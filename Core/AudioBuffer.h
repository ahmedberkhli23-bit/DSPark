// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file AudioBuffer.h
 * @brief Owning audio buffer and non-owning view for real-time DSP processing.
 *
 * Provides two complementary classes:
 *
 * - **AudioBufferView<T>**: A lightweight, non-owning view over channel pointers.
 *   Processors receive this type — it can reference any memory layout without copies.
 *   Supports zero-copy sub-views via getSubView().
 *
 * - **AudioBuffer<T, MaxChannels>**: An owning buffer with a single contiguous,
 *   32-byte aligned memory allocation. All memory is allocated in resize() (called
 *   during prepare()), never during process(). Move-only (copy is deleted).
 *
 * Dependencies: C++20 standard library only (<array>, <cstddef>, <cstring>, <new>).
 *
 * @note Memory layout:
 *       AudioBuffer stores all channel data in one contiguous block. Each channel
 *       is padded to a 32-byte boundary so SIMD operations can work on any channel
 *       without alignment faults.
 *
 *       ```
 *       [ch0_s0..ch0_sN | padding | ch1_s0..ch1_sN | padding | ...]
 *       ```
 *
 * @code
 *   // In prepare():
 *   dspark::AudioBuffer<float> buffer;
 *   buffer.resize(spec.numChannels, spec.maxBlockSize);
 *
 *   // Pass to a processor:
 *   myProcessor.process(buffer.toView());
 *
 *   // Create a sub-view for partial processing:
 *   auto firstHalf = buffer.toView().getSubView(0, 256);
 * @endcode
 */

#include "SimdOps.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <new>
#include <utility>

namespace dspark {

// ============================================================================
// AudioBufferView — Non-owning view
// ============================================================================

/**
 * @class AudioBufferView
 * @brief Non-owning view over audio channel data.
 *
 * Wraps a set of channel pointers and sample count without managing memory.
 * Lightweight and cheap to copy — designed to be passed by value into process()
 * functions. Supports const and non-const element types via the template parameter.
 *
 * @tparam T Sample type (typically `float`, `double`, `const float`, or `const double`).
 */
template <typename T, int MaxViewChannels = 16>
class AudioBufferView
{
public:
    /** @brief Default constructor. Creates an empty view (0 channels, 0 samples). */
    AudioBufferView() noexcept = default;

    /**
     * @brief Constructs a view from raw channel pointers.
     *
     * Channel pointers are copied into the view's internal storage, making
     * each view fully self-contained. Two consecutive getSubView() calls
     * will not invalidate each other.
     *
     * @param channelPtrs Array of pointers, one per channel.
     * @param numChannels Number of channels (must be <= MaxViewChannels).
     * @param numSamples  Number of samples per channel.
     */
    AudioBufferView(T** channelPtrs, int numChannels, int numSamples) noexcept
        : numChannels_(numChannels), numSamples_(numSamples)
    {
        assert(numChannels <= MaxViewChannels);
        for (int ch = 0; ch < numChannels; ++ch)
            channels_[ch] = channelPtrs[ch];
    }

    // -- Accessors -----------------------------------------------------------

    /**
     * @brief Returns a pointer to the sample data for the given channel.
     * @param ch Channel index (0-based).
     * @return Pointer to the first sample in the channel.
     */
    T* getChannel(int ch) noexcept
    {
        assert(ch >= 0 && ch < numChannels_);
        return channels_[ch];
    }

    /** @brief Const overload. */
    const T* getChannel(int ch) const noexcept
    {
        assert(ch >= 0 && ch < numChannels_);
        return channels_[ch];
    }

    /** @brief Returns the number of channels in this view. */
    [[nodiscard]] int getNumChannels() const noexcept { return numChannels_; }

    /** @brief Returns the number of samples per channel. */
    [[nodiscard]] int getNumSamples() const noexcept { return numSamples_; }

    // -- Sub-views -----------------------------------------------------------

    /**
     * @brief Returns a sub-view starting at a sample offset with a given length.
     *
     * This is a zero-copy operation — the returned view points into the same
     * underlying memory with adjusted pointers. Each sub-view is self-contained
     * (owns its own channel pointer array), so multiple consecutive calls are safe.
     *
     * @param startSample Offset from the beginning of each channel.
     * @param length       Number of samples in the sub-view.
     * @return A new AudioBufferView referencing the sub-range.
     */
    AudioBufferView getSubView(int startSample, int length) const noexcept
    {
        assert(startSample >= 0);
        assert(length >= 0);
        assert(startSample + length <= numSamples_);

        AudioBufferView sub;
        sub.numChannels_ = numChannels_;
        sub.numSamples_  = length;
        for (int ch = 0; ch < numChannels_; ++ch)
            sub.channels_[ch] = channels_[ch] + startSample;

        return sub;
    }

    // -- Operations ----------------------------------------------------------

    /** @brief Fills all channels with zero. */
    void clear() noexcept
    {
        for (int ch = 0; ch < numChannels_; ++ch)
            std::memset(channels_[ch], 0, static_cast<std::size_t>(numSamples_) * sizeof(T));
    }

    /**
     * @brief Copies samples from a source view into this view.
     *
     * The source may have a const-qualified sample type. Copies the minimum
     * of both views' channel and sample counts.
     *
     * @tparam U Source sample type (may be const-qualified).
     * @param src Source view to copy from.
     */
    template <typename U>
    void copyFrom(const AudioBufferView<U>& src) noexcept
    {
        const int chCount = std::min(numChannels_, src.getNumChannels());
        const int nSamples = std::min(numSamples_, src.getNumSamples());
        const auto bytes = static_cast<std::size_t>(nSamples) * sizeof(T);

        for (int ch = 0; ch < chCount; ++ch)
            std::memcpy(channels_[ch], src.getChannel(ch), bytes);
    }

    /**
     * @brief Adds samples from a source view into this view, optionally scaled.
     *
     * @tparam U Source sample type.
     * @param src  Source view.
     * @param gain Scaling factor applied to source samples before adding.
     */
    template <typename U>
    void addFrom(const AudioBufferView<U>& src, T gain = T(1)) noexcept
    {
        const int chCount = std::min(numChannels_, src.getNumChannels());
        const int nSamples = std::min(numSamples_, src.getNumSamples());

        for (int ch = 0; ch < chCount; ++ch)
        {
            T*       dst = channels_[ch];
            const auto* s = src.getChannel(ch);

            // SIMD fast path when source and destination share the same type
            if constexpr (std::is_same_v<std::remove_const_t<U>, T>)
            {
                simd::addWithGain(dst, s, gain, nSamples);
            }
            else
            {
                for (int i = 0; i < nSamples; ++i)
                    dst[i] += static_cast<T>(s[i]) * gain;
            }
        }
    }

    /**
     * @brief Multiplies all samples in all channels by a gain factor.
     * @param gain Scaling factor.
     */
    void applyGain(T gain) noexcept
    {
        for (int ch = 0; ch < numChannels_; ++ch)
            simd::applyGain(channels_[ch], gain, numSamples_);
    }

    /**
     * @brief Returns the peak absolute sample value across all channels.
     * @return Peak magnitude (>= 0).
     */
    [[nodiscard]] T getPeakLevel() const noexcept
    {
        T peak = T(0);
        for (int ch = 0; ch < numChannels_; ++ch)
        {
            T chPeak = simd::peakLevel(channels_[ch], numSamples_);
            if (chPeak > peak) peak = chPeak;
        }
        return peak;
    }

private:
    std::array<T*, MaxViewChannels> channels_ {};
    int numChannels_    = 0;
    int numSamples_     = 0;
};

// ============================================================================
// AudioBuffer — Owning, SIMD-aligned buffer
// ============================================================================

/**
 * @class AudioBuffer
 * @brief Owning audio buffer with contiguous, 32-byte aligned storage.
 *
 * Designed for real-time audio:
 * - **Single allocation**: all channel data lives in one contiguous block.
 * - **32-byte alignment**: each channel starts on a 32-byte boundary (AVX-ready).
 * - **Zero runtime allocation**: call resize() once in prepare(), then use
 *   toView() in process() with no allocations.
 * - **Move-only**: copying is deleted to prevent accidental allocations.
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum number of channels supported (compile-time bound).
 */
template <typename T, int MaxChannels = 16>
class AudioBuffer
{
    static constexpr std::size_t kAlignment = 32;

public:
    AudioBuffer() = default;

    ~AudioBuffer() { deallocate(); }

    // Move-only — copying would require allocation, violating RT safety.
    AudioBuffer(const AudioBuffer&)            = delete;
    AudioBuffer& operator=(const AudioBuffer&) = delete;

    AudioBuffer(AudioBuffer&& other) noexcept
        : rawData_(other.rawData_)
        , channelPtrs_(other.channelPtrs_)
        , numChannels_(other.numChannels_)
        , numSamples_(other.numSamples_)
        , allocatedBytes_(other.allocatedBytes_)
    {
        other.rawData_        = nullptr;
        other.channelPtrs_    = {};
        other.numChannels_    = 0;
        other.numSamples_     = 0;
        other.allocatedBytes_ = 0;
    }

    AudioBuffer& operator=(AudioBuffer&& other) noexcept
    {
        if (this != &other)
        {
            deallocate();

            rawData_        = other.rawData_;
            channelPtrs_    = other.channelPtrs_;
            numChannels_    = other.numChannels_;
            numSamples_     = other.numSamples_;
            allocatedBytes_ = other.allocatedBytes_;

            other.rawData_        = nullptr;
            other.channelPtrs_    = {};
            other.numChannels_    = 0;
            other.numSamples_     = 0;
            other.allocatedBytes_ = 0;
        }
        return *this;
    }

    // -- Allocation ----------------------------------------------------------

    /**
     * @brief Allocates (or re-allocates) the buffer for the given dimensions.
     *
     * This is the **only** function that allocates memory. Call it once during
     * prepare() — never during process(). If the current allocation already
     * satisfies the request, no reallocation occurs.
     *
     * Each channel is padded to a 32-byte boundary so SIMD intrinsics can
     * operate on any channel without alignment concerns.
     *
     * @param numChannels Number of channels (must be <= MaxChannels).
     * @param numSamples  Number of samples per channel.
     */
    void resize(int numChannels, int numSamples)
    {
        assert(numChannels >= 0 && numChannels <= MaxChannels);
        assert(numSamples >= 0);

        // Compute padded stride per channel (aligned to kAlignment)
        const auto samplesBytes = static_cast<std::size_t>(numSamples) * sizeof(T);
        const auto stride       = alignUp(samplesBytes, kAlignment);
        const auto totalBytes   = stride * static_cast<std::size_t>(numChannels);

        // Only reallocate if the existing block is too small
        if (totalBytes > allocatedBytes_)
        {
            deallocate();
            rawData_ = static_cast<T*>(::operator new(totalBytes, std::align_val_t(kAlignment)));
            allocatedBytes_ = totalBytes;
        }

        numChannels_ = numChannels;
        numSamples_  = numSamples;

        // Set up channel pointers into the contiguous block
        auto* base = reinterpret_cast<char*>(rawData_);
        for (int ch = 0; ch < numChannels; ++ch)
            channelPtrs_[ch] = reinterpret_cast<T*>(base + stride * static_cast<std::size_t>(ch));

        clear();
    }

    // -- View creation -------------------------------------------------------

    /**
     * @brief Returns a non-owning mutable view of this buffer.
     * @return AudioBufferView referencing this buffer's data.
     */
    AudioBufferView<T> toView() noexcept
    {
        return { channelPtrs_.data(), numChannels_, numSamples_ };
    }

    /**
     * @brief Returns a non-owning const view of this buffer.
     * @return Const AudioBufferView referencing this buffer's data.
     */
    AudioBufferView<const T> toView() const noexcept
    {
        // Safe cast: array of T* -> array of const T* (adding const).
        return { const_cast<const T**>(channelPtrs_.data()), numChannels_, numSamples_ };
    }

    // -- Accessors -----------------------------------------------------------

    /** @brief Returns a pointer to the sample data for the given channel. */
    T* getChannel(int ch) noexcept
    {
        assert(ch >= 0 && ch < numChannels_);
        return channelPtrs_[ch];
    }

    /** @brief Const overload. */
    const T* getChannel(int ch) const noexcept
    {
        assert(ch >= 0 && ch < numChannels_);
        return channelPtrs_[ch];
    }

    /** @brief Returns the number of active channels. */
    [[nodiscard]] int getNumChannels() const noexcept { return numChannels_; }

    /** @brief Returns the number of samples per channel. */
    [[nodiscard]] int getNumSamples() const noexcept { return numSamples_; }

    // -- Operations ----------------------------------------------------------

    /** @brief Fills all active channel data with zero. */
    void clear() noexcept
    {
        const auto bytes = static_cast<std::size_t>(numSamples_) * sizeof(T);
        for (int ch = 0; ch < numChannels_; ++ch)
            std::memset(channelPtrs_[ch], 0, bytes);
    }

private:
    /** @brief Rounds `value` up to the next multiple of `alignment`. */
    static constexpr std::size_t alignUp(std::size_t value, std::size_t alignment) noexcept
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    void deallocate() noexcept
    {
        if (rawData_)
        {
            ::operator delete(rawData_, std::align_val_t(kAlignment));
            rawData_ = nullptr;
        }
    }

    T*                          rawData_        = nullptr;
    std::array<T*, MaxChannels> channelPtrs_    {};
    int                         numChannels_    = 0;
    int                         numSamples_     = 0;
    std::size_t                 allocatedBytes_ = 0;
};

} // namespace dspark
