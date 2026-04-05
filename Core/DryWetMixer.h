// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DryWetMixer.h
 * @brief Real-time safe dry/wet mixer for effect processors.
 *
 * Captures a copy of the dry (unprocessed) signal before the effect runs,
 * then blends it with the wet (processed) signal using a mix proportion.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h.
 *
 * Usage pattern:
 * 1. In `prepare()`: call `prepare(spec)` to allocate the internal dry buffer.
 * 2. At the start of `process()`: call `pushDry(buffer)` to snapshot the input.
 * 3. Apply your effect to the buffer in-place.
 * 4. At the end of `process()`: call `mixWet(buffer, mix)` to blend dry + wet.
 *
 * @code
 *   dspark::DryWetMixer<float> mixer;
 *
 *   void prepare(const dspark::AudioSpec& spec) {
 *       mixer.prepare(spec);
 *   }
 *
 *   void process(dspark::AudioBufferView<float> buffer) {
 *       mixer.pushDry(buffer);
 *       applyEffect(buffer);       // Modifies buffer in-place (now wet)
 *       mixer.mixWet(buffer, 0.5f); // 50% dry, 50% wet
 *   }
 * @endcode
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"

#include <algorithm>
#include <cstring>

namespace dspark {

/**
 * @class DryWetMixer
 * @brief Pre-allocated dry/wet blender for real-time audio effects.
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum number of channels (compile-time bound).
 */
template <typename T, int MaxChannels = 16>
class DryWetMixer
{
public:
    /**
     * @brief Allocates the internal dry buffer for the given audio spec.
     *
     * Must be called before any processing. Only allocates if the current
     * buffer is too small.
     *
     * @param spec Audio environment specification.
     */
    void prepare(const AudioSpec& spec)
    {
        dryBuffer_.resize(spec.numChannels, spec.maxBlockSize);
    }

    /** @brief Resets the internal dry buffer to silence. */
    void reset() noexcept
    {
        dryBuffer_.clear();
    }

    /**
     * @brief Captures a snapshot of the dry (unprocessed) signal.
     *
     * Call this *before* applying the effect to the buffer. The view may
     * have fewer samples than the internal buffer's capacity — only the
     * actual sample count is copied.
     *
     * @param input The unprocessed audio buffer (read-only).
     */
    void pushDry(const AudioBufferView<const T>& input) noexcept
    {
        const int chCount  = std::min(input.getNumChannels(), dryBuffer_.getNumChannels());
        const int nSamples = std::min(input.getNumSamples(),  dryBuffer_.getNumSamples());
        const auto bytes   = static_cast<std::size_t>(nSamples) * sizeof(T);

        for (int ch = 0; ch < chCount; ++ch)
            std::memcpy(dryBuffer_.getChannel(ch), input.getChannel(ch), bytes);

        capturedSamples_ = nSamples;
    }

    /** @brief Overload accepting a mutable view (auto-converts to const). */
    void pushDry(const AudioBufferView<T>& input) noexcept
    {
        const int chCount  = std::min(input.getNumChannels(), dryBuffer_.getNumChannels());
        const int nSamples = std::min(input.getNumSamples(),  dryBuffer_.getNumSamples());
        const auto bytes   = static_cast<std::size_t>(nSamples) * sizeof(T);

        for (int ch = 0; ch < chCount; ++ch)
            std::memcpy(dryBuffer_.getChannel(ch), input.getChannel(ch), bytes);

        capturedSamples_ = nSamples;
    }

    /**
     * @brief Blends the stored dry signal with the current (wet) buffer.
     *
     * Applies a linear crossfade:
     *   output = dry * (1 - mix) + wet * mix
     *
     * @param wetBuffer     The processed buffer (modified in-place to the blended result).
     * @param mixProportion Mix amount: 0.0 = fully dry, 1.0 = fully wet.
     */
    void mixWet(AudioBufferView<T> wetBuffer, T mixProportion) noexcept
    {
        const T wet = std::clamp(mixProportion, T(0), T(1));
        const T dry = T(1) - wet;

        const int chCount  = std::min(wetBuffer.getNumChannels(), dryBuffer_.getNumChannels());
        const int nSamples = std::min(wetBuffer.getNumSamples(), capturedSamples_);

        for (int ch = 0; ch < chCount; ++ch)
        {
            T*       wetData = wetBuffer.getChannel(ch);
            const T* dryData = dryBuffer_.getChannel(ch);

            for (int i = 0; i < nSamples; ++i)
                wetData[i] = dryData[i] * dry + wetData[i] * wet;
        }
    }

    /**
     * @brief Returns a pointer to the dry channel data for the given channel.
     *
     * Valid only after pushDry() has been called.
     *
     * @param ch Channel index (0-based).
     * @return Pointer to the dry samples for this channel (read-only).
     */
    [[nodiscard]] const T* getDryChannel(int ch) const noexcept
    {
        return dryBuffer_.getChannel(ch);
    }

    /** @brief Returns the number of channels in the dry buffer. */
    [[nodiscard]] int getDryNumChannels() const noexcept { return dryBuffer_.getNumChannels(); }

    /** @brief Returns the number of samples captured in the last pushDry() call. */
    [[nodiscard]] int getDryCapturedSamples() const noexcept { return capturedSamples_; }

private:
    AudioBuffer<T, MaxChannels> dryBuffer_;
    int capturedSamples_ = 0;
};

} // namespace dspark
