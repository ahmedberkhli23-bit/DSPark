// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Reverb.h
 * @brief Convolution reverb with one-line IR loading and progressive API.
 *
 * Wraps the Convolver engine into a complete reverb effect with dry/wet mix,
 * pre-delay, and automatic IR management. Supports loading impulse responses
 * from WAV files or from raw sample data.
 *
 * Three levels of API complexity:
 *
 * - **Level 1 (simple):** `reverb.loadIR("hall.wav"); reverb.setMix(0.3f);`
 * - **Level 2 (intermediate):** Add pre-delay, different IR sample rate.
 * - **Level 3 (expert):** Direct access to Convolver and DryWetMixer internals.
 *
 * Dependencies: Convolver.h, DryWetMixer.h, RingBuffer.h, AudioSpec.h,
 *               AudioBuffer.h, WavFile.h.
 *
 * @code
 *   dspark::Reverb<float> reverb;
 *   reverb.prepare(spec);
 *   reverb.loadIR("hall.wav");   // One line — done
 *   reverb.setMix(0.3f);         // 30% wet
 *   reverb.processBlock(buffer);
 *
 *   // With pre-delay:
 *   reverb.setPreDelay(20.0f);   // 20 ms
 * @endcode
 */

#include "../Core/Convolver.h"
#include "../Core/DryWetMixer.h"
#include "../Core/RingBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/Resampler.h"
#include "../IO/WavFile.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>

namespace dspark {

/**
 * @class Reverb
 * @brief Convolution reverb with IR loading, dry/wet, and pre-delay.
 *
 * Internally manages one Convolver per channel. The IR is automatically
 * resampled if its sample rate differs from the processing sample rate.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Reverb
{
public:
    virtual ~Reverb() = default;

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the reverb for processing.
     *
     * Allocates internal buffers and sets up dry/wet mixer and pre-delay.
     * If an IR was loaded before prepare(), it will be re-applied.
     *
     * @param spec Audio environment (sample rate, block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        mixer_.prepare(spec);

        // Pre-delay ring buffers (one per channel, max 500ms)
        int maxDelaySamples = static_cast<int>(spec.sampleRate * 0.5) + 1;
        preDelayBuffers_.resize(spec.numChannels);
        for (auto& rb : preDelayBuffers_)
            rb.prepare(maxDelaySamples);

        updatePreDelay();

        // Re-apply IR if one was already loaded
        if (!irStorage_.empty())
            applyIR();
    }

    /**
     * @brief Processes audio through the reverb.
     *
     * Flow: pushDry -> pre-delay -> convolve -> mixWet.
     *
     * @param buffer Audio data to process in-place.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!loaded_.load(std::memory_order_acquire)) return;

        const int nCh = std::min(buffer.getNumChannels(), static_cast<int>(convolvers_.size()));
        const int nS  = buffer.getNumSamples();

        mixer_.pushDry(buffer);

        // Apply pre-delay + convolution per channel
        int preDelSamp = preDelaySamples_.load(std::memory_order_relaxed);
        T mixVal = mix_.load(std::memory_order_relaxed);

        for (int ch = 0; ch < nCh; ++ch)
        {
            T* data = buffer.getChannel(ch);

            if (preDelSamp > 0)
            {
                auto& ring = preDelayBuffers_[ch];
                for (int i = 0; i < nS; ++i)
                {
                    ring.push(data[i]);
                    data[i] = ring.read(preDelSamp);
                }
            }

            convolvers_[ch].processInPlace(data, nS);
        }

        mixer_.mixWet(buffer, mixVal);
    }

    /** @brief Resets all internal state (convolvers, pre-delay, mixer). */
    void reset() noexcept
    {
        loaded_.store(false, std::memory_order_relaxed);
        for (auto& conv : convolvers_)
            conv.reset();
        for (auto& rb : preDelayBuffers_)
            rb.reset();
        mixer_.reset();
    }

    // -- Level 1: Simple API ----------------------------------------------------

    /**
     * @brief Loads an impulse response from a WAV file.
     *
     * The IR is automatically resampled if the WAV sample rate differs from
     * the processing sample rate. Multi-channel IRs are supported (one
     * convolver per channel); mono IRs are duplicated across all channels.
     *
     * @param wavFilePath Path to the WAV file.
     * @return True if the IR was loaded successfully.
     */
    bool loadIR(const char* wavFilePath)
    {
        WavFile wav;
        if (!wav.openRead(wavFilePath))
            return false;

        auto info = wav.getInfo();

        AudioBuffer<T> irBuf;
        irBuf.resize(info.numChannels, static_cast<int>(info.numSamples));
        wav.readSamples(irBuf.toView());
        wav.close();

        // Store channel 0 (or all channels)
        irChannels_ = info.numChannels;
        int irLen = static_cast<int>(info.numSamples);

        irStorage_.resize(static_cast<size_t>(irChannels_) * static_cast<size_t>(irLen));
        for (int ch = 0; ch < irChannels_; ++ch)
        {
            const T* src = irBuf.getChannel(ch);
            T* dst = irStorage_.data() + static_cast<size_t>(ch) * static_cast<size_t>(irLen);
            std::copy_n(src, irLen, dst);
        }
        irLength_ = irLen;
        irSampleRate_ = info.sampleRate;

        if (spec_.sampleRate > 0)
            applyIR();

        return true;
    }

    /**
     * @brief Sets the dry/wet mix.
     * @param dryWet 0.0 = fully dry (no reverb), 1.0 = fully wet.
     */
    void setMix(T dryWet) noexcept { mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed); }

    // -- Level 2: Intermediate API ----------------------------------------------

    /**
     * @brief Loads an IR from raw sample data.
     *
     * @param data         Pointer to mono IR samples.
     * @param length       Number of samples.
     * @param irSampleRate Sample rate of the IR data.
     */
    void loadIR(const T* data, int length, double irSampleRate)
    {
        irChannels_ = 1;
        irLength_ = length;
        irSampleRate_ = irSampleRate;

        irStorage_.assign(data, data + length);

        if (spec_.sampleRate > 0)
            applyIR();
    }

    /**
     * @brief Sets the pre-delay time in milliseconds.
     *
     * Pre-delay adds a gap before the reverb tail starts, creating a sense
     * of room size. Typical values: 0-50 ms.
     *
     * @param ms Pre-delay in milliseconds (0 = off).
     */
    void setPreDelay(T ms) noexcept
    {
        preDelayMs_.store(std::max(ms, T(0)), std::memory_order_relaxed);
        updatePreDelay();
    }

    // -- Level 3: Expert API ----------------------------------------------------

    /**
     * @brief Direct access to a channel's Convolver.
     * @param channel Channel index.
     * @return Reference to the Convolver for this channel.
     */
    Convolver<T>& getConvolver(int channel = 0) { return convolvers_[channel]; }

    /** @brief Direct access to the DryWetMixer. */
    DryWetMixer<T>& getMixer() { return mixer_; }

    /** @brief Returns true if an IR has been loaded and applied. */
    [[nodiscard]] bool isLoaded() const noexcept { return loaded_.load(std::memory_order_acquire); }

    /** @brief Returns the current mix value. */
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }

    /** @brief Returns the current pre-delay in ms. */
    [[nodiscard]] T getPreDelay() const noexcept { return preDelayMs_.load(std::memory_order_relaxed); }

    /** @brief Returns the convolution latency in samples. */
    [[nodiscard]] int getLatency() const noexcept
    {
        return loaded_.load(std::memory_order_acquire) && !convolvers_.empty() ? convolvers_[0].getLatency() : 0;
    }

protected:
    void updatePreDelay() noexcept
    {
        if (spec_.sampleRate > 0)
            preDelaySamples_.store(static_cast<int>(static_cast<T>(spec_.sampleRate)
                                                * preDelayMs_.load(std::memory_order_relaxed) / T(1000)),
                                  std::memory_order_relaxed);
    }

    void applyIR()
    {
        if (irStorage_.empty() || irLength_ <= 0) return;

        int blockSize = spec_.maxBlockSize;
        // Round up to next power of two for FFT
        int fftBlock = 1;
        while (fftBlock < blockSize) fftBlock <<= 1;

        int numCh = spec_.numChannels;
        convolvers_.resize(numCh);

        for (int ch = 0; ch < numCh; ++ch)
        {
            // Pick IR channel: use corresponding channel if available, else mono (ch 0)
            int irCh = (ch < irChannels_) ? ch : 0;
            const T* irData = irStorage_.data()
                            + static_cast<size_t>(irCh) * static_cast<size_t>(irLength_);

            // Resample IR if sample rates differ
            if (std::abs(irSampleRate_ - spec_.sampleRate) > 1.0)
            {
                double ratio = spec_.sampleRate / irSampleRate_;
                int newLen = static_cast<int>(static_cast<double>(irLength_) * ratio) + 1;
                std::vector<T> resampled(newLen);

                Resampler<T> resampler;
                resampler.prepare(irSampleRate_, spec_.sampleRate);
                int produced = resampler.processBlock(irData, irLength_,
                                                      resampled.data());

                convolvers_[ch].prepare(fftBlock, resampled.data(), produced);
            }
            else
            {
                convolvers_[ch].prepare(fftBlock, irData, irLength_);
            }
        }

        loaded_.store(true, std::memory_order_release);
    }

    AudioSpec spec_ {};
    std::atomic<T> mix_ { T(0.3) };
    std::atomic<T> preDelayMs_ { T(0) };
    std::atomic<int> preDelaySamples_ { 0 };
    std::atomic<bool> loaded_ { false };

    // IR storage
    std::vector<T> irStorage_;
    int irLength_ = 0;
    int irChannels_ = 0;
    double irSampleRate_ = 0;

    // Processing
    std::vector<Convolver<T>> convolvers_;
    std::vector<RingBuffer<T>> preDelayBuffers_;
    DryWetMixer<T> mixer_;
};

} // namespace dspark
