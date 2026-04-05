// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Gain.h
 * @brief Smoothed gain processor with fade, mute, and channel linking.
 *
 * Applies gain to audio with click-free ramping. This is the most fundamental
 * audio processor — every signal chain needs gain control. Supports:
 * - Set gain in dB or linear
 * - Automatic smoothing (no zipper noise)
 * - Fade-in / fade-out with configurable time
 * - Mute with smooth transition
 * - Channel linking (same gain to all channels, or per-channel)
 * - Polarity inversion
 *
 * Dependencies: DspMath.h, SmoothedValue.h.
 *
 * @code
 *   dspark::Gain<float> gain;
 *   gain.prepare(48000.0, 10.0);  // 48 kHz, 10 ms ramp time
 *   gain.setGainDb(-6.0f);        // -6 dB
 *
 *   // In audio callback:
 *   gain.process(buffer, numSamples);  // in-place
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/SmoothedValue.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/SimdOps.h"

#include <algorithm>
#include <atomic>
#include <cmath>

namespace dspark {

/**
 * @class Gain
 * @brief Click-free gain processor with exponential smoothing and mute.
 *
 * Uses one-pole exponential smoothing (SmoothedValue) for perceptually
 * uniform gain transitions. This produces natural-sounding fades where
 * the perceived loudness change is constant over time.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Gain
{
public:
    ~Gain() = default;
    Gain()
    {
        gainSmooth_.reset(T(1));
    }

    /**
     * @brief Prepares the gain processor.
     *
     * @param sampleRate Sample rate in Hz.
     * @param rampTimeMs Smoothing time in milliseconds (default: 10 ms).
     * @param numChannels Number of channels (default: 2).
     */
    void prepare(double sampleRate, double rampTimeMs = 10.0,
                 int numChannels = 2)
    {
        sampleRate_ = sampleRate;
        numChannels_ = numChannels;

        gainSmooth_.prepare(sampleRate, rampTimeMs);
        gainSmooth_.setCurrentAndTarget(effectiveTarget());
    }

    /** @brief Prepares from AudioSpec (unified API). */
    void prepare(const AudioSpec& spec)
    {
        prepare(spec.sampleRate, 10.0, spec.numChannels);
    }

    /**
     * @brief Processes an AudioBufferView in-place (unified API).
     * @param buffer Audio buffer.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        const int nCh = buffer.getNumChannels();
        const int nS = buffer.getNumSamples();
        if (nCh == 1)
        {
            process(buffer.getChannel(0), nS);
        }
        else
        {
            T* channels[16];
            int useCh = nCh < 16 ? nCh : 16;
            for (int c = 0; c < useCh; ++c)
                channels[c] = buffer.getChannel(c);
            process(channels, useCh, nS);
        }
    }

    /**
     * @brief Sets the target gain in decibels.
     * @param dB Gain in dB (0 = unity, -6 = half, +6 = double).
     */
    void setGainDb(T dB) noexcept
    {
        targetGain_.store(decibelsToGain(dB), std::memory_order_relaxed);
        gainSmooth_.setTargetValue(effectiveTarget());
    }

    /**
     * @brief Sets the target gain as a linear multiplier.
     * @param linear Gain multiplier (0 = silence, 1 = unity, 2 = +6 dB).
     */
    void setGainLinear(T linear) noexcept
    {
        targetGain_.store(std::max(T(0), linear), std::memory_order_relaxed);
        gainSmooth_.setTargetValue(effectiveTarget());
    }

    /**
     * @brief Returns the current target gain in dB.
     */
    [[nodiscard]] T getGainDb() const noexcept
    {
        return gainToDecibels(targetGain_.load(std::memory_order_relaxed));
    }

    /**
     * @brief Returns the current target gain as linear multiplier.
     */
    [[nodiscard]] T getGainLinear() const noexcept
    {
        return targetGain_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Returns the current (smoothed) gain value.
     */
    [[nodiscard]] T getCurrentGain() const noexcept
    {
        return gainSmooth_.getCurrentValue();
    }

    /**
     * @brief Enables or disables mute (smooth transition to/from silence).
     * @param muted True to mute, false to unmute.
     */
    void setMuted(bool muted) noexcept
    {
        muted_.store(muted, std::memory_order_relaxed);
        gainSmooth_.setTargetValue(effectiveTarget());
    }

    /**
     * @brief Returns whether the processor is muted.
     */
    [[nodiscard]] bool isMuted() const noexcept { return muted_.load(std::memory_order_relaxed); }

    /**
     * @brief Enables or disables polarity inversion.
     * @param inverted True to invert polarity (multiply by -1).
     */
    void setInverted(bool inverted) noexcept { inverted_.store(inverted, std::memory_order_relaxed); }

    /**
     * @brief Returns whether polarity is inverted.
     */
    [[nodiscard]] bool isInverted() const noexcept { return inverted_.load(std::memory_order_relaxed); }

    /**
     * @brief Sets the smoothing ramp time.
     * @param rampTimeMs Ramp time in milliseconds.
     */
    void setRampTime(double rampTimeMs) noexcept
    {
        gainSmooth_.setRampTime(sampleRate_, rampTimeMs);
    }

    /**
     * @brief Processes a single sample (mono).
     * @param sample Input sample.
     * @return Gained sample.
     */
    [[nodiscard]] T processSample(T sample) noexcept
    {
        T g = gainSmooth_.getNextValue();
        if (inverted_.load(std::memory_order_relaxed)) g = -g;
        return sample * g;
    }

    /**
     * @brief Processes an interleaved or single-channel buffer in-place.
     *
     * @param data Audio samples (modified in-place).
     * @param numSamples Number of samples per channel.
     */
    void process(T* data, int numSamples) noexcept
    {
        const bool inv = inverted_.load(std::memory_order_relaxed);
        int i = 0;

        // Per-sample ramp while the smoother is still moving
        for (; i < numSamples && gainSmooth_.isSmoothing(); ++i)
        {
            T g = gainSmooth_.getNextValue();
            if (inv) g = -g;
            data[i] *= g;
        }

        // SIMD bulk path once gain has converged
        if (i < numSamples)
        {
            T g = gainSmooth_.getCurrentValue();
            if (inv) g = -g;
            simd::applyGain(data + i, g, numSamples - i);
        }
    }

    /**
     * @brief Processes separate channel buffers in-place.
     *
     * @param channelData Array of pointers to each channel's data.
     * @param numChannels Number of channels.
     * @param numSamples Number of samples per channel.
     */
    void process(T** channelData, int numChannels, int numSamples) noexcept
    {
        const bool inv = inverted_.load(std::memory_order_relaxed);
        int i = 0;

        // Per-sample ramp while the smoother is still moving
        for (; i < numSamples && gainSmooth_.isSmoothing(); ++i)
        {
            T g = gainSmooth_.getNextValue();
            if (inv) g = -g;

            for (int ch = 0; ch < numChannels; ++ch)
                channelData[ch][i] *= g;
        }

        // SIMD bulk path once gain has converged
        if (i < numSamples)
        {
            T g = gainSmooth_.getCurrentValue();
            if (inv) g = -g;
            const int remaining = numSamples - i;
            for (int ch = 0; ch < numChannels; ++ch)
                simd::applyGain(channelData[ch] + i, g, remaining);
        }
    }

    /**
     * @brief Processes input to output (not in-place).
     *
     * @param input Input buffer.
     * @param output Output buffer.
     * @param numSamples Number of samples.
     */
    void process(const T* input, T* output, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            T g = gainSmooth_.getNextValue();
            if (inverted_.load(std::memory_order_relaxed)) g = -g;
            output[i] = input[i] * g;
        }
    }

    /**
     * @brief Returns true if the gain is still ramping toward the target.
     */
    [[nodiscard]] bool isRamping() const noexcept
    {
        return gainSmooth_.isSmoothing();
    }

    /**
     * @brief Skips smoothing — immediately sets current gain to target.
     */
    void skipRamp() noexcept
    {
        gainSmooth_.skip();
    }

    /**
     * @brief Resets gain to the target value (no ramping on next process).
     */
    void reset() noexcept
    {
        gainSmooth_.setCurrentAndTarget(effectiveTarget());
    }

protected:
    T effectiveTarget() const noexcept
    {
        return muted_.load(std::memory_order_relaxed) ? T(0)
               : targetGain_.load(std::memory_order_relaxed);
    }

    double sampleRate_ = 48000.0;
    int numChannels_ = 2;

    std::atomic<T> targetGain_ { T(1) };
    SmoothedValue<T> gainSmooth_;

    std::atomic<bool> muted_ { false };
    std::atomic<bool> inverted_ { false };
};

} // namespace dspark
