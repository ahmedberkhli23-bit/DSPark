// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file AutoGain.h
 * @brief Automatic gain compensation for honest A/B comparison.
 *
 * Measures the input level before processing and adjusts the output level
 * after processing to match. This eliminates the loudness bias that makes
 * louder signals sound "better", enabling honest A/B testing.
 *
 * Usage pattern (sandwich):
 * ```
 *   autoGain.pushReference(buffer);   // measure input level
 *   myEffect.processBlock(buffer);    // apply your processing
 *   autoGain.compensate(buffer);      // adjust output to match input level
 * ```
 *
 * The compensation gain is smoothed to avoid pumping. A configurable maximum
 * compensation prevents runaway gain on silence.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::AutoGain<float> autoGain;
 *   autoGain.prepare(spec);
 *   autoGain.setMaxCompensation(12.0f);  // ±12 dB max
 *
 *   // In audio callback:
 *   autoGain.pushReference(buffer);
 *   compressor.processBlock(buffer);
 *   autoGain.compensate(buffer);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"

#include <algorithm>
#include <atomic>
#include <cmath>

namespace dspark {

/**
 * @class AutoGain
 * @brief Measures pre-processing level and compensates post-processing output.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class AutoGain
{
public:
    /**
     * @brief Prepares the auto-gain processor.
     * @param spec Audio environment specification.
     */
    void prepare(const AudioSpec& spec)
    {
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        // Default smoothing: ~100 ms
        smoothCoeff_ = static_cast<T>(
            1.0 - std::exp(-1.0 / (sampleRate_ * 0.100)));

        reset();
    }

    /**
     * @brief Snapshots the input level (call BEFORE processing).
     *
     * Computes the RMS level of the buffer and stores it as the reference.
     *
     * @param buffer Input audio (read-only measurement).
     */
    void pushReference(AudioBufferView<T> buffer) noexcept
    {
        refLevelDb_ = measureRmsDb(buffer);
    }

    /**
     * @brief Measures output level and applies gain compensation (call AFTER processing).
     *
     * Computes the difference between input and output levels and applies
     * a smoothed gain correction to the buffer.
     *
     * @param buffer Processed audio (modified in-place).
     */
    void compensate(AudioBufferView<T> buffer) noexcept
    {
        T outLevelDb = measureRmsDb(buffer);

        // Target compensation: make output match input level
        T targetDb = refLevelDb_ - outLevelDb;

        // Clamp to safety limits
        T maxComp = maxCompensation_.load(std::memory_order_relaxed);
        targetDb = std::clamp(targetDb, -maxComp, maxComp);

        // If both ref and output are silence, no compensation
        if (refLevelDb_ < T(-90) && outLevelDb < T(-90))
            targetDb = T(0);

        // Smooth the compensation to avoid pumping
        compensationDb_ += smoothCoeff_ * (targetDb - compensationDb_);

        // Apply
        T gain = decibelsToGain(compensationDb_);

        int numCh = std::min(buffer.getNumChannels(), numChannels_);
        int numSamples = buffer.getNumSamples();

        for (int ch = 0; ch < numCh; ++ch)
        {
            T* data = buffer.getChannel(ch);
            for (int i = 0; i < numSamples; ++i)
                data[i] *= gain;
        }
    }

    /** @brief Resets internal state. */
    void reset() noexcept
    {
        refLevelDb_ = T(-100);
        compensationDb_ = T(0);
    }

    /** @brief Returns the current compensation in dB (for metering). */
    [[nodiscard]] T getCompensationDb() const noexcept { return compensationDb_; }

    /**
     * @brief Sets the maximum allowed compensation in dB.
     * @param dB Max gain change (positive value, applies symmetrically). Default: 12.
     */
    void setMaxCompensation(T dB) noexcept { maxCompensation_.store(std::abs(dB), std::memory_order_relaxed); }

    /**
     * @brief Sets the smoothing time for gain changes.
     * @param ms Smoothing time in milliseconds (default: 100).
     */
    void setSmoothingTime(T ms) noexcept
    {
        double seconds = static_cast<double>(ms) * 0.001;
        smoothCoeff_ = static_cast<T>(
            1.0 - std::exp(-1.0 / (sampleRate_ * seconds)));
    }

private:
    /// Measures RMS level in dB across all channels.
    [[nodiscard]] T measureRmsDb(AudioBufferView<T> buffer) const noexcept
    {
        int numCh = std::min(buffer.getNumChannels(), numChannels_);
        int numSamples = buffer.getNumSamples();

        if (numSamples == 0) return T(-100);

        T sumSq = T(0);
        int totalSamples = 0;

        for (int ch = 0; ch < numCh; ++ch)
        {
            const T* data = buffer.getChannel(ch);
            for (int i = 0; i < numSamples; ++i)
                sumSq += data[i] * data[i];
            totalSamples += numSamples;
        }

        T rms = std::sqrt(sumSq / static_cast<T>(totalSamples));
        return gainToDecibels(rms);
    }

    double sampleRate_ = 44100.0;
    int numChannels_ = 2;

    T refLevelDb_ = T(-100);
    T compensationDb_ = T(0);
    std::atomic<T> maxCompensation_ { T(12) };
    T smoothCoeff_ = T(0.01);
};

} // namespace dspark
