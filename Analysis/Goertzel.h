// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Goertzel.h
 * @brief Goertzel algorithm for efficient single-frequency detection.
 *
 * The Goertzel algorithm computes the energy at a specific frequency in O(N)
 * time — much more efficient than a full FFT when you only need one or a few
 * frequency bins. Uses only 2 multiplies and 4 adds per sample (vs FFT's
 * O(N log N) for all bins).
 *
 * Use cases:
 * - **Guitar tuner:** Detect pitch of single notes
 * - **DTMF detection:** Decode phone tones (only 8 frequencies needed)
 * - **Harmonic analysis:** Measure specific harmonics (e.g., THD at 1 kHz)
 * - **Frequency presence:** Check if a tone is present in a signal
 *
 * Dependencies: DspMath.h.
 *
 * @code
 *   // Detect 440 Hz (A4) in a buffer
 *   dspark::Goertzel<float> detector;
 *   detector.prepare(48000.0, 440.0, 2048);  // 2048-sample window
 *
 *   detector.processBlock(audioData, 2048);
 *   float magnitude = detector.getMagnitude();
 *   float power = detector.getPower();
 * @endcode
 */

#include "../Core/DspMath.h"

#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class Goertzel
 * @brief Single-frequency magnitude detector using the Goertzel algorithm.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Goertzel
{
public:
    /**
     * @brief Prepares the detector for a specific frequency.
     *
     * @param sampleRate Sample rate in Hz.
     * @param targetFreqHz The frequency to detect.
     * @param blockSize Number of samples per analysis block (higher = more resolution).
     */
    void prepare(double sampleRate, double targetFreqHz, int blockSize) noexcept
    {
        sampleRate_ = sampleRate;
        targetFreq_ = targetFreqHz;
        blockSize_ = blockSize;

        // Normalised frequency: k = round(N * f / fs)
        double k = static_cast<double>(blockSize) * targetFreqHz / sampleRate;

        // Goertzel coefficient: 2 * cos(2 * pi * k / N)
        double omega = 2.0 * std::numbers::pi * k / static_cast<double>(blockSize);
        coeff_ = static_cast<T>(2.0 * std::cos(omega));
        cosOmega_ = static_cast<T>(std::cos(omega));
        sinOmega_ = static_cast<T>(std::sin(omega));

        reset();
    }

    /**
     * @brief Processes a complete block and computes the magnitude.
     *
     * After calling this, use getMagnitude() or getPower() to read the result.
     *
     * @param data Audio samples.
     * @param numSamples Number of samples (should match blockSize from prepare).
     */
    void processBlock(const T* data, int numSamples) noexcept
    {
        if (numSamples <= 0) return;

        T s0 = T(0), s1 = T(0), s2 = T(0);

        for (int i = 0; i < numSamples; ++i)
        {
            s0 = data[i] + coeff_ * s1 - s2;
            s2 = s1;
            s1 = s0;
        }

        // Compute real and imaginary parts
        real_ = s1 - s2 * cosOmega_;
        imag_ = s2 * sinOmega_;

        // Normalise by block size
        T invN = T(1) / static_cast<T>(numSamples);
        real_ *= invN;
        imag_ *= invN;
    }

    /**
     * @brief Feeds a single sample into the running Goertzel computation.
     *
     * Use this for streaming mode. Call `compute()` after pushing `blockSize`
     * samples to get the result.
     *
     * @param sample Input sample.
     */
    void pushSample(T sample) noexcept
    {
        T s0 = sample + coeff_ * s1_ - s2_;
        s2_ = s1_;
        s1_ = s0;

        ++sampleCount_;
        if (sampleCount_ >= blockSize_)
            compute();
    }

    /**
     * @brief Completes the streaming computation and stores the result.
     *
     * Called automatically by pushSample when blockSize is reached.
     * Can also be called manually.
     */
    void compute() noexcept
    {
        real_ = s1_ - s2_ * cosOmega_;
        imag_ = s2_ * sinOmega_;

        T invN = T(1) / static_cast<T>(std::max(sampleCount_, 1));
        real_ *= invN;
        imag_ *= invN;

        // Reset for next block
        s1_ = T(0);
        s2_ = T(0);
        sampleCount_ = 0;
    }

    // -- Results ------------------------------------------------------------------

    /**
     * @brief Returns the magnitude at the target frequency.
     * @return Magnitude (linear scale).
     */
    [[nodiscard]] T getMagnitude() const noexcept
    {
        return std::sqrt(real_ * real_ + imag_ * imag_);
    }

    /**
     * @brief Returns the power at the target frequency.
     * @return Power (magnitude squared).
     */
    [[nodiscard]] T getPower() const noexcept
    {
        return real_ * real_ + imag_ * imag_;
    }

    /**
     * @brief Returns the magnitude in decibels.
     * @return Magnitude in dB.
     */
    [[nodiscard]] T getMagnitudeDb() const noexcept
    {
        return gainToDecibels(getMagnitude());
    }

    /**
     * @brief Returns the phase angle at the target frequency.
     * @return Phase in radians.
     */
    [[nodiscard]] T getPhase() const noexcept
    {
        return std::atan2(imag_, real_);
    }

    /**
     * @brief Returns the target frequency.
     */
    [[nodiscard]] double getTargetFrequency() const noexcept { return targetFreq_; }

    /**
     * @brief Resets the internal state.
     */
    void reset() noexcept
    {
        s1_ = T(0);
        s2_ = T(0);
        sampleCount_ = 0;
        real_ = T(0);
        imag_ = T(0);
    }

private:
    double sampleRate_ = 48000.0;
    double targetFreq_ = 440.0;
    int blockSize_ = 2048;

    T coeff_ = T(0);
    T cosOmega_ = T(0);
    T sinOmega_ = T(0);

    // Streaming state
    T s1_ = T(0);
    T s2_ = T(0);
    int sampleCount_ = 0;

    // Results
    T real_ = T(0);
    T imag_ = T(0);
};

} // namespace dspark
