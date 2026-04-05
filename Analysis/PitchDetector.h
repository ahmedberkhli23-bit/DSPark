// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file PitchDetector.h
 * @brief Real-time monophonic pitch detection using the YIN algorithm.
 *
 * Implements the YIN autocorrelation method (de Cheveigné & Kawahara, 2002)
 * with cumulative mean normalized difference and parabolic interpolation.
 * Suitable for tuners, pitch correction, and musical analysis.
 *
 * The detector operates on a ring buffer — push samples in with pushSamples()
 * and query the current detected pitch at any time. Detection runs
 * automatically when enough samples have been accumulated.
 *
 * Dependencies: C++20 standard library only.
 *
 * @code
 *   dspark::PitchDetector<float> detector;
 *   detector.prepare(48000.0);
 *
 *   // In audio callback:
 *   detector.pushSamples(input, blockSize);
 *
 *   // Query results (can be called from any thread):
 *   float freq = detector.getFrequencyHz();
 *   float conf = detector.getConfidence();
 *   int note   = detector.getMidiNote();       // 69 = A4
 *   float cent = detector.getCentsOffset();    // -50..+50
 * @endcode
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace dspark {

/**
 * @class PitchDetector
 * @brief YIN-based monophonic pitch detector with parabolic interpolation.
 *
 * YIN steps:
 * 1. Compute the difference function d(τ).
 * 2. Compute the cumulative mean normalized difference d'(τ).
 * 3. Find the first dip below a threshold in d'(τ).
 * 4. Refine with parabolic interpolation for sub-sample accuracy.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class PitchDetector
{
public:
    /**
     * @brief Prepares the detector for a given sample rate.
     *
     * @param sampleRate  Sample rate in Hz.
     * @param windowSize  Analysis window size (default: 2048 — covers ~24 Hz min).
     */
    void prepare(double sampleRate, int windowSize = 2048)
    {
        sampleRate_ = sampleRate;
        windowSize_ = windowSize;
        halfWindow_ = windowSize / 2;

        buffer_.assign(static_cast<size_t>(windowSize), T(0));
        yinBuffer_.resize(static_cast<size_t>(halfWindow_));

        writePos_ = 0;
        samplesAccumulated_ = 0;

        frequency_ = T(0);
        confidence_ = T(0);
    }

    /**
     * @brief Pushes audio samples into the detector.
     *
     * When enough samples have been accumulated (windowSize), a full
     * YIN analysis is performed automatically.
     *
     * @param samples  Input audio data (mono).
     * @param numSamples Number of samples.
     */
    void pushSamples(const T* samples, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            buffer_[static_cast<size_t>(writePos_)] = samples[i];
            writePos_ = (writePos_ + 1) % windowSize_;
            ++samplesAccumulated_;

            if (samplesAccumulated_ >= windowSize_)
            {
                detect();
                samplesAccumulated_ = 0;
            }
        }
    }

    /** @brief Returns the detected fundamental frequency in Hz (0 if unvoiced). */
    [[nodiscard]] T getFrequencyHz() const noexcept { return frequency_; }

    /** @brief Returns detection confidence (0 = no pitch, 1 = very confident). */
    [[nodiscard]] T getConfidence() const noexcept { return confidence_; }

    /**
     * @brief Returns the closest MIDI note number (69 = A4 = 440 Hz).
     * @return MIDI note, or -1 if no pitch detected.
     */
    [[nodiscard]] int getMidiNote() const noexcept
    {
        if (frequency_ <= T(0)) return -1;
        return static_cast<int>(std::round(
            T(69) + T(12) * std::log2(frequency_ / T(440))));
    }

    /**
     * @brief Returns the offset in cents from the nearest MIDI note.
     * @return Cents offset in [-50, +50], or 0 if no pitch.
     */
    [[nodiscard]] T getCentsOffset() const noexcept
    {
        if (frequency_ <= T(0)) return T(0);
        T midiExact = T(69) + T(12) * std::log2(frequency_ / T(440));
        T nearest = std::round(midiExact);
        return (midiExact - nearest) * T(100);
    }

    /**
     * @brief Sets the YIN threshold.
     *
     * Lower values = more selective (fewer false positives, may miss quiet notes).
     * Higher values = more sensitive (catches quieter notes, more false positives).
     *
     * @param threshold Typical range: 0.05 – 0.20 (default: 0.10).
     */
    void setThreshold(T threshold) noexcept
    {
        threshold_ = std::clamp(threshold, T(0.01), T(0.5));
    }

    /** @brief Resets the detector state. */
    void reset() noexcept
    {
        std::fill(buffer_.begin(), buffer_.end(), T(0));
        writePos_ = 0;
        samplesAccumulated_ = 0;
        frequency_ = T(0);
        confidence_ = T(0);
    }

private:
    /// Runs the full YIN pitch detection on the current buffer.
    void detect() noexcept
    {
        // Check for silence — avoid false detections on zero-energy input
        T energy = T(0);
        for (int i = 0; i < windowSize_; ++i)
            energy += readBuffer(i) * readBuffer(i);
        if (energy < T(1e-10))
        {
            frequency_ = T(0);
            confidence_ = T(0);
            return;
        }

        // Step 1 + 2: Difference function + CMND
        computeCMND();

        // Step 3: Absolute threshold — find the first dip below threshold
        int tauEstimate = -1;

        for (int tau = 2; tau < halfWindow_; ++tau)
        {
            if (yinBuffer_[static_cast<size_t>(tau)] < threshold_)
            {
                // Find the local minimum in this valley
                while (tau + 1 < halfWindow_ &&
                       yinBuffer_[static_cast<size_t>(tau + 1)] <
                           yinBuffer_[static_cast<size_t>(tau)])
                {
                    ++tau;
                }
                tauEstimate = tau;
                break;
            }
        }

        if (tauEstimate < 0)
        {
            // No pitch found
            frequency_ = T(0);
            confidence_ = T(0);
            return;
        }

        // Step 4: Parabolic interpolation for sub-sample accuracy
        T betterTau = parabolicInterp(tauEstimate);

        frequency_ = static_cast<T>(sampleRate_) / betterTau;
        confidence_ = T(1) - yinBuffer_[static_cast<size_t>(tauEstimate)];
        confidence_ = std::clamp(confidence_, T(0), T(1));
    }

    /**
     * @brief Computes the cumulative mean normalized difference function.
     *
     * Combines steps 1 and 2 of YIN for efficiency:
     * d(τ) = Σ (x[n] - x[n+τ])²
     * d'(τ) = d(τ) / ((1/τ) Σ d(j) for j=1..τ)
     */
    void computeCMND() noexcept
    {
        yinBuffer_[0] = T(1);

        T runningSum = T(0);

        for (int tau = 1; tau < halfWindow_; ++tau)
        {
            T sum = T(0);
            for (int j = 0; j < halfWindow_; ++j)
            {
                T diff = readBuffer(j) - readBuffer(j + tau);
                sum += diff * diff;
            }

            runningSum += sum;
            yinBuffer_[static_cast<size_t>(tau)] =
                (runningSum > T(0)) ? sum * static_cast<T>(tau) / runningSum : T(0);
        }
    }

    /// Reads from the circular buffer (linearized from writePos).
    [[nodiscard]] T readBuffer(int index) const noexcept
    {
        int pos = (writePos_ - windowSize_ + index + windowSize_) % windowSize_;
        return buffer_[static_cast<size_t>(pos)];
    }

    /// Parabolic interpolation around a YIN minimum for sub-sample tau.
    [[nodiscard]] T parabolicInterp(int tau) const noexcept
    {
        if (tau < 1 || tau >= halfWindow_ - 1)
            return static_cast<T>(tau);

        T s0 = yinBuffer_[static_cast<size_t>(tau - 1)];
        T s1 = yinBuffer_[static_cast<size_t>(tau)];
        T s2 = yinBuffer_[static_cast<size_t>(tau + 1)];

        T adjustment = (s0 - s2) / (T(2) * (s0 - T(2) * s1 + s2));
        return static_cast<T>(tau) + adjustment;
    }

    double sampleRate_ = 44100.0;
    int windowSize_ = 2048;
    int halfWindow_ = 1024;
    int writePos_ = 0;
    int samplesAccumulated_ = 0;

    T threshold_ = T(0.10);
    T frequency_ = T(0);
    T confidence_ = T(0);

    std::vector<T> buffer_;
    std::vector<T> yinBuffer_;
};

} // namespace dspark
