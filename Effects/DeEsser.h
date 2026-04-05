// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DeEsser.h
 * @brief Frequency-selective dynamic processor for sibilance reduction.
 *
 * A split-band de-esser that detects energy in a configurable frequency band
 * (typically 4–10 kHz) and applies dynamic gain reduction only in that band.
 * More transparent than wideband de-essing because non-sibilant content
 * passes through unaffected.
 *
 * Architecture:
 * 1. Bandpass sidechain (Biquad) isolates the sibilant band.
 * 2. Envelope follower (peak with attack/release) tracks sibilance level.
 * 3. When level exceeds threshold, a parametric bell cut is applied.
 *
 * Dependencies: Biquad.h, DspMath.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::DeEsser<float> deesser;
 *   deesser.prepare(spec);
 *   deesser.setFrequency(7000.0f);   // centre of sibilance band
 *   deesser.setThreshold(-20.0f);     // dB threshold
 *   deesser.setReduction(12.0f);      // max reduction in dB
 *
 *   // In audio callback:
 *   deesser.processBlock(buffer);
 *   float gr = deesser.getGainReductionDb();  // for metering
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DspMath.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

namespace dspark {

/**
 * @class DeEsser
 * @brief Split-band de-esser with dynamic sibilance detection.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class DeEsser
{
public:
    /** @brief Detection mode for sibilance identification. */
    enum class DetectionMode
    {
        Bandpass,    ///< Standard bandpass filter detection (default).
        Derivative   ///< Multi-derivative cascade (DeBess-style). Only sustained sibilance triggers.
    };

    /**
     * @brief Prepares the de-esser.
     * @param spec Audio environment specification.
     */
    void prepare(const AudioSpec& spec)
    {
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        // Attack ~0.5 ms, release ~20 ms
        attackCoeff_ = static_cast<T>(
            1.0 - std::exp(-1.0 / (sampleRate_ * 0.0005)));
        releaseCoeff_ = static_cast<T>(
            1.0 - std::exp(-1.0 / (sampleRate_ * 0.020)));

        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            detector_[ch].reset();
            reduction_[ch].reset();
            envelope_[ch] = T(0);
            derivShift_[ch].fill(T(0));
        }

        updateCoefficients();
        gainReduction_.store(T(0), std::memory_order_relaxed);
    }

    /**
     * @brief Processes audio in-place (applies de-essing).
     * @param buffer Audio data.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        int numCh = std::min(buffer.getNumChannels(),
                             std::min(numChannels_, kMaxChannels));
        int numSamples = buffer.getNumSamples();

        // Sync atomics
        T freq    = frequency_.load(std::memory_order_relaxed);
        T bw      = bandwidth_.load(std::memory_order_relaxed);
        T thresh  = threshold_.load(std::memory_order_relaxed);
        T maxRed  = maxReduction_.load(std::memory_order_relaxed);
        auto mode = detectionMode_.load(std::memory_order_relaxed);

        // Update detector filter coefficients
        auto bpCoeffs = BiquadCoeffs<T>::makeBandPass(
            sampleRate_, static_cast<double>(freq), static_cast<double>(bw));
        for (int ch = 0; ch < numCh; ++ch)
            detector_[ch].setCoeffs(bpCoeffs);

        T maxGr = T(0);

        for (int ch = 0; ch < numCh; ++ch)
        {
            T* data = buffer.getChannel(ch);

            for (int i = 0; i < numSamples; ++i)
            {
                T level;

                if (mode == DetectionMode::Derivative)
                {
                    // DeBess-style multi-derivative cascade:
                    // Shift register + products of adjacent first differences.
                    // Only sustained high-frequency content survives the cascade.
                    auto& sr = derivShift_[ch];
                    for (int k = kDerivLen - 1; k > 0; --k)
                        sr[k] = sr[k - 1];
                    sr[0] = detector_[ch].processSample(data[i], 0);

                    // Product of adjacent differences
                    T product = T(1);
                    for (int k = 0; k < kDerivLen - 1; ++k)
                    {
                        T diff = (sr[k] - sr[k + 1]) * (k > 0 ? (sr[k - 1] - sr[k]) : sr[k]);
                        product *= std::clamp(diff, T(-1), T(1));
                    }
                    level = std::abs(product);
                }
                else
                {
                    // Standard bandpass detection
                    T sidechain = detector_[ch].processSample(data[i], 0);
                    level = std::abs(sidechain);
                }

                // Envelope follower (peak)
                T coeff = (level > envelope_[ch]) ? attackCoeff_ : releaseCoeff_;
                envelope_[ch] += coeff * (level - envelope_[ch]);

                // Compute gain reduction
                T envDb = gainToDecibels(envelope_[ch] + T(1e-30));
                T overDb = envDb - thresh;

                T grDb = T(0);
                if (overDb > T(0))
                    grDb = -std::min(overDb, maxRed);

                // Apply reduction via dynamic bell filter
                auto peakCoeffs = BiquadCoeffs<T>::makePeak(
                    sampleRate_, static_cast<double>(freq),
                    static_cast<double>(bw),
                    static_cast<double>(grDb < T(-0.1) ? grDb : T(0)));
                reduction_[ch].setCoeffs(peakCoeffs);
                data[i] = reduction_[ch].processSample(data[i], 0);

                if (-grDb > maxGr)
                    maxGr = -grDb;
            }
        }

        gainReduction_.store(maxGr, std::memory_order_relaxed);
    }

    /** @brief Resets internal state. */
    void reset() noexcept
    {
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            detector_[ch].reset();
            reduction_[ch].reset();
            envelope_[ch] = T(0);
            derivShift_[ch].fill(T(0));
        }
        gainReduction_.store(T(0), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the centre frequency of the sibilance band.
     * @param hz Typically 4000 – 10000 Hz (default: 7000).
     */
    void setFrequency(T hz) noexcept
    {
        frequency_.store(hz, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the detection bandwidth.
     * @param octaves Width in octaves (default: 1.5).
     */
    void setBandwidth(T octaves) noexcept
    {
        T bw = std::max(octaves, T(0.1));
        T q = T(1) / (T(2) * std::sinh(T(0.34657359) * bw));
        bandwidth_.store(q, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the detection threshold.
     * @param db Threshold in dB (typically -30 to -10).
     */
    void setThreshold(T db) noexcept { threshold_.store(db, std::memory_order_relaxed); }

    /**
     * @brief Sets the maximum gain reduction.
     * @param db Maximum cut in dB (positive value, e.g. 12.0).
     */
    void setReduction(T db) noexcept { maxReduction_.store(std::abs(db), std::memory_order_relaxed); }

    /**
     * @brief Sets the detection mode.
     *
     * - **Bandpass** (default): Standard bandpass-filtered envelope detection.
     * - **Derivative**: Multi-derivative cascade (DeBess/Airwindows-style).
     *   Only sustained sibilance survives the cascade of difference products,
     *   making it more selective than simple bandpass detection.
     */
    void setDetectionMode(DetectionMode mode) noexcept
    {
        detectionMode_.store(mode, std::memory_order_relaxed);
    }

    /** @brief Returns the current gain reduction in dB (positive value). */
    [[nodiscard]] T getGainReductionDb() const noexcept { return gainReduction_.load(std::memory_order_relaxed); }

    [[nodiscard]] T getFrequency() const noexcept { return frequency_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getThreshold() const noexcept { return threshold_.load(std::memory_order_relaxed); }
    [[nodiscard]] DetectionMode getDetectionMode() const noexcept { return detectionMode_.load(std::memory_order_relaxed); }

private:
    void updateCoefficients() noexcept
    {
        T freq = frequency_.load(std::memory_order_relaxed);
        T bw = bandwidth_.load(std::memory_order_relaxed);
        auto c = BiquadCoeffs<T>::makeBandPass(
            sampleRate_, static_cast<double>(freq), static_cast<double>(bw));
        for (auto& d : detector_)
            d.setCoeffs(c);
    }

    static constexpr int kMaxChannels = 2;
    static constexpr int kDerivLen = 8;  ///< Shift register length for derivative detection

    double sampleRate_ = 44100.0;
    int numChannels_ = 2;

    // Atomic parameters
    std::atomic<T> frequency_ { T(7000) };
    std::atomic<T> bandwidth_ { T(2) };       // Q value
    std::atomic<T> threshold_ { T(-20) };
    std::atomic<T> maxReduction_ { T(12) };
    std::atomic<DetectionMode> detectionMode_ { DetectionMode::Bandpass };

    T attackCoeff_ = T(0);
    T releaseCoeff_ = T(0);
    std::atomic<T> gainReduction_ { T(0) };

    Biquad<T, 1> detector_[kMaxChannels]{};
    Biquad<T, 1> reduction_[kMaxChannels]{};
    T envelope_[kMaxChannels]{};

    // Derivative detection shift registers
    std::array<std::array<T, kDerivLen>, kMaxChannels> derivShift_ {};
};

} // namespace dspark
