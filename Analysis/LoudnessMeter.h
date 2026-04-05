// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file LoudnessMeter.h
 * @brief EBU R128 / ITU-R BS.1770 loudness meter.
 *
 * Implements the international standard for loudness measurement used in
 * broadcast, streaming, and mastering. Measures loudness in LUFS (Loudness
 * Units relative to Full Scale) with three time scales.
 *
 * Measurements:
 * - **Momentary:** 400 ms sliding window (fast response for level tracking)
 * - **Short-term:** 3 second sliding window (medium-term loudness)
 * - **Integrated:** Entire programme with gating (overall loudness)
 *
 * The signal chain is: K-weighting filter → mean square → gating → LUFS.
 *
 * K-weighting consists of two cascaded biquad filters:
 * 1. Pre-filter (high shelf at ~1681 Hz, +3.9997 dB) — models head diffraction
 * 2. RLB (revised low-frequency B-weighting) high-pass at ~38.1 Hz
 *
 * Dependencies: DspMath.h.
 *
 * @code
 *   dspark::LoudnessMeter<float> meter;
 *   meter.prepare(48000.0, 2);  // 48 kHz, stereo
 *
 *   // In audio callback:
 *   meter.process(leftData, rightData, numSamples);
 *
 *   // Read loudness:
 *   float momentary = meter.getMomentaryLUFS();
 *   float shortTerm = meter.getShortTermLUFS();
 *   float integrated = meter.getIntegratedLUFS();
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class LoudnessMeter
 * @brief EBU R128 loudness meter with momentary, short-term, and integrated measurements.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class LoudnessMeter
{
public:
    /**
     * @brief Prepares the loudness meter.
     *
     * @param sampleRate Sample rate in Hz (must be 44100 or 48000 for standard compliance).
     * @param numChannels Number of channels (1 = mono, 2 = stereo).
     */
    void prepare(double sampleRate, int numChannels = 2)
    {
        sampleRate_ = sampleRate;
        numChannels_ = std::min(numChannels, kMaxChannels);

        // Compute K-weighting filter coefficients for this sample rate
        computeKWeighting(sampleRate);

        // Calculate block sizes for 100ms measurement blocks
        // (EBU R128 specifies 400ms with 75% overlap = 100ms hop)
        blockSamples_ = static_cast<int>(sampleRate * 0.1); // 100 ms
        momentaryBlocks_ = 4;  // 4 * 100ms = 400ms
        shortTermBlocks_ = 30; // 30 * 100ms = 3s

        // Allocate ring buffer for block powers
        blockPowers_.assign(static_cast<size_t>(shortTermBlocks_), T(0));
        blockWritePos_ = 0;
        currentBlockPower_ = T(0);
        currentBlockSamples_ = 0;

        // Integrated loudness gating — pre-allocate for up to 2 hours at 100ms blocks
        constexpr size_t kMaxBlocks = 72000; // 2h at 100ms per block
        allBlockPowers_.assign(kMaxBlocks, T(0));
        allBlockCount_ = 0;
        gatedPowerSum_ = 0.0;
        gatedBlockCount_ = 0;

        reset();
    }

    /** @brief Prepares from AudioSpec (unified API). */
    void prepare(const AudioSpec& spec)
    {
        prepare(spec.sampleRate, spec.numChannels);
    }

    /**
     * @brief Processes an AudioBufferView (unified API, read-only).
     * @param buffer Audio buffer.
     */
    void processBlock(AudioBufferView<const T> buffer) noexcept
    {
        const int nCh = buffer.getNumChannels();
        const int nS = buffer.getNumSamples();
        if (nCh >= 2)
            process(buffer.getChannel(0), buffer.getChannel(1), nS);
        else if (nCh == 1)
            process(buffer.getChannel(0), nS);
    }

    /**
     * @brief Processes mono audio.
     * @param data Audio samples.
     * @param numSamples Number of samples.
     */
    void process(const T* data, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            T filtered = applyKWeighting(data[i], 0);
            currentBlockPower_ += filtered * filtered;
            ++currentBlockSamples_;

            if (currentBlockSamples_ >= blockSamples_)
                commitBlock();
        }
    }

    /**
     * @brief Processes stereo audio.
     * @param left Left channel samples.
     * @param right Right channel samples.
     * @param numSamples Number of samples per channel.
     */
    void process(const T* left, const T* right, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            T filtL = applyKWeighting(left[i], 0);
            T filtR = applyKWeighting(right[i], 1);

            // Sum of per-channel mean-square (ITU-R BS.1770-4: sum, not average)
            T power = filtL * filtL + filtR * filtR;
            currentBlockPower_ += power;
            ++currentBlockSamples_;

            if (currentBlockSamples_ >= blockSamples_)
                commitBlock();
        }
    }

    // -- Loudness readouts --------------------------------------------------------

    /**
     * @brief Returns the momentary loudness (400 ms window).
     * @return Loudness in LUFS.
     */
    [[nodiscard]] T getMomentaryLUFS() const noexcept
    {
        T sum = T(0);
        int count = 0;
        for (int i = 0; i < momentaryBlocks_; ++i)
        {
            int idx = (blockWritePos_ - 1 - i + static_cast<int>(blockPowers_.size()))
                      % static_cast<int>(blockPowers_.size());
            sum += blockPowers_[static_cast<size_t>(idx)];
            ++count;
        }

        if (count == 0) return T(-100);
        T meanPower = sum / static_cast<T>(count);
        return powerToLUFS(meanPower);
    }

    /**
     * @brief Returns the short-term loudness (3 second window).
     * @return Loudness in LUFS.
     */
    [[nodiscard]] T getShortTermLUFS() const noexcept
    {
        T sum = T(0);
        int count = std::min(shortTermBlocks_, static_cast<int>(blockPowers_.size()));
        for (int i = 0; i < count; ++i)
        {
            int idx = (blockWritePos_ - 1 - i + static_cast<int>(blockPowers_.size()))
                      % static_cast<int>(blockPowers_.size());
            sum += blockPowers_[static_cast<size_t>(idx)];
        }

        if (count == 0) return T(-100);
        T meanPower = sum / static_cast<T>(count);
        return powerToLUFS(meanPower);
    }

    /**
     * @brief Returns the integrated loudness (entire programme with gating).
     *
     * Uses the two-pass gating algorithm specified by EBU R128:
     * 1. Absolute gate at -70 LUFS
     * 2. Relative gate at -10 LU below ungated mean
     *
     * @return Integrated loudness in LUFS.
     */
    [[nodiscard]] T getIntegratedLUFS() const noexcept
    {
        if (allBlockCount_ == 0) return T(-100);

        // Pass 1: absolute gate at -70 LUFS
        T absGatePower = lufsTopower(T(-70));
        double sum1 = 0.0;
        int count1 = 0;

        for (int b = 0; b < allBlockCount_; ++b)
        {
            T p = allBlockPowers_[static_cast<size_t>(b)];
            if (p > absGatePower)
            {
                sum1 += static_cast<double>(p);
                ++count1;
            }
        }

        if (count1 == 0) return T(-100);

        // Pass 2: relative gate at ungatedMean - 10 LU
        T ungatedMean = static_cast<T>(sum1 / count1);
        T relGateLUFS = powerToLUFS(ungatedMean) - T(10);
        T relGatePower = lufsTopower(relGateLUFS);

        double sum2 = 0.0;
        int count2 = 0;

        for (int b = 0; b < allBlockCount_; ++b)
        {
            T p = allBlockPowers_[static_cast<size_t>(b)];
            if (p > relGatePower)
            {
                sum2 += static_cast<double>(p);
                ++count2;
            }
        }

        if (count2 == 0) return T(-100);
        return powerToLUFS(static_cast<T>(sum2 / count2));
    }

    /**
     * @brief Resets the meter state.
     */
    void reset() noexcept
    {
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            preState_[ch] = {};
            rlbState_[ch] = {};
        }

        std::fill(blockPowers_.begin(), blockPowers_.end(), T(0));
        blockWritePos_ = 0;
        currentBlockPower_ = T(0);
        currentBlockSamples_ = 0;
        allBlockCount_ = 0;
        gatedPowerSum_ = 0.0;
        gatedBlockCount_ = 0;
    }

private:
    static constexpr int kMaxChannels = 8;

    // K-weighting biquad state
    struct BiquadState
    {
        T z1 = T(0), z2 = T(0);
    };

    // K-weighting biquad coefficients
    struct BiquadCoeff
    {
        T b0 = T(1), b1 = T(0), b2 = T(0);
        T a1 = T(0), a2 = T(0);
    };

    void computeKWeighting(double sr)
    {
        // Stage 1: Pre-filter (high shelf)
        // ITU-R BS.1770-4 coefficients derived analytically
        double fc = 1681.97;
        double G = 3.99984;  // dB
        double Q = 0.7071067811865476;

        double A = std::pow(10.0, G / 40.0);
        double w0 = 2.0 * std::numbers::pi * fc / sr;
        double cosw0 = std::cos(w0);
        double sinw0 = std::sin(w0);
        double alpha = sinw0 / (2.0 * Q);

        double a0 = (A + 1.0) - (A - 1.0) * cosw0 + 2.0 * std::sqrt(A) * alpha;
        pre_.b0 = static_cast<T>(( A * ((A + 1.0) + (A - 1.0) * cosw0 + 2.0 * std::sqrt(A) * alpha)) / a0);
        pre_.b1 = static_cast<T>((-2.0 * A * ((A - 1.0) + (A + 1.0) * cosw0)) / a0);
        pre_.b2 = static_cast<T>(( A * ((A + 1.0) + (A - 1.0) * cosw0 - 2.0 * std::sqrt(A) * alpha)) / a0);
        pre_.a1 = static_cast<T>(( 2.0 * ((A - 1.0) - (A + 1.0) * cosw0)) / a0);
        pre_.a2 = static_cast<T>(((A + 1.0) - (A - 1.0) * cosw0 - 2.0 * std::sqrt(A) * alpha) / a0);

        // Stage 2: RLB high-pass
        double fc2 = 38.13547087602444;
        double Q2 = 0.5003270373238773;

        double w02 = 2.0 * std::numbers::pi * fc2 / sr;
        double cosw02 = std::cos(w02);
        double sinw02 = std::sin(w02);
        double alpha2 = sinw02 / (2.0 * Q2);

        double a02 = 1.0 + alpha2;
        rlb_.b0 = static_cast<T>((1.0 + cosw02) / 2.0 / a02);
        rlb_.b1 = static_cast<T>(-(1.0 + cosw02) / a02);
        rlb_.b2 = static_cast<T>((1.0 + cosw02) / 2.0 / a02);
        rlb_.a1 = static_cast<T>(-2.0 * cosw02 / a02);
        rlb_.a2 = static_cast<T>((1.0 - alpha2) / a02);
    }

    T applyBiquad(T input, const BiquadCoeff& c, BiquadState& s) noexcept
    {
        T output = c.b0 * input + s.z1;
        s.z1 = c.b1 * input - c.a1 * output + s.z2;
        s.z2 = c.b2 * input - c.a2 * output;
        return output;
    }

    T applyKWeighting(T input, int channel) noexcept
    {
        T x = applyBiquad(input, pre_, preState_[channel]);
        return applyBiquad(x, rlb_, rlbState_[channel]);
    }

    void commitBlock() noexcept
    {
        T meanPower = currentBlockPower_ / static_cast<T>(currentBlockSamples_);

        blockPowers_[static_cast<size_t>(blockWritePos_)] = meanPower;
        blockWritePos_ = (blockWritePos_ + 1) % static_cast<int>(blockPowers_.size());

        // Store for integrated loudness (pre-allocated, no RT heap alloc)
        if (allBlockCount_ < static_cast<int>(allBlockPowers_.size()))
        {
            allBlockPowers_[static_cast<size_t>(allBlockCount_)] = meanPower;
            ++allBlockCount_;
        }
        // After buffer full, stop accumulating new blocks for integrated loudness.
        // The 2-hour window is sufficient for all practical broadcast use cases.

        currentBlockPower_ = T(0);
        currentBlockSamples_ = 0;
    }

    [[nodiscard]] static T powerToLUFS(T meanPower) noexcept
    {
        if (meanPower <= T(0)) return T(-100);
        return T(-0.691) + T(10) * std::log10(meanPower);
    }

    [[nodiscard]] static T lufsTopower(T lufs) noexcept
    {
        return std::pow(T(10), (lufs + T(0.691)) / T(10));
    }

    double sampleRate_ = 48000.0;
    int numChannels_ = 2;

    // K-weighting filters
    BiquadCoeff pre_, rlb_;
    BiquadState preState_[kMaxChannels], rlbState_[kMaxChannels];

    // Block measurement
    int blockSamples_ = 4800;      // 100 ms at 48 kHz
    int momentaryBlocks_ = 4;       // 400 ms
    int shortTermBlocks_ = 30;      // 3 s
    std::vector<T> blockPowers_;
    int blockWritePos_ = 0;
    T currentBlockPower_ = T(0);
    int currentBlockSamples_ = 0;

    // Integrated loudness (pre-allocated, no RT heap allocation)
    std::vector<T> allBlockPowers_;
    int allBlockCount_ = 0;
    double gatedPowerSum_ = 0.0;
    int gatedBlockCount_ = 0;
};

} // namespace dspark
