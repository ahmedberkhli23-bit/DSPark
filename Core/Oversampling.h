// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Oversampling.h
 * @brief Real-time oversampling with FIR half-band anti-aliasing filters.
 *
 * Reduces aliasing in nonlinear processes (saturation, waveshaping, distortion)
 * by upsampling the signal, processing at a higher rate, then downsampling with
 * high-quality anti-aliasing filters.
 *
 * Uses cascaded FIR half-band filters designed with the Kaiser window method.
 * Each 2x stage uses a symmetric half-band FIR whose even-indexed coefficients
 * (except the center tap) are exactly zero, enabling efficient computation at
 * roughly N/4 multiply-accumulates per sample instead of N (half-band zeros +
 * linear-phase symmetry folding combined).
 *
 * Four quality presets control stopband rejection:
 *
 * | Quality   | Taps/stage | Rejection | Passband* | Latency (4x) |
 * |-----------|------------|-----------|-----------|--------------|
 * | Low       | 31         | ~-40 dB   | ~0.88 Ny  | ~23 samples  |
 * | Medium    | 63         | ~-60 dB   | ~0.92 Ny  | ~47 samples  |
 * | High      | 127        | ~-80 dB   | ~0.96 Ny  | ~95 samples  |
 * | Maximum   | 255        | ~-100 dB  | ~0.97 Ny  | ~191 samples |
 *
 * (*) Passband = fraction of original Nyquist preserved within 0.1 dB.
 *
 * Factor must be a power of two (1, 2, 4, 8, 16).
 *
 * Dependencies: AudioBuffer.h, FIRFilter.h (for FIRDesign + Kaiser window), AudioSpec.h.
 *
 * @code
 *   dspark::Oversampling<float> os(4);  // 4x oversampling, High quality (-80 dB)
 *   os.prepare(spec);
 *
 *   // In process():
 *   auto upView = os.upsample(buffer);
 *   applyDistortion(upView);            // process at 4x sample rate
 *   os.downsample(buffer);
 *
 *   // Mastering-grade quality:
 *   dspark::Oversampling<float> os(4, dspark::Oversampling<float>::Quality::Maximum);
 * @endcode
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"
#include "FIRFilter.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <vector>

namespace dspark {

/**
 * @class Oversampling
 * @brief Power-of-two oversampling with cascaded FIR half-band anti-aliasing.
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum number of channels.
 */
template <typename T, int MaxChannels = 16>
class Oversampling
{
public:
    /**
     * @brief Quality presets controlling anti-aliasing filter order and rejection.
     *
     * Higher quality uses more FIR taps per stage, giving steeper rolloff and
     * deeper stopband rejection at the cost of more CPU and latency.
     */
    enum class Quality
    {
        Low,      ///< 31 taps/stage, ~-40 dB. Fast preview, low CPU.
        Medium,   ///< 63 taps/stage, ~-60 dB. Good balance for most uses.
        High,     ///< 127 taps/stage, ~-80 dB. Professional quality (default).
        Maximum   ///< 255 taps/stage, ~-100 dB. Mastering grade.
    };

    /**
     * @brief Constructs an oversampling processor.
     * @param factor  Oversampling factor (power of two: 1, 2, 4, 8, 16).
     * @param quality Anti-aliasing filter quality preset.
     */
    explicit Oversampling(int factor = 2, Quality quality = Quality::High)
        : factor_(factor), quality_(quality)
    {
        assert(factor >= 1 && (factor & (factor - 1)) == 0);
        numStages_ = 0;
        int f = factor;
        while (f > 1) { ++numStages_; f >>= 1; }
    }

    /**
     * @brief Prepares internal buffers and designs anti-aliasing filters.
     *
     * Allocates the oversampled buffer and computes FIR half-band coefficients
     * for each 2x stage using Kaiser-windowed sinc design. All memory allocation
     * happens here — process functions are allocation-free.
     *
     * @param spec Audio spec at the base (non-oversampled) rate.
     */
    void prepare(const AudioSpec& spec)
    {
        baseSpec_ = spec;
        upBuffer_.resize(spec.numChannels, spec.maxBlockSize * factor_);

        const int taps = tapsForQuality(quality_);
        const T beta = betaForQuality(quality_);

        for (int stage = 0; stage < numStages_; ++stage)
        {
            upFilters_[stage].design(taps, beta, spec.numChannels);
            downFilters_[stage].design(taps, beta, spec.numChannels);
        }

        reset();
    }

    /** @brief Resets all filter states and clears the oversampled buffer. */
    void reset() noexcept
    {
        upBuffer_.clear();
        for (int i = 0; i < kMaxStages; ++i)
        {
            upFilters_[i].reset();
            downFilters_[i].reset();
        }
    }

    /** @brief Returns the oversampling factor. */
    [[nodiscard]] int getFactor() const noexcept { return factor_; }

    /** @brief Returns the current quality preset. */
    [[nodiscard]] Quality getQuality() const noexcept { return quality_; }

    /**
     * @brief Returns the total latency in base-rate samples.
     *
     * Each stage contributes (numTaps - 1) / 2 samples of group delay at its
     * oversampled rate, for both the upsample and downsample filters. The total
     * is accumulated at the maximum oversampled rate and then divided back to
     * base-rate samples.
     *
     * Formula: latency = (numTaps - 1) * (1 - 1 / 2^numStages) base samples.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        if (numStages_ == 0) return 0;

        const int halfOrder = (tapsForQuality(quality_) - 1) / 2;

        // Accumulate delay at the maximum oversampled rate, then convert to base.
        // Stage k operates at 2^(k+1) * base rate. Each filter (up + down) adds
        // halfOrder samples at that rate = halfOrder * (factor / 2^(k+1)) at max rate.
        int totalAtMaxRate = 0;
        for (int stage = 0; stage < numStages_; ++stage)
        {
            int rateRatio = factor_ / (1 << (stage + 1));
            totalAtMaxRate += 2 * halfOrder * rateRatio;
        }

        // Convert to base-rate samples (ceiling for safe compensation)
        return (totalAtMaxRate + factor_ - 1) / factor_;
    }

    /**
     * @brief Upsamples the input buffer into the internal oversampled buffer.
     *
     * For each 2x stage: zero-stuffs between samples (with 2x gain compensation)
     * and applies the FIR half-band anti-imaging filter.
     *
     * @param input Base-rate audio buffer.
     * @return View to the oversampled buffer (factor * input samples).
     */
    [[nodiscard]] AudioBufferView<T> upsample(AudioBufferView<const T> input) noexcept
    {
        const int nCh = std::min(input.getNumChannels(), upBuffer_.getNumChannels());
        const int nS  = input.getNumSamples();

        if (numStages_ == 0)
        {
            for (int ch = 0; ch < nCh; ++ch)
                std::memcpy(upBuffer_.getChannel(ch), input.getChannel(ch),
                           static_cast<std::size_t>(nS) * sizeof(T));
            return upBuffer_.toView().getSubView(0, nS);
        }

        // Stage 0: zero-stuff from input
        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* src = input.getChannel(ch);
            T* dst = upBuffer_.getChannel(ch);
            for (int i = 0; i < nS; ++i)
            {
                dst[i * 2]     = src[i] * T(2); // Gain compensation for zero-stuffing
                dst[i * 2 + 1] = T(0);
            }
        }

        int currentLen = nS * 2;
        filterBlock(upFilters_[0], nCh, currentLen);

        // Additional stages: zero-stuff in-place (work backwards to avoid overwrite)
        for (int stage = 1; stage < numStages_; ++stage)
        {
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = upBuffer_.getChannel(ch);
                for (int i = currentLen - 1; i >= 0; --i)
                {
                    data[i * 2]     = data[i] * T(2);
                    data[i * 2 + 1] = T(0);
                }
            }
            currentLen *= 2;
            filterBlock(upFilters_[stage], nCh, currentLen);
        }

        return upBuffer_.toView().getSubView(0, nS * factor_);
    }

    /** @brief Overload accepting a mutable input view. */
    [[nodiscard]] AudioBufferView<T> upsample(AudioBufferView<T> input) noexcept
    {
        const int nCh = std::min(input.getNumChannels(), upBuffer_.getNumChannels());
        const int nS  = input.getNumSamples();

        if (numStages_ == 0)
        {
            for (int ch = 0; ch < nCh; ++ch)
                std::memcpy(upBuffer_.getChannel(ch), input.getChannel(ch),
                           static_cast<std::size_t>(nS) * sizeof(T));
            return upBuffer_.toView().getSubView(0, nS);
        }

        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* src = input.getChannel(ch);
            T* dst = upBuffer_.getChannel(ch);
            for (int i = 0; i < nS; ++i)
            {
                dst[i * 2]     = src[i] * T(2);
                dst[i * 2 + 1] = T(0);
            }
        }

        int currentLen = nS * 2;
        filterBlock(upFilters_[0], nCh, currentLen);

        for (int stage = 1; stage < numStages_; ++stage)
        {
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = upBuffer_.getChannel(ch);
                for (int i = currentLen - 1; i >= 0; --i)
                {
                    data[i * 2]     = data[i] * T(2);
                    data[i * 2 + 1] = T(0);
                }
            }
            currentLen *= 2;
            filterBlock(upFilters_[stage], nCh, currentLen);
        }

        return upBuffer_.toView().getSubView(0, nS * factor_);
    }

    /**
     * @brief Downsamples the internal oversampled buffer back to the output.
     *
     * Applies the FIR half-band anti-aliasing filter then decimates at each stage,
     * processed in reverse order from the highest rate down to the base rate.
     *
     * @param output Base-rate audio buffer to write the downsampled result.
     */
    void downsample(AudioBufferView<T> output) noexcept
    {
        const int nCh = std::min(output.getNumChannels(), upBuffer_.getNumChannels());
        const int nS  = output.getNumSamples();

        if (numStages_ == 0)
        {
            for (int ch = 0; ch < nCh; ++ch)
                std::memcpy(output.getChannel(ch), upBuffer_.getChannel(ch),
                           static_cast<std::size_t>(nS) * sizeof(T));
            return;
        }

        int currentLen = nS * factor_;

        // Apply stages in reverse (highest rate first)
        for (int stage = numStages_ - 1; stage >= 0; --stage)
        {
            filterBlock(downFilters_[stage], nCh, currentLen);

            // Decimate by 2
            int newLen = currentLen / 2;
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = upBuffer_.getChannel(ch);
                for (int i = 0; i < newLen; ++i)
                    data[i] = data[i * 2];
            }
            currentLen = newLen;
        }

        for (int ch = 0; ch < nCh; ++ch)
            std::memcpy(output.getChannel(ch), upBuffer_.getChannel(ch),
                       static_cast<std::size_t>(nS) * sizeof(T));
    }

    /** @brief Returns a mutable view to the oversampled buffer (after upsample). */
    [[nodiscard]] AudioBufferView<T> getOversampledView(int baseSamples) noexcept
    {
        return upBuffer_.toView().getSubView(0, baseSamples * factor_);
    }

private:
    static constexpr int kMaxStages = 4; // Up to 16x

    // ========================================================================
    // HalfBandFIR — Symmetric half-band FIR with zero-skip + symmetry folding
    // ========================================================================
    //
    // Exploits two mathematical properties of half-band FIR filters:
    //
    // 1. Linear-phase symmetry:  h[center-k] = h[center+k]
    //    -> Fold symmetric delay line pairs before multiplying.
    //
    // 2. Half-band zeros:  h[center +/- 2k] = 0 for all k >= 1
    //    (because sinc(0.5 * pi * 2k) = sin(k*pi) = 0)
    //    -> Only odd-offset coefficients are non-zero; skip even offsets.
    //
    // Combined: ~N/4 MACs per sample instead of N (4x speedup).
    // ========================================================================

    struct HalfBandFIR
    {
        std::vector<T> oddCoeffs;  // Unique non-zero coefficients: h[center+1], h[center+3], ...
        T centerCoeff = T(0);     // Center tap (approximately 0.5 for half-band)
        int numTaps = 0;
        int halfOrder = 0;        // (numTaps - 1) / 2
        int numOddCoeffs = 0;

        struct ChannelState
        {
            std::vector<T> delayLine;
            int writePos = 0;
        };
        std::vector<ChannelState> channels;

        void design(int taps, T beta, int numChannels)
        {
            numTaps = taps;
            halfOrder = (taps - 1) / 2;

            // Design half-band lowpass: cutoff at 0.25 * sampleRate (half-Nyquist).
            // Using canonical sampleRate=1.0 since half-band coefficients are rate-independent.
            auto fullCoeffs = FIRDesign<T>::lowPass(1.0, 0.25, taps, beta);

            centerCoeff = fullCoeffs[static_cast<std::size_t>(halfOrder)];

            // Extract unique non-zero odd-offset coefficients.
            // By symmetry h[center-k] == h[center+k], so store only one side.
            oddCoeffs.clear();
            for (int k = 1; k <= halfOrder; k += 2)
                oddCoeffs.push_back(fullCoeffs[static_cast<std::size_t>(halfOrder + k)]);

            numOddCoeffs = static_cast<int>(oddCoeffs.size());

            channels.resize(static_cast<std::size_t>(numChannels));
            for (auto& ch : channels)
            {
                ch.delayLine.assign(static_cast<std::size_t>(numTaps), T(0));
                ch.writePos = 0;
            }
        }

        void reset() noexcept
        {
            for (auto& ch : channels)
            {
                std::fill(ch.delayLine.begin(), ch.delayLine.end(), T(0));
                ch.writePos = 0;
            }
        }

        T processSample(T input, int channel) noexcept
        {
            auto& state = channels[static_cast<std::size_t>(channel)];
            auto& dl = state.delayLine;

            // Push sample into ring buffer
            dl[static_cast<std::size_t>(state.writePos)] = input;
            if (++state.writePos >= numTaps) state.writePos = 0;

            // Read center of the delay line (halfOrder samples back)
            int centerPos = state.writePos - 1 - halfOrder;
            if (centerPos < 0) centerPos += numTaps;

            T output = centerCoeff * dl[static_cast<std::size_t>(centerPos)];

            // Symmetric non-zero taps: k = 1, 3, 5, ...
            for (int i = 0; i < numOddCoeffs; ++i)
            {
                int k = i * 2 + 1;

                int posLeft = centerPos - k;
                if (posLeft < 0) posLeft += numTaps;

                int posRight = centerPos + k;
                if (posRight >= numTaps) posRight -= numTaps;

                output += oddCoeffs[static_cast<std::size_t>(i)]
                        * (dl[static_cast<std::size_t>(posLeft)]
                         + dl[static_cast<std::size_t>(posRight)]);
            }

            return output;
        }
    };

    // ========================================================================
    // Quality preset tables
    // ========================================================================

    static constexpr int tapsForQuality(Quality q) noexcept
    {
        switch (q)
        {
            case Quality::Low:     return 31;   // halfOrder = 15
            case Quality::Medium:  return 63;   // halfOrder = 31
            case Quality::High:    return 127;  // halfOrder = 63
            case Quality::Maximum: return 255;  // halfOrder = 127
        }
        return 127;
    }

    static constexpr T betaForQuality(Quality q) noexcept
    {
        // Kaiser beta from FIRDesign::estimateKaiserBeta():
        //   A > 50 dB:  beta = 0.1102 * (A - 8.7)
        //   21 <= A <= 50: beta = 0.5842*(A-21)^0.4 + 0.07886*(A-21)
        switch (q)
        {
            case Quality::Low:     return T(3.395);   // A ~ 40 dB
            case Quality::Medium:  return T(5.653);   // A ~ 60 dB
            case Quality::High:    return T(7.857);   // A ~ 80 dB
            case Quality::Maximum: return T(10.056);  // A ~ 100 dB
        }
        return T(7.857);
    }

    void filterBlock(HalfBandFIR& filter, int nCh, int nS) noexcept
    {
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* data = upBuffer_.getChannel(ch);
            for (int i = 0; i < nS; ++i)
                data[i] = filter.processSample(data[i], ch);
        }
    }

    int factor_;
    int numStages_;
    Quality quality_;
    AudioSpec baseSpec_ {};
    AudioBuffer<T, MaxChannels> upBuffer_;

    std::array<HalfBandFIR, kMaxStages> upFilters_;
    std::array<HalfBandFIR, kMaxStages> downFilters_;
};

} // namespace dspark
