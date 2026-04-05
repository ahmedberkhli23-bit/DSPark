// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Filters.h
 * @brief Multi-mode cascaded biquad filter engine with real-time parameter smoothing.
 *
 * Supports all standard filter shapes (LP, HP, BP, Peak, Shelves, Notch, Tilt,
 * AllPass) with configurable slopes from 6 to 48 dB/oct via cascaded biquad stages.
 * Parameters are smoothed per-sample to prevent zipper noise.
 *
 * Dependencies: Biquad.h, AudioBuffer.h, AudioSpec.h, Smoothers.h, AnalogRandom.h.
 *
 * @code
 *   dspark::FilterEngine<float> filter;
 *   filter.prepare(spec);
 *   filter.setLowPass(2000.0f, 0.707f, 24);  // 2kHz, Butterworth Q, 24dB/oct
 *   filter.processBlock(buffer);
 *
 *   // Real-time parameter changes (smoothed):
 *   filter.setFrequency(4000.0f);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DspMath.h"
#include "../Core/Smoothers.h"
#include "../Core/AnalogRandom.h"
#include "../Core/DenormalGuard.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

namespace dspark {

/**
 * @class FilterEngine
 * @brief Professional multi-mode filter with cascaded biquad stages.
 *
 * Each biquad stage provides 12 dB/oct (2nd order). Cascading N stages
 * gives 12*N dB/oct. For odd-order slopes (6, 18, 30, 42 dB/oct), the
 * last stage uses a 1st-order (6 dB) filter approximation.
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum number of audio channels.
 */
template <typename T, int MaxChannels = 16>
class FilterEngine
{
public:
    virtual ~FilterEngine() = default;
    enum class Shape
    {
        LowPass, HighPass, BandPass, Peak,
        LowShelf, HighShelf, Notch, AllPass, Tilt
    };

    // -- Lifecycle -----------------------------------------------------------

    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        freqSmoother_.reset(spec.sampleRate, 30.0f, 0.707f, 1000.0f);
        resSmoother_.reset(spec.sampleRate, 20.0f, 0.707f);
        gainSmoother_.reset(spec.sampleRate, 20.0f, 0.0f);
        reset();
    }

    void reset() noexcept
    {
        for (auto& stage : stages_) stage.reset();
        freqSmoother_.skip();
        resSmoother_.skip();
        gainSmoother_.skip();
    }

    // -- Configuration -------------------------------------------------------

    /**
     * @brief Configures a low-pass filter.
     * @param freq Cutoff frequency in Hz.
     * @param Q    Quality factor (0.707 = Butterworth).
     * @param slopeDb Slope in dB/oct (6, 12, 18, 24, 30, 36, 42, 48).
     */
    void setLowPass(float freq, float Q = 0.707f, int slopeDb = 12)
    {
        shape_ = Shape::LowPass;
        slopeDb_ = slopeDb;
        numStages_ = slopeToStages(slopeDb);
        freqSmoother_.setTargetValue(freq);
        resSmoother_.setTargetValue(Q);
    }

    void setHighPass(float freq, float Q = 0.707f, int slopeDb = 12)
    {
        shape_ = Shape::HighPass;
        slopeDb_ = slopeDb;
        numStages_ = slopeToStages(slopeDb);
        freqSmoother_.setTargetValue(freq);
        resSmoother_.setTargetValue(Q);
    }

    void setBandPass(float freq, float Q = 0.707f)
    {
        shape_ = Shape::BandPass;
        slopeDb_ = 12;
        numStages_ = 1;
        freqSmoother_.setTargetValue(freq);
        resSmoother_.setTargetValue(Q);
    }

    void setPeaking(float freq, float gainDb, float Q = 1.0f)
    {
        shape_ = Shape::Peak;
        slopeDb_ = 12;
        numStages_ = 1;
        freqSmoother_.setTargetValue(freq);
        resSmoother_.setTargetValue(Q);
        gainSmoother_.setTargetValue(gainDb);
    }

    void setLowShelf(float freq, float gainDb, float slope = 1.0f)
    {
        shape_ = Shape::LowShelf;
        slopeDb_ = 12;
        numStages_ = 1;
        freqSmoother_.setTargetValue(freq);
        gainSmoother_.setTargetValue(gainDb);
        shelfSlope_ = slope;
    }

    void setHighShelf(float freq, float gainDb, float slope = 1.0f)
    {
        shape_ = Shape::HighShelf;
        slopeDb_ = 12;
        numStages_ = 1;
        freqSmoother_.setTargetValue(freq);
        gainSmoother_.setTargetValue(gainDb);
        shelfSlope_ = slope;
    }

    void setNotch(float freq, float Q = 10.0f)
    {
        shape_ = Shape::Notch;
        slopeDb_ = 12;
        numStages_ = 1;
        freqSmoother_.setTargetValue(freq);
        resSmoother_.setTargetValue(Q);
    }

    void setAllPass(float freq, float Q = 0.707f)
    {
        shape_ = Shape::AllPass;
        slopeDb_ = 12;
        numStages_ = 1;
        freqSmoother_.setTargetValue(freq);
        resSmoother_.setTargetValue(Q);
    }

    /**
     * @brief Configures a tilt filter (boost highs + cut lows or vice versa).
     * @param centerFreq Center frequency.
     * @param gainDb Positive = bright, negative = dark.
     */
    void setTilt(float centerFreq, float gainDb)
    {
        shape_ = Shape::Tilt;
        slopeDb_ = 12;
        numStages_ = 1;
        freqSmoother_.setTargetValue(centerFreq);
        gainSmoother_.setTargetValue(gainDb);
    }

    // -- Real-time parameter changes -----------------------------------------

    void setFrequency(float freq) { freqSmoother_.setTargetValue(freq); }
    void setResonance(float Q)    { resSmoother_.setTargetValue(Q); }
    void setGain(float dB)        { gainSmoother_.setTargetValue(dB); }

    /**
     * @brief Sets the nonlinearity amount (Capacitor2-style).
     *
     * When > 0, the effective cutoff frequency modulates based on input
     * amplitude. Loud signals shift the cutoff, producing signal-dependent
     * filtering similar to analog capacitor nonlinearity.
     *
     * @param amount 0 = linear (default), 1 = full nonlinearity.
     */
    void setNonlinearity(T amount) noexcept
    {
        nonlinearity_.store(static_cast<float>(std::clamp(amount, T(0), T(1))), std::memory_order_relaxed);
    }

    // -- Analog drift --------------------------------------------------------

    void enableAnalogDrift(AnalogRandom::AnalogComponent component, float intensity = 0.5f)
    {
        driftEnabled_ = true;
        driftIntensity_ = intensity;
        driftGen_.setAnalogDefault(component);
        driftGen_.prepare(spec_.sampleRate);
    }

    void disableAnalogDrift() { driftEnabled_ = false; }

    // -- Processing ----------------------------------------------------------

    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), MaxChannels);
        const int nS  = buffer.getNumSamples();

        T nonLin = static_cast<T>(nonlinearity_.load(std::memory_order_relaxed));
        const bool needSmoothing = freqSmoother_.isSmoothing() || resSmoother_.isSmoothing()
                                   || gainSmoother_.isSmoothing() || driftEnabled_;
        // Nonlinearity modulates cutoff per-sample (signal-dependent) — cannot rate-limit.
        // Drift and smoother-only updates are safe to rate-limit.
        const bool perSampleCoeffs = nonLin > T(0);
        constexpr int kCoeffUpdateInterval = 32;

        for (int i = 0; i < nS; ++i)
        {
            // Advance smoothers every sample to maintain correct timing
            float freq = freqSmoother_.getNextValue();
            float res  = resSmoother_.getNextValue();
            float gain = gainSmoother_.getNextValue();

            // Advance drift generator every sample to keep LFO phase consistent
            float driftValue = 0.0f;
            if (driftEnabled_)
                driftValue = driftGen_.getNextSample() * driftIntensity_;

            if (perSampleCoeffs)
            {
                // -- Per-sample path: nonlinearity active (signal-dependent) --
                freq *= (1.0f + driftValue);

                // Capacitor2-style nonlinearity: modulate cutoff by signal amplitude
                T avgAbs = T(0);
                for (int ch = 0; ch < nCh; ++ch)
                    avgAbs += std::abs(buffer.getChannel(ch)[i]);
                avgAbs /= static_cast<T>(nCh);

                T dielectric = std::abs(T(2) - (avgAbs + nonLin) / nonLin);
                freq *= static_cast<float>(dielectric);

                float nyquist = static_cast<float>(spec_.sampleRate) * 0.499f;
                freq = std::clamp(freq, 10.0f, nyquist);
                res  = std::max(res, 0.1f);

                updateCoefficients(freq, res, gain);
            }
            else if (needSmoothing && (i % kCoeffUpdateInterval == 0))
            {
                // -- Rate-limited path: smoothing/drift active, no nonlinearity --
                freq *= (1.0f + driftValue);

                float nyquist = static_cast<float>(spec_.sampleRate) * 0.499f;
                freq = std::clamp(freq, 10.0f, nyquist);
                res  = std::max(res, 0.1f);

                updateCoefficients(freq, res, gain);
            }
            else if (!needSmoothing && i == 0)
            {
                // -- Static path: no smoothing, no drift, no nonlinearity --
                // Update once at the start of the block.
                float nyquist = static_cast<float>(spec_.sampleRate) * 0.499f;
                freq = std::clamp(freq, 10.0f, nyquist);
                res  = std::max(res, 0.1f);

                updateCoefficients(freq, res, gain);
            }

            for (int ch = 0; ch < nCh; ++ch)
            {
                T sample = buffer.getChannel(ch)[i];
                for (int s = 0; s < numStages_; ++s)
                    sample = stages_[s].processSample(sample, ch);
                buffer.getChannel(ch)[i] = sample;
            }
        }
    }

    /// @note processSample() applies the filter with current coefficients but does NOT
    /// advance parameter smoothers or recompute coefficients. For real-time parameter
    /// changes to take effect per-sample, use processBlock() which handles smoothing.
    /// This method is a lightweight hot path for use inside other processors that
    /// manage their own parameter updates externally.
    T processSample(T input, int channel) noexcept
    {
        T sample = input;
        for (int s = 0; s < numStages_; ++s)
            sample = stages_[s].processSample(sample, channel);
        return sample;
    }

protected:
    // Max biquad stages: order 8 = 4 second-order, or order 7 = 1 first-order + 3 second-order = 4 biquads
    static constexpr int kMaxStages = 4;
    static constexpr int kMaxOrder = 8; // 48 dB/oct max

    /**
     * @brief Butterworth cascade descriptor for a given filter order.
     *
     * For proper Butterworth response, each cascaded biquad stage needs a
     * specific Q value derived from the Butterworth polynomial poles.
     * Odd orders include a first-order (6 dB/oct) stage plus second-order sections.
     */
    struct ButterworthCascade
    {
        int order = 0;
        bool hasFirstOrder = false;   ///< True if order is odd (includes a 1st-order stage).
        int numSecondOrder = 0;       ///< Number of 2nd-order biquad sections.
        float qValues[kMaxStages] {}; ///< Q for each 2nd-order section.
    };

    /**
     * @brief Computes the Butterworth cascade parameters for a given slope.
     *
     * Uses a lookup table of exact Q values from the Butterworth polynomial
     * (source: Zolzer "DAFX", standard Butterworth filter tables).
     * Order = slopeDb/6 (not slopeDb/12), since each first-order section is 6 dB/oct.
     *
     * @param slopeDb Filter slope in dB/oct (6, 12, 18, 24, 30, 36, 42, 48).
     */
    static ButterworthCascade computeCascade(int slopeDb) noexcept
    {
        // Butterworth Q values for cascaded second-order sections, indexed by filter order.
        // For odd orders, the first-order (real pole) stage is handled separately.
        static constexpr float qTable[kMaxOrder + 1][kMaxStages] = {
            {},                                          // order 0 (unused)
            {},                                          // order 1 (first-order only, no 2nd-order stages)
            { 0.7071f },                                 // order 2
            { 1.0f },                                    // order 3 (+ first-order)
            { 0.5412f, 1.3066f },                        // order 4
            { 0.6180f, 1.6180f },                        // order 5 (+ first-order)
            { 0.5176f, 0.7071f, 1.9319f },               // order 6
            { 0.5549f, 0.8019f, 2.2470f },               // order 7 (+ first-order)
            { 0.5098f, 0.6013f, 0.9000f, 2.5628f }      // order 8
        };

        ButterworthCascade result {};
        result.order = std::clamp(slopeDb / 6, 1, kMaxOrder);
        result.hasFirstOrder = (result.order % 2 != 0);
        result.numSecondOrder = result.order / 2;
        for (int i = 0; i < result.numSecondOrder; ++i)
            result.qValues[i] = qTable[result.order][i];
        return result;
    }

    /**
     * @brief Computes the total number of biquad stages for a given slope.
     *
     * Even order: order/2 second-order stages.
     * Odd order: 1 first-order stage (as biquad with b2=a2=0) + (order-1)/2 second-order stages.
     * Total = (order + 1) / 2 using integer division.
     */
    static int slopeToStages(int slopeDb) noexcept
    {
        int order = std::clamp(slopeDb / 6, 1, kMaxOrder);
        return (order + 1) / 2;
    }

    void updateCoefficients(float freq, float Q, float gainDb) noexcept
    {
        double sr = spec_.sampleRate;
        double f  = static_cast<double>(freq);

        auto cascade = computeCascade(slopeDb_);
        int stageIdx = 0;

        // First-order stage for odd-order LP/HP Butterworth cascades
        if (cascade.hasFirstOrder)
        {
            BiquadCoeffs<T> c;
            switch (shape_)
            {
                case Shape::LowPass:   c = BiquadCoeffs<T>::makeFirstOrderLowPass(sr, f); break;
                case Shape::HighPass:   c = BiquadCoeffs<T>::makeFirstOrderHighPass(sr, f); break;
                // Non-LP/HP shapes don't use cascaded slopes — fall through to LP as safe default
                default:               c = BiquadCoeffs<T>::makeFirstOrderLowPass(sr, f); break;
            }
            stages_[stageIdx++].setCoeffs(c);
        }

        // Second-order stages with per-stage Butterworth Q values
        for (int s = 0; s < cascade.numSecondOrder; ++s)
        {
            float stageQ = cascade.qValues[s];

            // For resonant/user-Q shapes, use the user's Q instead of Butterworth Q.
            // Butterworth Q cascade only applies to LP/HP slope cascading.
            if (shape_ == Shape::Peak || shape_ == Shape::BandPass ||
                shape_ == Shape::Notch || shape_ == Shape::AllPass)
                stageQ = Q;

            BiquadCoeffs<T> c;
            switch (shape_)
            {
                case Shape::LowPass:   c = BiquadCoeffs<T>::makeLowPass(sr, f, stageQ);  break;
                case Shape::HighPass:   c = BiquadCoeffs<T>::makeHighPass(sr, f, stageQ); break;
                case Shape::BandPass:   c = BiquadCoeffs<T>::makeBandPass(sr, f, stageQ); break;
                case Shape::Peak:       c = BiquadCoeffs<T>::makePeak(sr, f, stageQ, gainDb); break;
                case Shape::LowShelf:   c = BiquadCoeffs<T>::makeLowShelf(sr, f, gainDb, shelfSlope_); break;
                case Shape::HighShelf:  c = BiquadCoeffs<T>::makeHighShelf(sr, f, gainDb, shelfSlope_); break;
                case Shape::Notch:      c = BiquadCoeffs<T>::makeNotch(sr, f, stageQ);  break;
                case Shape::AllPass:    c = BiquadCoeffs<T>::makeAllPass(sr, f, stageQ); break;
                case Shape::Tilt:       c = BiquadCoeffs<T>::makeTilt(sr, f, gainDb); break;
            }
            stages_[stageIdx++].setCoeffs(c);
        }

        numStages_ = stageIdx;
    }

    AudioSpec spec_ {};
    Shape shape_ = Shape::LowPass;
    int numStages_ = 1;
    int slopeDb_ = 12;         ///< Raw slope in dB/oct, needed by updateCoefficients for cascade computation.
    float shelfSlope_ = 1.0f;

    std::array<Biquad<T, MaxChannels>, kMaxStages> stages_ {};

    Smoothers::StateVariableSmoother freqSmoother_;
    Smoothers::LinearSmoother resSmoother_, gainSmoother_;

    std::atomic<float> nonlinearity_ { 0.0f };

    bool driftEnabled_ = false;
    float driftIntensity_ = 0.0f;
    AnalogRandom::Generator<float> driftGen_;
};

} // namespace dspark
