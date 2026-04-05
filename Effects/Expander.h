// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Expander.h
 * @brief Downward expander with configurable ratio, hysteresis, and sidechain.
 *
 * A generalization of the noise gate: instead of fully closing (infinite ratio),
 * the expander applies a configurable ratio below the threshold. At high ratios
 * (>20:1) it behaves like a gate; at low ratios (1.5:1–4:1) it provides gentle
 * dynamic range expansion.
 *
 * Uses the same state machine as NoiseGate (Open/Hold/Closed) with hysteresis
 * to prevent chattering. Supports internal and external sidechain with optional
 * high-pass filter.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h.
 *
 * @code
 *   dspark::Expander<float> exp;
 *   exp.prepare(spec);
 *   exp.setThreshold(-30.0f);
 *   exp.setRatio(4.0f);        // 4:1 expansion below threshold
 *   exp.setAttack(0.5f);
 *   exp.setRelease(100.0f);
 *   exp.processBlock(buffer);
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DenormalGuard.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class Expander
 * @brief Downward expander with ratio control, hysteresis, and sidechain.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Expander
{
public:
    virtual ~Expander() = default;

    enum class State { Closed, Open, Hold };

    // -- Lifecycle -----------------------------------------------------------

    void prepare(double sampleRate) noexcept
    {
        sampleRate_ = sampleRate;
        updateCoefficients();
        reset();
    }

    void prepare(const AudioSpec& spec)
    {
        prepare(spec.sampleRate);
    }

    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        const int nCh = buffer.getNumChannels();
        const int nS  = buffer.getNumSamples();

        cacheParams();

        if (nCh >= 2)
        {
            for (int i = 0; i < nS; ++i)
            {
                T left  = buffer.getChannel(0)[i];
                T right = buffer.getChannel(1)[i];
                T sc = std::max(std::abs(left), std::abs(right));
                if (cachedScHpfEnabled_) sc = applySidechainHPF(sc);

                T levelDb = gainToDecibels(sc);
                updateStateMachine(levelDb);
                T gain = computeGain(levelDb);

                for (int ch = 0; ch < nCh; ++ch)
                    buffer.getChannel(ch)[i] *= gain;
            }
        }
        else if (nCh == 1)
        {
            for (int i = 0; i < nS; ++i)
            {
                T input = buffer.getChannel(0)[i];
                T sc = std::abs(input);
                if (cachedScHpfEnabled_)
                    sc = std::abs(applySidechainHPF(input));
                T levelDb = gainToDecibels(sc);
                updateStateMachine(levelDb);
                buffer.getChannel(0)[i] = input * computeGain(levelDb);
            }
        }
    }

    /**
     * @brief Processes audio with external sidechain.
     */
    void processBlock(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {
        DenormalGuard guard;
        const int nCh  = audio.getNumChannels();
        const int nS   = audio.getNumSamples();
        const int scCh = sidechain.getNumChannels();

        cacheParams();

        for (int i = 0; i < nS; ++i)
        {
            T scMax = T(0);
            for (int c = 0; c < scCh; ++c)
            {
                T a = std::abs(sidechain.getChannel(c)[i]);
                if (a > scMax) scMax = a;
            }

            T levelDb = gainToDecibels(scMax);
            updateStateMachine(levelDb);
            T gain = computeGain(levelDb);

            for (int ch = 0; ch < nCh; ++ch)
                audio.getChannel(ch)[i] *= gain;
        }
    }

    [[nodiscard]] T processSample(T input) noexcept
    {
        return processSampleInternal(input, input);
    }

    [[nodiscard]] T processSampleWithSidechain(T input, T sidechain) noexcept
    {
        return processSampleInternal(input, sidechain);
    }

    // -- Parameters ----------------------------------------------------------

    void setThreshold(T dB) noexcept { threshold_.store(dB, std::memory_order_relaxed); }
    void setRatio(T ratio) noexcept { ratio_.store(std::max(ratio, T(1)), std::memory_order_relaxed); }
    void setHysteresis(T dB) noexcept { hysteresis_.store(std::max(dB, T(0)), std::memory_order_relaxed); }
    void setRange(T dB) noexcept
    {
        rangeDb_ = std::min(dB, T(0));
        rangeLinear_.store(decibelsToGain(rangeDb_), std::memory_order_relaxed);
    }

    void setAttack(T ms) noexcept
    {
        attackMs_.store(std::max(ms, T(0.01)), std::memory_order_relaxed);
        updateCoefficients();
    }

    void setHold(T ms) noexcept
    {
        holdMs_.store(std::max(ms, T(0)), std::memory_order_relaxed);
        holdSamples_.store(static_cast<int>(sampleRate_ * static_cast<double>(holdMs_.load(std::memory_order_relaxed)) / 1000.0),
                           std::memory_order_relaxed);
    }

    void setRelease(T ms) noexcept
    {
        releaseMs_.store(std::max(ms, T(0.01)), std::memory_order_relaxed);
        updateCoefficients();
    }

    void setSidechainHPF(bool enabled, double cutoffHz = 80.0) noexcept
    {
        scHpfEnabled_.store(enabled, std::memory_order_relaxed);
        scHpfCoeff_ = static_cast<T>(
            std::exp(-std::numbers::pi * 2.0 * cutoffHz / sampleRate_));
    }

    // -- Queries -------------------------------------------------------------

    [[nodiscard]] State getState() const noexcept { return state_; }
    [[nodiscard]] T getCurrentGainDb() const noexcept { return gainToDecibels(gateGain_); }

    void reset() noexcept
    {
        state_ = State::Closed;
        gateGain_ = rangeLinear_.load(std::memory_order_relaxed);
        holdCounter_ = 0;
        scHpfState_ = T(0);
        scHpfPrev_ = T(0);
    }

protected:
    void cacheParams() noexcept
    {
        cachedThreshold_    = threshold_.load(std::memory_order_relaxed);
        cachedRatio_        = ratio_.load(std::memory_order_relaxed);
        cachedHysteresis_   = hysteresis_.load(std::memory_order_relaxed);
        cachedRangeLinear_  = rangeLinear_.load(std::memory_order_relaxed);
        cachedAttackCoeff_  = attackCoeff_.load(std::memory_order_relaxed);
        cachedReleaseCoeff_ = releaseCoeff_.load(std::memory_order_relaxed);
        cachedHoldSamples_  = holdSamples_.load(std::memory_order_relaxed);
        cachedScHpfEnabled_ = scHpfEnabled_.load(std::memory_order_relaxed);
    }

    void updateStateMachine(T levelDb) noexcept
    {
        T openThresh  = cachedThreshold_;
        T closeThresh = cachedThreshold_ - cachedHysteresis_;

        switch (state_)
        {
            case State::Closed:
                if (levelDb > openThresh)
                    state_ = State::Open;
                break;
            case State::Open:
                if (levelDb < closeThresh)
                {
                    state_ = State::Hold;
                    holdCounter_ = cachedHoldSamples_;
                }
                break;
            case State::Hold:
                if (levelDb > openThresh)
                    state_ = State::Open;
                else if (--holdCounter_ <= 0)
                    state_ = State::Closed;
                break;
        }
    }

    /**
     * @brief Computes the output gain based on level and expansion ratio.
     *
     * When level is above threshold: gain = 1 (open).
     * When level is below threshold: gain = expansion curve with ratio.
     * Gain is smoothed and clamped to range.
     */
    [[nodiscard]] T computeGain(T levelDb) noexcept
    {
        T targetGain;

        if (state_ == State::Open || state_ == State::Hold)
        {
            targetGain = T(1);
        }
        else
        {
            // Expansion: how far below threshold
            T underDb = cachedThreshold_ - levelDb;
            // Gain reduction = underDb * (1 - 1/ratio)
            T reductionDb = underDb * (T(1) - T(1) / cachedRatio_);
            T expandedGain = decibelsToGain(-reductionDb);
            // Clamp to range floor
            targetGain = std::max(expandedGain, cachedRangeLinear_);
        }

        // Smooth transition
        T coeff = (targetGain > gateGain_) ? cachedAttackCoeff_ : cachedReleaseCoeff_;
        gateGain_ += coeff * (targetGain - gateGain_);

        return gateGain_;
    }

    [[nodiscard]] T processSampleInternal(T input, T sidechain) noexcept
    {
        cacheParams();
        T sc = std::abs(sidechain);
        if (cachedScHpfEnabled_)
            sc = std::abs(applySidechainHPF(sidechain));

        T levelDb = gainToDecibels(sc);
        updateStateMachine(levelDb);
        return input * computeGain(levelDb);
    }

    [[nodiscard]] T applySidechainHPF(T input) noexcept
    {
        T output = input - scHpfPrev_ + scHpfCoeff_ * scHpfState_;
        scHpfPrev_ = input;
        scHpfState_ = output;
        return output;
    }

    void updateCoefficients() noexcept
    {
        if (sampleRate_ <= 0.0) return;
        T fs = static_cast<T>(sampleRate_);
        attackCoeff_.store(T(1) - std::exp(T(-1) / (fs * attackMs_.load(std::memory_order_relaxed) / T(1000))),
                           std::memory_order_relaxed);
        releaseCoeff_.store(T(1) - std::exp(T(-1) / (fs * releaseMs_.load(std::memory_order_relaxed) / T(1000))),
                            std::memory_order_relaxed);
    }

    double sampleRate_ = 48000.0;

    std::atomic<T> threshold_ { T(-40) };
    std::atomic<T> ratio_ { T(4) };
    std::atomic<T> hysteresis_ { T(4) };
    std::atomic<T> attackMs_ { T(0.5) };
    std::atomic<T> holdMs_ { T(50) };
    std::atomic<T> releaseMs_ { T(100) };
    T rangeDb_ = T(-80);
    std::atomic<T> rangeLinear_ { T(0.0001) };

    std::atomic<bool> scHpfEnabled_ { false };
    T scHpfCoeff_ = T(0.995);
    T scHpfState_ = T(0);
    T scHpfPrev_ = T(0);

    std::atomic<T> attackCoeff_ { T(0) };
    std::atomic<T> releaseCoeff_ { T(0) };
    std::atomic<int> holdSamples_ { 0 };

    // Cached per-block
    T cachedThreshold_ = T(-40);
    T cachedRatio_ = T(4);
    T cachedHysteresis_ = T(4);
    T cachedRangeLinear_ = T(0.0001);
    T cachedAttackCoeff_ = T(0);
    T cachedReleaseCoeff_ = T(0);
    int cachedHoldSamples_ = 0;
    bool cachedScHpfEnabled_ = false;

    State state_ = State::Closed;
    T gateGain_ = T(0);
    int holdCounter_ = 0;
};

} // namespace dspark
