// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file TransientDesigner.h
 * @brief Transient shaping via dual-envelope analysis.
 *
 * Uses two envelope followers (fast peak + slow RMS) to separate transient
 * content from sustained content. The ratio between the fast and slow
 * envelopes indicates the transient intensity, allowing independent control
 * of attack (transient) and sustain (body) characteristics.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h.
 *
 * @code
 *   dspark::TransientDesigner<float> td;
 *   td.prepare(spec);
 *   td.setAttack(50.0f);    // +50% attack emphasis
 *   td.setSustain(-30.0f);  // -30% sustain reduction
 *   td.processBlock(buffer);
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DenormalGuard.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

namespace dspark {

/**
 * @class TransientDesigner
 * @brief Transient shaper with independent attack and sustain controls.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class TransientDesigner
{
public:
    virtual ~TransientDesigner() = default;

    // -- Lifecycle -----------------------------------------------------------

    void prepare(const AudioSpec& spec)
    {
        sampleRate_ = spec.sampleRate;
        updateCoefficients();
        reset();
    }

    void prepare(double sampleRate) noexcept
    {
        sampleRate_ = sampleRate;
        updateCoefficients();
        reset();
    }

    /**
     * @brief Processes an audio buffer in-place.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();

        T attAmt  = attackAmount_.load(std::memory_order_relaxed);
        T susAmt  = sustainAmount_.load(std::memory_order_relaxed);
        bool odr  = outputDepRecovery_.load(std::memory_order_relaxed);

        for (int i = 0; i < nS; ++i)
        {
            for (int ch = 0; ch < nCh; ++ch)
            {
                T sample = buffer.getChannel(ch)[i];
                T absSample = std::abs(sample);

                // Fast envelope (peak follower): detects transients
                T& fast = envFast_[ch];
                T fastCoeff = (absSample > fast) ? fastAttackCoeff_ : fastReleaseCoeff_;
                fast += fastCoeff * (absSample - fast);

                // Slow envelope (RMS-like): tracks sustained level
                T& slow = envSlow_[ch];
                T slowRelCoeff = slowReleaseCoeff_;

                // Output-dependent recovery: faster release when output is loud
                if (odr)
                    slowRelCoeff /= (T(1) + std::abs(lastOutput_[ch]));

                T slowCoeff = (absSample > slow) ? slowAttackCoeff_ : slowRelCoeff;
                slow += slowCoeff * (absSample - slow);

                // Compute gain from envelope ratio
                T gain = computeGain(fast, slow, attAmt, susAmt);
                T output = sample * gain;
                lastOutput_[ch] = output;
                buffer.getChannel(ch)[i] = output;
            }
        }
    }

    // -- Parameters ----------------------------------------------------------

    /**
     * @brief Sets attack (transient) emphasis.
     * @param amount -100 to +100 (%). Positive = boost transients, negative = soften.
     */
    void setAttack(T amount) noexcept
    {
        attackAmount_.store(std::clamp(amount, T(-100), T(100)) / T(100), std::memory_order_relaxed);
    }

    /**
     * @brief Sets sustain (body) emphasis.
     * @param amount -100 to +100 (%). Positive = boost sustain, negative = reduce.
     */
    void setSustain(T amount) noexcept
    {
        sustainAmount_.store(std::clamp(amount, T(-100), T(100)) / T(100), std::memory_order_relaxed);
    }

    /**
     * @brief Enables output-dependent recovery.
     *
     * When enabled, the slow envelope release speed scales inversely with
     * output level: louder output = faster recovery. Prevents pumping
     * artifacts on dynamic material.
     */
    void setOutputDepRecovery(bool enabled) noexcept
    {
        outputDepRecovery_.store(enabled, std::memory_order_relaxed);
    }

    /**
     * @brief Sets character as a single knob (-1 to +1).
     *
     * Maps to attack/sustain: -1 = soften transients + boost sustain,
     * 0 = neutral, +1 = boost transients + reduce sustain.
     */
    void setCharacter(T amount) noexcept
    {
        T c = std::clamp(amount, T(-1), T(1));
        attackAmount_.store(c, std::memory_order_relaxed);
        sustainAmount_.store(-c * T(0.5), std::memory_order_relaxed);
    }

    void reset() noexcept
    {
        envFast_.fill(T(0));
        envSlow_.fill(T(0));
        lastOutput_.fill(T(0));
    }

private:
    static constexpr int kMaxChannels = 16;

    [[nodiscard]] T computeGain(T fast, T slow, T attAmt, T susAmt) const noexcept
    {
        constexpr T eps = T(1e-10);
        T gain = T(1);

        T safeSlowInv = T(1) / std::max(slow, eps);
        T transientRatio = fast * safeSlowInv;

        if (std::abs(attAmt) > T(0.001))
        {
            T clamped = std::clamp(transientRatio, T(0.1), T(10));
            gain *= std::pow(clamped, attAmt);
        }

        if (std::abs(susAmt) > T(0.001))
        {
            T sustainRatio = slow * (T(1) / std::max(fast, eps));
            T clamped = std::clamp(sustainRatio, T(0.1), T(10));
            gain *= std::pow(clamped, susAmt);
        }

        return std::clamp(gain, T(0.01), T(10));
    }

    void updateCoefficients() noexcept
    {
        if (sampleRate_ <= 0.0) return;
        T fs = static_cast<T>(sampleRate_);
        fastAttackCoeff_  = T(1) - std::exp(T(-1) / (fs * T(0.0001)));
        fastReleaseCoeff_ = T(1) - std::exp(T(-1) / (fs * T(0.005)));
        slowAttackCoeff_  = T(1) - std::exp(T(-1) / (fs * T(0.020)));
        slowReleaseCoeff_ = T(1) - std::exp(T(-1) / (fs * T(0.200)));
    }

    double sampleRate_ = 48000.0;

    // Atomic parameters
    std::atomic<T> attackAmount_ { T(0) };
    std::atomic<T> sustainAmount_ { T(0) };
    std::atomic<bool> outputDepRecovery_ { false };

    // Envelope coefficients
    T fastAttackCoeff_ = T(0), fastReleaseCoeff_ = T(0);
    T slowAttackCoeff_ = T(0), slowReleaseCoeff_ = T(0);

    // Per-channel state
    std::array<T, kMaxChannels> envFast_ {};
    std::array<T, kMaxChannels> envSlow_ {};
    std::array<T, kMaxChannels> lastOutput_ {};
};

} // namespace dspark
