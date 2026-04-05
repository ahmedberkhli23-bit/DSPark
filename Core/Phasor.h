// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Phasor.h
 * @brief Phase accumulator for oscillators, LFOs, and modulation.
 *
 * A phasor generates a repeating ramp from 0 to 1 at a given frequency.
 * It is the fundamental building block for any periodic signal generator:
 * oscillators, LFOs, sequencers, wavetable readers, etc.
 *
 * Features:
 * - Phase output in [0, 1) range
 * - Hard sync and soft sync inputs
 * - Phase offset and reset
 * - Per-sample frequency modulation
 *
 * Dependencies: DspMath.h (for FloatType concept).
 *
 * @code
 *   dspark::Phasor<float> phasor;
 *   phasor.prepare(48000.0);
 *   phasor.setFrequency(440.0f);
 *
 *   for (int i = 0; i < numSamples; ++i)
 *   {
 *       float phase = phasor.advance();
 *       float sine = std::sin(phase * dspark::twoPi<float>);
 *   }
 * @endcode
 */

#include "DspMath.h"
#include "AudioSpec.h"

#include <cmath>

namespace dspark {

/**
 * @class Phasor
 * @brief Phase accumulator generating a [0, 1) ramp at a given frequency.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Phasor
{
public:
    /**
     * @brief Prepares the phasor for a given sample rate.
     * @param sampleRate Sample rate in Hz.
     */
    void prepare(double sampleRate) noexcept
    {
        sampleRate_ = sampleRate;
        invSampleRate_ = 1.0 / sampleRate;
        updateIncrement();
    }

    /** @brief Prepares from AudioSpec (unified API). */
    void prepare(const AudioSpec& spec) { prepare(spec.sampleRate); }

    /**
     * @brief Sets the oscillation frequency.
     * @param frequencyHz Frequency in Hz (can be negative for reverse).
     */
    void setFrequency(T frequencyHz) noexcept
    {
        frequency_ = frequencyHz;
        updateIncrement();
    }

    /**
     * @brief Returns the current frequency.
     */
    [[nodiscard]] T getFrequency() const noexcept { return frequency_; }

    /**
     * @brief Advances the phase by one sample and returns the current phase.
     *
     * @return Phase value in [0, 1).
     */
    [[nodiscard]] T advance() noexcept
    {
        T current = phase_;
        phase_ += increment_;

        // Wrap to [0, 1) — handles arbitrarily large increments (FM synthesis)
        phase_ -= std::floor(phase_);

        return current;
    }

    /**
     * @brief Advances the phase with per-sample frequency modulation.
     *
     * @param fmHz Additional frequency offset in Hz (e.g., from an FM source).
     * @return Phase value in [0, 1).
     */
    [[nodiscard]] T advanceWithFM(T fmHz) noexcept
    {
        T current = phase_;
        T modulatedIncrement = static_cast<T>(
            static_cast<double>(frequency_ + fmHz) * invSampleRate_);

        phase_ += modulatedIncrement;

        // Wrap to [0, 1) — handles arbitrarily large FM increments
        phase_ -= std::floor(phase_);

        return current;
    }

    /**
     * @brief Returns the current phase without advancing.
     * @return Phase value in [0, 1).
     */
    [[nodiscard]] T getPhase() const noexcept { return phase_; }

    /**
     * @brief Sets the current phase.
     * @param newPhase Phase value (will be wrapped to [0, 1)).
     */
    void setPhase(T newPhase) noexcept
    {
        phase_ = newPhase - std::floor(newPhase);
    }

    /**
     * @brief Hard sync: resets phase to zero.
     *
     * Use this when a master oscillator completes a cycle to synchronise
     * this (slave) oscillator.
     */
    void hardSync() noexcept { phase_ = T(0); }

    /**
     * @brief Soft sync: resets phase only if it is in the second half of its cycle.
     *
     * Produces a less harsh sync effect than hard sync.
     */
    void softSync() noexcept
    {
        if (phase_ >= T(0.5))
            phase_ = T(1) - phase_;
    }

    /**
     * @brief Resets the phasor to initial state.
     * @param startPhase Initial phase (default: 0).
     */
    void reset(T startPhase = T(0)) noexcept
    {
        phase_ = startPhase - std::floor(startPhase);
    }

    /**
     * @brief Returns the phase increment per sample.
     */
    [[nodiscard]] T getIncrement() const noexcept { return increment_; }

private:
    void updateIncrement() noexcept
    {
        increment_ = static_cast<T>(
            static_cast<double>(frequency_) * invSampleRate_);
    }

    double sampleRate_ = 48000.0;
    double invSampleRate_ = 1.0 / 48000.0;

    T frequency_ = T(0);
    T phase_ = T(0);
    T increment_ = T(0);
};

} // namespace dspark
