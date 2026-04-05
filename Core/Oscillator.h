// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Oscillator.h
 * @brief Band-limited oscillator for LFO, testing, and synthesis.
 *
 * Provides standard waveforms (sine, saw, square, triangle) with optional
 * PolyBLEP anti-aliasing for saw and square. Suitable both as an audio
 * oscillator and as a low-frequency modulation source.
 *
 * Dependencies: DspMath.h only.
 *
 * @code
 *   dspark::Oscillator<float> lfo;
 *   lfo.prepare(48000.0);
 *   lfo.setFrequency(5.0f);       // 5 Hz LFO
 *   lfo.setWaveform(dspark::Oscillator<float>::Waveform::Sine);
 *
 *   float mod = lfo.getNextSample();  // [-1, 1]
 * @endcode
 */

#include "DspMath.h"
#include "AudioSpec.h"

#include <cmath>

namespace dspark {

/**
 * @class Oscillator
 * @brief Per-sample waveform generator with PolyBLEP anti-aliasing.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class Oscillator
{
public:
    enum class Waveform { Sine, Saw, Square, Triangle };

    /** @brief Prepares the oscillator for the given sample rate. */
    void prepare(double sampleRate) noexcept
    {
        sampleRate_ = sampleRate;
        phaseInc_   = frequency_ / static_cast<T>(sampleRate_);
        updateTriNorm();
    }

    /** @brief Prepares from AudioSpec (unified API). */
    void prepare(const AudioSpec& spec) { prepare(spec.sampleRate); }

    void setFrequency(T freq) noexcept
    {
        frequency_ = freq;
        phaseInc_  = freq / static_cast<T>(sampleRate_);
        updateTriNorm();
    }

    void setWaveform(Waveform w) noexcept { waveform_ = w; }

    /** @brief Sets the phase in [0, 1). */
    void setPhase(T phase) noexcept { phase_ = phase; }

    /** @brief Resets the oscillator phase to zero. */
    void reset() noexcept { phase_ = T(0); }

    /** @brief Returns the next sample and advances the phase. */
    [[nodiscard]] T getNextSample() noexcept
    {
        T out = T(0);

        switch (waveform_)
        {
            case Waveform::Sine:
                out = std::sin(phase_ * twoPi<T>);
                break;

            case Waveform::Saw:
                out = T(2) * phase_ - T(1);
                out -= polyBlep(phase_, phaseInc_);
                break;

            case Waveform::Square:
            {
                T raw = (phase_ < T(0.5)) ? T(1) : T(-1);
                raw += polyBlep(phase_, phaseInc_);
                raw -= polyBlep(std::fmod(phase_ + T(0.5), T(1)), phaseInc_);
                out = raw;
                break;
            }

            case Waveform::Triangle:
            {
                // Integrated PolyBLEP square → anti-aliased triangle
                T raw = (phase_ < T(0.5)) ? T(1) : T(-1);
                raw += polyBlep(phase_, phaseInc_);
                raw -= polyBlep(std::fmod(phase_ + T(0.5), T(1)), phaseInc_);
                // Leaky integrator
                triState_ = phaseInc_ * raw + (T(1) - phaseInc_) * triState_;
                // Normalise using precomputed factor (frequency-independent amplitude)
                out = triState_ * triNorm_;
                break;
            }
        }

        // Advance phase
        phase_ += phaseInc_;
        if (phase_ >= T(1)) phase_ -= T(1);

        return out;
    }

    [[nodiscard]] T getPhase() const noexcept { return phase_; }
    [[nodiscard]] T getFrequency() const noexcept { return frequency_; }

private:
    /**
     * @brief PolyBLEP (Polynomial Band-Limited Step) correction.
     *
     * Reduces aliasing at discontinuities in saw and square waves.
     * Applied near the 0-crossing of the phase ramp.
     */
    static T polyBlep(T phase, T inc) noexcept
    {
        T absInc = std::abs(inc);
        if (absInc < T(1e-10)) return T(0);

        if (phase < absInc)
        {
            T t = phase / absInc;
            return t + t - t * t - T(1);
        }
        else if (phase > T(1) - absInc)
        {
            T t = (phase - T(1)) / absInc;
            return t * t + t + t + T(1);
        }
        return T(0);
    }

    /**
     * @brief Precomputes the triangle normalisation factor.
     *
     * The leaky integrator y[n] = a*x[n] + (1-a)*y[n-1] with a=phaseInc_
     * reaches a peak of (1 - (1-a)^(N/2)) after half a period (N/2 = 0.5/a samples).
     * We invert this to normalise the triangle to [-1, 1] at any frequency.
     */
    void updateTriNorm() noexcept
    {
        if (phaseInc_ > T(0) && phaseInc_ < T(1))
        {
            T leakCoeff = T(1) - phaseInc_;
            T halfPeriodSamples = T(0.5) / phaseInc_;
            T expectedPeak = T(1) - std::pow(leakCoeff, halfPeriodSamples);
            triNorm_ = (expectedPeak > T(0.001)) ? T(1) / expectedPeak : T(4);
        }
        else
        {
            triNorm_ = T(4); // Fallback for edge cases
        }
    }

    double   sampleRate_ = 48000.0;
    T        frequency_  = T(440);
    T        phase_      = T(0);
    T        phaseInc_   = T(0);
    T        triState_   = T(0);
    T        triNorm_    = T(4);
    Waveform waveform_   = Waveform::Sine;
};

} // namespace dspark
