// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file LadderFilter.h
 * @brief Moog-style 4-pole resonant ladder filter with TPT discretization.
 *
 * Classic analog-modelled filter using the Topology-Preserving Transform
 * (Zavalishin) for accurate analog behaviour. Features self-oscillation
 * at high resonance and optional nonlinear drive in the feedback path.
 *
 * Produces the warm, fat sound characteristic of Moog synthesizers.
 * Essential for subtractive synthesis and creative filtering.
 *
 * Multiple output modes from the same structure:
 * - **LP6/12/18/24**: 1/2/3/4-pole lowpass
 * - **BP12**: Bandpass (derived from stage outputs)
 * - **HP24**: Highpass (derived from stage outputs)
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h.
 *
 * @code
 *   dspark::LadderFilter<float> ladder;
 *   ladder.prepare(spec);
 *   ladder.setCutoff(1000.0f);    // 1 kHz cutoff
 *   ladder.setResonance(0.7f);    // Near self-oscillation
 *   ladder.setDrive(2.0f);        // Warm overdrive
 *   ladder.processBlock(buffer);
 * @endcode
 */

#include "DspMath.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "DenormalGuard.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace dspark {

/**
 * @class LadderFilter
 * @brief 4-pole resonant ladder filter with analog-modelled nonlinearity.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class LadderFilter
{
public:
    /** @brief Output filter mode. */
    enum class Mode
    {
        LP6,    ///< 1-pole lowpass (6 dB/oct).
        LP12,   ///< 2-pole lowpass (12 dB/oct).
        LP18,   ///< 3-pole lowpass (18 dB/oct).
        LP24,   ///< 4-pole lowpass (24 dB/oct) — classic Moog.
        BP12,   ///< Bandpass (12 dB/oct).
        HP24    ///< 4-pole highpass (24 dB/oct).
    };

    ~LadderFilter() = default;

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the filter.
     * @param spec Audio environment.
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        updateCoefficients();
        reset();
    }

    /**
     * @brief Processes an audio buffer in-place.
     * @param buffer Audio data.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();

        for (int i = 0; i < nS; ++i)
        {
            for (int ch = 0; ch < nCh; ++ch)
                buffer.getChannel(ch)[i] = processSample(buffer.getChannel(ch)[i], ch);
        }
    }

    /**
     * @brief Processes a single sample.
     * @param input Input sample.
     * @param channel Channel index.
     * @return Filtered output.
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        auto& s = state_[channel];

        // TPT integrator gain
        T G = g_ / (T(1) + g_);
        T G2 = G * G;
        T G3 = G2 * G;
        T G4 = G3 * G;

        // Estimate LP24 output from integrator states (zero-delay feedback).
        // Each state passes through the remaining integrator stages:
        //   Sest = G^3*(1-G)*z[0] + G^2*(1-G)*z[1] + G*(1-G)*z[2] + (1-G)*z[3]
        // Factor out (1-G) = 1/(1+g):
        T ig = T(1) / (T(1) + g_);
        T Sest = G3 * ig * s.z[0]
               + G2 * ig * s.z[1]
               + G  * ig * s.z[2]
               +      ig * s.z[3];

        // Resonance feedback coefficient
        T k = resonance_ * T(4);

        // Apply drive to estimated feedback (nonlinear saturation)
        T fbSignal = Sest;
        if (drive_ > T(1))
            fbSignal = fastTanh(fbSignal * drive_) / drive_;

        // Solve zero-delay feedback: u = (input - k * fb) / (1 + k * G^4)
        T u = (input - k * fbSignal) / (T(1) + k * G4);

        // Process through 4 TPT integrator stages
        T x = u;
        for (int i = 0; i < 4; ++i)
        {
            T v = (x - s.z[i]) * g_ / (T(1) + g_);
            T y = v + s.z[i];
            s.z[i] = y + v;
            s.stage[i] = y;
            x = y;
        }

        // Output mode selection
        return selectOutput(input, s);
    }

    /** @brief Resets internal state. */
    void reset() noexcept
    {
        for (auto& s : state_)
        {
            for (auto& z : s.z) z = T(0);
            for (auto& st : s.stage) st = T(0);
        }
    }

    // -- Parameters -------------------------------------------------------------

    /**
     * @brief Sets the cutoff frequency.
     * @param hz Cutoff in Hz (20 to Nyquist).
     */
    void setCutoff(T hz) noexcept
    {
        cutoff_ = std::clamp(hz, T(20), static_cast<T>(spec_.sampleRate) * T(0.499));
        updateCoefficients();
    }

    /**
     * @brief Sets resonance amount.
     * @param amount 0.0 = no resonance, 1.0 = self-oscillation.
     */
    void setResonance(T amount) noexcept
    {
        resonance_ = std::clamp(amount, T(0), T(1));
    }

    /**
     * @brief Sets the nonlinear drive amount.
     *
     * Values > 1 add analog-style saturation in the feedback path.
     * This prevents runaway self-oscillation and adds warmth.
     *
     * @param amount Drive (1.0 = clean, 2-5 = warm, >5 = aggressive).
     */
    void setDrive(T amount) noexcept { drive_ = std::max(amount, T(0.1)); }

    /**
     * @brief Sets the output mode.
     * @param mode Filter output type.
     */
    void setMode(Mode mode) noexcept { mode_ = mode; }

    /** @brief Returns the current cutoff. */
    [[nodiscard]] T getCutoff() const noexcept { return cutoff_; }
    /** @brief Returns the current resonance. */
    [[nodiscard]] T getResonance() const noexcept { return resonance_; }
    /** @brief Returns the current mode. */
    [[nodiscard]] Mode getMode() const noexcept { return mode_; }

protected:
    static constexpr int kMaxChannels = 16;

    struct ChannelState
    {
        std::array<T, 4> z {};      // Integrator states
        std::array<T, 4> stage {};  // Stage outputs (for mode selection)
    };

    std::array<ChannelState, kMaxChannels> state_ {};
    AudioSpec spec_ {};
    T cutoff_ = T(1000);
    T resonance_ = T(0);
    T drive_ = T(1);
    T g_ = T(0);  // TPT coefficient
    Mode mode_ = Mode::LP24;

private:
    void updateCoefficients() noexcept
    {
        if (spec_.sampleRate > 0)
            g_ = static_cast<T>(std::tan(pi<double> * static_cast<double>(cutoff_)
                                         / spec_.sampleRate));
    }

    [[nodiscard]] T selectOutput(T input, const ChannelState& s) const noexcept
    {
        switch (mode_)
        {
            case Mode::LP6:  return s.stage[0];
            case Mode::LP12: return s.stage[1];
            case Mode::LP18: return s.stage[2];
            case Mode::LP24: return s.stage[3];
            case Mode::BP12: return s.stage[0] - s.stage[2];
            case Mode::HP24:
            {
                // HP = input - 4*s1 + 6*s2 - 4*s3 + s4
                return input - T(4)*s.stage[0] + T(6)*s.stage[1]
                       - T(4)*s.stage[2] + s.stage[3];
            }
        }
        return s.stage[3]; // Default LP24
    }
};

} // namespace dspark
