// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file FrequencyShifter.h
 * @brief Frequency shifting (not pitch shifting) using Hilbert transform.
 *
 * Shifts all frequency components by a fixed amount in Hz. Unlike pitch
 * shifting, this does NOT preserve harmonic relationships — a 100 Hz
 * fundamental with 200 Hz harmonic shifted by +50 Hz becomes 150 Hz and
 * 250 Hz. This produces the characteristic "barber pole" effect.
 *
 * Implementation: Hilbert transformer produces an analytic signal (I + jQ),
 * which is multiplied by a complex exponential e^(j·2π·f·t) to shift
 * all frequencies. The real part of the result is the shifted signal.
 *
 * Dependencies: Hilbert.h, Phasor.h, DspMath.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::FrequencyShifter<float> shifter;
 *   shifter.prepare(spec);
 *   shifter.setShift(5.0f);   // shift up 5 Hz (barber pole phaser)
 *
 *   // In audio callback:
 *   shifter.processBlock(buffer);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/Hilbert.h"
#include "../Core/Phasor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class FrequencyShifter
 * @brief Constant-Hz frequency shift via Hilbert + complex modulation.
 *
 * Each channel has its own Hilbert transformer (independent allpass state).
 * The carrier oscillator is shared across channels.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class FrequencyShifter
{
public:
    /**
     * @brief Prepares the frequency shifter.
     * @param spec Audio environment specification.
     */
    void prepare(const AudioSpec& spec)
    {
        numChannels_ = spec.numChannels;
        phasor_.prepare(spec.sampleRate);
        phasor_.setFrequency(shift_.load(std::memory_order_relaxed));

        for (int ch = 0; ch < numChannels_ && ch < kMaxChannels; ++ch)
            hilberts_[ch].prepare(spec.sampleRate);
    }

    /**
     * @brief Processes audio in-place (applies frequency shift).
     * @param buffer Audio data.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        int numCh = std::min(buffer.getNumChannels(),
                             std::min(numChannels_, kMaxChannels));
        int numSamples = buffer.getNumSamples();

        constexpr T kTwoPi = static_cast<T>(2.0 * std::numbers::pi);
        T mixVal = mix_.load(std::memory_order_relaxed);

        phasor_.setFrequency(shift_.load(std::memory_order_relaxed));

        for (int i = 0; i < numSamples; ++i)
        {
            T phase = phasor_.advance();
            T cosPhase = std::cos(phase * kTwoPi);
            T sinPhase = std::sin(phase * kTwoPi);

            for (int ch = 0; ch < numCh; ++ch)
            {
                T* data = buffer.getChannel(ch);
                auto h = hilberts_[ch].process(data[i]);

                T shifted = h.real * cosPhase - h.imag * sinPhase;

                data[i] = data[i] + (shifted - data[i]) * mixVal;
            }
        }
    }

    /** @brief Resets internal state. */
    void reset() noexcept
    {
        phasor_.reset();
        for (auto& h : hilberts_)
            h.reset();
    }

    /**
     * @brief Sets the frequency shift amount.
     * @param hz Shift in Hz (-1000 to +1000 typical). Negative = down.
     */
    void setShift(T hz) noexcept
    {
        shift_.store(hz, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the dry/wet mix.
     * @param mix 0.0 = fully dry, 1.0 = fully wet.
     */
    void setMix(T mix) noexcept
    {
        mix_.store(std::clamp(mix, T(0), T(1)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getShift() const noexcept { return shift_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }

private:
    static constexpr int kMaxChannels = 2;

    int numChannels_ = 2;
    std::atomic<T> shift_ { T(0) };
    std::atomic<T> mix_ { T(1) };

    Phasor<T> phasor_;
    Hilbert<T> hilberts_[kMaxChannels]{};
};

} // namespace dspark
