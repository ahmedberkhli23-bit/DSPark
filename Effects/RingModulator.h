// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file RingModulator.h
 * @brief Ring modulation — multiplies the signal by an oscillator carrier.
 *
 * Produces sum and difference frequencies by multiplying the input signal
 * with a sine wave carrier. Classic effect for metallic, bell-like, or
 * robotic tones. Includes a dry/wet mix control.
 *
 * Dependencies: Phasor.h, DspMath.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::RingModulator<float> ring;
 *   ring.prepare(spec);
 *   ring.setFrequency(440.0f);   // carrier at 440 Hz
 *   ring.setMix(1.0f);           // 100% wet
 *
 *   // In audio callback:
 *   ring.processBlock(buffer);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/Phasor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class RingModulator
 * @brief Signal × carrier ring modulation with mix control.
 *
 * Uses a single shared carrier oscillator for all channels (coherent
 * modulation). The carrier is a pure sine — for richer timbres, chain
 * with a WaveshapeTable on the carrier or use multiple instances.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class RingModulator
{
public:
    /** @brief Modulation mode. */
    enum class Mode
    {
        Classic,        ///< Standard multiplication (sum & difference frequencies).
        GeometricMean   ///< Geometric-mean mode: sqrt(|in|*|carrier|) * sign — more musical.
    };

    void prepare(const AudioSpec& spec)
    {
        phasor_.prepare(spec.sampleRate);
        phasor_.setFrequency(frequency_.load(std::memory_order_relaxed));
        numChannels_ = spec.numChannels;
    }

    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        int numCh = std::min(buffer.getNumChannels(), numChannels_);
        int numSamples = buffer.getNumSamples();

        T freq = frequency_.load(std::memory_order_relaxed);
        T mixVal = mix_.load(std::memory_order_relaxed);
        T soarVal = soar_.load(std::memory_order_relaxed);
        auto modeVal = mode_.load(std::memory_order_relaxed);

        phasor_.setFrequency(freq);

        for (int i = 0; i < numSamples; ++i)
        {
            T phase = phasor_.advance();
            T carrier = std::sin(phase * T(2) * static_cast<T>(std::numbers::pi));

            for (int ch = 0; ch < numCh; ++ch)
            {
                T* data = buffer.getChannel(ch);
                T dry = data[i];
                T wet;

                if (modeVal == Mode::GeometricMean)
                {
                    // Geometric-mean ring mod: sqrt(|in| * |carrier|) with 4-quadrant sign
                    T absIn = std::abs(dry);
                    T absCarrier = std::abs(carrier);
                    T gm = std::sqrt(absIn * absCarrier + soarVal);
                    T sign = ((dry >= T(0)) == (carrier >= T(0))) ? T(1) : T(-1);
                    wet = gm * sign;
                }
                else
                {
                    wet = dry * carrier;
                }

                data[i] = dry + (wet - dry) * mixVal;
            }
        }
    }

    void reset() noexcept { phasor_.reset(); }

    /** @brief Sets the carrier frequency. */
    void setFrequency(T hz) noexcept
    {
        frequency_.store(hz, std::memory_order_relaxed);
    }

    /** @brief Sets the dry/wet mix. */
    void setMix(T mix) noexcept
    {
        mix_.store(std::clamp(mix, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the modulation mode.
     *
     * - **Classic**: Standard multiplication.
     * - **GeometricMean**: sqrt(|input|*|carrier|) with 4-quadrant sign.
     *   Produces more musical, less harsh ring modulation.
     */
    void setMode(Mode m) noexcept { mode_.store(m, std::memory_order_relaxed); }

    /**
     * @brief Sets the soar threshold for geometric-mean mode.
     *
     * Added to the product under the sqrt to prevent the output from
     * dropping to zero when either input or carrier crosses zero.
     * Higher values = smoother, more sustained output.
     *
     * @param amount 0 = strict geometric mean, 0.01 = subtle, 0.1 = strong.
     */
    void setSoar(T amount) noexcept
    {
        soar_.store(std::max(amount, T(0)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getFrequency() const noexcept { return frequency_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }
    [[nodiscard]] Mode getMode() const noexcept { return mode_.load(std::memory_order_relaxed); }

private:
    int numChannels_ = 2;
    std::atomic<T> frequency_ { T(440) };
    std::atomic<T> mix_ { T(1) };
    std::atomic<Mode> mode_ { Mode::Classic };
    std::atomic<T> soar_ { T(0) };

    Phasor<T> phasor_;
};

} // namespace dspark
