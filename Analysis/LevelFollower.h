// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file LevelFollower.h
 * @brief Envelope follower for real-time peak and RMS level metering.
 *
 * Tracks audio levels with configurable attack and release times using
 * one-pole smoothing. Suitable for metering, dynamics processing sidechain,
 * and visualisation.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, DspMath.h.
 *
 * @code
 *   dspark::LevelFollower<float> meter;
 *   meter.prepare(spec);
 *   meter.setAttackMs(1.0f);
 *   meter.setReleaseMs(100.0f);
 *
 *   // In process():
 *   meter.process(buffer.toView());
 *   float peakL = meter.getPeakLevel(0);
 *   float rmsL  = meter.getRmsLevel(0);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"

#include <array>
#include <cmath>

namespace dspark {

/**
 * @class LevelFollower
 * @brief Per-channel peak and RMS envelope follower.
 *
 * Uses one-pole smoothing with asymmetric attack/release for both peak and
 * RMS tracking. All state is pre-allocated — no runtime allocations.
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum number of channels supported.
 */
template <typename T, int MaxChannels = 16>
class LevelFollower
{
public:
    /**
     * @brief Prepares the follower for the given audio environment.
     * @param spec Audio specification (sample rate, block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        sampleRate_  = spec.sampleRate;
        numChannels_ = spec.numChannels;
        updateCoefficients();
        reset();
    }

    /**
     * @brief Sets the attack time (how fast the envelope rises).
     * @param ms Attack time in milliseconds.
     */
    void setAttackMs(float ms) noexcept
    {
        attackMs_ = ms;
        updateCoefficients();
    }

    /**
     * @brief Sets the release time (how fast the envelope falls).
     * @param ms Release time in milliseconds.
     */
    void setReleaseMs(float ms) noexcept
    {
        releaseMs_ = ms;
        updateCoefficients();
    }

    /** @brief Resets all envelope states to zero. */
    void reset() noexcept
    {
        for (auto& s : state_)
        {
            s.peak     = T(0);
            s.rmsAccum = T(0);
        }
    }

    /**
     * @brief Processes a block of audio and updates level tracking.
     * @param buffer Read-only audio buffer view.
     */
    void process(AudioBufferView<const T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS  = buffer.getNumSamples();

        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* data = buffer.getChannel(ch);
            auto& s = state_[ch];

            for (int i = 0; i < nS; ++i)
            {
                T absSample = std::abs(data[i]);

                // Peak follower (asymmetric one-pole)
                T peakCoeff = (absSample > s.peak) ? attackCoeff_ : releaseCoeff_;
                s.peak = absSample + peakCoeff * (s.peak - absSample);

                // RMS follower (squared domain, asymmetric one-pole)
                T squared   = data[i] * data[i];
                T rmsCoeff  = (squared > s.rmsAccum) ? attackCoeff_ : releaseCoeff_;
                s.rmsAccum  = squared + rmsCoeff * (s.rmsAccum - squared);
            }
        }
    }

    /** @brief Overload accepting a mutable view. */
    void process(AudioBufferView<T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS  = buffer.getNumSamples();

        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* data = buffer.getChannel(ch);
            auto& s = state_[ch];

            for (int i = 0; i < nS; ++i)
            {
                T absSample = std::abs(data[i]);

                T peakCoeff = (absSample > s.peak) ? attackCoeff_ : releaseCoeff_;
                s.peak = absSample + peakCoeff * (s.peak - absSample);

                T squared   = data[i] * data[i];
                T rmsCoeff  = (squared > s.rmsAccum) ? attackCoeff_ : releaseCoeff_;
                s.rmsAccum  = squared + rmsCoeff * (s.rmsAccum - squared);
            }
        }
    }

    /**
     * @brief Returns the current peak level for the given channel.
     * @param channel Channel index (0-based).
     * @return Peak level (linear, >= 0).
     */
    [[nodiscard]] T getPeakLevel(int channel) const noexcept
    {
        if (channel < 0 || channel >= MaxChannels) return T(0);
        return state_[channel].peak;
    }

    /**
     * @brief Returns the current RMS level for the given channel.
     * @param channel Channel index (0-based).
     * @return RMS level (linear, >= 0).
     */
    [[nodiscard]] T getRmsLevel(int channel) const noexcept
    {
        if (channel < 0 || channel >= MaxChannels) return T(0);
        return std::sqrt(std::max(state_[channel].rmsAccum, T(0)));
    }

    /**
     * @brief Returns the current peak level in decibels for the given channel.
     * @param channel Channel index (0-based).
     * @return Peak level in dB.
     */
    [[nodiscard]] T getPeakLevelDb(int channel) const noexcept
    {
        if (channel < 0 || channel >= MaxChannels) return T(-100);
        return gainToDecibels(state_[channel].peak);
    }

    /**
     * @brief Returns the current RMS level in decibels for the given channel.
     * @param channel Channel index (0-based).
     * @return RMS level in dB.
     */
    [[nodiscard]] T getRmsLevelDb(int channel) const noexcept
    {
        if (channel < 0 || channel >= MaxChannels) return T(-100);
        return gainToDecibels(getRmsLevel(channel));
    }

private:
    void updateCoefficients() noexcept
    {
        if (sampleRate_ <= 0.0) return;
        auto fs = static_cast<T>(sampleRate_);
        attackCoeff_  = (attackMs_  > T(0)) ? std::exp(T(-1) / (fs * attackMs_  / T(1000))) : T(0);
        releaseCoeff_ = (releaseMs_ > T(0)) ? std::exp(T(-1) / (fs * releaseMs_ / T(1000))) : T(0);
    }

    struct ChannelState
    {
        T peak     = T(0);
        T rmsAccum = T(0);
    };

    double sampleRate_  = 44100.0;
    int    numChannels_ = 0;
    T      attackMs_    = T(1);
    T      releaseMs_   = T(100);
    T      attackCoeff_ = T(0);
    T      releaseCoeff_= T(0);

    std::array<ChannelState, MaxChannels> state_ {};
};

} // namespace dspark
