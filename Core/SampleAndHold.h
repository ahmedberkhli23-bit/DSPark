// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file SampleAndHold.h
 * @brief Sample-and-hold processor for stepped modulation and bit-crushing.
 *
 * Captures an input sample and holds it for a configurable number of samples
 * or until an external trigger fires. This is useful for:
 * - **Bit-crushing effects:** Reduce effective sample rate by holding samples.
 * - **Stepped modulation:** Create staircase waveforms from smooth signals.
 * - **Triggered S&H:** Classic synth module triggered by an LFO or clock.
 *
 * Dependencies: DspMath.h (for FloatType concept).
 *
 * @code
 *   // Reduce effective sample rate to 1/4 (downsampling effect)
 *   dspark::SampleAndHold<float> sh;
 *   sh.setHoldSamples(4);
 *
 *   for (int i = 0; i < numSamples; ++i)
 *       output[i] = sh.process(input[i]);
 *
 *   // Trigger-based S&H (classic synth module)
 *   dspark::SampleAndHold<float> sh;
 *   sh.setMode(dspark::SampleAndHold<float>::Mode::Trigger);
 *   for (int i = 0; i < numSamples; ++i)
 *   {
 *       bool trigger = lfoOutput[i] > 0.0f && prevLfo <= 0.0f;  // zero-crossing
 *       output[i] = sh.process(input[i], trigger);
 *       prevLfo = lfoOutput[i];
 *   }
 * @endcode
 */

#include "DspMath.h"

namespace dspark {

/**
 * @class SampleAndHold
 * @brief Holds a sample value for N samples or until triggered.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class SampleAndHold
{
public:
    /** @brief Operating mode. */
    enum class Mode
    {
        Counter, ///< Hold for a fixed number of samples (default).
        Trigger  ///< Hold until an external trigger fires.
    };

    /**
     * @brief Sets the operating mode.
     * @param mode Counter or Trigger mode.
     */
    void setMode(Mode mode) noexcept { mode_ = mode; }

    /**
     * @brief Sets the number of samples to hold in Counter mode.
     *
     * A value of 1 means no hold (pass-through). A value of 4 means the
     * effective sample rate is reduced to 1/4.
     *
     * @param numSamples Hold period in samples (minimum 1).
     */
    void setHoldSamples(int numSamples) noexcept
    {
        holdPeriod_ = numSamples > 0 ? numSamples : 1;
    }

    /**
     * @brief Sets the hold period by effective sample rate.
     *
     * @param targetRate The desired effective sample rate in Hz.
     * @param actualRate The actual sample rate in Hz.
     */
    void setHoldRate(double targetRate, double actualRate) noexcept
    {
        int period = static_cast<int>(actualRate / targetRate);
        setHoldSamples(period);
    }

    /**
     * @brief Processes one sample.
     *
     * In Counter mode, captures a new sample every `holdPeriod` samples.
     * In Trigger mode, captures only when `trigger` is true.
     *
     * @param input The input sample.
     * @param trigger External trigger (only used in Trigger mode).
     * @return The held (output) sample.
     */
    [[nodiscard]] T process(T input, bool trigger = false) noexcept
    {
        if (mode_ == Mode::Counter)
        {
            ++counter_;
            if (counter_ >= holdPeriod_)
            {
                heldValue_ = input;
                counter_ = 0;
            }
        }
        else // Trigger mode
        {
            if (trigger)
                heldValue_ = input;
        }

        return heldValue_;
    }

    /**
     * @brief Processes a block of samples in-place.
     * @param data Audio buffer (modified in-place).
     * @param numSamples Number of samples.
     */
    void processBlock(T* data, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            data[i] = process(data[i]);
    }

    /**
     * @brief Returns the currently held value.
     */
    [[nodiscard]] T getHeldValue() const noexcept { return heldValue_; }

    /**
     * @brief Resets the held value and counter.
     * @param initialValue Value to hold after reset (default: 0).
     */
    void reset(T initialValue = T(0)) noexcept
    {
        heldValue_ = initialValue;
        counter_ = holdPeriod_; // will capture on next sample
    }

private:
    Mode mode_ = Mode::Counter;
    int holdPeriod_ = 1;
    int counter_ = 0;
    T heldValue_ = T(0);
};

} // namespace dspark
