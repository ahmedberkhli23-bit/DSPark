// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file SmoothedValue.h
 * @brief Lightweight per-sample parameter smoother for real-time DSP.
 *
 * A template-based smoother that provides click-free parameter transitions.
 * Designed to be used as a member variable inside processors — one
 * SmoothedValue per parameter that needs smoothing.
 *
 * Uses exponential (one-pole) smoothing by default, which produces
 * perceptually natural transitions for audio parameters like gain,
 * frequency, and resonance.
 *
 * Dependencies: DspMath.h (for FloatType concept).
 *
 * @code
 *   dspark::SmoothedValue<float> gainSmooth;
 *   gainSmooth.prepare(48000.0, 20.0);  // 20 ms ramp time
 *   gainSmooth.setTargetValue(0.5f);
 *
 *   for (int i = 0; i < numSamples; ++i)
 *   {
 *       float gain = gainSmooth.getNextValue();
 *       output[i] = input[i] * gain;
 *   }
 * @endcode
 */

#include "DspMath.h"

#include <cmath>

namespace dspark {

/**
 * @class SmoothedValue
 * @brief One-pole exponential smoother for real-time parameter interpolation.
 *
 * Call `prepare()` once with sample rate and ramp time. Then call
 * `setTargetValue()` whenever the parameter changes, and `getNextValue()`
 * once per sample in the audio callback.
 *
 * @tparam T Value type (float or double).
 */
template <FloatType T>
class SmoothedValue
{
public:
    /**
     * @brief Smoothing algorithm type.
     *
     * - **Exponential** (default): One-pole IIR. Natural for audio parameters
     *   (gain, frequency). Approaches target asymptotically (never truly arrives).
     * - **Linear**: Fixed-step ramp. Reaches target exactly after rampTimeMs.
     *   Good for crossfades, level changes, and anything where overshoot or
     *   zipper noise must be avoided.
     * - **Disabled**: No smoothing. Value jumps instantly to target.
     *   Use when latency from smoothing is unacceptable.
     * - **Chase**: Adaptive-speed smoother inspired by Airwindows. Jumps to
     *   maximum speed on target change, then decays exponentially+linearly.
     *   Combines the fast response of Disabled with the smoothness of
     *   Exponential — zero zipper artifacts, zero mushiness.
     */
    enum class SmoothingType
    {
        Exponential,  ///< One-pole IIR (default). Natural, musical.
        Linear,       ///< Fixed-step ramp. Exact arrival time.
        Disabled,     ///< No smoothing — instant jumps.
        Chase         ///< Adaptive-speed chase. Fast attack, smooth settle.
    };

    /**
     * @brief Configures the smoother for the given sample rate and ramp time.
     *
     * @param sampleRate Sample rate in Hz.
     * @param rampTimeMs Smoothing time in milliseconds. Larger = smoother but slower.
     */
    void prepare(double sampleRate, double rampTimeMs = 20.0) noexcept
    {
        sampleRate_ = sampleRate;
        rampTimeMs_ = rampTimeMs;

        if (sampleRate > 0.0 && rampTimeMs > 0.0)
        {
            // Exponential coefficient
            double tau = rampTimeMs / 1000.0;
            coeff_ = static_cast<T>(std::exp(-1.0 / (sampleRate * tau)));

            // Linear: samples to reach target
            linearSteps_ = static_cast<int>(sampleRate * rampTimeMs / 1000.0);
            if (linearSteps_ < 1) linearSteps_ = 1;
        }
        else
        {
            coeff_ = T(0);
            linearSteps_ = 1;
        }
    }

    /**
     * @brief Sets the smoothing type.
     *
     * Can be changed at any time without resetting state.
     *
     * @param type Smoothing algorithm to use.
     */
    void setSmoothingType(SmoothingType type) noexcept { type_ = type; }

    /** @brief Returns the current smoothing type. */
    [[nodiscard]] SmoothingType getSmoothingType() const noexcept { return type_; }

    /**
     * @brief Sets the target value to smooth towards.
     * @param newTarget The target value.
     */
    void setTargetValue(T newTarget) noexcept
    {
        if (type_ == SmoothingType::Linear && newTarget != target_)
        {
            linearStep_ = (newTarget - current_) / static_cast<T>(linearSteps_);
            linearCounter_ = linearSteps_;
        }
        if (type_ == SmoothingType::Chase && newTarget != target_)
            chaseSpeed_ = T(2500);
        target_ = newTarget;
    }

    /**
     * @brief Returns the next smoothed value and advances the state.
     *
     * Call this once per sample in the audio callback.
     *
     * @return The current smoothed value.
     */
    [[nodiscard]] T getNextValue() noexcept
    {
        switch (type_)
        {
            case SmoothingType::Exponential:
                current_ = target_ + coeff_ * (current_ - target_);
                break;

            case SmoothingType::Linear:
                if (linearCounter_ > 0)
                {
                    current_ += linearStep_;
                    --linearCounter_;
                    if (linearCounter_ == 0)
                        current_ = target_; // Snap to exact target
                }
                break;

            case SmoothingType::Disabled:
                current_ = target_;
                break;

            // Chase mode (Airwindows-inspired): asymptotic approach with adaptive speed.
            // chaseSpeed_ starts at 2500 (set on target change) and decays toward 350.
            // The weighted average: current = (current * speed + target) / (speed + 1)
            // produces exponential-like smoothing where higher speed = slower change.
            // Speed range [350, 2500] maps to ~[0.3%, 0.04%] step size per sample.
            // 0.9999 = multiplicative decay, 0.01 = additive decay per sample.
            // These constants are sample-rate dependent — designed for 44.1-96kHz range.
            case SmoothingType::Chase:
            {
                chaseSpeed_ *= T(0.9999);
                chaseSpeed_ -= T(0.01);
                if (chaseSpeed_ < T(350))  chaseSpeed_ = T(350);
                if (chaseSpeed_ > T(2500)) chaseSpeed_ = T(2500);
                current_ = (current_ * chaseSpeed_ + target_) / (chaseSpeed_ + T(1));
                break;
            }
        }
        return current_;
    }

    /** @brief Returns the current smoothed value without advancing. */
    [[nodiscard]] T getCurrentValue() const noexcept { return current_; }

    /** @brief Returns the target value. */
    [[nodiscard]] T getTargetValue() const noexcept { return target_; }

    /** @brief Returns true if the value is still moving toward the target. */
    [[nodiscard]] bool isSmoothing() const noexcept
    {
        if (type_ == SmoothingType::Disabled) return false;
        if (type_ == SmoothingType::Linear) return linearCounter_ > 0;
        return std::abs(current_ - target_) > T(1e-7); // Exponential & Chase
    }

    /** @brief Instantly jumps to the target value (bypasses smoothing). */
    void skip() noexcept { current_ = target_; linearCounter_ = 0; chaseSpeed_ = T(350); }

    /**
     * @brief Resets both current and target to the given value.
     * @param value The reset value.
     */
    void reset(T value = T(0)) noexcept
    {
        current_ = value;
        target_ = value;
        linearCounter_ = 0;
        chaseSpeed_ = T(350);
    }

    /**
     * @brief Sets the current value and target simultaneously.
     * @param value The value to set.
     */
    void setCurrentAndTarget(T value) noexcept
    {
        current_ = value;
        target_ = value;
        linearCounter_ = 0;
        chaseSpeed_ = T(350);
    }

    /**
     * @brief Changes the smoothing time without resetting state.
     * @param sampleRate Sample rate in Hz.
     * @param rampTimeMs New ramp time in milliseconds.
     */
    void setRampTime(double sampleRate, double rampTimeMs) noexcept
    {
        prepare(sampleRate, rampTimeMs);
    }

private:
    T current_ = T(0);
    T target_ = T(0);
    T coeff_ = T(0);       // Exponential coefficient
    SmoothingType type_ = SmoothingType::Exponential;

    // Linear ramp state
    T linearStep_ = T(0);
    int linearSteps_ = 1;
    int linearCounter_ = 0;

    // Chase state
    T chaseSpeed_ = T(350);

    double sampleRate_ = 0.0;
    double rampTimeMs_ = 20.0;
};

} // namespace dspark
