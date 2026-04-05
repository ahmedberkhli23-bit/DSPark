// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Crossfade.h
 * @brief Crossfade between two audio signals with selectable curve.
 *
 * Provides three crossfade curves for smooth transitions between audio sources.
 * Useful for preset morphing, scene transitions, A/B comparison, and layer mixing.
 *
 * Curves:
 * - **Linear:** Constant sum (A * (1-t) + B * t). Simple but has a -6 dB dip at centre.
 * - **EqualPower:** Constant power (A * cos(t*pi/2) + B * sin(t*pi/2)). No level dip.
 * - **SCurve:** Smooth S-curve (smoothstep). Starts and ends slowly, fast in the middle.
 *
 * Dependencies: DspMath.h.
 *
 * @code
 *   dspark::Crossfade<float> xfade;
 *   xfade.setCurve(dspark::Crossfade<float>::Curve::EqualPower);
 *   xfade.setPosition(0.5f);  // 50% blend
 *
 *   for (int i = 0; i < numSamples; ++i)
 *       output[i] = xfade.process(inputA[i], inputB[i]);
 * @endcode
 */

#include "../Core/DspMath.h"

#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class Crossfade
 * @brief Crossfades between two signals with configurable curve.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Crossfade
{
public:
    virtual ~Crossfade() = default;
    /** @brief Crossfade curve type. */
    enum class Curve
    {
        Linear,     ///< Linear interpolation (constant sum).
        EqualPower, ///< Equal-power crossfade (constant energy).
        SCurve      ///< Smooth S-curve (smoothstep).
    };

    /**
     * @brief Sets the crossfade curve.
     * @param curve Curve type.
     */
    void setCurve(Curve curve) noexcept { curve_.store(curve, std::memory_order_relaxed); }

    /**
     * @brief Sets the crossfade position.
     *
     * @param position Blend position: 0.0 = 100% A, 1.0 = 100% B.
     */
    void setPosition(T position) noexcept
    {
        position_.store(std::clamp(position, T(0), T(1)), std::memory_order_relaxed);
        updateGains();
    }

    /**
     * @brief Returns the current position.
     */
    [[nodiscard]] T getPosition() const noexcept { return position_.load(std::memory_order_relaxed); }

    /**
     * @brief Crossfades between two samples.
     *
     * @param a First input sample (dry / scene A).
     * @param b Second input sample (wet / scene B).
     * @return Blended output.
     */
    [[nodiscard]] T process(T a, T b) const noexcept
    {
        return a * gainA_ + b * gainB_;
    }

    /**
     * @brief Crossfades two buffers into an output buffer.
     *
     * @param inputA First input buffer.
     * @param inputB Second input buffer.
     * @param output Output buffer.
     * @param numSamples Number of samples.
     */
    void process(const T* inputA, const T* inputB, T* output,
                 int numSamples) const noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            output[i] = inputA[i] * gainA_ + inputB[i] * gainB_;
    }

    /**
     * @brief Crossfades with per-sample position automation.
     *
     * @param inputA First input buffer.
     * @param inputB Second input buffer.
     * @param positions Per-sample position values (0 to 1).
     * @param output Output buffer.
     * @param numSamples Number of samples.
     */
    void processAutomated(const T* inputA, const T* inputB,
                          const T* positions, T* output,
                          int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            position_ = std::clamp(positions[i], T(0), T(1));
            updateGains();
            output[i] = inputA[i] * gainA_ + inputB[i] * gainB_;
        }
    }

    /**
     * @brief Returns the current gain for signal A.
     */
    [[nodiscard]] T getGainA() const noexcept { return gainA_; }

    /**
     * @brief Returns the current gain for signal B.
     */
    [[nodiscard]] T getGainB() const noexcept { return gainB_; }

protected:
    void updateGains() noexcept
    {
        T pos = position_.load(std::memory_order_relaxed);
        switch (curve_.load(std::memory_order_relaxed))
        {
            case Curve::Linear:
                gainA_ = T(1) - pos;
                gainB_ = pos;
                break;

            case Curve::EqualPower:
            {
                constexpr T halfPi = pi<T> / T(2);
                gainA_ = std::cos(pos * halfPi);
                gainB_ = std::sin(pos * halfPi);
                break;
            }

            case Curve::SCurve:
            {
                T t = pos * pos * (T(3) - T(2) * pos);
                gainA_ = T(1) - t;
                gainB_ = t;
                break;
            }
        }
    }

    std::atomic<Curve> curve_ { Curve::EqualPower };
    std::atomic<T> position_ { T(0) };
    T gainA_ = T(1);
    T gainB_ = T(0);
};

} // namespace dspark
