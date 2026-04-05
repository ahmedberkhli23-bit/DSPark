// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DspMath.h
 * @brief Core mathematical utilities for digital signal processing.
 *
 * Provides constants, unit conversions, and fast approximations commonly needed
 * in audio DSP code. All functions are noexcept and suitable for real-time use.
 *
 * Dependencies: C++20 standard library only.
 *
 * @code
 *   // Convert dB to linear gain
 *   float gain = dspark::decibelsToGain(-6.0f);  // ~0.501
 *
 *   // Fast tanh for waveshaping
 *   float saturated = dspark::fastTanh(input * drive);
 *
 *   // Use typed constants
 *   double phase = dspark::twoPi<double> * freq / sampleRate;
 * @endcode
 */

#include <algorithm>
#include <cmath>
#include <concepts>
#include <numbers>
#include <type_traits>

namespace dspark {

// ============================================================================
// Concepts
// ============================================================================

/** @brief Constrains a type to IEEE floating-point (float or double). */
template <typename T>
concept FloatType = std::floating_point<T>;

// ============================================================================
// Constants
// ============================================================================

/** @brief Pi (3.14159...) for the given floating-point type. */
template <FloatType T> inline constexpr T pi     = std::numbers::pi_v<T>;

/** @brief 2 * Pi (6.28318...). */
template <FloatType T> inline constexpr T twoPi  = T(2) * std::numbers::pi_v<T>;

/** @brief Square root of 2 (1.41421...). */
template <FloatType T> inline constexpr T sqrt2  = std::numbers::sqrt2_v<T>;

/** @brief 1 / square root of 2 (0.70710...). Butterworth Q factor. */
template <FloatType T> inline constexpr T invSqrt2 = T(1) / std::numbers::sqrt2_v<T>;

// ============================================================================
// Decibel / Gain Conversions
// ============================================================================

/**
 * @brief Converts a value in decibels to linear gain.
 *
 * @param dB              Value in decibels.
 * @param minusInfinityDb Values at or below this threshold return 0. Default: -100 dB.
 * @return Linear gain (0 for silence, 1 for unity, >1 for boost).
 *
 * @code
 *   dspark::decibelsToGain(0.0f);   // 1.0
 *   dspark::decibelsToGain(-6.0f);  // ~0.501
 *   dspark::decibelsToGain(-120.0f); // 0.0 (below threshold)
 * @endcode
 */
template <FloatType T>
[[nodiscard]] inline T decibelsToGain(T dB, T minusInfinityDb = T(-100)) noexcept
{
    return dB <= minusInfinityDb ? T(0) : std::pow(T(10), dB / T(20));
}

/**
 * @brief Converts a linear gain value to decibels.
 *
 * @param gain            Linear gain (must be >= 0).
 * @param minusInfinityDb Returned for zero or negative gain. Default: -100 dB.
 * @return Value in decibels.
 *
 * @code
 *   dspark::gainToDecibels(1.0f);    // 0.0
 *   dspark::gainToDecibels(0.5f);    // ~-6.02
 *   dspark::gainToDecibels(0.0f);    // -100.0
 * @endcode
 */
template <FloatType T>
[[nodiscard]] inline T gainToDecibels(T gain, T minusInfinityDb = T(-100)) noexcept
{
    return gain > T(0) ? std::max(minusInfinityDb, T(20) * std::log10(gain))
                       : minusInfinityDb;
}

// ============================================================================
// Interpolation and Mapping
// ============================================================================

/**
 * @brief Maps a value from one range to another (linear interpolation).
 *
 * @param value The input value to remap.
 * @param inMin  Lower bound of the input range.
 * @param inMax  Upper bound of the input range.
 * @param outMin Lower bound of the output range.
 * @param outMax Upper bound of the output range.
 * @return Remapped value. NOT clamped — may exceed output range if input is out of input range.
 *
 * @note For a clamped version, use std::clamp on the result or on the input.
 * @note For simple linear interpolation (t in [0,1]), use std::lerp from <cmath>.
 */
template <FloatType T>
[[nodiscard]] inline T mapRange(T value, T inMin, T inMax, T outMin, T outMax) noexcept
{
    return outMin + (outMax - outMin) * ((value - inMin) / (inMax - inMin));
}

// ============================================================================
// Fast Approximations
// ============================================================================

/**
 * @brief Fast tanh approximation using Pade rational function.
 *
 * Approximately 5x faster than std::tanh with a maximum error < 0.004
 * in the range [-3, 3]. Clamped to +/-1 outside that range.
 * Exact at x = 0.
 *
 * @param x Input value.
 * @return Approximation of tanh(x), in the range [-1, 1].
 *
 * @note For maximum sound quality in saturation, prefer std::tanh.
 *       Use this where performance is critical and the small error is acceptable.
 */
template <FloatType T>
[[nodiscard]] inline T fastTanh(T x) noexcept
{
    if (x < T(-3)) return T(-1);
    if (x > T( 3)) return T( 1);
    const auto x2 = x * x;
    const auto x4 = x2 * x2;
    // Padé [5,4] approximant: max error < 0.05% for |x| <= 3
    return x * (T(945) + T(105) * x2 + x4) / (T(945) + T(420) * x2 + T(15) * x4);
}

/**
 * @brief Fast approximation of 10^x using exp2.
 *
 * Uses the identity 10^x = 2^(x * log2(10)).
 * Useful in dB conversions where exact precision is not critical.
 *
 * @param x Exponent.
 * @return Approximation of 10^x.
 */
template <FloatType T>
[[nodiscard]] inline T fastPow10(T x) noexcept
{
    return std::exp2(x * std::numbers::log2e_v<T> * std::numbers::ln10_v<T>);
}

// ============================================================================
// Utility
// ============================================================================

/**
 * @brief Normalises a phase value to the range [0, 2*pi).
 * @param phase Phase in radians.
 * @return Phase wrapped to [0, 2*pi).
 */
template <FloatType T>
[[nodiscard]] inline T wrapPhase(T phase) noexcept
{
    phase = std::fmod(phase, twoPi<T>);
    if (phase < T(0)) phase += twoPi<T>;
    return phase;
}

} // namespace dspark
