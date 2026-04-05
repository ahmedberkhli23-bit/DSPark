// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Interpolation.h
 * @brief Sample-accurate interpolation algorithms for audio signals.
 *
 * Provides five interpolation methods as free functions, suitable for use in
 * delay lines, wavetable oscillators, resamplers, and anywhere fractional-sample
 * reads are needed. Each function reads from a buffer at a fractional position.
 *
 * | Method    | Points | Quality       | CPU   | Use case                     |
 * |-----------|--------|---------------|-------|-------------------------------|
 * | Linear    | 2      | Low           | Lowest| Preview, non-critical         |
 * | Cubic     | 4      | Good          | Low   | Delay lines, general-purpose  |
 * | Hermite   | 4      | Good+         | Low   | Smooth curves, wavetables     |
 * | Lagrange  | 4      | High          | Medium| Precision resampling          |
 * | Allpass   | 2      | Frequency-dep | Low   | Fractional delay, phasing     |
 *
 * Dependencies: DspMath.h (for FloatType concept).
 *
 * @code
 *   float buffer[1024] = { ... };
 *   float pos = 512.7f;
 *
 *   float val = dspark::interpolateHermite(buffer, 1024, pos);
 * @endcode
 */

#include "DspMath.h"

#include <cassert>
#include <cmath>

namespace dspark {

/**
 * @brief Linear interpolation between two adjacent samples.
 *
 * @param buffer Sample buffer.
 * @param length Buffer length.
 * @param position Fractional read position (0 <= position < length).
 * @return Interpolated sample value.
 */
template <FloatType T>
[[nodiscard]] inline T interpolateLinear(const T* buffer, int length,
                                         T position) noexcept
{
    int idx0 = static_cast<int>(position);
    T frac = position - static_cast<T>(idx0);

    int idx1 = idx0 + 1;
    if (idx1 >= length) idx1 = 0; // wrap

    return buffer[idx0] + frac * (buffer[idx1] - buffer[idx0]);
}

/**
 * @brief Cubic (Catmull-Rom) interpolation using 4 points.
 *
 * Provides smooth C1-continuous interpolation. The standard choice for
 * delay lines and general audio interpolation.
 *
 * @param buffer Sample buffer.
 * @param length Buffer length.
 * @param position Fractional read position.
 * @return Interpolated sample value.
 */
template <FloatType T>
[[nodiscard]] inline T interpolateCubic(const T* buffer, int length,
                                        T position) noexcept
{
    int idx1 = static_cast<int>(position);
    T frac = position - static_cast<T>(idx1);

    auto wrap = [length](int i) -> int {
        return ((i % length) + length) % length;
    };

    T y0 = buffer[wrap(idx1 - 1)];
    T y1 = buffer[idx1];
    T y2 = buffer[wrap(idx1 + 1)];
    T y3 = buffer[wrap(idx1 + 2)];

    // Catmull-Rom coefficients
    T a0 = y1;
    T a1 = T(0.5) * (y2 - y0);
    T a2 = y0 - T(2.5) * y1 + T(2) * y2 - T(0.5) * y3;
    T a3 = T(0.5) * (y3 - y0) + T(1.5) * (y1 - y2);

    return ((a3 * frac + a2) * frac + a1) * frac + a0;
}

/**
 * @brief 4-point 3rd-order Hermite interpolation.
 *
 * Algebraically identical to Catmull-Rom cubic interpolation (the 4-point
 * Hermite with Catmull-Rom tangents produces the same polynomial). Provided
 * as an alias for API compatibility and naming clarity.
 *
 * @param buffer Sample buffer.
 * @param length Buffer length.
 * @param position Fractional read position.
 * @return Interpolated sample value.
 */
template <FloatType T>
[[nodiscard]] inline T interpolateHermite(const T* buffer, int length,
                                          T position) noexcept
{
    return interpolateCubic(buffer, length, position);
}

/**
 * @brief 4-point Lagrange interpolation.
 *
 * Higher accuracy than cubic/Hermite for smooth signals, at slightly higher
 * CPU cost. Best for precision resampling applications.
 *
 * @param buffer Sample buffer.
 * @param length Buffer length.
 * @param position Fractional read position.
 * @return Interpolated sample value.
 */
template <FloatType T>
[[nodiscard]] inline T interpolateLagrange(const T* buffer, int length,
                                           T position) noexcept
{
    int idx1 = static_cast<int>(position);
    T frac = position - static_cast<T>(idx1);

    auto wrap = [length](int i) -> int {
        return ((i % length) + length) % length;
    };

    T y0 = buffer[wrap(idx1 - 1)];
    T y1 = buffer[idx1];
    T y2 = buffer[wrap(idx1 + 1)];
    T y3 = buffer[wrap(idx1 + 2)];

    // Lagrange basis polynomials evaluated at (frac) with nodes at -1, 0, 1, 2
    T d = frac;
    T dm1 = d - T(1);
    T dm2 = d - T(2);
    T dp1 = d + T(1);

    T l0 = (d * dm1 * dm2) / T(-6);
    T l1 = (dp1 * dm1 * dm2) / T(2);
    T l2 = (dp1 * d * dm2) / T(-2);
    T l3 = (dp1 * d * dm1) / T(6);

    return l0 * y0 + l1 * y1 + l2 * y2 + l3 * y3;
}

/**
 * @brief Allpass interpolation (1-pole) for fractional delay.
 *
 * Uses a single allpass filter to achieve fractional delay. Unlike
 * polynomial methods, the frequency response is magnitude-flat (unity gain
 * at all frequencies) but has frequency-dependent phase. Best for small
 * fractional delays where phase linearity is less important than flatness.
 *
 * @note This function is stateful via the `state` parameter. The caller
 *       must maintain and initialise the state variable (typically to 0).
 *
 * @param currentSample The current input sample.
 * @param previousSample The previous input sample.
 * @param frac Fractional delay (0.0 to 1.0).
 * @param state Allpass filter state (caller-maintained, init to 0).
 * @return Interpolated sample value.
 */
template <FloatType T>
[[nodiscard]] inline T interpolateAllpass(T currentSample, T previousSample,
                                          T frac, T& state) noexcept
{
    // Allpass coefficient for fractional delay
    T coeff = (T(1) - frac) / (T(1) + frac);
    T output = coeff * (currentSample - state) + previousSample;
    state = output;
    return output;
}

} // namespace dspark
