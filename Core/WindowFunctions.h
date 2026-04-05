// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file WindowFunctions.h
 * @brief Standard window functions for spectral analysis and FIR filter design.
 *
 * Window functions taper the edges of an audio frame to reduce spectral leakage
 * when using the FFT. Each window has different trade-offs between main lobe width
 * (frequency resolution) and side lobe level (leakage suppression).
 *
 * Quick guide for choosing a window:
 *
 * | Window         | Main lobe | Side lobes | Best for                         |
 * |----------------|-----------|------------|----------------------------------|
 * | Hann           | Medium    | -31 dB     | General-purpose spectral analysis |
 * | Hamming        | Medium    | -42 dB     | Speech analysis, FIR design       |
 * | Blackman       | Wide      | -58 dB     | High dynamic range analysis       |
 * | BlackmanHarris | Wide      | -92 dB     | Precision measurement             |
 * | Kaiser         | Variable  | Variable   | Configurable — FIR design         |
 * | FlatTop        | Very wide | -93 dB     | Amplitude-accurate measurement    |
 * | Rectangular    | Narrowest | -13 dB     | Transient analysis (no windowing) |
 * | Triangular     | Medium    | -26 dB     | Simple overlap-add applications   |
 *
 * Dependencies: C++20 standard library only.
 *
 * @code
 *   // Apply a Hann window to a 1024-sample frame before FFT:
 *   std::vector<float> window(1024);
 *   dspark::WindowFunctions<float>::hann(window.data(), 1024);
 *
 *   for (int i = 0; i < 1024; ++i)
 *       frame[i] *= window[i];
 *
 *   fft.forward(frame, spectrum);
 * @endcode
 */

#include <cassert>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class WindowFunctions
 * @brief Static methods that fill a buffer with window function values.
 *
 * All methods write `size` values into the output array. The values are
 * normalised so the peak is 1.0 (except FlatTop, whose peak may exceed 1.0).
 *
 * For overlap-add applications, use the periodic form (default: `periodic = true`),
 * which ensures perfect reconstruction when overlapping at 50% or 75%.
 * For DFT analysis of a complete signal, use the symmetric form (`periodic = false`).
 *
 * @tparam T Output type (float or double).
 */
template <typename T>
class WindowFunctions
{
public:
    /**
     * @brief Rectangular window (no windowing — all ones).
     *
     * Narrowest main lobe but worst side lobe suppression (-13 dB).
     * Use when you want no modification, or for transient-preserving analysis.
     */
    static void rectangular(T* output, int size) noexcept
    {
        for (int i = 0; i < size; ++i)
            output[i] = T(1);
    }

    /**
     * @brief Triangular (Bartlett) window.
     *
     * Linear taper from 0 at the edges to 1 at the centre.
     * -26 dB side lobes. Simple and useful for overlap-add.
     */
    static void triangular(T* output, int size, bool periodic = true) noexcept
    {
        const int N = periodic ? size : size - 1;
        const T halfN = static_cast<T>(N) / T(2);

        for (int i = 0; i < size; ++i)
            output[i] = T(1) - std::abs((static_cast<T>(i) - halfN) / halfN);
    }

    /**
     * @brief Hann (raised cosine) window.
     *
     * The most commonly used window in audio. Good balance between frequency
     * resolution and leakage suppression (-31 dB side lobes).
     * Perfect for general spectral analysis and STFT.
     *
     * @param periodic If true (default), generates N-point periodic window for overlap-add.
     */
    static void hann(T* output, int size, bool periodic = true) noexcept
    {
        const int N = periodic ? size : size - 1;
        for (int i = 0; i < size; ++i)
        {
            T x = static_cast<T>(i) / static_cast<T>(N);
            output[i] = T(0.5) - T(0.5) * std::cos(T(2) * std::numbers::pi_v<T> * x);
        }
    }

    /**
     * @brief Hamming window.
     *
     * Similar to Hann but with slightly better side lobe suppression (-42 dB)
     * at the cost of not reaching zero at the edges. Popular for speech processing
     * and FIR filter design.
     */
    static void hamming(T* output, int size, bool periodic = true) noexcept
    {
        const int N = periodic ? size : size - 1;
        for (int i = 0; i < size; ++i)
        {
            T x = static_cast<T>(i) / static_cast<T>(N);
            output[i] = T(0.54) - T(0.46) * std::cos(T(2) * std::numbers::pi_v<T> * x);
        }
    }

    /**
     * @brief Blackman window.
     *
     * Wider main lobe but excellent side lobe suppression (-58 dB).
     * Use when you need high dynamic range in your spectral analysis.
     */
    static void blackman(T* output, int size, bool periodic = true) noexcept
    {
        const int N = periodic ? size : size - 1;
        constexpr T a0 = T(0.42);
        constexpr T a1 = T(0.5);
        constexpr T a2 = T(0.08);

        for (int i = 0; i < size; ++i)
        {
            T x = static_cast<T>(i) / static_cast<T>(N);
            T tp = T(2) * std::numbers::pi_v<T>;
            output[i] = a0 - a1 * std::cos(tp * x) + a2 * std::cos(T(2) * tp * x);
        }
    }

    /**
     * @brief Blackman-Harris window (4-term).
     *
     * Outstanding side lobe suppression (-92 dB) with a wider main lobe.
     * The best choice for high-precision spectral measurements.
     */
    static void blackmanHarris(T* output, int size, bool periodic = true) noexcept
    {
        const int N = periodic ? size : size - 1;
        constexpr T a0 = T(0.35875);
        constexpr T a1 = T(0.48829);
        constexpr T a2 = T(0.14128);
        constexpr T a3 = T(0.01168);

        for (int i = 0; i < size; ++i)
        {
            T x = static_cast<T>(i) / static_cast<T>(N);
            T tp = T(2) * std::numbers::pi_v<T>;
            output[i] = a0 - a1 * std::cos(tp * x)
                            + a2 * std::cos(T(2) * tp * x)
                            - a3 * std::cos(T(3) * tp * x);
        }
    }

    /**
     * @brief Flat-top window.
     *
     * Designed for amplitude-accurate measurements. The main lobe is very wide
     * (poor frequency resolution) but the amplitude response is nearly flat
     * at the top, minimising amplitude error (<0.01 dB). Side lobes at -93 dB.
     *
     * @note Peak value may exceed 1.0. Normalise if needed.
     */
    static void flatTop(T* output, int size, bool periodic = true) noexcept
    {
        const int N = periodic ? size : size - 1;
        constexpr T a0 = T(0.21557895);
        constexpr T a1 = T(0.41663158);
        constexpr T a2 = T(0.277263158);
        constexpr T a3 = T(0.083578947);
        constexpr T a4 = T(0.006947368);

        for (int i = 0; i < size; ++i)
        {
            T x = static_cast<T>(i) / static_cast<T>(N);
            T tp = T(2) * std::numbers::pi_v<T>;
            output[i] = a0 - a1 * std::cos(tp * x)
                            + a2 * std::cos(T(2) * tp * x)
                            - a3 * std::cos(T(3) * tp * x)
                            + a4 * std::cos(T(4) * tp * x);
        }
    }

    /**
     * @brief Kaiser window with configurable shape parameter beta.
     *
     * The Kaiser window is uniquely configurable: the `beta` parameter controls
     * the trade-off between main lobe width and side lobe attenuation.
     *
     * | beta  | Side lobe attenuation | Equivalent to     |
     * |-------|----------------------|-------------------|
     * | 0.0   | -13 dB               | Rectangular       |
     * | 5.0   | -36 dB               | ~Hamming          |
     * | 6.0   | -44 dB               | ~Hann             |
     * | 8.6   | -69 dB               | ~Blackman         |
     * | 14.0  | -100+ dB             | Very high isolation|
     *
     * Essential for FIR filter design (Parks-McClellan, windowed-sinc).
     *
     * @param beta Shape parameter (typically 0.0 to 14.0). Higher = more attenuation.
     */
    static void kaiser(T* output, int size, T beta, bool periodic = true) noexcept
    {
        const int N = periodic ? size : size - 1;
        const T denominator = bessel_I0(beta);

        for (int i = 0; i < size; ++i)
        {
            T x = T(2) * static_cast<T>(i) / static_cast<T>(N) - T(1);
            T arg = beta * std::sqrt(T(1) - x * x);
            output[i] = bessel_I0(arg) / denominator;
        }
    }

    // -- Utility methods -------------------------------------------------------

    /**
     * @brief Applies a window to a signal buffer in-place.
     *
     * Multiplies each sample by the corresponding window value.
     *
     * @param signal Signal buffer to window (modified in-place).
     * @param window Pre-computed window values.
     * @param size   Number of samples.
     */
    static void apply(T* signal, const T* window, int size) noexcept
    {
        for (int i = 0; i < size; ++i)
            signal[i] *= window[i];
    }

    /**
     * @brief Computes the coherent gain of a window.
     *
     * The coherent gain is the sum of all window values divided by N.
     * Used to compensate amplitude after windowing.
     *
     * @param window Window values.
     * @param size   Number of values.
     * @return Coherent gain (0.0 to 1.0).
     */
    [[nodiscard]] static T coherentGain(const T* window, int size) noexcept
    {
        T sum = T(0);
        for (int i = 0; i < size; ++i)
            sum += window[i];
        return sum / static_cast<T>(size);
    }

    /**
     * @brief Computes the energy gain of a window.
     *
     * The energy gain is the RMS of the window values. Used to compensate
     * power spectrum after windowing.
     *
     * @param window Window values.
     * @param size   Number of values.
     * @return Energy gain.
     */
    [[nodiscard]] static T energyGain(const T* window, int size) noexcept
    {
        T sum = T(0);
        for (int i = 0; i < size; ++i)
            sum += window[i] * window[i];
        return std::sqrt(sum / static_cast<T>(size));
    }

private:
    /**
     * @brief Modified Bessel function of the first kind, order 0 (I0).
     *
     * Computed via the series expansion. Converges rapidly for typical
     * Kaiser window beta values (0 to ~20).
     */
    [[nodiscard]] static T bessel_I0(T x) noexcept
    {
        T sum = T(1);
        T term = T(1);
        const T halfX = x / T(2);

        for (int k = 1; k < 25; ++k)
        {
            term *= (halfX / static_cast<T>(k));
            term *= (halfX / static_cast<T>(k));
            sum += term;
            if (term < sum * T(1e-12)) break; // Converged
        }

        return sum;
    }
};

} // namespace dspark
