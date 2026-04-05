// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file FIRFilter.h
 * @brief FIR (Finite Impulse Response) filter with windowed-sinc coefficient design.
 *
 * FIR filters provide **linear phase** response — they do not distort the phase
 * of the signal. This makes them essential for:
 * - Mastering-grade equalisation
 * - Linear-phase crossovers
 * - Sample rate conversion (anti-aliasing)
 * - Any application where phase accuracy matters
 *
 * Trade-offs vs IIR (Biquad):
 *
 * |              | IIR (Biquad)        | FIR                     |
 * |--------------|---------------------|-------------------------|
 * | Phase        | Non-linear          | **Linear** (symmetric)  |
 * | Efficiency   | Very few coefficients| Many coefficients needed|
 * | Latency      | Minimal             | N/2 samples             |
 * | Stability    | Can be unstable     | Always stable           |
 *
 * For short FIR filters (<= ~256 taps), direct convolution is used.
 * For longer filters, use the Convolver class which uses FFT-based
 * overlap-save for O(N log N) efficiency.
 *
 * Dependencies: WindowFunctions.h, DspMath.h.
 *
 * @code
 *   // Create a 255-tap low-pass FIR at 4 kHz (sample rate 48 kHz):
 *   auto coeffs = dspark::FIRDesign<float>::lowPass(48000.0, 4000.0, 255);
 *
 *   dspark::FIRFilter<float> filter;
 *   filter.setCoefficients(coeffs.data(), static_cast<int>(coeffs.size()));
 *   filter.prepare(2);  // 2 channels
 *
 *   // Process audio:
 *   for (int i = 0; i < numSamples; ++i)
 *       output[i] = filter.processSample(input[i], 0);
 * @endcode
 */

#include "DspMath.h"
#include "SimdOps.h"
#include "WindowFunctions.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numbers>
#include <type_traits>
#include <vector>

namespace dspark {

// ============================================================================
// FIRDesign — Coefficient design via windowed-sinc method
// ============================================================================

/**
 * @class FIRDesign
 * @brief Static methods for designing FIR filter coefficients.
 *
 * Uses the windowed-sinc method: an ideal (sinc) impulse response is
 * multiplied by a window function to produce a realisable FIR filter.
 *
 * All methods return a vector of filter coefficients (taps). The number
 * of taps determines the filter's steepness and stopband attenuation.
 * More taps = steeper transition but more latency and computation.
 *
 * Rule of thumb for number of taps:
 * - `N ≈ 4 / (transitionWidth / sampleRate)` for a Kaiser window
 * - For a 1 kHz transition band at 48 kHz: N ≈ 4 / (1000/48000) ≈ 192 taps
 * - Always use an **odd** number for symmetric (Type I) FIR filters
 *
 * @tparam T Coefficient type (float or double).
 */
template <typename T>
class FIRDesign
{
public:
    /**
     * @brief Designs a low-pass FIR filter.
     *
     * @param sampleRate Sample rate in Hz.
     * @param cutoffHz   Cutoff frequency in Hz (−6 dB point).
     * @param numTaps    Number of filter taps (must be odd, ≥ 3).
     * @param beta       Kaiser window beta (default: 5.0). Higher = more attenuation.
     * @return Vector of filter coefficients.
     */
    [[nodiscard]] static std::vector<T> lowPass(double sampleRate, double cutoffHz,
                                                 int numTaps, T beta = T(5)) noexcept
    {
        assert(numTaps >= 3 && (numTaps % 2 == 1));
        return designSinc(cutoffHz / sampleRate, numTaps, beta, false);
    }

    /**
     * @brief Designs a high-pass FIR filter.
     *
     * Created by spectrally inverting a low-pass filter.
     *
     * @param sampleRate Sample rate in Hz.
     * @param cutoffHz   Cutoff frequency in Hz.
     * @param numTaps    Number of filter taps (must be odd, ≥ 3).
     * @param beta       Kaiser window beta.
     * @return Vector of filter coefficients.
     */
    [[nodiscard]] static std::vector<T> highPass(double sampleRate, double cutoffHz,
                                                  int numTaps, T beta = T(5)) noexcept
    {
        assert(numTaps >= 3 && (numTaps % 2 == 1));
        return designSinc(cutoffHz / sampleRate, numTaps, beta, true);
    }

    /**
     * @brief Designs a band-pass FIR filter.
     *
     * Created by subtracting a low-pass at lowCutoff from a low-pass at highCutoff.
     *
     * @param sampleRate  Sample rate in Hz.
     * @param lowCutoffHz Lower cutoff frequency in Hz.
     * @param highCutoffHz Upper cutoff frequency in Hz.
     * @param numTaps     Number of filter taps (must be odd, ≥ 3).
     * @param beta        Kaiser window beta.
     * @return Vector of filter coefficients.
     */
    [[nodiscard]] static std::vector<T> bandPass(double sampleRate, double lowCutoffHz,
                                                  double highCutoffHz, int numTaps,
                                                  T beta = T(5)) noexcept
    {
        assert(numTaps >= 3 && (numTaps % 2 == 1));
        assert(lowCutoffHz < highCutoffHz);

        auto lp1 = designSinc(highCutoffHz / sampleRate, numTaps, beta, false);
        auto lp2 = designSinc(lowCutoffHz / sampleRate, numTaps, beta, false);

        for (int i = 0; i < numTaps; ++i)
            lp1[static_cast<size_t>(i)] -= lp2[static_cast<size_t>(i)];

        return lp1;
    }

    /**
     * @brief Designs a band-stop (notch) FIR filter.
     *
     * @param sampleRate  Sample rate in Hz.
     * @param lowCutoffHz Lower cutoff frequency in Hz.
     * @param highCutoffHz Upper cutoff frequency in Hz.
     * @param numTaps     Number of filter taps (must be odd, ≥ 3).
     * @param beta        Kaiser window beta.
     * @return Vector of filter coefficients.
     */
    [[nodiscard]] static std::vector<T> bandStop(double sampleRate, double lowCutoffHz,
                                                  double highCutoffHz, int numTaps,
                                                  T beta = T(5)) noexcept
    {
        auto bp = bandPass(sampleRate, lowCutoffHz, highCutoffHz, numTaps, beta);

        // Spectral inversion: negate all and add 1 to centre tap
        int centre = numTaps / 2;
        for (int i = 0; i < numTaps; ++i)
            bp[static_cast<size_t>(i)] = -bp[static_cast<size_t>(i)];
        bp[static_cast<size_t>(centre)] += T(1);

        return bp;
    }

    /**
     * @brief Estimates the required number of taps for a given specification.
     *
     * Uses the Kaiser formula to estimate the minimum number of odd taps needed
     * to achieve the desired stopband attenuation with a given transition bandwidth.
     *
     * @param sampleRate      Sample rate in Hz.
     * @param transitionHz    Width of the transition band in Hz.
     * @param attenuationDb   Desired stopband attenuation in dB (positive value, e.g., 60).
     * @return Estimated number of taps (always odd).
     */
    [[nodiscard]] static int estimateTaps(double sampleRate, double transitionHz,
                                           double attenuationDb) noexcept
    {
        double normTransition = transitionHz / sampleRate;
        int n = static_cast<int>(std::ceil((attenuationDb - 7.95) / (14.36 * normTransition)));
        if (n < 3) n = 3;
        if (n % 2 == 0) ++n; // Ensure odd
        return n;
    }

    /**
     * @brief Estimates the Kaiser beta parameter for a desired attenuation.
     *
     * @param attenuationDb Desired stopband attenuation in dB (positive).
     * @return Kaiser beta parameter.
     */
    [[nodiscard]] static T estimateKaiserBeta(double attenuationDb) noexcept
    {
        if (attenuationDb > 50.0)
            return static_cast<T>(0.1102 * (attenuationDb - 8.7));
        else if (attenuationDb >= 21.0)
            return static_cast<T>(0.5842 * std::pow(attenuationDb - 21.0, 0.4)
                                + 0.07886 * (attenuationDb - 21.0));
        else
            return T(0);
    }

private:
    /**
     * @brief Core windowed-sinc FIR design.
     *
     * @param normFreq Normalised cutoff frequency (cutoff / sampleRate), range [0, 0.5].
     * @param numTaps  Number of taps (odd).
     * @param beta     Kaiser window parameter.
     * @param invert   If true, spectrally invert for high-pass.
     * @return Coefficient vector.
     */
    [[nodiscard]] static std::vector<T> designSinc(double normFreq, int numTaps,
                                                    T beta, bool invert) noexcept
    {
        std::vector<T> coeffs(static_cast<size_t>(numTaps));
        std::vector<T> window(static_cast<size_t>(numTaps));

        // Generate Kaiser window
        WindowFunctions<T>::kaiser(window.data(), numTaps, beta, false);

        const int centre = numTaps / 2;
        const double fc = normFreq * 2.0; // Normalised to [0, 1] for sinc
        constexpr double kPi = std::numbers::pi;

        // Compute windowed sinc
        for (int i = 0; i < numTaps; ++i)
        {
            int n = i - centre;
            if (n == 0)
            {
                coeffs[static_cast<size_t>(i)] = static_cast<T>(fc);
            }
            else
            {
                double x = static_cast<double>(n) * kPi;
                coeffs[static_cast<size_t>(i)] = static_cast<T>(
                    std::sin(fc * x) / x);
            }

            // Apply window
            coeffs[static_cast<size_t>(i)] *= window[static_cast<size_t>(i)];
        }

        // Normalise for unity gain at DC (low-pass) or Nyquist (high-pass)
        T sum = T(0);
        for (auto c : coeffs) sum += c;
        if (std::abs(sum) > T(1e-10))
        {
            T invSum = T(1) / sum;
            for (auto& c : coeffs) c *= invSum;
        }

        // Spectral inversion for high-pass
        if (invert)
        {
            for (auto& c : coeffs) c = -c;
            coeffs[static_cast<size_t>(centre)] += T(1);
        }

        return coeffs;
    }
};

// ============================================================================
// FIRFilter — FIR filter processor (direct convolution)
// ============================================================================

/**
 * @class FIRFilter
 * @brief FIR filter using direct-form convolution.
 *
 * Suitable for filters up to ~256-512 taps. For longer filters (e.g., reverb
 * impulse responses), use the Convolver class which is FFT-based.
 *
 * Each channel has an independent delay line (ring buffer).
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class FIRFilter
{
public:
    /**
     * @brief Sets the filter coefficients.
     *
     * @param coeffs  Pointer to coefficient array.
     * @param numTaps Number of coefficients.
     */
    void setCoefficients(const T* coeffs, int numTaps)
    {
        coeffs_.assign(coeffs, coeffs + numTaps);
        numTaps_ = numTaps;
        buildReversedCoeffs();
    }

    /**
     * @brief Sets coefficients from a vector (e.g., from FIRDesign).
     * @param coeffs Coefficient vector.
     */
    void setCoefficients(const std::vector<T>& coeffs)
    {
        coeffs_ = coeffs;
        numTaps_ = static_cast<int>(coeffs.size());
        buildReversedCoeffs();
    }

    /**
     * @brief Prepares internal delay lines for the given number of channels.
     * @param numChannels Number of audio channels.
     */
    void prepare(int numChannels)
    {
        numChannels_ = numChannels;
        // Doubled delay line: write at wp AND wp+numTaps for contiguous SIMD reads
        delayLines_.resize(static_cast<size_t>(numChannels));
        for (auto& dl : delayLines_)
        {
            dl.resize(static_cast<size_t>(numTaps_ * 2), T(0));
            dl.shrink_to_fit();
        }
        writePositions_.resize(static_cast<size_t>(numChannels), 0);
    }

    /** @brief Resets all delay lines to zero. */
    void reset() noexcept
    {
        for (auto& dl : delayLines_)
            std::fill(dl.begin(), dl.end(), T(0));
        std::fill(writePositions_.begin(), writePositions_.end(), 0);
    }

    /**
     * @brief Processes a single sample through the FIR filter.
     *
     * Uses a doubled delay line and reversed coefficients for contiguous memory
     * access, enabling SIMD vectorization of the inner dot product (SSE2/NEON).
     *
     * @param input   Input sample.
     * @param channel Channel index.
     * @return Filtered output sample.
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        auto& dl = delayLines_[static_cast<size_t>(channel)];
        auto& wp = writePositions_[static_cast<size_t>(channel)];

        // Dual write: maintain mirror copy for contiguous reads
        dl[static_cast<size_t>(wp)] = input;
        dl[static_cast<size_t>(wp + numTaps_)] = input;

        // Contiguous dot product: revCoeffs_[k] * dl[wp+1+k] for k=0..numTaps-1
        const T* readPtr = dl.data() + wp + 1;
        T output = dotProduct(revCoeffs_.data(), readPtr, numTaps_);

        ++wp;
        if (wp >= numTaps_) wp = 0;

        return output;
    }

    /**
     * @brief Processes a block of audio in-place.
     *
     * @param data       Audio samples to process.
     * @param numSamples Number of samples.
     * @param channel    Channel index.
     */
    void processBlock(T* data, int numSamples, int channel) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            data[i] = processSample(data[i], channel);
    }

    /** @brief Returns the number of filter taps. */
    [[nodiscard]] int getNumTaps() const noexcept { return numTaps_; }

    /**
     * @brief Returns the filter latency in samples.
     *
     * A symmetric (linear-phase) FIR filter delays the signal by (N-1)/2 samples.
     */
    [[nodiscard]] int getLatency() const noexcept { return (numTaps_ - 1) / 2; }

private:
    void buildReversedCoeffs()
    {
        revCoeffs_.resize(static_cast<size_t>(numTaps_));
        for (int k = 0; k < numTaps_; ++k)
            revCoeffs_[static_cast<size_t>(k)] = coeffs_[static_cast<size_t>(numTaps_ - 1 - k)];
    }

    /// SIMD-accelerated dot product (delegates to simd::dotProduct).
    static T dotProduct(const T* a, const T* b, int n) noexcept
    {
        return simd::dotProduct(a, b, n);
    }

    std::vector<T> coeffs_;
    std::vector<T> revCoeffs_;    ///< Reversed coefficients for contiguous SIMD dot product.
    int numTaps_ = 0;
    int numChannels_ = 0;
    std::vector<std::vector<T>> delayLines_;
    std::vector<int> writePositions_;
};

} // namespace dspark
