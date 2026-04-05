// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file FFT.h
 * @brief Fast Fourier Transform (Cooley-Tukey radix-2) with SIMD acceleration.
 *
 * Provides forward and inverse FFT for both complex and real-valued signals.
 * Optimised for power-of-two sizes typical in audio (256, 512, 1024, 2048, 4096).
 *
 * Two classes:
 *
 * - **FFTComplex\<T\>**: General complex FFT. Operates on interleaved
 *   real/imaginary pairs: [re0, im0, re1, im1, ...].
 *
 * - **FFTReal\<T\>**: Optimised for real-valued input signals (the common case
 *   in audio). Uses a half-size complex FFT internally, saving ~50% computation.
 *   Returns only the positive-frequency half (N/2+1 complex bins) since the
 *   negative frequencies are conjugate-symmetric for real input.
 *
 * Performance features:
 * - **SIMD-accelerated butterfly**: SSE2 on x86-64, NEON on ARM64 (for float).
 *   Processes 2 complex butterflies per SIMD instruction (~2x throughput).
 * - **Pre-computed twiddle factors**: all cos/sin computed once at construction.
 *   Zero trigonometric calls during transform.
 * - **Pre-computed bit-reversal table**: O(1) per swap.
 * - Scalar fallback for double and platforms without SIMD support.
 *
 * All memory is pre-allocated in the constructor — no allocations during transform.
 *
 * Dependencies: C++20 standard library only (+ platform SIMD intrinsics when available).
 *
 * @code
 *   dspark::FFTReal<float> fft(1024);
 *
 *   std::vector<float> freqDomain(fft.getFrequencyDomainSize());
 *   fft.forward(timeDomain.data(), freqDomain.data());
 *
 *   // freqDomain[2*k] = real part of bin k
 *   // freqDomain[2*k+1] = imaginary part of bin k
 *
 *   fft.inverse(freqDomain.data(), timeDomain.data());
 * @endcode
 */

// --- Platform SIMD detection ------------------------------------------------
// x86-64: SSE2 is guaranteed by the AMD64 specification.
// ARM64:  NEON is guaranteed by the AArch64 specification.
// Both are unconditionally available — no runtime feature check needed.

#if defined(_M_AMD64) || defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
    #define DSPARK_FFT_SSE2 1
    #include <emmintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define DSPARK_FFT_NEON 1
    #include <arm_neon.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <type_traits>
#include <vector>

namespace dspark {

// ============================================================================
// FFTComplex — General complex-to-complex FFT
// ============================================================================

/**
 * @class FFTComplex
 * @brief In-place Cooley-Tukey radix-2 DIT FFT for complex data.
 *
 * Data layout: interleaved [re0, im0, re1, im1, ...], total 2*N elements.
 * SIMD-accelerated for T=float on x86-64 (SSE2) and ARM64 (NEON).
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class FFTComplex
{
public:
    /**
     * @brief Constructs an FFT processor for the given size.
     * @param size Number of complex samples (must be power of two, >= 2).
     */
    explicit FFTComplex(int size)
        : size_(size)
    {
        assert(size >= 2 && (size & (size - 1)) == 0);
        computeTwiddles();
        computeBitReversalTable();
    }

    /** @brief Returns the FFT size (number of complex points). */
    [[nodiscard]] int getSize() const noexcept { return size_; }

    /**
     * @brief Performs a forward (time->frequency) FFT in-place.
     * @param data Interleaved complex data [re, im, ...], 2*N elements.
     */
    void forward(T* data) const noexcept
    {
        bitReverse(data);
        butterflyPass(data, false);
    }

    /**
     * @brief Performs an inverse (frequency->time) FFT in-place.
     * Output is scaled by 1/N.
     * @param data Interleaved complex data, overwritten with time-domain result.
     */
    void inverse(T* data) const noexcept
    {
        bitReverse(data);
        butterflyPass(data, true);

        const T invN = T(1) / static_cast<T>(size_);
        const int total = size_ * 2;
        for (int i = 0; i < total; ++i)
            data[i] *= invN;
    }

private:
    void computeTwiddles()
    {
        twiddles_.clear();
        int numStages = 0;
        for (int s = size_; s > 1; s >>= 1) ++numStages;

        int stride = 2;
        for (int stage = 0; stage < numStages; ++stage)
        {
            int halfStride = stride / 2;
            for (int k = 0; k < halfStride; ++k)
            {
                double angle = -2.0 * std::numbers::pi * static_cast<double>(k)
                             / static_cast<double>(stride);
                twiddles_.push_back(static_cast<T>(std::cos(angle)));
                twiddles_.push_back(static_cast<T>(std::sin(angle)));
            }
            stride *= 2;
        }
    }

    void computeBitReversalTable()
    {
        bitrev_.resize(static_cast<size_t>(size_));
        int bits = 0;
        for (int s = size_; s > 1; s >>= 1) ++bits;

        for (int i = 0; i < size_; ++i)
        {
            int rev = 0;
            int val = i;
            for (int b = 0; b < bits; ++b)
            {
                rev = (rev << 1) | (val & 1);
                val >>= 1;
            }
            bitrev_[static_cast<size_t>(i)] = rev;
        }
    }

    void bitReverse(T* data) const noexcept
    {
        for (int i = 0; i < size_; ++i)
        {
            int j = bitrev_[static_cast<size_t>(i)];
            if (i < j)
            {
                std::swap(data[2 * i],     data[2 * j]);
                std::swap(data[2 * i + 1], data[2 * j + 1]);
            }
        }
    }

    void butterflyPass(T* data, bool isInverse) const noexcept
    {
        int twiddleOffset = 0;
        int stride = 2;

        while (stride <= size_)
        {
            int halfStride = stride / 2;

            for (int group = 0; group < size_; group += stride)
            {
                int k = 0;

                // --- SIMD path: float only, 2 butterflies at a time -----------
#if DSPARK_FFT_SSE2
                if constexpr (std::is_same_v<T, float>)
                {
                    // Sign mask for complex multiply: negate re part of p2
                    // addsub_mask[0]=-0, [1]=0, [2]=-0, [3]=0
                    alignas(16) static constexpr float kAddSub[4] =
                        { -0.0f, 0.0f, -0.0f, 0.0f };
                    const __m128 addsubMask = _mm_load_ps(kAddSub);

                    // Inverse: negate imaginary parts of twiddle
                    __m128 invTwMask = _mm_setzero_ps();
                    if (isInverse)
                    {
                        alignas(16) static constexpr float kInvTw[4] =
                            { 0.0f, -0.0f, 0.0f, -0.0f };
                        invTwMask = _mm_load_ps(kInvTw);
                    }

                    for (; k + 1 < halfStride; k += 2)
                    {
                        const int twIdx = twiddleOffset + k * 2;
                        const int eIdx  = 2 * (group + k);
                        const int oIdx  = 2 * (group + k + halfStride);

                        __m128 e = _mm_loadu_ps(&data[eIdx]);
                        __m128 o = _mm_loadu_ps(&data[oIdx]);
                        __m128 w = _mm_xor_ps(
                            _mm_loadu_ps(&twiddles_[static_cast<size_t>(twIdx)]),
                            invTwMask);

                        // Complex multiply t = w * o (2 complex values)
                        __m128 o_re  = _mm_shuffle_ps(o, o, _MM_SHUFFLE(2,2,0,0));
                        __m128 o_im  = _mm_shuffle_ps(o, o, _MM_SHUFFLE(3,3,1,1));
                        __m128 w_sw  = _mm_shuffle_ps(w, w, _MM_SHUFFLE(2,3,0,1));
                        __m128 p1    = _mm_mul_ps(w, o_re);
                        __m128 p2    = _mm_mul_ps(w_sw, o_im);
                        __m128 t     = _mm_add_ps(p1, _mm_xor_ps(p2, addsubMask));

                        _mm_storeu_ps(&data[eIdx], _mm_add_ps(e, t));
                        _mm_storeu_ps(&data[oIdx], _mm_sub_ps(e, t));
                    }
                }
#endif // DSPARK_FFT_SSE2

#if DSPARK_FFT_NEON
                if constexpr (std::is_same_v<T, float>)
                {
                    // Sign mask for addsub (negate elements 0 and 2)
                    static const uint32_t kAddSub[4] = { 0x80000000u, 0, 0x80000000u, 0 };
                    const uint32x4_t addsubMask = vld1q_u32(kAddSub);

                    // Inverse twiddle mask (negate elements 1 and 3)
                    static const uint32_t kInvTw[4] = { 0, 0x80000000u, 0, 0x80000000u };
                    const uint32x4_t invTwMask = isInverse ? vld1q_u32(kInvTw) : vdupq_n_u32(0);

                    for (; k + 1 < halfStride; k += 2)
                    {
                        const int twIdx = twiddleOffset + k * 2;
                        const int eIdx  = 2 * (group + k);
                        const int oIdx  = 2 * (group + k + halfStride);

                        float32x4_t e = vld1q_f32(&data[eIdx]);
                        float32x4_t o = vld1q_f32(&data[oIdx]);
                        float32x4_t w = vreinterpretq_f32_u32(veorq_u32(
                            vreinterpretq_u32_f32(
                                vld1q_f32(&twiddles_[static_cast<size_t>(twIdx)])),
                            invTwMask));

                        // Complex multiply t = w * o
                        float32x4_t o_re = vtrn1q_f32(o, o);      // [re,re,re,re]
                        float32x4_t o_im = vtrn2q_f32(o, o);      // [im,im,im,im]
                        float32x4_t w_sw = vrev64q_f32(w);        // swap re/im pairs
                        float32x4_t p1   = vmulq_f32(w, o_re);
                        float32x4_t p2   = vmulq_f32(w_sw, o_im);
                        float32x4_t t    = vaddq_f32(p1,
                            vreinterpretq_f32_u32(veorq_u32(
                                vreinterpretq_u32_f32(p2), addsubMask)));

                        vst1q_f32(&data[eIdx], vaddq_f32(e, t));
                        vst1q_f32(&data[oIdx], vsubq_f32(e, t));
                    }
                }
#endif // DSPARK_FFT_NEON

                // --- Scalar path: remainder + double + non-SIMD platforms -----
                for (; k < halfStride; ++k)
                {
                    int idx = twiddleOffset + k * 2;
                    T wr = twiddles_[static_cast<size_t>(idx)];
                    T wi = twiddles_[static_cast<size_t>(idx + 1)];

                    if (isInverse) wi = -wi;

                    int evenIdx = 2 * (group + k);
                    int oddIdx  = 2 * (group + k + halfStride);

                    T tr = wr * data[oddIdx] - wi * data[oddIdx + 1];
                    T ti = wr * data[oddIdx + 1] + wi * data[oddIdx];

                    data[oddIdx]     = data[evenIdx]     - tr;
                    data[oddIdx + 1] = data[evenIdx + 1] - ti;
                    data[evenIdx]     += tr;
                    data[evenIdx + 1] += ti;
                }
            }

            twiddleOffset += halfStride * 2;
            stride *= 2;
        }
    }

    int size_;
    std::vector<T>   twiddles_;
    std::vector<int> bitrev_;
};

// ============================================================================
// FFTReal — Optimised FFT for real-valued signals
// ============================================================================

/**
 * @class FFTReal
 * @brief FFT optimised for real-valued input signals (the common audio case).
 *
 * Uses a half-size complex FFT internally, then applies a post-processing step
 * to unpack the full spectrum. This saves ~50% computation vs treating the
 * real signal as complex with zero imaginary parts.
 *
 * **Frequency domain layout**: N+2 elements (interleaved complex).
 * - Bins 0 to N/2 inclusive -> (N/2 + 1) complex values -> (N + 2) floats.
 * - `data[2*k]` = real part of bin k.
 * - `data[2*k+1]` = imaginary part of bin k.
 * - Bin 0 = DC, bin N/2 = Nyquist.
 *
 * All twiddle factors for the real-FFT unpack/pack step are pre-computed at
 * construction. Zero trigonometric calls during forward() / inverse().
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class FFTReal
{
public:
    /**
     * @brief Constructs a real FFT processor.
     * @param size Number of real samples (must be power of two, >= 4).
     */
    explicit FFTReal(int size)
        : realSize_(size)
        , halfSize_(size / 2)
        , complexFFT_(size / 2)
    {
        assert(size >= 4 && (size & (size - 1)) == 0);
        computePostTwiddles();
        workBuffer_.resize(static_cast<size_t>(size));
    }

    /** @brief Returns the number of real input samples (N). */
    [[nodiscard]] int getSize() const noexcept { return realSize_; }

    /**
     * @brief Returns the frequency-domain buffer size in elements.
     * This is N + 2: (N/2 + 1) complex bins x 2 floats per bin.
     */
    [[nodiscard]] int getFrequencyDomainSize() const noexcept { return realSize_ + 2; }

    /**
     * @brief Returns the number of frequency bins (N/2 + 1).
     * Includes DC (bin 0) and Nyquist (bin N/2).
     */
    [[nodiscard]] int getNumBins() const noexcept { return halfSize_ + 1; }

    /**
     * @brief Forward transform: real time-domain -> complex frequency-domain.
     * @param timeData  Input: N real samples.
     * @param freqData  Output: N+2 elements (interleaved complex, N/2+1 bins).
     */
    void forward(const T* timeData, T* freqData) const noexcept
    {
        T* work = workBuffer_.data();
        for (int i = 0; i < halfSize_; ++i)
        {
            work[2 * i]     = timeData[2 * i];
            work[2 * i + 1] = timeData[2 * i + 1];
        }

        complexFFT_.forward(work);
        unpackForward(work, freqData);
    }

    /**
     * @brief Inverse transform: complex frequency-domain -> real time-domain.
     * @param freqData  Input: N+2 elements (interleaved complex, N/2+1 bins).
     * @param timeData  Output: N real samples.
     */
    void inverse(const T* freqData, T* timeData) const noexcept
    {
        T* work = workBuffer_.data();
        packInverse(freqData, work);
        complexFFT_.inverse(work);

        for (int i = 0; i < halfSize_; ++i)
        {
            timeData[2 * i]     = work[2 * i];
            timeData[2 * i + 1] = work[2 * i + 1];
        }
    }

    /**
     * @brief Computes the magnitude (absolute value) of each frequency bin.
     * @param freqData    Input: frequency-domain data from forward() (N+2 elements).
     * @param magnitudes  Output: N/2+1 magnitude values.
     */
    void computeMagnitudes(const T* freqData, T* magnitudes) const noexcept
    {
        for (int k = 0; k <= halfSize_; ++k)
        {
            T re = freqData[2 * k];
            T im = freqData[2 * k + 1];
            magnitudes[k] = std::sqrt(re * re + im * im);
        }
    }

    /**
     * @brief Computes the phase angle (in radians) of each frequency bin.
     * @param freqData  Input: frequency-domain data (N+2 elements).
     * @param phases    Output: N/2+1 phase values in radians (-pi to +pi).
     */
    void computePhases(const T* freqData, T* phases) const noexcept
    {
        for (int k = 0; k <= halfSize_; ++k)
        {
            T re = freqData[2 * k];
            T im = freqData[2 * k + 1];
            phases[k] = std::atan2(im, re);
        }
    }

    /**
     * @brief Computes the power spectrum (magnitude squared) of each bin.
     * @param freqData  Input: frequency-domain data (N+2 elements).
     * @param power     Output: N/2+1 power values.
     */
    void computePowerSpectrum(const T* freqData, T* power) const noexcept
    {
        for (int k = 0; k <= halfSize_; ++k)
        {
            T re = freqData[2 * k];
            T im = freqData[2 * k + 1];
            power[k] = re * re + im * im;
        }
    }

    /**
     * @brief Returns the frequency in Hz corresponding to a given bin index.
     * @param binIndex   Bin index (0 to N/2).
     * @param sampleRate Sample rate in Hz.
     * @param fftSize    FFT size (N).
     */
    [[nodiscard]] static T binToFrequency(int binIndex, double sampleRate, int fftSize) noexcept
    {
        return static_cast<T>(static_cast<double>(binIndex) * sampleRate
                            / static_cast<double>(fftSize));
    }

    /**
     * @brief Returns the bin index closest to a given frequency.
     * @param frequency  Frequency in Hz.
     * @param sampleRate Sample rate in Hz.
     * @param fftSize    FFT size (N).
     */
    [[nodiscard]] static int frequencyToBin(double frequency, double sampleRate, int fftSize) noexcept
    {
        return static_cast<int>(std::round(frequency * static_cast<double>(fftSize) / sampleRate));
    }

private:
    void computePostTwiddles()
    {
        // Pre-compute twiddle factors for the real-FFT unpack/pack step.
        // W_N^k = exp(-j * 2pi * k / N) for k = 0 .. N/2 - 1.
        // Stored as interleaved [cos, sin, cos, sin, ...].
        postTwiddles_.resize(static_cast<size_t>(halfSize_ * 2));
        for (int k = 0; k < halfSize_; ++k)
        {
            double angle = -2.0 * std::numbers::pi * static_cast<double>(k)
                         / static_cast<double>(realSize_);
            postTwiddles_[static_cast<size_t>(2 * k)]     = static_cast<T>(std::cos(angle));
            postTwiddles_[static_cast<size_t>(2 * k + 1)] = static_cast<T>(std::sin(angle));
        }
    }

    void unpackForward(const T* halfFFT, T* fullSpectrum) const noexcept
    {
        const int N2 = halfSize_;

        // DC and Nyquist bins
        T dcRe = halfFFT[0] + halfFFT[1];
        T nyRe = halfFFT[0] - halfFFT[1];

        fullSpectrum[0] = dcRe;
        fullSpectrum[1] = T(0);
        fullSpectrum[2 * N2]     = nyRe;
        fullSpectrum[2 * N2 + 1] = T(0);

        // Remaining bins via conjugate-symmetry unpack with pre-computed twiddles
        for (int k = 1; k < N2; ++k)
        {
            int kConj = N2 - k;

            T hkRe = halfFFT[2 * k];
            T hkIm = halfFFT[2 * k + 1];
            T hcRe = halfFFT[2 * kConj];
            T hcIm = halfFFT[2 * kConj + 1];

            // Xe[k] = 0.5 * (H[k] + conj(H[N/2-k]))
            T xeRe = T(0.5) * (hkRe + hcRe);
            T xeIm = T(0.5) * (hkIm - hcIm);

            // Xo[k] = 0.5 * (H[k] - conj(H[N/2-k]))
            T xoRe = T(0.5) * (hkRe - hcRe);
            T xoIm = T(0.5) * (hkIm + hcIm);

            // Pre-computed twiddle: W = exp(-j*2*pi*k/N)
            T wr = postTwiddles_[static_cast<size_t>(2 * k)];
            T wi = postTwiddles_[static_cast<size_t>(2 * k + 1)];

            // X[k] = Xe[k] + W * (-j * Xo[k])
            // -j * (a + jb) = b - ja
            T joRe = xoIm;
            T joIm = -xoRe;

            T twRe = wr * joRe - wi * joIm;
            T twIm = wr * joIm + wi * joRe;

            fullSpectrum[2 * k]     = xeRe + twRe;
            fullSpectrum[2 * k + 1] = xeIm + twIm;
        }
    }

    void packInverse(const T* fullSpectrum, T* halfFFT) const noexcept
    {
        const int N2 = halfSize_;

        T dcRe = fullSpectrum[0];
        T nyRe = fullSpectrum[2 * N2];

        halfFFT[0] = T(0.5) * (dcRe + nyRe);
        halfFFT[1] = T(0.5) * (dcRe - nyRe);

        for (int k = 1; k < N2; ++k)
        {
            int kConj = N2 - k;

            T xkRe = fullSpectrum[2 * k];
            T xkIm = fullSpectrum[2 * k + 1];
            T xcRe = fullSpectrum[2 * kConj];
            T xcIm = fullSpectrum[2 * kConj + 1];

            T xeRe = T(0.5) * (xkRe + xcRe);
            T xeIm = T(0.5) * (xkIm - xcIm);

            T diffRe = T(0.5) * (xkRe - xcRe);
            T diffIm = T(0.5) * (xkIm + xcIm);

            // Inverse twiddle: conjugate of forward = (wr, -wi)
            T wr =  postTwiddles_[static_cast<size_t>(2 * k)];
            T wi = -postTwiddles_[static_cast<size_t>(2 * k + 1)];

            T twRe = wr * diffRe - wi * diffIm;
            T twIm = wr * diffIm + wi * diffRe;

            // Undo -j multiply
            T xoRe = -twIm;
            T xoIm =  twRe;

            halfFFT[2 * k]     = xeRe + xoRe;
            halfFFT[2 * k + 1] = xeIm + xoIm;
        }
    }

    int realSize_;
    int halfSize_;
    FFTComplex<T> complexFFT_;
    std::vector<T> postTwiddles_;        // Pre-computed W_N^k for unpack/pack
    /// @note workBuffer_ is mutable to allow forward()/inverse() to be const.
    /// This is safe because FFTReal is designed for single-threaded audio use:
    /// only one transform runs at a time on a given instance. Do NOT share a
    /// single FFTReal instance across threads — use one instance per thread.
    mutable std::vector<T> workBuffer_;
};

} // namespace dspark
