// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DenormalGuard.h
 * @brief RAII guard to disable denormalised floating-point numbers.
 *
 * Denormals (subnormal floats) cause massive CPU spikes in IIR filters,
 * feedback loops, and any recursive DSP algorithm. This guard sets the
 * processor to flush denormals to zero on construction and restores the
 * previous state on destruction.
 *
 * Platform support:
 * - **x86/x64 (SSE):** Sets FTZ + DAZ bits via `_mm_setcsr` / `_mm_getcsr`.
 * - **ARM (NEON):** Sets FZ bit via inline assembly or intrinsic.
 * - **Other / WASM:** No-op (WebAssembly has no denormals by spec).
 *
 * Dependencies: none.
 *
 * @code
 *   void processBlock(float* data, int numSamples)
 *   {
 *       dspark::DenormalGuard guard;  // denormals disabled for this scope
 *
 *       for (int i = 0; i < numSamples; ++i)
 *           data[i] = myIIRFilter.processSample(data[i]);
 *
 *   }  // original FP state restored here
 * @endcode
 */

#include <cstdint>

#if defined(_MSC_VER) || defined(__SSE__)
    #include <immintrin.h>
#endif

namespace dspark {

/**
 * @class DenormalGuard
 * @brief RAII scope guard that disables denormalised floating-point numbers.
 *
 * Create an instance at the top of your audio callback or processing
 * function. Denormals are flushed to zero for the lifetime of the object.
 */
class DenormalGuard
{
public:
    /**
     * @brief Saves current FP state and enables flush-to-zero.
     *
     * On x86: sets FTZ (bit 15) and DAZ (bit 6) in the MXCSR register.
     * On ARM: sets FZ bit in the FPCR register.
     */
    DenormalGuard() noexcept
    {
#if defined(_MSC_VER) || defined(__SSE__)
        previousState_ = _mm_getcsr();
        _mm_setcsr(static_cast<unsigned int>(previousState_) | 0x8040u); // FTZ (bit 15) | DAZ (bit 6)
#elif defined(__aarch64__)
        unsigned long long fpcr;
        __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
        previousState_ = fpcr;
        fpcr |= (1ULL << 24); // FZ bit
        __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
#elif defined(__arm__)
        unsigned int fpscr;
        __asm__ __volatile__("vmrs %0, fpscr" : "=r"(fpscr));
        previousState_ = fpscr;
        fpscr |= (1u << 24); // FZ bit
        __asm__ __volatile__("vmsr fpscr, %0" : : "r"(fpscr));
#else
        // WASM, other: denormals don't exist or can't be controlled
        previousState_ = 0;
#endif
    }

    /**
     * @brief Restores the FP state that was active before construction.
     */
    ~DenormalGuard() noexcept
    {
#if defined(_MSC_VER) || defined(__SSE__)
        _mm_setcsr(static_cast<unsigned int>(previousState_));
#elif defined(__aarch64__)
        unsigned long long fpcr = static_cast<unsigned long long>(previousState_);
        __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
#elif defined(__arm__)
        { unsigned int fpscr = static_cast<unsigned int>(previousState_);
        __asm__ __volatile__("vmsr fpscr, %0" : : "r"(fpscr)); }
#endif
    }

    // Non-copyable, non-movable (RAII scope guard)
    DenormalGuard(const DenormalGuard&) = delete;
    DenormalGuard& operator=(const DenormalGuard&) = delete;
    DenormalGuard(DenormalGuard&&) = delete;
    DenormalGuard& operator=(DenormalGuard&&) = delete;

private:
    uint64_t previousState_ = 0;
};

} // namespace dspark
