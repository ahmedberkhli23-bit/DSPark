// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file SpinLock.h
 * @brief Lightweight spin lock for real-time audio thread synchronisation.
 *
 * A busy-wait mutex built on std::atomic_flag. Designed for protecting very
 * short critical sections (a few assignments) where sleeping would introduce
 * unacceptable latency — typically parameter hand-off between a GUI thread
 * and the audio thread.
 *
 * Dependencies: C++20 standard library only (<atomic>).
 *
 * @warning Do NOT use for long critical sections or high-contention scenarios.
 *          In those cases, prefer std::mutex or a lock-free data structure.
 *
 * @code
 *   dspark::SpinLock lock;
 *
 *   // From the GUI thread:
 *   {
 *       dspark::SpinLock::ScopedLock guard(lock);
 *       sharedParams = newParams;
 *   }
 *
 *   // From the audio thread (non-blocking attempt):
 *   {
 *       dspark::SpinLock::ScopedTryLock guard(lock);
 *       if (guard.isLocked())
 *           localCopy = sharedParams;
 *   }
 * @endcode
 */

#include <atomic>

#if defined(_MSC_VER) || defined(__SSE2__)
  #include <immintrin.h>
  #define DSPARK_SPIN_PAUSE() _mm_pause()
#elif defined(__aarch64__) || defined(__ARM_NEON)
  #define DSPARK_SPIN_PAUSE() __asm__ __volatile__("yield")
#else
  #define DSPARK_SPIN_PAUSE() ((void)0)
#endif

namespace dspark {

/**
 * @class SpinLock
 * @brief A minimal, real-time safe spin lock.
 *
 * - `lock()` busy-waits until the lock is acquired (suitable for RT threads
 *   when the protected section is guaranteed to be very short).
 * - `tryLock()` attempts a single acquire without waiting — ideal for the
 *   audio thread where blocking must be avoided.
 * - `unlock()` releases the lock.
 *
 * Use `ScopedLock` for RAII locking and `ScopedTryLock` for non-blocking attempts.
 */
class SpinLock
{
public:
    SpinLock() = default;

    SpinLock(const SpinLock&)            = delete;
    SpinLock& operator=(const SpinLock&) = delete;

    /**
     * @brief Acquires the lock, spinning until successful.
     * @note Call only when the lock holder is guaranteed to release quickly.
     */
    void lock() noexcept
    {
        while (flag_.test_and_set(std::memory_order_acquire))
            DSPARK_SPIN_PAUSE(); // Hint CPU to yield resources during spin
    }

    /**
     * @brief Attempts to acquire the lock without waiting.
     * @return true if the lock was acquired, false if it was already held.
     */
    [[nodiscard]] bool tryLock() noexcept
    {
        return !flag_.test_and_set(std::memory_order_acquire);
    }

    /**
     * @brief Releases the lock.
     */
    void unlock() noexcept
    {
        flag_.clear(std::memory_order_release);
    }

    // ========================================================================

    /**
     * @class ScopedLock
     * @brief RAII wrapper that acquires the lock on construction and releases on destruction.
     */
    class ScopedLock
    {
    public:
        explicit ScopedLock(SpinLock& spinLock) noexcept : lock_(spinLock) { lock_.lock(); }
        ~ScopedLock() noexcept { lock_.unlock(); }

        ScopedLock(const ScopedLock&)            = delete;
        ScopedLock& operator=(const ScopedLock&) = delete;

    private:
        SpinLock& lock_;
    };

    /**
     * @class ScopedTryLock
     * @brief RAII wrapper that *tries* to acquire the lock without blocking.
     *
     * Check `isLocked()` before accessing the protected resource.
     */
    class ScopedTryLock
    {
    public:
        explicit ScopedTryLock(SpinLock& spinLock) noexcept
            : lock_(spinLock), acquired_(spinLock.tryLock()) {}

        ~ScopedTryLock() noexcept { if (acquired_) lock_.unlock(); }

        /** @brief Returns true if the lock was successfully acquired. */
        [[nodiscard]] bool isLocked() const noexcept { return acquired_; }

        ScopedTryLock(const ScopedTryLock&)            = delete;
        ScopedTryLock& operator=(const ScopedTryLock&) = delete;

    private:
        SpinLock& lock_;
        bool      acquired_;
    };

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

} // namespace dspark
