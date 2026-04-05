// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Dither.h
 * @brief TPDF dithering for bit-depth reduction in audio.
 *
 * When converting audio from a higher bit-depth to a lower one (e.g., float → 16-bit,
 * 24-bit → 16-bit), simple truncation introduces correlated quantisation distortion.
 * Dithering adds a carefully shaped noise signal before quantisation, converting
 * the distortion into a constant, low-level noise floor — perceptually far less
 * objectionable.
 *
 * **TPDF (Triangular Probability Density Function)** is the industry standard for
 * audio dithering. It completely eliminates quantisation distortion at the cost of
 * a +4.77 dB noise floor increase (relative to the LSB). No noise shaping is applied
 * here — noise shaping can be added on top if desired.
 *
 * This implementation also supports optional **first-order noise shaping**, which
 * pushes dither noise energy toward higher frequencies where human hearing is less
 * sensitive.
 *
 * Dependencies: C++20 standard library only.
 *
 * @code
 *   // Dither float audio to 16-bit before writing to WAV:
 *   dspark::Dither<float> dither(16);
 *
 *   for (int i = 0; i < numSamples; ++i)
 *       output[i] = dither.processSample(input[i]);
 *   // output[i] is now quantised to 16-bit levels with TPDF dither
 * @endcode
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace dspark {

/**
 * @class Dither
 * @brief TPDF dithering processor for bit-depth reduction.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class Dither
{
public:
    /**
     * @brief Constructs a dithering processor for the target bit depth.
     *
     * @param targetBits Target bit depth (8, 16, 24, or 32). Default: 16.
     * @param noiseShaping Enable first-order noise shaping (default: false).
     */
    explicit Dither(int targetBits = 16, bool noiseShaping = false) noexcept
        : noiseShaping_(noiseShaping)
    {
        setTargetBitDepth(targetBits);
    }

    /**
     * @brief Sets the target bit depth.
     *
     * @param bits Target bit depth (8, 16, 24, or 32).
     */
    void setTargetBitDepth(int bits) noexcept
    {
        targetBits_ = std::clamp(bits, 1, 32);

        // Quantisation step size for the target bit depth
        // For N bits, there are 2^(N-1) levels in [0, 1], so the step is 1/2^(N-1)
        quantStep_ = T(1) / static_cast<T>(int64_t(1) << (targetBits_ - 1));
    }

    /** @brief Enables or disables first-order noise shaping. */
    void setNoiseShaping(bool enabled) noexcept { noiseShaping_ = enabled; }

    /** @brief Resets the noise shaping state and RNG to a clean state. */
    void reset() noexcept
    {
        for (auto& e : errorState_) e = T(0);
    }

    /**
     * @brief Applies TPDF dither and quantises a single sample.
     *
     * @param input  Input sample (typically in [-1, 1] float range).
     * @param channel Channel index for independent noise shaping state (default: 0).
     * @return Dithered and quantised sample.
     */
    [[nodiscard]] T processSample(T input, int channel = 0) noexcept
    {
        // Generate TPDF noise: sum of two uniform random values → triangular distribution
        T noise = (nextRandom() + nextRandom()) * quantStep_;

        T toQuantise = input;

        // Apply noise shaping feedback (first-order high-pass)
        if (noiseShaping_ && channel < kMaxChannels)
        {
            toQuantise -= errorState_[static_cast<size_t>(channel)];
        }

        // Add dither
        T dithered = toQuantise + noise;

        // Quantise: round to nearest quantisation step
        T quantised = std::round(dithered / quantStep_) * quantStep_;

        // Clamp to valid range
        quantised = std::clamp(quantised, T(-1), T(1));

        // Update noise shaping error (difference between quantised and original)
        if (noiseShaping_ && channel < kMaxChannels)
        {
            errorState_[static_cast<size_t>(channel)] = quantised - toQuantise;
        }

        return quantised;
    }

    /**
     * @brief Applies dithering to an entire audio buffer in-place.
     *
     * @param data       Pointer to interleaved or per-channel audio data.
     * @param numSamples Number of samples per channel.
     * @param channel    Channel index.
     */
    void processBlock(T* data, int numSamples, int channel = 0) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            data[i] = processSample(data[i], channel);
    }

    /** @brief Returns the current target bit depth. */
    [[nodiscard]] int getTargetBitDepth() const noexcept { return targetBits_; }

    /** @brief Returns the quantisation step size. */
    [[nodiscard]] T getQuantisationStep() const noexcept { return quantStep_; }

private:
    static constexpr int kMaxChannels = 16;

    /**
     * @brief Simple, fast PRNG for dither noise generation.
     *
     * Uses a 32-bit xorshift. The quality requirement for dither noise is low —
     * we only need a uniform distribution, not cryptographic randomness.
     * This is deliberately simpler and faster than AnalogRandom.
     *
     * @return Random value in [-0.5, +0.5).
     */
    [[nodiscard]] T nextRandom() noexcept
    {
        rngState_ ^= rngState_ << 13;
        rngState_ ^= rngState_ >> 17;
        rngState_ ^= rngState_ << 5;

        // Convert to [-0.5, 0.5) range
        constexpr T scale = T(1) / static_cast<T>(std::numeric_limits<uint32_t>::max());
        return static_cast<T>(rngState_) * scale - T(0.5);
    }

    int targetBits_ = 16;
    T quantStep_ = T(1) / T(32768);  // Default: 16-bit
    bool noiseShaping_ = false;
    uint32_t rngState_ = 0x12345678u;
    std::array<T, kMaxChannels> errorState_ {};
};

} // namespace dspark
