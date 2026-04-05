// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file MidSide.h
 * @brief Mid/Side stereo encoding and decoding for real-time audio.
 *
 * Provides lossless conversion between Left/Right and Mid/Side representations.
 * Useful for stereo processing where independent control over the centre image
 * (mid) and stereo width (side) is needed — e.g., mid-only saturation, side EQ,
 * or stereo width adjustment.
 *
 * Dependencies: AudioBuffer.h only.
 *
 * The transform is its own inverse (encode == decode), but separate functions
 * are provided for clarity at the call site.
 *
 * @note The buffer must have exactly 2 channels. Channel 0 = Left (or Mid),
 *       Channel 1 = Right (or Side).
 *
 * @code
 *   dspark::MidSide<float>::encode(buffer);  // L,R → M,S
 *   processOnlyMid(buffer.getChannel(0), numSamples);
 *   dspark::MidSide<float>::decode(buffer);  // M,S → L,R
 * @endcode
 */

#include "../Core/AudioBuffer.h"

#include <cassert>

namespace dspark {

/**
 * @struct MidSide
 * @brief Static utility for Mid/Side stereo encoding and decoding.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
struct MidSide
{
    /**
     * @brief Encodes a stereo buffer from Left/Right to Mid/Side.
     *
     * ```
     *   M = (L + R) * 0.5
     *   S = (L - R) * 0.5
     * ```
     *
     * The 0.5 scaling preserves energy (unity round-trip gain).
     *
     * @param buffer Stereo buffer (2 channels). Modified in-place.
     */
    static void encode(AudioBufferView<T> buffer) noexcept
    {
        assert(buffer.getNumChannels() >= 2);

        T* left  = buffer.getChannel(0);
        T* right = buffer.getChannel(1);
        const int n = buffer.getNumSamples();

        for (int i = 0; i < n; ++i)
        {
            const T l = left[i];
            const T r = right[i];
            left[i]  = (l + r) * T(0.5); // Mid
            right[i] = (l - r) * T(0.5); // Side
        }
    }

    /**
     * @brief Decodes a stereo buffer from Mid/Side back to Left/Right.
     *
     * ```
     *   L = M + S
     *   R = M - S
     * ```
     *
     * @param buffer Stereo buffer (2 channels, containing M/S). Modified in-place.
     */
    static void decode(AudioBufferView<T> buffer) noexcept
    {
        assert(buffer.getNumChannels() >= 2);

        T* mid  = buffer.getChannel(0);
        T* side = buffer.getChannel(1);
        const int n = buffer.getNumSamples();

        for (int i = 0; i < n; ++i)
        {
            const T m = mid[i];
            const T s = side[i];
            mid[i]  = m + s; // Left
            side[i] = m - s; // Right
        }
    }
};

} // namespace dspark
