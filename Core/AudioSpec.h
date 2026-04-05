// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file AudioSpec.h
 * @brief Describes the audio processing environment (sample rate, block size, channels).
 *
 * Every DSP processor receives an AudioSpec in its `prepare()` method to configure
 * internal resources (buffers, filter coefficients, smoothing rates, etc.).
 *
 * Dependencies: none.
 *
 * @code
 *   dspark::AudioSpec spec { .sampleRate = 48000.0, .maxBlockSize = 512, .numChannels = 2 };
 *   mySaturator.prepare(spec);
 * @endcode
 */

namespace dspark {

/**
 * @struct AudioSpec
 * @brief Describes the audio environment for a DSP processor.
 *
 * Passed to `prepare()` before processing begins. All processors must be
 * re-prepared if any of these values change (e.g., sample rate switch in a DAW).
 */
struct AudioSpec
{
    /**
     * @brief Sample rate in Hz (e.g., 44100.0, 48000.0, 96000.0).
     *
     * Used for computing filter coefficients, smoother time constants,
     * delay line lengths, and any time-dependent parameter.
     */
    double sampleRate = 44100.0;

    /**
     * @brief Maximum number of samples per processing block.
     *
     * Processors use this to pre-allocate internal buffers during `prepare()`.
     * The actual block size passed to `process()` may be smaller but will
     * never exceed this value.
     */
    int maxBlockSize = 512;

    /**
     * @brief Number of audio channels (e.g., 1 = mono, 2 = stereo).
     *
     * Processors use this to initialize per-channel state (filter states,
     * delay lines, envelope followers, etc.).
     */
    int numChannels = 2;
};

} // namespace dspark
