// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file AudioFile.h
 * @brief Abstract interface for reading and writing audio files.
 *
 * Defines a common contract for audio file I/O implementations (WAV, AIFF, etc.).
 * Concrete implementations handle format-specific details while exposing a
 * uniform API for loading audio into AudioBuffer and writing processed audio
 * back to disk.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h.
 *
 * @code
 *   dspark::WavFile wav;
 *   if (wav.openRead("input.wav"))
 *   {
 *       auto info = wav.getInfo();
 *       dspark::AudioBuffer<float> buffer;
 *       buffer.resize(info.numChannels, info.numSamples);
 *       wav.readSamples(buffer.toView());
 *       wav.close();
 *
 *       // Process buffer...
 *
 *       wav.openWrite("output.wav", info);
 *       wav.writeSamples(buffer.toView());
 *       wav.close();
 *   }
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"

#include <cstdint>

namespace dspark {

/**
 * @struct AudioFileInfo
 * @brief Metadata describing an audio file's format and dimensions.
 *
 * Returned by AudioFile::getInfo() after opening a file for reading.
 * Also passed to AudioFile::openWrite() to configure the output format.
 */
struct AudioFileInfo
{
    /** @brief Sample rate in Hz (e.g., 44100, 48000, 96000). */
    double sampleRate = 44100.0;

    /** @brief Number of audio channels (1 = mono, 2 = stereo). */
    int numChannels = 0;

    /** @brief Total number of sample frames in the file. */
    int64_t numSamples = 0;

    /** @brief Bits per sample in the stored format (8, 16, 24, 32, 64). */
    int bitsPerSample = 16;

    /** @brief True if the file stores floating-point samples (IEEE 754). */
    bool isFloatingPoint = false;

    /**
     * @brief Converts this file info to an AudioSpec for processor preparation.
     *
     * Uses the full file length as maxBlockSize. For streaming or chunked
     * processing, create the AudioSpec manually with a smaller block size.
     *
     * @return AudioSpec with matching sample rate, channels, and block size.
     */
    [[nodiscard]] AudioSpec toSpec() const noexcept
    {
        return {
            .sampleRate   = sampleRate,
            .maxBlockSize = static_cast<int>(numSamples),
            .numChannels  = numChannels
        };
    }
};

/**
 * @class AudioFile
 * @brief Abstract base class for audio file readers and writers.
 *
 * Concrete subclasses (WavFile, etc.) implement format-specific parsing
 * and encoding. All sample data is converted to/from the AudioBufferView<float>
 * format (normalised to [-1.0, 1.0]) regardless of the underlying file format.
 *
 * Typical usage flow:
 * 1. Call openRead() or openWrite() to open a file.
 * 2. Query getInfo() for metadata (after openRead).
 * 3. Call readSamples() or writeSamples() to transfer data.
 * 4. Call close() when done (also called by destructor).
 */
class AudioFile
{
public:
    virtual ~AudioFile() = default;

    /**
     * @brief Opens a file for reading.
     *
     * After a successful call, getInfo() returns valid metadata and
     * readSamples() can be used to load audio data.
     *
     * @param path File path (platform-dependent, null-terminated).
     * @return True if the file was opened and the header parsed successfully.
     */
    virtual bool openRead(const char* path) = 0;

    /**
     * @brief Opens a file for writing.
     *
     * Creates or overwrites the file at the given path. The provided
     * AudioFileInfo configures the output format (sample rate, channels,
     * bit depth, etc.).
     *
     * @param path File path (platform-dependent, null-terminated).
     * @param info Desired output format and metadata.
     * @return True if the file was created and the header written successfully.
     */
    virtual bool openWrite(const char* path, const AudioFileInfo& info) = 0;

    /**
     * @brief Returns metadata about the currently open file.
     *
     * Only valid after a successful openRead(). Returns a default-constructed
     * AudioFileInfo if no file is open.
     *
     * @return File metadata (sample rate, channels, sample count, bit depth).
     */
    [[nodiscard]] virtual AudioFileInfo getInfo() const = 0;

    /**
     * @brief Reads all samples from the file into the destination buffer.
     *
     * Samples are converted to float and normalised to [-1.0, 1.0].
     * The buffer must have at least as many channels and samples as
     * reported by getInfo(). Excess buffer space is left unchanged.
     *
     * @param dest Buffer to receive the audio data.
     * @return True if all samples were read successfully.
     */
    virtual bool readSamples(AudioBufferView<float> dest) = 0;

    /**
     * @brief Reads a range of sample frames from the file.
     *
     * @param dest        Buffer to receive the audio data.
     * @param startFrame  First frame to read (0-based).
     * @param numFrames   Number of frames to read.
     * @return True if the requested range was read successfully.
     */
    virtual bool readSamples(AudioBufferView<float> dest,
                             int64_t startFrame, int64_t numFrames) = 0;

    /**
     * @brief Writes samples from the source buffer to the file.
     *
     * Samples are converted from float [-1.0, 1.0] to the file's native
     * format (PCM integer or floating-point) as configured in openWrite().
     *
     * @param src Buffer containing the audio data to write.
     * @return True if all samples were written successfully.
     */
    virtual bool writeSamples(AudioBufferView<const float> src) = 0;

    /**
     * @brief Closes the file and releases resources.
     *
     * For files opened for writing, this finalises the header (e.g., updates
     * the RIFF chunk size in WAV files). Safe to call multiple times.
     */
    virtual void close() = 0;

    /** @brief Returns true if a file is currently open. */
    [[nodiscard]] virtual bool isOpen() const noexcept = 0;
};

} // namespace dspark
