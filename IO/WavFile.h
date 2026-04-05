// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file WavFile.h
 * @brief Pure C++ WAV file reader/writer (RIFF format).
 *
 * Supports reading and writing standard WAV files with the following formats:
 * - **PCM integer**: 8-bit unsigned, 16-bit, 24-bit, 32-bit signed
 * - **IEEE float**: 32-bit and 64-bit floating-point
 *
 * All sample data is normalised to [-1.0, 1.0] float on read, and converted
 * from [-1.0, 1.0] float to the target format on write. Multi-channel files
 * are deinterleaved on read and interleaved on write.
 *
 * This implementation handles:
 * - Standard RIFF/WAVE format with 'fmt ' and 'data' chunks
 * - Extensible format (WAVE_FORMAT_EXTENSIBLE) for >2 channels or >16-bit PCM
 * - Skipping unknown chunks gracefully
 * - Finalising the RIFF and data chunk sizes on close (streaming writes)
 *
 * Dependencies: AudioFile.h (base class), standard C++ library only.
 *
 * @code
 *   // Reading a WAV file:
 *   dspark::WavFile wav;
 *   if (wav.openRead("input.wav"))
 *   {
 *       auto info = wav.getInfo();
 *       dspark::AudioBuffer<float> buffer;
 *       buffer.resize(info.numChannels, static_cast<int>(info.numSamples));
 *       wav.readSamples(buffer.toView());
 *       wav.close();
 *   }
 *
 *   // Writing a WAV file:
 *   dspark::AudioFileInfo outInfo;
 *   outInfo.sampleRate     = 48000.0;
 *   outInfo.numChannels    = 2;
 *   outInfo.bitsPerSample  = 24;
 *   outInfo.isFloatingPoint = false;
 *
 *   dspark::WavFile writer;
 *   if (writer.openWrite("output.wav", outInfo))
 *   {
 *       writer.writeSamples(processedBuffer.toView());
 *       writer.close();
 *   }
 * @endcode
 */

#include "AudioFile.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

namespace dspark {

/**
 * @class WavFile
 * @brief Complete WAV file reader and writer in pure C++.
 *
 * Inherits from AudioFile to provide a uniform interface. Handles both
 * standard PCM and IEEE float formats, mono through multi-channel.
 */
class WavFile : public AudioFile
{
public:
    ~WavFile() override { close(); }

    // -- AudioFile interface ---------------------------------------------------

    bool openRead(const char* path) override
    {
        close();

        file_.open(path, std::ios::binary | std::ios::in);
        if (!file_.is_open()) return false;

        if (!parseHeader())
        {
            close();
            return false;
        }

        mode_ = Mode::Read;
        return true;
    }

    bool openWrite(const char* path, const AudioFileInfo& info) override
    {
        close();

        file_.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!file_.is_open()) return false;

        info_ = info;
        mode_ = Mode::Write;
        totalFramesWritten_ = 0;

        // Write placeholder header (sizes will be patched on close)
        if (!writeHeader())
        {
            close();
            return false;
        }

        return true;
    }

    [[nodiscard]] AudioFileInfo getInfo() const override { return info_; }

    bool readSamples(AudioBufferView<float> dest) override
    {
        return readSamples(dest, 0, info_.numSamples);
    }

    bool readSamples(AudioBufferView<float> dest,
                     int64_t startFrame, int64_t numFrames) override
    {
        if (mode_ != Mode::Read || !file_.is_open()) return false;
        if (startFrame < 0 || numFrames <= 0) return false;
        if (startFrame + numFrames > info_.numSamples) return false;

        const int bytesPerSample = info_.bitsPerSample / 8;
        const int frameSize = bytesPerSample * info_.numChannels;

        // Seek to the start frame
        auto seekPos = dataChunkOffset_ + static_cast<std::streamoff>(startFrame * frameSize);
        file_.seekg(seekPos, std::ios::beg);
        if (!file_.good()) return false;

        // Read in chunks to avoid huge allocations for very large files
        constexpr int64_t kChunkFrames = 8192;
        const int nCh = std::min(dest.getNumChannels(), info_.numChannels);
        int64_t framesRemaining = std::min(numFrames,
                                           static_cast<int64_t>(dest.getNumSamples()));

        std::vector<uint8_t> rawBuffer(
            static_cast<size_t>(kChunkFrames) * static_cast<size_t>(frameSize));

        int64_t destOffset = 0;
        while (framesRemaining > 0)
        {
            const auto toRead = std::min(framesRemaining, kChunkFrames);
            const auto rawBytes = static_cast<std::streamsize>(toRead * frameSize);

            file_.read(reinterpret_cast<char*>(rawBuffer.data()), rawBytes);
            if (file_.gcount() != rawBytes) return false;

            // Deinterleave and convert to float
            deinterleave(rawBuffer.data(), dest, nCh,
                         static_cast<int>(toRead), static_cast<int>(destOffset));

            destOffset += toRead;
            framesRemaining -= toRead;
        }

        return true;
    }

    bool writeSamples(AudioBufferView<const float> src) override
    {
        if (mode_ != Mode::Write || !file_.is_open()) return false;

        const int nCh = info_.numChannels;
        const int nS  = src.getNumSamples();
        const int bytesPerSample = info_.bitsPerSample / 8;
        const int frameSize = bytesPerSample * nCh;

        // Interleave and convert in chunks
        constexpr int kChunkFrames = 8192;
        std::vector<uint8_t> rawBuffer(
            static_cast<size_t>(kChunkFrames) * static_cast<size_t>(frameSize));

        int framesRemaining = nS;
        int srcOffset = 0;

        while (framesRemaining > 0)
        {
            const int toWrite = std::min(framesRemaining, kChunkFrames);

            interleave(src, rawBuffer.data(), nCh, toWrite, srcOffset);

            auto rawBytes = static_cast<std::streamsize>(
                static_cast<int64_t>(toWrite) * frameSize);
            file_.write(reinterpret_cast<const char*>(rawBuffer.data()), rawBytes);
            if (!file_.good()) return false;

            srcOffset += toWrite;
            framesRemaining -= toWrite;
        }

        totalFramesWritten_ += nS;
        return true;
    }

    void close() override
    {
        if (!file_.is_open()) { mode_ = Mode::Closed; return; }

        if (mode_ == Mode::Write)
            finaliseHeader();

        file_.close();
        mode_ = Mode::Closed;
    }

    [[nodiscard]] bool isOpen() const noexcept override
    {
        return file_.is_open() && mode_ != Mode::Closed;
    }

private:
    // -- RIFF/WAV constants ----------------------------------------------------

    static constexpr uint16_t kFormatPCM        = 1;
    static constexpr uint16_t kFormatIEEEFloat   = 3;
    static constexpr uint16_t kFormatExtensible  = 0xFFFE;

    enum class Mode { Closed, Read, Write };

    // -- Header parsing (read) -------------------------------------------------

    bool parseHeader()
    {
        // RIFF header
        char riffId[4];
        if (!readBytes(riffId, 4)) return false;
        if (std::memcmp(riffId, "RIFF", 4) != 0) return false;

        uint32_t riffSize = 0;
        if (!readLE32(riffSize)) return false;
        (void)riffSize; // not validated — some files have incorrect sizes

        char waveId[4];
        if (!readBytes(waveId, 4)) return false;
        if (std::memcmp(waveId, "WAVE", 4) != 0) return false;

        // Parse chunks until we have both 'fmt ' and 'data'
        bool hasFmt = false, hasData = false;

        while (file_.good() && !(hasFmt && hasData))
        {
            char chunkId[4];
            uint32_t chunkSize = 0;
            if (!readBytes(chunkId, 4)) break;
            if (!readLE32(chunkSize)) break;

            if (std::memcmp(chunkId, "fmt ", 4) == 0)
            {
                if (!parseFmtChunk(chunkSize)) return false;
                hasFmt = true;
            }
            else if (std::memcmp(chunkId, "data", 4) == 0)
            {
                // Validate: chunk cannot be larger than remaining file
                auto currentPos = file_.tellg();
                file_.seekg(0, std::ios::end);
                auto fileEnd = file_.tellg();
                file_.seekg(currentPos);
                auto remaining = fileEnd - currentPos;
                if (static_cast<std::streamoff>(chunkSize) > remaining)
                    chunkSize = static_cast<uint32_t>(remaining);

                dataChunkOffset_ = file_.tellg();
                dataChunkSize_ = chunkSize;

                // Calculate number of sample frames
                const int bytesPerFrame = (info_.bitsPerSample / 8) * info_.numChannels;
                if (bytesPerFrame > 0)
                    info_.numSamples = static_cast<int64_t>(chunkSize) / bytesPerFrame;

                hasData = true;

                // Don't seek past data — we'll read from here
                if (!hasFmt)
                {
                    // Need to skip data to find fmt (unusual but valid)
                    file_.seekg(static_cast<std::streamoff>(chunkSize), std::ios::cur);
                }
            }
            else
            {
                // Skip unknown chunk (pad to even boundary)
                auto skip = static_cast<std::streamoff>(chunkSize + (chunkSize & 1));
                file_.seekg(skip, std::ios::cur);
            }
        }

        return hasFmt && hasData;
    }

    bool parseFmtChunk(uint32_t chunkSize)
    {
        if (chunkSize < 16) return false;

        uint16_t audioFormat = 0;
        uint16_t numChannels = 0;
        uint32_t sampleRate = 0;
        uint32_t byteRate = 0;
        uint16_t blockAlign = 0;
        uint16_t bitsPerSample = 0;

        if (!readLE16(audioFormat)) return false;
        if (!readLE16(numChannels)) return false;
        if (!readLE32(sampleRate)) return false;
        if (!readLE32(byteRate)) return false;
        (void)byteRate;
        if (!readLE16(blockAlign)) return false;
        (void)blockAlign;
        if (!readLE16(bitsPerSample)) return false;

        // Handle WAVE_FORMAT_EXTENSIBLE
        if (audioFormat == kFormatExtensible && chunkSize >= 40)
        {
            uint16_t cbSize = 0;
            readLE16(cbSize);

            uint16_t validBitsPerSample = 0;
            readLE16(validBitsPerSample);

            uint32_t channelMask = 0;
            readLE32(channelMask);
            (void)channelMask;

            // Read the sub-format GUID (first 2 bytes are the actual format tag)
            uint16_t subFormat = 0;
            readLE16(subFormat);
            audioFormat = subFormat;

            // Skip remaining GUID bytes (14 bytes)
            file_.seekg(14, std::ios::cur);

            if (validBitsPerSample > 0)
                bitsPerSample = validBitsPerSample;
        }
        else if (chunkSize > 16)
        {
            // Skip extra fmt bytes
            auto skip = static_cast<std::streamoff>(chunkSize - 16);
            file_.seekg(skip, std::ios::cur);
        }

        info_.sampleRate      = static_cast<double>(sampleRate);
        info_.numChannels     = static_cast<int>(numChannels);
        info_.bitsPerSample   = static_cast<int>(bitsPerSample);
        info_.isFloatingPoint = (audioFormat == kFormatIEEEFloat);

        // Validate
        if (audioFormat != kFormatPCM && audioFormat != kFormatIEEEFloat)
            return false;
        if (numChannels == 0 || numChannels > 64) return false;
        if (bitsPerSample != 8 && bitsPerSample != 16 &&
            bitsPerSample != 24 && bitsPerSample != 32 &&
            bitsPerSample != 64)
            return false;
        if (info_.isFloatingPoint && bitsPerSample != 32 && bitsPerSample != 64)
            return false;

        return true;
    }

    // -- Header writing --------------------------------------------------------

    bool writeHeader()
    {
        // We'll write a placeholder header and patch sizes on close.
        // For float formats, use WAVE_FORMAT_EXTENSIBLE if >2 channels
        // or WAVE_FORMAT_IEEE_FLOAT for 1-2 channels.

        const uint16_t formatTag = getWriteFormatTag();
        const bool extensible = (formatTag == kFormatExtensible);
        const uint32_t fmtChunkSize = extensible ? 40u : 16u;

        const int bytesPerSample = info_.bitsPerSample / 8;
        const auto blockAlign = static_cast<uint16_t>(info_.numChannels * bytesPerSample);
        const auto byteRate = static_cast<uint32_t>(
            static_cast<int>(info_.sampleRate) * blockAlign);

        // RIFF header (placeholder size)
        writeBytes("RIFF", 4);
        writeLE32(0); // Placeholder — patched on close
        writeBytes("WAVE", 4);

        // fmt chunk
        writeBytes("fmt ", 4);
        writeLE32(fmtChunkSize);
        writeLE16(formatTag);
        writeLE16(static_cast<uint16_t>(info_.numChannels));
        writeLE32(static_cast<uint32_t>(info_.sampleRate));
        writeLE32(byteRate);
        writeLE16(blockAlign);
        writeLE16(static_cast<uint16_t>(info_.bitsPerSample));

        if (extensible)
        {
            writeLE16(22); // cbSize
            writeLE16(static_cast<uint16_t>(info_.bitsPerSample)); // validBitsPerSample
            writeLE32(0); // channelMask (default)

            // SubFormat GUID: first 2 bytes = actual format, rest = standard GUID suffix
            uint16_t subFormat = info_.isFloatingPoint ? kFormatIEEEFloat : kFormatPCM;
            writeLE16(subFormat);
            // Standard Microsoft GUID suffix: 00 00 00 00 10 00 80 00 00 AA 00 38 9B 71
            static constexpr uint8_t kGuidSuffix[14] = {
                0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00,
                0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71
            };
            writeBytes(reinterpret_cast<const char*>(kGuidSuffix), 14);
        }

        // data chunk header (placeholder size)
        writeBytes("data", 4);
        dataChunkSizeOffset_ = file_.tellp();
        writeLE32(0); // Placeholder — patched on close

        dataChunkOffset_ = file_.tellp();
        return file_.good();
    }

    void finaliseHeader()
    {
        if (!file_.is_open()) return;

        const int bytesPerSample = info_.bitsPerSample / 8;
        const auto dataSize = static_cast<uint32_t>(
            totalFramesWritten_ * info_.numChannels * bytesPerSample);

        // Patch data chunk size
        file_.seekp(dataChunkSizeOffset_, std::ios::beg);
        writeLE32(dataSize);

        // Patch RIFF chunk size (total file size - 8)
        file_.seekp(0, std::ios::end);
        auto fileSize = static_cast<uint32_t>(file_.tellp());
        uint32_t riffSize = fileSize - 8;
        file_.seekp(4, std::ios::beg);
        writeLE32(riffSize);

        file_.flush();
    }

    [[nodiscard]] uint16_t getWriteFormatTag() const noexcept
    {
        if (info_.numChannels > 2 || info_.bitsPerSample > 16)
            return kFormatExtensible;
        if (info_.isFloatingPoint)
            return kFormatIEEEFloat;
        return kFormatPCM;
    }

    // -- Sample conversion (read: raw → float) ---------------------------------

    void deinterleave(const uint8_t* raw, AudioBufferView<float>& dest,
                      int nCh, int numFrames, int destOffset) const
    {
        const int bytesPerSample = info_.bitsPerSample / 8;

        for (int frame = 0; frame < numFrames; ++frame)
        {
            for (int ch = 0; ch < nCh; ++ch)
            {
                const uint8_t* samplePtr =
                    raw + static_cast<size_t>(frame * info_.numChannels + ch) * bytesPerSample;

                float value = rawToFloat(samplePtr);
                dest.getChannel(ch)[destOffset + frame] = value;
            }

            // Skip extra channels in file that don't fit in dest
            // (handled implicitly by only writing to nCh channels)
        }
    }

    [[nodiscard]] float rawToFloat(const uint8_t* ptr) const noexcept
    {
        if (info_.isFloatingPoint)
        {
            if (info_.bitsPerSample == 32)
            {
                float v = 0.0f;
                std::memcpy(&v, ptr, 4);
                return v;
            }
            else // 64-bit
            {
                double v = 0.0;
                std::memcpy(&v, ptr, 8);
                return static_cast<float>(v);
            }
        }

        switch (info_.bitsPerSample)
        {
            case 8:
            {
                // 8-bit WAV is unsigned: 0–255, 128 = silence
                return (static_cast<float>(ptr[0]) - 128.0f) / 128.0f;
            }
            case 16:
            {
                auto val = static_cast<int16_t>(
                    static_cast<uint16_t>(ptr[0]) |
                    (static_cast<uint16_t>(ptr[1]) << 8));
                return static_cast<float>(val) / 32768.0f;
            }
            case 24:
            {
                // Sign-extend 24-bit to 32-bit
                int32_t val = static_cast<int32_t>(ptr[0])
                            | (static_cast<int32_t>(ptr[1]) << 8)
                            | (static_cast<int32_t>(ptr[2]) << 16);
                if (val & 0x800000) val |= static_cast<int32_t>(0xFF000000u);
                return static_cast<float>(val) / 8388608.0f;
            }
            case 32:
            {
                // Byte-by-byte LE reconstruction (portable, no endianness assumption)
                int32_t val = static_cast<int32_t>(ptr[0])
                            | (static_cast<int32_t>(ptr[1]) << 8)
                            | (static_cast<int32_t>(ptr[2]) << 16)
                            | (static_cast<int32_t>(ptr[3]) << 24);
                return static_cast<float>(val) / 2147483648.0f;
            }
            default:
                return 0.0f;
        }
    }

    // -- Sample conversion (write: float → raw) --------------------------------

    void interleave(AudioBufferView<const float> src, uint8_t* raw,
                    int nCh, int numFrames, int srcOffset) const
    {
        const int bytesPerSample = info_.bitsPerSample / 8;

        for (int frame = 0; frame < numFrames; ++frame)
        {
            for (int ch = 0; ch < nCh; ++ch)
            {
                float value = src.getChannel(ch)[srcOffset + frame];
                uint8_t* samplePtr =
                    raw + static_cast<size_t>(frame * nCh + ch) * bytesPerSample;

                floatToRaw(value, samplePtr);
            }
        }
    }

    void floatToRaw(float value, uint8_t* ptr) const noexcept
    {
        if (info_.isFloatingPoint)
        {
            if (info_.bitsPerSample == 32)
            {
                std::memcpy(ptr, &value, 4);
                return;
            }
            else // 64-bit
            {
                auto d = static_cast<double>(value);
                std::memcpy(ptr, &d, 8);
                return;
            }
        }

        // Clamp to [-1, 1] before converting
        value = std::clamp(value, -1.0f, 1.0f);

        switch (info_.bitsPerSample)
        {
            case 8:
            {
                auto val = static_cast<uint8_t>(
                    static_cast<int>(value * 127.0f) + 128);
                ptr[0] = val;
                break;
            }
            case 16:
            {
                auto val = static_cast<int16_t>(value * 32767.0f);
                ptr[0] = static_cast<uint8_t>(val & 0xFF);
                ptr[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
                break;
            }
            case 24:
            {
                auto val = static_cast<int32_t>(value * 8388607.0f);
                ptr[0] = static_cast<uint8_t>(val & 0xFF);
                ptr[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
                ptr[2] = static_cast<uint8_t>((val >> 16) & 0xFF);
                break;
            }
            case 32:
            {
                // Byte-by-byte LE write (portable, no endianness assumption)
                auto val = static_cast<int32_t>(
                    static_cast<double>(value) * 2147483647.0);
                ptr[0] = static_cast<uint8_t>(val & 0xFF);
                ptr[1] = static_cast<uint8_t>((val >> 8) & 0xFF);
                ptr[2] = static_cast<uint8_t>((val >> 16) & 0xFF);
                ptr[3] = static_cast<uint8_t>((val >> 24) & 0xFF);
                break;
            }
            default:
                break;
        }
    }

    // -- Low-level I/O helpers (little-endian) ---------------------------------

    bool readBytes(char* buf, int n)
    {
        file_.read(buf, n);
        return file_.gcount() == n;
    }

    bool readLE16(uint16_t& val)
    {
        uint8_t b[2];
        file_.read(reinterpret_cast<char*>(b), 2);
        if (file_.gcount() != 2) return false;
        val = static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
        return true;
    }

    bool readLE32(uint32_t& val)
    {
        uint8_t b[4];
        file_.read(reinterpret_cast<char*>(b), 4);
        if (file_.gcount() != 4) return false;
        val = static_cast<uint32_t>(b[0])
            | (static_cast<uint32_t>(b[1]) << 8)
            | (static_cast<uint32_t>(b[2]) << 16)
            | (static_cast<uint32_t>(b[3]) << 24);
        return true;
    }

    void writeBytes(const char* buf, int n)
    {
        file_.write(buf, n);
    }

    void writeLE16(uint16_t val)
    {
        uint8_t b[2] = {
            static_cast<uint8_t>(val & 0xFF),
            static_cast<uint8_t>((val >> 8) & 0xFF)
        };
        file_.write(reinterpret_cast<const char*>(b), 2);
    }

    void writeLE32(uint32_t val)
    {
        uint8_t b[4] = {
            static_cast<uint8_t>(val & 0xFF),
            static_cast<uint8_t>((val >> 8) & 0xFF),
            static_cast<uint8_t>((val >> 16) & 0xFF),
            static_cast<uint8_t>((val >> 24) & 0xFF)
        };
        file_.write(reinterpret_cast<const char*>(b), 4);
    }

    // -- Members ---------------------------------------------------------------

    std::fstream file_;
    Mode mode_ = Mode::Closed;
    AudioFileInfo info_ {};

    /** @brief File offset where the 'data' chunk's sample data begins. */
    std::streamoff dataChunkOffset_ = 0;

    /** @brief File offset where the 'data' chunk's size field is located (for patching). */
    std::streamoff dataChunkSizeOffset_ = 0;

    /** @brief Size of the data chunk in bytes (from the header, on read). */
    uint32_t dataChunkSize_ = 0;

    /** @brief Total frames written so far (for finalising the header on close). */
    int64_t totalFramesWritten_ = 0;
};

} // namespace dspark
