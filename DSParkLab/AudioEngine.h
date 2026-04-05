// DSParkLab — Audio Engine
// File loading (WAV/MP3), real-time playback via miniaudio, effect processing.

#pragma once

#include "../DSPark.h"
#include "EffectSlot.h"

#define MINIAUDIO_IMPLEMENTATION
#include "vendor/miniaudio.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <string>
#include <vector>

namespace dsplab {

class AudioEngine
{
public:
    AudioEngine() = default;

    ~AudioEngine()
    {
        if (deviceInit_)
        {
            ma_device_uninit(&device_);
            deviceInit_ = false;
        }
    }

    // --- File loading -------------------------------------------------------

    bool loadFile(const char* path)
    {
        stop();

        std::string p(path);
        std::string ext;
        if (auto dot = p.rfind('.'); dot != std::string::npos)
            ext = p.substr(dot);
        for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        bool ok = false;

        if (ext == ".wav")
        {
            dspark::WavFile wav;
            if (wav.openRead(path))
            {
                fileInfo_ = wav.getInfo();
                fileBuffer_.resize(fileInfo_.numChannels,
                                   static_cast<int>(fileInfo_.numSamples));
                wav.readSamples(fileBuffer_.toView());
                wav.close();
                ok = true;
            }
        }
        else if (ext == ".mp3")
        {
            dspark::Mp3File mp3;
            if (mp3.openRead(path))
            {
                fileInfo_ = mp3.getInfo();
                fileBuffer_.resize(fileInfo_.numChannels,
                                   static_cast<int>(fileInfo_.numSamples));
                mp3.readSamples(fileBuffer_.toView());
                mp3.close();
                ok = true;
            }
        }

        if (!ok) return false;

        filePath_ = path;
        position_.store(0);
        fileSampleRate_ = fileInfo_.sampleRate;
        fileChannels_   = fileInfo_.numChannels;
        fileSamples_    = static_cast<int>(fileInfo_.numSamples);

        // Prepare work buffer and processors
        spec_ = { fileSampleRate_, kBlockSize, std::min(fileChannels_, 2) };
        workBuffer_.resize(spec_.numChannels, kBlockSize);

        // Prepare analysis
        spectrum_.prepare(fileSampleRate_, 4096);
        meter_.prepare(spec_);
        meter_.setAttackMs(5.0f);
        meter_.setReleaseMs(100.0f);

        // Prepare all effects for this spec and apply UI defaults
        if (effects_)
            for (auto& e : *effects_)
            {
                e.prepare(spec_);
                e.applyAllDefaults();
            }

        // (Re)init audio device at file's sample rate
        initDevice();

        return true;
    }

    // --- Transport ----------------------------------------------------------

    void play()    { playing_.store(true); }
    void pause()   { playing_.store(false); }
    void stop()    { playing_.store(false); position_.store(0); }
    void togglePlay() { playing_.store(!playing_.load()); }

    void setLooping(bool v)           { looping_.store(v); }
    void seekTo(float normPos)
    {
        int pos = static_cast<int>(normPos * static_cast<float>(fileSamples_));
        pos = std::clamp(pos, 0, fileSamples_);
        position_.store(pos);
    }

    [[nodiscard]] bool  isPlaying()  const { return playing_.load(); }
    [[nodiscard]] bool  isLooping()  const { return looping_.load(); }
    [[nodiscard]] bool  hasFile()    const { return fileSamples_ > 0; }
    [[nodiscard]] float getPosition() const
    {
        if (fileSamples_ <= 0) return 0.0f;
        return static_cast<float>(position_.load()) / static_cast<float>(fileSamples_);
    }
    [[nodiscard]] float getDurationSeconds() const
    {
        if (fileSampleRate_ <= 0) return 0.0f;
        return static_cast<float>(fileSamples_) / static_cast<float>(fileSampleRate_);
    }
    [[nodiscard]] const std::string& getFilePath() const { return filePath_; }
    [[nodiscard]] double getSampleRate() const { return fileSampleRate_; }
    [[nodiscard]] int    getChannels()   const { return fileChannels_; }
    [[nodiscard]] const dspark::AudioSpec& getSpec() const { return spec_; }

    // --- Effect chain -------------------------------------------------------

    void setEffects(std::vector<EffectSlot>* effects) { effects_ = effects; }

    // --- Bypass (A/B) -------------------------------------------------------

    void setBypass(bool v) { bypass_.store(v); }
    [[nodiscard]] bool isBypassed() const { return bypass_.load(); }

    // --- Visualization data (read from GUI thread) --------------------------

    [[nodiscard]] float getPeakL() const { return peakL_.load(); }
    [[nodiscard]] float getPeakR() const { return peakR_.load(); }

    [[nodiscard]] const float* getSpectrumDb() const
    {
        return spectrum_.getMagnitudesDb();
    }
    [[nodiscard]] int getSpectrumBins() const
    {
        return spectrum_.getNumBins();
    }
    [[nodiscard]] float binToFrequency(int k) const
    {
        return spectrum_.binToFrequency(k);
    }

    // Waveform snapshot (last processed block)
    [[nodiscard]] const float* getWaveformL() const { return waveSnap_[0].data(); }
    [[nodiscard]] const float* getWaveformR() const { return waveSnap_[1].data(); }
    [[nodiscard]] int getWaveformSize() const { return waveSnapSize_.load(); }

private:
    static constexpr int kBlockSize = 512;

    // --- Audio callback (runs on audio thread) ------------------------------

    static void audioCallback(ma_device* device, void* output, const void*, ma_uint32 frameCount)
    {
        auto* self = static_cast<AudioEngine*>(device->pUserData);
        auto* out  = static_cast<float*>(output);
        const int frames = static_cast<int>(frameCount);
        const int outCh  = static_cast<int>(device->playback.channels);

        if (!self->playing_.load() || self->fileSamples_ <= 0)
        {
            std::memset(out, 0, sizeof(float) * frames * outCh);
            return;
        }

        int pos = self->position_.load();
        const int nCh = std::min(self->fileChannels_, 2);
        int written = 0;

        while (written < frames)
        {
            int remaining = self->fileSamples_ - pos;
            if (remaining <= 0)
            {
                if (self->looping_.load())
                    pos = 0;
                else
                {
                    // Fill rest with silence
                    std::memset(out + written * outCh, 0,
                                sizeof(float) * (frames - written) * outCh);
                    self->playing_.store(false);
                    break;
                }
                remaining = self->fileSamples_ - pos;
            }

            const int chunk = std::min(frames - written, std::min(remaining, self->kBlockSize));

            // Copy from source to work buffer
            for (int ch = 0; ch < nCh; ++ch)
            {
                const float* src = self->fileBuffer_.getChannel(ch) + pos;
                float* dst = self->workBuffer_.getChannel(ch);
                std::copy(src, src + chunk, dst);
            }

            // Apply effects (unless bypassed)
            if (!self->bypass_.load() && self->effects_)
            {
                auto view = self->workBuffer_.toView().getSubView(0, chunk);
                for (auto& e : *self->effects_)
                    e.process(view);
            }

            // Feed analysis
            self->spectrum_.pushSamples(self->workBuffer_.getChannel(0), chunk);
            auto meterView = self->workBuffer_.toView().getSubView(0, chunk);
            self->meter_.process(meterView);

            // Interleave to output
            for (int i = 0; i < chunk; ++i)
            {
                for (int c = 0; c < outCh; ++c)
                {
                    int srcCh = std::min(c, nCh - 1);
                    out[(written + i) * outCh + c] = self->workBuffer_.getChannel(srcCh)[i];
                }
            }

            // Waveform snapshot
            if (chunk > 0)
            {
                int snapN = std::min(chunk, static_cast<int>(self->waveSnap_[0].size()));
                for (int i = 0; i < snapN; ++i)
                {
                    self->waveSnap_[0][i] = self->workBuffer_.getChannel(0)[i];
                    if (nCh > 1)
                        self->waveSnap_[1][i] = self->workBuffer_.getChannel(1)[i];
                    else
                        self->waveSnap_[1][i] = self->waveSnap_[0][i];
                }
                self->waveSnapSize_.store(snapN);
            }

            pos += chunk;
            written += chunk;
        }

        self->position_.store(pos);

        // Update peak meters
        self->peakL_.store(self->meter_.getPeakLevel(0));
        if (nCh > 1)
            self->peakR_.store(self->meter_.getPeakLevel(1));
        else
            self->peakR_.store(self->peakL_.load());
    }

    // --- Device init --------------------------------------------------------

    void initDevice()
    {
        if (deviceInit_)
        {
            ma_device_uninit(&device_);
            deviceInit_ = false;
        }

        ma_device_config config = ma_device_config_init(ma_device_type_playback);
        config.playback.format   = ma_format_f32;
        config.playback.channels = static_cast<ma_uint32>(std::min(fileChannels_, 2));
        config.sampleRate        = static_cast<ma_uint32>(fileSampleRate_);
        config.dataCallback      = audioCallback;
        config.pUserData         = this;
        config.periodSizeInFrames = kBlockSize;

        if (ma_device_init(nullptr, &config, &device_) == MA_SUCCESS)
        {
            ma_device_start(&device_);
            deviceInit_ = true;
        }
    }

    // --- State --------------------------------------------------------------

    std::string              filePath_;
    dspark::AudioFileInfo    fileInfo_ {};
    dspark::AudioBuffer<float> fileBuffer_;
    dspark::AudioBuffer<float> workBuffer_;
    dspark::AudioSpec        spec_ {};

    double fileSampleRate_ = 0;
    int    fileChannels_   = 0;
    int    fileSamples_    = 0;

    std::atomic<int>  position_ { 0 };
    std::atomic<bool> playing_  { false };
    std::atomic<bool> looping_  { true };
    std::atomic<bool> bypass_   { false };

    // Analysis
    dspark::SpectrumAnalyzer<float> spectrum_;
    dspark::LevelFollower<float>    meter_;
    std::atomic<float> peakL_ { 0.0f };
    std::atomic<float> peakR_ { 0.0f };

    // Waveform snapshot
    std::array<std::vector<float>, 2> waveSnap_ = {
        std::vector<float>(kBlockSize, 0.0f),
        std::vector<float>(kBlockSize, 0.0f)
    };
    std::atomic<int> waveSnapSize_ { 0 };

    // Effects
    std::vector<EffectSlot>* effects_ = nullptr;

    // miniaudio
    ma_device device_ {};
    bool      deviceInit_ = false;
};

} // namespace dsplab
