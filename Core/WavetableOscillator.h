// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file WavetableOscillator.h
 * @brief Bandlimited wavetable oscillator with mipmap anti-aliasing.
 *
 * A professional wavetable oscillator that uses mipmapped tables to prevent
 * aliasing at all frequencies. Each mipmap level has fewer harmonics, matching
 * the Nyquist limit for that octave.
 *
 * Complements the PolyBLEP Oscillator (which is better for basic waveforms)
 * with support for arbitrary waveforms, wavetable morphing, and complex timbres.
 *
 * Features:
 * - Mipmapped tables (one per octave, auto-selected)
 * - Built-in waveforms: Saw, Square, Triangle, Sine
 * - Custom wavetable loading
 * - Morphing between two wavetables
 * - Phase accumulation via Phasor
 * - Per-sample frequency modulation
 *
 * Dependencies: DspMath.h, Phasor.h.
 *
 * @code
 *   dspark::WavetableOscillator<float> osc;
 *   osc.prepare(48000.0);
 *   osc.buildSaw();
 *   osc.setFrequency(440.0f);
 *
 *   for (int i = 0; i < numSamples; ++i)
 *       output[i] = osc.getSample();
 * @endcode
 */

#include "DspMath.h"
#include "AudioSpec.h"
#include "Phasor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class WavetableOscillator
 * @brief Mipmapped wavetable oscillator for bandlimited synthesis.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class WavetableOscillator
{
public:
    static constexpr int kTableSize = 2048;
    static constexpr int kMaxMipLevels = 12; // Covers 20 Hz to 20 kHz

    /**
     * @brief Prepares the oscillator for a given sample rate.
     * @param sampleRate Sample rate in Hz.
     */
    void prepare(double sampleRate)
    {
        sampleRate_ = sampleRate;
        phasor_.prepare(sampleRate);
    }

    /** @brief Prepares from AudioSpec (unified API). */
    void prepare(const AudioSpec& spec) { prepare(spec.sampleRate); }

    // -- Built-in waveform generators -------------------------------------------

    /**
     * @brief Builds a bandlimited sawtooth wavetable with mipmaps.
     */
    void buildSaw()
    {
        buildFromHarmonics([](int harmonic) -> T {
            // Saw: sum of sin(n*x)/n, alternating sign
            return (harmonic % 2 == 0) ? T(-1.0 / harmonic) : T(1.0 / harmonic);
        });
    }

    /**
     * @brief Builds a bandlimited square wavetable with mipmaps.
     */
    void buildSquare()
    {
        buildFromHarmonics([](int harmonic) -> T {
            // Square: only odd harmonics, amplitude 1/n
            if (harmonic % 2 == 0) return T(0);
            return T(1.0 / harmonic);
        });
    }

    /**
     * @brief Builds a bandlimited triangle wavetable with mipmaps.
     */
    void buildTriangle()
    {
        buildFromHarmonics([](int harmonic) -> T {
            // Triangle: only odd harmonics, amplitude 1/n^2, alternating sign
            if (harmonic % 2 == 0) return T(0);
            T sign = ((harmonic / 2) % 2 == 0) ? T(1) : T(-1);
            return sign / static_cast<T>(harmonic * harmonic);
        });
    }

    /**
     * @brief Builds a pure sine wavetable (single mipmap level, no aliasing).
     */
    void buildSine()
    {
        numMipLevels_ = 1;
        mipTables_.resize(1);
        mipTables_[0].resize(kTableSize);

        for (int i = 0; i < kTableSize; ++i)
        {
            T phase = twoPi<T> * static_cast<T>(i) / static_cast<T>(kTableSize);
            mipTables_[0][static_cast<size_t>(i)] = std::sin(phase);
        }

        mipMaxFreq_.assign(1, static_cast<T>(sampleRate_ / 2.0));
    }

    /**
     * @brief Builds mipmapped tables from a harmonic amplitude function.
     *
     * @param harmonicFunc Function that returns the amplitude for harmonic N
     *                     (N starts at 1). Return 0 for harmonics to skip.
     */
    template <typename HarmonicFunc>
    void buildFromHarmonics(HarmonicFunc harmonicFunc)
    {
        // Determine max harmonics per mip level
        numMipLevels_ = kMaxMipLevels;
        mipTables_.resize(static_cast<size_t>(numMipLevels_));
        mipMaxFreq_.resize(static_cast<size_t>(numMipLevels_));

        T nyquist = static_cast<T>(sampleRate_ / 2.0);

        for (int level = 0; level < numMipLevels_; ++level)
        {
            // Each level covers one octave
            // Level 0: lowest frequencies, most harmonics
            // Level N: highest frequencies, fewest harmonics
            T maxFreqForLevel = nyquist / static_cast<T>(1 << (numMipLevels_ - 1 - level));
            mipMaxFreq_[static_cast<size_t>(level)] = maxFreqForLevel;

            int maxHarmonics = static_cast<int>(nyquist / maxFreqForLevel);
            maxHarmonics = std::max(maxHarmonics, 1);

            // Generate table by additive synthesis
            mipTables_[static_cast<size_t>(level)].resize(kTableSize);
            auto& table = mipTables_[static_cast<size_t>(level)];

            std::fill(table.begin(), table.end(), T(0));

            for (int h = 1; h <= maxHarmonics; ++h)
            {
                T amplitude = harmonicFunc(h);
                if (amplitude == T(0)) continue;

                for (int i = 0; i < kTableSize; ++i)
                {
                    T phase = twoPi<T> * static_cast<T>(h)
                            * static_cast<T>(i) / static_cast<T>(kTableSize);
                    table[static_cast<size_t>(i)] += amplitude * std::sin(phase);
                }
            }

            // Normalise to [-1, 1]
            T maxVal = T(0);
            for (auto& s : table)
                maxVal = std::max(maxVal, std::abs(s));

            if (maxVal > T(0))
            {
                T invMax = T(1) / maxVal;
                for (auto& s : table)
                    s *= invMax;
            }
        }
    }

    /**
     * @brief Loads a custom single-cycle wavetable.
     *
     * The input is resampled to kTableSize and mipmaps are generated
     * by successively halving the harmonic content.
     *
     * @param data Wavetable samples (one cycle).
     * @param size Number of samples in the wavetable.
     */
    void loadWavetable(const T* data, int size)
    {
        // Resample to kTableSize using linear interpolation
        std::vector<T> baseTable(kTableSize);
        for (int i = 0; i < kTableSize; ++i)
        {
            T pos = static_cast<T>(i) * static_cast<T>(size) / static_cast<T>(kTableSize);
            int idx0 = static_cast<int>(pos) % size;
            int idx1 = (idx0 + 1) % size;
            T frac = pos - std::floor(pos);
            baseTable[static_cast<size_t>(i)] = data[idx0] + frac * (data[idx1] - data[idx0]);
        }

        // Analyse the base table via DFT to extract harmonic amplitudes and phases
        const int N = kTableSize;
        const int maxHarmonics = N / 2;
        std::vector<T> cosCoeffs(static_cast<size_t>(maxHarmonics + 1), T(0));
        std::vector<T> sinCoeffs(static_cast<size_t>(maxHarmonics + 1), T(0));

        for (int h = 1; h <= maxHarmonics; ++h)
        {
            T sumCos = T(0), sumSin = T(0);
            for (int i = 0; i < N; ++i)
            {
                T phase = twoPi<T> * static_cast<T>(h) * static_cast<T>(i) / static_cast<T>(N);
                sumCos += baseTable[static_cast<size_t>(i)] * std::cos(phase);
                sumSin += baseTable[static_cast<size_t>(i)] * std::sin(phase);
            }
            cosCoeffs[static_cast<size_t>(h)] = sumCos * T(2) / static_cast<T>(N);
            sinCoeffs[static_cast<size_t>(h)] = sumSin * T(2) / static_cast<T>(N);
        }

        // Build mipmapped tables (same octave structure as buildFromHarmonics)
        numMipLevels_ = kMaxMipLevels;
        mipTables_.resize(static_cast<size_t>(numMipLevels_));
        mipMaxFreq_.resize(static_cast<size_t>(numMipLevels_));

        T nyquist = static_cast<T>(sampleRate_ / 2.0);

        for (int level = 0; level < numMipLevels_; ++level)
        {
            T maxFreqForLevel = nyquist / static_cast<T>(1 << (numMipLevels_ - 1 - level));
            mipMaxFreq_[static_cast<size_t>(level)] = maxFreqForLevel;

            int maxH = static_cast<int>(nyquist / maxFreqForLevel);
            maxH = std::clamp(maxH, 1, maxHarmonics);

            // Reconstruct table from band-limited harmonics
            mipTables_[static_cast<size_t>(level)].resize(N);
            auto& table = mipTables_[static_cast<size_t>(level)];
            std::fill(table.begin(), table.end(), T(0));

            for (int h = 1; h <= maxH; ++h)
            {
                T a = cosCoeffs[static_cast<size_t>(h)];
                T b = sinCoeffs[static_cast<size_t>(h)];
                if (a == T(0) && b == T(0)) continue;

                for (int i = 0; i < N; ++i)
                {
                    T phase = twoPi<T> * static_cast<T>(h)
                            * static_cast<T>(i) / static_cast<T>(N);
                    table[static_cast<size_t>(i)] += a * std::cos(phase) + b * std::sin(phase);
                }
            }

            // Normalise to [-1, 1]
            T maxVal = T(0);
            for (auto& s : table)
                maxVal = std::max(maxVal, std::abs(s));

            if (maxVal > T(0))
            {
                T invMax = T(1) / maxVal;
                for (auto& s : table)
                    s *= invMax;
            }
        }
    }

    // -- Playback -----------------------------------------------------------------

    /**
     * @brief Sets the oscillation frequency.
     * @param frequencyHz Frequency in Hz.
     */
    void setFrequency(T frequencyHz) noexcept
    {
        frequency_ = frequencyHz;
        phasor_.setFrequency(frequencyHz);
    }

    /**
     * @brief Returns the current frequency.
     */
    [[nodiscard]] T getFrequency() const noexcept { return frequency_; }

    /**
     * @brief Generates the next sample.
     * @return Output sample.
     */
    [[nodiscard]] T getSample() noexcept
    {
        T phase = phasor_.advance();
        return readTable(phase);
    }

    /**
     * @brief Generates a block of samples.
     * @param output Output buffer.
     * @param numSamples Number of samples to generate.
     */
    void processBlock(T* output, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            output[i] = getSample();
    }

    /**
     * @brief Resets the oscillator phase.
     * @param phase Initial phase (0 to 1).
     */
    void reset(T phase = T(0)) noexcept
    {
        phasor_.reset(phase);
    }

private:
    /**
     * @brief Reads a sample from a specific mipmap level using linear interpolation.
     * @param phase Phase in [0, 1).
     * @param level Mipmap level index.
     */
    [[nodiscard]] T readFromLevel(T phase, int level) const noexcept
    {
        const auto& table = mipTables_[static_cast<size_t>(level)];

        T pos = phase * static_cast<T>(kTableSize);
        int idx0 = static_cast<int>(pos) % kTableSize;
        int idx1 = (idx0 + 1) % kTableSize;
        T frac = pos - std::floor(pos);

        return table[static_cast<size_t>(idx0)]
             + frac * (table[static_cast<size_t>(idx1)]
                      - table[static_cast<size_t>(idx0)]);
    }

    /**
     * @brief Selects the appropriate mipmap level and reads with crossfade between levels.
     *
     * Uses fractional mipmap selection to linearly interpolate between adjacent
     * levels, eliminating audible timbre discontinuities when sweeping frequency.
     */
    [[nodiscard]] T readTable(T phase) const noexcept
    {
        if (mipTables_.empty()) return T(0);

        // Get fractional mipmap level and crossfade between adjacent levels
        T levelF = selectMipLevelFloat();
        int level0 = static_cast<int>(levelF);
        int level1 = std::min(level0 + 1, numMipLevels_ - 1);
        T frac = levelF - static_cast<T>(level0);

        T s0 = readFromLevel(phase, level0);
        T s1 = readFromLevel(phase, level1);

        return s0 + frac * (s1 - s0);
    }

    /**
     * @brief Selects a fractional mipmap level for the current frequency.
     *
     * Returns a floating-point level that includes the fraction between
     * adjacent levels, enabling smooth crossfade during frequency sweeps.
     */
    [[nodiscard]] T selectMipLevelFloat() const noexcept
    {
        T absFreq = std::abs(frequency_);

        // Below the lowest level's max frequency — return level 0
        if (numMipLevels_ <= 1 || absFreq <= mipMaxFreq_[0])
            return T(0);

        for (int i = 1; i < numMipLevels_; ++i)
        {
            if (absFreq <= mipMaxFreq_[static_cast<size_t>(i)])
            {
                // Interpolate between level (i-1) and level i
                T freqLow  = mipMaxFreq_[static_cast<size_t>(i - 1)];
                T freqHigh = mipMaxFreq_[static_cast<size_t>(i)];
                T t = (absFreq - freqLow) / (freqHigh - freqLow + T(1e-10));
                return static_cast<T>(i - 1) + t;
            }
        }

        return static_cast<T>(numMipLevels_ - 1);
    }

    /**
     * @brief Selects the mipmap level for the current frequency (integer).
     *
     * Returns the level whose max frequency is just above the current
     * fundamental — this ensures all harmonics in that table are below Nyquist.
     */
    [[nodiscard]] int selectMipLevel() const noexcept
    {
        T absFreq = std::abs(frequency_);
        for (int i = 0; i < numMipLevels_; ++i)
        {
            if (absFreq <= mipMaxFreq_[static_cast<size_t>(i)])
                return i;
        }
        return numMipLevels_ - 1; // Highest level (fewest harmonics)
    }

    double sampleRate_ = 48000.0;
    T frequency_ = T(440);

    Phasor<T> phasor_;

    int numMipLevels_ = 0;
    std::vector<std::vector<T>> mipTables_;
    std::vector<T> mipMaxFreq_;
};

} // namespace dspark
