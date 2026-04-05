// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file WaveshapeTable.h
 * @brief Table-based waveshaping with interpolated lookup.
 *
 * Stores a transfer function as a lookup table and applies it to audio
 * signals with interpolated reads. This allows arbitrary waveshaping curves
 * without per-sample math — just a table lookup.
 *
 * Use cases:
 * - Custom saturation curves (tube simulation, tape emulation)
 * - Non-linear transfer functions for synthesis
 * - Sigmoid, asymmetric clipping, or any user-defined curve
 *
 * Built-in presets:
 * - **Tanh:** Smooth soft clipping
 * - **HardClip:** Sharp digital clipping
 * - **SoftClip:** Cubic soft clipping
 * - **Asymmetric:** Different positive/negative clipping (tube character)
 * - **Sine:** Sine waveshaping (foldback distortion)
 *
 * Dependencies: DspMath.h, Interpolation.h.
 *
 * @code
 *   dspark::WaveshapeTable<float> shaper;
 *   shaper.buildTanh(2.0f);  // tanh with drive = 2
 *
 *   for (int i = 0; i < numSamples; ++i)
 *       output[i] = shaper.process(input[i]);
 * @endcode
 */

#include "DspMath.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "Oversampling.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class WaveshapeTable
 * @brief Lookup-table waveshaper with built-in presets.
 *
 * The table maps input range [-1, 1] to output values. Input outside
 * this range is clamped. Interpolation between table entries ensures
 * smooth output even with moderate table sizes.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class WaveshapeTable
{
public:
    /**
     * @brief Builds the table from an arbitrary function.
     *
     * The function receives input in [-1, 1] and should return the shaped output.
     *
     * @param func Transfer function: T func(T input).
     * @param tableSize Number of table entries (default: 4096).
     */
    void buildFromFunction(std::function<T(T)> func, int tableSize = 4096)
    {
        assert(tableSize >= 4);
        tableSize_ = tableSize;
        table_.resize(static_cast<size_t>(tableSize));

        for (int i = 0; i < tableSize; ++i)
        {
            T x = T(-1) + T(2) * static_cast<T>(i) / static_cast<T>(tableSize - 1);
            table_[static_cast<size_t>(i)] = func(x);
        }
    }

    /**
     * @brief Builds a tanh (soft clip) table.
     * @param drive Drive amount (1.0 = mild, 5.0 = heavy saturation).
     * @param tableSize Table entries (default: 4096).
     */
    void buildTanh(T drive = T(1), int tableSize = 4096)
    {
        buildFromFunction([drive](T x) -> T {
            return std::tanh(x * drive);
        }, tableSize);
    }

    /**
     * @brief Builds a hard-clip table.
     * @param threshold Clipping threshold (0 to 1, default: 0.8).
     * @param tableSize Table entries.
     */
    void buildHardClip(T threshold = T(0.8), int tableSize = 4096)
    {
        buildFromFunction([threshold](T x) -> T {
            return std::clamp(x, -threshold, threshold);
        }, tableSize);
    }

    /**
     * @brief Builds a cubic soft-clip table.
     * @param tableSize Table entries.
     */
    void buildSoftClip(int tableSize = 4096)
    {
        buildFromFunction([](T x) -> T {
            if (x > T(1)) return T(2.0 / 3.0);
            if (x < T(-1)) return T(-2.0 / 3.0);
            return x - (x * x * x) / T(3);
        }, tableSize);
    }

    /**
     * @brief Builds an asymmetric clipping table (tube character).
     *
     * Positive excursions clip harder than negative, producing even harmonics.
     *
     * @param drive Drive amount.
     * @param tableSize Table entries.
     */
    void buildAsymmetric(T drive = T(2), int tableSize = 4096)
    {
        buildFromFunction([drive](T x) -> T {
            if (x >= T(0))
                return std::tanh(x * drive * T(1.2));
            else
                return std::tanh(x * drive * T(0.8));
        }, tableSize);
    }

    /**
     * @brief Builds a sine waveshaping table (foldback distortion).
     * @param drive Drive amount (higher = more folding).
     * @param tableSize Table entries.
     */
    void buildSineFold(T drive = T(1), int tableSize = 4096)
    {
        buildFromFunction([drive](T x) -> T {
            return std::sin(x * drive * pi<T>);
        }, tableSize);
    }

    /**
     * @brief Loads a custom table from raw data.
     *
     * @param data Pointer to table values.
     * @param size Number of entries.
     */
    void loadTable(const T* data, int size)
    {
        assert(size >= 4);
        tableSize_ = size;
        table_.assign(data, data + size);
    }

    /**
     * @brief Processes a single sample through the waveshaper.
     *
     * Input is expected in [-1, 1] range. Values outside this range are clamped.
     *
     * @param input Input sample.
     * @return Shaped output sample.
     */
    [[nodiscard]] T process(T input) const noexcept
    {
        // Map input [-1, 1] to table index [0, tableSize-1]
        T clamped = std::clamp(input, T(-1), T(1));
        T pos = (clamped + T(1)) * T(0.5) * static_cast<T>(tableSize_ - 1);

        // Linear interpolation
        int idx0 = static_cast<int>(pos);
        int idx1 = std::min(idx0 + 1, tableSize_ - 1);
        T frac = pos - static_cast<T>(idx0);

        return table_[static_cast<size_t>(idx0)]
             + frac * (table_[static_cast<size_t>(idx1)]
                      - table_[static_cast<size_t>(idx0)]);
    }

    /**
     * @brief Processes a buffer in-place.
     * @param data Audio samples.
     * @param numSamples Number of samples.
     */
    void process(T* data, int numSamples) const noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            data[i] = process(data[i]);
    }

    // -- Lifecycle (optional, required for oversampled processBlock) ----------

    /**
     * @brief Prepares the waveshaper for oversampled block processing.
     *
     * Only needed if using setOversampling() and processBlock().
     * The simple process(T) and process(T*, int) methods do not require this.
     *
     * @param spec Audio spec (sample rate, block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        if (oversampler_)
            oversampler_->prepare(spec);
    }

    /**
     * @brief Enables oversampling for anti-aliased waveshaping.
     *
     * Higher factors reduce aliasing from harmonic generation.
     * Requires prepare() to have been called.
     *
     * @param factor Oversampling factor (1 = off, 2, 4, 8, or 16).
     */
    void setOversampling(int factor)
    {
        assert(factor >= 1 && (factor & (factor - 1)) == 0);
        oversamplingFactor_ = factor;
        if (factor > 1)
        {
            oversampler_ = std::make_unique<Oversampling<T>>(factor);
            if (spec_.sampleRate > 0)
                oversampler_->prepare(spec_);
        }
        else
        {
            oversampler_.reset();
        }
    }

    /** @brief Returns the current oversampling factor. */
    [[nodiscard]] int getOversamplingFactor() const noexcept { return oversamplingFactor_; }

    /**
     * @brief Processes an audio buffer in-place with optional oversampling.
     *
     * Requires prepare() if oversampling is enabled.
     *
     * @param buffer Audio buffer to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (oversamplingFactor_ > 1 && oversampler_)
        {
            auto upView = oversampler_->upsample(buffer);
            for (int ch = 0; ch < upView.getNumChannels(); ++ch)
            {
                T* data = upView.getChannel(ch);
                for (int i = 0; i < upView.getNumSamples(); ++i)
                    data[i] = process(data[i]);
            }
            oversampler_->downsample(buffer);
        }
        else
        {
            for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
                process(buffer.getChannel(ch), buffer.getNumSamples());
        }
    }

    /** @brief Resets oversampling filter state. */
    void reset() noexcept
    {
        if (oversampler_)
            oversampler_->reset();
    }

    /** @brief Returns the table size. */
    [[nodiscard]] int getTableSize() const noexcept { return tableSize_; }

    /** @brief Returns true if the table has been built. */
    [[nodiscard]] bool isReady() const noexcept { return !table_.empty(); }

private:
    std::vector<T> table_;
    int tableSize_ = 0;

    AudioSpec spec_ {};
    std::unique_ptr<Oversampling<T>> oversampler_;
    int oversamplingFactor_ = 1;
};

} // namespace dspark
