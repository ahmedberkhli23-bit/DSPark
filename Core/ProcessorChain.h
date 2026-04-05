// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file ProcessorChain.h
 * @brief Compile-time processor chain for composing DSP effects.
 *
 * Chains multiple processors together so they are prepared, processed,
 * and reset as a single unit. Uses `std::tuple` internally — all dispatch
 * is resolved at compile time with zero runtime overhead.
 *
 * Each processor in the chain must satisfy the AudioProcessor concept
 * (i.e., have `prepare(AudioSpec)`, `processBlock(AudioBufferView<T>)`, and `reset()`).
 *
 * Dependencies: ProcessorTraits.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   // Build a chain: highpass → compressor → gain
 *   dspark::ProcessorChain<float,
 *       dspark::FilterEngine<float>,
 *       dspark::Compressor<float>,
 *       dspark::Gain<float>> chain;
 *
 *   // Configure
 *   chain.prepare(spec);
 *   chain.get<0>().setHighPass(80.0f);
 *   chain.get<1>().setThreshold(-20.0f);
 *   chain.get<1>().setRatio(4.0f);
 *   chain.get<2>().setGainDb(-3.0f);
 *
 *   // In audio callback — processes all three in order:
 *   chain.processBlock(buffer);
 * @endcode
 */

#include "ProcessorTraits.h"

#include <cstddef>
#include <tuple>
#include <utility>

namespace dspark {

/**
 * @class ProcessorChain
 * @brief Compile-time chain of audio processors.
 *
 * Processors are stored in a `std::tuple` and invoked in order.
 * Access individual processors via `get<Index>()` to configure parameters.
 *
 * @tparam T           Sample type (float or double).
 * @tparam Processors  Processor types (must each satisfy AudioProcessor<P, T>).
 */
template <typename T, typename... Processors>
class ProcessorChain
{
public:
    /**
     * @brief Prepares all processors in order.
     * @param spec Audio specification.
     */
    void prepare(const AudioSpec& spec)
    {
        std::apply([&spec](auto&... procs) {
            (procs.prepare(spec), ...);
        }, processors_);
    }

    /**
     * @brief Processes a buffer through all non-bypassed processors in order.
     * @param buffer Audio buffer (modified in-place by each processor).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        processBlockImpl(buffer, std::index_sequence_for<Processors...>{});
    }

    /**
     * @brief Resets all processors.
     */
    void reset() noexcept
    {
        std::apply([](auto&... procs) {
            (procs.reset(), ...);
        }, processors_);
    }

    /**
     * @brief Accesses the processor at the given index.
     *
     * @tparam Index Zero-based index into the chain.
     * @return Reference to the processor.
     *
     * @code
     *   chain.get<0>().setThreshold(-20.0f);
     *   chain.get<1>().setGainDb(-3.0f);
     * @endcode
     */
    template <std::size_t Index>
    [[nodiscard]] auto& get() noexcept
    {
        return std::get<Index>(processors_);
    }

    /**
     * @brief Const access to the processor at the given index.
     */
    template <std::size_t Index>
    [[nodiscard]] const auto& get() const noexcept
    {
        return std::get<Index>(processors_);
    }

    /**
     * @brief Returns the number of processors in the chain.
     */
    [[nodiscard]] static constexpr std::size_t size() noexcept
    {
        return sizeof...(Processors);
    }

    /// @brief Returns the total latency in samples across all processors.
    /// Only sums from processors that provide a getLatency() method.
    [[nodiscard]] int getLatency() const noexcept
    {
        return getLatencyImpl(std::index_sequence_for<Processors...>{});
    }

    // -- Bypass control ------------------------------------------------------

    /**
     * @brief Bypasses or enables a processor at the given index.
     *
     * A bypassed processor's processBlock() is skipped entirely —
     * audio passes through unchanged at that slot.
     *
     * @tparam Index Zero-based index into the chain.
     * @param bypassed True to bypass, false to enable.
     */
    template <std::size_t Index>
    void setBypassed(bool bypassed) noexcept
    {
        static_assert(Index < sizeof...(Processors), "Index out of range");
        bypassed_[Index] = bypassed;
    }

    /**
     * @brief Returns whether a processor at the given index is bypassed.
     */
    template <std::size_t Index>
    [[nodiscard]] bool isBypassed() const noexcept
    {
        static_assert(Index < sizeof...(Processors), "Index out of range");
        return bypassed_[Index];
    }

private:
    template <std::size_t... Is>
    [[nodiscard]] int getLatencyImpl(std::index_sequence<Is...>) const noexcept
    {
        return (getProcessorLatency<Is>() + ...);
    }

    template <std::size_t I>
    [[nodiscard]] int getProcessorLatency() const noexcept
    {
        if constexpr (requires { std::get<I>(processors_).getLatency(); })
            return std::get<I>(processors_).getLatency();
        else
            return 0;
    }

    template <std::size_t... Is>
    void processBlockImpl(AudioBufferView<T> buffer, std::index_sequence<Is...>) noexcept
    {
        ((!bypassed_[Is] ? std::get<Is>(processors_).processBlock(buffer) : (void)0), ...);
    }

    std::tuple<Processors...> processors_;
    std::array<bool, sizeof...(Processors)> bypassed_ {};
};

} // namespace dspark
