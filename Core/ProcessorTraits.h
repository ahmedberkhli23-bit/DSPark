// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file ProcessorTraits.h
 * @brief C++20 concepts defining the DSP processor contract.
 *
 * Provides compile-time concepts that formalise what it means to be a
 * DSP processor in this framework. Any class that satisfies these concepts
 * can be used with ProcessorChain and other generic utilities.
 *
 * No virtual functions, no base class inheritance, no runtime overhead.
 * Just compile-time constraints that produce clear error messages.
 *
 * The three levels of processor:
 *
 * | Concept          | Required methods                                    |
 * |------------------|-----------------------------------------------------|
 * | `AudioProcessor` | `prepare(AudioSpec)`, `processBlock(BufferView)`, `reset()` |
 * | `SampleProcessor`| Above + `processSample(T, int) -> T`                |
 * | `GeneratorProcessor` | `prepare(AudioSpec)`, `reset()`, `getSample() -> T` |
 *
 * Dependencies: AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   // At compile time, check that MyFilter satisfies AudioProcessor:
 *   static_assert(dspark::AudioProcessor<MyFilter, float>);
 *
 *   // Use in template constraints:
 *   template <dspark::AudioProcessor<float> P>
 *   void applyEffect(P& proc, dspark::AudioBufferView<float> buf) {
 *       proc.processBlock(buf);
 *   }
 * @endcode
 */

#include "AudioSpec.h"
#include "AudioBuffer.h"

#include <concepts>

namespace dspark {

/**
 * @concept AudioProcessor
 * @brief A type that can prepare, process audio blocks, and reset.
 *
 * This is the primary concept for processors in the framework.
 * Any class with these three methods can be used in a ProcessorChain.
 *
 * @tparam P Processor type.
 * @tparam T Sample type (float or double).
 */
template <typename P, typename T>
concept AudioProcessor = requires(P p, const AudioSpec& spec, AudioBufferView<T> buf) {
    p.prepare(spec);
    p.processBlock(buf);
    p.reset();
};

/**
 * @concept SampleProcessor
 * @brief An AudioProcessor that also supports per-sample processing.
 *
 * Adds the requirement for a `processSample(T, int) -> T` method,
 * where the int parameter is the channel index.
 *
 * @tparam P Processor type.
 * @tparam T Sample type.
 */
template <typename P, typename T>
concept SampleProcessor = AudioProcessor<P, T> &&
    requires(P p, T sample, int channel) {
        { p.processSample(sample, channel) } -> std::convertible_to<T>;
    };

/**
 * @concept GeneratorProcessor
 * @brief A processor that generates audio (oscillator, envelope, noise).
 *
 * Generators produce output rather than transforming input. They have
 * `prepare()` and `reset()` but use `getSample()` instead of `processBlock()`.
 *
 * @tparam P Processor type.
 * @tparam T Sample type.
 */
template <typename P, typename T>
concept GeneratorProcessor = requires(P p, const AudioSpec& spec) {
    p.prepare(spec);
    p.reset();
    { p.getSample() } -> std::convertible_to<T>;
};

} // namespace dspark
