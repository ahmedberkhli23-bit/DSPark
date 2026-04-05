// DSParkLab — Effect Registry
// Factory functions that create EffectSlot wrappers for all DSPark processors.

#pragma once

#include "../DSPark.h"
#include "EffectSlot.h"

#include <memory>

namespace dsplab {

// =============================================================================
// FILTERS
// =============================================================================

inline EffectSlot makeFilterEngine()
{
    auto p = std::make_shared<dspark::FilterEngine<float>>();
    EffectSlot s;
    s.name = "Filter"; s.category = "Filters";
    s.addChoice("Type", {"LowPass","HighPass","BandPass","Peak","LowShelf","HighShelf","Notch","AllPass","Tilt"}, 0);
    s.addSlider("Frequency", 20, 20000, 1000, "Hz", true);
    s.addSlider("Resonance", 0.1f, 20, 0.707f, "Q", true);
    s.addSlider("Gain", -24, 24, 0, "dB");
    s.addSlider("Slope", 6, 48, 12, "dB/oct");
    s.addSlider("Nonlinearity", 0, 1, 0, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: {
                switch(static_cast<int>(v)) {
                    case 0: p->setLowPass(1000, 0.707f, 12); break;
                    case 1: p->setHighPass(1000, 0.707f, 12); break;
                    case 2: p->setBandPass(1000, 0.707f); break;
                    case 3: p->setPeaking(1000, 0, 1.0f); break;
                    case 4: p->setLowShelf(1000, 0, 1.0f); break;
                    case 5: p->setHighShelf(1000, 0, 1.0f); break;
                    case 6: p->setNotch(1000, 0.707f); break;
                    case 7: p->setAllPass(1000, 0.707f); break;
                    case 8: p->setTilt(1000, 0); break;
                }
                break;
            }
            case 1: p->setFrequency(v); break;
            case 2: p->setResonance(v); break;
            case 3: p->setGain(v); break;
            case 4: break;
            case 5: p->setNonlinearity(v); break;
        }
    };
    return s;
}

inline EffectSlot makeEqualizer()
{
    auto p = std::make_shared<dspark::Equalizer<float>>();
    EffectSlot s;
    s.name = "Equalizer"; s.category = "Filters";
    s.addSlider("Band 1 Freq", 20, 20000, 100, "Hz", true);
    s.addSlider("Band 1 Gain", -24, 24, 0, "dB");
    s.addSlider("Band 1 Q", 0.1f, 10, 1, "");
    s.addSlider("Band 2 Freq", 20, 20000, 500, "Hz", true);
    s.addSlider("Band 2 Gain", -24, 24, 0, "dB");
    s.addSlider("Band 2 Q", 0.1f, 10, 1, "");
    s.addSlider("Band 3 Freq", 20, 20000, 2000, "Hz", true);
    s.addSlider("Band 3 Gain", -24, 24, 0, "dB");
    s.addSlider("Band 3 Q", 0.1f, 10, 1, "");
    s.addSlider("Band 4 Freq", 20, 20000, 8000, "Hz", true);
    s.addSlider("Band 4 Gain", -24, 24, 0, "dB");
    s.addSlider("Band 4 Q", 0.1f, 10, 1, "");
    s.prepareFn = [p](auto& sp) { p->setNumBands(4); p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        int band = i / 3;
        int param = i % 3;
        if (band >= 4) return;
        auto cfg = p->getBandConfig(band);
        switch(param) {
            case 0: cfg.frequency = v; break;
            case 1: cfg.gain = v; break;
            case 2: cfg.q = v; break;
        }
        p->setBand(band, cfg);
    };
    return s;
}

inline EffectSlot makeLadderFilter()
{
    auto p = std::make_shared<dspark::LadderFilter<float>>();
    EffectSlot s;
    s.name = "Ladder Filter"; s.category = "Filters";
    s.addSlider("Cutoff", 20, 20000, 1000, "Hz", true);
    s.addSlider("Resonance", 0, 1, 0.5f, "");
    s.addSlider("Drive", 0, 20, 0, "dB");
    s.addChoice("Mode", {"LP24","LP18","LP12","LP6","HP12","BP12"}, 0);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setCutoff(v); break;
            case 1: p->setResonance(v); break;
            case 2: p->setDrive(v); break;
            case 3: p->setMode(static_cast<typename dspark::LadderFilter<float>::Mode>(static_cast<int>(v))); break;
        }
    };
    return s;
}

inline EffectSlot makeStateVariableFilter()
{
    auto p = std::make_shared<dspark::StateVariableFilter<float>>();
    EffectSlot s;
    s.name = "SVF Filter"; s.category = "Filters";
    s.addSlider("Cutoff", 20, 20000, 1000, "Hz", true);
    s.addSlider("Q", 0.1f, 20, 0.707f, "");
    s.addSlider("Gain", -24, 24, 0, "dB");
    s.addChoice("Mode", {"LowPass","HighPass","BandPass","Notch","AllPass","Bell","LowShelf","HighShelf"}, 0);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setCutoff(v); break;
            case 1: p->setQ(v); break;
            case 2: p->setGain(v); break;
            case 3: p->setMode(static_cast<typename dspark::StateVariableFilter<float>::Mode>(static_cast<int>(v))); break;
        }
    };
    return s;
}

inline EffectSlot makeDCBlocker()
{
    auto p = std::make_shared<dspark::DCBlocker<float>>();
    auto sr = std::make_shared<double>(44100.0);
    EffectSlot s;
    s.name = "DC Blocker"; s.category = "Filters";
    s.addSlider("Cutoff", 1, 100, 5, "Hz");
    s.addSlider("Order", 1, 10, 1, "");
    s.prepareFn = [p, sr](auto& sp) { p->prepare(sp); *sr = sp.sampleRate; };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, sr](int i, float v) {
        switch(i) {
            case 0: p->setCutoff(*sr, static_cast<double>(v)); break;
            case 1: p->setOrder(static_cast<int>(v)); break;
        }
    };
    return s;
}

// =============================================================================
// DYNAMICS
// =============================================================================

inline EffectSlot makeCompressor()
{
    auto p = std::make_shared<dspark::Compressor<float>>();
    EffectSlot s;
    s.name = "Compressor"; s.category = "Dynamics";
    s.addSlider("Threshold", -60, 0, -20, "dB");
    s.addSlider("Ratio", 1, 20, 4, ":1");
    s.addSlider("Attack", 0.1f, 100, 10, "ms");
    s.addSlider("Release", 10, 1000, 100, "ms");
    s.addSlider("Knee", 0, 30, 6, "dB");
    s.addSlider("Makeup", -12, 30, 0, "dB");
    s.addToggle("Auto Makeup", true);
    s.addSlider("Mix", 0, 1, 1, "");
    s.addChoice("Detector", {"Peak","RMS","SplitPolarity"}, 0);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setThreshold(v); break;
            case 1: p->setRatio(v); break;
            case 2: p->setAttack(v); break;
            case 3: p->setRelease(v); break;
            case 4: p->setKnee(v); break;
            case 5: p->setMakeupGain(v); break;
            case 6: p->setAutoMakeup(v > 0.5f); break;
            case 7: p->setMix(v); break;
            case 8: p->setDetector(static_cast<dspark::Compressor<float>::DetectorType>(static_cast<int>(v))); break;
        }
    };
    return s;
}

inline EffectSlot makeLimiter()
{
    auto p = std::make_shared<dspark::Limiter<float>>();
    auto inputGainDb = std::make_shared<float>(0.0f);
    EffectSlot s;
    s.name = "Limiter"; s.category = "Dynamics";
    s.addSlider("Input Gain", 0, 30, 0, "dB");
    s.addSlider("Ceiling", -30, 0, -1, "dB");
    s.addSlider("Release", 10, 1000, 100, "ms");
    s.addSlider("Lookahead", 0, 10, 5, "ms");
    s.addToggle("Safety Clip", true);
    s.addToggle("Adaptive Release", true);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p, inputGainDb](auto b) {
        float gDb = *inputGainDb;
        p->processBlock(b);
        if (gDb > 0.01f) {
            float g = dspark::decibelsToGain(gDb);
            for (int ch = 0; ch < b.getNumChannels(); ++ch)
                for (int i = 0; i < b.getNumSamples(); ++i)
                    b.getChannel(ch)[i] *= g;
        }
    };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p, inputGainDb](int i, float v) {
        switch(i) {
            case 0: *inputGainDb = v; break;
            case 1: p->setCeiling(v); break;
            case 2: p->setRelease(v); break;
            case 3: p->setLookahead(v); break;
            case 4: p->setSafetyClip(v > 0.5f); break;
            case 5: p->setAdaptiveRelease(v > 0.5f); break;
        }
    };
    return s;
}

inline EffectSlot makeNoiseGate()
{
    auto p = std::make_shared<dspark::NoiseGate<float>>();
    EffectSlot s;
    s.name = "Noise Gate"; s.category = "Dynamics";
    s.addSlider("Threshold", -80, 0, -40, "dB");
    s.addSlider("Attack", 0.1f, 50, 1, "ms");
    s.addSlider("Hold", 0, 500, 50, "ms");
    s.addSlider("Release", 5, 500, 50, "ms");
    s.addSlider("Range", -80, 0, -80, "dB");
    s.addChoice("Gate Mode", {"Amplitude","Frequency"}, 0);
    s.addToggle("Adaptive Hold", false);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setThreshold(v); break;
            case 1: p->setAttack(v); break;
            case 2: p->setHold(v); break;
            case 3: p->setRelease(v); break;
            case 4: p->setRange(v); break;
            case 5: p->setGateMode(static_cast<dspark::NoiseGate<float>::GateMode>(static_cast<int>(v))); break;
            case 6: p->setAdaptiveHold(v > 0.5f); break;
        }
    };
    return s;
}

inline EffectSlot makeExpander()
{
    auto p = std::make_shared<dspark::Expander<float>>();
    EffectSlot s;
    s.name = "Expander"; s.category = "Dynamics";
    s.addSlider("Threshold", -60, 0, -30, "dB");
    s.addSlider("Ratio", 1, 20, 4, ":1");
    s.addSlider("Attack", 0.1f, 50, 5, "ms");
    s.addSlider("Hold", 0, 500, 50, "ms");
    s.addSlider("Release", 5, 500, 100, "ms");
    s.addSlider("Range", -80, 0, -60, "dB");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setThreshold(v); break;
            case 1: p->setRatio(v); break;
            case 2: p->setAttack(v); break;
            case 3: p->setHold(v); break;
            case 4: p->setRelease(v); break;
            case 5: p->setRange(v); break;
        }
    };
    return s;
}

inline EffectSlot makeTransientDesigner()
{
    auto p = std::make_shared<dspark::TransientDesigner<float>>();
    EffectSlot s;
    s.name = "Transient Designer"; s.category = "Dynamics";
    s.addSlider("Attack", -100, 100, 0, "%");
    s.addSlider("Sustain", -100, 100, 0, "%");
    s.addSlider("Character", -1, 1, 0, "");
    s.addToggle("Output-Dep Recovery", false);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setAttack(v); break;
            case 1: p->setSustain(v); break;
            case 2: p->setCharacter(v); break;
            case 3: p->setOutputDepRecovery(v > 0.5f); break;
        }
    };
    return s;
}

inline EffectSlot makeDeEsser()
{
    auto p = std::make_shared<dspark::DeEsser<float>>();
    EffectSlot s;
    s.name = "De-Esser"; s.category = "Dynamics";
    s.addSlider("Frequency", 2000, 12000, 6000, "Hz", true);
    s.addSlider("Bandwidth", 0.5f, 4, 1, "oct");
    s.addSlider("Threshold", -40, 0, -20, "dB");
    s.addSlider("Reduction", 0, 24, 6, "dB");
    s.addChoice("Detection", {"Bandpass","Derivative"}, 0);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setFrequency(v); break;
            case 1: p->setBandwidth(v); break;
            case 2: p->setThreshold(v); break;
            case 3: p->setReduction(v); break;
            case 4: p->setDetectionMode(static_cast<dspark::DeEsser<float>::DetectionMode>(static_cast<int>(v))); break;
        }
    };
    return s;
}

// =============================================================================
// DISTORTION
// =============================================================================

inline EffectSlot makeSaturation()
{
    auto p = std::make_shared<dspark::Saturation<float>>();
    EffectSlot s;
    s.name = "Saturation"; s.category = "Distortion";
    s.addChoice("Algorithm", {"SoftClip","HardClip","Tanh","Tube","Tape","Transformer","Foldback","Bitcrusher","Downsample","MultiStage"}, 0);
    s.addSlider("Drive", 0, 40, 6, "dB");
    s.addSlider("Mix", 0, 1, 1, "");
    s.addSlider("Output", -24, 12, 0, "dB");
    s.addToggle("Adaptive Blend", false);
    s.addSlider("Slew Sensitivity", 0, 1, 0, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->process(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setAlgorithm(static_cast<typename dspark::Saturation<float>::Algorithm>(static_cast<int>(v))); break;
            case 1: p->setDrive(v); break;
            case 2: p->setMix(v); break;
            case 3: p->setOutputGain(v); break;
            case 4: p->setAdaptiveBlend(v > 0.5f); break;
            case 5: p->setSlewSensitivity(v); break;
        }
    };
    return s;
}

inline EffectSlot makeClipper()
{
    auto p = std::make_shared<dspark::Clipper<float>>();
    EffectSlot s;
    s.name = "Clipper"; s.category = "Distortion";
    s.addChoice("Mode", {"Hard","Soft","Analog","GoldenRatio"}, 0);
    s.addSlider("Ceiling", -60, 0, 0, "dB");
    s.addSlider("Input Gain", 0, 48, 0, "dB");
    s.addSlider("Stages", 1, 4, 1, "");
    s.addSlider("Mix", 0, 1, 1, "");
    s.addSlider("Slew Limit", 0, 1, 0, "");
    s.addChoice("Oversampling", {"2x","4x","8x","16x"}, 0);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setMode(static_cast<dspark::Clipper<float>::Mode>(static_cast<int>(v))); break;
            case 1: p->setCeiling(v); break;
            case 2: p->setInputGain(v); break;
            case 3: p->setStages(static_cast<int>(v)); break;
            case 4: p->setMix(v); break;
            case 5: p->setSlewLimit(v); break;
            case 6: { int factors[] = {2,4,8,16}; p->setOversampling(factors[static_cast<int>(v)]); break; }
        }
    };
    return s;
}

// =============================================================================
// MODULATION
// =============================================================================

inline EffectSlot makeChorus()
{
    auto p = std::make_shared<dspark::Chorus<float>>();
    EffectSlot s;
    s.name = "Chorus"; s.category = "Modulation";
    s.addSlider("Rate", 0.1f, 10, 1.5f, "Hz");
    s.addSlider("Depth", 0, 1, 0.5f, "");
    s.addSlider("Mix", 0, 1, 0.5f, "");
    s.addSlider("Voices", 1, 6, 2, "");
    s.addSlider("Feedback", -0.95f, 0.95f, 0, "");
    s.addToggle("Auto Depth", false);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setRate(v); break;
            case 1: p->setDepth(v); break;
            case 2: p->setMix(v); break;
            case 3: p->setVoices(static_cast<int>(v)); break;
            case 4: p->setFeedback(v); break;
            case 5: p->setAutoDepth(v > 0.5f); break;
        }
    };
    return s;
}

inline EffectSlot makePhaser()
{
    auto p = std::make_shared<dspark::Phaser<float>>();
    EffectSlot s;
    s.name = "Phaser"; s.category = "Modulation";
    s.addSlider("Rate", 0.01f, 10, 0.5f, "Hz");
    s.addSlider("Depth", 0, 1, 0.7f, "");
    s.addSlider("Mix", 0, 1, 0.5f, "");
    s.addSlider("Stages", 2, 12, 4, "");
    s.addSlider("Feedback", -0.95f, 0.95f, 0.3f, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setRate(v); break;
            case 1: p->setDepth(v); break;
            case 2: p->setMix(v); break;
            case 3: p->setStages(static_cast<int>(v)); break;
            case 4: p->setFeedback(v); break;
        }
    };
    return s;
}

inline EffectSlot makeTremolo()
{
    auto p = std::make_shared<dspark::Tremolo<float>>();
    EffectSlot s;
    s.name = "Tremolo"; s.category = "Modulation";
    s.addSlider("Rate", 0.1f, 20, 4, "Hz");
    s.addSlider("Depth", 0, 1, 0.5f, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setRate(v); break;
            case 1: p->setDepth(v); break;
        }
    };
    return s;
}

inline EffectSlot makeVibrato()
{
    auto p = std::make_shared<dspark::Vibrato<float>>();
    EffectSlot s;
    s.name = "Vibrato"; s.category = "Modulation";
    s.addSlider("Rate", 0.1f, 15, 5, "Hz");
    s.addSlider("Depth", 0, 2, 0.5f, "st");
    s.addSlider("Mod Rate", 0, 5, 0, "Hz");
    s.addSlider("Mod Depth", 0, 1, 0, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setRate(v); break;
            case 1: p->setDepth(v); break;
            case 2: p->setModRate(v); break;
            case 3: p->setModDepth(v); break;
        }
    };
    return s;
}

inline EffectSlot makeRingModulator()
{
    auto p = std::make_shared<dspark::RingModulator<float>>();
    EffectSlot s;
    s.name = "Ring Modulator"; s.category = "Modulation";
    s.addSlider("Frequency", 20, 5000, 440, "Hz", true);
    s.addSlider("Mix", 0, 1, 0.5f, "");
    s.addChoice("Mode", {"Classic","GeometricMean"}, 0);
    s.addSlider("Soar", 0, 0.2f, 0, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setFrequency(v); break;
            case 1: p->setMix(v); break;
            case 2: p->setMode(static_cast<dspark::RingModulator<float>::Mode>(static_cast<int>(v))); break;
            case 3: p->setSoar(v); break;
        }
    };
    return s;
}

inline EffectSlot makeFrequencyShifter()
{
    auto p = std::make_shared<dspark::FrequencyShifter<float>>();
    EffectSlot s;
    s.name = "Freq Shifter"; s.category = "Modulation";
    s.addSlider("Shift", -500, 500, 0, "Hz");
    s.addSlider("Mix", 0, 1, 1, "");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setShift(v); break;
            case 1: p->setMix(v); break;
        }
    };
    return s;
}

// =============================================================================
// SPATIAL
// =============================================================================

inline EffectSlot makeDelay()
{
    struct Params { float ms=250; float fb=0.4f; float lp=8000; float hp=80; };
    auto p = std::make_shared<dspark::Delay<float>>();
    auto par = std::make_shared<Params>();
    EffectSlot s;
    s.name = "Delay"; s.category = "Spatial";
    s.addSlider("Time", 1, 1000, 250, "ms");
    s.addSlider("Feedback", 0, 0.95f, 0.4f, "");
    s.addSlider("LP Filter", 200, 20000, 8000, "Hz", true);
    s.addSlider("HP Filter", 20, 2000, 80, "Hz", true);
    s.prepareFn = [p](auto& sp) { p->prepare(sp, 2.0); };
    s.processFn = [p, par](auto b) { p->processBlock(b, par->ms, par->fb, par->lp, par->hp); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [par](int i, float v) {
        switch(i) {
            case 0: par->ms = v; break;
            case 1: par->fb = v; break;
            case 2: par->lp = v; break;
            case 3: par->hp = v; break;
        }
    };
    return s;
}

inline EffectSlot makeAlgorithmicReverb()
{
    auto p = std::make_shared<dspark::AlgorithmicReverb<float>>();
    EffectSlot s;
    s.name = "Reverb"; s.category = "Spatial";
    s.addChoice("Type", {"Room","Hall","Chamber","Plate","Spring","Cathedral"}, 0); // 0
    s.addSlider("Decay", 0.1f, 10, 1.5f, "s");              // 1
    s.addSlider("Size", 0, 1, 0.5f, "");                     // 2
    s.addSlider("Damping", 0, 1, 0.5f, "");                  // 3
    s.addSlider("Pre-Delay", 0, 100, 10, "ms");              // 4
    s.addSlider("Diffusion", 0, 1, 0.8f, "");                // 5
    s.addSlider("Modulation", 0, 1, 0.15f, "");              // 6
    s.addSlider("HF Decay", 0.1f, 2, 0.5f, "x");            // 7
    s.addSlider("Bass Decay", 0.5f, 2, 1.2f, "x");          // 8
    s.addSlider("HF Crossover", 500, 16000, 5000, "Hz", true); // 9 (log)
    s.addSlider("Bass Crossover", 50, 1000, 200, "Hz", true);  // 10 (log)
    s.addSlider("Tone Low Cut", 10, 500, 20, "Hz", true);    // 11 (log)
    s.addSlider("Tone High Cut", 2000, 20000, 16000, "Hz", true); // 12 (log)
    s.addSlider("ER-Late Gap", 0, 100, 10, "ms");             // 13
    s.addSlider("Mod Rate", 0.1f, 5, 1, "Hz");               // 14
    s.addSlider("Early Level", -20, 6, 0, "dB");             // 15
    s.addSlider("Late Level", -20, 6, 0, "dB");              // 16
    s.addSlider("Width", 0, 2, 1, "");                        // 17
    s.addSlider("Mix", 0, 1, 0.3f, "");                      // 18
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0:  p->setType(static_cast<typename dspark::AlgorithmicReverb<float>::Type>(static_cast<int>(v))); break;
            case 1:  p->setDecay(v); break;
            case 2:  p->setSize(v); break;
            case 3:  p->setDamping(v); break;
            case 4:  p->setPreDelay(v); break;
            case 5:  p->setDiffusion(v); break;
            case 6:  p->setModulation(v); break;
            case 7:  p->setHighDecayMultiplier(v); break;
            case 8:  p->setBassDecayMultiplier(v); break;
            case 9:  p->setHighCrossover(v); break;
            case 10: p->setBassCrossover(v); break;
            case 11: p->setToneLowCut(v); break;
            case 12: p->setToneHighCut(v); break;
            case 13: p->setErToLateDelay(v); break;
            case 14: p->setModRate(v); break;
            case 15: p->setEarlyLevel(v); break;
            case 16: p->setLateLevel(v); break;
            case 17: p->setWidth(v); break;
            case 18: p->setMix(v); break;
        }
    };
    return s;
}

inline EffectSlot makePanner()
{
    auto p = std::make_shared<dspark::Panner<float>>();
    EffectSlot s;
    s.name = "Panner"; s.category = "Spatial";
    s.addSlider("Pan", -1, 1, 0, "");                     // 0
    s.addChoice("Algorithm", {"EqualPower","Binaural","MidPan","SidePan","Haas","Spectral"}, 0); // 1
    s.addSlider("Binaural ITD", 0.1f, 5, 0.66f, "ms");    // 2
    s.addSlider("Haas Delay", 0.1f, 40, 30, "ms");        // 3
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setPan(v); break;
            case 1: p->setAlgorithm(static_cast<typename dspark::Panner<float>::Algorithm>(static_cast<int>(v))); break;
            case 2: p->setBinauralMaxITD(v); break;
            case 3: p->setHaasMaxDelay(v); break;
        }
    };
    return s;
}

inline EffectSlot makeStereoWidth()
{
    auto p = std::make_shared<dspark::StereoWidth<float>>();
    EffectSlot s;
    s.name = "Stereo Width"; s.category = "Spatial";
    s.addSlider("Width", 0, 2, 1, "");
    s.addToggle("Bass Mono", false);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setWidth(v); break;
            case 1: p->setBassMono(v > 0.5f); break;
        }
    };
    return s;
}

// =============================================================================
// UTILITY
// =============================================================================

inline EffectSlot makeGain()
{
    auto p = std::make_shared<dspark::Gain<float>>();
    EffectSlot s;
    s.name = "Gain"; s.category = "Utility";
    s.addSlider("Gain", -60, 24, 0, "dB");
    s.addToggle("Mute", false);
    s.addToggle("Invert", false);
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setGainDb(v); break;
            case 1: p->setMuted(v > 0.5f); break;
            case 2: p->setInverted(v > 0.5f); break;
        }
    };
    return s;
}

inline EffectSlot makeNoiseGenerator()
{
    auto p = std::make_shared<dspark::NoiseGenerator<float>>();
    EffectSlot s;
    s.name = "Noise Generator"; s.category = "Utility";
    s.addChoice("Type", {"White","Pink","Brown"}, 0);
    s.addSlider("Level", -60, 0, -20, "dB");
    s.prepareFn = [p](auto& sp) { p->prepare(sp); };
    s.processFn = [p](auto b) { p->processBlock(b); };
    s.resetFn   = [p]() { p->reset(); };
    s.setParamFn = [p](int i, float v) {
        switch(i) {
            case 0: p->setType(static_cast<typename dspark::NoiseGenerator<float>::Type>(static_cast<int>(v))); break;
            case 1: p->setLevel(v); break;
        }
    };
    return s;
}

// =============================================================================
// MASTER REGISTRY
// =============================================================================

inline std::vector<EffectSlot> createAllEffects()
{
    return {
        // Filters
        makeFilterEngine(),
        makeEqualizer(),
        makeLadderFilter(),
        makeStateVariableFilter(),
        makeDCBlocker(),
        // Dynamics
        makeCompressor(),
        makeLimiter(),
        makeNoiseGate(),
        makeExpander(),
        makeTransientDesigner(),
        makeDeEsser(),
        // Distortion
        makeSaturation(),
        makeClipper(),
        // Modulation
        makeChorus(),
        makePhaser(),
        makeTremolo(),
        makeVibrato(),
        makeRingModulator(),
        makeFrequencyShifter(),
        // Spatial
        makeDelay(),
        makeAlgorithmicReverb(),
        makePanner(),
        makeStereoWidth(),
        // Utility
        makeGain(),
        makeNoiseGenerator(),
    };
}

} // namespace dsplab
