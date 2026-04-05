// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file HarmonyConstants.h
 * @brief Comprehensive musical harmony calculations — scales, chords, MIDI, theory.
 *
 * A complete constexpr toolkit for working with musical scales and chords.
 * Fully functional at compile-time (`constexpr` / `consteval`) — can generate
 * static data tables for audio plugins, tuners, and musical analysis.
 *
 * Features:
 * - 61 common musical scales (bitmask representation, relative to C).
 * - 15 standard chord recipes with inversions.
 * - Context-aware note naming (sharp/flat by key).
 * - Scale transposition and diatonic chord generation.
 * - MIDI note conversion and octave helpers.
 *
 * Dependencies: C++20 standard library only.
 *
 * @code
 *   using namespace dspark::harmony;
 *   auto major = allScales[0];                    // Ionian (Major)
 *   auto inD = scaleAtRoot(major.mask, 2);        // D Major
 *   auto name = noteName(69, 0);                  // "A" (MIDI 69 = A4)
 *   auto chord = chordAtRootMidi(allChords[0], 60); // C Major triad
 * @endcode
 */

#include <array>
#include <string_view>
#include <cstdint>
#include <algorithm>
#include <optional>

namespace dspark {
namespace harmony
{
    //==============================================================================
    // 0. CORE TYPEDEFS
    //==============================================================================

    /**
     * @typedef NoteSet
     * @brief A 12-bit bitmask representing the 12 pitch-classes of the chromatic scale.
     *
     * Bits 0..11 correspond to semitones above the root (0=C, 1=C#/Db, ..., 11=B).
     */
    using NoteSet = std::uint16_t;

    /**
     * @typedef Degree
     * @brief Integer index used to select a degree inside standard diatonic representations.
     */
    using Degree = int;

    //==============================================================================
    // 1. CHORDTAG - A FILTER FOR CHORD TYPES
    //==============================================================================

    /**
     * @enum ChordTag
     * @brief Bitmask flags describing chord "families" compatible with a scale.
     *
     * Use these flags to filter scales by which chord types are naturally available
     * inside them (e.g., MajorTriad, Dominant7...).
     */
    enum class ChordTag : std::uint16_t
    {
        MajorTriad      = 1u << 0,   ///< Contains a major triad (e.g., C-E-G).
        MinorTriad      = 1u << 1,   ///< Contains a minor triad (e.g., C-Eb-G).
        DiminishedTriad = 1u << 2,   ///< Contains a diminished triad (e.g., C-Eb-Gb).
        AugmentedTriad  = 1u << 3,   ///< Contains an augmented triad (e.g., C-E-G#).
        Major7          = 1u << 4,   ///< Contains a major 7th chord.
        Dominant7       = 1u << 5,   ///< Contains a dominant 7th chord.
        Minor7          = 1u << 6,   ///< Contains a minor 7th chord.
        HalfDim7        = 1u << 7,   ///< Contains a half-diminished 7th chord.
        Dim7            = 1u << 8,   ///< Contains a fully-diminished 7th chord.
        Sus2Triad       = 1u << 9,   ///< Contains a suspended-2 triad.
        Sus4Triad       = 1u << 10,  ///< Contains a suspended-4 triad.
        Major9          = 1u << 11,  ///< Contains a major 9th chord.
        Dominant9       = 1u << 12,  ///< Contains a dominant 9th chord.
        Minor9          = 1u << 13,  ///< Contains a minor 9th chord.
        Major11         = 1u << 14,  ///< Contains a major 11th chord.
        Dominant11      = 1u << 15,  ///< Contains a dominant 11th chord.
        All             = 0xFFFFu    ///< Convenience: all flags set.
    };

    /**
     * @brief Bitwise OR operator for ChordTag flags.
     * @return Combination of both flags.
     */
    [[nodiscard]] constexpr ChordTag operator|(ChordTag lhs, ChordTag rhs) noexcept;


    //==============================================================================
    // 2. SCALE DESCRIPTOR
    //==============================================================================

    /**
     * @struct Scale
     * @brief Descriptor holding the name, pitch mask and chord tags for a scale.
     */
    struct Scale
    {
        std::string_view name; ///< Human-readable scale name (root = C in the database).
        NoteSet          mask; ///< 12-bit mask with set bits for the scale degrees.
        ChordTag         tags; ///< Flags describing chord families present in the scale.
    };


    //==============================================================================
    // 3. HELPER: BUILD A NOTESET
    //==============================================================================

    /**
     * @brief Build a NoteSet from 12 boolean flags (b0 = C, b1 = C#/Db, ... b11 = B).
     * @details `consteval` so it can be used in compile-time tables.
     */
    [[nodiscard]] consteval NoteSet makeMask(
        bool b0, bool b1, bool b2, bool b3,
        bool b4, bool b5, bool b6, bool b7,
        bool b8, bool b9, bool b10, bool b11) noexcept;

    /**
     * @brief Build a NoteSet from a list of semitone degrees (values may be >=12 or negative).
     * @details Degree values are taken modulo 12 so inputs like 14 -> 2 are accepted.
     * Negative degrees are ignored.
     * @param degrees Initializer list of semitone distances (0..n).
     */
    [[nodiscard]] consteval NoteSet makeMask(std::initializer_list<int> degrees) noexcept;


    //==============================================================================
    // 4. DATABASE OF SCALES
    //==============================================================================

    // allScales: defined below after makeMask is available.


    //==============================================================================
    // 5. CONTEXT-AWARE NOTE NAMES
    //==============================================================================

    // sharpNames, flatNames, useSharpsForRoot: defined below.

    /**
     * @brief Returns a human-readable note name for the given MIDI note (0..127)
     * and a root (0..11) used to choose sharp/flat presentation.
     */
    [[nodiscard]] constexpr std::string_view noteName(int midi, int root = 0) noexcept;

    /**
     * @brief Parse a simple note name (no octave) into a pitch-class 0..11.
     * @return std::optional<int> containing 0..11 on success, std::nullopt on failure.
     */
    [[nodiscard]] constexpr std::optional<int> parseNote(std::string_view s) noexcept;


    //==============================================================================
    // 6. TRANSPOSE A SCALE
    //==============================================================================

    /**
     * @brief Circularly rotate a NoteSet so it becomes rooted at `root` (0..11).
     * @param base NoteSet defined with root = C.
     * @param root Desired root as semitone offset from C.
     */
    [[nodiscard]] constexpr NoteSet
    scaleAtRoot(NoteSet base, int root) noexcept;


    //==============================================================================
    // 7. CHORD DESCRIPTOR
    //==============================================================================

    /**
     * @struct Chord
     * @brief A chord "recipe": name + intervals (root,3,5,7,9,11,13).
     *
     * Interval slots use -1 to indicate 'not present'. Intervals are measured in semitones
     * from the chord root (e.g., Major: 0,4,7,-1,-1,-1,-1).
     */
    struct Chord
    {
        std::string_view   name;
        std::array<int, 7> intervals; ///< -1 means the extension is absent
    };

    // allChords: defined below after Chord struct.


    //==============================================================================
    // 8. BUILD A CHORD AT A SPECIFIC ROOT (MIDI)
    //==============================================================================

    /**
     * @brief Build MIDI note numbers for a chord recipe located at rootMidi.
     * @param c Chord definition to use.
     * @param rootMidi MIDI note for chord root (0..127 typical).
     * @param inversion Which chord tone to place in bass (0=root position).
     * @return Array of 7 ints: valid MIDI numbers for present tones, unused slots = -1.
     */
    [[nodiscard]] constexpr std::array<int, 7>
    chordAtRootMidi(const Chord& c, int rootMidi, int inversion = 0) noexcept;


    //==============================================================================
    // 9. REVERSE LOOKUP
    //==============================================================================

    /**
     * @brief Return up to 16 pointers to scales that fully contain `chordMask`.
     * @details Remaining entries are nullptr.
     */
    [[nodiscard]] constexpr std::array<const Scale*, 16>
    scalesForChordMask(NoteSet chordMask) noexcept;

    /**
     * @brief Convenience wrapper: find scales that can contain the notes of a Chord.
     */
    [[nodiscard]] constexpr std::array<const Scale*, 16>
    scalesForChord(const Chord& chord) noexcept;


    //==============================================================================
    // 10. DIATONIC CHORD GENERATION
    //==============================================================================

    /**
     * @brief Internal helpers for diatonic chord generation.
     * @note These helpers are intended for library-internal use.
     */
    namespace detail
    {
        /**
         * @brief Extract the scale's active degrees (in semitones) and expand them so
         * there are always 7 ascending values to operate on.
         *
         * Example: for a pentatonic {0,2,4,7,9} this returns {0,2,4,7,9,12,14}
         * so that stacking thirds (which wraps across the octave) works correctly.
         */
        [[nodiscard]] constexpr std::array<int, 7>
        activeDegrees(NoteSet mask) noexcept;

        /**
         * @brief Interval in semitones between deg[degIdx] and the degree `skip` steps above it
         * when deg[] contains strictly ascending values (possibly >11 when expanded).
         */
        [[nodiscard]] constexpr int interval(
            const std::array<int, 7>& deg, int degIdx, int skip) noexcept;

        /**
         * @brief Safe small string copy used only for compile-time-friendly name building.
         * @note dst must have room for src.size()+1 bytes (including null terminator).
         */
        constexpr void copy(char* dst, std::string_view src, std::size_t dstCapacity) noexcept;
    }

    /**
     * @enum ChordLevel
     * @brief Which chord extensions to generate when building diatonic chords.
     */
    enum class ChordLevel : std::uint8_t
    {
        TriadsOnly   = 0, ///< Generate triads only.
        Triads7      = 1, ///< Generate up to 7th chords.
        Triads79     = 2, ///< Up to 9ths.
        Triads7911   = 3, ///< Up to 11ths.
        Triads791113 = 4  ///< Up to 13ths.
    };

    /**
     * @struct DiatonicChord
     * @brief Result container for a chord generated from a scale degree.
     *
     * name: small NUL-terminated char buffer. view() returns a string_view pointing to the buffer.
     */
    struct DiatonicChord
    {
        std::array<char, 16> name;      ///< NUL-terminated buffer (will contain small strings like "m7", "maj7"...)
        std::array<int, 7>   intervals; ///< Intervals in semitones for R-3-5-7-9-11-13 (-1 = absent)

        /**
         * @brief Safely obtain a string_view of the internal name buffer.
         */
        [[nodiscard]] constexpr std::string_view view() const noexcept;
    };

    /**
     * @brief Generate diatonic chords for a given scale and scale-degree.
     * @param sc Scale to generate from (database scales assume root=C).
     * @param degree Starting degree inside the scale (0..6 corresponds to scale degrees when expanded).
     * @param level Complexity level describing which extensions to include.
     * @return Array of up to 7 DiatonicChord objects. Unused entries will be default-initialized.
     */
    [[nodiscard]] constexpr std::array<DiatonicChord, 7>
    diatonicChords(const Scale& sc, Degree degree, ChordLevel level) noexcept;

    /**
     * @brief Convert a DiatonicChord's interval recipe into MIDI notes given a root MIDI note.
     * @return Array of 7 ints with MIDI notes or -1 for unused positions.
     */
    [[nodiscard]] constexpr std::array<int, 7>
    diatonicChordToMidi(const DiatonicChord& c, int rootMidi) noexcept;


    //==============================================================================
    // 11–12. OCTAVE HELPERS
    //==============================================================================

    /**
     * @brief Parse a note string with optional octave (e.g. "C#4"). Returns pitch-class (0..11).
     * @note If parsing fails this returns 0 (C). Consider using parseNote() directly for error handling.
     */
    [[nodiscard]] constexpr int parseNoteWithOctave(std::string_view note) noexcept;

    /**
     * @brief Extract octave number from a note string like "C#5". Defaults to 4 if not present.
     */
    [[nodiscard]] constexpr int getOctaveFromNote(std::string_view note) noexcept;

    /**
     * @brief Transpose a MIDI note by a number of full octaves (positive or negative).
     */
    [[nodiscard]] constexpr int transposeByOctaves(int midi, int octaveDelta) noexcept;


    /*------------------------------------------------------------
     * DEFINITIONS
     *-----------------------------------------------------------*/

    [[nodiscard]] constexpr ChordTag operator|(ChordTag lhs, ChordTag rhs) noexcept
    {
        using U = std::underlying_type_t<ChordTag>;
        return static_cast<ChordTag>(static_cast<U>(lhs) | static_cast<U>(rhs));
    }

    [[nodiscard]] consteval NoteSet makeMask(
        bool b0, bool b1, bool b2, bool b3,
        bool b4, bool b5, bool b6, bool b7,
        bool b8, bool b9, bool b10, bool b11) noexcept
    {
        return (static_cast<NoteSet>(b0)  << 0)  | (static_cast<NoteSet>(b1)  << 1)  |
               (static_cast<NoteSet>(b2)  << 2)  | (static_cast<NoteSet>(b3)  << 3)  |
               (static_cast<NoteSet>(b4)  << 4)  | (static_cast<NoteSet>(b5)  << 5)  |
               (static_cast<NoteSet>(b6)  << 6)  | (static_cast<NoteSet>(b7)  << 7)  |
               (static_cast<NoteSet>(b8)  << 8)  | (static_cast<NoteSet>(b9)  << 9)  |
               (static_cast<NoteSet>(b10) << 10) | (static_cast<NoteSet>(b11) << 11);
    }

    [[nodiscard]] consteval NoteSet makeMask(std::initializer_list<int> degrees) noexcept
    {
        NoteSet m = 0;
        for (int d : degrees)
            if (d >= 0)
                m |= static_cast<NoteSet>(1u << (d % 12));
        return m & 0x0FFFu;
    }

    inline constexpr std::array<Scale, 61> allScales = [](){
        std::array<Scale, 61> scales{};

        // Major modes
        scales[0]  = {"Ionian",          makeMask({0,2,4,5,7,9,11}), ChordTag::MajorTriad | ChordTag::Major7 | ChordTag::Dominant7};
        scales[1]  = {"Dorian",          makeMask({0,2,3,5,7,9,10}), ChordTag::MinorTriad | ChordTag::Minor7};
        scales[2]  = {"Phrygian",        makeMask({0,1,3,5,7,8,10}), ChordTag::MinorTriad | ChordTag::DiminishedTriad | ChordTag::HalfDim7};
        scales[3]  = {"Lydian",          makeMask({0,2,4,6,7,9,11}), ChordTag::MajorTriad | ChordTag::Major7};
        scales[4]  = {"Mixolydian",      makeMask({0,2,4,5,7,9,10}), ChordTag::MajorTriad | ChordTag::Dominant7};
        scales[5]  = {"Aeolian",         makeMask({0,2,3,5,7,8,10}), ChordTag::MinorTriad | ChordTag::Minor7};
        scales[6]  = {"Locrian",         makeMask({0,1,3,5,6,8,10}), ChordTag::MinorTriad | ChordTag::DiminishedTriad | ChordTag::HalfDim7};

        // Melodic-minor modes
        scales[7]  = {"MelodicMinor",    makeMask({0,2,3,5,7,9,11}), ChordTag::MinorTriad | ChordTag::Minor7};
        scales[8]  = {"Dorianb2",        makeMask({0,1,3,5,7,9,10}), ChordTag::MinorTriad};
        scales[9]  = {"LydianAugmented", makeMask({0,2,4,6,8,9,11}), ChordTag::MajorTriad | ChordTag::AugmentedTriad};
        scales[10] = {"LydianDominant",  makeMask({0,2,4,6,7,9,10}), ChordTag::Dominant7};
        scales[11] = {"Mixolydianb6",    makeMask({0,2,4,5,7,8,10}), ChordTag::MajorTriad | ChordTag::Dominant7};
        scales[12] = {"HalfDiminished",  makeMask({0,2,3,5,6,8,10}), ChordTag::MinorTriad | ChordTag::DiminishedTriad | ChordTag::HalfDim7};
        scales[13] = {"AlteredDominant", makeMask({0,1,3,4,6,8,10}), ChordTag::Dominant7};

        // Harmonic-minor modes
        scales[14] = {"HarmonicMinor",   makeMask({0,2,3,5,7,8,11}), ChordTag::MinorTriad | ChordTag::Minor7};
        scales[15] = {"Locrian6",        makeMask({0,1,3,5,6,9,10}), ChordTag::MinorTriad | ChordTag::DiminishedTriad};
        scales[16] = {"IonianAugmented", makeMask({0,2,4,5,8,9,11}), ChordTag::MajorTriad | ChordTag::AugmentedTriad};
        scales[17] = {"DorianSharp4",    makeMask({0,2,3,6,7,9,10}), ChordTag::MinorTriad};
        scales[18] = {"PhrygianDominant",makeMask({0,1,4,5,7,8,10}), ChordTag::MajorTriad | ChordTag::Dominant7};
        scales[19] = {"LydianSharp2",    makeMask({0,3,4,6,7,9,11}), ChordTag::MajorTriad | ChordTag::Major7};
        scales[20] = {"UltraLocrian",    makeMask({0,1,3,4,6,8,9}), ChordTag::DiminishedTriad | ChordTag::Dim7};

        // Harmonic-major modes
        scales[21] = {"HarmonicMajor",     makeMask({0,2,4,5,7,8,11}), ChordTag::MajorTriad | ChordTag::AugmentedTriad}; 
        scales[22] = {"Dorianb5",          makeMask({0,2,3,5,6,9,11}), ChordTag::MinorTriad | ChordTag::DiminishedTriad};
        scales[23] = {"Phrygianb4",        makeMask({0,1,3,4,6,9,10}), ChordTag::MinorTriad}; 
        scales[24] = {"LydianMinor",       makeMask({0,2,4,6,7,8,10}), ChordTag::MinorTriad | ChordTag::Minor7};
        scales[25] = {"Mixolydianb9",      makeMask({0,1,4,5,7,8,10}), ChordTag::MajorTriad | ChordTag::Dominant7};
        scales[26] = {"LydianAugmented2",  makeMask({0,3,4,6,7,9,11}), ChordTag::MajorTriad | ChordTag::Major7};
        scales[27] = {"LocrianDiminished", makeMask({0,1,3,4,6,7,9}), ChordTag::DiminishedTriad | ChordTag::Dim7}; 

        // Double-harmonic family
        scales[28] = {"DoubleHarmonic",    makeMask({0,1,4,5,7,8,11}), ChordTag::MajorTriad};
        scales[29] = {"HungarianMinor",    makeMask({0,2,3,6,7,8,11}), ChordTag::MinorTriad};
        scales[30] = {"Byzantine",         makeMask({0,1,4,5,7,8,11}), ChordTag::MajorTriad};
        scales[31] = {"Ionian b5",         makeMask({0,2,4,5,6,9,11}), ChordTag::MajorTriad | ChordTag::DiminishedTriad};
        scales[32] = {"Lydian #6",         makeMask({0,2,4,6,7,10,11}), ChordTag::MajorTriad | ChordTag::Major7};
        
        // Pentatonics
        scales[33] = {"MajorPentatonic",   makeMask({0,2,4,7,9}), ChordTag::MajorTriad};
        scales[34] = {"MinorPentatonic",   makeMask({0,3,5,7,10}), ChordTag::MinorTriad};
        scales[35] = {"EgyptianPentatonic",makeMask({0,2,5,7,10}), ChordTag::MajorTriad};
        scales[36] = {"Hirajoshi",         makeMask({0,2,3,7,8}), ChordTag::MinorTriad};
        scales[37] = {"InSen",             makeMask({0,1,5,7,10}), ChordTag::MinorTriad};
        scales[38] = {"Yo",                makeMask({0,4,5,7,11}), ChordTag::MajorTriad};

        // Symmetrical / exotic
        scales[39] = {"WholeTone",         makeMask({0,2,4,6,8,10}), ChordTag::AugmentedTriad};
        scales[40] = {"Chromatic",         makeMask({0,1,2,3,4,5,6,7,8,9,10,11}), ChordTag::All};
        scales[41] = {"Diminished",        makeMask({0,2,3,5,6,8,9,11}), ChordTag::MinorTriad | ChordTag::DiminishedTriad | ChordTag::HalfDim7 | ChordTag::Dim7};
        scales[42] = {"Diminished2",       makeMask({0,1,3,4,6,7,9,10}), ChordTag::MinorTriad | ChordTag::DiminishedTriad | ChordTag::HalfDim7 | ChordTag::Dim7};
        scales[43] = {"Augmented",         makeMask({0,3,4,7,8,11}), ChordTag::AugmentedTriad};

        // Additional Scales (some are enharmonic equivalents of others)
        scales[44] = {"Algerian",          makeMask({0,2,3,6,7,8,11}), ChordTag::MinorTriad};
        scales[45] = {"Arabian",           makeMask({0,2,4,5,6,8,10}), ChordTag::MajorTriad};
        scales[46] = {"Balinese",          makeMask({0,1,3,7,8}), ChordTag::MinorTriad};
        scales[47] = {"Chinese",           makeMask({0,4,6,7,11}), ChordTag::MajorTriad};
        scales[48] = {"Gypsy",             makeMask({0,1,4,5,7,8,10}), ChordTag::MajorTriad | ChordTag::Dominant7};
        scales[49] = {"Hindu",             makeMask({0,2,4,5,7,9,10}), ChordTag::MajorTriad | ChordTag::Dominant7};
        scales[50] = {"Hungarian",         makeMask({0,2,3,6,7,8,11}), ChordTag::MinorTriad};
        scales[51] = {"Japanese",          makeMask({0,1,5,7,8}), ChordTag::MinorTriad};
        scales[52] = {"Javanese",          makeMask({0,1,3,5,7,10}), ChordTag::MajorTriad};
        scales[53] = {"Mongolian",         makeMask({0,2,4,7,9}), ChordTag::MajorTriad};
        scales[54] = {"Neapolitan",        makeMask({0,1,3,5,7,8,11}), ChordTag::MinorTriad};
        scales[55] = {"Oriental",          makeMask({0,1,4,5,6,9,10}), ChordTag::MajorTriad};
        scales[56] = {"Persian",           makeMask({0,1,4,5,6,8,11}), ChordTag::MajorTriad};
        scales[57] = {"Prometheus",        makeMask({0,2,4,6,9,10}), ChordTag::MajorTriad};
        scales[58] = {"Spanish",           makeMask({0,1,3,4,5,7,8,10}), ChordTag::DiminishedTriad};
        scales[59] = {"Tritone",           makeMask({0,1,4,6,7,10}), ChordTag::Dominant7};
        scales[60] = {"Ukrainian",         makeMask({0,2,3,6,7,9,10}), ChordTag::MinorTriad};

        return scales;
    }();

    inline constexpr std::array<std::string_view, 12> sharpNames{
        "C","C#","D","D#","E","F","F#","G","G#","A","A#","B"
    };
    inline constexpr std::array<std::string_view, 12> flatNames{
        "C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"
    };

    inline constexpr std::array<bool,12> useSharpsForRoot{
        1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0
    };

    [[nodiscard]] constexpr std::string_view noteName(int midi, int root) noexcept
    {
        int idx = (midi % 12 + 12) % 12;
        int r   = (root % 12 + 12) % 12;
        return useSharpsForRoot[r] ? sharpNames[idx] : flatNames[idx];
    }

    [[nodiscard]] constexpr std::optional<int> parseNote(std::string_view s) noexcept
    {
        // Added a few enharmonic variants (E#, Fb, B#, Cb) for convenience.
        constexpr std::array<std::pair<std::string_view,int>, 28> lut{{
            {"C",0}, {"B#",0},
            {"C#",1}, {"Db",1},
            {"D",2}, {"D#",3}, {"Eb",3},
            {"E",4}, {"E#",5}, {"Fb",4},
            {"F",5}, {"F#",6}, {"Gb",6},
            {"G",7}, {"G#",8}, {"Ab",8},
            {"A",9}, {"A#",10},{"Bb",10},
            {"B",11}, {"Cb",11}
        }};

        for (const auto& [name,val] : lut)
            if (name.size() == s.size() &&
                std::equal(name.begin(), name.end(), s.begin(),
                           [](char a, char b){ return (a|32) == (b|32); }))
                return val;
        return std::nullopt;
    }

    [[nodiscard]] constexpr NoteSet
    scaleAtRoot(NoteSet base, int root) noexcept
    {
        root = (root % 12 + 12) % 12;
        // Perform circular shift on the 12-bit mask. Use a larger integer to avoid UB
        // during intermediate shifts and mask back to 12 bits.
        std::uint32_t b = static_cast<std::uint32_t>(base) & 0x0FFFu;
        std::uint32_t res = ((b << root) | (b >> (12 - root))) & 0x0FFFu;
        return static_cast<NoteSet>(res);
    }

    inline constexpr std::array<Chord, 15> allChords{{
        {"Major",      {0, 4, 7, -1, -1, -1, -1}},
        {"Minor",      {0, 3, 7, -1, -1, -1, -1}},
        {"Diminished", {0, 3, 6, -1, -1, -1, -1}},
        {"Augmented",  {0, 4, 8, -1, -1, -1, -1}},
        {"Major7",     {0, 4, 7, 11, -1, -1, -1}},
        {"Dominant7",  {0, 4, 7, 10, -1, -1, -1}},
        {"Minor7",     {0, 3, 7, 10, -1, -1, -1}},
        {"HalfDim7",   {0, 3, 6, 10, -1, -1, -1}},
        {"Dim7",       {0, 3, 6,  9, -1, -1, -1}},
        {"Sus2",       {0, 2, 7, -1, -1, -1, -1}},
        {"Sus4",       {0, 5, 7, -1, -1, -1, -1}},
        {"Major9",     {0, 4, 7, 11, 14, -1, -1}},
        {"Dominant9",  {0, 4, 7, 10, 14, -1, -1}},
        {"Minor9",     {0, 3, 7, 10, 14, -1, -1}},
        {"Major13",    {0, 4, 7, 11, 14, 17, 21}}
    }};

    [[nodiscard]] constexpr std::array<int, 7>
    chordAtRootMidi(const Chord& c, int rootMidi, int inversion) noexcept
    {
        std::array<int, 7> notes{};
        int count = 0;

        for (int i = 0; i < 7; ++i)
        {
            const int deg = c.intervals[i];
            if (deg < 0) break;
            notes[count++] = rootMidi + deg;
        }

        if (count == 0) return { -1,-1,-1,-1,-1,-1,-1 };

        if (count > 0)
        {
            inversion = (inversion >= 0) ? (inversion % count) : 0;
            for (int i = 0; i < inversion; ++i) notes[i] += 12;
            std::sort(notes.begin(), notes.begin() + count);
        }

        for (int i = count; i < 7; ++i) notes[i] = -1;
        return notes;
    }

    [[nodiscard]] constexpr std::array<const Scale*, 16>
    scalesForChordMask(NoteSet chordMask) noexcept
    {
        std::array<const Scale*, 16> out{};
        std::size_t idx = 0;
        for (const auto& s : allScales)
            if (((s.mask & chordMask) == chordMask) && idx < out.size())
                out[idx++] = &s;
        // remaining entries are zero-initialized (nullptr)
        return out;
    }

    [[nodiscard]] constexpr std::array<const Scale*, 16>
    scalesForChord(const Chord& chord) noexcept
    {
        NoteSet chordMask = 0;
        for (int d : chord.intervals)
            if (d >= 0) chordMask |= static_cast<NoteSet>(1u << (d % 12));
        return scalesForChordMask(chordMask);
    }

    namespace detail
    {
        [[nodiscard]] constexpr std::array<int, 7>
        activeDegrees(NoteSet mask) noexcept
        {
            // Collect active degrees in ascending order within the octave.
            std::array<int, 12> temp{};
            int tcount = 0;
            for (int i = 0; i < 12; ++i)
                if (mask & (1u << i)) temp[tcount++] = i;

            std::array<int, 7> out{};
            if (tcount == 0)
            {
                // empty scale -> return default ascending zero-filled values
                for (int i = 0; i < 7; ++i) out[i] = i;
                return out;
            }

            // Fill out[] by repeating the pattern across octaves so that values are strictly
            // ascending and usable for "stacking thirds" operations.
            for (int i = 0; i < 7; ++i)
            {
                int idx = i % tcount;
                int octave = i / tcount;
                out[i] = temp[idx] + octave * 12;
            }
            return out;
        }

        [[nodiscard]] constexpr int interval(
            const std::array<int, 7>& deg, int degIdx, int skip) noexcept
        {
            // deg is expected to contain strictly non-decreasing ascending values (possibly >11)
            int a = deg[degIdx];
            int b = deg[(degIdx + skip) % 7];
            // If b is not strictly > a (edge cases), normalize by adding 12 until it's above.
            while (b <= a) b += 12;
            return b - a;
        }

        constexpr void copy(char* dst, std::string_view src, std::size_t dstCapacity) noexcept
        {
            // Safe copy that will not overflow the destination if dstCapacity is respected by caller.
            std::size_t toCopy = src.size();
            if (toCopy + 1u > dstCapacity) toCopy = (dstCapacity > 0) ? dstCapacity - 1u : 0u;
            for (std::size_t i = 0; i < toCopy; ++i) dst[i] = src[i];
            if (dstCapacity > 0) dst[toCopy] = '\0';
        }
    } // namespace detail


    [[nodiscard]] constexpr std::array<DiatonicChord, 7>
    diatonicChords(const Scale& sc, Degree degree, ChordLevel level) noexcept
    {
        const auto deg = detail::activeDegrees(sc.mask);
        if (degree < 0 || degree >= 7) return {};

        std::array<DiatonicChord, 7> out{};
        std::size_t idx = 0;

        // Compute stack-of-thirds intervals (third,fifth,7th,9th,11th,13th) relative to the degree.
        const int third      = detail::interval(deg, degree, 2);
        const int fifth      = detail::interval(deg, degree, 4);
        const int seventh    = detail::interval(deg, degree, 6);
        const int ninth      = detail::interval(deg, degree, 1);
        const int eleventh   = detail::interval(deg, degree, 3);
        const int thirteenth = detail::interval(deg, degree, 5);

        auto push = [&](std::string_view baseName, ChordLevel lvl)
        {
            DiatonicChord c{};
            // Fill intervals: R-3-5-7-9-11-13 (use -1 for absent)
            c.intervals = {0, third, fifth,
                           (lvl >= ChordLevel::Triads7)      ? seventh    : -1,
                           (lvl >= ChordLevel::Triads79)     ? ninth      : -1,
                           (lvl >= ChordLevel::Triads7911)   ? eleventh   : -1,
                           (lvl >= ChordLevel::Triads791113) ? thirteenth : -1};

            char buf[20]{};
            std::size_t pos = 0;
            // safe copy helper
            detail::copy(buf + pos, baseName, sizeof(buf) - pos);
            pos += baseName.size();

            if (lvl >= ChordLevel::Triads79)     { detail::copy(buf + pos, "(9)",  sizeof(buf) - pos);  pos += 3; }
            if (lvl >= ChordLevel::Triads7911)   { detail::copy(buf + pos, "(11)", sizeof(buf) - pos);  pos += 4; }
            if (lvl >= ChordLevel::Triads791113) { detail::copy(buf + pos, "(13)", sizeof(buf) - pos);  pos += 4; }

            detail::copy(c.name.data(), std::string_view(buf, pos), c.name.size());
            out[idx++] = c;
        };

        /* Determine base triad quality */
        std::string_view base;
        if      (third == 4 && fifth == 7) base = "M";
        else if (third == 3 && fifth == 7) base = "m";
        else if (third == 3 && fifth == 6) base = "dim";
        else if (third == 4 && fifth == 8) base = "aug";
        else                               base = "?";

        /* Determine 7th chord symbol */
        std::string_view name7;
        if      (base == "dim" && seventh == 10) name7 = "m7b5";
        else if (base == "dim" && seventh == 9)  name7 = "dim7";
        else if (base == "M"   && seventh == 10) name7 = "7";
        else if (base == "M"   && seventh == 11) name7 = "maj7";
        else if (base == "m"   && seventh == 10) name7 = "m7";
        else if (base == "m"   && seventh == 11) name7 = "m(maj7)";
        else if (base == "aug" && seventh == 10) name7 = "aug7";
        else                                     name7 = base;

        switch (level)
        {
            case ChordLevel::TriadsOnly:   push(base,  ChordLevel::TriadsOnly); break;
            case ChordLevel::Triads7:      push(name7, ChordLevel::Triads7);    break;
            case ChordLevel::Triads79:     push(name7, ChordLevel::Triads79);   break;
            case ChordLevel::Triads7911:   push(name7, ChordLevel::Triads7911); break;
            case ChordLevel::Triads791113: push(name7, ChordLevel::Triads791113); break;
        }
        return out;
    }

    [[nodiscard]] constexpr std::array<int, 7>
    diatonicChordToMidi(const DiatonicChord& c, int rootMidi) noexcept
    {
        std::array<int, 7> notes{};
        for (std::size_t i = 0; i < 7; ++i)
            notes[i] = (c.intervals[i] >= 0) ? rootMidi + c.intervals[i] : -1;
        return notes;
    }

    [[nodiscard]] constexpr int parseNoteWithOctave(std::string_view note) noexcept
    {
        std::size_t len = note.size();
        while (len > 0 && note[len - 1] >= '0' && note[len - 1] <= '9')
            --len;

        std::string_view base = note.substr(0, len);
        if (auto pc = parseNote(base)) return *pc;
        return 0;
    }

    [[nodiscard]] constexpr int getOctaveFromNote(std::string_view note) noexcept
    {
        std::size_t len = note.size();
        int oct = 4;
        while (len && note[len - 1] >= '0' && note[len - 1] <= '9')
            --len;
        if (len < note.size())
        {
            int parsed = 0;
            for (std::size_t i = len; i < note.size(); ++i)
                parsed = parsed * 10 + (note[i] - '0');
            oct = parsed;
        }
        return oct;
    }

    [[nodiscard]] constexpr int transposeByOctaves(int midi, int octaveDelta) noexcept
    {
        return midi + octaveDelta * 12;
    }

} // namespace harmony
} // namespace dspark
