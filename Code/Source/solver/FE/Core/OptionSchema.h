/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CORE_OPTIONSCHEMA_H
#define SVMP_FE_CORE_OPTIONSCHEMA_H

/**
 * @file OptionSchema.h
 * @brief Declarative, typed description of an option struct and of its
 *        text input (decision D47, refactor step S13).
 *
 * One OptionSchema<T> declares every key of an option struct T once: its
 * canonical spelling and the ordered spellings a parser accepts, the codec
 * that turns text into a typed value and back, the location in T it fills,
 * the documentation, the cross-key rules ("implies", "conflicts", custom
 * checks) and the retired keys of the input.  The same declaration drives
 *
 *   - resolve():        text input -> typed options, with the provenance of
 *                       every key (default, explicit, deprecated spelling,
 *                       implied by another key, derived after resolution)
 *                       and the deprecated input that was accepted;
 *   - serialize():      typed options -> canonical text, in declaration order;
 *   - toJson():         the resolved options with provenance, for
 *                       effective-configuration artifacts;
 *   - documentation():  one entry per key for generated documentation;
 *   - spellings():      the accepted spellings, for XML acceptance lists.
 *
 * Parsing keeps the behaviour of the hand-written parsers as per-key
 * policies rather than one global rule:
 *
 *   - the first defined spelling in the declared order wins, unless the key
 *     rejects two defined spellings (DuplicatePolicy);
 *   - booleans use a strict or a relaxed lexicon (BoolLexicon);
 *   - numbers must consume the whole (trimmed) text;
 *   - an empty value counts as absent, is handed to the codec, or is an
 *     error (EmptyPolicy; each codec has a default);
 *   - "implies" sets a key that the input left alone;
 *   - modes of a resolve call (for example a legacy input contract) can
 *     enable or waive implications, conflicts and rules, and can make a
 *     retired key acceptable (it is then ignored and reported).
 *
 * The schema is physics-agnostic and depends only on the standard library and
 * FE/Core types; physics schemas live next to their option structs.
 */

#include "Core/Types.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace options {

// ---------------------------------------------------------------------------
// Errors and text helpers
// ---------------------------------------------------------------------------

/// An input that violates the schema: unknown, malformed, out of range,
/// retired, duplicated or inconsistent.  what() names the element and the key.
class OptionError : public std::runtime_error {
public:
    OptionError(std::string element_name, std::string key_name, const std::string& reason)
        : std::runtime_error("[" + element_name + "] " +
                             (key_name.empty() ? std::string{} : key_name + ": ") + reason),
          element_(std::move(element_name)),
          key_(std::move(key_name))
    {
    }

    [[nodiscard]] const std::string& element() const noexcept { return element_; }
    [[nodiscard]] const std::string& key() const noexcept { return key_; }

private:
    std::string element_;
    std::string key_;
};

namespace detail {

[[nodiscard]] inline std::string trimmed(std::string_view text)
{
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1u])) != 0) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

/// Lower case without '_', '-' and white space: the token comparison of the
/// enumeration lexicons ("LevelSetNegative" == "level_set negative").
[[nodiscard]] inline std::string normalizedToken(std::string_view text)
{
    std::string out;
    out.reserve(text.size());
    for (const char ch : text) {
        const auto uch = static_cast<unsigned char>(ch);
        if (ch == '_' || ch == '-' || std::isspace(uch) != 0) {
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(uch)));
    }
    return out;
}

[[nodiscard]] inline std::string lowerCase(std::string_view text)
{
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return out;
}

/// Shortest decimal text that reads back to the same double.
[[nodiscard]] inline std::string formatReal(double value)
{
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0.0 ? "inf" : "-inf";
    }
    std::array<char, 64> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    return std::string(buffer.data(), result.ptr);
}

[[nodiscard]] inline std::string jsonQuoted(std::string_view text)
{
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(text.size() + 2u);
    out.push_back('"');
    for (const char ch : text) {
        const auto uch = static_cast<unsigned char>(ch);
        switch (ch) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\t': out += "\\t"; break;
        case '\r': out += "\\r"; break;
        default:
            if (uch < 0x20) {
                out += "\\u00";
                out.push_back(hex[uch >> 4]);
                out.push_back(hex[uch & 0x0f]);
            } else {
                out.push_back(ch);
            }
            break;
        }
    }
    out.push_back('"');
    return out;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/// One supplied input value: the spelling the input used and its raw text.
struct OptionInputEntry {
    std::string spelling{};
    std::string text{};
};

/// The key/value input of one configuration element (for example one XML
/// element), in input order.  Only supplied values belong here; an XML key
/// that was given with an empty body is supplied with empty text.
class OptionInput {
public:
    OptionInput() = default;
    explicit OptionInput(std::vector<OptionInputEntry> entries) : entries_(std::move(entries)) {}

    void add(std::string spelling, std::string text)
    {
        entries_.push_back(OptionInputEntry{std::move(spelling), std::move(text)});
    }

    /// The first entry of this spelling, or nullptr.
    [[nodiscard]] const OptionInputEntry* find(std::string_view spelling) const
    {
        for (const auto& entry : entries_) {
            if (entry.spelling == spelling) {
                return &entry;
            }
        }
        return nullptr;
    }

    [[nodiscard]] const std::vector<OptionInputEntry>& entries() const noexcept { return entries_; }

private:
    std::vector<OptionInputEntry> entries_{};
};

// ---------------------------------------------------------------------------
// Key policies
// ---------------------------------------------------------------------------

/// What happens when an input defines two spellings of one key.
enum class DuplicatePolicy : std::uint8_t {
    /// The first defined spelling in the declared order wins.
    FirstDefinedWins,
    /// Two supplied spellings are an error, even when one of them is empty.
    Reject,
};

/// What an empty (white-space only) value means.
enum class EmptyPolicy : std::uint8_t {
    /// The spelling counts as absent; the next spelling is tried.
    Absent,
    /// The empty text goes to the codec like any other text (a relaxed
    /// boolean reads it as false; a number rejects it).
    Parse,
    /// An empty value is an error.
    Error,
};

namespace detail {

/// A codec may declare `static constexpr EmptyPolicy empty_policy`; codecs
/// that do not are treated as EmptyPolicy::Absent.
template <class Codec, class = void>
struct CodecEmptyPolicy {
    static constexpr EmptyPolicy value = EmptyPolicy::Absent;
};
template <class Codec>
struct CodecEmptyPolicy<Codec, std::void_t<decltype(Codec::empty_policy)>> {
    static constexpr EmptyPolicy value = Codec::empty_policy;
};

} // namespace detail

// ---------------------------------------------------------------------------
// Codecs: text <-> typed value, plus a description for documentation.
//
// A codec has `value_type`, `value_type parse(std::string_view, std::string&
// error) const` (sets a non-empty error on failure), `std::string
// format(const value_type&) const` and `std::string describe() const`.
// ---------------------------------------------------------------------------

enum class BoolLexicon : std::uint8_t {
    /// true/false, 1/0, yes/no, on/off (any case); any other text is an error.
    Strict,
    /// true/1/yes/on (any case) is true and any other text, including empty
    /// text, is false.  Kept for keys whose existing parsers behave this way.
    Relaxed,
};

struct BoolCodec {
    using value_type = bool;
    static constexpr EmptyPolicy empty_policy = EmptyPolicy::Parse;
    BoolLexicon lexicon{BoolLexicon::Strict};

    [[nodiscard]] bool parse(std::string_view text, std::string& error) const
    {
        const auto token = detail::lowerCase(text);
        if (token == "true" || token == "1" || token == "yes" || token == "on") {
            return true;
        }
        if (lexicon == BoolLexicon::Relaxed) {
            return false;
        }
        if (token == "false" || token == "0" || token == "no" || token == "off") {
            return false;
        }
        error = "expected true or false, got '" + std::string(text) + "'";
        return false;
    }
    [[nodiscard]] std::string format(bool value) const { return value ? "true" : "false"; }
    [[nodiscard]] std::string describe() const { return "bool"; }
};

/// A decimal integer with an optional sign; the whole text must be consumed.
struct IntCodec {
    using value_type = int;
    static constexpr EmptyPolicy empty_policy = EmptyPolicy::Parse;
    std::optional<int> minimum{};
    std::optional<int> maximum{};

    [[nodiscard]] int parse(std::string_view text, std::string& error) const
    {
        int value = 0;
        const auto* first = text.data();
        const auto* last = text.data() + text.size();
        if (first != last && *first == '+') {
            ++first;
            if (first != last && *first == '-') {
                first = last; // "+-5" is not a number
            }
        }
        const auto result = std::from_chars(first, last, value);
        if (result.ec == std::errc::result_out_of_range) {
            error = "integer '" + std::string(text) + "' out of range";
            return 0;
        }
        if (first == last || result.ec != std::errc{} || result.ptr != last) {
            error = "expected an integer, got '" + std::string(text) + "'";
            return 0;
        }
        if ((minimum && value < *minimum) || (maximum && value > *maximum)) {
            error = "integer " + std::to_string(value) + " outside " + range();
        }
        return value;
    }
    [[nodiscard]] std::string format(int value) const { return std::to_string(value); }
    [[nodiscard]] std::string range() const
    {
        return "[" + (minimum ? std::to_string(*minimum) : std::string("-inf")) + ", " +
               (maximum ? std::to_string(*maximum) : std::string("inf")) + "]";
    }
    [[nodiscard]] std::string describe() const
    {
        return (minimum || maximum) ? "int in " + range() : "int";
    }
};

/// A real number in std::stod spelling (decimal or scientific, optional
/// sign); the whole text must be consumed.
struct RealCodec {
    using value_type = Real;
    static constexpr EmptyPolicy empty_policy = EmptyPolicy::Parse;
    std::optional<double> minimum{};
    std::optional<double> maximum{};
    /// Exclusive bounds when true (for example "positive": minimum 0, open).
    bool open_minimum{false};
    bool open_maximum{false};
    bool require_finite{true};

    [[nodiscard]] Real parse(std::string_view text, std::string& error) const
    {
        const std::string copy(text);
        std::size_t consumed = 0;
        double value = 0.0;
        try {
            value = std::stod(copy, &consumed);
        } catch (const std::exception&) {
            consumed = 0;
        }
        if (copy.empty() || consumed != copy.size()) {
            error = "expected a number, got '" + copy + "'";
            return Real{0};
        }
        if (require_finite && !std::isfinite(value)) {
            error = "expected a finite number, got '" + copy + "'";
            return Real{0};
        }
        const bool below = minimum && (open_minimum ? !(value > *minimum) : value < *minimum);
        const bool above = maximum && (open_maximum ? !(value < *maximum) : value > *maximum);
        if (below || above) {
            error = "number " + detail::formatReal(value) + " outside " + range();
        }
        return static_cast<Real>(value);
    }
    [[nodiscard]] std::string format(Real value) const
    {
        return detail::formatReal(static_cast<double>(value));
    }
    [[nodiscard]] std::string range() const
    {
        return std::string(open_minimum ? "(" : "[") +
               (minimum ? detail::formatReal(*minimum) : std::string("-inf")) + ", " +
               (maximum ? detail::formatReal(*maximum) : std::string("inf")) +
               (open_maximum ? ")" : "]");
    }
    [[nodiscard]] std::string describe() const
    {
        return (minimum || maximum) ? "real in " + range() : "real";
    }
};

struct StringCodec {
    using value_type = std::string;
    static constexpr EmptyPolicy empty_policy = EmptyPolicy::Absent;
    [[nodiscard]] std::string parse(std::string_view text, std::string& /*error*/) const
    {
        return std::string(text);
    }
    [[nodiscard]] std::string format(const std::string& value) const { return value; }
    [[nodiscard]] std::string describe() const { return "text"; }
};

/// Exactly three reals separated by commas and/or white space.  Each
/// component must be a whole number token ("1 2 3x" is an error).
struct Vector3Codec {
    using value_type = std::array<Real, 3>;
    static constexpr EmptyPolicy empty_policy = EmptyPolicy::Absent;

    [[nodiscard]] value_type parse(std::string_view text, std::string& error) const
    {
        value_type out{Real{0}, Real{0}, Real{0}};
        std::vector<std::string> parts;
        std::string current;
        for (const char ch : text) {
            if (ch == ',' || std::isspace(static_cast<unsigned char>(ch)) != 0) {
                if (!current.empty()) {
                    parts.push_back(current);
                    current.clear();
                }
            } else {
                current.push_back(ch);
            }
        }
        if (!current.empty()) {
            parts.push_back(current);
        }
        if (parts.size() != 3u) {
            error = "expected three numbers, got '" + std::string(text) + "'";
            return out;
        }
        const RealCodec component{};
        for (std::size_t i = 0; i < 3u; ++i) {
            out[i] = component.parse(parts[i], error);
            if (!error.empty()) {
                return out;
            }
        }
        return out;
    }
    [[nodiscard]] std::string format(const value_type& value) const
    {
        const RealCodec component{};
        return component.format(value[0]) + " " + component.format(value[1]) + " " +
               component.format(value[2]);
    }
    [[nodiscard]] std::string describe() const { return "3 reals"; }
};

/// A semicolon-separated, non-empty list of values of another codec
/// ("80; 81").  Empty entries between separators are skipped.
template <class ElementCodec>
struct ListCodec {
    using element_type = typename ElementCodec::value_type;
    using value_type = std::vector<element_type>;
    static constexpr EmptyPolicy empty_policy = EmptyPolicy::Absent;
    ElementCodec element{};

    [[nodiscard]] value_type parse(std::string_view text, std::string& error) const
    {
        value_type out;
        std::size_t start = 0;
        while (start <= text.size()) {
            const auto end = text.find(';', start);
            const auto piece = detail::trimmed(
                text.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start));
            if (!piece.empty()) {
                out.push_back(element.parse(piece, error));
                if (!error.empty()) {
                    return out;
                }
            }
            if (end == std::string_view::npos) {
                break;
            }
            start = end + 1u;
        }
        if (out.empty()) {
            error = "expected a ';'-separated list, got '" + std::string(text) + "'";
        }
        return out;
    }
    [[nodiscard]] std::string format(const value_type& value) const
    {
        std::string out;
        for (std::size_t i = 0; i < value.size(); ++i) {
            if (i != 0u) {
                out += "; ";
            }
            out += element.format(value[i]);
        }
        return out;
    }
    [[nodiscard]] std::string describe() const { return "list of " + element.describe(); }
};

/// One value of an enumeration: the value, its canonical token and the
/// synonyms an input may use.  Tokens compare after normalization (case,
/// '_', '-' and white space ignored).
template <class E>
struct EnumToken {
    E value{};
    std::string canonical{};
    std::vector<std::string> synonyms{};
};

template <class E>
struct EnumCodec {
    using value_type = E;
    static constexpr EmptyPolicy empty_policy = EmptyPolicy::Absent;
    std::vector<EnumToken<E>> tokens{};

    [[nodiscard]] E parse(std::string_view text, std::string& error) const
    {
        const auto token = detail::normalizedToken(text);
        for (const auto& entry : tokens) {
            if (detail::normalizedToken(entry.canonical) == token) {
                return entry.value;
            }
            for (const auto& synonym : entry.synonyms) {
                if (detail::normalizedToken(synonym) == token) {
                    return entry.value;
                }
            }
        }
        error = "expected one of " + describe() + ", got '" + std::string(text) + "'";
        return tokens.empty() ? E{} : tokens.front().value;
    }
    [[nodiscard]] std::string format(E value) const
    {
        for (const auto& entry : tokens) {
            if (entry.value == value) {
                return entry.canonical;
            }
        }
        return "<unnamed>";
    }
    [[nodiscard]] std::string describe() const
    {
        std::string out;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (i != 0u) {
                out += " | ";
            }
            out += tokens[i].canonical;
        }
        return out;
    }
};

// ---------------------------------------------------------------------------
// Key settings, resolve context, provenance and the resolved result
// ---------------------------------------------------------------------------

/// One accepted spelling of a key.  A deprecated spelling is accepted and
/// reported (warning, artifact) with the canonical spelling as replacement.
struct OptionSpelling {
    std::string name{};
    bool deprecated{false};
};

/// Per-key settings of OptionSchema::key().
struct KeySettings {
    /// Accepted spellings in lookup order.  Empty: the canonical spelling only.
    /// The canonical spelling must be one of them and must not be deprecated.
    std::vector<OptionSpelling> spellings{};
    DuplicatePolicy duplicates{DuplicatePolicy::FirstDefinedWins};
    /// Unset: the codec's default (EmptyPolicy::Parse for booleans and
    /// numbers, EmptyPolicy::Absent for text, enumerations, vectors, lists).
    std::optional<EmptyPolicy> empty{};
    /// One line for the documentation.
    std::string doc{};
    /// Replaces the codec's reason when the value does not parse or is out
    /// of range.
    std::string invalid_message{};
    /// The key of the artifact entry.  Empty: the canonical spelling.
    std::string artifact_key{};
};

/// Modes are named switches of one resolve call (for example a legacy input
/// contract).  A schema never defines them; it only refers to them.
struct ResolveContext;

/// Restricts an implication, a conflict, a rule or the acceptance of a retired
/// key to resolve calls with or without a mode.
struct ModeCondition {
    /// Applies only when this mode is active (empty: no requirement).
    std::string only_in{};
    /// Does not apply when this mode is active (empty: no exclusion).
    std::string except_in{};

    [[nodiscard]] bool appliesIn(const ResolveContext& context) const;
};

/// What resolve() does with input spellings the schema does not know.
enum class UnknownKeyPolicy : std::uint8_t {
    /// Unknown spellings are left to other parsers of the same element.
    Ignore,
    /// Any unknown spelling is an error (fail-closed elements).
    Reject,
};

struct ResolveContext {
    UnknownKeyPolicy unknown{UnknownKeyPolicy::Ignore};
    std::vector<std::string> modes{};

    [[nodiscard]] bool hasMode(std::string_view mode) const
    {
        return std::find(modes.begin(), modes.end(), mode) != modes.end();
    }
};

inline bool ModeCondition::appliesIn(const ResolveContext& context) const
{
    return (only_in.empty() || context.hasMode(only_in)) &&
           (except_in.empty() || !context.hasMode(except_in));
}

/// Settings of OptionSchema::implies().
struct ImplicationSettings {
    /// Imply only when the trigger key resolves to this value (compared in
    /// the trigger codec's canonical text).  Unset: any value.
    std::optional<std::string> when_trigger_is{};
    /// When true the trigger must be given by the input; when false a
    /// default or implied trigger value counts as well.
    bool trigger_must_be_explicit{true};
    ModeCondition modes{};
};

/// How a retired key can still be accepted (it is then ignored and reported
/// as a deprecation).  Without either it is always an error.
struct RetiredAcceptance {
    /// The value decks may still restate (for example "false" for a retired
    /// switch), compared after normalization.
    std::optional<std::string> value{};
    /// Any value is accepted in this mode (for example a legacy contract that
    /// may keep an archived, ignored value).
    std::string any_value_in_mode{};
};

enum class OptionSource : std::uint8_t {
    Default,
    Explicit,
    DeprecatedSpelling,
    Implied,
    Derived,
};

[[nodiscard]] inline const char* optionSourceName(OptionSource source) noexcept
{
    switch (source) {
    case OptionSource::Default: return "default";
    case OptionSource::Explicit: return "explicit";
    case OptionSource::DeprecatedSpelling: return "deprecated_spelling";
    case OptionSource::Implied: return "implied";
    case OptionSource::Derived: return "derived";
    }
    return "unknown";
}

/// Where the value of one key came from.
struct OptionProvenance {
    std::string key{};
    OptionSource source{OptionSource::Default};
    /// The input spelling (Explicit, DeprecatedSpelling).
    std::string spelling{};
    /// The implying key (Implied) or the reason given to derive() (Derived).
    std::string origin{};
};

/// A deprecated input that was accepted: a deprecated spelling, or a retired
/// key whose value was acceptable.  The replacement is empty for retired keys.
struct OptionDeprecation {
    std::string spelling{};
    std::string replacement{};
    std::string message{};
};

template <class T>
struct ResolvedOptions {
    T value{};
    /// One entry per declared key, in declaration order.
    std::vector<OptionProvenance> provenance{};
    std::vector<OptionDeprecation> deprecations{};

    [[nodiscard]] const OptionProvenance* find(std::string_view key) const
    {
        for (const auto& item : provenance) {
            if (item.key == key) {
                return &item;
            }
        }
        return nullptr;
    }
    /// True when the input gave the key (under any spelling).
    [[nodiscard]] bool isExplicit(std::string_view key) const
    {
        const auto* item = find(key);
        return item != nullptr &&
               (item->source == OptionSource::Explicit ||
                item->source == OptionSource::DeprecatedSpelling);
    }
    /// Record that code after resolve() set the key (for example from the
    /// mesh or another element).  Returns false for an undeclared key.
    bool derive(std::string_view key, std::string origin)
    {
        for (auto& item : provenance) {
            if (item.key == key) {
                item.source = OptionSource::Derived;
                item.spelling.clear();
                item.origin = std::move(origin);
                return true;
            }
        }
        return false;
    }
};

/// One documented key.
struct OptionDocumentation {
    std::string key{};
    std::vector<std::string> spellings{};
    std::vector<std::string> deprecated_spellings{};
    std::string type{};
    std::string default_value{};
    std::string doc{};
};

/// An accessor for a member of a member: nestedMember(&T::guards, &Guards::n)
/// selects options.guards.n.
template <class Root, class Outer, class Inner>
[[nodiscard]] auto nestedMember(Outer Root::*outer, Inner Outer::*inner)
{
    return [outer, inner](Root& object) -> Inner& { return (object.*outer).*inner; };
}

// ---------------------------------------------------------------------------
// The schema
// ---------------------------------------------------------------------------

template <class T>
class OptionSchema {
public:
    explicit OptionSchema(std::string element_name) : element_(std::move(element_name)) {}

    [[nodiscard]] const std::string& element() const noexcept { return element_; }

    /// Declare a key.  `access` selects the location in T that the key fills:
    /// a member pointer (&T::member), nestedMember(...), or any callable
    /// T& -> V&.  An accessor must only select a location, never modify.
    template <class Access, class Codec>
    OptionSchema& key(std::string canonical, Access access, Codec codec, KeySettings settings = {})
    {
        static_assert(std::is_invocable_v<const Access&, T&>,
                      "OptionSchema::key: the accessor must be callable with T&");
        using Location = std::invoke_result_t<const Access&, T&>;
        static_assert(std::is_lvalue_reference_v<Location> &&
                          !std::is_const_v<std::remove_reference_t<Location>>,
                      "OptionSchema::key: the accessor must return a mutable reference into T");
        using V = std::remove_reference_t<Location>;
        using CodecValue = typename Codec::value_type;
        static_assert(std::is_convertible_v<CodecValue, V>,
                      "OptionSchema::key: the codec value type must convert to the member type");

        if (findKey(canonical) != nullptr) {
            throw std::logic_error("OptionSchema '" + element_ + "': key '" + canonical +
                                   "' declared twice");
        }
        Key entry;
        entry.canonical = canonical;
        entry.spellings = settings.spellings.empty()
                              ? std::vector<OptionSpelling>{OptionSpelling{canonical, false}}
                              : std::move(settings.spellings);
        const bool has_canonical = std::any_of(
            entry.spellings.begin(), entry.spellings.end(),
            [&](const OptionSpelling& s) { return s.name == canonical && !s.deprecated; });
        if (!has_canonical) {
            throw std::logic_error("OptionSchema '" + element_ + "': key '" + canonical +
                                   "' must list its canonical spelling as not deprecated");
        }
        for (std::size_t i = 0; i < entry.spellings.size(); ++i) {
            const auto& name = entry.spellings[i].name;
            const bool repeated = std::any_of(
                entry.spellings.begin(), entry.spellings.begin() + static_cast<std::ptrdiff_t>(i),
                [&](const OptionSpelling& s) { return s.name == name; });
            if (name.empty() || repeated || acceptsSpelling(name) || isRetiredSpelling(name)) {
                throw std::logic_error("OptionSchema '" + element_ + "': spelling '" + name +
                                       "' of key '" + canonical +
                                       "' is empty, repeated or belongs to another key");
            }
        }
        entry.duplicates = settings.duplicates;
        entry.empty = settings.empty.value_or(detail::CodecEmptyPolicy<Codec>::value);
        entry.doc = std::move(settings.doc);
        entry.invalid_message = std::move(settings.invalid_message);
        entry.artifact_key =
            settings.artifact_key.empty() ? canonical : std::move(settings.artifact_key);
        entry.type = codec.describe();
        auto shared_codec = std::make_shared<const Codec>(std::move(codec));
        entry.apply = [shared_codec, access](T& object, std::string_view text, std::string& error) {
            auto parsed = shared_codec->parse(text, error);
            if (error.empty()) {
                std::invoke(access, object) = static_cast<V>(std::move(parsed));
            }
        };
        entry.format = [shared_codec, access](const T& object) {
            // The accessor only selects a location; reading through the
            // mutable reference does not modify the object.
            const V& location = std::invoke(access, const_cast<T&>(object));
            return shared_codec->format(static_cast<CodecValue>(location));
        };
        keys_.push_back(std::move(entry));
        return *this;
    }

    /// Set the target key to `text` (parsed with the target's codec) when the
    /// trigger key is given, unless the input gave the target too.
    /// Implications apply in declaration order, after the input is read.
    OptionSchema& implies(std::string trigger, std::string target, std::string text,
                          ImplicationSettings settings = {})
    {
        const auto& target_key = requireKey(target, "implies");
        const auto& trigger_key = requireKey(trigger, "implies");
        Implication implication;
        implication.text = canonicalText(target_key, text, "implied value of '" + target + "'");
        if (settings.when_trigger_is) {
            implication.when_trigger_is = canonicalText(
                trigger_key, *settings.when_trigger_is, "trigger value of '" + trigger + "'");
        }
        implication.trigger = std::move(trigger);
        implication.target = std::move(target);
        implication.trigger_must_be_explicit = settings.trigger_must_be_explicit;
        implication.modes = std::move(settings.modes);
        implications_.push_back(std::move(implication));
        return *this;
    }

    /// Two keys that must not both be given.
    OptionSchema& conflicts(std::string first, std::string second, std::string reason,
                            ModeCondition modes = {})
    {
        (void)requireKey(first, "conflicts");
        (void)requireKey(second, "conflicts");
        conflicts_.push_back(
            Conflict{std::move(first), std::move(second), std::move(reason), std::move(modes)});
        return *this;
    }

    /// A check of the resolved options; the check returns an empty string when
    /// it passes and the reason otherwise.  Rules run after implications and
    /// conflicts, in declaration order.
    OptionSchema& rule(std::string name,
                       std::function<std::string(const ResolvedOptions<T>&)> check,
                       ModeCondition modes = {})
    {
        rules_.push_back(Rule{std::move(name), std::move(check), std::move(modes)});
        return *this;
    }

    /// Spellings of a retired key.  A supplied value is an error with the
    /// message unless the acceptance allows it; an accepted value is ignored
    /// and reported as a deprecation with the message.
    OptionSchema& retired(std::vector<std::string> names, std::string message,
                          RetiredAcceptance acceptance = {})
    {
        for (const auto& name : names) {
            if (name.empty() || acceptsSpelling(name) || isRetiredSpelling(name)) {
                throw std::logic_error("OptionSchema '" + element_ + "': retired spelling '" +
                                       name + "' is empty, repeated or an active spelling");
            }
        }
        retired_.push_back(Retired{std::move(names), std::move(message), std::move(acceptance)});
        return *this;
    }

    /// Resolve an input into typed options, starting from the given defaults.
    [[nodiscard]] ResolvedOptions<T> resolve(const OptionInput& input, T defaults = T{},
                                             const ResolveContext& context = {}) const
    {
        ResolvedOptions<T> out;
        out.value = std::move(defaults);
        out.provenance.reserve(keys_.size());

        if (context.unknown == UnknownKeyPolicy::Reject) {
            for (const auto& entry : input.entries()) {
                if (!acceptsSpelling(entry.spelling) && !isRetiredSpelling(entry.spelling)) {
                    throw OptionError(element_, entry.spelling, "unknown key");
                }
            }
        }
        for (const auto& item : retired_) {
            for (const auto& spelling : item.spellings) {
                const auto* entry = input.find(spelling);
                if (entry == nullptr) {
                    continue;
                }
                const bool value_accepted =
                    item.acceptance.value &&
                    detail::normalizedToken(entry->text) ==
                        detail::normalizedToken(*item.acceptance.value);
                const bool mode_accepted = !item.acceptance.any_value_in_mode.empty() &&
                                           context.hasMode(item.acceptance.any_value_in_mode);
                if (!value_accepted && !mode_accepted) {
                    throw OptionError(element_, spelling, item.message);
                }
                out.deprecations.push_back(OptionDeprecation{spelling, std::string{}, item.message});
            }
        }

        for (const auto& key_entry : keys_) {
            out.provenance.push_back(readKey(key_entry, input, out));
        }

        for (const auto& implication : implications_) {
            if (!implication.modes.appliesIn(context) || out.isExplicit(implication.target)) {
                continue;
            }
            if (implication.trigger_must_be_explicit && !out.isExplicit(implication.trigger)) {
                continue;
            }
            if (implication.when_trigger_is &&
                findKey(implication.trigger)->format(out.value) != *implication.when_trigger_is) {
                continue;
            }
            // The text was checked when the implication was declared.
            std::string error;
            findKey(implication.target)->apply(out.value, implication.text, error);
            for (auto& provenance : out.provenance) {
                if (provenance.key == implication.target) {
                    provenance.source = OptionSource::Implied;
                    provenance.spelling.clear();
                    provenance.origin = implication.trigger;
                }
            }
        }

        for (const auto& conflict : conflicts_) {
            if (conflict.modes.appliesIn(context) && out.isExplicit(conflict.first) &&
                out.isExplicit(conflict.second)) {
                throw OptionError(element_, conflict.first,
                                  "conflicts with " + conflict.second + ": " + conflict.reason);
            }
        }
        for (const auto& item : rules_) {
            if (!item.modes.appliesIn(context)) {
                continue;
            }
            const auto reason = item.check(out);
            if (!reason.empty()) {
                throw OptionError(element_, std::string{}, item.name + ": " + reason);
            }
        }
        return out;
    }

    /// Canonical text of every key, in declaration order.
    [[nodiscard]] std::vector<std::pair<std::string, std::string>> serialize(const T& options) const
    {
        std::vector<std::pair<std::string, std::string>> out;
        out.reserve(keys_.size());
        for (const auto& key_entry : keys_) {
            out.emplace_back(key_entry.canonical, key_entry.format(options));
        }
        return out;
    }

    /// A JSON object, in declaration order:
    ///   {"<artifact key>":{"value":"<text>","source":"<source>"
    ///                      [,"spelling":"<input spelling>"][,"origin":"<origin>"]},...}
    [[nodiscard]] std::string toJson(const ResolvedOptions<T>& resolved) const
    {
        std::string out = "{";
        for (std::size_t i = 0; i < keys_.size(); ++i) {
            const auto& key_entry = keys_[i];
            if (i != 0u) {
                out += ',';
            }
            out += detail::jsonQuoted(key_entry.artifact_key);
            out += ":{\"value\":";
            out += detail::jsonQuoted(key_entry.format(resolved.value));
            const auto* provenance = resolved.find(key_entry.canonical);
            const auto source = provenance != nullptr ? provenance->source : OptionSource::Default;
            out += ",\"source\":";
            out += detail::jsonQuoted(optionSourceName(source));
            if (provenance != nullptr && !provenance->spelling.empty()) {
                out += ",\"spelling\":";
                out += detail::jsonQuoted(provenance->spelling);
            }
            if (provenance != nullptr && !provenance->origin.empty()) {
                out += ",\"origin\":";
                out += detail::jsonQuoted(provenance->origin);
            }
            out += '}';
        }
        out += '}';
        return out;
    }

    /// One documentation entry per key; the default text comes from `defaults`.
    [[nodiscard]] std::vector<OptionDocumentation> documentation(const T& defaults = T{}) const
    {
        std::vector<OptionDocumentation> out;
        out.reserve(keys_.size());
        for (const auto& key_entry : keys_) {
            OptionDocumentation doc;
            doc.key = key_entry.canonical;
            for (const auto& spelling : key_entry.spellings) {
                (spelling.deprecated ? doc.deprecated_spellings : doc.spellings).push_back(spelling.name);
            }
            doc.type = key_entry.type;
            doc.default_value = key_entry.format(defaults);
            doc.doc = key_entry.doc;
            out.push_back(std::move(doc));
        }
        return out;
    }

    /// Every accepted spelling (active and deprecated), in declaration order.
    [[nodiscard]] std::vector<std::string> spellings() const
    {
        std::vector<std::string> out;
        for (const auto& key_entry : keys_) {
            for (const auto& spelling : key_entry.spellings) {
                out.push_back(spelling.name);
            }
        }
        return out;
    }

    /// Every retired spelling, in declaration order.  An XML acceptance list
    /// includes them so that resolve() reports the retirement message.
    [[nodiscard]] std::vector<std::string> retiredSpellings() const
    {
        std::vector<std::string> out;
        for (const auto& item : retired_) {
            out.insert(out.end(), item.spellings.begin(), item.spellings.end());
        }
        return out;
    }

    /// True when the spelling belongs to a declared key.
    [[nodiscard]] bool acceptsSpelling(std::string_view spelling) const
    {
        for (const auto& key_entry : keys_) {
            for (const auto& item : key_entry.spellings) {
                if (item.name == spelling) {
                    return true;
                }
            }
        }
        return false;
    }

    /// True when the spelling belongs to a retired key.
    [[nodiscard]] bool isRetiredSpelling(std::string_view spelling) const
    {
        for (const auto& item : retired_) {
            if (std::find(item.spellings.begin(), item.spellings.end(), spelling) !=
                item.spellings.end()) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] std::size_t size() const noexcept { return keys_.size(); }

private:
    struct Key {
        std::string canonical{};
        std::vector<OptionSpelling> spellings{};
        DuplicatePolicy duplicates{DuplicatePolicy::FirstDefinedWins};
        EmptyPolicy empty{EmptyPolicy::Absent};
        std::string doc{};
        std::string invalid_message{};
        std::string artifact_key{};
        std::string type{};
        std::function<void(T&, std::string_view, std::string&)> apply{};
        std::function<std::string(const T&)> format{};
    };
    struct Implication {
        std::string trigger{};
        std::string target{};
        std::string text{};
        std::optional<std::string> when_trigger_is{};
        bool trigger_must_be_explicit{true};
        ModeCondition modes{};
    };
    struct Conflict {
        std::string first{};
        std::string second{};
        std::string reason{};
        ModeCondition modes{};
    };
    struct Rule {
        std::string name{};
        std::function<std::string(const ResolvedOptions<T>&)> check{};
        ModeCondition modes{};
    };
    struct Retired {
        std::vector<std::string> spellings{};
        std::string message{};
        RetiredAcceptance acceptance{};
    };

    /// Read one key from the input into out.value; returns its provenance.
    OptionProvenance readKey(const Key& key_entry, const OptionInput& input,
                             ResolvedOptions<T>& out) const
    {
        OptionProvenance provenance{key_entry.canonical, OptionSource::Default, {}, {}};
        const OptionSpelling* chosen = nullptr;
        const OptionSpelling* first_supplied = nullptr;
        std::string chosen_text;
        for (const auto& spelling : key_entry.spellings) {
            const auto* entry = input.find(spelling.name);
            if (entry == nullptr) {
                continue;
            }
            if (first_supplied == nullptr) {
                first_supplied = &spelling;
            } else if (key_entry.duplicates == DuplicatePolicy::Reject) {
                throw OptionError(element_, key_entry.canonical,
                                  "defines more than one spelling ('" + first_supplied->name +
                                      "' and '" + spelling.name + "'); specify exactly one");
            }
            if (chosen != nullptr) {
                continue;
            }
            auto text = detail::trimmed(entry->text);
            if (text.empty()) {
                if (key_entry.empty == EmptyPolicy::Error) {
                    throw OptionError(element_, spelling.name, "empty value");
                }
                if (key_entry.empty == EmptyPolicy::Absent) {
                    continue;
                }
            }
            chosen = &spelling;
            chosen_text = std::move(text);
        }
        if (chosen == nullptr) {
            return provenance;
        }
        std::string error;
        key_entry.apply(out.value, chosen_text, error);
        if (!error.empty()) {
            throw OptionError(element_, chosen->name,
                              key_entry.invalid_message.empty() ? error : key_entry.invalid_message);
        }
        provenance.source =
            chosen->deprecated ? OptionSource::DeprecatedSpelling : OptionSource::Explicit;
        provenance.spelling = chosen->name;
        if (chosen->deprecated) {
            out.deprecations.push_back(
                OptionDeprecation{chosen->name, key_entry.canonical,
                                  "deprecated spelling; use '" + key_entry.canonical + "'"});
        }
        return provenance;
    }

    /// The canonical text of `text` under the key's codec; a declaration
    /// error when it does not parse.
    [[nodiscard]] std::string canonicalText(const Key& key_entry, const std::string& text,
                                            const std::string& what) const
    {
        T probe{};
        std::string error;
        key_entry.apply(probe, detail::trimmed(text), error);
        if (!error.empty()) {
            throw std::logic_error("OptionSchema '" + element_ + "': " + what +
                                   " does not parse: " + error);
        }
        return key_entry.format(probe);
    }

    [[nodiscard]] const Key* findKey(std::string_view canonical) const
    {
        for (const auto& key_entry : keys_) {
            if (key_entry.canonical == canonical) {
                return &key_entry;
            }
        }
        return nullptr;
    }
    [[nodiscard]] const Key& requireKey(const std::string& canonical, const char* what) const
    {
        const auto* key_entry = findKey(canonical);
        if (key_entry == nullptr) {
            throw std::logic_error("OptionSchema '" + element_ + "': " + what +
                                   " names undeclared key '" + canonical + "'");
        }
        return *key_entry;
    }

    std::string element_;
    std::vector<Key> keys_{};
    std::vector<Implication> implications_{};
    std::vector<Conflict> conflicts_{};
    std::vector<Rule> rules_{};
    std::vector<Retired> retired_{};
};

} // namespace options
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CORE_OPTIONSCHEMA_H
