/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CORE_ENVIRONMENT_NAMES_H
#define SVMP_FE_CORE_ENVIRONMENT_NAMES_H

/**
 * @file EnvironmentNames.h
 * @brief The environment-name registry: the one service through which the
 *        solver reads environment variables.
 *
 * Every name the solver reads is declared once, in the catalogue of
 * EnvironmentNames.cpp, with its kind, owning service, value type, value
 * when unset, and whether setting it makes the run a non-reference run.
 * Code reads a name only through the registry; reading a name that is not
 * declared is a programming error (std::logic_error).
 *
 * The registry keeps no state: each read consults the process environment,
 * as std::getenv does, so a reader that cached its value keeps doing so and
 * one that read on every call still does.  The parsing of a value stays
 * with its reader; the registry offers the raw text and the two rules that
 * most flag readers share.
 *
 * configurationRecord() is the "environment_names" member that
 * effective_configuration.json carries when a declared name other than a
 * diagnostic is set; with none set it is empty and the artifact is
 * unchanged.
 */

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace environment {

/// What setting a name changes.
enum class NameKind : std::uint8_t {
    Diagnostic,        ///< log lines, traces and dumps only
    Performance,       ///< execution strategy, caches, threads, timers
    TestHook,          ///< forces a path or a failure for a test
    NumericalOverride  ///< replaces a numerical choice of the solve
};

/// How the value is read.
enum class ValueType : std::uint8_t {
    Flag,     ///< on or off
    Integer,
    Real,
    Text,     ///< a path, a prefix or one word of a fixed set
    List      ///< comma- or space-separated words or numbers
};

/// Whether a run that sets the name can still serve as a reference run.
enum class RunStatus : std::uint8_t {
    Reference,
    NonReference
};

/// One declared environment name.  Every text is a string literal.
struct NameDeclaration {
    const char* name;
    NameKind kind;
    const char* owner;          ///< the owning service
    ValueType type;
    const char* default_value;  ///< what the reader does when the name is unset
    RunStatus run_status;
    const char* controls;       ///< what the name controls

    [[nodiscard]] constexpr bool marksNonReference() const noexcept
    {
        return run_status == RunStatus::NonReference;
    }
};

/// A declared name that is set in the process environment.
struct NameUse {
    const NameDeclaration* declaration{nullptr};
    std::string value{};
};

[[nodiscard]] const char* toString(NameKind kind) noexcept;
[[nodiscard]] const char* toString(ValueType type) noexcept;

class EnvironmentNameRegistry {
public:
    /// A registry over @p declarations (kept by reference).  Throws
    /// std::logic_error when a name is declared twice or a declaration is
    /// incomplete.
    explicit EnvironmentNameRegistry(std::span<const NameDeclaration> declarations);

    /// The registry of every name the solver reads.
    [[nodiscard]] static const EnvironmentNameRegistry& solver();

    /// The declarations in name order.
    [[nodiscard]] const std::vector<const NameDeclaration*>& declarations() const noexcept
    {
        return by_name_;
    }

    /// nullptr when @p name is not declared.
    [[nodiscard]] const NameDeclaration* find(std::string_view name) const noexcept;

    /// Throws std::logic_error when @p name is not declared.
    [[nodiscard]] const NameDeclaration& declaration(std::string_view name) const;

    /// The value of a declared name, or nullptr when it is unset (the
    /// result of std::getenv).
    [[nodiscard]] const char* value(std::string_view name) const;

    /// True when the declared name is set, to any value.
    [[nodiscard]] bool isSet(std::string_view name) const;

    /// True when the declared flag is set to a nonempty value other than
    /// 0, false, off or no (in any letter case).
    [[nodiscard]] bool enabled(std::string_view name) const;

    /// The declared names that are set now, in name order.
    [[nodiscard]] std::vector<NameUse> setNames() const;

    /// The `"environment_names":{...}` member of an effective-configuration
    /// artifact: every set name that is not a diagnostic, with its value,
    /// kind, value type and owner, and whether the run is still a reference
    /// run.  Empty when no such name is set.
    [[nodiscard]] std::string configurationRecord() const;

private:
    std::vector<const NameDeclaration*> by_name_{};
};

/// Reads through the solver registry.
[[nodiscard]] const char* value(std::string_view name);
[[nodiscard]] bool isSet(std::string_view name);
[[nodiscard]] bool enabled(std::string_view name);
[[nodiscard]] std::string configurationRecord();

} // namespace environment
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CORE_ENVIRONMENT_NAMES_H
