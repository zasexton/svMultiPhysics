/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CORE_LOG_RECORD_H
#define SVMP_FE_CORE_LOG_RECORD_H

/**
 * @file LogRecord.h
 * @brief A log line as data: an owner, a message and ordered key=value fields
 *
 * A record prints as
 *
 *   <owner>: <message> <key>=<value> <key>='<value>' ...
 *
 * or, without a message, "<owner>: <key>=<value> ...": the lines the run
 * scripts parse by key.  A formulation or an FE service declares its records
 * (the owner, the message, the keys in order and how each value is written)
 * and FE prints them (logRecord, FormulationBuilder::log), instead of
 * concatenating the text at each log point.  How a value is written is part
 * of the record: text as given, quoted text, an integer (std::to_string), a
 * fixed real (std::to_string: six decimals) or a general real (an output
 * stream's default: six significant digits).
 *
 * The record conventions of the services (agent-status/refactor/
 * logrecord-cleanup-proposal.md, approved 2026-10-24): one diagnostic word,
 * the first field (diagnostic()); unique snake_case keys, a repeated group
 * of fields printed as child lines (child()); reals in the shortest text that
 * reads back to the same double (real(): std::to_chars); booleans true/false,
 * an absent value none; a record of values reduced over the communicator
 * printed by rank 0 only (rootOnly(): as the Application's console stream),
 * every other record on every rank; WARNING and ERROR records on stderr (the
 * FE logger's streams).
 */

#include "Core/Logger.h"
#include "Core/RankLog.h"
#include "Core/Types.h"

#include <charconv>
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {

/// One field of a record: " key=value", or " key='value'" when quoted.
struct LogField {
    std::string key{};
    std::string value{};
    bool quoted{false};
};

class LogRecord {
public:
    explicit LogRecord(std::string owner, std::string message = {}, LogLevel level = LogLevel::INFO)
        : owner_(std::move(owner)), message_(std::move(message)), level_(level)
    {
    }

    /// key=value, the value as given.
    LogRecord& text(std::string key, std::string_view value)
    {
        fields_.push_back(LogField{.key = std::move(key), .value = std::string(value)});
        return *this;
    }

    /// key='value'.
    LogRecord& quoted(std::string key, std::string_view value)
    {
        fields_.push_back(LogField{.key = std::move(key), .value = std::string(value), .quoted = true});
        return *this;
    }

    /// key=std::to_string(value).
    template <std::integral T>
    LogRecord& integer(std::string key, T value)
    {
        return text(std::move(key), std::to_string(value));
    }

    /// key=std::to_string(value): six decimals.
    LogRecord& fixed(std::string key, Real value) { return text(std::move(key), std::to_string(value)); }

    /// key=value as an output stream writes it by default: six significant digits.
    LogRecord& general(std::string key, Real value)
    {
        std::ostringstream out;
        out << value;
        return text(std::move(key), out.str());
    }

    /// key=value in the shortest text that reads back to the same double
    /// (std::to_chars): 0.26, 1e-05, 2; nan, inf, -inf.
    LogRecord& real(std::string key, Real value) { return text(std::move(key), shortestReal(value)); }

    /// key=value (real()) or key=none.
    LogRecord& optionalReal(std::string key, const std::optional<Real>& value)
    {
        return value.has_value() ? real(std::move(key), *value) : text(std::move(key), "none");
    }

    /// key=true or key=false.
    LogRecord& boolean(std::string key, bool value) { return text(std::move(key), value ? "true" : "false"); }

    /// key=value by the value's type: boolean(), integer(), real(), or text.
    template <class T>
    LogRecord& value(std::string key, const T& value)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return boolean(std::move(key), value);
        } else if constexpr (std::is_integral_v<T>) {
            return integer(std::move(key), value);
        } else if constexpr (std::is_floating_point_v<T>) {
            return real(std::move(key), static_cast<Real>(value));
        } else {
            return text(std::move(key), std::string_view(value));
        }
    }

    /// diagnostic=word as the first field (one per record: a second call replaces the word).
    LogRecord& diagnostic(std::string_view word)
    {
        if (!fields_.empty() && fields_.front().key == "diagnostic") {
            fields_.front().value = std::string(word);
        } else {
            fields_.insert(fields_.begin(), LogField{.key = "diagnostic", .value = std::string(word)});
        }
        return *this;
    }

    /// Printed by rank 0 only: a record of values reduced over the
    /// communicator (rank_log::oopShouldLog(), as the Application's console stream).
    LogRecord& rootOnly(bool root_only = true) noexcept
    {
        root_only_ = root_only;
        return *this;
    }

    /**
     * @brief A child line, printed after this record: the same owner, message
     *        and level, diagnostic=@p word, then this record's fields named
     *        by @p identifying_keys (in that order); the caller adds the
     *        child's own fields.  The reference is valid until the next child.
     */
    LogRecord& child(std::string_view word, std::initializer_list<std::string_view> identifying_keys)
    {
        LogRecord line(owner_, message_, level_);
        line.diagnostic(word);
        line.rootOnly(root_only_);
        for (const auto key : identifying_keys) {
            for (const auto& field : fields_) {
                if (field.key == key) {
                    line.fields_.push_back(field);
                    break;
                }
            }
        }
        children_.push_back(std::move(line));
        return children_.back();
    }

    /// The shortest text that reads back to @p value (std::to_chars; "nan", "inf", "-inf").
    [[nodiscard]] static std::string shortestReal(Real value)
    {
        if (std::isnan(value)) {
            return "nan";
        }
        if (std::isinf(value)) {
            return value > 0 ? "inf" : "-inf";
        }
        char buffer[64];
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer), static_cast<double>(value));
        return std::string(buffer, result.ptr);
    }

    [[nodiscard]] const std::string& owner() const noexcept { return owner_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }
    [[nodiscard]] LogLevel level() const noexcept { return level_; }
    [[nodiscard]] const std::vector<LogField>& fields() const noexcept { return fields_; }
    [[nodiscard]] const std::vector<LogRecord>& children() const noexcept { return children_; }
    [[nodiscard]] bool isRootOnly() const noexcept { return root_only_; }

    /// The line.
    [[nodiscard]] std::string str() const
    {
        std::string line = owner_ + ":";
        if (!message_.empty()) {
            line += " " + message_;
        }
        for (const auto& field : fields_) {
            line += " " + field.key + "=";
            line += field.quoted ? "'" + field.value + "'" : field.value;
        }
        return line;
    }

    /// The line and then the child lines, in order.
    [[nodiscard]] std::vector<std::string> lines() const
    {
        std::vector<std::string> out{str()};
        for (const auto& line : children_) {
            auto nested = line.lines();
            out.insert(out.end(), nested.begin(), nested.end());
        }
        return out;
    }

private:
    std::string owner_{};
    std::string message_{};
    LogLevel level_{LogLevel::INFO};
    std::vector<LogField> fields_{};
    std::vector<LogRecord> children_{};
    bool root_only_{false};
};

/// Prints @p record and its child lines at its level (rank 0 only for a
/// rootOnly() record).
inline void logRecord(const LogRecord& record)
{
    if (record.isRootOnly() && !rank_log::oopShouldLog()) {
        return;
    }
    for (const auto& line : record.lines()) {
        FE_LOG(record.level(), line);
    }
}

/// Prints @p records in order.
inline void logRecords(std::span<const LogRecord> records)
{
    for (const auto& record : records) {
        logRecord(record);
    }
}

} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CORE_LOG_RECORD_H
