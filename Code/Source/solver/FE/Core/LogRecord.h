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
 */

#include "Core/Logger.h"
#include "Core/Types.h"

#include <concepts>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
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

    [[nodiscard]] const std::string& owner() const noexcept { return owner_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }
    [[nodiscard]] LogLevel level() const noexcept { return level_; }
    [[nodiscard]] const std::vector<LogField>& fields() const noexcept { return fields_; }

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

private:
    std::string owner_{};
    std::string message_{};
    LogLevel level_{LogLevel::INFO};
    std::vector<LogField> fields_{};
};

/// Prints @p record at its level.
inline void logRecord(const LogRecord& record)
{
    FE_LOG(record.level(), record.str());
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
