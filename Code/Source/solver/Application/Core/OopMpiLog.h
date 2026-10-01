#pragma once

#include "FE/Core/Logger.h"
#include "Mesh/Core/MeshComm.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <streambuf>
#include <string>

namespace application {
namespace core {

namespace detail {

inline std::string trim_copy(std::string s)
{
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

inline std::string lower_copy(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

inline bool parse_bool_relaxed(const std::string& raw)
{
  const auto v = lower_copy(trim_copy(raw));
  if (v == "true" || v == "1" || v == "yes" || v == "on") {
    return true;
  }
  if (v == "false" || v == "0" || v == "no" || v == "off") {
    return false;
  }
  return false;
}

class NullBuffer final : public std::streambuf {
public:
  int overflow(int ch) override { return traits_type::not_eof(ch); }
};

inline std::ostream& null_stream()
{
  static NullBuffer buf;
  static std::ostream os(&buf);
  return os;
}

// A stream without a buffer is in the bad state: every insertion returns at
// its sentry, before any formatting.
inline std::ostream& disabled_stream()
{
  static std::ostream os(nullptr);
  return os;
}

} // namespace detail

inline bool oopTraceEnabled()
{
  if (const char* env = std::getenv("SVMP_OOP_SOLVER_TRACE")) {
    return detail::parse_bool_relaxed(env);
  }
  return false;
}

inline bool oopIsRoot()
{
  return svmp::MeshComm::world().rank() == 0;
}

inline bool oopShouldLog()
{
  return oopIsRoot() || oopTraceEnabled();
}

inline std::ostream& oopCout()
{
  return oopShouldLog() ? std::cout : detail::null_stream();
}

/**
 * Whether this rank prints Application diagnostic lines of @p level: it logs
 * (oopShouldLog()) and the FE logger level admits @p level.  The logger level
 * comes from FE_LOG_LEVEL (default INFO), so FE_LOG_LEVEL=WARNING or higher
 * suppresses INFO diagnostics together with the FE INFO lines.  The answer is
 * rank-local: never use it to skip a collective operation.
 */
inline bool oopDiagnosticsEnabled(
    svmp::FE::LogLevel level = svmp::FE::LogLevel::INFO)
{
  return oopShouldLog() && svmp::FE::Logger::instance().get_level() <= level;
}

/**
 * Stream for Application diagnostic lines of @p level: std::cout when
 * oopDiagnosticsEnabled(level), otherwise a stream on which insertions do no
 * formatting.  Insertion arguments are still evaluated, so callers gate
 * values computed only for the line with oopDiagnosticsEnabled().
 */
inline std::ostream& oopDiagnosticsCout(
    svmp::FE::LogLevel level = svmp::FE::LogLevel::INFO)
{
  return oopDiagnosticsEnabled(level) ? std::cout : detail::disabled_stream();
}

} // namespace core
} // namespace application

