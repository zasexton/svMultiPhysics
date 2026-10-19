/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CORE_RANKLOG_H
#define SVMP_FE_CORE_RANKLOG_H

/**
 * @file RankLog.h
 * @brief Rank-aware console streams for solver progress and diagnostic lines.
 *
 * Rank 0 prints; other ranks print only when SVMP_OOP_SOLVER_TRACE is true.
 * The streams carry the caller's text unchanged; nothing is prefixed.
 */

#include "Core/FEConfig.h"
#include "Core/Logger.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <streambuf>
#include <string>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace rank_log {

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

/// Rank 0 of MPI_COMM_WORLD, or true before MPI is initialized.
inline bool oopIsRoot()
{
  int rank = 0;
#if FE_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
#endif
  return rank == 0;
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
 * Whether this rank prints diagnostic lines of @p level: it logs
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
 * Stream for diagnostic lines of @p level: std::cout when
 * oopDiagnosticsEnabled(level), otherwise a stream on which insertions do no
 * formatting.  Insertion arguments are still evaluated, so callers gate
 * values computed only for the line with oopDiagnosticsEnabled().
 */
inline std::ostream& oopDiagnosticsCout(
    svmp::FE::LogLevel level = svmp::FE::LogLevel::INFO)
{
  return oopDiagnosticsEnabled(level) ? std::cout : detail::disabled_stream();
}

} // namespace rank_log
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CORE_RANKLOG_H
