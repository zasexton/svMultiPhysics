#pragma once

// Forwarding header: the rank-aware console streams live in
// FE/Core/RankLog.h (namespace svmp::FE::rank_log).

#include "FE/Core/RankLog.h"

namespace application {
namespace core {

namespace detail {

using svmp::FE::rank_log::detail::NullBuffer;
using svmp::FE::rank_log::detail::disabled_stream;
using svmp::FE::rank_log::detail::lower_copy;
using svmp::FE::rank_log::detail::null_stream;
using svmp::FE::rank_log::detail::parse_bool_relaxed;
using svmp::FE::rank_log::detail::trim_copy;

} // namespace detail

using svmp::FE::rank_log::oopCout;
using svmp::FE::rank_log::oopDiagnosticsCout;
using svmp::FE::rank_log::oopDiagnosticsEnabled;
using svmp::FE::rank_log::oopIsRoot;
using svmp::FE::rank_log::oopShouldLog;
using svmp::FE::rank_log::oopTraceEnabled;

} // namespace core
} // namespace application
