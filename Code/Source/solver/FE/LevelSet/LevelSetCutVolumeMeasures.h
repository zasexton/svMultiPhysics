/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file LevelSetCutVolumeMeasures.h
 * @brief Owned reference and physical measures of generated cut-volume rules
 *
 * Moved from Application/Core/ActiveDomainOutput with the body and messages
 * unchanged; the Application header keeps the names through
 * using-declarations.
 */

#include "Core/Types.h"
#include "Geometry/CutQuadrature.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace svmp {
namespace FE {

namespace assembly {
class CutIntegrationContext;
class IMeshAccess;
} // namespace assembly

namespace level_set {

struct CutVolumeMeasureSummary {
  // Counts and measures include locally owned parent cells only.  Generated
  // rules on ghost cells remain available in the cut context for assembly but
  // must not contribute again to communicator-wide physical totals.
  // Revision identity is validated across every supplied volume rule before
  // ownership filtering so the complete consumer view cannot mix snapshots.
  svmp::FE::Real reference_measure{0.0};
  svmp::FE::Real physical_measure{0.0};
  std::uint64_t free_surface_snapshot_revision_key{0};
  std::uint64_t source_value_revision{0};
  std::size_t rule_count{0};
  std::size_t revisioned_rule_count{0};
  std::size_t physical_rule_count{0};
  std::size_t skipped_physical_rule_count{0};
};

// Rules stored classification-only in a cut integration context are
// measured through that context's exact rematerialization.
CutVolumeMeasureSummary collectCutVolumeMeasures(
    const svmp::FE::assembly::IMeshAccess& mesh,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules,
    const svmp::FE::assembly::CutIntegrationContext* context = nullptr);

} // namespace level_set
} // namespace FE
} // namespace svmp
