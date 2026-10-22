/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/LevelSetCutVolumeMeasures.h"

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Geometry/CutQuadratureMapping.h"

#include <stdexcept>

namespace svmp {
namespace FE {
namespace level_set {

// Moved from Application/Core/ActiveDomainOutput.cpp with the body and
// messages unchanged.

CutVolumeMeasureSummary collectCutVolumeMeasures(
    const svmp::FE::assembly::IMeshAccess& mesh,
    const std::vector<const svmp::FE::geometry::CutQuadratureRule*>& rules,
    const svmp::FE::assembly::CutIntegrationContext* context)
{
  CutVolumeMeasureSummary summary;
  bool found_revisioned_rule = false;
  bool found_unrevisioned_rule = false;
  for (const auto* rule : rules) {
    if (rule == nullptr ||
        rule->kind != svmp::FE::geometry::CutQuadratureKind::Volume) {
      continue;
    }
    const auto snapshot_revision_key =
        rule->provenance.free_surface_snapshot_revision_key;
    const auto source_value_revision =
        rule->provenance.source_value_revision;
    if ((snapshot_revision_key == 0u) != (source_value_revision == 0u)) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] Cut-volume measure rule has an "
          "incomplete free-surface snapshot revision.");
    }
    if (snapshot_revision_key != 0u) {
      found_revisioned_rule = true;
      if (summary.free_surface_snapshot_revision_key == 0u) {
        summary.free_surface_snapshot_revision_key = snapshot_revision_key;
        summary.source_value_revision = source_value_revision;
      } else if (summary.free_surface_snapshot_revision_key !=
                     snapshot_revision_key ||
                 summary.source_value_revision != source_value_revision) {
        throw std::invalid_argument(
            "[svMultiPhysics::Application] Cut-volume measure rules mix "
            "free-surface snapshot revisions.");
      }
    } else {
      found_unrevisioned_rule = true;
    }
  }
  if (found_revisioned_rule && found_unrevisioned_rule) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] Cut-volume measure rules mix "
        "revisioned and unrevisioned geometry.");
  }

  for (const auto* rule : rules) {
    if (rule == nullptr ||
        rule->kind != svmp::FE::geometry::CutQuadratureKind::Volume) {
      continue;
    }
    const auto parent_cell = static_cast<svmp::FE::GlobalIndex>(
        rule->provenance.parent_entity);
    if (parent_cell < 0 || parent_cell >= mesh.numCells() ||
        !mesh.isOwnedCell(parent_cell)) {
      continue;
    }
    if (found_revisioned_rule) {
      ++summary.revisioned_rule_count;
    }
    ++summary.rule_count;
    summary.reference_measure += rule->measure;
    try {
      if (rule->released_point_count != 0u) {
        if (context == nullptr) {
          throw std::invalid_argument(
              "[svMultiPhysics::Application] Cut-volume measure of a "
              "classification-only rule requires its integration context.");
        }
        summary.physical_measure +=
            context->physicalVolumeRuleMeasure(mesh, *rule);
      } else {
        summary.physical_measure +=
            svmp::FE::geometry::physicalCutQuadratureMeasure(mesh, *rule);
      }
      ++summary.physical_rule_count;
    } catch (...) {
      if (found_revisioned_rule) {
        throw;
      }
      ++summary.skipped_physical_rule_count;
    }
  }
  return summary;
}

} // namespace level_set
} // namespace FE
} // namespace svmp
