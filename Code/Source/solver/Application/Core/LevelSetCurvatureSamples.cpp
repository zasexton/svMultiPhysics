#include "Application/Core/LevelSetCurvatureSamples.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/Spaces/FunctionSpace.h"
#include "FE/Systems/FESystem.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>

namespace {

[[nodiscard]] bool allFinite(const std::array<svmp::FE::Real, 3>& point)
{
  return std::isfinite(point[0]) &&
         std::isfinite(point[1]) &&
         std::isfinite(point[2]);
}

[[nodiscard]] bool isZeroPoint(const std::array<svmp::FE::Real, 3>& point)
{
  return point[0] == svmp::FE::Real{0.0} &&
         point[1] == svmp::FE::Real{0.0} &&
         point[2] == svmp::FE::Real{0.0};
}

[[nodiscard]] std::shared_ptr<svmp::FE::geometry::GeometryMapping>
createCellGeometryMapping(const svmp::FE::assembly::IMeshAccess& mesh,
                          svmp::FE::GlobalIndex cell)
{
  if (cell < 0 || cell >= mesh.numCells()) {
    return nullptr;
  }

  std::vector<std::array<svmp::FE::Real, 3>> coords;
  mesh.getCellCoordinates(cell, coords);
  if (coords.empty()) {
    return nullptr;
  }

  std::vector<svmp::FE::math::Vector<svmp::FE::Real, 3>> nodes;
  nodes.reserve(coords.size());
  for (const auto& coord : coords) {
    svmp::FE::math::Vector<svmp::FE::Real, 3> node{};
    node[0] = coord[0];
    node[1] = coord[1];
    node[2] = coord[2];
    nodes.push_back(node);
  }

  svmp::FE::geometry::MappingRequest map_request;
  map_request.element_type = mesh.getCellType(cell);
  map_request.geometry_order = mesh.getCellGeometryOrder(cell);
  map_request.use_affine = map_request.geometry_order <= 1;
  return svmp::FE::geometry::MappingFactory::create(map_request, nodes);
}

[[nodiscard]] std::optional<std::array<svmp::FE::Real, 3>>
physicalCellPointAtReference(
    const svmp::FE::geometry::GeometryMapping& mapping,
    const std::array<svmp::FE::Real, 3>& reference_point)
{
  svmp::FE::math::Vector<svmp::FE::Real, 3> xi{};
  xi[0] = reference_point[0];
  xi[1] = reference_point[1];
  xi[2] = reference_point[2];
  const auto physical = mapping.map_to_physical(xi);
  return std::array<svmp::FE::Real, 3>{
      physical[0], physical[1], physical[2]};
}

[[nodiscard]] std::optional<svmp::FE::spaces::FunctionSpace::Value>
referenceCellInteriorPoint(svmp::FE::ElementType type, int order)
{
  try {
    const auto nodes =
        svmp::FE::basis::ReferenceNodeLayout::get_lagrange_node_coords(
            type, order);
    if (nodes.empty()) {
      return std::nullopt;
    }
    svmp::FE::spaces::FunctionSpace::Value point{};
    for (const auto& node : nodes) {
      point[0] += node[0];
      point[1] += node[1];
      point[2] += node[2];
    }
    const auto inverse_count =
        svmp::FE::Real{1.0} /
        static_cast<svmp::FE::Real>(nodes.size());
    point[0] *= inverse_count;
    point[1] *= inverse_count;
    point[2] *= inverse_count;
    return point;
  } catch (...) {
    return std::nullopt;
  }
}

} // namespace

namespace application {
namespace core {

std::optional<std::array<svmp::FE::Real, 3>>
mapLevelSetCurvatureReferenceSampleToPhysical(
    const svmp::FE::assembly::IMeshAccess& mesh,
    svmp::FE::GlobalIndex cell,
    const std::array<svmp::FE::Real, 3>& reference_point)
{
  if (!allFinite(reference_point)) {
    throw std::invalid_argument(
        "level-set curvature reference sample must be finite");
  }
  const auto mapping = createCellGeometryMapping(mesh, cell);
  if (mapping == nullptr) {
    return std::nullopt;
  }
  return physicalCellPointAtReference(*mapping, reference_point);
}

std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample>
collectLevelSetCurvatureCutVolumeSupplementalSamples(
    const svmp::FE::systems::FESystem& system,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::FieldId field,
    int interface_marker,
    svmp::FE::geometry::CutIntegrationSide side,
    std::uint64_t evaluated_state_source_revision)
{
  std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample> samples;
  if (side == svmp::FE::geometry::CutIntegrationSide::Interface) {
    return samples;
  }

  const auto* cut_context = system.cutIntegrationContext();
  if (cut_context == nullptr) {
    return samples;
  }

  std::uint64_t authoritative_snapshot_revision_key = 0u;
  std::uint64_t authoritative_source_value_revision = 0u;
  if (cut_context->hasFreeSurfaceGeometrySnapshotForMarker(
          interface_marker)) {
    cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
        system.meshAccess());
    authoritative_snapshot_revision_key =
        cut_context->freeSurfaceGeometrySnapshotRevisionForMarker(
            interface_marker);
    const auto& snapshots = cut_context->freeSurfaceGeometrySnapshots();
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     authoritative_snapshot_revision_key;
        });
    if (found == snapshots.end()) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] Cut-volume curvature sampling could "
          "not resolve its authoritative geometry snapshot.");
    }
    authoritative_source_value_revision =
        (*found)->revision().source_value_revision;
    if (evaluated_state_source_revision == 0u ||
        evaluated_state_source_revision !=
            authoritative_source_value_revision) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] Cut-volume curvature sampling state "
          "does not match its authoritative geometry source revision.");
    }
  }

  const auto& rec = system.fieldRecord(field);
  if (!rec.space || rec.components != 1) {
    return samples;
  }

  const auto rules =
      cut_context->generatedVolumeRulesForMarkerAndSide(interface_marker, side);
  if (rules.empty()) {
    return samples;
  }

  const auto& mesh = system.meshAccess();
  const auto& field_dofs = system.fieldDofHandler(field);
  const auto offset = system.fieldDofOffset(field);
  const bool use_prescribed =
      rec.source_kind == svmp::FE::systems::FieldSourceKind::PrescribedData;
  const auto prescribed_coefficients =
      use_prescribed ? system.prescribedFieldCoefficients(field)
                     : std::span<const svmp::FE::Real>{};

  std::unique_ptr<svmp::FE::assembly::GlobalSystemView> solution_view;
  if (!use_prescribed && state.u_vector != nullptr) {
    auto* vec = const_cast<svmp::FE::backends::GenericVector*>(state.u_vector);
    solution_view = vec->createAssemblyView();
  }

  std::map<svmp::FE::GlobalIndex, std::vector<svmp::FE::Real>>
      cell_coefficients_cache;
  auto coefficients_for_cell =
      [&](svmp::FE::GlobalIndex parent_cell)
          -> const std::vector<svmp::FE::Real>& {
    auto [it, inserted] =
        cell_coefficients_cache.emplace(parent_cell,
                                        std::vector<svmp::FE::Real>{});
    if (!inserted) {
      return it->second;
    }

    const auto cell_dofs = field_dofs.getCellDofs(parent_cell);
    const auto expected = rec.space->dofs_per_element(parent_cell);
    if (cell_dofs.size() != expected) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set curvature projection found "
          "a cut-volume level-set cell with incompatible DOF count.");
    }

    it->second.reserve(cell_dofs.size());
    for (const auto local_dof : cell_dofs) {
      const auto dof = use_prescribed ? local_dof : local_dof + offset;
      if (dof < 0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set curvature projection found "
            "a negative level-set DOF.");
      }
      if (use_prescribed) {
        const auto idx = static_cast<std::size_t>(dof);
        if (idx >= prescribed_coefficients.size()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set curvature projection "
              "found prescribed level-set coefficients that are too small.");
        }
        it->second.push_back(prescribed_coefficients[idx]);
      } else if (solution_view) {
        it->second.push_back(solution_view->getVectorEntry(dof));
      } else {
        const auto idx = static_cast<std::size_t>(dof);
        if (idx >= state.u.size()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set curvature projection "
              "found a level-set DOF outside the current state vector.");
        }
        it->second.push_back(state.u[idx]);
      }
    }
    return it->second;
  };

  std::map<svmp::FE::GlobalIndex,
           std::shared_ptr<svmp::FE::geometry::GeometryMapping>>
      mapping_cache;
  auto mapping_for_cell =
      [&](svmp::FE::GlobalIndex cell)
          -> std::shared_ptr<svmp::FE::geometry::GeometryMapping> {
    auto it = mapping_cache.find(cell);
    if (it != mapping_cache.end()) {
      return it->second;
    }
    auto mapping = createCellGeometryMapping(mesh, cell);
    mapping_cache.emplace(cell, mapping);
    return mapping;
  };

  auto append_sample =
      [&](svmp::FE::MeshIndex parent_cell,
          const std::array<svmp::FE::Real, 3>& coordinate,
          svmp::FE::Real value,
          std::uint64_t snapshot_revision_key,
          std::uint64_t source_value_revision,
          std::uint64_t cut_topology_revision) {
    if (!allFinite(coordinate) || !std::isfinite(value)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set curvature projection "
          "received a non-finite cut-volume supplemental sample.");
    }
    constexpr svmp::FE::Real duplicate_tol2 = svmp::FE::Real{1.0e-24};
    constexpr svmp::FE::Real duplicate_value_tol = svmp::FE::Real{1.0e-12};
    for (const auto& existing : samples) {
      if (existing.parent_cell != parent_cell) {
        continue;
      }
      const auto dx = existing.coordinate[0] - coordinate[0];
      const auto dy = existing.coordinate[1] - coordinate[1];
      const auto dz = existing.coordinate[2] - coordinate[2];
      const auto dist2 = dx * dx + dy * dy + dz * dz;
      if (dist2 <= duplicate_tol2 &&
          std::abs(existing.value - value) <= duplicate_value_tol) {
        if ((existing.free_surface_snapshot_revision_key != 0u ||
             snapshot_revision_key != 0u) &&
            (existing.free_surface_snapshot_revision_key !=
                 snapshot_revision_key ||
             existing.source_value_revision != source_value_revision)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set curvature projection "
              "found duplicate samples from different geometry snapshots.");
        }
        return;
      }
    }
    samples.push_back(
        svmp::FE::level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = parent_cell,
            .coordinate = coordinate,
            .value = value,
            .free_surface_snapshot_revision_key = snapshot_revision_key,
            .source_value_revision = source_value_revision,
            .cut_topology_revision = cut_topology_revision});
  };

  constexpr svmp::FE::Real cut_fraction_tol =
      svmp::FE::Real{16.0} * std::numeric_limits<svmp::FE::Real>::epsilon();
  for (const auto* rule : rules) {
    if (rule == nullptr ||
        rule->kind != svmp::FE::geometry::CutQuadratureKind::Volume ||
        rule->side != side ||
        rule->full_cell_equivalent) {
      continue;
    }
    const bool rule_is_revisioned =
        rule->provenance.free_surface_snapshot_revision_key != 0u ||
        rule->provenance.source_value_revision != 0u ||
        rule->provenance.cut_topology_revision != 0u;
    if (authoritative_snapshot_revision_key != 0u) {
      if (rule->provenance.free_surface_snapshot_revision_key !=
              authoritative_snapshot_revision_key ||
          rule->provenance.source_value_revision !=
              authoritative_source_value_revision ||
          rule->provenance.cut_topology_revision == 0u) {
        throw std::invalid_argument(
            "[svMultiPhysics::Application] Cut-volume curvature rule has "
            "incomplete or stale authoritative revision provenance.");
      }
    } else if (rule_is_revisioned) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] Revisioned cut-volume curvature rule "
          "has no authoritative geometry snapshot.");
    }
    if (std::isfinite(rule->volume_fraction) &&
        (rule->volume_fraction <= cut_fraction_tol ||
         rule->volume_fraction >= svmp::FE::Real{1.0} - cut_fraction_tol)) {
      continue;
    }
    const auto parent_cell = rule->provenance.parent_entity;
    if (parent_cell < 0 || parent_cell >= mesh.numCells()) {
      continue;
    }
    const auto& coefficients = coefficients_for_cell(parent_cell);
    if (coefficients.empty()) {
      continue;
    }

    for (const auto& point : rule->points) {
      if (!std::isfinite(point.weight) ||
          !(std::abs(point.weight) > svmp::FE::Real{0.0})) {
        continue;
      }

      auto reference_point = point.parent_coordinate;
      if (rule->frame == svmp::FE::geometry::CutGeometryFrame::Reference &&
          isZeroPoint(reference_point) &&
          !isZeroPoint(point.point)) {
        reference_point = point.point;
      }
      if (!allFinite(reference_point)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set curvature projection "
            "received a non-finite cut-volume reference sample.");
      }

      std::optional<std::array<svmp::FE::Real, 3>> physical_point;
      if (rule->frame == svmp::FE::geometry::CutGeometryFrame::Current) {
        if (!allFinite(point.point)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Level-set curvature projection "
              "received a non-finite cut-volume physical sample.");
        }
        physical_point = point.point;
      } else {
        auto mapping = mapping_for_cell(parent_cell);
        if (mapping != nullptr) {
          physical_point =
              physicalCellPointAtReference(*mapping, reference_point);
        }
      }
      if (!physical_point.has_value()) {
        continue;
      }

      svmp::FE::spaces::FunctionSpace::Value xi{};
      xi[0] = reference_point[0];
      xi[1] = reference_point[1];
      xi[2] = reference_point[2];
      const auto value = rec.space->evaluate_scalar(xi, coefficients);
      append_sample(parent_cell,
                    *physical_point,
                    value,
                    rule->provenance.free_surface_snapshot_revision_key,
                    rule->provenance.source_value_revision,
                    rule->provenance.cut_topology_revision);
    }
  }

  return samples;
}

std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample>
collectLevelSetCurvatureHighOrderSupplementalSamples(
    const svmp::FE::systems::FESystem& system,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::FieldId field,
    int interface_marker,
    std::uint64_t evaluated_state_source_revision)
{
  const auto& rec = system.fieldRecord(field);
  if (!rec.space || rec.components != 1) {
    return {};
  }

  const auto* cut_context = system.cutIntegrationContext();
  if (cut_context == nullptr ||
      !cut_context->hasFreeSurfaceGeometrySnapshotForMarker(
          interface_marker)) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] High-order curvature sampling requires "
        "an authoritative geometry snapshot.");
  }
  cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
      system.meshAccess());
  const auto snapshot_revision_key =
      cut_context->freeSurfaceGeometrySnapshotRevisionForMarker(
          interface_marker);
  const auto& snapshots = cut_context->freeSurfaceGeometrySnapshots();
  const auto snapshot = std::find_if(
      snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
        return candidate &&
               candidate->revision().snapshot_revision_key ==
                   snapshot_revision_key;
      });
  if (snapshot == snapshots.end()) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] High-order curvature sampling could "
        "not resolve its authoritative geometry snapshot.");
  }
  const auto source_value_revision =
      (*snapshot)->revision().source_value_revision;
  if (evaluated_state_source_revision == 0u ||
      evaluated_state_source_revision != source_value_revision) {
    throw std::invalid_argument(
        "[svMultiPhysics::Application] High-order curvature sampling state "
        "does not match its authoritative geometry source revision.");
  }

  std::map<svmp::FE::GlobalIndex, std::uint64_t> candidate_cells;
  for (const auto* rule :
       cut_context->interfaceRulesForMarker(interface_marker)) {
    if (rule == nullptr ||
        rule->kind != svmp::FE::geometry::CutQuadratureKind::Interface) {
      continue;
    }
    if (rule->provenance.free_surface_snapshot_revision_key !=
            snapshot_revision_key ||
        rule->provenance.source_value_revision != source_value_revision ||
        rule->provenance.cut_topology_revision == 0u) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] Authoritative interface rule has "
          "incomplete or stale revision provenance for high-order curvature "
          "sampling.");
    }
    const auto parent_cell = static_cast<svmp::FE::GlobalIndex>(
        rule->provenance.parent_entity);
    if (parent_cell < 0 || parent_cell >= system.meshAccess().numCells()) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] Authoritative interface rule has an "
          "invalid parent cell for high-order curvature sampling.");
    }
    auto [it, inserted] = candidate_cells.emplace(
        parent_cell, rule->provenance.cut_topology_revision);
    if (!inserted &&
        (it->second == 0u ||
         (rule->provenance.cut_topology_revision != 0u &&
          rule->provenance.cut_topology_revision < it->second))) {
      it->second = rule->provenance.cut_topology_revision;
    }
  }

  const auto& field_dofs = system.fieldDofHandler(field);
  const auto offset = system.fieldDofOffset(field);
  const bool use_prescribed =
      rec.source_kind == svmp::FE::systems::FieldSourceKind::PrescribedData;
  const auto prescribed_coefficients =
      use_prescribed ? system.prescribedFieldCoefficients(field)
                     : std::span<const svmp::FE::Real>{};
  std::unique_ptr<svmp::FE::assembly::GlobalSystemView> solution_view;
  if (!use_prescribed && state.u_vector != nullptr) {
    auto* vector =
        const_cast<svmp::FE::backends::GenericVector*>(state.u_vector);
    solution_view = vector->createAssemblyView();
  }

  std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample> samples;
  samples.reserve(candidate_cells.size());
  const auto& mesh = system.meshAccess();
  for (const auto& [parent_cell, cut_topology_revision] : candidate_cells) {
    const int order = rec.space->polynomial_order(parent_cell);
    if (order <= 1) {
      continue;
    }
    const auto reference_point =
        referenceCellInteriorPoint(mesh.getCellType(parent_cell), order);
    if (!reference_point.has_value()) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] High-order curvature sampling could "
          "not construct a certified interior reference point.");
    }
    const auto mapping = createCellGeometryMapping(mesh, parent_cell);
    if (mapping == nullptr) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] High-order curvature sampling could "
          "not construct the parent geometry mapping.");
    }
    const std::array<svmp::FE::Real, 3> reference_coordinate{{
        (*reference_point)[0],
        (*reference_point)[1],
        (*reference_point)[2],
    }};
    const auto physical_point =
        physicalCellPointAtReference(*mapping, reference_coordinate);
    if (!physical_point.has_value() || !allFinite(*physical_point)) {
      throw std::invalid_argument(
          "[svMultiPhysics::Application] High-order curvature sampling "
          "produced a non-finite physical point.");
    }

    const auto cell_dofs = field_dofs.getCellDofs(parent_cell);
    const auto expected = rec.space->dofs_per_element(parent_cell);
    if (cell_dofs.size() != expected) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] High-order curvature sampling found "
          "an incompatible cell DOF count.");
    }
    std::vector<svmp::FE::Real> coefficients;
    coefficients.reserve(cell_dofs.size());
    for (const auto local_dof : cell_dofs) {
      const auto dof = use_prescribed ? local_dof : local_dof + offset;
      if (dof < 0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] High-order curvature sampling "
            "found a negative field DOF.");
      }
      if (use_prescribed) {
        const auto index = static_cast<std::size_t>(dof);
        if (index >= prescribed_coefficients.size()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] High-order curvature sampling "
              "found an undersized prescribed coefficient vector.");
        }
        coefficients.push_back(prescribed_coefficients[index]);
      } else if (solution_view) {
        coefficients.push_back(solution_view->getVectorEntry(dof));
      } else {
        const auto index = static_cast<std::size_t>(dof);
        if (index >= state.u.size()) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] High-order curvature sampling "
              "found a field DOF outside the current state vector.");
        }
        coefficients.push_back(state.u[index]);
      }
    }
    const auto value =
        rec.space->evaluate_scalar(*reference_point, coefficients);
    if (!std::isfinite(value)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] High-order curvature sampling "
          "evaluated a non-finite field value.");
    }
    samples.push_back(
        svmp::FE::level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = static_cast<svmp::FE::MeshIndex>(parent_cell),
            .coordinate = *physical_point,
            .value = value,
            .free_surface_snapshot_revision_key = snapshot_revision_key,
            .source_value_revision = source_value_revision,
            .cut_topology_revision = cut_topology_revision});
  }
  return samples;
}

} // namespace core
} // namespace application
