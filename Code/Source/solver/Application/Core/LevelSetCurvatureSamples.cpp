#include "Application/Core/LevelSetCurvatureSamples.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Backends/Interfaces/GenericVector.h"
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

} // namespace

namespace application {
namespace core {

std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample>
collectLevelSetCurvatureCutVolumeSupplementalSamples(
    const svmp::FE::systems::FESystem& system,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::FieldId field,
    int interface_marker,
    svmp::FE::geometry::CutIntegrationSide side)
{
  std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample> samples;
  if (side == svmp::FE::geometry::CutIntegrationSide::Interface) {
    return samples;
  }

  const auto* cut_context = system.cutIntegrationContext();
  if (cut_context == nullptr) {
    return samples;
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
          svmp::FE::Real value) {
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
        return;
      }
    }
    samples.push_back(
        svmp::FE::level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = parent_cell,
            .coordinate = coordinate,
            .value = value});
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
      append_sample(parent_cell, *physical_point, value);
    }
  }

  return samples;
}

} // namespace core
} // namespace application

