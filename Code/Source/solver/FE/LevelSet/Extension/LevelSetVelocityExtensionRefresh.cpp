/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/Extension/LevelSetVelocityExtensionRefresh.h"

#include "Assembly/CutIntegrationContext.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Core/LogRecord.h"
#include "Core/RankLog.h"
#include "Dofs/EntityDofMap.h"
#include "Geometry/CutQuadrature.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "LevelSet/Extension/LevelSetVelocityExtensionSeed.h"
#include "LevelSet/LevelSetGeometryService.h"
#include "LevelSet/LevelSetVelocityExtensionConstraint.h"
#include "Mesh/Topology/CellTopology.h"
#include "Systems/FESystem.h"
#include "Systems/SystemCollectives.h"
#include "Systems/SystemState.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace svmp {
namespace FE {
namespace level_set {

std::array<bool, 3> strongZeroVelocityComponentMask(
    std::span<const int> effective_direction,
    int dimension)
{
  if (dimension <= 0 || dimension > 3) {
    throw std::invalid_argument(
        "strong zero-velocity component mask requires dimension 1, 2, or 3");
  }
  std::array<bool, 3> mask{true, true, true};
  if (effective_direction.empty()) {
    return mask;
  }

  std::array<bool, 3> requested{false, false, false};
  int active_count = 0;
  for (int d = 0; d < dimension; ++d) {
    const auto component = static_cast<std::size_t>(d);
    requested[component] =
        component < effective_direction.size() &&
        effective_direction[component] != 0;
    active_count += requested[component] ? 1 : 0;
  }
  // Match NavierStokesRegister.cpp exactly for a zero-valued steady
  // Dirichlet BC: only a nonempty proper subset changes the default
  // all-component constraint.
  if (active_count > 0 && active_count < dimension) {
    mask = requested;
  }
  return mask;
}

std::vector<double> orientedLevelSetForVelocityExtension(
    std::span<const double> level_set_values,
    double isovalue,
    LevelSetActiveSide retained_side)
{
  if (!std::isfinite(isovalue)) {
    throw std::invalid_argument(
        "velocity-extension orientation requires a finite isovalue");
  }
  std::vector<double> oriented(level_set_values.size(), 0.0);
  for (std::size_t vertex = 0u; vertex < level_set_values.size(); ++vertex) {
    const double centered = level_set_values[vertex] - isovalue;
    oriented[vertex] = retained_side == LevelSetActiveSide::Negative
                           ? centered
                           : -centered;
    if (!std::isfinite(oriented[vertex])) {
      throw std::invalid_argument(
          "velocity-extension orientation received a non-finite level-set value");
    }
  }
  return oriented;
}

std::optional<int> configuredInterfaceVelocityMarker(
    const svmp::FE::systems::FESystem& system,
    const LevelSetVelocityExtensionRequest& request)
{
  const auto* cut_context = system.cutIntegrationContext();
  if (request.requested_interface_marker >= 0) {
    return request.requested_interface_marker;
  }

  // Reconstruct the lifecycle's stable marker for the usual no-explicit-
  // marker configuration only when this request has keyed active-cut
  // provenance.  Integer-only system/context registration is insufficient:
  // an unregistered nodal request could otherwise match an unrelated marker.
  // The lifecycle registry rejects hash collisions rather than probing them.
  const auto phi_field = system.findFieldByName(request.level_set_field_name);
  if (request.active_cut_request_index.has_value() &&
      phi_field != svmp::FE::INVALID_FIELD_ID) {
    svmp::FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = svmp::FE::interfaces::LevelSetInterfaceSource::fromField(
        phi_field);
    key.domain_id = request.domain_id;
    key.isovalue = request.isovalue;
    key.requested_marker = -1;
    const int stable_marker =
        svmp::FE::interfaces::stableGeneratedInterfaceMarker(key);
    if (system.isGeneratedEmbeddedInterfaceMarkerRegistered(stable_marker) ||
        (cut_context != nullptr &&
         (cut_context->hasExpectedGeneratedSourceValueRevision(stable_marker) ||
          cut_context->hasGeneratedVolumeMarker(stable_marker) ||
          cut_context->hasGeneratedInterfaceMarker(stable_marker)))) {
      return stable_marker;
    }
  }

  if (cut_context != nullptr &&
      request.active_cut_request_index.has_value()) {
    const auto& markers = cut_context->generatedVolumeMarkers();
    const auto index = *request.active_cut_request_index;
    if (index < markers.size()) {
      return markers[index];
    } else {
      const auto& interface_markers =
          cut_context->generatedInterfaceMarkers();
      if (index < interface_markers.size()) {
        return interface_markers[index];
      }
    }
  }
  return std::nullopt;
}

std::optional<int> interfaceVelocitySampleMarker(
    const svmp::FE::systems::FESystem& system,
    const LevelSetVelocityExtensionRequest& request)
{
  const auto* cut_context = system.cutIntegrationContext();
  if (cut_context == nullptr) {
    return std::nullopt;
  }
  const auto marker = configuredInterfaceVelocityMarker(system, request);
  if (!marker.has_value() ||
      (!cut_context->hasGeneratedVolumeMarker(*marker) &&
       !cut_context->hasGeneratedInterfaceMarker(*marker))) {
    return std::nullopt;
  }
  return marker;
}

bool hasAuthoritativeInterfaceVelocityContext(
    const svmp::FE::systems::FESystem& system,
    const LevelSetVelocityExtensionRequest& request)
{
  const auto marker = configuredInterfaceVelocityMarker(system, request);
  const auto* cut_context = system.cutIntegrationContext();
  if (!marker.has_value()) {
    return false;
  }

  // The expected-source revision is registered even when this rank has no
  // retained rules for the generated interface.  Likewise, the FESystem
  // marker registration survives a locally empty cut context.  Either is an
  // authoritative declaration: a rank-local empty candidate set must remain
  // empty and must not be replaced by a nodal crossing search.
  return system.isGeneratedEmbeddedInterfaceMarkerRegistered(*marker) ||
         (cut_context != nullptr &&
          (cut_context->hasExpectedGeneratedSourceValueRevision(*marker) ||
           cut_context->hasGeneratedVolumeMarker(*marker) ||
           cut_context->hasGeneratedInterfaceMarker(*marker)));
}

std::vector<MeshIndex> interfaceVelocitySampleCandidateCells(
    const svmp::FE::systems::FESystem& system,
    const LevelSetVelocityExtensionRequest& request)
{
  const auto* cut_context = system.cutIntegrationContext();
  const auto marker = interfaceVelocitySampleMarker(system, request);
  if (cut_context == nullptr || !marker.has_value()) {
    return {};
  }
  cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
      system.meshAccess());

  std::vector<svmp::FE::MeshIndex> cells;
  if (cut_context->hasGeneratedVolumeMarker(*marker)) {
    cells = activeCutCellsForMarkerAndSide(
        *cut_context, *marker, request.active_side);
  }
  if (cut_context->hasGeneratedInterfaceMarker(*marker)) {
    for (const auto* rule : cut_context->interfaceRulesForMarker(*marker)) {
      if (rule == nullptr ||
          rule->kind != svmp::FE::geometry::CutQuadratureKind::Interface ||
          rule->provenance.parent_entity < 0) {
        continue;
      }
      cells.push_back(rule->provenance.parent_entity);
    }
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

bool refreshLevelSetVelocityExtensions(
    systems::FESystem& system,
    const svmp::Mesh& mesh,
    const systems::SystemStateView& state,
    std::span<const LevelSetVelocityExtensionRequest> requests,
    const LevelSetVelocityExtensionRefreshOptions& options,
    PdeVelocityExtensionCaches& caches)
{
  if (requests.empty()) {
    return false;
  }

  const auto extension_comm = systems::activeMeshCommunicator(system);
  const auto n_vertices = mesh.n_vertices();
  const int mesh_dim = mesh.dim();
  const bool trace_updates = options.trace;
  const auto activeSideContains =
      [](double phi, const LevelSetVelocityExtensionRequest& request) {
        return request.active_side == LevelSetActiveSide::Negative
                   ? phi <= request.isovalue
                   : phi >= request.isovalue;
      };

  bool updated = false;
  for (const auto& request : requests) {
    // The algebraic wall-compatible wet extension was retired (decisions D47
    // and D55); every request names a PDE method.
    const auto pde_operator =
        pdeVelocityExtensionOperatorFromToken(
            request.extension_method);
    if (!pde_operator.has_value()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set advection velocity "
          "extension method '" +
          request.extension_method +
          "' is not pde_harmonic or pde_normal; the algebraic "
          "wall-compatible wet extension was retired (decisions D47 and "
          "D55).");
    }
    const auto phi_field = system.findFieldByName(request.level_set_field_name);
    const auto source_field =
        system.findFieldByName(request.source_velocity_field_name);
    const auto target_field =
        system.findFieldByName(request.target_velocity_field_name);
    if (phi_field == svmp::FE::INVALID_FIELD_ID ||
        source_field == svmp::FE::INVALID_FIELD_ID ||
        target_field == svmp::FE::INVALID_FIELD_ID) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Could not find fields needed for the level-set advection velocity extension.");
    }

    const auto& phi_rec = system.fieldRecord(phi_field);
    const auto& target_rec = system.fieldRecord(target_field);
    auto extension_constraint =
        svmp::FE::level_set::
            findLevelSetVelocityExtensionConstraintKernel(
                system, request.operator_tag, target_field);
    const bool algebraic_extension =
        target_rec.source_kind ==
            svmp::FE::systems::FieldSourceKind::Unknown &&
        static_cast<bool>(extension_constraint);
    if (target_rec.source_kind !=
            svmp::FE::systems::FieldSourceKind::PrescribedData &&
        !algebraic_extension) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set advection velocity field '" +
          target_rec.name +
          "' must be either PrescribedData or an algebraic extension unknown with a registered exact constraint kernel.");
    }
    if (algebraic_extension &&
        extension_constraint->sourceVelocityField() != source_field) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Algebraic level-set velocity extension source does not match the registered physical velocity field.");
    }
    if (algebraic_extension && !options.refresh_frozen_rows) {
      // The rows depend on the known set and, for pde_normal, on the
      // level-set normals.  Freeze them throughout one nonlinear solve; only
      // E and the physical source velocity remain monolithic.
      continue;
    }
    if (algebraic_extension) {
      // Rebuilding the rows is fail-closed.  Any exception below, including
      // disappearance of the interface, must not leave the previous accepted
      // rows available to a later assembly.
      extension_constraint->invalidateFrozenMap();
    }

    const auto& source_rec = system.fieldRecord(source_field);
    if (source_rec.components <= 0 || target_rec.components <= 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set advection velocity update found an empty velocity field.");
    }
    const auto source_components =
        static_cast<std::size_t>(source_rec.components);
    const auto target_components =
        static_cast<std::size_t>(target_rec.components);
    const auto copy_components =
        std::min(source_components, target_components);
    // The extension reads and writes the fields at the mesh vertices, which
    // are the geometry nodes of the mesh: on a high-order cell they include
    // its edge, face and interior nodes.  Every field must therefore be a
    // fixed-order nodal Lagrange field whose nodes are exactly the mesh
    // vertices, i.e. whose order is the geometry order of every cell (P1 on
    // first-order cells; P2 on Triangle6, Quad9, Tetra10 or Hex27 cells).
    const auto require_nodal_field = [](const auto& record,
                                        std::string_view role) {
      if (!record.space || record.space->is_variable_order() ||
          record.space->polynomial_order() < 1 ||
          record.space->element().basis().basis_type() !=
              svmp::FE::BasisType::Lagrange) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] The level-set velocity "
            "extension requires a fixed-order nodal Lagrange " +
            std::string(role) + " field; field '" + record.name +
            "' is variable-order, non-nodal, or has no FE space.");
      }
    };
    if (phi_rec.components != 1) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] The level-set velocity "
          "extension requires a scalar level-set field.");
    }
    require_nodal_field(phi_rec, "level-set");
    require_nodal_field(source_rec, "source-velocity");
    require_nodal_field(target_rec, "target-velocity");
    if (source_components != target_components ||
        source_rec.space->value_dimension() !=
            target_rec.space->value_dimension()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] The level-set velocity extension requires source and target fields with identical component layouts.");
    }
    const int field_order = phi_rec.space->polynomial_order();
    if (source_rec.space->polynomial_order() != field_order ||
        target_rec.space->polynomial_order() != field_order) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] The level-set velocity extension "
          "requires level-set, source, and target fields of one polynomial "
          "order.");
    }
    const auto& target_dofs = system.fieldDofHandler(target_field);
    // Every mesh vertex carries one DOF per component of each field, and no
    // two vertices share a DOF.
    const auto require_vertex_nodes = [&](svmp::FE::FieldId field,
                                          std::size_t components) {
      const auto vertex_dofs = system.fieldDofsAtMeshVertices(field);
      std::vector<std::uint8_t> bound(
          static_cast<std::size_t>(system.fieldDofHandler(field).getNumDofs()),
          0u);
      bool complete = vertex_dofs.size() == n_vertices * components;
      for (std::size_t i = 0; complete && i < vertex_dofs.size(); ++i) {
        const auto dof = vertex_dofs[i];
        complete = dof >= 0 && static_cast<std::size_t>(dof) < bound.size() &&
                   bound[static_cast<std::size_t>(dof)] == 0u;
        if (complete) {
          bound[static_cast<std::size_t>(dof)] = 1u;
        }
      }
      if (!complete) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] The level-set velocity extension requires exactly one level-set, source, and target DOF per component at every mesh vertex.");
      }
    };
    require_vertex_nodes(phi_field, 1u);
    require_vertex_nodes(source_field, source_components);
    require_vertex_nodes(target_field, target_components);
    const auto& local_mesh = mesh.local_mesh();
    for (svmp::index_t cell = 0; cell < local_mesh.n_cells(); ++cell) {
      const int geometry_order = local_mesh.cell_shape(cell).order;
      if (geometry_order != field_order) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] The level-set velocity "
            "extension requires fields whose nodes are the mesh vertices: "
            "the field order " +
            std::to_string(field_order) +
            " differs from the geometry order " +
            std::to_string(geometry_order) + " of a mesh cell.");
      }
      auto [cell_vertices, cell_vertex_count] =
          local_mesh.cell_vertices_span(cell);
      const auto expected_cell_dofs =
          cell_vertex_count * source_components;
      if (cell_vertices == nullptr || cell_vertex_count == 0u ||
          phi_rec.space->dofs_per_element(cell) != cell_vertex_count ||
          source_rec.space->dofs_per_element(cell) != expected_cell_dofs ||
          target_rec.space->dofs_per_element(cell) != expected_cell_dofs) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] The level-set velocity extension requires identical nodal source/target cell layouts on every retained mesh cell.");
      }
    }

    const auto phi_values =
        evaluateVertexField(system,
                            mesh,
                            phi_field,
                            state,
                            1u,
                            "updating level-set advection velocity");
    const auto source_values =
        evaluateVertexField(system,
                            mesh,
                            source_field,
                            state,
                            source_components,
                            "updating level-set advection velocity");

    std::vector<std::uint8_t> active(n_vertices, 0u);
    std::vector<std::size_t> active_vertices;
    active_vertices.reserve(n_vertices);
    for (std::size_t v = 0; v < n_vertices; ++v) {
      const bool is_active = activeSideContains(phi_values[v], request);
      active[v] = is_active ? 1u : 0u;
      if (is_active) {
        active_vertices.push_back(v);
      }
    }

    std::vector<double> extended(n_vertices * target_components, 0.0);
    double max_active_speed = 0.0;
    double max_dry_extended_speed = 0.0;
    bool interface_sample_used_cut_context = false;
    std::size_t interface_sample_candidate_cell_count = 0u;
    std::size_t interface_sample_count = 0u;
    std::size_t interface_sample_context_rule_count = 0u;
    std::size_t trace_support_vertex_count = 0u;
    std::size_t dry_trace_support_vertex_count = 0u;
    std::size_t trace_seed_vertex_count = 0u;
    std::size_t wall_boundary_label_count = 0u;
    PdeVelocityExtensionReport pde_report{};
    std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow>
        algebraic_rows;
    // Set when the monolithic extension installs its rows.
    bool pde_algebraic = false;
    std::uint64_t pde_map_revision_key = 0u;

    auto record_speed = [&](std::size_t v) {
      double speed2 = 0.0;
      for (std::size_t c = 0; c < copy_components; ++c) {
        const auto value = extended[v * target_components + c];
        speed2 += value * value;
      }
      const double speed = std::sqrt(speed2);
      if (active[v]) {
        max_active_speed = std::max(max_active_speed, speed);
      } else {
        max_dry_extended_speed = std::max(max_dry_extended_speed, speed);
      }
    };

    if (mesh_dim < 1 || mesh_dim > 3) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] The level-set advection velocity "
          "extension requires mesh dimension in [1, 3].");
    }

    // For fixed nodal fields whose nodes are the mesh vertices (P1/Q1 on
    // first-order cells, P2/Q2 on quadratic cells, ...), equality at every
    // vertex of a retained interface cell is both necessary and sufficient
    // for the finite-element traces to agree pointwise on that cell:
    //
    //     E_h|_K = u_h|_K  =>  E_h|_(Gamma cap K) = u_h|_(Gamma cap K).
    //
    // The older sign-only seed marked wet vertices but graph-extrapolated
    // the dry vertices of cut cells, so E_h and u_h differed on Gamma even
    // though all wet nodal values agreed.  Prefer the generated cut context
    // (the geometry used by assembly), with a nodal crossing search only
    // when no authoritative generated-interface context exists anywhere on
    // the active communicator (on a high-order cell the nodal search sees
    // only sign changes between nodes).  A locally empty retained-rule set is
    // authoritative and must not resurrect an omitted cell through a nodal
    // fallback on that rank.
    const bool authoritative_interface_context = systems::collectiveAnyTrue(
        hasAuthoritativeInterfaceVelocityContext(system, request),
        extension_comm);
    auto candidate_cells = authoritative_interface_context
                               ? interfaceVelocitySampleCandidateCells(
                                     system, request)
                               : std::vector<svmp::FE::MeshIndex>{};
    if (!authoritative_interface_context) {
      candidate_cells = nodalVelocityExtensionInterfaceCells(
          mesh,
          std::span<const double>(phi_values.data(), phi_values.size()),
          request.isovalue);
    }
    interface_sample_used_cut_context = authoritative_interface_context;
    const auto local_owned_candidate_cells =
        static_cast<std::size_t>(std::count_if(
            candidate_cells.begin(),
            candidate_cells.end(),
            [&](svmp::FE::MeshIndex cell) {
              return cell >= 0 && mesh.is_owned_cell(
                                      static_cast<svmp::index_t>(cell));
            }));
    interface_sample_candidate_cell_count = systems::collectiveSumSize(
        local_owned_candidate_cells, extension_comm);

    std::vector<std::uint8_t> trace_support(n_vertices, 0u);
    markVelocityExtensionTraceSupportCells(
        mesh,
        std::span<const svmp::FE::MeshIndex>(candidate_cells.data(),
                                             candidate_cells.size()),
        trace_support);
    trace_support_vertex_count =
        synchronizeVelocityExtensionTraceSupportMask(
            mesh, extension_comm, trace_support);
    std::vector<std::uint8_t> trace_seed(active);
    std::vector<std::uint8_t> dry_trace_support(n_vertices, 0u);
    for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
      if (trace_support[vertex] != 0u) {
        trace_seed[vertex] = 1u;
        if (active[vertex] == 0u) {
          dry_trace_support[vertex] = 1u;
        }
      }
    }
    dry_trace_support_vertex_count =
        globalOwnedVelocityExtensionMaskCount(
            mesh, extension_comm, dry_trace_support);
    trace_seed_vertex_count = globalOwnedVelocityExtensionMaskCount(
        mesh, extension_comm, trace_seed);

    std::size_t local_interface_geometry_sample_count = 0u;

    auto add_interface_sample = [&](std::size_t va,
                                    std::size_t vb,
                                    double edge_t) {
      if (va >= n_vertices || vb >= n_vertices ||
          !std::isfinite(edge_t)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] The level-set velocity extension found an invalid nodal interface geometry sample.");
      }
      edge_t = std::clamp(edge_t, 0.0, 1.0);
      (void)edge_t;
      ++local_interface_geometry_sample_count;
    };

    auto add_cut_context_interface_samples = [&]() -> bool {
      const auto* cut_context = system.cutIntegrationContext();
      const auto marker = interfaceVelocitySampleMarker(system, request);
      if (cut_context == nullptr || !marker.has_value() ||
          !cut_context->hasGeneratedInterfaceMarker(*marker)) {
        return false;
      }
      bool added = false;
      for (const auto* rule :
           cut_context->interfaceRulesForMarker(*marker)) {
        if (rule == nullptr ||
            rule->kind != svmp::FE::geometry::CutQuadratureKind::Interface ||
            rule->provenance.parent_entity < 0) {
          continue;
        }
        ++interface_sample_context_rule_count;
        for (const auto& qp : rule->points) {
          if (!std::isfinite(qp.weight) ||
              !(std::abs(qp.weight) > 0.0)) {
            continue;
          }
          const auto& point = qp.point;
          if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
              !std::isfinite(point[2])) {
            throw std::runtime_error(
              "[svMultiPhysics::Application] The level-set advection velocity "
              "extension received a non-finite cut-interface sample.");
          }
          ++local_interface_geometry_sample_count;
          added = true;
        }
      }
      return added;
    };

    constexpr double zero_tol = 1.0e-12;
    auto add_interface_segment = [&](std::size_t va, std::size_t vb) {
      if (va >= n_vertices || vb >= n_vertices) {
        return;
      }
      const double phia = phi_values[va] - request.isovalue;
      const double phib = phi_values[vb] - request.isovalue;
      const bool a_zero = std::abs(phia) <= zero_tol;
      const bool b_zero = std::abs(phib) <= zero_tol;
      if (a_zero && b_zero) {
        add_interface_sample(va, vb, 0.0);
        add_interface_sample(va, vb, 1.0);
      } else if (a_zero) {
        add_interface_sample(va, vb, 0.0);
      } else if (b_zero) {
        add_interface_sample(va, vb, 1.0);
      } else if (phia * phib < 0.0) {
        add_interface_sample(va, vb, phia / (phia - phib));
      }
    };
    const bool used_cut_context_samples =
        add_cut_context_interface_samples();
    interface_sample_context_rule_count = systems::collectiveSumSize(
        interface_sample_context_rule_count, extension_comm);

    auto visit_cell_edges = [&](svmp::index_t cell) {
      if (cell < 0 || cell >= local_mesh.n_cells()) {
        return;
      }
      auto [cell_vertices, n_cell_vertices] =
          local_mesh.cell_vertices_span(cell);
      if (cell_vertices == nullptr || n_cell_vertices < 2u) {
        return;
      }
      const auto family = local_mesh.cell_shape(cell).family;
      if (family == svmp::CellFamily::Polygon) {
        const auto& shape = local_mesh.cell_shape(cell);
        const int polygon_corners =
            shape.num_corners > 0
                ? std::min<int>(shape.num_corners,
                                static_cast<int>(n_cell_vertices))
                : static_cast<int>(n_cell_vertices);
        if (polygon_corners < 2) {
          return;
        }
        const auto polygon_edges =
            svmp::CellTopology::get_polygon_edges_view(polygon_corners);
        for (int edge = 0; edge < polygon_edges.edge_count; ++edge) {
          auto edge_dofs = local_mesh.cell_edge_geometry_dofs(cell, edge);
          if (edge_dofs.size() < 2u) {
            const auto local_a = polygon_edges.pairs_flat[2 * edge];
            const auto local_b = polygon_edges.pairs_flat[2 * edge + 1];
            if (local_a < 0 || local_b < 0 ||
                static_cast<std::size_t>(local_a) >= n_cell_vertices ||
                static_cast<std::size_t>(local_b) >= n_cell_vertices) {
              continue;
            }
            edge_dofs = {
                cell_vertices[static_cast<std::size_t>(local_a)],
                cell_vertices[static_cast<std::size_t>(local_b)]};
          }
          for (std::size_t i = 0; i + 1u < edge_dofs.size(); ++i) {
            const auto raw_a = edge_dofs[i];
            const auto raw_b = edge_dofs[i + 1u];
            if (raw_a < 0 || raw_b < 0) {
              continue;
            }
            add_interface_segment(static_cast<std::size_t>(raw_a),
                                  static_cast<std::size_t>(raw_b));
          }
        }
        return;
      }

      const auto edge_view = svmp::CellTopology::get_edges_view(family);
      for (int edge = 0; edge < edge_view.edge_count; ++edge) {
        auto edge_dofs = local_mesh.cell_edge_geometry_dofs(cell, edge);
        if (edge_dofs.size() < 2u) {
          const auto local_a = edge_view.pairs_flat[2 * edge];
          const auto local_b = edge_view.pairs_flat[2 * edge + 1];
          if (local_a < 0 || local_b < 0 ||
              static_cast<std::size_t>(local_a) >= n_cell_vertices ||
              static_cast<std::size_t>(local_b) >= n_cell_vertices) {
            continue;
          }
          edge_dofs = {
              cell_vertices[static_cast<std::size_t>(local_a)],
              cell_vertices[static_cast<std::size_t>(local_b)]};
        }
        for (std::size_t i = 0; i + 1u < edge_dofs.size(); ++i) {
          const auto raw_a = edge_dofs[i];
          const auto raw_b = edge_dofs[i + 1u];
          if (raw_a < 0 || raw_b < 0) {
            continue;
          }
          add_interface_segment(static_cast<std::size_t>(raw_a),
                                static_cast<std::size_t>(raw_b));
        }
      }
    };
    if (!used_cut_context_samples && mesh_dim == 2) {
      if (authoritative_interface_context) {
        for (const auto cell : candidate_cells) {
          visit_cell_edges(static_cast<svmp::index_t>(cell));
        }
      } else {
        for (svmp::index_t cell = 0; cell < local_mesh.n_cells(); ++cell) {
          visit_cell_edges(cell);
        }
      }
    }

    interface_sample_count = globalVelocityExtensionGeometrySampleCount(
        local_interface_geometry_sample_count, extension_comm);
    if (interface_sample_count == 0u) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] The level-set velocity extension found no resolved interface geometry samples on any communicator rank; refusing to retain or install a stale extension map.");
    }
    if (trace_support_vertex_count == 0u) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] The level-set velocity extension resolved interface geometry but no trace-support vertices.");
    }

    std::vector<WallVelocityExtensionConstraint> wall_constraints;
    wall_constraints.reserve(request.wall_constraints.size());
    if (request.enforce_wall_impermeability) {
      if (request.wall_constraints.empty()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] The level-set velocity "
            "extension enforces wall impermeability, but no strong "
            "velocity Dirichlet component masks were resolved on the fluid "
            "walls. Give the walls strong velocity Dirichlet conditions: "
            "homogeneous (no-slip, or free-slip through "
            "Effective_direction) or with a prescribed velocity.");
      }
      for (const auto& wall : request.wall_constraints) {
        const auto label = mesh.label_from_name(wall.face_name);
        if (label == svmp::INVALID_LABEL) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Wet-extension wall face '" +
              wall.face_name + "' is not registered on the active mesh.");
        }
        wall_constraints.push_back(WallVelocityExtensionConstraint{
            .boundary_label = label,
            .constrained_components = strongZeroVelocityComponentMask(
                wall.effective_direction, mesh_dim),
            .prescribed_velocity = wall.prescribed_velocity});
      }
    }
    wall_boundary_label_count = wall_constraints.size();

    const auto oriented_level_set = orientedLevelSetForVelocityExtension(
        phi_values, request.isovalue, request.active_side);

    if (algebraic_extension) {
      // Monolithic coupling: the same discrete problem is installed as
      // frozen owner-local rows of the extension unknown.
      std::uint64_t free_surface_geometry_revision = 0u;
      if (const auto* cut_context = system.cutIntegrationContext()) {
        free_surface_geometry_revision = cut_context->contentRevision();
        if (const auto marker =
                interfaceVelocitySampleMarker(system, request);
            marker.has_value() &&
            cut_context->hasFreeSurfaceGeometrySnapshotForMarker(
                *marker)) {
          free_surface_geometry_revision =
              cut_context->freeSurfaceGeometrySnapshotRevisionForMarker(
                  *marker);
        }
      }
      const auto& mesh_access = system.meshAccess();
      // The frozen-map revision must agree across ranks.
      pde_map_revision_key = communicatorCombinedRevision(
          svmp::FE::level_set::velocityExtensionMapRevision(
              mesh_access.geometryRevision(),
              mesh_access.topologyRevision(),
              mesh_access.ownershipRevision(),
              mesh_access.numberingRevision(),
              free_surface_geometry_revision,
              std::span<const double>(oriented_level_set.data(),
                                      oriented_level_set.size()),
              std::span<const std::uint8_t>(trace_seed.data(),
                                            trace_seed.size()))
              .key(),
          extension_comm);
      pde_algebraic = true;
    }
    // Dry-region factorizations are reused while their content key is
    // unchanged (one cache per target field; see
    // PdeVelocityExtensionCache).
    auto& pde_extension_cache = caches[request.target_velocity_field_name];
    const auto& pde_mesh_access = system.meshAccess();
    pde_report = extendVelocityByPde(
        mesh,
        extension_comm,
        std::span<const double>(oriented_level_set.data(),
                                oriented_level_set.size()),
        std::span<const double>(source_values.data(),
                                source_values.size()),
        source_components,
        std::span<const std::uint8_t>(trace_seed.data(),
                                      trace_seed.size()),
        target_components,
        std::span<const WallVelocityExtensionConstraint>(
            wall_constraints),
        PdeVelocityExtensionOptions{
            .op = *pde_operator,
            // Every dry vertex is extended (no band truncation).
            .band_layers = 0,
            .enforce_wall_impermeability =
                request.enforce_wall_impermeability,
            .factorization = options.factorization},
        extended,
        pde_algebraic ? &algebraic_rows : nullptr,
        &pde_extension_cache,
        PdeVelocityExtensionMeshRevisions{
            .geometry = pde_mesh_access.geometryRevision(),
            .topology = pde_mesh_access.topologyRevision(),
            .ownership = pde_mesh_access.ownershipRevision(),
            .numbering = pde_mesh_access.numberingRevision()});
    // A list of the three components, joined by commas.
    const auto components = [](const std::array<std::size_t, 3>& counts) {
      return std::to_string(counts[0]) + "," + std::to_string(counts[1]) +
             "," + std::to_string(counts[2]);
    };
    if (trace_updates) {
      logRecord(
          LogRecord("LevelSetPdeVelocityExtension", "solve")
              .diagnostic("pde_extension_solve")
              .rootOnly()
              .text("method", request.extension_method)
              .value("known_vertices", pde_report.known_vertices)
              .value("extension_vertices", pde_report.extension_vertices)
              .value("outside_vertices", pde_report.outside_vertices)
              .value("extension_cells", pde_report.extension_cells)
              .text("unknowns", components(pde_report.unknowns))
              .text("wall_fixed", components(pde_report.wall_fixed))
              .real("max_relative_residual", pde_report.max_relative_residual)
              .real("max_known_speed", pde_report.max_known_speed)
              .real("max_extended_speed", pde_report.max_extended_speed)
              .real("max_wall_normal_velocity",
                    pde_report.max_wall_normal_velocity)
              .boolean("reused_factorization", pde_report.reused_factorization)
              .boolean("distributed_solves", pde_report.distributed_solves)
              .text("factorization",
                    pdeVelocityExtensionFactorizationName(options.factorization))
              .value("cache_entries", pde_report.cache_entries)
              .value("cache_capacity", pde_report.cache_capacity)
              .value("cache_hits", pde_extension_cache.statistics().hits)
              .value("cache_misses", pde_extension_cache.statistics().misses)
              .value("cache_bytes", pde_extension_cache.statistics().bytes)
              .value("cache_peak_bytes",
                     pde_extension_cache.statistics().peak_bytes)
              .value("solution_hash", pde_report.solution_hash)
              .real("elapsed_s", pde_report.elapsed_seconds)
              .real("component_solve_s", pde_report.component_solve_seconds));
    }
    if (pde_report.self_checked) {
      logRecord(
          LogRecord("LevelSetPdeVelocityExtension", "self-check")
              .diagnostic("pde_extension_self_check")
              .rootOnly()
              .boolean("reused_factorization", pde_report.reused_factorization)
              .boolean("distributed_solves", pde_report.distributed_solves)
              .value("cache_entries", pde_report.cache_entries)
              .integer("mismatches", 0));
    }
    constexpr double trace_equality_factor = 64.0;
    for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
      if (trace_support[vertex] == 0u) {
        continue;
      }
      for (std::size_t component = 0; component < copy_components;
           ++component) {
        const double source =
            source_values[vertex * source_components + component];
        const double extension =
            extended[vertex * target_components + component];
        const double tolerance =
            trace_equality_factor * std::numeric_limits<double>::epsilon() *
            std::max({1.0, std::abs(source), std::abs(extension)});
        if (!std::isfinite(source) || !std::isfinite(extension) ||
            std::abs(extension - source) > tolerance) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] The level-set velocity extension failed the exact nodal trace-support equality invariant.");
        }
      }
    }
    for (std::size_t v = 0; v < n_vertices; ++v) {
      record_speed(v);
    }

    max_active_speed = systems::collectiveMaxDouble(max_active_speed, extension_comm);
    max_dry_extended_speed =
        systems::collectiveMaxDouble(max_dry_extended_speed, extension_comm);
    const double velocity_scale_floor = 1.0e-12;
    // The prescribed wall velocities are data of the extension like the wet
    // velocities (zero without such walls).
    const double wet_to_dry_amplification =
        max_dry_extended_speed /
        std::max(std::max(max_active_speed,
                          pde_report.max_prescribed_wall_speed),
                 velocity_scale_floor);
    if (algebraic_extension &&
        wet_to_dry_amplification >
            kVelocityExtensionMaxWetToDryAmplification) {
      std::ostringstream message;
      message
          << "[svMultiPhysics::Application] The monolithic velocity "
             "extension refused to install a frozen map because the dry-region "
             "amplification exceeded its fixed guard: max_active_speed="
          << max_active_speed
          << " max_dry_extended_speed=" << max_dry_extended_speed
          << " amplification=" << wet_to_dry_amplification
          << " guard=" << kVelocityExtensionMaxWetToDryAmplification;
      throw std::runtime_error(message.str());
    }

    const auto* entity_map = target_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set advection velocity update requires vertex DOFs.");
    }
    std::size_t unassigned = 0u;
    if (algebraic_extension) {
      if (!pde_algebraic || algebraic_rows.empty()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] The monolithic velocity extension refresh produced no owned constraint rows.");
      }
      // The frozen-map revision agrees across ranks (see above).
      const std::uint64_t installed_map_revision_key = pde_map_revision_key;

      std::vector<svmp::FE::Real> projected_coefficients(
          static_cast<std::size_t>(target_dofs.getNumDofs()),
          svmp::FE::Real{0.0});
      std::vector<std::uint8_t> projected_assignment(
          projected_coefficients.size(), 0u);
      const auto projection =
          system.projectMeshVertexValuesToFieldCoefficients(
              target_field,
              std::span<const svmp::FE::Real>(extended.data(),
                                              extended.size()),
              target_components,
              std::span<svmp::FE::Real>(projected_coefficients.data(),
                                        projected_coefficients.size()),
              std::span<std::uint8_t>(projected_assignment.data(),
                                      projected_assignment.size()),
              "ApplicationDriver::refreshAlgebraicVelocityExtension");
      if (projection.unassigned_dofs != 0u || state.u_vector == nullptr) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] The monolithic velocity extension refresh requires a mutable state vector and a complete nodal preview projection.");
      }
      const auto owned_field_dofs =
          target_dofs.getPartition().locallyOwned().toVector();
      std::vector<svmp::FE::GlobalIndex> state_dofs;
      std::vector<svmp::FE::Real> state_values;
      state_dofs.reserve(owned_field_dofs.size());
      state_values.reserve(owned_field_dofs.size());
      const auto target_offset = system.fieldDofOffset(target_field);
      for (const auto local_dof : owned_field_dofs) {
        if (local_dof < 0 ||
            static_cast<std::size_t>(local_dof) >=
                projected_coefficients.size() ||
            projected_assignment[static_cast<std::size_t>(local_dof)] == 0u) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] The monolithic velocity extension preview omitted an owned extension DOF.");
        }
        state_dofs.push_back(target_offset + local_dof);
        state_values.push_back(
            projected_coefficients[static_cast<std::size_t>(local_dof)]);
      }
      auto* mutable_state =
          const_cast<svmp::FE::backends::GenericVector*>(state.u_vector);
      auto state_view = mutable_state->createAssemblyView();
      if (!state_view) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] The monolithic velocity extension refresh could not create a mutable state view.");
      }
      const bool map_changed =
          !extension_constraint->hasFrozenMap() ||
          extension_constraint->frozenMapRevision() !=
              installed_map_revision_key;
      extension_constraint->setFrozenRows(
          std::move(algebraic_rows),
          installed_map_revision_key);
      state_view->beginAssemblyPhase();
      state_view->setVectorEntries(
          std::span<const svmp::FE::GlobalIndex>(state_dofs.data(),
                                                 state_dofs.size()),
          std::span<const svmp::FE::Real>(state_values.data(),
                                          state_values.size()));
      state_view->endAssemblyPhase();
      state_view->finalizeAssembly();
      mutable_state->markModified();
      mutable_state->updateGhosts();
      if (trace_updates) {
        logRecord(
            LogRecord("LevelSetVelocityExtensionRefresh", "monolithic extension refreshed")
                .diagnostic("monolithic_extension_refresh")
                .rootOnly()
                .value("map_revision", installed_map_revision_key)
                .boolean("map_changed", map_changed)
                .value("reprojected_owned_dofs", state_dofs.size()));
      }
    } else {
      std::vector<svmp::FE::Real> coefficients(
          static_cast<std::size_t>(target_dofs.getNumDofs()),
          svmp::FE::Real{0.0});
      std::vector<std::uint8_t> assigned(coefficients.size(), 0u);

      const auto projection =
          system.projectMeshVertexValuesToFieldCoefficients(
              target_field,
              std::span<const svmp::FE::Real>(extended.data(),
                                              extended.size()),
              target_components,
              std::span<svmp::FE::Real>(coefficients.data(),
                                        coefficients.size()),
              std::span<std::uint8_t>(assigned.data(), assigned.size()),
              "ApplicationDriver::updateLevelSetAdvectionVelocities");
      if (projection.unassigned_dofs != 0u) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Level-set advection velocity update "
            "could not safely project " +
            std::to_string(projection.unassigned_dofs) +
            " target coefficient(s) from mesh vertices.");
      }
      for (const auto flag : assigned) {
        unassigned += flag ? 0u : 1u;
      }
      system.setPrescribedFieldCoefficients(
          target_field,
          std::span<const svmp::FE::Real>(coefficients.data(),
                                          coefficients.size()));
    }
    updated = true;
    if (trace_updates) {
      logRecord(
          LogRecord("LevelSetVelocityExtensionRefresh", "advection velocity updated")
              .diagnostic("level_set_advection_velocity_update")
              .rootOnly()
              .quoted("field", request.target_velocity_field_name)
              .quoted("source", request.source_velocity_field_name)
              .value("extension_method", request.extension_method)
              .text("linearization", algebraic_extension ? "frozen_sparse_algebraic" : "prescribed")
              .value("active_vertices", active_vertices.size())
              .value("dry_vertices", n_vertices - active_vertices.size())
              .value("interface_sample_candidates", interface_sample_candidate_cell_count)
              .value("interface_sample_context_rules", interface_sample_context_rule_count)
              .value("interface_samples", interface_sample_count)
              .value("interface_geometry_samples", interface_sample_count)
              .text("interface_geometry_sample_scope", "summed_rank_local")
              .text("interface_geometry_sample_exchange", "checked_scalar_allreduce")
              .integer("interface_samples_used_for_constraints", 0)
              .text("interface_sample_source", interface_sample_used_cut_context ? "cut_context" : "all_cells")
              .text("trace_constraint_source", "cut_cell_p1_support")
              .value("trace_support_vertices", trace_support_vertex_count)
              .value("dry_trace_support_vertices", dry_trace_support_vertex_count)
              .value("trace_seed_vertices", trace_seed_vertex_count)
              .value("extended_vertices", pde_report.extension_vertices)
              .value("vertices_outside_extension_band", pde_report.outside_vertices)
              .value("map_revision", pde_algebraic ? pde_map_revision_key : 0u)
              .value("map_max_seed_speed", pde_report.max_known_speed)
              .value("map_max_extended_speed", pde_report.max_extended_speed)
              .value("wall_boundary_labels", wall_boundary_label_count)
              .value("max_wall_normal_velocity", pde_report.max_wall_normal_velocity)
              .value("max_active_speed", max_active_speed)
              .value("max_dry_extended_speed", max_dry_extended_speed)
              .value("wet_to_dry_amplification", wet_to_dry_amplification)
              .value("wet_to_dry_amplification_guard", kVelocityExtensionMaxWetToDryAmplification)
              .text("target_value_semantics", algebraic_extension ? "generalized_alpha_stage_unknown" : "prescribed_current_coefficients")
              .text("computed_extension_semantics", "current_state_constraint_preview")
              .value("unassigned_dofs", unassigned));
    }
  }

  return updated;
}

LevelSetVelocityExtensionParticipant::LevelSetVelocityExtensionParticipant(
    systems::FESystem& system,
    const svmp::Mesh& mesh,
    std::vector<LevelSetVelocityExtensionRequest> requests,
    Options options,
    PdeVelocityExtensionCaches& caches)
    : system_(&system),
      mesh_(&mesh),
      requests_(std::move(requests)),
      options_(options),
      caches_(&caches)
{
}

void LevelSetVelocityExtensionParticipant::synchronize(
    const systems::SystemStateView& state,
    timestepping::GeneratedStateSync& sync)
{
  const bool refresh_frozen_rows =
      options_.solve == timestepping::GeneratedStateSolve::Steady
          ? options_.outer_fixed_point
          : timestepping::refreshesFrozenLevelSetExtensionAtStateSync(
                sync.point, options_.outer_fixed_point);
  (void)refreshLevelSetVelocityExtensions(
      *system_,
      *mesh_,
      state,
      requests_,
      LevelSetVelocityExtensionRefreshOptions{
          .refresh_frozen_rows = refresh_frozen_rows,
          .factorization = options_.factorization,
          .trace = options_.trace},
      *caches_);
}

void LevelSetVelocityExtensionParticipant::invalidate()
{
  for (const auto& request : requests_) {
    const auto target =
        system_->findFieldByName(request.target_velocity_field_name);
    if (target == INVALID_FIELD_ID) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Cut-topology rejection could not invalidate a level-set velocity extension target field.");
    }
    auto extension_constraint = findLevelSetVelocityExtensionConstraintKernel(
        *system_, request.operator_tag, target);
    const bool algebraic_extension =
        system_->fieldRecord(target).source_kind ==
        systems::FieldSourceKind::Unknown;
    if (algebraic_extension && !extension_constraint) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Cut-topology rejection could not resolve a frozen level-set velocity extension kernel.");
    }
    if (extension_constraint) {
      extension_constraint->invalidateFrozenMap();
    }
  }
}

} // namespace level_set
} // namespace FE
} // namespace svmp
