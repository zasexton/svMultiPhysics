/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FreeSurfaceCutStability.cpp
 * @brief Fixed and evolving-cut numerical stability regressions for the
 * unfitted free-surface block.
 *
 * The test deliberately goes beyond formulation-tree assertions.  For each
 * cut position it regenerates production level-set cut quadrature, constructs
 * the same active-side cut-adjacent facet set as ApplicationDriver, rebuilds
 * the active-support and AgFEM constraints, and assembles the actual transient
 * Navier--Stokes VMS/PSPG Jacobian with pressure ghost stabilization enabled.
 * The embedded natural-traction boundary anchors absolute pressure without a
 * pressure gauge constraint.  The test reduces by every registered constraint
 * line, including the componentwise velocity gauges and
 * active-support/aggregation lines, checks every remaining pressure row, and
 * evaluates the exact infinity-norm condition of the equilibrated matrix.  A
 * complementary sequence refreshes moving cuts on one persistent FESystem and
 * revisits an earlier position to detect stale quadrature/facet/constraint
 * state.  These are bounded algebraic regressions, not an inf-sup theorem.
 */

#include <gtest/gtest.h>

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Assembly/StandardAssembler.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/Vocabulary.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "FE/LevelSet/LevelSetInterfaceLifecycle.h"
#include "FE/Math/DenseLinearAlgebra.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/CutIntegrationInvalidation.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/TimeIntegrator.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#  include "Mesh/Topology/CellShape.h"
#endif

#if defined(FE_HAS_MPI) || defined(MESH_HAS_MPI)
#  include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp::Physics::test {
namespace {

namespace ns = formulations::navier_stokes;

[[nodiscard]] std::string realPropertyValue(FE::Real value)
{
    std::ostringstream output;
    output << std::setprecision(
                  std::numeric_limits<FE::Real>::max_digits10)
           << value;
    return output.str();
}

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* key, const char* value)
        : key_(key)
    {
        if (const char* prior = std::getenv(key_); prior != nullptr) {
            prior_ = std::string(prior);
        }
        ::setenv(key_, value, 1);
    }

    ~ScopedEnvVar()
    {
        if (prior_.has_value()) {
            ::setenv(key_, prior_->c_str(), 1);
        } else {
            ::unsetenv(key_);
        }
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    const char* key_;
    std::optional<std::string> prior_{};
};

struct PlaneCutPosition {
    std::string label;
    std::array<FE::Real, 3> normal;
    FE::Real offset{0.0};

    [[nodiscard]] FE::Real value(const std::array<FE::Real, 3>& x) const noexcept
    {
        return normal[0] * x[0] + normal[1] * x[1] + normal[2] * x[2] -
               offset;
    }
};

struct FieldConstraintCounts {
    std::size_t master_bearing{0u};
    std::size_t homogeneous_pins{0u};
};

struct AggregationConstraintMetrics {
    std::size_t master_bearing_lines{0u};
    std::size_t master_entries{0u};
    std::size_t maximum_masters_per_line{0u};
    FE::Real maximum_partition_of_unity_error{0.0};
    FE::Real maximum_inhomogeneity{0.0};
    FE::Real maximum_absolute_weight{0.0};
    FE::Real maximum_weight_l1{0.0};
    // The production BFS root distance is not retained in AffineConstraints.
    // This directly observable geometric proxy bounds how far the closed
    // extension reaches from a pressure slave to any surviving root master.
    FE::Real maximum_slave_master_distance_over_h{0.0};
};

struct PressureControlMetrics {
    std::size_t generalized_coupling_rank{0u};
    std::size_t pressure_dimension{0u};
    FE::Real generalized_coupling_smallest_singular_value{0.0};
    FE::Real stabilized_schur_smallest_generalized_eigenvalue{0.0};
    FE::Real stabilized_pressure_control{0.0};
    FE::Real constant_pressure_control{0.0};
    FE::Real velocity_block_relative_skew{0.0};
    FE::Real pressure_gradient_adjoint_relative_defect{0.0};
};

struct StabilityRegime {
    std::string_view id{"baseline"};
    FE::Real density{1.0};
    FE::Real viscosity{0.01};
    FE::Real dt{0.1};
    bool convection{false};
    FE::Real advective_speed{0.0};
};

struct KrylovTelemetry {
    std::size_t iterations{0u};
    std::size_t iteration_limit{0u};
    std::size_t diagonal_fallback_count{0u};
    bool converged{false};
    bool breakdown{false};
    FE::Real relative_residual{
        std::numeric_limits<FE::Real>::infinity()};
    FE::Real relative_solution_error{
        std::numeric_limits<FE::Real>::infinity()};
};

struct StabilitySample {
    std::string label;
    FE::Real minimum_active_cut_fraction{1.0};
    FE::Real designated_cut_fraction{
        std::numeric_limits<FE::Real>::quiet_NaN()};
    FE::Real reference_active_volume{0.0};
    FE::Real physical_active_volume{0.0};
    std::size_t cut_cells{0u};
    std::size_t cut_adjacent_facets{0u};
    std::uint64_t cut_adjacent_facet_gid_hash{0u};
    std::size_t pruned_volume_rules{0u};
    std::size_t backend_volume_quadrature_points{0u};
    std::size_t backend_fallback_cells{0u};
    bool pressure_natural_traction_anchor{false};
    bool pressure_anchor_has_no_gauge_enforcement{false};
    FieldConstraintCounts velocity_constraints{};
    FieldConstraintCounts pressure_constraints{};
    AggregationConstraintMetrics pressure_aggregation{};
    PressureControlMetrics pressure_control{};
    int mesh_cells_per_axis{0};
    FE::Real mesh_spacing{0.0};
    std::size_t free_velocity_dofs{0u};
    std::size_t free_pressure_dofs{0u};
    std::size_t zero_free_pressure_rows{0u};
    FE::Real pressure_ghost_norm{0.0};
    FE::Real pspg_pressure_gradient_norm{0.0};
    // Distributed samples retain canonically ordered, unconstrained operators
    // so different global DOF numberings can be compared entry-for-entry.
    std::vector<FE::Real> canonical_mixed_operator{};
    std::vector<FE::Real> canonical_pressure_ghost_operator{};
    std::vector<FE::Real> canonical_pressure_pspg_operator{};
    std::size_t equilibrated_rank{0u};
    std::size_t equilibrated_size{0u};
    FE::Real equilibrated_smallest_singular_value{0.0};
    FE::Real equilibrated_largest_singular_value{0.0};
    FE::Real equilibrated_condition_inf{
        std::numeric_limits<FE::Real>::infinity()};
    KrylovTelemetry krylov{};
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

class PhysicalCutVolumeMeasureKernel final
    : public FE::assembly::AssemblyKernel {
public:
    [[nodiscard]] FE::assembly::RequiredData getRequiredData() const override
    {
        return FE::assembly::RequiredData::BasisValues |
               FE::assembly::RequiredData::IntegrationWeights;
    }

    void computeCell(const FE::assembly::AssemblyContext& context,
                     FE::assembly::KernelOutput& output) override
    {
        const auto n_test = context.numTestDofs();
        output.reserve(n_test,
                       context.numTrialDofs(),
                       /*need_matrix=*/false,
                       /*need_vector=*/true);
        for (FE::LocalIndex q = 0; q < context.numQuadraturePoints(); ++q) {
            const auto weight = context.integrationWeight(q);
            for (FE::LocalIndex i = 0; i < n_test; ++i) {
                output.vectorEntry(i) +=
                    weight * context.basisValue(i, q);
            }
        }
    }
};

[[nodiscard]] FE::GlobalIndex structuredVertex(int i,
                                               int j,
                                               int k,
                                               int nodes_per_axis)
{
    return static_cast<FE::GlobalIndex>(
        i + nodes_per_axis * (j + nodes_per_axis * k));
}

[[nodiscard]] std::shared_ptr<Mesh> makeFixedTetraMesh(
    const PlaneCutPosition& cut,
    int cells_per_axis = 2)
{
    if (cells_per_axis < 2) {
        throw std::invalid_argument(
            "fixed-sweep tetra mesh requires at least two cells per axis");
    }
    auto base = std::make_shared<MeshBase>();
    const int nodes_per_axis = cells_per_axis + 1;
    const auto node_count = static_cast<std::size_t>(nodes_per_axis) *
                            static_cast<std::size_t>(nodes_per_axis) *
                            static_cast<std::size_t>(nodes_per_axis);
    const auto cell_count = std::size_t{6u} *
                            static_cast<std::size_t>(cells_per_axis) *
                            static_cast<std::size_t>(cells_per_axis) *
                            static_cast<std::size_t>(cells_per_axis);
    const FE::Real spacing =
        FE::Real{2.0} / static_cast<FE::Real>(cells_per_axis);

    std::vector<real_t> coordinates;
    coordinates.reserve(node_count * 3u);
    for (int k = 0; k < nodes_per_axis; ++k) {
        for (int j = 0; j < nodes_per_axis; ++j) {
            for (int i = 0; i < nodes_per_axis; ++i) {
                coordinates.push_back(
                    static_cast<real_t>(spacing * static_cast<FE::Real>(i)));
                coordinates.push_back(
                    static_cast<real_t>(spacing * static_cast<FE::Real>(j)));
                coordinates.push_back(
                    static_cast<real_t>(spacing * static_cast<FE::Real>(k)));
            }
        }
    }

    std::vector<offset_t> cell_offsets{0};
    std::vector<index_t> cell_vertices;
    cell_offsets.reserve(cell_count + 1u);
    cell_vertices.reserve(cell_count * 4u);
    // Freudenthal subdivision about the 000--111 body diagonal.  Reusing the
    // same orientation in every cube gives matching diagonals on shared faces.
    constexpr std::array<std::array<std::size_t, 4>, 6> tetrahedra = {{
        {{0, 1, 2, 6}},
        {{0, 2, 3, 6}},
        {{0, 3, 7, 6}},
        {{0, 7, 4, 6}},
        {{0, 4, 5, 6}},
        {{0, 5, 1, 6}},
    }};
    for (int k = 0; k < cells_per_axis; ++k) {
        for (int j = 0; j < cells_per_axis; ++j) {
            for (int i = 0; i < cells_per_axis; ++i) {
                const std::array<FE::GlobalIndex, 8> nodes = {
                    structuredVertex(i, j, k, nodes_per_axis),
                    structuredVertex(i + 1, j, k, nodes_per_axis),
                    structuredVertex(i + 1, j + 1, k, nodes_per_axis),
                    structuredVertex(i, j + 1, k, nodes_per_axis),
                    structuredVertex(i, j, k + 1, nodes_per_axis),
                    structuredVertex(i + 1, j, k + 1, nodes_per_axis),
                    structuredVertex(i + 1, j + 1, k + 1, nodes_per_axis),
                    structuredVertex(i, j + 1, k + 1, nodes_per_axis),
                };
                for (const auto& tetra : tetrahedra) {
                    for (const auto local : tetra) {
                        cell_vertices.push_back(
                            static_cast<index_t>(nodes[local]));
                    }
                    cell_offsets.push_back(
                        static_cast<offset_t>(cell_vertices.size()));
                }
            }
        }
    }

    const CellShape shape{CellFamily::Tetra, 4, 1};
    base->build_from_arrays(
        /*spatial_dim=*/3,
        coordinates,
        cell_offsets,
        cell_vertices,
        std::vector<CellShape>(cell_count, shape));
    base->finalize();

    const auto phi_handle = MeshFields::attach_field(
        *base,
        EntityKind::Vertex,
        "phi",
        FieldScalarType::Float64,
        1);
    auto* phi = MeshFields::field_data_as<real_t>(*base, phi_handle);
    if (phi == nullptr) {
        throw std::runtime_error("failed to allocate fixed-sweep level-set field");
    }
    for (index_t vertex = 0;
         vertex < static_cast<index_t>(base->n_vertices());
         ++vertex) {
        const auto x = base->get_vertex_coords(vertex);
        phi[static_cast<std::size_t>(vertex)] = static_cast<real_t>(
            cut.value({{static_cast<FE::Real>(x[0]),
                        static_cast<FE::Real>(x[1]),
                        static_cast<FE::Real>(x[2])}}));
    }

#if defined(MESH_HAS_MPI)
    return create_mesh(std::move(base), MeshComm(MPI_COMM_SELF));
#else
    return create_mesh(std::move(base));
#endif
}

[[nodiscard]] std::shared_ptr<Mesh> makeManufacturedOpenTankQuadMesh(
    const PlaneCutPosition& cut,
    int left_marker,
    int right_marker,
    int bottom_marker,
    int top_marker)
{
    auto base = std::make_shared<MeshBase>();
    const std::vector<real_t> coordinates = {
        -1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        -1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
        -1.0, 2.0, 0.0, 2.0, 1.0, 2.0,
    };
    const std::vector<offset_t> cell_offsets = {0, 4, 8, 12, 16};
    const std::vector<index_t> cell_vertices = {
        0, 1, 4, 3,
        1, 2, 5, 4,
        3, 4, 7, 6,
        4, 5, 8, 7,
    };
    const CellShape shape{CellFamily::Quad, 4, 1};
    base->build_from_arrays(
        /*spatial_dim=*/2,
        coordinates,
        cell_offsets,
        cell_vertices,
        std::vector<CellShape>(4u, shape));
    base->finalize();

    base->register_label("manufactured_wall_left",
                         static_cast<label_t>(left_marker));
    base->register_label("manufactured_wall_right",
                         static_cast<label_t>(right_marker));
    base->register_label("manufactured_wall_bottom",
                         static_cast<label_t>(bottom_marker));
    base->register_label("manufactured_wall_top",
                         static_cast<label_t>(top_marker));

    const auto coordinate = [&](index_t vertex, int component) {
        return base->X_ref().at(
            static_cast<std::size_t>(2 * vertex + component));
    };
    const auto allVerticesMatch = [&](std::span<const index_t> vertices,
                                      int component,
                                      real_t value) {
        return std::all_of(
            vertices.begin(), vertices.end(), [&](index_t vertex) {
                return std::abs(coordinate(vertex, component) - value) <
                       real_t{1.0e-14};
            });
    };
    for (index_t face = 0;
         face < static_cast<index_t>(base->n_faces());
         ++face) {
        const auto vertices = base->face_vertices(face);
        if (vertices.size() != 2u) {
            continue;
        }
        label_t marker = INVALID_LABEL;
        if (allVerticesMatch(vertices, /*component=*/1, real_t{0.0})) {
            marker = static_cast<label_t>(bottom_marker);
        } else if (allVerticesMatch(vertices, /*component=*/1, real_t{2.0})) {
            marker = static_cast<label_t>(top_marker);
        } else if (allVerticesMatch(vertices, /*component=*/0, real_t{-1.0})) {
            marker = static_cast<label_t>(left_marker);
        } else if (allVerticesMatch(vertices, /*component=*/0, real_t{1.0})) {
            marker = static_cast<label_t>(right_marker);
        }
        if (marker != INVALID_LABEL) {
            base->set_boundary_label(face, marker);
        }
    }

    const auto phi_handle = MeshFields::attach_field(
        *base,
        EntityKind::Vertex,
        "phi",
        FieldScalarType::Float64,
        1);
    auto* phi = MeshFields::field_data_as<real_t>(*base, phi_handle);
    if (phi == nullptr) {
        throw std::runtime_error(
            "failed to allocate manufactured level-set field");
    }
    for (index_t vertex = 0;
         vertex < static_cast<index_t>(base->n_vertices());
         ++vertex) {
        const auto x = base->get_vertex_coords(vertex);
        phi[static_cast<std::size_t>(vertex)] = static_cast<real_t>(
            cut.value({{static_cast<FE::Real>(x[0]),
                        static_cast<FE::Real>(x[1]),
                        FE::Real{0.0}}}));
    }

    return create_mesh(std::move(base));
}

[[nodiscard]] std::shared_ptr<Mesh> makePartialSlipSmallCutQuadStrip(
    const PlaneCutPosition& cut,
    int left_marker,
    int right_marker,
    int bottom_marker,
    int top_marker)
{
    auto base = std::make_shared<MeshBase>();
    // Three Q1 cells in x.  With phi=x-0.05 and the negative side active,
    // cell 0 is full-active, cell 1 has a 5% active sliver, and cell 2 is
    // inactive.  Vertex 2 is therefore an unsupported cut-cell node on the
    // interior of the bottom wall, while vertex 6 is its fully constrained
    // counterpart on the top wall.
    const std::vector<real_t> coordinates = {
        -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0,
        -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0,
    };
    const std::vector<offset_t> cell_offsets = {0, 4, 8, 12};
    const std::vector<index_t> cell_vertices = {
        0, 1, 5, 4,
        1, 2, 6, 5,
        2, 3, 7, 6,
    };
    const CellShape shape{CellFamily::Quad, 4, 1};
    base->build_from_arrays(
        /*spatial_dim=*/2,
        coordinates,
        cell_offsets,
        cell_vertices,
        std::vector<CellShape>(3u, shape));
    base->finalize();

    base->register_label("partial_slip_wall_left",
                         static_cast<label_t>(left_marker));
    base->register_label("partial_slip_wall_right",
                         static_cast<label_t>(right_marker));
    base->register_label("partial_slip_wall_bottom",
                         static_cast<label_t>(bottom_marker));
    base->register_label("partial_slip_wall_top",
                         static_cast<label_t>(top_marker));

    const auto coordinate = [&](index_t vertex, int component) {
        return base->X_ref().at(
            static_cast<std::size_t>(2 * vertex + component));
    };
    const auto allVerticesMatch = [&](std::span<const index_t> vertices,
                                      int component,
                                      real_t value) {
        return std::all_of(
            vertices.begin(), vertices.end(), [&](index_t vertex) {
                return std::abs(coordinate(vertex, component) - value) <
                       real_t{1.0e-14};
            });
    };
    for (index_t face = 0;
         face < static_cast<index_t>(base->n_faces());
         ++face) {
        const auto vertices = base->face_vertices(face);
        if (vertices.size() != 2u) {
            continue;
        }
        label_t marker = INVALID_LABEL;
        if (allVerticesMatch(vertices, /*component=*/1, real_t{0.0})) {
            marker = static_cast<label_t>(bottom_marker);
        } else if (allVerticesMatch(vertices, /*component=*/1, real_t{1.0})) {
            marker = static_cast<label_t>(top_marker);
        } else if (allVerticesMatch(vertices, /*component=*/0, real_t{-1.0})) {
            marker = static_cast<label_t>(left_marker);
        } else if (allVerticesMatch(vertices, /*component=*/0, real_t{2.0})) {
            marker = static_cast<label_t>(right_marker);
        }
        if (marker != INVALID_LABEL) {
            base->set_boundary_label(face, marker);
        }
    }

    const auto phi_handle = MeshFields::attach_field(
        *base,
        EntityKind::Vertex,
        "phi",
        FieldScalarType::Float64,
        1);
    auto* phi = MeshFields::field_data_as<real_t>(*base, phi_handle);
    if (phi == nullptr) {
        throw std::runtime_error(
            "failed to allocate partial-slip small-cut level-set field");
    }
    for (index_t vertex = 0;
         vertex < static_cast<index_t>(base->n_vertices());
         ++vertex) {
        const auto x = base->get_vertex_coords(vertex);
        phi[static_cast<std::size_t>(vertex)] = static_cast<real_t>(
            cut.value({{static_cast<FE::Real>(x[0]),
                        static_cast<FE::Real>(x[1]),
                        FE::Real{0.0}}}));
    }

    return create_mesh(std::move(base));
}

struct WetBlockQuadStripArrays {
    std::vector<real_t> coordinates{};
    std::vector<offset_t> cell_offsets{};
    std::vector<index_t> cell_vertices{};
    std::vector<CellShape> cell_shapes{};
    std::vector<real_t> level_set{};
};

[[nodiscard]] WetBlockQuadStripArrays makeWetBlockQuadStripArrays(
    std::span<const FE::Real> x_coordinates,
    std::span<const FE::Real> level_set_by_plane)
{
    if (x_coordinates.size() < 3u ||
        x_coordinates.size() != level_set_by_plane.size()) {
        throw std::invalid_argument(
            "wet-block strip requires matching coordinates and level-set values on at least three planes");
    }
    for (std::size_t plane = 0u; plane < x_coordinates.size(); ++plane) {
        if (!std::isfinite(x_coordinates[plane]) ||
            !std::isfinite(level_set_by_plane[plane]) ||
            (plane != 0u &&
             !(x_coordinates[plane] > x_coordinates[plane - 1u]))) {
            throw std::invalid_argument(
                "wet-block strip coordinates and level-set values must be finite with strictly increasing coordinates");
        }
    }

    WetBlockQuadStripArrays arrays;
    arrays.coordinates.reserve(2u * x_coordinates.size() * 2u);
    arrays.level_set.reserve(2u * level_set_by_plane.size());
    for (std::size_t plane = 0u; plane < x_coordinates.size(); ++plane) {
        for (int y = 0; y < 2; ++y) {
            arrays.coordinates.push_back(
                static_cast<real_t>(x_coordinates[plane]));
            arrays.coordinates.push_back(static_cast<real_t>(y));
            arrays.level_set.push_back(
                static_cast<real_t>(level_set_by_plane[plane]));
        }
    }

    arrays.cell_offsets.push_back(0);
    arrays.cell_vertices.reserve(4u * (x_coordinates.size() - 1u));
    arrays.cell_shapes.reserve(x_coordinates.size() - 1u);
    const CellShape shape{CellFamily::Quad, 4, 1};
    for (std::size_t cell = 0u; cell + 1u < x_coordinates.size(); ++cell) {
        const auto left = static_cast<index_t>(2u * cell);
        const auto right = static_cast<index_t>(2u * (cell + 1u));
        arrays.cell_vertices.insert(
            arrays.cell_vertices.end(),
            {left, right, static_cast<index_t>(right + 1),
             static_cast<index_t>(left + 1)});
        arrays.cell_offsets.push_back(
            static_cast<offset_t>(arrays.cell_vertices.size()));
        arrays.cell_shapes.push_back(shape);
    }
    return arrays;
}

void attachWetBlockLevelSet(MeshBase& mesh,
                            std::span<const real_t> level_set)
{
    if (level_set.size() != mesh.n_vertices()) {
        throw std::invalid_argument(
            "wet-block level-set data does not match the mesh vertices");
    }
    const auto handle = MeshFields::attach_field(
        mesh,
        EntityKind::Vertex,
        "phi",
        FieldScalarType::Float64,
        1);
    auto* values = MeshFields::field_data_as<real_t>(mesh, handle);
    if (values == nullptr) {
        throw std::runtime_error(
            "failed to allocate wet-block level-set field");
    }
    std::copy(level_set.begin(), level_set.end(), values);
}

[[nodiscard]] std::shared_ptr<Mesh> makeWetBlockQuadStrip(
    std::span<const FE::Real> x_coordinates,
    std::span<const FE::Real> level_set_by_plane)
{
    const auto arrays = makeWetBlockQuadStripArrays(
        x_coordinates, level_set_by_plane);
    auto base = std::make_shared<MeshBase>();
    base->build_from_arrays(
        /*spatial_dim=*/2,
        arrays.coordinates,
        arrays.cell_offsets,
        arrays.cell_vertices,
        arrays.cell_shapes);
    base->finalize();
    attachWetBlockLevelSet(*base, arrays.level_set);
    return create_mesh(std::move(base));
}

struct TetraStripArrays {
    std::vector<real_t> coordinates{};
    std::vector<offset_t> cell_offsets{};
    std::vector<index_t> cell_vertices{};
    std::vector<CellShape> cell_shapes{};
};

[[nodiscard]] FE::GlobalIndex stripVertex(int i, int j, int k)
{
    constexpr int nodes_x = 4;
    constexpr int nodes_y = 2;
    return static_cast<FE::GlobalIndex>(i + nodes_x * (j + nodes_y * k));
}

[[nodiscard]] TetraStripArrays makeThreeCubeTetraStripArrays()
{
    TetraStripArrays arrays;
    arrays.coordinates.reserve(16u * 3u);
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 4; ++i) {
                arrays.coordinates.push_back(static_cast<real_t>(i));
                arrays.coordinates.push_back(static_cast<real_t>(j));
                arrays.coordinates.push_back(static_cast<real_t>(k));
            }
        }
    }

    constexpr std::array<std::array<std::size_t, 4>, 6> tetrahedra = {{
        {{0, 1, 2, 6}},
        {{0, 2, 3, 6}},
        {{0, 3, 7, 6}},
        {{0, 7, 4, 6}},
        {{0, 4, 5, 6}},
        {{0, 5, 1, 6}},
    }};
    arrays.cell_offsets = {0};
    arrays.cell_offsets.reserve(19u);
    arrays.cell_vertices.reserve(18u * 4u);
    for (int i = 0; i < 3; ++i) {
        const std::array<FE::GlobalIndex, 8> nodes = {
            stripVertex(i, 0, 0),
            stripVertex(i + 1, 0, 0),
            stripVertex(i + 1, 1, 0),
            stripVertex(i, 1, 0),
            stripVertex(i, 0, 1),
            stripVertex(i + 1, 0, 1),
            stripVertex(i + 1, 1, 1),
            stripVertex(i, 1, 1),
        };
        for (const auto& tetra : tetrahedra) {
            for (const auto local : tetra) {
                arrays.cell_vertices.push_back(
                    static_cast<index_t>(nodes[local]));
            }
            arrays.cell_offsets.push_back(
                static_cast<offset_t>(arrays.cell_vertices.size()));
        }
    }
    arrays.cell_shapes.assign(
        18u, CellShape{CellFamily::Tetra, 4, 1});
    return arrays;
}

[[nodiscard]] FE::GlobalIndex nitscheStripVertex(int i, int j, int k)
{
    constexpr int nodes_x = 6;
    constexpr int nodes_y = 2;
    return static_cast<FE::GlobalIndex>(
        i + nodes_x * (j + nodes_y * k));
}

[[nodiscard]] TetraStripArrays makeFiveCubeTetraStripArrays(
    FE::Real mesh_scale)
{
    if (!(mesh_scale > FE::Real{0.0}) ||
        !std::isfinite(mesh_scale)) {
        throw std::invalid_argument(
            "Nitsche tetra strip requires a finite positive mesh scale");
    }

    TetraStripArrays arrays;
    arrays.coordinates.reserve(24u * 3u);
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 6; ++i) {
                arrays.coordinates.push_back(static_cast<real_t>(
                    mesh_scale * static_cast<FE::Real>(i)));
                arrays.coordinates.push_back(static_cast<real_t>(
                    mesh_scale * static_cast<FE::Real>(j)));
                arrays.coordinates.push_back(static_cast<real_t>(
                    mesh_scale * static_cast<FE::Real>(k)));
            }
        }
    }

    constexpr std::array<std::array<std::size_t, 4>, 6> tetrahedra = {{
        {{0, 1, 2, 6}},
        {{0, 2, 3, 6}},
        {{0, 3, 7, 6}},
        {{0, 7, 4, 6}},
        {{0, 4, 5, 6}},
        {{0, 5, 1, 6}},
    }};
    arrays.cell_offsets = {0};
    arrays.cell_offsets.reserve(31u);
    arrays.cell_vertices.reserve(30u * 4u);
    for (int i = 0; i < 5; ++i) {
        const std::array<FE::GlobalIndex, 8> nodes = {
            nitscheStripVertex(i, 0, 0),
            nitscheStripVertex(i + 1, 0, 0),
            nitscheStripVertex(i + 1, 1, 0),
            nitscheStripVertex(i, 1, 0),
            nitscheStripVertex(i, 0, 1),
            nitscheStripVertex(i + 1, 0, 1),
            nitscheStripVertex(i + 1, 1, 1),
            nitscheStripVertex(i, 1, 1),
        };
        for (const auto& tetra : tetrahedra) {
            for (const auto local : tetra) {
                arrays.cell_vertices.push_back(
                    static_cast<index_t>(nodes[local]));
            }
            arrays.cell_offsets.push_back(
                static_cast<offset_t>(arrays.cell_vertices.size()));
        }
    }
    arrays.cell_shapes.assign(
        30u, CellShape{CellFamily::Tetra, 4, 1});
    return arrays;
}

[[nodiscard]] std::shared_ptr<Mesh> makeNitscheEnergyTetraStripMesh(
    const PlaneCutPosition& cut,
    FE::Real mesh_scale,
    int wall_marker,
    int anchor_marker)
{
    const auto arrays =
        makeFiveCubeTetraStripArrays(mesh_scale);
    auto base = std::make_shared<MeshBase>();
    base->build_from_arrays(
        /*spatial_dim=*/3,
        arrays.coordinates,
        arrays.cell_offsets,
        arrays.cell_vertices,
        arrays.cell_shapes);
    base->finalize();
    base->register_label(
        "nitsche_energy_wall",
        static_cast<label_t>(wall_marker));
    base->register_label(
        "nitsche_energy_anchor",
        static_cast<label_t>(anchor_marker));

    std::size_t marked_face_count = 0u;
    std::size_t anchor_face_count = 0u;
    const auto on_plane = [](FE::Real value, FE::Real target) {
        return std::abs(value - target) <=
               FE::Real{32.0} *
                   std::numeric_limits<FE::Real>::epsilon() *
                   std::max(FE::Real{1.0}, std::abs(target));
    };
    for (index_t face = 0;
         face < static_cast<index_t>(base->n_faces());
         ++face) {
        const auto vertices = base->face_vertices(face);
        if (vertices.size() != 3u) {
            continue;
        }
        bool on_wall = true;
        bool on_anchor = true;
        std::size_t right_edge_vertex_count = 0u;
        for (const auto vertex : vertices) {
            const auto point = base->get_vertex_coords(vertex);
            on_wall =
                on_wall && on_plane(point[2], FE::Real{0.0}) &&
                point[0] >= FE::Real{3.0} * mesh_scale -
                                FE::Real{1.0e-13} &&
                point[0] <= FE::Real{4.0} * mesh_scale +
                                FE::Real{1.0e-13};
            right_edge_vertex_count += static_cast<std::size_t>(
                on_plane(
                    point[0], FE::Real{4.0} * mesh_scale));
            on_anchor =
                on_anchor && on_plane(point[0], FE::Real{0.0});
        }
        // Qualify one native triangular exterior face.  Its two vertices on
        // the right edge keep the requested 1e-8 wet fraction away from an
        // interface fragment whose area is below the authoritative geometry
        // tolerance; the sweep is about wet-boundary measure, not an
        // independently collapsed sub-tolerance interface sliver.
        if (on_wall && right_edge_vertex_count == 2u) {
            base->set_boundary_label(
                face, static_cast<label_t>(wall_marker));
            ++marked_face_count;
        }
        if (on_anchor) {
            base->set_boundary_label(
                face, static_cast<label_t>(anchor_marker));
            ++anchor_face_count;
        }
    }
    if (marked_face_count != 1u) {
        throw std::runtime_error(
            "Nitsche energy tetra strip requires exactly one marked triangular wall patch");
    }
    if (anchor_face_count == 0u) {
        throw std::runtime_error(
            "Nitsche energy tetra strip has no fixed strong anchor boundary");
    }

    const auto phi_handle = MeshFields::attach_field(
        *base,
        EntityKind::Vertex,
        "phi",
        FieldScalarType::Float64,
        1);
    auto* phi = MeshFields::field_data_as<real_t>(
        *base, phi_handle);
    if (phi == nullptr) {
        throw std::runtime_error(
            "failed to allocate Nitsche energy level-set field");
    }
    for (index_t vertex = 0;
         vertex < static_cast<index_t>(base->n_vertices());
         ++vertex) {
        const auto point = base->get_vertex_coords(vertex);
        phi[static_cast<std::size_t>(vertex)] =
            static_cast<real_t>(cut.value({{
                static_cast<FE::Real>(point[0]),
                static_cast<FE::Real>(point[1]),
                static_cast<FE::Real>(point[2]),
            }}));
    }
#if defined(MESH_HAS_MPI)
    return create_mesh(std::move(base), MeshComm(MPI_COMM_SELF));
#else
    return create_mesh(std::move(base));
#endif
}

#if defined(FE_HAS_MPI) && defined(MESH_HAS_MPI)

[[nodiscard]] TetraStripArrays makeStructuredTetraArrays(
    int cells_per_axis)
{
    if (cells_per_axis < 2) {
        throw std::invalid_argument(
            "distributed structured stability mesh requires at least two cells per axis");
    }
    TetraStripArrays arrays;
    const int nodes_per_axis = cells_per_axis + 1;
    const auto node_count =
        static_cast<std::size_t>(nodes_per_axis) *
        static_cast<std::size_t>(nodes_per_axis) *
        static_cast<std::size_t>(nodes_per_axis);
    const auto cube_count =
        static_cast<std::size_t>(cells_per_axis) *
        static_cast<std::size_t>(cells_per_axis) *
        static_cast<std::size_t>(cells_per_axis);
    const auto h =
        FE::Real{2.0} / static_cast<FE::Real>(cells_per_axis);
    arrays.coordinates.reserve(3u * node_count);
    for (int k = 0; k < nodes_per_axis; ++k) {
        for (int j = 0; j < nodes_per_axis; ++j) {
            for (int i = 0; i < nodes_per_axis; ++i) {
                arrays.coordinates.push_back(
                    static_cast<real_t>(h * static_cast<FE::Real>(i)));
                arrays.coordinates.push_back(
                    static_cast<real_t>(h * static_cast<FE::Real>(j)));
                arrays.coordinates.push_back(
                    static_cast<real_t>(h * static_cast<FE::Real>(k)));
            }
        }
    }
    const auto vertex = [&](int i, int j, int k) {
        return static_cast<FE::GlobalIndex>(
            i + nodes_per_axis * (j + nodes_per_axis * k));
    };
    constexpr std::array<std::array<std::size_t, 4>, 6> tetrahedra = {{
        {{0, 1, 2, 6}},
        {{0, 2, 3, 6}},
        {{0, 3, 7, 6}},
        {{0, 7, 4, 6}},
        {{0, 4, 5, 6}},
        {{0, 5, 1, 6}},
    }};
    arrays.cell_offsets = {0};
    arrays.cell_offsets.reserve(6u * cube_count + 1u);
    arrays.cell_vertices.reserve(24u * cube_count);
    for (int k = 0; k < cells_per_axis; ++k) {
        for (int j = 0; j < cells_per_axis; ++j) {
            for (int i = 0; i < cells_per_axis; ++i) {
                const std::array<FE::GlobalIndex, 8> nodes = {
                    vertex(i, j, k),
                    vertex(i + 1, j, k),
                    vertex(i + 1, j + 1, k),
                    vertex(i, j + 1, k),
                    vertex(i, j, k + 1),
                    vertex(i + 1, j, k + 1),
                    vertex(i + 1, j + 1, k + 1),
                    vertex(i, j + 1, k + 1),
                };
                for (const auto& tetra : tetrahedra) {
                    for (const auto local : tetra) {
                        arrays.cell_vertices.push_back(
                            static_cast<index_t>(nodes[local]));
                    }
                    arrays.cell_offsets.push_back(
                        static_cast<offset_t>(
                            arrays.cell_vertices.size()));
                }
            }
        }
    }
    arrays.cell_shapes.assign(
        6u * cube_count,
        CellShape{CellFamily::Tetra, 4, 1});
    return arrays;
}

[[nodiscard]] std::shared_ptr<Mesh> makeDistributedTetraStripMesh(
    const PlaneCutPosition& cut,
    MPI_Comm comm,
    std::string_view partition_method,
    int ghost_layers = 18)
{
    const auto arrays = makeThreeCubeTetraStripArrays();
    auto mesh = std::make_shared<Mesh>(MeshComm(comm));
    mesh->build_from_arrays_global_and_partition(
        /*spatial_dim=*/3,
        arrays.coordinates,
        arrays.cell_offsets,
        arrays.cell_vertices,
        arrays.cell_shapes,
        PartitionHint::Cells,
        // Small-cut aggregation follows the cut band to a full-active root.
        // Keep the complete three-cube qualification mesh in overlap so the
        // finite partition comparison exercises row ownership without making
        // its aggregate topology depend on a deliberately truncated stencil.
        ghost_layers,
        {{"partition_method", std::string(partition_method)}});

    auto& local_mesh = mesh->base();
    const auto phi_handle = MeshFields::attach_field(
        local_mesh,
        EntityKind::Vertex,
        "phi",
        FieldScalarType::Float64,
        1);
    auto* phi = MeshFields::field_data_as<real_t>(local_mesh, phi_handle);
    if (phi == nullptr) {
        throw std::runtime_error(
            "failed to allocate distributed stability level-set field");
    }
    for (index_t vertex = 0;
         vertex < static_cast<index_t>(local_mesh.n_vertices());
         ++vertex) {
        const auto x = local_mesh.get_vertex_coords(vertex);
        phi[static_cast<std::size_t>(vertex)] = static_cast<real_t>(
            cut.value({{static_cast<FE::Real>(x[0]),
                        static_cast<FE::Real>(x[1]),
                        static_cast<FE::Real>(x[2])}}));
    }
    return mesh;
}

[[nodiscard]] std::shared_ptr<Mesh> makeDistributedStructuredTetraMesh(
    const PlaneCutPosition& cut,
    MPI_Comm comm,
    std::string_view partition_method,
    int cells_per_axis)
{
    const auto arrays = makeStructuredTetraArrays(cells_per_axis);
    auto mesh = std::make_shared<Mesh>(MeshComm(comm));
    mesh->build_from_arrays_global_and_partition(
        /*spatial_dim=*/3,
        arrays.coordinates,
        arrays.cell_offsets,
        arrays.cell_vertices,
        arrays.cell_shapes,
        PartitionHint::Cells,
        static_cast<int>(arrays.cell_shapes.size()),
        {{"partition_method", std::string(partition_method)}});

    auto& local_mesh = mesh->base();
    const auto phi_handle = MeshFields::attach_field(
        local_mesh,
        EntityKind::Vertex,
        "phi",
        FieldScalarType::Float64,
        1);
    auto* phi = MeshFields::field_data_as<real_t>(
        local_mesh, phi_handle);
    if (phi == nullptr) {
        throw std::runtime_error(
            "failed to allocate distributed structured level-set field");
    }
    for (index_t vertex = 0;
         vertex < static_cast<index_t>(local_mesh.n_vertices());
         ++vertex) {
        const auto x = local_mesh.get_vertex_coords(vertex);
        phi[static_cast<std::size_t>(vertex)] =
            static_cast<real_t>(cut.value({{
                static_cast<FE::Real>(x[0]),
                static_cast<FE::Real>(x[1]),
                static_cast<FE::Real>(x[2]),
            }}));
    }
    return mesh;
}

[[nodiscard]] std::shared_ptr<Mesh> makeDistributedWetBlockQuadStrip(
    std::span<const FE::Real> x_coordinates,
    std::span<const FE::Real> level_set_by_plane,
    MPI_Comm comm,
    std::string_view partition_method)
{
    const auto arrays = makeWetBlockQuadStripArrays(
        x_coordinates, level_set_by_plane);
    auto mesh = std::make_shared<Mesh>(MeshComm(comm));
    mesh->build_from_arrays_global_and_partition(
        /*spatial_dim=*/2,
        arrays.coordinates,
        arrays.cell_offsets,
        arrays.cell_vertices,
        arrays.cell_shapes,
        PartitionHint::Cells,
        static_cast<int>(arrays.cell_shapes.size()),
        {{"partition_method", std::string(partition_method)}});
    if (mesh->global_n_cells() != arrays.cell_shapes.size() ||
        mesh->n_owned_cells() == 0u ||
        mesh->n_owned_cells() >= mesh->global_n_cells()) {
        throw std::runtime_error(
            "distributed wet-block strip is not genuinely partitioned");
    }

    const auto& gids = mesh->base().vertex_gids();
    std::vector<real_t> local_level_set(mesh->base().n_vertices(), 0.0);
    for (index_t vertex = 0;
         vertex < static_cast<index_t>(mesh->base().n_vertices());
         ++vertex) {
        const auto gid = gids.at(static_cast<std::size_t>(vertex));
        if (gid < 0 ||
            static_cast<std::size_t>(gid) >= arrays.level_set.size()) {
            throw std::runtime_error(
                "distributed wet-block vertex GID cannot resolve level-set data");
        }
        local_level_set[static_cast<std::size_t>(vertex)] =
            arrays.level_set[static_cast<std::size_t>(gid)];
    }
    attachWetBlockLevelSet(mesh->base(), local_level_set);
    return mesh;
}

#endif

[[nodiscard]] std::vector<FE::MeshIndex> retainedCells(
    const FE::assembly::CutIntegrationContext& context,
    int marker,
    bool cut_only)
{
    std::vector<FE::MeshIndex> cells;
    constexpr FE::Real full_fraction_tolerance = FE::Real{1.0e-12};
    for (const auto* metadata :
         context.generatedVolumeMetadataForMarkerAndSide(
             marker, FE::geometry::CutIntegrationSide::Negative)) {
        if (metadata == nullptr || metadata->parent_entity < 0 ||
            !std::isfinite(metadata->volume_fraction) ||
            metadata->volume_fraction <= FE::Real{0.0}) {
            continue;
        }
        if (cut_only &&
            metadata->volume_fraction >=
                FE::Real{1.0} - full_fraction_tolerance) {
            continue;
        }
        cells.push_back(metadata->parent_entity);
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    return cells;
}

/** Mirror ApplicationDriver::addGeneratedCutAdjacentFacetSet using its public
 * FE primitives so the diagnostic operator sees the production facet scope. */
[[nodiscard]] FE::assembly::CutFacetSetHandle addProductionFacetSet(
    FE::assembly::CutIntegrationContext& context,
    const FE::interfaces::LevelSetInterfaceDomain& domain,
    const FE::assembly::IMeshAccess& mesh)
{
    std::vector<FE::systems::CutInteriorFacetAdjacency> adjacency;
    mesh.forEachInteriorFace(
        [&](FE::GlobalIndex face,
            FE::GlobalIndex first,
            FE::GlobalIndex second) {
            adjacency.push_back(FE::systems::CutInteriorFacetAdjacency{
                .facet = static_cast<FE::MeshIndex>(face),
                .first_cell = static_cast<FE::MeshIndex>(first),
                .second_cell = static_cast<FE::MeshIndex>(second),
            });
        });

    auto cut_cells = retainedCells(context, domain.marker(), true);
    if (cut_cells.empty()) {
        cut_cells = domain.cutCells();
    }
    auto facets = FE::systems::identifyCutAdjacentInteriorFacets(
        cut_cells, adjacency);

    const auto active_cells = retainedCells(context, domain.marker(), false);
    const auto is_active = [&active_cells](FE::MeshIndex cell) {
        return std::binary_search(active_cells.begin(), active_cells.end(), cell);
    };
    facets.erase(
        std::remove_if(facets.begin(), facets.end(), [&](const auto& facet) {
            return !is_active(facet.first_cell) ||
                   !is_active(facet.second_cell);
        }),
        facets.end());

    const auto generated = FE::systems::makeCutAdjacentFacetSetHandle(
        domain.marker(), "generated-cut-adjacent-facets", facets);
    FE::assembly::CutFacetSetHandle handle;
    handle.marker = generated.marker;
    handle.name = generated.name;
    handle.facets = generated.facets;
    handle.stable_id = generated.stable_id;
    handle.facet_metadata.reserve(generated.facet_metadata.size());
    for (const auto& facet : generated.facet_metadata) {
        handle.facet_metadata.push_back(
            FE::assembly::CutFacetSetFacetMetadata{
                .facet = facet.facet,
                .first_cell = facet.first_cell,
                .second_cell = facet.second_cell,
                .stabilization_scale = facet.stabilization_scale,
                .stable_id = facet.stable_id,
            });
    }
    context.bindFacetStabilizationScalesForMarkerAndSide(
        handle,
        domain.marker(),
        FE::geometry::CutIntegrationSide::Negative);
    return context.addFacetSetHandle(std::move(handle));
}

void setScalarVertexField(std::vector<FE::Real>& values,
                          const FE::systems::FESystem& system,
                          FE::FieldId field,
                          const PlaneCutPosition& cut)
{
    const auto* entity_map =
        system.fieldDofHandler(field).getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("fixed-sweep scalar field has no entity map");
    }
    const auto offset = system.fieldDofOffset(field);
    for (FE::GlobalIndex vertex = 0;
         vertex < system.meshAccess().numVertices();
         ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        if (dofs.size() != 1u) {
            throw std::runtime_error(
                "fixed-sweep scalar field is not one-dof-per-vertex");
        }
        const auto global_dof = offset + dofs.front();
        values.at(static_cast<std::size_t>(global_dof)) =
            cut.value(system.meshAccess().getNodeCoordinates(vertex));
    }
}

void setConstantScalarVertexField(std::vector<FE::Real>& values,
                                  const FE::systems::FESystem& system,
                                  FE::FieldId field,
                                  FE::Real value)
{
    const auto* entity_map =
        system.fieldDofHandler(field).getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error(
            "manufactured scalar field has no entity map");
    }
    const auto offset = system.fieldDofOffset(field);
    for (FE::GlobalIndex vertex = 0;
         vertex < system.meshAccess().numVertices();
         ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        if (dofs.size() != 1u) {
            throw std::runtime_error(
                "manufactured scalar field is not one-dof-per-vertex");
        }
        values.at(static_cast<std::size_t>(offset + dofs.front())) = value;
    }
}

void setMeshVertexField(Mesh& mesh, const PlaneCutPosition& cut)
{
    auto& local_mesh = mesh.base();
    const auto handle = MeshFields::get_field_handle(
        local_mesh, EntityKind::Vertex, "phi");
    auto* phi = MeshFields::field_data_as<real_t>(local_mesh, handle);
    if (phi == nullptr) {
        throw std::runtime_error(
            "fixed-sweep mesh has no writable vertex level-set field");
    }
    for (index_t vertex = 0;
         vertex < static_cast<index_t>(local_mesh.n_vertices());
         ++vertex) {
        const auto x = local_mesh.get_vertex_coords(vertex);
        phi[static_cast<std::size_t>(vertex)] = static_cast<real_t>(
            cut.value({{static_cast<FE::Real>(x[0]),
                        static_cast<FE::Real>(x[1]),
                        static_cast<FE::Real>(x[2])}}));
    }
}

struct TargetFractionGeometrySample {
    FE::Real target_fraction{0.0};
    FE::Real generated_fraction{0.0};
    FE::Real expected_retained_volume{0.0};
    FE::Real generated_retained_volume{0.0};
    std::size_t cut_cells{0u};
    std::size_t backend_fallback_cells{0u};
};

[[nodiscard]] std::array<FE::Real, 3> crossProduct(
    const std::array<FE::Real, 3>& lhs,
    const std::array<FE::Real, 3>& rhs)
{
    return {{
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    }};
}

[[nodiscard]] FE::Real vectorNorm(
    const std::array<FE::Real, 3>& value)
{
    return std::sqrt(
        value[0] * value[0] +
        value[1] * value[1] +
        value[2] * value[2]);
}

[[nodiscard]] std::array<FE::Real, 3> normalizedVector(
    const std::array<FE::Real, 3>& value)
{
    const auto norm = vectorNorm(value);
    if (!(norm > FE::Real{0.0}) || !std::isfinite(norm)) {
        throw std::invalid_argument(
            "target-fraction orientation must have finite positive length");
    }
    return {{
        value[0] / norm,
        value[1] / norm,
        value[2] / norm,
    }};
}

[[nodiscard]] TargetFractionGeometrySample runTargetFractionGeometrySample(
    FE::Real target_fraction,
    const std::array<FE::Real, 3>& requested_normal,
    FE::Real h)
{
    if (!(target_fraction > FE::Real{0.0}) ||
        !(target_fraction < FE::Real{1.0}) ||
        !(h > FE::Real{0.0}) ||
        !std::isfinite(target_fraction) ||
        !std::isfinite(h)) {
        throw std::invalid_argument(
            "target-fraction geometry requires finite interior fraction and h");
    }

    const auto normal = normalizedVector(requested_normal);
    const std::array<FE::Real, 3> seed =
        std::abs(normal[2]) < FE::Real{0.9}
            ? std::array<FE::Real, 3>{{0.0, 0.0, 1.0}}
            : std::array<FE::Real, 3>{{0.0, 1.0, 0.0}};
    const auto first_tangent =
        normalizedVector(crossProduct(normal, seed));
    const auto second_tangent =
        normalizedVector(crossProduct(normal, first_tangent));
    const auto scaled_sum =
        [&](const std::array<FE::Real, 3>& first,
            const std::array<FE::Real, 3>& second) {
            return std::array<FE::Real, 3>{{
                h * (first[0] + second[0]),
                h * (first[1] + second[1]),
                h * (first[2] + second[2]),
            }};
        };

    const std::array<std::array<FE::Real, 3>, 4> vertices = {{
        {{0.0, 0.0, 0.0}},
        {{h * normal[0], h * normal[1], h * normal[2]}},
        scaled_sum(normal, first_tangent),
        scaled_sum(normal, second_tangent),
    }};
    std::vector<real_t> coordinates;
    coordinates.reserve(12u);
    for (const auto& vertex : vertices) {
        coordinates.push_back(static_cast<real_t>(vertex[0]));
        coordinates.push_back(static_cast<real_t>(vertex[1]));
        coordinates.push_back(static_cast<real_t>(vertex[2]));
    }

    auto base = std::make_shared<MeshBase>();
    base->build_from_arrays(
        /*spatial_dim=*/3,
        coordinates,
        std::vector<offset_t>{0, 4},
        std::vector<index_t>{0, 1, 2, 3},
        std::vector<CellShape>{
            CellShape{CellFamily::Tetra, 4, 1}});
    base->finalize();
    auto mesh = create_mesh(std::move(base));
    auto scalar_space = FE::spaces::SpaceFactory::create_h1(
        FE::ElementType::Tetra4, /*order=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::Unknown,
    });
    system.setup({});

    const auto edge_fraction = std::cbrt(target_fraction);
    const PlaneCutPosition cut{
        "target_fraction",
        normal,
        edge_fraction * h,
    };
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        FE::Real{0.0});
    setScalarVertexField(solution, system, phi, cut);

    FE::level_set::LevelSetGeneratedInterfaceOptions options;
    options.level_set_field_name = "phi";
    options.domain_id = "wp7_exact_target_fraction";
    options.requested_interface_marker = 27017;
    options.tolerance = FE::Real{1.0e-14};
    options.quadrature_order = 2;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;
    FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto generated = lifecycle.build(system, options, solution);
    if (!generated.success) {
        throw std::runtime_error(generated.diagnostic);
    }

    // LevelSetInterfaceDomain summary measures are stored in parent reference
    // coordinates.  Physical assembly applies the cell mapping later, so the
    // single reference tetra has measure 1/6 at every physical h.
    constexpr FE::Real full_volume =
        FE::Real{1.0} / FE::Real{6.0};
    TargetFractionGeometrySample sample;
    sample.target_fraction = target_fraction;
    sample.generated_retained_volume =
        generated.summary.negative_volume_measure;
    sample.expected_retained_volume = target_fraction * full_volume;
    sample.generated_fraction =
        sample.generated_retained_volume / full_volume;
    sample.cut_cells = generated.domain.cutCells().size();
    sample.backend_fallback_cells =
        generated.implicit_cut_fallback_cell_count;
    return sample;
}

[[nodiscard]] FE::Real tetrahedronVolume(
    const std::array<FE::Real, 3>& first,
    const std::array<FE::Real, 3>& second,
    const std::array<FE::Real, 3>& third,
    const std::array<FE::Real, 3>& fourth)
{
    const std::array<FE::Real, 3> a = {{
        second[0] - first[0],
        second[1] - first[1],
        second[2] - first[2],
    }};
    const std::array<FE::Real, 3> b = {{
        third[0] - first[0],
        third[1] - first[1],
        third[2] - first[2],
    }};
    const std::array<FE::Real, 3> c = {{
        fourth[0] - first[0],
        fourth[1] - first[1],
        fourth[2] - first[2],
    }};
    return std::abs(
               a[0] * (b[1] * c[2] - b[2] * c[1]) -
               a[1] * (b[0] * c[2] - b[2] * c[0]) +
               a[2] * (b[0] * c[1] - b[1] * c[0])) /
           FE::Real{6.0};
}

[[nodiscard]] FE::Real negativeReferenceTetraFraction(
    const std::array<FE::Real, 4>& values)
{
    constexpr std::array<std::array<FE::Real, 3>, 4> vertices = {{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}},
    }};
    std::array<std::size_t, 4> negative{};
    std::array<std::size_t, 4> positive{};
    std::size_t negative_count = 0u;
    std::size_t positive_count = 0u;
    for (std::size_t vertex = 0u; vertex < values.size(); ++vertex) {
        if (values[vertex] < FE::Real{0.0}) {
            negative[negative_count++] = vertex;
        } else {
            positive[positive_count++] = vertex;
        }
    }
    if (negative_count == 0u) {
        return FE::Real{0.0};
    }
    if (negative_count == 4u) {
        return FE::Real{1.0};
    }

    const auto intersection =
        [&](std::size_t first, std::size_t second) {
            const auto denominator = values[second] - values[first];
            if (!(std::abs(denominator) > FE::Real{0.0})) {
                throw std::runtime_error(
                    "target-cut edge has no signed-value separation");
            }
            const auto t = -values[first] / denominator;
            return std::array<FE::Real, 3>{{
                vertices[first][0] +
                    t * (vertices[second][0] - vertices[first][0]),
                vertices[first][1] +
                    t * (vertices[second][1] - vertices[first][1]),
                vertices[first][2] +
                    t * (vertices[second][2] - vertices[first][2]),
            }};
        };
    constexpr FE::Real full_volume = FE::Real{1.0} / FE::Real{6.0};
    if (negative_count == 1u) {
        const auto n = negative[0];
        const auto first = intersection(n, positive[0]);
        const auto second = intersection(n, positive[1]);
        const auto third = intersection(n, positive[2]);
        return tetrahedronVolume(
                   vertices[n], first, second, third) /
               full_volume;
    }
    if (negative_count == 3u) {
        const auto p = positive[0];
        const auto first = intersection(p, negative[0]);
        const auto second = intersection(p, negative[1]);
        const auto third = intersection(p, negative[2]);
        return FE::Real{1.0} -
               tetrahedronVolume(
                   vertices[p], first, second, third) /
                   full_volume;
    }

    const auto n0 = negative[0];
    const auto n1 = negative[1];
    const auto p0 = positive[0];
    const auto p1 = positive[1];
    const auto n0p0 = intersection(n0, p0);
    const auto n0p1 = intersection(n0, p1);
    const auto n1p0 = intersection(n1, p0);
    const auto n1p1 = intersection(n1, p1);
    const auto volume =
        tetrahedronVolume(
            vertices[n0], vertices[n1], n0p0, n0p1) +
        tetrahedronVolume(
            vertices[n1], n0p0, n0p1, n1p1) +
        tetrahedronVolume(
            vertices[n1], n0p0, n1p0, n1p1);
    return volume / full_volume;
}

struct TargetStructuredCut {
    PlaneCutPosition cut{};
    FE::MeshIndex designated_parent_cell{-1};
};

[[nodiscard]] TargetStructuredCut makeTargetStructuredCut(
    FE::Real target_fraction,
    const std::array<FE::Real, 3>& normal,
    int cells_per_axis,
    std::string label)
{
    if (!(target_fraction > FE::Real{0.0}) ||
        !(target_fraction < FE::Real{1.0}) ||
        cells_per_axis < 2) {
        throw std::invalid_argument(
            "structured target cut requires interior fraction and at least two cells per axis");
    }
    const auto h =
        FE::Real{2.0} / static_cast<FE::Real>(cells_per_axis);
    const auto base =
        h * static_cast<FE::Real>(cells_per_axis - 1);
    const std::array<std::array<FE::Real, 3>, 4> vertices = {{
        {{base, base, base}},
        {{base + h, base, base}},
        {{base + h, base + h, base}},
        {{base + h, base + h, base + h}},
    }};
    std::array<FE::Real, 4> projections{};
    for (std::size_t vertex = 0u; vertex < vertices.size(); ++vertex) {
        projections[vertex] =
            normal[0] * vertices[vertex][0] +
            normal[1] * vertices[vertex][1] +
            normal[2] * vertices[vertex][2];
    }
    auto lower =
        *std::min_element(projections.begin(), projections.end());
    auto upper =
        *std::max_element(projections.begin(), projections.end());
    for (int iteration = 0; iteration < 100; ++iteration) {
        const auto offset = FE::Real{0.5} * (lower + upper);
        std::array<FE::Real, 4> values{};
        for (std::size_t vertex = 0u;
             vertex < projections.size();
             ++vertex) {
            values[vertex] = projections[vertex] - offset;
        }
        if (negativeReferenceTetraFraction(values) < target_fraction) {
            lower = offset;
        } else {
            upper = offset;
        }
    }
    const auto cube_count =
        static_cast<std::size_t>(cells_per_axis) *
        static_cast<std::size_t>(cells_per_axis) *
        static_cast<std::size_t>(cells_per_axis);
    return TargetStructuredCut{
        .cut = PlaneCutPosition{
            std::move(label),
            normal,
            FE::Real{0.5} * (lower + upper),
        },
        .designated_parent_cell =
            static_cast<FE::MeshIndex>(6u * (cube_count - 1u)),
    };
}

[[nodiscard]] FieldConstraintCounts countFieldConstraints(
    const FE::systems::FESystem& system,
    FE::FieldId field)
{
    FieldConstraintCounts counts;
    const auto& constraints = system.constraints();
    const auto offset = system.fieldDofOffset(field);
    const auto count = system.fieldDofHandler(field).getNumDofs();
    for (FE::GlobalIndex local = 0; local < count; ++local) {
        const auto line = constraints.getConstraint(offset + local);
        if (!line.has_value()) {
            continue;
        }
        if (line->entries.empty()) {
            ++counts.homogeneous_pins;
        } else {
            ++counts.master_bearing;
        }
    }
    return counts;
}

[[nodiscard]] AggregationConstraintMetrics aggregationConstraintMetrics(
    const FE::systems::FESystem& system,
    FE::FieldId field,
    FE::Real mesh_spacing)
{
    if (!(mesh_spacing > FE::Real{0.0}) || !std::isfinite(mesh_spacing)) {
        throw std::invalid_argument(
            "aggregation metric requires a finite positive mesh spacing");
    }
    const auto* entity_map =
        system.fieldDofHandler(field).getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error(
            "aggregation metric field has no entity DOF map");
    }

    const auto offset = system.fieldDofOffset(field);
    const auto field_dofs = system.fieldDofHandler(field).getNumDofs();
    std::vector<std::array<FE::Real, 3>> dof_coordinates(
        static_cast<std::size_t>(field_dofs));
    std::vector<bool> has_coordinate(
        static_cast<std::size_t>(field_dofs), false);
    for (FE::GlobalIndex vertex = 0;
         vertex < system.meshAccess().numVertices();
         ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        if (dofs.size() != 1u || dofs.front() < 0 ||
            dofs.front() >= field_dofs) {
            throw std::runtime_error(
                "aggregation metric requires scalar P1 vertex DOFs");
        }
        const auto local = static_cast<std::size_t>(dofs.front());
        dof_coordinates[local] =
            system.meshAccess().getNodeCoordinates(vertex);
        has_coordinate[local] = true;
    }

    AggregationConstraintMetrics metrics;
    const auto& constraints = system.constraints();
    for (FE::GlobalIndex local = 0; local < field_dofs; ++local) {
        const auto line = constraints.getConstraint(offset + local);
        if (!line.has_value() || line->entries.empty()) {
            continue;
        }
        ++metrics.master_bearing_lines;
        metrics.master_entries += line->entries.size();
        metrics.maximum_masters_per_line = std::max(
            metrics.maximum_masters_per_line, line->entries.size());
        metrics.maximum_inhomogeneity = std::max(
            metrics.maximum_inhomogeneity,
            static_cast<FE::Real>(std::abs(line->inhomogeneity)));

        long double weight_sum = 0.0L;
        long double weight_l1 = 0.0L;
        if (!has_coordinate[static_cast<std::size_t>(local)]) {
            throw std::runtime_error(
                "aggregation slave has no vertex coordinate");
        }
        const auto& slave = dof_coordinates[static_cast<std::size_t>(local)];
        for (const auto& entry : line->entries) {
            const auto master_local = entry.master_dof - offset;
            if (master_local < 0 || master_local >= field_dofs ||
                !has_coordinate[static_cast<std::size_t>(master_local)]) {
                throw std::runtime_error(
                    "aggregation master is outside the scalar P1 field");
            }
            const auto weight = static_cast<long double>(entry.weight);
            weight_sum += weight;
            weight_l1 += std::abs(weight);
            metrics.maximum_absolute_weight = std::max(
                metrics.maximum_absolute_weight,
                static_cast<FE::Real>(std::abs(weight)));

            const auto& master =
                dof_coordinates[static_cast<std::size_t>(master_local)];
            FE::Real distance_squared = 0.0;
            for (std::size_t d = 0; d < 3u; ++d) {
                const auto delta = slave[d] - master[d];
                distance_squared += delta * delta;
            }
            metrics.maximum_slave_master_distance_over_h = std::max(
                metrics.maximum_slave_master_distance_over_h,
                std::sqrt(distance_squared) / mesh_spacing);
        }
        metrics.maximum_partition_of_unity_error = std::max(
            metrics.maximum_partition_of_unity_error,
            static_cast<FE::Real>(std::abs(weight_sum - 1.0L)));
        metrics.maximum_weight_l1 = std::max(
            metrics.maximum_weight_l1,
            static_cast<FE::Real>(weight_l1));
    }
    return metrics;
}

[[nodiscard]] std::vector<FE::GlobalIndex> freeFieldDofs(
    const FE::systems::FESystem& system,
    FE::FieldId field)
{
    std::vector<FE::GlobalIndex> dofs;
    const auto& constraints = system.constraints();
    const auto offset = system.fieldDofOffset(field);
    const auto count = system.fieldDofHandler(field).getNumDofs();
    dofs.reserve(static_cast<std::size_t>(count));
    for (FE::GlobalIndex local = 0; local < count; ++local) {
        const auto global = offset + local;
        if (!constraints.isConstrained(global)) {
            dofs.push_back(global);
        }
    }
    return dofs;
}

struct CanonicalP1Dof {
    gid_t vertex_gid{0};
    std::size_t component{0u};
    FE::GlobalIndex global_dof{FE::INVALID_GLOBAL_INDEX};
};

/** Return all P1 DOFs in physical vertex-GID/component order. */
[[nodiscard]] std::vector<CanonicalP1Dof> canonicalP1Dofs(
    const Mesh& mesh,
    const FE::systems::FESystem& system,
    FE::FieldId field,
    std::size_t components)
{
    const auto* entity_map =
        system.fieldDofHandler(field).getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error(
            "stability P1 field has no entity DOF map");
    }

    const auto& base = mesh.base();
    const auto& vertex_gids = base.vertex_gids();
    if (vertex_gids.size() != base.n_vertices()) {
        throw std::runtime_error(
            "stability mesh has incomplete vertex GIDs");
    }

    const auto offset = system.fieldDofOffset(field);
    std::vector<CanonicalP1Dof> canonical;
    canonical.reserve(vertex_gids.size() * components);
    for (index_t vertex = 0;
         vertex < static_cast<index_t>(base.n_vertices());
         ++vertex) {
        const auto vertex_dofs = entity_map->getVertexDofs(
            static_cast<FE::GlobalIndex>(vertex));
        if (vertex_dofs.size() != components) {
            throw std::runtime_error(
                "stability field is not the expected P1 layout");
        }
        for (std::size_t component = 0u;
             component < components;
             ++component) {
            canonical.push_back(CanonicalP1Dof{
                vertex_gids[static_cast<std::size_t>(vertex)],
                component,
                offset + vertex_dofs[component]});
        }
    }

    std::sort(
        canonical.begin(),
        canonical.end(),
        [](const CanonicalP1Dof& lhs, const CanonicalP1Dof& rhs) {
            if (lhs.vertex_gid != rhs.vertex_gid) {
                return lhs.vertex_gid < rhs.vertex_gid;
            }
            return lhs.component < rhs.component;
        });
    for (std::size_t index = 1u; index < canonical.size(); ++index) {
        if (canonical[index - 1u].vertex_gid == canonical[index].vertex_gid &&
            canonical[index - 1u].component == canonical[index].component) {
            throw std::runtime_error(
                "stability canonical P1 DOF is duplicated");
        }
    }
    return canonical;
}

[[nodiscard]] std::vector<FE::GlobalIndex> canonicalFreeP1Dofs(
    const Mesh& mesh,
    const FE::systems::FESystem& system,
    FE::FieldId field,
    std::size_t components)
{
    const auto canonical =
        canonicalP1Dofs(mesh, system, field, components);
    std::vector<FE::GlobalIndex> dofs;
    dofs.reserve(canonical.size());
    for (const auto& entry : canonical) {
        if (!system.constraints().isConstrained(entry.global_dof)) {
            dofs.push_back(entry.global_dof);
        }
    }
    return dofs;
}

struct PressureAnchorState {
    bool natural_traction_anchor{false};
    bool no_gauge_enforcement{false};
};

[[nodiscard]] PressureAnchorState pressureAnchorState(
    const FE::systems::FESystem& system,
    FE::FieldId pressure)
{
    PressureAnchorState state;
    const auto* registry = system.gaugeRegistryIfPresent();
    if (registry == nullptr) {
        return state;
    }
    constexpr std::string_view expected_source =
        "Unfitted CutVolume embedded free-surface natural traction anchors absolute pressure";
    state.natural_traction_anchor = std::any_of(
        registry->anchoring().begin(),
        registry->anchoring().end(),
        [&](const FE::gauge::AnchoringEvidence& evidence) {
            return evidence.field == pressure &&
                   evidence.family ==
                       FE::gauge::NullspaceModeFamily::ScalarConstant &&
                   evidence.verdict == FE::gauge::AnchoringVerdict::Anchored &&
                   evidence.source == expected_source;
        });
    state.no_gauge_enforcement = std::any_of(
        registry->resolvedModes().begin(),
        registry->resolvedModes().end(),
        [&](const FE::gauge::ResolvedMode& mode) {
            return mode.candidate.field == pressure &&
                   mode.candidate.family ==
                       FE::gauge::NullspaceModeFamily::ScalarConstant &&
                   mode.status == FE::gauge::GaugeStatus::Anchored &&
                   mode.policy == FE::gauge::EnforcementPolicy::None;
        });
    return state;
}

[[nodiscard]] FE::assembly::DenseMatrixView assembleOperatorMatrix(
    FE::systems::FESystem& system,
    const FE::systems::SystemStateView& state,
    std::string op)
{
    const auto n = system.dofHandler().getNumDofs();
    FE::assembly::DenseMatrixView matrix(n);
    matrix.zero();
    FE::systems::AssemblyRequest request;
    request.op = std::move(op);
    request.want_matrix = true;
    const auto result = system.assemble(request, state, &matrix, nullptr);
    if (!result.success) {
        throw std::runtime_error(result.error_message);
    }
    return matrix;
}

[[nodiscard]] std::vector<FE::Real> assembleOperatorResidual(
    FE::systems::FESystem& system,
    const FE::systems::SystemStateView& state,
    std::string op)
{
    const auto n = system.dofHandler().getNumDofs();
    FE::assembly::DenseVectorView residual(n);
    residual.zero();
    FE::systems::AssemblyRequest request;
    request.op = std::move(op);
    request.want_vector = true;
    const auto result = system.assemble(request, state, nullptr, &residual);
    if (!result.success) {
        throw std::runtime_error(result.error_message);
    }
    std::vector<FE::Real> values(static_cast<std::size_t>(n), FE::Real{0.0});
    for (FE::GlobalIndex dof = 0; dof < n; ++dof) {
        values[static_cast<std::size_t>(dof)] = residual[dof];
    }
    return values;
}

[[nodiscard]] FE::Real unconstrainedResidualNorm(
    const FE::systems::FESystem& system,
    std::span<const FE::Real> residual)
{
    if (residual.size() !=
        static_cast<std::size_t>(system.dofHandler().getNumDofs())) {
        throw std::invalid_argument(
            "manufactured residual size does not match the system");
    }
    FE::Real norm_squared = 0.0;
    for (FE::GlobalIndex dof = 0;
         dof < system.dofHandler().getNumDofs();
         ++dof) {
        if (system.constraints().isConstrained(dof)) {
            continue;
        }
        const auto value = residual[static_cast<std::size_t>(dof)];
        norm_squared += value * value;
    }
    return std::sqrt(norm_squared);
}

[[nodiscard]] FE::Real selectedFrobeniusNorm(
    const FE::assembly::DenseMatrixView& matrix,
    std::span<const FE::GlobalIndex> rows,
    std::span<const FE::GlobalIndex> columns)
{
    FE::Real norm_squared = 0.0;
    for (const auto row : rows) {
        for (const auto column : columns) {
            const auto value = matrix(row, column);
            norm_squared += value * value;
        }
    }
    return std::sqrt(norm_squared);
}

[[nodiscard]] FE::Real assemblePhysicalActiveVolume(
    FE::systems::FESystem& system,
    FE::FieldId scalar_field,
    const FE::spaces::FunctionSpace& scalar_space,
    const FE::assembly::CutIntegrationContext& context,
    int marker)
{
    FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(system.fieldDofHandler(scalar_field).getDofMap());
    assembler.initialize();
    FE::assembly::DenseVectorView rhs(
        system.fieldDofHandler(scalar_field).getNumDofs());
    PhysicalCutVolumeMeasureKernel kernel;
    const auto result = assembler.assembleCutVolumes(
        system.meshAccess(),
        context,
        marker,
        FE::geometry::CutIntegrationSide::Negative,
        scalar_space,
        scalar_space,
        kernel,
        /*matrix_view=*/nullptr,
        &rhs,
        /*assemble_matrix=*/false,
        /*assemble_vector=*/true);
    if (!result.success) {
        throw std::runtime_error(result.error_message);
    }
    return std::accumulate(rhs.data().begin(), rhs.data().end(), FE::Real{0.0});
}

[[nodiscard]] std::vector<FE::Real> extractReducedMatrix(
    const FE::assembly::DenseMatrixView& matrix,
    std::span<const FE::GlobalIndex> dofs)
{
    const auto n = dofs.size();
    std::vector<FE::Real> reduced(n * n, 0.0);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            reduced[row * n + column] = matrix(dofs[row], dofs[column]);
        }
    }
    return reduced;
}

[[nodiscard]] std::vector<FE::Real> extractRectangularMatrix(
    const FE::assembly::DenseMatrixView& matrix,
    std::span<const FE::GlobalIndex> rows,
    std::span<const FE::GlobalIndex> columns)
{
    std::vector<FE::Real> reduced(
        rows.size() * columns.size(), FE::Real{0.0});
    for (std::size_t row = 0; row < rows.size(); ++row) {
        for (std::size_t column = 0; column < columns.size(); ++column) {
            reduced[row * columns.size() + column] =
                matrix(rows[row], columns[column]);
        }
    }
    return reduced;
}

[[nodiscard]] std::vector<FE::Real> assembleRawActivePressureMass(
    FE::systems::FESystem& system,
    FE::FieldId pressure,
    const FE::spaces::FunctionSpace& pressure_space,
    const FE::assembly::CutIntegrationContext& context,
    int marker)
{
    const auto pressure_dofs =
        system.fieldDofHandler(pressure).getNumDofs();
    FE::assembly::DenseMatrixView mass(pressure_dofs);
    mass.zero();
    FE::assembly::MassKernel kernel(FE::Real{1.0});
    FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(
        system.fieldDofHandler(pressure).getDofMap());
    assembler.initialize();
    const auto result = assembler.assembleCutVolumes(
        system.meshAccess(),
        context,
        marker,
        FE::geometry::CutIntegrationSide::Negative,
        pressure_space,
        pressure_space,
        kernel,
        &mass,
        /*rhs=*/nullptr,
        /*assemble_matrix=*/true,
        /*assemble_vector=*/false);
    if (!result.success) {
        throw std::runtime_error(result.error_message);
    }
    return std::vector<FE::Real>(mass.data().begin(), mass.data().end());
}

[[nodiscard]] std::vector<FE::Real> reduceFieldMatrixByConstraints(
    std::span<const FE::Real> matrix,
    const FE::systems::FESystem& system,
    FE::FieldId field,
    std::span<const FE::GlobalIndex> free_dofs)
{
    const auto field_dofs = system.fieldDofHandler(field).getNumDofs();
    const auto field_size = static_cast<std::size_t>(field_dofs);
    if (matrix.size() != field_size * field_size) {
        throw std::invalid_argument(
            "raw field matrix does not match the field dimension");
    }
    const auto offset = system.fieldDofOffset(field);
    constexpr auto invalid_reduced_index =
        std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> free_index(
        field_size, invalid_reduced_index);
    for (std::size_t reduced = 0u;
         reduced < free_dofs.size();
         ++reduced) {
        const auto global = free_dofs[reduced];
        if (global < offset || global >= offset + field_dofs) {
            throw std::runtime_error(
                "free field DOF lies outside the field");
        }
        auto& index =
            free_index[static_cast<std::size_t>(global - offset)];
        if (index != invalid_reduced_index) {
            throw std::runtime_error("free field DOF is duplicated");
        }
        index = reduced;
    }
    const auto find_free = [&](FE::GlobalIndex global) -> std::size_t {
        if (global < offset || global >= offset + field_dofs) {
            throw std::runtime_error(
                "field constraint master lies outside the field");
        }
        const auto index =
            free_index[static_cast<std::size_t>(global - offset)];
        if (index == invalid_reduced_index) {
            throw std::runtime_error(
                "closed field constraint retains a constrained master");
        }
        return index;
    };

    using ExpansionEntry = std::pair<std::size_t, FE::Real>;
    std::vector<std::vector<ExpansionEntry>> expansions(field_size);
    const auto& constraints = system.constraints();
    for (FE::GlobalIndex local = 0; local < field_dofs; ++local) {
        const auto global = offset + local;
        const auto line = constraints.getConstraint(global);
        auto& expansion = expansions[static_cast<std::size_t>(local)];
        if (!line.has_value()) {
            expansion.push_back({find_free(global), FE::Real{1.0}});
            continue;
        }
        if (std::abs(line->inhomogeneity) > FE::Real{1.0e-14}) {
            throw std::runtime_error(
                "pressure mass reduction requires homogeneous constraints");
        }
        for (const auto& entry : line->entries) {
            if (entry.master_dof < offset ||
                entry.master_dof >= offset + field_dofs) {
                throw std::runtime_error(
                    "field constraint master lies outside the field");
            }
            expansion.push_back(
                {find_free(entry.master_dof),
                 static_cast<FE::Real>(entry.weight)});
        }
    }

    const auto reduced_size = free_dofs.size();
    std::vector<FE::Real> reduced(
        reduced_size * reduced_size, FE::Real{0.0});
    for (std::size_t row = 0; row < field_size; ++row) {
        for (std::size_t column = 0; column < field_size; ++column) {
            const auto value = matrix[row * field_size + column];
            if (value == FE::Real{0.0}) {
                continue;
            }
            for (const auto& [reduced_row, row_weight] : expansions[row]) {
                for (const auto& [reduced_column, column_weight] :
                     expansions[column]) {
                    reduced[reduced_row * reduced_size + reduced_column] +=
                        row_weight * value * column_weight;
                }
            }
        }
    }
    return reduced;
}

[[nodiscard]] FE::Real relativeMatrixSkew(
    std::span<const FE::Real> matrix,
    std::size_t n)
{
    if (matrix.size() != n * n) {
        throw std::invalid_argument("matrix skew size mismatch");
    }
    FE::Real maximum_skew = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = row + 1u; column < n; ++column) {
            maximum_skew = std::max(
                maximum_skew,
                std::abs(matrix[row * n + column] -
                         matrix[column * n + row]));
        }
    }
    return maximum_skew /
           std::max(FE::math::dense_matrix_max_abs(matrix),
                    std::numeric_limits<FE::Real>::min());
}

void symmetrize(std::vector<FE::Real>& matrix, std::size_t n)
{
    if (matrix.size() != n * n) {
        throw std::invalid_argument("matrix symmetrization size mismatch");
    }
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = row + 1u; column < n; ++column) {
            const auto average = FE::Real{0.5} *
                (matrix[row * n + column] + matrix[column * n + row]);
            matrix[row * n + column] = average;
            matrix[column * n + row] = average;
        }
    }
}

[[nodiscard]] std::vector<FE::Real> choleskyLower(
    std::span<const FE::Real> matrix,
    std::size_t n,
    std::string_view label)
{
    if (matrix.size() != n * n || n == 0u) {
        throw std::invalid_argument(
            std::string(label) + ": invalid Cholesky dimensions");
    }
    const auto scale = FE::math::dense_matrix_max_abs(matrix);
    const auto positive_tolerance =
        FE::Real{1024.0} * std::numeric_limits<FE::Real>::epsilon() *
        static_cast<FE::Real>(n) *
        std::max(scale, std::numeric_limits<FE::Real>::min());
    std::vector<FE::Real> lower(n * n, FE::Real{0.0});
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            FE::Real value = matrix[row * n + column];
            for (std::size_t k = 0; k < column; ++k) {
                value -= lower[row * n + k] * lower[column * n + k];
            }
            if (row == column) {
                if (!(value > positive_tolerance) || !std::isfinite(value)) {
                    throw std::runtime_error(
                        std::string(label) +
                        ": matrix is not numerically positive definite");
                }
                lower[row * n + column] = std::sqrt(value);
            } else {
                lower[row * n + column] =
                    value / lower[column * n + column];
            }
        }
    }
    return lower;
}

void leftMultiplyByInverseLower(std::vector<FE::Real>& matrix,
                                std::size_t rows,
                                std::size_t columns,
                                std::span<const FE::Real> lower)
{
    if (matrix.size() != rows * columns || lower.size() != rows * rows) {
        throw std::invalid_argument("left triangular solve size mismatch");
    }
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::size_t row = 0; row < rows; ++row) {
            FE::Real value = matrix[row * columns + column];
            for (std::size_t k = 0; k < row; ++k) {
                value -= lower[row * rows + k] *
                         matrix[k * columns + column];
            }
            matrix[row * columns + column] =
                value / lower[row * rows + row];
        }
    }
}

void rightMultiplyByInverseLowerTranspose(
    std::vector<FE::Real>& matrix,
    std::size_t rows,
    std::size_t columns,
    std::span<const FE::Real> lower)
{
    if (matrix.size() != rows * columns ||
        lower.size() != columns * columns) {
        throw std::invalid_argument("right triangular solve size mismatch");
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            FE::Real value = matrix[row * columns + column];
            for (std::size_t k = 0; k < column; ++k) {
                value -= lower[column * columns + k] *
                         matrix[row * columns + k];
            }
            matrix[row * columns + column] =
                value / lower[column * columns + column];
        }
    }
}

/** Finite-dimensional pressure-control diagnostics on the constrained spaces.
 * With A the symmetric transient-Stokes velocity block, M_p the active-domain
 * pressure mass, B the Galerkin continuity block, G the assembled momentum
 * pressure-gradient block, and S the production PSPG-plus-pressure-ghost
 * block, this computes
 *
 *   sigma_min(L_p^-1 B L_A^-T)
 *
 * and the smallest eigenvalue of
 *
 *   L_p^-1 (S - B A^-1 G) L_p^-T.
 *
 * The first is the generalized Galerkin pressure/velocity coupling value for
 * this finite space.  The second measures stabilized Schur pressure control.
 * Cholesky factorizations validate the stated energy/L2 norms and positive
 * definiteness before SVD diagnostics are accepted. */
[[nodiscard]] PressureControlMetrics pressureControlMetrics(
    const FE::assembly::DenseMatrixView& jacobian,
    const FE::assembly::DenseMatrixView& galerkin_continuity,
    const FE::assembly::DenseMatrixView& pressure_ghost,
    const FE::assembly::DenseMatrixView& pressure_pspg,
    std::span<const FE::Real> pressure_mass,
    std::span<const FE::GlobalIndex> free_velocity,
    std::span<const FE::GlobalIndex> free_pressure)
{
    const auto velocity_size = free_velocity.size();
    const auto pressure_size = free_pressure.size();
    if (velocity_size == 0u || pressure_size == 0u ||
        pressure_mass.size() != pressure_size * pressure_size) {
        throw std::invalid_argument(
            "pressure-control diagnostic has invalid dimensions");
    }

    auto velocity_energy = extractRectangularMatrix(
        jacobian, free_velocity, free_velocity);
    const auto velocity_skew =
        relativeMatrixSkew(velocity_energy, velocity_size);
    symmetrize(velocity_energy, velocity_size);
    auto pressure_l2 = std::vector<FE::Real>(
        pressure_mass.begin(), pressure_mass.end());
    symmetrize(pressure_l2, pressure_size);
    const auto velocity_lower = choleskyLower(
        velocity_energy, velocity_size, "free velocity energy block");
    const auto pressure_lower = choleskyLower(
        pressure_l2, pressure_size, "active pressure L2 mass");

    const auto galerkin_coupling = extractRectangularMatrix(
        galerkin_continuity, free_pressure, free_velocity);
    const auto assembled_pressure_gradient = extractRectangularMatrix(
        jacobian, free_velocity, free_pressure);
    long double adjoint_defect_squared = 0.0L;
    long double adjoint_scale_squared = 0.0L;
    for (std::size_t row = 0; row < pressure_size; ++row) {
        for (std::size_t column = 0; column < velocity_size; ++column) {
            const auto continuity = static_cast<long double>(
                galerkin_coupling[row * velocity_size + column]);
            const auto gradient = static_cast<long double>(
                assembled_pressure_gradient[column * pressure_size + row]);
            const auto defect = continuity + gradient;
            adjoint_defect_squared += defect * defect;
            adjoint_scale_squared +=
                continuity * continuity + gradient * gradient;
        }
    }

    auto energy_scaled_coupling = galerkin_coupling;
    rightMultiplyByInverseLowerTranspose(
        energy_scaled_coupling,
        pressure_size,
        velocity_size,
        velocity_lower);
    auto normalized_coupling = energy_scaled_coupling;
    leftMultiplyByInverseLower(
        normalized_coupling,
        pressure_size,
        velocity_size,
        pressure_lower);
    const auto coupling_diagnostics = FE::math::dense_matrix_diagnostics(
        normalized_coupling,
        pressure_size,
        velocity_size,
        "L2/energy-normalized pressure-velocity coupling");

    auto stabilization = extractRectangularMatrix(
        pressure_ghost, free_pressure, free_pressure);
    const auto pspg = extractRectangularMatrix(
        pressure_pspg, free_pressure, free_pressure);
    for (std::size_t i = 0; i < stabilization.size(); ++i) {
        stabilization[i] += pspg[i];
    }
    symmetrize(stabilization, pressure_size);

    auto energy_scaled_pressure_gradient = assembled_pressure_gradient;
    leftMultiplyByInverseLower(
        energy_scaled_pressure_gradient,
        velocity_size,
        pressure_size,
        velocity_lower);

    std::vector<FE::Real> schur = stabilization;
    for (std::size_t row = 0; row < pressure_size; ++row) {
        for (std::size_t column = 0; column < pressure_size; ++column) {
            FE::Real value = 0.0;
            for (std::size_t k = 0; k < velocity_size; ++k) {
                value -=
                    energy_scaled_coupling[row * velocity_size + k] *
                    energy_scaled_pressure_gradient[
                        k * pressure_size + column];
            }
            schur[row * pressure_size + column] += value;
        }
    }
    symmetrize(schur, pressure_size);
    long double constant_numerator = 0.0L;
    long double constant_denominator = 0.0L;
    for (std::size_t row = 0; row < pressure_size; ++row) {
        for (std::size_t column = 0; column < pressure_size; ++column) {
            constant_numerator += static_cast<long double>(
                schur[row * pressure_size + column]);
            constant_denominator += static_cast<long double>(
                pressure_l2[row * pressure_size + column]);
        }
    }
    leftMultiplyByInverseLower(
        schur, pressure_size, pressure_size, pressure_lower);
    rightMultiplyByInverseLowerTranspose(
        schur, pressure_size, pressure_size, pressure_lower);
    symmetrize(schur, pressure_size);
    (void)choleskyLower(
        schur, pressure_size, "normalized stabilized pressure Schur block");
    const auto schur_spectrum = FE::math::dense_symmetric_eigenvalue_bounds(
        schur,
        pressure_size,
        "normalized stabilized pressure Schur block");

    PressureControlMetrics metrics;
    metrics.generalized_coupling_rank = coupling_diagnostics.rank;
    metrics.pressure_dimension = pressure_size;
    metrics.generalized_coupling_smallest_singular_value =
        coupling_diagnostics.rank == pressure_size
            ? coupling_diagnostics.smallest_retained_singular_value
            : FE::Real{0.0};
    metrics.stabilized_schur_smallest_generalized_eigenvalue =
        schur_spectrum.smallest_eigenvalue;
    metrics.stabilized_pressure_control = std::sqrt(std::max(
        FE::Real{0.0},
        metrics.stabilized_schur_smallest_generalized_eigenvalue));
    metrics.constant_pressure_control = std::sqrt(std::max(
        FE::Real{0.0},
        static_cast<FE::Real>(constant_numerator / constant_denominator)));
    metrics.velocity_block_relative_skew = velocity_skew;
    metrics.pressure_gradient_adjoint_relative_defect =
        static_cast<FE::Real>(std::sqrt(adjoint_defect_squared) /
                              std::max(std::sqrt(adjoint_scale_squared),
                                       std::numeric_limits<long double>::min()));
    return metrics;
}

/** Ruiz-like row/column equilibration.  This does not change rank; it only
 * removes arbitrary velocity/pressure unit scaling from the condition
 * surrogate reported by the sweep. */
void equilibrate(std::vector<FE::Real>& matrix, std::size_t n)
{
    constexpr FE::Real floor = FE::Real{1.0e-30};
    for (int iteration = 0; iteration < 8; ++iteration) {
        for (std::size_t row = 0; row < n; ++row) {
            FE::Real norm = 0.0;
            for (std::size_t column = 0; column < n; ++column) {
                norm = std::max(norm, std::abs(matrix[row * n + column]));
            }
            if (norm > floor) {
                const auto scale = FE::Real{1.0} / std::sqrt(norm);
                for (std::size_t column = 0; column < n; ++column) {
                    matrix[row * n + column] *= scale;
                }
            }
        }
        for (std::size_t column = 0; column < n; ++column) {
            FE::Real norm = 0.0;
            for (std::size_t row = 0; row < n; ++row) {
                norm = std::max(norm, std::abs(matrix[row * n + column]));
            }
            if (norm > floor) {
                const auto scale = FE::Real{1.0} / std::sqrt(norm);
                for (std::size_t row = 0; row < n; ++row) {
                    matrix[row * n + column] *= scale;
                }
            }
        }
    }
}

/** Exact infinity-norm condition of the equilibrated matrix:
 * kappa_inf(D_r A D_c) = ||D_r A D_c||_inf
 *                        ||(D_r A D_c)^-1||_inf.
 * The inverse is obtained from one pivoted-LU factorization with all identity
 * columns solved as a block. */
[[nodiscard]] FE::Real infinityNormCondition(
    const std::vector<FE::Real>& matrix,
    std::size_t n)
{
    FE::Real matrix_norm = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
        FE::Real row_sum = 0.0;
        for (std::size_t column = 0; column < n; ++column) {
            row_sum += std::abs(matrix[row * n + column]);
        }
        matrix_norm = std::max(matrix_norm, row_sum);
    }

    auto solver = FE::math::factor_dense_matrix(
        matrix, n, "equilibrated free-surface mixed Jacobian");
    std::vector<FE::Real> inverse(n * n, 0.0);
    for (std::size_t diagonal = 0; diagonal < n; ++diagonal) {
        inverse[diagonal * n + diagonal] = 1.0;
    }
    solver.solve_in_place(inverse, n);

    FE::Real inverse_norm = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
        FE::Real row_sum = 0.0;
        for (std::size_t column = 0; column < n; ++column) {
            row_sum += std::abs(inverse[row * n + column]);
        }
        inverse_norm = std::max(inverse_norm, row_sum);
    }
    return matrix_norm * inverse_norm;
}

[[nodiscard]] KrylovTelemetry runEquilibratedJacobiBicgstab(
    std::span<const FE::Real> matrix,
    std::size_t n)
{
    if (n == 0u || matrix.size() != n * n) {
        throw std::invalid_argument(
            "Krylov telemetry requires a nonempty square dense matrix");
    }
    const auto dot = [](std::span<const FE::Real> lhs,
                        std::span<const FE::Real> rhs) {
        if (lhs.size() != rhs.size()) {
            throw std::invalid_argument(
                "Krylov dot product requires equal vector sizes");
        }
        long double value = 0.0L;
        for (std::size_t i = 0u; i < lhs.size(); ++i) {
            value += static_cast<long double>(lhs[i]) *
                     static_cast<long double>(rhs[i]);
        }
        return static_cast<FE::Real>(value);
    };
    const auto norm = [&](std::span<const FE::Real> value) {
        return std::sqrt(std::max(FE::Real{0.0}, dot(value, value)));
    };
    const auto multiply = [&](std::span<const FE::Real> input,
                              std::span<FE::Real> output) {
        if (input.size() != n || output.size() != n) {
            throw std::invalid_argument(
                "Krylov matrix-vector dimensions do not match");
        }
        for (std::size_t row = 0u; row < n; ++row) {
            long double value = 0.0L;
            for (std::size_t column = 0u; column < n; ++column) {
                value += static_cast<long double>(
                             matrix[row * n + column]) *
                         static_cast<long double>(input[column]);
            }
            output[row] = static_cast<FE::Real>(value);
        }
    };

    std::vector<FE::Real> exact(n, FE::Real{0.0});
    for (std::size_t i = 0u; i < n; ++i) {
        const auto index = static_cast<FE::Real>(i + 1u);
        exact[i] =
            std::sin(FE::Real{0.37} * index) +
            FE::Real{0.25} * std::cos(FE::Real{0.11} * index);
    }
    std::vector<FE::Real> rhs(n, FE::Real{0.0});
    multiply(exact, rhs);
    const auto rhs_norm = norm(rhs);
    if (!(rhs_norm > FE::Real{0.0}) || !std::isfinite(rhs_norm)) {
        throw std::runtime_error(
            "Krylov telemetry manufactured right-hand side is invalid");
    }

    KrylovTelemetry telemetry;
    telemetry.iteration_limit = std::max(std::size_t{32u}, 8u * n);
    std::vector<FE::Real> inverse_diagonal(n, FE::Real{1.0});
    for (std::size_t row = 0u; row < n; ++row) {
        FE::Real row_norm = 0.0;
        for (std::size_t column = 0u; column < n; ++column) {
            row_norm += std::abs(matrix[row * n + column]);
        }
        const auto diagonal = matrix[row * n + row];
        const auto tolerance =
            FE::Real{128.0} *
            std::numeric_limits<FE::Real>::epsilon() *
            std::max(FE::Real{1.0}, row_norm);
        if (std::abs(diagonal) > tolerance) {
            inverse_diagonal[row] = FE::Real{1.0} / diagonal;
        } else {
            inverse_diagonal[row] =
                FE::Real{1.0} / std::max(FE::Real{1.0}, row_norm);
            ++telemetry.diagonal_fallback_count;
        }
    }
    const auto apply_preconditioner =
        [&](std::span<const FE::Real> input,
            std::span<FE::Real> output) {
            for (std::size_t i = 0u; i < n; ++i) {
                output[i] = inverse_diagonal[i] * input[i];
            }
        };

    std::vector<FE::Real> solution(n, FE::Real{0.0});
    std::vector<FE::Real> residual = rhs;
    std::vector<FE::Real> shadow = residual;
    std::vector<FE::Real> direction(n, FE::Real{0.0});
    std::vector<FE::Real> operator_direction(n, FE::Real{0.0});
    std::vector<FE::Real> preconditioned_direction(n, FE::Real{0.0});
    std::vector<FE::Real> intermediate(n, FE::Real{0.0});
    std::vector<FE::Real> preconditioned_intermediate(n, FE::Real{0.0});
    std::vector<FE::Real> operator_intermediate(n, FE::Real{0.0});
    FE::Real rho_previous = FE::Real{1.0};
    FE::Real alpha = FE::Real{1.0};
    FE::Real omega = FE::Real{1.0};
    constexpr FE::Real relative_tolerance = FE::Real{1.0e-9};
    const auto breakdown_tolerance =
        FE::Real{1024.0} * std::numeric_limits<FE::Real>::epsilon();

    for (std::size_t iteration = 1u;
         iteration <= telemetry.iteration_limit;
         ++iteration) {
        const auto rho = dot(shadow, residual);
        const auto rho_scale = norm(shadow) * norm(residual);
        if (!std::isfinite(rho) ||
            !(rho_scale > FE::Real{0.0}) ||
            std::abs(rho) <= breakdown_tolerance * rho_scale) {
            telemetry.breakdown = true;
            break;
        }
        if (iteration == 1u) {
            direction = residual;
        } else {
            if (!std::isfinite(omega) ||
                std::abs(omega) <= breakdown_tolerance) {
                telemetry.breakdown = true;
                break;
            }
            const auto beta =
                (rho / rho_previous) * (alpha / omega);
            for (std::size_t i = 0u; i < n; ++i) {
                direction[i] =
                    residual[i] +
                    beta *
                        (direction[i] - omega * operator_direction[i]);
            }
        }

        apply_preconditioner(direction, preconditioned_direction);
        multiply(preconditioned_direction, operator_direction);
        const auto shadow_operator = dot(shadow, operator_direction);
        const auto shadow_operator_scale =
            norm(shadow) * norm(operator_direction);
        if (!std::isfinite(shadow_operator) ||
            !(shadow_operator_scale > FE::Real{0.0}) ||
            std::abs(shadow_operator) <=
                breakdown_tolerance * shadow_operator_scale) {
            telemetry.breakdown = true;
            break;
        }
        alpha = rho / shadow_operator;
        for (std::size_t i = 0u; i < n; ++i) {
            intermediate[i] =
                residual[i] - alpha * operator_direction[i];
        }
        if (norm(intermediate) / rhs_norm <= relative_tolerance) {
            for (std::size_t i = 0u; i < n; ++i) {
                solution[i] += alpha * preconditioned_direction[i];
            }
            telemetry.iterations = iteration;
            telemetry.converged = true;
            residual = intermediate;
            break;
        }

        apply_preconditioner(
            intermediate, preconditioned_intermediate);
        multiply(preconditioned_intermediate, operator_intermediate);
        const auto operator_norm_squared =
            dot(operator_intermediate, operator_intermediate);
        const auto intermediate_norm = norm(intermediate);
        const auto operator_breakdown_scale =
            breakdown_tolerance * intermediate_norm;
        if (!std::isfinite(operator_norm_squared) ||
            operator_norm_squared <=
                operator_breakdown_scale * operator_breakdown_scale) {
            telemetry.breakdown = true;
            break;
        }
        omega =
            dot(operator_intermediate, intermediate) /
            operator_norm_squared;
        if (!std::isfinite(omega)) {
            telemetry.breakdown = true;
            break;
        }
        for (std::size_t i = 0u; i < n; ++i) {
            solution[i] +=
                alpha * preconditioned_direction[i] +
                omega * preconditioned_intermediate[i];
            residual[i] =
                intermediate[i] - omega * operator_intermediate[i];
        }
        telemetry.iterations = iteration;
        telemetry.relative_residual = norm(residual) / rhs_norm;
        if (telemetry.relative_residual <= relative_tolerance) {
            telemetry.converged = true;
            break;
        }
        rho_previous = rho;
    }

    std::vector<FE::Real> recomputed(n, FE::Real{0.0});
    multiply(solution, recomputed);
    for (std::size_t i = 0u; i < n; ++i) {
        recomputed[i] = rhs[i] - recomputed[i];
    }
    telemetry.relative_residual = norm(recomputed) / rhs_norm;
    std::vector<FE::Real> solution_error(n, FE::Real{0.0});
    for (std::size_t i = 0u; i < n; ++i) {
        solution_error[i] = solution[i] - exact[i];
    }
    telemetry.relative_solution_error =
        norm(solution_error) /
        std::max(FE::Real{1.0}, norm(exact));
    telemetry.converged =
        telemetry.converged &&
        std::isfinite(telemetry.relative_residual) &&
        telemetry.relative_residual <= FE::Real{2.0e-9};
    return telemetry;
}

[[nodiscard]] ns::IncompressibleNavierStokesVMSOptions stabilityOptions(
    int interface_marker,
    std::string domain_id,
    const StabilityRegime& regime = {})
{
    ns::IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = "u";
    options.pressure_field_name = "p";
    options.density = regime.density;
    options.viscosity = regime.viscosity;
    options.enable_convection = regime.convection;
    options.enable_vms = true;
    options.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi",
            .generated_interface_domain_id = std::move(domain_id),
            .level_set_isovalue = 0.0,
            .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
            .active_domain_method = ns::FreeSurfaceActiveDomainMethod::CutVolume,
            .external_pressure = 0.0,
            .surface_tension = 0.0,
            .use_level_set_curvature = false,
            .cut_cell_stabilization = {
                .enabled = true,
                .pressure_gradient_penalty = 1.0,
                .pressure_policy =
                    ns::FreeSurfacePressureStabilizationPolicy::Enabled,
                .use_cut_metadata_scale = true,
                .cut_metadata_scale_cap = FE::Real{100.0},
            },
            .small_cut_aggregation = true,
        });
    return options;
}

struct CanonicalWetBlockDof {
    int field{0};
    gid_t vertex_gid{INVALID_GID};
    std::size_t component{0u};
    FE::GlobalIndex global_dof{FE::INVALID_GLOBAL_INDEX};
    std::array<FE::Real, 3> point{};
};

struct WetBlockAssemblySample {
    std::vector<CanonicalWetBlockDof> dofs{};
    std::vector<FE::Real> current_state{};
    std::vector<FE::Real> solved_state{};
    std::vector<FE::Real> residual{};
    std::vector<FE::Real> jacobian{};
    FE::Real dry_column_coupling_norm{0.0};
    std::size_t retained_vertices{0u};
    std::size_t constrained_dry_velocity_dofs{0u};
    std::size_t constrained_dry_pressure_dofs{0u};
};

struct ScaledWetBlockDifference {
    FE::Real residual{0.0};
    FE::Real jacobian{0.0};
    FE::Real solved_state{0.0};
    FE::Real residual_absolute{0.0};
    FE::Real jacobian_absolute{0.0};
    FE::Real solved_state_absolute{0.0};
};

[[nodiscard]] FE::Real vectorL2Norm(std::span<const FE::Real> values)
{
    long double squared = 0.0L;
    for (const auto value : values) {
        squared += static_cast<long double>(value) *
                   static_cast<long double>(value);
    }
    return static_cast<FE::Real>(std::sqrt(squared));
}

[[nodiscard]] FE::Real vectorDifferenceL2Norm(
    std::span<const FE::Real> lhs,
    std::span<const FE::Real> rhs)
{
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument(
            "wet-block comparison requires equal vector sizes");
    }
    long double squared = 0.0L;
    for (std::size_t index = 0u; index < lhs.size(); ++index) {
        const auto difference = static_cast<long double>(lhs[index]) -
                                static_cast<long double>(rhs[index]);
        squared += difference * difference;
    }
    return static_cast<FE::Real>(std::sqrt(squared));
}

[[nodiscard]] bool sameWetBlockDofIdentity(
    const CanonicalWetBlockDof& lhs,
    const CanonicalWetBlockDof& rhs) noexcept
{
    return lhs.field == rhs.field && lhs.vertex_gid == rhs.vertex_gid &&
           lhs.component == rhs.component && lhs.point == rhs.point;
}

[[nodiscard]] ScaledWetBlockDifference compareWetBlockSamples(
    const WetBlockAssemblySample& baseline,
    const WetBlockAssemblySample& candidate)
{
    if (baseline.dofs.size() != candidate.dofs.size() ||
        baseline.current_state.size() != candidate.current_state.size() ||
        baseline.solved_state.size() != candidate.solved_state.size() ||
        baseline.residual.size() != candidate.residual.size() ||
        baseline.jacobian.size() != candidate.jacobian.size()) {
        throw std::invalid_argument(
            "wet-block comparison requires matching canonical block sizes");
    }
    for (std::size_t index = 0u; index < baseline.dofs.size(); ++index) {
        if (!sameWetBlockDofIdentity(
                baseline.dofs[index], candidate.dofs[index])) {
            throw std::invalid_argument(
                "wet-block comparison encountered different canonical DOF identities");
        }
    }

    // The fixture is nondimensional.  These absolute floors prevent an
    // accidentally tiny reference block from making a roundoff-sized delta
    // appear large; the relative gates below are the predeclared WP-1 values.
    constexpr FE::Real residual_absolute_floor = FE::Real{1.0e-12};
    constexpr FE::Real jacobian_absolute_floor = FE::Real{1.0e-12};
    constexpr FE::Real solved_state_absolute_floor = FE::Real{1.0e-12};
    ScaledWetBlockDifference difference;
    difference.residual_absolute = vectorDifferenceL2Norm(
        baseline.residual, candidate.residual);
    difference.jacobian_absolute = vectorDifferenceL2Norm(
        baseline.jacobian, candidate.jacobian);
    difference.solved_state_absolute = vectorDifferenceL2Norm(
        baseline.solved_state, candidate.solved_state);
    difference.residual = difference.residual_absolute /
        (residual_absolute_floor +
         std::max(vectorL2Norm(baseline.residual),
                  vectorL2Norm(candidate.residual)));
    difference.jacobian = difference.jacobian_absolute /
        (jacobian_absolute_floor +
         std::max(vectorL2Norm(baseline.jacobian),
                  vectorL2Norm(candidate.jacobian)));
    difference.solved_state = difference.solved_state_absolute /
        (solved_state_absolute_floor +
         std::max(vectorL2Norm(baseline.solved_state),
                  vectorL2Norm(candidate.solved_state)));
    return difference;
}

[[nodiscard]] std::set<gid_t> retainedWetBlockVertexGids(
    const Mesh& mesh,
    const FE::assembly::CutIntegrationContext& context,
    int marker)
{
    const auto cells = retainedCells(context, marker, /*cut_only=*/false);
    const auto& base = mesh.base();
    const auto& vertex_gids = base.vertex_gids();
    if (vertex_gids.size() != base.n_vertices()) {
        throw std::runtime_error(
            "wet-block mesh has incomplete vertex global identities");
    }
    std::set<gid_t> retained;
    for (const auto cell : cells) {
        if (cell < 0 ||
            static_cast<std::size_t>(cell) >= base.n_cells()) {
            throw std::runtime_error(
                "wet-block retained cell is outside the local mesh");
        }
        for (const auto vertex : base.cell_vertices(
                 static_cast<index_t>(cell))) {
            retained.insert(vertex_gids.at(
                static_cast<std::size_t>(vertex)));
        }
    }
    if (retained.empty()) {
        throw std::runtime_error("wet-block retained vertex set is empty");
    }
    return retained;
}

void setWetBlockP1Field(
    std::vector<FE::Real>& state,
    const Mesh& mesh,
    const FE::systems::FESystem& system,
    FE::FieldId field,
    std::size_t components,
    const std::set<gid_t>& retained,
    FE::Real dry_state_scale,
    bool previous_state)
{
    const auto* entity_map =
        system.fieldDofHandler(field).getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error(
            "wet-block P1 field has no entity DOF map");
    }
    const auto offset = system.fieldDofOffset(field);
    const auto& base = mesh.base();
    const auto& vertex_gids = base.vertex_gids();
    for (index_t vertex = 0;
         vertex < static_cast<index_t>(base.n_vertices());
         ++vertex) {
        const auto dofs = entity_map->getVertexDofs(
            static_cast<FE::GlobalIndex>(vertex));
        if (dofs.size() != components) {
            throw std::runtime_error(
                "wet-block field does not have the expected P1 components");
        }
        const auto point = system.meshAccess().getNodeCoordinates(vertex);
        const bool wet_supported = retained.contains(
            vertex_gids.at(static_cast<std::size_t>(vertex)));
        for (std::size_t component = 0u; component < components; ++component) {
            FE::Real value = 0.0;
            if (wet_supported) {
                if (components == 1u) {
                    value = FE::Real{0.21} - FE::Real{0.07} * point[0] +
                            FE::Real{0.09} * point[1];
                } else if (component == 0u) {
                    value = FE::Real{0.73} + FE::Real{0.16} * point[0] -
                            FE::Real{0.08} * point[1];
                } else {
                    value = -FE::Real{0.31} + FE::Real{0.11} * point[0] +
                            FE::Real{0.17} * point[1];
                }
                if (previous_state && components > 1u) {
                    value -= FE::Real{0.03} *
                             static_cast<FE::Real>(component + 1u);
                }
            } else {
                value = dry_state_scale *
                    (FE::Real{1.0} + FE::Real{0.13} * point[0] +
                     FE::Real{0.19} * point[1] +
                     FE::Real{0.07} * static_cast<FE::Real>(component));
                if (previous_state) {
                    value *= FE::Real{-0.4};
                }
            }
            state.at(static_cast<std::size_t>(offset + dofs[component])) =
                value;
        }
    }
}

[[nodiscard]] WetBlockAssemblySample assembleSerialWetBlockSample(
    std::span<const FE::Real> x_coordinates,
    std::span<const FE::Real> level_set_by_plane,
    FE::Real dry_state_scale)
{
    constexpr int interface_marker = 27305;
    constexpr std::string_view domain_id =
        "wp1_serial_physical_wet_block_invariance";
    auto mesh = makeWetBlockQuadStrip(x_coordinates, level_set_by_plane);
    auto scalar_space = FE::spaces::SpaceFactory::create_h1(
        FE::ElementType::Quad4, /*order=*/1);
    auto velocity_space = FE::spaces::SpaceFactory::create_vector_h1(
        FE::ElementType::Quad4, /*order=*/1, /*components=*/2);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::Unknown,
    });

    ns::IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = "u";
    options.pressure_field_name = "p";
    options.density = FE::Real{1.2};
    options.viscosity = FE::Real{0.07};
    options.enable_convection = false;
    options.enable_vms = true;
    ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary free_surface;
    free_surface.implementation =
        ns::FreeSurfaceImplementation::UnfittedLevelSet;
    free_surface.interface_marker = interface_marker;
    free_surface.level_set_field_name = "phi";
    free_surface.generated_interface_domain_id = std::string(domain_id);
    free_surface.active_domain =
        ns::FreeSurfaceActiveDomain::LevelSetNegative;
    free_surface.active_domain_method =
        ns::FreeSurfaceActiveDomainMethod::CutVolume;
    free_surface.external_pressure = FE::Real{0.0};
    free_surface.surface_tension = FE::Real{0.0};
    free_surface.use_level_set_curvature = false;
    free_surface.cut_cell_stabilization.enabled = false;
    free_surface.small_cut_aggregation = false;
    free_surface.velocity_extension.enabled = false;
    options.free_surface.push_back(std::move(free_surface));

    ns::IncompressibleNavierStokesVMSModule module(
        velocity_space, scalar_space, options);
    module.registerOn(system);
    system.setup({});
    const auto velocity = system.findFieldByName("u");
    const auto pressure = system.findFieldByName("p");
    if (phi == FE::INVALID_FIELD_ID || velocity == FE::INVALID_FIELD_ID ||
        pressure == FE::INVALID_FIELD_ID) {
        throw std::runtime_error(
            "wet-block Navier--Stokes fields were not registered");
    }

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous = solution;
    const auto phi_handle = MeshFields::get_field_handle(
        mesh->base(), EntityKind::Vertex, "phi");
    const auto* mesh_phi = MeshFields::field_data_as<real_t>(
        mesh->base(), phi_handle);
    const auto* phi_map = system.fieldDofHandler(phi).getEntityDofMap();
    if (mesh_phi == nullptr || phi_map == nullptr) {
        throw std::runtime_error(
            "wet-block level-set field is unavailable after setup");
    }
    const auto phi_offset = system.fieldDofOffset(phi);
    for (index_t vertex = 0;
         vertex < static_cast<index_t>(mesh->base().n_vertices());
         ++vertex) {
        const auto dofs = phi_map->getVertexDofs(vertex);
        if (dofs.size() != 1u) {
            throw std::runtime_error(
                "wet-block level set is not scalar P1");
        }
        const auto value = mesh_phi[static_cast<std::size_t>(vertex)];
        solution.at(static_cast<std::size_t>(phi_offset + dofs.front())) =
            value;
        previous.at(static_cast<std::size_t>(phi_offset + dofs.front())) =
            value;
    }

    FE::level_set::LevelSetGeneratedInterfaceOptions cut_options;
    cut_options.level_set_field_name = "phi";
    cut_options.domain_id = std::string(domain_id);
    cut_options.requested_interface_marker = interface_marker;
    cut_options.tolerance = FE::Real{1.0e-12};
    cut_options.quadrature_order = 2;
    cut_options.interface_quadrature_order = 2;
    cut_options.volume_quadrature_order = 2;
    FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto generated = lifecycle.build(system, cut_options, solution);
    if (!generated.success) {
        throw std::runtime_error(generated.diagnostic);
    }
    auto context = std::make_shared<FE::assembly::CutIntegrationContext>();
    context->addGeneratedInterfaceDomain(generated.domain);
    system.setCutIntegrationContext(context);
    system.rebuildConstraintState();

    const auto retained = retainedWetBlockVertexGids(
        *mesh, *context, interface_marker);
    setWetBlockP1Field(
        solution,
        *mesh,
        system,
        velocity,
        /*components=*/2u,
        retained,
        dry_state_scale,
        /*previous_state=*/false);
    setWetBlockP1Field(
        previous,
        *mesh,
        system,
        velocity,
        /*components=*/2u,
        retained,
        dry_state_scale,
        /*previous_state=*/true);
    setWetBlockP1Field(
        solution,
        *mesh,
        system,
        pressure,
        /*components=*/1u,
        retained,
        dry_state_scale,
        /*previous_state=*/false);
    setWetBlockP1Field(
        previous,
        *mesh,
        system,
        pressure,
        /*components=*/1u,
        retained,
        dry_state_scale,
        /*previous_state=*/true);

    FE::systems::SystemStateView state;
    state.dt = FE::Real{0.125};
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;
    const auto matrix = assembleOperatorMatrix(system, state, "equations");
    const auto residual =
        assembleOperatorResidual(system, state, "equations");

    WetBlockAssemblySample sample;
    sample.retained_vertices = retained.size();
    const auto& base = mesh->base();
    const auto& gids = base.vertex_gids();
    std::vector<FE::GlobalIndex> dry_columns;
    const auto append_field = [&](FE::FieldId field,
                                  int field_index,
                                  std::size_t components,
                                  std::size_t& dry_constraint_count) {
        const auto* entity_map =
            system.fieldDofHandler(field).getEntityDofMap();
        if (entity_map == nullptr) {
            throw std::runtime_error(
                "wet-block canonical field has no entity map");
        }
        const auto offset = system.fieldDofOffset(field);
        for (index_t vertex = 0;
             vertex < static_cast<index_t>(base.n_vertices());
             ++vertex) {
            const auto vertex_dofs = entity_map->getVertexDofs(vertex);
            if (vertex_dofs.size() != components) {
                throw std::runtime_error(
                    "wet-block canonical field has an unexpected component count");
            }
            const auto gid = gids.at(static_cast<std::size_t>(vertex));
            const bool wet_supported = retained.contains(gid);
            for (std::size_t component = 0u;
                 component < components;
                 ++component) {
                const auto global = offset + vertex_dofs[component];
                if (wet_supported) {
                    if (system.constraints().isConstrained(global)) {
                        throw std::runtime_error(
                            "wet-supported physical DOF was constrained");
                    }
                    sample.dofs.push_back(CanonicalWetBlockDof{
                        .field = field_index,
                        .vertex_gid = gid,
                        .component = component,
                        .global_dof = global,
                        .point = system.meshAccess().getNodeCoordinates(vertex),
                    });
                } else {
                    const auto line =
                        system.constraints().getConstraint(global);
                    if (!line.has_value() || !line->isDirichlet() ||
                        line->inhomogeneity != FE::Real{0.0}) {
                        throw std::runtime_error(
                            "dry-only physical DOF was not homogeneously constrained");
                    }
                    ++dry_constraint_count;
                    dry_columns.push_back(global);
                }
            }
        }
    };
    append_field(
        velocity,
        /*field_index=*/0,
        /*components=*/2u,
        sample.constrained_dry_velocity_dofs);
    append_field(
        pressure,
        /*field_index=*/1,
        /*components=*/1u,
        sample.constrained_dry_pressure_dofs);
    std::sort(
        sample.dofs.begin(),
        sample.dofs.end(),
        [](const CanonicalWetBlockDof& lhs,
           const CanonicalWetBlockDof& rhs) {
            if (lhs.field != rhs.field) {
                return lhs.field < rhs.field;
            }
            if (lhs.vertex_gid != rhs.vertex_gid) {
                return lhs.vertex_gid < rhs.vertex_gid;
            }
            return lhs.component < rhs.component;
        });
    std::vector<FE::GlobalIndex> wet_dofs;
    wet_dofs.reserve(sample.dofs.size());
    sample.current_state.reserve(sample.dofs.size());
    sample.residual.reserve(sample.dofs.size());
    for (const auto& dof : sample.dofs) {
        wet_dofs.push_back(dof.global_dof);
        sample.current_state.push_back(
            solution.at(static_cast<std::size_t>(dof.global_dof)));
        sample.residual.push_back(
            residual.at(static_cast<std::size_t>(dof.global_dof)));
    }
    sample.jacobian = extractRectangularMatrix(
        matrix, wet_dofs, wet_dofs);
    auto correction = sample.residual;
    for (auto& value : correction) {
        value = -value;
    }
    auto solver = FE::math::factor_dense_matrix(
        sample.jacobian,
        sample.dofs.size(),
        "serial WP-1 physical wet-block Jacobian");
    solver.solve_in_place(correction, /*right_hand_sides=*/1u);
    sample.solved_state = sample.current_state;
    for (std::size_t index = 0u;
         index < sample.solved_state.size();
         ++index) {
        sample.solved_state[index] += correction[index];
    }
    sample.dry_column_coupling_norm = selectedFrobeniusNorm(
        matrix, wet_dofs, dry_columns);
    return sample;
}

struct ActiveCellTopologySample {
    FE::FieldId velocity{FE::INVALID_FIELD_ID};
    FE::FieldId pressure{FE::INVALID_FIELD_ID};
    FE::constraints::SmallCutAggregationRefreshReport velocity_report{};
    FE::constraints::SmallCutAggregationRefreshReport pressure_report{};
    FE::Real assembled_active_physical_volume{0.0};
};

[[nodiscard]] ActiveCellTopologySample
assembleSerialActiveCellTopologySample(
    std::span<const FE::Real> level_set_by_plane)
{
    constexpr int interface_marker = 27315;
    constexpr std::string_view domain_id =
        "wp7_active_cell_topology_policy";
    constexpr std::array<FE::Real, 7> x_coordinates = {
        FE::Real{0.0},
        FE::Real{1.0},
        FE::Real{2.0},
        FE::Real{3.0},
        FE::Real{4.0},
        FE::Real{5.0},
        FE::Real{6.0},
    };
    if (level_set_by_plane.size() != x_coordinates.size()) {
        throw std::invalid_argument(
            "active-cell topology sample requires seven level-set planes");
    }

    auto mesh = makeWetBlockQuadStrip(
        x_coordinates, level_set_by_plane);
    const auto& cell_gids = mesh->base().cell_gids();
    if (cell_gids.size() != x_coordinates.size() - 1u) {
        throw std::runtime_error(
            "active-cell topology strip has unexpected cell GIDs");
    }
    for (std::size_t cell = 0u; cell < cell_gids.size(); ++cell) {
        if (cell_gids[cell] != static_cast<gid_t>(cell)) {
            throw std::runtime_error(
                "active-cell topology strip requires canonical iota cell GIDs");
        }
    }

    auto scalar_space = FE::spaces::SpaceFactory::create_h1(
        FE::ElementType::Quad4, /*order=*/1);
    auto velocity_space =
        FE::spaces::SpaceFactory::create_vector_h1(
            FE::ElementType::Quad4,
            /*order=*/1,
            /*components=*/2);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::Unknown,
    });

    auto options =
        stabilityOptions(interface_marker, std::string(domain_id));
    ns::IncompressibleNavierStokesVMSModule module(
        velocity_space, scalar_space, options);
    module.registerOn(system);
    system.setup({});

    const auto velocity = system.findFieldByName("u");
    const auto pressure = system.findFieldByName("p");
    if (phi == FE::INVALID_FIELD_ID ||
        velocity == FE::INVALID_FIELD_ID ||
        pressure == FE::INVALID_FIELD_ID) {
        throw std::runtime_error(
            "active-cell topology fields were not registered");
    }

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(
            system.dofHandler().getNumDofs()),
        FE::Real{0.0});
    const auto phi_handle = MeshFields::get_field_handle(
        mesh->base(), EntityKind::Vertex, "phi");
    const auto* mesh_phi = MeshFields::field_data_as<real_t>(
        mesh->base(), phi_handle);
    const auto* phi_map =
        system.fieldDofHandler(phi).getEntityDofMap();
    if (mesh_phi == nullptr || phi_map == nullptr) {
        throw std::runtime_error(
            "active-cell topology level-set data is unavailable");
    }
    const auto phi_offset = system.fieldDofOffset(phi);
    for (index_t vertex = 0;
         vertex < static_cast<index_t>(
             mesh->base().n_vertices());
         ++vertex) {
        const auto dofs = phi_map->getVertexDofs(vertex);
        if (dofs.size() != 1u) {
            throw std::runtime_error(
                "active-cell topology level set is not scalar P1");
        }
        solution.at(static_cast<std::size_t>(
            phi_offset + dofs.front())) =
            static_cast<FE::Real>(
                mesh_phi[static_cast<std::size_t>(vertex)]);
    }

    FE::level_set::LevelSetGeneratedInterfaceOptions cut_options;
    cut_options.level_set_field_name = "phi";
    cut_options.domain_id = std::string(domain_id);
    cut_options.requested_interface_marker = interface_marker;
    cut_options.tolerance = FE::Real{1.0e-12};
    cut_options.quadrature_order = 2;
    cut_options.interface_quadrature_order = 2;
    cut_options.volume_quadrature_order = 2;
    FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto generated =
        lifecycle.build(system, cut_options, solution);
    if (!generated.success) {
        throw std::runtime_error(generated.diagnostic);
    }
    auto context =
        std::make_shared<FE::assembly::CutIntegrationContext>();
    context->addGeneratedInterfaceDomain(generated.domain);
    system.setCutIntegrationContext(context);
    system.rebuildConstraintState();

    const auto reports =
        system.completedSmallCutAggregationRefreshReports();
    const auto unique_report_for =
        [&](FE::FieldId field) {
            const auto count = static_cast<std::size_t>(
                std::count_if(
                    reports.begin(),
                    reports.end(),
                    [&](const auto& report) {
                        return report.field == field &&
                               report.interface_marker ==
                                   interface_marker &&
                               report.active_side ==
                                   FE::geometry::
                                       CutIntegrationSide::Negative;
                    }));
            if (count != 1u) {
                throw std::runtime_error(
                    "active-cell topology aggregation report is not unique");
            }
            return *std::find_if(
                reports.begin(),
                reports.end(),
                [&](const auto& report) {
                    return report.field == field &&
                           report.interface_marker ==
                               interface_marker &&
                           report.active_side ==
                               FE::geometry::
                                   CutIntegrationSide::Negative;
                });
        };

    return ActiveCellTopologySample{
        .velocity = velocity,
        .pressure = pressure,
        .velocity_report = unique_report_for(velocity),
        .pressure_report = unique_report_for(pressure),
        .assembled_active_physical_volume =
            assemblePhysicalActiveVolume(
                system,
                phi,
                *scalar_space,
                *context,
                interface_marker),
    };
}

struct ManufacturedAffineBalanceSample {
    FE::Real unconstrained_residual_norm{0.0};
    FE::Real repeated_residual_difference_norm{0.0};
    FE::Real physical_active_area{0.0};
    FE::Real expected_active_area{0.0};
    FE::Real maximum_q1_mixed_coefficient{0.0};
    FE::Real maximum_interface_normal_error{0.0};
    FE::Real maximum_contact_cosine_error{0.0};
    std::size_t interface_fragments{0u};
    std::size_t contact_fragments{0u};
    std::size_t free_velocity_dofs{0u};
};

/**
 * Assemble a manufactured, stationary sharp-interface state through the full
 * production Navier--Stokes/free-surface forms.  The interface is a globally
 * affine Q1 field, so LinearCorner geometry and the operator normal coincide
 * exactly.  A prescribed constant curvature is intentional here: this is an
 * algebraic Young--Laplace/contact-angle balance regression, not a curvature
 * recovery or physical sessile-drop test.
 */
[[nodiscard]] ManufacturedAffineBalanceSample
runManufacturedAffineQ1Balance(FE::Real geometry_angle,
                               FE::Real target_angle,
                               FE::Real pressure_multiplier)
{
    constexpr int left_marker = 27101;
    constexpr int right_marker = 27102;
    constexpr int bottom_marker = 27103;
    constexpr int top_marker = 27104;
    constexpr int interface_marker = 27105;
    constexpr FE::Real contact_x = FE::Real{0.6};
    constexpr FE::Real external_pressure = FE::Real{0.031};
    constexpr FE::Real surface_tension = FE::Real{0.072};
    constexpr FE::Real curvature = FE::Real{1.7};
    constexpr std::string_view domain_id =
        "manufactured_affine_q1_young_laplace_balance";

    const PlaneCutPosition cut{
        .label = "manufactured_affine_q1",
        .normal = {{std::sin(geometry_angle),
                    std::cos(geometry_angle),
                    FE::Real{0.0}}},
        .offset = std::sin(geometry_angle) * contact_x,
    };
    auto mesh = makeManufacturedOpenTankQuadMesh(
        cut, left_marker, right_marker, bottom_marker, top_marker);
    auto scalar_space = FE::spaces::SpaceFactory::create_h1(
        FE::ElementType::Quad4, /*order=*/1);
    auto velocity_space = FE::spaces::SpaceFactory::create_vector_h1(
        FE::ElementType::Quad4, /*order=*/1, /*components=*/2);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::Unknown,
    });
    system.addOperator("equations");
    const auto phi_state =
        FE::forms::StateField(phi, *scalar_space, "phi_balance_owner");
    const auto eta =
        FE::forms::TestField(phi, *scalar_space, "eta_balance_owner");
    (void)FE::systems::installFormulation(
        system,
        "equations",
        {phi},
        (FE::forms::dt(phi_state) * eta).dx());

    ns::IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = "u";
    options.pressure_field_name = "p";
    options.density = FE::Real{1.0};
    options.viscosity = FE::Real{0.01};
    options.enable_convection = false;
    options.enable_vms = true;
    for (const int marker : {left_marker, right_marker, top_marker}) {
        options.velocity_dirichlet.push_back(
            ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
                .boundary_marker = marker,
                .value = {FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}},
                .active_components = {true, true, false},
            });
    }
    // Only the wall-normal component is essential on the wetted wall; its
    // tangential trace remains free and therefore exercises the contact law.
    options.velocity_dirichlet.push_back(
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = bottom_marker,
            .value = {FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}},
            .active_components = {false, true, false},
        });

    ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary free_surface;
    free_surface.implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet;
    free_surface.interface_marker = interface_marker;
    free_surface.level_set_field_name = "phi";
    free_surface.generated_interface_domain_id = std::string(domain_id);
    free_surface.level_set_isovalue = FE::Real{0.0};
    free_surface.active_domain =
        ns::FreeSurfaceActiveDomain::LevelSetNegative;
    free_surface.active_domain_method =
        ns::FreeSurfaceActiveDomainMethod::CutVolume;
    free_surface.external_pressure = external_pressure;
    free_surface.surface_tension = surface_tension;
    // This fixture intentionally verifies the legacy prescribed-curvature
    // Young--Laplace algebra.  The unfitted Automatic default is now the
    // variational SurfaceStress form, which correctly ignores this supplied
    // scalar curvature and has a different discrete-equilibrium contract.
    free_surface.surface_tension_form =
        ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction;
    free_surface.curvature = curvature;
    free_surface.use_level_set_curvature = false;
    free_surface.cut_cell_stabilization.enabled = false;
    free_surface.small_cut_aggregation = false;
    ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine contact;
    contact.configuration =
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine::
            DynamicRenE{
                .wall_boundary_marker = bottom_marker,
                .contact_line_marker = svmp::INVALID_LABEL,
                .equilibrium_contact_angle_radians = target_angle,
                .wall_normal = {
                    FE::Real{0.0}, FE::Real{-1.0}, FE::Real{0.0}},
                .mobility = FE::Real{0.5},
                .slip_length = FE::Real{0.2},
            };
    free_surface.contact_lines.push_back(contact);
    options.free_surface.push_back(std::move(free_surface));

    ns::IncompressibleNavierStokesVMSModule module(
        velocity_space, scalar_space, options);
    module.registerOn(system);
    system.setup({});

    const auto velocity = system.findFieldByName("u");
    const auto pressure = system.findFieldByName("p");
    if (velocity == FE::INVALID_FIELD_ID || pressure == FE::INVALID_FIELD_ID) {
        throw std::runtime_error(
            "manufactured Navier--Stokes fields were not registered");
    }
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        FE::Real{0.0});
    setScalarVertexField(solution, system, phi, cut);
    const auto laplace_pressure =
        external_pressure + surface_tension * curvature;
    setConstantScalarVertexField(
        solution,
        system,
        pressure,
        pressure_multiplier * laplace_pressure);
    const std::vector<FE::Real> previous = solution;

    FE::level_set::LevelSetGeneratedInterfaceOptions cut_options;
    cut_options.level_set_field_name = "phi";
    cut_options.domain_id = std::string(domain_id);
    cut_options.requested_interface_marker = interface_marker;
    cut_options.tolerance = FE::Real{1.0e-12};
    cut_options.quadrature_order = 2;
    cut_options.interface_quadrature_order = 2;
    cut_options.volume_quadrature_order = 2;
    FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto generated = lifecycle.build(system, cut_options, solution);
    if (!generated.success) {
        throw std::runtime_error(generated.diagnostic);
    }

    FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey marker_key;
    marker_key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    marker_key.domain_id = std::string(domain_id);
    marker_key.isovalue = FE::Real{0.0};
    marker_key.interface_marker = interface_marker;
    marker_key.boundary_marker = bottom_marker;
    const int contact_marker =
        FE::interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(
            marker_key);

    const auto* phi_entity_map =
        system.fieldDofHandler(phi).getEntityDofMap();
    if (phi_entity_map == nullptr) {
        throw std::runtime_error(
            "manufactured level-set field has no entity map");
    }
    const auto phi_offset = system.fieldDofOffset(phi);
    const auto& interface_request = generated.domain.request();
    FE::interfaces::GeneratedInterfaceBoundaryIntersectionRequest
        contact_request;
    contact_request.source = interface_request.source;
    contact_request.generated_domain_id = std::string(domain_id);
    contact_request.isovalue = interface_request.isovalue;
    contact_request.interface_marker = interface_marker;
    contact_request.boundary_marker = bottom_marker;
    contact_request.intersection_marker = contact_marker;
    contact_request.tolerance = interface_request.tolerance;
    contact_request.quadrature_order = 2;
    contact_request.frame = interface_request.frame;
    contact_request.mesh_geometry_revision =
        interface_request.mesh_geometry_revision;
    contact_request.mesh_topology_revision =
        interface_request.mesh_topology_revision;
    contact_request.ownership_revision = interface_request.ownership_revision;
    contact_request.quadrature_policy_key =
        interface_request.quadrature_policy_key;
    contact_request.source_value_revision = generated.value_revision;
    const auto contact_domain =
        FE::interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
            std::move(contact_request),
            generated.domain,
            system.meshAccess());

    const auto make_active_boundary_request =
        [&](FE::geometry::CutIntegrationSide side) {
            FE::interfaces::GeneratedActiveBoundaryRequest request;
            request.source = interface_request.source;
            request.generated_domain_id = std::string(domain_id);
            request.isovalue = interface_request.isovalue;
            request.interface_marker = interface_marker;
            request.boundary_marker = bottom_marker;
            request.side = side;
            request.tolerance = interface_request.tolerance;
            request.quadrature_order = 2;
            request.frame = interface_request.frame;
            request.mesh_geometry_revision =
                interface_request.mesh_geometry_revision;
            request.mesh_topology_revision =
                interface_request.mesh_topology_revision;
            request.ownership_revision = interface_request.ownership_revision;
            request.quadrature_policy_key =
                interface_request.quadrature_policy_key;
            request.source_value_revision = generated.value_revision;
            return request;
        };
    FE::interfaces::GeneratedActiveBoundaryScalarField scalar_field;
    scalar_field.value_at_node = [&](FE::GlobalIndex vertex) {
        const auto dofs = phi_entity_map->getVertexDofs(vertex);
        if (dofs.empty()) {
            throw std::runtime_error(
                "manufactured level-set vertex has no scalar DOF");
        }
        return solution.at(static_cast<std::size_t>(
            phi_offset + dofs.front()));
    };
    const auto negative_active_boundary =
        FE::interfaces::buildGeneratedActiveBoundaryDomain(
            make_active_boundary_request(
                FE::geometry::CutIntegrationSide::Negative),
            generated.domain,
            contact_domain,
            system.meshAccess(),
            scalar_field);
    const auto positive_active_boundary =
        FE::interfaces::buildGeneratedActiveBoundaryDomain(
            make_active_boundary_request(
                FE::geometry::CutIntegrationSide::Positive),
            generated.domain,
            contact_domain,
            system.meshAccess(),
            scalar_field);
    const auto active_boundary_partition =
        FE::interfaces::validateGeneratedActiveBoundaryPartition(
            negative_active_boundary,
            positive_active_boundary,
            generated.domain,
            contact_domain,
            system.meshAccess());
    if (active_boundary_partition.orphan_source_reference_count != 0u ||
        active_boundary_partition.stale_revision_count != 0u) {
        throw std::runtime_error(
            "manufactured sharp active-boundary partition is inconsistent");
    }

    auto context = std::make_shared<FE::assembly::CutIntegrationContext>();
    context->addGeneratedInterfaceDomain(
        generated.domain, FE::geometry::CutIntegrationSide::Negative);
    context->addGeneratedInterfaceBoundaryIntersectionDomain(contact_domain);
    context->addGeneratedActiveBoundaryDomain(negative_active_boundary);
    context->addGeneratedActiveBoundaryDomain(positive_active_boundary);
    system.setCutIntegrationContext(context);
    system.rebuildConstraintState();

    FE::systems::SystemStateView state;
    state.dt = FE::Real{0.25};
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;
    const auto residual = assembleOperatorResidual(system, state, "equations");
    const auto repeated = assembleOperatorResidual(system, state, "equations");

    ManufacturedAffineBalanceSample sample;
    sample.unconstrained_residual_norm =
        unconstrainedResidualNorm(system, residual);
    std::vector<FE::Real> residual_difference(residual.size(), FE::Real{0.0});
    for (std::size_t i = 0; i < residual.size(); ++i) {
        residual_difference[i] = repeated[i] - residual[i];
    }
    sample.repeated_residual_difference_norm =
        unconstrainedResidualNorm(system, residual_difference);
    sample.physical_active_area = assemblePhysicalActiveVolume(
        system, pressure, *scalar_space, *context, interface_marker);
    sample.expected_active_area = FE::Real{2.0} *
        (FE::Real{1.0} + contact_x -
         std::cos(geometry_angle) / std::sin(geometry_angle));

    constexpr std::array<std::array<FE::GlobalIndex, 4>, 4> quad_vertices = {{
        {{0, 1, 4, 3}},
        {{1, 2, 5, 4}},
        {{3, 4, 7, 6}},
        {{4, 5, 8, 7}},
    }};
    const auto nodal_phi = [&](FE::GlobalIndex vertex) {
        const auto dofs = phi_entity_map->getVertexDofs(vertex);
        return solution.at(static_cast<std::size_t>(
            phi_offset + dofs.front()));
    };
    for (const auto& vertices : quad_vertices) {
        const auto mixed = nodal_phi(vertices[0]) - nodal_phi(vertices[1]) +
                           nodal_phi(vertices[2]) - nodal_phi(vertices[3]);
        sample.maximum_q1_mixed_coefficient = std::max(
            sample.maximum_q1_mixed_coefficient, std::abs(mixed));
    }

    const auto normal_error = [&](const std::array<FE::Real, 3>& normal) {
        const auto dx = normal[0] - cut.normal[0];
        const auto dy = normal[1] - cut.normal[1];
        const auto dz = normal[2] - cut.normal[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    for (const auto& fragment : generated.domain.fragments()) {
        if (!fragment.active()) {
            continue;
        }
        ++sample.interface_fragments;
        sample.maximum_interface_normal_error = std::max(
            sample.maximum_interface_normal_error,
            normal_error(fragment.normal));
    }
    const auto contact_summary = contact_domain.summary();
    sample.contact_fragments = contact_summary.active_fragment_count;
    for (const auto& fragment : contact_domain.fragments()) {
        if (!fragment.active()) {
            continue;
        }
        for (const auto& point : fragment.quadrature_points) {
            const auto dynamic_cosine = point.interface_normal[1];
            sample.maximum_contact_cosine_error = std::max(
                sample.maximum_contact_cosine_error,
                std::abs(dynamic_cosine - std::cos(target_angle)));
        }
    }
    sample.free_velocity_dofs = freeFieldDofs(system, velocity).size();
    return sample;
}

class PersistentStabilityProblem {
public:
    explicit PersistentStabilityProblem(const PlaneCutPosition& initial_cut,
                                        int cells_per_axis = 2,
                                        StabilityRegime regime = {})
        : mesh_(makeFixedTetraMesh(initial_cut, cells_per_axis)),
          velocity_space_(FE::spaces::SpaceFactory::create_vector_h1(
              FE::ElementType::Tetra4, /*order=*/1, /*components=*/3)),
          pressure_space_(FE::spaces::SpaceFactory::create_h1(
              FE::ElementType::Tetra4, /*order=*/1)),
          system_(mesh_),
          cells_per_axis_(cells_per_axis),
          mesh_spacing_(FE::Real{2.0} /
                        static_cast<FE::Real>(cells_per_axis)),
          regime_(regime)
    {
        // Keep phi as an unknown only so the production lifecycle receives its
        // normal coefficient span.  It owns no residual, and is intentionally
        // excluded from the mixed stability matrix below.
        phi_ = system_.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = pressure_space_,
            .components = 1,
            .source_kind = FE::systems::FieldSourceKind::Unknown,
        });

        auto options = stabilityOptions(
            interface_marker, std::string(domain_id), regime_);

        ns::IncompressibleNavierStokesVMSModule module(
            velocity_space_, pressure_space_, options);
        module.registerOn(system_);
        FE::systems::SetupOptions setup;
#if defined(FE_HAS_MPI) && FE_HAS_MPI
        setup.dof_options.my_rank = 0;
        setup.dof_options.world_size = 1;
        setup.dof_options.mpi_comm = MPI_COMM_SELF;
#endif
        system_.setup(setup);

        velocity_ = system_.findFieldByName("u");
        pressure_ = system_.findFieldByName("p");
        if (phi_ == FE::INVALID_FIELD_ID ||
            velocity_ == FE::INVALID_FIELD_ID ||
            pressure_ == FE::INVALID_FIELD_ID) {
            throw std::runtime_error(
                "fixed-sweep Navier--Stokes fields were not registered");
        }

        solution_.assign(
            static_cast<std::size_t>(system_.dofHandler().getNumDofs()), 0.0);
        const auto* velocity_map =
            system_.fieldDofHandler(velocity_).getEntityDofMap();
        if (velocity_map == nullptr) {
            throw std::runtime_error(
                "fixed-sweep velocity field has no entity map");
        }
        const auto velocity_offset = system_.fieldDofOffset(velocity_);
        for (FE::GlobalIndex vertex = 0;
             vertex < system_.meshAccess().numVertices();
             ++vertex) {
            const auto dofs = velocity_map->getVertexDofs(vertex);
            if (dofs.size() != 3u) {
                throw std::runtime_error(
                    "fixed-sweep velocity field is not vector P1");
            }
            solution_.at(static_cast<std::size_t>(
                velocity_offset + dofs[0])) = regime_.advective_speed;
        }
        previous_ = solution_;
    }

    [[nodiscard]] StabilitySample evaluate(
        const PlaneCutPosition& cut,
        std::optional<FE::MeshIndex> designated_parent_cell =
            std::nullopt)
    {
        setScalarVertexField(solution_, system_, phi_, cut);
        // Post-setup active-side and aggregation constraint rebuilds consume
        // the attached mesh field, while cut generation consumes the System
        // coefficient vector.  Keep both production views synchronized.
        setMeshVertexField(*mesh_, cut);
        if (!has_previous_sample_) {
            previous_ = solution_;
        }

        FE::level_set::LevelSetGeneratedInterfaceOptions cut_options;
        cut_options.level_set_field_name = "phi";
        cut_options.domain_id = std::string(domain_id);
        cut_options.requested_interface_marker = interface_marker;
        cut_options.tolerance = 1.0e-12;
        cut_options.quadrature_order = 2;
        cut_options.interface_quadrature_order = 1;
        cut_options.volume_quadrature_order = 2;
        const auto generated = lifecycle_.build(
            system_, cut_options, solution_);
        if (!generated.success) {
            throw std::runtime_error(generated.diagnostic);
        }

        auto context = std::make_shared<FE::assembly::CutIntegrationContext>();
        context->addGeneratedInterfaceDomain(generated.domain);
        const auto facet_handle = addProductionFacetSet(
            *context, generated.domain, system_.meshAccess());
        system_.setCutIntegrationContext(context);
        system_.rebuildConstraintState();
        const auto pressure_anchor = pressureAnchorState(system_, pressure_);

        FE::systems::SystemStateView state;
        state.dt = regime_.dt;
        state.u = std::span<const FE::Real>(solution_);
        state.u_prev = std::span<const FE::Real>(previous_);
        const FE::systems::BackwardDifferenceIntegrator integrator;
        const auto time_context =
            integrator.buildContext(/*max_time_derivative_order=*/1, state);
        state.time_integration = &time_context;

        const auto jacobian =
            assembleOperatorMatrix(system_, state, "equations");
        const auto ghost = assembleOperatorMatrix(
            system_, state,
            "equations_diagnostic_ns_pressure_ghost_penalty");
        const auto pspg = assembleOperatorMatrix(
            system_, state,
            "equations_diagnostic_ns_vms_pspg_pressure_gradient");
        const auto galerkin_continuity = assembleOperatorMatrix(
            system_, state,
            "equations_diagnostic_ns_galerkin_continuity");

        auto free_velocity =
            canonicalFreeP1Dofs(*mesh_, system_, velocity_, 3u);
        auto free_pressure =
            canonicalFreeP1Dofs(*mesh_, system_, pressure_, 1u);
        std::vector<FE::GlobalIndex> free_mixed = free_velocity;
        free_mixed.insert(
            free_mixed.end(), free_pressure.begin(), free_pressure.end());
        if (free_mixed.empty() || free_pressure.empty()) {
            throw std::runtime_error("fixed-sweep mixed free space is empty");
        }
        const auto raw_pressure_mass = assembleRawActivePressureMass(
            system_,
            pressure_,
            *pressure_space_,
            *context,
            interface_marker);
        const auto reduced_pressure_mass = reduceFieldMatrixByConstraints(
            raw_pressure_mass, system_, pressure_, free_pressure);
        const auto pressure_control = pressureControlMetrics(
            jacobian,
            galerkin_continuity,
            ghost,
            pspg,
            reduced_pressure_mass,
            free_velocity,
            free_pressure);

        auto reduced = extractReducedMatrix(jacobian, free_mixed);
        const auto canonical_mixed_operator = reduced;
        const auto canonical_pressure_ghost_operator =
            extractReducedMatrix(ghost, free_pressure);
        const auto canonical_pressure_pspg_operator =
            extractReducedMatrix(pspg, free_pressure);
        const auto reduced_max = FE::math::dense_matrix_max_abs(reduced);
        const auto zero_row_tolerance =
            std::max(FE::Real{1.0}, reduced_max) * FE::Real{1.0e-12};
        std::size_t zero_pressure_rows = 0u;
        for (std::size_t local = 0; local < free_pressure.size(); ++local) {
            const auto row = free_velocity.size() + local;
            FE::Real row_norm = 0.0;
            for (std::size_t column = 0; column < free_mixed.size(); ++column) {
                const auto value = reduced[row * free_mixed.size() + column];
                row_norm += value * value;
            }
            if (std::sqrt(row_norm) <= zero_row_tolerance) {
                ++zero_pressure_rows;
            }
        }

        equilibrate(reduced, free_mixed.size());
        const auto diagnostics = FE::math::dense_matrix_diagnostics(
            reduced,
            free_mixed.size(),
            free_mixed.size(),
            "equilibrated free-surface mixed Jacobian");

        StabilitySample sample;
        sample.label = cut.label;
        sample.reference_active_volume =
            generated.summary.negative_volume_measure;
        sample.physical_active_volume = assemblePhysicalActiveVolume(
            system_, pressure_, *pressure_space_, *context, interface_marker);
        sample.cut_cells = generated.domain.cutCells().size();
        sample.cut_adjacent_facets = facet_handle.facets.size();
        sample.pruned_volume_rules = context->generatedPrunedVolumeRuleCount();
        sample.backend_volume_quadrature_points =
            generated.backend_volume_quadrature_point_count;
        sample.backend_fallback_cells =
            generated.implicit_cut_fallback_cell_count;
        sample.pressure_natural_traction_anchor =
            pressure_anchor.natural_traction_anchor;
        sample.pressure_anchor_has_no_gauge_enforcement =
            pressure_anchor.no_gauge_enforcement;
        for (const auto& region : generated.domain.volumeRegions()) {
            if (!region.active() ||
                region.side != FE::geometry::CutIntegrationSide::Negative ||
                region.full_cell_equivalent ||
                !(region.volume_fraction > FE::Real{0.0}) ||
                !(region.volume_fraction < FE::Real{1.0})) {
                continue;
            }
            sample.minimum_active_cut_fraction = std::min(
                sample.minimum_active_cut_fraction, region.volume_fraction);
            if (designated_parent_cell.has_value() &&
                region.parent_cell == designated_parent_cell.value()) {
                sample.designated_cut_fraction = region.volume_fraction;
            }
        }
        sample.velocity_constraints = countFieldConstraints(system_, velocity_);
        sample.pressure_constraints = countFieldConstraints(system_, pressure_);
        sample.pressure_aggregation = aggregationConstraintMetrics(
            system_, pressure_, mesh_spacing_);
        sample.pressure_control = pressure_control;
        sample.mesh_cells_per_axis = cells_per_axis_;
        sample.mesh_spacing = mesh_spacing_;
        sample.free_velocity_dofs = free_velocity.size();
        sample.free_pressure_dofs = free_pressure.size();
        sample.zero_free_pressure_rows = zero_pressure_rows;
        sample.pressure_ghost_norm =
            selectedFrobeniusNorm(ghost, free_pressure, free_pressure);
        sample.pspg_pressure_gradient_norm =
            selectedFrobeniusNorm(pspg, free_pressure, free_pressure);
        sample.canonical_mixed_operator = canonical_mixed_operator;
        sample.canonical_pressure_ghost_operator =
            canonical_pressure_ghost_operator;
        sample.canonical_pressure_pspg_operator =
            canonical_pressure_pspg_operator;
        sample.equilibrated_rank = diagnostics.rank;
        sample.equilibrated_size = free_mixed.size();
        sample.equilibrated_smallest_singular_value =
            diagnostics.smallest_retained_singular_value;
        sample.equilibrated_largest_singular_value =
            diagnostics.largest_singular_value;
        sample.equilibrated_condition_inf =
            infinityNormCondition(reduced, free_mixed.size());
        sample.krylov = runEquilibratedJacobiBicgstab(
            reduced, free_mixed.size());

        previous_ = solution_;
        has_previous_sample_ = true;
        return sample;
    }

private:
    static constexpr int interface_marker = 27013;
    static constexpr std::string_view domain_id =
        "fs14_fixed_tetra_sweep";

    std::shared_ptr<Mesh> mesh_{};
    std::shared_ptr<FE::spaces::ProductSpace> velocity_space_{};
    std::shared_ptr<FE::spaces::H1Space> pressure_space_{};
    FE::systems::FESystem system_;
    FE::FieldId phi_{FE::INVALID_FIELD_ID};
    FE::FieldId velocity_{FE::INVALID_FIELD_ID};
    FE::FieldId pressure_{FE::INVALID_FIELD_ID};
    std::vector<FE::Real> solution_{};
    std::vector<FE::Real> previous_{};
    FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle_{};
    bool has_previous_sample_{false};
    int cells_per_axis_{2};
    FE::Real mesh_spacing_{1.0};
    StabilityRegime regime_{};
};

struct NitscheEnergyOrientation {
    std::string_view id;
    std::array<FE::Real, 3> raw_normal;
};

[[nodiscard]] FE::Real unitRightTriangleLinearCdf(
    FE::Real value,
    FE::Real second_span)
{
    if (!(second_span >= FE::Real{0.0}) ||
        !std::isfinite(second_span) ||
        !std::isfinite(value)) {
        throw std::invalid_argument(
            "triangle linear CDF requires finite nonnegative span and value");
    }
    if (value <= FE::Real{0.0}) {
        return FE::Real{0.0};
    }
    if (value >= FE::Real{1.0} + second_span) {
        return FE::Real{1.0};
    }

    using Point2 = std::array<FE::Real, 2>;
    std::vector<Point2> polygon{{
        {{0.0, 0.0}},
        {{1.0, 0.0}},
        {{1.0, 1.0}},
    }};
    std::vector<Point2> clipped;
    clipped.reserve(4u);
    auto previous = polygon.back();
    auto previous_level =
        previous[0] + second_span * previous[1] - value;
    bool previous_inside = previous_level <= FE::Real{0.0};
    for (const auto& current : polygon) {
        const auto current_level =
            current[0] + second_span * current[1] - value;
        const bool current_inside =
            current_level <= FE::Real{0.0};
        const auto append_intersection = [&] {
            const auto denominator =
                previous_level - current_level;
            const auto t =
                std::abs(denominator) >
                        std::numeric_limits<FE::Real>::min()
                    ? std::clamp(
                          previous_level / denominator,
                          FE::Real{0.0},
                          FE::Real{1.0})
                    : FE::Real{0.5};
            clipped.push_back({{
                (FE::Real{1.0} - t) * previous[0] +
                    t * current[0],
                (FE::Real{1.0} - t) * previous[1] +
                    t * current[1],
            }});
        };
        if (previous_inside && current_inside) {
            clipped.push_back(current);
        } else if (previous_inside && !current_inside) {
            append_intersection();
        } else if (!previous_inside && current_inside) {
            append_intersection();
            clipped.push_back(current);
        }
        previous = current;
        previous_level = current_level;
        previous_inside = current_inside;
    }
    if (clipped.size() < 3u) {
        return FE::Real{0.0};
    }
    FE::Real twice_signed_area = FE::Real{0.0};
    for (std::size_t i = 0u; i < clipped.size(); ++i) {
        const auto& a = clipped[i];
        const auto& b = clipped[(i + 1u) % clipped.size()];
        twice_signed_area += a[0] * b[1] - b[0] * a[1];
    }
    // The parent triangle area is 1/2, so its normalized fraction equals
    // the absolute shoelace sum.
    return std::clamp(
        std::abs(twice_signed_area),
        FE::Real{0.0},
        FE::Real{1.0});
}

[[nodiscard]] FE::Real inverseUnitRightTriangleLinearCdf(
    FE::Real fraction,
    FE::Real second_span)
{
    if (!(fraction >= FE::Real{0.0}) ||
        !(fraction <= FE::Real{1.0}) ||
        !std::isfinite(fraction)) {
        throw std::invalid_argument(
            "triangle linear inverse requires a fraction in [0,1]");
    }
    FE::Real lower = FE::Real{0.0};
    FE::Real upper = FE::Real{1.0} + second_span;
    for (int iteration = 0; iteration < 100; ++iteration) {
        const auto midpoint = FE::Real{0.5} * (lower + upper);
        if (unitRightTriangleLinearCdf(
                midpoint, second_span) < fraction) {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    return FE::Real{0.5} * (lower + upper);
}

[[nodiscard]] PlaneCutPosition nitscheEnergyStripCut(
    FE::Real active_wall_fraction,
    FE::Real mesh_scale,
    const NitscheEnergyOrientation& orientation,
    FE::geometry::CutIntegrationSide active_side)
{
    if (!(active_wall_fraction >= FE::Real{0.0}) ||
        !(active_wall_fraction <= FE::Real{1.0}) ||
        !(mesh_scale > FE::Real{0.0}) ||
        !std::isfinite(active_wall_fraction) ||
        !std::isfinite(mesh_scale)) {
        throw std::invalid_argument(
            "Nitsche energy strip cut has invalid fraction or scale");
    }
    const auto raw_norm = std::sqrt(
        orientation.raw_normal[0] * orientation.raw_normal[0] +
        orientation.raw_normal[1] * orientation.raw_normal[1] +
        orientation.raw_normal[2] * orientation.raw_normal[2]);
    if (!(raw_norm > FE::Real{0.0}) ||
        !std::isfinite(raw_norm) ||
        std::abs(
            orientation.raw_normal[0] - FE::Real{1.0}) >
            FE::Real{32.0} *
                std::numeric_limits<FE::Real>::epsilon() ||
        !(orientation.raw_normal[1] >= FE::Real{0.0}) ||
        !(orientation.raw_normal[2] >= FE::Real{0.0})) {
        throw std::invalid_argument(
            "Nitsche energy orientation must have unit x coefficient and nonnegative transverse coefficients");
    }

    FE::Real projected_patch_coordinate = FE::Real{0.0};
    if (active_wall_fraction == FE::Real{0.0}) {
        projected_patch_coordinate = FE::Real{-0.05};
    } else if (active_wall_fraction == FE::Real{1.0}) {
        projected_patch_coordinate =
            FE::Real{1.0} + orientation.raw_normal[1] +
            FE::Real{0.05};
    } else {
        projected_patch_coordinate =
            inverseUnitRightTriangleLinearCdf(
                active_wall_fraction,
                orientation.raw_normal[1]);
    }
    const auto physical_offset =
        mesh_scale *
        (FE::Real{3.0} * orientation.raw_normal[0] +
         projected_patch_coordinate);

    std::array<FE::Real, 3> unit_normal{{
        orientation.raw_normal[0] / raw_norm,
        orientation.raw_normal[1] / raw_norm,
        orientation.raw_normal[2] / raw_norm,
    }};
    FE::Real unit_offset = physical_offset / raw_norm;
    if (active_side ==
        FE::geometry::CutIntegrationSide::Positive) {
        for (auto& component : unit_normal) {
            component = -component;
        }
        unit_offset = -unit_offset;
    }
    return PlaneCutPosition{
        .label =
            std::string(orientation.id) + "_" +
            (active_side ==
                     FE::geometry::CutIntegrationSide::Negative
                 ? "negative"
                 : "positive") +
            "_" + realPropertyValue(active_wall_fraction),
        .normal = unit_normal,
        .offset = unit_offset,
    };
}

struct NitscheEnergyGeneratedContext {
    std::shared_ptr<FE::assembly::CutIntegrationContext> context{};
    int selected_boundary_marker{-1};
    FE::Real active_boundary_measure{0.0};
    FE::Real parent_boundary_measure{0.0};
    std::size_t active_rule_count{0u};
    std::size_t implicit_backend_fallback_count{0u};
};

[[nodiscard]] NitscheEnergyGeneratedContext
makeNitscheEnergyGeneratedContext(
    FE::systems::FESystem& system,
    FE::FieldId level_set,
    int interface_marker,
    int wall_marker,
    std::string_view domain_id,
    FE::geometry::CutIntegrationSide active_side,
    const FE::level_set::LevelSetGeneratedInterfaceResult& generated,
    std::span<const FE::Real> solution)
{
    const auto* entity_map =
        system.fieldDofHandler(level_set).getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error(
            "Nitsche energy level-set field has no entity map");
    }
    const auto field_offset =
        system.fieldDofOffset(level_set);
    const auto& interface_request = generated.domain.request();

    FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey
        contact_key;
    contact_key.source = interface_request.source;
    contact_key.domain_id = std::string(domain_id);
    contact_key.isovalue = interface_request.isovalue;
    contact_key.interface_marker = interface_marker;
    contact_key.boundary_marker = wall_marker;
    const int contact_marker =
        FE::interfaces::
            stableGeneratedInterfaceBoundaryIntersectionMarker(
                contact_key);

    FE::interfaces::GeneratedInterfaceBoundaryIntersectionRequest
        contact_request;
    contact_request.source = interface_request.source;
    contact_request.generated_domain_id = std::string(domain_id);
    contact_request.isovalue = interface_request.isovalue;
    contact_request.interface_marker = interface_marker;
    contact_request.boundary_marker = wall_marker;
    contact_request.intersection_marker = contact_marker;
    contact_request.tolerance = interface_request.tolerance;
    contact_request.quadrature_order = 2;
    contact_request.frame = interface_request.frame;
    contact_request.mesh_geometry_revision =
        interface_request.mesh_geometry_revision;
    contact_request.mesh_topology_revision =
        interface_request.mesh_topology_revision;
    contact_request.ownership_revision =
        interface_request.ownership_revision;
    contact_request.quadrature_policy_key =
        interface_request.quadrature_policy_key;
    contact_request.source_value_revision =
        generated.value_revision;
    const auto contact_domain =
        FE::interfaces::
            buildGeneratedInterfaceBoundaryIntersectionDomain(
                std::move(contact_request),
                generated.domain,
                system.meshAccess());

    const auto active_request =
        [&](FE::geometry::CutIntegrationSide side) {
            FE::interfaces::GeneratedActiveBoundaryRequest request;
            request.source = interface_request.source;
            request.generated_domain_id = std::string(domain_id);
            request.isovalue = interface_request.isovalue;
            request.interface_marker = interface_marker;
            request.boundary_marker = wall_marker;
            request.side = side;
            request.tolerance = interface_request.tolerance;
            request.quadrature_order = 2;
            request.frame = interface_request.frame;
            request.mesh_geometry_revision =
                interface_request.mesh_geometry_revision;
            request.mesh_topology_revision =
                interface_request.mesh_topology_revision;
            request.ownership_revision =
                interface_request.ownership_revision;
            request.quadrature_policy_key =
                interface_request.quadrature_policy_key;
            request.source_value_revision =
                generated.value_revision;
            return request;
        };
    FE::interfaces::GeneratedActiveBoundaryScalarField scalar_field;
    scalar_field.value_at_node =
        [&](FE::GlobalIndex vertex) {
            const auto dofs =
                entity_map->getVertexDofs(vertex);
            if (dofs.size() != 1u) {
                throw std::runtime_error(
                    "Nitsche energy level-set vertex is not scalar P1");
            }
            const auto global_dof = field_offset + dofs.front();
            if (global_dof < 0 ||
                static_cast<std::size_t>(global_dof) >=
                    solution.size()) {
                throw std::runtime_error(
                    "Nitsche energy level-set vertex DOF is out of range");
            }
            return solution[static_cast<std::size_t>(global_dof)];
        };
    const auto negative_active =
        FE::interfaces::buildGeneratedActiveBoundaryDomain(
            active_request(
                FE::geometry::CutIntegrationSide::Negative),
            generated.domain,
            contact_domain,
            system.meshAccess(),
            scalar_field);
    const auto positive_active =
        FE::interfaces::buildGeneratedActiveBoundaryDomain(
            active_request(
                FE::geometry::CutIntegrationSide::Positive),
            generated.domain,
            contact_domain,
            system.meshAccess(),
            scalar_field);
    const auto partition =
        FE::interfaces::validateGeneratedActiveBoundaryPartition(
            negative_active,
            positive_active,
            generated.domain,
            contact_domain,
            system.meshAccess());
    if (partition.orphan_source_reference_count != 0u ||
        partition.stale_revision_count != 0u) {
        throw std::runtime_error(
            "Nitsche energy generated boundary partition is inconsistent");
    }

    const auto& selected =
        active_side ==
                FE::geometry::CutIntegrationSide::Negative
            ? negative_active
            : positive_active;
    NitscheEnergyGeneratedContext result;
    result.selected_boundary_marker = selected.marker();
    result.active_boundary_measure =
        active_side ==
                FE::geometry::CutIntegrationSide::Negative
            ? partition.negative_boundary_measure
            : partition.positive_boundary_measure;
    result.parent_boundary_measure =
        partition.total_boundary_measure;
    for (const auto& fragment : selected.fragments()) {
        if (!fragment.active()) {
            continue;
        }
        result.implicit_backend_fallback_count +=
            static_cast<std::size_t>(
                fragment.represented_implicit_fallback_status !=
                "None");
    }
    result.active_rule_count =
        selected.boundaryQuadratureRules().size();

    result.context =
        std::make_shared<FE::assembly::CutIntegrationContext>();
    // Aggregation requires the complete two-sided cell classification even
    // though each form later selects one configured active side.
    result.context->addGeneratedInterfaceDomain(generated.domain);
    result.context
        ->addGeneratedInterfaceBoundaryIntersectionDomain(
            contact_domain);
    result.context->addGeneratedActiveBoundaryDomain(
        negative_active);
    result.context->addGeneratedActiveBoundaryDomain(
        positive_active);
    return result;
}

struct NitscheEnergySample {
    std::string case_id{};
    FE::Real target_wall_fraction{0.0};
    FE::Real observed_wall_fraction{0.0};
    FE::Real mesh_scale{0.0};
    std::size_t active_rule_count{0u};
    std::size_t implicit_backend_fallback_count{0u};
    std::size_t free_velocity_dofs{0u};
    std::size_t velocity_aggregate_lines{0u};
    std::size_t velocity_homogeneous_constraints{0u};
    std::size_t velocity_gauge_line_count{0u};
    int generated_active_boundary_marker{-1};
    std::array<std::size_t, 4>
        diagnostic_boundary_term_counts{};
    std::array<std::size_t, 4>
        diagnostic_interface_face_term_counts{};
    bool diagnostic_routes_use_generated_marker{false};
    bool generated_active_boundary_marker_registered{false};
    FE::constraints::SmallCutAggregationRefreshReport
        aggregation_report{};
    FE::Real symmetric_operator_relative_skew{0.0};
    FE::Real energy_norm_relative_skew{0.0};
    FE::Real production_reconstruction_relative_error{0.0};
    FE::Real energy_reconstruction_relative_error{0.0};
    FE::Real consistency_boundary_relative_norm{0.0};
    FE::Real penalty_boundary_relative_norm{0.0};
    FE::Real symmetric_boundary_relative_norm{0.0};
    FE::Real minimum_energy_norm_eigenvalue{0.0};
    FE::Real minimum_generalized_eigenvalue{0.0};
    FE::Real maximum_generalized_eigenvalue{0.0};
    FE::Real eigensolver_tolerance{0.0};
    FE::Real eigensolver_maximum_off_diagonal{0.0};
    std::size_t eigensolver_sweeps{0u};
    bool eigensolver_converged{false};
    std::vector<FE::Real> bulk_viscous{};
    std::vector<FE::Real> bulk_plus_consistency{};
    std::vector<FE::Real> symmetric_operator{};
    std::vector<FE::Real> energy_norm{};
};

[[nodiscard]] FE::Real relativeMatrixDifference(
    std::span<const FE::Real> lhs,
    std::span<const FE::Real> rhs)
{
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument(
            "relative matrix difference requires equal sizes");
    }
    FE::Real maximum_difference = 0.0;
    FE::Real maximum_scale = 0.0;
    for (std::size_t index = 0u; index < lhs.size(); ++index) {
        maximum_difference = std::max(
            maximum_difference,
            std::abs(lhs[index] - rhs[index]));
        maximum_scale = std::max(
            maximum_scale,
            std::max(std::abs(lhs[index]), std::abs(rhs[index])));
    }
    return maximum_difference /
           std::max(
               maximum_scale,
               std::numeric_limits<FE::Real>::min());
}

[[nodiscard]] std::vector<FE::Real> addMatrices(
    std::span<const FE::Real> first,
    std::span<const FE::Real> second,
    std::span<const FE::Real> third = {})
{
    if (first.size() != second.size() ||
        (!third.empty() && first.size() != third.size())) {
        throw std::invalid_argument(
            "matrix sum requires equal sizes");
    }
    std::vector<FE::Real> result(
        first.begin(), first.end());
    for (std::size_t index = 0u; index < result.size(); ++index) {
        result[index] += second[index];
        if (!third.empty()) {
            result[index] += third[index];
        }
    }
    return result;
}

[[nodiscard]] std::vector<FE::Real> subtractMatrices(
    std::span<const FE::Real> lhs,
    std::span<const FE::Real> rhs)
{
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument(
            "matrix difference requires equal sizes");
    }
    std::vector<FE::Real> result(lhs.begin(), lhs.end());
    for (std::size_t index = 0u; index < result.size(); ++index) {
        result[index] -= rhs[index];
    }
    return result;
}

constexpr std::size_t
    nitsche_energy_maximum_root_path_length{12u};
constexpr std::size_t
    nitsche_energy_default_maximum_root_path_length{
        FE::constraints::SmallCutAggregationGuardOptions{}
            .maximum_root_path_length};
constexpr FE::Real
    nitsche_energy_maximum_reference_extrapolation_distance{
        4.0};
constexpr FE::Real
    nitsche_energy_maximum_absolute_coefficient{16.0};
constexpr FE::Real
    nitsche_energy_maximum_row_l1_norm{32.0};

class PersistentNitscheEnergyProblem {
public:
    PersistentNitscheEnergyProblem(
        FE::Real mesh_scale,
        NitscheEnergyOrientation orientation,
        FE::geometry::CutIntegrationSide active_side,
        std::size_t maximum_root_path_length =
            nitsche_energy_maximum_root_path_length)
        : mesh_scale_(mesh_scale)
        , orientation_(std::move(orientation))
        , active_side_(active_side)
        , mesh_(makeNitscheEnergyTetraStripMesh(
              nitscheEnergyStripCut(
                  FE::Real{0.25},
                  mesh_scale_,
                  orientation_,
                  active_side_),
              mesh_scale_,
              wall_marker,
              anchor_marker))
        , velocity_space_(
              FE::spaces::SpaceFactory::create_vector_h1(
                  FE::ElementType::Tetra4,
                  /*order=*/1,
                  /*components=*/3))
        , pressure_space_(
              FE::spaces::SpaceFactory::create_h1(
                  FE::ElementType::Tetra4,
                  /*order=*/1))
        , system_(mesh_)
    {
        phi_ = system_.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = pressure_space_,
            .components = 1,
            .source_kind =
                FE::systems::FieldSourceKind::Unknown,
        });

        ns::IncompressibleNavierStokesVMSOptions options;
        options.velocity_field_name = "u";
        options.pressure_field_name = "p";
        options.density = FE::Real{1.0};
        options.viscosity = viscosity;
        options.enable_convection = false;
        options.enable_vms = false;
        options.jit_policy.enable = false;
        options.velocity_dirichlet_weak.push_back(
            ns::IncompressibleNavierStokesVMSOptions::
                VelocityDirichletBC{
                    .boundary_marker = wall_marker,
                    .value = {0.0, 0.0, 0.0},
                });
        options.velocity_dirichlet.push_back(
            ns::IncompressibleNavierStokesVMSOptions::
                VelocityDirichletBC{
                    .boundary_marker = anchor_marker,
                    .value = {0.0, 0.0, 0.0},
                });
        options.nitsche_gamma = nitsche_gamma;
        options.nitsche_symmetric = true;
        options.nitsche_scale_with_p = false;
        options.symmetric_nitsche_energy_qualification_scope =
            ns::SymmetricNitscheEnergyQualificationScope::
                JointLowLevelPrerequisite;
        options.free_surface.push_back(
            ns::IncompressibleNavierStokesVMSOptions::
                FreeSurfaceBoundary{
                    .implementation =
                        ns::FreeSurfaceImplementation::
                            UnfittedLevelSet,
                    .interface_marker = interface_marker,
                    .level_set_field_name = "phi",
                    .generated_interface_domain_id =
                        std::string(domain_id),
                    .level_set_isovalue = FE::Real{0.0},
                    .active_domain =
                        active_side_ ==
                                FE::geometry::
                                    CutIntegrationSide::Negative
                            ? ns::FreeSurfaceActiveDomain::
                                  LevelSetNegative
                            : ns::FreeSurfaceActiveDomain::
                                  LevelSetPositive,
                    .active_domain_method =
                        ns::FreeSurfaceActiveDomainMethod::
                            CutVolume,
                    .external_pressure = FE::Real{0.0},
                    .surface_tension = FE::Real{0.0},
                    .use_level_set_curvature = false,
                    .cut_cell_stabilization = {
                        .enabled = false,
                    },
                    .small_cut_aggregation = true,
                    .small_cut_aggregation_guards = {
                        .maximum_root_path_length =
                            maximum_root_path_length,
                        .maximum_reference_extrapolation_distance =
                            nitsche_energy_maximum_reference_extrapolation_distance,
                        .maximum_absolute_coefficient =
                            nitsche_energy_maximum_absolute_coefficient,
                        .maximum_row_l1_norm =
                            nitsche_energy_maximum_row_l1_norm,
                    },
                });

        ns::IncompressibleNavierStokesVMSModule module(
            velocity_space_, pressure_space_, options);
        module.registerOn(system_);
        FE::systems::SetupOptions setup;
#if defined(FE_HAS_MPI) && FE_HAS_MPI
        setup.dof_options.my_rank = 0;
        setup.dof_options.world_size = 1;
        setup.dof_options.mpi_comm = MPI_COMM_SELF;
#endif
        system_.setup(setup);

        velocity_ = system_.findFieldByName("u");
        if (velocity_ == FE::INVALID_FIELD_ID) {
            throw std::runtime_error(
                "Nitsche energy problem did not register velocity");
        }
        solution_.assign(
            static_cast<std::size_t>(
                system_.dofHandler().getNumDofs()),
            FE::Real{0.0});
        previous_ = solution_;
    }

    [[nodiscard]] NitscheEnergySample evaluate(
        FE::Real active_wall_fraction)
    {
        const auto cut = nitscheEnergyStripCut(
            active_wall_fraction,
            mesh_scale_,
            orientation_,
            active_side_);
        setScalarVertexField(
            solution_, system_, phi_, cut);
        setMeshVertexField(*mesh_, cut);
        previous_ = solution_;

        FE::level_set::LevelSetGeneratedInterfaceOptions
            cut_options;
        cut_options.level_set_field_name = "phi";
        cut_options.domain_id = std::string(domain_id);
        cut_options.requested_interface_marker =
            interface_marker;
        cut_options.tolerance = FE::Real{1.0e-12};
        cut_options.quadrature_order = 2;
        cut_options.interface_quadrature_order = 2;
        cut_options.volume_quadrature_order = 2;
        const auto generated = lifecycle_.build(
            system_, cut_options, solution_);
        if (!generated.success) {
            throw std::runtime_error(generated.diagnostic);
        }
        auto generated_context =
            makeNitscheEnergyGeneratedContext(
                system_,
                phi_,
                interface_marker,
                wall_marker,
                domain_id,
                active_side_,
                generated,
                solution_);
        system_.setCutIntegrationContext(
            generated_context.context);
        system_.rebuildConstraintState();

        constexpr std::array<std::string_view, 4>
            diagnostic_operator_tags = {{
                ns::SymmetricNitscheEnergyDiagnosticOperators::
                    bulk_viscous,
                ns::SymmetricNitscheEnergyDiagnosticOperators::
                    bulk_plus_consistency,
                ns::SymmetricNitscheEnergyDiagnosticOperators::
                    symmetric_operator,
                ns::SymmetricNitscheEnergyDiagnosticOperators::
                    energy_norm,
            }};
        std::array<std::size_t, 4>
            diagnostic_boundary_term_counts{};
        std::array<std::size_t, 4>
            diagnostic_interface_face_term_counts{};
        bool diagnostic_routes_use_generated_marker = true;
        for (std::size_t operator_index = 0u;
             operator_index < diagnostic_operator_tags.size();
             ++operator_index) {
            const auto& definition = system_.operatorDefinition(
                std::string(
                    diagnostic_operator_tags[operator_index]));
            diagnostic_boundary_term_counts[operator_index] =
                definition.boundary.size();
            diagnostic_interface_face_term_counts[operator_index] =
                definition.interface_faces.size();
            if (operator_index > 0u) {
                diagnostic_routes_use_generated_marker =
                    diagnostic_routes_use_generated_marker &&
                    !definition.interface_faces.empty() &&
                    std::all_of(
                        definition.interface_faces.begin(),
                        definition.interface_faces.end(),
                        [&](const auto& term) {
                            return term.marker ==
                                   generated_context
                                       .selected_boundary_marker;
                        });
            }
        }

        FE::systems::SystemStateView state;
        state.dt = FE::Real{1.0};
        state.u = std::span<const FE::Real>(solution_);
        state.u_prev =
            std::span<const FE::Real>(previous_);
        const FE::systems::BackwardDifferenceIntegrator
            integrator;
        const auto time_context =
            integrator.buildContext(1, state);
        state.time_integration = &time_context;

        const auto bulk = assembleOperatorMatrix(
            system_,
            state,
            std::string(
                ns::SymmetricNitscheEnergyDiagnosticOperators::
                    bulk_viscous));
        const auto bulk_plus_consistency = assembleOperatorMatrix(
            system_,
            state,
            std::string(
                ns::SymmetricNitscheEnergyDiagnosticOperators::
                    bulk_plus_consistency));
        const auto production = assembleOperatorMatrix(
            system_,
            state,
            std::string(
                ns::SymmetricNitscheEnergyDiagnosticOperators::
                    symmetric_operator));
        const auto energy = assembleOperatorMatrix(
            system_,
            state,
            std::string(
                ns::SymmetricNitscheEnergyDiagnosticOperators::
                    energy_norm));

        const auto free_velocity = canonicalFreeP1Dofs(
            *mesh_, system_, velocity_, 3u);
        if (free_velocity.empty()) {
            throw std::runtime_error(
                "Nitsche energy aggregate space is empty");
        }
        const auto reduced_bulk =
            extractReducedMatrix(bulk, free_velocity);
        const auto reduced_bulk_plus_consistency =
            extractReducedMatrix(
                bulk_plus_consistency, free_velocity);
        auto reduced_production =
            extractReducedMatrix(
                production, free_velocity);
        auto reduced_energy =
            extractReducedMatrix(energy, free_velocity);
        const auto reduced_consistency = subtractMatrices(
            reduced_bulk_plus_consistency, reduced_bulk);
        const auto reduced_penalty = subtractMatrices(
            reduced_energy, reduced_bulk);
        const auto reconstructed_production = addMatrices(
            reduced_bulk,
            reduced_consistency,
            reduced_penalty);
        const auto reconstructed_energy =
            addMatrices(reduced_bulk, reduced_penalty);

        NitscheEnergySample sample;
        sample.case_id = cut.label;
        sample.target_wall_fraction =
            active_wall_fraction;
        if (!(generated_context.parent_boundary_measure >
              FE::Real{0.0})) {
            throw std::runtime_error(
                "Nitsche energy wall patch has no parent measure");
        }
        sample.observed_wall_fraction =
            generated_context.active_boundary_measure /
            generated_context.parent_boundary_measure;
        sample.mesh_scale = mesh_scale_;
        sample.active_rule_count =
            generated_context.active_rule_count;
        sample.implicit_backend_fallback_count =
            generated_context.implicit_backend_fallback_count;
        sample.free_velocity_dofs =
            free_velocity.size();
        const auto constraint_counts =
            countFieldConstraints(system_, velocity_);
        sample.velocity_aggregate_lines =
            constraint_counts.master_bearing;
        sample.velocity_homogeneous_constraints =
            constraint_counts.homogeneous_pins;
        sample.generated_active_boundary_marker =
            generated_context.selected_boundary_marker;
        sample.diagnostic_boundary_term_counts =
            diagnostic_boundary_term_counts;
        sample.diagnostic_interface_face_term_counts =
            diagnostic_interface_face_term_counts;
        sample.diagnostic_routes_use_generated_marker =
            diagnostic_routes_use_generated_marker;
        sample.generated_active_boundary_marker_registered =
            generated_context.context
                ->hasGeneratedActiveBoundaryMarker(
                    generated_context.selected_boundary_marker);
        sample.production_reconstruction_relative_error =
            relativeMatrixDifference(
                reduced_production,
                reconstructed_production);
        sample.energy_reconstruction_relative_error =
            relativeMatrixDifference(
                reduced_energy, reconstructed_energy);
        sample.consistency_boundary_relative_norm =
            relativeMatrixDifference(
                reduced_bulk_plus_consistency, reduced_bulk);
        sample.penalty_boundary_relative_norm =
            relativeMatrixDifference(
                reduced_energy, reduced_bulk);
        sample.symmetric_boundary_relative_norm =
            relativeMatrixDifference(
                reduced_production, reduced_bulk);
        sample.symmetric_operator_relative_skew =
            relativeMatrixSkew(
                reduced_production, free_velocity.size());
        sample.energy_norm_relative_skew =
            relativeMatrixSkew(
                reduced_energy, free_velocity.size());

        const auto reports =
            system_
                .completedSmallCutAggregationRefreshReports();
        const auto report = std::find_if(
            reports.begin(),
            reports.end(),
            [&](const auto& candidate) {
                return candidate.field == velocity_ &&
                       candidate.interface_marker ==
                           interface_marker &&
                       candidate.active_side == active_side_;
            });
        if (report == reports.end()) {
            throw std::runtime_error(
                "Nitsche energy velocity aggregation report is missing");
        }
        sample.aggregation_report = *report;
        const auto* gauge =
            system_.gaugeRegistryIfPresent();
        if (gauge != nullptr) {
            sample.velocity_gauge_line_count =
                static_cast<std::size_t>(std::count_if(
                    gauge->resolvedModes().begin(),
                    gauge->resolvedModes().end(),
                    [&](const auto& mode) {
                        return mode.candidate.field ==
                                   velocity_ &&
                               mode.policy !=
                                   FE::gauge::
                                       EnforcementPolicy::None;
                    }));
        }

        if (active_wall_fraction == FE::Real{0.0}) {
            sample.bulk_viscous = reduced_bulk;
            sample.bulk_plus_consistency =
                reduced_bulk_plus_consistency;
            sample.symmetric_operator =
                std::move(reduced_production);
            sample.energy_norm =
                std::move(reduced_energy);
            return sample;
        }

        if (sample.symmetric_operator_relative_skew >
                FE::Real{1.0e-11} ||
            sample.energy_norm_relative_skew >
                FE::Real{1.0e-11}) {
            throw std::runtime_error(
                "Nitsche energy operator is not numerically symmetric");
        }
        symmetrize(
            reduced_production, free_velocity.size());
        symmetrize(
            reduced_energy, free_velocity.size());
        const auto energy_spectrum =
            FE::math::dense_symmetric_eigenvalue_bounds(
                reduced_energy,
                free_velocity.size(),
                "Nitsche energy norm");
        sample.minimum_energy_norm_eigenvalue =
            energy_spectrum.smallest_eigenvalue;
        const auto lower = choleskyLower(
            reduced_energy,
            free_velocity.size(),
            "Nitsche energy norm");
        auto normalized = reduced_production;
        leftMultiplyByInverseLower(
            normalized,
            free_velocity.size(),
            free_velocity.size(),
            lower);
        rightMultiplyByInverseLowerTranspose(
            normalized,
            free_velocity.size(),
            free_velocity.size(),
            lower);
        const auto normalized_skew =
            relativeMatrixSkew(
                normalized, free_velocity.size());
        if (normalized_skew > FE::Real{1.0e-10}) {
            throw std::runtime_error(
                "normalized Nitsche operator is not symmetric");
        }
        symmetrize(normalized, free_velocity.size());
        const auto generalized =
            FE::math::dense_symmetric_eigenvalue_bounds(
                normalized,
                free_velocity.size(),
                "energy-normalized symmetric Nitsche operator");
        sample.minimum_generalized_eigenvalue =
            generalized.smallest_eigenvalue;
        sample.maximum_generalized_eigenvalue =
            generalized.largest_eigenvalue;
        sample.eigensolver_tolerance =
            generalized.tolerance;
        sample.eigensolver_maximum_off_diagonal =
            generalized.maximum_off_diagonal;
        sample.eigensolver_sweeps =
            generalized.sweeps;
        sample.eigensolver_converged =
            generalized.converged;
        sample.bulk_viscous = reduced_bulk;
        sample.bulk_plus_consistency =
            reduced_bulk_plus_consistency;
        sample.symmetric_operator =
            std::move(reduced_production);
        sample.energy_norm =
            std::move(reduced_energy);
        return sample;
    }

private:
    static constexpr int interface_marker{27231};
    static constexpr int wall_marker{27232};
    static constexpr int anchor_marker{27233};
    static constexpr std::string_view domain_id =
        "symmetric_nitsche_energy_strip";
    static constexpr FE::Real viscosity{0.01};
    static constexpr FE::Real nitsche_gamma{12.0};

    FE::Real mesh_scale_{1.0};
    NitscheEnergyOrientation orientation_{};
    FE::geometry::CutIntegrationSide active_side_{
        FE::geometry::CutIntegrationSide::Negative};
    std::shared_ptr<Mesh> mesh_{};
    std::shared_ptr<FE::spaces::ProductSpace>
        velocity_space_{};
    std::shared_ptr<FE::spaces::H1Space>
        pressure_space_{};
    FE::systems::FESystem system_;
    FE::FieldId phi_{FE::INVALID_FIELD_ID};
    FE::FieldId velocity_{FE::INVALID_FIELD_ID};
    std::vector<FE::Real> solution_{};
    std::vector<FE::Real> previous_{};
    FE::level_set::LevelSetGeneratedInterfaceLifecycle
        lifecycle_{};
};

#if defined(FE_HAS_MPI) && defined(MESH_HAS_MPI)

[[nodiscard]] MPI_Datatype stabilityMpiRealType()
{
    if constexpr (sizeof(FE::Real) == sizeof(double)) {
        return MPI_DOUBLE;
    }
    if constexpr (sizeof(FE::Real) == sizeof(float)) {
        return MPI_FLOAT;
    }
    return MPI_LONG_DOUBLE;
}

[[nodiscard]] unsigned long long allreduceSumUnsigned(
    unsigned long long local,
    MPI_Comm comm)
{
    unsigned long long global = 0u;
    MPI_Allreduce(
        &local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    return global;
}

[[nodiscard]] FE::assembly::DenseMatrixView globalizeOwnedRows(
    const FE::assembly::DenseMatrixView& local,
    const FE::systems::FESystem& system,
    MPI_Comm comm)
{
    const auto n = local.numRows();
    if (n != local.numCols() || n < 0) {
        throw std::runtime_error(
            "distributed stability matrix is not a valid square matrix");
    }
    const auto count = static_cast<std::size_t>(n) *
                       static_cast<std::size_t>(n);
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "distributed stability matrix exceeds MPI count range");
    }

    std::vector<FE::Real> owned_rows(count, FE::Real{0.0});
    const auto& owned =
        system.dofHandler().getPartition().locallyOwned();
    for (const auto row : owned) {
        if (row < 0 || row >= n) {
            throw std::runtime_error(
                "distributed stability owned row is out of range");
        }
        const auto begin = static_cast<std::size_t>(row) *
                           static_cast<std::size_t>(n);
        std::copy_n(local.data().begin() +
                        static_cast<std::ptrdiff_t>(begin),
                    static_cast<std::size_t>(n),
                    owned_rows.begin() +
                        static_cast<std::ptrdiff_t>(begin));
    }

    std::vector<FE::Real> global(count, FE::Real{0.0});
    MPI_Allreduce(
        owned_rows.data(),
        global.data(),
        static_cast<int>(count),
        stabilityMpiRealType(),
        MPI_SUM,
        comm);
    FE::assembly::DenseMatrixView result(n);
    std::copy(global.begin(), global.end(), result.dataMutable().begin());
    return result;
}

[[nodiscard]] std::vector<FE::Real> globalizeOwnedResidual(
    std::span<const FE::Real> local,
    const FE::systems::FESystem& system,
    MPI_Comm comm)
{
    const auto n = system.dofHandler().getNumDofs();
    if (n < 0 || local.size() != static_cast<std::size_t>(n) ||
        static_cast<unsigned long long>(n) >
            static_cast<unsigned long long>(
                std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "distributed wet-block residual has an invalid size");
    }
    std::vector<FE::Real> owned(local.size(), FE::Real{0.0});
    for (const auto row : system.dofHandler().getPartition().locallyOwned()) {
        if (row < 0 || row >= n) {
            throw std::runtime_error(
                "distributed wet-block owned residual row is out of range");
        }
        owned[static_cast<std::size_t>(row)] =
            local[static_cast<std::size_t>(row)];
    }
    std::vector<FE::Real> global(local.size(), FE::Real{0.0});
    MPI_Allreduce(
        owned.data(),
        global.data(),
        static_cast<int>(n),
        stabilityMpiRealType(),
        MPI_SUM,
        comm);
    return global;
}

[[nodiscard]] std::vector<int> globalConstraintMask(
    const FE::systems::FESystem& system,
    MPI_Comm comm)
{
    const auto n = system.dofHandler().getNumDofs();
    if (n < 0 ||
        static_cast<unsigned long long>(n) >
            static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "distributed stability constraint mask exceeds MPI count range");
    }
    std::vector<int> local(static_cast<std::size_t>(n), 0);
    std::vector<int> global(static_cast<std::size_t>(n), 0);
    const auto& owned =
        system.dofHandler().getPartition().locallyOwned();
    for (const auto dof : owned) {
        if (system.constraints().isConstrained(dof)) {
            local.at(static_cast<std::size_t>(dof)) = 1;
        }
    }
    MPI_Allreduce(
        local.data(), global.data(), static_cast<int>(n), MPI_INT, MPI_MAX, comm);
    return global;
}

[[nodiscard]] std::vector<FE::GlobalIndex> globalFreeFieldDofs(
    const FE::systems::FESystem& system,
    FE::FieldId field,
    std::span<const int> constrained)
{
    std::vector<FE::GlobalIndex> dofs;
    const auto offset = system.fieldDofOffset(field);
    const auto count = system.fieldDofHandler(field).getNumDofs();
    dofs.reserve(static_cast<std::size_t>(count));
    for (FE::GlobalIndex local = 0; local < count; ++local) {
        const auto global = offset + local;
        if (global < 0 ||
            static_cast<std::size_t>(global) >= constrained.size()) {
            throw std::runtime_error(
                "distributed stability field DOF is out of range");
        }
        if (constrained[static_cast<std::size_t>(global)] == 0) {
            dofs.push_back(global);
        }
    }
    return dofs;
}

/** Return unconstrained P1 DOFs in physical vertex-GID/component order.
 *
 * Owner-contiguous global numbering deliberately changes when the cell
 * partition changes.  This fixture retains the complete mesh in overlap, so
 * every rank can independently construct the same physical ordering while
 * retaining the partition-specific global DOF used to index its matrix.
 */
[[nodiscard]] std::vector<FE::GlobalIndex> canonicalGlobalFreeP1Dofs(
    const Mesh& mesh,
    const FE::systems::FESystem& system,
    FE::FieldId field,
    std::size_t components,
    std::span<const int> constrained)
{
    const auto canonical =
        canonicalP1Dofs(mesh, system, field, components);

    std::vector<FE::GlobalIndex> dofs;
    dofs.reserve(canonical.size());
    for (const auto& entry : canonical) {
        if (entry.global_dof < 0 ||
            static_cast<std::size_t>(entry.global_dof) >=
                constrained.size()) {
            throw std::runtime_error(
                "distributed stability vertex DOF is out of range");
        }
        if (constrained[static_cast<std::size_t>(entry.global_dof)] == 0) {
            dofs.push_back(entry.global_dof);
        }
    }
    return dofs;
}

struct DenseOperatorDifference {
    FE::Real maximum_absolute_difference{0.0};
    FE::Real maximum_absolute_entry{0.0};
    std::size_t maximum_difference_index{0u};
};

[[nodiscard]] DenseOperatorDifference compareDenseOperators(
    std::span<const FE::Real> lhs,
    std::span<const FE::Real> rhs)
{
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument(
            "dense operator comparison requires equal sizes");
    }
    DenseOperatorDifference result;
    for (std::size_t index = 0u; index < lhs.size(); ++index) {
        result.maximum_absolute_entry = std::max(
            result.maximum_absolute_entry,
            std::max(std::abs(lhs[index]), std::abs(rhs[index])));
        const auto difference = std::abs(lhs[index] - rhs[index]);
        if (difference > result.maximum_absolute_difference) {
            result.maximum_absolute_difference = difference;
            result.maximum_difference_index = index;
        }
    }
    return result;
}

[[nodiscard]] FieldConstraintCounts globalFieldConstraintCounts(
    const FE::systems::FESystem& system,
    FE::FieldId field,
    MPI_Comm comm)
{
    unsigned long long local_master = 0u;
    unsigned long long local_homogeneous = 0u;
    const auto offset = system.fieldDofOffset(field);
    const auto count = system.fieldDofHandler(field).getNumDofs();
    const auto& owned =
        system.dofHandler().getPartition().locallyOwned();
    for (FE::GlobalIndex local = 0; local < count; ++local) {
        const auto global = offset + local;
        if (!owned.contains(global)) {
            continue;
        }
        const auto line = system.constraints().getConstraint(global);
        if (!line.has_value()) {
            continue;
        }
        if (line->entries.empty()) {
            ++local_homogeneous;
        } else {
            ++local_master;
        }
    }
    return FieldConstraintCounts{
        .master_bearing = static_cast<std::size_t>(
            allreduceSumUnsigned(local_master, comm)),
        .homogeneous_pins = static_cast<std::size_t>(
            allreduceSumUnsigned(local_homogeneous, comm)),
    };
}

[[nodiscard]] WetBlockAssemblySample assembleDistributedWetBlockSample(
    std::span<const FE::Real> x_coordinates,
    std::span<const FE::Real> level_set_by_plane,
    FE::Real dry_state_scale,
    MPI_Comm comm,
    std::string_view partition_method)
{
    constexpr int interface_marker = 27315;
    constexpr std::string_view domain_id =
        "wp1_distributed_physical_wet_block_invariance";
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size != 2) {
        throw std::runtime_error(
            "distributed wet-block sample requires exactly two ranks");
    }
    auto mesh = makeDistributedWetBlockQuadStrip(
        x_coordinates, level_set_by_plane, comm, partition_method);
    auto scalar_space = FE::spaces::SpaceFactory::create_h1(
        FE::ElementType::Quad4, /*order=*/1);
    auto velocity_space = FE::spaces::SpaceFactory::create_vector_h1(
        FE::ElementType::Quad4, /*order=*/1, /*components=*/2);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::Unknown,
    });

    ns::IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = "u";
    options.pressure_field_name = "p";
    options.density = FE::Real{1.2};
    options.viscosity = FE::Real{0.07};
    options.enable_convection = false;
    options.enable_vms = true;
    ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary free_surface;
    free_surface.implementation =
        ns::FreeSurfaceImplementation::UnfittedLevelSet;
    free_surface.interface_marker = interface_marker;
    free_surface.level_set_field_name = "phi";
    free_surface.generated_interface_domain_id = std::string(domain_id);
    free_surface.active_domain =
        ns::FreeSurfaceActiveDomain::LevelSetNegative;
    free_surface.active_domain_method =
        ns::FreeSurfaceActiveDomainMethod::CutVolume;
    free_surface.external_pressure = FE::Real{0.0};
    free_surface.surface_tension = FE::Real{0.0};
    free_surface.use_level_set_curvature = false;
    free_surface.cut_cell_stabilization.enabled = false;
    free_surface.small_cut_aggregation = false;
    free_surface.velocity_extension.enabled = false;
    options.free_surface.push_back(std::move(free_surface));
    ns::IncompressibleNavierStokesVMSModule module(
        velocity_space, scalar_space, options);
    module.registerOn(system);

    FE::systems::SetupOptions setup;
    setup.use_backend_row_ownership_for_assembly = true;
    setup.dof_options.global_numbering =
        FE::dofs::GlobalNumberingMode::OwnerContiguous;
    setup.dof_options.ownership = FE::dofs::OwnershipStrategy::VertexGID;
    setup.dof_options.my_rank = rank;
    setup.dof_options.world_size = size;
    setup.dof_options.mpi_comm = comm;
    system.setup(setup);
    const auto velocity = system.findFieldByName("u");
    const auto pressure = system.findFieldByName("p");
    if (phi == FE::INVALID_FIELD_ID || velocity == FE::INVALID_FIELD_ID ||
        pressure == FE::INVALID_FIELD_ID) {
        throw std::runtime_error(
            "distributed wet-block fields were not registered");
    }

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous = solution;
    const auto phi_handle = MeshFields::get_field_handle(
        mesh->base(), EntityKind::Vertex, "phi");
    const auto* mesh_phi = MeshFields::field_data_as<real_t>(
        mesh->base(), phi_handle);
    const auto* phi_map = system.fieldDofHandler(phi).getEntityDofMap();
    if (mesh_phi == nullptr || phi_map == nullptr) {
        throw std::runtime_error(
            "distributed wet-block level set is unavailable after setup");
    }
    const auto phi_offset = system.fieldDofOffset(phi);
    for (index_t vertex = 0;
         vertex < static_cast<index_t>(mesh->base().n_vertices());
         ++vertex) {
        const auto dofs = phi_map->getVertexDofs(vertex);
        if (dofs.size() != 1u) {
            throw std::runtime_error(
                "distributed wet-block level set is not scalar P1");
        }
        const auto value = mesh_phi[static_cast<std::size_t>(vertex)];
        solution.at(static_cast<std::size_t>(phi_offset + dofs.front())) =
            value;
        previous.at(static_cast<std::size_t>(phi_offset + dofs.front())) =
            value;
    }

    FE::level_set::LevelSetGeneratedInterfaceOptions cut_options;
    cut_options.level_set_field_name = "phi";
    cut_options.domain_id = std::string(domain_id);
    cut_options.requested_interface_marker = interface_marker;
    cut_options.tolerance = FE::Real{1.0e-12};
    cut_options.quadrature_order = 2;
    cut_options.interface_quadrature_order = 2;
    cut_options.volume_quadrature_order = 2;
    FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto generated = lifecycle.build(system, cut_options, solution);
    if (!generated.success) {
        throw std::runtime_error(generated.diagnostic);
    }
    auto context = std::make_shared<FE::assembly::CutIntegrationContext>();
    context->addGeneratedInterfaceDomain(generated.domain);
    system.setCutIntegrationContext(context);
    system.rebuildConstraintState();
    const auto retained = retainedWetBlockVertexGids(
        *mesh, *context, interface_marker);
    std::uint64_t retained_hash = 1469598103934665603ull;
    for (const auto gid : retained) {
        retained_hash ^= static_cast<std::uint64_t>(gid + 1);
        retained_hash *= 1099511628211ull;
    }
    unsigned long long minimum_retained_hash = 0u;
    unsigned long long maximum_retained_hash = 0u;
    const auto local_retained_hash =
        static_cast<unsigned long long>(retained_hash);
    MPI_Allreduce(
        &local_retained_hash,
        &minimum_retained_hash,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MIN,
        comm);
    MPI_Allreduce(
        &local_retained_hash,
        &maximum_retained_hash,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        comm);
    if (minimum_retained_hash != maximum_retained_hash) {
        throw std::runtime_error(
            "distributed wet-block ranks disagree on retained physical support");
    }

    setWetBlockP1Field(
        solution,
        *mesh,
        system,
        velocity,
        /*components=*/2u,
        retained,
        dry_state_scale,
        /*previous_state=*/false);
    setWetBlockP1Field(
        previous,
        *mesh,
        system,
        velocity,
        /*components=*/2u,
        retained,
        dry_state_scale,
        /*previous_state=*/true);
    setWetBlockP1Field(
        solution,
        *mesh,
        system,
        pressure,
        /*components=*/1u,
        retained,
        dry_state_scale,
        /*previous_state=*/false);
    setWetBlockP1Field(
        previous,
        *mesh,
        system,
        pressure,
        /*components=*/1u,
        retained,
        dry_state_scale,
        /*previous_state=*/true);

    FE::systems::SystemStateView state;
    state.dt = FE::Real{0.125};
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;
    const auto local_matrix =
        assembleOperatorMatrix(system, state, "equations");
    const auto local_residual =
        assembleOperatorResidual(system, state, "equations");
    const auto matrix = globalizeOwnedRows(local_matrix, system, comm);
    const auto residual =
        globalizeOwnedResidual(local_residual, system, comm);
    const auto constrained = globalConstraintMask(system, comm);

    WetBlockAssemblySample sample;
    sample.retained_vertices = retained.size();
    const auto& base = mesh->base();
    const auto& gids = base.vertex_gids();
    std::vector<FE::GlobalIndex> dry_columns;
    const auto append_field = [&](FE::FieldId field,
                                  int field_index,
                                  std::size_t components,
                                  std::size_t& dry_constraint_count) {
        const auto* entity_map =
            system.fieldDofHandler(field).getEntityDofMap();
        if (entity_map == nullptr) {
            throw std::runtime_error(
                "distributed wet-block canonical field has no entity map");
        }
        const auto offset = system.fieldDofOffset(field);
        for (index_t vertex = 0;
             vertex < static_cast<index_t>(base.n_vertices());
             ++vertex) {
            const auto vertex_dofs = entity_map->getVertexDofs(vertex);
            if (vertex_dofs.size() != components) {
                throw std::runtime_error(
                    "distributed wet-block field has an unexpected component count");
            }
            const auto gid = gids.at(static_cast<std::size_t>(vertex));
            const bool wet_supported = retained.contains(gid);
            for (std::size_t component = 0u;
                 component < components;
                 ++component) {
                const auto global = offset + vertex_dofs[component];
                if (global < 0 ||
                    static_cast<std::size_t>(global) >= constrained.size()) {
                    throw std::runtime_error(
                        "distributed wet-block DOF is outside the global constraint mask");
                }
                const bool is_constrained =
                    constrained[static_cast<std::size_t>(global)] != 0;
                if (wet_supported) {
                    if (is_constrained) {
                        throw std::runtime_error(
                            "distributed wet-supported physical DOF was constrained");
                    }
                    sample.dofs.push_back(CanonicalWetBlockDof{
                        .field = field_index,
                        .vertex_gid = gid,
                        .component = component,
                        .global_dof = global,
                        .point = system.meshAccess().getNodeCoordinates(vertex),
                    });
                } else {
                    if (!is_constrained) {
                        throw std::runtime_error(
                            "distributed dry-only physical DOF was not constrained");
                    }
                    ++dry_constraint_count;
                    dry_columns.push_back(global);
                }
            }
        }
    };
    append_field(
        velocity,
        /*field_index=*/0,
        /*components=*/2u,
        sample.constrained_dry_velocity_dofs);
    append_field(
        pressure,
        /*field_index=*/1,
        /*components=*/1u,
        sample.constrained_dry_pressure_dofs);
    std::sort(
        sample.dofs.begin(),
        sample.dofs.end(),
        [](const CanonicalWetBlockDof& lhs,
           const CanonicalWetBlockDof& rhs) {
            if (lhs.field != rhs.field) {
                return lhs.field < rhs.field;
            }
            if (lhs.vertex_gid != rhs.vertex_gid) {
                return lhs.vertex_gid < rhs.vertex_gid;
            }
            return lhs.component < rhs.component;
        });
    std::vector<FE::GlobalIndex> wet_dofs;
    wet_dofs.reserve(sample.dofs.size());
    sample.current_state.reserve(sample.dofs.size());
    sample.residual.reserve(sample.dofs.size());
    for (const auto& dof : sample.dofs) {
        wet_dofs.push_back(dof.global_dof);
        sample.current_state.push_back(
            solution.at(static_cast<std::size_t>(dof.global_dof)));
        sample.residual.push_back(
            residual.at(static_cast<std::size_t>(dof.global_dof)));
    }
    sample.jacobian = extractRectangularMatrix(
        matrix, wet_dofs, wet_dofs);
    auto correction = sample.residual;
    for (auto& value : correction) {
        value = -value;
    }
    auto solver = FE::math::factor_dense_matrix(
        sample.jacobian,
        sample.dofs.size(),
        "distributed WP-1 physical wet-block Jacobian");
    solver.solve_in_place(correction, /*right_hand_sides=*/1u);
    sample.solved_state = sample.current_state;
    for (std::size_t index = 0u;
         index < sample.solved_state.size();
         ++index) {
        sample.solved_state[index] += correction[index];
    }
    sample.dry_column_coupling_norm = selectedFrobeniusNorm(
        matrix, wet_dofs, dry_columns);
    return sample;
}

[[nodiscard]] std::uint64_t distributedCellOwnerHash(
    const Mesh& mesh,
    MPI_Comm comm)
{
    const auto global_cells = mesh.global_n_cells();
    if (global_cells == 0u ||
        global_cells >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "distributed stability partition has an invalid global cell count");
    }
    std::vector<int> local_owner(global_cells, -1);
    std::vector<int> global_owner(global_cells, -1);
    const auto& base = mesh.base();
    const auto& gids = base.cell_gids();
    for (index_t cell = 0;
         cell < static_cast<index_t>(base.n_cells());
         ++cell) {
        if (!mesh.is_owned_cell(cell)) {
            continue;
        }
        const auto gid = gids.at(static_cast<std::size_t>(cell));
        if (gid < 0 ||
            static_cast<std::size_t>(gid) >= global_cells) {
            throw std::runtime_error(
                "distributed stability cell GID is out of range");
        }
        local_owner[static_cast<std::size_t>(gid)] = mesh.rank();
    }
    MPI_Allreduce(
        local_owner.data(),
        global_owner.data(),
        static_cast<int>(global_cells),
        MPI_INT,
        MPI_MAX,
        comm);

    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t gid = 0; gid < global_cells; ++gid) {
        if (global_owner[gid] < 0) {
            throw std::runtime_error(
                "distributed stability partition has an unowned cell");
        }
        hash ^= static_cast<std::uint64_t>(gid + 1u);
        hash *= 1099511628211ull;
        hash ^= static_cast<std::uint64_t>(global_owner[gid] + 1);
        hash *= 1099511628211ull;
    }
    return hash;
}

class DistributedStabilityProblem {
public:
    DistributedStabilityProblem(
        const PlaneCutPosition& initial_cut,
        MPI_Comm comm,
        std::string partition_method,
        int ghost_layers = 18,
        int cells_per_axis = 0,
        StabilityRegime regime = {})
        : comm_(comm),
          mesh_(
              cells_per_axis > 0
                  ? makeDistributedStructuredTetraMesh(
                        initial_cut,
                        comm,
                        partition_method,
                        cells_per_axis)
                  : makeDistributedTetraStripMesh(
                        initial_cut,
                        comm,
                        partition_method,
                        ghost_layers)),
          velocity_space_(FE::spaces::SpaceFactory::create_vector_h1(
              FE::ElementType::Tetra4, /*order=*/1, /*components=*/3)),
          pressure_space_(FE::spaces::SpaceFactory::create_h1(
              FE::ElementType::Tetra4, /*order=*/1)),
          system_(mesh_),
          cells_per_axis_(cells_per_axis),
          regime_(regime)
    {
        int rank = 0;
        int size = 1;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &size);
        if (size != 2 && size != 4) {
            throw std::runtime_error(
                "distributed stability problem requires two or four ranks");
        }
        const auto expected_cells =
            cells_per_axis_ > 0
                ? 6u *
                      static_cast<std::size_t>(cells_per_axis_) *
                      static_cast<std::size_t>(cells_per_axis_) *
                      static_cast<std::size_t>(cells_per_axis_)
                : std::size_t{18u};
        if (mesh_->global_n_cells() != expected_cells ||
            mesh_->n_owned_cells() == 0u ||
            mesh_->n_owned_cells() >= mesh_->global_n_cells()) {
            throw std::runtime_error(
                "distributed stability mesh is not genuinely partitioned");
        }

        phi_ = system_.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = pressure_space_,
            .components = 1,
            .source_kind = FE::systems::FieldSourceKind::Unknown,
        });
        auto options = stabilityOptions(
            interface_marker, std::string(domain_id), regime_);
        ns::IncompressibleNavierStokesVMSModule module(
            velocity_space_, pressure_space_, options);
        module.registerOn(system_);

        FE::systems::SetupOptions setup;
        setup.use_backend_row_ownership_for_assembly = true;
        setup.dof_options.global_numbering =
            FE::dofs::GlobalNumberingMode::OwnerContiguous;
        setup.dof_options.ownership = FE::dofs::OwnershipStrategy::VertexGID;
        setup.dof_options.my_rank = rank;
        setup.dof_options.world_size = size;
        setup.dof_options.mpi_comm = comm_;
        try {
            system_.setup(setup);
        } catch (...) {
            if (const auto* registry = system_.gaugeRegistryIfPresent();
                registry != nullptr) {
                std::cerr << "FS14_distributed_setup_gauge rank=" << rank
                          << '\n'
                          << registry->diagnosticReport();
            }
            throw;
        }

        velocity_ = system_.findFieldByName("u");
        pressure_ = system_.findFieldByName("p");
        if (phi_ == FE::INVALID_FIELD_ID ||
            velocity_ == FE::INVALID_FIELD_ID ||
            pressure_ == FE::INVALID_FIELD_ID) {
            throw std::runtime_error(
                "distributed stability fields were not registered");
        }
        solution_.assign(
            static_cast<std::size_t>(system_.dofHandler().getNumDofs()), 0.0);
        const auto* velocity_map =
            system_.fieldDofHandler(velocity_).getEntityDofMap();
        if (velocity_map == nullptr) {
            throw std::runtime_error(
                "distributed stability velocity field has no entity map");
        }
        const auto velocity_offset = system_.fieldDofOffset(velocity_);
        for (FE::GlobalIndex vertex = 0;
             vertex < system_.meshAccess().numVertices();
             ++vertex) {
            const auto dofs = velocity_map->getVertexDofs(vertex);
            if (dofs.size() != 3u) {
                throw std::runtime_error(
                    "distributed stability velocity field is not vector P1");
            }
            solution_.at(static_cast<std::size_t>(
                velocity_offset + dofs[0])) = regime_.advective_speed;
        }
        previous_ = solution_;
        partition_hash_ = distributedCellOwnerHash(*mesh_, comm_);
    }

    [[nodiscard]] std::uint64_t partitionHash() const noexcept
    {
        return partition_hash_;
    }

    [[nodiscard]] StabilitySample evaluate(
        const PlaneCutPosition& cut,
        std::optional<FE::MeshIndex> designated_parent_cell =
            std::nullopt)
    {
        setScalarVertexField(solution_, system_, phi_, cut);
        setMeshVertexField(*mesh_, cut);
        if (!has_previous_sample_) {
            previous_ = solution_;
        }

        FE::level_set::LevelSetGeneratedInterfaceOptions cut_options;
        cut_options.level_set_field_name = "phi";
        cut_options.domain_id = std::string(domain_id);
        cut_options.requested_interface_marker = interface_marker;
        cut_options.tolerance = 1.0e-12;
        cut_options.quadrature_order = 2;
        cut_options.interface_quadrature_order = 1;
        cut_options.volume_quadrature_order = 2;
        const auto generated = lifecycle_.build(
            system_, cut_options, solution_);
        if (!generated.success) {
            throw std::runtime_error(generated.diagnostic);
        }

        auto context = std::make_shared<FE::assembly::CutIntegrationContext>();
        context->addGeneratedInterfaceDomain(generated.domain);
        const auto facet_handle = addProductionFacetSet(
            *context, generated.domain, system_.meshAccess());
        // Validate the aggregate declarations as soon as the production cut
        // context is installed.  The physical facet-set comparison below is a
        // second fail-closed guard: depending on which incomplete support is
        // visible first, a limited halo can be rejected by either guard.
        system_.setCutIntegrationContext(context);
        system_.rebuildConstraintState();
        std::vector<gid_t> facet_gids;
        const auto& mesh_face_gids = mesh_->base().face_gids();
        facet_gids.reserve(facet_handle.facets.size());
        for (const auto face : facet_handle.facets) {
            if (face < 0 ||
                static_cast<std::size_t>(face) >= mesh_face_gids.size()) {
                throw std::runtime_error(
                    "distributed stability facet has no canonical GID");
            }
            facet_gids.push_back(
                mesh_face_gids[static_cast<std::size_t>(face)]);
        }
        std::sort(facet_gids.begin(), facet_gids.end());
        facet_gids.erase(
            std::unique(facet_gids.begin(), facet_gids.end()),
            facet_gids.end());
        unsigned long long facet_gid_hash = 1469598103934665603ull;
        for (const auto gid : facet_gids) {
            facet_gid_hash ^= static_cast<unsigned long long>(gid);
            facet_gid_hash *= 1099511628211ull;
        }
        unsigned long long minimum_facet_gid_hash = 0u;
        unsigned long long maximum_facet_gid_hash = 0u;
        MPI_Allreduce(
            &facet_gid_hash,
            &minimum_facet_gid_hash,
            1,
            MPI_UNSIGNED_LONG_LONG,
            MPI_MIN,
            comm_);
        MPI_Allreduce(
            &facet_gid_hash,
            &maximum_facet_gid_hash,
            1,
            MPI_UNSIGNED_LONG_LONG,
            MPI_MAX,
            comm_);
        if (minimum_facet_gid_hash != maximum_facet_gid_hash) {
            throw std::runtime_error(
                "distributed stability ranks generated different physical "
                "cut-adjacent facet sets");
        }
        const auto pressure_anchor = pressureAnchorState(system_, pressure_);

        FE::systems::SystemStateView state;
        state.dt = regime_.dt;
        state.u = std::span<const FE::Real>(solution_);
        state.u_prev = std::span<const FE::Real>(previous_);
        const FE::systems::BackwardDifferenceIntegrator integrator;
        const auto time_context =
            integrator.buildContext(/*max_time_derivative_order=*/1, state);
        state.time_integration = &time_context;

        const auto local_jacobian =
            assembleOperatorMatrix(system_, state, "equations");
        const auto local_ghost = assembleOperatorMatrix(
            system_, state,
            "equations_diagnostic_ns_pressure_ghost_penalty");
        const auto local_pspg = assembleOperatorMatrix(
            system_, state,
            "equations_diagnostic_ns_vms_pspg_pressure_gradient");
        const auto jacobian =
            globalizeOwnedRows(local_jacobian, system_, comm_);
        const auto ghost =
            globalizeOwnedRows(local_ghost, system_, comm_);
        const auto pspg =
            globalizeOwnedRows(local_pspg, system_, comm_);

        const auto constrained = globalConstraintMask(system_, comm_);
        const auto numbered_free_velocity = globalFreeFieldDofs(
            system_, velocity_, constrained);
        const auto numbered_free_pressure = globalFreeFieldDofs(
            system_, pressure_, constrained);
        auto free_velocity = canonicalGlobalFreeP1Dofs(
            *mesh_, system_, velocity_, /*components=*/3u, constrained);
        auto free_pressure = canonicalGlobalFreeP1Dofs(
            *mesh_, system_, pressure_, /*components=*/1u, constrained);
        if (free_velocity.size() != numbered_free_velocity.size() ||
            free_pressure.size() != numbered_free_pressure.size()) {
            throw std::runtime_error(
                "distributed stability canonical ordering requires complete "
                "P1 vertex overlap");
        }
        std::vector<FE::GlobalIndex> free_mixed = free_velocity;
        free_mixed.insert(
            free_mixed.end(), free_pressure.begin(), free_pressure.end());
        if (free_mixed.empty() || free_pressure.empty()) {
            throw std::runtime_error(
                "distributed stability mixed free space is empty");
        }

        auto reduced = extractReducedMatrix(jacobian, free_mixed);
        const auto canonical_mixed_operator = reduced;
        const auto canonical_pressure_ghost_operator =
            extractReducedMatrix(ghost, free_pressure);
        const auto canonical_pressure_pspg_operator =
            extractReducedMatrix(pspg, free_pressure);
        const auto reduced_max = FE::math::dense_matrix_max_abs(reduced);
        const auto zero_row_tolerance =
            std::max(FE::Real{1.0}, reduced_max) * FE::Real{1.0e-12};
        std::size_t zero_pressure_rows = 0u;
        for (std::size_t local = 0; local < free_pressure.size(); ++local) {
            const auto row = free_velocity.size() + local;
            FE::Real row_norm = 0.0;
            for (std::size_t column = 0; column < free_mixed.size(); ++column) {
                const auto value = reduced[row * free_mixed.size() + column];
                row_norm += value * value;
            }
            if (std::sqrt(row_norm) <= zero_row_tolerance) {
                ++zero_pressure_rows;
            }
        }
        equilibrate(reduced, free_mixed.size());
        const auto diagnostics = FE::math::dense_matrix_diagnostics(
            reduced,
            free_mixed.size(),
            free_mixed.size(),
            "equilibrated distributed free-surface mixed Jacobian");

        unsigned long long local_cut_cells = 0u;
        for (const auto cell : generated.domain.cutCells()) {
            if (cell >= 0 && system_.meshAccess().isOwnedCell(cell)) {
                ++local_cut_cells;
            }
        }
        StabilitySample sample;
        sample.label = cut.label;
        sample.reference_active_volume =
            generated.summary.negative_volume_measure;
        sample.physical_active_volume = sample.reference_active_volume;
        sample.cut_cells = static_cast<std::size_t>(
            allreduceSumUnsigned(local_cut_cells, comm_));
        // This finite fixture intentionally retains its complete physical mesh
        // in overlap.  Canonical face GIDs above prove that every rank sees
        // the same physical facet set; counting "first-cell owned" is invalid
        // because the local minus/plus orientation is partition dependent.
        sample.cut_adjacent_facets = facet_gids.size();
        sample.cut_adjacent_facet_gid_hash =
            static_cast<std::uint64_t>(facet_gid_hash);
        sample.backend_volume_quadrature_points =
            generated.backend_volume_quadrature_point_count;
        sample.backend_fallback_cells =
            generated.implicit_cut_fallback_cell_count;
        sample.pressure_natural_traction_anchor =
            pressure_anchor.natural_traction_anchor;
        sample.pressure_anchor_has_no_gauge_enforcement =
            pressure_anchor.no_gauge_enforcement;
        for (const auto& region : generated.domain.volumeRegions()) {
            if (!region.active() ||
                region.side !=
                    FE::geometry::CutIntegrationSide::Negative ||
                region.full_cell_equivalent ||
                !(region.volume_fraction > FE::Real{0.0}) ||
                !(region.volume_fraction < FE::Real{1.0})) {
                continue;
            }
            sample.minimum_active_cut_fraction = std::min(
                sample.minimum_active_cut_fraction,
                region.volume_fraction);
            if (designated_parent_cell.has_value() &&
                region.parent_cell_global_id ==
                    static_cast<FE::GlobalIndex>(
                        designated_parent_cell.value())) {
                sample.designated_cut_fraction =
                    region.volume_fraction;
            }
        }
        sample.velocity_constraints = globalFieldConstraintCounts(
            system_, velocity_, comm_);
        sample.pressure_constraints = globalFieldConstraintCounts(
            system_, pressure_, comm_);
        sample.free_velocity_dofs = free_velocity.size();
        sample.free_pressure_dofs = free_pressure.size();
        sample.zero_free_pressure_rows = zero_pressure_rows;
        sample.pressure_ghost_norm =
            selectedFrobeniusNorm(ghost, free_pressure, free_pressure);
        sample.pspg_pressure_gradient_norm =
            selectedFrobeniusNorm(pspg, free_pressure, free_pressure);
        sample.canonical_mixed_operator = canonical_mixed_operator;
        sample.canonical_pressure_ghost_operator =
            canonical_pressure_ghost_operator;
        sample.canonical_pressure_pspg_operator =
            canonical_pressure_pspg_operator;
        sample.equilibrated_rank = diagnostics.rank;
        sample.equilibrated_size = free_mixed.size();
        sample.equilibrated_condition_inf =
            infinityNormCondition(reduced, free_mixed.size());
        sample.krylov = runEquilibratedJacobiBicgstab(
            reduced, free_mixed.size());

        previous_ = solution_;
        has_previous_sample_ = true;
        return sample;
    }

private:
    static constexpr int interface_marker = 27014;
    static constexpr std::string_view domain_id =
        "fs14_distributed_tetra_strip";

    MPI_Comm comm_{MPI_COMM_NULL};
    std::shared_ptr<Mesh> mesh_{};
    std::shared_ptr<FE::spaces::ProductSpace> velocity_space_{};
    std::shared_ptr<FE::spaces::H1Space> pressure_space_{};
    FE::systems::FESystem system_;
    FE::FieldId phi_{FE::INVALID_FIELD_ID};
    FE::FieldId velocity_{FE::INVALID_FIELD_ID};
    FE::FieldId pressure_{FE::INVALID_FIELD_ID};
    std::vector<FE::Real> solution_{};
    std::vector<FE::Real> previous_{};
    FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle_{};
    std::uint64_t partition_hash_{0u};
    bool has_previous_sample_{false};
    int cells_per_axis_{0};
    StabilityRegime regime_{};
};

void runDistributedFrozenMatrix(int expected_ranks)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    ASSERT_NE(initialized, 0) << "Run this test under mpiexec.";
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    ASSERT_EQ(size, expected_ranks);

    const ScopedEnvVar pressure_diagnostics(
        "SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC", "1");
    constexpr std::array<FE::Real, 7> fractions = {{
        FE::Real{1.0e-8},
        FE::Real{1.0e-6},
        FE::Real{1.0e-4},
        FE::Real{1.0e-2},
        FE::Real{0.1},
        FE::Real{0.25},
        FE::Real{0.49},
    }};
    constexpr std::array<std::array<FE::Real, 3>, 2> orientations = {{
        {{1.0, 0.0, 0.0}},
        {{1.0, 0.73, 0.41}},
    }};
    constexpr std::array<int, 3> resolutions = {{2, 3, 4}};
    constexpr std::array<StabilityRegime, 3> regimes = {{
        StabilityRegime{
            .id = "viscous",
            .density = FE::Real{1.0},
            .viscosity = FE::Real{1.0},
            .dt = FE::Real{0.1},
            .convection = false,
            .advective_speed = FE::Real{0.0},
        },
        StabilityRegime{
            .id = "transient",
            .density = FE::Real{1.0},
            .viscosity = FE::Real{0.01},
            .dt = FE::Real{0.001},
            .convection = false,
            .advective_speed = FE::Real{0.0},
        },
        StabilityRegime{
            .id = "advection",
            .density = FE::Real{1.0},
            .viscosity = FE::Real{0.001},
            .dt = FE::Real{0.1},
            .convection = true,
            .advective_speed = FE::Real{1.0},
        },
    }};

    std::size_t case_count = 0u;
    std::size_t distinct_partition_count = 0u;
    std::size_t krylov_nonconvergence_count = 0u;
    FE::Real maximum_fraction_relative_error = 0.0;
    FE::Real maximum_serial_block_operator_difference = 0.0;
    FE::Real maximum_serial_metis_operator_difference = 0.0;
    FE::Real maximum_block_metis_operator_difference = 0.0;
    FE::Real maximum_partition_condition_relative_difference = 0.0;
    FE::Real maximum_rank_condition_relative_difference = 0.0;

    const auto compare_operator =
        [&](std::span<const FE::Real> first,
            std::span<const FE::Real> second,
            std::string_view comparison) {
            EXPECT_EQ(second.size(), first.size()) << comparison;
            if (second.size() != first.size()) {
                return std::numeric_limits<FE::Real>::infinity();
            }
            const auto difference =
                compareDenseOperators(first, second);
            const auto tolerance =
                FE::Real{8192.0} *
                std::numeric_limits<FE::Real>::epsilon() *
                std::max(FE::Real{1.0},
                         difference.maximum_absolute_entry);
            EXPECT_LE(
                difference.maximum_absolute_difference, tolerance)
                << "comparison=" << comparison
                << " flat_index="
                << difference.maximum_difference_index
                << " scale=" << difference.maximum_absolute_entry;
            return difference.maximum_absolute_difference;
        };
    const auto relative_difference =
        [](FE::Real first, FE::Real second) {
            return std::abs(first - second) /
                   std::max(
                       FE::Real{1.0e-30},
                       std::max(std::abs(first), std::abs(second)));
        };

    for (const auto resolution : resolutions) {
        const auto structured_cell_count =
            6 * resolution * resolution * resolution;
        for (std::size_t orientation = 0u;
             orientation < orientations.size();
             ++orientation) {
            std::array<TargetStructuredCut, fractions.size()> cuts;
            for (std::size_t fraction = 0u;
                 fraction < fractions.size();
                 ++fraction) {
                cuts[fraction] = makeTargetStructuredCut(
                    fractions[fraction],
                    orientations[orientation],
                    resolution,
                    std::string("distributed_matrix_fraction_") +
                        std::to_string(fraction));
            }
            for (const auto& regime : regimes) {
                SCOPED_TRACE(
                    std::string("ranks=") +
                    std::to_string(expected_ranks) +
                    " resolution=" + std::to_string(resolution) +
                    " orientation=" + std::to_string(orientation) +
                    " regime=" + std::string(regime.id));
                PersistentStabilityProblem serial(
                    cuts.front().cut, resolution, regime);
                DistributedStabilityProblem block(
                    cuts.front().cut,
                    comm,
                    "block",
                    structured_cell_count,
                    resolution,
                    regime);
                DistributedStabilityProblem metis(
                    cuts.front().cut,
                    comm,
                    "metis",
                    structured_cell_count,
                    resolution,
                    regime);
                if (block.partitionHash() != metis.partitionHash()) {
                    ++distinct_partition_count;
                }

                for (std::size_t fraction = 0u;
                     fraction < fractions.size();
                     ++fraction) {
                    SCOPED_TRACE(
                        std::string("fraction=") +
                        std::to_string(fractions[fraction]));
                    const auto serial_sample = serial.evaluate(
                        cuts[fraction].cut,
                        cuts[fraction].designated_parent_cell);
                    const auto block_sample = block.evaluate(
                        cuts[fraction].cut,
                        cuts[fraction].designated_parent_cell);
                    const auto metis_sample = metis.evaluate(
                        cuts[fraction].cut,
                        cuts[fraction].designated_parent_cell);
                    for (const auto* sample :
                         {&serial_sample, &block_sample, &metis_sample}) {
                        ASSERT_TRUE(std::isfinite(
                            sample->designated_cut_fraction));
                        const auto fraction_error =
                            std::abs(
                                sample->designated_cut_fraction -
                                fractions[fraction]) /
                            fractions[fraction];
                        maximum_fraction_relative_error = std::max(
                            maximum_fraction_relative_error,
                            fraction_error);
                        EXPECT_LE(
                            fraction_error, FE::Real{5.0e-8});
                        EXPECT_GT(sample->cut_cells, 0u);
                        EXPECT_GT(sample->cut_adjacent_facets, 0u);
                        EXPECT_EQ(
                            sample->backend_fallback_cells, 0u);
                        EXPECT_EQ(
                            sample->zero_free_pressure_rows, 0u);
                        EXPECT_EQ(
                            sample->equilibrated_rank,
                            sample->equilibrated_size);
                        EXPECT_TRUE(std::isfinite(
                            sample->equilibrated_condition_inf));
                        if (!sample->krylov.converged) {
                            ++krylov_nonconvergence_count;
                        }
                        EXPECT_TRUE(sample->krylov.converged);
                        EXPECT_FALSE(sample->krylov.breakdown);
                    }

                    EXPECT_EQ(
                        block_sample.cut_cells,
                        serial_sample.cut_cells);
                    EXPECT_EQ(
                        metis_sample.cut_cells,
                        serial_sample.cut_cells);
                    EXPECT_EQ(
                        block_sample.cut_adjacent_facets,
                        serial_sample.cut_adjacent_facets);
                    EXPECT_EQ(
                        metis_sample.cut_adjacent_facets,
                        serial_sample.cut_adjacent_facets);
                    EXPECT_EQ(
                        block_sample.free_velocity_dofs,
                        serial_sample.free_velocity_dofs);
                    EXPECT_EQ(
                        metis_sample.free_velocity_dofs,
                        serial_sample.free_velocity_dofs);
                    EXPECT_EQ(
                        block_sample.free_pressure_dofs,
                        serial_sample.free_pressure_dofs);
                    EXPECT_EQ(
                        metis_sample.free_pressure_dofs,
                        serial_sample.free_pressure_dofs);
                    EXPECT_EQ(
                        block_sample.pressure_constraints.master_bearing,
                        serial_sample.pressure_constraints.master_bearing);
                    EXPECT_EQ(
                        metis_sample.pressure_constraints.master_bearing,
                        serial_sample.pressure_constraints.master_bearing);
                    EXPECT_EQ(
                        block_sample.pressure_constraints.homogeneous_pins,
                        serial_sample.pressure_constraints.homogeneous_pins);
                    EXPECT_EQ(
                        metis_sample.pressure_constraints.homogeneous_pins,
                        serial_sample.pressure_constraints.homogeneous_pins);

                    maximum_serial_block_operator_difference = std::max(
                        maximum_serial_block_operator_difference,
                        compare_operator(
                            serial_sample.canonical_mixed_operator,
                            block_sample.canonical_mixed_operator,
                            "serial-block mixed Jacobian"));
                    maximum_serial_metis_operator_difference = std::max(
                        maximum_serial_metis_operator_difference,
                        compare_operator(
                            serial_sample.canonical_mixed_operator,
                            metis_sample.canonical_mixed_operator,
                            "serial-METIS mixed Jacobian"));
                    maximum_block_metis_operator_difference = std::max(
                        maximum_block_metis_operator_difference,
                        compare_operator(
                            block_sample.canonical_mixed_operator,
                            metis_sample.canonical_mixed_operator,
                            "block-METIS mixed Jacobian"));
                    compare_operator(
                        serial_sample.canonical_pressure_ghost_operator,
                        block_sample.canonical_pressure_ghost_operator,
                        "serial-block pressure ghost");
                    compare_operator(
                        serial_sample.canonical_pressure_ghost_operator,
                        metis_sample.canonical_pressure_ghost_operator,
                        "serial-METIS pressure ghost");
                    compare_operator(
                        serial_sample.canonical_pressure_pspg_operator,
                        block_sample.canonical_pressure_pspg_operator,
                        "serial-block pressure PSPG");
                    compare_operator(
                        serial_sample.canonical_pressure_pspg_operator,
                        metis_sample.canonical_pressure_pspg_operator,
                        "serial-METIS pressure PSPG");

                    const auto partition_condition_difference =
                        relative_difference(
                            block_sample.equilibrated_condition_inf,
                            metis_sample.equilibrated_condition_inf);
                    const auto block_rank_condition_difference =
                        relative_difference(
                            serial_sample.equilibrated_condition_inf,
                            block_sample.equilibrated_condition_inf);
                    const auto metis_rank_condition_difference =
                        relative_difference(
                            serial_sample.equilibrated_condition_inf,
                            metis_sample.equilibrated_condition_inf);
                    maximum_partition_condition_relative_difference =
                        std::max(
                            maximum_partition_condition_relative_difference,
                            partition_condition_difference);
                    maximum_rank_condition_relative_difference = std::max(
                        maximum_rank_condition_relative_difference,
                        std::max(
                            block_rank_condition_difference,
                            metis_rank_condition_difference));
                    EXPECT_LE(
                        partition_condition_difference,
                        FE::Real{1.0e-9});
                    EXPECT_LE(
                        block_rank_condition_difference,
                        FE::Real{1.0e-9});
                    EXPECT_LE(
                        metis_rank_condition_difference,
                        FE::Real{1.0e-9});
                    ++case_count;
                }
            }
        }
    }

    const auto expected_case_count =
        fractions.size() * orientations.size() *
        resolutions.size() * regimes.size();
    EXPECT_EQ(case_count, expected_case_count);
    EXPECT_GT(distinct_partition_count, 0u);
    EXPECT_EQ(krylov_nonconvergence_count, 0u);
    if (rank == 0) {
        std::cout << std::setprecision(17)
                  << "WP7_distributed_full_matrix"
                  << " ranks=" << size
                  << " cases=" << case_count
                  << " distinct_partition_configurations="
                  << distinct_partition_count
                  << " maximum_fraction_relative_error="
                  << maximum_fraction_relative_error
                  << " maximum_serial_block_operator_difference="
                  << maximum_serial_block_operator_difference
                  << " maximum_serial_metis_operator_difference="
                  << maximum_serial_metis_operator_difference
                  << " maximum_block_metis_operator_difference="
                  << maximum_block_metis_operator_difference
                  << " maximum_partition_condition_relative_difference="
                  << maximum_partition_condition_relative_difference
                  << " maximum_rank_condition_relative_difference="
                  << maximum_rank_condition_relative_difference
                  << " krylov_nonconvergence_count="
                  << krylov_nonconvergence_count
                  << " scope=finite_p1_rank_equivalence_not_uniform_theorem"
                  << '\n';
    }
}

#endif

[[nodiscard]] StabilitySample runStabilitySample(
    const PlaneCutPosition& cut)
{
    PersistentStabilityProblem problem(cut);
    return problem.evaluate(cut);
}

#endif

} // namespace

TEST(FreeSurfaceCutStability,
     PhysicalWetBlocksAreInvariantToDryDepthAndState)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Wet-block invariance requires native mesh support.";
#else
    constexpr std::array<FE::Real, 4> baseline_x = {
        FE::Real{-1.0}, FE::Real{0.0}, FE::Real{1.0}, FE::Real{2.0}};
    constexpr std::array<FE::Real, 4> baseline_phi = {
        FE::Real{-1.25}, FE::Real{-0.25}, FE::Real{0.75},
        FE::Real{1.75}};
    constexpr std::array<FE::Real, 7> deep_x = {
        FE::Real{-1.0}, FE::Real{0.0}, FE::Real{1.0}, FE::Real{2.0},
        FE::Real{3.0}, FE::Real{4.0}, FE::Real{5.0}};
    constexpr std::array<FE::Real, 7> deep_phi = {
        FE::Real{-1.25}, FE::Real{-0.25}, FE::Real{0.75},
        FE::Real{1.75}, FE::Real{2.75}, FE::Real{3.75},
        FE::Real{4.75}};
    constexpr FE::Real serial_gate = FE::Real{1.0e-11};
    constexpr FE::Real serial_solved_gate = FE::Real{1.0e-10};
    constexpr FE::Real norm_floor = FE::Real{1.0e-12};

    const auto baseline = assembleSerialWetBlockSample(
        baseline_x,
        baseline_phi,
        /*dry_state_scale=*/FE::Real{3.0});
    const auto depth = assembleSerialWetBlockSample(
        deep_x,
        deep_phi,
        /*dry_state_scale=*/FE::Real{3.0});
    const auto dry_state = assembleSerialWetBlockSample(
        baseline_x,
        baseline_phi,
        // This changes every dry-only coefficient, including the exterior
        // right-boundary vertices, while preserving every retained-support
        // coefficient.
        /*dry_state_scale=*/FE::Real{1.0e6});

    ASSERT_EQ(baseline.retained_vertices, 6u);
    EXPECT_EQ(depth.retained_vertices, baseline.retained_vertices);
    EXPECT_EQ(baseline.constrained_dry_velocity_dofs, 4u);
    EXPECT_EQ(baseline.constrained_dry_pressure_dofs, 2u);
    EXPECT_EQ(depth.constrained_dry_velocity_dofs, 16u);
    EXPECT_EQ(depth.constrained_dry_pressure_dofs, 8u);
    EXPECT_GT(vectorL2Norm(baseline.residual), FE::Real{0.0});
    EXPECT_GT(vectorL2Norm(baseline.jacobian), FE::Real{0.0});

    const std::array<std::pair<std::string_view, ScaledWetBlockDifference>, 2>
        comparisons = {{
            {"dry_depth", compareWetBlockSamples(baseline, depth)},
            {"exterior_dry_values",
             compareWetBlockSamples(baseline, dry_state)},
        }};
    // The retired same-field diffusivity is intentionally absent from this
    // numerical matrix: NavierStokesLegacyBCs has separate negative tests
    // proving that enabling it, or specifying its coefficient while disabled,
    // fails before fields or assembly are created.
    for (const auto& [factor, difference] : comparisons) {
        SCOPED_TRACE(factor);
        EXPECT_LE(difference.residual, serial_gate);
        EXPECT_LE(difference.jacobian, serial_gate);
        EXPECT_LE(difference.solved_state, serial_solved_gate);
        std::cout << std::setprecision(17)
                  << "WP1_wet_block_invariance"
                  << " scope=serial"
                  << " factor=" << factor
                  << " residual_absolute_floor=" << norm_floor
                  << " jacobian_absolute_floor=" << norm_floor
                  << " scaled_residual_difference=" << difference.residual
                  << " scaled_jacobian_difference=" << difference.jacobian
                  << " scaled_solved_state_difference="
                  << difference.solved_state
                  << " residual_absolute_difference="
                  << difference.residual_absolute
                  << " jacobian_absolute_difference="
                  << difference.jacobian_absolute
                  << " solved_state_absolute_difference="
                  << difference.solved_state_absolute
                  << " accepted_gate=" << serial_gate
                  << " solved_state_gate=" << serial_solved_gate << '\n';
    }

    for (const auto* sample : {&baseline, &depth, &dry_state}) {
        const auto scaled_dry_coupling = sample->dry_column_coupling_norm /
            (norm_floor + vectorL2Norm(sample->jacobian));
        EXPECT_LE(scaled_dry_coupling, serial_gate);
    }
#endif
}

TEST(FreeSurfaceCutStability,
     DisconnectedLiquidIslandsHaveZeroPhysicalCrossCouplingThroughDryStrip)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Wet-block island decoupling requires native mesh support.";
#else
    constexpr std::array<FE::Real, 7> x = {
        FE::Real{0.0}, FE::Real{1.0}, FE::Real{2.0}, FE::Real{3.0},
        FE::Real{4.0}, FE::Real{5.0}, FE::Real{6.0}};
    constexpr std::array<FE::Real, 7> phi = {
        FE::Real{-1.0}, FE::Real{-1.0}, FE::Real{1.0}, FE::Real{1.0},
        FE::Real{1.0}, FE::Real{-1.0}, FE::Real{-1.0}};
    constexpr FE::Real serial_gate = FE::Real{1.0e-11};
    constexpr FE::Real serial_solved_gate = FE::Real{1.0e-10};
    constexpr FE::Real norm_floor = FE::Real{1.0e-12};
    const auto baseline = assembleSerialWetBlockSample(
        x,
        phi,
        /*dry_state_scale=*/FE::Real{2.0});
    const auto changed_dry_path = assembleSerialWetBlockSample(
        x,
        phi,
        /*dry_state_scale=*/FE::Real{1.0e6});
    ASSERT_EQ(baseline.retained_vertices, 12u);
    ASSERT_EQ(baseline.dofs.size(), 36u);
    const auto dry_path_difference = compareWetBlockSamples(
        baseline, changed_dry_path);
    EXPECT_LE(dry_path_difference.residual, serial_gate);
    EXPECT_LE(dry_path_difference.jacobian, serial_gate);
    EXPECT_LE(dry_path_difference.solved_state, serial_solved_gate);

    long double cross_squared = 0.0L;
    const auto n = baseline.dofs.size();
    for (std::size_t row = 0u; row < n; ++row) {
        const bool left_row = baseline.dofs[row].point[0] <= FE::Real{2.0};
        const bool right_row = baseline.dofs[row].point[0] >= FE::Real{4.0};
        ASSERT_TRUE(left_row || right_row);
        for (std::size_t column = 0u; column < n; ++column) {
            const bool left_column =
                baseline.dofs[column].point[0] <= FE::Real{2.0};
            const bool right_column =
                baseline.dofs[column].point[0] >= FE::Real{4.0};
            ASSERT_TRUE(left_column || right_column);
            if ((left_row && right_column) ||
                (right_row && left_column)) {
                const auto value = static_cast<long double>(
                    baseline.jacobian[row * n + column]);
                cross_squared += value * value;
            }
        }
    }
    const auto cross_norm =
        static_cast<FE::Real>(std::sqrt(cross_squared));
    const auto scaled_cross = cross_norm /
        (norm_floor + vectorL2Norm(baseline.jacobian));
    EXPECT_LE(scaled_cross, serial_gate);
    EXPECT_LE(baseline.dry_column_coupling_norm /
                  (norm_floor + vectorL2Norm(baseline.jacobian)),
              serial_gate);
    std::cout << std::setprecision(17)
              << "WP1_two_island_decoupling"
              << " scope=serial"
              << " retained_vertices=" << baseline.retained_vertices
              << " dry_velocity_constraints="
              << baseline.constrained_dry_velocity_dofs
              << " dry_pressure_constraints="
              << baseline.constrained_dry_pressure_dofs
              << " scaled_cross_jacobian=" << scaled_cross
              << " scaled_dry_path_residual_difference="
              << dry_path_difference.residual
              << " scaled_dry_path_jacobian_difference="
              << dry_path_difference.jacobian
              << " scaled_dry_path_solved_state_difference="
              << dry_path_difference.solved_state
              << " accepted_gate=" << serial_gate
              << " solved_state_gate=" << serial_solved_gate << '\n';
#endif
}

TEST(FreeSurfaceCutStability,
     ProductionSmallCutAggregationPreservesPartialWallDirichletPerComponent)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Partial-wall small-cut aggregation requires native mesh support.";
#else
    constexpr int left_marker = 27201;
    constexpr int right_marker = 27202;
    constexpr int bottom_marker = 27203;
    constexpr int top_marker = 27204;
    constexpr int interface_marker = 27205;
    constexpr std::string_view domain_id =
        "partial_slip_small_cut_component_precedence";
    const PlaneCutPosition cut{
        .label = "vertical_five_percent_active_sliver",
        .normal = {{FE::Real{1.0}, FE::Real{0.0}, FE::Real{0.0}}},
        .offset = FE::Real{0.05},
    };

    auto mesh = makePartialSlipSmallCutQuadStrip(
        cut, left_marker, right_marker, bottom_marker, top_marker);
    auto scalar_space = FE::spaces::SpaceFactory::create_h1(
        FE::ElementType::Quad4, /*order=*/1);
    auto velocity_space = FE::spaces::SpaceFactory::create_vector_h1(
        FE::ElementType::Quad4, /*order=*/1, /*components=*/2);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::Unknown,
    });
    system.addOperator("equations");
    const auto phi_state =
        FE::forms::StateField(phi, *scalar_space, "phi_partial_wall_owner");
    const auto eta =
        FE::forms::TestField(phi, *scalar_space, "eta_partial_wall_owner");
    (void)FE::systems::installFormulation(
        system,
        "equations",
        {phi},
        (FE::forms::dt(phi_state) * eta).dx());

    ns::IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = "u";
    options.pressure_field_name = "p";
    options.density = FE::Real{1.0};
    options.viscosity = FE::Real{0.01};
    options.enable_convection = false;
    options.enable_vms = true;
    for (const int marker : {left_marker, right_marker, top_marker}) {
        options.velocity_dirichlet.push_back(
            ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
                .boundary_marker = marker,
                .value = {FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}},
                .active_components = {true, true, false},
            });
    }
    // Impermeability is strong only in the y direction.  The bottom-wall x
    // trace intentionally remains free so the production aggregation
    // constraint, registered after the essential BC, must still support it.
    options.velocity_dirichlet.push_back(
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = bottom_marker,
            .value = {FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}},
            .active_components = {false, true, false},
        });

    ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary free_surface;
    free_surface.implementation =
        ns::FreeSurfaceImplementation::UnfittedLevelSet;
    free_surface.interface_marker = interface_marker;
    free_surface.level_set_field_name = "phi";
    free_surface.generated_interface_domain_id = std::string(domain_id);
    free_surface.level_set_isovalue = FE::Real{0.0};
    free_surface.active_domain =
        ns::FreeSurfaceActiveDomain::LevelSetNegative;
    free_surface.active_domain_method =
        ns::FreeSurfaceActiveDomainMethod::CutVolume;
    free_surface.external_pressure = FE::Real{0.0};
    free_surface.surface_tension = FE::Real{0.0};
    free_surface.use_level_set_curvature = false;
    free_surface.cut_cell_stabilization.enabled = false;
    // The physical velocity field is retained only on wet support.  Separate
    // auxiliary advection-velocity extension, when configured by an
    // application, is deliberately outside this momentum fixture.
    free_surface.small_cut_aggregation = true;
    options.free_surface.push_back(std::move(free_surface));

    ns::IncompressibleNavierStokesVMSModule module(
        velocity_space, scalar_space, options);
    module.registerOn(system);
    system.setup({});

    const auto velocity = system.findFieldByName("u");
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        FE::Real{0.0});
    setScalarVertexField(solution, system, phi, cut);

    FE::level_set::LevelSetGeneratedInterfaceOptions cut_options;
    cut_options.level_set_field_name = "phi";
    cut_options.domain_id = std::string(domain_id);
    cut_options.requested_interface_marker = interface_marker;
    cut_options.tolerance = FE::Real{1.0e-12};
    cut_options.quadrature_order = 2;
    cut_options.interface_quadrature_order = 2;
    cut_options.volume_quadrature_order = 2;
    FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto generated = lifecycle.build(system, cut_options, solution);
    ASSERT_TRUE(generated.success) << generated.diagnostic;
    ASSERT_EQ(generated.domain.cutCells().size(), 1u)
        << "the strip must expose exactly one small cut cell";

    auto context = std::make_shared<FE::assembly::CutIntegrationContext>();
    context->addGeneratedInterfaceDomain(generated.domain);
    system.setCutIntegrationContext(context);
    ASSERT_NO_THROW(system.rebuildConstraintState());

    const auto* velocity_entity_map =
        system.fieldDofHandler(velocity).getEntityDofMap();
    ASSERT_NE(velocity_entity_map, nullptr);
    const auto velocity_offset = system.fieldDofOffset(velocity);
    const auto velocity_dof_count =
        system.fieldDofHandler(velocity).getNumDofs();
    ASSERT_GT(velocity_offset, FE::GlobalIndex{0})
        << "the phi-first layout must exercise aggregation for a nonzero-offset "
           "field";
    const auto globalVelocityDofsAt = [&](FE::GlobalIndex vertex) {
        const auto local = velocity_entity_map->getVertexDofs(vertex);
        if (local.size() != 2u) {
            throw std::runtime_error(
                "partial-wall regression requires two velocity DOFs per vertex");
        }
        return std::array<FE::GlobalIndex, 2>{
            velocity_offset + local[0], velocity_offset + local[1]};
    };

    // Vertex 2=(1,0) belongs to the 5%-active cut cell and an inactive cell,
    // but to no full-active cell.  Its x component must be an AgFEM slave;
    // its y component must retain the homogeneous bottom-wall condition.
    const auto bottom_cut_dofs = globalVelocityDofsAt(/*vertex=*/2);
    const auto bottom_tangent =
        system.constraints().getConstraint(bottom_cut_dofs[0]);
    const auto bottom_normal =
        system.constraints().getConstraint(bottom_cut_dofs[1]);
    ASSERT_TRUE(bottom_tangent.has_value());
    EXPECT_FALSE(bottom_tangent->isDirichlet());
    EXPECT_FALSE(bottom_tangent->entries.empty())
        << "the normal-only wall marker must not suppress tangential aggregation";
    EXPECT_DOUBLE_EQ(bottom_tangent->inhomogeneity, 0.0);
    for (const auto& entry : bottom_tangent->entries) {
        EXPECT_GE(entry.master_dof, velocity_offset);
        EXPECT_LT(entry.master_dof, velocity_offset + velocity_dof_count);
    }
    ASSERT_TRUE(bottom_normal.has_value());
    EXPECT_TRUE(bottom_normal->isDirichlet());
    EXPECT_DOUBLE_EQ(bottom_normal->inhomogeneity, 0.0);

    // Vertex 6=(1,1) has the same unsupported cut support but lies on the
    // fully constrained top wall.  Both of its lines must remain strong.
    const auto top_cut_dofs = globalVelocityDofsAt(/*vertex=*/6);
    for (const auto dof : top_cut_dofs) {
        const auto line = system.constraints().getConstraint(dof);
        ASSERT_TRUE(line.has_value());
        EXPECT_TRUE(line->isDirichlet());
        EXPECT_DOUBLE_EQ(line->inhomogeneity, 0.0);
    }
#endif
}

TEST(FreeSurfaceCutStability,
     ConnectedDisconnectedAndRootlessFeaturesReportTopologyPolicy)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP()
        << "Active-cell topology telemetry requires native mesh support.";
#else
    using Disposition =
        FE::constraints::
            SmallCutAggregationActiveFeatureDisposition;
    using Feature =
        FE::constraints::SmallCutAggregationActiveFeatureReport;
    using Report =
        FE::constraints::SmallCutAggregationRefreshReport;

    struct ExpectedFeature {
        FE::GlobalIndex stable_id{FE::INVALID_GLOBAL_INDEX};
        std::uint64_t digest{0u};
        Disposition disposition{Disposition::Rootless};
        std::size_t cells{0u};
        std::size_t full_cells{0u};
        std::size_t cut_cells{0u};
        FE::Real retained_physical_volume{0.0};
    };
    struct ExpectedCase {
        std::string_view label{};
        std::array<FE::Real, 7> level_set{};
        std::vector<ExpectedFeature> features{};
        std::size_t rooted_features{0u};
        std::size_t rootless_features{0u};
        std::size_t candidate_vertices{0u};
        std::size_t rooted_candidate_vertices{0u};
        std::size_t rootless_candidate_vertices{0u};
        std::size_t pressure_aggregate_dofs{0u};
        std::size_t pressure_pinned_dofs{0u};
        FE::Real rootless_physical_volume{0.0};
        FE::Real active_physical_volume{0.0};
    };

    const ExpectedFeature left_rooted{
        .stable_id = 0,
        .digest = 590682968308805178ull,
        .disposition = Disposition::Rooted,
        .cells = 2u,
        .full_cells = 1u,
        .cut_cells = 1u,
        .retained_physical_volume = FE::Real{1.5},
    };
    const std::array<ExpectedCase, 3> cases = {{
        {
            .label = "connected_rooted",
            .level_set = {
                FE::Real{-1.0},
                FE::Real{-1.0},
                FE::Real{1.0},
                FE::Real{1.0},
                FE::Real{1.0},
                FE::Real{1.0},
                FE::Real{1.0},
            },
            .features = {left_rooted},
            .rooted_features = 1u,
            .rootless_features = 0u,
            .candidate_vertices = 2u,
            .rooted_candidate_vertices = 2u,
            .rootless_candidate_vertices = 0u,
            .pressure_aggregate_dofs = 2u,
            .pressure_pinned_dofs = 0u,
            .rootless_physical_volume = FE::Real{0.0},
            .active_physical_volume = FE::Real{1.5},
        },
        {
            .label = "disconnected_rooted",
            .level_set = {
                FE::Real{-1.0},
                FE::Real{-1.0},
                FE::Real{1.0},
                FE::Real{1.0},
                FE::Real{1.0},
                FE::Real{-1.0},
                FE::Real{-1.0},
            },
            .features = {
                left_rooted,
                ExpectedFeature{
                    .stable_id = 4,
                    .digest = 586861065889900642ull,
                    .disposition = Disposition::Rooted,
                    .cells = 2u,
                    .full_cells = 1u,
                    .cut_cells = 1u,
                    .retained_physical_volume = FE::Real{1.5},
                },
            },
            .rooted_features = 2u,
            .rootless_features = 0u,
            .candidate_vertices = 4u,
            .rooted_candidate_vertices = 4u,
            .rootless_candidate_vertices = 0u,
            .pressure_aggregate_dofs = 4u,
            .pressure_pinned_dofs = 0u,
            .rootless_physical_volume = FE::Real{0.0},
            .active_physical_volume = FE::Real{3.0},
        },
        {
            .label = "rooted_plus_rootless",
            .level_set = {
                FE::Real{-1.0},
                FE::Real{-1.0},
                FE::Real{1.0},
                FE::Real{1.0},
                FE::Real{-1.0},
                FE::Real{1.0},
                FE::Real{1.0},
            },
            .features = {
                left_rooted,
                ExpectedFeature{
                    .stable_id = 3,
                    .digest = 591645040983300578ull,
                    .disposition = Disposition::Rootless,
                    .cells = 2u,
                    .full_cells = 0u,
                    .cut_cells = 2u,
                    .retained_physical_volume = FE::Real{1.0},
                },
            },
            .rooted_features = 1u,
            .rootless_features = 1u,
            .candidate_vertices = 8u,
            .rooted_candidate_vertices = 2u,
            .rootless_candidate_vertices = 6u,
            .pressure_aggregate_dofs = 2u,
            .pressure_pinned_dofs = 6u,
            .rootless_physical_volume = FE::Real{1.0},
            .active_physical_volume = FE::Real{2.5},
        },
    }};

    const auto tolerance = [](FE::Real expected) {
        return FE::Real{1.0e-12} *
               std::max(FE::Real{1.0}, std::abs(expected));
    };
    const auto expect_report =
        [&](const Report& report,
            const ExpectedCase& expected,
            FE::FieldId field,
            std::size_t component_multiplier) {
            EXPECT_EQ(report.field, field);
            EXPECT_EQ(report.interface_marker, 27315);
            EXPECT_EQ(
                report.active_side,
                FE::geometry::CutIntegrationSide::Negative);
            EXPECT_EQ(
                report.canonical_active_feature_count,
                expected.features.size());
            EXPECT_EQ(
                report.canonical_rooted_active_feature_count,
                expected.rooted_features);
            EXPECT_EQ(
                report.canonical_rootless_active_feature_count,
                expected.rootless_features);
            EXPECT_EQ(
                report.canonical_candidate_vertices,
                expected.candidate_vertices);
            EXPECT_EQ(
                report.canonical_rooted_candidate_vertices,
                expected.rooted_candidate_vertices);
            EXPECT_EQ(
                report.canonical_rootless_candidate_vertices,
                expected.rootless_candidate_vertices);
            EXPECT_EQ(
                report.canonical_owned_aggregate_dofs,
                component_multiplier *
                    expected.pressure_aggregate_dofs);
            EXPECT_EQ(
                report.canonical_owned_pinned_dofs,
                component_multiplier *
                    expected.pressure_pinned_dofs);
            EXPECT_EQ(
                report.canonical_strong_suppressed_dofs, 0u);
            EXPECT_NEAR(
                report.canonical_rootless_active_physical_volume,
                expected.rootless_physical_volume,
                tolerance(expected.rootless_physical_volume));

            ASSERT_EQ(
                report.canonical_active_features.size(),
                expected.features.size());
            long double total_volume = 0.0L;
            long double rootless_volume = 0.0L;
            for (const auto& feature : expected.features) {
                const auto observed_it = std::find_if(
                    report.canonical_active_features.begin(),
                    report.canonical_active_features.end(),
                    [&](const auto& candidate) {
                        return candidate.stable_feature_id ==
                               feature.stable_id;
                    });
                ASSERT_NE(
                    observed_it,
                    report.canonical_active_features.end())
                    << "missing active-cell feature "
                    << feature.stable_id;
                const Feature& observed = *observed_it;
                EXPECT_EQ(
                    observed.stable_feature_id,
                    feature.stable_id);
                EXPECT_EQ(
                    observed.canonical_cell_gid_digest,
                    feature.digest);
                EXPECT_EQ(
                    observed.disposition,
                    feature.disposition);
                EXPECT_EQ(
                    observed.canonical_cell_count,
                    feature.cells);
                EXPECT_EQ(
                    observed.canonical_full_active_cell_count,
                    feature.full_cells);
                EXPECT_EQ(
                    observed.canonical_cut_cell_count,
                    feature.cut_cells);
                EXPECT_EQ(
                    observed.canonical_cell_count,
                    observed.canonical_full_active_cell_count +
                        observed.canonical_cut_cell_count);
                EXPECT_NEAR(
                    observed.canonical_retained_physical_volume,
                    feature.retained_physical_volume,
                    tolerance(
                        feature.retained_physical_volume));
                total_volume += static_cast<long double>(
                    observed.canonical_retained_physical_volume);
                if (observed.disposition ==
                    Disposition::Rootless) {
                    rootless_volume += static_cast<long double>(
                        observed.canonical_retained_physical_volume);
                }
            }
            EXPECT_NEAR(
                static_cast<FE::Real>(total_volume),
                expected.active_physical_volume,
                tolerance(expected.active_physical_volume));
            EXPECT_NEAR(
                static_cast<FE::Real>(rootless_volume),
                expected.rootless_physical_volume,
                tolerance(expected.rootless_physical_volume));
        };

    std::size_t observed_case_count = 0u;
    std::size_t observed_active_features = 0u;
    std::size_t observed_rooted_features = 0u;
    std::size_t observed_rootless_features = 0u;
    std::size_t observed_velocity_pressure_mismatch_count = 0u;
    FE::Real observed_rootless_physical_volume = 0.0;
    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.label);
        const auto sample =
            assembleSerialActiveCellTopologySample(
                expected.level_set);
        expect_report(
            sample.velocity_report,
            expected,
            sample.velocity,
            /*component_multiplier=*/2u);
        expect_report(
            sample.pressure_report,
            expected,
            sample.pressure,
            /*component_multiplier=*/1u);
        EXPECT_NE(
            sample.velocity,
            sample.pressure);
        EXPECT_NEAR(
            sample.assembled_active_physical_volume,
            expected.active_physical_volume,
            tolerance(expected.active_physical_volume));

        bool velocity_pressure_match =
            sample.velocity_report
                    .canonical_active_feature_count ==
                sample.pressure_report
                    .canonical_active_feature_count &&
            sample.velocity_report
                    .canonical_rooted_active_feature_count ==
                sample.pressure_report
                    .canonical_rooted_active_feature_count &&
            sample.velocity_report
                    .canonical_rootless_active_feature_count ==
                sample.pressure_report
                    .canonical_rootless_active_feature_count &&
            sample.velocity_report
                    .canonical_active_features.size() ==
                sample.pressure_report
                    .canonical_active_features.size();
        EXPECT_EQ(
            sample.velocity_report.canonical_active_features.size(),
            sample.pressure_report.canonical_active_features.size());
        for (const auto& pressure_feature :
             sample.pressure_report.canonical_active_features) {
            const auto velocity_it = std::find_if(
                sample.velocity_report
                    .canonical_active_features.begin(),
                sample.velocity_report
                    .canonical_active_features.end(),
                [&](const auto& candidate) {
                    return candidate.stable_feature_id ==
                           pressure_feature.stable_feature_id;
                });
            if (velocity_it ==
                sample.velocity_report
                    .canonical_active_features.end()) {
                velocity_pressure_match = false;
                ADD_FAILURE()
                    << "velocity report is missing active-cell feature "
                    << pressure_feature.stable_feature_id;
                continue;
            }
            const auto& velocity_feature = *velocity_it;
            EXPECT_EQ(
                velocity_feature.stable_feature_id,
                pressure_feature.stable_feature_id);
            EXPECT_EQ(
                velocity_feature.canonical_cell_gid_digest,
                pressure_feature.canonical_cell_gid_digest);
            EXPECT_EQ(
                velocity_feature.disposition,
                pressure_feature.disposition);
            EXPECT_EQ(
                velocity_feature.canonical_cell_count,
                pressure_feature.canonical_cell_count);
            EXPECT_EQ(
                velocity_feature.canonical_full_active_cell_count,
                pressure_feature.canonical_full_active_cell_count);
            EXPECT_EQ(
                velocity_feature.canonical_cut_cell_count,
                pressure_feature.canonical_cut_cell_count);
            EXPECT_NEAR(
                velocity_feature
                    .canonical_retained_physical_volume,
                pressure_feature
                    .canonical_retained_physical_volume,
                tolerance(
                    pressure_feature
                        .canonical_retained_physical_volume));
            velocity_pressure_match =
                velocity_pressure_match &&
                velocity_feature.canonical_cell_gid_digest ==
                    pressure_feature.canonical_cell_gid_digest &&
                velocity_feature.disposition ==
                    pressure_feature.disposition &&
                velocity_feature.canonical_cell_count ==
                    pressure_feature.canonical_cell_count &&
                velocity_feature.canonical_full_active_cell_count ==
                    pressure_feature
                        .canonical_full_active_cell_count &&
                velocity_feature.canonical_cut_cell_count ==
                    pressure_feature.canonical_cut_cell_count &&
                std::abs(
                    velocity_feature
                            .canonical_retained_physical_volume -
                    pressure_feature
                            .canonical_retained_physical_volume) <=
                    tolerance(
                        pressure_feature
                            .canonical_retained_physical_volume);
        }
        if (!velocity_pressure_match) {
            ++observed_velocity_pressure_mismatch_count;
        }

        ++observed_case_count;
        observed_active_features +=
            sample.pressure_report
                .canonical_active_feature_count;
        observed_rooted_features +=
            sample.pressure_report
                .canonical_rooted_active_feature_count;
        observed_rootless_features +=
            sample.pressure_report
                .canonical_rootless_active_feature_count;
        observed_rootless_physical_volume +=
            sample.pressure_report
                .canonical_rootless_active_physical_volume;
    }

    RecordProperty(
        "wp7_active_cell_topology_case_count",
        observed_case_count);
    RecordProperty(
        "wp7_active_cell_topology_feature_count",
        observed_active_features);
    RecordProperty(
        "wp7_active_cell_topology_rooted_feature_count",
        observed_rooted_features);
    RecordProperty(
        "wp7_active_cell_topology_rootless_feature_count",
        observed_rootless_features);
    RecordProperty(
        "wp7_active_cell_topology_rootless_retained_physical_volume",
        realPropertyValue(
            observed_rootless_physical_volume));
    RecordProperty(
        "wp7_active_cell_topology_velocity_pressure_mismatch_count",
        observed_velocity_pressure_mismatch_count);
#endif
}

TEST(FreeSurfaceCutStability,
     SelectedCombinedAggregateAndPressureStabilizationContractIsExplicit)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Cut-stability method contract requires native mesh support.";
#else
    const auto velocity_space =
        FE::spaces::SpaceFactory::create_vector_h1(
            FE::ElementType::Tetra4, /*order=*/1, /*components=*/3);
    const auto pressure_space =
        FE::spaces::SpaceFactory::create_h1(
            FE::ElementType::Tetra4, /*order=*/1);
    constexpr std::array<int, 1> supported_polynomial_orders = {{1}};
    const auto velocity_order = velocity_space->polynomial_order();
    const auto pressure_order = pressure_space->polynomial_order();
    const auto is_supported_order = [&](int order) {
        return std::find(
                   supported_polynomial_orders.begin(),
                   supported_polynomial_orders.end(),
                   order) != supported_polynomial_orders.end();
    };
    EXPECT_TRUE(is_supported_order(velocity_order));
    EXPECT_TRUE(is_supported_order(pressure_order));
    EXPECT_EQ(velocity_order, pressure_order);

    const auto options =
        stabilityOptions(27017, "wp7_selected_combined_method");
    ASSERT_EQ(options.free_surface.size(), 1u);
    const auto& free_surface = options.free_surface.front();

    EXPECT_EQ(free_surface.implementation,
              ns::FreeSurfaceImplementation::UnfittedLevelSet);
    EXPECT_EQ(free_surface.active_domain,
              ns::FreeSurfaceActiveDomain::LevelSetNegative);
    EXPECT_EQ(free_surface.active_domain_method,
              ns::FreeSurfaceActiveDomainMethod::CutVolume);
    EXPECT_TRUE(options.enable_vms);
    EXPECT_FALSE(options.enable_convection);
    EXPECT_TRUE(free_surface.small_cut_aggregation);
    EXPECT_TRUE(free_surface.cut_cell_stabilization.enabled);
    EXPECT_EQ(
        free_surface.cut_cell_stabilization.pressure_policy,
        ns::FreeSurfacePressureStabilizationPolicy::Enabled);
    ASSERT_TRUE(std::holds_alternative<FE::Real>(
        free_surface.cut_cell_stabilization
            .pressure_gradient_penalty));
    const auto pressure_gradient_penalty = std::get<FE::Real>(
        free_surface.cut_cell_stabilization
            .pressure_gradient_penalty);
    EXPECT_DOUBLE_EQ(
        pressure_gradient_penalty,
        FE::Real{1.0});
    EXPECT_TRUE(
        free_surface.cut_cell_stabilization.use_cut_metadata_scale);
    ASSERT_TRUE(
        free_surface.cut_cell_stabilization
            .cut_metadata_scale_cap.has_value());
    const auto cut_metadata_scale_cap =
        free_surface.cut_cell_stabilization
            .cut_metadata_scale_cap.value();
    EXPECT_DOUBLE_EQ(
        cut_metadata_scale_cap,
        FE::Real{100.0});

    const auto equal_order =
        velocity_order == pressure_order ? 1 : 0;
    const auto vms_pspg_enabled = options.enable_vms ? 1 : 0;
    const auto pressure_ghost_enabled =
        free_surface.cut_cell_stabilization.enabled &&
                free_surface.cut_cell_stabilization.pressure_policy ==
                    ns::FreeSurfacePressureStabilizationPolicy::Enabled
            ? 1
            : 0;
    const auto small_cut_aggregation_enabled =
        free_surface.small_cut_aggregation ? 1 : 0;
    RecordProperty(
        "wp7_selected_supported_polynomial_order_count",
        supported_polynomial_orders.size());
    RecordProperty(
        "wp7_selected_equal_order_velocity_pressure", equal_order);
    RecordProperty("wp7_selected_vms_pspg_enabled", vms_pspg_enabled);
    RecordProperty(
        "wp7_selected_pressure_ghost_enabled", pressure_ghost_enabled);
    RecordProperty(
        "wp7_selected_small_cut_aggregation_enabled",
        small_cut_aggregation_enabled);
    RecordProperty("wp7_selected_velocity_ghost_enabled", 0);
    RecordProperty(
        "wp7_selected_pressure_gradient_penalty",
        realPropertyValue(pressure_gradient_penalty));
    RecordProperty(
        "wp7_selected_cut_metadata_scale_cap",
        realPropertyValue(cut_metadata_scale_cap));
#endif
}

TEST(FreeSurfaceCutStability,
     ExactTargetFractionGeometryCoversAxisObliqueAndThreeHLevels)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Exact target-fraction sweep requires native mesh support.";
#else
    constexpr std::array<FE::Real, 7> fractions = {{
        FE::Real{1.0e-8},
        FE::Real{1.0e-6},
        FE::Real{1.0e-4},
        FE::Real{1.0e-2},
        FE::Real{0.1},
        FE::Real{0.25},
        FE::Real{0.49},
    }};
    constexpr std::array<std::array<FE::Real, 3>, 2> orientations = {{
        {{1.0, 0.0, 0.0}},
        {{1.0, 0.73, 0.41}},
    }};
    constexpr std::array<FE::Real, 3> h_levels = {{
        FE::Real{1.0},
        FE::Real{0.5},
        FE::Real{0.25},
    }};

    FE::Real maximum_fraction_relative_error = 0.0;
    FE::Real maximum_volume_relative_error = 0.0;
    std::size_t case_count = 0u;
    std::size_t fallback_count = 0u;
    for (std::size_t orientation = 0u;
         orientation < orientations.size();
         ++orientation) {
        for (const auto h : h_levels) {
            for (const auto fraction : fractions) {
                SCOPED_TRACE(
                    std::string("orientation=") +
                    std::to_string(orientation) +
                    " h=" + std::to_string(h) +
                    " fraction=" + std::to_string(fraction));
                const auto sample = runTargetFractionGeometrySample(
                    fraction, orientations[orientation], h);
                EXPECT_EQ(sample.cut_cells, 1u);
                EXPECT_EQ(sample.backend_fallback_cells, 0u);
                EXPECT_GT(sample.generated_retained_volume, FE::Real{0.0});
                EXPECT_NEAR(
                    sample.generated_fraction,
                    sample.target_fraction,
                    FE::Real{2.0e-11} *
                        std::max(FE::Real{1.0e-8},
                                 sample.target_fraction));
                EXPECT_NEAR(
                    sample.generated_retained_volume,
                    sample.expected_retained_volume,
                    FE::Real{2.0e-11} *
                        std::max(FE::Real{1.0e-12},
                                 sample.expected_retained_volume));

                maximum_fraction_relative_error = std::max(
                    maximum_fraction_relative_error,
                    std::abs(
                        sample.generated_fraction -
                        sample.target_fraction) /
                        sample.target_fraction);
                maximum_volume_relative_error = std::max(
                    maximum_volume_relative_error,
                    std::abs(
                        sample.generated_retained_volume -
                        sample.expected_retained_volume) /
                        sample.expected_retained_volume);
                fallback_count += sample.backend_fallback_cells;
                ++case_count;
            }
        }
    }

    EXPECT_EQ(case_count,
              fractions.size() * orientations.size() * h_levels.size());
    EXPECT_EQ(fallback_count, 0u);
    EXPECT_LE(maximum_fraction_relative_error, FE::Real{2.0e-11});
    EXPECT_LE(maximum_volume_relative_error, FE::Real{2.0e-11});
    RecordProperty("wp7_exact_fraction_case_count", case_count);
    RecordProperty(
        "wp7_exact_fraction_orientation_count", orientations.size());
    RecordProperty("wp7_exact_fraction_h_level_count", h_levels.size());
    RecordProperty("wp7_exact_fraction_backend_fallback_count",
                   fallback_count);
    RecordProperty(
        "wp7_exact_fraction_maximum_relative_error",
        realPropertyValue(maximum_fraction_relative_error));
    RecordProperty(
        "wp7_exact_fraction_maximum_volume_relative_error",
        realPropertyValue(maximum_volume_relative_error));
#endif
}

TEST(FreeSurfaceCutStability,
     FrozenFractionOrientationRefinementRegimeMatrixRecordsConditionAndIterations)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Full serial cut-stability matrix requires native mesh support.";
#else
    const ScopedEnvVar pressure_diagnostics(
        "SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC", "1");
    constexpr std::array<FE::Real, 7> fractions = {{
        FE::Real{1.0e-8},
        FE::Real{1.0e-6},
        FE::Real{1.0e-4},
        FE::Real{1.0e-2},
        FE::Real{0.1},
        FE::Real{0.25},
        FE::Real{0.49},
    }};
    constexpr std::array<std::array<FE::Real, 3>, 2> orientations = {{
        {{1.0, 0.0, 0.0}},
        {{1.0, 0.73, 0.41}},
    }};
    constexpr std::array<int, 3> resolutions = {{2, 3, 4}};
    constexpr std::array<StabilityRegime, 3> regimes = {{
        StabilityRegime{
            .id = "viscous",
            .density = FE::Real{1.0},
            .viscosity = FE::Real{1.0},
            .dt = FE::Real{0.1},
            .convection = false,
            .advective_speed = FE::Real{0.0},
        },
        StabilityRegime{
            .id = "transient",
            .density = FE::Real{1.0},
            .viscosity = FE::Real{0.01},
            .dt = FE::Real{0.001},
            .convection = false,
            .advective_speed = FE::Real{0.0},
        },
        StabilityRegime{
            .id = "advection",
            .density = FE::Real{1.0},
            .viscosity = FE::Real{0.001},
            .dt = FE::Real{0.1},
            .convection = true,
            .advective_speed = FE::Real{1.0},
        },
    }};

    std::size_t case_count = 0u;
    std::size_t aggregation_case_count = 0u;
    std::size_t rootless_case_count = 0u;
    std::size_t krylov_nonconvergence_count = 0u;
    std::size_t krylov_breakdown_count = 0u;
    std::size_t maximum_krylov_iterations = 0u;
    std::size_t maximum_krylov_diagonal_fallbacks = 0u;
    FE::Real maximum_fraction_relative_error = 0.0;
    FE::Real minimum_condition =
        std::numeric_limits<FE::Real>::infinity();
    FE::Real maximum_condition = 0.0;
    FE::Real maximum_krylov_relative_residual = 0.0;
    FE::Real maximum_krylov_relative_solution_error = 0.0;
    FE::Real minimum_pressure_control =
        std::numeric_limits<FE::Real>::infinity();

    for (const auto resolution : resolutions) {
        for (std::size_t orientation = 0u;
             orientation < orientations.size();
             ++orientation) {
            std::array<TargetStructuredCut, fractions.size()> cuts;
            for (std::size_t fraction = 0u;
                 fraction < fractions.size();
                 ++fraction) {
                cuts[fraction] = makeTargetStructuredCut(
                    fractions[fraction],
                    orientations[orientation],
                    resolution,
                    std::string("matrix_fraction_") +
                        std::to_string(fraction));
            }
            for (const auto& regime : regimes) {
                SCOPED_TRACE(
                    std::string("resolution=") +
                    std::to_string(resolution) +
                    " orientation=" +
                    std::to_string(orientation) +
                    " regime=" + std::string(regime.id));
                PersistentStabilityProblem problem(
                    cuts.front().cut, resolution, regime);
                for (std::size_t fraction = 0u;
                     fraction < fractions.size();
                     ++fraction) {
                    SCOPED_TRACE(
                        std::string("fraction=") +
                        std::to_string(fractions[fraction]));
                    const auto sample = problem.evaluate(
                        cuts[fraction].cut,
                        cuts[fraction].designated_parent_cell);
                    ASSERT_TRUE(std::isfinite(
                        sample.designated_cut_fraction));
                    const auto fraction_relative_error =
                        std::abs(
                            sample.designated_cut_fraction -
                            fractions[fraction]) /
                        fractions[fraction];
                    maximum_fraction_relative_error = std::max(
                        maximum_fraction_relative_error,
                        fraction_relative_error);
                    EXPECT_LE(fraction_relative_error, FE::Real{5.0e-8});
                    EXPECT_GT(sample.cut_cells, 0u);
                    EXPECT_GT(sample.cut_adjacent_facets, 0u);
                    EXPECT_GT(
                        sample.backend_volume_quadrature_points, 0u);
                    EXPECT_EQ(sample.backend_fallback_cells, 0u);
                    EXPECT_TRUE(sample.pressure_natural_traction_anchor);
                    EXPECT_TRUE(
                        sample.pressure_anchor_has_no_gauge_enforcement);
                    EXPECT_EQ(sample.zero_free_pressure_rows, 0u);
                    EXPECT_GT(sample.pressure_ghost_norm, FE::Real{0.0});
                    EXPECT_GT(
                        sample.pspg_pressure_gradient_norm, FE::Real{0.0});
                    EXPECT_EQ(
                        sample.equilibrated_rank,
                        sample.equilibrated_size);
                    EXPECT_TRUE(std::isfinite(
                        sample.equilibrated_condition_inf));
                    EXPECT_GT(
                        sample.equilibrated_condition_inf, FE::Real{0.0});
                    EXPECT_GT(
                        sample.pressure_control
                            .stabilized_pressure_control,
                        FE::Real{0.0});
                    if (sample.pressure_constraints.master_bearing > 0u) {
                        ++aggregation_case_count;
                    }
                    if (sample.pressure_constraints.homogeneous_pins > 0u) {
                        ++rootless_case_count;
                    }

                    if (!sample.krylov.converged) {
                        ++krylov_nonconvergence_count;
                    }
                    if (sample.krylov.breakdown) {
                        ++krylov_breakdown_count;
                    }
                    EXPECT_TRUE(sample.krylov.converged)
                        << "iterations=" << sample.krylov.iterations
                        << " limit=" << sample.krylov.iteration_limit
                        << " residual="
                        << sample.krylov.relative_residual;
                    EXPECT_FALSE(sample.krylov.breakdown);
                    EXPECT_LE(
                        sample.krylov.relative_residual,
                        FE::Real{2.0e-9});
                    EXPECT_TRUE(std::isfinite(
                        sample.krylov.relative_solution_error));
                    maximum_krylov_iterations = std::max(
                        maximum_krylov_iterations,
                        sample.krylov.iterations);
                    maximum_krylov_diagonal_fallbacks = std::max(
                        maximum_krylov_diagonal_fallbacks,
                        sample.krylov.diagonal_fallback_count);
                    maximum_krylov_relative_residual = std::max(
                        maximum_krylov_relative_residual,
                        sample.krylov.relative_residual);
                    maximum_krylov_relative_solution_error = std::max(
                        maximum_krylov_relative_solution_error,
                        sample.krylov.relative_solution_error);
                    minimum_condition = std::min(
                        minimum_condition,
                        sample.equilibrated_condition_inf);
                    maximum_condition = std::max(
                        maximum_condition,
                        sample.equilibrated_condition_inf);
                    minimum_pressure_control = std::min(
                        minimum_pressure_control,
                        sample.pressure_control
                            .stabilized_pressure_control);
                    ++case_count;
                }
            }
        }
    }

    const auto expected_case_count =
        fractions.size() * orientations.size() *
        resolutions.size() * regimes.size();
    ASSERT_EQ(case_count, expected_case_count);
    EXPECT_GT(aggregation_case_count, 0u);
    EXPECT_EQ(krylov_nonconvergence_count, 0u);
    EXPECT_EQ(krylov_breakdown_count, 0u);
    EXPECT_TRUE(std::isfinite(minimum_condition));
    EXPECT_TRUE(std::isfinite(maximum_condition));
    EXPECT_GE(maximum_condition, minimum_condition);
    EXPECT_GT(minimum_pressure_control, FE::Real{0.0});
    RecordProperty("wp7_full_serial_case_count", case_count);
    RecordProperty("wp7_full_serial_fraction_count", fractions.size());
    RecordProperty(
        "wp7_full_serial_orientation_count", orientations.size());
    RecordProperty("wp7_full_serial_h_level_count", resolutions.size());
    RecordProperty("wp7_full_serial_regime_count", regimes.size());
    RecordProperty(
        "wp7_full_serial_aggregation_case_count",
        aggregation_case_count);
    RecordProperty(
        "wp7_full_serial_rootless_case_count", rootless_case_count);
    RecordProperty(
        "wp7_full_serial_krylov_nonconvergence_count",
        krylov_nonconvergence_count);
    RecordProperty(
        "wp7_full_serial_krylov_breakdown_count",
        krylov_breakdown_count);
    RecordProperty(
        "wp7_full_serial_maximum_krylov_iterations",
        maximum_krylov_iterations);
    RecordProperty(
        "wp7_full_serial_maximum_krylov_diagonal_fallbacks",
        maximum_krylov_diagonal_fallbacks);
    RecordProperty(
        "wp7_full_serial_maximum_fraction_relative_error",
        realPropertyValue(maximum_fraction_relative_error));
    RecordProperty(
        "wp7_full_serial_minimum_condition",
        realPropertyValue(minimum_condition));
    RecordProperty(
        "wp7_full_serial_maximum_condition",
        realPropertyValue(maximum_condition));
    RecordProperty(
        "wp7_full_serial_condition_spread",
        realPropertyValue(maximum_condition / minimum_condition));
    RecordProperty(
        "wp7_full_serial_minimum_pressure_control",
        realPropertyValue(minimum_pressure_control));
    RecordProperty(
        "wp7_full_serial_maximum_krylov_relative_residual",
        realPropertyValue(maximum_krylov_relative_residual));
    RecordProperty(
        "wp7_full_serial_maximum_krylov_relative_solution_error",
        realPropertyValue(maximum_krylov_relative_solution_error));
    std::cout << std::setprecision(17)
              << "WP7_full_serial_matrix"
              << " cases=" << case_count
              << " aggregation_cases=" << aggregation_case_count
              << " rootless_cases=" << rootless_case_count
              << " minimum_condition=" << minimum_condition
              << " maximum_condition=" << maximum_condition
              << " condition_spread="
              << maximum_condition / minimum_condition
              << " minimum_pressure_control="
              << minimum_pressure_control
              << " maximum_krylov_iterations="
              << maximum_krylov_iterations
              << " maximum_krylov_diagonal_fallbacks="
              << maximum_krylov_diagonal_fallbacks
              << " maximum_krylov_relative_residual="
              << maximum_krylov_relative_residual
              << " maximum_krylov_relative_solution_error="
              << maximum_krylov_relative_solution_error
              << " scope=finite_p1_matrix_not_uniform_theorem"
              << '\n';
#endif
}

TEST(FreeSurfaceCutStability,
     ManufacturedAffineQ1YoungLaplaceAndContactAngleBalanceToRoundoff)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Manufactured cut balance requires native mesh support.";
#else
    constexpr FE::Real theta =
        FE::Real{1.04719755119659774615421446109316763};
    const auto balanced = runManufacturedAffineQ1Balance(
        theta, theta, FE::Real{1.0});

    EXPECT_GT(balanced.interface_fragments, 0u);
    EXPECT_EQ(balanced.contact_fragments, 1u)
        << "the affine interface must generate one sharp bottom-wall contact point";
    EXPECT_GT(balanced.free_velocity_dofs, 0u)
        << "the tangential wet-wall test space was accidentally eliminated";
    EXPECT_NEAR(balanced.physical_active_area,
                balanced.expected_active_area,
                FE::Real{2.0e-12});
    EXPECT_LT(balanced.maximum_q1_mixed_coefficient, FE::Real{2.0e-15})
        << "the manufactured level set is not locally affine/separable Q1";
    EXPECT_LT(balanced.maximum_interface_normal_error, FE::Real{2.0e-12})
        << "LinearCorner geometry and the affine Q1 operator normal disagree";
    EXPECT_LT(balanced.maximum_contact_cosine_error, FE::Real{2.0e-12})
        << "the generated wall contact does not satisfy the exact operator angle";
    EXPECT_LT(balanced.unconstrained_residual_norm, FE::Real{5.0e-11})
        << "constant pressure, prescribed constant curvature, capillary traction, "
           "and the dynamic contact law do not balance";
    EXPECT_LT(balanced.repeated_residual_difference_norm, FE::Real{1.0e-15})
        << "the stationary current/previous state is not one-step invariant";
    RecordProperty("wp7_balance_interface_fragment_count",
                   balanced.interface_fragments);
    RecordProperty("wp7_balance_contact_fragment_count",
                   balanced.contact_fragments);
    RecordProperty(
        "wp7_balance_maximum_interface_normal_error",
        realPropertyValue(balanced.maximum_interface_normal_error));
    RecordProperty(
        "wp7_balance_maximum_contact_cosine_error",
        realPropertyValue(balanced.maximum_contact_cosine_error));
    RecordProperty(
        "wp7_balance_unconstrained_residual_norm",
        realPropertyValue(balanced.unconstrained_residual_norm));
    RecordProperty(
        "wp7_balance_repeated_residual_difference_norm",
        realPropertyValue(
            balanced.repeated_residual_difference_norm));

    // Both controls retain the same production cut-volume/interface/contact
    // assembly.  They ensure the near-zero result is a resolved balance rather
    // than an empty or constrained-away momentum operator.
    const auto wrong_pressure = runManufacturedAffineQ1Balance(
        theta, theta, FE::Real{1.03});
    EXPECT_GT(wrong_pressure.unconstrained_residual_norm, FE::Real{1.0e-6});

    const auto wrong_angle = runManufacturedAffineQ1Balance(
        theta + FE::Real{0.08}, theta, FE::Real{1.0});
    EXPECT_GT(wrong_angle.maximum_contact_cosine_error, FE::Real{1.0e-3});
    EXPECT_GT(wrong_angle.unconstrained_residual_norm, FE::Real{1.0e-6});
#endif
}

TEST(FreeSurfaceCutStability,
     FixedTetraMeshGenericAndNearFeatureSweepHasNoFreePressureNullRows)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Fixed cut-position stability sweep requires native mesh support.";
#else
    // Installs the module's production diagnostic operators.  They are
    // assembled below to prove PSPG and pressure ghost terms contribute
    // numerically on the exact same constrained state as the coupled matrix.
    const ScopedEnvVar pressure_diagnostics(
        "SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC", "1");

    const std::vector<PlaneCutPosition> positions = {
        {"generic", {{1.0, 0.73, 0.41}}, 2.47},
        {"near_vertex_eps_1e-1", {{1.0, 1.0, 1.0}}, 3.1},
        {"near_vertex_eps_5e-2", {{1.0, 1.0, 1.0}}, 3.05},
        {"near_vertex_eps_2e-2", {{1.0, 1.0, 1.0}}, 3.02},
        {"near_edge_eps_1e-2", {{1.0, 1.0, 0.0}}, 2.01},
        {"near_edge_eps_3e-3", {{1.0, 1.0, 0.0}}, 2.003},
        {"near_edge_eps_1e-3", {{1.0, 1.0, 0.0}}, 2.001},
        {"near_face_eps_1e-3", {{1.0, 0.0, 0.0}}, 1.001},
        {"near_face_eps_1e-4", {{1.0, 0.0, 0.0}}, 1.0001},
        {"near_face_eps_1e-5", {{1.0, 0.0, 0.0}}, 1.00001},
    };

    std::vector<StabilitySample> samples;
    samples.reserve(positions.size());
    for (const auto& position : positions) {
        SCOPED_TRACE(position.label);
        samples.push_back(runStabilitySample(position));
        const auto& sample = samples.back();

        EXPECT_GT(sample.cut_cells, 0u);
        EXPECT_GT(sample.cut_adjacent_facets, 0u);
        EXPECT_GT(sample.backend_volume_quadrature_points, 0u)
            << "generated cut-volume backend was not numerically exercised";
        EXPECT_EQ(sample.backend_fallback_cells, 0u)
            << "generated cut-volume backend unexpectedly fell back";
        EXPECT_TRUE(sample.pressure_natural_traction_anchor)
            << "CutVolume free surface did not register its pressure anchor";
        EXPECT_TRUE(sample.pressure_anchor_has_no_gauge_enforcement)
            << "absolute pressure anchor unexpectedly installed a gauge constraint";
        EXPECT_GT(sample.physical_active_volume, FE::Real{0.0});
        EXPECT_LE(sample.physical_active_volume,
                  FE::Real{8.0} + FE::Real{1.0e-12});
        EXPECT_GT(sample.velocity_constraints.master_bearing, 0u)
            << "aggregation did not remove any velocity cut-band topology";
        EXPECT_GT(sample.pressure_constraints.master_bearing, 0u)
            << "aggregation did not remove any pressure cut-band topology";
        EXPECT_EQ(sample.pressure_constraints.master_bearing,
                  sample.pressure_aggregation.master_bearing_lines);
        EXPECT_LE(
            sample.pressure_aggregation.maximum_partition_of_unity_error,
            FE::Real{1.0e-10} *
                std::max(FE::Real{1.0},
                         sample.pressure_aggregation.maximum_weight_l1));
        EXPECT_LE(sample.pressure_aggregation.maximum_inhomogeneity,
                  FE::Real{1.0e-14});
        EXPECT_LE(sample.pressure_aggregation.maximum_weight_l1,
                  FE::Real{3.0} + FE::Real{1.0e-12});
        EXPECT_LE(
            sample.pressure_aggregation
                .maximum_slave_master_distance_over_h,
            FE::Real{3.5});
        EXPECT_EQ(sample.velocity_constraints.master_bearing,
                  3u * sample.pressure_constraints.master_bearing)
            << "velocity/pressure aggregation topology is inconsistent";
        EXPECT_EQ(sample.velocity_constraints.homogeneous_pins,
                  3u * sample.pressure_constraints.homogeneous_pins)
            << "pressure has a homogeneous pin beyond matching rootless-island removal";
        EXPECT_EQ(sample.zero_free_pressure_rows, 0u);
        EXPECT_GT(sample.pressure_ghost_norm, FE::Real{1.0e-14});
        EXPECT_GT(sample.pspg_pressure_gradient_norm, FE::Real{1.0e-14});
        EXPECT_EQ(sample.equilibrated_rank, sample.equilibrated_size);
        EXPECT_TRUE(std::isfinite(sample.equilibrated_condition_inf));
        EXPECT_GT(sample.equilibrated_condition_inf, FE::Real{0.0});
        EXPECT_EQ(sample.pressure_control.pressure_dimension,
                  sample.free_pressure_dofs);
        EXPECT_LE(sample.pressure_control.velocity_block_relative_skew,
                  FE::Real{1.0e-10});
        EXPECT_LE(
            sample.pressure_control
                .pressure_gradient_adjoint_relative_defect,
            FE::Real{1.0e-10});
        EXPECT_GT(sample.pressure_control.stabilized_pressure_control,
                  FE::Real{0.0});
        EXPECT_GE(sample.pressure_control.stabilized_pressure_control,
                  FE::Real{0.45});
    }

    ASSERT_FALSE(samples.empty());
    EXPECT_EQ(samples.front().pruned_volume_rules, 0u);
    EXPECT_GT(samples.back().pruned_volume_rules, 0u)
        << "the near-feature sweep did not exercise production tiny-sliver pruning";
    EXPECT_EQ(samples.front().pressure_constraints.homogeneous_pins, 0u)
        << "the natural-traction pressure anchor must not create a gauge pin";
    EXPECT_GT(samples.back().velocity_constraints.homogeneous_pins, 0u);
    EXPECT_GT(samples.back().pressure_constraints.homogeneous_pins, 0u)
        << "the smallest cut did not exercise rootless-island removal";

    FE::Real min_condition = std::numeric_limits<FE::Real>::infinity();
    FE::Real max_condition = 0.0;
    FE::Real min_stabilized_control =
        std::numeric_limits<FE::Real>::infinity();
    FE::Real max_stabilized_control = 0.0;
    FE::Real max_aggregate_weight_l1 = 0.0;
    FE::Real max_aggregate_reach_over_h = 0.0;
    for (const auto& sample : samples) {
        min_condition = std::min(
            min_condition, sample.equilibrated_condition_inf);
        max_condition = std::max(
            max_condition, sample.equilibrated_condition_inf);
        min_stabilized_control = std::min(
            min_stabilized_control,
            sample.pressure_control.stabilized_pressure_control);
        max_stabilized_control = std::max(
            max_stabilized_control,
            sample.pressure_control.stabilized_pressure_control);
        max_aggregate_weight_l1 = std::max(
            max_aggregate_weight_l1,
            sample.pressure_aggregation.maximum_weight_l1);
        max_aggregate_reach_over_h = std::max(
            max_aggregate_reach_over_h,
            sample.pressure_aggregation
                .maximum_slave_master_distance_over_h);
        std::cout << std::setprecision(12)
                  << "FS14_cut_sweep"
                  << " position=" << sample.label
                  << " min_active_cut_fraction="
                  << sample.minimum_active_cut_fraction
                  << " reference_active_volume="
                  << sample.reference_active_volume
                  << " physical_active_volume="
                  << sample.physical_active_volume
                  << " cut_cells=" << sample.cut_cells
                  << " cut_adjacent_facets=" << sample.cut_adjacent_facets
                  << " pruned_volume_rules=" << sample.pruned_volume_rules
                  << " backend_volume_quadrature_points="
                  << sample.backend_volume_quadrature_points
                  << " backend_fallback_cells="
                  << sample.backend_fallback_cells
                  << " pressure_natural_traction_anchor="
                  << (sample.pressure_natural_traction_anchor ? 1 : 0)
                  << " pressure_gauge_enforcement="
                  << (sample.pressure_anchor_has_no_gauge_enforcement ? 0 : 1)
                  << " velocity_aggregate_slaves="
                  << sample.velocity_constraints.master_bearing
                  << " velocity_homogeneous_pins="
                  << sample.velocity_constraints.homogeneous_pins
                  << " pressure_aggregate_slaves="
                  << sample.pressure_constraints.master_bearing
                  << " pressure_homogeneous_pins="
                  << sample.pressure_constraints.homogeneous_pins
                  << " pressure_aggregate_max_weight_l1="
                  << sample.pressure_aggregation.maximum_weight_l1
                  << " pressure_aggregate_max_reach_over_h="
                  << sample.pressure_aggregation
                         .maximum_slave_master_distance_over_h
                  << " free_velocity_dofs=" << sample.free_velocity_dofs
                  << " free_pressure_dofs=" << sample.free_pressure_dofs
                  << " zero_free_pressure_rows="
                  << sample.zero_free_pressure_rows
                  << " pressure_ghost_norm=" << sample.pressure_ghost_norm
                  << " pspg_pressure_gradient_norm="
                  << sample.pspg_pressure_gradient_norm
                  << " generalized_coupling_rank="
                  << sample.pressure_control.generalized_coupling_rank
                  << " pressure_dimension="
                  << sample.pressure_control.pressure_dimension
                  << " generalized_coupling_sigma_min="
                  << sample.pressure_control
                         .generalized_coupling_smallest_singular_value
                  << " stabilized_pressure_control="
                  << sample.pressure_control.stabilized_pressure_control
                  << " constant_pressure_control="
                  << sample.pressure_control.constant_pressure_control
                  << " pressure_gradient_adjoint_relative_defect="
                  << sample.pressure_control
                         .pressure_gradient_adjoint_relative_defect
                  << " equilibrated_condition_inf="
                  << sample.equilibrated_condition_inf << '\n';
    }
    // The no-pin system retains the physically anchored constant-pressure
    // direction.  Bound its absolute conditioning and the global variation
    // across topology/dimension changes, then separately require that
    // conditioning does not grow as epsilon tends toward each mesh feature.
    constexpr FE::Real maximum_accepted_condition_inf = FE::Real{400.0};
    constexpr FE::Real maximum_accepted_cut_position_spread = FE::Real{4.0};
    constexpr FE::Real maximum_near_feature_growth = FE::Real{1.1};
    constexpr FE::Real maximum_accepted_aggregate_weight_l1 =
        FE::Real{3.0} + FE::Real{1.0e-12};
    constexpr FE::Real maximum_accepted_aggregate_reach_over_h =
        FE::Real{3.5};
    constexpr FE::Real minimum_accepted_stabilized_pressure_control =
        FE::Real{0.45};
    constexpr FE::Real maximum_accepted_stabilized_control_spread =
        FE::Real{1.05};
    const auto condition_spread = max_condition / min_condition;
    const auto stabilized_control_spread =
        max_stabilized_control / min_stabilized_control;
    EXPECT_LE(max_condition, maximum_accepted_condition_inf)
        << "equilibrated mixed Jacobian exceeded the fixed-sweep stability bound";
    EXPECT_LE(condition_spread, maximum_accepted_cut_position_spread)
        << "mixed stability surrogate varies excessively with cut position";
    EXPECT_LE(max_aggregate_weight_l1,
              maximum_accepted_aggregate_weight_l1)
        << "closed pressure aggregation amplifies nodal data excessively";
    EXPECT_LE(max_aggregate_reach_over_h,
              maximum_accepted_aggregate_reach_over_h)
        << "closed pressure aggregation reaches too far in mesh units";
    EXPECT_GE(min_stabilized_control,
              minimum_accepted_stabilized_pressure_control)
        << "stabilized pressure Schur control fell below the finite-sweep bound";
    EXPECT_LE(stabilized_control_spread,
              maximum_accepted_stabilized_control_spread)
        << "stabilized pressure control varies excessively with cut position";
    const auto expect_no_near_feature_growth =
        [&](std::size_t first, std::size_t last) {
            ASSERT_LT(first, samples.size());
            ASSERT_LT(last, samples.size());
            ASSERT_LE(first, last);
            const auto bound = maximum_near_feature_growth *
                               samples[first].equilibrated_condition_inf;
            for (std::size_t index = first + 1u; index <= last; ++index) {
                EXPECT_LE(samples[index].equilibrated_condition_inf, bound)
                    << "condition grew as the cut approached a mesh feature; family="
                    << samples[first].label
                    << " position=" << samples[index].label;
            }
        };
    expect_no_near_feature_growth(/*near_vertex_first=*/1u,
                                  /*near_vertex_last=*/3u);
    expect_no_near_feature_growth(/*near_edge_first=*/4u,
                                  /*near_edge_last=*/6u);
    expect_no_near_feature_growth(/*near_face_first=*/7u,
                                  /*near_face_last=*/9u);
    RecordProperty("wp7_finite_cut_position_count", samples.size());
    RecordProperty(
        "wp7_finite_cut_maximum_condition",
        realPropertyValue(max_condition));
    RecordProperty(
        "wp7_finite_cut_condition_spread",
        realPropertyValue(condition_spread));
    RecordProperty(
        "wp7_finite_cut_minimum_pressure_control",
        realPropertyValue(min_stabilized_control));
    RecordProperty(
        "wp7_finite_cut_pressure_control_spread",
        realPropertyValue(stabilized_control_spread));
    RecordProperty(
        "wp7_finite_cut_maximum_aggregate_weight_l1",
        realPropertyValue(max_aggregate_weight_l1));
    RecordProperty(
        "wp7_finite_cut_maximum_aggregate_reach_over_h",
        realPropertyValue(max_aggregate_reach_over_h));
    std::cout << std::setprecision(12)
              << "FS14_cut_sweep_summary"
              << " positions=" << samples.size()
              << " min_equilibrated_condition_inf=" << min_condition
              << " max_equilibrated_condition_inf=" << max_condition
              << " condition_spread=" << condition_spread
              << " min_stabilized_pressure_control="
              << min_stabilized_control
              << " stabilized_pressure_control_spread="
              << stabilized_control_spread
              << " maximum_aggregate_weight_l1="
              << max_aggregate_weight_l1
              << " maximum_aggregate_reach_over_h="
              << max_aggregate_reach_over_h
              << " maximum_accepted_condition_inf="
              << maximum_accepted_condition_inf
              << " maximum_accepted_cut_position_spread="
              << maximum_accepted_cut_position_spread
              << " maximum_near_feature_growth="
              << maximum_near_feature_growth
              << " maximum_accepted_aggregate_weight_l1="
              << maximum_accepted_aggregate_weight_l1
              << " maximum_accepted_aggregate_reach_over_h="
              << maximum_accepted_aggregate_reach_over_h
              << " minimum_accepted_stabilized_pressure_control="
              << minimum_accepted_stabilized_pressure_control
              << " maximum_accepted_stabilized_control_spread="
              << maximum_accepted_stabilized_control_spread
              << '\n';
#endif
}

TEST(FreeSurfaceCutStability,
     FixedPhysicalCutThreeLevelRefinementBoundsPressureControlAndAggregation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Fixed-cut refinement requires native mesh support.";
#else
    const ScopedEnvVar pressure_diagnostics(
        "SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC", "1");

    // The physical domain and plane are identical at every level; only the
    // background tetrahedral spacing changes.  This is a finite three-level
    // regression.  It neither samples every relative cut position nor proves
    // an h-uniform inf-sup or aggregate-extension bound.
    const PlaneCutPosition cut{
        "fixed_physical_oblique_plane", {{1.0, 0.73, 0.41}}, 2.47};
    constexpr std::array<int, 3> resolutions = {{2, 3, 4}};

    std::vector<StabilitySample> samples;
    samples.reserve(resolutions.size());
    FE::Real minimum_stabilized_control =
        std::numeric_limits<FE::Real>::infinity();
    FE::Real maximum_stabilized_control = 0.0;
    FE::Real minimum_equilibrated_sigma =
        std::numeric_limits<FE::Real>::infinity();
    FE::Real maximum_weight_l1 = 0.0;
    FE::Real maximum_extension_reach = 0.0;

    for (const auto resolution : resolutions) {
        SCOPED_TRACE(std::string("cells_per_axis=") +
                     std::to_string(resolution));
        PersistentStabilityProblem problem(cut, resolution);
        samples.push_back(problem.evaluate(cut));
        const auto& sample = samples.back();

        EXPECT_EQ(sample.mesh_cells_per_axis, resolution);
        EXPECT_NEAR(sample.mesh_spacing,
                    FE::Real{2.0} / static_cast<FE::Real>(resolution),
                    FE::Real{1.0e-15});
        EXPECT_GT(sample.cut_cells, 0u);
        EXPECT_GT(sample.cut_adjacent_facets, 0u);
        EXPECT_GT(sample.pressure_constraints.master_bearing, 0u);
        EXPECT_EQ(sample.pressure_constraints.master_bearing,
                  sample.pressure_aggregation.master_bearing_lines);
        EXPECT_GT(sample.pressure_aggregation.master_entries, 0u);
        EXPECT_LE(
            sample.pressure_aggregation.maximum_partition_of_unity_error,
            FE::Real{1.0e-10} *
                std::max(FE::Real{1.0},
                         sample.pressure_aggregation.maximum_weight_l1));
        EXPECT_LE(sample.pressure_aggregation.maximum_inhomogeneity,
                  FE::Real{1.0e-14});
        EXPECT_TRUE(std::isfinite(
            sample.pressure_aggregation.maximum_weight_l1));
        EXPECT_TRUE(std::isfinite(
            sample.pressure_aggregation
                .maximum_slave_master_distance_over_h));
        EXPECT_EQ(sample.zero_free_pressure_rows, 0u);
        EXPECT_EQ(sample.equilibrated_rank, sample.equilibrated_size);
        EXPECT_GT(sample.equilibrated_smallest_singular_value,
                  FE::Real{0.0});
        EXPECT_LE(sample.equilibrated_condition_inf, FE::Real{400.0});
        EXPECT_EQ(sample.pressure_control.pressure_dimension,
                  sample.free_pressure_dofs);
        EXPECT_LE(sample.pressure_control.generalized_coupling_rank,
                  sample.pressure_control.pressure_dimension);
        EXPECT_LE(sample.pressure_control.velocity_block_relative_skew,
                  FE::Real{1.0e-10});
        EXPECT_LE(
            sample.pressure_control
                .pressure_gradient_adjoint_relative_defect,
            FE::Real{1.0e-10});
        EXPECT_GT(
            sample.pressure_control
                .stabilized_schur_smallest_generalized_eigenvalue,
            FE::Real{0.0});
        EXPECT_GT(sample.pressure_control.stabilized_pressure_control,
                  FE::Real{0.0});

        minimum_stabilized_control = std::min(
            minimum_stabilized_control,
            sample.pressure_control.stabilized_pressure_control);
        maximum_stabilized_control = std::max(
            maximum_stabilized_control,
            sample.pressure_control.stabilized_pressure_control);
        minimum_equilibrated_sigma = std::min(
            minimum_equilibrated_sigma,
            sample.equilibrated_smallest_singular_value);
        maximum_weight_l1 = std::max(
            maximum_weight_l1,
            sample.pressure_aggregation.maximum_weight_l1);
        maximum_extension_reach = std::max(
            maximum_extension_reach,
            sample.pressure_aggregation
                .maximum_slave_master_distance_over_h);

        std::cout << std::setprecision(12)
                  << "FS14_refinement"
                  << " cells_per_axis=" << resolution
                  << " h=" << sample.mesh_spacing
                  << " cut_cells=" << sample.cut_cells
                  << " pressure_aggregate_slaves="
                  << sample.pressure_constraints.master_bearing
                  << " pressure_aggregate_max_weight_l1="
                  << sample.pressure_aggregation.maximum_weight_l1
                  << " pressure_aggregate_max_abs_weight="
                  << sample.pressure_aggregation.maximum_absolute_weight
                  << " pressure_aggregate_max_reach_over_h="
                  << sample.pressure_aggregation
                         .maximum_slave_master_distance_over_h
                  << " pressure_aggregate_max_partition_error="
                  << sample.pressure_aggregation
                         .maximum_partition_of_unity_error
                  << " generalized_coupling_rank="
                  << sample.pressure_control.generalized_coupling_rank
                  << " pressure_dimension="
                  << sample.pressure_control.pressure_dimension
                  << " generalized_coupling_sigma_min="
                  << sample.pressure_control
                         .generalized_coupling_smallest_singular_value
                  << " stabilized_schur_lambda_min="
                  << sample.pressure_control
                         .stabilized_schur_smallest_generalized_eigenvalue
                  << " stabilized_pressure_control="
                  << sample.pressure_control.stabilized_pressure_control
                  << " equilibrated_sigma_min="
                  << sample.equilibrated_smallest_singular_value
                  << " equilibrated_condition_inf="
                  << sample.equilibrated_condition_inf << '\n';
    }

    ASSERT_EQ(samples.size(), resolutions.size());
    // Fixed finite-fixture gates.  They qualify only these three meshes and
    // this physical plane; they are not asymptotic lower/upper bounds.
    constexpr FE::Real minimum_accepted_stabilized_pressure_control =
        FE::Real{0.45};
    constexpr FE::Real maximum_accepted_stabilized_control_spread =
        FE::Real{1.75};
    constexpr FE::Real minimum_accepted_equilibrated_sigma = FE::Real{0.08};
    constexpr FE::Real maximum_accepted_aggregate_weight_l1 =
        FE::Real{5.0} + FE::Real{1.0e-12};
    constexpr FE::Real maximum_accepted_aggregate_reach_over_h =
        FE::Real{3.75};
    const auto stabilized_control_spread =
        maximum_stabilized_control / minimum_stabilized_control;
    EXPECT_GE(minimum_stabilized_control,
              minimum_accepted_stabilized_pressure_control);
    EXPECT_LE(stabilized_control_spread,
              maximum_accepted_stabilized_control_spread);
    EXPECT_GE(minimum_equilibrated_sigma,
              minimum_accepted_equilibrated_sigma);
    EXPECT_LE(maximum_weight_l1,
              maximum_accepted_aggregate_weight_l1);
    EXPECT_LE(maximum_extension_reach,
              maximum_accepted_aggregate_reach_over_h);
    RecordProperty("wp7_refinement_h_level_count", samples.size());
    RecordProperty(
        "wp7_refinement_minimum_pressure_control",
        realPropertyValue(minimum_stabilized_control));
    RecordProperty(
        "wp7_refinement_pressure_control_spread",
        realPropertyValue(stabilized_control_spread));
    RecordProperty(
        "wp7_refinement_minimum_equilibrated_sigma",
        realPropertyValue(minimum_equilibrated_sigma));
    RecordProperty(
        "wp7_refinement_maximum_aggregate_weight_l1",
        realPropertyValue(maximum_weight_l1));
    RecordProperty(
        "wp7_refinement_maximum_aggregate_reach_over_h",
        realPropertyValue(maximum_extension_reach));
    std::cout << std::setprecision(12)
              << "FS14_refinement_summary"
              << " levels=" << samples.size()
              << " minimum_stabilized_pressure_control="
              << minimum_stabilized_control
              << " stabilized_pressure_control_spread="
              << stabilized_control_spread
              << " minimum_equilibrated_sigma="
              << minimum_equilibrated_sigma
              << " maximum_aggregate_weight_l1=" << maximum_weight_l1
              << " maximum_aggregate_reach_over_h="
              << maximum_extension_reach
              << " minimum_accepted_stabilized_pressure_control="
              << minimum_accepted_stabilized_pressure_control
              << " maximum_accepted_stabilized_control_spread="
              << maximum_accepted_stabilized_control_spread
              << " minimum_accepted_equilibrated_sigma="
              << minimum_accepted_equilibrated_sigma
              << " maximum_accepted_aggregate_weight_l1="
              << maximum_accepted_aggregate_weight_l1
              << " maximum_accepted_aggregate_reach_over_h="
              << maximum_accepted_aggregate_reach_over_h << '\n';
#endif
}

TEST(FreeSurfaceCutStability,
     PersistentMovingCutRefreshesAggregationAndRetainsMixedRankWithoutPressurePin)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Moving cut-position stability sequence requires native mesh support.";
#else
    const ScopedEnvVar pressure_diagnostics(
        "SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC", "1");

    // Unlike the independent near-feature sweep, this sequence keeps one
    // FESystem alive while the oblique plane translates through multiple
    // element/aggregation topologies.  The final position revisits step 1 to
    // detect stale cut rules, facet sets, or constraints left by later steps.
    const std::vector<PlaneCutPosition> positions = {
        {"forward_0", {{1.0, 0.73, 0.41}}, 2.25},
        {"forward_1", {{1.0, 0.73, 0.41}}, 2.50},
        {"forward_2", {{1.0, 0.73, 0.41}}, 2.75},
        {"revisit_1", {{1.0, 0.73, 0.41}}, 2.50},
    };

    PersistentStabilityProblem problem(positions.front());
    std::vector<StabilitySample> samples;
    samples.reserve(positions.size());
    std::set<std::size_t> cut_cell_counts;
    std::set<std::size_t> cut_facet_counts;
    std::set<std::size_t> aggregate_pressure_slave_counts;
    FE::Real min_condition = std::numeric_limits<FE::Real>::infinity();
    FE::Real max_condition = 0.0;
    FE::Real min_stabilized_control =
        std::numeric_limits<FE::Real>::infinity();
    FE::Real max_stabilized_control = 0.0;

    for (const auto& position : positions) {
        SCOPED_TRACE(position.label);
        samples.push_back(problem.evaluate(position));
        const auto& sample = samples.back();

        EXPECT_GT(sample.cut_cells, 0u);
        EXPECT_GT(sample.cut_adjacent_facets, 0u);
        EXPECT_GT(sample.backend_volume_quadrature_points, 0u);
        EXPECT_EQ(sample.backend_fallback_cells, 0u);
        EXPECT_TRUE(sample.pressure_natural_traction_anchor);
        EXPECT_TRUE(sample.pressure_anchor_has_no_gauge_enforcement);
        EXPECT_EQ(sample.pressure_constraints.homogeneous_pins, 0u)
            << "the moving sequence must retain every pressure mode except "
               "master-bearing aggregation slaves";
        EXPECT_EQ(sample.velocity_constraints.homogeneous_pins, 0u);
        EXPECT_GT(sample.pressure_constraints.master_bearing, 0u);
        EXPECT_EQ(sample.pressure_constraints.master_bearing,
                  sample.pressure_aggregation.master_bearing_lines);
        EXPECT_LE(
            sample.pressure_aggregation.maximum_partition_of_unity_error,
            FE::Real{1.0e-10} *
                std::max(FE::Real{1.0},
                         sample.pressure_aggregation.maximum_weight_l1));
        EXPECT_LE(sample.pressure_aggregation.maximum_inhomogeneity,
                  FE::Real{1.0e-14});
        EXPECT_LE(sample.pressure_aggregation.maximum_weight_l1,
                  FE::Real{3.0} + FE::Real{1.0e-12});
        EXPECT_LE(
            sample.pressure_aggregation
                .maximum_slave_master_distance_over_h,
            FE::Real{3.5});
        EXPECT_EQ(sample.velocity_constraints.master_bearing,
                  3u * sample.pressure_constraints.master_bearing);
        EXPECT_GT(sample.free_pressure_dofs, 0u);
        EXPECT_EQ(sample.zero_free_pressure_rows, 0u);
        EXPECT_GT(sample.pressure_ghost_norm, FE::Real{1.0e-14});
        EXPECT_GT(sample.pspg_pressure_gradient_norm, FE::Real{1.0e-14});
        EXPECT_EQ(sample.equilibrated_rank, sample.equilibrated_size);
        EXPECT_TRUE(std::isfinite(sample.equilibrated_condition_inf));
        EXPECT_GT(sample.equilibrated_condition_inf, FE::Real{0.0});
        EXPECT_EQ(sample.pressure_control.pressure_dimension,
                  sample.free_pressure_dofs);
        EXPECT_LE(sample.pressure_control.velocity_block_relative_skew,
                  FE::Real{1.0e-10});
        EXPECT_LE(
            sample.pressure_control
                .pressure_gradient_adjoint_relative_defect,
            FE::Real{1.0e-10});
        EXPECT_GT(sample.pressure_control.stabilized_pressure_control,
                  FE::Real{0.0});
        EXPECT_GE(sample.pressure_control.stabilized_pressure_control,
                  FE::Real{0.45});

        cut_cell_counts.insert(sample.cut_cells);
        cut_facet_counts.insert(sample.cut_adjacent_facets);
        aggregate_pressure_slave_counts.insert(
            sample.pressure_constraints.master_bearing);
        min_condition = std::min(
            min_condition, sample.equilibrated_condition_inf);
        max_condition = std::max(
            max_condition, sample.equilibrated_condition_inf);
        min_stabilized_control = std::min(
            min_stabilized_control,
            sample.pressure_control.stabilized_pressure_control);
        max_stabilized_control = std::max(
            max_stabilized_control,
            sample.pressure_control.stabilized_pressure_control);

        std::cout << std::setprecision(12)
                  << "FS14_evolving_cut"
                  << " position=" << sample.label
                  << " physical_active_volume="
                  << sample.physical_active_volume
                  << " cut_cells=" << sample.cut_cells
                  << " cut_adjacent_facets=" << sample.cut_adjacent_facets
                  << " pressure_aggregate_slaves="
                  << sample.pressure_constraints.master_bearing
                  << " pressure_aggregate_max_weight_l1="
                  << sample.pressure_aggregation.maximum_weight_l1
                  << " pressure_aggregate_max_reach_over_h="
                  << sample.pressure_aggregation
                         .maximum_slave_master_distance_over_h
                  << " free_pressure_dofs=" << sample.free_pressure_dofs
                  << " zero_free_pressure_rows="
                  << sample.zero_free_pressure_rows
                  << " pressure_ghost_norm=" << sample.pressure_ghost_norm
                  << " pspg_pressure_gradient_norm="
                  << sample.pspg_pressure_gradient_norm
                  << " generalized_coupling_sigma_min="
                  << sample.pressure_control
                         .generalized_coupling_smallest_singular_value
                  << " stabilized_pressure_control="
                  << sample.pressure_control.stabilized_pressure_control
                  << " equilibrated_condition_inf="
                  << sample.equilibrated_condition_inf << '\n';
    }

    ASSERT_EQ(samples.size(), positions.size());
    EXPECT_GT(cut_cell_counts.size(), 1u)
        << "moving interface did not change cut-cell topology";
    EXPECT_GT(cut_facet_counts.size(), 1u)
        << "moving interface did not change pressure ghost-facet scope";
    EXPECT_GT(aggregate_pressure_slave_counts.size(), 1u)
        << "moving interface did not change aggregation topology";

    const auto& first_visit = samples[1];
    const auto& revisit = samples[3];
    EXPECT_EQ(revisit.cut_cells, first_visit.cut_cells);
    EXPECT_EQ(revisit.cut_adjacent_facets, first_visit.cut_adjacent_facets);
    EXPECT_EQ(revisit.pressure_constraints.master_bearing,
              first_visit.pressure_constraints.master_bearing);
    EXPECT_EQ(revisit.free_pressure_dofs, first_visit.free_pressure_dofs);
    EXPECT_NEAR(revisit.physical_active_volume,
                first_visit.physical_active_volume,
                FE::Real{1.0e-12});
    EXPECT_NEAR(revisit.pressure_ghost_norm,
                first_visit.pressure_ghost_norm,
                FE::Real{1.0e-12});
    EXPECT_NEAR(revisit.pspg_pressure_gradient_norm,
                first_visit.pspg_pressure_gradient_norm,
                FE::Real{1.0e-12});
    EXPECT_NEAR(revisit.equilibrated_condition_inf,
                first_visit.equilibrated_condition_inf,
                FE::Real{1.0e-9});
    EXPECT_NEAR(
        revisit.pressure_aggregation.maximum_weight_l1,
        first_visit.pressure_aggregation.maximum_weight_l1,
        FE::Real{1.0e-12});
    EXPECT_NEAR(
        revisit.pressure_aggregation.maximum_slave_master_distance_over_h,
        first_visit.pressure_aggregation.maximum_slave_master_distance_over_h,
        FE::Real{1.0e-12});
    EXPECT_EQ(revisit.pressure_control.generalized_coupling_rank,
              first_visit.pressure_control.generalized_coupling_rank);
    EXPECT_NEAR(
        revisit.pressure_control.stabilized_pressure_control,
        first_visit.pressure_control.stabilized_pressure_control,
        FE::Real{1.0e-12});

    // These are regression guards for this finite sequence, not a general
    // cut-position-independent inf-sup or conditioning result.
    constexpr FE::Real maximum_accepted_condition_inf = FE::Real{400.0};
    constexpr FE::Real maximum_accepted_sequence_spread = FE::Real{3.0};
    constexpr FE::Real maximum_accepted_stabilized_control_spread =
        FE::Real{1.05};
    const auto condition_spread = max_condition / min_condition;
    const auto stabilized_control_spread =
        max_stabilized_control / min_stabilized_control;
    EXPECT_LE(max_condition, maximum_accepted_condition_inf);
    EXPECT_LE(condition_spread, maximum_accepted_sequence_spread);
    EXPECT_LE(stabilized_control_spread,
              maximum_accepted_stabilized_control_spread);
    RecordProperty("wp7_moving_cut_position_count", samples.size());
    RecordProperty(
        "wp7_moving_cut_distinct_cell_topology_count",
        cut_cell_counts.size());
    RecordProperty(
        "wp7_moving_cut_distinct_facet_topology_count",
        cut_facet_counts.size());
    RecordProperty(
        "wp7_moving_cut_distinct_aggregate_topology_count",
        aggregate_pressure_slave_counts.size());
    RecordProperty(
        "wp7_moving_cut_condition_spread",
        realPropertyValue(condition_spread));
    RecordProperty(
        "wp7_moving_cut_pressure_control_spread",
        realPropertyValue(stabilized_control_spread));
    std::cout << std::setprecision(12)
              << "FS14_evolving_cut_summary"
              << " positions=" << samples.size()
              << " distinct_cut_cell_counts=" << cut_cell_counts.size()
              << " distinct_cut_facet_counts=" << cut_facet_counts.size()
              << " distinct_pressure_aggregate_counts="
              << aggregate_pressure_slave_counts.size()
              << " min_equilibrated_condition_inf=" << min_condition
              << " max_equilibrated_condition_inf=" << max_condition
              << " condition_spread=" << condition_spread
              << " min_stabilized_pressure_control="
              << min_stabilized_control
              << " stabilized_pressure_control_spread="
              << stabilized_control_spread
              << " maximum_accepted_condition_inf="
              << maximum_accepted_condition_inf
              << " maximum_accepted_sequence_spread="
              << maximum_accepted_sequence_spread
              << " maximum_accepted_stabilized_control_spread="
              << maximum_accepted_stabilized_control_spread
              << '\n';
#endif
}

TEST(FreeSurfaceCutStability,
     SymmetricNitscheFiniteSampleEnergySpectrumUsesSharpBoundaryAndAggregation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Symmetric Nitsche energy sampling requires native mesh support.";
#else
    const ScopedEnvVar energy_diagnostic(
        "SVMP_NS_SYMMETRIC_NITSCHE_ENERGY_DIAGNOSTIC", "1");
    const std::array<FE::Real, 9> fractions = {{
        FE::Real{0.0},
        FE::Real{1.0e-8},
        FE::Real{1.0e-6},
        FE::Real{1.0e-4},
        FE::Real{1.0e-2},
        FE::Real{0.1},
        FE::Real{0.25},
        FE::Real{0.49},
        FE::Real{1.0},
    }};
    const std::array<FE::Real, 3> mesh_scales = {{
        FE::Real{0.5},
        FE::Real{1.0} / FE::Real{3.0},
        FE::Real{0.25},
    }};
    const std::array<NitscheEnergyOrientation, 2> orientations = {{
        {"axis", {{1.0, 0.0, 0.0}}},
        {"oblique", {{1.0, 0.73, 0.41}}},
    }};

    constexpr FE::Real matrix_tolerance{1.0e-11};
    constexpr FE::Real fraction_tolerance{2.0e-11};
    constexpr FE::Real parity_tolerance{1.0e-10};
    constexpr std::size_t expected_case_count{108u};
    constexpr std::size_t expected_positive_case_count{96u};
    constexpr std::size_t expected_dry_case_count{12u};

    std::size_t case_count = 0u;
    std::size_t positive_case_count = 0u;
    std::size_t dry_case_count = 0u;
    std::size_t dry_exact_zero_case_count = 0u;
    std::size_t aggregation_exercised_case_count = 0u;
    std::size_t diagnostic_structure_verified_case_count = 0u;
    std::set<int> generated_active_boundary_markers;
    std::size_t maximum_observed_root_path = 0u;
    std::size_t maximum_root_path_guard_rejections = 0u;
    FE::Real maximum_observed_reference_extrapolation{0.0};
    std::size_t maximum_extrapolation_guard_rejections = 0u;
    FE::Real maximum_observed_absolute_coefficient{0.0};
    FE::Real maximum_observed_row_l1_norm{0.0};
    std::size_t maximum_line_guard_rejections = 0u;
    FE::Real maximum_fraction_absolute_error{0.0};
    FE::Real maximum_fraction_relative_error{0.0};
    FE::Real maximum_operator_reconstruction_error{0.0};
    FE::Real maximum_energy_reconstruction_error{0.0};
    FE::Real maximum_operator_skew{0.0};
    FE::Real maximum_energy_skew{0.0};
    FE::Real maximum_dry_consistency_boundary_norm{0.0};
    FE::Real maximum_dry_penalty_boundary_norm{0.0};
    FE::Real maximum_dry_symmetric_boundary_norm{0.0};
    FE::Real maximum_eigensolver_off_diagonal_ratio{0.0};
    FE::Real minimum_sampled_margin{
        std::numeric_limits<FE::Real>::infinity()};
    FE::Real minimum_sampled_margin_ratio{
        std::numeric_limits<FE::Real>::infinity()};
    FE::Real minimum_sampled_eigenvalue{
        std::numeric_limits<FE::Real>::infinity()};
    FE::Real maximum_sampled_eigenvalue{
        -std::numeric_limits<FE::Real>::infinity()};
    FE::Real maximum_active_side_operator_difference{0.0};
    FE::Real maximum_active_side_energy_difference{0.0};
    FE::Real maximum_active_side_eigenvalue_difference{0.0};
    FE::Real maximum_affine_scale_eigenvalue_spread{0.0};
    std::string minimum_sampled_case;

    const auto scalar_relative_difference =
        [](FE::Real lhs, FE::Real rhs) {
            return std::abs(lhs - rhs) /
                   std::max(
                       FE::Real{1.0},
                       std::max(std::abs(lhs), std::abs(rhs)));
        };
    const auto matrix_is_finite =
        [](std::span<const FE::Real> matrix) {
            return std::all_of(
                matrix.begin(), matrix.end(), [](FE::Real value) {
                    return std::isfinite(value);
                });
        };

    ASSERT_LT(
        nitsche_energy_default_maximum_root_path_length,
        nitsche_energy_maximum_root_path_length);
    std::string default_root_path_rejection_diagnostic;
    {
        PersistentNitscheEnergyProblem default_guard_problem(
            mesh_scales.front(),
            orientations.back(),
            FE::geometry::CutIntegrationSide::Negative,
            nitsche_energy_default_maximum_root_path_length);
        try {
            static_cast<void>(
                default_guard_problem.evaluate(FE::Real{1.0e-4}));
        } catch (const std::runtime_error& error) {
            default_root_path_rejection_diagnostic = error.what();
        }
    }
    const bool default_root_path_guard_rejection_verified =
        default_root_path_rejection_diagnostic.find(
            "diagnostic=root_path_guard_rejection") !=
            std::string::npos &&
        default_root_path_rejection_diagnostic.find(
            "maximum_observed_path=9") != std::string::npos &&
        default_root_path_rejection_diagnostic.find(
            "maximum_allowed_path=" +
            std::to_string(
                nitsche_energy_default_maximum_root_path_length)) !=
            std::string::npos;
    EXPECT_TRUE(default_root_path_guard_rejection_verified)
        << default_root_path_rejection_diagnostic;

    for (const auto& orientation : orientations) {
        std::array<
            std::array<std::vector<NitscheEnergySample>, 2>,
            3>
            samples_by_scale;
        for (std::size_t scale_index = 0u;
             scale_index < mesh_scales.size();
             ++scale_index) {
            const auto mesh_scale = mesh_scales[scale_index];
            PersistentNitscheEnergyProblem negative_problem(
                mesh_scale,
                orientation,
                FE::geometry::CutIntegrationSide::Negative);
            PersistentNitscheEnergyProblem positive_problem(
                mesh_scale,
                orientation,
                FE::geometry::CutIntegrationSide::Positive);
            samples_by_scale[scale_index][0].reserve(
                fractions.size());
            samples_by_scale[scale_index][1].reserve(
                fractions.size());

            for (const auto fraction : fractions) {
                SCOPED_TRACE(
                    std::string(orientation.id) + "_h_" +
                    realPropertyValue(mesh_scale) + "_fraction_" +
                    realPropertyValue(fraction));
                samples_by_scale[scale_index][0].push_back(
                    negative_problem.evaluate(fraction));
                samples_by_scale[scale_index][1].push_back(
                    positive_problem.evaluate(fraction));

                for (std::size_t side_index = 0u;
                     side_index < 2u;
                     ++side_index) {
                    const auto expected_side =
                        side_index == 0u
                            ? FE::geometry::
                                  CutIntegrationSide::Negative
                            : FE::geometry::
                                  CutIntegrationSide::Positive;
                    const auto& sample =
                        samples_by_scale[scale_index][side_index]
                                        .back();
                    const bool dry =
                        fraction == FE::Real{0.0};
                    SCOPED_TRACE(
                        sample.case_id + "_h_" +
                        realPropertyValue(mesh_scale));

                    ++case_count;
                    const auto fraction_absolute_error =
                        std::abs(
                            sample.observed_wall_fraction -
                            fraction);
                    maximum_fraction_absolute_error = std::max(
                        maximum_fraction_absolute_error,
                        fraction_absolute_error);
                    if (!dry) {
                        maximum_fraction_relative_error = std::max(
                            maximum_fraction_relative_error,
                            fraction_absolute_error / fraction);
                    }
                    EXPECT_LE(
                        fraction_absolute_error,
                        fraction_tolerance);
                    EXPECT_EQ(
                        sample.implicit_backend_fallback_count, 0u);
                    EXPECT_GT(sample.free_velocity_dofs, 0u);
                    EXPECT_EQ(
                        sample.aggregation_report.active_side,
                        expected_side);
                    EXPECT_EQ(
                        sample.aggregation_report.interface_marker,
                        27231);
                    EXPECT_NE(
                        sample.aggregation_report.field,
                        FE::INVALID_FIELD_ID);
                    EXPECT_EQ(
                        sample.aggregation_report
                            .maximum_root_path_length,
                        nitsche_energy_maximum_root_path_length);
                    EXPECT_LE(
                        sample.aggregation_report
                            .maximum_observed_root_path,
                        sample.aggregation_report
                            .maximum_root_path_length);
                    EXPECT_EQ(
                        sample.aggregation_report
                            .root_path_guard_rejections,
                        0u);
                    EXPECT_EQ(
                        sample.aggregation_report
                            .maximum_reference_extrapolation_distance,
                        nitsche_energy_maximum_reference_extrapolation_distance);
                    EXPECT_LE(
                        sample.aggregation_report
                            .maximum_observed_reference_extrapolation,
                        sample.aggregation_report
                            .maximum_reference_extrapolation_distance);
                    EXPECT_EQ(
                        sample.aggregation_report
                            .extrapolation_guard_rejections,
                        0u);
                    EXPECT_EQ(
                        sample.aggregation_report
                            .maximum_absolute_coefficient,
                        nitsche_energy_maximum_absolute_coefficient);
                    EXPECT_LE(
                        sample.aggregation_report
                            .maximum_observed_absolute_coefficient,
                        sample.aggregation_report
                            .maximum_absolute_coefficient);
                    EXPECT_EQ(
                        sample.aggregation_report
                            .maximum_row_l1_norm,
                        nitsche_energy_maximum_row_l1_norm);
                    EXPECT_LE(
                        sample.aggregation_report
                            .maximum_observed_row_l1_norm,
                        sample.aggregation_report
                            .maximum_row_l1_norm);
                    EXPECT_EQ(
                        sample.aggregation_report
                            .line_guard_rejections,
                        0u);
                    maximum_observed_root_path = std::max(
                        maximum_observed_root_path,
                        sample.aggregation_report
                            .maximum_observed_root_path);
                    maximum_root_path_guard_rejections = std::max(
                        maximum_root_path_guard_rejections,
                        sample.aggregation_report
                            .root_path_guard_rejections);
                    maximum_observed_reference_extrapolation =
                        std::max(
                            maximum_observed_reference_extrapolation,
                            sample.aggregation_report
                                .maximum_observed_reference_extrapolation);
                    maximum_extrapolation_guard_rejections =
                        std::max(
                            maximum_extrapolation_guard_rejections,
                            sample.aggregation_report
                                .extrapolation_guard_rejections);
                    maximum_observed_absolute_coefficient =
                        std::max(
                            maximum_observed_absolute_coefficient,
                            sample.aggregation_report
                                .maximum_observed_absolute_coefficient);
                    maximum_observed_row_l1_norm = std::max(
                        maximum_observed_row_l1_norm,
                        sample.aggregation_report
                            .maximum_observed_row_l1_norm);
                    maximum_line_guard_rejections = std::max(
                        maximum_line_guard_rejections,
                        sample.aggregation_report
                            .line_guard_rejections);
                    EXPECT_EQ(
                        sample.aggregation_report
                            .canonical_candidate_vertices,
                        sample.aggregation_report
                                .canonical_rooted_candidate_vertices +
                            sample.aggregation_report
                                .canonical_rootless_candidate_vertices);
                    EXPECT_EQ(
                        sample.aggregation_report
                            .canonical_rootless_candidate_vertices,
                        0u);
                    EXPECT_EQ(
                        sample.aggregation_report
                            .canonical_owned_pinned_dofs,
                        0u);
                    EXPECT_EQ(
                        sample.aggregation_report
                            .canonical_strong_suppressed_dofs,
                        0u);
                    EXPECT_EQ(
                        sample.velocity_aggregate_lines,
                        sample.aggregation_report
                            .canonical_owned_aggregate_dofs);
                    EXPECT_EQ(
                        sample.velocity_gauge_line_count, 0u);
                    if (sample.aggregation_report
                            .canonical_owned_aggregate_dofs >
                        0u) {
                        ++aggregation_exercised_case_count;
                    }

                    const auto ordinary_boundary_term_count =
                        std::accumulate(
                            sample
                                .diagnostic_boundary_term_counts
                                .begin(),
                            sample
                                .diagnostic_boundary_term_counts
                                .end(),
                            std::size_t{0u});
                    const bool boundary_bearing_operators_present =
                        std::all_of(
                            sample
                                    .diagnostic_interface_face_term_counts
                                    .begin() +
                                1,
                            sample
                                .diagnostic_interface_face_term_counts
                                .end(),
                            [](std::size_t count) {
                                return count > 0u;
                            });
                    EXPECT_EQ(ordinary_boundary_term_count, 0u);
                    EXPECT_EQ(
                        sample
                            .diagnostic_interface_face_term_counts
                            .front(),
                        0u);
                    EXPECT_TRUE(boundary_bearing_operators_present);
                    EXPECT_TRUE(
                        sample
                            .diagnostic_routes_use_generated_marker);
                    EXPECT_TRUE(
                        sample
                            .generated_active_boundary_marker_registered);
                    EXPECT_NE(
                        sample.generated_active_boundary_marker, -1);
                    EXPECT_NE(
                        sample.generated_active_boundary_marker,
                        27232);
                    generated_active_boundary_markers.insert(
                        sample.generated_active_boundary_marker);
                    if (ordinary_boundary_term_count == 0u &&
                        sample
                                .diagnostic_interface_face_term_counts
                                .front() ==
                            0u &&
                        boundary_bearing_operators_present &&
                        sample
                            .diagnostic_routes_use_generated_marker &&
                        sample
                            .generated_active_boundary_marker_registered) {
                        ++diagnostic_structure_verified_case_count;
                    }

                    maximum_operator_reconstruction_error =
                        std::max(
                            maximum_operator_reconstruction_error,
                            sample
                                .production_reconstruction_relative_error);
                    maximum_energy_reconstruction_error =
                        std::max(
                            maximum_energy_reconstruction_error,
                            sample
                                .energy_reconstruction_relative_error);
                    maximum_operator_skew = std::max(
                        maximum_operator_skew,
                        sample.symmetric_operator_relative_skew);
                    maximum_energy_skew = std::max(
                        maximum_energy_skew,
                        sample.energy_norm_relative_skew);
                    EXPECT_LE(
                        sample
                            .production_reconstruction_relative_error,
                        matrix_tolerance);
                    EXPECT_LE(
                        sample.energy_reconstruction_relative_error,
                        matrix_tolerance);
                    EXPECT_LE(
                        sample.symmetric_operator_relative_skew,
                        matrix_tolerance);
                    EXPECT_LE(
                        sample.energy_norm_relative_skew,
                        matrix_tolerance);
                    EXPECT_TRUE(matrix_is_finite(
                        sample.symmetric_operator));
                    EXPECT_TRUE(
                        matrix_is_finite(sample.energy_norm));

                    bool dry_boundary_exact_zero = false;
                    if (dry) {
                        ++dry_case_count;
                        EXPECT_EQ(sample.active_rule_count, 0u);
                        maximum_dry_consistency_boundary_norm =
                            std::max(
                                maximum_dry_consistency_boundary_norm,
                                sample
                                    .consistency_boundary_relative_norm);
                        maximum_dry_penalty_boundary_norm =
                            std::max(
                                maximum_dry_penalty_boundary_norm,
                                sample
                                    .penalty_boundary_relative_norm);
                        maximum_dry_symmetric_boundary_norm =
                            std::max(
                                maximum_dry_symmetric_boundary_norm,
                                sample
                                    .symmetric_boundary_relative_norm);
                        EXPECT_LE(
                            sample
                                .consistency_boundary_relative_norm,
                            matrix_tolerance);
                        EXPECT_LE(
                            sample.penalty_boundary_relative_norm,
                            matrix_tolerance);
                        EXPECT_LE(
                            sample.symmetric_boundary_relative_norm,
                            matrix_tolerance);
                        dry_boundary_exact_zero =
                            sample.bulk_plus_consistency ==
                                sample.bulk_viscous &&
                            sample.energy_norm ==
                                sample.bulk_viscous &&
                            sample.symmetric_operator ==
                                sample.bulk_viscous;
                        EXPECT_TRUE(dry_boundary_exact_zero);
                        EXPECT_GT(
                            FE::math::dense_matrix_max_abs(
                                sample.bulk_viscous),
                            FE::Real{0.0});
                        auto anchored_bulk =
                            sample.bulk_viscous;
                        symmetrize(
                            anchored_bulk,
                            sample.free_velocity_dofs);
                        EXPECT_NO_THROW((void)choleskyLower(
                            anchored_bulk,
                            sample.free_velocity_dofs,
                            "strongly anchored dry-boundary bulk"));
                        if (dry_boundary_exact_zero) {
                            ++dry_exact_zero_case_count;
                        }
                    } else {
                        ++positive_case_count;
                        EXPECT_GT(sample.active_rule_count, 0u);
                        EXPECT_GT(
                            sample.minimum_energy_norm_eigenvalue,
                            FE::Real{0.0});
                        EXPECT_TRUE(sample.eigensolver_converged);
                        EXPECT_GT(
                            sample.eigensolver_tolerance,
                            FE::Real{0.0});
                        const auto off_diagonal_ratio =
                            sample
                                .eigensolver_maximum_off_diagonal /
                            sample.eigensolver_tolerance;
                        maximum_eigensolver_off_diagonal_ratio =
                            std::max(
                                maximum_eigensolver_off_diagonal_ratio,
                                off_diagonal_ratio);
                        EXPECT_LE(
                            off_diagonal_ratio,
                            FE::Real{1.0});
                        const auto sampled_margin =
                            sample.minimum_generalized_eigenvalue -
                            sample.eigensolver_tolerance;
                        const auto sampled_margin_ratio =
                            sample.minimum_generalized_eigenvalue /
                            sample.eigensolver_tolerance;
                        minimum_sampled_margin = std::min(
                            minimum_sampled_margin,
                            sampled_margin);
                        minimum_sampled_margin_ratio = std::min(
                            minimum_sampled_margin_ratio,
                            sampled_margin_ratio);
                        EXPECT_GT(sampled_margin, FE::Real{0.0});
                        EXPECT_GT(
                            sampled_margin_ratio,
                            FE::Real{1.0});
                        if (sample.minimum_generalized_eigenvalue <
                            minimum_sampled_eigenvalue) {
                            minimum_sampled_eigenvalue =
                                sample
                                    .minimum_generalized_eigenvalue;
                            minimum_sampled_case =
                                sample.case_id + "_h_" +
                                realPropertyValue(mesh_scale);
                        }
                        maximum_sampled_eigenvalue = std::max(
                            maximum_sampled_eigenvalue,
                            sample
                                .maximum_generalized_eigenvalue);
                    }

                    std::cout
                        << std::setprecision(17)
                        << "WP3_WP7_NITSCHE_CASE {"
                        << "\"case_id\":\""
                        << sample.case_id << "\","
                        << "\"orientation\":\""
                        << orientation.id << "\","
                        << "\"active_side\":\""
                        << (side_index == 0u
                                ? "negative"
                                : "positive")
                        << "\","
                        << "\"mesh_scale\":"
                        << mesh_scale << ","
                        << "\"target_wall_fraction\":"
                        << fraction << ","
                        << "\"observed_wall_fraction\":"
                        << sample.observed_wall_fraction << ","
                        << "\"active_rule_count\":"
                        << sample.active_rule_count << ","
                        << "\"free_velocity_dofs\":"
                        << sample.free_velocity_dofs << ","
                        << "\"aggregate_dofs\":"
                        << sample.aggregation_report
                               .canonical_owned_aggregate_dofs
                        << ","
                        << "\"rootless_candidates\":"
                        << sample.aggregation_report
                               .canonical_rootless_candidate_vertices
                        << ","
                        << "\"owned_pins\":"
                        << sample.aggregation_report
                               .canonical_owned_pinned_dofs
                        << ","
                        << "\"maximum_root_path_length\":"
                        << sample.aggregation_report
                               .maximum_root_path_length
                        << ","
                        << "\"maximum_observed_root_path\":"
                        << sample.aggregation_report
                               .maximum_observed_root_path
                        << ","
                        << "\"root_path_guard_rejections\":"
                        << sample.aggregation_report
                               .root_path_guard_rejections
                        << ","
                        << "\"maximum_reference_extrapolation_distance\":"
                        << sample.aggregation_report
                               .maximum_reference_extrapolation_distance
                        << ","
                        << "\"maximum_observed_reference_extrapolation\":"
                        << sample.aggregation_report
                               .maximum_observed_reference_extrapolation
                        << ","
                        << "\"extrapolation_guard_rejections\":"
                        << sample.aggregation_report
                               .extrapolation_guard_rejections
                        << ","
                        << "\"maximum_absolute_coefficient\":"
                        << sample.aggregation_report
                               .maximum_absolute_coefficient
                        << ","
                        << "\"maximum_observed_absolute_coefficient\":"
                        << sample.aggregation_report
                               .maximum_observed_absolute_coefficient
                        << ","
                        << "\"maximum_row_l1_norm\":"
                        << sample.aggregation_report
                               .maximum_row_l1_norm
                        << ","
                        << "\"maximum_observed_row_l1_norm\":"
                        << sample.aggregation_report
                               .maximum_observed_row_l1_norm
                        << ","
                        << "\"line_guard_rejections\":"
                        << sample.aggregation_report
                               .line_guard_rejections
                        << ","
                        << "\"generated_active_boundary_marker\":"
                        << sample.generated_active_boundary_marker
                        << ","
                        << "\"ordinary_boundary_term_count\":"
                        << ordinary_boundary_term_count
                        << ","
                        << "\"bulk_interface_face_term_count\":"
                        << sample
                               .diagnostic_interface_face_term_counts[0]
                        << ","
                        << "\"bulk_plus_consistency_interface_face_term_count\":"
                        << sample
                               .diagnostic_interface_face_term_counts[1]
                        << ","
                        << "\"symmetric_operator_interface_face_term_count\":"
                        << sample
                               .diagnostic_interface_face_term_counts[2]
                        << ","
                        << "\"energy_norm_interface_face_term_count\":"
                        << sample
                               .diagnostic_interface_face_term_counts[3]
                        << ","
                        << "\"diagnostic_routes_use_generated_marker\":"
                        << (sample
                                    .diagnostic_routes_use_generated_marker
                                ? "true"
                                : "false")
                        << ","
                        << "\"generated_active_boundary_marker_registered\":"
                        << (sample
                                    .generated_active_boundary_marker_registered
                                ? "true"
                                : "false")
                        << ","
                        << "\"bulk_max_abs\":"
                        << FE::math::dense_matrix_max_abs(
                               sample.bulk_viscous)
                        << ","
                        << "\"bulk_plus_consistency_max_abs\":"
                        << FE::math::dense_matrix_max_abs(
                               sample.bulk_plus_consistency)
                        << ","
                        << "\"symmetric_operator_max_abs\":"
                        << FE::math::dense_matrix_max_abs(
                               sample.symmetric_operator)
                        << ","
                        << "\"energy_norm_max_abs\":"
                        << FE::math::dense_matrix_max_abs(
                               sample.energy_norm)
                        << ","
                        << "\"operator_reconstruction_error\":"
                        << sample
                               .production_reconstruction_relative_error
                        << ","
                        << "\"energy_reconstruction_error\":"
                        << sample
                               .energy_reconstruction_relative_error
                        << ","
                        << "\"operator_skew\":"
                        << sample.symmetric_operator_relative_skew
                        << ","
                        << "\"energy_skew\":"
                        << sample.energy_norm_relative_skew
                        << ","
                        << "\"dry_boundary_exact_zero\":"
                        << (dry_boundary_exact_zero ? 1 : 0)
                        << ","
                        << "\"lambda_min\":";
                    if (dry) {
                        std::cout << "null";
                    } else {
                        std::cout
                            << sample
                                   .minimum_generalized_eigenvalue;
                    }
                    std::cout << ",\"lambda_max\":";
                    if (dry) {
                        std::cout << "null";
                    } else {
                        std::cout
                            << sample
                                   .maximum_generalized_eigenvalue;
                    }
                    std::cout << ",\"eigensolver_tolerance\":";
                    if (dry) {
                        std::cout << "null";
                    } else {
                        std::cout
                            << sample.eigensolver_tolerance;
                    }
                    std::cout << "}" << '\n';
                }

                const auto& negative =
                    samples_by_scale[scale_index][0].back();
                const auto& positive =
                    samples_by_scale[scale_index][1].back();
                ASSERT_EQ(
                    negative.symmetric_operator.size(),
                    positive.symmetric_operator.size());
                ASSERT_EQ(
                    negative.energy_norm.size(),
                    positive.energy_norm.size());
                const auto operator_difference =
                    relativeMatrixDifference(
                        negative.symmetric_operator,
                        positive.symmetric_operator);
                const auto energy_difference =
                    relativeMatrixDifference(
                        negative.energy_norm,
                        positive.energy_norm);
                maximum_active_side_operator_difference =
                    std::max(
                        maximum_active_side_operator_difference,
                        operator_difference);
                maximum_active_side_energy_difference =
                    std::max(
                        maximum_active_side_energy_difference,
                        energy_difference);
                EXPECT_LE(
                    operator_difference, parity_tolerance);
                EXPECT_LE(
                    energy_difference, parity_tolerance);
                if (fraction > FE::Real{0.0}) {
                    const auto minimum_eigenvalue_difference =
                        scalar_relative_difference(
                            negative
                                .minimum_generalized_eigenvalue,
                            positive
                                .minimum_generalized_eigenvalue);
                    const auto maximum_eigenvalue_difference =
                        scalar_relative_difference(
                            negative
                                .maximum_generalized_eigenvalue,
                            positive
                                .maximum_generalized_eigenvalue);
                    maximum_active_side_eigenvalue_difference =
                        std::max(
                            maximum_active_side_eigenvalue_difference,
                            std::max(
                                minimum_eigenvalue_difference,
                                maximum_eigenvalue_difference));
                    EXPECT_LE(
                        minimum_eigenvalue_difference,
                        parity_tolerance);
                    EXPECT_LE(
                        maximum_eigenvalue_difference,
                        parity_tolerance);
                }
            }
        }

        for (std::size_t side_index = 0u;
             side_index < 2u;
             ++side_index) {
            for (std::size_t fraction_index = 1u;
                 fraction_index < fractions.size();
                 ++fraction_index) {
                const auto& reference =
                    samples_by_scale[0][side_index][fraction_index];
                for (std::size_t scale_index = 1u;
                     scale_index < mesh_scales.size();
                     ++scale_index) {
                    const auto& sample =
                        samples_by_scale[scale_index][side_index]
                                        [fraction_index];
                    const auto minimum_spread =
                        scalar_relative_difference(
                            reference
                                .minimum_generalized_eigenvalue,
                            sample
                                .minimum_generalized_eigenvalue);
                    const auto maximum_spread =
                        scalar_relative_difference(
                            reference
                                .maximum_generalized_eigenvalue,
                            sample
                                .maximum_generalized_eigenvalue);
                    maximum_affine_scale_eigenvalue_spread =
                        std::max(
                            maximum_affine_scale_eigenvalue_spread,
                            std::max(
                                minimum_spread,
                                maximum_spread));
                    EXPECT_LE(
                        minimum_spread, parity_tolerance);
                    EXPECT_LE(
                        maximum_spread, parity_tolerance);
                }
            }
        }
    }

    EXPECT_EQ(case_count, expected_case_count);
    EXPECT_EQ(
        positive_case_count,
        expected_positive_case_count);
    EXPECT_EQ(dry_case_count, expected_dry_case_count);
    EXPECT_EQ(
        dry_exact_zero_case_count,
        expected_dry_case_count);
    EXPECT_GT(aggregation_exercised_case_count, 0u);
    EXPECT_EQ(
        diagnostic_structure_verified_case_count,
        expected_case_count);
    EXPECT_EQ(generated_active_boundary_markers.size(), 2u);
    EXPECT_LE(
        maximum_observed_root_path,
        nitsche_energy_maximum_root_path_length);
    EXPECT_EQ(maximum_root_path_guard_rejections, 0u);
    EXPECT_LE(
        maximum_observed_reference_extrapolation,
        nitsche_energy_maximum_reference_extrapolation_distance);
    EXPECT_EQ(maximum_extrapolation_guard_rejections, 0u);
    EXPECT_LE(
        maximum_observed_absolute_coefficient,
        nitsche_energy_maximum_absolute_coefficient);
    EXPECT_LE(
        maximum_observed_row_l1_norm,
        nitsche_energy_maximum_row_l1_norm);
    EXPECT_EQ(maximum_line_guard_rejections, 0u);
    EXPECT_LE(
        maximum_fraction_absolute_error,
        fraction_tolerance);
    EXPECT_LE(
        maximum_operator_reconstruction_error,
        matrix_tolerance);
    EXPECT_LE(
        maximum_energy_reconstruction_error,
        matrix_tolerance);
    EXPECT_LE(maximum_operator_skew, matrix_tolerance);
    EXPECT_LE(maximum_energy_skew, matrix_tolerance);
    EXPECT_LE(
        maximum_dry_consistency_boundary_norm,
        matrix_tolerance);
    EXPECT_LE(
        maximum_dry_penalty_boundary_norm,
        matrix_tolerance);
    EXPECT_LE(
        maximum_dry_symmetric_boundary_norm,
        matrix_tolerance);
    EXPECT_LE(
        maximum_eigensolver_off_diagonal_ratio,
        FE::Real{1.0});
    EXPECT_GT(minimum_sampled_margin, FE::Real{0.0});
    EXPECT_GT(
        minimum_sampled_margin_ratio, FE::Real{1.0});
    EXPECT_LE(
        maximum_active_side_operator_difference,
        parity_tolerance);
    EXPECT_LE(
        maximum_active_side_energy_difference,
        parity_tolerance);
    EXPECT_LE(
        maximum_active_side_eigenvalue_difference,
        parity_tolerance);
    EXPECT_LE(
        maximum_affine_scale_eigenvalue_spread,
        parity_tolerance);

    RecordProperty("wp3_wp7_nitsche_case_count", case_count);
    RecordProperty(
        "wp3_wp7_nitsche_positive_case_count",
        positive_case_count);
    RecordProperty(
        "wp3_wp7_nitsche_dry_case_count",
        dry_case_count);
    RecordProperty(
        "wp3_wp7_nitsche_dry_exact_zero_case_count",
        dry_exact_zero_case_count);
    RecordProperty(
        "wp3_wp7_nitsche_fraction_count",
        fractions.size());
    RecordProperty(
        "wp3_wp7_nitsche_orientation_count",
        orientations.size());
    RecordProperty(
        "wp3_wp7_nitsche_mesh_scale_count",
        mesh_scales.size());
    RecordProperty(
        "wp3_wp7_nitsche_active_side_count", 2);
    RecordProperty(
        "wp3_wp7_nitsche_diagnostic_operator_count", 4);
    RecordProperty(
        "wp3_wp7_nitsche_diagnostic_structure_verified_case_count",
        diagnostic_structure_verified_case_count);
    RecordProperty(
        "wp3_wp7_nitsche_generated_active_boundary_marker_count",
        generated_active_boundary_markers.size());
    RecordProperty(
        "wp3_wp7_nitsche_aggregation_exercised_case_count",
        aggregation_exercised_case_count);
    RecordProperty(
        "wp3_wp7_nitsche_maximum_root_path_length",
        nitsche_energy_maximum_root_path_length);
    RecordProperty(
        "wp3_wp7_nitsche_default_maximum_root_path_length",
        nitsche_energy_default_maximum_root_path_length);
    RecordProperty(
        "wp3_wp7_nitsche_default_root_path_guard_rejection_verified",
        default_root_path_guard_rejection_verified ? 1 : 0);
    RecordProperty(
        "wp3_wp7_nitsche_maximum_observed_root_path",
        maximum_observed_root_path);
    RecordProperty(
        "wp3_wp7_nitsche_maximum_root_path_guard_rejections",
        maximum_root_path_guard_rejections);
    RecordProperty(
        "wp3_wp7_nitsche_maximum_reference_extrapolation_distance",
        realPropertyValue(
            nitsche_energy_maximum_reference_extrapolation_distance));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_observed_reference_extrapolation",
        realPropertyValue(
            maximum_observed_reference_extrapolation));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_extrapolation_guard_rejections",
        maximum_extrapolation_guard_rejections);
    RecordProperty(
        "wp3_wp7_nitsche_maximum_absolute_coefficient",
        realPropertyValue(
            nitsche_energy_maximum_absolute_coefficient));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_observed_absolute_coefficient",
        realPropertyValue(
            maximum_observed_absolute_coefficient));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_row_l1_norm",
        realPropertyValue(
            nitsche_energy_maximum_row_l1_norm));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_observed_row_l1_norm",
        realPropertyValue(maximum_observed_row_l1_norm));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_line_guard_rejections",
        maximum_line_guard_rejections);
    RecordProperty(
        "wp3_wp7_nitsche_maximum_fraction_absolute_error",
        realPropertyValue(maximum_fraction_absolute_error));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_fraction_relative_error",
        realPropertyValue(maximum_fraction_relative_error));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_operator_reconstruction_error",
        realPropertyValue(
            maximum_operator_reconstruction_error));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_energy_reconstruction_error",
        realPropertyValue(
            maximum_energy_reconstruction_error));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_dry_consistency_boundary_norm",
        realPropertyValue(
            maximum_dry_consistency_boundary_norm));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_dry_penalty_boundary_norm",
        realPropertyValue(
            maximum_dry_penalty_boundary_norm));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_dry_symmetric_boundary_norm",
        realPropertyValue(
            maximum_dry_symmetric_boundary_norm));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_operator_skew",
        realPropertyValue(maximum_operator_skew));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_energy_skew",
        realPropertyValue(maximum_energy_skew));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_eigensolver_off_diagonal_ratio",
        realPropertyValue(
            maximum_eigensolver_off_diagonal_ratio));
    RecordProperty(
        "wp3_wp7_nitsche_minimum_sampled_margin",
        realPropertyValue(minimum_sampled_margin));
    RecordProperty(
        "wp3_wp7_nitsche_minimum_sampled_margin_ratio",
        realPropertyValue(minimum_sampled_margin_ratio));
    RecordProperty(
        "wp3_wp7_nitsche_minimum_sampled_eigenvalue",
        realPropertyValue(minimum_sampled_eigenvalue));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_sampled_eigenvalue",
        realPropertyValue(maximum_sampled_eigenvalue));
    RecordProperty(
        "wp3_wp7_nitsche_minimum_sampled_case",
        minimum_sampled_case);
    RecordProperty(
        "wp3_wp7_nitsche_maximum_active_side_operator_difference",
        realPropertyValue(
            maximum_active_side_operator_difference));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_active_side_energy_difference",
        realPropertyValue(
            maximum_active_side_energy_difference));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_active_side_eigenvalue_difference",
        realPropertyValue(
            maximum_active_side_eigenvalue_difference));
    RecordProperty(
        "wp3_wp7_nitsche_maximum_affine_scale_eigenvalue_spread",
        realPropertyValue(
            maximum_affine_scale_eigenvalue_spread));
    RecordProperty(
        "wp3_wp7_nitsche_uniform_bound_status",
        "UNFROZEN_NO_BOUND_INVENTED");
    RecordProperty(
        "wp3_wp7_nitsche_accepted_claim",
        "joint_low_level_prerequisite");

    std::cout
        << std::setprecision(17)
        << "WP3_WP7_NITSCHE_SUMMARY {"
        << "\"case_count\":" << case_count << ","
        << "\"positive_case_count\":"
        << positive_case_count << ","
        << "\"dry_case_count\":" << dry_case_count << ","
        << "\"dry_exact_zero_case_count\":"
        << dry_exact_zero_case_count << ","
        << "\"aggregation_exercised_case_count\":"
        << aggregation_exercised_case_count << ","
        << "\"diagnostic_structure_verified_case_count\":"
        << diagnostic_structure_verified_case_count << ","
        << "\"generated_active_boundary_marker_count\":"
        << generated_active_boundary_markers.size() << ","
        << "\"maximum_root_path_length\":"
        << nitsche_energy_maximum_root_path_length << ","
        << "\"default_maximum_root_path_length\":"
        << nitsche_energy_default_maximum_root_path_length << ","
        << "\"default_root_path_guard_rejection_verified\":"
        << (default_root_path_guard_rejection_verified
                ? "true"
                : "false")
        << ","
        << "\"maximum_observed_root_path\":"
        << maximum_observed_root_path << ","
        << "\"maximum_root_path_guard_rejections\":"
        << maximum_root_path_guard_rejections << ","
        << "\"maximum_reference_extrapolation_distance\":"
        << nitsche_energy_maximum_reference_extrapolation_distance
        << ","
        << "\"maximum_observed_reference_extrapolation\":"
        << maximum_observed_reference_extrapolation << ","
        << "\"maximum_extrapolation_guard_rejections\":"
        << maximum_extrapolation_guard_rejections << ","
        << "\"maximum_absolute_coefficient\":"
        << nitsche_energy_maximum_absolute_coefficient << ","
        << "\"maximum_observed_absolute_coefficient\":"
        << maximum_observed_absolute_coefficient << ","
        << "\"maximum_row_l1_norm\":"
        << nitsche_energy_maximum_row_l1_norm << ","
        << "\"maximum_observed_row_l1_norm\":"
        << maximum_observed_row_l1_norm << ","
        << "\"maximum_line_guard_rejections\":"
        << maximum_line_guard_rejections << ","
        << "\"maximum_fraction_absolute_error\":"
        << maximum_fraction_absolute_error << ","
        << "\"maximum_fraction_relative_error\":"
        << maximum_fraction_relative_error << ","
        << "\"maximum_operator_reconstruction_error\":"
        << maximum_operator_reconstruction_error << ","
        << "\"maximum_energy_reconstruction_error\":"
        << maximum_energy_reconstruction_error << ","
        << "\"maximum_dry_consistency_boundary_norm\":"
        << maximum_dry_consistency_boundary_norm << ","
        << "\"maximum_dry_penalty_boundary_norm\":"
        << maximum_dry_penalty_boundary_norm << ","
        << "\"maximum_dry_symmetric_boundary_norm\":"
        << maximum_dry_symmetric_boundary_norm << ","
        << "\"maximum_operator_skew\":"
        << maximum_operator_skew << ","
        << "\"maximum_energy_skew\":"
        << maximum_energy_skew << ","
        << "\"maximum_eigensolver_off_diagonal_ratio\":"
        << maximum_eigensolver_off_diagonal_ratio << ","
        << "\"minimum_sampled_margin\":"
        << minimum_sampled_margin << ","
        << "\"minimum_sampled_margin_ratio\":"
        << minimum_sampled_margin_ratio << ","
        << "\"minimum_sampled_eigenvalue\":"
        << minimum_sampled_eigenvalue << ","
        << "\"maximum_sampled_eigenvalue\":"
        << maximum_sampled_eigenvalue << ","
        << "\"minimum_sampled_case\":\""
        << minimum_sampled_case << "\","
        << "\"maximum_active_side_operator_difference\":"
        << maximum_active_side_operator_difference << ","
        << "\"maximum_active_side_energy_difference\":"
        << maximum_active_side_energy_difference << ","
        << "\"maximum_active_side_eigenvalue_difference\":"
        << maximum_active_side_eigenvalue_difference << ","
        << "\"maximum_affine_scale_eigenvalue_spread\":"
        << maximum_affine_scale_eigenvalue_spread << ","
        << "\"method_coercivity_lower_bound\":null,"
        << "\"uniform_bound_status\":"
        << "\"UNFROZEN_NO_BOUND_INVENTED\","
        << "\"accepted_claim\":"
        << "\"joint_low_level_prerequisite\"}"
        << '\n';
#endif
}

TEST(FreeSurfaceCutStabilityMPI,
     PhysicalWetBlocksAndDisconnectedIslandsAreInvariantAcrossDryMPIData)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH && \
      defined(FE_HAS_MPI) && defined(MESH_HAS_MPI))
    GTEST_SKIP() << "Distributed wet-block invariance requires MPI-enabled FE and Mesh.";
#else
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized == 0) {
        GTEST_SKIP() << "Run this test under mpiexec.";
    }
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }

    constexpr std::array<FE::Real, 4> baseline_x = {
        FE::Real{-1.0}, FE::Real{0.0}, FE::Real{1.0}, FE::Real{2.0}};
    constexpr std::array<FE::Real, 4> baseline_phi = {
        FE::Real{-1.25}, FE::Real{-0.25}, FE::Real{0.75},
        FE::Real{1.75}};
    constexpr std::array<FE::Real, 7> deep_x = {
        FE::Real{-1.0}, FE::Real{0.0}, FE::Real{1.0}, FE::Real{2.0},
        FE::Real{3.0}, FE::Real{4.0}, FE::Real{5.0}};
    constexpr std::array<FE::Real, 7> deep_phi = {
        FE::Real{-1.25}, FE::Real{-0.25}, FE::Real{0.75},
        FE::Real{1.75}, FE::Real{2.75}, FE::Real{3.75},
        FE::Real{4.75}};
    constexpr std::array<FE::Real, 7> island_x = {
        FE::Real{0.0}, FE::Real{1.0}, FE::Real{2.0}, FE::Real{3.0},
        FE::Real{4.0}, FE::Real{5.0}, FE::Real{6.0}};
    constexpr std::array<FE::Real, 7> island_phi = {
        FE::Real{-1.0}, FE::Real{-1.0}, FE::Real{1.0}, FE::Real{1.0},
        FE::Real{1.0}, FE::Real{-1.0}, FE::Real{-1.0}};
    constexpr FE::Real mpi_gate = FE::Real{1.0e-9};
    constexpr FE::Real mpi_solved_gate = FE::Real{1.0e-8};
    constexpr FE::Real norm_floor = FE::Real{1.0e-12};
    constexpr std::string_view partition_method = "block";

    const auto baseline = assembleDistributedWetBlockSample(
        baseline_x,
        baseline_phi,
        /*dry_state_scale=*/FE::Real{3.0},
        comm,
        partition_method);
    const auto depth = assembleDistributedWetBlockSample(
        deep_x,
        deep_phi,
        /*dry_state_scale=*/FE::Real{3.0},
        comm,
        partition_method);
    const auto dry_state = assembleDistributedWetBlockSample(
        baseline_x,
        baseline_phi,
        // Includes the dry-only exterior vertices on both MPI owners.
        /*dry_state_scale=*/FE::Real{1.0e6},
        comm,
        partition_method);
    ASSERT_EQ(baseline.retained_vertices, 6u);
    ASSERT_EQ(depth.retained_vertices, baseline.retained_vertices);
    EXPECT_EQ(baseline.constrained_dry_velocity_dofs, 4u);
    EXPECT_EQ(baseline.constrained_dry_pressure_dofs, 2u);
    EXPECT_EQ(depth.constrained_dry_velocity_dofs, 16u);
    EXPECT_EQ(depth.constrained_dry_pressure_dofs, 8u);
    EXPECT_GT(vectorL2Norm(baseline.residual), FE::Real{0.0});
    EXPECT_GT(vectorL2Norm(baseline.jacobian), FE::Real{0.0});

    const std::array<std::pair<std::string_view, ScaledWetBlockDifference>, 2>
        comparisons = {{
            {"dry_depth", compareWetBlockSamples(baseline, depth)},
            {"exterior_dry_values",
             compareWetBlockSamples(baseline, dry_state)},
        }};
    for (const auto& [factor, difference] : comparisons) {
        SCOPED_TRACE(factor);
        EXPECT_LE(difference.residual, mpi_gate);
        EXPECT_LE(difference.jacobian, mpi_gate);
        EXPECT_LE(difference.solved_state, mpi_solved_gate);
        if (rank == 0) {
            std::cout << std::setprecision(17)
                      << "WP1_wet_block_invariance"
                      << " scope=mpi"
                      << " ranks=" << size
                      << " partition=" << partition_method
                      << " factor=" << factor
                      << " residual_absolute_floor=" << norm_floor
                      << " jacobian_absolute_floor=" << norm_floor
                      << " scaled_residual_difference="
                      << difference.residual
                      << " scaled_jacobian_difference="
                      << difference.jacobian
                      << " scaled_solved_state_difference="
                      << difference.solved_state
                      << " residual_absolute_difference="
                      << difference.residual_absolute
                      << " jacobian_absolute_difference="
                      << difference.jacobian_absolute
                      << " solved_state_absolute_difference="
                      << difference.solved_state_absolute
                      << " accepted_gate=" << mpi_gate
                      << " solved_state_gate=" << mpi_solved_gate << '\n';
        }
    }
    for (const auto* sample : {&baseline, &depth, &dry_state}) {
        EXPECT_LE(sample->dry_column_coupling_norm /
                      (norm_floor + vectorL2Norm(sample->jacobian)),
                  mpi_gate);
    }

    const auto islands = assembleDistributedWetBlockSample(
        island_x,
        island_phi,
        /*dry_state_scale=*/FE::Real{2.0},
        comm,
        partition_method);
    const auto changed_dry_path = assembleDistributedWetBlockSample(
        island_x,
        island_phi,
        /*dry_state_scale=*/FE::Real{1.0e6},
        comm,
        partition_method);
    ASSERT_EQ(islands.retained_vertices, 12u);
    ASSERT_EQ(islands.dofs.size(), 36u);
    const auto dry_path_difference = compareWetBlockSamples(
        islands, changed_dry_path);
    EXPECT_LE(dry_path_difference.residual, mpi_gate);
    EXPECT_LE(dry_path_difference.jacobian, mpi_gate);
    EXPECT_LE(dry_path_difference.solved_state, mpi_solved_gate);

    long double cross_squared = 0.0L;
    const auto n = islands.dofs.size();
    for (std::size_t row = 0u; row < n; ++row) {
        const bool left_row = islands.dofs[row].point[0] <= FE::Real{2.0};
        const bool right_row = islands.dofs[row].point[0] >= FE::Real{4.0};
        ASSERT_TRUE(left_row || right_row);
        for (std::size_t column = 0u; column < n; ++column) {
            const bool left_column =
                islands.dofs[column].point[0] <= FE::Real{2.0};
            const bool right_column =
                islands.dofs[column].point[0] >= FE::Real{4.0};
            ASSERT_TRUE(left_column || right_column);
            if ((left_row && right_column) ||
                (right_row && left_column)) {
                const auto value = static_cast<long double>(
                    islands.jacobian[row * n + column]);
                cross_squared += value * value;
            }
        }
    }
    const auto cross_norm =
        static_cast<FE::Real>(std::sqrt(cross_squared));
    const auto scaled_cross = cross_norm /
        (norm_floor + vectorL2Norm(islands.jacobian));
    EXPECT_LE(scaled_cross, mpi_gate);
    EXPECT_LE(islands.dry_column_coupling_norm /
                  (norm_floor + vectorL2Norm(islands.jacobian)),
              mpi_gate);
    if (rank == 0) {
        std::cout << std::setprecision(17)
                  << "WP1_two_island_decoupling"
                  << " scope=mpi"
                  << " ranks=" << size
                  << " partition=" << partition_method
                  << " retained_vertices=" << islands.retained_vertices
                  << " scaled_cross_jacobian=" << scaled_cross
                  << " scaled_dry_path_residual_difference="
                  << dry_path_difference.residual
                  << " scaled_dry_path_jacobian_difference="
                  << dry_path_difference.jacobian
                  << " scaled_dry_path_solved_state_difference="
                  << dry_path_difference.solved_state
                  << " accepted_gate=" << mpi_gate
                  << " solved_state_gate=" << mpi_solved_gate << '\n';
    }
#endif
}

TEST(FreeSurfaceCutStabilityMPI,
     FourRankFixedCutIsInvariantAcrossBlockAndMetisPartitions)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH && \
      defined(FE_HAS_MPI) && defined(MESH_HAS_MPI))
    GTEST_SKIP() << "Four-rank cut stability requires MPI-enabled FE and Mesh.";
#else
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized == 0) {
        GTEST_SKIP() << "Run this test under mpiexec.";
    }
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size != 4) {
        GTEST_SKIP() << "This test requires exactly four MPI ranks.";
    }

    const ScopedEnvVar pressure_diagnostics(
        "SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC", "1");
    const PlaneCutPosition cut{
        "four_rank_oblique_cut", {{1.0, 0.73, 0.41}}, 2.25};
    const std::array<std::string, 2> partition_methods = {
        "block", "metis"};
    std::array<StabilitySample, 2> samples;
    std::array<std::uint64_t, 2> partition_hashes{};

    for (std::size_t partition = 0u;
         partition < partition_methods.size();
         ++partition) {
        SCOPED_TRACE(partition_methods[partition]);
        DistributedStabilityProblem problem(
            cut, comm, partition_methods[partition]);
        partition_hashes[partition] = problem.partitionHash();
        samples[partition] = problem.evaluate(cut);
        const auto& sample = samples[partition];

        EXPECT_GT(sample.cut_cells, 0u);
        EXPECT_GT(sample.cut_adjacent_facets, 0u);
        EXPECT_GT(sample.backend_volume_quadrature_points, 0u);
        EXPECT_EQ(sample.backend_fallback_cells, 0u);
        EXPECT_TRUE(sample.pressure_natural_traction_anchor);
        EXPECT_TRUE(sample.pressure_anchor_has_no_gauge_enforcement);
        EXPECT_GT(sample.pressure_constraints.master_bearing, 0u);
        EXPECT_EQ(sample.velocity_constraints.master_bearing,
                  3u * sample.pressure_constraints.master_bearing);
        EXPECT_EQ(sample.velocity_constraints.homogeneous_pins,
                  3u * sample.pressure_constraints.homogeneous_pins);
        EXPECT_GT(sample.free_pressure_dofs, 0u);
        EXPECT_EQ(sample.zero_free_pressure_rows, 0u);
        EXPECT_GT(sample.pressure_ghost_norm, FE::Real{1.0e-14});
        EXPECT_GT(sample.pspg_pressure_gradient_norm, FE::Real{1.0e-14});
        EXPECT_EQ(sample.equilibrated_rank, sample.equilibrated_size);
        EXPECT_TRUE(std::isfinite(sample.equilibrated_condition_inf));
        EXPECT_LE(sample.equilibrated_condition_inf, FE::Real{1000.0});
    }

    EXPECT_NE(partition_hashes[0], partition_hashes[1])
        << "four-rank block and METIS partitions must exercise distinct owner maps";
    const auto expect_equal_operator =
        [&](std::span<const FE::Real> block,
            std::span<const FE::Real> metis,
            std::string_view name) {
            EXPECT_EQ(metis.size(), block.size()) << name;
            if (metis.size() != block.size()) {
                return std::numeric_limits<FE::Real>::infinity();
            }
            const auto difference = compareDenseOperators(block, metis);
            const auto tolerance =
                FE::Real{2048.0} *
                std::numeric_limits<FE::Real>::epsilon() *
                std::max(FE::Real{1.0},
                         difference.maximum_absolute_entry);
            EXPECT_LE(difference.maximum_absolute_difference, tolerance)
                << "operator=" << name
                << " flat_index=" << difference.maximum_difference_index;
            return difference.maximum_absolute_difference;
        };
    const auto mixed_difference = expect_equal_operator(
        samples[0].canonical_mixed_operator,
        samples[1].canonical_mixed_operator,
        "mixed Jacobian");
    const auto ghost_difference = expect_equal_operator(
        samples[0].canonical_pressure_ghost_operator,
        samples[1].canonical_pressure_ghost_operator,
        "pressure ghost penalty");
    const auto pspg_difference = expect_equal_operator(
        samples[0].canonical_pressure_pspg_operator,
        samples[1].canonical_pressure_pspg_operator,
        "pressure PSPG block");
    EXPECT_EQ(samples[1].cut_cells, samples[0].cut_cells);
    EXPECT_EQ(samples[1].cut_adjacent_facets,
              samples[0].cut_adjacent_facets);
    EXPECT_EQ(samples[1].cut_adjacent_facet_gid_hash,
              samples[0].cut_adjacent_facet_gid_hash);
    EXPECT_EQ(samples[1].free_velocity_dofs,
              samples[0].free_velocity_dofs);
    EXPECT_EQ(samples[1].free_pressure_dofs,
              samples[0].free_pressure_dofs);
    EXPECT_NEAR(samples[1].pressure_ghost_norm,
                samples[0].pressure_ghost_norm,
                FE::Real{1.0e-12});
    EXPECT_NEAR(samples[1].pspg_pressure_gradient_norm,
                samples[0].pspg_pressure_gradient_norm,
                FE::Real{1.0e-10});
    EXPECT_NEAR(samples[1].equilibrated_condition_inf,
                samples[0].equilibrated_condition_inf,
                FE::Real{1.0e-8});

    if (rank == 0) {
        std::cout << std::setprecision(17)
                  << "WP7_four_rank_partition_invariance"
                  << " ranks=" << size
                  << " cut_cells=" << samples[0].cut_cells
                  << " cut_adjacent_facets="
                  << samples[0].cut_adjacent_facets
                  << " free_velocity_dofs="
                  << samples[0].free_velocity_dofs
                  << " free_pressure_dofs="
                  << samples[0].free_pressure_dofs
                  << " block_condition="
                  << samples[0].equilibrated_condition_inf
                  << " metis_condition="
                  << samples[1].equilibrated_condition_inf
                  << " maximum_mixed_operator_difference="
                  << mixed_difference
                  << " maximum_pressure_ghost_operator_difference="
                  << ghost_difference
                  << " maximum_pressure_pspg_operator_difference="
                  << pspg_difference
                  << " qualification=finite_four_rank_partition_invariance"
                  << '\n';
    }
#endif
}

TEST(FreeSurfaceCutStabilityMPI,
     TwoRankFractionOrientationRegimeMatrixMatchesSerial)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH && \
      defined(FE_HAS_MPI) && defined(MESH_HAS_MPI))
    GTEST_SKIP() << "Two-rank matrix requires MPI-enabled FE and Mesh.";
#else
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized == 0) {
        GTEST_SKIP() << "Run this test under mpiexec.";
    }
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }
    runDistributedFrozenMatrix(/*expected_ranks=*/2);
#endif
}

TEST(FreeSurfaceCutStabilityMPI,
     FourRankFractionOrientationRegimeMatrixMatchesSerial)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH && \
      defined(FE_HAS_MPI) && defined(MESH_HAS_MPI))
    GTEST_SKIP() << "Four-rank matrix requires MPI-enabled FE and Mesh.";
#else
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized == 0) {
        GTEST_SKIP() << "Run this test under mpiexec.";
    }
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        GTEST_SKIP() << "This test requires exactly four MPI ranks.";
    }
    runDistributedFrozenMatrix(/*expected_ranks=*/4);
#endif
}

TEST(FreeSurfaceCutStabilityMPI,
     LimitedMetisHaloFailsClosedOnIncompleteAggregationSupport)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH && \
      defined(FE_HAS_MPI) && defined(MESH_HAS_MPI))
    GTEST_SKIP() << "Distributed cut stability requires MPI-enabled FE and Mesh.";
#else
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized == 0) {
        GTEST_SKIP() << "Run this test under mpiexec.";
    }
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }

    const PlaneCutPosition cut{
        "first_cube_cut", {{1.0, 0.0, 0.0}}, 1.25};
    DistributedStabilityProblem problem(
        cut, comm, "metis", /*ghost_layers=*/2);

    std::string diagnostic;
    try {
        (void)problem.evaluate(cut);
        FAIL() << "depth-2 METIS overlap was accepted despite incomplete "
                  "aggregate support";
    } catch (const std::runtime_error& error) {
        diagnostic = error.what();
    }
    const bool canonical_facet_set_mismatch =
        diagnostic.find(
            "distributed stability ranks generated different physical "
            "cut-adjacent facet sets") != std::string::npos;
    const bool incomplete_aggregation_support =
        diagnostic.find("incomplete_distributed_aggregation_halo") !=
            std::string::npos &&
        diagnostic.find("candidate_or_root_support_mismatch") !=
            std::string::npos &&
        diagnostic.find("Increase the mesh ghost overlap") !=
            std::string::npos;
    ASSERT_TRUE(canonical_facet_set_mismatch ||
                incomplete_aggregation_support)
        << "depth-2 METIS overlap did not fail closed with a recognized "
           "insufficient-halo diagnostic: "
        << diagnostic;
    if (rank == 0) {
        std::cout << "FS14_limited_halo_fail_closed diagnostic='"
                  << diagnostic << "'\n";
    }
#endif
}

TEST(FreeSurfaceCutStabilityMPI,
     DistributedMovingCutRemainsStableAcrossBlockAndMetisPartitions)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH && \
      defined(FE_HAS_MPI) && defined(MESH_HAS_MPI))
    GTEST_SKIP() << "Distributed cut stability requires MPI-enabled FE and Mesh.";
#else
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized == 0) {
        GTEST_SKIP() << "Run this test under mpiexec.";
    }
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size != 2) {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }

    const ScopedEnvVar pressure_diagnostics(
        "SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC", "1");
    const std::vector<PlaneCutPosition> positions = {
        {"first_cube_cut", {{1.0, 0.0, 0.0}}, 1.25},
        {"second_cube_cut", {{1.0, 0.0, 0.0}}, 2.25},
        {"revisit_first_cube_cut", {{1.0, 0.0, 0.0}}, 1.25},
    };
    const std::array<std::string, 2> partition_methods = {
        "block", "metis"};
    std::array<std::vector<StabilitySample>, 2> partition_samples;
    std::array<std::uint64_t, 2> partition_hashes{};

    for (std::size_t partition = 0;
         partition < partition_methods.size();
         ++partition) {
        SCOPED_TRACE(partition_methods[partition]);
        DistributedStabilityProblem problem(
            positions.front(), comm, partition_methods[partition]);
        partition_hashes[partition] = problem.partitionHash();
        auto& samples = partition_samples[partition];
        samples.reserve(positions.size());

        FE::Real min_condition = std::numeric_limits<FE::Real>::infinity();
        FE::Real max_condition = 0.0;
        for (const auto& position : positions) {
            SCOPED_TRACE(position.label);
            samples.push_back(problem.evaluate(position));
            const auto& sample = samples.back();

            EXPECT_GT(sample.cut_cells, 0u);
            EXPECT_GT(sample.cut_adjacent_facets, 0u);
            EXPECT_GT(sample.backend_volume_quadrature_points, 0u);
            EXPECT_EQ(sample.backend_fallback_cells, 0u);
            EXPECT_TRUE(sample.pressure_natural_traction_anchor);
            EXPECT_TRUE(sample.pressure_anchor_has_no_gauge_enforcement)
                << "the distributed free-surface pressure datum must not "
                   "install an algebraic pressure gauge";
            EXPECT_GT(sample.pressure_constraints.master_bearing, 0u);
            EXPECT_EQ(sample.velocity_constraints.master_bearing,
                      3u * sample.pressure_constraints.master_bearing);
            EXPECT_EQ(sample.velocity_constraints.homogeneous_pins,
                      3u * sample.pressure_constraints.homogeneous_pins)
                << "homogeneous constraints must be component copies of "
                   "active-support removal, not an unmatched pressure pin";
            EXPECT_GT(sample.free_pressure_dofs, 0u);
            EXPECT_EQ(sample.zero_free_pressure_rows, 0u);
            EXPECT_GT(sample.pressure_ghost_norm, FE::Real{1.0e-14});
            EXPECT_GT(sample.pspg_pressure_gradient_norm, FE::Real{1.0e-14});
            EXPECT_EQ(sample.equilibrated_rank, sample.equilibrated_size);
            EXPECT_TRUE(std::isfinite(sample.equilibrated_condition_inf));
            EXPECT_GT(sample.equilibrated_condition_inf, FE::Real{0.0});
            min_condition = std::min(
                min_condition, sample.equilibrated_condition_inf);
            max_condition = std::max(
                max_condition, sample.equilibrated_condition_inf);

            if (rank == 0) {
                std::cout << std::setprecision(12)
                          << "FS14_distributed_cut"
                          << " partition=" << partition_methods[partition]
                          << " position=" << sample.label
                          << " cut_cells=" << sample.cut_cells
                          << " cut_adjacent_facets="
                          << sample.cut_adjacent_facets
                          << " pressure_aggregate_slaves="
                          << sample.pressure_constraints.master_bearing
                          << " pressure_homogeneous_support_constraints="
                          << sample.pressure_constraints.homogeneous_pins
                          << " free_pressure_dofs="
                          << sample.free_pressure_dofs
                          << " zero_free_pressure_rows="
                          << sample.zero_free_pressure_rows
                          << " pressure_ghost_norm="
                          << sample.pressure_ghost_norm
                          << " pspg_pressure_gradient_norm="
                          << sample.pspg_pressure_gradient_norm
                          << " equilibrated_condition_inf="
                          << sample.equilibrated_condition_inf << '\n';
            }
        }

        ASSERT_EQ(samples.size(), positions.size());
        EXPECT_NE(samples[0].pressure_constraints.homogeneous_pins,
                  samples[1].pressure_constraints.homogeneous_pins)
            << "moving cut did not change active-support constraint topology";
        EXPECT_NE(samples[0].free_pressure_dofs,
                  samples[1].free_pressure_dofs)
            << "moving cut did not change the global free pressure space";

        const auto& first = samples[0];
        const auto& revisit = samples[2];
        EXPECT_EQ(revisit.cut_cells, first.cut_cells);
        EXPECT_EQ(revisit.cut_adjacent_facets, first.cut_adjacent_facets);
        EXPECT_EQ(revisit.pressure_constraints.master_bearing,
                  first.pressure_constraints.master_bearing);
        EXPECT_EQ(revisit.pressure_constraints.homogeneous_pins,
                  first.pressure_constraints.homogeneous_pins);
        EXPECT_EQ(revisit.free_pressure_dofs, first.free_pressure_dofs);
        EXPECT_NEAR(revisit.pressure_ghost_norm,
                    first.pressure_ghost_norm,
                    FE::Real{1.0e-11});
        EXPECT_NEAR(revisit.pspg_pressure_gradient_norm,
                    first.pspg_pressure_gradient_norm,
                    FE::Real{1.0e-11});
        EXPECT_NEAR(revisit.equilibrated_condition_inf,
                    first.equilibrated_condition_inf,
                    FE::Real{1.0e-8});

        constexpr FE::Real maximum_accepted_condition_inf =
            FE::Real{1000.0};
        constexpr FE::Real maximum_accepted_sequence_spread = FE::Real{5.0};
        EXPECT_LE(max_condition, maximum_accepted_condition_inf);
        EXPECT_LE(max_condition / min_condition,
                  maximum_accepted_sequence_spread);
    }

    EXPECT_NE(partition_hashes[0], partition_hashes[1])
        << "block and METIS produced the same cell-owner map; the test did "
           "not exercise partition variation";
    ASSERT_EQ(partition_samples[0].size(), partition_samples[1].size());
    FE::Real maximum_partition_condition_ratio = FE::Real{1.0};
    FE::Real maximum_partition_pressure_ghost_ratio = FE::Real{1.0};
    FE::Real maximum_partition_mixed_operator_difference = FE::Real{0.0};
    FE::Real maximum_partition_pressure_ghost_operator_difference =
        FE::Real{0.0};
    FE::Real maximum_partition_pressure_pspg_operator_difference =
        FE::Real{0.0};
    for (std::size_t position = 0;
         position < partition_samples[0].size();
         ++position) {
        SCOPED_TRACE(positions[position].label);
        const auto& block = partition_samples[0][position];
        const auto& metis = partition_samples[1][position];
        EXPECT_EQ(metis.cut_cells, block.cut_cells);
        EXPECT_EQ(metis.cut_adjacent_facets, block.cut_adjacent_facets);
        EXPECT_EQ(metis.cut_adjacent_facet_gid_hash,
                  block.cut_adjacent_facet_gid_hash);
        EXPECT_EQ(metis.velocity_constraints.master_bearing,
                  block.velocity_constraints.master_bearing);
        EXPECT_EQ(metis.velocity_constraints.homogeneous_pins,
                  block.velocity_constraints.homogeneous_pins);
        EXPECT_EQ(metis.pressure_constraints.master_bearing,
                  block.pressure_constraints.master_bearing);
        EXPECT_EQ(metis.pressure_constraints.homogeneous_pins,
                  block.pressure_constraints.homogeneous_pins);
        EXPECT_EQ(metis.free_velocity_dofs, block.free_velocity_dofs);
        EXPECT_EQ(metis.free_pressure_dofs, block.free_pressure_dofs);
        EXPECT_EQ(metis.zero_free_pressure_rows,
                  block.zero_free_pressure_rows);
        EXPECT_EQ(metis.equilibrated_rank, block.equilibrated_rank);
        EXPECT_EQ(metis.equilibrated_size, block.equilibrated_size);
        EXPECT_NEAR(metis.reference_active_volume,
                    block.reference_active_volume,
                    FE::Real{1.0e-12});
        EXPECT_NEAR(metis.pspg_pressure_gradient_norm,
                    block.pspg_pressure_gradient_norm,
                    FE::Real{1.0e-10});
        const auto condition_ratio = std::max(
            metis.equilibrated_condition_inf /
                block.equilibrated_condition_inf,
            block.equilibrated_condition_inf /
                metis.equilibrated_condition_inf);
        const auto pressure_ghost_ratio = std::max(
            metis.pressure_ghost_norm / block.pressure_ghost_norm,
            block.pressure_ghost_norm / metis.pressure_ghost_norm);
        maximum_partition_condition_ratio = std::max(
            maximum_partition_condition_ratio, condition_ratio);
        maximum_partition_pressure_ghost_ratio = std::max(
            maximum_partition_pressure_ghost_ratio,
            pressure_ghost_ratio);

        // Owner-contiguous numbering is partition dependent, so evaluate()
        // stores each free P1 operator in physical vertex-GID/component order.
        // The tolerance is a roundoff allowance for different local summation
        // orders; an omitted owned-row facet contribution is many orders of
        // magnitude larger (and was a factor-of-two ghost-norm error here).
        const auto expect_partition_operator_equality =
            [&](std::span<const FE::Real> block_operator,
                std::span<const FE::Real> metis_operator,
                std::string_view operator_name) {
                EXPECT_EQ(metis_operator.size(), block_operator.size())
                    << operator_name;
                if (metis_operator.size() != block_operator.size()) {
                    return DenseOperatorDifference{};
                }
                const auto difference = compareDenseOperators(
                    block_operator, metis_operator);
                const auto tolerance =
                    FE::Real{2048.0} *
                    std::numeric_limits<FE::Real>::epsilon() *
                    std::max(FE::Real{1.0},
                             difference.maximum_absolute_entry);
                EXPECT_LE(difference.maximum_absolute_difference, tolerance)
                    << "operator=" << operator_name
                    << " flat_index="
                    << difference.maximum_difference_index
                    << " scale=" << difference.maximum_absolute_entry
                    << " tolerance=" << tolerance;
                return difference;
            };
        const auto mixed_difference =
            expect_partition_operator_equality(
                block.canonical_mixed_operator,
                metis.canonical_mixed_operator,
                "mixed Jacobian");
        const auto pressure_ghost_difference =
            expect_partition_operator_equality(
                block.canonical_pressure_ghost_operator,
                metis.canonical_pressure_ghost_operator,
                "pressure ghost penalty");
        const auto pressure_pspg_difference =
            expect_partition_operator_equality(
                block.canonical_pressure_pspg_operator,
                metis.canonical_pressure_pspg_operator,
                "pressure PSPG block");
        maximum_partition_mixed_operator_difference = std::max(
            maximum_partition_mixed_operator_difference,
            mixed_difference.maximum_absolute_difference);
        maximum_partition_pressure_ghost_operator_difference = std::max(
            maximum_partition_pressure_ghost_operator_difference,
            pressure_ghost_difference.maximum_absolute_difference);
        maximum_partition_pressure_pspg_operator_difference = std::max(
            maximum_partition_pressure_pspg_operator_difference,
            pressure_pspg_difference.maximum_absolute_difference);

        EXPECT_NEAR(metis.pressure_ghost_norm,
                    block.pressure_ghost_norm,
                    FE::Real{1.0e-12});
        EXPECT_NEAR(metis.equilibrated_condition_inf,
                    block.equilibrated_condition_inf,
                    FE::Real{1.0e-8});
    }
    if (rank == 0) {
        std::cout << "FS14_distributed_cut_summary"
                  << " positions=" << positions.size()
                  << " partitions=" << partition_methods.size()
                  << " block_owner_hash=" << partition_hashes[0]
                  << " metis_owner_hash=" << partition_hashes[1]
                  << " maximum_partition_condition_ratio="
                  << maximum_partition_condition_ratio
                  << " maximum_partition_pressure_ghost_ratio="
                  << maximum_partition_pressure_ghost_ratio
                  << " maximum_partition_mixed_operator_difference="
                  << maximum_partition_mixed_operator_difference
                  << " maximum_partition_pressure_ghost_operator_difference="
                  << maximum_partition_pressure_ghost_operator_difference
                  << " maximum_partition_pressure_pspg_operator_difference="
                  << maximum_partition_pressure_pspg_operator_difference
                  << " qualification=finite_partition_operator_equality_not_inf_sup_proof"
                  << '\n';
    }
#endif
}

} // namespace svmp::Physics::test
