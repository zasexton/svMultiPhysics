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
 * gauge constraint.  The test removes only active-support/aggregation
 * constrained rows and columns, checks every remaining pressure row, and
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
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp::Physics::test {
namespace {

namespace ns = formulations::navier_stokes;

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

struct StabilitySample {
    std::string label;
    FE::Real minimum_active_cut_fraction{1.0};
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

    return create_mesh(std::move(base));
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

#if defined(FE_HAS_MPI) && defined(MESH_HAS_MPI)

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
    const auto find_free = [&](FE::GlobalIndex global) -> std::size_t {
        const auto found = std::lower_bound(
            free_dofs.begin(), free_dofs.end(), global);
        if (found == free_dofs.end() || *found != global) {
            throw std::runtime_error(
                "closed field constraint retains a constrained master");
        }
        return static_cast<std::size_t>(found - free_dofs.begin());
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

[[nodiscard]] ns::IncompressibleNavierStokesVMSOptions stabilityOptions(
    int interface_marker,
    std::string domain_id)
{
    ns::IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = "u";
    options.pressure_field_name = "p";
    options.density = 1.0;
    options.viscosity = 0.01;
    options.enable_convection = false;
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
                                        int cells_per_axis = 2)
        : mesh_(makeFixedTetraMesh(initial_cut, cells_per_axis)),
          velocity_space_(FE::spaces::SpaceFactory::create_vector_h1(
              FE::ElementType::Tetra4, /*order=*/1, /*components=*/3)),
          pressure_space_(FE::spaces::SpaceFactory::create_h1(
              FE::ElementType::Tetra4, /*order=*/1)),
          system_(mesh_),
          cells_per_axis_(cells_per_axis),
          mesh_spacing_(FE::Real{2.0} /
                        static_cast<FE::Real>(cells_per_axis))
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
            interface_marker, std::string(domain_id));

        ns::IncompressibleNavierStokesVMSModule module(
            velocity_space_, pressure_space_, options);
        module.registerOn(system_);
        system_.setup({});

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
        previous_ = solution_;
    }

    [[nodiscard]] StabilitySample evaluate(const PlaneCutPosition& cut)
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
        state.dt = 0.1;
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

        auto free_velocity = freeFieldDofs(system_, velocity_);
        auto free_pressure = freeFieldDofs(system_, pressure_);
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
        sample.equilibrated_rank = diagnostics.rank;
        sample.equilibrated_size = free_mixed.size();
        sample.equilibrated_smallest_singular_value =
            diagnostics.smallest_retained_singular_value;
        sample.equilibrated_largest_singular_value =
            diagnostics.largest_singular_value;
        sample.equilibrated_condition_inf =
            infinityNormCondition(reduced, free_mixed.size());

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

struct CanonicalP1Dof {
    gid_t vertex_gid{0};
    std::size_t component{0u};
    FE::GlobalIndex global_dof{FE::INVALID_GLOBAL_INDEX};
};

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
    const auto* entity_map =
        system.fieldDofHandler(field).getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error(
            "distributed stability P1 field has no entity DOF map");
    }

    const auto& base = mesh.base();
    const auto& vertex_gids = base.vertex_gids();
    if (vertex_gids.size() != base.n_vertices()) {
        throw std::runtime_error(
            "distributed stability mesh has incomplete vertex GIDs");
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
                "distributed stability field is not the expected P1 layout");
        }
        for (std::size_t component = 0u;
             component < components;
             ++component) {
            const auto global = offset + vertex_dofs[component];
            if (global < 0 ||
                static_cast<std::size_t>(global) >= constrained.size()) {
                throw std::runtime_error(
                    "distributed stability vertex DOF is out of range");
            }
            if (constrained[static_cast<std::size_t>(global)] == 0) {
                canonical.push_back(CanonicalP1Dof{
                    vertex_gids[static_cast<std::size_t>(vertex)],
                    component,
                    global});
            }
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
                "distributed stability canonical P1 DOF is duplicated");
        }
    }

    std::vector<FE::GlobalIndex> dofs;
    dofs.reserve(canonical.size());
    for (const auto& entry : canonical) {
        dofs.push_back(entry.global_dof);
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

[[nodiscard]] std::uint64_t distributedCellOwnerHash(
    const Mesh& mesh,
    MPI_Comm comm)
{
    constexpr std::size_t global_cells = 18u;
    std::array<int, global_cells> local_owner{};
    std::array<int, global_cells> global_owner{};
    local_owner.fill(-1);
    global_owner.fill(-1);
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
        int ghost_layers = 18)
        : comm_(comm),
          mesh_(makeDistributedTetraStripMesh(
              initial_cut, comm, partition_method, ghost_layers)),
          velocity_space_(FE::spaces::SpaceFactory::create_vector_h1(
              FE::ElementType::Tetra4, /*order=*/1, /*components=*/3)),
          pressure_space_(FE::spaces::SpaceFactory::create_h1(
              FE::ElementType::Tetra4, /*order=*/1)),
          system_(mesh_)
    {
        int rank = 0;
        int size = 1;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &size);
        if (size != 2) {
            throw std::runtime_error(
                "distributed stability problem requires exactly two ranks");
        }
        if (mesh_->global_n_cells() != 18u ||
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
            interface_marker, std::string(domain_id));
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
        previous_ = solution_;
        partition_hash_ = distributedCellOwnerHash(*mesh_, comm_);
    }

    [[nodiscard]] std::uint64_t partitionHash() const noexcept
    {
        return partition_hash_;
    }

    [[nodiscard]] StabilitySample evaluate(const PlaneCutPosition& cut)
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
        state.dt = 0.1;
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
        // This finite fixture intentionally retains its complete 18-cell mesh
        // in overlap.  Canonical face GIDs above prove that both ranks see the
        // same physical facet set; counting "first-cell owned" is invalid
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
};

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
