/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Classification-only (compact) storage of the dry full-cell records of an
// authoritative free-surface geometry snapshot.  Every consumer must get the
// same answers from the compact snapshot as from the fully materialized one,
// either from the retained classification or by exact recomputation.

#include "Assembly/CutIntegrationContext.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Geometry/CutQuadratureMapping.h"
#include "Interfaces/CopyOnWriteVector.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "LevelSet/LevelSetCellEvaluator.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace interfaces = svmp::FE::interfaces;
namespace geometry = svmp::FE::geometry;
namespace level_set = svmp::FE::level_set;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

using Point = std::array<FE::Real, 3>;

// Box [0,1]^3 split into six Kuhn tetrahedra per cube, slightly sheared so
// that the cell Jacobians differ between cells.
std::shared_ptr<svmp::Mesh> buildShearedTetraMesh(int n)
{
    auto base = std::make_shared<svmp::MeshBase>();
    const auto extent = static_cast<svmp::index_t>(n + 1);
    std::vector<svmp::real_t> x_ref;
    for (int k = 0; k <= n; ++k) {
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= n; ++i) {
                const auto x = static_cast<svmp::real_t>(i) / n;
                const auto y = static_cast<svmp::real_t>(j) / n;
                const auto z = static_cast<svmp::real_t>(k) / n;
                x_ref.push_back(x + svmp::real_t{0.07} * y * z);
                x_ref.push_back(y + svmp::real_t{0.05} * x * x);
                x_ref.push_back(z);
            }
        }
    }
    const auto vid = [&](int i, int j, int k) {
        return static_cast<svmp::index_t>((k * extent + j) * extent + i);
    };
    constexpr std::array<std::array<int, 3>, 6> paths{{
        {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
        {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}},
    }};
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> cell2vertex;
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                for (const auto& path : paths) {
                    std::array<int, 3> corner{{i, j, k}};
                    std::array<std::array<int, 3>, 4> tet{};
                    tet[0] = corner;
                    for (std::size_t step = 0; step < 3u; ++step) {
                        ++corner[static_cast<std::size_t>(path[step])];
                        tet[step + 1u] = corner;
                    }
                    std::array<std::array<int, 3>, 3> e{};
                    for (std::size_t q = 0; q < 3u; ++q) {
                        for (std::size_t c = 0; c < 3u; ++c) {
                            e[q][c] = tet[q + 1u][c] - tet[0][c];
                        }
                    }
                    const int det =
                        e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                        e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                        e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
                    if (det < 0) {
                        std::swap(tet[1], tet[2]);
                    }
                    for (const auto& vertex : tet) {
                        cell2vertex.push_back(
                            vid(vertex[0], vertex[1], vertex[2]));
                    }
                    offsets.push_back(
                        static_cast<svmp::offset_t>(cell2vertex.size()));
                }
            }
        }
    }
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Tetra;
    shape.num_corners = 4;
    shape.order = 1;
    std::vector<svmp::CellShape> shapes(offsets.size() - 1u, shape);
    base->build_from_arrays(/*spatial_dim=*/3, x_ref, offsets, cell2vertex,
                            shapes);
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

[[nodiscard]] bool sameBits(FE::Real a, FE::Real b) noexcept
{
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

template <std::size_t N>
[[nodiscard]] bool sameBits(const std::array<FE::Real, N>& a,
                            const std::array<FE::Real, N>& b) noexcept
{
    for (std::size_t i = 0; i < N; ++i) {
        if (!sameBits(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool sameBits(const geometry::CutGeometryJacobian& a,
                            const geometry::CutGeometryJacobian& b) noexcept
{
    return sameBits(a[0], b[0]) && sameBits(a[1], b[1]) &&
           sameBits(a[2], b[2]);
}

void expectSameProvenance(const geometry::CutQuadratureProvenance& a,
                          const geometry::CutQuadratureProvenance& b)
{
    EXPECT_EQ(a.embedded_geometry_id, b.embedded_geometry_id);
    EXPECT_EQ(a.cut_topology_id, b.cut_topology_id);
    EXPECT_EQ(a.parent_entity, b.parent_entity);
    EXPECT_EQ(a.parent_boundary_entity, b.parent_boundary_entity);
    EXPECT_EQ(a.parent_entity_global_id, b.parent_entity_global_id);
    EXPECT_EQ(a.parent_boundary_entity_global_id,
              b.parent_boundary_entity_global_id);
    EXPECT_EQ(a.owner_rank, b.owner_rank);
    EXPECT_EQ(a.marker, b.marker);
    EXPECT_EQ(a.cut_topology_revision, b.cut_topology_revision);
    EXPECT_EQ(a.predicate_policy_key, b.predicate_policy_key);
    EXPECT_EQ(a.coefficient_classification_policy,
              b.coefficient_classification_policy);
    EXPECT_TRUE(sameBits(a.coefficient_classification_band,
                         b.coefficient_classification_band));
    EXPECT_EQ(a.source_value_revision, b.source_value_revision);
    EXPECT_EQ(a.source_stable_id, b.source_stable_id);
    EXPECT_EQ(a.construction, b.construction);
    EXPECT_EQ(a.frame, b.frame);
    EXPECT_EQ(a.implicit_geometry_mode, b.implicit_geometry_mode);
    EXPECT_EQ(a.implicit_quadrature_backend, b.implicit_quadrature_backend);
    EXPECT_EQ(a.selected_implicit_quadrature_backend,
              b.selected_implicit_quadrature_backend);
    EXPECT_EQ(a.implicit_fallback_policy, b.implicit_fallback_policy);
    EXPECT_EQ(a.implicit_fallback_status, b.implicit_fallback_status);
    EXPECT_EQ(a.geometry_tangent_policy, b.geometry_tangent_policy);
    EXPECT_TRUE(sameBits(a.implicit_cut_root_tolerance,
                         b.implicit_cut_root_tolerance));
    EXPECT_TRUE(sameBits(a.implicit_cut_root_coordinate_tolerance,
                         b.implicit_cut_root_coordinate_tolerance));
    EXPECT_EQ(a.implicit_cut_root_max_iterations,
              b.implicit_cut_root_max_iterations);
    EXPECT_EQ(a.requested_quadrature_order, b.requested_quadrature_order);
    EXPECT_EQ(a.achieved_quadrature_order, b.achieved_quadrature_order);
    EXPECT_EQ(a.free_surface_snapshot_revision_key,
              b.free_surface_snapshot_revision_key);
}

// Every field except the point lists, the moments and the
// classification-only marker.
void expectSameClassification(const interfaces::FreeSurfaceGeometryRuleRecord& a,
                              const interfaces::FreeSurfaceGeometryRuleRecord& b)
{
    EXPECT_EQ(a.construction_observation, b.construction_observation);
    EXPECT_EQ(a.role, b.role);
    EXPECT_EQ(a.retention, b.retention);
    EXPECT_EQ(a.physical_boundary_marker, b.physical_boundary_marker);
    EXPECT_EQ(a.locally_owned, b.locally_owned);
    EXPECT_EQ(a.source_fragment_stable_ids, b.source_fragment_stable_ids);
    EXPECT_EQ(a.topology_id, b.topology_id);
    EXPECT_EQ(a.source_topology_key, b.source_topology_key);
    EXPECT_EQ(a.component_id, b.component_id);
    EXPECT_EQ(a.moment_certificate.polynomial_order,
              b.moment_certificate.polynomial_order);
    EXPECT_EQ(a.moment_certificate.ambient_dimension,
              b.moment_certificate.ambient_dimension);
    EXPECT_EQ(a.moment_certificate.source, b.moment_certificate.source);
    EXPECT_EQ(a.moment_certificate.phase_sign_certified,
              b.moment_certificate.phase_sign_certified);

    const auto& ar = a.reference_rule;
    const auto& br = b.reference_rule;
    EXPECT_EQ(ar.kind, br.kind);
    EXPECT_EQ(ar.side, br.side);
    EXPECT_EQ(ar.geometric_dimension, br.geometric_dimension);
    EXPECT_TRUE(sameBits(ar.measure, br.measure));
    EXPECT_TRUE(sameBits(ar.parent_measure, br.parent_measure));
    EXPECT_TRUE(sameBits(ar.volume_fraction, br.volume_fraction));
    EXPECT_EQ(ar.exact_for_constants, br.exact_for_constants);
    EXPECT_EQ(ar.exact_polynomial_order, br.exact_polynomial_order);
    EXPECT_EQ(ar.policy.kind, br.policy.kind);
    EXPECT_EQ(ar.policy.polynomial_order, br.policy.polynomial_order);
    EXPECT_EQ(ar.policy.moment_fitted, br.policy.moment_fitted);
    EXPECT_TRUE(sameBits(ar.policy.tolerance, br.policy.tolerance));
    EXPECT_EQ(ar.policy.name, br.policy.name);
    expectSameProvenance(ar.provenance, br.provenance);
    EXPECT_EQ(ar.provenance_id, br.provenance_id);
    EXPECT_EQ(ar.frame, br.frame);
    EXPECT_EQ(ar.curved_geometry, br.curved_geometry);
    EXPECT_EQ(ar.full_cell_equivalent, br.full_cell_equivalent);

    const auto& ap = a.physical_rule;
    const auto& bp = b.physical_rule;
    EXPECT_EQ(ap.kind, bp.kind);
    EXPECT_EQ(ap.side, bp.side);
    EXPECT_EQ(ap.geometric_dimension, bp.geometric_dimension);
    EXPECT_EQ(ap.parent_entity, bp.parent_entity);
    EXPECT_EQ(ap.marker, bp.marker);
    EXPECT_EQ(ap.source_stable_id, bp.source_stable_id);
    EXPECT_EQ(ap.cut_topology_revision, bp.cut_topology_revision);
    EXPECT_EQ(ap.source_value_revision, bp.source_value_revision);
    EXPECT_EQ(ap.free_surface_snapshot_revision_key,
              bp.free_surface_snapshot_revision_key);
    EXPECT_TRUE(sameBits(ap.reference_measure, bp.reference_measure));
    EXPECT_TRUE(sameBits(ap.physical_measure, bp.physical_measure));
}

// The content a classification-only record releases: points and moments.
void expectSamePoints(const interfaces::FreeSurfaceGeometryRuleRecord& a,
                      const interfaces::FreeSurfaceGeometryRuleRecord& b)
{
    ASSERT_EQ(a.moment_certificate.moments.size(),
              b.moment_certificate.moments.size());
    for (std::size_t i = 0; i < a.moment_certificate.moments.size(); ++i) {
        EXPECT_EQ(a.moment_certificate.moments[i].exponents,
                  b.moment_certificate.moments[i].exponents);
        EXPECT_TRUE(sameBits(a.moment_certificate.moments[i].value,
                             b.moment_certificate.moments[i].value));
    }
    ASSERT_EQ(a.reference_rule.points.size(), b.reference_rule.points.size());
    for (std::size_t q = 0; q < a.reference_rule.points.size(); ++q) {
        const auto& x = a.reference_rule.points[q];
        const auto& y = b.reference_rule.points[q];
        EXPECT_TRUE(sameBits(x.point, y.point));
        EXPECT_TRUE(sameBits(x.normal, y.normal));
        EXPECT_TRUE(sameBits(x.boundary_normal, y.boundary_normal));
        EXPECT_TRUE(sameBits(x.tangent, y.tangent));
        EXPECT_TRUE(sameBits(x.weight, y.weight));
        EXPECT_TRUE(sameBits(x.parent_coordinate, y.parent_coordinate));
        EXPECT_TRUE(sameBits(x.reference_measure_factor,
                             y.reference_measure_factor));
        EXPECT_TRUE(sameBits(x.level_set_residual, y.level_set_residual));
        EXPECT_TRUE(sameBits(x.gradient_norm, y.gradient_norm));
    }
    ASSERT_EQ(a.physical_rule.points.size(), b.physical_rule.points.size());
    for (std::size_t q = 0; q < a.physical_rule.points.size(); ++q) {
        const auto& x = a.physical_rule.points[q];
        const auto& y = b.physical_rule.points[q];
        EXPECT_TRUE(sameBits(x.reference_point, y.reference_point));
        EXPECT_TRUE(sameBits(x.physical_point, y.physical_point));
        EXPECT_TRUE(sameBits(x.jacobian, y.jacobian));
        EXPECT_TRUE(sameBits(x.inverse_jacobian, y.inverse_jacobian));
        EXPECT_TRUE(sameBits(x.absolute_jacobian_determinant,
                             y.absolute_jacobian_determinant));
        EXPECT_TRUE(sameBits(x.reference_weight, y.reference_weight));
        EXPECT_TRUE(sameBits(x.physical_weight, y.physical_weight));
        EXPECT_TRUE(sameBits(x.normal, y.normal));
        EXPECT_TRUE(sameBits(x.boundary_normal, y.boundary_normal));
        EXPECT_TRUE(sameBits(x.tangent, y.tangent));
    }
}

// Every scalar field except released_point_count.
void expectSameRuleClassification(const geometry::CutQuadratureRule& a,
                                  const geometry::CutQuadratureRule& b)
{
    EXPECT_EQ(a.kind, b.kind);
    EXPECT_EQ(a.side, b.side);
    EXPECT_EQ(a.geometric_dimension, b.geometric_dimension);
    EXPECT_TRUE(sameBits(a.measure, b.measure));
    EXPECT_TRUE(sameBits(a.parent_measure, b.parent_measure));
    EXPECT_TRUE(sameBits(a.volume_fraction, b.volume_fraction));
    EXPECT_EQ(a.exact_for_constants, b.exact_for_constants);
    EXPECT_EQ(a.exact_polynomial_order, b.exact_polynomial_order);
    EXPECT_EQ(a.policy.kind, b.policy.kind);
    EXPECT_EQ(a.policy.polynomial_order, b.policy.polynomial_order);
    EXPECT_EQ(a.policy.moment_fitted, b.policy.moment_fitted);
    EXPECT_TRUE(sameBits(a.policy.tolerance, b.policy.tolerance));
    EXPECT_EQ(a.policy.name, b.policy.name);
    expectSameProvenance(a.provenance, b.provenance);
    EXPECT_EQ(a.provenance_id, b.provenance_id);
    EXPECT_EQ(a.frame, b.frame);
    EXPECT_EQ(a.curved_geometry, b.curved_geometry);
    EXPECT_EQ(a.full_cell_equivalent, b.full_cell_equivalent);
    EXPECT_EQ(geometry::cutQuadratureRulePointCount(a),
              geometry::cutQuadratureRulePointCount(b));
}

// Every field, points bitwise.
void expectSameRules(const geometry::CutQuadratureRule& a,
                     const geometry::CutQuadratureRule& b)
{
    expectSameRuleClassification(a, b);
    EXPECT_EQ(a.released_point_count, b.released_point_count);
    ASSERT_EQ(a.points.size(), b.points.size());
    for (std::size_t q = 0; q < a.points.size(); ++q) {
        const auto& x = a.points[q];
        const auto& y = b.points[q];
        EXPECT_TRUE(sameBits(x.point, y.point));
        EXPECT_TRUE(sameBits(x.normal, y.normal));
        EXPECT_TRUE(sameBits(x.boundary_normal, y.boundary_normal));
        EXPECT_TRUE(sameBits(x.tangent, y.tangent));
        EXPECT_TRUE(sameBits(x.weight, y.weight));
        EXPECT_TRUE(sameBits(x.parent_coordinate, y.parent_coordinate));
        EXPECT_TRUE(sameBits(x.reference_measure_factor,
                             y.reference_measure_factor));
        EXPECT_TRUE(sameBits(x.level_set_residual, y.level_set_residual));
        EXPECT_TRUE(sameBits(x.gradient_norm, y.gradient_norm));
    }
}

// One off-centre sphere on a sheared box: dry (positive) full cells,
// wet (negative) full cells and cut cells.
struct SphereSnapshotFixture {
    std::shared_ptr<svmp::Mesh> mesh;
    FE::systems::FESystem system;
    FE::FieldId phi{FE::INVALID_FIELD_ID};
    std::vector<FE::Real> solution{};
    level_set::LevelSetGeneratedInterfaceResult generated{};
    std::shared_ptr<level_set::LevelSetCellEvaluator> evaluator{};

    SphereSnapshotFixture()
        : mesh(buildShearedTetraMesh(6)), system(mesh)
    {
        phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = std::make_shared<FE::spaces::H1Space>(
                FE::ElementType::Tetra4, /*order=*/1),
            .components = 1,
        });
        system.setup();
        const auto& dofs = system.fieldDofHandler(phi);
        const auto* map = dofs.getEntityDofMap();
        const auto offset =
            static_cast<std::size_t>(system.fieldDofOffset(phi));
        solution.assign(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()),
            FE::Real{0.0});
        for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
            const auto x = system.meshAccess().getNodeCoordinates(v);
            const FE::Real dx = x[0] - FE::Real{0.531};
            const FE::Real dy = x[1] - FE::Real{0.487};
            const FE::Real dz = x[2] - FE::Real{0.462};
            solution[offset + static_cast<std::size_t>(
                                  map->getVertexDofs(v).front())] =
                std::sqrt(dx * dx + dy * dy + dz * dz) - FE::Real{0.29};
        }
        level_set::LevelSetGeneratedInterfaceOptions options{};
        options.level_set_field_name = "phi";
        options.domain_id = "compact_records_sphere";
        options.requested_interface_marker = 4711;
        options.quadrature_order = 1;
        options.interface_quadrature_order = 2;
        options.volume_quadrature_order = 2;
        level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
        generated = lifecycle.build(system, options, solution);
        evaluator = std::make_shared<level_set::LevelSetCellEvaluator>(
            level_set::makeLevelSetCellEvaluator(system, phi, solution));
    }

    [[nodiscard]] interfaces::FreeSurfaceGeometryScalarEvaluator scalar() const
    {
        interfaces::FreeSurfaceGeometryScalarEvaluator result;
        result.value = [evaluator = evaluator](
                           FE::GlobalIndex cell,
                           const Point& xi,
                           const geometry::CutQuadratureProvenance&) {
            return evaluator->evaluateLinearCorner(cell, xi).value;
        };
        result.reference_gradient =
            [evaluator = evaluator](FE::GlobalIndex cell,
                                    const Point& xi,
                                    const geometry::CutQuadratureProvenance&) {
                return evaluator->evaluateLinearCorner(cell, xi)
                    .reference_gradient;
            };
        return result;
    }

    [[nodiscard]] std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot>
    snapshot(std::optional<geometry::CutIntegrationSide> compact_side) const
    {
        interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
        policy.require_complete_exterior_boundary_partition = false;
        policy.classification_only_full_cell_side = compact_side;
        return interfaces::buildFreeSurfaceGeometrySnapshot(
            generated.domain,
            {},
            {},
            system.meshAccess(),
            policy,
            scalar(),
            "compact_records_sphere");
    }
};

// Smooth reference-coordinate velocity that differs between cells.
[[nodiscard]] interfaces::FreeSurfaceDiscreteFunctionalVectorEvaluator
testVelocity(FE::Real scale)
{
    interfaces::FreeSurfaceDiscreteFunctionalVectorEvaluator velocity;
    velocity.value = [scale](FE::GlobalIndex cell,
                             const Point& xi,
                             const geometry::CutQuadratureProvenance&) {
        const FE::Real c = FE::Real{1.0e-3} * static_cast<FE::Real>(cell);
        return Point{{scale * (xi[0] + c), scale * xi[1] * xi[2],
                      scale * (FE::Real{0.5} - xi[0] * xi[1])}};
    };
    velocity.physical_gradient =
        [scale](FE::GlobalIndex cell,
                const Point& xi,
                const geometry::CutQuadratureProvenance&) {
            const FE::Real c = FE::Real{1.0e-3} * static_cast<FE::Real>(cell);
            interfaces::FreeSurfaceDiscreteFunctionalPhysicalGradient g{};
            g[0] = {{scale, c, FE::Real{0.0}}};
            g[1] = {{FE::Real{0.0}, scale * xi[2], scale * xi[1]}};
            g[2] = {{-scale * xi[1], -scale * xi[0], c}};
            return g;
        };
    return velocity;
}

TEST(FreeSurfaceSnapshotCompactRecords,
     DryFullCellRecordsAreClassificationOnlyAndRematerializeExactly)
{
    const SphereSnapshotFixture fixture;
    ASSERT_TRUE(fixture.generated.success) << fixture.generated.diagnostic;
    const auto full = fixture.snapshot(std::nullopt);
    const auto compact =
        fixture.snapshot(geometry::CutIntegrationSide::Positive);
    ASSERT_NE(full, nullptr);
    ASSERT_NE(compact, nullptr);

    // Same content-addressed revision and the same validation ledger.
    EXPECT_EQ(full->revision().snapshot_revision_key,
              compact->revision().snapshot_revision_key);
    static_assert(sizeof(interfaces::FreeSurfaceGeometryValidationLedger) %
                      sizeof(std::uint64_t) ==
                  0u);
    EXPECT_EQ(std::memcmp(&full->ledger(),
                          &compact->ledger(),
                          sizeof(interfaces::FreeSurfaceGeometryValidationLedger)),
              0);

    ASSERT_EQ(full->rules().size(), compact->rules().size());
    std::size_t compact_count = 0u;
    std::size_t dry_full_count = 0u;
    std::size_t wet_full_count = 0u;
    std::size_t cut_volume_count = 0u;
    for (std::size_t i = 0; i < full->rules().size(); ++i) {
        const auto& a = full->rules()[i];
        const auto& b = compact->rules()[i];
        EXPECT_FALSE(a.classification_only);
        expectSameClassification(a, b);
        const bool dry_full =
            a.role == interfaces::FreeSurfaceGeometryRuleRole::PositiveVolume &&
            a.reference_rule.full_cell_equivalent;
        dry_full_count += dry_full ? 1u : 0u;
        wet_full_count +=
            a.role == interfaces::FreeSurfaceGeometryRuleRole::NegativeVolume &&
                    a.reference_rule.full_cell_equivalent
                ? 1u
                : 0u;
        cut_volume_count += (a.role == interfaces::FreeSurfaceGeometryRuleRole::
                                           NegativeVolume ||
                             a.role == interfaces::FreeSurfaceGeometryRuleRole::
                                           PositiveVolume) &&
                                    !a.reference_rule.full_cell_equivalent
                                ? 1u
                                : 0u;
        EXPECT_EQ(b.classification_only, dry_full);
        EXPECT_EQ(interfaces::freeSurfaceGeometryRuleContentDigest(a),
                  interfaces::freeSurfaceGeometryRuleContentDigest(b));
        if (!b.classification_only) {
            expectSamePoints(a, b);
            continue;
        }
        ++compact_count;
        EXPECT_TRUE(b.reference_rule.points.empty());
        EXPECT_TRUE(b.physical_rule.points.empty());
        EXPECT_EQ(b.reference_rule.points.capacity(), 0u);
        EXPECT_EQ(b.physical_rule.points.capacity(), 0u);
        EXPECT_EQ(b.moment_certificate.moments.capacity(), 0u);
        EXPECT_FALSE(a.moment_certificate.moments.empty());
        // Exact recomputation of the released points.
        const auto materialized =
            interfaces::materializeFreeSurfaceGeometryRuleRecord(
                b, fixture.system.meshAccess());
        EXPECT_FALSE(materialized.classification_only);
        expectSameClassification(a, materialized);
        expectSamePoints(a, materialized);
        EXPECT_EQ(interfaces::freeSurfaceGeometryRuleContentDigest(
                      materialized),
                  interfaces::freeSurfaceGeometryRuleContentDigest(a));
    }
    // The record array is reserved exactly.
    EXPECT_EQ(compact->rules().capacity(), compact->rules().size());
    EXPECT_GT(dry_full_count, 0u);
    EXPECT_GT(wet_full_count, 0u);
    EXPECT_GT(cut_volume_count, 0u);
    EXPECT_EQ(compact_count, dry_full_count);
    EXPECT_LT(compact->residentBytes(), full->residentBytes());

    // Materialized records are returned unchanged.
    for (const auto& record : full->rules()) {
        const auto same = interfaces::materializeFreeSurfaceGeometryRuleRecord(
            record, fixture.system.meshAccess());
        expectSameClassification(record, same);
        expectSamePoints(record, same);
    }
}

TEST(FreeSurfaceSnapshotCompactRecords, ConsumersGetIdenticalAnswers)
{
    const SphereSnapshotFixture fixture;
    ASSERT_TRUE(fixture.generated.success) << fixture.generated.diagnostic;
    const auto full = fixture.snapshot(std::nullopt);
    const auto compact =
        fixture.snapshot(geometry::CutIntegrationSide::Positive);
    ASSERT_NE(full, nullptr);
    ASSERT_NE(compact, nullptr);

    // Measure-only functional over every role, both liquid sides.
    for (const auto liquid : {geometry::CutIntegrationSide::Negative,
                              geometry::CutIntegrationSide::Positive}) {
        interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
        parameters.liquid_side = liquid;
        parameters.surface_tension = FE::Real{0.7};
        parameters.volume_multiplier = FE::Real{-1.3};
        const auto a =
            interfaces::evaluateFreeSurfaceDiscreteFunctional(*full, parameters);
        const auto b = interfaces::evaluateFreeSurfaceDiscreteFunctional(
            *compact, parameters);
        EXPECT_TRUE(sameBits(a.owned_liquid_volume, b.owned_liquid_volume));
        EXPECT_TRUE(sameBits(a.owned_liquid_gas_area, b.owned_liquid_gas_area));
        EXPECT_TRUE(sameBits(a.total_potential, b.total_potential));
    }

    // Wet-side point integrals are unchanged.
    const auto velocity = testVelocity(FE::Real{0.4});
    interfaces::FreeSurfaceActiveVolumeEnergyParameters energy;
    energy.liquid_side = geometry::CutIntegrationSide::Negative;
    energy.density = FE::Real{1.2};
    energy.gravitational_acceleration = {{0.0, 0.0, -9.81}};
    const auto energy_full =
        interfaces::evaluateFreeSurfaceActiveVolumeEnergy(*full, energy,
                                                          velocity);
    const auto energy_compact =
        interfaces::evaluateFreeSurfaceActiveVolumeEnergy(*compact, energy,
                                                          velocity);
    EXPECT_EQ(energy_full.owned_quadrature_point_count,
              energy_compact.owned_quadrature_point_count);
    EXPECT_TRUE(sameBits(energy_full.owned_liquid_volume,
                         energy_compact.owned_liquid_volume));
    EXPECT_TRUE(sameBits(energy_full.kinetic_energy,
                         energy_compact.kinetic_energy));
    EXPECT_TRUE(sameBits(energy_full.gravitational_energy,
                         energy_compact.gravitational_energy));
    EXPECT_TRUE(sameBits(energy_full.gravitational_potential_power,
                         energy_compact.gravitational_potential_power));

    interfaces::FreeSurfaceActiveVolumeDissipationParameters dissipation;
    dissipation.liquid_side = geometry::CutIntegrationSide::Negative;
    dissipation.dynamic_viscosity = FE::Real{0.3};
    const auto dissipation_full =
        interfaces::evaluateFreeSurfaceActiveVolumeDissipation(
            *full, dissipation, velocity);
    const auto dissipation_compact =
        interfaces::evaluateFreeSurfaceActiveVolumeDissipation(
            *compact, dissipation, velocity);
    EXPECT_TRUE(sameBits(dissipation_full.bulk_viscous_dissipation_rate,
                         dissipation_compact.bulk_viscous_dissipation_rate));
    EXPECT_TRUE(sameBits(dissipation_full.owned_liquid_volume,
                         dissipation_compact.owned_liquid_volume));

    const auto previous = testVelocity(FE::Real{0.25});
    const auto work_full = interfaces::evaluateFreeSurfaceBackwardEulerKineticWork(
        *full, geometry::CutIntegrationSide::Negative, FE::Real{1.2}, 3u, 4u,
        previous, velocity);
    const auto work_compact =
        interfaces::evaluateFreeSurfaceBackwardEulerKineticWork(
            *compact, geometry::CutIntegrationSide::Negative, FE::Real{1.2},
            3u, 4u, previous, velocity);
    EXPECT_TRUE(sameBits(work_full.step_integrated_inertia_work,
                         work_compact.step_integrated_inertia_work));
    EXPECT_TRUE(sameBits(work_full.identity_residual,
                         work_compact.identity_residual));

    // A dry-side point integral fails closed instead of skipping volume.
    energy.liquid_side = geometry::CutIntegrationSide::Positive;
    EXPECT_NO_THROW((void)interfaces::evaluateFreeSurfaceActiveVolumeEnergy(
        *full, energy, velocity));
    EXPECT_THROW((void)interfaces::evaluateFreeSurfaceActiveVolumeEnergy(
                     *compact, energy, velocity),
                 std::invalid_argument);
    dissipation.liquid_side = geometry::CutIntegrationSide::Positive;
    EXPECT_THROW((void)interfaces::evaluateFreeSurfaceActiveVolumeDissipation(
                     *compact, dissipation, velocity),
                 std::invalid_argument);
    EXPECT_THROW(
        (void)interfaces::evaluateFreeSurfaceBackwardEulerKineticWork(
            *compact, geometry::CutIntegrationSide::Positive, FE::Real{1.2},
            3u, 4u, previous, velocity),
        std::invalid_argument);

    // The cut integration context imports identical rules from both.
    FE::assembly::CutIntegrationContext context_full;
    FE::assembly::CutIntegrationContext context_compact;
    context_full.addFreeSurfaceGeometrySnapshot(full);
    context_compact.addFreeSurfaceGeometrySnapshot(compact);
    ASSERT_EQ(context_full.volumeRules().size(),
              context_compact.volumeRules().size());
    for (std::size_t i = 0; i < context_full.volumeRules().size(); ++i) {
        expectSameRules(context_full.volumeRules()[i],
                        context_compact.volumeRules()[i]);
    }
    ASSERT_EQ(context_full.interfaceRules().size(),
              context_compact.interfaceRules().size());
    for (std::size_t i = 0; i < context_full.interfaceRules().size(); ++i) {
        expectSameRules(context_full.interfaceRules()[i],
                        context_compact.interfaceRules()[i]);
    }
}

TEST(FreeSurfaceSnapshotCompactRecords, PolicyValidation)
{
    const SphereSnapshotFixture fixture;
    ASSERT_TRUE(fixture.generated.success) << fixture.generated.diagnostic;
    EXPECT_THROW(
        (void)fixture.snapshot(geometry::CutIntegrationSide::Interface),
        std::invalid_argument);

    // Compacting the wet side instead keeps the dry side materialized.
    const auto full = fixture.snapshot(std::nullopt);
    const auto wet_compact =
        fixture.snapshot(geometry::CutIntegrationSide::Negative);
    ASSERT_EQ(full->rules().size(), wet_compact->rules().size());
    EXPECT_EQ(full->revision().snapshot_revision_key,
              wet_compact->revision().snapshot_revision_key);
    for (std::size_t i = 0; i < full->rules().size(); ++i) {
        const auto& record = wet_compact->rules()[i];
        EXPECT_EQ(record.classification_only,
                  record.role == interfaces::FreeSurfaceGeometryRuleRole::
                                     NegativeVolume &&
                      record.reference_rule.full_cell_equivalent);
    }
}

TEST(FreeSurfaceSnapshotCompactRecords, CopyOnWriteVectorSharesUntilModified)
{
    using Points = interfaces::CopyOnWriteVector<geometry::CutQuadraturePoint>;
    std::vector<geometry::CutQuadraturePoint> source(3u);
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i].point = {{0.1 * static_cast<FE::Real>(i), 0.2, 0.3}};
        source[i].weight = FE::Real{0.25} + static_cast<FE::Real>(i);
    }
    Points a = source;
    Points b = a;
    EXPECT_EQ(a.shareCount(), 2);
    EXPECT_EQ(a.data(), b.data());
    ASSERT_EQ(b.size(), source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
        EXPECT_TRUE(sameBits(b[i].point, source[i].point));
        EXPECT_TRUE(sameBits(b[i].weight, source[i].weight));
    }
    b.mutate()[1].weight = FE::Real{7.0};
    EXPECT_NE(a.data(), b.data());
    EXPECT_EQ(a.shareCount(), 1);
    EXPECT_TRUE(sameBits(a[1].weight, source[1].weight));
    EXPECT_TRUE(sameBits(b[1].weight, FE::Real{7.0}));
    const std::vector<geometry::CutQuadraturePoint>& view = a;
    EXPECT_EQ(view.size(), source.size());
    Points empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.begin(), empty.end());
    empty.push_back(source[0]);
    EXPECT_EQ(empty.size(), 1u);
    empty.clear();
    EXPECT_EQ(empty.shareCount(), 0);
}

TEST(FreeSurfaceSnapshotCompactRecords,
     GeneratedDomainCopiesShareRegionPointsAndYieldIdenticalRules)
{
    const SphereSnapshotFixture fixture;
    ASSERT_TRUE(fixture.generated.success) << fixture.generated.diagnostic;
    const auto& domain = fixture.generated.domain;
    const auto snapshot =
        fixture.snapshot(geometry::CutIntegrationSide::Positive);
    const auto& snapshot_regions = snapshot->interfaceDomain().volumeRegions();
    ASSERT_EQ(snapshot_regions.size(), domain.volumeRegions().size());

    // A region whose points own independent storage.
    const auto deep_copy = [](interfaces::CutInterfaceVolumeRegion region) {
        region.quadrature_points = std::vector<geometry::CutQuadraturePoint>(
            region.quadrature_points.values());
        return region;
    };
    std::size_t shared = 0u;
    for (std::size_t i = 0; i < snapshot_regions.size(); ++i) {
        const auto& region = domain.volumeRegions()[i];
        const auto& copy = snapshot_regions[i];
        if (!region.quadrature_points.empty()) {
            EXPECT_EQ(region.quadrature_points.data(),
                      copy.quadrature_points.data());
            shared += region.quadrature_points.shareCount() > 1 ? 1u : 0u;
        }
        if (!region.active()) {
            continue;
        }
        const auto independent = deep_copy(region);
        EXPECT_NE(independent.quadrature_points.data(),
                  region.quadrature_points.data());
        expectSameRules(region.toCutQuadratureRule(domain.request()),
                        independent.toCutQuadratureRule(domain.request()));
    }
    // Either the domain copies share one region array, or their regions
    // share point arrays.
    EXPECT_TRUE(shared > 0u || domain.volumeRegionShareCount() > 1);
}

TEST(FreeSurfaceSnapshotCompactRecords, DomainCopiesShareTheirArraysUntilModified)
{
    const SphereSnapshotFixture fixture;
    ASSERT_TRUE(fixture.generated.success) << fixture.generated.diagnostic;
    const auto& domain = fixture.generated.domain;
    ASSERT_FALSE(domain.volumeRegions().empty());
    ASSERT_FALSE(domain.fragments().empty());

    auto copy = domain;
    EXPECT_EQ(copy.volumeRegions().data(), domain.volumeRegions().data());
    EXPECT_EQ(copy.fragments().data(), domain.fragments().data());
    EXPECT_GE(domain.volumeRegionShareCount(), 2);

    // Adding to the copy gives it its own arrays; the original is unchanged.
    const auto original_regions = domain.volumeRegions().size();
    auto region = domain.volumeRegions().front();
    region.stable_id = 0u;
    region.local_region_index = FE::INVALID_LOCAL_INDEX;
    copy.addVolumeRegion(region);
    EXPECT_NE(copy.volumeRegions().data(), domain.volumeRegions().data());
    EXPECT_EQ(domain.volumeRegions().size(), original_regions);
    EXPECT_EQ(copy.volumeRegions().size(), original_regions + 1u);
    for (std::size_t i = 0; i < original_regions; ++i) {
        EXPECT_EQ(copy.volumeRegions()[i].stable_id,
                  domain.volumeRegions()[i].stable_id);
        EXPECT_EQ(copy.volumeRegions()[i].quadrature_points.data(),
                  domain.volumeRegions()[i].quadrature_points.data());
    }
    copy.clearFragments();
    EXPECT_TRUE(copy.fragments().empty());
    EXPECT_FALSE(domain.fragments().empty());

    // A snapshot built from the domain shares its arrays as well.
    const auto snapshot =
        fixture.snapshot(geometry::CutIntegrationSide::Positive);
    EXPECT_EQ(snapshot->interfaceDomain().volumeRegions().data(),
              domain.volumeRegions().data());
}

TEST(FreeSurfaceSnapshotCompactRecords,
     ContextKeepsDryFullCellRulesClassificationOnly)
{
    const SphereSnapshotFixture fixture;
    ASSERT_TRUE(fixture.generated.success) << fixture.generated.diagnostic;
    const auto snapshot =
        fixture.snapshot(geometry::CutIntegrationSide::Positive);
    const auto& mesh = fixture.system.meshAccess();

    FE::assembly::CutIntegrationContext full;
    full.addFreeSurfaceGeometrySnapshot(snapshot);
    auto compact = std::make_unique<FE::assembly::CutIntegrationContext>();
    compact->addFreeSurfaceGeometrySnapshot(
        snapshot, std::nullopt, geometry::CutIntegrationSide::Positive);
    // No extra modification events: the import is the same content change.
    EXPECT_EQ(full.contentRevision(), compact->contentRevision());
    EXPECT_EQ(full.classificationOnlyVolumeRuleCount(), 0u);

    ASSERT_EQ(full.volumeRules().size(), compact->volumeRules().size());
    ASSERT_EQ(full.metadata().size(), compact->metadata().size());
    ASSERT_EQ(full.bindings().size(), compact->bindings().size());
    std::size_t released = 0u;
    std::optional<std::size_t> first_released;
    for (std::size_t i = 0; i < full.volumeRules().size(); ++i) {
        const auto& a = full.volumeRules()[i];
        const auto& b = compact->volumeRules()[i];
        expectSameRuleClassification(a, b);
        EXPECT_EQ(full.metadata()[i].parent_entity,
                  compact->metadata()[i].parent_entity);
        EXPECT_TRUE(sameBits(full.metadata()[i].volume_fraction,
                             compact->metadata()[i].volume_fraction));
        EXPECT_TRUE(sameBits(full.metadata()[i].embedded_normal,
                             compact->metadata()[i].embedded_normal));
        EXPECT_EQ(full.bindings()[i].cut_revision_key,
                  compact->bindings()[i].cut_revision_key);
        const bool dry_full =
            a.side == geometry::CutIntegrationSide::Positive &&
            a.full_cell_equivalent;
        EXPECT_EQ(compact->volumeRuleIsClassificationOnly(i), dry_full);
        if (!dry_full) {
            expectSameRules(a, b);
            continue;
        }
        ++released;
        if (!first_released) {
            first_released = i;
        }
        EXPECT_TRUE(b.points.empty());
        EXPECT_EQ(b.points.capacity(), 0u);
        EXPECT_EQ(b.released_point_count, a.points.size());
        const auto materialized = compact->materializedVolumeRule(i);
        expectSameRules(a, materialized);
        expectSameRules(a, compact->materializedVolumeRule(b));
        EXPECT_TRUE(sameBits(
            geometry::physicalCutQuadratureMeasure(mesh, a),
            geometry::physicalCutQuadratureMeasure(mesh, materialized)));
    }
    ASSERT_TRUE(first_released.has_value());
    EXPECT_EQ(compact->classificationOnlyVolumeRuleCount(), released);
    // The per-rule arrays are reserved for the imported rules only.
    EXPECT_LE(compact->volumeRules().capacity() - compact->volumeRules().size(),
              compact->generatedPrunedVolumeRuleCount());
    EXPECT_LE(compact->metadata().capacity() - compact->metadata().size(),
              compact->generatedPrunedVolumeRuleCount());
    for (const auto side : {geometry::CutIntegrationSide::Negative,
                            geometry::CutIntegrationSide::Positive}) {
        const auto a = full.generatedVolumeDiagnosticsForMarkerAndSide(
            snapshot->interfaceDomain().marker(), side);
        const auto b = compact->generatedVolumeDiagnosticsForMarkerAndSide(
            snapshot->interfaceDomain().marker(), side);
        EXPECT_EQ(a.rule_count, b.rule_count);
        EXPECT_EQ(a.quadrature_points, b.quadrature_points);
        EXPECT_TRUE(sameBits(a.active_volume, b.active_volume));
    }

    // Integrating consumers fail closed; the full context still integrates.
    const auto one = [](const FE::assembly::CutScalarOperatorPoint&) {
        return FE::Real{1.0};
    };
    EXPECT_NO_THROW((void)full.evaluateScalarCutOperator(
        FE::assembly::CutIntegrationAssemblyPath::Standard, one, one));
    EXPECT_THROW((void)compact->evaluateScalarCutOperator(
                     FE::assembly::CutIntegrationAssemblyPath::Standard,
                     one,
                     one),
                 std::logic_error);

    // A copy keeps the snapshot, and with it the source regions, alive.
    const FE::assembly::CutIntegrationContext copy = *compact;
    compact.reset();
    for (std::size_t i = 0; i < full.volumeRules().size(); ++i) {
        if (copy.volumeRuleIsClassificationOnly(i)) {
            expectSameRules(full.volumeRules()[i],
                            copy.materializedVolumeRule(i));
        }
    }
    // A released rule that is not stored in the context has no source.
    const auto foreign = copy.volumeRules()[*first_released];
    EXPECT_THROW((void)copy.materializedVolumeRule(foreign), std::logic_error);
}

#endif

} // namespace
