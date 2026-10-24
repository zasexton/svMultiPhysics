/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TESTS_UNIT_SYSTEMS_FUNCTIONAL_FORM_EVALUATOR_FIXTURES_H
#define SVMP_FE_TESTS_UNIT_SYSTEMS_FUNCTIONAL_FORM_EVALUATOR_FIXTURES_H

/**
 * @file FunctionalFormEvaluatorFixtures.h
 * @brief A cut quadrilateral strip with an authoritative free-surface snapshot
 *
 * The strip [0, extent] x [0, 1] of Quad4 cells along x, with the wall y = 0
 * (or a faceted curved wall through y = wall_profile(x) at the wall nodes),
 * a level set phi(x, y) cut cell by cell (LinearCorner), its contact points
 * with the wall and its wetted wall (the wall where phi < 0), imported into a
 * cut integration context as one geometry snapshot.  Every rank sees every
 * cell; cell c is owned by rank c % size.  The functional assembler reads one
 * Q1 field with the given nodal values.
 */

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Assembly/FunctionalAssembler.h"
#include "Dofs/DofMap.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "Spaces/H1Space.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {
namespace functional_form_test {

inline constexpr int kWallMarker = 41;
inline constexpr int kInterfaceMarker = 3041;
inline constexpr int kContactMarker = 3042;
inline constexpr int kWettedWallMarker = 3043;
inline constexpr FieldId kField = 0;
inline constexpr std::uint64_t kSourceRevision = 5u;

/// A scalar of the physical coordinates.
using PlaneFunction = std::function<Real(Real, Real)>;

class QuadStripMeshAccess final : public assembly::IMeshAccess {
public:
    /// @p wall_profile, when given, puts the wall nodes at y = wall_profile(x)
    /// (a curved, faceted wall) instead of y = 0.
    QuadStripMeshAccess(int rank,
                        int size,
                        GlobalIndex cell_count,
                        Real extent,
                        std::function<Real(Real)> wall_profile = {})
        : rank_(rank)
        , size_(size)
        , cell_count_(cell_count)
        , extent_(extent)
        , wall_profile_(std::move(wall_profile))
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return cell_count_; }

    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        GlobalIndex count = 0;
        for (GlobalIndex cell = 0; cell < cell_count_; ++cell) {
            if (isOwnedCell(cell)) {
                ++count;
            }
        }
        return count;
    }

    [[nodiscard]] GlobalIndex numVertices() const override { return (cell_count_ + 1) * 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return cell_count_; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return cell_count_ - 1; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 1001u; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 2001u; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 3001u; }
    [[nodiscard]] std::uint64_t numberingRevision() const override { return 4001u; }

    [[nodiscard]] bool globalEntityIdsAvailable() const override { return true; }
    [[nodiscard]] GlobalIndex getCellGlobalId(GlobalIndex cell) const override { return 2000 + cell; }
    [[nodiscard]] GlobalIndex getBoundaryFaceGlobalId(GlobalIndex face) const override
    {
        return 20000 + face;
    }

    [[nodiscard]] int parallelRank() const override { return rank_; }
    [[nodiscard]] int parallelSize() const override { return size_; }

    [[nodiscard]] int getCellOwnerRank(GlobalIndex cell) const override
    {
        return static_cast<int>(cell % static_cast<GlobalIndex>(size_));
    }

    [[nodiscard]] int getBoundaryFaceOwnerRank(GlobalIndex, GlobalIndex parent_cell) const override
    {
        return getCellOwnerRank(parent_cell);
    }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell) const override
    {
        return getCellOwnerRank(cell) == rank_;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Quad4; }

    void getCellNodes(GlobalIndex cell, std::vector<GlobalIndex>& nodes) const override
    {
        nodes = {node(cell, 0), node(cell + 1, 0), node(cell + 1, 1), node(cell, 1)};
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        const GlobalIndex nx = cell_count_ + 1;
        const Real xi = x(node_id % nx);
        const bool wall_node = node_id / nx == 0;
        const Real yi = wall_node ? (wall_profile_ ? wall_profile_(xi) : Real{0.0}) : Real{1.0};
        return {{xi, yi, Real{0.0}}};
    }

    void getCellCoordinates(GlobalIndex cell, std::vector<std::array<Real, 3>>& coordinates) const override
    {
        std::vector<GlobalIndex> nodes;
        getCellNodes(cell, nodes);
        coordinates.clear();
        for (const auto node_id : nodes) {
            coordinates.push_back(getNodeCoordinates(node_id));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face, GlobalIndex cell) const override
    {
        return face == wallFace(cell) ? LocalIndex{0} : INVALID_LOCAL_INDEX;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face) const override
    {
        return face >= wallFace(0) && face < wallFace(cell_count_) ? kWallMarker : -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex face) const override
    {
        const GlobalIndex left = face - 400;
        return {left, left + 1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex cell = 0; cell < cell_count_; ++cell) {
            callback(cell);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex cell = 0; cell < cell_count_; ++cell) {
            if (isOwnedCell(cell)) {
                callback(cell);
            }
        }
    }

    /// The wall faces of every cell, owned or not (the consumers filter).
    void forEachBoundaryFace(int marker, std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker >= 0 && marker != kWallMarker) {
            return;
        }
        for (GlobalIndex cell = 0; cell < cell_count_; ++cell) {
            callback(wallFace(cell), cell);
        }
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        for (GlobalIndex face = 0; face + 1 < cell_count_; ++face) {
            callback(400 + face, face, face + 1);
        }
    }

    [[nodiscard]] Real extent() const noexcept { return extent_; }
    [[nodiscard]] Real cellWidth() const noexcept { return extent_ / static_cast<Real>(cell_count_); }
    [[nodiscard]] Real x(GlobalIndex i) const noexcept
    {
        return extent_ * static_cast<Real>(i) / static_cast<Real>(cell_count_);
    }
    [[nodiscard]] GlobalIndex node(GlobalIndex i, GlobalIndex j) const noexcept
    {
        return i + (cell_count_ + 1) * j;
    }
    [[nodiscard]] static GlobalIndex wallFace(GlobalIndex cell) noexcept { return 300 + cell; }

private:
    int rank_{0};
    int size_{1};
    GlobalIndex cell_count_{0};
    Real extent_{1.0};
    std::function<Real(Real)> wall_profile_{};
};

/// The Q1 DOFs of the strip: DOF = node.
[[nodiscard]] inline dofs::DofMap quadStripDofMap(const QuadStripMeshAccess& mesh)
{
    const GlobalIndex dof_count = mesh.numVertices();
    dofs::DofMap dof_map(mesh.numCells(), dof_count, /*dofs_per_cell=*/4);
    for (GlobalIndex cell = 0; cell < mesh.numCells(); ++cell) {
        std::vector<GlobalIndex> nodes;
        mesh.getCellNodes(cell, nodes);
        dof_map.setCellDofs(cell, nodes);
    }
    dof_map.setNumDofs(dof_count);
    dof_map.setNumLocalDofs(dof_count);
    dof_map.finalize();
    return dof_map;
}

/// @p f at every node of the strip.
[[nodiscard]] inline std::vector<Real> nodalValues(const QuadStripMeshAccess& mesh, const PlaneFunction& f)
{
    std::vector<Real> values(static_cast<std::size_t>(mesh.numVertices()));
    for (GlobalIndex node_id = 0; node_id < mesh.numVertices(); ++node_id) {
        const auto p = mesh.getNodeCoordinates(node_id);
        values[static_cast<std::size_t>(node_id)] = f(p[0], p[1]);
    }
    return values;
}

/// The assembler of the field kField (space @p space, DOFs @p dof_map) with
/// the given nodal values; the objects must outlive it.
inline void configureQuadStripAssembler(assembly::FunctionalAssembler& assembler,
                                        const QuadStripMeshAccess& mesh,
                                        const dofs::DofMap& dof_map,
                                        const spaces::FunctionSpace& space,
                                        std::span<const Real> solution)
{
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setPrimaryField(kField);
    assembler.setSolution(solution);
    assembly::FieldSolutionBinding binding;
    binding.field = kField;
    binding.space = &space;
    binding.dof_map = &dof_map;
    binding.dof_offset = 0;
    binding.field_global_size = dof_map.getNumDofs();
    binding.field_type = FieldType::Scalar;
    binding.value_dimension = 1;
    binding.n_components = 1;
    assembler.registerFieldBinding(binding);
}

template <typename Request>
void setMeshRevisions(const assembly::IMeshAccess& mesh, Request& request)
{
    request.mesh_geometry_revision = mesh.geometryRevision();
    request.mesh_topology_revision = mesh.topologyRevision();
    request.ownership_revision = mesh.ownershipRevision();
}

[[nodiscard]] inline interfaces::LevelSetInterfaceSource levelSetSource()
{
    return interfaces::LevelSetInterfaceSource::fromField(/*field_id=*/17, /*layout_revision=*/2, kSourceRevision);
}

[[nodiscard]] inline interfaces::LevelSetInterfaceDomain quadStripInterfaceDomain(const QuadStripMeshAccess& mesh,
                                                                                  const PlaneFunction& phi)
{
    interfaces::CutInterfaceDomainRequest request;
    request.source = levelSetSource();
    request.interface_marker = kInterfaceMarker;
    request.isovalue = 0.0;
    request.quadrature_order = 3;
    request.implicit_geometry_mode = "LinearCorner";
    request.implicit_quadrature_backend = "LinearCorner";
    request.implicit_fallback_status = "None";
    setMeshRevisions(mesh, request);

    interfaces::LevelSetInterfaceDomain domain(request);
    const std::vector<std::array<Real, 3>> reference_nodes{
        {{-1.0, -1.0, 0.0}},
        {{1.0, -1.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{-1.0, 1.0, 0.0}},
    };
    for (GlobalIndex cell = 0; cell < mesh.numCells(); ++cell) {
        const Real x0 = mesh.x(cell);
        const Real x1 = mesh.x(cell + 1);
        interfaces::LevelSetCellCutInput input;
        input.parent_cell = cell;
        input.element_type = ElementType::Quad4;
        input.node_coordinates = reference_nodes;
        input.level_set_values = {phi(x0, Real{0.0}), phi(x1, Real{0.0}), phi(x1, Real{1.0}), phi(x0, Real{1.0})};
        auto cut = interfaces::cutLinearLevelSetCell2D(request, input);
        for (auto& fragment : cut.fragments) {
            fragment.parent_cell_global_id = mesh.getCellGlobalId(cell);
            fragment.owner_rank = mesh.getCellOwnerRank(cell);
            fragment.stable_id = 0u;
            domain.addFragment(std::move(fragment));
        }
        for (auto& region : cut.volume_regions) {
            region.parent_cell_global_id = mesh.getCellGlobalId(cell);
            region.owner_rank = mesh.getCellOwnerRank(cell);
            region.stable_id = 0u;
            domain.addVolumeRegion(std::move(region));
        }
    }
    return domain;
}

/**
 * The geometry snapshot of phi on the strip: the interface (kInterfaceMarker)
 * with both cut volumes, the contact points with the wall (kContactMarker)
 * and the wetted wall phi < 0 (kWettedWallMarker).
 */
[[nodiscard]] inline std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot> buildQuadStripSnapshot(
    const QuadStripMeshAccess& mesh,
    const PlaneFunction& phi,
    interfaces::FreeSurfaceGeometryOwnershipCollective collective)
{
    auto interface_domain = quadStripInterfaceDomain(mesh, phi);

    interfaces::GeneratedInterfaceBoundaryIntersectionRequest contact_request;
    contact_request.source = levelSetSource();
    contact_request.generated_domain_id = "functional_form_contact";
    contact_request.interface_marker = kInterfaceMarker;
    contact_request.boundary_marker = kWallMarker;
    contact_request.intersection_marker = kContactMarker;
    contact_request.isovalue = 0.0;
    contact_request.quadrature_order = 3;
    contact_request.frame = geometry::CutGeometryFrame::Reference;
    contact_request.source_value_revision = kSourceRevision;
    setMeshRevisions(mesh, contact_request);
    auto contact_domain =
        interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(contact_request, interface_domain, mesh);

    interfaces::GeneratedActiveBoundaryRequest wall_request;
    wall_request.source = levelSetSource();
    wall_request.generated_domain_id = "functional_form_wetted_wall";
    wall_request.interface_marker = kInterfaceMarker;
    wall_request.boundary_marker = kWallMarker;
    wall_request.active_boundary_marker = kWettedWallMarker;
    wall_request.side = geometry::CutIntegrationSide::Negative;
    wall_request.quadrature_order = 3;
    wall_request.frame = geometry::CutGeometryFrame::Reference;
    wall_request.source_value_revision = kSourceRevision;
    setMeshRevisions(mesh, wall_request);
    interfaces::GeneratedActiveBoundaryScalarField wall_scalar;
    wall_scalar.value_at_node = [&mesh, phi](GlobalIndex node_id) {
        const auto p = mesh.getNodeCoordinates(node_id);
        return phi(p[0], p[1]);
    };
    auto wetted_wall = interfaces::buildGeneratedActiveBoundaryDomain(
        wall_request, interface_domain, contact_domain, mesh, wall_scalar);

    const auto corner_values = [&mesh, phi](GlobalIndex cell) {
        const Real x0 = mesh.x(cell);
        const Real x1 = mesh.x(cell + 1);
        return std::array<Real, 4>{{phi(x0, Real{0.0}), phi(x1, Real{0.0}), phi(x1, Real{1.0}), phi(x0, Real{1.0})}};
    };
    interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
    scalar.value = [corner_values](GlobalIndex cell,
                                   const std::array<Real, 3>& xi,
                                   const geometry::CutQuadratureProvenance&) {
        const auto v = corner_values(cell);
        return Real{0.25} * (v[0] * (Real{1.0} - xi[0]) * (Real{1.0} - xi[1]) +
                             v[1] * (Real{1.0} + xi[0]) * (Real{1.0} - xi[1]) +
                             v[2] * (Real{1.0} + xi[0]) * (Real{1.0} + xi[1]) +
                             v[3] * (Real{1.0} - xi[0]) * (Real{1.0} + xi[1]));
    };
    scalar.reference_gradient = [corner_values](GlobalIndex cell,
                                                const std::array<Real, 3>& xi,
                                                const geometry::CutQuadratureProvenance&) {
        const auto v = corner_values(cell);
        return std::array<Real, 3>{{
            Real{0.25} * (-v[0] * (Real{1.0} - xi[1]) + v[1] * (Real{1.0} - xi[1]) +
                          v[2] * (Real{1.0} + xi[1]) - v[3] * (Real{1.0} + xi[1])),
            Real{0.25} * (-v[0] * (Real{1.0} - xi[0]) - v[1] * (Real{1.0} + xi[0]) +
                          v[2] * (Real{1.0} + xi[0]) + v[3] * (Real{1.0} - xi[0])),
            Real{0.0},
        }};
    };

    std::vector<interfaces::GeneratedInterfaceBoundaryIntersectionDomain> contact_domains;
    contact_domains.push_back(std::move(contact_domain));
    std::vector<interfaces::GeneratedActiveBoundaryDomain> wall_domains;
    wall_domains.push_back(std::move(wetted_wall));
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition = false;
    policy.minimum_retained_volume_fraction = assembly::CutIntegrationContext::minGeneratedCutVolumeFraction();
    return interfaces::buildFreeSurfaceGeometrySnapshot(std::move(interface_domain),
                                                        std::move(contact_domains),
                                                        std::move(wall_domains),
                                                        mesh,
                                                        policy,
                                                        std::move(scalar),
                                                        "functional_form_evaluator",
                                                        std::move(collective));
}

/// A cut integration context holding the snapshot of phi on the strip.
[[nodiscard]] inline std::shared_ptr<assembly::CutIntegrationContext> quadStripCutContext(
    const QuadStripMeshAccess& mesh,
    const PlaneFunction& phi,
    interfaces::FreeSurfaceGeometryOwnershipCollective collective = {})
{
    auto context = std::make_shared<assembly::CutIntegrationContext>();
    context->addFreeSurfaceGeometrySnapshot(buildQuadStripSnapshot(mesh, phi, std::move(collective)));
    return context;
}

} // namespace functional_form_test
} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TESTS_UNIT_SYSTEMS_FUNCTIONAL_FORM_EVALUATOR_FIXTURES_H
