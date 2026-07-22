/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Geometry/CutQuadrature.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace svmp::FE;

namespace {

constexpr GlobalIndex kCellCount = 4;
constexpr int kWallMarker = 41;
constexpr int kInterfaceMarker = 3041;

[[nodiscard]] std::vector<std::uint64_t> allGatherUnsignedValues(
    std::span<const std::uint64_t> local)
{
    const int local_count = static_cast<int>(local.size());
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<int> counts(static_cast<std::size_t>(size), 0);
    MPI_Allgather(&local_count,
                  1,
                  MPI_INT,
                  counts.data(),
                  1,
                  MPI_INT,
                  MPI_COMM_WORLD);

    std::vector<int> displacements(static_cast<std::size_t>(size), 0);
    int global_count = 0;
    for (int peer = 0; peer < size; ++peer) {
        displacements[static_cast<std::size_t>(peer)] = global_count;
        global_count += counts[static_cast<std::size_t>(peer)];
    }
    std::vector<std::uint64_t> gathered(
        static_cast<std::size_t>(global_count));
    MPI_Allgatherv(local.empty() ? nullptr : local.data(),
                   local_count,
                   MPI_UINT64_T,
                   gathered.empty() ? nullptr : gathered.data(),
                   counts.data(),
                   displacements.data(),
                   MPI_UINT64_T,
                   MPI_COMM_WORLD);
    return gathered;
}

[[nodiscard]] interfaces::FreeSurfaceGeometryOwnershipCollective
mpiOwnershipCollective(int rank, int size)
{
    interfaces::FreeSurfaceGeometryOwnershipCollective collective;
    collective.rank = rank;
    collective.size = size;
    collective.all_gather_owned_rule_identity_values =
        allGatherUnsignedValues;
    collective.all_gather_revision_values = allGatherUnsignedValues;
    return collective;
}

class GhostedHexStripMeshAccess final : public assembly::IMeshAccess {
public:
    GhostedHexStripMeshAccess(int rank, int size, bool serial_owner_view)
        : rank_(rank)
        , size_(size)
        , serial_owner_view_(serial_owner_view)
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return kCellCount; }

    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        GlobalIndex count = 0;
        for (GlobalIndex cell = 0; cell < kCellCount; ++cell) {
            if (isOwnedCell(cell)) {
                ++count;
            }
        }
        return count;
    }

    [[nodiscard]] GlobalIndex numVertices() const override
    {
        return (kCellCount + 1) * 4;
    }

    [[nodiscard]] GlobalIndex numBoundaryFaces() const override
    {
        return kCellCount;
    }

    [[nodiscard]] GlobalIndex numInteriorFaces() const override
    {
        return kCellCount - 1;
    }

    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool globalEntityIdsAvailable() const override
    {
        return true;
    }

    [[nodiscard]] GlobalIndex getCellGlobalId(
        GlobalIndex cell_id) const override
    {
        return 1000 + cell_id;
    }

    [[nodiscard]] GlobalIndex getBoundaryFaceGlobalId(
        GlobalIndex face_id) const override
    {
        return 10000 + face_id;
    }

    [[nodiscard]] int parallelRank() const override { return rank_; }
    [[nodiscard]] int parallelSize() const override { return size_; }

    [[nodiscard]] int getCellOwnerRank(
        GlobalIndex cell_id) const override
    {
        if (serial_owner_view_) {
            return 0;
        }
        return static_cast<int>(cell_id % static_cast<GlobalIndex>(size_));
    }

    [[nodiscard]] int getBoundaryFaceOwnerRank(
        GlobalIndex, GlobalIndex parent_cell) const override
    {
        return getCellOwnerRank(parent_cell);
    }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        if (serial_owner_view_) {
            return true;
        }
        return static_cast<int>(cell_id % static_cast<GlobalIndex>(size_)) ==
               rank_;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex) const override
    {
        return ElementType::Hex8;
    }

    void getCellNodes(GlobalIndex cell_id,
                      std::vector<GlobalIndex>& nodes) const override
    {
        const auto node = [](GlobalIndex i, GlobalIndex j, GlobalIndex k) {
            return i + (kCellCount + 1) * (j + 2 * k);
        };
        nodes = {node(cell_id, 0, 0),
                 node(cell_id + 1, 0, 0),
                 node(cell_id + 1, 1, 0),
                 node(cell_id, 1, 0),
                 node(cell_id, 0, 1),
                 node(cell_id + 1, 0, 1),
                 node(cell_id + 1, 1, 1),
                 node(cell_id, 1, 1)};
    }

    [[nodiscard]] std::array<Real, 3>
    getNodeCoordinates(GlobalIndex node_id) const override
    {
        const GlobalIndex nx = kCellCount + 1;
        const GlobalIndex i = node_id % nx;
        const GlobalIndex plane_index = node_id / nx;
        const GlobalIndex j = plane_index % 2;
        const GlobalIndex k = plane_index / 2;
        return {{static_cast<Real>(i),
                 static_cast<Real>(j),
                 static_cast<Real>(k)}};
    }

    void getCellCoordinates(
        GlobalIndex cell_id,
        std::vector<std::array<Real, 3>>& coordinates) const override
    {
        std::vector<GlobalIndex> nodes;
        getCellNodes(cell_id, nodes);
        coordinates.clear();
        coordinates.reserve(nodes.size());
        for (const auto node : nodes) {
            coordinates.push_back(getNodeCoordinates(node));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex cell_id) const override
    {
        return face_id == 100 + cell_id ? LocalIndex{0}
                                        : INVALID_LOCAL_INDEX;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        return face_id >= 100 && face_id < 100 + kCellCount ? kWallMarker : -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex face_id) const override
    {
        const GlobalIndex left = face_id - 200;
        return {left, left + 1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex cell = 0; cell < kCellCount; ++cell) {
            callback(cell);
        }
    }

    void forEachOwnedCell(
        std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex cell = 0; cell < kCellCount; ++cell) {
            if (isOwnedCell(cell)) {
                callback(cell);
            }
        }
    }

    void forEachBoundaryFace(
        int marker,
        std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker >= 0 && marker != kWallMarker) {
            return;
        }
        // Every rank exposes every boundary face, including ghost faces. The
        // duplicate callback also exercises local face-id deduplication.
        for (GlobalIndex cell = 0; cell < kCellCount; ++cell) {
            callback(100 + cell, cell);
            callback(100 + cell, cell);
        }
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback)
        const override
    {
        for (GlobalIndex face = 0; face < kCellCount - 1; ++face) {
            callback(200 + face, face, face + 1);
        }
    }

private:
    int rank_{0};
    int size_{1};
    bool serial_owner_view_{false};
};

class GhostedQuadStripMeshAccess final : public assembly::IMeshAccess {
public:
    GhostedQuadStripMeshAccess(int rank,
                               int size,
                               bool serial_owner_view,
                               GlobalIndex cell_count = kCellCount,
                               Real extent = Real{4.0},
                               bool every_cell_owned_locally = false)
        : rank_(rank)
        , size_(size)
        , serial_owner_view_(serial_owner_view)
        , cell_count_(cell_count)
        , extent_(extent)
        , every_cell_owned_locally_(every_cell_owned_locally)
        , geometry_revision_(1001u + static_cast<std::uint64_t>(rank))
        , topology_revision_(2001u + static_cast<std::uint64_t>(rank))
        , ownership_revision_(3001u + static_cast<std::uint64_t>(rank))
        , numbering_revision_(4001u + static_cast<std::uint64_t>(rank))
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

    [[nodiscard]] GlobalIndex numVertices() const override
    {
        return (cell_count_ + 1) * 2;
    }

    [[nodiscard]] GlobalIndex numBoundaryFaces() const override
    {
        return cell_count_;
    }

    [[nodiscard]] GlobalIndex numInteriorFaces() const override
    {
        return cell_count_ - 1;
    }

    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool revisionTrackingAvailable() const override
    {
        return true;
    }

    [[nodiscard]] std::uint64_t geometryRevision() const override
    {
        return geometry_revision_;
    }

    [[nodiscard]] std::uint64_t topologyRevision() const override
    {
        return topology_revision_;
    }

    [[nodiscard]] std::uint64_t ownershipRevision() const override
    {
        return ownership_revision_;
    }

    [[nodiscard]] std::uint64_t numberingRevision() const override
    {
        return numbering_revision_;
    }

    [[nodiscard]] bool globalEntityIdsAvailable() const override
    {
        return true;
    }

    [[nodiscard]] GlobalIndex getCellGlobalId(
        GlobalIndex cell_id) const override
    {
        return 2000 + cell_id;
    }

    [[nodiscard]] GlobalIndex getBoundaryFaceGlobalId(
        GlobalIndex face_id) const override
    {
        return 20000 + face_id;
    }

    [[nodiscard]] int parallelRank() const override { return rank_; }
    [[nodiscard]] int parallelSize() const override { return size_; }

    [[nodiscard]] int getCellOwnerRank(
        GlobalIndex cell_id) const override
    {
        if (serial_owner_view_) {
            return 0;
        }
        if (every_cell_owned_locally_) {
            return rank_;
        }
        return static_cast<int>(cell_id % static_cast<GlobalIndex>(size_));
    }

    [[nodiscard]] int getBoundaryFaceOwnerRank(
        GlobalIndex, GlobalIndex parent_cell) const override
    {
        return getCellOwnerRank(parent_cell);
    }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        if (serial_owner_view_) {
            return true;
        }
        if (every_cell_owned_locally_) {
            return true;
        }
        return static_cast<int>(cell_id % static_cast<GlobalIndex>(size_)) ==
               rank_;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex) const override
    {
        return ElementType::Quad4;
    }

    void getCellNodes(GlobalIndex cell_id,
                      std::vector<GlobalIndex>& nodes) const override
    {
        const auto node = [this](GlobalIndex i, GlobalIndex j) {
            return i + (cell_count_ + 1) * j;
        };
        nodes = {node(cell_id, 0),
                 node(cell_id + 1, 0),
                 node(cell_id + 1, 1),
                 node(cell_id, 1)};
    }

    [[nodiscard]] std::array<Real, 3>
    getNodeCoordinates(GlobalIndex node_id) const override
    {
        const GlobalIndex nx = cell_count_ + 1;
        const GlobalIndex i = node_id % nx;
        const GlobalIndex j = node_id / nx;
        return {{extent_ * static_cast<Real>(i) /
                     static_cast<Real>(cell_count_),
                 static_cast<Real>(j),
                 Real{0.0}}};
    }

    void getCellCoordinates(
        GlobalIndex cell_id,
        std::vector<std::array<Real, 3>>& coordinates) const override
    {
        std::vector<GlobalIndex> nodes;
        getCellNodes(cell_id, nodes);
        coordinates.clear();
        coordinates.reserve(nodes.size());
        for (const auto node : nodes) {
            coordinates.push_back(getNodeCoordinates(node));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex cell_id) const override
    {
        return face_id == 300 + cell_id ? LocalIndex{0}
                                        : INVALID_LOCAL_INDEX;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        return face_id >= 300 && face_id < 300 + cell_count_ ? kWallMarker : -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex face_id) const override
    {
        const GlobalIndex left = face_id - 400;
        return {left, left + 1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex cell = 0; cell < cell_count_; ++cell) {
            callback(cell);
        }
    }

    void forEachOwnedCell(
        std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex cell = 0; cell < cell_count_; ++cell) {
            if (isOwnedCell(cell)) {
                callback(cell);
            }
        }
    }

    void forEachBoundaryFace(
        int marker,
        std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker >= 0 && marker != kWallMarker) {
            return;
        }
        // All boundary edges are visible on all ranks, including ghost data.
        // Repeating each callback verifies owner filtering and face-id
        // deduplication for the zero-dimensional contact domain.
        for (GlobalIndex cell = 0; cell < cell_count_; ++cell) {
            callback(300 + cell, cell);
            callback(300 + cell, cell);
        }
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback)
        const override
    {
        for (GlobalIndex face = 0; face < cell_count_ - 1; ++face) {
            callback(400 + face, face, face + 1);
        }
    }

    [[nodiscard]] Real extent() const noexcept { return extent_; }

    void advanceGeometryRevision() noexcept { ++geometry_revision_; }

private:
    int rank_{0};
    int size_{1};
    bool serial_owner_view_{false};
    GlobalIndex cell_count_{kCellCount};
    Real extent_{Real{4.0}};
    bool every_cell_owned_locally_{false};
    std::uint64_t geometry_revision_{0u};
    std::uint64_t topology_revision_{0u};
    std::uint64_t ownership_revision_{0u};
    std::uint64_t numbering_revision_{0u};
};

template <typename Request>
void applyTrackedMeshRevisions(Request& request,
                               const assembly::IMeshAccess* mesh)
{
    if (mesh == nullptr || !mesh->revisionTrackingAvailable()) {
        return;
    }
    request.mesh_geometry_revision = mesh->geometryRevision();
    request.mesh_topology_revision = mesh->topologyRevision();
    request.ownership_revision = mesh->ownershipRevision();
}

[[nodiscard]] interfaces::GeneratedInterfaceBoundaryIntersectionRequest
contactRequest(std::uint64_t source_revision = 5u,
               const assembly::IMeshAccess* mesh = nullptr)
{
    interfaces::GeneratedInterfaceBoundaryIntersectionRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromField(
        /*field_id=*/17,
        /*layout_revision=*/2,
        source_revision);
    request.generated_domain_id = "mpi_contact_measure";
    request.interface_marker = kInterfaceMarker;
    request.boundary_marker = kWallMarker;
    request.isovalue = 0.0;
    request.quadrature_order = 3;
    request.frame = geometry::CutGeometryFrame::Reference;
    request.source_value_revision = source_revision;
    applyTrackedMeshRevisions(request, mesh);
    return request;
}

[[nodiscard]] interfaces::GeneratedInterfaceBoundaryIntersectionRequest
pointContactRequest(std::uint64_t source_revision = 5u,
                    const assembly::IMeshAccess* mesh = nullptr)
{
    auto request = contactRequest(source_revision, mesh);
    request.generated_domain_id = "mpi_contact_point_measure";
    return request;
}

[[nodiscard]] interfaces::CutInterfaceDomainRequest interfaceRequest(
    std::uint64_t source_revision = 5u,
    const assembly::IMeshAccess* mesh = nullptr)
{
    interfaces::CutInterfaceDomainRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromField(
        /*field_id=*/17,
        /*layout_revision=*/2,
        source_revision);
    request.interface_marker = kInterfaceMarker;
    request.isovalue = 0.0;
    request.quadrature_order = 3;
    request.implicit_geometry_mode = "LinearCorner";
    request.implicit_quadrature_backend = "LinearCorner";
    request.implicit_fallback_status = "None";
    applyTrackedMeshRevisions(request, mesh);
    return request;
}

[[nodiscard]] interfaces::LevelSetInterfaceDomain hexInterfaceDomain()
{
    interfaces::LevelSetInterfaceDomain domain(interfaceRequest());
    for (GlobalIndex cell = 0; cell < kCellCount; ++cell) {
        interfaces::CutInterfaceFragment fragment;
        fragment.interface_marker = kInterfaceMarker;
        fragment.parent_cell = cell;
        fragment.kind = interfaces::CutInterfaceFragmentKind::Polygon;
        fragment.normal = {{0.0, 1.0, 0.0}};
        fragment.measure = 4.0;
        fragment.min_gradient_norm = 1.0;
        fragment.topology_id =
            "cell-" + std::to_string(cell) + "-horizontal-plane";
        fragment.implicit_quadrature_backend = "LinearCorner";
        fragment.implicit_fallback_status = "None";
        fragment.vertices = {
            {.point = {{-1.0, 0.0, -1.0}},
             .parent_coordinate = {{-1.0, 0.0, -1.0}}},
            {.point = {{1.0, 0.0, -1.0}},
             .parent_coordinate = {{1.0, 0.0, -1.0}}},
            {.point = {{1.0, 0.0, 1.0}},
             .parent_coordinate = {{1.0, 0.0, 1.0}}},
            {.point = {{-1.0, 0.0, 1.0}},
             .parent_coordinate = {{-1.0, 0.0, 1.0}}},
        };
        domain.addFragment(std::move(fragment));
    }
    return domain;
}

[[nodiscard]] interfaces::LevelSetInterfaceDomain quadInterfaceDomain()
{
    interfaces::LevelSetInterfaceDomain domain(interfaceRequest());
    for (GlobalIndex cell = 0; cell < kCellCount; ++cell) {
        const Real direction = cell % 2 == 0 ? Real{1.0} : Real{-1.0};
        interfaces::CutInterfaceFragment fragment;
        fragment.interface_marker = kInterfaceMarker;
        fragment.parent_cell = cell;
        fragment.kind = interfaces::CutInterfaceFragmentKind::Segment;
        fragment.normal = {{direction, 0.0, 0.0}};
        fragment.measure = 2.0;
        fragment.min_gradient_norm = 1.0;
        fragment.topology_id =
            "cell-" + std::to_string(cell) + "-vertical-line";
        fragment.implicit_quadrature_backend = "LinearCorner";
        fragment.implicit_fallback_status = "None";
        fragment.vertices = {
            {.point = {{0.0, -1.0, 0.0}},
             .parent_coordinate = {{0.0, -1.0, 0.0}}},
            {.point = {{0.0, 1.0, 0.0}},
             .parent_coordinate = {{0.0, 1.0, 0.0}}},
        };
        domain.addFragment(std::move(fragment));
    }
    return domain;
}

[[nodiscard]] interfaces::GeneratedInterfaceBoundaryIntersectionDomain
buildContactDomain(const GhostedHexStripMeshAccess& mesh)
{
    interfaces::GeneratedInterfaceBoundaryIntersectionScalarField field;
    field.value_at_node = [&mesh](GlobalIndex node) {
        return mesh.getNodeCoordinates(node)[1] - Real{0.5};
    };
    const auto interface_domain = hexInterfaceDomain();
    return interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
        contactRequest(), interface_domain, mesh, field);
}

[[nodiscard]] interfaces::GeneratedInterfaceBoundaryIntersectionDomain
buildPointContactDomain(const GhostedQuadStripMeshAccess& mesh)
{
    interfaces::GeneratedInterfaceBoundaryIntersectionScalarField field;
    field.value_at_node = [&mesh](GlobalIndex node) {
        const GlobalIndex i =
            static_cast<GlobalIndex>(mesh.getNodeCoordinates(node)[0]);
        return i % 2 == 0 ? Real{-1.0} : Real{1.0};
    };
    const auto interface_domain = quadInterfaceDomain();
    return interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
        pointContactRequest(), interface_domain, mesh, field);
}

struct RotatedSessileCap {
    Real center_x{2.1};
    Real center_y{0.2};
    Real radius_x{0.85};
    Real radius_y{0.55};
    Real angle{0.0};
};

[[nodiscard]] Real sessileCapValue(const RotatedSessileCap& cap,
                                   Real x,
                                   Real y)
{
    const Real c = std::cos(cap.angle);
    const Real s = std::sin(cap.angle);
    const Real dx = x - cap.center_x;
    const Real dy = y - cap.center_y;
    const Real u = c * dx + s * dy;
    const Real v = -s * dx + c * dy;
    return u * u / (cap.radius_x * cap.radius_x) +
           v * v / (cap.radius_y * cap.radius_y) - Real{1.0};
}

[[nodiscard]] std::array<Real, 2> sessileCapWallRoots(
    const RotatedSessileCap& cap)
{
    const Real c = std::cos(cap.angle);
    const Real s = std::sin(cap.angle);
    const Real inv_a2 = Real{1.0} / (cap.radius_x * cap.radius_x);
    const Real inv_b2 = Real{1.0} / (cap.radius_y * cap.radius_y);
    const Real A = c * c * inv_a2 + s * s * inv_b2;
    const Real B = Real{2.0} * c * s * cap.center_y *
                   (inv_b2 - inv_a2);
    const Real C = cap.center_y * cap.center_y *
                       (s * s * inv_a2 + c * c * inv_b2) -
                   Real{1.0};
    const Real discriminant = B * B - Real{4.0} * A * C;
    EXPECT_GT(discriminant, Real{0.0});
    const Real root = std::sqrt(std::max(discriminant, Real{0.0}));
    std::array<Real, 2> roots{{
        cap.center_x + (-B - root) / (Real{2.0} * A),
        cap.center_x + (-B + root) / (Real{2.0} * A),
    }};
    if (roots[1] < roots[0]) {
        std::swap(roots[0], roots[1]);
    }
    return roots;
}

[[nodiscard]] interfaces::LevelSetInterfaceDomain sessileCapInterfaceDomain(
    const GhostedQuadStripMeshAccess& mesh,
    const RotatedSessileCap& cap,
    std::uint64_t source_revision)
{
    const auto request = interfaceRequest(source_revision, &mesh);
    interfaces::LevelSetInterfaceDomain domain(request);
    const Real h = mesh.extent() / static_cast<Real>(mesh.numCells());
    const std::vector<std::array<Real, 3>> reference_nodes{
        {{-1.0, -1.0, 0.0}},
        {{1.0, -1.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{-1.0, 1.0, 0.0}},
    };
    for (GlobalIndex cell = 0; cell < mesh.numCells(); ++cell) {
        const Real x0 = h * static_cast<Real>(cell);
        const Real x1 = x0 + h;
        interfaces::LevelSetCellCutInput input;
        input.parent_cell = cell;
        input.element_type = ElementType::Quad4;
        input.node_coordinates = reference_nodes;
        input.level_set_values = {
            sessileCapValue(cap, x0, Real{0.0}),
            sessileCapValue(cap, x1, Real{0.0}),
            sessileCapValue(cap, x1, Real{1.0}),
            sessileCapValue(cap, x0, Real{1.0}),
        };
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

[[nodiscard]] interfaces::GeneratedInterfaceBoundaryIntersectionDomain
buildSessileCapContactDomain(const GhostedQuadStripMeshAccess& mesh,
                             const RotatedSessileCap& cap,
                             std::uint64_t source_revision)
{
    const auto interface_domain =
        sessileCapInterfaceDomain(mesh, cap, source_revision);
    return interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
        pointContactRequest(source_revision, &mesh), interface_domain, mesh);
}

[[nodiscard]] std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot>
buildSessileCapSnapshot(
    const GhostedQuadStripMeshAccess& mesh,
    const RotatedSessileCap& cap,
    std::uint64_t source_revision,
    interfaces::FreeSurfaceGeometryOwnershipCollective collective,
    std::string domain_id)
{
    auto interface_domain =
        sessileCapInterfaceDomain(mesh, cap, source_revision);
    const GlobalIndex cell_count = mesh.numCells();
    const Real cell_width = mesh.extent() / static_cast<Real>(cell_count);
    const auto corner_values = [cap, cell_count, cell_width](
                                   GlobalIndex parent_cell) {
        if (parent_cell < 0 || parent_cell >= cell_count) {
            throw std::invalid_argument(
                "sessile-cap scalar evaluator received an unknown parent cell");
        }
        const Real x0 = cell_width * static_cast<Real>(parent_cell);
        const Real x1 = x0 + cell_width;
        return std::array<Real, 4>{{
            sessileCapValue(cap, x0, Real{0.0}),
            sessileCapValue(cap, x1, Real{0.0}),
            sessileCapValue(cap, x1, Real{1.0}),
            sessileCapValue(cap, x0, Real{1.0}),
        }};
    };
    interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
    scalar.value = [corner_values](
                       GlobalIndex parent_cell,
                       const std::array<Real, 3>& xi,
                       const geometry::CutQuadratureProvenance&) {
        const auto values = corner_values(parent_cell);
        return Real{0.25} *
               (values[0] * (Real{1.0} - xi[0]) *
                    (Real{1.0} - xi[1]) +
                values[1] * (Real{1.0} + xi[0]) *
                    (Real{1.0} - xi[1]) +
                values[2] * (Real{1.0} + xi[0]) *
                    (Real{1.0} + xi[1]) +
                values[3] * (Real{1.0} - xi[0]) *
                    (Real{1.0} + xi[1]));
    };
    scalar.reference_gradient = [corner_values](
                                    GlobalIndex parent_cell,
                                    const std::array<Real, 3>& xi,
                                    const geometry::CutQuadratureProvenance&) {
        const auto values = corner_values(parent_cell);
        return std::array<Real, 3>{{
            Real{0.25} *
                (-values[0] * (Real{1.0} - xi[1]) +
                 values[1] * (Real{1.0} - xi[1]) +
                 values[2] * (Real{1.0} + xi[1]) -
                 values[3] * (Real{1.0} + xi[1])),
            Real{0.25} *
                (-values[0] * (Real{1.0} - xi[0]) -
                 values[1] * (Real{1.0} + xi[0]) +
                 values[2] * (Real{1.0} + xi[0]) +
                 values[3] * (Real{1.0} - xi[0])),
            Real{0.0},
        }};
    };
    std::vector<
        interfaces::GeneratedInterfaceBoundaryIntersectionDomain>
        contact_domains;
    contact_domains.push_back(
        interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(
            pointContactRequest(source_revision, &mesh),
            interface_domain,
            mesh));
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition = false;
    return interfaces::buildFreeSurfaceGeometrySnapshot(
        std::move(interface_domain),
        std::move(contact_domains),
        {},
        mesh,
        policy,
        std::move(scalar),
        std::move(domain_id),
        std::move(collective));
}

struct MovingContactState {
    std::uint64_t point_count{0u};
    Real counting_measure{0.0};
    Real x_sum{0.0};
    Real min_x{std::numeric_limits<Real>::infinity()};
    Real max_x{-std::numeric_limits<Real>::infinity()};
    long long min_parent_cell{std::numeric_limits<long long>::max()};
    std::uint64_t revision_mismatch_count{0u};
    std::vector<std::array<std::uint64_t, 7>> source_mappings{};
};

[[nodiscard]] MovingContactState movingContactState(
    const interfaces::GeneratedInterfaceBoundaryIntersectionDomain& domain,
    GlobalIndex cell_count,
    Real extent,
    std::uint64_t source_revision)
{
    MovingContactState state;
    const Real h = extent / static_cast<Real>(cell_count);
    for (const auto& rule : domain.intersectionQuadratureRules()) {
        EXPECT_EQ(rule.geometric_dimension, 0);
        EXPECT_EQ(rule.frame, geometry::CutGeometryFrame::Reference);
        if (rule.provenance.source_value_revision != source_revision) {
            ++state.revision_mismatch_count;
        }
        if (rule.points.size() != 1u) {
            ADD_FAILURE() << "moving contact rule must contain one point";
            continue;
        }
        const auto& point = rule.points.front();
        const auto cell = static_cast<GlobalIndex>(rule.provenance.parent_entity);
        const Real x = h * static_cast<Real>(cell) +
                       Real{0.5} * h *
                           (point.parent_coordinate[0] + Real{1.0});
        ++state.point_count;
        state.counting_measure += point.weight;
        state.x_sum += x;
        state.min_x = std::min(state.min_x, x);
        state.max_x = std::max(state.max_x, x);
        state.min_parent_cell =
            std::min(state.min_parent_cell, static_cast<long long>(cell));
    }
    return state;
}

struct SnapshotMovingContactState {
    MovingContactState contact{};
    std::uint64_t snapshot_revision_key{0u};
    std::uint64_t rule_snapshot_mismatch_count{0u};
    std::uint64_t rule_source_revision_mismatch_count{0u};
    std::uint64_t orphan_contact_count{0u};
    std::uint64_t missing_contact_count{0u};
};

[[nodiscard]] std::vector<std::array<std::uint64_t, 7>>
globalContactSourceMappings(
    const interfaces::GeneratedInterfaceBoundaryIntersectionDomain& domain);

[[nodiscard]] SnapshotMovingContactState movingContactSnapshotState(
    const interfaces::FreeSurfaceGeometrySnapshot& snapshot,
    std::uint64_t source_revision)
{
    SnapshotMovingContactState state;
    const auto& revision = snapshot.revision();
    EXPECT_TRUE(revision.complete());
    EXPECT_EQ(revision.source_layout_revision, 2u);
    state.snapshot_revision_key = revision.snapshot_revision_key;
    if (revision.source_value_revision != source_revision) {
        ++state.rule_source_revision_mismatch_count;
    }

    const auto& interface_request = snapshot.interfaceDomain().request();
    EXPECT_EQ(interface_request.source.layout_revision, 2u);
    if (interface_request.source.value_revision != source_revision) {
        ++state.rule_source_revision_mismatch_count;
    }
    for (const auto& contact_domain : snapshot.contactDomains()) {
        EXPECT_EQ(contact_domain.request().source.layout_revision, 2u);
        if (contact_domain.request().source.value_revision != source_revision ||
            contact_domain.request().source_value_revision != source_revision) {
            ++state.rule_source_revision_mismatch_count;
        }
    }

    for (const auto& record : snapshot.rules()) {
        if (record.reference_rule.provenance
                    .free_surface_snapshot_revision_key !=
                revision.snapshot_revision_key ||
            record.physical_rule.free_surface_snapshot_revision_key !=
                revision.snapshot_revision_key) {
            ++state.rule_snapshot_mismatch_count;
        }
        if (record.reference_rule.provenance.source_value_revision !=
                source_revision ||
            record.physical_rule.source_value_revision != source_revision) {
            ++state.rule_source_revision_mismatch_count;
        }
    }

    const auto contact_rules = snapshot.retainedRules(
        interfaces::FreeSurfaceGeometryRuleRole::Contact);
    for (const auto* record : contact_rules) {
        EXPECT_NE(record, nullptr);
        if (record == nullptr) {
            continue;
        }
        const auto& rule = record->reference_rule;
        const auto& mapped = record->physical_rule;
        EXPECT_EQ(rule.geometric_dimension, 0);
        EXPECT_EQ(rule.provenance.frame,
                  geometry::CutGeometryFrame::Reference);
        EXPECT_EQ(rule.points.size(), 1u);
        EXPECT_EQ(mapped.points.size(), rule.points.size());
        if (mapped.points.size() != 1u) {
            ADD_FAILURE() << "moving contact snapshot rule must contain one point";
            continue;
        }
        const auto& point = mapped.points.front();
        ++state.contact.point_count;
        state.contact.counting_measure += point.physical_weight;
        state.contact.x_sum += point.physical_point[0];
        state.contact.min_x =
            std::min(state.contact.min_x, point.physical_point[0]);
        state.contact.max_x =
            std::max(state.contact.max_x, point.physical_point[0]);
        state.contact.min_parent_cell = std::min(
            state.contact.min_parent_cell,
            static_cast<long long>(rule.provenance.parent_entity));
    }

    const auto& ledger = snapshot.ledger();
    state.orphan_contact_count = static_cast<std::uint64_t>(
        ledger.orphan_contact_fragment_count);
    state.missing_contact_count = static_cast<std::uint64_t>(
        ledger.missing_contact_fragment_count);
    EXPECT_EQ(snapshot.contactDomains().size(), 1u);
    if (snapshot.contactDomains().size() == 1u) {
        state.contact.source_mappings =
            globalContactSourceMappings(snapshot.contactDomains().front());
    }
    return state;
}

[[nodiscard]] std::vector<std::uint64_t> snapshotContactRuleSignature(
    const interfaces::FreeSurfaceGeometrySnapshot& snapshot)
{
    std::vector<std::uint64_t> signature;
    for (const auto* record : snapshot.retainedRules(
             interfaces::FreeSurfaceGeometryRuleRole::Contact)) {
        EXPECT_NE(record, nullptr);
        if (record == nullptr) {
            continue;
        }
        signature.push_back(record->reference_rule.provenance
                                .free_surface_snapshot_revision_key);
        signature.push_back(static_cast<std::uint64_t>(
            record->reference_rule.provenance.parent_entity_global_id));
        signature.push_back(
            record->reference_rule.provenance.cut_topology_revision);
        signature.push_back(record->reference_rule.provenance.source_stable_id);
        signature.push_back(
            record->reference_rule.provenance.source_value_revision);
    }
    return signature;
}

[[nodiscard]] std::vector<std::uint64_t> contactRuleSignature(
    const interfaces::GeneratedInterfaceBoundaryIntersectionDomain& domain)
{
    std::vector<std::uint64_t> signature;
    for (const auto& rule : domain.intersectionQuadratureRules()) {
        signature.push_back(static_cast<std::uint64_t>(
            rule.provenance.parent_entity_global_id));
        signature.push_back(rule.provenance.cut_topology_revision);
        signature.push_back(rule.provenance.source_stable_id);
        signature.push_back(rule.provenance.source_value_revision);
    }
    return signature;
}

[[nodiscard]] std::vector<std::array<std::uint64_t, 7>>
globalContactSourceMappings(
    const interfaces::GeneratedInterfaceBoundaryIntersectionDomain& domain)
{
    constexpr std::size_t width = 7u;
    std::vector<std::uint64_t> local;
    for (const auto& fragment : domain.fragments()) {
        if (!fragment.active()) {
            continue;
        }
        local.push_back(static_cast<std::uint64_t>(fragment.parent_cell));
        local.push_back(static_cast<std::uint64_t>(
            fragment.parent_cell_global_id));
        local.push_back(fragment.stable_id);
        local.push_back(fragment.source_interface_stable_id);
        local.push_back(domain.request().source_value_revision);
        local.push_back(static_cast<std::uint64_t>(fragment.degeneracy));
        local.push_back(static_cast<std::uint64_t>(fragment.owner_rank + 1));
    }
    const auto gathered = allGatherUnsignedValues(local);
    EXPECT_EQ(gathered.size() % width, 0u);
    std::vector<std::array<std::uint64_t, width>> mappings;
    if (gathered.size() % width != 0u) {
        return mappings;
    }
    mappings.reserve(gathered.size() / width);
    for (std::size_t offset = 0; offset < gathered.size(); offset += width) {
        std::array<std::uint64_t, width> mapping{};
        std::copy_n(gathered.begin() +
                        static_cast<std::ptrdiff_t>(offset),
                    static_cast<std::ptrdiff_t>(width),
                    mapping.begin());
        mappings.push_back(mapping);
    }
    std::sort(mappings.begin(), mappings.end());
    return mappings;
}

struct ContactMeasure {
    Real physical_length{0.0};
    std::uint64_t fragment_count{0};
    std::uint64_t quadrature_point_count{0};
};

[[nodiscard]] ContactMeasure mappedContactMeasure(
    const interfaces::GeneratedInterfaceBoundaryIntersectionDomain& domain)
{
    geometry::CutGeometryJacobian J{};
    geometry::CutGeometryJacobian J_inv{};
    for (std::size_t d = 0; d < 3; ++d) {
        J[d][d] = Real{0.5};
        J_inv[d][d] = Real{2.0};
    }

    ContactMeasure measure;
    const auto rules = domain.intersectionQuadratureRules();
    measure.fragment_count = static_cast<std::uint64_t>(rules.size());
    for (const auto& rule : rules) {
        EXPECT_EQ(rule.geometric_dimension, 1);
        EXPECT_EQ(rule.frame, geometry::CutGeometryFrame::Reference);
        for (const auto& point : rule.points) {
            const auto mapped = geometry::mapReferenceCutCodimensionTwoGeometry(
                point, /*parent_dimension=*/3, J, J_inv);
            measure.physical_length += mapped.weight;
            ++measure.quadrature_point_count;
        }
    }
    return measure;
}

struct ContactPointMeasure {
    Real counting_measure{0.0};
    std::uint64_t fragment_count{0};
    std::uint64_t quadrature_point_count{0};
};

[[nodiscard]] ContactPointMeasure mappedContactPointMeasure(
    const interfaces::GeneratedInterfaceBoundaryIntersectionDomain& domain)
{
    // Deliberately non-unit physical scaling: a zero-dimensional measure must
    // remain one under the production codimension-two mapping.
    geometry::CutGeometryJacobian J{};
    geometry::CutGeometryJacobian J_inv{};
    J[0][0] = Real{3.0};
    J[1][1] = Real{0.25};
    J[2][2] = Real{1.0};
    J_inv[0][0] = Real{1.0} / Real{3.0};
    J_inv[1][1] = Real{4.0};
    J_inv[2][2] = Real{1.0};

    ContactPointMeasure measure;
    const auto rules = domain.intersectionQuadratureRules();
    measure.fragment_count = static_cast<std::uint64_t>(rules.size());
    for (const auto& rule : rules) {
        EXPECT_EQ(rule.geometric_dimension, 0);
        EXPECT_EQ(rule.frame, geometry::CutGeometryFrame::Reference);
        EXPECT_EQ(rule.points.size(), 1u);
        for (const auto& point : rule.points) {
            EXPECT_DOUBLE_EQ(point.weight, Real{1.0});
            const auto mapped = geometry::mapReferenceCutCodimensionTwoGeometry(
                point, /*parent_dimension=*/2, J, J_inv);
            EXPECT_DOUBLE_EQ(mapped.measure_scale, Real{1.0});
            EXPECT_DOUBLE_EQ(mapped.weight, Real{1.0});
            measure.counting_measure += mapped.weight;
            ++measure.quadrature_point_count;
        }
    }
    return measure;
}

} // namespace

TEST(GeneratedContactMeasureMPI,
     SerialAndGhostedPartitionHaveExactRankIndependentMultiplicity)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ContactMeasure serial;
    if (rank == 0) {
        const GhostedHexStripMeshAccess serial_mesh(
            /*rank=*/0, /*size=*/1, /*serial_owner_view=*/true);
        serial = mappedContactMeasure(buildContactDomain(serial_mesh));
    }
    MPI_Bcast(&serial.physical_length, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&serial.fragment_count,
              1,
              MPI_UINT64_T,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(&serial.quadrature_point_count,
              1,
              MPI_UINT64_T,
              0,
              MPI_COMM_WORLD);

    const GhostedHexStripMeshAccess partitioned_mesh(
        rank, size, /*serial_owner_view=*/false);
    const auto local = mappedContactMeasure(buildContactDomain(partitioned_mesh));

    ContactMeasure global;
    MPI_Allreduce(&local.physical_length,
                  &global.physical_length,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local.fragment_count,
                  &global.fragment_count,
                  1,
                  MPI_UINT64_T,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local.quadrature_point_count,
                  &global.quadrature_point_count,
                  1,
                  MPI_UINT64_T,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    EXPECT_DOUBLE_EQ(serial.physical_length, Real{4.0});
    EXPECT_EQ(serial.fragment_count, 4u);
    EXPECT_EQ(serial.quadrature_point_count, 8u);
    EXPECT_DOUBLE_EQ(global.physical_length, serial.physical_length);
    EXPECT_EQ(global.fragment_count, serial.fragment_count);
    EXPECT_EQ(global.quadrature_point_count,
              serial.quadrature_point_count);

    if (rank == 0) {
        RecordProperty("mpi_rank_count", size);
        RecordProperty("global_contact_length",
                       ::testing::PrintToString(global.physical_length));
        RecordProperty("global_fragment_count", global.fragment_count);
        RecordProperty("global_quadrature_point_count",
                       global.quadrature_point_count);
    }
}

TEST(GeneratedContactMeasureMPI,
     TwoDimensionalContactPointsHaveExactRankIndependentCountingMeasure)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ContactPointMeasure serial;
    if (rank == 0) {
        const GhostedQuadStripMeshAccess serial_mesh(
            /*rank=*/0, /*size=*/1, /*serial_owner_view=*/true);
        serial = mappedContactPointMeasure(buildPointContactDomain(serial_mesh));
    }
    MPI_Bcast(&serial.counting_measure, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&serial.fragment_count,
              1,
              MPI_UINT64_T,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(&serial.quadrature_point_count,
              1,
              MPI_UINT64_T,
              0,
              MPI_COMM_WORLD);

    const GhostedQuadStripMeshAccess partitioned_mesh(
        rank, size, /*serial_owner_view=*/false);
    const auto local =
        mappedContactPointMeasure(buildPointContactDomain(partitioned_mesh));
    const auto local_owned = static_cast<std::uint64_t>(
        partitioned_mesh.numOwnedCells());
    EXPECT_EQ(local.fragment_count, local_owned);
    EXPECT_EQ(local.quadrature_point_count, local_owned);
    EXPECT_DOUBLE_EQ(local.counting_measure,
                     static_cast<Real>(local_owned));

    ContactPointMeasure global;
    MPI_Allreduce(&local.counting_measure,
                  &global.counting_measure,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local.fragment_count,
                  &global.fragment_count,
                  1,
                  MPI_UINT64_T,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local.quadrature_point_count,
                  &global.quadrature_point_count,
                  1,
                  MPI_UINT64_T,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    EXPECT_DOUBLE_EQ(serial.counting_measure, Real{4.0});
    EXPECT_EQ(serial.fragment_count, 4u);
    EXPECT_EQ(serial.quadrature_point_count, 4u);
    EXPECT_DOUBLE_EQ(global.counting_measure, serial.counting_measure);
    EXPECT_EQ(global.fragment_count, serial.fragment_count);
    EXPECT_EQ(global.quadrature_point_count,
              serial.quadrature_point_count);

    if (rank == 0) {
        RecordProperty("mpi_rank_count", size);
        RecordProperty("global_contact_point_counting_measure",
                       ::testing::PrintToString(global.counting_measure));
        RecordProperty("global_contact_point_fragment_count",
                       global.fragment_count);
        RecordProperty("global_contact_point_quadrature_point_count",
                       global.quadrature_point_count);
    }
}

TEST(GeneratedContactMeasureMPI,
     SnapshotCollectiveRejectsGeometryOwnedByBothRanks)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "distributed ownership rejection requires two ranks";
    }

    RotatedSessileCap cap;
    cap.angle = Real{0.35};
    constexpr GlobalIndex cell_count = 16;
    constexpr std::uint64_t source_revision = 301u;

    GhostedQuadStripMeshAccess valid_mesh(
        rank,
        size,
        /*serial_owner_view=*/false,
        cell_count);
    const auto valid_snapshot = buildSessileCapSnapshot(
        valid_mesh,
        cap,
        source_revision,
        mpiOwnershipCollective(rank, size),
        "mpi_unique_contact_ownership");
    ASSERT_TRUE(valid_snapshot);
    EXPECT_GT(valid_snapshot->ledger().global_owned_rule_count, 0u);
    const auto local_revision =
        valid_snapshot->revision().snapshot_revision_key;
    std::uint64_t minimum_revision = 0u;
    std::uint64_t maximum_revision = 0u;
    MPI_Allreduce(&local_revision,
                  &minimum_revision,
                  1,
                  MPI_UINT64_T,
                  MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_revision,
                  &maximum_revision,
                  1,
                  MPI_UINT64_T,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_NE(minimum_revision, 0u);
    EXPECT_EQ(minimum_revision, maximum_revision);

    assembly::CutIntegrationContext live_context;
    live_context.addFreeSurfaceGeometrySnapshot(valid_snapshot);
    EXPECT_NO_THROW(
        live_context.assertAllFreeSurfaceGeometrySnapshotsCurrent(valid_mesh));
    valid_mesh.advanceGeometryRevision();
    bool rejected_stale_consumption = false;
    try {
        live_context.assertAllFreeSurfaceGeometrySnapshotsCurrent(valid_mesh);
    } catch (const std::invalid_argument&) {
        rejected_stale_consumption = true;
    }
    int local_stale_rejection = rejected_stale_consumption ? 1 : 0;
    int global_stale_rejection = 0;
    MPI_Allreduce(&local_stale_rejection,
                  &global_stale_rejection,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  MPI_COMM_WORLD);
    EXPECT_EQ(global_stale_rejection, 1);

    const GhostedQuadStripMeshAccess duplicate_owner_mesh(
        rank,
        size,
        /*serial_owner_view=*/false,
        cell_count,
        Real{4.0},
        /*every_cell_owned_locally=*/true);
    bool rejected_duplicate_owner = false;
    try {
        (void)buildSessileCapSnapshot(
            duplicate_owner_mesh,
            cap,
            source_revision,
            mpiOwnershipCollective(rank, size),
            "mpi_duplicate_contact_ownership");
    } catch (const std::invalid_argument& error) {
        rejected_duplicate_owner =
            std::string(error.what()).find("owned by more than one rank") !=
            std::string::npos;
    }

    int local_rejected = rejected_duplicate_owner ? 1 : 0;
    int globally_rejected = 0;
    MPI_Allreduce(&local_rejected,
                  &globally_rejected,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  MPI_COMM_WORLD);
    EXPECT_EQ(globally_rejected, 1);

    if (rank == 0) {
        RecordProperty("mpi_rank_count", size);
        RecordProperty("valid_snapshot_revision", local_revision);
        RecordProperty("duplicate_owner_rejected", globally_rejected);
        RecordProperty(
            "distributed_snapshot_stale_consumption_rejection_count",
            global_stale_rejection);
    }
}

TEST(GeneratedContactMeasureMPI,
     RotatedSessileCapContactIsContinuousAcrossCellAndOwnerBoundaries)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const auto evaluate = [&](GlobalIndex cell_count,
                              const RotatedSessileCap& cap,
                              std::uint64_t source_revision) {
        MovingContactState serial;
        if (rank == 0) {
            const GhostedQuadStripMeshAccess serial_mesh(
                /*rank=*/0,
                /*size=*/1,
                /*serial_owner_view=*/true,
                cell_count);
            serial = movingContactState(
                buildSessileCapContactDomain(
                    serial_mesh, cap, source_revision),
                cell_count,
                serial_mesh.extent(),
                source_revision);
        }
        MPI_Bcast(&serial.point_count,
                  1,
                  MPI_UINT64_T,
                  0,
                  MPI_COMM_WORLD);
        MPI_Bcast(&serial.counting_measure,
                  1,
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);
        MPI_Bcast(&serial.x_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&serial.min_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&serial.max_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&serial.min_parent_cell,
                  1,
                  MPI_LONG_LONG_INT,
                  0,
                  MPI_COMM_WORLD);
        MPI_Bcast(&serial.revision_mismatch_count,
                  1,
                  MPI_UINT64_T,
                  0,
                  MPI_COMM_WORLD);

        const GhostedQuadStripMeshAccess partitioned_mesh(
            rank,
            size,
            /*serial_owner_view=*/false,
            cell_count);
        const auto first = buildSessileCapContactDomain(
            partitioned_mesh, cap, source_revision);
        const auto repeat = buildSessileCapContactDomain(
            partitioned_mesh, cap, source_revision);
        const auto first_signature = contactRuleSignature(first);
        EXPECT_EQ(first_signature, contactRuleSignature(repeat));
        const auto next_revision = buildSessileCapContactDomain(
            partitioned_mesh, cap, source_revision + 1000u);
        if (!first_signature.empty()) {
            EXPECT_NE(first_signature, contactRuleSignature(next_revision));
        }
        auto first_source_mappings = globalContactSourceMappings(first);
        EXPECT_EQ(first_source_mappings,
                  globalContactSourceMappings(repeat));
        EXPECT_NE(first_source_mappings,
                  globalContactSourceMappings(next_revision));

        const auto local = movingContactState(
            first,
            cell_count,
            partitioned_mesh.extent(),
            source_revision);
        MovingContactState global;
        MPI_Allreduce(&local.point_count,
                      &global.point_count,
                      1,
                      MPI_UINT64_T,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&local.counting_measure,
                      &global.counting_measure,
                      1,
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&local.x_sum,
                      &global.x_sum,
                      1,
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&local.min_x,
                      &global.min_x,
                      1,
                      MPI_DOUBLE,
                      MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&local.max_x,
                      &global.max_x,
                      1,
                      MPI_DOUBLE,
                      MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&local.min_parent_cell,
                      &global.min_parent_cell,
                      1,
                      MPI_LONG_LONG_INT,
                      MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&local.revision_mismatch_count,
                      &global.revision_mismatch_count,
                      1,
                      MPI_UINT64_T,
                      MPI_SUM,
                      MPI_COMM_WORLD);

        EXPECT_EQ(serial.point_count, 2u);
        EXPECT_DOUBLE_EQ(serial.counting_measure, Real{2.0});
        EXPECT_EQ(serial.revision_mismatch_count, 0u);
        EXPECT_EQ(global.point_count, serial.point_count);
        EXPECT_DOUBLE_EQ(global.counting_measure,
                         serial.counting_measure);
        EXPECT_NEAR(global.x_sum, serial.x_sum, 1.0e-13);
        EXPECT_NEAR(global.min_x, serial.min_x, 1.0e-13);
        EXPECT_NEAR(global.max_x, serial.max_x, 1.0e-13);
        EXPECT_EQ(global.min_parent_cell, serial.min_parent_cell);
        EXPECT_EQ(global.revision_mismatch_count, 0u);
        global.source_mappings = std::move(first_source_mappings);
        return global;
    };

    RotatedSessileCap cap;
    cap.angle = 0.35;
    const auto coarse = evaluate(/*cell_count=*/8, cap, 101u);
    const auto medium = evaluate(/*cell_count=*/16, cap, 102u);
    const auto fine = evaluate(/*cell_count=*/32, cap, 103u);
    const auto root_error = [](const MovingContactState& state,
                               const RotatedSessileCap& evaluated_cap) {
        const auto roots = sessileCapWallRoots(evaluated_cap);
        return std::max(std::abs(state.min_x - roots[0]),
                        std::abs(state.max_x - roots[1]));
    };
    const Real coarse_error = root_error(coarse, cap);
    const Real medium_error = root_error(medium, cap);
    const Real fine_error = root_error(fine, cap);
    EXPECT_LT(medium_error, coarse_error);
    EXPECT_LT(fine_error, medium_error);
    EXPECT_LT(fine_error, Real{3.0e-3});

    cap.angle = -0.35;
    EXPECT_LT(root_error(evaluate(/*cell_count=*/32, cap, 104u), cap),
              Real{4.0e-3});
    cap.angle = 0.0;
    EXPECT_LT(root_error(evaluate(/*cell_count=*/32, cap, 105u), cap),
              Real{4.0e-3});

    RotatedSessileCap crossing_cap;
    crossing_cap.angle = 0.35;
    crossing_cap.center_x = 0.0;
    const Real left_root_offset = sessileCapWallRoots(crossing_cap)[0];
    crossing_cap.center_x = Real{2.0} - left_root_offset;
    constexpr Real crossing_delta = Real{1.0e-7};
    auto before_cap = crossing_cap;
    before_cap.center_x -= crossing_delta;
    auto after_cap = crossing_cap;
    after_cap.center_x += crossing_delta;
    const auto before = evaluate(/*cell_count=*/16, before_cap, 201u);
    const auto crossing = evaluate(/*cell_count=*/16, crossing_cap, 202u);
    const auto after = evaluate(/*cell_count=*/16, after_cap, 203u);
    EXPECT_EQ(before.min_parent_cell, 7);
    EXPECT_EQ(after.min_parent_cell, 8);
    EXPECT_LT(std::abs(crossing.min_x - before.min_x), Real{1.0e-5});
    EXPECT_LT(std::abs(after.min_x - crossing.min_x), Real{1.0e-5});
    EXPECT_LT(std::abs(crossing.max_x - before.max_x), Real{1.0e-5});
    EXPECT_LT(std::abs(after.max_x - crossing.max_x), Real{1.0e-5});

    ASSERT_EQ(before.source_mappings.size(), 2u);
    ASSERT_EQ(crossing.source_mappings.size(), 2u);
    ASSERT_EQ(after.source_mappings.size(), 2u);
    const auto& old_left_mapping = before.source_mappings.front();
    const auto& new_left_mapping = after.source_mappings.front();
    // Each row is local parent, global parent, contact id, source id,
    // source revision, degeneracy, and encoded owner rank.
    EXPECT_EQ(old_left_mapping[0], 7u);
    EXPECT_EQ(old_left_mapping[1], 2007u);
    EXPECT_EQ(old_left_mapping[4], 201u);
    EXPECT_EQ(new_left_mapping[0], 8u);
    EXPECT_EQ(new_left_mapping[1], 2008u);
    EXPECT_EQ(new_left_mapping[4], 203u);
    EXPECT_NE(old_left_mapping[2], new_left_mapping[2]);
    EXPECT_NE(old_left_mapping[3], new_left_mapping[3]);
    if (size > 1) {
        EXPECT_NE(old_left_mapping[6], new_left_mapping[6]);
    }
    for (const auto& mapping : crossing.source_mappings) {
        EXPECT_NE(mapping[2], 0u);
        EXPECT_NE(mapping[3], 0u);
        EXPECT_EQ(mapping[4], 202u);
        EXPECT_EQ(
            mapping[5],
            static_cast<std::uint64_t>(
                interfaces::
                    GeneratedInterfaceBoundaryIntersectionDegeneracy::None));
    }
    EXPECT_NE(crossing.source_mappings[0][3],
              crossing.source_mappings[1][3]);

    if (rank == 0) {
        RecordProperty("mpi_rank_count", size);
        RecordProperty("coarse_contact_position_error",
                       ::testing::PrintToString(coarse_error));
        RecordProperty("medium_contact_position_error",
                       ::testing::PrintToString(medium_error));
        RecordProperty("fine_contact_position_error",
                       ::testing::PrintToString(fine_error));
        RecordProperty("crossing_contact_counting_measure",
                       ::testing::PrintToString(crossing.counting_measure));
        RecordProperty("owner_boundary_left_parent",
                       before.min_parent_cell);
        RecordProperty("owner_boundary_right_parent",
                       after.min_parent_cell);
        RecordProperty("old_left_source_id", old_left_mapping[3]);
        RecordProperty("new_left_source_id", new_left_mapping[3]);
        RecordProperty("crossing_source_map_count",
                       crossing.source_mappings.size());
    }
}

TEST(GeneratedContactMeasureMPI,
     RankOneSnapshotTrajectoryIsRevisionBoundContinuousAndDeterministic)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
        GTEST_SKIP() << "snapshot trajectory qualification requires one rank";
    }

    std::uint64_t next_source_revision = 7001u;
    std::uint64_t state_count = 0u;
    std::uint64_t nonzero_snapshot_key_count = 0u;
    std::uint64_t rule_snapshot_mismatch_count = 0u;
    std::uint64_t rule_source_revision_mismatch_count = 0u;
    std::uint64_t orphan_contact_count = 0u;
    std::uint64_t missing_contact_count = 0u;
    std::uint64_t deterministic_remap_mismatch_count = 0u;
    std::uint64_t stale_consumption_rejection_count = 0u;
    Real maximum_counting_measure_error = 0.0;
    std::vector<std::uint64_t> snapshot_keys;

    const auto evaluate = [&](GlobalIndex cell_count,
                              const RotatedSessileCap& cap) {
        const auto source_revision = next_source_revision++;
        SCOPED_TRACE(::testing::Message()
                     << "cell_count=" << cell_count
                     << " source_revision=" << source_revision
                     << " cap_center_x=" << cap.center_x);
        GhostedQuadStripMeshAccess mesh(
            rank,
            size,
            /*serial_owner_view=*/false,
            cell_count);
        std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot>
            snapshot;
        std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot> repeat;
        try {
            snapshot = buildSessileCapSnapshot(
                mesh,
                cap,
                source_revision,
                mpiOwnershipCollective(rank, size),
                "rank_one_contact_trajectory");
            repeat = buildSessileCapSnapshot(
                mesh,
                cap,
                source_revision,
                mpiOwnershipCollective(rank, size),
                "rank_one_contact_trajectory");
        } catch (const std::exception& error) {
            ADD_FAILURE() << "snapshot construction failed: " << error.what();
            return SnapshotMovingContactState{};
        }
        EXPECT_TRUE(snapshot);
        EXPECT_TRUE(repeat);
        if (!snapshot || !repeat) {
            return SnapshotMovingContactState{};
        }

        const auto current =
            movingContactSnapshotState(*snapshot, source_revision);
        const auto repeated =
            movingContactSnapshotState(*repeat, source_revision);
        ++state_count;
        snapshot_keys.push_back(current.snapshot_revision_key);
        if (current.snapshot_revision_key != 0u) {
            ++nonzero_snapshot_key_count;
        }
        rule_snapshot_mismatch_count +=
            current.rule_snapshot_mismatch_count +
            repeated.rule_snapshot_mismatch_count;
        rule_source_revision_mismatch_count +=
            current.rule_source_revision_mismatch_count +
            repeated.rule_source_revision_mismatch_count;
        orphan_contact_count +=
            current.orphan_contact_count + repeated.orphan_contact_count;
        missing_contact_count +=
            current.missing_contact_count + repeated.missing_contact_count;
        assembly::CutIntegrationContext live_context;
        live_context.addFreeSurfaceGeometrySnapshot(snapshot);
        EXPECT_NO_THROW(
            live_context.assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh));
        mesh.advanceGeometryRevision();
        try {
            live_context.assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh);
            ADD_FAILURE()
                << "stale sessile-cap snapshot consumption was accepted";
        } catch (const std::invalid_argument&) {
            ++stale_consumption_rejection_count;
        }
        maximum_counting_measure_error = std::max(
            maximum_counting_measure_error,
            std::abs(current.contact.counting_measure - Real{2.0}));

        const bool remap_mismatch =
            current.snapshot_revision_key !=
                repeated.snapshot_revision_key ||
            snapshotContactRuleSignature(*snapshot) !=
                snapshotContactRuleSignature(*repeat) ||
            current.contact.source_mappings !=
                repeated.contact.source_mappings ||
            current.contact.point_count != repeated.contact.point_count ||
            current.contact.counting_measure !=
                repeated.contact.counting_measure ||
            current.contact.min_x != repeated.contact.min_x ||
            current.contact.max_x != repeated.contact.max_x;
        if (remap_mismatch) {
            ++deterministic_remap_mismatch_count;
        }

        EXPECT_EQ(current.contact.point_count, 2u);
        EXPECT_DOUBLE_EQ(current.contact.counting_measure, Real{2.0});
        EXPECT_EQ(current.contact.source_mappings.size(), 2u);
        EXPECT_EQ(current.rule_snapshot_mismatch_count, 0u);
        EXPECT_EQ(current.rule_source_revision_mismatch_count, 0u);
        EXPECT_EQ(current.orphan_contact_count, 0u);
        EXPECT_EQ(current.missing_contact_count, 0u);
        return current;
    };

    RotatedSessileCap refinement_cap;
    refinement_cap.angle = Real{0.35};
    const auto coarse = evaluate(/*cell_count=*/8, refinement_cap);
    const auto medium = evaluate(/*cell_count=*/16, refinement_cap);
    const auto fine = evaluate(/*cell_count=*/32, refinement_cap);
    const auto position_error = [](const SnapshotMovingContactState& state,
                                   const RotatedSessileCap& cap) {
        const auto roots = sessileCapWallRoots(cap);
        return std::max(std::abs(state.contact.min_x - roots[0]),
                        std::abs(state.contact.max_x - roots[1]));
    };
    const Real coarse_error = position_error(coarse, refinement_cap);
    const Real medium_error = position_error(medium, refinement_cap);
    const Real fine_error = position_error(fine, refinement_cap);
    EXPECT_GT(coarse_error, Real{0.0});
    EXPECT_GT(medium_error, Real{0.0});
    EXPECT_GT(fine_error, Real{0.0});
    EXPECT_LT(medium_error, coarse_error);
    EXPECT_LT(fine_error, medium_error);
    EXPECT_LT(fine_error, Real{3.0e-3});
    const Real coarse_to_medium_order =
        std::log(coarse_error / medium_error) / std::log(Real{2.0});
    const Real medium_to_fine_order =
        std::log(medium_error / fine_error) / std::log(Real{2.0});
    const Real observed_position_order =
        std::min(coarse_to_medium_order, medium_to_fine_order);
    EXPECT_GT(observed_position_order, Real{0.0});

    RotatedSessileCap crossing_cap;
    crossing_cap.angle = Real{0.35};
    crossing_cap.center_x = 0.0;
    const Real left_root_offset = sessileCapWallRoots(crossing_cap)[0];
    crossing_cap.center_x = Real{2.0} - left_root_offset;
    constexpr std::array<Real, 9> trajectory_offsets{{
        Real{-4.0e-7},
        Real{-3.0e-7},
        Real{-2.0e-7},
        Real{-1.0e-7},
        Real{0.0},
        Real{1.0e-7},
        Real{2.0e-7},
        Real{3.0e-7},
        Real{4.0e-7},
    }};
    Real maximum_crossing_position_jump = 0.0;
    std::uint64_t cell_boundary_crossing_count = 0u;
    std::uint64_t source_id_change_count = 0u;
    SnapshotMovingContactState previous;
    bool have_previous = false;
    for (const auto offset : trajectory_offsets) {
        auto state_cap = crossing_cap;
        state_cap.center_x += offset;
        const auto current = evaluate(/*cell_count=*/16, state_cap);
        if (have_previous) {
            maximum_crossing_position_jump = std::max(
                maximum_crossing_position_jump,
                std::max(std::abs(current.contact.min_x -
                                  previous.contact.min_x),
                         std::abs(current.contact.max_x -
                                  previous.contact.max_x)));
            if (current.contact.min_parent_cell !=
                previous.contact.min_parent_cell) {
                ++cell_boundary_crossing_count;
                if (current.contact.source_mappings.size() == 2u &&
                    previous.contact.source_mappings.size() == 2u &&
                    current.contact.source_mappings.front()[3] !=
                        previous.contact.source_mappings.front()[3]) {
                    ++source_id_change_count;
                }
            }
        }
        previous = current;
        have_previous = true;
    }

    constexpr std::uint64_t frozen_state_count = 12u;
    std::sort(snapshot_keys.begin(), snapshot_keys.end());
    const auto distinct_end =
        std::unique(snapshot_keys.begin(), snapshot_keys.end());
    const auto distinct_snapshot_key_count =
        static_cast<std::uint64_t>(
            std::distance(snapshot_keys.begin(), distinct_end));
    EXPECT_EQ(state_count, frozen_state_count);
    EXPECT_EQ(nonzero_snapshot_key_count, frozen_state_count);
    EXPECT_EQ(distinct_snapshot_key_count, frozen_state_count);
    EXPECT_EQ(rule_snapshot_mismatch_count, 0u);
    EXPECT_EQ(rule_source_revision_mismatch_count, 0u);
    EXPECT_EQ(orphan_contact_count, 0u);
    EXPECT_EQ(missing_contact_count, 0u);
    EXPECT_LE(maximum_counting_measure_error, Real{1.0e-12});
    EXPECT_LT(maximum_crossing_position_jump, Real{1.0e-5});
    EXPECT_GE(cell_boundary_crossing_count, 1u);
    EXPECT_EQ(deterministic_remap_mismatch_count, 0u);
    EXPECT_GE(source_id_change_count, 1u);
    EXPECT_EQ(stale_consumption_rejection_count, frozen_state_count);

    RecordProperty("trajectory_state_count", state_count);
    RecordProperty("trajectory_nonzero_snapshot_key_count",
                   nonzero_snapshot_key_count);
    RecordProperty("trajectory_distinct_snapshot_key_count",
                   distinct_snapshot_key_count);
    RecordProperty("trajectory_rule_snapshot_mismatch_count",
                   rule_snapshot_mismatch_count);
    RecordProperty("trajectory_rule_source_revision_mismatch_count",
                   rule_source_revision_mismatch_count);
    RecordProperty("trajectory_orphan_contact_count", orphan_contact_count);
    RecordProperty("trajectory_missing_contact_count", missing_contact_count);
    RecordProperty("trajectory_max_counting_measure_error",
                   ::testing::PrintToString(maximum_counting_measure_error));
    RecordProperty("trajectory_coarse_position_error",
                   ::testing::PrintToString(coarse_error));
    RecordProperty("trajectory_medium_position_error",
                   ::testing::PrintToString(medium_error));
    RecordProperty("trajectory_fine_position_error",
                   ::testing::PrintToString(fine_error));
    RecordProperty("trajectory_observed_position_order",
                   ::testing::PrintToString(observed_position_order));
    RecordProperty("trajectory_max_crossing_position_jump",
                   ::testing::PrintToString(maximum_crossing_position_jump));
    RecordProperty("trajectory_cell_boundary_crossing_count",
                   cell_boundary_crossing_count);
    RecordProperty("trajectory_deterministic_remap_mismatch_count",
                   deterministic_remap_mismatch_count);
    RecordProperty("trajectory_source_id_change_count",
                   source_id_change_count);
    RecordProperty("snapshot_trajectory_stale_consumption_rejection_count",
                   stale_consumption_rejection_count);
}
