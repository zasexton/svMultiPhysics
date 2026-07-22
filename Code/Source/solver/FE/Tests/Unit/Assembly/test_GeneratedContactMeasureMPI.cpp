/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Assembly/Assembler.h"
#include "Geometry/CutQuadrature.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace svmp::FE;

namespace {

constexpr GlobalIndex kCellCount = 4;
constexpr int kWallMarker = 41;
constexpr int kInterfaceMarker = 3041;

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
    GhostedQuadStripMeshAccess(int rank, int size, bool serial_owner_view)
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
        return (kCellCount + 1) * 2;
    }

    [[nodiscard]] GlobalIndex numBoundaryFaces() const override
    {
        return kCellCount;
    }

    [[nodiscard]] GlobalIndex numInteriorFaces() const override
    {
        return kCellCount - 1;
    }

    [[nodiscard]] int dimension() const override { return 2; }

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
        return ElementType::Quad4;
    }

    void getCellNodes(GlobalIndex cell_id,
                      std::vector<GlobalIndex>& nodes) const override
    {
        const auto node = [](GlobalIndex i, GlobalIndex j) {
            return i + (kCellCount + 1) * j;
        };
        nodes = {node(cell_id, 0),
                 node(cell_id + 1, 0),
                 node(cell_id + 1, 1),
                 node(cell_id, 1)};
    }

    [[nodiscard]] std::array<Real, 3>
    getNodeCoordinates(GlobalIndex node_id) const override
    {
        const GlobalIndex nx = kCellCount + 1;
        const GlobalIndex i = node_id % nx;
        const GlobalIndex j = node_id / nx;
        return {{static_cast<Real>(i), static_cast<Real>(j), Real{0.0}}};
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
        return face_id >= 300 && face_id < 300 + kCellCount ? kWallMarker : -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex face_id) const override
    {
        const GlobalIndex left = face_id - 400;
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
        // All boundary edges are visible on all ranks, including ghost data.
        // Repeating each callback verifies owner filtering and face-id
        // deduplication for the zero-dimensional contact domain.
        for (GlobalIndex cell = 0; cell < kCellCount; ++cell) {
            callback(300 + cell, cell);
            callback(300 + cell, cell);
        }
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback)
        const override
    {
        for (GlobalIndex face = 0; face < kCellCount - 1; ++face) {
            callback(400 + face, face, face + 1);
        }
    }

private:
    int rank_{0};
    int size_{1};
    bool serial_owner_view_{false};
};

[[nodiscard]] interfaces::GeneratedInterfaceBoundaryIntersectionRequest
contactRequest()
{
    interfaces::GeneratedInterfaceBoundaryIntersectionRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromField(
        /*field_id=*/17,
        /*layout_revision=*/2,
        /*value_revision=*/5);
    request.generated_domain_id = "mpi_contact_measure";
    request.interface_marker = kInterfaceMarker;
    request.boundary_marker = kWallMarker;
    request.isovalue = 0.0;
    request.quadrature_order = 3;
    request.frame = geometry::CutGeometryFrame::Reference;
    request.source_value_revision = 5u;
    return request;
}

[[nodiscard]] interfaces::GeneratedInterfaceBoundaryIntersectionRequest
pointContactRequest()
{
    auto request = contactRequest();
    request.generated_domain_id = "mpi_contact_point_measure";
    return request;
}

[[nodiscard]] interfaces::CutInterfaceDomainRequest interfaceRequest()
{
    interfaces::CutInterfaceDomainRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromField(
        /*field_id=*/17,
        /*layout_revision=*/2,
        /*value_revision=*/5);
    request.interface_marker = kInterfaceMarker;
    request.isovalue = 0.0;
    request.quadrature_order = 3;
    request.implicit_geometry_mode = "LinearCorner";
    request.implicit_quadrature_backend = "LinearCorner";
    request.implicit_fallback_status = "None";
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
        RecordProperty("global_contact_length", global.physical_length);
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
                       global.counting_measure);
        RecordProperty("global_contact_point_fragment_count",
                       global.fragment_count);
        RecordProperty("global_contact_point_quadrature_point_count",
                       global.quadrature_point_count);
    }
}
