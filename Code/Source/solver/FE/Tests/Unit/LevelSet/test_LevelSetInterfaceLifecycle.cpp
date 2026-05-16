#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "LevelSet/LevelSetCellEvaluator.h"
#include "LevelSet/LevelSetImplicitCutQuadratureBackend.h"

#include "Assembly/Assembler.h"
#include "Assembly/CutDomainAssembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Interfaces/LevelSetInterfaceGeometryWriter.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class SingleTetraMeshAccess final : public FE::assembly::IMeshAccess {
public:
    SingleTetraMeshAccess()
        : SingleTetraMeshAccess({
              std::array<FE::Real, 3>{0.0, 0.0, 0.0},
              std::array<FE::Real, 3>{1.0, 0.0, 0.0},
              std::array<FE::Real, 3>{0.0, 1.0, 0.0},
              std::array<FE::Real, 3>{0.0, 0.0, 1.0},
          })
    {
    }

    explicit SingleTetraMeshAccess(std::vector<std::array<FE::Real, 3>> nodes)
        : nodes_(std::move(nodes))
    {
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Tetra4;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        FE::GlobalIndex /*cell_id*/,
        std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        coords = nodes_;
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex /*face_id*/,
        FE::GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex /*face_id*/) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>
            /*callback*/) const override
    {
    }

private:
    std::vector<std::array<FE::Real, 3>> nodes_{};
    std::array<FE::GlobalIndex, 4> cell_{};
};

class SingleTetra10GeometryMeshAccess final : public FE::assembly::IMeshAccess {
public:
    SingleTetra10GeometryMeshAccess()
        : nodes_{
              std::array<FE::Real, 3>{0.0, 0.0, 0.0},
              std::array<FE::Real, 3>{1.0, 0.0, 0.0},
              std::array<FE::Real, 3>{0.0, 1.0, 0.0},
              std::array<FE::Real, 3>{0.0, 0.0, 1.0},
              std::array<FE::Real, 3>{0.5, 0.0, 0.0},
              std::array<FE::Real, 3>{0.5, 0.5, 0.0},
              std::array<FE::Real, 3>{0.0, 0.5, 0.0},
              std::array<FE::Real, 3>{0.0, 0.0, 0.5},
              std::array<FE::Real, 3>{0.5, 0.0, 0.5},
              std::array<FE::Real, 3>{0.0, 0.5, 0.5},
          }
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 4; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Tetra10;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        FE::GlobalIndex /*cell_id*/,
        std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        coords.assign(nodes_.begin(), nodes_.end());
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex /*face_id*/,
        FE::GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex /*face_id*/) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>
            /*callback*/) const override
    {
    }

private:
    std::array<std::array<FE::Real, 3>, 10> nodes_{};
};

class SingleQuadMeshAccess final : public FE::assembly::IMeshAccess {
public:
    explicit SingleQuadMeshAccess(FE::ElementType type = FE::ElementType::Quad4)
        : type_(type)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 4; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return type_;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        if (type_ == FE::ElementType::Quad4) {
            nodes = {0, 1, 2, 3};
        } else {
            nodes = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        }
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        FE::GlobalIndex /*cell_id*/,
        std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        if (type_ == FE::ElementType::Quad4) {
            coords.assign(nodes_.begin(), nodes_.begin() + 4);
        } else {
            coords.assign(nodes_.begin(), nodes_.end());
        }
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex /*face_id*/,
        FE::GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex /*face_id*/) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>
            /*callback*/) const override
    {
    }

private:
    FE::ElementType type_{FE::ElementType::Quad4};
    std::array<std::array<FE::Real, 3>, 9> nodes_{{
        {{-1.0, -1.0, 0.0}},
        {{1.0, -1.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{-1.0, 1.0, 0.0}},
        {{0.0, -1.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{-1.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
    }};
};

class SingleTriangleMeshAccess final : public FE::assembly::IMeshAccess {
public:
    explicit SingleTriangleMeshAccess(
        FE::ElementType type = FE::ElementType::Triangle3)
        : type_(type)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 3; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return type_;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        if (type_ == FE::ElementType::Triangle3) {
            nodes = {0, 1, 2};
        } else {
            nodes = {0, 1, 2, 3, 4, 5};
        }
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        FE::GlobalIndex /*cell_id*/,
        std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        if (type_ == FE::ElementType::Triangle3) {
            coords.assign(nodes_.begin(), nodes_.begin() + 3);
        } else {
            coords.assign(nodes_.begin(), nodes_.end());
        }
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex /*face_id*/,
        FE::GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex /*face_id*/) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>
            /*callback*/) const override
    {
    }

private:
    FE::ElementType type_{FE::ElementType::Triangle3};
    std::array<std::array<FE::Real, 3>, 6> nodes_{{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.5, 0.0, 0.0}},
        {{0.5, 0.5, 0.0}},
        {{0.0, 0.5, 0.0}},
    }};
};

[[nodiscard]] FE::systems::SetupInputs makeSingleTetraSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 3;

    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

[[nodiscard]] FE::systems::SetupInputs makeSingleQuadSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

[[nodiscard]] FE::systems::SetupInputs makeSingleTriangleSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 3;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 3};
    topo.cell2vertex_data = {0, 1, 2};
    topo.vertex_gids = {0, 1, 2};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

void setFieldComponentValue(std::vector<FE::Real>& solution,
                            const FE::systems::FESystem& system,
                            FE::FieldId field,
                            FE::GlobalIndex vertex,
                            FE::Real value)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto offset = system.fieldDofOffset(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("setFieldComponentValue: field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (dofs.empty()) {
        throw std::runtime_error("setFieldComponentValue: missing vertex DOF");
    }
    const auto index = static_cast<std::size_t>(dofs.front() + offset);
    if (index >= solution.size()) {
        throw std::runtime_error("setFieldComponentValue: DOF index is out of range");
    }
    solution[index] = value;
}

FE::Real integrateVolumeMoment(
    const FE::interfaces::LevelSetInterfaceDomain& domain,
    FE::geometry::CutIntegrationSide side,
    const std::function<FE::Real(const std::array<FE::Real, 3>&)>& moment)
{
    FE::Real value = 0.0;
    for (const auto& rule : domain.volumeQuadratureRules()) {
        if (rule.side != side) {
            continue;
        }
        for (const auto& point : rule.points) {
            value += moment(point.point) * point.weight;
        }
    }
    return value;
}

FE::Real integrateInterfaceMoment(
    const FE::interfaces::LevelSetInterfaceDomain& domain,
    const std::function<FE::Real(const std::array<FE::Real, 3>&)>& moment)
{
    FE::Real value = 0.0;
    for (const auto& rule : domain.interfaceQuadratureRules()) {
        for (const auto& point : rule.points) {
            value += moment(point.point) * point.weight;
        }
    }
    return value;
}

class CutMeasureAssemblyKernel final : public FE::assembly::AssemblyKernel {
public:
    [[nodiscard]] FE::assembly::RequiredData getRequiredData() const override
    {
        return FE::assembly::RequiredData::IntegrationWeights;
    }

    void computeCell(const FE::assembly::AssemblyContext& ctx,
                     FE::assembly::KernelOutput& output) override
    {
        integrate(ctx, output);
    }

    void computeBoundaryFace(const FE::assembly::AssemblyContext& ctx,
                             int /*boundary_marker*/,
                             FE::assembly::KernelOutput& output) override
    {
        integrate(ctx, output);
    }

    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }

private:
    static void integrate(const FE::assembly::AssemblyContext& ctx,
                          FE::assembly::KernelOutput& output)
    {
        output.reserve(/*n_test=*/1, /*n_trial=*/0, /*need_matrix=*/false,
                       /*need_vector=*/true);
        for (FE::LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            output.vectorEntry(0) += ctx.integrationWeight(q);
        }
    }
};

void populateMeasureAssemblyContext(
    const FE::geometry::CutQuadratureRule& rule,
    FE::assembly::AssemblyContext& ctx)
{
    std::vector<std::array<FE::Real, 3>> points;
    std::vector<std::array<FE::Real, 3>> normals;
    std::vector<FE::Real> weights;
    points.reserve(rule.points.size());
    normals.reserve(rule.points.size());
    weights.reserve(rule.points.size());
    for (const auto& point : rule.points) {
        points.push_back(point.point);
        normals.push_back(point.normal);
        weights.push_back(point.weight);
    }
    ctx.setQuadratureData(points, weights);
    ctx.setPhysicalPoints(points);
    ctx.setIntegrationWeights(weights);
    ctx.setNormals(normals);
}

} // namespace

TEST(LevelSetInterfaceLifecycle, BuildsDomainFromScalarField)
{
    constexpr int interface_marker = 73;
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.tolerance = 1.0e-12;
    options.interface_quadrature_order = 0;
    options.volume_quadrature_order = 1;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.interface_marker, interface_marker);
    EXPECT_EQ(result.value_revision, 1u);
    EXPECT_EQ(lifecycle.valueRevision(), 1u);
    EXPECT_EQ(result.cell_count, 1u);
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.max_cell_node_count, 4u);
    EXPECT_EQ(result.max_corner_node_count, 4u);
    EXPECT_EQ(result.domain.marker(), interface_marker);
    EXPECT_EQ(result.domain.request().source.field_id, phi);
    EXPECT_EQ(result.domain.request().source.layout_revision, 17u);
    EXPECT_EQ(result.domain.request().source.value_revision, result.value_revision);
    EXPECT_EQ(result.domain.request().mesh_geometry_revision, 7u);
    EXPECT_EQ(result.domain.request().mesh_topology_revision, 11u);
    EXPECT_EQ(result.domain.request().ownership_revision, 13u);
    EXPECT_NE(result.domain.request().quadrature_policy_key, 0u);
    EXPECT_EQ(result.domain.request().implicit_geometry_mode, "LinearCorner");
    EXPECT_EQ(result.domain.request().implicit_quadrature_backend, "LinearCorner");
    EXPECT_EQ(result.domain.request().implicit_fallback_policy, "Fail");
    EXPECT_EQ(result.domain.request().resolvedInterfaceQuadratureOrder(), 0);
    EXPECT_EQ(result.domain.request().resolvedVolumeQuadratureOrder(), 1);
    EXPECT_EQ(result.summary.interface_marker, interface_marker);
    EXPECT_EQ(result.summary.active_fragment_count, 1u);
    EXPECT_EQ(result.summary.active_volume_region_count, 2u);
    EXPECT_EQ(result.summary.quadrature_point_count, 1u);
    EXPECT_GT(result.summary.measure, 0.0);
    EXPECT_GT(result.summary.negative_volume_measure, 0.0);
    EXPECT_GT(result.summary.positive_volume_measure, 0.0);
    ASSERT_EQ(result.domain.fragments().size(), 1u);
    EXPECT_EQ(result.domain.fragments().front().interface_marker, interface_marker);
    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_EQ(interface_rules.size(), 1u);
    EXPECT_EQ(interface_rules.front().exact_polynomial_order, 0);
    EXPECT_EQ(interface_rules.front().provenance.predicate_policy_key,
              result.domain.request().quadrature_policy_key);
    EXPECT_EQ(interface_rules.front().provenance.implicit_quadrature_backend,
              "LinearCorner");
    EXPECT_EQ(interface_rules.front().provenance.requested_quadrature_order, 0);
    EXPECT_EQ(interface_rules.front().provenance.achieved_quadrature_order, 0);
    ASSERT_EQ(result.domain.volumeRegions().size(), 2u);
    EXPECT_EQ(result.domain.volumeRegions().front().interface_marker, interface_marker);
    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_EQ(volume_rules.size(), 2u);
    EXPECT_EQ(volume_rules.front().exact_polynomial_order, 1);
    EXPECT_EQ(volume_rules.front().provenance.predicate_policy_key,
              result.domain.request().quadrature_policy_key);
    EXPECT_EQ(volume_rules.front().provenance.implicit_geometry_mode,
              "LinearCorner");
    EXPECT_EQ(volume_rules.front().provenance.requested_quadrature_order, 1);
    EXPECT_EQ(volume_rules.front().provenance.achieved_quadrature_order, 1);
}

TEST(LevelSetInterfaceLifecycle, QuadraturePolicyKeyChangesWithBackendOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions base_options{};
    base_options.level_set_field_name = "phi";
    base_options.requested_interface_marker = 74;
    base_options.domain_id = "water-air";
    base_options.interface_quadrature_order = 0;
    base_options.volume_quadrature_order = 1;

    auto changed_options = base_options;
    changed_options.implicit_cut_root_tolerance = 1.0e-8;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto base = lifecycle.build(system, base_options, solution);
    const auto changed = lifecycle.build(system, changed_options, solution);

    ASSERT_TRUE(base.success) << base.diagnostic;
    ASSERT_TRUE(changed.success) << changed.diagnostic;
    EXPECT_NE(base.domain.request().quadrature_policy_key,
              changed.domain.request().quadrature_policy_key);
}

TEST(LevelSetCellEvaluator, P1ReproducesCornerValuesAndReferenceGradient)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    const auto at_origin = evaluator.evaluate(0, {{0.0, 0.0, 0.0}});
    EXPECT_EQ(at_origin.interpolation_order, 1);
    EXPECT_EQ(at_origin.implicit_geometry_order, 1);
    EXPECT_NEAR(at_origin.value, -0.5, 1.0e-12);
    EXPECT_NEAR(at_origin.reference_gradient[0], 1.0, 1.0e-12);
    EXPECT_NEAR(at_origin.reference_gradient[1], 1.0, 1.0e-12);
    EXPECT_NEAR(at_origin.reference_gradient[2], 1.0, 1.0e-12);

    const auto at_vertex_one = evaluator.evaluate(0, {{1.0, 0.0, 0.0}});
    EXPECT_NEAR(at_vertex_one.value, 0.5, 1.0e-12);
}

TEST(LevelSetCellEvaluator, P2RespondsToEdgeDofsAtInteriorNodes)
{
    const auto mesh = std::make_shared<SingleTetra10GeometryMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 10u);

    const auto offset = system.fieldDofOffset(phi);
    solution[static_cast<std::size_t>(offset + cell_dofs[4])] = 2.0;

    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    const auto at_edge_node = evaluator.evaluate(0, {{0.5, 0.0, 0.0}});
    EXPECT_EQ(at_edge_node.interpolation_order, 2);
    EXPECT_EQ(at_edge_node.implicit_geometry_order, 2);
    EXPECT_NEAR(at_edge_node.value, 2.0, 1.0e-12);

    const auto at_vertex = evaluator.evaluate(0, {{0.0, 0.0, 0.0}});
    EXPECT_NEAR(at_vertex.value, 0.0, 1.0e-12);
}

TEST(LevelSetCellEvaluator, ReferenceGradientMatchesFiniteDifference)
{
    const auto mesh = std::make_shared<SingleTetra10GeometryMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 10u);

    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            FE::Real(0.1) * static_cast<FE::Real>(i + 1u);
    }

    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    const std::array<FE::Real, 3> xi{{0.2, 0.2, 0.2}};
    const auto evaluation = evaluator.evaluate(0, xi);
    constexpr FE::Real eps = 1.0e-6;
    for (std::size_t d = 0; d < 3u; ++d) {
        auto plus = xi;
        auto minus = xi;
        plus[d] += eps;
        minus[d] -= eps;
        const auto value_plus = evaluator.evaluate(0, plus).value;
        const auto value_minus = evaluator.evaluate(0, minus).value;
        const auto finite_difference =
            (value_plus - value_minus) / (FE::Real{2.0} * eps);
        EXPECT_NEAR(evaluation.reference_gradient[d],
                    finite_difference,
                    1.0e-8);
    }
}

TEST(LevelSetCellEvaluator, UsesFieldOffsetInFullStateVector)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto velocity_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/3);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    (void)system.addField(FE::systems::FieldSpec{
        .name = "Velocity",
        .space = velocity_space,
        .components = 3,
    });
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 1000.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);
    const auto at_origin = evaluator.evaluate(0, {{0.0, 0.0, 0.0}});
    EXPECT_NEAR(at_origin.value, -0.5, 1.0e-12);
    EXPECT_NEAR(at_origin.reference_gradient[0], 1.0, 1.0e-12);
    EXPECT_NEAR(at_origin.reference_gradient[1], 1.0, 1.0e-12);
    EXPECT_NEAR(at_origin.reference_gradient[2], 1.0, 1.0e-12);
}

TEST(LevelSetCellEvaluator, ReportsCellEvaluationFailure)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    try {
        (void)evaluator.evaluate(99, {{0.0, 0.0, 0.0}});
        FAIL() << "Expected invalid cell evaluation to fail";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("could not evaluate cell 99"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, RejectsHighOrderImplicitModeOnUnsupportedTetra)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    try {
        (void)lifecycle.build(system, options, solution);
        FAIL() << "Expected unsupported high-order tetrahedron backend to be rejected";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("quadrilateral"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, HighOrderImplicitDoesNotSilentlyUseLinearFallback)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::LinearCorner;
    options.implicit_cut_fallback_policy =
        level_set::ImplicitCutFallbackPolicy::LinearCorner;
    options.allow_corner_linearized_geometry = true;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    try {
        (void)lifecycle.build(system, options, std::span<const FE::Real>{});
        FAIL() << "Expected high-order implicit geometry to reject linear fallback";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("high-order implicit"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, RejectsNonlinearBackendForLinearCornerMode)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::LinearCorner;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    try {
        (void)lifecycle.build(system, options, std::span<const FE::Real>{});
        FAIL() << "Expected nonlinear backend to be rejected for linear geometry";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("LinearCorner"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, LinearBackendDriverReportsSupportAndOrders)
{
    const auto& backend =
        level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::LinearCorner);
    EXPECT_EQ(backend.kind(),
              level_set::ImplicitCutQuadratureBackend::LinearCorner);
    EXPECT_STREQ(backend.name(), "LinearCorner");
    EXPECT_TRUE(backend.supports(2, FE::ElementType::Quad4));
    EXPECT_TRUE(backend.supports(3, FE::ElementType::Tetra4));
    EXPECT_FALSE(backend.supports(3, FE::ElementType::Hex8));

    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source = FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
        "test-level-set", 0, 1);
    request.interface_marker = 10;
    request.quadrature_order = 3;
    request.interface_quadrature_order = 2;
    request.volume_quadrature_order = 3;

    EXPECT_EQ(backend.achievedInterfaceQuadratureOrder(request), 1);
    EXPECT_EQ(backend.achievedVolumeQuadratureOrder(request), 2);
}

TEST(LevelSetInterfaceLifecycle, BackendCapabilityReportsMilestoneContract)
{
    const auto linear_quad =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::LinearCorner,
            2,
            FE::ElementType::Quad4);
    EXPECT_TRUE(linear_quad.implemented);
    EXPECT_TRUE(linear_quad.supports_element_type);
    EXPECT_FALSE(linear_quad.supports_high_order_geometry);
    EXPECT_TRUE(linear_quad.requires_scalar_h1_c0_level_set);
    EXPECT_EQ(linear_quad.minimum_level_set_order, 1);
    EXPECT_EQ(linear_quad.validation_level_set_order, 1);
    EXPECT_EQ(linear_quad.maximum_reported_interface_order, 1);
    EXPECT_EQ(linear_quad.maximum_reported_volume_order, 2);
    EXPECT_TRUE(linear_quad.returns_reference_frame_rules);
    EXPECT_TRUE(linear_quad.requires_positive_volume_weights);
    EXPECT_TRUE(linear_quad.requires_deterministic_rule_order);
    EXPECT_TRUE(linear_quad.prunes_tiny_slivers_in_context);
    EXPECT_TRUE(linear_quad.near_tangent_requires_diagnostic);
    EXPECT_GT(linear_quad.tiny_sliver_volume_fraction, 0.0);

    const auto saye_quad =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
            2,
            FE::ElementType::Quad9);
    EXPECT_TRUE(saye_quad.implemented);
    EXPECT_TRUE(saye_quad.supports_element_type);
    EXPECT_TRUE(saye_quad.supports_high_order_geometry);
    EXPECT_EQ(saye_quad.minimum_level_set_order, 1);
    EXPECT_EQ(saye_quad.validation_level_set_order, 3);
    EXPECT_EQ(saye_quad.maximum_reported_interface_order, 1);
    EXPECT_EQ(saye_quad.maximum_reported_volume_order, 2);
    EXPECT_TRUE(saye_quad.requires_scalar_h1_c0_level_set);

    const auto saye_tri =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
            2,
            FE::ElementType::Triangle6);
    EXPECT_FALSE(saye_tri.supports_element_type);

    const auto subcell_tri =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
            2,
            FE::ElementType::Triangle6);
    EXPECT_TRUE(subcell_tri.implemented);
    EXPECT_TRUE(subcell_tri.supports_element_type);
    EXPECT_TRUE(subcell_tri.supports_high_order_geometry);
    EXPECT_EQ(subcell_tri.minimum_level_set_order, 1);
    EXPECT_EQ(subcell_tri.validation_level_set_order, 3);
    EXPECT_EQ(subcell_tri.maximum_reported_interface_order, 1);
    EXPECT_EQ(subcell_tri.maximum_reported_volume_order, 2);
    EXPECT_TRUE(subcell_tri.requires_scalar_h1_c0_level_set);

    const auto subcell_quad =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::HighOrderSubcell,
            2,
            FE::ElementType::Quad9);
    EXPECT_FALSE(subcell_quad.supports_element_type);

    const auto saye_hex =
        level_set::implicitCutQuadratureBackendCapability(
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle,
            3,
            FE::ElementType::Hex27);
    EXPECT_FALSE(saye_hex.supports_element_type);
}

TEST(LevelSetInterfaceLifecycle, BackendDiagnosticStatusNamesAreStable)
{
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::ExactNoCut),
        "ExactNoCut");
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::Cut),
        "Cut");
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::Tangent),
        "Tangent");
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::Degenerate),
        "Degenerate");
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::Fallback),
        "Fallback");
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::Unsupported),
        "Unsupported");
    EXPECT_STREQ(
        level_set::implicitCutQuadratureDiagnosticStatusName(
            level_set::ImplicitCutQuadratureDiagnosticStatus::Failed),
        "Failed");
}

TEST(LevelSetInterfaceLifecycle, LinearBackendOutputPassesCommonValidation)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source = FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
        "validation-level-set", 0, 1);
    request.interface_marker = 99;
    request.tolerance = 1.0e-12;
    request.quadrature_order = 2;
    request.interface_quadrature_order = 1;
    request.volume_quadrature_order = 2;

    FE::interfaces::LevelSetCellCutInput input{};
    input.parent_cell = 5;
    input.element_type = FE::ElementType::Tetra4;
    input.node_coordinates = {
        std::array<FE::Real, 3>{0.0, 0.0, 0.0},
        std::array<FE::Real, 3>{1.0, 0.0, 0.0},
        std::array<FE::Real, 3>{0.0, 1.0, 0.0},
        std::array<FE::Real, 3>{0.0, 0.0, 1.0},
    };
    input.level_set_values = {-0.25, 0.75, 0.75, 0.75};

    level_set::ImplicitCutQuadratureBackendCellInput backend_input{};
    backend_input.linearized_input = input;
    backend_input.evaluator = &evaluator;
    backend_input.isovalue = request.isovalue;
    backend_input.reference_min = {{0.0, 0.0, 0.0}};
    backend_input.reference_max = {{1.0, 1.0, 1.0}};

    const auto& backend =
        level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::LinearCorner);
    const auto result = backend.cut(3, request, backend_input);
    const auto validation =
        level_set::validateImplicitCutQuadratureBackendCellResult(
            request, backend_input, result);
    EXPECT_TRUE(validation.ok) << validation.diagnostic;
    EXPECT_EQ(validation.status,
              level_set::ImplicitCutQuadratureDiagnosticStatus::Cut);

    FE::Real parent_measure = 0.0;
    FE::Real side_measure = 0.0;
    for (const auto& region : result.cut.volume_regions) {
        parent_measure = std::max(parent_measure, region.parent_measure);
        side_measure += region.measure;
    }
    EXPECT_NEAR(side_measure, parent_measure, 1.0e-12);
}

TEST(LevelSetInterfaceLifecycle, InvalidBackendOutputIsRejected)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);

    FE::interfaces::CutInterfaceDomainRequest request{};
    request.source = FE::interfaces::LevelSetInterfaceSource::fromEvaluator(
        "validation-level-set", 0, 1);
    request.interface_marker = 12;
    request.tolerance = 1.0e-12;

    FE::interfaces::LevelSetCellCutInput input{};
    input.parent_cell = 2;
    input.element_type = FE::ElementType::Tetra4;
    level_set::ImplicitCutQuadratureBackendCellInput backend_input{};
    backend_input.linearized_input = input;
    backend_input.evaluator = &evaluator;

    level_set::ImplicitCutQuadratureBackendCellResult result{};
    result.cut.supported = true;
    result.achieved_interface_quadrature_order = 1;
    result.achieved_volume_quadrature_order = 1;
    result.diagnostic_status =
        level_set::ImplicitCutQuadratureDiagnosticStatus::Cut;

    FE::interfaces::CutInterfaceVolumeRegion bad_region{};
    bad_region.interface_marker = request.interface_marker;
    bad_region.parent_cell = input.parent_cell;
    bad_region.side = FE::geometry::CutIntegrationSide::Negative;
    bad_region.parent_measure = 1.0;
    bad_region.measure = std::numeric_limits<FE::Real>::quiet_NaN();
    bad_region.volume_fraction = 0.5;
    result.cut.volume_regions.push_back(bad_region);

    const auto validation =
        level_set::validateImplicitCutQuadratureBackendCellResult(
            request, backend_input, result);
    EXPECT_FALSE(validation.ok);
    EXPECT_NE(validation.diagnostic.find("invalid volume region"),
              std::string::npos);
}

TEST(LevelSetInterfaceLifecycle, BackendMetadataReachesCutIntegrationContext)
{
    constexpr int interface_marker = 83;
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.interface_quadrature_order = 0;
    options.volume_quadrature_order = 1;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);
    ASSERT_TRUE(result.success) << result.diagnostic;

    FE::assembly::CutIntegrationContext context;
    context.addGeneratedInterfaceDomain(result.domain);

    ASSERT_FALSE(context.volumeRules().empty());
    EXPECT_EQ(context.volumeRules().front().provenance.implicit_geometry_mode,
              "LinearCorner");
    EXPECT_EQ(context.volumeRules().front().provenance.implicit_quadrature_backend,
              "LinearCorner");
    EXPECT_EQ(context.volumeRules().front().provenance.predicate_policy_key,
              result.domain.request().quadrature_policy_key);
    ASSERT_FALSE(context.interfaceRules().empty());
    EXPECT_EQ(context.interfaceRules().front().provenance.marker,
              interface_marker);
    EXPECT_EQ(context.interfaceRules().front().provenance.implicit_fallback_policy,
              "Fail");
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleP1LineMatchesLinearMeasures)
{
    constexpr int interface_marker = 84;
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad4);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex, x[0]);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.interface_marker, interface_marker);
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle);
    EXPECT_NEAR(result.summary.negative_volume_measure, 2.0, 1.0e-12);
    EXPECT_NEAR(result.summary.positive_volume_measure, 2.0, 1.0e-12);
    EXPECT_NEAR(result.summary.measure, 2.0, 1.0e-12);
}

TEST(LevelSetInterfaceLifecycle, HighOrderSubcellP1LineMatchesLinearTriangleMeasures)
{
    constexpr int interface_marker = 87;
    const auto mesh =
        std::make_shared<SingleTriangleMeshAccess>(FE::ElementType::Triangle3);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTriangleSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 3; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex, x[0] - FE::Real{0.5});
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.interface_marker, interface_marker);
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.implicit_cut_quadrature_backend,
              level_set::ImplicitCutQuadratureBackend::HighOrderSubcell);
    EXPECT_NEAR(result.summary.negative_volume_measure, 0.375, 1.0e-12);
    EXPECT_NEAR(result.summary.positive_volume_measure, 0.125, 1.0e-12);
    EXPECT_NEAR(result.summary.measure, 0.5, 1.0e-12);
}

TEST(LevelSetInterfaceLifecycle, HighOrderSubcellP2CircleSegmentApproximatesAreaAndLength)
{
    constexpr int interface_marker = 88;
    constexpr FE::Real radius = 0.5;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const auto mesh =
        std::make_shared<SingleTriangleMeshAccess>(FE::ElementType::Triangle6);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTriangleSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 6u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 6u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] + x[1] * x[1] - radius * radius;
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.implicit_cut_max_subdivision_depth = 6;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.achieved_interface_quadrature_order, 1);
    EXPECT_EQ(result.achieved_volume_quadrature_order, 2);
    EXPECT_NE(result.diagnostic.find("HighOrderSubcell"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("linearized_leaves="), std::string::npos);
    EXPECT_GT(result.summary.active_fragment_count, 1u);
    EXPECT_NEAR(result.summary.negative_volume_measure,
                pi * radius * radius / 4.0,
                2.0e-2);
    EXPECT_NEAR(result.summary.positive_volume_measure,
                0.5 - pi * radius * radius / 4.0,
                2.0e-2);
    EXPECT_NEAR(result.summary.measure,
                pi * radius / 2.0,
                5.0e-2);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    EXPECT_EQ(interface_rules.front().provenance.implicit_quadrature_backend,
              "HighOrderSubcell");
    EXPECT_EQ(interface_rules.front().provenance.requested_quadrature_order, 2);
    EXPECT_EQ(interface_rules.front().provenance.achieved_quadrature_order, 1);

    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_FALSE(volume_rules.empty());
    EXPECT_EQ(volume_rules.front().provenance.implicit_quadrature_backend,
              "HighOrderSubcell");
    EXPECT_EQ(volume_rules.front().provenance.requested_quadrature_order, 2);
    EXPECT_EQ(volume_rules.front().provenance.achieved_quadrature_order, 2);
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleP2CircleApproximatesAreaAndLength)
{
    constexpr int interface_marker = 85;
    constexpr FE::Real radius = 0.5;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad9);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 9u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 9u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] + x[1] * x[1] - radius * radius;
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = 6;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.corner_linearized_cell_count, 0u);
    EXPECT_EQ(result.achieved_interface_quadrature_order, 1);
    EXPECT_EQ(result.achieved_volume_quadrature_order, 2);
    EXPECT_NE(result.diagnostic.find("SayeHyperrectangle"), std::string::npos);
    EXPECT_NE(result.diagnostic.find("max_depth_reached="), std::string::npos);
    EXPECT_NE(result.diagnostic.find("linearized_leaves="), std::string::npos);
    EXPECT_NE(result.diagnostic.find("interface_fragments="), std::string::npos);
    EXPECT_GT(result.summary.active_fragment_count, 1u);
    EXPECT_NEAR(result.summary.negative_volume_measure,
                pi * radius * radius,
                6.0e-2);
    EXPECT_NEAR(result.summary.measure,
                2.0 * pi * radius,
                1.2e-1);

    const auto x_moment = [](const std::array<FE::Real, 3>& x) { return x[0]; };
    const auto y_moment = [](const std::array<FE::Real, 3>& x) { return x[1]; };
    const auto r2_moment = [](const std::array<FE::Real, 3>& x) {
        return x[0] * x[0] + x[1] * x[1];
    };
    EXPECT_NEAR(integrateVolumeMoment(result.domain,
                                      FE::geometry::CutIntegrationSide::Negative,
                                      x_moment),
                0.0,
                2.0e-2);
    EXPECT_NEAR(integrateVolumeMoment(result.domain,
                                      FE::geometry::CutIntegrationSide::Negative,
                                      y_moment),
                0.0,
                2.0e-2);
    EXPECT_NEAR(integrateVolumeMoment(result.domain,
                                      FE::geometry::CutIntegrationSide::Negative,
                                      r2_moment),
                pi * radius * radius * radius * radius / 2.0,
                3.0e-2);
    EXPECT_NEAR(integrateInterfaceMoment(result.domain, x_moment), 0.0, 2.0e-2);
    EXPECT_NEAR(integrateInterfaceMoment(result.domain, y_moment), 0.0, 2.0e-2);
    EXPECT_NEAR(integrateInterfaceMoment(result.domain, r2_moment),
                2.0 * pi * radius * radius * radius,
                6.0e-2);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    EXPECT_EQ(interface_rules.front().provenance.requested_quadrature_order, 2);
    EXPECT_EQ(interface_rules.front().provenance.achieved_quadrature_order, 1);

    const std::string vtp =
        FE::interfaces::levelSetInterfaceGeometryVtpString(result.domain);
    EXPECT_NE(vtp.find("<VTKFile type=\"PolyData\""), std::string::npos);
    EXPECT_EQ(vtp.find("NumberOfLines=\"0\""), std::string::npos);
    EXPECT_NE(vtp.find("Name=\"level_set_value\""), std::string::npos);
    EXPECT_NE(vtp.find("Name=\"interface_normal\" NumberOfComponents=\"3\""),
              std::string::npos);
    EXPECT_NE(vtp.find("Name=\"negative_volume_fraction\""), std::string::npos);
    EXPECT_NE(vtp.find("Name=\"interface_marker\""), std::string::npos);
}

TEST(LevelSetInterfaceLifecycle, SayeHyperrectangleRulesAssembleFixedGeometryMeasures)
{
    constexpr int interface_marker = 86;
    constexpr FE::Real radius = 0.5;
    constexpr FE::Real pi = 3.141592653589793238462643383279502884;
    const auto mesh = std::make_shared<SingleQuadMeshAccess>(FE::ElementType::Quad9);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleQuadSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 9u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 9u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x[0] * x[0] + x[1] * x[1] - radius * radius;
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    options.implicit_cut_max_subdivision_depth = 6;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);
    ASSERT_TRUE(result.success) << result.diagnostic;

    FE::assembly::CutIntegrationContext context;
    ASSERT_NO_THROW(context.addGeneratedInterfaceDomain(result.domain));

    CutMeasureAssemblyKernel kernel;
    const auto build_context =
        [](const FE::assembly::CutRuleAssemblyRequest& request,
           FE::assembly::AssemblyContext& ctx) {
            ASSERT_NE(request.rule, nullptr);
            populateMeasureAssemblyContext(*request.rule, ctx);
        };

    FE::assembly::CutDomainAssemblyOptions volume_options;
    volume_options.include_interface_rules = false;
    volume_options.volume_marker = interface_marker;
    volume_options.volume_side = FE::geometry::CutIntegrationSide::Negative;
    const auto volume_summary =
        FE::assembly::assembleCutDomains(
            context, kernel, build_context, volume_options);

    ASSERT_GT(volume_summary.volume_rule_count, 0u);
    ASSERT_TRUE(volume_summary.hasVector());
    ASSERT_EQ(volume_summary.total_output.local_vector.size(), 1u);
    EXPECT_NEAR(volume_summary.total_output.local_vector[0],
                result.summary.negative_volume_measure,
                1.0e-12);
    EXPECT_NEAR(volume_summary.total_output.local_vector[0],
                pi * radius * radius,
                6.0e-2);

    FE::assembly::CutDomainAssemblyOptions interface_options;
    interface_options.include_volume_rules = false;
    interface_options.interface_marker = interface_marker;
    const auto interface_summary =
        FE::assembly::assembleCutDomains(
            context, kernel, build_context, interface_options);

    ASSERT_GT(interface_summary.interface_rule_count, 0u);
    ASSERT_TRUE(interface_summary.hasVector());
    ASSERT_EQ(interface_summary.total_output.local_vector.size(), 1u);
    EXPECT_NEAR(interface_summary.total_output.local_vector[0],
                result.summary.measure,
                1.0e-12);
    EXPECT_NEAR(interface_summary.total_output.local_vector[0],
                2.0 * pi * radius,
                1.2e-1);
}

TEST(LevelSetInterfaceLifecycle, UnimplementedBackendFactoryThrows)
{
    EXPECT_NO_THROW(
        (void)level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle));
    EXPECT_NO_THROW(
        (void)level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::HighOrderSubcell));
    EXPECT_THROW(
        (void)level_set::implicitCutQuadratureBackendDriver(
            level_set::ImplicitCutQuadratureBackend::MomentFit),
        std::invalid_argument);
}

TEST(LevelSetInterfaceLifecycle, RejectsNonH1LevelSetField)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::L2, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.domain_id = "water-air";

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    try {
        (void)lifecycle.build(system, options, solution);
        FAIL() << "Expected non-H1 level-set field to be rejected";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("H1/C0"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, FullSideVolumeRegionSucceedsWithoutInterfaceFragment)
{
    constexpr int interface_marker = 76;
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        setFieldComponentValue(solution, system, phi, vertex, FE::Real(-1.0));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water";
    options.tolerance = 1.0e-12;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.interface_marker, interface_marker);
    EXPECT_EQ(result.summary.active_fragment_count, 0u);
    EXPECT_EQ(result.summary.active_volume_region_count, 1u);
    EXPECT_GT(result.summary.negative_volume_measure, 0.0);
    EXPECT_EQ(result.summary.positive_volume_measure, 0.0);
}

TEST(LevelSetInterfaceLifecycle, GeneratedRulesUseReferenceCoordinatesOnPhysicalCells)
{
    constexpr int interface_marker = 79;
    const auto mesh = std::make_shared<SingleTetraMeshAccess>(
        std::vector<std::array<FE::Real, 3>>{
            std::array<FE::Real, 3>{10.0, 20.0, 30.0},
            std::array<FE::Real, 3>{12.0, 20.0, 30.0},
            std::array<FE::Real, 3>{10.0, 23.0, 30.0},
            std::array<FE::Real, 3>{10.0, 20.0, 34.0},
        });
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    setFieldComponentValue(solution, system, phi, 0, -0.5);
    setFieldComponentValue(solution, system, phi, 1, 0.5);
    setFieldComponentValue(solution, system, phi, 2, 0.5);
    setFieldComponentValue(solution, system, phi, 3, 0.5);

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.tolerance = 1.0e-12;
    options.interface_quadrature_order = 0;
    options.volume_quadrature_order = 1;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_EQ(interface_rules.size(), 1u);
    EXPECT_EQ(interface_rules.front().frame, FE::geometry::CutGeometryFrame::Reference);
    for (const auto& qp : interface_rules.front().points) {
        EXPECT_GE(qp.point[0], 0.0);
        EXPECT_GE(qp.point[1], 0.0);
        EXPECT_GE(qp.point[2], 0.0);
        EXPECT_LE(qp.point[0] + qp.point[1] + qp.point[2], 1.0);
    }

    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_EQ(volume_rules.size(), 2u);
    FE::Real measure = 0.0;
    for (const auto& rule : volume_rules) {
        EXPECT_EQ(rule.frame, FE::geometry::CutGeometryFrame::Reference);
        measure += rule.measure;
        for (const auto& qp : rule.points) {
            EXPECT_GE(qp.point[0], 0.0);
            EXPECT_GE(qp.point[1], 0.0);
            EXPECT_GE(qp.point[2], 0.0);
            EXPECT_LE(qp.point[0] + qp.point[1] + qp.point[2], 1.0);
        }
    }
    EXPECT_NEAR(measure, FE::Real(1.0) / FE::Real(6.0), 1.0e-12);
}

TEST(LevelSetInterfaceLifecycle, PreservesMarkerIdentity)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto make_solution = [&](FE::Real offset) {
        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
            const auto x = mesh->getNodeCoordinates(vertex);
            setFieldComponentValue(solution, system, phi, vertex,
                                   x[0] + x[1] + x[2] - offset);
        }
        return solution;
    };

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.domain_id = "water-air";

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto initial = lifecycle.build(system, options, make_solution(0.5));
    const auto updated = lifecycle.build(system, options, make_solution(0.75));

    ASSERT_TRUE(initial.success) << initial.diagnostic;
    ASSERT_TRUE(updated.success) << updated.diagnostic;
    EXPECT_GE(initial.interface_marker, 1000000);
    EXPECT_EQ(initial.interface_marker, updated.interface_marker);
    EXPECT_EQ(initial.domain.marker(), updated.domain.marker());
    EXPECT_EQ(updated.value_revision, initial.value_revision + 1u);
}

TEST(LevelSetInterfaceLifecycle, UpdatesGeometryAfterFieldChange)
{
    constexpr int interface_marker = 74;
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto make_solution = [&](FE::Real offset) {
        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
            const auto x = mesh->getNodeCoordinates(vertex);
            setFieldComponentValue(solution, system, phi, vertex,
                                   x[0] + x[1] + x[2] - offset);
        }
        return solution;
    };

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto initial = lifecycle.build(system, options, make_solution(0.5));
    const auto updated = lifecycle.build(system, options, make_solution(0.75));

    ASSERT_TRUE(initial.success) << initial.diagnostic;
    ASSERT_TRUE(updated.success) << updated.diagnostic;
    EXPECT_EQ(initial.value_revision, 1u);
    EXPECT_EQ(updated.value_revision, 2u);
    EXPECT_GT(initial.summary.measure, 0.0);
    EXPECT_GT(updated.summary.measure, 0.0);
    EXPECT_NE(initial.summary.measure, updated.summary.measure);
    ASSERT_EQ(initial.domain.fragments().size(), 1u);
    ASSERT_EQ(updated.domain.fragments().size(), 1u);
    EXPECT_NE(initial.domain.fragments().front().stable_id,
              updated.domain.fragments().front().stable_id);
}

TEST(LevelSetInterfaceLifecycle, CornerLinearizedHighOrderGeometryIsRejectedByDefault)
{
    constexpr int interface_marker = 81;
    const auto mesh = std::make_shared<SingleTetra10GeometryMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    try {
        (void)lifecycle.build(system, options, solution);
        FAIL() << "Expected high-order generated interface geometry to be rejected";
    } catch (const std::invalid_argument& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("corner-linearize"), std::string::npos);
    }
}

TEST(LevelSetInterfaceLifecycle, CornerLinearizedHighOrderGeometryReportsDiagnosticsWhenAllowed)
{
    constexpr int interface_marker = 82;
    const auto mesh = std::make_shared<SingleTetra10GeometryMeshAccess>();
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        setFieldComponentValue(solution, system, phi, vertex,
                               x[0] + x[1] + x[2] - FE::Real(0.5));
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = interface_marker;
    options.domain_id = "water-air";
    options.allow_corner_linearized_geometry = true;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.cell_count, 1u);
    EXPECT_EQ(result.corner_linearized_cell_count, 1u);
    EXPECT_EQ(result.max_cell_node_count, 10u);
    EXPECT_EQ(result.max_corner_node_count, 4u);
    EXPECT_EQ(result.interface_marker, interface_marker);
}
