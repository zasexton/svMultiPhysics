#include "LevelSet/LevelSetTransport.h"

#include "Assembly/Assembler.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/FormExpr.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"
#include "Systems/TimeIntegrator.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

using FE::forms::FormExpr;
using FE::forms::FormExprNode;
using FE::forms::FormExprType;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] std::shared_ptr<svmp::Mesh> buildNativeQuad9Mesh()
{
    auto base = std::make_shared<svmp::MeshBase>();

    const std::vector<svmp::real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
        0.5, 0.0,
        1.0, 0.5,
        0.5, 1.0,
        0.0, 0.5,
        0.5, 0.5,
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 9};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 2;
    base->build_from_arrays(/*spatial_dim=*/2,
                            x_ref,
                            cell2vertex_offsets,
                            cell2vertex,
                            {shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}
#endif

class SingleTetraMeshAccess final : public FE::assembly::IMeshAccess {
public:
    SingleTetraMeshAccess()
    {
        nodes_ = {
            std::array<FE::Real, 3>{0.0, 0.0, 0.0},
            std::array<FE::Real, 3>{1.0, 0.0, 0.0},
            std::array<FE::Real, 3>{0.0, 1.0, 0.0},
            std::array<FE::Real, 3>{0.0, 0.0, 1.0},
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
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

class SingleQuad9MeshAccess final : public FE::assembly::IMeshAccess {
public:
    SingleQuad9MeshAccess()
    {
        nodes_ = {
            std::array<FE::Real, 3>{0.0, 0.0, 0.0},
            std::array<FE::Real, 3>{1.0, 0.0, 0.0},
            std::array<FE::Real, 3>{1.0, 1.0, 0.0},
            std::array<FE::Real, 3>{0.0, 1.0, 0.0},
            std::array<FE::Real, 3>{0.5, 0.0, 0.0},
            std::array<FE::Real, 3>{1.0, 0.5, 0.0},
            std::array<FE::Real, 3>{0.5, 1.0, 0.0},
            std::array<FE::Real, 3>{0.0, 0.5, 0.0},
            std::array<FE::Real, 3>{0.5, 0.5, 0.0},
        };
        cell_ = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Quad9;
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
    std::array<FE::GlobalIndex, 9> cell_{};
};

class Quad9Patch2x2MeshAccess final : public FE::assembly::IMeshAccess {
public:
    Quad9Patch2x2MeshAccess()
    {
        nodes_.reserve(25u);
        for (int j = 0; j < 5; ++j) {
            for (int i = 0; i < 5; ++i) {
                nodes_.push_back(std::array<FE::Real, 3>{
                    FE::Real{0.25} * static_cast<FE::Real>(i),
                    FE::Real{0.25} * static_cast<FE::Real>(j),
                    0.0});
            }
        }

        for (int cy = 0; cy < 2; ++cy) {
            for (int cx = 0; cx < 2; ++cx) {
                const auto node = [](int ix, int iy) -> FE::GlobalIndex {
                    return static_cast<FE::GlobalIndex>(5 * iy + ix);
                };
                const int i = 2 * cx;
                const int j = 2 * cy;
                cells_.push_back(std::array<FE::GlobalIndex, 9>{
                    node(i, j),
                    node(i + 2, j),
                    node(i + 2, j + 2),
                    node(i, j + 2),
                    node(i + 1, j),
                    node(i + 2, j + 1),
                    node(i + 1, j + 2),
                    node(i, j + 1),
                    node(i + 1, j + 1)});
            }
        }
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override
    {
        return static_cast<FE::GlobalIndex>(cells_.size());
    }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return numCells(); }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Quad9;
    }

    void getCellNodes(FE::GlobalIndex cell_id,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        FE::GlobalIndex cell_id,
        std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        std::vector<FE::GlobalIndex> nodes;
        getCellNodes(cell_id, nodes);
        coords.clear();
        coords.reserve(nodes.size());
        for (const auto node : nodes) {
            coords.push_back(getNodeCoordinates(node));
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
        for (FE::GlobalIndex cell = 0; cell < numCells(); ++cell) {
            callback(cell);
        }
    }

    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
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
    std::vector<std::array<FE::GlobalIndex, 9>> cells_{};
};

[[nodiscard]] std::shared_ptr<FE::spaces::FunctionSpace> scalarSpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh)
{
    return FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
}

[[nodiscard]] std::shared_ptr<FE::spaces::FunctionSpace> scalarSpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh,
    int order)
{
    return FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, order, /*components=*/1);
}

[[nodiscard]] std::shared_ptr<FE::spaces::FunctionSpace> vectorSpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh)
{
    return FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/3);
}

bool containsExprType(const FormExprNode* node, FormExprType target)
{
    if (node == nullptr) {
        return false;
    }
    if (node->type() == target) {
        return true;
    }
    for (const auto* child : node->children()) {
        if (containsExprType(child, target)) {
            return true;
        }
    }
    return false;
}

bool formulationRecordsContain(const FE::systems::FESystem& system,
                               FormExprType target)
{
    for (const auto& record : system.formulationRecords()) {
        if (containsExprType(record.residual_expr.get(), target)) {
            return true;
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (containsExprType(expr.get(), target)) {
                return true;
            }
        }
    }
    return false;
}

void addScalarAndVelocityFields(FE::systems::FESystem& system,
                                const std::shared_ptr<const FE::spaces::FunctionSpace>& scalar_space,
                                const std::shared_ptr<const FE::spaces::FunctionSpace>& velocity_space,
                                FE::systems::FieldSourceKind velocity_source =
                                    FE::systems::FieldSourceKind::PrescribedData)
{
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    system.addField(FE::systems::FieldSpec{
        .name = "advecting_velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
        .source_kind = velocity_source,
    });
}

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

[[nodiscard]] FE::systems::SetupInputs makeSingleQuad9SetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 9;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 9};
    topo.cell2vertex_data = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

[[nodiscard]] FE::systems::SetupInputs makeQuad9Patch2x2SetupInputs()
{
    const Quad9Patch2x2MeshAccess mesh;
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 4;
    topo.n_vertices = 25;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 2;

    topo.cell2vertex_offsets.reserve(5u);
    topo.cell2vertex_offsets.push_back(0);
    for (FE::GlobalIndex cell = 0; cell < 4; ++cell) {
        std::vector<FE::GlobalIndex> nodes;
        mesh.getCellNodes(cell, nodes);
        topo.cell2vertex_data.insert(
            topo.cell2vertex_data.end(),
            nodes.begin(),
            nodes.end());
        topo.cell2vertex_offsets.push_back(
            static_cast<FE::GlobalIndex>(topo.cell2vertex_data.size()));
    }
    for (FE::GlobalIndex vertex = 0; vertex < 25; ++vertex) {
        topo.vertex_gids.push_back(vertex);
    }
    for (FE::GlobalIndex cell = 0; cell < 4; ++cell) {
        topo.cell_gids.push_back(cell);
        topo.cell_owner_ranks.push_back(0);
    }

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

std::vector<FE::Real> constantVectorTetraCoefficients(FE::Real x,
                                                      FE::Real y,
                                                      FE::Real z)
{
    std::vector<FE::Real> coefficients(12u, 0.0);
    for (std::size_t node = 0; node < 4u; ++node) {
        coefficients[node] = x;
        coefficients[4u + node] = y;
        coefficients[8u + node] = z;
    }
    return coefficients;
}

void setFieldComponentValue(std::vector<FE::Real>& solution,
                            const FE::systems::FESystem& system,
                            FE::FieldId field,
                            FE::GlobalIndex vertex,
                            int component,
                            FE::Real value)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto offset = system.fieldDofOffset(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("setFieldComponentValue: field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (component < 0 || static_cast<std::size_t>(component) >= dofs.size()) {
        throw std::runtime_error("setFieldComponentValue: component is out of range");
    }
    const auto index = static_cast<std::size_t>(
        dofs[static_cast<std::size_t>(component)] + offset);
    if (index >= solution.size()) {
        throw std::runtime_error("setFieldComponentValue: DOF index is out of range");
    }
    solution[index] = value;
}

void setScalarVertexValue(std::vector<FE::Real>& solution,
                          const FE::systems::FESystem& system,
                          FE::FieldId field,
                          FE::GlobalIndex vertex,
                          FE::Real value)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto offset = system.fieldDofOffset(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("setScalarVertexValue: field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (dofs.size() != 1u) {
        throw std::runtime_error("setScalarVertexValue: expected one scalar vertex DOF");
    }
    const auto index = static_cast<std::size_t>(dofs.front() + offset);
    if (index >= solution.size()) {
        throw std::runtime_error("setScalarVertexValue: DOF index is out of range");
    }
    solution[index] = value;
}

[[nodiscard]] std::vector<FE::Real> assembleLevelSetResidual(
    FE::systems::FESystem& system,
    const FE::systems::SystemStateView& state)
{
    const auto n = system.dofHandler().getNumDofs();
    FE::assembly::DenseVectorView residual(n);
    residual.zero();

    FE::systems::AssemblyRequest request;
    request.op = "level_set";
    request.want_vector = true;
    const auto result = system.assemble(request, state, nullptr, &residual);
    EXPECT_TRUE(result.success) << result.error_message;

    std::vector<FE::Real> out(static_cast<std::size_t>(n), 0.0);
    for (FE::GlobalIndex i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = residual.getVectorEntry(i);
    }
    return out;
}

[[nodiscard]] FE::Real l2Norm(std::span<const FE::Real> values)
{
    FE::Real sum = 0.0;
    for (const auto value : values) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

void expectOperatorJacobianMatchesCentralFD(FE::systems::FESystem& system,
                                            const FE::systems::SystemStateView& base_state,
                                            FE::Real eps,
                                            FE::Real rtol,
                                            FE::Real atol)
{
    const auto n = system.dofHandler().getNumDofs();
    ASSERT_GT(n, 0);

    const std::vector<FE::Real> base_u(base_state.u.begin(), base_state.u.end());
    ASSERT_EQ(static_cast<FE::GlobalIndex>(base_u.size()), n);

    FE::assembly::DenseMatrixView jacobian(n);
    {
        FE::systems::AssemblyRequest request;
        request.op = "level_set";
        request.want_matrix = true;
        const auto result = system.assemble(request, base_state, &jacobian, nullptr);
        ASSERT_TRUE(result.success) << result.error_message;
    }

    for (FE::GlobalIndex column = 0; column < n; ++column) {
        std::vector<FE::Real> u_plus = base_u;
        std::vector<FE::Real> u_minus = base_u;
        u_plus[static_cast<std::size_t>(column)] += eps;
        u_minus[static_cast<std::size_t>(column)] -= eps;

        FE::systems::SystemStateView state_plus = base_state;
        FE::systems::SystemStateView state_minus = base_state;
        state_plus.u = std::span<const FE::Real>(u_plus);
        state_minus.u = std::span<const FE::Real>(u_minus);

        FE::assembly::DenseVectorView r_plus(n);
        FE::assembly::DenseVectorView r_minus(n);
        {
            FE::systems::AssemblyRequest request;
            request.op = "level_set";
            request.want_vector = true;
            const auto result = system.assemble(request, state_plus, nullptr, &r_plus);
            ASSERT_TRUE(result.success) << result.error_message;
        }
        {
            FE::systems::AssemblyRequest request;
            request.op = "level_set";
            request.want_vector = true;
            const auto result = system.assemble(request, state_minus, nullptr, &r_minus);
            ASSERT_TRUE(result.success) << result.error_message;
        }

        for (FE::GlobalIndex row = 0; row < n; ++row) {
            const FE::Real finite_difference = (r_plus[row] - r_minus[row]) / (2.0 * eps);
            const FE::Real assembled = jacobian(row, column);
            const FE::Real tolerance =
                atol + rtol * std::max<FE::Real>(1.0, std::abs(finite_difference));
            EXPECT_NEAR(assembled, finite_difference, tolerance)
                << "Mismatch at (row=" << row << ", column=" << column << ")";
        }
    }
}

} // namespace

TEST(LevelSetTransport, ValidatesFieldOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem scalar_system(mesh);
    scalar_system.addField(FE::systems::FieldSpec{
        .name = "level_set",
        .space = phi_space,
        .components = 1,
    });
    scalar_system.addField(FE::systems::FieldSpec{
        .name = "Velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });
    EXPECT_NO_THROW(
        (void)level_set::installLevelSetTransport(scalar_system, phi_space, {}));

    FE::systems::FESystem vector_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(vector_system, velocity_space, {}),
        std::invalid_argument);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name.clear();
    FE::systems::FESystem empty_name_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(empty_name_system, phi_space, options),
        std::invalid_argument);

    options.level_set.field_name = "phi";
    options.velocity.field_name.clear();
    FE::systems::FESystem empty_velocity_name_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            empty_velocity_name_system,
            phi_space,
            options),
        std::invalid_argument);
}

TEST(LevelSetTransport, AutoRegistersConfiguredFields)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.source = level_set::LevelSetFieldSource::PrescribedData;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.velocity.auto_register_field = true;
    options.velocity.space = velocity_space;

    const auto kernels = level_set::installLevelSetTransport(system, phi_space, options);

    const auto phi = system.findFieldByName("phi");
    const auto velocity = system.findFieldByName("advecting_velocity");
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.fieldRecord(phi).source_kind, FE::systems::FieldSourceKind::Unknown);
    EXPECT_EQ(system.fieldRecord(velocity).source_kind,
              FE::systems::FieldSourceKind::PrescribedData);
    EXPECT_TRUE(system.hasOperator("level_set"));
    EXPECT_FALSE(kernels.residual.empty());
}

TEST(LevelSetTransport, InstallsOnConfiguredOperatorTag)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);

    level_set::LevelSetTransportOptions options{};
    options.operator_tag = "equations";
    options.level_set.field_name = "phi";
    options.level_set.source = level_set::LevelSetFieldSource::PrescribedData;
    options.velocity.field_name = "Velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;
    options.velocity.auto_register_field = true;
    options.velocity.space = velocity_space;

    const auto kernels = level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(system.hasOperator("equations"));
    EXPECT_FALSE(system.hasOperator("level_set"));
    EXPECT_FALSE(kernels.residual.empty());
}

TEST(LevelSetTransport, AutoRegistersCoupledVelocityAsUnknownWhenRequested)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.source = level_set::LevelSetFieldSource::PrescribedData;
    options.velocity.field_name = "Velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;
    options.velocity.auto_register_field = true;
    options.velocity.space = velocity_space;

    const auto kernels = level_set::installLevelSetTransport(system, phi_space, options);

    const auto phi = system.findFieldByName("phi");
    const auto velocity = system.findFieldByName("Velocity");
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.fieldRecord(phi).source_kind, FE::systems::FieldSourceKind::Unknown);
    EXPECT_EQ(system.fieldRecord(velocity).source_kind,
              FE::systems::FieldSourceKind::Unknown);
    EXPECT_TRUE(system.hasOperator("level_set"));
    EXPECT_FALSE(kernels.residual.empty());
}

TEST(LevelSetTransport, InstallsResidualFormStructure)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(system.hasOperator("level_set"));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::TimeDerivative));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::Divergence));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::CellDiameter));
}

TEST(LevelSetTransport, ConservativeDivergenceTransportUsesDivergenceResidual)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.transport_form = level_set::LevelSetTransportForm::ConservativeDivergence;
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(system.hasOperator("level_set"));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::TimeDerivative));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Divergence));
}

TEST(LevelSetTransport, SUPGAddsCellDiameterStabilization)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.supg.enabled = true;

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellDiameter));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::TimeDerivative));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
}

TEST(LevelSetTransport, Quad9FlatHorizontalNullModeHasZeroSpatialResidual)
{
    const auto run_case =
        [](const std::shared_ptr<FE::assembly::IMeshAccess>& mesh,
           const FE::systems::SetupInputs& setup) {
            auto phi_space = scalarSpace(mesh, /*order=*/2);

            FE::systems::FESystem system(mesh);
            const auto phi = system.addField(FE::systems::FieldSpec{
                .name = "phi",
                .space = phi_space,
                .components = 1,
            });

            level_set::LevelSetTransportOptions options{};
            options.level_set.field_name = "phi";
            options.level_set.auto_register_field = false;
            options.velocity.source = level_set::LevelSetVelocitySource::ConstantVector;
            options.velocity.constant_value = {0.1, 0.0, 0.0};
            options.supg.enabled = false;

            (void)level_set::installLevelSetTransport(system, phi_space, options);
            ASSERT_NO_THROW(system.setup({}, setup));

            std::vector<FE::Real> solution(
                static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
            const auto& phi_dofs = system.fieldDofHandler(phi);
            const auto offset = system.fieldDofOffset(phi);
            for (FE::GlobalIndex cell = 0; cell < mesh->numCells(); ++cell) {
                std::vector<FE::GlobalIndex> nodes;
                mesh->getCellNodes(cell, nodes);
                const auto dofs = phi_dofs.getCellDofs(cell);
                ASSERT_EQ(dofs.size(), nodes.size());
                for (std::size_t local = 0; local < nodes.size(); ++local) {
                    const auto x = mesh->getNodeCoordinates(nodes[local]);
                    const auto index = static_cast<std::size_t>(dofs[local] + offset);
                    ASSERT_LT(index, solution.size());
                    solution[index] = x[1] - FE::Real{0.375};
                }
            }
            const auto previous_solution = solution;

            FE::systems::SystemStateView state;
            state.dt = 0.1;
            state.u = std::span<const FE::Real>(solution);
            state.u_prev = std::span<const FE::Real>(previous_solution);
            const FE::systems::BackwardDifferenceIntegrator integrator;
            const auto time_context =
                integrator.buildContext(/*max_time_derivative_order=*/1, state);
            state.time_integration = &time_context;

            const auto residual = assembleLevelSetResidual(system, state);
            EXPECT_LT(l2Norm(std::span<const FE::Real>(residual)), 1.0e-12);
        };

    run_case(
        std::make_shared<SingleQuad9MeshAccess>(),
        makeSingleQuad9SetupInputs());
    run_case(
        std::make_shared<Quad9Patch2x2MeshAccess>(),
        makeQuad9Patch2x2SetupInputs());
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST(LevelSetTransport,
     NativeQuad9ProjectedFlatHorizontalNullModeHasZeroSpatialResidual)
{
    auto mesh = buildNativeQuad9Mesh();
    auto phi_space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Quad4,
        /*order=*/2);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
    });

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.source = level_set::LevelSetVelocitySource::ConstantVector;
    options.velocity.constant_value = {0.1, 0.0, 0.0};
    options.supg.enabled = false;

    (void)level_set::installLevelSetTransport(system, phi_space, options);
    ASSERT_NO_THROW(system.setup());

    std::vector<FE::Real> mesh_values(mesh->n_vertices(), FE::Real{0});
    for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
        const auto point =
            mesh->get_vertex_coords(static_cast<svmp::index_t>(vertex));
        mesh_values[vertex] = point[1] - FE::Real{0.375};
    }

    std::vector<FE::Real> phi_coefficients(
        static_cast<std::size_t>(system.fieldDofHandler(phi).getNumDofs()),
        FE::Real{0});
    std::vector<std::uint8_t> assigned(phi_coefficients.size(), 0u);
    const auto projection = system.projectMeshVertexValuesToFieldCoefficients(
        phi,
        std::span<const FE::Real>(mesh_values.data(), mesh_values.size()),
        /*mesh_components=*/1,
        std::span<FE::Real>(phi_coefficients.data(), phi_coefficients.size()),
        std::span<std::uint8_t>(assigned.data(), assigned.size()),
        "LevelSetTransport native Quad9 projection invariant");
    ASSERT_EQ(projection.unassigned_dofs, 0u);
    ASSERT_EQ(projection.values_written, phi_coefficients.size());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        FE::Real{0});
    const auto offset = system.fieldDofOffset(phi);
    ASSERT_GE(offset, 0);
    ASSERT_LE(static_cast<std::size_t>(offset) + phi_coefficients.size(),
              solution.size());
    std::copy(phi_coefficients.begin(),
              phi_coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));
    const auto previous_solution = solution;

    FE::systems::SystemStateView state;
    state.dt = 0.1;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    const auto residual = assembleLevelSetResidual(system, state);
    EXPECT_LT(l2Norm(std::span<const FE::Real>(residual)), 1.0e-12);
}
#endif

TEST(LevelSetTransport, InterfaceKinematicAddsInterfaceResidual)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.interface_kinematic.enabled = true;
    options.interface_kinematic.interface_marker = 77;
    options.interface_kinematic.weight_scale = 2.0;

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellDiameter));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::TimeDerivative));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
}

TEST(LevelSetTransport, ValidatesSUPGOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.supg.enabled = true;

    options.supg.tau_scale = 0.0;
    FE::systems::FESystem tau_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(tau_system, phi_space, options),
        std::invalid_argument);

    options.supg.tau_scale = 0.5;
    options.supg.velocity_epsilon = 0.0;
    FE::systems::FESystem epsilon_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(epsilon_system, phi_space, options),
        std::invalid_argument);
}

TEST(LevelSetTransport, ValidatesInterfaceKinematicOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.interface_kinematic.enabled = true;

    FE::systems::FESystem marker_system(mesh);
    addScalarAndVelocityFields(marker_system, phi_space, velocity_space);
    options.interface_kinematic.interface_marker = -1;
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(marker_system, phi_space, options),
        std::invalid_argument);

    FE::systems::FESystem weight_system(mesh);
    addScalarAndVelocityFields(weight_system, phi_space, velocity_space);
    options.interface_kinematic.interface_marker = 77;
    options.interface_kinematic.weight_scale = 0.0;
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(weight_system, phi_space, options),
        std::invalid_argument);

    FE::systems::FESystem valid_system(mesh);
    addScalarAndVelocityFields(valid_system, phi_space, velocity_space);
    options.interface_kinematic.weight_scale = 1.0;
    EXPECT_NO_THROW(
        (void)level_set::installLevelSetTransport(valid_system, phi_space, options));
}

TEST(LevelSetTransport, ValidatesReinitializationOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.reinitialization.enabled = true;

    options.reinitialization.cadence_steps = 0;
    FE::systems::FESystem cadence_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(cadence_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.cadence_steps = 1;
    options.reinitialization.max_iterations = 0;
    FE::systems::FESystem iterations_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(iterations_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.max_iterations = 10;
    options.reinitialization.pseudo_time_step_scale = 0.0;
    FE::systems::FESystem pseudo_time_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(pseudo_time_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.pseudo_time_step_scale = 0.3;
    options.reinitialization.interface_band_width = 0.0;
    FE::systems::FESystem band_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(band_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.interface_band_width = 3.0;
    options.reinitialization.signed_distance_tolerance = 0.0;
    FE::systems::FESystem tolerance_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(tolerance_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.signed_distance_tolerance = 1.0e-6;
    options.reinitialization.method =
        level_set::LevelSetReinitializationMethod::FastMarching;
    FE::systems::FESystem unsupported_method_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            unsupported_method_system, phi_space, options),
        std::invalid_argument);

    options.reinitialization.method =
        level_set::LevelSetReinitializationMethod::Projection;
    FE::systems::FESystem valid_system(mesh);
    valid_system.addField(FE::systems::FieldSpec{
        .name = "Velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });
    EXPECT_NO_THROW(
        (void)level_set::installLevelSetTransport(valid_system, phi_space, options));
}

TEST(LevelSetTransport, ValidatesVolumeCorrectionOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.volume_correction.enabled = true;

    options.volume_correction.cadence_steps = 0;
    FE::systems::FESystem cadence_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(cadence_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.cadence_steps = 1;
    options.volume_correction.volume_tolerance = 0.0;
    FE::systems::FESystem tolerance_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(tolerance_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.volume_tolerance = 1.0e-10;
    options.volume_correction.max_iterations = 0;
    FE::systems::FESystem iterations_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(iterations_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.max_iterations = 50;
    options.volume_correction.use_initial_negative_volume_as_target = false;
    options.volume_correction.target_negative_volume = -1.0;
    FE::systems::FESystem target_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(target_system, phi_space, options),
        std::invalid_argument);

    options.volume_correction.target_negative_volume = 0.125;
    FE::systems::FESystem valid_system(mesh);
    valid_system.addField(FE::systems::FieldSpec{
        .name = "Velocity",
        .space = velocity_space,
        .components = velocity_space->value_dimension(),
    });
    EXPECT_NO_THROW(
        (void)level_set::installLevelSetTransport(valid_system, phi_space, options));
}

TEST(LevelSetTransport, InflowBoundaryAddsUpwindPenalty)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.boundaries.inflow.push_back(level_set::LevelSetInflowBoundary{
        .boundary_marker = 4,
        .value = FE::Real{1.25},
        .penalty_scale = 2.0,
    });

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Normal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::AbsoluteValue));
}

TEST(LevelSetTransport, OutflowBoundaryIsNatural)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;
    options.boundaries.outflow.push_back(
        level_set::LevelSetOutflowBoundary{.boundary_marker = 5});

    (void)level_set::installLevelSetTransport(system, phi_space, options);

    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
}

TEST(LevelSetTransport, ValidatesBoundaryOptions)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);

    level_set::LevelSetTransportOptions options{};
    options.boundaries.inflow.push_back(level_set::LevelSetInflowBoundary{});
    FE::systems::FESystem missing_marker_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            missing_marker_system,
            phi_space,
            options),
        std::invalid_argument);

    options.boundaries.inflow.clear();
    options.boundaries.inflow.push_back(level_set::LevelSetInflowBoundary{
        .boundary_marker = 4,
        .penalty_scale = 0.0,
    });
    FE::systems::FESystem penalty_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(penalty_system, phi_space, options),
        std::invalid_argument);

    options.boundaries.inflow.clear();
    options.boundaries.inflow.push_back(
        level_set::LevelSetInflowBoundary{.boundary_marker = 4});
    options.boundaries.outflow.push_back(
        level_set::LevelSetOutflowBoundary{.boundary_marker = 4});
    FE::systems::FESystem duplicate_marker_system(mesh);
    EXPECT_THROW(
        (void)level_set::installLevelSetTransport(
            duplicate_marker_system,
            phi_space,
            options),
        std::invalid_argument);
}

TEST(LevelSetTransport, PrescribedVelocityJacobianMatchesFiniteDifference)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(system, phi_space, velocity_space);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::PrescribedData;

    (void)level_set::installLevelSetTransport(system, phi_space, options);
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto phi = system.findFieldByName("phi");
    const auto velocity = system.findFieldByName("advecting_velocity");
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);
    system.setPrescribedFieldCoefficients(
        velocity,
        constantVectorTetraCoefficients(0.70, -0.15, 0.25));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = static_cast<FE::Real>(vertex);
        setFieldComponentValue(
            solution,
            system,
            phi,
            vertex,
            0,
            FE::Real(0.20) + FE::Real(0.035) * x);
        setFieldComponentValue(
            previous_solution,
            system,
            phi,
            vertex,
            0,
            FE::Real(0.18) + FE::Real(0.025) * x);
    }

    FE::systems::SystemStateView state;
    state.dt = 0.1;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    expectOperatorJacobianMatchesCentralFD(
        system,
        state,
        1.0e-6,
        2.0e-5,
        1.0e-8);
}

TEST(LevelSetTransport, CoupledVelocityJacobianMatchesFiniteDifference)
{
    const auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto phi_space = scalarSpace(mesh);
    auto velocity_space = vectorSpace(mesh);

    FE::systems::FESystem system(mesh);
    addScalarAndVelocityFields(
        system,
        phi_space,
        velocity_space,
        FE::systems::FieldSourceKind::Unknown);

    level_set::LevelSetTransportOptions options{};
    options.level_set.field_name = "phi";
    options.level_set.auto_register_field = false;
    options.velocity.field_name = "advecting_velocity";
    options.velocity.source = level_set::LevelSetVelocitySource::CoupledField;

    (void)level_set::installLevelSetTransport(system, phi_space, options);
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto phi = system.findFieldByName("phi");
    const auto velocity = system.findFieldByName("advecting_velocity");
    ASSERT_NE(phi, FE::INVALID_FIELD_ID);
    ASSERT_NE(velocity, FE::INVALID_FIELD_ID);

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = static_cast<FE::Real>(vertex);
        setFieldComponentValue(
            solution,
            system,
            phi,
            vertex,
            0,
            FE::Real(0.15) + FE::Real(0.04) * x);
        setFieldComponentValue(
            previous_solution,
            system,
            phi,
            vertex,
            0,
            FE::Real(0.12) + FE::Real(0.03) * x);
        setFieldComponentValue(
            solution,
            system,
            velocity,
            vertex,
            0,
            FE::Real(0.40) + FE::Real(0.015) * x);
        setFieldComponentValue(
            solution,
            system,
            velocity,
            vertex,
            1,
            FE::Real(-0.20) + FE::Real(0.010) * x);
        setFieldComponentValue(
            solution,
            system,
            velocity,
            vertex,
            2,
            FE::Real(0.30) - FE::Real(0.005) * x);
    }

    FE::systems::SystemStateView state;
    state.dt = 0.1;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    expectOperatorJacobianMatchesCentralFD(
        system,
        state,
        1.0e-6,
        5.0e-5,
        1.0e-8);
}
