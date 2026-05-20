/* Copyright (c) Stanford University, The Regents of the University of California,
 * and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_LevelSetCellEvaluatorMPI.cpp
 * @brief MPI ownership checks for generated level-set field evaluation.
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"
#include "Dofs/EntityDofMap.h"
#include "Elements/ElementFactory.h"
#include "LevelSet/LevelSetCellEvaluator.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "Spaces/FunctionSpace.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class SharedTriangleMeshAccess final : public FE::assembly::IMeshAccess {
public:
    explicit SharedTriangleMeshAccess(int rank)
        : rank_(rank)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        return rank_ == 0 ? 1 : 0;
    }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 3; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override
    {
        return static_cast<std::uint64_t>(31 + rank_);
    }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override
    {
        return rank_ == 0;
    }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Triangle6;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes = {0, 1, 2, 3, 4, 5};
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
        if (rank_ == 0) {
            callback(0);
        }
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
    int rank_{0};
    std::array<std::array<FE::Real, 3>, 6> nodes_{{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.5, 0.0, 0.0}},
        {{0.5, 0.5, 0.0}},
        {{0.0, 0.5, 0.0}},
    }};
};

class SharedMixedQuadTriangleMeshAccess final
    : public FE::assembly::IMeshAccess {
public:
    explicit SharedMixedQuadTriangleMeshAccess(
        int rank,
        bool distributed_ownership = true)
        : rank_(rank)
        , distributed_ownership_(distributed_ownership)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        return distributed_ownership_ ? 1 : 2;
    }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 7; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 17; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 23; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override
    {
        return static_cast<std::uint64_t>(
            distributed_ownership_ ? 41 + rank_ : 41);
    }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 29; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex cell_id) const override
    {
        return !distributed_ownership_ ||
               cell_id == static_cast<FE::GlobalIndex>(rank_);
    }

    [[nodiscard]] FE::ElementType getCellType(
        FE::GlobalIndex cell_id) const override
    {
        return cell_id == 0 ? FE::ElementType::Quad4
                            : FE::ElementType::Triangle3;
    }

    void getCellNodes(FE::GlobalIndex cell_id,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        if (cell_id == 0) {
            nodes = {0, 1, 2, 3};
        } else {
            nodes = {4, 5, 6};
        }
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

    [[nodiscard]] int getBoundaryFaceMarker(
        FE::GlobalIndex /*face_id*/) const override
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
        if (rank_ == 0) {
            callback(0);
            callback(1);
        } else {
            callback(1);
            callback(0);
        }
    }

    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        if (!distributed_ownership_) {
            callback(0);
            callback(1);
            return;
        }
        callback(static_cast<FE::GlobalIndex>(rank_));
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
    int rank_{0};
    bool distributed_ownership_{true};
    std::array<std::array<FE::Real, 3>, 7> nodes_{{
        {{-1.0, -1.0, 0.0}},
        {{1.0, -1.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{-1.0, 1.0, 0.0}},
        {{2.0, 0.0, 0.0}},
        {{3.0, 0.0, 0.0}},
        {{2.0, 1.0, 0.0}},
    }};
};

class MixedQuadTriangleLinearH1Space final
    : public FE::spaces::FunctionSpace {
public:
    MixedQuadTriangleLinearH1Space()
        : quad_(makeElement(FE::ElementType::Quad4))
        , triangle_(makeElement(FE::ElementType::Triangle3))
    {
    }

    [[nodiscard]] FE::spaces::SpaceType space_type() const noexcept override
    {
        return FE::spaces::SpaceType::H1;
    }
    [[nodiscard]] FE::FieldType field_type() const noexcept override
    {
        return FE::FieldType::Scalar;
    }
    [[nodiscard]] FE::Continuity continuity() const noexcept override
    {
        return FE::Continuity::C0;
    }
    [[nodiscard]] int value_dimension() const noexcept override { return 1; }
    [[nodiscard]] int topological_dimension() const noexcept override { return 2; }
    [[nodiscard]] int polynomial_order() const noexcept override { return 1; }
    [[nodiscard]] int polynomial_order(
        FE::GlobalIndex /*cell_id*/) const noexcept override
    {
        return 1;
    }
    [[nodiscard]] bool is_variable_order() const noexcept override { return true; }
    [[nodiscard]] FE::ElementType element_type() const noexcept override
    {
        return FE::ElementType::Unknown;
    }
    [[nodiscard]] const FE::elements::Element& element() const noexcept override
    {
        return *quad_;
    }
    [[nodiscard]] const FE::elements::Element& getElement(
        FE::ElementType cell_type,
        FE::GlobalIndex /*cell_id*/) const noexcept override
    {
        return cell_type == FE::ElementType::Triangle3 ? *triangle_ : *quad_;
    }
    [[nodiscard]] std::shared_ptr<const FE::elements::Element> element_ptr()
        const noexcept override
    {
        return quad_;
    }
    [[nodiscard]] std::size_t dofs_per_element() const noexcept override
    {
        return quad_->num_dofs();
    }
    [[nodiscard]] std::size_t dofs_per_element(
        FE::GlobalIndex cell_id) const noexcept override
    {
        return cell_id == 1 ? triangle_->num_dofs() : quad_->num_dofs();
    }

    [[nodiscard]] Value evaluate(
        const Value& xi,
        const std::vector<FE::Real>& coefficients) const override
    {
        const auto& elem = elementForCoefficients(coefficients.size());
        std::vector<FE::Real> values(elem.num_dofs(), FE::Real{0});
        elem.basis().evaluate_values(xi, values);

        Value result{};
        for (std::size_t i = 0; i < values.size(); ++i) {
            result[0] += values[i] * coefficients[i];
        }
        return result;
    }

    [[nodiscard]] Gradient evaluate_gradient(
        const Value& xi,
        const std::vector<FE::Real>& coefficients) const override
    {
        const auto& elem = elementForCoefficients(coefficients.size());
        std::vector<FE::basis::Gradient> gradients(elem.num_dofs());
        elem.basis().evaluate_gradients(xi, gradients);

        Gradient result{};
        for (std::size_t i = 0; i < gradients.size(); ++i) {
            for (std::size_t d = 0; d < 3; ++d) {
                result[d] += gradients[i][d] * coefficients[i];
            }
        }
        return result;
    }

private:
    [[nodiscard]] static std::shared_ptr<FE::elements::Element> makeElement(
        FE::ElementType element_type)
    {
        FE::elements::ElementRequest request;
        request.element_type = element_type;
        request.field_type = FE::FieldType::Scalar;
        request.continuity = FE::Continuity::C0;
        request.basis_type = FE::BasisType::Lagrange;
        request.order = 1;
        return FE::elements::ElementFactory::create(request);
    }

    [[nodiscard]] const FE::elements::Element& elementForCoefficients(
        std::size_t coefficient_count) const
    {
        if (coefficient_count == quad_->num_dofs()) {
            return *quad_;
        }
        if (coefficient_count == triangle_->num_dofs()) {
            return *triangle_;
        }
        throw std::invalid_argument(
            "MixedQuadTriangleLinearH1Space: unexpected coefficient count");
    }

    std::shared_ptr<FE::elements::Element> quad_{};
    std::shared_ptr<FE::elements::Element> triangle_{};
};

[[nodiscard]] FE::systems::SetupInputs sharedTriangleSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 3;
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

[[nodiscard]] FE::systems::SetupInputs sharedMixedQuadTriangleSetupInputs(
    bool distributed_ownership = true)
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 7;
    topo.n_edges = 7;
    topo.n_faces = 0;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 4, 7};
    topo.cell2vertex_data = {0, 1, 2, 3, 4, 5, 6};
    topo.cell2edge_offsets = {0, 4, 7};
    topo.cell2edge_data = {0, 1, 2, 3, 4, 5, 6};
    topo.edge2vertex_data = {
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        4, 5,
        5, 6,
        6, 4,
    };
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6};
    topo.edge_gids = {0, 1, 2, 3, 4, 5, 6};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks =
        distributed_ownership ? std::vector<int>{0, 1}
                              : std::vector<int>{0, 0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

[[nodiscard]] FE::systems::SetupOptions mpiSetupOptions(
    MPI_Comm comm,
    int rank,
    int world_size)
{
    FE::systems::SetupOptions options;
    options.dof_options.global_numbering =
        FE::dofs::GlobalNumberingMode::OwnerContiguous;
    options.dof_options.ownership = FE::dofs::OwnershipStrategy::VertexGID;
    options.dof_options.my_rank = rank;
    options.dof_options.world_size = world_size;
    options.dof_options.mpi_comm = comm;
    return options;
}

[[nodiscard]] MPI_Datatype mpiRealType()
{
    if (sizeof(FE::Real) == sizeof(double)) {
        return MPI_DOUBLE;
    }
    if (sizeof(FE::Real) == sizeof(float)) {
        return MPI_FLOAT;
    }
    return MPI_LONG_DOUBLE;
}

[[nodiscard]] FE::Real allreduceMin(FE::Real value, MPI_Comm comm)
{
    FE::Real global = value;
    MPI_Allreduce(&value, &global, 1, mpiRealType(), MPI_MIN, comm);
    return global;
}

[[nodiscard]] FE::Real allreduceMax(FE::Real value, MPI_Comm comm)
{
    FE::Real global = value;
    MPI_Allreduce(&value, &global, 1, mpiRealType(), MPI_MAX, comm);
    return global;
}

void mixHash(std::uint64_t& hash, std::uint64_t value) noexcept
{
    hash ^= value;
    hash *= 1099511628211ull;
}

void mixStringHash(std::uint64_t& hash, const std::string& value) noexcept
{
    for (const char c : value) {
        mixHash(hash,
                static_cast<std::uint64_t>(static_cast<unsigned char>(c)));
    }
    mixHash(hash, 0xffu);
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
        throw std::runtime_error(
            "setFieldComponentValue: field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (dofs.empty()) {
        throw std::runtime_error("setFieldComponentValue: missing vertex DOF");
    }
    const auto index = static_cast<std::size_t>(dofs.front() + offset);
    if (index >= solution.size()) {
        throw std::runtime_error(
            "setFieldComponentValue: DOF index is out of range");
    }
    solution[index] = value;
}

[[nodiscard]] unsigned long long allreduceMinUnsigned(
    unsigned long long value,
    MPI_Comm comm)
{
    unsigned long long global = value;
    MPI_Allreduce(&value,
                  &global,
                  1,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN,
                  comm);
    return global;
}

[[nodiscard]] unsigned long long allreduceMaxUnsigned(
    unsigned long long value,
    MPI_Comm comm)
{
    unsigned long long global = value;
    MPI_Allreduce(&value,
                  &global,
                  1,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX,
                  comm);
    return global;
}

[[nodiscard]] FE::Real levelSetValueAtNode(const std::array<FE::Real, 3>& x)
{
    return x[0] * x[0] + x[1] * x[1] - FE::Real{0.25};
}

[[nodiscard]] std::uint64_t ruleSignatureHash(
    const std::vector<FE::geometry::CutQuadratureRule>& rules)
{
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto& rule : rules) {
        const auto& provenance = rule.provenance;
        mixHash(hash,
                static_cast<std::uint64_t>(provenance.parent_entity));
        mixHash(hash, static_cast<std::uint64_t>(static_cast<int>(rule.side)));
        mixStringHash(hash, provenance.implicit_quadrature_backend);
        mixStringHash(hash, provenance.selected_implicit_quadrature_backend);
        mixStringHash(hash, provenance.cut_topology_id);
        mixHash(hash, provenance.cut_topology_revision);
        mixHash(hash,
                static_cast<std::uint64_t>(
                    provenance.requested_quadrature_order));
        mixHash(hash,
                static_cast<std::uint64_t>(
                    provenance.achieved_quadrature_order));
        mixHash(hash, static_cast<std::uint64_t>(rule.points.size()));
    }
    return hash;
}

[[nodiscard]] level_set::LevelSetGeneratedInterfaceResult
buildSharedMixedQuadTriangleAutoResult(
    MPI_Comm comm,
    int rank,
    int world_size,
    bool distributed_ownership)
{
    const auto mesh =
        std::make_shared<SharedMixedQuadTriangleMeshAccess>(
            distributed_ownership ? rank : 0,
            distributed_ownership);
    auto scalar_space = std::make_shared<MixedQuadTriangleLinearH1Space>();

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    if (distributed_ownership) {
        system.setup(mpiSetupOptions(comm, rank, world_size),
                     sharedMixedQuadTriangleSetupInputs(
                         /*distributed_ownership=*/true));
    } else {
        system.setup({}, sharedMixedQuadTriangleSetupInputs(
                            /*distributed_ownership=*/false));
    }

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        const FE::Real value =
            vertex < 4 ? x[0] : x[0] - FE::Real{2.5};
        setFieldComponentValue(solution, system, phi, vertex, value);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = 902;
    options.domain_id = "water-air-auto";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::Auto;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 1;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    return lifecycle.build(system, options, solution);
}

} // namespace

TEST(LevelSetCellEvaluatorMPI, SharedOwnedAndGhostCellsUseDeterministicValues)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks";
    }

    const auto mesh = std::make_shared<SharedTriangleMeshAccess>(rank);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(
        system.setup(mpiSetupOptions(comm, rank, world_size),
                     sharedTriangleSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 6u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 6u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            levelSetValueAtNode(x);
    }

    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);
    const auto edge_midpoint = evaluator.evaluate(0, {{0.5, 0.0, 0.0}});
    EXPECT_NEAR(allreduceMin(edge_midpoint.value, comm),
                allreduceMax(edge_midpoint.value, comm),
                1.0e-14);

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = 87;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.implicit_cut_max_subdivision_depth = 5;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.domain.request().ownership_revision,
              static_cast<std::uint64_t>(31 + rank));
    EXPECT_EQ(mesh->isOwnedCell(0), rank == 0);
    EXPECT_EQ(result.achieved_interface_quadrature_order, 2);

    std::size_t root_polished_fragment_count = 0u;
    for (const auto& fragment : result.domain.fragments()) {
        if (fragment.active() && fragment.root_polished) {
            ++root_polished_fragment_count;
        }
    }
    EXPECT_GT(root_polished_fragment_count, 0u);

    std::set<std::string> interface_rule_topology_ids;
    std::uint64_t topology_hash = 1469598103934665603ull;
    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    for (const auto& rule : interface_rules) {
        EXPECT_TRUE(
            interface_rule_topology_ids.insert(rule.provenance.cut_topology_id)
                .second)
            << rule.provenance.cut_topology_id;
        EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                  "HighOrderSubcell");
        EXPECT_EQ(rule.provenance.achieved_quadrature_order, 2);
        mixStringHash(topology_hash, rule.provenance.cut_topology_id);
    }
    const auto local_topology_hash =
        static_cast<unsigned long long>(topology_hash);
    EXPECT_EQ(allreduceMinUnsigned(local_topology_hash, comm),
              allreduceMaxUnsigned(local_topology_hash, comm));

    EXPECT_NEAR(allreduceMin(result.summary.negative_volume_measure, comm),
                allreduceMax(result.summary.negative_volume_measure, comm),
                1.0e-14);
    EXPECT_NEAR(allreduceMin(result.summary.positive_volume_measure, comm),
                allreduceMax(result.summary.positive_volume_measure, comm),
                1.0e-14);
    EXPECT_NEAR(allreduceMin(result.summary.measure, comm),
                allreduceMax(result.summary.measure, comm),
                1.0e-14);
}

TEST(LevelSetCellEvaluatorMPI,
     AutoBackendMixedQuadTriangleDispatchIsRankDeterministic)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks";
    }

    const auto mesh =
        std::make_shared<SharedMixedQuadTriangleMeshAccess>(rank);
    auto scalar_space = std::make_shared<MixedQuadTriangleLinearH1Space>();

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(
        system.setup(mpiSetupOptions(comm, rank, world_size),
                     sharedMixedQuadTriangleSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        const auto x = mesh->getNodeCoordinates(vertex);
        const FE::Real value =
            vertex < 4 ? x[0] : x[0] - FE::Real{2.5};
        setFieldComponentValue(solution, system, phi, vertex, value);
    }

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = 902;
    options.domain_id = "water-air-auto";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::Auto;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 1;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.domain.request().ownership_revision,
              static_cast<std::uint64_t>(41 + rank));
    EXPECT_EQ(result.domain.request().implicit_quadrature_backend, "Auto");
    EXPECT_EQ(result.domain.request().implicit_fallback_status, "None");
    EXPECT_EQ(result.implicit_cut_fallback_cell_count, 0u);

    const auto interface_rules = result.domain.interfaceQuadratureRules();
    ASSERT_FALSE(interface_rules.empty());
    std::size_t quad_interface_rules = 0u;
    std::size_t triangle_interface_rules = 0u;
    for (const auto& rule : interface_rules) {
        EXPECT_EQ(rule.provenance.implicit_fallback_status, "None");
        if (rule.provenance.parent_entity == 0) {
            ++quad_interface_rules;
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      "SayeHyperrectangle");
            EXPECT_EQ(rule.provenance.selected_implicit_quadrature_backend,
                      "SayeHyperrectangle");
        } else {
            EXPECT_EQ(rule.provenance.parent_entity, 1);
            ++triangle_interface_rules;
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      "HighOrderSubcell");
            EXPECT_EQ(rule.provenance.selected_implicit_quadrature_backend,
                      "HighOrderSubcell");
        }
    }
    EXPECT_GT(quad_interface_rules, 0u);
    EXPECT_GT(triangle_interface_rules, 0u);

    const auto volume_rules = result.domain.volumeQuadratureRules();
    ASSERT_FALSE(volume_rules.empty());
    std::size_t quad_volume_rules = 0u;
    std::size_t triangle_volume_rules = 0u;
    for (const auto& rule : volume_rules) {
        EXPECT_EQ(rule.provenance.implicit_fallback_status, "None");
        if (rule.provenance.parent_entity == 0) {
            ++quad_volume_rules;
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      "SayeHyperrectangle");
            EXPECT_EQ(rule.provenance.selected_implicit_quadrature_backend,
                      "SayeHyperrectangle");
        } else {
            EXPECT_EQ(rule.provenance.parent_entity, 1);
            ++triangle_volume_rules;
            EXPECT_EQ(rule.provenance.implicit_quadrature_backend,
                      "HighOrderSubcell");
            EXPECT_EQ(rule.provenance.selected_implicit_quadrature_backend,
                      "HighOrderSubcell");
        }
    }
    EXPECT_GT(quad_volume_rules, 0u);
    EXPECT_GT(triangle_volume_rules, 0u);

    const auto interface_hash =
        static_cast<unsigned long long>(ruleSignatureHash(interface_rules));
    EXPECT_EQ(allreduceMinUnsigned(interface_hash, comm),
              allreduceMaxUnsigned(interface_hash, comm));
    const auto volume_hash =
        static_cast<unsigned long long>(ruleSignatureHash(volume_rules));
    EXPECT_EQ(allreduceMinUnsigned(volume_hash, comm),
              allreduceMaxUnsigned(volume_hash, comm));
    const auto active_fragment_count =
        static_cast<unsigned long long>(result.summary.active_fragment_count);
    EXPECT_EQ(allreduceMinUnsigned(active_fragment_count, comm),
              allreduceMaxUnsigned(active_fragment_count, comm));
    const auto active_volume_region_count =
        static_cast<unsigned long long>(
            result.summary.active_volume_region_count);
    EXPECT_EQ(allreduceMinUnsigned(active_volume_region_count, comm),
              allreduceMaxUnsigned(active_volume_region_count, comm));
}

TEST(LevelSetCellEvaluatorMPI,
     AutoBackendMixedQuadTriangleMeasuresMatchSerialReference)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks";
    }

    const auto serial_result =
        buildSharedMixedQuadTriangleAutoResult(
            comm,
            rank,
            world_size,
            /*distributed_ownership=*/false);
    const auto mpi_result =
        buildSharedMixedQuadTriangleAutoResult(
            comm,
            rank,
            world_size,
            /*distributed_ownership=*/true);

    ASSERT_TRUE(serial_result.success) << serial_result.diagnostic;
    ASSERT_TRUE(mpi_result.success) << mpi_result.diagnostic;
    EXPECT_EQ(serial_result.domain.request().ownership_revision, 41u);
    EXPECT_EQ(mpi_result.domain.request().ownership_revision,
              static_cast<std::uint64_t>(41 + rank));

    EXPECT_NEAR(mpi_result.summary.negative_volume_measure,
                serial_result.summary.negative_volume_measure,
                1.0e-14);
    EXPECT_NEAR(mpi_result.summary.positive_volume_measure,
                serial_result.summary.positive_volume_measure,
                1.0e-14);
    EXPECT_NEAR(mpi_result.summary.measure,
                serial_result.summary.measure,
                1.0e-14);
    EXPECT_EQ(mpi_result.summary.active_fragment_count,
              serial_result.summary.active_fragment_count);
    EXPECT_EQ(mpi_result.summary.active_volume_region_count,
              serial_result.summary.active_volume_region_count);

    const auto serial_interface_rules =
        serial_result.domain.interfaceQuadratureRules();
    const auto mpi_interface_rules =
        mpi_result.domain.interfaceQuadratureRules();
    const auto serial_volume_rules =
        serial_result.domain.volumeQuadratureRules();
    const auto mpi_volume_rules =
        mpi_result.domain.volumeQuadratureRules();
    EXPECT_EQ(mpi_interface_rules.size(), serial_interface_rules.size());
    EXPECT_EQ(mpi_volume_rules.size(), serial_volume_rules.size());
    EXPECT_EQ(ruleSignatureHash(mpi_interface_rules),
              ruleSignatureHash(serial_interface_rules));
    EXPECT_EQ(ruleSignatureHash(mpi_volume_rules),
              ruleSignatureHash(serial_volume_rules));

    EXPECT_NEAR(allreduceMin(mpi_result.summary.negative_volume_measure, comm),
                allreduceMax(mpi_result.summary.negative_volume_measure, comm),
                1.0e-14);
    EXPECT_NEAR(allreduceMin(mpi_result.summary.positive_volume_measure, comm),
                allreduceMax(mpi_result.summary.positive_volume_measure, comm),
                1.0e-14);
    EXPECT_NEAR(allreduceMin(mpi_result.summary.measure, comm),
                allreduceMax(mpi_result.summary.measure, comm),
                1.0e-14);

    const auto mpi_interface_hash =
        static_cast<unsigned long long>(
            ruleSignatureHash(mpi_interface_rules));
    const auto mpi_volume_hash =
        static_cast<unsigned long long>(ruleSignatureHash(mpi_volume_rules));
    EXPECT_EQ(allreduceMinUnsigned(mpi_interface_hash, comm),
              allreduceMaxUnsigned(mpi_interface_hash, comm));
    EXPECT_EQ(allreduceMinUnsigned(mpi_volume_hash, comm),
              allreduceMaxUnsigned(mpi_volume_hash, comm));
}
