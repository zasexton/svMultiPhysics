/**
 * @file test_SetupStoragePlan.cpp
 * @brief Unit tests for physics-agnostic FE setup storage planning.
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"

#include "Assembly/AssemblyKernel.h"
#include "Constraints/VertexDirichletConstraint.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Spaces/MixedSpace.h"

#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellShape.h"

#include <memory>
#include <string>
#include <vector>

namespace {

using svmp::CellFamily;
using svmp::CellShape;
using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;
using svmp::FE::assembly::AssemblyContext;
using svmp::FE::assembly::BilinearFormKernel;
using svmp::FE::assembly::KernelOutput;
using svmp::FE::assembly::RequiredData;
using svmp::FE::constraints::VertexDirichletConstraint;
using svmp::FE::constraints::VertexDirichletValue;
using svmp::FE::spaces::H1Space;
using svmp::FE::spaces::L2Space;
using svmp::FE::spaces::MixedSpace;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;

std::shared_ptr<svmp::Mesh> buildQuadMeshWithBoundaryFaces()
{
    auto base = std::make_shared<svmp::MeshBase>();

    const std::vector<svmp::real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0
    };
    const std::vector<svmp::offset_t> c2v_offsets = {0, 4};
    const std::vector<svmp::index_t> c2v = {0, 1, 2, 3};

    CellShape cell{};
    cell.family = CellFamily::Quad;
    cell.num_corners = 4;
    cell.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, x_ref, c2v_offsets, c2v, {cell});

    CellShape edge{};
    edge.family = CellFamily::Line;
    edge.num_corners = 2;
    edge.order = 1;
    base->set_faces_from_arrays(
        {edge, edge, edge, edge},
        {0, 2, 4, 6, 8},
        {0, 1, 1, 2, 2, 3, 3, 0},
        {{{0, svmp::INVALID_INDEX}},
         {{0, svmp::INVALID_INDEX}},
         {{0, svmp::INVALID_INDEX}},
         {{0, svmp::INVALID_INDEX}}});
    for (svmp::index_t f = 0; f < 4; ++f) {
        base->set_boundary_label(f, 7);
    }
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> buildQuadMeshWithoutDerivedTopology()
{
    auto base = std::make_shared<svmp::MeshBase>();

    const std::vector<svmp::real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0
    };
    const std::vector<svmp::offset_t> c2v_offsets = {0, 4};
    const std::vector<svmp::index_t> c2v = {0, 1, 2, 3};

    CellShape cell{};
    cell.family = CellFamily::Quad;
    cell.num_corners = 4;
    cell.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, x_ref, c2v_offsets, c2v, {cell});
    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> buildTwoQuadMeshWithBoundaryOnlyFaceMetadata()
{
    auto base = std::make_shared<svmp::MeshBase>();

    const std::vector<svmp::real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0
    };
    const std::vector<svmp::offset_t> c2v_offsets = {0, 4, 8};
    const std::vector<svmp::index_t> c2v = {
        0, 1, 4, 3,
        1, 2, 5, 4
    };

    CellShape cell{};
    cell.family = CellFamily::Quad;
    cell.num_corners = 4;
    cell.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, x_ref, c2v_offsets, c2v, {cell, cell});

    svmp::MeshFinalizeOptions options;
    options.codim1_storage = svmp::MeshCodim1StorageMode::BoundaryOnly;
    base->finalize(options);

    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
        base->set_boundary_label(f, 7);
        base->add_to_set(svmp::EntityKind::Face, "exterior", f);
    }

    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<H1Space> h1(int order)
{
    return std::make_shared<H1Space>(ElementType::Quad4, order);
}

std::shared_ptr<L2Space> l2(int order)
{
    return std::make_shared<L2Space>(ElementType::Quad4, order);
}

class CellOnlyKernel : public BilinearFormKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const override
    {
        return RequiredData::IntegrationWeights | RequiredData::BasisValues;
    }

    void computeCell(const AssemblyContext&, KernelOutput&) override {}
};

class BoundaryOnlyKernel final : public CellOnlyKernel {
public:
    [[nodiscard]] bool hasCell() const noexcept override { return false; }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }
};

class InteriorFaceOnlyKernel final : public CellOnlyKernel {
public:
    [[nodiscard]] bool hasCell() const noexcept override { return false; }
    [[nodiscard]] bool hasInteriorFace() const noexcept override { return true; }
};

FieldId addScalarField(FESystem& system, int order = 1, const char* name = "u")
{
    FieldSpec spec;
    spec.name = name;
    spec.space = h1(order);
    spec.components = 1;
    return system.addField(std::move(spec));
}

FieldId addField(FESystem& system,
                 std::shared_ptr<svmp::FE::spaces::FunctionSpace> space,
                 const char* name = "u")
{
    FieldSpec spec;
    spec.name = name;
    spec.components = space->value_dimension();
    spec.space = std::move(space);
    return system.addField(std::move(spec));
}

} // namespace

TEST(SetupStoragePlan, CellOnlyP1DoesNotRequireFaceOrEdgeTopology)
{
    FESystem system(buildQuadMeshWithBoundaryFaces());
    const auto u = addScalarField(system);
    system.addCellKernel("mass", u, std::make_shared<CellOnlyKernel>());

    const auto plan = system.computeSetupStoragePlan();

    EXPECT_FALSE(plan.requirements.edge_topology);
    EXPECT_FALSE(plan.requirements.boundary_face_topology);
    EXPECT_FALSE(plan.requirements.interior_face_topology);
    EXPECT_FALSE(plan.requirements.full_face_topology());
    EXPECT_TRUE(plan.can_alias_single_field_dof_map);
}

TEST(SetupStoragePlan, HigherOrderH1RequiresOnlyEntityTopologyUsedByDofs)
{
    FESystem system(buildQuadMeshWithBoundaryFaces());
    const auto u = addScalarField(system, /*order=*/2);
    system.addCellKernel("mass", u, std::make_shared<CellOnlyKernel>());

    const auto plan = system.computeSetupStoragePlan();

    EXPECT_TRUE(plan.requirements.edge_topology);
    EXPECT_FALSE(plan.requirements.interior_face_topology);
}

TEST(SetupStoragePlan, L2CellSpaceDoesNotRequireSharedEntityTopology)
{
    FESystem system(buildQuadMeshWithBoundaryFaces());
    const auto u = addField(system, l2(/*order=*/2));
    system.addCellKernel("mass", u, std::make_shared<CellOnlyKernel>());

    const auto plan = system.computeSetupStoragePlan();

    EXPECT_TRUE(plan.requirements.vertex_topology);
    EXPECT_TRUE(plan.requirements.cell_topology);
    EXPECT_FALSE(plan.requirements.edge_topology);
    EXPECT_FALSE(plan.requirements.boundary_face_topology);
    EXPECT_FALSE(plan.requirements.interior_face_topology);
}

TEST(SetupStoragePlan, MixedSpaceMergesComponentRequirements)
{
    auto mixed = std::make_shared<MixedSpace>();
    mixed->add_component("velocity", h1(/*order=*/2));
    mixed->add_component("pressure", l2(/*order=*/1));

    FESystem system(buildQuadMeshWithBoundaryFaces());
    const auto u = addField(system, std::move(mixed));
    system.addCellKernel("mixed_mass", u, std::make_shared<CellOnlyKernel>());

    const auto plan = system.computeSetupStoragePlan();

    EXPECT_TRUE(plan.requirements.edge_topology);
    EXPECT_FALSE(plan.requirements.boundary_face_topology);
    EXPECT_FALSE(plan.requirements.interior_face_topology);
}

TEST(SetupStoragePlan, BoundaryAndInteriorFaceOperatorsSetExactFaceNeeds)
{
    {
        FESystem system(buildQuadMeshWithBoundaryFaces());
        const auto u = addScalarField(system);
        system.addBoundaryKernel("boundary", 7, u, std::make_shared<BoundaryOnlyKernel>());

        const auto plan = system.computeSetupStoragePlan();
        EXPECT_TRUE(plan.requirements.boundary_face_topology);
        EXPECT_FALSE(plan.requirements.interior_face_topology);
    }

    {
        FESystem system(buildQuadMeshWithBoundaryFaces());
        const auto u = addScalarField(system);
        system.addInteriorFaceKernel("interior", u, std::make_shared<InteriorFaceOnlyKernel>());

        const auto plan = system.computeSetupStoragePlan();
        EXPECT_TRUE(plan.requirements.interior_face_topology);
        EXPECT_TRUE(plan.requirements.full_face_topology());
    }
}

TEST(SetupStoragePlan, VertexConstraintRequestsOnlyGlobalVertexLookup)
{
    FESystem system(buildQuadMeshWithBoundaryFaces());
    const auto u = addScalarField(system);
    system.addSystemConstraint(std::make_unique<VertexDirichletConstraint>(
        u,
        std::vector<VertexDirichletValue>{{GlobalIndex{2}, Real{3.0}}}));

    const auto plan = system.computeSetupStoragePlan();

    EXPECT_TRUE(plan.requirements.entity_dof_map);
    EXPECT_TRUE(plan.requirements.vertex_gids);
    EXPECT_TRUE(plan.requirements.global_vertex_lookup);
    EXPECT_FALSE(plan.requirements.global_face_lookup);
    EXPECT_FALSE(plan.requirements.boundary_face_topology);
}

TEST(SetupStoragePlan, SingleFieldSetupAliasesMonolithicDofMap)
{
    FESystem system(buildQuadMeshWithBoundaryFaces());
    const auto u = addScalarField(system);
    system.addCellKernel("mass", u, std::make_shared<CellOnlyKernel>());

    system.setup();

    EXPECT_TRUE(system.setupStoragePlan().uses_single_field_alias);
    EXPECT_EQ(&system.dofHandler().getDofMap(),
              &system.fieldDofHandler(u).getDofMap());
}

TEST(SetupStoragePlan, MultiFieldSetupDoesNotAliasMonolithicDofMap)
{
    FESystem system(buildQuadMeshWithBoundaryFaces());
    const auto u = addScalarField(system, 1, "u");
    const auto p = addScalarField(system, 1, "p");
    system.addCellKernel("u_mass", u, std::make_shared<CellOnlyKernel>());
    system.addCellKernel("p_mass", p, std::make_shared<CellOnlyKernel>());

    system.setup();

    EXPECT_FALSE(system.setupStoragePlan().uses_single_field_alias);
    EXPECT_NE(&system.dofHandler().getDofMap(),
              &system.fieldDofHandler(u).getDofMap());
}

TEST(SetupStoragePlan, SetupDoesNotMaterializeUnusedMeshTopology)
{
    auto mesh = buildQuadMeshWithoutDerivedTopology();
    FESystem system(mesh);
    const auto u = addScalarField(system);
    system.addCellKernel("mass", u, std::make_shared<CellOnlyKernel>());

    system.setup();

    EXPECT_FALSE(system.setupStoragePlan().requirements.edge_topology);
    EXPECT_FALSE(system.setupStoragePlan().requirements.boundary_face_topology);
    EXPECT_FALSE(system.setupStoragePlan().requirements.interior_face_topology);
    EXPECT_EQ(mesh->base().n_faces(), 0u);
    EXPECT_EQ(mesh->base().n_edges(), 0u);
}

TEST(SetupStoragePlan, SetupMaterializesOnlyPlannedEdgeTopology)
{
    auto mesh = buildQuadMeshWithoutDerivedTopology();
    FESystem system(mesh);
    const auto u = addScalarField(system, /*order=*/2);
    system.addCellKernel("mass", u, std::make_shared<CellOnlyKernel>());

    system.setup();

    EXPECT_TRUE(system.setupStoragePlan().requirements.edge_topology);
    EXPECT_FALSE(system.setupStoragePlan().requirements.boundary_face_topology);
    EXPECT_FALSE(system.setupStoragePlan().requirements.interior_face_topology);
    EXPECT_EQ(mesh->base().codim1_storage_mode(), svmp::MeshCodim1StorageMode::Full);
    EXPECT_EQ(mesh->base().n_edges(), 4u);
    EXPECT_EQ(mesh->base().n_faces(), 4u);
}

TEST(SetupStoragePlan, SetupMaterializesBoundaryOnlyTopologyForBoundaryWork)
{
    auto mesh = buildQuadMeshWithoutDerivedTopology();
    FESystem system(mesh);
    const auto u = addScalarField(system);
    system.addBoundaryKernel("boundary", 7, u, std::make_shared<BoundaryOnlyKernel>());

    system.setup();

    EXPECT_TRUE(system.setupStoragePlan().requirements.boundary_face_topology);
    EXPECT_FALSE(system.setupStoragePlan().requirements.interior_face_topology);
    EXPECT_EQ(mesh->base().codim1_storage_mode(), svmp::MeshCodim1StorageMode::BoundaryOnly);
    EXPECT_EQ(mesh->base().n_faces(), 4u);
}

TEST(SetupStoragePlan, SetupUpgradesBoundaryOnlyTopologyForInteriorFaceWork)
{
    auto mesh = buildTwoQuadMeshWithBoundaryOnlyFaceMetadata();
    ASSERT_EQ(mesh->base().codim1_storage_mode(), svmp::MeshCodim1StorageMode::BoundaryOnly);
    ASSERT_EQ(mesh->base().n_faces(), 6u);

    FESystem system(mesh);
    const auto u = addScalarField(system);
    system.addInteriorFaceKernel("interior", u, std::make_shared<InteriorFaceOnlyKernel>());

    ASSERT_NO_THROW(system.setup());

    EXPECT_TRUE(system.setupStoragePlan().requirements.interior_face_topology);
    EXPECT_EQ(mesh->base().codim1_storage_mode(), svmp::MeshCodim1StorageMode::Full);
    EXPECT_EQ(mesh->base().n_faces(), 7u);

    std::size_t labelled_faces = 0;
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh->base().n_faces()); ++f) {
        if (mesh->base().boundary_label(f) == 7) {
            ++labelled_faces;
        }
    }
    EXPECT_EQ(labelled_faces, 6u);

    const auto& exterior = mesh->base().get_set(svmp::EntityKind::Face, "exterior");
    EXPECT_EQ(exterior.size(), 6u);
    for (const auto face : exterior) {
        EXPECT_EQ(mesh->base().boundary_label(face), 7);
    }
}

TEST(SetupStoragePlan, SummaryReportsDecidedStorageStructures)
{
    FESystem system(buildQuadMeshWithoutDerivedTopology());
    const auto u = addScalarField(system);
    system.addCellKernel("mass", u, std::make_shared<CellOnlyKernel>());

    system.setup();

    const auto summary = system.setupStoragePlan().summary();
    EXPECT_NE(summary.find("edges=off"), std::string::npos);
    EXPECT_NE(summary.find("boundary_faces=off"), std::string::npos);
    EXPECT_NE(summary.find("interior_faces=off"), std::string::npos);
    EXPECT_NE(summary.find("monolithic_alias=on"), std::string::npos);
}

TEST(SetupStoragePlan, MeshGlobalLookupMapsAreBuiltOnlyOnDemand)
{
    auto mesh = buildQuadMeshWithBoundaryFaces();
    auto& base = mesh->base();

    EXPECT_FALSE(base.has_global_lookup_map(svmp::EntityKind::Vertex));
    EXPECT_FALSE(base.has_global_lookup_map(svmp::EntityKind::Face));

    EXPECT_EQ(base.global_to_local_vertex(2), 2);
    EXPECT_TRUE(base.has_global_lookup_map(svmp::EntityKind::Vertex));
    EXPECT_EQ(base.global_lookup_map_size(svmp::EntityKind::Vertex), 4u);
    EXPECT_FALSE(base.has_global_lookup_map(svmp::EntityKind::Face));

    EXPECT_EQ(base.global_to_local_face(1), 1);
    EXPECT_TRUE(base.has_global_lookup_map(svmp::EntityKind::Face));
    EXPECT_EQ(base.global_lookup_map_size(svmp::EntityKind::Face), 4u);
}
