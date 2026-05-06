/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyKernel.h"
#include "Assembly/CompositeMeshAccess.h"
#include "Assembly/GlobalSystemView.h"
#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"

#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellShape.h"

#include "Spaces/H1Space.h"

#include <memory>
#include <vector>

namespace {

std::shared_ptr<svmp::Mesh> buildQuadMesh(double x_offset)
{
    auto base = std::make_shared<svmp::MeshBase>();

    const std::vector<svmp::real_t> coordinates = {
        x_offset + 0.0, 0.0,
        x_offset + 1.0, 0.0,
        x_offset + 1.0, 1.0,
        x_offset + 0.0, 1.0,
    };
    const std::vector<svmp::offset_t> cell_to_vertex_offsets = {0, 4};
    const std::vector<svmp::index_t> cell_to_vertex = {0, 1, 2, 3};

    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2,
                            coordinates,
                            cell_to_vertex_offsets,
                            cell_to_vertex,
                            {shape});
    base->finalize();
    for (svmp::index_t face = 0; face < static_cast<svmp::index_t>(base->n_faces()); ++face) {
        base->set_boundary_label(face, 5);
    }

    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::FE::spaces::H1Space> quadScalarSpace()
{
    return std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4,
                                                       /*order=*/1);
}

} // namespace

TEST(FESystemMultiMesh, CompositeSetupWithNoPhysicsModules)
{
    auto left_mesh = buildQuadMesh(0.0);
    auto right_mesh = buildQuadMesh(2.0);

    auto access = std::make_shared<svmp::FE::assembly::CompositeMeshAccess>(
        std::vector<svmp::FE::assembly::CompositeMeshParticipant>{
            {.name = "left", .mesh = left_mesh.get(), .domain_id = 10},
            {.name = "right", .mesh = right_mesh.get(), .domain_id = 20},
        });

    svmp::FE::systems::FESystem system(access);
    EXPECT_FALSE(system.hasSingleNativeMesh());
    EXPECT_EQ(system.mesh(), nullptr);
    EXPECT_TRUE(system.hasCompositeMeshAccess());
    EXPECT_EQ(system.meshAccess().numCells(), 2);
    EXPECT_EQ(system.meshAccess().numVertices(), 8);

    ASSERT_EQ(system.meshParticipants().size(), 2u);
    EXPECT_EQ(system.meshParticipants()[0].name, "left");
    EXPECT_EQ(system.meshParticipants()[0].domain_id, 10);
    EXPECT_EQ(system.meshParticipants()[0].cell_offset, 0);
    EXPECT_EQ(system.meshParticipants()[0].num_cells, 1);
    EXPECT_EQ(system.meshParticipants()[1].name, "right");
    EXPECT_EQ(system.meshParticipants()[1].domain_id, 20);
    EXPECT_EQ(system.meshParticipants()[1].cell_offset, 1);
    EXPECT_EQ(system.meshParticipants()[1].num_cells, 1);

    ASSERT_NE(system.meshParticipantByName("left"), nullptr);
    EXPECT_EQ(system.meshParticipantByName("left")->domain_id, 10);
    ASSERT_NE(system.meshParticipantByDomain(20), nullptr);
    EXPECT_EQ(system.meshParticipantByDomain(20)->name, "right");
    ASSERT_NE(system.meshParticipantForCell(0), nullptr);
    EXPECT_EQ(system.meshParticipantForCell(0)->name, "left");
    ASSERT_NE(system.meshParticipantForCell(1), nullptr);
    EXPECT_EQ(system.meshParticipantForCell(1)->name, "right");

    const auto pressure = system.addField(svmp::FE::systems::FieldSpec{
        .name = "pressure",
        .space = quadScalarSpace(),
        .components = 1});
    system.setup();

    EXPECT_TRUE(system.isSetup());
    EXPECT_EQ(system.dofHandler().getNumDofs(), 8);
    EXPECT_EQ(system.fieldDofHandler(pressure).getNumDofs(), 8);
}

TEST(FESystemMultiMesh, SingleMeshSetupStillUsesNativeMesh)
{
    auto mesh = buildQuadMesh(0.0);

    svmp::FE::systems::MeshParticipantInfo participant{};
    participant.name = "primary";
    participant.domain_id = 7;
    svmp::FE::systems::FESystem system(mesh, std::move(participant));

    EXPECT_TRUE(system.hasSingleNativeMesh());
    EXPECT_EQ(system.mesh(), mesh.get());
    EXPECT_FALSE(system.hasCompositeMeshAccess());
    ASSERT_EQ(system.meshParticipants().size(), 1u);
    EXPECT_EQ(system.meshParticipants()[0].name, "primary");
    EXPECT_EQ(system.meshParticipants()[0].domain_id, 7);
    ASSERT_NE(system.meshParticipantForCell(0), nullptr);
    EXPECT_EQ(system.meshParticipantForCell(0)->name, "primary");
    EXPECT_EQ(&system.singleMesh("test"), mesh.get());

    const auto pressure = system.addField(svmp::FE::systems::FieldSpec{
        .name = "pressure",
        .space = quadScalarSpace(),
        .components = 1});
    system.setup();

    EXPECT_TRUE(system.isSetup());
    EXPECT_EQ(system.dofHandler().getNumDofs(), 4);
    EXPECT_EQ(system.fieldDofHandler(pressure).getNumDofs(), 4);
}

TEST(FESystemMultiMesh, NativeMeshOnlyApiReportsCompositeAccess)
{
    auto left_mesh = buildQuadMesh(0.0);
    auto right_mesh = buildQuadMesh(2.0);

    auto access = std::make_shared<svmp::FE::assembly::CompositeMeshAccess>(
        std::vector<svmp::FE::assembly::CompositeMeshParticipant>{
            {.name = "left", .mesh = left_mesh.get()},
            {.name = "right", .mesh = right_mesh.get()},
        });

    svmp::FE::systems::FESystem system(access);
    EXPECT_THROW((void)system.singleMesh("native mesh only"),
                 svmp::FE::systems::InvalidStateException);
}

TEST(FESystemMultiMesh, ParticipantScopedFieldsAllocateOnlySelectedParticipants)
{
    auto lumen_mesh = buildQuadMesh(0.0);
    auto wall_mesh = buildQuadMesh(2.0);

    auto access = std::make_shared<svmp::FE::assembly::CompositeMeshAccess>(
        std::vector<svmp::FE::assembly::CompositeMeshParticipant>{
            {.name = "lumen", .mesh = lumen_mesh.get(), .domain_id = 10},
            {.name = "wall", .mesh = wall_mesh.get(), .domain_id = 20},
        });

    svmp::FE::systems::FESystem system(access);
    const auto lumen_pressure = system.addField(svmp::FE::systems::FieldSpec{
        .name = "lumen_pressure",
        .space = quadScalarSpace(),
        .components = 1,
        .participant_name = std::string("lumen")});
    const auto wall_displacement = system.addField(svmp::FE::systems::FieldSpec{
        .name = "wall_displacement",
        .space = quadScalarSpace(),
        .components = 1,
        .participant_name = std::string("wall")});

    system.setup();

    ASSERT_NE(system.fieldMeshParticipant(lumen_pressure), nullptr);
    EXPECT_EQ(system.fieldMeshParticipant(lumen_pressure)->name, "lumen");
    ASSERT_NE(system.fieldMeshParticipant(wall_displacement), nullptr);
    EXPECT_EQ(system.fieldMeshParticipant(wall_displacement)->name, "wall");

    const auto& lumen_map = system.fieldDofHandler(lumen_pressure).getDofMap();
    const auto& wall_map = system.fieldDofHandler(wall_displacement).getDofMap();
    EXPECT_EQ(system.fieldDofHandler(lumen_pressure).getNumDofs(), 4);
    EXPECT_EQ(system.fieldDofHandler(wall_displacement).getNumDofs(), 4);
    EXPECT_EQ(system.dofHandler().getNumDofs(), 8);

    EXPECT_EQ(lumen_map.getCellDofs(0).size(), 4u);
    EXPECT_TRUE(lumen_map.getCellDofs(1).empty());
    EXPECT_TRUE(wall_map.getCellDofs(0).empty());
    EXPECT_EQ(wall_map.getCellDofs(1).size(), 4u);

    EXPECT_TRUE(system.fieldActiveOnCell(lumen_pressure, 0));
    EXPECT_FALSE(system.fieldActiveOnCell(lumen_pressure, 1));
    EXPECT_FALSE(system.fieldActiveOnCell(wall_displacement, 0));
    EXPECT_TRUE(system.fieldActiveOnCell(wall_displacement, 1));
}

TEST(FESystemMultiMesh, ParticipantScopedAssemblyRunsUncoupledTerms)
{
    auto lumen_mesh = buildQuadMesh(0.0);
    auto wall_mesh = buildQuadMesh(2.0);

    auto access = std::make_shared<svmp::FE::assembly::CompositeMeshAccess>(
        std::vector<svmp::FE::assembly::CompositeMeshParticipant>{
            {.name = "lumen", .mesh = lumen_mesh.get(), .domain_id = 10},
            {.name = "wall", .mesh = wall_mesh.get(), .domain_id = 20},
        });

    svmp::FE::systems::FESystem system(access);
    const auto lumen_pressure = system.addField(svmp::FE::systems::FieldSpec{
        .name = "lumen_pressure",
        .space = quadScalarSpace(),
        .components = 1,
        .participant_name = std::string("lumen")});
    const auto wall_temperature = system.addField(svmp::FE::systems::FieldSpec{
        .name = "wall_temperature",
        .space = quadScalarSpace(),
        .components = 1,
        .participant_name = std::string("wall")});

    system.addOperator("mass");
    system.addCellKernel(
        "mass",
        lumen_pressure,
        std::make_shared<svmp::FE::assembly::MassKernel>(1.0));
    system.addCellKernel(
        "mass",
        wall_temperature,
        std::make_shared<svmp::FE::assembly::MassKernel>(2.0));
    system.setup();

    svmp::FE::assembly::DenseMatrixView matrix(system.dofHandler().getNumDofs());
    svmp::FE::systems::SystemStateView state;
    const auto result = system.assembleMass(state, matrix);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.elements_assembled, 2);

    for (svmp::FE::GlobalIndex row = 0; row < 4; ++row) {
        EXPECT_GT(matrix.getMatrixEntry(row, row), 0.0);
        EXPECT_EQ(matrix.getMatrixEntry(row, row + 4), 0.0);
        EXPECT_EQ(matrix.getMatrixEntry(row + 4, row), 0.0);
        EXPECT_GT(matrix.getMatrixEntry(row + 4, row + 4), 0.0);
    }
}

TEST(FESystemMultiMesh, BoundaryMarkerMismatchIsRejectedForScopedField)
{
    auto lumen_mesh = buildQuadMesh(0.0);
    auto wall_mesh = buildQuadMesh(2.0);

    auto access = std::make_shared<svmp::FE::assembly::CompositeMeshAccess>(
        std::vector<svmp::FE::assembly::CompositeMeshParticipant>{
            {.name = "lumen", .mesh = lumen_mesh.get(), .domain_id = 10},
            {.name = "wall", .mesh = wall_mesh.get(), .domain_id = 20},
        });
    const int wall_boundary = access->globalBoundaryMarker("wall", 5);

    svmp::FE::systems::FESystem system(access);
    const auto lumen_pressure = system.addField(svmp::FE::systems::FieldSpec{
        .name = "lumen_pressure",
        .space = quadScalarSpace(),
        .components = 1,
        .participant_name = std::string("lumen")});
    system.addOperator("boundary");
    system.addBoundaryKernel(
        "boundary",
        wall_boundary,
        lumen_pressure,
        std::make_shared<svmp::FE::assembly::FacetMassKernel>(1.0));

    EXPECT_THROW(system.setup(), svmp::FE::InvalidArgumentException);
}
