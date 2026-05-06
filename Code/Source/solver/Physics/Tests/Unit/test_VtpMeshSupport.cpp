/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string_view>
#include <vector>

#include "Physics/Tests/Unit/PhysicsTestHelpers.h"

#include "FE/Forms/Vocabulary.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

namespace svmp {
namespace Physics {
namespace test {

#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)

TEST(VtpMeshSupport, RequiresFeWithMesh)
{
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
}

#else

namespace {

struct ExpectedSurfaceData {
    std::string_view name;
    std::size_t cells;
};

struct ExpectedParticipantMeshData {
    std::string_view variant;
    std::string_view participant;
    std::size_t points;
    std::size_t tetrahedra;
    std::vector<ExpectedSurfaceData> surfaces;
};

svmp::MeshBase loadVtkMeshBase(const std::filesystem::path& path,
                               std::string_view format)
{
    svmp::MeshIOOptions opts;
    opts.format = std::string(format);
    opts.path = path.string();
    opts.kv["force_min_dim"] = "3";
    opts.kv["codim1_topology"] = "none";
    opts.kv["edge_topology"] = "false";
    return svmp::MeshBase::load(opts);
}

} // namespace

TEST(VtpMeshSupport, LoadsSquareMeshAndAssemblesPoisson)
{
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    const auto mesh = loadSquareMeshWithMarkedBoundaries();
    ASSERT_TRUE(mesh);
    EXPECT_EQ(mesh->dim(), 2);
    EXPECT_GT(mesh->n_cells(), 0u);
    EXPECT_GT(mesh->n_vertices(), 0u);
    EXPECT_GT(mesh->n_faces(), 0u);

    const auto& base = mesh->base();
    const auto left = base.label_from_name("left");
    const auto right = base.label_from_name("right");
    const auto bottom = base.label_from_name("bottom");
    const auto top = base.label_from_name("top");

    ASSERT_NE(left, svmp::INVALID_LABEL);
    ASSERT_NE(right, svmp::INVALID_LABEL);
    ASSERT_NE(bottom, svmp::INVALID_LABEL);
    ASSERT_NE(top, svmp::INVALID_LABEL);

    EXPECT_EQ(base.faces_with_label(left).size(), 64u);
    EXPECT_EQ(base.faces_with_label(right).size(), 64u);
    EXPECT_EQ(base.faces_with_label(bottom).size(), 64u);
    EXPECT_EQ(base.faces_with_label(top).size(), 64u);

    FE::systems::FESystem system(mesh);

    auto space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Quad4, /*order=*/1);

    FE::systems::FieldSpec spec;
    spec.name = "u";
    spec.space = space;
    const FE::FieldId u_id = system.addField(std::move(spec));

    system.addOperator("residual");
    system.addOperator("jacobian");

    using namespace svmp::FE::forms;
    auto u = FormExpr::stateField(u_id, *space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    FE::systems::installFormulation(system, "residual", {u_id}, residual);
    FE::systems::installFormulation(system, "jacobian", {u_id}, residual);

    EXPECT_NO_THROW(system.setup());

    const auto ndofs = system.dofHandler().getNumDofs();
    EXPECT_EQ(static_cast<std::size_t>(ndofs), mesh->n_vertices());

    std::vector<FE::Real> uvec(mesh->n_vertices());
    for (std::size_t i = 0; i < uvec.size(); ++i) {
        uvec[i] = static_cast<FE::Real>(0.01 * std::sin(static_cast<double>(i)));
    }

    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(uvec);

    FE::assembly::DenseVectorView r(ndofs);
    r.zero();
    const auto res = system.assembleResidual(state, r);
    ASSERT_TRUE(res.success) << res.error_message;
    EXPECT_EQ(r.numRows(), ndofs);

    FE::assembly::DenseMatrixView J(ndofs);
    J.zero();
    const auto jac = system.assembleJacobian(state, J);
    ASSERT_TRUE(jac.success) << jac.error_message;
    EXPECT_EQ(J.numRows(), ndofs);
    EXPECT_EQ(J.numCols(), ndofs);
#  endif
}

TEST(VtpMeshSupport, LoadsElasticPipeFsiMeshes)
{
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    const std::array<ExpectedParticipantMeshData, 4> expected = {{
        ExpectedParticipantMeshData{
            .variant = "coarse",
            .participant = "fluid",
            .points = 325,
            .tetrahedra = 1344,
            .surfaces = {
                {"inlet", 112},
                {"outlet", 112},
                {"fsi_interface", 128},
            },
        },
        ExpectedParticipantMeshData{
            .variant = "coarse",
            .participant = "solid",
            .points = 160,
            .tetrahedra = 391,
            .surfaces = {
                {"solid_inlet", 32},
                {"solid_outlet", 32},
                {"fsi_interface", 128},
                {"outer_wall", 128},
            },
        },
        ExpectedParticipantMeshData{
            .variant = "refined",
            .participant = "fluid",
            .points = 1089,
            .tetrahedra = 5184,
            .surfaces = {
                {"inlet", 216},
                {"outlet", 216},
                {"fsi_interface", 384},
            },
        },
        ExpectedParticipantMeshData{
            .variant = "refined",
            .participant = "solid",
            .points = 432,
            .tetrahedra = 1163,
            .surfaces = {
                {"solid_inlet", 48},
                {"solid_outlet", 48},
                {"fsi_interface", 384},
                {"outer_wall", 384},
            },
        },
    }};

    for (const auto& mesh_data : expected) {
        const auto mesh_path = elasticPipeParticipantMeshVtuPath(
            mesh_data.variant,
            mesh_data.participant);
        ASSERT_TRUE(std::filesystem::exists(mesh_path)) << mesh_path;

        const auto mesh = loadVtkMeshBase(mesh_path, "vtu");
        EXPECT_EQ(mesh.dim(), 3) << mesh_path;
        EXPECT_EQ(mesh.n_vertices(), mesh_data.points) << mesh_path;
        EXPECT_EQ(mesh.n_cells(), mesh_data.tetrahedra) << mesh_path;

        for (const auto& surface_data : mesh_data.surfaces) {
            const auto surface_path = elasticPipeParticipantSurfaceVtpPath(
                mesh_data.variant,
                mesh_data.participant,
                surface_data.name);
            ASSERT_TRUE(std::filesystem::exists(surface_path)) << surface_path;

            const auto surface = loadVtkMeshBase(surface_path, "vtp");
            EXPECT_EQ(surface.dim(), 3) << surface_path;
            EXPECT_EQ(surface.n_cells(), surface_data.cells) << surface_path;
        }
    }
#  endif
}

#endif // SVMP_FE_WITH_MESH

} // namespace test
} // namespace Physics
} // namespace svmp
