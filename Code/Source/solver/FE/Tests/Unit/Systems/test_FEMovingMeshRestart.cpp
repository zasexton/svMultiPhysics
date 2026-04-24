/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/IO/MovingMeshRestart.h"
#include "Mesh/Mesh.h"
#include "Mesh/Motion/MotionFields.h"
#include "Mesh/Topology/CellShape.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;
using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;
using svmp::FE::assembly::DenseMatrixView;
using svmp::FE::assembly::MassKernel;
using svmp::FE::spaces::H1Space;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::SystemStateView;

std::string unique_restart_path()
{
    const auto stamp =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return (std::filesystem::temp_directory_path() /
            ("svmp_fe_moving_mesh_restart_" + std::to_string(static_cast<long long>(stamp)) + ".mmrst"))
        .string();
}

MeshBase build_current_quad_mesh()
{
    MeshBase mesh(2);
    const std::vector<svmp::real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
    };
    const std::vector<svmp::offset_t> offsets = {0, 4};
    const std::vector<svmp::index_t> conn = {0, 1, 2, 3};
    const std::vector<CellShape> shapes = {{CellFamily::Quad, 4, 1}};
    mesh.build_from_arrays(2, x_ref, offsets, conn, shapes);
    mesh.finalize();

    const auto handles = svmp::motion::attach_motion_fields(mesh, 2);
    auto* displacement = mesh.field_data_as<svmp::real_t>(handles.displacement);
    for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
        displacement[2 * v + 0] = x_ref[2 * v + 0];
        displacement[2 * v + 1] = 0.0;
    }

    std::vector<svmp::real_t> x_cur = x_ref;
    for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
        x_cur[2 * v + 0] *= 2.0;
    }
    mesh.set_current_coords(x_cur);
    mesh.use_current_configuration();
    return mesh;
}

DenseMatrixView assemble_current_mass(std::shared_ptr<Mesh> mesh)
{
    auto space = std::make_shared<H1Space>(ElementType::Quad4, 1);
    FESystem sys(std::move(mesh), svmp::Configuration::Current);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    sys.setup();

    DenseMatrixView mass(sys.dofHandler().getNumDofs());
    SystemStateView state;
    EXPECT_TRUE(sys.assembleMass(state, mass).success);
    return mass;
}

} // namespace

TEST(FEMovingMeshRestart, RestartedCurrentCoordinatesAreVisibleBeforeAssemblyCachesBuild)
{
    auto original_mesh_base = build_current_quad_mesh();
    const auto path = unique_restart_path();

    svmp::moving_mesh_restart::write(original_mesh_base, path);
    auto restarted_mesh_base = svmp::moving_mesh_restart::read(path);

    ASSERT_TRUE(restarted_mesh_base.has_current_coords());
    EXPECT_EQ(restarted_mesh_base.active_configuration(), svmp::Configuration::Current);
    ASSERT_EQ(restarted_mesh_base.X_cur().size(), original_mesh_base.X_cur().size());
    for (std::size_t i = 0; i < original_mesh_base.X_cur().size(); ++i) {
        EXPECT_NEAR(restarted_mesh_base.X_cur()[i], original_mesh_base.X_cur()[i], 1.0e-12);
    }

    auto original_mesh =
        svmp::create_mesh(std::make_shared<MeshBase>(std::move(original_mesh_base)));
    auto restarted_mesh =
        svmp::create_mesh(std::make_shared<MeshBase>(std::move(restarted_mesh_base)));

    const auto reference_mass = assemble_current_mass(std::move(original_mesh));
    const auto restarted_mass = assemble_current_mass(std::move(restarted_mesh));

    ASSERT_EQ(reference_mass.numRows(), restarted_mass.numRows());
    ASSERT_EQ(reference_mass.numCols(), restarted_mass.numCols());
    for (GlobalIndex i = 0; i < reference_mass.numRows(); ++i) {
        for (GlobalIndex j = 0; j < reference_mass.numCols(); ++j) {
            EXPECT_NEAR(restarted_mass.getMatrixEntry(i, j),
                        reference_mass.getMatrixEntry(i, j),
                        Real(1.0e-12));
        }
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}
