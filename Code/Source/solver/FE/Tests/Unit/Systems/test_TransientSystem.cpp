/**
 * @file test_TransientSystem.cpp
 * @brief Unit tests for Systems TransientSystem
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/TimeIntegrator.h"
#include "Systems/TransientSystem.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Spaces/H1Space.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <memory>
#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

using svmp::FE::assembly::DenseMatrixView;
using svmp::FE::assembly::MassKernel;

using svmp::FE::systems::AssemblyRequest;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::SystemStateView;
using svmp::FE::systems::TransientSystem;

namespace {

std::shared_ptr<Mesh> build_single_quad_mesh()
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

} // namespace

TEST(TransientSystemTest, TransientSystem_SystemAccessor_ReturnsReference)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    (void)sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.setup();

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    TransientSystem transient(sys, integrator);
    EXPECT_EQ(&transient.system(), &sys);
}

TEST(TransientSystemTest, TransientSystem_SystemAccessor_ConstCorrectness)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    (void)sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.setup();

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    TransientSystem transient(sys, integrator);

    const TransientSystem& cref = transient;
    const FESystem& sys_ref = cref.system();
    EXPECT_EQ(&sys_ref, &sys);
}

TEST(TransientSystemTest, TransientSystem_IntegratorAccessor_ReturnsReference)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    (void)sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.setup();

    auto integrator = std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>();
    TransientSystem transient(sys, integrator);
    EXPECT_EQ(&transient.integrator(), integrator.get());
}

TEST(TransientSystemTest, TransientSystem_SteadyFESystem_AssemblyWorks)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    sys.setup();

    const auto n_dofs = sys.dofHandler().getNumDofs();
    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);

    SystemStateView state;
    state.u = U;

    AssemblyRequest req;
    req.op = "mass";
    req.want_matrix = true;

    DenseMatrixView A_sys(n_dofs);
    DenseMatrixView A_transient(n_dofs);
    A_sys.zero();
    A_transient.zero();

    (void)sys.assemble(req, state, &A_sys, nullptr);

    TransientSystem transient(sys, std::make_shared<svmp::FE::systems::BackwardDifferenceIntegrator>());
    (void)transient.assemble(req, state, &A_transient, nullptr);

    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        for (GlobalIndex j = 0; j < n_dofs; ++j) {
            EXPECT_NEAR(A_transient.getMatrixEntry(i, j), A_sys.getMatrixEntry(i, j), 1e-12);
        }
    }
}

TEST(TransientSystemTest, TransientSystem_NullIntegrator_Throws)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Quad4, 1);

    FESystem sys(mesh);
    (void)sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.setup();

    std::shared_ptr<const svmp::FE::systems::TimeIntegrator> null_integrator{};
    EXPECT_THROW((void)TransientSystem(sys, null_integrator), svmp::FE::FEException);
}

