/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include <cmath>

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

TEST(VtpMeshSupport, LoadsSquareTriMeshAndAssemblesPoisson)
{
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    const auto mesh = loadSquareTriMeshFromVtp();
    ASSERT_TRUE(mesh);
    EXPECT_EQ(mesh->dim(), 2);
    EXPECT_GT(mesh->n_cells(), 0u);
    EXPECT_GT(mesh->n_vertices(), 0u);

    FE::systems::FESystem system(mesh);

    auto space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3, /*order=*/1);

    FE::systems::FieldSpec spec;
    spec.name = "u";
    spec.space = space;
    const FE::FieldId u_id = system.addField(std::move(spec));

    system.addOperator("residual");
    system.addOperator("jacobian");

    using namespace svmp::FE::forms;
    auto u = FormExpr::trialFunction(*space, "u");
    auto v = FormExpr::testFunction(*space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    FE::systems::installResidualForm(system, "residual", u_id, u_id, residual);
    FE::systems::installResidualForm(system, "jacobian", u_id, u_id, residual);

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

#endif // SVMP_FE_WITH_MESH

} // namespace test
} // namespace Physics
} // namespace svmp
