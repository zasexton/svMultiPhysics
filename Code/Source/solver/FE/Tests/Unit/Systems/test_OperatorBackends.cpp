/**
 * @file test_OperatorBackends.cpp
 * @brief Unit tests for Systems operator backends (matrix-free + functionals)
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/MatrixFreeAssembler.h"
#include "Assembly/FunctionalAssembler.h"

#include "Spaces/H1Space.h"

#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellShape.h"

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
using svmp::FE::assembly::L2NormKernel;

using svmp::FE::spaces::H1Space;

using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::SystemStateView;

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

std::vector<Real> applyDense(const DenseMatrixView& A, std::span<const Real> x)
{
    std::vector<Real> y(static_cast<std::size_t>(A.numRows()), 0.0);
    for (GlobalIndex i = 0; i < A.numRows(); ++i) {
        Real sum = 0.0;
        for (GlobalIndex j = 0; j < A.numCols(); ++j) {
            sum += A.getMatrixEntry(i, j) * x[static_cast<std::size_t>(j)];
        }
        y[static_cast<std::size_t>(i)] = sum;
    }
    return y;
}

} // namespace

TEST(OperatorBackends, MatrixFreeMatchesAssembledForMassKernel)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    auto mass_kernel = std::make_shared<MassKernel>(1.0);
    auto mf_kernel_unique = svmp::FE::assembly::wrapAsMatrixFreeKernel(*mass_kernel);
    std::shared_ptr<svmp::FE::assembly::IMatrixFreeKernel> mf_kernel = std::move(mf_kernel_unique);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, mass_kernel);
    sys.addMatrixFreeKernel("mass_mf", mf_kernel);
    sys.setup();

    DenseMatrixView assembled(sys.dofHandler().getNumDofs());
    SystemStateView state;
    sys.assembleMass(state, assembled);

    auto op = sys.matrixFreeOperator("mass_mf");
    ASSERT_TRUE(op);
    EXPECT_EQ(op->numRows(), assembled.numRows());

    std::vector<Real> x(static_cast<std::size_t>(assembled.numRows()), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<Real>(i + 1);
    }

    std::vector<Real> y(static_cast<std::size_t>(assembled.numRows()), 0.0);
    op->apply(x, y);

    auto y_ref = applyDense(assembled, x);
    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y[i], y_ref[i], 1e-12);
    }
}

TEST(OperatorBackends, AutoRegistersMatrixFreeForEligibleCellOnlyOperator)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    auto mass_kernel = std::make_shared<MassKernel>(1.0);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, mass_kernel);

    svmp::FE::systems::SetupOptions opts;
    opts.auto_register_matrix_free = true;
    sys.setup(opts);

    DenseMatrixView assembled(sys.dofHandler().getNumDofs());
    SystemStateView state;
    sys.assembleMass(state, assembled);

    auto op = sys.matrixFreeOperator("mass");
    ASSERT_TRUE(op);
    EXPECT_EQ(op->numRows(), assembled.numRows());

    std::vector<Real> x(static_cast<std::size_t>(assembled.numRows()), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<Real>(i + 1);
    }

    std::vector<Real> y(static_cast<std::size_t>(assembled.numRows()), 0.0);
    op->apply(x, y);

    auto y_ref = applyDense(assembled, x);
    for (std::size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y[i], y_ref[i], 1e-12);
    }
}

TEST(OperatorBackends, FunctionalL2NormOfConstantOneIsOne)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.setup();

    sys.addFunctionalKernel("qoi:l2", std::make_shared<L2NormKernel>());

    std::vector<Real> uvec(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 1.0);
    SystemStateView state;
    state.u = uvec;

    const Real value = sys.evaluateFunctional("qoi:l2", state);
    EXPECT_NEAR(value, 1.0, 1e-12);
}
