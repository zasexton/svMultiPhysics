/**
 * @file test_JITBasisBakingGeometry.cpp
 * @brief Specialized JIT kernels with baked basis tables on non-affine cells
 *
 * With specialization on, the JIT bakes the reference basis tables into the
 * kernel.  On a straight simplex the inverse Jacobian is constant over the
 * cell and the kernel rebuilds the physical gradients from the baked
 * reference gradients and that one inverse Jacobian.  A first-order Quad4,
 * Hex8, Wedge6 or Pyramid5 cell is affine only when it is a parallelogram
 * (parallelepiped); otherwise its inverse Jacobian varies over the cell.
 * Each test assembles a nonlinear gradient form on one distorted cell with the JIT
 * (specialization and basis baking on) and with the interpreter, and checks
 * that the two agree to round-off.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Forms/FormExpr.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace forms = svmp::FE::forms;
namespace systems = svmp::FE::systems;
using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

struct OneCell {
    std::string name;
    ElementType type;
    svmp::CellFamily family;
    int dim;
    std::vector<std::vector<double>> corners;  ///< reference corner order
};

std::shared_ptr<svmp::Mesh> oneCellMesh(const OneCell& cell)
{
    std::vector<svmp::real_t> x;
    std::vector<svmp::index_t> connectivity;
    for (std::size_t i = 0; i < cell.corners.size(); ++i) {
        for (int k = 0; k < cell.dim; ++k) {
            x.push_back(static_cast<svmp::real_t>(cell.corners[i][static_cast<std::size_t>(k)]));
        }
        connectivity.push_back(static_cast<svmp::index_t>(i));
    }
    svmp::CellShape shape{};
    shape.family = cell.family;
    shape.num_corners = static_cast<int>(cell.corners.size());
    shape.order = 1;
    auto base = std::make_shared<svmp::MeshBase>();
    base->build_from_arrays(cell.dim, x, {0, static_cast<svmp::offset_t>(connectivity.size())}, connectivity, {shape});
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

struct Assembled {
    std::vector<Real> vector;
    std::vector<Real> matrix;
    bool jit_ready{false};
};

// (1 + u^2) grad(u).grad(v) + u v over the cell, at a non-trivial state u.
Assembled assemble(const OneCell& cell, bool jit)
{
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(cell.type, 1);
    systems::FESystem system(oneCellMesh(cell));
    const auto u_id = system.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    system.addOperator("op");
    const auto u = forms::FormExpr::stateField(u_id, *space, "u");
    const auto v = forms::FormExpr::testFunction(u_id, *space, "v");

    systems::FormInstallOptions options;
    options.compiler_options.use_symbolic_tangent = true;
    options.compiler_options.jit.enable = jit;
    options.compiler_options.jit.specialization.enable = true;
    options.compiler_options.jit.basis_baking.enable = true;
    const auto installed = systems::installFormulation(
        system, "op", {u_id},
        ((forms::FormExpr::constant(1.0) + u * u) * forms::inner(forms::grad(u), forms::grad(v)) + u * v).dx(),
        options);
    system.setup();

    const auto n = system.dofHandler().getNumDofs();
    std::vector<Real> state(static_cast<std::size_t>(n));
    for (GlobalIndex i = 0; i < n; ++i) {
        state[static_cast<std::size_t>(i)] = Real(0.3) + Real(0.7) * std::sin(Real(1.9) * static_cast<Real>(i) + Real(0.4));
    }
    systems::SystemStateView view;
    view.u = state;
    systems::AssemblyRequest request;
    request.op = "op";
    request.want_matrix = true;
    request.want_vector = true;
    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();
    const auto result = system.assemble(request, view, &out, &out);
    EXPECT_TRUE(result.success) << result.error_message;

    Assembled a;
    for (GlobalIndex i = 0; i < n; ++i) {
        a.vector.push_back(out.getVectorEntry(i));
        for (GlobalIndex j = 0; j < n; ++j) {
            a.matrix.push_back(out.getMatrixEntry(i, j));
        }
    }
    if (!installed.residual.empty()) {
        if (const auto* wrapper = dynamic_cast<const forms::jit::JITKernelWrapper*>(installed.residual[0].get())) {
            a.jit_ready = wrapper->isJITReady();
        }
    }
    return a;
}

void expectJITMatchesTheInterpreter(const OneCell& cell)
{
    SCOPED_TRACE(cell.name);
    const auto interpreted = assemble(cell, false);
    const auto with_jit = assemble(cell, true);
    EXPECT_TRUE(with_jit.jit_ready);
    const auto compare = [](const std::vector<Real>& a, const std::vector<Real>& b, const char* what) {
        ASSERT_EQ(a.size(), b.size()) << what;
        Real scale = 1.0;
        for (const auto x : a) {
            scale = std::max(scale, std::abs(x));
        }
        for (std::size_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(a[i], b[i], 1e-12 * scale) << what << " entry " << i;
        }
    };
    compare(interpreted.vector, with_jit.vector, "residual");
    compare(interpreted.matrix, with_jit.matrix, "matrix");
}

} // namespace

TEST(JITBasisBakingGeometry, NonParallelogramFirstOrderCellsMatchTheInterpreter)
{
    requireLLVMJITOrSkip();
    expectJITMatchesTheInterpreter(
        {"Quad4", ElementType::Quad4, svmp::CellFamily::Quad, 2, {{0, 0}, {1, 0}, {1.2, 0.9}, {-0.1, 1.1}}});
    expectJITMatchesTheInterpreter({"Hex8", ElementType::Hex8, svmp::CellFamily::Hex, 3,
                                    {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                     {0, 0, 1}, {1, 0, 1}, {1.3, 1.2, 1.25}, {0, 1, 1}}});
    expectJITMatchesTheInterpreter({"Wedge6", ElementType::Wedge6, svmp::CellFamily::Wedge, 3,
                                    {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1.2, 0, 1.1}, {0, 0.9, 1.3}}});
    expectJITMatchesTheInterpreter({"Pyramid5", ElementType::Pyramid5, svmp::CellFamily::Pyramid, 3,
                                    {{0, 0, 0}, {1, 0, 0}, {1.2, 1.1, 0}, {0, 1, 0}, {0.4, 0.5, 1}}});
}

TEST(JITBasisBakingGeometry, StraightSimplicesMatchTheInterpreter)
{
    requireLLVMJITOrSkip();
    expectJITMatchesTheInterpreter(
        {"Triangle3", ElementType::Triangle3, svmp::CellFamily::Triangle, 2, {{0, 0}, {1, 0.1}, {0.2, 0.9}}});
    expectJITMatchesTheInterpreter({"Tetra4", ElementType::Tetra4, svmp::CellFamily::Tetra, 3,
                                    {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0.1, 0.2, 1}}});
}
