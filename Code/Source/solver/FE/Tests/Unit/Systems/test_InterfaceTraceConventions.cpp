/**
 * @file test_InterfaceTraceConventions.cpp
 * @brief Sign conventions of the interface trace helpers, checked by assembly
 *
 * Two unit squares share the interface x = 1 (the mesh of
 * test_MortarHybridInterface).  The field is discontinuous P1 (L2 Lagrange),
 * so the basis functions of a cell sum to one and v = 1 on one cell, 0 on the
 * other, is a test function.  Every expected value below is independent of
 * which cell is the minus side of the interface: with n(-) the outward normal
 * of the minus cell and jump(v) = v(-) - v(+), the product n(-) jump(v) does
 * not change when the sides are swapped.
 *
 * - Consistency of interior-penalty (SIPG) interface terms: for u = x
 *   (continuous, so jump(u) = 0) the consistency term
 *   -{k grad u}.n(-) jump(v) integrates, for v = 1 on the left cell, to
 *   -int_Gamma grad(u).n_left = -1 (k = 1, |Gamma| = 1): it cancels the
 *   interface part of the boundary flux of the left cell.
 * - Average of a normal trace: for a constant vector c = (2, 0),
 *   {c}.n(-) jump(v) integrates, for v = 1 on the left cell, to c.n_left = 2.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Forms/BoundaryConditions.h"
#include "Forms/InterfaceConditions.h"
#include "Forms/StandardBCs.h"
#include "Forms/Vocabulary.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/L2Space.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace svmp::FE::systems::test {
namespace {

constexpr int kInterfaceMarker = 17;

std::shared_ptr<svmp::Mesh> twoUnitSquaresSharingAnInterface()
{
    auto base = std::make_shared<svmp::MeshBase>();
    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0};
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 4, 3, 1, 2, 5, 4};
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape, shape});
    base->finalize();

    auto mesh = svmp::create_mesh(std::move(base));
    auto& mbase = mesh->local_mesh();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mbase.n_faces()); ++f) {
        const auto verts = mbase.face_vertices(f);
        if (verts.size() != 2u) {
            continue;
        }
        const bool has1 = (verts[0] == 1 || verts[1] == 1);
        const bool has4 = (verts[0] == 4 || verts[1] == 4);
        if (has1 && has4) {
            mbase.add_to_set(svmp::EntityKind::Face, "middle", f);
            break;
        }
    }
    return mesh;
}

/// Residual sums over the basis functions of each cell (v = 1 on that cell).
struct CellSums {
    Real left{0.0};    ///< cell 0, x in [0, 1]
    Real right{0.0};   ///< cell 1, x in [1, 2]
};

/**
 * Installs (0 u v) dx + interface(u, v) for a discontinuous P1 field u, sets u
 * to the interpolant of @p u_exact (a function of x), assembles the residual
 * and sums it over the DOFs of each cell.
 */
CellSums assembleCellSums(
    const std::function<forms::FormExpr(const forms::FormExpr&, const forms::FormExpr&)>& interface_terms,
    const std::function<Real(Real)>& u_exact)
{
    using forms::FormExpr;
    auto space = std::make_shared<spaces::L2Space>(ElementType::Quad4, 1);

    FESystem sys(twoUnitSquaresSharingAnInterface());
    sys.setInterfaceMeshFromFaceSet(kInterfaceMarker, "middle");
    const FieldId u_id = sys.addField(FieldSpec{.name = "u", .space = space});
    sys.addOperator("op");

    const auto u = forms::StateField(u_id, *space, "u");
    const auto v = forms::TestField(u_id, *space, "v");
    const auto residual = (FormExpr::constant(0.0) * u * v).dx() + interface_terms(u, v);

    FormInstallOptions options;
    options.compiler_options.jit.enable = false;
    (void)installFormulation(sys, "op", {u_id}, residual, options);
    sys.setup();

    const auto n_dofs = sys.dofHandler().getNumDofs();
    const auto offset = sys.fieldDofOffset(u_id);
    std::vector<Real> state_values(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex cell = 0; cell < 2; ++cell) {
        // Cell c maps the reference square [-1, 1]^2 onto [c, c + 1] x [0, 1]
        // with vertex 0 at (c, 0): x = c + (xi + 1) / 2.
        std::vector<Real> local;
        space->interpolate_scalar(
            [&](const spaces::FunctionSpace::Value& xi) {
                return u_exact(static_cast<Real>(cell) + Real(0.5) * (xi[0] + Real(1.0)));
            },
            local);
        const auto dofs = sys.fieldDofHandler(u_id).getCellDofs(cell);
        EXPECT_EQ(dofs.size(), local.size());
        for (std::size_t i = 0; i < dofs.size() && i < local.size(); ++i) {
            state_values[static_cast<std::size_t>(offset + dofs[i])] = local[i];
        }
    }

    SystemStateView state;
    state.u = state_values;
    AssemblyRequest request;
    request.op = "op";
    request.want_vector = true;
    assembly::DenseVectorView R(n_dofs);
    R.zero();
    const auto result = sys.assemble(request, state, nullptr, &R);
    EXPECT_TRUE(result.success);

    CellSums sums;
    for (GlobalIndex cell = 0; cell < 2; ++cell) {
        Real sum = 0.0;
        for (const auto dof : sys.fieldDofHandler(u_id).getCellDofs(cell)) {
            sum += R.getVectorEntry(offset + dof);
        }
        (cell == 0 ? sums.left : sums.right) = sum;
    }
    return sums;
}

Real linearX(Real x) { return x; }

} // namespace

TEST(InterfaceTraceConventions, CoreInterfacePenaltyIntegrandIsConsistentForALinearSolution)
{
    // The control: the integrand written with the core jump() and avg().
    const auto sums = assembleCellSums(
        [](const forms::FormExpr& u, const forms::FormExpr& v) {
            return forms::interface::nitschePoissonIntegrand(forms::FormExpr::constant(1.0), u, v)
                .dI(kInterfaceMarker);
        },
        linearX);
    EXPECT_NEAR(sums.left, -1.0, 1e-12);
    EXPECT_NEAR(sums.right, 1.0, 1e-12);
}

TEST(InterfaceTraceConventions, InterfaceNitschePoissonHelperIsConsistentForALinearSolution)
{
    // applyNitscheInterfacePoisson must state the same consistency term.
    const auto sums = assembleCellSums(
        [](const forms::FormExpr& u, const forms::FormExpr& v) {
            return forms::interface::applyNitscheInterfacePoisson(
                (forms::FormExpr::constant(0.0) * u * v).dx(), forms::FormExpr::constant(1.0), u, v,
                kInterfaceMarker);
        },
        linearX);
    EXPECT_NEAR(sums.left, -1.0, 1e-12);
    EXPECT_NEAR(sums.right, 1.0, 1e-12);
}

TEST(InterfaceTraceConventions, AverageOfTheNormalTraceIsTheAverageFlux)
{
    // {c}.n(-) jump(v) for c = (2, 0): +2 on the left cell, -2 on the right.
    const auto sums = assembleCellSums(
        [](const forms::FormExpr& /*u*/, const forms::FormExpr& v) {
            const auto c = forms::as_vector({forms::FormExpr::constant(2.0), forms::FormExpr::constant(0.0)});
            return (forms::bc::applyInterfaceScalarTrace(c, forms::bc::ScalarTraceOperator::NormalComponent,
                                                         forms::bc::InterfaceTraceReduction::Average) *
                    forms::jump(v))
                .dI(kInterfaceMarker);
        },
        linearX);
    EXPECT_NEAR(sums.left, 2.0, 1e-12);
    EXPECT_NEAR(sums.right, -2.0, 1e-12);
}

TEST(InterfaceTraceConventions, NormalComponentJumpIsTheCoreJumpAlongTheMinusNormal)
{
    // Holds before and after the conventions were aligned: the normal
    // reduction u(+).n(+) + u(-).n(-) equals jump(u).n(-).  With u the
    // indicator of the left cell and c = (1, 0) the vector field u c, the
    // jump term tested with the plus-side trace of v sums to the same value
    // as the core form on each cell.
    const auto with_helper = assembleCellSums(
        [](const forms::FormExpr& u, const forms::FormExpr& v) {
            const auto c = forms::as_vector({forms::FormExpr::constant(1.0), forms::FormExpr::constant(0.0)});
            return (forms::bc::applyInterfaceScalarTrace(u * c, forms::bc::ScalarTraceOperator::NormalComponent,
                                                         forms::bc::InterfaceTraceReduction::Jump) *
                    forms::jump(v))
                .dI(kInterfaceMarker);
        },
        [](Real x) { return x < 1.0 ? Real(1.0) : Real(0.0); });
    const auto with_core = assembleCellSums(
        [](const forms::FormExpr& u, const forms::FormExpr& v) {
            const auto c = forms::as_vector({forms::FormExpr::constant(1.0), forms::FormExpr::constant(0.0)});
            return (forms::inner(forms::jump(u * c), forms::FormExpr::normal().minus()) * forms::jump(v))
                .dI(kInterfaceMarker);
        },
        [](Real x) { return x < 1.0 ? Real(1.0) : Real(0.0); });
    EXPECT_NEAR(with_helper.left, with_core.left, 1e-12);
    EXPECT_NEAR(with_helper.right, with_core.right, 1e-12);
    EXPECT_GT(std::abs(with_core.left), 0.5);
}

TEST(InterfaceTraceConventions, IdentityJumpReductionIsTheCoreJump)
{
    // One jump convention: the Identity Jump reduction of the BC helpers is
    // jump(u) = u(-) - u(+), by tree and by value.
    spaces::L2Space space(ElementType::Quad4, 1);
    const auto u = forms::StateField(FieldId{0}, space, "u");
    EXPECT_EQ(forms::bc::applyInterfaceScalarTrace(u, forms::bc::ScalarTraceOperator::Identity,
                                                   forms::bc::InterfaceTraceReduction::Jump)
                  .toString(),
              (u.minus() - u.plus()).toString());

    const auto indicator = [](Real x) { return x < 1.0 ? Real(1.0) : Real(0.0); };
    const auto with_helper = assembleCellSums(
        [](const forms::FormExpr& w, const forms::FormExpr& v) {
            return (forms::bc::applyInterfaceScalarTrace(w, forms::bc::ScalarTraceOperator::Identity,
                                                         forms::bc::InterfaceTraceReduction::Jump) *
                    forms::bc::applyInterfaceScalarTrace(v, forms::bc::ScalarTraceOperator::Identity,
                                                         forms::bc::InterfaceTraceReduction::Jump))
                .dI(kInterfaceMarker);
        },
        indicator);
    const auto with_core = assembleCellSums(
        [](const forms::FormExpr& w, const forms::FormExpr& v) {
            return (forms::jump(w) * forms::jump(v)).dI(kInterfaceMarker);
        },
        indicator);
    EXPECT_NEAR(with_helper.left, with_core.left, 1e-12);
    EXPECT_NEAR(with_helper.right, with_core.right, 1e-12);
    // jump(u) jump(v) with u the left indicator: +1 on the left, -1 on the right.
    EXPECT_NEAR(with_core.left, 1.0, 1e-12);
    EXPECT_NEAR(with_core.right, -1.0, 1e-12);

    // The jump-penalty wrapper with a prescribed jump g states
    // alpha jump(u) jump(v) - g jump(v), the same form as the core.
    const auto penalty_wrapper = assembleCellSums(
        [](const forms::FormExpr& w, const forms::FormExpr& v) {
            forms::FormExpr residual = (forms::FormExpr::constant(0.0) * w * v).dx();
            forms::bc::InterfaceTraceJumpPenaltyBC bc(kInterfaceMarker, forms::FormExpr::constant(3.0),
                                                      forms::FormExpr::constant(2.0),
                                                      forms::bc::ScalarTraceOperator::Identity);
            bc.contributeToResidual(residual, w, v);
            return residual;
        },
        indicator);
    const auto penalty_core = assembleCellSums(
        [](const forms::FormExpr& w, const forms::FormExpr& v) {
            return (3.0 * forms::jump(w) * forms::jump(v)).dI(kInterfaceMarker) -
                   (2.0 * forms::jump(v)).dI(kInterfaceMarker);
        },
        indicator);
    EXPECT_NEAR(penalty_wrapper.left, penalty_core.left, 1e-12);
    EXPECT_NEAR(penalty_wrapper.right, penalty_core.right, 1e-12);
}

} // namespace svmp::FE::systems::test
