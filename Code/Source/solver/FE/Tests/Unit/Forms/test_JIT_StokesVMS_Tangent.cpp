/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITValidation.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <memory>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

struct StokesVMSParams {
    Real rho{1.0};
    Real mu{0.01};
    Real ct_m{1.0};
    Real ct_c{36.0};
    Real eps{1e-12};
};

FormExpr makeStokesVMSMomentumForm(const spaces::FunctionSpace& velocity_space,
                                  const FormExpr& u,
                                  const FormExpr& p,
                                  int dim,
                                  const StokesVMSParams& params)
{
    using namespace svmp::FE::forms;

    const auto v = TestFunction(velocity_space, "v");

    const auto rho = FormExpr::constant(params.rho);
    const auto mu = FormExpr::constant(params.mu);

    // Stokes: convection disabled (a = 0), body force f = 0.
    std::vector<FormExpr> zero_vec;
    zero_vec.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        zero_vec.push_back(FormExpr::constant(0.0));
    }
    const auto a = FormExpr::asVector(zero_vec);
    const auto f = FormExpr::asVector(std::move(zero_vec));

    const auto stress = FormExpr::constant(2.0) * mu * sym(grad(u));
    const auto r_m = rho * (dt(u) + grad(u) * a - f) + grad(p) - div(stress);

    // Galerkin terms.
    const auto inertia = rho * inner(dt(u), v);
    const auto viscous = FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v)));
    const auto forcing = -rho * inner(f, v);

    // Residual-based VMS static subscales.
    const auto eps = FormExpr::constant(params.eps);
    const auto dt_step = FormExpr::effectiveTimeStep();
    const auto ct_m = FormExpr::constant(params.ct_m);
    const auto ct_c = FormExpr::constant(params.ct_c);

    const auto Jinv_expr = Jinv();
    const auto K = transpose(Jinv_expr) * Jinv_expr;
    const auto nu = mu / rho;

    const auto kT = FormExpr::constant(4.0) * (ct_m * ct_m) / (dt_step * dt_step);
    const auto kU = inner(a, K * a);
    const auto kS = ct_c * doubleContraction(K, K) * (nu * nu);
    const auto tau_m = FormExpr::constant(1.0) / (rho * sqrt(kT + kU + kS + eps));
    const auto tau_c = FormExpr::constant(1.0) / (tau_m * trace(K) + eps);

    const auto u_sub = -tau_m * r_m;
    const auto p_sub = -tau_c * div(u);

    const auto p_adv = p + p_sub;
    const auto pressure_adv = -p_adv * div(v);

    // Convection-related terms are zero for Stokes (a = 0).
    const auto convection_adv = rho * inner(grad(u) * a, v);
    const auto supg = -rho * inner(grad(v) * a, u_sub);

    return (inertia + convection_adv + viscous + pressure_adv + forcing + supg).dx();
}

} // namespace

TEST(JITCompilerStokesVMS, TangentCellCompilesForMomentumBlocks)
{
    requireLLVMJITOrSkip();

    constexpr int dim = 2;
    const auto base = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);
    const auto vel_space = std::make_shared<spaces::ProductSpace>(base, dim);
    const auto p_space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    const FieldId u_id = static_cast<FieldId>(0);
    const FieldId p_id = static_cast<FieldId>(1);

    SymbolicOptions compiler_opts;
    compiler_opts.jit.enable = true;
    FormCompiler compiler(compiler_opts);

    const auto jit_opts = makeUnitTestJITOptions();
    auto jit_compiler = jit::JITCompiler::getOrCreate(jit_opts);
    ASSERT_TRUE(jit_compiler);

    jit::ValidationOptions vopt;
    vopt.strictness = jit::Strictness::AllowExternalCalls;

    const StokesVMSParams params{};

    // (1) Momentum Jacobian block dR_u/du: u is TrialFunction, p is DiscreteField.
    {
        const auto u = TrialFunction(*vel_space, "u");
        const auto p = FormExpr::discreteField(p_id, *p_space, "p");
        const auto form = makeStokesVMSMomentumForm(*vel_space, u, p, dim, params);
        auto ir = compiler.compileResidual(form);
        SymbolicNonlinearFormKernel k(std::move(ir), NonlinearKernelOutput::MatrixOnly);
        k.resolveInlinableConstitutives();

        const auto r = jit_compiler->compile(k.tangentIR(), vopt);
        EXPECT_TRUE(r.ok) << r.message;
    }

    // (2) Momentum Jacobian block dR_u/dp: p is TrialFunction, u is DiscreteField.
    {
        const auto u = FormExpr::discreteField(u_id, *vel_space, "u");
        const auto p = TrialFunction(*p_space, "p");
        const auto form = makeStokesVMSMomentumForm(*vel_space, u, p, dim, params);
        auto ir = compiler.compileResidual(form);
        SymbolicNonlinearFormKernel k(std::move(ir), NonlinearKernelOutput::MatrixOnly);
        k.resolveInlinableConstitutives();

        const auto r = jit_compiler->compile(k.tangentIR(), vopt);
        EXPECT_TRUE(r.ok) << r.message;
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

