/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Constitutive/DualOps.h"
#include "Constitutive/ExpressionLaw.h"
#include "Constitutive/LawAdapters.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITValidation.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <vector>

namespace svmp {
namespace FE {
namespace constitutive {
namespace test {

class ScaleScalarLaw final : public ScalarLawCRTP<ScaleScalarLaw> {
public:
    explicit ScaleScalarLaw(Real k) : k_(k) {}

    template <class Scalar, class Workspace>
    [[nodiscard]] Scalar evalScalar(const Scalar& x, const forms::ConstitutiveEvalContext& /*ctx*/, Workspace& ws) const
    {
        if constexpr (std::is_same_v<Scalar, forms::Dual>) {
            DualOps ops(ws);
            return ops.mul(x, k_);
        } else {
            (void)ws;
            return k_ * x;
        }
    }

private:
    Real k_{1.0};
};

class ScaleMatrixLaw final : public MatrixLawCRTP<ScaleMatrixLaw> {
public:
    explicit ScaleMatrixLaw(Real k) : k_(k) {}

    template <class Scalar, class Workspace>
    void evalMatrix(const MatrixConstRef<Scalar>& A,
                    const forms::ConstitutiveEvalContext& /*ctx*/,
                    Workspace& ws,
                    MatrixRef<Scalar> out) const
    {
        const auto rows = A.rows();
        const auto cols = A.cols();
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                if constexpr (std::is_same_v<Scalar, forms::Dual>) {
                    DualOps ops(ws);
                    out(r, c) = ops.mul(A(r, c), k_);
                } else {
                    (void)ws;
                    out(r, c) = k_ * A(r, c);
                }
            }
        }
    }

private:
    Real k_{1.0};
};

TEST(ScalarLawCRTPTest, EvaluatesRealAndDual)
{
    ScaleScalarLaw model(3.0);

    forms::Value<Real> in;
    in.kind = forms::Value<Real>::Kind::Scalar;
    in.s = 2.0;

    const auto out = model.evaluate(in, /*dim=*/3);
    EXPECT_EQ(out.kind, forms::Value<Real>::Kind::Scalar);
    EXPECT_DOUBLE_EQ(out.s, 6.0);

    forms::DualWorkspace ws;
    ws.reset(/*num_dofs=*/2);

    auto dspan = ws.alloc();
    auto x = forms::makeDual(2.5, dspan);
    x.deriv[0] = 1.0;
    x.deriv[1] = 0.0;

    forms::Value<forms::Dual> din;
    din.kind = forms::Value<forms::Dual>::Kind::Scalar;
    din.s = x;

    const auto dout = model.evaluate(din, /*dim=*/3, ws);
    EXPECT_EQ(dout.kind, forms::Value<forms::Dual>::Kind::Scalar);
    EXPECT_DOUBLE_EQ(dout.s.value, 7.5);
    ASSERT_EQ(dout.s.deriv.size(), 2u);
    EXPECT_DOUBLE_EQ(dout.s.deriv[0], 3.0);
    EXPECT_DOUBLE_EQ(dout.s.deriv[1], 0.0);
}

TEST(MatrixLawCRTPTest, PreservesMatrixShapeAndPropagatesDualDerivatives)
{
    ScaleMatrixLaw model(2.0);

    forms::ConstitutiveEvalContext ctx;
    ctx.dim = 3;

    forms::Value<Real> in;
    in.kind = forms::Value<Real>::Kind::Matrix;
    in.resizeMatrix(2u, 2u);
    in.matrixAt(0u, 0u) = 1.0;
    in.matrixAt(0u, 1u) = 2.0;
    in.matrixAt(1u, 0u) = 3.0;
    in.matrixAt(1u, 1u) = 4.0;

    const auto out = model.evaluate(in, ctx);
    EXPECT_EQ(out.kind, forms::Value<Real>::Kind::Matrix);
    EXPECT_EQ(out.matrixRows(), 2u);
    EXPECT_EQ(out.matrixCols(), 2u);
    EXPECT_DOUBLE_EQ(out.matrixAt(0u, 0u), 2.0);
    EXPECT_DOUBLE_EQ(out.matrixAt(0u, 1u), 4.0);
    EXPECT_DOUBLE_EQ(out.matrixAt(1u, 0u), 6.0);
    EXPECT_DOUBLE_EQ(out.matrixAt(1u, 1u), 8.0);

    forms::DualWorkspace ws;
    ws.reset(/*num_dofs=*/1);

    forms::Value<forms::Dual> din;
    din.kind = forms::Value<forms::Dual>::Kind::Matrix;
    din.resizeMatrix(2u, 2u);
    for (std::size_t r = 0; r < 2u; ++r) {
        for (std::size_t c = 0; c < 2u; ++c) {
            auto span = ws.alloc();
            auto d = forms::makeDual(static_cast<Real>(1.0 + r * 2.0 + c), span);
            d.deriv[0] = (r == 0u && c == 0u) ? 1.0 : 0.0;
            din.matrixAt(r, c) = d;
        }
    }

    const auto dout = model.evaluate(din, ctx, ws);
    EXPECT_EQ(dout.kind, forms::Value<forms::Dual>::Kind::Matrix);
    EXPECT_EQ(dout.matrixRows(), 2u);
    EXPECT_EQ(dout.matrixCols(), 2u);

    EXPECT_DOUBLE_EQ(dout.matrixAt(0u, 0u).value, 2.0);
    ASSERT_EQ(dout.matrixAt(0u, 0u).deriv.size(), 1u);
    EXPECT_DOUBLE_EQ(dout.matrixAt(0u, 0u).deriv[0], 2.0);
    EXPECT_DOUBLE_EQ(dout.matrixAt(1u, 1u).value, 8.0);
    ASSERT_EQ(dout.matrixAt(1u, 1u).deriv.size(), 1u);
    EXPECT_DOUBLE_EQ(dout.matrixAt(1u, 1u).deriv[0], 0.0);
}

TEST(MakeScalarLawTest, WrapsLambdaAsConstitutiveModel)
{
    auto model = makeScalarLaw("affine",
                               [](const auto& x, const forms::ConstitutiveEvalContext& ctx, auto& ws) {
                                   const Real a = ctx.requireParamAs<Real>("a");
                                   const Real b = ctx.paramOr<Real>("b", 0.0);
                                   if constexpr (std::is_same_v<std::decay_t<decltype(x)>, forms::Dual>) {
                                       DualOps ops(ws);
                                       return ops.add(ops.mul(x, b), a);
                                   } else {
                                       (void)ws;
                                       return a + b * x;
                                   }
                               });

    std::function<std::optional<Real>(std::string_view)> get_real_param =
        [](std::string_view key) -> std::optional<Real> {
        if (key == "a") return 1.5;
        if (key == "b") return 2.0;
        return std::nullopt;
    };

    forms::ConstitutiveEvalContext ctx;
    ctx.dim = 3;
    ctx.get_real_param = &get_real_param;

    forms::Value<Real> in;
    in.kind = forms::Value<Real>::Kind::Scalar;
    in.s = 2.0;
    const auto out = model->evaluate(in, ctx);
    EXPECT_DOUBLE_EQ(out.s, 1.5 + 2.0 * 2.0);

    forms::DualWorkspace ws;
    ws.reset(/*num_dofs=*/1);
    auto span = ws.alloc();
    auto x = forms::makeDual(2.0, span);
    x.deriv[0] = 1.0;

    forms::Value<forms::Dual> din;
    din.kind = forms::Value<forms::Dual>::Kind::Scalar;
    din.s = x;
    const auto dout = model->evaluate(din, ctx, ws);
    EXPECT_DOUBLE_EQ(dout.s.value, 1.5 + 2.0 * 2.0);
    ASSERT_EQ(dout.s.deriv.size(), 1u);
    EXPECT_DOUBLE_EQ(dout.s.deriv[0], 2.0);
}

TEST(MakeExpressionScalarLawTest, AllowsExpressionStyleLocalLaws)
{
    auto model = makeExpressionScalarLaw(
        "quadratic",
        [](const auto& x, const forms::ConstitutiveEvalContext& ctx) {
            const auto a = x.constant(ctx.requireParamAs<Real>("a"));
            const auto b = x.constant(ctx.requireParamAs<Real>("b"));
            return a + b * pow(x, 2.0);
        });

    std::function<std::optional<Real>(std::string_view)> get_real_param =
        [](std::string_view key) -> std::optional<Real> {
        if (key == "a") return 1.0;
        if (key == "b") return 3.0;
        return std::nullopt;
    };

    forms::ConstitutiveEvalContext ctx;
    ctx.dim = 3;
    ctx.get_real_param = &get_real_param;

    forms::Value<Real> in;
    in.kind = forms::Value<Real>::Kind::Scalar;
    in.s = 2.0;
    const auto out = model->evaluate(in, ctx);
    EXPECT_DOUBLE_EQ(out.s, 1.0 + 3.0 * 4.0);

    forms::DualWorkspace ws;
    ws.reset(/*num_dofs=*/1);
    auto span = ws.alloc();
    auto x = forms::makeDual(2.0, span);
    x.deriv[0] = 1.0;

    forms::Value<forms::Dual> din;
    din.kind = forms::Value<forms::Dual>::Kind::Scalar;
    din.s = x;
    const auto dout = model->evaluate(din, ctx, ws);
    EXPECT_DOUBLE_EQ(dout.s.value, 1.0 + 3.0 * 4.0);
    ASSERT_EQ(dout.s.deriv.size(), 1u);
    EXPECT_DOUBLE_EQ(dout.s.deriv[0], 12.0);
}

// A law written as forms: setup expands each call into the body's tree, so the
// residual and the symbolic tangent of a call equal those of the hand-written
// tree bit for bit, the expansion compiles in strict JIT mode, and there is no
// pointwise evaluation.
TEST(FormLawTest, CallsExpandToTheBodyTree)
{
    using forms::FormExpr;
    const auto body = [](const FormExpr& s, int) {
        return FormExpr::constant(1.0) + s * s / (FormExpr::constant(1.0) + s);
    };
    const auto metadata = analysis::dynamicViscosityMetadata(INVALID_FIELD_ID, 1.0);
    using Kind = forms::ConstitutiveModel::ValueKind;
    const auto mu = forms::law("test::quadratic_ratio", Kind::Scalar, Kind::Scalar, body, metadata);
    EXPECT_EQ(mu.model()->kindId(), forms::InlinableConstitutiveModel::fnv1a64("test::quadratic_ratio"));
    EXPECT_NE(mu.model()->inlinable(), nullptr);
    ASSERT_TRUE(mu.model()->constitutiveLawMetadata(0u).has_value());
    EXPECT_EQ(mu.model()->constitutiveLawMetadata(0u)->role, analysis::ConstitutiveLawRole::DynamicViscosity);
    forms::Value<Real> in;
    in.kind = forms::Value<Real>::Kind::Scalar;
    in.s = 2.0;
    EXPECT_THROW((void)mu.model()->evaluate(in, 3), std::logic_error);

    forms::test::SingleTetraMeshAccess mesh;
    auto dof_map = forms::test::createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto s = inner(grad(u), grad(u));
    const auto assemble = [&](const FormExpr& coefficient, assembly::DenseMatrixView& J, assembly::DenseVectorView& R) {
        forms::SymbolicOptions options;
        options.use_symbolic_tangent = true;
        forms::FormCompiler compiler(options);
        auto ir = compiler.compileResidual((coefficient * inner(grad(u), grad(v))).dx());
        forms::SymbolicNonlinearFormKernel kernel(std::move(ir), forms::NonlinearKernelOutput::Both);
        kernel.resolveInlinableConstitutives();
        assembly::StandardAssembler assembler;
        assembler.setDofMap(dof_map);
        const std::vector<Real> solution = {0.1, -0.3, 0.25, 0.4};
        assembler.setCurrentSolution(solution);
        J.zero();
        R.zero();
        (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);
    };
    const auto n = dof_map.getNumDofs();
    assembly::DenseMatrixView J_law(n), J_tree(n);
    assembly::DenseVectorView R_law(n), R_tree(n);
    assemble(mu(s), J_law, R_law);
    assemble(body(s, 3), J_tree, R_tree);
    for (GlobalIndex i = 0; i < n; ++i) {
        EXPECT_EQ(R_law.getVectorEntry(i), R_tree.getVectorEntry(i));
        for (GlobalIndex j = 0; j < n; ++j) {
            EXPECT_EQ(J_law.getMatrixEntry(i, j), J_tree.getMatrixEntry(i, j));
        }
    }
    EXPECT_NE(R_tree.getVectorEntry(0), 0.0);

    const std::vector<FormExpr> inputs{s};
    const auto expansion = mu.model()->inlineExpand(inputs, forms::InlinableConstitutiveContext{.dim = 3});
    ASSERT_EQ(expansion.outputs.size(), 1u);
    EXPECT_EQ(expansion.outputs.front().toString(), body(s, 3).toString());
    EXPECT_TRUE(forms::jit::canCompile(expansion.outputs.front() * v,
                                       forms::jit::ValidationOptions{.strictness = forms::jit::Strictness::Strict})
                    .ok);
}

} // namespace test
} // namespace constitutive
} // namespace FE
} // namespace svmp

