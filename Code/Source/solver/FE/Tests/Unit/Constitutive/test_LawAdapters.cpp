/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Constitutive/DualOps.h"
#include "Constitutive/ExpressionLaw.h"
#include "Constitutive/LawAdapters.h"

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

} // namespace test
} // namespace constitutive
} // namespace FE
} // namespace svmp

