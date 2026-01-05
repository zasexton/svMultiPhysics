/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Constitutive/ModelCRTP.h"

#include "Forms/Dual.h"

#include <type_traits>

namespace svmp {
namespace FE {
namespace constitutive {
namespace test {

class ScaleScalarModel final : public ModelCRTP<ScaleScalarModel> {
public:
    explicit ScaleScalarModel(Real k) : k_(k) {}

    template <class Scalar, class Workspace>
    [[nodiscard]] forms::Value<Scalar> evaluateImpl(const forms::Value<Scalar>& input,
                                                    int /*dim*/,
                                                    Workspace& ws) const
    {
        requireValueKind(input, forms::Value<Scalar>::Kind::Scalar, "ScaleScalarModel");

        forms::Value<Scalar> out;
        out.kind = forms::Value<Scalar>::Kind::Scalar;

        if constexpr (std::is_same_v<Scalar, forms::Dual>) {
            out.s = forms::mul(input.s, k_, forms::makeDualConstant(0.0, ws.alloc()));
        } else {
            out.s = k_ * input.s;
        }
        return out;
    }

private:
    Real k_{1.0};
};

class IdentityScalarModel final : public ModelCRTP<IdentityScalarModel> {
public:
    static constexpr ValueKind kExpectedInputKind = forms::Value<Real>::Kind::Scalar;
    static constexpr std::size_t kExpectedInputCount = 1;
    static constexpr StateSpec kStateSpec{sizeof(int), alignof(int)};

    template <class Scalar, class Workspace>
    [[nodiscard]] forms::Value<Scalar> evaluateImpl(const forms::Value<Scalar>& input,
                                                    int /*dim*/,
                                                    Workspace& /*ws*/) const
    {
        requireValueKind(input, forms::Value<Scalar>::Kind::Scalar, "IdentityScalarModel");
        return input;
    }
};

class LayoutDerivedStateModel final : public ModelCRTP<LayoutDerivedStateModel> {
public:
    inline static const StateLayout kStateLayout = []() {
        StateLayoutBuilder b("LayoutDerivedStateModel");
        b.add<int>("counter");
        return b.build();
    }();

    template <class Scalar, class Workspace>
    [[nodiscard]] forms::Value<Scalar> evaluateImpl(const forms::Value<Scalar>& input,
                                                    int /*dim*/,
                                                    Workspace& /*ws*/) const
    {
        requireValueKind(input, forms::Value<Scalar>::Kind::Scalar, "LayoutDerivedStateModel");
        return input;
    }
};

class ContextScaleScalarModel final : public ModelCRTP<ContextScaleScalarModel> {
public:
    template <class Scalar, class Workspace>
    [[nodiscard]] forms::Value<Scalar> evaluateImpl(const forms::Value<Scalar>& input,
                                                    const forms::ConstitutiveEvalContext& ctx,
                                                    Workspace& ws) const
    {
        requireValueKind(input, forms::Value<Scalar>::Kind::Scalar, "ContextScaleScalarModel");

        const Real factor = ctx.time + ctx.dt;
        forms::Value<Scalar> out;
        out.kind = forms::Value<Scalar>::Kind::Scalar;
        if constexpr (std::is_same_v<Scalar, forms::Dual>) {
            out.s = forms::mul(input.s, factor, forms::makeDualConstant(0.0, ws.alloc()));
        } else {
            out.s = factor * input.s;
        }
        return out;
    }
};

TEST(ModelCRTPTest, EvaluatesReal)
{
    ScaleScalarModel model(3.0);
    forms::Value<Real> in;
    in.kind = forms::Value<Real>::Kind::Scalar;
    in.s = 2.0;

    const auto out = model.evaluate(in, /*dim=*/3);
    EXPECT_EQ(out.kind, forms::Value<Real>::Kind::Scalar);
    EXPECT_DOUBLE_EQ(out.s, 6.0);
}

TEST(ModelCRTPTest, EvaluatesDualAndPropagatesDerivatives)
{
    ScaleScalarModel model(4.0);

    forms::DualWorkspace ws;
    ws.reset(/*num_dofs=*/2);

    auto in_deriv = ws.alloc();
    forms::Dual x = forms::makeDual(2.5, in_deriv);
    x.deriv[0] = 1.0;
    x.deriv[1] = 0.0;

    forms::Value<forms::Dual> in;
    in.kind = forms::Value<forms::Dual>::Kind::Scalar;
    in.s = x;

    const auto out = model.evaluate(in, /*dim=*/3, ws);
    EXPECT_EQ(out.kind, forms::Value<forms::Dual>::Kind::Scalar);
    EXPECT_DOUBLE_EQ(out.s.value, 10.0);
    ASSERT_EQ(out.s.deriv.size(), 2u);
    EXPECT_DOUBLE_EQ(out.s.deriv[0], 4.0);
    EXPECT_DOUBLE_EQ(out.s.deriv[1], 0.0);
}

TEST(ModelCRTPTest, ThrowsOnKindMismatch)
{
    ScaleScalarModel model(1.0);
    forms::Value<Real> in;
    in.kind = forms::Value<Real>::Kind::Vector;
    in.v = {1.0, 2.0, 3.0};

    EXPECT_THROW((void)model.evaluate(in, /*dim=*/3), InvalidArgumentException);
}

TEST(ModelCRTPTest, DefaultsMetadataWhenUnspecified)
{
    ScaleScalarModel model(1.0);
    const forms::ConstitutiveModel& base = model;

    EXPECT_FALSE(base.expectedInputKind().has_value());

    const auto spec = base.stateSpec();
    EXPECT_EQ(spec.bytes_per_qpt, 0u);
    EXPECT_TRUE(spec.empty());
}

TEST(ModelCRTPTest, ForwardsStaticMetadataViaTypeErasure)
{
    IdentityScalarModel model;
    const forms::ConstitutiveModel& base = model;

    const auto kind = base.expectedInputKind();
    ASSERT_TRUE(kind.has_value());
    EXPECT_EQ(*kind, forms::Value<Real>::Kind::Scalar);

    const auto spec = base.stateSpec();
    EXPECT_EQ(spec.bytes_per_qpt, sizeof(int));
    EXPECT_EQ(spec.alignment, alignof(int));
    EXPECT_FALSE(spec.empty());
}

TEST(ModelCRTPTest, ForwardsExpectedInputCountViaTypeErasure)
{
    IdentityScalarModel model;
    const forms::ConstitutiveModel& base = model;

    const auto count = base.expectedInputCount();
    ASSERT_TRUE(count.has_value());
    EXPECT_EQ(*count, 1u);
}

TEST(ModelCRTPTest, DerivesStateSpecAndExposesStateLayout)
{
    LayoutDerivedStateModel model;
    const forms::ConstitutiveModel& base = model;

    const auto spec = base.stateSpec();
    EXPECT_EQ(spec.bytes_per_qpt, LayoutDerivedStateModel::kStateLayout.bytesPerPoint());
    EXPECT_EQ(spec.alignment, LayoutDerivedStateModel::kStateLayout.alignment());
    EXPECT_FALSE(spec.empty());

    const auto* layout = base.stateLayout();
    ASSERT_NE(layout, nullptr);
    EXPECT_EQ(layout->bytesPerPoint(), LayoutDerivedStateModel::kStateLayout.bytesPerPoint());
    EXPECT_EQ(layout->alignment(), LayoutDerivedStateModel::kStateLayout.alignment());
}

TEST(ModelCRTPTest, EvaluatesWithContextOverload)
{
    ContextScaleScalarModel model;

    forms::ConstitutiveEvalContext ctx;
    ctx.dim = 3;
    ctx.time = 2.0;
    ctx.dt = 0.5;

    forms::Value<Real> in;
    in.kind = forms::Value<Real>::Kind::Scalar;
    in.s = 4.0;

    const auto out = model.evaluate(in, ctx);
    EXPECT_EQ(out.kind, forms::Value<Real>::Kind::Scalar);
    EXPECT_DOUBLE_EQ(out.s, (ctx.time + ctx.dt) * in.s);
}

TEST(ModelCRTPTest, EvaluatesDualWithContextOverload)
{
    ContextScaleScalarModel model;

    forms::ConstitutiveEvalContext ctx;
    ctx.dim = 3;
    ctx.time = 1.25;
    ctx.dt = 0.75;

    forms::DualWorkspace ws;
    ws.reset(/*num_dofs=*/2);

    auto in_deriv = ws.alloc();
    forms::Dual x = forms::makeDual(3.0, in_deriv);
    x.deriv[0] = 1.0;
    x.deriv[1] = -2.0;

    forms::Value<forms::Dual> in;
    in.kind = forms::Value<forms::Dual>::Kind::Scalar;
    in.s = x;

    const auto out = model.evaluate(in, ctx, ws);
    EXPECT_EQ(out.kind, forms::Value<forms::Dual>::Kind::Scalar);
    EXPECT_DOUBLE_EQ(out.s.value, (ctx.time + ctx.dt) * x.value);
    ASSERT_EQ(out.s.deriv.size(), 2u);
    EXPECT_DOUBLE_EQ(out.s.deriv[0], (ctx.time + ctx.dt) * 1.0);
    EXPECT_DOUBLE_EQ(out.s.deriv[1], (ctx.time + ctx.dt) * -2.0);
}

} // namespace test
} // namespace constitutive
} // namespace FE
} // namespace svmp
