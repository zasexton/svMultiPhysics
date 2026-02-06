/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_ConstitutiveModel.cpp
 * @brief Unit tests for FE/Forms constitutive operator integration boundary
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Constitutive/StateLayout.h"
#include "Constitutive/StateView.h"
#include "Core/FEException.h"
#include "Forms/ConstitutiveModel.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/InlinableConstitutiveModel.h"
#include "Forms/JIT/JITValidation.h"
#include "Forms/Vocabulary.h"
#include "Quadrature/QuadratureFactory.h"
#include "Quadrature/QuadratureRule.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Systems/MaterialStateProvider.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

class ScaleScalarModel final : public ConstitutiveModel {
public:
    explicit ScaleScalarModel(Real k) : k_(k) {}

    [[nodiscard]] Value<Real> evaluate(const Value<Real>& input, int /*dim*/) const override
    {
        if (input.kind != Value<Real>::Kind::Scalar) {
            throw std::invalid_argument("ScaleScalarModel: expected scalar input");
        }
        return Value<Real>{Value<Real>::Kind::Scalar, k_ * input.s};
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& input,
                                       int /*dim*/,
                                       DualWorkspace& ws) const override
    {
        if (input.kind != Value<Dual>::Kind::Scalar) {
            throw std::invalid_argument("ScaleScalarModel: expected scalar input (dual)");
        }
        Value<Dual> out;
        out.kind = Value<Dual>::Kind::Scalar;
        out.s = mul(input.s, k_, makeDualConstant(0.0, ws.alloc()));
        return out;
    }

private:
    Real k_{1.0};
};

class ScaleVectorModel final : public ConstitutiveModel {
public:
    explicit ScaleVectorModel(Real k) : k_(k) {}

    [[nodiscard]] Value<Real> evaluate(const Value<Real>& input, int /*dim*/) const override
    {
        if (input.kind != Value<Real>::Kind::Vector) {
            throw std::invalid_argument("ScaleVectorModel: expected vector input");
        }
        Value<Real> out;
        out.kind = Value<Real>::Kind::Vector;
        for (int d = 0; d < 3; ++d) out.v[static_cast<std::size_t>(d)] = k_ * input.v[static_cast<std::size_t>(d)];
        return out;
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& input,
                                       int /*dim*/,
                                       DualWorkspace& ws) const override
    {
        if (input.kind != Value<Dual>::Kind::Vector) {
            throw std::invalid_argument("ScaleVectorModel: expected vector input (dual)");
        }
        Value<Dual> out;
        out.kind = Value<Dual>::Kind::Vector;
        for (int d = 0; d < 3; ++d) {
            out.v[static_cast<std::size_t>(d)] = mul(input.v[static_cast<std::size_t>(d)],
                                                     k_,
                                                     makeDualConstant(0.0, ws.alloc()));
        }
        return out;
    }

private:
    Real k_{1.0};
};

class SumTwoScalarsModel final : public ConstitutiveModel {
public:
    [[nodiscard]] Value<Real> evaluate(const Value<Real>& /*input*/, int /*dim*/) const override
    {
        throw std::logic_error("SumTwoScalarsModel: unary evaluate() should not be called");
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& /*input*/,
                                       int /*dim*/,
                                       DualWorkspace& /*ws*/) const override
    {
        throw std::logic_error("SumTwoScalarsModel: unary evaluate() should not be called (dual)");
    }

    [[nodiscard]] std::optional<std::size_t> expectedInputCount() const override { return 2u; }

    [[nodiscard]] std::optional<ValueKind> expectedInputKind(std::size_t /*input_index*/) const override
    {
        return ValueKind::Scalar;
    }

    [[nodiscard]] Value<Real> evaluateNary(std::span<const Value<Real>> inputs,
                                           const ConstitutiveEvalContext& /*ctx*/) const override
    {
        if (inputs.size() != 2u) {
            throw std::invalid_argument("SumTwoScalarsModel: expected exactly 2 inputs");
        }
        if (inputs[0].kind != Value<Real>::Kind::Scalar || inputs[1].kind != Value<Real>::Kind::Scalar) {
            throw std::invalid_argument("SumTwoScalarsModel: expected scalar inputs");
        }
        return Value<Real>{Value<Real>::Kind::Scalar, inputs[0].s + inputs[1].s};
    }

    [[nodiscard]] Value<Dual> evaluateNary(std::span<const Value<Dual>> inputs,
                                           const ConstitutiveEvalContext& /*ctx*/,
                                           DualWorkspace& ws) const override
    {
        if (inputs.size() != 2u) {
            throw std::invalid_argument("SumTwoScalarsModel: expected exactly 2 inputs (dual)");
        }
        if (inputs[0].kind != Value<Dual>::Kind::Scalar || inputs[1].kind != Value<Dual>::Kind::Scalar) {
            throw std::invalid_argument("SumTwoScalarsModel: expected scalar inputs (dual)");
        }
        Value<Dual> out;
        out.kind = Value<Dual>::Kind::Scalar;
        out.s = add(inputs[0].s, inputs[1].s, makeDualConstant(0.0, ws.alloc()));
        return out;
    }
};

class ContextFactorModel final : public ConstitutiveModel {
public:
    [[nodiscard]] Value<Real> evaluate(const Value<Real>& /*input*/, int /*dim*/) const override
    {
        throw std::logic_error("ContextFactorModel: unary evaluate() should not be called");
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& /*input*/,
                                       int /*dim*/,
                                       DualWorkspace& /*ws*/) const override
    {
        throw std::logic_error("ContextFactorModel: unary evaluate() should not be called (dual)");
    }

    [[nodiscard]] Value<Real> evaluate(const Value<Real>& input, const ConstitutiveEvalContext& ctx) const override
    {
        if (input.kind != Value<Real>::Kind::Scalar) {
            throw std::invalid_argument("ContextFactorModel: expected scalar input");
        }

        const Real k = ctx.realParam("k").value_or(0.0);
        const Real factor = k + ctx.time + ctx.dt;
        return Value<Real>{Value<Real>::Kind::Scalar, factor * input.s};
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& input,
                                       const ConstitutiveEvalContext& ctx,
                                       DualWorkspace& ws) const override
    {
        if (input.kind != Value<Dual>::Kind::Scalar) {
            throw std::invalid_argument("ContextFactorModel: expected scalar input (dual)");
        }

        const Real k = ctx.realParam("k").value_or(0.0);
        const Real factor = k + ctx.time + ctx.dt;

        Value<Dual> out;
        out.kind = Value<Dual>::Kind::Scalar;
        out.s = mul(input.s, factor, makeDualConstant(0.0, ws.alloc()));
        return out;
    }
};

class StatefulCounterModel final : public ConstitutiveModel {
public:
    [[nodiscard]] StateSpec stateSpec() const noexcept override
    {
        return StateSpec{sizeof(int), alignof(int)};
    }

    [[nodiscard]] Value<Real> evaluate(const Value<Real>& /*input*/, int /*dim*/) const override
    {
        throw std::logic_error("StatefulCounterModel: unary evaluate() should not be called");
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& /*input*/,
                                       int /*dim*/,
                                       DualWorkspace& /*ws*/) const override
    {
        throw std::logic_error("StatefulCounterModel: unary evaluate() should not be called (dual)");
    }

    [[nodiscard]] Value<Real> evaluate(const Value<Real>& input, const ConstitutiveEvalContext& ctx) const override
    {
        if (input.kind != Value<Real>::Kind::Scalar) {
            throw std::invalid_argument("StatefulCounterModel: expected scalar input");
        }
        if (ctx.state_work.size() != sizeof(int)) {
            throw std::invalid_argument("StatefulCounterModel: expected sizeof(int) state");
        }

        auto* counter = reinterpret_cast<int*>(ctx.state_work.data());
        *counter += 1;
        return Value<Real>{Value<Real>::Kind::Scalar, static_cast<Real>(*counter) * input.s};
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& input,
                                       const ConstitutiveEvalContext& ctx,
                                       DualWorkspace& ws) const override
    {
        if (input.kind != Value<Dual>::Kind::Scalar) {
            throw std::invalid_argument("StatefulCounterModel: expected scalar input (dual)");
        }
        if (ctx.state_work.size() != sizeof(int)) {
            throw std::invalid_argument("StatefulCounterModel: expected sizeof(int) state (dual)");
        }

        auto* counter = reinterpret_cast<int*>(ctx.state_work.data());
        *counter += 1;

        const Real factor = static_cast<Real>(*counter);
        Value<Dual> out;
        out.kind = Value<Dual>::Kind::Scalar;
        out.s = mul(input.s, factor, makeDualConstant(0.0, ws.alloc()));
        return out;
    }
};

class StateLayoutOnlyCounterModel final : public ConstitutiveModel {
public:
    [[nodiscard]] const constitutive::StateLayout* stateLayout() const noexcept override
    {
        return &kStateLayout;
    }

    [[nodiscard]] Value<Real> evaluate(const Value<Real>& /*input*/, int /*dim*/) const override
    {
        throw std::logic_error("StateLayoutOnlyCounterModel: unary evaluate() should not be called");
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& /*input*/,
                                       int /*dim*/,
                                       DualWorkspace& /*ws*/) const override
    {
        throw std::logic_error("StateLayoutOnlyCounterModel: unary evaluate() should not be called (dual)");
    }

    [[nodiscard]] Value<Real> evaluate(const Value<Real>& input, const ConstitutiveEvalContext& ctx) const override
    {
        if (input.kind != Value<Real>::Kind::Scalar) {
            throw std::invalid_argument("StateLayoutOnlyCounterModel: expected scalar input");
        }
        if (ctx.state_work.size() != kStateLayout.bytesPerPoint()) {
            throw std::invalid_argument("StateLayoutOnlyCounterModel: unexpected state size");
        }

        constitutive::StateView view(ctx.state_work);
        auto& counter = view.get<int>(0);
        counter += 1;
        return Value<Real>{Value<Real>::Kind::Scalar, static_cast<Real>(counter) * input.s};
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& input,
                                       const ConstitutiveEvalContext& ctx,
                                       DualWorkspace& ws) const override
    {
        if (input.kind != Value<Dual>::Kind::Scalar) {
            throw std::invalid_argument("StateLayoutOnlyCounterModel: expected scalar input (dual)");
        }
        if (ctx.state_work.size() != kStateLayout.bytesPerPoint()) {
            throw std::invalid_argument("StateLayoutOnlyCounterModel: unexpected state size (dual)");
        }

        constitutive::StateView view(ctx.state_work);
        auto& counter = view.get<int>(0);
        counter += 1;

        Value<Dual> out;
        out.kind = Value<Dual>::Kind::Scalar;
        out.s = mul(input.s, static_cast<Real>(counter), makeDualConstant(0.0, ws.alloc()));
        return out;
    }

private:
    inline static const constitutive::StateLayout kStateLayout = []() {
        constitutive::StateLayoutBuilder b("StateLayoutOnlyCounterModel");
        b.add<int>("counter");
        return b.build();
    }();
};

struct UserContext {
    Real value{0.0};
};

class ParamsAndUserDataModel final : public ConstitutiveModel {
public:
    [[nodiscard]] Value<Real> evaluate(const Value<Real>& /*input*/, int /*dim*/) const override
    {
        throw std::logic_error("ParamsAndUserDataModel: unary evaluate() should not be called");
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& /*input*/,
                                       int /*dim*/,
                                       DualWorkspace& /*ws*/) const override
    {
        throw std::logic_error("ParamsAndUserDataModel: unary evaluate() should not be called (dual)");
    }

    [[nodiscard]] Value<Real> evaluate(const Value<Real>& input, const ConstitutiveEvalContext& ctx) const override
    {
        if (input.kind != Value<Real>::Kind::Scalar) {
            throw std::invalid_argument("ParamsAndUserDataModel: expected scalar input");
        }

        const Real k = ctx.realParam("k").value_or(0.0);
        const int i = ctx.paramAs<int>("i").value_or(0);
        const bool flag = ctx.paramAs<bool>("flag").value_or(false);
        const auto* uctx = static_cast<const UserContext*>(ctx.user_data);
        const Real u = uctx ? uctx->value : 0.0;

        const Real factor = k + static_cast<Real>(i) + (flag ? 1.0 : 0.0) + u;
        return Value<Real>{Value<Real>::Kind::Scalar, factor * input.s};
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& input,
                                       const ConstitutiveEvalContext& ctx,
                                       DualWorkspace& ws) const override
    {
        if (input.kind != Value<Dual>::Kind::Scalar) {
            throw std::invalid_argument("ParamsAndUserDataModel: expected scalar input (dual)");
        }

        const Real k = ctx.realParam("k").value_or(0.0);
        const int i = ctx.paramAs<int>("i").value_or(0);
        const bool flag = ctx.paramAs<bool>("flag").value_or(false);
        const auto* uctx = static_cast<const UserContext*>(ctx.user_data);
        const Real u = uctx ? uctx->value : 0.0;

        const Real factor = k + static_cast<Real>(i) + (flag ? 1.0 : 0.0) + u;

        Value<Dual> out;
        out.kind = Value<Dual>::Kind::Scalar;
        out.s = mul(input.s, factor, makeDualConstant(0.0, ws.alloc()));
        return out;
    }
};

class ExtendedParamsModel final : public ConstitutiveModel {
public:
    [[nodiscard]] Value<Real> evaluate(const Value<Real>& /*input*/, int /*dim*/) const override
    {
        throw std::logic_error("ExtendedParamsModel: unary evaluate() should not be called");
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& /*input*/,
                                       int /*dim*/,
                                       DualWorkspace& /*ws*/) const override
    {
        throw std::logic_error("ExtendedParamsModel: unary evaluate() should not be called (dual)");
    }

    [[nodiscard]] Value<Real> evaluate(const Value<Real>& input, const ConstitutiveEvalContext& ctx) const override
    {
        if (input.kind != Value<Real>::Kind::Scalar) {
            throw std::invalid_argument("ExtendedParamsModel: expected scalar input");
        }

        const Real k = ctx.paramAs<Real>("k").value_or(0.0);
        const auto name = ctx.paramAs<std::string>("name").value_or(std::string{});
        const auto vec = ctx.paramAs<std::vector<Real>>("vec").value_or(std::vector<Real>{});
        const auto M = ctx.paramAs<params::DenseMatrix>("M").value_or(params::DenseMatrix{});

        Real sum_vec = 0.0;
        for (const auto v : vec) sum_vec += v;
        Real sum_M = 0.0;
        for (const auto v : M.data) sum_M += v;

        const Real factor = k + static_cast<Real>(name.size()) + sum_vec + sum_M;
        return Value<Real>{Value<Real>::Kind::Scalar, factor * input.s};
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& input,
                                       const ConstitutiveEvalContext& ctx,
                                       DualWorkspace& ws) const override
    {
        if (input.kind != Value<Dual>::Kind::Scalar) {
            throw std::invalid_argument("ExtendedParamsModel: expected scalar input (dual)");
        }

        const Real k = ctx.paramAs<Real>("k").value_or(0.0);
        const auto name = ctx.paramAs<std::string>("name").value_or(std::string{});
        const auto vec = ctx.paramAs<std::vector<Real>>("vec").value_or(std::vector<Real>{});
        const auto M = ctx.paramAs<params::DenseMatrix>("M").value_or(params::DenseMatrix{});

        Real sum_vec = 0.0;
        for (const auto v : vec) sum_vec += v;
        Real sum_M = 0.0;
        for (const auto v : M.data) sum_M += v;

        const Real factor = k + static_cast<Real>(name.size()) + sum_vec + sum_M;

        Value<Dual> out;
        out.kind = Value<Dual>::Kind::Scalar;
        out.s = mul(input.s, factor, makeDualConstant(0.0, ws.alloc()));
        return out;
    }
};

class MultiOutputScaleModel final : public ConstitutiveModel {
public:
    [[nodiscard]] Value<Real> evaluate(const Value<Real>& input, int /*dim*/) const override
    {
        if (input.kind != Value<Real>::Kind::Scalar) {
            throw std::invalid_argument("MultiOutputScaleModel: expected scalar input");
        }
        return Value<Real>{Value<Real>::Kind::Scalar, input.s};
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& input,
                                       int /*dim*/,
                                       DualWorkspace& ws) const override
    {
        if (input.kind != Value<Dual>::Kind::Scalar) {
            throw std::invalid_argument("MultiOutputScaleModel: expected scalar input (dual)");
        }
        Value<Dual> out;
        out.kind = Value<Dual>::Kind::Scalar;
        out.s = input.s;
        (void)ws;
        return out;
    }

    [[nodiscard]] std::size_t outputCount() const noexcept override { return 2u; }

    void evaluateNaryOutputs(std::span<const Value<Real>> inputs,
                             const ConstitutiveEvalContext& /*ctx*/,
                             std::span<Value<Real>> outputs) const override
    {
        if (inputs.size() != 1u) {
            throw std::invalid_argument("MultiOutputScaleModel: expected 1 input");
        }
        if (outputs.size() != 2u) {
            throw std::invalid_argument("MultiOutputScaleModel: expected 2 outputs");
        }
        if (inputs[0].kind != Value<Real>::Kind::Scalar) {
            throw std::invalid_argument("MultiOutputScaleModel: expected scalar input");
        }

        ++real_calls;
        outputs[0] = Value<Real>{Value<Real>::Kind::Scalar, inputs[0].s};
        outputs[1] = Value<Real>{Value<Real>::Kind::Scalar, 2.0 * inputs[0].s};
    }

    void evaluateNaryOutputs(std::span<const Value<Dual>> inputs,
                             const ConstitutiveEvalContext& /*ctx*/,
                             DualWorkspace& ws,
                             std::span<Value<Dual>> outputs) const override
    {
        if (inputs.size() != 1u) {
            throw std::invalid_argument("MultiOutputScaleModel: expected 1 input (dual)");
        }
        if (outputs.size() != 2u) {
            throw std::invalid_argument("MultiOutputScaleModel: expected 2 outputs (dual)");
        }
        if (inputs[0].kind != Value<Dual>::Kind::Scalar) {
            throw std::invalid_argument("MultiOutputScaleModel: expected scalar input (dual)");
        }

        ++dual_calls;
        outputs[0].kind = Value<Dual>::Kind::Scalar;
        outputs[0].s = inputs[0].s;

        outputs[1].kind = Value<Dual>::Kind::Scalar;
        outputs[1].s = mul(inputs[0].s, 2.0, makeDualConstant(0.0, ws.alloc()));
    }

    mutable std::size_t real_calls{0};
    mutable std::size_t dual_calls{0};
};

class NonlocalEchoModel final : public ConstitutiveModel {
public:
    [[nodiscard]] Value<Real> evaluate(const Value<Real>& input, int /*dim*/) const override
    {
        if (input.kind != Value<Real>::Kind::Scalar) {
            throw std::invalid_argument("NonlocalEchoModel: expected scalar input");
        }
        return input;
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& input,
                                       int /*dim*/,
                                       DualWorkspace& /*ws*/) const override
    {
        if (input.kind != Value<Dual>::Kind::Scalar) {
            throw std::invalid_argument("NonlocalEchoModel: expected scalar input (dual)");
        }
        return input;
    }

    [[nodiscard]] Value<Real> evaluate(const Value<Real>& input, const ConstitutiveEvalContext& ctx) const override
    {
        if (input.kind != Value<Real>::Kind::Scalar) {
            throw std::invalid_argument("NonlocalEchoModel: expected scalar input");
        }

        FE_THROW_IF(!ctx.hasNonlocalAccess(), FEException,
                    "NonlocalEchoModel: missing nonlocal access");
        FE_THROW_IF(ctx.num_qpts == 0u, FEException,
                    "NonlocalEchoModel: invalid num_qpts");
        FE_THROW_IF(ctx.q >= ctx.num_qpts, FEException,
                    "NonlocalEchoModel: invalid q");

        const auto same = ctx.inputAt(/*input_index=*/0u, ctx.q);
        FE_THROW_IF(same.kind != input.kind, FEException,
                    "NonlocalEchoModel: nonlocal input kind mismatch");
        FE_THROW_IF(same.s != input.s, FEException,
                    "NonlocalEchoModel: nonlocal input value mismatch");

        const auto x = ctx.physicalPointAt(ctx.q);
        for (int d = 0; d < 3; ++d) {
            FE_THROW_IF(x[static_cast<std::size_t>(d)] != ctx.x[static_cast<std::size_t>(d)], FEException,
                        "NonlocalEchoModel: physicalPointAt mismatch");
        }

        (void)ctx.integrationWeightAt(ctx.q);
        return input;
    }
};

class InlinableScaleByParamModel final : public ConstitutiveModel, public InlinableConstitutiveModel {
public:
    [[nodiscard]] Value<Real> evaluate(const Value<Real>& /*input*/, int /*dim*/) const override
    {
        throw std::logic_error("InlinableScaleByParamModel: unary evaluate() should not be called");
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& /*input*/,
                                       int /*dim*/,
                                       DualWorkspace& /*ws*/) const override
    {
        throw std::logic_error("InlinableScaleByParamModel: unary evaluate() should not be called (dual)");
    }

    [[nodiscard]] Value<Real> evaluate(const Value<Real>& input, const ConstitutiveEvalContext& ctx) const override
    {
        if (input.kind != Value<Real>::Kind::Scalar) {
            throw std::invalid_argument("InlinableScaleByParamModel: expected scalar input");
        }
        const Real k = ctx.requireParamAs<Real>("k");
        ++real_calls;
        return Value<Real>{Value<Real>::Kind::Scalar, k * input.s};
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& input,
                                       const ConstitutiveEvalContext& ctx,
                                       DualWorkspace& ws) const override
    {
        if (input.kind != Value<Dual>::Kind::Scalar) {
            throw std::invalid_argument("InlinableScaleByParamModel: expected scalar input (dual)");
        }
        const Real k = ctx.requireParamAs<Real>("k");
        ++dual_calls;

        Value<Dual> out;
        out.kind = Value<Dual>::Kind::Scalar;
        out.s = mul(input.s, k, makeDualConstant(0.0, ws.alloc()));
        return out;
    }

    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override
    {
        return {params::Spec{.key = "k", .type = params::ValueType::Real, .required = true}};
    }

    [[nodiscard]] const InlinableConstitutiveModel* inlinable() const noexcept override { return this; }

    [[nodiscard]] std::uint64_t kindId() const noexcept override
    {
        return InlinableConstitutiveModel::fnv1a64("InlinableScaleByParamModel");
    }

    [[nodiscard]] MaterialStateAccess stateAccess() const noexcept override { return MaterialStateAccess::None; }

    [[nodiscard]] InlinedConstitutiveExpansion inlineExpand(std::span<const FormExpr> inputs,
                                                            const InlinableConstitutiveContext& /*ctx*/) const override
    {
        FE_THROW_IF(inputs.size() != 1u, InvalidArgumentException,
                    "InlinableScaleByParamModel: expected exactly 1 input");
        InlinedConstitutiveExpansion out;
        out.outputs.push_back(inputs[0] * FormExpr::parameter("k"));
        return out;
    }

    mutable std::size_t real_calls{0};
    mutable std::size_t dual_calls{0};
};

class InlinableTwoOutputParamModel final : public ConstitutiveModel, public InlinableConstitutiveModel {
public:
    [[nodiscard]] Value<Real> evaluate(const Value<Real>& /*input*/, int /*dim*/) const override
    {
        throw std::logic_error("InlinableTwoOutputParamModel: evaluate() should not be called after inlining");
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& /*input*/,
                                       int /*dim*/,
                                       DualWorkspace& /*ws*/) const override
    {
        throw std::logic_error("InlinableTwoOutputParamModel: evaluate() should not be called after inlining (dual)");
    }

    [[nodiscard]] std::size_t outputCount() const noexcept override { return 2u; }

    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override
    {
        return {params::Spec{.key = "k", .type = params::ValueType::Real, .required = true}};
    }

    [[nodiscard]] const InlinableConstitutiveModel* inlinable() const noexcept override { return this; }

    [[nodiscard]] std::uint64_t kindId() const noexcept override
    {
        return InlinableConstitutiveModel::fnv1a64("InlinableTwoOutputParamModel");
    }

    [[nodiscard]] MaterialStateAccess stateAccess() const noexcept override { return MaterialStateAccess::None; }

    [[nodiscard]] InlinedConstitutiveExpansion inlineExpand(std::span<const FormExpr> inputs,
                                                            const InlinableConstitutiveContext& /*ctx*/) const override
    {
        FE_THROW_IF(inputs.size() != 1u, InvalidArgumentException,
                    "InlinableTwoOutputParamModel: expected exactly 1 input");
        InlinedConstitutiveExpansion out;
        out.outputs.push_back(inputs[0]);
        out.outputs.push_back(inputs[0] * FormExpr::parameter("k"));
        return out;
    }
};

class InlinableStatefulWriteModel final : public ConstitutiveModel, public InlinableConstitutiveModel {
public:
    [[nodiscard]] Value<Real> evaluate(const Value<Real>& /*input*/, int /*dim*/) const override
    {
        throw std::logic_error("InlinableStatefulWriteModel: evaluate() should not be called after inlining");
    }

    [[nodiscard]] Value<Dual> evaluate(const Value<Dual>& /*input*/,
                                       int /*dim*/,
                                       DualWorkspace& /*ws*/) const override
    {
        throw std::logic_error("InlinableStatefulWriteModel: evaluate() should not be called after inlining (dual)");
    }

    [[nodiscard]] StateSpec stateSpec() const noexcept override
    {
        StateSpec s;
        s.bytes_per_qpt = sizeof(Real);
        s.alignment = alignof(Real);
        return s;
    }

    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override
    {
        return {params::Spec{.key = "k", .type = params::ValueType::Real, .required = true}};
    }

    [[nodiscard]] const InlinableConstitutiveModel* inlinable() const noexcept override { return this; }

    [[nodiscard]] std::uint64_t kindId() const noexcept override
    {
        return InlinableConstitutiveModel::fnv1a64("InlinableStatefulWriteModel");
    }

    [[nodiscard]] MaterialStateAccess stateAccess() const noexcept override { return MaterialStateAccess::ReadWrite; }

    [[nodiscard]] InlinedConstitutiveExpansion inlineExpand(std::span<const FormExpr> inputs,
                                                            const InlinableConstitutiveContext& ctx) const override
    {
        FE_THROW_IF(inputs.size() != 1u, InvalidArgumentException,
                    "InlinableStatefulWriteModel: expected exactly 1 input");

        const std::uint32_t state_off = ctx.state_base_offset_bytes;
        const auto state_value = inputs[0] * FormExpr::parameter("k");

        InlinedConstitutiveExpansion out;
        out.state_updates.push_back(MaterialStateUpdateOp{.offset_bytes = state_off, .value = state_value});
        out.outputs.push_back(FormExpr::materialStateWorkRef(state_off));
        return out;
    }
};

static dofs::DofMap createSingleTetraP0DofMap()
{
    dofs::DofMap dof_map(1, 1, 1);
    std::vector<GlobalIndex> cell_dofs = {0};
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(1);
    dof_map.setNumLocalDofs(1);
    dof_map.finalize();
    return dof_map;
}

TEST(ConstitutiveModelTest, BilinearScalingOfGradients)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const Real k = 3.0;
    auto model = std::make_shared<ScaleVectorModel>(k);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = inner(FormExpr::constitutive(model, grad(u)), grad(v)).dx();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(4);
    mat.zero();

    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);

    const Real V = 1.0 / 6.0;
    const Real e00 = k * (3.0 * V);
    const Real em = k * (-1.0 * V);
    const Real e11 = k * (1.0 * V);

    const Real expected[4][4] = {
        {e00, em,  em,  em},
        {em,  e11, 0.0, 0.0},
        {em,  0.0, e11, 0.0},
        {em,  0.0, 0.0, e11}
    };

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected[i][j], 1e-12);
        }
    }
}

TEST(ConstitutiveModelTest, SupportsMultiInputConstitutiveNodes)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    auto model = std::make_shared<SumTwoScalarsModel>();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto Ku_variadic = constitutive(model, u, u);
    const auto Ku_list = constitutive(model, {u, u});
    const auto residual = ((Ku_variadic + Ku_list) * v).dx();

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.2, -0.1, 0.05, 0.4};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();

    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    const Real V = 1.0 / 6.0;
    const Real diag = 4.0 * (V / 10.0);
    const Real off = 4.0 * (V / 20.0);

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = (i == j) ? diag : off;
            EXPECT_NEAR(J.getMatrixEntry(i, j), expected, 1e-12);
        }
    }

    for (GlobalIndex i = 0; i < 4; ++i) {
        Real expected = 0.0;
        for (GlobalIndex j = 0; j < 4; ++j) {
            expected += ((i == j) ? diag : off) * U[static_cast<std::size_t>(j)];
        }
        EXPECT_NEAR(R.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(ConstitutiveEvalContextTest, ProvidesRequireAndDefaultedParameterAccessors)
{
    std::function<std::optional<Real>(std::string_view)> get_real_param =
        [](std::string_view key) -> std::optional<Real> {
        if (key == "k") return 2.0;
        return std::nullopt;
    };

    std::function<std::optional<params::Value>(std::string_view)> get_param =
        [](std::string_view key) -> std::optional<params::Value> {
        if (key == "name") return params::Value{std::string("abc")};
        return std::nullopt;
    };

    ConstitutiveEvalContext ctx;
    ctx.get_real_param = &get_real_param;
    ctx.get_param = &get_param;

    EXPECT_DOUBLE_EQ(ctx.paramOr<Real>("k", 0.0), 2.0);
    EXPECT_DOUBLE_EQ(ctx.requireParamAs<Real>("k"), 2.0);
    EXPECT_DOUBLE_EQ(ctx.paramOr<Real>("missing", 3.0), 3.0);
    EXPECT_THROW((void)ctx.requireParamAs<Real>("missing"), std::invalid_argument);

    EXPECT_EQ(ctx.paramOr<std::string>("name", "def"), "abc");
    EXPECT_EQ(ctx.requireParamAs<std::string>("name"), "abc");
    EXPECT_EQ(ctx.paramOr<std::string>("missing", "def"), "def");
    EXPECT_THROW((void)ctx.requireParamAs<std::vector<Real>>("name"), std::invalid_argument);
}

TEST(ConstitutiveModelTest, PlumbsTimeDtAndParametersIntoConstitutiveContext)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    auto model = std::make_shared<ContextFactorModel>();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto Ku = FormExpr::constitutive(model, u);
    const auto residual = (Ku * v).dx();

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setTime(1.0);
    assembler.setTimeStep(0.5);

    std::function<std::optional<Real>(std::string_view)> get_param =
        [](std::string_view key) -> std::optional<Real> {
        if (key == "k") return 2.0;
        return std::nullopt;
    };
    assembler.setRealParameterGetter(&get_param);

    std::vector<Real> U = {0.2, -0.1, 0.05, 0.4};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();

    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    const Real V = 1.0 / 6.0;
    const Real factor = 2.0 + 1.0 + 0.5;
    const Real diag = factor * (V / 10.0);
    const Real off = factor * (V / 20.0);

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = (i == j) ? diag : off;
            EXPECT_NEAR(J.getMatrixEntry(i, j), expected, 1e-12);
        }
    }

    for (GlobalIndex i = 0; i < 4; ++i) {
        Real expected = 0.0;
        for (GlobalIndex j = 0; j < 4; ++j) {
            expected += ((i == j) ? diag : off) * U[static_cast<std::size_t>(j)];
        }
        EXPECT_NEAR(R.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(ConstitutiveModelTest, PlumbsMaterialStateIntoConstitutiveContext)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraP0DofMap();
    spaces::L2Space space(ElementType::Tetra4, 0);

    auto model = std::make_shared<StatefulCounterModel>();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto Ku = FormExpr::constitutive(model, u);
    const auto form = (Ku * v).dx();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    systems::MaterialStateProvider provider(/*num_cells=*/mesh.numCells());
    provider.addKernel(kernel, kernel.materialStateSpec(), /*max_qpts=*/64);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setMaterialStateProvider(&provider);

    assembly::DenseMatrixView mat(1);
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
    EXPECT_NEAR(mat.getMatrixEntry(0, 0), 1.0 / 6.0, 1e-12);

    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
    EXPECT_NEAR(mat.getMatrixEntry(0, 0), 2.0 / 6.0, 1e-12);
}

TEST(ConstitutiveModelTest, StateLayoutMetadataDrivesMaterialStateAllocation)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraP0DofMap();
    spaces::L2Space space(ElementType::Tetra4, 0);

    auto model = std::make_shared<StateLayoutOnlyCounterModel>();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto Ku = FormExpr::constitutive(model, u);
    const auto form = (Ku * v).dx();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    systems::MaterialStateProvider provider(/*num_cells=*/mesh.numCells());
    provider.addKernel(kernel, kernel.materialStateSpec(), /*max_qpts=*/64);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setMaterialStateProvider(&provider);

    assembly::DenseMatrixView mat(1);
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
    EXPECT_NEAR(mat.getMatrixEntry(0, 0), 1.0 / 6.0, 1e-12);

    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
    EXPECT_NEAR(mat.getMatrixEntry(0, 0), 2.0 / 6.0, 1e-12);
}

TEST(ConstitutiveModelTest, PlumbsTypedParametersAndUserDataIntoConstitutiveContext)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    auto model = std::make_shared<ParamsAndUserDataModel>();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto Ku = FormExpr::constitutive(model, u);
    const auto form = (Ku * v).dx();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    UserContext user;
    user.value = 4.0;

    std::function<std::optional<params::Value>(std::string_view)> get_param =
        [](std::string_view key) -> std::optional<params::Value> {
        if (key == "k") return params::Value{Real(5.0)};
        if (key == "i") return params::Value{3};
        if (key == "flag") return params::Value{true};
        return std::nullopt;
    };

    std::function<std::optional<Real>(std::string_view)> get_real_param =
        [](std::string_view key) -> std::optional<Real> {
        if (key == "k") return 2.0;
        return std::nullopt;
    };

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setUserData(&user);
    assembler.setParameterGetter(&get_param);
    assembler.setRealParameterGetter(&get_real_param);

    assembly::DenseMatrixView mat(4);
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);

    const Real V = 1.0 / 6.0;
    const Real factor = 5.0 + 3.0 + 1.0 + 4.0;
    const Real diag = factor * (V / 10.0);
    const Real off = factor * (V / 20.0);

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = (i == j) ? diag : off;
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 1e-12);
        }
    }
}

TEST(ConstitutiveModelTest, PlumbsExtendedParameterTypesIntoConstitutiveContext)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    auto model = std::make_shared<ExtendedParamsModel>();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto Ku = FormExpr::constitutive(model, u);
    const auto form = (Ku * v).dx();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    std::function<std::optional<params::Value>(std::string_view)> get_param =
        [](std::string_view key) -> std::optional<params::Value> {
        if (key == "k") return params::Value{Real(2.0)};
        if (key == "name") return params::Value{std::string("abc")};
        if (key == "vec") return params::Value{std::vector<Real>{1.0, 2.0}};
        if (key == "M") return params::Value{params::DenseMatrix{2u, 2u, std::vector<Real>{1.0, 1.0, 1.0, 1.0}}};
        return std::nullopt;
    };

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setParameterGetter(&get_param);

    assembly::DenseMatrixView mat(4);
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);

    const Real V = 1.0 / 6.0;
    const Real factor = 2.0 + 3.0 + 3.0 + 4.0;
    const Real diag = factor * (V / 10.0);
    const Real off = factor * (V / 20.0);

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = (i == j) ? diag : off;
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 1e-12);
        }
    }
}

TEST(ConstitutiveModelTest, SupportsMultiOutputConstitutiveNodes)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    auto model = std::make_shared<MultiOutputScaleModel>();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto call = constitutive(model, u);
    const auto out1 = call.out(1u);
    const auto form = ((call + out1) * v).dx();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(4);
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);

    const Real V = 1.0 / 6.0;
    const Real factor = 3.0;
    const Real diag = factor * (V / 10.0);
    const Real off = factor * (V / 20.0);

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = (i == j) ? diag : off;
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 1e-12);
        }
    }

    // This constitutive call depends on TrialFunction, so caching reuses values across
    // test-basis loops but not across trial-basis loops.
    auto quad_rule = space.getElement(ElementType::Tetra4, 0).quadrature();
    if (!quad_rule) {
        const int order = quadrature::QuadratureFactory::recommended_order(
            space.getElement(ElementType::Tetra4, 0).polynomial_order(), false);
        quad_rule = quadrature::QuadratureFactory::create(ElementType::Tetra4, order);
    }
    ASSERT_TRUE(static_cast<bool>(quad_rule));
    const auto n_qpts = static_cast<std::size_t>(quad_rule->num_points());
    const auto n_trial = dof_map.getCellDofs(0).size();
    EXPECT_EQ(model->real_calls, n_qpts * n_trial);
}

TEST(ConstitutiveModelTest, ProvidesNonlocalInputAccessToConstitutiveContext)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    auto model = std::make_shared<NonlocalEchoModel>();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto Ku = FormExpr::constitutive(model, u);
    const auto form = (Ku * v).dx();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(4);
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);

    const Real V = 1.0 / 6.0;
    const Real diag = V / 10.0;
    const Real off = V / 20.0;

    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = (i == j) ? diag : off;
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 1e-12);
        }
    }
}

TEST(ConstitutiveModelTest, PlumbsMaterialStateIntoBoundaryFaceConstitutiveContext)
{
    constexpr int kMarker = 2;
    SingleTetraOneBoundaryFaceMeshAccess mesh(kMarker);
    auto dof_map = createSingleTetraP0DofMap();
    spaces::L2Space space(ElementType::Tetra4, 0);

    auto model = std::make_shared<StatefulCounterModel>();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto Ku = FormExpr::constitutive(model, u);
    const auto form = (Ku * v).ds(kMarker);

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    systems::MaterialStateProvider provider(/*num_cells=*/mesh.numCells(),
                                            /*boundary_face_ids=*/std::vector<GlobalIndex>{0},
                                            /*interior_face_ids=*/{});
    provider.addKernel(kernel, kernel.materialStateSpec(),
                       /*max_cell_qpts=*/0,
                       /*max_boundary_face_qpts=*/64,
                       /*max_interior_face_qpts=*/0);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setMaterialStateProvider(&provider);

    assembly::DenseMatrixView mat(1);
    mat.zero();
    (void)assembler.assembleBoundaryFaces(mesh, kMarker, space, space, kernel, &mat, nullptr);
    EXPECT_NEAR(mat.getMatrixEntry(0, 0), 0.5, 1e-12);

    mat.zero();
    (void)assembler.assembleBoundaryFaces(mesh, kMarker, space, space, kernel, &mat, nullptr);
    EXPECT_NEAR(mat.getMatrixEntry(0, 0), 1.0, 1e-12);
}

TEST(ConstitutiveModelTest, ResidualAndJacobianScalingOfMassMatrix)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const Real k = 2.0;
    auto model = std::make_shared<ScaleScalarModel>(k);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto Ku = FormExpr::constitutive(model, u);
    const auto residual = (Ku * v).dx();

    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    std::vector<Real> U = {0.2, -0.1, 0.05, 0.4};
    assembler.setCurrentSolution(U);

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();

    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);

    const Real V = 1.0 / 6.0;
    const Real diag = k * (V / 10.0);
    const Real off = k * (V / 20.0);

    // J should be k * mass matrix
    for (GlobalIndex i = 0; i < 4; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            const Real expected = (i == j) ? diag : off;
            EXPECT_NEAR(J.getMatrixEntry(i, j), expected, 1e-12);
        }
    }

    // R should be J * U
    for (GlobalIndex i = 0; i < 4; ++i) {
        Real expected = 0.0;
        for (GlobalIndex j = 0; j < 4; ++j) {
            expected += ((i == j) ? diag : off) * U[static_cast<std::size_t>(j)];
        }
        EXPECT_NEAR(R.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(ConstitutiveModelTest, NonInlinableConstitutiveRejectedInStrictButAllowedInRelaxedMode)
{
    auto model = std::make_shared<ScaleScalarModel>(Real(2.0));
    const auto expr = FormExpr::constitutive(model, FormExpr::constant(Real(1.0)));

    const auto strict = jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_FALSE(strict.ok);
    ASSERT_TRUE(strict.first_issue.has_value());
    EXPECT_EQ(strict.first_issue->type, FormExprType::Constitutive);

    const auto relaxed =
        jit::canCompile(expr, jit::ValidationOptions{.strictness = jit::Strictness::AllowExternalCalls});
    EXPECT_TRUE(relaxed.ok);
    EXPECT_FALSE(relaxed.cacheable);
}

TEST(ConstitutiveModelTest, InlinableConstitutiveMatchesInterpreterAndAvoidsVirtualCalls)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraP0DofMap();
    spaces::L2Space space(ElementType::Tetra4, 0);

    auto model = std::make_shared<InlinableScaleByParamModel>();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto K = FormExpr::constitutive(model, FormExpr::constant(2.0));
    const auto form = (K * u * v).dx();

    // Interpreter baseline: model called with string parameter getter.
    auto ir_baseline = compiler.compileBilinear(form);
    FormKernel baseline_kernel(std::move(ir_baseline));

    std::function<std::optional<Real>(std::string_view)> get_real_param =
        [](std::string_view key) -> std::optional<Real> {
        if (key == "k") return Real(3.0);
        return std::nullopt;
    };

    assembly::StandardAssembler baseline_assembler;
    baseline_assembler.setDofMap(dof_map);
    baseline_assembler.setRealParameterGetter(&get_real_param);

    assembly::DenseMatrixView mat_baseline(1);
    mat_baseline.zero();
    (void)baseline_assembler.assembleMatrix(mesh, space, space, baseline_kernel, mat_baseline);

    const auto calls_after_baseline = model->real_calls;
    EXPECT_GT(calls_after_baseline, 0u);

    // Inlined + slot-resolved: no virtual call, no string lookup.
    auto ir_inlined = compiler.compileBilinear(form);
    FormKernel inlined_kernel(std::move(ir_inlined));
    inlined_kernel.resolveInlinableConstitutives();
    inlined_kernel.resolveParameterSlots([](std::string_view key) -> std::optional<std::uint32_t> {
        if (key == "k") return 0u;
        return std::nullopt;
    });

    assembly::StandardAssembler inlined_assembler;
    inlined_assembler.setDofMap(dof_map);
    const std::vector<Real> inlined_constants = {Real(3.0)};
    inlined_assembler.setJITConstants(inlined_constants);

    assembly::DenseMatrixView mat_inlined(1);
    mat_inlined.zero();
    (void)inlined_assembler.assembleMatrix(mesh, space, space, inlined_kernel, mat_inlined);

    EXPECT_NEAR(mat_inlined.getMatrixEntry(0, 0), mat_baseline.getMatrixEntry(0, 0), 1e-12);
    EXPECT_EQ(model->real_calls, calls_after_baseline);
}

TEST(ConstitutiveModelTest, InlinableConstitutiveOutputInlinesAndPassesStrictJITValidation)
{
    spaces::L2Space space(ElementType::Tetra4, 0);
    auto model = std::make_shared<InlinableTwoOutputParamModel>();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto call = constitutive(model, FormExpr::constant(2.0));
    const auto K = call.out(1);
    const auto form = (K * u * v).dx();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    {
        const auto r = jit::canCompile(kernel.ir(), jit::ValidationOptions{.strictness = jit::Strictness::Strict});
        EXPECT_FALSE(r.ok);
        ASSERT_TRUE(r.first_issue.has_value());
        EXPECT_EQ(r.first_issue->type, FormExprType::Constitutive);
    }

    kernel.resolveInlinableConstitutives();
    kernel.resolveParameterSlots([](std::string_view key) -> std::optional<std::uint32_t> {
        if (key == "k") return 0u;
        return std::nullopt;
    });

    const auto r = jit::canCompile(kernel.ir(), jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_TRUE(r.ok);
    EXPECT_TRUE(r.cacheable);
}

TEST(ConstitutiveModelTest, InlinableStatefulConstitutiveWritesStateAndUsesSlotParameters)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraP0DofMap();
    spaces::L2Space space(ElementType::Tetra4, 0);

    auto model = std::make_shared<InlinableStatefulWriteModel>();

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto K = FormExpr::constitutive(model, FormExpr::constant(2.0));
    const auto form = (K * u * v).dx();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    kernel.resolveInlinableConstitutives();
    kernel.resolveParameterSlots([](std::string_view key) -> std::optional<std::uint32_t> {
        if (key == "k") return 0u;
        return std::nullopt;
    });

    const auto jit_check =
        jit::canCompile(kernel.ir(), jit::ValidationOptions{.strictness = jit::Strictness::Strict});
    EXPECT_TRUE(jit_check.ok);
    EXPECT_TRUE(jit_check.cacheable);

    systems::MaterialStateProvider provider(/*num_cells=*/mesh.numCells());
    provider.addKernel(kernel, kernel.materialStateSpec(), /*max_qpts=*/64);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setMaterialStateProvider(&provider);

    const std::vector<Real> constants = {3.0};
    assembler.setJITConstants(constants);

    assembly::DenseMatrixView mat(1);
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);

    // input=2, k=3 => state=6. For a unit tetra: V=1/6 => entry=6*V=1.
    EXPECT_NEAR(mat.getMatrixEntry(0, 0), 1.0, 1e-12);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
