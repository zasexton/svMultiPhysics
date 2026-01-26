/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/CoupledBoundaryManager.h"

#include "Assembly/FunctionalAssembler.h"
#include "Forms/BoundaryFunctional.h"
#include "Forms/PointEvaluator.h"
#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

namespace {

class BoundaryMeasureKernel final : public assembly::FunctionalKernel {
public:
    [[nodiscard]] assembly::RequiredData getRequiredData() const noexcept override
    {
        return assembly::RequiredData::IntegrationWeights;
    }

    [[nodiscard]] bool hasCell() const noexcept override { return false; }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }

    [[nodiscard]] Real evaluateCell(const assembly::AssemblyContext& /*ctx*/, LocalIndex /*q*/) override
    {
        return 0.0;
    }

    [[nodiscard]] Real evaluateBoundaryFace(const assembly::AssemblyContext& /*ctx*/,
                                            LocalIndex /*q*/,
                                            int /*boundary_marker*/) override
    {
        return 1.0;
    }

    [[nodiscard]] std::string name() const override { return "BoundaryMeasure"; }
};

void gatherParameterSymbols(const forms::FormExprNode& node, std::vector<std::string_view>& names)
{
    if (node.type() == forms::FormExprType::ParameterSymbol) {
        const auto nm = node.symbolName();
        if (nm && !nm->empty()) {
            names.push_back(*nm);
        }
    }

    for (const auto& child : node.childrenShared()) {
        if (child) gatherParameterSymbols(*child, names);
    }
}

} // namespace

CoupledBoundaryManager::CoupledBoundaryManager(FESystem& system, FieldId primary_field)
    : system_(system)
    , primary_field_(primary_field)
{
    FE_THROW_IF(primary_field_ == INVALID_FIELD_ID, InvalidArgumentException,
                "CoupledBoundaryManager: primary_field is invalid");
}

CoupledBoundaryManager::~CoupledBoundaryManager() = default;

void CoupledBoundaryManager::requireSetup() const
{
    FE_THROW_IF(!system_.isSetup(), InvalidArgumentException,
                "CoupledBoundaryManager: system.setup() has not been called");
}

void CoupledBoundaryManager::setCompilerOptions(forms::SymbolicOptions options)
{
    compiler_options_ = std::move(options);

    // Invalidate any already-compiled boundary functional kernels so they can be
    // recompiled under the updated options on demand.
    for (auto& entry : functionals_) {
        entry.kernel.reset();
    }
}

void CoupledBoundaryManager::addBoundaryFunctional(forms::BoundaryFunctional functional)
{
    FE_THROW_IF(functional.name.empty(), InvalidArgumentException,
                "CoupledBoundaryManager::addBoundaryFunctional: empty name");
    FE_THROW_IF(!functional.integrand.isValid(), InvalidArgumentException,
                "CoupledBoundaryManager::addBoundaryFunctional: invalid integrand");

    auto it = name_to_functional_.find(functional.name);
    if (it != name_to_functional_.end()) {
        const auto& existing = functionals_.at(it->second).def;
        FE_THROW_IF(existing.boundary_marker != functional.boundary_marker, InvalidArgumentException,
                    "CoupledBoundaryManager::addBoundaryFunctional: name '" + functional.name + "' already registered with different boundary_marker");
        FE_THROW_IF(existing.reduction != functional.reduction, InvalidArgumentException,
                    "CoupledBoundaryManager::addBoundaryFunctional: name '" + functional.name + "' already registered with different reduction");
        FE_THROW_IF(existing.integrand.toString() != functional.integrand.toString(), InvalidArgumentException,
                    "CoupledBoundaryManager::addBoundaryFunctional: name '" + functional.name + "' already registered with different integrand");
        return;
    }

    const auto idx = functionals_.size();
    functionals_.push_back(CompiledFunctional{std::move(functional), nullptr});
    name_to_functional_.emplace(functionals_.back().def.name, idx);

    // Pre-register the name in the results container so the index is stable.
    integrals_.set(functionals_.back().def.name, 0.0);
}

void CoupledBoundaryManager::addCoupledNeumannBC(constraints::CoupledNeumannBC bc)
{
    for (const auto& f : bc.requiredIntegrals()) {
        addBoundaryFunctional(f);
    }
    coupled_neumann_.push_back(std::move(bc));
}

void CoupledBoundaryManager::addCoupledRobinBC(constraints::CoupledRobinBC bc)
{
    for (const auto& f : bc.requiredIntegrals()) {
        addBoundaryFunctional(f);
    }
    coupled_robin_.push_back(std::move(bc));
}

void CoupledBoundaryManager::addAuxiliaryState(AuxiliaryStateRegistration registration)
{
    for (const auto& f : registration.required_integrals) {
        addBoundaryFunctional(f);
    }

    FE_THROW_IF(!registration.rhs.isValid(), InvalidArgumentException,
                "CoupledBoundaryManager::addAuxiliaryState: RHS expression is invalid");
    FE_THROW_IF(registration.spec.size != 1, NotImplementedException,
                "CoupledBoundaryManager: auxiliary state spec.size != 1 is not supported yet");

    // Allow repeated registration (common when multiple coupled helpers are called),
    // but do not duplicate storage or integrate twice.
    for (const auto& existing : aux_registrations_) {
        if (existing.spec.name == registration.spec.name) {
            return;
        }
    }

    aux_state_.registerState(registration.spec, registration.initial_values);

    // Resolve this variable's slot now (stable after registerState).
    const auto slot_u = aux_state_.indexOf(registration.spec.name);
    FE_THROW_IF(slot_u > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()),
                InvalidArgumentException,
                "CoupledBoundaryManager::addAuxiliaryState: auxiliary slot overflow");
    registration.slot = static_cast<std::uint32_t>(slot_u);

    // Resolve coupled placeholders in RHS/Jacobian to slot-based terminals.
    auto resolve_expr = [&](const forms::FormExpr& expr) -> forms::FormExpr {
        return expr.transformNodes([&](const forms::FormExprNode& n) -> std::optional<forms::FormExpr> {
            if (n.type() == forms::FormExprType::AuxiliaryStateSymbol) {
                const auto nm = n.symbolName();
                FE_THROW_IF(!nm || nm->empty(), InvalidArgumentException,
                            "CoupledBoundaryManager: auxiliaryState(...) missing name in ODE expression");
                const auto idx = aux_state_.indexOf(*nm);
                FE_THROW_IF(idx > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()),
                            InvalidArgumentException,
                            "CoupledBoundaryManager: auxiliary slot overflow in ODE expression");
                return forms::FormExpr::auxiliaryStateRef(static_cast<std::uint32_t>(idx));
            }

            if (n.type() == forms::FormExprType::BoundaryFunctionalSymbol ||
                n.type() == forms::FormExprType::BoundaryIntegralSymbol) {
                const auto nm = n.symbolName();
                FE_THROW_IF(!nm || nm->empty(), InvalidArgumentException,
                            "CoupledBoundaryManager: boundary integral symbol missing name in ODE expression");
                const auto idx = integrals_.indexOf(*nm);
                FE_THROW_IF(idx > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()),
                            InvalidArgumentException,
                            "CoupledBoundaryManager: boundary integral slot overflow in ODE expression");
                return forms::FormExpr::boundaryIntegralRef(static_cast<std::uint32_t>(idx));
            }

            return std::nullopt;
        });
    };

    registration.rhs = resolve_expr(registration.rhs);
    if (registration.d_rhs_dX && registration.d_rhs_dX->isValid()) {
        registration.d_rhs_dX = resolve_expr(*registration.d_rhs_dX);
    }

    aux_registrations_.push_back(std::move(registration));
}

std::vector<params::Spec> CoupledBoundaryManager::parameterSpecs() const
{
    std::vector<std::string_view> param_names;

    for (const auto& entry : functionals_) {
        const auto* root = entry.def.integrand.node();
        if (!root) continue;
        gatherParameterSymbols(*root, param_names);
    }

    for (const auto& reg : aux_registrations_) {
        if (reg.rhs.isValid()) {
            if (const auto* root = reg.rhs.node()) {
                gatherParameterSymbols(*root, param_names);
            }
        }
        if (reg.d_rhs_dX && reg.d_rhs_dX->isValid()) {
            if (const auto* root = reg.d_rhs_dX->node()) {
                gatherParameterSymbols(*root, param_names);
            }
        }
    }

    if (param_names.empty()) {
        return {};
    }

    std::vector<std::string> keys;
    keys.reserve(param_names.size());
    for (const auto nm : param_names) {
        keys.emplace_back(nm);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    std::vector<params::Spec> out;
    out.reserve(keys.size());
    for (auto& key : keys) {
        out.push_back(params::Spec{.key = std::move(key),
                                   .type = params::ValueType::Real,
                                   .required = true});
    }
    return out;
}

void CoupledBoundaryManager::resolveParameterSlots(
    const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param)
{
    auto resolve_expr = [&](forms::FormExpr expr, std::string_view where) -> forms::FormExpr {
        if (!expr.isValid()) {
            return expr;
        }

        return expr.transformNodes([&](const forms::FormExprNode& n) -> std::optional<forms::FormExpr> {
            if (n.type() != forms::FormExprType::ParameterSymbol) {
                return std::nullopt;
            }
            const auto key = n.symbolName();
            FE_THROW_IF(!key || key->empty(), InvalidArgumentException,
                        std::string(where) + ": ParameterSymbol node missing name");
            const auto slot = slot_of_real_param(*key);
            FE_THROW_IF(!slot.has_value(), InvalidArgumentException,
                        std::string(where) + ": could not resolve parameter slot for '" + std::string(*key) + "'");
            return forms::FormExpr::parameterRef(*slot);
        });
    };

    for (auto& entry : functionals_) {
        entry.def.integrand = resolve_expr(entry.def.integrand, "CoupledBoundaryManager::functionals");
        entry.kernel.reset();
    }
    for (auto& reg : aux_registrations_) {
        reg.rhs = resolve_expr(reg.rhs, "CoupledBoundaryManager::aux_state_rhs");
        if (reg.d_rhs_dX && reg.d_rhs_dX->isValid()) {
            reg.d_rhs_dX = resolve_expr(*reg.d_rhs_dX, "CoupledBoundaryManager::aux_state_jacobian");
        }
    }
}

void CoupledBoundaryManager::compileFunctionalIfNeeded(const forms::BoundaryFunctional& functional)
{
    auto it = name_to_functional_.find(functional.name);
    FE_THROW_IF(it == name_to_functional_.end(), InvalidArgumentException,
                "CoupledBoundaryManager: unknown boundary functional '" + functional.name + "'");
    auto& entry = functionals_.at(it->second);
    if (entry.kernel) return;
    entry.kernel = forms::compileBoundaryFunctionalKernel(entry.def, compiler_options_);
}

Real CoupledBoundaryManager::boundaryMeasure(int boundary_marker, const SystemStateView& state)
{
    auto it = boundary_measure_cache_.find(boundary_marker);
    if (it != boundary_measure_cache_.end()) {
        return it->second;
    }

    requireSetup();

    const auto& rec = system_.fieldRecord(primary_field_);
    FE_CHECK_NOT_NULL(rec.space.get(), "CoupledBoundaryManager::boundaryMeasure: field space");

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(system_.meshAccess());
    assembler.setDofMap(system_.dofHandler().getDofMap());
    assembler.setSpace(*rec.space);
    assembler.setPrimaryField(primary_field_);
    assembler.setTimeIntegrationContext(state.time_integration);
    assembler.setTime(static_cast<Real>(state.time));
    assembler.setTimeStep(static_cast<Real>(state.dt));
    const auto& preg = system_.parameterRegistry();
    const bool have_param_contracts = !preg.specs().empty();
    std::function<std::optional<Real>(std::string_view)> get_real_param_wrapped{};
    std::function<std::optional<params::Value>(std::string_view)> get_param_wrapped{};
    if (have_param_contracts) {
        get_real_param_wrapped = preg.makeRealGetter(state);
        get_param_wrapped = preg.makeParamGetter(state);
    }
    assembler.setRealParameterGetter(have_param_contracts
                                         ? &get_real_param_wrapped
                                         : (state.getRealParam ? &state.getRealParam : nullptr));
    assembler.setParameterGetter(have_param_contracts
                                     ? &get_param_wrapped
                                     : (state.getParam ? &state.getParam : nullptr));
    assembler.setUserData(state.user_data);
    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> jit_constants;
    if (have_param_contracts && preg.slotCount() > 0u) {
        const auto slots = preg.evaluateRealSlots(state);
        jit_constants.assign(slots.begin(), slots.end());
        assembler.setJITConstants(jit_constants);
    } else {
        assembler.setJITConstants({});
    }
    assembler.setCoupledValues({}, {});

    BoundaryMeasureKernel measure_kernel;
    const Real area = assembler.assembleBoundaryScalar(measure_kernel, boundary_marker);
    boundary_measure_cache_.emplace(boundary_marker, area);
    return area;
}

Real CoupledBoundaryManager::evaluateFunctional(const CompiledFunctional& entry, const SystemStateView& state)
{
    requireSetup();
    FE_CHECK_NOT_NULL(entry.kernel.get(), "CoupledBoundaryManager::evaluateFunctional: kernel");

    const auto& rec = system_.fieldRecord(primary_field_);
    FE_CHECK_NOT_NULL(rec.space.get(), "CoupledBoundaryManager::evaluateFunctional: field space");

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(system_.meshAccess());
    assembler.setDofMap(system_.dofHandler().getDofMap());
    assembler.setSpace(*rec.space);
    assembler.setPrimaryField(primary_field_);
    assembler.setSolution(state.u);
    assembler.setTimeIntegrationContext(state.time_integration);
    assembler.setTime(static_cast<Real>(state.time));
    assembler.setTimeStep(static_cast<Real>(state.dt));
    const auto& preg = system_.parameterRegistry();
    const bool have_param_contracts = !preg.specs().empty();
    std::function<std::optional<Real>(std::string_view)> get_real_param_wrapped{};
    std::function<std::optional<params::Value>(std::string_view)> get_param_wrapped{};
    if (have_param_contracts) {
        get_real_param_wrapped = preg.makeRealGetter(state);
        get_param_wrapped = preg.makeParamGetter(state);
    }
    assembler.setRealParameterGetter(have_param_contracts
                                         ? &get_real_param_wrapped
                                         : (state.getRealParam ? &state.getRealParam : nullptr));
    assembler.setParameterGetter(have_param_contracts
                                     ? &get_param_wrapped
                                     : (state.getParam ? &state.getParam : nullptr));
    assembler.setUserData(state.user_data);
    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> jit_constants;
    if (have_param_contracts && preg.slotCount() > 0u) {
        const auto slots = preg.evaluateRealSlots(state);
        jit_constants.assign(slots.begin(), slots.end());
        assembler.setJITConstants(jit_constants);
    } else {
        assembler.setJITConstants({});
    }
    assembler.setCoupledValues({}, {});

    if (!state.u_history.empty()) {
        for (std::size_t k = 0; k < state.u_history.size(); ++k) {
            assembler.setPreviousSolutionK(static_cast<int>(k + 1), state.u_history[k]);
        }
    } else {
        assembler.setPreviousSolution(state.u_prev);
        assembler.setPreviousSolution2(state.u_prev2);
    }

    Real raw = assembler.assembleBoundaryScalar(*entry.kernel, entry.def.boundary_marker);

    switch (entry.def.reduction) {
        case forms::BoundaryFunctional::Reduction::Sum:
            return raw;
        case forms::BoundaryFunctional::Reduction::Average: {
            const Real area = boundaryMeasure(entry.def.boundary_marker, state);
            FE_THROW_IF(std::abs(area) < 1e-14, InvalidArgumentException,
                        "CoupledBoundaryManager: boundary measure is near zero for Average reduction");
            return raw / area;
        }
        case forms::BoundaryFunctional::Reduction::Max:
        case forms::BoundaryFunctional::Reduction::Min:
            FE_THROW(NotImplementedException,
                     "CoupledBoundaryManager: Max/Min reductions are not implemented");
    }

    return raw;
}

void CoupledBoundaryManager::prepareForAssembly(const SystemStateView& state)
{
    requireSetup();

    ctx_.t = static_cast<Real>(state.time);
    ctx_.dt = static_cast<Real>(state.dt);

    aux_state_.resetToCommitted();

    for (auto& entry : functionals_) {
        if (!entry.kernel) {
            compileFunctionalIfNeeded(entry.def);
        }
        const Real v = evaluateFunctional(entry, state);
        integrals_.set(entry.def.name, v);
    }

    for (auto& reg : aux_registrations_) {
        FE_THROW_IF(!reg.rhs.isValid(), InvalidArgumentException,
                    "CoupledBoundaryManager::prepareForAssembly: invalid RHS expression");
        FE_THROW_IF(reg.slot == AuxiliaryStateRegistration::kInvalidSlot, InvalidStateException,
                    "CoupledBoundaryManager::prepareForAssembly: auxiliary slot is not resolved");

        // Real-valued parameter slots (optional).
        std::vector<Real> jit_constants;
        const auto& preg = system_.parameterRegistry();
        const bool have_param_contracts = !preg.specs().empty();
        if (have_param_contracts && preg.slotCount() > 0u) {
            jit_constants = preg.evaluateRealSlots(state);
        }

        ODEIntegrator::advance(reg.integrator,
                               reg.slot,
                               aux_state_,
                               reg.rhs,
                               reg.d_rhs_dX,
                               integrals_.all(),
                               jit_constants,
                               ctx_.t,
                               ctx_.dt);
    }
}

void CoupledBoundaryManager::beginTimeStep()
{
    aux_state_.resetToCommitted();
}

void CoupledBoundaryManager::commitTimeStep()
{
    aux_state_.commitTimeStep();
}

std::vector<Real>
CoupledBoundaryManager::computeAuxiliaryStateForIntegrals(std::span<const Real> integrals_override,
                                                         const SystemStateView& state) const
{
    requireSetup();

    if (aux_registrations_.empty()) {
        return {};
    }

    AuxiliaryState tmp = aux_state_;
    tmp.resetToCommitted();

    // Real-valued parameter slots (optional).
    std::vector<Real> jit_constants;
    const auto& preg = system_.parameterRegistry();
    const bool have_param_contracts = !preg.specs().empty();
    if (have_param_contracts && preg.slotCount() > 0u) {
        jit_constants = preg.evaluateRealSlots(state);
    }

    const Real t = static_cast<Real>(state.time);
    const Real dt = static_cast<Real>(state.dt);

    for (const auto& reg : aux_registrations_) {
        if (!reg.rhs.isValid()) {
            continue;
        }
        if (reg.slot == AuxiliaryStateRegistration::kInvalidSlot) {
            continue;
        }

        ODEIntegrator::advance(reg.integrator,
                               reg.slot,
                               tmp,
                               reg.rhs,
                               reg.d_rhs_dX,
                               integrals_override,
                               jit_constants,
                               t,
                               dt);
    }

    auto vals = tmp.values();
    return std::vector<Real>(vals.begin(), vals.end());
}

std::vector<Real>
CoupledBoundaryManager::computeAuxiliarySensitivityForIntegrals(std::span<const Real> integrals_override,
                                                                const SystemStateView& state) const
{
    requireSetup();

    if (aux_registrations_.empty()) {
        return {};
    }

    const std::size_t n_q = integrals_override.size();
    if (n_q == 0u) {
        return {};
    }

    AuxiliaryState tmp = aux_state_;
    tmp.resetToCommitted();

    const std::size_t n_aux = tmp.size();
    std::vector<Real> sens(n_aux * n_q, 0.0);

    // Real-valued parameter slots (optional).
    std::vector<Real> jit_constants;
    const auto& preg = system_.parameterRegistry();
    const bool have_param_contracts = !preg.specs().empty();
    if (have_param_contracts && preg.slotCount() > 0u) {
        jit_constants = preg.evaluateRealSlots(state);
    }

    const Real t = static_cast<Real>(state.time);
    const Real dt = static_cast<Real>(state.dt);

    if (dt <= Real(0.0)) {
        return sens;
    }

    auto row_span = [&](std::uint32_t slot) -> std::span<Real> {
        if (slot >= n_aux) {
            return {};
        }
        return std::span<Real>(sens.data() + static_cast<std::size_t>(slot) * n_q, n_q);
    };

    forms::DualWorkspace ws;
    std::vector<Real> dx_zero(n_q, 0.0);

    for (const auto& reg : aux_registrations_) {
        if (!reg.rhs.isValid()) {
            continue;
        }
        if (reg.slot == AuxiliaryStateRegistration::kInvalidSlot) {
            continue;
        }

        const std::uint32_t slot = reg.slot;
        if (slot >= n_aux) {
            continue;
        }

        auto vals = tmp.values();
        const Real x_prev = vals[slot];
        const auto dx_prev_span = row_span(slot);
        std::vector<Real> dx_prev(dx_prev_span.begin(), dx_prev_span.end());

        const auto eval_rhs_dual = [&](Real x_value,
                                       std::span<const Real> dx_value,
                                       Real t_eval) -> forms::Dual {
            forms::PointEvalContext pctx;
            pctx.x = {0.0, 0.0, 0.0};
            pctx.time = t_eval;
            pctx.dt = dt;
            pctx.jit_constants = jit_constants;
            pctx.coupled_integrals = integrals_override;
            pctx.coupled_aux = tmp.values();

            forms::PointDualSeedContext seeds;
            seeds.deriv_dim = n_q;
            seeds.aux_dseed = std::span<const Real>(sens.data(), sens.size());
            forms::PointDualSeedContext::AuxOverride ov;
            ov.slot = slot;
            ov.value = x_value;
            ov.deriv = dx_value;
            seeds.aux_override = ov;

            const auto out = forms::evaluateScalarAtDual(reg.rhs, pctx, ws, seeds);
            return out;
        };

        const auto eval_drhs = [&](Real t_eval) -> Real {
            FE_THROW_IF(!reg.d_rhs_dX || !reg.d_rhs_dX->isValid(), InvalidArgumentException,
                        "CoupledBoundaryManager::computeAuxiliarySensitivityForIntegrals: missing d_rhs_dX for implicit ODE method");
            forms::PointEvalContext pctx;
            pctx.x = {0.0, 0.0, 0.0};
            pctx.time = t_eval;
            pctx.dt = dt;
            pctx.jit_constants = jit_constants;
            pctx.coupled_integrals = integrals_override;
            pctx.coupled_aux = tmp.values();
            return forms::evaluateScalarAt(*reg.d_rhs_dX, pctx);
        };

        const auto update_row = [&](std::span<const Real> dx_new) {
            auto row = row_span(slot);
            FE_THROW_IF(row.size() != dx_new.size(), InvalidArgumentException,
                        "CoupledBoundaryManager::computeAuxiliarySensitivityForIntegrals: derivative size mismatch");
            std::copy(dx_new.begin(), dx_new.end(), row.begin());
        };

        switch (reg.integrator) {
            case ODEMethod::ForwardEuler: {
                const auto k1 = eval_rhs_dual(x_prev, dx_prev, t);

                std::vector<Real> dx_new(n_q);
                for (std::size_t j = 0; j < n_q; ++j) {
                    dx_new[j] = dx_prev[j] + dt * k1.deriv[j];
                }
                vals[slot] = x_prev + dt * k1.value;
                update_row(dx_new);
                break;
            }

            case ODEMethod::RK4: {
                const Real half_dt = Real(0.5) * dt;

                const auto k1 = eval_rhs_dual(x_prev, dx_prev, t);
                std::vector<Real> dk1(k1.deriv.begin(), k1.deriv.end());

                const Real x2 = x_prev + half_dt * k1.value;
                std::vector<Real> dx2(n_q);
                for (std::size_t j = 0; j < n_q; ++j) {
                    dx2[j] = dx_prev[j] + half_dt * dk1[j];
                }

                const auto k2 = eval_rhs_dual(x2, dx2, t + half_dt);
                std::vector<Real> dk2(k2.deriv.begin(), k2.deriv.end());

                const Real x3 = x_prev + half_dt * k2.value;
                std::vector<Real> dx3(n_q);
                for (std::size_t j = 0; j < n_q; ++j) {
                    dx3[j] = dx_prev[j] + half_dt * dk2[j];
                }

                const auto k3 = eval_rhs_dual(x3, dx3, t + half_dt);
                std::vector<Real> dk3(k3.deriv.begin(), k3.deriv.end());

                const Real x4 = x_prev + dt * k3.value;
                std::vector<Real> dx4(n_q);
                for (std::size_t j = 0; j < n_q; ++j) {
                    dx4[j] = dx_prev[j] + dt * dk3[j];
                }

                const auto k4 = eval_rhs_dual(x4, dx4, t + dt);
                std::vector<Real> dk4(k4.deriv.begin(), k4.deriv.end());

                const Real x_new =
                    x_prev + (dt / Real(6.0)) * (k1.value + Real(2.0) * k2.value + Real(2.0) * k3.value + k4.value);

                std::vector<Real> dx_new(n_q);
                for (std::size_t j = 0; j < n_q; ++j) {
                    dx_new[j] =
                        dx_prev[j] + (dt / Real(6.0)) * (dk1[j] + Real(2.0) * dk2[j] + Real(2.0) * dk3[j] + dk4[j]);
                }

                vals[slot] = x_new;
                update_row(dx_new);
                break;
            }

            case ODEMethod::BackwardEuler: {
                ODEIntegrator::advance(ODEMethod::BackwardEuler,
                                       slot,
                                       tmp,
                                       reg.rhs,
                                       reg.d_rhs_dX,
                                       integrals_override,
                                       jit_constants,
                                       t,
                                       dt);

                const auto vals_now = tmp.values();
                const Real x_new = vals_now[slot];

                const auto f = eval_rhs_dual(x_new, std::span<const Real>(dx_zero.data(), dx_zero.size()), t);
                const Real dfdx = eval_drhs(t);
                const Real denom = Real(1.0) - dt * dfdx;
                FE_THROW_IF(std::abs(denom) < Real(1e-16), InvalidStateException,
                            "CoupledBoundaryManager::computeAuxiliarySensitivityForIntegrals: BackwardEuler sensitivity denominator near zero");

                std::vector<Real> dx_new(n_q);
                for (std::size_t j = 0; j < n_q; ++j) {
                    dx_new[j] = (dx_prev[j] + dt * f.deriv[j]) / denom;
                }
                update_row(dx_new);
                break;
            }

            case ODEMethod::BDF2: {
                const bool have_hist = tmp.hasHistory(/*steps_back=*/2);

                ODEIntegrator::advance(ODEMethod::BDF2,
                                       slot,
                                       tmp,
                                       reg.rhs,
                                       reg.d_rhs_dX,
                                       integrals_override,
                                       jit_constants,
                                       t,
                                       dt);

                if (!have_hist) {
                    const auto vals_now = tmp.values();
                    const Real x_new = vals_now[slot];
                    const auto f = eval_rhs_dual(x_new, std::span<const Real>(dx_zero.data(), dx_zero.size()), t);
                    const Real dfdx = eval_drhs(t);
                    const Real denom = Real(1.0) - dt * dfdx;
                    FE_THROW_IF(std::abs(denom) < Real(1e-16), InvalidStateException,
                                "CoupledBoundaryManager::computeAuxiliarySensitivityForIntegrals: BE sensitivity denominator near zero (BDF2 fallback)");
                    std::vector<Real> dx_new(n_q);
                    for (std::size_t j = 0; j < n_q; ++j) {
                        dx_new[j] = (dx_prev[j] + dt * f.deriv[j]) / denom;
                    }
                    update_row(dx_new);
                    break;
                }

                const auto vals_now = tmp.values();
                const Real x_new = vals_now[slot];
                const auto f = eval_rhs_dual(x_new, std::span<const Real>(dx_zero.data(), dx_zero.size()), t);
                const Real dfdx = eval_drhs(t);
                const Real denom = Real(3.0) - Real(2.0) * dt * dfdx;
                FE_THROW_IF(std::abs(denom) < Real(1e-16), InvalidStateException,
                            "CoupledBoundaryManager::computeAuxiliarySensitivityForIntegrals: BDF2 sensitivity denominator near zero");

                std::vector<Real> dx_new(n_q);
                for (std::size_t j = 0; j < n_q; ++j) {
                    dx_new[j] = (Real(4.0) * dx_prev[j] + Real(2.0) * dt * f.deriv[j]) / denom;
                }
                update_row(dx_new);
                break;
            }
        }
    }

    return sens;
}

std::vector<CoupledBoundaryManager::RegisteredBoundaryFunctional>
CoupledBoundaryManager::registeredBoundaryFunctionals() const
{
    std::vector<RegisteredBoundaryFunctional> out;
    out.reserve(functionals_.size());
    for (const auto& entry : functionals_) {
        const auto idx = integrals_.indexOf(entry.def.name);
        FE_THROW_IF(idx > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()),
                    InvalidArgumentException,
                    "CoupledBoundaryManager::registeredBoundaryFunctionals: slot overflow");
        out.push_back(RegisteredBoundaryFunctional{entry.def, static_cast<std::uint32_t>(idx)});
    }
    return out;
}

} // namespace systems
} // namespace FE
} // namespace svmp
