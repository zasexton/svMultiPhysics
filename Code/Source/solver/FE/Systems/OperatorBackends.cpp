/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/OperatorBackends.h"

#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"

#include "Assembly/MatrixFreeAssembler.h"
#include "Assembly/FunctionalAssembler.h"

#include <functional>
#include <unordered_map>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

struct OperatorBackends::Impl {
    struct MatrixFreeEntry {
        std::shared_ptr<assembly::IMatrixFreeKernel> kernel;
        assembly::MatrixFreeOptions options{};
        bool has_custom_options{false};

        std::unique_ptr<assembly::MatrixFreeAssembler> assembler;
        std::shared_ptr<assembly::MatrixFreeOperator> op;
    };

    std::unordered_map<OperatorTag, MatrixFreeEntry> matrix_free;
    std::unordered_map<std::string, std::shared_ptr<assembly::FunctionalKernel>> functionals;
};

OperatorBackends::OperatorBackends()
    : impl_(std::make_unique<Impl>())
{}

OperatorBackends::~OperatorBackends() = default;

OperatorBackends::OperatorBackends(OperatorBackends&&) noexcept = default;
OperatorBackends& OperatorBackends::operator=(OperatorBackends&&) noexcept = default;

void OperatorBackends::clear()
{
    impl_->matrix_free.clear();
    impl_->functionals.clear();
}

void OperatorBackends::invalidateCache()
{
    for (auto& kv : impl_->matrix_free) {
        kv.second.op.reset();
        kv.second.assembler.reset();
    }
}

void OperatorBackends::registerMatrixFree(OperatorTag tag,
                                          std::shared_ptr<assembly::IMatrixFreeKernel> kernel)
{
    FE_THROW_IF(tag.empty(), InvalidArgumentException, "OperatorBackends::registerMatrixFree: empty tag");
    FE_CHECK_NOT_NULL(kernel.get(), "OperatorBackends::registerMatrixFree: kernel");

    auto& entry = impl_->matrix_free[tag];
    entry.kernel = std::move(kernel);
    entry.has_custom_options = false;
    entry.op.reset();
    entry.assembler.reset();
}

void OperatorBackends::registerMatrixFree(OperatorTag tag,
                                          std::shared_ptr<assembly::IMatrixFreeKernel> kernel,
                                          const assembly::MatrixFreeOptions& options)
{
    FE_THROW_IF(tag.empty(), InvalidArgumentException, "OperatorBackends::registerMatrixFree: empty tag");
    FE_CHECK_NOT_NULL(kernel.get(), "OperatorBackends::registerMatrixFree: kernel");

    auto& entry = impl_->matrix_free[tag];
    entry.kernel = std::move(kernel);
    entry.options = options;
    entry.has_custom_options = true;
    entry.op.reset();
    entry.assembler.reset();
}

bool OperatorBackends::hasMatrixFree(const OperatorTag& tag) const noexcept
{
    if (!impl_) {
        return false;
    }
    return impl_->matrix_free.find(tag) != impl_->matrix_free.end();
}

std::shared_ptr<assembly::MatrixFreeOperator>
OperatorBackends::matrixFreeOperator(const FESystem& system, const OperatorTag& tag) const
{
    FE_THROW_IF(!system.is_setup_, InvalidStateException,
                "OperatorBackends::matrixFreeOperator: system is not set up");
    FE_THROW_IF(system.field_registry_.size() != 1u, NotImplementedException,
                "OperatorBackends::matrixFreeOperator: multi-field not implemented");

    auto it = impl_->matrix_free.find(tag);
    FE_THROW_IF(it == impl_->matrix_free.end(), InvalidArgumentException,
                "OperatorBackends::matrixFreeOperator: unknown matrix-free operator '" + tag + "'");

    auto& entry = it->second;
    FE_CHECK_NOT_NULL(entry.kernel.get(), "OperatorBackends::matrixFreeOperator: kernel");

    if (entry.op && entry.assembler) {
        return entry.op;
    }

    const auto& space = *system.field_registry_.records().front().space;
    const auto& dof_map = system.dof_handler_.getDofMap();
    const auto& mesh = system.meshAccess();

    if (entry.has_custom_options) {
        entry.assembler = assembly::createMatrixFreeAssembler(entry.options);
    } else {
        entry.assembler = assembly::createMatrixFreeAssembler();
    }

    FE_CHECK_NOT_NULL(entry.assembler.get(), "OperatorBackends::matrixFreeOperator: assembler");

    entry.assembler->setMesh(mesh);
    entry.assembler->setDofMap(dof_map);
    entry.assembler->setSpace(space);
    entry.assembler->setKernel(*entry.kernel);
    entry.assembler->setConstraints(system.affine_constraints_);
    entry.assembler->setup();

    entry.op = entry.assembler->getOperator();
    FE_CHECK_NOT_NULL(entry.op.get(), "OperatorBackends::matrixFreeOperator: operator");
    return entry.op;
}

void OperatorBackends::registerFunctional(std::string tag,
                                          std::shared_ptr<assembly::FunctionalKernel> kernel)
{
    FE_THROW_IF(tag.empty(), InvalidArgumentException, "OperatorBackends::registerFunctional: empty tag");
    FE_CHECK_NOT_NULL(kernel.get(), "OperatorBackends::registerFunctional: kernel");
    impl_->functionals[std::move(tag)] = std::move(kernel);
}

Real OperatorBackends::evaluateFunctional(const FESystem& system,
                                          const std::string& tag,
                                          const SystemStateView& state) const
{
    FE_THROW_IF(!system.is_setup_, InvalidStateException,
                "OperatorBackends::evaluateFunctional: system is not set up");
    FE_THROW_IF(system.field_registry_.size() != 1u, NotImplementedException,
                "OperatorBackends::evaluateFunctional: multi-field not implemented");

    auto it = impl_->functionals.find(tag);
    FE_THROW_IF(it == impl_->functionals.end(), InvalidArgumentException,
                "OperatorBackends::evaluateFunctional: unknown functional '" + tag + "'");

    auto& kernel = *it->second;
    const auto& field = system.field_registry_.records().front();
    const auto& space = *field.space;

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(system.meshAccess());
    assembler.setDofMap(system.dof_handler_.getDofMap());
    assembler.setSpace(space);
    assembler.setPrimaryField(field.id);
    assembler.setSolution(state.u);
    assembler.setTimeIntegrationContext(state.time_integration);
    assembler.setTime(static_cast<Real>(state.time));
    assembler.setTimeStep(static_cast<Real>(state.dt));
    const auto& preg = system.parameterRegistry();
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
    std::vector<Real> jit_constants;
    if (have_param_contracts && preg.slotCount() > 0u) {
        jit_constants = preg.evaluateRealSlots(state);
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
    return assembler.assembleScalar(kernel);
}

Real OperatorBackends::evaluateBoundaryFunctional(const FESystem& system,
                                                  const std::string& tag,
                                                  int boundary_marker,
                                                  const SystemStateView& state) const
{
    FE_THROW_IF(!system.is_setup_, InvalidStateException,
                "OperatorBackends::evaluateBoundaryFunctional: system is not set up");
    FE_THROW_IF(system.field_registry_.size() != 1u, NotImplementedException,
                "OperatorBackends::evaluateBoundaryFunctional: multi-field not implemented");

    auto it = impl_->functionals.find(tag);
    FE_THROW_IF(it == impl_->functionals.end(), InvalidArgumentException,
                "OperatorBackends::evaluateBoundaryFunctional: unknown functional '" + tag + "'");

    auto& kernel = *it->second;
    const auto& field = system.field_registry_.records().front();
    const auto& space = *field.space;

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(system.meshAccess());
    assembler.setDofMap(system.dof_handler_.getDofMap());
    assembler.setSpace(space);
    assembler.setPrimaryField(field.id);
    assembler.setSolution(state.u);
    assembler.setTimeIntegrationContext(state.time_integration);
    assembler.setTime(static_cast<Real>(state.time));
    assembler.setTimeStep(static_cast<Real>(state.dt));
    const auto& preg = system.parameterRegistry();
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
    std::vector<Real> jit_constants;
    if (have_param_contracts && preg.slotCount() > 0u) {
        jit_constants = preg.evaluateRealSlots(state);
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
    return assembler.assembleBoundaryScalar(kernel, boundary_marker);
}

} // namespace systems
} // namespace FE
} // namespace svmp
