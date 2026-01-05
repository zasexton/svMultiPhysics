/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/TransientSystem.h"

#include "Core/FEException.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace systems {

TransientSystem::TransientSystem(FESystem& system, std::shared_ptr<const TimeIntegrator> integrator)
    : system_(system)
    , integrator_(std::move(integrator))
{
    FE_CHECK_NOT_NULL(integrator_.get(), "TransientSystem::integrator");
}

const TimeIntegrator& TransientSystem::integrator() const
{
    FE_CHECK_NOT_NULL(integrator_.get(), "TransientSystem::integrator");
    return *integrator_;
}

void TransientSystem::validateState(const SystemStateView& state, int required_history_states) const
{
    FE_THROW_IF(!system_.isSetup(), InvalidStateException, "TransientSystem: system.setup() has not been called");

    const auto n_dofs = static_cast<std::size_t>(system_.dofHandler().getNumDofs());
    FE_THROW_IF(state.u.size() != n_dofs, InvalidArgumentException,
                "TransientSystem: state.u size does not match system DOF count");

    if (required_history_states <= 0) {
        return;
    }

    if (!state.u_history.empty()) {
        FE_THROW_IF(static_cast<int>(state.u_history.size()) < required_history_states, InvalidArgumentException,
                    "TransientSystem: insufficient history states for dt(...) stencils");
        for (int k = 1; k <= required_history_states; ++k) {
            FE_THROW_IF(state.u_history[static_cast<std::size_t>(k - 1)].size() != n_dofs, InvalidArgumentException,
                        "TransientSystem: history state size does not match system DOF count");
        }
        return;
    }

    // Backward-compatible path (supports up to 2 history states).
    if (required_history_states >= 1) {
        FE_THROW_IF(state.u_prev.size() != n_dofs, InvalidArgumentException,
                    "TransientSystem: dt(...) requires state.u_prev with matching DOF count");
    }
    if (required_history_states >= 2) {
        FE_THROW_IF(state.u_prev2.size() != n_dofs, InvalidArgumentException,
                    "TransientSystem: time integration requires state.u_prev2 with matching DOF count");
    }
    FE_THROW_IF(required_history_states > 2, InvalidArgumentException,
                "TransientSystem: time integration requires more than 2 history states, but state.u_history was not provided");
}

assembly::AssemblyResult TransientSystem::assemble(
    const AssemblyRequest& req,
    const SystemStateView& state,
    assembly::GlobalSystemView* matrix_out,
    assembly::GlobalSystemView* vector_out)
{
    const int max_order = system_.temporalOrder();
    if (max_order <= 0) {
        // No dt(...) present anywhere; assemble as steady.
        return system_.assemble(req, state, matrix_out, vector_out);
    }

    FE_THROW_IF(max_order > integrator().maxSupportedDerivativeOrder(),
                NotImplementedException,
                "TransientSystem: integrator '" + integrator().name() + "' does not support dt(·," + std::to_string(max_order) + ")");

    ctx_ = integrator().buildContext(max_order, state);

    int required_history = 0;
    if (ctx_.dt1) {
        required_history = std::max(required_history, ctx_.dt1->requiredHistoryStates());
    }
    if (ctx_.dt2) {
        required_history = std::max(required_history, ctx_.dt2->requiredHistoryStates());
    }
    validateState(state, required_history);

    SystemStateView transient_state = state;
    transient_state.time_integration = &ctx_;

    return system_.assemble(req, transient_state, matrix_out, vector_out);
}

} // namespace systems
} // namespace FE
} // namespace svmp
