/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/SystemAssembly.h"

#include "Systems/FESystem.h"
#include "Systems/CoupledBoundaryManager.h"
#include "Systems/GlobalKernel.h"
#include "Systems/SystemsExceptions.h"

#include "Assembly/GlobalSystemView.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/TimeIntegrationContext.h"

#include "Backends/Interfaces/GenericVector.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace systems {

namespace {

void mergeAssemblyResult(assembly::AssemblyResult& total, const assembly::AssemblyResult& part)
{
    if (!part.success && total.success) {
        total.success = false;
        total.error_message = part.error_message;
    } else if (!part.success && !part.error_message.empty()) {
        if (!total.error_message.empty()) {
            total.error_message += "\n";
        }
        total.error_message += part.error_message;
    }

    total.elements_assembled += part.elements_assembled;
    total.boundary_faces_assembled += part.boundary_faces_assembled;
    total.interior_faces_assembled += part.interior_faces_assembled;
    total.elapsed_time_seconds += part.elapsed_time_seconds;
    total.matrix_entries_inserted += part.matrix_entries_inserted;
    total.vector_entries_inserted += part.vector_entries_inserted;
}

} // namespace

assembly::AssemblyResult assembleOperator(
    FESystem& system,
    const AssemblyRequest& request,
    const SystemStateView& state,
    assembly::GlobalSystemView* matrix_out,
    assembly::GlobalSystemView* vector_out)
{
    system.requireSetup();

    FE_THROW_IF(request.op.empty(), InvalidArgumentException, "assembleOperator: empty operator tag");
    FE_THROW_IF(!system.operator_registry_.has(request.op), InvalidArgumentException,
                "assembleOperator: unknown operator '" + request.op + "'");

    FE_THROW_IF(request.want_matrix && matrix_out == nullptr, InvalidArgumentException,
                "assembleOperator: want_matrix but matrix_out is null");
    FE_THROW_IF(request.want_vector && vector_out == nullptr, InvalidArgumentException,
                "assembleOperator: want_vector but vector_out is null");
    FE_THROW_IF(!request.want_matrix && !request.want_vector, InvalidArgumentException,
                "assembleOperator: nothing requested (want_matrix=false and want_vector=false)");

    FE_CHECK_NOT_NULL(system.assembler_.get(), "FESystem::assembler");

    // Coupled boundary-condition orchestration (boundary integrals + auxiliary state).
    // Per design, this runs immediately before PDE assembly so coupled coefficients
    // can read a consistent, up-to-date context during kernel evaluation.
    if (system.coupled_boundary_) {
        system.coupled_boundary_->prepareForAssembly(state);
    }

    if (request.zero_outputs) {
        if (request.want_matrix) {
            matrix_out->zero();
        }
        if (request.want_vector && vector_out != matrix_out) {
            vector_out->zero();
        } else if (request.want_vector && vector_out == matrix_out) {
            vector_out->zero();
        }
    }

    if (request.want_matrix) {
        auto it = system.sparsity_by_op_.find(request.op);
        if (it != system.sparsity_by_op_.end() && it->second) {
            system.assembler_->setSparsityPattern(it->second.get());
        }
    } else {
        system.assembler_->setSparsityPattern(nullptr);
    }

    auto& assembler = *system.assembler_;
    assembler.setCurrentSolution(state.u);
    std::unique_ptr<assembly::GlobalSystemView> current_solution_view;
    if (state.u_vector != nullptr) {
        // `createAssemblyView()` is non-const for historical reasons; treat this as read-only use.
        auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
        current_solution_view = vec->createAssemblyView();
    }
    assembler.setCurrentSolutionView(current_solution_view.get());
    {
        std::vector<assembly::FieldSolutionAccess> access;
        access.reserve(system.field_registry_.size());
        for (const auto& rec : system.field_registry_.records()) {
            FE_CHECK_NOT_NULL(rec.space.get(), "assembleOperator: field space");
            const auto idx = static_cast<std::size_t>(rec.id);
            FE_THROW_IF(idx >= system.field_dof_handlers_.size(), InvalidStateException,
                        "assembleOperator: invalid field DOF handler index for field '" + rec.name + "'");
            FE_THROW_IF(idx >= system.field_dof_offsets_.size(), InvalidStateException,
                        "assembleOperator: invalid field DOF offset index for field '" + rec.name + "'");
            access.push_back(assembly::FieldSolutionAccess{
                rec.id,
                rec.space.get(),
                &system.field_dof_handlers_[idx].getDofMap(),
                system.field_dof_offsets_[idx],
            });
        }
        assembler.setFieldSolutionAccess(access);
    }
    assembler.setTimeIntegrationContext(state.time_integration);

    int required_history = 0;
    if (state.time_integration != nullptr) {
        if (state.time_integration->dt1) {
            required_history = std::max(required_history, state.time_integration->dt1->requiredHistoryStates());
        }
        if (state.time_integration->dt2) {
            required_history = std::max(required_history, state.time_integration->dt2->requiredHistoryStates());
        }
    }

    if (required_history > 0) {
        if (!state.u_history.empty()) {
            FE_THROW_IF(static_cast<int>(state.u_history.size()) < required_history, InvalidArgumentException,
                        "assembleOperator: insufficient solution history (need " + std::to_string(required_history) +
                            ", have " + std::to_string(state.u_history.size()) + ")");
            for (int k = 1; k <= required_history; ++k) {
                assembler.setPreviousSolutionK(k, state.u_history[static_cast<std::size_t>(k - 1)]);
            }
        } else {
            // Backward-compatible path (supports up to 2 history states).
            FE_THROW_IF(required_history > 2, InvalidArgumentException,
                        "assembleOperator: time integration requires more than 2 history states, but state.u_history was not provided");
            assembler.setPreviousSolution(state.u_prev);
            assembler.setPreviousSolution2(state.u_prev2);
        }
    } else {
        // No dt(...) required by the active time-integration context.
        assembler.setPreviousSolution({});
        assembler.setPreviousSolution2({});
    }

    // Optional parameter validation + defaults.
    system.parameter_registry_.validate(state);
    std::function<std::optional<Real>(std::string_view)> get_real_param_wrapped{};
    std::function<std::optional<params::Value>(std::string_view)> get_param_wrapped{};

    const bool have_param_contracts = !system.parameter_registry_.specs().empty();
    if (have_param_contracts) {
        get_real_param_wrapped = system.parameter_registry_.makeRealGetter(state);
        get_param_wrapped = system.parameter_registry_.makeParamGetter(state);
    }

    assembler.setTime(static_cast<Real>(state.time));
    assembler.setTimeStep(static_cast<Real>(state.dt));
    assembler.setRealParameterGetter(have_param_contracts
                                         ? &get_real_param_wrapped
                                         : (state.getRealParam ? &state.getRealParam : nullptr));
    assembler.setParameterGetter(have_param_contracts
                                     ? &get_param_wrapped
                                     : (state.getParam ? &state.getParam : nullptr));
    assembler.setUserData(state.user_data);

    // JIT-friendly constant slots (Real-valued parameters resolved to stable indices).
    std::vector<Real> jit_constants;
    if (have_param_contracts && system.parameter_registry_.slotCount() > 0u) {
        jit_constants = system.parameter_registry_.evaluateRealSlots(state);
        assembler.setJITConstants(jit_constants);
    } else {
        assembler.setJITConstants({});
    }

    if (system.coupled_boundary_) {
        assembler.setCoupledValues(system.coupled_boundary_->integrals().all(),
                                   system.coupled_boundary_->auxiliaryState().values());
    } else {
        assembler.setCoupledValues({}, {});
    }
    const auto& mesh = system.meshAccess();

    const auto& def = system.operator_registry_.get(request.op);

    assembly::AssemblyResult total;

    // Cell terms
    for (const auto& term : def.cells) {
        FE_CHECK_NOT_NULL(term.kernel.get(), "assembleOperator: cell term kernel");
        const auto& test_field = system.field_registry_.get(term.test_field);
        const auto& trial_field = system.field_registry_.get(term.trial_field);

        FE_CHECK_NOT_NULL(test_field.space.get(), "assembleOperator: test_field.space");
        FE_CHECK_NOT_NULL(trial_field.space.get(), "assembleOperator: trial_field.space");

        const auto test_idx = static_cast<std::size_t>(test_field.id);
        const auto trial_idx = static_cast<std::size_t>(trial_field.id);
        FE_THROW_IF(test_field.id < 0 || test_idx >= system.field_dof_handlers_.size(), InvalidStateException,
                    "assembleOperator: invalid test field DOF handler");
        FE_THROW_IF(trial_field.id < 0 || trial_idx >= system.field_dof_handlers_.size(), InvalidStateException,
                    "assembleOperator: invalid trial field DOF handler");

        assembler.setRowDofMap(system.field_dof_handlers_[test_idx].getDofMap(),
                               system.field_dof_offsets_[test_idx]);
	        assembler.setColDofMap(system.field_dof_handlers_[trial_idx].getDofMap(),
	                               system.field_dof_offsets_[trial_idx]);

	        const bool want_matrix = request.want_matrix && !term.kernel->isVectorOnly();
	        const bool want_vector = request.want_vector && !term.kernel->isMatrixOnly();

	        if (want_matrix && want_vector) {
	            auto r = assembler.assembleBoth(mesh, *test_field.space, *trial_field.space, *term.kernel,
	                                            *matrix_out, *vector_out);
	            mergeAssemblyResult(total, r);
	        } else if (want_matrix) {
	            auto r = assembler.assembleMatrix(mesh, *test_field.space, *trial_field.space, *term.kernel,
	                                              *matrix_out);
	            mergeAssemblyResult(total, r);
	        } else if (want_vector) {
	            if (term.test_field == term.trial_field) {
	                auto r = assembler.assembleVector(mesh, *test_field.space, *term.kernel, *vector_out);
	                mergeAssemblyResult(total, r);
	            } else {
	                assembly::DenseVectorView dummy_matrix(0);
	                auto r = assembler.assembleBoth(mesh, *test_field.space, *trial_field.space, *term.kernel,
	                                                dummy_matrix, *vector_out);
	                mergeAssemblyResult(total, r);
	            }
	        }
	    }

    // Boundary terms
    if (request.assemble_boundary_terms) {
        for (const auto& term : def.boundary) {
            FE_CHECK_NOT_NULL(term.kernel.get(), "assembleOperator: boundary term kernel");
            const auto& test_field = system.field_registry_.get(term.test_field);
            const auto& trial_field = system.field_registry_.get(term.trial_field);

            FE_CHECK_NOT_NULL(test_field.space.get(), "assembleOperator: test_field.space");
            FE_CHECK_NOT_NULL(trial_field.space.get(), "assembleOperator: trial_field.space");

            const auto test_idx = static_cast<std::size_t>(test_field.id);
            const auto trial_idx = static_cast<std::size_t>(trial_field.id);
            FE_THROW_IF(test_field.id < 0 || test_idx >= system.field_dof_handlers_.size(), InvalidStateException,
                        "assembleOperator: invalid test field DOF handler");
            FE_THROW_IF(trial_field.id < 0 || trial_idx >= system.field_dof_handlers_.size(), InvalidStateException,
                        "assembleOperator: invalid trial field DOF handler");

            assembler.setRowDofMap(system.field_dof_handlers_[test_idx].getDofMap(),
                                   system.field_dof_offsets_[test_idx]);
            assembler.setColDofMap(system.field_dof_handlers_[trial_idx].getDofMap(),
                                   system.field_dof_offsets_[trial_idx]);

            auto r = assembler.assembleBoundaryFaces(
                mesh, term.marker, *test_field.space, *trial_field.space, *term.kernel,
                request.want_matrix ? matrix_out : nullptr,
                request.want_vector ? vector_out : nullptr);
            mergeAssemblyResult(total, r);
        }
    }

	    // Interior face terms (DG)
	    if (request.assemble_interior_face_terms) {
	        for (const auto& term : def.interior) {
	            FE_CHECK_NOT_NULL(term.kernel.get(), "assembleOperator: interior-face term kernel");
	            const auto& test_field = system.field_registry_.get(term.test_field);
	            const auto& trial_field = system.field_registry_.get(term.trial_field);

            FE_CHECK_NOT_NULL(test_field.space.get(), "assembleOperator: test_field.space");
            FE_CHECK_NOT_NULL(trial_field.space.get(), "assembleOperator: trial_field.space");

            const auto test_idx = static_cast<std::size_t>(test_field.id);
            const auto trial_idx = static_cast<std::size_t>(trial_field.id);
            FE_THROW_IF(test_field.id < 0 || test_idx >= system.field_dof_handlers_.size(), InvalidStateException,
                        "assembleOperator: invalid test field DOF handler");
            FE_THROW_IF(trial_field.id < 0 || trial_idx >= system.field_dof_handlers_.size(), InvalidStateException,
                        "assembleOperator: invalid trial field DOF handler");

	            assembler.setRowDofMap(system.field_dof_handlers_[test_idx].getDofMap(),
	                                   system.field_dof_offsets_[test_idx]);
	            assembler.setColDofMap(system.field_dof_handlers_[trial_idx].getDofMap(),
	                                   system.field_dof_offsets_[trial_idx]);

	            const bool want_matrix = request.want_matrix && !term.kernel->isVectorOnly();
	            const bool want_vector = request.want_vector && !term.kernel->isMatrixOnly();
	            if (!want_matrix && !want_vector) {
	                continue;
	            }

	            if (want_matrix) {
	                auto r = assembler.assembleInteriorFaces(
	                    mesh, *test_field.space, *trial_field.space, *term.kernel, *matrix_out,
	                    want_vector ? vector_out : nullptr);
	                mergeAssemblyResult(total, r);
	            } else {
	                assembly::DenseVectorView dummy_matrix(0);
	                auto r = assembler.assembleInteriorFaces(
	                    mesh, *test_field.space, *trial_field.space, *term.kernel, dummy_matrix,
	                    vector_out);
	                mergeAssemblyResult(total, r);
	            }
	        }
	    }

    // Global (non-element-local) terms (e.g., contact)
    if (request.assemble_global_terms) {
        for (const auto& kernel : def.global) {
            FE_CHECK_NOT_NULL(kernel.get(), "assembleOperator: global term kernel");
            auto r = kernel->assemble(system, request, state,
                                      request.want_matrix ? matrix_out : nullptr,
                                      request.want_vector ? vector_out : nullptr);
            mergeAssemblyResult(total, r);
        }
    }

    assembler.finalize(request.want_matrix ? matrix_out : nullptr,
                       request.want_vector ? vector_out : nullptr);

    return total;
}

} // namespace systems
} // namespace FE
} // namespace svmp
