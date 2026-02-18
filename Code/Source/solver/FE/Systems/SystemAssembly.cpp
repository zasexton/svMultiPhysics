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

#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"

#include "Backends/Interfaces/GenericVector.h"
#include "Backends/FSILS/FsilsMatrix.h"
#include "Core/Logger.h"
#include "Core/Alignment.h"
#include "Core/AlignedAllocator.h"
#include "Core/FEConfig.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace systems {

namespace {

[[nodiscard]] bool oopTraceEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_OOP_SOLVER_TRACE");
        if (env == nullptr) {
            return false;
        }
        std::string v(env);
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return !(v == "0" || v == "false" || v == "off" || v == "no");
    }();
    return enabled;
}

void traceLog(const std::string& msg)
{
    if (!oopTraceEnabled()) {
        return;
    }
    FE_LOG_INFO(msg);
}

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
    total.interface_faces_assembled += part.interface_faces_assembled;
    total.elapsed_time_seconds += part.elapsed_time_seconds;
    total.matrix_entries_inserted += part.matrix_entries_inserted;
    total.vector_entries_inserted += part.vector_entries_inserted;
}

#if FE_HAS_MPI
MPI_Datatype mpiRealType()
{
    if (sizeof(Real) == sizeof(double)) {
        return MPI_DOUBLE;
    }
    if (sizeof(Real) == sizeof(float)) {
        return MPI_FLOAT;
    }
    return MPI_LONG_DOUBLE;
}

MPI_Datatype mpiGlobalIndexType()
{
    if (sizeof(GlobalIndex) == sizeof(std::int64_t)) {
        return MPI_INT64_T;
    }
    if (sizeof(GlobalIndex) == sizeof(long long)) {
        return MPI_LONG_LONG;
    }
    if (sizeof(GlobalIndex) == sizeof(long)) {
        return MPI_LONG;
    }
    return MPI_LONG_LONG;
}

std::vector<std::pair<GlobalIndex, Real>> allreduceSumSparsePairs(std::vector<std::pair<GlobalIndex, Real>> local,
                                                                  MPI_Comm comm)
{
    int comm_size = 1;
    MPI_Comm_size(comm, &comm_size);
    if (comm_size <= 1) {
        return local;
    }

    const int local_n = static_cast<int>(local.size());
    std::vector<int> counts(static_cast<std::size_t>(comm_size), 0);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

    std::vector<int> displs(static_cast<std::size_t>(comm_size), 0);
    int total_n = 0;
    for (int r = 0; r < comm_size; ++r) {
        displs[static_cast<std::size_t>(r)] = total_n;
        total_n += counts[static_cast<std::size_t>(r)];
    }

    std::vector<GlobalIndex> idx_local(static_cast<std::size_t>(local_n), GlobalIndex(0));
    std::vector<Real> val_local(static_cast<std::size_t>(local_n), Real(0.0));
    for (int i = 0; i < local_n; ++i) {
        idx_local[static_cast<std::size_t>(i)] = local[static_cast<std::size_t>(i)].first;
        val_local[static_cast<std::size_t>(i)] = local[static_cast<std::size_t>(i)].second;
    }

    std::vector<GlobalIndex> idx_all(static_cast<std::size_t>(total_n), GlobalIndex(0));
    std::vector<Real> val_all(static_cast<std::size_t>(total_n), Real(0.0));
    MPI_Allgatherv(idx_local.data(),
                   local_n,
                   mpiGlobalIndexType(),
                   idx_all.data(),
                   counts.data(),
                   displs.data(),
                   mpiGlobalIndexType(),
                   comm);
    MPI_Allgatherv(val_local.data(),
                   local_n,
                   mpiRealType(),
                   val_all.data(),
                   counts.data(),
                   displs.data(),
                   mpiRealType(),
                   comm);

    std::vector<std::pair<GlobalIndex, Real>> merged;
    merged.reserve(static_cast<std::size_t>(total_n));
    for (int i = 0; i < total_n; ++i) {
        merged.emplace_back(idx_all[static_cast<std::size_t>(i)], val_all[static_cast<std::size_t>(i)]);
    }

    std::sort(merged.begin(), merged.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::pair<GlobalIndex, Real>> out;
    out.reserve(merged.size());
    for (const auto& kv : merged) {
        if (out.empty() || kv.first != out.back().first) {
            out.push_back(kv);
        } else {
            out.back().second += kv.second;
        }
    }
    return out;
}
#endif

class SparseVectorAccumulatorView final : public assembly::GlobalSystemView {
public:
    explicit SparseVectorAccumulatorView(GlobalIndex size)
        : size_(size)
    {
    }

    // Matrix operations: not supported for this view.
    void addMatrixEntries(std::span<const GlobalIndex> /*dofs*/,
                          std::span<const Real> /*local_matrix*/,
                          assembly::AddMode /*mode*/ = assembly::AddMode::Add) override
    {
    }

    void addMatrixEntries(std::span<const GlobalIndex> /*row_dofs*/,
                          std::span<const GlobalIndex> /*col_dofs*/,
                          std::span<const Real> /*local_matrix*/,
                          assembly::AddMode /*mode*/ = assembly::AddMode::Add) override
    {
    }

    void addMatrixEntry(GlobalIndex /*row*/,
                        GlobalIndex /*col*/,
                        Real /*value*/,
                        assembly::AddMode /*mode*/ = assembly::AddMode::Add) override
    {
    }

    void setDiagonal(std::span<const GlobalIndex> /*dofs*/,
                     std::span<const Real> /*values*/) override
    {
    }

    void setDiagonal(GlobalIndex /*dof*/, Real /*value*/) override
    {
    }

    void zeroRows(std::span<const GlobalIndex> /*rows*/,
                  bool /*set_diagonal*/ = true) override
    {
    }

    // Vector operations
    void addVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_vector,
                          assembly::AddMode mode = assembly::AddMode::Add) override
    {
        FE_THROW_IF(dofs.size() != local_vector.size(), InvalidArgumentException,
                    "SparseVectorAccumulatorView::addVectorEntries: size mismatch");
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            addVectorEntry(dofs[i], local_vector[i], mode);
        }
    }

    void addVectorEntry(GlobalIndex dof,
                        Real value,
                        assembly::AddMode mode = assembly::AddMode::Add) override
    {
        if (dof < 0 || dof >= size_) {
            return;
        }
        switch (mode) {
            case assembly::AddMode::Add:
                values_[dof] += value;
                break;
            case assembly::AddMode::Insert:
                values_[dof] = value;
                break;
            case assembly::AddMode::Max: {
                auto& v = values_[dof];
                v = std::max(v, value);
                break;
            }
            case assembly::AddMode::Min: {
                auto& v = values_[dof];
                v = std::min(v, value);
                break;
            }
        }
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> values) override
    {
        FE_THROW_IF(dofs.size() != values.size(), InvalidArgumentException,
                    "SparseVectorAccumulatorView::setVectorEntries: size mismatch");
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            const auto dof = dofs[i];
            if (dof < 0 || dof >= size_) {
                continue;
            }
            values_[dof] = values[i];
        }
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
    {
        for (const auto dof : dofs) {
            if (dof < 0 || dof >= size_) continue;
            values_.erase(dof);
        }
    }

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override
    {
        if (dof < 0 || dof >= size_) {
            return 0.0;
        }
        auto it = values_.find(dof);
        if (it == values_.end()) {
            return 0.0;
        }
        return it->second;
    }

    void beginAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Building; }
    void endAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Flushing; }
    void finalizeAssembly() override { phase_ = assembly::AssemblyPhase::Finalized; }
    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return phase_; }

    [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
    [[nodiscard]] bool hasVector() const noexcept override { return true; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return size_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return 1; }
    [[nodiscard]] std::string backendName() const override { return "SparseVectorAccumulator"; }

    void zero() override { values_.clear(); }

    [[nodiscard]] const std::unordered_map<GlobalIndex, Real>& values() const noexcept { return values_; }

    [[nodiscard]] std::vector<std::pair<GlobalIndex, Real>> entriesSorted(Real abs_tol = 0.0) const
    {
        std::vector<std::pair<GlobalIndex, Real>> out;
        out.reserve(values_.size());
        for (const auto& kv : values_) {
            if (std::abs(kv.second) <= abs_tol) continue;
            out.emplace_back(kv.first, kv.second);
        }
        std::sort(out.begin(), out.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        return out;
    }

private:
    GlobalIndex size_{0};
    std::unordered_map<GlobalIndex, Real> values_{};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

} // namespace

assembly::AssemblyResult assembleOperator(
    FESystem& system,
    const AssemblyRequest& request,
    const SystemStateView& state,
    assembly::GlobalSystemView* matrix_out,
    assembly::GlobalSystemView* vector_out)
{
    system.requireSetup();
    system.last_rank_one_updates_.clear();

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
        if (oopTraceEnabled()) {
            traceLog("assembleOperator: coupled_boundary.prepareForAssembly() begin");
        }
        const auto t0 = std::chrono::steady_clock::now();
        system.coupled_boundary_->prepareForAssembly(state);
        const auto t1 = std::chrono::steady_clock::now();
        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "assembleOperator: coupled_boundary.prepareForAssembly() done time="
                << std::chrono::duration<double>(t1 - t0).count();
            traceLog(oss.str());
        }
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
        for (const auto& s : state.time_integration->dt_extra) {
            if (s) {
                required_history = std::max(required_history, s->requiredHistoryStates());
            }
        }
    }

    std::unique_ptr<assembly::GlobalSystemView> prev_solution_view;
    std::unique_ptr<assembly::GlobalSystemView> prev2_solution_view;

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

        // Provide global-indexed views for history states when available (needed by some backends).
        if (required_history >= 1) {
            if (state.u_prev_vector != nullptr) {
                auto* vec = const_cast<backends::GenericVector*>(state.u_prev_vector);
                prev_solution_view = vec->createAssemblyView();
                assembler.setPreviousSolutionView(prev_solution_view.get());
            } else {
                assembler.setPreviousSolutionView(nullptr);
            }
        }
        if (required_history >= 2) {
            if (state.u_prev2_vector != nullptr) {
                auto* vec = const_cast<backends::GenericVector*>(state.u_prev2_vector);
                prev2_solution_view = vec->createAssemblyView();
                assembler.setPreviousSolution2View(prev2_solution_view.get());
            } else {
                assembler.setPreviousSolution2View(nullptr);
            }
        }
    } else {
        // No dt(...) required by the active time-integration context.
        assembler.setPreviousSolution({});
        assembler.setPreviousSolution2({});
        assembler.setPreviousSolutionView(nullptr);
        assembler.setPreviousSolution2View(nullptr);
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
    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> jit_constants;
    if (have_param_contracts && system.parameter_registry_.slotCount() > 0u) {
        const auto slots = system.parameter_registry_.evaluateRealSlots(state);
        jit_constants.assign(slots.begin(), slots.end());
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

    assembler.setSuppressConstraintInhomogeneity(request.suppress_constraint_inhomogeneity);

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
	        if (!want_matrix && !want_vector) {
	            continue;
	        }

	        if (oopTraceEnabled()) {
	            std::ostringstream oss;
	            oss << "assembleOperator: op='" << request.op << "' cell term test='" << test_field.name
	                << "' trial='" << trial_field.name << "' want_matrix=" << (want_matrix ? 1 : 0)
	                << " want_vector=" << (want_vector ? 1 : 0);
	            traceLog(oss.str());
	        }
	        const auto term_t0 = std::chrono::steady_clock::now();

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

	        if (oopTraceEnabled()) {
	            const auto term_t1 = std::chrono::steady_clock::now();
	            std::ostringstream oss;
	            oss << "assembleOperator: op='" << request.op << "' cell term done test='" << test_field.name
	                << "' trial='" << trial_field.name << "' time="
	                << std::chrono::duration<double>(term_t1 - term_t0).count();
	            traceLog(oss.str());
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

	            const bool want_matrix = request.want_matrix && !term.kernel->isVectorOnly();
	            const bool want_vector = request.want_vector && !term.kernel->isMatrixOnly();
	            if (oopTraceEnabled() && (want_matrix || want_vector)) {
	                std::ostringstream oss;
	                oss << "assembleOperator: op='" << request.op << "' boundary term marker=" << term.marker
	                    << " test='" << test_field.name << "' trial='" << trial_field.name << "' want_matrix="
	                    << (want_matrix ? 1 : 0) << " want_vector=" << (want_vector ? 1 : 0);
	                traceLog(oss.str());
	            }
	            const auto term_t0 = std::chrono::steady_clock::now();

	            auto r = assembler.assembleBoundaryFaces(
	                mesh, term.marker, *test_field.space, *trial_field.space, *term.kernel,
	                request.want_matrix ? matrix_out : nullptr,
	                request.want_vector ? vector_out : nullptr);
	            mergeAssemblyResult(total, r);

	            if (oopTraceEnabled() && (want_matrix || want_vector)) {
	                const auto term_t1 = std::chrono::steady_clock::now();
	                std::ostringstream oss;
	                oss << "assembleOperator: op='" << request.op << "' boundary term done marker=" << term.marker
	                    << " test='" << test_field.name << "' trial='" << trial_field.name << "' time="
	                    << std::chrono::duration<double>(term_t1 - term_t0).count();
	                traceLog(oss.str());
	            }
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

		            if (oopTraceEnabled()) {
		                std::ostringstream oss;
		                oss << "assembleOperator: op='" << request.op << "' interior-face term test='" << test_field.name
		                    << "' trial='" << trial_field.name << "' want_matrix=" << (want_matrix ? 1 : 0)
		                    << " want_vector=" << (want_vector ? 1 : 0);
		                traceLog(oss.str());
		            }
		            const auto term_t0 = std::chrono::steady_clock::now();

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

		            if (oopTraceEnabled()) {
		                const auto term_t1 = std::chrono::steady_clock::now();
		                std::ostringstream oss;
		                oss << "assembleOperator: op='" << request.op << "' interior-face term done test='"
		                    << test_field.name << "' trial='" << trial_field.name << "' time="
		                    << std::chrono::duration<double>(term_t1 - term_t0).count();
		                traceLog(oss.str());
		            }
			        }
			    }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
		    // Interface face terms (InterfaceMesh subset)
		    if (request.assemble_interface_face_terms) {
		        for (const auto& term : def.interface_faces) {
		            FE_CHECK_NOT_NULL(term.kernel.get(), "assembleOperator: interface-face term kernel");
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

		            auto assemble_on_marker = [&](int marker) {
		                auto it = system.interface_meshes_.find(marker);
		                FE_THROW_IF(it == system.interface_meshes_.end() || !it->second, InvalidArgumentException,
		                            "assembleOperator: missing InterfaceMesh for interface marker " + std::to_string(marker));
		                const auto& iface_mesh = *it->second;

		                if (want_matrix) {
		                    auto r = assembler.assembleInterfaceFaces(
		                        mesh, iface_mesh, marker, *test_field.space, *trial_field.space, *term.kernel, *matrix_out,
		                        want_vector ? vector_out : nullptr);
		                    mergeAssemblyResult(total, r);
		                } else {
		                    assembly::DenseVectorView dummy_matrix(0);
		                    auto r = assembler.assembleInterfaceFaces(
		                        mesh, iface_mesh, marker, *test_field.space, *trial_field.space, *term.kernel, dummy_matrix,
		                        vector_out);
		                    mergeAssemblyResult(total, r);
		                }
		            };

		            if (term.marker < 0) {
		                FE_THROW_IF(system.interface_meshes_.empty(), InvalidArgumentException,
		                            "assembleOperator: interface-face term requested for all interface markers, but no InterfaceMesh was registered");
		                for (const auto& kv : system.interface_meshes_) {
		                    if (!kv.second) continue;
		                    assemble_on_marker(kv.first);
		                }
		            } else {
		                assemble_on_marker(term.marker);
		            }
		        }
		    }
#endif

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

    if (request.want_matrix && matrix_out != nullptr && system.coupled_boundary_) {
        bool skip_coupled_jacobian = false;
        // Coupled BC Jacobian contributions (e.g., resistance/RCR) are dt-free terms.
        // When assembling a dt-only Jacobian (used for PTC mass-lumping), the caller sets
        // non_time_derivative_term_weight=0 to suppress dt-free contributions. Skip coupled
        // BC Jacobian handling in that case to avoid spurious "symmetry check" warnings and
        // unnecessary work.
        if (state.time_integration != nullptr) {
            constexpr Real kWeightTol = 1e-14;
            const Real w = state.time_integration->non_time_derivative_term_weight;
            if (std::abs(w) < kWeightTol) {
                skip_coupled_jacobian = true;
            }
        }
        if (skip_coupled_jacobian) {
            traceLog("coupled Jacobian: skipped (non_time_derivative_term_weight ~ 0)");
        } else {
            const auto integrals = system.coupled_boundary_->integrals().all();
            if (!integrals.empty()) {
                const auto regs = system.coupled_boundary_->registeredBoundaryFunctionals();
                if (!regs.empty()) {
                    const auto coupling_t0 = std::chrono::steady_clock::now();
                    const auto dropped_before = backends::FsilsMatrix::droppedEntryCount();
                    traceLog("coupled Jacobian: " + std::to_string(regs.size()) +
                             " registered functionals, " + std::to_string(integrals.size()) +
                             " integral slots");

                    const auto n_dofs = system.dof_handler_.getDofMap().getNumDofs();
                    const auto& global_map = system.dof_handler_.getDofMap();
                    const auto& opts = assembler.getOptions();
                    const bool owned_rows_only = (opts.ghost_policy == assembly::GhostPolicy::OwnedRowsOnly);

                    const auto& rec = system.fieldRecord(system.coupled_boundary_->primaryField());
                    FE_CHECK_NOT_NULL(rec.space.get(), "assembleOperator: coupled primary field space");

                    const auto primary_idx = static_cast<std::size_t>(rec.id);
                    FE_THROW_IF(rec.id < 0 || primary_idx >= system.field_dof_handlers_.size(), InvalidStateException,
                                "assembleOperator: invalid coupled primary field DOF handler index");

                    const auto& primary_map = system.field_dof_handlers_[primary_idx].getDofMap();
                    const auto primary_offset = system.field_dof_offsets_[primary_idx];

                auto build_integrand_with_trial = [&](const forms::FormExpr& integrand) -> forms::FormExpr {
                    const auto trial = forms::FormExpr::trialFunction(*rec.space, "u");
                    return integrand.transformNodes([&](const forms::FormExprNode& n) -> std::optional<forms::FormExpr> {
                        if (n.type() != forms::FormExprType::DiscreteField &&
                            n.type() != forms::FormExprType::StateField) {
                            return std::nullopt;
                        }
                        const auto fid = n.fieldId();
                        if (!fid || *fid != rec.id) {
                            return std::nullopt;
                        }
                        return trial;
                    });
                };

                std::vector<std::vector<std::pair<GlobalIndex, Real>>> dQ_du_by_slot(integrals.size());
                for (const auto& entry : regs) {
                    if (entry.slot >= integrals.size()) {
                        continue;
                    }

                    traceLog("  computing dQ/du for slot " + std::to_string(entry.slot) +
                             " marker=" + std::to_string(entry.def.boundary_marker));

                    const auto integrand_trial = build_integrand_with_trial(entry.def.integrand);
                    forms::BoundaryFunctionalGradientKernel g_kernel(integrand_trial, entry.def.boundary_marker);

                    SparseVectorAccumulatorView g_view(n_dofs);
                    assembler.setRowDofMap(primary_map, primary_offset);
                    assembler.setColDofMap(primary_map, primary_offset);
                    traceLog("    assembleBoundaryFaces start");
                    auto r = assembler.assembleBoundaryFaces(system.meshAccess(),
                                                             entry.def.boundary_marker,
                                                             *rec.space,
                                                             *rec.space,
                                                             g_kernel,
                                                             /*matrix_out=*/nullptr,
                                                             /*vector_out=*/&g_view);
	                    (void)r;
                    traceLog("    assembleBoundaryFaces done");

	                    auto g_pairs = g_view.entriesSorted(/*abs_tol=*/1e-16);

#if FE_HAS_MPI
                        {
                            int mpi_initialized = 0;
                            MPI_Initialized(&mpi_initialized);
                            if (mpi_initialized) {
                                g_pairs = allreduceSumSparsePairs(std::move(g_pairs), MPI_COMM_WORLD);
                            }
                        }
#endif

	                    if (entry.def.reduction == forms::BoundaryFunctional::Reduction::Average) {
	                        const Real area = system.coupled_boundary_->boundaryMeasure(entry.def.boundary_marker, state);
	                        FE_THROW_IF(std::abs(area) < 1e-14, InvalidArgumentException,
	                                    "assembleOperator: coupled boundary measure near zero for Average reduction");
	                        for (auto& kv : g_pairs) {
                            kv.second /= area;
                        }
                    }

                    traceLog("  slot " + std::to_string(entry.slot) +
                             " dQ/du: " + std::to_string(g_pairs.size()) + " nonzero entries");

                    dQ_du_by_slot[entry.slot] = std::move(g_pairs);
                }

                const auto aux_state_base = system.coupled_boundary_->auxiliaryState().values();
                std::vector<Real> integrals_base(integrals.begin(), integrals.end());
                assembler.setCoupledValues(integrals_base, aux_state_base);

                const auto aux_sens =
                    system.coupled_boundary_->computeAuxiliarySensitivityForIntegrals(integrals_base, state);
                std::span<const Real> aux_sens_span{};
                if (!aux_sens.empty()) {
                    const auto need = aux_state_base.size() * integrals_base.size();
                    FE_THROW_IF(aux_sens.size() != need, InvalidStateException,
                                "assembleOperator: coupled auxiliary sensitivity size mismatch");
                    aux_sens_span = std::span<const Real>(aux_sens.data(), aux_sens.size());
                }

                constexpr Real kAbsTol = 1e-14;

                std::vector<Real> col_vals;
                std::vector<GlobalIndex> col_dofs;
                std::array<GlobalIndex, 1> row_dof{};

                for (std::size_t k = 0; k < integrals_base.size(); ++k) {
                    if (k >= dQ_du_by_slot.size() || dQ_du_by_slot[k].empty()) {
                        continue;
                    }
                    traceLog("  dR/dQ sensitivity: slot " + std::to_string(k) + " start");
                    SparseVectorAccumulatorView dR_view(static_cast<GlobalIndex>(n_dofs));
                    assembler.setCoupledValues(integrals_base, aux_state_base);

	                    // Cell terms.
                    {
                        int cell_term_idx = 0;
	                    for (const auto& term : def.cells) {
	                        if (term.kernel->isMatrixOnly()) {
                                ++cell_term_idx;
	                            continue;
	                        }

	                        const auto& test_field = system.field_registry_.get(term.test_field);
	                        const auto& trial_field = system.field_registry_.get(term.trial_field);

                        traceLog("    cell term " + std::to_string(cell_term_idx) +
                                 " test='" + test_field.name + "' trial='" + trial_field.name + "' start");

                        assembler.setRowDofMap(system.field_dof_handlers_[static_cast<std::size_t>(test_field.id)].getDofMap(),
                                               system.field_dof_offsets_[static_cast<std::size_t>(test_field.id)]);
                        assembler.setColDofMap(system.field_dof_handlers_[static_cast<std::size_t>(trial_field.id)].getDofMap(),
                                               system.field_dof_offsets_[static_cast<std::size_t>(trial_field.id)]);

	                        auto assemble_sensitivity = [&](const auto& base_kernel) {
	                            forms::CoupledResidualSensitivityKernel s_kernel(base_kernel,
	                                                                             static_cast<std::uint32_t>(k),
	                                                                             aux_sens_span,
	                                                                             integrals_base.size());
	                            if (term.test_field == term.trial_field) {
	                                auto rr = assembler.assembleVector(system.meshAccess(), *test_field.space, s_kernel, dR_view);
	                                FE_THROW_IF(!rr.success, InvalidStateException,
	                                            "assembleOperator: coupled dR/dQ sensitivity assembly failed");
	                            } else {
	                                assembly::DenseVectorView dummy_matrix(0);
	                                auto rr = assembler.assembleBoth(system.meshAccess(), *test_field.space, *trial_field.space, s_kernel,
	                                                                 dummy_matrix, dR_view);
	                                FE_THROW_IF(!rr.success, InvalidStateException,
	                                            "assembleOperator: coupled dR/dQ sensitivity assembly failed");
	                            }
	                        };

	                        const auto* kernel = term.kernel.get();
	                        if (const auto* jit = dynamic_cast<const forms::jit::JITKernelWrapper*>(kernel)) {
	                            kernel = &jit->fallbackKernel();
	                        }

	                        if (const auto* base = dynamic_cast<const forms::NonlinearFormKernel*>(kernel)) {
	                            assemble_sensitivity(*base);
	                        } else if (const auto* base = dynamic_cast<const forms::SymbolicNonlinearFormKernel*>(kernel)) {
	                            assemble_sensitivity(*base);
	                        } else {
                                ++cell_term_idx;
	                            continue;
	                        }
                        traceLog("    cell term " + std::to_string(cell_term_idx) + " done");
                        ++cell_term_idx;
	                    }
                    }

                    // Boundary terms.
	                    if (request.assemble_boundary_terms) {
	                        for (const auto& term : def.boundary) {
	                            if (term.kernel->isMatrixOnly()) {
	                                continue;
	                            }

	                            const auto& test_field = system.field_registry_.get(term.test_field);
	                            const auto& trial_field = system.field_registry_.get(term.trial_field);

                            assembler.setRowDofMap(system.field_dof_handlers_[static_cast<std::size_t>(test_field.id)].getDofMap(),
                                                   system.field_dof_offsets_[static_cast<std::size_t>(test_field.id)]);
                            assembler.setColDofMap(system.field_dof_handlers_[static_cast<std::size_t>(trial_field.id)].getDofMap(),
                                                   system.field_dof_offsets_[static_cast<std::size_t>(trial_field.id)]);

	                            auto assemble_sensitivity = [&](const auto& base_kernel) {
	                                forms::CoupledResidualSensitivityKernel s_kernel(base_kernel,
	                                                                                 static_cast<std::uint32_t>(k),
	                                                                                 aux_sens_span,
	                                                                                 integrals_base.size());
	                                auto rr = assembler.assembleBoundaryFaces(system.meshAccess(),
	                                                                          term.marker,
	                                                                          *test_field.space,
	                                                                          *trial_field.space,
	                                                                          s_kernel,
	                                                                          /*matrix_out=*/nullptr,
	                                                                          /*vector_out=*/&dR_view);
	                                FE_THROW_IF(!rr.success, InvalidStateException,
	                                            "assembleOperator: coupled dR/dQ sensitivity assembly failed");
	                            };

	                            const auto* kernel = term.kernel.get();
	                            if (const auto* jit = dynamic_cast<const forms::jit::JITKernelWrapper*>(kernel)) {
	                                kernel = &jit->fallbackKernel();
	                            }

	                            if (const auto* base = dynamic_cast<const forms::NonlinearFormKernel*>(kernel)) {
	                                assemble_sensitivity(*base);
	                            } else if (const auto* base = dynamic_cast<const forms::SymbolicNonlinearFormKernel*>(kernel)) {
	                                assemble_sensitivity(*base);
	                            } else {
	                                continue;
	                            }
	                        }
	                    }

                    // Interior-face terms.
	                    if (request.assemble_interior_face_terms) {
	                        for (const auto& term : def.interior) {
	                            if (term.kernel->isMatrixOnly()) {
	                                continue;
	                            }

	                            const auto& test_field = system.field_registry_.get(term.test_field);
	                            const auto& trial_field = system.field_registry_.get(term.trial_field);

                            assembler.setRowDofMap(system.field_dof_handlers_[static_cast<std::size_t>(test_field.id)].getDofMap(),
                                                   system.field_dof_offsets_[static_cast<std::size_t>(test_field.id)]);
                            assembler.setColDofMap(system.field_dof_handlers_[static_cast<std::size_t>(trial_field.id)].getDofMap(),
                                                   system.field_dof_offsets_[static_cast<std::size_t>(trial_field.id)]);

	                            auto assemble_sensitivity = [&](const auto& base_kernel) {
	                                forms::CoupledResidualSensitivityKernel s_kernel(base_kernel,
	                                                                                 static_cast<std::uint32_t>(k),
	                                                                                 aux_sens_span,
	                                                                                 integrals_base.size());
	                                assembly::DenseVectorView dummy_matrix(0);
	                                auto rr = assembler.assembleInteriorFaces(system.meshAccess(),
	                                                                          *test_field.space,
	                                                                          *trial_field.space,
	                                                                          s_kernel,
	                                                                          dummy_matrix,
	                                                                          &dR_view);
	                                FE_THROW_IF(!rr.success, InvalidStateException,
	                                            "assembleOperator: coupled dR/dQ sensitivity assembly failed");
	                            };

	                            const auto* kernel = term.kernel.get();
	                            if (const auto* jit = dynamic_cast<const forms::jit::JITKernelWrapper*>(kernel)) {
	                                kernel = &jit->fallbackKernel();
	                            }

	                            if (const auto* base = dynamic_cast<const forms::NonlinearFormKernel*>(kernel)) {
	                                assemble_sensitivity(*base);
	                            } else if (const auto* base = dynamic_cast<const forms::SymbolicNonlinearFormKernel*>(kernel)) {
	                                assemble_sensitivity(*base);
	                            } else {
	                                continue;
	                            }
	                        }
	                    }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
                    // Interface-face terms.
	                    if (request.assemble_interface_face_terms) {
	                        for (const auto& term : def.interface_faces) {
	                            if (term.kernel->isMatrixOnly()) {
	                                continue;
	                            }

	                            const auto& test_field = system.field_registry_.get(term.test_field);
	                            const auto& trial_field = system.field_registry_.get(term.trial_field);

                            assembler.setRowDofMap(system.field_dof_handlers_[static_cast<std::size_t>(test_field.id)].getDofMap(),
                                                   system.field_dof_offsets_[static_cast<std::size_t>(test_field.id)]);
                            assembler.setColDofMap(system.field_dof_handlers_[static_cast<std::size_t>(trial_field.id)].getDofMap(),
                                                   system.field_dof_offsets_[static_cast<std::size_t>(trial_field.id)]);

	                            auto assemble_sensitivity = [&](const auto& base_kernel) {
	                                forms::CoupledResidualSensitivityKernel s_kernel(base_kernel,
	                                                                                 static_cast<std::uint32_t>(k),
	                                                                                 aux_sens_span,
	                                                                                 integrals_base.size());

	                                auto assemble_on_marker = [&](int marker) {
	                                    auto it = system.interface_meshes_.find(marker);
	                                    FE_THROW_IF(it == system.interface_meshes_.end() || !it->second, InvalidArgumentException,
	                                                "assembleOperator: missing InterfaceMesh for interface marker " + std::to_string(marker));
	                                    const auto& iface_mesh = *it->second;
	                                    assembly::DenseVectorView dummy_matrix(0);
	                                    auto rr = assembler.assembleInterfaceFaces(system.meshAccess(),
	                                                                               iface_mesh,
	                                                                               marker,
	                                                                               *test_field.space,
	                                                                               *trial_field.space,
	                                                                               s_kernel,
	                                                                               dummy_matrix,
	                                                                               &dR_view);
	                                    FE_THROW_IF(!rr.success, InvalidStateException,
	                                                "assembleOperator: coupled dR/dQ sensitivity assembly failed");
	                                };

	                                if (term.marker < 0) {
	                                    for (const auto& kv : system.interface_meshes_) {
	                                        if (!kv.second) continue;
	                                        assemble_on_marker(kv.first);
	                                    }
	                                } else {
	                                    assemble_on_marker(term.marker);
	                                }
	                            };

	                            const auto* kernel = term.kernel.get();
	                            if (const auto* jit = dynamic_cast<const forms::jit::JITKernelWrapper*>(kernel)) {
	                                kernel = &jit->fallbackKernel();
	                            }

	                            if (const auto* base = dynamic_cast<const forms::NonlinearFormKernel*>(kernel)) {
	                                assemble_sensitivity(*base);
	                            } else if (const auto* base = dynamic_cast<const forms::SymbolicNonlinearFormKernel*>(kernel)) {
	                                assemble_sensitivity(*base);
	                            } else {
	                                continue;
	                            }
	                        }
	                    }
#endif

                    auto dR_dQ = dR_view.entriesSorted(kAbsTol);

                    if (owned_rows_only) {
                        dR_dQ.erase(std::remove_if(dR_dQ.begin(), dR_dQ.end(),
                                                   [&](const auto& kv) { return !global_map.isOwnedDof(kv.first); }),
                                    dR_dQ.end());
                    }

                    // Globally reduce dR_dQ for consistent rank-1 detection across MPI ranks.
                    // Each rank has only owned-row entries; allreduce merges them so all ranks
                    // see the same dR_dQ and make identical symmetry decisions.
                    auto dR_dQ_global = dR_dQ;
#if FE_HAS_MPI
                    {
                        int mpi_initialized = 0;
                        MPI_Initialized(&mpi_initialized);
                        if (mpi_initialized) {
                            dR_dQ_global = allreduceSumSparsePairs(std::move(dR_dQ_global), MPI_COMM_WORLD);
                        }
                    }
#endif

                    const auto& g = dQ_du_by_slot[k];

                    traceLog("  slot " + std::to_string(k) +
                             " dR/dQ: " + std::to_string(dR_dQ.size()) +
                             " (local) " + std::to_string(dR_dQ_global.size()) +
                             " (global) entries, dQ/du: " + std::to_string(g.size()) +
                             " -> rank-1 writes " + std::to_string(dR_dQ.size()) + " x " +
                             std::to_string(g.size()) + " = " +
                             std::to_string(dR_dQ.size() * g.size()) + " matrix entries");

                    // Detect symmetric rank-1 structure: dR_dQ = sigma * dQ_du
                    // If so, store as RankOneUpdate{sigma, dQ_du} for preconditioner correction.
                    // Use globally-reduced dR_dQ so all MPI ranks make the same decision.
                    bool extracted_rank_one = false;
                    Real sym_rel_err = -1.0;
                    if (!g.empty() && !dR_dQ_global.empty()) {
                        // Build index maps for dQ_du (g) to enable lookup by DOF index.
                        std::unordered_map<GlobalIndex, Real> g_map;
                        g_map.reserve(g.size());
                        Real g_dot_g = 0.0;
                        for (const auto& kv : g) {
                            g_map[kv.first] = kv.second;
                            g_dot_g += kv.second * kv.second;
                        }

                        if (g_dot_g > kAbsTol * kAbsTol) {
                            // sigma = (dR_dQ . dQ_du) / (dQ_du . dQ_du)
                            Real dR_dot_g = 0.0;
                            for (const auto& kv : dR_dQ_global) {
                                auto it = g_map.find(kv.first);
                                if (it != g_map.end()) {
                                    dR_dot_g += kv.second * it->second;
                                }
                            }
                            const Real sigma = dR_dot_g / g_dot_g;

                            // Compute remainder: w_rem = dR_dQ - sigma * dQ_du
                            Real rem_norm_sq = 0.0;
                            Real dR_norm_sq = 0.0;
                            // Contributions from DOFs in dR_dQ (global)
                            for (const auto& kv : dR_dQ_global) {
                                dR_norm_sq += kv.second * kv.second;
                                auto it = g_map.find(kv.first);
                                const Real g_val = (it != g_map.end()) ? it->second : Real(0.0);
                                const Real rem = kv.second - sigma * g_val;
                                rem_norm_sq += rem * rem;
                            }
                            // Contributions from DOFs in dQ_du but not in dR_dQ (global)
                            std::unordered_map<GlobalIndex, Real> dR_map;
                            dR_map.reserve(dR_dQ_global.size());
                            for (const auto& kv : dR_dQ_global) {
                                dR_map[kv.first] = kv.second;
                            }
                            for (const auto& kv : g) {
                                if (dR_map.find(kv.first) == dR_map.end()) {
                                    const Real rem = -sigma * kv.second;
                                    rem_norm_sq += rem * rem;
                                }
                            }

                            const Real dR_norm = std::sqrt(dR_norm_sq);
                            const Real rem_norm = std::sqrt(rem_norm_sq);
                            sym_rel_err = (dR_norm > kAbsTol) ? (rem_norm / dR_norm) : -1.0;
                            constexpr Real kSymTol = 1e-2;

                            if (dR_norm > kAbsTol && rem_norm / dR_norm < kSymTol) {
                                // Symmetric! Store as rank-1 update.
                                backends::RankOneUpdate update;
                                update.sigma = sigma;
                                update.v = g; // dQ_du
                                system.last_rank_one_updates_.push_back(std::move(update));
                                extracted_rank_one = true;

                                traceLog("  slot " + std::to_string(k) +
                                         " -> symmetric rank-1 update: sigma=" + std::to_string(static_cast<double>(sigma)) +
                                         " |rem|/|dR|=" + std::to_string(static_cast<double>(rem_norm / dR_norm)));
                            } else {
                                traceLog("  slot " + std::to_string(k) +
                                         " -> NOT symmetric: sigma=" + std::to_string(static_cast<double>(sigma)) +
                                         " |rem|/|dR|=" + std::to_string(
                                             dR_norm > kAbsTol ? static_cast<double>(rem_norm / dR_norm) : 0.0) +
                                         " -> assembling full outer product");
                            }
                        }
                    }

                    // Fallback: assemble full outer product into sparse matrix.
                    if (!extracted_rank_one) {
                        if (g.empty() || dR_dQ.empty()) {
                            // Nothing to assemble (e.g., dt-only Jacobian or inactive coupling).
                            continue;
                        }
                        std::string extra;
                        if (sym_rel_err >= 0.0 && std::isfinite(static_cast<double>(sym_rel_err))) {
                            extra = " (rel_err=" + std::to_string(static_cast<double>(sym_rel_err)) + ")";
                        }
                        FE_LOG_WARNING("Coupled BC slot " + std::to_string(k) +
                                       " failed symmetry check" + extra +
                                       ". Falling back to sparse assembly, which may drop entries in FSILS.");

                        col_dofs.resize(g.size());
                        col_vals.resize(g.size());
                        for (std::size_t j = 0; j < g.size(); ++j) {
                            col_dofs[j] = g[j].first;
                            col_vals[j] = g[j].second;
                        }

                        std::vector<Real> row_vals(col_dofs.size());
                        for (const auto& ri : dR_dQ) {
                            row_dof[0] = ri.first;
                            const Real scale = ri.second;
                            for (std::size_t j = 0; j < col_vals.size(); ++j) {
                                row_vals[j] = scale * col_vals[j];
                            }
                            matrix_out->addMatrixEntries(std::span<const GlobalIndex>(row_dof.data(), row_dof.size()),
                                                         std::span<const GlobalIndex>(col_dofs.data(), col_dofs.size()),
                                                         std::span<const Real>(row_vals.data(), row_vals.size()),
                                                         assembly::AddMode::Add);
                        }
                    }
                }

                const auto dropped_after = backends::FsilsMatrix::droppedEntryCount();
                const auto dropped_coupling = dropped_after - dropped_before;
                const auto coupling_elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - coupling_t0).count();

                traceLog("coupled Jacobian: completed in " +
                         std::to_string(coupling_elapsed) + " s, " +
                         std::to_string(dropped_coupling) + " entries dropped by sparsity filter");
                if (dropped_coupling > 0) {
                    FE_LOG_WARNING("coupled Jacobian: " + std::to_string(dropped_coupling) +
                                   " matrix entries dropped (outside sparsity pattern)");
                }

                assembler.setCoupledValues(integrals_base, aux_state_base);
                }
            }
        }
    }

    assembler.finalize(request.want_matrix ? matrix_out : nullptr,
                       request.want_vector ? vector_out : nullptr);

    return total;
}

} // namespace systems
} // namespace FE
} // namespace svmp
