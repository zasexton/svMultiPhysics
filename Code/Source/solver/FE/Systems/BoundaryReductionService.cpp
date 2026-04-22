/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/BoundaryReductionService.h"

#include "Assembly/FunctionalAssembler.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Forms/BoundaryFunctional.h"
#include "Forms/FormExpr.h"
#include "Forms/FormKernels.h"  // for BoundaryFunctionalGradientKernel
#include "Forms/JIT/ExternalCalls.h"
#include "Systems/FESystem.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/H1Space.h"
#include "Systems/SystemsExceptions.h"
#include "Core/FEConfig.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace systems {

namespace {

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

Real allreduceSum(Real local, MPI_Comm comm)
{
    Real global = local;
    MPI_Allreduce(&local, &global, 1, mpiRealType(), MPI_SUM, comm);
    return global;
}
#endif

/// Trivial FunctionalKernel that integrates 1.0 over boundary faces to compute
/// the geometric measure (area in 3D, length in 2D) of a boundary marker.
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

} // namespace

// ---------------------------------------------------------------------------
//  Construction
// ---------------------------------------------------------------------------

BoundaryReductionService::BoundaryReductionService(FESystem& system, FieldId primary_field)
    : system_(system)
    , primary_field_(primary_field)
{
    FE_THROW_IF(primary_field_ == INVALID_FIELD_ID, InvalidArgumentException,
                "BoundaryReductionService: primary_field is invalid");
    // GEOMETRY_FIELD_ID is accepted — it means geometry-only evaluation
    // using a default P1 space for quadrature.
}

BoundaryReductionService::~BoundaryReductionService() = default;

const spaces::FunctionSpace& BoundaryReductionService::geometrySpace() const
{
    if (!geometry_space_) {
        // Create a default P1 Lagrange space from the mesh's element type.
        // This provides quadrature context for geometry-only integrands.
        const auto& mesh = system_.meshAccess();
        const auto cell_type = mesh.getCellType(0);
        geometry_space_ = std::make_shared<spaces::H1Space>(cell_type, 1);
    }
    return *geometry_space_;
}

// ---------------------------------------------------------------------------
//  Registration
// ---------------------------------------------------------------------------

void BoundaryReductionService::addBoundaryFunctional(forms::BoundaryFunctional functional)
{
    FE_THROW_IF(functional.name.empty(), InvalidArgumentException,
                "BoundaryReductionService::addBoundaryFunctional: empty name");
    FE_THROW_IF(!functional.integrand.isValid(), InvalidArgumentException,
                "BoundaryReductionService::addBoundaryFunctional: invalid integrand");

    auto it = name_to_functional_.find(functional.name);
    if (it != name_to_functional_.end()) {
        // Duplicate with identical properties is accepted silently.
        const auto& existing = functionals_.at(it->second).def;
        FE_THROW_IF(existing.boundary_marker != functional.boundary_marker, InvalidArgumentException,
                    "BoundaryReductionService::addBoundaryFunctional: name '" + functional.name +
                    "' already registered with different boundary_marker");
        FE_THROW_IF(existing.reduction != functional.reduction, InvalidArgumentException,
                    "BoundaryReductionService::addBoundaryFunctional: name '" + functional.name +
                    "' already registered with different reduction");
        FE_THROW_IF(existing.integrand.toString() != functional.integrand.toString(), InvalidArgumentException,
                    "BoundaryReductionService::addBoundaryFunctional: name '" + functional.name +
                    "' already registered with different integrand");
        return;
    }

    const auto idx = functionals_.size();
    functionals_.push_back(CompiledFunctional{std::move(functional), nullptr});
    name_to_functional_.emplace(functionals_.back().def.name, idx);
}

bool BoundaryReductionService::hasFunctional(std::string_view name) const noexcept
{
    return name_to_functional_.find(std::string(name)) != name_to_functional_.end();
}

// ---------------------------------------------------------------------------
//  Compilation options
// ---------------------------------------------------------------------------

void BoundaryReductionService::setCompilerOptions(const forms::SymbolicOptions& options)
{
    compiler_options_ = options;

    // Invalidate compiled kernels.
    for (auto& entry : functionals_) {
        entry.kernel.reset();
    }
}

// ---------------------------------------------------------------------------
//  Multi-field support
// ---------------------------------------------------------------------------

void BoundaryReductionService::registerSecondaryField(const assembly::FieldSolutionBinding& binding)
{
    for (auto& existing : secondary_fields_) {
        if (existing.field == binding.field) {
            existing = binding;
            return;
        }
    }
    secondary_fields_.push_back(binding);
}

void BoundaryReductionService::setDofPerNode(int dof_per_node) noexcept
{
    dof_per_node_ = dof_per_node;
}

// ---------------------------------------------------------------------------
//  Compilation
// ---------------------------------------------------------------------------

void BoundaryReductionService::compileFunctionalIfNeeded(CompiledFunctional& entry)
{
    if (entry.kernel) return;
    entry.kernel = forms::compileBoundaryFunctionalKernel(entry.def, compiler_options_);
}

// ---------------------------------------------------------------------------
//  Assembler configuration (shared between evaluateFunctional and boundaryMeasure)
// ---------------------------------------------------------------------------

void BoundaryReductionService::configureAssembler(assembly::FunctionalAssembler& assembler,
                                                   const SystemStateView& state,
                                                   bool bind_solution) const
{
    FE_THROW_IF(!system_.isSetup(), InvalidArgumentException,
                "BoundaryReductionService: system.setup() has not been called");

    assembler.setMesh(system_.meshAccess());

    // For GEOMETRY_FIELD_ID, use the default geometry space instead of a
    // field record.  This enables geometry-only integrands (∫ 1 ds, etc.)
    // without any registered FE field.
    const auto* primary_rec =
        (primary_field_ == GEOMETRY_FIELD_ID) ? nullptr : &system_.fieldRecord(primary_field_);
    if (primary_field_ == GEOMETRY_FIELD_ID) {
        assembler.setDofMap(system_.dofHandler().getDofMap());
        assembler.setPrimaryFieldDofOffset(0);
        assembler.setSpace(geometrySpace());
        // No primary field to bind — geometry-only evaluation.
    } else {
        const auto& rec = *primary_rec;
        FE_CHECK_NOT_NULL(rec.space.get(), "BoundaryReductionService: field space");
        if (bind_solution) {
            assembler.setDofMap(system_.dofHandler().getDofMap());
            assembler.setPrimaryFieldDofOffset(0);
        } else {
            assembler.setDofMap(system_.fieldDofHandler(primary_field_).getDofMap());
            assembler.setPrimaryFieldDofOffset(system_.fieldDofOffset(primary_field_));
        }
        assembler.setSpace(*rec.space);
        assembler.setPrimaryField(primary_field_);
    }
    assembler.setTimeIntegrationContext(state.time_integration);
    assembler.setTime(static_cast<Real>(state.time));
    assembler.setTimeStep(static_cast<Real>(state.dt));

    // Parameter contracts.
    const auto& preg = system_.parameterRegistry();
    const bool have_param_contracts = !preg.specs().empty();
    // Note: these are stack locals; assembler must use them within this scope.
    // They are passed by pointer-to-function, so they must outlive the assembly call.
    // The caller is responsible for keeping them alive.
    // We use thread_local to avoid repeated allocations across calls.
    thread_local std::function<std::optional<Real>(std::string_view)> get_real_param_wrapped{};
    thread_local std::function<std::optional<params::Value>(std::string_view)> get_param_wrapped{};
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

    // JIT external call table.
    thread_local forms::jit::external::ExternalCallTableV1 jit_table;
    jit_table.context = state.user_data;
    assembler.setUserData(&jit_table);

    // JIT constants from parameter registry.
    thread_local std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> jit_constants;
    if (have_param_contracts && preg.slotCount() > 0u) {
        const auto slots = preg.evaluateRealSlots(state);
        jit_constants.assign(slots.begin(), slots.end());
        assembler.setJITConstants(jit_constants);
    } else {
        assembler.setJITConstants({});
    }

    assembler.setCoupledValues({}, {});

    if (bind_solution) {
        // Set primary field solution. The DofMap maps cell DOFs to global
        // indices in the block layout, and sol[global_dof] gives the value.
        assembler.setSolution(state.u);
        if (state.u_vector != nullptr) {
            auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
            (void)vec;
        }

        // When sampling solution coefficients through a backend vector view,
        // stay on the monolithic system DOF map and extract per-field slices
        // via explicit bindings. Distributed field-local DOF maps can diverge
        // from the system-global ids expected by the view even for
        // primary-only evaluations.
        if (system_.isSetup()) {
            assembler.setDofPerNode(0);  // block DOF layout mode

            auto register_field_binding = [&](FieldId field_id,
                                              const spaces::FunctionSpace& field_space,
                                              int components) {
                const auto& sec_dh = system_.fieldDofHandler(field_id);

                assembly::FieldSolutionBinding binding;
                binding.field = field_id;
                binding.space = &field_space;
                binding.dof_map = &sec_dh.getDofMap();
                binding.dof_offset = system_.fieldDofOffset(field_id);
                binding.field_global_size = sec_dh.getNumDofs();
                binding.field_type = field_space.field_type();
                binding.value_dimension = components;
                binding.n_components = components;
                assembler.registerFieldBinding(binding);
            };

            FE_CHECK_NOT_NULL(primary_rec, "BoundaryReductionService: primary record");
            register_field_binding(primary_field_,
                                   *primary_rec->space,
                                   primary_rec->components);
            for (const auto& fb : secondary_fields_) {
                register_field_binding(fb.field, *fb.space, fb.n_components);
            }
        }

        // Previous solutions.
        if (!state.u_history.empty()) {
            for (std::size_t k = 0; k < state.u_history.size(); ++k) {
                assembler.setPreviousSolutionK(static_cast<int>(k + 1), state.u_history[k]);
            }
        } else {
            assembler.setPreviousSolution(state.u_prev);
            assembler.setPreviousSolution2(state.u_prev2);
        }
    }
}

// ---------------------------------------------------------------------------
//  Evaluation
// ---------------------------------------------------------------------------

Real BoundaryReductionService::evaluateFunctionalEntry(CompiledFunctional& entry,
                                                        const SystemStateView& state)
{
    compileFunctionalIfNeeded(entry);
    FE_CHECK_NOT_NULL(entry.kernel.get(), "BoundaryReductionService::evaluateFunctional: kernel");

    auto refreshGhostedCoefficients = [](const backends::GenericVector* vec_ptr) {
        if (vec_ptr == nullptr) {
            return;
        }
        // Explicit sampled reductions read FE coefficients through backend
        // vector views. On distributed backends those views must see fresh
        // owner-to-ghost copies or sampled partitioned inputs can silently
        // evaluate with stale zeros on ghosted DOFs.
        auto* vec = const_cast<backends::GenericVector*>(vec_ptr);
        vec->updateGhosts();
    };

    refreshGhostedCoefficients(state.u_vector);
    refreshGhostedCoefficients(state.u_prev_vector);
    refreshGhostedCoefficients(state.u_prev2_vector);

    assembly::FunctionalAssembler assembler;
    configureAssembler(assembler, state, /*bind_solution=*/true);

    // Bind solution view for MPI-aware DOF access.
    std::unique_ptr<assembly::GlobalSystemView> solution_view;
    if (state.u_vector != nullptr) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
        solution_view = vec->createAssemblyView();
        assembler.setSolutionView(solution_view.get());
    }

    if (const char* trace_dofs_env = std::getenv("SVMP_MONO_AUX_TRACE_DOFS");
        trace_dofs_env != nullptr && *trace_dofs_env != '\0' && solution_view != nullptr) {
        static thread_local int trace_budget = 32;
        if (trace_budget > 0) {
            std::vector<GlobalIndex> trace_dofs;
            const char* cursor = trace_dofs_env;
            while (*cursor != '\0') {
                char* end = nullptr;
                const long value = std::strtol(cursor, &end, 10);
                if (end != cursor) {
                    trace_dofs.push_back(static_cast<GlobalIndex>(value));
                    cursor = end;
                }
                while (*cursor == ',' || *cursor == ' ' || *cursor == ';') {
                    ++cursor;
                }
                if (end == cursor && *cursor != '\0') {
                    ++cursor;
                }
            }
            if (!trace_dofs.empty()) {
                std::vector<Real> trace_values(trace_dofs.size(), 0.0);
                solution_view->getVectorEntries(trace_dofs, trace_values);
                int rank = 0;
#if FE_HAS_MPI
                int mpi_initialized = 0;
                MPI_Initialized(&mpi_initialized);
                if (mpi_initialized) {
                    MPI_Comm_rank(system_.dofHandler().mpiComm(), &rank);
                }
#endif
                const auto& constraints = system_.constraints();
                std::fprintf(stderr,
                             "[BoundaryReductionTraceDofs] rank=%d functional='%s' marker=%d",
                             rank,
                             entry.def.name.c_str(),
                             entry.def.boundary_marker);
                for (std::size_t i = 0; i < trace_dofs.size(); ++i) {
                    std::fprintf(stderr,
                                 " dof=%lld constrained=%d value=%.17g",
                                 static_cast<long long>(trace_dofs[i]),
                                 constraints.isConstrained(trace_dofs[i]) ? 1 : 0,
                                 static_cast<double>(trace_values[i]));
                }
                std::fprintf(stderr, "\n");
                --trace_budget;
            }
        }
    }

    // Previous solution views for MPI.
    std::unique_ptr<assembly::GlobalSystemView> prev_solution_view;
    std::unique_ptr<assembly::GlobalSystemView> prev2_solution_view;
    if (state.u_prev_vector != nullptr) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_prev_vector);
        prev_solution_view = vec->createAssemblyView();
        assembler.setPreviousSolutionView(prev_solution_view.get());
    }
    if (state.u_prev2_vector != nullptr) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_prev2_vector);
        prev2_solution_view = vec->createAssemblyView();
        assembler.setPreviousSolution2View(prev2_solution_view.get());
    }

    Real raw = 0.0;
    if (entry.def.is_domain_functional) {
        if (entry.def.region_marker >= 0) {
            std::vector<GlobalIndex> cells;
            system_.meshAccess().forEachCell([&](GlobalIndex cell_id) {
                if (system_.meshAccess().getCellDomainId(cell_id) == entry.def.region_marker) {
                    cells.push_back(cell_id);
                }
            });
            raw = assembler.assembleScalarOverCells(*entry.kernel, cells);
        } else {
            raw = assembler.assembleScalar(*entry.kernel);
        }
    } else {
        raw = assembler.assembleBoundaryScalar(*entry.kernel, entry.def.boundary_marker);
    }

    const auto local_result = assembler.getLastResult();

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        raw = allreduceSum(raw, system_.dofHandler().mpiComm());
    }
#endif

    switch (entry.def.reduction) {
        case forms::BoundaryFunctional::Reduction::Sum:
            if (std::getenv("SVMP_MONO_AUX_TRACE") != nullptr) {
                std::fprintf(stderr,
                             "[BoundaryReductionService] functional='%s' marker=%d local_faces=%lld local_raw=%.17g\n",
                             entry.def.name.c_str(),
                             entry.def.boundary_marker,
                             static_cast<long long>(local_result.faces_processed),
                             static_cast<double>(raw));
            }
            return raw;
        case forms::BoundaryFunctional::Reduction::Average: {
            const Real area = const_cast<BoundaryReductionService*>(this)->boundaryMeasure(
                entry.def.boundary_marker, state);
            FE_THROW_IF(std::abs(area) < 1e-14, InvalidArgumentException,
                        "BoundaryReductionService: boundary measure is near zero for Average reduction");
            return raw / area;
        }
        case forms::BoundaryFunctional::Reduction::Max:
        case forms::BoundaryFunctional::Reduction::Min:
            FE_THROW(NotImplementedException,
                     "BoundaryReductionService: Max/Min reductions are not implemented");
    }

    return raw;
}

Real BoundaryReductionService::evaluateFunctional(std::string_view name, const SystemStateView& state)
{
    auto it = name_to_functional_.find(std::string(name));
    FE_THROW_IF(it == name_to_functional_.end(), InvalidArgumentException,
                "BoundaryReductionService::evaluateFunctional: unknown functional '" +
                std::string(name) + "'");
    return evaluateFunctionalEntry(functionals_.at(it->second), state);
}

Real BoundaryReductionService::evaluateFunctionalOverCells(
    std::string_view name,
    std::span<const GlobalIndex> cell_ids,
    const SystemStateView& state)
{
    auto it = name_to_functional_.find(std::string(name));
    FE_THROW_IF(it == name_to_functional_.end(), InvalidArgumentException,
                "BoundaryReductionService::evaluateFunctionalOverCells: unknown functional '" +
                std::string(name) + "'");

    auto& entry = functionals_.at(it->second);
    FE_THROW_IF(!entry.def.is_domain_functional, InvalidArgumentException,
                "BoundaryReductionService::evaluateFunctionalOverCells: functional '" +
                std::string(name) + "' is not a domain functional");

    compileFunctionalIfNeeded(entry);
    FE_CHECK_NOT_NULL(entry.kernel.get(),
                      "BoundaryReductionService::evaluateFunctionalOverCells: kernel");

    auto refreshGhostedCoefficients = [](const backends::GenericVector* vec_ptr) {
        if (vec_ptr == nullptr) {
            return;
        }
        auto* vec = const_cast<backends::GenericVector*>(vec_ptr);
        vec->updateGhosts();
    };

    refreshGhostedCoefficients(state.u_vector);
    refreshGhostedCoefficients(state.u_prev_vector);
    refreshGhostedCoefficients(state.u_prev2_vector);

    assembly::FunctionalAssembler assembler;
    configureAssembler(assembler, state, /*bind_solution=*/true);

    std::unique_ptr<assembly::GlobalSystemView> solution_view;
    if (state.u_vector != nullptr) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
        solution_view = vec->createAssemblyView();
        assembler.setSolutionView(solution_view.get());
    }

    std::unique_ptr<assembly::GlobalSystemView> prev_solution_view;
    std::unique_ptr<assembly::GlobalSystemView> prev2_solution_view;
    if (state.u_prev_vector != nullptr) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_prev_vector);
        prev_solution_view = vec->createAssemblyView();
        assembler.setPreviousSolutionView(prev_solution_view.get());
    }
    if (state.u_prev2_vector != nullptr) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_prev2_vector);
        prev2_solution_view = vec->createAssemblyView();
        assembler.setPreviousSolution2View(prev2_solution_view.get());
    }

    Real raw = assembler.assembleScalarOverCells(*entry.kernel, cell_ids);

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        raw = allreduceSum(raw, system_.dofHandler().mpiComm());
    }
#endif

    if (entry.def.reduction == forms::BoundaryFunctional::Reduction::Sum) {
        return raw;
    }

    FE_THROW(NotImplementedException,
             "BoundaryReductionService::evaluateFunctionalOverCells: only Sum reduction is supported");
}

std::vector<Real> BoundaryReductionService::evaluateAll(const SystemStateView& state)
{
    std::vector<Real> results;
    results.reserve(functionals_.size());
    for (auto& entry : functionals_) {
        results.push_back(evaluateFunctionalEntry(entry, state));
    }
    return results;
}

Real BoundaryReductionService::boundaryMeasure(int boundary_marker, const SystemStateView& state)
{
    auto it = boundary_measure_cache_.find(boundary_marker);
    if (it != boundary_measure_cache_.end()) {
        return it->second;
    }

    assembly::FunctionalAssembler assembler;
    configureAssembler(assembler, state, /*bind_solution=*/false);

    BoundaryMeasureKernel measure_kernel;
    Real area = assembler.assembleBoundaryScalar(measure_kernel, boundary_marker);

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        area = allreduceSum(area, system_.dofHandler().mpiComm());
    }
#endif

    boundary_measure_cache_.emplace(boundary_marker, area);
    return area;
}

// ---------------------------------------------------------------------------
//  Sensitivity
// ---------------------------------------------------------------------------

std::vector<BoundaryReductionService::SensitivityEntry>
BoundaryReductionService::evaluateFunctionalGradient(std::string_view name,
                                                      const SystemStateView& state,
                                                      bool apply_constraints)
{
    // Default: linearize w.r.t. the primary field.
    return evaluateFunctionalGradient(name, primary_field_, state, apply_constraints);
}

std::vector<BoundaryReductionService::SensitivityEntry>
BoundaryReductionService::evaluateFunctionalGradient(std::string_view name,
                                                      FieldId target_field,
                                                      const SystemStateView& state,
                                                      bool apply_constraints)
{
    auto it = name_to_functional_.find(std::string(name));
    FE_THROW_IF(it == name_to_functional_.end(), InvalidArgumentException,
                "BoundaryReductionService::evaluateFunctionalGradient: unknown functional '" +
                std::string(name) + "'");

    auto& entry = functionals_.at(it->second);
    compileFunctionalIfNeeded(entry);

    FE_THROW_IF(!system_.isSetup(), InvalidArgumentException,
                "BoundaryReductionService::evaluateFunctionalGradient: system.setup() not called");

    // Geometry-only functionals have no field dependence.
    if (target_field == GEOMETRY_FIELD_ID) {
        return {};
    }

    const auto& rec = system_.fieldRecord(target_field);
    FE_CHECK_NOT_NULL(rec.space.get(),
                      "BoundaryReductionService::evaluateFunctionalGradient: field space");

    // Build integrand with trial: replace DiscreteField(target_field) → TrialFunction.
    // Other fields' DiscreteField nodes remain as constants.
    const auto trial = forms::FormExpr::trialFunction(*rec.space, "u");
    auto integrand_trial = entry.def.integrand.transformNodes(
        [&](const forms::FormExprNode& n) -> std::optional<forms::FormExpr> {
            if (n.type() != forms::FormExprType::DiscreteField &&
                n.type() != forms::FormExprType::StateField) {
                return std::nullopt;
            }
            const auto fid = n.fieldId();
            if (!fid || *fid != target_field) {
                return std::nullopt;
            }
            return trial;
        });

    // Symbolic gradient via BoundaryFunctionalGradientKernel + GradAccumulator.
    const int region_marker =
        entry.def.is_domain_functional ? entry.def.region_marker : -1;
    auto grad_entries = system_.assembleBoundaryGradient(
        target_field,
        integrand_trial,
        entry.def.boundary_marker,
        state,
        apply_constraints,
        region_marker);

    // Apply reduction: for Average, divide by boundary measure.
    if (entry.def.reduction == forms::BoundaryFunctional::Reduction::Average) {
        const Real measure = boundaryMeasure(entry.def.boundary_marker, state);
        if (measure > 0.0) {
            for (auto& se : grad_entries) se.value /= measure;
        }
    }

    return grad_entries;
}

std::vector<BoundaryReductionService::SensitivityEntry>
BoundaryReductionService::evaluateFunctionalGradientOverCells(
    std::string_view name,
    FieldId target_field,
    std::span<const GlobalIndex> cell_ids,
    const SystemStateView& state,
    bool apply_constraints)
{
    auto it = name_to_functional_.find(std::string(name));
    FE_THROW_IF(it == name_to_functional_.end(), InvalidArgumentException,
                "BoundaryReductionService::evaluateFunctionalGradientOverCells: "
                "unknown functional '" + std::string(name) + "'");

    auto& entry = functionals_.at(it->second);
    FE_THROW_IF(!entry.def.is_domain_functional, InvalidArgumentException,
                "BoundaryReductionService::evaluateFunctionalGradientOverCells: "
                "functional '" + std::string(name) +
                "' is not a domain functional");
    compileFunctionalIfNeeded(entry);

    FE_THROW_IF(!system_.isSetup(), InvalidArgumentException,
                "BoundaryReductionService::evaluateFunctionalGradientOverCells: "
                "system.setup() not called");

    if (target_field == GEOMETRY_FIELD_ID || cell_ids.empty()) {
        return {};
    }

    const auto& rec = system_.fieldRecord(target_field);
    FE_CHECK_NOT_NULL(rec.space.get(),
                      "BoundaryReductionService::evaluateFunctionalGradientOverCells: "
                      "field space");

    const auto trial = forms::FormExpr::trialFunction(*rec.space, "u");
    auto integrand_trial = entry.def.integrand.transformNodes(
        [&](const forms::FormExprNode& n) -> std::optional<forms::FormExpr> {
            if (n.type() != forms::FormExprType::DiscreteField &&
                n.type() != forms::FormExprType::StateField) {
                return std::nullopt;
            }
            const auto fid = n.fieldId();
            if (!fid || *fid != target_field) {
                return std::nullopt;
            }
            return trial;
        });

    return system_.assembleBoundaryGradient(
        target_field,
        integrand_trial,
        entry.def.boundary_marker,
        state,
        apply_constraints,
        /*region_marker=*/-1,
        cell_ids);
}

// ---------------------------------------------------------------------------
//  Accessors
// ---------------------------------------------------------------------------

const forms::BoundaryFunctional& BoundaryReductionService::functionalDef(std::string_view name) const
{
    auto it = name_to_functional_.find(std::string(name));
    FE_THROW_IF(it == name_to_functional_.end(), InvalidArgumentException,
                "BoundaryReductionService::functionalDef: unknown functional '" +
                std::string(name) + "'");
    return functionals_.at(it->second).def;
}

std::vector<forms::BoundaryFunctional> BoundaryReductionService::allFunctionalDefs() const
{
    std::vector<forms::BoundaryFunctional> out;
    out.reserve(functionals_.size());
    for (const auto& entry : functionals_) {
        out.push_back(entry.def);
    }
    return out;
}

} // namespace systems
} // namespace FE
} // namespace svmp
