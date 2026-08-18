/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/BoundaryReductionService.h"

#include "Assembly/FunctionalAssembler.h"
#include "Assembly/CutIntegrationContext.h"
#if defined(FE_HAS_FSILS)
#include "Backends/FSILS/FsilsVector.h"
#endif
#include "Backends/Interfaces/BlockVector.h"
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
#include "Core/MpiCollectiveTrace.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <numeric>
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
MPI_Comm boundaryReductionCommunicator(
    const FESystem& system) noexcept
{
    return system.isSetup()
        ? system.dofHandler().mpiComm()
        : system.activeMpiCommunicator();
}

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

int allreduceSum(int local, MPI_Comm comm)
{
    int global = local;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, comm);
    return global;
}

bool mpiUsesMultipleRanks(MPI_Comm comm)
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        return false;
    }
    int size = 1;
    MPI_Comm_size(comm, &size);
    return size > 1;
}
#endif

void coordinateBoundaryReductionLocalFailure(
    const FESystem& system,
    const std::exception_ptr& local_exception,
    std::string_view phase)
{
    bool any_failed =
        local_exception != nullptr;
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (initialized != 0 && finalized == 0) {
        const auto communicator =
            boundaryReductionCommunicator(system);
        if (communicator != MPI_COMM_NULL &&
            mpiUsesMultipleRanks(communicator)) {
            const int local_ok =
                local_exception == nullptr ? 1 : 0;
            int all_ok = 0;
            const auto sequence =
                debug::nextMpiCollectiveTraceSeq();
            debug::traceMpiCollective(
                "before",
                sequence,
                "BoundaryReductionService::coordinateLocalFailure",
                1,
                MPI_INT,
                MPI_MIN,
                communicator);
            MPI_Allreduce(
                &local_ok,
                &all_ok,
                1,
                MPI_INT,
                MPI_MIN,
                communicator);
            debug::traceMpiCollective(
                "after",
                sequence,
                "BoundaryReductionService::coordinateLocalFailure",
                1,
                MPI_INT,
                MPI_MIN,
                communicator);
            any_failed = all_ok == 0;
        }
    }
#else
    static_cast<void>(system);
#endif
    if (!any_failed) {
        return;
    }
    if (local_exception != nullptr) {
        std::rethrow_exception(local_exception);
    }
    throw InvalidStateException(
        "BoundaryReductionService: another communicator rank "
        "failed local phase '" +
        std::string(phase) + "'");
}

void refreshBoundaryReductionGhostedCoefficients(
    const FESystem& system,
    const SystemStateView& state,
    std::string_view phase)
{
    std::exception_ptr local_exception;
    try {
        const auto refresh =
            [](const backends::GenericVector*
                   vector) {
                if (vector == nullptr) {
                    return;
                }
                auto* mutable_vector =
                    const_cast<
                        backends::GenericVector*>(
                        vector);
                mutable_vector->updateGhosts();
            };
        refresh(state.u_vector);
        refresh(state.u_prev_vector);
        refresh(state.u_prev2_vector);
    } catch (...) {
        local_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system, local_exception, phase);
}

void requireMatchingBoundaryReductionShape(
    const FESystem& system,
    const std::array<std::uint64_t, 3>& local_shape)
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (initialized == 0 || finalized != 0) {
        return;
    }
    const auto communicator =
        boundaryReductionCommunicator(system);
    if (communicator == MPI_COMM_NULL ||
        !mpiUsesMultipleRanks(communicator)) {
        return;
    }
    std::array<std::uint64_t, 3>
        minimum_shape{};
    std::array<std::uint64_t, 3>
        maximum_shape{};
    const auto reduce =
        [&](auto& reduced, MPI_Op operation) {
            const auto sequence =
                debug::nextMpiCollectiveTraceSeq();
            debug::traceMpiCollective(
                "before",
                sequence,
                "BoundaryReductionService::requireMatchingRequest",
                static_cast<int>(local_shape.size()),
                MPI_UINT64_T,
                operation,
                communicator);
            MPI_Allreduce(
                local_shape.data(),
                reduced.data(),
                static_cast<int>(local_shape.size()),
                MPI_UINT64_T,
                operation,
                communicator);
            debug::traceMpiCollective(
                "after",
                sequence,
                "BoundaryReductionService::requireMatchingRequest",
                static_cast<int>(local_shape.size()),
                MPI_UINT64_T,
                operation,
                communicator);
        };
    reduce(minimum_shape, MPI_MIN);
    reduce(maximum_shape, MPI_MAX);
    FE_THROW_IF(
        minimum_shape != maximum_shape,
        InvalidArgumentException,
        "BoundaryReductionService: collective request differs "
        "across communicator ranks");
#else
    static_cast<void>(system);
    static_cast<void>(local_shape);
#endif
}

bool allRanksHaveBoundaryMeasureCacheEntry(
    const FESystem& system,
    bool local_hit)
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (initialized != 0 && finalized == 0) {
        const auto communicator =
            boundaryReductionCommunicator(system);
        if (communicator != MPI_COMM_NULL &&
            mpiUsesMultipleRanks(communicator)) {
            const int local_has =
                local_hit ? 1 : 0;
            int every_rank_has = 0;
            const auto sequence =
                debug::nextMpiCollectiveTraceSeq();
            debug::traceMpiCollective(
                "before",
                sequence,
                "BoundaryReductionService::boundaryMeasureCache",
                1,
                MPI_INT,
                MPI_MIN,
                communicator);
            MPI_Allreduce(
                &local_has,
                &every_rank_has,
                1,
                MPI_INT,
                MPI_MIN,
                communicator);
            debug::traceMpiCollective(
                "after",
                sequence,
                "BoundaryReductionService::boundaryMeasureCache",
                1,
                MPI_INT,
                MPI_MIN,
                communicator);
            return every_rank_has != 0;
        }
    }
#else
    static_cast<void>(system);
#endif
    return local_hit;
}

void requireMatchingBoundaryMeasureValue(
    const FESystem& system,
    Real local_value,
    std::string_view phase)
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (initialized != 0 && finalized == 0) {
        const auto communicator =
            boundaryReductionCommunicator(system);
        if (communicator != MPI_COMM_NULL &&
            mpiUsesMultipleRanks(communicator)) {
            const int local_finite =
                std::isfinite(local_value) ? 1 : 0;
            int all_finite = 0;
            MPI_Allreduce(
                &local_finite,
                &all_finite,
                1,
                MPI_INT,
                MPI_MIN,
                communicator);
            FE_THROW_IF(
                all_finite == 0,
                InvalidStateException,
                "BoundaryReductionService: non-finite "
                "boundary measure during '" +
                    std::string(phase) + "'");

            Real minimum_value = local_value;
            Real maximum_value = local_value;
            const auto reduce =
                [&](Real& reduced,
                    MPI_Op operation) {
                    const auto sequence =
                        debug::
                            nextMpiCollectiveTraceSeq();
                    debug::traceMpiCollective(
                        "before",
                        sequence,
                        "BoundaryReductionService::requireMatchingBoundaryMeasureValue",
                        1,
                        mpiRealType(),
                        operation,
                        communicator);
                    MPI_Allreduce(
                        &local_value,
                        &reduced,
                        1,
                        mpiRealType(),
                        operation,
                        communicator);
                    debug::traceMpiCollective(
                        "after",
                        sequence,
                        "BoundaryReductionService::requireMatchingBoundaryMeasureValue",
                        1,
                        mpiRealType(),
                        operation,
                        communicator);
                };
            reduce(minimum_value, MPI_MIN);
            reduce(maximum_value, MPI_MAX);
            FE_THROW_IF(
                minimum_value != maximum_value,
                InvalidStateException,
                "BoundaryReductionService: cached "
                "boundary measure differs across "
                "communicator ranks");
            return;
        }
    }
#else
    static_cast<void>(system);
#endif
    FE_THROW_IF(
        !std::isfinite(local_value),
        InvalidStateException,
        "BoundaryReductionService: non-finite boundary "
        "measure during '" +
            std::string(phase) + "'");
}

[[nodiscard]] SystemStateView makeFunctionalEvaluationState(
    const SystemStateView& state)
{
    auto out = state;

    // Backend history spans are local/overlap storage and therefore are not
    // indexed by public global DOF. Preserve the established two-history
    // contract by binding u_prev/u_prev2 through their global-indexed views.
    if (!out.u_history.empty() &&
        (state.u_prev_vector != nullptr || state.u_prev2_vector != nullptr)) {
        out.u_history = {};
    }

    return out;
}

void traceSampledVectorDofs(const FESystem& system,
                            const forms::BoundaryFunctional& functional,
                            const backends::GenericVector* vec_ptr)
{
    const char* trace_dofs_env = std::getenv("SVMP_MONO_AUX_TRACE_DOFS");
    if (trace_dofs_env == nullptr || *trace_dofs_env == '\0' || vec_ptr == nullptr) {
        return;
    }

    static thread_local int trace_budget = 64;
    if (trace_budget <= 0) {
        return;
    }

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
    if (trace_dofs.empty()) {
        return;
    }

    auto* vec = const_cast<backends::GenericVector*>(vec_ptr);
    auto view = vec->createAssemblyView();
    if (!view) {
        return;
    }

    std::vector<Real> trace_values(trace_dofs.size(), 0.0);
    view->getVectorEntries(trace_dofs, trace_values);

    int rank = 0;
#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        MPI_Comm_rank(system.dofHandler().mpiComm(), &rank);
    }
#endif

    const auto& constraints = system.constraints();
#if defined(FE_HAS_FSILS)
    const auto* fsils_vec = dynamic_cast<const backends::FsilsVector*>(vec_ptr);
#else
    const void* fsils_vec = nullptr;
#endif
    std::fprintf(stderr,
                 "[BoundaryReductionVectorDofs] rank=%d functional='%s' marker=%d",
                 rank,
                 functional.name.c_str(),
                 functional.boundary_marker);
    for (std::size_t i = 0; i < trace_dofs.size(); ++i) {
        std::fprintf(stderr,
                     " dof=%lld constrained=%d owned=%d value=%.17g",
                     static_cast<long long>(trace_dofs[i]),
                     constraints.isConstrained(trace_dofs[i]) ? 1 : 0,
#if defined(FE_HAS_FSILS)
                     fsils_vec != nullptr && fsils_vec->ownsFeDof(trace_dofs[i]) ? 1 : 0,
#else
                     0,
#endif
                     static_cast<double>(trace_values[i]));
    }
    std::fprintf(stderr, "\n");
    --trace_budget;
}

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

void BoundaryReductionService::
    validateExteriorBoundaryMeasureAgainstCutContext(
        const forms::BoundaryFunctional& functional,
        const assembly::CutIntegrationContext* context,
        bool require_generated_active_context)
{
    if (functional.is_domain_functional) {
        FE_THROW_IF(
            functional.generated_active_boundary_marker.has_value(),
            InvalidArgumentException,
            "BoundaryReductionService: domain functionals cannot select a "
            "generated active boundary");
        return;
    }

    FE_THROW_IF(
        functional.boundary_marker < 0,
        InvalidArgumentException,
        "BoundaryReductionService: exterior-boundary functional requires a "
        "nonnegative physical boundary marker");
    FE_THROW_IF(
        functional.generated_active_boundary_marker.has_value() &&
            *functional.generated_active_boundary_marker < 0,
        InvalidArgumentException,
        "BoundaryReductionService: generated active-boundary marker must be "
        "nonnegative");

    const auto measure = functional.exteriorBoundaryMeasure();
    if (measure.isFullPhysical()) {
        return;
    }
    if (context == nullptr) {
        FE_THROW_IF(
            require_generated_active_context,
            InvalidStateException,
            "BoundaryReductionService: generated active-boundary functional "
            "is pending a cut integration context");
        return;
    }

    const auto* provenance =
        context->findGeneratedActiveBoundaryProvenance(
            measure.generatedActiveBoundaryMarker());
    FE_THROW_IF(
        provenance == nullptr,
        InvalidArgumentException,
        "BoundaryReductionService: generated active-boundary functional "
        "marker has no provenance in the candidate cut context");
    FE_THROW_IF(
        provenance->physicalBoundaryMarker() !=
            measure.physicalBoundaryMarker(),
        InvalidArgumentException,
        "BoundaryReductionService: generated active-boundary functional "
        "physical marker does not match candidate cut-context provenance");
}

void BoundaryReductionService::
    validateExteriorBoundaryMeasuresAgainstCutContext(
        const assembly::CutIntegrationContext* context,
        bool require_generated_active_context) const
{
    for (const auto& entry : functionals_) {
        validateExteriorBoundaryMeasureAgainstCutContext(
            entry.def,
            context,
            require_generated_active_context);
    }
}

void BoundaryReductionService::addBoundaryFunctional(forms::BoundaryFunctional functional)
{
    FE_THROW_IF(functional.name.empty(), InvalidArgumentException,
                "BoundaryReductionService::addBoundaryFunctional: empty name");
    FE_THROW_IF(!functional.integrand.isValid(), InvalidArgumentException,
                "BoundaryReductionService::addBoundaryFunctional: invalid integrand");
    FE_THROW_IF(functional.generated_active_boundary_marker.has_value() &&
                    *functional.generated_active_boundary_marker < 0,
                InvalidArgumentException,
                "BoundaryReductionService::addBoundaryFunctional: "
                "generated_active_boundary_marker must be >= 0");
    FE_THROW_IF(functional.is_domain_functional &&
                    functional.generated_active_boundary_marker.has_value(),
                InvalidArgumentException,
                "BoundaryReductionService::addBoundaryFunctional: domain "
                "functionals cannot use generated_active_boundary_marker");
    FE_THROW_IF(
        functional.is_domain_functional &&
            functional.reduction !=
                forms::BoundaryFunctional::Reduction::Sum,
        InvalidArgumentException,
        "BoundaryReductionService::addBoundaryFunctional: domain "
        "functionals support only Sum reduction");
    validateExteriorBoundaryMeasureAgainstCutContext(
        functional,
        system_.cutIntegrationContext(),
        /*require_generated_active_context=*/false);
    if (functional.generated_active_boundary_marker.has_value()) {
        const auto validate_generated_expression =
            [&](const auto& self,
                const forms::FormExprNode& node) -> void {
                if (const auto field = node.fieldId(); field.has_value()) {
                    FE_THROW_IF(
                        system_.fieldRecord(*field).source_kind !=
                            FieldSourceKind::Unknown,
                        NotImplementedException,
                        "BoundaryReductionService::addBoundaryFunctional: "
                        "generated active-boundary functionals currently "
                        "require unknown-vector field sources");
                }
                FE_THROW_IF(
                    node.type() == forms::FormExprType::PreviousSolutionRef ||
                        node.type() == forms::FormExprType::TimeDerivative ||
                        node.type() ==
                            forms::FormExprType::HistoryWeightedSum ||
                        node.type() ==
                            forms::FormExprType::HistoryConvolution,
                    NotImplementedException,
                    "BoundaryReductionService::addBoundaryFunctional: "
                    "generated active-boundary functionals with "
                    "solution-history operators are unsupported");
                for (const auto* child : node.children()) {
                    if (child != nullptr) {
                        self(self, *child);
                    }
                }
            };
        validate_generated_expression(
            validate_generated_expression, *functional.integrand.node());
    }

    auto it = name_to_functional_.find(functional.name);
    if (it != name_to_functional_.end()) {
        // Duplicate with identical properties is accepted silently.
        const auto& existing = functionals_.at(it->second).def;
        FE_THROW_IF(existing.boundary_marker != functional.boundary_marker, InvalidArgumentException,
                    "BoundaryReductionService::addBoundaryFunctional: name '" + functional.name +
                    "' already registered with different boundary_marker");
        FE_THROW_IF(existing.generated_active_boundary_marker !=
                        functional.generated_active_boundary_marker,
                    InvalidArgumentException,
                    "BoundaryReductionService::addBoundaryFunctional: name '" +
                        functional.name +
                        "' already registered with different "
                        "generated_active_boundary_marker");
        FE_THROW_IF(existing.reduction != functional.reduction, InvalidArgumentException,
                    "BoundaryReductionService::addBoundaryFunctional: name '" + functional.name +
                    "' already registered with different reduction");
        FE_THROW_IF(existing.is_domain_functional !=
                        functional.is_domain_functional,
                    InvalidArgumentException,
                    "BoundaryReductionService::addBoundaryFunctional: name '" +
                        functional.name +
                        "' already registered with different functional domain");
        FE_THROW_IF(existing.region_marker != functional.region_marker,
                    InvalidArgumentException,
                    "BoundaryReductionService::addBoundaryFunctional: name '" +
                        functional.name +
                        "' already registered with different region_marker");
        FE_THROW_IF(existing.integrand.toString() != functional.integrand.toString(), InvalidArgumentException,
                    "BoundaryReductionService::addBoundaryFunctional: name '" + functional.name +
                    "' already registered with different integrand");
        return;
    }

    const auto idx = functionals_.size();
    functionals_.reserve(idx + 1u);
    name_to_functional_.reserve(
        name_to_functional_.size() + 1u);
    const auto [name_entry, inserted] =
        name_to_functional_.emplace(
            functional.name, idx);
    FE_THROW_IF(
        !inserted,
        InvalidStateException,
        "BoundaryReductionService::addBoundaryFunctional: "
        "functional-name index insertion failed");
    try {
        functionals_.push_back(
            CompiledFunctional{
                std::move(functional),
                nullptr});
    } catch (...) {
        name_to_functional_.erase(
            name_entry);
        throw;
    }
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

            if (primary_rec != nullptr) {
                register_field_binding(
                    primary_field_,
                    *primary_rec->space,
                    primary_rec->components);
            }
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

BoundaryReductionService::CompiledFunctional&
BoundaryReductionService::requireCollectiveFunctional(
    std::string_view name,
    CollectiveOperation operation,
    const SystemStateView& state,
    FieldId target_field,
    bool apply_constraints)
{
    CompiledFunctional* entry = nullptr;
    for (auto& candidate : functionals_) {
        if (candidate.def.name == name) {
            entry = &candidate;
            break;
        }
    }
    requireCollectiveRequest(
        operation,
        name,
        entry == nullptr ? nullptr : &entry->def,
        entry != nullptr,
        state,
        target_field,
        apply_constraints);
    FE_THROW_IF(
        entry == nullptr,
        InvalidArgumentException,
        "BoundaryReductionService: unknown functional '" +
            std::string(name) + "'");
    return *entry;
}

void BoundaryReductionService::requireCollectiveRequest(
    CollectiveOperation operation,
    std::string_view name,
    const forms::BoundaryFunctional* functional,
    bool request_valid,
    const SystemStateView& state,
    FieldId target_field,
    bool apply_constraints) const
{
    constexpr std::uint64_t offset =
        14695981039346656037ull;
    constexpr std::uint64_t prime =
        1099511628211ull;
    std::uint64_t digest = offset;
    const auto mix_byte = [&](std::uint8_t value) {
        digest ^= value;
        digest *= prime;
    };
    const auto mix_u64 = [&](std::uint64_t value) {
        for (unsigned int byte = 0u;
             byte < 8u;
             ++byte) {
            mix_byte(static_cast<std::uint8_t>(
                (value >> (byte * 8u)) &
                std::uint64_t{0xffu}));
        }
    };
    const auto mix_signed = [&](auto value) {
        mix_u64(static_cast<std::uint64_t>(
            static_cast<std::int64_t>(value)));
    };
    const auto mix_string = [&](std::string_view value) {
        mix_u64(static_cast<std::uint64_t>(
            value.size()));
        for (const unsigned char character :
             value) {
            mix_byte(character);
        }
    };

    mix_byte(static_cast<std::uint8_t>(
        operation));
    mix_signed(primary_field_);
    mix_string(name);
    mix_byte(request_valid
                 ? std::uint8_t{1u}
                 : std::uint8_t{0u});
    mix_byte(system_.isSetup()
                 ? std::uint8_t{1u}
                 : std::uint8_t{0u});
    mix_signed(target_field);
    mix_byte(apply_constraints
                 ? std::uint8_t{1u}
                 : std::uint8_t{0u});
    mix_byte(state.u_vector != nullptr
                 ? std::uint8_t{1u}
                 : std::uint8_t{0u});
    mix_byte(state.u_prev_vector != nullptr
                 ? std::uint8_t{1u}
                 : std::uint8_t{0u});
    mix_byte(state.u_prev2_vector != nullptr
                 ? std::uint8_t{1u}
                 : std::uint8_t{0u});
    mix_byte(!state.u_history.empty()
                 ? std::uint8_t{1u}
                 : std::uint8_t{0u});
    const auto mix_time_value =
        [&](double value) {
            const double canonical =
                value == 0.0 ? 0.0 : value;
            mix_u64(std::bit_cast<
                    std::uint64_t>(
                canonical));
        };
    mix_time_value(state.time);
    mix_time_value(state.dt);
    mix_time_value(state.effective_dt);
    mix_time_value(state.dt_prev);
    mix_byte(state.time_integration != nullptr
                 ? std::uint8_t{1u}
                 : std::uint8_t{0u});
    if (state.time_integration != nullptr) {
        const auto& context = *state.time_integration;
        const auto mix_stencil =
            [&](const std::optional<
                    assembly::TimeDerivativeStencil>& stencil) {
                mix_byte(stencil.has_value()
                             ? std::uint8_t{1u}
                             : std::uint8_t{0u});
                if (!stencil.has_value()) {
                    return;
                }
                mix_signed(stencil->order);
                mix_u64(static_cast<std::uint64_t>(
                    stencil->a.size()));
                for (const Real coefficient : stencil->a) {
                    mix_time_value(coefficient);
                }
            };
        mix_string(context.integrator_name);
        mix_stencil(context.dt1);
        mix_stencil(context.dt2);
        mix_u64(static_cast<std::uint64_t>(
            context.dt_extra.size()));
        for (const auto& stencil : context.dt_extra) {
            mix_stencil(stencil);
        }
        mix_time_value(context.time_derivative_term_weight);
        mix_time_value(context.non_time_derivative_term_weight);
        mix_time_value(context.dt1_term_weight);
        mix_time_value(context.dt2_term_weight);
        mix_u64(static_cast<std::uint64_t>(
            context.dt_extra_term_weight.size()));
        for (const Real weight :
             context.dt_extra_term_weight) {
            mix_time_value(weight);
        }
    }
    const auto mix_vector_shape =
        [&](const backends::GenericVector* vector,
            std::span<const Real> fallback) {
            if (vector == nullptr) {
                mix_byte(std::uint8_t{0xffu});
                mix_signed(static_cast<GlobalIndex>(
                    fallback.size()));
                return;
            }

            const auto mix_backend_layout =
                [&](auto&& self,
                    const backends::GenericVector& current)
                    -> void {
                    mix_byte(static_cast<std::uint8_t>(
                        current.backendKind()));
                    mix_signed(current.size());
                    mix_byte(
                        current.ghostUpdateRequiresCollectiveParticipation()
                            ? std::uint8_t{1u}
                            : std::uint8_t{0u});
                    const auto* blocks = dynamic_cast<
                        const backends::BlockVector*>(&current);
                    mix_byte(blocks != nullptr
                                 ? std::uint8_t{1u}
                                 : std::uint8_t{0u});
                    if (blocks == nullptr) {
                        return;
                    }
                    mix_u64(static_cast<std::uint64_t>(
                        blocks->numBlocks()));
                    for (std::size_t block = 0u;
                         block < blocks->numBlocks();
                         ++block) {
                        self(self, blocks->block(block));
                    }
                };
            mix_backend_layout(
                mix_backend_layout, *vector);
        };
    mix_vector_shape(
        state.u_vector, state.u);
    mix_vector_shape(
        state.u_prev_vector, state.u_prev);
    mix_vector_shape(
        state.u_prev2_vector, state.u_prev2);
    const bool dense_history_is_used =
        state.u_prev_vector == nullptr &&
        state.u_prev2_vector == nullptr;
    mix_u64(
        dense_history_is_used
            ? static_cast<std::uint64_t>(
                  state.u_history.size())
            : std::uint64_t{0u});
    if (dense_history_is_used) {
        for (const auto history :
             state.u_history) {
            mix_u64(
                static_cast<std::uint64_t>(
                    history.size()));
        }
    }
    mix_u64(static_cast<std::uint64_t>(
        state.dt_history.size()));
    for (const double history_dt :
         state.dt_history) {
        mix_time_value(history_dt);
    }
    mix_u64(static_cast<std::uint64_t>(
        functionals_.size()));
    // Mesh revision epochs are rank-local cache-invalidation metadata on a
    // distributed partition, not part of the replicated logical request.
    // boundaryMeasurePreflighted() retains them in each rank's local cache key
    // and coordinates cache-hit/recompute decisions separately.
    if (functional != nullptr) {
        mix_string(functional->name);
        mix_byte(static_cast<std::uint8_t>(
            functional->reduction));
        mix_byte(
            functional->is_domain_functional
                ? std::uint8_t{1u}
                : std::uint8_t{0u});
        mix_signed(functional->region_marker);
        mix_signed(functional->boundary_marker);
        mix_signed(
            functional
                ->generated_active_boundary_marker
                .value_or(-1));
    }

    requireMatchingBoundaryReductionShape(
        system_,
        std::array<std::uint64_t, 3>{{
            request_valid
                ? std::uint64_t{1u}
                : std::uint64_t{0u},
            static_cast<std::uint64_t>(
                name.size()),
            digest,
        }});
    FE_THROW_IF(
        !request_valid,
        InvalidArgumentException,
        "BoundaryReductionService: collective request is "
        "not valid on this rank");
    const auto finite_stencil =
        [](const std::optional<
               assembly::TimeDerivativeStencil>& stencil) {
            return !stencil.has_value() ||
                   std::all_of(
                       stencil->a.begin(),
                       stencil->a.end(),
                       [](Real value) {
                           return std::isfinite(value);
                       });
        };
    const bool finite_time_integration =
        state.time_integration == nullptr ||
        (finite_stencil(state.time_integration->dt1) &&
         finite_stencil(state.time_integration->dt2) &&
         std::all_of(
             state.time_integration->dt_extra.begin(),
             state.time_integration->dt_extra.end(),
             finite_stencil) &&
         std::isfinite(
             state.time_integration
                 ->time_derivative_term_weight) &&
         std::isfinite(
             state.time_integration
                 ->non_time_derivative_term_weight) &&
         std::isfinite(
             state.time_integration->dt1_term_weight) &&
         std::isfinite(
             state.time_integration->dt2_term_weight) &&
         std::all_of(
             state.time_integration
                 ->dt_extra_term_weight.begin(),
             state.time_integration
                 ->dt_extra_term_weight.end(),
             [](Real value) {
                 return std::isfinite(value);
             }));
    FE_THROW_IF(
        !std::isfinite(state.time) ||
            !std::isfinite(state.dt) ||
            !std::isfinite(state.effective_dt) ||
            !std::isfinite(state.dt_prev) ||
            !finite_time_integration ||
            std::any_of(
                state.dt_history.begin(),
                state.dt_history.end(),
                [](double value) {
                    return !std::isfinite(value);
                }),
        InvalidArgumentException,
        "BoundaryReductionService: state time, dt, and time-integration "
        "coefficients must be finite");
    const auto has_valid_size =
        [](const backends::GenericVector* vector) {
            return vector == nullptr || vector->size() >= 0;
        };
    FE_THROW_IF(
        !has_valid_size(state.u_vector) ||
            !has_valid_size(state.u_prev_vector) ||
            !has_valid_size(state.u_prev2_vector),
        InvalidArgumentException,
        "BoundaryReductionService: backend vector size must be nonnegative");
    FE_THROW_IF(
        !system_.isSetup(),
        InvalidStateException,
        "BoundaryReductionService: system.setup() not called");
}

Real BoundaryReductionService::evaluateFunctionalEntryPreflighted(
    CompiledFunctional& entry,
    const SystemStateView& state)
{
    std::exception_ptr local_compile_exception;
    try {
        compileFunctionalIfNeeded(entry);
        FE_CHECK_NOT_NULL(
            entry.kernel.get(),
            "BoundaryReductionService::evaluateFunctional: kernel");
    } catch (...) {
        local_compile_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_compile_exception,
        "functional_compile");

    Real raw = Real{0.0};
    assembly::FunctionalResult local_result;
    std::exception_ptr local_assembly_exception;

    std::exception_ptr local_ghost_exception;
    try {
        const auto refreshGhostedCoefficients =
            [](const backends::GenericVector* vec_ptr) {
                if (vec_ptr == nullptr) {
                    return;
                }
                // Explicit sampled reductions read FE coefficients through
                // backend vector views. Distributed views must see fresh
                // owner-to-ghost copies.
                auto* vec =
                    const_cast<backends::GenericVector*>(
                        vec_ptr);
                vec->updateGhosts();
            };
        refreshGhostedCoefficients(
            state.u_vector);
        refreshGhostedCoefficients(
            state.u_prev_vector);
        refreshGhostedCoefficients(
            state.u_prev2_vector);
    } catch (...) {
        local_ghost_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_ghost_exception,
        "functional_ghost_refresh");

    std::exception_ptr local_trace_exception;
    try {
        traceSampledVectorDofs(
            system_, entry.def, state.u_vector);
    } catch (...) {
        local_trace_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_trace_exception,
        "functional_trace_preflight");

    auto eval_state = makeFunctionalEvaluationState(state);

    try {
    assembly::FunctionalAssembler assembler;
    configureAssembler(assembler, eval_state, /*bind_solution=*/true);

    // Bind solution view for MPI-aware DOF access.
    std::unique_ptr<assembly::GlobalSystemView> solution_view;
    if (eval_state.u_vector != nullptr) {
        auto* vec = const_cast<backends::GenericVector*>(eval_state.u_vector);
        solution_view = vec->createGhostedReadView();
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
    if (eval_state.u_prev_vector != nullptr) {
        auto* vec = const_cast<backends::GenericVector*>(eval_state.u_prev_vector);
        prev_solution_view = vec->createGhostedReadView();
        assembler.setPreviousSolutionView(prev_solution_view.get());
    }
    if (eval_state.u_prev2_vector != nullptr) {
        auto* vec = const_cast<backends::GenericVector*>(eval_state.u_prev2_vector);
        prev2_solution_view = vec->createGhostedReadView();
        assembler.setPreviousSolution2View(prev2_solution_view.get());
    }

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
    } else if (entry.def.generated_active_boundary_marker.has_value()) {
        const auto* cut_context = system_.cutIntegrationContext();
        FE_THROW_IF(cut_context == nullptr, InvalidStateException,
                    "BoundaryReductionService: generated active-boundary "
                    "functional requires a cut integration context");
        const int marker = *entry.def.generated_active_boundary_marker;
        FE_THROW_IF(!cut_context->hasGeneratedActiveBoundaryMarker(marker),
                    InvalidArgumentException,
                    "BoundaryReductionService: marker " +
                        std::to_string(marker) +
                        " is not a generated active-boundary marker");
        raw = assembler.assembleCutInterfaceScalar(
            *entry.kernel, *cut_context, marker);
    } else {
        raw = assembler.assembleBoundaryScalar(*entry.kernel, entry.def.boundary_marker);
    }

    local_result = assembler.getLastResult();
    } catch (...) {
        local_assembly_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_assembly_exception,
        "functional_local_assembly");

    int assembly_failure_count = local_result.success ? 0 : 1;
#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        const auto comm = system_.dofHandler().mpiComm();
        assembly_failure_count =
            allreduceSum(assembly_failure_count, comm);
    }
#endif
    FE_THROW_IF(
        assembly_failure_count != 0,
        InvalidStateException,
        "BoundaryReductionService: functional '" + entry.def.name +
            "' assembly failed" +
            (local_result.error_message.empty()
                 ? std::string(" on at least one rank")
                 : std::string(": ") + local_result.error_message));
#if FE_HAS_MPI
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
            const Real area = boundaryMeasurePreflighted(
                entry.def, state);
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
    auto& entry = requireCollectiveFunctional(
        name,
        CollectiveOperation::Value,
        state);
    system_.requireCurrentBoundaryReductionExteriorMeasures();
    return evaluateFunctionalEntryPreflighted(
        entry, state);
}

Real BoundaryReductionService::evaluateFunctionalOverCells(
    std::string_view name,
    std::span<const GlobalIndex> cell_ids,
    const SystemStateView& state)
{
    auto& entry = requireCollectiveFunctional(
        name,
        CollectiveOperation::ValueOverCells,
        state);
    system_.requireCurrentBoundaryReductionExteriorMeasures();
    FE_THROW_IF(!entry.def.is_domain_functional, InvalidArgumentException,
                "BoundaryReductionService::evaluateFunctionalOverCells: functional '" +
                std::string(name) + "' is not a domain functional");
    FE_THROW_IF(
        entry.def.reduction !=
            forms::BoundaryFunctional::Reduction::Sum,
        NotImplementedException,
        "BoundaryReductionService::evaluateFunctionalOverCells: "
        "only Sum reduction is supported");

    std::exception_ptr local_compile_exception;
    try {
        compileFunctionalIfNeeded(entry);
        FE_CHECK_NOT_NULL(
            entry.kernel.get(),
            "BoundaryReductionService::evaluateFunctionalOverCells: kernel");
    } catch (...) {
        local_compile_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_compile_exception,
        "cell_functional_compile");

    std::exception_ptr local_ghost_exception;
    try {
        const auto refreshGhostedCoefficients =
            [](const backends::GenericVector* vec_ptr) {
                if (vec_ptr == nullptr) {
                    return;
                }
                auto* vec =
                    const_cast<backends::GenericVector*>(
                        vec_ptr);
                vec->updateGhosts();
            };
        refreshGhostedCoefficients(
            state.u_vector);
        refreshGhostedCoefficients(
            state.u_prev_vector);
        refreshGhostedCoefficients(
            state.u_prev2_vector);
    } catch (...) {
        local_ghost_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_ghost_exception,
        "cell_functional_ghost_refresh");

    auto eval_state = makeFunctionalEvaluationState(state);

    Real raw = Real{0.0};
    assembly::FunctionalResult local_result;
    std::exception_ptr local_assembly_exception;
    try {
        assembly::FunctionalAssembler assembler;
        configureAssembler(
            assembler,
            eval_state,
            /*bind_solution=*/true);

        std::unique_ptr<
            assembly::GlobalSystemView>
            solution_view;
        if (eval_state.u_vector != nullptr) {
            auto* vec = const_cast<
                backends::GenericVector*>(
                    eval_state.u_vector);
            solution_view =
                vec->createGhostedReadView();
            assembler.setSolutionView(
                solution_view.get());
        }

        std::unique_ptr<
            assembly::GlobalSystemView>
            prev_solution_view;
        std::unique_ptr<
            assembly::GlobalSystemView>
            prev2_solution_view;
        if (eval_state.u_prev_vector !=
            nullptr) {
            auto* vec = const_cast<
                backends::GenericVector*>(
                    eval_state.u_prev_vector);
            prev_solution_view =
                vec->createGhostedReadView();
            assembler.setPreviousSolutionView(
                prev_solution_view.get());
        }
        if (eval_state.u_prev2_vector !=
            nullptr) {
            auto* vec = const_cast<
                backends::GenericVector*>(
                    eval_state.u_prev2_vector);
            prev2_solution_view =
                vec->createGhostedReadView();
            assembler.setPreviousSolution2View(
                prev2_solution_view.get());
        }

        raw = assembler.assembleScalarOverCells(
            *entry.kernel, cell_ids);
        local_result =
            assembler.getLastResult();
    } catch (...) {
        local_assembly_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_assembly_exception,
        "cell_functional_local_assembly");

    int assembly_failure_count =
        local_result.success ? 0 : 1;
#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        assembly_failure_count = allreduceSum(
            assembly_failure_count,
            system_.dofHandler().mpiComm());
        raw = allreduceSum(raw, system_.dofHandler().mpiComm());
    }
#endif
    FE_THROW_IF(
        assembly_failure_count != 0,
        InvalidStateException,
        "BoundaryReductionService::evaluateFunctionalOverCells: "
        "assembly failed" +
            (local_result.error_message.empty()
                 ? std::string(" on at least one rank")
                 : std::string(": ") +
                       local_result.error_message));

    if (entry.def.reduction == forms::BoundaryFunctional::Reduction::Sum) {
        return raw;
    }

    FE_THROW(NotImplementedException,
             "BoundaryReductionService::evaluateFunctionalOverCells: only Sum reduction is supported");
}

std::vector<Real> BoundaryReductionService::evaluateAll(const SystemStateView& state)
{
    requireCollectiveRequest(
        CollectiveOperation::EvaluateAll,
        {},
        nullptr,
        /*request_valid=*/true,
        state);
    system_.requireCurrentBoundaryReductionExteriorMeasures();

    std::vector<std::size_t> order;
    std::vector<Real> results;
    std::exception_ptr local_schedule_exception;
    try {
        order.resize(functionals_.size());
        std::iota(
            order.begin(), order.end(),
            std::size_t{0u});
        std::sort(
            order.begin(),
            order.end(),
            [&](std::size_t lhs,
                std::size_t rhs) {
                return functionals_[lhs]
                           .def.name <
                       functionals_[rhs]
                           .def.name;
            });
        results.assign(
            functionals_.size(),
            Real{0.0});
    } catch (...) {
        local_schedule_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_schedule_exception,
        "evaluate_all_schedule");
    for (const auto index : order) {
        results[index] =
            evaluateFunctionalEntryPreflighted(
                functionals_[index],
                state);
    }
    return results;
}

Real BoundaryReductionService::boundaryMeasure(
    int boundary_marker,
    const SystemStateView& state)
{
    forms::BoundaryFunctional functional;
    functional.boundary_marker = boundary_marker;
    return boundaryMeasure(functional, state);
}

Real BoundaryReductionService::boundaryMeasure(
    const forms::BoundaryFunctional& functional,
    const SystemStateView& state)
{
    requireCollectiveRequest(
        CollectiveOperation::Measure,
        functional.name,
        &functional,
        /*request_valid=*/true,
        state);
    FE_THROW_IF(
        functional.is_domain_functional,
        InvalidArgumentException,
        "BoundaryReductionService::boundaryMeasure: domain "
        "functionals do not have a boundary measure");
    system_.requireCurrentBoundaryReductionExteriorMeasures();
    system_.requireBoundaryReductionExteriorMeasure(
        functional);
    return boundaryMeasurePreflighted(
        functional, state);
}

Real BoundaryReductionService::boundaryMeasurePreflighted(
    const forms::BoundaryFunctional& functional,
    const SystemStateView& state)
{
    std::array<std::uint64_t, 7>
        current_mesh_revision{};
    bool mesh_revisions_available = false;
    std::exception_ptr local_revision_exception;
    try {
        const auto& mesh = system_.meshAccess();
        mesh_revisions_available =
            mesh.revisionTrackingAvailable();
        if (mesh_revisions_available) {
            current_mesh_revision = {{
                mesh.geometryRevision(),
                mesh.topologyRevision(),
                mesh.ownershipRevision(),
                mesh.numberingRevision(),
                mesh.labelRevision(),
                mesh.activeConfigurationEpoch(),
                mesh.coordinateConfigurationKey(),
            }};
        }
    } catch (...) {
        local_revision_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_revision_exception,
        "boundary_measure_revision");

    if (!functional.generated_active_boundary_marker.has_value()) {
        const auto cached = boundary_measure_cache_.find(
            functional.boundary_marker);
        const bool local_cache_hit =
            mesh_revisions_available &&
            cached != boundary_measure_cache_.end() &&
            cached->second.mesh_revision == current_mesh_revision;
        // Revision-tracking availability is itself allowed to be rank-local.
        // Every rank must nevertheless enter the same cache-hit consensus;
        // an unavailable rank contributes a miss and forces recomputation.
        if (allRanksHaveBoundaryMeasureCacheEntry(
                system_,
                local_cache_hit)) {
            FE_THROW_IF(
                !local_cache_hit,
                InvalidStateException,
                "BoundaryReductionService: boundary-measure "
                "cache agreement failed locally");
            requireMatchingBoundaryMeasureValue(
                system_,
                cached->second.value,
                "cache_hit");
            return cached->second.value;
        }
    }

    Real area = Real{0.0};
    assembly::FunctionalResult local_result;
    std::exception_ptr local_assembly_exception;
    try {
        assembly::FunctionalAssembler assembler;
        configureAssembler(
            assembler,
            state,
            /*bind_solution=*/false);

        BoundaryMeasureKernel measure_kernel;
        if (functional
                .generated_active_boundary_marker
                .has_value()) {
            const auto* cut_context =
                system_.cutIntegrationContext();
            FE_THROW_IF(
                cut_context == nullptr,
                InvalidStateException,
                "BoundaryReductionService::boundaryMeasure: generated "
                "active boundary requires a cut integration context");
            const int marker =
                *functional
                     .generated_active_boundary_marker;
            FE_THROW_IF(
                !cut_context
                     ->hasGeneratedActiveBoundaryMarker(
                         marker),
                InvalidArgumentException,
                "BoundaryReductionService::boundaryMeasure: marker " +
                    std::to_string(marker) +
                    " is not a generated active-boundary marker");
            area =
                assembler
                    .assembleCutInterfaceScalar(
                        measure_kernel,
                        *cut_context,
                        marker);
        } else {
            area =
                assembler.assembleBoundaryScalar(
                    measure_kernel,
                    functional.boundary_marker);
        }

        local_result =
            assembler.getLastResult();
    } catch (...) {
        local_assembly_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_assembly_exception,
        "boundary_measure_local_assembly");
    int assembly_failure_count = local_result.success ? 0 : 1;
#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        assembly_failure_count = allreduceSum(
            assembly_failure_count, system_.dofHandler().mpiComm());
    }
#endif
    FE_THROW_IF(
        assembly_failure_count != 0,
        InvalidStateException,
        "BoundaryReductionService::boundaryMeasure: assembly failed" +
            (local_result.error_message.empty()
                 ? std::string(" on at least one rank")
                 : std::string(": ") + local_result.error_message));
#if FE_HAS_MPI
    if (mpi_initialized) {
        area = allreduceSum(area, system_.dofHandler().mpiComm());
    }
#endif
    requireMatchingBoundaryMeasureValue(
        system_, area, "assembly");

    std::exception_ptr local_cache_publication_exception;
    if (!functional.generated_active_boundary_marker.has_value() &&
        mesh_revisions_available) {
        try {
            boundary_measure_cache_.insert_or_assign(
                functional.boundary_marker,
                BoundaryMeasureCacheEntry{
                    .value = area,
                    .mesh_revision =
                        current_mesh_revision,
                });
        } catch (...) {
            local_cache_publication_exception =
                std::current_exception();
        }
    }
    if (!functional.generated_active_boundary_marker.has_value()) {
        // Cache publication is optional on ranks without revision metadata,
        // but failure coordination must remain in the same collective order
        // on every rank serving the physical-boundary request.
        coordinateBoundaryReductionLocalFailure(
            system_,
            local_cache_publication_exception,
            "boundary_measure_cache_publication");
    }
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
    auto& entry = requireCollectiveFunctional(
        name,
        CollectiveOperation::Gradient,
        state,
        target_field,
        apply_constraints);
    system_.requireCurrentBoundaryReductionExteriorMeasures();
    FE_THROW_IF(
        entry.def.reduction ==
                forms::BoundaryFunctional::Reduction::Max ||
            entry.def.reduction ==
                forms::BoundaryFunctional::Reduction::Min,
        NotImplementedException,
        "BoundaryReductionService: Max/Min gradient "
        "reductions are not implemented");

    // Geometry-only functionals have no field dependence.
    if (target_field == GEOMETRY_FIELD_ID) {
        return {};
    }
    refreshBoundaryReductionGhostedCoefficients(
        system_,
        state,
        "functional_gradient_ghost_refresh");

    std::optional<forms::FormExpr>
        integrand_trial;
    std::exception_ptr local_gradient_exception;
    try {
        compileFunctionalIfNeeded(entry);
        const auto& rec =
            system_.fieldRecord(target_field);
        FE_CHECK_NOT_NULL(
            rec.space.get(),
            "BoundaryReductionService::evaluateFunctionalGradient: field space");

        const auto trial =
            forms::FormExpr::trialFunction(
                *rec.space, "u");
        integrand_trial =
            entry.def.integrand.transformNodes(
                [&](const forms::FormExprNode& n)
                    -> std::optional<
                        forms::FormExpr> {
                    if (n.type() !=
                            forms::FormExprType::
                                DiscreteField &&
                        n.type() !=
                            forms::FormExprType::
                                StateField) {
                        return std::nullopt;
                    }
                    const auto fid =
                        n.fieldId();
                    if (!fid ||
                        *fid != target_field) {
                        return std::nullopt;
                    }
                    return trial;
                });
    } catch (...) {
        local_gradient_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_gradient_exception,
        "functional_gradient_preflight");
    FE_THROW_IF(
        !integrand_trial.has_value(),
        InvalidStateException,
        "BoundaryReductionService: functional gradient "
        "preflight produced no integrand");

    // Symbolic gradient via BoundaryFunctionalGradientKernel + GradAccumulator.
    const int region_marker =
        entry.def.is_domain_functional ? entry.def.region_marker : -1;
    std::vector<SensitivityEntry>
        grad_entries;
    std::exception_ptr
        local_assembly_exception;
    try {
        grad_entries =
            system_.assembleBoundaryGradient(
                target_field,
                *integrand_trial,
                entry.def.boundary_marker,
                state,
                apply_constraints,
                region_marker,
                {},
                entry.def
                    .generated_active_boundary_marker);
    } catch (...) {
        local_assembly_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_assembly_exception,
        "functional_gradient_local_assembly");

    // Apply reduction: for Average, divide by boundary measure.
    if (entry.def.reduction == forms::BoundaryFunctional::Reduction::Average) {
        const Real measure =
            boundaryMeasurePreflighted(
                entry.def, state);
        FE_THROW_IF(std::abs(measure) < 1e-14, InvalidArgumentException,
                    "BoundaryReductionService: boundary measure is near zero "
                    "for Average reduction gradient");
        for (auto& se : grad_entries) se.value /= measure;
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
    auto& entry = requireCollectiveFunctional(
        name,
        CollectiveOperation::GradientOverCells,
        state,
        target_field,
        apply_constraints);
    system_.requireCurrentBoundaryReductionExteriorMeasures();
    FE_THROW_IF(!entry.def.is_domain_functional, InvalidArgumentException,
                "BoundaryReductionService::evaluateFunctionalGradientOverCells: "
                "functional '" + std::string(name) +
                "' is not a domain functional");
    FE_THROW_IF(
        entry.def.reduction !=
            forms::BoundaryFunctional::Reduction::Sum,
        NotImplementedException,
        "BoundaryReductionService::evaluateFunctionalGradientOverCells: "
        "only Sum reduction is supported");
    if (target_field == GEOMETRY_FIELD_ID) {
        return {};
    }
    refreshBoundaryReductionGhostedCoefficients(
        system_,
        state,
        "cell_functional_gradient_ghost_refresh");

    std::optional<forms::FormExpr>
        integrand_trial;
    std::exception_ptr local_gradient_exception;
    try {
        compileFunctionalIfNeeded(entry);
        const auto& rec =
            system_.fieldRecord(target_field);
        FE_CHECK_NOT_NULL(
            rec.space.get(),
            "BoundaryReductionService::evaluateFunctionalGradientOverCells: "
            "field space");

        const auto trial =
            forms::FormExpr::trialFunction(
                *rec.space, "u");
        integrand_trial =
            entry.def.integrand.transformNodes(
                [&](const forms::FormExprNode& n)
                    -> std::optional<
                        forms::FormExpr> {
                    if (n.type() !=
                            forms::FormExprType::
                                DiscreteField &&
                        n.type() !=
                            forms::FormExprType::
                                StateField) {
                        return std::nullopt;
                    }
                    const auto fid =
                        n.fieldId();
                    if (!fid ||
                        *fid != target_field) {
                        return std::nullopt;
                    }
                    return trial;
                });
    } catch (...) {
        local_gradient_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_gradient_exception,
        "cell_functional_gradient_preflight");
    FE_THROW_IF(
        !integrand_trial.has_value(),
        InvalidStateException,
        "BoundaryReductionService: cell functional "
        "gradient preflight produced no integrand");

    std::vector<SensitivityEntry> result;
    std::exception_ptr
        local_assembly_exception;
    try {
        result =
            system_.assembleBoundaryGradient(
                target_field,
                *integrand_trial,
                entry.def.boundary_marker,
                state,
                apply_constraints,
                /*region_marker=*/-1,
                cell_ids,
                std::nullopt,
                /*explicit_cell_filter=*/true);
    } catch (...) {
        local_assembly_exception =
            std::current_exception();
    }
    coordinateBoundaryReductionLocalFailure(
        system_,
        local_assembly_exception,
        "cell_functional_gradient_local_assembly");
    return result;
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
