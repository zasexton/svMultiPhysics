/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FESYSTEM_H
#define SVMP_FE_SYSTEMS_FESYSTEM_H

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Forms/BoundaryFunctional.h"
#include "Systems/AuxiliaryBindings.h"
#include "Systems/AuxiliaryInputRegistry.h"
#include "Systems/AuxiliaryStateTypes.h"
#include "Systems/FEQuantityRegistry.h"
#include "Systems/BoundaryReductionService.h"
#include "Systems/FieldRegistry.h"
#include "Systems/GlobalKernel.h"
#include "Systems/GlobalKernelStateProvider.h"
#include "Systems/OperatorRegistry.h"
#include "Systems/ParameterRegistry.h"
#include "Systems/SearchAccess.h"
#include "Systems/SystemConstraint.h"
#include "Systems/SystemState.h"
#include "Systems/SystemSetup.h"

#include "Assembly/Assembler.h"

#include "Backends/Interfaces/LinearSolver.h"

#include "Constraints/AffineConstraints.h"
#include "Constraints/Constraint.h"
#include "Constraints/GaugeRegistry.h"

#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/FormulationRecord.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/TopologyAnalysisContext.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/ConstraintAnalysisSummary.h"
#include "Analysis/InterfaceTopologyContext.h"

#include "Dofs/DofHandler.h"
#include "Dofs/FieldDofMap.h"
#include "Dofs/BlockDofMap.h"

#include "Sparsity/SparsityPattern.h"
#include "Sparsity/SparsityBuilder.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <array>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
namespace svmp {
class InterfaceMesh;
}
#endif

namespace svmp {
namespace FE {

namespace sparsity {
class DistributedSparsityPattern;
} // namespace sparsity

namespace assembly {
class GlobalSystemView;
struct MatrixFreeOptions;
class IMatrixFreeKernel;
class MatrixFreeOperator;
class FunctionalKernel;
}

namespace backends {
struct DofPermutation;
} // namespace backends

namespace systems {

using BoundaryId = int;
using InterfaceId = int;
class OperatorBackends;
class BoundaryReductionService;
class CoupledBoundaryManager;
class AuxiliaryStateManager;
class AuxiliaryOperatorRegistry;
class AuxiliaryInputRegistry;
class AuxiliaryDeployedInstance;
class AuxiliaryInputHandle;
class AuxiliaryInstanceHandle;
class AuxiliaryStateModel;
class AuxiliaryMultirateScheduler;
class AuxiliaryStateStepper;
class AuxiliaryDerivativeProvider;
struct MixedSystemLayout;

struct SetupOptions {
    dofs::DofDistributionOptions dof_options{};
    assembly::AssemblyOptions assembly_options{};
    sparsity::SparsityBuildOptions sparsity_options{};

    std::string assembler_name{"StandardAssembler"};

    sparsity::CouplingMode coupling_mode{sparsity::CouplingMode::Full};
    std::vector<sparsity::FieldCoupling> custom_couplings{};

    bool use_constraints_in_assembly{true};

    // Iterative-solver leverage (explicit opt-in): auto-register eligible matrix-free operators.
    bool auto_register_matrix_free{false};

    /// When true, print a detailed GaugeRegistry diagnostic report to stderr
    /// after nullspace resolution during setup().  Useful for debugging
    /// nullspace detection and enforcement decisions.
    bool gauge_diagnostics{false};
};

struct AssemblyRequest {
    OperatorTag op;
    bool want_matrix{false};
    bool want_vector{false};
    bool zero_outputs{true};
    bool assemble_boundary_terms{true};
    bool assemble_interior_face_terms{true};
    bool assemble_interface_face_terms{true};
    bool assemble_global_terms{true};

    /// When true, constrained assembly distributes matrix and vector independently
    /// (suppressing the -K*g Dirichlet inhomogeneity correction that joint distribution
    /// adds).  Set to true for nonlinear Newton solves where the residual R(u) is already
    /// evaluated at the constrained state.
    bool suppress_constraint_inhomogeneity{false};

    /// When true, EachNonlinearIteration auxiliary inputs are refreshed.
    /// Set to true on each Newton iteration within a time step.
    bool is_nonlinear_iteration{false};
};

class FESystem {
public:
    explicit FESystem(std::shared_ptr<const assembly::IMeshAccess> mesh_access);
    ~FESystem();

    FESystem(FESystem&&) noexcept;
    FESystem& operator=(FESystem&&) noexcept;

    FESystem(const FESystem&) = delete;
    FESystem& operator=(const FESystem&) = delete;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    explicit FESystem(std::shared_ptr<const svmp::Mesh> mesh,
                      svmp::Configuration coord_cfg = svmp::Configuration::Reference);
#endif

    // ---- Definition phase ----
    FieldId addField(FieldSpec spec);
    void addConstraint(std::unique_ptr<constraints::Constraint> c);
    void addSystemConstraint(std::unique_ptr<ISystemConstraint> c);

    void addOperator(OperatorTag name);

    /// @name Kernel registration (internal — do not use in physics modules)
    ///
    /// These methods are called by FormsInstaller internally. Physics modules
    /// should use the public FormsInstaller API instead:
    ///   - installFormulation()    for residual physics
    ///   - installMixedBilinear()  for mixed bilinear operators
    ///   - installMixedLinear()    for mixed linear operators
    /// @{

    void addCellKernel(OperatorTag op, FieldId field,
                       std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addCellKernel(OperatorTag op, FieldId test_field, FieldId trial_field,
                       std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addBoundaryKernel(OperatorTag op, BoundaryId boundary, FieldId field,
                           std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addBoundaryKernel(OperatorTag op, BoundaryId boundary, FieldId test_field, FieldId trial_field,
                           std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addInteriorFaceKernel(OperatorTag op, FieldId field,
                               std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addInteriorFaceKernel(OperatorTag op, FieldId test_field, FieldId trial_field,
                               std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addInterfaceFaceKernel(OperatorTag op, InterfaceId interface_marker, FieldId field,
                                std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addInterfaceFaceKernel(OperatorTag op, InterfaceId interface_marker, FieldId test_field, FieldId trial_field,
                                std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addGlobalKernel(OperatorTag op,
                         std::shared_ptr<GlobalKernel> kernel);

    /// @}

    // ---- Optional operator backends (Milestone 5) ----
    void addMatrixFreeKernel(OperatorTag op,
                             std::shared_ptr<assembly::IMatrixFreeKernel> kernel);
    void addMatrixFreeKernel(OperatorTag op,
                             std::shared_ptr<assembly::IMatrixFreeKernel> kernel,
                             const assembly::MatrixFreeOptions& options);
    [[nodiscard]] std::shared_ptr<assembly::MatrixFreeOperator>
    matrixFreeOperator(const OperatorTag& op) const;

    void addFunctionalKernel(std::string tag,
                             std::shared_ptr<assembly::FunctionalKernel> kernel);
    [[nodiscard]] Real evaluateFunctional(const std::string& tag,
                                          const SystemStateView& state) const;
    [[nodiscard]] Real evaluateBoundaryFunctional(const std::string& tag,
                                                  int boundary_marker,
                                                  const SystemStateView& state) const;

    /**
     * @brief Enable coupled boundary-condition infrastructure for a primary field
     *
     * This creates (or returns) a CoupledBoundaryManager used to orchestrate
     * boundary functionals and auxiliary (0D) state updates. The primary field
     * is used as the discrete solution source for BoundaryFunctional evaluation.
     */
    CoupledBoundaryManager& coupledBoundaryManager(FieldId primary_field);
    [[nodiscard]] CoupledBoundaryManager* coupledBoundaryManager() noexcept { return coupled_boundary_.get(); }
    [[nodiscard]] const CoupledBoundaryManager* coupledBoundaryManager() const noexcept { return coupled_boundary_.get(); }

    /**
     * @brief Access the boundary reduction service for a given primary field.
     *
     * Lazily creates the service on first access.  The service provides
     * physics-agnostic boundary-integral evaluation and is shared with
     * the CoupledBoundaryManager (if present) and AuxiliaryInputRegistry.
     */
    BoundaryReductionService& boundaryReductionService(FieldId primary_field);
    [[nodiscard]] BoundaryReductionService* boundaryReductionServiceIfPresent(FieldId primary_field) noexcept
    {
        auto it = boundary_reduction_services_.find(primary_field);
        return it != boundary_reduction_services_.end() ? it->second.get() : nullptr;
    }

    /**
     * @brief Access the generalized auxiliary state manager.
     *
     * Lazily creates the manager on first access.  The manager owns
     * auxiliary blocks with any scope (Global, Node, Cell, etc.) and
     * provides distributed ownership, sync, and checkpoint APIs.
     */
    AuxiliaryStateManager& auxiliaryStateManager();
    [[nodiscard]] AuxiliaryStateManager* auxiliaryStateManagerIfPresent() noexcept
    {
        return auxiliary_state_manager_.get();
    }
    [[nodiscard]] const AuxiliaryStateManager* auxiliaryStateManagerIfPresent() const noexcept
    {
        return auxiliary_state_manager_.get();
    }

    /**
     * @brief Access the auxiliary operator registry.
     *
     * Lazily creates the registry on first access.  Owns auxiliary
     * operators, coupling graph, and monolithic unknown layouts.
     */
    AuxiliaryOperatorRegistry& auxiliaryOperatorRegistry();
    [[nodiscard]] AuxiliaryOperatorRegistry* auxiliaryOperatorRegistryIfPresent() noexcept
    {
        return auxiliary_operator_registry_.get();
    }
    [[nodiscard]] const AuxiliaryOperatorRegistry* auxiliaryOperatorRegistryIfPresent() const noexcept
    {
        return auxiliary_operator_registry_.get();
    }

    /**
     * @brief Access the auxiliary input registry.
     *
     * Lazily creates the registry on first access.
     */
    AuxiliaryInputRegistry& auxiliaryInputRegistry();
    [[nodiscard]] AuxiliaryInputRegistry* auxiliaryInputRegistryIfPresent() noexcept
    {
        return auxiliary_input_registry_.get();
    }
    [[nodiscard]] const AuxiliaryInputRegistry* auxiliaryInputRegistryIfPresent() const noexcept
    {
        return auxiliary_input_registry_.get();
    }

    /**
     * @brief Access the FE-backed quantity definition registry.
     *
     * Lazily creates the registry on first access.
     */
    FEQuantityRegistry& feQuantityRegistry();
    [[nodiscard]] const FEQuantityRegistry* feQuantityRegistryIfPresent() const noexcept
    {
        return fe_quantity_registry_.get();
    }

    /**
     * @brief Register a sampled-state-field auxiliary input.
     *
     * Creates an entity-local input that samples the named FE field at
     * each node using direct DOF lookup (fast path for Lagrange elements).
     * Must be called after `setup()` (so DOF handlers exist) and before
     * `finalizeAuxiliaryLayout()`.
     *
     * @param input_name   Registry name for the input.
     * @param field_name   Name of the FE field to sample.
     * @param n_entities   Number of entities (nodes).
     */
    void registerSampledFieldInput(
        const std::string& input_name,
        const std::string& field_name,
        std::size_t n_entities);

    /**
     * @brief Register a boundary-face nodal sum auxiliary input.
     *
     * Creates a global input that sums all field DOF components at unique
     * boundary face vertices with the given marker.  The output size
     * equals the field component count.
     *
     * Requires `setup()` to have been called and a vertex-based (Lagrange)
     * FE space; throws if the field has no vertex DOFs.  For a quadrature-
     * weighted boundary integral, use the BoundaryFunctional assembly pipeline.
     */
    void registerBoundaryNodalSumInput(
        const std::string& input_name,
        const std::string& field_name,
        int boundary_marker);

    /**
     * @brief Register a true quadrature-weighted boundary integral as an auxiliary input.
     *
     * Creates a global scalar input backed by a real FE boundary functional
     * (not a nodal sum surrogate).  The input is evaluated via the
     * BoundaryReductionService and stored in the AuxiliaryInputRegistry
     * so that `AuxiliaryInput("name")` resolves to the integral value.
     *
     * This is the physics-agnostic API for registering boundary-integral
     * auxiliary inputs.  It supports:
     *
     * - `Sum` reduction (default): raw integral value.
     * - `Average` reduction: integral divided by boundary measure.
     * - Room for `Min`/`Max` in future extensions.
     *
     * ## Lifecycle
     *
     * May be called before or after `setup()`.  Must be called before
     * `installFormulation()` if the input name appears in an
     * `AuxiliaryInput(...)` symbol, and before `finalizeAuxiliaryLayout()`
     * if the input feeds an auxiliary model.  Both constraints are naturally
     * satisfied when called from a module's `registerOn()` method before
     * form installation.
     *
     * ## Multi-field support
     *
     * Integrands may reference multiple FE fields.  The first referenced
     * field provides the DOF layout and quadrature context; secondary
     * fields are automatically bound via `registerSecondaryField()` with
     * correct `field_type`, `component_offset`, and block DOF mapping.
     *
     * @param input_name   Registry name for the input (e.g., "Q").
     * @param functional   Boundary functional definition (integrand, marker, reduction).
     * @param schedule     When the input is re-evaluated (default: OncePerTimeStep).
     */
    void registerBoundaryIntegralInput(
        const std::string& input_name,
        forms::BoundaryFunctional functional,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a true quadrature-weighted boundary integral as an auxiliary input.
     *
     * Convenience overload that constructs a BoundaryFunctional from components.
     *
     * @param input_name       Registry name for the input.
     * @param integrand        Scalar-valued integrand expression.
     * @param boundary_marker  Boundary label to integrate over.
     * @param reduction        Reduction mode (default: Sum).
     * @param schedule         When the input is re-evaluated.
     */
    void registerBoundaryIntegralInput(
        const std::string& input_name,
        forms::FormExpr integrand,
        int boundary_marker,
        forms::BoundaryFunctional::Reduction reduction = forms::BoundaryFunctional::Reduction::Sum,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    // ---- Setup phase ----
    void setup(const SetupOptions& opts = {}, const SetupInputs& inputs = {});

    // ---- Constraints lifecycle ----
    void updateConstraints(double time, double dt = 0.0);

    // ---- Assembly phase ----
    assembly::AssemblyResult assemble(
        const AssemblyRequest& req,
        const SystemStateView& state,
        assembly::GlobalSystemView* matrix_out,
        assembly::GlobalSystemView* vector_out);

    assembly::AssemblyResult assembleResidual(
        const SystemStateView& state,
        assembly::GlobalSystemView& rhs_out);

    assembly::AssemblyResult assembleJacobian(
        const SystemStateView& state,
        assembly::GlobalSystemView& jac_out);

    assembly::AssemblyResult assembleMass(
        const SystemStateView& state,
        assembly::GlobalSystemView& mass_out);

    // ---- Time stepping lifecycle ----
    void beginTimeStep(bool reset_auxiliary_state = true,
                       bool invalidate_auxiliary_inputs = true);
    void commitTimeStep();

    // ---- Auxiliary model deployment ----

    /**
     * @brief Deploy an auxiliary model instance into the system.
     *
     * Collects the instance for setup-time finalization.  Must be called
     * before `setup()`.  During `finalizeAuxiliaryLayout()`, deployed
     * instances are registered as blocks, inputs, and steppers.
     */
    void deployAuxiliaryModel(AuxiliaryDeployedInstance instance);

    /**
     * @brief Deploy an auxiliary model and return a typed instance handle.
     *
     * Preferred over `deployAuxiliaryModel()` — returns a handle for
     * string-free output access.
     *
     * ```cpp
     * auto rcr = system.deploy(use(model).name("rcr_1")...);
     * auto p_out = rcr.output("P_out");
     * ```
     */
    AuxiliaryInstanceHandle deploy(AuxiliaryDeployedInstance instance);

    // ---- Handle-returning auxiliary input registration ----

    /**
     * @brief Register a boundary integral as an auxiliary input and return a handle.
     *
     * ```cpp
     * auto Q = system.boundaryIntegral(inner(u, n), marker);
     * ```
     */
    [[deprecated("boundaryIntegral(name, ...) is deprecated; use boundaryIntegral(...) without an explicit name")]]
    AuxiliaryInputHandle boundaryIntegral(
        const std::string& input_name,
        forms::FormExpr integrand,
        int boundary_marker,
        forms::BoundaryFunctional::Reduction reduction = forms::BoundaryFunctional::Reduction::Sum,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a boundary integral with an auto-generated internal name.
     */
    AuxiliaryInputHandle boundaryIntegral(
        forms::FormExpr integrand,
        int boundary_marker,
        forms::BoundaryFunctional::Reduction reduction = forms::BoundaryFunctional::Reduction::Sum,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a boundary integral (full functional) and return a handle.
     */
    [[deprecated("boundaryIntegral(name, functional, ...) is deprecated; use boundaryIntegral(functional, ...) without an explicit name")]]
    AuxiliaryInputHandle boundaryIntegral(
        const std::string& input_name,
        forms::BoundaryFunctional functional,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a boundary integral functional with an auto-generated internal name.
     */
    AuxiliaryInputHandle boundaryIntegral(
        forms::BoundaryFunctional functional,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a derived auxiliary input computed from an expression.
     *
     * Auto-discovers dependencies on other auxiliary inputs referenced in `expr`.
     * The expression is evaluated using the current auxiliary input values.
     *
     * ```cpp
     * auto P_out = system.derivedInput("P_out", Pd + (Rp + Rd) * Q);
     * ```
     *
     * @param name  Registry name for the derived input.
     * @param expr  Expression to evaluate (may reference other auxiliary inputs).
     * @param schedule  When the input is re-evaluated.
     * @return Handle for binding and expression use.
     */
    AuxiliaryInputHandle derivedInput(
        const std::string& name,
        forms::FormExpr expr,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a sampled FE field as an auxiliary input and return a handle.
     *
     * ```cpp
     * auto u_sample = system.sampledField("u_sample", "u", n_nodes);
     * ```
     */
    AuxiliaryInputHandle sampledField(
        const std::string& input_name,
        const std::string& field_name,
        std::size_t n_entities);

    /**
     * @brief Register a boundary nodal sum as an auxiliary input and return a handle.
     *
     * ```cpp
     * auto Q_nodal = system.boundaryNodalSum("Q_nodal", "u", marker);
     * ```
     */
    AuxiliaryInputHandle boundaryNodalSum(
        const std::string& input_name,
        const std::string& field_name,
        int boundary_marker);

    /**
     * @brief Register a boundary average as an auxiliary input.
     *
     * Computes `∫_Γ expr ds / ∫_Γ 1 ds` on the boundary with the given marker.
     */
    AuxiliaryInputHandle boundaryAverage(
        const std::string& input_name,
        forms::FormExpr integrand,
        int boundary_marker,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a domain integral as an auxiliary input.
     *
     * Computes `∫_Ω expr dx` over all cells.
     */
    AuxiliaryInputHandle domainIntegral(
        const std::string& input_name,
        forms::FormExpr integrand,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a domain average as an auxiliary input.
     *
     * Computes `∫_Ω expr dx / ∫_Ω 1 dx` over all cells.
     */
    AuxiliaryInputHandle domainAverage(
        const std::string& input_name,
        forms::FormExpr integrand,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a region-restricted integral as an auxiliary input.
     *
     * Computes `∫_R expr dx` over cells matching the given marker.
     */
    AuxiliaryInputHandle regionIntegral(
        const std::string& input_name,
        forms::FormExpr integrand,
        int region_marker,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a region-restricted average as an auxiliary input.
     *
     * Computes `∫_R expr dx / ∫_R 1 dx` over cells matching the given marker.
     */
    AuxiliaryInputHandle regionAverage(
        const std::string& input_name,
        forms::FormExpr integrand,
        int region_marker,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a generic FE expression as an auxiliary input.
     *
     * Evaluates the expression at a representative point (cell centroid)
     * for each entity.  For scalar global quantities, use domainIntegral()
     * or boundaryIntegral() instead.
     */
    AuxiliaryInputHandle feExpression(
        const std::string& input_name,
        forms::FormExpr expression,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    // ---- Auxiliary state lifecycle ----

    /**
     * @brief Prepare auxiliary inputs and state for assembly.
     *
     * Evaluates all auxiliary input providers (respecting schedules),
     * and binds auxiliary values into the assembler context.
     * Called before each PDE assembly (including within Newton iterations).
     *
     * @param state  Current system state view.
     * @param is_nonlinear_iteration  If true, refreshes inputs with
     *        `EachNonlinearIteration` schedule.
     */
    void prepareAuxiliaryForAssembly(const SystemStateView& state,
                                      bool is_nonlinear_iteration = false);

    /**
     * @brief Advance all Partitioned auxiliary blocks by one time step.
     *
     * Dispatches to the per-block stepper.  Monolithic blocks are NOT
     * advanced here — their time discretization is part of the global
     * assembled solve.
     *
     * Respects per-block scheduling (SingleRate, Subcycled, Multirate).
     *
     * @warning This overload does NOT update the cached system state used by
     * FE-coupled auxiliary input callbacks (boundary integrals, sampled fields).
     * Those callbacks read from cached spans/pointers that were populated by the
     * most recent call to `prepareAuxiliaryForAssembly()` or
     * `advanceAuxiliaryState(const SystemStateView&)`.  If no such call has been
     * made in this time step, or if the underlying data (solution vectors) has
     * been freed or overwritten, the callbacks will read stale or invalid data.
     *
     * Use `advanceAuxiliaryState(const SystemStateView&)` instead when any
     * registered auxiliary input depends on FE field state.
     *
     * @param time  Current simulation time.
     * @param dt    PDE time step.
     */
    void advanceAuxiliaryState(Real time, Real dt);

    /**
     * @brief Advance auxiliary state with full system state context.
     *
     * Caches the system state (solution, previous solutions, time integration
     * context, user data) before evaluating auxiliary inputs and stepping.
     * This ensures that FE-coupled input callbacks (boundary integrals, sampled
     * fields) have access to valid, current data.
     *
     * **This is the preferred overload** when any registered auxiliary input
     * depends on FE field state (e.g., boundary-integral inputs registered via
     * `registerBoundaryIntegralInput()`).
     *
     * @param state  Full system state (time, dt, solution, history, etc.).
     */
    void advanceAuxiliaryState(const SystemStateView& state);

    /**
     * @brief Advance auxiliary state with full system state context and
     *        nonlinear-iteration-aware input refresh.
     *
     * Behaves like `advanceAuxiliaryState(const SystemStateView&)`, but when
     * `is_nonlinear_iteration` is true it also refreshes auxiliary inputs whose
     * update schedule is `EachNonlinearIteration` before stepping.
     *
     * @param state  Full system state (time, dt, solution, history, etc.).
     * @param is_nonlinear_iteration  If true, refreshes inputs with
     *        `EachNonlinearIteration` schedule before stepping.
     */
    void advanceAuxiliaryState(const SystemStateView& state,
                               bool is_nonlinear_iteration);

    /**
     * @brief Assemble monolithic auxiliary residual and Jacobian.
     *
     * Evaluates the residual F(xdot, x, ...) and Jacobian dF/dx for all
     * monolithic auxiliary blocks.  Results are stored in the provided
     * dense vectors/matrices.
     *
     * @param time  Current time.
     * @param dt    Time step (for xdot computation).
     * @param residual_out  Output residual vector (sized to total_aux_unknowns).
     * @param jacobian_out  Output Jacobian matrix (row-major, n×n).
     * @param is_nonlinear_iteration  When true, EachNonlinearIteration
     *        auxiliary inputs are refreshed before assembly.  Pass true
     *        on each Newton iteration; false (default) on the first call.
     */
    void assembleMonolithicAuxiliary(
        Real time, Real dt,
        std::span<Real> residual_out,
        std::span<Real> jacobian_out,
        bool is_nonlinear_iteration = false);

    /**
     * @brief Assemble mixed auxiliary contributions into dense outputs (test helper).
     *
     * Assembles the monolithic auxiliary blocks + chain-rule field-auxiliary
     * coupling into dense vector/matrix outputs sized for the mixed system
     * (n_field_dofs + n_aux_dofs).
     *
     * @param state           System state.
     * @param n_field_dofs    Number of FE field DOFs.
     * @param residual_out    Dense vector (n_field + n_aux).
     * @param matrix_out      Dense row-major matrix ((n_field+n_aux)^2).
     */
    void assembleMixedAuxiliaryDense(
        const SystemStateView& state,
        std::size_t n_field_dofs,
        std::vector<Real>& residual_out,
        std::vector<Real>& matrix_out);

    /**
     * @brief Get the composed mixed system layout.
     *
     * Only valid after `finalizeAuxiliaryLayout()`.
     * @param n_field_unknowns  Number of FE field DOFs.
     */
    [[nodiscard]] MixedSystemLayout composeMixedSystemLayout(
        std::size_t n_field_unknowns = 0) const;

    /**
     * @brief Rollback all auxiliary blocks to their committed state.
     *
     * Used after a failed nonlinear solve or rejected time step.
     */
    void rollbackAuxiliaryState();

    /**
     * @brief Convert monolithic auxiliary stage values to end-of-step values.
     *
     * Generalized-alpha stage solves store differential auxiliary variables at
     * the stage state x_{n+alpha_f}. Before commit, these must be mapped back
     * to x_{n+1}. Algebraic rows are left unchanged.
     *
     * @param alpha_f    Generalized-alpha stage weight.
     * @param final_time Physical time of the accepted end-of-step state.
     */
    void finalizeMonolithicAuxiliaryStageState(Real alpha_f, Real final_time);

    /**
     * @brief Finalize monolithic auxiliary stage values and update stored
     *        committed rates for first-order generalized-alpha.
     *
     * @param alpha_f    Generalized-alpha stage weight.
     * @param gamma      Generalized-alpha gamma parameter.
     * @param dt         Full time-step size.
     * @param final_time Physical time of the accepted end-of-step state.
     */
    void finalizeMonolithicAuxiliaryStageState(Real alpha_f, Real gamma, Real dt, Real final_time);

    /**
     * @brief Finalize auxiliary layouts during setup.
     *
     * Called from `setup()` after all auxiliary model instances have been
     * deployed.  Finalizes monolithic unknown layouts and builds any
     * requested symbolic derivative artifacts.
     */
    void finalizeAuxiliaryLayout();

    /**
     * @brief Pack all auxiliary state for checkpoint.
     */
    [[nodiscard]] std::vector<Real> checkpointAuxiliaryState() const;

    /**
     * @brief Restore auxiliary state from checkpoint data.
     */
    void restoreAuxiliaryState(std::span<const Real> data);

    /**
     * @brief Get the flattened evaluated auxiliary output values.
     *
     * Updated by `prepareAuxiliaryForAssembly()`.  Empty if no
     * deployed models have output expressions.
     */
    [[nodiscard]] std::span<const Real> auxiliaryOutputValues() const noexcept;

    /**
     * @brief Get the flattened slot index of a named auxiliary output.
     *
     * Outputs are flattened across all deployed models in deployment order.
     * Each model contributes `n_entities * n_outputs` slots in the flat
     * buffer.  Returns the entity-0 slot for the named output; per-entity
     * access is `slot + entity_index * n_outputs_for_that_model`.
     *
     * Safe to call after `finalizeAuxiliaryLayout()` (does not depend on
     * runtime-populated output buffers).
     *
     * @return Slot index, or `std::size_t(-1)` if not found.
     */
    [[nodiscard]] std::size_t auxiliaryOutputSlotOf(std::string_view output_name) const;

    /**
     * @brief Instance-qualified output slot lookup.
     *
     * Use this overload when multiple deployed models have outputs with
     * the same name (e.g., two RCR models each exposing "P_out").
     *
     * @param instance_name  The deployed instance name.
     * @param output_name    The output name within that instance.
     * @return Slot index, or `std::size_t(-1)` if not found.
     */
    [[nodiscard]] std::size_t auxiliaryOutputSlotOf(
        std::string_view instance_name, std::string_view output_name) const;

    /**
     * @brief Lowered algebraic output expression lookup by symbolic output name.
     *
     * When a deployed AuxiliaryState output can be expressed directly in terms
     * of runtime-available terminals (for example AuxiliaryStateRef,
     * AuxiliaryInputRef, and constants), setup may lower that output to a
     * direct expression for assembly-time substitution in a plain NaturalBC.
     *
     * For live monolithic blocks, formulation metadata may still preserve the
     * original AuxiliaryOutputRef so the bordered direct-coupling path can
     * extract dR/d(output). Fully lowered direct-only blocks instead lower the
     * metadata as well because there is no live auxiliary unknown to couple.
     */
    [[nodiscard]] std::optional<forms::FormExpr>
    loweredAuxiliaryOutputExpr(std::string_view output_name) const;

    /// Slot-based lowered algebraic output lookup.
    [[nodiscard]] std::optional<forms::FormExpr>
    loweredAuxiliaryOutputExpr(std::size_t slot) const;

    /**
     * @brief Boundary-BC lowering for algebraic outputs that can be rewritten
     *        entirely in terms of coupled boundary functionals.
     *
     * This is used by `BoundaryConditionManager` so a simplified `NaturalBC`
     * authored against an algebraic AuxiliaryState output can still route
     * through the native coupled-boundary Jacobian path when the output is
     * exactly expressible using `boundaryIntegral(...)` placeholders.
     */
    [[nodiscard]] std::optional<forms::FormExpr>
    coupledBoundaryCompatibleAuxiliaryOutputExpr(std::string_view output_name) const;

    /**
     * @brief Return true when formulation metadata should keep AuxiliaryOutputRef.
     *
     * Live monolithic blocks preserve the output reference in metadata so the
     * bordered direct-coupling path can still assemble dR/d(output). Direct-only
     * lowered blocks return false because their metadata should use the same
     * lowered expression as assembly.
     */
    [[nodiscard]] bool auxiliaryOutputMetadataUsesRef(std::string_view output_name) const;

    /**
     * @brief Get an analysis summary of auxiliary blocks and inputs.
     */
    struct AuxiliaryAnalysisSummary {
        std::size_t n_blocks{0};
        std::size_t n_partitioned{0};
        std::size_t n_monolithic{0};
        std::size_t n_inputs{0};
        std::size_t total_aux_unknowns{0};
        std::vector<std::string> block_names{};
        std::vector<std::string> input_names{};
    };
    [[nodiscard]] AuxiliaryAnalysisSummary auxiliaryAnalysisSummary() const;

	    // ---- Accessors ----
	    [[nodiscard]] const assembly::IMeshAccess& meshAccess() const;
	    [[nodiscard]] std::string assemblerName() const;
	    [[nodiscard]] std::string assemblerSelectionReport() const;
	    [[nodiscard]] const ISearchAccess* searchAccess() const noexcept { return search_access_.get(); }
	    void setSearchAccess(std::shared_ptr<const ISearchAccess> access) { search_access_ = std::move(access); }

	    /**
	     * @brief Locate a physical point in the mesh using the configured search access.
	     */
	    [[nodiscard]] ISearchAccess::PointLocation locatePoint(const std::array<Real, 3>& point,
	                                                           GlobalIndex hint_cell = INVALID_GLOBAL_INDEX) const;

	    /**
	     * @brief Evaluate a field at a physical point (search + reference-space interpolation).
	     *
	     * @return nullopt if no search access is configured or the point is not located in the mesh.
	     */
	    [[nodiscard]] std::optional<std::array<Real, 3>> evaluateFieldAtPoint(FieldId field,
	                                                                          const SystemStateView& state,
	                                                                          const std::array<Real, 3>& point,
	                                                                          GlobalIndex hint_cell = INVALID_GLOBAL_INDEX) const;

	    /**
	     * @brief Evaluate a field at all mesh vertices by direct DOF coefficient lookup.
	     *
	     * For Lagrange elements, basis functions equal 1 at their associated node and 0
	     * at all others, so the field value at a vertex equals the DOF coefficient.
	     * This avoids the expensive locatePoint + evaluate per vertex.
	     *
	     * @param field       Field to evaluate
	     * @param state       Current system state
	     * @param n_vertices  Number of mesh vertices
	     * @param out         Output buffer, size >= n_vertices * max(1, components)
	     * @return true if direct nodal evaluation was used, false if not supported
	     *         (caller should fall back to evaluateFieldAtPoint)
	     */
	    [[nodiscard]] bool evaluateFieldAtVertices(FieldId field,
	                                               const SystemStateView& state,
	                                               GlobalIndex n_vertices,
	                                               std::span<double> out) const;

	    // ---- Global-kernel persistent state (optional) ----
	    [[nodiscard]] assembly::MaterialStateView globalKernelCellState(const GlobalKernel& kernel,
	                                                                    GlobalIndex cell_id,
	                                                                    LocalIndex num_qpts) const;
    [[nodiscard]] assembly::MaterialStateView globalKernelBoundaryFaceState(const GlobalKernel& kernel,
                                                                            GlobalIndex face_id,
                                                                            LocalIndex num_qpts) const;
    [[nodiscard]] assembly::MaterialStateView globalKernelInteriorFaceState(const GlobalKernel& kernel,
                                                                            GlobalIndex face_id,
                                                                            LocalIndex num_qpts) const;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
	    [[nodiscard]] const svmp::Mesh* mesh() const noexcept { return mesh_.get(); }
	    [[nodiscard]] svmp::Configuration coordinateConfiguration() const noexcept { return coord_cfg_; }

	    void setInterfaceMesh(InterfaceId marker, std::shared_ptr<const svmp::InterfaceMesh> mesh);
	    [[nodiscard]] bool hasInterfaceMesh(InterfaceId marker) const noexcept;
	    [[nodiscard]] const svmp::InterfaceMesh& interfaceMesh(InterfaceId marker) const;

	    void setInterfaceMeshFromFaceSet(InterfaceId marker,
	                                     const std::string& face_set_name,
	                                     bool compute_orientation = true);
	    void setInterfaceMeshFromBoundaryLabel(InterfaceId marker,
	                                           int boundary_label,
	                                           bool compute_orientation = true);
#endif

    [[nodiscard]] const dofs::DofHandler& dofHandler() const noexcept { return dof_handler_; }
    [[nodiscard]] const FieldRecord& fieldRecord(FieldId field) const;
    [[nodiscard]] const dofs::DofHandler& fieldDofHandler(FieldId field) const;
    [[nodiscard]] GlobalIndex fieldDofOffset(FieldId field) const;
    [[nodiscard]] const dofs::FieldDofMap& fieldMap() const noexcept { return field_map_; }
    [[nodiscard]] const dofs::BlockDofMap* blockMap() const noexcept { return block_map_.get(); }
    [[nodiscard]] const constraints::AffineConstraints& constraints() const noexcept { return affine_constraints_; }
    [[nodiscard]] const sparsity::SparsityPattern& sparsity(const OperatorTag& op) const;
    [[nodiscard]] const sparsity::DistributedSparsityPattern* distributedSparsityIfAvailable(const OperatorTag& op) const noexcept;
    [[nodiscard]] std::shared_ptr<const backends::DofPermutation> dofPermutation() const noexcept { return dof_permutation_; }

		    [[nodiscard]] bool isSetup() const noexcept { return is_setup_; }
		    [[nodiscard]] int temporalOrder() const noexcept;
		    [[nodiscard]] bool isTransient() const noexcept { return temporalOrder() > 0; }
		    [[nodiscard]] std::vector<FieldId> timeDerivativeFields(const OperatorTag& op) const;
		    [[nodiscard]] std::vector<FieldId> timeDerivativeFields() const;

	    // ---- Parameter requirements (optional) ----
	    [[nodiscard]] const ParameterRegistry& parameterRegistry() const noexcept { return parameter_registry_; }

    // ---- Gauge / nullspace detection (optional) ----
    /**
     * @brief Access the GaugeRegistry, creating it on first call
     *
     * The GaugeRegistry is an optional component for automatic nullspace
     * detection and enforcement.  It is created lazily on first access.
     */
    [[nodiscard]] gauge::GaugeRegistry& gaugeRegistry();
    [[nodiscard]] const gauge::GaugeRegistry* gaugeRegistryIfPresent() const noexcept {
        return gauge_registry_.get();
    }
    [[nodiscard]] bool hasGaugeRegistry() const noexcept { return gauge_registry_ != nullptr; }

    // ---- Problem analysis subsystem ----

    void addFormulationRecord(analysis::FormulationRecord record);
    void addBoundaryConditionDescriptor(analysis::BoundaryConditionDescriptor desc);
    void addContribution(analysis::ContributionDescriptor desc);
    void addVariableDescriptor(analysis::VariableDescriptor desc);

    [[nodiscard]] const std::vector<analysis::FormulationRecord>& formulationRecords() const noexcept {
        return formulation_records_;
    }
    [[nodiscard]] const std::vector<analysis::BoundaryConditionDescriptor>& boundaryConditionDescriptors() const noexcept {
        return bc_descriptors_;
    }
    [[nodiscard]] const std::vector<analysis::VariableDescriptor>& variableDescriptors() const noexcept {
        return variable_descriptors_;
    }
    [[nodiscard]] const std::vector<analysis::ContributionDescriptor>& contributionDescriptors() const noexcept {
        return contributions_;
    }

    /// Build and store topology context from the mesh
    void buildTopologyContext();
    [[nodiscard]] const analysis::TopologyAnalysisContext* topologyContext() const noexcept {
        return topology_context_ ? &*topology_context_ : nullptr;
    }

    /// Build and store interface topology from registered InterfaceMesh objects
    void buildInterfaceTopologyContext();
    [[nodiscard]] const analysis::InterfaceTopologyContext* interfaceTopologyContext() const noexcept {
        return interface_topology_context_ ? &*interface_topology_context_ : nullptr;
    }

    /// Build and store constraint summary from current AffineConstraints
    void buildConstraintSummary();
    [[nodiscard]] const analysis::ConstraintAnalysisSummary* constraintSummary() const noexcept {
        return constraint_summary_ ? &*constraint_summary_ : nullptr;
    }

    /// Invalidate cached analysis report (called automatically by mutation methods)
    void invalidateAnalysisCache() noexcept;

    /// Run all analysis passes and return a fresh report
    [[nodiscard]] analysis::ProblemAnalysisReport runProblemAnalysis() const;

    /// Cached version — re-runs only if inputs have changed
    [[nodiscard]] const analysis::ProblemAnalysisReport& analysisReport() const;

    // ---- Operator registry query (for tests and diagnostics) ----

    /**
     * @brief Query the registered operator definition for an operator tag
     *
     * Returns the OperatorDefinition containing all registered cell, boundary,
     * interior-face, interface-face, and global terms. This is the structural
     * view of what was installed — useful for parity tests that verify mixed
     * and manual installation paths produce identical block structure.
     */
    [[nodiscard]] const OperatorDefinition& operatorDefinition(const OperatorTag& op) const {
        return operator_registry_.get(op);
    }

    [[nodiscard]] bool hasOperator(const OperatorTag& op) const noexcept {
        return operator_registry_.has(op);
    }

    // ---- Rank-1 updates from coupled Jacobian assembly ----
    [[nodiscard]] std::span<const backends::RankOneUpdate> lastRankOneUpdates() const noexcept;
    void clearRankOneUpdates() noexcept;
    [[nodiscard]] std::span<const backends::ReducedFieldUpdate> lastReducedFieldUpdates() const noexcept;
    void clearReducedFieldUpdates() noexcept;
    [[nodiscard]] std::span<const Real> lastLocalCondensedRhsShift() const noexcept;
    [[nodiscard]] bool hasLocalCondensedRecovery() const noexcept
    {
        return !last_local_condensed_records_.empty();
    }
    void applyLocalCondensedRecovery(std::span<const Real> dense_du, Real alpha = Real(1.0));
    void clearLocalCondensedRecovery() noexcept;

    /// @cond INTERNAL
    // Internal — used by FormsInstaller for transactional kernel registration.
    // Public only because C++ templates cannot be called from non-friend TUs
    // when private. Do not call from physics modules.
    template <typename Fn>
    auto executeWithOperatorRollback_(Fn&& fn) -> decltype(fn()) {
        auto snap = operator_registry_.snapshot();
        try {
            return fn();
        } catch (...) {
            operator_registry_.rollback(snap);
            throw;
        }
    }
    /// @endcond

private:

    struct PlannedCellTerm {
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        assembly::AssemblyKernel* kernel{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
        assembly::SemanticKernelKind semantic_kind{assembly::SemanticKernelKind::SingleForm};
        bool matrix_capable{false};
        bool vector_capable{false};
    };

    struct LocalCondensedEntityRecord {
        std::string block_name{};
        std::size_t entity_index{0};
        std::vector<std::vector<std::pair<GlobalIndex, Real>>> B_columns{};
        std::vector<std::vector<std::pair<GlobalIndex, Real>>> Ct_rows{};
        std::vector<Real> D_inv{};
        std::vector<Real> g{};
    };

    struct PlannedBoundaryTerm {
        int marker{0};
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        assembly::AssemblyKernel* kernel{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
        bool matrix_capable{false};
        bool vector_capable{false};
    };

    struct PlannedInteriorFaceTerm {
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        assembly::AssemblyKernel* kernel{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
        bool matrix_capable{false};
        bool vector_capable{false};
    };

    struct PlannedInterfaceFaceTerm {
        int marker{0};
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        assembly::AssemblyKernel* kernel{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
        bool matrix_capable{false};
        bool vector_capable{false};
    };

    struct OperatorAssemblyPlan {
        std::vector<PlannedCellTerm> cell_terms{};
        std::vector<PlannedBoundaryTerm> boundary_terms{};
        std::vector<PlannedInteriorFaceTerm> interior_terms{};
        std::vector<PlannedInterfaceFaceTerm> interface_terms{};
        std::vector<GlobalKernel*> global_terms{};
    };

    friend assembly::AssemblyResult assembleOperator(
        FESystem& system,
        const AssemblyRequest& request,
        const SystemStateView& state,
        assembly::GlobalSystemView* matrix_out,
        assembly::GlobalSystemView* vector_out);
    friend class OperatorBackends;

    void invalidateSetup() noexcept;
    void requireSetup() const;
    void requireSingleFieldSetup() const;
    void buildAssemblyPlans();

    [[nodiscard]] const FieldRecord& singleField() const;

    std::shared_ptr<const assembly::IMeshAccess> mesh_access_;
    std::shared_ptr<const ISearchAccess> search_access_{};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
	    std::shared_ptr<const svmp::Mesh> mesh_{};
	    svmp::Configuration coord_cfg_{svmp::Configuration::Reference};
	    std::unordered_map<InterfaceId, std::shared_ptr<const svmp::InterfaceMesh>> interface_meshes_{};
#endif

    FieldRegistry field_registry_;
    OperatorRegistry operator_registry_;
    std::vector<std::unique_ptr<constraints::Constraint>> constraint_defs_;
    std::vector<std::unique_ptr<ISystemConstraint>> system_constraint_defs_;

    dofs::DofHandler dof_handler_{};
    std::vector<dofs::DofHandler> field_dof_handlers_{};
    std::vector<GlobalIndex> field_dof_offsets_{};
    dofs::FieldDofMap field_map_{};
    std::unique_ptr<dofs::BlockDofMap> block_map_{};
	    constraints::AffineConstraints affine_constraints_{};

		    std::unordered_map<OperatorTag, std::unique_ptr<sparsity::SparsityPattern>> sparsity_by_op_{};
		    std::unordered_map<OperatorTag, std::unique_ptr<sparsity::DistributedSparsityPattern>> distributed_sparsity_by_op_{};
		    std::shared_ptr<const backends::DofPermutation> dof_permutation_{};

	    std::unique_ptr<assembly::Assembler> assembler_{};
        bool use_constraints_in_assembly_{true};
		    std::string assembler_selection_report_{};
		    std::unique_ptr<assembly::IMaterialStateProvider> material_state_provider_{};
	    std::unique_ptr<GlobalKernelStateProvider> global_kernel_state_provider_{};
    std::unique_ptr<OperatorBackends> operator_backends_{};
    std::unique_ptr<CoupledBoundaryManager> coupled_boundary_{};
    std::unordered_map<FieldId, std::unique_ptr<BoundaryReductionService>> boundary_reduction_services_{};
    std::unique_ptr<AuxiliaryStateManager> auxiliary_state_manager_{};
    std::unique_ptr<AuxiliaryOperatorRegistry> auxiliary_operator_registry_{};
    std::unique_ptr<AuxiliaryInputRegistry> auxiliary_input_registry_{};
    std::unique_ptr<FEQuantityRegistry> fe_quantity_registry_{};

    /// Cached system state for FE-coupled auxiliary input callbacks.
    /// Set by cacheSystemState() which is called from prepareAuxiliaryForAssembly(),
    /// assembleMixedAuxiliaryIntoGlobal(), and advanceAuxiliaryState().
    /// Callbacks capture `this` and read from these members.
    mutable std::span<const Real> cached_solution_u_{};
    mutable const backends::GenericVector* cached_solution_vector_{nullptr};
    mutable std::span<const Real> cached_solution_u_prev_{};
    mutable const backends::GenericVector* cached_solution_prev_vector_{nullptr};
    mutable std::span<const Real> cached_solution_u_prev2_{};
    mutable const backends::GenericVector* cached_solution_prev2_vector_{nullptr};
    mutable const assembly::TimeIntegrationContext* cached_time_integration_{nullptr};
    mutable const void* cached_user_data_{nullptr};

    /// Cache a SystemStateView's fields for auxiliary input callbacks.
    void cacheSystemState(const SystemStateView& state) const;

    mutable std::vector<Real> field_endpoint_scratch_src_{}; ///< Scratch for distributed field source endpoint.
    mutable std::vector<Real> field_endpoint_scratch_tgt_{}; ///< Scratch for distributed field target endpoint.

    // Deployed auxiliary model instances (collected before setup, consumed during finalize).
    struct DeployedAuxEntry {
        std::shared_ptr<AuxiliaryStateModel> model{};
        std::string instance_name{};
        AuxiliaryStateSpec spec{};
        AuxiliaryStepperSpec stepper_spec{};
        std::vector<Real> initial_values{};
        std::map<std::string, std::string> input_bindings{}; ///< Ordered for deterministic iteration
        std::unordered_map<std::string, AuxiliaryInputHandle> coupled_bindings{}; ///< For chain-rule coupling
        std::unordered_map<std::string, Real> param_values{};
        std::size_t explicit_entity_count{0}; ///< 0 = auto from scope/mesh
        std::unique_ptr<AuxiliaryStateStepper> stepper{};
        std::unique_ptr<AuxiliaryDerivativeProvider> deriv_provider{};
        std::vector<Real> output_buffer{}; ///< Evaluated output values
        bool lower_to_direct_only{false};  ///< Keep semantic block/output, but exclude from live monolithic solve.
        bool local_condensed{false};       ///< Eliminate locally into reduced field updates instead of dense bordered layout.
        /// Entity map: indices of entities this block covers.
        /// Empty = all entities (WholeDomain / no restriction).
        std::vector<std::size_t> entity_map{};
        std::vector<std::size_t> qp_offsets{};
    };
    std::vector<DeployedAuxEntry> deployed_aux_entries_{};
    std::unordered_map<std::string, forms::FormExpr> lowered_aux_output_exprs_by_name_{};
    std::unordered_map<std::size_t, forms::FormExpr> lowered_aux_output_exprs_by_slot_{};
    [[nodiscard]] bool canLowerAlgebraicAuxiliaryToDirectOnly_(const DeployedAuxEntry& entry) const;
    [[nodiscard]] std::optional<forms::FormExpr>
    synthesizeLoweredAuxiliaryOutputExpr_(const DeployedAuxEntry& entry,
                                          std::string_view output_name) const;
    [[nodiscard]] std::optional<forms::FormExpr>
    synthesizeCoupledBoundaryAuxiliaryOutputExpr_(const DeployedAuxEntry& entry,
                                                  std::string_view output_name) const;
    void buildAuxiliaryOutputBindings_();
    /// Deferred dependency pairs (dependent, dependency) from derivedInput().
    /// Wired by finalizeDeferredInputDeps() when all inputs are registered.
    std::vector<std::pair<std::string, std::string>> deferred_input_deps_{};
    /// Deferred derived-input expressions needing AuxiliaryInputSymbol→Ref resolution.
    std::vector<std::pair<std::string, std::shared_ptr<forms::FormExpr>>> deferred_derived_exprs_{};
    /// Resolve deferred derived-input expressions and wire dependency edges.
    /// Safe to call multiple times — clears the deferred lists on first run.
    void finalizeDeferredInputDeps();
    void buildLoweredAuxiliaryOutputExpressions_();
    AuxiliaryInputHandle registerBoundaryIntegralHandle_(
        const std::string& input_name,
        forms::FormExpr integrand,
        int boundary_marker,
        forms::BoundaryFunctional::Reduction reduction,
        AuxiliaryInputUpdateSchedule schedule);
    AuxiliaryInputHandle registerBoundaryIntegralHandle_(
        const std::string& input_name,
        forms::BoundaryFunctional functional,
        AuxiliaryInputUpdateSchedule schedule);
    [[nodiscard]] std::string generateUniqueAuxiliaryInputName_(std::string_view prefix);
    [[nodiscard]] std::string makeScopeAwareInstanceBaseName_(const AuxiliaryDeployedInstance& instance) const;
    [[nodiscard]] std::string resolveDeploymentInstanceName_(const AuxiliaryDeployedInstance& instance) const;
    [[nodiscard]] bool hasDeployedInstanceName_(std::string_view instance_name) const;
    std::size_t generated_boundary_input_counter_{0};

    /// Bind secondary fields and set dof_per_node on a BoundaryReductionService
    /// for multi-field integrand evaluation.
    void bindSecondaryFields(BoundaryReductionService& svc,
                              FieldId primary_fid,
                              const std::vector<FieldId>& referenced_fields);
    std::unique_ptr<AuxiliaryMultirateScheduler> aux_scheduler_{};

public:
    /// Assemble boundary gradient dI/du for a functional with the given
    /// integrand (already transformed: DiscreteField → TrialFunction).
    /// Returns sparse (DOF, value) pairs.
    std::vector<BoundaryReductionService::SensitivityEntry>
    assembleBoundaryGradient(FieldId field,
                              const forms::FormExpr& integrand_trial,
                              int boundary_marker,
                              const SystemStateView& state,
                              bool apply_constraints = true,
                              int region_marker = -1);

private:

    void advanceOneEntry(DeployedAuxEntry& entry, Real time, Real dt, int substep_count);

    /// Build ordered parameter vector for a deployed entry.
    [[nodiscard]] std::vector<Real> buildParamVector(const DeployedAuxEntry& entry) const;

    /// Build ordered input vector for a deployed entry (non-entity-local).
    [[nodiscard]] std::vector<Real> buildInputVector(const DeployedAuxEntry& entry) const;

    /// Rebuild input vector for a generic (non-built) model at a specific entity,
    /// using declared input names with name:size parsing.
    void rebuildGenericInputsForEntity(
        const DeployedAuxEntry& entry, std::size_t entity_index,
        std::vector<Real>& out) const;

    /// Ensure monolithic auxiliary committed-rate buffers exist and are seeded
    /// for first-order generalized-alpha stage assembly.
    void ensureMonolithicCommittedRates(const SystemStateView& state);

    /// Seed one monolithic block's committed-rate buffer from the committed
    /// state and previous-step FE inputs.
    void initializeMonolithicCommittedRate(const DeployedAuxEntry& entry,
                                           const SystemStateView& prev_state);

    /// Ensure a flat committed-rate buffer exists for the given block.
    void ensureMonolithicCommittedRateBuffer(const DeployedAuxEntry& entry,
                                             std::size_t storage_size);

    /// Gather a per-entity committed-rate vector from the flat block buffer.
    [[nodiscard]] std::vector<Real> gatherMonolithicCommittedRate(
        const DeployedAuxEntry& entry,
        std::size_t entity_index) const;

    /// Scatter a per-entity committed-rate vector into the flat block buffer.
    void scatterMonolithicCommittedRate(const DeployedAuxEntry& entry,
                                        std::size_t entity_index,
                                        std::span<const Real> values);

    /// Wire FE-coupled auxiliary input providers during finalization.
    void wireFECoupledInputProviders();

    /// Assemble monolithic auxiliary contributions into a global system view.
    /// @param n_field_dofs  Number of FE field DOFs (for mixed offset computation).
    /// @param is_nonlinear_iteration  When true, EachNonlinearIteration inputs refresh.
    void assembleMixedAuxiliaryIntoGlobal(
        const SystemStateView& state,
        assembly::GlobalSystemView* matrix_out,
        assembly::GlobalSystemView* vector_out,
        bool want_matrix, bool want_vector,
        std::size_t n_field_dofs,
        bool is_nonlinear_iteration = false);

    /// Parse "name:size" suffix from a declared input name.
    /// Returns (base_name, component_count).
    static std::pair<std::string, int> parseDeclaredInputName(const std::string& raw);

    /// Validate all declared input names at deployment time (catches malformed suffixes).
    static void validateDeclaredInputNames(const AuxiliaryStateModel& model);

	    ParameterRegistry parameter_registry_{};
    std::unique_ptr<gauge::GaugeRegistry> gauge_registry_{};
    std::vector<backends::RankOneUpdate> last_rank_one_updates_{};
    std::vector<backends::ReducedFieldUpdate> last_reduced_field_updates_{};
    std::vector<LocalCondensedEntityRecord> last_local_condensed_records_{};
    std::vector<Real> last_local_condensed_rhs_shift_{};
    std::unordered_map<OperatorTag, OperatorAssemblyPlan> assembly_plan_by_op_{};

    // Cached coupled Jacobian results.
    // For time-invariant sensitivities (e.g., RCR/resistance BCs), keyed by dt only.
    // For time-variant sensitivities, keyed by (time, dt) pair.
    // Time-invariance is auto-detected on the first computation by walking the
    // FormExpr trees for solution/time references.
    struct CoupledJacobianCache {
        double time{-1e30};
        double dt{-1e30};
        bool valid{false};
        bool is_time_invariant{false};  ///< True if sensitivity depends only on geometry + dt
        std::vector<backends::RankOneUpdate> rank_one_updates{};
        // For non-symmetric cases: cached outer-product matrix entries.
        struct SparseEntry {
            GlobalIndex row;
            std::vector<GlobalIndex> col_dofs;
            std::vector<Real> col_vals;
        };
        std::vector<SparseEntry> outer_product_entries{};

        void clear() noexcept {
            time = -1e30;
            dt = -1e30;
            valid = false;
            is_time_invariant = false;
            rank_one_updates.clear();
            outer_product_entries.clear();
        }
    };
    CoupledJacobianCache coupled_jac_cache_{};
    std::unordered_map<std::string, std::vector<Real>> monolithic_aux_committed_rates_{};
    std::unordered_set<std::string> monolithic_aux_committed_rates_valid_{};

    // ---- Analysis subsystem storage ----
    std::vector<analysis::FormulationRecord> formulation_records_;
    std::vector<analysis::ContributionDescriptor> contributions_;
    std::size_t contributions_def_count_{0}; ///< Watermark for definition-phase contributions
    std::vector<analysis::BoundaryConditionDescriptor> bc_descriptors_;
    std::vector<analysis::VariableDescriptor> variable_descriptors_;

    std::optional<analysis::TopologyAnalysisContext> topology_context_;
    std::optional<analysis::InterfaceTopologyContext> interface_topology_context_;
    std::optional<analysis::ConstraintAnalysisSummary> constraint_summary_;
    mutable std::optional<analysis::ProblemAnalysisReport> analysis_report_cache_;
    mutable std::uint64_t analysis_inputs_version_{0};
    mutable std::uint64_t analysis_report_version_{std::numeric_limits<std::uint64_t>::max()};

	    bool is_setup_{false};
	    Real last_auxiliary_advance_time_{0.0};
        bool partitioned_auxiliary_advance_valid_{false};
        Real partitioned_auxiliary_advance_time_{std::numeric_limits<Real>::quiet_NaN()};
        Real partitioned_auxiliary_advance_dt_{std::numeric_limits<Real>::quiet_NaN()};
    mutable std::vector<Real> aux_output_flat_{}; ///< Flattened output values for assembly
    std::vector<assembly::AuxiliaryOutputBinding> auxiliary_output_bindings_{};

public:
    /// Bordered coupling data for monolithic auxiliary DOFs.
    /// Populated by assembleMixedAuxiliaryIntoGlobal when monolithic blocks exist.
    /// Consumed by the Newton solver to apply a bordered system correction
    /// after the PDE linear solve.
    struct BorderedCouplingData {
        bool active{false};             ///< True if monolithic aux DOFs exist
        bool globally_reduced{false};   ///< True once dense bordered blocks have been summed for replicated MPI use
        int n_aux{0};                   ///< Number of auxiliary unknowns
        std::size_t n_field_dofs{0};    ///< Number of PDE DOFs
        std::vector<Real> D;            ///< Aux-aux Jacobian (n_aux × n_aux, row-major)
        std::vector<Real> g;            ///< Auxiliary residual (n_aux)
        std::vector<Real> B;            ///< dR_PDE/dx_aux columns (n_field_dofs × n_aux, col-major)
        std::vector<Real> Ct;           ///< dR_aux/du rows (n_aux × n_field_dofs, row-major)
        std::vector<Real> dF_dxdot;     ///< Raw dF/dxdot block (n_aux × n_aux, row-major)
        std::vector<AuxiliaryVariableKind> aux_variable_kinds{}; ///< Per-aux unknown classification in mixed-storage order

        void clear() {
            active = false;
            globally_reduced = false;
            n_aux = 0;
            n_field_dofs = 0;
            D.clear(); g.clear(); B.clear(); Ct.clear();
            dF_dxdot.clear();
            aux_variable_kinds.clear();
            aux_blocks.clear();
            dF_dinputs.clear();
            dO_dx.clear();
            dO_dI.clear();
            direct_coupling_records.clear();
        }
        /// Per-block info for auxiliary state update after solve.
        struct AuxBlock { std::string name; int dim; };
        std::vector<AuxBlock> aux_blocks;

        struct DirectCouplingRecord {
            std::size_t output_slot{static_cast<std::size_t>(-1)};
            std::size_t entity_index{0};
            std::vector<std::size_t> aux_local_indices{};
            std::vector<Real> dF_dinputs{};
            std::vector<Real> dO_dx{};
            std::vector<Real> dO_dI{};
            std::vector<std::vector<std::pair<GlobalIndex, Real>>> input_gradients{};
            std::vector<std::pair<GlobalIndex, Real>> output_gradient{};
        };

        /// dF/d(inputs) per aux DOF (needed for B computation from Ct).
        std::vector<Real> dF_dinputs;  ///< (n_aux × n_inputs_per_block)
        /// d(output)/d(state) per aux DOF.
        std::vector<Real> dO_dx;       ///< (n_outputs × n_aux)
        std::vector<Real> dO_dI;       ///< d(output)/d(input), e.g., Rp for RCR
        /// Per-output/entity direct-coupling metadata for debugging and
        /// verification. Unlike the flat compatibility vectors above, these do
        /// not collapse multiple outputs or boundary entities into one record.
        std::vector<DirectCouplingRecord> direct_coupling_records{};

        void resize(int na, std::size_t nf) {
            n_aux = na;
            n_field_dofs = nf;
            globally_reduced = false;
            D.assign(static_cast<std::size_t>(na * na), 0.0);
            g.assign(static_cast<std::size_t>(na), 0.0);
            B.assign(nf * static_cast<std::size_t>(na), 0.0);
            Ct.assign(static_cast<std::size_t>(na) * nf, 0.0);
            dF_dxdot.assign(static_cast<std::size_t>(na * na), 0.0);
            aux_variable_kinds.assign(static_cast<std::size_t>(na),
                                      AuxiliaryVariableKind::Differential);
            dF_dinputs.clear();
            dO_dx.clear();
            dO_dI.clear();
            direct_coupling_records.clear();
            active = true;
        }
    };

    /// Access bordered coupling data (populated during assembly).
    [[nodiscard]] BorderedCouplingData& borderedCoupling() noexcept { return bordered_coupling_; }
    [[nodiscard]] const BorderedCouplingData& borderedCoupling() const noexcept { return bordered_coupling_; }
    [[nodiscard]] std::span<const assembly::AuxiliaryOutputBinding>
    auxiliaryOutputBindings() const noexcept
    {
        return auxiliary_output_bindings_;
    }

private:
    BorderedCouplingData bordered_coupling_{};
	};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FESYSTEM_H
