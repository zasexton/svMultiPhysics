#ifndef SVMP_FE_SYSTEMS_AUXILIARY_STATE_TYPES_H
#define SVMP_FE_SYSTEMS_AUXILIARY_STATE_TYPES_H

/**
 * @file AuxiliaryStateTypes.h
 * @brief Core type definitions for the generalized AuxiliaryState subsystem
 *
 * This header defines the physics-agnostic vocabulary for auxiliary (non-PDE)
 * state variables managed by the FE library.  Auxiliary state is FE-library
 * infrastructure — not a boundary-condition feature and not a physics-specific
 * concept.  Boundary functionals, EP-like ionic models, metabolism models,
 * reduced models, and future coupled subsystems all use the same neutral
 * AuxiliaryState infrastructure.
 *
 * ## Key design principles
 *
 * - **Block identity** is string-based by unique block name within an
 *   FESystem.  Scope is metadata carried on the block, not part of the
 *   public identity key.
 *
 * - **Mixed differential and algebraic variables** may coexist within a
 *   single auxiliary block.
 *
 * - **Solve mode** (`Partitioned` vs `Monolithic`) is fixed once deployed
 *   auxiliary instances are finalized during `system.setup()`.
 *
 * - **History depth** is block-wide in phase 1.
 *
 * - **Fixed-stride layout** is the default.  Ragged layout is an explicit
 *   choice with canonical per-entity offsets; grouped or archetyped fast
 *   paths are an internal optimization, not public API.
 *
 * - **Entity ordering** follows owned mesh or DOF-layer ordering, appends
 *   ghosts explicitly, and defaults to `ByEntityThenComponent` unless a
 *   formulation selects otherwise.
 *
 * - **Auxiliary block names** are the durable public handles.  Numeric block
 *   ids and slot ids are finalized at setup time and are internal,
 *   setup-stable implementation details.
 *
 * - **All auxiliary blocks** are owned by `FESystem` and must be finalized
 *   before `system.setup()`.  Adding new blocks afterward requires a future
 *   re-finalization workflow not part of phase 1.
 *
 * - **`schedule(...)`** controls when and how often auxiliary advancement
 *   occurs relative to the PDE step; **`stepper(...)`** controls the
 *   numerical method used when that advancement executes.
 *
 * - **Local stepper selection**, local nonlinear-solver options, substep
 *   input-refresh policy, and substep commit policy are valid only for
 *   `Partitioned` solve mode.
 *
 * - **`Monolithic` auxiliary unknowns** use auxiliary-specific unknown maps
 *   and layouts rather than reusing FE field DOF maps.  `FESystem` composes
 *   FE field unknown layouts and auxiliary-specific unknown layouts into one
 *   mixed system layout for assembly and solves.
 *
 * - **Derivative policy** is configured at the model level in phase 1 and
 *   inherited by deployed instances.  Precedence: analytic override first,
 *   then explicit model policy, then default symbolic generation when
 *   expressions are available.  Lower-level residual implementations that
 *   cannot satisfy the resolved policy must provide a compatible analytic,
 *   AD, or finite-difference path or fail setup with a clear diagnostic.
 *
 * - **Lowering contract** from math-first auxiliary model rows to residual
 *   `FormExpr` trees used for symbolic differentiation:
 *   - `ode(x, rhs)` lowers to `dot(x) - rhs`
 *   - `algebraic(z, expr)` lowers to `expr`
 *   - `residual(name, expr)` is taken as a raw residual row
 *
 * - **Symbolic differentiation targets** for auxiliary models:
 *   - Primary solve targets: auxiliary state, auxiliary time derivative,
 *     and coupled FE fields (for monolithic models).
 *   - Auxiliary-input sensitivities: optional.
 *   - Auxiliary outputs: derived expressions, not primary solve targets.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Storage scope
// ---------------------------------------------------------------------------

/**
 * @brief Storage scope for an auxiliary state block.
 *
 * The scope determines the entity set that owns instances of the block
 * and the associated indexing, ownership, and ghost/sync semantics.
 *
 * Storage scope is orthogonal to solve mode: `Global` scope does NOT
 * imply monolithic solve participation, and `Monolithic` solve mode
 * does NOT imply `Global` scope.
 */
enum class AuxiliaryStateScope : std::uint8_t {
    /// One instance for the entire system (lumped boundary models, etc.).
    Global,

    /// One instance per mesh node.  Supports owned + ghost partitioning.
    Node,

    /// One instance per mesh cell (element).
    Cell,

    /// One instance per quadrature point per cell.
    QuadraturePoint,

    /// One instance per boundary entity (face, edge, or vertex on a named
    /// boundary).  Supports stable entity indexing on boundary subsets.
    BoundaryEntity
};

// ---------------------------------------------------------------------------
//  Variable kind
// ---------------------------------------------------------------------------

/**
 * @brief Classification of an individual variable within an auxiliary block.
 *
 * A single block may contain both differential and algebraic variables.
 * The kind informs the stepper/solver how to treat each row of the local
 * residual system.
 */
enum class AuxiliaryVariableKind : std::uint8_t {
    /// Evolves via a time derivative: F(dot(x), x, ...) = 0
    Differential,

    /// Determined by an algebraic constraint: g(x, ...) = 0
    Algebraic
};

// ---------------------------------------------------------------------------
//  Solve mode
// ---------------------------------------------------------------------------

/**
 * @brief How an auxiliary block participates in the global solve.
 *
 * Solve mode is fixed once deployed auxiliary instances are finalized
 * during `system.setup()`.
 */
enum class AuxiliarySolveMode : std::uint8_t {
    /// Advanced independently via a local stepper between PDE steps.
    /// Supports local time-advancement method selection, substepping,
    /// and related staggered-advance policies.
    Partitioned,

    /// Participates as first-class unknowns in assembled residual/Jacobian
    /// systems.  Uses auxiliary-specific unknown layouts composed into a
    /// mixed system layout.  Does NOT use an independent local stepper;
    /// time discretization is owned by the global assembled solve.
    Monolithic
};

// ---------------------------------------------------------------------------
//  History
// ---------------------------------------------------------------------------

/// History retention depth for an auxiliary block (block-wide in phase 1).
enum class AuxiliaryHistoryMode : std::uint8_t {
    /// No history retained beyond committed state.
    None,

    /// One step of history (committed + one prior).
    SingleStep,

    /// Multiple steps of history for multi-step methods.
    MultiStep
};

/// Interpolation policy for off-grid history access.
enum class AuxiliaryHistoryInterpolationPolicy : std::uint8_t {
    /// No interpolation; only step-aligned access.
    None,

    /// Linear interpolation between time-stamped history snapshots.
    Linear,

    /// Formulation provides a custom interpolation callback.
    FormulationDefined
};

// ---------------------------------------------------------------------------
//  Layout
// ---------------------------------------------------------------------------

/// Memory layout mode for auxiliary block storage.
enum class AuxiliaryLayoutMode : std::uint8_t {
    /// Fixed number of components per entity (default).
    FixedStride,

    /// Variable number of components per entity with canonical per-entity
    /// offsets.  Grouped or archetyped fast paths are an internal
    /// optimization, not public API.
    Ragged
};

/// Entity-component ordering within flat storage.
enum class AuxiliaryEntityOrdering : std::uint8_t {
    /// [entity0_comp0, entity0_comp1, ..., entity1_comp0, ...] (default).
    ByEntityThenComponent,

    /// [comp0_entity0, comp0_entity1, ..., comp1_entity0, ...].
    ByComponentThenEntity
};

// ---------------------------------------------------------------------------
//  Synchronization and transfer
// ---------------------------------------------------------------------------

/// MPI synchronization policy for distributed auxiliary blocks.
enum class AuxiliarySyncPolicy : std::uint8_t {
    /// No synchronization (Global scope, or serial runs).
    None,

    /// Only owned entities participate; no ghost exchange.
    OwnedOnly,

    /// Full owned + ghost synchronization.
    OwnedAndGhost
};

/// Data transfer policy for mesh adaptation or repartitioning.
enum class AuxiliaryTransferPolicy : std::uint8_t {
    /// No transfer support; data is discarded on remesh.
    None,

    /// Interpolate from old mesh to new mesh.
    Interpolate,

    /// Copy from nearest entity on old mesh.
    CopyNearest,

    /// Formulation provides a custom transfer callback.
    FormulationDefined
};

// ---------------------------------------------------------------------------
//  Scheduling
// ---------------------------------------------------------------------------

/**
 * @brief Advancement schedule for an auxiliary block relative to the PDE step.
 *
 * `schedule(...)` selects advancement timing and rate.
 * `stepper(...)` (on the deployed instance) selects the numerical
 * integration method used when advancement executes.
 */
enum class AuxiliaryScheduleMode : std::uint8_t {
    /// Advances once per PDE time step.
    SingleRate,

    /// Advances multiple substeps per PDE time step.
    Subcycled,

    /// Participates in a multirate scheme with independent rate selection.
    Multirate
};

// ---------------------------------------------------------------------------
//  Event / nonsmooth hooks
// ---------------------------------------------------------------------------

/// Extension hooks for event detection and nonsmooth behavior.
enum class AuxiliaryEventMode : std::uint8_t {
    /// No event hooks.
    None,

    /// Formulation provides event functions and state-reset hooks.
    EventHook,

    /// Formulation provides active-set detection hooks.
    ActiveSetHook,

    /// Formulation provides complementarity condition hooks.
    ComplementarityHook
};

// ---------------------------------------------------------------------------
//  Derivative infrastructure
// ---------------------------------------------------------------------------

/// Source for automatically generated derivatives (Jacobians, Hessians).
enum class AuxiliaryDerivativeSource : std::uint8_t {
    /// Differentiate lowered residual FormExpr trees symbolically (default
    /// when expressions are available).
    Symbolic,

    /// Use finite differencing (fallback for custom models without expressions).
    FiniteDifference,

    /// Model provides analytic Jacobian via evaluateJacobian().
    Analytic
};

/// What second-derivative information, if any, is requested.
enum class AuxiliarySecondDerivativeMode : std::uint8_t {
    /// No second derivatives.
    None,

    /// Full dense Hessian.
    Hessian,

    /// Hessian-vector products only (matrix-free).
    HessianVectorProduct,

    /// Selected second-derivative blocks (sparse).
    SelectedBlocks
};

// ---------------------------------------------------------------------------
//  Partitioned stepper policies
// ---------------------------------------------------------------------------

/**
 * @brief Input refresh policy for local substepping (Partitioned only).
 *
 * Controls whether externally-supplied auxiliary inputs are refreshed
 * between local substeps.
 */
enum class AuxiliaryInputRefreshPolicy : std::uint8_t {
    /// Hold the last sampled value for the entire outer step.
    HoldLastSample,

    /// Re-evaluate inputs at each local substep.
    RefreshEachSubstep,

    /// Formulation provides a custom refresh callback.
    FormulationDefined
};

/**
 * @brief Commit policy for local substepping (Partitioned only).
 *
 * Controls when substep results become visible to the rest of the system.
 */
enum class AuxiliarySubstepCommitPolicy : std::uint8_t {
    /// Only the final substep result is committed.
    CommitAtEnd,

    /// Each substep result is committed immediately.
    CommitEachSubstep,

    /// Formulation provides a custom commit/rollback policy.
    FormulationDefined
};

// ---------------------------------------------------------------------------
//  Output state view
// ---------------------------------------------------------------------------

/**
 * @brief Which state view an auxiliary output is evaluated against.
 *
 * Output evaluation always occurs against an explicit state view.
 * This avoids ambiguity about which version of auxiliary state
 * (committed, in-progress, mid-substep, etc.) the output represents.
 */
enum class AuxiliaryOutputStateView : std::uint8_t {
    /// Last committed time-step state.
    Committed,

    /// Current work buffer (may be mid-iteration).
    Work,

    /// Current stage in a multi-stage time integrator.
    Stage,

    /// Current nonlinear iterate.
    NonlinearIterate,

    /// Current local substep within partitioned advancement.
    Substep
};

// ---------------------------------------------------------------------------
//  Deployment region
// ---------------------------------------------------------------------------

/**
 * @brief Kind of region that restricts where auxiliary state is deployed.
 *
 * Deployment region is orthogonal to storage scope: scope defines the
 * entity type (node, cell, etc.), while region defines which subset of
 * those entities has auxiliary storage.
 */
enum class AuxiliaryRegionKind : std::uint8_t {
    /// Entire domain (no restriction).
    WholeDomain,

    /// Selected cells by set or marker.
    CellSet,

    /// Selected boundary faces/edges by marker.
    BoundarySet,

    /// Selected cells by material ID.
    MaterialIdSet,

    /// Selected interface entities.
    InterfaceSet,

    /// Formulation provides a custom entity selector.
    FormulationDefined
};

/**
 * @brief Deployment region descriptor.
 *
 * Specifies which subset of the mesh receives auxiliary storage.
 */
struct AuxiliaryDeploymentRegion {
    /// Kind of region restriction.
    AuxiliaryRegionKind kind{AuxiliaryRegionKind::WholeDomain};

    /// Stable identity token for the region (e.g., marker ID, set name).
    std::string identity{};

    /// Optional version or schema hash for the region definition.
    std::string version{};

    /// Explicit entity indices for FormulationDefined regions.
    /// When non-empty, this is the authoritative entity set regardless
    /// of `kind`.  For mesh-marker-based kinds (CellSet, BoundarySet,
    /// MaterialIdSet, InterfaceSet), the entity expansion uses the
    /// marker + mesh topology; the explicit set is a fallback or
    /// override for callers without mesh access.
    std::vector<std::size_t> explicit_entities{};

    /// Whether this region restricts to a subset (kind != WholeDomain
    /// or explicit_entities is non-empty).
    [[nodiscard]] bool isRestricted() const noexcept
    {
        return kind != AuxiliaryRegionKind::WholeDomain ||
               !explicit_entities.empty();
    }
};

// ---------------------------------------------------------------------------
//  Failure policy
// ---------------------------------------------------------------------------

/**
 * @brief Policy for handling failures during auxiliary state advancement.
 */
struct AuxiliaryFailurePolicy {
    /// Maximum retries for recoverable local solve failures.
    int max_local_retries{3};

    /// Whether a local failure should trigger global time-step rejection.
    bool reject_timestep_on_failure{true};

    /// Whether singular Jacobians should be treated as fatal.
    bool fatal_on_singular_jacobian{false};

    /// Whether event-localization failure should be treated as fatal.
    bool fatal_on_event_failure{false};
};

// ---------------------------------------------------------------------------
//  Composite spec structs
// ---------------------------------------------------------------------------

/**
 * @brief Derivative policy for an auxiliary model.
 *
 * Configured at the model level in phase 1; deployed instances inherit the
 * resolved policy.
 *
 * Precedence rules:
 * 1. Analytic override (if provided and `analytic_override_enabled` is true).
 * 2. Explicit model-level policy (`jacobian_source`, `second_deriv_source`).
 * 3. Default: symbolic generation when lowered residual expressions are
 *    available; setup failure with a clear diagnostic otherwise.
 */
struct AuxiliaryDerivativePolicy {
    /// Default derivative source for Jacobians.
    AuxiliaryDerivativeSource jacobian_source{AuxiliaryDerivativeSource::Symbolic};

    /// Default derivative source for requested second derivatives.
    AuxiliaryDerivativeSource second_deriv_source{AuxiliaryDerivativeSource::Symbolic};

    /// What second-derivative information is requested.
    AuxiliarySecondDerivativeMode second_deriv_mode{AuxiliarySecondDerivativeMode::None};

    /// Whether user-provided analytic derivatives override automatic
    /// generation.  When true and analytic derivatives are supplied,
    /// they take precedence over `jacobian_source`.
    bool analytic_override_enabled{true};

    /// Perturbation size for finite-difference derivatives (when selected).
    double fd_epsilon{1.0e-7};

    /// Seed dimension for AD (0 = auto-detect from block size).
    std::size_t ad_seed_dim{0};
};

/**
 * @brief Stepper configuration for a Partitioned auxiliary block.
 *
 * Local stepper selection, nonlinear-solver options, substep input-refresh
 * policy, and substep commit policy are valid only for `Partitioned` solve
 * mode.  `Monolithic` instances must not use this configuration surface.
 */
struct AuxiliaryStepperSpec {
    /// Stable name of the time-integration method (e.g. "BackwardEuler",
    /// "RK4", "BDF2").
    std::string method_name{"BackwardEuler"};

    /// Arbitrary method-specific options (variant payload).
    std::unordered_map<std::string, double> method_options{};

    /// How auxiliary inputs are refreshed during substepping.
    AuxiliaryInputRefreshPolicy input_refresh{AuxiliaryInputRefreshPolicy::HoldLastSample};

    /// When substep results become visible.
    AuxiliarySubstepCommitPolicy commit_policy{AuxiliarySubstepCommitPolicy::CommitAtEnd};

    /// Maximum nonlinear iterations for implicit local solves.
    int max_nonlinear_iters{50};

    /// Absolute tolerance for implicit local solves.
    double nonlinear_tol_abs{1.0e-12};

    /// Relative tolerance for implicit local solves.
    double nonlinear_tol_rel{1.0e-10};

    /// Number of substeps per outer PDE step (1 = no substepping).
    int substep_count{1};
};

/**
 * @brief Generalized specification for an auxiliary state block.
 *
 * This replaces the older boundary-specific `AuxiliaryStateSpec` as the
 * canonical registration descriptor.  Block identity is the `name` string,
 * unique within an FESystem.
 *
 * The spec is physics-agnostic: it describes storage layout, scope,
 * scheduling, and derivative policy without reference to any particular
 * physical model.
 */
struct AuxiliaryStateSpec {
    /// Stable block name — the durable public handle.  Must be unique
    /// within an FESystem.  Numeric block ids and slot ids are internal.
    std::string name{};

    /// Number of components (state variables) in this block.
    int size{0};

    /// Optional human-readable names for individual components.
    /// If non-empty, must have exactly `size` entries.
    std::vector<std::string> component_names{};

    /// Per-component classification as differential or algebraic.
    /// If empty, all components default to `Differential`.
    /// If non-empty, must have exactly `size` entries.
    std::vector<AuxiliaryVariableKind> variable_kinds{};

    /// Storage scope (Global, Node, Cell, QuadraturePoint, BoundaryEntity).
    AuxiliaryStateScope scope{AuxiliaryStateScope::Global};

    /// Solve mode (Partitioned or Monolithic).
    AuxiliarySolveMode solve_mode{AuxiliarySolveMode::Partitioned};

    /// Memory layout mode.
    AuxiliaryLayoutMode layout_mode{AuxiliaryLayoutMode::FixedStride};

    /// Entity-component ordering within flat storage.
    AuxiliaryEntityOrdering ordering{AuxiliaryEntityOrdering::ByEntityThenComponent};

    /// History retention mode.
    AuxiliaryHistoryMode history_mode{AuxiliaryHistoryMode::SingleStep};

    /// Maximum number of history snapshots (block-wide in phase 1).
    /// Only meaningful when `history_mode` is `MultiStep`.
    int history_depth{1};

    /// Interpolation policy for off-grid history access.
    AuxiliaryHistoryInterpolationPolicy history_interpolation{
        AuxiliaryHistoryInterpolationPolicy::None};

    /// MPI synchronization policy.
    AuxiliarySyncPolicy sync_policy{AuxiliarySyncPolicy::None};

    /// Data transfer policy for mesh adaptation.
    AuxiliaryTransferPolicy transfer_policy{AuxiliaryTransferPolicy::None};

    /// Advancement schedule relative to PDE steps.
    AuxiliaryScheduleMode schedule_mode{AuxiliaryScheduleMode::SingleRate};

    /// Event/nonsmooth extension hooks.
    AuxiliaryEventMode event_mode{AuxiliaryEventMode::None};

    /// Derivative policy for Jacobians and optional second derivatives.
    AuxiliaryDerivativePolicy derivative_policy{};

    /// Deployment region (restricts which entities get auxiliary storage).
    AuxiliaryDeploymentRegion deployment_region{};

    /// Failure handling policy for local solve failures.
    AuxiliaryFailurePolicy failure_policy{};

    /// Optional formulation-owned metadata (key-value pairs).
    std::unordered_map<std::string, std::string> metadata{};
};

/**
 * @brief Finalized layout information for an auxiliary block after setup.
 *
 * Block ids and slot ids are internal, setup-stable identifiers —
 * not part of the public API contract.
 */
struct AuxiliaryStateBlockLayout {
    /// Internal block id (setup-stable, not public).
    std::uint32_t block_id{0};

    /// Component stride (= spec.size for fixed-stride).
    int component_stride{0};

    /// Number of entities in this block on this rank.
    std::size_t entity_count{0};

    /// Total local storage size in Real values (entity_count * stride).
    std::size_t local_storage_size{0};

    /// Number of owned entities (for distributed scopes).
    std::size_t owned_entity_count{0};

    /// Owned storage size in Real values.
    std::size_t owned_storage_size{0};

    /// Total history storage size in Real values across all retained steps.
    std::size_t history_storage_size{0};
};

/**
 * @brief Summary of total auxiliary state storage (for debugging / tests).
 */
struct AuxiliaryStateStorageSummary {
    /// Number of registered blocks.
    std::size_t block_count{0};

    /// Total work-buffer storage in Real values.
    std::size_t total_work_storage{0};

    /// Total committed-buffer storage in Real values.
    std::size_t total_committed_storage{0};

    /// Total history storage in Real values.
    std::size_t total_history_storage{0};
};

/**
 * @brief Options that control block registration behavior.
 */
struct AuxiliaryStateRegistrationOptions {
    /// If true, registration of a block with a duplicate name is an error.
    /// If false, the existing block is returned (must have compatible spec).
    bool error_on_duplicate{true};

    /// If true, validate that component_names and variable_kinds sizes
    /// match spec.size during registration.
    bool validate_sizes{true};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_AUXILIARY_STATE_TYPES_H
