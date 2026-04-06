#ifndef SVMP_FE_AUXILIARY_INPUT_REGISTRY_H
#define SVMP_FE_AUXILIARY_INPUT_REGISTRY_H

/**
 * @file AuxiliaryInputRegistry.h
 * @brief Generalized registry for externally supplied auxiliary input values.
 *
 * Auxiliary inputs are named, typed values that flow into auxiliary-state
 * model evaluation.  The registry provides a neutral abstraction:
 * boundary functionals are only one input-provider implementation, not
 * the core abstraction.
 *
 * ## Provider types
 *
 * | Producer                  | Description                                       |
 * |---------------------------|---------------------------------------------------|
 * | `BoundaryReduction`       | Boundary-integrated scalar (replaces direct        |
 * |                           | BoundaryFunctional coupling).                     |
 * | `FormulationCallback`     | Formulation supplies a callable.                  |
 * | `ParameterDerived`        | Computed from system parameters.                  |
 * | `DirectUserData`          | User sets value explicitly.                       |
 * | `AuxiliaryOutput`         | Produced from another auxiliary model's output.   |
 * | `SampledStateField`       | Sampled from an FE field (committed/iterate/etc.).|
 * | `CoupledField`            | Symbolic FE field dependency (monolithic).        |
 * | `CellAverage`             | Cell-averaged FE field quantity.                  |
 * | `CellSample`              | Point-sampled FE field within a cell.             |
 * | `DomainAverage`           | Domain-averaged FE field quantity.                |
 * | `DomainIntegral`          | Domain-integrated FE field quantity.              |
 * | `SampledBoundaryTrace`    | FE field sampled on a boundary.                   |
 * | `CoupledBoundaryTrace`    | Symbolic boundary field dependency (monolithic).  |
 * | `SampledBoundaryReduction`| Boundary-reduced FE field quantity (explicit).    |
 * | `CoupledBoundaryReduction`| Symbolic boundary reduction (monolithic).         |
 *
 * ## Stage selection
 *
 * When a sampled field input is taken from FE system state, the caller
 * must specify which stage of the solution to sample:
 * - `Committed` — the last committed time-step state.
 * - `PreviousStep` — the state from the prior time step.
 * - `CurrentIterate` — the current nonlinear iterate.
 * - `StageState` — the current stage in a multi-stage method.
 *
 * ## Scope validity
 *
 * | Provider                  | Global | Node | Cell | QP  | BndEnt |
 * |---------------------------|--------|------|------|-----|--------|
 * | BoundaryReduction         |  yes   |  no  |  no  |  no |  no    |
 * | FormulationCallback       |  yes   |  yes |  yes |  yes|  yes   |
 * | ParameterDerived          |  yes   |  yes |  yes |  yes|  yes   |
 * | DirectUserData            |  yes   |  yes |  yes |  yes|  yes   |
 * | AuxiliaryOutput           |  yes   |  yes |  yes |  yes|  yes   |
 * | SampledStateField         |  no    |  yes |  yes |  yes|  no    |
 * | CoupledField              |  no    |  yes |  yes |  yes|  no    |
 * | CellAverage               |  no    |  no  |  yes |  yes|  no    |
 * | CellSample                |  no    |  no  |  yes |  yes|  no    |
 * | DomainAverage             |  yes   |  no  |  no  |  no |  no    |
 * | DomainIntegral            |  yes   |  no  |  no  |  no |  no    |
 * | SampledBoundaryTrace      |  no    |  no  |  no  |  no |  yes   |
 * | CoupledBoundaryTrace      |  no    |  no  |  no  |  no |  yes   |
 * | SampledBoundaryReduction  |  yes   |  no  |  no  |  no |  no    |
 * | CoupledBoundaryReduction  |  yes   |  no  |  no  |  no |  no    |
 *
 * `Global` scope requires explicit reductions — no implicit field-to-scalar
 * collapse.  `CoupledField` and `CoupledBoundaryTrace` lower into symbolic
 * field dependencies for `Monolithic` auxiliary blocks rather than frozen
 * cached inputs.
 *
 * ## Invalidation and caching
 *
 * Inputs are evaluated once per time step by default.  Within nonlinear
 * iterations, `CoupledField` inputs are re-evaluated (they track the
 * current iterate), while `Sampled*` inputs hold their last-sampled value.
 * The `AuxiliaryInputRefreshPolicy` on the consuming block controls
 * whether inputs are refreshed during local substepping.
 *
 * ## Dependency ordering
 *
 * If input A depends on input B (e.g., A is an AuxiliaryOutput that
 * consumes B), the registry resolves evaluation order topologically.
 */

#include "Core/Types.h"
#include "Core/Alignment.h"
#include "Core/AlignedAllocator.h"
#include "Core/FEException.h"

#include "Auxiliary/AuxiliaryStateTypes.h"
#include "Systems/SystemsExceptions.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Producer type
// ---------------------------------------------------------------------------

/**
 * @brief Classification of how an auxiliary input is produced.
 */
enum class AuxiliaryInputProducer : std::uint8_t {
    /// Boundary-integrated scalar (MPI-reduced).
    BoundaryReduction,

    /// Formulation-supplied callback.
    FormulationCallback,

    /// Computed from system parameters.
    ParameterDerived,

    /// User sets value explicitly.
    DirectUserData,

    /// Produced from another auxiliary model's output.
    AuxiliaryOutput,

    // -- Field-binding providers --

    /// Sampled from a committed/iterate FE field.
    SampledStateField,

    /// Symbolic FE field dependency for monolithic blocks.
    CoupledField,

    /// Cell-averaged FE field quantity.
    CellAverage,

    /// Point-sampled FE field within a cell.
    CellSample,

    /// Domain-averaged FE field quantity.
    DomainAverage,

    /// Domain-integrated FE field quantity.
    DomainIntegral,

    // -- Boundary-binding providers --

    /// FE field sampled on a boundary (pointwise/entity-local).
    SampledBoundaryTrace,

    /// Symbolic boundary field dependency for monolithic blocks.
    CoupledBoundaryTrace,

    /// Boundary-reduced FE field quantity (explicit sampling).
    SampledBoundaryReduction,

    /// Symbolic boundary reduction for monolithic blocks.
    CoupledBoundaryReduction
};

// ---------------------------------------------------------------------------
//  Stage selection
// ---------------------------------------------------------------------------

/**
 * @brief Which stage of the FE solution to sample for field-binding inputs.
 */
enum class AuxiliaryFieldStage : std::uint8_t {
    /// Last committed time-step state.
    Committed,

    /// State from the prior time step.
    PreviousStep,

    /// Current nonlinear iterate.
    CurrentIterate,

    /// Current stage in a multi-stage time integrator.
    StageState
};

// ---------------------------------------------------------------------------
//  Update schedule
// ---------------------------------------------------------------------------

/**
 * @brief When an input is re-evaluated.
 */
enum class AuxiliaryInputUpdateSchedule : std::uint8_t {
    /// Once per time step (default).
    OncePerTimeStep,

    /// Once per nonlinear iteration.
    EachNonlinearIteration,

    /// Once at setup time (constant).
    OnceAtSetup,

    /// Manually triggered by the formulation.
    Manual
};

// ---------------------------------------------------------------------------
//  Input spec
// ---------------------------------------------------------------------------

/**
 * @brief Specification for one auxiliary input.
 */
struct AuxiliaryInputSpec {
    /// Stable input name — the public handle.
    std::string name{};

    /// Number of scalar components in this input.
    int size{1};

    /// Optional human-readable component names.
    std::vector<std::string> component_names{};

    /// How this input is produced.
    AuxiliaryInputProducer producer{AuxiliaryInputProducer::DirectUserData};

    /// When this input is re-evaluated.
    AuxiliaryInputUpdateSchedule update_schedule{
        AuxiliaryInputUpdateSchedule::OncePerTimeStep};

    /// For field-binding producers: which solution stage to sample.
    AuxiliaryFieldStage field_stage{AuxiliaryFieldStage::CurrentIterate};

    /// For BoundaryReduction/SampledBoundaryReduction: boundary marker.
    int boundary_marker{-1};

    /// Name of the source auxiliary output (for AuxiliaryOutput producer).
    std::string source_output_name{};

    /// Source FE field name (for SampledStateField, SampledBoundaryTrace,
    /// SampledBoundaryReduction, CellAverage, DomainAverage producers).
    std::string source_field_name{};

    /// Whether this input requires MPI reduction.
    bool requires_mpi_reduction{false};

    /// Number of entities for entity-local inputs (0 = global/scalar).
    /// When > 0, the flat storage is `entity_count * size` values,
    /// laid out as [entity0_comp0, entity0_comp1, ..., entity1_comp0, ...].
    std::size_t entity_count{0};
};

// ---------------------------------------------------------------------------
//  Input provider callbacks
// ---------------------------------------------------------------------------

/**
 * @brief Callback that produces global (non-entity-local) input values.
 *
 * Arguments: (time, dt, output_buffer)
 * The callback writes `spec.size` values into `output_buffer`.
 */
using AuxiliaryInputCallback = std::function<void(
    Real time, Real dt, std::span<Real> output)>;

/**
 * @brief Callback that produces entity-local input values.
 *
 * Arguments: (time, dt, entity_index, output_buffer)
 * The callback writes `spec.size` values for one entity.
 * Called once per entity during evaluation.
 */
using AuxiliaryEntityInputCallback = std::function<void(
    Real time, Real dt, std::size_t entity_index, std::span<Real> output)>;

// ---------------------------------------------------------------------------
//  Registry
// ---------------------------------------------------------------------------

/**
 * @brief Registry for auxiliary input values.
 *
 * Owns the value storage for all registered inputs.  Provides:
 * - Name-based registration and lookup
 * - Stable slot assignment
 * - Invalidation and refresh lifecycle
 * - Dependency-ordered evaluation
 * - Debug inspection
 */
class AuxiliaryInputRegistry {
public:
    AuxiliaryInputRegistry() = default;

    // -----------------------------------------------------------------
    //  Registration
    // -----------------------------------------------------------------

    /**
     * @brief Register a new input.
     *
     * @param spec     Input specification.
     * @param callback Optional evaluation callback.
     * @return Slot index for this input's first component in the flat
     *         value array.
     */
    std::size_t registerInput(const AuxiliaryInputSpec& spec,
                              AuxiliaryInputCallback callback = {});

    /**
     * @brief Register an entity-local input.
     *
     * Storage is `spec.entity_count * spec.size` values.
     * The callback is invoked once per entity during evaluation.
     */
    std::size_t registerEntityInput(const AuxiliaryInputSpec& spec,
                                     AuxiliaryEntityInputCallback callback);

    // -----------------------------------------------------------------
    //  Access
    // -----------------------------------------------------------------

    /// Number of registered inputs.
    [[nodiscard]] std::size_t inputCount() const noexcept { return entries_.size(); }

    /// Total number of scalar values across all inputs.
    [[nodiscard]] std::size_t totalSize() const noexcept { return values_.size(); }

    /// Whether an input with the given name exists.
    [[nodiscard]] bool hasInput(std::string_view name) const noexcept;

    /// Get the slot (offset into flat array) for an input by name.
    [[nodiscard]] std::size_t slotOf(std::string_view name) const;

    /// Get the spec for an input by name.
    [[nodiscard]] const AuxiliaryInputSpec& specOf(std::string_view name) const;

    /// Read-only view of all values (flat, slot-indexed).
    [[nodiscard]] std::span<const Real> all() const noexcept { return values_; }

    /// Read-only view of a single input's values (all entities for entity-local).
    [[nodiscard]] std::span<const Real> valuesOf(std::string_view name) const;

    /// Read-only view of a single entity's values for an entity-local input.
    /// Returns spec.size values for entity `entity_index`.
    /// For global inputs, entity_index is ignored and the full values are returned.
    [[nodiscard]] std::span<const Real> valuesOf(std::string_view name,
                                                  std::size_t entity_index) const;

    /// Whether an input is entity-local (entity_count > 0).
    [[nodiscard]] bool isEntityLocal(std::string_view name) const;

    /// Mutable view of a single input's values (for DirectUserData / callbacks).
    [[nodiscard]] std::span<Real> mutableValuesOf(std::string_view name);

    /// Get value of a scalar input by name (convenience for size==1).
    [[nodiscard]] Real get(std::string_view name) const;

    /// Set value of a scalar input by name (convenience for size==1).
    void set(std::string_view name, Real value);

    /// Get all input names in registration order.
    [[nodiscard]] std::vector<std::string> inputNames() const;

    // -----------------------------------------------------------------
    //  Evaluation lifecycle
    // -----------------------------------------------------------------

    /**
     * @brief Evaluate all inputs that are due for refresh.
     *
     * Calls registered callbacks in dependency order.
     * Inputs with `OnceAtSetup` schedule are skipped after first eval.
     * Inputs with `Manual` schedule are only evaluated if explicitly
     * marked dirty.
     *
     * @param time Current simulation time.
     * @param dt   Current time step.
     * @param is_nonlinear_iteration If true, also refresh inputs with
     *        `EachNonlinearIteration` schedule.
     */
    void evaluate(Real time, Real dt, bool is_nonlinear_iteration = false);

    /**
     * @brief Mark an input as dirty (needing re-evaluation).
     *
     * Used for Manual-schedule inputs.
     */
    void markDirty(std::string_view name);

    /**
     * @brief Mark all inputs as needing re-evaluation.
     *
     * Called at the start of each time step.
     */
    void invalidateAll();

    /**
     * @brief Clear all registrations and values.
     */
    void clear();

    // -----------------------------------------------------------------
    //  Dependency ordering
    // -----------------------------------------------------------------

    /**
     * @brief Declare that input `dependent` depends on input `dependency`.
     *
     * The registry will ensure `dependency` is evaluated before `dependent`.
     */
    void addDependency(std::string_view dependent, std::string_view dependency);

    /**
     * @brief Get the topologically sorted evaluation order.
     *
     * Returns input indices in the order they should be evaluated.
     */
    [[nodiscard]] std::vector<std::size_t> evaluationOrder() const;

    // -----------------------------------------------------------------
    //  Debug inspection
    // -----------------------------------------------------------------

    /**
     * @brief Print all input names, slots, values, and states to a string.
     */
    [[nodiscard]] std::string debugDump() const;

private:
    struct InputEntry {
        AuxiliaryInputSpec spec{};
        AuxiliaryInputCallback callback{};
        AuxiliaryEntityInputCallback entity_callback{}; ///< For entity-local inputs.
        std::size_t slot{0};           ///< Offset into values_ array.
        bool evaluated_at_setup{false}; ///< For OnceAtSetup inputs.
        bool dirty{true};              ///< Needs re-evaluation.
        std::vector<std::size_t> depends_on{}; ///< Input indices this depends on.
    };

    std::vector<InputEntry> entries_{};
    std::unordered_map<std::string, std::size_t> name_to_index_{};

    /// Flat value storage (slot-indexed, SIMD-aligned).
    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> values_{};

    [[nodiscard]] std::size_t entryIndex(std::string_view name) const;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_INPUT_REGISTRY_H
