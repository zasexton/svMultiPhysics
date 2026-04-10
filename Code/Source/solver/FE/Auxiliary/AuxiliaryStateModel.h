#ifndef SVMP_FE_AUXILIARY_STATE_MODEL_H
#define SVMP_FE_AUXILIARY_STATE_MODEL_H

/**
 * @file AuxiliaryStateModel.h
 * @brief DAE-capable auxiliary model interface for residual-based local systems.
 *
 * This replaces the scalar-RHS contract with a residual-based interface
 * that can express ODE-like and DAE-like local systems:
 *
 *     F(dot(x), x, history, inputs, t, dt) = 0
 *
 * ## Key design decisions
 *
 * - `xdot` is always available in the residual interface.  For algebraic
 *   rows, `xdot` entries are zero and the residual is purely `g(x, ...) = 0`.
 * - Mixed blocks: differential and algebraic rows may coexist.
 *   `AuxiliaryStructuralMetadata::variable_kinds` classifies each row.
 * - Jacobian hooks are optional.  If not provided, the derivative provider
 *   generates them (symbolic, AD, or FD) according to the derivative policy.
 * - Hessian hooks are optional and demand-driven.  Models need not provide
 *   Hessians unless a solver or coupling path explicitly requests them.
 * - Consistent initialization hooks handle algebraic-variable initialization.
 * - Event and nonsmooth hooks extend the interface beyond smooth DAEs.
 *
 * ## Canonical lowering from math-first builder rows
 *
 * - `ode(x, rhs)` → residual row: `dot(x) - rhs`
 * - `algebraic(z, expr)` → residual row: `expr`
 * - `residual(name, expr)` → raw residual row: `expr`
 */

#include "Core/Types.h"

#include "Auxiliary/AuxiliaryStateTypes.h"

#include "Forms/FormExpr.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Context passed to model evaluations
// ---------------------------------------------------------------------------

/**
 * @brief Runtime context for auxiliary model evaluation.
 *
 * Contains everything a residual/Jacobian evaluation needs:
 * time, time step, state, time derivatives, history, inputs, params.
 */
struct AuxiliaryLocalContext {
    /// Current simulation time.
    Real time{0.0};

    /// Outer PDE time step.
    Real dt{0.0};

    /// Effective step size for the local stepper (may differ from dt
    /// if substepping is active).
    Real effective_dt{0.0};

    /// Current state vector x for this block.
    std::span<const Real> x{};

    /// Time derivative dot(x) for this block (zero for algebraic rows).
    std::span<const Real> xdot{};

    /// History views: history[k] is (k+1)-th prior committed state.
    /// Empty if no history is available.
    std::span<const std::span<const Real>> history{};

    /// Auxiliary input values (from input providers).
    std::span<const Real> inputs{};

    /// Parameter values (constants or provider-driven).
    std::span<const Real> params{};

    /// Entity index for per-entity models (Node, Cell, QP, Facet).
    /// Ignored for Global scope (always 0).
    std::size_t entity_index{0};

    /// FE field values at the entity location, for models that directly
    /// reference DiscreteField/StateField nodes in their expressions.
    /// Each entry carries a FieldId with 1+ component values (scalar,
    /// vector, or tensor).  Populated by the assembly path when the
    /// derivative provider's artifact references FE fields.
    std::span<const FieldValueEntry> field_values{};

    /// Optional formulation user data.
    const void* user_data{nullptr};
};

// ---------------------------------------------------------------------------
//  Request structs
// ---------------------------------------------------------------------------

/**
 * @brief What to compute during residual evaluation.
 */
struct AuxiliaryResidualRequest {
    /// Output buffer for residual F.  Must be pre-sized to block.size.
    std::span<Real> residual{};
};

/**
 * @brief What to compute during Jacobian evaluation.
 */
struct AuxiliaryJacobianRequest {
    /// dF/dx — row-major dense, size = n × n.
    std::span<Real> dF_dx{};

    /// dF/d(dot(x)) — row-major dense, size = n × n.
    /// Only meaningful for differential rows.
    std::span<Real> dF_dxdot{};

    /// Whether dF_dxdot is requested.
    bool want_dF_dxdot{false};

    /// dF/d(inputs) — row-major dense, size = n × n_inputs.
    /// Optional: only computed if this span is non-empty.
    std::span<Real> dF_dinputs{};

    /// Block dimension (n = number of state components).
    int n{0};

    /// Number of auxiliary inputs.
    int n_inputs{0};
};

/**
 * @brief What to compute for second-derivative information.
 */
struct AuxiliaryHessianRequest {
    /// Mode of second-derivative computation.
    AuxiliarySecondDerivativeMode mode{AuxiliarySecondDerivativeMode::None};

    /// For HessianVectorProduct: direction vector v (size n).
    std::span<const Real> direction{};

    /// Output: full Hessian (n × n × n tensor, row-major) when mode == Hessian.
    std::span<Real> hessian{};

    /// Output: Hessian-vector product (n × n) when mode == HessianVectorProduct.
    std::span<Real> hvp{};

    /// Output: selected second-derivative blocks when mode == SelectedBlocks.
    /// Layout depends on the specific blocks requested.
    std::span<Real> selected_blocks{};

    int n{0};
};

/**
 * @brief Request for consistent initialization of algebraic variables.
 */
struct AuxiliaryInitializationRequest {
    /// Initial state vector (differential variables already set).
    /// Algebraic variables should be solved for.
    std::span<Real> x{};

    /// Time for initialization.
    Real time{0.0};

    /// Auxiliary inputs at initialization time.
    std::span<const Real> inputs{};

    /// Parameters.
    std::span<const Real> params{};
};

// ---------------------------------------------------------------------------
//  Structural metadata
// ---------------------------------------------------------------------------

/**
 * @brief Structural metadata describing the local system.
 *
 * Allows advanced solvers to distinguish differential rows, algebraic
 * constraints, and solver hints.
 */
struct AuxiliaryStructuralMetadata {
    /// Per-row classification as Differential or Algebraic.
    /// Size == block dimension.
    std::vector<AuxiliaryVariableKind> variable_kinds{};

    /// Optional constraint grouping (e.g., for block-structured solvers).
    /// Each entry is a list of row indices forming one constraint group.
    std::vector<std::vector<int>> constraint_groups{};

    /// Optional differential index hint for DAE solvers.
    /// -1 means unspecified.
    int dae_index_hint{-1};

    /// Whether this model has event functions.
    bool has_events{false};

    /// Whether this model has nonsmooth/complementarity behavior.
    bool has_nonsmooth{false};

    /// Number of event functions (0 if !has_events).
    int n_event_functions{0};
};

// ---------------------------------------------------------------------------
//  Event / nonsmooth hooks
// ---------------------------------------------------------------------------

/**
 * @brief Result of event function evaluation.
 */
struct AuxiliaryEventResult {
    /// Event function values g_i(x, t).  Sign change triggers event.
    std::vector<Real> values{};

    /// Optional: which events are currently active.
    std::vector<bool> active{};
};

// ---------------------------------------------------------------------------
//  Abstract model interface
// ---------------------------------------------------------------------------

/**
 * @brief Abstract interface for an auxiliary state model.
 *
 * The residual function F(dot(x), x, ...) = 0 is the primary contract.
 * All other hooks (Jacobian, Hessian, initialization, events) are optional.
 *
 * ## Evaluation semantics
 *
 * - `evaluateResidual()`: REQUIRED.  Evaluate F for the given state.
 * - `evaluateJacobian()`: Optional.  If not overridden, the derivative
 *   provider generates Jacobians according to the derivative policy.
 * - `evaluateHessian()`: Optional.  Only called when explicitly requested.
 * - `initializeAlgebraic()`: Optional.  Called at t=0 or restart to find
 *   consistent algebraic variable values.
 * - `evaluateEvents()`: Optional.  Called during event detection.
 * - `resetAfterEvent()`: Optional.  Called after an event is detected.
 *
 * ## Expression-based models
 *
 * Models defined via math-first builder expressions carry `FormExpr` trees
 * for the residual rows.  The derivative provider differentiates these
 * symbolically at setup time.  The `residualExpressions()` method returns
 * these expressions for the provider.
 */
class AuxiliaryStateModel {
public:
    virtual ~AuxiliaryStateModel() = default;

    /// Human-readable model name (for diagnostics).
    [[nodiscard]] virtual std::string modelName() const = 0;

    /// Block dimension (number of state components).
    [[nodiscard]] virtual int dimension() const = 0;

    /// Structural metadata for this model.
    [[nodiscard]] virtual AuxiliaryStructuralMetadata structuralMetadata() const = 0;

    // -----------------------------------------------------------------
    //  Residual (REQUIRED)
    // -----------------------------------------------------------------

    /**
     * @brief Evaluate the residual F(dot(x), x, history, inputs, t, dt) = 0.
     *
     * @param ctx     Current evaluation context (state, time, inputs, etc.).
     * @param request Output buffer for residual values.
     */
    virtual void evaluateResidual(const AuxiliaryLocalContext& ctx,
                                  AuxiliaryResidualRequest& request) const = 0;

    // -----------------------------------------------------------------
    //  Jacobian (optional analytic override)
    // -----------------------------------------------------------------

    /**
     * @brief Whether this model provides analytic Jacobians.
     *
     * If true, `evaluateJacobian()` is used instead of automatic generation.
     */
    [[nodiscard]] virtual bool hasAnalyticJacobian() const { return false; }

    /**
     * @brief Evaluate the Jacobian analytically.
     *
     * Only called when `hasAnalyticJacobian()` returns true.
     */
    virtual void evaluateJacobian(const AuxiliaryLocalContext& ctx,
                                  AuxiliaryJacobianRequest& request) const
    {
        (void)ctx; (void)request;
    }

    // -----------------------------------------------------------------
    //  Hessian (optional)
    // -----------------------------------------------------------------

    [[nodiscard]] virtual bool hasAnalyticHessian() const { return false; }

    virtual void evaluateHessian(const AuxiliaryLocalContext& ctx,
                                 AuxiliaryHessianRequest& request) const
    {
        (void)ctx; (void)request;
    }

    // -----------------------------------------------------------------
    //  Consistent initialization (optional)
    // -----------------------------------------------------------------

    [[nodiscard]] virtual bool hasConsistentInitialization() const { return false; }

    virtual void initializeAlgebraic(AuxiliaryInitializationRequest& request) const
    {
        (void)request;
    }

    // -----------------------------------------------------------------
    //  Events (optional)
    // -----------------------------------------------------------------

    [[nodiscard]] virtual bool hasEventFunctions() const { return false; }

    virtual AuxiliaryEventResult evaluateEvents(const AuxiliaryLocalContext& ctx) const
    {
        (void)ctx;
        return {};
    }

    virtual void resetAfterEvent(const AuxiliaryLocalContext& ctx,
                                 int event_index,
                                 std::span<Real> x_new) const
    {
        (void)ctx; (void)event_index; (void)x_new;
    }

    // -----------------------------------------------------------------
    //  Nonsmooth / complementarity (optional)
    // -----------------------------------------------------------------

    [[nodiscard]] virtual bool hasNonsmoothHooks() const { return false; }

    /// Evaluate complementarity conditions.  Returns per-component values.
    virtual std::vector<Real> evaluateComplementarity(
        const AuxiliaryLocalContext& ctx) const
    {
        (void)ctx;
        return {};
    }

    // -----------------------------------------------------------------
    //  Expression access (for symbolic derivative generation)
    // -----------------------------------------------------------------

    /**
     * @brief Whether this model carries lowered residual FormExpr trees.
     *
     * When true, the derivative provider can differentiate these expressions
     * symbolically.  When false, the model must provide analytic Jacobians
     * or the provider must use AD/FD.
     */
    [[nodiscard]] virtual bool hasResidualExpressions() const { return false; }

    /**
     * @brief Get the lowered residual expressions (one per row).
     *
     * Only valid when `hasResidualExpressions()` returns true.
     */
    [[nodiscard]] virtual std::vector<forms::FormExpr> residualExpressions() const
    {
        return {};
    }

    // -----------------------------------------------------------------
    //  Mass-like metadata (optional, for DAE solvers)
    // -----------------------------------------------------------------

    /**
     * @brief Whether this model has a mass matrix M in M*dot(x) = f(x).
     *
     * When true, `massDiagonal()` or `massMatrix()` may be queried.
     * When false, the mass matrix is assumed to be the identity for
     * differential rows.
     */
    [[nodiscard]] virtual bool hasMassMatrix() const { return false; }

    /**
     * @brief Diagonal mass matrix entries (for lumped-mass models).
     *
     * Size == dimension().  Zero for algebraic rows.
     */
    [[nodiscard]] virtual std::vector<Real> massDiagonal() const { return {}; }

    // -----------------------------------------------------------------
    //  Outputs (optional)
    // -----------------------------------------------------------------

    /**
     * @brief Number of named outputs this model provides.
     *
     * Override to expose outputs for assembly/JIT consumption via
     * `AuxiliaryOutputRef(output_id)`.
     */
    [[nodiscard]] virtual int outputCount() const { return 0; }

    /**
     * @brief Names of the outputs, in declaration order.
     *
     * Size must equal `outputCount()`.
     */
    [[nodiscard]] virtual std::vector<std::string> outputNames() const { return {}; }

    /**
     * @brief Evaluate all outputs for the given context.
     *
     * @param ctx    Current evaluation context (state, time, inputs, etc.).
     * @param output Buffer of size `outputCount()` to write values into.
     */
    virtual void evaluateOutputs(const AuxiliaryLocalContext& ctx,
                                  std::span<Real> output) const
    {
        (void)ctx; (void)output;
    }

    // -----------------------------------------------------------------
    //  Input and parameter signatures (optional)
    // -----------------------------------------------------------------

    /**
     * @brief Declared input names in canonical order.
     *
     * Override to define the input ordering contract for this model.
     * When non-empty, FESystem uses this order (not lexicographic
     * binding-key order) to build the `ctx.inputs` vector.
     *
     * Each name should match a `.bind(model_input, registry_input)` key.
     * Entries may include a size suffix for multi-component inputs,
     * e.g., "velocity:3".
     */
    [[nodiscard]] virtual std::vector<std::string> declaredInputNames() const
    {
        return {};
    }

    /**
     * @brief Declared parameter names in canonical order.
     *
     * Override to define the parameter ordering contract.
     * When non-empty, FESystem uses this order to build the
     * `ctx.params` vector.
     */
    [[nodiscard]] virtual std::vector<std::string> declaredParameterNames() const
    {
        return {};
    }
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_STATE_MODEL_H
