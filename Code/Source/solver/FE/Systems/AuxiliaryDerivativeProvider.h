#ifndef SVMP_FE_SYSTEMS_AUXILIARY_DERIVATIVE_PROVIDER_H
#define SVMP_FE_SYSTEMS_AUXILIARY_DERIVATIVE_PROVIDER_H

/**
 * @file AuxiliaryDerivativeProvider.h
 * @brief Derivative supply for auxiliary state models.
 *
 * Resolves the derivative policy for an auxiliary model and supplies
 * Jacobians (and optional second derivatives) from the appropriate source:
 *
 * 1. **Analytic override** — if the model provides `hasAnalyticJacobian()`,
 *    the model's own `evaluateJacobian()` is used.
 * 2. **Symbolic** — if the model carries lowered `FormExpr` residual
 *    expressions, they are differentiated symbolically at setup time and
 *    the compiled result is reused at runtime.
 * 3. **FiniteDifference** — forward finite differencing of the residual
 *    callback (fallback for custom models without expressions).
 *
 * ## Symbolic coverage
 *
 * Symbolic differentiation supports all scalar operations relevant to
 * auxiliary state models (EP, metabolism, reduced-order, etc.):
 *
 * - **Arithmetic**: `+`, `-`, `*`, `/`, negate.
 * - **Transcendental**: `pow` (const and variable exponent), `sqrt`,
 *   `exp`, `log`.
 * - **Piecewise**: `abs`, `sign`, `min`, `max`, `conditional`,
 *   comparisons (`<`, `<=`, `>`, `>=`, `==`, `!=`).
 * - **Smooth approximations**: `smoothAbs`, `smoothSign`,
 *   `smoothHeaviside`, `smoothMin`, `smoothMax`.
 * - **Scalar tensor ops**: `inner`, `doubleContraction`, `outer`
 *   (product rule), `trace`, `transpose`, `det` (passthrough),
 *   `inv` (reciprocal rule), `norm` (abs derivative), `normalize`
 *   (zero), `sym` (passthrough).
 * - **History operators**: `HistoryWeightedSum`, `HistoryConvolution`
 *   (constant w.r.t. current state).
 * - **Terminals**: `AuxiliaryStateRef` (variable), all other terminals
 *   (inputs, parameters, time, constants, coordinates, etc.) treated
 *   as constants.
 *
 * FE-specific operations (gradient, divergence, test/trial functions)
 * and matrix-specific operations trigger FD fallback.  The reason is
 * stored in `artifact().fallback_reason` and logged to stderr.
 *
 * Implemented derivative targets:
 * - `dF/dx` — symbolic for all supported scalar expressions, FD fallback
 *   for unsupported ops.
 * - `dF/d(xdot)` — synthesized from row kind (identity for ODE, zero
 *   for algebraic).
 *
 * - `dF/d(inputs)` — symbolic for all supported scalar expressions,
 *   FD fallback for unsupported ops.  Generated when the model has
 *   input-dependent residual expressions.
 *
 * - `dF/d(fields)` — symbolic per-component differentiation for
 *   Node-scoped models that reference `DiscreteField`/`StateField` nodes.
 *   Supports scalar, vector, and tensor fields (up to 9 components).
 *   Requires C0-continuous (nodal Lagrange) spaces for the Kronecker
 *   delta property — scalar H1 and Product/Vector H1 spaces.
 *   Per-component derivatives via `DiffTarget::Field` with
 *   `target_component`: `component(u, i)` → δ_{ik}, `inner(u, u)` →
 *   2*u_k, product-rule compositions.  `FieldValueEntry` carries
 *   multi-component values per FieldId.  Non-Node scopes and non-C0
 *   spaces are rejected at setup.
 *
 * ## Setup-time generation
 *
 * For expression-defined models, symbolic derivatives are generated once
 * per finalized model instance during setup.  The resulting expressions
 * are evaluated at runtime via `PointEvaluator`.
 *
 * ## Workspace reuse
 *
 * The provider maintains per-block scratch buffers for Jacobian evaluation
 * to avoid per-call allocation.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Systems/AuxiliaryStateTypes.h"
#include "Systems/AuxiliaryStateModel.h"
#include "Systems/SystemsExceptions.h"

#include "Forms/FormExpr.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Derivative targets
// ---------------------------------------------------------------------------

/**
 * @brief Enumeration of what the derivative is taken with respect to.
 *
 * Implemented: State (symbolic + FD), TimeDeriv (synthesized),
 * Inputs (symbolic + FD), CoupledField (symbolic via DiffTarget::Field).
 */
enum class AuxiliaryDerivativeTarget : std::uint8_t {
    /// d/dx — auxiliary state variables.  Symbolic + FD.
    State,

    /// d/d(dot(x)) — time derivatives of state.  Synthesized from row kind.
    TimeDeriv,

    /// d/d(inputs) — auxiliary input sensitivities.  Symbolic + FD.
    Inputs,

    /// d/d(u) — coupled FE field variables (monolithic).  Symbolic via
    /// DiffTarget::Field for models with DiscreteField/StateField terminals.
    CoupledField
};

// ---------------------------------------------------------------------------
//  Cached derivative artifact
// ---------------------------------------------------------------------------

/**
 * @brief A cached, compiled derivative artifact for one model.
 *
 * Holds either symbolic FormExpr Jacobian expressions (to be evaluated at
 * runtime) or compiled function pointers, depending on the generation path.
 */
struct AuxiliaryDerivativeArtifact {
    /// The model name this artifact was generated for.
    std::string model_name{};

    /// Derivative source used.
    AuxiliaryDerivativeSource source{AuxiliaryDerivativeSource::Symbolic};

    /// Symbolic Jacobian expressions dF_i/dx_j (row-major, n×n).
    /// Non-empty when source == Symbolic and hasResidualExpressions.
    std::vector<forms::FormExpr> dF_dx_exprs{};

    /// Symbolic dF/d(dot(x)) expressions (n×n), if generated.
    std::vector<forms::FormExpr> dF_dxdot_exprs{};

    /// Whether this artifact has been successfully generated.
    bool valid{false};

    /// Per-row variable kind (Differential or Algebraic), stored for dF/dxdot.
    std::vector<AuxiliaryVariableKind> variable_kinds{};

    /// Symbolic dF/d(inputs) expressions (n_rows × n_inputs), row-major.
    /// Non-empty when source == Symbolic and the model has input-dependent
    /// residual expressions.
    std::vector<forms::FormExpr> dF_dinputs_exprs{};

    /// Number of input components for dF/d(inputs).
    int n_inputs{0};

    /// Symbolic d(output_k)/d(state_j) expressions (n_outputs × n, row-major).
    /// Non-empty when source == Symbolic and model has output expressions.
    std::vector<forms::FormExpr> dOutput_dx_exprs{};

    /// Number of outputs.
    int n_outputs{0};

    /// Symbolic dF_i/d(field_value_comp_j) expressions per referenced FE field.
    /// Map: FieldId → vector of n_rows * n_field_components expressions (row-major).
    /// Index: [row * n_comp + comp].  For scalar fields, n_comp=1 (same as before).
    std::unordered_map<FieldId, std::vector<forms::FormExpr>> dF_dfield_exprs{};

    /// Number of components per referenced field (for per-component layout).
    std::unordered_map<FieldId, int> dF_dfield_ncomp{};

    /// FieldIds referenced by the residual expressions.
    std::vector<FieldId> referenced_fields{};

    /// Symbolic d²F_i/(dx_j dx_k) expressions (n×n×n, row-major: [i*n*n + j*n + k]).
    /// Generated on demand when Hessian mode is requested.
    std::vector<forms::FormExpr> d2F_dx2_exprs{};

    /// Whether Hessian expressions have been generated.
    bool hessian_generated{false};

    /// Block dimension.
    int n{0};

    /// Diagnostic message when symbolic differentiation was attempted but
    /// fell back to another source.  Empty when symbolic succeeded or was
    /// never attempted.
    std::string fallback_reason{};
};

// ---------------------------------------------------------------------------
//  Provider
// ---------------------------------------------------------------------------

/**
 * @brief Derivative provider for one auxiliary model.
 *
 * Created per model instance.  Resolves the derivative policy and
 * generates/caches derivative artifacts at setup time.
 */
class AuxiliaryDerivativeProvider {
public:
    AuxiliaryDerivativeProvider() = default;

    /**
     * @brief Initialize the provider for a given model and policy.
     *
     * Resolves which derivative source to use:
     * 1. Analytic (if model provides it and policy allows).
     * 2. Symbolic (if model has residual expressions).
     * 3. FiniteDifference (explicit opt-in or fallback).
     *
     * For symbolic, generates and caches derivative expressions.
     */
    void setup(const AuxiliaryStateModel& model,
               const AuxiliaryDerivativePolicy& policy);

    /// Whether the provider is set up and ready.
    [[nodiscard]] bool isSetup() const noexcept { return is_setup_; }

    /// The resolved derivative source being used.
    [[nodiscard]] AuxiliaryDerivativeSource resolvedSource() const noexcept
    {
        return resolved_source_;
    }

    /// Whether analytic Jacobians are available (from model override).
    [[nodiscard]] bool hasAnalyticJacobian() const noexcept
    {
        return use_analytic_;
    }

    /// Whether symbolic derivative artifacts are available.
    [[nodiscard]] bool hasSymbolicArtifacts() const noexcept
    {
        return artifact_.valid;
    }

    /**
     * @brief Evaluate the Jacobian for the given context.
     *
     * Dispatches to the resolved source:
     * - Analytic: calls model->evaluateJacobian().
     * - Symbolic: evaluates cached symbolic expressions.
     * - FiniteDifference: perturbs and calls model->evaluateResidual().
     *
     * @param model   The model to evaluate.
     * @param ctx     Current evaluation context.
     * @param request Output request for Jacobian matrices.
     */
    void evaluateJacobian(const AuxiliaryStateModel& model,
                          const AuxiliaryLocalContext& ctx,
                          AuxiliaryJacobianRequest& request) const;

    /**
     * @brief Evaluate dF/d(field_value) for a specific FE field.
     *
     * Returns a vector of n values: dF_i/d(field_value) for each row i.
     * The derivative is w.r.t. the field value at the entity's location.
     * To get per-DOF contributions, multiply by basis function values.
     *
     * @param field  FieldId of the FE field.
     * @param ctx    Current evaluation context.
     * @return Empty vector if the field is not referenced.
     */
    [[nodiscard]] std::vector<Real> evaluateFieldDerivative(
        FieldId field,
        const AuxiliaryLocalContext& ctx) const;

    /**
     * @brief Evaluate the Hessian (if available).
     */
    void evaluateHessian(const AuxiliaryStateModel& model,
                         const AuxiliaryLocalContext& ctx,
                         AuxiliaryHessianRequest& request) const;

    /// Get the cached derivative artifact (for inspection/testing).
    [[nodiscard]] const AuxiliaryDerivativeArtifact& artifact() const noexcept
    {
        return artifact_;
    }

private:
    void evaluateJacobianFD(const AuxiliaryStateModel& model,
                            const AuxiliaryLocalContext& ctx,
                            AuxiliaryJacobianRequest& request) const;

    bool is_setup_{false};
    AuxiliaryDerivativeSource resolved_source_{AuxiliaryDerivativeSource::Symbolic};
    bool use_analytic_{false};
    AuxiliaryDerivativePolicy policy_{};
    mutable AuxiliaryDerivativeArtifact artifact_{};

    /// Scratch buffers for FD perturbation (avoid per-call allocation).
    mutable std::vector<Real> fd_scratch_residual_{};
    mutable std::vector<Real> fd_scratch_perturbed_{};
    mutable std::vector<Real> fd_scratch_x_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_AUXILIARY_DERIVATIVE_PROVIDER_H
