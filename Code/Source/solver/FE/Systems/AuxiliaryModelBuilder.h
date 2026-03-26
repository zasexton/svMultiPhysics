#ifndef SVMP_FE_SYSTEMS_AUXILIARY_MODEL_BUILDER_H
#define SVMP_FE_SYSTEMS_AUXILIARY_MODEL_BUILDER_H

/**
 * @file AuxiliaryModelBuilder.h
 * @brief Math-first declarative builder for auxiliary ODE/DAE-like models.
 *
 * Provides a readable, composable API for defining local auxiliary models:
 *
 * ```cpp
 * auto rcr = AuxiliaryModelBuilder("RCR")
 *     .input("Q")                              // flow rate input
 *     .state("P_d")                            // distal pressure (ODE)
 *     .param("R_p")                            // proximal resistance
 *     .param("C")                              // capacitance
 *     .param("R_d")                            // distal resistance
 *     .ode("P_d", (input("Q") - state("P_d") / param("R_d")) / param("C"))
 *     .output("P_out", param("R_p") * input("Q") + state("P_d"))
 *     .build();
 * ```
 *
 * The built model lowers to `AuxiliaryStateModel` with residual FormExpr
 * trees suitable for symbolic differentiation.
 *
 * ## Canonical lowering
 *
 * - `ode(x, rhs)` → residual: `dot(x) - rhs`
 * - `algebraic(z, expr)` → residual: `expr`
 * - `residual(name, expr)` → raw residual row
 *
 * ## Signature
 *
 * `AuxiliaryModelSignature` captures the reusable contract (inputs, outputs,
 * parameters) independently of internal state names.  Two models are
 * interchangeable if their signatures are compatible.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Forms/FormExpr.h"

#include "Systems/AuxiliaryStateTypes.h"
#include "Systems/AuxiliaryStateModel.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Model signature
// ---------------------------------------------------------------------------

/**
 * @brief Descriptor for one input, output, or parameter in a model signature.
 */
struct AuxiliaryModelPortDescriptor {
    std::string name{};
    int size{1};
    bool optional{false};
    std::optional<Real> default_value{};  ///< Default value for optional ports.
    std::optional<Real> lower_bound{};    ///< Optional lower bound metadata.
    std::optional<Real> upper_bound{};    ///< Optional upper bound metadata.
    std::optional<Real> scale{};          ///< Optional solver scaling metadata.
};

/**
 * @brief Reusable contract for an auxiliary model.
 *
 * Two models are interchangeable when their signatures have compatible
 * inputs, outputs, and parameters (name + size).  Internal state names
 * are NOT part of the signature.
 */
struct AuxiliaryModelSignature {
    std::string model_name{};
    std::vector<AuxiliaryModelPortDescriptor> inputs{};
    std::vector<AuxiliaryModelPortDescriptor> outputs{};
    std::vector<AuxiliaryModelPortDescriptor> parameters{};

    [[nodiscard]] bool isCompatibleWith(const AuxiliaryModelSignature& other) const;
};

// ---------------------------------------------------------------------------
//  Symbolic placeholders for model-definition expressions
// ---------------------------------------------------------------------------

/// Reference to a model input by name (used inside builder expressions).
[[nodiscard]] inline forms::FormExpr modelInput(const std::string& name)
{
    return forms::FormExpr::auxiliaryInput(name);
}

/// Reference to a model state variable by name.
[[nodiscard]] inline forms::FormExpr modelState(const std::string& name)
{
    return forms::FormExpr::auxiliaryState(name);
}

/// Reference to a model parameter by name.
[[nodiscard]] inline forms::FormExpr modelParam(const std::string& name)
{
    return forms::FormExpr::parameter(name);
}

// ---------------------------------------------------------------------------
//  Row declarations (internal)
// ---------------------------------------------------------------------------

/// Kind of row declaration in the builder.
enum class AuxiliaryRowKind : std::uint8_t {
    ODE,         ///< ode(state, rhs) → dot(state) - rhs
    Algebraic,   ///< algebraic(state, expr) → expr
    Residual     ///< residual(name, expr) → raw row
};

/**
 * @brief One row declaration from the builder.
 */
struct AuxiliaryRowDeclaration {
    AuxiliaryRowKind kind{AuxiliaryRowKind::ODE};
    std::string state_name{};
    forms::FormExpr expression{};
};

// ---------------------------------------------------------------------------
//  Built model (expression-defined AuxiliaryStateModel)
// ---------------------------------------------------------------------------

/**
 * @brief An AuxiliaryStateModel built from the declarative builder.
 *
 * Carries lowered residual FormExpr trees so the derivative provider
 * can differentiate them symbolically at setup time.
 */
class BuiltAuxiliaryModel final : public AuxiliaryStateModel {
public:
    BuiltAuxiliaryModel(std::string name,
                         int dim,
                         AuxiliaryStructuralMetadata metadata,
                         std::vector<forms::FormExpr> residual_exprs,
                         std::vector<std::string> state_names,
                         std::vector<AuxiliaryModelPortDescriptor> inputs,
                         std::vector<AuxiliaryModelPortDescriptor> outputs,
                         std::vector<AuxiliaryModelPortDescriptor> params,
                         AuxiliaryDerivativePolicy deriv_policy);

    [[nodiscard]] std::string modelName() const override { return name_; }
    [[nodiscard]] int dimension() const override { return dim_; }
    [[nodiscard]] AuxiliaryStructuralMetadata structuralMetadata() const override
    {
        return metadata_;
    }

    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& request) const override;

    [[nodiscard]] bool hasResidualExpressions() const override { return true; }
    [[nodiscard]] std::vector<forms::FormExpr> residualExpressions() const override
    {
        return residual_exprs_;
    }

    /// Model signature (inputs, outputs, parameters).
    [[nodiscard]] const AuxiliaryModelSignature& signature() const noexcept
    {
        return signature_;
    }

    /// State variable names.
    [[nodiscard]] const std::vector<std::string>& stateNames() const noexcept
    {
        return state_names_;
    }

    /// Per-state algebraic initial guesses (empty optional for differential states).
    [[nodiscard]] const std::vector<std::optional<Real>>& initialGuesses() const noexcept
    {
        return initial_guesses_;
    }

    /// Output expressions (BuiltAuxiliaryModel-specific accessor).
    [[nodiscard]] const std::vector<std::pair<std::string, forms::FormExpr>>&
    outputExpressions() const noexcept { return output_exprs_; }

    // Base-class output interface overrides.
    [[nodiscard]] int outputCount() const override
    {
        return static_cast<int>(output_exprs_.size());
    }
    [[nodiscard]] std::vector<std::string> outputNames() const override
    {
        std::vector<std::string> names;
        names.reserve(output_exprs_.size());
        for (const auto& [n, e] : output_exprs_) names.push_back(n);
        return names;
    }
    void evaluateOutputs(const AuxiliaryLocalContext& ctx,
                          std::span<Real> output) const override;

    /// Set output expressions (called by builder after construction).
    void setOutputExpressions(
        std::vector<std::pair<std::string, forms::FormExpr>> exprs)
    {
        output_exprs_ = std::move(exprs);
    }

    /// Set initial guesses for algebraic states (called by builder).
    void setInitialGuesses(std::vector<std::optional<Real>> guesses)
    {
        initial_guesses_ = std::move(guesses);
    }

private:
    std::string name_{};
    int dim_{0};
    AuxiliaryStructuralMetadata metadata_{};
    std::vector<forms::FormExpr> residual_exprs_{};
    std::vector<std::string> state_names_{};
    std::vector<std::optional<Real>> initial_guesses_{};
    AuxiliaryModelSignature signature_{};
    AuxiliaryDerivativePolicy deriv_policy_{};
    std::vector<std::pair<std::string, forms::FormExpr>> output_exprs_{};

public:
    /// Derivative policy configured on the model.
    [[nodiscard]] const AuxiliaryDerivativePolicy& derivativePolicy() const noexcept
    {
        return deriv_policy_;
    }
};

// ---------------------------------------------------------------------------
//  Builder
// ---------------------------------------------------------------------------

/**
 * @brief Fluent builder for auxiliary ODE/DAE-like models.
 *
 * Usage:
 * ```cpp
 * auto model = AuxiliaryModelBuilder("decay")
 *     .state("x")
 *     .param("k")
 *     .ode("x", -modelParam("k") * modelState("x"))
 *     .output("y", modelState("x"))
 *     .build();
 * ```
 */
class AuxiliaryModelBuilder {
public:
    explicit AuxiliaryModelBuilder(std::string name);

    /// Declare an input port (externally supplied value).
    AuxiliaryModelBuilder& input(std::string name, int size = 1);

    /// Declare a state variable (differential by default).
    AuxiliaryModelBuilder& state(std::string name,
                                  AuxiliaryVariableKind kind =
                                      AuxiliaryVariableKind::Differential);

    /// Declare a parameter (constant or provider-driven).
    AuxiliaryModelBuilder& param(std::string name);

    /// Declare a parameter with a default value (optional at deployment time).
    AuxiliaryModelBuilder& param(std::string name, Real default_value);

    /// Declare an optional input with a default value.
    AuxiliaryModelBuilder& optionalInput(std::string name, Real default_value, int size = 1);

    /// Declare a derived quantity (computed, not a solve target).
    AuxiliaryModelBuilder& derived(std::string name, forms::FormExpr expr);

    /// Declare an ODE row: dot(state) = rhs → residual: dot(state) - rhs.
    AuxiliaryModelBuilder& ode(const std::string& state_name,
                                forms::FormExpr rhs);

    /// Declare an algebraic row: expr = 0.
    AuxiliaryModelBuilder& algebraic(const std::string& state_name,
                                      forms::FormExpr expr);

    /// Declare a raw residual row.
    AuxiliaryModelBuilder& residual(std::string name, forms::FormExpr expr);

    /// Declare an output (derived expression, first-class coupling surface).
    AuxiliaryModelBuilder& output(std::string name, forms::FormExpr expr);

    /// Control unused-symbol diagnostics: "warn" (default), "error", "silent".
    AuxiliaryModelBuilder& unusedSymbolPolicy(const std::string& policy);

    /// Set derivative policy for the built model.
    AuxiliaryModelBuilder& derivatives(AuxiliaryDerivativePolicy policy);

    /// Build the model.  Validates and lowers row declarations to residual
    /// FormExpr trees.
    [[nodiscard]] std::shared_ptr<BuiltAuxiliaryModel> build() const;

    // ---- DSL equation insertion ----

    /// Insert an equation object from the math-first DSL.
    /// Used via `builder << ddt(X) == rhs` or `builder << out("name") == expr`.
    /// See AuxiliaryModelDSL.h for the full DSL surface.
    AuxiliaryModelBuilder& insertEquation(
        int kind,  // 0=ODE, 1=Algebraic, 2=Output
        std::string target_name,
        forms::FormExpr rhs);

    // ---- Introspection ----

    /// Model name.
    [[nodiscard]] const std::string& modelName() const noexcept { return name_; }

    /// Declared state names in declaration order.
    [[nodiscard]] std::vector<std::string> stateNames() const;

    /// Declared input names in declaration order.
    [[nodiscard]] std::vector<std::string> inputNames() const;

    /// Declared parameter names in declaration order.
    [[nodiscard]] std::vector<std::string> parameterNames() const;

    /// Declared output names in declaration order.
    [[nodiscard]] std::vector<std::string> outputNames() const;

    /// Number of declared states.
    [[nodiscard]] std::size_t stateCount() const noexcept { return states_.size(); }

    /// Total flat input slot count (sum of all input sizes).
    [[nodiscard]] std::size_t flatInputSize() const noexcept {
        std::size_t n = 0;
        for (const auto& p : inputs_) n += static_cast<std::size_t>(p.size);
        return n;
    }

    /// Number of declared parameters.
    [[nodiscard]] std::size_t paramCount() const noexcept { return params_.size(); }

    /// Number of declared rows (equations).
    [[nodiscard]] std::size_t rowCount() const noexcept { return rows_.size(); }

    /// Pretty-print a model summary showing all declared symbols and their metadata.
    [[nodiscard]] std::string summary() const;

    // ---- Metadata setters ----

    /// Set lower/upper bounds on a state or parameter by name.
    AuxiliaryModelBuilder& setBounds(const std::string& name, Real lower, Real upper);
    /// Set nonnegative bound (lower=0) on a state by name.
    AuxiliaryModelBuilder& setNonnegative(const std::string& name);
    /// Set solver scaling on a state by name.
    AuxiliaryModelBuilder& setScale(const std::string& name, Real s);
    /// Set algebraic initial guess on a state by name.
    AuxiliaryModelBuilder& setInitialGuess(const std::string& name, Real guess);

private:
    std::string name_{};
    std::vector<AuxiliaryModelPortDescriptor> inputs_{};
    struct StateDecl {
        std::string name{};
        AuxiliaryVariableKind kind{AuxiliaryVariableKind::Differential};
        std::optional<Real> initial_guess{};  ///< For algebraic states.
        std::optional<Real> lower_bound{};
        std::optional<Real> upper_bound{};
        std::optional<Real> scale{};
    };
    std::vector<StateDecl> states_{};
    std::vector<AuxiliaryModelPortDescriptor> params_{};
    std::vector<std::pair<std::string, forms::FormExpr>> derived_{};
    std::vector<AuxiliaryRowDeclaration> rows_{};
    std::vector<std::pair<std::string, forms::FormExpr>> outputs_{};
    AuxiliaryDerivativePolicy deriv_policy_{};

    std::unordered_set<std::string> all_names_{};
    std::string unused_symbol_policy_{"warn"};  ///< "warn", "error", "silent"

    void validateUniqueName(const std::string& name) const;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_AUXILIARY_MODEL_BUILDER_H
