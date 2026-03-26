#ifndef SVMP_FE_SYSTEMS_AUXILIARY_MODEL_DSL_H
#define SVMP_FE_SYSTEMS_AUXILIARY_MODEL_DSL_H

/**
 * @file AuxiliaryModelDSL.h
 * @brief Math-first DSL for auxiliary model authoring.
 *
 * Provides typed symbol handles, equation-like syntax, grouped declarations,
 * named intermediates, and a lambda-based model construction front end.
 *
 * ## Quick start
 *
 * ```cpp
 * using namespace svmp::FE::systems::aux;
 *
 * auto rcr = model("rcr", [&](auto& m) {
 *     auto Q = m.input("Q");
 *     auto X = m.state("X");
 *     auto [Rp, C, Rd, Pd] = m.params("Rp", "C", "Rd", "Pd");
 *
 *     m << ddt(X) == (Q - (X - Pd) / Rd) / C;
 *     m << out("P_out") == X + Rp * Q;
 * });
 * ```
 *
 * ## Equation syntax
 *
 * - `m << ddt(X) == rhs`        — ODE row: dX/dt = rhs
 * - `m << alg(Z) == constraint`  — algebraic row: constraint = 0
 * - `m << out("name") == expr`   — named output
 *
 * ## Grouped declarations
 *
 * - `auto [a, b, c] = m.params("a", "b", "c");`
 * - `auto [x, y] = m.states("x", "y");`
 * - `auto [u, v] = m.inputs("u", "v");`
 *
 * ## Named intermediates
 *
 * - `auto v_rate = m.let("v_rate", k * S / (Km + S));`
 * - `m.expose(v_rate, "rate_output");`
 *
 * ## Design
 *
 * The DSL is a thin front end over `AuxiliaryModelBuilder`.  All equation
 * forms lower to the existing builder methods (`.ode()`, `.algebraic()`,
 * `.output()`).  The typed symbols inherit from `FormExpr` so they
 * participate directly in arithmetic expression trees.
 *
 * This header is entirely physics-agnostic.
 */

#include "Core/Types.h"

#include "Forms/FormExpr.h"

#include "Systems/AuxiliaryModelBuilder.h"
#include "Systems/AuxiliaryStateTypes.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ============================================================================
//  Typed symbol handles
// ============================================================================

/**
 * @brief A declared input port, usable directly in FormExpr arithmetic.
 *
 * Created by `ModelFacade::input()`.  Inherits from `FormExpr` so that
 * expressions like `Q * Rp + X` work without explicit conversion.
 */
class InputSymbol : public forms::FormExpr {
    std::string decl_name_;
public:
    explicit InputSymbol(const std::string& name)
        : forms::FormExpr(forms::FormExpr::auxiliaryInput(name))
        , decl_name_(name) {}

    [[nodiscard]] const std::string& declName() const noexcept { return decl_name_; }
};

/**
 * @brief A declared state variable, usable directly in FormExpr arithmetic.
 *
 * Created by `ModelFacade::state()`.  Pass to `ddt()` or `alg()` to form
 * governing equations.
 */
class StateSymbol : public forms::FormExpr {
    std::string decl_name_;
public:
    explicit StateSymbol(const std::string& name)
        : forms::FormExpr(forms::FormExpr::auxiliaryState(name))
        , decl_name_(name) {}

    [[nodiscard]] const std::string& declName() const noexcept { return decl_name_; }
};

/**
 * @brief A declared parameter, usable directly in FormExpr arithmetic.
 *
 * Created by `ModelFacade::param()`.
 */
class ParamSymbol : public forms::FormExpr {
    std::string decl_name_;
public:
    explicit ParamSymbol(const std::string& name)
        : forms::FormExpr(forms::FormExpr::parameter(name))
        , decl_name_(name) {}

    [[nodiscard]] const std::string& declName() const noexcept { return decl_name_; }
};

// ============================================================================
//  Equation targets and equation objects
// ============================================================================

/// Target for `ddt(X) == rhs` (ODE row).
struct DdtTarget {
    std::string state_name;
};

/// Target for `alg(Z) == constraint` (algebraic row).
struct AlgTarget {
    std::string state_name;
};

/// Target for `out("name") == expr` (output equation).
struct OutputTarget {
    std::string output_name;
};

/// Create a differential-equation target from a state symbol.
[[nodiscard]] inline DdtTarget ddt(const StateSymbol& s)
{
    return DdtTarget{s.declName()};
}

/// Create an algebraic-equation target from a state symbol.
[[nodiscard]] inline AlgTarget alg(const StateSymbol& s)
{
    return AlgTarget{s.declName()};
}

/// Create an output-equation target from a name.
[[nodiscard]] inline OutputTarget out(std::string name)
{
    return OutputTarget{std::move(name)};
}

/// Kind of equation in the DSL.
enum class AuxEquationKind : std::uint8_t {
    ODE,        ///< ddt(X) == rhs
    Algebraic,  ///< alg(Z) == constraint
    Output      ///< out("name") == expr
};

/**
 * @brief A fully formed equation ready for insertion into a builder.
 *
 * Created by `operator==` on equation targets, or via the proxy mechanism
 * when used with `m << ddt(X) == rhs` syntax (where `<<` binds before `==`).
 */
struct AuxEquation {
    AuxEquationKind kind{};
    std::string target_name{};
    forms::FormExpr rhs{};
};

/// Form an ODE equation: `ddt(X) == rhs`.
[[nodiscard]] inline AuxEquation operator==(DdtTarget target, forms::FormExpr rhs)
{
    return AuxEquation{AuxEquationKind::ODE, std::move(target.state_name), std::move(rhs)};
}

/// Form an algebraic equation: `alg(Z) == constraint`.
[[nodiscard]] inline AuxEquation operator==(AlgTarget target, forms::FormExpr rhs)
{
    return AuxEquation{AuxEquationKind::Algebraic, std::move(target.state_name), std::move(rhs)};
}

/// Form an output equation: `out("name") == expr`.
[[nodiscard]] inline AuxEquation operator==(OutputTarget target, forms::FormExpr rhs)
{
    return AuxEquation{AuxEquationKind::Output, std::move(target.output_name), std::move(rhs)};
}

// ============================================================================
//  Pending equation proxy (handles << precedence over ==)
// ============================================================================

// Forward declaration
class ModelFacade;

/**
 * @brief Proxy object returned by `m << ddt(X)`.
 *
 * Because `<<` has higher precedence than `==`, the expression
 * `m << ddt(X) == rhs` is parsed as `(m << ddt(X)) == rhs`.
 * The proxy captures the pending equation target and facade reference,
 * then `operator==` completes and inserts the equation.
 */
struct PendingEquation {
    ModelFacade& facade;
    AuxEquationKind kind;
    std::string target_name;
};

/// Complete a pending equation: `(m << ddt(X)) == rhs`.
/// Inserts the equation into the facade's builder.
ModelFacade& operator==(PendingEquation pending, forms::FormExpr rhs);

// ============================================================================
//  Model facade (builder wrapper for lambda-based construction)
// ============================================================================

/**
 * @brief Builder facade exposing the math-first DSL surface.
 *
 * Wraps `AuxiliaryModelBuilder` and provides:
 * - Typed symbol declarations returning `InputSymbol` / `StateSymbol` / `ParamSymbol`
 * - Grouped declaration helpers (`params(...)`, `states(...)`, `inputs(...)`)
 * - Named intermediate expressions (`let(...)`)
 * - Equation insertion via `operator<<`
 *
 * Not intended for direct construction — use `aux::model(name, lambda)`.
 */
class ModelFacade {
public:
    explicit ModelFacade(AuxiliaryModelBuilder& builder) : builder_(builder) {}

    // ---- Single declarations ----

    /// Declare an input port and return a typed symbol.
    [[nodiscard]] InputSymbol input(const std::string& name, int size = 1)
    {
        builder_.input(std::string(name), size);
        return InputSymbol(name);
    }

    /// Declare a state variable and return a typed symbol.
    [[nodiscard]] StateSymbol state(const std::string& name,
                                     AuxiliaryVariableKind kind =
                                         AuxiliaryVariableKind::Differential)
    {
        builder_.state(std::string(name), kind);
        return StateSymbol(name);
    }

    /// Declare a parameter and return a typed symbol.
    [[nodiscard]] ParamSymbol param(const std::string& name)
    {
        builder_.param(std::string(name));
        return ParamSymbol(name);
    }

    /// Declare a parameter with a default value (optional at deployment).
    [[nodiscard]] ParamSymbol param(const std::string& name, Real default_value)
    {
        builder_.param(std::string(name), default_value);
        return ParamSymbol(name);
    }

    /// Declare an optional input with a default value.
    [[nodiscard]] InputSymbol optionalInput(const std::string& name,
                                             Real default_value, int size = 1)
    {
        builder_.optionalInput(std::string(name), default_value, size);
        return InputSymbol(name);
    }

    // ---- Metadata setters (modify last-declared symbol) ----

    /// Mark a state as nonnegative (lower_bound = 0).
    ModelFacade& nonnegative(const std::string& symbol_name)
    {
        builder_.setNonnegative(symbol_name);
        return *this;
    }

    /// Set bounds on a state/param.
    ModelFacade& bounded(const std::string& symbol_name, Real lower, Real upper)
    {
        builder_.setBounds(symbol_name, lower, upper);
        return *this;
    }

    /// Set solver scaling on a state.
    ModelFacade& scale(const std::string& symbol_name, Real s)
    {
        builder_.setScale(symbol_name, s);
        return *this;
    }

    /// Set algebraic initial guess on an algebraic state.
    ModelFacade& initialGuess(const std::string& symbol_name, Real guess)
    {
        builder_.setInitialGuess(symbol_name, guess);
        return *this;
    }

    // ---- Grouped declarations ----

    /// Declare multiple parameters and return as a tuple.
    template<typename... Names>
    [[nodiscard]] auto params(Names&&... names)
    {
        return std::make_tuple(param(std::string(std::forward<Names>(names)))...);
    }

    /// Declare multiple states (all Differential) and return as a tuple.
    template<typename... Names>
    [[nodiscard]] auto states(Names&&... names)
    {
        return std::make_tuple(state(std::string(std::forward<Names>(names)))...);
    }

    /// Declare multiple inputs (all size=1) and return as a tuple.
    template<typename... Names>
    [[nodiscard]] auto inputs(Names&&... names)
    {
        return std::make_tuple(input(std::string(std::forward<Names>(names)))...);
    }

    /// Declare multiple parameters and return as a vector.
    [[nodiscard]] std::vector<ParamSymbol> paramVec(
        std::initializer_list<std::string> names)
    {
        std::vector<ParamSymbol> result;
        result.reserve(names.size());
        for (const auto& n : names) {
            result.push_back(param(n));
        }
        return result;
    }

    /// Declare multiple states and return as a vector.
    [[nodiscard]] std::vector<StateSymbol> stateVec(
        std::initializer_list<std::string> names,
        AuxiliaryVariableKind kind = AuxiliaryVariableKind::Differential)
    {
        std::vector<StateSymbol> result;
        result.reserve(names.size());
        for (const auto& n : names) {
            result.push_back(state(n, kind));
        }
        return result;
    }

    /// Declare multiple inputs and return as a vector.
    [[nodiscard]] std::vector<InputSymbol> inputVec(
        std::initializer_list<std::string> names)
    {
        std::vector<InputSymbol> result;
        result.reserve(names.size());
        for (const auto& n : names) {
            result.push_back(input(n));
        }
        return result;
    }

    // ---- Named intermediates ----

    /**
     * @brief Define a named intermediate expression.
     *
     * The expression is stored in the builder for introspection/debug and
     * the `FormExpr` is returned for reuse in equations and outputs.
     *
     * ```cpp
     * auto v_gly = m.let("v_gly", k_gly * Glc / (Km_Glc + Glc));
     * m << ddt(ATP) == v_gly - v_demand;
     * ```
     */
    [[nodiscard]] forms::FormExpr let(const std::string& name, forms::FormExpr expr)
    {
        builder_.derived(std::string(name), expr);
        return expr;
    }

    /**
     * @brief Expose a named intermediate as an output.
     *
     * Shorthand for `m << out(name) == expr`.
     */
    ModelFacade& expose(const forms::FormExpr& expr, const std::string& output_name)
    {
        builder_.output(std::string(output_name), expr);
        return *this;
    }

    // ---- Equation insertion ----

    /**
     * @brief Begin an ODE equation: `m << ddt(X) == rhs`.
     *
     * Because `<<` has higher precedence than `==`, this returns a proxy
     * that captures the target.  The `==` then completes the equation.
     */
    PendingEquation operator<<(DdtTarget target)
    {
        return PendingEquation{*this, AuxEquationKind::ODE, std::move(target.state_name)};
    }

    /// Begin an algebraic equation: `m << alg(Z) == constraint`.
    PendingEquation operator<<(AlgTarget target)
    {
        return PendingEquation{*this, AuxEquationKind::Algebraic, std::move(target.state_name)};
    }

    /// Begin an output equation: `m << out("name") == expr`.
    PendingEquation operator<<(OutputTarget target)
    {
        return PendingEquation{*this, AuxEquationKind::Output, std::move(target.output_name)};
    }

    /// Insert a pre-formed equation (for programmatic use).
    ModelFacade& operator<<(AuxEquation eq)
    {
        insertEquation(eq.kind, std::move(eq.target_name), std::move(eq.rhs));
        return *this;
    }

    /// Insert an equation into the builder (used by PendingEquation proxy).
    void insertEquation(AuxEquationKind kind, std::string target_name,
                         forms::FormExpr rhs)
    {
        switch (kind) {
            case AuxEquationKind::ODE:
                builder_.ode(target_name, std::move(rhs));
                break;
            case AuxEquationKind::Algebraic:
                builder_.algebraic(target_name, std::move(rhs));
                break;
            case AuxEquationKind::Output:
                builder_.output(std::move(target_name), std::move(rhs));
                break;
        }
    }

    // ---- Conservation / invariant helpers ----

    /**
     * @brief Declare a conservation constraint as an algebraic equation.
     *
     * Lowers to `alg(dependent_state) == lhs - rhs` where the constraint
     * is expressed as `lhs == rhs`.  The `dependent_state` is the state
     * that will be solved from the conservation law rather than having
     * its own ODE.
     *
     * ```cpp
     * auto ADP = m.state("ADP", Algebraic);
     * m.conservation(ADP, ATP + ADP - A_tot);
     * // Equivalent to: m << alg(ADP) == ATP + ADP - A_tot;
     * ```
     */
    ModelFacade& conservation(const StateSymbol& dependent_state,
                               forms::FormExpr constraint)
    {
        return *this << (alg(dependent_state) == std::move(constraint));
    }

    // ---- Derivative policy ----

    /// Set the derivative policy for the model.
    ModelFacade& derivatives(AuxiliaryDerivativePolicy policy)
    {
        builder_.derivatives(policy);
        return *this;
    }

    // ---- Direct builder access (advanced) ----

    /// Access the underlying builder for advanced/raw operations.
    [[nodiscard]] AuxiliaryModelBuilder& builder() noexcept { return builder_; }

    // ---- Symbol grouping / namespacing ----

    /**
     * @brief Create a namespaced sub-facade with a prefix.
     *
     * All declarations through the returned facade have the prefix prepended:
     * ```cpp
     * auto mito = m.group("mito");
     * auto ATP = mito.state("ATP");  // declares "mito.ATP"
     * ```
     */
    class GroupFacade {
    public:
        GroupFacade(ModelFacade& parent, std::string prefix)
            : parent_(parent), prefix_(std::move(prefix)) {}

        [[nodiscard]] InputSymbol input(const std::string& name, int size = 1) {
            return parent_.input(prefix_ + name, size);
        }
        [[nodiscard]] StateSymbol state(const std::string& name,
                                         AuxiliaryVariableKind kind =
                                             AuxiliaryVariableKind::Differential) {
            return parent_.state(prefix_ + name, kind);
        }
        [[nodiscard]] ParamSymbol param(const std::string& name) {
            return parent_.param(prefix_ + name);
        }
        [[nodiscard]] ParamSymbol param(const std::string& name, Real default_value) {
            return parent_.param(prefix_ + name, default_value);
        }
        [[nodiscard]] forms::FormExpr let(const std::string& name, forms::FormExpr expr) {
            return parent_.let(prefix_ + name, std::move(expr));
        }

    private:
        ModelFacade& parent_;
        std::string prefix_;
    };

    [[nodiscard]] GroupFacade group(const std::string& prefix) {
        return GroupFacade(*this, prefix + ".");
    }

    // ---- Submodel composition ----

    /**
     * @brief Include a submodel's declarations with a prefix.
     *
     * Copies all states, inputs, params, and equations from the submodel
     * into the current builder, prefixing all names to avoid collisions.
     *
     * Note: the submodel must be a `BuiltAuxiliaryModel` built from
     * `AuxiliaryModelBuilder`. The expressions are not re-resolved —
     * they use the submodel's internal slot indices, which are remapped
     * during the include operation.
     *
     * ```cpp
     * auto glycolysis = aux::model("gly", ...);
     * auto mito = aux::model("mito", ...);
     *
     * auto full = aux::model("cell", [&](auto& m) {
     *     m.include(glycolysis, "gly");
     *     m.include(mito, "mito");
     *     // ... cross-model coupling equations ...
     * });
     * ```
     *
     * Currently a placeholder — full slot remapping is future work.
     * For now, use grouped declarations with the same model lambda.
     */
    /**
     * @brief Include a built submodel with prefixed names and remapped slots.
     *
     * Copies all states, inputs, params, equations, and outputs from the
     * submodel into the current builder.  Expression slot indices are
     * remapped to account for existing declarations in the host model.
     */
    void include(const std::shared_ptr<BuiltAuxiliaryModel>& submodel,
                  const std::string& prefix)
    {
        if (!submodel) return;
        const std::string pfx = prefix + ".";

        // Record offsets for slot remapping.
        const auto state_offset = static_cast<std::uint32_t>(builder_.stateCount());
        const auto input_offset = static_cast<std::uint32_t>(builder_.flatInputSize());
        const auto param_offset = static_cast<std::uint32_t>(builder_.paramCount());

        // Copy declarations with prefix.
        const auto& sig = submodel->signature();
        const auto& state_names = submodel->stateNames();
        const auto meta = submodel->structuralMetadata();

        for (std::size_t i = 0; i < state_names.size(); ++i) {
            const auto kind = (i < meta.variable_kinds.size())
                ? meta.variable_kinds[i] : AuxiliaryVariableKind::Differential;
            builder_.state(pfx + state_names[i], kind);
        }
        for (const auto& inp : sig.inputs) {
            builder_.input(pfx + inp.name, inp.size);
        }
        for (const auto& p : sig.parameters) {
            if (p.default_value) {
                builder_.param(pfx + p.name, *p.default_value);
            } else {
                builder_.param(pfx + p.name);
            }
        }

        // Slot remapping transform.
        auto remap = [&](const forms::FormExprNode& node)
            -> std::optional<forms::FormExpr> {
            const auto slot = node.slotIndex();
            if (!slot) return std::nullopt;
            switch (node.type()) {
                case forms::FormExprType::AuxiliaryStateRef:
                    return forms::FormExpr::auxiliaryStateRef(*slot + state_offset);
                case forms::FormExprType::AuxiliaryInputRef:
                    return forms::FormExpr::auxiliaryInputRef(*slot + input_offset);
                case forms::FormExprType::ParameterRef:
                    return forms::FormExpr::parameterRef(*slot + param_offset);
                default:
                    return std::nullopt;
            }
        };

        // Copy equations with remapped expressions.
        if (submodel->hasResidualExpressions()) {
            const auto& residuals = submodel->residualExpressions();
            for (std::size_t i = 0; i < residuals.size() && i < state_names.size(); ++i) {
                const auto kind = (i < meta.variable_kinds.size())
                    ? meta.variable_kinds[i] : AuxiliaryVariableKind::Differential;
                auto remapped = residuals[i].transformNodes(remap);
                if (kind == AuxiliaryVariableKind::Differential) {
                    builder_.ode(pfx + state_names[i], std::move(remapped));
                } else {
                    builder_.algebraic(pfx + state_names[i], std::move(remapped));
                }
            }
        }

        // Copy outputs with remapped expressions.
        for (const auto& [oname, oexpr] : submodel->outputExpressions()) {
            builder_.output(pfx + oname, oexpr.transformNodes(remap));
        }
    }

private:
    AuxiliaryModelBuilder& builder_;
};

// ============================================================================
//  Lambda-based model construction
// ============================================================================

namespace aux {

/**
 * @brief Build an auxiliary model using the math-first DSL.
 *
 * ```cpp
 * auto rcr = aux::model("rcr", [&](auto& m) {
 *     auto Q = m.input("Q");
 *     auto X = m.state("X");
 *     auto [Rp, C, Rd, Pd] = m.params("Rp", "C", "Rd", "Pd");
 *
 *     m << ddt(X) == (Q - (X - Pd) / Rd) / C;
 *     m << out("P_out") == X + Rp * Q;
 * });
 * ```
 *
 * @param name   Model name (used in diagnostics and instance naming).
 * @param f      Lambda receiving a `ModelFacade&`.
 * @return Built model ready for deployment.
 */
template<typename F>
[[nodiscard]] std::shared_ptr<BuiltAuxiliaryModel> model(std::string name, F&& f)
{
    AuxiliaryModelBuilder builder(std::move(name));
    ModelFacade facade(builder);
    std::forward<F>(f)(facade);
    return builder.build();
}

} // namespace aux

// ============================================================================
//  PendingEquation operator== (must be after ModelFacade is complete)
// ============================================================================

inline ModelFacade& operator==(PendingEquation pending, forms::FormExpr rhs)
{
    pending.facade.insertEquation(
        pending.kind, std::move(pending.target_name), std::move(rhs));
    return pending.facade;
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_AUXILIARY_MODEL_DSL_H
