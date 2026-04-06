#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Forms/PointEvaluator.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  AuxiliaryModelSignature
// ---------------------------------------------------------------------------

bool AuxiliaryModelSignature::isCompatibleWith(
    const AuxiliaryModelSignature& other) const
{
    if (inputs.size() != other.inputs.size()) return false;
    if (outputs.size() != other.outputs.size()) return false;
    if (parameters.size() != other.parameters.size()) return false;

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].name != other.inputs[i].name ||
            inputs[i].size != other.inputs[i].size)
            return false;
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].name != other.outputs[i].name ||
            outputs[i].size != other.outputs[i].size)
            return false;
    }
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        if (parameters[i].name != other.parameters[i].name) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
//  BuiltAuxiliaryModel
// ---------------------------------------------------------------------------

BuiltAuxiliaryModel::BuiltAuxiliaryModel(
    std::string name,
    int dim,
    AuxiliaryStructuralMetadata metadata,
    std::vector<forms::FormExpr> residual_exprs,
    std::vector<std::string> state_names,
    std::vector<AuxiliaryModelPortDescriptor> inputs,
    std::vector<AuxiliaryModelPortDescriptor> outputs,
    std::vector<AuxiliaryModelPortDescriptor> params,
    AuxiliaryDerivativePolicy deriv_policy)
    : name_(std::move(name))
    , dim_(dim)
    , metadata_(std::move(metadata))
    , residual_exprs_(std::move(residual_exprs))
    , state_names_(std::move(state_names))
    , deriv_policy_(deriv_policy)
{
    signature_.model_name = name_;
    signature_.inputs = std::move(inputs);
    signature_.outputs = std::move(outputs);
    signature_.parameters = std::move(params);
}

void BuiltAuxiliaryModel::evaluateResidual(
    const AuxiliaryLocalContext& ctx,
    AuxiliaryResidualRequest& request) const
{
    // Build a PointEvalContext from the auxiliary local context.
    forms::PointEvalContext pctx;
    pctx.time = ctx.time;
    pctx.dt = (ctx.effective_dt > 0.0) ? ctx.effective_dt : ctx.dt;
    pctx.coupled_aux = ctx.x;        // AuxiliaryStateRef slots map to state x
    pctx.auxiliary_inputs = ctx.inputs; // AuxiliaryInputRef slots map to inputs
    pctx.jit_constants = ctx.params;   // ParameterRef slots map to params
    pctx.field_values = ctx.field_values; // DiscreteField/StateField values

    const auto n = static_cast<std::size_t>(dim_);
    for (std::size_t i = 0; i < n; ++i) {
        // Evaluate the lowered residual expression.
        // For ODE rows: F = xdot - rhs, where xdot is supplied externally.
        // The expression tree encodes "-rhs", so: F = xdot + eval(expr)
        // Actually, the lowered expression IS the full residual.
        // For ODE: expr = dot(x) - rhs. We substitute xdot for dot(x).
        // For simplicity in this implementation, we evaluate the RHS part
        // and construct the residual.
        if (metadata_.variable_kinds[i] == AuxiliaryVariableKind::Differential) {
            // Residual = xdot[i] - rhs[i]
            // residual_exprs_[i] stores the RHS (from ode() declaration)
            Real rhs = forms::evaluateScalarAt(residual_exprs_[i], pctx);
            request.residual[i] = ctx.xdot[i] - rhs;
        } else {
            // Algebraic: residual = expr (already the constraint)
            request.residual[i] = forms::evaluateScalarAt(residual_exprs_[i], pctx);
        }
    }
}

void BuiltAuxiliaryModel::evaluateOutputs(
    const AuxiliaryLocalContext& ctx,
    std::span<Real> output) const
{
    forms::PointEvalContext pctx;
    pctx.time = ctx.time;
    pctx.dt = (ctx.effective_dt > 0.0) ? ctx.effective_dt : ctx.dt;
    pctx.coupled_aux = ctx.x;
    pctx.auxiliary_inputs = ctx.inputs;
    pctx.jit_constants = ctx.params;
    pctx.field_values = ctx.field_values;

    for (std::size_t i = 0; i < output_exprs_.size() && i < output.size(); ++i) {
        output[i] = forms::evaluateScalarAt(output_exprs_[i].second, pctx);
    }
}

// ---------------------------------------------------------------------------
//  AuxiliaryModelBuilder
// ---------------------------------------------------------------------------

AuxiliaryModelBuilder::AuxiliaryModelBuilder(std::string name)
    : name_(std::move(name))
{
    FE_THROW_IF(name_.empty(), InvalidArgumentException,
                "AuxiliaryModelBuilder: empty model name");
}

void AuxiliaryModelBuilder::validateUniqueName(const std::string& name) const
{
    FE_THROW_IF(name.empty(), InvalidArgumentException,
                "AuxiliaryModelBuilder('" + name_ + "'): empty declaration name");
    FE_THROW_IF(name.find('/') != std::string::npos, InvalidArgumentException,
                "AuxiliaryModelBuilder('" + name_ + "'): name '" + name +
                "' contains '/' which is reserved as the instance/output separator");
    FE_THROW_IF(all_names_.count(name) != 0u, InvalidArgumentException,
                "AuxiliaryModelBuilder('" + name_ + "'): duplicate name '" + name + "'");
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::input(std::string name, int size)
{
    validateUniqueName(name);
    FE_THROW_IF(size <= 0, InvalidArgumentException,
                "AuxiliaryModelBuilder: input size must be > 0");
    all_names_.insert(name);
    inputs_.push_back({std::move(name), size, false});
    return *this;
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::state(
    std::string name, AuxiliaryVariableKind kind)
{
    validateUniqueName(name);
    all_names_.insert(name);
    StateDecl sd;
    sd.name = std::move(name);
    sd.kind = kind;
    states_.push_back(std::move(sd));
    return *this;
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::param(std::string name)
{
    validateUniqueName(name);
    all_names_.insert(name);
    params_.push_back({std::move(name), 1, false, std::nullopt});
    return *this;
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::param(std::string name, Real default_value)
{
    validateUniqueName(name);
    all_names_.insert(name);
    params_.push_back({std::move(name), 1, true, default_value});
    return *this;
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::optionalInput(
    std::string name, Real default_value, int size)
{
    validateUniqueName(name);
    FE_THROW_IF(size <= 0, InvalidArgumentException,
                "AuxiliaryModelBuilder: input size must be > 0");
    all_names_.insert(name);
    inputs_.push_back({std::move(name), size, true, default_value});
    return *this;
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::derived(
    std::string name, forms::FormExpr expr)
{
    validateUniqueName(name);
    all_names_.insert(name);
    derived_.emplace_back(std::move(name), std::move(expr));
    return *this;
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::ode(
    const std::string& state_name, forms::FormExpr rhs)
{
    // Verify the state was declared.
    bool found = false;
    for (const auto& sd : states_) {
        if (sd.name == state_name) {
            FE_THROW_IF(sd.kind != AuxiliaryVariableKind::Differential,
                        InvalidArgumentException,
                        "AuxiliaryModelBuilder('" + name_ + "'): ode() requires "
                            "Differential state '" + state_name + "'");
            found = true;
            break;
        }
    }
    FE_THROW_IF(!found, InvalidArgumentException,
                "AuxiliaryModelBuilder('" + name_ + "'): undeclared state '"
                    + state_name + "' in ode()");

    // Check for duplicate equation targeting the same state.
    for (const auto& row : rows_) {
        FE_THROW_IF(row.state_name == state_name, InvalidArgumentException,
                    "AuxiliaryModelBuilder('" + name_ + "'): duplicate equation "
                        "for state '" + state_name + "'");
    }

    rows_.push_back({AuxiliaryRowKind::ODE, state_name, std::move(rhs)});
    return *this;
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::algebraic(
    const std::string& state_name, forms::FormExpr expr)
{
    bool found = false;
    for (const auto& sd : states_) {
        if (sd.name == state_name) {
            FE_THROW_IF(sd.kind != AuxiliaryVariableKind::Algebraic,
                        InvalidArgumentException,
                        "AuxiliaryModelBuilder('" + name_ + "'): algebraic() requires "
                            "Algebraic state '" + state_name + "'");
            found = true;
            break;
        }
    }
    FE_THROW_IF(!found, InvalidArgumentException,
                "AuxiliaryModelBuilder('" + name_ + "'): undeclared state '"
                    + state_name + "' in algebraic()");

    // Check for duplicate equation targeting the same state.
    for (const auto& row : rows_) {
        FE_THROW_IF(row.state_name == state_name, InvalidArgumentException,
                    "AuxiliaryModelBuilder('" + name_ + "'): duplicate equation "
                        "for state '" + state_name + "'");
    }

    rows_.push_back({AuxiliaryRowKind::Algebraic, state_name, std::move(expr)});
    return *this;
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::residual(
    std::string name, forms::FormExpr expr)
{
    rows_.push_back({AuxiliaryRowKind::Residual, std::move(name), std::move(expr)});
    return *this;
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::output(
    std::string name, forms::FormExpr expr)
{
    validateUniqueName(name);
    all_names_.insert(name);
    outputs_.emplace_back(std::move(name), std::move(expr));
    return *this;
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::unusedSymbolPolicy(const std::string& policy)
{
    FE_THROW_IF(policy != "warn" && policy != "error" && policy != "silent",
                InvalidArgumentException,
                "AuxiliaryModelBuilder: unusedSymbolPolicy must be 'warn', 'error', or 'silent'");
    unused_symbol_policy_ = policy;
    return *this;
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::derivatives(
    AuxiliaryDerivativePolicy policy)
{
    deriv_policy_ = policy;
    return *this;
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::insertEquation(
    int kind, std::string target_name, forms::FormExpr rhs)
{
    switch (kind) {
        case 0: // ODE
            return ode(target_name, std::move(rhs));
        case 1: // Algebraic
            return algebraic(target_name, std::move(rhs));
        case 2: // Output
            return output(std::move(target_name), std::move(rhs));
        default:
            FE_THROW(InvalidArgumentException,
                     "AuxiliaryModelBuilder('" + name_ + "'): unknown equation kind "
                         + std::to_string(kind));
    }
}

std::vector<std::string> AuxiliaryModelBuilder::stateNames() const
{
    std::vector<std::string> result;
    result.reserve(states_.size());
    for (const auto& sd : states_) result.push_back(sd.name);
    return result;
}

std::vector<std::string> AuxiliaryModelBuilder::inputNames() const
{
    std::vector<std::string> result;
    result.reserve(inputs_.size());
    for (const auto& p : inputs_) result.push_back(p.name);
    return result;
}

std::vector<std::string> AuxiliaryModelBuilder::parameterNames() const
{
    std::vector<std::string> result;
    result.reserve(params_.size());
    for (const auto& p : params_) result.push_back(p.name);
    return result;
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::setBounds(
    const std::string& name, Real lower, Real upper)
{
    for (auto& s : states_) {
        if (s.name == name) { s.lower_bound = lower; s.upper_bound = upper; return *this; }
    }
    for (auto& p : params_) {
        if (p.name == name) { p.lower_bound = lower; p.upper_bound = upper; return *this; }
    }
    FE_THROW(InvalidArgumentException,
             "AuxiliaryModelBuilder('" + name_ + "'): setBounds: unknown symbol '" + name + "'");
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::setNonnegative(const std::string& name)
{
    return setBounds(name, 0.0, std::numeric_limits<Real>::infinity());
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::setScale(const std::string& name, Real s)
{
    for (auto& st : states_) { if (st.name == name) { st.scale = s; return *this; } }
    FE_THROW(InvalidArgumentException,
             "AuxiliaryModelBuilder('" + name_ + "'): setScale: unknown state '" + name + "'");
}

AuxiliaryModelBuilder& AuxiliaryModelBuilder::setInitialGuess(const std::string& name, Real guess)
{
    for (auto& st : states_) {
        if (st.name == name) {
            FE_THROW_IF(st.kind != AuxiliaryVariableKind::Algebraic, InvalidArgumentException,
                        "AuxiliaryModelBuilder('" + name_ + "'): setInitialGuess requires "
                            "Algebraic state '" + name + "'");
            st.initial_guess = guess;
            return *this;
        }
    }
    FE_THROW(InvalidArgumentException,
             "AuxiliaryModelBuilder('" + name_ + "'): setInitialGuess: unknown state '" + name + "'");
}

std::vector<std::string> AuxiliaryModelBuilder::outputNames() const
{
    std::vector<std::string> result;
    result.reserve(outputs_.size());
    for (const auto& [n, e] : outputs_) result.push_back(n);
    return result;
}

std::string AuxiliaryModelBuilder::summary() const
{
    std::string out;
    out += "Model: " + name_ + "\n";

    if (!states_.empty()) {
        out += "  States (" + std::to_string(states_.size()) + "):\n";
        for (const auto& s : states_) {
            out += "    " + s.name;
            out += (s.kind == AuxiliaryVariableKind::Algebraic) ? " [algebraic]" : " [differential]";
            if (s.lower_bound) out += " lower=" + std::to_string(*s.lower_bound);
            if (s.upper_bound) out += " upper=" + std::to_string(*s.upper_bound);
            if (s.scale) out += " scale=" + std::to_string(*s.scale);
            if (s.initial_guess) out += " guess=" + std::to_string(*s.initial_guess);
            out += "\n";
        }
    }

    if (!inputs_.empty()) {
        out += "  Inputs (" + std::to_string(inputs_.size()) + "):\n";
        for (const auto& p : inputs_) {
            out += "    " + p.name;
            if (p.size > 1) out += "[" + std::to_string(p.size) + "]";
            if (p.optional) out += " [optional]";
            if (p.default_value) out += " default=" + std::to_string(*p.default_value);
            out += "\n";
        }
    }

    if (!params_.empty()) {
        out += "  Parameters (" + std::to_string(params_.size()) + "):\n";
        for (const auto& p : params_) {
            out += "    " + p.name;
            if (p.optional) out += " [optional]";
            if (p.default_value) out += " default=" + std::to_string(*p.default_value);
            out += "\n";
        }
    }

    if (!derived_.empty()) {
        out += "  Intermediates (" + std::to_string(derived_.size()) + "):\n";
        for (const auto& [n, e] : derived_) {
            out += "    " + n + "\n";
        }
    }

    if (!rows_.empty()) {
        out += "  Equations (" + std::to_string(rows_.size()) + "):\n";
        for (const auto& row : rows_) {
            out += "    ";
            switch (row.kind) {
                case AuxiliaryRowKind::ODE: out += "ddt(" + row.state_name + ") == ..."; break;
                case AuxiliaryRowKind::Algebraic: out += "alg(" + row.state_name + ") == ..."; break;
                case AuxiliaryRowKind::Residual: out += "residual(" + row.state_name + ") == ..."; break;
            }
            out += "\n";
        }
    }

    if (!outputs_.empty()) {
        out += "  Outputs (" + std::to_string(outputs_.size()) + "):\n";
        for (const auto& [n, e] : outputs_) {
            out += "    " + n + "\n";
        }
    }

    return out;
}

std::shared_ptr<BuiltAuxiliaryModel> AuxiliaryModelBuilder::build() const
{
    // Validate: must have at least one state.
    FE_THROW_IF(states_.empty(), InvalidArgumentException,
                "AuxiliaryModelBuilder('" + name_ + "'): no states declared");

    // Validate: every declared state must have exactly one governing equation.
    // Detect missing equations and duplicates with specific diagnostics.
    {
        std::unordered_map<std::string, int> state_row_count;
        for (const auto& sd : states_) {
            state_row_count[sd.name] = 0;
        }
        for (const auto& row : rows_) {
            auto it = state_row_count.find(row.state_name);
            FE_THROW_IF(it == state_row_count.end(), InvalidArgumentException,
                        "AuxiliaryModelBuilder('" + name_ + "'): equation targets "
                            "undeclared state '" + row.state_name + "'");
            it->second++;
            FE_THROW_IF(it->second > 1, InvalidArgumentException,
                        "AuxiliaryModelBuilder('" + name_ + "'): duplicate equation "
                            "for state '" + row.state_name + "'");
        }
        for (const auto& [sname, count] : state_row_count) {
            FE_THROW_IF(count == 0, InvalidArgumentException,
                        "AuxiliaryModelBuilder('" + name_ + "'): declared state '"
                            + sname + "' has no governing equation");
        }
    }

    // Warn about declared inputs and parameters that are never referenced
    // in any equation or output expression.  This catches typos and stale
    // declarations in large models.
    {
        std::unordered_set<std::string> referenced_inputs;
        std::unordered_set<std::string> referenced_params;

        auto scanExpr = [&](const forms::FormExpr& expr) {
            if (!expr.isValid() || !expr.node()) return;
            std::function<void(const forms::FormExprNode&)> walk =
                [&](const forms::FormExprNode& n) {
                    if (n.type() == forms::FormExprType::AuxiliaryInputSymbol) {
                        if (auto s = n.symbolName()) referenced_inputs.insert(std::string(*s));
                    }
                    if (n.type() == forms::FormExprType::ParameterSymbol) {
                        if (auto s = n.symbolName()) referenced_params.insert(std::string(*s));
                    }
                    for (const auto* c : n.children()) {
                        if (c) walk(*c);
                    }
                };
            walk(*expr.node());
        };

        for (const auto& row : rows_) scanExpr(row.expression);
        for (const auto& [oname, oexpr] : outputs_) scanExpr(oexpr);

        // Report unused symbols based on the configured policy.
        auto report = [&](const std::string& category, const std::string& sym_name) {
            if (unused_symbol_policy_ == "error") {
                FE_THROW(InvalidArgumentException,
                         "AuxiliaryModelBuilder('" + name_ + "'): declared " +
                             category + " '" + sym_name +
                             "' is never referenced in any equation or output");
            } else if (unused_symbol_policy_ == "warn") {
                std::fprintf(stderr,
                    "[AuxiliaryModelBuilder] model '%s': declared %s '%s' "
                    "is never referenced in any equation or output\n",
                    name_.c_str(), category.c_str(), sym_name.c_str());
            }
            // "silent": do nothing.
        };

        for (const auto& inp : inputs_) {
            if (!inp.optional && referenced_inputs.count(inp.name) == 0) {
                report("input", inp.name);
            }
        }
        for (const auto& p : params_) {
            if (!p.optional && referenced_params.count(p.name) == 0) {
                report("parameter", p.name);
            }
        }
    }

    const int dim = static_cast<int>(states_.size());

    // Build structural metadata.
    AuxiliaryStructuralMetadata metadata;
    metadata.variable_kinds.reserve(static_cast<std::size_t>(dim));
    for (const auto& sd : states_) {
        metadata.variable_kinds.push_back(sd.kind);
    }

    // Build state names list and initial guesses.
    std::vector<std::string> state_names;
    std::vector<std::optional<Real>> initial_guesses;
    state_names.reserve(static_cast<std::size_t>(dim));
    initial_guesses.reserve(static_cast<std::size_t>(dim));
    for (const auto& sd : states_) {
        state_names.push_back(sd.name);
        initial_guesses.push_back(sd.initial_guess);
    }

    // Build residual expressions (one per state, in declaration order).
    // For ODE rows, store the RHS expression (dot(x) - rhs lowering
    // is handled at evaluation time).
    // For algebraic rows, store the constraint expression.
    std::vector<forms::FormExpr> residual_exprs;
    residual_exprs.reserve(static_cast<std::size_t>(dim));

    // Map state names to row indices.
    std::unordered_map<std::string, std::size_t> state_to_row;
    for (std::size_t i = 0; i < states_.size(); ++i) {
        state_to_row[states_[i].name] = i;
    }

    // Build name→slot maps for symbol resolution.
    std::unordered_map<std::string, std::size_t> state_name_to_slot;
    for (std::size_t i = 0; i < states_.size(); ++i) {
        state_name_to_slot[states_[i].name] = i;
    }
    std::unordered_map<std::string, std::size_t> input_name_to_slot;
    {
        std::size_t flat_offset = 0;
        for (std::size_t i = 0; i < inputs_.size(); ++i) {
            input_name_to_slot[inputs_[i].name] = flat_offset;
            flat_offset += static_cast<std::size_t>(inputs_[i].size);
        }
    }
    std::unordered_map<std::string, std::size_t> param_name_to_slot;
    for (std::size_t i = 0; i < params_.size(); ++i) {
        param_name_to_slot[params_[i].name] = i;
    }

    // Resolution transform: symbolic names → slot-indexed refs.
    auto resolve = [&](const forms::FormExprNode& node) -> std::optional<forms::FormExpr> {
        auto sym = node.symbolName();
        if (!sym) return std::nullopt;
        const std::string name_str{*sym};

        switch (node.type()) {
            case forms::FormExprType::AuxiliaryStateSymbol: {
                auto it = state_name_to_slot.find(name_str);
                if (it != state_name_to_slot.end())
                    return forms::FormExpr::auxiliaryStateRef(static_cast<std::uint32_t>(it->second));
                break;
            }
            case forms::FormExprType::AuxiliaryInputSymbol: {
                auto it = input_name_to_slot.find(name_str);
                if (it != input_name_to_slot.end())
                    return forms::FormExpr::auxiliaryInputRef(static_cast<std::uint32_t>(it->second));
                break;
            }
            case forms::FormExprType::ParameterSymbol: {
                auto it = param_name_to_slot.find(name_str);
                if (it != param_name_to_slot.end())
                    return forms::FormExpr::parameterRef(static_cast<std::uint32_t>(it->second));
                break;
            }
            default:
                break;
        }
        return std::nullopt;
    };

    // Order rows to match state declaration order, then resolve symbols.
    residual_exprs.resize(static_cast<std::size_t>(dim));
    for (const auto& row : rows_) {
        auto it = state_to_row.find(row.state_name);
        FE_THROW_IF(it == state_to_row.end(), InvalidArgumentException,
                    "AuxiliaryModelBuilder('" + name_ +
                        "'): row references unknown state '" + row.state_name + "'");
        residual_exprs[it->second] = row.expression.transformNodes(resolve);
    }

    // Build port descriptors.
    std::vector<AuxiliaryModelPortDescriptor> out_inputs = inputs_;
    std::vector<AuxiliaryModelPortDescriptor> out_outputs;
    for (const auto& [oname, oexpr] : outputs_) {
        out_outputs.push_back({oname, 1, false});
    }
    std::vector<AuxiliaryModelPortDescriptor> out_params = params_;

    auto model = std::make_shared<BuiltAuxiliaryModel>(
        name_, dim, std::move(metadata), std::move(residual_exprs),
        std::move(state_names), std::move(out_inputs),
        std::move(out_outputs), std::move(out_params), deriv_policy_);

    // Resolve and attach output expressions.
    std::vector<std::pair<std::string, forms::FormExpr>> resolved_outputs;
    for (const auto& [oname, oexpr] : outputs_) {
        resolved_outputs.emplace_back(oname, oexpr.transformNodes(resolve));
    }
    model->setOutputExpressions(std::move(resolved_outputs));
    model->setInitialGuesses(std::move(initial_guesses));

    return model;
}

} // namespace systems
} // namespace FE
} // namespace svmp
