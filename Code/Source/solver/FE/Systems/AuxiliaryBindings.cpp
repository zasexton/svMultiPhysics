#include "Systems/AuxiliaryBindings.h"

#include <unordered_set>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  AuxiliaryDeployedInstance
// ---------------------------------------------------------------------------

AuxiliaryDeployedInstance::AuxiliaryDeployedInstance(
    std::shared_ptr<AuxiliaryStateModel> model)
    : model_(std::move(model))
{
    FE_THROW_IF(!model_, InvalidArgumentException,
                "AuxiliaryDeployedInstance: null model");
    instance_name_ = model_->modelName();
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::name(std::string n)
{
    instance_name_ = std::move(n);
    has_explicit_name_ = true;
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::scope(AuxiliaryStateScope s)
{
    scope_ = s;
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::region(AuxiliaryDeploymentRegion r)
{
    region_ = std::move(r);
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::solveMode(AuxiliarySolveMode m)
{
    solve_mode_ = m;
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::schedule(AuxiliaryScheduleMode m)
{
    schedule_ = m;
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::stepper(AuxiliaryStepperSpec spec)
{
    stepper_spec_ = std::move(spec);
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::bind(
    const std::string& model_input, const std::string& registry_input)
{
    input_bindings_[model_input] = registry_input;
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::bind(
    const std::string& model_input, const AuxiliaryInputHandle& handle)
{
    input_bindings_[model_input] = handle.registryName();
    coupled_bindings_[model_input] = handle;
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::bindCoupled(
    const std::string& model_input, const AuxiliaryInputHandle& handle)
{
    return bind(model_input, handle);
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::param(
    const std::string& param_name, Real value)
{
    param_values_[param_name] = value;
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::initialize(std::vector<Real> values)
{
    initial_values_ = std::move(values);
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::entityCount(std::size_t count)
{
    entity_count_ = count;
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::qpOffsets(
    std::vector<std::size_t> offsets)
{
    qp_offsets_ = std::move(offsets);
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::layoutMode(AuxiliaryLayoutMode m)
{
    layout_mode_ = m;
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::entityOrdering(AuxiliaryEntityOrdering o)
{
    entity_ordering_ = o;
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::params(
    std::initializer_list<std::pair<std::string, Real>> values)
{
    for (const auto& [pname, pval] : values) {
        param_values_[pname] = pval;
    }
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::initialState(
    std::initializer_list<std::pair<std::string, Real>> named_values)
{
    // Resolve state names to positional indices via BuiltAuxiliaryModel.
    auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(model_.get());
    FE_THROW_IF(!built, InvalidArgumentException,
                "AuxiliaryDeployedInstance::initialState(): model '" +
                    (model_ ? model_->modelName() : "<null>") +
                    "' is not a BuiltAuxiliaryModel — use .initialize(vector) instead");

    const auto& state_names = built->stateNames();
    const auto dim = static_cast<std::size_t>(built->dimension());
    initial_values_.resize(dim);

    // Track which states were provided.
    std::vector<bool> provided(dim, false);

    for (const auto& [sname, sval] : named_values) {
        bool found = false;
        for (std::size_t i = 0; i < state_names.size(); ++i) {
            if (state_names[i] == sname) {
                initial_values_[i] = sval;
                provided[i] = true;
                found = true;
                break;
            }
        }
        FE_THROW_IF(!found, InvalidArgumentException,
                    "AuxiliaryDeployedInstance::initialState(): unknown state '"
                        + sname + "' in model '" + built->modelName() + "'");
    }

    // For states not explicitly provided, use algebraic initial guesses
    // if available.  Otherwise require explicit values.
    const auto& guesses = built->initialGuesses();
    std::string missing;
    for (std::size_t i = 0; i < dim; ++i) {
        if (!provided[i]) {
            if (i < guesses.size() && guesses[i].has_value()) {
                initial_values_[i] = *guesses[i];
                provided[i] = true;
            } else {
                if (!missing.empty()) missing += ", ";
                missing += "'" + state_names[i] + "'";
            }
        }
    }
    FE_THROW_IF(!missing.empty(), InvalidArgumentException,
                "AuxiliaryDeployedInstance::initialState(): model '" +
                    built->modelName() + "' has " + std::to_string(dim) +
                    " states but initialState() is missing: " + missing +
                    ". Provide explicit values for all states, or use "
                    ".initialize(vector) for positional initialization. "
                    "Algebraic states with .initialGuess() are auto-filled.");
    return *this;
}

AuxiliaryDeployedInstance& AuxiliaryDeployedInstance::derivatives(AuxiliaryDerivativePolicy policy)
{
    derivative_policy_ = policy;
    has_explicit_derivative_policy_ = true;
    return *this;
}

std::string AuxiliaryDeployedInstance::validate() const
{
    if (instance_name_.empty()) {
        return "instance name is empty";
    }
    if (instance_name_.find('/') != std::string::npos) {
        return "instance name '" + instance_name_ +
               "' contains '/' which is reserved as the instance/output separator";
    }

    if (!model_) {
        return "model is null";
    }

    // Stepper must not be explicitly configured for Monolithic instances.
    if (solve_mode_ == AuxiliarySolveMode::Monolithic &&
        stepper_spec_.method_name != "BackwardEuler") {
        return "Monolithic instance '" + instance_name_ +
               "' has explicit stepper '" + stepper_spec_.method_name +
               "'; Monolithic blocks use the global assembled time discretization, "
               "not a local stepper";
    }
    if (solve_mode_ == AuxiliarySolveMode::Monolithic &&
        stepper_spec_.substep_count > 1) {
        return "Monolithic instance '" + instance_name_ +
               "' has substep_count=" + std::to_string(stepper_spec_.substep_count) +
               "; Monolithic blocks cannot substep independently";
    }

    // Check initial values size if provided.
    if (!initial_values_.empty()) {
        const auto dim = static_cast<std::size_t>(model_->dimension());
        if (initial_values_.size() != dim) {
            return "initial_values size (" +
                   std::to_string(initial_values_.size()) +
                   ") != model dimension (" + std::to_string(dim) + ")";
        }
    }

    // Validate output contract consistency.
    {
        const auto n_out = model_->outputCount();
        const auto out_names = model_->outputNames();
        if (n_out < 0) {
            return "outputCount() is negative (" + std::to_string(n_out) + ")";
        }
        if (static_cast<int>(out_names.size()) != n_out) {
            return "outputNames().size() (" + std::to_string(out_names.size()) +
                   ") != outputCount() (" + std::to_string(n_out) + ")";
        }
        // Check for duplicate or invalid output names within the instance.
        std::unordered_set<std::string> seen_names;
        for (const auto& oname : out_names) {
            if (oname.find('/') != std::string::npos) {
                return "output name '" + oname + "' in model '" +
                       model_->modelName() +
                       "' contains '/' which is reserved as the "
                       "instance/output separator";
            }
            if (!seen_names.insert(oname).second) {
                return "duplicate output name '" + oname + "' in model '" +
                       model_->modelName() + "'";
            }
        }
    }

    // Check that the model has inputs and params matching the signature.
    if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(model_.get())) {
        const auto& sig = built->signature();
        for (const auto& inp : sig.inputs) {
            if (!inp.optional && input_bindings_.find(inp.name) == input_bindings_.end()) {
                return "required input '" + inp.name + "' is not bound";
            }
        }
        for (const auto& p : sig.parameters) {
            if (!p.optional && param_values_.find(p.name) == param_values_.end()) {
                return "required parameter '" + p.name + "' is not set";
            }
        }
        // Reject unknown parameter names.
        for (const auto& [pname, pval] : param_values_) {
            bool known = false;
            for (const auto& p : sig.parameters) {
                if (p.name == pname) { known = true; break; }
            }
            if (!known) {
                return "unknown parameter '" + pname + "' in model '" +
                       built->modelName() + "'";
            }
        }
        // Reject unknown initial-state names (already resolved to positional
        // in initialState(), but check size consistency).
    }

    if (!qp_offsets_.empty()) {
        if (scope_ != AuxiliaryStateScope::QuadraturePoint) {
            return "instance '" + instance_name_ +
                   "' provides qpOffsets() but is not QuadraturePoint scoped";
        }
        if (qp_offsets_.size() < 2u) {
            return "instance '" + instance_name_ +
                   "' has qpOffsets() with fewer than 2 entries";
        }
        if (qp_offsets_.front() != 0u) {
            return "instance '" + instance_name_ +
                   "' has qpOffsets() whose first entry is not 0";
        }
        for (std::size_t i = 1; i < qp_offsets_.size(); ++i) {
            if (qp_offsets_[i] < qp_offsets_[i - 1]) {
                return "instance '" + instance_name_ +
                       "' has non-monotone qpOffsets()";
            }
        }
        if (entity_count_ != 0 && entity_count_ != qp_offsets_.back()) {
            return "instance '" + instance_name_ +
                   "' has entityCount()=" + std::to_string(entity_count_) +
                   " but qpOffsets().back()=" + std::to_string(qp_offsets_.back());
        }
    }

    // Validate FE-backed handle bindings.
    if (!coupled_bindings_.empty()) {
        if (solve_mode_ == AuxiliarySolveMode::Monolithic) {
            for (const auto& [model_input, handle] : coupled_bindings_) {
                if (handle.hasDefinition() && !handle.supportsMonolithicLinearization()) {
                    return "instance '" + instance_name_ + "': handle binding '" +
                           model_input + "' references FE quantity '" +
                           handle.registryName() + "' which does not support "
                           "monolithic linearization (dI/du)";
                }
            }
        }
    }

    // Validate shape consistency for all bindings with FE quantity metadata.
    // For BuiltAuxiliaryModel, each model input has size 1 (scalar) unless
    // explicitly multi-component.  FE-backed handles with vector/tensor shape
    // must match the expected input size.
    if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(model_.get())) {
        const auto& sig = built->signature();
        for (const auto& [model_input, reg_input] : input_bindings_) {
            // Find the model input size.
            int expected_size = 1;
            for (const auto& inp : sig.inputs) {
                if (inp.name == model_input) {
                    expected_size = inp.size;
                    break;
                }
            }

            // Check FE-backed handle bindings for shape mismatch.
            auto cb_it = coupled_bindings_.find(model_input);
            if (cb_it != coupled_bindings_.end() && cb_it->second.hasDefinition()) {
                const auto actual_size = cb_it->second.shape().components;
                if (actual_size != expected_size) {
                    return "instance '" + instance_name_ + "': binding '" +
                           model_input + "' expects " + std::to_string(expected_size) +
                           " component(s) but FE quantity '" + reg_input +
                           "' has shape with " + std::to_string(actual_size) +
                           " component(s)";
                }
            }
        }
    }

    return {}; // valid
}

} // namespace systems
} // namespace FE
} // namespace svmp
