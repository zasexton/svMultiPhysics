#ifndef SVMP_FE_AUXILIARY_BINDINGS_H
#define SVMP_FE_AUXILIARY_BINDINGS_H

/**
 * @file AuxiliaryBindings.h
 * @brief Deployment and binding API for auxiliary models.
 *
 * ## Preferred deployment workflow (math-first DSL)
 *
 * ```cpp
 * auto rcr = system.deploy(
 *     use(rcr_model)
 *         .name("outlet_rcr")
 *         .boundary(marker)
 *         .partitioned("BackwardEuler")
 *         .params({{"Rp", 100.0}, {"C", 0.001}, {"Rd", 1000.0}, {"Pd", 0.0}})
 *         .bind(Q)
 *         .initialState({{"X", 0.0}})
 * );
 * auto p_out = rcr.output("P_out");
 * ```
 *
 * ## Legacy deployment workflow
 *
 * ```cpp
 * system.deployAuxiliaryModel(
 *     use(model)
 *         .name("instance")
 *         .scope(AuxiliaryStateScope::Global)
 *         .solveMode(AuxiliarySolveMode::Partitioned)
 *         .stepper({"BackwardEuler"})
 *         .bind("Q", "flow_rate")
 *         .param("R_p", 100.0)
 *         .initialize({0.0}));
 * ```
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Forms/FormExpr.h"

#include "Auxiliary/AuxiliaryStateTypes.h"
#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Systems/FEQuantityDefinition.h"

#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// Forward declarations
class AuxiliaryInputHandle;

// ============================================================================
//  AuxiliaryInputHandle
// ============================================================================

/**
 * @brief Typed handle for an FE-coupled auxiliary input.
 *
 * Returned by `FESystem::boundaryIntegral()`, `FESystem::derivedInput()`,
 * and related registration APIs.  Convertible to `FormExpr` for use in
 * expressions and deployable via handle-based `.bind()`.
 *
 * Carries optional FE-backed quantity metadata (kind, shape, referenced
 * fields, capability flags) when created through an FEQuantityRegistry-
 * aware registration path.  Plain derived/callback inputs are also
 * representable (definition is null for those).
 */
class AuxiliaryInputHandle {
public:
    AuxiliaryInputHandle() = default;

    /// Construct with registry name only (no FE quantity metadata).
    explicit AuxiliaryInputHandle(std::string registry_name)
        : registry_name_(std::move(registry_name)) {}

    /// Construct with registry name and FE quantity definition.
    AuxiliaryInputHandle(std::string registry_name,
                          std::shared_ptr<const FEQuantityDefinition> def)
        : registry_name_(std::move(registry_name))
        , definition_(std::move(def)) {}

    /// The registry input name (for binding and lookup).
    [[nodiscard]] const std::string& registryName() const noexcept
    {
        return registry_name_;
    }

    /// Convert to a FormExpr referencing this auxiliary input.
    [[nodiscard]] operator forms::FormExpr() const  // NOLINT(google-explicit-constructor)
    {
        return forms::FormExpr::auxiliaryInput(registry_name_);
    }

    /// Explicit conversion to FormExpr.
    [[nodiscard]] forms::FormExpr expr() const
    {
        return forms::FormExpr::auxiliaryInput(registry_name_);
    }

    // ---- FE quantity metadata ----

    /// Whether this handle carries FE-backed quantity metadata.
    [[nodiscard]] bool hasDefinition() const noexcept
    {
        return definition_ != nullptr;
    }

    /// The FE quantity definition, or nullptr if not FE-backed.
    [[nodiscard]] const FEQuantityDefinition* definition() const noexcept
    {
        return definition_.get();
    }

    /// Quantity kind (FEQuantityKind::DerivedCallback if no definition).
    [[nodiscard]] FEQuantityKind kind() const noexcept
    {
        return definition_ ? definition_->kind : FEQuantityKind::DerivedCallback;
    }

    /// Result shape (scalar if no definition).
    [[nodiscard]] FEQuantityShape shape() const noexcept
    {
        return definition_ ? definition_->shape : FEQuantityShape::scalar();
    }

    /// Whether explicit (partitioned) evaluation is supported.
    [[nodiscard]] bool supportsExplicitEvaluation() const noexcept
    {
        return !definition_ || definition_->capabilities.explicit_evaluation;
    }

    /// Whether exact monolithic linearization (dI/du) is supported.
    [[nodiscard]] bool supportsMonolithicLinearization() const noexcept
    {
        return definition_ && definition_->capabilities.monolithic_linearization;
    }

    /// FE fields referenced by this quantity (empty if not FE-backed).
    [[nodiscard]] const std::vector<FieldId>& referencedFields() const noexcept
    {
        static const std::vector<FieldId> empty;
        return definition_ ? definition_->referenced_fields : empty;
    }

private:
    std::string registry_name_{};
    std::shared_ptr<const FEQuantityDefinition> definition_{};
};

// ============================================================================
//  Shape helpers for FE-backed quantity handles
// ============================================================================

/**
 * @brief Extract the i-th component of a vector/tensor FE quantity.
 *
 * ```cpp
 * auto uz = comp(u_handle, 2);
 * ```
 */
[[nodiscard]] inline forms::FormExpr comp(const AuxiliaryInputHandle& h, int i)
{
    return forms::FormExpr::auxiliaryInput(
        h.registryName() + ":c" + std::to_string(i));
}

/**
 * @brief Inner product of a vector FE quantity handle with another handle.
 *
 * Both handles must have the same number of components.
 *
 * ```cpp
 * auto kinetic = dot(u_handle, u_handle);  // |u|^2
 * ```
 */
[[nodiscard]] inline forms::FormExpr dot(const AuxiliaryInputHandle& a,
                                          const AuxiliaryInputHandle& b)
{
    const int n = a.shape().components;
    if (n <= 1) return a.expr() * b.expr();
    auto result = comp(a, 0) * comp(b, 0);
    for (int i = 1; i < n; ++i) {
        result = result + comp(a, i) * comp(b, i);
    }
    return result;
}

/**
 * @brief Trace of a tensor FE quantity handle.
 *
 * ```cpp
 * auto tr = trace(sigma_handle);
 * ```
 */
[[nodiscard]] inline forms::FormExpr trace(const AuxiliaryInputHandle& h)
{
    const auto shape = h.shape();
    if (shape.kind != FEQuantityShapeKind::Tensor || shape.spatial_dim <= 0) {
        return h.expr();  // degenerate: treat as scalar
    }
    const int dim = shape.spatial_dim;
    auto result = comp(h, 0);  // (0,0)
    for (int i = 1; i < dim; ++i) {
        result = result + comp(h, i * dim + i);
    }
    return result;
}

/**
 * @brief Euclidean norm of a vector FE quantity handle.
 *
 * ```cpp
 * auto speed = norm(u_handle);
 * ```
 */
[[nodiscard]] inline forms::FormExpr norm(const AuxiliaryInputHandle& h)
{
    return forms::sqrt(dot(h, h));
}

// ============================================================================
//  AuxiliaryInstanceHandle
// ============================================================================

/**
 * @brief Typed handle for a deployed auxiliary model instance.
 *
 * Returned by `FESystem::deploy()`.  Provides handle-based access to
 * instance outputs without string concatenation.
 */
class AuxiliaryInstanceHandle {
public:
    AuxiliaryInstanceHandle() = default;
    explicit AuxiliaryInstanceHandle(std::string instance_name)
        : instance_name_(std::move(instance_name)) {}

    /// The deployed instance name.
    [[nodiscard]] const std::string& instanceName() const noexcept
    {
        return instance_name_;
    }

    /**
     * @brief Get a FormExpr referencing a named output of this instance.
     *
     * ```cpp
     * auto rcr = system.deploy(use(model).name("rcr_1")...);
     * auto p_out = rcr.output("P_out");
     * // Equivalent to: FormExpr::auxiliaryOutput("rcr_1/P_out")
     * ```
     */
    [[nodiscard]] forms::FormExpr output(const std::string& output_name) const
    {
        return forms::FormExpr::auxiliaryOutput(instance_name_ + "/" + output_name);
    }

    /// Get a FormExpr referencing a named input of this instance (advanced).
    [[nodiscard]] forms::FormExpr input(const std::string& input_name) const
    {
        return forms::FormExpr::auxiliaryInput(instance_name_ + "/" + input_name);
    }

    /// Get a FormExpr referencing a named state of this instance (advanced).
    [[nodiscard]] forms::FormExpr state(const std::string& state_name) const
    {
        return forms::FormExpr::auxiliaryState(instance_name_ + "/" + state_name);
    }

private:
    std::string instance_name_{};
};

// ============================================================================
//  AuxiliaryDeployedInstance
// ============================================================================

/**
 * @brief A deployed auxiliary model instance with formulation-time bindings.
 *
 * Created by `use(model)`.  Configured via fluent setters before
 * finalization during `system.setup()`.
 */
class AuxiliaryDeployedInstance {
public:
    explicit AuxiliaryDeployedInstance(std::shared_ptr<AuxiliaryStateModel> model);

    /// Set the instance name (defaults to the model name until deployment finalization).
    AuxiliaryDeployedInstance& name(std::string instance_name);

    /// Set the storage scope.
    AuxiliaryDeployedInstance& scope(AuxiliaryStateScope s);

    /// Set the deployment region.
    AuxiliaryDeployedInstance& region(AuxiliaryDeploymentRegion r);

    /// Set the solve mode.
    AuxiliaryDeployedInstance& solveMode(AuxiliarySolveMode m);

    /// Set the advancement schedule.
    AuxiliaryDeployedInstance& schedule(AuxiliaryScheduleMode m);

    /// Set the storage layout mode (default: FixedStride).
    AuxiliaryDeployedInstance& layoutMode(AuxiliaryLayoutMode m);

    /// Set the entity-component ordering (default: ByEntityThenComponent).
    AuxiliaryDeployedInstance& entityOrdering(AuxiliaryEntityOrdering o);

    /// Set the stepper configuration (Partitioned only).
    AuxiliaryDeployedInstance& stepper(AuxiliaryStepperSpec spec);

    // ---- Convenience scope sugar ----

    /// Shorthand for `.scope(AuxiliaryStateScope::Global)`.
    AuxiliaryDeployedInstance& global()
    {
        return scope(AuxiliaryStateScope::Global);
    }

    /// Shorthand for `.scope(AuxiliaryStateScope::Node)`.
    AuxiliaryDeployedInstance& node()
    {
        return scope(AuxiliaryStateScope::Node);
    }

    /// Shorthand for `.scope(AuxiliaryStateScope::Cell)`.
    AuxiliaryDeployedInstance& cell()
    {
        return scope(AuxiliaryStateScope::Cell);
    }

    /// Shorthand for `.scope(AuxiliaryStateScope::QuadraturePoint)`.
    AuxiliaryDeployedInstance& quadraturePoint()
    {
        return scope(AuxiliaryStateScope::QuadraturePoint);
    }

    /// Shorthand for `.scope(AuxiliaryStateScope::Boundary)` with a boundary
    /// marker region.  Use for lumped-parameter models on a named boundary
    /// (e.g., RCR windkessel on an outlet cap).
    AuxiliaryDeployedInstance& boundary(int marker)
    {
        scope(AuxiliaryStateScope::Boundary);
        return region(AuxiliaryDeploymentRegion{
            AuxiliaryRegionKind::BoundarySet, std::to_string(marker)});
    }

    /// Shorthand for `.scope(AuxiliaryStateScope::Facet)`.
    AuxiliaryDeployedInstance& facet()
    {
        return scope(AuxiliaryStateScope::Facet);
    }

    // ---- Convenience solve-mode sugar ----

    /**
     * @brief Set Partitioned solve mode with the named stepper.
     *
     * ```cpp
     * .partitioned("BackwardEuler")
     * // equivalent to:
     * .solveMode(AuxiliarySolveMode::Partitioned).stepper({"BackwardEuler"})
     * ```
     */
    AuxiliaryDeployedInstance& partitioned(const std::string& stepper_name)
    {
        return solveMode(AuxiliarySolveMode::Partitioned)
            .stepper(AuxiliaryStepperSpec{stepper_name});
    }

    /// Shorthand for `.solveMode(AuxiliarySolveMode::Monolithic)`.
    AuxiliaryDeployedInstance& monolithic()
    {
        return solveMode(AuxiliarySolveMode::Monolithic);
    }

    // ---- Schedule sugar ----

    /// Shorthand for `.schedule(AuxiliaryScheduleMode::SingleRate)`.
    AuxiliaryDeployedInstance& singleRate()
    {
        return schedule(AuxiliaryScheduleMode::SingleRate);
    }

    /// Shorthand for `.schedule(Subcycled).stepper({name, ..., substep_count})`.
    AuxiliaryDeployedInstance& subcycled(const std::string& stepper_name, int substep_count)
    {
        AuxiliaryStepperSpec spec{stepper_name};
        spec.substep_count = substep_count;
        return solveMode(AuxiliarySolveMode::Partitioned)
            .schedule(AuxiliaryScheduleMode::Subcycled)
            .stepper(std::move(spec));
    }

    // ---- Input binding ----

    /// Bind a model input to a named auxiliary input in the registry.
    AuxiliaryDeployedInstance& bind(const std::string& model_input,
                                     const std::string& registry_input);

    /// Bind a model input to an AuxiliaryInputHandle.
    AuxiliaryDeployedInstance& bind(const std::string& model_input,
                                     const AuxiliaryInputHandle& handle);

    /**
     * @brief Auto-bind an input handle by matching its registry name to a
     *        model input with the same name.
     *
     * ```cpp
     * auto Q = system.boundaryIntegral(...);
     * .bind(Q)  // binds model input "Q" to registry input "Q"
     * ```
     */
    AuxiliaryDeployedInstance& bind(const AuxiliaryInputHandle& handle)
    {
        return bind(handle.registryName(), handle);
    }

    /**
     * @brief Deprecated compatibility shim for exact monolithic FE coupling.
     *
     * FE-backed input handles now preserve their metadata through
     * `bind(model_input, handle)`. In monolithic solve mode the chain rule
     * `dF/du = dF/dI * dI/du` is assembled automatically for those bindings.
     *
     * ```cpp
     * use(model).monolithic().bind("Q", Q_handle);
     * ```
     */
    [[deprecated("bindCoupled() is deprecated; use bind(model_input, handle) instead")]]
    AuxiliaryDeployedInstance& bindCoupled(const std::string& model_input,
                                            const AuxiliaryInputHandle& handle);

    /// Auto-bind coupled by name match.
    [[deprecated("bindCoupled() is deprecated; use bind(handle) instead")]]
    AuxiliaryDeployedInstance& bindCoupled(const AuxiliaryInputHandle& handle)
    {
        return bindCoupled(handle.registryName(), handle);
    }

    // ---- Parameter setting ----

    /// Set a parameter to a literal value.
    AuxiliaryDeployedInstance& param(const std::string& param_name, Real value);

    /**
     * @brief Set multiple parameters at once.
     *
     * ```cpp
     * .params({{"Rp", 100.0}, {"C", 0.001}, {"Rd", 1000.0}, {"Pd", 0.0}})
     * ```
     */
    AuxiliaryDeployedInstance& params(
        std::initializer_list<std::pair<std::string, Real>> values);

    // ---- Initialization ----

    /// Set initial values for all state variables (positional, legacy).
    AuxiliaryDeployedInstance& initialize(std::vector<Real> values);

    /**
     * @brief Set initial state values by name.
     *
     * Maps state names to initial values using the model's state ordering.
     * Only works with `BuiltAuxiliaryModel` (which carries state names).
     *
     * ```cpp
     * .initialState({{"X", 0.0}, {"Y", 1.0}})
     * ```
     */
    AuxiliaryDeployedInstance& initialState(
        std::initializer_list<std::pair<std::string, Real>> named_values);

    // ---- Entity count ----

    /// Explicit entity count override.
    AuxiliaryDeployedInstance& entityCount(std::size_t count);

    /// Optional CSR-style QP offsets for QuadraturePoint scope.
    AuxiliaryDeployedInstance& qpOffsets(std::vector<std::size_t> offsets);

    // ---- Derivative policy ----

    /// Set the derivative policy explicitly.
    AuxiliaryDeployedInstance& derivatives(AuxiliaryDerivativePolicy policy);

    // -----------------------------------------------------------------
    //  Accessors (read after configuration)
    // -----------------------------------------------------------------

    [[nodiscard]] const std::string& instanceName() const noexcept
    {
        return instance_name_;
    }
    [[nodiscard]] const std::shared_ptr<AuxiliaryStateModel>& model() const noexcept
    {
        return model_;
    }
    [[nodiscard]] AuxiliaryStateScope getScope() const noexcept { return scope_; }
    [[nodiscard]] AuxiliarySolveMode getSolveMode() const noexcept
    {
        return solve_mode_;
    }
    [[nodiscard]] AuxiliaryScheduleMode getSchedule() const noexcept
    {
        return schedule_;
    }
    [[nodiscard]] const AuxiliaryStepperSpec& getStepperSpec() const noexcept
    {
        return stepper_spec_;
    }
    [[nodiscard]] const AuxiliaryDeploymentRegion& getRegion() const noexcept
    {
        return region_;
    }
    [[nodiscard]] const std::unordered_map<std::string, std::string>&
    inputBindings() const noexcept { return input_bindings_; }
    [[nodiscard]] const std::unordered_map<std::string, AuxiliaryInputHandle>&
    coupledBindings() const noexcept { return coupled_bindings_; }
    [[nodiscard]] const std::unordered_map<std::string, Real>&
    paramValues() const noexcept { return param_values_; }
    [[nodiscard]] const std::vector<Real>& initialValues() const noexcept
    {
        return initial_values_;
    }
    [[nodiscard]] std::size_t getEntityCount() const noexcept
    {
        return entity_count_;
    }
    [[nodiscard]] bool hasExplicitName() const noexcept
    {
        return has_explicit_name_;
    }
    [[nodiscard]] std::span<const std::size_t> qpOffsets() const noexcept
    {
        return qp_offsets_;
    }
    void setResolvedInstanceName(std::string instance_name)
    {
        instance_name_ = std::move(instance_name);
        has_explicit_name_ = true;
    }
    [[nodiscard]] const AuxiliaryDerivativePolicy& getDerivativePolicy() const noexcept
    {
        return derivative_policy_;
    }
    [[nodiscard]] bool hasExplicitDerivativePolicy() const noexcept
    {
        return has_explicit_derivative_policy_;
    }

    [[nodiscard]] AuxiliaryLayoutMode getLayoutMode() const noexcept
    {
        return layout_mode_;
    }
    [[nodiscard]] AuxiliaryEntityOrdering getEntityOrdering() const noexcept
    {
        return entity_ordering_;
    }

    // -----------------------------------------------------------------
    //  Validation
    // -----------------------------------------------------------------

    /**
     * @brief Validate the deployment configuration.
     *
     * Checks:
     * - Instance name is non-empty.
     * - Stepper is not set for Monolithic instances.
     * - All required model inputs have bindings.
     * - All parameters have values.
     * - Initial values size matches model dimension.
     *
     * @return Empty string if valid; diagnostic message if not.
     */
    [[nodiscard]] std::string validate() const;

private:
    std::shared_ptr<AuxiliaryStateModel> model_{};
    std::string instance_name_{};
    bool has_explicit_name_{false};
    AuxiliaryStateScope scope_{AuxiliaryStateScope::Global};
    AuxiliaryDeploymentRegion region_{};
    AuxiliarySolveMode solve_mode_{AuxiliarySolveMode::Partitioned};
    AuxiliaryScheduleMode schedule_{AuxiliaryScheduleMode::SingleRate};
    AuxiliaryStepperSpec stepper_spec_{};
    std::unordered_map<std::string, std::string> input_bindings_{};
    /// FE-backed handle bindings keyed by model input name.
    std::unordered_map<std::string, AuxiliaryInputHandle> coupled_bindings_{};
    std::unordered_map<std::string, Real> param_values_{};
    std::vector<Real> initial_values_{};
    std::size_t entity_count_{0}; ///< 0 = auto-detect from scope/mesh
    std::vector<std::size_t> qp_offsets_{};
    AuxiliaryDerivativePolicy derivative_policy_{};
    bool has_explicit_derivative_policy_{false};
    AuxiliaryLayoutMode layout_mode_{AuxiliaryLayoutMode::FixedStride};
    AuxiliaryEntityOrdering entity_ordering_{AuxiliaryEntityOrdering::ByEntityThenComponent};
};

/**
 * @brief Create a deployment handle for an auxiliary model.
 *
 * This is the canonical entry point for deploying auxiliary models.
 *
 * @param model  The model to deploy (built or custom).
 * @return A configurable deployment instance.
 */
[[nodiscard]] inline AuxiliaryDeployedInstance use(
    std::shared_ptr<AuxiliaryStateModel> model)
{
    return AuxiliaryDeployedInstance(std::move(model));
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_BINDINGS_H
