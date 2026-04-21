#ifndef SVMP_FE_FORMS_STANDARD_BCS_H
#define SVMP_FE_FORMS_STANDARD_BCS_H

/**
 * @file StandardBCs.h
 * @brief Standard physics-agnostic boundary condition implementations
 */

#include "Constraints/HDivNormalConstraint.h"
#include "Forms/BoundaryCondition.h"
#include "Systems/FESystem.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace bc {

namespace detail {

[[nodiscard]] inline analysis::TraceKind traceKindFor(ScalarTraceOperator op)
{
    return op == ScalarTraceOperator::NormalComponent
               ? analysis::TraceKind::NormalComponent
               : analysis::TraceKind::Value;
}

[[nodiscard]] inline analysis::InequalitySense inequalitySenseFor(TraceInequalitySense sense)
{
    switch (sense) {
    case TraceInequalitySense::LessEqual:
        return analysis::InequalitySense::LessEqual;
    case TraceInequalitySense::GreaterEqual:
        return analysis::InequalitySense::GreaterEqual;
    }

    return analysis::InequalitySense::None;
}

[[nodiscard]] inline std::string sourceFor(std::string_view name,
                                           analysis::DomainKind domain,
                                           int marker)
{
    const std::string marker_name =
        (domain == analysis::DomainKind::InterfaceFace) ? "interface " : "marker ";
    return std::string(name) + " on " + marker_name + std::to_string(marker);
}

} // namespace detail

/**
 * @brief Marker-only boundary condition used for conflict validation
 *
 * This BC reserves a boundary marker in BoundaryConditionManager::validate()
 * without contributing weak-form terms or strong constraints.
 */
class ReservedBC final : public BoundaryCondition {
public:
    explicit ReservedBC(int boundary_marker)
        : boundary_marker_(boundary_marker)
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("ReservedBC: boundary_marker must be >= 0");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }
    [[nodiscard]] bool hasWeakTerms() const override { return false; }

    void contributeToResidual(FormExpr& /*residual*/,
                              const FormExpr& /*u*/,
                              const FormExpr& /*v*/) const override
    {
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId /*field_id*/, const systems::FESystem* /*system*/) const override
    {
        return {};  // Reserved BC — no mathematical constraint
    }

private:
    int boundary_marker_{-1};
};

/**
 * @brief Natural (Neumann/traction) BC: adds -∫ flux · v ds(marker)
 */
class NaturalBC : public BoundaryCondition {
public:
    NaturalBC(int boundary_marker, FormExpr flux)
        : boundary_marker_(boundary_marker)
        , flux_(std::move(flux))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("NaturalBC: boundary_marker must be >= 0");
        }
        if (!flux_.isValid()) {
            throw std::invalid_argument("NaturalBC: invalid flux expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& /*u*/,
                              const FormExpr& v) const override
    {
        residual = residual - inner(flux_, v).ds(boundary_marker_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = analysis::TraceKind::Flux;
        d.enforcement_kind = analysis::EnforcementKind::WeakConsistent;
        d.anchors_constant_mode = false;
        d.anchors_rigid_body_translation = false;
        d.anchors_rigid_body_rotation = false;
        d.source = "NaturalBC on marker " + std::to_string(boundary_marker_);
        return {d};
    }

private:
protected:
    int boundary_marker_{-1};
    FormExpr flux_{};
};

/**
 * @brief Robin BC: adds ∫ alpha * (u · v) ds(marker) - ∫ rhs · v ds(marker)
 */
class RobinBC : public BoundaryCondition {
public:
    RobinBC(int boundary_marker, FormExpr alpha, FormExpr rhs)
        : boundary_marker_(boundary_marker)
        , alpha_(std::move(alpha))
        , rhs_(std::move(rhs))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("RobinBC: boundary_marker must be >= 0");
        }
        if (!alpha_.isValid()) {
            throw std::invalid_argument("RobinBC: invalid alpha expression");
        }
        if (!rhs_.isValid()) {
            throw std::invalid_argument("RobinBC: invalid rhs expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        residual = residual + (alpha_ * inner(u, v)).ds(boundary_marker_) - inner(rhs_, v).ds(boundary_marker_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = analysis::TraceKind::Mixed;
        d.enforcement_kind = analysis::EnforcementKind::WeakPenalty;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = "RobinBC on marker " + std::to_string(boundary_marker_);
        return {d};
    }

private:
protected:
    int boundary_marker_{-1};
    FormExpr alpha_{};
    FormExpr rhs_{};
};

/**
 * @brief Essential (Dirichlet) BC: declares u = g on ds(marker)
 *
 * For vector-valued fields, this returns one StrongDirichlet per component.
 */
class EssentialBC final : public BoundaryCondition {
public:
    EssentialBC(int boundary_marker, FormExpr value, std::string symbol = "u")
        : boundary_marker_(boundary_marker)
        , value_components_{std::move(value)}
        , symbol_(std::move(symbol))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("EssentialBC: boundary_marker must be >= 0");
        }
        if (!value_components_.front().isValid()) {
            throw std::invalid_argument("EssentialBC: invalid value expression");
        }
    }

    EssentialBC(int boundary_marker, std::vector<FormExpr> value_components, std::string symbol = "u")
        : boundary_marker_(boundary_marker)
        , value_components_(std::move(value_components))
        , symbol_(std::move(symbol))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("EssentialBC: boundary_marker must be >= 0");
        }
        if (value_components_.empty()) {
            throw std::invalid_argument("EssentialBC: empty value component list");
        }
        for (const auto& c : value_components_) {
            if (!c.isValid()) {
                throw std::invalid_argument("EssentialBC: invalid value component expression");
            }
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }
    [[nodiscard]] bool hasWeakTerms() const override { return false; }

    void contributeToResidual(FormExpr& /*residual*/,
                              const FormExpr& /*u*/,
                              const FormExpr& /*v*/) const override
    {
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId field_id) const override
    {
        std::vector<StrongDirichlet> out;
        out.reserve(value_components_.size());

        if (value_components_.size() == 1u) {
            out.push_back(strongDirichlet(field_id, boundary_marker_, value_components_.front(), symbol_));
            return out;
        }

        for (std::size_t comp = 0; comp < value_components_.size(); ++comp) {
            out.push_back(strongDirichlet(field_id,
                                          boundary_marker_,
                                          value_components_[comp],
                                          symbol_,
                                          static_cast<int>(comp)));
        }
        return out;
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        std::vector<analysis::BoundaryConditionDescriptor> descs;

        if (value_components_.size() <= 1u) {
            // Scalar field or single-component vector: field-wide descriptor
            analysis::BoundaryConditionDescriptor d;
            d.primary_variable = analysis::VariableKey::field(field_id);
            d.component = -1;
            d.boundary_marker = boundary_marker_;
            d.trace_kind = analysis::TraceKind::Value;
            d.enforcement_kind = analysis::EnforcementKind::Strong;
            d.anchors_constant_mode = true;
            d.anchors_rigid_body_translation = true;
            d.anchors_rigid_body_rotation = false;
            d.source = "EssentialBC on marker " + std::to_string(boundary_marker_);
            descs.push_back(std::move(d));
        } else {
            // Multi-component: emit one descriptor per component so that
            // partially constrained vector fields are correctly analyzed.
            for (std::size_t comp = 0; comp < value_components_.size(); ++comp) {
                analysis::BoundaryConditionDescriptor d;
                d.primary_variable = analysis::VariableKey::field(field_id, static_cast<int>(comp));
                d.component = static_cast<int>(comp);
                d.boundary_marker = boundary_marker_;
                d.trace_kind = analysis::TraceKind::Value;
                d.enforcement_kind = analysis::EnforcementKind::Strong;
                d.anchors_constant_mode = true;
                d.anchors_rigid_body_translation = true;
                d.anchors_rigid_body_rotation = false;
                d.source = "EssentialBC comp " + std::to_string(comp) +
                           " on marker " + std::to_string(boundary_marker_);
                descs.push_back(std::move(d));
            }
        }
        return descs;
    }

private:
    int boundary_marker_{-1};
    std::vector<FormExpr> value_components_{};
    std::string symbol_{"u"};
};

/**
 * @brief Essential BC on the scalar normal trace of an H(div) field: u·n = g
 */
class NormalTraceEssentialBC final : public BoundaryCondition {
public:
    explicit NormalTraceEssentialBC(int boundary_marker, FormExpr value)
        : boundary_marker_(boundary_marker)
        , value_(std::move(value))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("NormalTraceEssentialBC: boundary_marker must be >= 0");
        }
        if (!value_.isValid()) {
            throw std::invalid_argument("NormalTraceEssentialBC: invalid value expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }
    [[nodiscard]] bool hasWeakTerms() const override { return false; }

    void contributeToResidual(FormExpr& /*residual*/,
                              const FormExpr& /*u*/,
                              const FormExpr& /*v*/) const override
    {
    }

    void installSystemConstraints(systems::FESystem& system, FieldId field_id) const override
    {
        system.addSystemConstraint(
            std::make_unique<constraints::HDivNormalConstraint>(field_id, boundary_marker_, value_));
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = analysis::TraceKind::NormalComponent;
        d.enforcement_kind = analysis::EnforcementKind::Strong;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = "NormalTraceEssentialBC on marker " + std::to_string(boundary_marker_);
        return {d};
    }

private:
    int boundary_marker_{-1};
    FormExpr value_{};
};

/**
 * @brief Weak trace load BC: adds -∫ g * tau(v) ds(marker)
 *
 * The default trace operator is the H(div) normal component tau(w) = w·n.
 */
class TraceLoadBC final : public BoundaryCondition {
public:
    TraceLoadBC(int boundary_marker,
                FormExpr value,
                ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent)
        : boundary_marker_(boundary_marker)
        , value_(std::move(value))
        , trace_operator_(trace_operator)
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("TraceLoadBC: boundary_marker must be >= 0");
        }
        if (!value_.isValid()) {
            throw std::invalid_argument("TraceLoadBC: invalid value expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& /*u*/,
                              const FormExpr& v) const override
    {
        residual = residual - (value_ * applyScalarTrace(v, trace_operator_)).ds(boundary_marker_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = detail::traceKindFor(trace_operator_);
        d.enforcement_kind = analysis::EnforcementKind::WeakConsistent;
        d.anchors_constant_mode = false;
        d.anchors_rigid_body_translation = false;
        d.anchors_rigid_body_rotation = false;
        d.source = "TraceLoadBC on marker " + std::to_string(boundary_marker_);
        return {d};
    }

private:
    int boundary_marker_{-1};
    FormExpr value_{};
    ScalarTraceOperator trace_operator_{ScalarTraceOperator::NormalComponent};
};

/**
 * @brief Weak Robin relation on a scalar trace:
 *        ∫ alpha * tau(u) * tau(v) ds - ∫ rhs * tau(v) ds
 *
 * The default trace operator is the H(div) normal component tau(w) = w·n.
 */
class TraceRobinBC final : public BoundaryCondition {
public:
    TraceRobinBC(int boundary_marker,
                 FormExpr alpha,
                 FormExpr rhs,
                 ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent)
        : boundary_marker_(boundary_marker)
        , alpha_(std::move(alpha))
        , rhs_(std::move(rhs))
        , trace_operator_(trace_operator)
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("TraceRobinBC: boundary_marker must be >= 0");
        }
        if (!alpha_.isValid()) {
            throw std::invalid_argument("TraceRobinBC: invalid alpha expression");
        }
        if (!rhs_.isValid()) {
            throw std::invalid_argument("TraceRobinBC: invalid rhs expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        const auto tau_u = applyScalarTrace(u, trace_operator_);
        const auto tau_v = applyScalarTrace(v, trace_operator_);
        residual = residual + (alpha_ * tau_u * tau_v).ds(boundary_marker_)
                            - (rhs_ * tau_v).ds(boundary_marker_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = detail::traceKindFor(trace_operator_);
        d.enforcement_kind = analysis::EnforcementKind::WeakPenalty;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = "TraceRobinBC on marker " + std::to_string(boundary_marker_);
        return {d};
    }

private:
    int boundary_marker_{-1};
    FormExpr alpha_{};
    FormExpr rhs_{};
    ScalarTraceOperator trace_operator_{ScalarTraceOperator::NormalComponent};
};

/**
 * @brief Weak one-sided scalar-trace law:
 *        ∫ sign * penalty * positive_part(signed_gap) * tau(v) ds
 *
 * This is the first FE-side nonlinear inequality contract for scalar traces.
 * In semismooth mode it uses `max(0, signed_gap)`; in smooth mode it uses a
 * regularized `smoothMax`.
 */
class TraceInequalityBC final : public BoundaryCondition {
public:
    TraceInequalityBC(int boundary_marker,
                      FormExpr bound,
                      FormExpr penalty,
                      TraceInequalityOptions options = {})
        : boundary_marker_(boundary_marker)
        , bound_(std::move(bound))
        , penalty_(std::move(penalty))
        , options_(std::move(options))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("TraceInequalityBC: boundary_marker must be >= 0");
        }
        if (!bound_.isValid()) {
            throw std::invalid_argument("TraceInequalityBC: invalid bound expression");
        }
        if (!penalty_.isValid()) {
            throw std::invalid_argument("TraceInequalityBC: invalid penalty expression");
        }
        if (bound_.hasTest() || bound_.hasTrial()) {
            throw std::invalid_argument("TraceInequalityBC: bound must not contain test/trial functions");
        }
        if (penalty_.hasTest() || penalty_.hasTrial()) {
            throw std::invalid_argument("TraceInequalityBC: penalty must not contain test/trial functions");
        }
        if (options_.linearization == TraceInequalityLinearization::Smooth) {
            if (!options_.smoothing_epsilon.isValid()) {
                throw std::invalid_argument(
                    "TraceInequalityBC: smooth linearization requires a valid smoothing epsilon");
            }
            if (options_.smoothing_epsilon.hasTest() || options_.smoothing_epsilon.hasTrial()) {
                throw std::invalid_argument(
                    "TraceInequalityBC: smoothing epsilon must not contain test/trial functions");
            }
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }
    [[nodiscard]] bool allowsMarkerSharing() const override { return false; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        residual = applyTraceInequality(
            std::move(residual), u, v, boundary_marker_, bound_, penalty_, options_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = detail::traceKindFor(options_.trace_operator);
        d.enforcement_kind = analysis::EnforcementKind::WeakInequality;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.inequality_sense = detail::inequalitySenseFor(options_.sense);
        d.state_dependent_activation = true;
        d.source = "TraceInequalityBC on marker " + std::to_string(boundary_marker_);
        return {d};
    }

private:
    int boundary_marker_{-1};
    FormExpr bound_{};
    FormExpr penalty_{};
    TraceInequalityOptions options_{};
};

/**
 * @brief Weak trace load condition on an interface marker.
 *
 * Adds:
 *   -∫ g * tau_red(v) dI(marker)
 * where tau_red is a scalar trace on one side, or a jump/average reduction.
 */
class InterfaceTraceLoadBC final : public BoundaryCondition {
public:
    InterfaceTraceLoadBC(int interface_marker,
                         FormExpr value,
                         ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent,
                         InterfaceTraceReduction reduction = InterfaceTraceReduction::Minus)
        : interface_marker_(interface_marker)
        , value_(std::move(value))
        , trace_operator_(trace_operator)
        , reduction_(reduction)
    {
        if (interface_marker_ < 0) {
            throw std::invalid_argument("InterfaceTraceLoadBC: interface_marker must be >= 0");
        }
        if (!value_.isValid()) {
            throw std::invalid_argument("InterfaceTraceLoadBC: invalid value expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return -1; }
    [[nodiscard]] int interfaceMarker() const override { return interface_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& /*u*/,
                              const FormExpr& v) const override
    {
        residual = residual - (value_ * applyInterfaceScalarTrace(v, trace_operator_, reduction_))
                                  .dI(interface_marker_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.domain = analysis::DomainKind::InterfaceFace;
        d.interface_marker = interface_marker_;
        d.trace_kind = detail::traceKindFor(trace_operator_);
        d.enforcement_kind = analysis::EnforcementKind::WeakConsistent;
        d.anchors_constant_mode = false;
        d.anchors_rigid_body_translation = false;
        d.anchors_rigid_body_rotation = false;
        d.source = detail::sourceFor("InterfaceTraceLoadBC",
                                     analysis::DomainKind::InterfaceFace,
                                     interface_marker_);
        return {d};
    }

private:
    int interface_marker_{-1};
    FormExpr value_{};
    ScalarTraceOperator trace_operator_{ScalarTraceOperator::NormalComponent};
    InterfaceTraceReduction reduction_{InterfaceTraceReduction::Minus};
};

/**
 * @brief Weak Robin relation on an interface scalar trace.
 *
 * Adds:
 *   ∫ alpha * tau_red(u) * tau_red(v) dI(marker) - ∫ rhs * tau_red(v) dI(marker)
 */
class InterfaceTraceRobinBC final : public BoundaryCondition {
public:
    InterfaceTraceRobinBC(int interface_marker,
                          FormExpr alpha,
                          FormExpr rhs,
                          ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent,
                          InterfaceTraceReduction reduction = InterfaceTraceReduction::Minus)
        : interface_marker_(interface_marker)
        , alpha_(std::move(alpha))
        , rhs_(std::move(rhs))
        , trace_operator_(trace_operator)
        , reduction_(reduction)
    {
        if (interface_marker_ < 0) {
            throw std::invalid_argument("InterfaceTraceRobinBC: interface_marker must be >= 0");
        }
        if (!alpha_.isValid()) {
            throw std::invalid_argument("InterfaceTraceRobinBC: invalid alpha expression");
        }
        if (!rhs_.isValid()) {
            throw std::invalid_argument("InterfaceTraceRobinBC: invalid rhs expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return -1; }
    [[nodiscard]] int interfaceMarker() const override { return interface_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        const auto tau_u = applyInterfaceScalarTrace(u, trace_operator_, reduction_);
        const auto tau_v = applyInterfaceScalarTrace(v, trace_operator_, reduction_);
        residual = residual + (alpha_ * tau_u * tau_v).dI(interface_marker_)
                            - (rhs_ * tau_v).dI(interface_marker_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.domain = analysis::DomainKind::InterfaceFace;
        d.interface_marker = interface_marker_;
        d.trace_kind = detail::traceKindFor(trace_operator_);
        d.enforcement_kind = analysis::EnforcementKind::WeakPenalty;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = detail::sourceFor("InterfaceTraceRobinBC",
                                     analysis::DomainKind::InterfaceFace,
                                     interface_marker_);
        return {d};
    }

private:
    int interface_marker_{-1};
    FormExpr alpha_{};
    FormExpr rhs_{};
    ScalarTraceOperator trace_operator_{ScalarTraceOperator::NormalComponent};
    InterfaceTraceReduction reduction_{InterfaceTraceReduction::Minus};
};

/**
 * @brief Weak continuity penalty on an interface scalar trace jump.
 *
 * Adds:
 *   ∫ alpha * [[tau(u)]] * [[tau(v)]] dI(marker) - ∫ rhs * [[tau(v)]] dI(marker)
 *
 * For `ScalarTraceOperator::NormalComponent`, the jump uses outward normals on
 * each side, i.e. `u^-·n^- + u^+·n^+`.
 */
class InterfaceTraceJumpPenaltyBC final : public BoundaryCondition {
public:
    InterfaceTraceJumpPenaltyBC(int interface_marker,
                                FormExpr alpha,
                                FormExpr rhs,
                                ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent)
        : interface_marker_(interface_marker)
        , alpha_(std::move(alpha))
        , rhs_(std::move(rhs))
        , trace_operator_(trace_operator)
    {
        if (interface_marker_ < 0) {
            throw std::invalid_argument("InterfaceTraceJumpPenaltyBC: interface_marker must be >= 0");
        }
        if (!alpha_.isValid()) {
            throw std::invalid_argument("InterfaceTraceJumpPenaltyBC: invalid alpha expression");
        }
        if (!rhs_.isValid()) {
            throw std::invalid_argument("InterfaceTraceJumpPenaltyBC: invalid rhs expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return -1; }
    [[nodiscard]] int interfaceMarker() const override { return interface_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        const auto jump_u =
            applyInterfaceScalarTrace(u, trace_operator_, InterfaceTraceReduction::Jump);
        const auto jump_v =
            applyInterfaceScalarTrace(v, trace_operator_, InterfaceTraceReduction::Jump);
        residual = residual + (alpha_ * jump_u * jump_v).dI(interface_marker_)
                            - (rhs_ * jump_v).dI(interface_marker_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.domain = analysis::DomainKind::InterfaceFace;
        d.interface_marker = interface_marker_;
        d.trace_kind = detail::traceKindFor(trace_operator_);
        d.enforcement_kind = analysis::EnforcementKind::WeakPenalty;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = detail::sourceFor("InterfaceTraceJumpPenaltyBC",
                                     analysis::DomainKind::InterfaceFace,
                                     interface_marker_);
        return {d};
    }

private:
    int interface_marker_{-1};
    FormExpr alpha_{};
    FormExpr rhs_{};
    ScalarTraceOperator trace_operator_{ScalarTraceOperator::NormalComponent};
};

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeNormalTraceEssentialBC(int boundary_marker, FormExpr value)
{
    return std::make_unique<NormalTraceEssentialBC>(boundary_marker, std::move(value));
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeTraceLoadBC(int boundary_marker,
                FormExpr value,
                ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent)
{
    return std::make_unique<TraceLoadBC>(
        boundary_marker, std::move(value), trace_operator);
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeTraceRobinBC(int boundary_marker,
                 FormExpr alpha,
                 FormExpr rhs,
                 ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent)
{
    return std::make_unique<TraceRobinBC>(
        boundary_marker, std::move(alpha), std::move(rhs), trace_operator);
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeTraceInequalityBC(int boundary_marker,
                      FormExpr bound,
                      FormExpr penalty,
                      TraceInequalityOptions options = {})
{
    return std::make_unique<TraceInequalityBC>(
        boundary_marker, std::move(bound), std::move(penalty), std::move(options));
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeInterfaceTraceLoadBC(int interface_marker,
                         FormExpr value,
                         ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent,
                         InterfaceTraceReduction reduction = InterfaceTraceReduction::Minus)
{
    return std::make_unique<InterfaceTraceLoadBC>(
        interface_marker, std::move(value), trace_operator, reduction);
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeInterfaceTraceRobinBC(int interface_marker,
                          FormExpr alpha,
                          FormExpr rhs,
                          ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent,
                          InterfaceTraceReduction reduction = InterfaceTraceReduction::Minus)
{
    return std::make_unique<InterfaceTraceRobinBC>(
        interface_marker, std::move(alpha), std::move(rhs), trace_operator, reduction);
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeInterfaceTraceJumpPenaltyBC(int interface_marker,
                                FormExpr alpha,
                                FormExpr rhs,
                                ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent)
{
    return std::make_unique<InterfaceTraceJumpPenaltyBC>(
        interface_marker, std::move(alpha), std::move(rhs), trace_operator);
}

} // namespace bc
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_STANDARD_BCS_H
