#ifndef SVMP_FE_FORMS_NITSCHE_BC_H
#define SVMP_FE_FORMS_NITSCHE_BC_H

/**
 * @file NitscheBC.h
 * @brief Weak Dirichlet boundary conditions (Nitsche) as BoundaryCondition objects
 */

#include "Forms/BoundaryCondition.h"

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace bc {

/**
 * @brief Generic scalar-trace Nitsche BC on a boundary marker
 *
 * Imposes tau(u) = g weakly using user-supplied consistency and adjoint flux
 * expressions plus a penalty weight. FE stays physics-agnostic by not assuming
 * a specific PDE flux law.
 */
class TraceNitscheBC final : public BoundaryCondition {
public:
    TraceNitscheBC(ExteriorBoundaryMeasure measure,
                   FormExpr value,
                   FormExpr consistency_flux,
                   FormExpr adjoint_flux,
                   FormExpr penalty_weight,
                   ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent,
                   Real penalty_gamma = 10.0,
                   bool symmetric = true,
                   bool scale_with_p = true)
        : boundary_marker_(
              measure.physicalBoundaryMarker())
        , value_(std::move(value))
        , consistency_flux_(std::move(consistency_flux))
        , adjoint_flux_(std::move(adjoint_flux))
        , penalty_weight_(std::move(penalty_weight))
        , trace_operator_(trace_operator)
        , penalty_gamma_(penalty_gamma)
        , symmetric_(symmetric)
        , scale_with_p_(scale_with_p)
        , exterior_boundary_measure_(measure)
    {
        validate();
    }

    TraceNitscheBC(int boundary_marker,
                   FormExpr value,
                   FormExpr consistency_flux,
                   FormExpr adjoint_flux,
                   FormExpr penalty_weight,
                   ScalarTraceOperator trace_operator =
                       ScalarTraceOperator::NormalComponent,
                   Real penalty_gamma = 10.0,
                   bool symmetric = true,
                   bool scale_with_p = true)
        : boundary_marker_(boundary_marker)
        , value_(std::move(value))
        , consistency_flux_(std::move(consistency_flux))
        , adjoint_flux_(std::move(adjoint_flux))
        , penalty_weight_(std::move(penalty_weight))
        , trace_operator_(trace_operator)
        , penalty_gamma_(penalty_gamma)
        , symmetric_(symmetric)
        , scale_with_p_(scale_with_p)
    {
        validate();
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }
    [[nodiscard]] std::string_view label() const override { return "TraceNitscheBC"; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        Nitsche opts;
        opts.gamma = penalty_gamma_;
        opts.variant = symmetric_ ? NitscheVariant::Symmetric
                                  : NitscheVariant::Unsymmetric;
        opts.scale_with_p = scale_with_p_;

        residual =
            exterior_boundary_measure_.has_value()
                ? applyTraceNitsche(
                      std::move(residual),
                      u,
                      v,
                      *exterior_boundary_measure_,
                      value_,
                      consistency_flux_,
                      adjoint_flux_,
                      penalty_weight_,
                      trace_operator_,
                      opts)
                : applyTraceNitsche(
                      std::move(residual),
                      u,
                      v,
                      boundary_marker_,
                      value_,
                      consistency_flux_,
                      adjoint_flux_,
                      penalty_weight_,
                      trace_operator_,
                      opts);
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
        d.trace_kind = (trace_operator_ == ScalarTraceOperator::NormalComponent)
                           ? analysis::TraceKind::NormalComponent
                           : analysis::TraceKind::Value;
        d.enforcement_kind = analysis::EnforcementKind::WeakNitsche;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = "TraceNitscheBC on marker " + std::to_string(boundary_marker_);
        analysis::NitscheMetadata nitsche;
        nitsche.variant = symmetric_
            ? analysis::NitscheVariant::Symmetric
            : analysis::NitscheVariant::Nonsymmetric;
        nitsche.primal_consistency_terms_present = true;
        nitsche.adjoint_consistency_terms_present = symmetric_;
        nitsche.rhs_consistency_terms_present = true;
        nitsche.penalty_positive = penalty_gamma_ > Real(0.0);
        nitsche.penalty_scaling_verified =
            scale_with_p_ &&
            (!exterior_boundary_measure_.has_value() ||
             exterior_boundary_measure_->isFullPhysical());
        d.nitsche = nitsche;
        return {d};
    }

private:
    void validate() const
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument(
                "TraceNitscheBC: boundary_marker must be >= 0");
        }
        if (!value_.isValid()) {
            throw std::invalid_argument(
                "TraceNitscheBC: invalid value expression");
        }
        if (!consistency_flux_.isValid()) {
            throw std::invalid_argument(
                "TraceNitscheBC: invalid consistency flux expression");
        }
        if (!adjoint_flux_.isValid()) {
            throw std::invalid_argument(
                "TraceNitscheBC: invalid adjoint flux expression");
        }
        if (!penalty_weight_.isValid()) {
            throw std::invalid_argument(
                "TraceNitscheBC: invalid penalty weight expression");
        }
        if (!(penalty_gamma_ > Real(0.0))) {
            throw std::invalid_argument(
                "TraceNitscheBC: penalty_gamma must be > 0");
        }
        if (value_.hasTest() || value_.hasTrial()) {
            throw std::invalid_argument(
                "TraceNitscheBC: value must not contain test/trial functions");
        }
        if (penalty_weight_.hasTest() ||
            penalty_weight_.hasTrial()) {
            throw std::invalid_argument(
                "TraceNitscheBC: penalty weight must not contain test/trial functions");
        }
    }

    int boundary_marker_{-1};
    FormExpr value_{};
    FormExpr consistency_flux_{};
    FormExpr adjoint_flux_{};
    FormExpr penalty_weight_{};
    ScalarTraceOperator trace_operator_{ScalarTraceOperator::NormalComponent};
    Real penalty_gamma_{10.0};
    bool symmetric_{true};
    bool scale_with_p_{true};
    std::optional<ExteriorBoundaryMeasure>
        exterior_boundary_measure_{};
};

/**
 * @brief Generic scalar-trace Nitsche relation on an interface marker
 *
 * Imposes tau_red(u) = g weakly on `.dI(marker)`, where tau_red is a selected
 * side, jump, or average reduction of a scalar trace operator.
 */
class InterfaceTraceNitscheBC final : public BoundaryCondition {
public:
    InterfaceTraceNitscheBC(int interface_marker,
                            FormExpr value,
                            FormExpr consistency_flux,
                            FormExpr adjoint_flux,
                            FormExpr penalty_weight,
                            ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent,
                            InterfaceTraceReduction reduction = InterfaceTraceReduction::Jump,
                            Real penalty_gamma = 10.0,
                            bool symmetric = true,
                            bool scale_with_p = true)
        : interface_marker_(interface_marker)
        , value_(std::move(value))
        , consistency_flux_(std::move(consistency_flux))
        , adjoint_flux_(std::move(adjoint_flux))
        , penalty_weight_(std::move(penalty_weight))
        , trace_operator_(trace_operator)
        , reduction_(reduction)
        , penalty_gamma_(penalty_gamma)
        , symmetric_(symmetric)
        , scale_with_p_(scale_with_p)
    {
        if (interface_marker_ < 0) {
            throw std::invalid_argument("InterfaceTraceNitscheBC: interface_marker must be >= 0");
        }
        if (!value_.isValid()) {
            throw std::invalid_argument("InterfaceTraceNitscheBC: invalid value expression");
        }
        if (!consistency_flux_.isValid()) {
            throw std::invalid_argument("InterfaceTraceNitscheBC: invalid consistency flux expression");
        }
        if (!adjoint_flux_.isValid()) {
            throw std::invalid_argument("InterfaceTraceNitscheBC: invalid adjoint flux expression");
        }
        if (!penalty_weight_.isValid()) {
            throw std::invalid_argument("InterfaceTraceNitscheBC: invalid penalty weight expression");
        }
        if (!(penalty_gamma_ > Real(0.0))) {
            throw std::invalid_argument("InterfaceTraceNitscheBC: penalty_gamma must be > 0");
        }
        if (value_.hasTest() || value_.hasTrial()) {
            throw std::invalid_argument("InterfaceTraceNitscheBC: value must not contain test/trial functions");
        }
        if (penalty_weight_.hasTest() || penalty_weight_.hasTrial()) {
            throw std::invalid_argument("InterfaceTraceNitscheBC: penalty weight must not contain test/trial functions");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return -1; }
    [[nodiscard]] std::string_view label() const override { return "InterfaceTraceNitscheBC"; }
    [[nodiscard]] int interfaceMarker() const override { return interface_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        Nitsche opts;
        opts.gamma = penalty_gamma_;
        opts.variant = symmetric_ ? NitscheVariant::Symmetric
                                  : NitscheVariant::Unsymmetric;
        opts.scale_with_p = scale_with_p_;

        residual = applyInterfaceTraceNitsche(std::move(residual),
                                              u,
                                              v,
                                              interface_marker_,
                                              value_,
                                              consistency_flux_,
                                              adjoint_flux_,
                                              penalty_weight_,
                                              trace_operator_,
                                              reduction_,
                                              opts);
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
        d.trace_kind = (trace_operator_ == ScalarTraceOperator::NormalComponent)
                           ? analysis::TraceKind::NormalComponent
                           : analysis::TraceKind::Value;
        d.enforcement_kind = analysis::EnforcementKind::WeakNitsche;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = "InterfaceTraceNitscheBC on interface " + std::to_string(interface_marker_);
        analysis::NitscheMetadata nitsche;
        nitsche.variant = symmetric_
            ? analysis::NitscheVariant::Symmetric
            : analysis::NitscheVariant::Nonsymmetric;
        nitsche.primal_consistency_terms_present = true;
        nitsche.adjoint_consistency_terms_present = symmetric_;
        nitsche.rhs_consistency_terms_present = true;
        nitsche.penalty_positive = penalty_gamma_ > Real(0.0);
        nitsche.penalty_scaling_verified = scale_with_p_;
        d.nitsche = nitsche;
        return {d};
    }

private:
    int interface_marker_{-1};
    FormExpr value_{};
    FormExpr consistency_flux_{};
    FormExpr adjoint_flux_{};
    FormExpr penalty_weight_{};
    ScalarTraceOperator trace_operator_{ScalarTraceOperator::NormalComponent};
    InterfaceTraceReduction reduction_{InterfaceTraceReduction::Jump};
    Real penalty_gamma_{10.0};
    bool symmetric_{true};
    bool scale_with_p_{true};
};

} // namespace bc
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_NITSCHE_BC_H
