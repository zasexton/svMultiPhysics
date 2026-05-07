#ifndef SVMP_FE_FORMS_NITSCHE_BC_H
#define SVMP_FE_FORMS_NITSCHE_BC_H

/**
 * @file NitscheBC.h
 * @brief Weak Dirichlet boundary conditions (Nitsche) as BoundaryCondition objects
 */

#include "Forms/BoundaryCondition.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace bc {

/**
 * @brief Legacy scalar Nitsche Dirichlet BC for diffusion-type operators
 *
 * New physics modules should prefer `TraceNitscheBC`, which receives explicit
 * consistency flux, adjoint flux, and penalty-weight expressions from the
 * formulation.
 *
 * Imposes u = g weakly on boundary marker Γ(m) via Nitsche's method:
 *  1) Consistency: -∫ k (∇u·n) v ds
 *  2) Adjoint:     ∓∫ k (∇v·n) (u-g) ds  (symmetric: -, unsymmetric: +)
 *  3) Penalty:     +∫ (γ k p^2 / h) (u-g) v ds
 *
 * where p is the trial polynomial order when available, and h is the facet-normal
 * element size h_n = 2|K|/|F| (cell volume divided by facet area).
 */
class ScalarNitscheBC final : public BoundaryCondition {
public:
    ScalarNitscheBC(int boundary_marker,
                    FormExpr value,
                    FormExpr diffusion_coeff,
                    Real penalty_gamma,
                    bool symmetric,
                    bool scale_with_p = true)
        : boundary_marker_(boundary_marker)
        , value_(std::move(value))
        , diffusion_coeff_(std::move(diffusion_coeff))
        , penalty_gamma_(penalty_gamma)
        , symmetric_(symmetric)
        , scale_with_p_(scale_with_p)
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("ScalarNitscheBC: boundary_marker must be >= 0");
        }
        if (!value_.isValid()) {
            throw std::invalid_argument("ScalarNitscheBC: invalid value expression");
        }
        if (!diffusion_coeff_.isValid()) {
            throw std::invalid_argument("ScalarNitscheBC: invalid diffusion coefficient expression");
        }
        if (!(penalty_gamma_ > Real(0.0))) {
            throw std::invalid_argument("ScalarNitscheBC: penalty_gamma must be > 0");
        }
        if (value_.hasTest() || value_.hasTrial()) {
            throw std::invalid_argument("ScalarNitscheBC: value must not contain test/trial functions");
        }
        if (diffusion_coeff_.hasTest() || diffusion_coeff_.hasTrial()) {
            throw std::invalid_argument("ScalarNitscheBC: diffusion_coeff must not contain test/trial functions");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        const auto n = FormExpr::normal();
        const auto h = (2.0 * FormExpr::cellVolume()) / FormExpr::facetArea();
        TraceNitscheOptions opts;
        opts.gamma = penalty_gamma_;
        opts.variant = symmetric_ ? NitscheVariant::Symmetric
                                  : NitscheVariant::Unsymmetric;
        opts.scale_with_p = scale_with_p_;

        residual = applyTraceNitsche(std::move(residual),
                                     u,
                                     v,
                                     boundary_marker_,
                                     value_,
                                     diffusion_coeff_ * inner(grad(u), n),
                                     diffusion_coeff_ * inner(grad(v), n),
                                     diffusion_coeff_ / h,
                                     ScalarTraceOperator::Identity,
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
        d.trace_kind = analysis::TraceKind::Value;
        d.enforcement_kind = analysis::EnforcementKind::WeakNitsche;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = "ScalarNitscheBC on marker " + std::to_string(boundary_marker_);
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
    int boundary_marker_{-1};
    FormExpr value_{};
    FormExpr diffusion_coeff_{};
    Real penalty_gamma_{10.0};
    bool symmetric_{true};
    bool scale_with_p_{true};
};

/**
 * @brief Generic scalar-trace Nitsche BC on a boundary marker
 *
 * Imposes tau(u) = g weakly using user-supplied consistency and adjoint flux
 * expressions plus a penalty weight. FE stays physics-agnostic by not assuming
 * a specific PDE flux law.
 */
class TraceNitscheBC final : public BoundaryCondition {
public:
    TraceNitscheBC(int boundary_marker,
                   FormExpr value,
                   FormExpr consistency_flux,
                   FormExpr adjoint_flux,
                   FormExpr penalty_weight,
                   ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent,
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
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("TraceNitscheBC: boundary_marker must be >= 0");
        }
        if (!value_.isValid()) {
            throw std::invalid_argument("TraceNitscheBC: invalid value expression");
        }
        if (!consistency_flux_.isValid()) {
            throw std::invalid_argument("TraceNitscheBC: invalid consistency flux expression");
        }
        if (!adjoint_flux_.isValid()) {
            throw std::invalid_argument("TraceNitscheBC: invalid adjoint flux expression");
        }
        if (!penalty_weight_.isValid()) {
            throw std::invalid_argument("TraceNitscheBC: invalid penalty weight expression");
        }
        if (!(penalty_gamma_ > Real(0.0))) {
            throw std::invalid_argument("TraceNitscheBC: penalty_gamma must be > 0");
        }
        if (value_.hasTest() || value_.hasTrial()) {
            throw std::invalid_argument("TraceNitscheBC: value must not contain test/trial functions");
        }
        if (penalty_weight_.hasTest() || penalty_weight_.hasTrial()) {
            throw std::invalid_argument("TraceNitscheBC: penalty weight must not contain test/trial functions");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        TraceNitscheOptions opts;
        opts.gamma = penalty_gamma_;
        opts.variant = symmetric_ ? NitscheVariant::Symmetric
                                  : NitscheVariant::Unsymmetric;
        opts.scale_with_p = scale_with_p_;

        residual = applyTraceNitsche(std::move(residual),
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
        nitsche.penalty_scaling_verified = scale_with_p_;
        d.nitsche = nitsche;
        return {d};
    }

private:
    int boundary_marker_{-1};
    FormExpr value_{};
    FormExpr consistency_flux_{};
    FormExpr adjoint_flux_{};
    FormExpr penalty_weight_{};
    ScalarTraceOperator trace_operator_{ScalarTraceOperator::NormalComponent};
    Real penalty_gamma_{10.0};
    bool symmetric_{true};
    bool scale_with_p_{true};
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
    [[nodiscard]] int interfaceMarker() const override { return interface_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        TraceNitscheOptions opts;
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
