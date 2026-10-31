/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_FORMS_LAGRANGEDIRICHLETBC_H
#define SVMP_FE_FORMS_LAGRANGEDIRICHLETBC_H

/**
 * @file LagrangeDirichletBC.h
 * @brief Dirichlet condition u = g imposed weakly through a Lagrange multiplier
 *
 * The multiplier lambda is a volume field restricted to the boundary (D54
 * T1): the condition registers SupportRestrictionConstraint(lambda, the
 * boundary), and adds to the residual of u
 *
 *     (lambda * v).ds(b) + (mu * (u - g)).ds(b)
 *
 * with mu the test function of lambda; lambda is then -k du/dn for the
 * operator (k grad u, grad v).  With beta > 0 it also adds the Barbosa-Hughes
 * stabilization
 *
 *     - (beta * h * (lambda + k grad(u).n) * (mu + k grad(v).n)).ds(b)
 *
 * (h the cell diameter, k the diffusivity given to the condition), which
 * makes equal-order pairs (P1 u, P1 lambda) stable.  The residual must be
 * installed with lambda in the field list:
 *
 * @code
 *   bcs.add(bc::makeLagrangeDirichletBC(1, g, lam_id));
 *   bcs.applyAll(system, residual, u, v, u_id);
 *   installFormulation(system, "op", {u_id, lam_id}, residual);
 * @endcode
 *
 * The condition is exclusive on its marker: the BoundaryConditionManager
 * refuses it together with an EssentialBC (or any other condition) on the
 * same boundary of the field.  The multiplier's rows anchor the constant of u
 * (setup reads them), so a problem whose only Dirichlet condition is this one
 * gets no automatic gauge.  Scalar fields only.
 */

#include "Constraints/SupportRestrictionConstraint.h"
#include "Forms/BoundaryCondition.h"
#include "Forms/FormExpr.h"
#include "Forms/Vocabulary.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace bc {

class LagrangeDirichletBC final : public BoundaryCondition {
public:
    /**
     * @param boundary_marker the physical boundary b
     * @param value           g (a FormExpr in x, t)
     * @param multiplier      the multiplier field (a scalar H1 volume field)
     * @param beta            Barbosa-Hughes stabilization (0: none)
     * @param diffusivity     k of the stabilization's flux k grad(u).n
     */
    LagrangeDirichletBC(int boundary_marker,
                        FormExpr value,
                        FieldId multiplier,
                        Real beta = Real(0.0),
                        FormExpr diffusivity = FormExpr::constant(Real(1.0)))
        : boundary_marker_(boundary_marker)
        , value_(std::move(value))
        , multiplier_(multiplier)
        , beta_(beta)
        , diffusivity_(std::move(diffusivity))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("LagrangeDirichletBC: the boundary marker must be >= 0");
        }
        if (!value_.isValid()) {
            throw std::invalid_argument("LagrangeDirichletBC: invalid value expression");
        }
        if (multiplier_ == INVALID_FIELD_ID) {
            throw std::invalid_argument("LagrangeDirichletBC: invalid multiplier field");
        }
        if (!(beta_ >= Real(0.0))) {
            throw std::invalid_argument("LagrangeDirichletBC: the stabilization beta must be >= 0");
        }
        if (!diffusivity_.isValid()) {
            throw std::invalid_argument("LagrangeDirichletBC: invalid diffusivity expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }
    [[nodiscard]] std::string_view label() const override { return "LagrangeDirichletBC"; }
    [[nodiscard]] FieldId multiplier() const noexcept { return multiplier_; }

    /// Binds the multiplier's symbols; the multiplier must be a scalar volume
    /// field other than the field the condition is applied to.
    void setup(systems::FESystem& system, FieldId field_id) override
    {
        const std::string context = "LagrangeDirichletBC on boundary_marker " + std::to_string(boundary_marker_);
        if (multiplier_ == field_id) {
            throw std::invalid_argument(context + ": the multiplier must be a field of its own");
        }
        const auto& rec = system.fieldRecord(multiplier_);
        if (!rec.space || rec.scope != systems::FieldScope::VolumeCell || rec.components != 1) {
            throw std::invalid_argument(context + ": the multiplier '" + rec.name +
                                        "' must be a scalar volume field (restricted to the boundary)");
        }
        multiplier_space_ = rec.space;
        multiplier_name_ = rec.name;
    }

    /// Registers the multiplier's support restriction to the boundary.
    void installSystemConstraints(systems::FESystem& system, FieldId /*field_id*/) const override
    {
        system.addSystemConstraint(std::make_unique<constraints::SupportRestrictionConstraint>(
            multiplier_, ExteriorBoundaryMeasure::fullPhysical(boundary_marker_)));
    }

    void contributeToResidual(FormExpr& residual, const FormExpr& u, const FormExpr& v) const override
    {
        if (!multiplier_space_) {
            throw std::logic_error("LagrangeDirichletBC: setup() binds the multiplier before contributeToResidual()");
        }
        const auto lam = StateField(multiplier_, *multiplier_space_, multiplier_name_);
        const auto mu = TestField(multiplier_, *multiplier_space_, "test_" + multiplier_name_);
        // One integral per test function (the installer splits the residual
        // by its integrals).
        auto terms = (lam * v).ds(boundary_marker_) + (mu * (u - value_)).ds(boundary_marker_);
        if (beta_ > Real(0.0)) {
            const auto n = FormExpr::normal();
            const auto h = FormExpr::cellDiameter();
            const auto weighted = FormExpr::constant(beta_) * h * (lam + diffusivity_ * inner(grad(u), n));
            terms = terms - (weighted * mu).ds(boundary_marker_) -
                    (weighted * diffusivity_ * inner(grad(v), n)).ds(boundary_marker_);
        }
        residual = residual.isValid() ? residual + terms : terms;
    }

    [[nodiscard]] bool hasWeakTerms() const override { return true; }

    /// Exclusive on its boundary: the condition fixes u there.
    [[nodiscard]] bool allowsMarkerSharing() const override { return false; }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

private:
    int boundary_marker_{-1};
    FormExpr value_;
    FieldId multiplier_{INVALID_FIELD_ID};
    Real beta_{0.0};
    FormExpr diffusivity_;
    std::shared_ptr<const spaces::FunctionSpace> multiplier_space_{};
    std::string multiplier_name_{};
};

/// The Lagrange-multiplier Dirichlet condition u = g on boundary @p boundary_marker.
[[nodiscard]] inline std::unique_ptr<LagrangeDirichletBC> makeLagrangeDirichletBC(
    int boundary_marker,
    FormExpr value,
    FieldId multiplier,
    Real beta = Real(0.0),
    FormExpr diffusivity = FormExpr::constant(Real(1.0)))
{
    return std::make_unique<LagrangeDirichletBC>(boundary_marker, std::move(value), multiplier, beta,
                                                 std::move(diffusivity));
}

} // namespace bc
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_LAGRANGEDIRICHLETBC_H
