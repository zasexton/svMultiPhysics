/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/EnrichedSpace.h"

namespace svmp {
namespace FE {
namespace spaces {

EnrichedSpace::EnrichedSpace(std::shared_ptr<FunctionSpace> base_space,
                             std::shared_ptr<FunctionSpace> enrichment_space)
    : base_(std::move(base_space)),
      enrichment_(std::move(enrichment_space)) {
    FE_CHECK_NOT_NULL(base_.get(), "EnrichedSpace base_space");
    FE_CHECK_NOT_NULL(enrichment_.get(), "EnrichedSpace enrichment_space");
    FE_CHECK_ARG(base_->element_type() == enrichment_->element_type(),
                 "EnrichedSpace requires matching element types");
    FE_CHECK_ARG(base_->field_type() == enrichment_->field_type(),
                 "EnrichedSpace requires matching field types");
}

namespace {

template<typename Operation>
auto evaluate_sum(const FunctionSpace& base, const FunctionSpace& enrichment,
                  const FunctionSpace::Value& xi, const std::vector<Real>& coefficients,
                  Operation operation) {
    const auto n = base.dofs_per_element();
    FE_CHECK_ARG(coefficients.size() == n + enrichment.dofs_per_element(),
                 "EnrichedSpace: coefficient size mismatch");
    const auto split = coefficients.begin() + static_cast<std::ptrdiff_t>(n);
    return (base.*operation)(xi, std::vector<Real>(coefficients.begin(), split)) +
           (enrichment.*operation)(xi, std::vector<Real>(split, coefficients.end()));
}

} // namespace

FunctionSpace::Value EnrichedSpace::evaluate(const Value& xi,
                                             const std::vector<Real>& coefficients) const {
    return evaluate_sum(*base_, *enrichment_, xi, coefficients, &FunctionSpace::evaluate);
}

FunctionSpace::Jacobian EnrichedSpace::evaluate_jacobian(
    const Value& xi, const std::vector<Real>& coefficients) const {
    return evaluate_sum(*base_, *enrichment_, xi, coefficients, &FunctionSpace::evaluate_jacobian);
}

void EnrichedSpace::interpolate(const ValueFunction& function,
                                std::vector<Real>& coefficients) const {
    // Default strategy: project into base space only, zero enrichment
    std::vector<Real> base_coeffs;
    base_->interpolate(function, base_coeffs);

    const std::size_t base_dofs = base_->dofs_per_element();
    const std::size_t enr_dofs  = enrichment_->dofs_per_element();

    coefficients.assign(base_dofs + enr_dofs, Real(0));
    for (std::size_t i = 0; i < base_dofs && i < base_coeffs.size(); ++i) {
        coefficients[i] = base_coeffs[i];
    }
}

} // namespace spaces
} // namespace FE
} // namespace svmp

