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

FunctionSpace::Value EnrichedSpace::evaluate(const Value& xi,
                                             const std::vector<Real>& coefficients) const {
    const std::size_t base_dofs = base_->dofs_per_element();
    const std::size_t enr_dofs  = enrichment_->dofs_per_element();
    FE_CHECK_ARG(coefficients.size() == base_dofs + enr_dofs,
                 "EnrichedSpace::evaluate: coefficient size mismatch");

    std::vector<Real> base_coeffs(base_dofs);
    std::vector<Real> enr_coeffs(enr_dofs);
    for (std::size_t i = 0; i < base_dofs; ++i) {
        base_coeffs[i] = coefficients[i];
    }
    for (std::size_t i = 0; i < enr_dofs; ++i) {
        enr_coeffs[i] = coefficients[base_dofs + i];
    }

    Value u_base = base_->evaluate(xi, base_coeffs);
    Value u_enr  = enrichment_->evaluate(xi, enr_coeffs);
    return u_base + u_enr;
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

