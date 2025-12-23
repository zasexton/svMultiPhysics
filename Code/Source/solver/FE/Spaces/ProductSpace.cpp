/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/ProductSpace.h"

namespace svmp {
namespace FE {
namespace spaces {

ProductSpace::ProductSpace(std::shared_ptr<FunctionSpace> base_space,
                           int components)
    : base_(std::move(base_space)),
      components_(components) {
    FE_CHECK_NOT_NULL(base_.get(), "ProductSpace base_space");
    FE_CHECK_ARG(components_ > 0 && components_ <= 3,
                 "ProductSpace currently supports 1–3 components");
}

void ProductSpace::interpolate(const ValueFunction& function,
                               std::vector<Real>& coefficients) const {
    const std::size_t per_comp = scalar_dofs_per_component();
    coefficients.assign(dofs_per_element(), Real(0));

    std::vector<Real> comp_coeffs;
    comp_coeffs.reserve(per_comp);

    for (int c = 0; c < components_; ++c) {
        base_->interpolate_scalar(
            [&function, c](const Value& xi) -> Real {
                return function(xi)[static_cast<std::size_t>(c)];
            },
            comp_coeffs);

        FE_CHECK_ARG(comp_coeffs.size() == per_comp,
                     "ProductSpace::interpolate: component coefficient size mismatch");

        const std::size_t offset = static_cast<std::size_t>(c) * per_comp;
        for (std::size_t i = 0; i < per_comp; ++i) {
            coefficients[offset + i] = comp_coeffs[i];
        }
    }
}

FunctionSpace::Value ProductSpace::evaluate(const Value& xi,
                                            const std::vector<Real>& coefficients) const {
    const std::size_t per_comp = scalar_dofs_per_component();
    FE_CHECK_ARG(coefficients.size() == per_comp * static_cast<std::size_t>(components_),
                 "ProductSpace::evaluate: coefficient size mismatch");

    Value result{};

    std::vector<Real> coef_view;
    coef_view.resize(per_comp);

    for (int c = 0; c < components_; ++c) {
        const std::size_t offset = static_cast<std::size_t>(c) * per_comp;
        for (std::size_t i = 0; i < per_comp; ++i) {
            coef_view[i] = coefficients[offset + i];
        }

        const Real value_c = base_->evaluate_scalar(xi, coef_view);
        result[static_cast<std::size_t>(c)] = value_c;
    }

    return result;
}

} // namespace spaces
} // namespace FE
} // namespace svmp
