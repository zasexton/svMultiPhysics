/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Basis/CompatibleTensorVectorBasis.h"

#include "Core/FEException.h"

#include <sstream>

namespace svmp {
namespace FE {
namespace basis {

CompatibleTensorVectorBasis::CompatibleTensorVectorBasis(
    Family family,
    BasisType semantic_basis_type,
    std::shared_ptr<BasisFunction> first_component_basis,
    std::shared_ptr<BasisFunction> second_component_basis,
    std::vector<DofAssociation> associations,
    int order,
    ElementType element_type)
    : family_(family)
    , semantic_basis_type_(semantic_basis_type)
    , element_type_(element_type)
    , order_(order)
    , first_component_basis_(std::move(first_component_basis))
    , second_component_basis_(std::move(second_component_basis))
    , associations_(std::move(associations)) {
    FE_CHECK_NOT_NULL(first_component_basis_.get(),
                      "CompatibleTensorVectorBasis: first component basis");
    FE_CHECK_NOT_NULL(second_component_basis_.get(),
                      "CompatibleTensorVectorBasis: second component basis");
    FE_CHECK_ARG(!first_component_basis_->is_vector_valued() &&
                     !second_component_basis_->is_vector_valued(),
                 "CompatibleTensorVectorBasis: component bases must be scalar");
    FE_CHECK_ARG(first_component_basis_->dimension() == 2 &&
                     second_component_basis_->dimension() == 2,
                 "CompatibleTensorVectorBasis: component bases must be two-dimensional");
    FE_CHECK_ARG(first_component_basis_->element_type() == ElementType::Quad4 &&
                     second_component_basis_->element_type() == ElementType::Quad4,
                 "CompatibleTensorVectorBasis: component bases must be quadrilateral tensor-product bases");

    size_ = first_component_basis_->size() + second_component_basis_->size();
    FE_CHECK_ARG(associations_.size() == size_,
                 "CompatibleTensorVectorBasis: DOF association size mismatch");
}

std::string CompatibleTensorVectorBasis::cache_identity() const {
    std::ostringstream oss;
    oss << "CompatibleTensorVectorBasis"
        << "|family=" << static_cast<int>(family_)
        << "|semantic=" << static_cast<int>(semantic_basis_type_)
        << "|elem=" << static_cast<int>(element_type_)
        << "|order=" << order_
        << "|first=" << first_component_basis_->cache_identity()
        << "|second=" << second_component_basis_->cache_identity();
    return oss.str();
}

void CompatibleTensorVectorBasis::evaluate_vector_values(
    const math::Vector<Real, 3>& xi,
    std::vector<math::Vector<Real, 3>>& values) const {
    std::vector<Real> first_values;
    std::vector<Real> second_values;
    first_component_basis_->evaluate_values(xi, first_values);
    second_component_basis_->evaluate_values(xi, second_values);

    values.assign(size_, math::Vector<Real, 3>{});
    for (std::size_t i = 0; i < first_values.size(); ++i) {
        values[i][0] = first_values[i];
    }
    const std::size_t offset = first_values.size();
    for (std::size_t i = 0; i < second_values.size(); ++i) {
        values[offset + i][1] = second_values[i];
    }
}

void CompatibleTensorVectorBasis::evaluate_divergence(
    const math::Vector<Real, 3>& xi,
    std::vector<Real>& divergence) const {
    FE_CHECK_ARG(family_ == Family::HDiv,
                 "CompatibleTensorVectorBasis::evaluate_divergence is only valid for H(div)");

    std::vector<Gradient> first_gradients;
    std::vector<Gradient> second_gradients;
    first_component_basis_->evaluate_gradients(xi, first_gradients);
    second_component_basis_->evaluate_gradients(xi, second_gradients);

    divergence.assign(size_, Real(0));
    for (std::size_t i = 0; i < first_gradients.size(); ++i) {
        divergence[i] = first_gradients[i][0];
    }
    const std::size_t offset = first_gradients.size();
    for (std::size_t i = 0; i < second_gradients.size(); ++i) {
        divergence[offset + i] = second_gradients[i][1];
    }
}

void CompatibleTensorVectorBasis::evaluate_curl(
    const math::Vector<Real, 3>& xi,
    std::vector<math::Vector<Real, 3>>& curl) const {
    FE_CHECK_ARG(family_ == Family::HCurl,
                 "CompatibleTensorVectorBasis::evaluate_curl is only valid for H(curl)");

    std::vector<Gradient> first_gradients;
    std::vector<Gradient> second_gradients;
    first_component_basis_->evaluate_gradients(xi, first_gradients);
    second_component_basis_->evaluate_gradients(xi, second_gradients);

    curl.assign(size_, math::Vector<Real, 3>{});
    for (std::size_t i = 0; i < first_gradients.size(); ++i) {
        curl[i][2] = -first_gradients[i][1];
    }
    const std::size_t offset = first_gradients.size();
    for (std::size_t i = 0; i < second_gradients.size(); ++i) {
        curl[offset + i][2] = second_gradients[i][0];
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
