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
    : CompatibleTensorVectorBasis(family,
                                  semantic_basis_type,
                                  std::vector<std::shared_ptr<BasisFunction>>{
                                      std::move(first_component_basis),
                                      std::move(second_component_basis)},
                                  std::move(associations),
                                  order,
                                  element_type) {}

CompatibleTensorVectorBasis::CompatibleTensorVectorBasis(
    Family family,
    BasisType semantic_basis_type,
    std::vector<std::shared_ptr<BasisFunction>> component_bases,
    std::vector<DofAssociation> associations,
    int order,
    ElementType element_type)
    : family_(family)
    , semantic_basis_type_(semantic_basis_type)
    , element_type_(element_type)
    , order_(order)
    , component_bases_(std::move(component_bases))
    , associations_(std::move(associations)) {
    FE_CHECK_ARG(element_type_ == ElementType::Quad4 || element_type_ == ElementType::Hex8,
                 "CompatibleTensorVectorBasis: only Quad4 and Hex8 tensor-product cells are supported");

    dimension_ = (element_type_ == ElementType::Quad4) ? 2 : 3;
    FE_CHECK_ARG(component_bases_.size() == static_cast<std::size_t>(dimension_),
                 "CompatibleTensorVectorBasis: component count must match tensor dimension");

    size_ = 0;
    for (const auto& basis : component_bases_) {
        FE_CHECK_NOT_NULL(basis.get(), "CompatibleTensorVectorBasis: null component basis");
        FE_CHECK_ARG(!basis->is_vector_valued(),
                     "CompatibleTensorVectorBasis: component bases must be scalar");
        FE_CHECK_ARG(basis->dimension() == dimension_,
                     "CompatibleTensorVectorBasis: component basis dimension mismatch");
        FE_CHECK_ARG(basis->element_type() == element_type_,
                     "CompatibleTensorVectorBasis: component basis element mismatch");
        size_ += basis->size();
    }

    FE_CHECK_ARG(associations_.size() == size_,
                 "CompatibleTensorVectorBasis: DOF association size mismatch");
}

std::string CompatibleTensorVectorBasis::cache_identity() const {
    std::ostringstream oss;
    oss << "CompatibleTensorVectorBasis"
        << "|family=" << static_cast<int>(family_)
        << "|semantic=" << static_cast<int>(semantic_basis_type_)
        << "|elem=" << static_cast<int>(element_type_)
        << "|order=" << order_;
    for (std::size_t c = 0; c < component_bases_.size(); ++c) {
        oss << "|component" << c << '=' << component_bases_[c]->cache_identity();
    }
    return oss.str();
}

void CompatibleTensorVectorBasis::evaluate_vector_values(
    const math::Vector<Real, 3>& xi,
    std::vector<math::Vector<Real, 3>>& values) const {
    values.assign(size_, math::Vector<Real, 3>{});
    std::size_t offset = 0;
    for (std::size_t component = 0; component < component_bases_.size(); ++component) {
        std::vector<Real> scalar_values;
        component_bases_[component]->evaluate_values(xi, scalar_values);
        for (std::size_t i = 0; i < scalar_values.size(); ++i) {
            values[offset + i][component] = scalar_values[i];
        }
        offset += scalar_values.size();
    }
}

void CompatibleTensorVectorBasis::evaluate_vector_jacobians(
    const math::Vector<Real, 3>& xi,
    std::vector<VectorJacobian>& jacobians) const {
    jacobians.assign(size_, VectorJacobian{});
    std::size_t offset = 0;
    for (std::size_t component = 0; component < component_bases_.size(); ++component) {
        std::vector<Gradient> gradients;
        component_bases_[component]->evaluate_gradients(xi, gradients);
        for (std::size_t i = 0; i < gradients.size(); ++i) {
            for (int d = 0; d < dimension_; ++d) {
                jacobians[offset + i](component, static_cast<std::size_t>(d)) =
                    gradients[i][static_cast<std::size_t>(d)];
            }
        }
        offset += gradients.size();
    }
}

void CompatibleTensorVectorBasis::evaluate_divergence(
    const math::Vector<Real, 3>& xi,
    std::vector<Real>& divergence) const {
    FE_CHECK_ARG(family_ == Family::HDiv,
                 "CompatibleTensorVectorBasis::evaluate_divergence is only valid for H(div)");

    divergence.assign(size_, Real(0));
    std::size_t offset = 0;
    for (std::size_t component = 0; component < component_bases_.size(); ++component) {
        std::vector<Gradient> gradients;
        component_bases_[component]->evaluate_gradients(xi, gradients);
        for (std::size_t i = 0; i < gradients.size(); ++i) {
            divergence[offset + i] = gradients[i][component];
        }
        offset += gradients.size();
    }
}

void CompatibleTensorVectorBasis::evaluate_curl(
    const math::Vector<Real, 3>& xi,
    std::vector<math::Vector<Real, 3>>& curl) const {
    FE_CHECK_ARG(family_ == Family::HCurl,
                 "CompatibleTensorVectorBasis::evaluate_curl is only valid for H(curl)");

    curl.assign(size_, math::Vector<Real, 3>{});
    std::size_t offset = 0;
    for (std::size_t component = 0; component < component_bases_.size(); ++component) {
        std::vector<Gradient> gradients;
        component_bases_[component]->evaluate_gradients(xi, gradients);
        for (std::size_t i = 0; i < gradients.size(); ++i) {
            auto& value = curl[offset + i];
            if (dimension_ == 2) {
                if (component == 0u) {
                    value[2] = -gradients[i][1];
                } else {
                    value[2] = gradients[i][0];
                }
            } else {
                if (component == 0u) {
                    value[1] = gradients[i][2];
                    value[2] = -gradients[i][1];
                } else if (component == 1u) {
                    value[0] = -gradients[i][2];
                    value[2] = gradients[i][0];
                } else {
                    value[0] = gradients[i][1];
                    value[1] = -gradients[i][0];
                }
            }
        }
        offset += gradients.size();
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
