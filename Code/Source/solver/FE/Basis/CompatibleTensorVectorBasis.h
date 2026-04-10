/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_COMPATIBLETENSORVECTORBASIS_H
#define SVMP_FE_BASIS_COMPATIBLETENSORVECTORBASIS_H

/**
 * @file CompatibleTensorVectorBasis.h
 * @brief Tensor-product compatible vector bases for quad spline/NURBS spaces
 */

#include "Basis/VectorBasis.h"
#include <memory>

namespace svmp {
namespace FE {
namespace basis {

/**
 * @brief Tensor-product compatible H(div)/H(curl) basis on Quad4
 *
 * This basis provides first-class vector-valued spline/NURBS spaces on a
 * single quadrilateral reference patch. Each vector component is represented
 * by a compatible tensor-product scalar basis:
 *
 * - H(curl): S_{p_x-1,p_y} e_x  ⊕  S_{p_x,p_y-1} e_y
 * - H(div):  S_{p_x,p_y-1} e_x  ⊕  S_{p_x-1,p_y} e_y
 *
 * The current implementation targets the 2D quadrilateral surface needed by
 * the parity roadmap. Higher-dimensional compatible spline spaces can be
 * added later without changing this public contract.
 */
class CompatibleTensorVectorBasis : public VectorBasisFunction {
public:
    enum class Family : std::uint8_t {
        HCurl,
        HDiv
    };

    CompatibleTensorVectorBasis(Family family,
                                BasisType semantic_basis_type,
                                std::shared_ptr<BasisFunction> first_component_basis,
                                std::shared_ptr<BasisFunction> second_component_basis,
                                std::vector<DofAssociation> associations,
                                int order,
                                ElementType element_type = ElementType::Quad4);

    BasisType basis_type() const noexcept override { return semantic_basis_type_; }
    ElementType element_type() const noexcept override { return element_type_; }
    int dimension() const noexcept override { return dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return size_; }

    std::string cache_identity() const override;

    void evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                std::vector<math::Vector<Real, 3>>& values) const override;

    void evaluate_divergence(const math::Vector<Real, 3>& xi,
                             std::vector<Real>& divergence) const override;

    void evaluate_curl(const math::Vector<Real, 3>& xi,
                       std::vector<math::Vector<Real, 3>>& curl) const override;

    std::vector<DofAssociation> dof_associations() const override { return associations_; }

private:
    Family family_;
    BasisType semantic_basis_type_;
    ElementType element_type_;
    int dimension_{2};
    int order_{0};
    std::size_t size_{0};
    std::shared_ptr<BasisFunction> first_component_basis_;
    std::shared_ptr<BasisFunction> second_component_basis_;
    std::vector<DofAssociation> associations_;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_COMPATIBLETENSORVECTORBASIS_H
