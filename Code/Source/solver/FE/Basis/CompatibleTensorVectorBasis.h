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
 * @brief Tensor-product compatible vector bases for spline/NURBS tensor-product cells
 */

#include "Basis/VectorBasis.h"
#include <memory>

namespace svmp {
namespace FE {
namespace basis {

/**
 * @brief Tensor-product compatible H(div)/H(curl) basis on Quad4/Hex8
 *
 * This basis provides first-class vector-valued spline/NURBS spaces on a
 * single tensor-product reference patch/cell. Each vector component is
 * represented by a compatible tensor-product scalar basis:
 *
 * - 2D H(curl): S_{p_x-1,p_y} e_x  ⊕  S_{p_x,p_y-1} e_y
 * - 2D H(div):  S_{p_x,p_y-1} e_x  ⊕  S_{p_x-1,p_y} e_y
 * - 3D H(curl): S_{p_x-1,p_y,p_z} e_x
 *             ⊕ S_{p_x,p_y-1,p_z} e_y
 *             ⊕ S_{p_x,p_y,p_z-1} e_z
 * - 3D H(div):  S_{p_x,p_y-1,p_z-1} e_x
 *             ⊕ S_{p_x-1,p_y,p_z-1} e_y
 *             ⊕ S_{p_x-1,p_y-1,p_z} e_z
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

    CompatibleTensorVectorBasis(Family family,
                                BasisType semantic_basis_type,
                                std::vector<std::shared_ptr<BasisFunction>> component_bases,
                                std::vector<DofAssociation> associations,
                                int order,
                                ElementType element_type);

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
    std::vector<std::shared_ptr<BasisFunction>> component_bases_;
    std::vector<DofAssociation> associations_;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_COMPATIBLETENSORVECTORBASIS_H
