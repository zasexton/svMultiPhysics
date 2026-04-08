/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_LAGRANGEBASIS_H
#define SVMP_FE_BASIS_LAGRANGEBASIS_H

/**
 * @file LagrangeBasis.h
 * @brief Nodal Lagrange polynomial basis on reference elements
 */

#include "BasisFunction.h"
#include <array>

namespace svmp {
namespace FE {
namespace basis {

/**
 * @brief Complete nodal H1 Lagrange basis on canonical reference topologies
 *
 * Supports arbitrary polynomial order on the canonical complete families:
 * `Line2`, `Triangle3`, `Quad4`, `Tetra4`, `Hex8`, `Wedge6`, and `Pyramid5`.
 * Low-order complete-family aliases (`Line3`, `Triangle6`, `Quad9`,
 * `Tetra10`, `Hex27`, `Wedge18`, `Pyramid14`) normalize to their canonical
 * topology plus order. Serendipity variants remain intentionally excluded.
 *
 * Node locations are generated on canonical reference elements using
 * equispaced coordinates on tensor-product elements, barycentric grids on
 * simplices, tensorized triangle-line grids on wedges, and a rational nodal
 * pyramid construction on `Pyramid5`.
 */
class LagrangeBasis : public BasisFunction {
public:
    LagrangeBasis(ElementType type, int order);

    BasisType basis_type() const noexcept override { return BasisType::Lagrange; }
    ElementType element_type() const noexcept override { return element_type_; }
    int dimension() const noexcept override { return dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return nodes_.size(); }

    const std::vector<math::Vector<Real, 3>>& nodes() const noexcept { return nodes_; }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override;
    void evaluate_gradients(const math::Vector<Real, 3>& xi,
                            std::vector<Gradient>& gradients) const override;
    void evaluate_hessians(const math::Vector<Real, 3>& xi,
                           std::vector<Hessian>& hessians) const override;

private:
    using TensorNodeIndex = std::array<std::size_t, 3>;
    using WedgeNodeIndex = std::array<std::size_t, 2>;

    ElementType element_type_;
    int dimension_;
    int order_;

    std::vector<Real> nodes_1d_;
    std::vector<math::Vector<Real, 3>> nodes_;
    std::vector<TensorNodeIndex> tensor_indices_;
    std::vector<std::array<int, 4>> simplex_exponents_;
    std::vector<WedgeNodeIndex> wedge_indices_;

    void init_nodes();
    void build_point_nodes();
    void build_tensor_product_nodes(int dimensions);
    void build_simplex_nodes();
    void build_wedge_nodes();
    void build_pyramid_nodes();
    void init_equispaced_1d_nodes();
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_LAGRANGEBASIS_H
