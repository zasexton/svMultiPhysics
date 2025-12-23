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
 * @brief Lagrange basis implementation for common element families
 *
 * Supports lines, triangles, quadrilaterals, tetrahedra, and hexahedra with
 * arbitrary polynomial order. Node locations are generated on canonical
 * reference elements with equispaced coordinates (or barycentric grids on
 * simplices).
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
    ElementType element_type_;
    int dimension_;
    int order_;

    std::vector<Real> nodes_1d_;
    std::vector<math::Vector<Real, 3>> nodes_;
    std::vector<Real> denominators_1d_;
    std::vector<std::array<int, 4>> simplex_exponents_;
    std::vector<Real> simplex_coefficients_;
    // Optional permutation that maps external (NodeOrderingConventions/VTK)
    // node ordering to the internal basis enumeration used by evaluators.
    // If empty, the internal ordering is already the public ordering.
    std::vector<std::size_t> external_to_internal_;

    void init_nodes();
    void init_simplex_table();

    // 1D helpers
    std::vector<Real> evaluate_1d(Real xi) const;
    std::vector<Real> evaluate_1d_derivative(Real xi) const;
    std::vector<Real> evaluate_1d_second_derivative(Real xi) const;

    // Wedge helpers reuse triangle simplex tables and 1D nodes along z
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_LAGRANGEBASIS_H
