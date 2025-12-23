/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_BASISFUNCTION_H
#define SVMP_FE_BASIS_BASISFUNCTION_H

/**
 * @file BasisFunction.h
 * @brief Abstract interface for basis function evaluation on reference elements
 *
 * The Basis module operates purely on reference elements and is independent of
 * mesh-specific data structures. Implementations may leverage Math and
 * Quadrature utilities but must not read mesh connectivity or geometry.
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Math/Vector.h"
#include "Math/Matrix.h"
#include <vector>
#include <functional>

namespace svmp {
namespace FE {
namespace basis {

using Gradient = math::Vector<Real, 3>;
using Hessian  = math::Matrix<Real, 3, 3>;

/**
 * @brief Base interface for scalar and vector-valued basis families
 *
 * All basis implementations operate in reference space. Physical mappings are
 * handled by the Geometry module. Derivatives are returned with unused
 * components set to zero for lower dimensional elements.
 */
class BasisFunction {
public:
    virtual ~BasisFunction() = default;

    /// Basis family identifier
    virtual BasisType basis_type() const noexcept = 0;

    /// Underlying element type on the reference domain
    virtual ElementType element_type() const noexcept = 0;

    /// Reference dimensionality (1, 2, or 3)
    virtual int dimension() const noexcept = 0;

    /// Polynomial order (modal/nodal definition dependent)
    virtual int order() const noexcept = 0;

    /// Number of basis functions (scalar or vector-valued)
    virtual std::size_t size() const noexcept = 0;

    /// Whether the basis is vector-valued (H(div)/H(curl))
    virtual bool is_vector_valued() const noexcept { return false; }

    /**
     * @brief Evaluate scalar basis values at a reference point
     * @param xi Reference coordinates (unused entries are ignored)
     * @param[out] values Output array resized to size()
     */
    virtual void evaluate_values(const math::Vector<Real, 3>& xi,
                                 std::vector<Real>& values) const = 0;

    /**
     * @brief Evaluate gradients of scalar basis functions
     *
     * Default implementation uses central finite differences on
     * evaluate_values; override for analytic derivatives.
     */
    virtual void evaluate_gradients(const math::Vector<Real, 3>& xi,
                                    std::vector<Gradient>& gradients) const;

    /**
     * @brief Evaluate Hessians of scalar basis functions
     *
     * Default implementation differentiates evaluate_gradients using
     * finite differences. Override for analytic second derivatives.
     */
    virtual void evaluate_hessians(const math::Vector<Real, 3>& xi,
                                   std::vector<Hessian>& hessians) const;

    /**
     * @brief Evaluate vector-valued basis functions (H(div)/H(curl))
     *
     * Default implementation throws; vector bases must override.
     */
    virtual void evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                        std::vector<math::Vector<Real, 3>>& values) const;

    /// Evaluate divergence of vector-valued basis functions (if applicable)
    virtual void evaluate_divergence(const math::Vector<Real, 3>& xi,
                                     std::vector<Real>& divergence) const;

    /// Evaluate curl of vector-valued basis functions (if applicable)
    virtual void evaluate_curl(const math::Vector<Real, 3>& xi,
                               std::vector<math::Vector<Real, 3>>& curl) const;

protected:
    /// Finite-difference helper for gradients of scalar bases
    void numerical_gradient(const math::Vector<Real, 3>& xi,
                            std::vector<Gradient>& gradients,
                            Real eps = Real(1e-6)) const;

    /// Finite-difference helper for Hessians of scalar bases
    void numerical_hessian(const math::Vector<Real, 3>& xi,
                           std::vector<Hessian>& hessians,
                           Real eps = Real(1e-5)) const;
};

// -----------------------------------------------------------------------------
// Inline implementations
// -----------------------------------------------------------------------------

inline void BasisFunction::evaluate_gradients(const math::Vector<Real, 3>& xi,
                                              std::vector<Gradient>& gradients) const {
    numerical_gradient(xi, gradients);
}

inline void BasisFunction::evaluate_hessians(const math::Vector<Real, 3>& xi,
                                             std::vector<Hessian>& hessians) const {
    numerical_hessian(xi, hessians);
}

inline void BasisFunction::evaluate_vector_values(const math::Vector<Real, 3>&,
                                                  std::vector<math::Vector<Real, 3>>&) const {
    throw FEException("Vector-valued evaluation requested on scalar basis",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

inline void BasisFunction::evaluate_divergence(const math::Vector<Real, 3>&,
                                               std::vector<Real>&) const {
    throw FEException("Divergence requested on scalar basis",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

inline void BasisFunction::evaluate_curl(const math::Vector<Real, 3>&,
                                         std::vector<math::Vector<Real, 3>>&) const {
    throw FEException("Curl requested on scalar basis",
                      __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
}

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_BASISFUNCTION_H
