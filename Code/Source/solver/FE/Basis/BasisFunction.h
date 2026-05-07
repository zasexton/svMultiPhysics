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
#include "BasisExceptions.h"
#include "Math/Vector.h"
#include "Math/Matrix.h"
#include <functional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace basis {

using Gradient = math::Vector<Real, 3>;
using Hessian  = math::Matrix<Real, 3, 3>;
using VectorJacobian = math::Matrix<Real, 3, 3>;

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
     * @brief Stable semantic identity used by BasisCache
     *
     * Derived classes should override this when evaluation depends on
     * additional state beyond basis family / element / order metadata.
     */
    virtual std::string cache_identity() const;

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
     * @brief Fused evaluation of values, gradients, and Hessians at one point
     *
     * Default implementation calls evaluate_values, evaluate_gradients, and
     * evaluate_hessians in sequence. Bases that share intermediate
     * computations (e.g., LagrangeBasis sharing per-axis 1D evaluations)
     * should override this to avoid redundant work.
     */
    virtual void evaluate_all(const math::Vector<Real, 3>& xi,
                              std::vector<Real>& values,
                              std::vector<Gradient>& gradients,
                              std::vector<Hessian>& hessians) const;

    /**
     * @brief Fill SoA buffers with basis evaluations at all quadrature points
     *
     * Outputs are written directly to caller-provided strided buffers in
     * DOF-major SoA layout — no scratch+transpose required by the caller.
     * Pass `nullptr` for any output that is not needed.
     *
     *   values_out:    size num_dofs * num_qpts; element [d * num_qpts + q]
     *   gradients_out: size num_dofs * 3 * num_qpts; element [(d*3 + c) * num_qpts + q]
     *   hessians_out:  size num_dofs * 9 * num_qpts; element [(d*9 + r*3 + c) * num_qpts + q]
     *
     * Default implementation calls evaluate_all (or evaluate_values/gradients/
     * hessians as appropriate) per QP, materializing into temp buffers then
     * scatter-writing to the output. Bases that can amortize per-QP setup
     * across the quadrature rule should override.
     */
    virtual void evaluate_at_quadrature_points(
        const std::vector<math::Vector<Real, 3>>& points,
        Real* values_out,
        Real* gradients_out,
        Real* hessians_out) const;

    /**
     * @brief Evaluate scalar basis values into a caller-provided raw buffer (D3)
     *
     * Caller is responsible for providing a buffer of at least size() Real
     * entries. This avoids the per-call std::vector::resize() cost of the
     * vector-output overload. Default implementation forwards through a temp
     * vector; bases should override for direct write.
     */
    virtual void evaluate_values_to(const math::Vector<Real, 3>& xi,
                                    Real* values_out) const;

    /**
     * @brief Evaluate gradients into a flat caller-provided buffer (D3)
     *
     * Layout: gradients_out[i * 3 + c] = component c of gradient of basis i.
     * Caller provides a buffer of size() * 3 Real entries.
     */
    virtual void evaluate_gradients_to(const math::Vector<Real, 3>& xi,
                                       Real* gradients_out) const;

    /**
     * @brief Evaluate Hessians into a flat caller-provided buffer (D3)
     *
     * Layout: hessians_out[i * 9 + r * 3 + c] = H_i(r, c).
     */
    virtual void evaluate_hessians_to(const math::Vector<Real, 3>& xi,
                                      Real* hessians_out) const;

    /**
     * @brief Evaluate vector-valued basis functions (H(div)/H(curl))
     *
     * Default implementation throws; vector bases must override.
     */
    virtual void evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                        std::vector<math::Vector<Real, 3>>& values) const;

    /**
     * @brief Evaluate reference-space Jacobians of vector-valued basis functions
     *
     * The returned matrix for basis function `i` has entries
     * `jacobians[i](component, derivative_direction) = d phi_i_component / d xi_direction`.
     * Unused rows/columns are zero-filled for lower-dimensional elements.
     */
    virtual void evaluate_vector_jacobians(const math::Vector<Real, 3>& xi,
                                           std::vector<VectorJacobian>& jacobians) const;

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

inline void BasisFunction::evaluate_all(const math::Vector<Real, 3>& xi,
                                        std::vector<Real>& values,
                                        std::vector<Gradient>& gradients,
                                        std::vector<Hessian>& hessians) const {
    evaluate_values(xi, values);
    evaluate_gradients(xi, gradients);
    evaluate_hessians(xi, hessians);
}

inline void BasisFunction::evaluate_values_to(const math::Vector<Real, 3>& xi,
                                              Real* values_out) const {
    std::vector<Real> tmp(size());
    evaluate_values(xi, tmp);
    for (std::size_t i = 0; i < tmp.size(); ++i) values_out[i] = tmp[i];
}

inline void BasisFunction::evaluate_gradients_to(const math::Vector<Real, 3>& xi,
                                                 Real* gradients_out) const {
    std::vector<Gradient> tmp(size());
    evaluate_gradients(xi, tmp);
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        gradients_out[i * 3 + 0] = tmp[i][0];
        gradients_out[i * 3 + 1] = tmp[i][1];
        gradients_out[i * 3 + 2] = tmp[i][2];
    }
}

inline void BasisFunction::evaluate_hessians_to(const math::Vector<Real, 3>& xi,
                                                Real* hessians_out) const {
    std::vector<Hessian> tmp(size());
    evaluate_hessians(xi, tmp);
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                hessians_out[i * 9 + static_cast<std::size_t>(r * 3 + c)] =
                    tmp[i](static_cast<std::size_t>(r), static_cast<std::size_t>(c));
            }
        }
    }
}

inline void BasisFunction::evaluate_at_quadrature_points(
    const std::vector<math::Vector<Real, 3>>& points,
    Real* values_out,
    Real* gradients_out,
    Real* hessians_out) const {
    const std::size_t num_qpts = points.size();
    const std::size_t num_dofs = size();

    std::vector<Real> v_tmp;
    std::vector<Gradient> g_tmp;
    std::vector<Hessian> h_tmp;
    if (values_out)    v_tmp.resize(num_dofs);
    if (gradients_out) g_tmp.resize(num_dofs);
    if (hessians_out)  h_tmp.resize(num_dofs);

    for (std::size_t q = 0; q < num_qpts; ++q) {
        if (values_out && gradients_out && hessians_out) {
            evaluate_all(points[q], v_tmp, g_tmp, h_tmp);
        } else {
            if (values_out)    evaluate_values(points[q], v_tmp);
            if (gradients_out) evaluate_gradients(points[q], g_tmp);
            if (hessians_out)  evaluate_hessians(points[q], h_tmp);
        }

        if (values_out) {
            for (std::size_t d = 0; d < num_dofs; ++d) {
                values_out[d * num_qpts + q] = v_tmp[d];
            }
        }
        if (gradients_out) {
            for (std::size_t d = 0; d < num_dofs; ++d) {
                for (int c = 0; c < 3; ++c) {
                    gradients_out[(d * 3 + static_cast<std::size_t>(c)) * num_qpts + q] = g_tmp[d][static_cast<std::size_t>(c)];
                }
            }
        }
        if (hessians_out) {
            for (std::size_t d = 0; d < num_dofs; ++d) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        hessians_out[(d * 9 + static_cast<std::size_t>(r * 3 + c)) * num_qpts + q] =
                            h_tmp[d](static_cast<std::size_t>(r), static_cast<std::size_t>(c));
                    }
                }
            }
        }
    }
}

inline void BasisFunction::evaluate_vector_values(const math::Vector<Real, 3>&,
                                                  std::vector<math::Vector<Real, 3>>&) const {
    throw BasisEvaluationException("Vector-valued evaluation requested on scalar basis",
                                   __FILE__, __LINE__, __func__);
}

inline void BasisFunction::evaluate_vector_jacobians(const math::Vector<Real, 3>&,
                                                     std::vector<VectorJacobian>&) const {
    throw BasisEvaluationException("Vector-basis Jacobian evaluation requested on scalar basis",
                                   __FILE__, __LINE__, __func__);
}

inline void BasisFunction::evaluate_divergence(const math::Vector<Real, 3>&,
                                               std::vector<Real>&) const {
    throw BasisEvaluationException("Divergence requested on scalar basis",
                                   __FILE__, __LINE__, __func__);
}

inline void BasisFunction::evaluate_curl(const math::Vector<Real, 3>&,
                                         std::vector<math::Vector<Real, 3>>&) const {
    throw BasisEvaluationException("Curl requested on scalar basis",
                                   __FILE__, __LINE__, __func__);
}

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_BASISFUNCTION_H
