/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/FunctionSpace.h"

#include "Basis/BasisFunction.h"
#include "Spaces/SpaceCache.h"

#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace spaces {

namespace {

/// Simple dynamic dense solver for small linear systems using Gaussian elimination
inline void solve_dense_system(std::vector<Real>& A,
                               std::vector<Real>& b,
                               std::size_t n) {
    FE_CHECK_ARG(A.size() == n * n, "solve_dense_system: matrix size mismatch");
    FE_CHECK_ARG(b.size() == n, "solve_dense_system: RHS size mismatch");

    const Real eps = std::numeric_limits<Real>::epsilon();

    // Forward elimination with partial pivoting
    for (std::size_t k = 0; k < n; ++k) {
        // Find pivot row
        std::size_t pivot_row = k;
        Real pivot_val = std::abs(A[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            Real val = std::abs(A[i * n + k]);
            if (val > pivot_val) {
                pivot_val = val;
                pivot_row = i;
            }
        }

        if (pivot_val <= eps) {
            throw AssemblyException("solve_dense_system: singular matrix in interpolation",
                                    __FILE__, static_cast<int>(__LINE__));
        }

        // Swap rows if needed
        if (pivot_row != k) {
            for (std::size_t j = k; j < n; ++j) {
                std::swap(A[k * n + j], A[pivot_row * n + j]);
            }
            std::swap(b[k], b[pivot_row]);
        }

        // Eliminate below pivot
        for (std::size_t i = k + 1; i < n; ++i) {
            const Real factor = A[i * n + k] / A[k * n + k];
            A[i * n + k] = Real(0);
            for (std::size_t j = k + 1; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Backward substitution
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
            Real sum = b[static_cast<std::size_t>(i)];
        for (std::size_t j = static_cast<std::size_t>(i) + 1; j < n; ++j) {
            sum -= A[static_cast<std::size_t>(i) * n + j] * b[j];
        }
        const Real diag = A[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(i)];
        if (std::abs(diag) <= eps) {
            throw AssemblyException("solve_dense_system: zero pivot encountered",
                                    __FILE__, static_cast<int>(__LINE__));
        }
        b[static_cast<std::size_t>(i)] = sum / diag;
    }
}

} // namespace

void FunctionSpace::interpolate(const ValueFunction& function,
                                std::vector<Real>& coefficients) const {
    const auto elem_ptr = element_ptr();
    FE_CHECK_NOT_NULL(elem_ptr.get(), "FunctionSpace::element_ptr");

    const auto& elem = *elem_ptr;
    const auto& basis = elem.basis();
    const auto quad = elem.quadrature();
    FE_CHECK_NOT_NULL(quad.get(), "FunctionSpace::quadrature");

    const std::size_t ndofs = elem.num_dofs();
    if (ndofs == 0) {
        coefficients.clear();
        return;
    }

    // Assemble local mass matrix and RHS: M_ij = ∫ (phi_i · phi_j), b_i = ∫ (f · phi_i)
    std::vector<Real> M(ndofs * ndofs, Real(0));
    std::vector<Real> b(ndofs, Real(0));
    std::vector<Real> scalar_values;
    std::vector<math::Vector<Real, 3>> vector_values;
    const bool vector_basis = basis.is_vector_valued();

    const std::size_t nqp = quad->num_points();

    // Use cached scalar basis values when available
    const SpaceCache::CachedData* cached = nullptr;
    if (!vector_basis) {
        cached = &SpaceCache::instance().get(elem, polynomial_order());
        // If cache is empty (e.g., no quadrature), fall back to direct evaluation
        if (cached->num_qpts != nqp || cached->num_dofs != ndofs) {
            cached = nullptr;
        }
    }

    for (std::size_t q = 0; q < nqp; ++q) {
        const auto xi = quad->point(q);
        const Real w = quad->weight(q);

        const Value f_val = function(xi);

        if (vector_basis) {
            vector_values.resize(ndofs);
            basis.evaluate_vector_values(xi, vector_values);

            for (std::size_t i = 0; i < ndofs; ++i) {
                const Real fi = f_val.dot(vector_values[i]);
                b[i] += w * fi;

                for (std::size_t j = 0; j < ndofs; ++j) {
                    const Real mij = vector_values[i].dot(vector_values[j]);
                    M[i * ndofs + j] += w * mij;
                }
            }
        } else {
            if (cached) {
                const auto& values_iq = cached->basis_values;
                const Real f_scalar = f_val[0];
                for (std::size_t i = 0; i < ndofs; ++i) {
                    const Real phi_i = values_iq[i][q];
                    b[i] += w * f_scalar * phi_i;
                    for (std::size_t j = 0; j < ndofs; ++j) {
                        M[i * ndofs + j] += w * phi_i * values_iq[j][q];
                    }
                }
            } else {
                scalar_values.resize(ndofs);
                basis.evaluate_values(xi, scalar_values);

                const Real f_scalar = f_val[0];
                for (std::size_t i = 0; i < ndofs; ++i) {
                    const Real phi_i = scalar_values[i];
                    b[i] += w * f_scalar * phi_i;

                    for (std::size_t j = 0; j < ndofs; ++j) {
                        M[i * ndofs + j] += w * phi_i * scalar_values[j];
                    }
                }
            }
        }
    }

    // Solve M c = b in-place (solution overwrites b)
    solve_dense_system(M, b, ndofs);
    coefficients = std::move(b);
}

void FunctionSpace::interpolate_scalar(const std::function<Real(const Value&)>& function,
                                       std::vector<Real>& coefficients) const {
    ValueFunction wrapper = [&function](const Value& x) -> Value {
        Value v{};
        v[0] = function(x);
        return v;
    };
    interpolate(wrapper, coefficients);
}

FunctionSpace::Value FunctionSpace::evaluate(const Value& xi,
                                             const std::vector<Real>& coefficients) const {
    const auto elem_ptr = element_ptr();
    FE_CHECK_NOT_NULL(elem_ptr.get(), "FunctionSpace::element_ptr");

    const auto& elem = *elem_ptr;
    const auto& basis = elem.basis();

    const std::size_t ndofs = elem.num_dofs();
    FE_CHECK_ARG(coefficients.size() == ndofs,
                 "FunctionSpace::evaluate: coefficient size mismatch");

    Value result{};

    if (basis.is_vector_valued()) {
        std::vector<math::Vector<Real, 3>> values;
        values.resize(ndofs);
        basis.evaluate_vector_values(xi, values);

        for (std::size_t i = 0; i < ndofs; ++i) {
            result += values[i] * coefficients[i];
        }
    } else {
        std::vector<Real> values;
        values.resize(ndofs);
        basis.evaluate_values(xi, values);

        for (std::size_t i = 0; i < ndofs; ++i) {
            result[0] += values[i] * coefficients[i];
        }
    }

    return result;
}

FunctionSpace::Gradient FunctionSpace::evaluate_gradient(
    const Value& xi,
    const std::vector<Real>& coefficients) const {
    const auto elem_ptr = element_ptr();
    FE_CHECK_NOT_NULL(elem_ptr.get(), "FunctionSpace::element_ptr");

    const auto& elem = *elem_ptr;
    const auto& basis = elem.basis();

    if (basis.is_vector_valued()) {
        FE_THROW(NotImplementedException,
                 "FunctionSpace::evaluate_gradient is not implemented for vector-valued bases");
    }

    const std::size_t ndofs = elem.num_dofs();
    FE_CHECK_ARG(coefficients.size() == ndofs,
                 "FunctionSpace::evaluate_gradient: coefficient size mismatch");

    Gradient result{};
    if (ndofs == 0) {
        return result;
    }

    std::vector<basis::Gradient> gradients;
    gradients.resize(ndofs);
    basis.evaluate_gradients(xi, gradients);

    for (std::size_t i = 0; i < ndofs; ++i) {
        result += gradients[i] * coefficients[i];
    }

    return result;
}

Real FunctionSpace::evaluate_divergence(
    const Value& xi,
    const std::vector<Real>& coefficients) const {
    const auto elem_ptr = element_ptr();
    FE_CHECK_NOT_NULL(elem_ptr.get(), "FunctionSpace::element_ptr");

    const auto& elem = *elem_ptr;
    const auto& basis = elem.basis();

    if (!basis.is_vector_valued()) {
        FE_THROW(NotImplementedException,
                 "FunctionSpace::evaluate_divergence is only defined for vector-valued bases");
    }

    const std::size_t ndofs = elem.num_dofs();
    FE_CHECK_ARG(coefficients.size() == ndofs,
                 "FunctionSpace::evaluate_divergence: coefficient size mismatch");

    std::vector<Real> divergence;
    divergence.resize(ndofs);
    basis.evaluate_divergence(xi, divergence);

    Real result = Real(0);
    for (std::size_t i = 0; i < ndofs; ++i) {
        result += divergence[i] * coefficients[i];
    }
    return result;
}

FunctionSpace::Value FunctionSpace::evaluate_curl(
    const Value& xi,
    const std::vector<Real>& coefficients) const {
    const auto elem_ptr = element_ptr();
    FE_CHECK_NOT_NULL(elem_ptr.get(), "FunctionSpace::element_ptr");

    const auto& elem = *elem_ptr;
    const auto& basis = elem.basis();

    if (!basis.is_vector_valued()) {
        FE_THROW(NotImplementedException,
                 "FunctionSpace::evaluate_curl is only defined for vector-valued bases");
    }

    const std::size_t ndofs = elem.num_dofs();
    FE_CHECK_ARG(coefficients.size() == ndofs,
                 "FunctionSpace::evaluate_curl: coefficient size mismatch");

    std::vector<Value> curls;
    curls.resize(ndofs);
    basis.evaluate_curl(xi, curls);

    Value result{};
    for (std::size_t i = 0; i < ndofs; ++i) {
        result += curls[i] * coefficients[i];
    }
    return result;
}

} // namespace spaces
} // namespace FE
} // namespace svmp
