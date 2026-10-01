/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_EIGEN_RIGHT_GMRES_H
#define SVMP_FE_BACKENDS_EIGEN_RIGHT_GMRES_H

#if defined(FE_HAS_EIGEN)

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace svmp {
namespace FE {
namespace backends {
namespace eigen_detail {

struct RightGmresResult {
    int iterations{0};
    bool converged{false};
    double initial_residual{0.0};
    double final_residual{0.0};
};

/**
 * @brief Restarted GMRES with right preconditioning for A x = b, x0 = 0.
 *
 * The Krylov space is built for A M^{-1}; the iterate is x = M^{-1} V y.  With
 * right preconditioning the Arnoldi residual is the true residual (in exact
 * arithmetic), so the stopping test uses the true residual norm:
 * ||b - A x|| <= target, verified with an explicit residual at the end of
 * every cycle.  Orthogonalization is modified Gram-Schmidt.
 *
 * @param apply_A     callable (const VectorXd& in, VectorXd& out): out = A in
 * @param apply_Minv  callable (const VectorXd& in, VectorXd& out): out = M^{-1} in
 */
template <class ApplyA, class ApplyMinv>
RightGmresResult rightPreconditionedGmres(const ApplyA& apply_A,
                                          const ApplyMinv& apply_Minv,
                                          const Eigen::VectorXd& b,
                                          Eigen::VectorXd& x,
                                          int restart,
                                          int max_iterations,
                                          double target)
{
    const Eigen::Index n = b.size();
    x.setZero(n);
    RightGmresResult result;
    result.initial_residual = b.norm();
    result.final_residual = result.initial_residual;
    if (!std::isfinite(result.initial_residual)) {
        return result;
    }
    if (result.initial_residual <= target) {
        result.converged = true;
        return result;
    }

    const int m = std::max(1, restart);
    Eigen::MatrixXd V(n, m + 1);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m + 1, m);
    Eigen::VectorXd cs = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd sn = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(m + 1);
    Eigen::VectorXd w(n);
    Eigen::VectorXd z(n);
    Eigen::VectorXd vk(n);
    Eigen::VectorXd r = b;
    double beta = result.initial_residual;
    int total = 0;

    while (total < max_iterations) {
        V.col(0) = r / beta;
        g.setZero();
        g(0) = beta;
        H.setZero();
        int k = 0;
        bool cycle_done = false;
        while (!cycle_done) {
            vk = V.col(k);
            apply_Minv(vk, z);
            apply_A(z, w);
            ++total;
            for (int j = 0; j <= k; ++j) {
                const double hjk = V.col(j).dot(w);
                H(j, k) = hjk;
                w.noalias() -= hjk * V.col(j);
            }
            const double hnext = w.norm();
            H(k + 1, k) = hnext;
            const bool breakdown = !(hnext > 0.0) || !std::isfinite(hnext);
            if (!breakdown) {
                V.col(k + 1) = w / hnext;
            }
            for (int j = 0; j < k; ++j) {
                const double t = cs(j) * H(j, k) + sn(j) * H(j + 1, k);
                H(j + 1, k) = -sn(j) * H(j, k) + cs(j) * H(j + 1, k);
                H(j, k) = t;
            }
            const double a = H(k, k);
            const double c = H(k + 1, k);
            const double rho = std::hypot(a, c);
            cs(k) = (rho > 0.0) ? a / rho : 1.0;
            sn(k) = (rho > 0.0) ? c / rho : 0.0;
            H(k, k) = rho;
            H(k + 1, k) = 0.0;
            g(k + 1) = -sn(k) * g(k);
            g(k) = cs(k) * g(k);
            ++k;
            cycle_done = breakdown || std::abs(g(k)) <= target || k >= m || total >= max_iterations;
        }

        // y = H^{-1} g on the leading k x k triangle (skip vanishing pivots).
        Eigen::VectorXd y = Eigen::VectorXd::Zero(k);
        for (int i = k - 1; i >= 0; --i) {
            double s = g(i);
            for (int j = i + 1; j < k; ++j) {
                s -= H(i, j) * y(j);
            }
            y(i) = (H(i, i) != 0.0) ? s / H(i, i) : 0.0;
        }
        w.noalias() = V.leftCols(k) * y;
        apply_Minv(w, z);
        x += z;

        apply_A(x, w);
        r = b - w;
        beta = r.norm();
        result.final_residual = beta;
        result.iterations = total;
        if (beta <= target) {
            result.converged = true;
            break;
        }
        if (!(beta > 0.0) || !std::isfinite(beta)) {
            break;
        }
    }
    result.iterations = total;
    return result;
}

} // namespace eigen_detail
} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_EIGEN

#endif // SVMP_FE_BACKENDS_EIGEN_RIGHT_GMRES_H
