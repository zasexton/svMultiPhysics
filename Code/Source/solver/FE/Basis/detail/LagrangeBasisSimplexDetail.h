#ifndef SVMP_FE_BASIS_DETAIL_LAGRANGEBASISSIMPLEXDETAIL_H
#define SVMP_FE_BASIS_DETAIL_LAGRANGEBASISSIMPLEXDETAIL_H

// Private helper for LagrangeBasis.cpp.
// This header is only intended to be included after the FE basis aliases are
// already available.

#include <array>
#include <vector>

namespace svmp {
namespace FE {
namespace basis {
namespace detail {

// Falling-factorial (equispaced barycentric) Lagrange factors for simplex nodes.
//
// For a fixed polynomial order p and barycentric coordinate lambda in [0, 1],
// define
//   phi_a(lambda) = product_{m=0}^{a-1} (p * lambda - m) / (a - m), a = 0..p
// Then for a multi-index (i0, i1, ..., id) with sum i_k = p, the simplex
// Lagrange basis function is product_k phi_{i_k}(lambda_k), nodal on the
// barycentric lattice.
//
// Output buffers must each be sized to at least p+1 entries; the function
// writes every output slot (no pre-zero required by the caller).
inline void simplex_lagrange_factor_sequence(int p,
                                             Real lambda,
                                             Real* phi,
                                             Real* dphi,
                                             Real* d2phi) {
    phi[0] = Real(1);
    dphi[0] = Real(0);
    d2phi[0] = Real(0);
    if (p == 0) {
        return;
    }

    const Real t = static_cast<Real>(p) * lambda;
    const Real dt_dlambda = static_cast<Real>(p);

    Real dphi_dt_prev = Real(0);
    Real d2phi_dt2_prev = Real(0);

    for (int a = 1; a <= p; ++a) {
        const std::size_t au = static_cast<std::size_t>(a);
        const Real inv_a = Real(1) / static_cast<Real>(a);
        const Real s = (t - static_cast<Real>(a - 1)) * inv_a;

        phi[au] = s * phi[au - 1];

        const Real dphi_dt = inv_a * phi[au - 1] + s * dphi_dt_prev;
        const Real d2phi_dt2 = Real(2) * inv_a * dphi_dt_prev + s * d2phi_dt2_prev;

        dphi[au] = dt_dlambda * dphi_dt;
        d2phi[au] = dt_dlambda * dt_dlambda * d2phi_dt2;

        dphi_dt_prev = dphi_dt;
        d2phi_dt2_prev = d2phi_dt2;
    }
}

// Per-thread scratch space for simplex factor sequences. Each scratch holds
// one barycentric coordinate's (phi, dphi, d2phi) triple of length p+1.
struct SimplexAxisScratch {
    std::vector<Real> phi;
    std::vector<Real> dphi;
    std::vector<Real> d2phi;

    void reserveFor(std::size_t n) {
        if (phi.size() < n) phi.resize(n);
        if (dphi.size() < n) dphi.resize(n);
        if (d2phi.size() < n) d2phi.resize(n);
    }
};

inline SimplexAxisScratch& simplex_axis_scratch_slot(int slot) {
    thread_local SimplexAxisScratch s[4];
    return s[slot];
}

inline void evaluate_triangle_simplex_basis(const std::vector<std::array<int, 4>>& simplex_exponents,
                                            int order,
                                            const math::Vector<Real, 3>& xi,
                                            std::vector<Real>* values = nullptr,
                                            std::vector<Gradient>* gradients = nullptr,
                                            std::vector<Hessian>* hessians = nullptr) {
    const Real l1 = xi[0];
    const Real l2 = xi[1];
    const Real l0 = Real(1) - l1 - l2;

    const std::size_t n = static_cast<std::size_t>(order + 1);
    SimplexAxisScratch& s0 = simplex_axis_scratch_slot(0);
    SimplexAxisScratch& s1 = simplex_axis_scratch_slot(1);
    SimplexAxisScratch& s2 = simplex_axis_scratch_slot(2);
    s0.reserveFor(n);
    s1.reserveFor(n);
    s2.reserveFor(n);

    simplex_lagrange_factor_sequence(order, l0, s0.phi.data(), s0.dphi.data(), s0.d2phi.data());
    simplex_lagrange_factor_sequence(order, l1, s1.phi.data(), s1.dphi.data(), s1.d2phi.data());
    simplex_lagrange_factor_sequence(order, l2, s2.phi.data(), s2.dphi.data(), s2.d2phi.data());

    const std::size_t num_nodes = simplex_exponents.size();
    if (values != nullptr) {
        values->resize(num_nodes);
    }
    if (gradients != nullptr) {
        gradients->resize(num_nodes);
    }
    if (hessians != nullptr) {
        hessians->resize(num_nodes);
    }

    for (std::size_t n_idx = 0; n_idx < num_nodes; ++n_idx) {
        const auto& e = simplex_exponents[n_idx];
        const std::size_t i0 = static_cast<std::size_t>(e[0]);
        const std::size_t i1 = static_cast<std::size_t>(e[1]);
        const std::size_t i2 = static_cast<std::size_t>(e[2]);

        const Real v0 = s0.phi[i0];
        const Real v1 = s1.phi[i1];
        const Real v2 = s2.phi[i2];
        if (values != nullptr) {
            (*values)[n_idx] = v0 * v1 * v2;
        }

        const Real D0 = s0.dphi[i0];
        const Real D1 = s1.dphi[i1];
        const Real D2 = s2.dphi[i2];

        const Real dl0 = D0 * v1 * v2;
        const Real dl1 = v0 * D1 * v2;
        const Real dl2 = v0 * v1 * D2;

        if (gradients != nullptr) {
            (*gradients)[n_idx][0] = dl1 - dl0;
            (*gradients)[n_idx][1] = dl2 - dl0;
            (*gradients)[n_idx][2] = Real(0);
        }

        if (hessians != nullptr) {
            const Real DD0 = s0.d2phi[i0];
            const Real DD1 = s1.d2phi[i1];
            const Real DD2 = s2.d2phi[i2];

            const Real H00 = DD0 * v1 * v2;
            const Real H11 = v0 * DD1 * v2;
            const Real H22 = v0 * v1 * DD2;
            const Real H01 = D0 * D1 * v2;
            const Real H02 = D0 * v1 * D2;
            const Real H12 = v0 * D1 * D2;

            Hessian H{};
            H(0, 0) = H00 - Real(2) * H01 + H11;
            H(1, 1) = H00 - Real(2) * H02 + H22;
            H(0, 1) = H00 - H01 - H02 + H12;
            H(1, 0) = H(0, 1);
            (*hessians)[n_idx] = H;
        }
    }
}

inline void evaluate_tetrahedron_simplex_basis(const std::vector<std::array<int, 4>>& simplex_exponents,
                                               int order,
                                               const math::Vector<Real, 3>& xi,
                                               std::vector<Real>* values = nullptr,
                                               std::vector<Gradient>* gradients = nullptr,
                                               std::vector<Hessian>* hessians = nullptr) {
    const Real l1 = xi[0];
    const Real l2 = xi[1];
    const Real l3 = xi[2];
    const Real l0 = Real(1) - l1 - l2 - l3;

    const std::size_t n = static_cast<std::size_t>(order + 1);
    SimplexAxisScratch& s0 = simplex_axis_scratch_slot(0);
    SimplexAxisScratch& s1 = simplex_axis_scratch_slot(1);
    SimplexAxisScratch& s2 = simplex_axis_scratch_slot(2);
    SimplexAxisScratch& s3 = simplex_axis_scratch_slot(3);
    s0.reserveFor(n);
    s1.reserveFor(n);
    s2.reserveFor(n);
    s3.reserveFor(n);

    simplex_lagrange_factor_sequence(order, l0, s0.phi.data(), s0.dphi.data(), s0.d2phi.data());
    simplex_lagrange_factor_sequence(order, l1, s1.phi.data(), s1.dphi.data(), s1.d2phi.data());
    simplex_lagrange_factor_sequence(order, l2, s2.phi.data(), s2.dphi.data(), s2.d2phi.data());
    simplex_lagrange_factor_sequence(order, l3, s3.phi.data(), s3.dphi.data(), s3.d2phi.data());

    const std::size_t num_nodes = simplex_exponents.size();
    if (values != nullptr) {
        values->resize(num_nodes);
    }
    if (gradients != nullptr) {
        gradients->resize(num_nodes);
    }
    if (hessians != nullptr) {
        hessians->resize(num_nodes);
    }

    for (std::size_t n_idx = 0; n_idx < num_nodes; ++n_idx) {
        const auto& e = simplex_exponents[n_idx];
        const std::size_t i0 = static_cast<std::size_t>(e[0]);
        const std::size_t i1 = static_cast<std::size_t>(e[1]);
        const std::size_t i2 = static_cast<std::size_t>(e[2]);
        const std::size_t i3 = static_cast<std::size_t>(e[3]);

        const Real v0 = s0.phi[i0];
        const Real v1 = s1.phi[i1];
        const Real v2 = s2.phi[i2];
        const Real v3 = s3.phi[i3];
        if (values != nullptr) {
            (*values)[n_idx] = v0 * v1 * v2 * v3;
        }

        const Real D0 = s0.dphi[i0];
        const Real D1 = s1.dphi[i1];
        const Real D2 = s2.dphi[i2];
        const Real D3 = s3.dphi[i3];

        const Real dl0 = D0 * v1 * v2 * v3;
        const Real dl1 = v0 * D1 * v2 * v3;
        const Real dl2 = v0 * v1 * D2 * v3;
        const Real dl3 = v0 * v1 * v2 * D3;

        if (gradients != nullptr) {
            (*gradients)[n_idx][0] = dl1 - dl0;
            (*gradients)[n_idx][1] = dl2 - dl0;
            (*gradients)[n_idx][2] = dl3 - dl0;
        }

        if (hessians != nullptr) {
            const Real DD0 = s0.d2phi[i0];
            const Real DD1 = s1.d2phi[i1];
            const Real DD2 = s2.d2phi[i2];
            const Real DD3 = s3.d2phi[i3];

            const Real H00 = DD0 * v1 * v2 * v3;
            const Real H11 = v0 * DD1 * v2 * v3;
            const Real H22 = v0 * v1 * DD2 * v3;
            const Real H33 = v0 * v1 * v2 * DD3;

            const Real H01 = D0 * D1 * v2 * v3;
            const Real H02 = D0 * v1 * D2 * v3;
            const Real H03 = D0 * v1 * v2 * D3;
            const Real H12 = v0 * D1 * D2 * v3;
            const Real H13 = v0 * D1 * v2 * D3;
            const Real H23 = v0 * v1 * D2 * D3;

            Hessian H{};
            H(0, 0) = H00 - Real(2) * H01 + H11;
            H(1, 1) = H00 - Real(2) * H02 + H22;
            H(2, 2) = H00 - Real(2) * H03 + H33;
            H(0, 1) = H00 - H01 - H02 + H12;
            H(1, 0) = H(0, 1);
            H(0, 2) = H00 - H01 - H03 + H13;
            H(2, 0) = H(0, 2);
            H(1, 2) = H00 - H02 - H03 + H23;
            H(2, 1) = H(1, 2);
            (*hessians)[n_idx] = H;
        }
    }
}

} // namespace detail
} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_DETAIL_LAGRANGEBASISSIMPLEXDETAIL_H
