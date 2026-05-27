#include "LagrangeBasisSimplex.h"

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
void simplex_lagrange_factor_sequence(int p,
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

constexpr int kFixedSimplexAxisOrder = 12;
constexpr std::size_t kFixedSimplexAxisSize =
    static_cast<std::size_t>(kFixedSimplexAxisOrder + 1);

// Per-thread scratch space for simplex factor sequences. Common low orders use
// fixed storage; higher orders fall back to dynamic vectors.
struct SimplexAxisScratch {
    std::size_t size{0};
    std::array<Real, kFixedSimplexAxisSize> phi_fixed{};
    std::array<Real, kFixedSimplexAxisSize> dphi_fixed{};
    std::array<Real, kFixedSimplexAxisSize> d2phi_fixed{};
    std::vector<Real> phi_dynamic;
    std::vector<Real> dphi_dynamic;
    std::vector<Real> d2phi_dynamic;

    void reserveFor(std::size_t n) {
        size = n;
        if (n <= kFixedSimplexAxisSize) {
            return;
        }
        if (phi_dynamic.size() < n) phi_dynamic.resize(n);
        if (dphi_dynamic.size() < n) dphi_dynamic.resize(n);
        if (d2phi_dynamic.size() < n) d2phi_dynamic.resize(n);
    }

    Real* phi() noexcept {
        return size <= kFixedSimplexAxisSize ? phi_fixed.data() : phi_dynamic.data();
    }

    Real* dphi() noexcept {
        return size <= kFixedSimplexAxisSize ? dphi_fixed.data() : dphi_dynamic.data();
    }

    Real* d2phi() noexcept {
        return size <= kFixedSimplexAxisSize ? d2phi_fixed.data() : d2phi_dynamic.data();
    }

    const Real* phi() const noexcept {
        return size <= kFixedSimplexAxisSize ? phi_fixed.data() : phi_dynamic.data();
    }

    const Real* dphi() const noexcept {
        return size <= kFixedSimplexAxisSize ? dphi_fixed.data() : dphi_dynamic.data();
    }

    const Real* d2phi() const noexcept {
        return size <= kFixedSimplexAxisSize ? d2phi_fixed.data() : d2phi_dynamic.data();
    }
};

SimplexAxisScratch& simplex_axis_scratch_slot(int slot) {
    thread_local SimplexAxisScratch s[4];
    return s[slot];
}

struct SimplexVectorSink {
    std::vector<Real>* values;
    std::vector<Gradient>* gradients;
    std::vector<Hessian>* hessians;

    bool wants_values() const noexcept { return values != nullptr; }
    bool wants_gradients() const noexcept { return gradients != nullptr; }
    bool wants_hessians() const noexcept { return hessians != nullptr; }

    void prepare(std::size_t n_nodes) const {
        if (values)    values->resize(n_nodes);
        if (gradients) gradients->resize(n_nodes);
        if (hessians)  hessians->resize(n_nodes);
    }

    void write_value(std::size_t n, Real value) const {
        (*values)[n] = value;
    }

    void write_gradient(std::size_t n, Real x, Real y, Real z) const {
        auto& gradient = (*gradients)[n];
        gradient[0] = x;
        gradient[1] = y;
        gradient[2] = z;
    }

    void write_hessian(std::size_t n,
                       Real xx,
                       Real yy,
                       Real zz,
                       Real xy,
                       Real xz,
                       Real yz) const {
        Hessian hessian{};
        hessian(0, 0) = xx;
        hessian(1, 1) = yy;
        hessian(2, 2) = zz;
        hessian(0, 1) = xy; hessian(1, 0) = xy;
        hessian(0, 2) = xz; hessian(2, 0) = xz;
        hessian(1, 2) = yz; hessian(2, 1) = yz;
        (*hessians)[n] = hessian;
    }
};

struct SimplexRawSink {
    Real* values;
    Real* gradients;
    Real* hessians;

    bool wants_values() const noexcept { return values != nullptr; }
    bool wants_gradients() const noexcept { return gradients != nullptr; }
    bool wants_hessians() const noexcept { return hessians != nullptr; }

    void prepare(std::size_t) const {}

    void write_value(std::size_t n, Real value) const {
        values[n] = value;
    }

    void write_gradient(std::size_t n, Real x, Real y, Real z) const {
        Real* gradient = gradients + n * 3u;
        gradient[0] = x;
        gradient[1] = y;
        gradient[2] = z;
    }

    void write_hessian(std::size_t n,
                       Real xx,
                       Real yy,
                       Real zz,
                       Real xy,
                       Real xz,
                       Real yz) const {
        Real* hessian = hessians + n * 9u;
        hessian[0] = xx;
        hessian[1] = xy;
        hessian[2] = xz;
        hessian[3] = xy;
        hessian[4] = yy;
        hessian[5] = yz;
        hessian[6] = xz;
        hessian[7] = yz;
        hessian[8] = zz;
    }
};

template <typename Sink>
void evaluate_triangle_simplex_basis_impl(const std::vector<std::array<int, 4>>& simplex_exponents,
                                          int order,
                                          const math::Vector<Real, 3>& xi,
                                          const Sink& sink) {
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

    simplex_lagrange_factor_sequence(order, l0, s0.phi(), s0.dphi(), s0.d2phi());
    simplex_lagrange_factor_sequence(order, l1, s1.phi(), s1.dphi(), s1.d2phi());
    simplex_lagrange_factor_sequence(order, l2, s2.phi(), s2.dphi(), s2.d2phi());
    const Real* phi0 = s0.phi();
    const Real* phi1 = s1.phi();
    const Real* phi2 = s2.phi();
    const Real* dphi0 = s0.dphi();
    const Real* dphi1 = s1.dphi();
    const Real* dphi2 = s2.dphi();
    const Real* d2phi0 = s0.d2phi();
    const Real* d2phi1 = s1.d2phi();
    const Real* d2phi2 = s2.d2phi();

    const std::size_t num_nodes = simplex_exponents.size();
    sink.prepare(num_nodes);
    const bool need_values = sink.wants_values();
    const bool need_gradients = sink.wants_gradients();
    const bool need_hessians = sink.wants_hessians();

    for (std::size_t n_idx = 0; n_idx < num_nodes; ++n_idx) {
        const auto& e = simplex_exponents[n_idx];
        const std::size_t i0 = static_cast<std::size_t>(e[0]);
        const std::size_t i1 = static_cast<std::size_t>(e[1]);
        const std::size_t i2 = static_cast<std::size_t>(e[2]);

        const Real v0 = phi0[i0];
        const Real v1 = phi1[i1];
        const Real v2 = phi2[i2];
        if (need_values) {
            sink.write_value(n_idx, v0 * v1 * v2);
        }

        const Real D0 = dphi0[i0];
        const Real D1 = dphi1[i1];
        const Real D2 = dphi2[i2];

        const Real dl0 = D0 * v1 * v2;
        const Real dl1 = v0 * D1 * v2;
        const Real dl2 = v0 * v1 * D2;

        if (need_gradients) {
            sink.write_gradient(n_idx, dl1 - dl0, dl2 - dl0, Real(0));
        }

        if (need_hessians) {
            const Real DD0 = d2phi0[i0];
            const Real DD1 = d2phi1[i1];
            const Real DD2 = d2phi2[i2];

            const Real H00 = DD0 * v1 * v2;
            const Real H11 = v0 * DD1 * v2;
            const Real H22 = v0 * v1 * DD2;
            const Real H01 = D0 * D1 * v2;
            const Real H02 = D0 * v1 * D2;
            const Real H12 = v0 * D1 * D2;

            sink.write_hessian(n_idx,
                               H00 - Real(2) * H01 + H11,
                               H00 - Real(2) * H02 + H22,
                               Real(0),
                               H00 - H01 - H02 + H12,
                               Real(0),
                               Real(0));
        }
    }
}

void evaluate_triangle_simplex_basis(const std::vector<std::array<int, 4>>& simplex_exponents,
                                     int order,
                                     const math::Vector<Real, 3>& xi,
                                     std::vector<Real>* values,
                                     std::vector<Gradient>* gradients,
                                     std::vector<Hessian>* hessians) {
    const SimplexVectorSink sink{values, gradients, hessians};
    evaluate_triangle_simplex_basis_impl(simplex_exponents, order, xi, sink);
}

void evaluate_triangle_simplex_basis_to(const std::vector<std::array<int, 4>>& simplex_exponents,
                                        int order,
                                        const math::Vector<Real, 3>& xi,
                                        Real* SVMP_RESTRICT values_out,
                                        Real* SVMP_RESTRICT gradients_out,
                                        Real* SVMP_RESTRICT hessians_out) {
    const SimplexRawSink sink{values_out, gradients_out, hessians_out};
    evaluate_triangle_simplex_basis_impl(simplex_exponents, order, xi, sink);
}

void evaluate_triangle_simplex_basis_strided(
    const std::vector<std::array<int, 4>>& simplex_exponents,
    int order,
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) {
    const std::size_t num_nodes = simplex_exponents.size();
    if (points.empty() || num_nodes == 0u) {
        return;
    }

    const std::size_t sequence_size = static_cast<std::size_t>(order + 1);
    SimplexAxisScratch& s0 = simplex_axis_scratch_slot(0);
    SimplexAxisScratch& s1 = simplex_axis_scratch_slot(1);
    SimplexAxisScratch& s2 = simplex_axis_scratch_slot(2);
    s0.reserveFor(sequence_size);
    s1.reserveFor(sequence_size);
    s2.reserveFor(sequence_size);

    for (std::size_t q = 0; q < points.size(); ++q) {
        const auto& xi = points[q];
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l0 = Real(1) - l1 - l2;

        simplex_lagrange_factor_sequence(order, l0, s0.phi(), s0.dphi(), s0.d2phi());
        simplex_lagrange_factor_sequence(order, l1, s1.phi(), s1.dphi(), s1.d2phi());
        simplex_lagrange_factor_sequence(order, l2, s2.phi(), s2.dphi(), s2.d2phi());
        const Real* phi0 = s0.phi();
        const Real* phi1 = s1.phi();
        const Real* phi2 = s2.phi();
        const Real* dphi0 = s0.dphi();
        const Real* dphi1 = s1.dphi();
        const Real* dphi2 = s2.dphi();
        const Real* d2phi0 = s0.d2phi();
        const Real* d2phi1 = s1.d2phi();
        const Real* d2phi2 = s2.d2phi();

        for (std::size_t node = 0; node < num_nodes; ++node) {
            const auto& e = simplex_exponents[node];
            const std::size_t i0 = static_cast<std::size_t>(e[0]);
            const std::size_t i1 = static_cast<std::size_t>(e[1]);
            const std::size_t i2 = static_cast<std::size_t>(e[2]);

            const Real v0 = phi0[i0];
            const Real v1 = phi1[i1];
            const Real v2 = phi2[i2];
            const Real value = v0 * v1 * v2;
            if (values_out != nullptr) {
                values_out[node * output_stride + q] = value;
            }

            const Real D0 = dphi0[i0];
            const Real D1 = dphi1[i1];
            const Real D2 = dphi2[i2];

            const Real dl0 = D0 * v1 * v2;
            const Real dl1 = v0 * D1 * v2;
            const Real dl2 = v0 * v1 * D2;

            if (gradients_out != nullptr) {
                Real* g = gradients_out + node * 3u * output_stride;
                g[0u * output_stride + q] = dl1 - dl0;
                g[1u * output_stride + q] = dl2 - dl0;
                g[2u * output_stride + q] = Real(0);
            }

            if (hessians_out != nullptr) {
                const Real DD0 = d2phi0[i0];
                const Real DD1 = d2phi1[i1];
                const Real DD2 = d2phi2[i2];

                const Real H00 = DD0 * v1 * v2;
                const Real H11 = v0 * DD1 * v2;
                const Real H22 = v0 * v1 * DD2;
                const Real H01 = D0 * D1 * v2;
                const Real H02 = D0 * v1 * D2;
                const Real H12 = v0 * D1 * D2;

                Real* H = hessians_out + node * 9u * output_stride;
                const Real h01 = H00 - H01 - H02 + H12;
                H[0u * output_stride + q] = H00 - Real(2) * H01 + H11;
                H[1u * output_stride + q] = h01;
                H[2u * output_stride + q] = Real(0);
                H[3u * output_stride + q] = h01;
                H[4u * output_stride + q] = H00 - Real(2) * H02 + H22;
                H[5u * output_stride + q] = Real(0);
                H[6u * output_stride + q] = Real(0);
                H[7u * output_stride + q] = Real(0);
                H[8u * output_stride + q] = Real(0);
            }
        }
    }
}

template <typename Sink>
void evaluate_tetrahedron_simplex_basis_impl(const std::vector<std::array<int, 4>>& simplex_exponents,
                                             int order,
                                             const math::Vector<Real, 3>& xi,
                                             const Sink& sink) {
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

    simplex_lagrange_factor_sequence(order, l0, s0.phi(), s0.dphi(), s0.d2phi());
    simplex_lagrange_factor_sequence(order, l1, s1.phi(), s1.dphi(), s1.d2phi());
    simplex_lagrange_factor_sequence(order, l2, s2.phi(), s2.dphi(), s2.d2phi());
    simplex_lagrange_factor_sequence(order, l3, s3.phi(), s3.dphi(), s3.d2phi());
    const Real* phi0 = s0.phi();
    const Real* phi1 = s1.phi();
    const Real* phi2 = s2.phi();
    const Real* phi3 = s3.phi();
    const Real* dphi0 = s0.dphi();
    const Real* dphi1 = s1.dphi();
    const Real* dphi2 = s2.dphi();
    const Real* dphi3 = s3.dphi();
    const Real* d2phi0 = s0.d2phi();
    const Real* d2phi1 = s1.d2phi();
    const Real* d2phi2 = s2.d2phi();
    const Real* d2phi3 = s3.d2phi();

    const std::size_t num_nodes = simplex_exponents.size();
    sink.prepare(num_nodes);
    const bool need_values = sink.wants_values();
    const bool need_gradients = sink.wants_gradients();
    const bool need_hessians = sink.wants_hessians();

    for (std::size_t n_idx = 0; n_idx < num_nodes; ++n_idx) {
        const auto& e = simplex_exponents[n_idx];
        const std::size_t i0 = static_cast<std::size_t>(e[0]);
        const std::size_t i1 = static_cast<std::size_t>(e[1]);
        const std::size_t i2 = static_cast<std::size_t>(e[2]);
        const std::size_t i3 = static_cast<std::size_t>(e[3]);

        const Real v0 = phi0[i0];
        const Real v1 = phi1[i1];
        const Real v2 = phi2[i2];
        const Real v3 = phi3[i3];
        if (need_values) {
            sink.write_value(n_idx, v0 * v1 * v2 * v3);
        }

        const Real D0 = dphi0[i0];
        const Real D1 = dphi1[i1];
        const Real D2 = dphi2[i2];
        const Real D3 = dphi3[i3];

        const Real dl0 = D0 * v1 * v2 * v3;
        const Real dl1 = v0 * D1 * v2 * v3;
        const Real dl2 = v0 * v1 * D2 * v3;
        const Real dl3 = v0 * v1 * v2 * D3;

        if (need_gradients) {
            sink.write_gradient(n_idx, dl1 - dl0, dl2 - dl0, dl3 - dl0);
        }

        if (need_hessians) {
            const Real DD0 = d2phi0[i0];
            const Real DD1 = d2phi1[i1];
            const Real DD2 = d2phi2[i2];
            const Real DD3 = d2phi3[i3];

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

            sink.write_hessian(n_idx,
                               H00 - Real(2) * H01 + H11,
                               H00 - Real(2) * H02 + H22,
                               H00 - Real(2) * H03 + H33,
                               H00 - H01 - H02 + H12,
                               H00 - H01 - H03 + H13,
                               H00 - H02 - H03 + H23);
        }
    }
}

void evaluate_tetrahedron_simplex_basis(const std::vector<std::array<int, 4>>& simplex_exponents,
                                        int order,
                                        const math::Vector<Real, 3>& xi,
                                        std::vector<Real>* values,
                                        std::vector<Gradient>* gradients,
                                        std::vector<Hessian>* hessians) {
    const SimplexVectorSink sink{values, gradients, hessians};
    evaluate_tetrahedron_simplex_basis_impl(simplex_exponents, order, xi, sink);
}

void evaluate_tetrahedron_simplex_basis_to(const std::vector<std::array<int, 4>>& simplex_exponents,
                                           int order,
                                           const math::Vector<Real, 3>& xi,
                                           Real* SVMP_RESTRICT values_out,
                                           Real* SVMP_RESTRICT gradients_out,
                                           Real* SVMP_RESTRICT hessians_out) {
    const SimplexRawSink sink{values_out, gradients_out, hessians_out};
    evaluate_tetrahedron_simplex_basis_impl(simplex_exponents, order, xi, sink);
}

void evaluate_tetrahedron_simplex_basis_strided(
    const std::vector<std::array<int, 4>>& simplex_exponents,
    int order,
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT gradients_out,
    Real* SVMP_RESTRICT hessians_out) {
    const std::size_t num_nodes = simplex_exponents.size();
    if (points.empty() || num_nodes == 0u) {
        return;
    }

    const std::size_t sequence_size = static_cast<std::size_t>(order + 1);
    SimplexAxisScratch& s0 = simplex_axis_scratch_slot(0);
    SimplexAxisScratch& s1 = simplex_axis_scratch_slot(1);
    SimplexAxisScratch& s2 = simplex_axis_scratch_slot(2);
    SimplexAxisScratch& s3 = simplex_axis_scratch_slot(3);
    s0.reserveFor(sequence_size);
    s1.reserveFor(sequence_size);
    s2.reserveFor(sequence_size);
    s3.reserveFor(sequence_size);

    for (std::size_t q = 0; q < points.size(); ++q) {
        const auto& xi = points[q];
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l3 = xi[2];
        const Real l0 = Real(1) - l1 - l2 - l3;

        simplex_lagrange_factor_sequence(order, l0, s0.phi(), s0.dphi(), s0.d2phi());
        simplex_lagrange_factor_sequence(order, l1, s1.phi(), s1.dphi(), s1.d2phi());
        simplex_lagrange_factor_sequence(order, l2, s2.phi(), s2.dphi(), s2.d2phi());
        simplex_lagrange_factor_sequence(order, l3, s3.phi(), s3.dphi(), s3.d2phi());
        const Real* phi0 = s0.phi();
        const Real* phi1 = s1.phi();
        const Real* phi2 = s2.phi();
        const Real* phi3 = s3.phi();
        const Real* dphi0 = s0.dphi();
        const Real* dphi1 = s1.dphi();
        const Real* dphi2 = s2.dphi();
        const Real* dphi3 = s3.dphi();
        const Real* d2phi0 = s0.d2phi();
        const Real* d2phi1 = s1.d2phi();
        const Real* d2phi2 = s2.d2phi();
        const Real* d2phi3 = s3.d2phi();

        for (std::size_t node = 0; node < num_nodes; ++node) {
            const auto& e = simplex_exponents[node];
            const std::size_t i0 = static_cast<std::size_t>(e[0]);
            const std::size_t i1 = static_cast<std::size_t>(e[1]);
            const std::size_t i2 = static_cast<std::size_t>(e[2]);
            const std::size_t i3 = static_cast<std::size_t>(e[3]);

            const Real v0 = phi0[i0];
            const Real v1 = phi1[i1];
            const Real v2 = phi2[i2];
            const Real v3 = phi3[i3];
            if (values_out != nullptr) {
                values_out[node * output_stride + q] = v0 * v1 * v2 * v3;
            }

            const Real D0 = dphi0[i0];
            const Real D1 = dphi1[i1];
            const Real D2 = dphi2[i2];
            const Real D3 = dphi3[i3];

            const Real dl0 = D0 * v1 * v2 * v3;
            const Real dl1 = v0 * D1 * v2 * v3;
            const Real dl2 = v0 * v1 * D2 * v3;
            const Real dl3 = v0 * v1 * v2 * D3;

            if (gradients_out != nullptr) {
                Real* g = gradients_out + node * 3u * output_stride;
                g[0u * output_stride + q] = dl1 - dl0;
                g[1u * output_stride + q] = dl2 - dl0;
                g[2u * output_stride + q] = dl3 - dl0;
            }

            if (hessians_out != nullptr) {
                const Real DD0 = d2phi0[i0];
                const Real DD1 = d2phi1[i1];
                const Real DD2 = d2phi2[i2];
                const Real DD3 = d2phi3[i3];

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

                const Real h01 = H00 - H01 - H02 + H12;
                const Real h02 = H00 - H01 - H03 + H13;
                const Real h12 = H00 - H02 - H03 + H23;

                Real* H = hessians_out + node * 9u * output_stride;
                H[0u * output_stride + q] = H00 - Real(2) * H01 + H11;
                H[1u * output_stride + q] = h01;
                H[2u * output_stride + q] = h02;
                H[3u * output_stride + q] = h01;
                H[4u * output_stride + q] = H00 - Real(2) * H02 + H22;
                H[5u * output_stride + q] = h12;
                H[6u * output_stride + q] = h02;
                H[7u * output_stride + q] = h12;
                H[8u * output_stride + q] = H00 - Real(2) * H03 + H33;
            }
        }
    }
}

} // namespace detail
} // namespace basis
} // namespace FE
} // namespace svmp
