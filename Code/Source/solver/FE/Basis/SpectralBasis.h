/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_SPECTRALBASIS_H
#define SVMP_FE_BASIS_SPECTRALBASIS_H

/**
 * @file SpectralBasis.h
 * @brief Gauss-Lobatto-Legendre spectral bases for high-order elements
 *
 * Supported reference element families:
 * - Line
 * - Quadrilateral
 * - Hexahedron
 * - Triangle
 * - Tetrahedron
 * - Wedge (triangle spectral face tensorized with 1D GLL)
 * - Pyramid (spectral face-compatible nodal set over the standard rational
 *   pyramid modal space)
 */

#include "BasisFunction.h"
#include "Math/Vector.h"
#include <memory>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace basis {

namespace pyramid_modal {
struct Term;
} // namespace pyramid_modal
namespace detail {
struct SpectralModalMatrixData;
} // namespace detail

class SpectralBasis : public BasisFunction {
public:
    SpectralBasis(ElementType type, int order);
    ~SpectralBasis() override;
    SpectralBasis(const SpectralBasis&);
    SpectralBasis(SpectralBasis&&);
    SpectralBasis& operator=(const SpectralBasis&);
    SpectralBasis& operator=(SpectralBasis&&);

    BasisType basis_type() const noexcept override { return BasisType::Spectral; }
    ElementType element_type() const noexcept override { return element_type_; }
    int dimension() const noexcept override { return dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return size_; }
    bool cache_identity_is_structural() const noexcept override { return true; }

    const std::vector<Real>& nodes_1d() const noexcept { return nodes_1d_; }
    const std::vector<math::Vector<Real, 3>>& interpolation_nodes() const noexcept { return simplex_nodes_; }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override;
    void evaluate_gradients(const math::Vector<Real, 3>& xi,
                            std::vector<Gradient>& gradients) const override;
    void evaluate_hessians(const math::Vector<Real, 3>& xi,
                           std::vector<Hessian>& hessians) const override;
    void evaluate_all(const math::Vector<Real, 3>& xi,
                      std::vector<Real>& values,
                      std::vector<Gradient>& gradients,
                      std::vector<Hessian>& hessians) const override;
    void evaluate_values_to(const math::Vector<Real, 3>& xi,
                            Real* SVMP_RESTRICT values_out) const override;
    void evaluate_gradients_to(const math::Vector<Real, 3>& xi,
                               Real* SVMP_RESTRICT gradients_out) const override;
    void evaluate_hessians_to(const math::Vector<Real, 3>& xi,
                              Real* SVMP_RESTRICT hessians_out) const override;
    void evaluate_at_quadrature_points(
        const std::vector<math::Vector<Real, 3>>& points,
        Real* SVMP_RESTRICT values_out,
        Real* SVMP_RESTRICT gradients_out,
        Real* SVMP_RESTRICT hessians_out) const override;
    void evaluate_at_quadrature_points_strided(
        const std::vector<math::Vector<Real, 3>>& points,
        std::size_t output_stride,
        Real* SVMP_RESTRICT values_out,
        Real* SVMP_RESTRICT gradients_out,
        Real* SVMP_RESTRICT hessians_out) const override;

    /// Whether this is a simplex spectral element.
    bool is_simplex() const noexcept { return is_simplex_; }

private:
    struct FlatOutputLayout {
        std::size_t output_stride{1u};
        std::size_t q{0u};
    };

    ElementType element_type_;
    int dimension_;
    int order_;
    std::size_t size_;
    bool is_simplex_{false};
    bool is_wedge_tensor_product_{false};
    bool is_pyramid_modal_{false};

    // Tensor-product (line/quad/hex) members
    std::vector<Real> nodes_1d_;
    std::vector<Real> barycentric_weights_;
    std::vector<Real> derivative_matrix_1d_;
    std::vector<Real> second_derivative_matrix_1d_;
    std::shared_ptr<SpectralBasis> face_basis_;
    std::shared_ptr<SpectralBasis> axis_basis_;

    void build_nodes();
    void build_1d_derivative_matrices();
    void eval_1d_to(Real x, std::span<Real> values) const;
    void eval_1d_derivative_to(Real x, std::span<Real> derivatives) const;
    void eval_1d_second_derivative_to(Real x, std::span<Real> second_derivatives) const;
    void eval_1d_all_to(Real x,
                        std::span<Real> values,
                        std::span<Real> derivatives,
                        std::span<Real> second_derivatives) const;
    std::vector<Real> eval_1d(Real x) const;
    std::vector<Real> eval_1d_derivative(Real x) const;
    std::vector<Real> eval_1d_second_derivative(Real x) const;
    void eval_1d_all(Real x,
                     std::vector<Real>& values,
                     std::vector<Real>& derivatives,
                     std::vector<Real>& second_derivatives) const;

    // Simplex (triangle/tet) members -- collapsed GLL nodes + modal transform
    std::vector<math::Vector<Real, 3>> simplex_nodes_;
    std::shared_ptr<const detail::SpectralModalMatrixData> modal_matrices_;
    std::vector<pyramid_modal::Term> pyramid_modal_terms_;

    void build_simplex_nodes_triangle();
    void build_simplex_nodes_tetrahedron();
    void build_pyramid_nodes();
    void build_inverse_vandermonde();
    const std::vector<Real>& modal_to_nodal_by_basis() const noexcept;
    void apply_modal_to_nodal(std::span<const Real> modal,
                              std::span<Real> nodal) const;
    void apply_modal_gradients_to_nodal(std::span<const Gradient> modal_gradients,
                                        std::span<Gradient> nodal_gradients,
                                        int components) const;
    void apply_modal_hessians_to_nodal(std::span<const Hessian> modal_hessians,
                                       std::span<Hessian> nodal_hessians,
                                       int components) const;
    void apply_modal_values_to_nodal_strided(std::span<const Real> modal,
                                             Real* SVMP_RESTRICT values_out,
                                             std::size_t output_stride,
                                             std::size_t q) const;
    void apply_modal_gradients_to_nodal_strided(std::span<const Gradient> modal_gradients,
                                                Real* SVMP_RESTRICT gradients_out,
                                                int components,
                                                std::size_t output_stride,
                                                std::size_t q) const;
    void apply_modal_hessians_to_nodal_strided(std::span<const Hessian> modal_hessians,
                                               Real* SVMP_RESTRICT hessians_out,
                                               int components,
                                               std::size_t output_stride,
                                               std::size_t q) const;
    void evaluate_spectral_flat(const math::Vector<Real, 3>& xi,
                                Real* SVMP_RESTRICT values_out,
                                Real* SVMP_RESTRICT gradients_out,
                                Real* SVMP_RESTRICT hessians_out,
                                FlatOutputLayout layout) const;
    void write_values_strided(const math::Vector<Real, 3>& xi,
                              Real* SVMP_RESTRICT values_out,
                              std::size_t output_stride,
                              std::size_t q) const;
    void write_gradients_strided(const math::Vector<Real, 3>& xi,
                                 Real* SVMP_RESTRICT gradients_out,
                                 std::size_t output_stride,
                                 std::size_t q) const;
    void write_hessians_strided(const math::Vector<Real, 3>& xi,
                                Real* SVMP_RESTRICT hessians_out,
                                std::size_t output_stride,
                                std::size_t q) const;
    void write_all_strided(const math::Vector<Real, 3>& xi,
                           Real* SVMP_RESTRICT values_out,
                           Real* SVMP_RESTRICT gradients_out,
                           Real* SVMP_RESTRICT hessians_out,
                           std::size_t output_stride,
                           std::size_t q) const;
    void write_tensor_product_batch_strided(
        const std::vector<math::Vector<Real, 3>>& points,
        std::size_t output_stride,
        Real* SVMP_RESTRICT values_out,
        Real* SVMP_RESTRICT gradients_out,
        Real* SVMP_RESTRICT hessians_out) const;
    void write_wedge_batch_strided(
        const std::vector<math::Vector<Real, 3>>& points,
        std::size_t output_stride,
        Real* SVMP_RESTRICT values_out,
        Real* SVMP_RESTRICT gradients_out,
        Real* SVMP_RESTRICT hessians_out) const;
    void write_modal_batch_strided(
        const std::vector<math::Vector<Real, 3>>& points,
        std::size_t output_stride,
        Real* SVMP_RESTRICT values_out,
        Real* SVMP_RESTRICT gradients_out,
        Real* SVMP_RESTRICT hessians_out) const;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_SPECTRALBASIS_H
