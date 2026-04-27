/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "PushForward.h"

#include "Geometry/FrameAwareTransform.h"

#include <cmath>

namespace svmp {
namespace FE {
namespace geometry {

namespace {

[[nodiscard]] bool hessian_is_zero(const GeometryMapping::MappingHessian& H, int dim)
{
    constexpr Real tol = Real(1e-14);
    for (std::size_t r = 0; r < 3; ++r) {
        for (int a = 0; a < dim; ++a) {
            for (int b = 0; b < dim; ++b) {
                if (std::abs(H[r](static_cast<std::size_t>(a),
                                  static_cast<std::size_t>(b))) > tol) {
                    return false;
                }
            }
        }
    }
    return true;
}

[[nodiscard]] bool piola_gradient_mapping_is_affine(
    const GeometryMapping::MappingHessian& H,
    int dim,
    bool affine_mapping)
{
    return affine_mapping || hessian_is_zero(H, dim);
}

} // namespace

math::Vector<Real, 3> PushForward::gradient(const GeometryMapping& mapping,
                                            const math::Vector<Real, 3>& grad_ref,
                                            const math::Vector<Real, 3>& xi) {
    return mapping.transform_gradient(grad_ref, xi);
}

math::Vector<Real, 3> PushForward::gradient(const GeometryMapping& mapping,
                                            const math::Vector<Real, 3>& grad_ref,
                                            const math::Matrix<Real, 3, 3>& jacobian_inverse) {
    return mapping.transform_gradient(grad_ref, jacobian_inverse);
}

math::Vector<Real, 3> PushForward::hdiv_vector(const GeometryMapping& mapping,
                                               const math::Vector<Real, 3>& v_ref,
                                               const math::Vector<Real, 3>& xi) {
    auto J = mapping.jacobian(xi);
    const Real det = J.determinant();
    return hdiv_vector(mapping, v_ref, J, det);
}

math::Vector<Real, 3> PushForward::hdiv_vector(const GeometryMapping& mapping,
                                               const math::Vector<Real, 3>& v_ref,
                                               const math::Matrix<Real, 3, 3>& jacobian,
                                               Real det_jacobian) {
    DeformationFrame frame;
    frame.J = jacobian;
    frame.detJ = det_jacobian;
    frame.dim = mapping.dimension();
    return FrameAwareTransform::hdivPushForward(v_ref, frame);
}

math::Vector<Real, 3> PushForward::hcurl_vector(const GeometryMapping& mapping,
                                               const math::Vector<Real, 3>& v_ref,
                                               const math::Vector<Real, 3>& xi) {
    return hcurl_vector(mapping, v_ref, mapping.jacobian_inverse(xi));
}

math::Vector<Real, 3> PushForward::hcurl_vector(const GeometryMapping& mapping,
                                                const math::Vector<Real, 3>& v_ref,
                                                const math::Matrix<Real, 3, 3>& jacobian_inverse) {
    DeformationFrame frame;
    frame.Jinv = jacobian_inverse;
    frame.dim = mapping.dimension();
    return FrameAwareTransform::hcurlPushForward(v_ref, frame);
}

math::Matrix<Real, 3, 3> PushForward::vector_jacobian(
    const GeometryMapping& mapping,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Vector<Real, 3>& xi)
{
    return vector_jacobian(mapping, jac_ref, mapping.jacobian_inverse(xi));
}

math::Matrix<Real, 3, 3> PushForward::vector_jacobian(
    const GeometryMapping& mapping,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Matrix<Real, 3, 3>& jacobian_inverse)
{
    const std::size_t dim = static_cast<std::size_t>(mapping.dimension());
    math::Matrix<Real, 3, 3> out{};
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < dim; ++c) {
            Real sum = Real(0);
            for (std::size_t a = 0; a < dim; ++a) {
                sum += jac_ref(r, a) * jacobian_inverse(a, c);
            }
            out(r, c) = sum;
        }
    }
    return out;
}

PushForward::PiolaVectorGradientGeometryData
PushForward::piola_vector_gradient_geometry_data(const GeometryMapping& mapping,
                                                 const math::Vector<Real, 3>& xi)
{
    const auto J = mapping.jacobian(xi);
    return piola_vector_gradient_geometry_data(mapping.dimension(),
                                               J,
                                               J.inverse(),
                                               J.determinant(),
                                               mapping.mapping_hessian(xi),
                                               mapping.isAffine());
}

PushForward::PiolaVectorGradientGeometryData
PushForward::piola_vector_gradient_geometry_data(
    int dimension,
    const math::Matrix<Real, 3, 3>& jacobian,
    const math::Matrix<Real, 3, 3>& jacobian_inverse,
    Real det_jacobian,
    const GeometryMapping::MappingHessian& mapping_hessian,
    bool affine_mapping)
{
    FE_THROW_IF(dimension < 1 || dimension > 3, FEException,
                "curved Piola vector-gradient geometry requires mapping dimension 1, 2, or 3");

    PiolaVectorGradientGeometryData data;
    data.jacobian = jacobian;
    data.jacobian_inverse = jacobian_inverse;
    data.mapping_hessian = mapping_hessian;
    data.determinant = det_jacobian;
    data.dimension = dimension;
    data.affine = piola_gradient_mapping_is_affine(mapping_hessian, dimension, affine_mapping);

    if (data.affine) {
        return data;
    }

    FE_THROW_IF(dimension != 3, FEException,
                "curved Piola vector-gradient derivatives are enabled for non-affine 3D volume mappings; "
                "lower-dimensional curved mappings require surface/curve frame derivatives");

    for (std::size_t c = 0; c < 3; ++c) {
        auto& dJdx = data.jacobian_derivatives_x[c];
        for (std::size_t r = 0; r < 3; ++r) {
            for (std::size_t a = 0; a < 3; ++a) {
                Real sum = Real(0);
                for (std::size_t k = 0; k < 3; ++k) {
                    sum += mapping_hessian[r](a, k) * jacobian_inverse(k, c);
                }
                dJdx(r, a) = sum;
            }
        }

        Real tr_Jinv_dJ = Real(0);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t a = 0; a < 3; ++a) {
                tr_Jinv_dJ += jacobian_inverse(i, a) * dJdx(a, i);
            }
        }
        data.determinant_derivatives_x[c] = det_jacobian * tr_Jinv_dJ;

        auto& dJinv_dx = data.inverse_jacobian_derivatives_x[c];
        auto& dJinvT_dx = data.inverse_transpose_jacobian_derivatives_x[c];
        for (std::size_t a = 0; a < 3; ++a) {
            for (std::size_t r = 0; r < 3; ++r) {
                Real sum = Real(0);
                for (std::size_t m = 0; m < 3; ++m) {
                    for (std::size_t n = 0; n < 3; ++n) {
                        sum += jacobian_inverse(a, m) * dJdx(m, n) * jacobian_inverse(n, r);
                    }
                }
                dJinv_dx(a, r) = -sum;
                dJinvT_dx(r, a) = -sum;
            }
        }
    }

    return data;
}

math::Matrix<Real, 3, 3> PushForward::hdiv_vector_jacobian(
    const GeometryMapping& mapping,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Vector<Real, 3>& xi)
{
    return hdiv_vector_jacobian(mapping,
                                jac_ref,
                                mapping.jacobian(xi),
                                mapping.jacobian_inverse(xi),
                                mapping.jacobian_determinant(xi));
}

math::Matrix<Real, 3, 3> PushForward::hdiv_vector_jacobian(
    const GeometryMapping& mapping,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Matrix<Real, 3, 3>& jacobian,
    const math::Matrix<Real, 3, 3>& jacobian_inverse,
    Real det_jacobian)
{
    if (!mapping.isAffine()) {
        FE_THROW(FEException,
                 "H(div) vector-basis Jacobians require an affine geometry mapping; "
                 "non-affine curved Piola gradients also require the reference vector value; "
                 "use hdiv_vector_jacobian(mapping, v_ref, jac_ref, xi) or reusable Piola geometry data");
    }
    return (jacobian * jac_ref * jacobian_inverse) * (Real(1) / det_jacobian);
}

math::Matrix<Real, 3, 3> PushForward::hdiv_vector_jacobian(
    const GeometryMapping& mapping,
    const math::Vector<Real, 3>& v_ref,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Vector<Real, 3>& xi)
{
    return hdiv_vector_jacobian(v_ref,
                                jac_ref,
                                piola_vector_gradient_geometry_data(mapping, xi));
}

math::Matrix<Real, 3, 3> PushForward::hdiv_vector_jacobian(
    const math::Vector<Real, 3>& v_ref,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const PiolaVectorGradientGeometryData& geometry)
{
    if (geometry.affine) {
        return (geometry.jacobian * jac_ref * geometry.jacobian_inverse) *
               (Real(1) / geometry.determinant);
    }

    math::Matrix<Real, 3, 3> out{};
    math::Vector<Real, 3> Jv{};
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t a = 0; a < 3; ++a) {
            Jv[r] += geometry.jacobian(r, a) * v_ref[a];
        }
    }

    const Real inv_det = Real(1) / geometry.determinant;
    const Real inv_det_sq = inv_det * inv_det;
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            Real geom_value_term = Real(0);
            Real reference_gradient_term = Real(0);
            for (std::size_t a = 0; a < 3; ++a) {
                geom_value_term += geometry.jacobian_derivatives_x[c](r, a) * v_ref[a];
                for (std::size_t b = 0; b < 3; ++b) {
                    reference_gradient_term +=
                        geometry.jacobian(r, a) * jac_ref(a, b) * geometry.jacobian_inverse(b, c);
                }
            }
            out(r, c) = inv_det * (geom_value_term + reference_gradient_term) -
                        Jv[r] * geometry.determinant_derivatives_x[c] * inv_det_sq;
        }
    }
    return out;
}

math::Matrix<Real, 3, 3> PushForward::hcurl_vector_jacobian(
    const GeometryMapping& mapping,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Vector<Real, 3>& xi)
{
    return hcurl_vector_jacobian(mapping, jac_ref, mapping.jacobian_inverse(xi));
}

math::Matrix<Real, 3, 3> PushForward::hcurl_vector_jacobian(
    const GeometryMapping& mapping,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Matrix<Real, 3, 3>& jacobian_inverse)
{
    if (!mapping.isAffine()) {
        FE_THROW(FEException,
                 "H(curl) vector-basis Jacobians require an affine geometry mapping; "
                 "non-affine curved Piola gradients also require the reference vector value; "
                 "use hcurl_vector_jacobian(mapping, v_ref, jac_ref, xi) or reusable Piola geometry data");
    }
    return jacobian_inverse.transpose() * jac_ref * jacobian_inverse;
}

math::Matrix<Real, 3, 3> PushForward::hcurl_vector_jacobian(
    const GeometryMapping& mapping,
    const math::Vector<Real, 3>& v_ref,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const math::Vector<Real, 3>& xi)
{
    return hcurl_vector_jacobian(v_ref,
                                 jac_ref,
                                 piola_vector_gradient_geometry_data(mapping, xi));
}

math::Matrix<Real, 3, 3> PushForward::hcurl_vector_jacobian(
    const math::Vector<Real, 3>& v_ref,
    const math::Matrix<Real, 3, 3>& jac_ref,
    const PiolaVectorGradientGeometryData& geometry)
{
    if (geometry.affine) {
        return geometry.jacobian_inverse.transpose() * jac_ref * geometry.jacobian_inverse;
    }

    math::Matrix<Real, 3, 3> out{};
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            Real geom_value_term = Real(0);
            Real reference_gradient_term = Real(0);
            for (std::size_t a = 0; a < 3; ++a) {
                geom_value_term += geometry.inverse_jacobian_derivatives_x[c](a, r) * v_ref[a];
                for (std::size_t b = 0; b < 3; ++b) {
                    reference_gradient_term +=
                        geometry.jacobian_inverse(a, r) * jac_ref(a, b) *
                        geometry.jacobian_inverse(b, c);
                }
            }
            out(r, c) = geom_value_term + reference_gradient_term;
        }
    }
    return out;
}

} // namespace geometry
} // namespace FE
} // namespace svmp
