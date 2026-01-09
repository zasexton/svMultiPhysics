#ifndef SVMP_PHYSICS_MATERIALS_COMMON_TENSOR_OPS_H
#define SVMP_PHYSICS_MATERIALS_COMMON_TENSOR_OPS_H

/**
 * @file TensorOps.h
 * @brief Small tensor helpers for Physics constitutive models (Real + Forms Dual)
 *
 * This header provides minimal scalar and small-matrix utilities needed by
 * constitutive models implemented in Physics/. It intentionally avoids
 * introducing a new tensor library dependency.
 */

#include "FE/Core/Types.h"
#include "FE/Forms/Dual.h"
#include "FE/Forms/Value.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace svmp {
namespace Physics {
namespace materials {
namespace common {

template <class Scalar, class Workspace>
[[nodiscard]] inline Scalar constant(FE::Real value, Workspace& ws)
{
    if constexpr (std::is_same_v<Scalar, FE::Real>) {
        (void)ws;
        return value;
    } else if constexpr (std::is_same_v<Scalar, FE::forms::Dual>) {
        return FE::forms::makeDualConstant(value, ws.alloc());
    } else {
        static_assert(std::is_same_v<Scalar, FE::Real> || std::is_same_v<Scalar, FE::forms::Dual>,
                      "materials::common::constant: unsupported Scalar type");
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline Scalar add(const Scalar& a, const Scalar& b, Workspace& ws)
{
    if constexpr (std::is_same_v<Scalar, FE::Real>) {
        (void)ws;
        return a + b;
    } else {
        return FE::forms::add(a, b, FE::forms::makeDualConstant(0.0, ws.alloc()));
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline Scalar sub(const Scalar& a, const Scalar& b, Workspace& ws)
{
    if constexpr (std::is_same_v<Scalar, FE::Real>) {
        (void)ws;
        return a - b;
    } else {
        return FE::forms::sub(a, b, FE::forms::makeDualConstant(0.0, ws.alloc()));
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline Scalar mul(const Scalar& a, const Scalar& b, Workspace& ws)
{
    if constexpr (std::is_same_v<Scalar, FE::Real>) {
        (void)ws;
        return a * b;
    } else {
        return FE::forms::mul(a, b, FE::forms::makeDualConstant(0.0, ws.alloc()));
    }
}

template <class Scalar, class Workspace,
          std::enable_if_t<!std::is_same_v<Scalar, FE::Real>, int> = 0>
[[nodiscard]] inline Scalar mul(const Scalar& a, FE::Real b, Workspace& ws)
{
    return FE::forms::mul(a, b, FE::forms::makeDualConstant(0.0, ws.alloc()));
}

template <class Scalar, class Workspace>
[[nodiscard]] inline Scalar div(const Scalar& a, const Scalar& b, Workspace& ws)
{
    if constexpr (std::is_same_v<Scalar, FE::Real>) {
        (void)ws;
        return a / b;
    } else {
        return FE::forms::div(a, b, FE::forms::makeDualConstant(0.0, ws.alloc()));
    }
}

template <class Scalar, class Workspace,
          std::enable_if_t<!std::is_same_v<Scalar, FE::Real>, int> = 0>
[[nodiscard]] inline Scalar div(const Scalar& a, FE::Real b, Workspace& ws)
{
    return FE::forms::div(a, b, FE::forms::makeDualConstant(0.0, ws.alloc()));
}

template <class Scalar, class Workspace>
[[nodiscard]] inline Scalar neg(const Scalar& a, Workspace& ws)
{
    if constexpr (std::is_same_v<Scalar, FE::Real>) {
        (void)ws;
        return -a;
    } else {
        return FE::forms::neg(a, FE::forms::makeDualConstant(0.0, ws.alloc()));
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline Scalar sqrt(const Scalar& a, Workspace& ws)
{
    if constexpr (std::is_same_v<Scalar, FE::Real>) {
        (void)ws;
        return std::sqrt(a);
    } else {
        return FE::forms::sqrt(a, FE::forms::makeDualConstant(0.0, ws.alloc()));
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline Scalar log(const Scalar& a, Workspace& ws)
{
    if constexpr (std::is_same_v<Scalar, FE::Real>) {
        (void)ws;
        return std::log(a);
    } else {
        return FE::forms::log(a, FE::forms::makeDualConstant(0.0, ws.alloc()));
    }
}

template <class Scalar, class Workspace>
[[nodiscard]] inline Scalar pow(const Scalar& a, FE::Real b, Workspace& ws)
{
    if constexpr (std::is_same_v<Scalar, FE::Real>) {
        (void)ws;
        return std::pow(a, b);
    } else {
        return FE::forms::pow(a, b, FE::forms::makeDualConstant(0.0, ws.alloc()));
    }
}

template <class Scalar>
[[nodiscard]] inline FE::Real scalarValue(const Scalar& a)
{
    if constexpr (std::is_same_v<Scalar, FE::Real>) {
        return a;
    } else {
        return a.value;
    }
}

template <class Scalar>
[[nodiscard]] inline bool isMatrixLike(typename FE::forms::Value<Scalar>::Kind kind)
{
    using Kind = typename FE::forms::Value<Scalar>::Kind;
    return kind == Kind::Matrix || kind == Kind::SymmetricMatrix || kind == Kind::SkewMatrix;
}

template <class Scalar, class Workspace>
[[nodiscard]] inline Scalar det(const FE::forms::Value<Scalar>& A, int dim, Workspace& ws)
{
    if (!isMatrixLike<Scalar>(A.kind)) {
        throw std::invalid_argument("materials::common::det: expected matrix-like Value");
    }

    if (dim == 2) {
        const auto a00 = A.matrixAt(0, 0);
        const auto a01 = A.matrixAt(0, 1);
        const auto a10 = A.matrixAt(1, 0);
        const auto a11 = A.matrixAt(1, 1);
        return sub(mul(a00, a11, ws), mul(a01, a10, ws), ws);
    }
    if (dim == 3) {
        const auto a00 = A.matrixAt(0, 0);
        const auto a01 = A.matrixAt(0, 1);
        const auto a02 = A.matrixAt(0, 2);
        const auto a10 = A.matrixAt(1, 0);
        const auto a11 = A.matrixAt(1, 1);
        const auto a12 = A.matrixAt(1, 2);
        const auto a20 = A.matrixAt(2, 0);
        const auto a21 = A.matrixAt(2, 1);
        const auto a22 = A.matrixAt(2, 2);

        const auto m11 = sub(mul(a11, a22, ws), mul(a12, a21, ws), ws);
        const auto m12 = sub(mul(a10, a22, ws), mul(a12, a20, ws), ws);
        const auto m13 = sub(mul(a10, a21, ws), mul(a11, a20, ws), ws);

        const auto t0 = mul(a00, m11, ws);
        const auto t1 = mul(a01, m12, ws);
        const auto t2 = mul(a02, m13, ws);

        return add(sub(t0, t1, ws), t2, ws);
    }

    throw std::invalid_argument("materials::common::det: only dim=2 or dim=3 supported");
}

template <class Scalar, class Workspace>
[[nodiscard]] inline FE::forms::Value<Scalar> inverse(const FE::forms::Value<Scalar>& A, int dim, Workspace& ws)
{
    if (!isMatrixLike<Scalar>(A.kind)) {
        throw std::invalid_argument("materials::common::inverse: expected matrix-like Value");
    }

    FE::forms::Value<Scalar> inv;
    inv.kind = FE::forms::Value<Scalar>::Kind::Matrix;
    inv.resizeMatrix(static_cast<std::size_t>(dim), static_cast<std::size_t>(dim));

    if (dim == 2) {
        const auto a00 = A.matrixAt(0, 0);
        const auto a01 = A.matrixAt(0, 1);
        const auto a10 = A.matrixAt(1, 0);
        const auto a11 = A.matrixAt(1, 1);

        const auto detA = sub(mul(a00, a11, ws), mul(a01, a10, ws), ws);
        if (scalarValue(detA) == 0.0) {
            throw std::invalid_argument("materials::common::inverse: singular 2x2 matrix");
        }
        const auto inv_det = div(constant<Scalar>(1.0, ws), detA, ws);

        inv.matrixAt(0, 0) = mul(a11, inv_det, ws);
        inv.matrixAt(0, 1) = mul(neg(a01, ws), inv_det, ws);
        inv.matrixAt(1, 0) = mul(neg(a10, ws), inv_det, ws);
        inv.matrixAt(1, 1) = mul(a00, inv_det, ws);
        return inv;
    }

    if (dim == 3) {
        const auto a00 = A.matrixAt(0, 0);
        const auto a01 = A.matrixAt(0, 1);
        const auto a02 = A.matrixAt(0, 2);
        const auto a10 = A.matrixAt(1, 0);
        const auto a11 = A.matrixAt(1, 1);
        const auto a12 = A.matrixAt(1, 2);
        const auto a20 = A.matrixAt(2, 0);
        const auto a21 = A.matrixAt(2, 1);
        const auto a22 = A.matrixAt(2, 2);

        // Cofactors (adjugate transpose).
        const auto c00 = sub(mul(a11, a22, ws), mul(a12, a21, ws), ws);
        const auto c01 = neg(sub(mul(a10, a22, ws), mul(a12, a20, ws), ws), ws);
        const auto c02 = sub(mul(a10, a21, ws), mul(a11, a20, ws), ws);

        const auto c10 = neg(sub(mul(a01, a22, ws), mul(a02, a21, ws), ws), ws);
        const auto c11 = sub(mul(a00, a22, ws), mul(a02, a20, ws), ws);
        const auto c12 = neg(sub(mul(a00, a21, ws), mul(a01, a20, ws), ws), ws);

        const auto c20 = sub(mul(a01, a12, ws), mul(a02, a11, ws), ws);
        const auto c21 = neg(sub(mul(a00, a12, ws), mul(a02, a10, ws), ws), ws);
        const auto c22 = sub(mul(a00, a11, ws), mul(a01, a10, ws), ws);

        const auto detA = add(add(mul(a00, c00, ws), mul(a01, c01, ws), ws), mul(a02, c02, ws), ws);
        if (scalarValue(detA) == 0.0) {
            throw std::invalid_argument("materials::common::inverse: singular 3x3 matrix");
        }
        const auto inv_det = div(constant<Scalar>(1.0, ws), detA, ws);

        // inv(A) = adj(A) / det(A) where adj(A) = C^T.
        inv.matrixAt(0, 0) = mul(c00, inv_det, ws);
        inv.matrixAt(0, 1) = mul(c10, inv_det, ws);
        inv.matrixAt(0, 2) = mul(c20, inv_det, ws);

        inv.matrixAt(1, 0) = mul(c01, inv_det, ws);
        inv.matrixAt(1, 1) = mul(c11, inv_det, ws);
        inv.matrixAt(1, 2) = mul(c21, inv_det, ws);

        inv.matrixAt(2, 0) = mul(c02, inv_det, ws);
        inv.matrixAt(2, 1) = mul(c12, inv_det, ws);
        inv.matrixAt(2, 2) = mul(c22, inv_det, ws);

        return inv;
    }

    throw std::invalid_argument("materials::common::inverse: only dim=2 or dim=3 supported");
}

} // namespace common
} // namespace materials
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_MATERIALS_COMMON_TENSOR_OPS_H
