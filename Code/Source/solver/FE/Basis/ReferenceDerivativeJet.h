/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_REFERENCEDERIVATIVEJET_H
#define SVMP_FE_BASIS_REFERENCEDERIVATIVEJET_H

#include "Core/Types.h"
#include "Math/Matrix.h"
#include "Math/Vector.h"

namespace svmp {
namespace FE {
namespace basis {
namespace detail {

// Basis-local forward-mode AD for reference-coordinate modal derivative
// construction. Promote this out of Basis only if Assembly/Operator code needs
// the same value/gradient/Hessian contract.
struct Jet3 {
    Real value{Real(0)};
    math::Vector<Real, 3> gradient{};
    math::Matrix<Real, 3, 3> hessian{};
};

inline constexpr Jet3 constant_jet(Real value) {
    Jet3 jet;
    jet.value = value;
    return jet;
}

inline constexpr Jet3 variable_jet(int axis, Real value) {
    Jet3 jet = constant_jet(value);
    jet.gradient[static_cast<std::size_t>(axis)] = Real(1);
    return jet;
}

inline constexpr Jet3 operator+(const Jet3& lhs, const Jet3& rhs) {
    Jet3 out;
    out.value = lhs.value + rhs.value;
    for (int i = 0; i < 3; ++i) {
        const std::size_t si = static_cast<std::size_t>(i);
        out.gradient[si] = lhs.gradient[si] + rhs.gradient[si];
        for (int j = 0; j < 3; ++j) {
            const std::size_t sj = static_cast<std::size_t>(j);
            out.hessian(si, sj) = lhs.hessian(si, sj) + rhs.hessian(si, sj);
        }
    }
    return out;
}

inline constexpr Jet3 operator-(const Jet3& lhs, const Jet3& rhs) {
    Jet3 out;
    out.value = lhs.value - rhs.value;
    for (int i = 0; i < 3; ++i) {
        const std::size_t si = static_cast<std::size_t>(i);
        out.gradient[si] = lhs.gradient[si] - rhs.gradient[si];
        for (int j = 0; j < 3; ++j) {
            const std::size_t sj = static_cast<std::size_t>(j);
            out.hessian(si, sj) = lhs.hessian(si, sj) - rhs.hessian(si, sj);
        }
    }
    return out;
}

inline constexpr Jet3 operator-(const Jet3& jet) {
    Jet3 out;
    out.value = -jet.value;
    for (int i = 0; i < 3; ++i) {
        const std::size_t si = static_cast<std::size_t>(i);
        out.gradient[si] = -jet.gradient[si];
        for (int j = 0; j < 3; ++j) {
            const std::size_t sj = static_cast<std::size_t>(j);
            out.hessian(si, sj) = -jet.hessian(si, sj);
        }
    }
    return out;
}

inline constexpr Jet3 operator*(const Jet3& lhs, const Jet3& rhs) {
    Jet3 out;
    out.value = lhs.value * rhs.value;
    for (int i = 0; i < 3; ++i) {
        const std::size_t si = static_cast<std::size_t>(i);
        out.gradient[si] = lhs.gradient[si] * rhs.value + rhs.gradient[si] * lhs.value;
        for (int j = 0; j < 3; ++j) {
            const std::size_t sj = static_cast<std::size_t>(j);
            out.hessian(si, sj) =
                lhs.hessian(si, sj) * rhs.value +
                rhs.hessian(si, sj) * lhs.value +
                lhs.gradient[si] * rhs.gradient[sj] +
                rhs.gradient[si] * lhs.gradient[sj];
        }
    }
    return out;
}

inline constexpr Jet3 operator*(const Jet3& jet, Real scalar) {
    Jet3 out;
    out.value = jet.value * scalar;
    for (int i = 0; i < 3; ++i) {
        const std::size_t si = static_cast<std::size_t>(i);
        out.gradient[si] = jet.gradient[si] * scalar;
        for (int j = 0; j < 3; ++j) {
            const std::size_t sj = static_cast<std::size_t>(j);
            out.hessian(si, sj) = jet.hessian(si, sj) * scalar;
        }
    }
    return out;
}

inline constexpr Jet3 operator*(Real scalar, const Jet3& jet) {
    return jet * scalar;
}

inline constexpr Jet3 operator+(const Jet3& jet, Real scalar) {
    Jet3 out = jet;
    out.value += scalar;
    return out;
}

inline constexpr Jet3 operator+(Real scalar, const Jet3& jet) {
    return jet + scalar;
}

inline constexpr Jet3 operator-(const Jet3& jet, Real scalar) {
    Jet3 out = jet;
    out.value -= scalar;
    return out;
}

inline constexpr Jet3 operator-(Real scalar, const Jet3& jet) {
    return constant_jet(scalar) - jet;
}

inline constexpr Jet3 compose_univariate(const Jet3& jet,
                                         Real value,
                                         Real first_derivative,
                                         Real second_derivative) {
    Jet3 out;
    out.value = value;
    for (int i = 0; i < 3; ++i) {
        const std::size_t si = static_cast<std::size_t>(i);
        out.gradient[si] = first_derivative * jet.gradient[si];
        for (int j = 0; j < 3; ++j) {
            const std::size_t sj = static_cast<std::size_t>(j);
            out.hessian(si, sj) =
                second_derivative * jet.gradient[si] * jet.gradient[sj] +
                first_derivative * jet.hessian(si, sj);
        }
    }
    return out;
}

inline constexpr Jet3 reciprocal(const Jet3& jet) {
    const Real inv = Real(1) / jet.value;
    return compose_univariate(jet, inv, -inv * inv, Real(2) * inv * inv * inv);
}

inline constexpr Jet3 operator/(const Jet3& lhs, const Jet3& rhs) {
    return lhs * reciprocal(rhs);
}

inline constexpr Jet3 operator/(const Jet3& jet, Real scalar) {
    return jet * (Real(1) / scalar);
}

inline constexpr Jet3 operator/(Real scalar, const Jet3& jet) {
    return constant_jet(scalar) / jet;
}

inline constexpr Jet3 pow_int(const Jet3& jet, int exponent) {
    if (exponent <= 0) {
        return constant_jet(Real(1));
    }
    if (exponent == 1) {
        return jet;
    }
    if (exponent == 2) {
        return jet * jet;
    }

    Jet3 result = constant_jet(Real(1));
    Jet3 base = jet;
    int power = exponent;
    while (power > 0) {
        if ((power & 1) != 0) {
            result = result * base;
        }
        power >>= 1;
        if (power > 0) {
            base = base * base;
        }
    }
    return result;
}

} // namespace detail
} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_REFERENCEDERIVATIVEJET_H
