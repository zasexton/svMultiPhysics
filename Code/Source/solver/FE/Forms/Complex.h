#ifndef SVMP_FE_FORMS_COMPLEX_H
#define SVMP_FE_FORMS_COMPLEX_H

/**
 * @file Complex.h
 * @brief Minimal complex-valued vocabulary for FE/Forms via real/imag splitting
 *
 * FE/Forms is real-valued at the kernel level. This header provides a small
 * complex-valued vocabulary implemented as `{re, im}` pairs of real `FormExpr`.
 *
 * To assemble complex-valued systems using real-valued backends, use the
 * standard 2x2 real block lifting:
 *   A = Re(A) + i Im(A)
 *   [Re(A) -Im(A); Im(A) Re(A)]
 *
 * This remains purely declarative: Forms does not introduce complex linear
 * algebra backends, and it does not pick time-integration schemes.
 */

#include "Forms/BlockForm.h"

#include <complex>
#include <utility>

namespace svmp {
namespace FE {
namespace forms {

struct ComplexScalar {
    FormExpr re{};
    FormExpr im{};

    [[nodiscard]] bool isValid() const noexcept { return re.isValid() && im.isValid(); }

    static ComplexScalar constant(Real re_val, Real im_val = 0.0)
    {
        return ComplexScalar{FormExpr::constant(re_val), FormExpr::constant(im_val)};
    }

    static ComplexScalar constant(std::complex<Real> z)
    {
        return constant(z.real(), z.imag());
    }
};

inline ComplexScalar I()
{
    return ComplexScalar::constant(0.0, 1.0);
}

inline ComplexScalar conj(const ComplexScalar& z)
{
    return ComplexScalar{z.re, -z.im};
}

inline ComplexScalar operator+(const ComplexScalar& a, const ComplexScalar& b)
{
    return ComplexScalar{a.re + b.re, a.im + b.im};
}

inline ComplexScalar operator-(const ComplexScalar& a, const ComplexScalar& b)
{
    return ComplexScalar{a.re - b.re, a.im - b.im};
}

inline ComplexScalar operator-(const ComplexScalar& a)
{
    return ComplexScalar{-a.re, -a.im};
}

inline ComplexScalar operator*(const ComplexScalar& a, const ComplexScalar& b)
{
    // (a + i b)(c + i d) = (ac - bd) + i(ad + bc)
    return ComplexScalar{
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re,
    };
}

inline ComplexScalar operator*(const ComplexScalar& a, const FormExpr& b)
{
    return ComplexScalar{a.re * b, a.im * b};
}

inline ComplexScalar operator*(const FormExpr& a, const ComplexScalar& b)
{
    return b * a;
}

inline ComplexScalar operator*(Real a, const ComplexScalar& b)
{
    return ComplexScalar{b.re * a, b.im * a};
}

inline ComplexScalar operator*(const ComplexScalar& a, Real b)
{
    return b * a;
}

struct ComplexBilinearForm {
    FormExpr re{};
    FormExpr im{};

    [[nodiscard]] bool isValid() const noexcept { return re.isValid() && im.isValid(); }
};

struct ComplexLinearForm {
    FormExpr re{};
    FormExpr im{};

    [[nodiscard]] bool isValid() const noexcept { return re.isValid() && im.isValid(); }
};

inline BlockBilinearForm toRealBlock2x2(const ComplexBilinearForm& a)
{
    BlockBilinearForm blocks(2, 2);
    blocks.setBlock(0, 0, a.re);
    blocks.setBlock(0, 1, -a.im);
    blocks.setBlock(1, 0, a.im);
    blocks.setBlock(1, 1, a.re);
    return blocks;
}

inline BlockLinearForm toRealBlock2x1(const ComplexLinearForm& f)
{
    BlockLinearForm blocks(2);
    blocks.setBlock(0, f.re);
    blocks.setBlock(1, f.im);
    return blocks;
}

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_COMPLEX_H

