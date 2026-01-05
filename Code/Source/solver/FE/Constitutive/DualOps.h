#ifndef SVMP_FE_CONSTITUTIVE_DUAL_OPS_H
#define SVMP_FE_CONSTITUTIVE_DUAL_OPS_H

#include "Forms/Dual.h"

namespace svmp {
namespace FE {
namespace constitutive {

/**
 * @brief Convenience allocator-backed scalar operations for forms::Dual
 *
 * This is intended to reduce boilerplate in constitutive models by hiding the
 * repeated pattern:
 * `forms::foo(a, b, forms::makeDualConstant(0.0, ws.alloc()))`.
 */
class DualOps {
public:
    explicit DualOps(forms::DualWorkspace& ws) : ws_(&ws) {}

    [[nodiscard]] forms::Dual constant(Real value) const { return forms::makeDualConstant(value, ws_->alloc()); }

    [[nodiscard]] forms::Dual copy(const forms::Dual& a) const { return forms::copy(a, tmp()); }

    [[nodiscard]] forms::Dual add(const forms::Dual& a, const forms::Dual& b) const { return forms::add(a, b, tmp()); }

    [[nodiscard]] forms::Dual add(const forms::Dual& a, Real b) const
    {
        auto out = tmp();
        out.value = a.value + b;
        const std::size_t n = out.deriv.size();
        for (std::size_t k = 0; k < n; ++k) out.deriv[k] = a.deriv[k];
        return out;
    }

    [[nodiscard]] forms::Dual add(Real a, const forms::Dual& b) const { return add(b, a); }

    [[nodiscard]] forms::Dual sub(const forms::Dual& a, const forms::Dual& b) const { return forms::sub(a, b, tmp()); }

    [[nodiscard]] forms::Dual sub(const forms::Dual& a, Real b) const
    {
        auto out = tmp();
        out.value = a.value - b;
        const std::size_t n = out.deriv.size();
        for (std::size_t k = 0; k < n; ++k) out.deriv[k] = a.deriv[k];
        return out;
    }

    [[nodiscard]] forms::Dual sub(Real a, const forms::Dual& b) const
    {
        auto out = tmp();
        out.value = a - b.value;
        const std::size_t n = out.deriv.size();
        for (std::size_t k = 0; k < n; ++k) out.deriv[k] = -b.deriv[k];
        return out;
    }

    [[nodiscard]] forms::Dual mul(const forms::Dual& a, const forms::Dual& b) const { return forms::mul(a, b, tmp()); }
    [[nodiscard]] forms::Dual mul(const forms::Dual& a, Real b) const { return forms::mul(a, b, tmp()); }
    [[nodiscard]] forms::Dual mul(Real a, const forms::Dual& b) const { return forms::mul(a, b, tmp()); }

    [[nodiscard]] forms::Dual div(const forms::Dual& a, const forms::Dual& b) const { return forms::div(a, b, tmp()); }
    [[nodiscard]] forms::Dual div(const forms::Dual& a, Real b) const { return forms::div(a, b, tmp()); }
    [[nodiscard]] forms::Dual div(Real a, const forms::Dual& b) const { return forms::div(a, b, tmp()); }

    [[nodiscard]] forms::Dual neg(const forms::Dual& a) const { return forms::neg(a, tmp()); }
    [[nodiscard]] forms::Dual abs(const forms::Dual& a) const { return forms::abs(a, tmp()); }
    [[nodiscard]] forms::Dual sign(const forms::Dual& a) const { return forms::sign(a, tmp()); }

    [[nodiscard]] forms::Dual sqrt(const forms::Dual& a) const { return forms::sqrt(a, tmp()); }
    [[nodiscard]] forms::Dual exp(const forms::Dual& a) const { return forms::exp(a, tmp()); }
    [[nodiscard]] forms::Dual log(const forms::Dual& a) const { return forms::log(a, tmp()); }

    [[nodiscard]] forms::Dual pow(const forms::Dual& a, const forms::Dual& b) const { return forms::pow(a, b, tmp()); }
    [[nodiscard]] forms::Dual pow(const forms::Dual& a, Real b) const { return forms::pow(a, b, tmp()); }

private:
    [[nodiscard]] forms::Dual tmp() const { return forms::makeDualConstant(0.0, ws_->alloc()); }

    forms::DualWorkspace* ws_{nullptr};
};

} // namespace constitutive
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTITUTIVE_DUAL_OPS_H

