#ifndef SVMP_FE_FORMS_DUAL_H
#define SVMP_FE_FORMS_DUAL_H

/**
 * @file Dual.h
 * @brief Minimal forward-mode AD scalar used by FE/Forms for Jacobian assembly
 *
 * This is intentionally lightweight and scoped to per-element Jacobian assembly.
 * It is not intended to replace more complete AD libraries (e.g., Sacado).
 */

#include "Core/Types.h"
#include "Core/AlignedAllocator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>
#include <span>

namespace svmp {
namespace FE {
namespace forms {

inline constexpr std::size_t kDualDerivativeAlignmentBytes = 64u;

#if defined(__clang__)
#define SVMP_DUAL_VECTORIZE _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
#define SVMP_DUAL_VECTORIZE _Pragma("GCC ivdep")
#else
#define SVMP_DUAL_VECTORIZE
#endif

/**
 * @brief Forward-mode dual number with externally managed derivative storage
 *
 * The derivative array is a non-owning span into a caller-provided workspace.
 */
struct Dual {
    Real value{0.0};
    std::span<Real> deriv{};
};

/**
 * @brief Simple per-thread scratch allocator for Dual derivative storage
 *
 * Allocations are linear (stack-like) until reset(). Intended for per-element
 * evaluation and not for persistent storage.
 */
class DualWorkspace {
public:
    void reset(std::size_t num_dofs)
    {
        if (num_dofs != num_dofs_) {
            blocks_.clear();
            num_dofs_ = num_dofs;
            stride_ = paddedStride(num_dofs_);
        }
        block_index_ = 0;
        offset_ = 0;

        if (num_dofs_ == 0) return;
        if (blocks_.empty()) {
            allocateBlock(kInitialSlots);
        }
    }

    [[nodiscard]] std::size_t numDofs() const noexcept { return num_dofs_; }

    std::span<Real> alloc()
    {
        if (num_dofs_ == 0) return {};
        if (blocks_.empty()) {
            allocateBlock(kInitialSlots);
        }

        auto& block = blocks_[block_index_];
        if (offset_ + stride_ > block.size()) {
            ++block_index_;
            offset_ = 0;

            if (block_index_ >= blocks_.size()) {
                const auto prev_slots = blocks_.back().size() / stride_;
                const auto next_slots = std::max(prev_slots * 2, kInitialSlots);
                allocateBlock(next_slots);
            }
        }

        auto& current = blocks_[block_index_];
        auto span = std::span<Real>(current.data() + offset_, num_dofs_);
        offset_ += stride_;
        return span;
    }

private:
    static constexpr std::size_t kInitialSlots = 64;
    using AlignedRealVector = std::vector<Real, AlignedAllocator<Real, kDualDerivativeAlignmentBytes>>;

    [[nodiscard]] static std::size_t paddedStride(std::size_t num_dofs) noexcept
    {
        if (num_dofs == 0u) return 0u;
        constexpr std::size_t reals_per_alignment =
            (kDualDerivativeAlignmentBytes / sizeof(Real)) > 0u
                ? (kDualDerivativeAlignmentBytes / sizeof(Real))
                : 1u;
        const auto blocks = (num_dofs + reals_per_alignment - 1u) / reals_per_alignment;
        return blocks * reals_per_alignment;
    }

    void allocateBlock(std::size_t slots)
    {
        blocks_.emplace_back(AlignedRealVector(slots * stride_, 0.0));
    }

    std::vector<AlignedRealVector> blocks_{};
    std::size_t num_dofs_{0};
    std::size_t stride_{0};
    std::size_t block_index_{0};
    std::size_t offset_{0};
};

inline Dual makeDual(Real value, std::span<Real> deriv) {
    Dual d;
    d.value = value;
    d.deriv = deriv;
    std::fill(d.deriv.begin(), d.deriv.end(), 0.0);
    return d;
}

inline Dual makeDualConstant(Real value, std::span<Real> deriv) {
    Dual d;
    d.value = value;
    d.deriv = deriv;
    std::fill(d.deriv.begin(), d.deriv.end(), 0.0);
    return d;
}

inline Dual add(const Dual& a, const Dual& b, Dual out) {
    out.value = a.value + b.value;
    const std::size_t n = out.deriv.size();
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = a.deriv[k] + b.deriv[k];
    }
    return out;
}

inline Dual sub(const Dual& a, const Dual& b, Dual out) {
    out.value = a.value - b.value;
    const std::size_t n = out.deriv.size();
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = a.deriv[k] - b.deriv[k];
    }
    return out;
}

inline Dual mul(const Dual& a, const Dual& b, Dual out) {
    out.value = a.value * b.value;
    const std::size_t n = out.deriv.size();
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = a.deriv[k] * b.value + a.value * b.deriv[k];
    }
    return out;
}

inline Dual mul(const Dual& a, Real b, Dual out) {
    out.value = a.value * b;
    const std::size_t n = out.deriv.size();
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = a.deriv[k] * b;
    }
    return out;
}

inline Dual mul(Real a, const Dual& b, Dual out) {
    return mul(b, a, out);
}

inline Dual neg(const Dual& a, Dual out) {
    out.value = -a.value;
    const std::size_t n = out.deriv.size();
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = -a.deriv[k];
    }
    return out;
}

inline Dual copy(const Dual& a, Dual out)
{
    out.value = a.value;
    const std::size_t n = out.deriv.size();
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = a.deriv[k];
    }
    return out;
}

inline Dual div(const Dual& a, const Dual& b, Dual out)
{
    out.value = a.value / b.value;
    const Real inv_b = 1.0 / b.value;
    const Real inv_b2 = inv_b * inv_b;
    const std::size_t n = out.deriv.size();
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = (a.deriv[k] * b.value - a.value * b.deriv[k]) * inv_b2;
    }
    return out;
}

inline Dual div(const Dual& a, Real b, Dual out)
{
    out.value = a.value / b;
    const Real inv_b = 1.0 / b;
    const std::size_t n = out.deriv.size();
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = a.deriv[k] * inv_b;
    }
    return out;
}

inline Dual div(Real a, const Dual& b, Dual out)
{
    out.value = a / b.value;
    const Real inv_b = 1.0 / b.value;
    const Real inv_b2 = inv_b * inv_b;
    const std::size_t n = out.deriv.size();
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = (-a) * b.deriv[k] * inv_b2;
    }
    return out;
}

inline Dual abs(const Dual& a, Dual out)
{
    if (a.value >= 0.0) {
        return copy(a, out);
    }
    return neg(a, out);
}

inline Dual sign(const Dual& a, Dual out)
{
    out.value = (a.value > 0.0) ? 1.0 : ((a.value < 0.0) ? -1.0 : 0.0);
    std::fill(out.deriv.begin(), out.deriv.end(), 0.0);
    return out;
}

inline Dual sqrt(const Dual& a, Dual out)
{
    out.value = std::sqrt(a.value);
    const Real denom = (out.value != 0.0) ? (2.0 * out.value) : 1.0;
    const std::size_t n = out.deriv.size();
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = a.deriv[k] / denom;
    }
    return out;
}

inline Dual exp(const Dual& a, Dual out)
{
    out.value = std::exp(a.value);
    const std::size_t n = out.deriv.size();
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = out.value * a.deriv[k];
    }
    return out;
}

inline Dual log(const Dual& a, Dual out)
{
    out.value = std::log(a.value);
    const Real inv = 1.0 / a.value;
    const std::size_t n = out.deriv.size();
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = a.deriv[k] * inv;
    }
    return out;
}

inline Dual pow(const Dual& a, const Dual& b, Dual out)
{
    out.value = std::pow(a.value, b.value);
    if (out.value == 0.0) {
        std::fill(out.deriv.begin(), out.deriv.end(), 0.0);
        return out;
    }

    const Real inv_a = 1.0 / a.value;
    const Real log_a = std::log(a.value);
    const std::size_t n = out.deriv.size();
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = out.value * (b.deriv[k] * log_a + b.value * a.deriv[k] * inv_a);
    }
    return out;
}

inline Dual pow(const Dual& a, Real b, Dual out)
{
    out.value = std::pow(a.value, b);
    const std::size_t n = out.deriv.size();
    if (b == 0.0) {
        std::fill(out.deriv.begin(), out.deriv.end(), 0.0);
        return out;
    }
    const Real scale = b * std::pow(a.value, b - 1.0);
    SVMP_DUAL_VECTORIZE
    for (std::size_t k = 0; k < n; ++k) {
        out.deriv[k] = scale * a.deriv[k];
    }
    return out;
}

#undef SVMP_DUAL_VECTORIZE

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_DUAL_H
