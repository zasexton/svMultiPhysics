/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_EXACTACCUMULATOR_H
#define SVMP_FE_SYSTEMS_EXACTACCUMULATOR_H

/**
 * @file ExactAccumulator.h
 * @brief Exact, order-independent summation of doubles.
 *
 * An ExactAccumulator holds the exact sum of the finite doubles added to it
 * as a fixed-point integer that covers the whole double range: the sum is
 * N * 2^-1074, and N is kept in kDigits signed 64-bit slots of 32-bit digits
 * (digit k weighs 2^(32 k)).  Adding a double adds its mantissa, shifted by
 * its exponent, into at most three digits, so add() is exact and the sum
 * does not depend on the order of the terms.  value() rounds the exact sum
 * once, to nearest with ties to even.
 *
 * Accumulators of a partition of the terms (threads, cells, ranks) combine
 * by adding their integers, again exactly, so every partition gives the
 * same value: collectiveExactSum (SystemCollectives.h) combines the ranks'
 * accumulators with one integer MPI_Allreduce(SUM) of their slots().
 *
 * Non-finite terms are counted apart: the value is NaN after a NaN term or
 * after infinities of both signs, and +/-infinity after infinities of one
 * sign only.  A sum that is exactly zero is +0.  A finite sum beyond the
 * double range rounds to +/-infinity.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace svmp {
namespace FE {
namespace systems {

class ExactAccumulator {
public:
    /// 32-bit digits from 2^-1074: 2045 exponent steps and a 53-bit
    /// mantissa need 66 digits; the top one also takes the carries.
    static constexpr std::size_t kDigits = 67;
    /// Slots of slots(): the digits, then the counts of NaN, +infinity and
    /// -infinity terms.
    static constexpr std::size_t kSlots = kDigits + 3;

    ExactAccumulator() = default;

    /// Adds @p term exactly.
    void add(double term) noexcept;

    /// Adds every term exactly.
    void add(std::span<const double> terms) noexcept;

    /// Adds the exact sum of @p other.
    ExactAccumulator& operator+=(const ExactAccumulator& other) noexcept;

    /// The exact sum rounded to nearest, ties to even (non-finite terms as
    /// above).
    [[nodiscard]] double value() const noexcept;

    /**
     * @brief The accumulator as integers in canonical form.
     *
     * Digits 0 to kDigits - 2 are in [0, 2^32), the top digit is signed and
     * small, the three counts follow.  Equal sums give equal slots, and the
     * element-wise sum of the slots of several accumulators (over at most
     * 2^30 of them) does not overflow.
     */
    [[nodiscard]] std::array<std::int64_t, kSlots> slots() const noexcept;

    /// The accumulator whose exact sum is that of @p slots, for example the
    /// element-wise sum of the slots() of several accumulators.
    [[nodiscard]] static ExactAccumulator fromSlots(
        std::span<const std::int64_t, kSlots> slots) noexcept;

private:
    void normalize() noexcept;

    std::array<std::int64_t, kDigits> digits_{};
    std::int64_t nan_terms_{0};
    std::int64_t positive_infinite_terms_{0};
    std::int64_t negative_infinite_terms_{0};
    std::uint32_t adds_since_normalized_{0};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_EXACTACCUMULATOR_H
