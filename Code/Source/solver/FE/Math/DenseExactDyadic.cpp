/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "DenseLinearAlgebra.h"

#include "Core/FEException.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#define DENSE_EXACT_CHECK(condition, message) \
    FE_THROW_IF(!(condition), FEException, message)

namespace svmp::FE::math {
namespace {

using boost::multiprecision::cpp_int;

constexpr std::size_t kMaximumExactDimension = 32u;
constexpr std::size_t kMaximumIntegerBits = 262144u;
constexpr std::size_t kMaximumExactUpdates = 2000000u;
constexpr std::size_t kMaximumBinary64SearchSteps = 64u;
constexpr std::size_t kMaximumModeledIntegerBytes =
    64u * 1024u * 1024u;

struct ExactBudget {
    std::size_t update_count{0u};
    std::size_t maximum_integer_bits{0u};
};

[[nodiscard]] std::size_t integerBits(const cpp_int& value)
{
    if (value == 0) {
        return 0u;
    }
    cpp_int magnitude = value;
    if (magnitude < 0) {
        magnitude = -magnitude;
    }
    return static_cast<std::size_t>(
               boost::multiprecision::msb(magnitude)) +
           1u;
}

void accountInteger(const cpp_int& value,
                    ExactBudget& budget,
                    std::string_view label)
{
    const auto bits = integerBits(value);
    DENSE_EXACT_CHECK(
        bits <= kMaximumIntegerBits,
        std::string(label) +
            ": exact integer exceeds the fixed bit cap");
    budget.maximum_integer_bits =
        std::max(budget.maximum_integer_bits, bits);
}

void accountUpdate(ExactBudget& budget,
                   std::string_view label)
{
    DENSE_EXACT_CHECK(
        budget.update_count < kMaximumExactUpdates,
        std::string(label) +
            ": exact arithmetic update cap exceeded");
    ++budget.update_count;
}

[[nodiscard]] cpp_int exactProduct(const cpp_int& left,
                                   const cpp_int& right,
                                   ExactBudget& budget,
                                   std::string_view label)
{
    const auto left_bits = integerBits(left);
    const auto right_bits = integerBits(right);
    DENSE_EXACT_CHECK(
        left_bits == 0u || right_bits == 0u ||
            left_bits + right_bits <=
                kMaximumIntegerBits + 1u,
        std::string(label) +
            ": exact product would exceed the fixed bit cap");
    accountUpdate(budget, label);
    cpp_int product = left * right;
    accountInteger(product, budget, label);
    return product;
}

[[nodiscard]] cpp_int exactDifference(cpp_int left,
                                      const cpp_int& right,
                                      ExactBudget& budget,
                                      std::string_view label)
{
    accountUpdate(budget, label);
    left -= right;
    accountInteger(left, budget, label);
    return left;
}

struct Binary64Dyadic {
    bool negative{false};
    std::uint64_t significand{0u};
    int exponent{0};
};

[[nodiscard]] Binary64Dyadic decodeBinary64(
    Real value,
    std::string_view label)
{
    static_assert(
        sizeof(Real) == sizeof(std::uint64_t) &&
            std::numeric_limits<Real>::is_iec559 &&
            std::numeric_limits<Real>::digits == 53,
        "exact dyadic dense certification requires IEEE binary64 Real");
    constexpr std::uint64_t kFractionMask =
        (UINT64_C(1) << 52u) - UINT64_C(1);
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const auto exponent_bits =
        (bits >> 52u) & UINT64_C(0x7ff);
    DENSE_EXACT_CHECK(
        exponent_bits != UINT64_C(0x7ff),
        std::string(label) +
            ": exact dyadic input is nonfinite");

    Binary64Dyadic result;
    result.negative = (bits >> 63u) != 0u;
    const auto fraction = bits & kFractionMask;
    if (exponent_bits == 0u) {
        result.significand = fraction;
        result.exponent = -1074;
    } else {
        result.significand =
            (UINT64_C(1) << 52u) | fraction;
        result.exponent =
            static_cast<int>(exponent_bits) - 1023 - 52;
    }
    if (result.significand == 0u) {
        result.negative = false;
        return result;
    }
    while ((result.significand & UINT64_C(1)) == 0u) {
        result.significand >>= 1u;
        ++result.exponent;
    }
    return result;
}

[[nodiscard]] std::size_t checkedSquare(
    std::size_t n,
    std::string_view label)
{
    DENSE_EXACT_CHECK(
        n > 0u && n <= kMaximumExactDimension,
        std::string(label) +
            ": exact quotient dimension is outside the fixed cap");
    DENSE_EXACT_CHECK(
        n <= std::numeric_limits<std::size_t>::max() / n,
        std::string(label) +
            ": exact quotient dimension product overflows size_t");
    const auto entries = n * n;
    DENSE_EXACT_CHECK(
        entries <=
            kMaximumModeledIntegerBytes /
                (kMaximumIntegerBits / 8u),
        std::string(label) +
            ": exact quotient modeled integer storage exceeds its cap");
    return entries;
}

void validateSymmetricBinary64Inputs(
    std::span<const Real> numerator,
    std::span<const Real> denominator,
    std::size_t n,
    std::string_view label)
{
    const auto entries = checkedSquare(n, label);
    DENSE_EXACT_CHECK(
        numerator.size() == entries,
        std::string(label) +
            ": exact numerator size mismatch");
    DENSE_EXACT_CHECK(
        denominator.size() == entries,
        std::string(label) +
            ": exact denominator size mismatch");
    for (std::size_t row = 0u; row < n; ++row) {
        for (std::size_t column = 0u;
             column < n;
             ++column) {
            DENSE_EXACT_CHECK(
                std::isfinite(numerator[row * n + column]) &&
                    std::isfinite(
                        denominator[row * n + column]),
                std::string(label) +
                    ": exact quotient contains a nonfinite entry");
        }
        for (std::size_t column = row + 1u;
             column < n;
             ++column) {
            DENSE_EXACT_CHECK(
                numerator[row * n + column] ==
                    numerator[column * n + row],
                std::string(label) +
                    ": exact numerator is not symmetric");
            DENSE_EXACT_CHECK(
                denominator[row * n + column] ==
                    denominator[column * n + row],
                std::string(label) +
                    ": exact denominator is not symmetric");
        }
    }
}

void removeCommonPowerOfTwo(std::vector<cpp_int>& matrix)
{
    std::size_t common =
        std::numeric_limits<std::size_t>::max();
    for (const auto& value : matrix) {
        if (value == 0) {
            continue;
        }
        cpp_int magnitude = value;
        if (magnitude < 0) {
            magnitude = -magnitude;
        }
        common = std::min(
            common,
            static_cast<std::size_t>(
                boost::multiprecision::lsb(magnitude)));
    }
    if (common ==
            std::numeric_limits<std::size_t>::max() ||
        common == 0u) {
        return;
    }
    for (auto& value : matrix) {
        value >>= common;
    }
}

[[nodiscard]] std::vector<cpp_int> integerizeSingleMatrix(
    std::span<const Binary64Dyadic> decoded,
    ExactBudget& budget,
    std::string_view label)
{
    int minimum_exponent = std::numeric_limits<int>::max();
    for (const auto& value : decoded) {
        if (value.significand != 0u) {
            minimum_exponent =
                std::min(minimum_exponent, value.exponent);
        }
    }
    std::vector<cpp_int> result(decoded.size(), cpp_int{0});
    if (minimum_exponent ==
        std::numeric_limits<int>::max()) {
        return result;
    }
    for (std::size_t index = 0u;
         index < decoded.size();
         ++index) {
        const auto& value = decoded[index];
        if (value.significand == 0u) {
            continue;
        }
        const auto shift = static_cast<std::size_t>(
            value.exponent - minimum_exponent);
        cpp_int integer = value.significand;
        DENSE_EXACT_CHECK(
            integerBits(integer) + shift <=
                kMaximumIntegerBits,
            std::string(label) +
                ": exact dyadic exponent span exceeds the bit cap");
        integer <<= shift;
        if (value.negative) {
            integer = -integer;
        }
        accountInteger(integer, budget, label);
        result[index] = std::move(integer);
    }
    removeCommonPowerOfTwo(result);
    return result;
}

class ExactDyadicPencil {
public:
    ExactDyadicPencil(std::span<const Real> numerator,
                      std::span<const Real> denominator,
                      std::size_t n,
                      std::string_view label)
        : label_(label)
    {
        validateSymmetricBinary64Inputs(
            numerator, denominator, n, label);
        numerator_.reserve(numerator.size());
        denominator_.reserve(denominator.size());
        for (const auto value : numerator) {
            numerator_.push_back(
                decodeBinary64(value, label_));
        }
        for (const auto value : denominator) {
            denominator_.push_back(
                decodeBinary64(value, label_));
        }
    }

    [[nodiscard]] std::vector<cpp_int> numeratorIntegers(
        ExactBudget& budget) const
    {
        return integerizeSingleMatrix(
            numerator_, budget,
            label_ + " numerator");
    }

    [[nodiscard]] std::vector<cpp_int> denominatorIntegers(
        ExactBudget& budget) const
    {
        return integerizeSingleMatrix(
            denominator_, budget,
            label_ + " denominator");
    }

    [[nodiscard]] std::vector<cpp_int> differenceIntegers(
        Real coefficient,
        ExactBudget& budget) const
    {
        const auto factor =
            decodeBinary64(coefficient, label_ + " coefficient");
        DENSE_EXACT_CHECK(
            !factor.negative,
            label_ +
                ": exact generalized coefficient is negative");

        int minimum_exponent =
            std::numeric_limits<int>::max();
        for (std::size_t index = 0u;
             index < numerator_.size();
             ++index) {
            if (factor.significand != 0u &&
                denominator_[index].significand != 0u) {
                minimum_exponent = std::min(
                    minimum_exponent,
                    factor.exponent +
                        denominator_[index].exponent);
            }
            if (numerator_[index].significand != 0u) {
                minimum_exponent = std::min(
                    minimum_exponent,
                    numerator_[index].exponent);
            }
        }

        std::vector<cpp_int> result(
            numerator_.size(), cpp_int{0});
        if (minimum_exponent ==
            std::numeric_limits<int>::max()) {
            return result;
        }
        for (std::size_t index = 0u;
             index < result.size();
             ++index) {
            cpp_int value{0};
            const auto& denominator = denominator_[index];
            if (factor.significand != 0u &&
                denominator.significand != 0u) {
                cpp_int product = exactProduct(
                    cpp_int{factor.significand},
                    cpp_int{denominator.significand},
                    budget,
                    label_ + " coefficient-denominator product");
                const auto shift = static_cast<std::size_t>(
                    factor.exponent + denominator.exponent -
                    minimum_exponent);
                DENSE_EXACT_CHECK(
                    integerBits(product) + shift <=
                        kMaximumIntegerBits,
                    label_ +
                        ": exact pencil exponent span exceeds the bit cap");
                product <<= shift;
                if (denominator.negative) {
                    product = -product;
                }
                accountInteger(product, budget, label_);
                value = std::move(product);
            }
            const auto& numerator = numerator_[index];
            if (numerator.significand != 0u) {
                cpp_int term = numerator.significand;
                const auto shift = static_cast<std::size_t>(
                    numerator.exponent - minimum_exponent);
                DENSE_EXACT_CHECK(
                    integerBits(term) + shift <=
                        kMaximumIntegerBits,
                    label_ +
                        ": exact pencil exponent span exceeds the bit cap");
                term <<= shift;
                if (numerator.negative) {
                    term = -term;
                }
                value = exactDifference(
                    std::move(value), term, budget,
                    label_ + " coefficient denominator minus numerator");
            }
            accountInteger(value, budget, label_);
            result[index] = std::move(value);
        }
        removeCommonPowerOfTwo(result);
        return result;
    }

private:
    std::string label_{};
    std::vector<Binary64Dyadic> numerator_{};
    std::vector<Binary64Dyadic> denominator_{};
};

enum class ExactPsdStatus : std::uint8_t {
    NotPositiveSemidefinite = 0u,
    PositiveSemidefinite = 1u,
};

struct ExactPsdResult {
    ExactPsdStatus status{
        ExactPsdStatus::NotPositiveSemidefinite};
    std::size_t rank{0u};
};

void symmetricSwap(std::vector<cpp_int>& matrix,
                   std::size_t n,
                   std::size_t left,
                   std::size_t right)
{
    if (left == right) {
        return;
    }
    for (std::size_t column = 0u;
         column < n;
         ++column) {
        std::swap(
            matrix[left * n + column],
            matrix[right * n + column]);
    }
    for (std::size_t row = 0u; row < n; ++row) {
        std::swap(
            matrix[row * n + left],
            matrix[row * n + right]);
    }
}

[[nodiscard]] ExactPsdResult exactPositiveSemidefiniteRank(
    std::vector<cpp_int> matrix,
    std::size_t n,
    ExactBudget& budget,
    std::string_view label)
{
    DENSE_EXACT_CHECK(
        matrix.size() == n * n,
        std::string(label) +
            ": exact PSD matrix size mismatch");
    cpp_int previous_pivot{1};
    std::size_t rank = 0u;
    for (std::size_t step = 0u;
         step < n;
         ++step) {
        std::size_t pivot_coordinate = n;
        for (std::size_t coordinate = step;
             coordinate < n;
             ++coordinate) {
            const auto& diagonal =
                matrix[coordinate * n + coordinate];
            if (diagonal < 0) {
                return {};
            }
            if (pivot_coordinate == n && diagonal > 0) {
                pivot_coordinate = coordinate;
            }
        }
        if (pivot_coordinate == n) {
            for (std::size_t row = step;
                 row < n;
                 ++row) {
                for (std::size_t column = step;
                     column < n;
                     ++column) {
                    if (matrix[row * n + column] != 0) {
                        return {};
                    }
                }
            }
            return {
                ExactPsdStatus::PositiveSemidefinite,
                rank};
        }

        symmetricSwap(
            matrix, n, step, pivot_coordinate);
        const cpp_int pivot = matrix[step * n + step];
        DENSE_EXACT_CHECK(
            pivot > 0 && previous_pivot > 0,
            std::string(label) +
                ": exact PSD pivot invariant failed");

        for (std::size_t row = step + 1u;
             row < n;
             ++row) {
            for (std::size_t column = row;
                 column < n;
                 ++column) {
                auto numerator = exactDifference(
                    exactProduct(
                        pivot,
                        matrix[row * n + column],
                        budget,
                        label),
                    exactProduct(
                        matrix[row * n + step],
                        matrix[step * n + column],
                        budget,
                        label),
                    budget,
                    label);
                if (step != 0u) {
                    DENSE_EXACT_CHECK(
                        numerator % previous_pivot == 0,
                        std::string(label) +
                            ": fraction-free PSD division is not exact");
                    accountUpdate(budget, label);
                    numerator /= previous_pivot;
                    accountInteger(numerator, budget, label);
                }
                matrix[row * n + column] = numerator;
                matrix[column * n + row] =
                    std::move(numerator);
            }
        }
        for (std::size_t coordinate = step + 1u;
             coordinate < n;
             ++coordinate) {
            matrix[step * n + coordinate] = 0;
            matrix[coordinate * n + step] = 0;
        }
        previous_pivot = pivot;
        ++rank;
    }
    return {
        ExactPsdStatus::PositiveSemidefinite,
        rank};
}

[[nodiscard]] Real nonnegativeBinary64FromOrderedBits(
    std::uint64_t bits)
{
    DENSE_EXACT_CHECK(
        bits <= UINT64_C(0x7fefffffffffffff),
        "exact dyadic binary64 search produced an invalid bit pattern");
    const Real value = std::bit_cast<Real>(bits);
    DENSE_EXACT_CHECK(
        std::isfinite(value) && value >= Real{0},
        "exact dyadic binary64 search produced an invalid value");
    return value;
}

} // namespace

DenseExactDyadicSpdGeneralizedUpperBound
dense_exact_dyadic_spd_generalized_upper_bound(
    std::span<const Real> numerator,
    std::span<const Real> denominator,
    std::size_t n,
    std::string_view label)
{
    const std::string label_text(label);
    ExactDyadicPencil pencil(
        numerator, denominator, n, label);
    ExactBudget budget;

    DenseExactDyadicSpdGeneralizedUpperBound result;
    result.applied = true;
    result.dimension = n;

    const auto denominator_psd =
        exactPositiveSemidefiniteRank(
            pencil.denominatorIntegers(budget),
            n,
            budget,
            label_text + " denominator");
    ++result.psd_oracle_calls;
    DENSE_EXACT_CHECK(
        denominator_psd.status ==
                ExactPsdStatus::PositiveSemidefinite &&
            denominator_psd.rank == n,
        label_text +
            ": exact denominator is not positive definite");
    result.denominator_rank = denominator_psd.rank;
    result.denominator_positive_definite_proven = true;

    const auto numerator_psd =
        exactPositiveSemidefiniteRank(
            pencil.numeratorIntegers(budget),
            n,
            budget,
            label_text + " numerator");
    ++result.psd_oracle_calls;
    DENSE_EXACT_CHECK(
        numerator_psd.status ==
            ExactPsdStatus::PositiveSemidefinite,
        label_text +
            ": exact numerator is not positive semidefinite");
    result.numerator_rank = numerator_psd.rank;
    result.numerator_positive_semidefinite_proven = true;

    const auto proves_upper =
        [&](Real coefficient) {
            const auto proof =
                exactPositiveSemidefiniteRank(
                    pencil.differenceIntegers(
                        coefficient, budget),
                    n,
                    budget,
                    label_text +
                        " coefficient denominator minus numerator");
            ++result.psd_oracle_calls;
            return proof.status ==
                   ExactPsdStatus::PositiveSemidefinite;
        };

    if (proves_upper(Real{0})) {
        result.directly_proven_upper_bound = Real{0};
        result.upper_inequality_proven = true;
        result.exact_update_count = budget.update_count;
        result.maximum_integer_bits =
            budget.maximum_integer_bits;
        return result;
    }
    result.failing_lower_bound_proven = true;

    constexpr std::uint64_t kMaximumFiniteBits =
        UINT64_C(0x7fefffffffffffff);
    DENSE_EXACT_CHECK(
        proves_upper(
            nonnegativeBinary64FromOrderedBits(
                kMaximumFiniteBits)),
        label_text +
            ": no finite binary64 generalized upper bound was proved");

    std::uint64_t failing_bits = 0u;
    std::uint64_t passing_bits = kMaximumFiniteBits;
    while (passing_bits - failing_bits > UINT64_C(1)) {
        const auto middle_bits =
            failing_bits +
            (passing_bits - failing_bits) / UINT64_C(2);
        const Real candidate =
            nonnegativeBinary64FromOrderedBits(middle_bits);
        ++result.binary64_search_steps;
        DENSE_EXACT_CHECK(
            result.binary64_search_steps <=
                kMaximumBinary64SearchSteps,
            label_text +
                ": exact binary64 search-step cap exceeded");
        if (proves_upper(candidate)) {
            passing_bits = middle_bits;
        } else {
            failing_bits = middle_bits;
        }
    }

    result.directly_proven_upper_bound =
        nonnegativeBinary64FromOrderedBits(passing_bits);
    result.largest_failing_lower_bound =
        nonnegativeBinary64FromOrderedBits(failing_bits);
    result.upper_inequality_proven = true;
    result.exact_update_count = budget.update_count;
    result.maximum_integer_bits =
        budget.maximum_integer_bits;
    return result;
}

} // namespace svmp::FE::math
