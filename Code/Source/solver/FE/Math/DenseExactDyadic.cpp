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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#define DENSE_EXACT_CHECK(condition, message) \
    FE_THROW_IF(!(condition), FEException, message)

namespace svmp::FE::math {
namespace {

using boost::multiprecision::cpp_int;

constexpr std::size_t kMaximumExactDimension =
    dense_exact_dyadic_maximum_dimension;
constexpr std::size_t kMaximumIntegerBits = 262144u;
constexpr std::size_t kMaximumLegacyExactUpdates = 2000000u;
constexpr std::size_t kMaximumExactGramBlocks = 16384u;
constexpr std::size_t kMaximumExactGramRows = 262144u;
constexpr std::size_t kMaximumExactWeightTerms = 1048576u;
constexpr std::size_t kMaximumExactWeightProductFactors = 4u;
constexpr std::size_t kMaximumExactTransformEntries = 1048576u;
constexpr std::size_t kMaximumExactInputScalars = 1048576u;
constexpr std::size_t kMaximumExactTransformProducts = 1048576u;
constexpr std::size_t kMaximumExactOuterProducts = 8388608u;
constexpr std::size_t kMaximumBinary64SearchSteps = 64u;
constexpr std::size_t kMaximumModeledIntegerBytes =
    64u * 1024u * 1024u;

struct ExactBudget {
    std::size_t update_count{0u};
    std::size_t maximum_integer_bits{0u};
    std::size_t update_limit{kMaximumLegacyExactUpdates};
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
        budget.update_count < budget.update_limit,
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

[[nodiscard]] cpp_int exactSum(cpp_int left,
                               const cpp_int& right,
                               ExactBudget& budget,
                               std::string_view label)
{
    accountUpdate(budget, label);
    left += right;
    accountInteger(left, budget, label);
    return left;
}

struct Binary64Dyadic {
    bool negative{false};
    std::uint64_t significand{0u};
    int exponent{0};
};

template <typename To, typename From>
[[nodiscard]] To bitwiseCopy(const From& source) noexcept
{
    static_assert(sizeof(To) == sizeof(From));
    static_assert(std::is_trivially_copyable_v<To>);
    static_assert(std::is_trivially_copyable_v<From>);
    To destination{};
    std::memcpy(&destination, &source, sizeof(destination));
    return destination;
}

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
    const auto bits = bitwiseCopy<std::uint64_t>(value);
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

[[nodiscard]] std::size_t removeCommonPowerOfTwo(
    std::vector<cpp_int>& matrix)
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
        return 0u;
    }
    for (auto& value : matrix) {
        value >>= common;
    }
    return common;
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
    static_cast<void>(removeCommonPowerOfTwo(result));
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
        static_cast<void>(removeCommonPowerOfTwo(result));
        return result;
    }

private:
    std::string label_{};
    std::vector<Binary64Dyadic> numerator_{};
    std::vector<Binary64Dyadic> denominator_{};
};

struct ExactScaledIntegerMatrix {
    std::vector<cpp_int> entries{};
    int exponent{0};
};

struct ExactDyadicScalar {
    cpp_int integer{};
    int exponent{0};
};

[[nodiscard]] int checkedExactExponent(
    long long exponent,
    std::string_view label)
{
    DENSE_EXACT_CHECK(
        exponent >= static_cast<long long>(
                        std::numeric_limits<int>::min()) &&
            exponent <= static_cast<long long>(
                            std::numeric_limits<int>::max()),
        std::string(label) +
            ": exact dyadic exponent exceeds the supported range");
    return static_cast<int>(exponent);
}

void normalizeExactDyadic(ExactDyadicScalar& value,
                          std::string_view label)
{
    if (value.integer == 0) {
        value.exponent = 0;
        return;
    }
    const bool negative = value.integer < 0;
    cpp_int magnitude = negative ? -value.integer : value.integer;
    const auto power = static_cast<std::size_t>(
        boost::multiprecision::lsb(magnitude));
    if (power == 0u) {
        return;
    }
    magnitude >>= power;
    value.integer = negative ? -magnitude : magnitude;
    value.exponent = checkedExactExponent(
        static_cast<long long>(value.exponent) +
            static_cast<long long>(power),
        label);
}

[[nodiscard]] ExactDyadicScalar exactDyadicFromBinary64(
    Real value,
    std::string_view label)
{
    const auto decoded = decodeBinary64(value, label);
    ExactDyadicScalar result{
        cpp_int{decoded.significand}, decoded.exponent};
    if (decoded.negative) {
        result.integer = -result.integer;
    }
    normalizeExactDyadic(result, label);
    return result;
}

[[nodiscard]] ExactDyadicScalar exactDyadicFromUnsigned(
    std::uint64_t value,
    std::string_view label)
{
    DENSE_EXACT_CHECK(
        value != 0u,
        std::string(label) +
            ": exact positive integer multiplier is zero");
    ExactDyadicScalar result{cpp_int{value}, 0};
    normalizeExactDyadic(result, label);
    return result;
}

[[nodiscard]] ExactDyadicScalar exactDyadicProduct(
    const ExactDyadicScalar& left,
    const ExactDyadicScalar& right,
    ExactBudget& budget,
    std::string_view label)
{
    if (left.integer == 0 || right.integer == 0) {
        return {};
    }
    ExactDyadicScalar result;
    result.integer = exactProduct(
        left.integer, right.integer, budget, label);
    result.exponent = checkedExactExponent(
        static_cast<long long>(left.exponent) +
            static_cast<long long>(right.exponent),
        label);
    normalizeExactDyadic(result, label);
    return result;
}

[[nodiscard]] ExactDyadicScalar exactDyadicSum(
    const ExactDyadicScalar& left,
    const ExactDyadicScalar& right,
    ExactBudget& budget,
    std::string_view label)
{
    if (left.integer == 0) {
        return right;
    }
    if (right.integer == 0) {
        return left;
    }
    const int exponent = std::min(left.exponent, right.exponent);
    const auto shifted =
        [&](const ExactDyadicScalar& value) {
            const auto shift = static_cast<std::size_t>(
                static_cast<long long>(value.exponent) -
                static_cast<long long>(exponent));
            DENSE_EXACT_CHECK(
                integerBits(value.integer) + shift <=
                    kMaximumIntegerBits,
                std::string(label) +
                    ": exact dyadic sum exponent span exceeds the bit cap");
            cpp_int result = value.integer;
            result <<= shift;
            accountInteger(result, budget, label);
            return result;
        };
    ExactDyadicScalar result;
    result.integer = exactSum(
        shifted(left), shifted(right), budget, label);
    result.exponent = exponent;
    normalizeExactDyadic(result, label);
    return result;
}

[[nodiscard]] ExactScaledIntegerMatrix integerizeExactDyadicMatrix(
    std::span<const ExactDyadicScalar> matrix,
    std::size_t n,
    ExactBudget& budget,
    std::string_view label)
{
    const auto entries = checkedSquare(n, label);
    DENSE_EXACT_CHECK(
        matrix.size() == entries,
        std::string(label) +
            ": exact factorized matrix size mismatch");
    int minimum_exponent = std::numeric_limits<int>::max();
    for (const auto& value : matrix) {
        if (value.integer != 0) {
            minimum_exponent =
                std::min(minimum_exponent, value.exponent);
        }
    }
    ExactScaledIntegerMatrix result;
    result.entries.assign(entries, cpp_int{0});
    if (minimum_exponent == std::numeric_limits<int>::max()) {
        return result;
    }
    result.exponent = minimum_exponent;
    for (std::size_t index = 0u; index < entries; ++index) {
        const auto& value = matrix[index];
        if (value.integer == 0) {
            continue;
        }
        const auto shift = static_cast<std::size_t>(
            static_cast<long long>(value.exponent) -
            static_cast<long long>(minimum_exponent));
        DENSE_EXACT_CHECK(
            integerBits(value.integer) + shift <=
                kMaximumIntegerBits,
            std::string(label) +
                ": exact factorized matrix exponent span exceeds the bit cap");
        auto integer = value.integer;
        integer <<= shift;
        accountInteger(integer, budget, label);
        result.entries[index] = std::move(integer);
    }
    const auto common = removeCommonPowerOfTwo(result.entries);
    result.exponent = checkedExactExponent(
        static_cast<long long>(result.exponent) +
            static_cast<long long>(common),
        label);
    return result;
}

struct FactorizedPreflight {
    std::size_t block_count{0u};
    std::size_t numerator_block_count{0u};
    std::size_t denominator_block_count{0u};
    std::size_t numerator_row_count{0u};
    std::size_t denominator_row_count{0u};
    std::size_t numerator_weight_term_count{0u};
    std::size_t denominator_weight_term_count{0u};
    std::size_t raw_factor_coefficient_count{0u};
    std::size_t transform_visit_count{0u};
    std::size_t outer_pair_count{0u};
    std::size_t positive_scale_update_count{0u};
    std::size_t modeled_input_bytes{0u};
    std::uint64_t digest{UINT64_C(1469598103934665603)};
};

void addCappedCount(std::size_t& total,
                    std::size_t additional,
                    std::size_t cap,
                    std::string_view label)
{
    DENSE_EXACT_CHECK(
        additional <= cap - std::min(total, cap),
        std::string(label) + " exceeds the fixed cap");
    total += additional;
}

void addModeledBytes(FactorizedPreflight& preflight,
                     std::size_t count,
                     std::size_t element_bytes,
                     std::string_view label)
{
    constexpr std::size_t kMaximumFactorizedModeledBytes =
        64u * 1024u * 1024u;
    DENSE_EXACT_CHECK(
        element_bytes == 0u ||
            count <=
                (kMaximumFactorizedModeledBytes -
                 std::min(preflight.modeled_input_bytes,
                          kMaximumFactorizedModeledBytes)) /
                    element_bytes,
        std::string(label) +
            ": exact factorized input exceeds the modeled-byte cap");
    preflight.modeled_input_bytes += count * element_bytes;
}

void exactDigestMix(std::uint64_t& digest,
                    std::uint64_t value) noexcept
{
    for (unsigned byte = 0u; byte < 8u; ++byte) {
        digest ^= (value >> (byte * 8u)) & UINT64_C(0xff);
        digest *= UINT64_C(1099511628211);
    }
}

void exactDigestMixReal(std::uint64_t& digest,
                        Real value) noexcept
{
    exactDigestMix(
        digest, bitwiseCopy<std::uint64_t>(value));
}

void preflightSparseMapShape(
    DenseExactDyadicSparseMapView map,
    FactorizedPreflight& preflight,
    std::string_view label)
{
    DENSE_EXACT_CHECK(
        map.input_dimension > 0u &&
            map.input_dimension <= kMaximumExactTransformEntries,
        std::string(label) +
            ": exact sparse-map input dimension is outside its cap");
    DENSE_EXACT_CHECK(
        map.output_dimension <= kMaximumExactDimension,
        std::string(label) +
            ": exact sparse-map output dimension " +
            std::to_string(map.output_dimension) +
            " exceeds the fixed cap " +
            std::to_string(kMaximumExactDimension));
    if (map.output_dimension != 0u) {
        static_cast<void>(checkedSquare(map.output_dimension, label));
    }
    DENSE_EXACT_CHECK(
        map.row_offsets.size() == map.input_dimension + 1u,
        std::string(label) +
            ": exact sparse-map row-offset shape mismatch");
    DENSE_EXACT_CHECK(
        map.entries.size() <= kMaximumExactTransformEntries,
        std::string(label) +
            ": exact sparse-map entry count exceeds its cap");
    addModeledBytes(
        preflight, map.row_offsets.size(), sizeof(std::size_t), label);
    addModeledBytes(
        preflight,
        map.entries.size(),
        sizeof(DenseExactDyadicSparseMapEntry),
        label);
}

void validateSparseMapOffsets(
    DenseExactDyadicSparseMapView map,
    std::string_view label)
{
    DENSE_EXACT_CHECK(
        !map.row_offsets.empty() &&
            map.row_offsets.front() == 0u &&
            map.row_offsets.back() == map.entries.size(),
        std::string(label) +
            ": exact sparse-map offsets do not span the entries");
    for (std::size_t row = 0u; row < map.input_dimension; ++row) {
        const auto begin = map.row_offsets[row];
        const auto end = map.row_offsets[row + 1u];
        DENSE_EXACT_CHECK(
            begin <= end && end <= map.entries.size(),
            std::string(label) +
                ": exact sparse-map offsets are not monotone");
    }
}

void validateSparseMapEntriesAndDigest(
    DenseExactDyadicSparseMapView map,
    FactorizedPreflight& preflight,
    std::string_view label)
{
    for (std::size_t row = 0u; row < map.input_dimension; ++row) {
        const auto begin = map.row_offsets[row];
        const auto end = map.row_offsets[row + 1u];
        std::size_t previous = map.output_dimension;
        for (std::size_t index = begin; index < end; ++index) {
            const auto& entry = map.entries[index];
            DENSE_EXACT_CHECK(
                entry.output_coordinate < map.output_dimension &&
                    std::isfinite(entry.coefficient) &&
                    entry.coefficient != Real{0},
                std::string(label) +
                    ": exact sparse-map entry is invalid");
            DENSE_EXACT_CHECK(
                previous == map.output_dimension ||
                    entry.output_coordinate > previous,
                std::string(label) +
                    ": exact sparse-map row is not strictly ordered");
            previous = entry.output_coordinate;
        }
    }
    exactDigestMix(preflight.digest, UINT64_C(0x5350415253454d41));
    exactDigestMix(preflight.digest, map.input_dimension);
    exactDigestMix(preflight.digest, map.output_dimension);
    exactDigestMix(preflight.digest, map.row_offsets.size());
    for (const auto offset : map.row_offsets) {
        exactDigestMix(preflight.digest, offset);
    }
    exactDigestMix(preflight.digest, map.entries.size());
    for (const auto& entry : map.entries) {
        exactDigestMix(preflight.digest, entry.output_coordinate);
        exactDigestMixReal(preflight.digest, entry.coefficient);
    }
}

void preflightGramBlockShapes(
    std::span<const DenseExactDyadicGramBlockView> blocks,
    DenseExactDyadicSparseMapView map,
    bool numerator,
    FactorizedPreflight& preflight,
    std::string_view label)
{
    addCappedCount(
        preflight.block_count,
        blocks.size(),
        kMaximumExactGramBlocks,
        label);
    auto& block_count = numerator
        ? preflight.numerator_block_count
        : preflight.denominator_block_count;
    auto& row_count = numerator
        ? preflight.numerator_row_count
        : preflight.denominator_row_count;
    auto& weight_term_count = numerator
        ? preflight.numerator_weight_term_count
        : preflight.denominator_weight_term_count;
    block_count += blocks.size();
    const auto quotient_pairs =
        map.output_dimension * (map.output_dimension + 1u) / 2u;
    for (const auto& block : blocks) {
        DENSE_EXACT_CHECK(
            !block.map_rows.empty() &&
                block.factor_row_count > 0u,
            std::string(label) +
                ": exact Gram block is empty");
        DENSE_EXACT_CHECK(
            block.factor_row_count <=
                std::numeric_limits<std::size_t>::max() /
                    block.map_rows.size() &&
                block.row_major_raw_factors.size() ==
                    block.factor_row_count * block.map_rows.size(),
            std::string(label) +
                ": exact Gram block factor shape mismatch");
        DENSE_EXACT_CHECK(
            block.row_multipliers.empty() ||
                block.row_multipliers.size() == block.factor_row_count,
            std::string(label) +
                ": exact Gram row-multiplier shape mismatch");
        DENSE_EXACT_CHECK(
            !block.scale.positive_sum_terms.empty() &&
                block.scale.positive_product_factors.size() <=
                    kMaximumExactWeightProductFactors,
            std::string(label) +
                ": exact Gram positive scale is malformed");

        // Apply every shape-derived count, work, and modeled-byte cap before
        // scanning any caller-owned payload. Transform work is the only
        // remaining dependent count; it is derived later from already-bounded
        // sparse-map metadata before scalar payloads are inspected.
        addCappedCount(
            row_count,
            block.factor_row_count,
            kMaximumExactGramRows,
            label);
        addCappedCount(
            weight_term_count,
            block.scale.positive_sum_terms.size(),
            kMaximumExactWeightTerms,
            label);
        addCappedCount(
            preflight.raw_factor_coefficient_count,
            block.row_major_raw_factors.size(),
            kMaximumExactInputScalars,
            label);
        addModeledBytes(
            preflight, block.map_rows.size(), sizeof(std::size_t), label);
        addModeledBytes(
            preflight,
            block.row_major_raw_factors.size(),
            sizeof(Real),
            label);
        addModeledBytes(
            preflight,
            block.row_multipliers.size(),
            sizeof(std::uint64_t),
            label);
        addModeledBytes(
            preflight,
            block.scale.positive_sum_terms.size(),
            sizeof(Real),
            label);
        addModeledBytes(
            preflight,
            block.scale.positive_product_factors.size(),
            sizeof(Real),
            label);
        DENSE_EXACT_CHECK(
            block.factor_row_count == 0u ||
                quotient_pairs <=
                    (kMaximumExactOuterProducts -
                     std::min(preflight.outer_pair_count,
                              kMaximumExactOuterProducts)) /
                        block.factor_row_count,
            std::string(label) +
                ": exact Gram outer-product count exceeds its cap");
        preflight.outer_pair_count +=
            quotient_pairs * block.factor_row_count;
        addCappedCount(
            preflight.positive_scale_update_count,
            block.scale.positive_sum_terms.size() - 1u +
                block.scale.positive_product_factors.size() + 1u +
                block.factor_row_count,
            kMaximumLegacyExactUpdates,
            label);
    }
}

void preflightGramBlockTransformWork(
    std::span<const DenseExactDyadicGramBlockView> blocks,
    DenseExactDyadicSparseMapView map,
    FactorizedPreflight& preflight,
    std::string_view label)
{
    for (const auto& block : blocks) {
        std::size_t entries_per_factor_row = 0u;
        for (std::size_t local = 0u;
             local < block.map_rows.size();
             ++local) {
            const auto map_row = block.map_rows[local];
            DENSE_EXACT_CHECK(
                map_row < map.input_dimension,
                std::string(label) +
                    ": exact Gram block references an unknown map row");
            DENSE_EXACT_CHECK(
                local == 0u ||
                    map_row > block.map_rows[local - 1u],
                std::string(label) +
                    ": exact Gram block map rows are not strictly ordered");
            addCappedCount(
                entries_per_factor_row,
                map.row_offsets[map_row + 1u] -
                    map.row_offsets[map_row],
                kMaximumExactTransformEntries,
                label);
        }
        DENSE_EXACT_CHECK(
            entries_per_factor_row <=
                (kMaximumExactTransformProducts -
                 std::min(preflight.transform_visit_count,
                          kMaximumExactTransformProducts)) /
                    block.factor_row_count,
            std::string(label) +
                ": exact Gram transform-visit count exceeds its cap");
        preflight.transform_visit_count +=
            entries_per_factor_row * block.factor_row_count;
    }
}

void validateGramBlockContentsAndDigest(
    std::span<const DenseExactDyadicGramBlockView> blocks,
    bool numerator,
    FactorizedPreflight& preflight,
    std::string_view label)
{
    exactDigestMix(
        preflight.digest,
        numerator ? UINT64_C(0x4e554d455241544f)
                  : UINT64_C(0x44454e4f4d494e41));
    exactDigestMix(preflight.digest, blocks.size());
    for (const auto& block : blocks) {
        DENSE_EXACT_CHECK(
            block.scale.integer_multiplier != 0u,
            std::string(label) +
                ": exact Gram positive scale is malformed");
        for (const auto value : block.row_major_raw_factors) {
            DENSE_EXACT_CHECK(
                std::isfinite(value),
                std::string(label) +
                    ": exact Gram raw factor is nonfinite");
        }
        for (const auto multiplier : block.row_multipliers) {
            DENSE_EXACT_CHECK(
                multiplier != 0u,
                std::string(label) +
                    ": exact Gram row multiplier is zero");
        }
        for (const Real term : block.scale.positive_sum_terms) {
            DENSE_EXACT_CHECK(
                std::isfinite(term) && term > Real{0},
                std::string(label) +
                    ": exact Gram sum term is not finite and positive");
        }
        for (const Real factor : block.scale.positive_product_factors) {
            DENSE_EXACT_CHECK(
                std::isfinite(factor) && factor > Real{0},
                std::string(label) +
                    ": exact Gram scale factor is not finite and positive");
        }

        exactDigestMix(preflight.digest, block.map_rows.size());
        for (const auto map_row : block.map_rows) {
            exactDigestMix(preflight.digest, map_row);
        }
        exactDigestMix(preflight.digest, block.factor_row_count);
        exactDigestMix(
            preflight.digest, block.row_major_raw_factors.size());
        for (const auto factor : block.row_major_raw_factors) {
            exactDigestMixReal(preflight.digest, factor);
        }
        exactDigestMix(preflight.digest, block.row_multipliers.size());
        for (const auto multiplier : block.row_multipliers) {
            exactDigestMix(preflight.digest, multiplier);
        }
        exactDigestMix(
            preflight.digest, block.scale.integer_multiplier);
        exactDigestMix(
            preflight.digest,
            block.scale.positive_sum_terms.size());
        for (const auto term : block.scale.positive_sum_terms) {
            exactDigestMixReal(preflight.digest, term);
        }
        exactDigestMix(
            preflight.digest,
            block.scale.positive_product_factors.size());
        for (const auto factor : block.scale.positive_product_factors) {
            exactDigestMixReal(preflight.digest, factor);
        }
    }
}

[[nodiscard]] std::size_t worstExactPsdUpdates(
    std::size_t n) noexcept
{
    const auto triangular = [](std::size_t value) noexcept {
        return value * (value + 1u) / 2u;
    };
    std::size_t result = n == 0u
        ? 0u
        : 3u * triangular(n - 1u);
    for (std::size_t remaining = 0u;
         remaining + 1u < n;
         ++remaining) {
        result += 4u * triangular(remaining);
    }
    return result;
}

void validateFactorizedWorkCap(
    const FactorizedPreflight& preflight,
    std::size_t n,
    std::string_view label,
    bool discover_common_kernel = false)
{
    std::size_t worst_updates = 0u;
    const auto add = [&](std::size_t value) {
        addCappedCount(
            worst_updates,
            value,
            kMaximumLegacyExactUpdates,
            label);
    };
    add(preflight.positive_scale_update_count);
    add(2u * preflight.transform_visit_count);
    add(3u * preflight.outer_pair_count);
    add((discover_common_kernel ? 71u : 68u) *
        worstExactPsdUpdates(n));
    add(66u * 2u * n * n);
    if (discover_common_kernel) {
        add(n * n);
    }
}

[[nodiscard]] ExactScaledIntegerMatrix materializeFactorizedMatrix(
    std::span<const DenseExactDyadicGramBlockView> blocks,
    DenseExactDyadicSparseMapView map,
    ExactBudget& budget,
    std::size_t& nonzero_outer_pairs,
    std::string_view label)
{
    const std::size_t n = map.output_dimension;
    std::vector<ExactDyadicScalar> matrix(
        checkedSquare(n, label));
    for (const auto& block : blocks) {
        ExactDyadicScalar weight_sum;
        for (const Real term : block.scale.positive_sum_terms) {
            weight_sum = exactDyadicSum(
                weight_sum,
                exactDyadicFromBinary64(term, label),
                budget,
                std::string(label) + " positive weight sum");
        }
        auto block_scale = exactDyadicProduct(
            exactDyadicFromUnsigned(
                block.scale.integer_multiplier, label),
            weight_sum,
            budget,
            std::string(label) + " positive integer scale");
        for (const Real factor :
             block.scale.positive_product_factors) {
            block_scale = exactDyadicProduct(
                block_scale,
                exactDyadicFromBinary64(factor, label),
                budget,
                std::string(label) + " positive product scale");
        }
        DENSE_EXACT_CHECK(
            block_scale.integer > 0,
            std::string(label) +
                ": exact Gram block scale is not positive");

        for (std::size_t factor_row = 0u;
             factor_row < block.factor_row_count;
             ++factor_row) {
            std::vector<ExactDyadicScalar> transformed(n);
            for (std::size_t local = 0u;
                 local < block.map_rows.size();
                 ++local) {
                const Real raw =
                    block.row_major_raw_factors[
                        factor_row * block.map_rows.size() + local];
                const auto raw_exact =
                    exactDyadicFromBinary64(raw, label);
                const auto map_row = block.map_rows[local];
                for (std::size_t entry_index =
                         map.row_offsets[map_row];
                     entry_index < map.row_offsets[map_row + 1u];
                     ++entry_index) {
                    const auto& entry = map.entries[entry_index];
                    const auto term = exactDyadicProduct(
                        raw_exact,
                        exactDyadicFromBinary64(
                            entry.coefficient, label),
                        budget,
                        std::string(label) +
                            " exact sparse transform product");
                    transformed[entry.output_coordinate] =
                        exactDyadicSum(
                            transformed[entry.output_coordinate],
                            term,
                            budget,
                            std::string(label) +
                                " exact sparse transform sum");
                }
            }
            const std::uint64_t row_multiplier =
                block.row_multipliers.empty()
                    ? UINT64_C(1)
                    : block.row_multipliers[factor_row];
            const auto row_scale = exactDyadicProduct(
                block_scale,
                exactDyadicFromUnsigned(row_multiplier, label),
                budget,
                std::string(label) + " exact Gram row scale");
            for (std::size_t row = 0u; row < n; ++row) {
                if (transformed[row].integer == 0) {
                    continue;
                }
                for (std::size_t column = row;
                     column < n;
                     ++column) {
                    if (transformed[column].integer == 0) {
                        continue;
                    }
                    DENSE_EXACT_CHECK(
                        nonzero_outer_pairs <
                            kMaximumExactOuterProducts,
                        std::string(label) +
                            ": exact nonzero outer-product count exceeds "
                            "its cap");
                    ++nonzero_outer_pairs;
                    auto term = exactDyadicProduct(
                        row_scale,
                        transformed[row],
                        budget,
                        std::string(label) + " exact Gram product");
                    term = exactDyadicProduct(
                        term,
                        transformed[column],
                        budget,
                        std::string(label) + " exact Gram product");
                    const auto index = row * n + column;
                    matrix[index] = exactDyadicSum(
                        matrix[index],
                        term,
                        budget,
                        std::string(label) + " exact Gram sum");
                }
            }
        }
    }
    for (std::size_t row = 0u; row < n; ++row) {
        for (std::size_t column = row + 1u;
             column < n;
             ++column) {
            matrix[column * n + row] =
                matrix[row * n + column];
        }
    }
    return integerizeExactDyadicMatrix(
        matrix, n, budget, label);
}

[[nodiscard]] bool scaledMatrixHasNonzero(
    const ExactScaledIntegerMatrix& matrix) noexcept
{
    return std::any_of(
        matrix.entries.begin(),
        matrix.entries.end(),
        [](const cpp_int& value) { return value != 0; });
}

[[nodiscard]] ExactScaledIntegerMatrix exactScaledMatrixSum(
    const ExactScaledIntegerMatrix& left,
    const ExactScaledIntegerMatrix& right,
    std::size_t n,
    ExactBudget& budget,
    std::string_view label)
{
    DENSE_EXACT_CHECK(
        left.entries.size() == n * n &&
            right.entries.size() == n * n,
        std::string(label) +
            ": exact scaled-matrix sum shape mismatch");
    const bool left_nonzero = scaledMatrixHasNonzero(left);
    const bool right_nonzero = scaledMatrixHasNonzero(right);
    if (!left_nonzero) {
        return right;
    }
    if (!right_nonzero) {
        return left;
    }

    ExactScaledIntegerMatrix result;
    result.exponent = std::min(left.exponent, right.exponent);
    result.entries.resize(n * n);
    const auto shifted =
        [&](const cpp_int& value, int exponent) {
            if (value == 0) {
                return cpp_int{0};
            }
            const auto shift = static_cast<std::size_t>(
                static_cast<long long>(exponent) -
                static_cast<long long>(result.exponent));
            DENSE_EXACT_CHECK(
                integerBits(value) + shift <= kMaximumIntegerBits,
                std::string(label) +
                    ": exact scaled-matrix sum exponent span exceeds "
                    "the bit cap");
            cpp_int term = value;
            term <<= shift;
            accountInteger(term, budget, label);
            return term;
        };
    for (std::size_t index = 0u;
         index < result.entries.size();
         ++index) {
        result.entries[index] = exactSum(
            shifted(left.entries[index], left.exponent),
            shifted(right.entries[index], right.exponent),
            budget,
            label);
    }
    const auto common = removeCommonPowerOfTwo(result.entries);
    result.exponent = checkedExactExponent(
        static_cast<long long>(result.exponent) +
            static_cast<long long>(common),
        label);
    return result;
}

[[nodiscard]] ExactScaledIntegerMatrix exactPrincipalSubmatrix(
    const ExactScaledIntegerMatrix& matrix,
    std::size_t input_dimension,
    std::span<const std::size_t> coordinates,
    std::string_view label)
{
    DENSE_EXACT_CHECK(
        matrix.entries.size() == input_dimension * input_dimension,
        std::string(label) +
            ": exact principal-submatrix input shape mismatch");
    ExactScaledIntegerMatrix result;
    result.exponent = matrix.exponent;
    result.entries.resize(coordinates.size() * coordinates.size());
    for (std::size_t row = 0u;
         row < coordinates.size();
         ++row) {
        DENSE_EXACT_CHECK(
            coordinates[row] < input_dimension &&
                (row == 0u ||
                 coordinates[row] > coordinates[row - 1u]),
            std::string(label) +
                ": exact principal coordinates are not strictly ordered");
        for (std::size_t column = 0u;
             column < coordinates.size();
             ++column) {
            result.entries[row * coordinates.size() + column] =
                matrix.entries[
                    coordinates[row] * input_dimension +
                    coordinates[column]];
        }
    }
    return result;
}

class ExactFactorizedPencil {
public:
    ExactFactorizedPencil(
        ExactScaledIntegerMatrix numerator,
        ExactScaledIntegerMatrix denominator,
        std::string_view label)
        : label_(label),
          numerator_(std::move(numerator)),
          denominator_(std::move(denominator))
    {
    }

    [[nodiscard]] std::vector<cpp_int> numeratorIntegers(
        ExactBudget&) const
    {
        return numerator_.entries;
    }

    [[nodiscard]] std::vector<cpp_int> denominatorIntegers(
        ExactBudget&) const
    {
        return denominator_.entries;
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

        int minimum_exponent = std::numeric_limits<int>::max();
        for (std::size_t index = 0u;
             index < numerator_.entries.size();
             ++index) {
            if (factor.significand != 0u &&
                denominator_.entries[index] != 0) {
                minimum_exponent = std::min(
                    minimum_exponent,
                    factor.exponent + denominator_.exponent);
            }
            if (numerator_.entries[index] != 0) {
                minimum_exponent = std::min(
                    minimum_exponent,
                    numerator_.exponent);
            }
        }

        std::vector<cpp_int> result(
            numerator_.entries.size(), cpp_int{0});
        if (minimum_exponent == std::numeric_limits<int>::max()) {
            return result;
        }
        for (std::size_t index = 0u;
             index < result.size();
             ++index) {
            cpp_int value{0};
            if (factor.significand != 0u &&
                denominator_.entries[index] != 0) {
                value = exactProduct(
                    cpp_int{factor.significand},
                    denominator_.entries[index],
                    budget,
                    label_ + " coefficient-denominator product");
                const auto shift = static_cast<std::size_t>(
                    factor.exponent + denominator_.exponent -
                    minimum_exponent);
                DENSE_EXACT_CHECK(
                    integerBits(value) + shift <=
                        kMaximumIntegerBits,
                    label_ +
                        ": exact pencil exponent span exceeds the bit cap");
                value <<= shift;
                accountInteger(value, budget, label_);
            }
            if (numerator_.entries[index] != 0) {
                cpp_int term = numerator_.entries[index];
                const auto shift = static_cast<std::size_t>(
                    numerator_.exponent - minimum_exponent);
                DENSE_EXACT_CHECK(
                    integerBits(term) + shift <=
                        kMaximumIntegerBits,
                    label_ +
                        ": exact pencil exponent span exceeds the bit cap");
                term <<= shift;
                value = exactDifference(
                    std::move(value),
                    term,
                    budget,
                    label_ +
                        " coefficient denominator minus numerator");
            }
            accountInteger(value, budget, label_);
            result[index] = std::move(value);
        }
        static_cast<void>(removeCommonPowerOfTwo(result));
        return result;
    }

private:
    std::string label_{};
    ExactScaledIntegerMatrix numerator_{};
    ExactScaledIntegerMatrix denominator_{};
};

enum class ExactPsdStatus : std::uint8_t {
    NotPositiveSemidefinite = 0u,
    PositiveSemidefinite = 1u,
};

struct ExactPsdResult {
    ExactPsdStatus status{
        ExactPsdStatus::NotPositiveSemidefinite};
    std::size_t rank{0u};
    std::vector<std::size_t> positive_coordinates{};
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
    std::vector<std::size_t> coordinate_permutation(n);
    for (std::size_t coordinate = 0u;
         coordinate < n;
         ++coordinate) {
        coordinate_permutation[coordinate] = coordinate;
    }
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
            coordinate_permutation.resize(rank);
            return {
                ExactPsdStatus::PositiveSemidefinite,
                rank,
                std::move(coordinate_permutation)};
        }

        symmetricSwap(
            matrix, n, step, pivot_coordinate);
        std::swap(
            coordinate_permutation[step],
            coordinate_permutation[pivot_coordinate]);
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
    coordinate_permutation.resize(rank);
    return {
        ExactPsdStatus::PositiveSemidefinite,
        rank,
        std::move(coordinate_permutation)};
}

[[nodiscard]] Real nonnegativeBinary64FromOrderedBits(
    std::uint64_t bits)
{
    DENSE_EXACT_CHECK(
        bits <= UINT64_C(0x7fefffffffffffff),
        "exact dyadic binary64 search produced an invalid bit pattern");
    const Real value = bitwiseCopy<Real>(bits);
    DENSE_EXACT_CHECK(
        std::isfinite(value) && value >= Real{0},
        "exact dyadic binary64 search produced an invalid value");
    return value;
}

template <typename Pencil>
[[nodiscard]] DenseExactDyadicSpdGeneralizedUpperBound
proveExactGeneralizedUpperBound(
    const Pencil& pencil,
    std::size_t n,
    const std::string& label_text,
    ExactBudget& budget)
{
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
            ": exact denominator is not positive definite (status=" +
            std::to_string(
                static_cast<unsigned int>(denominator_psd.status)) +
            ", rank=" + std::to_string(denominator_psd.rank) +
            ", dimension=" + std::to_string(n) + ")");
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
    return proveExactGeneralizedUpperBound(
        pencil, n, label_text, budget);
}

DenseExactDyadicSpdGeneralizedUpperBound
dense_exact_dyadic_spd_generalized_factorized_upper_bound(
    std::span<const DenseExactDyadicGramBlockView> numerator,
    std::span<const DenseExactDyadicGramBlockView> denominator,
    DenseExactDyadicSparseMapView raw_to_quotient,
    std::string_view label)
{
    const std::string label_text(label);
    FactorizedPreflight preflight;
    preflightSparseMapShape(
        raw_to_quotient,
        preflight,
        label_text + " sparse map");
    addModeledBytes(
        preflight,
        numerator.size(),
        sizeof(DenseExactDyadicGramBlockView),
        label_text);
    addModeledBytes(
        preflight,
        denominator.size(),
        sizeof(DenseExactDyadicGramBlockView),
        label_text);
    preflightGramBlockShapes(
        numerator,
        raw_to_quotient,
        true,
        preflight,
        label_text + " numerator");
    preflightGramBlockShapes(
        denominator,
        raw_to_quotient,
        false,
        preflight,
        label_text + " denominator");
    validateSparseMapOffsets(
        raw_to_quotient,
        label_text + " sparse map");
    preflightGramBlockTransformWork(
        numerator,
        raw_to_quotient,
        preflight,
        label_text + " numerator");
    preflightGramBlockTransformWork(
        denominator,
        raw_to_quotient,
        preflight,
        label_text + " denominator");
    validateFactorizedWorkCap(
        preflight,
        raw_to_quotient.output_dimension,
        label_text + " modeled exact work");
    validateSparseMapEntriesAndDigest(
        raw_to_quotient,
        preflight,
        label_text + " sparse map");
    validateGramBlockContentsAndDigest(
        numerator,
        true,
        preflight,
        label_text + " numerator");
    validateGramBlockContentsAndDigest(
        denominator,
        false,
        preflight,
        label_text + " denominator");

    const auto populate_factorized_diagnostics =
        [&](DenseExactDyadicSpdGeneralizedUpperBound& result,
            std::size_t materialization_updates,
            std::size_t nonzero_outer_pairs) {
            result.proof_input =
                DenseExactDyadicProofInput::
                    FactorizedBinary64PositiveForm;
            result.exact_factorized_materialization_proven = true;
            result.exact_sparse_map_applied = true;
            result.numerator_gram_block_count =
                preflight.numerator_block_count;
            result.denominator_gram_block_count =
                preflight.denominator_block_count;
            result.numerator_gram_row_count =
                preflight.numerator_row_count;
            result.denominator_gram_row_count =
                preflight.denominator_row_count;
            result.numerator_weight_term_count =
                preflight.numerator_weight_term_count;
            result.denominator_weight_term_count =
                preflight.denominator_weight_term_count;
            result.transform_entry_count =
                raw_to_quotient.entries.size();
            result.exact_transform_visit_count =
                preflight.transform_visit_count;
            result.exact_nonzero_outer_pair_count =
                nonzero_outer_pairs;
            result.factor_materialization_update_count =
                materialization_updates;
            result.modeled_input_bytes =
                preflight.modeled_input_bytes;
            result.factorized_input_digest = preflight.digest;
            result.factorized_input_dimension =
                raw_to_quotient.output_dimension;
            result.exact_common_kernel_proven = true;
            result.exact_common_kernel_quotient_applied = false;
            result.exact_common_kernel_nullity = 0u;
            result.exact_common_kernel_eliminated_coordinates.clear();
        };

    if (raw_to_quotient.output_dimension == 0u) {
        // An empty principal quotient is vacuously SPD/PSD. Validation above
        // proves that the sparse map has no output entries and binds every raw
        // positive-form primitive into the factorized-input digest.
        DenseExactDyadicSpdGeneralizedUpperBound result;
        result.applied = true;
        result.denominator_positive_definite_proven = true;
        result.numerator_positive_semidefinite_proven = true;
        result.upper_inequality_proven = true;
        populate_factorized_diagnostics(result, 0u, 0u);
        return result;
    }

    ExactBudget budget;
    std::size_t nonzero_outer_pairs = 0u;
    auto numerator_matrix = materializeFactorizedMatrix(
        numerator,
        raw_to_quotient,
        budget,
        nonzero_outer_pairs,
        label_text + " numerator");
    auto denominator_matrix = materializeFactorizedMatrix(
        denominator,
        raw_to_quotient,
        budget,
        nonzero_outer_pairs,
        label_text + " denominator");
    const auto materialization_updates = budget.update_count;
    ExactFactorizedPencil pencil(
        std::move(numerator_matrix),
        std::move(denominator_matrix),
        label);
    auto result = proveExactGeneralizedUpperBound(
        pencil,
        raw_to_quotient.output_dimension,
        label_text,
        budget);
    populate_factorized_diagnostics(
        result, materialization_updates, nonzero_outer_pairs);
    return result;
}

DenseExactDyadicSpdGeneralizedUpperBound
dense_exact_dyadic_psd_generalized_factorized_upper_bound(
    std::span<const DenseExactDyadicGramBlockView> numerator,
    std::span<const DenseExactDyadicGramBlockView> denominator,
    DenseExactDyadicSparseMapView raw_to_quotient,
    std::string_view label)
{
    const std::string label_text(label);
    FactorizedPreflight preflight;
    preflightSparseMapShape(
        raw_to_quotient,
        preflight,
        label_text + " sparse map");
    addModeledBytes(
        preflight,
        numerator.size(),
        sizeof(DenseExactDyadicGramBlockView),
        label_text);
    addModeledBytes(
        preflight,
        denominator.size(),
        sizeof(DenseExactDyadicGramBlockView),
        label_text);
    preflightGramBlockShapes(
        numerator,
        raw_to_quotient,
        true,
        preflight,
        label_text + " numerator");
    preflightGramBlockShapes(
        denominator,
        raw_to_quotient,
        false,
        preflight,
        label_text + " denominator");
    validateSparseMapOffsets(
        raw_to_quotient,
        label_text + " sparse map");
    preflightGramBlockTransformWork(
        numerator,
        raw_to_quotient,
        preflight,
        label_text + " numerator");
    preflightGramBlockTransformWork(
        denominator,
        raw_to_quotient,
        preflight,
        label_text + " denominator");
    validateFactorizedWorkCap(
        preflight,
        raw_to_quotient.output_dimension,
        label_text + " modeled exact work",
        true);
    validateSparseMapEntriesAndDigest(
        raw_to_quotient,
        preflight,
        label_text + " sparse map");
    validateGramBlockContentsAndDigest(
        numerator,
        true,
        preflight,
        label_text + " numerator");
    validateGramBlockContentsAndDigest(
        denominator,
        false,
        preflight,
        label_text + " denominator");

    const auto populate_factorized_diagnostics =
        [&](DenseExactDyadicSpdGeneralizedUpperBound& result,
            std::size_t materialization_updates,
            std::size_t nonzero_outer_pairs,
            std::span<const std::size_t> eliminated_coordinates) {
            result.proof_input =
                DenseExactDyadicProofInput::
                    FactorizedBinary64PositiveForm;
            result.exact_factorized_materialization_proven = true;
            result.exact_sparse_map_applied = true;
            result.numerator_gram_block_count =
                preflight.numerator_block_count;
            result.denominator_gram_block_count =
                preflight.denominator_block_count;
            result.numerator_gram_row_count =
                preflight.numerator_row_count;
            result.denominator_gram_row_count =
                preflight.denominator_row_count;
            result.numerator_weight_term_count =
                preflight.numerator_weight_term_count;
            result.denominator_weight_term_count =
                preflight.denominator_weight_term_count;
            result.transform_entry_count =
                raw_to_quotient.entries.size();
            result.exact_transform_visit_count =
                preflight.transform_visit_count;
            result.exact_nonzero_outer_pair_count =
                nonzero_outer_pairs;
            result.factor_materialization_update_count =
                materialization_updates;
            result.modeled_input_bytes =
                preflight.modeled_input_bytes;
            result.factorized_input_digest = preflight.digest;
            result.factorized_input_dimension =
                raw_to_quotient.output_dimension;
            result.exact_common_kernel_proven = true;
            result.exact_common_kernel_nullity =
                eliminated_coordinates.size();
            result.exact_common_kernel_quotient_applied =
                !eliminated_coordinates.empty();
            result.exact_common_kernel_eliminated_coordinates.assign(
                eliminated_coordinates.begin(),
                eliminated_coordinates.end());
        };

    if (raw_to_quotient.output_dimension == 0u) {
        DenseExactDyadicSpdGeneralizedUpperBound result;
        result.applied = true;
        result.denominator_positive_definite_proven = true;
        result.numerator_positive_semidefinite_proven = true;
        result.upper_inequality_proven = true;
        populate_factorized_diagnostics(result, 0u, 0u, {});
        return result;
    }

    const std::size_t input_dimension =
        raw_to_quotient.output_dimension;
    ExactBudget budget;
    std::size_t nonzero_outer_pairs = 0u;
    auto numerator_matrix = materializeFactorizedMatrix(
        numerator,
        raw_to_quotient,
        budget,
        nonzero_outer_pairs,
        label_text + " numerator");
    auto denominator_matrix = materializeFactorizedMatrix(
        denominator,
        raw_to_quotient,
        budget,
        nonzero_outer_pairs,
        label_text + " denominator");
    const auto materialization_updates = budget.update_count;

    auto denominator_psd = exactPositiveSemidefiniteRank(
        denominator_matrix.entries,
        input_dimension,
        budget,
        label_text + " denominator common-kernel preflight");
    DENSE_EXACT_CHECK(
        denominator_psd.status ==
            ExactPsdStatus::PositiveSemidefinite,
        label_text +
            ": exact factorized denominator is not positive semidefinite");
    const auto numerator_psd = exactPositiveSemidefiniteRank(
        numerator_matrix.entries,
        input_dimension,
        budget,
        label_text + " numerator common-kernel preflight");
    DENSE_EXACT_CHECK(
        numerator_psd.status ==
            ExactPsdStatus::PositiveSemidefinite,
        label_text +
            ": exact factorized numerator is not positive semidefinite");
    auto pencil_sum = exactScaledMatrixSum(
        numerator_matrix,
        denominator_matrix,
        input_dimension,
        budget,
        label_text + " common-kernel rank sum");
    const auto sum_psd = exactPositiveSemidefiniteRank(
        std::move(pencil_sum.entries),
        input_dimension,
        budget,
        label_text + " common-kernel rank sum");
    DENSE_EXACT_CHECK(
        sum_psd.status == ExactPsdStatus::PositiveSemidefinite,
        label_text +
            ": exact factorized positive-form sum is not positive "
            "semidefinite");
    DENSE_EXACT_CHECK(
        sum_psd.rank == denominator_psd.rank,
        label_text +
            ": exact numerator acts on the denominator kernel "
            "(denominator_rank=" +
            std::to_string(denominator_psd.rank) +
            ", combined_rank=" + std::to_string(sum_psd.rank) +
            ", dimension=" + std::to_string(input_dimension) + ")");

    auto retained_coordinates =
        std::move(denominator_psd.positive_coordinates);
    std::sort(
        retained_coordinates.begin(),
        retained_coordinates.end());
    std::vector<std::size_t> eliminated_coordinates;
    eliminated_coordinates.reserve(
        input_dimension - retained_coordinates.size());
    std::size_t retained_index = 0u;
    for (std::size_t coordinate = 0u;
         coordinate < input_dimension;
         ++coordinate) {
        if (retained_index < retained_coordinates.size() &&
            retained_coordinates[retained_index] == coordinate) {
            ++retained_index;
        } else {
            eliminated_coordinates.push_back(coordinate);
        }
    }
    DENSE_EXACT_CHECK(
        retained_index == retained_coordinates.size() &&
            retained_coordinates.size() == denominator_psd.rank,
        label_text +
            ": exact common-kernel coordinate rank is inconsistent");

    if (retained_coordinates.empty()) {
        DenseExactDyadicSpdGeneralizedUpperBound result;
        result.applied = true;
        result.denominator_positive_definite_proven = true;
        result.numerator_positive_semidefinite_proven = true;
        result.upper_inequality_proven = true;
        result.psd_oracle_calls = 3u;
        result.exact_update_count = budget.update_count;
        result.maximum_integer_bits = budget.maximum_integer_bits;
        populate_factorized_diagnostics(
            result,
            materialization_updates,
            nonzero_outer_pairs,
            eliminated_coordinates);
        return result;
    }

    auto quotient_numerator = exactPrincipalSubmatrix(
        numerator_matrix,
        input_dimension,
        retained_coordinates,
        label_text + " numerator common-kernel quotient");
    numerator_matrix = {};
    auto quotient_denominator = exactPrincipalSubmatrix(
        denominator_matrix,
        input_dimension,
        retained_coordinates,
        label_text + " denominator common-kernel quotient");
    denominator_matrix = {};
    ExactFactorizedPencil quotient_pencil(
        std::move(quotient_numerator),
        std::move(quotient_denominator),
        label);
    auto result = proveExactGeneralizedUpperBound(
        quotient_pencil,
        retained_coordinates.size(),
        label_text + " exact common-kernel coordinate quotient",
        budget);
    result.psd_oracle_calls += 3u;
    result.exact_update_count = budget.update_count;
    result.maximum_integer_bits = budget.maximum_integer_bits;
    populate_factorized_diagnostics(
        result,
        materialization_updates,
        nonzero_outer_pairs,
        eliminated_coordinates);
    return result;
}

} // namespace svmp::FE::math
