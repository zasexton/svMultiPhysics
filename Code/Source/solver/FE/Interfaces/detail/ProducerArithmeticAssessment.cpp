/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/detail/ProducerArithmeticAssessment.h"

#include <array>
#include <bit>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace svmp::FE::interfaces::detail {
namespace {

#if defined(__GNUC__) && !defined(__clang__)
#  define SVMP_PRODUCER_ARITHMETIC_ENTRY __attribute__((noipa))
#else
#  define SVMP_PRODUCER_ARITHMETIC_ENTRY
#endif

constexpr std::size_t kLimbBits = 64u;
constexpr std::size_t kLimbCount = 68u;
constexpr std::size_t kMaximumOperandLimbs = 33u;
constexpr unsigned kBinary64Scale = 1074u;
constexpr std::uint64_t kSignMask = UINT64_C(0x8000000000000000);
constexpr std::uint64_t kFractionMask = UINT64_C(0x000fffffffffffff);
constexpr std::uint64_t kExponentMask = UINT64_C(0x7ff0000000000000);
constexpr std::uint64_t kMinimumNormal = UINT64_C(0x0010000000000000);
constexpr std::uint64_t kMaximumFinite = UINT64_C(0x7fefffffffffffff);
constexpr std::uint64_t kPositiveInfinity = UINT64_C(0x7ff0000000000000);
constexpr std::uint64_t kOne = UINT64_C(0x3ff0000000000000);
constexpr std::uint64_t kHelperDenominatorMargin =
    UINT64_C(0x39b4484bfeebc2a0); // Stored binary64 1e-30.

struct UInt4352 {
    std::array<std::uint64_t, kLimbCount> limbs{};
    std::size_t used{0u};
};

enum class TargetKind { Direct, ShiftedCandidate, CandidateProduct, Square };

struct SearchTarget {
    TargetKind kind{TargetKind::Direct};
    UInt4352 fixed{};
    UInt4352 factor{};
};

struct SignedInteger {
    UInt4352 magnitude{};
    int sign{0};
};

struct DecodedScalar {
    UInt4352 magnitude{};
    int sign{0};
};

struct MagnitudeBracket {
    std::uint64_t lower{0u};
    std::uint64_t upper{0u};
    bool valid{false};
};

template <typename T>
[[nodiscard]] std::uint64_t bitsOf(T value) noexcept
{
    if constexpr (sizeof(T) == sizeof(std::uint64_t) &&
                  std::is_trivially_copyable_v<T>) {
        return std::bit_cast<std::uint64_t>(value);
    } else {
        return 0u;
    }
}

template <typename T>
[[nodiscard]] T valueFromBits(std::uint64_t bits) noexcept
{
    if constexpr (sizeof(T) == sizeof(std::uint64_t) &&
                  std::is_trivially_copyable_v<T>) {
        return std::bit_cast<T>(bits);
    } else {
        return T{};
    }
}

[[nodiscard]] bool supportedRepresentation() noexcept
{
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && \
    defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ && __GNUC__ == 12 && \
    __GNUC_MINOR__ == 4
    constexpr bool properties =
        CHAR_BIT == 8 && sizeof(std::uint64_t) == 8u &&
        sizeof(Real) == 8u && std::numeric_limits<Real>::radix == 2 &&
        std::numeric_limits<Real>::digits == 53 &&
        std::numeric_limits<Real>::min_exponent == -1021 &&
        std::numeric_limits<Real>::max_exponent == 1024 &&
        std::numeric_limits<Real>::is_iec559 &&
        std::numeric_limits<Real>::has_denorm == std::denorm_present &&
        FLT_EVAL_METHOD == 0;
    if constexpr (!properties) {
        return false;
    }
    return bitsOf(Real{0}) == UINT64_C(0x0000000000000000) &&
           bitsOf(Real{1}) == UINT64_C(0x3ff0000000000000) &&
           bitsOf(Real{-1}) == UINT64_C(0xbff0000000000000) &&
           bitsOf(std::numeric_limits<Real>::min()) == kMinimumNormal &&
           bitsOf(std::numeric_limits<Real>::max()) == kMaximumFinite;
#else
    return false;
#endif
}

void clear(UInt4352& value) noexcept
{
    for (std::size_t index = 0u; index < kLimbCount; ++index) {
        value.limbs[index] = 0u;
    }
    value.used = 0u;
}

void normalize(UInt4352& value) noexcept
{
    while (value.used > 0u && value.limbs[value.used - 1u] == 0u) {
        --value.used;
    }
}

[[nodiscard]] bool isZero(const UInt4352& value) noexcept
{
    return value.used == 0u;
}

[[nodiscard]] int compare(const UInt4352& left,
                          const UInt4352& right) noexcept
{
    if (left.used != right.used) {
        return left.used < right.used ? -1 : 1;
    }
    for (std::size_t index = left.used; index > 0u; --index) {
        const auto limb_index = index - 1u;
        if (left.limbs[limb_index] != right.limbs[limb_index]) {
            return left.limbs[limb_index] < right.limbs[limb_index] ? -1 : 1;
        }
    }
    return 0;
}

[[nodiscard]] bool add(const UInt4352& left,
                       const UInt4352& right,
                       UInt4352& result) noexcept
{
    clear(result);
    const auto count = left.used > right.used ? left.used : right.used;
    if (count >= kLimbCount) {
        return false;
    }
    __uint128_t carry = 0u;
    for (std::size_t index = 0u; index < count; ++index) {
        const auto left_limb = index < left.used ? left.limbs[index] : 0u;
        const auto right_limb = index < right.used ? right.limbs[index] : 0u;
        const __uint128_t sum = static_cast<__uint128_t>(left_limb) +
                                static_cast<__uint128_t>(right_limb) + carry;
        result.limbs[index] = static_cast<std::uint64_t>(sum);
        carry = sum >> kLimbBits;
    }
    result.used = count;
    if (carry != 0u) {
        result.limbs[count] = static_cast<std::uint64_t>(carry);
        result.used = count + 1u;
    }
    return true;
}

[[nodiscard]] bool subtract(const UInt4352& left,
                            const UInt4352& right,
                            UInt4352& result) noexcept
{
    if (compare(left, right) < 0) {
        return false;
    }
    clear(result);
    std::uint64_t borrow = 0u;
    for (std::size_t index = 0u; index < left.used; ++index) {
        const auto left_limb = left.limbs[index];
        const auto right_limb = index < right.used ? right.limbs[index] : 0u;
        result.limbs[index] =
            static_cast<std::uint64_t>(left_limb - right_limb - borrow);
        const bool borrowed_from_right = left_limb < right_limb;
        const bool borrowed_for_carry =
            borrow != 0u && left_limb == right_limb;
        borrow = borrowed_from_right || borrowed_for_carry ? 1u : 0u;
    }
    result.used = left.used;
    normalize(result);
    return borrow == 0u;
}

[[nodiscard]] bool shiftLeft(const UInt4352& input,
                             unsigned shift,
                             UInt4352& result) noexcept
{
    clear(result);
    if (isZero(input)) {
        return true;
    }
    const auto word_shift = static_cast<std::size_t>(shift / kLimbBits);
    const auto bit_shift = shift % kLimbBits;
    if (word_shift >= kLimbCount ||
        input.used > kLimbCount - word_shift) {
        return false;
    }
    for (std::size_t index = 0u; index < input.used; ++index) {
        const auto destination = index + word_shift;
        result.limbs[destination] |= input.limbs[index] << bit_shift;
        if (bit_shift != 0u) {
            const auto carry = input.limbs[index] >> (kLimbBits - bit_shift);
            if (carry != 0u) {
                if (destination + 1u >= kLimbCount) {
                    return false;
                }
                result.limbs[destination + 1u] |= carry;
            }
        }
    }
    result.used = input.used + word_shift + (bit_shift == 0u ? 0u : 1u);
    if (result.used > kLimbCount) {
        return false;
    }
    normalize(result);
    return true;
}

[[nodiscard]] bool multiply(const UInt4352& left,
                            const UInt4352& right,
                            UInt4352& result) noexcept
{
    clear(result);
    if (isZero(left) || isZero(right)) {
        return true;
    }
    if (left.used > kMaximumOperandLimbs ||
        right.used > kMaximumOperandLimbs ||
        left.used + right.used > kLimbCount) {
        return false;
    }
    for (std::size_t left_index = 0u;
         left_index < left.used;
         ++left_index) {
        if (left.limbs[left_index] == 0u) {
            continue;
        }
        __uint128_t carry = 0u;
        for (std::size_t right_index = 0u;
             right_index < right.used;
             ++right_index) {
            const auto destination = left_index + right_index;
            const __uint128_t product =
                static_cast<__uint128_t>(left.limbs[left_index]) *
                    static_cast<__uint128_t>(right.limbs[right_index]) +
                static_cast<__uint128_t>(result.limbs[destination]) + carry;
            result.limbs[destination] = static_cast<std::uint64_t>(product);
            carry = product >> kLimbBits;
        }
        const auto carry_index = left_index + right.used;
        if (carry != 0u) {
            result.limbs[carry_index] = static_cast<std::uint64_t>(carry);
        }
    }
    result.used = left.used + right.used;
    normalize(result);
    return true;
}

[[nodiscard]] bool decodeMagnitude(std::uint64_t magnitude_bits,
                                   UInt4352& result) noexcept
{
    clear(result);
    if (magnitude_bits > kMaximumFinite) {
        return false;
    }
    const auto exponent = magnitude_bits >> 52u;
    const auto fraction = magnitude_bits & kFractionMask;
    if (exponent == 0u) {
        if (fraction != 0u) {
            result.limbs[0] = fraction;
            result.used = 1u;
        }
        return true;
    }
    UInt4352 significand;
    clear(significand);
    significand.limbs[0] = (UINT64_C(1) << 52u) | fraction;
    significand.used = 1u;
    return shiftLeft(significand,
                     static_cast<unsigned>(exponent - 1u),
                     result);
}

[[nodiscard]] bool decodeScalar(std::uint64_t bits,
                                DecodedScalar& result) noexcept
{
    const auto magnitude_bits = bits & ~kSignMask;
    if (!decodeMagnitude(magnitude_bits, result.magnitude)) {
        return false;
    }
    if (isZero(result.magnitude)) {
        result.sign = 0;
    } else {
        result.sign = (bits & kSignMask) == 0u ? 1 : -1;
    }
    return true;
}

[[nodiscard]] bool validSubmittedScalarBits(std::uint64_t bits) noexcept
{
    const auto magnitude = bits & ~kSignMask;
    const auto exponent = magnitude & kExponentMask;
    const auto fraction = magnitude & kFractionMask;
    if (exponent == kExponentMask) {
        return false;
    }
    return exponent != 0u || fraction == 0u;
}

[[nodiscard]] int compareRealBits(std::uint64_t left,
                                  std::uint64_t right) noexcept
{
    const auto left_magnitude = left & ~kSignMask;
    const auto right_magnitude = right & ~kSignMask;
    if (left_magnitude == 0u && right_magnitude == 0u) {
        return 0;
    }
    const bool left_negative = (left & kSignMask) != 0u;
    const bool right_negative = (right & kSignMask) != 0u;
    if (left_negative != right_negative) {
        return left_negative ? -1 : 1;
    }
    if (left_magnitude == right_magnitude) {
        return 0;
    }
    if (left_negative) {
        return left_magnitude > right_magnitude ? -1 : 1;
    }
    return left_magnitude < right_magnitude ? -1 : 1;
}

[[nodiscard]] bool validInterval(ArithmeticInterval interval) noexcept
{
    const auto lower = bitsOf(interval.lower);
    const auto upper = bitsOf(interval.upper);
    return validSubmittedScalarBits(lower) &&
           validSubmittedScalarBits(upper) &&
           compareRealBits(lower, upper) <= 0;
}

[[nodiscard]] bool addSigned(const DecodedScalar& left,
                             const DecodedScalar& right,
                             SignedInteger& result) noexcept
{
    clear(result.magnitude);
    result.sign = 0;
    if (left.sign == 0) {
        result.magnitude = right.magnitude;
        result.sign = right.sign;
        return true;
    }
    if (right.sign == 0) {
        result.magnitude = left.magnitude;
        result.sign = left.sign;
        return true;
    }
    if (left.sign == right.sign) {
        if (!add(left.magnitude, right.magnitude, result.magnitude)) {
            return false;
        }
        result.sign = left.sign;
        return true;
    }
    const auto ordering = compare(left.magnitude, right.magnitude);
    if (ordering == 0) {
        return true;
    }
    if (ordering > 0) {
        if (!subtract(left.magnitude, right.magnitude, result.magnitude)) {
            return false;
        }
        result.sign = left.sign;
    } else {
        if (!subtract(right.magnitude, left.magnitude, result.magnitude)) {
            return false;
        }
        result.sign = right.sign;
    }
    return true;
}

[[nodiscard]] bool compareCandidate(std::uint64_t candidate_bits,
                                    const SearchTarget& target,
                                    int& ordering) noexcept
{
    UInt4352 candidate;
    if (!decodeMagnitude(candidate_bits, candidate)) {
        return false;
    }
    UInt4352 evaluated;
    switch (target.kind) {
    case TargetKind::Direct:
        ordering = compare(candidate, target.fixed);
        return true;
    case TargetKind::ShiftedCandidate:
        if (!shiftLeft(candidate, kBinary64Scale, evaluated)) {
            return false;
        }
        ordering = compare(evaluated, target.fixed);
        return true;
    case TargetKind::CandidateProduct:
        if (!multiply(candidate, target.factor, evaluated)) {
            return false;
        }
        ordering = compare(evaluated, target.fixed);
        return true;
    case TargetKind::Square:
        if (!multiply(candidate, candidate, evaluated)) {
            return false;
        }
        ordering = compare(evaluated, target.fixed);
        return true;
    }
    return false;
}

[[nodiscard]] MagnitudeBracket searchMagnitude(
    const SearchTarget& target) noexcept
{
    if (isZero(target.fixed)) {
        return {0u, 0u, true};
    }
    int maximum_ordering = 0;
    if (!compareCandidate(kMaximumFinite, target, maximum_ordering) ||
        maximum_ordering < 0) {
        return {};
    }
    if (maximum_ordering == 0) {
        return {kMaximumFinite, kMaximumFinite, true};
    }

    std::uint64_t lower = 0u;
    std::uint64_t upper = kPositiveInfinity;
    unsigned iterations = 0u;
    while (upper - lower > 1u && iterations < 64u) {
        const auto middle = lower + (upper - lower) / 2u;
        int ordering = 0;
        if (!compareCandidate(middle, target, ordering)) {
            return {};
        }
        if (ordering <= 0) {
            lower = middle;
        } else {
            upper = middle;
        }
        ++iterations;
    }
    if (upper - lower != 1u) {
        return {};
    }
    int lower_ordering = 0;
    if (!compareCandidate(lower, target, lower_ordering)) {
        return {};
    }
    if (lower_ordering == 0) {
        if (lower != 0u && lower < kMinimumNormal) {
            return {};
        }
        return {lower, lower, true};
    }
    if (lower == 0u || upper > kMaximumFinite ||
        lower < kMinimumNormal || upper < kMinimumNormal) {
        return {};
    }
    return {lower, upper, true};
}

[[nodiscard]] IntervalAssessment unavailable(
    ArithmeticFailure failure) noexcept
{
    IntervalAssessment result;
    result.failure = failure;
    return result;
}

[[nodiscard]] IntervalAssessment fromMagnitudeBracket(
    MagnitudeBracket bracket,
    int sign) noexcept
{
    if (!bracket.valid) {
        return unavailable(ArithmeticFailure::ArithmeticRange);
    }
    IntervalAssessment result;
    if (sign < 0 && bracket.upper != 0u) {
        result.interval.lower =
            valueFromBits<Real>(bracket.upper | kSignMask);
        result.interval.upper =
            valueFromBits<Real>(bracket.lower | kSignMask);
    } else {
        result.interval.lower = valueFromBits<Real>(bracket.lower);
        result.interval.upper = valueFromBits<Real>(bracket.upper);
    }
    result.failure = ArithmeticFailure::None;
    return result;
}

[[nodiscard]] IntervalAssessment scalarAdd(std::uint64_t left_bits,
                                           std::uint64_t right_bits,
                                           bool subtract_right) noexcept
{
    DecodedScalar left;
    DecodedScalar right;
    if (!decodeScalar(left_bits, left) || !decodeScalar(right_bits, right)) {
        return unavailable(ArithmeticFailure::InvalidInput);
    }
    if (subtract_right) {
        right.sign = -right.sign;
    }
    SignedInteger exact;
    if (!addSigned(left, right, exact)) {
        return unavailable(ArithmeticFailure::ArithmeticRange);
    }
    SearchTarget target;
    target.kind = TargetKind::Direct;
    target.fixed = exact.magnitude;
    return fromMagnitudeBracket(searchMagnitude(target), exact.sign);
}

[[nodiscard]] IntervalAssessment scalarMultiply(
    std::uint64_t left_bits,
    std::uint64_t right_bits) noexcept
{
    DecodedScalar left;
    DecodedScalar right;
    if (!decodeScalar(left_bits, left) || !decodeScalar(right_bits, right)) {
        return unavailable(ArithmeticFailure::InvalidInput);
    }
    UInt4352 exact_product;
    if (!multiply(left.magnitude, right.magnitude, exact_product)) {
        return unavailable(ArithmeticFailure::ArithmeticRange);
    }
    SearchTarget target;
    target.kind = TargetKind::ShiftedCandidate;
    target.fixed = exact_product;
    const auto sign = left.sign == 0 || right.sign == 0
                          ? 0
                          : left.sign * right.sign;
    return fromMagnitudeBracket(searchMagnitude(target), sign);
}

[[nodiscard]] IntervalAssessment scalarDivide(
    std::uint64_t numerator_bits,
    std::uint64_t denominator_bits) noexcept
{
    DecodedScalar numerator;
    DecodedScalar denominator;
    if (!decodeScalar(numerator_bits, numerator) ||
        !decodeScalar(denominator_bits, denominator)) {
        return unavailable(ArithmeticFailure::InvalidInput);
    }
    if (denominator.sign == 0) {
        return unavailable(ArithmeticFailure::UnresolvedDenominator);
    }
    SearchTarget target;
    target.kind = TargetKind::CandidateProduct;
    if (!shiftLeft(numerator.magnitude, kBinary64Scale, target.fixed)) {
        return unavailable(ArithmeticFailure::ArithmeticRange);
    }
    target.factor = denominator.magnitude;
    const auto sign = numerator.sign == 0
                          ? 0
                          : numerator.sign * denominator.sign;
    return fromMagnitudeBracket(searchMagnitude(target), sign);
}

[[nodiscard]] IntervalAssessment scalarSqrt(std::uint64_t input_bits) noexcept
{
    DecodedScalar input;
    if (!decodeScalar(input_bits, input) || input.sign < 0) {
        return unavailable(ArithmeticFailure::InvalidInput);
    }
    SearchTarget target;
    target.kind = TargetKind::Square;
    if (!shiftLeft(input.magnitude, kBinary64Scale, target.fixed)) {
        return unavailable(ArithmeticFailure::ArithmeticRange);
    }
    return fromMagnitudeBracket(searchMagnitude(target), input.sign);
}

[[nodiscard]] std::uint64_t smallerBits(std::uint64_t left,
                                        std::uint64_t right) noexcept
{
    return compareRealBits(left, right) <= 0 ? left : right;
}

[[nodiscard]] std::uint64_t largerBits(std::uint64_t left,
                                       std::uint64_t right) noexcept
{
    return compareRealBits(left, right) >= 0 ? left : right;
}

[[nodiscard]] IntervalAssessment combineFour(
    IntervalOperation operation,
    ArithmeticInterval left,
    ArithmeticInterval right) noexcept
{
    const std::array<std::uint64_t, 2> left_bits{
        bitsOf(left.lower), bitsOf(left.upper)};
    const std::array<std::uint64_t, 2> right_bits{
        bitsOf(right.lower), bitsOf(right.upper)};
    const auto left_count = left_bits[0] == left_bits[1] ? 1u : 2u;
    const auto right_count = right_bits[0] == right_bits[1] ? 1u : 2u;
    bool initialized = false;
    std::uint64_t lower = 0u;
    std::uint64_t upper = 0u;
    for (std::size_t left_index = 0u; left_index < left_count;
         ++left_index) {
        for (std::size_t right_index = 0u; right_index < right_count;
             ++right_index) {
            const auto left_endpoint = left_bits[left_index];
            const auto right_endpoint = right_bits[right_index];
            const auto scalar = operation == IntervalOperation::Multiply
                                    ? scalarMultiply(left_endpoint,
                                                     right_endpoint)
                                    : scalarDivide(left_endpoint,
                                                   right_endpoint);
            if (!scalar.available()) {
                return scalar;
            }
            const auto scalar_lower = bitsOf(scalar.interval.lower);
            const auto scalar_upper = bitsOf(scalar.interval.upper);
            if (!initialized) {
                lower = scalar_lower;
                upper = scalar_upper;
                initialized = true;
            } else {
                lower = smallerBits(lower, scalar_lower);
                upper = largerBits(upper, scalar_upper);
            }
        }
    }
    IntervalAssessment result;
    result.interval.lower = valueFromBits<Real>(lower);
    result.interval.upper = valueFromBits<Real>(upper);
    result.failure = ArithmeticFailure::None;
    return result;
}

[[nodiscard]] std::uint64_t absoluteBits(std::uint64_t bits) noexcept
{
    return bits & ~kSignMask;
}

[[nodiscard]] std::uint64_t maximumAbsoluteEndpoint(
    ArithmeticInterval interval) noexcept
{
    const auto lower = absoluteBits(bitsOf(interval.lower));
    const auto upper = absoluteBits(bitsOf(interval.upper));
    return lower >= upper ? lower : upper;
}

[[nodiscard]] bool strictlyPositive(std::uint64_t bits) noexcept
{
    return compareRealBits(bits, 0u) > 0;
}

[[nodiscard]] bool strictlyNegative(std::uint64_t bits) noexcept
{
    return compareRealBits(bits, 0u) < 0;
}

[[nodiscard]] bool intervalExcludesZero(ArithmeticInterval interval) noexcept
{
    return strictlyPositive(bitsOf(interval.lower)) ||
           strictlyNegative(bitsOf(interval.upper));
}

[[nodiscard]] ArithmeticFailure coordinateRadius(
    Real emitted,
    ArithmeticInterval ideal,
    Real& radius) noexcept
{
    const auto difference = assessIntervalOperation(
        IntervalOperation::Subtract, {emitted, emitted}, ideal);
    if (!difference.available()) {
        return difference.failure;
    }
    radius = valueFromBits<Real>(maximumAbsoluteEndpoint(difference.interval));
    return ArithmeticFailure::None;
}

[[nodiscard]] bool validPoint(const AssessmentPoint& point) noexcept
{
    for (const auto coordinate : point) {
        if (!validSubmittedScalarBits(bitsOf(coordinate))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool validPointAssessment(const PointAssessment& point) noexcept
{
    if (!point.available()) {
        return false;
    }
    for (std::size_t coordinate = 0u; coordinate < 3u; ++coordinate) {
        const auto radius = bitsOf(point.radius[coordinate]);
        if (!validInterval(point.ideal[coordinate]) ||
            !validSubmittedScalarBits(radius) || strictlyNegative(radius)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool sameBits(Real left, Real right) noexcept
{
    return bitsOf(left) == bitsOf(right);
}

[[nodiscard]] bool signedBandSatisfied(ArithmeticInterval a,
                                       ArithmeticInterval b,
                                       Real band,
                                       bool& a_negative) noexcept
{
    const auto band_bits = bitsOf(band);
    const auto negative_band = band_bits | kSignMask;
    const bool first_order =
        compareRealBits(bitsOf(a.upper), negative_band) < 0 &&
        compareRealBits(bitsOf(b.lower), band_bits) > 0;
    const bool reverse_order =
        compareRealBits(bitsOf(b.upper), negative_band) < 0 &&
        compareRealBits(bitsOf(a.lower), band_bits) > 0;
    a_negative = first_order;
    return first_order || reverse_order;
}

[[nodiscard]] bool actualClassificationAgrees(
    const OriginalEdgeObservation& input,
    bool a_negative) noexcept
{
    const auto band = bitsOf(input.signed_band);
    const auto negative_band = band | kSignMask;
    const auto actual_a = bitsOf(input.actual_signed_a);
    const auto actual_b = bitsOf(input.actual_signed_b);
    if (a_negative) {
        return compareRealBits(actual_a, negative_band) < 0 &&
               compareRealBits(actual_b, band) > 0;
    }
    return compareRealBits(actual_b, negative_band) < 0 &&
           compareRealBits(actual_a, band) > 0;
}

[[nodiscard]] bool denominatorMarginSatisfied(
    ArithmeticInterval denominator) noexcept
{
    std::uint64_t minimum_magnitude = 0u;
    if (strictlyPositive(bitsOf(denominator.lower))) {
        minimum_magnitude = absoluteBits(bitsOf(denominator.lower));
    } else if (strictlyNegative(bitsOf(denominator.upper))) {
        minimum_magnitude = absoluteBits(bitsOf(denominator.upper));
    } else {
        return false;
    }
    return minimum_magnitude > kHelperDenominatorMargin;
}

[[nodiscard]] bool actualDenominatorAgrees(
    ArithmeticInterval denominator,
    Real actual) noexcept
{
    const auto actual_bits = bitsOf(actual);
    if (strictlyPositive(bitsOf(denominator.lower))) {
        return strictlyPositive(actual_bits);
    }
    if (strictlyNegative(bitsOf(denominator.upper))) {
        return strictlyNegative(actual_bits);
    }
    return false;
}

[[nodiscard]] IntervalAssessment squaredDistance(
    const std::array<ArithmeticInterval, 3>& left,
    const std::array<ArithmeticInterval, 3>& right,
    unsigned dimension) noexcept
{
    IntervalAssessment sum;
    sum.interval = {Real{0}, Real{0}};
    sum.failure = ArithmeticFailure::None;
    for (unsigned coordinate = 0u; coordinate < dimension; ++coordinate) {
        const auto difference = assessIntervalOperation(
            IntervalOperation::Subtract,
            left[coordinate],
            right[coordinate]);
        if (!difference.available()) {
            return difference;
        }
        const auto square = assessIntervalSquare(difference.interval);
        if (!square.available()) {
            return square;
        }
        sum = assessIntervalOperation(
            IntervalOperation::Add, sum.interval, square.interval);
        if (!sum.available()) {
            return sum;
        }
    }
    return sum;
}

[[nodiscard]] IntervalAssessment distanceInterval(
    const std::array<ArithmeticInterval, 3>& left,
    const std::array<ArithmeticInterval, 3>& right,
    unsigned dimension) noexcept
{
    const auto squared = squaredDistance(left, right, dimension);
    if (!squared.available()) {
        return squared;
    }
    return assessIntervalSqrt(squared.interval);
}

[[nodiscard]] std::array<ArithmeticInterval, 3> singletonPoint(
    const AssessmentPoint& point) noexcept
{
    return {{{point[0], point[0]},
             {point[1], point[1]},
             {point[2], point[2]}}};
}

[[nodiscard]] bool matchesSingletonPoint(
    const std::array<ArithmeticInterval, 3>& intervals,
    const AssessmentPoint& point,
    unsigned dimension) noexcept
{
    for (unsigned coordinate = 0u; coordinate < dimension; ++coordinate) {
        const auto point_bits = bitsOf(point[coordinate]);
        if (bitsOf(intervals[coordinate].lower) != point_bits ||
            bitsOf(intervals[coordinate].upper) != point_bits) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] DistanceAssessment unavailableDistance(
    ArithmeticFailure failure) noexcept
{
    DistanceAssessment result;
    result.failure = failure;
    return result;
}

} // namespace

SVMP_PRODUCER_ARITHMETIC_ENTRY IntervalAssessment assessIntervalOperation(
    IntervalOperation operation,
    ArithmeticInterval left,
    ArithmeticInterval right) noexcept
{
    if (!supportedRepresentation()) {
        return unavailable(ArithmeticFailure::UnsupportedRepresentation);
    }
    if (!validInterval(left) || !validInterval(right)) {
        return unavailable(ArithmeticFailure::InvalidInput);
    }
    if (operation == IntervalOperation::Divide &&
        !intervalExcludesZero(right)) {
        return unavailable(ArithmeticFailure::UnresolvedDenominator);
    }
    if (operation == IntervalOperation::Multiply ||
        operation == IntervalOperation::Divide) {
        return combineFour(operation, left, right);
    }

    IntervalAssessment lower;
    IntervalAssessment upper;
    if (operation == IntervalOperation::Add) {
        lower = scalarAdd(bitsOf(left.lower), bitsOf(right.lower), false);
        upper = bitsOf(left.lower) == bitsOf(left.upper) &&
                        bitsOf(right.lower) == bitsOf(right.upper)
                    ? lower
                    : scalarAdd(bitsOf(left.upper), bitsOf(right.upper), false);
    } else if (operation == IntervalOperation::Subtract) {
        lower = scalarAdd(bitsOf(left.lower), bitsOf(right.upper), true);
        upper = bitsOf(left.lower) == bitsOf(left.upper) &&
                        bitsOf(right.upper) == bitsOf(right.lower)
                    ? lower
                    : scalarAdd(bitsOf(left.upper), bitsOf(right.lower), true);
    } else {
        return unavailable(ArithmeticFailure::InvalidInput);
    }
    if (!lower.available()) {
        return lower;
    }
    if (!upper.available()) {
        return upper;
    }
    IntervalAssessment result;
    result.interval = {lower.interval.lower, upper.interval.upper};
    result.failure = ArithmeticFailure::None;
    return result;
}

SVMP_PRODUCER_ARITHMETIC_ENTRY IntervalAssessment assessIntervalSquare(
    ArithmeticInterval value) noexcept
{
    if (!supportedRepresentation()) {
        return unavailable(ArithmeticFailure::UnsupportedRepresentation);
    }
    if (!validInterval(value)) {
        return unavailable(ArithmeticFailure::InvalidInput);
    }
    const auto lower_bits = bitsOf(value.lower);
    const auto upper_bits = bitsOf(value.upper);
    if (!strictlyNegative(upper_bits)) {
        if (!strictlyPositive(lower_bits)) {
            const auto maximum = maximumAbsoluteEndpoint(value);
            const auto upper = scalarMultiply(maximum, maximum);
            if (!upper.available()) {
                return upper;
            }
            IntervalAssessment result;
            result.interval = {Real{0}, upper.interval.upper};
            result.failure = ArithmeticFailure::None;
            return result;
        }
        const auto lower = scalarMultiply(lower_bits, lower_bits);
        const auto upper = lower_bits == upper_bits
                               ? lower
                               : scalarMultiply(upper_bits, upper_bits);
        if (!lower.available()) {
            return lower;
        }
        if (!upper.available()) {
            return upper;
        }
        IntervalAssessment result;
        result.interval = {lower.interval.lower, upper.interval.upper};
        result.failure = ArithmeticFailure::None;
        return result;
    }
    const auto smaller_magnitude = absoluteBits(upper_bits);
    const auto larger_magnitude = absoluteBits(lower_bits);
    const auto lower = scalarMultiply(smaller_magnitude, smaller_magnitude);
    const auto upper = smaller_magnitude == larger_magnitude
                           ? lower
                           : scalarMultiply(larger_magnitude,
                                            larger_magnitude);
    if (!lower.available()) {
        return lower;
    }
    if (!upper.available()) {
        return upper;
    }
    IntervalAssessment result;
    result.interval = {lower.interval.lower, upper.interval.upper};
    result.failure = ArithmeticFailure::None;
    return result;
}

SVMP_PRODUCER_ARITHMETIC_ENTRY IntervalAssessment assessIntervalSqrt(
    ArithmeticInterval value) noexcept
{
    if (!supportedRepresentation()) {
        return unavailable(ArithmeticFailure::UnsupportedRepresentation);
    }
    if (!validInterval(value) || strictlyNegative(bitsOf(value.lower))) {
        return unavailable(ArithmeticFailure::InvalidInput);
    }
    const auto lower = scalarSqrt(bitsOf(value.lower));
    const auto upper = bitsOf(value.lower) == bitsOf(value.upper)
                           ? lower
                           : scalarSqrt(bitsOf(value.upper));
    if (!lower.available()) {
        return lower;
    }
    if (!upper.available()) {
        return upper;
    }
    IntervalAssessment result;
    result.interval = {lower.interval.lower, upper.interval.upper};
    result.failure = ArithmeticFailure::None;
    return result;
}

SVMP_PRODUCER_ARITHMETIC_ENTRY PointAssessment assessOriginalCorner(
    const AssessmentPoint& original,
    const AssessmentPoint& emitted) noexcept
{
    PointAssessment result;
    if (!supportedRepresentation()) {
        result.failure = ArithmeticFailure::UnsupportedRepresentation;
        return result;
    }
    if (!validPoint(original) || !validPoint(emitted)) {
        result.failure = ArithmeticFailure::InvalidInput;
        return result;
    }
    for (std::size_t coordinate = 0u; coordinate < 3u; ++coordinate) {
        result.ideal[coordinate] = {original[coordinate], original[coordinate]};
        const auto failure = coordinateRadius(
            emitted[coordinate], result.ideal[coordinate],
            result.radius[coordinate]);
        if (failure != ArithmeticFailure::None) {
            result.failure = failure;
            return result;
        }
    }
    result.failure = ArithmeticFailure::None;
    return result;
}

SVMP_PRODUCER_ARITHMETIC_ENTRY PointAssessment assessOriginalEdge(
    const OriginalEdgeObservation& input) noexcept
{
    PointAssessment result;
    if (!supportedRepresentation()) {
        result.failure = ArithmeticFailure::UnsupportedRepresentation;
        return result;
    }
    if (!validPoint(input.a) || !validPoint(input.b) ||
        !validPoint(input.emitted) ||
        !validSubmittedScalarBits(bitsOf(input.phi_a)) ||
        !validSubmittedScalarBits(bitsOf(input.phi_b)) ||
        !validSubmittedScalarBits(bitsOf(input.isovalue)) ||
        !validSubmittedScalarBits(bitsOf(input.signed_band)) ||
        !validSubmittedScalarBits(bitsOf(input.actual_signed_a)) ||
        !validSubmittedScalarBits(bitsOf(input.actual_signed_b)) ||
        !validSubmittedScalarBits(bitsOf(input.actual_denominator)) ||
        !validSubmittedScalarBits(bitsOf(input.actual_quotient)) ||
        !validSubmittedScalarBits(bitsOf(input.actual_clamped)) ||
        strictlyNegative(bitsOf(input.signed_band))) {
        result.failure = ArithmeticFailure::InvalidInput;
        return result;
    }
    if (input.canonicalization_changed) {
        result.failure = ArithmeticFailure::ChangedBranch;
        return result;
    }

    const auto signed_a = assessIntervalOperation(
        IntervalOperation::Subtract,
        {input.phi_a, input.phi_a},
        {input.isovalue, input.isovalue});
    const auto signed_b = assessIntervalOperation(
        IntervalOperation::Subtract,
        {input.phi_b, input.phi_b},
        {input.isovalue, input.isovalue});
    if (!signed_a.available() || !signed_b.available()) {
        result.failure = ArithmeticFailure::UnresolvedBand;
        return result;
    }
    bool a_negative = false;
    if (!signedBandSatisfied(signed_a.interval,
                             signed_b.interval,
                             input.signed_band,
                             a_negative)) {
        result.failure = ArithmeticFailure::UnresolvedBand;
        return result;
    }
    if (!actualClassificationAgrees(input, a_negative)) {
        result.failure = ArithmeticFailure::ChangedBranch;
        return result;
    }

    const auto denominator = assessIntervalOperation(
        IntervalOperation::Subtract,
        signed_a.interval,
        signed_b.interval);
    if (!denominator.available() ||
        !intervalExcludesZero(denominator.interval)) {
        result.failure = ArithmeticFailure::UnresolvedDenominator;
        return result;
    }
    if (!input.division_taken ||
        !actualDenominatorAgrees(denominator.interval,
                                 input.actual_denominator)) {
        result.failure = ArithmeticFailure::ChangedBranch;
        return result;
    }
    if (input.helper_denominator_guard &&
        !denominatorMarginSatisfied(denominator.interval)) {
        result.failure = ArithmeticFailure::UnresolvedDenominator;
        return result;
    }

    const auto interpolation = assessIntervalOperation(
        IntervalOperation::Divide,
        signed_a.interval,
        denominator.interval);
    if (!interpolation.available() ||
        compareRealBits(bitsOf(interpolation.interval.lower), 0u) <= 0 ||
        compareRealBits(bitsOf(interpolation.interval.upper), kOne) >= 0) {
        result.failure = ArithmeticFailure::UnresolvedInterior;
        return result;
    }
    const auto actual_quotient = bitsOf(input.actual_quotient);
    if (compareRealBits(actual_quotient, 0u) <= 0 ||
        compareRealBits(actual_quotient, kOne) >= 0 ||
        !sameBits(input.actual_quotient, input.actual_clamped)) {
        result.failure = ArithmeticFailure::ChangedBranch;
        return result;
    }

    const auto one_minus_interpolation = assessIntervalOperation(
        IntervalOperation::Subtract,
        {valueFromBits<Real>(kOne), valueFromBits<Real>(kOne)},
        interpolation.interval);
    if (!one_minus_interpolation.available()) {
        result.failure = one_minus_interpolation.failure;
        return result;
    }
    for (std::size_t coordinate = 0u; coordinate < 3u; ++coordinate) {
        const auto left = assessIntervalOperation(
            IntervalOperation::Multiply,
            one_minus_interpolation.interval,
            {input.a[coordinate], input.a[coordinate]});
        const auto right = assessIntervalOperation(
            IntervalOperation::Multiply,
            interpolation.interval,
            {input.b[coordinate], input.b[coordinate]});
        if (!left.available() || !right.available()) {
            result.failure = ArithmeticFailure::ArithmeticRange;
            return result;
        }
        const auto ideal = assessIntervalOperation(
            IntervalOperation::Add, left.interval, right.interval);
        if (!ideal.available()) {
            result.failure = ideal.failure;
            return result;
        }
        result.ideal[coordinate] = ideal.interval;
        const auto failure = coordinateRadius(
            input.emitted[coordinate], ideal.interval,
            result.radius[coordinate]);
        if (failure != ArithmeticFailure::None) {
            result.failure = failure;
            return result;
        }
    }
    result.failure = ArithmeticFailure::None;
    return result;
}

SVMP_PRODUCER_ARITHMETIC_ENTRY DistanceAssessment assessDistance(
    const PointAssessment& a,
    const AssessmentPoint& emitted_a,
    const PointAssessment& b,
    const AssessmentPoint& emitted_b,
    Real tolerance,
    unsigned dimension,
    OriginRelation relation,
    DistanceObservation observation) noexcept
{
    if (!supportedRepresentation()) {
        return unavailableDistance(ArithmeticFailure::UnsupportedRepresentation);
    }
    const auto tolerance_bits = bitsOf(tolerance);
    if (!validPointAssessment(a) || !validPointAssessment(b) ||
        !validPoint(emitted_a) || !validPoint(emitted_b) ||
        !validSubmittedScalarBits(tolerance_bits) ||
        !strictlyPositive(tolerance_bits) ||
        (dimension != 2u && dimension != 3u)) {
        return unavailableDistance(ArithmeticFailure::InvalidInput);
    }
    if (relation == OriginRelation::Unknown) {
        return unavailableDistance(ArithmeticFailure::UnknownOrigin);
    }
    if (!observation.executed &&
        (relation != OriginRelation::DistinctOriginal ||
         observation.comparison_result || observation.removed)) {
        return unavailableDistance(ArithmeticFailure::InvalidInput);
    }
    if (observation.executed &&
        (!validSubmittedScalarBits(bitsOf(observation.distance)) ||
         strictlyNegative(bitsOf(observation.distance)) ||
         observation.comparison_result != observation.removed)) {
        return unavailableDistance(ArithmeticFailure::InvalidInput);
    }
    if (relation == OriginRelation::SameOriginal &&
        (!observation.executed || !observation.comparison_result ||
         !observation.removed)) {
        return unavailableDistance(ArithmeticFailure::RetainedRepeat);
    }
    if (relation == OriginRelation::DistinctOriginal &&
        observation.executed &&
        (observation.comparison_result || observation.removed)) {
        return unavailableDistance(ArithmeticFailure::DistinctOriginMerge);
    }

    const auto ideal_distance = distanceInterval(a.ideal, b.ideal, dimension);
    const auto emitted_distance = matchesSingletonPoint(
                                      a.ideal, emitted_a, dimension) &&
                                          matchesSingletonPoint(
                                              b.ideal, emitted_b, dimension)
                                      ? ideal_distance
                                      : distanceInterval(
                                            singletonPoint(emitted_a),
                                            singletonPoint(emitted_b),
                                            dimension);
    if (!ideal_distance.available() || !emitted_distance.available()) {
        return unavailableDistance(ArithmeticFailure::UnresolvedDistance);
    }
    auto lower = smallerBits(bitsOf(ideal_distance.interval.lower),
                             bitsOf(emitted_distance.interval.lower));
    auto upper = largerBits(bitsOf(ideal_distance.interval.upper),
                            bitsOf(emitted_distance.interval.upper));
    if (observation.executed) {
        const auto observed = bitsOf(observation.distance);
        lower = smallerBits(lower, observed);
        upper = largerBits(upper, observed);
    }

    if (relation == OriginRelation::SameOriginal) {
        for (unsigned coordinate = 0u; coordinate < dimension; ++coordinate) {
            const auto measured = assessIntervalOperation(
                IntervalOperation::Subtract,
                {emitted_a[coordinate], emitted_a[coordinate]},
                {emitted_b[coordinate], emitted_b[coordinate]});
            const auto radius_sum = assessIntervalOperation(
                IntervalOperation::Add,
                {a.radius[coordinate], a.radius[coordinate]},
                {b.radius[coordinate], b.radius[coordinate]});
            if (!measured.available() || !radius_sum.available() ||
                maximumAbsoluteEndpoint(measured.interval) >
                    bitsOf(radius_sum.interval.upper)) {
                return unavailableDistance(
                    ArithmeticFailure::UnresolvedDistance);
            }
        }
        if (compareRealBits(upper, tolerance_bits) >= 0) {
            return unavailableDistance(ArithmeticFailure::UnresolvedDistance);
        }
    } else if (compareRealBits(lower, tolerance_bits) <= 0) {
        return unavailableDistance(ArithmeticFailure::UnresolvedDistance);
    }

    DistanceAssessment result;
    result.hull = {valueFromBits<Real>(lower), valueFromBits<Real>(upper)};
    result.failure = ArithmeticFailure::None;
    return result;
}

#undef SVMP_PRODUCER_ARITHMETIC_ENTRY

} // namespace svmp::FE::interfaces::detail
