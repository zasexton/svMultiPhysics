/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/ExactAccumulator.h"

#include <bit>
#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace systems {

namespace {

constexpr std::int64_t kDigitBase = std::int64_t{1} << 32;
constexpr std::uint64_t kDigitMask = 0xffffffffull;
// One add changes a digit by less than 2^32, so 2^30 adds keep a normalized
// digit far below 2^63.
constexpr std::uint32_t kAddsBetweenNormalizations = std::uint32_t{1} << 30;
// The weight of digit 0, bit 0: the smallest subnormal.
constexpr int kLowestExponent = -1074;

// Bits [start, start + count) of the magnitude held in little-endian 32-bit
// words, count <= 64; bits beyond the words are zero.
std::uint64_t bitsOf(const std::array<std::uint32_t, ExactAccumulator::kDigits + 1>& words,
                     int start,
                     int count) noexcept
{
  std::uint64_t out = 0;
  for (int i = 0; i < count; ++i) {
    const int bit = start + i;
    if (bit < 0) {
      continue;
    }
    const auto word = static_cast<std::size_t>(bit / 32);
    if (word >= words.size()) {
      break;
    }
    if ((words[word] >> (bit % 32)) & 1u) {
      out |= std::uint64_t{1} << i;
    }
  }
  return out;
}

} // namespace

void ExactAccumulator::add(double term) noexcept
{
  const auto bits = std::bit_cast<std::uint64_t>(term);
  const bool negative = (bits >> 63) != 0;
  const auto biased_exponent = static_cast<unsigned>((bits >> 52) & 0x7ffu);
  std::uint64_t mantissa = bits & ((std::uint64_t{1} << 52) - 1u);
  if (biased_exponent == 0x7ffu) {
    if (mantissa != 0) {
      ++nan_terms_;
    } else if (negative) {
      ++negative_infinite_terms_;
    } else {
      ++positive_infinite_terms_;
    }
    return;
  }
  // term = mantissa * 2^(shift - 1074)
  unsigned shift = 0;
  if (biased_exponent == 0) {
    if (mantissa == 0) {
      return;
    }
  } else {
    mantissa |= std::uint64_t{1} << 52;
    shift = biased_exponent - 1u;
  }
  const std::size_t digit = shift / 32u;
  const unsigned offset = shift % 32u;
  // mantissa << offset spans at most 84 bits: three digits.
  const std::uint64_t low = mantissa << offset;
  const std::uint64_t high = offset == 0 ? 0 : (mantissa >> (64u - offset));
  const auto d0 = static_cast<std::int64_t>(low & kDigitMask);
  const auto d1 = static_cast<std::int64_t>(low >> 32);
  const auto d2 = static_cast<std::int64_t>(high);
  if (negative) {
    digits_[digit] -= d0;
    digits_[digit + 1] -= d1;
    digits_[digit + 2] -= d2;
  } else {
    digits_[digit] += d0;
    digits_[digit + 1] += d1;
    digits_[digit + 2] += d2;
  }
  if (++adds_since_normalized_ >= kAddsBetweenNormalizations) {
    normalize();
  }
}

void ExactAccumulator::add(std::span<const double> terms) noexcept
{
  for (const double term : terms) {
    add(term);
  }
}

ExactAccumulator& ExactAccumulator::operator+=(const ExactAccumulator& other) noexcept
{
  normalize();
  ExactAccumulator normalized = other;
  normalized.normalize();
  for (std::size_t k = 0; k < kDigits; ++k) {
    digits_[k] += normalized.digits_[k];
  }
  nan_terms_ += normalized.nan_terms_;
  positive_infinite_terms_ += normalized.positive_infinite_terms_;
  negative_infinite_terms_ += normalized.negative_infinite_terms_;
  normalize();
  return *this;
}

void ExactAccumulator::normalize() noexcept
{
  for (std::size_t k = 0; k + 1 < kDigits; ++k) {
    const std::int64_t carry = digits_[k] >> 32;  // floor division by 2^32
    digits_[k] -= carry * kDigitBase;
    digits_[k + 1] += carry;
  }
  adds_since_normalized_ = 0;
}

std::array<std::int64_t, ExactAccumulator::kSlots> ExactAccumulator::slots() const noexcept
{
  ExactAccumulator normalized = *this;
  normalized.normalize();
  std::array<std::int64_t, kSlots> out{};
  for (std::size_t k = 0; k < kDigits; ++k) {
    out[k] = normalized.digits_[k];
  }
  out[kDigits] = normalized.nan_terms_;
  out[kDigits + 1] = normalized.positive_infinite_terms_;
  out[kDigits + 2] = normalized.negative_infinite_terms_;
  return out;
}

ExactAccumulator ExactAccumulator::fromSlots(
    std::span<const std::int64_t, kSlots> slots) noexcept
{
  ExactAccumulator out;
  for (std::size_t k = 0; k < kDigits; ++k) {
    out.digits_[k] = slots[k];
  }
  out.nan_terms_ = slots[kDigits];
  out.positive_infinite_terms_ = slots[kDigits + 1];
  out.negative_infinite_terms_ = slots[kDigits + 2];
  out.normalize();
  return out;
}

double ExactAccumulator::value() const noexcept
{
  if (nan_terms_ > 0 || (positive_infinite_terms_ > 0 && negative_infinite_terms_ > 0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (positive_infinite_terms_ > 0) {
    return std::numeric_limits<double>::infinity();
  }
  if (negative_infinite_terms_ > 0) {
    return -std::numeric_limits<double>::infinity();
  }

  ExactAccumulator magnitude = *this;
  magnitude.normalize();
  // Below the top digit every digit is in [0, 2^32), so the sign of N is the
  // sign of the top digit.
  const bool negative = magnitude.digits_[kDigits - 1] < 0;
  if (negative) {
    for (auto& digit : magnitude.digits_) {
      digit = -digit;
    }
    magnitude.normalize();
  }
  // |N| as little-endian 32-bit words; the top digit is non-negative here and
  // may exceed 32 bits.
  std::array<std::uint32_t, kDigits + 1> words{};
  for (std::size_t k = 0; k + 1 < kDigits; ++k) {
    words[k] = static_cast<std::uint32_t>(magnitude.digits_[k]);
  }
  const auto top = static_cast<std::uint64_t>(magnitude.digits_[kDigits - 1]);
  words[kDigits - 1] = static_cast<std::uint32_t>(top & kDigitMask);
  words[kDigits] = static_cast<std::uint32_t>(top >> 32);

  int highest_word = -1;
  for (int k = static_cast<int>(words.size()) - 1; k >= 0; --k) {
    if (words[static_cast<std::size_t>(k)] != 0) {
      highest_word = k;
      break;
    }
  }
  if (highest_word < 0) {
    return 0.0;
  }
  const int bit_length =
      32 * highest_word + std::bit_width(words[static_cast<std::size_t>(highest_word)]);

  double rounded = 0.0;
  if (bit_length <= 53) {
    // Exact: |N| < 2^53 and |N| * 2^-1074 is a subnormal or the smallest
    // binade of normals.
    rounded = std::ldexp(static_cast<double>(bitsOf(words, 0, bit_length)), kLowestExponent);
  } else {
    // The leading 53 bits, the next 11 and a sticky bit for the rest.
    const std::uint64_t leading = bitsOf(words, bit_length - 64, 64);
    bool sticky = false;
    for (int bit = 0; bit < bit_length - 64 && !sticky; ++bit) {
      sticky = ((words[static_cast<std::size_t>(bit / 32)] >> (bit % 32)) & 1u) != 0;
    }
    std::uint64_t mantissa = leading >> 11;
    const std::uint64_t rest = leading & 0x7ffu;
    constexpr std::uint64_t kHalf = 0x400u;
    if (rest > kHalf || (rest == kHalf && (sticky || (mantissa & 1u) != 0))) {
      ++mantissa;
    }
    int exponent = bit_length - 53 + kLowestExponent;
    if (mantissa == (std::uint64_t{1} << 53)) {
      mantissa >>= 1;
      ++exponent;
    }
    // Overflow rounds to infinity in ldexp.
    rounded = std::ldexp(static_cast<double>(mantissa), exponent);
  }
  return negative ? -rounded : rounded;
}

} // namespace systems
} // namespace FE
} // namespace svmp
