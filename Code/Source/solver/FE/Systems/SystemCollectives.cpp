/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/SystemCollectives.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <sstream>

namespace svmp {
namespace FE {
namespace systems {

#if FE_HAS_MPI
namespace {

// Rank-local report of a failed agreement, written before the caller acts
// on the collective result.  Lists the first differing word positions with
// this rank's value and the communicator minimum and maximum, so the
// offending canonical field can be identified.
constexpr std::size_t kMaxReportedWordDifferences = 16u;
using WordDifference = std::array<std::uint64_t, 4>;

void logCanonicalWordDisagreement(
    MPI_Comm comm,
    const CanonicalWordsDisagreementReport& report,
    std::string_view context,
    std::uint64_t local_word_count,
    std::uint64_t minimum_word_count,
    std::uint64_t maximum_word_count,
    std::span<const WordDifference> differences) noexcept
{
  try {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  std::ostringstream message;
  message << report.prefix << " [R" << rank
          << "] " << report.subject
          << " diagnostic=" << report.diagnostic
          << " context='" << context << "'"
          << " local_word_count=" << local_word_count
          << " minimum_word_count=" << minimum_word_count
          << " maximum_word_count=" << maximum_word_count
          << " differing_words_shown=" << differences.size();
  for (const auto& difference : differences) {
    message << " [index=" << difference[0]
            << " local=" << difference[1]
            << " min=" << difference[2]
            << " max=" << difference[3] << "]";
  }
  message << "\n";
  std::cerr << message.str() << std::flush;
  } catch (...) {
  }
}

} // namespace

bool collectiveCanonicalWordsAgree(
    std::span<const std::uint64_t> local_words,
    MPI_Comm comm,
    std::string_view context,
    const CanonicalWordsDisagreementReport& report)
{
  int size = 1;
  MPI_Comm_size(comm, &size);
  if (size <= 1) {
    return true;
  }

  const auto local_word_count =
      static_cast<std::uint64_t>(local_words.size());
  std::uint64_t minimum_word_count{0u};
  std::uint64_t maximum_word_count{0u};
  MPI_Allreduce(
      &local_word_count,
      &minimum_word_count,
      1,
      MPI_UINT64_T,
      MPI_MIN,
      comm);
  MPI_Allreduce(
      &local_word_count,
      &maximum_word_count,
      1,
      MPI_UINT64_T,
      MPI_MAX,
      comm);

  // Keep the comparison scratch bounded and allocation-free after ranks have
  // entered the collective schedule. A rank-local allocation failure here
  // would otherwise strand peers inside the first value reduction.
  constexpr std::size_t chunk_words = 4096u;
  std::array<std::uint64_t, chunk_words> words{};
  std::array<std::uint64_t, chunk_words> minimum_words{};
  std::array<std::uint64_t, chunk_words> maximum_words{};
  bool values_agree = true;
  std::array<WordDifference, kMaxReportedWordDifferences> differences{};
  std::size_t difference_count = 0u;
  const auto global_word_count =
      static_cast<std::size_t>(maximum_word_count);
  for (std::size_t offset = 0u; offset < global_word_count;
       offset += chunk_words) {
    const auto words_this_chunk = std::min(
        chunk_words, global_word_count - offset);
    std::fill_n(words.begin(), words_this_chunk, 0u);
    const auto local_words_this_chunk = offset < local_words.size()
        ? std::min(words_this_chunk, local_words.size() - offset)
        : 0u;
    std::copy_n(
        local_words.begin() + static_cast<std::ptrdiff_t>(
                                  std::min(offset, local_words.size())),
        local_words_this_chunk,
        words.begin());
    const auto count = static_cast<int>(words_this_chunk);
    MPI_Allreduce(
        words.data(),
        minimum_words.data(),
        count,
        MPI_UINT64_T,
        MPI_MIN,
        comm);
    MPI_Allreduce(
        words.data(),
        maximum_words.data(),
        count,
        MPI_UINT64_T,
        MPI_MAX,
        comm);
    for (std::size_t i = 0u; i < words_this_chunk; ++i) {
      if (minimum_words[i] != maximum_words[i]) {
        values_agree = false;
        if (difference_count < differences.size()) {
          differences[difference_count++] = WordDifference{
              static_cast<std::uint64_t>(offset + i), words[i],
              minimum_words[i], maximum_words[i]};
        }
      }
    }
  }

  const bool agree =
      minimum_word_count == maximum_word_count && values_agree;
  if (!agree) {
    logCanonicalWordDisagreement(
        comm, report, context, local_word_count, minimum_word_count,
        maximum_word_count,
        std::span<const WordDifference>(
            differences.data(), difference_count));
  }
  return agree;
}
#endif

} // namespace systems
} // namespace FE
} // namespace svmp
