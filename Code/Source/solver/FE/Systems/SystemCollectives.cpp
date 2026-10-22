/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/SystemCollectives.h"

#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Constraints/AffineConstraints.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace svmp {
namespace FE {
namespace systems {

namespace {

constexpr std::string_view kLocalPreparationFailure{
    "failed local preparation for collective phase"};
constexpr std::string_view kRecoveryPhaseFailure{
    "failed postaccept maintenance recovery phase"};

// FNV-1a-style 64-bit mixing with the cut-context key constants of the
// application driver; the offset constant is not the standard FNV basis, and
// keys must stay equal to the driver's.
constexpr std::uint64_t kFnv1aOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnv1aPrime = 1099511628211ull;

void mixFnv1a(std::uint64_t& h, std::uint64_t value) noexcept
{
  h ^= value;
  h *= kFnv1aPrime;
}

double maxAbsDifference(std::span<const svmp::FE::Real> left,
                        std::span<const svmp::FE::Real> right)
{
  if (left.size() != right.size()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Cannot compare solution history vectors with mismatched sizes.");
  }

  double local = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    local = std::max(
        local,
        static_cast<double>(std::abs(left[i] - right[i])));
  }

  return local;
}

} // namespace

#if FE_HAS_MPI
CollectiveCommunicator::CollectiveCommunicator(MPI_Comm comm) : comm_(comm)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized && comm_ != MPI_COMM_NULL) {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
    }
}
#endif

bool collectiveAnyTrue(bool local, const CollectiveCommunicator& comm)
{
#if FE_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    const int local_value = local ? 1 : 0;
    int global_value = 0;
    MPI_Allreduce(
        &local_value, &global_value, 1, MPI_INT, MPI_MAX, comm.native());
    return global_value != 0;
  }
#else
  (void)comm;
#endif

  return local;
}

double collectiveMinDouble(double local, const CollectiveCommunicator& comm)
{
#if FE_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MIN, comm.native());
    return global;
  }
#else
  (void)comm;
#endif

  return local;
}

double collectiveMaxDouble(double local, const CollectiveCommunicator& comm)
{
#if FE_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm.native());
    return global;
  }
#else
  (void)comm;
#endif

  return local;
}

std::pair<std::uint64_t, std::uint64_t> collectiveMinMaxUint64(
    std::uint64_t local,
    const CollectiveCommunicator& comm)
{
#if FE_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    std::uint64_t minimum = 0u;
    std::uint64_t maximum = 0u;
#ifdef MPI_UINT64_T
    const MPI_Datatype datatype = MPI_UINT64_T;
#else
    const MPI_Datatype datatype = MPI_UNSIGNED_LONG_LONG;
#endif
    MPI_Allreduce(&local, &minimum, 1, datatype, MPI_MIN, comm.native());
    MPI_Allreduce(&local, &maximum, 1, datatype, MPI_MAX, comm.native());
    return {minimum, maximum};
  }
#else
  (void)comm;
#endif

  return {local, local};
}

// Moved from Application/Core/ApplicationDriver.cpp (globalSumSize, lines
// 7676-7692 at 7d9ddc8a) with the body unchanged.
std::size_t collectiveSumSize(
    std::size_t local,
    const CollectiveCommunicator& comm)
{
  auto local_count = static_cast<long long>(local);
#if FE_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    long long global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, comm.native());
    return static_cast<std::size_t>(std::max<long long>(0, global_count));
  }
#else
  (void)comm;
#endif

  return static_cast<std::size_t>(std::max<long long>(0, local_count));
}

void requireCollectivePhase(
    const std::exception_ptr& local_failure,
    const CollectiveCommunicator& comm,
    std::string_view failure,
    std::string_view phase)
{
  if (!collectiveAnyTrue(local_failure != nullptr, comm)) {
    return;
  }
  if (local_failure != nullptr) {
    std::rethrow_exception(local_failure);
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Another active FE communicator rank " +
      std::string(failure) + " '" + std::string(phase) + "'.");
}

void requireCollectivePhasePreparation(
    const std::exception_ptr& local_failure,
    const CollectiveCommunicator& comm,
    std::string_view phase)
{
  requireCollectivePhase(
      local_failure,
      comm,
      "failed local preparation for collective phase",
      phase);
}

#if FE_HAS_MPI
namespace {

// Rank-local report of a failed agreement, written before the caller acts
// on the collective result.  Lists the first differing word positions with
// this rank's value and the communicator minimum and maximum, so the
// offending canonical field can be identified.
constexpr std::size_t kMaxReportedWordDifferences = 16u;
using WordDifference = std::array<std::uint64_t, 4>;

void logCanonicalWordDisagreement(
    const CollectiveCommunicator& comm,
    const CanonicalWordsDisagreementReport& report,
    std::string_view context,
    std::uint64_t local_word_count,
    std::uint64_t minimum_word_count,
    std::uint64_t maximum_word_count,
    std::span<const WordDifference> differences) noexcept
{
  try {
  std::ostringstream message;
  message << report.prefix << " [R" << comm.rank()
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
#endif

bool collectiveCanonicalWordsAgree(
    std::span<const std::uint64_t> local_words,
    const CollectiveCommunicator& comm,
    std::string_view context,
    const CanonicalWordsDisagreementReport& report)
{
  if (!comm.isParallel()) {
    return true;
  }

#if FE_HAS_MPI
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
      comm.native());
  MPI_Allreduce(
      &local_word_count,
      &maximum_word_count,
      1,
      MPI_UINT64_T,
      MPI_MAX,
      comm.native());

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
        comm.native());
    MPI_Allreduce(
        words.data(),
        maximum_words.data(),
        count,
        MPI_UINT64_T,
        MPI_MAX,
        comm.native());
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
#else
  (void)local_words;
  (void)context;
  (void)report;
  return true;
#endif
}

std::vector<Real> captureFeOrderedVectorCollectively(
    svmp::FE::backends::GenericVector& solution,
    const CollectiveCommunicator& comm)
{
  svmp::FE::GlobalIndex local_size{-1};
  std::exception_ptr local_size_failure;
  try {
    local_size = solution.size();
    if (local_size < 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Collective FE-vector capture cannot "
          "capture a vector with negative global size.");
    }
  } catch (...) {
    local_size_failure = std::current_exception();
  }
  requireCollectivePhase(
      local_size_failure,
      comm,
      kLocalPreparationFailure,
      "collective_fe_vector_capture_size_before_consensus");
  const auto global_size = static_cast<std::uint64_t>(local_size);
  const auto [minimum_size, maximum_size] =
      collectiveMinMaxUint64(global_size, comm);
  if (minimum_size != maximum_size) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Collective FE-vector sizes "
        "differ across the active FE communicator.");
  }

  std::vector<svmp::FE::Real> values;
  std::vector<svmp::FE::GlobalIndex> owned_rows;
  std::unique_ptr<svmp::FE::assembly::GlobalSystemView> read_view;
  std::exception_ptr local_failure;
  try {
    values.resize(static_cast<std::size_t>(global_size));
    owned_rows = solution.ownedGlobalRows();
    if (std::adjacent_find(
            owned_rows.begin(), owned_rows.end(),
            [](svmp::FE::GlobalIndex left,
               svmp::FE::GlobalIndex right) { return left >= right; }) !=
            owned_rows.end() ||
        !std::all_of(
            owned_rows.begin(), owned_rows.end(),
            [&](svmp::FE::GlobalIndex row) {
              return row >= 0 && row < local_size;
            })) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Collective FE-vector capture "
          "requires strictly ordered, unique, in-range backend-owned rows.");
    }
    read_view = solution.createGhostedReadView();
    if (!read_view) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Collective FE-vector capture "
          "requires an owned-row read view.");
    }
  } catch (...) {
    local_failure = std::current_exception();
  }
  requireCollectivePhase(
      local_failure,
      comm,
      kLocalPreparationFailure,
      "collective_fe_vector_capture_prepare_before_owned_reads");

  constexpr std::size_t chunk_capacity = 4096u;
  static_assert(
      sizeof(svmp::FE::Real) == sizeof(std::uint64_t),
      "Exact collective FE-vector capture requires a 64-bit FE scalar.");
  std::array<std::uint64_t, chunk_capacity> local_bits{};
  std::array<std::uint64_t, chunk_capacity> global_bits{};
  std::array<std::uint32_t, chunk_capacity> local_counts{};
  std::array<std::uint32_t, chunk_capacity> global_counts{};
  std::array<svmp::FE::Real, chunk_capacity> owned_values{};
  std::size_t owned_cursor = 0u;
  for (std::uint64_t chunk_begin = 0u; chunk_begin < global_size;
       chunk_begin += chunk_capacity) {
    const auto chunk_size = static_cast<std::size_t>(
        std::min<std::uint64_t>(
            chunk_capacity, global_size - chunk_begin));
    const auto chunk_end = chunk_begin + chunk_size;
    std::fill_n(local_bits.begin(), chunk_size, std::uint64_t{0u});
    std::fill_n(global_bits.begin(), chunk_size, std::uint64_t{0u});
    std::fill_n(local_counts.begin(), chunk_size, std::uint32_t{0u});
    std::fill_n(global_counts.begin(), chunk_size, std::uint32_t{0u});

    const auto first = owned_rows.begin() +
        static_cast<std::ptrdiff_t>(owned_cursor);
    const auto last =
        std::lower_bound(first, owned_rows.end(),
                         static_cast<svmp::FE::GlobalIndex>(chunk_end));
    const auto owned_count =
        static_cast<std::size_t>(std::distance(first, last));
    local_failure = nullptr;
    try {
      if (owned_count != 0u) {
        read_view->getVectorEntries(
            std::span<const svmp::FE::GlobalIndex>(
                owned_rows.data() +
                    static_cast<std::ptrdiff_t>(owned_cursor),
                owned_count),
            std::span<svmp::FE::Real>(owned_values.data(), owned_count));
      }
      for (std::size_t i = 0u; i < owned_count; ++i) {
        const auto row = owned_rows[owned_cursor + i];
        if (row < static_cast<svmp::FE::GlobalIndex>(chunk_begin) ||
            row >= static_cast<svmp::FE::GlobalIndex>(chunk_end)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Collective FE-vector owned "
              "read escaped its fixed global chunk.");
        }
        const auto relative = static_cast<std::size_t>(
            static_cast<std::uint64_t>(row) - chunk_begin);
        local_bits[relative] =
            std::bit_cast<std::uint64_t>(owned_values[i]);
        local_counts[relative] = 1u;
      }
    } catch (...) {
      local_failure = std::current_exception();
    }
    requireCollectivePhase(
        local_failure,
        comm,
        kLocalPreparationFailure,
        "collective_fe_vector_capture_owned_read_before_replication");

#if FE_HAS_MPI
    if (comm.isParallel()) {
#ifdef MPI_UINT32_T
      const MPI_Datatype count_type = MPI_UINT32_T;
#else
      static_assert(sizeof(std::uint32_t) == sizeof(unsigned));
      const MPI_Datatype count_type = MPI_UNSIGNED;
#endif
#ifdef MPI_UINT64_T
      const MPI_Datatype value_type = MPI_UINT64_T;
#else
      static_assert(sizeof(std::uint64_t) == sizeof(unsigned long long));
      const MPI_Datatype value_type = MPI_UNSIGNED_LONG_LONG;
#endif
      MPI_Allreduce(
          local_counts.data(), global_counts.data(),
          static_cast<int>(chunk_size), count_type, MPI_SUM, comm.native());
      MPI_Allreduce(
          local_bits.data(), global_bits.data(),
          static_cast<int>(chunk_size), value_type, MPI_BOR, comm.native());
    } else {
      std::copy_n(local_counts.begin(), chunk_size, global_counts.begin());
      std::copy_n(local_bits.begin(), chunk_size, global_bits.begin());
    }
#else
    if (comm.isParallel()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Distributed collective FE-vector "
          "capture requires MPI.");
    }
    std::copy_n(local_counts.begin(), chunk_size, global_counts.begin());
    std::copy_n(local_bits.begin(), chunk_size, global_bits.begin());
#endif
    if (std::any_of(
            global_counts.begin(),
            global_counts.begin() +
                static_cast<std::ptrdiff_t>(chunk_size),
            [](std::uint32_t count) { return count != 1u; })) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Collective FE-vector capture "
          "requires exactly one backend owner for every FE row.");
    }
    for (std::size_t i = 0u; i < chunk_size; ++i) {
      values[static_cast<std::size_t>(chunk_begin) + i] =
          std::bit_cast<svmp::FE::Real>(global_bits[i]);
    }
    owned_cursor += owned_count;
  }
  if (owned_cursor != owned_rows.size()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Collective FE-vector capture did "
        "not consume every backend-owned row.");
  }
  return values;
}

void assignFeOrderedVectorCollectively(
    svmp::FE::backends::GenericVector& solution,
    std::span<const svmp::FE::Real> values,
    const CollectiveCommunicator& comm)
{
  std::unique_ptr<svmp::FE::assembly::GlobalSystemView> view;
  std::span<svmp::FE::Real> local_values;
  std::vector<svmp::FE::GlobalIndex> owned_rows;
  std::vector<svmp::FE::Real> owned_values;
  std::exception_ptr local_failure;
  try {
    if (solution.size() < 0 ||
        static_cast<std::size_t>(solution.size()) != values.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Postaccept maintenance assignment "
          "found a vector/checkpoint size mismatch.");
    }
    view = solution.createAssemblyView();
    if (view) {
      owned_rows = solution.ownedGlobalRows();
      const bool owned_rows_valid = std::adjacent_find(
          owned_rows.begin(),
          owned_rows.end(),
          [](svmp::FE::GlobalIndex left,
             svmp::FE::GlobalIndex right) {
            return left >= right;
          }) == owned_rows.end() &&
          std::all_of(
              owned_rows.begin(),
              owned_rows.end(),
              [&](svmp::FE::GlobalIndex row) {
                return row >= 0 && row < solution.size();
              });
      if (!owned_rows_valid) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Postaccept maintenance assignment "
            "requires strictly ordered, unique, in-range owned vector rows.");
      }
      owned_values.reserve(owned_rows.size());
      for (const auto row : owned_rows) {
        owned_values.push_back(
            values[static_cast<std::size_t>(row)]);
      }
      view->beginAssemblyPhase();
    } else {
      local_values = solution.localSpan();
      if (local_values.size() != values.size()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Postaccept maintenance rollback "
            "could not obtain a complete local vector span.");
      }
    }
  } catch (...) {
    local_failure = std::current_exception();
  }
  requireCollectivePhase(
      local_failure,
      comm,
      kRecoveryPhaseFailure,
      "restore_vector_prepare_before_assembly");

  local_failure = nullptr;
  try {
    if (view) {
      constexpr std::size_t chunk_size = 4096u;
      for (std::size_t offset = 0u;
           offset < owned_rows.size(); offset += chunk_size) {
        const auto chunk =
            std::min(chunk_size, owned_rows.size() - offset);
        view->setVectorEntries(
            std::span<const svmp::FE::GlobalIndex>(
                owned_rows.data() + offset, chunk),
            std::span<const svmp::FE::Real>(
                owned_values.data() + offset, chunk));
      }
    } else {
      std::copy(values.begin(), values.end(), local_values.begin());
    }
  } catch (...) {
    local_failure = std::current_exception();
  }
  requireCollectivePhase(
      local_failure,
      comm,
      kRecoveryPhaseFailure,
      "restore_vector_values_before_assembly_flush");

  if (view) {
    local_failure = nullptr;
    try {
      view->endAssemblyPhase();
    } catch (...) {
      local_failure = std::current_exception();
    }
    requireCollectivePhase(
        local_failure,
        comm,
        kRecoveryPhaseFailure,
        "restore_vector_assembly_flush_before_finalize");

    local_failure = nullptr;
    try {
      view->finalizeAssembly();
    } catch (...) {
      local_failure = std::current_exception();
    }
    requireCollectivePhase(
        local_failure,
        comm,
        kRecoveryPhaseFailure,
        "restore_vector_finalize_before_ghost_update");
  }

  local_failure = nullptr;
  try {
    solution.updateGhosts();
  } catch (...) {
    local_failure = std::current_exception();
  }
  requireCollectivePhase(
      local_failure,
      comm,
      kRecoveryPhaseFailure,
      "restore_vector_ghost_update_complete");
}

double collectiveMaxAbsDifference(std::span<const Real> left,
                                  std::span<const Real> right,
                                  const CollectiveCommunicator& comm)
{
  const double local = maxAbsDifference(left, right);

#if FE_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized && comm.size() > 1) {
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm.native());
    return global;
  }
#else
  (void)comm;
#endif

  return local;
}

std::vector<std::size_t> collectiveIndexUnion(
    std::vector<std::size_t> local_indices,
    std::size_t upper_bound,
    const CollectiveCommunicator& comm,
    std::string_view purpose)
{
  std::sort(local_indices.begin(), local_indices.end());
  local_indices.erase(
      std::unique(local_indices.begin(), local_indices.end()),
      local_indices.end());
  bool index_out_of_range =
      !local_indices.empty() && local_indices.back() >= upper_bound;
#if FE_HAS_MPI
  if (comm.isParallel()) {
    const int local_invalid = index_out_of_range ? 1 : 0;
    int any_invalid = 0;
    MPI_Allreduce(&local_invalid,
                  &any_invalid,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  comm.native());
    index_out_of_range = any_invalid != 0;
  }
#endif
  if (index_out_of_range) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] " + std::string(purpose) +
        " contains an out-of-range local index.");
  }

#if FE_HAS_MPI
  if (comm.isParallel()) {
    const int local_count_overflow =
        local_indices.size() >
                static_cast<std::size_t>(std::numeric_limits<int>::max())
            ? 1
            : 0;
    int any_count_overflow = 0;
    MPI_Allreduce(&local_count_overflow,
                  &any_count_overflow,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  comm.native());
    if (any_count_overflow != 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] " + std::string(purpose) +
          " exceeds MPI local-count limits.");
    }

    std::vector<std::uint64_t> local_values;
    local_values.reserve(local_indices.size());
    for (const auto index : local_indices) {
      local_values.push_back(static_cast<std::uint64_t>(index));
    }

    const int local_count = static_cast<int>(local_values.size());
    std::vector<int> counts(static_cast<std::size_t>(comm.size()), 0);
    MPI_Allgather(&local_count,
                  1,
                  MPI_INT,
                  counts.data(),
                  1,
                  MPI_INT,
                  comm.native());

    std::vector<int> displacements(counts.size(), 0);
    int total_count = 0;
    for (std::size_t rank = 0u; rank < counts.size(); ++rank) {
      if (counts[rank] < 0 ||
          counts[rank] > std::numeric_limits<int>::max() - total_count) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] " + std::string(purpose) +
            " exceeds MPI aggregate-count limits.");
      }
      displacements[rank] = total_count;
      total_count += counts[rank];
    }

    std::vector<std::uint64_t> gathered(
        static_cast<std::size_t>(total_count), 0u);
#ifdef MPI_UINT64_T
    const MPI_Datatype index_type = MPI_UINT64_T;
#else
    const MPI_Datatype index_type = MPI_UNSIGNED_LONG_LONG;
#endif
    MPI_Allgatherv(local_values.empty() ? nullptr : local_values.data(),
                   local_count,
                   index_type,
                   gathered.empty() ? nullptr : gathered.data(),
                   counts.data(),
                   displacements.data(),
                   index_type,
                   comm.native());

    local_indices.clear();
    local_indices.reserve(gathered.size());
    for (const auto value : gathered) {
      if (value >= static_cast<std::uint64_t>(upper_bound) ||
          value >
              static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] " + std::string(purpose) +
            " contains an out-of-range communicator index.");
      }
      local_indices.push_back(static_cast<std::size_t>(value));
    }
    std::sort(local_indices.begin(), local_indices.end());
    local_indices.erase(
        std::unique(local_indices.begin(), local_indices.end()),
        local_indices.end());
  }
#else
  (void)comm;
#endif
  return local_indices;
}

void scatterFeOrderedSolution(
    backends::GenericVector& solution,
    std::span<const Real> values)
{
  if (static_cast<std::size_t>(solution.size()) != values.size()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Cannot scatter FE-ordered solution values with mismatched size.");
  }

  auto view = solution.createAssemblyView();
  if (!view) {
    auto span = solution.localSpan();
    if (span.size() != values.size()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Could not scatter FE-ordered solution values.");
    }
    std::copy(values.begin(), values.end(), span.begin());
    solution.updateGhosts();
    return;
  }

  constexpr svmp::FE::GlobalIndex chunk_size = 4096;
  std::vector<svmp::FE::GlobalIndex> dofs;
  dofs.reserve(static_cast<std::size_t>(std::min(solution.size(), chunk_size)));
  view->beginAssemblyPhase();
  for (svmp::FE::GlobalIndex offset = 0; offset < solution.size();
       offset += chunk_size) {
    const auto chunk =
        std::min<svmp::FE::GlobalIndex>(chunk_size, solution.size() - offset);
    dofs.resize(static_cast<std::size_t>(chunk));
    for (svmp::FE::GlobalIndex i = 0; i < chunk; ++i) {
      dofs[static_cast<std::size_t>(i)] = offset + i;
    }
    view->setVectorEntries(
        std::span<const svmp::FE::GlobalIndex>(dofs.data(), dofs.size()),
        values.subspan(static_cast<std::size_t>(offset),
                       static_cast<std::size_t>(chunk)));
  }
  view->endAssemblyPhase();
  view->finalizeAssembly();
  solution.updateGhosts();
}

interfaces::FreeSurfaceGeometryOwnershipCollective ownershipCollective(
    const CollectiveCommunicator& comm)
{
  svmp::FE::interfaces::FreeSurfaceGeometryOwnershipCollective collective;
  collective.rank = comm.rank();
  collective.size = comm.size();
  collective.all_gather_owned_rule_identity_values =
      [comm](std::span<const std::uint64_t> local_values) {
        std::vector<std::uint64_t> gathered(local_values.begin(),
                                            local_values.end());
#if FE_HAS_MPI
        if (!comm.isParallel()) {
          return gathered;
        }
        if (local_values.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
          throw std::runtime_error(
              "Free-surface snapshot ownership stream exceeds MPI limits.");
        }
        const int local_count = static_cast<int>(local_values.size());
        std::vector<int> counts(static_cast<std::size_t>(comm.size()), 0);
        MPI_Allgather(&local_count,
                      1,
                      MPI_INT,
                      counts.data(),
                      1,
                      MPI_INT,
                      comm.native());
        std::vector<int> displacements(counts.size(), 0);
        int total_values = 0;
        for (std::size_t rank = 0; rank < counts.size(); ++rank) {
          if (counts[rank] < 0 ||
              counts[rank] >
                  std::numeric_limits<int>::max() - total_values) {
            throw std::runtime_error(
                "Free-surface snapshot ownership gather exceeds MPI limits.");
          }
          displacements[rank] = total_values;
          total_values += counts[rank];
        }
        gathered.assign(static_cast<std::size_t>(total_values), 0u);
#ifdef MPI_UINT64_T
        const MPI_Datatype identity_type = MPI_UINT64_T;
#else
        const MPI_Datatype identity_type = MPI_UNSIGNED_LONG_LONG;
#endif
        MPI_Allgatherv(local_values.empty() ? nullptr : local_values.data(),
                       local_count,
                       identity_type,
                       gathered.empty() ? nullptr : gathered.data(),
                       counts.data(),
                       displacements.data(),
                       identity_type,
                       comm.native());
#else
        if (comm.isParallel()) {
          throw std::runtime_error(
              "Free-surface snapshot ownership validation requires MPI in a parallel run.");
        }
#endif
        return gathered;
      };
  collective.all_gather_revision_values =
      collective.all_gather_owned_rule_identity_values;
  return collective;
}

std::uint64_t collectiveConstraintSemanticsKey(
    const svmp::FE::systems::FESystem& system,
    const CollectiveCommunicator& comm)
{
  const auto local =
      svmp::FE::constraints::constraintSemanticFingerprint(
          system.constraints());
  const std::array<std::uint64_t, 4> local_words{
      local.hash_a,
      local.hash_b,
      local.line_count,
      local.entry_count,
  };
  const auto collective = ownershipCollective(comm);
  const auto global_words = collective.all_gather_revision_values(
      std::span<const std::uint64_t>(
          local_words.data(), local_words.size()));

  std::uint64_t key = kFnv1aOffset;
  mixFnv1a(
      key, static_cast<std::uint64_t>(comm.size()));
  mixFnv1a(
      key, static_cast<std::uint64_t>(global_words.size()));
  for (const auto word : global_words) {
    mixFnv1a(key, word);
  }
  return key == 0u ? 1u : key;
}

} // namespace systems
} // namespace FE
} // namespace svmp
