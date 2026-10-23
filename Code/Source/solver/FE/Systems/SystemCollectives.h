/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_SYSTEMCOLLECTIVES_H
#define SVMP_FE_SYSTEMS_SYSTEMCOLLECTIVES_H

/**
 * @file SystemCollectives.h
 * @brief Collective operations shared by distributed system services.
 *
 * Every routine here is collective over the communicator it receives: all
 * ranks call it in the same order, and a rank-local failure is agreed on
 * before any rank acts on it, so no rank is left inside a reduction.
 */

#include "Core/FEConfig.h"
#include "Core/Types.h"
#include "Systems/ExactAccumulator.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
class MeshComm;
#endif

namespace FE {
namespace backends {
class GenericVector;
} // namespace backends

namespace interfaces {
struct FreeSurfaceGeometryOwnershipCollective;
} // namespace interfaces

namespace systems {

class FESystem;
struct SystemStateView;

/**
 * @brief Communicator of the collectives below.
 *
 * Wraps an MPI communicator when FE is built with MPI and is a single rank
 * otherwise.  Rank and size are read once, at construction, and only when
 * MPI is initialized; a communicator built before MPI_Init therefore stays
 * a single rank.
 */
class CollectiveCommunicator {
public:
    /// A single rank (MPI_COMM_SELF when FE is built with MPI).
    CollectiveCommunicator() = default;

#if FE_HAS_MPI
    /// Reads rank and size from @p comm if MPI is initialized.
    explicit CollectiveCommunicator(MPI_Comm comm);

    /// Uses the given rank and size, for callers that already hold them.
    CollectiveCommunicator(MPI_Comm comm, int rank, int size) noexcept
        : comm_(comm), rank_(rank), size_(size)
    {
    }

    [[nodiscard]] MPI_Comm native() const noexcept { return comm_; }
#endif

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    /// The communicator, rank and size of a Mesh communicator (read when
    /// @p comm was made), so every collective here takes a MeshComm.
    CollectiveCommunicator(const svmp::MeshComm& comm) noexcept;
#endif

    [[nodiscard]] int rank() const noexcept { return rank_; }
    [[nodiscard]] int size() const noexcept { return size_; }
    [[nodiscard]] bool isParallel() const noexcept { return size_ > 1; }

private:
#if FE_HAS_MPI
    MPI_Comm comm_{MPI_COMM_SELF};
#endif
    int rank_{0};
    int size_{1};
};

/// True on every rank when @p local is true on at least one rank.
[[nodiscard]] bool collectiveAnyTrue(bool local, const CollectiveCommunicator& comm);

/// Sum of @p local over all ranks.
[[nodiscard]] double collectiveSumDouble(double local, const CollectiveCommunicator& comm);

/// Minimum of @p local over all ranks.
[[nodiscard]] double collectiveMinDouble(double local, const CollectiveCommunicator& comm);

/// Maximum of @p local over all ranks.
[[nodiscard]] double collectiveMaxDouble(double local, const CollectiveCommunicator& comm);

/// Largest |left[i] - right[i]| over all ranks; the local sizes must match.
[[nodiscard]] double collectiveMaxAbsDifference(
    std::span<const Real> left,
    std::span<const Real> right,
    const CollectiveCommunicator& comm);

/// Minimum and maximum of @p local over all ranks.
[[nodiscard]] std::pair<std::uint64_t, std::uint64_t> collectiveMinMaxUint64(
    std::uint64_t local,
    const CollectiveCommunicator& comm);

/// A rank-local value (for example a wall time) with its minimum, mean and
/// maximum over all ranks.
struct CollectiveTimingStatistics {
    double local{0.0};
    double min{0.0};
    double mean{0.0};
    double max{0.0};
};

/// Minimum, mean and maximum of @p local over all ranks.
[[nodiscard]] CollectiveTimingStatistics collectiveTimingStatistics(
    double local,
    const CollectiveCommunicator& comm);

/// Sum of @p local over all ranks, reduced as long long and clamped below at
/// zero.
[[nodiscard]] std::size_t collectiveSumSize(
    std::size_t local,
    const CollectiveCommunicator& comm);

/**
 * @brief Exact sum over all ranks, correctly rounded.
 *
 * The exact sum of every term added to the ranks' accumulators, rounded once
 * to nearest (ExactAccumulator).  The value is the same for any rank count
 * and any distribution of the terms over the ranks, and equals the value()
 * of one accumulator that holds all the terms.  One integer
 * MPI_Allreduce(SUM) of ExactAccumulator::kSlots words; nothing is
 * communicated on a single rank.
 */
[[nodiscard]] double collectiveExactSum(
    const ExactAccumulator& local,
    const CollectiveCommunicator& comm);

/// collectiveExactSum of each accumulator, with one reduction for all.
[[nodiscard]] std::vector<double> collectiveExactSums(
    std::span<const ExactAccumulator> local,
    const CollectiveCommunicator& comm);

/// collectiveExactSum of the local terms.
[[nodiscard]] double collectiveExactSum(
    std::span<const double> local_terms,
    const CollectiveCommunicator& comm);

/**
 * @brief Agree on a rank-local failure of one collective phase.
 *
 * Returns when no rank failed.  Otherwise a failing rank rethrows its own
 * exception and every other rank throws std::runtime_error with the text
 * "[svMultiPhysics::Application] Another active FE communicator rank
 * <failure> '<phase>'.".
 */
void requireCollectivePhase(
    const std::exception_ptr& local_failure,
    const CollectiveCommunicator& comm,
    std::string_view failure,
    std::string_view phase);

/**
 * @brief requireCollectivePhase for a failed local preparation.
 *
 * The other ranks throw "[svMultiPhysics::Application] Another active FE
 * communicator rank failed local preparation for collective phase
 * '<phase>'.".
 */
void requireCollectivePhasePreparation(
    const std::exception_ptr& local_failure,
    const CollectiveCommunicator& comm,
    std::string_view phase);

/**
 * @brief Log "<owner>: diagnostic=rank_local_collective_failure phase='<phase>'
 * reason='<what>'" for a rank-local failure (nothing for a null @p failure).
 *
 * Called before the collective that propagates the failure, so that the
 * failing rank's reason reaches the log even when the job is aborted from
 * another rank first.
 */
void logRankLocalPhaseFailure(
    std::string_view owner,
    std::string_view phase,
    const std::exception_ptr& failure);

/**
 * @brief Agree on the outcome of a rank-local phase.
 *
 * Each rank reports whether its local phase failed (@p local_failure) or
 * succeeded with (@p local_available) or without the data the next phase
 * needs.  The ranks take the minimum of failed (0), unavailable (1) and
 * available (2); before it, a failing rank logs
 * "<owner>: diagnostic=rank_local_collective_failure phase='<phase>'
 * reason='<what>'".  When any rank failed, a failing rank rethrows its own
 * exception and every other rank throws std::runtime_error with the text
 * "<owner>: another communicator rank rejected cut-context callback phase
 * '<phase>'".  Otherwise returns whether every rank has the data.  On a
 * single rank nothing is communicated or logged.
 */
[[nodiscard]] bool agreeOnLocalPhase(
    const std::exception_ptr& local_failure,
    bool local_available,
    const CollectiveCommunicator& comm,
    std::string_view phase,
    std::string_view owner);

/**
 * @brief agreeOnLocalPhase on the system's active communicator, when MPI is
 * initialized and not finalized (otherwise on a single rank).
 */
[[nodiscard]] bool agreeOnLocalPhase(
    const FESystem& system,
    const std::exception_ptr& local_failure,
    bool local_available,
    std::string_view phase,
    std::string_view owner);

/**
 * @brief Text of the rank-local report written when canonical words disagree.
 *
 * The report line is
 * `<prefix> [R<rank>] <subject> diagnostic=<diagnostic> context='<context>'`
 * followed by the word counts and the first differing word positions with
 * this rank's value and the communicator minimum and maximum.
 */
struct CanonicalWordsDisagreementReport {
    std::string_view prefix{"[svMultiPhysics::FE]"};
    std::string_view subject{"Canonical words disagree"};
    std::string_view diagnostic{"canonical_words_disagreement"};
};

/**
 * @brief Exact all-rank agreement for a canonical sequence of 64-bit words.
 *
 * Length and every word are compared directly.  The routine deliberately
 * does not hash the input, so agreement cannot be produced by a digest
 * collision.  All ranks execute the same padded collective schedule even
 * when their local sequence lengths differ, and the comparison scratch is
 * bounded and allocated before the first value reduction.  On disagreement
 * each rank writes one report line to std::cerr before returning false.
 * Returns true without communicating on a single rank.
 *
 * @param local_words this rank's canonical words
 * @param comm communicator; collective over it
 * @param context free text copied into the report
 * @param report text of the report line
 */
[[nodiscard]] bool collectiveCanonicalWordsAgree(
    std::span<const std::uint64_t> local_words,
    const CollectiveCommunicator& comm,
    std::string_view context,
    const CanonicalWordsDisagreementReport& report = {});

/**
 * @brief Replicate a distributed vector on every rank, in FE (global row) order.
 *
 * Every rank reads its backend-owned rows; the rows are combined exactly
 * (bit patterns, one owner per row, checked) in bounded chunks.  Throws on
 * every rank when sizes differ, ownership is not a partition of the rows,
 * or any rank fails to read.
 */
[[nodiscard]] std::vector<Real> captureFeOrderedVectorCollectively(
    backends::GenericVector& solution,
    const CollectiveCommunicator& comm);

/**
 * @brief The FE-ordered values of the current state of @p state on every rank.
 *
 * The distributed vector u_vector when present (collective, as above), else
 * a copy of the replicated span u; throws when the view has neither.
 */
[[nodiscard]] std::vector<Real> captureFeOrderedVectorCollectively(
    const SystemStateView& state,
    const CollectiveCommunicator& comm);

/**
 * @brief Assign a replicated FE-ordered vector to a distributed vector.
 *
 * Each rank sets its owned rows from @p values (all ranks pass the same
 * values), finalizes the assembly view and updates ghosts.  A failure on
 * any rank is agreed on between the phases, so ranks leave together.
 */
void assignFeOrderedVectorCollectively(
    backends::GenericVector& solution,
    std::span<const Real> values,
    const CollectiveCommunicator& comm);

/**
 * @brief Sorted, duplicate-free union over all ranks of local indices.
 *
 * Every index must be below @p upper_bound on every rank; otherwise every
 * rank throws.  @p purpose names the index set in error messages.
 */
[[nodiscard]] std::vector<std::size_t> collectiveIndexUnion(
    std::vector<std::size_t> local_indices,
    std::size_t upper_bound,
    const CollectiveCommunicator& comm,
    std::string_view purpose);

/**
 * @brief Write a complete FE-ordered vector into @p solution and update ghosts.
 *
 * Every rank passes the same values for all global rows.
 */
void scatterFeOrderedSolution(
    backends::GenericVector& solution,
    std::span<const Real> values);

/**
 * @brief Ownership collective for generated-geometry snapshot validation.
 *
 * Its gathers concatenate every rank's 64-bit words in rank order.  The
 * returned functions keep a copy of @p comm.
 */
[[nodiscard]] interfaces::FreeSurfaceGeometryOwnershipCollective ownershipCollective(
    const CollectiveCommunicator& comm);

/**
 * @brief Communicator-wide key of the system's constraint semantics.
 *
 * FNV-1a-style 64-bit mixing (with the driver's cut-context constants) of
 * the rank count and every rank's constraint semantic fingerprint, in rank
 * order; never zero.
 */
[[nodiscard]] std::uint64_t collectiveConstraintSemanticsKey(
    const FESystem& system,
    const CollectiveCommunicator& comm);

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
/// The active communicator of @p system (FESystem::activeMpiCommunicator)
/// as a Mesh communicator; MeshComm::self() without MPI.
[[nodiscard]] svmp::MeshComm activeMeshCommunicator(const FESystem& system);
#endif

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_SYSTEMCOLLECTIVES_H
