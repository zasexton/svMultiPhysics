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
 */

#include <cstdint>
#include <span>
#include <string_view>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace systems {

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

#if FE_HAS_MPI
/**
 * @brief Exact all-rank agreement for a canonical sequence of 64-bit words.
 *
 * Length and every word are compared directly.  The routine deliberately
 * does not hash the input, so agreement cannot be produced by a digest
 * collision.  All ranks execute the same padded collective schedule even
 * when their local sequence lengths differ, and the comparison scratch is
 * bounded and allocated before the first value reduction.  On disagreement
 * each rank writes one report line to std::cerr before returning false.
 * Returns true without communicating on a single-rank communicator.
 *
 * @param local_words this rank's canonical words
 * @param comm communicator of an initialized MPI; collective over it
 * @param context free text copied into the report
 * @param report text of the report line
 */
[[nodiscard]] bool collectiveCanonicalWordsAgree(
    std::span<const std::uint64_t> local_words,
    MPI_Comm comm,
    std::string_view context,
    const CanonicalWordsDisagreementReport& report = {});
#endif

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_SYSTEMCOLLECTIVES_H
