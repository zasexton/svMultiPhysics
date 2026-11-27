/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file ConstraintRangeCertificateLog.cpp
 * @brief The log adapter of the constraint-range certificate
 *
 * The text of the certificate's log lines: a sample of the declared loads
 * and of the range distance, the samples it could not take, the two
 * acceptance gates and the initializer.  The labels, their order and the
 * reasons follow the record vocabulary supplied by the formulation.
 * ConstraintRangeCertificate.cpp
 * computes what they print and decides which rank writes them.
 */

#include "Analysis/ConstraintRangeCertificate.h"

#include "Core/Logger.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace svmp {
namespace FE {
namespace analysis {
namespace detail {
namespace {

[[nodiscard]] const char* phaseName(const char* phase) noexcept
{
    return phase != nullptr ? phase : "unknown";
}

[[nodiscard]] const char* syncPointName(
    timestepping::NewtonOptions::StateSynchronizationPoint point) noexcept
{
    return timestepping::stateSynchronizationPointName(point);
}

/// The position of an acceptance event on a gate line.
void writePosition(std::ostringstream& oss, const timestepping::NewtonAcceptance& acceptance)
{
    oss << " iteration=" << acceptance.iteration;
    if (acceptance.outer_iteration.has_value()) {
        oss << " outer_iteration=" << *acceptance.outer_iteration;
    }
    oss << " phase='" << phaseName(acceptance.phase) << "'";
}

void writeUnavailable(std::ostringstream& out, const ConstraintRangeRecordVocabulary& labels,
                      std::string_view reason)
{
    out << " " << labels.range << "_available=0"
        << " " << labels.range << "_method=lsqr"
        << " " << labels.range << "_convergence=normal_equation_stationarity"
        << " " << labels.range << "_distance_gate_applied=0"
        << " " << labels.range << "_claimed=0"
        << " " << labels.range << "_reason=" << reason
        << " " << labels.constant << "_kkt_available=0"
        << " " << labels.constant << "_kkt_reason=" << reason;
}

} // namespace

void logConstraintRangeEnablementDiffers(const ConstraintRangeRecordVocabulary& labels)
{
    std::ostringstream skipped;
    skipped << labels.sample_heading
        << " available=0"
        << " reason=diagnostic_enablement_differs_across_communicator";
    writeUnavailable(skipped, labels, "diagnostic_enablement_differs_across_communicator");
    FE_LOG_INFO(skipped.str());
}

void logConstraintRangeNotDeclared(const ConstraintRangeRecordVocabulary& labels,
                                   const timestepping::NewtonObservation& observation)
{
    std::ostringstream skipped;
    skipped
        << labels.sample_heading
        << " available=0"
        << " reason=certificate_not_declared";
    writeUnavailable(skipped, labels, "conservative_balance_operator_unavailable");
    skipped << " iteration=" << observation.iteration
        << " phase='"
        << phaseName(observation.phase) << "'"
        << " sync_point=" << syncPointName(observation.sync_point);
    FE_LOG_INFO(skipped.str());
}

void logConstraintRangeOperatorNotInstalled(const ConstraintRangeRecordVocabulary& labels,
                                            const timestepping::NewtonObservation& observation,
                                            std::string_view op)
{
    std::ostringstream skipped;
    skipped
        << labels.sample_heading
        << " available=0"
        << " reason=operator_not_installed_on_all_ranks"
        << " missing_op='" << op << "'";
    writeUnavailable(skipped, labels, "conservative_balance_operator_unavailable");
    skipped << " iteration=" << observation.iteration
        << " phase='"
        << phaseName(observation.phase) << "'"
        << " sync_point=" << syncPointName(observation.sync_point)
        << labels.scope
        << labels.sample_claims;
    FE_LOG_INFO(skipped.str());
}

void logConstraintRangeSample(const ConstraintRangeRecordVocabulary& labels,
                              const timestepping::NewtonObservation& observation,
                              int rank,
                              const std::array<double, 6>& norms,
                              const ConstraintRangeCacheTrace& cache,
                              const ConstraintRangeReport& record,
                              const ConstraintRangeLsqrResult& range,
                              std::uint64_t active_constraint_dofs,
                              int iteration_cap)
{
    // The balance of the sampled loads, normalized by the constraint work
    // and the potential work.
    const double denominator = norms[0] + norms[3];
    const bool normalization_available =
        std::isfinite(denominator) && denominator > 0.0;
    const double normalized_imbalance =
        normalization_available
            ? norms[5] / denominator
            : std::numeric_limits<double>::quiet_NaN();
    const bool alignment_available =
        std::isfinite(norms[0]) && std::isfinite(norms[3]) &&
        std::isfinite(norms[5]) && norms[0] > 0.0 && norms[3] > 0.0;
    double alignment_cosine = std::numeric_limits<double>::quiet_NaN();
    if (alignment_available) {
        alignment_cosine =
            (norms[5] * norms[5] - norms[0] * norms[0] -
             norms[3] * norms[3]) /
            (2.0 * norms[0] * norms[3]);
        // Roundoff can place a mathematically valid cosine just
        // outside the closed interval.
        alignment_cosine = std::clamp(alignment_cosine, -1.0, 1.0);
    }
    const double magnitude_mismatch =
        normalization_available
            ? std::abs(norms[0] - norms[3]) / denominator
            : std::numeric_limits<double>::quiet_NaN();

    const auto& kkt = record.constant_field_kkt;
    std::ostringstream oss;
    oss << std::setprecision(17)
        << labels.sample_heading
        << " available=1"
        << " rank=" << rank
        << " iteration=" << observation.iteration
        << " phase='"
        << phaseName(observation.phase) << "'"
        << " sync_point=" << syncPointName(observation.sync_point)
        << " " << labels.work_norms[0] << "=" << norms[0]
        << " " << labels.work_norms[1] << "=" << norms[1]
        << " " << labels.work_norms[2] << "=" << norms[2]
        << " " << labels.work_norms[3] << "=" << norms[3]
        << " " << labels.work_norms[4] << "="
        << norms[4]
        << " " << labels.work_norms[5] << "=" << norms[5]
        << " normalization=" << labels.normalization
        << " normalized_imbalance=" << normalized_imbalance
        << " magnitude_mismatch=" << magnitude_mismatch
        << " alignment_cosine=" << alignment_cosine
        << " " << labels.range << "_cache_revision_key_available="
        << (cache.revision_key_available ? 1 : 0)
        << " " << labels.range << "_pair_every_assembly="
        << (cache.every_assembly ? 1 : 0)
        << " " << labels.range << "_available="
        << (record.range.available ? 1 : 0)
        << " " << labels.range << "_cache_hit="
        << (cache.hit ? 1 : 0)
        << " " << labels.range << "_pair_matrix_state_independent="
        << (cache.pair_matrix_state_independent ? 1 : 0)
        << " " << labels.range << "_cache_prior_sample="
        << (cache.prior_sample ? 1 : 0)
        << " " << labels.range << "_cache_geometry_match="
        << (cache.geometry_match ? 1 : 0)
        << " " << labels.range << "_cache_candidate="
        << (cache.candidate ? 1 : 0)
        << " " << labels.range << "_cache_load_equality_checked="
        << (cache.load_equality_checked ? 1 : 0)
        << " " << labels.range << "_cache_load_equal="
        << (cache.load_equal ? 1 : 0)
        << " " << labels.range << "_cache_load_difference_norm="
        << cache.load_difference_norm
        << " " << labels.range << "_cache_rejection_reason="
        << cache.rejection_reason
        << " " << labels.range << "_cache_policy=geometry_constraints_state_independent_pair_and_exact_load"
        << " " << labels.range << "_method=lsqr"
        << " " << labels.range << "_convergence=normal_equation_stationarity"
        << " " << labels.range << "_distance_gate_applied=0"
        << " " << labels.range << "_claimed=0";
    if (record.range.available) {
        oss << " " << labels.range << "_residual_norm="
            << range.residual_norm
            << " " << labels.range << "_relative_residual="
            << range.relative_residual
            << " " << labels.range << "_normal_residual_norm="
            << range.normal_residual_norm
            << " " << labels.range << "_relative_normal_residual="
            << range.relative_normal_residual
            << " " << labels.range << "_" << labels.field << "_norm="
            << range.solution_norm
            << " " << labels.range << "_iterations="
            << range.iterations
            << " " << labels.range << "_normal_residual_refinements="
            << range.normal_residual_refinements
            << " " << labels.range << "_converged="
            << (range.converged ? 1 : 0)
            << " " << labels.range << "_breakdown="
            << (range.breakdown ? 1 : 0)
            << " " << labels.range << "_active_" << labels.field << "_dofs="
            << active_constraint_dofs
            << " " << labels.range << "_iteration_cap="
            << iteration_cap;
    } else {
        oss << " " << labels.range << "_reason="
            << record.range.reason;
    }
    oss << " " << labels.constant << "_kkt_available="
        << (kkt.available ? 1 : 0)
        << " " << labels.constant << "_unit_coefficients_represent_constant="
        << (kkt.unit_coefficients_represent_constant ? 1 : 0)
        << " " << labels.constant << "_constraints_preserve_constants="
        << (kkt.constraints_preserve_constants ? 1 : 0)
        << " " << labels.constant << "_constraint_scope="
        << kkt.constraint_scope
        << " " << labels.constant << "_constraint_scope_reason="
        << kkt.constraint_scope_reason
        << " " << labels.constant << "_kkt_method=" << labels.constant_fit
        << " " << labels.constant << "_kkt_force_projection_applied=0";
    if (kkt.available) {
        oss << " " << labels.constant << "_kkt_" << labels.field << "_jump="
            << kkt.jump
            << " " << labels.constant << "_kkt_" << labels.multiplier << "="
            << kkt.multiplier
            << " " << labels.constant << "_kkt_direction_norm="
            << kkt.direction_norm
            << " " << labels.constant << "_kkt_residual_norm="
            << kkt.residual_norm
            << " " << labels.constant << "_kkt_relative_distance="
            << kkt.relative_distance
            << " " << labels.constant << "_kkt_relative_orthogonality="
            << kkt.relative_orthogonality;
    } else {
        oss << " " << labels.constant << "_kkt_reason="
            << kkt.reason;
    }
    oss << " " << labels.range << "_norm=constrained_reduced_coefficient_l2"
        << " " << labels.range << "_load=" << labels.load
        << labels.scope
        << labels.exclusions
        << labels.sample_claims;
    FE_LOG_INFO(oss.str());
}

void logConstraintRangeDistanceGate(const ConstraintRangeRecordVocabulary& labels,
                                    const timestepping::NewtonAcceptance& acceptance,
                                    const ConstraintRangeReport& candidate,
                                    double threshold,
                                    bool passed,
                                    std::string_view reason)
{
    std::ostringstream oss;
    oss << std::setprecision(17)
        << labels.range_gate_heading
        << " accepted_static_state=1";
    writePosition(oss, acceptance);
    oss << " " << labels.range << "_distance_gate_applied=1"
        << " " << labels.range << "_available="
        << (candidate.range.available ? 1 : 0)
        << " " << labels.range << "_converged="
        << (candidate.range.converged ? 1 : 0)
        << " " << labels.range << "_breakdown="
        << (candidate.range.breakdown ? 1 : 0)
        << " " << labels.range << "_relative_residual="
        << candidate.range.relative_distance
        << " " << labels.range << "_max_relative_distance="
        << threshold
        << " " << labels.range << "_distance_gate_passed="
        << (passed ? 1 : 0)
        << " " << labels.range << "_claimed="
        << (passed ? 1 : 0)
        << " " << labels.range << "_reason="
        << candidate.range.reason
        << " reason=" << reason;
    FE_LOG_INFO(oss.str());
}

void logConstantFieldKktDistanceGate(const ConstraintRangeRecordVocabulary& labels,
                                     const timestepping::NewtonAcceptance& acceptance,
                                     const ConstraintRangeReport& candidate,
                                     double threshold,
                                     bool passed,
                                     std::string_view reason)
{
    std::ostringstream oss;
    oss << std::setprecision(17)
        << labels.constant_gate_heading
        << " accepted_static_state=1";
    writePosition(oss, acceptance);
    oss << " " << labels.constant << "_kkt_distance_gate_applied=1"
        << " " << labels.constant << "_kkt_available="
        << (candidate.constant_field_kkt.available ? 1 : 0)
        << " " << labels.constant << "_unit_coefficients_represent_constant="
        << (candidate.constant_field_kkt.unit_coefficients_represent_constant ? 1 : 0)
        << " " << labels.constant << "_constraints_preserve_constants="
        << (candidate.constant_field_kkt.constraints_preserve_constants ? 1 : 0)
        << " " << labels.constant << "_kkt_relative_distance="
        << candidate.constant_field_kkt.relative_distance
        << " " << labels.constant << "_kkt_max_relative_distance="
        << threshold
        << " " << labels.constant << "_kkt_distance_gate_passed="
        << (passed ? 1 : 0)
        << " " << labels.constant << "_kkt_claimed="
        << (passed ? 1 : 0)
        << " " << labels.constant << "_kkt_reason="
        << candidate.constant_field_kkt.reason
        << " " << labels.constant << "_kkt_force_projection_applied=0"
        << " reason=" << reason;
    FE_LOG_INFO(oss.str());
}

void logConstraintRangeInitializer(const ConstraintRangeRecordVocabulary& labels,
                                   const ConstraintRangeReport& record,
                                   double threshold,
                                   const ConstraintRangeLsqrResult& correction,
                                   double correction_load_norm,
                                   bool applied,
                                   bool passed,
                                   std::string_view reason)
{
    std::ostringstream oss;
    oss << std::setprecision(17)
        << labels.initializer_heading
        << " requested=1"
        << " applied=" << (applied ? 1 : 0)
        << " passed=" << (passed ? 1 : 0)
        << " reason=" << reason
        << " " << labels.range << "_available="
        << (record.range.available ? 1 : 0)
        << " " << labels.range << "_converged="
        << (record.range.converged ? 1 : 0)
        << " " << labels.range << "_breakdown="
        << (record.range.breakdown ? 1 : 0)
        << " " << labels.range << "_relative_residual="
        << record.range.relative_distance
        << " " << labels.range << "_max_relative_distance="
        << threshold
        << " " << labels.field << "_correction_residual_norm="
        << correction.residual_norm
        << " " << labels.field << "_correction_relative_residual="
        << correction.relative_residual
        << " " << labels.field << "_correction_load_norm="
        << correction_load_norm
        << " " << labels.field << "_correction_normal_residual_norm="
        << correction.normal_residual_norm
        << " " << labels.field << "_correction_norm="
        << correction.solution_norm
        << " " << labels.field << "_correction_iterations="
        << correction.iterations
        << " " << labels.field << "_correction_method=direct_balance_residual_lsqr"
        << labels.initializer_scope
        << " " << labels.field << "_update=additive"
        << " " << labels.field << "_increment=conservative_balance_residual_correction"
        << " existing_" << labels.field << "_baseline_preserved=1"
        << " committed_history_or_rate_slots_mutated=0"
        << " scope=one_shot_current_" << labels.field << "_initial_guess";
    FE_LOG_INFO(oss.str());
}

} // namespace detail
} // namespace analysis
} // namespace FE
} // namespace svmp
