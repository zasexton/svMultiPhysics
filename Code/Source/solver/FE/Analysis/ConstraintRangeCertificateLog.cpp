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
 * reasons keep the vocabulary of the qualification records (free-surface
 * pressure certificate), which scripts parse.  ConstraintRangeCertificate.cpp
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

} // namespace

void logConstraintRangeEnablementDiffers()
{
    FE_LOG_INFO(
        "NewtonSolver: free-surface conservative balance"
        " diagnostic=free_surface_conservative_balance"
        " available=0"
        " reason=diagnostic_enablement_differs_across_communicator"
        " pressure_representability_available=0"
        " pressure_representability_method=lsqr"
        " pressure_representability_convergence=normal_equation_stationarity"
        " pressure_representability_distance_gate_applied=0"
        " pressure_representability_claimed=0"
        " pressure_representability_reason=diagnostic_enablement_differs_across_communicator"
        " constant_pressure_kkt_available=0"
        " constant_pressure_kkt_reason=diagnostic_enablement_differs_across_communicator");
}

void logConstraintRangeNotDeclared(const timestepping::NewtonObservation& observation)
{
    std::ostringstream skipped;
    skipped
        << "NewtonSolver: free-surface conservative balance"
        << " diagnostic=free_surface_conservative_balance"
        << " available=0"
        << " reason=certificate_not_declared"
        << " pressure_representability_available=0"
        << " pressure_representability_method=lsqr"
        << " pressure_representability_convergence=normal_equation_stationarity"
        << " pressure_representability_distance_gate_applied=0"
        << " pressure_representability_claimed=0"
        << " pressure_representability_reason=conservative_balance_operator_unavailable"
        << " constant_pressure_kkt_available=0"
        << " constant_pressure_kkt_reason=conservative_balance_operator_unavailable"
        << " iteration=" << observation.iteration
        << " phase='"
        << phaseName(observation.phase) << "'"
        << " sync_point=" << syncPointName(observation.sync_point);
    FE_LOG_INFO(skipped.str());
}

void logConstraintRangeOperatorNotInstalled(const timestepping::NewtonObservation& observation,
                                            std::string_view op)
{
    std::ostringstream skipped;
    skipped
        << "NewtonSolver: free-surface conservative balance"
        << " diagnostic=free_surface_conservative_balance"
        << " available=0"
        << " reason=operator_not_installed_on_all_ranks"
        << " missing_op='" << op << "'"
        << " pressure_representability_available=0"
        << " pressure_representability_method=lsqr"
        << " pressure_representability_convergence=normal_equation_stationarity"
        << " pressure_representability_distance_gate_applied=0"
        << " pressure_representability_claimed=0"
        << " pressure_representability_reason=conservative_balance_operator_unavailable"
        << " constant_pressure_kkt_available=0"
        << " constant_pressure_kkt_reason=conservative_balance_operator_unavailable"
        << " iteration=" << observation.iteration
        << " phase='"
        << phaseName(observation.phase) << "'"
        << " sync_point=" << syncPointName(observation.sync_point)
        << " scope=pressure_and_physical_potential_first_variations_only"
        << " contract=instantaneous_constrained_velocity_test_virtual_work"
        << " total_momentum_equilibrium_claimed=0"
        << " discrete_energy_theorem_claimed=0";
    FE_LOG_INFO(skipped.str());
}

void logConstraintRangeSample(const timestepping::NewtonObservation& observation,
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
        << "NewtonSolver: free-surface conservative balance"
        << " diagnostic=free_surface_conservative_balance"
        << " available=1"
        << " rank=" << rank
        << " iteration=" << observation.iteration
        << " phase='"
        << phaseName(observation.phase) << "'"
        << " sync_point=" << syncPointName(observation.sync_point)
        << " pressure_virtual_work_norm=" << norms[0]
        << " surface_energy_virtual_work_norm=" << norms[1]
        << " gravitational_potential_virtual_work_norm=" << norms[2]
        << " physical_potential_virtual_work_norm=" << norms[3]
        << " pressure_representability_load_virtual_work_norm="
        << norms[4]
        << " conservative_balance_norm=" << norms[5]
        << " normalization=pressure_plus_physical_potential_norms"
        << " normalized_imbalance=" << normalized_imbalance
        << " magnitude_mismatch=" << magnitude_mismatch
        << " alignment_cosine=" << alignment_cosine
        << " pressure_representability_cache_revision_key_available="
        << (cache.revision_key_available ? 1 : 0)
        << " pressure_representability_pair_every_assembly="
        << (cache.every_assembly ? 1 : 0)
        << " pressure_representability_available="
        << (record.range.available ? 1 : 0)
        << " pressure_representability_cache_hit="
        << (cache.hit ? 1 : 0)
        << " pressure_representability_pair_matrix_state_independent="
        << (cache.pair_matrix_state_independent ? 1 : 0)
        << " pressure_representability_cache_prior_sample="
        << (cache.prior_sample ? 1 : 0)
        << " pressure_representability_cache_geometry_match="
        << (cache.geometry_match ? 1 : 0)
        << " pressure_representability_cache_candidate="
        << (cache.candidate ? 1 : 0)
        << " pressure_representability_cache_load_equality_checked="
        << (cache.load_equality_checked ? 1 : 0)
        << " pressure_representability_cache_load_equal="
        << (cache.load_equal ? 1 : 0)
        << " pressure_representability_cache_load_difference_norm="
        << cache.load_difference_norm
        << " pressure_representability_cache_rejection_reason="
        << cache.rejection_reason
        << " pressure_representability_cache_policy="
           "geometry_constraints_state_independent_pair_and_exact_load"
        << " pressure_representability_method=lsqr"
        << " pressure_representability_convergence=normal_equation_stationarity"
        << " pressure_representability_distance_gate_applied=0"
        << " pressure_representability_claimed=0";
    if (record.range.available) {
        oss << " pressure_representability_residual_norm="
            << range.residual_norm
            << " pressure_representability_relative_residual="
            << range.relative_residual
            << " pressure_representability_normal_residual_norm="
            << range.normal_residual_norm
            << " pressure_representability_relative_normal_residual="
            << range.relative_normal_residual
            << " pressure_representability_pressure_norm="
            << range.solution_norm
            << " pressure_representability_iterations="
            << range.iterations
            << " pressure_representability_normal_residual_"
               "refinements="
            << range.normal_residual_refinements
            << " pressure_representability_converged="
            << (range.converged ? 1 : 0)
            << " pressure_representability_breakdown="
            << (range.breakdown ? 1 : 0)
            << " pressure_representability_active_pressure_dofs="
            << active_constraint_dofs
            << " pressure_representability_iteration_cap="
            << iteration_cap;
    } else {
        oss << " pressure_representability_reason="
            << record.range.reason;
    }
    oss << " constant_pressure_kkt_available="
        << (kkt.available ? 1 : 0)
        << " constant_pressure_unit_coefficients_represent_constant="
        << (kkt.unit_coefficients_represent_constant ? 1 : 0)
        << " constant_pressure_constraints_preserve_constants="
        << (kkt.constraints_preserve_constants ? 1 : 0)
        << " constant_pressure_constraint_scope="
        << kkt.constraint_scope
        << " constant_pressure_constraint_scope_reason="
        << kkt.constraint_scope_reason
        << " constant_pressure_kkt_method="
           "closed_form_one_dimensional_pressure_trace"
        << " constant_pressure_kkt_force_projection_applied=0";
    if (kkt.available) {
        oss << " constant_pressure_kkt_pressure_jump="
            << kkt.jump
            << " constant_pressure_kkt_volume_multiplier="
            << kkt.multiplier
            << " constant_pressure_kkt_direction_norm="
            << kkt.direction_norm
            << " constant_pressure_kkt_residual_norm="
            << kkt.residual_norm
            << " constant_pressure_kkt_relative_distance="
            << kkt.relative_distance
            << " constant_pressure_kkt_relative_orthogonality="
            << kkt.relative_orthogonality;
    } else {
        oss << " constant_pressure_kkt_reason="
            << kkt.reason;
    }
    oss << " pressure_representability_norm=constrained_reduced_coefficient_l2"
        << " pressure_representability_load=prescribed_external_pressure_plus_surface_area_variation_plus_young_wall_energy_plus_gravitational_potential"
        << " scope=pressure_and_physical_potential_first_variations_only"
        << " contract=instantaneous_constrained_velocity_test_virtual_work"
        << " excludes=line_friction_and_wetted_wall_navier_dissipation"
        << " total_momentum_equilibrium_claimed=0"
        << " discrete_energy_theorem_claimed=0";
    FE_LOG_INFO(oss.str());
}

void logConstraintRangeDistanceGate(const timestepping::NewtonAcceptance& acceptance,
                                    const ConstraintRangeReport& candidate,
                                    double threshold,
                                    bool passed,
                                    std::string_view reason)
{
    std::ostringstream oss;
    oss << std::setprecision(17)
        << "NewtonSolver: accepted static pressure-representability distance gate"
        << " diagnostic=free_surface_pressure_representability_distance_gate"
        << " accepted_static_state=1";
    writePosition(oss, acceptance);
    oss << " pressure_representability_distance_gate_applied=1"
        << " pressure_representability_available="
        << (candidate.range.available ? 1 : 0)
        << " pressure_representability_converged="
        << (candidate.range.converged ? 1 : 0)
        << " pressure_representability_breakdown="
        << (candidate.range.breakdown ? 1 : 0)
        << " pressure_representability_relative_residual="
        << candidate.range.relative_distance
        << " pressure_representability_max_relative_distance="
        << threshold
        << " pressure_representability_distance_gate_passed="
        << (passed ? 1 : 0)
        << " pressure_representability_claimed="
        << (passed ? 1 : 0)
        << " pressure_representability_reason="
        << candidate.range.reason
        << " reason=" << reason;
    FE_LOG_INFO(oss.str());
}

void logConstantFieldKktDistanceGate(const timestepping::NewtonAcceptance& acceptance,
                                     const ConstraintRangeReport& candidate,
                                     double threshold,
                                     bool passed,
                                     std::string_view reason)
{
    std::ostringstream oss;
    oss << std::setprecision(17)
        << "NewtonSolver: accepted static constant-pressure KKT distance gate"
        << " diagnostic=free_surface_constant_pressure_kkt_distance_gate"
        << " accepted_static_state=1";
    writePosition(oss, acceptance);
    oss << " constant_pressure_kkt_distance_gate_applied=1"
        << " constant_pressure_kkt_available="
        << (candidate.constant_field_kkt.available ? 1 : 0)
        << " constant_pressure_unit_coefficients_represent_constant="
        << (candidate.constant_field_kkt.unit_coefficients_represent_constant ? 1 : 0)
        << " constant_pressure_constraints_preserve_constants="
        << (candidate.constant_field_kkt.constraints_preserve_constants ? 1 : 0)
        << " constant_pressure_kkt_relative_distance="
        << candidate.constant_field_kkt.relative_distance
        << " constant_pressure_kkt_max_relative_distance="
        << threshold
        << " constant_pressure_kkt_distance_gate_passed="
        << (passed ? 1 : 0)
        << " constant_pressure_kkt_claimed="
        << (passed ? 1 : 0)
        << " constant_pressure_kkt_reason="
        << candidate.constant_field_kkt.reason
        << " constant_pressure_kkt_force_projection_applied=0"
        << " reason=" << reason;
    FE_LOG_INFO(oss.str());
}

void logConstraintRangeInitializer(const ConstraintRangeReport& record,
                                   double threshold,
                                   const ConstraintRangeLsqrResult& correction,
                                   double correction_load_norm,
                                   bool applied,
                                   bool passed,
                                   std::string_view reason)
{
    std::ostringstream oss;
    oss << std::setprecision(17)
        << "NewtonSolver: static compatible free-surface pressure initializer"
        << " diagnostic=free_surface_static_compatible_pressure_initializer"
        << " requested=1"
        << " applied=" << (applied ? 1 : 0)
        << " passed=" << (passed ? 1 : 0)
        << " reason=" << reason
        << " pressure_representability_available="
        << (record.range.available ? 1 : 0)
        << " pressure_representability_converged="
        << (record.range.converged ? 1 : 0)
        << " pressure_representability_breakdown="
        << (record.range.breakdown ? 1 : 0)
        << " pressure_representability_relative_residual="
        << record.range.relative_distance
        << " pressure_representability_max_relative_distance="
        << threshold
        << " pressure_correction_residual_norm="
        << correction.residual_norm
        << " pressure_correction_relative_residual="
        << correction.relative_residual
        << " pressure_correction_load_norm="
        << correction_load_norm
        << " pressure_correction_normal_residual_norm="
        << correction.normal_residual_norm
        << " pressure_correction_norm="
        << correction.solution_norm
        << " pressure_correction_iterations="
        << correction.iterations
        << " pressure_correction_method="
           "direct_balance_residual_lsqr"
        << " force_projection_applied=0"
        << " balanced_force_evidence=0"
        << " total_momentum_equilibrium_claimed=0"
        << " production_capillary_operator_changed=0"
        << " pressure_update=additive"
        << " pressure_increment=conservative_balance_residual_correction"
        << " existing_pressure_baseline_preserved=1"
        << " committed_history_or_rate_slots_mutated=0"
        << " scope=one_shot_current_pressure_initial_guess";
    FE_LOG_INFO(oss.str());
}

} // namespace detail
} // namespace analysis
} // namespace FE
} // namespace svmp
