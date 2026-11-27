/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ConstraintRangeCertificate.h"

#include "Analysis/RowContributionObserver.h"

#include "Assembly/CutIntegrationContext.h"
#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/GenericMatrix.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/SmallCutAggregationConstraint.h"
#include "Core/FEException.h"
#include "Dofs/DofIndexSet.h"
#include "Geometry/CutQuadrature.h"
#include "Interfaces/LevelSetGeometrySnapshot.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"
#include "Systems/SystemCollectives.h"
#include "Systems/SystemsExceptions.h"
#include "Systems/TransientSystem.h"
#include "Core/EnvironmentNames.h"

#if defined(FE_HAS_FSILS)
#  include "Backends/FSILS/FsilsMatrix.h"
#  include "Backends/FSILS/FsilsVector.h"
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string_view>
#include <vector>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace analysis {
namespace {

void axpy(backends::GenericVector& y, Real alpha, const backends::GenericVector& x)
{
    auto ys = y.localSpan();
    auto xs = x.localSpan();
    FE_CHECK_ARG(ys.size() == xs.size(), "ConstraintRangeCertificate: axpy size mismatch");
    bool changed = false;
    for (std::size_t i = 0; i < ys.size(); ++i) {
        const auto old = ys[i];
        ys[i] += alpha * xs[i];
        changed = changed || ys[i] != old;
    }
    if (changed) {
        y.markModified();
    }
}

void zeroVectorEntries(std::span<const GlobalIndex> dofs, backends::GenericVector& vec)
{
    auto view = vec.createAssemblyView();
    FE_CHECK_NOT_NULL(view.get(), "ConstraintRangeCertificate: zeroVectorEntries view");
    view->beginAssemblyPhase();
    if (!dofs.empty()) {
        view->zeroVectorEntries(dofs);
    }
    // Distributed backends may require every communicator rank to finalize,
    // including ranks whose local constrained-row list is empty.
    view->finalizeAssembly();
}

// The one cut-volume side on which every term of @p pair in @p field lives.
struct ConstantFieldCutVolumeBinding {
    int marker{-1};
    geometry::CutIntegrationSide side{
        geometry::CutIntegrationSide::Interface};
};

std::optional<ConstantFieldCutVolumeBinding>
constantFieldCutVolumeBinding(
    const systems::OperatorDefinition& pair,
    FieldId constraint_field)
{
    bool unsupported_constraint_field_term = !pair.global.empty();
    const auto constraint_field_bearing = [constraint_field](const auto& term) {
        return term.test_field == constraint_field ||
               term.trial_field == constraint_field;
    };
    const auto inspect_nonvolume = [&](const auto& terms) {
        unsupported_constraint_field_term =
            unsupported_constraint_field_term ||
            std::any_of(terms.begin(), terms.end(), constraint_field_bearing);
    };
    inspect_nonvolume(pair.cells);
    inspect_nonvolume(pair.boundary);
    inspect_nonvolume(pair.interior);
    inspect_nonvolume(pair.interface_faces);

    std::optional<ConstantFieldCutVolumeBinding> binding;
    for (const auto& term : pair.cut_volumes) {
        if (!constraint_field_bearing(term)) {
            continue;
        }
        if (term.marker < 0 ||
            term.side == geometry::CutIntegrationSide::Interface) {
            unsupported_constraint_field_term = true;
            continue;
        }
        const ConstantFieldCutVolumeBinding candidate{
            term.marker, term.side};
        if (!binding.has_value()) {
            binding = candidate;
        } else if (binding->marker != candidate.marker ||
                   binding->side != candidate.side) {
            unsupported_constraint_field_term = true;
        }
    }
    if (unsupported_constraint_field_term || !binding.has_value()) {
        return std::nullopt;
    }
    return binding;
}

} // namespace

void ConstraintRangeVectors::recreate(const backends::BackendFactory& factory, GlobalIndex n_dofs)
{
    FE_THROW_IF(
        n_dofs <= 0,
        systems::InvalidStateException,
        "ConstraintRangeVectors::recreate: the vector layout has no DOFs");
    const auto create_vector = [&]() {
        auto vector = factory.createVector(n_dofs);
        FE_CHECK_NOT_NULL(
            vector.get(),
            "ConstraintRangeVectors::recreate vector");
        return vector;
    };

    // Construct the complete replacement set before releasing any old
    // vectors. Besides providing basic exception safety, this makes a layout
    // refresh observable and prevents an allocator from recycling an old
    // object address while the set is only partially rebuilt.
    auto new_load = create_vector();
    auto new_correction_load = create_vector();
    auto new_solution = create_vector();
    auto new_left_basis = create_vector();
    auto new_right_basis = create_vector();
    auto new_direction = create_vector();
    auto new_work = create_vector();
    auto new_residual = create_vector();
    auto new_normal_residual = create_vector();

    load = std::move(new_load);
    correction_load = std::move(new_correction_load);
    solution = std::move(new_solution);
    left_basis = std::move(new_left_basis);
    right_basis = std::move(new_right_basis);
    direction = std::move(new_direction);
    work = std::move(new_work);
    residual = std::move(new_residual);
    normal_residual = std::move(new_normal_residual);
}

ConstraintRangeLsqrResult solveConstraintRangeLsqr(
    const backends::GenericMatrix& pair,
    const backends::GenericVector& load,
    backends::GenericVector& coefficients,
    backends::GenericVector& left_basis,
    backends::GenericVector& right_basis,
    backends::GenericVector& direction,
    backends::GenericVector& work,
    backends::GenericVector& residual,
    backends::GenericVector& normal_residual,
    int max_iterations,
    std::optional<double> requested_max_relative_residual,
    const std::function<bool(bool)>& all_ranks)
{
    ConstraintRangeLsqrResult result;
    coefficients.zero();
    left_basis.zero();
    right_basis.zero();
    direction.zero();
    work.zero();
    residual.zero();
    normal_residual.zero();

    const auto vector_is_finite = [&](const backends::GenericVector& vector) {
        const auto values = vector.localSpan();
        const bool local_finite = std::all_of(
            values.begin(), values.end(), [](Real value) {
                return std::isfinite(static_cast<double>(value));
            });
        return all_ranks(local_finite);
    };
    const auto finite_nonnegative = [&](double value) {
        return all_ranks(std::isfinite(value) && value >= 0.0);
    };

    const double load_norm = load.norm();
    if (!vector_is_finite(load) || !finite_nonnegative(load_norm)) {
        result.breakdown = true;
        return result;
    }
    const bool local_relative_target_present =
        requested_max_relative_residual.has_value();
    const bool relative_target_present_on_all_ranks =
        all_ranks(local_relative_target_present);
    const bool relative_target_absent_on_all_ranks =
        all_ranks(!local_relative_target_present);
    if ((!relative_target_present_on_all_ranks &&
         !relative_target_absent_on_all_ranks) ||
        (relative_target_present_on_all_ranks &&
         !all_ranks(
             std::isfinite(*requested_max_relative_residual) &&
             *requested_max_relative_residual >= 0.0))) {
        result.breakdown = true;
        return result;
    }

    auto evaluate_fresh_residual = [&]() {
        coefficients.updateGhosts();
        pair.mult(coefficients, residual);
        residual.markModified();
        axpy(residual, Real{1.0}, load);
        residual.updateGhosts();
        pair.mult(residual, normal_residual);
        normal_residual.markModified();

        result.residual_norm = residual.norm();
        result.normal_residual_norm = normal_residual.norm();
        result.solution_norm = coefficients.norm();
        return vector_is_finite(residual) &&
               vector_is_finite(normal_residual) &&
               vector_is_finite(coefficients) &&
               finite_nonnegative(result.residual_norm) &&
               finite_nonnegative(result.normal_residual_norm) &&
               finite_nonnegative(result.solution_norm);
    };

    if (!evaluate_fresh_residual()) {
        result.breakdown = true;
        return result;
    }
    const double initial_normal_residual_norm =
        result.normal_residual_norm;
    const double absolute_stationarity_tolerance =
        100.0 * std::numeric_limits<Real>::epsilon() *
        std::max(1.0, initial_normal_residual_norm);
    // Keep normal-equation stationarity materially tighter than the primal
    // representability gates because mixed-pair conditioning can amplify the
    // remaining normal residual into a larger primal distance.
    const double stationarity_tolerance =
        absolute_stationarity_tolerance +
        1.0e-12 * initial_normal_residual_norm;
    const double residual_roundoff_floor =
        100.0 * std::numeric_limits<Real>::epsilon() *
        std::max(1.0, load_norm);
    const double normal_roundoff_floor =
        100.0 * std::numeric_limits<Real>::epsilon() *
        std::max(1.0, initial_normal_residual_norm);

    const auto refresh_relative_metrics = [&]() {
        // A load that is numerical zero has no meaningful relative scale.
        // Report zero distance only when its absolute residual is also at the
        // coefficient-scaled roundoff floor; the independent absolute gate
        // remains authoritative.
        result.relative_residual =
            load_norm > residual_roundoff_floor
                ? result.residual_norm / load_norm
                : (result.residual_norm <= residual_roundoff_floor
                       ? 0.0
                       : std::numeric_limits<double>::infinity());
        result.relative_normal_residual =
            initial_normal_residual_norm > normal_roundoff_floor
                ? result.normal_residual_norm /
                      initial_normal_residual_norm
                : (result.normal_residual_norm <=
                           normal_roundoff_floor
                       ? 0.0
                       : std::numeric_limits<double>::infinity());
    };
    refresh_relative_metrics();
    const auto requested_relative_residual_reached = [&]() {
        return !requested_max_relative_residual.has_value() ||
               all_ranks(
                   result.relative_residual <=
                   *requested_max_relative_residual);
    };

    // If G^T f is already at the scale-aware roundoff floor, p=0 is the
    // least-squares stationarity certificate.  This includes loads wholly in
    // the orthogonal complement of range(G).
    const bool initially_stationary = all_ranks(
        result.normal_residual_norm <= absolute_stationarity_tolerance);
    if (initially_stationary) {
        result.converged = true;
        return result;
    }
    if (max_iterations <= 0 || !(load_norm > 0.0)) {
        return result;
    }

    // LSQR solves G p = -f.  The vectors remain monolithic, but the zero
    // diagonal blocks ensure the Golub--Kahan bases alternate exactly between
    // primary and constraint subspaces.
    left_basis.copyFrom(load);
    left_basis.scale(static_cast<Real>(-1.0 / load_norm));
    left_basis.updateGhosts();
    pair.mult(left_basis, right_basis);
    right_basis.markModified();
    double alpha = right_basis.norm();
    if (!vector_is_finite(right_basis) || !finite_nonnegative(alpha) ||
        !(alpha > 0.0)) {
        result.breakdown = true;
        (void)evaluate_fresh_residual();
        refresh_relative_metrics();
        return result;
    }
    right_basis.scale(static_cast<Real>(1.0 / alpha));
    direction.copyFrom(right_basis);

    double rho_bar = alpha;
    double phi_bar = load_norm;
    // Preserve one slot in the fixed cap for a normal-residual-minimizing
    // refinement if finite-precision LSQR has not reached stationarity.
    const int lsqr_iteration_limit = std::max(0, max_iterations - 1);
    for (int iteration = 0; iteration < lsqr_iteration_limit; ++iteration) {
        right_basis.updateGhosts();
        pair.mult(right_basis, work);
        work.markModified();
        axpy(work, static_cast<Real>(-alpha), left_basis);
        // Correct the component along the immediately preceding basis vector
        // before normalization.  This local reorthogonalization bounds the
        // finite-precision drift without retaining the full Krylov basis.
        const double left_local_projection = work.dot(left_basis);
        axpy(work,
             static_cast<Real>(-left_local_projection),
             left_basis);
        double beta = work.norm();
        bool recurrence_finite =
            vector_is_finite(work) &&
            all_ranks(std::isfinite(left_local_projection)) &&
            finite_nonnegative(beta);
        if (!recurrence_finite) {
            result.breakdown = true;
            break;
        }

        if (beta > 0.0) {
            left_basis.copyFrom(work);
            left_basis.scale(static_cast<Real>(1.0 / beta));
            left_basis.updateGhosts();
            pair.mult(left_basis, work);
            work.markModified();
            axpy(work, static_cast<Real>(-beta), right_basis);
            // Apply the matching correction on the constraint-field basis.
            const double right_local_projection =
                work.dot(right_basis);
            axpy(work,
                 static_cast<Real>(-right_local_projection),
                 right_basis);
            alpha = work.norm();
            recurrence_finite =
                vector_is_finite(work) &&
                all_ranks(std::isfinite(right_local_projection)) &&
                finite_nonnegative(alpha);
            if (!recurrence_finite) {
                result.breakdown = true;
                break;
            }
            if (alpha > 0.0) {
                right_basis.copyFrom(work);
                right_basis.scale(static_cast<Real>(1.0 / alpha));
            } else {
                right_basis.zero();
            }
        } else {
            left_basis.zero();
            right_basis.zero();
            alpha = 0.0;
        }

        const double rho = std::hypot(rho_bar, beta);
        const bool rotation_finite = all_ranks(
            std::isfinite(rho) && rho > 0.0 &&
            std::isfinite(rho_bar) && std::isfinite(phi_bar) &&
            std::isfinite(alpha));
        if (!rotation_finite) {
            result.breakdown = true;
            break;
        }
        const double cosine = rho_bar / rho;
        const double sine = beta / rho;
        const double theta = sine * alpha;
        rho_bar = -cosine * alpha;
        const double phi = cosine * phi_bar;
        phi_bar = sine * phi_bar;
        const double solution_scale = phi / rho;
        const double direction_scale = theta / rho;
        const bool coefficients_finite = all_ranks(
            std::isfinite(cosine) && std::isfinite(sine) &&
            std::isfinite(theta) && std::isfinite(rho_bar) &&
            std::isfinite(phi) && std::isfinite(phi_bar) &&
            std::isfinite(solution_scale) &&
            std::isfinite(direction_scale));
        if (!coefficients_finite) {
            result.breakdown = true;
            break;
        }

        axpy(coefficients, static_cast<Real>(solution_scale), direction);
        direction.scale(static_cast<Real>(-direction_scale));
        axpy(direction, Real{1.0}, right_basis);
        result.iterations = iteration + 1;

        if (!evaluate_fresh_residual()) {
            result.breakdown = true;
            break;
        }
        refresh_relative_metrics();
        const bool stationary = all_ranks(
            result.normal_residual_norm <= stationarity_tolerance);
        if (stationary && requested_relative_residual_reached()) {
            result.converged = true;
            break;
        }

        // Exact bidiagonal termination without stationarity is a genuine
        // numerical breakdown; it is not silently relabelled convergence.
        const bool exact_recurrence_terminated =
            all_ranks(alpha == 0.0 || beta == 0.0);
        if (exact_recurrence_terminated) {
            result.converged = stationary;
            result.breakdown = !stationary;
            break;
        }

    }

    if (!result.converged && !result.breakdown &&
        result.iterations == lsqr_iteration_limit &&
        result.iterations < max_iterations) {
        // With g=G^T(Gp+f), choose the final correction along -g that
        // minimizes ||g-alpha*G^T*G*g||.  This gives the stationarity metric
        // one deterministic, matrix-free refinement without increasing the
        // fixed iteration cap or changing any acceptance tolerance.
        direction.copyFrom(normal_residual);
        direction.updateGhosts();
        pair.mult(direction, work);
        work.markModified();
        work.updateGhosts();
        pair.mult(work, left_basis);
        left_basis.markModified();
        const double numerator = work.dot(work);
        const double denominator = left_basis.dot(left_basis);
        const bool refinement_finite =
            vector_is_finite(direction) && vector_is_finite(work) &&
            vector_is_finite(left_basis) &&
            finite_nonnegative(numerator) &&
            finite_nonnegative(denominator) &&
            all_ranks(numerator > 0.0 && denominator > 0.0);
        if (!refinement_finite) {
            result.breakdown = true;
        } else {
            const double refinement_scale = numerator / denominator;
            if (!all_ranks(std::isfinite(refinement_scale) &&
                           refinement_scale > 0.0)) {
                result.breakdown = true;
            } else {
                axpy(coefficients,
                     static_cast<Real>(-refinement_scale),
                     direction);
                ++result.iterations;
                ++result.normal_residual_refinements;
                if (!evaluate_fresh_residual()) {
                    result.breakdown = true;
                } else {
                    refresh_relative_metrics();
                    result.converged = all_ranks(
                        result.normal_residual_norm <=
                        stationarity_tolerance);
                }
            }
        }
    }

    // Do not rely on recurrence estimates for qualification.  Recompute both
    // r=Gp+f and G^T r from the assembled operator after the final update (or
    // after a breakdown) and publish only those fresh norms.
    if (!evaluate_fresh_residual()) {
        result.breakdown = true;
        result.converged = false;
    }
    refresh_relative_metrics();
    if (result.converged) {
        result.converged = all_ranks(
            !result.breakdown &&
            result.normal_residual_norm <= stationarity_tolerance);
    }
    return result;
}

bool unitCoefficientsRepresentConstant(
    const spaces::FunctionSpace& constraint_field_space)
{
    try {
        if (constraint_field_space.field_type() != FieldType::Scalar ||
            constraint_field_space.value_dimension() != 1 ||
            constraint_field_space.dofs_per_element() == 0u ||
            constraint_field_space.is_variable_order()) {
            return false;
        }
        const auto quadrature = constraint_field_space.element().quadrature();
        if (!quadrature || quadrature->num_points() == 0u) {
            return false;
        }
        const std::vector<Real> unit_coefficients(
            constraint_field_space.dofs_per_element(), Real{1.0});
        const Real tolerance =
            Real{1000.0} * std::numeric_limits<Real>::epsilon() *
            std::max(Real{1.0},
                     static_cast<Real>(constraint_field_space.dofs_per_element()));
        for (std::size_t point = 0;
             point < quadrature->num_points();
             ++point) {
            const auto value = constraint_field_space.evaluate(
                quadrature->point(point), unit_coefficients);
            if (!std::isfinite(value[0]) ||
                std::abs(value[0] - Real{1.0}) > tolerance) {
                return false;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool constraintsPreserveUnitField(
    const constraints::AffineConstraints& affine_constraints,
    GlobalIndex constraint_begin,
    GlobalIndex constraint_count,
    const std::set<GlobalIndex>* retained_support)
{
    if (constraint_begin < 0 || constraint_count <= 0) {
        return false;
    }
    const GlobalIndex constraint_field_end = constraint_begin + constraint_count;
    bool compatible = true;
    affine_constraints.forEach(
        [&](const constraints::AffineConstraints::ConstraintView& line) {
            const bool constraint_field_slave =
                line.slave_dof >= constraint_begin &&
                line.slave_dof < constraint_field_end;
            bool has_constraint_field_master = false;
            bool has_other_field_master = false;
            double weight_sum = 0.0;
            double absolute_weight_sum = 0.0;
            for (const auto& entry : line.entries) {
                const bool constraint_field_master =
                    entry.master_dof >= constraint_begin &&
                    entry.master_dof < constraint_field_end;
                has_constraint_field_master =
                    has_constraint_field_master || constraint_field_master;
                has_other_field_master =
                    has_other_field_master || !constraint_field_master;
                weight_sum += entry.weight;
                absolute_weight_sum += std::abs(entry.weight);
            }

            // A constraint_field Dirichlet line removes the freely variable constant
            // mode.  A cross-field relation cannot be interpreted as a unit
            // scalar constraint_field trace.  Homogeneous constraint_field MPCs preserve the
            // mode only when their weights form a partition of unity.
            if (constraint_field_slave) {
                if (retained_support != nullptr &&
                    !retained_support->contains(line.slave_dof) &&
                    line.entries.empty() &&
                    std::isfinite(line.inhomogeneity) &&
                    line.inhomogeneity == 0.0) {
                    return;
                }
                const double tolerance =
                    100.0 * std::numeric_limits<Real>::epsilon() *
                    std::max(1.0, absolute_weight_sum);
                compatible =
                    compatible && !line.entries.empty() &&
                    line.inhomogeneity == 0.0 &&
                    !has_other_field_master &&
                    std::isfinite(weight_sum) &&
                    std::isfinite(absolute_weight_sum) &&
                    std::abs(weight_sum - 1.0) <= tolerance;
            } else if (has_constraint_field_master) {
                compatible = false;
            }
        });
    return compatible;
}



RetainedFieldSupportResult retainedFieldSupport(
    const systems::FESystem& system,
    const systems::OperatorDefinition& pair,
    FieldId constraint_field,
    GlobalIndex constraint_begin,
    GlobalIndex constraint_count,
    const CertificateCollectives& collectives)
{
    const auto& all_ranks = collectives.all_ranks;
    const auto& any_rank = collectives.any_rank;
    RetainedFieldSupportResult result;
    const auto binding =
        constantFieldCutVolumeBinding(pair, constraint_field);
    if (!all_ranks(binding.has_value())) {
        return result;
    }

    const auto binding_key =
        (static_cast<std::uint64_t>(binding->marker) << 8u) |
        static_cast<std::uint64_t>(binding->side);
    if (!collectives.agrees_on_unsigned(binding_key)) {
        result.reason = "pair_support_inconsistent";
        return result;
    }

    const auto reports =
        system.finalizedSmallCutAggregationProlongations();
    std::vector<std::shared_ptr<const constraints::
        SmallCutAggregationProlongationReport>> matches;
    for (const auto& candidate : reports) {
        if (candidate && candidate->field == constraint_field &&
            candidate->interface_marker == binding->marker &&
            candidate->active_side == binding->side) {
            matches.push_back(candidate);
        }
    }
    const bool ambiguous = any_rank(matches.size() > 1u);
    const bool unavailable = any_rank(matches.empty());
    if (ambiguous) {
        result.reason = "aggregation_report_ambiguous";
        return result;
    }
    if (unavailable || !all_ranks(matches.size() == 1u)) {
        result.reason = "aggregation_report_unavailable";
        return result;
    }

    const auto& report = *matches.front();
    const bool ineligible = any_rank(!report.trace_bound_eligible);
    const bool invalid_identity = any_rank(
        report.active_cells.empty() ||
        report.canonical_content_digest == 0u);
    if (ineligible) {
        result.reason = "aggregation_report_ineligible";
        return result;
    }
    if (invalid_identity) {
        result.reason = "aggregation_report_invalid";
        return result;
    }
    if (!collectives.agrees_on_unsigned(report.canonical_content_digest)) {
        result.reason = "aggregation_report_inconsistent";
        return result;
    }

    const auto& affine_constraints = system.constraints();
    const int rank = collectives.rank;
    const int size = collectives.size;
    const bool local_communicator_current =
        report.revision.local_rank == rank &&
        report.revision.communicator_size == size;
    if (!all_ranks(local_communicator_current)) {
        result.reason = "aggregation_report_inconsistent";
        return result;
    }

    const bool local_constraint_state_current =
        affine_constraints.isClosed() &&
        !system.constraintStateStaleForCurrentRevisions() &&
        report.revision.constraint ==
            system.constraintRevisionSnapshot() &&
        report.revision.affine_constraint_layout_revision ==
            affine_constraints.constraintLayoutRevision();
    if (!all_ranks(local_constraint_state_current)) {
        result.reason = "aggregation_report_stale";
        return result;
    }

    const auto* cut = system.cutIntegrationContext();
    bool local_cut_state_current = cut != nullptr;
    if (local_cut_state_current) {
        try {
            cut->assertAllLevelSetGeometrySnapshotsCurrent(
                system.meshAccess());
        } catch (...) {
            local_cut_state_current = false;
        }
    }
    local_cut_state_current =
        local_cut_state_current &&
        cut->contentRevision() ==
            report.revision.cut_context_content_revision &&
        cut->hasGeneratedVolumeMarker(binding->marker) &&
        report.revision.has_level_set_snapshot_revision &&
        report.revision.has_source_value_revision &&
        cut->hasLevelSetGeometrySnapshotForMarker(binding->marker);
    if (!all_ranks(local_cut_state_current)) {
        result.reason = "aggregation_report_stale";
        return result;
    }

    const auto snapshot_revision =
        cut->levelSetGeometrySnapshotRevisionForMarker(binding->marker);
    std::shared_ptr<const interfaces::LevelSetGeometrySnapshot>
        selected_snapshot;
    std::size_t local_snapshot_matches = 0u;
    for (const auto& snapshot : cut->levelSetGeometrySnapshots()) {
        if (snapshot &&
            snapshot->revision().snapshot_revision_key ==
                snapshot_revision) {
            ++local_snapshot_matches;
            selected_snapshot = snapshot;
        }
    }
    const bool local_snapshot_current =
        snapshot_revision ==
            report.revision.level_set_snapshot_revision &&
        local_snapshot_matches == 1u && selected_snapshot != nullptr &&
        selected_snapshot->revision().complete() &&
        selected_snapshot->revision().interface_marker == binding->marker &&
        selected_snapshot->revision().source_value_revision ==
            report.revision.source_value_revision;
    if (!all_ranks(local_snapshot_current)) {
        result.reason = "aggregation_report_stale";
        return result;
    }

    const GlobalIndex constraint_field_end = constraint_begin + constraint_count;
    bool local_cells_valid = true;
    GlobalIndex previous_cell = INVALID_GLOBAL_INDEX;
    std::set<GlobalIndex> support;
    for (const auto& cell : report.active_cells) {
        const bool valid_kind =
            cell.kind == constraints::
                             SmallCutAggregationActiveCellKind::FullActive ||
            cell.kind ==
                constraints::SmallCutAggregationActiveCellKind::Cut;
        const bool canonical_dofs =
            !cell.field_dofs.empty() &&
            std::is_sorted(
                cell.field_dofs.begin(), cell.field_dofs.end()) &&
            std::adjacent_find(
                cell.field_dofs.begin(), cell.field_dofs.end()) ==
                cell.field_dofs.end();
        const bool valid_cell =
            cell.cell_gid >= 0 && cell.cell_gid > previous_cell &&
            cell.owner_rank >= 0 && cell.owner_rank < size &&
            cell.retained_measure_provider_rank >= 0 &&
            cell.retained_measure_provider_rank < size &&
            cell.active_feature_id >= 0 && valid_kind &&
            std::isfinite(cell.retained_physical_volume) &&
            cell.retained_physical_volume > 0.0 && canonical_dofs;
        if (!valid_cell) {
            local_cells_valid = false;
            break;
        }
        for (const auto dof : cell.field_dofs) {
            if (dof < constraint_begin || dof >= constraint_field_end) {
                local_cells_valid = false;
                break;
            }
        }
        if (!local_cells_valid) {
            break;
        }
        support.insert(cell.field_dofs.begin(), cell.field_dofs.end());
        previous_cell = cell.cell_gid;
    }
    local_cells_valid = local_cells_valid && !support.empty();
    if (!all_ranks(local_cells_valid)) {
        result.reason = "aggregation_report_invalid";
        return result;
    }

    result.support = std::move(support);
    result.reason = "available";
    return result;
}

ConstantFieldKktResult evaluateConstantFieldKkt(
    const backends::GenericMatrix& pair,
    const backends::GenericVector& load,
    backends::GenericVector& unit_field,
    backends::GenericVector& constraint_action,
    backends::GenericVector& kkt_residual,
    GlobalIndex constraint_begin,
    GlobalIndex constraint_count,
    GlobalIndex primary_begin,
    GlobalIndex primary_count,
    std::span<const GlobalIndex> constrained_dofs,
    std::span<const GlobalIndex> vector_owned_dofs,
    const std::function<bool(bool)>& all_ranks)
{
    ConstantFieldKktResult result;
    if (constraint_begin < 0 || constraint_count <= 0 ||
        primary_begin < 0 || primary_count <= 0) {
        result.reason = "invalid_primary_or_constraint_range";
        return result;
    }

    const auto vector_is_finite =
        [&all_ranks](const backends::GenericVector& vector) {
            const auto values = vector.localSpan();
            const bool local_finite = std::all_of(
                values.begin(), values.end(), [](Real value) {
                    return std::isfinite(static_cast<double>(value));
                });
            return all_ranks(local_finite);
        };
    const auto finite_nonnegative = [&all_ranks](double value) {
        return all_ranks(std::isfinite(value) && value >= 0.0);
    };

    unit_field.zero();
    std::vector<GlobalIndex> owned_active_constraint_field_dofs;
    owned_active_constraint_field_dofs.reserve(vector_owned_dofs.size());
    const GlobalIndex constraint_field_end = constraint_begin + constraint_count;
    for (const auto dof : vector_owned_dofs) {
        if (dof >= constraint_begin && dof < constraint_field_end &&
            !std::binary_search(
                constrained_dofs.begin(), constrained_dofs.end(), dof)) {
            owned_active_constraint_field_dofs.push_back(dof);
        }
    }
    std::vector<Real> unit_values(
        owned_active_constraint_field_dofs.size(), Real{1.0});
    auto unit_view = unit_field.createAssemblyView();
    FE_CHECK_NOT_NULL(
        unit_view.get(), "evaluateConstantFieldKkt: unit-field view");
    unit_view->beginAssemblyPhase();
    if (!owned_active_constraint_field_dofs.empty()) {
        unit_view->setVectorEntries(
            owned_active_constraint_field_dofs, unit_values);
    }
    unit_view->finalizeAssembly();
    unit_field.updateGhosts();
    if (!vector_is_finite(unit_field)) {
        result.reason = "unit_constraint_field_nonfinite";
        return result;
    }

    pair.mult(unit_field, constraint_action);
    constraint_action.markModified();
    if (!vector_is_finite(constraint_action)) {
        result.reason = "constant_field_action_nonfinite";
        return result;
    }
    result.direction_norm = constraint_action.norm();
    if (!finite_nonnegative(result.direction_norm) ||
        !(result.direction_norm > 0.0)) {
        result.reason = "constant_field_direction_zero_or_nonfinite";
        return result;
    }

    // K*c must lie exclusively in the primary-test rows of the constrained
    // system. Check that invariant before interpreting its coefficient as a
    // constraint coefficient.
    unit_field.copyFrom(constraint_action);
    const GlobalIndex primary_end = primary_begin + primary_count;
    std::vector<GlobalIndex> owned_primary_dofs;
    owned_primary_dofs.reserve(vector_owned_dofs.size());
    for (const auto dof : vector_owned_dofs) {
        if (dof >= primary_begin && dof < primary_end) {
            owned_primary_dofs.push_back(dof);
        }
    }
    zeroVectorEntries(owned_primary_dofs, unit_field);
    unit_field.updateGhosts();
    const double off_primary_action_norm = unit_field.norm();
    const double subspace_tolerance =
        100.0 * std::numeric_limits<Real>::epsilon() *
        std::max(1.0, result.direction_norm);
    if (!finite_nonnegative(off_primary_action_norm) ||
        !all_ranks(off_primary_action_norm <= subspace_tolerance)) {
        result.reason = "constant_field_action_outside_primary_trace";
        return result;
    }

    const double load_norm = load.norm();
    const double direction_squared = constraint_action.dot(constraint_action);
    const double direction_load = constraint_action.dot(load);
    if (!finite_nonnegative(load_norm) ||
        !all_ranks(std::isfinite(direction_squared) &&
                   direction_squared > 0.0 &&
                   std::isfinite(direction_load))) {
        result.reason = "constant_field_projection_nonfinite";
        return result;
    }

    result.jump = -direction_load / direction_squared;
    result.multiplier = -result.jump;
    if (!all_ranks(std::isfinite(result.jump) &&
                   std::isfinite(result.multiplier))) {
        result.reason = "constant_field_multiplier_nonfinite";
        return result;
    }

    kkt_residual.copyFrom(load);
    axpy(kkt_residual,
         static_cast<Real>(result.jump),
         constraint_action);
    result.residual_norm = kkt_residual.norm();
    if (!vector_is_finite(kkt_residual) ||
        !finite_nonnegative(result.residual_norm)) {
        result.reason = "constant_field_residual_nonfinite";
        return result;
    }

    const double roundoff_floor =
        100.0 * std::numeric_limits<Real>::epsilon() *
        std::max({1.0, load_norm, result.direction_norm});
    // In the zero-load limit, dividing two roundoff-scale norms produces an
    // order-one ratio even though the absolute KKT residual is numerical
    // zero.  Preserve the scaled distance only above the coefficient-scale
    // roundoff floor; the independent absolute gate still bounds the result.
    result.relative_distance =
        load_norm > roundoff_floor
            ? result.residual_norm / load_norm
            : (result.residual_norm <= roundoff_floor
                   ? 0.0
                   : std::numeric_limits<double>::infinity());
    const double orthogonality_numerator =
        std::abs(constraint_action.dot(kkt_residual));
    const double orthogonality_denominator =
        result.direction_norm *
        std::max({load_norm, result.residual_norm, roundoff_floor});
    result.relative_orthogonality =
        orthogonality_numerator / orthogonality_denominator;
    if (!all_ranks(std::isfinite(result.relative_distance) &&
                   result.relative_distance >= 0.0 &&
                   std::isfinite(result.relative_orthogonality) &&
                   result.relative_orthogonality >= 0.0)) {
        result.reason = "constant_field_metrics_nonfinite";
        return result;
    }

    result.available = true;
    result.reason = "available";
    return result;
}

namespace {

bool reassembleEachCertificateSample() noexcept
{
    return environment::enabled(constraint_range_environment::reassemble);
}

} // namespace

// ---------------------------------------------------------------------------
// The certificate in a Newton solve
// ---------------------------------------------------------------------------

namespace {

#if FE_HAS_MPI
using CertificateCommunicator = MPI_Comm;
#else
using CertificateCommunicator = int;
constexpr CertificateCommunicator kSerialCertificateCommunicator = 0;
#endif

using constraints::ConstraintSemanticFingerprint;
using constraints::constraintSemanticFingerprint;
using timestepping::CertificateDecision;
using timestepping::CertificateStatus;

[[nodiscard]] CertificateCommunicator systemCommunicator(
    const systems::FESystem& system) noexcept
{
#if FE_HAS_MPI
    return system.activeMpiCommunicator();
#else
    (void)system;
    return kSerialCertificateCommunicator;
#endif
}

[[nodiscard]] bool anyRankOf(CertificateCommunicator communicator, bool local_value)
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized != 0 && finalized == 0) {
        const int local = local_value ? 1 : 0;
        int global = local;
        MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, communicator);
        return global != 0;
    }
#else
    (void)communicator;
#endif
    return local_value;
}

[[nodiscard]] int communicatorRank(CertificateCommunicator communicator) noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (!initialized || finalized || communicator == MPI_COMM_NULL) {
        return 0;
    }
    int rank = 0;
    (void)MPI_Comm_rank(communicator, &rank);
    return rank;
#else
    (void)communicator;
    return 0;
#endif
}

[[nodiscard]] int communicatorSize(CertificateCommunicator communicator) noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (!initialized || finalized || communicator == MPI_COMM_NULL) {
        return 1;
    }
    int size = 1;
    (void)MPI_Comm_size(communicator, &size);
    return size;
#else
    (void)communicator;
    return 1;
#endif
}

[[nodiscard]] bool communicatorAgreesOnUnsigned(
    std::uint64_t value,
    CertificateCommunicator communicator) noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized != 0 && finalized == 0 &&
        communicator != MPI_COMM_NULL) {
        std::uint64_t minimum = value;
        std::uint64_t maximum = value;
        MPI_Allreduce(&value, &minimum, 1, MPI_UINT64_T, MPI_MIN, communicator);
        MPI_Allreduce(&value, &maximum, 1, MPI_UINT64_T, MPI_MAX, communicator);
        return minimum == maximum;
    }
#else
    (void)communicator;
#endif
    return true;
}

[[nodiscard]] bool mpiMultiTaskActive() noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (!initialized || finalized) {
        return false;
    }
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size > 1;
#else
    return false;
#endif
}

[[nodiscard]] std::vector<GlobalIndex> ownedDofsForVector(
    const backends::GenericVector& vec,
    const dofs::IndexSet& fe_owned_dofs)
{
#if defined(FE_HAS_FSILS)
    if (const auto* fs = dynamic_cast<const backends::FsilsVector*>(&vec);
        fs != nullptr && fs->usesOwnedRowLayout()) {
        return fs->ownedFeDofs();
    }
#endif
    return fe_owned_dofs.toVector();
}

/// What the mixed pair/LSQR cache of a sample depends on besides the load.
struct DiagnosticGeometryKey {
    bool revision_tracking_available{false};
    std::uint64_t mesh_geometry_revision{0u};
    std::uint64_t mesh_topology_revision{0u};
    std::uint64_t mesh_ownership_revision{0u};
    std::uint64_t mesh_numbering_revision{0u};
    std::uint64_t mesh_field_layout_revision{0u};
    std::uint64_t mesh_label_revision{0u};
    std::uint64_t mesh_active_configuration_epoch{0u};
    std::uint64_t mesh_coordinate_configuration_key{0u};
    std::uintptr_t cut_context_identity{0u};
    std::uint64_t cut_context_content_revision{0u};
    std::uint64_t system_layout_revision{0u};
    std::uint64_t sparsity_pattern_revision{0u};
    ConstraintSemanticFingerprint constraint_semantics{};

    [[nodiscard]] bool operator==(const DiagnosticGeometryKey&) const noexcept = default;
};

[[nodiscard]] DiagnosticGeometryKey diagnosticGeometryKey(
    const systems::FESystem& system,
    const constraints::AffineConstraints& affine_constraints)
{
    const auto& mesh = system.meshAccess();
    const auto* cut_context = system.cutIntegrationContext();
    DiagnosticGeometryKey key;
    // Both fitted and unfitted operators depend on the background mesh.
    // A cut context tracks generated quadrature/content changes, but it
    // cannot invalidate stale entries after untracked background geometry,
    // topology, ownership, numbering, field-layout, or label changes.
    // Cache only when IMeshAccess explicitly promises that all revision
    // queries below form a trustworthy invalidation domain.
    key.revision_tracking_available = mesh.revisionTrackingAvailable();
    key.mesh_geometry_revision = mesh.geometryRevision();
    key.mesh_topology_revision = mesh.topologyRevision();
    key.mesh_ownership_revision = mesh.ownershipRevision();
    key.mesh_numbering_revision = mesh.numberingRevision();
    key.mesh_field_layout_revision = mesh.fieldLayoutRevision();
    key.mesh_label_revision = mesh.labelRevision();
    key.mesh_active_configuration_epoch = mesh.activeConfigurationEpoch();
    key.mesh_coordinate_configuration_key = mesh.coordinateConfigurationKey();
    key.cut_context_identity = reinterpret_cast<std::uintptr_t>(cut_context);
    key.cut_context_content_revision =
        cut_context != nullptr ? cut_context->contentRevision() : 0u;
    key.system_layout_revision = system.systemLayoutRevision();
    key.sparsity_pattern_revision = system.sparsityPatternRevision();
    key.constraint_semantics = constraintSemanticFingerprint(affine_constraints);
    return key;
}

[[nodiscard]] std::unique_ptr<backends::GenericMatrix> createPairMatrix(
    const timestepping::NewtonStorageContext& context,
    const std::string& pair_op)
{
    const auto* pair_dist = context.system.distributedSparsityIfAvailable(pair_op);
    if (pair_dist != nullptr &&
        context.factory.backendKind() != backends::BackendKind::Eigen) {
        return context.factory.createMatrix(*pair_dist);
    }
    return context.factory.createMatrix(context.system.sparsity(pair_op));
}

void transferOwnedEntries(backends::GenericVector& destination,
                          backends::GenericVector& source,
                          CertificateCommunicator communicator)
{
    const auto destination_owned = destination.ownedGlobalRows();
    const auto source_owned = source.ownedGlobalRows();
    const bool local_layout_compatible =
        destination.size() == source.size() && destination_owned == source_owned;
    FE_THROW_IF(
        anyRankOf(communicator, !local_layout_compatible),
        systems::InvalidStateException,
        "NewtonSolver: static compatible constraint-field initializer cannot transfer between vectors with different owned-row layouts");
    backends::copyOwnedEntriesBetweenLayouts(destination, source, destination_owned);
}

[[nodiscard]] CertificateStatus gateStatus(bool passed, std::string_view reason) noexcept
{
    if (passed) {
        return CertificateStatus::Passed;
    }
    return reason == "diagnostic_not_sampled" || reason == "diagnostic_unavailable" ||
                   reason == "constant_field_kkt_unavailable"
               ? CertificateStatus::Unavailable
               : CertificateStatus::Failed;
}

} // namespace

/// What one frozen solve keeps between samples, and the proposed correction.
struct ConstraintRangeCertificate::State {
    struct RepresentabilityCache {
        bool valid{false};
        DiagnosticGeometryKey geometry{};
        ConstraintRangeLsqrResult result{};
    };
    RepresentabilityCache cache{};
    std::optional<FieldId> constraint_field{};
    bool unavailable_logged{false};

    struct Correction {
        ConstraintRangeLsqrResult lsqr{};
        double load_norm{std::numeric_limits<double>::quiet_NaN()};
    };
    Correction correction{};
    ConstraintRangeStorage* proposal_storage{nullptr};
};

namespace {

/// The record a certificate keeps in a NewtonReport.
class ConstraintRangeRecord final : public timestepping::NewtonExtensionRecord {
public:
    [[nodiscard]] std::unique_ptr<timestepping::NewtonExtensionRecord> clone() const override
    {
        return std::make_unique<ConstraintRangeRecord>(*this);
    }

    ConstraintRangeReport value{};
};

[[nodiscard]] const ConstraintRangeReport& defaultConstraintRangeReport()
{
    static const ConstraintRangeReport absent{};
    return absent;
}

} // namespace

const ConstraintRangeReport& ConstraintRangeCertificate::record(
    const timestepping::NewtonReport& report) const
{
    const auto* record =
        dynamic_cast<const ConstraintRangeRecord*>(report.extension_records.find(*this));
    return record != nullptr ? record->value : defaultConstraintRangeReport();
}

ConstraintRangeReport& ConstraintRangeCertificate::mutableRecord(
    timestepping::NewtonReport& report) const
{
    auto* record = dynamic_cast<ConstraintRangeRecord*>(report.extension_records.find(*this));
    if (record == nullptr) {
        auto created = std::make_unique<ConstraintRangeRecord>();
        record = created.get();
        report.extension_records.set(*this, std::move(created));
    }
    return record->value;
}

bool constraintRangeMethodRequested(const ConstraintRangeMethod& method) noexcept
{
    return method.observe || method.initialize_constraint_field ||
           method.max_range_distance.has_value() ||
           method.max_constant_field_kkt_distance.has_value();
}

const ConstraintRangeRecordVocabulary& ConstraintRangeMethod::vocabulary() const noexcept
{
    static constexpr ConstraintRangeRecordVocabulary defaults;
    return record_vocabulary ? *record_vocabulary : defaults;
}

ConstraintRangeMethod constraintRangeMethodFromEnvironment()
{
    namespace controls = constraint_range_environment;
    const auto& names = environment::EnvironmentNameRegistry::solver();
    return ConstraintRangeMethod{
        .observe = names.enabled(controls::observe),
        .max_range_distance = names.optionalReal(controls::range_distance),
        .max_constant_field_kkt_distance = names.optionalReal(controls::constant_kkt_distance),
        .initialize_constraint_field = names.enabled(controls::initialize),
        .reassemble_each_sample = reassembleEachCertificateSample};
}

ConstraintRangeStorage* findConstraintRangeStorage(timestepping::NewtonWorkspace& workspace)
{
    for (auto& storage : workspace.extension_storage) {
        if (auto* found = dynamic_cast<ConstraintRangeStorage*>(storage.get())) {
            return found;
        }
    }
    return nullptr;
}

ConstraintRangeCertificate::ConstraintRangeCertificate(
    std::optional<ConstraintRangeDeclaration> declaration,
    ConstraintRangeMethod method)
    : declaration_(std::move(declaration))
    , method_(std::move(method))
    , state_(std::make_unique<State>())
{
    const bool defer_threshold_validation = mpiMultiTaskActive();
    if (method_.max_range_distance.has_value() && !defer_threshold_validation) {
        const double threshold = *method_.max_range_distance;
        FE_THROW_IF(
            threshold < 0.0 || !std::isfinite(threshold),
            InvalidArgumentException,
            "NewtonSolver: accepted static constraint-range maximum relative distance must be finite and >= 0");
    }
    if (method_.max_constant_field_kkt_distance.has_value() && !defer_threshold_validation) {
        const double threshold = *method_.max_constant_field_kkt_distance;
        FE_THROW_IF(
            threshold < 0.0 || !std::isfinite(threshold),
            InvalidArgumentException,
            "NewtonSolver: accepted static constant-field KKT maximum relative distance must be finite and >= 0");
    }
    FE_THROW_IF(
        method_.initialize_constraint_field &&
            !method_.max_range_distance.has_value() &&
            !defer_threshold_validation,
        InvalidArgumentException,
        "NewtonSolver: the static compatible constraint-field initializer requires an accepted static constraint-range maximum relative distance");
}

ConstraintRangeCertificate::~ConstraintRangeCertificate() = default;

void ConstraintRangeCertificate::beginSolve()
{
    state_->cache = {};
    state_->constraint_field.reset();
    state_->unavailable_logged = false;
}

std::unique_ptr<timestepping::NewtonExtensionStorage> ConstraintRangeCertificate::allocate(
    const timestepping::NewtonStorageContext& context) const
{
    const auto& system = context.system;
    bool allocate_pair =
        constraintRangeMethodRequested(method_) &&
        declaration_.has_value() && declaration_->pair.installedOn(system);
#if FE_HAS_MPI
    {
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        MPI_Finalized(&finalized);
        if (initialized != 0 && finalized == 0) {
            const int local = allocate_pair ? 1 : 0;
            int global = local;
            MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN,
                          system.activeMpiCommunicator());
            allocate_pair = global != 0;
        }
    }
#endif
    if (!allocate_pair) {
        return nullptr;
    }
    // The pair matrix first and its vectors immediately after it, so they
    // take the pair layout; the solver builds the production Jacobian last.
    auto storage = std::make_unique<ConstraintRangeStorage>();
    storage->pair = createPairMatrix(context, declaration_->pair.tag);
    storage->vectors.recreate(context.factory, system.dofHandler().getNumDofs());
    FE_CHECK_NOT_NULL(storage->pair.get(),
                      "NewtonSolver workspace.range_pair_matrix");
    return storage;
}

void ConstraintRangeCertificate::refreshStorage(const timestepping::NewtonStorageContext& context,
                                                timestepping::NewtonExtensionStorage* storage,
                                                bool after_production) const
{
    auto* pair_storage = dynamic_cast<ConstraintRangeStorage*>(storage);
    if (after_production || pair_storage == nullptr || !pair_storage->pair) {
        return;
    }
    // Refresh the diagnostic pair before the production Jacobian.  A matrix
    // replacement updates the backend factory's cached vector layout; doing
    // the production refresh last restores that cache for any vectors created
    // later in this nonlinear solve.  FSILS requires in-place refresh here so
    // the pair matrix remains compatible with its already allocated LSQR
    // vectors.
    const auto& system = context.system;
    const auto& factory = context.factory;
    FE_THROW_IF(
        !declaration_.has_value() || !declaration_->pair.installedOn(system),
        systems::InvalidStateException,
        "NewtonSolver: constraint-range pair operator disappeared during a sparsity refresh");
    const std::string& pair_op = declaration_->pair.tag;
    const auto* pair_dist = system.distributedSparsityIfAvailable(pair_op);
    const bool use_pair_distributed =
        pair_dist != nullptr && factory.backendKind() != backends::BackendKind::Eigen;
    bool pair_in_place =
        use_pair_distributed
            ? pair_storage->pair->reinitFromPattern(*pair_dist)
            : pair_storage->pair->reinitFromPattern(system.sparsity(pair_op));
#if FE_HAS_MPI
    if (use_pair_distributed) {
        const int local_pair_in_place = pair_in_place ? 1 : 0;
        int global_pair_in_place = local_pair_in_place;
        MPI_Allreduce(&local_pair_in_place, &global_pair_in_place, 1, MPI_INT, MPI_MIN,
                      system.activeMpiCommunicator());
        pair_in_place = global_pair_in_place != 0;
    }
#endif
    FE_THROW_IF(
        !pair_in_place && factory.backendKind() == backends::BackendKind::FSILS,
        systems::InvalidStateException,
        "NewtonSolver: FSILS constraint-range matrix sparsity refresh changed its vector layout; rebuild the nonlinear workspace at a time-step boundary");
    if (!pair_in_place) {
        pair_storage->pair = createPairMatrix(context, pair_op);
        FE_CHECK_NOT_NULL(pair_storage->pair.get(),
                          "NewtonSolver: constraint-range matrix sparsity refresh");
        // The replacement may carry a different PETSc/Trilinos owned and
        // ghost map even when its global dimensions are unchanged.  The
        // factory cache currently describes the refreshed pair pattern;
        // recreate every LSQR vector now, before the production-J refresh
        // deliberately restores the factory's production layout.
        pair_storage->vectors.recreate(factory, system.dofHandler().getNumDofs());
    }
    FE_CHECK_NOT_NULL(pair_storage->pair.get(),
                      "NewtonSolver: constraint-range matrix sparsity refresh");
}

void ConstraintRangeCertificate::validateCommunicatorPolicy(const systems::FESystem& system) const
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    const auto communicator = system.activeMpiCommunicator();
    if (initialized == 0 || finalized != 0 || communicator == MPI_COMM_NULL) {
        return;
    }

    const bool threshold_present = method_.max_range_distance.has_value();
    const double threshold = threshold_present ? *method_.max_range_distance : 0.0;
    const bool threshold_valid =
        !threshold_present || (threshold >= 0.0 && std::isfinite(threshold));
    const bool constant_threshold_present =
        method_.max_constant_field_kkt_distance.has_value();
    const double constant_threshold =
        constant_threshold_present ? *method_.max_constant_field_kkt_distance : 0.0;
    const bool constant_threshold_valid =
        !constant_threshold_present ||
        (constant_threshold >= 0.0 && std::isfinite(constant_threshold));
    const std::array<int, 6> local_flags{
        method_.initialize_constraint_field ? 1 : 0,
        threshold_present ? 1 : 0,
        threshold_valid ? 1 : 0,
        constant_threshold_present ? 1 : 0,
        constant_threshold_valid ? 1 : 0,
        constraintRangeMethodRequested(method_) ? 1 : 0,
    };
    std::array<int, local_flags.size()> minimum_flags{};
    std::array<int, local_flags.size()> maximum_flags{};
    MPI_Allreduce(local_flags.data(), minimum_flags.data(),
                  static_cast<int>(local_flags.size()), MPI_INT, MPI_MIN, communicator);
    MPI_Allreduce(local_flags.data(), maximum_flags.data(),
                  static_cast<int>(local_flags.size()), MPI_INT, MPI_MAX, communicator);
    const bool threshold_value_agrees =
        communicatorAgreesOnUnsigned(std::bit_cast<std::uint64_t>(threshold), communicator);
    const bool constant_threshold_value_agrees = communicatorAgreesOnUnsigned(
        std::bit_cast<std::uint64_t>(constant_threshold), communicator);

    const auto differs = [&](std::size_t index) {
        return minimum_flags[index] != maximum_flags[index];
    };
    FE_THROW_IF(
        differs(0),
        systems::InvalidStateException,
        "NewtonSolver: static compatible constraint-field initializer enablement "
        "differs across the active system communicator");
    FE_THROW_IF(
        differs(1),
        systems::InvalidStateException,
        "NewtonSolver: accepted static constraint-range threshold "
        "presence differs across the active system communicator");
    FE_THROW_IF(
        differs(2),
        systems::InvalidStateException,
        "NewtonSolver: accepted static constraint-range threshold "
        "validity differs across the active system communicator");
    FE_THROW_IF(
        threshold_present && maximum_flags[2] == 0,
        InvalidArgumentException,
        "NewtonSolver: accepted static constraint-range maximum "
        "relative distance must be finite and >= 0");
    FE_THROW_IF(
        threshold_present && !threshold_value_agrees,
        systems::InvalidStateException,
        "NewtonSolver: accepted static constraint-range threshold "
        "value differs across the active system communicator");
    FE_THROW_IF(
        differs(3),
        systems::InvalidStateException,
        "NewtonSolver: accepted static constant-field KKT threshold "
        "presence differs across the active system communicator");
    FE_THROW_IF(
        differs(4),
        systems::InvalidStateException,
        "NewtonSolver: accepted static constant-field KKT threshold "
        "validity differs across the active system communicator");
    FE_THROW_IF(
        constant_threshold_present && maximum_flags[4] == 0,
        InvalidArgumentException,
        "NewtonSolver: accepted static constant-field KKT maximum "
        "relative distance must be finite and >= 0");
    FE_THROW_IF(
        constant_threshold_present && !constant_threshold_value_agrees,
        systems::InvalidStateException,
        "NewtonSolver: accepted static constant-field KKT threshold "
        "value differs across the active system communicator");
    FE_THROW_IF(
        differs(5),
        systems::InvalidStateException,
        "NewtonSolver: constraint-range diagnostic "
        "enablement differs across the active system communicator");
    FE_THROW_IF(
        maximum_flags[0] != 0 && maximum_flags[1] == 0,
        InvalidArgumentException,
        "NewtonSolver: the static compatible constraint-field "
        "initializer requires an accepted static constraint-range "
        "maximum relative distance");
#else
    (void)system;
#endif
}

void ConstraintRangeCertificate::observe(timestepping::NewtonObservation& observation)
{
    auto& report = mutableRecord(observation.report);
    auto& transient = observation.transient;
    auto& residual_scratch = observation.scratch;
    const auto constrained_dofs = observation.constrained_dofs;
    const auto& constraints = transient.system().constraints();
    const auto system_communicator = systemCommunicator(transient.system());
    const auto anyRank = [system_communicator](bool local_value) {
        return anyRankOf(system_communicator, local_value);
    };
    const auto allRanks = [&anyRank](bool local_value) {
        return !anyRank(!local_value);
    };
    ConstraintRangeStorage absent_storage;
    auto* found_storage = dynamic_cast<ConstraintRangeStorage*>(observation.storage);
    auto& storage = found_storage != nullptr ? *found_storage : absent_storage;

    const bool local_diagnostic_enabled =
        constraintRangeMethodRequested(method_);
    const bool diagnostic_enabled_on_any_rank =
        anyRank(local_diagnostic_enabled);
    if (!diagnostic_enabled_on_any_rank) {
        return;
    }
    report.sampled = true;
    report.range.available = false;
    report.range.converged = false;
    report.range.breakdown = false;
    report.range.residual_norm =
        std::numeric_limits<double>::quiet_NaN();
    report.range.relative_distance =
        std::numeric_limits<double>::quiet_NaN();
    report.range.reason = "diagnostic_pending";
    report.constant_field_kkt.available = false;
    report.constant_field_kkt.unit_coefficients_represent_constant =
        false;
    report.constant_field_kkt.constraints_preserve_constants = false;
    report.constant_field_kkt.constraint_scope = "full_field";
    report.constant_field_kkt.constraint_scope_reason =
        "not_attempted";
    report.constant_field_kkt.jump =
        std::numeric_limits<double>::quiet_NaN();
    report.constant_field_kkt.multiplier =
        std::numeric_limits<double>::quiet_NaN();
    report.constant_field_kkt.direction_norm =
        std::numeric_limits<double>::quiet_NaN();
    report.constant_field_kkt.residual_norm =
        std::numeric_limits<double>::quiet_NaN();
    report.constant_field_kkt.relative_distance =
        std::numeric_limits<double>::quiet_NaN();
    report.constant_field_kkt.relative_orthogonality =
        std::numeric_limits<double>::quiet_NaN();
    report.constant_field_kkt.reason = "diagnostic_pending";
    state_->constraint_field.reset();
    if (!allRanks(local_diagnostic_enabled)) {
        report.range.reason =
            "diagnostic_enablement_differs_across_communicator";
        report.constant_field_kkt.reason =
            "diagnostic_enablement_differs_across_communicator";
        if (!state_->unavailable_logged &&
            communicatorRank(system_communicator) == 0) {
            detail::logConstraintRangeEnablementDiffers(method_.vocabulary());
        }
        state_->unavailable_logged = true;
        return;
    }

    // Assemble every requested diagnostic sample.  Equality of the
    // current coefficient vector, time scalars, and geometry is not a
    // complete residual-state key: supported forms may also depend on
    // solution/time-step history, auxiliary state/inputs, user data,
    // or runtime parameter callbacks.  Suppressing a full sample on a
    // partial fingerprint could therefore hide a changed constraint_field or
    // constraint-independent load and would lose accepted-state
    // provenance.  Only the
    // mixed constraint pair/LSQR work below is cached, after the current
    // constrained constraint-independent load has been reassembled and
    // compared exactly.  The qualification switch forces even that
    // safe cache off so every pair matrix and LSQR solve is repeated.
    const bool every_assembly_on_any_rank = anyRank(
        method_.reassemble_each_sample ? method_.reassemble_each_sample() : reassembleEachCertificateSample());
    const auto geometry_key =
        diagnosticGeometryKey(transient.system(), constraints);

    // The formulation declares the fields and the six sampled
    // operators plus the symmetric pair of the certificate in the
    // system's diagnostic registry; it installs them only on request.
    const auto* declaration =
        declaration_.has_value() ? &*declaration_ : nullptr;
    if (!allRanks(declaration != nullptr)) {
        report.range.reason =
            "conservative_balance_operator_unavailable";
        report.constant_field_kkt.reason =
            "conservative_balance_operator_unavailable";
        if (!state_->unavailable_logged &&
            communicatorRank(system_communicator) == 0) {
            detail::logConstraintRangeNotDeclared(method_.vocabulary(), observation);
        }
        state_->unavailable_logged = true;
        return;
    }
    std::array<std::string, 6> ops;
    const auto sampled_operators = declaration->sampledOperators();
    for (std::size_t i = 0; i < ops.size(); ++i) {
        ops[i] = sampled_operators[i]->tag;
    }
    for (const auto& op : ops) {
        const bool local_operator_installed =
            transient.system().hasOperator(op);
        if (!allRanks(local_operator_installed)) {
            report.range.reason =
                "conservative_balance_operator_unavailable";
            report.constant_field_kkt.reason =
                "conservative_balance_operator_unavailable";
            if (!state_->unavailable_logged &&
                communicatorRank(system_communicator) == 0) {
                detail::logConstraintRangeOperatorNotInstalled(method_.vocabulary(), observation, op);
            }
            state_->unavailable_logged = true;
            return;
        }
    }

    const std::string& pair_op = declaration->pair.tag;
    const bool pair_operator_available = allRanks(
        transient.system().hasOperator(pair_op));
    const bool local_representability_workspace_available =
        storage.pair != nullptr &&
        storage.vectors.load != nullptr &&
        storage.vectors.correction_load != nullptr &&
        storage.vectors.solution != nullptr &&
        storage.vectors.left_basis != nullptr &&
        storage.vectors.right_basis != nullptr &&
        storage.vectors.direction != nullptr &&
        storage.vectors.work != nullptr &&
        storage.vectors.residual != nullptr &&
        storage.vectors.normal_residual != nullptr;
    const bool representability_workspace_available = allRanks(
        local_representability_workspace_available);

    bool local_representability_layout_compatible = false;
    if (local_representability_workspace_available) {
        const auto& pair_matrix =
            *storage.pair;
        const std::array<const backends::GenericVector*, 9>
            pair_vectors{
                storage.vectors.load.get(),
                storage.vectors.correction_load
                    .get(),
                storage.vectors.solution.get(),
                storage.vectors.left_basis.get(),
                storage.vectors.right_basis.get(),
                storage.vectors.direction.get(),
                storage.vectors.work.get(),
                storage.vectors.residual.get(),
                storage.vectors.normal_residual
                    .get(),
            };
        const auto expected_size = pair_matrix.numRows();
        const auto expected_backend = pair_matrix.backendKind();
        const auto expected_local_size =
            pair_vectors.front()->localSpan().size();
        local_representability_layout_compatible =
            expected_size > 0 &&
            pair_matrix.numCols() == expected_size &&
            std::all_of(
                pair_vectors.begin(),
                pair_vectors.end(),
                [&](const backends::GenericVector* vector) {
                    return vector != nullptr &&
                           vector->backendKind() == expected_backend &&
                           vector->size() == expected_size &&
                           vector->localSpan().size() ==
                               expected_local_size;
                });
#if defined(FE_HAS_FSILS)
        // FSILS matrix-vector products require identity of the shared
        // overlap/permutation layout, not merely equal global and
        // local sizes.  Fail closed before assembly/LSQR if a caller
        // has replaced any diagnostic vector with another layout.
        if (local_representability_layout_compatible &&
            expected_backend == backends::BackendKind::FSILS) {
            const auto* fsils_matrix =
                dynamic_cast<const backends::FsilsMatrix*>(
                    &pair_matrix);
            const auto* shared =
                fsils_matrix != nullptr
                    ? fsils_matrix->shared().get()
                    : nullptr;
            local_representability_layout_compatible =
                shared != nullptr &&
                std::all_of(
                    pair_vectors.begin(),
                    pair_vectors.end(),
                    [shared](const backends::GenericVector* vector) {
                        const auto* fsils_vector =
                            dynamic_cast<const backends::FsilsVector*>(
                                vector);
                        return fsils_vector != nullptr &&
                               fsils_vector->shared() == shared;
                    });
        }
#endif
    }
    const bool representability_layout_compatible = allRanks(
        local_representability_layout_compatible);
    const bool representability_storage_usable =
        representability_workspace_available &&
        representability_layout_compatible;

    const bool constraint_field_pair_matrix_state_independent =
        pair_operator_available &&
        allRanks(transient.system().operatorMatrixStateIndependent(
            pair_op));
    const bool local_range_cache_prior_sample =
        state_->cache.valid;
    const bool local_range_cache_geometry_match =
        local_range_cache_prior_sample &&
        state_->cache.geometry ==
            geometry_key;
    const bool local_range_cache_candidate =
        !every_assembly_on_any_rank &&
        geometry_key.revision_tracking_available &&
        representability_storage_usable &&
        constraint_field_pair_matrix_state_independent &&
        local_range_cache_geometry_match;
    const bool range_cache_candidate =
        allRanks(
            local_range_cache_candidate);
    std::string range_cache_rejection_reason;
    if (every_assembly_on_any_rank) {
        range_cache_rejection_reason =
            "every_assembly_policy";
    } else if (!pair_operator_available) {
        range_cache_rejection_reason =
            "pair_operator_unavailable";
    } else if (!geometry_key.revision_tracking_available) {
        range_cache_rejection_reason =
            "revision_key_unavailable";
    } else if (!representability_storage_usable) {
        range_cache_rejection_reason =
            "storage_or_layout_unusable";
    } else if (!constraint_field_pair_matrix_state_independent) {
        range_cache_rejection_reason =
            "pair_matrix_state_dependent";
    } else if (!local_range_cache_prior_sample) {
        range_cache_rejection_reason =
            "no_prior_sample";
    } else if (!local_range_cache_geometry_match) {
        range_cache_rejection_reason =
            "geometry_or_constraints_changed";
    } else if (!range_cache_candidate) {
        range_cache_rejection_reason =
            "candidate_unavailable_on_another_rank";
    } else {
        range_cache_rejection_reason =
            "exact_load_check_pending";
    }
    if (range_cache_candidate) {
        // Preserve the previous constrained physical-potential load
        // in an LSQR
        // work vector before the current state reassembles it below.
        // A geometry key alone is deliberately insufficient: exact
        // load equality also catches state/parameter dependencies that
        // an operator may carry despite unchanged cut geometry.
        storage.vectors.work->copyFrom(
            *storage.vectors.load);
    }

    std::array<double, 6> norms{};
    for (std::size_t i = 0; i < ops.size(); ++i) {
        // Assemble the constraint-independent exterior-constraint_field plus
        // physical-potential load directly into its pair-owned RHS.
        // Production-layout and pair-layout vectors may have
        // different ghost sets or FSILS shared orderings, so
        // copyFrom/local-span positional transfer is not a valid map.
        auto& diagnostic_vector =
            i == 4u && representability_storage_usable
                ? *storage.vectors.load
                : residual_scratch;
        diagnostic_vector.zero();
        auto scratch_view = diagnostic_vector.createAssemblyView();
        FE_CHECK_NOT_NULL(
            scratch_view.get(),
            "NewtonSolver: constraint-range diagnostic view");
        systems::AssemblyRequest req;
        req.op = ops[i];
        req.want_vector = true;
        req.suppress_constraint_inhomogeneity = true;
        req.suppress_auxiliary_coupling_assembly = true;
        req.is_nonlinear_iteration = true;
        const auto ar = observation.assemble(
                method_.vocabulary().assembly_source, req, nullptr, scratch_view.get());
        FE_THROW_IF(
            !ar.success,
            FEException,
            "NewtonSolver: constraint-range diagnostic assembly failed for op '" +
                std::string(ops[i]) + "': " + ar.error_message);

        // Match the constrained residual space used by Newton.  The
        // diagnostic operators contain primary-test rows only; the
        // vector norm therefore is exactly their global constrained
        // virtual-work coefficient norm.
        zeroVectorEntries(constrained_dofs, diagnostic_vector);
        norms[i] = diagnostic_vector.norm();
        if (i == 4u && representability_storage_usable) {
            diagnostic_vector.updateGhosts();
        }
    }

    bool range_available = false;
    std::string range_reason;
    analysis::ConstraintRangeLsqrResult range{};
    analysis::ConstantFieldKktResult constant_field_kkt{};
    std::uint64_t active_constraint_field_dofs = 0u;
    int range_iteration_cap = 0;
    bool constraint_field_unit_coefficients_represent_constant = false;
    bool constraints_preserve_constants = false;
    std::string constraint_scope{"full_field"};
    std::string constraint_scope_reason{"not_attempted"};

    std::optional<FieldId> primary_field;
    std::optional<FieldId> constraint_field;
    if (!pair_operator_available) {
        range_reason =
            "pair_operator_not_installed_on_all_ranks";
    } else if (!representability_workspace_available) {
        range_reason =
            "workspace_not_allocated_on_all_ranks";
    } else if (!representability_layout_compatible) {
        range_reason =
            "pair_matrix_vector_layout_mismatch";
    } else {
        const auto& load_definition =
            transient.system().operatorDefinition(ops[4]);
        std::set<FieldId> load_test_fields;
        const auto collect_test_fields =
            [&load_test_fields](const auto& terms) {
                for (const auto& term : terms) {
                    if (term.test_field != INVALID_FIELD_ID) {
                        load_test_fields.insert(term.test_field);
                    }
                }
            };
        collect_test_fields(load_definition.cells);
        collect_test_fields(load_definition.boundary);
        collect_test_fields(load_definition.interior);
        collect_test_fields(load_definition.interface_faces);
        collect_test_fields(load_definition.cut_volumes);
        if (load_test_fields.size() == 1u) {
            primary_field = *load_test_fields.begin();
        }

        const auto& pair_definition =
            transient.system().operatorDefinition(pair_op);
        std::set<FieldId> pair_fields;
        const auto collect_pair_fields =
            [&pair_fields](const auto& terms) {
                for (const auto& term : terms) {
                    if (term.test_field != INVALID_FIELD_ID) {
                        pair_fields.insert(term.test_field);
                    }
                    if (term.trial_field != INVALID_FIELD_ID) {
                        pair_fields.insert(term.trial_field);
                    }
                }
            };
        collect_pair_fields(pair_definition.cells);
        collect_pair_fields(pair_definition.boundary);
        collect_pair_fields(pair_definition.interior);
        collect_pair_fields(pair_definition.interface_faces);
        collect_pair_fields(pair_definition.cut_volumes);
        if (primary_field.has_value()) {
            pair_fields.erase(*primary_field);
        }
        if (pair_fields.size() == 1u) {
            constraint_field = *pair_fields.begin();
        }
        // The operator terms must confirm the declared fields.
        if (primary_field.has_value() &&
            *primary_field != declaration->primary_field) {
            primary_field.reset();
        }
        if (constraint_field.has_value() &&
            *constraint_field != declaration->constraint_field) {
            constraint_field.reset();
        }

        const bool fields_identified_on_all_ranks = allRanks(
            primary_field.has_value() && constraint_field.has_value());
        if (!fields_identified_on_all_ranks) {
            range_reason =
                "mixed_primary_constraint_fields_not_identifiable";
        } else {
            state_->constraint_field =
                *constraint_field;
            report.range.constraint_field =
                *constraint_field;
        }
    }

    if (range_reason.empty()) {
        const auto constraint_begin =
            transient.system().fieldDofOffset(*constraint_field);
        const auto constraint_count =
            transient.system()
                .fieldDofHandler(*constraint_field)
                .getNumDofs();
        const auto constraint_field_end = constraint_begin + constraint_count;
        const auto& constraint_field_record =
            transient.system().fieldRecord(*constraint_field);
        const bool local_unit_coefficients_represent_constant =
            constraint_field_record.components == 1 &&
            constraint_field_record.space != nullptr &&
            analysis::unitCoefficientsRepresentConstant(
                *constraint_field_record.space);
        constraint_field_unit_coefficients_represent_constant = allRanks(
            local_unit_coefficients_represent_constant);
        if (constraint_field_unit_coefficients_represent_constant) {
            constraints_preserve_constants = allRanks(
                analysis::constraintsPreserveUnitField(
                    constraints,
                    constraint_begin,
                    constraint_count));
            constraint_scope_reason =
                constraints_preserve_constants
                    ? "available"
                    : "pair_support_unavailable";
            if (!constraints_preserve_constants) {
                const auto support = analysis::retainedFieldSupport(
                    transient.system(),
                    transient.system().operatorDefinition(pair_op),
                    *constraint_field,
                    constraint_begin,
                    constraint_count,
                    analysis::CertificateCollectives{
                        .all_ranks = allRanks,
                        .any_rank = anyRank,
                        .agrees_on_unsigned =
                            [&](std::uint64_t value) {
                                return communicatorAgreesOnUnsigned(
                                    value, system_communicator);
                            },
                        .rank = communicatorRank(system_communicator),
                        .size = communicatorSize(system_communicator)});
                constraint_scope_reason = support.reason;
                if (support.support.has_value()) {
                    constraint_scope = "retained_volume";
                    constraint_scope_reason = "available";
                    constraints_preserve_constants =
                        allRanks(
                            analysis::constraintsPreserveUnitField(
                                constraints,
                                constraint_begin,
                                constraint_count,
                                &*support.support));
                }
            }
        }

        bool local_nonzero_constraint_field_inhomogeneity = false;
        constraints.forEach(
            [&](const constraints::AffineConstraints::ConstraintView&
                    line) {
                if (line.slave_dof >= constraint_begin &&
                    line.slave_dof < constraint_field_end &&
                    (!std::isfinite(line.inhomogeneity) ||
                     line.inhomogeneity != 0.0)) {
                    local_nonzero_constraint_field_inhomogeneity = true;
                }
            });
        if (anyRank(local_nonzero_constraint_field_inhomogeneity)) {
            range_reason =
                "nonzero_constraint_field_inhomogeneity";
        }

        const auto& owned = transient.system()
                                .dofHandler()
                                .getPartition()
                                .locallyOwned();
        std::uint64_t local_active_constraint_field_dofs = 0u;
        for (GlobalIndex dof = constraint_begin; dof < constraint_field_end;
             ++dof) {
            if (owned.contains(dof) &&
                !std::binary_search(constrained_dofs.begin(),
                                    constrained_dofs.end(),
                                    dof)) {
                ++local_active_constraint_field_dofs;
            }
        }
        active_constraint_field_dofs = local_active_constraint_field_dofs;
#if FE_HAS_MPI
        {
            int initialized = 0;
            int finalized = 0;
            MPI_Initialized(&initialized);
            MPI_Finalized(&finalized);
            if (initialized != 0 && finalized == 0) {
                std::uint64_t global_active_constraint_field_dofs = 0u;
                MPI_Allreduce(&local_active_constraint_field_dofs,
                              &global_active_constraint_field_dofs,
                              1,
                              MPI_UINT64_T,
                              MPI_SUM,
                              system_communicator);
                active_constraint_field_dofs =
                    global_active_constraint_field_dofs;
            }
        }
#endif
        // In exact arithmetic Golub--Kahan LSQR terminates within the
        // constraint space dimension.  Finite-precision loss of Krylov
        // orthogonality can require more than n iterations, however;
        // Algorithm 583 recommends 4*n when the operator is not known
        // to be well conditioned.  Retain the existing hard ceiling so
        // this diagnostic cannot acquire unbounded solve cost.
        range_iteration_cap =
            static_cast<int>(
                4u * std::min<std::uint64_t>(
                         active_constraint_field_dofs, 250u));
    }

    bool range_cache_hit = false;
    bool range_cache_load_equality_checked = false;
    bool range_cache_load_equal = false;
    double range_cache_load_difference_norm =
        std::numeric_limits<double>::quiet_NaN();
    if (range_reason.empty() &&
        range_cache_candidate) {
        axpy(*storage.vectors.work,
             static_cast<Real>(-1.0),
             *storage.vectors.load);
        range_cache_load_difference_norm =
            storage.vectors.work->norm();
        range_cache_load_equality_checked = true;
        range_cache_load_equal = allRanks(
            std::isfinite(
                range_cache_load_difference_norm) &&
            range_cache_load_difference_norm ==
                0.0);
        range_cache_hit =
            range_cache_load_equal;
        range_cache_rejection_reason =
            range_cache_hit
                ? "none"
                : "load_changed_or_nonfinite";
        if (range_cache_hit) {
            range =
                state_->cache.result;
            range_available = true;
        }
    }

    if (range_cache_rejection_reason ==
            "exact_load_check_pending" &&
        !range_cache_load_equality_checked) {
        range_cache_rejection_reason =
            "diagnostic_unavailable_before_load_check";
    }

    if (range_reason.empty() &&
        !range_cache_hit) {
        auto& pair_matrix =
            *storage.pair;
        pair_matrix.zero();
        auto pair_view = pair_matrix.createAssemblyView();
        FE_CHECK_NOT_NULL(
            pair_view.get(),
            "NewtonSolver: constraint-range pair matrix view");
        systems::AssemblyRequest pair_request;
        pair_request.op = pair_op;
        pair_request.want_matrix = true;
        pair_request.want_vector = false;
        pair_request.suppress_constraint_inhomogeneity = true;
        pair_request.suppress_auxiliary_coupling_assembly = true;
        pair_request.is_nonlinear_iteration = true;
        const auto pair_assembly = observation.assemble(
                method_.vocabulary().assembly_source, pair_request, pair_view.get(), nullptr);
        FE_THROW_IF(
            !pair_assembly.success,
            FEException,
            "NewtonSolver: constraint-range pair assembly failed for op '" +
                pair_op + "': " + pair_assembly.error_message);

        // Affine substitution has already moved slave-column and
        // slave-test effects onto masters.  The only remaining slave
        // matrix entries are artificial identity diagonals stamped by
        // constrained assembly; remove them instead of allowing LSQR
        // to interpret constraints as physical constraint_field work.
        auto constrained_pair_view =
            pair_matrix.createAssemblyView();
        FE_CHECK_NOT_NULL(
            constrained_pair_view.get(),
            "NewtonSolver: constraint-range constrained pair view");
        constrained_pair_view->beginAssemblyPhase();
        constrained_pair_view->zeroRows(
            constrained_dofs, /*set_diagonal=*/false);
        constrained_pair_view->finalizeAssembly();
        zeroVectorEntries(
            constrained_dofs,
            *storage.vectors.load);
        storage.vectors.load->updateGhosts();

        range =
            analysis::solveConstraintRangeLsqr(
                pair_matrix,
                *storage.vectors.load,
                *storage.vectors.solution,
                *storage.vectors.left_basis,
                *storage.vectors.right_basis,
                *storage.vectors.direction,
                *storage.vectors.work,
                *storage.vectors.residual,
                *storage.vectors.normal_residual,
                range_iteration_cap,
                method_.max_range_distance,
                allRanks);
        range_available = true;
        if (geometry_key.revision_tracking_available &&
            constraint_field_pair_matrix_state_independent) {
            state_->cache.valid =
                true;
            state_->cache.geometry =
                geometry_key;
            state_->cache.result =
                range;
        }
    }

    if (!range_available) {
        constant_field_kkt.reason =
            range_reason.empty()
                ? "constraint_pair_or_load_unavailable"
                : std::string(method_.vocabulary().reason(range_reason));
    } else if (!constraint_field_unit_coefficients_represent_constant) {
        constant_field_kkt.reason =
            "constraint_space_unit_coefficients_do_not_represent_constant";
    } else if (!constraints_preserve_constants) {
        constant_field_kkt.reason =
            "field_constraints_do_not_preserve_constants";
    } else if (!primary_field.has_value() ||
               !constraint_field.has_value()) {
        constant_field_kkt.reason =
            "mixed_primary_constraint_fields_not_identifiable";
    } else {
        const auto constraint_begin =
            transient.system().fieldDofOffset(*constraint_field);
        const auto constraint_count =
            transient.system()
                .fieldDofHandler(*constraint_field)
                .getNumDofs();
        const auto primary_begin =
            transient.system().fieldDofOffset(*primary_field);
        const auto primary_count =
            transient.system()
                .fieldDofHandler(*primary_field)
                .getNumDofs();
        constant_field_kkt = analysis::evaluateConstantFieldKkt(
            *storage.pair,
            *storage.vectors.load,
            *storage.vectors.left_basis,
            *storage.vectors.right_basis,
            *storage.vectors.direction,
            constraint_begin,
            constraint_count,
            primary_begin,
            primary_count,
            constrained_dofs,
            ownedDofsForVector(
                *storage.vectors.left_basis,
                transient.system()
                    .dofHandler()
                    .getPartition()
                    .locallyOwned()),
            allRanks);
    }

    report.range.available =
        range_available;
    report.range.converged =
        range_available &&
        range.converged;
    report.range.breakdown =
        range_available &&
        range.breakdown;
    report.range.residual_norm =
        range_available
            ? range.residual_norm
            : std::numeric_limits<double>::quiet_NaN();
    report.range.relative_distance =
        range_available
            ? range.relative_residual
            : std::numeric_limits<double>::quiet_NaN();
    report.range.reason =
        range_available
            ? "available"
            : std::string(method_.vocabulary().reason(range_reason));
    report.constant_field_kkt.available =
        constant_field_kkt.available;
    report.constant_field_kkt.unit_coefficients_represent_constant =
        constraint_field_unit_coefficients_represent_constant;
    report.constant_field_kkt.constraints_preserve_constants =
        constraints_preserve_constants;
    report.constant_field_kkt.constraint_scope =
        constraint_scope;
    report.constant_field_kkt.constraint_scope_reason =
        constraint_scope_reason;
    report.constant_field_kkt.jump =
        constant_field_kkt.jump;
    report.constant_field_kkt.multiplier =
        constant_field_kkt.multiplier;
    report.constant_field_kkt.direction_norm =
        constant_field_kkt.direction_norm;
    report.constant_field_kkt.residual_norm =
        constant_field_kkt.residual_norm;
    report.constant_field_kkt.relative_distance =
        constant_field_kkt.relative_distance;
    report.constant_field_kkt.relative_orthogonality =
        constant_field_kkt.relative_orthogonality;
    report.constant_field_kkt.reason =
        method_.vocabulary().reason(constant_field_kkt.reason);

    if (communicatorRank(system_communicator) == 0) {
        detail::logConstraintRangeSample(method_.vocabulary(),
            observation,
            communicatorRank(system_communicator),
            norms,
            detail::ConstraintRangeCacheTrace{
                .revision_key_available = geometry_key.revision_tracking_available,
                .every_assembly = every_assembly_on_any_rank,
                .pair_matrix_state_independent = constraint_field_pair_matrix_state_independent,
                .prior_sample = local_range_cache_prior_sample,
                .geometry_match = local_range_cache_geometry_match,
                .candidate = range_cache_candidate,
                .load_equality_checked = range_cache_load_equality_checked,
                .load_equal = range_cache_load_equal,
                .hit = range_cache_hit,
                .load_difference_norm = range_cache_load_difference_norm,
                .rejection_reason = range_cache_rejection_reason},
            report,
            range,
            active_constraint_field_dofs,
            range_iteration_cap);
    }
}

bool ConstraintRangeCertificate::requiresAcceptedObservation() const
{
    return method_.max_range_distance.has_value() ||
           method_.max_constant_field_kkt_distance.has_value();
}

std::vector<CertificateDecision> ConstraintRangeCertificate::assess(
    timestepping::NewtonAcceptance& acceptance)
{
    auto& candidate = mutableRecord(acceptance.candidate);
    const auto system_communicator = systemCommunicator(acceptance.system);
    std::vector<CertificateDecision> decisions;

    // The constrained constraint space distance of the accepted static state.
    if (const auto threshold = method_.max_range_distance; threshold.has_value()) {
        bool local_passed = true;
        const char* reason = "within_threshold";
        if (!candidate.sampled) {
            local_passed = false;
            reason = "diagnostic_not_sampled";
        } else if (!candidate.range.available) {
            local_passed = false;
            reason = "diagnostic_unavailable";
        } else if (candidate.range.breakdown) {
            local_passed = false;
            reason = "diagnostic_breakdown";
        } else if (!candidate.range.converged) {
            local_passed = false;
            reason = "normal_equation_not_stationary";
        } else if (!std::isfinite(candidate.range.relative_distance)) {
            local_passed = false;
            reason = "relative_distance_nonfinite";
        } else if (candidate.range.relative_distance > *threshold) {
            local_passed = false;
            reason = "relative_distance_exceeds_threshold";
        }
        const bool passed = !anyRankOf(system_communicator, !local_passed);
        if (!passed && local_passed) {
            reason = "remote_rank_rejected";
        }

        candidate.range_gate.applied = true;
        candidate.range_gate.passed = passed;
        candidate.range_gate.max_relative_distance = *threshold;
        if (communicatorRank(system_communicator) == 0) {
            detail::logConstraintRangeDistanceGate(method_.vocabulary(), acceptance, candidate, *threshold,
                                                    passed, method_.vocabulary().reason(reason));
        }
        decisions.push_back(CertificateDecision{
            .id = "constraint_range.range_distance",
            .status = gateStatus(passed, reason),
            .reason = std::string(method_.vocabulary().reason(reason)),
            .value = candidate.range.relative_distance,
            .threshold = *threshold});
    }

    // The constant-field KKT distance of the accepted static state.
    if (const auto threshold = method_.max_constant_field_kkt_distance; threshold.has_value()) {
        bool local_passed = true;
        const char* reason = "within_threshold";
        if (!candidate.sampled) {
            local_passed = false;
            reason = "diagnostic_not_sampled";
        } else if (!candidate.constant_field_kkt.available) {
            local_passed = false;
            reason = "constant_field_kkt_unavailable";
        } else if (!std::isfinite(candidate.constant_field_kkt.relative_distance)) {
            local_passed = false;
            reason = "relative_distance_nonfinite";
        } else if (candidate.constant_field_kkt.relative_distance > *threshold) {
            local_passed = false;
            reason = "relative_distance_exceeds_threshold";
        }
        const bool passed = !anyRankOf(system_communicator, !local_passed);
        if (!passed && local_passed) {
            reason = "remote_rank_rejected";
        }

        candidate.constant_field_kkt_gate.applied = true;
        candidate.constant_field_kkt_gate.passed = passed;
        candidate.constant_field_kkt_gate.max_relative_distance = *threshold;
        if (communicatorRank(system_communicator) == 0) {
            detail::logConstantFieldKktDistanceGate(method_.vocabulary(), acceptance, candidate, *threshold,
                                                     passed, method_.vocabulary().reason(reason));
        }
        decisions.push_back(CertificateDecision{
            .id = "constraint_range.constant_field_kkt_distance",
            .status = gateStatus(passed, reason),
            .reason = std::string(method_.vocabulary().reason(reason)),
            .value = candidate.constant_field_kkt.relative_distance,
            .threshold = *threshold});
    }
    return decisions;
}

void ConstraintRangeCertificate::logInitializer(const systems::FESystem& system,
                                                timestepping::NewtonReport& newton_report,
                                                bool applied,
                                                bool passed,
                                                std::string_view reason) const
{
    auto& record = mutableRecord(newton_report);
    record.initializer.applied = applied;
    record.initializer.passed = passed;
    record.initializer.reason = method_.vocabulary().reason(reason);
    if (communicatorRank(systemCommunicator(system)) == 0) {
        detail::logConstraintRangeInitializer(method_.vocabulary(), record,
                                              *method_.max_range_distance,
                                              state_->correction.lsqr,
                                              state_->correction.load_norm,
                                              applied,
                                              passed,
                                              record.initializer.reason);
    }
}

void ConstraintRangeCertificate::reportConsumed(const systems::FESystem& system,
                                                timestepping::NewtonReport& report)
{
    mutableRecord(report).initializer.requested = true;
    state_->correction = {};
    logInitializer(system, report, /*applied=*/false, /*passed=*/true, "already_initialized");
}

bool ConstraintRangeCertificate::propose(timestepping::NewtonInitialization& initialization)
{
    auto& observation = initialization.observation;
    auto& newton_report = observation.report;
    auto& report = mutableRecord(newton_report);
    auto& transient = observation.transient;
    auto& current = initialization.current;
    const auto constrained_dofs = observation.constrained_dofs;
    const auto system_communicator = systemCommunicator(transient.system());
    const auto allRanks = [system_communicator](bool local_value) {
        return !anyRankOf(system_communicator, !local_value);
    };
    report.initializer.requested = true;
    state_->correction = {};
    state_->proposal_storage = nullptr;
    auto& correction_state = state_->correction;
    const double threshold = *method_.max_range_distance;

    // The certificate of the synchronized current guess.
    auto sample = observation;
    sample.phase = method_.vocabulary().initializer_phase;
    observe(sample);

    auto* storage = dynamic_cast<ConstraintRangeStorage*>(observation.storage);
    std::string_view failure_reason;
    const bool local_storage_available =
        storage != nullptr &&
        storage->pair != nullptr &&
        storage->vectors.load != nullptr &&
        storage->vectors.correction_load != nullptr &&
        storage->vectors.solution != nullptr &&
        storage->vectors.left_basis != nullptr &&
        storage->vectors.right_basis != nullptr &&
        storage->vectors.direction != nullptr &&
        storage->vectors.work != nullptr &&
        storage->vectors.residual != nullptr &&
        storage->vectors.normal_residual != nullptr &&
        state_->constraint_field.has_value();
    if (!report.sampled) {
        failure_reason = "diagnostic_not_sampled";
    } else if (!report.range.available) {
        failure_reason = "diagnostic_unavailable";
    } else if (report.range.breakdown) {
        failure_reason = "diagnostic_breakdown";
    } else if (!report.range.converged) {
        failure_reason = "normal_equation_not_stationary";
    } else if (!std::isfinite(report.range.relative_distance)) {
        failure_reason = "relative_distance_nonfinite";
    } else if (report.range.relative_distance > threshold) {
        failure_reason = "relative_distance_exceeds_threshold";
    } else if (!allRanks(local_storage_available)) {
        failure_reason = "constraint_solution_storage_unavailable";
    }
    if (!failure_reason.empty()) {
        logInitializer(transient.system(), newton_report, /*applied=*/false, /*passed=*/false, failure_reason);
        return false;
    }

    const auto constraint_field = *state_->constraint_field;
    const auto constraint_begin = transient.system().fieldDofOffset(constraint_field);
    const auto constraint_count =
        transient.system().fieldDofHandler(constraint_field).getNumDofs();
    FE_THROW_IF(
        constraint_begin < 0 || constraint_count <= 0 ||
            constraint_begin + constraint_count > current.size(),
        systems::InvalidStateException,
        "NewtonSolver: static compatible constraint-field initializer found an invalid constraint-field DOF range");
    std::vector<GlobalIndex> constraint_field_dofs(static_cast<std::size_t>(constraint_count));
    std::iota(constraint_field_dofs.begin(), constraint_field_dofs.end(), constraint_begin);

    auto& compatible_constraint_field = *storage->vectors.solution;
    auto& pair_layout_work = *storage->vectors.work;
    compatible_constraint_field.updateGhosts();

    // The representability solve above certifies the complete
    // constraint-independent physical load.  The one-shot initializer is
    // additive, so solve a second LSQR problem for the balance remaining
    // after the constraint_field already present at entry.  This retains the
    // current representative of a compatible constraint space, avoids
    // counting exterior or existing constraint load twice, and leaves an
    // already compatible state unchanged.
    auto& correction_load = *storage->vectors.correction_load;

    transferOwnedEntries(pair_layout_work, current, system_communicator);
    correction_load.copyFrom(pair_layout_work);
    zeroVectorEntries(constraint_field_dofs, correction_load);
    axpy(pair_layout_work, Real{-1.0}, correction_load);
    pair_layout_work.updateGhosts();

    auto& pair_matrix = *storage->pair;
    pair_matrix.mult(pair_layout_work, correction_load);
    correction_load.markModified();
    axpy(correction_load, Real{1.0}, *storage->vectors.load);
    zeroVectorEntries(constrained_dofs, correction_load);
    correction_load.updateGhosts();
    correction_state.load_norm = correction_load.norm();
    const auto correction_iteration_cap = static_cast<int>(
        4u * std::min<std::uint64_t>(static_cast<std::uint64_t>(constraint_count), 250u));
    const auto correction = solveConstraintRangeLsqr(
        pair_matrix,
        correction_load,
        compatible_constraint_field,
        *storage->vectors.left_basis,
        *storage->vectors.right_basis,
        *storage->vectors.direction,
        pair_layout_work,
        *storage->vectors.residual,
        *storage->vectors.normal_residual,
        correction_iteration_cap,
        std::nullopt,
        allRanks);
    correction_state.lsqr = correction;
    if (correction.breakdown) {
        logInitializer(transient.system(), newton_report, false, false, "constraint_correction_breakdown");
        return false;
    }
    if (!correction.converged) {
        logInitializer(transient.system(), newton_report, false, false, "constraint_correction_not_stationary");
        return false;
    }
    if (!allRanks(std::isfinite(correction_state.load_norm) &&
                  std::isfinite(correction.residual_norm) &&
                  std::isfinite(correction.normal_residual_norm) &&
                  std::isfinite(correction.solution_norm))) {
        logInitializer(transient.system(), newton_report, false, false, "constraint_correction_nonfinite");
        return false;
    }
    const double nonincrease_tolerance =
        1000.0 * std::numeric_limits<Real>::epsilon() *
        std::max({1.0, correction_state.load_norm, correction.residual_norm});
    if (!allRanks(correction.residual_norm <=
                  correction_state.load_norm + nonincrease_tolerance)) {
        logInitializer(transient.system(), newton_report, false, false, "constraint_correction_increased_residual");
        return false;
    }

    // A valid mixed-pair LSQR recurrence places the solution entirely in
    // the constraint block.  Check that invariant explicitly before the
    // monolithic state is touched; this prevents an accidentally polluted
    // diagnostic pair from changing primary or any auxiliary field.
    pair_layout_work.copyFrom(compatible_constraint_field);
    zeroVectorEntries(constraint_field_dofs, pair_layout_work);
    const double other_field_solution_norm = pair_layout_work.norm();
    const double constraint_field_solution_norm = compatible_constraint_field.norm();
    const double subspace_tolerance =
        100.0 * std::numeric_limits<Real>::epsilon() * std::max(1.0, constraint_field_solution_norm);
    if (!allRanks(std::isfinite(other_field_solution_norm) &&
                  other_field_solution_norm <= subspace_tolerance)) {
        logInitializer(transient.system(), newton_report, false, false, "solution_outside_constraint_subspace");
        return false;
    }

    // Remove the verified roundoff-sized non-constraint_field component before
    // the addition so every other field remains bitwise unchanged.
    axpy(compatible_constraint_field, Real{-1.0}, pair_layout_work);
    compatible_constraint_field.updateGhosts();
    state_->proposal_storage = storage;
    return true;
}

void ConstraintRangeCertificate::apply(const systems::FESystem& system,
                                       backends::GenericVector& current)
{
    const auto communicator = systemCommunicator(system);
    auto* storage = state_->proposal_storage;
    FE_CHECK_NOT_NULL(storage, "ConstraintRangeCertificate: no proposed correction");
    // Pair and production vectors may have different ghost maps.  Transfer
    // their common authoritative owned rows in public numbering, then add
    // the verified residual-correction constraint block to the existing
    // current state.  Committed history and rate slots are not touched.
    auto& pair_layout_work = *storage->vectors.work;
    transferOwnedEntries(pair_layout_work, current, communicator);
    axpy(pair_layout_work, Real{1.0}, *storage->vectors.solution);
    transferOwnedEntries(current, pair_layout_work, communicator);
    state_->proposal_storage = nullptr;
}

void ConstraintRangeCertificate::reportApplied(const systems::FESystem& system,
                                               timestepping::NewtonReport& report)
{
    logInitializer(system, report, /*applied=*/true, /*passed=*/true,
                   "additive_initial_guess_within_threshold");
}

ConstraintRangeProbeEvidence ConstraintRangeCertificate::probeEvidence(
    const timestepping::NewtonReport& report,
    const systems::FESystem& system,
    bool constant_field_is_physical,
    FieldId distinct_field) const
{
    using Defect = ConstraintRangeProbeEvidence::Defect;
#if FE_HAS_MPI
    const systems::CollectiveCommunicator comm(system.activeMpiCommunicator());
#else
    (void)system;
    const systems::CollectiveCommunicator comm{};
#endif
    const auto any_rank = [&](bool local) { return systems::collectiveAnyTrue(local, comm); };
    const auto identical = [&](double local) {
        return systems::collectiveMinDouble(local, comm) ==
               systems::collectiveMaxDouble(local, comm);
    };
    const auto flag = [](bool value) { return value ? 1.0 : 0.0; };

    const auto& record = this->record(report);
    ConstraintRangeProbeEvidence evidence;
    evidence.production_residual_norm = report.residual_norm;
    const auto fail = [&](Defect defect) {
        evidence.defect = defect;
        return evidence;
    };

    if (any_rank(!std::isfinite(report.residual_norm))) {
        return fail(Defect::ProductionResidualNonfinite);
    }
    if (!identical(report.residual_norm)) {
        return fail(Defect::ProductionResidualDiffersAcrossRanks);
    }
    const bool local_kkt_scalar_invalid =
        record.constant_field_kkt.available &&
        (!std::isfinite(record.constant_field_kkt.residual_norm) ||
         !std::isfinite(record.constant_field_kkt.relative_distance) ||
         !std::isfinite(record.constant_field_kkt.jump) ||
         !std::isfinite(record.constant_field_kkt.multiplier));
    if (any_rank(local_kkt_scalar_invalid)) {
        return fail(Defect::ConstantFieldKktNonfinite);
    }
    const bool local_sampled = record.sampled;
    const bool local_range_available =
        local_sampled && record.range.available;
    const bool local_field_invalid =
        local_range_available &&
        (record.range.constraint_field == INVALID_FIELD_ID ||
         record.range.constraint_field == distinct_field);
    if (any_rank(local_field_invalid)) {
        return fail(Defect::ConstraintFieldInvalid);
    }
    const bool local_range_scalar_invalid =
        local_range_available &&
        (!std::isfinite(record.range.residual_norm) ||
         !std::isfinite(record.range.relative_distance));
    if (any_rank(local_range_scalar_invalid)) {
        return fail(Defect::RangeNonfinite);
    }
    const bool local_kkt_available = local_sampled && record.constant_field_kkt.available;
    const bool sampled_identical = identical(flag(local_sampled));
    const bool range_available_identical = identical(flag(local_range_available));
    const bool field_identical =
        identical(static_cast<double>(record.range.constraint_field));
    const bool kkt_available_identical = identical(flag(local_kkt_available));
    const bool range_convergence_identical =
        identical(flag(record.range.converged)) &&
        identical(flag(record.range.breakdown));
    const bool constant_trace_flags_identical =
        identical(flag(record.constant_field_kkt.unit_coefficients_represent_constant)) &&
        identical(flag(record.constant_field_kkt.constraints_preserve_constants));
    if (!sampled_identical || !range_available_identical || !field_identical ||
        !kkt_available_identical || !range_convergence_identical ||
        !constant_trace_flags_identical) {
        return fail(Defect::AvailabilityDiffersAcrossRanks);
    }
    if (local_range_available) {
        const bool range_scalars_identical =
            identical(record.range.residual_norm) &&
            identical(record.range.relative_distance);
        if (!range_scalars_identical) {
            return fail(Defect::RangeDiffersAcrossRanks);
        }
    }
    if (local_kkt_available) {
        const bool kkt_scalars_identical =
            identical(record.constant_field_kkt.residual_norm) &&
            identical(record.constant_field_kkt.relative_distance) &&
            identical(record.constant_field_kkt.jump) &&
            identical(record.constant_field_kkt.multiplier);
        if (!kkt_scalars_identical) {
            return fail(Defect::ConstantFieldKktDiffersAcrossRanks);
        }
    }

    evidence.range_available = local_range_available;
    evidence.range_converged =
        local_range_available && record.range.converged;
    evidence.range_breakdown =
        local_range_available && record.range.breakdown;
    evidence.range_residual_norm = record.range.residual_norm;
    evidence.range_relative_distance = record.range.relative_distance;
    evidence.range_reason = record.range.reason;
    evidence.constraint_field = record.range.constraint_field;
    evidence.constant_field_kkt_required =
        constant_field_is_physical &&
        record.constant_field_kkt.unit_coefficients_represent_constant &&
        record.constant_field_kkt.constraints_preserve_constants;
    evidence.constant_field_kkt_available = local_kkt_available;
    evidence.constant_field_kkt_residual_norm = record.constant_field_kkt.residual_norm;
    evidence.constant_field_kkt_relative_distance = record.constant_field_kkt.relative_distance;
    evidence.constant_field_jump = record.constant_field_kkt.jump;
    evidence.constant_field_multiplier = record.constant_field_kkt.multiplier;
    return evidence;
}

NewtonAnalysisMethod newtonAnalysisMethodFromEnvironment()
{
    NewtonAnalysisMethod method;
    method.observe_field_residual = fieldResidualObserverSwitchedOn();
    method.observe_row_contributions = rowContributionObserverSwitchedOn();
    method.constraint_range = constraintRangeMethodFromEnvironment();
    return method;
}

namespace {

/// The first declaration of type Declaration of the formulations of
/// @p system, or none.
template <class Declaration>
[[nodiscard]] std::optional<Declaration> declarationOf(const systems::FESystem& system)
{
    const auto* declaration = systems::findDiagnosticDeclaration<Declaration>(system);
    return declaration != nullptr ? std::optional<Declaration>(*declaration) : std::nullopt;
}

} // namespace

NewtonAnalysisExtensions buildNewtonAnalysisExtensions(
    const systems::FESystem& system,
    const NewtonAnalysisMethod& method)
{
    NewtonAnalysisExtensions analysis;
    auto& extensions = analysis.extensions;
    if (method.observe_field_residual) {
        extensions.observers.push_back(std::make_shared<FieldResidualObserver>());
    }
    if (method.observe_row_contributions) {
        extensions.observers.push_back(std::make_shared<RowContributionObserver>(
            declarationOf<RowContributionDeclaration>(system)));
    }
    if (constraintRangeMethodRequested(method.constraint_range)) {
        auto certificate = std::make_shared<ConstraintRangeCertificate>(
            declarationOf<ConstraintRangeDeclaration>(system), method.constraint_range);
        extensions.observers.push_back(certificate);
        extensions.acceptance_checks.push_back(certificate);
        extensions.initializers.push_back(certificate);
        analysis.constraint_range = std::move(certificate);
    }
    return analysis;
}

timestepping::NewtonOptions constraintRangeProbeOptions(
    const systems::OperatorTag& op,
    const NewtonAnalysisExtensions& analysis)
{
    const auto& certificate = analysis.constraint_range;
    FE_CHECK_NOT_NULL(certificate.get(), "constraintRangeProbeOptions: certificate");
    timestepping::NewtonOptions options;
    options.residual_op = op;
    options.jacobian_op = op;
    options.max_iterations = 1;
    options.min_iterations = 0;
    options.abs_tolerance = std::numeric_limits<double>::max();
    options.rel_tolerance = 0.0;
    options.step_tolerance = 0.0;
    options.use_line_search = false;
    options.assemble_both_when_possible = false;
    options.initial_residual_only_certificate = !certificate->enabled();
    options.external_state_fixed_point.enabled = false;
    options.extensions = analysis.extensions;
    return options;
}

} // namespace analysis
} // namespace FE
} // namespace svmp
