/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ConstraintRangeCertificate.h"

#include "Assembly/CutIntegrationContext.h"
#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/GenericMatrix.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/SmallCutAggregationConstraint.h"
#include "Core/FEException.h"
#include "Geometry/CutQuadrature.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {
namespace {

bool environmentFlagEnabled(const char* name) noexcept
{
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    std::string v(env);
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(v == "0" || v == "false" || v == "off" || v == "no");
}

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
    FieldId pressure_field)
{
    bool unsupported_pressure_term = !pair.global.empty();
    const auto pressure_bearing = [pressure_field](const auto& term) {
        return term.test_field == pressure_field ||
               term.trial_field == pressure_field;
    };
    const auto inspect_nonvolume = [&](const auto& terms) {
        unsupported_pressure_term =
            unsupported_pressure_term ||
            std::any_of(terms.begin(), terms.end(), pressure_bearing);
    };
    inspect_nonvolume(pair.cells);
    inspect_nonvolume(pair.boundary);
    inspect_nonvolume(pair.interior);
    inspect_nonvolume(pair.interface_faces);

    std::optional<ConstantFieldCutVolumeBinding> binding;
    for (const auto& term : pair.cut_volumes) {
        if (!pressure_bearing(term)) {
            continue;
        }
        if (term.marker < 0 ||
            term.side == geometry::CutIntegrationSide::Interface) {
            unsupported_pressure_term = true;
            continue;
        }
        const ConstantFieldCutVolumeBinding candidate{
            term.marker, term.side};
        if (!binding.has_value()) {
            binding = candidate;
        } else if (binding->marker != candidate.marker ||
                   binding->side != candidate.side) {
            unsupported_pressure_term = true;
        }
    }
    if (unsupported_pressure_term || !binding.has_value()) {
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
    backends::GenericVector& pressure,
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
    pressure.zero();
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
        pressure.updateGhosts();
        pair.mult(pressure, residual);
        residual.markModified();
        axpy(residual, Real{1.0}, load);
        residual.updateGhosts();
        pair.mult(residual, normal_residual);
        normal_residual.markModified();

        result.residual_norm = residual.norm();
        result.normal_residual_norm = normal_residual.norm();
        result.solution_norm = pressure.norm();
        return vector_is_finite(residual) &&
               vector_is_finite(normal_residual) &&
               vector_is_finite(pressure) &&
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
    // velocity and pressure subspaces.
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
            // Apply the matching correction on the pressure-space basis.
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

        axpy(pressure, static_cast<Real>(solution_scale), direction);
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
                axpy(pressure,
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
    const spaces::FunctionSpace& pressure_space)
{
    try {
        if (pressure_space.field_type() != FieldType::Scalar ||
            pressure_space.value_dimension() != 1 ||
            pressure_space.dofs_per_element() == 0u ||
            pressure_space.is_variable_order()) {
            return false;
        }
        const auto quadrature = pressure_space.element().quadrature();
        if (!quadrature || quadrature->num_points() == 0u) {
            return false;
        }
        const std::vector<Real> unit_coefficients(
            pressure_space.dofs_per_element(), Real{1.0});
        const Real tolerance =
            Real{1000.0} * std::numeric_limits<Real>::epsilon() *
            std::max(Real{1.0},
                     static_cast<Real>(pressure_space.dofs_per_element()));
        for (std::size_t point = 0;
             point < quadrature->num_points();
             ++point) {
            const auto value = pressure_space.evaluate(
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
    GlobalIndex pressure_begin,
    GlobalIndex pressure_count,
    const std::set<GlobalIndex>* retained_pressure_support)
{
    if (pressure_begin < 0 || pressure_count <= 0) {
        return false;
    }
    const GlobalIndex pressure_end = pressure_begin + pressure_count;
    bool compatible = true;
    affine_constraints.forEach(
        [&](const constraints::AffineConstraints::ConstraintView& line) {
            const bool pressure_slave =
                line.slave_dof >= pressure_begin &&
                line.slave_dof < pressure_end;
            bool has_pressure_master = false;
            bool has_nonpressure_master = false;
            double weight_sum = 0.0;
            double absolute_weight_sum = 0.0;
            for (const auto& entry : line.entries) {
                const bool pressure_master =
                    entry.master_dof >= pressure_begin &&
                    entry.master_dof < pressure_end;
                has_pressure_master =
                    has_pressure_master || pressure_master;
                has_nonpressure_master =
                    has_nonpressure_master || !pressure_master;
                weight_sum += entry.weight;
                absolute_weight_sum += std::abs(entry.weight);
            }

            // A pressure Dirichlet line removes the freely variable constant
            // mode.  A cross-field relation cannot be interpreted as a unit
            // scalar pressure trace.  Homogeneous pressure MPCs preserve the
            // mode only when their weights form a partition of unity.
            if (pressure_slave) {
                if (retained_pressure_support != nullptr &&
                    !retained_pressure_support->contains(line.slave_dof) &&
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
                    !has_nonpressure_master &&
                    std::isfinite(weight_sum) &&
                    std::isfinite(absolute_weight_sum) &&
                    std::abs(weight_sum - 1.0) <= tolerance;
            } else if (has_pressure_master) {
                compatible = false;
            }
        });
    return compatible;
}



RetainedFieldSupportResult retainedFieldSupport(
    const systems::FESystem& system,
    const systems::OperatorDefinition& pair,
    FieldId pressure_field,
    GlobalIndex pressure_begin,
    GlobalIndex pressure_count,
    const CertificateCollectives& collectives)
{
    const auto& all_ranks = collectives.all_ranks;
    const auto& any_rank = collectives.any_rank;
    RetainedFieldSupportResult result;
    const auto binding =
        constantFieldCutVolumeBinding(pair, pressure_field);
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
        if (candidate && candidate->field == pressure_field &&
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
            cut->assertAllFreeSurfaceGeometrySnapshotsCurrent(
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
        report.revision.has_free_surface_snapshot_revision &&
        report.revision.has_source_value_revision &&
        cut->hasFreeSurfaceGeometrySnapshotForMarker(binding->marker);
    if (!all_ranks(local_cut_state_current)) {
        result.reason = "aggregation_report_stale";
        return result;
    }

    const auto snapshot_revision =
        cut->freeSurfaceGeometrySnapshotRevisionForMarker(binding->marker);
    std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot>
        selected_snapshot;
    std::size_t local_snapshot_matches = 0u;
    for (const auto& snapshot : cut->freeSurfaceGeometrySnapshots()) {
        if (snapshot &&
            snapshot->revision().snapshot_revision_key ==
                snapshot_revision) {
            ++local_snapshot_matches;
            selected_snapshot = snapshot;
        }
    }
    const bool local_snapshot_current =
        snapshot_revision ==
            report.revision.free_surface_snapshot_revision &&
        local_snapshot_matches == 1u && selected_snapshot != nullptr &&
        selected_snapshot->revision().complete() &&
        selected_snapshot->revision().interface_marker == binding->marker &&
        selected_snapshot->revision().source_value_revision ==
            report.revision.source_value_revision;
    if (!all_ranks(local_snapshot_current)) {
        result.reason = "aggregation_report_stale";
        return result;
    }

    const GlobalIndex pressure_end = pressure_begin + pressure_count;
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
            if (dof < pressure_begin || dof >= pressure_end) {
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
    backends::GenericVector& unit_pressure,
    backends::GenericVector& pressure_action,
    backends::GenericVector& kkt_residual,
    GlobalIndex pressure_begin,
    GlobalIndex pressure_count,
    GlobalIndex velocity_begin,
    GlobalIndex velocity_count,
    std::span<const GlobalIndex> constrained_dofs,
    std::span<const GlobalIndex> vector_owned_dofs,
    const std::function<bool(bool)>& all_ranks)
{
    ConstantFieldKktResult result;
    if (pressure_begin < 0 || pressure_count <= 0 ||
        velocity_begin < 0 || velocity_count <= 0) {
        result.reason = "invalid_velocity_or_pressure_range";
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

    unit_pressure.zero();
    std::vector<GlobalIndex> owned_active_pressure_dofs;
    owned_active_pressure_dofs.reserve(vector_owned_dofs.size());
    const GlobalIndex pressure_end = pressure_begin + pressure_count;
    for (const auto dof : vector_owned_dofs) {
        if (dof >= pressure_begin && dof < pressure_end &&
            !std::binary_search(
                constrained_dofs.begin(), constrained_dofs.end(), dof)) {
            owned_active_pressure_dofs.push_back(dof);
        }
    }
    std::vector<Real> unit_values(
        owned_active_pressure_dofs.size(), Real{1.0});
    auto unit_view = unit_pressure.createAssemblyView();
    FE_CHECK_NOT_NULL(
        unit_view.get(), "evaluateConstantFieldKkt: unit-field view");
    unit_view->beginAssemblyPhase();
    if (!owned_active_pressure_dofs.empty()) {
        unit_view->setVectorEntries(
            owned_active_pressure_dofs, unit_values);
    }
    unit_view->finalizeAssembly();
    unit_pressure.updateGhosts();
    if (!vector_is_finite(unit_pressure)) {
        result.reason = "unit_pressure_field_nonfinite";
        return result;
    }

    pair.mult(unit_pressure, pressure_action);
    pressure_action.markModified();
    if (!vector_is_finite(pressure_action)) {
        result.reason = "constant_pressure_action_nonfinite";
        return result;
    }
    result.direction_norm = pressure_action.norm();
    if (!finite_nonnegative(result.direction_norm) ||
        !(result.direction_norm > 0.0)) {
        result.reason = "constant_pressure_direction_zero_or_nonfinite";
        return result;
    }

    // K*c must lie exclusively in the velocity-test rows of the constrained
    // system. Check that invariant before interpreting its coefficient as a
    // pressure jump.
    unit_pressure.copyFrom(pressure_action);
    const GlobalIndex velocity_end = velocity_begin + velocity_count;
    std::vector<GlobalIndex> owned_velocity_dofs;
    owned_velocity_dofs.reserve(vector_owned_dofs.size());
    for (const auto dof : vector_owned_dofs) {
        if (dof >= velocity_begin && dof < velocity_end) {
            owned_velocity_dofs.push_back(dof);
        }
    }
    zeroVectorEntries(owned_velocity_dofs, unit_pressure);
    unit_pressure.updateGhosts();
    const double nonvelocity_action_norm = unit_pressure.norm();
    const double subspace_tolerance =
        100.0 * std::numeric_limits<Real>::epsilon() *
        std::max(1.0, result.direction_norm);
    if (!finite_nonnegative(nonvelocity_action_norm) ||
        !all_ranks(nonvelocity_action_norm <= subspace_tolerance)) {
        result.reason = "constant_pressure_action_outside_velocity_trace";
        return result;
    }

    const double load_norm = load.norm();
    const double direction_squared = pressure_action.dot(pressure_action);
    const double direction_load = pressure_action.dot(load);
    if (!finite_nonnegative(load_norm) ||
        !all_ranks(std::isfinite(direction_squared) &&
                   direction_squared > 0.0 &&
                   std::isfinite(direction_load))) {
        result.reason = "constant_pressure_projection_nonfinite";
        return result;
    }

    result.pressure_jump = -direction_load / direction_squared;
    result.volume_multiplier = -result.pressure_jump;
    if (!all_ranks(std::isfinite(result.pressure_jump) &&
                   std::isfinite(result.volume_multiplier))) {
        result.reason = "constant_pressure_multiplier_nonfinite";
        return result;
    }

    kkt_residual.copyFrom(load);
    axpy(kkt_residual,
         static_cast<Real>(result.pressure_jump),
         pressure_action);
    result.residual_norm = kkt_residual.norm();
    if (!vector_is_finite(kkt_residual) ||
        !finite_nonnegative(result.residual_norm)) {
        result.reason = "constant_pressure_residual_nonfinite";
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
        std::abs(pressure_action.dot(kkt_residual));
    const double orthogonality_denominator =
        result.direction_norm *
        std::max({load_norm, result.residual_norm, roundoff_floor});
    result.relative_orthogonality =
        orthogonality_numerator / orthogonality_denominator;
    if (!all_ranks(std::isfinite(result.relative_distance) &&
                   result.relative_distance >= 0.0 &&
                   std::isfinite(result.relative_orthogonality) &&
                   result.relative_orthogonality >= 0.0)) {
        result.reason = "constant_pressure_metrics_nonfinite";
        return result;
    }

    result.available = true;
    result.reason = "available";
    return result;
}

bool staticCertificateDiagnosticEnabled() noexcept
{
    // Preserve the same precedence as the Navier--Stokes registration gate so
    // an explicit primary setting of "0" cannot be overridden by an ambient
    // compatibility alias.  Do not cache this value: unit and integration
    // harnesses can scope environment variables within one process.
    constexpr std::array<const char*, 4> names{
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
        "SVMP_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
        "SVMP_NS_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC",
        "SVMP_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC",
    };
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            return environmentFlagEnabled(name);
        }
    }
    return false;
}

std::optional<double> staticCertificateRepresentabilityDistance() noexcept
{
    constexpr std::array<const char*, 2> names{
        "SVMP_NS_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE",
        "SVMP_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE",
    };
    for (const char* name : names) {
        const char* text = std::getenv(name);
        if (text == nullptr || text[0] == '\0') {
            continue;
        }
        char* end = nullptr;
        const double value = std::strtod(text, &end);
        if (end == text || end == nullptr || end[0] != '\0') {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return value;
    }
    return std::nullopt;
}

std::optional<double> staticCertificateConstantKktDistance() noexcept
{
    constexpr std::array<const char*, 2> names{
        "SVMP_NS_FREE_SURFACE_CONSTANT_PRESSURE_KKT_MAX_RELATIVE_DISTANCE",
        "SVMP_FREE_SURFACE_CONSTANT_PRESSURE_KKT_MAX_RELATIVE_DISTANCE",
    };
    for (const char* name : names) {
        const char* text = std::getenv(name);
        if (text == nullptr || text[0] == '\0') {
            continue;
        }
        char* end = nullptr;
        const double value = std::strtod(text, &end);
        if (end == text || end == nullptr || end[0] != '\0') {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return value;
    }
    return std::nullopt;
}

bool staticCertificateInitializerRequested() noexcept
{
    constexpr std::array<const char*, 2> names{
        "SVMP_NS_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER",
        "SVMP_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER",
    };
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            return environmentFlagEnabled(name);
        }
    }
    return false;
}

bool staticCertificateEveryAssembly() noexcept
{
    // Every requested full diagnostic sample is assembled.  Qualification
    // runs may additionally force the revision/exact-load-guarded mixed pair
    // and LSQR cache off, so those expensive operations repeat for every
    // sample (including both halves of a line-search accepted refresh).
    constexpr std::array<const char*, 2> names{
        "SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY",
        "SVMP_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY",
    };
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            return environmentFlagEnabled(name);
        }
    }
    return false;
}

} // namespace analysis
} // namespace FE
} // namespace svmp
