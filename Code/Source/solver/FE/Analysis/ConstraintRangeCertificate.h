/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_CONSTRAINT_RANGE_CERTIFICATE_H
#define SVMP_FE_ANALYSIS_CONSTRAINT_RANGE_CERTIFICATE_H

/**
 * @file ConstraintRangeCertificate.h
 * @brief Whether a load lies in the range of a constraint field's gradient
 *
 * Linear algebra of the static certificates of a saddle-point system with a
 * primary field (velocity) and a scalar constraint field (pressure), on the
 * constrained mixed pair K = [0, G; G^T, 0] assembled as a GenericMatrix:
 *
 * - solveConstraintRangeLsqr: min_c ||G c + f|| by Golub--Kahan LSQR, the
 *   distance of a load f from the range of G;
 * - evaluateConstantFieldKkt: the best constant of the constraint field,
 *   min_p ||f + p G 1||, and its residual;
 * - the checks that the unit field is representable and preserved by the
 *   constraints, and its retained support on an aggregated cut side.
 *
 * Everything is matrix-free through GenericMatrix/GenericVector: no normal
 * matrix, no backend storage, no global gather.  The collective decisions go
 * through the caller's reductions, so every rank takes the same branch.
 *
 * The environment gates of the static free-surface certificates (decision
 * D61 item 7) are read here until the diagnostics registry replaces them.
 */

#include "Core/Types.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>

namespace svmp {
namespace FE {
namespace backends {
class BackendFactory;
class GenericMatrix;
class GenericVector;
} // namespace backends
namespace constraints {
class AffineConstraints;
} // namespace constraints
namespace spaces {
class FunctionSpace;
} // namespace spaces
namespace systems {
class FESystem;
struct OperatorDefinition;
} // namespace systems

namespace analysis {

/// Reductions over the communicator of the system, called on every rank.
struct CertificateCollectives {
    std::function<bool(bool)> all_ranks{};
    std::function<bool(bool)> any_rank{};
    std::function<bool(std::uint64_t)> agrees_on_unsigned{};
    int rank{0};
    int size{1};
};

struct ConstraintRangeLsqrResult {
    double residual_norm{std::numeric_limits<double>::quiet_NaN()};
    double relative_residual{std::numeric_limits<double>::quiet_NaN()};
    double normal_residual_norm{std::numeric_limits<double>::quiet_NaN()};
    double relative_normal_residual{
        std::numeric_limits<double>::quiet_NaN()};
    double solution_norm{std::numeric_limits<double>::quiet_NaN()};
    int iterations{0};
    int normal_residual_refinements{0};
    bool converged{false};
    bool breakdown{false};
};

/**
 * Solve min_c ||G c + f|| with Golub--Kahan LSQR from c=0.
 *
 * `pair` is the constrained, symmetric mixed operator [0,G;G^T,0].  Inputs
 * carrying constraint-field coefficients therefore produce G-actions, while
 * inputs carrying primary-field residuals produce G^T-actions through the
 * same generic matrix multiply.  No normal matrix is formed and no backend
 * storage is inspected or globally gathered.
 */
[[nodiscard]] ConstraintRangeLsqrResult solveConstraintRangeLsqr(
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
    const std::function<bool(bool)>& all_ranks);

/// The vectors of the certificate solves, in the layout of the mixed pair.
struct ConstraintRangeVectors {
    std::unique_ptr<backends::GenericVector> load{};
    /// The balance left after an existing constraint-field baseline.
    std::unique_ptr<backends::GenericVector> correction_load{};
    std::unique_ptr<backends::GenericVector> solution{};
    std::unique_ptr<backends::GenericVector> left_basis{};
    std::unique_ptr<backends::GenericVector> right_basis{};
    std::unique_ptr<backends::GenericVector> direction{};
    std::unique_ptr<backends::GenericVector> work{};
    std::unique_ptr<backends::GenericVector> residual{};
    std::unique_ptr<backends::GenericVector> normal_residual{};

    /// Replaces every vector with a new one of @p n_dofs entries, building
    /// the whole set before releasing the old one.
    void recreate(const backends::BackendFactory& factory, GlobalIndex n_dofs);
};

struct ConstantFieldKktResult {
    bool available{false};
    double pressure_jump{std::numeric_limits<double>::quiet_NaN()};
    double volume_multiplier{std::numeric_limits<double>::quiet_NaN()};
    double direction_norm{std::numeric_limits<double>::quiet_NaN()};
    double residual_norm{std::numeric_limits<double>::quiet_NaN()};
    double relative_distance{std::numeric_limits<double>::quiet_NaN()};
    double relative_orthogonality{
        std::numeric_limits<double>::quiet_NaN()};
    std::string reason{"not_evaluated"};
};

struct RetainedFieldSupportResult {
    std::optional<std::set<GlobalIndex>> support{};
    std::string reason{"pair_support_unavailable"};
};

/// True when the unit coefficient vector of @p space is the constant 1.
[[nodiscard]] bool unitCoefficientsRepresentConstant(const spaces::FunctionSpace& space);

/// True when the constraints on the field DOFs [begin, begin + count)
/// preserve its unit field: homogeneous partition-of-unity relations among
/// those DOFs only.  Zero lines outside @p retained_support are ignored.
[[nodiscard]] bool constraintsPreserveUnitField(
    const constraints::AffineConstraints& affine_constraints,
    GlobalIndex pressure_begin,
    GlobalIndex pressure_count,
    const std::set<GlobalIndex>* retained_pressure_support = nullptr);

/// The field DOFs that the aggregation of the field's cut side retains.
[[nodiscard]] RetainedFieldSupportResult retainedFieldSupport(
    const systems::FESystem& system,
    const systems::OperatorDefinition& pair,
    FieldId pressure_field,
    GlobalIndex pressure_begin,
    GlobalIndex pressure_count,
    const CertificateCollectives& collectives);

/**
 * Evaluate the constant-field KKT subproblem without changing the load.
 *
 * The symmetric pair is K=[0,G;G^T,0].  A reduced coefficient vector c that
 * represents the unit constraint field produces g=G*c.  The returned jump
 * minimizes ||f+p*g|| in the same constrained coefficient norm as the full
 * range diagnostic.  The functional multiplier is -p, matching
 * E_h + lambda*V_h to the momentum convention -p*D V_h.
 * @p vector_owned_dofs are the DOFs the vectors' layout owns on this rank.
 */
[[nodiscard]] ConstantFieldKktResult evaluateConstantFieldKkt(
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
    const std::function<bool(bool)>& all_ranks);

// The environment gates of the static free-surface certificates, each a
// Navier--Stokes name and its short alias; the first one set decides.
/// SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC or
/// ..._EQUILIBRIUM_DIAGNOSTIC: sample the diagnostic (not cached).
[[nodiscard]] bool staticCertificateDiagnosticEnabled() noexcept;
/// ..._PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE (NaN when unparsable).
[[nodiscard]] std::optional<double> staticCertificateRepresentabilityDistance() noexcept;
/// ..._CONSTANT_PRESSURE_KKT_MAX_RELATIVE_DISTANCE (NaN when unparsable).
[[nodiscard]] std::optional<double> staticCertificateConstantKktDistance() noexcept;
/// ..._STATIC_COMPATIBLE_PRESSURE_INITIALIZER.
[[nodiscard]] bool staticCertificateInitializerRequested() noexcept;
/// ..._CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY.
[[nodiscard]] bool staticCertificateEveryAssembly() noexcept;

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_CONSTRAINT_RANGE_CERTIFICATE_H
