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
 * primary field and a scalar constraint field, on the
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
 * A formulation states the fields and the diagnostic operators of the
 * certificate as a ConstraintRangeDeclaration in the system's diagnostic
 * registry (DiagnosticRegistry::declare).
 *
 * ConstraintRangeCertificate is the certificate in a Newton solve, for that
 * declaration and one ConstraintRangeMethod: an observer that samples the
 * declared loads and the range distance at every residual assembly, an
 * acceptance check that vetoes a converged candidate whose distances exceed
 * the method's thresholds, and an initializer that adds the least-squares
 * constraint-field correction of the remaining balance to the current guess
 * once per workspace lifecycle.
 * Its log lines, written by its log adapter (ConstraintRangeCertificateLog.cpp,
 * declared in the detail namespace below), keep the vocabulary of the
 * qualification records consumed by solver clients.
 *
 * buildNewtonAnalysisExtensions installs the analysis extensions of a Newton
 * solve in one step, from the declarations of the system's formulations and
 * a resolved NewtonAnalysisMethod: the residual observers and the
 * certificate.
 */

#include "Analysis/ConstraintRangeRecordVocabulary.h"
#include "Core/Types.h"
#include "Systems/DiagnosticRegistry.h"
#include "TimeStepping/NewtonSolver.h"

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

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

/**
 * What a formulation declares for the constraint-range certificate of its
 * saddle-point pair: the primary field (the test field of every sampled
 * load), the scalar constraint field, and the diagnostic operators, which it
 * may install only on request.  The six sampled operators are the constraint
 * work, the two parts of the potential work, their sum, the load that is
 * certified to lie in the range of G, and the balance (constraint work plus
 * potential work); `pair` is the symmetric mixed operator [0, G; G^T, 0].
 */
struct ConstraintRangeDeclaration {
    FieldId primary_field{INVALID_FIELD_ID};
    FieldId constraint_field{INVALID_FIELD_ID};
    systems::DiagnosticOperatorHandle constraint_work{};
    std::array<systems::DiagnosticOperatorHandle, 2> potential_work_parts{};
    systems::DiagnosticOperatorHandle potential_work{};
    systems::DiagnosticOperatorHandle load{};
    systems::DiagnosticOperatorHandle balance{};
    systems::DiagnosticOperatorHandle pair{};

    /// The six sampled operators, in the order of the certificate record.
    [[nodiscard]] std::array<const systems::DiagnosticOperatorHandle*, 6> sampledOperators() const noexcept
    {
        return {&constraint_work, &potential_work_parts[0], &potential_work_parts[1],
                &potential_work, &load, &balance};
    }

    friend bool operator==(const ConstraintRangeDeclaration&, const ConstraintRangeDeclaration&) = default;
};

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
    backends::GenericVector& coefficients,
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
    double jump{std::numeric_limits<double>::quiet_NaN()};
    double multiplier{std::numeric_limits<double>::quiet_NaN()};
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
    GlobalIndex constraint_begin,
    GlobalIndex constraint_count,
    const std::set<GlobalIndex>* retained_support = nullptr);

/// The field DOFs that the aggregation of the field's cut side retains.
[[nodiscard]] RetainedFieldSupportResult retainedFieldSupport(
    const systems::FESystem& system,
    const systems::OperatorDefinition& pair,
    FieldId constraint_field,
    GlobalIndex constraint_begin,
    GlobalIndex constraint_count,
    const CertificateCollectives& collectives);

/**
 * Evaluate the constant-field KKT subproblem without changing the load.
 *
 * The symmetric pair is K=[0,G;G^T,0].  A reduced coefficient vector c that
 * represents the unit constraint field produces g=G*c.  The returned jump
 * minimizes ||f+p*g|| in the same constrained coefficient norm as the full
 * range diagnostic.  The functional multiplier is -p, matching
 * a Lagrangian whose constraint derivative is the negative of G.
 * @p vector_owned_dofs are the DOFs the vectors' layout owns on this rank.
 */
[[nodiscard]] ConstantFieldKktResult evaluateConstantFieldKkt(
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
    const std::function<bool(bool)>& all_ranks);

/**
 * The resolved method of the certificate.  A threshold makes the
 * corresponding gate mandatory at acceptance; without any, the certificate
 * only observes.  The initializer requires the range threshold.
 */
struct ConstraintRangeMethod {
    /// Sample at every residual assembly even without a threshold.
    bool observe{false};
    /// Maximum relative distance of the load from range(G).
    std::optional<double> max_range_distance{};
    /// Maximum relative KKT residual after fitting only a constant field.
    std::optional<double> max_constant_field_kkt_distance{};
    /// Add the least-squares constraint-field correction of the remaining
    /// balance to the current guess, once per workspace lifecycle.
    bool initialize_constraint_field{false};
    /// Read before each sample. An explicit method without a reader uses
    /// the base catalogue's cache policy.
    bool (*reassemble_each_sample)() noexcept{nullptr};
    /// Static record labels supplied by the owner of the observed balance.
    const ConstraintRangeRecordVocabulary* record_vocabulary{nullptr};
    [[nodiscard]] const ConstraintRangeRecordVocabulary& vocabulary() const noexcept;
};

/// Whether @p method samples at all.
[[nodiscard]] bool constraintRangeMethodRequested(const ConstraintRangeMethod& method) noexcept;

/// The method the environment gates below state.
[[nodiscard]] ConstraintRangeMethod constraintRangeMethodFromEnvironment();

/// The pair matrix and its vectors in a Newton workspace.
class ConstraintRangeStorage final : public timestepping::NewtonExtensionStorage {
public:
    std::unique_ptr<backends::GenericMatrix> pair{};
    ConstraintRangeVectors vectors{};
};

/// The certificate's storage in @p workspace, or null.
[[nodiscard]] ConstraintRangeStorage* findConstraintRangeStorage(timestepping::NewtonWorkspace& workspace);

/**
 * What the certificate records in a Newton report: its last sample (the
 * range distance of the load and the constant-field KKT), the gates applied
 * at acceptance, and the initializer.  Reached through the certificate
 * (ConstraintRangeCertificate::record).
 */
struct ConstraintRangeReport {
    /// Whether a sample was taken in the solve.
    bool sampled{false};

    struct Range {
        bool available{false};
        bool converged{false};
        bool breakdown{false};
        FieldId constraint_field{INVALID_FIELD_ID};
        double residual_norm{std::numeric_limits<double>::quiet_NaN()};
        double relative_distance{std::numeric_limits<double>::quiet_NaN()};
        std::string reason{"not_sampled"};
    };
    /// The distance of the load from range(G).
    Range range{};

    struct ConstantFieldKkt {
        bool available{false};
        bool unit_coefficients_represent_constant{false};
        bool constraints_preserve_constants{false};
        std::string constraint_scope{"full_field"};
        std::string constraint_scope_reason{"not_attempted"};
        double jump{std::numeric_limits<double>::quiet_NaN()};
        double multiplier{std::numeric_limits<double>::quiet_NaN()};
        double direction_norm{std::numeric_limits<double>::quiet_NaN()};
        double residual_norm{std::numeric_limits<double>::quiet_NaN()};
        double relative_distance{std::numeric_limits<double>::quiet_NaN()};
        double relative_orthogonality{std::numeric_limits<double>::quiet_NaN()};
        std::string reason{"not_sampled"};
    };
    /// The best constant constraint field and the residual it leaves; it
    /// never replaces or projects the load.
    ConstantFieldKkt constant_field_kkt{};

    struct Gate {
        bool applied{false};
        bool passed{false};
        double max_relative_distance{std::numeric_limits<double>::quiet_NaN()};
    };
    Gate range_gate{};
    Gate constant_field_kkt_gate{};

    struct Initializer {
        bool requested{false};
        bool applied{false};
        bool passed{false};
        std::string reason{"not_requested"};
    };
    Initializer initializer{};
};

/**
 * What a consumer of one probe of the certificate reads (the static
 * startup): the range certificate, the production residual of the probed
 * state, and the constant-field KKT, each checked to be finite and the same
 * on every rank.  `defect` names the first check that failed.
 */
struct ConstraintRangeProbeEvidence {
    enum class Defect : std::uint8_t {
        None,
        ProductionResidualNonfinite,
        ProductionResidualDiffersAcrossRanks,
        ConstantFieldKktNonfinite,
        ConstraintFieldInvalid,
        RangeNonfinite,
        AvailabilityDiffersAcrossRanks,
        RangeDiffersAcrossRanks,
        ConstantFieldKktDiffersAcrossRanks
    };
    Defect defect{Defect::None};

    bool range_available{false};
    bool range_converged{false};
    bool range_breakdown{false};
    double range_residual_norm{std::numeric_limits<double>::quiet_NaN()};
    double range_relative_distance{std::numeric_limits<double>::quiet_NaN()};
    /// Why the range certificate is unavailable.
    std::string range_reason{};
    FieldId constraint_field{INVALID_FIELD_ID};

    double production_residual_norm{std::numeric_limits<double>::quiet_NaN()};

    /// The constant-field KKT is required when the constant field is a
    /// physical mode of the problem, the space represents the unit field
    /// and the constraints preserve it.
    bool constant_field_kkt_required{false};
    bool constant_field_kkt_available{false};
    double constant_field_kkt_residual_norm{std::numeric_limits<double>::quiet_NaN()};
    double constant_field_kkt_relative_distance{std::numeric_limits<double>::quiet_NaN()};
    double constant_field_jump{std::numeric_limits<double>::quiet_NaN()};
    double constant_field_multiplier{std::numeric_limits<double>::quiet_NaN()};
};

class ConstraintRangeCertificate final : public timestepping::NewtonObserver,
                                         public timestepping::NewtonAcceptanceCheck,
                                         public timestepping::NewtonInitializer {
public:
    /// The certificate of @p declaration (when empty, a system that declares
    /// none: it samples nothing and says so) for @p method.  Throws for a
    /// negative or nonfinite threshold and for an initializer without a
    /// range threshold (deferred to the collective policy check when MPI
    /// runs several tasks).
    ConstraintRangeCertificate(std::optional<ConstraintRangeDeclaration> declaration,
                               ConstraintRangeMethod method);
    ~ConstraintRangeCertificate() override;
    ConstraintRangeCertificate(const ConstraintRangeCertificate&) = delete;
    ConstraintRangeCertificate& operator=(const ConstraintRangeCertificate&) = delete;

    [[nodiscard]] const ConstraintRangeMethod& method() const noexcept { return method_; }

    /// What this certificate recorded in @p report (a default record when
    /// it recorded nothing).
    [[nodiscard]] const ConstraintRangeReport& record(const timestepping::NewtonReport& report) const;

    /// The evidence of @p report, a probe of this certificate on @p system.
    /// @p constant_field_is_physical states whether a constant constraint
    /// field is a physical mode of the problem; @p distinct_field is a field
    /// the constraint field must differ from (or INVALID_FIELD_ID).
    /// Collective on the active communicator of @p system.
    [[nodiscard]] ConstraintRangeProbeEvidence probeEvidence(
        const timestepping::NewtonReport& report,
        const systems::FESystem& system,
        bool constant_field_is_physical,
        FieldId distinct_field) const;

    [[nodiscard]] std::string_view id() const override { return "constraint_range"; }
    [[nodiscard]] std::unique_ptr<timestepping::NewtonExtensionStorage> allocate(
        const timestepping::NewtonStorageContext& context) const override;
    void refreshStorage(const timestepping::NewtonStorageContext& context,
                        timestepping::NewtonExtensionStorage* storage,
                        bool after_production) const override;
    void beginSolve() override;
    void validateCommunicatorPolicy(const systems::FESystem& system) const override;

    void observe(timestepping::NewtonObservation& observation) override;

    [[nodiscard]] std::vector<timestepping::CertificateDecision> assess(
        timestepping::NewtonAcceptance& acceptance) override;
    [[nodiscard]] bool requiresAcceptedObservation() const override;

    [[nodiscard]] bool enabled() const override { return method_.initialize_constraint_field; }
    [[nodiscard]] bool propose(timestepping::NewtonInitialization& initialization) override;
    void apply(const systems::FESystem& system, backends::GenericVector& current) override;
    void reportApplied(const systems::FESystem& system, timestepping::NewtonReport& report) override;
    void reportConsumed(const systems::FESystem& system, timestepping::NewtonReport& report) override;

private:
    struct State;

    /// The record of this certificate in @p report, created on first use.
    ConstraintRangeReport& mutableRecord(timestepping::NewtonReport& report) const;

    void logInitializer(const systems::FESystem& system, timestepping::NewtonReport& report,
                        bool applied, bool passed, std::string_view reason) const;

    std::optional<ConstraintRangeDeclaration> declaration_;
    ConstraintRangeMethod method_;
    std::unique_ptr<State> state_;
};

/**
 * The resolved method of the analysis extensions of a Newton solve: the
 * residual observers it switches on and the method of the constraint-range
 * certificate, which is installed when it requests a sample.
 */
struct NewtonAnalysisMethod {
    /// FieldResidualObserver.
    bool observe_field_residual{false};
    /// RowContributionObserver.
    bool observe_row_contributions{false};
    ConstraintRangeMethod constraint_range{};
};

/// The method the environment states: the switches of the two residual
/// observers, then the six certificate gates (constraintRangeMethodFromEnvironment).
[[nodiscard]] NewtonAnalysisMethod newtonAnalysisMethodFromEnvironment();

/// The analysis extensions of a Newton solve, and the certificate among
/// them, through which its results are read (null when none is installed).
struct NewtonAnalysisExtensions {
    timestepping::NewtonExtensions extensions{};
    std::shared_ptr<ConstraintRangeCertificate> constraint_range{};
};

/**
 * The extensions @p method requests on @p system, each with the declaration
 * of the system's formulations it reads (none when the system declares
 * none), in installation order: the field-residual observer, the
 * row-contribution observer, then the certificate as observer, acceptance
 * check and initializer.
 */
[[nodiscard]] NewtonAnalysisExtensions buildNewtonAnalysisExtensions(
    const systems::FESystem& system,
    const NewtonAnalysisMethod& method);

/**
 * The options of a probe of the certificate of @p analysis at a supplied
 * state: one Newton iteration of @p op, the entry residual alone (or, when
 * the certificate initializes, its correction and nothing more), no line
 * search, no outer fixed point, and only the extensions of @p analysis.
 */
[[nodiscard]] timestepping::NewtonOptions constraintRangeProbeOptions(
    const systems::OperatorTag& op,
    const NewtonAnalysisExtensions& analysis);

/// Controls of the numerical certificate. A formulation may supply ordered
/// aliases for these settings when it configures Newton analysis.
namespace constraint_range_environment {
inline constexpr char observe[] = "SVMP_FE_CONSTRAINT_RANGE_DIAGNOSTIC";
inline constexpr char reassemble[] = "SVMP_FE_CONSTRAINT_RANGE_REASSEMBLE_EACH_SAMPLE";
inline constexpr char range_distance[] = "SVMP_FE_CONSTRAINT_RANGE_MAX_RELATIVE_DISTANCE";
inline constexpr char constant_kkt_distance[] = "SVMP_FE_CONSTRAINT_RANGE_CONSTANT_FIELD_KKT_MAX_RELATIVE_DISTANCE";
inline constexpr char initialize[] = "SVMP_FE_CONSTRAINT_RANGE_INITIALIZE_CONSTRAINT_FIELD";
} // namespace constraint_range_environment

namespace detail {

/// What the mixed-pair/LSQR cache decided for one sample, for its log line.
struct ConstraintRangeCacheTrace {
    bool revision_key_available{false};
    bool every_assembly{false};
    bool pair_matrix_state_independent{false};
    bool prior_sample{false};
    bool geometry_match{false};
    bool candidate{false};
    bool load_equality_checked{false};
    bool load_equal{false};
    bool hit{false};
    double load_difference_norm{std::numeric_limits<double>::quiet_NaN()};
    std::string rejection_reason{};
};

// The log adapter of the certificate (ConstraintRangeCertificateLog.cpp):
// the text of its log lines, written by the rank the certificate chooses.

/// A sample not taken: the communicator disagrees on whether to sample.
void logConstraintRangeEnablementDiffers(const ConstraintRangeRecordVocabulary& labels);
/// A sample not taken: the system declares no certificate.
void logConstraintRangeNotDeclared(const ConstraintRangeRecordVocabulary& labels,
                                   const timestepping::NewtonObservation& observation);
/// A sample not taken: the declared operator @p op is not installed on every rank.
void logConstraintRangeOperatorNotInstalled(const ConstraintRangeRecordVocabulary& labels,
                                            const timestepping::NewtonObservation& observation,
                                            std::string_view op);
/// A sample: the norms of the six sampled loads, the cache, and the range
/// and constant-field KKT parts of @p record (@p range, the LSQR solve, when
/// the range is available).
void logConstraintRangeSample(const ConstraintRangeRecordVocabulary& labels,
                              const timestepping::NewtonObservation& observation,
                              int rank,
                              const std::array<double, 6>& norms,
                              const ConstraintRangeCacheTrace& cache,
                              const ConstraintRangeReport& record,
                              const ConstraintRangeLsqrResult& range,
                              std::uint64_t active_constraint_dofs,
                              int iteration_cap);
/// The range-distance gate at an acceptance event.
void logConstraintRangeDistanceGate(const ConstraintRangeRecordVocabulary& labels,
                                    const timestepping::NewtonAcceptance& acceptance,
                                    const ConstraintRangeReport& candidate,
                                    double threshold,
                                    bool passed,
                                    std::string_view reason);
/// The constant-field KKT distance gate at an acceptance event.
void logConstantFieldKktDistanceGate(const ConstraintRangeRecordVocabulary& labels,
                                     const timestepping::NewtonAcceptance& acceptance,
                                     const ConstraintRangeReport& candidate,
                                     double threshold,
                                     bool passed,
                                     std::string_view reason);
/// The initializer: applied, refused, or consumed earlier.
void logConstraintRangeInitializer(const ConstraintRangeRecordVocabulary& labels,
                                   const ConstraintRangeReport& record,
                                   double threshold,
                                   const ConstraintRangeLsqrResult& correction,
                                   double correction_load_norm,
                                   bool applied,
                                   bool passed,
                                   std::string_view reason);

} // namespace detail

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_CONSTRAINT_RANGE_CERTIFICATE_H
