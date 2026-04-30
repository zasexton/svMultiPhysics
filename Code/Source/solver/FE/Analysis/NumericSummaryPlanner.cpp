/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/NumericSummaryPlanner.h"

#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/ContributionDescriptor.h"

#include <algorithm>
#include <sstream>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

template <typename T>
void appendUnique(std::vector<T>& values, const T& value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

void appendVariables(std::vector<VariableKey>& target,
                     const std::vector<VariableKey>& values) {
    for (const auto& value : values) {
        appendUnique(target, value);
    }
}

std::string sourceAnalyzer(const PropertyClaim& claim) {
    if (!claim.claim_origin.empty()) {
        return claim.claim_origin;
    }
    for (const auto& evidence : claim.evidence) {
        if (!evidence.source.empty()) {
            return evidence.source;
        }
    }
    return "UnknownAnalyzer";
}

bool hasSourceAnalyzer(const PropertyClaim& claim, const char* analyzer) {
    if (claim.claim_origin == analyzer) {
        return true;
    }
    return std::any_of(claim.evidence.begin(), claim.evidence.end(),
                       [analyzer](const PropertyEvidence& evidence) {
                           return evidence.source == analyzer;
                       });
}

AnalysisConfidence strongerConfidence(AnalysisConfidence a, AnalysisConfidence b) {
    auto rank = [](AnalysisConfidence c) {
        switch (c) {
            case AnalysisConfidence::High: return 3;
            case AnalysisConfidence::Medium: return 2;
            case AnalysisConfidence::Low: return 1;
        }
        return 0;
    };
    return rank(b) > rank(a) ? b : a;
}

const FieldDescriptor* fieldDescriptorForClaim(const ProblemAnalysisContext& context,
                                               const PropertyClaim& claim) {
    if (claim.field != INVALID_FIELD_ID) {
        if (const auto* fd = context.fieldDescriptor(claim.field)) {
            return fd;
        }
    }

    for (const auto& variable : claim.variables) {
        if (variable.kind != VariableKind::FieldComponent ||
            variable.field_id == INVALID_FIELD_ID) {
            continue;
        }
        if (const auto* fd = context.fieldDescriptor(variable.field_id)) {
            return fd;
        }
    }

    return nullptr;
}

bool isScalarFieldClaim(const ProblemAnalysisContext& context,
                        const PropertyClaim& claim) {
    const auto* fd = fieldDescriptorForClaim(context, claim);
    if (!fd) {
        return false;
    }
    return fd->value_dimension == 1 && fd->field_type == FieldType::Scalar;
}

std::vector<VariableKey> variablesForFields(const std::vector<FieldId>& fields) {
    std::vector<VariableKey> variables;
    variables.reserve(fields.size());
    for (FieldId field : fields) {
        if (field != INVALID_FIELD_ID) {
            appendUnique(variables, VariableKey::field(field));
        }
    }
    return variables;
}

bool containsVariable(const std::vector<VariableKey>& values,
                      const VariableKey& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool sameVariableSet(const std::vector<VariableKey>& a,
                     const std::vector<VariableKey>& b) {
    return a.size() == b.size() &&
           std::all_of(a.begin(), a.end(), [&](const VariableKey& value) {
               return containsVariable(b, value);
           });
}

void appendVariableId(std::ostringstream& os, const VariableKey& variable) {
    os << static_cast<int>(variable.kind) << ':'
       << variable.field_id << ':'
       << variable.component << ':'
       << variable.name;
}

std::string variablesScopeId(const std::vector<VariableKey>& variables) {
    std::ostringstream os;
    for (const auto& variable : variables) {
        os << '[';
        appendVariableId(os, variable);
        os << ']';
    }
    return os.str();
}

std::string optionalString(const std::optional<std::string>& value) {
    return value ? *value : std::string{};
}

std::string claimScopeId(const PropertyClaim& claim) {
    if (claim.estimate_scope) return *claim.estimate_scope;
    if (claim.tested_block_id) return *claim.tested_block_id;
    if (claim.coefficient_id) return *claim.coefficient_id;
    if (claim.invariant_set_id) return *claim.invariant_set_id;
    if (claim.equilibrium_id) return *claim.equilibrium_id;
    return {};
}

AnalysisSummaryRequest& ensureRequest(AnalysisRequestPlan& plan,
                                      AnalysisSummaryKind kind,
                                      DomainKind domain,
                                      const std::vector<VariableKey>& variables,
                                      const std::string& block_id,
                                      const std::string& contribution_id,
                                      const std::string& scope_id) {
    auto it = std::find_if(plan.summary_requests.begin(),
                           plan.summary_requests.end(),
                           [&](const AnalysisSummaryRequest& request) {
                               return request.summary_kind == kind &&
                                      request.domain == domain &&
                                      sameVariableSet(request.variables,
                                                      variables) &&
                                      request.block_id == block_id &&
                                      request.contribution_id == contribution_id &&
                                      request.scope_id == scope_id;
                           });
    if (it != plan.summary_requests.end()) {
        return *it;
    }

    AnalysisSummaryRequest request;
    request.summary_kind = kind;
    request.domain = domain;
    request.variables = variables;
    request.block_id = block_id;
    request.contribution_id = contribution_id;
    request.scope_id = scope_id;
    request.request_id = std::string(toString(kind)) + ":" + toString(domain);
    if (!block_id.empty()) request.request_id += ":block=" + block_id;
    if (!contribution_id.empty()) {
        request.request_id += ":contribution=" + contribution_id;
    }
    if (!scope_id.empty()) request.request_id += ":scope=" + scope_id;
    if (!variables.empty()) {
        request.request_id += ":vars=" + variablesScopeId(variables);
    }
    plan.summary_requests.push_back(std::move(request));
    return plan.summary_requests.back();
}

void addRequest(AnalysisRequestPlan& plan,
                const ProblemAnalysisContext& context,
                AnalysisSummaryKind kind,
                DomainKind domain,
                const std::vector<VariableKey>& variables,
                std::size_t source_claim_index,
                const PropertyClaim& claim,
                const std::string& reason) {
    auto& request = ensureRequest(plan,
                                  kind,
                                  domain,
                                  variables,
                                  optionalString(claim.tested_block_id),
                                  std::string{},
                                  claimScopeId(claim));
    request.already_available = request.already_available || context.hasSummaryKind(kind);
    request.confidence = strongerConfidence(request.confidence, claim.confidence);
    appendVariables(request.variables, variables);
    appendUnique(request.source_claim_indices, source_claim_index);
    appendUnique(request.source_claim_kinds, claim.kind);
    appendUnique(request.source_analyzers, sourceAnalyzer(claim));
    appendUnique(request.reasons, reason);
}

void addContextRequest(AnalysisRequestPlan& plan,
                       const ProblemAnalysisContext& context,
                       AnalysisSummaryKind kind,
                       DomainKind domain,
                       const std::vector<VariableKey>& variables,
                       const std::string& reason,
                       AnalysisConfidence confidence = AnalysisConfidence::Medium,
                       const std::string& block_id = {},
                       const std::string& contribution_id = {},
                       const std::string& scope_id = {}) {
    auto& request = ensureRequest(plan,
                                  kind,
                                  domain,
                                  variables,
                                  block_id,
                                  contribution_id,
                                  scope_id);
    request.already_available = request.already_available || context.hasSummaryKind(kind);
    request.confidence = strongerConfidence(request.confidence, confidence);
    appendVariables(request.variables, variables);
    appendUnique(request.source_analyzers, std::string("ProblemAnalysisContext"));
    appendUnique(request.reasons, reason);
}

bool claimIsUsableSymbolicEvidence(const PropertyClaim& claim) noexcept {
    return claim.status == PropertyStatus::Exact ||
           claim.status == PropertyStatus::Likely ||
           claim.status == PropertyStatus::Preserved ||
           claim.status == PropertyStatus::Violated ||
           claim.status == PropertyStatus::Unknown;
}

bool isWeakPenaltyLike(EnforcementKind kind) noexcept {
    return kind == EnforcementKind::WeakPenalty ||
           kind == EnforcementKind::WeakNitsche ||
           kind == EnforcementKind::WeakInequality;
}

} // namespace

std::string NumericSummaryPlanner::name() const {
    return "NumericSummaryPlanner";
}

void NumericSummaryPlanner::run(const ProblemAnalysisContext& context,
                                ProblemAnalysisReport& report) const {
    for (std::size_t i = 0; i < report.claims.size(); ++i) {
        const auto& claim = report.claims[i];
        if (!claimIsUsableSymbolicEvidence(claim)) {
            continue;
        }

        const auto source = sourceAnalyzer(claim);

        if (claim.kind == PropertyKind::OperatorSymmetry ||
            claim.kind == PropertyKind::OperatorDefiniteness ||
            hasSourceAnalyzer(claim, "OperatorClassAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::DiscreteMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " classified an elliptic/operator block; request sparse matrix symmetry, sign, row-sum, conditioning, finite tolerance, and theorem-specific M-matrix/SPD diagnostics");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " classified an operator block; request reduced free-free evidence after constraints");
            addRequest(report.request_plan, context, AnalysisSummaryKind::CoefficientProperties,
                       claim.domain, claim.variables, i, claim,
                       source + " classified an elliptic/coercive structure; request coefficient positivity and anisotropy metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::MeshGeometryQuality,
                       claim.domain, claim.variables, i, claim,
                       source + " classified an elliptic operator; request native mesh geometry quality evidence");

            if (isScalarFieldClaim(context, claim)) {
                addRequest(report.request_plan, context, AnalysisSummaryKind::LocalStencil,
                           claim.domain, claim.variables, i, claim,
                           source + " classified scalar diffusion; request local stencil sign checks for DMP/Z-matrix/M-matrix monotonicity");
            }
        }

        if (claim.kind == PropertyKind::Nullspace &&
            (hasSourceAnalyzer(claim, "KernelAnalyzer") ||
             hasSourceAnalyzer(claim, "MixedOperatorAnalyzer"))) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " detected a nullspace; request reduced operator and nullspace-handling evidence");
        }

        if (claim.kind == PropertyKind::MixedSaddlePoint ||
            hasSourceAnalyzer(claim, "MixedOperatorAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupEstimate,
                       claim.domain, claim.variables, i, claim,
                       source + " detected saddle-point structure; request numerical inf-sup estimate");
            addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupPairCertification,
                       claim.domain, claim.variables, i, claim,
                       source + " detected saddle-point structure; request known stable-pair, Fortin, mesh, and domain assumption evidence");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " detected saddle-point structure; request constrained/reduced block classification");
            addRequest(report.request_plan, context, AnalysisSummaryKind::SchurComplement,
                       claim.domain, claim.variables, i, claim,
                       source + " detected saddle-point structure; request Schur complement conditioning, equivalence, and preconditioner evidence");
            addRequest(report.request_plan, context, AnalysisSummaryKind::DiscreteMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " detected saddle-point structure; request block matrix diagnostics and Schur-ready evidence");
        }

        if (claim.kind == PropertyKind::InfSupCondition ||
            hasSourceAnalyzer(claim, "InfSupAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupEstimate,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted an inf-sup claim; request estimate value, scope, and nullspace handling");
            addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupPairCertification,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted an inf-sup claim; request stable-pair/Fortin certification metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted an inf-sup claim; request reduced saddle-point block evidence");
        }

        if (claim.kind == PropertyKind::OperatorTransportCharacter ||
            hasSourceAnalyzer(claim, "TransportCharacterAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::TemporalStability,
                       claim.domain, claim.variables, i, claim,
                       source + " detected first-order transport; request finite CFL/eigenvalue-scale, stability-region, norm, operator-scope, nonnormal/pseudospectral, accepted-growth, and time-horizon metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ParameterScale,
                       claim.domain, claim.variables, i, claim,
                       source + " detected first-order transport; request finite Peclet-like and local transport-diffusion scale metadata with theorem/scope identifiers");
            addRequest(report.request_plan, context, AnalysisSummaryKind::DiscreteMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " detected first-order transport; request finite nonnormality/skew split matrix diagnostics and operator-scope provenance");
            addRequest(report.request_plan, context, AnalysisSummaryKind::InvariantDomain,
                       claim.domain, claim.variables, i, claim,
                       source + " detected transport character; request overshoot/undershoot and invariant-domain evidence when available");
        }

        if (claim.kind == PropertyKind::Stabilization ||
            hasSourceAnalyzer(claim, "StabilizationAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::ParameterScale,
                       claim.domain, claim.variables, i, claim,
                       source + " detected stabilization; request stabilization parameter and scaling metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::StabilizationAdequacy,
                       claim.domain, claim.variables, i, claim,
                       source + " detected stabilization; request parameter-formula, consistency, quantitative Peclet-regime, and CFL-bound adequacy evidence");
            addRequest(report.request_plan, context, AnalysisSummaryKind::DiscreteMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " detected stabilization; request retained matrix diagnostics for consistency and conditioning");
        }

        if (claim.kind == PropertyKind::ConservationStructure ||
            hasSourceAnalyzer(claim, "ConservationAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted a conservation/balance claim; request finite declared tolerance plus finite local, global, and interface flux-balance residuals with closure/source/orientation/time-scope metadata");
        }

        if (claim.kind == PropertyKind::DifferentialAlgebraicStructure ||
            hasSourceAnalyzer(claim, "DAEStructureAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::TemporalStability,
                       claim.domain, claim.variables, i, claim,
                       source + " classified temporal/DAE structure; request time scheme stability metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::DAEStructureEvidence,
                       claim.domain, claim.variables, i, claim,
                       source + " classified temporal/DAE structure; request mass-rank, algebraic-Jacobian rank, hidden-constraint, regular descriptor-pencil/index theorem, projector, and finite consistent-initialization residual/tolerance evidence");
            addRequest(report.request_plan, context, AnalysisSummaryKind::InitialCompatibility,
                       claim.domain, claim.variables, i, claim,
                       source + " classified temporal/DAE structure; request initial constraint and boundary residual evidence");
        }

        if (claim.kind == PropertyKind::SpaceCompatibility ||
            hasSourceAnalyzer(claim, "SpaceCompatibilityAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted a space/trace compatibility claim; request trace coverage and boundary-symbol evidence");
            if (claim.variables.size() >= 2) {
                addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupEstimate,
                           claim.domain, claim.variables, i, claim,
                           source + " emitted a mixed space compatibility claim; request inf-sup evidence for the space pair");
                addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupPairCertification,
                           claim.domain, claim.variables, i, claim,
                           source + " emitted a mixed space compatibility claim; request known stable-pair and Fortin metadata");
            }
        }

        if (claim.kind == PropertyKind::UnderConstraint ||
            claim.kind == PropertyKind::OverConstraint ||
            hasSourceAnalyzer(claim, "ConstraintRankAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted a constraint-rank claim; request constrained/free DOF and reduced matrix evidence");
            addRequest(report.request_plan, context, AnalysisSummaryKind::InitialCompatibility,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted a constraint-rank claim; request initial constraint residual evidence for transient/DAE use");
        }

        if (claim.kind == PropertyKind::TopologyScopedKernel ||
            hasSourceAnalyzer(claim, "TopologyScopeAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::MeshGeometryQuality,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted topology-scoped kernel evidence; request mesh region quality and revision metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted topology-scoped kernel evidence; request per-region reduced constraint evidence");
        }

        if (claim.kind == PropertyKind::CompatibilityCondition ||
            hasSourceAnalyzer(claim, "CompatibilityAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::InitialCompatibility,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted a compatibility condition; request initial/boundary compatibility residual evidence");
        }

        if (claim.kind == PropertyKind::InterfaceCondition ||
            claim.kind == PropertyKind::WeakBoundaryCoercivity ||
            claim.kind == PropertyKind::BoundaryComplementingCondition) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted interface/boundary coercivity evidence; request boundary-symbol and penalty adequacy metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted interface/boundary evidence; request numerical flux-balance metadata");
        }

        if (claim.kind == PropertyKind::CompatibleComplexStructure) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::CompatibleComplex,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted compatible-complex evidence; request exact-sequence, bounded commuting-projection, projection-bound, mesh-family, and shape-regularity metadata");
        }

        if (claim.kind == PropertyKind::TemporalStability) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::TemporalStability,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted temporal-stability evidence; request finite CFL, eigenvalue scale, amplification-radius, norm/theorem/scope, time-horizon, and nonnormal/pseudospectral metadata");
        }

        if (claim.kind == PropertyKind::EnergyStability ||
            claim.kind == PropertyKind::EntropyStability) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::EnergyEntropyBalance,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted energy/entropy law evidence; request finite balance/production/tolerance, theorem, functional/norm, coercivity, norm-equivalence, dissipation, entropy-variable, entropy-flux, convexity, and boundary/source accounting summaries");
        }

        if (claim.kind == PropertyKind::CoefficientPositivity ||
            claim.kind == PropertyKind::ParameterRobustness) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::CoefficientProperties,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted coefficient/parameter robustness evidence; request coefficient spectral bounds and contrast metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ParameterScale,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted parameter robustness evidence; request nondimensional parameter-scale summaries");
        }

        if (claim.kind == PropertyKind::NonlinearTangentStructure) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::NonlinearTangent,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted nonlinear tangent evidence; request exact/approximate tangent and finite-difference action summaries");
            addRequest(report.request_plan, context, AnalysisSummaryKind::DiscreteMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted nonlinear tangent evidence; request tangent block matrix diagnostics");
        }

        if (claim.kind == PropertyKind::LockingRisk) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::InfSupEstimate,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted locking/overstiffness evidence; request inf-sup and constraint-space estimates");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ReducedMatrix,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted locking/overstiffness evidence; request reduced block conditioning metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::ParameterScale,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted locking/overstiffness evidence; request parameter and resolution scales");
        }

        if (claim.kind == PropertyKind::SpectralCorrectness) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::SpectralStructure,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted spectral correctness evidence; request self-adjointness, compactness, and spurious-mode summaries");
            addRequest(report.request_plan, context, AnalysisSummaryKind::CompatibleComplex,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted spectral correctness evidence; request compatible-complex support where applicable");
        }

        if (claim.kind == PropertyKind::ErrorEstimatorEligibility) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::ErrorEstimator,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted estimator eligibility evidence; request residual, jump, flux-reconstruction, goal-weight, norm scope, PDE class, boundary residual, data oscillation, coefficient/source regularity, shape-regularity, reliability/efficiency, effectivity, and refinement metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted estimator eligibility evidence; request flux-balance residuals used by estimator checks");
        }

        if (claim.kind == PropertyKind::QuadratureAdequacy) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::QuadratureAdequacy,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted quadrature adequacy evidence; request exactness, aliasing, and hourglass-control metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::LocalStencil,
                              claim.domain, claim.variables, i, claim,
                              source + " emitted quadrature adequacy evidence; request local stencil diagnostics for zero-energy modes");
        }

        if (claim.kind == PropertyKind::MinimumResidualStability) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::MinimumResidualStability,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted minimum-residual/Petrov-Galerkin evidence; request trial/test space, Riesz map, Fortin/optimal-test, enrichment, residual-control, conditioning estimates, accepted conditioning bounds, and condition-scope metadata");
        }

        if (claim.kind == PropertyKind::InvariantDomainPreservation) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::InvariantDomain,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted invariant-domain evidence; request bound, limiter, quantitative CFL/wave-speed, theorem, and post-step violation summaries");
        }

        if (claim.kind == PropertyKind::EquilibriumPreservation) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::EquilibriumPreservation,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted equilibrium-preservation evidence; request finite residual/tolerance plus flux/source, reconstruction, quadrature, and boundary-compatibility metadata");
        }

        if (claim.kind == PropertyKind::GeometricConservation) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::MovingDomain,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted moving-domain geometric conservation evidence; request finite mapping Jacobian interval, declared GCL tolerance, theorem, metric-identity, free-stream residual, mesh-update, mesh-velocity, time-integration, and remap summaries");
        }

        if (claim.kind == PropertyKind::TransferOperatorCompatibility) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::TransferOperator,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted transfer-operator evidence; request finite residual/tolerance, projection conservation, constant preservation, interface scope, projection consistency, mortar/dual consistency, interface mass conditioning, action-reaction flux, and rank metadata");
        }

        if (claim.kind == PropertyKind::AdjointConsistency) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::AdjointConsistency,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted adjoint-consistency evidence; request transpose-backend, goal-functional, boundary-adjoint, stabilization-adjoint, goal-linearization, and finite discrete-adjoint residual/tolerance summaries");
        }

        if (claim.kind == PropertyKind::CoupledSystemStructure &&
            hasSourceAnalyzer(claim, "CoupledSystemStabilityAnalyzer")) {
            addRequest(report.request_plan, context, AnalysisSummaryKind::CoupledSystemStability,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted coupled-system stability evidence; request finite exchange residual, constraint drift, tolerance, partition spectral radius, contraction/coercive/nonnormal bounds, coupled-operator evidence, relaxation, added-mass, and partition metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::TemporalStability,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted coupled-system stability evidence; request coupled temporal stability metadata");
            addRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                       claim.domain, claim.variables, i, claim,
                       source + " emitted coupled-system stability evidence; request exchange balance summaries");
        }
    }

    for (const auto& contribution : context.contributions()) {
        const auto variables = contribution.test_variables.empty()
            ? contribution.trial_variables
            : contribution.test_variables;

        if (contribution.domain == DomainKind::InteriorFace ||
            contribution.domain == DomainKind::InterfaceFace) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              contribution.domain, variables,
                              "ProblemAnalysisContext has face contribution '" +
                                  contribution.operator_tag +
                                  "'; request trace/penalty boundary-symbol metadata",
                              AnalysisConfidence::Medium,
                              contribution.operator_tag,
                              contribution.contribution_id);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                              contribution.domain, variables,
                              "ProblemAnalysisContext has face contribution '" +
                                  contribution.operator_tag +
                                  "'; request numerical flux and interface-pair balance residuals",
                              AnalysisConfidence::Medium,
                              contribution.operator_tag,
                              contribution.contribution_id);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::LocalStencil,
                              contribution.domain, variables,
                              "ProblemAnalysisContext has DG/interface face contribution '" +
                                  contribution.operator_tag +
                                  "'; request local face stencil diagnostics",
                              AnalysisConfidence::Medium,
                              contribution.operator_tag,
                              contribution.contribution_id);
        }

        if (contribution.domain == DomainKind::InterfaceFace) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::TransferOperator,
                              contribution.domain, variables,
                              "ProblemAnalysisContext has interface contribution '" +
                                  contribution.operator_tag +
                                  "'; request projection/mortar transfer compatibility evidence",
                              AnalysisConfidence::Medium,
                              contribution.operator_tag,
                              contribution.contribution_id);
        }

        if (contribution.role == ContributionRole::BoundaryConstraint ||
            contribution.domain == DomainKind::Boundary) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              contribution.domain, variables,
                              "ProblemAnalysisContext has boundary contribution '" +
                                  contribution.operator_tag +
                                  "'; request weak-boundary coercivity and trace metadata",
                              AnalysisConfidence::Medium,
                              contribution.operator_tag,
                              contribution.contribution_id);
            if (contribution.adjoint_consistency.has_value()) {
                addContextRequest(report.request_plan, context, AnalysisSummaryKind::AdjointConsistency,
                                  contribution.domain, variables,
                                  "ProblemAnalysisContext has boundary contribution '" +
                                      contribution.operator_tag +
                                      "' with adjoint-consistency metadata; request adjoint consistency summary",
                                  AnalysisConfidence::Medium,
                                  contribution.operator_tag,
                                  contribution.contribution_id);
            }
        }
    }

    for (const auto& record : context.formulationRecords()) {
        const auto variables = record.active_variables.empty()
            ? variablesForFields(record.active_fields)
            : record.active_variables;

        bool has_interior_face = record.has_interior_face_terms;
        bool has_interface_face = false;
        bool has_boundary_face = false;
        for (DomainKind domain : record.active_domains) {
            has_interior_face = has_interior_face || domain == DomainKind::InteriorFace;
            has_interface_face = has_interface_face || domain == DomainKind::InterfaceFace;
            has_boundary_face = has_boundary_face || domain == DomainKind::Boundary;
        }

        if (has_interior_face) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              DomainKind::InteriorFace, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has interior-face/DG terms; request penalty and trace-symbol metadata",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                              DomainKind::InteriorFace, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has interior-face/DG terms; request numerical flux-balance summaries",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::LocalStencil,
                              DomainKind::InteriorFace, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has interior-face/DG terms; request local face-stencil diagnostics",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
        }

        if (has_interface_face) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              DomainKind::InterfaceFace, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has interface-face terms; request interface trace and penalty metadata",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                              DomainKind::InterfaceFace, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has interface-face terms; request interface flux-balance summaries",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::TransferOperator,
                              DomainKind::InterfaceFace, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has interface-face terms; request transfer/projection compatibility evidence",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
        }

        if (has_boundary_face) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              DomainKind::Boundary, variables,
                              "FormulationRecord '" + record.operator_tag +
                                  "' has boundary-face terms; request boundary-symbol metadata",
                              AnalysisConfidence::Medium,
                              record.operator_tag);
        }
    }

    for (const auto& bc : context.bcDescriptors()) {
        std::vector<VariableKey> variables{bc.primary_variable};
        appendVariables(variables, bc.related_variables);

        if (bc.domain == DomainKind::InterfaceFace) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              DomainKind::InterfaceFace, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' targets an interface; request interface boundary-symbol metadata",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                              DomainKind::InterfaceFace, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' targets an interface; request interface flux-balance metadata",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::TransferOperator,
                              DomainKind::InterfaceFace, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' targets an interface; request transfer/projection compatibility metadata",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
        }

        if (isWeakPenaltyLike(bc.enforcement_kind)) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::BoundarySymbol,
                              bc.domain, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' uses weak penalty/Nitsche enforcement; request penalty adequacy and trace metadata",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::ParameterScale,
                              bc.domain, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' uses weak penalty/Nitsche enforcement; request penalty scaling metadata",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::FluxBalance,
                              bc.domain, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' uses weak enforcement; request boundary/interface flux-balance metadata",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
        }

        if (bc.adjoint_consistency.has_value()) {
            addContextRequest(report.request_plan, context, AnalysisSummaryKind::AdjointConsistency,
                              bc.domain, variables,
                              "BoundaryConditionDescriptor from '" + bc.source +
                                  "' provides adjoint-consistency metadata; request adjoint consistency summary",
                              AnalysisConfidence::Medium,
                              {},
                              {},
                              bc.source);
        }
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
