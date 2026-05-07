/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/FortinOperatorAutogeneration.h"

#include "Basis/BubbleBasis.h"
#include "Basis/LagrangeBasis.h"
#include "Basis/VectorBasis.h"
#include "Forms/FormExpr.h"
#include "Quadrature/QuadratureFactory.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <memory>
#include <sstream>
#include <utility>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

template <typename T>
void appendUnique(std::vector<T>& values, const T& value)
{
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

bool sameUnorderedPair(const VariableKey& a,
                       const VariableKey& b,
                       const VariableKey& c,
                       const VariableKey& d)
{
    return (a == c && b == d) || (a == d && b == c);
}

template <typename T>
bool contains(const std::vector<T>& values, const T& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool isScalarField(const ProblemAnalysisContext& context,
                   const VariableKey& variable)
{
    if (variable.kind != VariableKind::FieldComponent) {
        return false;
    }
    const auto* fd = context.fieldDescriptor(variable.field_id);
    return fd && fd->value_dimension == 1 &&
           fd->field_type == FieldType::Scalar;
}

bool isPrimalVectorLikeField(const ProblemAnalysisContext& context,
                             const VariableKey& variable)
{
    if (variable.kind != VariableKind::FieldComponent) {
        return false;
    }
    const auto* fd = context.fieldDescriptor(variable.field_id);
    if (!fd) {
        return false;
    }
    return fd->field_type == FieldType::Vector ||
           fd->value_dimension > 1 ||
           fd->space_family == SpaceFamily::HDiv ||
           fd->space_family == SpaceFamily::HCurl;
}

std::pair<VariableKey, VariableKey>
selectPrimalMultiplier(const ProblemAnalysisContext& context,
                       const VariableKey& a,
                       const VariableKey& b)
{
    const bool a_primal = isPrimalVectorLikeField(context, a);
    const bool b_primal = isPrimalVectorLikeField(context, b);
    const bool a_scalar = isScalarField(context, a);
    const bool b_scalar = isScalarField(context, b);

    if (a_primal && b_scalar) {
        return {a, b};
    }
    if (b_primal && a_scalar) {
        return {b, a};
    }
    return {a, b};
}

bool nodeContainsFieldTerminal(const forms::FormExprNode& node, FieldId field)
{
    using FT = forms::FormExprType;
    switch (node.type()) {
        case FT::TestFunction:
        case FT::TrialFunction:
        case FT::DiscreteField:
        case FT::StateField:
            if (const auto id = node.fieldId()) {
                if (*id == field) {
                    return true;
                }
            }
            break;
        default:
            break;
    }

    for (const auto* child : node.children()) {
        if (child && nodeContainsFieldTerminal(*child, field)) {
            return true;
        }
    }
    return false;
}

bool nodeContainsDifferentialField(const forms::FormExprNode& node,
                                   forms::FormExprType op,
                                   FieldId field)
{
    if (node.type() == op) {
        for (const auto* child : node.children()) {
            if (child && nodeContainsFieldTerminal(*child, field)) {
                return true;
            }
        }
    }
    for (const auto* child : node.children()) {
        if (child && nodeContainsDifferentialField(*child, op, field)) {
            return true;
        }
    }
    return false;
}

bool expressionHasDivergenceCoupling(const ContributionDescriptor& contribution,
                                     FieldId primal_field,
                                     FieldId multiplier_field)
{
    if (!contribution.source_expression) {
        return false;
    }
    const auto product_level_coupling =
        [&](const auto& self, const forms::FormExprNode& node) -> bool {
            using FT = forms::FormExprType;
            const auto ty = node.type();
            if (ty == FT::Multiply ||
                ty == FT::InnerProduct ||
                ty == FT::DoubleContraction) {
                if (nodeContainsDifferentialField(node,
                                                  FT::Divergence,
                                                  primal_field) &&
                    nodeContainsFieldTerminal(node, multiplier_field)) {
                    return true;
                }
            }
            for (const auto* child : node.children()) {
                if (child && self(self, *child)) {
                    return true;
                }
            }
            return false;
        };
    return product_level_coupling(product_level_coupling,
                                  *contribution.source_expression);
}

bool expressionHasGradientAdjointCoupling(
    const forms::FormExprNode& node,
    FieldId primal_field,
    FieldId multiplier_field)
{
    using FT = forms::FormExprType;
    if (node.type() == FT::InnerProduct) {
        const auto children = node.children();
        if (children.size() >= 2u && children[0] && children[1]) {
            const bool left_grad_multiplier =
                nodeContainsDifferentialField(*children[0],
                                              FT::Gradient,
                                              multiplier_field);
            const bool right_grad_multiplier =
                nodeContainsDifferentialField(*children[1],
                                              FT::Gradient,
                                              multiplier_field);
            const bool left_primal =
                nodeContainsFieldTerminal(*children[0], primal_field);
            const bool right_primal =
                nodeContainsFieldTerminal(*children[1], primal_field);
            if ((left_grad_multiplier && right_primal) ||
                (right_grad_multiplier && left_primal)) {
                return true;
            }
        }
    }

    for (const auto* child : node.children()) {
        if (child &&
            expressionHasGradientAdjointCoupling(*child,
                                                 primal_field,
                                                 multiplier_field)) {
            return true;
        }
    }
    return false;
}

bool expressionHasGradientAdjointCoupling(const ContributionDescriptor& contribution,
                                          FieldId primal_field,
                                          FieldId multiplier_field)
{
    return contribution.source_expression &&
           expressionHasGradientAdjointCoupling(*contribution.source_expression,
                                                primal_field,
                                                multiplier_field);
}

MixedCouplingFamily familyFromSpaces(const ProblemAnalysisContext& context,
                                     const VariableKey& primal,
                                     const VariableKey& multiplier,
                                     DomainKind domain)
{
    if (domain == DomainKind::Boundary ||
        domain == DomainKind::InterfaceFace) {
        const auto* m = multiplier.kind == VariableKind::FieldComponent
            ? context.fieldDescriptor(multiplier.field_id)
            : nullptr;
        if (m && m->element_family == ElementFamily::Mortar) {
            return MixedCouplingFamily::MortarConstraint;
        }
        return MixedCouplingFamily::TraceMultiplier;
    }

    const auto* p = primal.kind == VariableKind::FieldComponent
        ? context.fieldDescriptor(primal.field_id)
        : nullptr;
    const auto* m = multiplier.kind == VariableKind::FieldComponent
        ? context.fieldDescriptor(multiplier.field_id)
        : nullptr;
    if (p && m && p->space_family == SpaceFamily::HDiv &&
        (m->space_family == SpaceFamily::L2 ||
         m->space_family == SpaceFamily::DG)) {
        return MixedCouplingFamily::MixedHDivDivergence;
    }
    if (p && m && isPrimalVectorLikeField(context, primal) &&
        isScalarField(context, multiplier)) {
        return MixedCouplingFamily::VectorDivergenceScalarMultiplier;
    }
    return MixedCouplingFamily::GenericMultiplierConstraint;
}

MixedCouplingEvidenceStrength stronger(MixedCouplingEvidenceStrength a,
                                       MixedCouplingEvidenceStrength b) noexcept
{
    auto rank = [](MixedCouplingEvidenceStrength s) {
        switch (s) {
            case MixedCouplingEvidenceStrength::Strong: return 3;
            case MixedCouplingEvidenceStrength::Weak: return 2;
            case MixedCouplingEvidenceStrength::Ambiguous: return 1;
            case MixedCouplingEvidenceStrength::None: return 0;
        }
        return 0;
    };
    return rank(b) > rank(a) ? b : a;
}

void mergeDescriptor(std::vector<MixedCouplingDescriptor>& descriptors,
                     MixedCouplingDescriptor descriptor)
{
    auto it = std::find_if(
        descriptors.begin(),
        descriptors.end(),
        [&](const MixedCouplingDescriptor& existing) {
            return sameUnorderedPair(existing.primal_variable,
                                     existing.multiplier_variable,
                                     descriptor.primal_variable,
                                     descriptor.multiplier_variable) &&
                   existing.domain == descriptor.domain &&
                   existing.boundary_marker == descriptor.boundary_marker &&
                   existing.interface_marker == descriptor.interface_marker &&
                   existing.pairing_group == descriptor.pairing_group;
        });
    if (it == descriptors.end()) {
        descriptors.push_back(std::move(descriptor));
        return;
    }

    for (const auto& id : descriptor.contribution_ids) {
        appendUnique(it->contribution_ids, id);
    }
    it->evidence_from_dag =
        it->evidence_from_dag || descriptor.evidence_from_dag;
    it->evidence_from_contribution_role =
        it->evidence_from_contribution_role ||
        descriptor.evidence_from_contribution_role;
    it->evidence_from_pairing_metadata =
        it->evidence_from_pairing_metadata ||
        descriptor.evidence_from_pairing_metadata;
    it->has_stabilization_surrogate =
        it->has_stabilization_surrogate ||
        descriptor.has_stabilization_surrogate;
    it->evidence_strength =
        stronger(it->evidence_strength, descriptor.evidence_strength);
    if (it->coupling_family == MixedCouplingFamily::Unknown ||
        it->coupling_family == MixedCouplingFamily::GenericMultiplierConstraint) {
        it->coupling_family = descriptor.coupling_family;
    }
    for (const auto& diagnostic : descriptor.diagnostics) {
        appendUnique(it->diagnostics, diagnostic);
    }
}

MixedCouplingDescriptor descriptorFromVariables(
    const ProblemAnalysisContext& context,
    const ContributionDescriptor& contribution,
    const VariableKey& a,
    const VariableKey& b,
    PairingKind pairing_kind,
    const std::string& pairing_group,
    bool pairing_has_stabilizing_surrogate)
{
    auto [primal, multiplier] = selectPrimalMultiplier(context, a, b);

    MixedCouplingDescriptor descriptor;
    descriptor.primal_variable = primal;
    descriptor.multiplier_variable = multiplier;
    descriptor.domain = contribution.domain;
    descriptor.boundary_marker = contribution.boundary_marker;
    descriptor.interface_marker = contribution.interface_marker;
    descriptor.operator_tag = contribution.operator_tag;
    descriptor.pairing_group = pairing_group;
    descriptor.contribution_ids.push_back(contribution.contribution_id);
    descriptor.evidence_from_contribution_role =
        contribution.role == ContributionRole::ConstraintBlock ||
        contribution.role == ContributionRole::OffDiagonalBlock ||
        contribution.role == ContributionRole::BoundaryConstraint ||
        contribution.role == ContributionRole::InterfaceCoupling;
    descriptor.evidence_from_pairing_metadata =
        pairing_kind != PairingKind::Unknown;
    descriptor.has_stabilization_surrogate =
        pairing_kind == PairingKind::StabilizedConstraintPair ||
        pairing_has_stabilizing_surrogate;

    descriptor.coupling_family =
        familyFromSpaces(context, primal, multiplier, contribution.domain);
    descriptor.evidence_strength =
        descriptor.evidence_from_pairing_metadata ||
        descriptor.evidence_from_contribution_role
            ? MixedCouplingEvidenceStrength::Weak
            : MixedCouplingEvidenceStrength::Ambiguous;

    if (primal.kind == VariableKind::FieldComponent &&
        multiplier.kind == VariableKind::FieldComponent) {
        const bool div_evidence =
            expressionHasDivergenceCoupling(contribution,
                                            primal.field_id,
                                            multiplier.field_id);
        const bool grad_adj_evidence =
            expressionHasGradientAdjointCoupling(contribution,
                                                primal.field_id,
                                                multiplier.field_id);
        if (div_evidence) {
            descriptor.evidence_from_dag = true;
            descriptor.evidence_strength =
                MixedCouplingEvidenceStrength::Strong;
            descriptor.coupling_family =
                familyFromSpaces(context, primal, multiplier, contribution.domain);
            descriptor.diagnostics.push_back(
                "DAG contains multiplier field coupled to div(primal)");
        } else if (grad_adj_evidence) {
            descriptor.evidence_from_dag = true;
            descriptor.evidence_strength =
                MixedCouplingEvidenceStrength::Strong;
            descriptor.coupling_family =
                MixedCouplingFamily::GradientDivergenceAdjoint;
            descriptor.diagnostics.push_back(
                "DAG contains grad(multiplier) dot primal adjoint form");
        } else if (!contribution.source_expression) {
            descriptor.diagnostics.push_back(
                "No source FormExpr DAG; classification uses contribution metadata");
        } else {
            descriptor.diagnostics.push_back(
                "FormExpr DAG did not expose a supported divergence/adjoint pattern");
        }
    }

    if (descriptor.has_stabilization_surrogate) {
        descriptor.diagnostics.push_back(
            "Pairing declares a stabilization surrogate; Fortin certification is blocked");
    }
    return descriptor;
}

bool couplingCompatible(MixedCouplingFamily actual,
                        MixedCouplingFamily required) noexcept
{
    if (actual == required) {
        return true;
    }
    return required == MixedCouplingFamily::VectorDivergenceScalarMultiplier &&
           actual == MixedCouplingFamily::GradientDivergenceAdjoint;
}

void addReason(FortinRegistryMatch& match, FortinRejectionReason reason)
{
    appendUnique(match.rejection_reasons, reason);
}

bool metadataKnownForRequirement(const FieldDescriptor& field,
                                 const FortinSpaceRequirement& req)
{
    const bool order_in_range =
        field.polynomial_order >= req.minimum_order &&
        (req.maximum_order < 0 || field.polynomial_order <= req.maximum_order);
    return field.space_family != SpaceFamily::Unknown &&
           field.element_family != ElementFamily::Unknown &&
           field.continuity_class != SpaceContinuityClass::Unknown &&
           field.mapping_transform != MappingTransform::Unknown &&
           order_in_range;
}

bool spaceRequirementSatisfied(const FieldDescriptor& field,
                               const FortinSpaceRequirement& req)
{
    const bool space_ok = req.space_families.empty() ||
        contains(req.space_families, field.space_family);
    const bool element_ok = req.element_families.empty() ||
        contains(req.element_families, field.element_family);
    const bool continuity_ok =
        req.continuity_class == SpaceContinuityClass::Unknown ||
        field.continuity_class == req.continuity_class;
    const bool mapping_ok =
        req.mapping_transform == MappingTransform::Unknown ||
        field.mapping_transform == req.mapping_transform;
    const bool order_ok =
        field.polynomial_order >= req.minimum_order &&
        (req.maximum_order < 0 || field.polynomial_order <= req.maximum_order);
    return space_ok && element_ok && continuity_ok && mapping_ok &&
           order_ok;
}

bool orderRelationSatisfied(const FieldDescriptor& primal,
                            const FieldDescriptor& multiplier,
                            FortinPolynomialOrderRelation relation)
{
    switch (relation) {
        case FortinPolynomialOrderRelation::Equal:
            return primal.polynomial_order == multiplier.polynomial_order;
        case FortinPolynomialOrderRelation::PrimalOneHigher:
            return primal.polynomial_order == multiplier.polynomial_order + 1;
        case FortinPolynomialOrderRelation::MiniP1P1:
            return primal.polynomial_order == 1 &&
                   multiplier.polynomial_order == 1 &&
                   primal.enrichment.visible_to_analysis &&
                   primal.enrichment.bubble_degree > 0;
    }
    return false;
}

void addCandidateReason(FortinCandidate& candidate,
                        FortinRejectionReason reason)
{
    appendUnique(candidate.missing_or_rejected_reasons, reason);
}

bool meshSummarySatisfiesShapeRegularity(const AnalysisSummarySet* summaries)
{
    if (!summaries) {
        return false;
    }
    return std::any_of(
        summaries->mesh_geometry_quality.begin(),
        summaries->mesh_geometry_quality.end(),
        [](const MeshGeometryQualitySummary& summary) {
            return summary.shape_regular_evidence_present &&
                   summary.mesh_family_scope_present &&
                   summary.inverted_element_count == 0u;
        });
}

bool meshAssumptionSatisfied(const ProblemAnalysisContext& context,
                             const FieldDescriptor& primal,
                             const FieldDescriptor& multiplier)
{
    if (primal.shape_regular_mesh_assumed &&
        multiplier.shape_regular_mesh_assumed) {
        return true;
    }
    return meshSummarySatisfiesShapeRegularity(context.analysisSummaries());
}

bool domainAssumptionSatisfied(const FieldDescriptor& primal,
                               const FieldDescriptor& multiplier)
{
    return primal.domain_assumptions_present &&
           multiplier.domain_assumptions_present &&
           primal.lipschitz_domain_assumed &&
           multiplier.lipschitz_domain_assumed;
}

bool multiplierGaugeHandled(const FieldDescriptor& multiplier)
{
    return multiplier.mean_zero_constraint_present ||
           multiplier.gauge_fixing_metadata_present;
}

bool boundaryNullspaceAssumptionSatisfied(
    const MixedCouplingDescriptor& coupling,
    const FieldDescriptor& primal,
    const FieldDescriptor& multiplier)
{
    if (coupling.coupling_family == MixedCouplingFamily::VectorDivergenceScalarMultiplier ||
        coupling.coupling_family ==
            MixedCouplingFamily::GradientDivergenceAdjoint) {
        return primal.strong_dirichlet_boundary_present &&
               multiplierGaugeHandled(multiplier);
    }
    if (coupling.coupling_family == MixedCouplingFamily::MixedHDivDivergence) {
        return primal.normal_trace_boundary_scope_present ||
               multiplierGaugeHandled(multiplier) ||
               primal.boundary_condition_scope_metadata_present;
    }
    if (coupling.coupling_family == MixedCouplingFamily::TraceMultiplier ||
        coupling.coupling_family == MixedCouplingFamily::MortarConstraint) {
        return false;
    }
    return primal.boundary_condition_scope_metadata_present ||
           multiplierGaugeHandled(multiplier);
}

std::string globalConstraintHandling(const FieldDescriptor& multiplier)
{
    if (multiplier.mean_zero_constraint_present) {
        return "mean-zero multiplier constraint";
    }
    if (multiplier.gauge_fixing_metadata_present) {
        return "multiplier gauge fixing metadata";
    }
    return "not declared";
}

std::string variableShortLabel(const VariableKey& variable)
{
    if (variable.kind == VariableKind::FieldComponent) {
        return "field=" + std::to_string(variable.field_id);
    }
    return std::string(toString(variable.kind)) + "=" + variable.name;
}

std::string reasonsLabel(const std::vector<FortinRejectionReason>& reasons)
{
    std::string label;
    for (std::size_t i = 0; i < reasons.size(); ++i) {
        if (i != 0u) {
            label += ",";
        }
        label += toString(reasons[i]);
    }
    return label;
}

bool reportAlreadyHasCertifiedPair(const ProblemAnalysisReport& report,
                                   const VariableKey& a,
                                   const VariableKey& b)
{
    return std::any_of(
        report.claims.begin(),
        report.claims.end(),
        [&](const PropertyClaim& claim) {
            return claim.kind == PropertyKind::InfSupCondition &&
                   claim.certification_class &&
                   *claim.certification_class == CertificationClass::Certified &&
                   claim.variables.size() >= 2u &&
                   sameUnorderedPair(a, b, claim.variables[0], claim.variables[1]);
        });
}

struct DenseMatrix {
    int rows{0};
    int cols{0};
    std::vector<Real> values;

    DenseMatrix() = default;
    DenseMatrix(int r, int c)
        : rows(r), cols(c), values(static_cast<std::size_t>(r * c), Real(0)) {}

    Real& operator()(int r, int c) {
        return values[static_cast<std::size_t>(r * cols + c)];
    }

    Real operator()(int r, int c) const {
        return values[static_cast<std::size_t>(r * cols + c)];
    }
};

[[nodiscard]] Real denseFrobeniusNorm(const DenseMatrix& matrix)
{
    Real sum = Real(0);
    for (const auto value : matrix.values) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

[[nodiscard]] Real denseMaxAbs(const DenseMatrix& matrix)
{
    Real max_value = Real(0);
    for (const auto value : matrix.values) {
        max_value = std::max(max_value, std::abs(value));
    }
    return max_value;
}

[[nodiscard]] Real denseMaxAbs(const std::vector<Real>& values)
{
    Real max_value = Real(0);
    for (const auto value : values) {
        max_value = std::max(max_value, std::abs(value));
    }
    return max_value;
}

[[nodiscard]] bool solveDenseSquare(DenseMatrix matrix,
                                    std::vector<Real> rhs,
                                    std::vector<Real>& solution,
                                    Real tolerance,
                                    std::string& diagnostic)
{
    const int n = matrix.rows;
    if (matrix.rows != matrix.cols ||
        static_cast<int>(rhs.size()) != n) {
        diagnostic = "dense solve received incompatible dimensions";
        return false;
    }

    for (int k = 0; k < n; ++k) {
        int pivot = k;
        Real pivot_abs = std::abs(matrix(k, k));
        for (int r = k + 1; r < n; ++r) {
            const Real value = std::abs(matrix(r, k));
            if (value > pivot_abs) {
                pivot_abs = value;
                pivot = r;
            }
        }
        if (pivot_abs <= tolerance) {
            diagnostic = "dense solve found a singular or rank-deficient local system";
            return false;
        }
        if (pivot != k) {
            for (int c = k; c < n; ++c) {
                std::swap(matrix(k, c), matrix(pivot, c));
            }
            std::swap(rhs[static_cast<std::size_t>(k)],
                      rhs[static_cast<std::size_t>(pivot)]);
        }

        const Real diag = matrix(k, k);
        for (int r = k + 1; r < n; ++r) {
            const Real factor = matrix(r, k) / diag;
            matrix(r, k) = Real(0);
            for (int c = k + 1; c < n; ++c) {
                matrix(r, c) -= factor * matrix(k, c);
            }
            rhs[static_cast<std::size_t>(r)] -=
                factor * rhs[static_cast<std::size_t>(k)];
        }
    }

    solution.assign(static_cast<std::size_t>(n), Real(0));
    for (int i = n - 1; i >= 0; --i) {
        Real value = rhs[static_cast<std::size_t>(i)];
        for (int c = i + 1; c < n; ++c) {
            value -= matrix(i, c) * solution[static_cast<std::size_t>(c)];
        }
        const Real diag = matrix(i, i);
        if (std::abs(diag) <= tolerance) {
            diagnostic = "dense solve found a zero diagonal during back substitution";
            return false;
        }
        solution[static_cast<std::size_t>(i)] = value / diag;
    }
    return true;
}

[[nodiscard]] std::vector<int> independentConstraintRows(const DenseMatrix& rows,
                                                         Real tolerance)
{
    std::vector<int> selected;
    std::vector<std::vector<Real>> basis_rows;
    std::vector<int> pivots;

    for (int r = 0; r < rows.rows; ++r) {
        std::vector<Real> candidate(static_cast<std::size_t>(rows.cols), Real(0));
        for (int c = 0; c < rows.cols; ++c) {
            candidate[static_cast<std::size_t>(c)] = rows(r, c);
        }

        for (std::size_t i = 0; i < basis_rows.size(); ++i) {
            const int pivot_col = pivots[i];
            const Real factor = candidate[static_cast<std::size_t>(pivot_col)];
            if (std::abs(factor) <= tolerance) {
                continue;
            }
            const auto& basis = basis_rows[i];
            for (int c = pivot_col; c < rows.cols; ++c) {
                candidate[static_cast<std::size_t>(c)] -=
                    factor * basis[static_cast<std::size_t>(c)];
            }
        }

        int pivot_col = -1;
        Real pivot_abs = tolerance;
        for (int c = 0; c < rows.cols; ++c) {
            const Real value = std::abs(candidate[static_cast<std::size_t>(c)]);
            if (value > pivot_abs) {
                pivot_abs = value;
                pivot_col = c;
            }
        }
        if (pivot_col < 0) {
            continue;
        }

        const Real inv_pivot =
            Real(1) / candidate[static_cast<std::size_t>(pivot_col)];
        for (auto& value : candidate) {
            value *= inv_pivot;
        }
        selected.push_back(r);
        pivots.push_back(pivot_col);
        basis_rows.push_back(std::move(candidate));
    }

    return selected;
}

[[nodiscard]] ElementType canonicalElementFor(ReferenceCellFamily family,
                                              int dimension)
{
    if (family == ReferenceCellFamily::Simplex) {
        if (dimension == 1) return ElementType::Line2;
        if (dimension == 2) return ElementType::Triangle3;
        if (dimension == 3) return ElementType::Tetra4;
    }
    if (family == ReferenceCellFamily::TensorProduct) {
        if (dimension == 1) return ElementType::Line2;
        if (dimension == 2) return ElementType::Quad4;
        if (dimension == 3) return ElementType::Hex8;
    }
    if (family == ReferenceCellFamily::Wedge && dimension == 3) {
        return ElementType::Wedge6;
    }
    if (family == ReferenceCellFamily::Pyramid && dimension == 3) {
        return ElementType::Pyramid5;
    }
    return ElementType::Unknown;
}

struct ScalarReferenceBasis {
    std::shared_ptr<const basis::LagrangeBasis> lagrange;
    std::shared_ptr<const basis::BubbleBasis> bubble;
    bool include_bubble{false};

    [[nodiscard]] int size() const noexcept {
        int count = lagrange ? static_cast<int>(lagrange->size()) : 0;
        if (include_bubble && bubble) {
            ++count;
        }
        return count;
    }

    [[nodiscard]] int polynomialOrder() const noexcept {
        int order = lagrange ? lagrange->order() : 0;
        if (include_bubble && bubble) {
            order = std::max(order, bubble->order());
        }
        return order;
    }

    void evaluateValues(const math::Vector<Real, 3>& xi,
                        std::vector<Real>& values) const
    {
        values.clear();
        if (lagrange) {
            lagrange->evaluate_values(xi, values);
        }
        if (include_bubble && bubble) {
            std::vector<Real> bubble_values;
            bubble->evaluate_values(xi, bubble_values);
            values.push_back(bubble_values.empty() ? Real(0) : bubble_values[0]);
        }
    }

    void evaluateGradients(const math::Vector<Real, 3>& xi,
                           std::vector<basis::Gradient>& gradients) const
    {
        gradients.clear();
        if (lagrange) {
            lagrange->evaluate_gradients(xi, gradients);
        }
        if (include_bubble && bubble) {
            std::vector<basis::Gradient> bubble_gradients;
            bubble->evaluate_gradients(xi, bubble_gradients);
            gradients.push_back(bubble_gradients.empty()
                                    ? basis::Gradient{}
                                    : bubble_gradients[0]);
        }
    }
};

struct VectorReferenceBasis {
    enum class Kind {
        ComponentScalar,
        HDivVector,
    };

    Kind kind{Kind::ComponentScalar};
    int dimension{0};
    int value_dimension{0};
    ElementFamily element_family{ElementFamily::Unknown};
    ScalarReferenceBasis scalar;
    std::shared_ptr<const basis::BasisFunction> vector_basis;

    [[nodiscard]] int size() const noexcept {
        if (kind == Kind::HDivVector) {
            return vector_basis ? static_cast<int>(vector_basis->size()) : 0;
        }
        return scalar.size() * value_dimension;
    }

    [[nodiscard]] int polynomialOrder() const noexcept {
        if (kind == Kind::HDivVector) {
            return vector_basis ? vector_basis->order() : 0;
        }
        return scalar.polynomialOrder();
    }

    void evaluateValues(const math::Vector<Real, 3>& xi,
                        std::vector<math::Vector<Real, 3>>& values) const
    {
        values.assign(static_cast<std::size_t>(size()), math::Vector<Real, 3>{});
        if (kind == Kind::HDivVector) {
            if (vector_basis) {
                vector_basis->evaluate_vector_values(xi, values);
            }
            return;
        }

        std::vector<Real> scalar_values;
        scalar.evaluateValues(xi, scalar_values);
        const int scalar_count = static_cast<int>(scalar_values.size());
        for (int component = 0; component < value_dimension; ++component) {
            for (int i = 0; i < scalar_count; ++i) {
                const int dof = component * scalar_count + i;
                if (component < 3) {
                    values[static_cast<std::size_t>(dof)][component] =
                        scalar_values[static_cast<std::size_t>(i)];
                }
            }
        }
    }

    void evaluateDivergence(const math::Vector<Real, 3>& xi,
                            std::vector<Real>& divergence) const
    {
        divergence.assign(static_cast<std::size_t>(size()), Real(0));
        if (kind == Kind::HDivVector) {
            if (vector_basis) {
                vector_basis->evaluate_divergence(xi, divergence);
            }
            return;
        }

        std::vector<basis::Gradient> gradients;
        scalar.evaluateGradients(xi, gradients);
        const int scalar_count = static_cast<int>(gradients.size());
        for (int component = 0; component < value_dimension; ++component) {
            for (int i = 0; i < scalar_count; ++i) {
                const int dof = component * scalar_count + i;
                divergence[static_cast<std::size_t>(dof)] =
                    component < dimension && component < 3
                        ? gradients[static_cast<std::size_t>(i)][component]
                        : Real(0);
            }
        }
    }
};

struct LocalProjectionBuildSpec {
    FieldDescriptor primal;
    FieldDescriptor multiplier;
    ReferenceCellFamily reference_cell_family{ReferenceCellFamily::Unknown};
    int dimension{0};
    int value_dimension{0};
};

[[nodiscard]] FieldDescriptor fallbackPrimalDescriptor(
    const FortinCandidate& candidate)
{
    FieldDescriptor fd;
    fd.field_id = candidate.coupling.primal_variable.field_id;
    fd.field_type = FieldType::Vector;
    fd.value_dimension = !candidate.theorem.supported_dimensions.empty()
        ? candidate.theorem.supported_dimensions.front()
        : 2;
    fd.topological_dimension = fd.value_dimension;
    fd.polynomial_order = candidate.theorem.primal_space.minimum_order;
    fd.space_family = candidate.theorem.primal_space.space_families.empty()
        ? SpaceFamily::Unknown
        : candidate.theorem.primal_space.space_families.front();
    fd.element_family = candidate.theorem.primal_space.element_families.empty()
        ? ElementFamily::Unknown
        : candidate.theorem.primal_space.element_families.front();
    fd.reference_cell_family = candidate.theorem.supported_cell_families.empty()
        ? ReferenceCellFamily::Simplex
        : candidate.theorem.supported_cell_families.front();
    return fd;
}

[[nodiscard]] FieldDescriptor fallbackMultiplierDescriptor(
    const FortinCandidate& candidate)
{
    FieldDescriptor fd;
    fd.field_id = candidate.coupling.multiplier_variable.field_id;
    fd.field_type = FieldType::Scalar;
    fd.value_dimension = 1;
    fd.topological_dimension = !candidate.theorem.supported_dimensions.empty()
        ? candidate.theorem.supported_dimensions.front()
        : 2;
    fd.polynomial_order = candidate.theorem.multiplier_space.minimum_order;
    fd.space_family = candidate.theorem.multiplier_space.space_families.empty()
        ? SpaceFamily::Unknown
        : candidate.theorem.multiplier_space.space_families.front();
    fd.element_family = candidate.theorem.multiplier_space.element_families.empty()
        ? ElementFamily::Unknown
        : candidate.theorem.multiplier_space.element_families.front();
    fd.reference_cell_family = candidate.theorem.supported_cell_families.empty()
        ? ReferenceCellFamily::Simplex
        : candidate.theorem.supported_cell_families.front();
    return fd;
}

[[nodiscard]] LocalProjectionBuildSpec makeLocalProjectionBuildSpec(
    const FortinCandidate& candidate,
    const ProblemAnalysisContext* context)
{
    LocalProjectionBuildSpec spec;
    const auto* primal =
        context && candidate.coupling.primal_variable.kind ==
                       VariableKind::FieldComponent
            ? context->fieldDescriptor(candidate.coupling.primal_variable.field_id)
            : nullptr;
    const auto* multiplier =
        context && candidate.coupling.multiplier_variable.kind ==
                       VariableKind::FieldComponent
            ? context->fieldDescriptor(candidate.coupling.multiplier_variable.field_id)
            : nullptr;

    spec.primal = primal ? *primal : fallbackPrimalDescriptor(candidate);
    spec.multiplier =
        multiplier ? *multiplier : fallbackMultiplierDescriptor(candidate);
    spec.dimension = spec.primal.topological_dimension > 0
        ? spec.primal.topological_dimension
        : spec.multiplier.topological_dimension;
    if (spec.dimension <= 0 &&
        !candidate.theorem.supported_dimensions.empty()) {
        spec.dimension = candidate.theorem.supported_dimensions.front();
    }
    spec.value_dimension = spec.primal.value_dimension > 0
        ? spec.primal.value_dimension
        : spec.dimension;
    spec.reference_cell_family =
        spec.primal.reference_cell_family != ReferenceCellFamily::Unknown
            ? spec.primal.reference_cell_family
            : spec.multiplier.reference_cell_family;
    if (spec.reference_cell_family == ReferenceCellFamily::Unknown &&
        !candidate.theorem.supported_cell_families.empty()) {
        spec.reference_cell_family =
            candidate.theorem.supported_cell_families.front();
    }
    return spec;
}

[[nodiscard]] ScalarReferenceBasis makeScalarReferenceBasis(
    ElementType element_type,
    int order,
    bool include_bubble)
{
    ScalarReferenceBasis basis;
    basis.lagrange =
        std::make_shared<basis::LagrangeBasis>(element_type, order);
    basis.include_bubble = include_bubble;
    if (include_bubble) {
        basis.bubble =
            std::make_shared<basis::BubbleBasis>(element_type);
    }
    return basis;
}

[[nodiscard]] VectorReferenceBasis makeComponentVectorBasis(
    ElementType element_type,
    int dimension,
    int value_dimension,
    int order,
    bool include_bubble,
    ElementFamily family)
{
    VectorReferenceBasis basis;
    basis.kind = VectorReferenceBasis::Kind::ComponentScalar;
    basis.dimension = dimension;
    basis.value_dimension = std::min(std::max(value_dimension, 1), 3);
    basis.element_family = family;
    basis.scalar = makeScalarReferenceBasis(element_type, order, include_bubble);
    return basis;
}

[[nodiscard]] VectorReferenceBasis makeHDivVectorBasis(ElementType element_type,
                                                       int dimension,
                                                       int order,
                                                       ElementFamily family)
{
    VectorReferenceBasis basis;
    basis.kind = VectorReferenceBasis::Kind::HDivVector;
    basis.dimension = dimension;
    basis.value_dimension = dimension;
    basis.element_family = family;
    if (family == ElementFamily::RaviartThomas) {
        basis.vector_basis =
            std::make_shared<basis::RaviartThomasBasis>(element_type, order);
    } else {
        basis.vector_basis =
            std::make_shared<basis::BDMBasis>(element_type, order);
    }
    return basis;
}

[[nodiscard]] VectorReferenceBasis makeTargetVectorBasis(
    const LocalProjectionBuildSpec& spec,
    ElementType element_type)
{
    const bool mini =
        spec.primal.element_family == ElementFamily::BubbleEnrichedLagrange ||
        (spec.primal.enrichment.visible_to_analysis &&
         spec.primal.enrichment.bubble_degree > 0);
    if (spec.primal.element_family == ElementFamily::RaviartThomas ||
        spec.primal.element_family == ElementFamily::BDM) {
        return makeHDivVectorBasis(element_type,
                                   spec.dimension,
                                   spec.primal.polynomial_order,
                                   spec.primal.element_family);
    }
    return makeComponentVectorBasis(element_type,
                                    spec.dimension,
                                    spec.value_dimension,
                                    spec.primal.polynomial_order,
                                    mini,
                                    spec.primal.element_family);
}

[[nodiscard]] VectorReferenceBasis makeSourceVectorBasis(
    const FortinCandidate& candidate,
    const LocalProjectionBuildSpec& spec,
    ElementType element_type)
{
    if (candidate.commuting_projection &&
        (spec.primal.element_family == ElementFamily::RaviartThomas ||
         spec.primal.element_family == ElementFamily::BDM)) {
        return makeHDivVectorBasis(element_type,
                                   spec.dimension,
                                   spec.primal.polynomial_order,
                                   spec.primal.element_family);
    }

    const int source_order = std::max(spec.primal.polynomial_order + 1,
                                      spec.primal.polynomial_order);
    return makeComponentVectorBasis(element_type,
                                    spec.dimension,
                                    spec.value_dimension,
                                    source_order,
                                    false,
                                    ElementFamily::Lagrange);
}

[[nodiscard]] int localProjectionQuadratureOrder(
    const VectorReferenceBasis& target,
    const VectorReferenceBasis& source,
    const ScalarReferenceBasis& multiplier,
    const LocalFortinProjectionOptions& options)
{
    if (options.quadrature_order > 0) {
        return options.quadrature_order;
    }
    const int max_order = std::max({target.polynomialOrder(),
                                    source.polynomialOrder(),
                                    multiplier.polynomialOrder()});
    return std::max(2, 2 * max_order + 3);
}

[[nodiscard]] Real dotVector(const math::Vector<Real, 3>& a,
                             const math::Vector<Real, 3>& b,
                             int dimension)
{
    Real value = Real(0);
    for (int d = 0; d < std::min(dimension, 3); ++d) {
        value += a[d] * b[d];
    }
    return value;
}

struct ProjectionAssemblies {
    DenseMatrix target_mass;
    DenseMatrix target_source_mass;
    DenseMatrix target_divergence_moments;
    DenseMatrix source_divergence_moments;
};

[[nodiscard]] ProjectionAssemblies assembleProjectionSystems(
    const VectorReferenceBasis& target,
    const VectorReferenceBasis& source,
    const ScalarReferenceBasis& multiplier,
    const quadrature::QuadratureRule& quadrature)
{
    ProjectionAssemblies assemblies{
        DenseMatrix(target.size(), target.size()),
        DenseMatrix(target.size(), source.size()),
        DenseMatrix(multiplier.size(), target.size()),
        DenseMatrix(multiplier.size(), source.size())};

    std::vector<math::Vector<Real, 3>> target_values;
    std::vector<math::Vector<Real, 3>> source_values;
    std::vector<Real> target_divergence;
    std::vector<Real> source_divergence;
    std::vector<Real> multiplier_values;

    for (std::size_t q = 0; q < quadrature.num_points(); ++q) {
        const auto xi = quadrature.point(q);
        const Real weight = quadrature.weight(q);
        target.evaluateValues(xi, target_values);
        source.evaluateValues(xi, source_values);
        target.evaluateDivergence(xi, target_divergence);
        source.evaluateDivergence(xi, source_divergence);
        multiplier.evaluateValues(xi, multiplier_values);

        for (int i = 0; i < target.size(); ++i) {
            for (int j = 0; j < target.size(); ++j) {
                assemblies.target_mass(i, j) +=
                    weight * dotVector(target_values[static_cast<std::size_t>(i)],
                                       target_values[static_cast<std::size_t>(j)],
                                       target.dimension);
            }
            for (int s = 0; s < source.size(); ++s) {
                assemblies.target_source_mass(i, s) +=
                    weight * dotVector(target_values[static_cast<std::size_t>(i)],
                                       source_values[static_cast<std::size_t>(s)],
                                       target.dimension);
            }
        }

        for (int m = 0; m < multiplier.size(); ++m) {
            const Real q_value = multiplier_values[static_cast<std::size_t>(m)];
            for (int i = 0; i < target.size(); ++i) {
                assemblies.target_divergence_moments(m, i) +=
                    weight * q_value *
                    target_divergence[static_cast<std::size_t>(i)];
            }
            for (int s = 0; s < source.size(); ++s) {
                assemblies.source_divergence_moments(m, s) +=
                    weight * q_value *
                    source_divergence[static_cast<std::size_t>(s)];
            }
        }
    }

    return assemblies;
}

[[nodiscard]] bool buildConstrainedProjectionMatrix(
    const ProjectionAssemblies& assemblies,
    DenseMatrix& projection,
    int& independent_constraint_count,
    std::string& diagnostic)
{
    const int target_dofs = assemblies.target_mass.rows;
    const int source_dofs = assemblies.target_source_mass.cols;
    const Real matrix_scale =
        std::max<Real>(Real(1), denseMaxAbs(assemblies.target_mass.values));
    const Real tolerance = Real(1e-12) * matrix_scale;

    const auto independent_rows = independentConstraintRows(
        assemblies.target_divergence_moments,
        Real(1e-11) *
            std::max<Real>(Real(1),
                           denseMaxAbs(assemblies.target_divergence_moments.values)));
    independent_constraint_count =
        static_cast<int>(independent_rows.size());
    const int system_size = target_dofs + independent_constraint_count;
    projection = DenseMatrix(target_dofs, source_dofs);

    for (int s = 0; s < source_dofs; ++s) {
        DenseMatrix system(system_size, system_size);
        std::vector<Real> rhs(static_cast<std::size_t>(system_size), Real(0));

        for (int i = 0; i < target_dofs; ++i) {
            for (int j = 0; j < target_dofs; ++j) {
                system(i, j) = assemblies.target_mass(i, j);
            }
            rhs[static_cast<std::size_t>(i)] =
                assemblies.target_source_mass(i, s);
        }

        for (int row = 0; row < independent_constraint_count; ++row) {
            const int moment_row = independent_rows[static_cast<std::size_t>(row)];
            const int kkt_row = target_dofs + row;
            for (int j = 0; j < target_dofs; ++j) {
                const Real value =
                    assemblies.target_divergence_moments(moment_row, j);
                system(j, kkt_row) = value;
                system(kkt_row, j) = value;
            }
            rhs[static_cast<std::size_t>(kkt_row)] =
                assemblies.source_divergence_moments(moment_row, s);
        }

        std::vector<Real> solution;
        if (!solveDenseSquare(system, rhs, solution, tolerance, diagnostic)) {
            return false;
        }
        for (int i = 0; i < target_dofs; ++i) {
            projection(i, s) = solution[static_cast<std::size_t>(i)];
        }
    }

    return true;
}

[[nodiscard]] DenseMatrix divergencePreservationResidual(
    const ProjectionAssemblies& assemblies,
    const DenseMatrix& projection)
{
    DenseMatrix residual(assemblies.target_divergence_moments.rows,
                         assemblies.source_divergence_moments.cols);
    for (int m = 0; m < residual.rows; ++m) {
        for (int s = 0; s < residual.cols; ++s) {
            Real projected = Real(0);
            for (int i = 0; i < projection.rows; ++i) {
                projected +=
                    assemblies.target_divergence_moments(m, i) *
                    projection(i, s);
            }
            residual(m, s) =
                projected - assemblies.source_divergence_moments(m, s);
        }
    }
    return residual;
}

[[nodiscard]] std::string localProjectionId(
    const FortinCandidate& candidate,
    const LocalProjectionBuildSpec& spec)
{
    std::ostringstream out;
    out << candidate.theorem.theorem_id
        << ":cell=" << toString(spec.reference_cell_family)
        << ":dim=" << spec.dimension
        << ":p=" << spec.primal.polynomial_order
        << ":q=" << spec.multiplier.polynomial_order;
    return out.str();
}

FortinProjectionPlan projectionPlanFor(const FortinTheoremEntry& theorem,
                                       const FieldDescriptor& primal,
                                       const FieldDescriptor& multiplier)
{
    FortinProjectionPlan plan;
    plan.local_operator_shape_metadata_present = true;
    plan.local_rows = std::max(0, multiplier.polynomial_order + 1);
    plan.local_cols = std::max(0, primal.polynomial_order + 1);

    if (theorem.evidence_kind ==
        FortinEvidenceKind::ExplicitFortinConstruction) {
        plan.projection_type = "bounded Fortin projection";
        plan.interpolation_or_projection =
            theorem.pair_family.find("MINI") != std::string::npos
                ? "constrained local projection with bubble correction"
                : "constrained local projection with divergence moment correction";
        plan.local_moment_constraints =
            "preserve divergence moments against the multiplier space";
        plan.correction_space =
            theorem.pair_family.find("MINI") != std::string::npos
                ? "element bubble enrichment"
                : "local H1 correction functions";
        plan.preserved_quantity = "discrete divergence pairing";
        plan.target_norm = "H1";
        return plan;
    }

    if (theorem.evidence_kind == FortinEvidenceKind::CommutingProjection) {
        plan.projection_type = "commuting projection";
        plan.interpolation_or_projection =
            "canonical H(div) interpolation/projection";
        plan.local_moment_constraints =
            "normal trace moments and interior vector moments";
        plan.correction_space = "cell moment space";
        plan.preserved_quantity = "divergence projection";
        plan.target_norm = "H(div)";
        return plan;
    }

    plan.projection_type = "stable-pair theorem";
    plan.interpolation_or_projection = "not constructive";
    plan.target_norm = "not applicable";
    plan.local_operator_shape_metadata_present = false;
    plan.local_rows = -1;
    plan.local_cols = -1;
    return plan;
}

} // namespace

const char* toString(MixedCouplingFamily family) noexcept
{
    switch (family) {
        case MixedCouplingFamily::VectorDivergenceScalarMultiplier:
            return "VectorDivergenceScalarMultiplier";
        case MixedCouplingFamily::GradientDivergenceAdjoint:
            return "GradientDivergenceAdjoint";
        case MixedCouplingFamily::MixedHDivDivergence:
            return "MixedHDivDivergence";
        case MixedCouplingFamily::TraceMultiplier:
            return "TraceMultiplier";
        case MixedCouplingFamily::MortarConstraint:
            return "MortarConstraint";
        case MixedCouplingFamily::CurlDivDeRham:
            return "CurlDivDeRham";
        case MixedCouplingFamily::GenericMultiplierConstraint:
            return "GenericMultiplierConstraint";
        case MixedCouplingFamily::Unknown:
            return "Unknown";
    }
    return "Unknown";
}

const char* toString(MixedCouplingEvidenceStrength strength) noexcept
{
    switch (strength) {
        case MixedCouplingEvidenceStrength::Strong: return "Strong";
        case MixedCouplingEvidenceStrength::Weak: return "Weak";
        case MixedCouplingEvidenceStrength::Ambiguous: return "Ambiguous";
        case MixedCouplingEvidenceStrength::None: return "None";
    }
    return "None";
}

const char* toString(FortinEvidenceKind kind) noexcept
{
    switch (kind) {
        case FortinEvidenceKind::KnownStablePair:
            return "KnownStablePair";
        case FortinEvidenceKind::ExplicitFortinConstruction:
            return "ExplicitFortinConstruction";
        case FortinEvidenceKind::CommutingProjection:
            return "CommutingProjection";
        case FortinEvidenceKind::NumericEstimateOnly:
            return "NumericEstimateOnly";
        case FortinEvidenceKind::StabilizedSurrogate:
            return "StabilizedSurrogate";
    }
    return "KnownStablePair";
}

const char* toString(FortinBoundAvailability availability) noexcept
{
    switch (availability) {
        case FortinBoundAvailability::Unavailable: return "Unavailable";
        case FortinBoundAvailability::Symbolic: return "Symbolic";
        case FortinBoundAvailability::Scoped: return "Scoped";
        case FortinBoundAvailability::Numeric: return "Numeric";
    }
    return "Unavailable";
}

const char* toString(FortinRejectionReason reason) noexcept
{
    switch (reason) {
        case FortinRejectionReason::UnsupportedPair:
            return "UnsupportedPair";
        case FortinRejectionReason::MissingMetadata:
            return "MissingMetadata";
        case FortinRejectionReason::WrongOrderRelation:
            return "WrongOrderRelation";
        case FortinRejectionReason::WrongMeshFamily:
            return "WrongMeshFamily";
        case FortinRejectionReason::MissingBoundaryNullspaceAssumption:
            return "MissingBoundaryNullspaceAssumption";
        case FortinRejectionReason::ContradictedAssumption:
            return "ContradictedAssumption";
        case FortinRejectionReason::StabilizedSurrogate:
            return "StabilizedSurrogate";
        case FortinRejectionReason::AmbiguousCoupling:
            return "AmbiguousCoupling";
    }
    return "UnsupportedPair";
}

const char* toString(FortinCandidateStatus status) noexcept
{
    switch (status) {
        case FortinCandidateStatus::Complete: return "Complete";
        case FortinCandidateStatus::Incomplete: return "Incomplete";
        case FortinCandidateStatus::Blocked: return "Blocked";
    }
    return "Incomplete";
}

const char* toString(LocalFortinProjectionStatus status) noexcept
{
    switch (status) {
        case LocalFortinProjectionStatus::NotRequested:
            return "NotRequested";
        case LocalFortinProjectionStatus::MetadataOnly:
            return "MetadataOnly";
        case LocalFortinProjectionStatus::Constructed:
            return "Constructed";
        case LocalFortinProjectionStatus::LocalDiagnosticPass:
            return "LocalDiagnosticPass";
        case LocalFortinProjectionStatus::Failed:
            return "Failed";
        case LocalFortinProjectionStatus::Unsupported:
            return "Unsupported";
    }
    return "NotRequested";
}

std::vector<MixedCouplingDescriptor>
MixedCouplingClassifier::classify(const ProblemAnalysisContext& context) const
{
    std::vector<MixedCouplingDescriptor> descriptors;

    for (const auto& contribution : context.contributions()) {
        bool emitted_from_pairing = false;
        for (const auto& pairing : contribution.pairings) {
            if (pairing.kind != PairingKind::ConstraintPair &&
                pairing.kind != PairingKind::FormalAdjointPair &&
                pairing.kind != PairingKind::StabilizedConstraintPair) {
                continue;
            }
            mergeDescriptor(
                descriptors,
                descriptorFromVariables(context,
                                        contribution,
                                        pairing.row_var,
                                        pairing.col_var,
                                        pairing.kind,
                                        pairing.pairing_group,
                                        pairing.has_stabilizing_surrogate));
            emitted_from_pairing = true;
        }

        if (emitted_from_pairing) {
            continue;
        }

        if ((contribution.role == ContributionRole::ConstraintBlock ||
             contribution.role == ContributionRole::OffDiagonalBlock ||
             contribution.role == ContributionRole::BoundaryConstraint ||
             contribution.role == ContributionRole::InterfaceCoupling) &&
            !contribution.test_variables.empty() &&
            !contribution.trial_variables.empty()) {
            mergeDescriptor(
                descriptors,
                descriptorFromVariables(context,
                                        contribution,
                                        contribution.test_variables.front(),
                                        contribution.trial_variables.front(),
                                        PairingKind::Unknown,
                                        {},
                                        false));
        }
    }

    return descriptors;
}

FortinTheoremRegistry::FortinTheoremRegistry()
{
    FortinTheoremEntry taylor_hood;
    taylor_hood.theorem_id = "fortin:taylor-hood-p2-p1-simplex";
    taylor_hood.pair_family = "Taylor-Hood P2/P1";
    taylor_hood.coupling_family = MixedCouplingFamily::VectorDivergenceScalarMultiplier;
    taylor_hood.primal_space.space_families = {SpaceFamily::H1};
    taylor_hood.primal_space.element_families = {ElementFamily::Lagrange};
    taylor_hood.primal_space.continuity_class =
        SpaceContinuityClass::Continuous;
    taylor_hood.primal_space.mapping_transform = MappingTransform::Identity;
    taylor_hood.primal_space.minimum_order = 2;
    taylor_hood.primal_space.maximum_order = 2;
    taylor_hood.multiplier_space.space_families = {SpaceFamily::H1};
    taylor_hood.multiplier_space.element_families = {ElementFamily::Lagrange};
    taylor_hood.multiplier_space.continuity_class =
        SpaceContinuityClass::Continuous;
    taylor_hood.multiplier_space.mapping_transform =
        MappingTransform::Identity;
    taylor_hood.multiplier_space.minimum_order = 1;
    taylor_hood.multiplier_space.maximum_order = 1;
    taylor_hood.order_relation =
        FortinPolynomialOrderRelation::PrimalOneHigher;
    taylor_hood.supported_dimensions = {2, 3};
    taylor_hood.supported_cell_families = {ReferenceCellFamily::Simplex};
    taylor_hood.mesh_assumption = "shape-regular simplex mesh";
    taylor_hood.domain_assumption = "bounded Lipschitz domain";
    taylor_hood.boundary_nullspace_assumption =
        "primal trace boundary scope and scalar multiplier gauge/mean-zero handling";
    taylor_hood.evidence_kind =
        FortinEvidenceKind::ExplicitFortinConstruction;
    taylor_hood.beta_bound = FortinBoundAvailability::Symbolic;
    taylor_hood.fortin_norm_bound = FortinBoundAvailability::Symbolic;
    entries_.push_back(std::move(taylor_hood));

    FortinTheoremEntry mini;
    mini.theorem_id = "fortin:mini-p1bubble-p1-simplex";
    mini.pair_family = "MINI P1+bubble/P1";
    mini.coupling_family = MixedCouplingFamily::VectorDivergenceScalarMultiplier;
    mini.primal_space.space_families = {SpaceFamily::H1};
    mini.primal_space.element_families =
        {ElementFamily::BubbleEnrichedLagrange};
    mini.primal_space.continuity_class = SpaceContinuityClass::Continuous;
    mini.primal_space.mapping_transform = MappingTransform::Identity;
    mini.primal_space.minimum_order = 1;
    mini.primal_space.maximum_order = 1;
    mini.multiplier_space.space_families = {SpaceFamily::H1};
    mini.multiplier_space.element_families = {ElementFamily::Lagrange};
    mini.multiplier_space.continuity_class = SpaceContinuityClass::Continuous;
    mini.multiplier_space.mapping_transform = MappingTransform::Identity;
    mini.multiplier_space.minimum_order = 1;
    mini.multiplier_space.maximum_order = 1;
    mini.order_relation = FortinPolynomialOrderRelation::MiniP1P1;
    mini.supported_dimensions = {2, 3};
    mini.supported_cell_families = {ReferenceCellFamily::Simplex};
    mini.mesh_assumption = "shape-regular simplex mesh";
    mini.domain_assumption = "bounded Lipschitz domain";
    mini.boundary_nullspace_assumption =
        "primal trace boundary scope and scalar multiplier gauge/mean-zero handling";
    mini.evidence_kind = FortinEvidenceKind::ExplicitFortinConstruction;
    mini.beta_bound = FortinBoundAvailability::Symbolic;
    mini.fortin_norm_bound = FortinBoundAvailability::Symbolic;
    entries_.push_back(std::move(mini));

    FortinTheoremEntry rt;
    rt.theorem_id = "fortin:rtk-dgk-hdiv-divergence";
    rt.pair_family = "RT_k/DG_k";
    rt.coupling_family = MixedCouplingFamily::MixedHDivDivergence;
    rt.primal_space.space_families = {SpaceFamily::HDiv};
    rt.primal_space.element_families = {ElementFamily::RaviartThomas};
    rt.primal_space.continuity_class =
        SpaceContinuityClass::NormalContinuous;
    rt.primal_space.mapping_transform =
        MappingTransform::ContravariantPiola;
    rt.multiplier_space.space_families = {SpaceFamily::L2, SpaceFamily::DG};
    rt.multiplier_space.element_families = {ElementFamily::DG};
    rt.multiplier_space.continuity_class =
        SpaceContinuityClass::Discontinuous;
    rt.multiplier_space.mapping_transform = MappingTransform::Identity;
    rt.order_relation = FortinPolynomialOrderRelation::Equal;
    rt.supported_dimensions = {2, 3};
    rt.supported_cell_families = {ReferenceCellFamily::Simplex,
                                  ReferenceCellFamily::TensorProduct};
    rt.mesh_assumption = "shape-regular affine mesh";
    rt.domain_assumption = "bounded Lipschitz domain";
    rt.boundary_nullspace_assumption =
        "compatible divergence range and multiplier gauge when boundary flux is constrained";
    rt.evidence_kind = FortinEvidenceKind::CommutingProjection;
    rt.beta_bound = FortinBoundAvailability::Symbolic;
    rt.fortin_norm_bound = FortinBoundAvailability::Symbolic;
    entries_.push_back(std::move(rt));

    FortinTheoremEntry bdm;
    bdm.theorem_id = "fortin:bdmk-dgkminus1-hdiv-divergence";
    bdm.pair_family = "BDM_k/DG_{k-1}";
    bdm.coupling_family = MixedCouplingFamily::MixedHDivDivergence;
    bdm.primal_space.space_families = {SpaceFamily::HDiv};
    bdm.primal_space.element_families = {ElementFamily::BDM};
    bdm.primal_space.continuity_class =
        SpaceContinuityClass::NormalContinuous;
    bdm.primal_space.mapping_transform =
        MappingTransform::ContravariantPiola;
    bdm.primal_space.minimum_order = 1;
    bdm.multiplier_space.space_families = {SpaceFamily::L2, SpaceFamily::DG};
    bdm.multiplier_space.element_families = {ElementFamily::DG};
    bdm.multiplier_space.continuity_class =
        SpaceContinuityClass::Discontinuous;
    bdm.multiplier_space.mapping_transform = MappingTransform::Identity;
    bdm.order_relation = FortinPolynomialOrderRelation::PrimalOneHigher;
    bdm.supported_dimensions = {2, 3};
    bdm.supported_cell_families = {ReferenceCellFamily::Simplex,
                                   ReferenceCellFamily::TensorProduct};
    bdm.mesh_assumption = "shape-regular affine mesh";
    bdm.domain_assumption = "bounded Lipschitz domain";
    bdm.boundary_nullspace_assumption =
        "compatible divergence range and multiplier gauge when boundary flux is constrained";
    bdm.evidence_kind = FortinEvidenceKind::CommutingProjection;
    bdm.beta_bound = FortinBoundAvailability::Symbolic;
    bdm.fortin_norm_bound = FortinBoundAvailability::Symbolic;
    entries_.push_back(std::move(bdm));
}

FortinRegistryMatch FortinTheoremRegistry::match(
    const MixedCouplingDescriptor& coupling,
    const FieldDescriptor& primal,
    const FieldDescriptor& multiplier) const
{
    FortinRegistryMatch aggregate_miss;
    bool saw_coupling_family = false;

    for (const auto& entry : entries_) {
        if (!couplingCompatible(coupling.coupling_family,
                                entry.coupling_family)) {
            continue;
        }
        saw_coupling_family = true;

        FortinRegistryMatch local;
        local.entry = &entry;
        if (coupling.has_stabilization_surrogate) {
            addReason(local, FortinRejectionReason::StabilizedSurrogate);
            local.diagnostics.push_back(
                "stabilized surrogate pairs route to StabilizationAdequacy");
        }
        if (coupling.evidence_strength ==
                MixedCouplingEvidenceStrength::Ambiguous ||
            coupling.evidence_strength == MixedCouplingEvidenceStrength::None) {
            addReason(local, FortinRejectionReason::AmbiguousCoupling);
        }
        if (!metadataKnownForRequirement(primal, entry.primal_space) ||
            !metadataKnownForRequirement(multiplier, entry.multiplier_space)) {
            addReason(local, FortinRejectionReason::MissingMetadata);
        }
        if (!spaceRequirementSatisfied(primal, entry.primal_space) ||
            !spaceRequirementSatisfied(multiplier, entry.multiplier_space)) {
            addReason(local, FortinRejectionReason::UnsupportedPair);
        }
        if (!contains(entry.supported_dimensions,
                      primal.topological_dimension) ||
            primal.topological_dimension != multiplier.topological_dimension) {
            addReason(local, FortinRejectionReason::MissingMetadata);
            local.diagnostics.push_back(
                "dimension metadata is missing or unsupported");
        }
        if (!contains(entry.supported_cell_families,
                      primal.reference_cell_family) ||
            primal.reference_cell_family != multiplier.reference_cell_family) {
            addReason(local, FortinRejectionReason::WrongMeshFamily);
        }
        if (!orderRelationSatisfied(primal,
                                    multiplier,
                                    entry.order_relation)) {
            addReason(local, FortinRejectionReason::WrongOrderRelation);
        }

        if (local.rejection_reasons.empty()) {
            local.matched = true;
            local.diagnostics.push_back("matched theorem " + entry.theorem_id);
            return local;
        }

        for (auto reason : local.rejection_reasons) {
            addReason(aggregate_miss, reason);
        }
        for (const auto& diagnostic : local.diagnostics) {
            appendUnique(aggregate_miss.diagnostics, diagnostic);
        }
    }

    if (!saw_coupling_family) {
        addReason(aggregate_miss, FortinRejectionReason::UnsupportedPair);
        aggregate_miss.diagnostics.push_back(
            "no Fortin theorem entry supports coupling family " +
            std::string(toString(coupling.coupling_family)));
    }
    return aggregate_miss;
}

FortinCandidateBuilder::FortinCandidateBuilder() = default;

FortinCandidateBuildResult FortinCandidateBuilder::build(
    const ProblemAnalysisContext& context) const
{
    FortinCandidateBuildResult result;
    const auto couplings = coupling_classifier_.classify(context);
    if (couplings.empty()) {
        result.diagnostics.push_back(
            "Fortin candidate builder found no mixed coupling descriptors");
        return result;
    }

    for (const auto& coupling : couplings) {
        FortinCandidate candidate;
        candidate.coupling = coupling;
        candidate.status = FortinCandidateStatus::Incomplete;
        candidate.diagnostics = coupling.diagnostics;

        if (coupling.primal_variable.kind != VariableKind::FieldComponent ||
            coupling.multiplier_variable.kind != VariableKind::FieldComponent) {
            candidate.status = FortinCandidateStatus::Blocked;
            addCandidateReason(candidate, FortinRejectionReason::MissingMetadata);
            candidate.diagnostics.push_back(
                "Fortin registry requires FE field variables");
            result.candidates.push_back(std::move(candidate));
            continue;
        }

        const auto* primal =
            context.fieldDescriptor(coupling.primal_variable.field_id);
        const auto* multiplier =
            context.fieldDescriptor(coupling.multiplier_variable.field_id);
        if (!primal || !multiplier) {
            candidate.status = FortinCandidateStatus::Blocked;
            addCandidateReason(candidate, FortinRejectionReason::MissingMetadata);
            candidate.diagnostics.push_back(
                "missing FieldDescriptor for primal or multiplier variable");
            result.candidates.push_back(std::move(candidate));
            continue;
        }

        const auto match = registry_.match(coupling, *primal, *multiplier);
        if (!match.matched || !match.entry) {
            candidate.status = coupling.has_stabilization_surrogate
                ? FortinCandidateStatus::Blocked
                : FortinCandidateStatus::Incomplete;
            for (auto reason : match.rejection_reasons) {
                addCandidateReason(candidate, reason);
            }
            for (const auto& diagnostic : match.diagnostics) {
                appendUnique(candidate.diagnostics, diagnostic);
            }
            result.candidates.push_back(std::move(candidate));
            continue;
        }

        candidate.theorem = *match.entry;
        candidate.stable_pair_only =
            match.entry->evidence_kind == FortinEvidenceKind::KnownStablePair;
        candidate.constructive_fortin =
            match.entry->evidence_kind ==
            FortinEvidenceKind::ExplicitFortinConstruction;
        candidate.commuting_projection =
            match.entry->evidence_kind == FortinEvidenceKind::CommutingProjection;
        candidate.global_constraint_handling =
            globalConstraintHandling(*multiplier);
        candidate.mesh_assumption_satisfied =
            meshAssumptionSatisfied(context, *primal, *multiplier);
        candidate.domain_assumption_satisfied =
            domainAssumptionSatisfied(*primal, *multiplier);
        candidate.boundary_nullspace_assumption_satisfied =
            boundaryNullspaceAssumptionSatisfied(coupling,
                                                 *primal,
                                                 *multiplier);

        if (!candidate.mesh_assumption_satisfied) {
            addCandidateReason(candidate, FortinRejectionReason::WrongMeshFamily);
            candidate.diagnostics.push_back(
                "missing shape-regular mesh evidence for " +
                match.entry->mesh_assumption);
        }
        if (!candidate.domain_assumption_satisfied) {
            addCandidateReason(candidate, FortinRejectionReason::MissingMetadata);
            candidate.diagnostics.push_back(
                "missing domain assumption evidence for " +
                match.entry->domain_assumption);
        }
        if (!candidate.boundary_nullspace_assumption_satisfied) {
            addCandidateReason(
                candidate,
                FortinRejectionReason::MissingBoundaryNullspaceAssumption);
            candidate.diagnostics.push_back(
                "missing boundary/nullspace assumption evidence: " +
                match.entry->boundary_nullspace_assumption);
        }

        if (candidate.constructive_fortin || candidate.commuting_projection) {
            candidate.projection_plan =
                projectionPlanFor(*match.entry, *primal, *multiplier);
        }

        candidate.status = candidate.missing_or_rejected_reasons.empty()
            ? FortinCandidateStatus::Complete
            : FortinCandidateStatus::Incomplete;
        candidate.diagnostics.push_back(
            "Fortin candidate " +
            std::string(toString(candidate.status)) +
            " for theorem " + match.entry->theorem_id);
        result.candidates.push_back(std::move(candidate));
    }

    return result;
}

LocalFortinProjectionResult LocalFortinProjectionBuilder::build(
    const FortinCandidate& candidate,
    const LocalFortinProjectionOptions& options) const
{
    return build(candidate, static_cast<const ProblemAnalysisContext*>(nullptr),
                 options);
}

LocalFortinProjectionResult LocalFortinProjectionBuilder::build(
    const FortinCandidate& candidate,
    const ProblemAnalysisContext& context,
    const LocalFortinProjectionOptions& options) const
{
    return build(candidate, &context, options);
}

LocalFortinProjectionResult LocalFortinProjectionBuilder::build(
    const FortinCandidate& candidate,
    const ProblemAnalysisContext* context,
    const LocalFortinProjectionOptions& options) const
{
    LocalFortinProjectionResult result;

    if (!candidate.projection_plan) {
        result.status = LocalFortinProjectionStatus::Unsupported;
        result.diagnostics.push_back(
            "candidate has no constructive or commuting projection plan");
        return result;
    }

    result.plan = *candidate.projection_plan;
    result.local_operator_shape_metadata_present =
        result.plan.local_operator_shape_metadata_present;
    result.divergence_preservation_metadata_present =
        result.plan.preserved_quantity.find("divergence") != std::string::npos;
    result.trace_preservation_metadata_present =
        result.plan.local_moment_constraints.find("normal trace") !=
        std::string::npos;
    result.commuting_projection_metadata_present =
        candidate.commuting_projection;

    const auto spec = makeLocalProjectionBuildSpec(candidate, context);
    result.reference_projection_id = localProjectionId(candidate, spec);
    result.reference_cell_family = spec.reference_cell_family;
    result.topological_dimension = spec.dimension;

    if (!options.build_local_projection_matrices &&
        !options.verify_preservation_identities &&
        !options.estimate_norm_bound) {
        result.status = LocalFortinProjectionStatus::MetadataOnly;
        result.diagnostics.push_back(
            "local Fortin construction returned metadata only");
        return result;
    }

    if (candidate.status != FortinCandidateStatus::Complete) {
        result.status = LocalFortinProjectionStatus::Failed;
        result.diagnostics.push_back(
            "local projection construction requires a complete theorem candidate");
        return result;
    }

    if (candidate.coupling.coupling_family == MixedCouplingFamily::TraceMultiplier ||
        candidate.coupling.coupling_family == MixedCouplingFamily::MortarConstraint) {
        result.status = LocalFortinProjectionStatus::Unsupported;
        result.diagnostics.push_back(
            "trace and mortar Fortin projection construction requires complete trace DOF metadata");
        return result;
    }

    const ElementType element_type =
        canonicalElementFor(spec.reference_cell_family, spec.dimension);
    if (element_type == ElementType::Unknown ||
        spec.dimension <= 0 ||
        spec.dimension > 3) {
        result.status = LocalFortinProjectionStatus::Unsupported;
        result.diagnostics.push_back(
            "reference cell family or dimension is not supported for local Fortin construction");
        return result;
    }

    try {
        const auto target = makeTargetVectorBasis(spec, element_type);
        const auto source = makeSourceVectorBasis(candidate, spec, element_type);
        const auto multiplier = makeScalarReferenceBasis(
            element_type, spec.multiplier.polynomial_order, false);
        const int quadrature_order =
            localProjectionQuadratureOrder(target, source, multiplier, options);
        const auto quadrature = quadrature::QuadratureFactory::create(
            element_type, quadrature_order, QuadratureType::GaussLegendre, true);

        result.target_dof_count = target.size();
        result.source_dof_count = source.size();
        result.multiplier_dof_count = multiplier.size();
        result.source_polynomial_order = source.polynomialOrder();
        result.quadrature_order = quadrature_order;
        result.plan.local_rows = result.target_dof_count;
        result.plan.local_cols = result.source_dof_count;

        const auto assemblies =
            assembleProjectionSystems(target, source, multiplier, *quadrature);
        DenseMatrix projection;
        int independent_constraints = 0;
        std::string diagnostic;
        if (!buildConstrainedProjectionMatrix(assemblies,
                                              projection,
                                              independent_constraints,
                                              diagnostic)) {
            result.status = LocalFortinProjectionStatus::Failed;
            result.diagnostics.push_back(
                "local constrained Fortin projection solve failed: " +
                diagnostic);
            return result;
        }

        result.independent_moment_constraint_count =
            independent_constraints;
        result.local_projection_matrix_present = true;
        result.local_operator_shape_metadata_present = true;
        LocalFortinMatrix matrix;
        matrix.matrix_id = result.reference_projection_id;
        matrix.rows = projection.rows;
        matrix.cols = projection.cols;
        matrix.row_major_values = projection.values;
        result.local_matrices.push_back(std::move(matrix));

        const auto residual =
            divergencePreservationResidual(assemblies, projection);
        result.preservation_residual_frobenius_norm =
            denseFrobeniusNorm(residual);
        result.preservation_residual_max_abs = denseMaxAbs(residual);
        const Real source_scale =
            std::max<Real>(Real(1),
                           denseMaxAbs(assemblies.source_divergence_moments.values));
        result.preservation_residual_tolerance = Real(1e-8) * source_scale;

        if (options.verify_preservation_identities) {
            result.preservation_identity_verified =
                result.preservation_residual_max_abs <=
                result.preservation_residual_tolerance;
            if (!result.preservation_identity_verified) {
                result.status = LocalFortinProjectionStatus::Failed;
                result.diagnostics.push_back(
                    "reference-element divergence preservation residual exceeded tolerance");
                return result;
            }
        }

        if (options.estimate_norm_bound) {
            result.norm_bound_estimate_present = true;
            result.norm_bound_estimate = denseFrobeniusNorm(projection);
            result.norm_bound_estimate_kind =
                "reference projection Frobenius diagnostic (not a global Fortin norm bound)";
        }

        result.status = options.verify_preservation_identities
            ? LocalFortinProjectionStatus::LocalDiagnosticPass
            : LocalFortinProjectionStatus::Constructed;
        result.diagnostics.push_back(
            "reference-element constrained projection matrix constructed");
        if (options.verify_preservation_identities) {
            result.diagnostics.push_back(
                "divergence moment preservation verified by quadrature on the reference element");
        }
    } catch (const std::exception& ex) {
        result.status = LocalFortinProjectionStatus::Failed;
        result.diagnostics.push_back(
            std::string("local Fortin projection construction failed: ") +
            ex.what());
    }

    return result;
}

namespace {

void populateSummaryFieldMetadata(const ProblemAnalysisContext& context,
                                  InfSupPairCertificationSummary& summary)
{
    const auto* primal =
        summary.primal_variable.kind == VariableKind::FieldComponent
            ? context.fieldDescriptor(summary.primal_variable.field_id)
            : nullptr;
    const auto* multiplier =
        summary.multiplier_variable.kind == VariableKind::FieldComponent
            ? context.fieldDescriptor(summary.multiplier_variable.field_id)
            : nullptr;
    if (primal) {
        summary.primal_polynomial_order = primal->polynomial_order;
        summary.primal_space_family = primal->space_family;
        summary.primal_element_family = primal->element_family;
        summary.reference_cell_family = primal->reference_cell_family;
    }
    if (multiplier) {
        summary.multiplier_polynomial_order = multiplier->polynomial_order;
        summary.multiplier_space_family = multiplier->space_family;
        summary.multiplier_element_family = multiplier->element_family;
        if (summary.reference_cell_family == ReferenceCellFamily::Unknown) {
            summary.reference_cell_family = multiplier->reference_cell_family;
        }
    }
}

} // namespace

std::optional<InfSupPairCertificationSummary>
makeInfSupPairCertificationSummary(const FortinCandidate& candidate)
{
    if (candidate.status != FortinCandidateStatus::Complete) {
        return std::nullopt;
    }

    InfSupPairCertificationSummary summary;
    summary.primal_variable = candidate.coupling.primal_variable;
    summary.multiplier_variable = candidate.coupling.multiplier_variable;
    summary.pair_family = candidate.theorem.pair_family;
    summary.inf_sup_theorem_id = candidate.theorem.theorem_id;
    summary.coupling_family = toString(candidate.coupling.coupling_family);
    summary.theorem_evidence_kind = toString(candidate.theorem.evidence_kind);
    summary.mesh_assumption_scope = candidate.theorem.mesh_assumption;
    summary.domain_assumption_scope = candidate.theorem.domain_assumption;
    summary.boundary_nullspace_scope =
        candidate.theorem.boundary_nullspace_assumption;
    summary.global_constraint_handling = candidate.global_constraint_handling;
    summary.block.domain = candidate.coupling.domain;
    summary.block.role = ContributionRole::ConstraintBlock;
    summary.block.operator_tag = candidate.coupling.operator_tag;
    summary.block.test_variables = {summary.primal_variable,
                                    summary.multiplier_variable};
    summary.block.trial_variables = summary.block.test_variables;
    summary.block.marker = candidate.coupling.domain == DomainKind::Boundary
        ? candidate.coupling.boundary_marker
        : candidate.coupling.interface_marker;
    if (!candidate.coupling.contribution_ids.empty()) {
        summary.block.contribution_id =
            candidate.coupling.contribution_ids.front();
    }

    summary.known_stable_pair =
        candidate.theorem.evidence_kind !=
            FortinEvidenceKind::NumericEstimateOnly &&
        candidate.theorem.evidence_kind !=
            FortinEvidenceKind::StabilizedSurrogate;
    summary.fortin_operator_evidence_present =
        candidate.constructive_fortin || candidate.commuting_projection;
    summary.mesh_assumption_evidence_present =
        candidate.mesh_assumption_satisfied;
    summary.domain_assumption_evidence_present =
        candidate.domain_assumption_satisfied;
    summary.boundary_condition_scope_present =
        candidate.boundary_nullspace_assumption_satisfied;

    if (candidate.theorem.beta_bound == FortinBoundAvailability::Numeric) {
        summary.beta_lower_bound_present = true;
        summary.beta_lower_bound = candidate.theorem.beta_lower_bound;
    } else if (candidate.theorem.beta_bound !=
               FortinBoundAvailability::Unavailable) {
        summary.beta_lower_bound_symbolic_present = true;
    }

    if (candidate.theorem.fortin_norm_bound ==
        FortinBoundAvailability::Numeric) {
        summary.fortin_operator_norm_bound_present = true;
        summary.fortin_operator_norm_bound =
            candidate.theorem.fortin_norm_bound_value;
    } else if (candidate.theorem.fortin_norm_bound !=
               FortinBoundAvailability::Unavailable) {
        summary.fortin_operator_norm_bound_symbolic_present = true;
    }

    if (candidate.projection_plan) {
        summary.projection_plan_present = true;
        summary.projection_plan_id = candidate.projection_plan->projection_type;
        summary.projection_preserved_quantity =
            candidate.projection_plan->preserved_quantity;
        summary.projection_target_norm = candidate.projection_plan->target_norm;
    }

    summary.diagnostics = candidate.diagnostics;
    return summary;
}

std::optional<InfSupPairCertificationSummary>
makeInfSupPairCertificationSummary(const FortinCandidate& candidate,
                                   const ProblemAnalysisContext& context)
{
    auto summary = makeInfSupPairCertificationSummary(candidate);
    if (summary) {
        populateSummaryFieldMetadata(context, *summary);
    }
    return summary;
}

namespace {

void emitCertifiedFortinClaim(const ProblemAnalysisContext& context,
                              ProblemAnalysisReport& report,
                              InfSupPairCertificationSummary summary)
{
    populateSummaryFieldMetadata(context, summary);

    PropertyClaim claim;
    claim.kind = PropertyKind::InfSupCondition;
    claim.status = PropertyStatus::Preserved;
    claim.confidence = AnalysisConfidence::High;
    claim.inf_sup_class = InfSupClass::StructurallySupported;
    claim.certification_class = CertificationClass::Certified;
    claim.domain = summary.block.domain;
    claim.variables.push_back(summary.primal_variable);
    claim.variables.push_back(summary.multiplier_variable);
    claim.tested_block_id = summary.block.operator_tag;
    claim.estimate_scope = summary.pair_family;
    claim.claim_origin = "FortinCertificationAnalyzer";
    claim.description =
        "Inf-sup condition certified by FE Fortin/stable-pair theorem registry";
    claim.addEvidence(
        "FortinCertificationAnalyzer",
        "theorem='" + summary.inf_sup_theorem_id +
            "', pair_family='" + summary.pair_family +
            "', coupling='" + summary.coupling_family +
            "', evidence_kind='" + summary.theorem_evidence_kind +
            "', mesh_assumption='" + summary.mesh_assumption_scope +
            "', domain_assumption='" + summary.domain_assumption_scope +
            "', boundary_nullspace='" + summary.boundary_nullspace_scope +
            "', global_constraint='" + summary.global_constraint_handling +
            "', beta_symbolic=" +
            std::string(summary.beta_lower_bound_symbolic_present ? "true" : "false") +
            ", fortin_norm_symbolic=" +
            std::string(summary.fortin_operator_norm_bound_symbolic_present ? "true" : "false"),
        AnalysisConfidence::High);
    report.claims.push_back(std::move(claim));
}

} // namespace

std::string FortinCertificationAnalyzer::name() const
{
    return "FortinCertificationAnalyzer";
}

void FortinCertificationAnalyzer::run(const ProblemAnalysisContext& context,
                                      ProblemAnalysisReport& report) const
{
    AnalyzerRunLogSummary run_log;
    run_log.analyzer = name();
    run_log.summary_id = "FortinOperatorAutogeneration";

    if (!report.request_plan.has(AnalysisSummaryKind::InfSupPairCertification)) {
        run_log.status = "skipped";
        run_log.skipped_count = 1;
        run_log.message =
            "InfSupPairCertification was not requested by the planner";
        report.run_logs.push_back(std::move(run_log));
        return;
    }

    FortinCandidateBuilder builder;
    const auto build = builder.build(context);
    run_log.attempted_count = build.candidates.size();
    for (const auto& diagnostic : build.diagnostics) {
        appendUnique(run_log.diagnostics, diagnostic);
    }

    if (build.candidates.empty()) {
        AnalysisIssue issue;
        issue.severity = IssueSeverity::Info;
        issue.message =
            "FortinCertificationAnalyzer: no Fortin candidates were found";
        report.issues.push_back(std::move(issue));
        run_log.status = "unsupported";
        run_log.unsupported_count = 1;
        run_log.message =
            "no Fortin candidates were found for requested certification";
        report.run_logs.push_back(std::move(run_log));
        return;
    }

    for (const auto& candidate : build.candidates) {
        std::string detail =
            "candidate " +
            variableShortLabel(candidate.coupling.primal_variable) +
            "/" +
            variableShortLabel(candidate.coupling.multiplier_variable) +
            " status=" + toString(candidate.status) +
            " coupling=" + toString(candidate.coupling.coupling_family);
        if (!candidate.theorem.theorem_id.empty()) {
            detail += " theorem=" + candidate.theorem.theorem_id;
        }
        if (!candidate.missing_or_rejected_reasons.empty()) {
            detail += " reasons=" +
                      reasonsLabel(candidate.missing_or_rejected_reasons);
        }
        run_log.detail_lines.push_back(detail);
        for (const auto& diagnostic : candidate.diagnostics) {
            appendUnique(run_log.diagnostics, diagnostic);
        }

        if (candidate.status == FortinCandidateStatus::Complete) {
            if (reportAlreadyHasCertifiedPair(
                    report,
                    candidate.coupling.primal_variable,
                    candidate.coupling.multiplier_variable)) {
                ++run_log.skipped_count;
                run_log.detail_lines.push_back(
                    "skipped duplicate certified pair for " +
                    variableShortLabel(candidate.coupling.primal_variable) +
                    "/" +
                    variableShortLabel(candidate.coupling.multiplier_variable));
                continue;
            }
            auto summary = makeInfSupPairCertificationSummary(candidate, context);
            if (summary) {
                emitCertifiedFortinClaim(context, report, std::move(*summary));
                ++run_log.certified_count;
            } else {
                ++run_log.incomplete_count;
                run_log.diagnostics.push_back(
                    "complete candidate could not be converted to InfSupPairCertificationSummary");
            }
            continue;
        }

        if (candidate.status == FortinCandidateStatus::Blocked) {
            ++run_log.blocked_count;
        } else if (contains(candidate.missing_or_rejected_reasons,
                           FortinRejectionReason::UnsupportedPair)) {
            ++run_log.unsupported_count;
        } else {
            ++run_log.incomplete_count;
        }

        AnalysisIssue issue;
        issue.severity = IssueSeverity::Info;
        issue.message =
            "FortinCertificationAnalyzer: " +
            std::string(toString(candidate.status)) +
            " candidate for " +
            variableShortLabel(candidate.coupling.primal_variable) +
            " / " +
            variableShortLabel(candidate.coupling.multiplier_variable) +
            " coupling=" + toString(candidate.coupling.coupling_family) +
            " reasons=" + reasonsLabel(candidate.missing_or_rejected_reasons);
        if (!candidate.diagnostics.empty()) {
            issue.message += " diagnostic=" + candidate.diagnostics.front();
        }
        report.issues.push_back(std::move(issue));
    }

    if (run_log.certified_count > 0) {
        run_log.status = run_log.incomplete_count == 0 &&
                                 run_log.blocked_count == 0 &&
                                 run_log.unsupported_count == 0
                             ? "certified"
                             : "partial";
    } else if (run_log.blocked_count > 0) {
        run_log.status = "blocked";
    } else if (run_log.incomplete_count > 0) {
        run_log.status = "incomplete";
    } else if (run_log.unsupported_count > 0) {
        run_log.status = "unsupported";
    } else {
        run_log.status = "skipped";
    }
    run_log.message =
        "Fortin theorem matching and local certification diagnostics completed";
    report.run_logs.push_back(std::move(run_log));
}

} // namespace analysis
} // namespace FE
} // namespace svmp
