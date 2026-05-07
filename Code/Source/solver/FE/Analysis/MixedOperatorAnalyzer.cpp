/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/MixedOperatorAnalyzer.h"
#include "Analysis/AnalysisNumericGuards.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/FormStructureAnalyzer.h"
#include "Forms/FormExpr.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

void appendUnique(std::vector<VariableKey>& values, const VariableKey& value)
{
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

std::vector<VariableKey> blockVariables(const OperatorBlockId& block)
{
    std::vector<VariableKey> variables;
    for (const auto& v : block.test_variables) {
        appendUnique(variables, v);
    }
    for (const auto& v : block.trial_variables) {
        appendUnique(variables, v);
    }
    return variables;
}

struct LocalDeclaredNullspaceMetadata {
    bool present{false};
    NullspaceFamily family{NullspaceFamily::UserDefined};
    AnalysisConfidence confidence{AnalysisConfidence::Medium};
    NullspaceEvidenceKind evidence_kind{NullspaceEvidenceKind::DescriptorHint};
    std::string evidence;
};

bool variableIsField(const VariableKey& variable, FieldId field) noexcept
{
    return variable.kind == VariableKind::FieldComponent &&
           variable.field_id == field;
}

bool variablesContainField(const std::vector<VariableKey>& variables,
                           FieldId field)
{
    return std::any_of(variables.begin(), variables.end(),
                       [field](const VariableKey& variable) {
                           return variableIsField(variable, field);
                       });
}

bool contributionHasTestField(const ContributionDescriptor& contribution,
                              FieldId field)
{
    return variablesContainField(contribution.test_variables, field);
}

bool contributionHasTrialField(const ContributionDescriptor& contribution,
                               FieldId field)
{
    return variablesContainField(contribution.trial_variables, field);
}

bool isScalarFieldDescriptor(const ProblemAnalysisContext& context,
                             FieldId field)
{
    const auto* descriptor = context.fieldDescriptor(field);
    return descriptor &&
           descriptor->field_type == FieldType::Scalar &&
           descriptor->value_dimension <= 1;
}

bool isVectorFieldDescriptor(const ProblemAnalysisContext& context,
                             FieldId field)
{
    const auto* descriptor = context.fieldDescriptor(field);
    return descriptor &&
           (descriptor->field_type == FieldType::Vector ||
            descriptor->value_dimension > 1);
}

bool isZeroLikeExpression(const forms::FormExprNode& node)
{
    using forms::FormExprType;

    if (node.type() == FormExprType::TypedZero) {
        return true;
    }
    if (node.type() == FormExprType::Constant) {
        const auto value = node.constantValue();
        return value.has_value() && *value == Real(0.0);
    }

    const auto children = node.children();
    if (children.empty()) {
        return false;
    }

    switch (node.type()) {
        case FormExprType::Negate:
        case FormExprType::Gradient:
        case FormExprType::Divergence:
        case FormExprType::Curl:
        case FormExprType::Hessian:
        case FormExprType::TimeDerivative:
        case FormExprType::RestrictMinus:
        case FormExprType::RestrictPlus:
        case FormExprType::Jump:
        case FormExprType::Average:
        case FormExprType::Transpose:
        case FormExprType::Trace:
        case FormExprType::Deviator:
        case FormExprType::SymmetricPart:
        case FormExprType::SkewPart:
        case FormExprType::Norm:
        case FormExprType::Normalize:
        case FormExprType::AbsoluteValue:
        case FormExprType::Sign:
        case FormExprType::Sqrt:
        case FormExprType::CellIntegral:
        case FormExprType::BoundaryIntegral:
        case FormExprType::InteriorFaceIntegral:
        case FormExprType::InterfaceIntegral:
            return children.front() && isZeroLikeExpression(*children.front());

        case FormExprType::Add:
        case FormExprType::Subtract:
            return std::all_of(children.begin(), children.end(),
                               [](const forms::FormExprNode* child) {
                                   return child && isZeroLikeExpression(*child);
                               });

        case FormExprType::Multiply:
        case FormExprType::InnerProduct:
        case FormExprType::DoubleContraction:
        case FormExprType::OuterProduct:
        case FormExprType::CrossProduct:
            return std::any_of(children.begin(), children.end(),
                               [](const forms::FormExprNode* child) {
                                   return child && isZeroLikeExpression(*child);
                               });

        case FormExprType::Divide:
            // A zero numerator is not enough to prove a quotient is zero:
            // the denominator may be zero on part of the expression domain.
            return false;

        case FormExprType::AsVector:
        case FormExprType::AsTensor:
            return std::all_of(children.begin(), children.end(),
                               [](const forms::FormExprNode* child) {
                                   return child && isZeroLikeExpression(*child);
                               });

        default:
            return false;
    }
}

bool staticallyZeroContribution(const ContributionDescriptor& contribution)
{
    return contribution.source_expression &&
           isZeroLikeExpression(*contribution.source_expression);
}

bool fieldHasNonzeroTrialOccurrence(const forms::FormExprNode& node,
                                    FieldId field)
{
    using forms::FormExprType;

    if (isZeroLikeExpression(node)) {
        return false;
    }

    switch (node.type()) {
        case FormExprType::DiscreteField:
        case FormExprType::StateField:
        case FormExprType::TrialFunction: {
            const auto fid = node.fieldId();
            return fid.has_value() && *fid == field;
        }
        default:
            break;
    }

    const auto children = node.children();
    return std::any_of(children.begin(), children.end(),
                       [field](const forms::FormExprNode* child) {
                           return child &&
                                  fieldHasNonzeroTrialOccurrence(*child, field);
                       });
}

bool nodeIsTrialField(const forms::FormExprNode& node, FieldId field)
{
    using forms::FormExprType;
    switch (node.type()) {
        case FormExprType::DiscreteField:
        case FormExprType::StateField:
        case FormExprType::TrialFunction: {
            const auto fid = node.fieldId();
            return fid.has_value() && *fid == field;
        }
        default:
            return false;
    }
}

bool nodeIsTestField(const forms::FormExprNode& node, FieldId field)
{
    if (node.type() != forms::FormExprType::TestFunction) {
        return false;
    }
    const auto fid = node.fieldId();
    return fid.has_value() && *fid == field;
}

bool subtreeContainsTestField(const forms::FormExprNode& node, FieldId field)
{
    if (isZeroLikeExpression(node)) {
        return false;
    }
    if (nodeIsTestField(node, field)) {
        return true;
    }

    const auto children = node.children();
    return std::any_of(children.begin(), children.end(),
                       [field](const forms::FormExprNode* child) {
                           return child &&
                                  subtreeContainsTestField(*child, field);
                       });
}

bool subtreeContainsTrialField(const forms::FormExprNode& node, FieldId field)
{
    if (isZeroLikeExpression(node)) {
        return false;
    }
    if (nodeIsTrialField(node, field)) {
        return true;
    }

    const auto children = node.children();
    return std::any_of(children.begin(), children.end(),
                       [field](const forms::FormExprNode* child) {
                           return child &&
                                  subtreeContainsTrialField(*child, field);
                       });
}

bool subtreeContainsNonScalarMultiplierTrialField(const forms::FormExprNode& node,
                                         FieldId multiplier_field)
{
    if (isZeroLikeExpression(node)) {
        return false;
    }

    using forms::FormExprType;
    switch (node.type()) {
        case FormExprType::DiscreteField:
        case FormExprType::StateField:
        case FormExprType::TrialFunction: {
            const auto fid = node.fieldId();
            return fid.has_value() && *fid != multiplier_field;
        }
        default:
            break;
    }

    const auto children = node.children();
    return std::any_of(children.begin(), children.end(),
                       [multiplier_field](const forms::FormExprNode* child) {
                           return child &&
                                  subtreeContainsNonScalarMultiplierTrialField(
                                      *child, multiplier_field);
                       });
}

struct TrialUseState {
    bool under_annihilating_operator{false};
    bool under_time_derivative{false};
};

bool subtreeContainsUndifferentiatedTrialField(
    const forms::FormExprNode& node,
    FieldId field,
    TrialUseState state = {})
{
    using forms::FormExprType;

    if (isZeroLikeExpression(node)) {
        return false;
    }
    if (nodeIsTrialField(node, field)) {
        return !state.under_annihilating_operator &&
               !state.under_time_derivative;
    }

    TrialUseState child_state = state;
    switch (node.type()) {
        case FormExprType::Gradient:
        case FormExprType::Divergence:
        case FormExprType::Curl:
        case FormExprType::Hessian:
            child_state.under_annihilating_operator = true;
            break;
        case FormExprType::TimeDerivative:
            child_state.under_time_derivative = true;
            break;
        default:
            break;
    }

    const auto children = node.children();
    return std::any_of(children.begin(), children.end(),
                       [field, child_state](const forms::FormExprNode* child) {
                           return child &&
                                  subtreeContainsUndifferentiatedTrialField(
                                      *child, field, child_state);
                       });
}

bool subtreeContainsDivergenceOfTestField(const forms::FormExprNode& node,
                                          FieldId field)
{
    if (isZeroLikeExpression(node)) {
        return false;
    }
    if (node.type() == forms::FormExprType::Divergence) {
        const auto children = node.children();
        return !children.empty() &&
               children.front() &&
               subtreeContainsTestField(*children.front(), field);
    }

    const auto children = node.children();
    return std::any_of(children.begin(), children.end(),
                       [field](const forms::FormExprNode* child) {
                           return child &&
                                  subtreeContainsDivergenceOfTestField(
                                      *child, field);
                       });
}

bool subtreeContainsDivergenceOfTrialField(const forms::FormExprNode& node,
                                           FieldId field)
{
    if (isZeroLikeExpression(node)) {
        return false;
    }
    if (node.type() == forms::FormExprType::Divergence) {
        const auto children = node.children();
        return !children.empty() &&
               children.front() &&
               subtreeContainsTrialField(*children.front(), field);
    }

    const auto children = node.children();
    return std::any_of(children.begin(), children.end(),
                       [field](const forms::FormExprNode* child) {
                           return child &&
                                  subtreeContainsDivergenceOfTrialField(
                                      *child, field);
                       });
}

bool subtreeContainsAnyPrimalVectorTestDivergence(
    const forms::FormExprNode& node,
    const std::unordered_set<FieldId>& primal_vector_fields)
{
    return std::any_of(primal_vector_fields.begin(), primal_vector_fields.end(),
                       [&node](FieldId primal_vector_field) {
                           return subtreeContainsDivergenceOfTestField(
                               node, primal_vector_field);
                       });
}

bool subtreeContainsAnyPrimalVectorTrialDivergence(
    const forms::FormExprNode& node,
    const std::unordered_set<FieldId>& primal_vector_fields)
{
    return std::any_of(primal_vector_fields.begin(), primal_vector_fields.end(),
                       [&node](FieldId primal_vector_field) {
                           return subtreeContainsDivergenceOfTrialField(
                               node, primal_vector_field);
                       });
}

bool subtreeIsCertifiedScalarMultiplierBlock(
    const forms::FormExprNode& node,
    FieldId multiplier_field,
    const std::unordered_set<FieldId>& primal_vector_fields)
{
    using forms::FormExprType;
    if (node.type() != FormExprType::Multiply &&
        node.type() != FormExprType::InnerProduct &&
        node.type() != FormExprType::DoubleContraction) {
        return false;
    }

    return subtreeContainsUndifferentiatedTrialField(node, multiplier_field) &&
           subtreeContainsAnyPrimalVectorTestDivergence(node, primal_vector_fields) &&
           !subtreeContainsNonScalarMultiplierTrialField(node, multiplier_field);
}

bool multiplierTrialUseOutsideCertifiedGaugePattern(
    const forms::FormExprNode& node,
    FieldId multiplier_field,
    const std::unordered_set<FieldId>& primal_vector_fields,
    TrialUseState state = {})
{
    using forms::FormExprType;

    if (isZeroLikeExpression(node)) {
        return false;
    }
    if (!state.under_annihilating_operator &&
        !state.under_time_derivative &&
        subtreeIsCertifiedScalarMultiplierBlock(
            node, multiplier_field, primal_vector_fields)) {
        return false;
    }
    if (nodeIsTrialField(node, multiplier_field)) {
        if (state.under_time_derivative) {
            return true;
        }
        return !state.under_annihilating_operator;
    }

    TrialUseState child_state = state;
    switch (node.type()) {
        case FormExprType::Gradient:
        case FormExprType::Divergence:
        case FormExprType::Curl:
        case FormExprType::Hessian:
            child_state.under_annihilating_operator = true;
            break;
        case FormExprType::TimeDerivative:
            child_state.under_time_derivative = true;
            break;
        default:
            break;
    }

    const auto children = node.children();
    return std::any_of(children.begin(), children.end(),
                       [multiplier_field, &primal_vector_fields, child_state](
                           const forms::FormExprNode* child) {
                           return child &&
                                  multiplierTrialUseOutsideCertifiedGaugePattern(
                                      *child,
                                      multiplier_field,
                                      primal_vector_fields,
                                      child_state);
                       });
}

bool subtreeContainsCertifiedScalarMultiplierBlock(
    const forms::FormExprNode& node,
    FieldId multiplier_field,
    const std::unordered_set<FieldId>& primal_vector_fields)
{
    if (isZeroLikeExpression(node)) {
        return false;
    }
    if (subtreeIsCertifiedScalarMultiplierBlock(
            node, multiplier_field, primal_vector_fields)) {
        return true;
    }

    const auto children = node.children();
    return std::any_of(children.begin(), children.end(),
                       [multiplier_field, &primal_vector_fields](
                           const forms::FormExprNode* child) {
                           return child &&
                                  subtreeContainsCertifiedScalarMultiplierBlock(
                                      *child,
                                      multiplier_field,
                                      primal_vector_fields);
                       });
}

bool subtreeIsScalarMultiplierContinuityBlock(
    const forms::FormExprNode& node,
    FieldId multiplier_field,
    const std::unordered_set<FieldId>& primal_vector_fields)
{
    using forms::FormExprType;
    if (node.type() != FormExprType::Multiply &&
        node.type() != FormExprType::InnerProduct &&
        node.type() != FormExprType::DoubleContraction) {
        return false;
    }

    return subtreeContainsTestField(node, multiplier_field) &&
           subtreeContainsAnyPrimalVectorTrialDivergence(node, primal_vector_fields);
}

bool subtreeContainsScalarMultiplierContinuityBlock(
    const forms::FormExprNode& node,
    FieldId multiplier_field,
    const std::unordered_set<FieldId>& primal_vector_fields)
{
    if (isZeroLikeExpression(node)) {
        return false;
    }
    if (subtreeIsScalarMultiplierContinuityBlock(
            node, multiplier_field, primal_vector_fields)) {
        return true;
    }

    const auto children = node.children();
    return std::any_of(children.begin(), children.end(),
                       [multiplier_field, &primal_vector_fields](
                           const forms::FormExprNode* child) {
                           return child &&
                                  subtreeContainsScalarMultiplierContinuityBlock(
                                      *child,
                                      multiplier_field,
                                      primal_vector_fields);
                       });
}

bool fieldSummaryHasOnlyConstantAnnihilatingTrialUse(
    const ContributionDescriptor& contribution,
    FieldId field,
    FormStructureAnalyzer& fsa)
{
    if (!contribution.source_expression) {
        return false;
    }
    if (!fieldHasNonzeroTrialOccurrence(*contribution.source_expression, field)) {
        return false;
    }

    const auto fs = fsa.analyzeField(*contribution.source_expression, field);
    return fs.occurrence_count > 0 &&
           fs.only_through_annihilating_ops &&
           !fs.has_absolute_value &&
           !fs.has_time_derivative;
}

bool fieldSummaryHasUndifferentiatedTrialUse(
    const ContributionDescriptor& contribution,
    FieldId field,
    FormStructureAnalyzer& fsa)
{
    if (!contribution.source_expression) {
        return false;
    }
    if (!fieldHasNonzeroTrialOccurrence(*contribution.source_expression, field)) {
        return false;
    }

    const auto fs = fsa.analyzeField(*contribution.source_expression, field);
    return fs.occurrence_count > 0 &&
           fs.has_absolute_value &&
           !fs.has_time_derivative;
}

bool fieldSummaryHasDivergenceTrialUse(const ContributionDescriptor& contribution,
                                       FieldId field,
                                       FormStructureAnalyzer& fsa)
{
    if (!contribution.source_expression) {
        return false;
    }
    if (!fieldHasNonzeroTrialOccurrence(*contribution.source_expression, field)) {
        return false;
    }

    const auto fs = fsa.analyzeField(*contribution.source_expression, field);
    return fs.occurrence_count > 0 &&
           fs.has_divergence &&
           !fs.has_absolute_value &&
           !fs.has_time_derivative;
}

bool hasPairing(const ContributionDescriptor& contribution,
                FieldId row_field,
                FieldId col_field,
                PairingKind kind)
{
    return std::any_of(contribution.pairings.begin(),
                       contribution.pairings.end(),
                       [row_field, col_field, kind](const PairingDescriptor& pairing) {
                           return pairing.kind == kind &&
                                  variableIsField(pairing.row_var, row_field) &&
                                  variableIsField(pairing.col_var, col_field);
                       });
}

bool hasUndifferentiatedConstraintPair(const ContributionDescriptor& contribution,
                                       FieldId row_field,
                                       FieldId col_field)
{
    return std::any_of(contribution.pairings.begin(),
                       contribution.pairings.end(),
                       [row_field, col_field](const PairingDescriptor& pairing) {
                           return pairing.kind == PairingKind::ConstraintPair &&
                                  pairing.trial_has_undifferentiated &&
                                  variableIsField(pairing.row_var, row_field) &&
                                  variableIsField(pairing.col_var, col_field);
                       });
}

bool multiplierTrialContributionIsAllowedForGaugeInference(
    const ContributionDescriptor& contribution,
    FieldId multiplier_field,
    const std::unordered_set<FieldId>& certified_primal_vector_fields,
    FormStructureAnalyzer& fsa)
{
    if (!contributionHasTrialField(contribution, multiplier_field)) {
        return true;
    }
    if (staticallyZeroContribution(contribution)) {
        return true;
    }
    if (contribution.source_expression &&
        !fieldHasNonzeroTrialOccurrence(*contribution.source_expression,
                                        multiplier_field)) {
        return true;
    }

    if (contributionHasTestField(contribution, multiplier_field)) {
        if (!contribution.source_expression) {
            return !hasFlag(contribution.traits, OperatorTraitFlags::HasMass) &&
                   !hasFlag(contribution.traits, OperatorTraitFlags::HasFirstOrder) &&
                   !hasFlag(contribution.traits, OperatorTraitFlags::NullspaceLifting);
        }

        const auto fs =
            fsa.analyzeField(*contribution.source_expression, multiplier_field);
        return fs.occurrence_count > 0 &&
               fs.only_through_annihilating_ops &&
               !fs.has_absolute_value &&
               !fs.has_time_derivative &&
               !hasFlag(contribution.traits, OperatorTraitFlags::HasMass) &&
               !hasFlag(contribution.traits, OperatorTraitFlags::HasFirstOrder) &&
               !hasFlag(contribution.traits, OperatorTraitFlags::NullspaceLifting);
    }

    const bool tests_certified_primal_vector =
        std::any_of(certified_primal_vector_fields.begin(),
                    certified_primal_vector_fields.end(),
                    [&contribution](FieldId primal_vector_field) {
                        return contributionHasTestField(contribution, primal_vector_field);
                    });
    if (tests_certified_primal_vector &&
        fieldSummaryHasUndifferentiatedTrialUse(
            contribution, multiplier_field, fsa)) {
        return true;
    }

    return fieldSummaryHasOnlyConstantAnnihilatingTrialUse(
        contribution, multiplier_field, fsa);
}

struct InferredScalarMultiplierNullspace {
    bool certified{false};
    std::string evidence;
};

InferredScalarMultiplierNullspace inferScalarMultiplierConstantNullspace(
    const ProblemAnalysisContext& context,
    const VariableKey& constraint_variable,
    const std::unordered_set<VariableKey, VariableKeyHash>& partners,
    const std::vector<const ContributionDescriptor*>& contributions)
{
    InferredScalarMultiplierNullspace result;

    if (constraint_variable.kind != VariableKind::FieldComponent ||
        constraint_variable.field_id == INVALID_FIELD_ID ||
        !isScalarFieldDescriptor(context, constraint_variable.field_id)) {
        return result;
    }

    FormStructureAnalyzer fsa;
    std::unordered_set<FieldId> certified_primal_vector_fields;

    for (const auto& primal_vector_variable : partners) {
        if (primal_vector_variable.kind != VariableKind::FieldComponent ||
            primal_vector_variable.field_id == INVALID_FIELD_ID ||
            !isVectorFieldDescriptor(context, primal_vector_variable.field_id)) {
            continue;
        }

        bool has_multiplier_in_primal_vector_equation = false;
        bool has_divergence_in_constraint_equation = false;

        for (const auto* contribution : contributions) {
            if (!contribution) {
                continue;
            }
            if (staticallyZeroContribution(*contribution)) {
                continue;
            }

            if (contributionHasTestField(*contribution,
                                         primal_vector_variable.field_id) &&
                contributionHasTrialField(*contribution,
                                          constraint_variable.field_id) &&
                hasUndifferentiatedConstraintPair(
                    *contribution,
                    primal_vector_variable.field_id,
                    constraint_variable.field_id) &&
                fieldSummaryHasUndifferentiatedTrialUse(
                    *contribution,
                    constraint_variable.field_id,
                    fsa)) {
                has_multiplier_in_primal_vector_equation = true;
            }

            if (contributionHasTestField(*contribution,
                                         constraint_variable.field_id) &&
                contributionHasTrialField(*contribution,
                                          primal_vector_variable.field_id) &&
                hasPairing(*contribution,
                           constraint_variable.field_id,
                           primal_vector_variable.field_id,
                           PairingKind::FormalAdjointPair) &&
                fieldSummaryHasDivergenceTrialUse(
                    *contribution,
                    primal_vector_variable.field_id,
                    fsa)) {
                has_divergence_in_constraint_equation = true;
            }
        }

        if (has_multiplier_in_primal_vector_equation &&
            has_divergence_in_constraint_equation) {
            certified_primal_vector_fields.insert(primal_vector_variable.field_id);
        }
    }

    if (certified_primal_vector_fields.empty()) {
        return result;
    }

    for (const auto* contribution : contributions) {
        if (!contribution) {
            continue;
        }
        if (!multiplierTrialContributionIsAllowedForGaugeInference(
                *contribution,
                constraint_variable.field_id,
                certified_primal_vector_fields,
                fsa)) {
            return result;
        }
    }

    result.certified = true;
    result.evidence =
        "Structural scalar-constant multiplier-gauge candidate: scalar constraint field has "
        "an undifferentiated multiplier block against a primal vector "
        "divergence test operator, the adjoint constraint equation contains "
        "div(primal vector), and all nonzero trial appearances of the scalar field "
        "are either that multiplier block or constant-annihilating terms; no "
        "nonzero scalar mass/reaction/time/lifting block was found";
    return result;
}

InferredScalarMultiplierNullspace inferScalarMultiplierConstantNullspaceFromResidual(
    const ProblemAnalysisContext& context,
    const forms::FormExprNode& residual,
    FieldId multiplier_field,
    const std::vector<FieldId>& candidate_primal_vector_fields)
{
    InferredScalarMultiplierNullspace result;
    if (!isScalarFieldDescriptor(context, multiplier_field)) {
        return result;
    }

    std::unordered_set<FieldId> primal_vector_fields;
    for (FieldId field : candidate_primal_vector_fields) {
        if (isVectorFieldDescriptor(context, field)) {
            primal_vector_fields.insert(field);
        }
    }
    if (primal_vector_fields.empty()) {
        return result;
    }

    if (!subtreeContainsCertifiedScalarMultiplierBlock(
            residual, multiplier_field, primal_vector_fields)) {
        return result;
    }
    if (!subtreeContainsScalarMultiplierContinuityBlock(
            residual, multiplier_field, primal_vector_fields)) {
        return result;
    }
    if (multiplierTrialUseOutsideCertifiedGaugePattern(
            residual, multiplier_field, primal_vector_fields)) {
        return result;
    }

    result.certified = true;
    result.evidence =
        "Structural scalar-constant multiplier-gauge candidate from residual structure: "
        "the residual contains scalar-multiplier times primal-vector-divergence coupling and adjoint divergence coupling; "
        "all nonzero multiplier trial occurrences are either in that multiplier "
        "coupling or under constant-annihilating differential operators";
    return result;
}

LocalDeclaredNullspaceMetadata declaredNullspaceForField(
    const ProblemAnalysisContext& context,
    FieldId field,
    int component,
    const std::vector<const ContributionDescriptor*>* contributions)
{
    if (contributions) {
        for (const auto* contribution : *contributions) {
            for (const auto& hint : contribution->nullspace_hints) {
                if (hint.field != field) {
                    continue;
                }
                if (component >= 0 && hint.component >= 0 &&
                    hint.component != component) {
                    continue;
                }
                LocalDeclaredNullspaceMetadata metadata;
                metadata.present = true;
                metadata.family = hint.family;
                metadata.confidence = hint.confidence;
                metadata.evidence_kind = hint.evidence_kind;
                metadata.evidence =
        "ContributionDescriptor declared scalar-multiplier nullspace family=" +
                    std::string(toString(hint.family));
                if (!hint.reason.empty()) {
                    metadata.evidence += ": " + hint.reason;
                }
                return metadata;
            }
        }
    }

    const auto* descriptor = context.fieldDescriptor(field);
    if (descriptor && descriptor->declared_nullspace_metadata_present) {
        LocalDeclaredNullspaceMetadata metadata;
        metadata.present = true;
        metadata.family = descriptor->declared_nullspace_family;
        metadata.confidence = AnalysisConfidence::High;
        metadata.evidence_kind = descriptor->declared_nullspaces.empty()
            ? NullspaceEvidenceKind::DescriptorHint
            : descriptor->declared_nullspaces.front().evidence_kind;
        metadata.evidence =
            "FieldDescriptor declared scalar-multiplier nullspace family=" +
            std::string(toString(descriptor->declared_nullspace_family));
        if (!descriptor->declared_nullspace_scope.empty()) {
            metadata.evidence += ", scope='" +
                                 descriptor->declared_nullspace_scope + "'";
        }
        return metadata;
    }

    return {};
}

void emitSchurSummaryClaim(ProblemAnalysisReport& report,
                           const ReducedMatrixSummary& summary)
{
    const auto& matrix = summary.free_free_matrix;
    const bool tagged_schur =
        matrix.block.role == ContributionRole::ConstraintBlock ||
        matrix.block.operator_tag.find("schur") != std::string::npos ||
        matrix.block.operator_tag.find("Schur") != std::string::npos;
    if (!tagged_schur) {
        return;
    }

    const bool exact_reduction =
        summary.reduction_exact_for_analysis &&
        matrix.rows > 0 &&
        matrix.cols > 0;

    PropertyClaim claim;
    claim.kind = PropertyKind::IndefiniteOperatorResolution;
    claim.status = exact_reduction ? PropertyStatus::Likely
                                   : PropertyStatus::Unknown;
    claim.confidence = exact_reduction ? AnalysisConfidence::Medium
                                       : AnalysisConfidence::Medium;
    claim.domain = matrix.block.domain;
    claim.variables = blockVariables(matrix.block);
    claim.reduced_definiteness_class =
        exact_reduction ? CertificationClass::NotCertified
                        : CertificationClass::Unknown;
    claim.tested_block_id = matrix.block.operator_tag;
    claim.description = exact_reduction
        ? "Reduced Schur/constraint summary is available, but stability certification requires inf-sup and Schur conditioning/equivalence evidence"
        : "Reduced Schur/constraint summary is incomplete for indefinite block resolution";
    claim.claim_origin = "MixedOperatorAnalyzer";
    claim.addEvidence("MixedOperatorAnalyzer",
        "ReducedMatrixSummary for Schur-like block has rows=" +
        std::to_string(matrix.rows) +
        ", cols=" + std::to_string(matrix.cols) +
        ", exact_reduction=" +
        std::string(summary.reduction_exact_for_analysis ? "true" : "false"),
        claim.confidence);
    report.claims.push_back(std::move(claim));
}

bool nullspaceHandlingAcceptable(NullspaceHandlingClass handling) noexcept
{
    return handling == NullspaceHandlingClass::NotApplicable ||
           handling == NullspaceHandlingClass::AnchoredByConstraints ||
           handling == NullspaceHandlingClass::ProjectedOut;
}

bool positiveSchurEvidence(PositivityClass positivity) noexcept
{
    return positivity == PositivityClass::Positive;
}

bool quotientSchurEvidence(const SchurComplementSummary& summary) noexcept
{
    return summary.schur_positivity == PositivityClass::Nonnegative &&
           summary.schur_positive_on_quotient &&
           summary.quotient_space_scope_present &&
           summary.multiplier_gauge_projected &&
           summary.nullspace_basis_matches &&
           !summary.quotient_norm_id.empty() &&
           numeric::finitePositiveOrdered(
               summary.schur_lower_bound_on_quotient,
               summary.schur_upper_bound_on_quotient);
}

bool validPositiveOrderedBounds(Real lower, Real upper) noexcept
{
    return numeric::finitePositiveOrdered(lower, upper);
}

bool schurCertificationComplete(const SchurComplementSummary& summary) noexcept
{
    const bool spectral_bounds_valid =
        summary.spectral_equivalence_bounds_present &&
        validPositiveOrderedBounds(summary.spectral_equivalence_lower_bound,
                                   summary.spectral_equivalence_upper_bound);
    const bool preconditioner_bounds_valid =
        summary.preconditioner_equivalence_bounds_present &&
        validPositiveOrderedBounds(
            summary.preconditioner_equivalence_lower_bound,
            summary.preconditioner_equivalence_upper_bound);
    return summary.schur_available &&
           summary.reduction_exact_for_analysis &&
           summary.primal_block_invertible_evidence_present &&
           summary.inf_sup_evidence_present &&
           summary.nullspace_handling_evidence_present &&
           nullspaceHandlingAcceptable(summary.nullspace_handling) &&
           summary.schur_definiteness_evidence_present &&
           (positiveSchurEvidence(summary.schur_positivity) ||
            quotientSchurEvidence(summary)) &&
           spectral_bounds_valid &&
           preconditioner_bounds_valid;
}

void emitSchurComplementClaim(ProblemAnalysisReport& report,
                              const SchurComplementSummary& summary)
{
    PropertyClaim claim;
    claim.kind = PropertyKind::IndefiniteOperatorResolution;
    claim.domain = summary.block.domain;
    claim.variables = !summary.variables.empty()
        ? summary.variables
        : blockVariables(summary.block);
    claim.tested_block_id = summary.block.operator_tag.empty()
        ? summary.schur_id
        : summary.block.operator_tag;
    claim.estimate_scope = summary.schur_id;
    claim.claim_origin = "MixedOperatorAnalyzer";
    claim.nullspace_handling_class = summary.nullspace_handling;
    claim.evidence_level = EvidenceLevel::ScopedNumericSummary;

    const bool certified = schurCertificationComplete(summary);
    if (!summary.schur_available) {
        claim.status = PropertyStatus::Unknown;
        claim.confidence = AnalysisConfidence::Medium;
        claim.reduced_definiteness_class = CertificationClass::Unknown;
        claim.description =
            "Schur complement resolution is unknown because no Schur operator is available";
    } else if (certified) {
        claim.status = PropertyStatus::Preserved;
        claim.confidence = AnalysisConfidence::High;
        claim.reduced_definiteness_class = CertificationClass::Certified;
        claim.evidence_level = EvidenceLevel::CertifiedNumericTheorem;
        claim.description =
            "Schur complement resolution is certified by exact reduction, inf-sup, quotient/nullspace handling where needed, and spectral/preconditioner equivalence evidence";
    } else {
        claim.status = PropertyStatus::Likely;
        claim.confidence = AnalysisConfidence::Medium;
        claim.reduced_definiteness_class = CertificationClass::NotCertified;
        claim.description =
            "Schur complement is available but lacks complete inf-sup, nullspace, definiteness, or spectral/preconditioner equivalence evidence";
    }

    claim.addEvidence("MixedOperatorAnalyzer",
        "SchurComplementSummary id='" + summary.schur_id +
        "', available=" +
        std::string(summary.schur_available ? "true" : "false") +
        ", exact_reduction=" +
        std::string(summary.reduction_exact_for_analysis ? "true" : "false") +
        ", primal_invertible=" +
        std::string(summary.primal_block_invertible_evidence_present ? "true" : "false") +
        ", inf_sup=" +
        std::string(summary.inf_sup_evidence_present ? "true" : "false") +
        ", schur_positivity=" +
        std::to_string(static_cast<int>(summary.schur_positivity)) +
        ", positive_on_quotient=" +
        std::string(summary.schur_positive_on_quotient ? "true" : "false") +
        ", quotient_scope=" +
        std::string(summary.quotient_space_scope_present ? "true" : "false") +
        ", gauge_projected=" +
        std::string(summary.multiplier_gauge_projected ? "true" : "false") +
        ", nullspace_basis_matches=" +
        std::string(summary.nullspace_basis_matches ? "true" : "false") +
        ", quotient_norm='" + summary.quotient_norm_id + "'" +
        ", spectral_equivalence=" +
        std::string(summary.spectral_equivalence_bounds_present ? "true" : "false") +
        ", spectral_bounds=[" +
        std::to_string(summary.spectral_equivalence_lower_bound) +
        ", " +
        std::to_string(summary.spectral_equivalence_upper_bound) +
        "]" +
        ", preconditioner_equivalence=" +
        std::string(summary.preconditioner_equivalence_bounds_present ? "true" : "false") +
        ", preconditioner_bounds=[" +
        std::to_string(summary.preconditioner_equivalence_lower_bound) +
        ", " +
        std::to_string(summary.preconditioner_equivalence_upper_bound) +
        "]",
        claim.confidence);
    report.claims.push_back(std::move(claim));
}

void emitSchurSummaryHooks(const ProblemAnalysisContext& context,
                           ProblemAnalysisReport& report)
{
    const auto* summaries = context.analysisSummaries();
    if (!summaries) {
        return;
    }

    for (const auto& summary : summaries->reduced_matrices) {
        emitSchurSummaryClaim(report, summary);
    }
    for (const auto& summary : summaries->schur_complements) {
        emitSchurComplementClaim(report, summary);
    }
}

} // namespace

std::string MixedOperatorAnalyzer::name() const {
    return "MixedOperatorAnalyzer";
}

void MixedOperatorAnalyzer::run(const ProblemAnalysisContext& context,
                                ProblemAnalysisReport& report) const
{
    // Dedup for MixedSaddlePoint claims must include scope.  The same pair of
    // variables can appear in distinct cell, boundary, interface, auxiliary,
    // or contribution-local saddle systems.
    struct MixedPairScopeKey {
        VariableKey primal_vector;
        VariableKey constraint;
        DomainKind domain{DomainKind::Cell};
        int marker{-1};
        std::string operator_tag;
        std::string pairing_group;
        std::string contribution_id;
        bool operator==(const MixedPairScopeKey& o) const {
            return primal_vector == o.primal_vector &&
                   constraint == o.constraint &&
                   domain == o.domain &&
                   marker == o.marker &&
                   operator_tag == o.operator_tag &&
                   pairing_group == o.pairing_group &&
                   contribution_id == o.contribution_id;
        }
    };
    struct MixedPairScopeKeyHash {
        std::size_t operator()(const MixedPairScopeKey& key) const {
            std::size_t seed = VariableKeyHash{}(key.primal_vector);
            seed ^= VariableKeyHash{}(key.constraint) + 0x9e3779b97f4a7c15ULL +
                    (seed << 6U) + (seed >> 2U);
            seed ^= std::hash<int>{}(static_cast<int>(key.domain)) +
                    0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
            seed ^= std::hash<int>{}(key.marker) + 0x9e3779b97f4a7c15ULL +
                    (seed << 6U) + (seed >> 2U);
            seed ^= std::hash<std::string>{}(key.operator_tag) +
                    0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
            seed ^= std::hash<std::string>{}(key.pairing_group) +
                    0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
            seed ^= std::hash<std::string>{}(key.contribution_id) +
                    0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
            return seed;
        }
    };
    std::unordered_set<MixedPairScopeKey, MixedPairScopeKeyHash>
        emitted_mixed_pairs;

    // Nullspace claims are field-intrinsic (constant nullspace) — one per
    // constraint field regardless of how many primal-vector partners it has.
    std::unordered_set<FieldId> emitted_nullspace_fields;

    // =====================================================================
    // PRIMARY PATH: Connected-component analysis on the variable graph
    //
    // Build an adjacency graph where variables are nodes and each
    // contribution creates edges between its test and trial variables.
    // Connected components naturally separate:
    //   - Distinct formulations that share an operator_tag but use
    //     different fields (Issue 1)
    //   - Independent saddle-point subsystems within one formulation
    //     that don't share variables
    //
    // Within each component, constraint variables are paired only with
    // the primal-vector variables they actually couple with via
    // ConstraintBlock/OffDiagonalBlock contributions (Issue 2).
    // =====================================================================
    const auto& contributions = context.contributions();
    if (!contributions.empty()) {
        // --- Phase 1: Build variable adjacency graph ---
        std::unordered_map<VariableKey,
            std::unordered_set<VariableKey, VariableKeyHash>,
            VariableKeyHash> adj;

        for (const auto& contrib : contributions) {
            // Collect all variables from this contribution
            std::vector<VariableKey> vars;
            for (const auto& v : contrib.test_variables) vars.push_back(v);
            for (const auto& v : contrib.trial_variables) vars.push_back(v);

            // Ensure all nodes exist (including isolated ones)
            for (const auto& v : vars) adj[v];

            // Connect all pairs within this contribution
            for (std::size_t i = 0; i < vars.size(); ++i) {
                for (std::size_t j = i + 1; j < vars.size(); ++j) {
                    adj[vars[i]].insert(vars[j]);
                    adj[vars[j]].insert(vars[i]);
                }
            }
        }

        // --- Phase 2: BFS to find connected components ---
        std::unordered_map<VariableKey, int, VariableKeyHash> var_to_comp;
        int num_components = 0;

        for (const auto& [v, _] : adj) {
            if (var_to_comp.count(v)) continue;

            int comp_id = num_components++;
            std::vector<VariableKey> queue;
            queue.push_back(v);
            var_to_comp[v] = comp_id;

            for (std::size_t qi = 0; qi < queue.size(); ++qi) {
                for (const auto& neighbor : adj[queue[qi]]) {
                    if (!var_to_comp.count(neighbor)) {
                        var_to_comp[neighbor] = comp_id;
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // --- Phase 3: Assign contributions to components ---
        std::unordered_map<int, std::vector<const ContributionDescriptor*>>
            comp_contribs;
        for (const auto& contrib : contributions) {
            int comp_id = -1;
            for (const auto& v : contrib.test_variables) {
                auto it = var_to_comp.find(v);
                if (it != var_to_comp.end()) { comp_id = it->second; break; }
            }
            if (comp_id < 0) {
                for (const auto& v : contrib.trial_variables) {
                    auto it = var_to_comp.find(v);
                    if (it != var_to_comp.end()) { comp_id = it->second; break; }
                }
            }
            if (comp_id >= 0) {
                comp_contribs[comp_id].push_back(&contrib);
            }
        }

        // --- Phase 4: Analyze each component for saddle-point structure ---
        for (const auto& [comp_id, comp_group] : comp_contribs) {
            struct VarInfo {
                bool has_primal_diagonal_candidate = false;
                bool appears_in_constraint = false;
                bool appears_in_offdiagonal = false;
                bool has_constraint_pair_metadata = false;
                bool has_formal_adjoint_pair_metadata = false;
            };
            std::unordered_map<VariableKey, VarInfo, VariableKeyHash> var_info;

            for (const auto* contrib : comp_group) {
                for (const auto& tv : contrib->test_variables) var_info[tv];
                for (const auto& tv : contrib->trial_variables) var_info[tv];

                if (contrib->role == ContributionRole::DiagonalBlock) {
                    if (hasFlag(contrib->traits,
                                OperatorTraitFlags::HasSecondOrder)) {
                        for (const auto& tv : contrib->test_variables) {
                            var_info[tv].has_primal_diagonal_candidate = true;
                        }
                    }
                } else if (contrib->role == ContributionRole::ConstraintBlock) {
                    for (const auto& tv : contrib->test_variables)
                        var_info[tv].appears_in_constraint = true;
                    for (const auto& tv : contrib->trial_variables)
                        var_info[tv].appears_in_constraint = true;
                } else if (contrib->role == ContributionRole::OffDiagonalBlock) {
                    for (const auto& tv : contrib->test_variables)
                        var_info[tv].appears_in_offdiagonal = true;
                    for (const auto& tv : contrib->trial_variables)
                        var_info[tv].appears_in_offdiagonal = true;
                }
                for (const auto& pairing : contrib->pairings) {
                    auto& row_info = var_info[pairing.row_var];
                    auto& col_info = var_info[pairing.col_var];
                    if (pairing.kind == PairingKind::ConstraintPair ||
                        pairing.kind == PairingKind::StabilizedConstraintPair) {
                        row_info.has_constraint_pair_metadata = true;
                        col_info.has_constraint_pair_metadata = true;
                    }
                    if (pairing.kind == PairingKind::FormalAdjointPair) {
                        row_info.has_formal_adjoint_pair_metadata = true;
                        col_info.has_formal_adjoint_pair_metadata = true;
                    }
                }
            }

            // Identify primal-vector and constraint variables
            std::unordered_set<VariableKey, VariableKeyHash> primal_vector_set;
            std::unordered_set<VariableKey, VariableKeyHash> constraint_set;

            for (const auto& [vk, info] : var_info) {
                if (info.has_primal_diagonal_candidate) {
                    primal_vector_set.insert(vk);
                }
                const bool explicit_constraint_metadata =
                    info.appears_in_constraint ||
                    info.has_constraint_pair_metadata ||
                    (info.appears_in_offdiagonal &&
                     info.has_formal_adjoint_pair_metadata);
                if (explicit_constraint_metadata &&
                    !info.has_primal_diagonal_candidate) {
                    constraint_set.insert(vk);
                }
            }

            if (primal_vector_set.empty() || constraint_set.empty()) continue;

            // Build per-constraint coupling map: which primal-vector vars does
            // each constraint var actually couple with via
            // ConstraintBlock/OffDiagonalBlock contributions?
            struct MixedPairCandidate {
                VariableKey primal_vector;
                VariableKey constraint;
                DomainKind domain{DomainKind::Cell};
                int marker{-1};
                std::string operator_tag;
                std::string pairing_group;
                std::string contribution_id;
            };
            std::unordered_map<VariableKey,
                std::vector<MixedPairCandidate>,
                VariableKeyHash> constraint_to_primal_vector;

            for (const auto* contrib : comp_group) {
                if (contrib->role != ContributionRole::ConstraintBlock &&
                    contrib->role != ContributionRole::OffDiagonalBlock) {
                    continue;
                }
                auto pairingGroupFor = [&](const VariableKey& row,
                                           const VariableKey& col) {
                    for (const auto& pairing : contrib->pairings) {
                        if ((pairing.row_var == row && pairing.col_var == col) ||
                            (pairing.row_var == col && pairing.col_var == row)) {
                            return pairing.pairing_group;
                        }
                    }
                    return std::string{};
                };
                auto appendCandidate = [&](const VariableKey& constraint,
                                           const VariableKey& primal_vector,
                                           const VariableKey& row,
                                           const VariableKey& col) {
                    MixedPairCandidate candidate;
                    candidate.primal_vector = primal_vector;
                    candidate.constraint = constraint;
                    candidate.domain = contrib->domain;
                    candidate.marker = contrib->interface_marker >= 0
                        ? contrib->interface_marker
                        : contrib->boundary_marker;
                    candidate.operator_tag = contrib->operator_tag;
                    candidate.pairing_group = pairingGroupFor(row, col);
                    candidate.contribution_id = contrib->contribution_id;
                    constraint_to_primal_vector[constraint].push_back(
                        std::move(candidate));
                };
                // Check all (test, trial) pairs for primal-vector/constraint links
                for (const auto& tv : contrib->test_variables) {
                    for (const auto& trv : contrib->trial_variables) {
                        if (constraint_set.count(tv) && primal_vector_set.count(trv))
                            appendCandidate(tv, trv, tv, trv);
                        if (constraint_set.count(trv) && primal_vector_set.count(tv))
                            appendCandidate(trv, tv, tv, trv);
                    }
                }
            }

            // Emit one MixedSaddlePoint claim per (primal-vector, constraint) pair.
            // Each claim carries exactly two variables so InfSupAnalyzer's
            // pair-based coverage check is precisely scoped.
            for (const auto& cv : constraint_set) {
                FieldId constraint_fid = cv.field_id;
                std::vector<VariableKey> nullspace_partners;

                // Determine this constraint's primal-vector partners
                auto coup_it = constraint_to_primal_vector.find(cv);
                if (coup_it == constraint_to_primal_vector.end() ||
                    coup_it->second.empty()) {
                    PropertyClaim claim;
                    claim.kind = PropertyKind::MixedSaddlePoint;
                    claim.status = PropertyStatus::Unknown;
                    claim.confidence = AnalysisConfidence::Medium;
                    claim.certification_class =
                        CertificationClass::NotCertified;
                    claim.evidence_level = EvidenceLevel::StructuralMetadata;
                    claim.definiteness_interpretation =
                        DefinitenessInterpretation::SaddlePointExpected;
                    claim.domain = DomainKind::Cell;
                    claim.variables.push_back(cv);
                    claim.claim_origin = "MixedOperatorAnalyzer";
                    claim.description =
                        "Saddle-point structure requires inf-sup evidence for constraint/multiplier variable " +
                        (cv.kind == VariableKind::FieldComponent
                            ? ("field " + std::to_string(constraint_fid))
                            : cv.name) +
                        ", but no unique primal partner was identified";
                    claim.addEvidence("MixedOperatorAnalyzer",
                        "Constraint/multiplier metadata is present, but pair-level primal block scope is not unique");
                    report.claims.push_back(std::move(claim));
                } else {
                    for (const auto& candidate : coup_it->second) {
                        const auto& mv = candidate.primal_vector;
                        appendUnique(nullspace_partners, mv);
                        MixedPairScopeKey key;
                        key.primal_vector = mv;
                        key.constraint = cv;
                        key.domain = candidate.domain;
                        key.marker = candidate.marker;
                        key.operator_tag = candidate.operator_tag;
                        key.pairing_group = candidate.pairing_group;
                        key.contribution_id = candidate.contribution_id;
                        if (emitted_mixed_pairs.count(key)) continue;
                        emitted_mixed_pairs.insert(std::move(key));

                        PropertyClaim claim;
                        claim.kind = PropertyKind::MixedSaddlePoint;
                        claim.status = PropertyStatus::Likely;
                        claim.confidence = AnalysisConfidence::Medium;
                        claim.certification_class =
                            CertificationClass::NotCertified;
                        claim.evidence_level =
                            EvidenceLevel::StructuralMetadata;
                        claim.definiteness_interpretation =
                            DefinitenessInterpretation::SaddlePointExpected;
                        claim.domain = candidate.domain;
                        claim.variables.push_back(mv);
                        claim.variables.push_back(cv);
                        if (!candidate.operator_tag.empty()) {
                            claim.tested_block_id = candidate.operator_tag;
                        }
                        if (!candidate.pairing_group.empty()) {
                            claim.estimate_scope = candidate.pairing_group;
                        } else if (!candidate.contribution_id.empty()) {
                            claim.estimate_scope = candidate.contribution_id;
                        }
                        claim.claim_origin = "MixedOperatorAnalyzer";

                        claim.description =
                            "Saddle-point structure: variable " +
                            (cv.kind == VariableKind::FieldComponent
                                ? ("field " + std::to_string(constraint_fid))
                                : cv.name) +
                            " (constraint/multiplier, no diagonal elliptic"
                            " block)";
                        claim.addEvidence("MixedOperatorAnalyzer",
                            "Constraint/multiplier metadata with paired structural primal-vector block; no scoped inf-sup/rank theorem evidence is attached. Pair scope domain=" +
                                std::string(toString(candidate.domain)) +
                                ", marker=" + std::to_string(candidate.marker) +
                                ", operator_tag='" + candidate.operator_tag +
                                "', contribution_id='" + candidate.contribution_id + "'");
                        report.claims.push_back(std::move(claim));
                    }
                }

                // Nullspace claim: one per constraint field.  Explicit
                // metadata remains authoritative, and otherwise the analyzer
                // can certify the canonical multiplier-gauge case from the
                // block operator structure itself.
                if (cv.kind == VariableKind::FieldComponent &&
                    constraint_fid != INVALID_FIELD_ID &&
                    !emitted_nullspace_fields.count(constraint_fid)) {
                    const auto metadata = declaredNullspaceForField(
                        context, constraint_fid, cv.component, &comp_group);
                    const auto inferred = metadata.present
                        ? InferredScalarMultiplierNullspace{}
                        : inferScalarMultiplierConstantNullspace(
                              context,
                              cv,
                              std::unordered_set<VariableKey, VariableKeyHash>(
                                  nullspace_partners.begin(),
                                  nullspace_partners.end()),
                              comp_group);
                    if (metadata.present || inferred.certified) {
                        PropertyClaim ns_claim;
                        ns_claim.kind = PropertyKind::Nullspace;
                        ns_claim.status = PropertyStatus::Likely;
                        ns_claim.confidence = metadata.present
                            ? metadata.confidence
                            : AnalysisConfidence::Medium;
                        ns_claim.certification_class =
                            CertificationClass::NotCertified;
                        ns_claim.evidence_level = metadata.present
                            ? EvidenceLevel::DescriptorHint
                            : EvidenceLevel::StructuralMetadata;
                        ns_claim.nullspace_evidence_kind = metadata.present
                            ? metadata.evidence_kind
                            : NullspaceEvidenceKind::SymbolicOperatorIdentity;
                        ns_claim.field = constraint_fid;
                        ns_claim.component = cv.component;
                        ns_claim.domain = DomainKind::Cell;
                        ns_claim.variables.push_back(
                            cv.component >= 0
                                ? VariableKey::field(constraint_fid, cv.component)
                                : VariableKey::field(constraint_fid));
                        ns_claim.description =
                            metadata.present
                                ? ("Constraint field " +
                                   std::to_string(constraint_fid) +
                                   " in saddle-point system has declared multiplier"
                                   " nullspace metadata")
                                : ("Scalar constraint field " +
                                   std::to_string(constraint_fid) +
                                   " in saddle-point system has structural"
                                   " multiplier-gauge constant nullspace");
                        ns_claim.nullspace_family = metadata.present
                            ? metadata.family
                            : NullspaceFamily::ScalarConstant;
                        ns_claim.claim_origin = "MixedOperatorAnalyzer";
                        ns_claim.addEvidence("MixedOperatorAnalyzer",
                            metadata.present ? metadata.evidence
                                             : inferred.evidence,
                            ns_claim.confidence);
                        report.claims.push_back(std::move(ns_claim));
                        emitted_nullspace_fields.insert(constraint_fid);
                    }
                }
            }
        }
    }

    // =====================================================================
    // FALLBACK PATH: FormulationRecords
    // Emits one MixedSaddlePoint claim per constraint field with a SINGLE
    // variable (the constraint). Without block expressions we can verify
    // that a field is a constraint (no gradient operator) but NOT which
    // primal-vector field it couples with. Rather than fabricating pair
    // structure that downstream passes (SpaceCompatibilityAnalyzer,
    // InfSupAnalyzer) would consume as verified, we emit only what's
    // known. Downstream passes handle single-variable claims:
    //   - InfSupAnalyzer: emits generic InfSupCondition::Required
    //   - SpaceCompatibilityAnalyzer: skips (no pair to check)
    // =====================================================================
    {
        const auto& records = context.formulationRecords();

        FormStructureAnalyzer fsa;

        for (const auto& rec : records) {
            if (!rec.residual_expr) continue;
            if (rec.active_fields.size() < 2) continue;

            // Collect ALL primal-vector and constraint fields
            std::vector<FieldId> primal_vector_fids;
            std::vector<FieldId> constraint_fids;

            for (FieldId fid : rec.active_fields) {
                auto fs = fsa.analyzeField(*rec.residual_expr, fid);
                if (fs.occurrence_count == 0) continue;

                if (fs.value_dimension <= 1 &&
                    fs.has_absolute_value &&
                    !fs.has_gradient && !fs.has_sym_grad &&
                    !fs.has_stabilization) {
                    constraint_fids.push_back(fid);
                }

                if (fs.value_dimension > 1 &&
                    (fs.has_gradient || fs.has_sym_grad)) {
                    primal_vector_fids.push_back(fid);
                }
            }

            if (primal_vector_fids.empty() || constraint_fids.empty()) continue;

            // Emit one claim per constraint field (single variable, no
            // fabricated pair). The claim asserts: "this field is a
            // constraint/multiplier in a saddle-point system."
            for (FieldId cfid : constraint_fids) {
                // Skip if primary path already covered this constraint
                if (emitted_nullspace_fields.count(cfid)) continue;

                PropertyClaim claim;
                claim.kind = PropertyKind::MixedSaddlePoint;
                claim.status = PropertyStatus::Likely;
                claim.confidence = AnalysisConfidence::Low;
                claim.certification_class = CertificationClass::NotCertified;
                claim.evidence_level = EvidenceLevel::SyntaxPattern;
                claim.definiteness_interpretation =
                    DefinitenessInterpretation::SaddlePointExpected;
                claim.domain = DomainKind::Cell;
                claim.variables.push_back(VariableKey::field(cfid));
                claim.claim_origin = "MixedOperatorAnalyzer";

                claim.description =
                    "Formulation '" + rec.operator_tag +
                    "' has saddle-point structure: scalar field " +
                    std::to_string(cfid) +
                    " (constraint/multiplier, no diagonal elliptic"
                    " block)";
                claim.addEvidence("MixedOperatorAnalyzer",
                    "Constraint field " + std::to_string(cfid) +
                    " has no own gradient operator (no block structure"
                    " available to verify specific primal-vector partner)");
                report.claims.push_back(std::move(claim));

                // Nullspace: one per constraint field.  Explicit metadata is
                // accepted, and residual-only records can still certify the
                // multiplier gauge when the full residual exposes the p*div(v)
                // and q*div(u) pattern with no nonzero multiplier reaction.
                if (!emitted_nullspace_fields.count(cfid)) {
                    const auto metadata = declaredNullspaceForField(
                        context, cfid, -1, nullptr);
                    const auto inferred = metadata.present
                        ? InferredScalarMultiplierNullspace{}
                        : inferScalarMultiplierConstantNullspaceFromResidual(
                              context, *rec.residual_expr, cfid, primal_vector_fids);
                    if (!metadata.present && !inferred.certified) {
                        continue;
                    }
                    PropertyClaim ns_claim;
                    ns_claim.kind = PropertyKind::Nullspace;
                    ns_claim.status = PropertyStatus::Likely;
                    ns_claim.confidence = metadata.present
                        ? metadata.confidence
                        : AnalysisConfidence::Medium;
                    ns_claim.certification_class =
                        CertificationClass::NotCertified;
                    ns_claim.evidence_level = metadata.present
                        ? EvidenceLevel::DescriptorHint
                        : EvidenceLevel::SyntaxPattern;
                    ns_claim.nullspace_evidence_kind = metadata.present
                        ? metadata.evidence_kind
                        : NullspaceEvidenceKind::DescriptorHint;
                    ns_claim.field = cfid;
                    ns_claim.component = -1;
                    ns_claim.domain = DomainKind::Cell;
                    ns_claim.variables.push_back(
                        VariableKey::field(cfid));
                    ns_claim.description =
                        metadata.present
                            ? ("Constraint field " + std::to_string(cfid) +
                               " in saddle-point system has declared multiplier"
                               " nullspace metadata")
                            : ("Scalar constraint field " +
                               std::to_string(cfid) +
                               " in residual-defined saddle-point system has"
                               " structural multiplier-gauge constant nullspace");
                    ns_claim.nullspace_family = metadata.present
                        ? metadata.family
                        : NullspaceFamily::ScalarConstant;
                    ns_claim.claim_origin = "MixedOperatorAnalyzer";
                    ns_claim.addEvidence("MixedOperatorAnalyzer",
                        metadata.present ? metadata.evidence
                                         : inferred.evidence,
                        ns_claim.confidence);
                    report.claims.push_back(std::move(ns_claim));
                    emitted_nullspace_fields.insert(cfid);
                }
            }
        }
    }

    emitSchurSummaryHooks(context, report);
}

} // namespace analysis
} // namespace FE
} // namespace svmp
