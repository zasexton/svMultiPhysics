/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ProblemAnalysisContext.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace analysis {

// ============================================================================
// Field descriptors
// ============================================================================

void ProblemAnalysisContext::addFieldDescriptor(FieldDescriptor desc) {
    auto id = desc.field_id;
    auto it = field_id_to_index_.find(id);
    if (it != field_id_to_index_.end()) {
        field_descriptors_[it->second] = std::move(desc);
    } else {
        field_id_to_index_[id] = field_descriptors_.size();
        field_descriptors_.push_back(std::move(desc));
    }
    bumpVersion();
}

const FieldDescriptor* ProblemAnalysisContext::fieldDescriptor(FieldId id) const noexcept {
    auto it = field_id_to_index_.find(id);
    if (it == field_id_to_index_.end()) return nullptr;
    return &field_descriptors_[it->second];
}

// ============================================================================
// Variable descriptors
// ============================================================================

void ProblemAnalysisContext::addVariableDescriptor(VariableDescriptor desc) {
    // Check for existing descriptor with same key and update
    auto it = std::find_if(variable_descriptors_.begin(), variable_descriptors_.end(),
                           [&](const VariableDescriptor& vd) { return vd.key == desc.key; });
    if (it != variable_descriptors_.end()) {
        *it = std::move(desc);
    } else {
        variable_descriptors_.push_back(std::move(desc));
    }
    bumpVersion();
}

const VariableDescriptor* ProblemAnalysisContext::variableDescriptor(const VariableKey& key) const noexcept {
    auto it = std::find_if(variable_descriptors_.begin(), variable_descriptors_.end(),
                           [&](const VariableDescriptor& vd) { return vd.key == key; });
    if (it == variable_descriptors_.end()) return nullptr;
    return &*it;
}

// ============================================================================
// Formulation records (Phase 2)
// ============================================================================

void ProblemAnalysisContext::addFormulationRecord(FormulationRecord record) {
    formulation_records_.push_back(std::move(record));
    bumpVersion();
}

// ============================================================================
// Normalized contributions (Phase 10)
// ============================================================================

void ProblemAnalysisContext::addContribution(ContributionDescriptor desc) {
    contributions_.push_back(std::move(desc));
    bumpVersion();
}

// ============================================================================
// BC descriptors (Phase 4)
// ============================================================================

void ProblemAnalysisContext::addBCDescriptor(BoundaryConditionDescriptor desc) {
    bc_descriptors_.push_back(std::move(desc));
    bumpVersion();
}

// ============================================================================
// Topology context (Phase 5)
// ============================================================================

void ProblemAnalysisContext::setTopologyContext(TopologyAnalysisContext ctx) {
    topology_context_.emplace(std::move(ctx));
    bumpVersion();
}

// ============================================================================
// Interface topology (Phase 14)
// ============================================================================

void ProblemAnalysisContext::setInterfaceTopologyContext(InterfaceTopologyContext ctx) {
    interface_topology_context_.emplace(std::move(ctx));
    bumpVersion();
}

// ============================================================================
// Constraint summary (Phase 6)
// ============================================================================

void ProblemAnalysisContext::setConstraintSummary(ConstraintAnalysisSummary summary) {
    constraint_summary_.emplace(std::move(summary));
    bumpVersion();
}

// ============================================================================
// Optional numeric/discrete summaries (Phase 2 metadata contracts)
// ============================================================================

void ProblemAnalysisContext::setAnalysisSummaries(AnalysisSummarySet summaries) {
    analysis_summaries_.emplace(std::move(summaries));
    bumpVersion();
}

void ProblemAnalysisContext::clearAnalysisSummaries() {
    analysis_summaries_.reset();
    bumpVersion();
}

bool ProblemAnalysisContext::hasSummaryKind(AnalysisSummaryKind kind) const noexcept {
    return analysis_summaries_ && analysis_summaries_->has(kind);
}

CertificationClass ProblemAnalysisContext::summaryCertificationOrUnknown(
    AnalysisSummaryKind kind,
    CertificationClass when_present) const noexcept
{
    return analysis_summaries_
        ? analysis_summaries_->certificationOrUnknown(kind, when_present)
        : CertificationClass::Unknown;
}

// ============================================================================
// Optional solver choices for compatibility checks
// ============================================================================

void ProblemAnalysisContext::setSolverOptions(backends::SolverOptions options) {
    solver_options_.emplace(std::move(options));
    bumpVersion();
}

void ProblemAnalysisContext::clearSolverOptions() {
    solver_options_.reset();
    bumpVersion();
}

// ============================================================================
// Convenience
// ============================================================================

bool ProblemAnalysisContext::empty() const noexcept {
    return field_descriptors_.empty()
        && variable_descriptors_.empty()
        && formulation_records_.empty()
        && contributions_.empty()
        && bc_descriptors_.empty()
        && !topology_context_.has_value()
        && !interface_topology_context_.has_value()
        && !constraint_summary_.has_value()
        && (!analysis_summaries_.has_value() || analysis_summaries_->empty())
        && !solver_options_.has_value();
}

} // namespace analysis
} // namespace FE
} // namespace svmp
