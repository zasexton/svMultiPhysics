/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_PROBLEM_ANALYSIS_CONTEXT_H
#define SVMP_FE_ANALYSIS_PROBLEM_ANALYSIS_CONTEXT_H

/**
 * @file ProblemAnalysisContext.h
 * @brief Central aggregation point for all problem metadata consumed by analyzers
 *
 * ProblemAnalysisContext collects:
 *  - FormulationRecords (operator structure, active fields, block couplings)
 *  - BoundaryConditionDescriptors (trace kind, enforcement, anchoring)
 *  - TopologyAnalysisContext (connected components, boundary-region mapping)
 *  - ConstraintAnalysisSummary (constrained DOFs, conflicts)
 *  - Field/space descriptors (FieldId → SpaceSignature map)
 *
 * All sections are optional/empty by default so the context can be built
 * incrementally as each subsystem populates its portion.
 *
 * @see ProblemAnalyzer for the orchestrator that consumes this context
 * @see ProblemAnalysisTypes.h for the output types
 */

#include "Core/Types.h"
#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/FormulationRecord.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/TopologyAnalysisContext.h"
#include "Analysis/InterfaceTopologyContext.h"
#include "Analysis/ConstraintAnalysisSummary.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Backends/Utils/BackendOptions.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

/**
 * @brief Field/space descriptor for analysis purposes
 *
 * Lightweight summary of a field's function space characteristics,
 * decoupled from the full SpaceSignature to avoid heavy includes.
 */
struct FieldDescriptor {
    FieldId field_id{INVALID_FIELD_ID};
    std::string name;                    ///< Human-readable field name (e.g. "velocity", "pressure")
    FieldType field_type{FieldType::Scalar};
    int value_dimension{1};              ///< 1=scalar, 2/3=vector, 4/6/9=tensor
    int topological_dimension{0};        ///< Spatial dimension of the mesh
    int polynomial_order{1};
    Continuity continuity{Continuity::C0};
    DomainKind domain{DomainKind::Cell};
    int interface_marker{-1};

    /// True when per-component DOF extraction is supported (ProductSpace / H1 vectors).
    /// False for vector-basis fields (HDiv/HCurl) where DOFs are on vector-valued
    /// basis functions and component extraction is not defined.
    /// Default true for backward compatibility with tests that don't set this.
    bool component_extractable{true};

    // Phase 21 extensions
    SpaceFamily space_family{SpaceFamily::Unknown};
    TraceCapabilityFlags trace_capabilities{TraceCapabilityFlags::None};
    bool has_exact_sequence_structure{false};   ///< HDiv/HCurl in a de Rham sequence
    bool supports_local_balance_closure{false}; ///< Spaces with conservative trace (HDiv)
};

/**
 * @brief Central metadata aggregation for problem analysis
 *
 * Default-constructible with all sections empty.  Populated incrementally
 * by FormsInstaller (formulations), BoundaryConditionManager (BCs),
 * SystemSetup (topology, constraints), CoupledBoundaryManager, etc.
 */
class ProblemAnalysisContext {
public:
    ProblemAnalysisContext() = default;

    // ---- Field descriptors (FE fields) ----

    void addFieldDescriptor(FieldDescriptor desc);
    [[nodiscard]] const FieldDescriptor* fieldDescriptor(FieldId id) const noexcept;
    [[nodiscard]] const std::vector<FieldDescriptor>& fieldDescriptors() const noexcept {
        return field_descriptors_;
    }

    // ---- Variable descriptors (generic: FE fields + aux states + boundary functionals + global scalars) ----

    void addVariableDescriptor(VariableDescriptor desc);
    [[nodiscard]] const VariableDescriptor* variableDescriptor(const VariableKey& key) const noexcept;
    [[nodiscard]] const std::vector<VariableDescriptor>& variableDescriptors() const noexcept {
        return variable_descriptors_;
    }

    // ---- Formulation records (Phase 2) ----

    void addFormulationRecord(FormulationRecord record);
    [[nodiscard]] const std::vector<FormulationRecord>& formulationRecords() const noexcept {
        return formulation_records_;
    }

    // ---- Normalized contributions (Phase 10) ----

    void addContribution(ContributionDescriptor desc);
    [[nodiscard]] const std::vector<ContributionDescriptor>& contributions() const noexcept {
        return contributions_;
    }

    // ---- BC descriptors (Phase 4) ----

    void addBCDescriptor(BoundaryConditionDescriptor desc);
    [[nodiscard]] const std::vector<BoundaryConditionDescriptor>& bcDescriptors() const noexcept {
        return bc_descriptors_;
    }

    // ---- Topology context (Phase 5) ----

    void setTopologyContext(TopologyAnalysisContext ctx);
    [[nodiscard]] const TopologyAnalysisContext* topologyContext() const noexcept {
        return topology_context_ ? &*topology_context_ : nullptr;
    }

    // ---- Interface topology (Phase 14) ----

    void setInterfaceTopologyContext(InterfaceTopologyContext ctx);
    [[nodiscard]] const InterfaceTopologyContext* interfaceTopologyContext() const noexcept {
        return interface_topology_context_ ? &*interface_topology_context_ : nullptr;
    }

    // ---- Constraint summary (Phase 6) ----

    void setConstraintSummary(ConstraintAnalysisSummary summary);
    [[nodiscard]] const ConstraintAnalysisSummary* constraintSummary() const noexcept {
        return constraint_summary_ ? &*constraint_summary_ : nullptr;
    }

    // ---- Optional numeric/discrete summaries (Phase 2 metadata contracts) ----

    void setAnalysisSummaries(AnalysisSummarySet summaries);
    void clearAnalysisSummaries();
    [[nodiscard]] const AnalysisSummarySet* analysisSummaries() const noexcept {
        return analysis_summaries_ ? &*analysis_summaries_ : nullptr;
    }
    [[nodiscard]] bool hasSummaryKind(AnalysisSummaryKind kind) const noexcept;
    [[nodiscard]] CertificationClass summaryCertificationOrUnknown(
        AnalysisSummaryKind kind,
        CertificationClass when_present) const noexcept;

    // ---- Optional solver choices for compatibility checks ----

    void setSolverOptions(backends::SolverOptions options);
    void clearSolverOptions();
    [[nodiscard]] const backends::SolverOptions* solverOptions() const noexcept {
        return solver_options_ ? &*solver_options_ : nullptr;
    }

    // ---- Version tracking for cache invalidation ----

    /// Monotonically increasing version; incremented on every mutation
    [[nodiscard]] std::uint64_t inputsVersion() const noexcept { return inputs_version_; }

    // ---- Convenience ----

    /// True if no metadata has been populated
    [[nodiscard]] bool empty() const noexcept;

private:
    void bumpVersion() noexcept { ++inputs_version_; }

    std::vector<FieldDescriptor> field_descriptors_;
    std::unordered_map<FieldId, std::size_t> field_id_to_index_;

    std::vector<VariableDescriptor> variable_descriptors_;

    std::vector<FormulationRecord> formulation_records_;
    std::vector<ContributionDescriptor> contributions_;
    std::vector<BoundaryConditionDescriptor> bc_descriptors_;
    std::optional<TopologyAnalysisContext> topology_context_;
    std::optional<InterfaceTopologyContext> interface_topology_context_;
    std::optional<ConstraintAnalysisSummary> constraint_summary_;
    std::optional<AnalysisSummarySet> analysis_summaries_;
    std::optional<backends::SolverOptions> solver_options_;

    std::uint64_t inputs_version_{0};
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_PROBLEM_ANALYSIS_CONTEXT_H
