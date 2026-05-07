/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/CouplingGraphAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

#include <set>
#include <string>
#include <tuple>

namespace svmp {
namespace FE {
namespace analysis {

namespace {

struct CouplingEdgeKey {
    VariableKey a;
    VariableKey b;
    DomainKind domain{DomainKind::Cell};
    int marker{-1};
    std::string contribution_id;
    std::string operator_tag;

    bool operator<(const CouplingEdgeKey& other) const noexcept
    {
        return std::tie(a, b, domain, marker,
                        contribution_id, operator_tag) <
               std::tie(other.a, other.b, other.domain, other.marker,
                        other.contribution_id, other.operator_tag);
    }
};

[[nodiscard]] int scopedMarker(DomainKind domain,
                               int boundary_marker,
                               int interface_marker) noexcept
{
    return domain == DomainKind::InterfaceFace
        ? interface_marker
        : boundary_marker;
}

} // namespace

std::string CouplingGraphAnalyzer::name() const {
    return "CouplingGraphAnalyzer";
}

void CouplingGraphAnalyzer::run(const ProblemAnalysisContext& context,
                                ProblemAnalysisReport& report) const
{
    // Track emitted coupling edges by variable pair and mathematical scope.
    std::set<CouplingEdgeKey> emitted;

    auto emit_coupling = [&](const VariableKey& a, const VariableKey& b,
                             DomainKind domain, int marker,
                             const std::string& contribution_id,
                             const std::string& operator_tag,
                             const std::string& source_tag)
    {
        if (a == b) return;

        // Canonical ordering to avoid duplicates
        CouplingEdgeKey key;
        if (a < b) {
            key.a = a;
            key.b = b;
        } else {
            key.a = b;
            key.b = a;
        }
        key.domain = domain;
        key.marker = marker;
        key.contribution_id = contribution_id;
        key.operator_tag = operator_tag;
        if (!emitted.insert(key).second) return;

        PropertyClaim claim;
        claim.kind = PropertyKind::CoupledSystemStructure;
        claim.status = PropertyStatus::Exact;
        claim.confidence = AnalysisConfidence::High;
        claim.domain = domain;
        claim.variables.push_back(key.a);
        claim.variables.push_back(key.b);
        claim.tested_block_id = operator_tag.empty()
            ? contribution_id
            : operator_tag;
        claim.description = "Variables are coupled through " + source_tag;
        claim.addEvidence("CouplingGraphAnalyzer",
            "Coupling discovered in " + source_tag +
            ", marker=" + std::to_string(marker) +
            ", contribution_id='" + contribution_id + "'");
        report.claims.push_back(std::move(claim));
    };

    auto emit_interface = [&](const VariableKey& a, const VariableKey& b,
                              const std::string& source_tag)
    {
        PropertyClaim claim;
        claim.kind = PropertyKind::InterfaceCondition;
        claim.status = PropertyStatus::Exact;
        claim.confidence = AnalysisConfidence::High;
        claim.domain = DomainKind::InterfaceFace;
        claim.variables.push_back(a);
        claim.variables.push_back(b);
        claim.description =
            "Variables coupled only through interface/global operator: " +
            source_tag;
        claim.addEvidence("CouplingGraphAnalyzer",
            "Interface-only coupling in " + source_tag);
        report.claims.push_back(std::move(claim));
    };

    const auto dependency_domain = [](const VariableKey& dependency) {
        switch (dependency.kind) {
            case VariableKind::BoundaryFunctional:
                return DomainKind::CoupledBoundary;
            case VariableKind::AuxiliaryState:
            case VariableKind::AuxiliaryInput:
            case VariableKind::AuxiliaryOutput:
                return DomainKind::AuxiliaryCoupling;
            case VariableKind::GlobalScalar:
                return DomainKind::Global;
            case VariableKind::FieldComponent:
                return DomainKind::Cell;
        }
        return DomainKind::Cell;
    };

    const auto coupling_domain = [&](const VariableKey& a, const VariableKey& b) {
        if (b.kind != VariableKind::FieldComponent) {
            return dependency_domain(b);
        }
        if (a.kind != VariableKind::FieldComponent) {
            return dependency_domain(a);
        }
        return DomainKind::Cell;
    };

    const auto emit_record_dependency =
        [&](const FormulationRecord& rec,
            const VariableKey& dependency,
            const std::string& source_tag)
    {
        for (FieldId fid : rec.active_fields) {
            emit_coupling(VariableKey::field(fid),
                          dependency,
                          dependency_domain(dependency),
                          -1,
                          {},
                          rec.operator_tag,
                          source_tag + " in '" + rec.operator_tag + "'");
        }
    };

    const auto emit_formulation_record = [&](const FormulationRecord& rec) {
        for (const auto& [test_fid, trial_fid] : rec.block_couplings) {
            auto a = VariableKey::field(test_fid);
            auto b = VariableKey::field(trial_fid);
            emit_coupling(a, b, DomainKind::Cell,
                          -1, {}, rec.operator_tag,
                          "formulation '" + rec.operator_tag + "'");
        }

        for (const auto& [va, vb] : rec.variable_couplings) {
            emit_coupling(va, vb, coupling_domain(va, vb),
                          -1, {}, rec.operator_tag,
                          "formulation '" + rec.operator_tag + "'");
        }

        for (const auto& bf : rec.boundary_functional_dependencies) {
            emit_record_dependency(rec, bf, "boundary functional");
        }

        for (const auto& aux : rec.auxiliary_state_dependencies) {
            emit_record_dependency(rec, aux, "auxiliary state");
        }

        for (const auto& input : rec.auxiliary_input_dependencies) {
            emit_record_dependency(rec, input, "auxiliary input");
        }

        for (const auto& output : rec.auxiliary_output_dependencies) {
            emit_record_dependency(rec, output, "auxiliary output");
        }
    };

    // =====================================================================
    // PRIMARY PATH: Consume ContributionDescriptors
    // =====================================================================
    const auto& contributions = context.contributions();
    if (!contributions.empty()) {
        for (const auto& contrib : contributions) {
            // Cross-product of test × trial variables for coupling edges
            const int marker = scopedMarker(contrib.domain,
                                            contrib.boundary_marker,
                                            contrib.interface_marker);
            for (const auto& tv : contrib.test_variables) {
                for (const auto& tr : contrib.trial_variables) {
                    DomainKind dom = contrib.domain;

                    // Detect interface-only coupling
                    bool is_interface_only =
                        (dom == DomainKind::InterfaceFace ||
                         dom == DomainKind::Global);

                    if (is_interface_only) {
                        emit_interface(tv, tr,
                            "contribution '" + contrib.operator_tag + "'");
                    }
                    emit_coupling(tv, tr, dom,
                        marker,
                        contrib.contribution_id,
                        contrib.operator_tag,
                        "contribution '" + contrib.operator_tag + "'");
                }

                // Related variables (data dependencies)
                for (const auto& rv : contrib.related_variables) {
                    emit_coupling(tv, rv, contrib.domain,
                        marker,
                        contrib.contribution_id,
                        contrib.operator_tag,
                        "contribution '" + contrib.operator_tag + "' (data dep)");
                }
            }

            // Non-FE variable couplings are already emitted above through
            // the test x trial cross-product and related variables.
        }

        // Also include fallback paths that may carry additional information

        // --- Formulation records for block_couplings/variable_couplings/
        //     non-field dependency lists
        //     that might not have been lowered to contributions yet ---
        for (const auto& rec : context.formulationRecords()) {
            emit_formulation_record(rec);
        }

        // --- BC descriptors ---
        for (const auto& bc : context.bcDescriptors()) {
            for (const auto& rv : bc.related_variables) {
                const std::string marker_desc =
                    (bc.domain == DomainKind::InterfaceFace && bc.interface_marker >= 0)
                        ? ("interface " + std::to_string(bc.interface_marker))
                        : ("boundary " + std::to_string(bc.boundary_marker));
                emit_coupling(bc.primary_variable, rv, bc.domain,
                              scopedMarker(bc.domain,
                                           bc.boundary_marker,
                                           bc.interface_marker),
                              {},
                              "bc",
                              "condition on " + marker_desc);
            }
        }
        return;
    }

    // =====================================================================
    // FALLBACK PATH: FormulationRecords
    // =====================================================================

    // --- Formulation records ---
    for (const auto& rec : context.formulationRecords()) {
        emit_formulation_record(rec);
    }

    // --- BC descriptors ---
    for (const auto& bc : context.bcDescriptors()) {
        // BCs with related variables create couplings
        for (const auto& rv : bc.related_variables) {
            const std::string marker_desc =
                (bc.domain == DomainKind::InterfaceFace && bc.interface_marker >= 0)
                    ? ("interface " + std::to_string(bc.interface_marker))
                    : ("boundary " + std::to_string(bc.boundary_marker));
            emit_coupling(bc.primary_variable, rv, bc.domain,
                          scopedMarker(bc.domain,
                                       bc.boundary_marker,
                                       bc.interface_marker),
                          {},
                          "bc",
                          "condition on " + marker_desc);
        }
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
