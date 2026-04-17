/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/CouplingGraphAnalyzer.h"
#include "Analysis/ContributionDescriptor.h"

#include <set>

namespace svmp {
namespace FE {
namespace analysis {

std::string CouplingGraphAnalyzer::name() const {
    return "CouplingGraphAnalyzer";
}

void CouplingGraphAnalyzer::run(const ProblemAnalysisContext& context,
                                ProblemAnalysisReport& report) const
{
    // Track emitted coupling pairs to avoid duplicates.
    // Use a set of ordered VariableKey pairs.
    std::set<std::pair<VariableKey, VariableKey>> emitted;

    auto emit_coupling = [&](const VariableKey& a, const VariableKey& b,
                             DomainKind domain, const std::string& source_tag)
    {
        // Canonical ordering to avoid duplicates
        auto key = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
        if (!emitted.insert(key).second) return;

        PropertyClaim claim;
        claim.kind = PropertyKind::CoupledSystemStructure;
        claim.status = PropertyStatus::Exact;
        claim.confidence = AnalysisConfidence::High;
        claim.domain = domain;
        claim.variables.push_back(key.first);
        claim.variables.push_back(key.second);
        claim.description = "Variables are coupled through " + source_tag;
        claim.addEvidence("CouplingGraphAnalyzer",
            "Coupling discovered in " + source_tag);
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

    // =====================================================================
    // PRIMARY PATH: Consume ContributionDescriptors
    // =====================================================================
    const auto& contributions = context.contributions();
    if (!contributions.empty()) {
        for (const auto& contrib : contributions) {
            // Cross-product of test × trial variables for coupling edges
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
                        "contribution '" + contrib.operator_tag + "'");
                }

                // Related variables (data dependencies)
                for (const auto& rv : contrib.related_variables) {
                    emit_coupling(tv, rv, contrib.domain,
                        "contribution '" + contrib.operator_tag + "' (data dep)");
                }
            }

            // Detect CoupledSystemStructure when non-FE variables are involved
            bool has_non_fe = false;
            for (const auto& tv : contrib.test_variables) {
                if (tv.kind != VariableKind::FieldComponent) has_non_fe = true;
            }
            for (const auto& tv : contrib.trial_variables) {
                if (tv.kind != VariableKind::FieldComponent) has_non_fe = true;
            }
            for (const auto& rv : contrib.related_variables) {
                if (rv.kind != VariableKind::FieldComponent) has_non_fe = true;
            }

            // Non-FE variable couplings are already emitted above through
            // the test x trial cross-product and related variables.
        }

        // Also include fallback paths that may carry additional information

        // --- Formulation records for block_couplings/variable_couplings/
        //     boundary_functional_dependencies/auxiliary_state_dependencies
        //     that might not have been lowered to contributions yet ---
        for (const auto& rec : context.formulationRecords()) {
            for (const auto& [test_fid, trial_fid] : rec.block_couplings) {
                auto a = VariableKey::field(test_fid);
                auto b = VariableKey::field(trial_fid);
                emit_coupling(a, b, DomainKind::Cell,
                              "formulation '" + rec.operator_tag + "'");
            }

            for (const auto& [va, vb] : rec.variable_couplings) {
                emit_coupling(va, vb, DomainKind::Cell,
                              "formulation '" + rec.operator_tag + "'");
            }

            for (const auto& bf : rec.boundary_functional_dependencies) {
                for (FieldId fid : rec.active_fields) {
                    auto fk = VariableKey::field(fid);
                    emit_coupling(fk, bf, DomainKind::CoupledBoundary,
                                  "boundary functional in '" + rec.operator_tag + "'");
                }
            }

            for (const auto& aux : rec.auxiliary_state_dependencies) {
                for (FieldId fid : rec.active_fields) {
                    auto fk = VariableKey::field(fid);
                    emit_coupling(fk, aux, DomainKind::Cell,
                                  "auxiliary state in '" + rec.operator_tag + "'");
                }
            }
        }

        // --- BC descriptors ---
        for (const auto& bc : context.bcDescriptors()) {
            for (const auto& rv : bc.related_variables) {
                const std::string marker_desc =
                    (bc.domain == DomainKind::InterfaceFace && bc.interface_marker >= 0)
                        ? ("interface " + std::to_string(bc.interface_marker))
                        : ("boundary " + std::to_string(bc.boundary_marker));
                emit_coupling(bc.primary_variable, rv, bc.domain,
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
        // Block couplings (FE field pairs)
        for (const auto& [test_fid, trial_fid] : rec.block_couplings) {
            auto a = VariableKey::field(test_fid);
            auto b = VariableKey::field(trial_fid);
            emit_coupling(a, b, DomainKind::Cell,
                          "formulation '" + rec.operator_tag + "'");
        }

        // Generic variable couplings
        for (const auto& [va, vb] : rec.variable_couplings) {
            emit_coupling(va, vb, DomainKind::Cell,
                          "formulation '" + rec.operator_tag + "'");
        }

        // Boundary functional dependencies couple to active fields
        for (const auto& bf : rec.boundary_functional_dependencies) {
            for (FieldId fid : rec.active_fields) {
                auto fk = VariableKey::field(fid);
                emit_coupling(fk, bf, DomainKind::CoupledBoundary,
                              "boundary functional in '" + rec.operator_tag + "'");
            }
        }

        // Auxiliary state dependencies couple to active fields
        for (const auto& aux : rec.auxiliary_state_dependencies) {
            for (FieldId fid : rec.active_fields) {
                auto fk = VariableKey::field(fid);
                emit_coupling(fk, aux, DomainKind::Cell,
                              "auxiliary state in '" + rec.operator_tag + "'");
            }
        }
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
                          "condition on " + marker_desc);
        }
    }
}

} // namespace analysis
} // namespace FE
} // namespace svmp
