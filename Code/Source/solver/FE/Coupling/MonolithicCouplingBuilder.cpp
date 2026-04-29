/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/MonolithicCouplingBuilder.h"

#include "Coupling/CouplingGraph.h"
#include "Core/FEException.h"
#include "Systems/FormsInstaller.h"

#include <algorithm>
#include <string>

namespace svmp {
namespace FE {
namespace coupling {

CouplingValidationResult MonolithicCouplingBuilder::validateDeclarations(
    const CouplingContext& context,
    std::span<const CouplingContractDeclaration> declarations) const
{
    CouplingGraph graph;
    return graph.buildDeclarationGraph(context, declarations);
}

ResolvedCouplingFormContribution MonolithicCouplingBuilder::resolveFormContribution(
    const CouplingContext& context,
    const CouplingFormContribution& contribution) const
{
    FE_THROW_IF(contribution.contribution_name.empty(), InvalidArgumentException,
                "coupling form contribution requires a contribution name");
    FE_THROW_IF(contribution.origin.empty(), InvalidArgumentException,
                "coupling form contribution requires a diagnostic origin");
    FE_THROW_IF(contribution.field_uses.empty(), InvalidArgumentException,
                "coupling form contribution requires at least one primary field use");
    FE_THROW_IF(!contribution.residual.isValid(), InvalidArgumentException,
                "coupling form contribution requires a residual expression");

    ResolvedCouplingFormContribution resolved;
    resolved.contribution_name = contribution.contribution_name;
    resolved.origin = contribution.origin;
    resolved.operator_name = contribution.operator_name;
    resolved.install_options = contribution.install_options;

    const systems::FESystem* owning_system = nullptr;
    auto resolve_field = [&](const CouplingFieldUse& use) {
        const auto ref = context.field(use.participant_name, use.field_name);
        if (owning_system == nullptr) {
            owning_system = ref.system;
            resolved.system_name = ref.system_name;
        }
        FE_THROW_IF(ref.system != owning_system, InvalidArgumentException,
                    "monolithic coupling form fields must resolve to one owning system");
        return ref.field_id;
    };

    for (const auto& use : contribution.field_uses) {
        const auto field = resolve_field(use);
        FE_THROW_IF(std::find(resolved.fields.begin(), resolved.fields.end(), field) !=
                        resolved.fields.end(),
                    InvalidArgumentException,
                    "duplicate primary field use in coupling form contribution");
        resolved.fields.push_back(field);
    }

    resolved.install_options.extra_trial_fields.clear();
    for (const auto& use : contribution.extra_trial_field_uses) {
        const auto field = resolve_field(use);
        FE_THROW_IF(std::find(resolved.fields.begin(), resolved.fields.end(), field) !=
                        resolved.fields.end(),
                    InvalidArgumentException,
                    "extra trial field overlaps a primary coupling form field");
        FE_THROW_IF(std::find(resolved.extra_trial_fields.begin(),
                              resolved.extra_trial_fields.end(),
                              field) != resolved.extra_trial_fields.end(),
                    InvalidArgumentException,
                    "duplicate extra trial field use in coupling form contribution");
        resolved.extra_trial_fields.push_back(field);
        resolved.install_options.extra_trial_fields.push_back(field);
    }

    resolved.residual = contribution.residual;
    return resolved;
}

CouplingFormAnalysisMetadata MonolithicCouplingBuilder::installResolvedFormContribution(
    systems::FESystem& system,
    const ResolvedCouplingFormContribution& contribution) const
{
    analysis::FormAnalysisBridgeOptions bridge_options;
    bridge_options.contribution_name = contribution.contribution_name;
    bridge_options.origin = contribution.origin;
    bridge_options.system_name = contribution.system_name;
    bridge_options.geometry_sensitivity =
        contribution.install_options.compiler_options.geometry_sensitivity;

    const auto installed = systems::installFormulationWithMetadata(
        system,
        contribution.operator_name,
        contribution.fields,
        contribution.residual,
        contribution.install_options,
        bridge_options);

    return adaptFormAnalysisMetadata(installed.analysis);
}

std::vector<CouplingFormAnalysisMetadata> MonolithicCouplingBuilder::installFormContributions(
    systems::FESystem& system,
    const CouplingContext& context,
    std::span<const CouplingFormContribution> contributions) const
{
    std::vector<CouplingFormAnalysisMetadata> installed;
    installed.reserve(contributions.size());
    for (const auto& contribution : contributions) {
        const auto resolved = resolveFormContribution(context, contribution);
        installed.push_back(installResolvedFormContribution(system, resolved));
    }
    return installed;
}

CouplingFormAnalysisMetadata MonolithicCouplingBuilder::adaptFormAnalysisMetadata(
    const analysis::FormContributionAnalysisMetadata& metadata)
{
    CouplingFormAnalysisMetadata adapted;
    adapted.contribution_name = metadata.contribution_name;
    adapted.origin = metadata.origin;
    adapted.system_name = metadata.system_name;
    adapted.operator_name = metadata.operator_tag;
    adapted.installed_fields = metadata.installed_fields;
    adapted.feature_gates = metadata.feature_gates;

    adapted.installed_dependencies.reserve(metadata.installed_dependencies.size());
    for (const auto& dependency : metadata.installed_dependencies) {
        adapted.installed_dependencies.push_back(CouplingInstalledDependency{
            .residual_row = dependency.residual_row,
            .dependency = dependency.dependency,
            .mode = CouplingDependencyMode::ImplicitMonolithic,
            .domain = dependency.domain,
            .contributes_matrix_block = dependency.contributes_matrix_block,
            .contributes_vector = dependency.contributes_vector,
            .provider = dependency.provider,
        });
    }

    adapted.installed_blocks.reserve(metadata.installed_blocks.size());
    for (const auto& block : metadata.installed_blocks) {
        adapted.installed_blocks.push_back(CouplingInstalledBlockProvenance{
            .residual_row = block.residual_row,
            .dependency = block.dependency,
            .domains = block.domains,
            .has_matrix = block.has_matrix,
            .has_vector = block.has_vector,
        });
    }

    return adapted;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
