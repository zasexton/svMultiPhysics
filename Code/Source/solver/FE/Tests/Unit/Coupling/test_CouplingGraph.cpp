#include "Coupling/CouplingGraph.h"

#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

const systems::FESystem* graphSystemToken()
{
    return reinterpret_cast<const systems::FESystem*>(1);
}

CouplingContext graphContext()
{
    const auto* system = graphSystemToken();
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    const CouplingRegionRef surface{
        .participant_name = "left",
        .system_name = "system",
        .system = system,
        .region_name = "surface",
        .kind = CouplingRegionKind::Boundary,
        .marker = 4,
    };

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "left",
        .system_name = "system",
        .system = system,
    });
    builder.addField({
        .participant_name = "left",
        .system_name = "system",
        .system = system,
        .field_name = "primary",
        .field_id = 1,
        .space = space,
        .components = 1,
    });
    builder.addRegion(surface);
    builder.addSharedRegion(SharedRegionRef{
        .name = "interface",
        .required_region_kind = CouplingRegionKind::Boundary,
        .participant_regions = {surface},
    });
    return builder.build();
}

CouplingContext twoParticipantGraphContext()
{
    const auto* system = graphSystemToken();
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "left",
        .system_name = "system",
        .system = system,
    });
    builder.addParticipant({
        .participant_name = "right",
        .system_name = "system",
        .system = system,
    });
    builder.addField({
        .participant_name = "left",
        .system_name = "system",
        .system = system,
        .field_name = "primary",
        .field_id = 1,
        .space = space,
        .components = 1,
    });
    builder.addField({
        .participant_name = "right",
        .system_name = "system",
        .system = system,
        .field_name = "primary",
        .field_id = 2,
        .space = space,
        .components = 1,
    });
    return builder.build();
}

CouplingContractDeclaration graphDeclaration()
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
    declaration.regions.push_back({
        .participant_name = "left",
        .region_name = "surface",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    declaration.shared_regions.push_back({
        .shared_region_name = "interface",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    return declaration;
}

CouplingContractDeclaration twoParticipantDependencyDeclaration()
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
    declaration.fields.push_back({.participant_name = "right", .field_name = "primary"});
    declaration.dependencies.push_back(CouplingResidualDependency{
        .residual_row = {
            .kind = CouplingVariableKind::Field,
            .participant_name = "right",
            .name = "primary",
        },
        .dependency = {
            .kind = CouplingVariableKind::Field,
            .participant_name = "left",
            .name = "primary",
        },
    });
    declaration.expected_blocks.push_back(CouplingBlockExpectation{
        .residual_row = declaration.dependencies.back().residual_row,
        .dependency = declaration.dependencies.back().dependency,
    });
    return declaration;
}

CouplingFormAnalysisMetadata installedDependencyMetadata()
{
    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "generic_cell_coupling";
    metadata.origin = "CouplingGraphTest";
    metadata.system_name = "system";
    metadata.installed_fields = {2, 1};
    metadata.installed_dependencies.push_back(CouplingInstalledDependency{
        .residual_row = analysis::VariableKey::field(2),
        .dependency = analysis::VariableKey::field(1),
        .mode = CouplingDependencyMode::ImplicitMonolithic,
        .domain = analysis::DomainKind::Cell,
        .contributes_matrix_block = true,
        .contributes_vector = true,
        .provider = "FormsInstaller",
    });
    metadata.installed_blocks.push_back(CouplingInstalledBlockProvenance{
        .residual_row = analysis::VariableKey::field(2),
        .dependency = analysis::VariableKey::field(1),
        .domains = {analysis::DomainKind::Cell},
        .has_matrix = true,
        .has_vector = true,
    });
    return metadata;
}

CouplingValidationResult buildGraph(const CouplingContext& context,
                                    const CouplingContractDeclaration& declaration)
{
    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    return graph.buildDeclarationGraph(
        context,
        std::span<const CouplingContractDeclaration>(declarations));
}

CouplingValidationResult buildFinalizedGraph(
    const CouplingContext& context,
    const CouplingContractDeclaration& declaration,
    const std::vector<CouplingFormAnalysisMetadata>& installed_forms)
{
    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    return graph.buildFinalizedGraph(
        context,
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed_forms));
}

} // namespace

TEST(CouplingGraph, AcceptsResolvableRequiredContextReferences)
{
    const auto validation = buildGraph(graphContext(), graphDeclaration());
    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(CouplingGraph, RejectsMissingRequiredContextReferences)
{
    auto declaration = graphDeclaration();
    declaration.participants.push_back({.participant_name = "right"});
    declaration.fields.push_back({.participant_name = "right", .field_name = "primary"});
    declaration.regions.push_back({.participant_name = "right", .region_name = "surface"});
    declaration.shared_regions.push_back({.shared_region_name = "other_interface"});

    const auto validation = buildGraph(graphContext(), declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("required coupling participant is missing"), std::string::npos);
    EXPECT_NE(text.find("required coupling field is missing"), std::string::npos);
    EXPECT_NE(text.find("required coupling region is missing"), std::string::npos);
    EXPECT_NE(text.find("required shared region is missing"), std::string::npos);
}

TEST(CouplingGraph, AllowsAbsentOptionalContextReferences)
{
    auto declaration = graphDeclaration();
    declaration.participants.push_back({
        .participant_name = "right",
        .requirement = CouplingRequirement::Optional,
    });
    declaration.fields.push_back({
        .participant_name = "right",
        .field_name = "primary",
        .requirement = CouplingRequirement::Optional,
    });
    declaration.regions.push_back({
        .participant_name = "right",
        .region_name = "surface",
        .requirement = CouplingRequirement::Optional,
    });
    declaration.shared_regions.push_back({
        .shared_region_name = "other_interface",
        .requirement = CouplingRequirement::Optional,
    });

    const auto validation = buildGraph(graphContext(), declaration);
    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(CouplingGraph, RejectsRegionKindMismatches)
{
    auto declaration = graphDeclaration();
    declaration.regions[0].required_region_kind = CouplingRegionKind::InterfaceFace;

    const auto validation = buildGraph(graphContext(), declaration);
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("region kind"), std::string::npos);
}

TEST(CouplingGraph, RejectsSharedRegionKindMismatches)
{
    auto declaration = graphDeclaration();
    declaration.shared_regions[0].required_region_kind = CouplingRegionKind::InterfaceFace;

    const auto validation = buildGraph(graphContext(), declaration);
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("shared-region participant mapping"),
              std::string::npos);
}

TEST(CouplingGraph, AcceptsInstalledImplicitDependencyAndExpectedBlock)
{
    const std::vector<CouplingFormAnalysisMetadata> installed_forms{
        installedDependencyMetadata()};

    const auto validation = buildFinalizedGraph(
        twoParticipantGraphContext(),
        twoParticipantDependencyDeclaration(),
        installed_forms);

    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(CouplingGraph, RejectsDeclaredImplicitDependencyMissingFromInstalledMetadata)
{
    const auto validation = buildFinalizedGraph(
        twoParticipantGraphContext(),
        twoParticipantDependencyDeclaration(),
        {});

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find(
                  "declared implicit coupling dependency is not reported"),
              std::string::npos);
}

TEST(CouplingGraph, RejectsExpectedBlockWithoutInstalledMatrixEvidence)
{
    auto metadata = installedDependencyMetadata();
    metadata.installed_dependencies[0].contributes_matrix_block = false;
    metadata.installed_blocks.clear();

    const std::vector<CouplingFormAnalysisMetadata> installed_forms{metadata};
    const auto validation = buildFinalizedGraph(
        twoParticipantGraphContext(),
        twoParticipantDependencyDeclaration(),
        installed_forms);

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find(
                  "expected monolithic block is missing installed matrix evidence"),
              std::string::npos);
}

TEST(CouplingGraph, RejectsExpectedMatrixBlockForExternalDependency)
{
    auto declaration = twoParticipantDependencyDeclaration();
    declaration.dependencies[0].mode = CouplingDependencyMode::ExternalLagged;

    const std::vector<CouplingFormAnalysisMetadata> installed_forms{
        installedDependencyMetadata()};
    const auto validation = buildFinalizedGraph(
        twoParticipantGraphContext(),
        declaration,
        installed_forms);

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find(
                  "external or lagged coupling dependency cannot require"),
              std::string::npos);
}

TEST(CouplingGraph, RejectsUndeclaredInstalledBlocks)
{
    auto declaration = twoParticipantDependencyDeclaration();
    declaration.dependencies.clear();
    declaration.expected_blocks.clear();

    const std::vector<CouplingFormAnalysisMetadata> installed_forms{
        installedDependencyMetadata()};
    const auto validation = buildFinalizedGraph(
        twoParticipantGraphContext(),
        declaration,
        installed_forms);

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find(
                  "installed coupling block has no declared implicit dependency"),
              std::string::npos);
}

TEST(CouplingGraph, RejectsExpectedZeroBlockWithInstalledEvidence)
{
    auto declaration = twoParticipantDependencyDeclaration();
    declaration.expected_blocks[0].expected_nonzero = false;

    const std::vector<CouplingFormAnalysisMetadata> installed_forms{
        installedDependencyMetadata()};
    const auto validation = buildFinalizedGraph(
        twoParticipantGraphContext(),
        declaration,
        installed_forms);

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find(
                  "expected zero coupling block has installed evidence"),
              std::string::npos);
}
