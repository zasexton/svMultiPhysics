#include "Coupling/CouplingGraph.h"
#include "Coupling/PartitionedCouplingPlanGenerator.h"

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

CouplingEndpointRef graphFieldEndpoint(std::string participant)
{
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::Field,
        .participant_name = std::move(participant),
        .endpoint_name = "primary",
        .temporal = CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::Current,
        },
    };
}

CouplingContractDeclaration partitionedGraphDeclaration()
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
    declaration.fields.push_back({.participant_name = "right", .field_name = "primary"});
    declaration.partitioned_exchange_declarations.push_back(CouplingExchangeDeclaration{
        .producer_port = {
            .contract_instance_name = "generic_instance",
            .port_name = "left_out",
        },
        .consumer_port = {
            .contract_instance_name = "generic_instance",
            .port_name = "right_in",
        },
        .value = CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        .producer = graphFieldEndpoint("left"),
        .consumer = graphFieldEndpoint("right"),
        .transfer = CouplingTransferDeclaration{
            .kind = CouplingTransferKind::Identity,
        },
    });
    declaration.group_hints.push_back(CouplingGroupHint{
        .name = "pair",
        .participant_names = {"left", "right"},
    });
    return declaration;
}

CouplingFormAnalysisMetadata installedDependencyMetadata()
{
    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "generic_cell_coupling";
    metadata.origin = "CouplingGraphTest";
    metadata.system_name = "system";
    metadata.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledDependencies,
        analysis::FormBridgeFeatureStatus::Available,
        "Synthetic installed dependency fixture"});
    metadata.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledBlocks,
        analysis::FormBridgeFeatureStatus::Available,
        "Synthetic installed block fixture"});
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

CouplingValidationResult buildFinalizedGraph(
    const CouplingContext& context,
    const CouplingContractDeclaration& declaration,
    const PartitionedCouplingPlan& plan)
{
    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const std::array<CouplingFormAnalysisMetadata, 0> installed_forms{};
    return graph.buildFinalizedGraph(
        context,
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed_forms),
        plan);
}

CouplingValidationResult buildFinalizedGraph(
    const CouplingContext& context,
    const CouplingContractDeclaration& declaration,
    const PartitionedCouplingPlan& plan,
    std::span<const CouplingExchangeDeclaration> exchange_templates)
{
    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const std::array<CouplingFormAnalysisMetadata, 0> installed_forms{};
    return graph.buildFinalizedGraph(
        context,
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed_forms),
        plan,
        exchange_templates);
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

TEST(CouplingGraph, RejectsInstalledMetadataWithoutBridgeReadinessGates)
{
    auto metadata = installedDependencyMetadata();
    metadata.feature_gates.clear();

    const std::vector<CouplingFormAnalysisMetadata> installed_forms{metadata};
    const auto validation = buildFinalizedGraph(
        twoParticipantGraphContext(),
        twoParticipantDependencyDeclaration(),
        installed_forms);

    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("installed-form validators require public bridge feature gate"),
              std::string::npos);
    EXPECT_NE(text.find("InstalledDependencies"), std::string::npos);
    EXPECT_NE(text.find("InstalledBlocks"), std::string::npos);
}

TEST(CouplingGraph, RejectsDeclaredImplicitDependencyMissingFromInstalledMetadata)
{
    const auto validation = buildFinalizedGraph(
        twoParticipantGraphContext(),
        twoParticipantDependencyDeclaration(),
        std::vector<CouplingFormAnalysisMetadata>{});

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

TEST(CouplingGraph, AcceptsGeneratedPartitionedPlanMatchingDeclarations)
{
    const auto context = twoParticipantGraphContext();
    const auto declaration = partitionedGraphDeclaration();
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};

    const PartitionedCouplingPlanGenerator generator;
    const auto plan = generator.generate(
        context,
        std::span<const CouplingContractDeclaration>(declarations));

    const auto validation = buildFinalizedGraph(context, declaration, plan);
    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(CouplingGraph, RejectsGeneratedPartitionedPlanMissingDeclaredExchange)
{
    const auto context = twoParticipantGraphContext();
    const auto declaration = partitionedGraphDeclaration();
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};

    const PartitionedCouplingPlanGenerator generator;
    auto plan = generator.generate(
        context,
        std::span<const CouplingContractDeclaration>(declarations));
    plan.exchanges.clear();

    const auto validation = buildFinalizedGraph(context, declaration, plan);
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("declared partitioned exchange is missing"),
              std::string::npos);
}

TEST(CouplingGraph, RejectsGeneratedPartitionedPlanWithUndeclaredExchange)
{
    const auto context = twoParticipantGraphContext();
    const auto declaration = partitionedGraphDeclaration();
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};

    const PartitionedCouplingPlanGenerator generator;
    auto plan = generator.generate(
        context,
        std::span<const CouplingContractDeclaration>(declarations));
    auto extra = plan.exchanges.front();
    extra.producer_port.port_name = "extra_out";
    plan.exchanges.push_back(extra);

    const auto validation = buildFinalizedGraph(context, declaration, plan);
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("generated partitioned exchange has no declaration"),
              std::string::npos);
}

TEST(CouplingGraph, AcceptsGeneratedPartitionedPlanWithTemplateExchange)
{
    const auto context = twoParticipantGraphContext();
    const auto declaration = partitionedGraphDeclaration();
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};

    auto extra_exchange = declaration.partitioned_exchange_declarations.front();
    extra_exchange.producer_port.port_name = "right_in";
    extra_exchange.consumer_port.port_name = "left_out";
    extra_exchange.producer = graphFieldEndpoint("right");
    extra_exchange.consumer = graphFieldEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> templates{extra_exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto plan = generator.generate(
        context,
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingExchangeDeclaration>(templates));

    const auto validation = buildFinalizedGraph(
        context,
        declaration,
        plan,
        std::span<const CouplingExchangeDeclaration>(templates));
    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(CouplingGraph, RejectsGeneratedPartitionedPlanMissingGroupHint)
{
    const auto context = twoParticipantGraphContext();
    const auto declaration = partitionedGraphDeclaration();
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};

    const PartitionedCouplingPlanGenerator generator;
    auto plan = generator.generate(
        context,
        std::span<const CouplingContractDeclaration>(declarations));
    plan.group_hints.clear();

    const auto validation = buildFinalizedGraph(context, declaration, plan);
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("declared partitioned group hint is missing"),
              std::string::npos);
}
