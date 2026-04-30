#include "Coupling/CouplingGraph.h"
#include "Coupling/MonolithicCouplingBuilder.h"
#include "Coupling/PartitionedCouplingPlanGenerator.h"

#include "Analysis/CouplingGraphAnalyzer.h"
#include "Analysis/FormulationRecord.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Assembly/AssemblyContext.h"
#include "Assembly/AssemblyKernel.h"
#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryStateManager.h"
#include "Core/FEException.h"
#include "Forms/BoundaryFunctional.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

const systems::FESystem* graphSystemToken()
{
    return reinterpret_cast<const systems::FESystem*>(1);
}

bool hasAnalysisCouplingClaim(const analysis::ProblemAnalysisReport& report,
                              const analysis::VariableKey& a,
                              const analysis::VariableKey& b,
                              analysis::DomainKind domain)
{
    for (const auto& claim : report.claims) {
        if (claim.kind != analysis::PropertyKind::CoupledSystemStructure ||
            claim.domain != domain ||
            claim.variables.size() != 2u) {
            continue;
        }
        const bool same_order = claim.variables[0] == a && claim.variables[1] == b;
        const bool reverse_order = claim.variables[0] == b && claim.variables[1] == a;
        if (same_order || reverse_order) {
            return true;
        }
    }
    return false;
}

class GraphParameterKernel final : public assembly::BilinearFormKernel {
public:
    [[nodiscard]] assembly::RequiredData getRequiredData() const override
    {
        return assembly::RequiredData::None;
    }

    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override
    {
        params::Spec spec;
        spec.key = "coefficient";
        spec.type = params::ValueType::Real;
        spec.required = true;
        spec.default_value = params::Value{Real(1.0)};
        return {spec};
    }

    void computeCell(const assembly::AssemblyContext&,
                     assembly::KernelOutput& output) override
    {
        output.reserve(0, 0, true, false);
    }

    [[nodiscard]] std::string name() const override
    {
        return "GraphParameterKernel";
    }
};

class GraphScalarOutputModel final : public systems::AuxiliaryStateModel {
public:
    [[nodiscard]] std::string modelName() const override
    {
        return "GraphScalarOutputModel";
    }

    [[nodiscard]] int dimension() const override
    {
        return 1;
    }

    [[nodiscard]] systems::AuxiliaryStructuralMetadata structuralMetadata()
        const override
    {
        systems::AuxiliaryStructuralMetadata metadata;
        metadata.variable_kinds = {systems::AuxiliaryVariableKind::Differential};
        return metadata;
    }

    void evaluateResidual(const systems::AuxiliaryLocalContext&,
                          systems::AuxiliaryResidualRequest& request) const override
    {
        if (!request.residual.empty()) {
            request.residual[0] = 0.0;
        }
    }

    [[nodiscard]] int outputCount() const override
    {
        return 1;
    }

    [[nodiscard]] std::vector<std::string> outputNames() const override
    {
        return {"out_value"};
    }

    void evaluateOutputs(const systems::AuxiliaryLocalContext& ctx,
                         std::span<Real> output) const override
    {
        if (!output.empty()) {
            output[0] = ctx.x.empty() ? 0.0 : ctx.x[0];
        }
    }
};

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

CouplingContext twoBoundaryRegionGraphContext()
{
    const auto* system = graphSystemToken();
    const CouplingRegionRef inlet{
        .participant_name = "left",
        .system_name = "system",
        .system = system,
        .region_name = "inlet",
        .kind = CouplingRegionKind::Boundary,
        .marker = 4,
    };
    const CouplingRegionRef outlet{
        .participant_name = "left",
        .system_name = "system",
        .system = system,
        .region_name = "outlet",
        .kind = CouplingRegionKind::Boundary,
        .marker = 5,
    };

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "left",
        .system_name = "system",
        .system = system,
    });
    builder.addRegion(inlet);
    builder.addRegion(outlet);
    return builder.build();
}

struct NonFieldGraphFixture {
    std::shared_ptr<spaces::H1Space> space;
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh;
    systems::FESystem system;
    FieldId field{INVALID_FIELD_ID};
    CouplingContext context;

    explicit NonFieldGraphFixture(bool register_variables)
        : space(std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1))
        , mesh(std::make_shared<forms::test::SingleTetraMeshAccess>())
        , system(mesh)
    {
        field = system.addField(systems::FieldSpec{
            .name = "primary",
            .space = space,
            .components = 1,
        });

        if (register_variables) {
            system.auxiliaryStateManager().registerBlock(
                systems::AuxiliaryStateSpec{
                    .name = "aux_state",
                    .size = 1,
                    .scope = systems::AuxiliaryStateScope::Global,
                },
                1);

            systems::AuxiliaryInputSpec input;
            input.name = "driver_input";
            input.size = 1;
            input.producer = systems::AuxiliaryInputProducer::DirectUserData;
            system.auxiliaryInputRegistry().registerInput(input);

            system.deployAuxiliaryModel(
                systems::use(std::make_shared<GraphScalarOutputModel>())
                    .name("output_block")
                    .global()
                    .partitioned("ForwardEuler")
                    .initialize({0.0}));

            forms::BoundaryFunctional functional;
            functional.name = "surface_measure";
            functional.integrand = forms::FormExpr::constant(1.0);
            functional.boundary_marker = 1;
            system.boundaryReductionService(field)
                .addBoundaryFunctional(std::move(functional));

            system.addOperator("op");
            system.addCellKernel(
                "op",
                field,
                std::make_shared<GraphParameterKernel>());
            systems::SetupInputs inputs;
            inputs.topology_override = timestepping::test::singleTetraTopology();
            system.setup({}, inputs);
        }

        CouplingRegionRef surface{
            .participant_name = "left",
            .system_name = "left_system",
            .system = &system,
            .region_name = "surface",
            .kind = CouplingRegionKind::Boundary,
            .marker = 1,
        };

        CouplingContextBuilder builder;
        builder.addParticipant({
            .participant_name = "left",
            .system_name = "left_system",
            .system = &system,
        });
        builder.addField({
            .participant_name = "left",
            .system_name = "left_system",
            .system = &system,
            .field_name = "primary",
            .field_id = field,
            .space = space,
            .components = 1,
        });
        builder.addRegion(surface);
        builder.addSharedRegion(SharedRegionRef{
            .name = "interface",
            .required_region_kind = CouplingRegionKind::Boundary,
            .participant_regions = {surface},
        });
        context = builder.build();
    }
};

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

CouplingContext interfaceGraphContext(int left_marker = 17,
                                      int right_marker = 17)
{
    const auto* system = graphSystemToken();
    const CouplingRegionRef left_region{
        .participant_name = "left",
        .system_name = "system",
        .system = system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = left_marker,
        .side = CouplingInterfaceSide::Minus,
    };
    const CouplingRegionRef right_region{
        .participant_name = "right",
        .system_name = "system",
        .system = system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = right_marker,
        .side = CouplingInterfaceSide::Plus,
    };

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
    builder.addRegion(left_region);
    builder.addRegion(right_region);
    builder.addSharedRegion(SharedRegionRef{
        .name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .participant_regions = {left_region, right_region},
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

CouplingAdditionalFieldDeclaration graphAdditionalField(
    CouplingAdditionalFieldNamespace field_namespace,
    std::string namespace_name,
    std::string field_name,
    std::string system_participant_name = {})
{
    return CouplingAdditionalFieldDeclaration{
        .field_namespace = field_namespace,
        .namespace_name = std::move(namespace_name),
        .system_participant_name = std::move(system_participant_name),
        .field_name = std::move(field_name),
        .space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1),
        .components = 1,
    };
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

CouplingContractDeclaration nonFieldGraphDeclaration()
{
    auto declaration = graphDeclaration();
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::AuxiliaryState,
        .participant_name = "left",
        .name = "aux_state",
    });
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::AuxiliaryInput,
        .participant_name = "left",
        .name = "driver_input",
    });
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::AuxiliaryOutput,
        .participant_name = "left",
        .name = "out_value",
    });
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::BoundaryFunctional,
        .participant_name = "left",
        .name = "surface_measure",
    });
    declaration.dependencies.push_back(CouplingResidualDependency{
        .residual_row = {
            .kind = CouplingVariableKind::Field,
            .participant_name = "left",
            .name = "primary",
        },
        .dependency = {
            .kind = CouplingVariableKind::GlobalScalar,
            .participant_name = "left",
            .name = "coefficient",
        },
    });
    return declaration;
}

CouplingFormAnalysisMetadata nonFieldGraphDependencyMetadata(FieldId row_field)
{
    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "non_field_variable_coupling";
    metadata.system_name = "left_system";
    metadata.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledDependencies,
        analysis::FormBridgeFeatureStatus::Available,
        "Synthetic non-field dependency fixture"});
    metadata.variable_dependencies.push_back(CouplingFormVariableDependencyProvenance{
        .residual_row = analysis::VariableKey::field(row_field),
        .dependency = analysis::VariableKey::named(
            analysis::VariableKind::GlobalScalar,
            "left_system/coefficient"),
        .mode = CouplingDependencyMode::ImplicitMonolithic,
        .domain = analysis::DomainKind::Global,
        .contributes_vector = true,
        .provider = "forms",
    });
    return metadata;
}

CouplingContractDeclaration providerMetadataDeclaration()
{
    auto declaration = graphDeclaration();
    declaration.non_field_dependencies = {
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::Parameter,
            .participant_name = "left",
            .name = "penalty",
            .region = CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "surface",
                .shared_region_name = "interface",
            },
            .required_region_kind = CouplingRegionKind::Boundary,
            .expected_parameter_value_type = params::ValueType::Real,
            .expected_value_type = "scalar",
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::Coefficient,
            .participant_name = "left",
            .name = "wall_speed",
            .expected_value_type = "vector",
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::MaterialStateOld,
            .participant_name = "left",
            .name = "history_old",
            .expected_value_type = "tensor",
            .material_state_byte_offset = 8,
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::MaterialStateWork,
            .participant_name = "left",
            .name = "history_work",
            .expected_value_type = "tensor",
            .material_state_byte_offset = 16,
        },
        CouplingNonFieldDependencyRequirement{
            .kind = CouplingNonFieldDependencyRequirementKind::BoundaryIntegral,
            .participant_name = "left",
            .name = "traction_integral",
            .region = CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "surface",
                .shared_region_name = "interface",
            },
            .required_region_kind = CouplingRegionKind::Boundary,
            .expected_value_type = "scalar",
        },
    };
    return declaration;
}

CouplingFormAnalysisMetadata providerMetadataFixture()
{
    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "provider_metadata_coupling";
    metadata.system_name = "system";
    metadata.non_field_dependencies = {
        CouplingFormNonFieldDependencyProvenance{
            .kind = CouplingFormNonFieldDependencyKind::Parameter,
            .participant_name = "left",
            .system_name = "system",
            .name = "penalty",
            .domain = analysis::DomainKind::Boundary,
            .region_name = "surface",
            .shared_region_name = "interface",
            .marker = 4,
            .slot = 5,
            .provider = "forms",
            .value_type = "scalar",
            .parameter_value_type = params::ValueType::Real,
        },
        CouplingFormNonFieldDependencyProvenance{
            .kind = CouplingFormNonFieldDependencyKind::Coefficient,
            .participant_name = "left",
            .system_name = "system",
            .name = "wall_speed",
            .provider = "forms",
            .value_type = "vector",
        },
        CouplingFormNonFieldDependencyProvenance{
            .kind = CouplingFormNonFieldDependencyKind::MaterialStateOld,
            .participant_name = "left",
            .system_name = "system",
            .name = "history_old",
            .byte_offset = 8,
            .provider = "forms",
            .value_type = "tensor",
        },
        CouplingFormNonFieldDependencyProvenance{
            .kind = CouplingFormNonFieldDependencyKind::MaterialStateWork,
            .participant_name = "left",
            .system_name = "system",
            .name = "history_work",
            .byte_offset = 16,
            .provider = "forms",
            .value_type = "tensor",
        },
        CouplingFormNonFieldDependencyProvenance{
            .kind = CouplingFormNonFieldDependencyKind::BoundaryIntegral,
            .participant_name = "left",
            .system_name = "system",
            .name = "traction_integral",
            .domain = analysis::DomainKind::Boundary,
            .region_name = "surface",
            .shared_region_name = "interface",
            .marker = 4,
            .slot = 6,
            .provider = "forms",
            .value_type = "scalar",
        },
    };
    return metadata;
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

CouplingInstallMetadata expertInstallMetadata()
{
    CouplingInstallMetadata metadata;
    metadata.contribution_name = "expert_balance";
    metadata.origin = "expert_fixture";
    metadata.system_name = "system";
    metadata.operator_name = "equations";
    metadata.installed_dependencies.push_back(CouplingInstalledDependency{
        .residual_row = analysis::VariableKey::field(2),
        .dependency = analysis::VariableKey::field(1),
        .mode = CouplingDependencyMode::ImplicitMonolithic,
        .domain = analysis::DomainKind::Cell,
        .contributes_matrix_block = true,
        .contributes_vector = true,
        .provider = "expert_fixture",
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

TEST(CouplingGraph, BuildsGraphForMultipleContractsSharingContext)
{
    auto first = graphDeclaration();
    first.contract_name = "wall_interface";

    auto second = graphDeclaration();
    second.contract_name = "thermal_interface";

    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 2> declarations{
        first,
        second,
    };
    const auto validation = graph.buildDeclarationGraph(
        graphContext(),
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto& snapshot = graph.snapshot();
    EXPECT_EQ(snapshot.participants.size(), 1u);
    EXPECT_EQ(snapshot.fields.size(), 1u);
    EXPECT_EQ(snapshot.regions.size(), 1u);
    EXPECT_EQ(snapshot.shared_regions.size(), 1u);
    ASSERT_EQ(snapshot.contract_types.size(), 1u);
    EXPECT_EQ(snapshot.contract_types[0].contract_type, "generic");
    ASSERT_EQ(snapshot.contract_instances.size(), 2u);
    EXPECT_EQ(snapshot.contract_instances[0].contract_name, "wall_interface");
    EXPECT_EQ(snapshot.contract_instances[1].contract_name, "thermal_interface");
}

TEST(CouplingGraph, SeparatesContractTypesFromInstances)
{
    auto first = graphDeclaration();
    first.contract_type = "fsi";
    first.contract_name = "wall_interface";

    auto second = graphDeclaration();
    second.contract_type = "thermal";
    second.contract_name = "thermal_interface";

    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 2> declarations{
        first,
        second,
    };
    const auto validation = graph.buildDeclarationGraph(
        graphContext(),
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto& snapshot = graph.snapshot();
    ASSERT_EQ(snapshot.contract_types.size(), 2u);
    EXPECT_EQ(snapshot.contract_types[0].contract_type, "fsi");
    EXPECT_EQ(snapshot.contract_types[1].contract_type, "thermal");
    ASSERT_EQ(snapshot.contract_instances.size(), 2u);
    EXPECT_EQ(snapshot.contract_instances[0].contract_type, "fsi");
    EXPECT_EQ(snapshot.contract_instances[0].contract_name, "wall_interface");
    EXPECT_EQ(snapshot.contract_instances[1].contract_type, "thermal");
    EXPECT_EQ(snapshot.contract_instances[1].contract_name, "thermal_interface");

    auto duplicate = second;
    duplicate.contract_name = first.contract_name;
    const std::array<CouplingContractDeclaration, 2> duplicate_declarations{
        first,
        duplicate,
    };
    CouplingGraph duplicate_graph;
    const auto duplicate_validation = duplicate_graph.buildDeclarationGraph(
        graphContext(),
        std::span<const CouplingContractDeclaration>(duplicate_declarations));
    EXPECT_FALSE(duplicate_validation.ok());
    EXPECT_NE(formatDiagnostics(duplicate_validation)
                  .find("duplicate coupling contract instance name"),
              std::string::npos);
}

TEST(CouplingGraph, RecordsDeclarationGraphNodeCategories)
{
    auto declaration = graphDeclaration();
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::AuxiliaryInput,
        .participant_name = "left",
        .name = "inlet_signal",
    });
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::Parameter,
        .participant_name = "left",
        .name = "density",
    });
    declaration.dependencies.push_back(CouplingResidualDependency{
        .residual_row = {
            .kind = CouplingVariableKind::Field,
            .participant_name = "left",
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
    declaration.temporal_requirements.push_back({
        .quantity = CouplingTemporalQuantity::TimeStep,
    });
    declaration.geometry_requirements.push_back({
        .quantity = CouplingGeometryTerminalQuantity::Coordinate,
    });
    declaration.partitioned_exchange_declarations.push_back(
        CouplingExchangeDeclaration{
            .producer_port = {
                .contract_instance_name = "generic_instance",
                .port_name = "left_out",
            },
            .consumer_port = {
                .contract_instance_name = "generic_instance",
                .port_name = "left_in",
            },
            .value = CouplingValueDescriptor{
                .rank = CouplingValueRank::Scalar,
                .components = 1,
            },
            .producer = graphFieldEndpoint("left"),
            .consumer = graphFieldEndpoint("left"),
            .transfer = CouplingTransferDeclaration{
                .kind = CouplingTransferKind::Identity,
            },
        });

    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const auto validation = graph.buildDeclarationGraph(
        graphContext(),
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto& snapshot = graph.snapshot();
    EXPECT_EQ(snapshot.participants.size(), 1u);
    EXPECT_EQ(snapshot.fields.size(), 1u);
    EXPECT_EQ(snapshot.regions.size(), 1u);
    EXPECT_EQ(snapshot.shared_regions.size(), 1u);
    ASSERT_EQ(snapshot.contract_types.size(), 1u);
    EXPECT_EQ(snapshot.contract_types[0].contract_type, "generic");
    ASSERT_EQ(snapshot.contract_instances.size(), 1u);
    EXPECT_EQ(snapshot.contract_instances[0].contract_name,
              "generic_instance");
    ASSERT_EQ(snapshot.non_field_variables.size(), 1u);
    ASSERT_TRUE(snapshot.non_field_variables[0].variable.has_value());
    EXPECT_EQ(snapshot.non_field_variables[0].variable->kind,
              analysis::VariableKind::AuxiliaryInput);
    ASSERT_EQ(snapshot.provider_metadata_requirements.size(), 1u);
    EXPECT_EQ(snapshot.provider_metadata_requirements[0].requirement.kind,
              CouplingNonFieldDependencyRequirementKind::Parameter);
    EXPECT_EQ(snapshot.temporal_requirements.size(), 1u);
    EXPECT_EQ(snapshot.geometry_requirements.size(), 1u);
    EXPECT_EQ(snapshot.partitioned_exchange_declarations.size(), 1u);
    ASSERT_EQ(snapshot.dependency_expectations.size(), 1u);
    EXPECT_TRUE(snapshot.dependency_expectations[0].residual_row.has_value());
    EXPECT_TRUE(snapshot.dependency_expectations[0].dependency.has_value());
    ASSERT_EQ(snapshot.expected_blocks.size(), 1u);
    EXPECT_TRUE(snapshot.expected_blocks[0].residual_row.has_value());
    EXPECT_TRUE(snapshot.expected_blocks[0].dependency.has_value());
}

TEST(CouplingGraph, AggregatesGeometryTerminalRequirementsAcrossContracts)
{
    CouplingContractDeclaration left;
    left.contract_type = "generic";
    left.contract_name = "left_geometry";
    left.participants.push_back({.participant_name = "left"});
    left.regions.push_back({
        .participant_name = "left",
        .region_name = "inlet",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    left.temporal_requirements.push_back({
        .quantity = CouplingTemporalQuantity::TimeStep,
    });
    left.geometry_requirements.push_back({
        .quantity = CouplingGeometryTerminalQuantity::CurrentNormal,
        .scope = CouplingGeometryTerminalScope{
            .participant_name = "left",
            .region = CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "inlet",
            },
            .location = CouplingGeometryTerminalLocationDeclaration{
                .region_kind = CouplingRegionKind::Boundary,
                .coordinate_configuration =
                    forms::GeometryConfiguration::Current,
                .transform_from_configuration =
                    forms::GeometryConfiguration::Reference,
                .transform_to_configuration =
                    forms::GeometryConfiguration::Current,
                .quadrature_policy_key = 42,
            },
        },
    });

    CouplingContractDeclaration right;
    right.contract_type = "generic";
    right.contract_name = "right_geometry";
    right.participants.push_back({.participant_name = "left"});
    right.regions.push_back({
        .participant_name = "left",
        .region_name = "outlet",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    right.geometry_requirements.push_back({
        .quantity = CouplingGeometryTerminalQuantity::CellDomainId,
        .scope = CouplingGeometryTerminalScope{
            .participant_name = "left",
            .region = CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "outlet",
            },
            .location = CouplingGeometryTerminalLocationDeclaration{
                .region_kind = CouplingRegionKind::Boundary,
            },
        },
    });

    const auto context = twoBoundaryRegionGraphContext();
    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 2> declarations{left, right};
    const auto validation = graph.buildDeclarationGraph(
        context,
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto& snapshot = graph.snapshot();
    ASSERT_EQ(snapshot.geometry_requirements.size(), 2u);
    EXPECT_EQ(snapshot.geometry_requirements[0].contract_name, "left_geometry");
    ASSERT_TRUE(snapshot.geometry_requirements[0]
                    .requirement.scope.location.has_value());
    EXPECT_EQ(snapshot.geometry_requirements[0]
                  .requirement.scope.location->region_kind,
              CouplingRegionKind::Boundary);
    EXPECT_EQ(snapshot.geometry_requirements[0]
                  .requirement.scope.location->coordinate_configuration,
              forms::GeometryConfiguration::Current);
    EXPECT_EQ(snapshot.geometry_requirements[0]
                  .requirement.scope.location->transform_from_configuration,
              forms::GeometryConfiguration::Reference);
    EXPECT_EQ(snapshot.geometry_requirements[0]
                  .requirement.scope.location->transform_to_configuration,
              forms::GeometryConfiguration::Current);
    EXPECT_EQ(snapshot.geometry_requirements[0]
                  .requirement.scope.location->quadrature_policy_key,
              42u);
    EXPECT_EQ(snapshot.geometry_requirements[1].contract_name, "right_geometry");
    ASSERT_TRUE(snapshot.geometry_requirements[1]
                    .requirement.scope.location.has_value());
    EXPECT_EQ(snapshot.geometry_requirements[1]
                  .requirement.scope.location->region_kind,
              CouplingRegionKind::Boundary);

    CouplingTemporalAvailability temporal_availability;
    temporal_availability.provides_time_step = false;
    const auto temporal_validation =
        graph.validateTemporalRequirements(temporal_availability);
    EXPECT_FALSE(temporal_validation.ok());
    EXPECT_NE(formatDiagnostics(temporal_validation).find("time-step"),
              std::string::npos);

    const CouplingGeometryTerminalAvailability geometry_availability{
        .supported_quantities = {CouplingGeometryTerminalQuantity::CellDomainId},
        .supported_domains = {analysis::DomainKind::Boundary},
        .supports_reference_configuration = true,
        .supports_current_configuration = true,
    };
    const auto geometry_validation =
        graph.validateGeometryTerminalRequirements(context, geometry_availability);
    EXPECT_FALSE(geometry_validation.ok());
    const auto text = formatDiagnostics(geometry_validation);
    EXPECT_NE(text.find("geometry terminal quantity"), std::string::npos);
    EXPECT_EQ(text.find("time-step"), std::string::npos);
}

TEST(CouplingGraph, ValidatesRequiredNonFieldGraphVariablesThroughSystemRegistries)
{
    NonFieldGraphFixture fixture(/*register_variables=*/true);
    const auto declaration = nonFieldGraphDeclaration();
    const std::vector<CouplingFormAnalysisMetadata> installed_forms{
        nonFieldGraphDependencyMetadata(fixture.field)};
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};

    CouplingGraph graph;
    const auto validation = graph.buildFinalizedGraph(
        fixture.context,
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed_forms));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto& snapshot = graph.snapshot();
    auto has_variable = [&](analysis::VariableKind kind,
                            const std::string& name) {
        return std::any_of(
            snapshot.non_field_variables.begin(),
            snapshot.non_field_variables.end(),
            [&](const CouplingGraphNonFieldVariableNode& node) {
                return node.variable.has_value() &&
                       node.variable->kind == kind &&
                       node.variable->name == name;
            });
    };

    EXPECT_TRUE(has_variable(analysis::VariableKind::AuxiliaryState,
                             "left_system/aux_state"));
    EXPECT_TRUE(has_variable(analysis::VariableKind::AuxiliaryInput,
                             "left_system/driver_input"));
    EXPECT_TRUE(has_variable(analysis::VariableKind::AuxiliaryOutput,
                             "left_system/out_value"));
    EXPECT_TRUE(has_variable(analysis::VariableKind::BoundaryFunctional,
                             "left_system/surface_measure"));
    ASSERT_EQ(snapshot.dependency_expectations.size(), 1u);
    ASSERT_TRUE(snapshot.dependency_expectations[0].dependency.has_value());
    EXPECT_EQ(snapshot.dependency_expectations[0].dependency->kind,
              analysis::VariableKind::GlobalScalar);
    EXPECT_EQ(snapshot.dependency_expectations[0].dependency->name,
              "left_system/coefficient");
}

TEST(CouplingGraph, RejectsMissingRequiredNonFieldGraphVariablesInSystemRegistries)
{
    NonFieldGraphFixture fixture(/*register_variables=*/false);
    const auto declaration = nonFieldGraphDeclaration();
    const std::vector<CouplingFormAnalysisMetadata> installed_forms{
        nonFieldGraphDependencyMetadata(fixture.field)};

    const auto validation = buildFinalizedGraph(
        fixture.context,
        declaration,
        installed_forms);

    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("AuxiliaryStateManager"), std::string::npos);
    EXPECT_NE(text.find("AuxiliaryInputRegistry"), std::string::npos);
    EXPECT_NE(text.find("AuxiliaryOutputRegistry"), std::string::npos);
    EXPECT_NE(text.find("BoundaryReductionService"), std::string::npos);
    EXPECT_NE(text.find("ParameterRegistry"), std::string::npos);
    EXPECT_NE(text.find("left_system/coefficient"), std::string::npos);
}

TEST(CouplingGraph, ValidatesProviderOnlyNonFieldRequirements)
{
    const auto declaration = providerMetadataDeclaration();
    const std::vector<CouplingFormAnalysisMetadata> installed_forms{
        providerMetadataFixture()};
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};

    CouplingGraph graph;
    const auto validation = graph.buildFinalizedGraph(
        graphContext(),
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed_forms));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto& snapshot = graph.snapshot();
    EXPECT_TRUE(snapshot.non_field_variables.empty());
    ASSERT_EQ(snapshot.provider_metadata_requirements.size(), 5u);
    EXPECT_EQ(snapshot.provider_metadata_requirements.front().requirement.kind,
              CouplingNonFieldDependencyRequirementKind::Parameter);
    EXPECT_EQ(snapshot.provider_metadata_requirements.back().requirement.kind,
              CouplingNonFieldDependencyRequirementKind::BoundaryIntegral);
}

TEST(CouplingGraph, RejectsMissingProviderOnlyNonFieldMetadata)
{
    const auto declaration = providerMetadataDeclaration();
    auto metadata = providerMetadataFixture();
    metadata.non_field_dependencies.pop_back();
    const std::vector<CouplingFormAnalysisMetadata> installed_forms{metadata};

    const auto validation = buildFinalizedGraph(
        graphContext(),
        declaration,
        installed_forms);

    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("required provider metadata is missing"),
              std::string::npos);
    EXPECT_NE(text.find("BoundaryIntegral(left/traction_integral)"),
              std::string::npos);
}

TEST(CouplingGraph, RejectsIncompleteProviderMetadataProvenance)
{
    const auto declaration = providerMetadataDeclaration();
    auto missing_region = providerMetadataFixture();
    missing_region.non_field_dependencies[0].region_name.reset();
    auto wrong_marker = providerMetadataFixture();
    wrong_marker.non_field_dependencies[0].marker = 99;
    auto missing_provider = providerMetadataFixture();
    missing_provider.non_field_dependencies[0].provider.clear();

    auto expect_rejected = [&](const CouplingFormAnalysisMetadata& metadata) {
        const std::vector<CouplingFormAnalysisMetadata> installed_forms{metadata};
        const auto validation = buildFinalizedGraph(
            graphContext(),
            declaration,
            installed_forms);
        EXPECT_FALSE(validation.ok());
        EXPECT_NE(formatDiagnostics(validation).find("Parameter(left/penalty)"),
                  std::string::npos);
    };

    expect_rejected(missing_region);
    expect_rejected(wrong_marker);
    expect_rejected(missing_provider);
}

TEST(CouplingGraph, DistinguishesProviderMetadataWithSameNameByRegion)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.regions.push_back({
        .participant_name = "left",
        .region_name = "inlet",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    declaration.regions.push_back({
        .participant_name = "left",
        .region_name = "outlet",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::Parameter,
        .participant_name = "left",
        .name = "penalty",
        .region = CouplingRegionEndpointDeclaration{
            .participant_name = "left",
            .region_name = "inlet",
        },
        .required_region_kind = CouplingRegionKind::Boundary,
        .expected_parameter_value_type = params::ValueType::Real,
        .expected_value_type = "scalar",
    });
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::Parameter,
        .participant_name = "left",
        .name = "penalty",
        .region = CouplingRegionEndpointDeclaration{
            .participant_name = "left",
            .region_name = "outlet",
        },
        .required_region_kind = CouplingRegionKind::Boundary,
        .expected_parameter_value_type = params::ValueType::Real,
        .expected_value_type = "scalar",
    });

    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "regional_provider_metadata";
    metadata.system_name = "system";
    metadata.non_field_dependencies = {
        CouplingFormNonFieldDependencyProvenance{
            .kind = CouplingFormNonFieldDependencyKind::Parameter,
            .participant_name = "left",
            .system_name = "system",
            .name = "penalty",
            .domain = analysis::DomainKind::Boundary,
            .region_name = "inlet",
            .marker = 4,
            .provider = "forms",
            .value_type = "scalar",
            .parameter_value_type = params::ValueType::Real,
        },
        CouplingFormNonFieldDependencyProvenance{
            .kind = CouplingFormNonFieldDependencyKind::Parameter,
            .participant_name = "left",
            .system_name = "system",
            .name = "penalty",
            .domain = analysis::DomainKind::Boundary,
            .region_name = "outlet",
            .marker = 5,
            .provider = "forms",
            .value_type = "scalar",
            .parameter_value_type = params::ValueType::Real,
        },
    };

    const std::vector<CouplingFormAnalysisMetadata> installed_forms{metadata};
    const auto validation = buildFinalizedGraph(
        twoBoundaryRegionGraphContext(),
        declaration,
        installed_forms);
    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(CouplingGraph, ValidatesInterfaceProviderMetadataSideProvenance)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::BoundaryIntegral,
        .participant_name = "left",
        .name = "interface_flux",
        .region = CouplingRegionEndpointDeclaration{
            .participant_name = "left",
            .region_name = "interface",
            .shared_region_name = "interface",
        },
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .expected_value_type = "scalar",
    });

    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "interface_provider_metadata";
    metadata.system_name = "system";
    metadata.non_field_dependencies.push_back({
        .kind = CouplingFormNonFieldDependencyKind::BoundaryIntegral,
        .participant_name = "left",
        .system_name = "system",
        .name = "interface_flux",
        .domain = analysis::DomainKind::InterfaceFace,
        .region_name = "interface",
        .shared_region_name = "interface",
        .marker = 17,
        .side = CouplingInterfaceSide::Minus,
        .provider = "forms",
        .value_type = "scalar",
    });

    const std::vector<CouplingFormAnalysisMetadata> installed_forms{metadata};
    const auto accepted = buildFinalizedGraph(
        interfaceGraphContext(17, 17),
        declaration,
        installed_forms);
    EXPECT_TRUE(accepted.ok()) << formatDiagnostics(accepted);

    metadata.non_field_dependencies[0].side = CouplingInterfaceSide::Plus;
    const std::vector<CouplingFormAnalysisMetadata> wrong_side_forms{metadata};
    const auto wrong_side = buildFinalizedGraph(
        interfaceGraphContext(17, 17),
        declaration,
        wrong_side_forms);
    EXPECT_FALSE(wrong_side.ok());
    EXPECT_NE(formatDiagnostics(wrong_side).find(
                  "BoundaryIntegral(left/interface_flux)"),
              std::string::npos);
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

TEST(CouplingGraph, FormatsMissingContextDiagnosticsWithLookupNames)
{
    auto declaration = graphDeclaration();
    declaration.participants.push_back({.participant_name = "right"});
    declaration.fields.push_back({.participant_name = "right", .field_name = "primary"});
    declaration.regions.push_back({.participant_name = "right", .region_name = "surface"});
    declaration.shared_regions.push_back({.shared_region_name = "other_interface"});

    const auto validation = buildGraph(graphContext(), declaration);
    ASSERT_FALSE(validation.ok());

    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("contract='generic_instance'"), std::string::npos);
    EXPECT_NE(text.find("participant='right'"), std::string::npos);
    EXPECT_NE(text.find("field='primary'"), std::string::npos);
    EXPECT_NE(text.find("region='surface'"), std::string::npos);
    EXPECT_NE(text.find("region='other_interface'"), std::string::npos);
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

TEST(CouplingGraph, AcceptsUniqueAdditionalFieldsAcrossContracts)
{
    auto first = graphDeclaration();
    first.contract_name = "first_instance";
    first.additional_fields.push_back(graphAdditionalField(
        CouplingAdditionalFieldNamespace::Participant,
        "left",
        "lambda"));

    auto second = graphDeclaration();
    second.contract_name = "second_instance";
    second.additional_fields.push_back(graphAdditionalField(
        CouplingAdditionalFieldNamespace::Contract,
        "second_instance",
        "lambda",
        "left"));

    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 2> declarations{
        first,
        second,
    };
    const auto validation = graph.buildDeclarationGraph(
        graphContext(),
        std::span<const CouplingContractDeclaration>(declarations));

    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);
    EXPECT_EQ(graph.snapshot().additional_fields.size(), 2u);
}

TEST(CouplingGraph, RejectsDuplicateAdditionalFieldsAcrossContractsAndBaseCollisions)
{
    auto first = graphDeclaration();
    first.contract_name = "first_instance";
    first.additional_fields.push_back(graphAdditionalField(
        CouplingAdditionalFieldNamespace::Participant,
        "left",
        "lambda"));

    auto duplicate = graphDeclaration();
    duplicate.contract_name = "second_instance";
    duplicate.additional_fields.push_back(graphAdditionalField(
        CouplingAdditionalFieldNamespace::Participant,
        "left",
        "lambda"));

    auto collision = graphDeclaration();
    collision.contract_name = "third_instance";
    collision.additional_fields.push_back(graphAdditionalField(
        CouplingAdditionalFieldNamespace::Participant,
        "left",
        "primary"));

    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 3> declarations{
        first,
        duplicate,
        collision,
    };
    const auto validation = graph.buildDeclarationGraph(
        graphContext(),
        std::span<const CouplingContractDeclaration>(declarations));

    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("duplicate additional field declaration"),
              std::string::npos);
    EXPECT_NE(text.find("collides with a base field"), std::string::npos);
}

TEST(CouplingGraph, ValidatesContractOwnedAdditionalFieldNamespaceAndTarget)
{
    auto valid = graphDeclaration();
    valid.additional_fields.push_back(graphAdditionalField(
        CouplingAdditionalFieldNamespace::Contract,
        "generic_instance",
        "lambda",
        "left"));
    EXPECT_TRUE(buildGraph(graphContext(), valid).ok());

    auto wrong_namespace = graphDeclaration();
    wrong_namespace.contract_name = "wrong_namespace_instance";
    wrong_namespace.additional_fields.push_back(graphAdditionalField(
        CouplingAdditionalFieldNamespace::Contract,
        "generic_instance",
        "lambda",
        "left"));

    auto missing_target = graphDeclaration();
    missing_target.contract_name = "missing_target_instance";
    missing_target.additional_fields.push_back(graphAdditionalField(
        CouplingAdditionalFieldNamespace::Contract,
        "missing_target_instance",
        "lambda"));

    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 2> declarations{
        wrong_namespace,
        missing_target,
    };
    const auto validation = graph.buildDeclarationGraph(
        graphContext(),
        std::span<const CouplingContractDeclaration>(declarations));

    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("namespace must match the contract instance name"),
              std::string::npos);
    EXPECT_NE(text.find("does not resolve to a target system"),
              std::string::npos);
}

TEST(CouplingGraph, RejectsSkippedOptionalAdditionalFieldReferences)
{
    auto skipped_owner = graphDeclaration();
    skipped_owner.contract_name = "optional_instance";
    auto skipped_field = graphAdditionalField(
        CouplingAdditionalFieldNamespace::Contract,
        "optional_instance",
        "lambda",
        "left");
    skipped_field.requirement = CouplingRequirement::Optional;
    skipped_field.enabled = false;
    skipped_owner.additional_fields.push_back(skipped_field);

    auto dependent = graphDeclaration();
    dependent.contract_name = "dependent_instance";
    const CouplingVariableUse row{
        .kind = CouplingVariableKind::Field,
        .participant_name = "left",
        .name = "primary",
    };
    const CouplingVariableUse skipped_dependency{
        .kind = CouplingVariableKind::Field,
        .participant_name = "optional_instance",
        .name = "lambda",
    };
    dependent.dependencies.push_back(CouplingResidualDependency{
        .residual_row = row,
        .dependency = skipped_dependency,
    });
    dependent.expected_blocks.push_back(CouplingBlockExpectation{
        .residual_row = row,
        .dependency = skipped_dependency,
    });

    const std::array<CouplingContractDeclaration, 2> skipped_declarations{
        skipped_owner,
        dependent,
    };
    CouplingGraph graph;
    const auto graph_validation = graph.buildDeclarationGraph(
        graphContext(),
        std::span<const CouplingContractDeclaration>(skipped_declarations));

    EXPECT_FALSE(graph_validation.ok());
    EXPECT_NE(formatDiagnostics(graph_validation).find(
                  "disabled optional additional field is referenced by another contract"),
              std::string::npos);

    CouplingFormContribution contribution;
    contribution.contribution_name = "skipped_optional_use";
    contribution.origin = "SkippedOptionalFieldTest";
    contribution.field_uses = {{
        .participant_name = "optional_instance",
        .field_name = "lambda",
    }};
    const std::array<CouplingFormContribution, 1> contributions{contribution};
    const auto contribution_validation = validateFormContributionDeclarations(
        std::span<const CouplingContractDeclaration>(skipped_declarations),
        std::span<const CouplingFormContribution>(contributions));

    EXPECT_FALSE(contribution_validation.ok());
    EXPECT_NE(formatDiagnostics(contribution_validation).find(
                  "disabled optional additional field is referenced by a form contribution"),
              std::string::npos);

    skipped_owner.additional_fields[0].enabled = true;
    const std::array<CouplingContractDeclaration, 2> selected_declarations{
        skipped_owner,
        dependent,
    };
    CouplingGraph selected_graph;
    const auto selected_graph_validation = selected_graph.buildDeclarationGraph(
        graphContext(),
        std::span<const CouplingContractDeclaration>(selected_declarations));
    EXPECT_TRUE(selected_graph_validation.ok())
        << formatDiagnostics(selected_graph_validation);
    EXPECT_TRUE(validateFormContributionDeclarations(
                    std::span<const CouplingContractDeclaration>(
                        selected_declarations),
                    std::span<const CouplingFormContribution>(contributions))
                    .ok());
}

TEST(CouplingGraph, RejectsAdditionalFieldsThatCannotLowerToFieldRegistration)
{
    auto compatible = graphDeclaration();
    auto inferred_components = graphAdditionalField(
        CouplingAdditionalFieldNamespace::Participant,
        "left",
        "lambda");
    inferred_components.components = 0;
    compatible.additional_fields.push_back(inferred_components);
    EXPECT_TRUE(buildGraph(graphContext(), compatible).ok());

    auto missing_target = graphDeclaration();
    missing_target.contract_name = "missing_target_instance";
    missing_target.additional_fields.push_back(graphAdditionalField(
        CouplingAdditionalFieldNamespace::Participant,
        "unknown_participant",
        "lambda"));

    const auto validation = buildGraph(graphContext(), missing_target);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("cannot be lowered to an FE field registration target"),
              std::string::npos);
    EXPECT_NE(text.find("participant='unknown_participant'"),
              std::string::npos);
}

TEST(CouplingGraph, ValidatesInterfaceAdditionalFieldMarkerResolution)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Participant,
        .namespace_name = "left",
        .field_name = "trace",
        .space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1),
        .components = 1,
        .scope = CouplingAdditionalFieldScope::InterfaceFace,
        .region_name = "interface",
    });
    declaration.additional_fields.push_back({
        .field_namespace = CouplingAdditionalFieldNamespace::Contract,
        .namespace_name = "generic_instance",
        .field_name = "lambda",
        .space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1),
        .components = 1,
        .scope = CouplingAdditionalFieldScope::InterfaceFace,
        .shared_region_name = "interface",
    });

    const auto context = interfaceGraphContext(17, 17);
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    CouplingGraph graph;
    const auto validation = graph.buildDeclarationGraph(
        context,
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const MonolithicCouplingBuilder builder;
    const auto resolved = builder.resolveAdditionalFields(
        context,
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_EQ(resolved.size(), 2u);
    EXPECT_EQ(resolved[0].declaration.region_name.value(), "interface");
    EXPECT_EQ(resolved[0].field_spec.interface_marker, 17);
    EXPECT_EQ(resolved[1].declaration.shared_region_name.value(), "interface");
    EXPECT_EQ(resolved[1].field_spec.interface_marker, 17);

    CouplingGraph mismatch_graph;
    const auto mismatch = mismatch_graph.buildDeclarationGraph(
        interfaceGraphContext(17, 18),
        std::span<const CouplingContractDeclaration>(declarations));
    EXPECT_FALSE(mismatch.ok());
    EXPECT_NE(formatDiagnostics(mismatch).find(
                  "shared-region markers must agree"),
              std::string::npos);

    auto missing_region = declaration;
    missing_region.additional_fields[0].region_name = "missing_interface";
    const std::array<CouplingContractDeclaration, 1> missing_declarations{
        missing_region,
    };
    CouplingGraph missing_graph;
    const auto missing = missing_graph.buildDeclarationGraph(
        context,
        std::span<const CouplingContractDeclaration>(missing_declarations));
    EXPECT_FALSE(missing.ok());
    EXPECT_NE(formatDiagnostics(missing).find(
                  "participant region is missing"),
              std::string::npos);
}

TEST(CouplingGraph, RejectsInvalidAdditionalFieldAttachmentCombinations)
{
    auto volume_with_region = graphDeclaration();
    volume_with_region.contract_name = "volume_with_region_instance";
    auto invalid_volume = graphAdditionalField(
        CouplingAdditionalFieldNamespace::Participant,
        "left",
        "lambda");
    invalid_volume.scope = CouplingAdditionalFieldScope::VolumeCell;
    invalid_volume.region_name = "surface";
    volume_with_region.additional_fields.push_back(invalid_volume);

    auto interface_without_region = graphDeclaration();
    interface_without_region.contract_name = "interface_without_region_instance";
    auto missing_attachment = graphAdditionalField(
        CouplingAdditionalFieldNamespace::Participant,
        "left",
        "trace");
    missing_attachment.scope = CouplingAdditionalFieldScope::InterfaceFace;
    interface_without_region.additional_fields.push_back(missing_attachment);

    auto interface_with_both = graphDeclaration();
    interface_with_both.contract_name = "interface_with_both_instance";
    auto duplicate_attachment = graphAdditionalField(
        CouplingAdditionalFieldNamespace::Participant,
        "left",
        "trace");
    duplicate_attachment.scope = CouplingAdditionalFieldScope::InterfaceFace;
    duplicate_attachment.region_name = "surface";
    duplicate_attachment.shared_region_name = "interface";
    interface_with_both.additional_fields.push_back(duplicate_attachment);

    const std::array<CouplingContractDeclaration, 3> declarations{
        volume_with_region,
        interface_without_region,
        interface_with_both,
    };
    CouplingGraph graph;
    const auto validation = graph.buildDeclarationGraph(
        graphContext(),
        std::span<const CouplingContractDeclaration>(declarations));

    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("volume additional fields must not declare region attachments"),
              std::string::npos);
    EXPECT_NE(text.find("interface additional fields require exactly one region attachment"),
              std::string::npos);
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

TEST(CouplingGraph, AcceptsDeclaredImplicitDependencyFromFieldUseMetadata)
{
    auto declaration = twoParticipantDependencyDeclaration();
    declaration.expected_blocks.clear();

    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "field_use_coupling";
    metadata.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledDependencies,
        analysis::FormBridgeFeatureStatus::Available,
        "Synthetic field-use dependency fixture"});
    metadata.field_uses.push_back(CouplingFormFieldProvenance{
        .residual_row = 2,
        .field = 1,
        .appears_as_state_field = true,
    });

    const std::vector<CouplingFormAnalysisMetadata> installed_forms{metadata};
    const auto validation = buildFinalizedGraph(
        twoParticipantGraphContext(),
        declaration,
        installed_forms);

    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(CouplingGraph, AcceptsDeclaredImplicitDependencyFromVariableMetadata)
{
    auto declaration = twoParticipantDependencyDeclaration();
    declaration.expected_blocks.clear();
    declaration.dependencies[0].dependency = CouplingVariableUse{
        .kind = CouplingVariableKind::GlobalScalar,
        .name = "lambda",
    };

    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "variable_coupling";
    metadata.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledDependencies,
        analysis::FormBridgeFeatureStatus::Available,
        "Synthetic variable dependency fixture"});
    metadata.variable_dependencies.push_back(CouplingFormVariableDependencyProvenance{
        .residual_row = analysis::VariableKey::field(2),
        .dependency = analysis::VariableKey::named(
            analysis::VariableKind::GlobalScalar,
            "lambda"),
        .mode = CouplingDependencyMode::ImplicitMonolithic,
        .domain = analysis::DomainKind::Global,
        .contributes_vector = true,
        .provider = "forms",
    });

    const std::vector<CouplingFormAnalysisMetadata> installed_forms{metadata};
    const auto validation = buildFinalizedGraph(
        twoParticipantGraphContext(),
        declaration,
        installed_forms);

    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(CouplingGraph, RejectsIncompleteExpertInstallMetadataRecords)
{
    const auto expect_rejected = [](CouplingInstallMetadata metadata) {
        EXPECT_THROW(static_cast<void>(
                         MonolithicCouplingBuilder::adaptInstallMetadata(metadata)),
                     InvalidArgumentException);
    };

    {
        auto metadata = expertInstallMetadata();
        metadata.contribution_name.clear();
        expect_rejected(std::move(metadata));
    }
    {
        auto metadata = expertInstallMetadata();
        metadata.system_name.clear();
        expect_rejected(std::move(metadata));
    }
    {
        auto metadata = expertInstallMetadata();
        metadata.installed_dependencies[0].residual_row = analysis::VariableKey{};
        expect_rejected(std::move(metadata));
    }
    {
        auto metadata = expertInstallMetadata();
        metadata.installed_dependencies[0].provider.clear();
        expect_rejected(std::move(metadata));
    }
    {
        auto metadata = expertInstallMetadata();
        metadata.installed_dependencies[0].contributes_matrix_block = false;
        metadata.installed_dependencies[0].contributes_vector = false;
        expect_rejected(std::move(metadata));
    }
    {
        auto metadata = expertInstallMetadata();
        metadata.installed_blocks[0].domains.clear();
        expect_rejected(std::move(metadata));
    }
    {
        auto metadata = expertInstallMetadata();
        metadata.installed_blocks[0].has_matrix = false;
        metadata.installed_blocks[0].has_vector = false;
        expect_rejected(std::move(metadata));
    }
}

TEST(CouplingGraph, FallbackGraphAnalyzerConsumesFormulationNonFieldDependencies)
{
    analysis::ProblemAnalysisContext context;
    analysis::FormulationRecord record;
    record.operator_tag = "coupled_form";
    record.active_fields = {2};

    const auto row = analysis::VariableKey::field(2);
    const auto boundary = analysis::VariableKey::named(
        analysis::VariableKind::BoundaryFunctional,
        "flow");
    const auto aux_input = analysis::VariableKey::named(
        analysis::VariableKind::AuxiliaryInput,
        "driver/inlet");
    const auto aux_output = analysis::VariableKey::named(
        analysis::VariableKind::AuxiliaryOutput,
        "model/pressure");
    const auto global = analysis::VariableKey::named(
        analysis::VariableKind::GlobalScalar,
        "lambda");

    record.boundary_functional_dependencies.push_back(boundary);
    record.auxiliary_input_dependencies.push_back(aux_input);
    record.auxiliary_output_dependencies.push_back(aux_output);
    record.variable_couplings.emplace_back(row, global);
    context.addFormulationRecord(record);

    analysis::ProblemAnalysisReport report;
    analysis::CouplingGraphAnalyzer analyzer;
    analyzer.run(context, report);

    EXPECT_TRUE(hasAnalysisCouplingClaim(
        report, row, boundary, analysis::DomainKind::CoupledBoundary));
    EXPECT_TRUE(hasAnalysisCouplingClaim(
        report, row, aux_input, analysis::DomainKind::AuxiliaryCoupling));
    EXPECT_TRUE(hasAnalysisCouplingClaim(
        report, row, aux_output, analysis::DomainKind::AuxiliaryCoupling));
    EXPECT_TRUE(hasAnalysisCouplingClaim(
        report, row, global, analysis::DomainKind::Global));
}

TEST(CouplingGraph, ValidatesNonFieldDependencyExpectationsThroughVariableKeys)
{
    NonFieldGraphFixture fixture(true);
    auto declaration = graphDeclaration();
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::AuxiliaryState,
        .participant_name = "left",
        .name = "aux_state",
    });
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::AuxiliaryInput,
        .participant_name = "left",
        .name = "driver_input",
    });
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::AuxiliaryOutput,
        .participant_name = "left",
        .name = "out_value",
    });
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::BoundaryFunctional,
        .participant_name = "left",
        .name = "surface_measure",
    });

    const CouplingVariableUse row{
        .kind = CouplingVariableKind::Field,
        .participant_name = "left",
        .name = "primary",
    };
    auto add_dependency = [&](CouplingVariableKind kind,
                              std::string name,
                              analysis::VariableKind analysis_kind) {
        CouplingVariableUse dependency{
            .kind = kind,
            .participant_name = "left",
            .name = std::move(name),
        };
        declaration.dependencies.push_back(CouplingResidualDependency{
            .residual_row = row,
            .dependency = dependency,
        });
        declaration.expected_blocks.push_back(CouplingBlockExpectation{
            .residual_row = row,
            .dependency = dependency,
            .expect_matrix_block = false,
        });
        return analysis::VariableKey::named(
            analysis_kind,
            "left_system/" + dependency.name);
    };

    std::vector<analysis::VariableKey> dependencies;
    dependencies.push_back(add_dependency(
        CouplingVariableKind::BoundaryFunctional,
        "surface_measure",
        analysis::VariableKind::BoundaryFunctional));
    dependencies.push_back(add_dependency(CouplingVariableKind::AuxiliaryState,
                                          "aux_state",
                                          analysis::VariableKind::AuxiliaryState));
    dependencies.push_back(add_dependency(CouplingVariableKind::AuxiliaryInput,
                                          "driver_input",
                                          analysis::VariableKind::AuxiliaryInput));
    dependencies.push_back(add_dependency(CouplingVariableKind::AuxiliaryOutput,
                                          "out_value",
                                          analysis::VariableKind::AuxiliaryOutput));
    dependencies.push_back(add_dependency(CouplingVariableKind::GlobalScalar,
                                          "coefficient",
                                          analysis::VariableKind::GlobalScalar));

    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "non_field_expectation_coupling";
    metadata.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledDependencies,
        analysis::FormBridgeFeatureStatus::Available,
        "Synthetic non-field expectation fixture"});
    metadata.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledBlocks,
        analysis::FormBridgeFeatureStatus::Available,
        "Synthetic non-field expectation fixture"});
    for (const auto& dependency : dependencies) {
        metadata.variable_dependencies.push_back(
            CouplingFormVariableDependencyProvenance{
                .residual_row = analysis::VariableKey::field(fixture.field),
                .dependency = dependency,
                .mode = CouplingDependencyMode::ImplicitMonolithic,
                .domain = analysis::DomainKind::Global,
                .contributes_vector = true,
                .provider = "forms",
            });
    }

    const std::vector<CouplingFormAnalysisMetadata> installed_forms{metadata};
    const auto validation = buildFinalizedGraph(
        fixture.context,
        declaration,
        installed_forms);
    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);

    declaration.expected_blocks[0].expect_matrix_block = true;
    const auto matrix_validation = buildFinalizedGraph(
        fixture.context,
        declaration,
        installed_forms);
    EXPECT_FALSE(matrix_validation.ok());
    EXPECT_NE(formatDiagnostics(matrix_validation).find(
                  "expected monolithic block is missing installed matrix evidence"),
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

TEST(CouplingGraph, RecordsResolvedPartitionedExchangeNodes)
{
    const auto context = twoParticipantGraphContext();
    const auto declaration = partitionedGraphDeclaration();
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};

    const PartitionedCouplingPlanGenerator generator;
    const auto plan = generator.generate(
        context,
        std::span<const CouplingContractDeclaration>(declarations));

    CouplingGraph graph;
    const std::array<CouplingFormAnalysisMetadata, 0> installed_forms{};
    const auto validation = graph.buildFinalizedGraph(
        context,
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed_forms),
        plan);
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto& snapshot = graph.snapshot();
    ASSERT_EQ(snapshot.resolved_partitioned_exchanges.size(),
              plan.exchanges.size());
    EXPECT_EQ(snapshot.resolved_partitioned_exchanges[0]
                  .exchange.producer_port.port_name,
              "left_out");
}

TEST(CouplingGraph, RejectsInvalidPartitionedExchangeDeclarations)
{
    const auto context = twoParticipantGraphContext();

    {
        auto declaration = partitionedGraphDeclaration();
        declaration.partitioned_exchange_declarations[0]
            .producer->participant_name = "missing";
        const auto validation = buildGraph(context, declaration);
        EXPECT_FALSE(validation.ok());
        EXPECT_NE(formatDiagnostics(validation).find("missing from the context"),
                  std::string::npos);
    }
    {
        auto declaration = partitionedGraphDeclaration();
        declaration.partitioned_exchange_declarations[0]
            .producer->endpoint_name.clear();
        const auto validation = buildGraph(context, declaration);
        EXPECT_FALSE(validation.ok());
        EXPECT_NE(formatDiagnostics(validation).find("registry key"),
                  std::string::npos);
    }
    {
        auto declaration = partitionedGraphDeclaration();
        declaration.partitioned_exchange_declarations[0]
            .producer->endpoint_name = "missing";
        const auto validation = buildGraph(context, declaration);
        EXPECT_FALSE(validation.ok());
        EXPECT_NE(formatDiagnostics(validation).find("field endpoint is missing"),
                  std::string::npos);
    }
    {
        auto declaration = partitionedGraphDeclaration();
        declaration.partitioned_exchange_declarations[0]
            .producer->temporal = CouplingTemporalSlotDescriptor{
                .slot = CouplingTemporalSlot::History,
            };
        const auto validation = buildGraph(context, declaration);
        EXPECT_FALSE(validation.ok());
        EXPECT_NE(formatDiagnostics(validation).find("history temporal slots"),
                  std::string::npos);
    }
    {
        auto declaration = partitionedGraphDeclaration();
        declaration.partitioned_exchange_declarations[0]
            .producer->temporal = CouplingTemporalSlotDescriptor{
                .slot = CouplingTemporalSlot::Current,
                .stage_index = 0,
            };
        const auto validation = buildGraph(context, declaration);
        EXPECT_FALSE(validation.ok());
        EXPECT_NE(formatDiagnostics(validation).find("stage index is valid only"),
                  std::string::npos);
    }
    {
        auto declaration = partitionedGraphDeclaration();
        declaration.partitioned_exchange_declarations[0].value.components = 0;
        const auto validation = buildGraph(context, declaration);
        EXPECT_FALSE(validation.ok());
        EXPECT_NE(formatDiagnostics(validation).find("scalar coupling values"),
                  std::string::npos);
    }
    {
        auto declaration = partitionedGraphDeclaration();
        declaration.partitioned_exchange_declarations[0].value = CouplingValueDescriptor{
            .rank = CouplingValueRank::Vector,
            .components = 2,
        };
        const auto validation = buildGraph(context, declaration);
        EXPECT_FALSE(validation.ok());
        EXPECT_NE(formatDiagnostics(validation).find("component count does not match"),
                  std::string::npos);
    }
    {
        auto declaration = partitionedGraphDeclaration();
        declaration.partitioned_exchange_declarations[0].transfer.kind =
            CouplingTransferKind::Unspecified;
        const auto validation = buildGraph(context, declaration);
        EXPECT_FALSE(validation.ok());
        EXPECT_NE(formatDiagnostics(validation).find("explicit transfer"),
                  std::string::npos);
    }
}

TEST(CouplingGraph, RejectsPartitionedEndpointsMissingProviderRegistryEntries)
{
    NonFieldGraphFixture fixture(/*register_variables=*/false);

    auto declaration = graphDeclaration();
    declaration.partitioned_exchange_declarations.push_back(CouplingExchangeDeclaration{
        .producer_port = {
            .contract_instance_name = "generic_instance",
            .port_name = "parameter_out",
        },
        .consumer_port = {
            .contract_instance_name = "generic_instance",
            .port_name = "field_in",
        },
        .value = CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        .producer = CouplingEndpointRef{
            .kind = CouplingEndpointKind::Parameter,
            .participant_name = "left",
            .endpoint_name = "missing_parameter",
            .temporal = CouplingTemporalSlotDescriptor{
                .slot = CouplingTemporalSlot::Current,
            },
        },
        .consumer = graphFieldEndpoint("left"),
        .transfer = CouplingTransferDeclaration{
            .kind = CouplingTransferKind::Identity,
        },
    });

    auto parameter_validation = buildGraph(fixture.context, declaration);
    EXPECT_FALSE(parameter_validation.ok());
    EXPECT_NE(formatDiagnostics(parameter_validation).find(
                  "ParameterRegistry entry"),
              std::string::npos);

    declaration.partitioned_exchange_declarations[0].producer =
        CouplingEndpointRef{
            .kind = CouplingEndpointKind::AuxiliaryInput,
            .participant_name = "left",
            .endpoint_name = "missing_input",
            .temporal = CouplingTemporalSlotDescriptor{
                .slot = CouplingTemporalSlot::Current,
            },
        };

    const auto auxiliary_validation = buildGraph(fixture.context, declaration);
    EXPECT_FALSE(auxiliary_validation.ok());
    EXPECT_NE(formatDiagnostics(auxiliary_validation).find(
                  "AuxiliaryInputRegistry entry"),
              std::string::npos);
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
