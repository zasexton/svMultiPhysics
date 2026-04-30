#include "Coupling/CouplingGraph.h"
#include "Coupling/PartitionedCouplingPlanGenerator.h"

#include "Assembly/AssemblyContext.h"
#include "Assembly/AssemblyKernel.h"
#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryStateManager.h"
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
