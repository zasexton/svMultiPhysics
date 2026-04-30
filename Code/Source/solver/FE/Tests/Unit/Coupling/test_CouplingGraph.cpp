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
#include "Mesh/Core/InterfaceMesh.h"
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

const systems::FESystem* otherGraphSystemToken()
{
    return reinterpret_cast<const systems::FESystem*>(2);
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

CouplingContext duplicateRawFieldIdGraphContext()
{
    const auto* left_system = graphSystemToken();
    const auto* right_system = otherGraphSystemToken();
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "left",
        .system_name = "left_system",
        .system = left_system,
    });
    builder.addParticipant({
        .participant_name = "right",
        .system_name = "right_system",
        .system = right_system,
    });
    builder.addField({
        .participant_name = "left",
        .system_name = "left_system",
        .system = left_system,
        .field_name = "primary",
        .field_id = 1,
        .space = space,
        .components = 1,
    });
    builder.addField({
        .participant_name = "right",
        .system_name = "right_system",
        .system = right_system,
        .field_name = "primary",
        .field_id = 1,
        .space = space,
        .components = 1,
    });
    return builder.build();
}

CouplingContext multiParticipantGraphContext()
{
    const auto* system = graphSystemToken();
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);

    CouplingContextBuilder builder;
    const std::array<std::string, 3> participants{"left", "middle", "right"};
    for (std::size_t i = 0; i < participants.size(); ++i) {
        builder.addParticipant({
            .participant_name = participants[i],
            .system_name = "system",
            .system = system,
        });
        builder.addField({
            .participant_name = participants[i],
            .system_name = "system",
            .system = system,
            .field_name = "primary",
            .field_id = static_cast<FieldId>(i + 1u),
            .space = space,
            .components = 1,
        });
    }
    return builder.build();
}

CouplingVariableUse graphFieldVariable(std::string participant)
{
    return CouplingVariableUse{
        .kind = CouplingVariableKind::Field,
        .participant_name = std::move(participant),
        .name = "primary",
    };
}

CouplingResidualDependency graphFieldDependency(std::string row_participant,
                                                std::string dependency_participant)
{
    return CouplingResidualDependency{
        .residual_row = graphFieldVariable(std::move(row_participant)),
        .dependency = graphFieldVariable(std::move(dependency_participant)),
    };
}

CouplingContractDeclaration graphNParticipantContract(
    std::string contract_name,
    std::string row_participant,
    std::string dependency_participant)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "n_participant_scalar";
    declaration.contract_name = std::move(contract_name);
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "middle"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
    declaration.fields.push_back({.participant_name = "middle", .field_name = "primary"});
    declaration.fields.push_back({.participant_name = "right", .field_name = "primary"});
    declaration.shared_regions.push_back({
        .shared_region_name = "chain_boundary",
        .required_region_kind = CouplingRegionKind::Boundary,
    });
    declaration.dependencies.push_back(graphFieldDependency(
        std::move(row_participant),
        std::move(dependency_participant)));
    declaration.expected_blocks.push_back(CouplingBlockExpectation{
        .residual_row = declaration.dependencies.back().residual_row,
        .dependency = declaration.dependencies.back().dependency,
    });
    return declaration;
}

CouplingFormAnalysisMetadata graphNParticipantInstalledBlock(
    std::string contribution_name,
    FieldId row,
    FieldId dependency)
{
    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = std::move(contribution_name);
    metadata.origin = "NParticipantGraphFixture";
    metadata.system_name = "system";
    metadata.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledDependencies,
        analysis::FormBridgeFeatureStatus::Available,
        "Synthetic N-participant dependency fixture"});
    metadata.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledBlocks,
        analysis::FormBridgeFeatureStatus::Available,
        "Synthetic N-participant block fixture"});
    metadata.installed_fields = {row, dependency};
    metadata.installed_dependencies.push_back(CouplingInstalledDependency{
        .residual_row = analysis::VariableKey::field(row),
        .dependency = analysis::VariableKey::field(dependency),
        .mode = CouplingDependencyMode::ImplicitMonolithic,
        .domain = analysis::DomainKind::Boundary,
        .contributes_matrix_block = true,
        .contributes_vector = true,
        .provider = "synthetic_fixture",
    });
    metadata.installed_blocks.push_back(CouplingInstalledBlockProvenance{
        .residual_row = analysis::VariableKey::field(row),
        .dependency = analysis::VariableKey::field(dependency),
        .domains = {analysis::DomainKind::Boundary},
        .has_matrix = true,
        .has_vector = true,
    });
    return metadata;
}

struct NParticipantGraphFixture {
    CouplingContext context;
    std::array<CouplingContractDeclaration, 2> declarations;
    std::vector<CouplingFormAnalysisMetadata> installed_forms;
};

NParticipantGraphFixture nParticipantGraphFixture()
{
    const auto* system = graphSystemToken();
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);

    const CouplingRegionRef left_region{
        .participant_name = "left",
        .system_name = "system",
        .system = system,
        .region_name = "boundary",
        .kind = CouplingRegionKind::Boundary,
        .marker = 1,
    };
    const CouplingRegionRef middle_region{
        .participant_name = "middle",
        .system_name = "system",
        .system = system,
        .region_name = "boundary",
        .kind = CouplingRegionKind::Boundary,
        .marker = 2,
    };
    const CouplingRegionRef right_region{
        .participant_name = "right",
        .system_name = "system",
        .system = system,
        .region_name = "boundary",
        .kind = CouplingRegionKind::Boundary,
        .marker = 3,
    };

    CouplingContextBuilder builder;
    const std::array<std::string, 3> participants{"left", "middle", "right"};
    for (std::size_t i = 0; i < participants.size(); ++i) {
        builder.addParticipant({
            .participant_name = participants[i],
            .system_name = "system",
            .system = system,
        });
        builder.addField({
            .participant_name = participants[i],
            .system_name = "system",
            .system = system,
            .field_name = "primary",
            .field_id = static_cast<FieldId>(i + 1u),
            .space = space,
            .components = 1,
        });
    }
    builder.addRegion(left_region);
    builder.addRegion(middle_region);
    builder.addRegion(right_region);
    builder.addSharedRegion(SharedRegionRef{
        .name = "chain_boundary",
        .required_region_kind = CouplingRegionKind::Boundary,
        .participant_regions = {left_region, middle_region, right_region},
    });

    return NParticipantGraphFixture{
        .context = builder.build(),
        .declarations = {graphNParticipantContract("left_middle_bridge",
                                                   "middle",
                                                   "left"),
                         graphNParticipantContract("middle_right_bridge",
                                                   "right",
                                                   "middle")},
        .installed_forms = {
            graphNParticipantInstalledBlock(
                "left_middle_block",
                FieldId{2},
                FieldId{1}),
            graphNParticipantInstalledBlock(
                "middle_right_block",
                FieldId{3},
                FieldId{2}),
        },
    };
}

CouplingContext interfaceGraphContext(
    int left_marker = 17,
    int right_marker = 17,
    CouplingInterfaceSide left_side = CouplingInterfaceSide::Minus,
    CouplingInterfaceSide right_side = CouplingInterfaceSide::Plus)
{
    const auto* system = graphSystemToken();
    const CouplingRegionRef left_region{
        .participant_name = "left",
        .system_name = "system",
        .system = system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = left_marker,
        .side = left_side,
    };
    const CouplingRegionRef right_region{
        .participant_name = "right",
        .system_name = "system",
        .system = system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = right_marker,
        .side = right_side,
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

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
svmp::search::LogicalInterfaceRegionId graphLogicalInterfaceRegion(
    std::string persistent_id,
    std::string name,
    int physical_label)
{
    return svmp::search::LogicalInterfaceRegionId{
        .kind = svmp::search::LogicalInterfaceRegionKind::SlidingInterface,
        .persistent_id = std::move(persistent_id),
        .name = std::move(name),
        .physical_label = physical_label,
        .provenance_epoch = 3,
    };
}

CouplingContext logicalInterfaceProviderGraphContext(int left_marker = 17,
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
        .logical_region = graphLogicalInterfaceRegion(
            "left-interface-region",
            "left_interface",
            left_marker),
    };
    const CouplingRegionRef right_region{
        .participant_name = "right",
        .system_name = "system",
        .system = system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = right_marker,
        .side = CouplingInterfaceSide::Plus,
        .logical_region = graphLogicalInterfaceRegion(
            "right-interface-region",
            "right_interface",
            right_marker),
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

svmp::search::InterfaceRevisionSnapshot graphInterfaceRevisionSnapshot(
    svmp::Configuration configuration,
    std::uint64_t offset)
{
    return svmp::search::InterfaceRevisionSnapshot{
        .configuration = configuration,
        .geometry_revision = 10 + offset,
        .reference_geometry_revision = 20 + offset,
        .current_geometry_revision = 30 + offset,
        .topology_revision = 40 + offset,
        .ownership_revision = 50 + offset,
        .numbering_revision = 60 + offset,
        .field_layout_revision = 70 + offset,
        .label_revision = 80 + offset,
        .active_configuration_epoch = 90 + offset,
    };
}

svmp::search::InterfaceMap graphInterfaceMap(std::string name)
{
    svmp::search::InterfaceMap map;
    map.name = std::move(name);
    map.source.boundary_label = 21;
    map.source.configuration = svmp::Configuration::Reference;
    map.source.logical_region =
        graphLogicalInterfaceRegion("left-interface-region", "left_interface", 21);
    map.target.boundary_label = 22;
    map.target.configuration = svmp::Configuration::Current;
    map.target.logical_region =
        graphLogicalInterfaceRegion("right-interface-region", "right_interface", 22);
    map.source_revision =
        graphInterfaceRevisionSnapshot(svmp::Configuration::Reference, 1);
    map.target_revision =
        graphInterfaceRevisionSnapshot(svmp::Configuration::Current, 2);
    map.state = svmp::search::InterfaceMapState::Trial;
    return map;
}

systems::SlidingInterfaceMap graphSlidingInterfaceMap()
{
    systems::SlidingInterfaceMap sliding_map;
    sliding_map.name = "interface_map";
    sliding_map.map_kind = systems::SlidingInterfaceMapKind::RotatingSliding;
    sliding_map.interface_map = graphInterfaceMap("interface_map");
    sliding_map.state = systems::InterfaceOperatorState::Trial;
    sliding_map.accepted_revision_key = 47;
    sliding_map.trial_revision_key = 53;
    sliding_map.time = 0.125;
    sliding_map.time_level_epoch = 59;
    return sliding_map;
}

CouplingContext interfacePartitionedGraphContext(
    const svmp::search::InterfaceSearchRegistry& registry,
    const systems::SlidingInterfaceMap& sliding_map)
{
    const auto* left_system = graphSystemToken();
    const auto* right_system = otherGraphSystemToken();
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);

    const auto source_logical =
        graphLogicalInterfaceRegion("left-interface-region", "left_interface", 21);
    const auto target_logical =
        graphLogicalInterfaceRegion("right-interface-region", "right_interface", 22);
    const auto source_revision =
        graphInterfaceRevisionSnapshot(svmp::Configuration::Reference, 1);
    const auto target_revision =
        graphInterfaceRevisionSnapshot(svmp::Configuration::Current, 2);

    const CouplingRegionRef left_region{
        .participant_name = "left",
        .system_name = "left_system",
        .system = left_system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 21,
        .side = CouplingInterfaceSide::Minus,
        .coordinate_configuration = CouplingCoordinateConfiguration::Reference,
        .logical_region = source_logical,
        .revision_snapshot = source_revision,
    };
    const CouplingRegionRef right_region{
        .participant_name = "right",
        .system_name = "right_system",
        .system = right_system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 22,
        .side = CouplingInterfaceSide::Plus,
        .coordinate_configuration = CouplingCoordinateConfiguration::Current,
        .logical_region = target_logical,
        .revision_snapshot = target_revision,
    };

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "left",
        .system_name = "left_system",
        .system = left_system,
    });
    builder.addParticipant({
        .participant_name = "right",
        .system_name = "right_system",
        .system = right_system,
    });
    builder.addField({
        .participant_name = "left",
        .system_name = "left_system",
        .system = left_system,
        .field_name = "primary",
        .field_id = 1,
        .space = space,
        .components = 3,
    });
    builder.addField({
        .participant_name = "right",
        .system_name = "right_system",
        .system = right_system,
        .field_name = "primary",
        .field_id = 2,
        .space = space,
        .components = 3,
    });
    builder.addRegion(left_region);
    builder.addRegion(right_region);
    builder.addSharedRegion(SharedRegionRef{
        .name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .participant_regions = {left_region, right_region},
    });
    builder.addInterfaceSearchRegistry(CouplingInterfaceSearchRegistryRegistration{
        .registry_name = "interface_search",
        .registry = &registry,
    });
    builder.addSlidingInterfaceMap(CouplingSlidingInterfaceMapRegistration{
        .interface_map_name = "interface_map",
        .sliding_map = &sliding_map,
    });
    return builder.build();
}

CouplingInterfaceMapProvenance graphInterfaceMapProvenance()
{
    const auto map = graphInterfaceMap("interface_map");
    return CouplingInterfaceMapProvenance{
        .interface_map_name = "interface_map",
        .interface_entry_name = "interface_entry",
        .interface_search_registry_name = "interface_search",
        .source_system_name = "left_system",
        .target_system_name = "right_system",
        .source_interface_marker = 21,
        .target_interface_marker = 22,
        .sliding_map_kind = systems::SlidingInterfaceMapKind::RotatingSliding,
        .source_configuration = svmp::Configuration::Reference,
        .target_configuration = svmp::Configuration::Current,
        .source_logical_region = graphLogicalInterfaceRegion(
            "left-interface-region",
            "left_interface",
            21),
        .target_logical_region = graphLogicalInterfaceRegion(
            "right-interface-region",
            "right_interface",
            22),
        .source_revision_snapshot = graphInterfaceRevisionSnapshot(
            svmp::Configuration::Reference,
            1),
        .target_revision_snapshot = graphInterfaceRevisionSnapshot(
            svmp::Configuration::Current,
            2),
        .source_search_revision_key = map.source_revision.revision_key(),
        .target_search_revision_key = map.target_revision.revision_key(),
        .map_revision_key = map.revision_key(),
        .map_state = svmp::search::InterfaceMapState::Trial,
        .operator_state = systems::InterfaceOperatorState::Trial,
        .accepted_revision_key = 47,
        .trial_revision_key = 53,
        .time = 0.125,
        .time_level_epoch = 59,
    };
}
#endif

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

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
CouplingContractDeclaration interfacePartitionedGraphDeclaration()
{
    auto interface_declaration = CouplingInterfaceTransferDeclaration{
        .frame_policy = CouplingInterfaceFramePolicy::SourceToTargetVector,
    };
    interface_declaration.source_to_target_rotation[0][0] = 0.0;
    interface_declaration.source_to_target_rotation[0][1] = -1.0;
    interface_declaration.source_to_target_rotation[1][0] = 1.0;
    interface_declaration.source_to_target_rotation[1][1] = 0.0;
    interface_declaration.conservation_tolerance = 1.0e-9;

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
            .port_name = "left_interface_out",
        },
        .consumer_port = {
            .contract_instance_name = "generic_instance",
            .port_name = "right_interface_in",
        },
        .value = CouplingValueDescriptor{
            .rank = CouplingValueRank::Vector,
            .components = 3,
        },
        .producer = graphFieldEndpoint("left"),
        .consumer = graphFieldEndpoint("right"),
        .shared_region_name = "interface",
        .producer_region = CouplingRegionEndpointDeclaration{
            .participant_name = "left",
            .region_name = "interface",
        },
        .consumer_region = CouplingRegionEndpointDeclaration{
            .participant_name = "right",
            .region_name = "interface",
        },
        .transfer = CouplingTransferDeclaration{
            .kind = CouplingTransferKind::InterfacePointwiseInterpolation,
            .interface_declaration = interface_declaration,
            .interface_map = graphInterfaceMapProvenance(),
        },
    });
    declaration.group_hints.push_back(CouplingGroupHint{
        .name = "interface_pair",
        .participant_names = {"left", "right"},
    });
    return declaration;
}
#endif

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

TEST(CouplingGraph, ValidatesFieldRequirementShapeAgainstContext)
{
    auto declaration = graphDeclaration();
    declaration.field_requirements.push_back({
        .field = {.participant_name = "left", .field_name = "primary"},
        .value = {.rank = CouplingValueRank::Scalar, .components = 1},
        .required_scope = systems::FieldScope::VolumeCell,
    });

    EXPECT_TRUE(buildGraph(graphContext(), declaration).ok());

    declaration.field_requirements.front().value =
        CouplingValueDescriptor{.rank = CouplingValueRank::Vector, .components = 3};
    declaration.field_requirements.front().required_scope =
        systems::FieldScope::InterfaceFace;

    const auto validation = buildGraph(graphContext(), declaration);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("coupling field component count does not satisfy"),
              std::string::npos);
    EXPECT_NE(text.find("coupling field scope does not satisfy"),
              std::string::npos);
}

TEST(CouplingGraph, ValidatesRequiredFieldRequirementPresence)
{
    auto declaration = graphDeclaration();
    declaration.field_requirements.push_back({
        .field = {.participant_name = "left", .field_name = "missing"},
        .value = {.rank = CouplingValueRank::Scalar, .components = 1},
    });

    const auto validation = buildGraph(graphContext(), declaration);
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find(
                  "required coupling field-shape field is missing from the context"),
              std::string::npos);
}

TEST(CouplingGraph, ValidatesSharedInterfaceRequirementsAgainstContext)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.shared_interface_requirements.push_back({
        .shared_region_name = "interface",
        .participant_names = {"left", "right"},
    });

    EXPECT_TRUE(buildGraph(interfaceGraphContext(), declaration).ok());

    auto missing_mapping = declaration;
    missing_mapping.shared_interface_requirements.front().participant_names = {
        "left",
        "middle",
    };
    const auto missing_validation =
        buildGraph(interfaceGraphContext(), missing_mapping);
    EXPECT_FALSE(missing_validation.ok());
    EXPECT_NE(formatDiagnostics(missing_validation).find(
                  "shared-interface participant mapping is missing from the context"),
              std::string::npos);

    CouplingContractDeclaration boundary_declaration;
    boundary_declaration.contract_type = "generic";
    boundary_declaration.contract_name = "generic_instance";
    boundary_declaration.participants.push_back({.participant_name = "left"});
    boundary_declaration.shared_interface_requirements.push_back({
        .shared_region_name = "interface",
        .participant_names = {"left"},
    });
    const auto boundary_validation = buildGraph(graphContext(), boundary_declaration);
    EXPECT_FALSE(boundary_validation.ok());
    EXPECT_NE(formatDiagnostics(boundary_validation).find(
                  "shared-interface participant mapping does not satisfy the declaration"),
              std::string::npos);
}

TEST(CouplingGraph, ValidatesRegionRelationRequirementsAgainstContext)
{
    const auto fixture = nParticipantGraphFixture();
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "middle"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.region_relation_requirements.push_back({
        .relation_name = "chain_balance",
        .relation_kind = CouplingRegionRelationKind::NWayInterface,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "boundary",
                .shared_region_name = "chain_boundary",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "middle",
                .region_name = "boundary",
                .shared_region_name = "chain_boundary",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "right",
                .region_name = "boundary",
                .shared_region_name = "chain_boundary",
            },
        },
        .lowering_capabilities = {
            CouplingRelationLoweringCapability{
                .lowering_kind =
                    CouplingRelationLoweringKind::PartitionedExchange,
            },
        },
        .required_region_kind = CouplingRegionKind::Boundary,
    });

    EXPECT_TRUE(buildGraph(fixture.context, declaration).ok());

    auto missing_endpoint = declaration;
    missing_endpoint.region_relation_requirements.front()
        .endpoints[1]
        .region_name = "missing";
    const auto missing_validation = buildGraph(fixture.context, missing_endpoint);
    EXPECT_FALSE(missing_validation.ok());
    EXPECT_NE(formatDiagnostics(missing_validation).find(
                  "region-relation endpoint is missing from the context"),
              std::string::npos);

    auto optional_endpoint = missing_endpoint;
    optional_endpoint.region_relation_requirements.front().require_all_endpoints =
        false;
    EXPECT_TRUE(buildGraph(fixture.context, optional_endpoint).ok());

    auto bad_kind = declaration;
    bad_kind.region_relation_requirements.front().required_region_kind =
        CouplingRegionKind::InterfaceFace;
    const auto kind_validation = buildGraph(fixture.context, bad_kind);
    EXPECT_FALSE(kind_validation.ok());
    EXPECT_NE(formatDiagnostics(kind_validation).find(
                  "region-relation endpoint kind does not satisfy the declaration"),
              std::string::npos);

    auto bad_shared_region = declaration;
    bad_shared_region.region_relation_requirements.front()
        .endpoints[0]
        .shared_region_name = "missing_shared_region";
    const auto shared_region_validation =
        buildGraph(fixture.context, bad_shared_region);
    EXPECT_FALSE(shared_region_validation.ok());
    EXPECT_NE(formatDiagnostics(shared_region_validation).find(
                  "region-relation endpoint shared region is missing from the context"),
              std::string::npos);
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

TEST(CouplingGraph, BuildsGraphForMultiwayContract)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "multiway";
    declaration.contract_name = "three_way_interface";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "middle"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
    declaration.fields.push_back({.participant_name = "middle", .field_name = "primary"});
    declaration.fields.push_back({.participant_name = "right", .field_name = "primary"});

    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const auto validation = graph.buildDeclarationGraph(
        multiParticipantGraphContext(),
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto& snapshot = graph.snapshot();
    EXPECT_EQ(snapshot.participants.size(), 3u);
    EXPECT_EQ(snapshot.fields.size(), 3u);
    ASSERT_EQ(snapshot.contract_instances.size(), 1u);
    EXPECT_EQ(snapshot.contract_instances[0].contract_name,
              "three_way_interface");
}

TEST(CouplingGraph, NParticipantFixtureValidatesSharedRegionReuseAndBlocks)
{
    const auto fixture = nParticipantGraphFixture();

    CouplingGraph graph;
    const auto validation = graph.buildFinalizedGraph(
        fixture.context,
        std::span<const CouplingContractDeclaration>(fixture.declarations),
        std::span<const CouplingFormAnalysisMetadata>(fixture.installed_forms));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto& snapshot = graph.snapshot();
    EXPECT_EQ(snapshot.participants.size(), 3u);
    EXPECT_EQ(snapshot.fields.size(), 3u);
    EXPECT_EQ(snapshot.regions.size(), 3u);
    ASSERT_EQ(snapshot.shared_regions.size(), 1u);
    EXPECT_EQ(snapshot.shared_regions[0].shared_region.name, "chain_boundary");
    EXPECT_EQ(snapshot.shared_regions[0]
                  .shared_region.participant_regions.size(),
              3u);
    ASSERT_EQ(snapshot.contract_types.size(), 1u);
    EXPECT_EQ(snapshot.contract_types[0].contract_type,
              "n_participant_scalar");
    ASSERT_EQ(snapshot.contract_instances.size(), 2u);
    EXPECT_EQ(snapshot.contract_instances[0].contract_name,
              "left_middle_bridge");
    EXPECT_EQ(snapshot.contract_instances[1].contract_name,
              "middle_right_bridge");
    ASSERT_EQ(snapshot.dependency_expectations.size(), 2u);
    ASSERT_TRUE(snapshot.dependency_expectations[0].residual_row.has_value());
    ASSERT_TRUE(snapshot.dependency_expectations[0].dependency.has_value());
    EXPECT_EQ(*snapshot.dependency_expectations[0].residual_row,
              analysis::VariableKey::field(FieldId{2}));
    EXPECT_EQ(*snapshot.dependency_expectations[0].dependency,
              analysis::VariableKey::field(FieldId{1}));
    ASSERT_EQ(snapshot.expected_blocks.size(), 2u);
    EXPECT_EQ(snapshot.expected_blocks[1].contract_name,
              "middle_right_bridge");
    ASSERT_EQ(graph.installedFormMetadata().size(), 2u);
    EXPECT_EQ(graph.installedFormMetadata()[1].installed_blocks[0].domains[0],
              analysis::DomainKind::Boundary);
}

TEST(CouplingGraph, NParticipantFixtureReportsDeterministicBlockDiagnostics)
{
    auto fixture = nParticipantGraphFixture();
    for (auto& metadata : fixture.installed_forms) {
        metadata.installed_dependencies[0].contributes_matrix_block = false;
        metadata.installed_blocks.clear();
    }

    const auto build_diagnostic_text = [&fixture]() {
        CouplingGraph graph;
        const auto validation = graph.buildFinalizedGraph(
            fixture.context,
            std::span<const CouplingContractDeclaration>(fixture.declarations),
            std::span<const CouplingFormAnalysisMetadata>(
                fixture.installed_forms));
        EXPECT_FALSE(validation.ok());
        return formatDiagnostics(validation);
    };

    const auto first_text = build_diagnostic_text();
    const auto second_text = build_diagnostic_text();
    EXPECT_EQ(first_text, second_text);
    EXPECT_NE(first_text.find(
                  "expected monolithic block is missing installed matrix evidence"),
              std::string::npos);
    const auto first_contract = first_text.find("left_middle_bridge");
    const auto second_contract = first_text.find("middle_right_bridge");
    ASSERT_NE(first_contract, std::string::npos);
    ASSERT_NE(second_contract, std::string::npos);
    EXPECT_LT(first_contract, second_contract);
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

TEST(CouplingGraph, AggregatesFieldHistoryAndMeshTemporalRequirementsAcrossContracts)
{
    CouplingContractDeclaration history_contract;
    history_contract.contract_type = "generic";
    history_contract.contract_name = "history_contract";
    history_contract.participants.push_back({.participant_name = "left"});
    history_contract.fields.push_back({
        .participant_name = "left",
        .field_name = "primary",
    });
    history_contract.temporal_requirements.push_back({
        .quantity = CouplingTemporalQuantity::FieldHistoryValue,
        .field = CouplingFieldUse{
            .participant_name = "left",
            .field_name = "primary",
        },
        .history_index = 2,
    });

    CouplingContractDeclaration mesh_contract;
    mesh_contract.contract_type = "generic";
    mesh_contract.contract_name = "mesh_temporal_contract";
    mesh_contract.participants.push_back({.participant_name = "right"});
    mesh_contract.temporal_requirements.push_back({
        .quantity = CouplingTemporalQuantity::MeshVelocity,
        .mesh_motion_scope = CouplingGeometryTerminalScope{
            .participant_name = "right",
        },
        .mesh_motion_role = systems::MeshMotionFieldRole::Velocity,
    });

    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "multi_temporal_form";
    metadata.temporal_symbols.push_back({
        .field = FieldId{1},
        .quantity = CouplingTemporalQuantity::FieldHistoryValue,
        .history_index = 2,
    });
    metadata.temporal_symbols.push_back({
        .mesh_motion_scope = CouplingGeometryTerminalScope{
            .participant_name = "right",
        },
        .mesh_motion_role = systems::MeshMotionFieldRole::Velocity,
        .quantity = CouplingTemporalQuantity::MeshVelocity,
    });

    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 2> declarations{
        history_contract,
        mesh_contract,
    };
    const std::array<CouplingFormAnalysisMetadata, 1> installed{metadata};
    const auto validation = graph.buildFinalizedGraph(
        twoParticipantGraphContext(),
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    ASSERT_EQ(graph.snapshot().temporal_requirements.size(), 2u);
    EXPECT_EQ(graph.snapshot().temporal_requirements[0].contract_name,
              "history_contract");
    EXPECT_EQ(graph.snapshot().temporal_requirements[1].contract_name,
              "mesh_temporal_contract");
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

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    auto logical_metadata = metadata;
    logical_metadata.non_field_dependencies[0].side =
        CouplingInterfaceSide::Minus;
    logical_metadata.non_field_dependencies[0].logical_region =
        graphLogicalInterfaceRegion("left-interface-region",
                                    "left_interface",
                                    17);
    const std::vector<CouplingFormAnalysisMetadata> logical_forms{
        logical_metadata};
    const auto logical_ok = buildFinalizedGraph(
        logicalInterfaceProviderGraphContext(17, 17),
        declaration,
        logical_forms);
    EXPECT_TRUE(logical_ok.ok()) << formatDiagnostics(logical_ok);

    auto wrong_logical_region = logical_metadata;
    wrong_logical_region.non_field_dependencies[0]
        .logical_region->persistent_id = "other-interface-region";
    const std::vector<CouplingFormAnalysisMetadata> wrong_logical_forms{
        wrong_logical_region};
    const auto wrong_logical = buildFinalizedGraph(
        logicalInterfaceProviderGraphContext(17, 17),
        declaration,
        wrong_logical_forms);
    EXPECT_FALSE(wrong_logical.ok());
    EXPECT_NE(formatDiagnostics(wrong_logical).find(
                  "BoundaryIntegral(left/interface_flux)"),
              std::string::npos);
#endif
}

TEST(CouplingGraph, AcceptsBoundaryIntegralProvenanceAsBoundaryFunctionalGraphIdentity)
{
    NonFieldGraphFixture fixture(/*register_variables=*/true);
    auto declaration = graphDeclaration();
    declaration.non_field_dependencies.push_back({
        .kind = CouplingNonFieldDependencyRequirementKind::BoundaryIntegral,
        .participant_name = "left",
        .name = "surface_measure",
        .region = CouplingRegionEndpointDeclaration{
            .participant_name = "left",
            .region_name = "surface",
        },
        .required_region_kind = CouplingRegionKind::Boundary,
        .expected_value_type = "scalar",
    });
    declaration.dependencies.push_back(CouplingResidualDependency{
        .residual_row = {
            .kind = CouplingVariableKind::Field,
            .participant_name = "left",
            .name = "primary",
        },
        .dependency = {
            .kind = CouplingVariableKind::BoundaryFunctional,
            .participant_name = "left",
            .name = "surface_measure",
        },
    });
    declaration.expected_blocks.push_back(CouplingBlockExpectation{
        .residual_row = declaration.dependencies.back().residual_row,
        .dependency = declaration.dependencies.back().dependency,
    });

    const auto boundary_functional =
        analysis::VariableKey::named(analysis::VariableKind::BoundaryFunctional,
                                     "left_system/surface_measure");
    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "boundary_integral_block";
    metadata.origin = "CouplingGraphTest";
    metadata.system_name = "left_system";
    metadata.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledDependencies,
        analysis::FormBridgeFeatureStatus::Available,
        "Synthetic installed dependency fixture"});
    metadata.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledBlocks,
        analysis::FormBridgeFeatureStatus::Available,
        "Synthetic installed block fixture"});
    metadata.non_field_dependencies.push_back({
        .kind = CouplingFormNonFieldDependencyKind::BoundaryIntegral,
        .participant_name = "left",
        .system_name = "left_system",
        .name = "surface_measure",
        .domain = analysis::DomainKind::Boundary,
        .region_name = "surface",
        .marker = 1,
        .provider = "forms",
        .value_type = "scalar",
    });
    metadata.installed_dependencies.push_back(CouplingInstalledDependency{
        .residual_row = analysis::VariableKey::field(fixture.field),
        .dependency = boundary_functional,
        .mode = CouplingDependencyMode::ImplicitMonolithic,
        .domain = analysis::DomainKind::Boundary,
        .contributes_matrix_block = true,
        .contributes_vector = true,
        .provider = "FormsInstaller",
    });
    metadata.installed_blocks.push_back(CouplingInstalledBlockProvenance{
        .residual_row = analysis::VariableKey::field(fixture.field),
        .dependency = boundary_functional,
        .domains = {analysis::DomainKind::Boundary},
        .has_matrix = true,
        .has_vector = true,
    });

    const std::vector<CouplingFormAnalysisMetadata> installed_forms{metadata};
    const auto validation = buildFinalizedGraph(
        fixture.context,
        declaration,
        installed_forms);
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

TEST(CouplingGraph, DistinguishesDeclarationAndFinalizedValidationForCouplingOwnedFields)
{
    auto declaration = graphDeclaration();
    declaration.additional_fields.push_back(graphAdditionalField(
        CouplingAdditionalFieldNamespace::Contract,
        "generic_instance",
        "lambda",
        "left"));
    declaration.dependencies.push_back(CouplingResidualDependency{
        .residual_row = {
            .kind = CouplingVariableKind::Field,
            .participant_name = "left",
            .name = "primary",
        },
        .dependency = {
            .kind = CouplingVariableKind::Field,
            .participant_name = "generic_instance",
            .name = "lambda",
        },
    });
    declaration.expected_blocks.push_back(CouplingBlockExpectation{
        .residual_row = declaration.dependencies.back().residual_row,
        .dependency = declaration.dependencies.back().dependency,
    });

    CouplingGraph declaration_graph;
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const auto declaration_validation =
        declaration_graph.buildDeclarationGraph(
            graphContext(),
            std::span<const CouplingContractDeclaration>(declarations));
    EXPECT_TRUE(declaration_validation.ok())
        << formatDiagnostics(declaration_validation);

    CouplingGraph finalized_graph;
    const std::array<CouplingFormAnalysisMetadata, 0> installed_forms{};
    const auto finalized_validation = finalized_graph.buildFinalizedGraph(
        graphContext(),
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed_forms));
    EXPECT_FALSE(finalized_validation.ok());
    const auto text = formatDiagnostics(finalized_validation);
    EXPECT_NE(text.find("coupling graph variable field is missing from the context"),
              std::string::npos);
    EXPECT_NE(text.find("field='lambda'"), std::string::npos);
}

TEST(CouplingGraph, RejectsInterfaceContractWithoutRegisteredTopology)
{
    NonFieldGraphFixture fixture(/*register_variables=*/false);
    const CouplingRegionRef left_interface{
        .participant_name = "left",
        .system_name = "shared_system",
        .system = &fixture.system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 17,
        .side = CouplingInterfaceSide::Minus,
    };
    const CouplingRegionRef right_interface{
        .participant_name = "right",
        .system_name = "shared_system",
        .system = &fixture.system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 17,
        .side = CouplingInterfaceSide::Plus,
    };

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "left",
        .system_name = "shared_system",
        .system = &fixture.system,
    });
    builder.addParticipant({
        .participant_name = "right",
        .system_name = "shared_system",
        .system = &fixture.system,
    });
    builder.addRegion(left_interface);
    builder.addRegion(right_interface);
    builder.addSharedRegion(SharedRegionRef{
        .name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .participant_regions = {left_interface, right_interface},
    });
    const auto context = builder.build();

    CouplingContractDeclaration declaration;
    declaration.contract_type = "interface";
    declaration.contract_name = "interface_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.shared_regions.push_back({
        .shared_region_name = "interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
    });

    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const auto validation = graph.buildDeclarationGraph(
        context,
        std::span<const CouplingContractDeclaration>(declarations));
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find(
                  "interface-face coupling region is missing registered interface topology"),
              std::string::npos);
}

TEST(CouplingGraph, ValidatesRelationEndpointRegisteredTopology)
{
    NonFieldGraphFixture fixture(/*register_variables=*/false);
    const CouplingRegionRef left_interface{
        .participant_name = "left",
        .system_name = "shared_system",
        .system = &fixture.system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 17,
        .side = CouplingInterfaceSide::Minus,
    };
    const CouplingRegionRef right_interface{
        .participant_name = "right",
        .system_name = "shared_system",
        .system = &fixture.system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 17,
        .side = CouplingInterfaceSide::Plus,
    };

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "left",
        .system_name = "shared_system",
        .system = &fixture.system,
    });
    builder.addParticipant({
        .participant_name = "right",
        .system_name = "shared_system",
        .system = &fixture.system,
    });
    builder.addRegion(left_interface);
    builder.addRegion(right_interface);
    const auto context = builder.build();

    CouplingContractDeclaration declaration;
    declaration.contract_type = "interface";
    declaration.contract_name = "interface_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.region_relation_requirements.push_back({
        .relation_name = "interface_pair",
        .relation_kind = CouplingRegionRelationKind::SidePairedInterface,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "interface",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "right",
                .region_name = "interface",
            },
        },
        .lowering_capabilities = {
            CouplingRelationLoweringCapability{
                .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
            },
        },
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .require_opposite_sides_for_side_pair = true,
        .require_registered_topology = true,
    });

    const auto missing = buildGraph(context, declaration);
    EXPECT_FALSE(missing.ok());
    EXPECT_NE(formatDiagnostics(missing).find(
                  "interface-face coupling region is missing registered interface topology"),
              std::string::npos);

    fixture.system.setInterfaceMesh(17,
                                    std::make_shared<const svmp::InterfaceMesh>());
    const auto accepted = buildGraph(context, declaration);
    EXPECT_TRUE(accepted.ok()) << formatDiagnostics(accepted);
}

TEST(CouplingGraph, ValidatesRelationCommonMonolithicSystem)
{
    auto make_context = [](bool same_system) {
        const auto* left_system = graphSystemToken();
        const auto* right_system =
            same_system ? graphSystemToken() : otherGraphSystemToken();
        const std::string left_system_name =
            same_system ? "system" : "left_system";
        const std::string right_system_name =
            same_system ? "system" : "right_system";
        const CouplingRegionRef left_region{
            .participant_name = "left",
            .system_name = left_system_name,
            .system = left_system,
            .region_name = "surface",
            .kind = CouplingRegionKind::Boundary,
            .marker = 1,
        };
        const CouplingRegionRef right_region{
            .participant_name = "right",
            .system_name = right_system_name,
            .system = right_system,
            .region_name = "surface",
            .kind = CouplingRegionKind::Boundary,
            .marker = 2,
        };

        CouplingContextBuilder builder;
        builder.addParticipant({
            .participant_name = "left",
            .system_name = left_system_name,
            .system = left_system,
        });
        builder.addParticipant({
            .participant_name = "right",
            .system_name = right_system_name,
            .system = right_system,
        });
        builder.addRegion(left_region);
        builder.addRegion(right_region);
        return builder.build();
    };

    CouplingContractDeclaration declaration;
    declaration.contract_type = "surface_pair";
    declaration.contract_name = "surface_pair_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.region_relation_requirements.push_back({
        .relation_name = "surface_pair",
        .relation_kind = CouplingRegionRelationKind::VolumeBoundaryRelation,
        .endpoints = {
            CouplingRegionEndpointDeclaration{
                .participant_name = "left",
                .region_name = "surface",
            },
            CouplingRegionEndpointDeclaration{
                .participant_name = "right",
                .region_name = "surface",
            },
        },
        .lowering_capabilities = {
            CouplingRelationLoweringCapability{
                .lowering_kind = CouplingRelationLoweringKind::MonolithicForms,
            },
        },
        .required_region_kind = CouplingRegionKind::Boundary,
        .require_common_monolithic_system = true,
    });

    EXPECT_TRUE(buildGraph(make_context(true), declaration).ok());

    const auto validation = buildGraph(make_context(false), declaration);
    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find(
                  "region-relation endpoints must resolve to one owning system"),
              std::string::npos);
}

TEST(CouplingGraph, RejectsRawFieldIdMatchesAcrossDifferentSystems)
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "cross_system_instance";
    declaration.participants.push_back({.participant_name = "left"});
    declaration.participants.push_back({.participant_name = "right"});
    declaration.fields.push_back({.participant_name = "left", .field_name = "primary"});
    declaration.fields.push_back({.participant_name = "right", .field_name = "primary"});
    declaration.dependencies.push_back(CouplingResidualDependency{
        .residual_row = {
            .kind = CouplingVariableKind::Field,
            .participant_name = "left",
            .name = "primary",
        },
        .dependency = {
            .kind = CouplingVariableKind::Field,
            .participant_name = "right",
            .name = "primary",
        },
    });
    declaration.expected_blocks.push_back(CouplingBlockExpectation{
        .residual_row = declaration.dependencies.back().residual_row,
        .dependency = declaration.dependencies.back().dependency,
    });

    auto metadata = installedDependencyMetadata();
    metadata.system_name = "left_system";
    metadata.installed_fields = {1};
    metadata.installed_dependencies[0].residual_row =
        analysis::VariableKey::field(1);
    metadata.installed_dependencies[0].dependency =
        analysis::VariableKey::field(1);
    metadata.installed_blocks[0].residual_row =
        analysis::VariableKey::field(1);
    metadata.installed_blocks[0].dependency =
        analysis::VariableKey::field(1);

    CouplingGraph graph;
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const std::array<CouplingFormAnalysisMetadata, 1> installed_forms{metadata};
    const auto validation = graph.buildFinalizedGraph(
        duplicateRawFieldIdGraphContext(),
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed_forms));

    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("fields must resolve to one owning system"),
              std::string::npos);
    EXPECT_NE(text.find("right/primary"), std::string::npos);
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

TEST(CouplingGraph, PreservesInstalledBlockDomainProvenance)
{
    auto metadata = installedDependencyMetadata();
    metadata.installed_blocks[0].domains = {
        analysis::DomainKind::Global,
        analysis::DomainKind::CoupledBoundary,
        analysis::DomainKind::AuxiliaryCoupling,
    };

    CouplingGraph graph;
    const auto declaration = twoParticipantDependencyDeclaration();
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};
    const std::array<CouplingFormAnalysisMetadata, 1> installed_forms{metadata};
    const auto validation = graph.buildFinalizedGraph(
        twoParticipantGraphContext(),
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingFormAnalysisMetadata>(installed_forms));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    ASSERT_EQ(graph.installedFormMetadata().size(), 1u);
    ASSERT_EQ(graph.installedFormMetadata()[0].installed_blocks.size(), 1u);
    const auto& domains =
        graph.installedFormMetadata()[0].installed_blocks[0].domains;
    ASSERT_EQ(domains.size(), 3u);
    EXPECT_EQ(domains[0], analysis::DomainKind::Global);
    EXPECT_EQ(domains[1], analysis::DomainKind::CoupledBoundary);
    EXPECT_EQ(domains[2], analysis::DomainKind::AuxiliaryCoupling);
}

TEST(CouplingGraph, AcceptsGeometrySensitivityDependencyEvidence)
{
    auto declaration = twoParticipantDependencyDeclaration();
    declaration.expected_blocks[0].expect_matrix_block = false;

    CouplingFormAnalysisMetadata metadata;
    metadata.contribution_name = "geometry_sensitivity_coupling";
    metadata.origin = "CouplingGraphTest";
    metadata.system_name = "system";
    metadata.feature_gates.push_back(analysis::FormBridgeFeatureGate{
        analysis::FormBridgeFeature::InstalledDependencies,
        analysis::FormBridgeFeatureStatus::Available,
        "Synthetic installed dependency fixture"});
    metadata.field_uses.push_back(CouplingFormFieldProvenance{
        .residual_row = 2,
        .field = 1,
        .appears_as_geometry_sensitivity = true,
    });

    const std::vector<CouplingFormAnalysisMetadata> installed_forms{metadata};
    const auto validation = buildFinalizedGraph(
        twoParticipantGraphContext(),
        declaration,
        installed_forms);
    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);

    metadata.field_uses[0].appears_as_geometry_sensitivity = false;
    const std::vector<CouplingFormAnalysisMetadata> missing_forms{metadata};
    const auto missing = buildFinalizedGraph(
        twoParticipantGraphContext(),
        declaration,
        missing_forms);
    EXPECT_FALSE(missing.ok());
    EXPECT_NE(formatDiagnostics(missing).find(
                  "declared implicit coupling dependency is not reported"),
              std::string::npos);
}

TEST(CouplingGraph, DistinguishesExternalLaggedDependenciesFromImplicitEdges)
{
    auto implicit = twoParticipantDependencyDeclaration();
    implicit.expected_blocks.clear();
    const std::vector<CouplingFormAnalysisMetadata> no_installed_forms;
    const auto implicit_validation = buildFinalizedGraph(
        twoParticipantGraphContext(),
        implicit,
        no_installed_forms);
    EXPECT_FALSE(implicit_validation.ok());
    EXPECT_NE(formatDiagnostics(implicit_validation).find(
                  "declared implicit coupling dependency is not reported"),
              std::string::npos);

    auto external = implicit;
    external.dependencies[0].mode = CouplingDependencyMode::ExternalLagged;
    const auto external_validation = buildFinalizedGraph(
        twoParticipantGraphContext(),
        external,
        no_installed_forms);
    EXPECT_TRUE(external_validation.ok()) << formatDiagnostics(external_validation);
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

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST(CouplingGraph, RecordsInterfaceTransferProvenanceInResolvedPartitionedExchanges)
{
    svmp::search::InterfaceSearchRegistry registry;
    const auto sliding_map = graphSlidingInterfaceMap();
    const auto context = interfacePartitionedGraphContext(registry, sliding_map);
    const auto declaration = interfacePartitionedGraphDeclaration();
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
    ASSERT_EQ(snapshot.resolved_partitioned_exchanges.size(), 1u);
    const auto& exchange = snapshot.resolved_partitioned_exchanges[0].exchange;
    ASSERT_TRUE(exchange.transfer.interface_options.has_value());
    EXPECT_EQ(exchange.transfer.kind,
              CouplingTransferKind::InterfacePointwiseInterpolation);
    EXPECT_EQ(exchange.transfer.interface_options->field_kind,
              systems::InterfaceFieldKind::Vector);
    EXPECT_EQ(exchange.transfer.interface_options->frame_policy,
              systems::InterfaceFrameTransformPolicy::SourceToTargetVector);
    EXPECT_EQ(exchange.transfer.interface_options->component_count, 3u);
    EXPECT_EQ(exchange.transfer.interface_options->source_to_target_rotation[0][1],
              -1.0);
    EXPECT_EQ(exchange.transfer.interface_options->conservation_tolerance,
              1.0e-9);

    ASSERT_TRUE(exchange.transfer.interface_map.has_value());
    const auto& provenance = *exchange.transfer.interface_map;
    EXPECT_EQ(provenance.interface_map_name, "interface_map");
    EXPECT_EQ(provenance.interface_entry_name, "interface_entry");
    EXPECT_EQ(provenance.interface_search_registry_name, "interface_search");
    EXPECT_NE(provenance.interface_map_name, provenance.interface_entry_name);
    EXPECT_EQ(provenance.source_system_name, "left_system");
    EXPECT_EQ(provenance.target_system_name, "right_system");
    EXPECT_EQ(provenance.source_interface_marker, 21);
    EXPECT_EQ(provenance.target_interface_marker, 22);
    EXPECT_EQ(provenance.sliding_map_kind,
              systems::SlidingInterfaceMapKind::RotatingSliding);
    EXPECT_EQ(provenance.source_configuration, svmp::Configuration::Reference);
    EXPECT_EQ(provenance.target_configuration, svmp::Configuration::Current);
    EXPECT_EQ(provenance.source_logical_region.persistent_id,
              "left-interface-region");
    EXPECT_EQ(provenance.target_logical_region.persistent_id,
              "right-interface-region");
    EXPECT_EQ(provenance.source_revision_snapshot.revision_key(),
              graphInterfaceRevisionSnapshot(svmp::Configuration::Reference, 1)
                  .revision_key());
    EXPECT_EQ(provenance.target_revision_snapshot.revision_key(),
              graphInterfaceRevisionSnapshot(svmp::Configuration::Current, 2)
                  .revision_key());
    const auto interface_map = graphInterfaceMap("interface_map");
    EXPECT_EQ(provenance.source_search_revision_key,
              interface_map.source_revision.revision_key());
    EXPECT_EQ(provenance.target_search_revision_key,
              interface_map.target_revision.revision_key());
    EXPECT_EQ(provenance.map_revision_key, interface_map.revision_key());
    EXPECT_EQ(provenance.map_state, svmp::search::InterfaceMapState::Trial);
    EXPECT_EQ(provenance.operator_state, systems::InterfaceOperatorState::Trial);
    EXPECT_EQ(provenance.accepted_revision_key, 47u);
    EXPECT_EQ(provenance.trial_revision_key, 53u);
    EXPECT_EQ(provenance.time_level_epoch, 59u);

    ASSERT_TRUE(exchange.producer_region.has_value());
    ASSERT_TRUE(exchange.consumer_region.has_value());
    EXPECT_EQ(exchange.producer_region->coordinate_configuration,
              CouplingCoordinateConfiguration::Reference);
    EXPECT_EQ(exchange.consumer_region->coordinate_configuration,
              CouplingCoordinateConfiguration::Current);
    ASSERT_TRUE(exchange.producer_region->logical_region.has_value());
    ASSERT_TRUE(exchange.consumer_region->logical_region.has_value());
    EXPECT_EQ(exchange.producer_region->logical_region->persistent_id,
              "left-interface-region");
    EXPECT_EQ(exchange.consumer_region->logical_region->persistent_id,
              "right-interface-region");
}

TEST(CouplingGraph, RejectsMismatchedInterfaceMapRuntimeHandles)
{
    svmp::search::InterfaceSearchRegistry registry;
    const auto sliding_map = graphSlidingInterfaceMap();
    const auto context = interfacePartitionedGraphContext(registry, sliding_map);
    const auto declaration = interfacePartitionedGraphDeclaration();
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};

    const PartitionedCouplingPlanGenerator generator;
    const auto plan = generator.generate(
        context,
        std::span<const CouplingContractDeclaration>(declarations));

    const auto expect_rejected =
        [&](auto mutate_provenance, std::string_view expected_text) {
            auto mutated_plan = plan;
            ASSERT_FALSE(mutated_plan.exchanges.empty());
            ASSERT_TRUE(mutated_plan.exchanges[0].transfer.interface_map.has_value());
            mutate_provenance(*mutated_plan.exchanges[0].transfer.interface_map);

            const auto validation =
                buildFinalizedGraph(context, declaration, mutated_plan);
            EXPECT_FALSE(validation.ok());
            EXPECT_NE(formatDiagnostics(validation).find(expected_text),
                      std::string::npos)
                << "expected text: " << expected_text << "\n"
                << formatDiagnostics(validation);
        };

    expect_rejected(
        [](CouplingInterfaceMapProvenance& provenance) {
            provenance.map_revision_key += 1;
        },
        "runtime revision");
    expect_rejected(
        [](CouplingInterfaceMapProvenance& provenance) {
            provenance.source_system_name = "right_system";
        },
        "source system does not match");
    expect_rejected(
        [](CouplingInterfaceMapProvenance& provenance) {
            provenance.operator_state =
                systems::InterfaceOperatorState::AcceptedTimeStep;
        },
        "operator state does not match");
}
#endif

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
