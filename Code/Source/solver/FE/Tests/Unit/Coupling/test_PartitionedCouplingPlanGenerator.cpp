#include "Coupling/CouplingDeclaration.h"
#include "Coupling/PartitionedCouplingPlanGenerator.h"
#include "Assembly/AssemblyContext.h"
#include "Assembly/AssemblyKernel.h"
#include "Auxiliary/AuxiliaryBindings.h"
#include "Forms/BoundaryFunctional.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"

#include <gtest/gtest.h>

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

class ParameterDeclaringKernel final : public assembly::BilinearFormKernel {
public:
    explicit ParameterDeclaringKernel(params::ValueType type)
        : type_(type)
    {
    }

    [[nodiscard]] assembly::RequiredData getRequiredData() const override
    {
        return assembly::RequiredData::None;
    }

    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override
    {
        params::Spec spec;
        spec.key = "coefficient";
        spec.type = type_;
        spec.required = true;
        if (type_ == params::ValueType::Real) {
            spec.default_value = params::Value{Real(1.0)};
        } else if (type_ == params::ValueType::Int) {
            spec.default_value = params::Value{1};
        }
        return {spec};
    }

    void computeCell(const assembly::AssemblyContext&,
                     assembly::KernelOutput& output) override
    {
        output.reserve(0, 0, true, false);
    }

    [[nodiscard]] std::string name() const override { return "ParameterDeclaringKernel"; }

private:
    params::ValueType type_{params::ValueType::Real};
};

class ScalarOutputModel final : public systems::AuxiliaryStateModel {
public:
    [[nodiscard]] std::string modelName() const override
    {
        return "ScalarOutputModel";
    }

    [[nodiscard]] int dimension() const override { return 1; }

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

    [[nodiscard]] int outputCount() const override { return 1; }

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

const systems::FESystem* partitionedSystemToken(int index)
{
    return reinterpret_cast<const systems::FESystem*>(
        static_cast<std::uintptr_t>(index));
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
svmp::search::InterfaceRevisionSnapshot interfaceRevisionSnapshot(bool is_left)
{
    svmp::search::InterfaceRevisionSnapshot snapshot;
    snapshot.configuration = svmp::Configuration::Reference;
    snapshot.geometry_revision = is_left ? 101 : 201;
    snapshot.reference_geometry_revision = is_left ? 102 : 202;
    snapshot.current_geometry_revision = is_left ? 103 : 203;
    snapshot.topology_revision = is_left ? 104 : 204;
    snapshot.ownership_revision = is_left ? 105 : 205;
    snapshot.numbering_revision = is_left ? 106 : 206;
    snapshot.field_layout_revision = is_left ? 107 : 207;
    snapshot.label_revision = is_left ? 108 : 208;
    snapshot.active_configuration_epoch = is_left ? 109 : 209;
    return snapshot;
}
#endif

CouplingContextBuilder partitionedContextBuilder(int components)
{
    const auto left_system = partitionedSystemToken(1);
    const auto right_system = partitionedSystemToken(2);
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
        .components = components,
    });
    builder.addField({
        .participant_name = "right",
        .system_name = "right_system",
        .system = right_system,
        .field_name = "primary",
        .field_id = 2,
        .space = space,
        .components = components,
    });
    return builder;
}

CouplingContext partitionedContextWithComponents(int components)
{
    return partitionedContextBuilder(components).build();
}

CouplingContext partitionedContext()
{
    return partitionedContextWithComponents(1);
}

struct ParameterEndpointFixture {
    std::shared_ptr<spaces::H1Space> space;
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh;
    systems::FESystem left_system;
    FieldId left_field{INVALID_FIELD_ID};
    CouplingContext context;

    explicit ParameterEndpointFixture(
        params::ValueType parameter_type = params::ValueType::Real)
        : space(std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1))
        , mesh(std::make_shared<forms::test::SingleTetraMeshAccess>())
        , left_system(mesh)
    {
        left_field = left_system.addField(systems::FieldSpec{
            .name = "primary",
            .space = space,
            .components = 1,
        });
        left_system.addOperator("op");
        left_system.addCellKernel(
            "op", left_field, std::make_shared<ParameterDeclaringKernel>(parameter_type));
        systems::SetupInputs inputs;
        inputs.topology_override = timestepping::test::singleTetraTopology();
        left_system.setup({}, inputs);

        CouplingContextBuilder builder;
        builder.addParticipant({
            .participant_name = "left",
            .system_name = "left_system",
            .system = &left_system,
        });
        builder.addParticipant({
            .participant_name = "right",
            .system_name = "right_system",
            .system = partitionedSystemToken(2),
        });
        builder.addField({
            .participant_name = "left",
            .system_name = "left_system",
            .system = &left_system,
            .field_name = "primary",
            .field_id = left_field,
            .space = space,
            .components = 1,
        });
        builder.addField({
            .participant_name = "right",
            .system_name = "right_system",
            .system = partitionedSystemToken(2),
            .field_name = "primary",
            .field_id = 2,
            .space = space,
            .components = 1,
        });
        context = builder.build();
    }
};

struct AuxiliaryInputEndpointFixture {
    std::shared_ptr<spaces::H1Space> space;
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh;
    systems::FESystem left_system;
    FieldId left_field{INVALID_FIELD_ID};
    CouplingContext context;

    explicit AuxiliaryInputEndpointFixture(int input_size = 1,
                                           bool register_input = true)
        : space(std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1))
        , mesh(std::make_shared<forms::test::SingleTetraMeshAccess>())
        , left_system(mesh)
    {
        left_field = left_system.addField(systems::FieldSpec{
            .name = "primary",
            .space = space,
            .components = 1,
        });
        if (register_input) {
            systems::AuxiliaryInputSpec spec;
            spec.name = "driver_input";
            spec.size = input_size;
            spec.producer = systems::AuxiliaryInputProducer::DirectUserData;
            left_system.auxiliaryInputRegistry().registerInput(spec);
        }

        CouplingContextBuilder builder;
        builder.addParticipant({
            .participant_name = "left",
            .system_name = "left_system",
            .system = &left_system,
        });
        builder.addParticipant({
            .participant_name = "right",
            .system_name = "right_system",
            .system = partitionedSystemToken(2),
        });
        builder.addField({
            .participant_name = "left",
            .system_name = "left_system",
            .system = &left_system,
            .field_name = "primary",
            .field_id = left_field,
            .space = space,
            .components = 1,
        });
        builder.addField({
            .participant_name = "right",
            .system_name = "right_system",
            .system = partitionedSystemToken(2),
            .field_name = "primary",
            .field_id = 2,
            .space = space,
            .components = 1,
        });
        context = builder.build();
    }
};

struct AuxiliaryOutputEndpointFixture {
    std::shared_ptr<spaces::H1Space> space;
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh;
    systems::FESystem left_system;
    FieldId left_field{INVALID_FIELD_ID};
    CouplingContext context;

    explicit AuxiliaryOutputEndpointFixture(bool deploy_output = true)
        : space(std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1))
        , mesh(std::make_shared<forms::test::SingleTetraMeshAccess>())
        , left_system(mesh)
    {
        left_field = left_system.addField(systems::FieldSpec{
            .name = "primary",
            .space = space,
            .components = 1,
        });
        if (deploy_output) {
            auto model = std::make_shared<ScalarOutputModel>();
            left_system.deployAuxiliaryModel(
                systems::use(model)
                    .name("output_block")
                    .global()
                    .partitioned("ForwardEuler")
                    .initialize({0.0}));
            left_system.finalizeAuxiliaryLayout();
        }

        CouplingContextBuilder builder;
        builder.addParticipant({
            .participant_name = "left",
            .system_name = "left_system",
            .system = &left_system,
        });
        builder.addParticipant({
            .participant_name = "right",
            .system_name = "right_system",
            .system = partitionedSystemToken(2),
        });
        builder.addField({
            .participant_name = "left",
            .system_name = "left_system",
            .system = &left_system,
            .field_name = "primary",
            .field_id = left_field,
            .space = space,
            .components = 1,
        });
        builder.addField({
            .participant_name = "right",
            .system_name = "right_system",
            .system = partitionedSystemToken(2),
            .field_name = "primary",
            .field_id = 2,
            .space = space,
            .components = 1,
        });
        context = builder.build();
    }
};

struct RegionDataEndpointFixture {
    std::shared_ptr<spaces::H1Space> space;
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh;
    systems::FESystem left_system;
    FieldId left_field{INVALID_FIELD_ID};
    CouplingContext context;

    explicit RegionDataEndpointFixture(
        systems::FEQuantityShape shape = systems::FEQuantityShape::scalar(),
        bool explicit_evaluation = true,
        bool register_quantity = true)
        : space(std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1))
        , mesh(std::make_shared<forms::test::SingleTetraMeshAccess>())
        , left_system(mesh)
    {
        left_field = left_system.addField(systems::FieldSpec{
            .name = "primary",
            .space = space,
            .components = 1,
        });
        if (register_quantity) {
            systems::FEQuantityDefinition definition;
            definition.name = "surface_measure";
            definition.kind = systems::FEQuantityKind::RegionIntegral;
            definition.shape = shape;
            definition.capabilities.explicit_evaluation = explicit_evaluation;
            definition.referenced_fields = {left_field};
            left_system.feQuantityRegistry().registerDefinition(std::move(definition));
        }

        CouplingContextBuilder builder;
        builder.addParticipant({
            .participant_name = "left",
            .system_name = "left_system",
            .system = &left_system,
        });
        builder.addParticipant({
            .participant_name = "right",
            .system_name = "right_system",
            .system = partitionedSystemToken(2),
        });
        builder.addField({
            .participant_name = "left",
            .system_name = "left_system",
            .system = &left_system,
            .field_name = "primary",
            .field_id = left_field,
            .space = space,
            .components = 1,
        });
        builder.addField({
            .participant_name = "right",
            .system_name = "right_system",
            .system = partitionedSystemToken(2),
            .field_name = "primary",
            .field_id = 2,
            .space = space,
            .components = 1,
        });
        context = builder.build();
    }
};

struct BoundaryReductionEndpointFixture {
    std::shared_ptr<spaces::H1Space> space;
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh;
    systems::FESystem left_system;
    FieldId left_field{INVALID_FIELD_ID};
    CouplingContext context;

    BoundaryReductionEndpointFixture()
        : space(std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1))
        , mesh(std::make_shared<forms::test::SingleTetraMeshAccess>())
        , left_system(mesh)
    {
        left_field = left_system.addField(systems::FieldSpec{
            .name = "primary",
            .space = space,
            .components = 1,
        });

        forms::BoundaryFunctional functional;
        functional.name = "surface_measure";
        functional.integrand = forms::FormExpr::constant(1.0);
        functional.boundary_marker = 1;
        left_system.boundaryReductionService(left_field)
            .addBoundaryFunctional(std::move(functional));

        CouplingContextBuilder builder;
        builder.addParticipant({
            .participant_name = "left",
            .system_name = "left_system",
            .system = &left_system,
        });
        builder.addParticipant({
            .participant_name = "right",
            .system_name = "right_system",
            .system = partitionedSystemToken(2),
        });
        builder.addField({
            .participant_name = "left",
            .system_name = "left_system",
            .system = &left_system,
            .field_name = "primary",
            .field_id = left_field,
            .space = space,
            .components = 1,
        });
        builder.addField({
            .participant_name = "right",
            .system_name = "right_system",
            .system = partitionedSystemToken(2),
            .field_name = "primary",
            .field_id = 2,
            .space = space,
            .components = 1,
        });
        context = builder.build();
    }
};

CouplingRegionRef partitionedRegion(std::string participant,
                                    std::string system_name,
                                    const systems::FESystem* system,
                                    CouplingInterfaceSide side,
                                    int marker)
{
    const bool is_left = participant == "left";
    return CouplingRegionRef{
        .participant_name = std::move(participant),
        .system_name = std::move(system_name),
        .system = system,
        .region_name = "surface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = marker,
        .side = side,
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
        .logical_region = svmp::search::LogicalInterfaceRegionId{
            .persistent_id = is_left ? "left_interface" : "right_interface",
            .name = is_left ? "left_surface" : "right_surface",
        },
        .revision_snapshot = interfaceRevisionSnapshot(is_left),
#endif
    };
}

CouplingContext partitionedContextWithSharedRegion(int components = 1)
{
    auto builder = partitionedContextBuilder(components);
    const auto left = partitionedRegion(
        "left", "left_system", partitionedSystemToken(1),
        CouplingInterfaceSide::Minus, 10);
    const auto right = partitionedRegion(
        "right", "right_system", partitionedSystemToken(2),
        CouplingInterfaceSide::Plus, 11);
    builder.addRegion(left)
        .addRegion(right)
        .addSharedRegion(SharedRegionRef{
            .name = "interface",
            .required_region_kind = CouplingRegionKind::InterfaceFace,
            .participant_regions = {left, right},
        });
    return builder.build();
}

CouplingEndpointRef fieldEndpoint(std::string participant,
                                  CouplingTemporalSlotDescriptor temporal)
{
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::Field,
        .participant_name = std::move(participant),
        .endpoint_name = "primary",
        .temporal = temporal,
    };
}

CouplingEndpointRef fieldEndpoint(std::string participant)
{
    return fieldEndpoint(
        std::move(participant),
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current});
}

CouplingEndpointRef externalBufferEndpoint(std::string key)
{
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::ExternalBuffer,
        .endpoint_name = std::move(key),
        .temporal = CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::External,
        },
    };
}

CouplingEndpointRef participantExternalBufferEndpoint(std::string participant,
                                                      std::string key)
{
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::ExternalBuffer,
        .participant_name = std::move(participant),
        .endpoint_name = std::move(key),
        .temporal = CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::External,
        },
    };
}

CouplingEndpointRef parameterEndpoint(
    std::string participant,
    std::string key = "coefficient",
    CouplingTemporalSlotDescriptor temporal =
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current})
{
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::Parameter,
        .participant_name = std::move(participant),
        .endpoint_name = std::move(key),
        .temporal = temporal,
    };
}

CouplingEndpointRef auxiliaryInputEndpoint(
    std::string participant,
    std::string key = "driver_input",
    CouplingTemporalSlotDescriptor temporal =
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current})
{
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::AuxiliaryInput,
        .participant_name = std::move(participant),
        .endpoint_name = std::move(key),
        .temporal = temporal,
    };
}

CouplingEndpointRef auxiliaryOutputEndpoint(
    std::string participant,
    std::string key = "out_value",
    CouplingTemporalSlotDescriptor temporal =
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current})
{
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::AuxiliaryOutput,
        .participant_name = std::move(participant),
        .endpoint_name = std::move(key),
        .temporal = temporal,
    };
}

CouplingEndpointRef auxiliaryStateEndpoint(
    std::string participant,
    std::string key = "output_block",
    CouplingTemporalSlotDescriptor temporal =
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current})
{
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::AuxiliaryState,
        .participant_name = std::move(participant),
        .endpoint_name = std::move(key),
        .temporal = temporal,
    };
}

CouplingEndpointRef regionDataEndpoint(
    std::string participant,
    std::string key = "surface_measure",
    CouplingTemporalSlotDescriptor temporal =
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current})
{
    return CouplingEndpointRef{
        .kind = CouplingEndpointKind::RegionData,
        .participant_name = std::move(participant),
        .endpoint_name = std::move(key),
        .temporal = temporal,
    };
}

CouplingPortId port(std::string name)
{
    return CouplingPortId{
        .contract_instance_name = "generic_instance",
        .port_name = std::move(name),
    };
}

CouplingExchangeDeclaration identityExchange()
{
    CouplingExchangeDeclaration exchange;
    exchange.producer_port = port("left_out");
    exchange.consumer_port = port("right_in");
    exchange.value = CouplingValueDescriptor{
        .rank = CouplingValueRank::Scalar,
        .components = 1,
    };
    exchange.producer = fieldEndpoint("left");
    exchange.consumer = fieldEndpoint("right");
    exchange.transfer.kind = CouplingTransferKind::Identity;
    return exchange;
}

CouplingExternalBufferDescriptor externalBufferDescriptor(
    std::string name,
    CouplingValueDescriptor value,
    CouplingExternalBufferAccess access,
    std::vector<CouplingTemporalSlotDescriptor> supported_temporal_slots)
{
    return CouplingExternalBufferDescriptor{
        .buffer_name = std::move(name),
        .value = std::move(value),
        .access = access,
        .extents = {1},
        .strides = {1},
        .packing = "contiguous",
        .supported_temporal_slots = std::move(supported_temporal_slots),
        .layout_revision_key = 3,
        .data_revision_key = 5,
    };
}

CouplingDriverOwnedTransferDescriptor driverOwnedTransferDescriptor(
    std::string name,
    std::vector<CouplingValueRank> supported_ranks)
{
    return CouplingDriverOwnedTransferDescriptor{
        .transfer_name = std::move(name),
        .supported_ranks = std::move(supported_ranks),
        .supported_source_temporal_slots = {
            CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current}},
        .supported_target_temporal_slots = {
            CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current}},
        .registry_revision_key = 11,
    };
}

CouplingExchangeDeclaration interfaceExchange(CouplingValueDescriptor value,
                                              CouplingInterfaceFramePolicy frame_policy)
{
    auto exchange = identityExchange();
    exchange.value = std::move(value);
    exchange.shared_region_name = "interface";
    exchange.producer_region = CouplingRegionEndpointDeclaration{
        .participant_name = "left",
        .region_name = "surface",
    };
    exchange.consumer_region = CouplingRegionEndpointDeclaration{
        .participant_name = "right",
        .region_name = "surface",
    };
    exchange.transfer.kind = CouplingTransferKind::InterfacePointwiseInterpolation;
    exchange.transfer.interface_declaration = CouplingInterfaceTransferDeclaration{
        .frame_policy = frame_policy,
    };
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    exchange.transfer.interface_map = CouplingInterfaceMapProvenance{
        .interface_map_name = "interface_map",
        .interface_entry_name = "interface",
        .interface_search_registry_name = "default_search",
        .source_system_name = "left_system",
        .target_system_name = "right_system",
        .source_interface_marker = 10,
        .target_interface_marker = 11,
        .source_logical_region = svmp::search::LogicalInterfaceRegionId{
            .persistent_id = "left_interface",
            .name = "left_surface",
        },
        .target_logical_region = svmp::search::LogicalInterfaceRegionId{
            .persistent_id = "right_interface",
            .name = "right_surface",
        },
        .source_revision_snapshot = interfaceRevisionSnapshot(true),
        .target_revision_snapshot = interfaceRevisionSnapshot(false),
        .source_search_revision_key = 3,
        .target_search_revision_key = 5,
        .map_revision_key = 7,
        .map_state = svmp::search::InterfaceMapState::Committed,
        .operator_state = systems::InterfaceOperatorState::AcceptedTimeStep,
        .accepted_revision_key = 11,
        .trial_revision_key = 13,
        .time = 0.25,
        .time_level_epoch = 17,
    };
#endif
    return exchange;
}

CouplingContractDeclaration partitionedDeclaration()
{
    CouplingContractDeclaration declaration;
    declaration.contract_type = "generic";
    declaration.contract_name = "generic_instance";
    declaration.partitioned_exchange_declarations.push_back(identityExchange());
    declaration.group_hints.push_back(CouplingGroupHint{
        .name = "sync_group",
        .participant_names = {"left", "right"},
    });
    return declaration;
}

} // namespace

TEST(PartitionedCouplingPlanGenerator, GeneratesFieldIdentityExchange)
{
    const PartitionedCouplingPlanGenerator generator;
    const std::array<CouplingExchangeDeclaration, 1> exchanges{identityExchange()};

    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_EQ(plan.exchanges.size(), 1u);
    EXPECT_EQ(plan.exchanges[0].producer.field_id, 1);
    EXPECT_EQ(plan.exchanges[0].producer.system_name, "left_system");
    EXPECT_EQ(plan.exchanges[0].consumer.field_id, 2);
    EXPECT_EQ(plan.exchanges[0].transfer.kind, CouplingTransferKind::Identity);
    EXPECT_TRUE(plan.cycles.empty());
}

TEST(PartitionedCouplingPlanGenerator, GeneratesVectorFieldIdentityExchange)
{
    auto exchange = identityExchange();
    exchange.value = CouplingValueDescriptor{
        .rank = CouplingValueRank::Vector,
        .components = 2,
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithComponents(2),
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        partitionedContextWithComponents(2),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_EQ(plan.exchanges.size(), 1u);
    EXPECT_EQ(plan.exchanges[0].value.rank, CouplingValueRank::Vector);
    EXPECT_EQ(plan.exchanges[0].value.components, 2);
}

TEST(PartitionedCouplingPlanGenerator, RejectsFieldEndpointComponentMismatch)
{
    auto exchange = identityExchange();
    exchange.value = CouplingValueDescriptor{
        .rank = CouplingValueRank::Vector,
        .components = 2,
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("component count does not match"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsGeneralTensorWithoutDriverOwnedTransfer)
{
    auto exchange = identityExchange();
    exchange.value = CouplingValueDescriptor{
        .rank = CouplingValueRank::GeneralTensor,
        .components = 4,
        .tensor_extents = {2, 2},
        .tensor_packing = "row_major",
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithComponents(4),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("general tensor partitioned values"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, AcceptsScalarInterfaceTransferMetadata)
{
    const auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::None);
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

TEST(PartitionedCouplingPlanGenerator, AcceptsVectorInterfaceFrameTransform)
{
    const auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Vector,
            .components = 3,
        },
        CouplingInterfaceFramePolicy::SourceToTargetVector);
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(3),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_TRUE(validation.ok()) << formatDiagnostics(validation);
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST(PartitionedCouplingPlanGenerator, GeneratesResolvedInterfaceTransferOptions)
{
    auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Vector,
            .components = 3,
        },
        CouplingInterfaceFramePolicy::SourceToTargetVector);
    exchange.transfer.interface_declaration->source_to_target_rotation[0][0] = 0.0;
    exchange.transfer.interface_declaration->source_to_target_rotation[0][1] = -1.0;
    exchange.transfer.interface_declaration->source_to_target_rotation[1][0] = 1.0;
    exchange.transfer.interface_declaration->source_to_target_rotation[1][1] = 0.0;
    exchange.transfer.interface_declaration->conservation_tolerance = 1.0e-8;
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto context = partitionedContextWithSharedRegion(3);
    const auto validation = generator.validate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_TRUE(plan.exchanges[0].transfer.interface_options.has_value());
    EXPECT_EQ(plan.exchanges[0].transfer.source_embedding_policy,
              CouplingFrameSourceEmbeddingPolicy::None);
    EXPECT_EQ(plan.exchanges[0].transfer.target_restriction_policy,
              CouplingFrameTargetRestrictionPolicy::None);
    const auto& options = *plan.exchanges[0].transfer.interface_options;
    EXPECT_EQ(options.field_kind, systems::InterfaceFieldKind::Vector);
    EXPECT_EQ(options.frame_policy,
              systems::InterfaceFrameTransformPolicy::SourceToTargetVector);
    EXPECT_EQ(options.component_count, 3u);
    EXPECT_EQ(options.source_to_target_rotation[0][1], -1.0);
    EXPECT_EQ(options.source_to_target_rotation[1][0], 1.0);
    EXPECT_EQ(options.conservation_tolerance, 1.0e-8);
    ASSERT_TRUE(plan.exchanges[0].transfer.interface_map.has_value());
    const auto& provenance = *plan.exchanges[0].transfer.interface_map;
    EXPECT_EQ(provenance.interface_map_name, "interface_map");
    EXPECT_EQ(provenance.interface_entry_name, "interface");
    EXPECT_EQ(provenance.interface_search_registry_name, "default_search");
    EXPECT_EQ(provenance.source_system_name, "left_system");
    EXPECT_EQ(provenance.target_system_name, "right_system");
    EXPECT_EQ(provenance.source_interface_marker, 10);
    EXPECT_EQ(provenance.target_interface_marker, 11);
    EXPECT_EQ(provenance.map_state, svmp::search::InterfaceMapState::Committed);
    EXPECT_EQ(provenance.operator_state,
              systems::InterfaceOperatorState::AcceptedTimeStep);
    EXPECT_EQ(provenance.map_revision_key, 7u);
}
#endif

TEST(PartitionedCouplingPlanGenerator, RejectsInterfaceTransferWithoutRegionEndpoints)
{
    auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::None);
    exchange.producer_region.reset();
    exchange.consumer_region.reset();
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires producer region endpoint"),
              std::string::npos);
    EXPECT_NE(formatDiagnostics(validation).find("requires consumer region endpoint"),
              std::string::npos);
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST(PartitionedCouplingPlanGenerator, RejectsInterfaceTransferWithoutMapProvenance)
{
    auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::None);
    exchange.transfer.interface_map.reset();
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires interface map provenance"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsInterfaceMapProvenanceRegionMismatch)
{
    auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::None);
    exchange.transfer.interface_map->source_interface_marker = 99;
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("source marker does not match"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsInterfaceMapProvenanceConfigurationMismatch)
{
    auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::None);
    exchange.transfer.interface_map->source_configuration = svmp::Configuration::Current;
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("source configuration does not match"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsInterfaceMapProvenanceLogicalRegionMismatch)
{
    auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::None);
    exchange.transfer.interface_map->source_logical_region.persistent_id =
        "other_interface";
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("source logical region does not match"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsInterfaceMapProvenanceMissingRevisionSnapshot)
{
    auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::None);
    exchange.transfer.interface_map->source_revision_snapshot =
        svmp::search::InterfaceRevisionSnapshot{};
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires a source revision snapshot"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsInterfaceMapProvenanceRevisionSnapshotMismatch)
{
    auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::None);
    exchange.transfer.interface_map->source_revision_snapshot.geometry_revision += 1;
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("source revision snapshot does not match"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsInterfaceMapProvenanceMissingRevisionKeys)
{
    auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::None);
    exchange.transfer.interface_map->source_search_revision_key = 0;
    exchange.transfer.interface_map->target_search_revision_key = 0;
    exchange.transfer.interface_map->map_revision_key = 0;
    exchange.transfer.interface_map->accepted_revision_key = 0;
    exchange.transfer.interface_map->trial_revision_key = 0;
    exchange.transfer.interface_map->time_level_epoch = 0;
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    const auto diagnostics = formatDiagnostics(validation);
    EXPECT_NE(diagnostics.find("source search revision key"), std::string::npos);
    EXPECT_NE(diagnostics.find("target search revision key"), std::string::npos);
    EXPECT_NE(diagnostics.find("interface map revision key"), std::string::npos);
    EXPECT_NE(diagnostics.find("accepted revision key"), std::string::npos);
    EXPECT_NE(diagnostics.find("trial revision key"), std::string::npos);
    EXPECT_NE(diagnostics.find("time-level epoch"), std::string::npos);
}
#endif

TEST(PartitionedCouplingPlanGenerator, RejectsInterfaceTransferNonInterfaceRegions)
{
    auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::None);
    exchange.shared_region_name.reset();

    auto builder = partitionedContextBuilder(1);
    builder.addRegion(CouplingRegionRef{
        .participant_name = "left",
        .system_name = "left_system",
        .system = partitionedSystemToken(1),
        .region_name = "surface",
        .kind = CouplingRegionKind::Boundary,
        .marker = 10,
    });
    builder.addRegion(CouplingRegionRef{
        .participant_name = "right",
        .system_name = "right_system",
        .system = partitionedSystemToken(2),
        .region_name = "surface",
        .kind = CouplingRegionKind::Boundary,
        .marker = 11,
    });
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        builder.build(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("must resolve to an interface face"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsUserDefinedInterfaceTransferRegions)
{
    auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::None);
    exchange.shared_region_name.reset();

    auto builder = partitionedContextBuilder(1);
    builder.addRegion(CouplingRegionRef{
        .participant_name = "left",
        .system_name = "left_system",
        .system = partitionedSystemToken(1),
        .region_name = "surface",
        .kind = CouplingRegionKind::UserDefined,
    });
    builder.addRegion(CouplingRegionRef{
        .participant_name = "right",
        .system_name = "right_system",
        .system = partitionedSystemToken(2),
        .region_name = "surface",
        .kind = CouplingRegionKind::UserDefined,
    });
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        builder.build(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("must resolve to an interface face"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsInterfaceFramePayloadMismatch)
{
    const auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::SourceToTargetVector);
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("vector frame transforms require vector"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsSymmetricTensorInterfaceTransfer)
{
    const auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::SymmetricTensor,
            .components = 6,
        },
        CouplingInterfaceFramePolicy::None);
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(6),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("symmetric tensor interface transfers"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsTrue2DVectorInterfaceTransformWithoutAdapter)
{
    auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Vector,
            .components = 2,
        },
        CouplingInterfaceFramePolicy::SourceToTargetVector);
    exchange.transfer.interface_declaration->source_embedding_policy =
        CouplingFrameSourceEmbeddingPolicy::Embed2DInXY;
    exchange.transfer.interface_declaration->target_restriction_policy =
        CouplingFrameTargetRestrictionPolicy::RestrictToXY;
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(2),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("true 2D vector interface transforms"),
              std::string::npos);
}

#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
TEST(PartitionedCouplingPlanGenerator, RejectsInterfaceTransferWhenMeshSupportDisabled)
{
    const auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Scalar,
            .components = 1,
        },
        CouplingInterfaceFramePolicy::None);
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find(
                  "interface partitioned transfers require mesh interface support"),
              std::string::npos);
}
#endif

TEST(PartitionedCouplingPlanGenerator, RejectsVectorFramePassThroughWithoutLayout)
{
    const auto exchange = interfaceExchange(
        CouplingValueDescriptor{
            .rank = CouplingValueRank::Vector,
            .components = 4,
        },
        CouplingInterfaceFramePolicy::SourceToTargetVector);
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(4),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("pass-through components require component layout"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, ResolvesFieldHistoryTemporalSlot)
{
    auto exchange = identityExchange();
    exchange.producer = fieldEndpoint(
        "left",
        CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::History,
            .history_index = 2,
        });
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_EQ(plan.exchanges[0].producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::SystemStateHistory);
    ASSERT_TRUE(plan.exchanges[0].producer.temporal.storage_index.has_value());
    EXPECT_EQ(*plan.exchanges[0].producer.temporal.storage_index, 1);
    ASSERT_TRUE(plan.exchanges[0].producer.temporal.request.history_index.has_value());
    EXPECT_EQ(*plan.exchanges[0].producer.temporal.request.history_index, 2);
}

TEST(PartitionedCouplingPlanGenerator, RejectsUnsupportedFieldTemporalSlots)
{
    const std::vector<CouplingTemporalSlotDescriptor> unsupported_temporal_slots{
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Accepted},
        CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::Stage,
            .stage_index = 0,
        },
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::External},
    };

    const PartitionedCouplingPlanGenerator generator;
    for (const auto& temporal : unsupported_temporal_slots) {
        SCOPED_TRACE(toString(temporal.slot));
        auto exchange = identityExchange();
        exchange.producer = fieldEndpoint("left", temporal);
        const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

        const auto validation = generator.validate(
            partitionedContext(),
            std::span<const CouplingExchangeDeclaration>(exchanges));

        EXPECT_FALSE(validation.ok());
        EXPECT_NE(formatDiagnostics(validation).find("field endpoint temporal slot"),
                  std::string::npos);
    }
}

TEST(PartitionedCouplingPlanGenerator, ResolvesFieldPredictedTemporalSlot)
{
    auto exchange = identityExchange();
    exchange.producer = fieldEndpoint(
        "left",
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Predicted});
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_EQ(plan.exchanges[0].producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::SystemStatePredicted);
    EXPECT_NE(plan.exchanges[0].producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::SystemStateCurrent);
    EXPECT_NE(plan.exchanges[0].producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::SystemStateAccepted);
}

TEST(PartitionedCouplingPlanGenerator, GeneratesAuxiliaryStateEndpointFromBlock)
{
    AuxiliaryOutputEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = auxiliaryStateEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_EQ(plan.exchanges.size(), 1u);
    const auto& producer = plan.exchanges[0].producer;
    EXPECT_EQ(producer.resolved_kind, CouplingEndpointKind::AuxiliaryState);
    EXPECT_EQ(producer.system_name, "left_system");
    EXPECT_EQ(producer.system, &fixture.left_system);
    EXPECT_EQ(producer.registry_provider, "AuxiliaryStateManager");
    EXPECT_EQ(producer.temporal.provider_name, "AuxiliaryStateManager");
    EXPECT_EQ(producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::AuxiliaryCurrent);
    EXPECT_EQ(producer.auxiliary_kind,
              CouplingAuxiliaryEndpointResolutionKind::BlockState);
    ASSERT_TRUE(producer.auxiliary_block_index.has_value());
    EXPECT_EQ(*producer.auxiliary_block_index, 0u);
    EXPECT_EQ(producer.auxiliary_key, "output_block");
}

TEST(PartitionedCouplingPlanGenerator, ResolvesAcceptedAuxiliaryStateEndpoint)
{
    AuxiliaryOutputEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = auxiliaryStateEndpoint(
        "left",
        "output_block",
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Accepted});
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_EQ(plan.exchanges[0].producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::AuxiliaryCommitted);
}

TEST(PartitionedCouplingPlanGenerator, ResolvesPredictedAuxiliaryStateEndpoint)
{
    AuxiliaryOutputEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = auxiliaryStateEndpoint(
        "left",
        "output_block",
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Predicted});
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_EQ(plan.exchanges[0].producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::AuxiliaryPredicted);
    EXPECT_NE(plan.exchanges[0].producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::AuxiliaryCurrent);
    EXPECT_NE(plan.exchanges[0].producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::AuxiliaryCommitted);
}

TEST(PartitionedCouplingPlanGenerator, RejectsAuxiliaryStateWithoutBlock)
{
    AuxiliaryOutputEndpointFixture fixture(false);
    auto exchange = identityExchange();
    exchange.producer = auxiliaryStateEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires an AuxiliaryState block"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsAuxiliaryStateUnsupportedTemporalSlot)
{
    AuxiliaryOutputEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = auxiliaryStateEndpoint(
        "left",
        "output_block",
        CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::Stage,
            .stage_index = 0,
        });
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("auxiliary state endpoint temporal slot"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, GeneratesAuxiliaryOutputEndpointFromRegistry)
{
    AuxiliaryOutputEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = auxiliaryOutputEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_EQ(plan.exchanges.size(), 1u);
    const auto& producer = plan.exchanges[0].producer;
    EXPECT_EQ(producer.resolved_kind, CouplingEndpointKind::AuxiliaryOutput);
    EXPECT_EQ(producer.system_name, "left_system");
    EXPECT_EQ(producer.system, &fixture.left_system);
    EXPECT_EQ(producer.registry_provider, "AuxiliaryOutputRegistry");
    EXPECT_EQ(producer.temporal.provider_name, "AuxiliaryOutputRegistry");
    EXPECT_EQ(producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::AuxiliaryCurrent);
    EXPECT_EQ(producer.auxiliary_kind,
              CouplingAuxiliaryEndpointResolutionKind::Output);
    ASSERT_TRUE(producer.auxiliary_output_id.has_value());
    EXPECT_EQ(*producer.auxiliary_output_id, 0u);
    ASSERT_TRUE(producer.auxiliary_output_flat_slot.has_value());
    EXPECT_EQ(*producer.auxiliary_output_flat_slot, 0u);
    EXPECT_EQ(producer.auxiliary_key, "out_value");
}

TEST(PartitionedCouplingPlanGenerator, RejectsAuxiliaryOutputWithoutDeployedOutput)
{
    AuxiliaryOutputEndpointFixture fixture(false);
    auto exchange = identityExchange();
    exchange.producer = auxiliaryOutputEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires a deployed auxiliary output"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsAuxiliaryOutputUnsupportedTemporalSlot)
{
    AuxiliaryOutputEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = auxiliaryOutputEndpoint(
        "left",
        "out_value",
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Accepted});
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("auxiliary output endpoint temporal slot"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, GeneratesAuxiliaryInputEndpointFromRegistry)
{
    AuxiliaryInputEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = auxiliaryInputEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_EQ(plan.exchanges.size(), 1u);
    const auto& producer = plan.exchanges[0].producer;
    EXPECT_EQ(producer.resolved_kind, CouplingEndpointKind::AuxiliaryInput);
    EXPECT_EQ(producer.system_name, "left_system");
    EXPECT_EQ(producer.system, &fixture.left_system);
    EXPECT_EQ(producer.registry_provider, "AuxiliaryInputRegistry");
    EXPECT_EQ(producer.temporal.provider_name, "AuxiliaryInputRegistry");
    EXPECT_EQ(producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::AuxiliaryCurrent);
    EXPECT_EQ(producer.auxiliary_kind,
              CouplingAuxiliaryEndpointResolutionKind::InputSlot);
    ASSERT_TRUE(producer.auxiliary_input_slot.has_value());
    EXPECT_EQ(*producer.auxiliary_input_slot, 0u);
    EXPECT_EQ(producer.auxiliary_key, "driver_input");
}

TEST(PartitionedCouplingPlanGenerator, RejectsAuxiliaryInputWithoutRegistryEntry)
{
    AuxiliaryInputEndpointFixture fixture(1, false);
    auto exchange = identityExchange();
    exchange.producer = auxiliaryInputEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires an AuxiliaryInputRegistry entry"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsAuxiliaryInputComponentMismatch)
{
    AuxiliaryInputEndpointFixture fixture(2);
    auto exchange = identityExchange();
    exchange.producer = auxiliaryInputEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("component count does not match"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsAuxiliaryInputUnsupportedTemporalSlot)
{
    AuxiliaryInputEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = auxiliaryInputEndpoint(
        "left",
        "driver_input",
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Accepted});
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("auxiliary input endpoint temporal slot"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, GeneratesRegionDataEndpointFromRegistry)
{
    RegionDataEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = regionDataEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_EQ(plan.exchanges.size(), 1u);
    const auto& producer = plan.exchanges[0].producer;
    EXPECT_EQ(producer.resolved_kind, CouplingEndpointKind::RegionData);
    EXPECT_EQ(producer.system_name, "left_system");
    EXPECT_EQ(producer.system, &fixture.left_system);
    EXPECT_EQ(producer.registry_provider, "FEQuantityRegistry");
    EXPECT_EQ(producer.temporal.provider_name, "FEQuantityRegistry");
    EXPECT_EQ(producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::ProviderDefined);
    EXPECT_EQ(producer.region_data_provider_kind,
              CouplingRegionDataProviderKind::FEQuantity);
    EXPECT_EQ(producer.region_data_provider_name, "surface_measure");
    ASSERT_TRUE(producer.fe_quantity_id.has_value());
    EXPECT_EQ(*producer.fe_quantity_id, 0u);
}

TEST(PartitionedCouplingPlanGenerator, RejectsRegionDataWithoutRegistryEntry)
{
    RegionDataEndpointFixture fixture(systems::FEQuantityShape::scalar(),
                                      true,
                                      false);
    auto exchange = identityExchange();
    exchange.producer = regionDataEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires an FE quantity"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, GeneratesRegionDataFromBoundaryReductionService)
{
    BoundaryReductionEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = regionDataEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_EQ(plan.exchanges.size(), 1u);
    const auto& producer = plan.exchanges[0].producer;
    EXPECT_EQ(producer.registry_provider, "BoundaryReductionService");
    EXPECT_EQ(producer.temporal.provider_name, "BoundaryReductionService");
    EXPECT_EQ(producer.region_data_provider_kind,
              CouplingRegionDataProviderKind::BoundaryReductionFunctional);
    EXPECT_EQ(producer.boundary_functional_name, "surface_measure");
    EXPECT_EQ(producer.boundary_reduction_primary_field, fixture.left_field);
    EXPECT_FALSE(producer.fe_quantity_id.has_value());
}

TEST(PartitionedCouplingPlanGenerator, RejectsRegionDataShapeMismatch)
{
    RegionDataEndpointFixture fixture(systems::FEQuantityShape::vector(2));
    auto exchange = identityExchange();
    exchange.producer = regionDataEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("value shape does not match"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsRegionDataWithoutExplicitEvaluation)
{
    RegionDataEndpointFixture fixture(systems::FEQuantityShape::scalar(), false);
    auto exchange = identityExchange();
    exchange.producer = regionDataEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires explicit evaluation support"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsRegionDataUnsupportedTemporalSlot)
{
    RegionDataEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = regionDataEndpoint(
        "left",
        "surface_measure",
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Accepted});
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("region data endpoint temporal slot"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, GeneratesParameterEndpointFromRegistry)
{
    ParameterEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = parameterEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_EQ(plan.exchanges.size(), 1u);
    const auto& producer = plan.exchanges[0].producer;
    EXPECT_EQ(producer.resolved_kind, CouplingEndpointKind::Parameter);
    EXPECT_EQ(producer.system_name, "left_system");
    EXPECT_EQ(producer.system, &fixture.left_system);
    EXPECT_EQ(producer.registry_provider, "ParameterRegistry");
    EXPECT_EQ(producer.temporal.provider_name, "ParameterRegistry");
    EXPECT_EQ(producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::ProviderDefined);
    ASSERT_TRUE(producer.parameter_slot.has_value());
    EXPECT_EQ(*producer.parameter_slot, 0u);
    EXPECT_EQ(producer.parameter_value_type, params::ValueType::Real);
}

TEST(PartitionedCouplingPlanGenerator, RejectsParameterEndpointWithoutRegistryEntry)
{
    ParameterEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = parameterEndpoint("left", "missing");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires a ParameterRegistry entry"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsParameterEndpointNonRealType)
{
    ParameterEndpointFixture fixture(params::ValueType::Int);
    auto exchange = identityExchange();
    exchange.producer = parameterEndpoint("left");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires a Real ParameterRegistry entry"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsParameterEndpointUnsupportedTemporalSlot)
{
    ParameterEndpointFixture fixture;
    auto exchange = identityExchange();
    exchange.producer = parameterEndpoint(
        "left",
        "coefficient",
        CouplingTemporalSlotDescriptor{
            .slot = CouplingTemporalSlot::History,
            .history_index = 1,
        });
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        fixture.context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("parameter endpoint temporal slot"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsExternalBufferWithoutDescriptor)
{
    auto exchange = identityExchange();
    exchange.producer = externalBufferEndpoint("driver_value");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires a registered descriptor"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, GeneratesExternalBufferEndpointWithDescriptor)
{
    auto exchange = identityExchange();
    exchange.producer = externalBufferEndpoint("driver_value");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto builder = partitionedContextBuilder(1);
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = externalBufferDescriptor(
            "driver_value",
            exchange.value,
            CouplingExternalBufferAccess::ReadOnly,
            {CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::External}}),
    });
    const auto context = builder.build();

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_TRUE(plan.exchanges[0].producer.external_buffer.has_value());
    EXPECT_EQ(plan.exchanges[0].producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::ExternalBuffer);
    EXPECT_EQ(plan.exchanges[0].producer.layout_revision_key, 3u);
    EXPECT_EQ(plan.exchanges[0].producer.registry_revision_key, 5u);
}

TEST(PartitionedCouplingPlanGenerator, ResolvesPredictedExternalBufferEndpoint)
{
    auto exchange = identityExchange();
    exchange.producer = externalBufferEndpoint("driver_value");
    exchange.producer->temporal =
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Predicted};
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto builder = partitionedContextBuilder(1);
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = externalBufferDescriptor(
            "driver_value",
            exchange.value,
            CouplingExternalBufferAccess::ReadOnly,
            {CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Predicted}}),
    });
    const auto context = builder.build();

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_EQ(plan.exchanges[0].producer.temporal.backing,
              CouplingResolvedTemporalBackingKind::ExternalBuffer);
    EXPECT_EQ(plan.exchanges[0].producer.temporal.request.slot,
              CouplingTemporalSlot::Predicted);
    EXPECT_EQ(plan.exchanges[0].producer.temporal.provided.slot,
              CouplingTemporalSlot::Predicted);
}

TEST(PartitionedCouplingPlanGenerator, ResolvesParticipantScopedExternalBufferEndpoint)
{
    auto exchange = identityExchange();
    exchange.producer = participantExternalBufferEndpoint("left", "driver_value");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto global_descriptor = externalBufferDescriptor(
        "driver_value",
        exchange.value,
        CouplingExternalBufferAccess::ReadOnly,
        {CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::External}});
    global_descriptor.data_revision_key = 5;

    auto scoped_descriptor = externalBufferDescriptor(
        "driver_value",
        exchange.value,
        CouplingExternalBufferAccess::ReadOnly,
        {CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::External}});
    scoped_descriptor.data_revision_key = 9;

    auto builder = partitionedContextBuilder(1);
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = global_descriptor,
    });
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .participant_name = "left",
        .descriptor = scoped_descriptor,
    });
    const auto context = builder.build();

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_TRUE(plan.exchanges[0].producer.resolved_participant_name.has_value());
    EXPECT_EQ(*plan.exchanges[0].producer.resolved_participant_name, "left");
    EXPECT_EQ(plan.exchanges[0].producer.system_name, "left_system");
    EXPECT_EQ(plan.exchanges[0].producer.system, partitionedSystemToken(1));
    ASSERT_TRUE(plan.exchanges[0].producer.external_buffer.has_value());
    EXPECT_EQ(plan.exchanges[0].producer.external_buffer->data_revision_key, 9u);
}

TEST(PartitionedCouplingPlanGenerator, RejectsExternalBufferUnsupportedTemporalSlot)
{
    auto exchange = identityExchange();
    exchange.producer = externalBufferEndpoint("driver_value");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto builder = partitionedContextBuilder(1);
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = externalBufferDescriptor(
            "driver_value",
            exchange.value,
            CouplingExternalBufferAccess::ReadOnly,
            {CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::Current}}),
    });

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        builder.build(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("does not support the requested temporal slot"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsExternalBufferProducerWithoutReadAccess)
{
    auto exchange = identityExchange();
    exchange.producer = externalBufferEndpoint("driver_value");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto builder = partitionedContextBuilder(1);
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = externalBufferDescriptor(
            "driver_value",
            exchange.value,
            CouplingExternalBufferAccess::WriteOnly,
            {CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::External}}),
    });

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        builder.build(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires read access"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, GeneratesDriverOwnedTransferDescriptor)
{
    auto exchange = identityExchange();
    exchange.transfer.kind = CouplingTransferKind::DriverOwned;
    exchange.transfer.driver_owned_name = "copy";
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto builder = partitionedContextBuilder(1);
    builder.addDriverOwnedTransfer(
        driverOwnedTransferDescriptor("copy", {CouplingValueRank::Scalar}));
    const auto context = builder.build();

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_TRUE(plan.exchanges[0].transfer.driver_owned_descriptor.has_value());
    EXPECT_EQ(plan.exchanges[0].transfer.driver_owned_descriptor->registry_revision_key,
              11u);
}

TEST(PartitionedCouplingPlanGenerator, RejectsDriverOwnedTransferWithoutDescriptor)
{
    auto exchange = identityExchange();
    exchange.transfer.kind = CouplingTransferKind::DriverOwned;
    exchange.transfer.driver_owned_name = "copy";
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("requires a registered descriptor"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsDriverOwnedTransferUnsupportedRank)
{
    auto exchange = identityExchange();
    exchange.value = CouplingValueDescriptor{
        .rank = CouplingValueRank::Vector,
        .components = 2,
    };
    exchange.transfer.kind = CouplingTransferKind::DriverOwned;
    exchange.transfer.driver_owned_name = "copy";
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto builder = partitionedContextBuilder(2);
    builder.addDriverOwnedTransfer(
        driverOwnedTransferDescriptor("copy", {CouplingValueRank::Scalar}));

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        builder.build(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("does not support the exchange value rank"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsDriverOwnedGeneralTensorFieldEndpoints)
{
    auto exchange = identityExchange();
    exchange.value = CouplingValueDescriptor{
        .rank = CouplingValueRank::GeneralTensor,
        .components = 4,
        .tensor_extents = {2, 2},
        .tensor_packing = "row_major",
    };
    exchange.transfer.kind = CouplingTransferKind::DriverOwned;
    exchange.transfer.driver_owned_name = "copy";
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto builder = partitionedContextBuilder(4);
    builder.addDriverOwnedTransfer(
        driverOwnedTransferDescriptor("copy", {CouplingValueRank::GeneralTensor}));

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        builder.build(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("general tensor producer requires"),
              std::string::npos);
    EXPECT_NE(formatDiagnostics(validation).find("general tensor consumer requires"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, GeneratesDriverOwnedGeneralTensorExternalBuffers)
{
    auto exchange = identityExchange();
    exchange.value = CouplingValueDescriptor{
        .rank = CouplingValueRank::GeneralTensor,
        .components = 4,
        .tensor_extents = {2, 2},
        .tensor_packing = "row_major",
    };
    exchange.producer = externalBufferEndpoint("tensor_source");
    exchange.consumer = externalBufferEndpoint("tensor_target");
    exchange.transfer.kind = CouplingTransferKind::DriverOwned;
    exchange.transfer.driver_owned_name = "tensor_copy";
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    auto producer_descriptor = externalBufferDescriptor(
        "tensor_source",
        exchange.value,
        CouplingExternalBufferAccess::ReadOnly,
        {CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::External}});
    producer_descriptor.extents = {2, 2};
    producer_descriptor.strides = {2, 1};
    producer_descriptor.packing = "row_major";

    auto consumer_descriptor = externalBufferDescriptor(
        "tensor_target",
        exchange.value,
        CouplingExternalBufferAccess::WriteOnly,
        {CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::External}});
    consumer_descriptor.extents = {2, 2};
    consumer_descriptor.strides = {2, 1};
    consumer_descriptor.packing = "row_major";

    CouplingDriverOwnedTransferDescriptor transfer_descriptor;
    transfer_descriptor.transfer_name = "tensor_copy";
    transfer_descriptor.supported_ranks = {CouplingValueRank::GeneralTensor};
    transfer_descriptor.supported_source_temporal_slots = {
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::External}};
    transfer_descriptor.supported_target_temporal_slots = {
        CouplingTemporalSlotDescriptor{.slot = CouplingTemporalSlot::External}};
    transfer_descriptor.registry_revision_key = 17;

    auto builder = partitionedContextBuilder(4);
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = producer_descriptor,
    });
    builder.addExternalBuffer(CouplingExternalBufferRegistration{
        .descriptor = consumer_descriptor,
    });
    builder.addDriverOwnedTransfer(transfer_descriptor);
    const auto context = builder.build();

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_EQ(plan.exchanges.size(), 1u);
    ASSERT_TRUE(plan.exchanges[0].producer.external_buffer.has_value());
    ASSERT_TRUE(plan.exchanges[0].consumer.external_buffer.has_value());
    EXPECT_EQ(plan.exchanges[0].producer.external_buffer->packing, "row_major");
    EXPECT_EQ(plan.exchanges[0].consumer.external_buffer->value.tensor_extents,
              std::vector<int>({2, 2}));
    ASSERT_TRUE(plan.exchanges[0].transfer.driver_owned_descriptor.has_value());
    EXPECT_EQ(plan.exchanges[0].transfer.driver_owned_descriptor->registry_revision_key,
              17u);
}

TEST(PartitionedCouplingPlanGenerator, InheritsExchangeSharedRegionForEndpointRegions)
{
    auto exchange = identityExchange();
    exchange.shared_region_name = "interface";
    exchange.producer_region = CouplingRegionEndpointDeclaration{
        .participant_name = "left",
        .region_name = "surface",
    };
    exchange.consumer_region = CouplingRegionEndpointDeclaration{
        .participant_name = "right",
        .region_name = "surface",
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto context = partitionedContextWithSharedRegion();
    const auto validation = generator.validate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        context,
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_TRUE(plan.exchanges[0].producer_region.has_value());
    ASSERT_TRUE(plan.exchanges[0].consumer_region.has_value());
    EXPECT_EQ(plan.exchanges[0].producer_region->marker, 10);
    EXPECT_EQ(plan.exchanges[0].producer_region->side, CouplingInterfaceSide::Minus);
    EXPECT_EQ(plan.exchanges[0].consumer_region->marker, 11);
    EXPECT_EQ(plan.exchanges[0].consumer_region->side, CouplingInterfaceSide::Plus);
}

TEST(PartitionedCouplingPlanGenerator, RejectsConflictingEndpointSharedRegion)
{
    auto exchange = identityExchange();
    exchange.shared_region_name = "interface";
    exchange.producer_region = CouplingRegionEndpointDeclaration{
        .participant_name = "left",
        .region_name = "surface",
        .shared_region_name = "other_interface",
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContextWithSharedRegion(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("conflicts with the exchange shared region"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsMissingParticipantRegionEndpoint)
{
    auto exchange = identityExchange();
    exchange.producer_region = CouplingRegionEndpointDeclaration{
        .participant_name = "left",
        .region_name = "missing",
    };
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("region endpoint is missing"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsUnspecifiedTransfer)
{
    auto exchange = identityExchange();
    exchange.transfer.kind = CouplingTransferKind::Unspecified;
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("explicit transfer"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsTransferDeclarationExtraMetadata)
{
    auto exchange = identityExchange();
    exchange.transfer.interface_declaration = CouplingInterfaceTransferDeclaration{};
    exchange.transfer.driver_owned_name = "copy";
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("interface transfer metadata"),
              std::string::npos);
    EXPECT_NE(formatDiagnostics(validation).find("driver-owned transfer names"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RejectsUnknownFieldEndpoint)
{
    auto exchange = identityExchange();
    exchange.consumer = fieldEndpoint("missing");
    const std::array<CouplingExchangeDeclaration, 1> exchanges{exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("consumer field endpoint is missing"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, RecordsDirectedExchangeCycles)
{
    auto forward = identityExchange();
    auto backward = identityExchange();
    backward.producer_port = port("right_out");
    backward.consumer_port = port("left_in");
    backward.producer = fieldEndpoint("right");
    backward.consumer = fieldEndpoint("left");

    forward.consumer_port = backward.producer_port;
    backward.consumer_port = forward.producer_port;

    const std::array<CouplingExchangeDeclaration, 2> exchanges{forward, backward};
    const PartitionedCouplingPlanGenerator generator;
    const auto plan = generator.generate(
        partitionedContext(),
        std::span<const CouplingExchangeDeclaration>(exchanges));

    ASSERT_FALSE(plan.cycles.empty());
    EXPECT_GE(plan.cycles[0].ports.size(), 3u);
    EXPECT_EQ(plan.cycles[0].ports.front(), plan.cycles[0].ports.back());
}

TEST(PartitionedCouplingPlanGenerator, GeneratesFromContractDeclarations)
{
    const std::array<CouplingContractDeclaration, 1> declarations{
        partitionedDeclaration()};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        partitionedContext(),
        std::span<const CouplingContractDeclaration>(declarations));

    ASSERT_EQ(plan.exchanges.size(), 1u);
    ASSERT_EQ(plan.group_hints.size(), 1u);
    EXPECT_EQ(plan.group_hints[0].name, "sync_group");
    EXPECT_EQ(plan.group_hints[0].participant_names.size(), 2u);
}

TEST(PartitionedCouplingPlanGenerator, MergesDeclarationAndTemplateExchanges)
{
    auto extra_exchange = identityExchange();
    extra_exchange.producer_port = port("right_in");
    extra_exchange.consumer_port = port("left_out");
    extra_exchange.producer = fieldEndpoint("right");
    extra_exchange.consumer = fieldEndpoint("left");

    const std::array<CouplingContractDeclaration, 1> declarations{
        partitionedDeclaration()};
    const std::array<CouplingExchangeDeclaration, 1> templates{extra_exchange};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingExchangeDeclaration>(templates));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        partitionedContext(),
        std::span<const CouplingContractDeclaration>(declarations),
        std::span<const CouplingExchangeDeclaration>(templates));

    EXPECT_EQ(plan.exchanges.size(), 2u);
    EXPECT_EQ(plan.group_hints.size(), 1u);
    EXPECT_FALSE(plan.cycles.empty());
}

TEST(PartitionedCouplingPlanGenerator, RejectsGroupHintWithUnknownParticipant)
{
    auto declaration = partitionedDeclaration();
    declaration.group_hints[0].participant_names.push_back("missing");
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingContractDeclaration>(declarations));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("group hint references an unknown participant"),
              std::string::npos);
}

TEST(PartitionedCouplingPlanGenerator, MergesIdenticalGroupHints)
{
    auto declaration = partitionedDeclaration();
    declaration.group_hints.push_back(declaration.group_hints.front());
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingContractDeclaration>(declarations));
    ASSERT_TRUE(validation.ok()) << formatDiagnostics(validation);

    const auto plan = generator.generate(
        partitionedContext(),
        std::span<const CouplingContractDeclaration>(declarations));

    ASSERT_EQ(plan.group_hints.size(), 1u);
    EXPECT_EQ(plan.group_hints[0].name, "sync_group");
}

TEST(PartitionedCouplingPlanGenerator, RejectsAmbiguousDuplicateGroupHints)
{
    auto declaration = partitionedDeclaration();
    declaration.group_hints.push_back(CouplingGroupHint{
        .name = "sync_group",
        .participant_names = {"left"},
    });
    const std::array<CouplingContractDeclaration, 1> declarations{declaration};

    const PartitionedCouplingPlanGenerator generator;
    const auto validation = generator.validate(
        partitionedContext(),
        std::span<const CouplingContractDeclaration>(declarations));

    EXPECT_FALSE(validation.ok());
    EXPECT_NE(formatDiagnostics(validation).find("duplicates a name with different participants"),
              std::string::npos);
}
