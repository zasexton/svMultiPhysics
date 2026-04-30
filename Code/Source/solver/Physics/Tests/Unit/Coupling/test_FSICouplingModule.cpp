#include "Physics/Coupling/FSICouplingModule.h"

#include <gtest/gtest.h>

#include "Coupling/CouplingFormBuilder.h"
#include "Coupling/CouplingContext.h"
#include "Core/FEException.h"
#include "Mesh/Core/InterfaceMesh.h"
#include "Physics/Tests/Unit/PhysicsTestHelpers.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace svmp;
using namespace svmp::Physics::coupling;
namespace fec = svmp::FE::coupling;
namespace forms = svmp::FE::forms;

namespace {

std::shared_ptr<const FE::spaces::FunctionSpace> multiplierSpace()
{
    return FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Triangle3,
        1,
        3);
}

bool hasField(const fec::CouplingContractDeclaration& declaration,
              const std::string& participant,
              const std::string& field)
{
    return std::any_of(declaration.fields.begin(),
                       declaration.fields.end(),
                       [&](const fec::CouplingFieldUse& use) {
                           return use.participant_name == participant &&
                                  use.field_name == field;
                       });
}

bool hasFieldUse(const std::vector<fec::CouplingFieldUse>& fields,
                 const std::string& participant,
                 const std::string& field)
{
    return std::any_of(fields.begin(),
                       fields.end(),
                       [&](const fec::CouplingFieldUse& use) {
                           return use.participant_name == participant &&
                                  use.field_name == field;
                       });
}

bool containsFormExprType(const forms::FormExprNode& node,
                          forms::FormExprType type)
{
    if (node.type() == type) {
        return true;
    }
    const auto children = node.children();
    return std::any_of(children.begin(),
                       children.end(),
                       [&](const forms::FormExprNode* child) {
                           return child != nullptr &&
                                  containsFormExprType(*child, type);
                       });
}

struct FSIFieldComponents {
    int fluid_velocity{3};
    int fluid_pressure{1};
    int solid_displacement{3};
    int solid_velocity{3};
    int mesh_displacement{3};
};

std::shared_ptr<const FE::spaces::FunctionSpace> space(int components)
{
    return FE::spaces::Space(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Triangle3,
        1,
        components);
}

fec::CouplingFieldRef field(std::string participant_name,
                            std::string system_name,
                            const FE::systems::FESystem* system,
                            std::string field_name,
                            FE::FieldId field_id,
                            int components)
{
    return fec::CouplingFieldRef{
        .participant_name = std::move(participant_name),
        .system_name = std::move(system_name),
        .system = system,
        .field_name = std::move(field_name),
        .field_id = field_id,
        .space = space(components),
        .components = components,
    };
}

fec::CouplingRegionRef interfaceRegion(std::string participant_name,
                                       std::string system_name,
                                       const FE::systems::FESystem* system,
                                       std::string region_name,
                                       int marker,
                                       fec::CouplingInterfaceSide side)
{
    return fec::CouplingRegionRef{
        .participant_name = std::move(participant_name),
        .system_name = std::move(system_name),
        .system = system,
        .region_name = std::move(region_name),
        .kind = fec::CouplingRegionKind::InterfaceFace,
        .marker = marker,
        .side = side,
    };
}

struct FSIContextFixture {
    std::shared_ptr<svmp::Physics::test::SingleTetraMeshAccess> mesh{
        std::make_shared<svmp::Physics::test::SingleTetraMeshAccess>()};
    FE::systems::FESystem fluid_system{mesh};
    FE::systems::FESystem solid_system{mesh};
    FE::systems::FESystem mesh_system{mesh};
    fec::CouplingContext context;

    explicit FSIContextFixture(const FSIFieldComponents& components = {},
                               bool include_mesh = false,
                               bool include_shared_region = true,
                               bool include_fluid_mapping = true,
                               bool include_solid_mapping = true,
                               bool include_mesh_displacement = true)
    {
        fluid_system.setInterfaceMesh(10, std::make_shared<const svmp::InterfaceMesh>());
        solid_system.setInterfaceMesh(20, std::make_shared<const svmp::InterfaceMesh>());

        auto fluid_region = interfaceRegion("fluid",
                                            "fluid_system",
                                            &fluid_system,
                                            "fluid_interface",
                                            10,
                                            fec::CouplingInterfaceSide::Minus);
        auto solid_region = interfaceRegion("solid",
                                            "solid_system",
                                            &solid_system,
                                            "solid_interface",
                                            20,
                                            fec::CouplingInterfaceSide::Plus);
        std::vector<fec::CouplingRegionRef> shared_regions;
        if (include_fluid_mapping) {
            shared_regions.push_back(fluid_region);
        }
        if (include_solid_mapping) {
            shared_regions.push_back(solid_region);
        }

        fec::CouplingContextBuilder builder;
        builder.addParticipant(fec::CouplingParticipantRef{
                   .participant_name = "fluid",
                   .system_name = "fluid_system",
                   .system = &fluid_system,
               })
            .addParticipant(fec::CouplingParticipantRef{
                .participant_name = "solid",
                .system_name = "solid_system",
                .system = &solid_system,
            })
            .addField(field("fluid",
                            "fluid_system",
                            &fluid_system,
                            "velocity",
                            static_cast<FE::FieldId>(1),
                            components.fluid_velocity))
            .addField(field("fluid",
                            "fluid_system",
                            &fluid_system,
                            "pressure",
                            static_cast<FE::FieldId>(2),
                            components.fluid_pressure))
            .addField(field("solid",
                            "solid_system",
                            &solid_system,
                            "displacement",
                            static_cast<FE::FieldId>(3),
                            components.solid_displacement))
            .addField(field("solid",
                            "solid_system",
                            &solid_system,
                            "velocity",
                            static_cast<FE::FieldId>(4),
                            components.solid_velocity))
            .addRegion(fluid_region)
            .addRegion(solid_region);

        if (include_mesh) {
            builder.addParticipant(fec::CouplingParticipantRef{
                       .participant_name = "mesh",
                       .system_name = "mesh_system",
                       .system = &mesh_system,
                   });
            if (include_mesh_displacement) {
                builder.addField(field("mesh",
                                       "mesh_system",
                                       &mesh_system,
                                       "displacement",
                                       static_cast<FE::FieldId>(5),
                                       components.mesh_displacement));
            }
        }

        if (include_shared_region) {
            builder.addSharedRegion(fec::SharedRegionRef{
                .name = "interface",
                .required_region_kind = fec::CouplingRegionKind::InterfaceFace,
                .participant_regions = std::move(shared_regions),
            });
        }
        context = builder.build();
    }
};

void expectValidationFailureContaining(const FSICouplingModule& module,
                                       const fec::CouplingContext& context,
                                       const std::string& message)
{
    try {
        module.validate(context);
        FAIL() << "expected validation to fail";
    } catch (const FE::InvalidArgumentException& exception) {
        EXPECT_NE(std::string(exception.what()).find(message), std::string::npos)
            << exception.what();
    }
}

} // namespace

TEST(FSICouplingModule, DeclaresMonolithicRequirements)
{
    FSICouplingOptions options;
    options.mesh_name = "mesh";

    const FSICouplingModule module(options);
    const auto declaration = module.declare();

    EXPECT_EQ(declaration.contract_type, "fsi");
    EXPECT_EQ(declaration.contract_name, "fsi");
    ASSERT_EQ(declaration.participants.size(), 3u);
    EXPECT_EQ(declaration.participants[0].participant_name, "fluid");
    EXPECT_EQ(declaration.participants[1].participant_name, "solid");
    EXPECT_EQ(declaration.participants[2].participant_name, "mesh");

    EXPECT_TRUE(hasField(declaration, "fluid", "velocity"));
    EXPECT_TRUE(hasField(declaration, "fluid", "pressure"));
    EXPECT_TRUE(hasField(declaration, "solid", "displacement"));
    EXPECT_TRUE(hasField(declaration, "solid", "velocity"));
    EXPECT_TRUE(hasField(declaration, "mesh", "displacement"));

    ASSERT_EQ(declaration.shared_regions.size(), 1u);
    EXPECT_EQ(declaration.shared_regions[0].shared_region_name, "interface");
    ASSERT_TRUE(declaration.shared_regions[0].required_region_kind.has_value());
    EXPECT_EQ(*declaration.shared_regions[0].required_region_kind,
              fec::CouplingRegionKind::InterfaceFace);

    EXPECT_TRUE(declaration.partitioned_exchange_declarations.empty());
}

TEST(FSICouplingModule, DeclaresMultiplierFieldWhenEnabled)
{
    FSICouplingOptions options;
    options.multiplier.enabled = true;
    options.multiplier.contract_field_namespace = "fsi_interface";
    options.multiplier.system_participant_name = "solid";
    options.multiplier.field_name = "interface_multiplier";
    options.multiplier.space = multiplierSpace();
    options.multiplier.components = 3;

    const FSICouplingModule module(options);
    const auto declaration = module.declare();

    ASSERT_EQ(declaration.additional_fields.size(), 1u);
    const auto& field = declaration.additional_fields.front();
    EXPECT_EQ(field.field_namespace, fec::CouplingAdditionalFieldNamespace::Contract);
    EXPECT_EQ(field.namespace_name, "fsi_interface");
    EXPECT_EQ(field.system_participant_name, "solid");
    EXPECT_EQ(field.field_name, "interface_multiplier");
    ASSERT_NE(field.space, nullptr);
    EXPECT_EQ(field.space->value_dimension(), 3);
    EXPECT_EQ(field.components, 3);
    EXPECT_EQ(field.scope, fec::CouplingAdditionalFieldScope::InterfaceFace);
    ASSERT_TRUE(field.shared_region_name.has_value());
    EXPECT_EQ(*field.shared_region_name, "interface");
}

TEST(FSICouplingModule, DeclaresTemporalDerivativeWhenSolidVelocityDerived)
{
    FSICouplingOptions options;
    options.use_solid_displacement_derivative = true;

    const FSICouplingModule module(options);
    const auto declaration = module.declare();

    EXPECT_FALSE(hasField(declaration, "solid", "velocity"));
    ASSERT_EQ(declaration.temporal_requirements.size(), 1u);
    const auto& requirement = declaration.temporal_requirements.front();
    EXPECT_EQ(requirement.quantity, fec::CouplingTemporalQuantity::FieldDerivative);
    ASSERT_TRUE(requirement.field.has_value());
    EXPECT_EQ(requirement.field->participant_name, "solid");
    EXPECT_EQ(requirement.field->field_name, "displacement");
    EXPECT_EQ(requirement.derivative_order, 1);
}

TEST(FSICouplingModule, DeclaresPartitionedExchanges)
{
    FSICouplingOptions options;
    options.mode = fec::CouplingMode::Partitioned;
    options.solid_to_fluid_transfer.kind = fec::CouplingTransferKind::Identity;
    options.fluid_to_solid_transfer.kind = fec::CouplingTransferKind::Identity;

    const FSICouplingModule module(options);
    const auto declaration = module.declare();

    ASSERT_EQ(declaration.partitioned_exchange_declarations.size(), 2u);
    const auto& displacement = declaration.partitioned_exchange_declarations[0];
    EXPECT_EQ(displacement.producer_port.contract_instance_name, "fsi");
    EXPECT_EQ(displacement.producer_port.port_name, "solid_displacement");
    EXPECT_EQ(displacement.consumer_port.port_name, "fluid_displacement");
    EXPECT_EQ(displacement.value.rank, fec::CouplingValueRank::Vector);
    EXPECT_EQ(displacement.value.components, 3);
    ASSERT_TRUE(displacement.producer.has_value());
    EXPECT_EQ(displacement.producer->participant_name, "solid");
    EXPECT_EQ(displacement.producer->endpoint_name, "displacement");
    ASSERT_TRUE(displacement.consumer.has_value());
    EXPECT_EQ(displacement.consumer->participant_name, "fluid");
    EXPECT_EQ(displacement.consumer->endpoint_name, "velocity");
    EXPECT_EQ(displacement.shared_region_name, "interface");
    EXPECT_EQ(displacement.transfer.kind, fec::CouplingTransferKind::Identity);

    const auto& load = declaration.partitioned_exchange_declarations[1];
    EXPECT_EQ(load.producer_port.port_name, "fluid_load");
    EXPECT_EQ(load.consumer_port.port_name, "solid_load");
    ASSERT_TRUE(load.producer.has_value());
    EXPECT_EQ(load.producer->participant_name, "fluid");
    ASSERT_TRUE(load.consumer.has_value());
    EXPECT_EQ(load.consumer->participant_name, "solid");

    ASSERT_EQ(declaration.group_hints.size(), 1u);
    EXPECT_EQ(declaration.group_hints[0].name, "fsi_participants");
    EXPECT_EQ(declaration.group_hints[0].participant_names,
              (std::vector<std::string>{"fluid", "solid"}));

    EXPECT_TRUE(module.buildPartitionedExchangeDeclarations(fec::CouplingContext{}).empty());
}

TEST(FSICouplingModule, BuildsFormsAuthoredVelocityContinuity)
{
    FSIContextFixture fixture;
    const fec::CouplingFormBuilder form_builder(fixture.context);
    const FSICouplingModule module;

    EXPECT_TRUE(module.supportsMonolithicLowering());
    const auto contributions = module.buildMonolithicForms(fixture.context,
                                                           form_builder);
    ASSERT_EQ(contributions.size(), 1u);
    const auto& contribution = contributions.front();
    EXPECT_EQ(contribution.contribution_name, "fsi_velocity_continuity");
    EXPECT_EQ(contribution.origin, "FSICouplingModule");
    EXPECT_EQ(contribution.operator_name, "equations");
    EXPECT_TRUE(hasFieldUse(contribution.field_uses, "fluid", "velocity"));
    EXPECT_TRUE(
        hasFieldUse(contribution.extra_trial_field_uses, "solid", "velocity"));
    EXPECT_TRUE(contribution.terminal_provenance.empty());

    ASSERT_TRUE(contribution.residual.isValid());
    ASSERT_EQ(contribution.residual.node()->type(),
              forms::FormExprType::InterfaceIntegral);
    ASSERT_TRUE(contribution.residual.node()->interfaceMarker().has_value());
    EXPECT_EQ(*contribution.residual.node()->interfaceMarker(), 10);
    EXPECT_TRUE(containsFormExprType(*contribution.residual.node(),
                                     forms::FormExprType::RestrictMinus));
    EXPECT_TRUE(containsFormExprType(*contribution.residual.node(),
                                     forms::FormExprType::RestrictPlus));
}

TEST(FSICouplingModule, BuildsDisplacementDerivativeVelocityContinuity)
{
    FSIContextFixture fixture;
    const fec::CouplingFormBuilder form_builder(fixture.context);

    FSICouplingOptions options;
    options.use_solid_displacement_derivative = true;
    const FSICouplingModule module(options);

    const auto contributions = module.buildMonolithicForms(fixture.context,
                                                           form_builder);
    ASSERT_EQ(contributions.size(), 1u);
    const auto& contribution = contributions.front();
    EXPECT_TRUE(hasFieldUse(contribution.field_uses, "fluid", "velocity"));
    EXPECT_TRUE(hasFieldUse(contribution.extra_trial_field_uses,
                            "solid",
                            "displacement"));
    EXPECT_TRUE(containsFormExprType(*contribution.residual.node(),
                                     forms::FormExprType::TimeDerivative));
}

TEST(FSICouplingModule, RejectsInvalidOptionsDuringValidation)
{
    FSICouplingOptions options;
    options.interface_components = 0;

    const FSICouplingModule module(options);
    EXPECT_THROW(module.validate(fec::CouplingContext{}), FE::InvalidArgumentException);
}

TEST(FSICouplingModule, ValidatesFieldComponentCountsAgainstInterfaceDimension)
{
    FSICouplingOptions options;
    options.mesh_name = "mesh";
    const FSICouplingModule module(options);

    FSIContextFixture valid_fixture(FSIFieldComponents{}, true);
    EXPECT_NO_THROW(module.validate(valid_fixture.context));

    auto components = FSIFieldComponents{};
    components.fluid_velocity = 2;
    FSIContextFixture fluid_velocity_fixture(components, true);
    expectValidationFailureContaining(
        module,
        fluid_velocity_fixture.context,
        "FSI fluid velocity field component count must match the interface component count");

    components = FSIFieldComponents{};
    components.fluid_pressure = 2;
    FSIContextFixture fluid_pressure_fixture(components, true);
    expectValidationFailureContaining(
        module,
        fluid_pressure_fixture.context,
        "FSI fluid pressure field component count must be scalar");

    components = FSIFieldComponents{};
    components.solid_displacement = 2;
    FSIContextFixture solid_displacement_fixture(components, true);
    expectValidationFailureContaining(
        module,
        solid_displacement_fixture.context,
        "FSI solid displacement field component count must match the interface component count");

    components = FSIFieldComponents{};
    components.solid_velocity = 2;
    FSIContextFixture solid_velocity_fixture(components, true);
    expectValidationFailureContaining(
        module,
        solid_velocity_fixture.context,
        "FSI solid velocity field component count must match the interface component count");

    components = FSIFieldComponents{};
    components.mesh_displacement = 2;
    FSIContextFixture mesh_displacement_fixture(components, true);
    expectValidationFailureContaining(
        module,
        mesh_displacement_fixture.context,
        "FSI mesh displacement field component count must match the interface component count");
}

TEST(FSICouplingModule, ValidatesInterfaceSharedRegionMappings)
{
    FSICouplingOptions options;
    const FSICouplingModule module(options);

    FSIContextFixture missing_group(FSIFieldComponents{}, false, false);
    expectValidationFailureContaining(
        module,
        missing_group.context,
        "FSI interface shared region is missing");

    FSIContextFixture missing_fluid(FSIFieldComponents{}, false, true, false, true);
    expectValidationFailureContaining(
        module,
        missing_fluid.context,
        "FSI interface shared region must map the fluid participant");

    FSIContextFixture missing_solid(FSIFieldComponents{}, false, true, true, false);
    expectValidationFailureContaining(
        module,
        missing_solid.context,
        "FSI interface shared region must map the solid participant");
}

TEST(FSICouplingModule, RejectsALEWithoutMeshDisplacement)
{
    FSICouplingOptions options;
    options.mesh_name = "mesh";

    {
        auto missing_field_options = options;
        missing_field_options.mesh_displacement_field = std::nullopt;
        const FSICouplingModule module(missing_field_options);
        FSIContextFixture fixture(FSIFieldComponents{}, true);
        expectValidationFailureContaining(
            module,
            fixture.context,
            "FSI ALE mode requires a mesh displacement field");
    }

    {
        const FSICouplingModule module(options);
        FSIContextFixture fixture(FSIFieldComponents{}, false);
        expectValidationFailureContaining(
            module,
            fixture.context,
            "FSI ALE mode requires a registered mesh participant");
    }

    {
        const FSICouplingModule module(options);
        FSIContextFixture fixture(FSIFieldComponents{},
                                  true,
                                  true,
                                  true,
                                  true,
                                  false);
        expectValidationFailureContaining(
            module,
            fixture.context,
            "FSI ALE mode requires a registered mesh displacement field");
    }
}

TEST(FSICouplingModule, RejectsUnconfiguredPartitionedTransfers)
{
    FSICouplingOptions options;
    options.mode = fec::CouplingMode::Partitioned;

    const FSICouplingModule module(options);
    EXPECT_THROW(module.validate(fec::CouplingContext{}), FE::InvalidArgumentException);
}
