#include "Coupling/CouplingTypes.h"
#include "Coupling/TransferPlan.h"
#include "Tests/Unit/Coupling/CouplingTestHelpers.h"

#include <gtest/gtest.h>

using namespace svmp::FE;
using namespace svmp::FE::coupling;
namespace coupling_test = svmp::FE::coupling::test;

TEST(CouplingTypes, PortIdentityUsesContractInstanceAndPortName)
{
    const CouplingPortId a{"contract_a", "out"};
    const CouplingPortId b{"contract_a", "in"};
    const CouplingPortId c{"contract_b", "out"};

    EXPECT_TRUE(a.valid());
    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
    EXPECT_TRUE(a < c);
    EXPECT_TRUE(validateCouplingPortId(a).ok());

    const CouplingPortId invalid{"", "out"};
    EXPECT_FALSE(validateCouplingPortId(invalid).ok());
}

TEST(CouplingTypes, GeneratedNamesUseStableContractRelationAndLocalParts)
{
    CouplingGeneratedNameRequest request{
        .contract_name = "fsi_wall",
        .relation_name = "interface",
        .local_name = "traction_balance",
    };

    EXPECT_TRUE(validateCouplingGeneratedNameRequest(request).ok());
    EXPECT_EQ(makeCouplingGeneratedName(request),
              "fsi_wall.interface.traction_balance");

    request.explicit_name = "custom_traction_form";
    EXPECT_TRUE(validateCouplingGeneratedNameRequest(request).ok());
    EXPECT_EQ(makeCouplingGeneratedName(request), "custom_traction_form");
}

TEST(CouplingTypes, GeneratedNameValidationRejectsAmbiguousParts)
{
    CouplingGeneratedNameRequest request{
        .contract_name = "fsi.wall",
        .relation_name = "",
        .local_name = "traction.balance",
    };

    const auto validation = validateCouplingGeneratedNameRequest(request);
    EXPECT_FALSE(validation.ok());
    const auto text = formatDiagnostics(validation);
    EXPECT_NE(text.find("contract name part must not contain"), std::string::npos);
    EXPECT_NE(text.find("relation name part must be nonempty"), std::string::npos);
    EXPECT_NE(text.find("local name part must not contain"), std::string::npos);

    request.explicit_name = "";
    const auto override_validation =
        validateCouplingGeneratedNameRequest(request);
    EXPECT_FALSE(override_validation.ok());
    EXPECT_NE(formatDiagnostics(override_validation).find(
                  "explicit generated-name override must be nonempty"),
              std::string::npos);
}

TEST(CouplingTypes, ValueDescriptorValidatesGeneralTensorShape)
{
    CouplingValueDescriptor value;
    value.rank = CouplingValueRank::GeneralTensor;
    value.components = 6;
    value.tensor_extents = {2, 3};
    value.tensor_packing = "row_major";

    EXPECT_TRUE(validateCouplingValueDescriptor(value).ok());
    EXPECT_EQ(couplingTensorExtentProduct(value.tensor_extents), 6u);

    value.components = 5;
    EXPECT_FALSE(validateCouplingValueDescriptor(value).ok());
}

TEST(CouplingTypes, CoordinateConfigurationMapsExplicitlyToMeshConfiguration)
{
    EXPECT_EQ(toMeshConfiguration(CouplingCoordinateConfiguration::Reference),
              svmp::Configuration::Reference);
    EXPECT_EQ(toMeshConfiguration(CouplingCoordinateConfiguration::Current),
              svmp::Configuration::Current);
    EXPECT_FALSE(toMeshConfiguration(
                     static_cast<CouplingCoordinateConfiguration>(255))
                     .has_value());
}

TEST(CouplingTypes, ValueDescriptorRequiresMixedBlockLayout)
{
    CouplingValueDescriptor value;
    value.rank = CouplingValueRank::MixedBlock;
    value.components = 2;
    EXPECT_FALSE(validateCouplingValueDescriptor(value).ok());

    value.component_layout = {"primary", "secondary"};
    EXPECT_TRUE(validateCouplingValueDescriptor(value).ok());
}

TEST(CouplingTypes, ValueDescriptorRejectsAmbiguousComponentLayout)
{
    CouplingValueDescriptor value;
    value.rank = CouplingValueRank::Vector;
    value.components = 2;
    value.component_layout = {"normal", ""};
    EXPECT_FALSE(validateCouplingValueDescriptor(value).ok());

    value.component_layout = {"normal", "normal"};
    EXPECT_FALSE(validateCouplingValueDescriptor(value).ok());

    value.component_layout = {"normal", "tangent"};
    EXPECT_TRUE(validateCouplingValueDescriptor(value).ok());
}

TEST(CouplingTypes, ValueDescriptorValidatesRankComponentMinimums)
{
    CouplingValueDescriptor value;
    value.rank = CouplingValueRank::Vector;
    value.components = 1;
    EXPECT_FALSE(validateCouplingValueDescriptor(value).ok());

    value.rank = CouplingValueRank::Rank2Tensor;
    value.components = 3;
    EXPECT_FALSE(validateCouplingValueDescriptor(value).ok());

    value.rank = CouplingValueRank::SymmetricTensor;
    value.components = 2;
    EXPECT_FALSE(validateCouplingValueDescriptor(value).ok());

    value.rank = CouplingValueRank::Rank2Tensor;
    value.components = 4;
    EXPECT_TRUE(validateCouplingValueDescriptor(value).ok());
}

TEST(CouplingTypes, ValueDescriptorChecksUnitAndDimensionMetadataWhenPresent)
{
    CouplingValueDescriptor pressure;
    pressure.rank = CouplingValueRank::Scalar;
    pressure.components = 1;
    pressure.physical_dimension = "pressure";
    pressure.unit = "Pa";
    EXPECT_TRUE(validateCouplingValueDescriptor(pressure).ok());

    CouplingValueDescriptor incomplete_unit = pressure;
    incomplete_unit.physical_dimension.clear();
    EXPECT_FALSE(validateCouplingValueDescriptor(incomplete_unit).ok());

    CouplingValueDescriptor unspecified;
    unspecified.rank = CouplingValueRank::Scalar;
    unspecified.components = 1;
    EXPECT_TRUE(couplingValueDescriptorsCompatible(pressure, unspecified));

    CouplingValueDescriptor pressure_kpa = pressure;
    pressure_kpa.unit = "kPa";
    EXPECT_FALSE(couplingValueDescriptorsCompatible(pressure, pressure_kpa));

    CouplingValueDescriptor force = pressure;
    force.physical_dimension = "force";
    EXPECT_FALSE(couplingValueDescriptorsCompatible(pressure, force));
}

TEST(CouplingTypes, TemporalSlotValidationUsesLogicalHistoryAndStageRules)
{
    CouplingTemporalSlotDescriptor history;
    history.slot = CouplingTemporalSlot::History;
    history.history_index = 1;
    EXPECT_TRUE(validateCouplingTemporalSlot(history).ok());

    history.history_index = 0;
    EXPECT_FALSE(validateCouplingTemporalSlot(history).ok());

    CouplingTemporalSlotDescriptor current;
    current.slot = CouplingTemporalSlot::Current;
    current.history_index = 1;
    EXPECT_FALSE(validateCouplingTemporalSlot(current).ok());

    CouplingTemporalSlotDescriptor stage;
    stage.slot = CouplingTemporalSlot::Stage;
    stage.stage_index = 0;
    EXPECT_TRUE(validateCouplingTemporalSlot(stage).ok());
}

TEST(CouplingTypes, EndpointValidationRequiresParticipantScopeForFEBackedKinds)
{
    CouplingEndpointRef endpoint;
    endpoint.kind = CouplingEndpointKind::Field;
    endpoint.endpoint_name = "primary";
    endpoint.participant_name = "participant";
    EXPECT_TRUE(validateCouplingEndpointRef(endpoint).ok());

    endpoint.participant_name.reset();
    EXPECT_FALSE(validateCouplingEndpointRef(endpoint).ok());

    endpoint.kind = CouplingEndpointKind::ExternalBuffer;
    EXPECT_TRUE(validateCouplingEndpointRef(endpoint).ok());
}

TEST(CouplingTypes, TransferKindIdentifiesInterfaceTransfers)
{
    EXPECT_FALSE(isInterfaceTransferKind(CouplingTransferKind::Identity));
    EXPECT_TRUE(isInterfaceTransferKind(CouplingTransferKind::InterfacePointwiseInterpolation));
    EXPECT_STREQ(toString(CouplingTransferKind::DriverOwned), "driver_owned");
}

TEST(CouplingTypes, TestHelpersConstructReusablePhaseEightFixtureRecords)
{
    const auto left = coupling_test::participantBinding("left", 1u);
    const auto right = coupling_test::participantBinding("right", 2u);
    EXPECT_EQ(left.system_name, "left_system");
    EXPECT_NE(left.system, nullptr);

    const auto context = coupling_test::twoParticipantContext(3);
    EXPECT_TRUE(context.hasParticipant("left"));
    EXPECT_TRUE(context.hasParticipant("right"));
    EXPECT_EQ(context.field("left", "primary").components, 3);
    const auto shared = context.sharedRegionGroup("interface");
    ASSERT_EQ(shared.participant_regions.size(), 2u);
    EXPECT_EQ(shared.participant_regions[0].side, CouplingInterfaceSide::Minus);
    EXPECT_EQ(shared.participant_regions[1].side, CouplingInterfaceSide::Plus);
    ASSERT_EQ(context.externalBuffers().size(), 1u);
    EXPECT_EQ(context.externalBuffers()[0].descriptor.buffer_name, "driver_buffer");
    ASSERT_EQ(context.driverOwnedTransfers().size(), 1u);
    EXPECT_EQ(context.driverOwnedTransfers()[0].transfer_name, "driver_transfer");

    const auto contract_type = coupling_test::contractTypeKey("surface_balance");
    const auto contract_instance =
        coupling_test::contractInstanceName("surface_balance", 2);
    EXPECT_EQ(contract_type, "fixture.surface_balance");
    EXPECT_EQ(contract_instance, "surface_balance_2");
    EXPECT_TRUE(validateCouplingPortId(
                    coupling_test::portId(contract_instance, "out"))
                    .ok());

    const auto vector_value = coupling_test::vectorValueDescriptor(3);
    EXPECT_TRUE(validateCouplingValueDescriptor(vector_value).ok());
    const auto tensor_value =
        coupling_test::generalTensorValueDescriptor({2, 3});
    EXPECT_TRUE(validateCouplingValueDescriptor(tensor_value).ok());
    EXPECT_EQ(tensor_value.components, 6);

    const auto history = coupling_test::historySlot(2);
    const auto stage = coupling_test::stageSlot(0);
    EXPECT_TRUE(validateCouplingTemporalSlot(history).ok());
    EXPECT_TRUE(validateCouplingTemporalSlot(stage).ok());
    EXPECT_EQ(history.history_index, 2);

    const auto field_endpoint = coupling_test::endpointRef(
        CouplingEndpointKind::Field,
        left.participant_name,
        "primary",
        history);
    EXPECT_TRUE(validateCouplingEndpointRef(field_endpoint).ok());
    const auto resolved_endpoint = coupling_test::resolvedFieldEndpoint(
        left,
        "primary",
        1,
        vector_value,
        coupling_test::currentSlot());
    EXPECT_EQ(resolved_endpoint.resolved_kind, CouplingEndpointKind::Field);
    EXPECT_EQ(resolved_endpoint.system_name, "left_system");
    EXPECT_EQ(resolved_endpoint.field_id, 1);

    const auto driver_descriptor =
        coupling_test::driverOwnedTransferDescriptor("driver_transfer");
    EXPECT_EQ(driver_descriptor.supported_ranks.back(),
              CouplingValueRank::GeneralTensor);
    const auto driver_transfer =
        coupling_test::driverOwnedTransfer(driver_descriptor.transfer_name);
    EXPECT_EQ(driver_transfer.kind, CouplingTransferKind::DriverOwned);
    const auto resolved_transfer =
        coupling_test::resolvedDriverOwnedTransfer(driver_descriptor);
    ASSERT_TRUE(resolved_transfer.driver_owned_descriptor.has_value());
    EXPECT_EQ(resolved_transfer.driver_owned_descriptor->transfer_name,
              "driver_transfer");
    EXPECT_EQ(coupling_test::unspecifiedTransfer().kind,
              CouplingTransferKind::Unspecified);
    EXPECT_EQ(coupling_test::identityTransfer().kind,
              CouplingTransferKind::Identity);

    const auto space =
        std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    const auto participant_field =
        coupling_test::participantAdditionalField(left, space);
    EXPECT_EQ(participant_field.field_namespace,
              CouplingAdditionalFieldNamespace::Participant);
    EXPECT_EQ(participant_field.namespace_name, "left");
    const auto contract_field = coupling_test::contractAdditionalField(
        left,
        space,
        contract_instance);
    EXPECT_EQ(contract_field.field_namespace,
              CouplingAdditionalFieldNamespace::Contract);
    EXPECT_EQ(contract_field.shared_region_name, "interface");

    const auto location = coupling_test::geometryTerminalLocation(
        CouplingRegionKind::InterfaceFace,
        std::optional<std::string>("interface"));
    EXPECT_EQ(location.region_kind, CouplingRegionKind::InterfaceFace);
    EXPECT_EQ(location.side, CouplingInterfaceSide::Minus);
    ASSERT_TRUE(location.transform_from_configuration.has_value());
    EXPECT_EQ(*location.transform_to_configuration,
              forms::GeometryConfiguration::Current);
    const auto requirement = coupling_test::geometryTerminalRequirement(
        CouplingGeometryTerminalQuantity::SurfaceJacobian,
        left,
        "interface",
        CouplingRegionKind::InterfaceFace);
    ASSERT_TRUE(requirement.scope.location.has_value());
    EXPECT_EQ(requirement.scope.location->shared_region_name, "interface");
    const auto terminal_provenance =
        coupling_test::geometryTerminalProvenance(
            CouplingGeometryTerminalQuantity::SurfaceJacobian,
            left,
            "interface",
            7,
            CouplingRegionKind::InterfaceFace);
    EXPECT_EQ(terminal_provenance.analysis_domain,
              analysis::DomainKind::InterfaceFace);
    ASSERT_TRUE(terminal_provenance.owner.has_value());
    EXPECT_EQ(terminal_provenance.owner->system_name, "left_system");

    const auto sensitivity_declaration =
        coupling_test::geometrySensitivityDeclaration(left);
    ASSERT_TRUE(sensitivity_declaration.mesh_motion_field.has_value());
    EXPECT_EQ(sensitivity_declaration.mesh_motion_field->participant_name,
              "left");
    const auto sensitivity_provenance =
        coupling_test::geometrySensitivityProvenance(3);
    EXPECT_TRUE(sensitivity_provenance.ad_compatible);
    EXPECT_EQ(sensitivity_provenance.geometry_fields[0], 3);

    const auto install_declaration =
        coupling_test::formInstallOptionsDeclaration(left);
    EXPECT_EQ(install_declaration.ad_mode, forms::ADMode::Reverse);
    ASSERT_TRUE(install_declaration.geometry_sensitivity.has_value());
    const auto install_options = coupling_test::resolvedFormInstallOptions();
    EXPECT_EQ(install_options.ad_mode, forms::ADMode::Reverse);
    EXPECT_EQ(install_options.compiler_options.ad_mode,
              forms::ADMode::Reverse);
    EXPECT_EQ(install_options.extra_trial_fields[0], 3);
    const auto contribution = coupling_test::formContribution(
        left,
        "surface_balance_form",
        "coupling_test");
    EXPECT_EQ(contribution.contribution_name, "surface_balance_form");
    ASSERT_EQ(contribution.field_uses.size(), 1u);
    ASSERT_EQ(contribution.extra_trial_field_uses.size(), 1u);

    const auto expert_metadata =
        coupling_test::expertInstallMetadata(left, "expert_surface_balance");
    EXPECT_EQ(expert_metadata.system_name, "left_system");
    ASSERT_EQ(expert_metadata.installed_dependencies.size(), 1u);
    ASSERT_EQ(expert_metadata.installed_blocks.size(), 1u);
    EXPECT_TRUE(expert_metadata.installed_blocks[0].has_matrix);

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto map_provenance = coupling_test::interfaceMapProvenance();
    EXPECT_EQ(map_provenance.interface_entry_name, "left_to_right_entry");
    EXPECT_EQ(map_provenance.source_logical_region.physical_label, 7);
    EXPECT_EQ(map_provenance.target_logical_region.physical_label, 8);
    EXPECT_EQ(map_provenance.map_state,
              svmp::search::InterfaceMapState::Committed);
    const auto interface_transfer = coupling_test::interfaceTransfer();
    EXPECT_EQ(interface_transfer.kind,
              CouplingTransferKind::InterfacePointwiseInterpolation);
    ASSERT_TRUE(interface_transfer.interface_declaration.has_value());
    ASSERT_TRUE(interface_transfer.interface_map.has_value());
    EXPECT_EQ(interface_transfer.interface_declaration->source_embedding_policy,
              CouplingFrameSourceEmbeddingPolicy::Embed2DInXY);
#endif
}
