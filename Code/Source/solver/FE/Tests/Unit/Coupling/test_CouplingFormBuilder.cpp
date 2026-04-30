#include "Coupling/CouplingFormBuilder.h"

#include "Core/FEException.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

using namespace svmp::FE;
using namespace svmp::FE::coupling;

namespace {

const systems::FESystem* builderSystemToken()
{
    return reinterpret_cast<const systems::FESystem*>(1);
}

CouplingContext makeBuilderContext()
{
    const auto* system = builderSystemToken();
    const auto space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);

    CouplingContextBuilder builder;
    builder.addParticipant({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
    });
    builder.addField({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
        .field_name = "primary",
        .field_id = 7,
        .space = space,
        .components = 1,
    });
    builder.addRegion({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
        .region_name = "volume",
        .kind = CouplingRegionKind::Domain,
    });
    builder.addRegion({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
        .region_name = "surface",
        .kind = CouplingRegionKind::Boundary,
        .marker = 12,
    });
    builder.addRegion({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
        .region_name = "interior",
        .kind = CouplingRegionKind::InteriorFace,
    });
    builder.addRegion({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
        .region_name = "interface",
        .kind = CouplingRegionKind::InterfaceFace,
        .marker = 17,
        .side = CouplingInterfaceSide::Minus,
    });
    builder.addRegion({
        .participant_name = "participant",
        .system_name = "system",
        .system = system,
        .region_name = "provider_owned",
        .kind = CouplingRegionKind::UserDefined,
    });
    builder.addSharedRegion(SharedRegionRef{
        .name = "shared_interface",
        .required_region_kind = CouplingRegionKind::InterfaceFace,
        .participant_regions = {{
            .participant_name = "participant",
            .system_name = "system",
            .system = system,
            .region_name = "interface",
            .kind = CouplingRegionKind::InterfaceFace,
            .marker = 17,
            .side = CouplingInterfaceSide::Minus,
        }},
    });
    return builder.build();
}

} // namespace

TEST(CouplingFormBuilder, BuildsFieldBoundStateAndTestSymbols)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);

    const auto state = builder.state("participant", "primary", "u");
    const auto test = builder.test("participant", "primary", "w");

    ASSERT_TRUE(state.isValid());
    ASSERT_TRUE(test.isValid());
    EXPECT_EQ(state.node()->type(), forms::FormExprType::StateField);
    EXPECT_EQ(test.node()->type(), forms::FormExprType::TestFunction);
    ASSERT_TRUE(state.node()->fieldId().has_value());
    ASSERT_TRUE(test.node()->fieldId().has_value());
    EXPECT_EQ(*state.node()->fieldId(), 7);
    EXPECT_EQ(*test.node()->fieldId(), 7);
    EXPECT_NE(state.toString().find("u"), std::string::npos);
}

TEST(CouplingFormBuilder, BuildsTemporalTermsThroughFormsVocabulary)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);

    const auto derivative = builder.timeDerivative("participant", "primary", "u", 2);
    ASSERT_TRUE(derivative.isValid());
    EXPECT_EQ(derivative.node()->type(), forms::FormExprType::TimeDerivative);
    ASSERT_TRUE(derivative.node()->timeDerivativeOrder().has_value());
    EXPECT_EQ(*derivative.node()->timeDerivativeOrder(), 2);

    const auto previous = builder.previousSolution("participant", "primary", 3);
    ASSERT_TRUE(previous.isValid());
    EXPECT_EQ(previous.node()->type(), forms::FormExprType::PreviousSolutionRef);
    ASSERT_TRUE(previous.node()->historyIndex().has_value());
    EXPECT_EQ(*previous.node()->historyIndex(), 3);

    EXPECT_EQ(builder.time().node()->type(), forms::FormExprType::Time);
    EXPECT_EQ(builder.timeStep().node()->type(), forms::FormExprType::TimeStep);
    EXPECT_EQ(builder.effectiveTimeStep().node()->type(), forms::FormExprType::EffectiveTimeStep);
}

TEST(CouplingFormBuilder, RejectsInvalidTemporalRequestsAndUnknownFields)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);

    EXPECT_THROW(static_cast<void>(builder.timeDerivative("participant", "primary", "u", 0)),
                 InvalidArgumentException);
    EXPECT_THROW(static_cast<void>(builder.previousSolution("participant", "primary", 0)),
                 InvalidArgumentException);
    EXPECT_THROW(static_cast<void>(builder.state("participant", "missing", "u")),
                 InvalidArgumentException);
}

TEST(CouplingFormBuilder, DelegatesRegionLookupsToContext)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);

    const auto region = builder.region("participant", "surface");
    EXPECT_EQ(region.marker, 12);
    EXPECT_EQ(region.kind, CouplingRegionKind::Boundary);
}

TEST(CouplingFormBuilder, LowersRegionKindsToFormsMeasures)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);
    const auto integrand =
        builder.state("participant", "primary", "u") *
        builder.test("participant", "primary", "w");

    const auto cell = builder.integrate(integrand, "participant", "volume");
    ASSERT_TRUE(cell.isValid());
    EXPECT_EQ(cell.node()->type(), forms::FormExprType::CellIntegral);

    const auto boundary = builder.integrate(integrand, "participant", "surface");
    ASSERT_TRUE(boundary.isValid());
    EXPECT_EQ(boundary.node()->type(), forms::FormExprType::BoundaryIntegral);
    ASSERT_TRUE(boundary.node()->boundaryMarker().has_value());
    EXPECT_EQ(*boundary.node()->boundaryMarker(), 12);

    const auto interior = builder.integrate(integrand, "participant", "interior");
    ASSERT_TRUE(interior.isValid());
    EXPECT_EQ(interior.node()->type(), forms::FormExprType::InteriorFaceIntegral);

    const auto interface = builder.integrate(integrand, "participant", "interface");
    ASSERT_TRUE(interface.isValid());
    EXPECT_EQ(interface.node()->type(), forms::FormExprType::InterfaceIntegral);
    ASSERT_TRUE(interface.node()->interfaceMarker().has_value());
    EXPECT_EQ(*interface.node()->interfaceMarker(), 17);

    const auto shared_interface =
        builder.integrateShared(integrand, "shared_interface", "participant");
    ASSERT_TRUE(shared_interface.isValid());
    EXPECT_EQ(shared_interface.node()->type(), forms::FormExprType::InterfaceIntegral);
    ASSERT_TRUE(shared_interface.node()->interfaceMarker().has_value());
    EXPECT_EQ(*shared_interface.node()->interfaceMarker(), 17);
}

TEST(CouplingFormBuilder, RejectsUserDefinedRegionFormsLowering)
{
    const auto context = makeBuilderContext();
    const CouplingFormBuilder builder(context);
    const auto integrand =
        builder.state("participant", "primary", "u") *
        builder.test("participant", "primary", "w");

    EXPECT_THROW(static_cast<void>(
                     builder.integrate(integrand, "participant", "provider_owned")),
                 InvalidArgumentException);
}
