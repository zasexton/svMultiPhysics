/**
 * @file test_LevelSetGeometry.cpp
 * @brief Typed domains of a bound level-set geometry use the stable generated markers
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Forms/IntegrationDomain.h"
#include "Forms/Vocabulary.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "LevelSet/LevelSetGeometry.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/SystemManifest.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <memory>
#include <stdexcept>
#include <vector>

namespace svmp::FE::level_set::test {
namespace {

using forms::CutVolumeSide;
using forms::FormExpr;

struct GeometrySystem {
    std::shared_ptr<forms::test::SingleTetraMeshAccess> mesh{
        std::make_shared<forms::test::SingleTetraMeshAccess>()};
    std::shared_ptr<spaces::H1Space> scalar{
        std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1)};
    std::shared_ptr<spaces::ProductSpace> vector{
        std::make_shared<spaces::ProductSpace>(scalar, 3)};
    std::unique_ptr<systems::FESystem> system{};
    FieldId phi{INVALID_FIELD_ID};
    FieldId u{INVALID_FIELD_ID};

    GeometrySystem()
    {
        system = std::make_unique<systems::FESystem>(mesh);
        u = system->addField(systems::FieldSpec{.name = "u", .space = vector, .components = 3});
        phi = system->addField(systems::FieldSpec{.name = "phi", .space = scalar, .components = 1});
    }
};

TEST(LevelSetGeometry, BindResolvesTheStableGeneratedInterfaceMarker)
{
    GeometrySystem s;
    const auto G = LevelSetGeometry::bind(
        *s.system, {.level_set_field_name = "phi", .domain_id = "drop", .isovalue = 0.25});
    interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = interfaces::LevelSetInterfaceSource::fromField(s.phi);
    key.domain_id = "drop";
    key.isovalue = 0.25;
    key.requested_marker = -1;
    EXPECT_EQ(G.levelSetField(), s.phi);
    EXPECT_EQ(G.interfaceMarker(), interfaces::stableGeneratedInterfaceMarker(key));
    EXPECT_EQ(G.interface().marker(), G.interfaceMarker());

    // By field id, and with an explicit marker.
    const auto by_id = LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop", .isovalue = 0.25});
    EXPECT_EQ(by_id.interfaceMarker(), G.interfaceMarker());
    const auto explicit_marker =
        LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop", .interface_marker = 12});
    EXPECT_EQ(explicit_marker.interfaceMarker(), 12);
}

TEST(LevelSetGeometry, BindRejectsUnknownOrVectorFieldsAndMissingDomainIds)
{
    GeometrySystem s;
    EXPECT_THROW((void)LevelSetGeometry::bind(*s.system, {.level_set_field_name = "psi", .domain_id = "drop"}),
                 std::invalid_argument);
    EXPECT_THROW((void)LevelSetGeometry::bind(*s.system, {.level_set_field = s.u, .domain_id = "drop"}),
                 std::invalid_argument);
    EXPECT_THROW((void)LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi}), std::invalid_argument);
}

TEST(LevelSetGeometry, SideDomainsUseTheInterfaceMarkerAndSide)
{
    GeometrySystem s;
    const auto G = LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop"});
    for (const auto side : {CutVolumeSide::Negative, CutVolumeSide::Positive}) {
        const auto L = G.side(side);
        EXPECT_EQ(L.side(), side);
        EXPECT_EQ(L.volume(), forms::VolumeDomain::cut(G.interfaceMarker(), side));
        EXPECT_EQ(L.interface(), forms::InterfaceDomain(G.interfaceMarker()));
        EXPECT_EQ(L.cutFacets(), forms::FacetDomain::cutAdjacent(G.interfaceMarker(), side));
    }
}

TEST(LevelSetGeometry, OutwardNormalIsTheRuleNormalFlippedOnThePositiveSide)
{
    GeometrySystem s;
    const auto G = LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop"});
    const auto n = FormExpr::normal();
    EXPECT_EQ(systems::formExprTreeHash(G.side(CutVolumeSide::Negative).outwardNormal()),
              systems::formExprTreeHash(n));
    EXPECT_EQ(systems::formExprTreeHash(G.side(CutVolumeSide::Positive).outwardNormal()),
              systems::formExprTreeHash(-n));
    EXPECT_EQ(G.side(CutVolumeSide::Positive).outwardNormal().toString(), (-n).toString());
}

TEST(LevelSetGeometry, ContactLineUsesTheStableIntersectionMarker)
{
    GeometrySystem s;
    const auto G = LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop", .isovalue = 0.0});
    interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key{};
    key.source = interfaces::LevelSetInterfaceSource::fromField(s.phi);
    key.domain_id = "drop";
    key.isovalue = 0.0;
    key.interface_marker = G.interfaceMarker();
    key.boundary_marker = 4;
    key.requested_marker = -1;
    EXPECT_EQ(G.contactLineKey(4).stableKey(), key.stableKey());
    const auto CL = G.contactLine(4);
    EXPECT_EQ(CL.marker(), interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(key));
    EXPECT_EQ(CL.wallBoundaryMarker(), 4);
    EXPECT_EQ(G.side(CutVolumeSide::Negative).contactLine(4), CL);
    key.requested_marker = 77;
    EXPECT_EQ(G.contactLine(4, 77).marker(),
              interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(key));
}

TEST(LevelSetGeometry, WettedBoundaryIsTheGeneratedActiveSubsetOfTheSide)
{
    GeometrySystem s;
    const auto G = LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop"});
    for (const auto side : {CutVolumeSide::Negative, CutVolumeSide::Positive}) {
        const auto L = G.side(side);
        interfaces::GeneratedActiveBoundaryMarkerKey key;
        key.source = interfaces::LevelSetInterfaceSource::fromField(s.phi);
        key.domain_id = "drop";
        key.isovalue = 0.0;
        key.interface_marker = G.interfaceMarker();
        key.boundary_marker = 2;
        key.side = side == CutVolumeSide::Negative ? geometry::CutIntegrationSide::Negative
                                                   : geometry::CutIntegrationSide::Positive;
        EXPECT_EQ(L.boundaryKey(2).stableKey(), key.stableKey());
        const auto expected = forms::BoundaryDomain::generatedActiveSubset(
            2, interfaces::stableGeneratedActiveBoundaryMarker(key));
        EXPECT_EQ(L.boundary(2), expected);
        EXPECT_EQ(L.wetted(2), expected);
    }
    EXPECT_NE(G.side(CutVolumeSide::Negative).wetted(2), G.side(CutVolumeSide::Positive).wetted(2));
}

TEST(LevelSetGeometry, DomainsGiveTheSameFormsAsMarkersWrittenByHand)
{
    GeometrySystem s;
    const auto G = LevelSetGeometry::bind(*s.system, {.level_set_field = s.phi, .domain_id = "drop"});
    const auto L = G.side(CutVolumeSide::Positive);
    const auto v = forms::TestField(s.u, *s.vector, "v");
    const auto u = forms::StateField(s.u, *s.vector, "u");
    const auto integrand = forms::inner(L.outwardNormal(), v);
    const int m = G.interfaceMarker();
    EXPECT_EQ(systems::formExprTreeHash(integrand.dI(L.interface())),
              systems::formExprTreeHash(integrand.dI(m)));
    EXPECT_EQ(systems::formExprTreeHash(forms::inner(u, v).dx(L.volume())),
              systems::formExprTreeHash(forms::inner(u, v).dCutVolume(m, CutVolumeSide::Positive)));
    const auto CL = G.contactLine(4);
    EXPECT_EQ(systems::formExprTreeHash(integrand.dL(CL)),
              systems::formExprTreeHash(integrand.dI(CL.marker())));
    const auto W = L.wetted(4);
    EXPECT_EQ(systems::formExprTreeHash(forms::inner(u, v).ds(W)),
              systems::formExprTreeHash(forms::inner(u, v).dExteriorBoundary(
                  forms::ExteriorBoundaryMeasure::generatedActiveSubset(4, W.generatedActiveBoundaryMarker()))));
}

TEST(LevelSetGeometry, CutVolumeRequestPolicyKeyIgnoresProvenance)
{
    ActiveCutVolumeRequest a;
    a.level_set_field_name = "phi";
    a.domain_id = "drop";
    ActiveCutVolumeRequest b = a;
    b.origin = ActiveCutVolumeRequestOrigin::FreeSurfaceBoundary;
    b.equation_type = "fluid";
    b.geometry_tangent_policy_explicit = true;
    EXPECT_EQ(activeCutVolumeRequestPolicyKey({a}), activeCutVolumeRequestPolicyKey({b}));
    b.active_side = LevelSetActiveSide::Positive;
    EXPECT_NE(activeCutVolumeRequestPolicyKey({a}), activeCutVolumeRequestPolicyKey({b}));
    EXPECT_FALSE(hasHighOrderGeneratedInterfaceGeometry({a}));
}

} // namespace
} // namespace svmp::FE::level_set::test
