/**
 * @file test_LevelSetGeometry.cpp
 * @brief Typed domains of a bound level-set geometry use the stable generated markers
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/ConstraintPlan.h"
#include "Core/FEConfig.h"
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
#include "Systems/FormsInstaller.h"
#include "Systems/FunctionalFormEvaluator.h"
#include "Systems/SystemCollectives.h"
#include "Systems/SystemManifest.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#if SVMP_FE_WITH_MESH
#include "LevelSet/LevelSetGeometryService.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Tests/Unit/Systems/TraceMultiplierTestMeshes.h"
#endif

#if FE_HAS_EIGEN
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#endif

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
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

TEST(LevelSetGeometry, ScalarLevelSetFieldIsResolvedWithTheOwnersMessages)
{
    GeometrySystem s;
    EXPECT_EQ(resolveScalarLevelSetField(*s.system, "phi", "Owner: unknown field '", "Owner"), s.phi);
    try {
        (void)resolveScalarLevelSetField(*s.system, "psi", "Owner: unknown field '", "Owner");
        FAIL() << "unknown field accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()), "Owner: unknown field 'psi'");
    }
    try {
        (void)resolveScalarLevelSetField(*s.system, "u", "Owner: unknown field '", "Owner");
        FAIL() << "vector field accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()), "Owner: level-set field 'u' must be scalar");
    }
}

TEST(LevelSetGeometry, GeneratedActiveBoundaryMarkerMustAgreeOverItsOwners)
{
    interfaces::GeneratedActiveBoundaryMarkerKey first{};
    first.source = interfaces::LevelSetInterfaceSource::fromField(FieldId{1});
    first.domain_id = "drop";
    first.interface_marker = 5;
    first.boundary_marker = 2;
    auto second = first;
    second.side = geometry::CutIntegrationSide::Positive;
    const std::vector<std::optional<interfaces::GeneratedActiveBoundaryMarkerKey>> agree{
        std::nullopt, first, first};
    std::vector<std::size_t> asked;
    const auto marker = resolveGeneratedActiveBoundaryMarker(
        2, agree.size(), [&](std::size_t i) { asked.push_back(i); return agree[i]; }, "Owner");
    ASSERT_TRUE(marker.has_value());
    EXPECT_EQ(*marker, interfaces::stableGeneratedActiveBoundaryMarker(first));
    EXPECT_EQ(asked, (std::vector<std::size_t>{0, 1, 2}));
    EXPECT_FALSE(resolveGeneratedActiveBoundaryMarker(
                     2, 1, [](std::size_t) { return std::optional<interfaces::GeneratedActiveBoundaryMarkerKey>{}; },
                     "Owner")
                     .has_value());

    const std::vector<std::optional<interfaces::GeneratedActiveBoundaryMarkerKey>> ambiguous{first, second};
    try {
        (void)resolveGeneratedActiveBoundaryMarker(
            2, ambiguous.size(), [&](std::size_t i) { return ambiguous[i]; }, "Owner");
        FAIL() << "ambiguous owners accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()),
                  "Owner: physical boundary marker 2 is covered by multiple ambiguous unfitted active "
                  "domains; select one explicit active-domain owner");
    }
}

TEST(LevelSetGeometry, WettedBoundaryResolverRegistersTheGeneratedMarkers)
{
    GeometrySystem s;
    const auto resolve = wettedBoundaryResolver(
        *s.system, [](int marker) -> std::optional<int> {
            return marker == 1 ? std::optional<int>(3000007) : std::nullopt;
        });
    EXPECT_FALSE(resolve(2).has_value());
    EXPECT_FALSE(s.system->isGeneratedEmbeddedInterfaceMarkerRegistered(3000007));
    const auto generated = resolve(1);
    ASSERT_TRUE(generated.has_value());
    EXPECT_EQ(*generated, 3000007);
    EXPECT_TRUE(s.system->isGeneratedEmbeddedInterfaceMarkerRegistered(3000007));
}

TEST(LevelSetGeometry, ActiveBoundaryResolverOwnsItsInputsUntilExplicitRegistration)
{
    std::function<std::optional<int>(int)> resolve;
    int expected_marker = -1;
    {
        GeometrySystem source;
        const auto geometry = LevelSetGeometry::bind(*source.system,
            {.level_set_field = source.phi, .domain_id = "owned", .interface_marker = 17});
        std::vector<CutSide> owners{geometry.side(CutVolumeSide::Negative)};
        expected_marker = owners.front().boundary(4).generatedActiveBoundaryMarker();
        resolve = generatedActiveBoundaryResolver(std::move(owners), "Owner", "negative marker");
    }
    auto moved = std::move(resolve);
    EXPECT_EQ(moved(4), expected_marker);
    try {
        (void)moved(-1);
        FAIL() << "negative marker was accepted with an owner";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()), "negative marker");
    }

    GeometrySystem consumer;
    const auto wetted = wettedBoundaryResolver(*consumer.system, std::move(moved));
    EXPECT_FALSE(consumer.system->isGeneratedEmbeddedInterfaceMarkerRegistered(expected_marker));
    EXPECT_EQ(wetted(4), expected_marker);
    EXPECT_TRUE(consumer.system->isGeneratedEmbeddedInterfaceMarkerRegistered(expected_marker));
}

TEST(LevelSetGeometry, ActiveBoundaryResolverAllowsEmptyAndRepeatedOwnersButRejectsAmbiguity)
{
    const auto empty = generatedActiveBoundaryResolver({}, "Owner", "negative marker");
    EXPECT_FALSE(empty(-1));
    EXPECT_FALSE(empty(4));

    GeometrySystem source;
    const auto geometry = LevelSetGeometry::bind(*source.system,
        {.level_set_field = source.phi, .domain_id = "shared", .interface_marker = 17});
    const auto negative = geometry.side(CutVolumeSide::Negative);
    const auto repeated = generatedActiveBoundaryResolver({negative, negative}, "Owner", "negative marker");
    EXPECT_EQ(repeated(4), negative.boundary(4).generatedActiveBoundaryMarker());
    const auto ambiguous = generatedActiveBoundaryResolver(
        {negative, geometry.side(CutVolumeSide::Positive)}, "Owner", "negative marker");
    try {
        (void)ambiguous(4);
        FAIL() << "ambiguous owners were accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(std::string(error.what()),
            "Owner: physical boundary marker 4 is covered by multiple ambiguous unfitted active "
            "domains; select one explicit active-domain owner");
    }
}

#if SVMP_FE_WITH_MESH && FE_HAS_EIGEN

// ---------------------------------------------------------------------------
// A standalone cut geometry: an unfitted Poisson problem with a Robin
// condition on a circle or sphere, written with cutGeometry, activeOn and the
// typed measures only (design A, section 2.6).
// ---------------------------------------------------------------------------

constexpr Real kPi = 3.14159265358979323846;

/// The unit square (Triangle3, two per square; Quad4) or the unit cube
/// (Tetra4, six per cube around its main diagonal; Hex8), n cells per side.
std::shared_ptr<svmp::Mesh> unitBoxMesh(svmp::CellFamily family, int n)
{
    const int dim = family == svmp::CellFamily::Triangle || family == svmp::CellFamily::Quad ? 2 : 3;
    const int nz = dim == 3 ? n : 0;
    const auto id = [n](int i, int j, int k) {
        return static_cast<svmp::index_t>((k * (n + 1) + j) * (n + 1) + i);
    };
    std::vector<svmp::real_t> x;
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= n; ++i) {
                x.push_back(static_cast<svmp::real_t>(i) / n);
                x.push_back(static_cast<svmp::real_t>(j) / n);
                if (dim == 3) {
                    x.push_back(static_cast<svmp::real_t>(k) / n);
                }
            }
        }
    }
    svmp::CellShape shape{};
    shape.family = family;
    shape.num_corners = family == svmp::CellFamily::Triangle ? 3
                        : family == svmp::CellFamily::Hex    ? 8
                                                             : 4;
    shape.order = 1;
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> connectivity;
    const auto add = [&](std::initializer_list<svmp::index_t> cell) {
        connectivity.insert(connectivity.end(), cell);
        offsets.push_back(static_cast<svmp::offset_t>(connectivity.size()));
    };
    for (int k = 0; k < std::max(nz, 1); ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const auto c = [&](int a, int b, int d) { return id(i + a, j + b, dim == 3 ? k + d : 0); };
                if (family == svmp::CellFamily::Triangle) {
                    add({c(0, 0, 0), c(1, 0, 0), c(1, 1, 0)});
                    add({c(0, 0, 0), c(1, 1, 0), c(0, 1, 0)});
                } else if (family == svmp::CellFamily::Quad) {
                    add({c(0, 0, 0), c(1, 0, 0), c(1, 1, 0), c(0, 1, 0)});
                } else if (family == svmp::CellFamily::Hex) {
                    add({c(0, 0, 0), c(1, 0, 0), c(1, 1, 0), c(0, 1, 0),
                         c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1)});
                } else {
                    // The six axis paths from (0,0,0) to (1,1,1); a path with
                    // a negative volume swaps its first two steps.
                    constexpr std::array<std::array<int, 3>, 6> paths{
                        {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {1, 0, 2}, {0, 2, 1}, {2, 1, 0}}};
                    for (std::size_t p = 0; p < paths.size(); ++p) {
                        std::array<std::array<int, 3>, 4> v{};
                        for (std::size_t s_ = 1; s_ < 4u; ++s_) {
                            v[s_] = v[s_ - 1];
                            v[s_][static_cast<std::size_t>(paths[p][s_ - 1])] = 1;
                        }
                        if (p >= 3u) {
                            std::swap(v[1], v[2]);
                        }
                        add({c(v[0][0], v[0][1], v[0][2]), c(v[1][0], v[1][1], v[1][2]),
                             c(v[2][0], v[2][1], v[2][2]), c(v[3][0], v[3][1], v[3][2])});
                    }
                }
            }
        }
    }
    auto base = std::make_shared<svmp::MeshBase>();
    base->build_from_arrays(dim, x, offsets, connectivity,
                            std::vector<svmp::CellShape>(offsets.size() - 1u, shape));
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

ElementType linearElement(svmp::CellFamily family)
{
    switch (family) {
    case svmp::CellFamily::Triangle:
        return ElementType::Triangle3;
    case svmp::CellFamily::Quad:
        return ElementType::Quad4;
    case svmp::CellFamily::Tetra:
        return ElementType::Tetra4;
    default:
        return ElementType::Hex8;
    }
}

/// The inclusion {phi < 0}: a circle or sphere of radius 0.31 about a point
/// off the grid, or (hexahedra) the half-cube below a tilted plane.  The cut
/// backends linearize a curved interface in a hexahedron, and the linearized
/// facets are off the trilinear zero set, so hexahedra take a planar cut.
constexpr Real kInclusionRadius = 0.31;

Real inclusionLevelSet(bool planar, int dim, const std::array<Real, 3>& x)
{
    if (planar) {
        return 0.2 * x[0] + 0.1 * x[1] + x[2] - 0.55;
    }
    const std::array<Real, 3> c =
        dim == 2 ? std::array<Real, 3>{0.5013, 0.4987, 0.0} : std::array<Real, 3>{0.5031, 0.4969, 0.5017};
    Real r2 = 0.0;
    for (int d = 0; d < dim; ++d) {
        const auto dx = x[static_cast<std::size_t>(d)] - c[static_cast<std::size_t>(d)];
        r2 += dx * dx;
    }
    return std::sqrt(r2) - kInclusionRadius;
}

Real inclusionMeasure(bool planar, int dim)
{
    return planar ? 0.40
                  : (dim == 2 ? kPi * kInclusionRadius * kInclusionRadius
                              : 4.0 / 3.0 * kPi * kInclusionRadius * kInclusionRadius * kInclusionRadius);
}

/// The exact solution cos(pi x) cos(pi y) cos(pi z), whose flux through the
/// faces of the unit box vanishes, and its gradient.
Real exactT(Real x, Real y, Real z) { return std::cos(kPi * x) * std::cos(kPi * y) * std::cos(kPi * z); }
std::array<Real, 3> exactGradT(Real x, Real y, Real z)
{
    return {-kPi * std::sin(kPi * x) * std::cos(kPi * y) * std::cos(kPi * z),
            -kPi * std::cos(kPi * x) * std::sin(kPi * y) * std::cos(kPi * z),
            -kPi * std::cos(kPi * x) * std::cos(kPi * y) * std::sin(kPi * z)};
}

struct CutPoisson {
    std::shared_ptr<svmp::Mesh> mesh;
    std::unique_ptr<systems::FESystem> system;
    std::shared_ptr<spaces::H1Space> space;
    std::optional<CutGeometry> geometry;
    FieldId phi{INVALID_FIELD_ID};
    FieldId T{INVALID_FIELD_ID};
};

FormExpr scalarCoefficient(std::string name, std::function<Real(Real, Real, Real)> f)
{
    return FormExpr::coefficient(std::move(name), forms::ScalarCoefficient(std::move(f)));
}

/// -k Laplace(T) = f inside the inclusion L = {phi < 0}, k dT/dn + alpha (T - T_inf) = g on its
/// interface (and no flux through the faces of the box), with f and g from the exact solution and
/// the normal of the generated interface, so that T is the exact solution on the cut domain.
CutPoisson buildCutPoisson(svmp::CellFamily family, int n, bool aggregated)
{
    const bool planar = family == svmp::CellFamily::Hex;
    CutPoisson out;
    out.mesh = unitBoxMesh(family, n);
    const int dim = out.mesh->dim();
    out.space = std::make_shared<spaces::H1Space>(linearElement(family), 1);
    out.system = std::make_unique<systems::FESystem>(out.mesh);
    auto& system = *out.system;
    out.phi = system.addField(systems::FieldSpec{
        .name = "phi", .space = out.space, .source_kind = systems::FieldSourceKind::PrescribedData});
    out.T = system.addField(systems::FieldSpec{.name = "T", .space = out.space});
    system.addOperator("op");

    // Linear corners cut simplices and quadrilaterals; hexahedra take the
    // implicit backend, whose cut of a trilinear level set is linearized.
    CutGeometryPolicy policy;
    if (family == svmp::CellFamily::Hex) {
        policy.geometry_mode = GeneratedInterfaceGeometryMode::HighOrderImplicit;
        policy.implicit_cut_backend = ImplicitCutQuadratureBackend::SayeHyperrectangle;
        policy.implicit_cut_fallback_policy = ImplicitCutFallbackPolicy::LinearCorner;
    }
    out.geometry.emplace(
        cutGeometry(system, *out.mesh, {.level_set_field_name = "phi", .domain_id = "inclusion"}, policy));
    const auto L = out.geometry->side(CutVolumeSide::Negative);
    const auto n_out = L.outwardNormal();

    const auto T = forms::StateField(out.T, *out.space, "T");
    const auto s_ = forms::TestField(out.T, *out.space, "s");
    const auto k = FormExpr::constant(2.0);
    const auto alpha = FormExpr::constant(3.0);
    const auto T_inf = FormExpr::constant(0.25);
    const auto T_exact = scalarCoefficient("T_exact", exactT);
    std::vector<FormExpr> grad_components;
    for (int d = 0; d < dim; ++d) {
        grad_components.push_back(scalarCoefficient(
            "dT_exact_" + std::to_string(d),
            [d](Real x, Real y, Real z) { return exactGradT(x, y, z)[static_cast<std::size_t>(d)]; }));
    }
    const auto grad_T_exact = FormExpr::asVector(grad_components);
    const auto f = scalarCoefficient("f", [dim](Real x, Real y, Real z) {
        return 2.0 * static_cast<Real>(dim) * kPi * kPi * exactT(x, y, z);
    });
    const auto g = k * forms::inner(grad_T_exact, n_out) + alpha * (T_exact - T_inf);
    const auto residual = (k * forms::inner(forms::grad(T), forms::grad(s_)) - f * s_).dx(L.volume()) +
                          ((alpha * (T - T_inf) - g) * s_).dI(L.interface());
    (void)systems::installFormulation(system, "op", {out.T}, residual);

    constraints::ConstraintPlan plan(system);
    plan.activeOn({out.T}, L, {.aggregate = aggregated});
    plan.commit();
    system.setup();

    std::vector<Real> phi_at_vertices(out.mesh->n_vertices());
    for (std::size_t v = 0; v < phi_at_vertices.size(); ++v) {
        std::array<Real, 3> x{0.0, 0.0, 0.0};
        for (int d = 0; d < dim; ++d) {
            x[static_cast<std::size_t>(d)] = static_cast<Real>(
                out.mesh->X_ref()[v * static_cast<std::size_t>(dim) + static_cast<std::size_t>(d)]);
        }
        phi_at_vertices[v] = inclusionLevelSet(planar, dim, x);
    }
    std::vector<Real> coefficients(static_cast<std::size_t>(system.fieldDofHandler(out.phi).getNumDofs()), 0.0);
    const auto projection =
        system.projectMeshVertexValuesToFieldCoefficients(out.phi, phi_at_vertices, 1u, coefficients);
    EXPECT_EQ(projection.unassigned_dofs, 0u);
    system.setPrescribedFieldCoefficients(out.phi, coefficients);
    (void)out.geometry->refresh();
    return out;
}

/// One Newton step from the constraint values (the problem is linear).
std::vector<Real> solveCutPoisson(systems::FESystem& system)
{
    const auto n = system.dofHandler().getNumDofs();
    std::vector<Real> state(static_cast<std::size_t>(n), 0.0);
    system.constraints().distribute(state);
    assembly::DenseSystemView out(n);
    out.zero();
    systems::SystemStateView view;
    view.u = state;
    systems::AssemblyRequest request;
    request.op = "op";
    request.want_matrix = true;
    request.want_vector = true;
    request.suppress_constraint_inhomogeneity = true;
    EXPECT_TRUE(system.assemble(request, view, &out, &out).success);
    std::vector<Eigen::Triplet<Real>> entries;
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            const auto v = out.matrixData()[static_cast<std::size_t>(i * n + j)];
            if (v != Real(0.0)) {
                entries.emplace_back(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j), v);
            }
        }
    }
    Eigen::SparseMatrix<Real> A(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    A.setFromTriplets(entries.begin(), entries.end());
    Eigen::SparseLU<Eigen::SparseMatrix<Real>> lu;
    lu.compute(A);
    EXPECT_EQ(lu.info(), Eigen::Success);
    Eigen::Matrix<Real, Eigen::Dynamic, 1> rhs(static_cast<Eigen::Index>(n));
    for (GlobalIndex i = 0; i < n; ++i) {
        rhs[static_cast<Eigen::Index>(i)] = -out.vectorData()[static_cast<std::size_t>(i)];
    }
    const Eigen::Matrix<Real, Eigen::Dynamic, 1> step = lu.solve(rhs);
    for (GlobalIndex i = 0; i < n; ++i) {
        state[static_cast<std::size_t>(i)] += step[static_cast<Eigen::Index>(i)];
    }
    system.constraints().distribute(state);
    return state;
}

/// The L2 error of T on the inclusion, and the inclusion's measure.
std::array<Real, 2> cutPoissonError(const CutPoisson& problem, const std::vector<Real>& state)
{
    const auto L = problem.geometry->side(CutVolumeSide::Negative);
    const auto e = forms::StateField(problem.T, *problem.space, "T") - scalarCoefficient("T_exact", exactT);
    const std::vector<FormExpr> functionals{(e * e).dx(L.volume()), FormExpr::constant(1.0).dx(L.volume())};
    systems::SystemStateView view;
    view.u = state;
    const auto values =
        systems::evaluateFunctionalForms(*problem.system, functionals, view, systems::CollectiveCommunicator{});
    return {std::sqrt(values[0].value), values[1].value};
}

TEST(CutGeometry, CutPoissonWithARobinInclusionConvergesAtSecondOrder)
{
    struct Case {
        svmp::CellFamily family;
        const char* name;
        std::array<int, 2> n;
    };
    const std::vector<Case> cases{{svmp::CellFamily::Triangle, "Triangle3", {16, 32}},
                                  {svmp::CellFamily::Quad, "Quad4", {16, 32}},
                                  {svmp::CellFamily::Tetra, "Tetra4", {8, 16}},
                                  {svmp::CellFamily::Hex, "Hex8", {6, 12}}};
    for (const auto& [family, name, sizes] : cases) {
        SCOPED_TRACE(name);
        std::array<Real, 2> errors{};
        for (std::size_t m = 0; m < sizes.size(); ++m) {
            // Aggregation on the implicit cut of the hexahedra leaves no
            // degree of freedom of T free (every one becomes a Dirichlet
            // line), so the hexahedra run without it.
            auto problem = buildCutPoisson(family, sizes[m], /*aggregated=*/family != svmp::CellFamily::Hex);
            const auto state = solveCutPoisson(*problem.system);
            const auto [error, measure] = cutPoissonError(problem, state);
            const Real exact_measure = inclusionMeasure(family == svmp::CellFamily::Hex, problem.mesh->dim());
            EXPECT_NEAR(measure, exact_measure, 0.15 * exact_measure);
            errors[m] = error;
            std::cout << "[ cut Poisson " << name << " n=" << sizes[m] << " ] e_T=" << error
                      << " |L|=" << measure << std::endl;
        }
        EXPECT_GT(std::log2(errors[0] / errors[1]), 1.6) << errors[0] << " " << errors[1];
    }
}

// The cut of the trace-multiplier test I3 (a circle of radius 1/2 in the
// 2 x 2 square of Quad4 cells, quadrature order 4) made by cutGeometry
// assembles the matrix of the context the test builds cell by cell with the
// linear cut, to round-off: the generated context integrates the cells away
// from the circle with rules of its own, so their entries differ in the last
// bits as well.
TEST(CutGeometry, CutOfTheI3CircleMatchesTheHandBuiltContext)
{
    constexpr int kMarker = 41;
    const auto phi = [](Real x, Real y) { return std::sqrt((x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0)) - 0.5; };
    const auto assembled = [&](bool by_cut_geometry) {
        auto mesh = test_trace_multiplier::buildQuadGrid(16, 16, 2.0, 2.0, 1, test_trace_multiplier::kWholeBoundary);
        auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
        systems::FESystem system(mesh);
        const auto phi_id = system.addField(
            systems::FieldSpec{.name = "phi", .space = space, .source_kind = systems::FieldSourceKind::PrescribedData});
        const auto u_id = system.addField(systems::FieldSpec{.name = "u", .space = space});
        system.addOperator("op");
        std::optional<CutGeometry> G;
        if (by_cut_geometry) {
            G.emplace(cutGeometry(system, *mesh,
                                  {.level_set_field_name = "phi", .domain_id = "i3", .interface_marker = kMarker},
                                  {.quadrature_order = 4, .interface_quadrature_order = 4, .volume_quadrature_order = 4}));
        } else {
            system.setCutIntegrationContext(test_trace_multiplier::generatedInterfaceContext(*mesh, kMarker, phi, 1u, 4));
        }
        const auto inside = forms::VolumeDomain::cut(kMarker, CutVolumeSide::Negative);
        const auto outside = forms::VolumeDomain::cut(kMarker, CutVolumeSide::Positive);
        if (G) {
            EXPECT_EQ(G->side(CutVolumeSide::Negative).volume(), inside);
            EXPECT_EQ(G->side(CutVolumeSide::Positive).volume(), outside);
        }
        const auto u = forms::StateField(u_id, *space, "u");
        const auto v = forms::TestField(u_id, *space, "v");
        (void)systems::installFormulation(
            system, "op", {u_id},
            (u * v).dx(inside) + (FormExpr::constant(2.0) * forms::inner(forms::grad(u), forms::grad(v))).dx(outside) +
                (FormExpr::constant(3.0) * u * v).dI(forms::InterfaceDomain(kMarker)));
        system.setup();
        std::vector<Real> values(mesh->n_vertices());
        for (std::size_t vtx = 0; vtx < values.size(); ++vtx) {
            const auto x = mesh->local_mesh().get_vertex_coords(static_cast<svmp::index_t>(vtx));
            values[vtx] = phi(static_cast<Real>(x[0]), static_cast<Real>(x[1]));
        }
        std::vector<Real> coefficients(static_cast<std::size_t>(system.fieldDofHandler(phi_id).getNumDofs()), 0.0);
        (void)system.projectMeshVertexValuesToFieldCoefficients(phi_id, values, 1u, coefficients);
        system.setPrescribedFieldCoefficients(phi_id, coefficients);
        if (G) {
            (void)G->refresh();
        }
        const auto n = system.dofHandler().getNumDofs();
        assembly::DenseSystemView out(n);
        out.zero();
        const std::vector<Real> zero(static_cast<std::size_t>(n), 0.0);
        systems::SystemStateView view;
        view.u = zero;
        systems::AssemblyRequest request;
        request.op = "op";
        request.want_matrix = true;
        EXPECT_TRUE(system.assemble(request, view, &out, nullptr).success);
        return std::vector<Real>(out.matrixData().begin(), out.matrixData().end());
    };
    const auto by_hand = assembled(false);
    const auto by_cut_geometry = assembled(true);
    ASSERT_EQ(by_hand.size(), by_cut_geometry.size());
    std::size_t nonzero = 0u;
    Real scale = 0.0;
    Real difference = 0.0;
    for (std::size_t i = 0; i < by_hand.size(); ++i) {
        nonzero += by_hand[i] != 0.0 ? 1u : 0u;
        scale = std::max(scale, std::abs(by_hand[i]));
        difference = std::max(difference, std::abs(by_hand[i] - by_cut_geometry[i]));
    }
    EXPECT_GT(nonzero, 0u);
    EXPECT_LT(difference, 1e-13 * scale);
}

// ---------------------------------------------------------------------------
// The cut-structure key of HighOrderImplicit geometry
// ---------------------------------------------------------------------------

struct CutKey {
    std::uint64_t key{0};
    std::size_t interface_fragments{0};
};

/// 8 x 8 Quad9 cells of the unit square without boundary labels.
std::shared_ptr<svmp::Mesh> unlabeledQuad9Mesh()
{
    constexpr int n = 8;
    constexpr int m = 2 * n;
    std::vector<svmp::real_t> x;
    for (int j = 0; j <= m; ++j) {
        for (int i = 0; i <= m; ++i) {
            x.push_back(static_cast<svmp::real_t>(i) / m);
            x.push_back(static_cast<svmp::real_t>(j) / m);
        }
    }
    const auto id = [](int i, int j) { return static_cast<svmp::index_t>(j * (m + 1) + i); };
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 2;
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> connectivity;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int p = 2 * i, q = 2 * j;
            connectivity.insert(connectivity.end(),
                                {id(p, q), id(p + 2, q), id(p + 2, q + 2), id(p, q + 2), id(p + 1, q),
                                 id(p + 2, q + 1), id(p + 1, q + 2), id(p, q + 1), id(p + 1, q + 1)});
            offsets.push_back(static_cast<svmp::offset_t>(connectivity.size()));
        }
    }
    auto base = std::make_shared<svmp::MeshBase>();
    base->build_from_arrays(2, x, offsets, connectivity, std::vector<svmp::CellShape>(offsets.size() - 1u, shape));
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

/// A HighOrderImplicit cut (SayeHyperrectangle unless given) of a P2 level set: the system,
/// its geometry and the level-set field, before the level set is set.
struct HighOrderCut {
    std::shared_ptr<svmp::Mesh> mesh;
    std::unique_ptr<systems::FESystem> system;
    std::optional<CutGeometry> geometry;
    FieldId phi{INVALID_FIELD_ID};
};

HighOrderCut highOrderCut(std::shared_ptr<svmp::Mesh> mesh,
                          std::shared_ptr<spaces::H1Space> space,
                          int depth,
                          ImplicitCutQuadratureBackend backend = ImplicitCutQuadratureBackend::SayeHyperrectangle)
{
    HighOrderCut out;
    out.mesh = std::move(mesh);
    out.system = std::make_unique<systems::FESystem>(out.mesh);
    out.phi = out.system->addField(
        systems::FieldSpec{.name = "phi", .space = space, .source_kind = systems::FieldSourceKind::PrescribedData});
    (void)out.system->addField(systems::FieldSpec{.name = "T", .space = space});
    out.geometry.emplace(cutGeometry(*out.system, *out.mesh, {.level_set_field_name = "phi", .domain_id = "wave"},
                                     {.quadrature_order = 2,
                                      .geometry_mode = GeneratedInterfaceGeometryMode::HighOrderImplicit,
                                      .implicit_cut_backend = backend,
                                      .implicit_cut_max_subdivision_depth = depth}));
    return out;
}

/// Sets the level set to @p phi at the mesh vertices, changed by @p edit, and cuts.
ActiveCutContextRefreshReport cutAt(HighOrderCut& cut,
                                    const std::function<Real(Real, Real)>& phi,
                                    const std::function<void(const svmp::Mesh&, std::vector<Real>&)>& edit = {})
{
    std::vector<Real> values(cut.mesh->n_vertices());
    for (std::size_t v = 0; v < values.size(); ++v) {
        const auto x = cut.mesh->local_mesh().get_vertex_coords(static_cast<svmp::index_t>(v));
        values[v] = phi(static_cast<Real>(x[0]), static_cast<Real>(x[1]));
    }
    if (edit) {
        edit(*cut.mesh, values);
    }
    auto& system = *cut.system;
    std::vector<Real> coefficients(static_cast<std::size_t>(system.fieldDofHandler(cut.phi).getNumDofs()), 0.0);
    (void)system.projectMeshVertexValuesToFieldCoefficients(cut.phi, values, 1u, coefficients);
    system.setPrescribedFieldCoefficients(cut.phi, coefficients);
    return cut.geometry->refresh();
}

/// The topology key of the HighOrderImplicit cut of the P2 level set @p phi
/// on 8 x 8 Quad9 cells of the unit square.
CutKey highOrderCutKey(const std::function<Real(Real, Real)>& phi,
                       int depth,
                       const std::function<void(const svmp::Mesh&, std::vector<Real>&)>& edit = {})
{
    auto cut = highOrderCut(unlabeledQuad9Mesh(), test_trace_multiplier::quadSpace(2), depth);
    (void)cut.geometry->side(CutVolumeSide::Negative);
    cut.system->setup();
    const auto report = cutAt(cut, phi, edit);
    return {report.topology_key, report.interface_fragments};
}

Real waveLevelSet(Real x, Real y, Real shift)
{
    return y - 0.43 - shift - 0.05 * std::sin(2.0 * kPi * x);
}

// The key reads the discrete cut structure, not the backend's subdivision: a
// motion that changes no node sign, no cut cell and no wetted face keeps the
// key at every subdivision depth, although it changes the backend's leaves.
// (The depth itself is part of the request, so keys of different depths
// differ.)
TEST(CutStructureKey, DoesNotChangeWithMotionWithinAStructureClass)
{
    for (const int depth : {4, 6, 8}) {
        SCOPED_TRACE("depth " + std::to_string(depth));
        const auto base = highOrderCutKey([](Real x, Real y) { return waveLevelSet(x, y, 0.0); }, depth);
        const auto moved = highOrderCutKey([](Real x, Real y) { return waveLevelSet(x, y, 0.003); }, depth);
        EXPECT_NE(base.key, 0u);
        EXPECT_EQ(moved.key, base.key);
        std::cout << "[ cut-structure key ] depth " << depth << ": interface fragments " << base.interface_fragments
                  << " -> " << moved.interface_fragments << std::endl;
    }
}

// Each change of the discrete structure changes the key: a level-set DOF
// changes sign (here a mid-edge node above the interface dips below it while
// the corners of its edge keep their sign), and cells become cut or uncut.
TEST(CutStructureKey, ChangesWithEveryChangeOfTheDiscreteStructure)
{
    const auto wave = [](Real x, Real y) { return waveLevelSet(x, y, 0.0); };
    const auto base = highOrderCutKey(wave, 6);

    // The mid-edge node at (0.5 + 1/16, 0.75) only: its edge's corners stay
    // positive.
    const auto dipped = highOrderCutKey(wave, 6, [](const svmp::Mesh& mesh, std::vector<Real>& values) {
        std::size_t edited = 0u;
        for (std::size_t v = 0; v < values.size(); ++v) {
            const auto x = mesh.local_mesh().get_vertex_coords(static_cast<svmp::index_t>(v));
            if (std::abs(x[0] - 0.5625) < 1e-12 && std::abs(x[1] - 0.75) < 1e-12) {
                values[v] = -0.01;
                ++edited;
            }
        }
        EXPECT_EQ(edited, 1u);
    });
    EXPECT_NE(dipped.key, base.key);

    // A larger shift moves the interface across nodes and cells.
    const auto shifted = highOrderCutKey([](Real x, Real y) { return waveLevelSet(x, y, 0.07); }, 6);
    EXPECT_NE(shifted.key, base.key);
}

// The sharp active boundary of HighOrderImplicit geometry restricts the P2
// level set to each wall edge: on Quad9 and Triangle6 (a P2 field on
// Triangle3 cells) the wetted part of the left wall ends at the contact point
// of the root of the level set on that edge, and the contact point, with the
// level set's gradient as its normal, passes the snapshot's checks.
TEST(CutStructureKey, HighOrderActiveBoundaryRestrictsTheLevelSetToTheWall)
{
    for (const bool triangles : {false, true}) {
        SCOPED_TRACE(triangles ? "Triangle6" : "Quad9");
        // The wave, and a parabola across the wall; both are linear along the
        // left wall, with roots 0.43 and 0.464 there.
        const std::vector<std::pair<std::function<Real(Real, Real)>, Real>> cases{
            {[](Real x, Real y) { return waveLevelSet(x, y, 0.0); }, 0.43},
            {[](Real x, Real y) { return y - 0.41 - 0.6 * (x - 0.3) * (x - 0.3); }, 0.464}};
        for (const auto& [phi, root] : cases) {
            auto cut = highOrderCut(triangles ? test_trace_multiplier::buildTriangleGrid(8)
                                              : test_trace_multiplier::buildQuadGrid(8, 8, 1.0, 1.0, 2),
                                    triangles ? test_trace_multiplier::triangleSpace(2)
                                              : test_trace_multiplier::quadSpace(2),
                                    6,
                                    triangles ? ImplicitCutQuadratureBackend::HighOrderSubcell
                                              : ImplicitCutQuadratureBackend::SayeHyperrectangle);
            const auto L = cut.geometry->side(CutVolumeSide::Negative);
            const auto wetted = L.wetted(test_trace_multiplier::kLeft);
            const auto contact = L.contactLine(test_trace_multiplier::kLeft);
            cut.system->registerGeneratedEmbeddedInterfaceMarker(wetted.generatedActiveBoundaryMarker());
            cut.system->registerGeneratedEmbeddedInterfaceMarker(contact.marker());
            cut.system->setup();
            (void)cutAt(cut, phi);
            const std::vector<FormExpr> functionals{FormExpr::constant(1.0).ds(wetted),
                                                    FormExpr::constant(1.0).dL(contact)};
            const std::vector<Real> zero(static_cast<std::size_t>(cut.system->dofHandler().getNumDofs()), 0.0);
            systems::SystemStateView view;
            view.u = zero;
            const auto measures =
                systems::evaluateFunctionalForms(*cut.system, functionals, view, systems::CollectiveCommunicator{});
            EXPECT_NEAR(measures[0].value, root, 1e-10);
            EXPECT_NEAR(measures[1].value, 1.0, 1e-12);
        }
    }
}

#endif // SVMP_FE_WITH_MESH && FE_HAS_EIGEN

} // namespace
} // namespace svmp::FE::level_set::test
