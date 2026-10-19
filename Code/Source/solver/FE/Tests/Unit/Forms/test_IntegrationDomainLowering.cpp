/**
 * @file test_IntegrationDomainLowering.cpp
 * @brief The typed measures and helpers build exactly the trees of the untyped calls
 *
 * The free-surface refactor (D42) requires the typed domains and helpers to be
 * sugar only: the expression trees, and hence the IR and the JIT kernels, must
 * equal those written with markers and explicit algebra.  Each test compares
 * the ordered tree hash of the manifest (node types, operand order and
 * association, constant bits, markers, sides) and the printed tree.
 */

#include <gtest/gtest.h>

#include "Forms/CutCellForms.h"
#include "Forms/FormExpr.h"
#include "Forms/IntegrationDomain.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/SystemManifest.h"

#include <memory>
#include <stdexcept>

namespace svmp::FE::forms::test {
namespace {

void expectSameTree(const FormExpr& actual, const FormExpr& expected)
{
    ASSERT_TRUE(actual.isValid());
    ASSERT_TRUE(expected.isValid());
    EXPECT_EQ(systems::formExprTreeHash(actual), systems::formExprTreeHash(expected));
    EXPECT_EQ(actual.toString(), expected.toString());
}

struct Fields {
    std::shared_ptr<spaces::H1Space> scalar{
        std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1)};
    std::shared_ptr<spaces::ProductSpace> vector{
        std::make_shared<spaces::ProductSpace>(scalar, 2)};
    FormExpr u{StateField(FieldId{0}, *vector, "u")};
    FormExpr v{TestField(FieldId{0}, *vector, "v")};
    FormExpr p{StateField(FieldId{1}, *scalar, "p")};
    FormExpr q{TestField(FieldId{1}, *scalar, "q")};
};

TEST(IntegrationDomainLowering, VolumeDomainsLowerToDxAndDCutVolume)
{
    Fields f;
    const auto integrand = f.p * div(f.v);
    expectSameTree(integrand.dx(VolumeDomain::full()), integrand.dx());
    expectSameTree(integrand.dx(VolumeDomain::cut(7, CutVolumeSide::Negative)),
                   integrand.dCutVolume(7, CutVolumeSide::Negative));
    expectSameTree(integrand.dx(VolumeDomain::cut(7, CutVolumeSide::Positive)),
                   integrand.dCutVolume(7, CutVolumeSide::Positive));
    expectSameTree(integrate(integrand, VolumeIntegrationDomain::cutVolume(7, CutVolumeSide::Positive)),
                   integrand.dCutVolume(7, CutVolumeSide::Positive));
    EXPECT_NE(systems::formExprTreeHash(integrand.dx(VolumeDomain::cut(7, CutVolumeSide::Negative))),
              systems::formExprTreeHash(integrand.dx(VolumeDomain::cut(7, CutVolumeSide::Positive))));
    EXPECT_THROW((void)VolumeDomain::cut(-1, CutVolumeSide::Negative), std::invalid_argument);
}

TEST(IntegrationDomainLowering, InterfaceAndContactLineDomainsLowerToDI)
{
    Fields f;
    const auto integrand = inner(FormExpr::normal(), f.v);
    expectSameTree(integrand.dI(InterfaceDomain(1000003)), integrand.dI(1000003));
    expectSameTree(integrand.dL(ContactLineDomain(2000011, 4)), integrand.dI(2000011));
    expectSameTree(dInterfaceBoundary(integrand, ContactLineDomain(2000011, 4)),
                   dInterfaceBoundary(integrand, 2000011));
    EXPECT_THROW((void)InterfaceDomain(-1), std::invalid_argument);
    EXPECT_THROW((void)ContactLineDomain(-1), std::invalid_argument);
}

TEST(IntegrationDomainLowering, BoundaryDomainsLowerToDExteriorBoundary)
{
    Fields f;
    const auto integrand = dot(f.u, f.v);
    const auto full = BoundaryDomain::fullPhysical(3);
    const auto wetted = BoundaryDomain::generatedActiveSubset(3, 3000123);
    expectSameTree(integrand.ds(full), integrand.dExteriorBoundary(full));
    expectSameTree(integrand.ds(wetted), integrand.dExteriorBoundary(wetted));
    EXPECT_NE(systems::formExprTreeHash(integrand.ds(full)),
              systems::formExprTreeHash(integrand.ds(wetted)));
    // The typed full-physical measure is not the legacy marker-only ds(m).
    EXPECT_NE(systems::formExprTreeHash(integrand.ds(full)), systems::formExprTreeHash(integrand.ds(3)));
}

TEST(IntegrationDomainLowering, FacetDomainsLowerToCutAdjacentFacetIntegrals)
{
    Fields f;
    const auto integrand = inner(jump(grad(f.p)), jump(grad(f.q)));
    expectSameTree(integrand.dS(FacetDomain::cutAdjacent(1000003, CutVolumeSide::Negative)),
                   cutAdjacentFacetIntegral(integrand, 1000003, CutVolumeSide::Negative));
    expectSameTree(integrand.dS(FacetDomain::cutAdjacent(1000003)),
                   cutAdjacentFacetIntegral(integrand, 1000003));
    EXPECT_NE(systems::formExprTreeHash(integrand.dS(FacetDomain::cutAdjacent(1000003))),
              systems::formExprTreeHash(
                  integrand.dS(FacetDomain::cutAdjacent(1000003, CutVolumeSide::Positive))));
    EXPECT_THROW((void)FacetDomain::cutAdjacent(-1), std::invalid_argument);
}

TEST(IntegrationDomainLowering, ProjectorAndConormalExpandToTheNavierStokesTrees)
{
    Fields f;
    const auto n = FormExpr::normal();
    const auto wall_n = FormExpr::asVector({FormExpr::constant(0.0), FormExpr::constant(-1.0)});
    // gamma (I - n (x) n) : grad(v), as in the surface-stress term.
    expectSameTree(tangentialProjector(n, 2), FormExpr::identity(2) - outer(n, n));
    // The contact-line footprint direction.
    expectSameTree(conormal(n, wall_n), safeNormalize(n - dot(n, wall_n) * wall_n));
    // Navier-slip tangential traces.
    expectSameTree(tangentialProjection(f.u, wall_n), f.u - dot(f.u, wall_n) * wall_n);
    // The algebraically equal surface divergence is a different tree.
    EXPECT_NE(systems::formExprTreeHash(inner(tangentialProjector(n, 2), grad(f.v))),
              systems::formExprTreeHash(surfaceDivergence(f.v, n)));
}

TEST(IntegrationDomainLowering, TreeHashSeesOperandOrderAndAssociation)
{
    Fields f;
    const auto a = FormExpr::constant(2.0);
    EXPECT_NE(systems::formExprTreeHash(a * f.p), systems::formExprTreeHash(f.p * a));
    EXPECT_NE(systems::formExprTreeHash((a * f.p) * f.q), systems::formExprTreeHash(a * (f.p * f.q)));
    EXPECT_NE(systems::formExprTreeHash(-f.p * div(f.v)), systems::formExprTreeHash(-(f.p * div(f.v))));
}

} // namespace
} // namespace svmp::FE::forms::test
