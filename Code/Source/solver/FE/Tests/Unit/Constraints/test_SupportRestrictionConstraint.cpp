/**
 * @file test_SupportRestrictionConstraint.cpp
 * @brief A volume field restricted to a boundary (D54 LM3, T1 trace multipliers)
 *
 * The support of a field restricted to a physical boundary is the set of
 * degrees of freedom on the closure of its faces: exactly the set a strong
 * Dirichlet condition on that boundary constrains.  The restriction pins every
 * other degree of freedom to 0, on quadrilaterals (Q1, Q2), triangles (P1,
 * P2) and hexahedra (Q1).  On a generated interface the support is the cells
 * the interface cuts, and it follows the cut when a new cut integration
 * context is installed, without a new setup.
 */

#include <gtest/gtest.h>

#include "Constraints/AffineConstraints.h"
#include "Constraints/ConstraintPlan.h"
#include "Constraints/SupportRestrictionConstraint.h"
#include "Forms/BoundaryConditions.h"
#include "Forms/FormExpr.h"
#include "Forms/IntegrationDomain.h"
#include "Forms/Vocabulary.h"
#include "Spaces/GlobalSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include "../Systems/TraceMultiplierTestMeshes.h"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace svmp::FE::test_trace_multiplier {
namespace {

using systems::FESystem;
using systems::FieldSpec;

/// A system with one field of @p space on @p mesh and a mass operator.
std::unique_ptr<FESystem> massSystem(std::shared_ptr<svmp::Mesh> mesh,
                                     std::shared_ptr<spaces::H1Space> space,
                                     FieldId& field)
{
    auto sys = std::make_unique<FESystem>(std::move(mesh));
    field = sys->addField(FieldSpec{.name = "lambda", .space = space});
    sys->addOperator("op");
    const auto u = forms::StateField(field, *space, "lambda");
    const auto v = forms::TestField(field, *space, "mu");
    (void)systems::installFormulation(*sys, "op", {field}, (u * v).dx());
    return sys;
}

/// The field-local degrees of freedom the system constrains.
std::set<GlobalIndex> constrainedDofs(const FESystem& sys, FieldId field)
{
    std::set<GlobalIndex> out;
    const auto offset = sys.fieldDofOffset(field);
    for (GlobalIndex d = 0; d < sys.fieldDofHandler(field).getNumDofs(); ++d) {
        if (sys.constraints().isConstrained(offset + d)) {
            out.insert(d);
        }
    }
    return out;
}

/// The degrees of freedom a strong Dirichlet condition on @p marker constrains.
std::set<GlobalIndex> strongDirichletDofs(std::shared_ptr<svmp::Mesh> mesh,
                                          std::shared_ptr<spaces::H1Space> space,
                                          int marker)
{
    FieldId field = INVALID_FIELD_ID;
    auto sys = massSystem(std::move(mesh), std::move(space), field);
    const std::array<forms::bc::StrongDirichlet, 1> bcs{
        forms::bc::strongDirichlet(field, marker, forms::FormExpr::constant(0.0), "lambda")};
    systems::installStrongDirichlet(*sys, std::span<const forms::bc::StrongDirichlet>(bcs));
    sys->setup();
    return constrainedDofs(*sys, field);
}

struct Restricted {
    std::set<GlobalIndex> pinned;
    GlobalIndex n_dofs{0};
};

/// The degrees of freedom the restriction of the field to @p marker pins.
Restricted restrictedTo(std::shared_ptr<svmp::Mesh> mesh, std::shared_ptr<spaces::H1Space> space, int marker)
{
    FieldId field = INVALID_FIELD_ID;
    auto sys = massSystem(std::move(mesh), std::move(space), field);
    constraints::ConstraintPlan plan(*sys);
    plan.restrictSupport({field}, forms::BoundaryDomain::fullPhysical(marker));
    plan.commit();
    sys->setup();
    for (const auto dof : constrainedDofs(*sys, field)) {
        EXPECT_EQ(sys->constraints().getInhomogeneity(sys->fieldDofOffset(field) + dof), 0.0);
    }
    return {constrainedDofs(*sys, field), sys->fieldDofHandler(field).getNumDofs()};
}

/// The restriction pins exactly the complement of the strong-Dirichlet set,
/// which has @p expected_support members.
void expectSupportIsTheClosure(const std::string& what,
                               const std::function<std::shared_ptr<svmp::Mesh>()>& mesh,
                               std::shared_ptr<spaces::H1Space> space,
                               int marker,
                               std::size_t expected_support)
{
    SCOPED_TRACE(what);
    const auto support = strongDirichletDofs(mesh(), space, marker);
    const auto restricted = restrictedTo(mesh(), space, marker);
    EXPECT_EQ(support.size(), expected_support);
    std::set<GlobalIndex> complement;
    for (GlobalIndex d = 0; d < restricted.n_dofs; ++d) {
        if (support.count(d) == 0u) {
            complement.insert(d);
        }
    }
    EXPECT_EQ(restricted.pinned, complement);
}

} // namespace

TEST(SupportRestriction, PinsEverythingOffTheClosureOfTheBoundaryOnQuadrilaterals)
{
    expectSupportIsTheClosure("Q1", [] { return buildQuadGrid(3, 2, 1.0, 1.0, 1); }, quadSpace(1), kLeft, 3u);
    expectSupportIsTheClosure("Q2", [] { return buildQuadGrid(3, 2, 1.0, 1.0, 2); }, quadSpace(2), kLeft, 5u);
    expectSupportIsTheClosure("Q2 bottom", [] { return buildQuadGrid(3, 2, 1.0, 1.0, 2); }, quadSpace(2), kBottom,
                              7u);
}

TEST(SupportRestriction, PinsEverythingOffTheClosureOfTheBoundaryOnTriangles)
{
    expectSupportIsTheClosure("P1", [] { return buildTriangleGrid(3); }, triangleSpace(1), kLeft, 4u);
    expectSupportIsTheClosure("P2", [] { return buildTriangleGrid(3); }, triangleSpace(2), kLeft, 7u);
    expectSupportIsTheClosure("P2 top", [] { return buildTriangleGrid(3); }, triangleSpace(2), kTop, 7u);
}

TEST(SupportRestriction, PinsEverythingOffTheClosureOfTheBoundaryOnHexahedra)
{
    expectSupportIsTheClosure("Q1 x=0", [] { return buildHexGrid(2); }, hexSpace(1), kLeft, 9u);
    expectSupportIsTheClosure("Q1 z=1", [] { return buildHexGrid(2); }, hexSpace(1), kFront, 9u);
}

TEST(SupportRestriction, AnEarlierConstraintKeepsItsDofs)
{
    // An essential condition at x = 1 (value 2) comes before the restriction
    // to x = 0: its DOFs keep the value 2, every other DOF off x = 0 is 0.
    FieldId field = INVALID_FIELD_ID;
    auto sys = massSystem(buildQuadGrid(3, 2, 1.0, 1.0, 1), quadSpace(1), field);
    constraints::ConstraintPlan plan(*sys);
    plan.restrictSupport({field}, forms::BoundaryDomain::fullPhysical(kLeft));
    plan.essentials([&](FESystem& s) {
        const std::array<forms::bc::StrongDirichlet, 1> bcs{
            forms::bc::strongDirichlet(field, kRight, forms::FormExpr::constant(2.0), "lambda")};
        systems::installStrongDirichlet(s, std::span<const forms::bc::StrongDirichlet>(bcs));
    });
    plan.commit();
    sys->setup();

    const auto right = strongDirichletDofs(buildQuadGrid(3, 2, 1.0, 1.0, 1), quadSpace(1), kRight);
    const auto left = strongDirichletDofs(buildQuadGrid(3, 2, 1.0, 1.0, 1), quadSpace(1), kLeft);
    const auto offset = sys->fieldDofOffset(field);
    for (GlobalIndex d = 0; d < sys->fieldDofHandler(field).getNumDofs(); ++d) {
        const auto dof = offset + d;
        if (left.count(d) != 0u) {
            EXPECT_FALSE(sys->constraints().isConstrained(dof)) << d;
        } else if (right.count(d) != 0u) {
            ASSERT_TRUE(sys->constraints().isConstrained(dof)) << d;
            EXPECT_EQ(sys->constraints().getInhomogeneity(dof), 2.0) << d;
        } else {
            ASSERT_TRUE(sys->constraints().isConstrained(dof)) << d;
            EXPECT_EQ(sys->constraints().getInhomogeneity(dof), 0.0) << d;
        }
    }
}

TEST(SupportRestriction, IsDeclaredInTheInactiveSupportPhaseWithItsManifestLine)
{
    FieldId field = INVALID_FIELD_ID;
    auto sys = massSystem(buildQuadGrid(2, 2, 1.0, 1.0, 1), quadSpace(1), field);
    constraints::ConstraintPlan plan(*sys);
    plan.restrictSupport({field}, forms::BoundaryDomain::fullPhysical(kTop));
    EXPECT_EQ(plan.pendingCount(), 1u);
    plan.commitThrough(constraints::ConstraintPhase::Aggregation);
    EXPECT_THROW(plan.restrictSupport({field}, forms::BoundaryDomain::fullPhysical(kLeft)), std::logic_error);

    bool found = false;
    for (const auto& line : sys->constraintManifestLines()) {
        if (line.find("SupportRestrictionConstraint") != std::string::npos) {
            found = true;
            EXPECT_NE(line.find("domain=physical_boundary marker=4"), std::string::npos) << line;
        }
    }
    EXPECT_TRUE(found);
}

TEST(SupportRestriction, RefusesWhatItCannotRestrict)
{
    // A generated active subset is not a support domain.
    EXPECT_THROW(constraints::SupportRestrictionConstraint(
                     0, forms::BoundaryDomain::generatedActiveSubset(kLeft, 7)),
                 std::invalid_argument);

    // A fitted interface needs its InterfaceMesh.
    {
        FieldId field = INVALID_FIELD_ID;
        auto sys = massSystem(buildQuadGrid(2, 2, 1.0, 1.0, 1), quadSpace(1), field);
        constraints::ConstraintPlan plan(*sys);
        plan.restrictSupport({field}, forms::InterfaceDomain(9));
        plan.commit();
        try {
            sys->setup();
            ADD_FAILURE() << "a fitted interface without an InterfaceMesh was accepted";
        } catch (const std::exception& e) {
            EXPECT_NE(std::string(e.what()).find("no InterfaceMesh is registered"), std::string::npos) << e.what();
        }
    }

    // A global field has no trace.
    {
        auto sys = std::make_unique<FESystem>(buildQuadGrid(2, 2, 1.0, 1.0, 1));
        auto h1 = quadSpace(1);
        auto r1 = std::make_shared<spaces::GlobalSpace>(1);
        const auto u = sys->addField(FieldSpec{.name = "u", .space = h1});
        const auto lam = sys->addField(FieldSpec{.name = "lambda", .space = r1});
        sys->addOperator("op");
        const auto us = forms::StateField(u, *h1, "u");
        const auto v = forms::TestField(u, *h1, "v");
        const auto ls = forms::StateField(lam, *r1, "lambda");
        const auto mu = forms::TestField(lam, *r1, "mu");
        (void)systems::installFormulation(*sys, "op", {u, lam}, (us * v + ls * v).dx() + (mu * us).dx());
        constraints::ConstraintPlan plan(*sys);
        plan.restrictSupport({lam}, forms::BoundaryDomain::fullPhysical(kLeft));
        plan.commit();
        try {
            sys->setup();
            ADD_FAILURE() << "a global field was restricted";
        } catch (const std::exception& e) {
            EXPECT_NE(std::string(e.what()).find("must be an H1 (C0) scalar or Product H1 vector field"),
                      std::string::npos)
                << e.what();
        }
    }
}

namespace {

/// The cells of a quad grid that phi(x, y) = x - c cuts strictly: those with x0 < c < x1.
std::set<GlobalIndex> dofsOfCellsCutAt(const FESystem& sys, FieldId field, int nx, Real c)
{
    std::set<GlobalIndex> out;
    const auto& dof_map = sys.fieldDofHandler(field).getDofMap();
    const auto& base = sys.mesh()->local_mesh();
    for (std::size_t cell = 0; cell < base.n_cells(); ++cell) {
        const auto [vertices, count] = base.cell_vertices_span(static_cast<svmp::index_t>(cell));
        Real x0 = 1e30;
        Real x1 = -1e30;
        for (std::size_t i = 0; i < count; ++i) {
            const auto x = base.get_vertex_coords(vertices[i]);
            x0 = std::min(x0, static_cast<Real>(x[0]));
            x1 = std::max(x1, static_cast<Real>(x[0]));
        }
        (void)nx;
        if (x0 < c && c < x1) {
            for (const auto dof : dof_map.getCellDofs(static_cast<GlobalIndex>(cell))) {
                out.insert(dof);
            }
        }
    }
    return out;
}

} // namespace

TEST(SupportRestriction, FollowsTheCutOfAGeneratedInterfaceWithoutANewSetup)
{
    constexpr int marker = 61;
    for (const int order : {1, 2}) {
        SCOPED_TRACE("Q" + std::to_string(order) + " on Quad4 cells");
        // Q2 on the Quad4 cells: the cut cells are the linear cells.
        auto mesh = buildQuadGrid(4, 2, 1.0, 1.0, 1);
        auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, order);
        FieldId field = INVALID_FIELD_ID;
        auto sys = massSystem(mesh, space, field);
        sys->setCutIntegrationContext(
            generatedInterfaceContext(*mesh, marker, [](Real x, Real) { return x - Real{0.3}; }));
        sys->addSystemConstraint(
            std::make_unique<constraints::SupportRestrictionConstraint>(field, marker, std::string("phi")));
        sys->setup();
        const auto n_dofs = sys->dofHandler().getNumDofs();

        const auto expect_support = [&](Real c) {
            const auto support = dofsOfCellsCutAt(*sys, field, 4, c);
            ASSERT_FALSE(support.empty());
            const auto pinned = constrainedDofs(*sys, field);
            for (GlobalIndex d = 0; d < sys->fieldDofHandler(field).getNumDofs(); ++d) {
                EXPECT_EQ(pinned.count(d) == 0u, support.count(d) != 0u) << "x = " << c << " dof " << d;
            }
        };
        expect_support(Real{0.3});

        // A new cut epoch: the interface moves to x = 0.6.
        sys->setCutIntegrationContext(
            generatedInterfaceContext(*mesh, marker, [](Real x, Real) { return x - Real{0.6}; }, 2u));
        sys->rebuildConstraintState();
        EXPECT_EQ(sys->dofHandler().getNumDofs(), n_dofs);
        expect_support(Real{0.6});
    }
}

TEST(SupportRestriction, AGeneratedInterfaceOutsideTheMeshPinsEverything)
{
    constexpr int marker = 62;
    auto mesh = buildQuadGrid(3, 2, 1.0, 1.0, 1);
    FieldId field = INVALID_FIELD_ID;
    auto sys = massSystem(mesh, quadSpace(1), field);
    sys->setCutIntegrationContext(
        generatedInterfaceContext(*mesh, marker, [](Real x, Real) { return x - Real{2.0}; }));
    sys->addSystemConstraint(
        std::make_unique<constraints::SupportRestrictionConstraint>(field, marker, std::string("phi")));
    sys->setup();
    EXPECT_EQ(constrainedDofs(*sys, field).size(),
              static_cast<std::size_t>(sys->fieldDofHandler(field).getNumDofs()));
}

} // namespace svmp::FE::test_trace_multiplier
