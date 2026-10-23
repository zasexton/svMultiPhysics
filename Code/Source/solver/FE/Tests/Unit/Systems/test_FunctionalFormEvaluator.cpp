/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/CutIntegrationContext.h"
#include "Assembly/FunctionalAssembler.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/FormExpr.h"
#include "Forms/IntegrationDomain.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/FunctionalFormEvaluator.h"
#include "Systems/FunctionalForms.h"
#include "Systems/SystemCollectives.h"
#include "Systems/SystemState.h"
#include "Tests/Unit/Systems/FunctionalFormEvaluatorFixtures.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace forms = svmp::FE::forms;
namespace systems = svmp::FE::systems;
namespace fixture = svmp::FE::systems::functional_form_test;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;
using forms::FormExpr;

constexpr GlobalIndex kCellCount = 4;
constexpr Real kExtent = 4.0;
constexpr Real kInterfaceX = 1.3; // the level set x - kInterfaceX

/// u = 1 + 2x + 3y on the strip, cut by the vertical line x = kInterfaceX.
struct PlanarCut {
    fixture::QuadStripMeshAccess mesh{0, 1, kCellCount, kExtent};
    svmp::FE::dofs::DofMap dof_map{fixture::quadStripDofMap(mesh)};
    svmp::FE::spaces::H1Space space{svmp::FE::ElementType::Quad4, 1};
    std::vector<Real> u_values{
        fixture::nodalValues(mesh, [](Real x, Real y) { return Real{1.0} + Real{2.0} * x + Real{3.0} * y; })};
    std::shared_ptr<svmp::FE::assembly::CutIntegrationContext> cut_context{
        fixture::quadStripCutContext(mesh, [](Real x, Real) { return x - kInterfaceX; })};
    svmp::FE::assembly::FunctionalAssembler assembler{};

    PlanarCut() { fixture::configureQuadStripAssembler(assembler, mesh, dof_map, space, u_values); }

    [[nodiscard]] FormExpr u() const { return FormExpr::discreteField(fixture::kField, space, "u"); }

    [[nodiscard]] std::vector<systems::FunctionalFormValue> evaluate(const std::vector<FormExpr>& forms)
    {
        return systems::evaluateFunctionalForms(assembler, forms, cut_context.get(), systems::CollectiveCommunicator{});
    }
};

[[nodiscard]] FormExpr one() { return FormExpr::constant(1.0); }
[[nodiscard]] FormExpr coordinateX() { return FormExpr::coordinate().component(0); }
[[nodiscard]] FormExpr coordinateY() { return FormExpr::coordinate().component(1); }
[[nodiscard]] forms::BoundaryDomain wettedWall()
{
    return forms::BoundaryDomain::generatedActiveSubset(fixture::kWallMarker, fixture::kWettedWallMarker);
}
[[nodiscard]] forms::ContactLineDomain contactLine()
{
    return forms::ContactLineDomain(fixture::kContactMarker, fixture::kWallMarker);
}

void expectRejected(PlanarCut& cut,
                    const FormExpr& form,
                    const std::string& reason,
                    const svmp::FE::assembly::CutIntegrationContext* context)
{
    const std::vector<FormExpr> forms{form};
    try {
        (void)systems::evaluateFunctionalForms(cut.assembler, forms, context, systems::CollectiveCommunicator{});
        ADD_FAILURE() << "expected a rejection: " << reason;
    } catch (const svmp::FE::InvalidArgumentException& error) {
        EXPECT_NE(std::string(error.what()).find(reason), std::string::npos) << error.what();
    }
}

} // namespace

TEST(FunctionalFormEvaluator, IntegratesEveryMeasureOfAPlanarCutExactly)
{
    PlanarCut cut;
    const auto u = cut.u();
    const auto x = coordinateX();
    const Real c = kInterfaceX;

    struct Case {
        const char* name;
        FormExpr form;
        Real expected;
    };
    const std::vector<Case> cases{
        {"area", one().dx(), kExtent},
        {"u over the strip", u.dx(), Real{26.0}},
        {"x over the wall", x.ds(fixture::kWallMarker), Real{8.0}},
        {"negative area", one().dx(forms::VolumeDomain::cut(fixture::kInterfaceMarker, forms::CutVolumeSide::Negative)),
         c},
        {"x over the negative side", x.dCutVolume(fixture::kInterfaceMarker, forms::CutVolumeSide::Negative),
         Real{0.5} * c * c},
        {"u over the positive side", u.dCutVolume(fixture::kInterfaceMarker, forms::CutVolumeSide::Positive),
         (kExtent - c) * Real{2.5} + (kExtent * kExtent - c * c)},
        {"interface length", one().dI(forms::InterfaceDomain(fixture::kInterfaceMarker)), Real{1.0}},
        {"u over the interface", u.dI(fixture::kInterfaceMarker), Real{1.0} + Real{2.0} * c + Real{1.5}},
        {"contact count", one().dL(contactLine()), Real{1.0}},
        {"x at the contact point", x.dL(contactLine()), c},
        {"wetted wall length", one().ds(wettedWall()), c},
        {"x over the wetted wall", x.ds(wettedWall()), Real{0.5} * c * c},
    };
    std::vector<FormExpr> forms;
    for (const auto& item : cases) {
        forms.push_back(item.form);
    }

    const auto values = cut.evaluate(forms);
    ASSERT_EQ(values.size(), cases.size());
    for (std::size_t i = 0; i < cases.size(); ++i) {
        SCOPED_TRACE(cases[i].name);
        EXPECT_NEAR(values[i].value, cases[i].expected, Real{1.0e-12} * std::max(Real{1.0}, cases[i].expected));
        // Nonnegative integrands: the absolute mass sums the same terms.
        EXPECT_EQ(values[i].absolute_mass, values[i].value);
    }
}

TEST(FunctionalFormEvaluator, SumsTheSignedTermsOfAFormExactly)
{
    PlanarCut cut;
    const auto u = cut.u();
    const auto x = coordinateX();
    const auto negative = forms::CutVolumeSide::Negative;
    const auto positive = forms::CutVolumeSide::Positive;

    const std::vector<FormExpr> forms{
        u.dI(fixture::kInterfaceMarker),
        u.dI(fixture::kInterfaceMarker) - u.dI(fixture::kInterfaceMarker),
        x.dx(),
        -(x.dx()),
        one().dCutVolume(fixture::kInterfaceMarker, negative) +
            one().dCutVolume(fixture::kInterfaceMarker, positive) - one().dx(),
        (x - FormExpr::constant(2.0)).dx(),
    };
    const auto values = cut.evaluate(forms);
    ASSERT_EQ(values.size(), forms.size());

    // The same terms with both signs cancel exactly, to +0.
    EXPECT_EQ(values[1].value, Real{0.0});
    EXPECT_FALSE(std::signbit(values[1].value));
    EXPECT_EQ(values[1].absolute_mass, Real{2.0} * values[0].value);
    // Negation is exact.
    EXPECT_EQ(values[3].value, -values[2].value);
    EXPECT_EQ(values[3].absolute_mass, values[2].value);
    // The two cut volumes partition the strip.
    EXPECT_NEAR(values[4].value, Real{0.0}, Real{1.0e-13});
    EXPECT_NEAR(values[4].absolute_mass, Real{2.0} * kExtent, Real{1.0e-12});
    // A sign change inside the domain: the mass is the integral of |x - 2|.
    EXPECT_NEAR(values[5].value, Real{0.0}, Real{1.0e-13});
    EXPECT_NEAR(values[5].absolute_mass, Real{4.0}, Real{1.0e-12});
}

TEST(FunctionalFormEvaluator, EvaluatesTheDeclaredFunctionalsOfASystemInOrder)
{
    auto mesh = std::make_shared<fixture::QuadStripMeshAccess>(0, 1, kCellCount, kExtent);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4, 1);
    systems::FESystem sys(mesh);
    const auto u_field = sys.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    const auto u = FormExpr::discreteField(u_field, *space, "u");
    systems::declareFunctionalForm(
        sys, {.name = "energy", .kind = systems::FunctionalKind::Stored, .form = (Real{0.5} * u * u).dx(), .owner = "t"});
    systems::declareFunctionalForm(
        sys, {.name = "wetted_wall", .kind = systems::FunctionalKind::Measure, .form = one().ds(wettedWall()), .owner = "t"});
    systems::declareFunctionalForm(
        sys, {.name = "interface_u", .kind = systems::FunctionalKind::Power, .form = u.dI(fixture::kInterfaceMarker), .owner = "t"});

    svmp::FE::dofs::MeshTopologyInfo topology;
    topology.n_cells = kCellCount;
    topology.n_vertices = mesh->numVertices();
    topology.dim = 2;
    using Offset = decltype(topology.cell2vertex_offsets)::value_type;
    using Vertex = decltype(topology.cell2vertex_data)::value_type;
    using Gid = decltype(topology.vertex_gids)::value_type;
    for (GlobalIndex cell = 0; cell < kCellCount; ++cell) {
        std::vector<GlobalIndex> nodes;
        mesh->getCellNodes(cell, nodes);
        topology.cell2vertex_offsets.push_back(static_cast<Offset>(topology.cell2vertex_data.size()));
        for (const auto node : nodes) {
            topology.cell2vertex_data.push_back(static_cast<Vertex>(node));
        }
        topology.cell_gids.push_back(static_cast<Gid>(mesh->getCellGlobalId(cell)));
        topology.cell_owner_ranks.push_back(0);
    }
    topology.cell2vertex_offsets.push_back(static_cast<Offset>(topology.cell2vertex_data.size()));
    for (GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        topology.vertex_gids.push_back(static_cast<Gid>(vertex));
    }
    systems::SetupInputs inputs;
    inputs.topology_override = topology;
    sys.setup({}, inputs);

    // u = 1 + 2x + 3y at the vertex DOFs.
    std::vector<Real> u_values(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), Real{0.0});
    const auto* vertex_dofs = sys.fieldDofHandler(u_field).getEntityDofMap();
    ASSERT_NE(vertex_dofs, nullptr);
    for (GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        const auto p = mesh->getNodeCoordinates(vertex);
        for (const auto dof : vertex_dofs->getVertexDofs(vertex)) {
            u_values[static_cast<std::size_t>(sys.fieldDofOffset(u_field) + dof)] =
                Real{1.0} + Real{2.0} * p[0] + Real{3.0} * p[1];
        }
    }
    sys.setCutIntegrationContext(
        fixture::quadStripCutContext(*mesh, [](Real x, Real) { return x - kInterfaceX; }));

    systems::SystemStateView state;
    state.u = u_values;
    const auto values = systems::evaluateDeclaredFunctionalForms(sys, state, systems::CollectiveCommunicator{});
    ASSERT_EQ(values.size(), 3u);
    // 0.5 * integral of (1 + 2x + 3y)^2 over [0, 4] x [0, 1].
    EXPECT_NEAR(values[0].value, Real{290.0} / Real{3.0}, Real{1.0e-11});
    EXPECT_NEAR(values[1].value, kInterfaceX, Real{1.0e-12});
    EXPECT_NEAR(values[2].value, Real{1.0} + Real{2.0} * kInterfaceX + Real{1.5}, Real{1.0e-12});

    // The same forms through a standalone assembler give the same values.
    PlanarCut cut;
    const auto standalone = cut.evaluate({(Real{0.5} * cut.u() * cut.u()).dx(),
                                          one().ds(wettedWall()),
                                          cut.u().dI(fixture::kInterfaceMarker)});
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_NEAR(values[i].value, standalone[i].value, Real{1.0e-12} * std::max(Real{1.0}, values[i].value));
    }
}

TEST(FunctionalFormEvaluator, RejectsFormsItCannotEvaluate)
{
    PlanarCut cut;
    const auto u = cut.u();
    const auto* context = cut.cut_context.get();
    expectRejected(cut, FormExpr{}, "the form is empty", context);
    expectRejected(cut, u * u, "not a sum of integrals", context);
    expectRejected(cut, FormExpr::constant(2.0) * u.dx(), "not a sum of integrals", context);
    expectRejected(cut, FormExpr::testFunction(cut.space, "v").dx(), "test or trial function", context);
    expectRejected(cut, u.dS(5), "dS (interior facets) is not evaluated", context);
    expectRejected(cut, u.dI(), "need the marker of their rules", context);
    expectRejected(cut, (u * one().dx()).dx(), "an integrand contains an integral", context);
    expectRejected(cut, one().dCutVolume(999, forms::CutVolumeSide::Negative), "has no cut-volume rules", context);
    expectRejected(cut, one().dI(999), "is not an interface, contact-line or active-boundary marker", context);
    expectRejected(cut, one().dI(fixture::kInterfaceMarker), "needs a cut integration context", nullptr);
    // Without a cut measure no cut context is needed.
    const std::vector<FormExpr> forms{one().dx()};
    EXPECT_NEAR(systems::evaluateFunctionalForms(cut.assembler, forms, nullptr, systems::CollectiveCommunicator{})
                    .front()
                    .value,
                kExtent,
                Real{1.0e-12});
}
