/**
 * @file test_VectorBasisConstraints.cpp
 * @brief Unit tests for Systems-side essential constraints on H(curl)/H(div) vector bases
 */

#include <gtest/gtest.h>

#include "Basis/VectorBasis.h"
#include "Elements/ReferenceElement.h"
#include "Constraints/HCurlTangentialConstraint.h"
#include "Constraints/HDivNormalConstraint.h"
#include "Forms/FormCompiler.h"
#include "Forms/StandardBCs.h"
#include "Forms/Vocabulary.h"
#include "Spaces/HCurlSpace.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/TraceSpace.h"
#include "Systems/BoundaryConditionManager.h"
#include "Systems/FESystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <algorithm>
#include <memory>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace systems {
namespace test {

namespace {

dofs::MeshTopologyInfo singleTetraTopology()
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    return topo;
}

[[nodiscard]] std::vector<int> localEdgesOnFace(const elements::ReferenceElement& ref, LocalIndex local_face_id)
{
    std::vector<int> edges;
    const auto& fn = ref.face_nodes(static_cast<std::size_t>(local_face_id));
    if (fn.size() < 2u) {
        return edges;
    }

    auto on_face = [&](LocalIndex v) -> bool {
        for (auto fv : fn) {
            if (fv == v) return true;
        }
        return false;
    };

    edges.reserve(ref.num_edges());
    for (std::size_t e = 0; e < ref.num_edges(); ++e) {
        const auto& en = ref.edge_nodes(e);
        if (en.size() != 2u) continue;
        if (on_face(en[0]) && on_face(en[1])) {
            edges.push_back(static_cast<int>(e));
        }
    }
    return edges;
}

template <class SpaceT>
std::unordered_set<GlobalIndex> expectedBoundaryDofsForMarker(
    const FESystem& sys,
    FieldId field,
    const SpaceT& space,
    int marker)
{
    const auto* vbf = dynamic_cast<const basis::VectorBasisFunction*>(&space.element().basis());
    EXPECT_NE(vbf, nullptr);

    const auto assoc = vbf->dof_associations();
    const auto ref = elements::ReferenceElement::create(ElementType::Tetra4);

    std::unordered_set<GlobalIndex> expected;
    const auto offset = sys.fieldDofOffset(field);
    const auto cell_dofs = sys.fieldDofHandler(field).getDofMap().getCellDofs(0);

    sys.meshAccess().forEachBoundaryFace(marker, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        (void)face_id;
        ASSERT_EQ(cell_id, 0);
        const auto local_face = sys.meshAccess().getLocalFaceIndex(0, 0);
        const auto face_edges = localEdgesOnFace(ref, local_face);

        auto is_face_edge = [&](int edge_id) -> bool {
            for (int e : face_edges) {
                if (e == edge_id) return true;
            }
            return false;
        };

        for (std::size_t ldof = 0; ldof < assoc.size(); ++ldof) {
            const auto& a = assoc[ldof];
            bool constrain = false;
            if (space.continuity() == Continuity::H_curl) {
                if (a.entity_type == basis::DofEntity::Edge) {
                    constrain = is_face_edge(a.entity_id);
                } else if (a.entity_type == basis::DofEntity::Face) {
                    constrain = (a.entity_id == static_cast<int>(local_face));
                }
            } else if (space.continuity() == Continuity::H_div) {
                if (a.entity_type == basis::DofEntity::Face) {
                    constrain = (a.entity_id == static_cast<int>(local_face));
                }
            }
            if (!constrain) continue;
            expected.insert(cell_dofs[ldof] + offset);
        }
    });

    return expected;
}

[[nodiscard]] int localFaceForMarker(const FESystem& sys, int marker)
{
    int local_face = -1;
    int hits = 0;
    sys.meshAccess().forEachBoundaryFace(marker, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        EXPECT_EQ(cell_id, 0);
        local_face = static_cast<int>(sys.meshAccess().getLocalFaceIndex(face_id, cell_id));
        ++hits;
    });
    EXPECT_EQ(hits, 1);
    return local_face;
}

[[nodiscard]] std::vector<Real> constrainedCellCoefficients(const FESystem& sys, FieldId field)
{
    const auto& rec = sys.fieldRecord(field);
    const auto& cell_dofs = sys.fieldDofHandler(field).getDofMap().getCellDofs(0);
    const auto offset = sys.fieldDofOffset(field);

    std::vector<Real> coeffs(rec.space->dofs_per_element(), Real(0));
    EXPECT_EQ(coeffs.size(), cell_dofs.size());
    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
        const auto dof = cell_dofs[i] + offset;
        if (sys.constraints().isConstrained(dof)) {
            coeffs[i] = static_cast<Real>(sys.constraints().getInhomogeneity(dof));
        }
    }
    return coeffs;
}

void expectBoundaryTraceEquals(const FESystem& sys,
                               FieldId field,
                               int marker,
                               Real expected,
                               Real tol = 1e-11)
{
    const auto local_face = localFaceForMarker(sys, marker);
    auto volume_space = std::const_pointer_cast<spaces::FunctionSpace>(sys.fieldRecord(field).space);
    spaces::TraceSpace trace(volume_space, local_face);

    const auto cell_coeffs = constrainedCellCoefficients(sys, field);
    const auto face_coeffs = trace.restrict(cell_coeffs);

    auto quad = trace.element().quadrature();
    ASSERT_TRUE(quad);
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto xi = quad->point(q);
        const auto approx = trace.evaluate_scalar(xi, face_coeffs);
        EXPECT_NEAR(approx, expected, tol);
    }
}

} // namespace

TEST(VectorBasisConstraints, HCurlTangentialConstrainsBoundaryFacetDofs)
{
    const int marker = 7;
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<spaces::HCurlSpace>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto E = sys.addField(FieldSpec{.name = "E", .space = space, .components = space->value_dimension()});
    sys.addSystemConstraint(std::make_unique<constraints::HCurlTangentialConstraint>(E, marker));

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto expected = expectedBoundaryDofsForMarker(sys, E, *space, marker);
    EXPECT_FALSE(expected.empty());

    for (GlobalIndex dof = 0; dof < sys.dofHandler().getNumDofs(); ++dof) {
        const bool should = (expected.find(dof) != expected.end());
        EXPECT_EQ(sys.constraints().isConstrained(dof), should);
    }
}

TEST(VectorBasisConstraints, HDivNormalConstrainsBoundaryFacetDofs)
{
    const int marker = 7;
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<spaces::HDivSpace>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto B = sys.addField(FieldSpec{.name = "B", .space = space, .components = space->value_dimension()});
    sys.addSystemConstraint(std::make_unique<constraints::HDivNormalConstraint>(B, marker));

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto expected = expectedBoundaryDofsForMarker(sys, B, *space, marker);
    EXPECT_FALSE(expected.empty());

    for (GlobalIndex dof = 0; dof < sys.dofHandler().getNumDofs(); ++dof) {
        const bool should = (expected.find(dof) != expected.end());
        EXPECT_EQ(sys.constraints().isConstrained(dof), should);
    }
}

TEST(VectorBasisConstraints, HDivNormalConstraintInterpolatesInhomogeneousBoundaryTraceData)
{
    const int marker = 7;
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<spaces::HDivSpace>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto B = sys.addField(FieldSpec{.name = "B", .space = space, .components = space->value_dimension()});
    sys.addSystemConstraint(
        std::make_unique<constraints::HDivNormalConstraint>(B, marker, forms::FormExpr::constant(2.75)));

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    expectBoundaryTraceEquals(sys, B, marker, /*expected=*/2.75, /*tol=*/1e-10);
}

TEST(VectorBasisConstraints, HDivNormalConstraintUpdatesTimeDependentBoundaryTraceData)
{
    const int marker = 7;
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<spaces::HDivSpace>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto B = sys.addField(FieldSpec{.name = "B", .space = space, .components = space->value_dimension()});
    sys.addSystemConstraint(
        std::make_unique<constraints::HDivNormalConstraint>(B, marker, forms::t() + forms::deltat() + forms::FormExpr::constant(1.0)));

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    expectBoundaryTraceEquals(sys, B, marker, /*expected=*/1.0, /*tol=*/1e-10);

    sys.updateConstraints(/*time=*/2.0, /*dt=*/0.5);
    expectBoundaryTraceEquals(sys, B, marker, /*expected=*/3.5, /*tol=*/1e-10);
}

TEST(VectorBasisConstraints, BoundaryConditionManagerInstallsNormalTraceEssentialBC)
{
    const int marker = 7;
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<spaces::HDivSpace>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto B = sys.addField(FieldSpec{.name = "B", .space = space, .components = space->value_dimension()});

    BoundaryConditionManager bc_manager;
    bc_manager.add(std::make_unique<forms::bc::NormalTraceEssentialBC>(marker, forms::FormExpr::constant(4.0)));
    bc_manager.applyAll(sys, B);

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    expectBoundaryTraceEquals(sys, B, marker, /*expected=*/4.0, /*tol=*/1e-10);
}

TEST(VectorBasisConstraints, BoundaryConditionManagerSupportsMixedStrongAndWeakTraceBCPaths)
{
    const int strong_marker = 7;
    const int weak_marker = 8;
    auto mesh = std::make_shared<forms::test::SingleTetraTwoBoundaryFaceMeshAccess>(
        strong_marker, weak_marker);
    auto space = std::make_shared<spaces::HDivSpace>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto B = sys.addField(FieldSpec{.name = "B", .space = space, .components = space->value_dimension()});

    const auto u = forms::FormExpr::trialFunction(*space, "B");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    auto residual = forms::dot(u, v).dx();

    BoundaryConditionManager bc_manager;
    bc_manager.add(
        std::make_unique<forms::bc::NormalTraceEssentialBC>(strong_marker, forms::FormExpr::constant(1.5)));
    bc_manager.add(
        std::make_unique<forms::bc::TraceLoadBC>(weak_marker, forms::FormExpr::constant(2.0)));
    bc_manager.applyAll(sys, residual, u, v, B);

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto expected = expectedBoundaryDofsForMarker(sys, B, *space, strong_marker);
    EXPECT_FALSE(expected.empty());
    for (GlobalIndex dof = 0; dof < sys.dofHandler().getNumDofs(); ++dof) {
        const bool should = (expected.find(dof) != expected.end());
        EXPECT_EQ(sys.constraints().isConstrained(dof), should);
    }
    expectBoundaryTraceEquals(sys, B, strong_marker, /*expected=*/1.5, /*tol=*/1e-10);

    forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int weak_terms = 0;
    int strong_terms = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != forms::IntegralDomain::Boundary) {
            continue;
        }
        if (term.boundary_marker == weak_marker) {
            ++weak_terms;
        }
        if (term.boundary_marker == strong_marker) {
            ++strong_terms;
        }
    }

    EXPECT_EQ(weak_terms, 1);
    EXPECT_EQ(strong_terms, 0);
}

TEST(VectorBasisConstraints, BoundaryConditionManagerAllowsMultipleWeakTraceConditionsOnSameMarker)
{
    BoundaryConditionManager bc_manager;
    bc_manager.add(
        std::make_unique<forms::bc::TraceLoadBC>(7, forms::FormExpr::constant(2.0)));
    bc_manager.add(
        std::make_unique<forms::bc::TraceRobinBC>(7,
                                                  forms::FormExpr::constant(3.0),
                                                  forms::FormExpr::constant(0.0)));

    EXPECT_NO_THROW(bc_manager.validate());
}

TEST(VectorBasisConstraints, BoundaryConditionManagerRejectsStrongAndWeakTraceConditionsOnSameMarker)
{
    BoundaryConditionManager bc_manager;
    bc_manager.add(
        std::make_unique<forms::bc::NormalTraceEssentialBC>(7, forms::FormExpr::constant(1.0)));
    bc_manager.add(
        std::make_unique<forms::bc::TraceLoadBC>(7, forms::FormExpr::constant(2.0)));

    EXPECT_THROW((void)bc_manager.validate(), std::invalid_argument);
}

TEST(VectorBasisConstraints, BoundaryConditionManagerSeparatesBoundaryAndInterfaceMarkers)
{
    BoundaryConditionManager bc_manager;
    bc_manager.add(
        std::make_unique<forms::bc::TraceLoadBC>(7, forms::FormExpr::constant(2.0)));
    bc_manager.add(
        std::make_unique<forms::bc::InterfaceTraceLoadBC>(7, forms::FormExpr::constant(3.0)));

    EXPECT_NO_THROW(bc_manager.validate());
}

TEST(VectorBasisConstraints, BoundaryConditionManagerInstallsInterfaceTraceMetadataAndResidualTerms)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::HDivSpace>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto B = sys.addField(FieldSpec{.name = "B", .space = space, .components = space->value_dimension()});

    const auto u = forms::FormExpr::trialFunction(*space, "B");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    auto residual = forms::dot(u, v).dx();

    BoundaryConditionManager bc_manager;
    bc_manager.add(
        std::make_unique<forms::bc::InterfaceTraceLoadBC>(9, forms::FormExpr::constant(2.0)));
    bc_manager.add(
        std::make_unique<forms::bc::InterfaceTraceJumpPenaltyBC>(9,
                                                                 forms::FormExpr::constant(5.0),
                                                                 forms::FormExpr::constant(0.0)));
    bc_manager.applyAll(sys, residual, u, v, B);

    ASSERT_EQ(sys.boundaryConditionDescriptors().size(), 2u);
    for (const auto& desc : sys.boundaryConditionDescriptors()) {
        EXPECT_EQ(desc.domain, analysis::DomainKind::InterfaceFace);
        EXPECT_EQ(desc.interface_marker, 9);
    }

    forms::FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);

    int interface_terms = 0;
    for (const auto& term : ir.terms()) {
        if (term.domain != forms::IntegralDomain::InterfaceFace) continue;
        if (term.interface_marker == 9) {
            ++interface_terms;
        }
    }

    EXPECT_EQ(interface_terms, 3);
}

TEST(VectorBasisConstraints, MonolithicDofHandlerPreservesCellOrientationsForVectorBasisFields)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::HCurlSpace>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    (void)sys.addField(FieldSpec{.name = "E", .space = space, .components = space->value_dimension()});
    sys.addOperator("op");

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    EXPECT_TRUE(sys.dofHandler().hasCellOrientations());
}

} // namespace test
} // namespace systems
} // namespace FE
} // namespace svmp
