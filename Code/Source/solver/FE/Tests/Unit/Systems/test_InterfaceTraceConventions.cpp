/**
 * @file test_InterfaceTraceConventions.cpp
 * @brief Sign conventions of the interface trace helpers, checked by assembly
 *
 * Two unit squares share the interface x = 1 (the mesh of
 * test_MortarHybridInterface).  The field is discontinuous P1 (L2 Lagrange),
 * so the basis functions of a cell sum to one and v = 1 on one cell, 0 on the
 * other, is a test function.  Every expected value below is independent of
 * which cell is the minus side of the interface: with n(-) the outward normal
 * of the minus cell and jump(v) = v(-) - v(+), the product n(-) jump(v) does
 * not change when the sides are swapped.
 *
 * - Consistency of interior-penalty (SIPG) interface terms: for u = x
 *   (continuous, so jump(u) = 0) the consistency term
 *   -{k grad u}.n(-) jump(v) integrates, for v = 1 on the left cell, to
 *   -int_Gamma grad(u).n_left = -1 (k = 1, |Gamma| = 1): it cancels the
 *   interface part of the boundary flux of the left cell.
 * - Average of a normal trace: for a constant vector c = (2, 0),
 *   {c}.n(-) jump(v) integrates, for v = 1 on the left cell, to c.n_left = 2.
 *
 * InterfaceFaceGeometry: two-sided face terms on 3D faces against the exact
 * face mass of a continuous first-order field, on a tetrahedral interface
 * face that the mesh and the reference element number differently (faces 2
 * and 3 of Tetra4) and on a quadrilateral face that the two sides' facets
 * run in opposite directions, as an interior and as an interface face.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Forms/BoundaryConditions.h"
#include "Forms/InterfaceConditions.h"
#include "Forms/StandardBCs.h"
#include "Forms/Vocabulary.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::systems::test {
namespace {

constexpr int kInterfaceMarker = 17;

std::shared_ptr<svmp::Mesh> twoUnitSquaresSharingAnInterface()
{
    auto base = std::make_shared<svmp::MeshBase>();
    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0};
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 4, 3, 1, 2, 5, 4};
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape, shape});
    base->finalize();

    auto mesh = svmp::create_mesh(std::move(base));
    auto& mbase = mesh->local_mesh();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mbase.n_faces()); ++f) {
        const auto verts = mbase.face_vertices(f);
        if (verts.size() != 2u) {
            continue;
        }
        const bool has1 = (verts[0] == 1 || verts[1] == 1);
        const bool has4 = (verts[0] == 4 || verts[1] == 4);
        if (has1 && has4) {
            mbase.add_to_set(svmp::EntityKind::Face, "middle", f);
            break;
        }
    }
    return mesh;
}

/// Residual sums over the basis functions of each cell (v = 1 on that cell).
struct CellSums {
    Real left{0.0};    ///< cell 0, x in [0, 1]
    Real right{0.0};   ///< cell 1, x in [1, 2]
};

/**
 * Installs (0 u v) dx + interface(u, v) for a discontinuous P1 field u, sets u
 * to the interpolant of @p u_exact (a function of x), assembles the residual
 * and sums it over the DOFs of each cell.
 */
CellSums assembleCellSums(
    const std::function<forms::FormExpr(const forms::FormExpr&, const forms::FormExpr&)>& interface_terms,
    const std::function<Real(Real)>& u_exact)
{
    using forms::FormExpr;
    auto space = std::make_shared<spaces::L2Space>(ElementType::Quad4, 1);

    FESystem sys(twoUnitSquaresSharingAnInterface());
    sys.setInterfaceMeshFromFaceSet(kInterfaceMarker, "middle");
    const FieldId u_id = sys.addField(FieldSpec{.name = "u", .space = space});
    sys.addOperator("op");

    const auto u = forms::StateField(u_id, *space, "u");
    const auto v = forms::TestField(u_id, *space, "v");
    const auto residual = (FormExpr::constant(0.0) * u * v).dx() + interface_terms(u, v);

    FormInstallOptions options;
    options.compiler_options.jit.enable = false;
    (void)installFormulation(sys, "op", {u_id}, residual, options);
    sys.setup();

    const auto n_dofs = sys.dofHandler().getNumDofs();
    const auto offset = sys.fieldDofOffset(u_id);
    std::vector<Real> state_values(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex cell = 0; cell < 2; ++cell) {
        // Cell c maps the reference square [-1, 1]^2 onto [c, c + 1] x [0, 1]
        // with vertex 0 at (c, 0): x = c + (xi + 1) / 2.
        std::vector<Real> local;
        space->interpolate_scalar(
            [&](const spaces::FunctionSpace::Value& xi) {
                return u_exact(static_cast<Real>(cell) + Real(0.5) * (xi[0] + Real(1.0)));
            },
            local);
        const auto dofs = sys.fieldDofHandler(u_id).getCellDofs(cell);
        EXPECT_EQ(dofs.size(), local.size());
        for (std::size_t i = 0; i < dofs.size() && i < local.size(); ++i) {
            state_values[static_cast<std::size_t>(offset + dofs[i])] = local[i];
        }
    }

    SystemStateView state;
    state.u = state_values;
    AssemblyRequest request;
    request.op = "op";
    request.want_vector = true;
    assembly::DenseVectorView R(n_dofs);
    R.zero();
    const auto result = sys.assemble(request, state, nullptr, &R);
    EXPECT_TRUE(result.success);

    CellSums sums;
    for (GlobalIndex cell = 0; cell < 2; ++cell) {
        Real sum = 0.0;
        for (const auto dof : sys.fieldDofHandler(u_id).getCellDofs(cell)) {
            sum += R.getVectorEntry(offset + dof);
        }
        (cell == 0 ? sums.left : sums.right) = sum;
    }
    return sums;
}

Real linearX(Real x) { return x; }

} // namespace

TEST(InterfaceTraceConventions, CoreInterfacePenaltyIntegrandIsConsistentForALinearSolution)
{
    // The control: the integrand written with the core jump() and avg().
    const auto sums = assembleCellSums(
        [](const forms::FormExpr& u, const forms::FormExpr& v) {
            return forms::interface::nitschePoissonIntegrand(forms::FormExpr::constant(1.0), u, v)
                .dI(kInterfaceMarker);
        },
        linearX);
    EXPECT_NEAR(sums.left, -1.0, 1e-12);
    EXPECT_NEAR(sums.right, 1.0, 1e-12);
}

TEST(InterfaceTraceConventions, InterfaceNitschePoissonHelperIsConsistentForALinearSolution)
{
    // applyNitscheInterfacePoisson must state the same consistency term.
    const auto sums = assembleCellSums(
        [](const forms::FormExpr& u, const forms::FormExpr& v) {
            return forms::interface::applyNitscheInterfacePoisson(
                (forms::FormExpr::constant(0.0) * u * v).dx(), forms::FormExpr::constant(1.0), u, v,
                kInterfaceMarker);
        },
        linearX);
    EXPECT_NEAR(sums.left, -1.0, 1e-12);
    EXPECT_NEAR(sums.right, 1.0, 1e-12);
}

TEST(InterfaceTraceConventions, AverageOfTheNormalTraceIsTheAverageFlux)
{
    // {c}.n(-) jump(v) for c = (2, 0): +2 on the left cell, -2 on the right.
    const auto sums = assembleCellSums(
        [](const forms::FormExpr& /*u*/, const forms::FormExpr& v) {
            const auto c = forms::as_vector({forms::FormExpr::constant(2.0), forms::FormExpr::constant(0.0)});
            return (forms::bc::applyInterfaceScalarTrace(c, forms::bc::ScalarTraceOperator::NormalComponent,
                                                         forms::bc::InterfaceTraceReduction::Average) *
                    forms::jump(v))
                .dI(kInterfaceMarker);
        },
        linearX);
    EXPECT_NEAR(sums.left, 2.0, 1e-12);
    EXPECT_NEAR(sums.right, -2.0, 1e-12);
}

TEST(InterfaceTraceConventions, NormalComponentJumpIsTheCoreJumpAlongTheMinusNormal)
{
    // Holds before and after the conventions were aligned: the normal
    // reduction u(+).n(+) + u(-).n(-) equals jump(u).n(-).  With u the
    // indicator of the left cell and c = (1, 0) the vector field u c, the
    // jump term tested with the plus-side trace of v sums to the same value
    // as the core form on each cell.
    const auto with_helper = assembleCellSums(
        [](const forms::FormExpr& u, const forms::FormExpr& v) {
            const auto c = forms::as_vector({forms::FormExpr::constant(1.0), forms::FormExpr::constant(0.0)});
            return (forms::bc::applyInterfaceScalarTrace(u * c, forms::bc::ScalarTraceOperator::NormalComponent,
                                                         forms::bc::InterfaceTraceReduction::Jump) *
                    forms::jump(v))
                .dI(kInterfaceMarker);
        },
        [](Real x) { return x < 1.0 ? Real(1.0) : Real(0.0); });
    const auto with_core = assembleCellSums(
        [](const forms::FormExpr& u, const forms::FormExpr& v) {
            const auto c = forms::as_vector({forms::FormExpr::constant(1.0), forms::FormExpr::constant(0.0)});
            return (forms::inner(forms::jump(u * c), forms::FormExpr::normal().minus()) * forms::jump(v))
                .dI(kInterfaceMarker);
        },
        [](Real x) { return x < 1.0 ? Real(1.0) : Real(0.0); });
    EXPECT_NEAR(with_helper.left, with_core.left, 1e-12);
    EXPECT_NEAR(with_helper.right, with_core.right, 1e-12);
    EXPECT_GT(std::abs(with_core.left), 0.5);
}

TEST(InterfaceTraceConventions, IdentityJumpReductionIsTheCoreJump)
{
    // One jump convention: the Identity Jump reduction of the BC helpers is
    // jump(u) = u(-) - u(+), by tree and by value.
    spaces::L2Space space(ElementType::Quad4, 1);
    const auto u = forms::StateField(FieldId{0}, space, "u");
    EXPECT_EQ(forms::bc::applyInterfaceScalarTrace(u, forms::bc::ScalarTraceOperator::Identity,
                                                   forms::bc::InterfaceTraceReduction::Jump)
                  .toString(),
              (u.minus() - u.plus()).toString());

    const auto indicator = [](Real x) { return x < 1.0 ? Real(1.0) : Real(0.0); };
    const auto with_helper = assembleCellSums(
        [](const forms::FormExpr& w, const forms::FormExpr& v) {
            return (forms::bc::applyInterfaceScalarTrace(w, forms::bc::ScalarTraceOperator::Identity,
                                                         forms::bc::InterfaceTraceReduction::Jump) *
                    forms::bc::applyInterfaceScalarTrace(v, forms::bc::ScalarTraceOperator::Identity,
                                                         forms::bc::InterfaceTraceReduction::Jump))
                .dI(kInterfaceMarker);
        },
        indicator);
    const auto with_core = assembleCellSums(
        [](const forms::FormExpr& w, const forms::FormExpr& v) {
            return (forms::jump(w) * forms::jump(v)).dI(kInterfaceMarker);
        },
        indicator);
    EXPECT_NEAR(with_helper.left, with_core.left, 1e-12);
    EXPECT_NEAR(with_helper.right, with_core.right, 1e-12);
    // jump(u) jump(v) with u the left indicator: +1 on the left, -1 on the right.
    EXPECT_NEAR(with_core.left, 1.0, 1e-12);
    EXPECT_NEAR(with_core.right, -1.0, 1e-12);

    // The jump-penalty wrapper with a prescribed jump g states
    // alpha jump(u) jump(v) - g jump(v), the same form as the core.
    const auto penalty_wrapper = assembleCellSums(
        [](const forms::FormExpr& w, const forms::FormExpr& v) {
            forms::FormExpr residual = (forms::FormExpr::constant(0.0) * w * v).dx();
            forms::bc::InterfaceTraceJumpPenaltyBC bc(kInterfaceMarker, forms::FormExpr::constant(3.0),
                                                      forms::FormExpr::constant(2.0),
                                                      forms::bc::ScalarTraceOperator::Identity);
            bc.contributeToResidual(residual, w, v);
            return residual;
        },
        indicator);
    const auto penalty_core = assembleCellSums(
        [](const forms::FormExpr& w, const forms::FormExpr& v) {
            return (3.0 * forms::jump(w) * forms::jump(v)).dI(kInterfaceMarker) -
                   (2.0 * forms::jump(v)).dI(kInterfaceMarker);
        },
        indicator);
    EXPECT_NEAR(penalty_wrapper.left, penalty_core.left, 1e-12);
    EXPECT_NEAR(penalty_wrapper.right, penalty_core.right, 1e-12);
}

namespace {

/**
 * Two cells of one shape, every face stored, the shared face in the face set
 * "shared".
 */
std::shared_ptr<svmp::Mesh> twoCells(svmp::CellFamily family,
                                     const std::vector<svmp::real_t>& X,
                                     const std::vector<svmp::index_t>& conn)
{
    svmp::CellShape shape{};
    shape.family = family;
    shape.num_corners = static_cast<int>(conn.size() / 2u);
    shape.order = 1;
    const auto corners = static_cast<svmp::offset_t>(shape.num_corners);
    auto base = std::make_shared<svmp::MeshBase>();
    base->build_from_arrays(/*spatial_dim=*/3, X, {0, corners, 2 * corners}, conn, {shape, shape});
    base->finalize();

    auto mesh = svmp::create_mesh(std::move(base));
    auto& mbase = mesh->local_mesh();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mbase.n_faces()); ++f) {
        if (mbase.face2cell()[static_cast<std::size_t>(f)][1] != svmp::INVALID_INDEX) {
            mbase.add_to_set(svmp::EntityKind::Face, "shared", f);
        }
    }
    return mesh;
}

using FaceTerms = std::function<forms::FormExpr(const forms::FormExpr&, const forms::FormExpr&)>;

/**
 * The matrix of @p face_terms(u, v) for a continuous first-order field u on
 * @p mesh (the two cells @p conn), keyed by the mesh vertices of its row and
 * column (the cells' DOFs in their vertex order).
 */
std::map<std::pair<svmp::index_t, svmp::index_t>, Real> vertexMatrix(const std::shared_ptr<svmp::Mesh>& mesh,
                                                                     ElementType cell_type,
                                                                     const std::vector<svmp::index_t>& conn,
                                                                     const FaceTerms& face_terms)
{
    using forms::FormExpr;
    auto space = std::make_shared<spaces::H1Space>(cell_type, 1);
    FESystem sys(mesh);
    sys.setInterfaceMeshFromFaceSet(kInterfaceMarker, "shared");
    const FieldId u_id = sys.addField(FieldSpec{.name = "u", .space = space});
    sys.addOperator("op");
    const auto u = forms::StateField(u_id, *space, "u");
    const auto v = forms::TestField(u_id, *space, "v");
    FormInstallOptions options;
    options.compiler_options.jit.enable = false;
    (void)installFormulation(sys, "op", {u_id}, (FormExpr::constant(0.0) * u * v).dx() + face_terms(u, v),
                             options);
    sys.setup();

    const auto n_dofs = sys.dofHandler().getNumDofs();
    std::vector<Real> zero(static_cast<std::size_t>(n_dofs), 0.0);
    SystemStateView state;
    state.u = zero;
    AssemblyRequest request;
    request.op = "op";
    request.want_matrix = true;
    assembly::DenseMatrixView A(n_dofs);
    EXPECT_TRUE(sys.assemble(request, state, &A, nullptr).success);

    const auto corners = conn.size() / 2u;
    std::map<GlobalIndex, svmp::index_t> vertex_of_dof;
    for (GlobalIndex cell = 0; cell < 2; ++cell) {
        const auto dofs = sys.fieldDofHandler(u_id).getCellDofs(cell);
        EXPECT_EQ(dofs.size(), corners);
        for (std::size_t k = 0; k < dofs.size() && k < corners; ++k) {
            vertex_of_dof[sys.fieldDofOffset(u_id) + dofs[k]] = conn[static_cast<std::size_t>(cell) * corners + k];
        }
    }
    std::map<std::pair<svmp::index_t, svmp::index_t>, Real> out;
    for (const auto& [i, vi] : vertex_of_dof) {
        for (const auto& [j, vj] : vertex_of_dof) {
            out[{vi, vj}] = A.getMatrixEntry(i, j);
        }
    }
    return out;
}

/// u v on the shared face for each pair of sides, and jump(u) jump(v).
std::vector<std::pair<std::string, FaceTerms>> sidedFaceTerms(
    const std::function<forms::FormExpr(const forms::FormExpr&)>& on_face)
{
    using forms::FormExpr;
    return {
        {"minus-minus", [=](const FormExpr& u, const FormExpr& v) { return on_face(u.minus() * v.minus()); }},
        {"plus-plus", [=](const FormExpr& u, const FormExpr& v) { return on_face(u.plus() * v.plus()); }},
        {"minus-plus", [=](const FormExpr& u, const FormExpr& v) { return on_face(u.minus() * v.plus()); }},
        {"plus-minus", [=](const FormExpr& u, const FormExpr& v) { return on_face(u.plus() * v.minus()); }},
        {"jump-jump",
         [=](const FormExpr& u, const FormExpr& v) { return on_face(forms::jump(u) * forms::jump(v)); }},
    };
}

} // namespace

TEST(InterfaceFaceGeometry, TwoSidedTermsOnATetrahedralInterfaceFaceGiveTheExactFaceMass)
{
    // The shared face {1, 2, 3} is reference face 2 of cell 0 and reference
    // face 3 of cell 1: ReferenceElement numbers the Tetra4 faces {0,1,2},
    // {0,1,3}, {1,2,3}, {0,2,3}, and the mesh's canonical table (which the
    // InterfaceMesh records) swaps the last two.  For a continuous P1 field
    // each side pairing of u v on the face is the P1 face mass of the face's
    // vertices, |F|/12 (1 + delta_ij) with |F| = sqrt(3)/2, and jump(u) = 0.
    const std::vector<svmp::real_t> X = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
    const std::vector<svmp::index_t> conn = {0, 1, 2, 3, 1, 4, 2, 3};
    const Real area = std::sqrt(Real(3.0)) / Real(2.0);
    const std::set<svmp::index_t> face = {1, 2, 3};
    for (const auto& [name, terms] :
         sidedFaceTerms([](const forms::FormExpr& e) { return e.dI(kInterfaceMarker); })) {
        SCOPED_TRACE(name);
        const auto M = vertexMatrix(twoCells(svmp::CellFamily::Tetra, X, conn), ElementType::Tetra4, conn, terms);
        for (const auto& [key, value] : M) {
            const auto [a, b] = key;
            const bool on_face = face.contains(a) && face.contains(b);
            const Real expected =
                (name == "jump-jump" || !on_face) ? Real(0.0) : area / Real(12.0) * (a == b ? Real(2.0) : Real(1.0));
            EXPECT_NEAR(value, expected, 1e-13) << "vertices " << a << ", " << b;
        }
    }
}

TEST(InterfaceFaceGeometry, TwoSidedTermsOnAQuadrilateralFaceGiveTheExactFaceMass)
{
    // Two unit cubes in the same vertex order share the face x = 1, which the
    // two sides' reference facets run in opposite directions.  For a
    // continuous Q1 field each side pairing of u v on the face, as an
    // interior face (dS) and as an interface face (dI), is the Q1 face mass
    // of the unit square ((4, 2, 1)/36 for the same vertex, an edge neighbour
    // and the opposite vertex), and jump(u) = 0.
    std::vector<svmp::real_t> X;
    for (int k = 0; k < 2; ++k) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 3; ++i) {
                X.insert(X.end(), {static_cast<svmp::real_t>(i), static_cast<svmp::real_t>(j),
                                   static_cast<svmp::real_t>(k)});
            }
        }
    }
    const auto vid = [](int i, int j, int k) { return static_cast<svmp::index_t>(i + 3 * (j + 2 * k)); };
    std::vector<svmp::index_t> conn;
    for (int c = 0; c < 2; ++c) {
        conn.insert(conn.end(), {vid(c, 0, 0), vid(c + 1, 0, 0), vid(c + 1, 1, 0), vid(c, 1, 0),
                                 vid(c, 0, 1), vid(c + 1, 0, 1), vid(c + 1, 1, 1), vid(c, 1, 1)});
    }
    std::vector<std::pair<std::string, FaceTerms>> cases;
    for (auto& [name, terms] : sidedFaceTerms([](const forms::FormExpr& e) { return e.dS(); })) {
        cases.emplace_back("dS " + name, std::move(terms));
    }
    for (auto& [name, terms] : sidedFaceTerms([](const forms::FormExpr& e) { return e.dI(kInterfaceMarker); })) {
        cases.emplace_back("dI " + name, std::move(terms));
    }
    for (const auto& [name, terms] : cases) {
        SCOPED_TRACE(name);
        const auto M = vertexMatrix(twoCells(svmp::CellFamily::Hex, X, conn), ElementType::Hex8, conn, terms);
        for (const auto& [key, value] : M) {
            const auto [a, b] = key;
            const bool on_face = (a % 3 == 1) && (b % 3 == 1);
            Real expected = 0.0;
            if (on_face && !name.ends_with("jump-jump")) {
                const int apart = std::abs(static_cast<int>((a / 3) % 2) - static_cast<int>((b / 3) % 2)) +
                                  std::abs(static_cast<int>(a / 6) - static_cast<int>(b / 6));
                expected = (apart == 0 ? Real(4.0) : apart == 1 ? Real(2.0) : Real(1.0)) / Real(36.0);
            }
            EXPECT_NEAR(value, expected, 1e-13) << "vertices " << a << ", " << b;
        }
    }
}

} // namespace svmp::FE::systems::test
