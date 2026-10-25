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
 * run in opposite directions, as an interior and as an interface face; on an
 * interface face whose stored face id setup renumbers; and on the face
 * between two cells of different types (Quad4-Triangle3, Pyramid5-Tetra4,
 * Wedge6-Tetra4, Hex8-Wedge6, Hex8-Pyramid5), with a first-order field on
 * each cell's own element.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/MeshAccess.h"
#include "Elements/ElementFactory.h"
#include "Elements/ReferenceElement.h"
#include "Forms/BoundaryConditions.h"
#include "Forms/InterfaceConditions.h"
#include "Forms/StandardBCs.h"
#include "Forms/Vocabulary.h"
#include "Mesh/Core/InterfaceMesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include <algorithm>
#include <array>
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


namespace {

/// First-order Lagrange on each cell's own reference element: a continuous
/// P1/Q1 field on a mesh of several cell types (numbered through the
/// variable-order DOF path, which reads each cell's element).
class FirstOrderOnEachCellType final : public spaces::FunctionSpace {
public:
    FirstOrderOnEachCellType(int dim, std::vector<ElementType> cell_types)
        : dim_(dim), cell_types_(std::move(cell_types))
    {
        for (const auto type : cell_types_) {
            if (elements_.find(type) == elements_.end()) {
                elements::ElementRequest request;
                request.element_type = type;
                request.field_type = FieldType::Scalar;
                request.continuity = Continuity::C0;
                request.basis_type = BasisType::Lagrange;
                request.order = 1;
                elements_[type] = elements::ElementFactory::create(request);
            }
        }
    }

    spaces::SpaceType space_type() const noexcept override { return spaces::SpaceType::H1; }
    FieldType field_type() const noexcept override { return FieldType::Scalar; }
    Continuity continuity() const noexcept override { return Continuity::C0; }
    int value_dimension() const noexcept override { return 1; }
    int topological_dimension() const noexcept override { return dim_; }
    int polynomial_order() const noexcept override { return 1; }
    int polynomial_order(GlobalIndex /*cell_id*/) const noexcept override { return 1; }
    bool is_variable_order() const noexcept override { return true; }
    ElementType element_type() const noexcept override { return ElementType::Unknown; }
    const elements::Element& element() const noexcept override { return *elements_.at(cell_types_.front()); }
    const elements::Element& getElement(ElementType cell_type, GlobalIndex /*cell_id*/) const noexcept override
    {
        return *elements_.at(cell_type);
    }
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override
    {
        return elements_.at(cell_types_.front());
    }
    std::size_t dofs_per_element() const noexcept override { return element().num_dofs(); }
    std::size_t dofs_per_element(GlobalIndex cell_id) const noexcept override
    {
        return elements_.at(cell_types_.at(static_cast<std::size_t>(cell_id)))->num_dofs();
    }
    void interpolate(const ValueFunction& /*function*/, std::vector<Real>& /*coefficients*/) const override {}
    Value evaluate(const Value& /*xi*/, const std::vector<Real>& /*coefficients*/) const override { return {}; }

private:
    int dim_;
    std::vector<ElementType> cell_types_;
    std::map<ElementType, std::shared_ptr<elements::Element>> elements_;
};

/// Two cells of different types sharing one face; cell 0 is the minus side.
struct MixedCellPair {
    std::string name;
    int dim{3};
    std::vector<svmp::real_t> X;
    std::array<ElementType, 2> types{};
    std::array<std::vector<svmp::index_t>, 2> cells;   ///< vertices in each type's reference order
};

svmp::CellFamily familyOf(ElementType type)
{
    switch (type) {
    case ElementType::Triangle3: return svmp::CellFamily::Triangle;
    case ElementType::Quad4: return svmp::CellFamily::Quad;
    case ElementType::Tetra4: return svmp::CellFamily::Tetra;
    case ElementType::Hex8: return svmp::CellFamily::Hex;
    case ElementType::Wedge6: return svmp::CellFamily::Wedge;
    case ElementType::Pyramid5: return svmp::CellFamily::Pyramid;
    default: return svmp::CellFamily::Point;
    }
}

/// The same pair with its cells in the other order (the other type is minus).
MixedCellPair swapped(MixedCellPair pair)
{
    std::swap(pair.types[0], pair.types[1]);
    std::swap(pair.cells[0], pair.cells[1]);
    pair.name += " (swapped)";
    return pair;
}

/// The topology the variable-order DOF numbering reads: the cells' vertices
/// and their edges (and faces in 3D) in each reference element's order.
dofs::MeshTopologyInfo topologyOf(const MixedCellPair& pair)
{
    dofs::MeshTopologyInfo topo;
    topo.dim = pair.dim;
    topo.n_cells = 2;
    topo.n_vertices = static_cast<GlobalIndex>(pair.X.size() / static_cast<std::size_t>(pair.dim));
    std::map<std::vector<svmp::index_t>, MeshIndex> edges;
    std::map<std::vector<svmp::index_t>, MeshIndex> faces;
    topo.cell2vertex_offsets = {0};
    topo.cell2edge_offsets = {0};
    topo.cell2face_offsets = {0};
    topo.face2vertex_offsets = {0};
    for (std::size_t c = 0; c < 2; ++c) {
        const auto& cell = pair.cells[c];
        const auto& ref = elements::ReferenceElement::shared(pair.types[c]);
        topo.cell2vertex_data.insert(topo.cell2vertex_data.end(), cell.begin(), cell.end());
        topo.cell2vertex_offsets.push_back(static_cast<MeshOffset>(topo.cell2vertex_data.size()));
        for (std::size_t e = 0; e < ref.num_edges(); ++e) {
            std::vector<svmp::index_t> key;
            for (const auto node : ref.edge_nodes(e)) {
                key.push_back(cell[static_cast<std::size_t>(node)]);
            }
            std::sort(key.begin(), key.end());
            const auto [it, added] = edges.emplace(key, static_cast<MeshIndex>(edges.size()));
            if (added) {
                topo.edge2vertex_data.insert(topo.edge2vertex_data.end(), key.begin(), key.end());
            }
            topo.cell2edge_data.push_back(it->second);
        }
        topo.cell2edge_offsets.push_back(static_cast<MeshOffset>(topo.cell2edge_data.size()));
        if (pair.dim == 3) {
            for (std::size_t f = 0; f < ref.num_faces(); ++f) {
                std::vector<svmp::index_t> key;
                for (const auto node : ref.face_nodes(f)) {
                    key.push_back(cell[static_cast<std::size_t>(node)]);
                }
                std::vector<svmp::index_t> sorted = key;
                std::sort(sorted.begin(), sorted.end());
                const auto [it, added] = faces.emplace(sorted, static_cast<MeshIndex>(faces.size()));
                if (added) {
                    topo.face2vertex_data.insert(topo.face2vertex_data.end(), key.begin(), key.end());
                    topo.face2vertex_offsets.push_back(static_cast<MeshOffset>(topo.face2vertex_data.size()));
                }
                topo.cell2face_data.push_back(it->second);
            }
            topo.cell2face_offsets.push_back(static_cast<MeshOffset>(topo.cell2face_data.size()));
        }
    }
    topo.n_edges = static_cast<GlobalIndex>(edges.size());
    if (pair.dim == 3) {
        topo.n_faces = static_cast<GlobalIndex>(faces.size());
    } else {
        topo.cell2face_offsets.clear();
        topo.face2vertex_offsets.clear();
    }
    for (GlobalIndex v = 0; v < topo.n_vertices; ++v) {
        topo.vertex_gids.push_back(static_cast<gid_t>(v));
    }
    for (GlobalIndex e = 0; e < topo.n_edges; ++e) {
        topo.edge_gids.push_back(static_cast<gid_t>(e));
    }
    for (GlobalIndex f = 0; f < topo.n_faces; ++f) {
        topo.face_gids.push_back(static_cast<gid_t>(f));
    }
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 0};
    return topo;
}

/**
 * The matrix of @p face_terms(u, v) for a continuous first-order field u on
 * the two cells of @p pair, the shared face also an InterfaceMesh face, keyed
 * by the mesh vertices of its row and column.
 */
std::map<std::pair<svmp::index_t, svmp::index_t>, Real> mixedVertexMatrix(const MixedCellPair& pair,
                                                                          const FaceTerms& face_terms)
{
    using forms::FormExpr;
    std::vector<svmp::CellShape> shapes;
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> conn;
    for (std::size_t c = 0; c < 2; ++c) {
        svmp::CellShape shape{};
        shape.family = familyOf(pair.types[c]);
        shape.num_corners = static_cast<int>(pair.cells[c].size());
        shape.order = 1;
        shapes.push_back(shape);
        conn.insert(conn.end(), pair.cells[c].begin(), pair.cells[c].end());
        offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
    }
    auto base = std::make_shared<svmp::MeshBase>();
    base->build_from_arrays(pair.dim, pair.X, offsets, conn, shapes);
    base->finalize();
    auto mesh = svmp::create_mesh(std::move(base));
    auto& mbase = mesh->local_mesh();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mbase.n_faces()); ++f) {
        if (mbase.face2cell()[static_cast<std::size_t>(f)][1] != svmp::INVALID_INDEX) {
            mbase.add_to_set(svmp::EntityKind::Face, "shared", f);
        }
    }

    auto space = std::make_shared<FirstOrderOnEachCellType>(pair.dim, std::vector<ElementType>{pair.types[0], pair.types[1]});
    FESystem sys(std::make_shared<assembly::MeshAccess>(*mesh));
    sys.setInterfaceMesh(kInterfaceMarker, std::make_shared<svmp::InterfaceMesh>(
                                               svmp::InterfaceMesh::build_from_face_set(mbase, "shared", false)));
    const FieldId u_id = sys.addField(FieldSpec{.name = "u", .space = space});
    sys.addOperator("op");
    const auto u = forms::StateField(u_id, *space, "u");
    const auto v = forms::TestField(u_id, *space, "v");
    FormInstallOptions options;
    options.compiler_options.jit.enable = false;
    (void)installFormulation(sys, "op", {u_id}, (FormExpr::constant(0.0) * u * v).dx() + face_terms(u, v),
                             options);
    SetupInputs inputs;
    inputs.topology_override = topologyOf(pair);
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    std::vector<Real> zero(static_cast<std::size_t>(n_dofs), 0.0);
    SystemStateView state;
    state.u = zero;
    AssemblyRequest request;
    request.op = "op";
    request.want_matrix = true;
    assembly::DenseMatrixView A(n_dofs);
    EXPECT_TRUE(sys.assemble(request, state, &A, nullptr).success);

    std::map<GlobalIndex, svmp::index_t> vertex_of_dof;
    for (GlobalIndex cell = 0; cell < 2; ++cell) {
        const auto& vertices = pair.cells[static_cast<std::size_t>(cell)];
        const auto dofs = sys.fieldDofHandler(u_id).getCellDofs(cell);
        EXPECT_EQ(dofs.size(), vertices.size());
        for (std::size_t k = 0; k < dofs.size() && k < vertices.size(); ++k) {
            vertex_of_dof[sys.fieldDofOffset(u_id) + dofs[k]] = vertices[k];
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

/// The exact first-order face mass of a planar segment, triangle or unit square
/// with corners @p face, between mesh vertices @p a and @p b.
Real exactFaceMass(const MixedCellPair& pair, const std::vector<svmp::index_t>& face, svmp::index_t a,
                   svmp::index_t b)
{
    const auto dim = static_cast<std::size_t>(pair.dim);
    const auto point = [&](svmp::index_t vertex) {
        std::array<Real, 3> x{0.0, 0.0, 0.0};
        for (std::size_t d = 0; d < dim; ++d) {
            x[d] = pair.X[static_cast<std::size_t>(vertex) * dim + d];
        }
        return x;
    };
    const auto sub = [](const std::array<Real, 3>& p, const std::array<Real, 3>& q) {
        return std::array<Real, 3>{p[0] - q[0], p[1] - q[1], p[2] - q[2]};
    };
    const auto norm = [](const std::array<Real, 3>& p) { return std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]); };
    if (face.size() == 2u) {
        return norm(sub(point(face[1]), point(face[0]))) / 6.0 * (a == b ? 2.0 : 1.0);
    }
    if (face.size() == 3u) {
        const auto e1 = sub(point(face[1]), point(face[0]));
        const auto e2 = sub(point(face[2]), point(face[0]));
        const std::array<Real, 3> n{e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                                    e1[0] * e2[1] - e1[1] * e2[0]};
        return norm(n) / 2.0 / 12.0 * (a == b ? 2.0 : 1.0);
    }
    const Real distance = norm(sub(point(a), point(b)));   // unit square: 0, 1 or sqrt(2)
    return (distance < 0.5 ? 4.0 : distance < 1.2 ? 2.0 : 1.0) / 36.0;
}

std::vector<MixedCellPair> mixedCellPairs()
{
    std::vector<MixedCellPair> pairs;
    // Quad4 [0,1] x [0,1] and a triangle on its side x = 1.
    pairs.push_back({"Quad4-Triangle3", 2, {0, 0, 1, 0, 2, 0.5, 0, 1, 1, 1},
                     {ElementType::Quad4, ElementType::Triangle3}, {{{0, 1, 4, 3}, {1, 2, 4}}}});
    // A pyramid on the unit square and a tetrahedron on its side face
    // {0, 1, 4}, which the tetrahedron's facet runs turned by one place.
    pairs.push_back({"Pyramid5-Tetra4", 3,
                     {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.5, 0.5, 1, 0.5, -1, 0.5},
                     {ElementType::Pyramid5, ElementType::Tetra4}, {{{0, 1, 2, 3, 4}, {1, 4, 0, 5}}}});
    // A unit-height wedge and a tetrahedron on its top face {3, 4, 5}, turned by one place.
    pairs.push_back({"Wedge6-Tetra4", 3,
                     {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0.3, 0.3, 2},
                     {ElementType::Wedge6, ElementType::Tetra4}, {{{0, 1, 2, 3, 4, 5}, {4, 5, 3, 6}}}});
    // The unit cube and a wedge on its face x = 1, which the wedge's facet runs mirrored.
    const std::vector<svmp::real_t> cube_and_tip = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1,
                                                    0, 1, 1, 2, 0.5, 0, 2, 0.5, 1};
    pairs.push_back({"Hex8-Wedge6", 3, cube_and_tip, {ElementType::Hex8, ElementType::Wedge6},
                     {{{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 2, 5, 9, 6}}}});
    // The unit cube and a pyramid on its face x = 1, its base turned a quarter.
    std::vector<svmp::real_t> cube_and_apex(cube_and_tip.begin(), cube_and_tip.begin() + 24);
    cube_and_apex.insert(cube_and_apex.end(), {1.5, 0.5, 0.5});
    pairs.push_back({"Hex8-Pyramid5", 3, cube_and_apex, {ElementType::Hex8, ElementType::Pyramid5},
                     {{{0, 1, 2, 3, 4, 5, 6, 7}, {2, 6, 5, 1, 8}}}});
    return pairs;
}

} // namespace

TEST(InterfaceFaceGeometry, TwoSidedTermsOnAFaceBetweenCellsOfDifferentTypesGiveTheExactFaceMass)
{
    // For a continuous first-order field on two cells of different types,
    // each side pairing of u v on the shared face, as an interior face (dS)
    // and as an interface face (dI), is the face mass of the face's vertices,
    // and jump(u) = 0; either cell as the minus side.
    for (const auto& base_pair : mixedCellPairs()) {
        for (const auto& pair : {base_pair, swapped(base_pair)}) {
            std::set<svmp::index_t> shared(pair.cells[0].begin(), pair.cells[0].end());
            std::vector<svmp::index_t> face;
            for (const auto vertex : pair.cells[1]) {
                if (shared.contains(vertex)) {
                    face.push_back(vertex);
                }
            }
            const std::set<svmp::index_t> on_face(face.begin(), face.end());
            std::vector<std::pair<std::string, FaceTerms>> cases;
            for (auto& [name, terms] : sidedFaceTerms([](const forms::FormExpr& e) { return e.dS(); })) {
                cases.emplace_back("dS " + name, std::move(terms));
            }
            for (auto& [name, terms] :
                 sidedFaceTerms([](const forms::FormExpr& e) { return e.dI(kInterfaceMarker); })) {
                cases.emplace_back("dI " + name, std::move(terms));
            }
            for (const auto& [name, terms] : cases) {
                SCOPED_TRACE(pair.name + ": " + name);
                const auto M = mixedVertexMatrix(pair, terms);
                for (const auto& [key, value] : M) {
                    const auto [a, b] = key;
                    const Real expected = (!name.ends_with("jump-jump") && on_face.contains(a) && on_face.contains(b))
                                              ? exactFaceMass(pair, face, a, b)
                                              : Real(0.0);
                    EXPECT_NEAR(value, expected, 1e-13) << "vertices " << a << ", " << b;
                }
            }
        }
    }
}


TEST(InterfaceFaceGeometry, TwoSidedInterfaceFacesSurviveTheFacesSetupStores)
{
    // The two tetrahedra of the first test, the mesh storing its boundary
    // faces and the shared face only, the shared face last.  The interior
    // face terms make setup store every face, which renumbers them after the
    // InterfaceMesh recorded the shared face's id; the two-sided assembly
    // must find each cell's facet by the face's vertices.
    using forms::FormExpr;
    const std::vector<svmp::real_t> X = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
    const std::vector<svmp::index_t> conn = {0, 1, 2, 3, 1, 4, 2, 3};
    const Real area = std::sqrt(Real(3.0)) / Real(2.0);
    const std::set<svmp::index_t> face = {1, 2, 3};
    for (const auto& [name, terms] :
         sidedFaceTerms([](const FormExpr& e) { return e.dI(kInterfaceMarker); })) {
        SCOPED_TRACE(name);
        svmp::CellShape tet{};
        tet.family = svmp::CellFamily::Tetra;
        tet.num_corners = 4;
        tet.order = 1;
        auto base = std::make_shared<svmp::MeshBase>();
        base->build_from_arrays(/*spatial_dim=*/3, X, {0, 4, 8}, conn, {tet, tet});
        svmp::CellShape tri{};
        tri.family = svmp::CellFamily::Triangle;
        tri.num_corners = 3;
        tri.order = 1;
        std::vector<svmp::index_t> face_conn = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 4, 1, 4, 3, 2, 3, 4, 1, 2, 3};
        std::vector<std::array<svmp::index_t, 2>> face_cells = {
            {0, svmp::INVALID_INDEX}, {0, svmp::INVALID_INDEX}, {0, svmp::INVALID_INDEX}, {1, svmp::INVALID_INDEX},
            {1, svmp::INVALID_INDEX}, {1, svmp::INVALID_INDEX}, {0, 1}};
        std::vector<svmp::offset_t> face_offsets;
        for (std::size_t f = 0; f <= face_cells.size(); ++f) {
            face_offsets.push_back(static_cast<svmp::offset_t>(3 * f));
        }
        base->set_faces_from_arrays(std::vector<svmp::CellShape>(face_cells.size(), tri), std::move(face_offsets),
                                    std::move(face_conn), std::move(face_cells),
                                    svmp::MeshCodim1StorageMode::BoundaryOnly);
        base->add_to_set(svmp::EntityKind::Face, "shared", 6);
        auto mesh = svmp::create_mesh(std::move(base));

        auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
        FESystem sys(mesh);
        sys.setInterfaceMeshFromFaceSet(kInterfaceMarker, "shared", /*compute_orientation=*/false);
        const auto recorded_face = sys.interfaceMesh(kInterfaceMarker).volume_face(0);
        const FieldId u_id = sys.addField(FieldSpec{.name = "u", .space = space});
        sys.addOperator("op");
        const auto u = forms::StateField(u_id, *space, "u");
        const auto v = forms::TestField(u_id, *space, "v");
        FormInstallOptions options;
        options.compiler_options.jit.enable = false;
        (void)installFormulation(sys, "op", {u_id},
                                 (FormExpr::constant(0.0) * u * v).dx() + terms(u, v) +
                                     (FormExpr::constant(0.0) * forms::jump(u) * forms::jump(v)).dS(),
                                 options);
        sys.setup();

        // Setup stored every face, and the recorded id is now another face.
        const auto& mbase = mesh->local_mesh();
        ASSERT_EQ(mbase.n_faces(), 7u);
        const auto [recorded_ptr, recorded_count] =
            mbase.face_vertices_span(static_cast<svmp::index_t>(recorded_face));
        std::set<svmp::index_t> recorded(recorded_ptr, recorded_ptr + recorded_count);
        EXPECT_NE(recorded, face);

        const auto n_dofs = sys.dofHandler().getNumDofs();
        std::vector<Real> zero(static_cast<std::size_t>(n_dofs), 0.0);
        SystemStateView state;
        state.u = zero;
        AssemblyRequest request;
        request.op = "op";
        request.want_matrix = true;
        assembly::DenseMatrixView A(n_dofs);
        ASSERT_TRUE(sys.assemble(request, state, &A, nullptr).success);
        std::map<GlobalIndex, svmp::index_t> vertex_of_dof;
        for (GlobalIndex cell = 0; cell < 2; ++cell) {
            const auto dofs = sys.fieldDofHandler(u_id).getCellDofs(cell);
            for (std::size_t k = 0; k < dofs.size(); ++k) {
                vertex_of_dof[sys.fieldDofOffset(u_id) + dofs[k]] = conn[static_cast<std::size_t>(cell) * 4u + k];
            }
        }
        for (const auto& [i, a] : vertex_of_dof) {
            for (const auto& [j, b] : vertex_of_dof) {
                const bool on_face = face.contains(a) && face.contains(b);
                const Real expected = (name == "jump-jump" || !on_face)
                                          ? Real(0.0)
                                          : area / Real(12.0) * (a == b ? Real(2.0) : Real(1.0));
                EXPECT_NEAR(A.getMatrixEntry(i, j), expected, 1e-13) << "vertices " << a << ", " << b;
            }
        }
    }
}

} // namespace svmp::FE::systems::test
