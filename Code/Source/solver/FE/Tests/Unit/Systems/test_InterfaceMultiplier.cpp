/**
 * @file test_InterfaceMultiplier.cpp
 * @brief Interface-owned multipliers in forms, one side at a time (D54 LM4, T2)
 *
 * A field on an InterfaceMesh (FieldScope::InterfaceFace) is single-valued on
 * the interface; the installer splits the dI terms that hold one into minus
 * and plus single-sided kernels (Systems/InterfaceSideSplit.h).
 *
 * I1: the unit square split at x = 1/2 into two subdomains with fields u1
 * (left) and u2 (right), each solving -Delta u = f on its side with u = g on
 * its outer Dirichlet side (x = 0, x = 1; natural elsewhere), glued on the
 * interface by a multiplier lambda in the full trace space of u (P1 on the
 * interface faces).  The glued solution equals the single-domain solution of
 * the same mesh to solver tolerance.
 *
 * InterfaceFaceAlignment: on a triangular or quadrilateral interface face
 * turned against the side's facet (3-cycles and quarter turns, which are not
 * their own inverses), and on a tetrahedral facet that the mesh and the
 * reference element number differently, a first-order interface field couples
 * to a first-order volume field as the exact face mass of the mesh vertices.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/SystemConstraint.h"
#include "Core/FEException.h"
#include "Dofs/EntityDofMap.h"
#include "Elements/ReferenceElement.h"
#include "Forms/BoundaryConditions.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/InterfaceSideSplit.h"

#include "Mesh/Core/InterfaceMesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#if FE_HAS_EIGEN
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::test_interface_multiplier {
namespace {

using forms::FormExpr;
using systems::FESystem;
using systems::FieldSpec;

constexpr int kLeft = 1;
constexpr int kRight = 2;
constexpr int kOther = 3;
constexpr int kGamma = 31;

/**
 * The n x n Quad4 grid of the unit square (n even), all faces stored: the
 * boundary faces at x = 0 and x = 1 labelled kLeft and kRight, the others
 * kOther, and the interior faces on x = 1/2 in the face set "gamma".  Cell
 * cy n + cx: on gamma the left cell has the lower id (the minus side).
 */
std::shared_ptr<svmp::Mesh> buildSplitSquare(int n)
{
    auto base = std::make_shared<svmp::MeshBase>();
    const auto id = [&](int i, int j) { return static_cast<svmp::index_t>(j * (n + 1) + i); };
    std::vector<svmp::real_t> X;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            X.push_back(static_cast<svmp::real_t>(static_cast<Real>(i) / static_cast<Real>(n)));
            X.push_back(static_cast<svmp::real_t>(static_cast<Real>(j) / static_cast<Real>(n)));
        }
    }
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> conn;
    std::vector<svmp::CellShape> shapes;
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    for (int cy = 0; cy < n; ++cy) {
        for (int cx = 0; cx < n; ++cx) {
            conn.insert(conn.end(), {id(cx, cy), id(cx + 1, cy), id(cx + 1, cy + 1), id(cx, cy + 1)});
            offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
            shapes.push_back(shape);
        }
    }
    base->build_from_arrays(/*spatial_dim=*/2, X, offsets, conn, shapes);
    base->finalize();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
        const auto vertices = base->face_vertices(f);
        Real x0 = 1e30;
        Real x1 = -1e30;
        for (const auto v : vertices) {
            const auto x = base->get_vertex_coords(v);
            x0 = std::min(x0, static_cast<Real>(x[0]));
            x1 = std::max(x1, static_cast<Real>(x[0]));
        }
        const bool boundary = base->face_cells(f)[1] == svmp::INVALID_INDEX;
        if (boundary) {
            base->set_boundary_label(f, x1 < 1e-12 ? kLeft : (x0 > 1.0 - 1e-12 ? kRight : kOther));
        } else if (std::abs(x0 - 0.5) < 1e-12 && std::abs(x1 - 0.5) < 1e-12) {
            base->add_to_set(svmp::EntityKind::Face, "gamma", f);
        }
    }
    return svmp::create_mesh(std::move(base));
}

/// Pins the DOFs of a field that only cells off its subdomain touch.
class OffSubdomainPin final : public constraints::ISystemConstraint {
public:
    OffSubdomainPin(FieldId field, std::function<bool(Real)> in_subdomain)
        : field_(field)
        , in_subdomain_(std::move(in_subdomain))
    {
    }

    void apply(const systems::FESystem& system, constraints::AffineConstraints& constraints) override
    {
        const auto& dof_map = system.fieldDofHandler(field_).getDofMap();
        const auto& base = system.mesh()->local_mesh();
        std::set<GlobalIndex> inside;
        std::set<GlobalIndex> outside;
        for (std::size_t c = 0; c < base.n_cells(); ++c) {
            const auto [vertices, count] = base.cell_vertices_span(static_cast<svmp::index_t>(c));
            Real xc = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                xc += static_cast<Real>(base.get_vertex_coords(vertices[i])[0]) / static_cast<Real>(count);
            }
            auto& set = in_subdomain_(xc) ? inside : outside;
            for (const auto dof : dof_map.getCellDofs(static_cast<GlobalIndex>(c))) {
                set.insert(dof);
            }
        }
        const auto offset = system.fieldDofOffset(field_);
        for (const auto dof : outside) {
            if (inside.count(dof) == 0u && !constraints.isConstrained(offset + dof)) {
                constraints.addDirichlet(offset + dof, 0.0);
            }
        }
    }
    bool updateValues(const systems::FESystem&, constraints::AffineConstraints&, double, double) override
    {
        return false;
    }
    [[nodiscard]] bool isTimeDependent() const noexcept override { return false; }
    [[nodiscard]] systems::SetupStorageRequirements storageRequirements() const noexcept override
    {
        systems::SetupStorageRequirements req;
        req.cell_topology = true;
        return req;
    }

private:
    FieldId field_;
    std::function<bool(Real)> in_subdomain_;
};

FormExpr coefficient(std::string name, std::function<Real(Real, Real)> f)
{
    return FormExpr::coefficient(std::move(name),
                                 forms::ScalarCoefficient([f](Real x, Real y, Real) { return f(x, y); }));
}

/// The (linear) problem's solution: one Newton step from the constraint values.
std::vector<Real> solveLinear(FESystem& sys)
{
    const auto n = sys.dofHandler().getNumDofs();
    std::vector<Real> state(static_cast<std::size_t>(n), 0.0);
    sys.constraints().distribute(state);
    assembly::DenseSystemView out(n);
    out.zero();
    systems::SystemStateView view;
    view.u = state;
    systems::AssemblyRequest request;
    request.op = "op";
    request.want_matrix = true;
    request.want_vector = true;
    request.suppress_constraint_inhomogeneity = true;
    EXPECT_TRUE(sys.assemble(request, view, &out, &out).success);
#if FE_HAS_EIGEN
    std::vector<Eigen::Triplet<Real>> entries;
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            const auto v = out.getMatrixEntry(i, j);
            if (v != Real(0.0)) {
                entries.emplace_back(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j), v);
            }
        }
    }
    Eigen::SparseMatrix<Real> A(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    A.setFromTriplets(entries.begin(), entries.end());
    Eigen::SparseLU<Eigen::SparseMatrix<Real>> lu;
    lu.compute(A);
    EXPECT_EQ(lu.info(), Eigen::Success) << "singular matrix";
    Eigen::Matrix<Real, Eigen::Dynamic, 1> rhs(static_cast<Eigen::Index>(n));
    for (GlobalIndex i = 0; i < n; ++i) {
        rhs[static_cast<Eigen::Index>(i)] = -out.getVectorEntry(i);
    }
    const Eigen::Matrix<Real, Eigen::Dynamic, 1> step = lu.solve(rhs);
    for (GlobalIndex i = 0; i < n; ++i) {
        state[static_cast<std::size_t>(i)] += step[static_cast<Eigen::Index>(i)];
    }
#else
    ADD_FAILURE() << "this test needs Eigen";
#endif
    sys.constraints().distribute(state);
    return state;
}

/// The value of a vertex-DOF field at every mesh vertex.
std::vector<Real> vertexValues(const FESystem& sys, FieldId field, const std::vector<Real>& state)
{
    const auto* map = sys.fieldDofHandler(field).getEntityDofMap();
    EXPECT_NE(map, nullptr);
    const auto n_vertices = sys.mesh()->local_mesh().n_vertices();
    std::vector<Real> out(n_vertices, 0.0);
    for (std::size_t v = 0; v < n_vertices; ++v) {
        const auto dofs = map->getVertexDofs(static_cast<GlobalIndex>(v));
        if (!dofs.empty()) {
            out[v] = state[static_cast<std::size_t>(sys.fieldDofOffset(field) + dofs[0])];
        }
    }
    return out;
}

/**
 * Two cells of @p cell_type sharing one face, every face stored, the shared
 * face in the face set "gamma".  The shared face's vertex list (which the
 * InterfaceMesh takes as its face) is cell 0's facet in the node order of
 * the reference element (elements::ReferenceElement::face_nodes), turned by
 * @p turn places: cell 0, the minus side, sees the interface face turned
 * against its facet.
 */
std::shared_ptr<svmp::Mesh> buildCellPair(ElementType cell_type,
                                          const std::vector<svmp::real_t>& X,
                                          const std::vector<svmp::index_t>& conn,
                                          int turn)
{
    svmp::CellShape shape{};
    shape.family = cell_type == ElementType::Tetra4 ? svmp::CellFamily::Tetra : svmp::CellFamily::Hex;
    shape.num_corners = static_cast<int>(conn.size() / 2u);
    shape.order = 1;
    const auto corners = static_cast<svmp::offset_t>(shape.num_corners);
    const std::vector<svmp::offset_t> offsets{0, corners, 2 * corners};

    // The faces as finalize() derives them.
    svmp::MeshBase derived;
    derived.build_from_arrays(/*spatial_dim=*/3, X, offsets, conn, {shape, shape});
    derived.finalize();
    auto face_shapes = derived.face_shapes();
    auto face_offsets = derived.face2vertex_offsets();
    auto face_conn = derived.face2vertex();
    auto face_cells = derived.face2cell();
    std::size_t shared = face_cells.size();
    for (std::size_t f = 0; f < face_cells.size(); ++f) {
        if (face_cells[f][1] != svmp::INVALID_INDEX) {
            shared = f;
        }
    }
    EXPECT_LT(shared, face_cells.size());
    const auto begin = static_cast<std::ptrdiff_t>(face_offsets[shared]);
    const auto end = static_cast<std::ptrdiff_t>(face_offsets[shared + 1u]);
    std::vector<svmp::index_t> face(face_conn.begin() + begin, face_conn.begin() + end);
    std::sort(face.begin(), face.end());

    // Cell 0's facet on it, in reference-element order, turned.
    const auto& ref = elements::ReferenceElement::shared(cell_type);
    std::vector<svmp::index_t> facet;
    for (std::size_t lf = 0; lf < ref.num_faces(); ++lf) {
        std::vector<svmp::index_t> candidate;
        for (const auto node : ref.face_nodes(lf)) {
            candidate.push_back(conn[static_cast<std::size_t>(node)]);
        }
        auto sorted = candidate;
        std::sort(sorted.begin(), sorted.end());
        if (sorted == face) {
            facet = candidate;
        }
    }
    EXPECT_EQ(facet.size(), face.size());
    std::rotate(facet.begin(), facet.begin() + turn, facet.end());
    std::copy(facet.begin(), facet.end(), face_conn.begin() + begin);

    auto base = std::make_shared<svmp::MeshBase>();
    base->build_from_arrays(/*spatial_dim=*/3, X, offsets, conn, {shape, shape});
    base->set_faces_from_arrays(std::move(face_shapes), std::move(face_offsets), std::move(face_conn),
                                std::move(face_cells));
    base->finalize();
    base->add_to_set(svmp::EntityKind::Face, "gamma", static_cast<svmp::index_t>(shared));
    return svmp::create_mesh(std::move(base));
}

/**
 * The coupling of a first-order interface field lambda on "gamma" with a
 * first-order volume field u, (lam * w.side()).dI + (mu * u.side()).dI on one
 * side: both blocks equal the face mass of the mesh vertices, @p face_mass,
 * and vanish for the vertices off the face.
 */
void expectFaceMassCoupling(const std::shared_ptr<svmp::Mesh>& mesh,
                            ElementType cell_type,
                            ElementType face_type,
                            bool plus,
                            const std::function<Real(svmp::index_t, svmp::index_t)>& face_mass)
{
    FESystem sys(mesh);
    sys.setInterfaceMeshFromFaceSet(kGamma, "gamma");
    auto volume = std::make_shared<spaces::H1Space>(cell_type, 1);
    auto trace = std::make_shared<spaces::H1Space>(face_type, 1);
    const auto u_id = sys.addField(FieldSpec{.name = "u", .space = volume});
    const auto lam_id = sys.addInterfaceField("lambda", trace, kGamma);
    sys.addOperator("op");
    {
        const auto u = forms::StateField(u_id, *volume, "u");
        const auto w = forms::TestField(u_id, *volume, "w");
        const auto lam = forms::StateField(lam_id, *trace, "lambda");
        const auto mu = forms::TestField(lam_id, *trace, "mu");
        const auto on_side = [plus](const FormExpr& e) { return plus ? e.plus() : e.minus(); };
        (void)systems::installFormulation(sys, "op", {u_id, lam_id},
                                          (u * w).dx() + (lam * on_side(w)).dI(kGamma) +
                                              (mu * on_side(u)).dI(kGamma));
    }
    sys.setup();
    const auto n = sys.dofHandler().getNumDofs();
    std::vector<Real> zero(static_cast<std::size_t>(n), 0.0);
    assembly::DenseSystemView out(n);
    out.zero();
    systems::SystemStateView view;
    view.u = zero;
    systems::AssemblyRequest request;
    request.op = "op";
    request.want_matrix = true;
    ASSERT_TRUE(sys.assemble(request, view, &out, nullptr).success);

    const auto& iface = sys.interfaceMesh(kGamma);
    ASSERT_EQ(iface.n_faces(), 1u);
    const auto [face_vertices, count] = iface.face_vertices_span(0);
    std::vector<svmp::index_t> on_face;
    for (std::size_t a = 0; a < count; ++a) {
        on_face.push_back(iface.volume_vertex(face_vertices[a]));
    }
    const auto lam_dofs = sys.fieldDofHandler(lam_id).getCellDofs(0);
    ASSERT_EQ(lam_dofs.size(), count);
    const auto* u_map = sys.fieldDofHandler(u_id).getEntityDofMap();
    ASSERT_NE(u_map, nullptr);
    const auto l0 = sys.fieldDofOffset(lam_id);
    const auto u0 = sys.fieldDofOffset(u_id);
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t v = 0; v < mesh->local_mesh().n_vertices(); ++v) {
            const auto vertex = static_cast<svmp::index_t>(v);
            const auto u_dofs = u_map->getVertexDofs(static_cast<GlobalIndex>(v));
            ASSERT_EQ(u_dofs.size(), 1u);
            const bool face_vertex = std::find(on_face.begin(), on_face.end(), vertex) != on_face.end();
            const Real expected = face_vertex ? face_mass(on_face[a], vertex) : Real(0.0);
            EXPECT_NEAR(out.getMatrixEntry(l0 + lam_dofs[a], u0 + u_dofs[0]), expected, 1e-14)
                << "lambda at vertex " << on_face[a] << ", u at vertex " << v;
            EXPECT_NEAR(out.getMatrixEntry(u0 + u_dofs[0], l0 + lam_dofs[a]), expected, 1e-14)
                << "u at vertex " << v << ", lambda at vertex " << on_face[a];
        }
    }
}

const auto kSource = [](Real x, Real y) { return Real(1.0) + x * y; };
const auto kData = [](Real x, Real y) { return x + Real(0.5) * y * y; };

} // namespace

TEST(InterfaceMultiplierI1, TwoSubdomainsGluedByAFullTraceMultiplierEqualTheSingleDomain)
{
    const int n = 6;
    auto q1 = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    const auto f = coefficient("f", kSource);
    const auto g = coefficient("g", kData);

    // The single domain.
    FESystem single(buildSplitSquare(n));
    const auto u_id = single.addField(FieldSpec{.name = "u", .space = q1});
    single.addOperator("op");
    {
        const auto u = forms::StateField(u_id, *q1, "u");
        const auto v = forms::TestField(u_id, *q1, "v");
        (void)systems::installFormulation(single, "op", {u_id},
                                          (forms::inner(forms::grad(u), forms::grad(v)) - f * v).dx());
        const std::array<forms::bc::StrongDirichlet, 2> bcs{forms::bc::strongDirichlet(u_id, kLeft, g, "u"),
                                                            forms::bc::strongDirichlet(u_id, kRight, g, "u")};
        systems::installStrongDirichlet(single, std::span<const forms::bc::StrongDirichlet>(bcs));
    }
    single.setup();
    const auto single_state = solveLinear(single);

    // Two subdomains glued by an interface-owned multiplier.
    FESystem glued(buildSplitSquare(n));
    glued.setInterfaceMeshFromFaceSet(kGamma, "gamma");
    const auto u1_id = glued.addField(FieldSpec{.name = "u1", .space = q1});
    const auto u2_id = glued.addField(FieldSpec{.name = "u2", .space = q1});
    auto p1_line = std::make_shared<spaces::H1Space>(ElementType::Line2, 1);
    const auto lam_id = glued.addInterfaceField("lambda", p1_line, kGamma);
    glued.addOperator("op");
    {
        const auto u1 = forms::StateField(u1_id, *q1, "u1");
        const auto v1 = forms::TestField(u1_id, *q1, "v1");
        const auto u2 = forms::StateField(u2_id, *q1, "u2");
        const auto v2 = forms::TestField(u2_id, *q1, "v2");
        const auto lam = forms::StateField(lam_id, *p1_line, "lambda");
        const auto mu = forms::TestField(lam_id, *p1_line, "mu");
        const auto left = coefficient("left", [](Real x, Real) { return x < 0.5 ? Real(1.0) : Real(0.0); });
        const auto right = coefficient("right", [](Real x, Real) { return x > 0.5 ? Real(1.0) : Real(0.0); });
        const auto residual =
            (left * (forms::inner(forms::grad(u1), forms::grad(v1)) - f * v1)).dx() +
            (right * (forms::inner(forms::grad(u2), forms::grad(v2)) - f * v2)).dx() +
            (lam * v1.minus()).dI(kGamma) - (lam * v2.plus()).dI(kGamma) +
            (mu * (u1.minus() - u2.plus())).dI(kGamma);
        (void)systems::installFormulation(glued, "op", {u1_id, u2_id, lam_id}, residual);
        const std::array<forms::bc::StrongDirichlet, 2> bcs{forms::bc::strongDirichlet(u1_id, kLeft, g, "u1"),
                                                            forms::bc::strongDirichlet(u2_id, kRight, g, "u2")};
        systems::installStrongDirichlet(glued, std::span<const forms::bc::StrongDirichlet>(bcs));
        glued.addSystemConstraint(std::make_unique<OffSubdomainPin>(u1_id, [](Real x) { return x < 0.5; }));
        glued.addSystemConstraint(std::make_unique<OffSubdomainPin>(u2_id, [](Real x) { return x > 0.5; }));
    }
    glued.setup();
    const auto glued_state = solveLinear(glued);
    const auto u = vertexValues(single, u_id, single_state);
    const auto u1 = vertexValues(glued, u1_id, glued_state);
    const auto u2 = vertexValues(glued, u2_id, glued_state);
    const auto& base = single.mesh()->local_mesh();
    Real scale = 0.0;
    for (const auto value : u) {
        scale = std::max(scale, std::abs(value));
    }
    for (std::size_t vtx = 0; vtx < u.size(); ++vtx) {
        const Real x = static_cast<Real>(base.get_vertex_coords(static_cast<svmp::index_t>(vtx))[0]);
        if (x <= 0.5 + 1e-12) {
            EXPECT_NEAR(u1[vtx], u[vtx], 1e-11 * scale) << "u1 at vertex " << vtx;
        }
        if (x >= 0.5 - 1e-12) {
            EXPECT_NEAR(u2[vtx], u[vtx], 1e-11 * scale) << "u2 at vertex " << vtx;
        }
    }
}

TEST(InterfaceMultiplierSplit, InterfaceTermsAreSplitBySideWithTheNormalResigned)
{
    FESystem sys(buildSplitSquare(2));
    sys.setInterfaceMeshFromFaceSet(kGamma, "gamma");
    auto q1 = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    auto p1_line = std::make_shared<spaces::H1Space>(ElementType::Line2, 1);
    const auto u_id = sys.addField(FieldSpec{.name = "u", .space = q1});
    const auto lam_id = sys.addInterfaceField("lambda", p1_line, kGamma);
    const auto u = forms::StateField(u_id, *q1, "u");
    const auto lam = forms::StateField(lam_id, *p1_line, "lambda");
    const auto mu = forms::TestField(lam_id, *p1_line, "mu");
    const auto n = FormExpr::normal();

    // mu (grad(u).minus() - grad(u).plus()) . n, and a term without a volume field.
    const auto row = (mu * forms::inner(forms::jump(forms::grad(u)), n)).dI(kGamma) +
                     (lam * mu).dI(kGamma) + (mu * u).dx();
    const auto split = systems::splitInterfaceSides(sys, row, "lambda");
    ASSERT_TRUE(split.rest.isValid());
    ASSERT_EQ(split.parts.size(), 2u);
    for (const auto& part : split.parts) {
        EXPECT_EQ(part.marker, kGamma);
        if (part.side == assembly::InterfaceEvaluationSide::Minus) {
            ASSERT_EQ(part.products.size(), 2u);
            EXPECT_EQ(part.products[0].toString(), (mu * forms::inner(forms::grad(u), n)).toString());
            EXPECT_EQ(part.products[1].toString(), (lam * mu).toString());
        } else {
            ASSERT_EQ(part.products.size(), 1u);
            // mu (-grad u) . (-n): the plus side's restriction dropped, n = -n_ctx there.
            EXPECT_EQ(part.products[0].toString(), (mu * forms::inner(-forms::grad(u), -n)).toString());
        }
    }
}

TEST(InterfaceMultiplierSplit, RefusesWhatOneSideCannotHold)
{
    FESystem sys(buildSplitSquare(2));
    sys.setInterfaceMeshFromFaceSet(kGamma, "gamma");
    auto q1 = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    auto p1_line = std::make_shared<spaces::H1Space>(ElementType::Line2, 1);
    const auto u_id = sys.addField(FieldSpec{.name = "u", .space = q1});
    const auto lam_id = sys.addInterfaceField("lambda", p1_line, kGamma);
    const auto u = forms::StateField(u_id, *q1, "u");
    const auto v = forms::TestField(u_id, *q1, "v");
    const auto lam = forms::StateField(lam_id, *p1_line, "lambda");
    const auto mu = forms::TestField(lam_id, *p1_line, "mu");

    const auto refused = [&](const FormExpr& row, const std::string& text) {
        try {
            (void)systems::splitInterfaceSides(sys, row, "row");
            ADD_FAILURE() << "accepted: " << text;
        } catch (const std::exception& e) {
            EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << e.what();
        }
    };
    refused((lam.minus() * v.minus()).dI(kGamma), "takes no .minus(), .plus(), jump() or avg()");
    refused((lam * v).dI(kGamma), "needs its side: write .minus() or .plus()");
    refused((mu * u.minus() * u.plus()).dI(kGamma), "holds both sides of the interface");
    refused((lam * u.minus() * v.minus()).dI(kGamma), "a product of an interface-owned unknown with another unknown");
    refused((lam * v.minus()).dI(), "without a marker");
}

TEST(InterfaceFaceAlignment, ATetrahedralInterfaceFieldCouplesAsTheFaceMass)
{
    // The face z = 0 (vertices 0, 1, 2; area 1/2) between a cell above
    // (apex 3) and one below (apex 4).  The interface face is cell 0's facet
    // turned by 0 (the control), 1 and 2 places (3-cycles, which are not
    // their own inverses); the facet is the cells' reference face 0 or, with
    // the apex first, face 2 (the mesh numbers that face 3).
    const std::vector<svmp::real_t> X{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                      0.2, 0.3, 1.0, 0.3, 0.2, -1.0};
    const std::vector<std::vector<svmp::index_t>> pairs{{0, 1, 2, 3, 0, 2, 1, 4},  // facet 0 on both sides
                                                        {3, 0, 2, 1, 4, 0, 1, 2}}; // facet 2 on both sides
    const auto face_mass = [](svmp::index_t a, svmp::index_t b) {
        return a == b ? Real(0.5) / Real(6.0) : Real(0.5) / Real(12.0);
    };
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        for (int turn = 0; turn < 3; ++turn) {
            for (const bool plus : {false, true}) {
                SCOPED_TRACE("cell pair " + std::to_string(k) + ", turn " + std::to_string(turn) +
                             (plus ? ", plus side" : ", minus side"));
                expectFaceMassCoupling(buildCellPair(ElementType::Tetra4, X, pairs[k], turn), ElementType::Tetra4,
                                       ElementType::Triangle3, plus, face_mass);
            }
        }
    }
}

TEST(InterfaceFaceAlignment, AHexahedralInterfaceFieldCouplesAsTheFaceMass)
{
    // The unit square z = 1 between the cubes [0,1]^2 x [0,1] and
    // [0,1]^2 x [1,2]; the interface face is the lower cube's top facet turned
    // by 0 (the control), 1 and 3 quarter turns (not their own inverses) and
    // a half turn.  The Q1 face mass is the product of the 1D P1 masses.
    std::vector<svmp::real_t> X;
    for (int z = 0; z <= 2; ++z) {
        for (const auto& [x, y] : std::array<std::pair<int, int>, 4>{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}}) {
            X.insert(X.end(), {static_cast<svmp::real_t>(x), static_cast<svmp::real_t>(y),
                               static_cast<svmp::real_t>(z)});
        }
    }
    const std::vector<svmp::index_t> conn{0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11};
    const auto face_mass = [&](svmp::index_t a, svmp::index_t b) {
        const auto ax = X[static_cast<std::size_t>(3 * a)];
        const auto ay = X[static_cast<std::size_t>(3 * a + 1)];
        const auto bx = X[static_cast<std::size_t>(3 * b)];
        const auto by = X[static_cast<std::size_t>(3 * b + 1)];
        const Real mx = ax == bx ? Real(1.0) / Real(3.0) : Real(1.0) / Real(6.0);
        const Real my = ay == by ? Real(1.0) / Real(3.0) : Real(1.0) / Real(6.0);
        return mx * my;
    };
    for (int turn = 0; turn < 4; ++turn) {
        for (const bool plus : {false, true}) {
            SCOPED_TRACE("turn " + std::to_string(turn) + (plus ? ", plus side" : ", minus side"));
            expectFaceMassCoupling(buildCellPair(ElementType::Hex8, X, conn, turn), ElementType::Hex8,
                                   ElementType::Quad4, plus, face_mass);
        }
    }
}

} // namespace svmp::FE::test_interface_multiplier
