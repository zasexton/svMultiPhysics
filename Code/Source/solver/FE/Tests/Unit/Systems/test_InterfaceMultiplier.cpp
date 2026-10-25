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
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/SystemConstraint.h"
#include "Core/FEException.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/BoundaryConditions.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/InterfaceSideSplit.h"

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

} // namespace svmp::FE::test_interface_multiplier
