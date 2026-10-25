/**
 * @file test_GlobalMultiplier.cpp
 * @brief Global multipliers as forms: the global field, its rows and the G1-G3 cases
 *
 * A global field lives on a GlobalSpace and has one DOF for the whole domain.
 * Its test function's row is a measured 0-form plus dGlobal terms; the
 * element kernels give the row's Jacobian and a GlobalRowKernel adds its
 * residual entry exactly.
 *
 * G1 (mean-constrained Poisson): on (0,1) x (0,b), -Delta u = lambda with
 * u = 0 at x = 0 and x = 1 and mean(u) = c gives u = 6c x(1-x) and
 * lambda = 12c.  Q2 is exact; on Q1 the nodal values are those of u and the
 * trapezoidal mean gives lambda_h = 12c / (1 - h^2), so the error is O(h^2).
 *
 * G2 (pure Neumann, mean zero): the multiplier's row fixes the constant of u,
 * so the gauge adds nothing, and u_h equals the solution under the automatic
 * weighted mean-zero gauge.  G3 (Taylor-Hood Stokes with a mean-zero pressure
 * multiplier): the velocities equal those with a pinned pressure, and the
 * pressures differ by a constant.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/DirichletBC.h"
#include "Core/FEException.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/BoundaryConditions.h"
#include "Forms/Vocabulary.h"
#include "Spaces/GlobalSpace.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Sparsity/SparsityPattern.h"
#include "Constraints/GaugeRegistry.h"
#include "Systems/FESystem.h"
#include "Systems/FieldBinding.h"
#include "Systems/FormulationBuilder.h"
#include "Systems/FormsInstaller.h"
#include "Systems/FormsInstallerDetail.h"
#include "Systems/FunctionalFormEvaluator.h"
#include "Systems/SystemCollectives.h"
#include "Systems/SystemManifest.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::systems::test_global_multiplier {
namespace {

using forms::FormExpr;

constexpr int kLeft = 1;
constexpr int kRight = 2;
constexpr int kBottom = 3;
constexpr int kTop = 4;

/**
 * A grid of nx x ny quadrilaterals (Quad4 for order 1, Quad9 for order 2) on
 * (0, lx) x (0, ly), with its boundary edges labelled left 1, right 2,
 * bottom 3 and top 4.
 */
std::shared_ptr<svmp::Mesh> buildGrid(int nx, int ny, Real lx, Real ly, int order)
{
    auto base = std::make_shared<svmp::MeshBase>();
    const int px = order * nx;
    const int py = order * ny;
    const auto id = [&](int i, int j) { return static_cast<svmp::index_t>(j * (px + 1) + i); };

    std::vector<svmp::real_t> X;
    X.reserve(static_cast<std::size_t>(2 * (px + 1) * (py + 1)));
    for (int j = 0; j <= py; ++j) {
        for (int i = 0; i <= px; ++i) {
            X.push_back(static_cast<svmp::real_t>(lx * static_cast<Real>(i) / static_cast<Real>(px)));
            X.push_back(static_cast<svmp::real_t>(ly * static_cast<Real>(j) / static_cast<Real>(py)));
        }
    }

    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> conn;
    std::vector<svmp::CellShape> shapes;
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = order;
    const auto cell_of = [&](int cx, int cy) { return static_cast<svmp::index_t>(cy * nx + cx); };
    for (int cy = 0; cy < ny; ++cy) {
        for (int cx = 0; cx < nx; ++cx) {
            const int i0 = order * cx;
            const int j0 = order * cy;
            if (order == 1) {
                conn.insert(conn.end(), {id(i0, j0), id(i0 + 1, j0), id(i0 + 1, j0 + 1), id(i0, j0 + 1)});
            } else {
                conn.insert(conn.end(), {id(i0, j0), id(i0 + 2, j0), id(i0 + 2, j0 + 2), id(i0, j0 + 2),
                                         id(i0 + 1, j0), id(i0 + 2, j0 + 1), id(i0 + 1, j0 + 2),
                                         id(i0, j0 + 1), id(i0 + 1, j0 + 1)});
            }
            offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
            shapes.push_back(shape);
        }
    }
    base->build_from_arrays(/*spatial_dim=*/2, X, offsets, conn, shapes);

    std::vector<svmp::CellShape> face_shapes;
    std::vector<svmp::offset_t> face_offsets{0};
    std::vector<svmp::index_t> face_conn;
    std::vector<std::array<svmp::index_t, 2>> face_cells;
    std::vector<int> labels;
    svmp::CellShape line{};
    line.family = svmp::CellFamily::Line;
    line.num_corners = 2;
    line.order = 1;
    const auto add_face = [&](svmp::index_t a, svmp::index_t b, svmp::index_t cell, int label) {
        face_shapes.push_back(line);
        face_conn.insert(face_conn.end(), {a, b});
        face_offsets.push_back(static_cast<svmp::offset_t>(face_conn.size()));
        face_cells.push_back({cell, svmp::INVALID_INDEX});
        labels.push_back(label);
    };
    for (int cx = 0; cx < nx; ++cx) {
        add_face(id(order * cx, 0), id(order * (cx + 1), 0), cell_of(cx, 0), kBottom);
    }
    for (int cy = 0; cy < ny; ++cy) {
        add_face(id(px, order * cy), id(px, order * (cy + 1)), cell_of(nx - 1, cy), kRight);
    }
    for (int cx = 0; cx < nx; ++cx) {
        add_face(id(order * (cx + 1), py), id(order * cx, py), cell_of(cx, ny - 1), kTop);
    }
    for (int cy = 0; cy < ny; ++cy) {
        add_face(id(0, order * (cy + 1)), id(0, order * cy), cell_of(0, cy), kLeft);
    }
    base->set_faces_from_arrays(std::move(face_shapes),
                                std::move(face_offsets),
                                std::move(face_conn),
                                std::move(face_cells),
                                svmp::MeshCodim1StorageMode::BoundaryOnly);
    for (std::size_t f = 0; f < labels.size(); ++f) {
        base->set_boundary_label(static_cast<svmp::index_t>(f), labels[f]);
    }
    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<spaces::H1Space> quadSpace(int order)
{
    return std::make_shared<spaces::H1Space>(order == 1 ? ElementType::Quad4 : ElementType::Quad9, order);
}

/**
 * The n x n grid of the unit square, each square split into two Triangle3
 * cells along its diagonal, with the boundary edges labelled as buildGrid.
 * Fields of any order live on these linear cells.
 */
std::shared_ptr<svmp::Mesh> buildTriangleGrid(int n)
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
    shape.family = svmp::CellFamily::Triangle;
    shape.num_corners = 3;
    shape.order = 1;
    // Cell 2 (j n + i) is the lower triangle of square (i, j), 2 (j n + i) + 1 the upper.
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            conn.insert(conn.end(), {id(i, j), id(i + 1, j), id(i + 1, j + 1)});
            offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
            conn.insert(conn.end(), {id(i, j), id(i + 1, j + 1), id(i, j + 1)});
            offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
            shapes.push_back(shape);
            shapes.push_back(shape);
        }
    }
    base->build_from_arrays(/*spatial_dim=*/2, X, offsets, conn, shapes);

    std::vector<svmp::CellShape> face_shapes;
    std::vector<svmp::offset_t> face_offsets{0};
    std::vector<svmp::index_t> face_conn;
    std::vector<std::array<svmp::index_t, 2>> face_cells;
    std::vector<int> labels;
    svmp::CellShape line{};
    line.family = svmp::CellFamily::Line;
    line.num_corners = 2;
    line.order = 1;
    const auto lower = [&](int i, int j) { return static_cast<svmp::index_t>(2 * (j * n + i)); };
    const auto add_face = [&](svmp::index_t a, svmp::index_t b, svmp::index_t cell, int label) {
        face_shapes.push_back(line);
        face_conn.insert(face_conn.end(), {a, b});
        face_offsets.push_back(static_cast<svmp::offset_t>(face_conn.size()));
        face_cells.push_back({cell, svmp::INVALID_INDEX});
        labels.push_back(label);
    };
    for (int i = 0; i < n; ++i) {
        add_face(id(i, 0), id(i + 1, 0), lower(i, 0), kBottom);
        add_face(id(i + 1, n), id(i, n), lower(i, n - 1) + 1, kTop);
    }
    for (int j = 0; j < n; ++j) {
        add_face(id(n, j), id(n, j + 1), lower(n - 1, j), kRight);
        add_face(id(0, j + 1), id(0, j), lower(0, j) + 1, kLeft);
    }
    base->set_faces_from_arrays(std::move(face_shapes),
                                std::move(face_offsets),
                                std::move(face_conn),
                                std::move(face_cells),
                                svmp::MeshCodim1StorageMode::BoundaryOnly);
    for (std::size_t f = 0; f < labels.size(); ++f) {
        base->set_boundary_label(static_cast<svmp::index_t>(f), labels[f]);
    }
    return svmp::create_mesh(std::move(base));
}

/// Solves A x = b (dense, partial pivoting; the saddle-point diagonal has zeros).
std::vector<Real> solveDense(std::vector<Real> a, std::vector<Real> b)
{
    const auto n = b.size();
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t p = k;
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(a[i * n + k]) > std::abs(a[p * n + k])) {
                p = i;
            }
        }
        EXPECT_GT(std::abs(a[p * n + k]), 1e-14) << "singular matrix at column " << k;
        if (p != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a[k * n + j], a[p * n + j]);
            }
            std::swap(b[k], b[p]);
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            const Real f = a[i * n + k] / a[k * n + k];
            if (f == Real(0.0)) {
                continue;
            }
            for (std::size_t j = k; j < n; ++j) {
                a[i * n + j] -= f * a[k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    std::vector<Real> x(n, 0.0);
    for (std::size_t k = n; k-- > 0;) {
        Real s = b[k];
        for (std::size_t j = k + 1; j < n; ++j) {
            s -= a[k * n + j] * x[j];
        }
        x[k] = s / a[k * n + k];
    }
    return x;
}

/// Mean-constrained Poisson on a grid: u, its multiplier lambda, and the operator.
struct MeanConstrainedPoisson {
    std::shared_ptr<svmp::Mesh> mesh;
    std::shared_ptr<spaces::H1Space> u_space;
    std::shared_ptr<spaces::GlobalSpace> r1;
    std::unique_ptr<FESystem> system;
    FieldId u{INVALID_FIELD_ID};
    FieldId lambda{INVALID_FIELD_ID};
};

struct BuildOptions {
    bool dirichlet{true};  ///< u = 0 at x = 0 and x = 1
    bool point_term{true}; ///< the dGlobal term; without it the integral of u is 0
    int threads{1};        ///< assembly threads
};

/**
 * R = (grad u . grad v - lambda v) dx - (mu u) dx + (mu c |Omega|) dGlobal:
 * -Delta u = lambda, the integral of u equal to c |Omega| (mean c).  The
 * coupling blocks are transposes, so the Jacobian is symmetric.
 */
MeanConstrainedPoisson buildMeanConstrainedPoisson(int nx, int ny, Real height, int order, Real c,
                                                   BuildOptions build = {})
{
    MeanConstrainedPoisson out;
    out.mesh = buildGrid(nx, ny, Real(1.0), height, order);
    out.u_space = quadSpace(order);
    out.r1 = std::make_shared<spaces::GlobalSpace>(1);
    out.system = std::make_unique<FESystem>(out.mesh);
    auto& sys = *out.system;
    out.u = sys.addField(FieldSpec{.name = "u", .space = out.u_space});
    out.lambda = sys.addField(FieldSpec{.name = "lambda", .space = out.r1, .scope = FieldScope::Global});
    sys.addOperator("poisson");

    const auto u = forms::StateField(out.u, *out.u_space, "u");
    const auto v = forms::TestField(out.u, *out.u_space, "v");
    const auto lam = forms::StateField(out.lambda, *out.r1, "lambda");
    const auto mu = forms::TestField(out.lambda, *out.r1, "mu");
    auto residual = (forms::inner(forms::grad(u), forms::grad(v)) - lam * v).dx() - (mu * u).dx();
    if (build.point_term) {
        residual = residual + (mu * FormExpr::constant(c * height)).dGlobal();
    }
    (void)installFormulation(sys, "poisson", {out.u, out.lambda}, residual);

    if (build.dirichlet) {
        const std::array<forms::bc::StrongDirichlet, 2> bcs{
            forms::bc::strongDirichlet(out.u, kLeft, FormExpr::constant(0.0), "u"),
            forms::bc::strongDirichlet(out.u, kRight, FormExpr::constant(0.0), "u")};
        installStrongDirichlet(sys, std::span<const forms::bc::StrongDirichlet>(bcs));
    }
    SetupOptions options;
    options.assembly_options.num_threads = build.threads;
    sys.setup(options);
    return out;
}

struct Assembled {
    std::vector<Real> matrix;
    std::vector<Real> vector;
};

Assembled assembleAt(FESystem& sys, std::span<const Real> u, bool want_matrix = true,
                     const OperatorTag& op = "poisson")
{
    const auto n = sys.dofHandler().getNumDofs();
    assembly::DenseSystemView out(n);
    out.zero();
    SystemStateView state;
    state.u = u;
    AssemblyRequest request;
    request.op = op;
    request.want_matrix = want_matrix;
    request.want_vector = true;
    request.suppress_constraint_inhomogeneity = true;
    const auto result = sys.assemble(request, state, &out, &out);
    EXPECT_TRUE(result.success);
    Assembled assembled;
    assembled.matrix.assign(out.matrixData().begin(), out.matrixData().end());
    assembled.vector.assign(out.vectorData().begin(), out.vectorData().end());
    return assembled;
}

/// One Newton step from 0 (the problem is linear): the solution.
std::vector<Real> solveFromZero(FESystem& sys, const OperatorTag& op = "poisson")
{
    const auto n = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> zero(n, 0.0);
    const auto assembled = assembleAt(sys, zero, /*want_matrix=*/true, op);
    std::vector<Real> rhs(n);
    for (std::size_t i = 0; i < n; ++i) {
        rhs[i] = -assembled.vector[i];
    }
    return solveDense(assembled.matrix, rhs);
}

/// The message of the FEException @p f throws (empty if it throws none).
template <typename F>
std::string feMessage(F&& f)
{
    try {
        f();
    } catch (const FEException& e) {
        return e.what();
    }
    return {};
}

/// Every DOF of the field (they are a contiguous block of the monolithic numbering).
std::vector<GlobalIndex> fieldDofs(const FESystem& sys, FieldId field)
{
    std::vector<GlobalIndex> dofs;
    const auto offset = sys.fieldDofOffset(field);
    for (GlobalIndex d = 0; d < sys.fieldDofHandler(field).getNumDofs(); ++d) {
        dofs.push_back(offset + d);
    }
    return dofs;
}

} // namespace

// ---------------------------------------------------------------------------
// Fields and setup
// ---------------------------------------------------------------------------

TEST(GlobalField, ScopeFollowsTheGlobalSpace)
{
    auto mesh = buildGrid(2, 1, 1.0, 1.0, 1);
    auto r1 = std::make_shared<spaces::GlobalSpace>(1);
    auto h1 = quadSpace(1);
    FESystem sys(mesh);
    const auto lam = sys.addField(FieldSpec{.name = "lambda", .space = r1});
    EXPECT_EQ(sys.fieldRecord(lam).scope, FieldScope::Global);
    EXPECT_TRUE(sys.hasGlobalFields());
    EXPECT_THROW((void)sys.addField(FieldSpec{.name = "w", .space = h1, .scope = FieldScope::Global}),
                 FEException);
    EXPECT_THROW((void)sys.addField(FieldSpec{.name = "m",
                                              .space = r1,
                                              .scope = FieldScope::InterfaceFace,
                                              .interface_marker = 3}),
                 FEException);
    EXPECT_THROW((void)sys.addField(FieldSpec{.name = "d",
                                              .space = r1,
                                              .source_kind = FieldSourceKind::PrescribedData}),
                 FEException);

    FESystem plain(mesh);
    (void)plain.addField(FieldSpec{.name = "u", .space = h1});
    EXPECT_FALSE(plain.hasGlobalFields());
}

TEST(GlobalField, DofsFollowTheFieldDofsAndCoupleToEveryCell)
{
    auto problem = buildMeanConstrainedPoisson(4, 2, 0.5, 1, 1.0, {.point_term = false});
    auto& sys = *problem.system;

    const auto n_u = sys.fieldDofHandler(problem.u).getNumDofs();
    EXPECT_EQ(n_u, 15);
    EXPECT_EQ(sys.fieldDofHandler(problem.lambda).getNumDofs(), 1);
    EXPECT_EQ(sys.fieldDofOffset(problem.lambda), n_u);
    EXPECT_EQ(sys.dofHandler().getNumDofs(), n_u + 1);

    const auto lam = sys.fieldDofOffset(problem.lambda);
    const auto& map = sys.dofHandler().getDofMap();
    EXPECT_TRUE(map.isOwnedDof(lam));
    for (GlobalIndex cell = 0; cell < sys.meshAccess().numCells(); ++cell) {
        const auto dofs = map.getCellDofs(cell);
        EXPECT_EQ(std::count(dofs.begin(), dofs.end(), lam), 1) << "cell " << cell;
    }

    // The global DOF has no mesh entity.
    const auto* entities = sys.dofHandler().getEntityDofMap();
    ASSERT_NE(entities, nullptr);
    EXPECT_FALSE(entities->getDofEntity(lam).has_value());

    // The row and the column of lambda are dense over u (the lambda-lambda
    // entry is structural: every cell lists the DOF).
    const auto& pattern = sys.sparsity("poisson");
    for (const auto dof : fieldDofs(sys, problem.u)) {
        EXPECT_TRUE(pattern.hasEntry(lam, dof)) << dof;
        EXPECT_TRUE(pattern.hasEntry(dof, lam)) << dof;
    }
    EXPECT_EQ(pattern.getRowNnz(lam), n_u + 1);

    // The rows of the global test function use the highest field's rule.
    EXPECT_EQ(problem.r1->quadratureOrder(), problem.u_space->element().quadrature()->order());
    EXPECT_TRUE(problem.r1->hasBoundQuadrature(ElementType::Quad4));
}

TEST(GlobalField, ColoredAssemblyIsRefused)
{
    auto mesh = buildGrid(2, 1, 1.0, 1.0, 1);
    auto h1 = quadSpace(1);
    auto r1 = std::make_shared<spaces::GlobalSpace>(1);
    FESystem sys(mesh);
    const auto u_id = sys.addField(FieldSpec{.name = "u", .space = h1});
    const auto lam_id = sys.addField(FieldSpec{.name = "lambda", .space = r1});
    sys.addOperator("op");
    const auto u = forms::StateField(u_id, *h1, "u");
    const auto v = forms::TestField(u_id, *h1, "v");
    const auto lam = forms::StateField(lam_id, *r1, "lambda");
    const auto mu = forms::TestField(lam_id, *r1, "mu");
    (void)installFormulation(sys, "op", {u_id, lam_id},
                             (forms::inner(forms::grad(u), forms::grad(v)) + lam * v).dx() + (mu * u).dx());
    SetupOptions options;
    options.assembly_options.use_coloring = true;
    EXPECT_THROW(sys.setup(options), FEException);
}

TEST(GlobalField, FieldBindingChecksTheScope)
{
    auto mesh = buildGrid(2, 1, 1.0, 1.0, 1);
    auto h1 = quadSpace(1);
    FESystem sys(mesh);
    FieldBinding fields(sys, "Test::registerOn");
    const auto lam = fields.global("lambda", "multiplier");
    EXPECT_EQ(sys.fieldRecord(lam).scope, FieldScope::Global);
    EXPECT_EQ(sys.fieldRecord(lam).space->space_type(), spaces::SpaceType::Global);
    // The registered global field is reused, by global() or by its spec.
    EXPECT_EQ(fields.global("lambda", "multiplier"), lam);
    auto r1 = std::make_shared<spaces::GlobalSpace>(1);
    EXPECT_EQ(fields.unknown(FieldSpec{.name = "lambda", .space = r1, .scope = FieldScope::Global}, "multiplier"),
              lam);
    // A volume field of that name is not a global field, and the reverse.
    const auto u = fields.unknown(FieldSpec{.name = "u", .space = h1}, "unknown");
    EXPECT_THROW((void)fields.global("u", "multiplier"), std::invalid_argument);
    EXPECT_THROW((void)fields.unknown(FieldSpec{.name = "lambda", .space = h1}, "unknown"), std::invalid_argument);
    EXPECT_EQ(fields.unknown(FieldSpec{.name = "u", .space = h1}, "unknown"), u);
    // The same space with another scope is another field.
    try {
        (void)fields.unknown(
            FieldSpec{.name = "u", .space = h1, .scope = FieldScope::InterfaceFace, .interface_marker = 3},
            "trace");
        ADD_FAILURE() << "a volume field was reused as an interface field";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("is a volume field, expected an interface field on marker 3"),
                  std::string::npos)
            << e.what();
    }
}

TEST(GlobalField, LoweringIsInMatrixAndTheBorderedRouteIsRefused)
{
    {
        auto problem = buildMeanConstrainedPoisson(4, 1, 0.5, 1, 1.0);
        EXPECT_EQ(problem.system->multiplierLowering(), MultiplierLowering::InMatrix);
    }
    auto mesh = buildGrid(2, 1, 1.0, 1.0, 1);
    auto h1 = quadSpace(1);
    auto r1 = std::make_shared<spaces::GlobalSpace>(1);
    FESystem sys(mesh);
    const auto u_id = sys.addField(FieldSpec{.name = "u", .space = h1});
    const auto lam_id = sys.addField(FieldSpec{.name = "lambda", .space = r1});
    FormulationBuilder F(sys, "op", FormInstallOptions{});
    const auto u = forms::StateField(u_id, *h1, "u");
    const auto v = forms::TestField(u_id, *h1, "v");
    const auto lam = forms::StateField(lam_id, *r1, "lambda");
    const auto mu = forms::TestField(lam_id, *r1, "mu");
    F.multiplierLowering(MultiplierLowering::Bordered);
    (void)F.install({u_id, lam_id}, (forms::inner(forms::grad(u), forms::grad(v)) + lam * v).dx() + (mu * u).dx());
    const auto bordered = feMessage([&] { sys.setup(); });
    EXPECT_NE(bordered.find("bordered lowering"), std::string::npos) << bordered;
    EXPECT_NE(bordered.find("LM2"), std::string::npos) << bordered;
    // A second, different request is refused; the same one is accepted.
    EXPECT_NO_THROW(sys.requestMultiplierLowering(MultiplierLowering::Bordered, "other"));
    EXPECT_THROW(sys.requestMultiplierLowering(MultiplierLowering::InMatrix, "other"), FEException);
    // Options that contradict the request are refused at setup.
    SetupOptions options;
    options.multiplier_lowering = MultiplierLowering::InMatrix;
    const auto conflict = feMessage([&] { sys.setup(options); });
    EXPECT_NE(conflict.find("asks for Bordered"), std::string::npos) << conflict;
}

TEST(GlobalField, AConstrainedMultiplierIsRefused)
{
    auto mesh = buildGrid(3, 1, 1.0, 1.0, 1);
    auto h1 = quadSpace(1);
    auto r1 = std::make_shared<spaces::GlobalSpace>(1);
    FESystem sys(mesh);
    const auto u_id = sys.addField(FieldSpec{.name = "u", .space = h1});
    const auto lam_id = sys.addField(FieldSpec{.name = "lambda", .space = r1});
    sys.addOperator("op");
    const auto u = forms::StateField(u_id, *h1, "u");
    const auto v = forms::TestField(u_id, *h1, "v");
    const auto lam = forms::StateField(lam_id, *r1, "lambda");
    const auto mu = forms::TestField(lam_id, *r1, "mu");
    (void)installFormulation(sys, "op", {u_id, lam_id},
                             (forms::inner(forms::grad(u), forms::grad(v)) + lam * v).dx() + (mu * u).dx());
    // The multiplier's DOF follows u's 8 DOFs.
    sys.addConstraint(std::make_unique<constraints::DirichletBC>(std::vector<GlobalIndex>{8}, 0.0));
    const auto message = feMessage([&] { sys.setup(); });
    EXPECT_NE(message.find("the DOF of the global field 'lambda' is constrained"), std::string::npos) << message;
}

TEST(GlobalField, AMultiplierAndAnEssentialConditionOnTheSameDofsAreRefused)
{
    // One Q1 cell: every DOF of u is on a Dirichlet boundary, so the mean
    // constraint has nothing left to act on.
    auto mesh = buildGrid(1, 1, 1.0, 1.0, 1);
    auto h1 = quadSpace(1);
    auto r1 = std::make_shared<spaces::GlobalSpace>(1);
    FESystem sys(mesh);
    const auto u_id = sys.addField(FieldSpec{.name = "u", .space = h1});
    const auto lam_id = sys.addField(FieldSpec{.name = "lambda", .space = r1});
    sys.addOperator("op");
    const auto u = forms::StateField(u_id, *h1, "u");
    const auto v = forms::TestField(u_id, *h1, "v");
    const auto lam = forms::StateField(lam_id, *r1, "lambda");
    const auto mu = forms::TestField(lam_id, *r1, "mu");
    (void)installFormulation(sys, "op", {u_id, lam_id},
                             (forms::inner(forms::grad(u), forms::grad(v)) - lam * v).dx() - (mu * u).dx() +
                                 (mu * FormExpr::constant(1.0)).dGlobal());
    const std::array<forms::bc::StrongDirichlet, 2> bcs{
        forms::bc::strongDirichlet(u_id, kLeft, FormExpr::constant(0.0), "u"),
        forms::bc::strongDirichlet(u_id, kRight, FormExpr::constant(0.0), "u")};
    installStrongDirichlet(sys, std::span<const forms::bc::StrongDirichlet>(bcs));
    const auto message = feMessage([&] { sys.setup(); });
    EXPECT_NE(message.find("every DOF the row of the multiplier 'lambda' couples to"), std::string::npos)
        << message;
}

TEST(GlobalField, AMultiplierNeedsItsRowAndItsColumn)
{
    auto mesh = buildGrid(2, 1, 1.0, 1.0, 1);
    auto h1 = quadSpace(1);
    auto r1 = std::make_shared<spaces::GlobalSpace>(1);
    FESystem sys(mesh);
    const auto u_id = sys.addField(FieldSpec{.name = "u", .space = h1});
    const auto lam_id = sys.addField(FieldSpec{.name = "lambda", .space = r1});
    sys.addOperator("op");
    const auto u = forms::StateField(u_id, *h1, "u");
    const auto v = forms::TestField(u_id, *h1, "v");
    const auto lam = forms::StateField(lam_id, *r1, "lambda");
    const auto mu = forms::TestField(lam_id, *r1, "mu");
    const auto no_row = feMessage([&] {
        (void)installFormulation(sys, "op", {u_id, lam_id},
                                 (forms::inner(forms::grad(u), forms::grad(v)) + lam * v).dx());
    });
    EXPECT_NE(no_row.find("the global field 'lambda' has no row"), std::string::npos) << no_row;
    const auto no_column = feMessage([&] {
        (void)installFormulation(sys, "op", {u_id, lam_id},
                                 (forms::inner(forms::grad(u), forms::grad(v))).dx() + (mu * u).dx());
    });
    EXPECT_NE(no_column.find("the global field 'lambda' enters no row"), std::string::npos) << no_column;
}

// ---------------------------------------------------------------------------
// Installation of a global row
// ---------------------------------------------------------------------------

TEST(GlobalRow, IsLoweredToAGlobalKernelAndMatrixOnlyElementKernels)
{
    auto problem = buildMeanConstrainedPoisson(4, 1, 0.5, 1, 1.0);
    const auto lines = kernelManifestLines(*problem.system);
    std::string manifest;
    for (const auto& line : lines) {
        manifest += line + "\n";
    }
    EXPECT_NE(manifest.find("GlobalRowKernel"), std::string::npos) << manifest;
}

TEST(GlobalRow, ResidualEntryIsTheExactMeasuredFormPlusThePointTerms)
{
    const Real c = 0.75;
    const Real height = 0.5;
    auto problem = buildMeanConstrainedPoisson(6, 2, height, 1, c);
    auto& sys = *problem.system;
    const auto n = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    const auto lam = static_cast<std::size_t>(sys.fieldDofOffset(problem.lambda));

    std::vector<Real> state(n);
    for (std::size_t i = 0; i < n; ++i) {
        state[i] = std::sin(0.7 * static_cast<Real>(i) + 0.3);
    }
    sys.constraints().distribute(state);
    const auto assembled = assembleAt(sys, state);

    // The entry: -(integral of u) + c |Omega|, the integral evaluated exactly.
    SystemStateView view;
    view.u = state;
    const auto u = forms::StateField(problem.u, *problem.u_space, "u");
    const std::array<FormExpr, 1> forms{(-u).dx()};
    const auto integral = evaluateFunctionalForms(sys, forms, view, CollectiveCommunicator{});
    EXPECT_EQ(assembled.vector[lam], integral.front().value + c * height);

    // The row is linear in u: its Jacobian row reproduces the residual change.
    std::vector<Real> direction(n, 0.0);
    for (std::size_t i = 0; i < lam; ++i) {
        direction[i] = std::cos(1.3 * static_cast<Real>(i));
    }
    sys.constraints().distributeHomogeneous(direction);
    auto shifted = state;
    for (std::size_t i = 0; i < n; ++i) {
        shifted[i] += direction[i];
    }
    const auto assembled_shifted = assembleAt(sys, shifted, /*want_matrix=*/false);
    Real predicted = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        predicted += assembled.matrix[lam * n + j] * direction[j];
    }
    EXPECT_NEAR(assembled_shifted.vector[lam] - assembled.vector[lam], predicted, 1e-13);
    EXPECT_EQ(assembled.matrix[lam * n + lam], 0.0);
}

TEST(GlobalRow, ThreadedAssemblyEqualsSerialAssembly)
{
    auto serial = buildMeanConstrainedPoisson(8, 3, 0.5, 2, 0.75);
    auto threaded = buildMeanConstrainedPoisson(8, 3, 0.5, 2, 0.75, {.threads = 4});
    const auto n = static_cast<std::size_t>(serial.system->dofHandler().getNumDofs());
    std::vector<Real> state(n);
    for (std::size_t i = 0; i < n; ++i) {
        state[i] = std::cos(0.41 * static_cast<Real>(i));
    }
    serial.system->constraints().distribute(state);
    const auto one = assembleAt(*serial.system, state);
    const auto four = assembleAt(*threaded.system, state);
    EXPECT_EQ(one.vector, four.vector);
    EXPECT_EQ(one.matrix, four.matrix);
}

TEST(GlobalRow, PointTermsWithTheMultiplierGiveTheDBlock)
{
    auto mesh = buildGrid(3, 1, 1.0, 1.0, 1);
    auto h1 = quadSpace(1);
    auto r1 = std::make_shared<spaces::GlobalSpace>(1);
    FESystem sys(mesh);
    const auto u_id = sys.addField(FieldSpec{.name = "u", .space = h1});
    const auto lam_id = sys.addField(FieldSpec{.name = "lambda", .space = r1});
    sys.addOperator("poisson");
    const auto u = forms::StateField(u_id, *h1, "u");
    const auto v = forms::TestField(u_id, *h1, "v");
    const auto lam = forms::StateField(lam_id, *r1, "lambda");
    const auto mu = forms::TestField(lam_id, *r1, "mu");
    const Real eps = 0.25;
    const auto residual = (forms::inner(forms::grad(u), forms::grad(v)) - lam * v).dx() -
                          (mu * u).dx() - (FormExpr::constant(eps) * lam * lam * mu).dGlobal() +
                          (FormExpr::constant(2.0) * mu).dGlobal();
    (void)installFormulation(sys, "poisson", {u_id, lam_id}, residual);
    sys.setup();

    const auto n = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    const auto row = static_cast<std::size_t>(sys.fieldDofOffset(lam_id));
    EXPECT_TRUE(sys.sparsity("poisson").hasEntry(static_cast<GlobalIndex>(row),
                                                 static_cast<GlobalIndex>(row)));
    std::vector<Real> state(n, 0.0);
    state[row] = 3.0;
    const auto assembled = assembleAt(sys, state);
    EXPECT_DOUBLE_EQ(assembled.vector[row], -eps * 9.0 + 2.0);
    EXPECT_DOUBLE_EQ(assembled.matrix[row * n + row], -2.0 * eps * 3.0);
}

TEST(GlobalRow, PointTermsHoldOnlyGlobalSymbols)
{
    auto mesh = buildGrid(2, 1, 1.0, 1.0, 1);
    auto h1 = quadSpace(1);
    auto r1 = std::make_shared<spaces::GlobalSpace>(1);
    FESystem sys(mesh);
    const auto u_id = sys.addField(FieldSpec{.name = "u", .space = h1});
    const auto lam_id = sys.addField(FieldSpec{.name = "lambda", .space = r1});
    sys.addOperator("op");
    const auto u = forms::StateField(u_id, *h1, "u");
    const auto v = forms::TestField(u_id, *h1, "v");
    const auto lam = forms::StateField(lam_id, *r1, "lambda");
    const auto mu = forms::TestField(lam_id, *r1, "mu");
    const auto base = (forms::inner(forms::grad(u), forms::grad(v)) + lam * v).dx() + (mu * u).dx();

    // A field that is not global inside dGlobal.
    const auto volume_field = feMessage([&] {
        (void)installFormulation(sys, "op", {u_id, lam_id}, base + (mu * u).dGlobal());
    });
    EXPECT_NE(volume_field.find("a field that is not global"), std::string::npos) << volume_field;
    // A spatial terminal inside dGlobal.
    const auto spatial = feMessage([&] {
        (void)installFormulation(sys, "op", {u_id, lam_id},
                                 base + (mu * forms::component(FormExpr::coordinate(), 0)).dGlobal());
    });
    EXPECT_NE(spatial.find("spatial or unsupported"), std::string::npos) << spatial;
    // A dGlobal term in the row of a volume field.
    const auto volume_row = feMessage([&] {
        (void)installFormulation(sys, "op", {u_id, lam_id}, base + (lam * v).dGlobal());
    });
    EXPECT_NE(volume_row.find("tests field 'u'"), std::string::npos) << volume_row;
    // dGlobal nested inside another node.
    const auto nested = feMessage([&] {
        (void)installFormulation(sys, "op", {u_id, lam_id},
                                 base + FormExpr::constant(2.0) * (mu * lam).dGlobal());
    });
    EXPECT_NE(nested.find("must be a term of the residual's sum"), std::string::npos) << nested;
    // The row of a global field through the one-block installer.
    const auto one_block = feMessage([&] {
        (void)installResidualForm(sys, "op", lam_id, u_id, (mu * u).dx());
    });
    EXPECT_NE(one_block.find("installed by installFormulation"), std::string::npos) << one_block;
}

// ---------------------------------------------------------------------------
// G1: mean-constrained Poisson
// ---------------------------------------------------------------------------

TEST(GlobalMultiplierG1, QuadraticElementsAreExact)
{
    const Real c = 1.5;
    const Real height = 0.5;
    auto problem = buildMeanConstrainedPoisson(4, 1, height, 2, c);
    auto& sys = *problem.system;
    const auto solution = solveFromZero(sys);
    const auto lam = static_cast<std::size_t>(sys.fieldDofOffset(problem.lambda));
    EXPECT_NEAR(solution[lam], 12.0 * c, 1e-11 * 12.0 * c);

    SystemStateView state;
    state.u = solution;
    for (const Real x : {0.1, 0.3, 0.5, 0.77, 0.95}) {
        for (const Real y : {0.05, 0.25, 0.45}) {
            const auto value = sys.evaluateFieldAtPoint(problem.u, state, {x, y, 0.0});
            ASSERT_TRUE(value.has_value()) << x << "," << y;
            EXPECT_NEAR((*value)[0], 6.0 * c * x * (1.0 - x), 1e-11) << x << "," << y;
        }
    }
}

TEST(GlobalMultiplierG1, BilinearElementsConvergeAtSecondOrder)
{
    const Real c = 1.5;
    const Real height = 0.5;
    std::vector<Real> errors;
    for (const int nx : {4, 8, 16, 32}) {
        auto problem = buildMeanConstrainedPoisson(nx, 1, height, 1, c);
        auto& sys = *problem.system;
        const auto solution = solveFromZero(sys);
        const Real lambda_h = solution[static_cast<std::size_t>(sys.fieldDofOffset(problem.lambda))];
        // Exact nodal values, trapezoidal mean: lambda_h = 12c / (1 - h^2).
        const Real h = 1.0 / static_cast<Real>(nx);
        EXPECT_NEAR(lambda_h, 12.0 * c / (1.0 - h * h), 1e-11 * 12.0 * c) << "nx=" << nx;
        errors.push_back(std::abs(lambda_h - 12.0 * c));
    }
    for (std::size_t i = 1; i < errors.size(); ++i) {
        const Real rate = std::log2(errors[i - 1] / errors[i]);
        EXPECT_NEAR(rate, 2.0, 0.1) << "level " << i;
    }
}

// ---------------------------------------------------------------------------
// G2: pure Neumann, mean zero
// ---------------------------------------------------------------------------

namespace {

/// One Newton step from 0, the constraints' slaves filled in from their masters.
std::vector<Real> solveConstrainedFromZero(FESystem& sys, const OperatorTag& op = "poisson")
{
    auto solution = solveFromZero(sys, op);
    sys.constraints().distribute(solution);
    return solution;
}

} // namespace

TEST(GlobalMultiplierG2, MeanZeroMultiplierMatchesTheAutomaticGauge)
{
    // u = cos(pi x) cos(pi y) on the unit square: -Delta u = 2 pi^2 u with
    // natural boundaries and mean zero.  The multiplier's row fixes the
    // constant of u, so the gauge adds nothing for u, and u_h equals the
    // solution under the automatic weighted mean-zero gauge (on P1 triangles
    // its weights are the integrals of the basis functions); lambda_h is the
    // mean of the discrete load and goes to 0.
    constexpr Real pi = 3.14159265358979323846;
    const auto f = FormExpr::coefficient("f", [](Real x, Real y, Real) {
        return 2.0 * pi * pi * std::cos(pi * x) * std::cos(pi * y);
    });
    std::vector<Real> lambdas;
    for (const int n : {4, 8, 16}) {
        auto h1 = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
        auto r1 = std::make_shared<spaces::GlobalSpace>(1);

        FESystem with_multiplier(buildTriangleGrid(n));
        const auto u_id = with_multiplier.addField(FieldSpec{.name = "u", .space = h1});
        const auto lam_id = with_multiplier.addField(FieldSpec{.name = "lambda", .space = r1});
        with_multiplier.addOperator("poisson");
        const auto u = forms::StateField(u_id, *h1, "u");
        const auto v = forms::TestField(u_id, *h1, "v");
        const auto lam = forms::StateField(lam_id, *r1, "lambda");
        const auto mu = forms::TestField(lam_id, *r1, "mu");
        (void)installFormulation(with_multiplier, "poisson", {u_id, lam_id},
                                 (forms::inner(forms::grad(u), forms::grad(v)) - f * v - lam * v).dx() -
                                     (mu * u).dx());
        with_multiplier.setup();
        EXPECT_EQ(with_multiplier.constraints().numConstraints(), 0) << "no gauge for u or lambda";
        const auto* registry = with_multiplier.gaugeRegistryIfPresent();
        if (registry != nullptr) {
            for (const auto& candidate : registry->candidates()) {
                EXPECT_NE(candidate.field, lam_id) << "a multiplier is never a gauge candidate";
            }
            for (const auto& mode : registry->resolvedModes()) {
                if (mode.candidate.field == u_id) {
                    EXPECT_EQ(mode.status, gauge::GaugeStatus::Anchored);
                    EXPECT_EQ(mode.policy, gauge::EnforcementPolicy::None);
                }
            }
        }

        FESystem gauged(buildTriangleGrid(n));
        const auto w_id = gauged.addField(FieldSpec{.name = "u", .space = h1});
        gauged.addOperator("poisson");
        const auto w = forms::StateField(w_id, *h1, "u");
        const auto z = forms::TestField(w_id, *h1, "v");
        (void)installFormulation(gauged, "poisson", {w_id},
                                 (forms::inner(forms::grad(w), forms::grad(z)) - f * z).dx());
        gauged.setup();
        EXPECT_GT(gauged.constraints().numConstraints(), 0) << "the automatic gauge fixes u";

        const auto multiplier_solution = solveFromZero(with_multiplier);
        const auto gauged_solution = solveConstrainedFromZero(gauged);
        const auto n_u = static_cast<std::size_t>(with_multiplier.fieldDofHandler(u_id).getNumDofs());
        ASSERT_EQ(gauged_solution.size(), n_u);
        Real scale = 0.0;
        for (std::size_t i = 0; i < n_u; ++i) {
            scale = std::max(scale, std::abs(gauged_solution[i]));
        }
        for (std::size_t i = 0; i < n_u; ++i) {
            EXPECT_NEAR(multiplier_solution[i], gauged_solution[i], 1e-10 * scale) << "n=" << n << " dof " << i;
        }
        lambdas.push_back(std::abs(multiplier_solution[n_u]));
    }
    // lambda_h -> 0 (the discrete load's mean, O(h^2)).
    EXPECT_LT(lambdas[1], lambdas[0]);
    EXPECT_LT(lambdas[2], lambdas[1]);
    EXPECT_LT(lambdas[2], 0.05);
}

// ---------------------------------------------------------------------------
// G3: Taylor-Hood Stokes with a mean-zero pressure multiplier
// ---------------------------------------------------------------------------

namespace {

struct StokesSystem {
    std::unique_ptr<FESystem> system;
    FieldId u{INVALID_FIELD_ID};
    FieldId p{INVALID_FIELD_ID};
    FieldId lambda{INVALID_FIELD_ID};
};

/**
 * P2/P1 Stokes on the unit square with u = 0 on the boundary and the forcing of
 * u = curl(sin^2(pi x) sin^2(pi y)), p = cos(pi x) cos(pi y):
 * (grad u : grad v - p div v - q div u - f . v) dx, plus, with the multiplier,
 * (lambda q) dx + (mu p) dx (mean-zero pressure).  Without the multiplier the
 * first pressure DOF is pinned instead: the automatic gauge does not fix the
 * pressure of this enclosed flow (its integration-by-parts evidence takes
 * boundaryDofsByMarker's P2 DOFs, some of which are interior, for natural
 * boundaries).
 */
StokesSystem buildStokes(int n, bool multiplier)
{
    constexpr Real pi = 3.14159265358979323846;
    StokesSystem out;
    auto p2 = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 2);
    auto velocity_space = std::make_shared<spaces::ProductSpace>(p2, 2);
    auto pressure_space = std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1);
    auto r1 = std::make_shared<spaces::GlobalSpace>(1);
    out.system = std::make_unique<FESystem>(buildTriangleGrid(n));
    auto& sys = *out.system;
    out.u = sys.addField(FieldSpec{.name = "u", .space = velocity_space, .components = 2});
    out.p = sys.addField(FieldSpec{.name = "p", .space = pressure_space});
    if (multiplier) {
        out.lambda = sys.addField(FieldSpec{.name = "lambda", .space = r1});
    }
    sys.addOperator("stokes");
    const auto f = FormExpr::coefficient("f", [](Real x, Real y, Real) {
        const Real two_pi_cubed = 2.0 * pi * pi * pi;
        return std::array<Real, 3>{
            -two_pi_cubed * std::sin(2.0 * pi * y) * (2.0 * std::cos(2.0 * pi * x) - 1.0) -
                pi * std::sin(pi * x) * std::cos(pi * y),
            two_pi_cubed * std::sin(2.0 * pi * x) * (2.0 * std::cos(2.0 * pi * y) - 1.0) -
                pi * std::cos(pi * x) * std::sin(pi * y),
            0.0};
    });
    const auto u = forms::StateField(out.u, *velocity_space, "u");
    const auto v = forms::TestField(out.u, *velocity_space, "v");
    const auto p = forms::StateField(out.p, *pressure_space, "p");
    const auto q = forms::TestField(out.p, *pressure_space, "q");
    auto residual = (forms::inner(forms::grad(u), forms::grad(v)) - p * forms::div(v) - forms::inner(f, v)).dx() -
                    (q * forms::div(u)).dx();
    if (multiplier) {
        const auto lam = forms::StateField(out.lambda, *r1, "lambda");
        const auto mu = forms::TestField(out.lambda, *r1, "mu");
        residual = residual + (lam * q).dx() + (mu * p).dx();
        (void)installFormulation(sys, "stokes", {out.u, out.p, out.lambda}, residual);
    } else {
        (void)installFormulation(sys, "stokes", {out.u, out.p}, residual);
    }
    std::vector<forms::bc::StrongDirichlet> bcs;
    for (const int marker : {kLeft, kRight, kBottom, kTop}) {
        for (const int component : {0, 1}) {
            bcs.push_back(forms::bc::strongDirichlet(out.u, marker, FormExpr::constant(0.0), "u", component));
        }
    }
    installStrongDirichlet(sys, std::span<const forms::bc::StrongDirichlet>(bcs));
    if (!multiplier) {
        // The pressure DOFs follow the 2 (2n + 1)^2 velocity DOFs.
        const auto first_pressure = static_cast<GlobalIndex>(2 * (2 * n + 1) * (2 * n + 1));
        sys.addConstraint(std::make_unique<constraints::DirichletBC>(std::vector<GlobalIndex>{first_pressure}, 0.0));
    }
    sys.setup();
    if (!multiplier) {
        EXPECT_EQ(sys.fieldDofOffset(out.p), static_cast<GlobalIndex>(2 * (2 * n + 1) * (2 * n + 1)));
    }
    return out;
}

} // namespace

TEST(GlobalMultiplierG3, MeanZeroPressureMultiplierMatchesAPinnedPressure)
{
    for (const int n : {4, 8}) {
        auto with_multiplier = buildStokes(n, /*multiplier=*/true);
        auto pinned = buildStokes(n, /*multiplier=*/false);

        // The multiplier's row fixes the pressure constant: no constraint on
        // p or lambda.
        auto& sys = *with_multiplier.system;
        const auto p_offset = sys.fieldDofOffset(with_multiplier.p);
        const auto p_count = sys.fieldDofHandler(with_multiplier.p).getNumDofs();
        for (GlobalIndex d = p_offset; d < p_offset + p_count + 1; ++d) {
            EXPECT_FALSE(sys.constraints().isConstrained(d)) << "n=" << n << " dof " << d;
        }

        const auto a = solveFromZero(sys, "stokes");
        const auto b = solveConstrainedFromZero(*pinned.system, "stokes");
        const auto n_u = static_cast<std::size_t>(sys.fieldDofHandler(with_multiplier.u).getNumDofs());
        const auto n_p = static_cast<std::size_t>(p_count);
        Real velocity_scale = 0.0;
        for (std::size_t i = 0; i < n_u; ++i) {
            velocity_scale = std::max(velocity_scale, std::abs(b[i]));
        }
        for (std::size_t i = 0; i < n_u; ++i) {
            EXPECT_NEAR(a[i], b[i], 1e-10 * velocity_scale) << "n=" << n << " velocity dof " << i;
        }
        // The pressures differ by a constant only.
        Real lowest = std::numeric_limits<Real>::max();
        Real highest = std::numeric_limits<Real>::lowest();
        Real pressure_scale = 0.0;
        for (std::size_t i = n_u; i < n_u + n_p; ++i) {
            lowest = std::min(lowest, a[i] - b[i]);
            highest = std::max(highest, a[i] - b[i]);
            pressure_scale = std::max(pressure_scale, std::abs(a[i]));
        }
        EXPECT_LT(highest - lowest, 1e-9 * pressure_scale) << "n=" << n;
        // The multiplier's pressure has zero integral, and the multiplier is 0:
        // the divergence of a velocity that vanishes on the boundary integrates
        // to 0.
        SystemStateView state;
        state.u = a;
        const auto p = forms::StateField(with_multiplier.p, *sys.fieldRecord(with_multiplier.p).space, "p");
        const std::array<FormExpr, 1> mean{p.dx()};
        const auto integral = evaluateFunctionalForms(sys, mean, state, CollectiveCommunicator{});
        EXPECT_LT(std::abs(integral.front().value), 1e-12 * pressure_scale) << "n=" << n;
        EXPECT_LT(std::abs(a[n_u + n_p]), 1e-10 * pressure_scale) << "n=" << n;
    }
}

} // namespace svmp::FE::systems::test_global_multiplier
