/**
 * @file test_GlobalMultiplier.cpp
 * @brief Global multipliers as forms: the global field, its rows and the G1/G2 cases
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
 * G2 (pure Neumann, mean zero) needs the multiplier's anchoring evidence in the
 * gauge registry (step LM1b) and stays disabled.
 *
 * The cases are written against the target API of decision D54 (GlobalSpace,
 * FieldScope::Global, dGlobal()).  Step LM1a adds that API and builds this
 * file; until then it is not compiled.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Constraints/AffineConstraints.h"
#include "Core/FEException.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/BoundaryConditions.h"
#include "Forms/Vocabulary.h"
#include "Spaces/GlobalSpace.h"
#include "Spaces/H1Space.h"
#include "Sparsity/SparsityPattern.h"
#include "Systems/FESystem.h"
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

Assembled assembleAt(FESystem& sys, std::span<const Real> u, bool want_matrix = true)
{
    const auto n = sys.dofHandler().getNumDofs();
    assembly::DenseSystemView out(n);
    out.zero();
    SystemStateView state;
    state.u = u;
    AssemblyRequest request;
    request.op = "poisson";
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
std::vector<Real> solveFromZero(FESystem& sys)
{
    const auto n = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    std::vector<Real> zero(n, 0.0);
    const auto assembled = assembleAt(sys, zero);
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
// G1: mean-constrained Poisson
// ---------------------------------------------------------------------------

TEST(GlobalMultiplierG1, DISABLED_QuadraticElementsAreExact)
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

TEST(GlobalMultiplierG1, DISABLED_BilinearElementsConvergeAtSecondOrder)
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
// G2: pure Neumann, mean zero (needs the multiplier's gauge evidence, LM1b)
// ---------------------------------------------------------------------------

TEST(GlobalMultiplierG2, DISABLED_MeanZeroMultiplierMatchesTheAutomaticGauge)
{
    // u = cos(pi x) cos(pi y) on the unit square, -Delta u = 2 pi^2 u, natural
    // boundaries, mean zero.  With the multiplier, u_h equals the solution of
    // the automatic mean-zero gauge, and lambda_h -> 0.
    constexpr Real pi = 3.14159265358979323846;
    const auto f = FormExpr::coefficient("f", [](Real x, Real y, Real) {
        return 2.0 * pi * pi * std::cos(pi * x) * std::cos(pi * y);
    });
    for (const int n : {8, 16}) {
        auto mesh = buildGrid(n, n, 1.0, 1.0, 1);
        auto h1 = quadSpace(1);
        auto r1 = std::make_shared<spaces::GlobalSpace>(1);

        FESystem with_multiplier(mesh);
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

        auto mesh_gauge = buildGrid(n, n, 1.0, 1.0, 1);
        FESystem gauged(mesh_gauge);
        const auto w_id = gauged.addField(FieldSpec{.name = "u", .space = h1});
        gauged.addOperator("poisson");
        const auto w = forms::StateField(w_id, *h1, "u");
        const auto z = forms::TestField(w_id, *h1, "v");
        (void)installFormulation(gauged, "poisson", {w_id},
                                 (forms::inner(forms::grad(w), forms::grad(z)) - f * z).dx());
        gauged.setup();

        const auto multiplier_solution = solveFromZero(with_multiplier);
        const auto gauged_solution = solveFromZero(gauged);
        const auto n_u = static_cast<std::size_t>(with_multiplier.fieldDofHandler(u_id).getNumDofs());
        for (std::size_t i = 0; i < n_u; ++i) {
            EXPECT_NEAR(multiplier_solution[i], gauged_solution[i], 1e-10);
        }
        EXPECT_LT(std::abs(multiplier_solution[n_u]), 1.0 / static_cast<Real>(n));
    }
}

} // namespace svmp::FE::systems::test_global_multiplier
