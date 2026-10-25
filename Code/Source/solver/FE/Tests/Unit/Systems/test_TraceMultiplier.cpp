/**
 * @file test_TraceMultiplier.cpp
 * @brief Trace multipliers written as restricted volume fields (D54 LM3, T1)
 *
 * B1 (weak Dirichlet, MMS): -Delta u = f on the unit square with
 * u = sin(pi x) e^y on the boundary, imposed through one multiplier per side
 * restricted to that side (bc::LagrangeDirichletBC).  P2 u with P1
 * lambda (quadrilaterals and triangles): the L2 error of u is O(h^3) and
 * lambda tends to -du/dn on the boundary at second order.  P1 u with P1
 * lambda and the Barbosa-Hughes stabilization: the L2 error of u is O(h^2).
 *
 * B2: with matching full-trace spaces (P1 u, P1 lambda on the whole boundary)
 * and boundary data in the trace space, u_h equals the strong-Dirichlet
 * solution, and B^T lambda_h is minus its reaction (K u - F on the boundary
 * rows).
 *
 * Setup: the multiplier's rows anchor the constant of u (no automatic gauge),
 * a multiplier that couples only to DOFs an essential condition fixes is
 * refused, the condition is exclusive on its marker, and an unstabilized
 * multiplier is reported for the FSILS refusal.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/ConstraintPlan.h"
#include "Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "Constraints/SupportRestrictionConstraint.h"
#include "Core/FEException.h"
#include "Forms/BoundaryConditions.h"
#include "Forms/LagrangeDirichletBC.h"
#include "Forms/StandardBCs.h"
#include "Forms/Vocabulary.h"
#include "Systems/BoundaryConditionManager.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/FunctionalFormEvaluator.h"
#include "Systems/SystemCollectives.h"

#include "TraceMultiplierTestMeshes.h"

#include "Mesh/Fields/MeshFields.h"

#if FE_HAS_EIGEN
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::test_trace_multiplier {
namespace {

using forms::FormExpr;
using systems::FESystem;
using systems::FieldSpec;

/// The message of the exception @p f throws (empty if it throws none).
template <typename F>
std::string messageOf(F&& f)
{
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}

} // namespace

// ---------------------------------------------------------------------------
// Setup: anchoring and refusals
// ---------------------------------------------------------------------------

TEST(TraceMultiplierSetup, TheMultiplierAnchorsTheConstantOfU)
{
    // -Delta u = 1 with u = 0 on the boundary only through the multiplier's
    // rows: no gauge constraint is added, so the only constrained DOFs are the
    // multiplier's off the boundary.
    auto mesh = buildQuadGrid(3, 3, 1.0, 1.0, 1, kWholeBoundary);
    auto h1 = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    FESystem sys(mesh);
    const auto u_id = sys.addField(FieldSpec{.name = "u", .space = h1});
    const auto lam_id = sys.addField(FieldSpec{.name = "lambda", .space = h1});
    sys.addOperator("op");
    const auto u = forms::StateField(u_id, *h1, "u");
    const auto v = forms::TestField(u_id, *h1, "v");
    const auto lam = forms::StateField(lam_id, *h1, "lambda");
    const auto mu = forms::TestField(lam_id, *h1, "mu");
    (void)systems::installFormulation(
        sys, "op", {u_id, lam_id},
        (forms::inner(forms::grad(u), forms::grad(v)) - FormExpr::constant(1.0) * v).dx() +
            (lam * v).ds(kWholeBoundary) + (mu * u).ds(kWholeBoundary));
    constraints::ConstraintPlan plan(sys);
    plan.restrictSupport({lam_id}, forms::BoundaryDomain::fullPhysical(kWholeBoundary));
    plan.commit();
    sys.setup();

    const auto offset_u = sys.fieldDofOffset(u_id);
    for (GlobalIndex d = 0; d < sys.fieldDofHandler(u_id).getNumDofs(); ++d) {
        EXPECT_FALSE(sys.constraints().isConstrained(offset_u + d)) << "u dof " << d;
    }
    // The 4 interior vertices of the 3 x 3 grid carry no multiplier.
    std::size_t pinned = 0;
    const auto offset_l = sys.fieldDofOffset(lam_id);
    for (GlobalIndex d = 0; d < sys.fieldDofHandler(lam_id).getNumDofs(); ++d) {
        pinned += sys.constraints().isConstrained(offset_l + d) ? 1u : 0u;
    }
    EXPECT_EQ(pinned, 4u);
    EXPECT_EQ(sys.traceMultiplierFields(), std::vector<FieldId>{lam_id});
    // The evidence comes from the multiplier's rows; the multiplier itself is
    // no gauge candidate.
    bool anchored_by_rows = false;
    for (const auto& evidence : sys.gaugeRegistry().anchoring()) {
        anchored_by_rows = anchored_by_rows ||
                           (evidence.field == u_id &&
                            evidence.source.find("trace multiplier 'lambda': its rows fix the constant") !=
                                std::string::npos);
    }
    EXPECT_TRUE(anchored_by_rows);
    for (const auto& candidate : sys.gaugeRegistry().candidates()) {
        EXPECT_NE(candidate.field, lam_id);
    }
}

TEST(TraceMultiplierSetup, AMultiplierOnDofsAnEssentialConditionFixesIsRefused)
{
    // lambda restricted to the left boundary, where u also has an essential
    // condition: every DOF lambda's rows couple to is fixed.
    auto mesh = buildQuadGrid(3, 2, 1.0, 1.0, 1);
    auto h1 = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    FESystem sys(mesh);
    const auto u_id = sys.addField(FieldSpec{.name = "u", .space = h1});
    const auto lam_id = sys.addField(FieldSpec{.name = "lambda", .space = h1});
    sys.addOperator("op");
    const auto u = forms::StateField(u_id, *h1, "u");
    const auto v = forms::TestField(u_id, *h1, "v");
    const auto lam = forms::StateField(lam_id, *h1, "lambda");
    const auto mu = forms::TestField(lam_id, *h1, "mu");
    (void)systems::installFormulation(
        sys, "op", {u_id, lam_id},
        (forms::inner(forms::grad(u), forms::grad(v))).dx() + (lam * v).ds(kLeft) + (mu * u).ds(kLeft));
    constraints::ConstraintPlan plan(sys);
    plan.restrictSupport({lam_id}, forms::BoundaryDomain::fullPhysical(kLeft));
    plan.essentials([&](FESystem& s) {
        const std::array<forms::bc::StrongDirichlet, 1> bcs{
            forms::bc::strongDirichlet(u_id, kLeft, FormExpr::constant(0.0), "u")};
        systems::installStrongDirichlet(s, std::span<const forms::bc::StrongDirichlet>(bcs));
    });
    plan.commit();
    const auto message = messageOf([&] { sys.setup(); });
    EXPECT_NE(message.find("of the trace multiplier 'lambda' (restricted to physical_boundary 1) couple only to "
                           "fixed degrees of freedom of 'u'"),
              std::string::npos)
        << message;
}

namespace {

constexpr Real kPi = 3.14159265358979323846;

Real exactU(Real x, Real y) { return std::sin(kPi * x) * std::exp(y); }

/// The assembled residual and Jacobian at @p state (constraints applied, the
/// constrained rows homogeneous, as Newton assembles them).
struct Assembled {
    std::vector<Real> matrix;
    std::vector<Real> vector;
};

Assembled assembleAt(FESystem& sys, std::span<const Real> state, const systems::OperatorTag& op = "op")
{
    const auto n = sys.dofHandler().getNumDofs();
    assembly::DenseSystemView out(n);
    out.zero();
    systems::SystemStateView view;
    view.u = state;
    systems::AssemblyRequest request;
    request.op = op;
    request.want_matrix = true;
    request.want_vector = true;
    request.suppress_constraint_inhomogeneity = true;
    const auto result = sys.assemble(request, view, &out, &out);
    EXPECT_TRUE(result.success);
    return {std::vector<Real>(out.matrixData().begin(), out.matrixData().end()),
            std::vector<Real>(out.vectorData().begin(), out.vectorData().end())};
}

/// Solves A x = b (sparse LU with pivoting: the saddle-point diagonal has zeros).
std::vector<Real> solve(const std::vector<Real>& a, const std::vector<Real>& b)
{
#if FE_HAS_EIGEN
    const auto n = static_cast<Eigen::Index>(b.size());
    std::vector<Eigen::Triplet<Real>> entries;
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            const auto v = a[static_cast<std::size_t>(i * n + j)];
            if (v != Real(0.0)) {
                entries.emplace_back(i, j, v);
            }
        }
    }
    Eigen::SparseMatrix<Real> A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    Eigen::SparseLU<Eigen::SparseMatrix<Real>> lu;
    lu.compute(A);
    EXPECT_EQ(lu.info(), Eigen::Success) << "singular saddle-point matrix";
    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> rhs(b.data(), n);
    const Eigen::Matrix<Real, Eigen::Dynamic, 1> x = lu.solve(rhs);
    return std::vector<Real>(x.data(), x.data() + n);
#else
    (void)a;
    ADD_FAILURE() << "these tests need Eigen";
    return std::vector<Real>(b.size(), 0.0);
#endif
}

/// The solution of the (linear) problem: one Newton step from the
/// constraint values.
std::vector<Real> solveLinear(FESystem& sys, const systems::OperatorTag& op = "op")
{
    std::vector<Real> state(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    sys.constraints().distribute(state);
    const auto assembled = assembleAt(sys, state, op);
    std::vector<Real> rhs(assembled.vector.size());
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] = -assembled.vector[i];
    }
    const auto step = solve(assembled.matrix, rhs);
    for (std::size_t i = 0; i < state.size(); ++i) {
        state[i] += step[i];
    }
    // The constrained rows are homogeneous (as Newton assembles them): the
    // constraint values come back with the update.
    sys.constraints().distribute(state);
    return state;
}

FormExpr coefficient(std::string name, std::function<Real(Real, Real)> f)
{
    return FormExpr::coefficient(std::move(name),
                                 forms::ScalarCoefficient([f](Real x, Real y, Real) { return f(x, y); }));
}

/// -Delta u = f, u = g on the boundary through one multiplier per boundary
/// marker (each restricted to its boundary, bc::LagrangeDirichletBC).
struct WeakDirichletPoisson {
    std::unique_ptr<FESystem> system;
    std::shared_ptr<spaces::H1Space> u_space;
    std::shared_ptr<spaces::H1Space> lambda_space;
    FieldId u{INVALID_FIELD_ID};
    std::vector<FieldId> lambdas;
    std::vector<int> markers;
    FieldId lambda{INVALID_FIELD_ID}; ///< the first multiplier
};

WeakDirichletPoisson buildWeakDirichletPoisson(std::shared_ptr<svmp::Mesh> mesh,
                                               std::shared_ptr<spaces::H1Space> u_space,
                                               std::shared_ptr<spaces::H1Space> lambda_space,
                                               const FormExpr& f,
                                               const FormExpr& g,
                                               Real beta = Real(0.0),
                                               std::vector<int> markers = {kWholeBoundary})
{
    WeakDirichletPoisson out;
    out.system = std::make_unique<FESystem>(std::move(mesh));
    out.u_space = std::move(u_space);
    out.lambda_space = std::move(lambda_space);
    out.markers = std::move(markers);
    auto& sys = *out.system;
    out.u = sys.addField(FieldSpec{.name = "u", .space = out.u_space});
    for (const int marker : out.markers) {
        out.lambdas.push_back(
            sys.addField(FieldSpec{.name = "lambda_" + std::to_string(marker), .space = out.lambda_space}));
    }
    out.lambda = out.lambdas.front();
    sys.addOperator("op");
    const auto u = forms::StateField(out.u, *out.u_space, "u");
    const auto v = forms::TestField(out.u, *out.u_space, "v");
    auto residual = (forms::inner(forms::grad(u), forms::grad(v)) - f * v).dx();
    systems::BoundaryConditionManager bcs;
    for (std::size_t k = 0; k < out.markers.size(); ++k) {
        bcs.add(forms::bc::makeLagrangeDirichletBC(out.markers[k], g, out.lambdas[k], beta));
    }
    bcs.applyAll(sys, residual, u, v, out.u);
    std::vector<FieldId> fields{out.u};
    fields.insert(fields.end(), out.lambdas.begin(), out.lambdas.end());
    (void)systems::installFormulation(sys, "op", fields, residual);
    sys.setup();
    return out;
}

struct Errors {
    Real u{0.0};      ///< L2 error of u
    Real lambda{0.0}; ///< L2(boundary) error of the multipliers against -du/dn
};

Errors b1Errors(const WeakDirichletPoisson& problem, const std::vector<Real>& state)
{
    auto& sys = *problem.system;
    const auto u = forms::StateField(problem.u, *problem.u_space, "u");
    const auto uex = coefficient("u_exact", exactU);
    const auto grad_uex = FormExpr::coefficient(
        "grad_u_exact", forms::VectorCoefficient([](Real x, Real y, Real) {
            return std::array<Real, 3>{kPi * std::cos(kPi * x) * std::exp(y), std::sin(kPi * x) * std::exp(y),
                                       Real(0.0)};
        }));
    const auto flux = forms::inner(grad_uex, FormExpr::normal());
    std::vector<FormExpr> forms_{((u - uex) * (u - uex)).dx()};
    for (std::size_t k = 0; k < problem.lambdas.size(); ++k) {
        const auto lam = forms::StateField(problem.lambdas[k], *problem.lambda_space, "lambda");
        forms_.push_back(((lam + flux) * (lam + flux)).ds(problem.markers[k]));
    }
    systems::SystemStateView view;
    view.u = state;
    const auto values = systems::evaluateFunctionalForms(sys, forms_, view, systems::CollectiveCommunicator{});
    Real lambda_sq = 0.0;
    for (std::size_t k = 1; k < values.size(); ++k) {
        lambda_sq += values[k].value;
    }
    return {std::sqrt(values[0].value), std::sqrt(lambda_sq)};
}

struct Family {
    std::string name;
    /// The n x n mesh; label -1: the four sides labelled apart, else every
    /// boundary face carries the label.
    std::function<std::shared_ptr<svmp::Mesh>(int, int)> mesh;
    std::function<std::shared_ptr<spaces::H1Space>(int)> space;
};

std::vector<Family> families()
{
    return {
        Family{"quadrilaterals", [](int n, int label) { return buildQuadGrid(n, n, 1.0, 1.0, 1, label); },
               [](int order) { return std::make_shared<spaces::H1Space>(ElementType::Quad4, order); }},
        Family{"triangles", [](int n, int label) { return buildTriangleGrid(n, label); },
               [](int order) { return triangleSpace(order); }},
    };
}

/// One multiplier per side: the flux -du/dn jumps at the corners, which a
/// multiplier continuous around a corner cannot follow (its error would be
/// O(h^1/2) and would pollute u).
const std::vector<int> kSides{kLeft, kRight, kBottom, kTop};

FormExpr b1Source()
{
    return coefficient("f", [](Real x, Real y) { return (kPi * kPi - Real(1.0)) * exactU(x, y); });
}

} // namespace

// ---------------------------------------------------------------------------
// B1: weak Dirichlet through a multiplier, manufactured solution
// ---------------------------------------------------------------------------

TEST(TraceMultiplierB1, P2WithP1MultiplierConvergesAtThirdOrder)
{
    for (const auto& family : families()) {
        SCOPED_TRACE(family.name);
        std::vector<Errors> errors;
        for (const int n : {4, 8, 16}) {
            auto problem = buildWeakDirichletPoisson(family.mesh(n, -1), family.space(2), family.space(1), b1Source(),
                                                     coefficient("g", exactU), Real(0.0), kSides);
            const auto state = solveLinear(*problem.system);
            errors.push_back(b1Errors(problem, state));
        }
        for (std::size_t k = 1; k < errors.size(); ++k) {
            const Real rate_u = std::log2(errors[k - 1].u / errors[k].u);
            const Real rate_lambda = std::log2(errors[k - 1].lambda / errors[k].lambda);
            std::cout << "[ B1 " << family.name << " P2/P1 ] e_u=" << errors[k].u << " rate_u=" << rate_u
                      << " e_lambda=" << errors[k].lambda << " rate_lambda=" << rate_lambda << std::endl;
            EXPECT_GT(rate_u, 2.8);
            EXPECT_GT(rate_lambda, 1.8);
        }
    }
}

TEST(TraceMultiplierB1, StabilizedP1WithP1MultiplierConvergesAtSecondOrder)
{
    for (const auto& family : families()) {
        SCOPED_TRACE(family.name);
        std::vector<Errors> errors;
        for (const int n : {8, 16, 32}) {
            auto problem = buildWeakDirichletPoisson(family.mesh(n, -1), family.space(1), family.space(1), b1Source(),
                                                     coefficient("g", exactU), /*beta=*/Real(0.1), kSides);
            const auto state = solveLinear(*problem.system);
            errors.push_back(b1Errors(problem, state));
        }
        for (std::size_t k = 1; k < errors.size(); ++k) {
            const Real rate_u = std::log2(errors[k - 1].u / errors[k].u);
            const Real rate_lambda = std::log2(errors[k - 1].lambda / errors[k].lambda);
            std::cout << "[ B1 " << family.name << " P1/P1 Barbosa-Hughes ] e_u=" << errors[k].u
                      << " rate_u=" << rate_u << " e_lambda=" << errors[k].lambda << " rate_lambda=" << rate_lambda
                      << std::endl;
            EXPECT_GT(rate_u, 1.8);
            EXPECT_GT(rate_lambda, 1.5);
        }
    }
}

// ---------------------------------------------------------------------------
// B2: the multiplier is the reaction of the strong-Dirichlet solve
// ---------------------------------------------------------------------------

TEST(TraceMultiplierB2, MultiplierIsTheReactionOfTheStrongDirichletSolve)
{
    // g = 1 + x + 2y is linear on every boundary edge (in the P1 trace space).
    const auto g = coefficient("g", [](Real x, Real y) { return Real(1.0) + x + Real(2.0) * y; });
    const auto f = coefficient("f", [](Real x, Real y) { return Real(1.0) + x * y; });
    for (const auto& family : families()) {
        SCOPED_TRACE(family.name);
        const int n = 6;
        auto weak = buildWeakDirichletPoisson(family.mesh(n, kWholeBoundary), family.space(1), family.space(1), f, g);
        const auto weak_state = solveLinear(*weak.system);

        // The strong-Dirichlet solve of the same problem (same mesh and space:
        // the u DOFs are numbered alike, first).
        FESystem strong(family.mesh(n, kWholeBoundary));
        auto space = family.space(1);
        const auto u_id = strong.addField(FieldSpec{.name = "u", .space = space});
        strong.addOperator("op");
        const auto u = forms::StateField(u_id, *space, "u");
        const auto v = forms::TestField(u_id, *space, "v");
        (void)systems::installFormulation(strong, "op", {u_id},
                                          (forms::inner(forms::grad(u), forms::grad(v)) - f * v).dx());
        const std::array<forms::bc::StrongDirichlet, 1> bcs{forms::bc::strongDirichlet(u_id, kWholeBoundary, g, "u")};
        systems::installStrongDirichlet(strong, std::span<const forms::bc::StrongDirichlet>(bcs));
        strong.setup();
        const auto strong_state = solveLinear(strong);

        const auto n_u = static_cast<std::size_t>(strong.dofHandler().getNumDofs());
        ASSERT_EQ(static_cast<std::size_t>(weak.system->fieldDofOffset(weak.lambda)), n_u);
        Real scale = 0.0;
        for (std::size_t i = 0; i < n_u; ++i) {
            scale = std::max(scale, std::abs(strong_state[i]));
        }
        for (std::size_t i = 0; i < n_u; ++i) {
            EXPECT_NEAR(weak_state[i], strong_state[i], 1e-10 * scale) << "u dof " << i;
        }

        // The u rows of the multiplier system: K u - F + B^T lambda.  At
        // (u_s, 0) they are the strong solve's reaction r (zero off the
        // boundary), at (0, lambda_h) minus at (0, 0) they are B^T lambda_h.
        const auto rows = [&](const std::vector<Real>& state) {
            auto assembled = assembleAt(*weak.system, state);
            assembled.vector.resize(n_u);
            return assembled.vector;
        };
        std::vector<Real> at_strong(weak_state.size(), 0.0);
        std::copy(strong_state.begin(), strong_state.end(), at_strong.begin());
        std::vector<Real> at_lambda(weak_state.size(), 0.0);
        std::copy(weak_state.begin() + static_cast<std::ptrdiff_t>(n_u), weak_state.end(),
                  at_lambda.begin() + static_cast<std::ptrdiff_t>(n_u));
        const std::vector<Real> zero(weak_state.size(), 0.0);
        const auto reaction = rows(at_strong);
        const auto bt_lambda = rows(at_lambda);
        const auto source = rows(zero);
        Real reaction_scale = 0.0;
        for (std::size_t i = 0; i < n_u; ++i) {
            reaction_scale = std::max(reaction_scale, std::abs(reaction[i]));
        }
        ASSERT_GT(reaction_scale, 0.0);
        for (std::size_t i = 0; i < n_u; ++i) {
            EXPECT_NEAR(reaction[i], -(bt_lambda[i] - source[i]), 1e-9 * reaction_scale) << "u dof " << i;
        }
    }
}

// ---------------------------------------------------------------------------
// The condition on its marker, and the solvers
// ---------------------------------------------------------------------------

TEST(TraceMultiplierSetup, TheConditionIsExclusiveOnItsBoundary)
{
    auto mesh = buildQuadGrid(2, 2, 1.0, 1.0, 1);
    auto h1 = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    FESystem sys(mesh);
    const auto u_id = sys.addField(FieldSpec{.name = "u", .space = h1});
    const auto lam_id = sys.addField(FieldSpec{.name = "lambda", .space = h1});
    sys.addOperator("op");
    const auto u = forms::StateField(u_id, *h1, "u");
    const auto v = forms::TestField(u_id, *h1, "v");
    auto residual = (forms::inner(forms::grad(u), forms::grad(v))).dx();
    systems::BoundaryConditionManager bcs;
    bcs.add(forms::bc::makeLagrangeDirichletBC(kLeft, FormExpr::constant(0.0), lam_id));
    bcs.add(std::make_unique<forms::bc::EssentialBC>(kLeft, FormExpr::constant(0.0)));
    EXPECT_THROW(bcs.applyAll(sys, residual, u, v, u_id), std::invalid_argument);
}

TEST(TraceMultiplierSetup, AnUnstabilizedMultiplierIsReportedForTheIterativeSolvers)
{
    const auto quad1 = [] { return std::make_shared<spaces::H1Space>(ElementType::Quad4, 1); };
    auto plain = buildWeakDirichletPoisson(buildQuadGrid(2, 2, 1.0, 1.0, 1, kWholeBoundary), quad1(), quad1(),
                                           FormExpr::constant(1.0), FormExpr::constant(0.0));
    EXPECT_EQ(plain.system->unstabilizedTraceMultiplierFields(), std::vector<FieldId>{plain.lambda});
    auto stabilized = buildWeakDirichletPoisson(buildQuadGrid(2, 2, 1.0, 1.0, 1, kWholeBoundary), quad1(), quad1(),
                                                FormExpr::constant(1.0), FormExpr::constant(0.0), Real(0.1));
    EXPECT_TRUE(stabilized.system->unstabilizedTraceMultiplierFields().empty());
}

} // namespace svmp::FE::test_trace_multiplier
