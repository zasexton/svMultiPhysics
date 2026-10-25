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
 * I3 (cut interface): a circle of radius 0.5 in a square of side 2, k1 = 1
 * inside and k2 = 10 outside, u = r^2 / k_i + c_i (continuous), f = -4, so
 * the flux k du/dn on the circle is 2r = 1.  Two phase fields on the
 * background mesh (each pinned off its side) are glued by a multiplier
 * restricted to the cut cells, stabilized as Barbosa-Hughes on the interface;
 * the result is compared with the symmetric Nitsche coupling of the same
 * fields.  The Analysis subsystem gives no inf-sup verdict for this pair: it
 * classifies it as TraceMultiplier, but builds no Fortin projection for trace
 * pairs and has no numerical inf-sup estimator (AssemblyAccess::infSupEstimate
 * forwards a supplied summary only).
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

// ---------------------------------------------------------------------------
// I3: a cut interface glued by a stabilized multiplier, against Nitsche
// ---------------------------------------------------------------------------

namespace {

constexpr int kInterface = 41;
constexpr Real kInner = 1.0;  ///< k1, inside the circle
constexpr Real kOuter = 10.0; ///< k2, outside
constexpr Real kRadius = 0.5;

Real radius(Real x, Real y) { return std::sqrt((x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0)); }
Real exactInner(Real x, Real y) { return radius(x, y) * radius(x, y) / kInner + kRadius * kRadius * (1.0 / kOuter - 1.0 / kInner); }
Real exactOuter(Real x, Real y) { return radius(x, y) * radius(x, y) / kOuter; }

struct CutProblem {
    std::shared_ptr<svmp::Mesh> mesh;
    std::unique_ptr<FESystem> system;
    std::shared_ptr<spaces::H1Space> space;
    FieldId u1{INVALID_FIELD_ID};
    FieldId u2{INVALID_FIELD_ID};
    FieldId lambda{INVALID_FIELD_ID};
};

/// u1 inside, u2 outside, coupled on the circle by a multiplier (@p multiplier)
/// or by symmetric Nitsche.
CutProblem buildCutProblem(int n, bool multiplier, Real beta)
{
    CutProblem out;
    out.mesh = buildQuadGrid(n, n, 2.0, 2.0, 1, kWholeBoundary);
    const auto phi = [](Real x, Real y) { return radius(x, y) - kRadius; };
    {
        auto& base = out.mesh->local_mesh();
        const auto handle = svmp::MeshFields::attach_field(base, svmp::EntityKind::Vertex, "phi",
                                                           svmp::FieldScalarType::Float64, 1);
        auto* values = svmp::MeshFields::field_data_as<svmp::real_t>(base, handle);
        for (std::size_t v = 0; v < base.n_vertices(); ++v) {
            const auto x = base.get_vertex_coords(static_cast<svmp::index_t>(v));
            values[v] = static_cast<svmp::real_t>(phi(static_cast<Real>(x[0]), static_cast<Real>(x[1])));
        }
    }
    out.space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    out.system = std::make_unique<FESystem>(out.mesh);
    auto& sys = *out.system;
    sys.setCutIntegrationContext(generatedInterfaceContext(*out.mesh, kInterface, phi, 1u, /*quadrature_order=*/4));
    out.u1 = sys.addField(FieldSpec{.name = "u1", .space = out.space});
    out.u2 = sys.addField(FieldSpec{.name = "u2", .space = out.space});
    if (multiplier) {
        out.lambda = sys.addField(FieldSpec{.name = "lambda", .space = out.space});
    }
    sys.addOperator("op");

    const auto& V = *out.space;
    const auto u1 = forms::StateField(out.u1, V, "u1");
    const auto v1 = forms::TestField(out.u1, V, "v1");
    const auto u2 = forms::StateField(out.u2, V, "u2");
    const auto v2 = forms::TestField(out.u2, V, "v2");
    const auto k1 = FormExpr::constant(kInner);
    const auto k2 = FormExpr::constant(kOuter);
    const auto f = FormExpr::constant(-4.0);
    const auto n_ = FormExpr::normal(); // from u1's side (phi < 0) to u2's
    const auto h = FormExpr::cellDiameter();
    const auto inside = forms::CutVolumeSide::Negative;
    const auto outside = forms::CutVolumeSide::Positive;
    auto residual = (k1 * forms::inner(forms::grad(u1), forms::grad(v1)) - f * v1).dCutVolume(kInterface, inside) +
                    (k2 * forms::inner(forms::grad(u2), forms::grad(v2)) - f * v2).dCutVolume(kInterface, outside);
    // The mean flux {k du/dn} and its test counterparts.
    const auto flux_u = FormExpr::constant(0.5) * (k1 * forms::inner(forms::grad(u1), n_) +
                                                   k2 * forms::inner(forms::grad(u2), n_));
    const auto half_flux_v1 = FormExpr::constant(0.5) * k1 * forms::inner(forms::grad(v1), n_);
    const auto half_flux_v2 = FormExpr::constant(0.5) * k2 * forms::inner(forms::grad(v2), n_);
    std::vector<FieldId> fields{out.u1, out.u2};
    if (multiplier) {
        const auto lam = forms::StateField(out.lambda, V, "lambda");
        const auto mu = forms::TestField(out.lambda, V, "mu");
        // lambda (v2 - v1) + mu (u2 - u1): lambda is the flux k du/dn.
        residual = residual + (lam * v2).dI(kInterface) - (lam * v1).dI(kInterface) +
                   (mu * (u2 - u1)).dI(kInterface);
        // - beta h (lambda - {k du/dn}) (mu - {k dv/dn}), one integral per test function.
        const auto weighted = FormExpr::constant(beta) * h * (lam - flux_u);
        residual = residual - (weighted * mu).dI(kInterface) + (weighted * half_flux_v1).dI(kInterface) +
                   (weighted * half_flux_v2).dI(kInterface);
        fields.push_back(out.lambda);
    } else {
        // Symmetric Nitsche: - {k du/dn}[v] - {k dv/dn}[u] + gamma k2 / h [u][v], [w] = w1 - w2.
        const auto jump_u = u1 - u2;
        const auto penalty = FormExpr::constant(10.0 * kOuter) / h;
        residual = residual - (flux_u * v1).dI(kInterface) + (flux_u * v2).dI(kInterface) -
                   (half_flux_v1 * jump_u).dI(kInterface) - (half_flux_v2 * jump_u).dI(kInterface) +
                   (penalty * jump_u * v1).dI(kInterface) - (penalty * jump_u * v2).dI(kInterface);
    }
    (void)systems::installFormulation(sys, "op", fields, residual);

    sys.addSystemConstraint(std::make_unique<constraints::LevelSetActiveSideVertexDirichletConstraint>(
        out.u1, "phi", constraints::LevelSetConstraintSide::Negative, Real(0.0), Real(0.0), kInterface));
    sys.addSystemConstraint(std::make_unique<constraints::LevelSetActiveSideVertexDirichletConstraint>(
        out.u2, "phi", constraints::LevelSetConstraintSide::Positive, Real(0.0), Real(0.0), kInterface));
    const std::array<forms::bc::StrongDirichlet, 1> bcs{
        forms::bc::strongDirichlet(out.u2, kWholeBoundary, coefficient("u2_exact", exactOuter), "u2")};
    systems::installStrongDirichlet(sys, std::span<const forms::bc::StrongDirichlet>(bcs));
    if (multiplier) {
        sys.addSystemConstraint(
            std::make_unique<constraints::SupportRestrictionConstraint>(out.lambda, kInterface, std::string("phi")));
    }
    sys.setup();
    return out;
}

struct CutErrors {
    Real u{0.0};      ///< L2 error of u1 inside plus u2 outside
    Real lambda{0.0}; ///< L2 error of lambda on the circle against the flux 1
};

CutErrors cutErrors(const CutProblem& problem, const std::vector<Real>& state)
{
    const auto& V = *problem.space;
    const auto u1 = forms::StateField(problem.u1, V, "u1");
    const auto u2 = forms::StateField(problem.u2, V, "u2");
    const auto e1 = u1 - coefficient("u1_exact", exactInner);
    const auto e2 = u2 - coefficient("u2_exact", exactOuter);
    std::vector<FormExpr> forms_{(e1 * e1).dCutVolume(kInterface, forms::CutVolumeSide::Negative),
                                 (e2 * e2).dCutVolume(kInterface, forms::CutVolumeSide::Positive)};
    if (problem.lambda != INVALID_FIELD_ID) {
        const auto lam = forms::StateField(problem.lambda, V, "lambda");
        forms_.push_back(((lam - FormExpr::constant(1.0)) * (lam - FormExpr::constant(1.0))).dI(kInterface));
    }
    systems::SystemStateView view;
    view.u = state;
    const auto values =
        systems::evaluateFunctionalForms(*problem.system, forms_, view, systems::CollectiveCommunicator{});
    CutErrors out;
    out.u = std::sqrt(values[0].value + values[1].value);
    if (values.size() > 2u) {
        out.lambda = std::sqrt(values[2].value);
    }
    return out;
}

} // namespace

TEST(TraceMultiplierI3, AStabilizedMultiplierOnACutInterfaceMatchesNitsche)
{
    std::vector<CutErrors> with_multiplier;
    std::vector<CutErrors> with_nitsche;
    for (const int n : {16, 32}) {
        auto multiplier = buildCutProblem(n, /*multiplier=*/true, /*beta=*/0.05);
        with_multiplier.push_back(cutErrors(multiplier, solveLinear(*multiplier.system)));
        auto nitsche = buildCutProblem(n, /*multiplier=*/false, 0.0);
        with_nitsche.push_back(cutErrors(nitsche, solveLinear(*nitsche.system)));
        std::cout << "[ I3 n=" << n << " ] multiplier e_u=" << with_multiplier.back().u
                  << " e_lambda=" << with_multiplier.back().lambda << "  nitsche e_u=" << with_nitsche.back().u
                  << std::endl;
    }
    for (std::size_t k = 0; k < with_multiplier.size(); ++k) {
        // The multiplier coupling is as accurate as Nitsche's.
        EXPECT_LT(with_multiplier[k].u, 3.0 * with_nitsche[k].u) << "mesh " << k;
    }
    // Both converge at second order in u; lambda tends to the flux.
    EXPECT_GT(std::log2(with_multiplier[0].u / with_multiplier[1].u), 1.5);
    EXPECT_GT(std::log2(with_nitsche[0].u / with_nitsche[1].u), 1.5);
    EXPECT_LT(with_multiplier[1].lambda, with_multiplier[0].lambda);
    EXPECT_LT(with_multiplier[1].lambda, 0.1);
}

} // namespace svmp::FE::test_trace_multiplier
