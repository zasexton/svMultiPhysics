/**
 * @file test_CanonicalWorkflow.cpp
 * @brief End-to-end tests for the canonical formulation workflow
 *
 * Tests the recommended pattern:
 *   StateField + TestField + BoundaryConditionManager::applyAll + installFormulation
 *
 * Covers:
 *   - Same-space multi-field residual assembly (not just metadata)
 *   - BC convenience flow with actual strong constraints
 *   - Full canonical workflow from registration through assembly
 *
 * Expert/manual path parity tests live in test_MixedManualParity.cpp.
 */

#include <gtest/gtest.h>

#include "Systems/BoundaryConditionManager.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include "Forms/FormExpr.h"
#include "Forms/StandardBCs.h"
#include "Forms/Vocabulary.h"

#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <span>
#include <stdexcept>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;
using namespace svmp::FE::forms;

namespace {

svmp::FE::dofs::MeshTopologyInfo singleTetraTopology()
{
    svmp::FE::dofs::MeshTopologyInfo topo;
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

} // namespace

TEST(CanonicalWorkflow, SameSpaceWeightedInterfaceSplitsByTestField)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto scalar = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    auto vector = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar, 3);
    svmp::FE::systems::FESystem sys(mesh);
    const auto un_id = sys.addField({.name = "u_negative", .space = vector, .components = 3});
    const auto pn_id = sys.addField({.name = "p_negative", .space = scalar});
    const auto up_id = sys.addField({.name = "u_positive", .space = vector, .components = 3});
    const auto pp_id = sys.addField({.name = "p_positive", .space = scalar});
    sys.addOperator("interface");
    sys.addOperator("compact");

    // Each phase uses the same spaces and symbol names. Only FieldId identifies
    // the phase; unequal viscosities give unequal complementary weights.
    const auto un = StateField(un_id, *vector, "u");
    const auto pn = StateField(pn_id, *scalar, "p");
    const auto up = StateField(up_id, *vector, "u");
    const auto pp = StateField(pp_id, *scalar, "p");
    const auto vn = TestField(un_id, *vector, "v");
    const auto qn = TestField(pn_id, *scalar, "q");
    const auto vp = TestField(up_id, *vector, "v");
    const auto qp = TestField(pp_id, *scalar, "q");
    const auto n = FormExpr::normal();
    const auto identity = FormExpr::identity(3);
    const auto tn = (2.0 * 2.0 * sym(grad(un))) * n;
    const auto tp = (2.0 * 8.0 * sym(grad(up))) * n;
    const auto sn = (2.0 * 2.0 * sym(grad(vn))) * n;
    const auto sp = (2.0 * 8.0 * sym(grad(vp))) * n;
    const auto ptn = (-pn * identity) * n;
    const auto ptp = (-pp * identity) * n;
    const auto qtn = (-qn * identity) * n;
    const auto qtp = (-qp * identity) * n;
    const auto wn = FormExpr::constant(0.8);
    const auto wp = FormExpr::constant(0.2);
    const auto penalty = FormExpr::constant(6.0);
    constexpr int marker = 71;

    // The Nitsche coupling as one integral over the two phases' fields.
    const SidePair U{un, up};
    const SidePair P{pn, pp};
    const SidePair V{vn, vp};
    const SidePair Q{qn, qp};
    const auto traction = [&](const FormExpr& w, const FormExpr& r, CutVolumeSide side) {
        const Real mu = side == CutVolumeSide::Negative ? 2.0 : 8.0;
        return (2.0 * mu * sym(grad(w))) * n + (-r * identity) * n;
    };
    const auto compact =
        (-inner(avg(map(traction, U, P), wn, wp), jump(V)) -
         inner(avg(map(traction, V, Q), wn, wp), jump(U)) +
         penalty * inner(jump(U), jump(V))).dI(marker);
    // The record's block couplings follow the test functions' order of
    // appearance (v_neg, v_pos, q_neg, q_pos), so the fields are listed in it.
    const std::array compact_fields{un_id, up_id, pn_id, pp_id};
    svmp::FE::systems::CoupledResidualKernels compact_installed;
    ASSERT_NO_THROW(compact_installed = svmp::FE::systems::installFormulation(
        sys, "compact", compact_fields, compact));

    // The production-style spelling keeps the viscous and pressure pieces
    // separable, with each phase's negative row before its positive row.
    const auto consistency =
        (-wn * inner(tn, vn)).dI(marker) + (-wn * inner(ptn, vn)).dI(marker) +
        (wn * inner(tn, vp)).dI(marker) + (wn * inner(ptn, vp)).dI(marker) +
        (-wp * inner(tp, vn)).dI(marker) + (-wp * inner(ptp, vn)).dI(marker) +
        (wp * inner(tp, vp)).dI(marker) + (wp * inner(ptp, vp)).dI(marker);
    const auto adjoint =
        (-wn * inner(sn, un)).dI(marker) + (-wn * inner(qtn, un)).dI(marker) +
        (wn * inner(sn, up)).dI(marker) + (wn * inner(qtn, up)).dI(marker) +
        (-wp * inner(sp, un)).dI(marker) + (-wp * inner(qtp, un)).dI(marker) +
        (wp * inner(sp, up)).dI(marker) + (wp * inner(qtp, up)).dI(marker);
    const auto penalty_terms =
        (penalty * inner(un, vn)).dI(marker) + (-penalty * inner(un, vp)).dI(marker) +
        (-penalty * inner(up, vn)).dI(marker) + (penalty * inner(up, vp)).dI(marker);
    // Reverse the requested row order to exclude incidental traversal mapping.
    const std::array fields{pp_id, up_id, pn_id, un_id};
    svmp::FE::systems::CoupledResidualKernels installed;
    ASSERT_NO_THROW(installed = svmp::FE::systems::installFormulation(
        sys, "interface", fields, consistency + adjoint + penalty_terms));

    const std::set<std::pair<FieldId, FieldId>> expected{
        {un_id, un_id}, {un_id, pn_id}, {un_id, up_id}, {un_id, pp_id},
        {up_id, un_id}, {up_id, pn_id}, {up_id, up_id}, {up_id, pp_id},
        {pn_id, un_id}, {pn_id, up_id}, {pp_id, un_id}, {pp_id, up_id}};
    std::set<std::pair<FieldId, FieldId>> actual;
    for (const auto& term : sys.operatorDefinition("interface").interface_faces) {
        EXPECT_EQ(term.marker, marker);
        EXPECT_NE(term.kernel, nullptr);
        actual.emplace(term.test_field, term.trial_field);
    }
    EXPECT_EQ(actual, expected);
    // The compact integral installs the blocks of the expanded spelling.
    std::set<std::pair<FieldId, FieldId>> compact_actual;
    for (const auto& term : sys.operatorDefinition("compact").interface_faces) {
        EXPECT_EQ(term.marker, marker);
        compact_actual.emplace(term.test_field, term.trial_field);
    }
    EXPECT_EQ(compact_actual, expected);
    ASSERT_EQ(sys.formulationRecords().size(), 2u);
    const std::set<std::pair<FieldId, FieldId>> compact_couplings(
        sys.formulationRecords()[0].block_couplings.begin(),
        sys.formulationRecords()[0].block_couplings.end());
    EXPECT_EQ(compact_couplings, expected);

    ASSERT_NE(installed.mixed_plan, nullptr);
    std::set<std::pair<FieldId, FieldId>> planned;
    for (const auto& block : installed.mixed_plan->blocks) {
        EXPECT_TRUE(block.has_interface);
        EXPECT_TRUE(block.want_matrix);
        planned.emplace(block.test_field, block.trial_field);
    }
    EXPECT_EQ(planned, expected);
    ASSERT_EQ(installed.jacobian_blocks.size(), fields.size());
    for (std::size_t row = 0; row < fields.size(); ++row) {
        ASSERT_EQ(installed.jacobian_blocks[row].size(), fields.size());
        for (std::size_t col = 0; col < fields.size(); ++col) {
            const auto& kernel = installed.jacobian_blocks[row][col];
            EXPECT_EQ(kernel != nullptr, expected.contains({fields[row], fields[col]}));
            for (const auto& term : sys.operatorDefinition("interface").interface_faces) {
                if (term.test_field == fields[row] && term.trial_field == fields[col]) {
                    EXPECT_EQ(term.kernel, kernel);
                }
            }
        }
    }
}

// ============================================================================
// Same-space assembly: two scalar fields on the same H1 space
// ============================================================================

TEST(CanonicalWorkflow, SameSpace_TwoField_Assembly)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto sys = std::make_unique<svmp::FE::systems::FESystem>(mesh);
    const auto T_f = sys->addField({.name = "T", .space = space, .components = 1});
    const auto C_f = sys->addField({.name = "C", .space = space, .components = 1});
    sys->addOperator("op");

    // Canonical workflow: StateField + TestField with field bindings
    auto T = StateField(T_f, *space, "T");
    auto C = StateField(C_f, *space, "C");
    auto w = TestField(T_f, *space, "w");
    auto r = TestField(C_f, *space, "r");

    // Coupled system: T equation depends on T only, C equation depends on C and T
    auto residual = (inner(grad(T), grad(w))).dx()
                  + (inner(grad(C), grad(r)) + T * r).dx();

    svmp::FE::systems::installFormulation(*sys, "op", {T_f, C_f}, residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys->setup({}, inputs);

    const auto n = sys->dofHandler().getNumDofs();
    ASSERT_EQ(n, 8);  // 4 dofs/field * 2 fields

    std::vector<Real> U(static_cast<std::size_t>(n), 0.0);
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = 0.1 * static_cast<Real>(i + 1);
    }

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();
    (void)sys->assemble(req, state, &out, &out);

    // Residual should be non-zero
    double vec_norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        vec_norm += out.getVectorEntry(i) * out.getVectorEntry(i);
    }
    EXPECT_GT(vec_norm, 0.0);

    // Jacobian should be non-zero
    double mat_norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            mat_norm += out.getMatrixEntry(i, j) * out.getMatrixEntry(i, j);
        }
    }
    EXPECT_GT(mat_norm, 0.0);

    // Verify block structure: T equation (rows 0-3) should have no coupling to C DOFs (cols 4-7)
    // in the off-diagonal TT block, since grad(T):grad(w) has no C dependency.
    // C equation (rows 4-7) SHOULD couple to T DOFs (cols 0-3) via the T*r term.
    bool has_CT_coupling = false;
    for (GlobalIndex i = 4; i < 8; ++i) {
        for (GlobalIndex j = 0; j < 4; ++j) {
            if (std::abs(out.getMatrixEntry(i, j)) > 1e-14) {
                has_CT_coupling = true;
                break;
            }
        }
    }
    EXPECT_TRUE(has_CT_coupling) << "C equation should couple to T via the T*r term";
}

// ============================================================================
// One integral holding the test functions of two fields
// ============================================================================

namespace {

struct AssembledTwoField {
    std::vector<Real> vector;
    std::vector<Real> matrix;
};

AssembledTwoField assembleTwoField(bool one_integral)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    svmp::FE::systems::FESystem sys(mesh);
    const auto T_f = sys.addField({.name = "T", .space = space, .components = 1});
    const auto C_f = sys.addField({.name = "C", .space = space, .components = 1});
    sys.addOperator("op");
    const auto T = StateField(T_f, *space, "T");
    const auto C = StateField(C_f, *space, "C");
    const auto w = TestField(T_f, *space, "w");
    const auto r = TestField(C_f, *space, "r");
    const auto residual =
        one_integral ? (inner(grad(T), grad(w)) + inner(grad(C), grad(r)) + T * C * r).dx()
                     : (inner(grad(T), grad(w))).dx() + (inner(grad(C), grad(r)) + T * C * r).dx();
    svmp::FE::systems::installFormulation(sys, "op", {T_f, C_f}, residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);
    const auto n = sys.dofHandler().getNumDofs();
    std::vector<Real> U(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = 0.1 * static_cast<Real>(i + 1);
    }
    svmp::FE::systems::SystemStateView state;
    state.u = U;
    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();
    (void)sys.assemble(req, state, &out, &out);
    AssembledTwoField assembled;
    for (GlobalIndex i = 0; i < n; ++i) {
        assembled.vector.push_back(out.getVectorEntry(i));
        for (GlobalIndex j = 0; j < n; ++j) {
            assembled.matrix.push_back(out.getMatrixEntry(i, j));
        }
    }
    return assembled;
}

} // namespace

TEST(CanonicalWorkflow, OneIntegralOverTwoTestFieldsAssemblesAsOneIntegralPerField)
{
    // The integral is split into its part in each test field's row; a part
    // that is a whole summand keeps its tree, so the kernels and the
    // assembled values are those of the expanded spelling.
    const auto one = assembleTwoField(/*one_integral=*/true);
    const auto two = assembleTwoField(/*one_integral=*/false);
    EXPECT_EQ(one.vector, two.vector);
    EXPECT_EQ(one.matrix, two.matrix);
}

TEST(CanonicalWorkflow, ATermNonlinearInTwoTestFieldsIsRefused)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    svmp::FE::systems::FESystem sys(mesh);
    const auto T_f = sys.addField({.name = "T", .space = space, .components = 1});
    const auto C_f = sys.addField({.name = "C", .space = space, .components = 1});
    sys.addOperator("op");
    const auto T = StateField(T_f, *space, "T");
    const auto w = TestField(T_f, *space, "w");
    const auto r = TestField(C_f, *space, "r");
    try {
        svmp::FE::systems::installFormulation(sys, "op", {T_f, C_f}, (inner(grad(T), grad(w)) + T * w * r).dx());
        FAIL() << "A product of two test functions is not a residual";
    } catch (const svmp::FE::InvalidArgumentException& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("cannot split the term"), std::string::npos) << message;
        EXPECT_NE(message.find("both factors of a product hold test functions"), std::string::npos) << message;
    }
    EXPECT_TRUE(sys.operatorDefinition("op").cells.empty());
    EXPECT_TRUE(sys.formulationRecords().empty());
}

// ============================================================================
// Install-time checks: unknowns outside the trial fields, history reads, shape errors
// ============================================================================

TEST(CanonicalWorkflow, AnUnknownOutsideTheTrialFieldsIsRefusedNamingTheTerm)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    svmp::FE::systems::FESystem sys(mesh);
    const auto T_f = sys.addField({.name = "T", .space = space});
    const auto C_f = sys.addField({.name = "C", .space = space});
    const auto g_f = sys.addField({.name = "g", .space = space,
                                   .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData});
    sys.addOperator("op");
    sys.addOperator("work");
    const auto T = StateField(T_f, *space, "T");
    const auto C = StateField(C_f, *space, "C");
    const auto w = TestField(T_f, *space, "w");
    const auto coupling = (C * w).dx();
    try {
        svmp::FE::systems::installFormulation(sys, "op", {T_f}, (inner(grad(T), grad(w))).dx() + coupling);
        FAIL() << "the columns of C would be dropped";
    } catch (const svmp::FE::InvalidArgumentException& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("the term '" + coupling.toString() + "'"), std::string::npos) << message;
        EXPECT_NE(message.find("holds the unknown 'C'"), std::string::npos) << message;
        EXPECT_NE(message.find("frozen()"), std::string::npos) << message;
    }
    EXPECT_TRUE(sys.operatorDefinition("op").cells.empty());
    EXPECT_TRUE(sys.formulationRecords().empty());

    // C as a trial field, C held fixed as data (frozen() or a DiscreteField),
    // a prescribed field's StateField, and a measurement install that is
    // assembled for its vector all install.
    svmp::FE::systems::FormInstallOptions with_c;
    with_c.extra_trial_fields = {C_f};
    EXPECT_NO_THROW(svmp::FE::systems::installFormulation(sys, "op", {T_f}, (inner(grad(T), grad(w))).dx() + coupling,
                                                          with_c));
    EXPECT_NO_THROW(svmp::FE::systems::installFormulation(
        sys, "op", {T_f}, (inner(grad(T), grad(w)) + FormExpr::discreteField(C_f, *space, "C") * w).dx()));
    EXPECT_NO_THROW(svmp::FE::systems::installFormulation(
        sys, "op", {T_f}, (inner(grad(T), grad(w)) + frozen(C) * w).dx()));
    EXPECT_NO_THROW(svmp::FE::systems::installFormulation(
        sys, "op", {T_f}, (inner(grad(T), grad(w)) + StateField(g_f, *space, "g") * w).dx()));
    svmp::FE::systems::FormInstallOptions measurement;
    measurement.contributes_to_problem_analysis = false;
    EXPECT_NO_THROW(svmp::FE::systems::installFormulation(sys, "work", {T_f}, coupling, measurement));
}

TEST(CanonicalWorkflow, PreviousSolutionInARowOfTwoUnknownsIsRefused)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    svmp::FE::systems::FESystem sys(mesh);
    const auto T_f = sys.addField({.name = "T", .space = space});
    const auto C_f = sys.addField({.name = "C", .space = space});
    sys.addOperator("op");
    const auto T = StateField(T_f, *space, "T");
    const auto C = StateField(C_f, *space, "C");
    const auto w = TestField(T_f, *space, "w");
    const auto r = TestField(C_f, *space, "r");
    // previousSolution(1) is the history of the block's trial field: C's in
    // the (T, C) block.
    const auto lagged_term = ((T - FormExpr::previousSolution(1)) * w).dx();
    try {
        svmp::FE::systems::installFormulation(sys, "op", {T_f, C_f},
                                              lagged_term + (C * w).dx() + (inner(grad(C), grad(r))).dx());
        FAIL() << "previousSolution would read C's history in the (T, C) block";
    } catch (const svmp::FE::InvalidArgumentException& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("the term '" + lagged_term.toString() + "'"), std::string::npos) << message;
        EXPECT_NE(message.find("also depends on 'C'"), std::string::npos) << message;
    }
    EXPECT_TRUE(sys.formulationRecords().empty());
    // A row that depends on its own field alone reads that field's history.
    EXPECT_NO_THROW(svmp::FE::systems::installFormulation(
        sys, "op", {T_f, C_f}, lagged_term + (inner(grad(C), grad(r)) + T * r).dx()));
}

TEST(CanonicalWorkflow, AShapeErrorOfAProductNamesTheTerm)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto scalar = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    auto vector = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar, 3);
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = vector, .components = 3});
    sys.addOperator("op");
    const auto u = StateField(u_f, *vector, "u");
    const auto v = TestField(u_f, *vector, "v");
    svmp::FE::systems::FormInstallOptions options;
    options.compiler_options.jit.enable = false;
    svmp::FE::systems::installFormulation(sys, "op", {u_f}, (inner(u * u, v)).dx(), options);
    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);
    const auto n = sys.dofHandler().getNumDofs();
    std::vector<Real> U(static_cast<std::size_t>(n), 0.1);
    svmp::FE::systems::SystemStateView state;
    state.u = U;
    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    svmp::FE::assembly::DenseSystemView out(n);
    try {
        (void)sys.assemble(req, state, &out, &out);
        FAIL() << "a product of two vectors is not defined";
    } catch (const std::exception& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("unsupported multiplication kinds"), std::string::npos) << message;
        EXPECT_NE(message.find("lhs=Vector"), std::string::npos) << message;
    }
}

// ============================================================================
// Variations and linearization: derivative() and frozen()
// ============================================================================

namespace {

struct AssembledSystem {
    std::vector<Real> vector;
    std::vector<Real> matrix;
};

/// Residual and Jacobian of @p residual of the unknown @p field at a fixed
/// state, on the single tetrahedron.
AssembledSystem assembleAtState(svmp::FE::systems::FESystem& sys, const std::vector<FieldId>& fields,
                                const FormExpr& residual, bool jit)
{
    svmp::FE::systems::FormInstallOptions options;
    options.compiler_options.jit.enable = jit;
    svmp::FE::systems::installFormulation(sys, "op", std::span<const FieldId>(fields), residual, options);
    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);
    const auto n = sys.dofHandler().getNumDofs();
    std::vector<Real> U(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = 0.05 * std::sin(0.9 * static_cast<Real>(i) + 0.4);
    }
    svmp::FE::systems::SystemStateView state;
    state.u = U;
    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();
    (void)sys.assemble(req, state, &out, &out);
    AssembledSystem assembled;
    for (GlobalIndex i = 0; i < n; ++i) {
        assembled.vector.push_back(out.getVectorEntry(i));
        for (GlobalIndex j = 0; j < n; ++j) {
            assembled.matrix.push_back(out.getMatrixEntry(i, j));
        }
    }
    return assembled;
}

Real maxRelativeDifference(const std::vector<Real>& a, const std::vector<Real>& b)
{
    EXPECT_EQ(a.size(), b.size());
    Real scale = 0.0;
    Real difference = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        scale = std::max(scale, std::abs(a[i]));
        difference = std::max(difference, std::abs(a[i] - b[i]));
    }
    return scale > 0.0 ? difference / scale : difference;
}

} // namespace

TEST(CanonicalWorkflow, TheDerivativeOfThePoissonEnergyIsItsResidual)
{
    for (const bool jit : {false, true}) {
        const auto build = [&](bool from_energy) {
            auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
            auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
            svmp::FE::systems::FESystem sys(mesh);
            const auto u_id = sys.addField({.name = "u", .space = space});
            sys.addOperator("op");
            const auto u = StateField(u_id, *space, "u");
            const auto v = TestField(u_id, *space, "v");
            const auto f = 1.0 + x().component(0);
            const auto energy = (0.5 * inner(grad(u), grad(u)) + 0.25 * u * u * u * u - f * u).dx();
            const auto residual = from_energy ? derivative(energy, u, v)
                                              : (inner(grad(u), grad(v)) + u * u * u * v - f * v).dx();
            return assembleAtState(sys, {u_id}, residual, jit);
        };
        const auto derived = build(true);
        const auto hand = build(false);
        EXPECT_LE(maxRelativeDifference(derived.vector, hand.vector), 1e-14) << "jit=" << jit;
        EXPECT_LE(maxRelativeDifference(derived.matrix, hand.matrix), 1e-14) << "jit=" << jit;
    }
}

TEST(CanonicalWorkflow, TheDerivativeOfTheNeoHookeanEnergyIsItsFirstPiolaResidual)
{
    const auto build = [&](bool from_energy) {
        auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
        auto scalar = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
        auto vector = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar, 3);
        svmp::FE::systems::FESystem sys(mesh);
        const auto u_id = sys.addField({.name = "u", .space = vector, .components = 3});
        sys.addOperator("op");
        const auto u = StateField(u_id, *vector, "u");
        const auto v = TestField(u_id, *vector, "v");
        const Real mu = 1.5;
        const Real lambda = 4.0;
        const auto F = FormExpr::identity(3) + grad(u);
        const auto J = det(F);
        const auto C = transpose(F) * F;
        const auto FinvT = transpose(inv(F));
        if (from_energy) {
            const auto psi = 0.5 * mu * (trace(C) - 3.0) - mu * log(J) + 0.5 * lambda * log(J) * log(J);
            return assembleAtState(sys, {u_id}, derivative(psi.dx(), u, v), false);
        }
        const auto P = mu * (F - FinvT) + lambda * log(J) * FinvT;
        return assembleAtState(sys, {u_id}, doubleContraction(P, grad(v)).dx(), false);
    };
    const auto derived = build(true);
    const auto hand = build(false);
    EXPECT_LE(maxRelativeDifference(derived.vector, hand.vector), 1e-12);
    EXPECT_LE(maxRelativeDifference(derived.matrix, hand.matrix), 1e-12);
}

TEST(CanonicalWorkflow, DerivativeRefusesWhatIsNotAVariation)
{
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    const auto u = StateField(0, *space, "u");
    const auto w = StateField(1, *space, "w");
    const auto v = TestField(0, *space, "v");
    const auto energy = (0.5 * u * u).dx();
    EXPECT_THROW((void)derivative(energy, TrialFunction(*space, "u"), v), std::invalid_argument);
    EXPECT_THROW((void)derivative(energy, w, v), std::invalid_argument);
    // A field the form does not depend on drops out of the sum.
    const auto both = derivative(energy, {{u, v}, {w, TestField(1, *space, "q")}});
    EXPECT_EQ(both.toString(), derivative(energy, u, v).toString());
}

TEST(CanonicalWorkflow, FrozenIsTheDiscreteFieldOfTheSameField)
{
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    const auto u = StateField(3, *space, "u");
    const auto frozen_u = frozen(u);
    const auto discrete_u = FormExpr::discreteField(3, *space, "u");
    ASSERT_EQ(frozen_u.node()->type(), FormExprType::DiscreteField);
    EXPECT_EQ(frozen_u.toString(), discrete_u.toString());
    EXPECT_EQ(frozen_u.node()->fieldId(), discrete_u.node()->fieldId());
    EXPECT_EQ(frozen(u * grad(u)).toString(), (discrete_u * grad(discrete_u)).toString());
    EXPECT_THROW((void)frozen(TrialFunction(*space, "u")), std::invalid_argument);
    EXPECT_THROW((void)frozen(TestFunction(*space, "v")), std::invalid_argument);
    EXPECT_THROW((void)frozen((u * u).dx()), std::invalid_argument);
    EXPECT_THROW((void)frozen(u * currentJacobianDeterminant()), std::invalid_argument);
}

TEST(CanonicalWorkflow, AFrozenStateGivesAZeroResidualAndTheMassMatrix)
{
    const auto build = [&](bool frozen_difference) {
        auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
        auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
        svmp::FE::systems::FESystem sys(mesh);
        const auto u_id = sys.addField({.name = "u", .space = space});
        sys.addOperator("op");
        const auto u = StateField(u_id, *space, "u");
        const auto v = TestField(u_id, *space, "v");
        const auto residual = frozen_difference ? ((u - frozen(u)) * v).dx() : (u * v).dx();
        return assembleAtState(sys, {u_id}, residual, false);
    };
    const auto difference = build(true);
    const auto mass = build(false);
    // u and its frozen value are evaluated by two paths, equal to round-off.
    for (const auto value : difference.vector) {
        EXPECT_NEAR(value, 0.0, 1e-16);
    }
    EXPECT_LE(maxRelativeDifference(difference.matrix, mass.matrix), 1e-15);
}

TEST(CanonicalWorkflow, PicardConvectionWithFrozenIsTheDiscreteFieldSpelling)
{
    const auto build = [&](bool with_frozen) {
        auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
        auto scalar = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
        auto vector = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar, 3);
        svmp::FE::systems::FESystem sys(mesh);
        const auto u_id = sys.addField({.name = "u", .space = vector, .components = 3});
        sys.addOperator("op");
        const auto u = StateField(u_id, *vector, "u");
        const auto v = TestField(u_id, *vector, "v");
        const auto a = with_frozen ? frozen(u) : FormExpr::discreteField(u_id, *vector, "u");
        const auto residual = (inner(advect(a, u), v) + 0.1 * inner(grad(u), grad(v))).dx();
        return assembleAtState(sys, {u_id}, residual, false);
    };
    const auto with_frozen = build(true);
    const auto discrete = build(false);
    EXPECT_EQ(with_frozen.vector, discrete.vector);
    EXPECT_EQ(with_frozen.matrix, discrete.matrix);
}

// ============================================================================
// BC workflow: strong Dirichlet via direct installStrongDirichlet
// ============================================================================

TEST(CanonicalWorkflow, BC_StrongDirichlet_Constraints)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = StateField(u_f, *space, "u");
    auto v = TestField(u_f, *space, "v");
    auto residual = (inner(grad(u), grad(v))).dx();

    auto bc = svmp::FE::forms::bc::strongDirichlet(
        u_f, marker, FormExpr::constant(2.5), "u");
    svmp::FE::systems::installStrongDirichlet(sys,
        std::span<const svmp::FE::forms::bc::StrongDirichlet>(&bc, 1));

    svmp::FE::systems::installFormulation(sys, "op", {u_f}, residual);

    svmp::FE::systems::SetupInputs inputs;
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    inputs.topology_override = topo;
    sys.setup({}, inputs);

    // Boundary face (0,1,2) should be constrained
    EXPECT_TRUE(sys.constraints().isConstrained(0));
    EXPECT_TRUE(sys.constraints().isConstrained(1));
    EXPECT_TRUE(sys.constraints().isConstrained(2));
    EXPECT_FALSE(sys.constraints().isConstrained(3));

    // Constrained DOFs should have inhomogeneity = 2.5
    for (GlobalIndex dof : {0, 1, 2}) {
        EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 2.5, 1e-15);
    }

    // Assembly should apply constraint: constrained rows get identity, off-diag zeroed
    const auto n = sys.dofHandler().getNumDofs();
    svmp::FE::assembly::DenseMatrixView mat(n);
    mat.zero();

    svmp::FE::systems::SystemStateView state;
    std::vector<Real> U = {2.5, 2.5, 2.5, 0.4};
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    (void)sys.assemble(req, state, &mat, nullptr);

    // Constrained rows: diagonal = 1, off-diagonal = 0
    for (GlobalIndex dof : {0, 1, 2}) {
        EXPECT_NEAR(mat.getMatrixEntry(dof, dof), 1.0, 1e-12);
        for (GlobalIndex j = 0; j < n; ++j) {
            if (j == dof) continue;
            EXPECT_NEAR(mat.getMatrixEntry(dof, j), 0.0, 1e-12);
        }
    }
}

// ============================================================================
// BC convenience: applyAll with a real EssentialBC object
// ============================================================================

TEST(CanonicalWorkflow, BCApplyAll_WithEssentialBC)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = StateField(u_f, *space, "u");
    auto v = TestField(u_f, *space, "v");
    auto residual = (inner(grad(u), grad(v))).dx();

    // Use BoundaryConditionManager with a concrete EssentialBC
    svmp::FE::systems::BoundaryConditionManager bc_manager;
    bc_manager.add(std::make_unique<svmp::FE::forms::bc::EssentialBC>(
        marker, FormExpr::constant(3.0), "u"));

    // One-call convenience: validate + apply weak terms + install strong constraints
    bc_manager.applyAll(sys, residual, u, v, u_f);

    svmp::FE::systems::installFormulation(sys, "op", {u_f}, residual);

    svmp::FE::systems::SetupInputs inputs;
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    inputs.topology_override = topo;
    sys.setup({}, inputs);

    // Boundary face nodes (0,1,2) should be constrained with value 3.0
    EXPECT_TRUE(sys.constraints().isConstrained(0));
    EXPECT_TRUE(sys.constraints().isConstrained(1));
    EXPECT_TRUE(sys.constraints().isConstrained(2));
    EXPECT_FALSE(sys.constraints().isConstrained(3));

    for (GlobalIndex dof : {0, 1, 2}) {
        EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 3.0, 1e-15);
    }
}

// ============================================================================
// 2-arg applyAll: strong-only BCs succeed
// ============================================================================

TEST(CanonicalWorkflow, BCApplyAll_FieldOnly_StrongBCsSucceed)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    // 2-arg overload with a strong-only BC should succeed
    svmp::FE::systems::BoundaryConditionManager bc_manager;
    bc_manager.add(std::make_unique<svmp::FE::forms::bc::EssentialBC>(
        marker, FormExpr::constant(1.0), "p"));

    EXPECT_NO_THROW(bc_manager.applyAll(sys, p_f));

    // Verify the constraint was actually installed
    svmp::FE::systems::SetupInputs inputs;
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    inputs.topology_override = topo;
    sys.setup({}, inputs);

    EXPECT_TRUE(sys.constraints().isConstrained(0));
    EXPECT_TRUE(sys.constraints().isConstrained(1));
    EXPECT_TRUE(sys.constraints().isConstrained(2));
    EXPECT_FALSE(sys.constraints().isConstrained(3));
}

// ============================================================================
// 2-arg applyAll: weak BCs are rejected before system mutation
// ============================================================================

TEST(CanonicalWorkflow, BCApplyAll_FieldOnly_WeakBCThrows)
{
    const int marker = 5;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Snapshot system state before the rejected call
    const auto bc_descs_before = sys.boundaryConditionDescriptors().size();
    const auto contribs_before = sys.contributionDescriptors().size();

    // A NaturalBC has weak terms — 2-arg overload must reject it
    svmp::FE::systems::BoundaryConditionManager bc_manager;
    bc_manager.add(std::make_unique<svmp::FE::forms::bc::NaturalBC>(
        marker, FormExpr::constant(1.0)));

    EXPECT_THROW(bc_manager.applyAll(sys, u_f), std::invalid_argument);

    // System must be unchanged — no descriptors, contributions, or constraints added
    EXPECT_EQ(sys.boundaryConditionDescriptors().size(), bc_descs_before)
        << "Rejected applyAll must not add BC descriptors";
    EXPECT_EQ(sys.contributionDescriptors().size(), contribs_before)
        << "Rejected applyAll must not add contribution descriptors";
}

TEST(CanonicalWorkflow, BCApplyAll_FieldOnly_MixedStrongAndWeakThrows)
{
    const int marker_d = 5;
    const int marker_n = 6;
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker_d);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    // Snapshot system state before the rejected call
    const auto bc_descs_before = sys.boundaryConditionDescriptors().size();
    const auto contribs_before = sys.contributionDescriptors().size();

    // Mix of strong + weak — must still reject
    svmp::FE::systems::BoundaryConditionManager bc_manager;
    bc_manager.add(std::make_unique<svmp::FE::forms::bc::EssentialBC>(
        marker_d, FormExpr::constant(0.0), "u"));
    bc_manager.add(std::make_unique<svmp::FE::forms::bc::NaturalBC>(
        marker_n, FormExpr::constant(1.0)));

    EXPECT_THROW(bc_manager.applyAll(sys, u_f), std::invalid_argument);

    // System must be unchanged — no descriptors, contributions, or constraints added
    EXPECT_EQ(sys.boundaryConditionDescriptors().size(), bc_descs_before)
        << "Rejected applyAll must not add BC descriptors";
    EXPECT_EQ(sys.contributionDescriptors().size(), contribs_before)
        << "Rejected applyAll must not add contribution descriptors";
}

// ============================================================================
// Same-space, same-name test functions disambiguated by field binding
// ============================================================================

TEST(CanonicalWorkflow, SameSpace_SameName_DifferentBindings)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto sys = std::make_unique<svmp::FE::systems::FESystem>(mesh);
    const auto T_f = sys->addField({.name = "T", .space = space, .components = 1});
    const auto C_f = sys->addField({.name = "C", .space = space, .components = 1});
    sys->addOperator("op");

    // Same space, same test function name "v", different field bindings.
    // Asymmetric diffusion (k_T=1, k_C=10) and REVERSED residual order
    // (C term first) so an order-based mapping bug would put the wrong
    // coefficient in the wrong block.
    auto T = StateField(T_f, *space, "T");
    auto C = StateField(C_f, *space, "C");
    auto v_T = TestField(T_f, *space, "v");
    auto v_C = TestField(C_f, *space, "v");
    const auto k_T = FormExpr::constant(1.0);
    const auto k_C = FormExpr::constant(10.0);

    // C term listed FIRST, T term second — reversed from field order {T_f, C_f}
    auto residual = (k_C * inner(grad(C), grad(v_C))).dx()
                  + (k_T * inner(grad(T), grad(v_T))).dx();

    EXPECT_NO_THROW(
        svmp::FE::systems::installFormulation(*sys, "op", {T_f, C_f}, residual));

    // Go through setup and assembly
    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys->setup({}, inputs);

    const auto n = sys->dofHandler().getNumDofs();
    ASSERT_EQ(n, 8);  // 4 dofs/field * 2 fields

    std::vector<Real> U(static_cast<std::size_t>(n), 0.0);
    for (std::size_t i = 0; i < U.size(); ++i) U[i] = 0.1 * static_cast<Real>(i + 1);

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();

    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    (void)sys->assemble(req, state, &out, &out);

    // T block (rows 0-3, cols 0-3) should have k_T=1 scaling
    // and C block (rows 4-7, cols 4-7) should have k_C=10 scaling.
    // Automatic gauges add unscaled algebraic rows to both blocks, so compare
    // the corresponding unconstrained sub-blocks where the form scaling is
    // preserved.
    double T_block_norm = 0.0, C_block_norm = 0.0;
    for (GlobalIndex i = 0; i < 4; ++i) {
        ASSERT_EQ(sys->constraints().isConstrained(i),
                  sys->constraints().isConstrained(i + 4));
        for (GlobalIndex j = 0; j < 4; ++j) {
            if (sys->constraints().isConstrained(i) ||
                sys->constraints().isConstrained(j)) {
                continue;
            }
            const auto T_entry = out.getMatrixEntry(i, j);
            const auto C_entry = out.getMatrixEntry(i + 4, j + 4);
            T_block_norm += T_entry * T_entry;
            C_block_norm += C_entry * C_entry;
        }
    }

    EXPECT_GT(T_block_norm, 0.0) << "T diagonal block should be non-zero";
    EXPECT_GT(C_block_norm, 0.0) << "C diagonal block should be non-zero";

    // k_C/k_T = 10 → Frobenius norms scale quadratically → ratio ~100
    const double ratio = C_block_norm / T_block_norm;
    EXPECT_NEAR(ratio, 100.0, 1.0e-10)
        << "C block (k=10) should have 100x the unconstrained Frobenius norm² "
           "of T block (k=1). "
           "Got ratio=" << ratio << ". If ~1.0, rows are swapped.";
}

// ============================================================================
// Negative: same-FieldId conflicting name or space is rejected
// ============================================================================

TEST(CanonicalWorkflow, ConflictingBoundTests_SameFieldId_DifferentName_Throws)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    auto u = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
    auto p = svmp::FE::forms::FormExpr::stateField(p_f, *space, "p");

    // Two test functions bound to the SAME FieldId but with DIFFERENT names
    auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "v");
    auto w = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "w");

    auto residual = (u * v).dx() + (p * w).dx();

    try {
        svmp::FE::systems::installFormulation(sys, "op", {u_f, p_f}, residual);
        FAIL() << "Expected InvalidArgumentException for same-FieldId/different-name conflict";
    } catch (const svmp::FE::InvalidArgumentException& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("conflicting test functions"), std::string::npos)
            << "Error should mention 'conflicting test functions', got: " << msg;
        EXPECT_NE(msg.find("consistent space and name"), std::string::npos)
            << "Error should mention 'consistent space and name', got: " << msg;
    }
}

TEST(CanonicalWorkflow, ConflictingBoundTests_SameFieldId_DifferentSpace_Throws)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space_p1 = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
    auto space_p2 = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 2);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space_p1, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space_p1, .components = 1});
    sys.addOperator("op");

    auto u = svmp::FE::forms::FormExpr::stateField(u_f, *space_p1, "u");
    auto p = svmp::FE::forms::FormExpr::stateField(p_f, *space_p1, "p");

    // Two test functions bound to the SAME FieldId but with DIFFERENT spaces
    auto v1 = svmp::FE::forms::FormExpr::testFunction(u_f, *space_p1, "v");
    auto v2 = svmp::FE::forms::FormExpr::testFunction(u_f, *space_p2, "v");

    auto residual = (u * v1).dx() + (p * v2).dx();

    try {
        svmp::FE::systems::installFormulation(sys, "op", {u_f, p_f}, residual);
        FAIL() << "Expected InvalidArgumentException for same-FieldId/different-space conflict";
    } catch (const svmp::FE::InvalidArgumentException& e) {
        const std::string msg = e.what();
        // May be caught by the same-FieldId conflict check or by the
        // duplicate-name-across-different-spaces check, depending on which
        // fires first. Both are valid early diagnostics.
        const bool has_conflict_msg = msg.find("conflicting test functions") != std::string::npos;
        const bool has_duplicate_msg = msg.find("duplicate TestFunction name") != std::string::npos;
        EXPECT_TRUE(has_conflict_msg || has_duplicate_msg)
            << "Error should mention 'conflicting test functions' or "
               "'duplicate TestFunction name', got: " << msg;
    }
}

// ============================================================================
// Full canonical workflow: register → symbols → BCs → installFormulation → setup → assemble
// ============================================================================

TEST(CanonicalWorkflow, FullWorkflow_SingleField)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    // 1. Register fields and operator
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    sys.addOperator("equations");

    // 2. Build field-bound symbols
    auto u = StateField(u_f, *space, "u");
    auto v = TestField(u_f, *space, "v");
    const auto k = FormExpr::constant(2.0);
    const auto f = FormExpr::constant(1.0);

    // 3. Write the residual
    auto residual = (k * inner(grad(u), grad(v)) - f * v).dx();

    // 4. Apply BCs (empty manager for this test)
    svmp::FE::systems::BoundaryConditionManager bc_manager;
    bc_manager.applyAll(sys, residual, u, v, u_f);

    // 5. Install
    svmp::FE::systems::installFormulation(sys, "equations", {u_f}, residual);

    // 6. Setup
    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    // 7. Assemble and verify
    const auto n = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n, 4);

    std::vector<Real> U = {0.1, 0.2, 0.3, 0.4};
    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();

    svmp::FE::systems::AssemblyRequest req;
    req.op = "equations";
    req.want_matrix = true;
    req.want_vector = true;
    (void)sys.assemble(req, state, &out, &out);

    // Matrix should have stiffness structure (symmetric, non-zero)
    double mat_norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            mat_norm += out.getMatrixEntry(i, j) * out.getMatrixEntry(i, j);
        }
    }
    EXPECT_GT(mat_norm, 0.0);

    // Vector should have both stiffness and source contributions
    double vec_norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        vec_norm += out.getVectorEntry(i) * out.getVectorEntry(i);
    }
    EXPECT_GT(vec_norm, 0.0);
}
