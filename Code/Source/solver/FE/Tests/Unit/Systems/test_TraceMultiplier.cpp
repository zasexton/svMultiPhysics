/**
 * @file test_TraceMultiplier.cpp
 * @brief Trace multipliers written as restricted volume fields (D54 LM3, T1)
 *
 * Setup: a scalar field held by the rows of a multiplier restricted to the
 * boundary is anchored by them (no automatic gauge), the multiplier is no
 * gauge candidate, and a multiplier whose rows couple only to DOFs an
 * essential condition fixes is refused.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/ConstraintPlan.h"
#include "Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "Constraints/SupportRestrictionConstraint.h"
#include "Core/FEException.h"
#include "Forms/BoundaryConditions.h"
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

} // namespace svmp::FE::test_trace_multiplier
