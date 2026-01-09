/**
 * @file test_ODEIntegrator.cpp
 * @brief Unit tests for Systems/ODEIntegrator
 */

#include <gtest/gtest.h>

#include <cmath>

#include "Forms/BoundaryFunctional.h"
#include "Systems/AuxiliaryState.h"
#include "Systems/ODEIntegrator.h"

using svmp::FE::Real;

namespace {

svmp::FE::systems::AuxiliaryState makeState(std::string name, Real x0)
{
    svmp::FE::systems::AuxiliaryState st;
    svmp::FE::systems::AuxiliaryStateSpec spec;
    spec.size = 1;
    spec.name = std::move(name);
    st.registerState(spec, std::span<const Real>(&x0, 1));
    return st;
}

} // namespace

TEST(ODEIntegrator, ForwardEuler_LinearDecay)
{
    using namespace svmp::FE;

    auto st = makeState("X", 1.0);
    forms::BoundaryFunctionalResults integrals;

    systems::ODEIntegrator::advance(
        systems::ODEMethod::ForwardEuler,
        "X",
        st,
        [](const systems::AuxiliaryState& state, const forms::BoundaryFunctionalResults&, Real) {
            return -state["X"];
        },
        integrals,
        /*t=*/0.1,
        /*dt=*/0.1);

    EXPECT_NEAR(st["X"], 0.9, 1e-12);
}

TEST(ODEIntegrator, BackwardEuler_LinearDecay)
{
    using namespace svmp::FE;

    auto st = makeState("X", 1.0);
    forms::BoundaryFunctionalResults integrals;

    systems::ODEIntegrator::advance(
        systems::ODEMethod::BackwardEuler,
        "X",
        st,
        [](const systems::AuxiliaryState& state, const forms::BoundaryFunctionalResults&, Real) {
            return -state["X"];
        },
        integrals,
        /*t=*/0.1,
        /*dt=*/0.1);

    EXPECT_NEAR(st["X"], 1.0 / 1.1, 1e-10);
}

TEST(ODEIntegrator, RK4_LinearDecay)
{
    using namespace svmp::FE;

    auto st = makeState("X", 1.0);
    forms::BoundaryFunctionalResults integrals;

    systems::ODEIntegrator::advance(
        systems::ODEMethod::RK4,
        "X",
        st,
        [](const systems::AuxiliaryState& state, const forms::BoundaryFunctionalResults&, Real) {
            return -state["X"];
        },
        integrals,
        /*t=*/0.0,
        /*dt=*/0.1);

    // Compare to exact exp(-dt).
    EXPECT_NEAR(st["X"], std::exp(-0.1), 1e-6);
}

TEST(ODEIntegrator, BDF2_LinearDecay_WithHistory)
{
    using namespace svmp::FE;

    const Real dt = 0.1;

    // Start at X0 = 1.0 (committed).
    auto st = makeState("X", 1.0);
    forms::BoundaryFunctionalResults integrals;

    // Step 1: Backward Euler to compute X1, then commit to create history.
    systems::ODEIntegrator::advance(
        systems::ODEMethod::BackwardEuler,
        "X",
        st,
        [](const systems::AuxiliaryState& state, const forms::BoundaryFunctionalResults&, Real) {
            return -state["X"];
        },
        integrals,
        /*t=*/dt,
        /*dt=*/dt);

    const Real x1 = st["X"];
    st.commitTimeStep();

    // Step 2: BDF2 using (X1, X0) to compute X2.
    systems::ODEIntegrator::advance(
        systems::ODEMethod::BDF2,
        "X",
        st,
        [](const systems::AuxiliaryState& state, const forms::BoundaryFunctionalResults&, Real) {
            return -state["X"];
        },
        integrals,
        /*t=*/2.0 * dt,
        /*dt=*/dt);

    const Real expected = (4.0 * x1 - 1.0) / (3.0 + 2.0 * dt);
    EXPECT_NEAR(st["X"], expected, 1e-8);
}
