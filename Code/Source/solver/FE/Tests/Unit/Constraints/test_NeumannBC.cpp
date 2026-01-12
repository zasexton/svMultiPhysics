/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_NeumannBC.cpp
 * @brief Unit tests for NeumannBC class
 */

#include <gtest/gtest.h>

#include "Constraints/NeumannBC.h"
#include "Constraints/CoupledNeumannBC.h"

#include <array>
#include <cmath>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

TEST(NeumannBCTest, ConstantFluxConstructionAndEvaluate) {
    NeumannBC bc(7, 2.5);

    EXPECT_EQ(bc.getBoundaryId(), 7);
    EXPECT_FALSE(bc.isVectorValued());
    EXPECT_FALSE(bc.isTimeDependent());
    EXPECT_DOUBLE_EQ(bc.getConstantFlux(), 2.5);
    EXPECT_DOUBLE_EQ(bc.evaluateFlux(1.0, 2.0, 3.0), 2.5);
}

TEST(NeumannBCTest, SpatialFluxFunctionEvaluate) {
    NeumannBC bc(3, [](double x, double y, double z) { return x + 2.0 * y + 3.0 * z; });

    EXPECT_EQ(bc.getBoundaryId(), 3);
    EXPECT_FALSE(bc.isVectorValued());
    EXPECT_FALSE(bc.isTimeDependent());
    EXPECT_DOUBLE_EQ(bc.evaluateFlux(1.0, 2.0, 3.0), 1.0 + 4.0 + 9.0);
}

TEST(NeumannBCTest, TimeDependentFluxFunctionEvaluate) {
    NeumannBC bc(3, [](double x, double y, double z, double t) { return x + y + z + t; });

    EXPECT_EQ(bc.getBoundaryId(), 3);
    EXPECT_FALSE(bc.isVectorValued());
    EXPECT_TRUE(bc.isTimeDependent());
    EXPECT_DOUBLE_EQ(bc.evaluateFlux(1.0, 2.0, 3.0, 4.0), 10.0);
}

TEST(NeumannBCTest, ConstantTractionConstructionAndEvaluate) {
    NeumannBC bc(9, std::array<double, 3>{{1.0, 2.0, 3.0}});

    EXPECT_EQ(bc.getBoundaryId(), 9);
    EXPECT_TRUE(bc.isVectorValued());
    EXPECT_FALSE(bc.isTimeDependent());

    auto traction = bc.evaluateTraction(0.0, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(traction[0], 1.0);
    EXPECT_DOUBLE_EQ(traction[1], 2.0);
    EXPECT_DOUBLE_EQ(traction[2], 3.0);
}

TEST(NeumannBCTest, TractionFunctionEvaluate) {
    NeumannBC bc(2, [](double x, double y, double z) {
        return std::array<double, 3>{{x, y, z}};
    });

    EXPECT_TRUE(bc.isVectorValued());
    EXPECT_FALSE(bc.isTimeDependent());

    auto traction = bc.evaluateTraction(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(traction[0], 1.0);
    EXPECT_DOUBLE_EQ(traction[1], 2.0);
    EXPECT_DOUBLE_EQ(traction[2], 3.0);
}

TEST(NeumannBCTest, TimeDependentTractionFunctionEvaluate) {
    NeumannBC bc(2, [](double x, double y, double z, double t) {
        return std::array<double, 3>{{x + t, y + t, z + t}};
    });

    EXPECT_TRUE(bc.isVectorValued());
    EXPECT_TRUE(bc.isTimeDependent());

    auto traction = bc.evaluateTraction(1.0, 2.0, 3.0, 4.0);
    EXPECT_DOUBLE_EQ(traction[0], 5.0);
    EXPECT_DOUBLE_EQ(traction[1], 6.0);
    EXPECT_DOUBLE_EQ(traction[2], 7.0);
}

TEST(NeumannBCTest, TimeDependentPulsatileFlux) {
    const double Q_max = 2.0;
    const double period = 1.0;
    const double pi = std::acos(-1.0);

    NeumannBC bc(5, [Q_max, period, pi](double /*x*/, double /*y*/, double /*z*/, double t) {
        return Q_max * std::sin(2.0 * pi * t / period);
    });

    EXPECT_TRUE(bc.isTimeDependent());

    EXPECT_NEAR(bc.evaluateFlux(0.0, 0.0, 0.0, 0.0), 0.0, 1e-14);
    EXPECT_NEAR(bc.evaluateFlux(0.0, 0.0, 0.0, period / 4.0), Q_max, 1e-14);
    EXPECT_NEAR(bc.evaluateFlux(0.0, 0.0, 0.0, period / 2.0), 0.0, 1e-14);
    EXPECT_NEAR(bc.evaluateFlux(0.0, 0.0, 0.0, 3.0 * period / 4.0), -Q_max, 1e-14);
}

TEST(NeumannBCTest, SetFluxResetsVectorAndTimeFlags) {
    NeumannBC bc(1, [](double x, double /*y*/, double /*z*/, double t) { return x + t; });
    EXPECT_TRUE(bc.isTimeDependent());

    bc.setFlux(4.2);
    EXPECT_FALSE(bc.isVectorValued());
    EXPECT_FALSE(bc.isTimeDependent());
    EXPECT_DOUBLE_EQ(bc.evaluateFlux(1.0, 0.0, 0.0, 9.0), 4.2);
}

TEST(NeumannBCTest, SetTractionResetsScalarAndTimeFlags) {
    NeumannBC bc(1, 3.14);
    EXPECT_FALSE(bc.isVectorValued());

    bc.setTraction({{1.0, 0.0, -1.0}});
    EXPECT_TRUE(bc.isVectorValued());
    EXPECT_FALSE(bc.isTimeDependent());

    auto traction = bc.evaluateTraction(0.0, 0.0, 0.0, 10.0);
    EXPECT_DOUBLE_EQ(traction[0], 1.0);
    EXPECT_DOUBLE_EQ(traction[1], 0.0);
    EXPECT_DOUBLE_EQ(traction[2], -1.0);
}

TEST(NeumannBCTest, CollectionLookup) {
    NeumannBCCollection bcs;
    EXPECT_TRUE(bcs.empty());

    bcs.add(NeumannBC(1, 1.0));
    bcs.add(NeumannBC(2, std::array<double, 3>{{1.0, 2.0, 3.0}}));

    EXPECT_EQ(bcs.size(), 2u);
    EXPECT_TRUE(bcs.hasBC(1));
    EXPECT_TRUE(bcs.hasBC(2));
    EXPECT_FALSE(bcs.hasBC(3));

    const auto* bc = bcs.find(2);
    ASSERT_NE(bc, nullptr);
    EXPECT_TRUE(bc->isVectorValued());
}

TEST(CoupledBCTest, ContextPropagation) {
    forms::BoundaryFunctionalResults integrals;
    integrals.set("Q", 3.0);

    systems::AuxiliaryState aux;
    systems::AuxiliaryStateSpec spec;
    spec.size = 1;
    spec.name = "R";
    std::array<Real, 1> initial{{2.0}};
    aux.registerState(spec, initial);

    CoupledBCContext ctx{integrals, aux, /*t=*/1.0, /*dt=*/0.1};
    EXPECT_EQ(ctx.integralsValues().size(), 1u);
    EXPECT_EQ(ctx.auxValues().size(), 1u);
    EXPECT_DOUBLE_EQ(ctx.integrals.get("Q"), 3.0);
    EXPECT_DOUBLE_EQ(ctx.aux_state["R"], 2.0);
    EXPECT_DOUBLE_EQ(ctx.t, 1.0);
    EXPECT_DOUBLE_EQ(ctx.dt, 0.1);
}

TEST(CoupledNeumannBCTest, ScalarEvaluator) {
    forms::BoundaryFunctionalResults integrals;
    integrals.set("Q", 3.0);

    systems::AuxiliaryState aux;
    systems::AuxiliaryStateSpec spec;
    spec.size = 1;
    spec.name = "R";
    std::array<Real, 1> initial{{2.0}};
    aux.registerState(spec, initial);

    CoupledBCContext ctx{integrals, aux, /*t=*/0.0, /*dt=*/0.0};

    CoupledNeumannBC bc(
        /*boundary_marker=*/7,
        /*required_integrals=*/{},
        [](const CoupledBCContext& c, Real x, Real /*y*/, Real /*z*/) {
            return c.integrals.get("Q") + c.aux_state["R"] + x;
        });

    EXPECT_EQ(bc.boundaryMarker(), 7);
    EXPECT_DOUBLE_EQ(bc.evaluate(ctx, 1.0, 0.0, 0.0), 6.0);
    auto v = bc.evaluateVector(ctx, 1.0, 0.0, 0.0, {{1.0, 0.0, 0.0}});
    EXPECT_DOUBLE_EQ(v[0], 6.0);
    EXPECT_DOUBLE_EQ(v[1], 0.0);
    EXPECT_DOUBLE_EQ(v[2], 0.0);
}

TEST(CoupledNeumannBCTest, VectorEvaluator) {
    forms::BoundaryFunctionalResults integrals;
    systems::AuxiliaryState aux;
    CoupledBCContext ctx{integrals, aux, /*t=*/0.0, /*dt=*/0.0};

    CoupledNeumannBC bc(
        /*boundary_marker=*/1,
        /*required_integrals=*/{},
        [](const CoupledBCContext& /*c*/, Real /*x*/, Real /*y*/, Real /*z*/, const std::array<Real, 3>& normal) {
            return std::array<Real, 3>{{2.0 * normal[0], 2.0 * normal[1], 2.0 * normal[2]}};
        });

    EXPECT_THROW((void)bc.evaluate(ctx, 0.0, 0.0, 0.0), NotImplementedException);

    const auto traction = bc.evaluateVector(ctx, 0.0, 0.0, 0.0, {{0.0, -1.0, 0.5}});
    EXPECT_DOUBLE_EQ(traction[0], 0.0);
    EXPECT_DOUBLE_EQ(traction[1], -2.0);
    EXPECT_DOUBLE_EQ(traction[2], 1.0);
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
