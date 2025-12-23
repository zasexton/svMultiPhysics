/**
 * @file test_VectorSpaceTraces.cpp
 * @brief Unit tests for HCurlSpace/HDivSpace trace operators
 */

#include <gtest/gtest.h>

#include "FE/Spaces/HCurlSpace.h"
#include "FE/Spaces/HDivSpace.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

namespace {

FunctionSpace::Value as_vec3(Real x, Real y, Real z = Real(0)) {
    FunctionSpace::Value v{};
    v[0] = x;
    v[1] = y;
    v[2] = z;
    return v;
}

} // namespace

TEST(VectorSpaceTraces, HCurlTangentialTraceIsProjection) {
    HCurlSpace space(ElementType::Quad4, 0);
    const auto ndofs = space.dofs_per_element();
    ASSERT_GT(ndofs, 0u);

    std::vector<Real> dof_values(ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        dof_values[i] = Real(0.25) * Real(i + 1);
    }

    std::vector<HCurlSpace::Vec3> points = {
        as_vec3(Real(-1), Real(0)),
        as_vec3(Real(1), Real(0)),
        as_vec3(Real(0), Real(-1)),
        as_vec3(Real(0), Real(1)),
        as_vec3(Real(0), Real(0)),
    };

    const HCurlSpace::Vec3 normal = as_vec3(Real(0), Real(0), Real(1));
    auto tangential = space.tangential_trace(dof_values, points, normal);

    ASSERT_EQ(tangential.size(), points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto v = space.evaluate(points[i], dof_values);
        const Real vdotn = v.dot(normal);
        HCurlSpace::Vec3 expected{};
        expected[0] = v[0] - normal[0] * vdotn;
        expected[1] = v[1] - normal[1] * vdotn;
        expected[2] = v[2] - normal[2] * vdotn;

        EXPECT_NEAR(tangential[i][0], expected[0], 1e-12);
        EXPECT_NEAR(tangential[i][1], expected[1], 1e-12);
        EXPECT_NEAR(tangential[i][2], expected[2], 1e-12);
        EXPECT_NEAR(tangential[i].dot(normal), 0.0, 1e-12);
    }
}

TEST(VectorSpaceTraces, HDivNormalTraceMatchesDotProduct) {
    HDivSpace space(ElementType::Quad4, 0);
    const auto ndofs = space.dofs_per_element();
    ASSERT_GT(ndofs, 0u);

    std::vector<Real> dof_values(ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        dof_values[i] = Real(-0.5) + Real(0.1) * Real(i);
    }

    std::vector<HDivSpace::Vec3> points = {
        as_vec3(Real(-1), Real(0)),
        as_vec3(Real(1), Real(0)),
        as_vec3(Real(0), Real(0)),
    };

    const HDivSpace::Vec3 normal = as_vec3(Real(1), Real(0), Real(0));
    auto normal_trace = space.normal_trace(dof_values, points, normal);

    ASSERT_EQ(normal_trace.size(), points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto v = space.evaluate(points[i], dof_values);
        const Real expected = v.dot(normal);
        EXPECT_NEAR(normal_trace[i], expected, 1e-12);
    }
}
