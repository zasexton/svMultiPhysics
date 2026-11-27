/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TESTS_UNIT_FORMS_RUNTIME_SCALAR_FUNCTIONAL_TEST_H
#define SVMP_FE_TESTS_UNIT_FORMS_RUNTIME_SCALAR_FUNCTIONAL_TEST_H

#include "Forms/JIT/JITFunctionalKernelWrapper.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>

namespace svmp::FE::forms::test {

class RuntimeScalarFunctionalTest : public ::testing::Test {
protected:
    struct Sample {
        std::array<Real, 2> operands;
        Real expected;
    };

    void SetUp() override
    {
        requireLLVMJITOrSkip();
    }

    void expectSamples(const FormExpr& expression, std::initializer_list<Sample> samples)
    {
        SingleTetraMeshAccess mesh;
        auto dofs = createSingleTetraDofMap();
        spaces::H1Space space(ElementType::Tetra4, 1);
        assembly::FunctionalAssembler assembler;
        assembler.setMesh(mesh);
        assembler.setDofMap(dofs);
        assembler.setSpace(space);
        auto interpreted = makeFunctionalFormKernel(expression, FunctionalFormKernel::Domain::Cell);
        auto fallback = makeThrowingTotalKernelFor(expression, true, false);
        jit::JITFunctionalKernelWrapper compiled(
            fallback, expression, jit::JITFunctionalKernelWrapper::Domain::Cell, makeUnitTestJITOptions());

        // Both operators integrate a spatially constant runtime expression on the unit tetrahedron.
        for (const auto& sample : samples) {
            SCOPED_TRACE(::testing::Message() << "a=" << sample.operands[0] << ", b=" << sample.operands[1]);
            assembler.setJITConstants(sample.operands);
            const Real tolerance = 64 * std::numeric_limits<Real>::epsilon() *
                                   std::max(Real(1), std::abs(sample.expected));
            EXPECT_NEAR(6 * assembler.assembleScalar(*interpreted), sample.expected, tolerance);
            EXPECT_NEAR(6 * assembler.assembleScalar(compiled), sample.expected, tolerance);
        }
    }

    const FormExpr a = FormExpr::parameterRef(0);
    const FormExpr b = FormExpr::parameterRef(1);
};

} // namespace svmp::FE::forms::test

#endif
