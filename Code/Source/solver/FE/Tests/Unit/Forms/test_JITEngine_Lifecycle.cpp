/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/JIT/JITEngine.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

namespace svmp {
namespace FE {
namespace forms {
namespace test {

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

TEST(JITEngine, CreateAndQueryTargetProperties)
{
    auto options = makeUnitTestJITOptions();
    options.dump_directory = "svmp_fe_jit_dumps_tests_engine";

    auto engine = jit::JITEngine::create(options);

#if SVMP_FE_ENABLE_LLVM_JIT
    ASSERT_NE(engine, nullptr);
    EXPECT_TRUE(engine->available());
    EXPECT_FALSE(engine->targetTriple().empty());
    EXPECT_FALSE(engine->dataLayoutString().empty());
    EXPECT_FALSE(engine->cpuName().empty());

    (void)engine->cpuFeaturesString();

    engine->resetObjectCacheStats();
    (void)engine->objectCacheStats();
#else
    EXPECT_EQ(engine, nullptr);
#endif
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

