/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/JIT/JITEngine.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <cstdlib>
#include <optional>
#include <string>

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

#if SVMP_FE_ENABLE_LLVM_JIT
namespace {

// Sets an environment variable for the lifetime of the object and restores
// the previous value afterwards.
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value)
        : name_(name)
    {
        if (const char* old = std::getenv(name)) {
            old_ = std::string(old);
        }
        ::setenv(name, value, 1);
    }
    ~ScopedEnv()
    {
        if (old_) {
            ::setenv(name_, old_->c_str(), 1);
        } else {
            ::unsetenv(name_);
        }
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    const char* name_;
    std::optional<std::string> old_{};
};

[[nodiscard]] bool hasFeature(const std::string& features, const std::string& name)
{
    return ("," + features + ",").find("," + name + ",") != std::string::npos;
}

} // namespace

TEST(JITEngine, DefaultTargetIsHostWithoutCodegenOptions)
{
    auto engine = jit::JITEngine::create(makeUnitTestJITOptions());
    ASSERT_NE(engine, nullptr);
    EXPECT_TRUE(engine->codegenOptionsString().empty());
}

TEST(JITEngine, StrictFPContractionIsReportedAsCodegenOption)
{
    ScopedEnv contract("SVMP_JIT_FP_CONTRACT", "off");
    auto engine = jit::JITEngine::create(makeUnitTestJITOptions());
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->codegenOptionsString(), "fp-contract=off");
}

#if defined(__x86_64__) || defined(_M_X64)
TEST(JITEngine, GenericX86LevelReplacesHostTarget)
{
    auto host = jit::JITEngine::create(makeUnitTestJITOptions());
    ASSERT_NE(host, nullptr);

    ScopedEnv cpu("SVMP_JIT_CPU", "x86-64");
    auto engine = jit::JITEngine::create(makeUnitTestJITOptions());
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->cpuName(), "x86-64");
    EXPECT_TRUE(hasFeature(engine->cpuFeaturesString(), "sse2"));
    EXPECT_FALSE(hasFeature(engine->cpuFeaturesString(), "avx"));
    EXPECT_EQ(engine->dataLayoutString(), host->dataLayoutString());
}

TEST(JITEngine, UnknownTargetFallsBackToHost)
{
    auto host = jit::JITEngine::create(makeUnitTestJITOptions());
    ASSERT_NE(host, nullptr);

    ScopedEnv cpu("SVMP_JIT_CPU", "not-a-cpu");
    auto engine = jit::JITEngine::create(makeUnitTestJITOptions());
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->cpuName(), host->cpuName());
    EXPECT_EQ(engine->cpuFeaturesString(), host->cpuFeaturesString());
}
#endif
#endif

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

