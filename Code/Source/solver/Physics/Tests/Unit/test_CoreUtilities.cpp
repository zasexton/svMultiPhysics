/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Core/Domain.h"
#include "Physics/Core/JITRuntimePolicy.h"
#include "Physics/Core/ParameterSchema.h"

#include "FE/Core/FEException.h"
#include "FE/Systems/SystemState.h"

#include <cstdlib>
#include <optional>
#include <string>

namespace svmp {
namespace Physics {
namespace test {

namespace {

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* key, std::optional<std::string> value)
        : key_(key)
    {
        if (const char* existing = std::getenv(key_); existing != nullptr) {
            original_ = std::string(existing);
        }
        set(std::move(value));
    }

    ~ScopedEnvVar() { set(original_); }

private:
    void set(std::optional<std::string> value)
    {
        if (value.has_value()) {
            setenv(key_, value->c_str(), 1);
        } else {
            unsetenv(key_);
        }
    }

    const char* key_{nullptr};
    std::optional<std::string> original_{};
};

} // namespace

TEST(MarkerSet, EmptyIsAll)
{
    MarkerSet s;
    EXPECT_TRUE(s.isAll());
    EXPECT_TRUE(s.contains(0));
    EXPECT_TRUE(s.contains(123));
    EXPECT_TRUE(s.contains(-7));
}

TEST(MarkerSet, NegativeMarkerIsAll)
{
    MarkerSet s;
    s.markers = {kAllBoundaryMarkers};
    EXPECT_TRUE(s.isAll());
    EXPECT_TRUE(s.contains(5));
}

TEST(MarkerSet, ContainsSpecificMarkers)
{
    MarkerSet s;
    s.markers = {1, 3, 7};
    EXPECT_FALSE(s.isAll());
    EXPECT_TRUE(s.contains(1));
    EXPECT_TRUE(s.contains(7));
    EXPECT_FALSE(s.contains(2));
}

TEST(ParameterSchema, ValidateHonorsDefaults)
{
    ParameterSchema schema;

    FE::params::Spec s;
    s.key = "alpha";
    s.type = FE::params::ValueType::Real;
    s.required = true;
    s.default_value = FE::params::Value{FE::Real(2.0)};
    schema.add(std::move(s), "test");

    FE::systems::SystemStateView state;
    EXPECT_NO_THROW(schema.validate(state));
}

TEST(ParameterSchema, ValidateThrowsWhenMissingRequired)
{
    ParameterSchema schema;

    FE::params::Spec s;
    s.key = "alpha";
    s.type = FE::params::ValueType::Real;
    s.required = true;
    schema.add(std::move(s), "test");

    FE::systems::SystemStateView state;
    EXPECT_THROW(schema.validate(state), FE::InvalidArgumentException);
}

TEST(ParameterSchema, FindAndClear)
{
    ParameterSchema schema;

    FE::params::Spec s;
    s.key = "alpha";
    s.type = FE::params::ValueType::Real;
    s.required = false;
    schema.add(std::move(s), "test");

    ASSERT_NE(schema.find("alpha"), nullptr);
    EXPECT_EQ(schema.find("alpha")->type, FE::params::ValueType::Real);

    schema.clear();
    EXPECT_TRUE(schema.specs().empty());
    EXPECT_EQ(schema.find("alpha"), nullptr);
}

TEST(JitRuntimePolicy, ModuleOptionsOverrideEnvironment)
{
    ScopedEnvVar env("SVMP_OOP_JIT_ENABLE", std::string("1"));

    EquationModuleInput input;
    input.module_options = "jit = false";

    EXPECT_FALSE(core::resolveOopJitEnable(input, true));
}

TEST(JitRuntimePolicy, EquationParamOverridesEnvironment)
{
    ScopedEnvVar env("SVMP_OOP_JIT_ENABLE", std::string("1"));

    EquationModuleInput input;
    input.equation_params["Enable_jit"] = ParameterValue{true, "off"};

    EXPECT_FALSE(core::resolveOopJitEnable(input, true));
}

TEST(JitRuntimePolicy, EnvironmentOverridesDefaultWhenNoEquationOverrideExists)
{
    ScopedEnvVar env("SVMP_OOP_JIT_ENABLE", std::string("0"));

    EquationModuleInput input;

    EXPECT_FALSE(core::resolveOopJitEnable(input, true));
}

TEST(JitRuntimePolicy, SpecializationModuleOptionsOverrideEnvironment)
{
    ScopedEnvVar env("SVMP_OOP_JIT_SPECIALIZATION_ENABLE", std::string("1"));

    EquationModuleInput input;
    input.module_options = "jit_specialization = false";

    EXPECT_FALSE(core::resolveOopJitSpecializationEnable(input, true));
}

TEST(JitRuntimePolicy, SpecializationEquationParamOverridesEnvironment)
{
    ScopedEnvVar env("SVMP_OOP_JIT_SPECIALIZATION_ENABLE", std::string("1"));

    EquationModuleInput input;
    input.equation_params["Enable_jit_specialization"] = ParameterValue{true, "off"};

    EXPECT_FALSE(core::resolveOopJitSpecializationEnable(input, true));
}

TEST(JitRuntimePolicy, SpecializationEnvironmentOverridesDefaultWhenNoEquationOverrideExists)
{
    ScopedEnvVar env("SVMP_OOP_JIT_SPECIALIZATION_ENABLE", std::string("0"));

    EquationModuleInput input;

    EXPECT_FALSE(core::resolveOopJitSpecializationEnable(input, true));
}

} // namespace test
} // namespace Physics
} // namespace svmp
