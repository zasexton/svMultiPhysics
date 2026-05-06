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
#include "Physics/Core/TemporalValues.h"

#include "FE/Core/FEException.h"
#include "FE/Systems/SystemState.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
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

TEST(TemporalValues, ReadsAndInterpolatesClampedScalarTable)
{
    const auto path = std::filesystem::temp_directory_path() / "svmp_temporal_values_clamped.dat";
    {
        std::ofstream out(path);
        out << "3 1\n"
            << "0.0 0.0\n"
            << "0.5 10.0\n"
            << "1.0 20.0\n";
    }

    const auto values = readTemporalValuesFile(path.string(), /*num_components=*/1, TemporalEndBehavior::Clamp);
    ASSERT_EQ(values->num_time_points, 3);
    ASSERT_EQ(values->num_components, 1);
    EXPECT_DOUBLE_EQ(values->firstTime(), 0.0);
    EXPECT_DOUBLE_EQ(values->lastTime(), 1.0);
    EXPECT_DOUBLE_EQ(values->firstValue(), 0.0);
    EXPECT_DOUBLE_EQ(values->lastValue(), 20.0);
    EXPECT_DOUBLE_EQ(values->interpolate(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(values->interpolate(0.25), 5.0);
    EXPECT_DOUBLE_EQ(values->interpolate(0.75), 15.0);
    EXPECT_DOUBLE_EQ(values->interpolate(2.0), 20.0);

    std::filesystem::remove(path);
}

TEST(TemporalValues, SupportsPeriodicEndBehaviorForLegacyFlowFiles)
{
    const auto path = std::filesystem::temp_directory_path() / "svmp_temporal_values_periodic.dat";
    {
        std::ofstream out(path);
        out << "2 1\n"
            << "0.0 0.0\n"
            << "1.0 10.0\n";
    }

    const auto values = readTemporalValuesFile(path.string(), /*num_components=*/1, TemporalEndBehavior::Periodic);
    EXPECT_DOUBLE_EQ(values->interpolate(0.25), 2.5);
    EXPECT_DOUBLE_EQ(values->interpolate(1.25), 2.5);
    EXPECT_DOUBLE_EQ(values->interpolate(1.0), 0.0);

    std::filesystem::remove(path);
}

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
