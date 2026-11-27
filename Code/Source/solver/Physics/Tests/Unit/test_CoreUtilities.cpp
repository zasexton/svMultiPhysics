/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Core/Domain.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/CertificateEnvironment.h"
#include "Physics/Core/JITRuntimePolicy.h"
#include "Physics/Core/PhysicsModule.h"
#include "Physics/Core/TemporalValues.h"

#include "FE/Forms/JIT/LLVMJITBuildInfo.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/TimeSeries.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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

class ExposedPhysicsModule final : public PhysicsModule {
public:
    using PhysicsModule::applyPhysicsJITOptions;
    using PhysicsModule::physicsCompilerOptions;
    using PhysicsModule::physicsInstallOptions;

    void registerOn(FE::systems::FESystem& /*system*/) const override {}
};

} // namespace

class FreeSurfaceCertificateEnvironment : public ::testing::Test {
protected:
    void SetUp() override
    {
        namespace fs = formulations::navier_stokes::free_surface;
        for (const auto* setting : fs::environmentRegistry().declarations()) {
            if (std::string_view(setting->owner) == "FE/Analysis/ConstraintRangeCertificate") {
                clean_.push_back(std::make_unique<ScopedEnvVar>(setting->name, std::nullopt));
            }
        }
    }

private:
    std::vector<std::unique_ptr<ScopedEnvVar>> clean_;
};

TEST_F(FreeSurfaceCertificateEnvironment, KeepsAliasPrecedenceAndReadsCachePolicyAtEachSample)
{
    namespace fs = formulations::navier_stokes::free_surface;
    ScopedEnvVar primary("SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "off");
    ScopedEnvVar fallback("SVMP_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC", "1");
    ScopedEnvVar canonical("SVMP_FE_CONSTRAINT_RANGE_DIAGNOSTIC", std::nullopt);
    ScopedEnvVar repeat("SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY", "0");
    const auto method = fs::newtonAnalysisMethodFromEnvironment().constraint_range;
    EXPECT_FALSE(method.observe);
    ASSERT_NE(method.reassemble_each_sample, nullptr);
    EXPECT_FALSE(method.reassemble_each_sample());
    {
        ScopedEnvVar scoped_repeat("SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC_EVERY_ASSEMBLY", "on");
        EXPECT_TRUE(method.reassemble_each_sample());
        ScopedEnvVar canonical_repeat("SVMP_FE_CONSTRAINT_RANGE_REASSEMBLE_EACH_SAMPLE", "false");
        EXPECT_FALSE(method.reassemble_each_sample());
    }
    ScopedEnvVar empty_primary("SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC", "");
    EXPECT_TRUE(fs::newtonAnalysisMethodFromEnvironment().constraint_range.observe);
    EXPECT_FALSE(FE::analysis::constraintRangeMethodFromEnvironment().observe);
    ScopedEnvVar override("SVMP_FE_CONSTRAINT_RANGE_DIAGNOSTIC", "0");
    EXPECT_FALSE(fs::newtonAnalysisMethodFromEnvironment().constraint_range.observe);
}

TEST(FreeSurfaceGeometryEnvironment, PreservesFullCellPointRetentionFlag)
{
    namespace fs = formulations::navier_stokes::free_surface;
    constexpr auto legacy = "SVMP_KEEP_DRY_CELL_POINTS";
    EXPECT_EQ(FE::environment::EnvironmentNameRegistry::solver().find(legacy), nullptr);
    EXPECT_NE(fs::environmentRegistry().find(legacy), nullptr);
    EXPECT_NE(FE::environment::EnvironmentNameRegistry::solver().find("SVMP_KEEP_FULL_CELL_POINTS"), nullptr);
    for (const auto* value : {"", "0", "01"}) {
        ScopedEnvVar flag(legacy, value);
        EXPECT_FALSE(fs::keepFullCellPointsFromEnvironment());
    }
    for (const auto* value : {"1", "yes", "false"}) {
        ScopedEnvVar flag(legacy, value);
        EXPECT_TRUE(fs::keepFullCellPointsFromEnvironment());
    }
}

TEST_F(FreeSurfaceCertificateEnvironment, SuppliesThePhysicalRecordVocabulary)
{
    namespace fs = formulations::navier_stokes::free_surface;
    const auto method = fs::newtonAnalysisMethodFromEnvironment();
    const auto& records = method.constraint_range.vocabulary();
    EXPECT_EQ(records.range, "pressure_representability");
    EXPECT_EQ(records.work_norms[2], "gravitational_potential_virtual_work_norm");
    EXPECT_EQ(records.reason("constant_field_action_nonfinite"), "constant_pressure_action_nonfinite");
    EXPECT_EQ(FE::analysis::ConstraintRangeMethod{}.vocabulary().range, "constraint_range");
}

TEST_F(FreeSurfaceCertificateEnvironment, PreservesThresholdValidationAndEffectiveConfiguration)
{
    namespace fs = formulations::navier_stokes::free_surface;
    constexpr auto range = "SVMP_NS_FREE_SURFACE_PRESSURE_REPRESENTABILITY_MAX_RELATIVE_DISTANCE";
    constexpr auto kkt = "SVMP_FREE_SURFACE_CONSTANT_PRESSURE_KKT_MAX_RELATIVE_DISTANCE";
    constexpr auto initialize = "SVMP_FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER";
    ScopedEnvVar threshold(range, "0.25");
    ScopedEnvVar constant(kkt, "1e-4");
    ScopedEnvVar initializer(initialize, "yes");
    auto method = fs::newtonAnalysisMethodFromEnvironment().constraint_range;
    EXPECT_EQ(method.max_range_distance, 0.25);
    EXPECT_EQ(method.max_constant_field_kkt_distance, 1e-4);
    EXPECT_TRUE(method.initialize_constraint_field);
    const auto& names = fs::environmentRegistry();
    const auto& base = FE::environment::EnvironmentNameRegistry::solver();
    std::size_t certificate_aliases = 0;
    for (const auto* declaration : names.declarations()) {
        if (std::string_view(declaration->owner) == "FE/Analysis/ConstraintRangeCertificate" &&
            !base.find(declaration->name)) {
            ++certificate_aliases;
        }
    }
    EXPECT_EQ(certificate_aliases, 12u);
    EXPECT_TRUE(names.declaration(range).marksNonReference());
    EXPECT_STREQ(names.declaration(range).owner, "FE/Analysis/ConstraintRangeCertificate");
    EXPECT_EQ(FE::environment::EnvironmentNameRegistry::solver().find(range), nullptr);
    EXPECT_TRUE(names.declaration("SVMP_FE_CONSTRAINT_RANGE_MAX_RELATIVE_DISTANCE").marksNonReference());
    const auto record = names.configurationRecord();
    EXPECT_NE(record.find(range), std::string::npos);
    EXPECT_NE(record.find(kkt), std::string::npos);
    EXPECT_NE(record.find(initialize), std::string::npos);
    ScopedEnvVar malformed("SVMP_FE_CONSTRAINT_RANGE_MAX_RELATIVE_DISTANCE", "0.01bad");
    const auto failed = fs::newtonAnalysisMethodFromEnvironment().constraint_range.max_range_distance;
    ASSERT_TRUE(failed.has_value());
    EXPECT_TRUE(std::isnan(*failed));
}

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

TEST(TemporalValues, FETimeSeriesInterpolatesTheDeckFilesBitForBit)
{
    // The temporal-values files of decks under tests/cases: the inflow of the
    // RCR pipe (33 points, periodic in the solver), the genBC inflow (200
    // points) and the ventricle pressure (1001 points, clamped).
    auto root = std::filesystem::path(__FILE__).parent_path();
    for (int i = 0; i < 6; ++i) {
        root = root.parent_path();
    }
    const std::array<std::filesystem::path, 3> files{
        root / "tests" / "cases" / "fluid" / "pipe_RCR_3d" / "lumen_inlet.flow",
        root / "tests" / "cases" / "fluid" / "pipe_RCR_genBC" / "lumen_inlet.flw",
        root / "tests" / "cases" / "struct" / "LV_HolzapfelOgden_passive_CANN" / "endo_pressure.dat"};
    const std::array<std::pair<TemporalEndBehavior, FE::systems::TimeSeriesEnd>, 2> ends{
        std::pair{TemporalEndBehavior::Periodic, FE::systems::TimeSeriesEnd::Periodic},
        std::pair{TemporalEndBehavior::Clamp, FE::systems::TimeSeriesEnd::Clamp}};
    for (const auto& file : files) {
        ASSERT_TRUE(std::filesystem::exists(file)) << file;
        for (const auto& [physics_end, fe_end] : ends) {
            const auto reference = readTemporalValuesFile(file.string(), /*num_components=*/1, physics_end);
            const auto series = FE::systems::readTimeSeries(file.string(), fe_end);
            ASSERT_EQ(series.times().size(), static_cast<std::size_t>(reference->num_time_points));
            for (int i = 0; i < reference->num_time_points; ++i) {
                const double ti = reference->t[static_cast<std::size_t>(i)];
                EXPECT_EQ(series(ti), reference->interpolate(ti)) << file << " t=" << ti;
            }
            // Before, inside and after the table, over three periods, off the table times.
            const double t0 = reference->firstTime();
            const double span = reference->period();
            for (int k = -97; k <= 3 * 97; ++k) {
                const double t = t0 + span * (static_cast<double>(k) + 0.371) / 97.0;
                EXPECT_EQ(series(t), reference->interpolate(t)) << file << " t=" << t;
            }
        }
    }
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

TEST(JitRuntimePolicy, PolicyDefaultsToImplicitJITRequest)
{
    const core::PhysicsJITPolicy policy{};

    EXPECT_TRUE(policy.enable);
    EXPECT_FALSE(policy.enable_was_explicitly_set);
    EXPECT_TRUE(policy.specialization);
    EXPECT_EQ(policy.optimization_level, 3);
    EXPECT_TRUE(policy.specialize_n_qpts);
    EXPECT_TRUE(policy.specialize_dofs);
}

TEST(JitRuntimePolicy, PolicyModuleOptionsOverrideEnvironment)
{
    ScopedEnvVar env("SVMP_OOP_JIT_ENABLE", std::string("1"));
    ScopedEnvVar specialization_env("SVMP_OOP_JIT_SPECIALIZATION_ENABLE", std::string("1"));

    EquationModuleInput input;
    input.module_options = "jit = false; jit_specialization = false";

    const auto policy = core::resolveOopJitPolicy(input);
    EXPECT_FALSE(policy.enable);
    EXPECT_TRUE(policy.enable_was_explicitly_set);
    EXPECT_FALSE(policy.specialization);
}

TEST(JitRuntimePolicy, PolicyEquationParamsOverrideEnvironment)
{
    ScopedEnvVar env("SVMP_OOP_JIT_ENABLE", std::string("1"));
    ScopedEnvVar specialization_env("SVMP_OOP_JIT_SPECIALIZATION_ENABLE", std::string("1"));

    EquationModuleInput input;
    input.equation_params["Enable_jit"] = ParameterValue{true, "off"};
    input.equation_params["Enable_jit_specialization"] = ParameterValue{true, "off"};

    const auto policy = core::resolveOopJitPolicy(input);
    EXPECT_FALSE(policy.enable);
    EXPECT_TRUE(policy.enable_was_explicitly_set);
    EXPECT_FALSE(policy.specialization);
}

TEST(JitRuntimePolicy, PolicyEnvironmentOverridesDefault)
{
    ScopedEnvVar env("SVMP_OOP_JIT_ENABLE", std::string("0"));
    ScopedEnvVar specialization_env("SVMP_OOP_JIT_SPECIALIZATION_ENABLE", std::string("0"));

    EquationModuleInput input;

    const auto policy = core::resolveOopJitPolicy(input, core::PhysicsJITPolicy{
                                                            .enable = true,
                                                            .specialization = true});
    EXPECT_FALSE(policy.enable);
    EXPECT_TRUE(policy.enable_was_explicitly_set);
    EXPECT_FALSE(policy.specialization);
}

TEST(JitRuntimePolicy, PolicyWithoutOverrideRemainsImplicit)
{
    ScopedEnvVar env("SVMP_OOP_JIT_ENABLE", std::nullopt);
    ScopedEnvVar fe_env("SVMP_FE_JIT_ENABLE", std::nullopt);

    EquationModuleInput input;
    const auto policy = core::resolveOopJitPolicy(input);

    EXPECT_TRUE(policy.enable);
    EXPECT_FALSE(policy.enable_was_explicitly_set);
}

TEST(JitRuntimePolicy, ExplicitTrueIsRecordedFromModuleOptions)
{
    ScopedEnvVar env("SVMP_OOP_JIT_ENABLE", std::string("0"));

    EquationModuleInput input;
    input.module_options = "jit = true";
    const auto policy = core::resolveOopJitPolicy(input);

    EXPECT_TRUE(policy.enable);
    EXPECT_TRUE(policy.enable_was_explicitly_set);
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

TEST(PhysicsModuleJITOptions, CompilerOptionsMaskJITWhenUnavailable)
{
    ExposedPhysicsModule module;
    core::PhysicsJITPolicy policy{};
    policy.enable = true;

    const auto options = module.physicsCompilerOptions(policy);

    EXPECT_EQ(options.jit.enable, FE::forms::jit::llvmJITEnabled());
    EXPECT_EQ(options.jit.optimization_level, 3);
    EXPECT_TRUE(options.jit.specialization.enable);
    EXPECT_TRUE(options.jit.specialization.specialize_n_qpts);
    EXPECT_TRUE(options.jit.specialization.specialize_dofs);
}

TEST(PhysicsModuleJITOptions, ImplicitUnavailableJITFallsBack)
{
    ExposedPhysicsModule module;
    const core::PhysicsJITPolicy policy{};

    EXPECT_NO_THROW({
        const auto options = module.physicsCompilerOptions(policy);
        EXPECT_EQ(options.jit.enable, FE::forms::jit::llvmJITEnabled());
    });
}

TEST(PhysicsModuleJITOptions, ExplicitUnavailableJITFailsClosed)
{
    ExposedPhysicsModule module;
    core::PhysicsJITPolicy policy{};
    policy.enable = true;
    policy.enable_was_explicitly_set = true;

    if (FE::forms::jit::llvmJITEnabled()) {
        EXPECT_TRUE(module.physicsCompilerOptions(policy).jit.enable);
        return;
    }

    try {
        (void)module.physicsCompilerOptions(policy);
        FAIL() << "explicit jit=true unexpectedly fell back to the interpreter";
    } catch (const std::runtime_error& error) {
        const std::string diagnostic = error.what();
        EXPECT_NE(diagnostic.find("jit=true was explicitly requested"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("FE_ENABLE_LLVM_JIT"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("set jit=false"), std::string::npos);
    }
}

TEST(PhysicsModuleJITOptions, ExplicitFalseRemainsDisabled)
{
    ExposedPhysicsModule module;
    core::PhysicsJITPolicy policy{};
    policy.enable = false;
    policy.enable_was_explicitly_set = true;

    EXPECT_FALSE(module.physicsCompilerOptions(policy).jit.enable);
}

TEST(PhysicsModuleJITOptions, CompilerOptionsHonorDisabledPolicy)
{
    ExposedPhysicsModule module;
    core::PhysicsJITPolicy policy{};
    policy.enable = false;
    policy.specialization = false;
    policy.specialize_n_qpts = false;
    policy.specialize_dofs = false;

    auto options = module.physicsCompilerOptions(policy);

    EXPECT_FALSE(options.jit.enable);
    EXPECT_EQ(options.jit.optimization_level, 3);
    EXPECT_FALSE(options.jit.specialization.enable);
    EXPECT_FALSE(options.jit.specialization.specialize_n_qpts);
    EXPECT_FALSE(options.jit.specialization.specialize_dofs);

    policy.optimization_level = 1;
    module.applyPhysicsJITOptions(options, policy);
    EXPECT_EQ(options.jit.optimization_level, 1);
}

TEST(PhysicsModuleJITOptions, InstallOptionsUseSharedCompilerPolicy)
{
    ExposedPhysicsModule module;
    core::PhysicsJITPolicy policy{};
    policy.enable = true;
    policy.optimization_level = 3;
    policy.specialization = false;

    const auto install = module.physicsInstallOptions(policy);

    EXPECT_EQ(install.compiler_options.jit.enable, FE::forms::jit::llvmJITEnabled());
    EXPECT_EQ(install.compiler_options.jit.optimization_level, 3);
    EXPECT_FALSE(install.compiler_options.jit.specialization.enable);
}

} // namespace test
} // namespace Physics
} // namespace svmp
