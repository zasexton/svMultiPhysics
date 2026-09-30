/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Logger level configuration tests.
//
// FE_LOG_LEVEL is read when the logger singleton is first used, so the
// environment-driven case runs in its own process: CTest registers
// FE_Core_Logger_Environment_Tests, which runs LoggerEnvironment.* with
// FE_LOG_LEVEL=WARNING.  In the plain FE_Core_Tests run (variable unset) that
// suite skips and LoggerDefaults checks that the default level is unchanged.
//
// The test only uses header-inline Logger API plus parse_log_level(), mirroring
// a solver binary that links the static FE library.

#include <gtest/gtest.h>

#include "Core/Logger.h"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>

namespace svmp {
namespace FE {
namespace test {

namespace {

constexpr const char* kProbeTag = "svmp-logger-level-probe";

std::atomic<int> g_info_probes{0};
std::atomic<int> g_warning_probes{0};

void installProbeCounterOnce()
{
    static std::once_flag once;
    std::call_once(once, []() {
        Logger::instance().add_handler([](const LogMessage& msg) {
            if (msg.message.find(kProbeTag) == std::string::npos) {
                return;
            }
            if (msg.level == LogLevel::INFO) {
                g_info_probes.fetch_add(1);
            } else if (msg.level == LogLevel::WARNING) {
                g_warning_probes.fetch_add(1);
            }
        });
    });
}

[[nodiscard]] const char* envLogLevel()
{
    return std::getenv("FE_LOG_LEVEL");
}

} // namespace

TEST(LoggerLevelParsing, AcceptsDocumentedNamesCaseInsensitively)
{
    EXPECT_EQ(parse_log_level("DEBUG"), LogLevel::DEBUG);
    EXPECT_EQ(parse_log_level("info"), LogLevel::INFO);
    EXPECT_EQ(parse_log_level("WARNING"), LogLevel::WARNING);
    EXPECT_EQ(parse_log_level("warn"), LogLevel::WARNING);
    EXPECT_EQ(parse_log_level("Error"), LogLevel::ERROR);
    EXPECT_EQ(parse_log_level("CRITICAL"), LogLevel::CRITICAL);
    EXPECT_EQ(parse_log_level("crit"), LogLevel::CRITICAL);
    EXPECT_EQ(parse_log_level("off"), LogLevel::OFF);

    EXPECT_FALSE(parse_log_level("").has_value());
    EXPECT_FALSE(parse_log_level("verbose").has_value());
    EXPECT_FALSE(parse_log_level("WARNINGS").has_value());
}

TEST(LoggerDefaults, DefaultLevelIsInfoWithoutEnvironment)
{
    if (envLogLevel() != nullptr) {
        GTEST_SKIP() << "FE_LOG_LEVEL is set in this environment";
    }

    auto& logger = Logger::instance();
    installProbeCounterOnce();
    EXPECT_EQ(logger.get_level(), LogLevel::INFO);

    const int info_before = g_info_probes.load();
    testing::internal::CaptureStdout();
    FE_LOG_INFO(std::string(kProbeTag) + " default-info");
    const std::string out = testing::internal::GetCapturedStdout();
    EXPECT_EQ(g_info_probes.load(), info_before + 1);
    EXPECT_NE(out.find(std::string(kProbeTag) + " default-info"), std::string::npos) << out;
}

TEST(LoggerEnvironment, WarningLevelFromEnvironmentSuppressesInfo)
{
    const char* env = envLogLevel();
    if (env == nullptr || parse_log_level(env) != LogLevel::WARNING) {
        GTEST_SKIP() << "runs as FE_Core_Logger_Environment_Tests with FE_LOG_LEVEL=WARNING";
    }

    auto& logger = Logger::instance();
    installProbeCounterOnce();
    ASSERT_EQ(logger.get_level(), LogLevel::WARNING);

    const int info_before = g_info_probes.load();
    const int warning_before = g_warning_probes.load();

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    FE_LOG_INFO(std::string(kProbeTag) + " suppressed-info");
    FE_LOG_WARNING(std::string(kProbeTag) + " emitted-warning");
    const std::string err = testing::internal::GetCapturedStderr();
    const std::string out = testing::internal::GetCapturedStdout();

    EXPECT_EQ(g_info_probes.load(), info_before);
    EXPECT_EQ(g_warning_probes.load(), warning_before + 1);
    EXPECT_EQ(out.find("suppressed-info"), std::string::npos) << out;
    EXPECT_EQ(err.find("suppressed-info"), std::string::npos) << err;
    EXPECT_NE(err.find("emitted-warning"), std::string::npos) << err;

    // An explicit set_level() made after start-up still takes precedence.
    logger.set_level(LogLevel::INFO);
    testing::internal::CaptureStdout();
    FE_LOG_INFO(std::string(kProbeTag) + " explicit-info");
    const std::string explicit_out = testing::internal::GetCapturedStdout();
    logger.set_level(LogLevel::WARNING);
    EXPECT_EQ(g_info_probes.load(), info_before + 1);
    EXPECT_NE(explicit_out.find("explicit-info"), std::string::npos) << explicit_out;
}

} // namespace test
} // namespace FE
} // namespace svmp
