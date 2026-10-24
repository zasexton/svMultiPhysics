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
#include "Core/LogRecord.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <limits>
#include <string>
#include <vector>

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

TEST(LogRecord, PrintsItsOwnerMessageAndFieldsInOrder)
{
    LogRecord record("Owner", "installed a term");
    record.integer("marker", 7)
        .quoted("level_set_field", "level_set")
        .text("form", "a_b")
        .fixed("mobility", 0.25)
        .general("calibration", 0.01)
        .general("cap", 2.5)
        .integer("count", std::size_t{3});
    EXPECT_EQ(record.str(),
              "Owner: installed a term marker=7 level_set_field='level_set' form=a_b mobility=0.250000 "
              "calibration=0.01 cap=2.5 count=3");
    EXPECT_EQ(record.level(), LogLevel::INFO);
    ASSERT_EQ(record.fields().size(), 7u);
    EXPECT_EQ(record.fields()[1].key, "level_set_field");
    EXPECT_TRUE(record.fields()[1].quoted);

    // Without a message the fields follow the owner; without fields the message is the line.
    EXPECT_EQ(LogRecord("Owner").text("diagnostic", "d").text("status", "declared").str(),
              "Owner: diagnostic=d status=declared");
    EXPECT_EQ(LogRecord("Owner", "free text", LogLevel::WARNING).str(), "Owner: free text");
}

TEST(LogRecord, IsLoggedAtItsLevel)
{
    testing::internal::CaptureStderr();
    logRecord(LogRecord("svmp-log-record-probe", "line", LogLevel::WARNING).integer("n", 1));
    logRecords(std::vector<LogRecord>{LogRecord("svmp-log-record-probe", "", LogLevel::WARNING).text("a", "b"),
                                      LogRecord("svmp-log-record-probe", "last", LogLevel::WARNING)});
    const std::string err = testing::internal::GetCapturedStderr();
    const auto first = err.find("[WARN] svmp-log-record-probe: line n=1\n");
    const auto second = err.find("[WARN] svmp-log-record-probe: a=b\n");
    const auto third = err.find("[WARN] svmp-log-record-probe: last\n");
    EXPECT_NE(first, std::string::npos) << err;
    EXPECT_NE(second, std::string::npos) << err;
    EXPECT_NE(third, std::string::npos) << err;
    EXPECT_LT(first, second);
    EXPECT_LT(second, third);
}

TEST(LogRecord, WritesRealsInTheShortestRoundTripText)
{
    EXPECT_EQ(LogRecord::shortestReal(0.26), "0.26");
    EXPECT_EQ(LogRecord::shortestReal(1.0e-05), "1e-05");
    EXPECT_EQ(LogRecord::shortestReal(2.0), "2");
    EXPECT_EQ(LogRecord::shortestReal(0.1 + 0.2), "0.30000000000000004");
    EXPECT_EQ(LogRecord::shortestReal(1.0e16), "1e+16");
    EXPECT_EQ(LogRecord::shortestReal(std::numeric_limits<Real>::quiet_NaN()), "nan");
    EXPECT_EQ(LogRecord::shortestReal(-std::numeric_limits<Real>::infinity()), "-inf");
    // The text reads back to the same double.
    for (const Real value : {0.50707106781186551, 5.50265214346803738e-04, 129.81019335983763, -3.2e-300}) {
        EXPECT_EQ(std::stod(LogRecord::shortestReal(value)), value);
    }
    EXPECT_EQ(LogRecord("Owner").real("x", 0.26).optionalReal("y", std::nullopt).boolean("z", true).str(),
              "Owner: x=0.26 y=none z=true");
    // value() writes by type.
    EXPECT_EQ(LogRecord("Owner")
                  .value("b", false)
                  .value("n", std::uint64_t{7})
                  .value("x", 0.5)
                  .value("w", std::string("word"))
                  .str(),
              "Owner: b=false n=7 x=0.5 w=word");
}

TEST(LogRecord, PutsTheDiagnosticFirstAndPrintsChildLines)
{
    LogRecord record("Owner", "accepted record");
    record.integer("accepted_step", 3).integer("interface_marker", 7).real("area", 0.5).diagnostic("record_history");
    for (const int wall : {1, 4}) {
        record.child("record_wall_history", {"accepted_step", "interface_marker"})
            .integer("wall_marker", wall)
            .real("wetted_area", 0.25 * wall);
    }
    EXPECT_EQ(record.str(), "Owner: accepted record diagnostic=record_history accepted_step=3 interface_marker=7 area=0.5");
    const std::vector<std::string> expected{
        "Owner: accepted record diagnostic=record_history accepted_step=3 interface_marker=7 area=0.5",
        "Owner: accepted record diagnostic=record_wall_history accepted_step=3 interface_marker=7 wall_marker=1 "
        "wetted_area=0.25",
        "Owner: accepted record diagnostic=record_wall_history accepted_step=3 interface_marker=7 wall_marker=4 "
        "wetted_area=1",
    };
    EXPECT_EQ(record.lines(), expected);
    // A second diagnostic replaces the first.
    EXPECT_EQ(LogRecord("Owner").diagnostic("a").text("k", "v").diagnostic("b").str(), "Owner: diagnostic=b k=v");

    // Printed in order at the record's level (WARNING: stderr); a root-only record prints on rank 0.
    LogRecord warning("svmp-log-record-child-probe", "", LogLevel::WARNING);
    warning.integer("step", 2).diagnostic("probe");
    warning.child("probe_point", {"step"}).integer("point", 0);
    testing::internal::CaptureStderr();
    logRecord(warning.rootOnly());
    const std::string err = testing::internal::GetCapturedStderr();
    const auto parent = err.find("[WARN] svmp-log-record-child-probe: diagnostic=probe step=2\n");
    const auto child = err.find("[WARN] svmp-log-record-child-probe: diagnostic=probe_point step=2 point=0\n");
    EXPECT_NE(parent, std::string::npos) << err;
    EXPECT_NE(child, std::string::npos) << err;
    EXPECT_LT(parent, child);
}

} // namespace test
} // namespace FE
} // namespace svmp
