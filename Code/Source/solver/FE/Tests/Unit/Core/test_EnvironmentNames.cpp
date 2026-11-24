/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Core/EnvironmentNames.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace svmp::FE::environment;

constexpr NameDeclaration kTestNames[] = {
    {"SVMP_TEST_ENVIRONMENT_TRACE", NameKind::Diagnostic, "test", ValueType::Flag, "off",
     RunStatus::Reference, "a trace"},
    {"SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES", NameKind::Performance, "test", ValueType::Integer,
     "64", RunStatus::Reference, "a cache size"},
    {"SVMP_TEST_ENVIRONMENT_FORCE_PATH", NameKind::TestHook, "test", ValueType::Flag, "off",
     RunStatus::NonReference, "a forced path"},
    {"SVMP_TEST_ENVIRONMENT_TOLERANCE", NameKind::NumericalOverride, "test", ValueType::Real,
     "the deck value", RunStatus::NonReference, "a tolerance"},
};

/// Sets or unsets one variable for the lifetime of the guard.
class ScopedVariable {
public:
    ScopedVariable(const char* name, const char* value) : name_(name)
    {
        if (const char* previous = std::getenv(name)) {
            previous_ = previous;
            had_previous_ = true;
        }
        if (value == nullptr) {
            ::unsetenv(name);
        } else {
            ::setenv(name, value, 1);
        }
    }
    ~ScopedVariable()
    {
        if (had_previous_) {
            ::setenv(name_, previous_.c_str(), 1);
        } else {
            ::unsetenv(name_);
        }
    }
    ScopedVariable(const ScopedVariable&) = delete;
    ScopedVariable& operator=(const ScopedVariable&) = delete;

private:
    const char* name_;
    std::string previous_{};
    bool had_previous_{false};
};

class EnvironmentNamesTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        for (const auto& declaration : kTestNames) {
            guards_.emplace_back(std::make_unique<ScopedVariable>(declaration.name, nullptr));
        }
    }
    void TearDown() override { guards_.clear(); }

    const EnvironmentNameRegistry registry_{kTestNames};

private:
    std::vector<std::unique_ptr<ScopedVariable>> guards_{};
};

TEST_F(EnvironmentNamesTest, ReadingAnUndeclaredNameIsAProgrammingError)
{
    EXPECT_EQ(registry_.find("SVMP_TEST_ENVIRONMENT_UNDECLARED"), nullptr);
    EXPECT_THROW((void)registry_.declaration("SVMP_TEST_ENVIRONMENT_UNDECLARED"), std::logic_error);
    EXPECT_THROW((void)registry_.value("SVMP_TEST_ENVIRONMENT_UNDECLARED"), std::logic_error);
    EXPECT_THROW((void)registry_.isSet("SVMP_TEST_ENVIRONMENT_UNDECLARED"), std::logic_error);
    EXPECT_THROW((void)registry_.enabled("SVMP_TEST_ENVIRONMENT_UNDECLARED"), std::logic_error);

    ScopedVariable set("SVMP_TEST_ENVIRONMENT_UNDECLARED", "1");
    EXPECT_THROW((void)value("SVMP_TEST_ENVIRONMENT_UNDECLARED"), std::logic_error);
    EXPECT_THROW((void)isSet("SVMP_TEST_ENVIRONMENT_UNDECLARED"), std::logic_error);
    EXPECT_THROW((void)enabled("SVMP_TEST_ENVIRONMENT_UNDECLARED"), std::logic_error);
}

TEST_F(EnvironmentNamesTest, RejectsDuplicateAndIncompleteDeclarations)
{
    constexpr NameDeclaration duplicate[] = {
        {"SVMP_TEST_ENVIRONMENT_TRACE", NameKind::Diagnostic, "a", ValueType::Flag, "off",
         RunStatus::Reference, "a trace"},
        {"SVMP_TEST_ENVIRONMENT_TRACE", NameKind::Diagnostic, "b", ValueType::Flag, "off",
         RunStatus::Reference, "a trace"},
    };
    EXPECT_THROW(EnvironmentNameRegistry{duplicate}, std::logic_error);

    constexpr NameDeclaration lower_case[] = {
        {"svmp_test_environment_trace", NameKind::Diagnostic, "a", ValueType::Flag, "off",
         RunStatus::Reference, "a trace"},
    };
    EXPECT_THROW(EnvironmentNameRegistry{lower_case}, std::logic_error);

    constexpr NameDeclaration no_owner[] = {
        {"SVMP_TEST_ENVIRONMENT_TRACE", NameKind::Diagnostic, "", ValueType::Flag, "off",
         RunStatus::Reference, "a trace"},
    };
    EXPECT_THROW(EnvironmentNameRegistry{no_owner}, std::logic_error);
}

TEST_F(EnvironmentNamesTest, ReadsTheProcessEnvironmentOnEveryCall)
{
    EXPECT_EQ(registry_.value("SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES"), nullptr);
    EXPECT_FALSE(registry_.isSet("SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES"));
    {
        ScopedVariable set("SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES", "128");
        ASSERT_NE(registry_.value("SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES"), nullptr);
        EXPECT_STREQ(registry_.value("SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES"), "128");
        EXPECT_TRUE(registry_.isSet("SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES"));
    }
    {
        ScopedVariable set("SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES", "");
        EXPECT_TRUE(registry_.isSet("SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES"));
        EXPECT_STREQ(registry_.value("SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES"), "");
    }
    EXPECT_FALSE(registry_.isSet("SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES"));
}

TEST_F(EnvironmentNamesTest, EnabledFollowsTheSharedFlagRule)
{
    const char* name = "SVMP_TEST_ENVIRONMENT_TRACE";
    EXPECT_FALSE(registry_.enabled(name));
    const std::map<std::string, bool> expected{
        {"", false},   {"0", false},    {"false", false}, {"FALSE", false}, {"off", false},
        {"Off", false}, {"no", false},  {"NO", false},    {"1", true},      {"true", true},
        {"on", true},  {"yes", true},   {"2", true},      {" 0", true},     {"00", true},
    };
    for (const auto& [text, enabled] : expected) {
        ScopedVariable set(name, text.c_str());
        EXPECT_EQ(registry_.enabled(name), enabled) << "value '" << text << "'";
    }
    EXPECT_THROW((void)registry_.enabled("SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES"), std::logic_error);
}

TEST_F(EnvironmentNamesTest, ConfigurationRecordIsEmptyUnlessANameThatIsNotADiagnosticIsSet)
{
    EXPECT_TRUE(registry_.setNames().empty());
    EXPECT_EQ(registry_.configurationRecord(), "");

    ScopedVariable trace("SVMP_TEST_ENVIRONMENT_TRACE", "1");
    ASSERT_EQ(registry_.setNames().size(), 1u);
    EXPECT_EQ(registry_.configurationRecord(), "");

    {
        ScopedVariable cache("SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES", "32");
        EXPECT_EQ(registry_.configurationRecord(),
                  "\"environment_names\":{\"reference_run\":true,\"set\":["
                  "{\"name\":\"SVMP_TEST_ENVIRONMENT_CACHE_ENTRIES\",\"value\":\"32\","
                  "\"kind\":\"performance\",\"value_type\":\"integer\",\"owner\":\"test\","
                  "\"marks_non_reference\":false}]}");
    }

    ScopedVariable force("SVMP_TEST_ENVIRONMENT_FORCE_PATH", "on");
    ScopedVariable tolerance("SVMP_TEST_ENVIRONMENT_TOLERANCE", "1e-3\"\n");
    EXPECT_EQ(registry_.configurationRecord(),
              "\"environment_names\":{\"reference_run\":false,\"set\":["
              "{\"name\":\"SVMP_TEST_ENVIRONMENT_FORCE_PATH\",\"value\":\"on\","
              "\"kind\":\"test_hook\",\"value_type\":\"flag\",\"owner\":\"test\","
              "\"marks_non_reference\":true},"
              "{\"name\":\"SVMP_TEST_ENVIRONMENT_TOLERANCE\",\"value\":\"1e-3\\\"\\n\","
              "\"kind\":\"numerical_override\",\"value_type\":\"real\",\"owner\":\"test\","
              "\"marks_non_reference\":true}]}");
}

// ---------------------------------------------------------------------------
// The solver catalogue against the solver sources
// ---------------------------------------------------------------------------

struct SourceFile {
    std::filesystem::path path{};
    std::string text{};
};

/// Every production source file (tests excluded) of FE, Physics and
/// Application.
std::vector<SourceFile> productionSources()
{
    std::vector<SourceFile> files;
    const std::filesystem::path root(SVMP_SOLVER_SOURCE_DIR);
    for (const char* module : {"FE", "Physics", "Application"}) {
        const auto directory = root / module;
        if (!std::filesystem::is_directory(directory)) {
            continue;
        }
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const auto& path = entry.path();
            const auto extension = path.extension().string();
            if (extension != ".cpp" && extension != ".h" && extension != ".hpp") {
                continue;
            }
            bool in_tests = false;
            for (const auto& part : path.lexically_relative(root)) {
                in_tests = in_tests || part == "Tests";
            }
            if (in_tests) {
                continue;
            }
            std::ifstream input(path);
            files.push_back(SourceFile{
                .path = path,
                .text = std::string(std::istreambuf_iterator<char>(input),
                                    std::istreambuf_iterator<char>())});
        }
    }
    return files;
}

/// The string literal of upper-case letters, digits and underscores that
/// starts at @p position after optional white space, or "" when there is none.
std::string literalAt(const std::string& text, std::size_t position)
{
    while (position < text.size() &&
           (text[position] == ' ' || text[position] == '\n' || text[position] == '\t' ||
            text[position] == '\r')) {
        ++position;
    }
    if (position >= text.size() || text[position] != '"') {
        return {};
    }
    const auto begin = position + 1u;
    auto end = begin;
    while (end < text.size() &&
           ((text[end] >= 'A' && text[end] <= 'Z') || (text[end] >= '0' && text[end] <= '9') ||
            text[end] == '_')) {
        ++end;
    }
    if (end == begin || end >= text.size() || text[end] != '"') {
        return {};
    }
    return text.substr(begin, end - begin);
}

/// True when @p text reads an environment name through the registry.
bool readsRegistry(const std::string& text)
{
    for (const char* call : {"environment::value(", "environment::isSet(", "environment::enabled("}) {
        if (text.find(call) != std::string::npos) {
            return true;
        }
    }
    return false;
}

/// The quoted environment-name spellings (SVMP_... and FE_LOG_...) of @p text.
std::set<std::string> quotedNames(const std::string& text)
{
    std::set<std::string> names;
    for (auto at = text.find('"'); at != std::string::npos; at = text.find('"', at + 1u)) {
        auto literal = literalAt(text, at);
        if (literal.rfind("SVMP_", 0) == 0 || literal.rfind("FE_LOG_", 0) == 0) {
            names.insert(std::move(literal));
        }
    }
    return names;
}

/// The names that @p text reads with std::getenv and a literal argument.
std::set<std::string> directlyReadNames(const std::string& text)
{
    std::set<std::string> names;
    const std::string call = "getenv(";
    for (auto at = text.find(call); at != std::string::npos; at = text.find(call, at + call.size())) {
        if (auto name = literalAt(text, at + call.size()); !name.empty()) {
            names.insert(std::move(name));
        }
    }
    return names;
}

// A file that reads through the registry spells a name only if the name is
// declared or the file still reads it directly (a family not converted yet):
// a misspelled name passed to a local reading helper fails here.
TEST(EnvironmentNameCatalogue, EveryNameSpelledWhereTheRegistryIsReadIsDeclared)
{
    const auto files = productionSources();
    if (files.empty()) {
        GTEST_SKIP() << "solver sources not found at " << SVMP_SOLVER_SOURCE_DIR;
    }
    const auto& registry = EnvironmentNameRegistry::solver();
    for (const auto& file : files) {
        if (!readsRegistry(file.text) || file.path.filename() == "EnvironmentNames.cpp") {
            continue;
        }
        const auto direct = directlyReadNames(file.text);
        for (const auto& name : quotedNames(file.text)) {
            EXPECT_TRUE(registry.find(name) != nullptr || direct.count(name) != 0u)
                << file.path.string() << " spells the undeclared name " << name;
        }
    }
}

TEST(EnvironmentNameCatalogue, DeclaredNamesAreReadOnlyThroughTheRegistry)
{
    const auto files = productionSources();
    if (files.empty()) {
        GTEST_SKIP() << "solver sources not found at " << SVMP_SOLVER_SOURCE_DIR;
    }
    const auto& registry = EnvironmentNameRegistry::solver();
    const std::string call = "getenv(";
    for (const auto& file : files) {
        if (file.path.filename() == "EnvironmentNames.cpp") {
            continue;
        }
        const bool reads_registry = readsRegistry(file.text);
        for (auto at = file.text.find(call); at != std::string::npos;
             at = file.text.find(call, at + call.size())) {
            const auto name = literalAt(file.text, at + call.size());
            if (name.empty()) {
                // A file that reads through the registry reads no name it
                // does not spell.
                EXPECT_FALSE(reads_registry)
                    << file.path.string() << " calls getenv with a computed name";
                continue;
            }
            EXPECT_EQ(registry.find(name), nullptr)
                << file.path.string() << " reads the declared name " << name
                << " without the registry";
        }
    }
}

// Outside the registry the solver reads only system names.
TEST(EnvironmentNameCatalogue, OnlySystemNamesAreReadWithoutTheRegistry)
{
    const auto files = productionSources();
    if (files.empty()) {
        GTEST_SKIP() << "solver sources not found at " << SVMP_SOLVER_SOURCE_DIR;
    }
    const std::set<std::string> system_names{"COMPUTERNAME", "HOME", "OMP_NUM_THREADS"};
    const std::string call = "getenv(";
    for (const auto& file : files) {
        if (file.path.filename() == "EnvironmentNames.cpp") {
            continue;
        }
        for (auto at = file.text.find(call); at != std::string::npos;
             at = file.text.find(call, at + call.size())) {
            const auto name = literalAt(file.text, at + call.size());
            if (name.empty()) {
                // The MPI launchers' rank variables, tried in turn.
                EXPECT_EQ(file.path.filename(), "PhysicsModule.cpp")
                    << file.path.string() << " calls getenv with a computed name";
                continue;
            }
            EXPECT_TRUE(system_names.count(name) != 0u)
                << file.path.string() << " reads " << name << " without the registry";
        }
    }
}

TEST(EnvironmentNameCatalogue, EveryDeclaredNameHasAReader)
{
    const auto files = productionSources();
    if (files.empty()) {
        GTEST_SKIP() << "solver sources not found at " << SVMP_SOLVER_SOURCE_DIR;
    }
    std::set<std::string> spelled;
    for (const auto& file : files) {
        if (file.path.filename() != "EnvironmentNames.cpp") {
            spelled.merge(quotedNames(file.text));
        }
    }
    for (const auto* declaration : EnvironmentNameRegistry::solver().declarations()) {
        EXPECT_TRUE(spelled.count(declaration->name) != 0u)
            << declaration->name << " is declared but no solver source names it";
    }
}

} // namespace
