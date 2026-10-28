/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Core/OptionSchema.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <locale>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace svmp::FE::options;
using svmp::FE::Real;

enum class Side { Negative, Positive };
enum class Enforcement { None, Penalty, MeshNitsche };

struct Guards {
    int path_length{8};
    Real coefficient{16.0};
};

struct Example {
    bool enabled{false};
    bool relaxed_flag{false};
    int layers{0};
    Real tension{0.0};
    Real tolerance{1.0e-10};
    Side side{Side::Negative};
    std::string field{"level_set"};
    std::array<Real, 3> normal{0.0, 0.0, 0.0};
    std::vector<int> markers{};
    Guards guards{};
    bool stabilization{false};
    Real penalty{1.0};
};

EnumCodec<Side> sideCodec()
{
    return EnumCodec<Side>{{
        EnumToken<Side>{Side::Negative, "LevelSetNegative", {"negative", "phiNegative"}},
        EnumToken<Side>{Side::Positive, "LevelSetPositive", {"positive"}},
    }};
}

OptionSchema<Example> exampleSchema()
{
    OptionSchema<Example> schema("Test_element");
    schema
        .key("Enabled", &Example::enabled, BoolCodec{},
             KeySettings{.spellings = {{"Enabled", false}, {"Enable_it", false}},
                         .doc = "switch"})
        .key("Relaxed_flag", &Example::relaxed_flag, BoolCodec{BoolLexicon::Relaxed})
        .key("Layers", &Example::layers, IntCodec{.minimum = 0, .maximum = 16},
             KeySettings{.spellings = {{"Layers", false}, {"Layer_count", false}},
                         .duplicates = DuplicatePolicy::Reject})
        .key("Surface_tension", &Example::tension,
             RealCodec{.minimum = 0.0},
             KeySettings{.spellings = {{"Old_tension", true}, {"Surface_tension", false}}})
        .key("Tolerance", &Example::tolerance, RealCodec{.require_finite = false})
        .key("Active_domain", &Example::side, sideCodec())
        .key("Level_set_field_name", &Example::field, StringCodec{},
             KeySettings{.empty = EmptyPolicy::Error})
        .key("Wall_normal", &Example::normal, Vector3Codec{},
             KeySettings{.spellings = {{"Wall_normal", false}, {"Contact_wall_normal", false}}})
        .key("Wall_markers", &Example::markers, ListCodec<IntCodec>{})
        .key("Root_path_length", nestedMember(&Example::guards, &Guards::path_length),
             IntCodec{.minimum = 1})
        .key("Guard_coefficient",
             [](Example& e) -> Real& { return e.guards.coefficient; },
             RealCodec{.minimum = 0.0, .open_minimum = true},
             KeySettings{.invalid_message = "must be finite and positive",
                         .artifact_key = "guard_coefficient"})
        .key("Stabilization", &Example::stabilization, BoolCodec{})
        .key("Penalty", &Example::penalty, RealCodec{.minimum = 0.0, .open_minimum = true})
        .implies("Penalty", "Stabilization", "true")
        .retired({"Velocity_extension"}, "retired (decision D42)",
                 RetiredAcceptance{.value = "false"});
    return schema;
}

// Index of a key in the declaration order of exampleSchema().
constexpr std::size_t kTensionIndex = 3;

OptionInput input(std::vector<std::pair<std::string, std::string>> entries)
{
    OptionInput out;
    for (auto& [spelling, text] : entries) {
        out.add(spelling, text);
    }
    return out;
}

std::string errorText(const std::function<void()>& action)
{
    try {
        action();
    } catch (const OptionError& error) {
        return error.what();
    }
    return {};
}

TEST(OptionSchema, DefaultsAreKeptAndRecordedWhenNothingIsGiven)
{
    const auto schema = exampleSchema();
    const auto resolved = schema.resolve(OptionInput{});
    EXPECT_FALSE(resolved.value.enabled);
    EXPECT_EQ(resolved.value.layers, 0);
    EXPECT_EQ(resolved.value.field, "level_set");
    EXPECT_EQ(resolved.value.guards.coefficient, 16.0);
    ASSERT_EQ(resolved.provenance.size(), schema.size());
    for (const auto& item : resolved.provenance) {
        EXPECT_EQ(item.source, OptionSource::Default) << item.key;
    }
    EXPECT_TRUE(resolved.deprecations.empty());
}

TEST(OptionSchema, ResolveStartsFromTheGivenDefaults)
{
    const auto schema = exampleSchema();
    Example defaults;
    defaults.layers = 5;
    defaults.side = Side::Positive;
    const auto resolved = schema.resolve(input({{"Layers", "2"}}), defaults);
    EXPECT_EQ(resolved.value.layers, 2);
    EXPECT_EQ(resolved.value.side, Side::Positive);
    EXPECT_EQ(resolved.find("Active_domain")->source, OptionSource::Default);
}

TEST(OptionSchema, FirstDefinedSpellingInDeclaredOrderWins)
{
    const auto schema = exampleSchema();
    // Input order is the reverse of the declared order.
    const auto resolved =
        schema.resolve(input({{"Enable_it", "false"}, {"Enabled", "true"}}));
    EXPECT_TRUE(resolved.value.enabled);
    ASSERT_NE(resolved.find("Enabled"), nullptr);
    EXPECT_EQ(resolved.find("Enabled")->spelling, "Enabled");
    EXPECT_TRUE(resolved.isExplicit("Enabled"));
    // A later spelling is not parsed once an earlier one is chosen.
    EXPECT_TRUE(schema.resolve(input({{"Enabled", "true"}, {"Enable_it", "garbage"}})).value.enabled);
}

TEST(OptionSchema, RejectPolicyRefusesTwoSuppliedSpellingsOfOneKey)
{
    const auto schema = exampleSchema();
    const auto message = errorText([&] { (void)schema.resolve(input({{"Layers", "2"}, {"Layer_count", "3"}})); });
    EXPECT_NE(message.find("'Layers' and 'Layer_count'"), std::string::npos) << message;
    // An empty spelling still counts as supplied, as in the hand-written check.
    EXPECT_THROW((void)schema.resolve(input({{"Layers", " "}, {"Layer_count", "3"}})), OptionError);
    EXPECT_EQ(schema.resolve(input({{"Layer_count", "3"}})).value.layers, 3);
}

TEST(OptionSchema, DeprecatedSpellingIsAppliedAndReported)
{
    const auto schema = exampleSchema();
    const auto resolved = schema.resolve(input({{"Old_tension", "0.07"}}));
    EXPECT_DOUBLE_EQ(resolved.value.tension, 0.07);
    ASSERT_NE(resolved.find("Surface_tension"), nullptr);
    EXPECT_EQ(resolved.find("Surface_tension")->source, OptionSource::DeprecatedSpelling);
    EXPECT_TRUE(resolved.isExplicit("Surface_tension"));
    ASSERT_EQ(resolved.deprecations.size(), 1u);
    EXPECT_EQ(resolved.deprecations.front().spelling, "Old_tension");
    EXPECT_EQ(resolved.deprecations.front().replacement, "Surface_tension");
    // The deprecated spelling is first in the declared order, as in the
    // hand-written parser it replaces.
    EXPECT_DOUBLE_EQ(
        schema.resolve(input({{"Surface_tension", "1"}, {"Old_tension", "2"}})).value.tension, 2.0);
}

TEST(OptionSchema, BooleanLexiconsAreStrictOrRelaxedPerKey)
{
    const auto schema = exampleSchema();
    for (const auto* text : {"true", "TRUE", "1", "yes", "On", " true "}) {
        EXPECT_TRUE(schema.resolve(input({{"Enabled", text}})).value.enabled) << text;
    }
    for (const auto* text : {"false", "0", "No", "off"}) {
        EXPECT_FALSE(schema.resolve(input({{"Enabled", text}})).value.enabled) << text;
    }
    EXPECT_THROW((void)schema.resolve(input({{"Enabled", "maybe"}})), OptionError);
    EXPECT_FALSE(schema.resolve(input({{"Relaxed_flag", "maybe"}})).value.relaxed_flag);
    EXPECT_TRUE(schema.resolve(input({{"Relaxed_flag", " yes "}})).value.relaxed_flag);
}

TEST(OptionSchema, NumbersMustConsumeTheWholeTextAndRespectBounds)
{
    const auto schema = exampleSchema();
    EXPECT_EQ(schema.resolve(input({{"Layers", "+12"}})).value.layers, 12);
    EXPECT_EQ(schema.resolve(input({{"Layers", " 7 "}})).value.layers, 7);
    for (const auto* text : {"2tail", "1.5", "0x10", "+-5", "+", "-", "17", "-1"}) {
        EXPECT_THROW((void)schema.resolve(input({{"Layers", text}})), OptionError) << text;
    }
    const auto overflow = errorText([&] { (void)schema.resolve(input({{"Layers", "99999999999999999999"}})); });
    EXPECT_NE(overflow.find("out of range"), std::string::npos) << overflow;
    EXPECT_THROW((void)schema.resolve(input({{"Surface_tension", "1.25tail"}})), OptionError);
    EXPECT_THROW((void)schema.resolve(input({{"Surface_tension", "-0.5"}})), OptionError);
    EXPECT_THROW((void)schema.resolve(input({{"Surface_tension", "inf"}})), OptionError);
    EXPECT_THROW((void)schema.resolve(input({{"Surface_tension", "1e999"}})), OptionError);
    EXPECT_DOUBLE_EQ(schema.resolve(input({{"Surface_tension", "7.2e-2"}})).value.tension, 0.072);
    EXPECT_DOUBLE_EQ(schema.resolve(input({{"Surface_tension", "+0.5"}})).value.tension, 0.5);
    // A key that admits non-finite values keeps them.
    EXPECT_TRUE(std::isnan(schema.resolve(input({{"Tolerance", "nan"}})).value.tolerance));
    // Open lower bound: zero is outside (0, inf].
    EXPECT_THROW((void)schema.resolve(input({{"Penalty", "0"}})), OptionError);
    EXPECT_DOUBLE_EQ(schema.resolve(input({{"Penalty", "1e-300"}})).value.penalty, 1e-300);
}

TEST(OptionSchema, InvalidMessageReplacesTheCodecReason)
{
    const auto schema = exampleSchema();
    const auto message = errorText([&] { (void)schema.resolve(input({{"Guard_coefficient", "0"}})); });
    EXPECT_EQ(message, "[Test_element] Guard_coefficient: must be finite and positive");
    const auto codec_message = errorText([&] { (void)schema.resolve(input({{"Penalty", "0"}})); });
    EXPECT_NE(codec_message.find("outside (0, inf]"), std::string::npos) << codec_message;
}

TEST(OptionSchema, EnumerationTokensIgnoreCaseAndSeparators)
{
    const auto schema = exampleSchema();
    for (const auto* text : {"LevelSetPositive", "level_set_positive", "POSITIVE", "Level Set-Positive"}) {
        EXPECT_EQ(schema.resolve(input({{"Active_domain", text}})).value.side, Side::Positive) << text;
    }
    EXPECT_EQ(schema.resolve(input({{"Active_domain", "phiNegative"}})).value.side, Side::Negative);
    const auto message = errorText([&] { (void)schema.resolve(input({{"Active_domain", "None"}})); });
    EXPECT_NE(message.find("LevelSetNegative | LevelSetPositive"), std::string::npos) << message;
    const auto serialized = schema.serialize(schema.resolve(input({{"Active_domain", "positive"}})).value);
    const auto it = std::find_if(serialized.begin(), serialized.end(),
                                 [](const auto& item) { return item.first == "Active_domain"; });
    ASSERT_NE(it, serialized.end());
    EXPECT_EQ(it->second, "LevelSetPositive");
}

TEST(OptionSchema, VectorsListsAndNestedMembers)
{
    const auto schema = exampleSchema();
    const auto resolved = schema.resolve(input({{"Wall_normal", "0, -1 0"},
                                                {"Wall_markers", "80; 81;"},
                                                {"Root_path_length", "6"},
                                                {"Guard_coefficient", "4.5"}}));
    EXPECT_EQ(resolved.value.normal, (std::array<Real, 3>{0.0, -1.0, 0.0}));
    EXPECT_EQ(resolved.value.markers, (std::vector<int>{80, 81}));
    EXPECT_EQ(resolved.value.guards.path_length, 6);
    EXPECT_EQ(resolved.value.guards.coefficient, 4.5);
    EXPECT_THROW((void)schema.resolve(input({{"Wall_normal", "1 0"}})), OptionError);
    EXPECT_THROW((void)schema.resolve(input({{"Wall_normal", "1 2 3x"}})), OptionError);
    EXPECT_THROW((void)schema.resolve(input({{"Wall_normal", "1 2 3 4"}})), OptionError);
    EXPECT_THROW((void)schema.resolve(input({{"Wall_markers", " ; "}})), OptionError);
    EXPECT_THROW((void)schema.resolve(input({{"Wall_markers", "80; x"}})), OptionError);
    EXPECT_THROW((void)schema.resolve(input({{"Root_path_length", "0"}})), OptionError);
    const auto serialized = schema.serialize(resolved.value);
    EXPECT_EQ(serialized[7].second, "0 -1 0");
    EXPECT_EQ(serialized[8].second, "80; 81");
}

TEST(OptionSchema, EmptyValuesFollowTheCodecDefaultOrTheKeySetting)
{
    const auto schema = exampleSchema();
    // Vectors (and text, enumerations, lists): an empty value is absent and
    // the next spelling is used.
    EXPECT_EQ(schema.resolve(input({{"Wall_normal", "   "}, {"Contact_wall_normal", "1 0 0"}})).value.normal,
              (std::array<Real, 3>{1.0, 0.0, 0.0}));
    EXPECT_EQ(schema.resolve(input({{"Wall_normal", ""}})).find("Wall_normal")->source,
              OptionSource::Default);
    EXPECT_EQ(schema.resolve(input({{"Active_domain", " "}})).find("Active_domain")->source,
              OptionSource::Default);
    // Booleans and numbers: the empty text is parsed.  A strict boolean and
    // a number reject it; a relaxed boolean reads it as an explicit false.
    EXPECT_THROW((void)schema.resolve(input({{"Enabled", " "}})), OptionError);
    EXPECT_THROW((void)schema.resolve(input({{"Surface_tension", ""}})), OptionError);
    EXPECT_THROW((void)schema.resolve(input({{"Root_path_length", ""}})), OptionError);
    Example defaults;
    defaults.relaxed_flag = true;
    const auto relaxed = schema.resolve(input({{"Relaxed_flag", ""}}), defaults);
    EXPECT_FALSE(relaxed.value.relaxed_flag);
    EXPECT_TRUE(relaxed.isExplicit("Relaxed_flag"));
    // A key may make an empty value an error.
    EXPECT_THROW((void)schema.resolve(input({{"Level_set_field_name", "  "}})), OptionError);
    EXPECT_EQ(schema.resolve(input({{"Level_set_field_name", " phi "}})).value.field, "phi");
}

TEST(OptionSchema, ImpliedValuesYieldToExplicitOnes)
{
    const auto schema = exampleSchema();
    const auto implied = schema.resolve(input({{"Penalty", "2.0"}}));
    EXPECT_TRUE(implied.value.stabilization);
    EXPECT_EQ(implied.find("Stabilization")->source, OptionSource::Implied);
    EXPECT_EQ(implied.find("Stabilization")->origin, "Penalty");
    EXPECT_FALSE(implied.isExplicit("Stabilization"));
    const auto explicit_off =
        schema.resolve(input({{"Penalty", "2.0"}, {"Stabilization", "false"}}));
    EXPECT_FALSE(explicit_off.value.stabilization);
    EXPECT_EQ(explicit_off.find("Stabilization")->source, OptionSource::Explicit);
    // Without the trigger nothing is implied.
    EXPECT_EQ(schema.resolve(OptionInput{}).find("Stabilization")->source, OptionSource::Default);
}

struct Boundary {
    Side side{Side::Negative};
    Enforcement enforcement{Enforcement::None};
    Real penalty{0.0};
    int archived_order{0};
};

OptionSchema<Boundary> boundarySchema()
{
    OptionSchema<Boundary> schema("Boundary");
    schema
        .key("Active_domain", &Boundary::side, sideCodec())
        .key("Enforcement", &Boundary::enforcement,
             EnumCodec<Enforcement>{{
                 EnumToken<Enforcement>{Enforcement::None, "None", {}},
                 EnumToken<Enforcement>{Enforcement::Penalty, "Penalty", {}},
                 EnumToken<Enforcement>{Enforcement::MeshNitsche, "MeshNitsche", {}},
             }})
        .key("Penalty", &Boundary::penalty, RealCodec{.minimum = 0.0})
        // A positive active domain defaults the enforcement to MeshNitsche,
        // whether or not the domain was given, except under the legacy mode.
        .implies("Active_domain", "Enforcement", "MeshNitsche",
                 ImplicationSettings{.when_trigger_is = "positive",
                                     .trigger_must_be_explicit = false,
                                     .modes = {.except_in = "legacy"}})
        // Under the legacy mode a penalty promotes the enforcement.
        .implies("Penalty", "Enforcement", "Penalty",
                 ImplicationSettings{.modes = {.only_in = "legacy"}})
        .rule("penalty needs Penalty enforcement",
              [](const ResolvedOptions<Boundary>& r) {
                  return r.isExplicit("Penalty") && r.value.enforcement != Enforcement::Penalty
                             ? std::string("set Enforcement=Penalty")
                             : std::string{};
              },
              ModeCondition{.except_in = "legacy"})
        .retired({"Archived_order"}, "retired; only legacy input may keep an ignored value",
                 RetiredAcceptance{.any_value_in_mode = "legacy"});
    return schema;
}

TEST(OptionSchema, ConditionalImplicationsFollowTheTriggerValue)
{
    const auto schema = boundarySchema();
    Boundary positive_default;
    positive_default.side = Side::Positive;
    const auto from_default = schema.resolve(OptionInput{}, positive_default);
    EXPECT_EQ(from_default.value.enforcement, Enforcement::MeshNitsche);
    EXPECT_EQ(from_default.find("Enforcement")->source, OptionSource::Implied);
    EXPECT_EQ(from_default.find("Enforcement")->origin, "Active_domain");
    EXPECT_EQ(schema.resolve(input({{"Active_domain", "LevelSetPositive"}})).value.enforcement,
              Enforcement::MeshNitsche);
    EXPECT_EQ(schema.resolve(input({{"Active_domain", "negative"}})).value.enforcement, Enforcement::None);
    EXPECT_EQ(schema.resolve(input({{"Active_domain", "positive"}, {"Enforcement", "none"}})).value.enforcement,
              Enforcement::None);
    // Declaration mistakes in the implied or trigger text are found at once.
    auto broken = boundarySchema();
    EXPECT_THROW(broken.implies("Penalty", "Enforcement", "Sometimes"), std::logic_error);
    EXPECT_THROW(broken.implies("Active_domain", "Penalty", "1",
                                ImplicationSettings{.when_trigger_is = "upward"}),
                 std::logic_error);
}

TEST(OptionSchema, ModesEnableAndWaiveImplicationsRulesAndRetiredKeys)
{
    const auto schema = boundarySchema();
    const ResolveContext legacy{.modes = {"legacy"}};
    const ResolveContext current{};

    // The rule applies outside the legacy mode only.
    EXPECT_THROW((void)schema.resolve(input({{"Penalty", "3"}}), Boundary{}, current), OptionError);
    const auto promoted = schema.resolve(input({{"Penalty", "3"}}), Boundary{}, legacy);
    EXPECT_EQ(promoted.value.enforcement, Enforcement::Penalty);
    EXPECT_EQ(promoted.find("Enforcement")->origin, "Penalty");
    EXPECT_NO_THROW((void)schema.resolve(input({{"Penalty", "3"}, {"Enforcement", "Penalty"}}), Boundary{},
                                         current));

    // The legacy mode waives the MeshNitsche default.
    Boundary positive_default;
    positive_default.side = Side::Positive;
    EXPECT_EQ(schema.resolve(OptionInput{}, positive_default, legacy).value.enforcement, Enforcement::None);

    // A retired key is accepted (ignored and reported) only in the legacy mode.
    EXPECT_THROW((void)schema.resolve(input({{"Archived_order", "2"}}), Boundary{}, current), OptionError);
    const auto archived = schema.resolve(input({{"Archived_order", "2"}}), Boundary{}, legacy);
    EXPECT_EQ(archived.value.archived_order, 0);
    ASSERT_EQ(archived.deprecations.size(), 1u);
    EXPECT_EQ(archived.deprecations.front().spelling, "Archived_order");
    EXPECT_TRUE(archived.deprecations.front().replacement.empty());
}

TEST(OptionSchema, ConflictsAndRulesFailWithTheirReason)
{
    auto schema = exampleSchema();
    schema.conflicts("Layers", "Root_path_length", "use one of them");
    schema.rule("tension needs the switch", [](const ResolvedOptions<Example>& r) {
        return r.value.tension > 0.0 && !r.value.enabled ? std::string("set Enabled") : std::string{};
    });
    const auto conflict =
        errorText([&] { (void)schema.resolve(input({{"Layers", "1"}, {"Root_path_length", "2"}})); });
    EXPECT_NE(conflict.find("conflicts with Root_path_length: use one of them"), std::string::npos) << conflict;
    EXPECT_NO_THROW((void)schema.resolve(input({{"Layers", "1"}})));
    const auto rule = errorText([&] { (void)schema.resolve(input({{"Surface_tension", "1"}})); });
    EXPECT_NE(rule.find("set Enabled"), std::string::npos) << rule;
    EXPECT_NE(rule.find("[Test_element]"), std::string::npos) << rule;
    EXPECT_NO_THROW((void)schema.resolve(input({{"Surface_tension", "1"}, {"Enabled", "true"}})));
}

TEST(OptionSchema, RetiredKeysFailUnlessTheyRestateTheAcceptedValue)
{
    const auto schema = exampleSchema();
    const auto message = errorText([&] { (void)schema.resolve(input({{"Velocity_extension", "true"}})); });
    EXPECT_EQ(message, "[Test_element] Velocity_extension: retired (decision D42)");
    const auto accepted = schema.resolve(input({{"Velocity_extension", " FALSE "}}));
    ASSERT_EQ(accepted.deprecations.size(), 1u);
    EXPECT_EQ(accepted.deprecations.front().spelling, "Velocity_extension");
    EXPECT_EQ(accepted.deprecations.front().message, "retired (decision D42)");
    EXPECT_FALSE(schema.acceptsSpelling("Velocity_extension"));
    EXPECT_TRUE(schema.isRetiredSpelling("Velocity_extension"));
    EXPECT_EQ(schema.retiredSpellings(), (std::vector<std::string>{"Velocity_extension"}));
}

TEST(OptionSchema, UnknownSpellingsAreIgnoredOrRejected)
{
    const auto schema = exampleSchema();
    const ResolveContext fail_closed{.unknown = UnknownKeyPolicy::Reject};
    const auto in = input({{"Enabled", "true"}, {"Some_other_key", "1"}});
    EXPECT_NO_THROW((void)schema.resolve(in));
    const auto message = errorText([&] { (void)schema.resolve(in, Example{}, fail_closed); });
    EXPECT_EQ(message, "[Test_element] Some_other_key: unknown key");
    // Deprecated and retired spellings are known spellings.
    EXPECT_NO_THROW((void)schema.resolve(input({{"Old_tension", "1"}, {"Velocity_extension", "false"}}),
                                         Example{}, fail_closed));
}

TEST(OptionSchema, DerivedValuesAreRecordedAfterResolution)
{
    const auto schema = exampleSchema();
    auto resolved = schema.resolve(input({{"Layers", "2"}}));
    resolved.value.markers = {7};
    EXPECT_TRUE(resolved.derive("Wall_markers", "mesh face 'wall'"));
    EXPECT_FALSE(resolved.derive("Not_a_key", "nothing"));
    EXPECT_EQ(resolved.find("Wall_markers")->source, OptionSource::Derived);
    const auto json = schema.toJson(resolved);
    EXPECT_NE(json.find("\"Wall_markers\":{\"value\":\"7\",\"source\":\"derived\",\"origin\":\"mesh face 'wall'\"}"),
              std::string::npos)
        << json;
}

TEST(OptionSchema, SerializationAndDocumentationFollowDeclarationOrder)
{
    const auto schema = exampleSchema();
    const auto resolved = schema.resolve(input({{"Enabled", "yes"}, {"Old_tension", "0.5"}, {"Penalty", "2"}}));
    const auto serialized = schema.serialize(resolved.value);
    ASSERT_EQ(serialized.size(), schema.size());
    EXPECT_EQ(serialized.front(), (std::pair<std::string, std::string>{"Enabled", "true"}));
    EXPECT_EQ(serialized[kTensionIndex], (std::pair<std::string, std::string>{"Surface_tension", "0.5"}));
    const auto json = schema.toJson(resolved);
    EXPECT_EQ(json.rfind("{\"Enabled\":{\"value\":\"true\",\"source\":\"explicit\",\"spelling\":\"Enabled\"}", 0),
              0u)
        << json;
    EXPECT_NE(json.find("\"Surface_tension\":{\"value\":\"0.5\",\"source\":\"deprecated_spelling\","
                        "\"spelling\":\"Old_tension\"}"),
              std::string::npos)
        << json;
    EXPECT_NE(json.find("\"Stabilization\":{\"value\":\"true\",\"source\":\"implied\",\"origin\":\"Penalty\"}"),
              std::string::npos)
        << json;
    // The artifact key may differ from the canonical spelling.
    EXPECT_NE(json.find("\"guard_coefficient\":{\"value\":\"16\",\"source\":\"default\"}"), std::string::npos)
        << json;
    EXPECT_EQ(json.back(), '}');
    const auto docs = schema.documentation();
    ASSERT_EQ(docs.size(), schema.size());
    EXPECT_EQ(docs.front().key, "Enabled");
    EXPECT_EQ(docs.front().spellings, (std::vector<std::string>{"Enabled", "Enable_it"}));
    EXPECT_EQ(docs.front().default_value, "false");
    EXPECT_EQ(docs.front().doc, "switch");
    EXPECT_EQ(docs[kTensionIndex].deprecated_spellings, (std::vector<std::string>{"Old_tension"}));
    EXPECT_EQ(docs[kTensionIndex].spellings, (std::vector<std::string>{"Surface_tension"}));
    EXPECT_EQ(docs[2].type, "int in [0, 16]");
    EXPECT_EQ(docs[kTensionIndex].type, "real in [0, inf]");
    const auto spellings = schema.spellings();
    EXPECT_NE(std::find(spellings.begin(), spellings.end(), "Old_tension"), spellings.end());
    EXPECT_EQ(std::find(spellings.begin(), spellings.end(), "Velocity_extension"), spellings.end());
}

TEST(OptionSchema, DescribeRecordsTheProvenanceOfValuesResolvedElsewhere)
{
    const auto schema = exampleSchema();
    // Another parser read this input, normalized the tension text and set
    // the markers from somewhere else.
    const auto in = input({{"Enabled", "yes"},
                           {"Old_tension", "0.50"},
                           {"Active_domain", "LevelSetPositive"},
                           {"Wall_normal", " "},
                           {"Some_other_key", "1"}});
    Example value;
    value.enabled = true;
    value.tension = 0.5;
    value.side = Side::Positive;
    value.markers = {7, 8};
    const auto described = schema.describe(in, value, Example{});

    EXPECT_EQ(described.value.markers, (std::vector<int>{7, 8}));
    ASSERT_EQ(described.provenance.size(), schema.size());
    const auto* enabled = described.find("Enabled");
    EXPECT_EQ(enabled->source, OptionSource::Explicit);
    EXPECT_EQ(enabled->spelling, "Enabled");
    EXPECT_EQ(enabled->input, "yes");   // canonical text is "true"
    const auto* tension = described.find("Surface_tension");
    EXPECT_EQ(tension->source, OptionSource::DeprecatedSpelling);
    EXPECT_EQ(tension->input, "0.50");
    ASSERT_EQ(described.deprecations.size(), 1u);
    EXPECT_EQ(described.deprecations.front().spelling, "Old_tension");
    const auto* side = described.find("Active_domain");
    EXPECT_EQ(side->source, OptionSource::Explicit);
    EXPECT_TRUE(side->input.empty());   // the input is the canonical text
    // An empty vector text counts as absent; the value equals the default.
    EXPECT_EQ(described.find("Wall_normal")->source, OptionSource::Default);
    const auto* markers = described.find("Wall_markers");
    EXPECT_EQ(markers->source, OptionSource::Derived);
    EXPECT_EQ(markers->origin, "resolution");
    EXPECT_EQ(described.find("Layers")->source, OptionSource::Default);

    const auto json = schema.toJson(described);
    EXPECT_NE(json.find("\"Enabled\":{\"value\":\"true\",\"source\":\"explicit\",\"spelling\":\"Enabled\","
                        "\"input\":\"yes\"}"),
              std::string::npos)
        << json;
    EXPECT_NE(json.find("\"Wall_markers\":{\"value\":\"7; 8\",\"source\":\"derived\",\"origin\":\"resolution\"}"),
              std::string::npos)
        << json;
    EXPECT_NE(json.find("\"Active_domain\":{\"value\":\"LevelSetPositive\",\"source\":\"explicit\","
                        "\"spelling\":\"Active_domain\"}"),
              std::string::npos)
        << json;
}

TEST(OptionSchema, OptionalValuesFormatAsNoneWhenUnset)
{
    struct Capped {
        std::optional<Real> cap{};
    };
    OptionSchema<Capped> schema("Capped");
    schema.key("Cap", &Capped::cap, OptionalCodec<RealCodec>{RealCodec{.minimum = 1.0}});
    EXPECT_FALSE(schema.resolve(OptionInput{}).value.cap.has_value());
    EXPECT_EQ(schema.serialize(Capped{}).front().second, "none");
    const auto capped = schema.resolve(input({{"Cap", "4"}}));
    ASSERT_TRUE(capped.value.cap.has_value());
    EXPECT_EQ(*capped.value.cap, 4.0);
    EXPECT_EQ(schema.serialize(capped.value).front().second, "4");
    EXPECT_FALSE(schema.resolve(input({{"Cap", "None"}})).value.cap.has_value());
    EXPECT_THROW((void)schema.resolve(input({{"Cap", "0.5"}})), OptionError);
    EXPECT_EQ(schema.documentation().front().type, "optional real in [1, inf]");
}

TEST(OptionSchema, JsonEscapesControlCharactersAndQuotes)
{
    const auto schema = exampleSchema();
    const auto resolved = schema.resolve(input({{"Level_set_field_name", "a\"b\\c\x01"}}));
    const auto json = schema.toJson(resolved);
    EXPECT_NE(json.find("\"value\":\"a\\\"b\\\\c\\u0001\""), std::string::npos) << json;
}

TEST(OptionSchema, JsonStringStylesPreserveSchemaAndArtifactControlEscapes)
{
    const std::string text{"\"\\\b\f\n\r\t\x01\x1f/"};
    EXPECT_EQ(json::quote(text), "\"\\\"\\\\\\b\\f\\n\\r\\t\\u0001\\u001f/\"");
    EXPECT_EQ(json::quote(text, json::ControlEscapes::UnicodeBackspaceAndFormFeed),
              "\"\\\"\\\\\\u0008\\u000c\\n\\r\\t\\u0001\\u001f/\"");
    EXPECT_EQ(json::quote(std::string("a\0b", 3)), "\"a\\u0000b\"");
    EXPECT_EQ(json::quote("caf\xc3\xa9"), "\"caf\xc3\xa9\"");

    const auto resolved = exampleSchema().resolve(input({{"Level_set_field_name", text}}));
    EXPECT_NE(exampleSchema().toJson(resolved).find(
                  "\"value\":\"\\\"\\\\\\u0008\\u000c\\n\\r\\t\\u0001\\u001f/\""),
              std::string::npos);
}

TEST(OptionSchema, JsonNumbersKeepArtifactPrecisionAndRejectNonfiniteValues)
{
    struct CommaDecimal : std::numpunct<char> {
        char do_decimal_point() const override { return ','; }
    };
    struct RestoreLocale {
        std::locale previous{std::locale()};
        ~RestoreLocale() { std::locale::global(previous); }
    } restore_locale;
    std::locale::global(std::locale(restore_locale.previous, new CommaDecimal));

    EXPECT_EQ(json::number(Real{0.1}, "nonfinite artifact value"), "0.10000000000000001");
    EXPECT_EQ(RealCodec{}.format(Real{0.1}), "0.1");
    EXPECT_EQ(json::number(Real{-0.0}, "nonfinite artifact value"), "-0");
    EXPECT_EQ(json::number(Real{1.0e-12}, "nonfinite artifact value"), "9.9999999999999998e-13");
    for (const auto value : {std::numeric_limits<Real>::infinity(),
                             -std::numeric_limits<Real>::infinity(),
                             std::numeric_limits<Real>::quiet_NaN()}) {
        try {
            static_cast<void>(json::number(value, "nonfinite artifact value"));
            FAIL() << "Expected nonfinite JSON number rejection";
        } catch (const std::invalid_argument& error) {
            EXPECT_EQ(std::string(error.what()), "nonfinite artifact value");
        }
    }
}

TEST(OptionSchema, InputSelectionPreservesRawEntriesAndSpellingPrecedence)
{
    const auto supplied = input({{"Surface_tension", " 7 "},
                                 {"Old_tension", " 3 "},
                                 {"Enabled", "true"},
                                 {"Old_tension", " 5 "},
                                 {"Unrelated", ""}});
    const auto schema = exampleSchema();
    const auto spellings = schema.spellings({"Surface_tension"});
    EXPECT_EQ(spellings, (std::vector<std::string>{"Old_tension", "Surface_tension"}));
    const auto selected = supplied.selectSpellings(spellings);
    ASSERT_EQ(selected.entries().size(), 3u);
    EXPECT_EQ(selected.entries()[0].spelling, "Surface_tension");
    EXPECT_EQ(selected.entries()[0].text, " 7 ");
    EXPECT_EQ(selected.entries()[1].spelling, "Old_tension");
    EXPECT_EQ(selected.entries()[1].text, " 3 ");
    EXPECT_EQ(selected.entries()[2].text, " 5 ");
    const auto resolved = schema.resolve(selected);
    EXPECT_EQ(resolved.value.tension, Real{3});
    ASSERT_NE(resolved.find("Surface_tension"), nullptr);
    EXPECT_EQ(resolved.find("Surface_tension")->source, OptionSource::DeprecatedSpelling);
    EXPECT_EQ(resolved.find("Surface_tension")->spelling, "Old_tension");

    const auto excluded = supplied.excludeSpellings(spellings);
    ASSERT_EQ(excluded.entries().size(), 2u);
    EXPECT_EQ(excluded.entries()[0].spelling, "Enabled");
    EXPECT_EQ(excluded.entries()[1].spelling, "Unrelated");
    EXPECT_TRUE(excluded.entries()[1].text.empty());
    EXPECT_TRUE(supplied.selectSpellings({}).entries().empty());
    EXPECT_EQ(supplied.excludeSpellings({}).entries().size(), supplied.entries().size());
    EXPECT_EQ(schema.spellings({"Surface_tension", "Enabled"}),
              (std::vector<std::string>{"Enabled", "Enable_it", "Old_tension", "Surface_tension"}));
    EXPECT_TRUE(schema.spellings({"not_a_key"}).empty());
}

TEST(OptionSchema, RealsSerializeToTheShortestExactText)
{
    const auto schema = exampleSchema();
    for (const auto* text : {"0.1", "1e-12", "0.30000000000000004", "1234567.875", "2.2250738585072014e-308"}) {
        const auto resolved = schema.resolve(input({{"Surface_tension", text}}));
        const auto serialized = schema.serialize(resolved.value);
        const auto again = schema.resolve(input({{"Surface_tension", serialized[kTensionIndex].second}}));
        EXPECT_EQ(again.value.tension, resolved.value.tension) << text << " -> " << serialized[kTensionIndex].second;
    }
}

TEST(OptionSchema, DeclarationMistakesAreLogicErrors)
{
    OptionSchema<Example> schema("Broken");
    schema.key("Enabled", &Example::enabled, BoolCodec{});
    // A key declared twice.
    EXPECT_THROW(schema.key("Enabled", &Example::relaxed_flag, BoolCodec{}), std::logic_error);
    // A spelling of two keys.
    EXPECT_THROW(schema.key("Other", &Example::relaxed_flag, BoolCodec{},
                            KeySettings{.spellings = {{"Enabled", false}, {"Other", false}}}),
                 std::logic_error);
    // A key without its canonical spelling, or with it only as deprecated.
    EXPECT_THROW(schema.key("Third", &Example::relaxed_flag, BoolCodec{},
                            KeySettings{.spellings = {{"Third_alias", false}}}),
                 std::logic_error);
    EXPECT_THROW(schema.key("Fourth", &Example::relaxed_flag, BoolCodec{},
                            KeySettings{.spellings = {{"Fourth", true}}}),
                 std::logic_error);
    // A spelling repeated within one key.
    EXPECT_THROW(schema.key("Fifth", &Example::relaxed_flag, BoolCodec{},
                            KeySettings{.spellings = {{"Fifth", false}, {"Fifth", false}}}),
                 std::logic_error);
    // Rules naming undeclared keys, and retired spellings that are active.
    EXPECT_THROW(schema.implies("Enabled", "Missing", "true"), std::logic_error);
    EXPECT_THROW(schema.conflicts("Missing", "Enabled", "no"), std::logic_error);
    EXPECT_THROW(schema.retired({"Enabled"}, "no"), std::logic_error);
    schema.retired({"Gone"}, "retired");
    EXPECT_THROW(schema.key("Gone", &Example::relaxed_flag, BoolCodec{}), std::logic_error);
    EXPECT_EQ(schema.size(), 1u);
}

} // namespace
