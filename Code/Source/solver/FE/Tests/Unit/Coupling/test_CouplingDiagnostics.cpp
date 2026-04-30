#include "Coupling/CouplingDiagnostics.h"

#include "Coupling/CouplingGraph.h"
#include "Coupling/PartitionedCouplingPlan.h"

#include <gtest/gtest.h>

#include <string>

using namespace svmp::FE::coupling;

namespace {

void expectBefore(const std::string& text,
                  const std::string& first,
                  const std::string& second)
{
    const auto first_pos = text.find(first);
    const auto second_pos = text.find(second);
    ASSERT_NE(first_pos, std::string::npos) << text;
    ASSERT_NE(second_pos, std::string::npos) << text;
    EXPECT_LT(first_pos, second_pos) << text;
}

CouplingGraphSnapshot diagnosticSnapshot()
{
    CouplingGraphSnapshot snapshot;
    snapshot.participants.push_back({
        .participant = {
            .participant_name = "left",
            .system_name = "left_system",
        },
    });
    snapshot.participants.push_back({
        .participant = {
            .participant_name = "right",
            .system_name = "right_system",
        },
    });
    snapshot.fields.push_back({
        .field = {
            .participant_name = "left",
            .system_name = "left_system",
            .field_name = "velocity",
            .field_id = 3,
        },
    });
    snapshot.regions.push_back({
        .region = {
            .participant_name = "left",
            .system_name = "left_system",
            .region_name = "interface",
            .kind = CouplingRegionKind::InterfaceFace,
            .marker = 7,
        },
    });
    snapshot.shared_regions.push_back({
        .shared_region = {
            .name = "fsi_interface",
        },
    });
    snapshot.contract_instances.push_back({
        .contract_type = "fsi",
        .contract_name = "fsi_wall",
    });
    snapshot.dependency_expectations.push_back({
        .contract_name = "fsi_wall",
    });
    snapshot.expected_blocks.push_back({
        .contract_name = "fsi_wall",
    });
    snapshot.temporal_requirements.push_back({
        .contract_name = "fsi_wall",
    });
    snapshot.geometry_requirements.push_back({
        .contract_name = "fsi_wall",
    });
    snapshot.partitioned_exchange_declarations.push_back({
        .contract_name = "fsi_wall",
        .declaration = {
            .producer_port = {"fsi_wall", "fluid_out"},
            .consumer_port = {"fsi_wall", "solid_in"},
        },
    });
    return snapshot;
}

PartitionedCouplingPlan diagnosticPartitionedPlan()
{
    CouplingExchange exchange;
    exchange.producer_port = {"fsi_wall", "fluid_out"};
    exchange.consumer_port = {"fsi_wall", "solid_in"};

    PartitionedCouplingPlan plan;
    plan.exchanges.push_back(exchange);
    plan.group_hints.push_back({
        .name = "interface_pair",
        .participant_names = {"left", "right"},
    });
    plan.cycles.push_back({
        .ports = {
            {"fsi_wall", "fluid_out"},
            {"fsi_wall", "solid_in"},
            {"fsi_wall", "fluid_out"},
        },
    });
    return plan;
}

CouplingValidationResult categorizedDiagnostics()
{
    CouplingValidationResult result;
    result.add(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .contract_name = "fsi_wall",
        .participant_name = "left",
        .field_name = "velocity",
        .message = "required coupling field is missing from the context",
    });
    result.add(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .contract_name = "fsi_wall",
        .participant_name = "left",
        .message = "required provider metadata is missing for non-field dependency: Parameter(left/penalty)",
    });
    result.add(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .contract_name = "fsi_wall",
        .participant_name = "left",
        .endpoint_name = "velocity",
        .message = "field endpoint temporal slot does not support stage values",
    });
    result.add(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .contract_name = "fsi_wall",
        .endpoint_name = "fluid_out",
        .message = "interface partitioned transfer requires interface map provenance",
    });
    result.add(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Warning,
        .category = CouplingDiagnosticCategory::CycleVisibility,
        .contract_name = "fsi_wall",
        .message = "partitioned exchange cycle is visible to the driver",
    });
    result.add(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .contract_name = "fsi_wall",
        .message = "expected monolithic block is missing installed matrix evidence",
    });
    return result;
}

} // namespace

TEST(CouplingDiagnostics, FormatsGraphAndPartitionedSummaries)
{
    const auto snapshot = diagnosticSnapshot();
    const auto plan = diagnosticPartitionedPlan();
    const CouplingValidationResult validation;

    const auto text = formatDiagnosticsReport(snapshot, validation, plan);
    EXPECT_NE(text.find("graph summary: participants=2 [left, right]"),
              std::string::npos);
    EXPECT_NE(text.find("fields=1 [left/velocity]"), std::string::npos);
    EXPECT_NE(text.find("shared_regions=1 [fsi_interface]"),
              std::string::npos);
    EXPECT_NE(text.find("contracts=1 [fsi_wall(fsi)]"), std::string::npos);
    EXPECT_NE(text.find("dependencies=1"), std::string::npos);
    EXPECT_NE(text.find("expected_blocks=1"), std::string::npos);
    EXPECT_NE(text.find("partitioned plan summary: exchanges=1 "
                        "[fsi_wall/fluid_out -> fsi_wall/solid_in]"),
              std::string::npos);
    EXPECT_NE(text.find("group_hints=1 [interface_pair(left,right)]"),
              std::string::npos);
    EXPECT_NE(text.find("cycles=1 [fsi_wall/fluid_out -> "
                        "fsi_wall/solid_in -> fsi_wall/fluid_out]"),
              std::string::npos);
    EXPECT_NE(text.find("diagnostics: total=0 status=ok"), std::string::npos);
}

TEST(CouplingDiagnostics, FormatsStableActionableDiagnosticCategories)
{
    const auto text = formatDiagnosticsReport(categorizedDiagnostics());

    expectBefore(text, "[missing-context]", "[dependency-mismatch]");
    expectBefore(text, "[dependency-mismatch]", "[temporal-policy]");
    expectBefore(text, "[temporal-policy]", "[transfer]");
    expectBefore(text, "[transfer]", "[cycle]");
    expectBefore(text, "[cycle]", "[block-coverage]");

    EXPECT_NE(text.find("contract='fsi_wall'"), std::string::npos);
    EXPECT_NE(text.find("participant='left'"), std::string::npos);
    EXPECT_NE(text.find("field='velocity'"), std::string::npos);
    EXPECT_NE(text.find("endpoint='fluid_out'"), std::string::npos);
    EXPECT_NE(text.find("remediation='register the missing participant"),
              std::string::npos);
    EXPECT_NE(text.find("remediation='align the contract dependency declaration"),
              std::string::npos);
    EXPECT_NE(text.find("remediation='declare the required temporal symbol"),
              std::string::npos);
    EXPECT_NE(text.find("remediation='fix the partitioned exchange endpoint"),
              std::string::npos);
    EXPECT_NE(text.find("remediation='handle the reported partitioned exchange cycle"),
              std::string::npos);
    EXPECT_NE(text.find("remediation='update expected block declarations"),
              std::string::npos);
}
