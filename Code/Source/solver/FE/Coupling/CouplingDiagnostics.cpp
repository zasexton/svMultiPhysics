/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingDiagnostics.h"

#include "Coupling/CouplingGraph.h"
#include "Coupling/PartitionedCouplingPlan.h"
#include "Core/FEException.h"

#include <algorithm>
#include <sstream>
#include <string_view>
#include <utility>

namespace svmp {
namespace FE {
namespace coupling {

namespace {

constexpr CouplingDiagnosticCategory kReportCategories[] = {
    CouplingDiagnosticCategory::MissingContextValue,
    CouplingDiagnosticCategory::DependencyMismatch,
    CouplingDiagnosticCategory::TemporalPolicyFailure,
    CouplingDiagnosticCategory::TransferFailure,
    CouplingDiagnosticCategory::CycleVisibility,
    CouplingDiagnosticCategory::BlockCoverageMismatch,
    CouplingDiagnosticCategory::General,
};

bool contains(std::string_view text, std::string_view needle) noexcept
{
    return text.find(needle) != std::string_view::npos;
}

std::string portLabel(const CouplingPortId& port)
{
    return port.contract_instance_name + "/" + port.port_name;
}

template <class Range, class Label>
void appendLabelList(std::ostringstream& os,
                     std::string_view title,
                     const Range& range,
                     Label label)
{
    os << title << "=" << range.size();
    if (range.empty()) {
        return;
    }

    os << " [";
    bool first = true;
    for (const auto& item : range) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << label(item);
    }
    os << "]";
}

const char* remediationForCategory(CouplingDiagnosticCategory category) noexcept
{
    switch (category) {
    case CouplingDiagnosticCategory::MissingContextValue:
        return "register the missing participant, field, region, or provider in the CouplingContext, or mark the declaration optional";
    case CouplingDiagnosticCategory::DependencyMismatch:
        return "align the contract dependency declaration with the installed Forms/Systems analysis metadata";
    case CouplingDiagnosticCategory::TemporalPolicyFailure:
        return "declare the required temporal symbol or choose an endpoint slot supported by the provider";
    case CouplingDiagnosticCategory::TransferFailure:
        return "fix the partitioned exchange endpoint, value descriptor, region, or transfer declaration";
    case CouplingDiagnosticCategory::CycleVisibility:
        return "handle the reported partitioned exchange cycle in the driver solve strategy";
    case CouplingDiagnosticCategory::BlockCoverageMismatch:
        return "update expected block declarations or installed coupling metadata so matrix/vector coverage agrees";
    case CouplingDiagnosticCategory::General:
        return "inspect the reported contract and context labels";
    }
    return "inspect the reported contract and context labels";
}

void appendDiagnosticReportLine(std::ostringstream& os,
                                const CouplingDiagnostic& diagnostic)
{
    const auto category = classifyDiagnostic(diagnostic);
    os << "[" << toString(category) << "] "
       << formatDiagnostic(diagnostic)
       << " remediation='" << remediationForCategory(category) << "'";
}

} // namespace

bool CouplingValidationResult::ok() const noexcept
{
    for (const auto& diagnostic : diagnostics) {
        if (diagnostic.severity == CouplingDiagnosticSeverity::Error) {
            return false;
        }
    }
    return true;
}

void CouplingValidationResult::add(CouplingDiagnostic diagnostic)
{
    diagnostics.push_back(std::move(diagnostic));
}

void CouplingValidationResult::addError(std::string message)
{
    diagnostics.push_back(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Error,
        .message = std::move(message),
    });
}

void CouplingValidationResult::addWarning(std::string message)
{
    diagnostics.push_back(CouplingDiagnostic{
        .severity = CouplingDiagnosticSeverity::Warning,
        .message = std::move(message),
    });
}

void CouplingValidationResult::append(const CouplingValidationResult& other)
{
    diagnostics.insert(diagnostics.end(), other.diagnostics.begin(), other.diagnostics.end());
}

const char* toString(CouplingDiagnosticSeverity severity) noexcept
{
    switch (severity) {
    case CouplingDiagnosticSeverity::Info:
        return "info";
    case CouplingDiagnosticSeverity::Warning:
        return "warning";
    case CouplingDiagnosticSeverity::Error:
        return "error";
    }
    return "unknown";
}

const char* toString(CouplingDiagnosticCategory category) noexcept
{
    switch (category) {
    case CouplingDiagnosticCategory::General:
        return "general";
    case CouplingDiagnosticCategory::MissingContextValue:
        return "missing-context";
    case CouplingDiagnosticCategory::DependencyMismatch:
        return "dependency-mismatch";
    case CouplingDiagnosticCategory::TemporalPolicyFailure:
        return "temporal-policy";
    case CouplingDiagnosticCategory::TransferFailure:
        return "transfer";
    case CouplingDiagnosticCategory::CycleVisibility:
        return "cycle";
    case CouplingDiagnosticCategory::BlockCoverageMismatch:
        return "block-coverage";
    }
    return "general";
}

CouplingDiagnosticCategory classifyDiagnostic(
    const CouplingDiagnostic& diagnostic)
{
    if (diagnostic.category != CouplingDiagnosticCategory::General) {
        return diagnostic.category;
    }

    const std::string_view message = diagnostic.message;
    if (contains(message, "block") ||
        contains(message, "matrix evidence") ||
        contains(message, "installed matrix") ||
        contains(message, "zero coupling")) {
        return CouplingDiagnosticCategory::BlockCoverageMismatch;
    }
    if (contains(message, "cycle")) {
        return CouplingDiagnosticCategory::CycleVisibility;
    }
    if (contains(message, "temporal") ||
        contains(message, "time-step") ||
        contains(message, "history") ||
        contains(message, "stage")) {
        return CouplingDiagnosticCategory::TemporalPolicyFailure;
    }
    if (contains(message, "transfer") ||
        contains(message, "endpoint") ||
        contains(message, "exchange") ||
        contains(message, "value descriptor") ||
        contains(message, "component count") ||
        contains(message, "ParameterRegistry") ||
        contains(message, "AuxiliaryInputRegistry") ||
        contains(message, "AuxiliaryOutputRegistry") ||
        contains(message, "ExternalBuffer") ||
        contains(message, "interface map provenance")) {
        return CouplingDiagnosticCategory::TransferFailure;
    }
    if (contains(message, "dependency") ||
        contains(message, "provider metadata") ||
        contains(message, "graph variable") ||
        contains(message, "installed metadata")) {
        return CouplingDiagnosticCategory::DependencyMismatch;
    }
    if (contains(message, "missing from the context") ||
        contains(message, "is missing") ||
        contains(message, "has no owning system") ||
        contains(message, "cannot be lowered")) {
        return CouplingDiagnosticCategory::MissingContextValue;
    }
    return CouplingDiagnosticCategory::General;
}

std::string formatDiagnostic(const CouplingDiagnostic& diagnostic)
{
    std::ostringstream os;
    os << toString(diagnostic.severity) << ": " << diagnostic.message;
    if (!diagnostic.contract_name.empty()) {
        os << " contract='" << diagnostic.contract_name << "'";
    }
    if (!diagnostic.participant_name.empty()) {
        os << " participant='" << diagnostic.participant_name << "'";
    }
    if (!diagnostic.field_name.empty()) {
        os << " field='" << diagnostic.field_name << "'";
    }
    if (!diagnostic.region_name.empty()) {
        os << " region='" << diagnostic.region_name << "'";
    }
    if (!diagnostic.endpoint_name.empty()) {
        os << " endpoint='" << diagnostic.endpoint_name << "'";
    }
    return os.str();
}

std::string formatGraphSummary(const CouplingGraphSnapshot& snapshot)
{
    std::ostringstream os;
    os << "graph summary: ";
    appendLabelList(
        os,
        "participants",
        snapshot.participants,
        [](const CouplingGraphParticipantNode& node) {
            return node.participant.participant_name;
        });
    os << "; ";
    appendLabelList(
        os,
        "fields",
        snapshot.fields,
        [](const CouplingGraphFieldNode& node) {
            return node.field.participant_name + "/" + node.field.field_name;
        });
    os << "; ";
    appendLabelList(
        os,
        "regions",
        snapshot.regions,
        [](const CouplingGraphRegionNode& node) {
            return node.region.participant_name + "/" + node.region.region_name;
        });
    os << "; ";
    appendLabelList(
        os,
        "shared_regions",
        snapshot.shared_regions,
        [](const CouplingGraphSharedRegionNode& node) {
            return node.shared_region.name;
        });
    os << "; ";
    appendLabelList(
        os,
        "contracts",
        snapshot.contract_instances,
        [](const CouplingGraphContractInstanceNode& node) {
            if (node.contract_type.empty()) {
                return node.contract_name;
            }
            return node.contract_name + "(" + node.contract_type + ")";
        });
    os << "; dependencies=" << snapshot.dependency_expectations.size()
       << "; expected_blocks=" << snapshot.expected_blocks.size()
       << "; temporal_requirements=" << snapshot.temporal_requirements.size()
       << "; geometry_requirements=" << snapshot.geometry_requirements.size()
       << "; partitioned_exchanges="
       << snapshot.partitioned_exchange_declarations.size();
    return os.str();
}

std::string formatPartitionedPlanSummary(const PartitionedCouplingPlan& plan)
{
    std::ostringstream os;
    os << "partitioned plan summary: exchanges=" << plan.exchanges.size();
    if (!plan.exchanges.empty()) {
        os << " [";
        for (std::size_t i = 0; i < plan.exchanges.size(); ++i) {
            if (i != 0u) {
                os << ", ";
            }
            os << portLabel(plan.exchanges[i].producer_port)
               << " -> "
               << portLabel(plan.exchanges[i].consumer_port);
        }
        os << "]";
    }
    os << "; group_hints=" << plan.group_hints.size();
    if (!plan.group_hints.empty()) {
        os << " [";
        for (std::size_t i = 0; i < plan.group_hints.size(); ++i) {
            if (i != 0u) {
                os << ", ";
            }
            os << plan.group_hints[i].name << "(";
            for (std::size_t j = 0; j < plan.group_hints[i].participant_names.size(); ++j) {
                if (j != 0u) {
                    os << ",";
                }
                os << plan.group_hints[i].participant_names[j];
            }
            os << ")";
        }
        os << "]";
    }
    os << "; cycles=" << plan.cycles.size();
    if (!plan.cycles.empty()) {
        os << " [";
        for (std::size_t i = 0; i < plan.cycles.size(); ++i) {
            if (i != 0u) {
                os << ", ";
            }
            for (std::size_t j = 0; j < plan.cycles[i].ports.size(); ++j) {
                if (j != 0u) {
                    os << " -> ";
                }
                os << portLabel(plan.cycles[i].ports[j]);
            }
        }
        os << "]";
    }
    return os.str();
}

std::string formatDiagnosticsReport(const CouplingValidationResult& result)
{
    std::ostringstream os;
    os << "diagnostics: total=" << result.diagnostics.size()
       << " status=" << (result.ok() ? "ok" : "failed");
    for (const auto category : kReportCategories) {
        for (const auto& diagnostic : result.diagnostics) {
            if (classifyDiagnostic(diagnostic) != category) {
                continue;
            }
            os << '\n';
            appendDiagnosticReportLine(os, diagnostic);
        }
    }
    return os.str();
}

std::string formatDiagnosticsReport(const CouplingGraphSnapshot& snapshot,
                                    const CouplingValidationResult& result)
{
    return formatGraphSummary(snapshot) + "\n" +
           formatDiagnosticsReport(result);
}

std::string formatDiagnosticsReport(const CouplingGraphSnapshot& snapshot,
                                    const CouplingValidationResult& result,
                                    const PartitionedCouplingPlan& plan)
{
    return formatGraphSummary(snapshot) + "\n" +
           formatPartitionedPlanSummary(plan) + "\n" +
           formatDiagnosticsReport(result);
}

std::string formatDiagnostics(const CouplingValidationResult& result)
{
    std::ostringstream os;
    for (std::size_t i = 0; i < result.diagnostics.size(); ++i) {
        if (i != 0u) {
            os << '\n';
        }
        os << formatDiagnostic(result.diagnostics[i]);
    }
    return os.str();
}

void throwIfInvalid(const CouplingValidationResult& result)
{
    if (!result.ok()) {
        FE_THROW(InvalidArgumentException, formatDiagnostics(result));
    }
}

} // namespace coupling
} // namespace FE
} // namespace svmp
