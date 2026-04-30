/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingRegistry.h"

#include "Core/FEException.h"

#include <algorithm>
#include <utility>

namespace svmp {
namespace FE {
namespace coupling {

void CouplingRegistry::registerContract(std::string name, Factory factory)
{
    FE_THROW_IF(name.empty(), InvalidArgumentException,
                "coupling contract registry entry requires a name");
    FE_THROW_IF(!factory, InvalidArgumentException,
                "coupling contract registry entry requires a factory");
    FE_THROW_IF(contains(name), InvalidArgumentException,
                "duplicate coupling contract registry entry: " + name);
    entries_.push_back(Entry{std::move(name), std::move(factory)});
}

bool CouplingRegistry::contains(std::string_view name) const noexcept
{
    return std::any_of(entries_.begin(), entries_.end(),
                       [name](const Entry& entry) { return entry.name == name; });
}

bool CouplingRegistry::supportsMode(std::string_view name, CouplingMode mode) const
{
    const auto it = std::find_if(entries_.begin(), entries_.end(),
                                 [name](const Entry& entry) {
                                     return entry.name == name;
                                 });
    FE_THROW_IF(it == entries_.end(), InvalidArgumentException,
                "unknown coupling contract registry entry");
    const auto contract = it->factory();
    FE_THROW_IF(contract == nullptr, InvalidArgumentException,
                "coupling contract factory returned null");
    return contract->supportsCouplingMode(mode);
}

std::unique_ptr<CouplingContract> CouplingRegistry::create(std::string_view name) const
{
    const auto it = std::find_if(entries_.begin(), entries_.end(),
                                 [name](const Entry& entry) {
                                     return entry.name == name;
                                 });
    FE_THROW_IF(it == entries_.end(), InvalidArgumentException,
                "unknown coupling contract registry entry");
    auto contract = it->factory();
    FE_THROW_IF(contract == nullptr, InvalidArgumentException,
                "coupling contract factory returned null");
    return contract;
}

CouplingValidationResult CouplingRegistry::validateDeclarations(
    std::span<const CouplingContractDeclaration> declarations) const
{
    CouplingValidationResult result;
    for (std::size_t i = 0; i < declarations.size(); ++i) {
        const auto& declaration = declarations[i];
        if (!declaration.contract_type.empty()) {
            const auto it = std::find_if(entries_.begin(),
                                         entries_.end(),
                                         [&](const Entry& entry) {
                                             return entry.name ==
                                                    declaration.contract_type;
                                         });
            if (it == entries_.end()) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .message = "coupling contract declaration type is not registered",
                });
            } else {
                const auto contract = it->factory();
                if (contract == nullptr) {
                    result.add(CouplingDiagnostic{
                        .severity = CouplingDiagnosticSeverity::Error,
                        .contract_name = declaration.contract_name,
                        .message = "coupling contract factory returned null",
                    });
                } else if (contract->name() != declaration.contract_type) {
                    result.add(CouplingDiagnostic{
                        .severity = CouplingDiagnosticSeverity::Error,
                        .contract_name = declaration.contract_name,
                        .message = "coupling contract declaration type does not match the registered contract name",
                    });
                }
            }
        }

        if (declaration.contract_name.empty()) {
            continue;
        }
        for (std::size_t j = i + 1u; j < declarations.size(); ++j) {
            if (declaration.contract_name == declarations[j].contract_name) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .contract_name = declaration.contract_name,
                    .message = "duplicate coupling contract instance name",
                });
            }
        }
    }
    return result;
}

std::vector<std::string> CouplingRegistry::names() const
{
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto& entry : entries_) {
        out.push_back(entry.name);
    }
    return out;
}

std::vector<std::string> CouplingRegistry::namesSupporting(CouplingMode mode) const
{
    std::vector<std::string> out;
    for (const auto& entry : entries_) {
        const auto contract = entry.factory();
        FE_THROW_IF(contract == nullptr, InvalidArgumentException,
                    "coupling contract factory returned null");
        if (contract->supportsCouplingMode(mode)) {
            out.push_back(entry.name);
        }
    }
    return out;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
