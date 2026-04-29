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

std::unique_ptr<CouplingContract> CouplingRegistry::create(std::string_view name) const
{
    const auto it = std::find_if(entries_.begin(), entries_.end(),
                                 [name](const Entry& entry) {
                                     return entry.name == name;
                                 });
    FE_THROW_IF(it == entries_.end(), InvalidArgumentException,
                "unknown coupling contract registry entry");
    return it->factory();
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

} // namespace coupling
} // namespace FE
} // namespace svmp
