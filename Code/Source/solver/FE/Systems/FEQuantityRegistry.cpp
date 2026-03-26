#include "Systems/FEQuantityRegistry.h"

namespace svmp {
namespace FE {
namespace systems {

std::size_t FEQuantityRegistry::registerDefinition(FEQuantityDefinition def)
{
    FE_THROW_IF(def.name.empty(), InvalidArgumentException,
                "FEQuantityRegistry: empty definition name");
    FE_THROW_IF(name_to_id_.count(def.name) != 0, InvalidArgumentException,
                "FEQuantityRegistry: duplicate definition '" + def.name + "'");

    const auto id = definitions_.size();
    name_to_id_[def.name] = id;
    definitions_.push_back(std::move(def));
    return id;
}

bool FEQuantityRegistry::hasDefinition(std::string_view name) const noexcept
{
    return name_to_id_.count(std::string(name)) != 0;
}

const FEQuantityDefinition& FEQuantityRegistry::get(std::string_view name) const
{
    auto it = name_to_id_.find(std::string(name));
    FE_THROW_IF(it == name_to_id_.end(), InvalidArgumentException,
                "FEQuantityRegistry: unknown quantity '" + std::string(name) + "'");
    return definitions_[it->second];
}

const FEQuantityDefinition& FEQuantityRegistry::get(std::size_t id) const
{
    FE_THROW_IF(id >= definitions_.size(), InvalidArgumentException,
                "FEQuantityRegistry: ID " + std::to_string(id) + " out of range");
    return definitions_[id];
}

std::optional<std::size_t> FEQuantityRegistry::idOf(std::string_view name) const noexcept
{
    auto it = name_to_id_.find(std::string(name));
    if (it == name_to_id_.end()) return std::nullopt;
    return it->second;
}

std::vector<const FEQuantityDefinition*> FEQuantityRegistry::byKind(FEQuantityKind kind) const
{
    std::vector<const FEQuantityDefinition*> result;
    for (const auto& def : definitions_) {
        if (def.kind == kind) result.push_back(&def);
    }
    return result;
}

std::vector<const FEQuantityDefinition*> FEQuantityRegistry::byField(FieldId fid) const
{
    std::vector<const FEQuantityDefinition*> result;
    for (const auto& def : definitions_) {
        for (const auto f : def.referenced_fields) {
            if (f == fid) { result.push_back(&def); break; }
        }
    }
    return result;
}

std::vector<const FEQuantityDefinition*> FEQuantityRegistry::monolithicCapable() const
{
    std::vector<const FEQuantityDefinition*> result;
    for (const auto& def : definitions_) {
        if (def.capabilities.monolithic_linearization) result.push_back(&def);
    }
    return result;
}

void FEQuantityRegistry::clear()
{
    definitions_.clear();
    name_to_id_.clear();
}

} // namespace systems
} // namespace FE
} // namespace svmp
