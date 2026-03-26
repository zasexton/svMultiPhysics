#ifndef SVMP_FE_SYSTEMS_FE_QUANTITY_REGISTRY_H
#define SVMP_FE_SYSTEMS_FE_QUANTITY_REGISTRY_H

/**
 * @file FEQuantityRegistry.h
 * @brief Registry for FE-backed auxiliary input quantity definitions.
 *
 * `FEQuantityRegistry` stores `FEQuantityDefinition` entries — the
 * metadata describing each FE-backed auxiliary input (kind, shape,
 * referenced fields, capabilities).  It is separate from
 * `AuxiliaryInputRegistry` (which stores evaluated numeric values
 * and dependency ordering).
 *
 * Owned by `FESystem` and consulted during:
 * - binding validation (shape checks, coupling-mode checks)
 * - monolithic assembly (dI/du linearization queries)
 * - diagnostics and introspection
 *
 * This class is physics-agnostic.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Systems/FEQuantityDefinition.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Registry of FE-backed quantity definitions.
 */
class FEQuantityRegistry {
public:
    FEQuantityRegistry() = default;

    /// Register a new FE-backed quantity definition.
    /// Returns the definition ID (stable index).
    std::size_t registerDefinition(FEQuantityDefinition def);

    /// Check if a definition with the given name exists.
    [[nodiscard]] bool hasDefinition(std::string_view name) const noexcept;

    /// Get a definition by name.  Throws if not found.
    [[nodiscard]] const FEQuantityDefinition& get(std::string_view name) const;

    /// Get a definition by ID.  Throws if out of range.
    [[nodiscard]] const FEQuantityDefinition& get(std::size_t id) const;

    /// Get the definition ID for a name.  Returns nullopt if not found.
    [[nodiscard]] std::optional<std::size_t> idOf(std::string_view name) const noexcept;

    /// Number of registered definitions.
    [[nodiscard]] std::size_t count() const noexcept { return definitions_.size(); }

    /// All registered definitions (in registration order).
    [[nodiscard]] const std::vector<FEQuantityDefinition>& all() const noexcept
    {
        return definitions_;
    }

    /// All definitions with the given kind.
    [[nodiscard]] std::vector<const FEQuantityDefinition*> byKind(FEQuantityKind kind) const;

    /// All definitions that reference a given field.
    [[nodiscard]] std::vector<const FEQuantityDefinition*> byField(FieldId fid) const;

    /// All definitions that support monolithic linearization.
    [[nodiscard]] std::vector<const FEQuantityDefinition*> monolithicCapable() const;

    /// Clear all definitions.
    void clear();

private:
    std::vector<FEQuantityDefinition> definitions_{};
    std::unordered_map<std::string, std::size_t> name_to_id_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FE_QUANTITY_REGISTRY_H
