#ifndef SVMP_FE_POSTPROCESSING_DERIVED_RESULT_REGISTRY_H
#define SVMP_FE_POSTPROCESSING_DERIVED_RESULT_REGISTRY_H

/**
 * @file DerivedResultRegistry.h
 * @brief Ordered registry for physics-agnostic derived result definitions.
 */

#include "PostProcessing/DerivedResultTypes.h"

#include <cstddef>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace post {

struct DerivedResultHandle {
    std::size_t id{static_cast<std::size_t>(-1)};

    [[nodiscard]] bool valid() const noexcept
    {
        return id != static_cast<std::size_t>(-1);
    }
};

class DerivedResultRegistry {
public:
    [[nodiscard]] DerivedResultHandle registerDefinition(DerivedResultDefinition def);

    [[nodiscard]] const DerivedResultDefinition& get(DerivedResultHandle handle) const;
    [[nodiscard]] const DerivedResultDefinition& get(std::string_view name) const;

    [[nodiscard]] bool contains(std::string_view name) const noexcept;
    [[nodiscard]] std::span<const DerivedResultDefinition> all() const noexcept
    {
        return definitions_;
    }

private:
    std::vector<DerivedResultDefinition> definitions_{};
    std::unordered_map<std::string, std::size_t> name_to_id_{};
};

[[nodiscard]] std::vector<FieldId> collectReferencedFields(const forms::FormExpr& expr);
void validateDerivedResultDefinition(DerivedResultDefinition& def);

} // namespace post
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_POSTPROCESSING_DERIVED_RESULT_REGISTRY_H
