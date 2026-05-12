/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_LEVELSETINTERFACEBUILDER_H
#define SVMP_FE_INTERFACES_LEVELSETINTERFACEBUILDER_H

/**
 * @file LevelSetInterfaceBuilder.h
 * @brief Linear level-set cell cutters for generated interface domains.
 */

#include "Core/Types.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#include <array>
#include <string>
#include <functional>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace interfaces {

struct LevelSetCellCutInput {
    MeshIndex parent_cell{static_cast<MeshIndex>(-1)};
    ElementType element_type{ElementType::Unknown};
    std::vector<std::array<Real, 3>> node_coordinates{};
    std::vector<Real> level_set_values{};
};

struct LevelSetCellCutResult {
    std::vector<CutInterfaceFragment> fragments{};
    CutInterfaceDegeneracy degeneracy{CutInterfaceDegeneracy::None};
    bool supported{true};
    std::string diagnostic{};

    [[nodiscard]] bool hasActiveFragments() const noexcept {
        for (const auto& fragment : fragments) {
            if (fragment.active()) {
                return true;
            }
        }
        return false;
    }
};

using LevelSetCellCutFunction =
    std::function<LevelSetCellCutResult(const CutInterfaceDomainRequest&,
                                        const LevelSetCellCutInput&)>;

struct LevelSetCellCutExtension {
    ElementType element_type{ElementType::Unknown};
    int dimension{0};
    std::string name{};
    LevelSetCellCutFunction cutter{};
};

class LevelSetCellCutExtensionRegistry {
public:
    void registerCutter(LevelSetCellCutExtension extension);

    [[nodiscard]] bool hasCutter(ElementType element_type) const noexcept;

    [[nodiscard]] std::vector<ElementType> registeredElementTypes() const;

    [[nodiscard]] LevelSetCellCutResult cut(const CutInterfaceDomainRequest& request,
                                            const LevelSetCellCutInput& input) const;

private:
    std::unordered_map<std::uint8_t, LevelSetCellCutExtension> extensions_{};
};

[[nodiscard]] bool supportsLinearLevelSetCellCut2D(ElementType element_type) noexcept;

[[nodiscard]] bool supportsLinearLevelSetCellCut3D(ElementType element_type) noexcept;

[[nodiscard]] bool isLevelSetCellCutExtensionElement(ElementType element_type) noexcept;

[[nodiscard]] LevelSetCellCutResult cutLinearLevelSetCell2D(
    const CutInterfaceDomainRequest& request,
    const LevelSetCellCutInput& input);

[[nodiscard]] LevelSetCellCutResult cutLinearLevelSetCell3D(
    const CutInterfaceDomainRequest& request,
    const LevelSetCellCutInput& input);

void appendLinearLevelSetCellCut2D(LevelSetInterfaceDomain& domain,
                                   const LevelSetCellCutInput& input);

void appendLinearLevelSetCellCut3D(LevelSetInterfaceDomain& domain,
                                   const LevelSetCellCutInput& input);

} // namespace interfaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_INTERFACES_LEVELSETINTERFACEBUILDER_H
