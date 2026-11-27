/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved. See Copyright-SimVascular.txt for details.
 */

#pragma once

#include "FE/LevelSet/LevelSetConservativePhaseRegions.h"

#include <cctype>
#include <stdexcept>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

/// Read fixed free-surface observation regions, including the original morphology aliases.
[[nodiscard]] inline std::vector<FE::level_set::LevelSetPhaseRegionBox>
parsePhaseRegionBoxes(std::string_view boxes)
{
    return FE::level_set::parseLevelSetPhaseRegionBoxes(boxes, [](std::string_view kind) -> std::string {
        std::string key;
        for (unsigned char c : kind) {
            if (c != '_' && c != '-' && !std::isspace(c)) {
                key.push_back(static_cast<char>(std::tolower(c)));
            }
        }
        if (key == "wallfilm") return "wall_film";
        if (key == "satellite" || key == "resolvedsatellite") return "resolved_satellite";
        if (key == "sheet" || key == "rim" || key == "observer") return key;
        throw std::invalid_argument("unknown conservative phase region kind '" + std::string(kind) + "'");
    });
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
