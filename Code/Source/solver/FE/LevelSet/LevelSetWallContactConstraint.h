/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file LevelSetWallContactConstraint.h
 * @ingroup fe_level_set
 * @brief Identity of one accepted wall-contact cell of a level-set interface.
 */

#include "Core/Types.h"

#include <cstdint>

namespace svmp {
namespace FE {
namespace level_set {

/**
 * @brief One parent cell of an accepted contact rule on a contact wall.
 *
 * The record names the accepted contact geometry and nothing else: the
 * contact angle has one owner, the momentum-side Young term (decision D4 in
 * Documentation/free_surface_program_tracker.md), so no target angle or
 * physical frame is carried.  Maintenance that must not move the accepted
 * contact line (the conservative-phase geometry reconciliation) leaves the
 * coefficients of these cells unchanged.
 *
 * The parent identity is partition independent.  A caller may hold only the
 * locally owned records; consumers form one sorted, duplicate-free
 * communicator snapshot before using them.
 */
struct LevelSetWallContactConstraint {
    int interface_marker{-1};
    int boundary_marker{-1};
    GlobalIndex parent_cell_global_id{INVALID_GLOBAL_INDEX};
    std::uint64_t geometry_revision{0u};
};

} // namespace level_set
} // namespace FE
} // namespace svmp
