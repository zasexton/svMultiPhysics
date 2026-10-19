/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_LEVELSET_LEVELSETMAINTENANCESUBSTAGE_H
#define SVMP_FE_LEVELSET_LEVELSETMAINTENANCESUBSTAGE_H

/**
 * @file LevelSetMaintenanceSubstage.h
 * @brief Substages of level-set maintenance reported to work observers.
 */

#include <cstdint>

namespace svmp {
namespace FE {
namespace level_set {

/**
 * @brief The maintenance substage that produced a change of the level set.
 *
 * Observers (for example an energy ledger) receive one of these with every
 * staged change.  The enumerator values are part of canonical consensus
 * words and must not be reordered.
 */
enum class MaintenanceSubstage : std::uint8_t {
    Transport,
    Limiting,
    Reinitialization,
    GeometryReconciliation,
    GlobalCorrection
};

} // namespace level_set
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_LEVELSET_LEVELSETMAINTENANCESUBSTAGE_H
