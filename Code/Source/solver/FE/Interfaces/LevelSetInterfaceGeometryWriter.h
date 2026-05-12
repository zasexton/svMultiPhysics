/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_LEVELSETINTERFACEGEOMETRYWRITER_H
#define SVMP_FE_INTERFACES_LEVELSETINTERFACEGEOMETRYWRITER_H

/**
 * @file LevelSetInterfaceGeometryWriter.h
 * @brief Lightweight geometry output for generated level-set interface domains.
 */

#include "Interfaces/LevelSetInterfaceDomain.h"

#include <iosfwd>
#include <string>

namespace svmp {
namespace FE {
namespace interfaces {

void writeLevelSetInterfaceGeometryVtp(const LevelSetInterfaceDomain& domain,
                                       std::ostream& out);

[[nodiscard]] std::string levelSetInterfaceGeometryVtpString(
    const LevelSetInterfaceDomain& domain);

} // namespace interfaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_INTERFACES_LEVELSETINTERFACEGEOMETRYWRITER_H
