/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_SYSTEMSEXCEPTIONS_H
#define SVMP_FE_SYSTEMS_SYSTEMSEXCEPTIONS_H

#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace systems {

class SystemsException : public FEException {
public:
    SystemsException(const std::string& message,
                     const char* file = "",
                     int line = 0,
                     const char* function = "")
        : FEException(message, file, line, function) {}
};

class InvalidStateException : public FEException {
public:
    InvalidStateException(const std::string& message,
                          const char* file = "",
                          int line = 0,
                          const char* function = "")
        : FEException(message, file, line, function) {}
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_SYSTEMSEXCEPTIONS_H
