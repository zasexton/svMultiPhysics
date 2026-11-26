/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_SPACECOMPATIBILITY_H
#define SVMP_FE_SPACES_SPACECOMPATIBILITY_H

/**
 * @file SpaceCompatibility.h
 * @brief Utilities for checking compatibility between function spaces
 */

#include "Spaces/FunctionSpace.h"
#include <string>

namespace svmp {
namespace FE {
namespace spaces {

class SpaceCompatibility {
public:
    struct Result {
        bool ok{false};
        std::string message;
    };

    /// Check basic conformity between two spaces (element type, continuity, etc.)
    static Result check_conformity(const FunctionSpace& a,
                                   const FunctionSpace& b);

    /// Heuristic inf-sup stability check for vector-field/multiplier pairs
    static Result check_inf_sup(const FunctionSpace& vector_field_space,
                                const FunctionSpace& multiplier_space);
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_SPACECOMPATIBILITY_H

