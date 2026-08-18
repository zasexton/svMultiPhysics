/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Materials/Solid/IsochoricNeoHookeanPK1.h"

namespace svmp {
namespace Physics {
namespace materials {
namespace solid {

IsochoricNeoHookeanPK1::IsochoricNeoHookeanPK1(FE::Real mu)
    : mu_(mu)
{
    if (mu_ <= 0.0) {
        throw std::invalid_argument("IsochoricNeoHookeanPK1: mu must be > 0");
    }
}

} // namespace solid
} // namespace materials
} // namespace Physics
} // namespace svmp
