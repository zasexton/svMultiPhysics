/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_FORM_HELPERS_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_FORM_HELPERS_H

/**
 * @file NavierStokesFormHelpers.h
 * @brief Small helpers of the Navier-Stokes formulation (the form helpers are
 *        FE/Forms constantVector, sumOf and bc::toVectorExpr)
 */
#include <algorithm>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

template <class T>
void appendUnique(std::vector<T>& values, const T& value)
{
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_NAVIER_STOKES_FORM_HELPERS_H
