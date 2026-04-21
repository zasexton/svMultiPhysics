/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_CONSTRAINTSYNC_H
#define SVMP_FE_TIMESTEPPING_CONSTRAINTSYNC_H

#include "Constraints/AffineConstraints.h"
#include "TimeStepping/TimeHistory.h"

namespace svmp {
namespace FE {
namespace timestepping {
namespace detail {

inline void distributeConstraints(constraints::AffineConstraints const& constraints,
                                  backends::GenericVector& vec)
{
    if (constraints.empty()) {
        return;
    }

    constraints.distribute(vec);
}

inline void distributeConstraints(constraints::AffineConstraints const& constraints,
                                  TimeHistory& history)
{
    if (constraints.empty()) {
        return;
    }

    constraints.distribute(history.u());
    for (int k = 1; k <= history.historyDepth(); ++k) {
        constraints.distribute(history.uPrevK(k));
    }
}

inline void updateGhostsAndDistributeConstraints(constraints::AffineConstraints const& constraints,
                                                 backends::GenericVector& vec)
{
    vec.updateGhosts();
    distributeConstraints(constraints, vec);
}

inline void updateGhostsAndDistributeConstraints(constraints::AffineConstraints const& constraints,
                                                 TimeHistory& history)
{
    history.updateGhosts();
    distributeConstraints(constraints, history);
}

} // namespace detail
} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_CONSTRAINTSYNC_H
