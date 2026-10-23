/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FESYSTEMHANDLE_H
#define SVMP_FE_SYSTEMS_FESYSTEMHANDLE_H

/**
 * @file FESystemHandle.h
 * @brief The FE system of a run, as the components of a host hold it.
 */

#include <concepts>
#include <memory>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

class FESystem;

/**
 * @brief Refers to the owning pointer of a run's FE system.
 *
 * Converts implicitly from any object with an fe_system member of that type
 * (for example the simulation components of an application), so a function
 * taking an FESystemHandle accepts those components.  Valid while the
 * referenced pointer is.
 */
struct FESystemHandle {
    template <class Components>
        requires std::convertible_to<
            decltype((std::declval<Components&>().fe_system)),
            const std::unique_ptr<FESystem>&>
    FESystemHandle(Components& components) noexcept
        : fe_system(components.fe_system)
    {
    }

    explicit FESystemHandle(const std::unique_ptr<FESystem>& system) noexcept
        : fe_system(system)
    {
    }

    const std::unique_ptr<FESystem>& fe_system;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FESYSTEMHANDLE_H
