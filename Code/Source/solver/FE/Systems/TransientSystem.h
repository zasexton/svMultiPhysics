/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_TRANSIENT_SYSTEM_H
#define SVMP_FE_SYSTEMS_TRANSIENT_SYSTEM_H

/**
 * @file TransientSystem.h
 * @brief Systems-level transient orchestration for forms containing symbolic `dt(·,k)`
 *
 * This class is intentionally lightweight: it does not implement a full time loop.
 * It exists to enforce the design boundary:
 * - FE/Forms: symbolic dt() only
 * - FE/Systems: chooses an integrator, validates derivative orders, manages history vectors,
 *               and supplies an assembly-time context so dt() can be lowered.
 */

#include "Systems/FESystem.h"
#include "Systems/TimeIntegrator.h"

#include <memory>

namespace svmp {
namespace FE {
namespace systems {

class TransientSystem {
public:
    TransientSystem(FESystem& system, std::shared_ptr<const TimeIntegrator> integrator);

    [[nodiscard]] const FESystem& system() const noexcept { return system_; }
    [[nodiscard]] FESystem& system() noexcept { return system_; }
    [[nodiscard]] const TimeIntegrator& integrator() const;

    /**
     * @brief Assemble an operator in a transient time-integration context
     *
     * This method:
     * 1) detects dt orders via registered kernels (system.temporalOrder()),
     * 2) validates integrator compatibility and history availability,
     * 3) builds an `assembly::TimeIntegrationContext`,
     * 4) calls into FESystem assembly with that context attached.
     */
    assembly::AssemblyResult assemble(
        const AssemblyRequest& req,
        const SystemStateView& state,
        assembly::GlobalSystemView* matrix_out,
        assembly::GlobalSystemView* vector_out);

private:
    void validateState(const SystemStateView& state, int required_history_states) const;

    FESystem& system_;
    std::shared_ptr<const TimeIntegrator> integrator_;
    assembly::TimeIntegrationContext ctx_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_TRANSIENT_SYSTEM_H
