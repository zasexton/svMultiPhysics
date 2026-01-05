/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_GLOBALKERNEL_H
#define SVMP_FE_SYSTEMS_GLOBALKERNEL_H

#include "Core/Types.h"
#include "Core/ParameterValue.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace svmp {
namespace FE {

namespace sparsity {
class SparsityPattern;
}

namespace assembly {
struct AssemblyResult;
class GlobalSystemView;
}

namespace systems {

class FESystem;
struct AssemblyRequest;
struct SystemStateView;

/**
 * @brief Optional persistent state storage for global kernels
 *
 * Global kernels may request stable, per-entity state storage with old/work
 * double-buffering (similar to per-integration-point MaterialState for element
 * kernels). This is primarily intended for history-dependent global terms such
 * as contact/friction, where state is naturally associated with boundary faces.
 */
struct GlobalStateSpec {
    enum class Domain : std::uint8_t {
        None,
        Cell,
        BoundaryFace,
        InteriorFace,
    };

    Domain domain{Domain::None};
    std::size_t bytes_per_qpt{0};
    std::size_t alignment{alignof(std::max_align_t)};
    LocalIndex max_qpts{0};

    [[nodiscard]] bool empty() const noexcept { return domain == Domain::None || bytes_per_qpt == 0u; }
};

/**
 * @brief Systems-level kernel for non-element-local operator contributions
 *
 * GlobalKernel enables operator terms that are not naturally expressed as a
 * cell/boundary/interior-face loop (e.g., contact, search-driven constraints,
 * or other globally coupled evaluations). These kernels are invoked during
 * `systems::assembleOperator` after the standard Assembly loops and before
 * finalization/communication.
 */
class GlobalKernel {
public:
    virtual ~GlobalKernel() = default;
    [[nodiscard]] virtual std::string name() const { return "GlobalKernel"; }

    /**
     * @brief Optional sparsity augmentation hook
     *
     * Global terms are not represented in the element/face loops used by the
     * default sparsity builder. Kernels may override this to conservatively add
     * couplings required for their global insertions.
     */
    virtual void addSparsityCouplings(const FESystem& /*system*/,
                                      sparsity::SparsityPattern& /*pattern*/) const
    {
    }

    /**
     * @brief Optional state requirement for this global kernel
     */
    [[nodiscard]] virtual GlobalStateSpec globalStateSpec() const noexcept { return {}; }

    /**
     * @brief Optional parameter requirements for this global kernel
     */
    [[nodiscard]] virtual std::vector<params::Spec> parameterSpecs() const { return {}; }

    [[nodiscard]] virtual assembly::AssemblyResult assemble(const FESystem& system,
                                                            const AssemblyRequest& request,
                                                            const SystemStateView& state,
                                                            assembly::GlobalSystemView* matrix_out,
                                                            assembly::GlobalSystemView* vector_out) = 0;
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_GLOBALKERNEL_H
