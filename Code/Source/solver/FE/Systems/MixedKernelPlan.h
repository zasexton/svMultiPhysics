#ifndef SVMP_FE_SYSTEMS_MIXEDKERNELPLAN_H
#define SVMP_FE_SYSTEMS_MIXEDKERNELPLAN_H

/**
 * @file MixedKernelPlan.h
 * @brief Explicit lowering plan for mixed FE kernels
 */

#include "Core/Types.h"

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

enum class MixedKernelSemanticType : std::uint8_t {
    MixedBlockSet = 0u,
    MonolithicCell
};

[[nodiscard]] inline const char* mixedKernelSemanticTypeName(MixedKernelSemanticType type) noexcept
{
    switch (type) {
        case MixedKernelSemanticType::MixedBlockSet:
            return "MixedBlockSet";
        case MixedKernelSemanticType::MonolithicCell:
            return "MonolithicCell";
    }
    return "Unknown";
}

struct MixedKernelPlanBlock {
    FieldId test_field{INVALID_FIELD_ID};
    FieldId trial_field{INVALID_FIELD_ID};
    FieldId residual_owner_field{INVALID_FIELD_ID};
    bool has_cell{false};
    bool has_boundary{false};
    bool has_interior{false};
    bool has_interface{false};
    bool want_matrix{false};
    bool want_vector{false};
};

struct MixedKernelPlan {
    MixedKernelSemanticType semantic_type{MixedKernelSemanticType::MixedBlockSet};
    bool jit_requested{false};
    bool monolithic_cell_requested{false};
    bool monolithic_cell_enabled{false};
    std::vector<MixedKernelPlanBlock> blocks{};

    [[nodiscard]] bool usesMonolithicCellKernel() const noexcept
    {
        return semantic_type == MixedKernelSemanticType::MonolithicCell && monolithic_cell_enabled;
    }

    [[nodiscard]] std::size_t numCellBlocks() const noexcept
    {
        std::size_t count = 0;
        for (const auto& block : blocks) {
            if (block.has_cell) {
                ++count;
            }
        }
        return count;
    }

    [[nodiscard]] std::string describe() const
    {
        std::ostringstream oss;
        oss << "semantic=" << mixedKernelSemanticTypeName(semantic_type)
            << " jit_requested=" << (jit_requested ? 1 : 0)
            << " monolithic_requested=" << (monolithic_cell_requested ? 1 : 0)
            << " monolithic_enabled=" << (monolithic_cell_enabled ? 1 : 0)
            << " blocks=" << blocks.size()
            << " cell_blocks=" << numCellBlocks();
        return oss.str();
    }
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_MIXEDKERNELPLAN_H
