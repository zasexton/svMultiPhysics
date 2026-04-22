#ifndef SVMP_FE_AUXILIARY_ROW_OWNERSHIP_H
#define SVMP_FE_AUXILIARY_ROW_OWNERSHIP_H

/**
 * @file AuxiliaryRowOwnership.h
 * @brief Scope-aware owner-map helpers for monolithic auxiliary rows.
 *
 * These helpers produce concrete owner ranks for auxiliary unknown rows.  They
 * are intentionally independent of assembly so distributed sparsity and numeric
 * insertion can consume the same owner map.
 */

#include "Core/Types.h"

#include "Auxiliary/AuxiliaryStateTypes.h"
#include "Backends/Utils/BackendOptions.h"

#include <cstddef>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

struct AuxiliaryRowOwnershipSpec {
    AuxiliaryStateScope scope{AuxiliaryStateScope::Global};
    backends::MixedRowOwnershipPolicy policy{
        backends::MixedRowOwnershipPolicy::Unspecified};
    std::size_t entity_count{0};
    int stride{0};
    int single_owner_rank{-1};

    /// Owner rank per auxiliary entity. Required for non-single-owner policies.
    std::span<const int> entity_owner_ranks{};
};

/**
 * @brief Expand entity owners to one owner per auxiliary row/component.
 */
[[nodiscard]] std::vector<int>
buildAuxiliaryRowOwnerRanks(const AuxiliaryRowOwnershipSpec& spec);

/**
 * @brief Expand cell owners through QP offsets to one owner per QP row.
 */
[[nodiscard]] std::vector<int>
buildQuadraturePointRowOwnerRanks(std::span<const int> cell_owner_ranks,
                                  std::span<const std::size_t> qp_offsets,
                                  int stride);

/**
 * @brief Derive region entity owners from globally ordered cell metadata.
 *
 * The first cell encountered for each region is treated as the globally lowest
 * cell for that region.  MPI callers should provide globally-minloc-reduced
 * cell ordering/owners before using this helper for distributed ownership.
 */
[[nodiscard]] std::vector<int>
buildRegionEntityOwnerRanksFromCells(std::span<const int> cell_region_ids,
                                     std::span<const int> cell_owner_ranks,
                                     std::size_t n_regions);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_ROW_OWNERSHIP_H
