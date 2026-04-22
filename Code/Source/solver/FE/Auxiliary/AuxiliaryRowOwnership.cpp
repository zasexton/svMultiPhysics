#include "Auxiliary/AuxiliaryRowOwnership.h"

#include "Core/FEException.h"

#include <algorithm>
#include <string>
#include <string_view>

namespace svmp {
namespace FE {
namespace systems {

namespace {

void validateStride(int stride, std::string_view context)
{
    FE_THROW_IF(stride <= 0, InvalidArgumentException,
                std::string(context) + ": stride must be > 0");
}

void validateOwnerRank(int owner, std::string_view context)
{
    FE_THROW_IF(owner < 0, InvalidArgumentException,
                std::string(context) + ": owner rank must be >= 0");
}

[[nodiscard]] bool isSingleOwnerPolicy(backends::MixedRowOwnershipPolicy policy) noexcept
{
    return policy == backends::MixedRowOwnershipPolicy::SingleOwner;
}

} // namespace

std::vector<int>
buildAuxiliaryRowOwnerRanks(const AuxiliaryRowOwnershipSpec& spec)
{
    validateStride(spec.stride, "buildAuxiliaryRowOwnerRanks");

    const auto row_count =
        spec.entity_count * static_cast<std::size_t>(spec.stride);
    std::vector<int> owners;
    owners.reserve(row_count);

    if (isSingleOwnerPolicy(spec.policy)) {
        validateOwnerRank(spec.single_owner_rank, "buildAuxiliaryRowOwnerRanks");
        owners.assign(row_count, spec.single_owner_rank);
        return owners;
    }

    FE_THROW_IF(spec.policy == backends::MixedRowOwnershipPolicy::Unspecified,
                InvalidArgumentException,
                "buildAuxiliaryRowOwnerRanks: ownership policy is unspecified");
    FE_THROW_IF(spec.entity_owner_ranks.size() != spec.entity_count,
                InvalidArgumentException,
                "buildAuxiliaryRowOwnerRanks: entity owner count must match entity_count");

    for (const int owner : spec.entity_owner_ranks) {
        validateOwnerRank(owner, "buildAuxiliaryRowOwnerRanks");
        for (int c = 0; c < spec.stride; ++c) {
            owners.push_back(owner);
        }
    }
    return owners;
}

std::vector<int>
buildQuadraturePointRowOwnerRanks(std::span<const int> cell_owner_ranks,
                                  std::span<const std::size_t> qp_offsets,
                                  int stride)
{
    validateStride(stride, "buildQuadraturePointRowOwnerRanks");
    FE_THROW_IF(qp_offsets.size() != cell_owner_ranks.size() + 1u,
                InvalidArgumentException,
                "buildQuadraturePointRowOwnerRanks: qp_offsets size must be n_cells + 1");
    FE_THROW_IF(!qp_offsets.empty() && qp_offsets.front() != 0u,
                InvalidArgumentException,
                "buildQuadraturePointRowOwnerRanks: qp_offsets[0] must be 0");
    FE_THROW_IF(!std::is_sorted(qp_offsets.begin(), qp_offsets.end()),
                InvalidArgumentException,
                "buildQuadraturePointRowOwnerRanks: qp_offsets must be nondecreasing");

    std::vector<int> owners;
    owners.reserve(qp_offsets.empty()
                       ? 0u
                       : qp_offsets.back() * static_cast<std::size_t>(stride));

    for (std::size_t cell = 0; cell < cell_owner_ranks.size(); ++cell) {
        const int owner = cell_owner_ranks[cell];
        validateOwnerRank(owner, "buildQuadraturePointRowOwnerRanks");
        const auto n_qp = qp_offsets[cell + 1u] - qp_offsets[cell];
        for (std::size_t qp = 0; qp < n_qp; ++qp) {
            for (int c = 0; c < stride; ++c) {
                owners.push_back(owner);
            }
        }
    }
    return owners;
}

std::vector<int>
buildRegionEntityOwnerRanksFromCells(std::span<const int> cell_region_ids,
                                     std::span<const int> cell_owner_ranks,
                                     std::size_t n_regions)
{
    FE_THROW_IF(cell_region_ids.size() != cell_owner_ranks.size(),
                InvalidArgumentException,
                "buildRegionEntityOwnerRanksFromCells: cell metadata sizes must match");

    std::vector<int> region_owners(n_regions, -1);
    for (std::size_t cell = 0; cell < cell_region_ids.size(); ++cell) {
        const int region = cell_region_ids[cell];
        if (region < 0) {
            continue;
        }
        FE_THROW_IF(static_cast<std::size_t>(region) >= n_regions,
                    InvalidArgumentException,
                    "buildRegionEntityOwnerRanksFromCells: region id is out of range");
        const int owner = cell_owner_ranks[cell];
        validateOwnerRank(owner, "buildRegionEntityOwnerRanksFromCells");
        auto& region_owner = region_owners[static_cast<std::size_t>(region)];
        if (region_owner < 0) {
            region_owner = owner;
        }
    }

    for (int owner : region_owners) {
        validateOwnerRank(owner, "buildRegionEntityOwnerRanksFromCells");
    }
    return region_owners;
}

} // namespace systems
} // namespace FE
} // namespace svmp
