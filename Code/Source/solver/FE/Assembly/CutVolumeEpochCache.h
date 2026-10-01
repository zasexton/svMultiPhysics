/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_CUTVOLUMEEPOCHCACHE_H
#define SVMP_FE_ASSEMBLY_CUTVOLUMEEPOCHCACHE_H

/**
 * @file CutVolumeEpochCache.h
 * @brief Reuse of cut-volume integration data across assemblies.
 *
 * While the geometry is frozen, every residual and Jacobian assembly visits
 * the same cut-volume rules on the same cells and spaces, so the transient
 * quadrature rule built from each cut rule and the scalar basis tabulations
 * at its points (values, reference and physical gradients and Hessians) are
 * the same numbers every time. This cache keeps them and gives back exactly
 * the arrays the uncached path computes, so assembled results are bitwise
 * unchanged.
 *
 * Keys are content, never object addresses:
 *  - each entry belongs to one (parent cell, marker, side) and is reused only
 *    while the rule's points, weights and order and the cell's type, geometry
 *    order and node coordinates are bitwise identical to what it was built
 *    from, so it also survives a new cut epoch for cells whose rule did not
 *    change;
 *  - each tabulation is keyed on the basis identity, the space's DOF layout,
 *    the role (test or trial) and whether Hessians were requested;
 *  - a change of the mesh geometry, topology, ownership, numbering or
 *    configuration revisions releases everything, and a new cut-context
 *    content signature (rule provenance, snapshot revision keys, measures)
 *    starts a new generation that releases entries left unused for a whole
 *    generation.
 *
 * The arrays are packed losslessly: an array whose bytes are all zero is not
 * stored, an array that is constant over the quadrature points of each DOF
 * (affine P1 gradients) keeps one value per DOF, and identical component
 * blocks of a product space are kept once. Every packing decision is made by
 * an exact bitwise comparison, never by tolerance.
 */

#include "Core/Types.h"
#include "Quadrature/QuadratureRule.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
namespace detail {

/// Index order of a packed per-DOF, per-quadrature-point array.
enum class CutVolumePackedLayout : std::uint8_t {
    DofMajor, ///< element (i, q) at i * n_qpts + q
    QptMajor  ///< element (i, q) at q * n_dofs + i
};

[[nodiscard]] inline bool cutVolumeBytesEqual(const void* a, const void* b, std::size_t bytes) noexcept
{
    return bytes == 0u || std::memcmp(a, b, bytes) == 0;
}

[[nodiscard]] inline bool cutVolumeAllBytesZero(const void* data, std::size_t bytes) noexcept
{
    const auto* p = static_cast<const unsigned char*>(data);
    std::size_t i = 0;
    std::uint64_t accumulated = 0;
    for (; i + 8u <= bytes; i += 8u) {
        std::uint64_t word = 0;
        std::memcpy(&word, p + i, 8u);
        accumulated |= word;
        if ((i & 0x3ffu) == 0u && accumulated != 0u) {
            return false;
        }
    }
    for (; i < bytes; ++i) {
        accumulated |= p[i];
    }
    return accumulated == 0u;
}

/**
 * @brief Lossless packed copy of an (n_dofs x n_qpts) array.
 *
 * unpack() reproduces the packed array bit for bit. Arrays whose size is not
 * n_dofs * n_qpts are kept verbatim.
 */
template <class T>
class CutVolumePackedArray {
public:
    void pack(std::span<const T> data,
              std::size_t n_dofs,
              std::size_t n_qpts,
              CutVolumePackedLayout layout,
              std::size_t n_scalar_dofs)
    {
        layout_ = layout;
        n_dofs_ = n_dofs;
        n_qpts_ = n_qpts;
        size_ = data.size();
        rows_ = n_dofs;
        replicated_ = false;
        data_.clear();
        if (data.empty()) {
            mode_ = Mode::Empty;
            data_.shrink_to_fit();
            return;
        }
        if (n_dofs == 0u || n_qpts == 0u || data.size() != n_dofs * n_qpts) {
            mode_ = Mode::Verbatim;
            data_.assign(data.begin(), data.end());
            data_.shrink_to_fit();
            return;
        }
        const T* d = data.data();
        constexpr std::size_t es = sizeof(T);
        const bool dof_major = (layout == CutVolumePackedLayout::DofMajor);

        if (cutVolumeAllBytesZero(d, size_ * es)) {
            mode_ = Mode::Zero;
            data_.shrink_to_fit();
            return;
        }

        // Product spaces evaluate component c of scalar DOF s as DOF
        // c * n_scalar_dofs + s. Every block repeats the first one exactly
        // when the array equals itself shifted by one block.
        if (n_scalar_dofs > 0u && n_scalar_dofs < n_dofs && n_dofs % n_scalar_dofs == 0u) {
            bool repeated = true;
            if (dof_major) {
                repeated = cutVolumeBytesEqual(d + n_scalar_dofs * n_qpts, d,
                                               (n_dofs - n_scalar_dofs) * n_qpts * es);
            } else {
                for (std::size_t q = 0; q < n_qpts && repeated; ++q) {
                    const T* block = d + q * n_dofs;
                    repeated = cutVolumeBytesEqual(block + n_scalar_dofs, block,
                                                   (n_dofs - n_scalar_dofs) * es);
                }
            }
            if (repeated) {
                replicated_ = true;
                rows_ = n_scalar_dofs;
            }
        }

        // Constant over the points of every DOF: again a one-step shift test.
        bool constant_per_dof = true;
        if (dof_major) {
            for (std::size_t r = 0; r < rows_ && constant_per_dof; ++r) {
                const T* row = d + r * n_qpts;
                constant_per_dof = cutVolumeBytesEqual(row + 1, row, (n_qpts - 1u) * es);
            }
        } else {
            constant_per_dof = cutVolumeBytesEqual(d + n_dofs, d, (n_qpts - 1u) * n_dofs * es);
        }

        if (constant_per_dof) {
            mode_ = Mode::DofConstant;
            data_.resize(rows_);
            for (std::size_t r = 0; r < rows_; ++r) {
                data_[r] = dof_major ? d[r * n_qpts] : d[r];
            }
        } else {
            mode_ = Mode::Full;
            if (!replicated_ || dof_major) {
                // Dof-major blocks are contiguous: the first rows_ rows.
                data_.assign(d, d + rows_ * n_qpts);
            } else {
                data_.resize(rows_ * n_qpts);
                for (std::size_t q = 0; q < n_qpts; ++q) {
                    std::memcpy(static_cast<void*>(data_.data() + q * rows_), d + q * n_dofs,
                                rows_ * es);
                }
            }
        }
        data_.shrink_to_fit();
    }

    /**
     * @brief Exact copy of the packed array.
     *
     * Returns a view of the stored data when it already is the full array, a
     * view of @p zeros (a buffer that only ever holds value-initialized, i.e.
     * all-zero, elements) for an all-zero array, and otherwise expands into
     * @p scratch and returns a view of it.
     */
    [[nodiscard]] std::span<const T> unpack(std::vector<T>& scratch,
                                            std::vector<T>& zeros) const
    {
        constexpr std::size_t es = sizeof(T);
        switch (mode_) {
            case Mode::Empty:
                return {};
            case Mode::Verbatim:
                return std::span<const T>(data_);
            case Mode::Zero:
                if (zeros.size() < size_) {
                    zeros.resize(size_);
                }
                return std::span<const T>(zeros.data(), size_);
            case Mode::Full:
                if (!replicated_) {
                    return std::span<const T>(data_);
                }
                break;
            case Mode::DofConstant:
                break;
        }
        scratch.resize(size_);
        T* out = scratch.data();
        if (layout_ == CutVolumePackedLayout::DofMajor) {
            for (std::size_t i = 0; i < n_dofs_; ++i) {
                const std::size_t r = i % rows_;
                if (mode_ == Mode::DofConstant) {
                    std::fill(out + i * n_qpts_, out + (i + 1u) * n_qpts_, data_[r]);
                } else {
                    std::memcpy(static_cast<void*>(out + i * n_qpts_), data_.data() + r * n_qpts_,
                                n_qpts_ * es);
                }
            }
        } else {
            for (std::size_t q = 0; q < n_qpts_; ++q) {
                const T* source = (mode_ == Mode::DofConstant) ? data_.data()
                                                               : data_.data() + q * rows_;
                for (std::size_t c = 0; c < n_dofs_; c += rows_) {
                    std::memcpy(static_cast<void*>(out + q * n_dofs_ + c), source, rows_ * es);
                }
            }
        }
        return std::span<const T>(scratch.data(), size_);
    }

    [[nodiscard]] std::size_t bytes() const noexcept
    {
        return data_.capacity() * sizeof(T);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    enum class Mode : std::uint8_t { Empty, Verbatim, Zero, DofConstant, Full };

    Mode mode_{Mode::Empty};
    CutVolumePackedLayout layout_{CutVolumePackedLayout::DofMajor};
    bool replicated_{false};
    std::size_t n_dofs_{0};
    std::size_t n_qpts_{0};
    std::size_t rows_{0};
    std::size_t size_{0};
    std::vector<T> data_{};
};

/**
 * @brief Basis arrays of one space in one role (test or trial) on one rule,
 *        in the exact layout passed to the AssemblyContext setters.
 */
struct CutVolumeEpochTabulation {
    using Vector3D = std::array<Real, 3>;
    using Matrix3x3 = std::array<std::array<Real, 3>, 3>;

    std::uint32_t space_key{0};
    CutVolumePackedArray<Real> values{};              ///< dof-major
    CutVolumePackedArray<Vector3D> ref_gradients{};   ///< dof-major
    CutVolumePackedArray<Vector3D> phys_gradients{};  ///< qpt-major
    CutVolumePackedArray<Matrix3x3> ref_hessians{};   ///< dof-major
    CutVolumePackedArray<Matrix3x3> phys_hessians{};  ///< dof-major

    [[nodiscard]] std::size_t bytes() const noexcept
    {
        return sizeof(*this) + values.bytes() + ref_gradients.bytes() +
               phys_gradients.bytes() + ref_hessians.bytes() + phys_hessians.bytes();
    }
};

/// Expansion buffers for one role when a packed tabulation is restored.
struct CutVolumeEpochUnpackScratch {
    std::vector<Real> values{};
    std::vector<CutVolumeEpochTabulation::Vector3D> ref_gradients{};
    std::vector<CutVolumeEpochTabulation::Vector3D> phys_gradients{};
    std::vector<CutVolumeEpochTabulation::Matrix3x3> ref_hessians{};
    std::vector<CutVolumeEpochTabulation::Matrix3x3> phys_hessians{};
    // Only ever grown by value-initialization, so they hold zeros only.
    std::vector<Real> zero_values{};
    std::vector<CutVolumeEpochTabulation::Vector3D> zero_vectors{};
    std::vector<CutVolumeEpochTabulation::Matrix3x3> zero_matrices{};
};

/**
 * @brief Cached data of one cut-volume rule: its transient quadrature rule,
 *        the cell it was validated against, and its basis tabulations.
 */
struct CutVolumeEpochRuleEntry {
    std::unique_ptr<const quadrature::QuadratureRule> rule{};
    GlobalIndex cell_id{-1};
    ElementType cell_type{ElementType::Unknown};
    int geometry_order{-1};
    std::vector<std::array<Real, 3>> cell_coords{};
    std::vector<CutVolumeEpochTabulation> tabulations{};
    std::size_t bytes{0};
    std::uint64_t last_generation{0};

    [[nodiscard]] const CutVolumeEpochTabulation* find(std::uint32_t space_key) const noexcept
    {
        for (const auto& tabulation : tabulations) {
            if (tabulation.space_key == space_key) {
                return &tabulation;
            }
        }
        return nullptr;
    }
};

/// Where a cut-volume rule lives: its parent cell, marker and side.
struct CutVolumeEpochRuleSlot {
    GlobalIndex cell{-1};
    int marker{-1};
    std::uint8_t side{0};

    [[nodiscard]] friend bool operator==(const CutVolumeEpochRuleSlot&,
                                         const CutVolumeEpochRuleSlot&) = default;
};

struct CutVolumeEpochRuleSlotHash {
    [[nodiscard]] std::size_t operator()(const CutVolumeEpochRuleSlot& slot) const noexcept
    {
        std::uint64_t h = static_cast<std::uint64_t>(slot.cell) * 0x9e3779b97f4a7c15ULL;
        h ^= (static_cast<std::uint64_t>(static_cast<std::uint32_t>(slot.marker)) << 8u) ^
             static_cast<std::uint64_t>(slot.side);
        h ^= h >> 31u;
        return static_cast<std::size_t>(h);
    }
};

/**
 * @brief Everything outside the rules themselves that the cache depends on.
 *
 * A change of any mesh field releases the whole cache; a change of the
 * content signature alone starts a new generation.
 */
struct CutVolumeEpochKey {
    std::uint64_t content_signature{0};
    std::uint64_t mesh_geometry_revision{0};
    std::uint64_t mesh_topology_revision{0};
    std::uint64_t mesh_ownership_revision{0};
    std::uint64_t mesh_numbering_revision{0};
    std::uint64_t mesh_active_configuration_epoch{0};
    std::uint64_t mesh_coordinate_configuration_key{0};
    GlobalIndex mesh_cell_count{0};
    int mesh_dimension{0};

    /// Bit mask of the mesh fields that differ (0: same mesh state).
    [[nodiscard]] std::uint32_t meshDifference(const CutVolumeEpochKey& other) const noexcept
    {
        std::uint32_t mask = 0u;
        mask |= (mesh_geometry_revision != other.mesh_geometry_revision) ? 0x1u : 0u;
        mask |= (mesh_topology_revision != other.mesh_topology_revision) ? 0x2u : 0u;
        mask |= (mesh_ownership_revision != other.mesh_ownership_revision) ? 0x4u : 0u;
        mask |= (mesh_numbering_revision != other.mesh_numbering_revision) ? 0x8u : 0u;
        mask |= (mesh_active_configuration_epoch != other.mesh_active_configuration_epoch) ? 0x10u : 0u;
        mask |= (mesh_coordinate_configuration_key != other.mesh_coordinate_configuration_key) ? 0x20u : 0u;
        mask |= (mesh_cell_count != other.mesh_cell_count) ? 0x40u : 0u;
        mask |= (mesh_dimension != other.mesh_dimension) ? 0x80u : 0u;
        return mask;
    }
};

/// Counters reported by StandardAssembler::cutVolumeEpochCacheDiagnostics().
struct CutVolumeEpochCacheStatistics {
    std::size_t resets{0};             ///< full releases (mesh change or first use)
    std::uint32_t reset_reasons{0};    ///< OR of reset causes: 0x1 geometry, 0x2 topology,
                                       ///< 0x4 ownership, 0x8 numbering, 0x10 active
                                       ///< configuration, 0x20 coordinate configuration,
                                       ///< 0x40 cell count, 0x80 dimension, 0x100 first use
                                       ///< or explicit invalidation
    std::size_t epochs{0};             ///< cut-context content generations started
    std::size_t stale_releases{0};     ///< entries released after a generation unused
    std::size_t rule_hits{0};          ///< rules whose cached quadrature rule was reused
    std::size_t rule_misses{0};        ///< rules whose quadrature rule was (re)built
    std::size_t rule_invalidations{0}; ///< cached data dropped because the content changed
    std::size_t basis_hits{0};         ///< term evaluations restored from the cache
    std::size_t basis_misses{0};       ///< term evaluations computed by prepareBasis
    std::size_t tabulations_stored{0};
    std::size_t over_budget_skips{0};  ///< data computed but not stored (memory budget)
    std::size_t bytes{0};              ///< current approximate footprint
    std::size_t peak_bytes{0};
    std::size_t max_bytes{0};          ///< budget (0: cache disabled)
    std::size_t entries{0};            ///< rules with a cached quadrature rule
    std::size_t tabulations{0};        ///< tabulations held
};

/**
 * @brief Per-assembler store of cut-volume rule entries.
 */
class CutVolumeEpochCache {
public:
    /// Approximate footprint of one map node beyond the entry's own data.
    static constexpr std::size_t kEntryOverheadBytes =
        sizeof(CutVolumeEpochRuleSlot) + sizeof(CutVolumeEpochRuleEntry) + 4u * sizeof(void*);

    void clear() noexcept
    {
        entries_.clear();
        space_keys_.clear();
        statistics_.bytes = 0;
        statistics_.entries = 0;
        statistics_.tabulations = 0;
        has_key_ = false;
    }

    /// Start using the cache for an assembly call; returns false when disabled.
    bool begin(const CutVolumeEpochKey& key, std::size_t max_bytes)
    {
        statistics_.max_bytes = max_bytes;
        if (max_bytes == 0u) {
            if (has_key_ || !entries_.empty()) {
                clear();
            }
            return false;
        }
        const std::uint32_t mesh_difference = has_key_ ? key.meshDifference(key_) : 0x100u;
        if (mesh_difference != 0u) {
            statistics_.reset_reasons |= mesh_difference;
            clear();
            key_ = key;
            has_key_ = true;
            ++generation_;
            ++statistics_.resets;
            ++statistics_.epochs;
        } else if (key.content_signature != key_.content_signature) {
            key_.content_signature = key.content_signature;
            ++generation_;
            ++statistics_.epochs;
            releaseStale();
        }
        return true;
    }

    /// Entry of a rule slot, created empty on first use (nullptr over budget).
    [[nodiscard]] CutVolumeEpochRuleEntry* entry(const CutVolumeEpochRuleSlot& slot)
    {
        auto found = entries_.find(slot);
        if (found == entries_.end()) {
            if (!fits(kEntryOverheadBytes)) {
                return nullptr;
            }
            found = entries_.emplace(slot, CutVolumeEpochRuleEntry{}).first;
            addBytes(found->second, kEntryOverheadBytes);
        }
        found->second.last_generation = generation_;
        return &found->second;
    }

    [[nodiscard]] std::uint32_t internSpaceKey(const std::string& key)
    {
        const auto found = space_keys_.find(key);
        if (found != space_keys_.end()) {
            return found->second;
        }
        const auto id = static_cast<std::uint32_t>(space_keys_.size() + 1u);
        space_keys_.emplace(key, id);
        return id;
    }

    [[nodiscard]] bool fits(std::size_t extra_bytes) const noexcept
    {
        return statistics_.bytes + extra_bytes <= statistics_.max_bytes;
    }

    void addBytes(CutVolumeEpochRuleEntry& entry, std::size_t bytes) noexcept
    {
        entry.bytes += bytes;
        statistics_.bytes += bytes;
        if (statistics_.bytes > statistics_.peak_bytes) {
            statistics_.peak_bytes = statistics_.bytes;
        }
    }

    void removeBytes(CutVolumeEpochRuleEntry& entry, std::size_t bytes) noexcept
    {
        const auto removed = std::min(entry.bytes, bytes);
        entry.bytes -= removed;
        statistics_.bytes -= std::min(statistics_.bytes, removed);
    }

    void addTabulation(CutVolumeEpochRuleEntry& entry, CutVolumeEpochTabulation tabulation)
    {
        const auto bytes = tabulation.bytes();
        entry.tabulations.push_back(std::move(tabulation));
        addBytes(entry, bytes);
        ++statistics_.tabulations;
        ++statistics_.tabulations_stored;
    }

    void dropTabulations(CutVolumeEpochRuleEntry& entry) noexcept
    {
        std::size_t bytes = 0;
        for (const auto& tabulation : entry.tabulations) {
            bytes += tabulation.bytes();
        }
        statistics_.tabulations -= std::min(statistics_.tabulations, entry.tabulations.size());
        entry.tabulations.clear();
        entry.tabulations.shrink_to_fit();
        removeBytes(entry, bytes);
    }

    /// Drop everything an entry holds except its slot.
    void resetEntry(CutVolumeEpochRuleEntry& entry) noexcept
    {
        dropTabulations(entry);
        if (entry.rule) {
            statistics_.entries -= std::min<std::size_t>(statistics_.entries, 1u);
        }
        const auto generation = entry.last_generation;
        removeBytes(entry, entry.bytes > kEntryOverheadBytes ? entry.bytes - kEntryOverheadBytes : 0u);
        entry.rule.reset();
        entry.cell_id = -1;
        entry.cell_type = ElementType::Unknown;
        entry.geometry_order = -1;
        entry.cell_coords.clear();
        entry.cell_coords.shrink_to_fit();
        entry.last_generation = generation;
    }

    [[nodiscard]] CutVolumeEpochCacheStatistics& statistics() noexcept { return statistics_; }
    [[nodiscard]] const CutVolumeEpochCacheStatistics& statistics() const noexcept
    {
        return statistics_;
    }

private:
    void releaseStale() noexcept
    {
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->second.last_generation + 1u < generation_) {
                auto& entry = it->second;
                statistics_.bytes -= std::min(statistics_.bytes, entry.bytes);
                if (entry.rule) {
                    statistics_.entries -= std::min<std::size_t>(statistics_.entries, 1u);
                }
                statistics_.tabulations -=
                    std::min(statistics_.tabulations, entry.tabulations.size());
                ++statistics_.stale_releases;
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::unordered_map<CutVolumeEpochRuleSlot, CutVolumeEpochRuleEntry, CutVolumeEpochRuleSlotHash>
        entries_{};
    std::unordered_map<std::string, std::uint32_t> space_keys_{};
    CutVolumeEpochKey key_{};
    bool has_key_{false};
    std::uint64_t generation_{0};
    CutVolumeEpochCacheStatistics statistics_{};
};

} // namespace detail
} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_CUTVOLUMEEPOCHCACHE_H
