#ifndef SVMP_FE_AUXILIARY_STATE_H
#define SVMP_FE_AUXILIARY_STATE_H

/**
 * @file AuxiliaryState.h
 * @brief Generalized auxiliary state container for the FE library
 *
 * This module provides the core runtime container for auxiliary (non-PDE)
 * state variables managed by the FE library.  Auxiliary state is
 * FE-library infrastructure — not a boundary-condition feature and not a
 * physics-specific concept.  Boundary functionals, EP-like ionic models,
 * metabolism models, reduced models, and future coupled subsystems all
 * use the same neutral AuxiliaryState infrastructure.
 *
 * ## Public API surface
 *
 * - `AuxiliaryStateTypes.h`: Core enums and specification structs.
 * - `AuxiliaryStateStorage.h`: Per-block storage with committed/work/history.
 * - `AuxiliaryHistoryBuffer.h`: Time-stamped history snapshots.
 * - `AuxiliaryStateIndexing.h`: Scope-specific entity indexing.
 * - `AuxiliaryState`: Mutable runtime container with both a legacy flat
 *   API (for backward compatibility) and a block-based API supporting
 *   multiple blocks with distinct scopes.
 * - `AuxiliaryStateRegistration`: Legacy registration record for ODE-based
 *   auxiliary state used by the coupled-boundary path.
 *
 * See `AuxiliaryStateTypes.h` for the full vocabulary of scopes, solve
 * modes, derivative policies, and layout options.
 */

#include "Core/Types.h"
#include "Core/Alignment.h"
#include "Core/AlignedAllocator.h"
#include "Core/FEException.h"

#include "Auxiliary/AuxiliaryStateTypes.h"
#include "Auxiliary/AuxiliaryStateStorage.h"
#include "Auxiliary/AuxiliaryStateIndexing.h"
#include "Systems/ODEIntegrator.h"

#include "Forms/BoundaryFunctional.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/**
 * @brief Mutable runtime container for named auxiliary state variables.
 *
 * Provides two complementary access modes:
 *
 * ## Legacy flat API (backward-compatible)
 *
 * - `values()` — flattened work buffer across all legacy-registered vars.
 * - `previous()` / `previous(k)` — committed / history access.
 * - `operator[]` — named scalar access.
 * - `registerState()` — register into the flat buffer.
 *
 * ## Block-based API (generalized)
 *
 * - `registerBlock()` — register a typed block with a specific scope.
 * - `getBlock()` / `hasBlock()` — block lookup by name.
 * - Each block is an `AuxiliaryBlockStorage` with its own work/committed
 *   buffers, history, and scope-specific indexing.
 * - Blocks may have different scopes in the same `AuxiliaryState`.
 *
 * Block names are the durable public handles.  Numeric indices are
 * internal, setup-stable identifiers.
 */
class AuxiliaryState {
public:
    AuxiliaryState() = default;

    // =================================================================
    //  Legacy flat API (backward-compatible)
    // =================================================================

    [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

    [[nodiscard]] std::span<Real> values() noexcept { return values_; }
    [[nodiscard]] std::span<const Real> values() const noexcept { return values_; }

    [[nodiscard]] std::span<const Real> previous() const noexcept { return committed_; }

    [[nodiscard]] std::span<const Real> previous(int steps_back) const
    {
        FE_THROW_IF(steps_back <= 0, InvalidArgumentException,
                    "AuxiliaryState::previous(k): k must be >= 1");
        if (steps_back == 1) {
            return committed_;
        }
        const auto idx = static_cast<std::size_t>(steps_back - 2);
        FE_THROW_IF(idx >= flat_history_.size(), InvalidArgumentException,
                    "AuxiliaryState::previous(k): insufficient history");
        return flat_history_[idx];
    }

    [[nodiscard]] bool has(std::string_view name) const noexcept
    {
        return name_to_index_.find(std::string(name)) != name_to_index_.end();
    }

    [[nodiscard]] std::optional<std::size_t> tryIndexOf(std::string_view name) const noexcept
    {
        auto it = name_to_index_.find(std::string(name));
        if (it == name_to_index_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    [[nodiscard]] std::size_t indexOf(std::string_view name) const
    {
        auto it = name_to_index_.find(std::string(name));
        FE_THROW_IF(it == name_to_index_.end(), InvalidArgumentException,
                    "AuxiliaryState: unknown variable '" + std::string(name) + "'");
        return it->second;
    }

    [[nodiscard]] bool hasHistory(int steps_back) const noexcept
    {
        if (steps_back <= 1) {
            return true;
        }
        const auto idx = static_cast<std::size_t>(steps_back - 2);
        return idx < flat_history_.size();
    }

    [[nodiscard]] Real previousValue(std::string_view name, int steps_back) const
    {
        auto it = name_to_index_.find(std::string(name));
        FE_THROW_IF(it == name_to_index_.end(), InvalidArgumentException,
                    "AuxiliaryState: unknown variable '" + std::string(name) + "'");
        const auto vec = previous(steps_back);
        FE_THROW_IF(it->second >= vec.size(), InvalidArgumentException,
                    "AuxiliaryState::previousValue: internal index out of range");
        return vec[it->second];
    }

    [[nodiscard]] Real& operator[](std::string_view name)
    {
        auto it = name_to_index_.find(std::string(name));
        FE_THROW_IF(it == name_to_index_.end(), InvalidArgumentException,
                    "AuxiliaryState: unknown variable '" + std::string(name) + "'");
        return values_.at(it->second);
    }

    [[nodiscard]] Real operator[](std::string_view name) const
    {
        auto it = name_to_index_.find(std::string(name));
        FE_THROW_IF(it == name_to_index_.end(), InvalidArgumentException,
                    "AuxiliaryState: unknown variable '" + std::string(name) + "'");
        return values_.at(it->second);
    }

    void clear()
    {
        committed_.clear();
        values_.clear();
        flat_history_.clear();
        name_to_index_.clear();
        blocks_.clear();
        block_name_to_index_.clear();
    }

    /**
     * @brief Register into the legacy flat buffer.
     *
     * For `size==1`, the variable name is `spec.name` unless
     * `component_names` is provided.  For `size>1`, `component_names`
     * (if provided) must have `size` entries; otherwise names are
     * generated as `name[i]`.
     */
    void registerState(const AuxiliaryStateSpec& spec, std::span<const Real> initial_values = {})
    {
        FE_THROW_IF(spec.size <= 0, InvalidArgumentException,
                    "AuxiliaryState::registerState: spec.size must be > 0");
        FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                    "AuxiliaryState::registerState: spec.name is empty");
        if (!spec.component_names.empty()) {
            FE_THROW_IF(static_cast<int>(spec.component_names.size()) != spec.size, InvalidArgumentException,
                        "AuxiliaryState::registerState: component_names.size() must equal spec.size");
        }
        if (!initial_values.empty()) {
            FE_THROW_IF(initial_values.size() != static_cast<std::size_t>(spec.size), InvalidArgumentException,
                        "AuxiliaryState::registerState: initial_values size mismatch");
        }

        std::vector<std::string> names;
        names.reserve(static_cast<std::size_t>(spec.size));
        if (!spec.component_names.empty()) {
            for (const auto& nm : spec.component_names) {
                FE_THROW_IF(nm.empty(), InvalidArgumentException,
                            "AuxiliaryState::registerState: empty component name");
                names.push_back(nm);
            }
        } else if (spec.size == 1) {
            names.push_back(spec.name);
        } else {
            for (int i = 0; i < spec.size; ++i) {
                names.push_back(spec.name + "[" + std::to_string(i) + "]");
            }
        }

        for (const auto& nm : names) {
            FE_THROW_IF(name_to_index_.count(nm) != 0u, InvalidArgumentException,
                        "AuxiliaryState::registerState: duplicate variable '" + nm + "'");
        }

        const auto offset = values_.size();
        values_.resize(offset + static_cast<std::size_t>(spec.size), 0.0);
        committed_.resize(values_.size(), 0.0);

        for (int i = 0; i < spec.size; ++i) {
            const auto idx = offset + static_cast<std::size_t>(i);
            const Real v0 = initial_values.empty() ? Real{0.0} : initial_values[static_cast<std::size_t>(i)];
            values_[idx] = v0;
            committed_[idx] = v0;
            name_to_index_.emplace(names[static_cast<std::size_t>(i)], idx);
        }
    }

    void resetToCommitted()
    {
        values_ = committed_;
    }

    void commitTimeStep(std::size_t max_history = 8)
    {
        if (values_.empty()) return;
        if (!committed_.empty()) {
            flat_history_.insert(flat_history_.begin(), committed_);
            if (flat_history_.size() > max_history) {
                flat_history_.resize(max_history);
            }
        }
        committed_ = values_;
    }

    // =================================================================
    //  Block-based API (generalized)
    // =================================================================

    /**
     * @brief Register a new typed auxiliary block.
     *
     * @param spec         Block specification.
     * @param entity_count Number of entities (1 for Global, mesh count for others).
     * @param initial_values Optional initial values (must be entity_count * spec.size).
     *
     * @return Index of the newly registered block.
     */
    std::size_t registerBlock(const AuxiliaryStateSpec& spec,
                              std::size_t entity_count,
                              std::span<const Real> initial_values = {})
    {
        FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                    "AuxiliaryState::registerBlock: empty block name");
        FE_THROW_IF(block_name_to_index_.count(spec.name) != 0u, InvalidArgumentException,
                    "AuxiliaryState::registerBlock: duplicate block '" + spec.name + "'");

        const auto idx = blocks_.size();
        blocks_.emplace_back();
        auto& block = blocks_.back();

        if (spec.layout_mode == AuxiliaryLayoutMode::FixedStride) {
            block.setupFixedStride(spec, entity_count);
        } else {
            FE_THROW(NotImplementedException,
                     "AuxiliaryState::registerBlock: ragged layout requires "
                     "registerBlockRagged()");
        }

        if (!initial_values.empty()) {
            block.initialize(initial_values);
        }

        block_name_to_index_.emplace(spec.name, idx);
        return idx;
    }

    /**
     * @brief Register a new block with ragged per-entity layout.
     *
     * @param spec    Block specification.
     * @param offsets Per-entity offsets (size = entity_count + 1).
     * @param initial_values Optional initial values (must be offsets.back()).
     *
     * @return Index of the newly registered block.
     */
    std::size_t registerBlockRagged(const AuxiliaryStateSpec& spec,
                                    std::span<const std::size_t> offsets,
                                    std::span<const Real> initial_values = {})
    {
        FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                    "AuxiliaryState::registerBlockRagged: empty block name");
        FE_THROW_IF(block_name_to_index_.count(spec.name) != 0u, InvalidArgumentException,
                    "AuxiliaryState::registerBlockRagged: duplicate block '" + spec.name + "'");

        const auto idx = blocks_.size();
        blocks_.emplace_back();
        auto& block = blocks_.back();
        block.setupRagged(spec, offsets);

        if (!initial_values.empty()) {
            block.initialize(initial_values);
        }

        block_name_to_index_.emplace(spec.name, idx);
        return idx;
    }

    /// Number of registered blocks.
    [[nodiscard]] std::size_t blockCount() const noexcept { return blocks_.size(); }

    /// Whether a block with the given name exists.
    [[nodiscard]] bool hasBlock(std::string_view name) const noexcept
    {
        return block_name_to_index_.find(std::string(name)) !=
               block_name_to_index_.end();
    }

    /// Get block index by name (throws if not found).
    [[nodiscard]] std::size_t blockIndex(std::string_view name) const
    {
        auto it = block_name_to_index_.find(std::string(name));
        FE_THROW_IF(it == block_name_to_index_.end(), InvalidArgumentException,
                    "AuxiliaryState: unknown block '" + std::string(name) + "'");
        return it->second;
    }

    /// Get block by index.
    [[nodiscard]] AuxiliaryBlockStorage& block(std::size_t idx)
    {
        FE_THROW_IF(idx >= blocks_.size(), InvalidArgumentException,
                    "AuxiliaryState::block: index out of range");
        return blocks_[idx];
    }
    [[nodiscard]] const AuxiliaryBlockStorage& block(std::size_t idx) const
    {
        FE_THROW_IF(idx >= blocks_.size(), InvalidArgumentException,
                    "AuxiliaryState::block: index out of range");
        return blocks_[idx];
    }

    /// Get block by name.
    [[nodiscard]] AuxiliaryBlockStorage& getBlock(std::string_view name)
    {
        return blocks_[blockIndex(name)];
    }
    [[nodiscard]] const AuxiliaryBlockStorage& getBlock(std::string_view name) const
    {
        return blocks_[blockIndex(name)];
    }

    /// Get all block names (in registration order).
    [[nodiscard]] std::vector<std::string> blockNames() const
    {
        std::vector<std::string> names;
        names.reserve(blocks_.size());
        for (const auto& b : blocks_) {
            names.push_back(b.name());
        }
        return names;
    }

    /**
     * @brief Reset all blocks to their committed state.
     */
    void resetAllBlocks()
    {
        for (auto& b : blocks_) {
            b.resetToCommitted();
        }
    }

    /**
     * @brief Commit all blocks at the given time.
     */
    void commitAllBlocks(Real time)
    {
        for (auto& b : blocks_) {
            b.commitTimeStep(time);
        }
    }

    /**
     * @brief Rollback all blocks to their committed state.
     */
    void rollbackAllBlocks()
    {
        for (auto& b : blocks_) {
            b.rollback();
        }
    }

    /**
     * @brief Get a storage summary across all blocks.
     */
    [[nodiscard]] AuxiliaryStateStorageSummary storageSummary() const noexcept
    {
        AuxiliaryStateStorageSummary summary;
        summary.block_count = blocks_.size();
        for (const auto& b : blocks_) {
            summary.total_work_storage += b.storageSize();
            summary.total_committed_storage += b.storageSize();
            summary.total_history_storage += b.history().totalHistoryStorage();
        }
        return summary;
    }

private:
    // --- Legacy flat storage ---
    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> committed_{};
    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> values_{};
    std::vector<std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>>> flat_history_{};
    std::unordered_map<std::string, std::size_t> name_to_index_{};

    // --- Block-based storage ---
    std::vector<AuxiliaryBlockStorage> blocks_{};
    std::unordered_map<std::string, std::size_t> block_name_to_index_{};
};

/**
 * @brief Registration record for an auxiliary state variable with ODE/DAE
 *        time integration.
 *
 * This struct captures an auxiliary model's specification, initial values,
 * boundary functional dependencies, time integration method, and optional
 * analytic derivatives.  It is used by the coupled-boundary path and will
 * be superseded by the declarative `AuxiliaryModel` deployment API in a
 * future phase.
 *
 * The RHS expression must be a pure auxiliary expression: it must not
 * contain `TestFunction`, `TrialFunction`, `DiscreteField`, or
 * `StateField`, and must not contain measures (`dx`/`ds`/`dS`).
 * Coupled placeholders must be resolved to slot references before use.
 */
struct AuxiliaryStateRegistration {
    AuxiliaryStateSpec spec{};
    std::vector<Real> initial_values{};
    std::vector<forms::BoundaryFunctional> required_integrals{};

    /// Boundary markers where this auxiliary state is used.
    /// This is registration metadata — not part of the auxiliary block
    /// identity or storage specification.
    std::vector<int> associated_markers{};

    /// Setup-resolved slot for this variable in AuxiliaryState::values().
    static constexpr std::uint32_t kInvalidSlot = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t slot{kInvalidSlot};

    /// RHS expression: dX/dt = rhs(aux, integrals, t, dt, params)
    forms::FormExpr rhs{};

    /// Optional analytic derivative drhs/dX for implicit methods.
    std::optional<forms::FormExpr> d_rhs_dX{};

    ODEMethod integrator{ODEMethod::BackwardEuler};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_AUXILIARY_STATE_H
