#ifndef SVMP_FE_SYSTEMS_AUXILIARY_STATE_H
#define SVMP_FE_SYSTEMS_AUXILIARY_STATE_H

/**
 * @file AuxiliaryState.h
 * @brief Non-PDE auxiliary state variables for coupled boundary conditions
 *
 * This module provides a small container for auxiliary (0D) state variables
 * that evolve via user-supplied ODE callbacks and can be coupled back into PDE
 * boundary conditions through FE/Forms coefficients.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

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

struct AuxiliaryStateSpec {
    int size{0};
    std::string name{};
    std::vector<int> associated_markers{};
    std::vector<std::string> component_names{};
};

/**
 * @brief Mutable container for named scalar auxiliary variables
 *
 * Storage model:
 * - `committed_` is the last committed time-step state (u^{n-1} for the ODE).
 * - `values_` is the current "work" state used during assembly (may be updated
 *   multiple times within a time step / nonlinear iteration).
 * - `history_` stores older committed states (u^{n-2}, u^{n-3}, ...) for
 *   multi-step auxiliary integrators.
 */
class AuxiliaryState {
public:
    AuxiliaryState() = default;

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
        FE_THROW_IF(idx >= history_.size(), InvalidArgumentException,
                    "AuxiliaryState::previous(k): insufficient history");
        return history_[idx];
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
        return idx < history_.size();
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
        history_.clear();
        name_to_index_.clear();
    }

    /**
     * @brief Register a new auxiliary state block
     *
     * For size==1, the variable name is spec.name unless component_names is provided.
     * For size>1, component_names (if provided) must have size entries; otherwise
     * names are generated as `name[i]`.
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

    /**
     * @brief Reset work values to the last committed state (does not change history)
     */
    void resetToCommitted()
    {
        values_ = committed_;
    }

    /**
     * @brief Commit the current work values as the new state and push history
     */
    void commitTimeStep(std::size_t max_history = 8)
    {
        if (values_.empty()) return;
        if (!committed_.empty()) {
            history_.insert(history_.begin(), committed_);
            if (history_.size() > max_history) {
                history_.resize(max_history);
            }
        }
        committed_ = values_;
    }

private:
    std::vector<Real> committed_{};
    std::vector<Real> values_{};
    std::vector<std::vector<Real>> history_{};
    std::unordered_map<std::string, std::size_t> name_to_index_{};
};

/**
 * @brief Registration for an auxiliary scalar ODE state variable
 *
 * Users provide only the RHS: dX/dt = rhs(state, integrals, t). Time integration
 * is handled by `systems::ODEIntegrator`.
 *
 * Current scope: scalar variables (spec.size == 1).
 */
struct AuxiliaryStateRegistration {
    AuxiliaryStateSpec spec{};
    std::vector<Real> initial_values{};
    std::vector<forms::BoundaryFunctional> required_integrals{};

    // Setup-resolved slot for this scalar variable in AuxiliaryState::values().
    static constexpr std::uint32_t kInvalidSlot = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t slot{kInvalidSlot};

    // Scalar ODE RHS: dX/dt = rhs(aux, integrals, t, dt, params)
    // Requirements:
    // - Must not contain TestFunction/TrialFunction/DiscreteField/StateField
    // - Must not contain measures (dx/ds/dS)
    // - Coupled placeholders must be resolved to slot refs before use
    forms::FormExpr rhs{};

    // Optional analytic derivative drhs/dX for implicit methods (BackwardEuler/BDF2).
    std::optional<forms::FormExpr> d_rhs_dX{};

    ODEMethod integrator{ODEMethod::BackwardEuler};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_AUXILIARY_STATE_H
