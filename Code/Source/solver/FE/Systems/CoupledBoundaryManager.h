#ifndef SVMP_FE_SYSTEMS_COUPLED_BOUNDARY_MANAGER_H
#define SVMP_FE_SYSTEMS_COUPLED_BOUNDARY_MANAGER_H

/**
 * @file CoupledBoundaryManager.h
 * @brief Orchestration for coupled boundary conditions (boundary functionals + aux state)
 *
 * This class provides the FE-side infrastructure for non-local / 0D-3D coupled
 * boundary conditions:
 * - computes required boundary integrals (BoundaryFunctional),
 * - evolves auxiliary state via user-provided callbacks,
 * - provides a stable CoupledBCContext object for coefficient evaluators.
 *
 * Current scope:
 * - Single-field Systems only (primary field specified at construction).
 * - BoundaryFunctional reductions: Sum and Average (Max/Min throw for now).
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Core/ParameterValue.h"

#include "Constraints/CoupledBCContext.h"
#include "Constraints/CoupledNeumannBC.h"
#include "Constraints/CoupledRobinBC.h"
#include "Forms/BoundaryFunctional.h"
#include "Systems/AuxiliaryState.h"
#include "Systems/SystemState.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {

namespace assembly {
class FunctionalKernel;
}

namespace systems {

class FESystem;

class CoupledBoundaryManager {
public:
    CoupledBoundaryManager(FESystem& system, FieldId primary_field);
    ~CoupledBoundaryManager();

    CoupledBoundaryManager(CoupledBoundaryManager&&) noexcept = delete;
    CoupledBoundaryManager& operator=(CoupledBoundaryManager&&) noexcept = delete;
    CoupledBoundaryManager(const CoupledBoundaryManager&) = delete;
    CoupledBoundaryManager& operator=(const CoupledBoundaryManager&) = delete;

    [[nodiscard]] FieldId primaryField() const noexcept { return primary_field_; }

    // ---------------------------------------------------------------------
    // Registration
    // ---------------------------------------------------------------------

    void addBoundaryFunctional(forms::BoundaryFunctional functional);
    void addCoupledNeumannBC(constraints::CoupledNeumannBC bc);
    void addCoupledRobinBC(constraints::CoupledRobinBC bc);
    void addAuxiliaryState(AuxiliaryStateRegistration registration);

    // ---------------------------------------------------------------------
    // Parameter contract integration
    // ---------------------------------------------------------------------

    /**
     * @brief Parameter requirements implied by stored expressions (boundary functionals + ODEs)
     *
     * This allows Systems to register parameter contracts for coupled BC
     * infrastructure alongside regular AssemblyKernel requirements.
     */
    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const;

    /**
     * @brief Resolve ParameterSymbol terminals to ParameterRef(slot) for JIT-friendly evaluation
     *
     * Must be called after Systems has finalized the ParameterRegistry slot layout.
     */
    void resolveParameterSlots(
        const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param);

    // ---------------------------------------------------------------------
    // Assembly-time update
    // ---------------------------------------------------------------------

    /**
     * @brief Update integrals + auxiliary state for an upcoming assembly call
     *
     * Safe to call multiple times (e.g., during nonlinear iterations):
     * the auxiliary state work values are reset to the committed state each call.
     */
    void prepareForAssembly(const SystemStateView& state);

    // ---------------------------------------------------------------------
    // Time-step lifecycle
    // ---------------------------------------------------------------------

    void beginTimeStep();
    void commitTimeStep();

    // ---------------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------------

    [[nodiscard]] const forms::BoundaryFunctionalResults& integrals() const noexcept { return integrals_; }
    [[nodiscard]] const AuxiliaryState& auxiliaryState() const noexcept { return aux_state_; }
    [[nodiscard]] AuxiliaryState& auxiliaryState() noexcept { return aux_state_; }

    [[nodiscard]] const constraints::CoupledBCContext& context() const noexcept { return ctx_; }
    [[nodiscard]] const constraints::CoupledBCContext* contextPtr() const noexcept { return &ctx_; }

private:
    struct CompiledFunctional {
        forms::BoundaryFunctional def{};
        std::shared_ptr<assembly::FunctionalKernel> kernel{};
    };

    void requireSetup() const;
    void compileFunctionalIfNeeded(const forms::BoundaryFunctional& functional);
    Real evaluateFunctional(const CompiledFunctional& entry, const SystemStateView& state);
    Real boundaryMeasure(int boundary_marker, const SystemStateView& state);

    FESystem& system_;
    FieldId primary_field_{INVALID_FIELD_ID};

    std::vector<CompiledFunctional> functionals_{};
    std::unordered_map<std::string, std::size_t> name_to_functional_{};
    std::unordered_map<int, Real> boundary_measure_cache_{};

    std::vector<constraints::CoupledNeumannBC> coupled_neumann_{};
    std::vector<constraints::CoupledRobinBC> coupled_robin_{};
    std::vector<AuxiliaryStateRegistration> aux_registrations_{};

    forms::BoundaryFunctionalResults integrals_{};
    AuxiliaryState aux_state_{};
    constraints::CoupledBCContext ctx_{integrals_, aux_state_, 0.0, 0.0};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_COUPLED_BOUNDARY_MANAGER_H
