#ifndef SVMP_FE_CONSTITUTIVE_GLOBAL_LAW_H
#define SVMP_FE_CONSTITUTIVE_GLOBAL_LAW_H

#include "Core/FEException.h"
#include "Core/Types.h"

#include "Assembly/Assembler.h"
#include "Assembly/GlobalSystemView.h"

#include "Forms/ConstitutiveModel.h"

#include "Systems/GlobalKernel.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {

namespace sparsity {
class SparsityPattern;
}

namespace systems {
class FESystem;
struct AssemblyRequest;
struct SystemStateView;
} // namespace systems

namespace constitutive {

/**
 * @brief Constitutive-level definition of a globally-coupled operator term
 *
 * GlobalLaw is authored in `FE/Constitutive`, but can be installed into a
 * `systems::FESystem` by emitting `systems::GlobalKernel` adapters.
 *
 * This supports intrinsically nonlocal / mesh-wide terms while keeping the
 * authoring surface in the Constitutive module.
 */
class GlobalLaw : public std::enable_shared_from_this<GlobalLaw> {
public:
    struct Emitted {
        std::vector<std::shared_ptr<const forms::ConstitutiveModel>> pointwise_models{};
        std::vector<std::shared_ptr<systems::GlobalKernel>> global_kernels{};
    };

    virtual ~GlobalLaw() = default;

    [[nodiscard]] virtual std::string name() const { return "GlobalLaw"; }

    /**
     * @brief Optional Forms-compatible pointwise models provided by this law
     *
     * Many laws are purely global and return an empty list; others may provide
     * one or more local `forms::ConstitutiveModel` implementations used inside
     * element/face weak-form terms.
     */
    [[nodiscard]] virtual std::vector<std::shared_ptr<const forms::ConstitutiveModel>> pointwiseModels() const
    {
        return {};
    }

    /**
     * @brief Optional sparsity augmentation hook for the global contributions
     */
    virtual void addSparsityCouplings(const systems::FESystem& /*system*/,
                                      sparsity::SparsityPattern& /*pattern*/) const
    {
    }

    /**
     * @brief Assemble global contributions into the requested outputs
     */
    [[nodiscard]] virtual assembly::AssemblyResult assemble(const systems::FESystem& system,
                                                            const systems::AssemblyRequest& request,
                                                            const systems::SystemStateView& state,
                                                            assembly::GlobalSystemView* matrix_out,
                                                            assembly::GlobalSystemView* vector_out) const = 0;

    /**
     * @brief Emit Systems adapters for this law
     *
     * The default implementation emits a single `systems::GlobalKernel` adapter
     * that forwards to this `GlobalLaw`. Override to emit multiple kernels.
     *
     * @note Calling this method requires that the object is owned by a
     * `std::shared_ptr` (uses `shared_from_this()`).
     */
    [[nodiscard]] virtual std::vector<std::shared_ptr<systems::GlobalKernel>> globalKernels() const;

    /**
     * @brief Emit both pointwise models and global kernels in one call
     */
    [[nodiscard]] virtual Emitted emit() const
    {
        Emitted out;
        out.pointwise_models = pointwiseModels();
        out.global_kernels = globalKernels();
        return out;
    }
};

/**
 * @brief Create a Systems GlobalKernel adapter for a GlobalLaw
 */
[[nodiscard]] std::shared_ptr<systems::GlobalKernel>
makeGlobalKernelAdapter(std::shared_ptr<const GlobalLaw> law);

/**
 * @brief Convenience: install all emitted global kernels into a system operator
 *
 * This is a template to avoid forcing `Constitutive/GlobalLaw.h` to include
 * `Systems/FESystem.h`. Any `SystemLike` with an `addGlobalKernel(op, kernel)`
 * method is supported.
 */
template <class SystemLike>
void installGlobalLawKernels(SystemLike& system,
                             std::string op,
                             std::shared_ptr<const GlobalLaw> law)
{
    FE_THROW_IF(!law, InvalidArgumentException, "installGlobalLawKernels: law is null");
    for (auto& kernel : law->globalKernels()) {
        system.addGlobalKernel(op, std::move(kernel));
    }
}

} // namespace constitutive

namespace systems {

/**
 * @brief Systems::GlobalKernel adapter that forwards to a constitutive::GlobalLaw
 *
 * This keeps the authoring API in `FE/Constitutive` while integrating with the
 * Systems assembly pipeline via `systems::GlobalKernel`.
 */
class GlobalLawKernelAdapter final : public GlobalKernel {
public:
    explicit GlobalLawKernelAdapter(std::shared_ptr<const constitutive::GlobalLaw> law)
        : law_(std::move(law))
    {
        FE_THROW_IF(!law_, InvalidArgumentException, "GlobalLawKernelAdapter: law is null");
    }

    [[nodiscard]] std::string name() const override { return law_->name(); }

    void addSparsityCouplings(const FESystem& system,
                              sparsity::SparsityPattern& pattern) const override
    {
        law_->addSparsityCouplings(system, pattern);
    }

    [[nodiscard]] assembly::AssemblyResult assemble(const FESystem& system,
                                                    const AssemblyRequest& request,
                                                    const SystemStateView& state,
                                                    assembly::GlobalSystemView* matrix_out,
                                                    assembly::GlobalSystemView* vector_out) override
    {
        return law_->assemble(system, request, state, matrix_out, vector_out);
    }

private:
    std::shared_ptr<const constitutive::GlobalLaw> law_{};
};

} // namespace systems

namespace constitutive {

inline std::vector<std::shared_ptr<systems::GlobalKernel>> GlobalLaw::globalKernels() const
{
    return {makeGlobalKernelAdapter(shared_from_this())};
}

inline std::shared_ptr<systems::GlobalKernel>
makeGlobalKernelAdapter(std::shared_ptr<const GlobalLaw> law)
{
    return std::make_shared<systems::GlobalLawKernelAdapter>(std::move(law));
}

} // namespace constitutive

} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTITUTIVE_GLOBAL_LAW_H
