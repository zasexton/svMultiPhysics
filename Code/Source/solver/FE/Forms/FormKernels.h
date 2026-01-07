#ifndef SVMP_FE_FORMS_FORM_KERNELS_H
#define SVMP_FE_FORMS_FORM_KERNELS_H

/**
 * @file FormKernels.h
 * @brief Assembly-kernel adapters for compiled FE/Forms forms
 */

#include "Assembly/AssemblyKernel.h"
#include "Forms/FormIR.h"

#include <memory>

namespace svmp {
namespace FE {
namespace forms {

struct ConstitutiveStateLayout;

/**
 * @brief Output mode for nonlinear (residual) kernels
 *
 * - Both: assemble residual vector and Jacobian matrix
 * - MatrixOnly: assemble only the Jacobian matrix
 * - VectorOnly: assemble only the residual vector
 */
enum class NonlinearKernelOutput : std::uint8_t {
    Both,
    MatrixOnly,
    VectorOnly
};

/**
 * @brief Kernel for linear/bilinear forms (no solution dependence)
 */
class FormKernel final : public assembly::AssemblyKernel {
public:
    explicit FormKernel(FormIR ir);
    ~FormKernel() override;

    FormKernel(FormKernel&&) noexcept;
    FormKernel& operator=(FormKernel&&) noexcept;

    FormKernel(const FormKernel&) = delete;
    FormKernel& operator=(const FormKernel&) = delete;

    [[nodiscard]] assembly::RequiredData getRequiredData() const noexcept override;
    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override;
    [[nodiscard]] assembly::MaterialStateSpec materialStateSpec() const noexcept override;
    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override;
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override { return ir_.maxTimeDerivativeOrder(); }

    [[nodiscard]] bool hasCell() const noexcept override;
    [[nodiscard]] bool hasBoundaryFace() const noexcept override;
    [[nodiscard]] bool hasInteriorFace() const noexcept override;

    void computeCell(const assembly::AssemblyContext& ctx,
                     assembly::KernelOutput& output) override;

    void computeBoundaryFace(const assembly::AssemblyContext& ctx,
                             int boundary_marker,
                             assembly::KernelOutput& output) override;

    void computeInteriorFace(const assembly::AssemblyContext& ctx_minus,
                             const assembly::AssemblyContext& ctx_plus,
                             assembly::KernelOutput& output_minus,
                             assembly::KernelOutput& output_plus,
                             assembly::KernelOutput& coupling_mp,
                             assembly::KernelOutput& coupling_pm) override;

    [[nodiscard]] std::string name() const override { return "Forms::FormKernel"; }

private:
    FormIR ir_;
    std::shared_ptr<const ConstitutiveStateLayout> constitutive_state_{};
    assembly::MaterialStateSpec material_state_spec_{};
};

/**
 * @brief Kernel for nonlinear residual forms with AD-based Jacobians
 *
 * This kernel can assemble:
 * - residual vector (has_vector=true)
 * - Jacobian matrix (has_matrix=true)
 * in a single pass, using forward-mode AD at the element level.
 */
class NonlinearFormKernel final : public assembly::AssemblyKernel {
public:
    explicit NonlinearFormKernel(FormIR residual_ir,
                                 ADMode ad_mode = ADMode::Forward,
                                 NonlinearKernelOutput output = NonlinearKernelOutput::Both);
    ~NonlinearFormKernel() override;

    NonlinearFormKernel(NonlinearFormKernel&&) noexcept;
    NonlinearFormKernel& operator=(NonlinearFormKernel&&) noexcept;

    NonlinearFormKernel(const NonlinearFormKernel&) = delete;
    NonlinearFormKernel& operator=(const NonlinearFormKernel&) = delete;

    [[nodiscard]] assembly::RequiredData getRequiredData() const noexcept override;
    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override;
    [[nodiscard]] assembly::MaterialStateSpec materialStateSpec() const noexcept override;
    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override;
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override {
        return residual_ir_.maxTimeDerivativeOrder();
    }

    [[nodiscard]] bool hasCell() const noexcept override;
    [[nodiscard]] bool hasBoundaryFace() const noexcept override;
    [[nodiscard]] bool hasInteriorFace() const noexcept override;

    void computeCell(const assembly::AssemblyContext& ctx,
                     assembly::KernelOutput& output) override;

    void computeBoundaryFace(const assembly::AssemblyContext& ctx,
                             int boundary_marker,
                             assembly::KernelOutput& output) override;

    void computeInteriorFace(const assembly::AssemblyContext& ctx_minus,
                             const assembly::AssemblyContext& ctx_plus,
                             assembly::KernelOutput& output_minus,
                             assembly::KernelOutput& output_plus,
                             assembly::KernelOutput& coupling_mp,
                             assembly::KernelOutput& coupling_pm) override;

    [[nodiscard]] std::string name() const override { return "Forms::NonlinearFormKernel"; }
    [[nodiscard]] bool isMatrixOnly() const noexcept override { return output_ == NonlinearKernelOutput::MatrixOnly; }
    [[nodiscard]] bool isVectorOnly() const noexcept override { return output_ == NonlinearKernelOutput::VectorOnly; }

private:
    FormIR residual_ir_;
    ADMode ad_mode_{ADMode::Forward};
    NonlinearKernelOutput output_{NonlinearKernelOutput::Both};
    std::shared_ptr<const ConstitutiveStateLayout> constitutive_state_{};
    assembly::MaterialStateSpec material_state_spec_{};
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_FORM_KERNELS_H
