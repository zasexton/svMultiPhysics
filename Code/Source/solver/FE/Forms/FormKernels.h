#ifndef SVMP_FE_FORMS_FORM_KERNELS_H
#define SVMP_FE_FORMS_FORM_KERNELS_H

/**
 * @file FormKernels.h
 * @brief Assembly-kernel adapters for compiled FE/Forms forms
 */

#include "Assembly/AssemblyKernel.h"
#include "Assembly/FunctionalAssembler.h"
#include "Forms/FormIR.h"

#include <memory>
#include <optional>

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
 * @brief Output mode for affine (linear-in-trial) residual kernels
 */
enum class LinearKernelOutput : std::uint8_t {
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
    void resolveParameterSlots(
        const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param) override;
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
 * @brief Kernel for residuals affine in the TrialFunction (R(u;v) = a(u,v) + L(v))
 *
 * This kernel assembles the Jacobian from the bilinear part `a(·,·)` and
 * assembles the residual vector by applying the bilinear operator to the
 * current element-local coefficient vector and adding the linear part `L(v)`.
 *
 * The intent is to avoid AD for residuals that are provably affine in the
 * TrialFunction while preserving the `installResidualForm(...)` API.
 */
class LinearFormKernel final : public assembly::AssemblyKernel {
public:
    explicit LinearFormKernel(FormIR bilinear_ir,
                              std::optional<FormIR> linear_ir = std::nullopt,
                              LinearKernelOutput output = LinearKernelOutput::Both);
    ~LinearFormKernel() override;

    LinearFormKernel(LinearFormKernel&&) noexcept;
    LinearFormKernel& operator=(LinearFormKernel&&) noexcept;

    LinearFormKernel(const LinearFormKernel&) = delete;
    LinearFormKernel& operator=(const LinearFormKernel&) = delete;

    [[nodiscard]] assembly::RequiredData getRequiredData() const noexcept override;
    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override;
    [[nodiscard]] assembly::MaterialStateSpec materialStateSpec() const noexcept override;
    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override;
    void resolveParameterSlots(
        const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param) override;
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override;

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

    [[nodiscard]] std::string name() const override { return "Forms::LinearFormKernel"; }
    [[nodiscard]] bool isMatrixOnly() const noexcept override { return output_ == LinearKernelOutput::MatrixOnly; }
    [[nodiscard]] bool isVectorOnly() const noexcept override { return output_ == LinearKernelOutput::VectorOnly; }

private:
    FormIR bilinear_ir_;
    std::optional<FormIR> linear_ir_{};
    LinearKernelOutput output_{LinearKernelOutput::Both};
    std::vector<assembly::FieldRequirement> field_requirements_{};
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
    void resolveParameterSlots(
        const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param) override;
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

/**
 * @brief Functional-kernel adapter for integrating a scalar FormExpr
 *
 * This is a lightweight bridge from FE/Forms to FE/Assembly's FunctionalAssembler,
 * intended for scalar quantities of interest (QoIs) and coupled boundary-condition
 * infrastructure (boundary functionals).
 *
 * Constraints:
 * - The expression must evaluate to a scalar at quadrature points.
 * - The expression must not contain TestFunction/TrialFunction nodes (use DiscreteField/StateField instead).
 */
class FunctionalFormKernel final : public assembly::FunctionalKernel {
public:
    enum class Domain : std::uint8_t {
        Cell,
        BoundaryFace
    };

    FunctionalFormKernel(FormExpr integrand,
                         Domain domain,
                         assembly::RequiredData required,
                         std::vector<assembly::FieldRequirement> field_requirements = {});

    ~FunctionalFormKernel() override = default;

    [[nodiscard]] assembly::RequiredData getRequiredData() const noexcept override { return required_data_; }
    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override { return field_requirements_; }

    [[nodiscard]] bool hasCell() const noexcept override { return domain_ == Domain::Cell; }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return domain_ == Domain::BoundaryFace; }

    [[nodiscard]] Real evaluateCell(const assembly::AssemblyContext& ctx, LocalIndex q) override;

    [[nodiscard]] Real evaluateBoundaryFace(const assembly::AssemblyContext& ctx,
                                            LocalIndex q,
                                            int boundary_marker) override;

    [[nodiscard]] std::string name() const override { return "Forms::FunctionalFormKernel"; }

private:
    FormExpr integrand_{};
    Domain domain_{Domain::Cell};
    assembly::RequiredData required_data_{assembly::RequiredData::None};
    std::vector<assembly::FieldRequirement> field_requirements_{};
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_FORM_KERNELS_H
