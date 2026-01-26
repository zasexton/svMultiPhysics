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
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace forms {

struct ConstitutiveStateLayout;

struct MaterialStateUpdate {
    std::uint32_t offset_bytes{0u};
    FormExpr value{};
};

struct InlinedMaterialStateUpdateProgram {
    std::vector<MaterialStateUpdate> cell{};
    std::vector<MaterialStateUpdate> boundary_all{};
    std::unordered_map<int, std::vector<MaterialStateUpdate>> boundary_by_marker{};
    std::vector<MaterialStateUpdate> interior_face{};
    std::vector<MaterialStateUpdate> interface_face{};

    [[nodiscard]] bool empty() const noexcept
    {
        return cell.empty() &&
               boundary_all.empty() &&
               boundary_by_marker.empty() &&
               interior_face.empty() &&
               interface_face.empty();
    }

    void clear()
    {
        cell.clear();
        boundary_all.clear();
        boundary_by_marker.clear();
        interior_face.clear();
        interface_face.clear();
    }
};

/**
 * @brief Side selector for DG (minus/plus) evaluations.
 */
enum class Side : std::uint8_t { Minus, Plus };

void applyInlinedMaterialStateUpdatesReal(const assembly::AssemblyContext& ctx_minus,
                                          const assembly::AssemblyContext* ctx_plus,
                                          FormKind kind,
                                          const ConstitutiveStateLayout* constitutive_state,
                                          const std::vector<MaterialStateUpdate>& updates,
                                          Side side,
                                          LocalIndex q);

void applyInlinedMaterialStateUpdatesDual(const assembly::AssemblyContext& ctx_minus,
                                          const assembly::AssemblyContext* ctx_plus,
                                          const ConstitutiveStateLayout* constitutive_state,
                                          const std::vector<MaterialStateUpdate>& updates,
                                          Side side,
                                          LocalIndex q);

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
    void resolveInlinableConstitutives() override;
    void resolveParameterSlots(
        const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param) override;
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override { return ir_.maxTimeDerivativeOrder(); }

    [[nodiscard]] bool hasCell() const noexcept override;
    [[nodiscard]] bool hasBoundaryFace() const noexcept override;
    [[nodiscard]] bool hasInteriorFace() const noexcept override;
    [[nodiscard]] bool hasInterfaceFace() const noexcept override;

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

    void computeInterfaceFace(const assembly::AssemblyContext& ctx_minus,
                              const assembly::AssemblyContext& ctx_plus,
                              int interface_marker,
                              assembly::KernelOutput& output_minus,
                              assembly::KernelOutput& output_plus,
                              assembly::KernelOutput& coupling_mp,
                              assembly::KernelOutput& coupling_pm) override;

    [[nodiscard]] std::string name() const override { return "Forms::FormKernel"; }
    [[nodiscard]] const FormIR& ir() const noexcept { return ir_; }
    [[nodiscard]] const InlinedMaterialStateUpdateProgram& inlinedStateUpdates() const noexcept
    {
        return inlined_state_updates_;
    }
    [[nodiscard]] const ConstitutiveStateLayout* constitutiveStateLayout() const noexcept
    {
        return constitutive_state_.get();
    }

private:
    void ensureInterpreterLoweredIndexedAccess();

    FormIR ir_;
    std::vector<params::Spec> parameter_specs_{};
    std::shared_ptr<const ConstitutiveStateLayout> constitutive_state_{};
    assembly::MaterialStateSpec material_state_spec_{};
    InlinedMaterialStateUpdateProgram inlined_state_updates_{};
    std::unique_ptr<std::once_flag> indexed_lowering_once_{std::make_unique<std::once_flag>()};
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
    void resolveInlinableConstitutives() override;
    void resolveParameterSlots(
        const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param) override;
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override;

    [[nodiscard]] bool hasCell() const noexcept override;
    [[nodiscard]] bool hasBoundaryFace() const noexcept override;
    [[nodiscard]] bool hasInteriorFace() const noexcept override;
    [[nodiscard]] bool hasInterfaceFace() const noexcept override;

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

    void computeInterfaceFace(const assembly::AssemblyContext& ctx_minus,
                              const assembly::AssemblyContext& ctx_plus,
                              int interface_marker,
                              assembly::KernelOutput& output_minus,
                              assembly::KernelOutput& output_plus,
                              assembly::KernelOutput& coupling_mp,
                              assembly::KernelOutput& coupling_pm) override;

    [[nodiscard]] std::string name() const override { return "Forms::LinearFormKernel"; }
    [[nodiscard]] bool isMatrixOnly() const noexcept override { return output_ == LinearKernelOutput::MatrixOnly; }
    [[nodiscard]] bool isVectorOnly() const noexcept override { return output_ == LinearKernelOutput::VectorOnly; }
    [[nodiscard]] const FormIR& bilinearIR() const noexcept { return bilinear_ir_; }
    [[nodiscard]] const std::optional<FormIR>& linearIR() const noexcept { return linear_ir_; }
    [[nodiscard]] const InlinedMaterialStateUpdateProgram& inlinedStateUpdates() const noexcept
    {
        return inlined_state_updates_;
    }
    [[nodiscard]] const ConstitutiveStateLayout* constitutiveStateLayout() const noexcept
    {
        return constitutive_state_.get();
    }

private:
    void ensureInterpreterLoweredIndexedAccess();

    FormIR bilinear_ir_;
    std::optional<FormIR> linear_ir_{};
    LinearKernelOutput output_{LinearKernelOutput::Both};
    std::vector<assembly::FieldRequirement> field_requirements_{};
    std::vector<params::Spec> parameter_specs_{};
    std::shared_ptr<const ConstitutiveStateLayout> constitutive_state_{};
    assembly::MaterialStateSpec material_state_spec_{};
    InlinedMaterialStateUpdateProgram inlined_state_updates_{};
    std::unique_ptr<std::once_flag> indexed_lowering_once_{std::make_unique<std::once_flag>()};
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
    void resolveInlinableConstitutives() override;
    void resolveParameterSlots(
        const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param) override;
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override {
        return residual_ir_.maxTimeDerivativeOrder();
    }

    [[nodiscard]] bool hasCell() const noexcept override;
    [[nodiscard]] bool hasBoundaryFace() const noexcept override;
    [[nodiscard]] bool hasInteriorFace() const noexcept override;
    [[nodiscard]] bool hasInterfaceFace() const noexcept override;

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

    void computeInterfaceFace(const assembly::AssemblyContext& ctx_minus,
                              const assembly::AssemblyContext& ctx_plus,
                              int interface_marker,
                              assembly::KernelOutput& output_minus,
                              assembly::KernelOutput& output_plus,
                              assembly::KernelOutput& coupling_mp,
                              assembly::KernelOutput& coupling_pm) override;

    [[nodiscard]] std::string name() const override { return "Forms::NonlinearFormKernel"; }
    [[nodiscard]] bool isMatrixOnly() const noexcept override { return output_ == NonlinearKernelOutput::MatrixOnly; }
    [[nodiscard]] bool isVectorOnly() const noexcept override { return output_ == NonlinearKernelOutput::VectorOnly; }
    [[nodiscard]] const FormIR& residualIR() const noexcept { return residual_ir_; }
    [[nodiscard]] const InlinedMaterialStateUpdateProgram& inlinedStateUpdates() const noexcept { return inlined_state_updates_; }
    [[nodiscard]] const ConstitutiveStateLayout* constitutiveStateLayout() const noexcept { return constitutive_state_.get(); }

private:
    void ensureInterpreterLoweredIndexedAccess();

    FormIR residual_ir_;
    ADMode ad_mode_{ADMode::Forward};
    NonlinearKernelOutput output_{NonlinearKernelOutput::Both};
    std::vector<params::Spec> parameter_specs_{};
    std::shared_ptr<const ConstitutiveStateLayout> constitutive_state_{};
    assembly::MaterialStateSpec material_state_spec_{};
    InlinedMaterialStateUpdateProgram inlined_state_updates_{};
    std::unique_ptr<std::once_flag> indexed_lowering_once_{std::make_unique<std::once_flag>()};
};

/**
 * @brief Kernel for nonlinear residual forms using symbolic tangent decomposition (no Dual arrays)
 *
 * Strategy:
 * - Residual: evaluate the residual FormIR (FormKind::Residual) without seeding any DOF derivatives.
 * - Jacobian: build a bilinear tangent form a(δu,v) symbolically, compile to FormIR (FormKind::Bilinear),
 *   and assemble the matrix with pure scalar evaluation.
 *
 * Note: Constitutive calls must be eliminated via resolveInlinableConstitutives() for differentiation.
 */
class SymbolicNonlinearFormKernel final : public assembly::AssemblyKernel {
public:
    explicit SymbolicNonlinearFormKernel(FormIR residual_ir,
                                         NonlinearKernelOutput output = NonlinearKernelOutput::Both);
    ~SymbolicNonlinearFormKernel() override;

    SymbolicNonlinearFormKernel(SymbolicNonlinearFormKernel&&) noexcept;
    SymbolicNonlinearFormKernel& operator=(SymbolicNonlinearFormKernel&&) noexcept;

    SymbolicNonlinearFormKernel(const SymbolicNonlinearFormKernel&) = delete;
    SymbolicNonlinearFormKernel& operator=(const SymbolicNonlinearFormKernel&) = delete;

    [[nodiscard]] assembly::RequiredData getRequiredData() const noexcept override;
    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override;
    [[nodiscard]] assembly::MaterialStateSpec materialStateSpec() const noexcept override;
    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override;
    void resolveInlinableConstitutives() override;
    void resolveParameterSlots(
        const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param) override;
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override {
        return residual_ir_.maxTimeDerivativeOrder();
    }

    [[nodiscard]] bool hasCell() const noexcept override;
    [[nodiscard]] bool hasBoundaryFace() const noexcept override;
    [[nodiscard]] bool hasInteriorFace() const noexcept override;
    [[nodiscard]] bool hasInterfaceFace() const noexcept override;

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

    void computeInterfaceFace(const assembly::AssemblyContext& ctx_minus,
                              const assembly::AssemblyContext& ctx_plus,
                              int interface_marker,
                              assembly::KernelOutput& output_minus,
                              assembly::KernelOutput& output_plus,
                              assembly::KernelOutput& coupling_mp,
                              assembly::KernelOutput& coupling_pm) override;

    [[nodiscard]] std::string name() const override { return "Forms::SymbolicNonlinearFormKernel"; }
    [[nodiscard]] bool isMatrixOnly() const noexcept override { return output_ == NonlinearKernelOutput::MatrixOnly; }
    [[nodiscard]] bool isVectorOnly() const noexcept override { return output_ == NonlinearKernelOutput::VectorOnly; }
    [[nodiscard]] const FormIR& residualIR() const noexcept { return residual_ir_; }
    [[nodiscard]] const FormIR& tangentIR() const noexcept { return tangent_ir_; }
    [[nodiscard]] const InlinedMaterialStateUpdateProgram& inlinedStateUpdates() const noexcept { return inlined_state_updates_; }
    [[nodiscard]] const ConstitutiveStateLayout* constitutiveStateLayout() const noexcept { return constitutive_state_.get(); }

private:
    void rebuildTangentIR();
    void rewriteResidualTrialToState();
    void ensureInterpreterLoweredIndexedAccess();

    FormIR residual_ir_;
    FormIR tangent_ir_;
    bool tangent_ready_{false};
    bool residual_scalar_ready_{false};
    bool inlinable_constitutives_resolved_{false};
    NonlinearKernelOutput output_{NonlinearKernelOutput::Both};
    std::vector<params::Spec> parameter_specs_{};
    std::shared_ptr<const ConstitutiveStateLayout> constitutive_state_{};
    assembly::MaterialStateSpec material_state_spec_{};
    InlinedMaterialStateUpdateProgram inlined_state_updates_{};
    std::unique_ptr<std::once_flag> indexed_lowering_once_{std::make_unique<std::once_flag>()};
};

/**
 * @brief Sensitivity kernel for d/dQ of a residual form, where Q is a coupled scalar
 *
 * This kernel assembles the DOF-vector sensitivities:
 *   s_i = ∂R_i/∂Q
 * for a single coupled scalar slot Q, with optional indirect dependence through
 * AuxiliaryStateRef terminals seeded by a provided d(aux)/dQ column.
 *
 * Notes:
 * - The TrialFunction is treated as constant (no derivative seeding).
 * - This is intended as infrastructure for coupled-BC Jacobian chain-rule updates.
 */
class CoupledResidualSensitivityKernel final : public assembly::AssemblyKernel {
public:
    CoupledResidualSensitivityKernel(const NonlinearFormKernel& base,
                                     std::uint32_t coupled_integral_slot,
                                     std::span<const Real> daux_dintegrals,
                                     std::size_t num_integrals);
    ~CoupledResidualSensitivityKernel() override = default;

    CoupledResidualSensitivityKernel(const CoupledResidualSensitivityKernel&) = delete;
    CoupledResidualSensitivityKernel& operator=(const CoupledResidualSensitivityKernel&) = delete;

    [[nodiscard]] assembly::RequiredData getRequiredData() const noexcept override;
    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override;
    [[nodiscard]] assembly::MaterialStateSpec materialStateSpec() const noexcept override;
    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override;
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override;

    [[nodiscard]] bool hasCell() const noexcept override;
    [[nodiscard]] bool hasBoundaryFace() const noexcept override;
    [[nodiscard]] bool hasInteriorFace() const noexcept override;
    [[nodiscard]] bool hasInterfaceFace() const noexcept override;

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

    void computeInterfaceFace(const assembly::AssemblyContext& ctx_minus,
                              const assembly::AssemblyContext& ctx_plus,
                              int interface_marker,
                              assembly::KernelOutput& output_minus,
                              assembly::KernelOutput& output_plus,
                              assembly::KernelOutput& coupling_mp,
                              assembly::KernelOutput& coupling_pm) override;

    [[nodiscard]] std::string name() const override { return "Forms::CoupledResidualSensitivityKernel"; }

private:
    const NonlinearFormKernel* base_{nullptr};
    std::uint32_t coupled_integral_slot_{0u};
    std::span<const Real> daux_dintegrals_{};
    std::size_t num_integrals_{0u};
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
    void ensureInterpreterLoweredIndexedAccess();

    FormExpr integrand_{};
    Domain domain_{Domain::Cell};
    assembly::RequiredData required_data_{assembly::RequiredData::None};
    std::vector<assembly::FieldRequirement> field_requirements_{};
    std::unique_ptr<std::once_flag> indexed_lowering_once_{std::make_unique<std::once_flag>()};
};

/**
 * @brief Boundary functional gradient kernel (d/dU of a boundary integral)
 *
 * This kernel assembles the DOF-gradient of a scalar boundary integral:
 *   Q(U) = ∫_Γ q(u_h, ∇u_h, ...) ds
 * into a global vector g where g_j = ∂Q/∂U_j.
 *
 * The stored integrand must be a scalar-valued expression that may contain a
 * TrialFunction but must not contain a TestFunction. (For boundary functionals,
 * callers typically transform DiscreteField/StateField terminals into a TrialFunction
 * prior to constructing this kernel.)
 *
 * Note: Reduction (Sum/Average) is handled by the caller; this kernel always assembles
 * the raw integral gradient for the specified boundary marker.
 */
class BoundaryFunctionalGradientKernel final : public assembly::AssemblyKernel {
public:
    explicit BoundaryFunctionalGradientKernel(FormExpr integrand, int boundary_marker);
    ~BoundaryFunctionalGradientKernel() override = default;

    [[nodiscard]] assembly::RequiredData getRequiredData() const noexcept override { return required_data_; }
    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override { return field_requirements_; }

    [[nodiscard]] bool hasCell() const noexcept override { return false; }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }
    [[nodiscard]] bool hasInteriorFace() const noexcept override { return false; }
    [[nodiscard]] bool hasInterfaceFace() const noexcept override { return false; }

    void computeCell(const assembly::AssemblyContext& /*ctx*/,
                     assembly::KernelOutput& output) override
    {
        output.reserve(0, 0, false, false);
    }

    void computeBoundaryFace(const assembly::AssemblyContext& ctx,
                             int boundary_marker,
                             assembly::KernelOutput& output) override;

    void computeInteriorFace(const assembly::AssemblyContext& /*ctx_minus*/,
                             const assembly::AssemblyContext& /*ctx_plus*/,
                             assembly::KernelOutput& output_minus,
                             assembly::KernelOutput& output_plus,
                             assembly::KernelOutput& coupling_mp,
                             assembly::KernelOutput& coupling_pm) override
    {
        output_minus.reserve(0, 0, false, false);
        output_plus.reserve(0, 0, false, false);
        coupling_mp.reserve(0, 0, false, false);
        coupling_pm.reserve(0, 0, false, false);
    }

    void computeInterfaceFace(const assembly::AssemblyContext& /*ctx_minus*/,
                              const assembly::AssemblyContext& /*ctx_plus*/,
                              int /*interface_marker*/,
                              assembly::KernelOutput& output_minus,
                              assembly::KernelOutput& output_plus,
                              assembly::KernelOutput& coupling_mp,
                              assembly::KernelOutput& coupling_pm) override
    {
        output_minus.reserve(0, 0, false, false);
        output_plus.reserve(0, 0, false, false);
        coupling_mp.reserve(0, 0, false, false);
        coupling_pm.reserve(0, 0, false, false);
    }

    [[nodiscard]] std::string name() const override { return "Forms::BoundaryFunctionalGradientKernel"; }

private:
    FormExpr integrand_{};
    int boundary_marker_{-1};
    assembly::RequiredData required_data_{assembly::RequiredData::None};
    std::vector<assembly::FieldRequirement> field_requirements_{};
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_FORM_KERNELS_H
