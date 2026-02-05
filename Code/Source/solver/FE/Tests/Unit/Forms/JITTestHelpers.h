#ifndef SVMP_FE_TESTS_UNIT_FORMS_JIT_TEST_HELPERS_H
#define SVMP_FE_TESTS_UNIT_FORMS_JIT_TEST_HELPERS_H

#include <gtest/gtest.h>

#include "Assembly/FunctionalAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"

#include <stdexcept>

namespace svmp {
namespace FE {
namespace forms {
namespace detail {

assembly::RequiredData analyzeRequiredData(const FormExprNode& node, FormKind kind);
std::vector<assembly::FieldRequirement> analyzeFieldRequirements(const FormExprNode& node);

} // namespace detail

namespace test {

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

#if !SVMP_FE_ENABLE_LLVM_JIT
#define requireLLVMJITOrSkip() \
    do {                       \
        GTEST_SKIP() << "FE was built without LLVM JIT support (SVMP_FE_ENABLE_LLVM_JIT=0)"; \
    } while (0)
#else
#define requireLLVMJITOrSkip() \
    do {                       \
    } while (0)
#endif

class ThrowingTotalKernel final : public assembly::FunctionalKernel {
public:
    ThrowingTotalKernel(assembly::RequiredData required,
                        std::vector<assembly::FieldRequirement> field_requirements,
                        bool has_cell,
                        bool has_boundary)
        : required_(required),
          field_requirements_(std::move(field_requirements)),
          has_cell_(has_cell),
          has_boundary_(has_boundary)
    {
    }

    [[nodiscard]] assembly::RequiredData getRequiredData() const noexcept override { return required_; }
    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override { return field_requirements_; }

    [[nodiscard]] bool hasCell() const noexcept override { return has_cell_; }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return has_boundary_; }

    [[nodiscard]] Real evaluateCell(const assembly::AssemblyContext& /*ctx*/, LocalIndex /*q*/) override { return Real(0.0); }

    [[nodiscard]] Real evaluateBoundaryFace(const assembly::AssemblyContext& /*ctx*/,
                                            LocalIndex /*q*/,
                                            int /*boundary_marker*/) override
    {
        return Real(0.0);
    }

    [[nodiscard]] Real evaluateCellTotal(const assembly::AssemblyContext& /*ctx*/) override
    {
        throw std::runtime_error("ThrowingTotalKernel: interpreter fallback was called for evaluateCellTotal()");
    }

    [[nodiscard]] Real evaluateBoundaryFaceTotal(const assembly::AssemblyContext& /*ctx*/,
                                                 int /*boundary_marker*/) override
    {
        throw std::runtime_error("ThrowingTotalKernel: interpreter fallback was called for evaluateBoundaryFaceTotal()");
    }

    [[nodiscard]] std::string name() const override { return "ThrowingTotalKernel"; }

private:
    assembly::RequiredData required_{assembly::RequiredData::None};
    std::vector<assembly::FieldRequirement> field_requirements_{};
    bool has_cell_{true};
    bool has_boundary_{false};
};

[[nodiscard]] inline std::shared_ptr<ThrowingTotalKernel> makeThrowingTotalKernelFor(
    const FormExpr& integrand,
    bool has_cell,
    bool has_boundary)
{
    FE_CHECK_NOT_NULL(integrand.node(), "makeThrowingTotalKernelFor: integrand node");
    return std::make_shared<ThrowingTotalKernel>(
        detail::analyzeRequiredData(*integrand.node(), FormKind::Linear),
        detail::analyzeFieldRequirements(*integrand.node()),
        has_cell,
        has_boundary);
}

[[nodiscard]] inline JITOptions makeUnitTestJITOptions()
{
    JITOptions opt;
    opt.enable = true;
    opt.optimization_level = 0;
    opt.cache_kernels = true;
    opt.vectorize = false;
    opt.cache_directory.clear();
    opt.cache_diagnostics = false;
    opt.max_in_memory_kernels = 0;
    opt.dump_kernel_ir = false;
    opt.dump_llvm_ir = false;
    opt.dump_llvm_ir_optimized = false;
    opt.debug_info = false;
    opt.dump_directory = "svmp_fe_jit_dumps_tests";
    opt.specialization.enable = false;
    return opt;
}

[[nodiscard]] inline std::shared_ptr<FunctionalFormKernel> makeFunctionalFormKernel(
    const FormExpr& integrand,
    FunctionalFormKernel::Domain domain)
{
    FE_CHECK_NOT_NULL(integrand.node(), "makeFunctionalFormKernel: integrand node");

    const auto required = detail::analyzeRequiredData(*integrand.node(), FormKind::Linear);
    auto field_reqs = detail::analyzeFieldRequirements(*integrand.node());
    return std::make_shared<FunctionalFormKernel>(integrand, domain, required, std::move(field_reqs));
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TESTS_UNIT_FORMS_JIT_TEST_HELPERS_H
