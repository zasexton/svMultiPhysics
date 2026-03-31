#ifndef SVMP_FE_FORMS_JIT_LLVM_GEN_H
#define SVMP_FE_FORMS_JIT_LLVM_GEN_H

/**
 * @file LLVMGen.h
 * @brief Lowering of deterministic KernelIR to LLVM IR modules
 *
 * This header contains no LLVM dependencies.
 */

#include "Forms/FormIR.h"
#include "Forms/JIT/JITEngine.h"
#include "Forms/JIT/JITSpecialization.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

struct LLVMGenResult {
    bool ok{false};
    std::string message{};
};

// Opaque type for optional internal codegen context shared with
// compileAndAddKernelImpl(). Defined in LLVMGen.cpp.
struct LLVMGenFusedInfo;

class LLVMGen final {
public:
    explicit LLVMGen(JITOptions options);

    LLVMGen(const LLVMGen&) = delete;
    LLVMGen& operator=(const LLVMGen&) = delete;

    [[nodiscard]] LLVMGenResult compileAndAddKernel(JITEngine& engine,
                                                    const FormIR& ir,
                                                    std::span<const std::size_t> term_indices,
                                                    IntegralDomain domain,
                                                    int boundary_marker,
                                                    int interface_marker,
                                                    std::string_view symbol,
                                                    std::uintptr_t& out_address,
                                                    const JITCompileSpecialization* specialization = nullptr) const;

    struct MonolithicBlockInfo {
        const FormIR* tangent_ir{nullptr};
        const FormIR* residual_ir{nullptr};
        bool want_matrix{false};
        bool want_vector{false};
    };

    /** Compile a monolithic coupled kernel that evaluates all NxM blocks
     *  in a single pass with shared geometry and QP-level intermediates.
     *  The kernel uses the CoupledCellKernelArgsV1 ABI. */
    [[nodiscard]] LLVMGenResult compileAndAddCoupledKernel(JITEngine& engine,
                                                           std::span<const MonolithicBlockInfo> blocks,
                                                           std::string_view symbol,
                                                           std::uintptr_t& out_address) const;

    struct ColocatedKernelSpec {
        const FormIR* ir{nullptr};
        std::vector<std::size_t> term_indices{};
        IntegralDomain domain{IntegralDomain::Cell};
        int boundary_marker{-1};
        int interface_marker{-1};
        std::string symbol{};
        const JITCompileSpecialization* specialization{nullptr};
    };

    struct ColocatedResult {
        std::string symbol{};
        std::uintptr_t address{0};
    };

    /** Compile multiple kernel functions into a single LLVM module for
     *  contiguous .text layout. Reduces L1i cache thrashing when cycling
     *  through multiple kernels per element in coupled assembly. */
    [[nodiscard]] LLVMGenResult compileAndAddColocatedKernels(
        JITEngine& engine,
        std::span<const ColocatedKernelSpec> specs,
        std::vector<ColocatedResult>& out_results) const;

private:
    [[nodiscard]] LLVMGenResult compileAndAddKernelImpl(JITEngine& engine,
                                                        const FormIR& ir,
                                                        std::span<const std::size_t> term_indices,
                                                        IntegralDomain domain,
                                                        int boundary_marker,
                                                        int interface_marker,
                                                        std::string_view symbol,
                                                        std::uintptr_t& out_address,
                                                        const JITCompileSpecialization* specialization,
                                                        LLVMGenFusedInfo* fused,
                                                        void* coupled_info = nullptr,
                                                        void* external_ctx = nullptr,
                                                        void* external_module = nullptr) const;

    JITOptions options_{};
};

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_LLVM_GEN_H
