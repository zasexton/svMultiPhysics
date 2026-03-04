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

// Opaque type for passing pre-lowered fused terms from compileAndAddFusedKernel
// to the shared codegen path. Defined in LLVMGen.cpp.
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

    /** Compile a fused tangent+residual kernel that computes both element
     *  matrix (from tangent_ir) and element vector (from residual_ir) in a
     *  single QP loop pass. Falls back gracefully — if fused compilation
     *  fails, JITKernelWrapper retains separate tangent/residual kernels. */
    [[nodiscard]] LLVMGenResult compileAndAddFusedKernel(JITEngine& engine,
                                                         const FormIR& tangent_ir,
                                                         std::span<const std::size_t> tangent_indices,
                                                         const FormIR& residual_ir,
                                                         std::span<const std::size_t> residual_indices,
                                                         IntegralDomain domain,
                                                         int boundary_marker,
                                                         int interface_marker,
                                                         std::string_view symbol,
                                                         std::uintptr_t& out_address) const;

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
                                                        LLVMGenFusedInfo* fused) const;

    JITOptions options_{};
};

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_LLVM_GEN_H
