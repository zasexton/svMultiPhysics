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

private:
    JITOptions options_{};
};

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_LLVM_GEN_H
