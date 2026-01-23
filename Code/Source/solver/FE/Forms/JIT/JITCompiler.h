#ifndef SVMP_FE_FORMS_JIT_JIT_COMPILER_H
#define SVMP_FE_FORMS_JIT_JIT_COMPILER_H

/**
 * @file JITCompiler.h
 * @brief Orchestration layer for compiling FormIR to future JIT kernels
 *
 * This header intentionally contains no LLVM dependencies.
 */

#include "Forms/FormIR.h"
#include "Forms/JIT/JITValidation.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

struct JITCompiledKernel {
    IntegralDomain domain{IntegralDomain::Cell};
    int boundary_marker{-1};
    int interface_marker{-1};

    std::uint64_t cache_key{0};
    bool cacheable{true};

    std::string symbol{};
    std::uintptr_t address{0};
};

struct JITCompileResult {
    bool ok{false};
    bool cacheable{true};
    std::string message{};
    std::vector<JITCompiledKernel> kernels{};
};

class JITCompiler final {
public:
    [[nodiscard]] static std::shared_ptr<JITCompiler> getOrCreate(const JITOptions& options);

    [[nodiscard]] JITCompileResult compile(const FormIR& ir,
                                           const ValidationOptions& validation = {});

    JITCompiler(const JITCompiler&) = delete;
    JITCompiler& operator=(const JITCompiler&) = delete;

private:
    explicit JITCompiler(JITOptions options);

    struct Impl;
    std::unique_ptr<Impl> impl_{};
};

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_JIT_COMPILER_H

