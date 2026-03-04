#ifndef SVMP_FE_FORMS_JIT_JIT_COMPILER_H
#define SVMP_FE_FORMS_JIT_JIT_COMPILER_H

/**
 * @file JITCompiler.h
 * @brief Orchestration layer for compiling FormIR to future JIT kernels
 *
 * This header intentionally contains no LLVM dependencies.
 */

#include "Forms/FormIR.h"
#include "Forms/JIT/JITCacheStats.h"
#include "Forms/JIT/JITSpecialization.h"
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

    [[nodiscard]] JITCompileResult compileSpecialized(const FormIR& ir,
                                                      const JITCompileSpecialization& specialization,
                                                      const ValidationOptions& validation = {});

    [[nodiscard]] JITCompileResult compileFunctional(const FormExpr& integrand,
                                                     IntegralDomain domain,
                                                     std::uint32_t dim_hint,
                                                     const ValidationOptions& validation = {});

    [[nodiscard]] JITCompileResult compileFunctional(const FormExpr& integrand,
                                                     IntegralDomain domain,
                                                     const ValidationOptions& validation = {});

    /** Compile a fused tangent+residual kernel. Returns a single Cell-domain
     *  kernel that computes both element matrix and element vector in one pass.
     *  Boundary/face terms are not fused and fall back to separate kernels. */
    [[nodiscard]] JITCompileResult compileFused(const FormIR& tangent_ir,
                                                const FormIR& residual_ir,
                                                const ValidationOptions& validation = {});

    [[nodiscard]] JITCacheStats cacheStats() const;
    void resetCacheStats();

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
