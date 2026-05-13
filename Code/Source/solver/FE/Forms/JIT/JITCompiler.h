#ifndef SVMP_FE_FORMS_JIT_JIT_COMPILER_H
#define SVMP_FE_FORMS_JIT_JIT_COMPILER_H

/**
 * @file JITCompiler.h
 * @brief Orchestration layer for compiling FormIR to the production LLVM JIT kernels
 *
 * This header intentionally contains no LLVM dependencies.
 */

#include "Forms/FormIR.h"
#include "Forms/JIT/JITCacheStats.h"
#include "Forms/JIT/JITSpecialization.h"
#include "Forms/JIT/JITValidation.h"

#include <cstdint>
#include <memory>
#include <span>
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
    CutVolumeSide cut_volume_side{CutVolumeSide::Negative};

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

    struct MonolithicBlockSpec {
        const FormIR* tangent_ir{nullptr};   // Bilinear form (matrix), may be nullptr
        const FormIR* residual_ir{nullptr};  // Residual form (vector), may be nullptr
        bool want_matrix{false};
        bool want_vector{false};
    };

    /** Compile a monolithic coupled kernel that evaluates all blocks in a
     *  single pass with shared geometry and QP-level intermediates.
     *  Uses the CoupledCellKernelArgsV1 ABI. Falls back gracefully if
     *  monolithic codegen is not available. */
    [[nodiscard]] JITCompileResult compileMonolithic(
        std::span<const MonolithicBlockSpec> blocks,
        const ValidationOptions& validation = {});

    struct ColocatedKernelSpec {
        const FormIR* ir{nullptr};
        std::vector<std::size_t> term_indices{};
        IntegralDomain domain{IntegralDomain::Cell};
        int boundary_marker{-1};
        int interface_marker{-1};
        const JITCompileSpecialization* specialization{nullptr};
    };

    struct ColocatedKernelResult {
        std::string symbol{};
        std::uintptr_t address{0};
    };

    /** Compile multiple kernels into a single LLVM module for contiguous
     *  .text layout. Returns per-kernel resolved addresses. */
    [[nodiscard]] JITCompileResult compileColocated(
        std::span<const ColocatedKernelSpec> specs,
        std::vector<ColocatedKernelResult>& out_results);

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
