#ifndef SVMP_FE_FORMS_JIT_JIT_ENGINE_H
#define SVMP_FE_FORMS_JIT_JIT_ENGINE_H

/**
 * @file JITEngine.h
 * @brief Thin wrapper around LLVM OrcJIT (LLJIT) runtime
 *
 * This header intentionally avoids including LLVM headers to keep LLVM
 * dependencies isolated to the Forms/JIT implementation.
 */

#include "Forms/FormExpr.h"
#include "Forms/JIT/JITCacheStats.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace llvm {
namespace orc {
class ThreadSafeModule;
} // namespace orc
} // namespace llvm

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

class JITEngine final {
public:
    using SymbolAddress = std::uintptr_t;

    /**
     * @brief Create a JITEngine instance when LLVM support is available.
     *
     * Returns nullptr when FE was built without LLVM JIT support or when the
     * engine cannot be created at runtime.
     */
    [[nodiscard]] static std::unique_ptr<JITEngine> create(const JITOptions& options);

    ~JITEngine();

    JITEngine(const JITEngine&) = delete;
    JITEngine& operator=(const JITEngine&) = delete;

    [[nodiscard]] bool available() const noexcept;

    void addModule(llvm::orc::ThreadSafeModule&& module);

    [[nodiscard]] SymbolAddress lookup(std::string_view name);
    [[nodiscard]] bool tryLookup(std::string_view name, SymbolAddress& out) noexcept;

    template <typename Fn>
    [[nodiscard]] Fn lookupAs(std::string_view name)
    {
        return reinterpret_cast<Fn>(lookup(name));
    }

    [[nodiscard]] std::string targetTriple() const;
    [[nodiscard]] std::string dataLayoutString() const;
    [[nodiscard]] std::string cpuName() const;
    [[nodiscard]] std::string cpuFeaturesString() const;
    [[nodiscard]] JITObjectCacheStats objectCacheStats() const;
    void resetObjectCacheStats();

private:
    JITEngine() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_{};
    mutable std::mutex mutex_{};
};

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_JIT_ENGINE_H
