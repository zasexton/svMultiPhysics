/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_CONCURRENTCOMPUTE_H
#define SVMP_FE_ASSEMBLY_CONCURRENTCOMPUTE_H

/**
 * @file ConcurrentCompute.h
 * @brief "No lazy work" mode of the threaded assembly compute phase.
 *
 * Kernels and assemblers do some work once, on first use: JIT compiles of
 * generic and specialized kernels, the interpreter's lowering of indexed
 * access in the form IR, the switch to the interpreter after a JIT runtime
 * failure, and the first build of the assembler's shared tables. Such work is
 * not safe to run on several threads at once, and its outcome may depend on
 * the order in which it happens (the JIT's code generation depends on earlier
 * compiles). While a ConcurrentComputeScope is active on a thread, code that
 * would do such work calls requireSerial(), which throws DeferredSerialWork;
 * the threaded assembly then discards its attempt and runs the loop serially,
 * so the one-time work happens in the serial order.
 * See FE/Docs/ThreadedAssembly.md.
 */

#include <stdexcept>
#include <string>

namespace svmp {
namespace FE {
namespace assembly {

/// Thrown on an assembly thread when lazy one-time work would run there.
class DeferredSerialWork : public std::runtime_error {
public:
    explicit DeferredSerialWork(const std::string& what)
        : std::runtime_error("deferred to serial assembly: " + what)
    {
    }
};

/// True while a ConcurrentComputeScope is active on the calling thread.
[[nodiscard]] bool concurrentComputeActive() noexcept;

/// Throws DeferredSerialWork(what) when concurrentComputeActive().
void requireSerial(const char* what);

/// Activates the "no lazy work" mode on the calling thread for its lifetime.
class ConcurrentComputeScope {
public:
    ConcurrentComputeScope() noexcept;
    ~ConcurrentComputeScope();
    ConcurrentComputeScope(const ConcurrentComputeScope&) = delete;
    ConcurrentComputeScope& operator=(const ConcurrentComputeScope&) = delete;

private:
    bool previous_;
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_CONCURRENTCOMPUTE_H
