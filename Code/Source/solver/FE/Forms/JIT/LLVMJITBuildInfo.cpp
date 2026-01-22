/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/LLVMJITBuildInfo.h"

#include <string>

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

#if SVMP_FE_ENABLE_LLVM_JIT
#include <llvm/Config/llvm-config.h>
#endif

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

bool llvmJITEnabled() noexcept
{
    return SVMP_FE_ENABLE_LLVM_JIT != 0;
}

std::string llvmVersionString()
{
#if SVMP_FE_ENABLE_LLVM_JIT
    return LLVM_VERSION_STRING;
#else
    return {};
#endif
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

