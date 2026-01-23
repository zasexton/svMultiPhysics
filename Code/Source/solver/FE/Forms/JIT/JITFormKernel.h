#ifndef SVMP_FE_FORMS_JIT_JIT_FORM_KERNEL_H
#define SVMP_FE_FORMS_JIT_JIT_FORM_KERNEL_H

/**
 * @file JITFormKernel.h
 * @brief Convenience alias for the LLVM-JIT executing AssemblyKernel wrapper
 *
 * The concrete implementation currently lives in `JITKernelWrapper`.
 */

#include "Forms/JIT/JITKernelWrapper.h"

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

using JITFormKernel = JITKernelWrapper;

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_JIT_FORM_KERNEL_H

