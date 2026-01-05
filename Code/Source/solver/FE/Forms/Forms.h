#ifndef SVMP_FE_FORMS_FORMS_H
#define SVMP_FE_FORMS_FORMS_H

/**
 * @file Forms.h
 * @brief Public umbrella header for the FE/Forms module
 *
 * FE/Forms provides a vocabulary for building weak forms and compiling them to
 * assembly kernels consumable by FE/Assembly and FE/Systems.
 */

#include "Forms/FormExpr.h"
#include "Forms/Index.h"
#include "Forms/Einsum.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/ConstitutiveModel.h"
#include "Forms/BlockForm.h"
#include "Forms/Complex.h"
#include "Forms/Vocabulary.h"

#endif // SVMP_FE_FORMS_FORMS_H
