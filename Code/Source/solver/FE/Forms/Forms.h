#ifndef SVMP_FE_FORMS_FORMS_H
#define SVMP_FE_FORMS_FORMS_H

/**
 * @file Forms.h
 * @brief Umbrella header — includes the full FE/Forms surface
 *
 * @note **Not recommended as a starting include for physics modules.**
 * Most formulation code needs only:
 *   - `Forms/Vocabulary.h` (expression helpers: stateField, testFunction, grad, ...)
 *   - `Systems/FormsInstaller.h` (installFormulation, installStrongDirichlet)
 *
 * This umbrella pulls in expert/internal headers (BlockForm, Complex,
 * FormKernels, FormCompiler) that are not needed for standard workflows.
 */

// Core authoring surface
#include "Forms/FormExpr.h"
#include "Forms/Vocabulary.h"
#include "Forms/BoundaryConditions.h"
#include "Forms/InterfaceConditions.h"
#include "Forms/BoundaryFunctional.h"
#include "Forms/ConstitutiveModel.h"

// Expert/advanced headers — include directly when needed:
//   #include "Forms/FormCompiler.h"   // explicit compilation
//   #include "Forms/FormKernels.h"    // kernel internals
//   #include "Forms/BlockForm.h"      // manual block decomposition
//   #include "Forms/Complex.h"        // complex-valued PDE adapter
//   #include "Forms/Index.h"          // Einstein index notation
//   #include "Forms/Einsum.h"         // einsum lowering

#endif // SVMP_FE_FORMS_FORMS_H
