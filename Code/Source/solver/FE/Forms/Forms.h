#ifndef SVMP_FE_FORMS_FORMS_H
#define SVMP_FE_FORMS_FORMS_H

/**
 * @file Forms.h
 * @brief Core FE/Forms authoring surface
 *
 * @note A formulation module includes `Forms/Authoring.h`, the form
 * language together with the FE services of the canonical skeleton
 * (Forms/SYSTEMS_INTEGRATION.md section 2).  This header is the core form
 * language only.
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
