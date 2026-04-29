/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGFORMBUILDER_H
#define SVMP_FE_COUPLING_COUPLINGFORMBUILDER_H

/**
 * @file CouplingFormBuilder.h
 * @brief Adapter from coupling participant names to public Forms vocabulary.
 */

#include "Coupling/CouplingContext.h"
#include "Coupling/CouplingGeometryRequirements.h"
#include "Forms/FormExpr.h"
#include "Forms/Vocabulary.h"

#include <string_view>

namespace svmp {
namespace FE {
namespace coupling {

class CouplingFormBuilder {
public:
    explicit CouplingFormBuilder(const CouplingContext& context);

    [[nodiscard]] const CouplingContext& context() const noexcept;

private:
    const CouplingContext* context_{nullptr};
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGFORMBUILDER_H
