/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_CUTINTERFACEFIELDEVALUATION_H
#define SVMP_FE_INTERFACES_CUTINTERFACEFIELDEVALUATION_H

/**
 * @file CutInterfaceFieldEvaluation.h
 * @brief H1 field evaluation helpers for generated cut-interface fragments.
 */

#include "Core/Types.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#include <vector>

namespace svmp {
namespace FE {
namespace interfaces {

struct H1NodalFieldData {
    ElementType element_type{ElementType::Unknown};
    int components{1};
    std::vector<Real> nodal_values{};
};

struct CutInterfaceFieldValue {
    std::vector<Real> components{};
};

[[nodiscard]] std::vector<Real> linearH1ShapeValues(
    ElementType element_type,
    const std::array<Real, 3>& parent_coordinate);

[[nodiscard]] CutInterfaceFieldValue evaluateH1FieldValueAtPoint(
    const H1NodalFieldData& field,
    const std::array<Real, 3>& parent_coordinate);

[[nodiscard]] std::vector<CutInterfaceFieldValue> evaluateH1FieldValuesOnFragment(
    const H1NodalFieldData& field,
    const CutInterfaceFragment& fragment);

} // namespace interfaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_INTERFACES_CUTINTERFACEFIELDEVALUATION_H
