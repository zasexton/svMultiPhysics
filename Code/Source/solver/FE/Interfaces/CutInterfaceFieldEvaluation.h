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
    std::vector<std::array<Real, 3>> node_coordinates{};
};

struct CutInterfaceFieldValue {
    std::vector<Real> components{};
};

struct CutInterfaceFieldGradient {
    std::vector<std::array<Real, 3>> components{};
};

struct CutInterfaceTwoSidedFieldValue {
    int interface_marker{-1};
    MeshIndex parent_cell{static_cast<MeshIndex>(-1)};
    std::uint64_t interface_stable_id{0};
    CutInterfaceSideTag minus_side{CutInterfaceSideTag::Negative};
    CutInterfaceSideTag plus_side{CutInterfaceSideTag::Positive};
    CutInterfaceFieldValue minus{};
    CutInterfaceFieldValue plus{};
    CutInterfaceFieldValue jump{};
    CutInterfaceFieldValue average{};
};

struct CutInterfaceTwoSidedFieldGradient {
    int interface_marker{-1};
    MeshIndex parent_cell{static_cast<MeshIndex>(-1)};
    std::uint64_t interface_stable_id{0};
    CutInterfaceSideTag minus_side{CutInterfaceSideTag::Negative};
    CutInterfaceSideTag plus_side{CutInterfaceSideTag::Positive};
    CutInterfaceFieldGradient minus{};
    CutInterfaceFieldGradient plus{};
    CutInterfaceFieldGradient jump{};
    CutInterfaceFieldGradient average{};
};

[[nodiscard]] std::vector<Real> linearH1ShapeValues(
    ElementType element_type,
    const std::array<Real, 3>& parent_coordinate);

[[nodiscard]] std::vector<std::array<Real, 3>> linearH1ShapeGradients(
    ElementType element_type,
    const std::array<Real, 3>& parent_coordinate);

[[nodiscard]] CutInterfaceFieldValue evaluateH1FieldValueAtPoint(
    const H1NodalFieldData& field,
    const std::array<Real, 3>& parent_coordinate);

[[nodiscard]] std::vector<CutInterfaceFieldValue> evaluateH1FieldValuesOnFragment(
    const H1NodalFieldData& field,
    const CutInterfaceFragment& fragment);

[[nodiscard]] CutInterfaceFieldGradient evaluateH1FieldGradientAtPoint(
    const H1NodalFieldData& field,
    const std::array<Real, 3>& parent_coordinate);

[[nodiscard]] std::vector<CutInterfaceFieldGradient> evaluateH1FieldGradientsOnFragment(
    const H1NodalFieldData& field,
    const CutInterfaceFragment& fragment);

[[nodiscard]] CutInterfaceTwoSidedFieldValue evaluateH1TwoSidedFieldValueAtPoint(
    const H1NodalFieldData& negative_side_field,
    const H1NodalFieldData& positive_side_field,
    const GeneratedInterfaceTwoSidedBinding& binding,
    const std::array<Real, 3>& parent_coordinate);

[[nodiscard]] std::vector<CutInterfaceTwoSidedFieldValue>
evaluateH1TwoSidedFieldValuesOnFragment(
    const H1NodalFieldData& negative_side_field,
    const H1NodalFieldData& positive_side_field,
    const GeneratedInterfaceTwoSidedBinding& binding,
    const CutInterfaceFragment& fragment);

[[nodiscard]] CutInterfaceTwoSidedFieldGradient
evaluateH1TwoSidedFieldGradientAtPoint(
    const H1NodalFieldData& negative_side_field,
    const H1NodalFieldData& positive_side_field,
    const GeneratedInterfaceTwoSidedBinding& binding,
    const std::array<Real, 3>& parent_coordinate);

[[nodiscard]] std::vector<CutInterfaceTwoSidedFieldGradient>
evaluateH1TwoSidedFieldGradientsOnFragment(
    const H1NodalFieldData& negative_side_field,
    const H1NodalFieldData& positive_side_field,
    const GeneratedInterfaceTwoSidedBinding& binding,
    const CutInterfaceFragment& fragment);

} // namespace interfaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_INTERFACES_CUTINTERFACEFIELDEVALUATION_H
