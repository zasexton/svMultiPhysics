/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

#include "Core/FEException.h"
#include "Forms/FormExpr.h"

namespace svmp::FE::forms::detail {

inline std::optional<FormExpr> bindNamedParameter(
    const FormExprNode& node,
    const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of)
{
    if (node.type() != FormExprType::ParameterSymbol && node.type() != FormExprType::ParameterRef) {
        return std::nullopt;
    }
    const auto name = node.symbolName();
    if (!name) return std::nullopt;  // Explicit positional reference.
    const auto slot = slot_of(*name);
    FE_THROW_IF(!slot, InvalidArgumentException,
                "Forms: could not resolve parameter slot for '" + std::string(*name) + "'");
    if (node.slotIndex() == slot) return std::nullopt;
    return FormExpr::boundParameter(std::string(*name), *slot);
}

} // namespace svmp::FE::forms::detail
