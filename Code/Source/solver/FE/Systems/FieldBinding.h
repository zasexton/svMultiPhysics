/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FIELDBINDING_H
#define SVMP_FE_SYSTEMS_FIELDBINDING_H

/**
 * @file FieldBinding.h
 * @brief Registration or reuse of the fields of one module, with checks
 *
 * A module registers its unknowns and its prescribed data through one
 * FieldBinding.  A field of the same name that is already registered (by
 * another module or an earlier call) is reused when its source kind,
 * component count and function space agree; otherwise the call throws
 * std::invalid_argument.  Fields are added in call order, which is the DOF
 * order.  Every message starts with "<owner> <role>: ", or with the context
 * passed to check().
 */

#include "Core/Types.h"
#include "Systems/FieldRegistry.h"

#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

namespace svmp {
namespace FE {
namespace spaces {
class FunctionSpace;
} // namespace spaces

namespace systems {

class FESystem;

/// Same space type, field type, value and topological dimension, polynomial
/// order and element type.
[[nodiscard]] bool spacesCompatible(const spaces::FunctionSpace& lhs,
                                    const spaces::FunctionSpace& rhs) noexcept;

class FieldBinding {
public:
    /// @p owner prefixes the messages, e.g. "MyModule::registerOn".
    FieldBinding(FESystem& system, std::string owner);

    /**
     * @brief Checks @p spec against the registered field of the same name
     * without changing the system.
     *
     * Throws when the name is empty, the spec has no space, the field is
     * missing and @p allow_missing is false, or the registered field differs
     * in source kind, component count or space.
     */
    static void check(const FESystem& system,
                      const FieldSpec& spec,
                      FieldSourceKind expected_source,
                      bool allow_missing,
                      std::string_view context);

    /// An unknown field: registered, or the compatible registered one.
    [[nodiscard]] FieldId unknown(FieldSpec spec, std::string_view role);

    /// Prescribed data: registered when missing and @p auto_register is
    /// true, otherwise the compatible registered field.
    [[nodiscard]] FieldId prescribed(FieldSpec spec, bool auto_register, std::string_view role);

    /**
     * @brief Throws "<owner>: field name '<name>' is assigned to both <a> and
     * <b>" when two roles name the same field; empty names are skipped.
     */
    static void requireDistinctNames(
        std::string_view owner,
        std::initializer_list<std::pair<std::string_view, std::string_view>> role_names);

private:
    [[nodiscard]] std::string roleContext(std::string_view role) const;

    FESystem* system_{nullptr};
    std::string owner_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FIELDBINDING_H
