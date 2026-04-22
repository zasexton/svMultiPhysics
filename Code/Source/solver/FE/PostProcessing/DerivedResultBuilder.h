#ifndef SVMP_FE_POSTPROCESSING_DERIVED_RESULT_BUILDER_H
#define SVMP_FE_POSTPROCESSING_DERIVED_RESULT_BUILDER_H

/**
 * @file DerivedResultBuilder.h
 * @brief Fluent C++ builder for physics modules registering derived results.
 */

#include "Core/FEException.h"
#include "PostProcessing/DerivedResultTypes.h"

#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace post {

class DerivedResultBuilder {
public:
    explicit DerivedResultBuilder(std::string name)
    {
        def_.name = std::move(name);
    }

    DerivedResultBuilder& scope(DerivedResultScope value) noexcept
    {
        def_.scope = value;
        return *this;
    }

    DerivedResultBuilder& policy(DerivedResultPolicy value) noexcept
    {
        def_.policy = value;
        return *this;
    }

    DerivedResultBuilder& shape(systems::FEQuantityShape value) noexcept
    {
        def_.shape = value;
        shape_set_ = true;
        return *this;
    }

    DerivedResultBuilder& expression(forms::FormExpr value)
    {
        def_.expression = std::move(value);
        return *this;
    }

    DerivedResultBuilder& referencedField(FieldId field)
    {
        def_.referenced_fields.push_back(field);
        return *this;
    }

    DerivedResultBuilder& referencedFields(std::vector<FieldId> fields)
    {
        def_.referenced_fields = std::move(fields);
        return *this;
    }

    DerivedResultBuilder& marker(int value)
    {
        def_.marker = value;
        return *this;
    }

    DerivedResultBuilder& enabled(bool value) noexcept
    {
        def_.enabled = value;
        return *this;
    }

    [[nodiscard]] DerivedResultDefinition build() const
    {
        FE_THROW_IF(def_.name.empty(), InvalidArgumentException,
                    "DerivedResultBuilder: derived result name must not be empty");
        FE_THROW_IF(!shape_set_, InvalidArgumentException,
                    "DerivedResultBuilder('" + def_.name + "'): shape must be specified");
        FE_THROW_IF(!def_.expression.isValid(), InvalidArgumentException,
                    "DerivedResultBuilder('" + def_.name + "'): expression must be valid");
        return def_;
    }

private:
    DerivedResultDefinition def_{};
    bool shape_set_{false};
};

} // namespace post
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_POSTPROCESSING_DERIVED_RESULT_BUILDER_H
