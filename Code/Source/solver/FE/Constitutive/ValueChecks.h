#ifndef SVMP_FE_CONSTITUTIVE_VALUECHECKS_H
#define SVMP_FE_CONSTITUTIVE_VALUECHECKS_H

#include "Core/FEException.h"
#include "Forms/Value.h"

#include <string>
#include <string_view>

namespace svmp {
namespace FE {
namespace constitutive {

template <class Scalar>
[[nodiscard]] inline std::string valueKindToString(typename forms::Value<Scalar>::Kind kind)
{
    using Kind = typename forms::Value<Scalar>::Kind;
    switch (kind) {
        case Kind::Scalar:
            return "Scalar";
        case Kind::Vector:
            return "Vector";
        case Kind::Matrix:
            return "Matrix";
        case Kind::SymmetricMatrix:
            return "SymmetricMatrix";
        case Kind::SkewMatrix:
            return "SkewMatrix";
        case Kind::Tensor4:
            return "Tensor4";
        default:
            return "Unknown";
    }
}

template <class Scalar>
inline void requireValueKind(const forms::Value<Scalar>& value,
                             typename forms::Value<Scalar>::Kind expected,
                             std::string_view where)
{
    FE_THROW_IF(value.kind != expected, InvalidArgumentException,
                std::string(where) + ": expected Value kind " + valueKindToString<Scalar>(expected) +
                    " but got " + valueKindToString<Scalar>(value.kind));
}

} // namespace constitutive
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTITUTIVE_VALUECHECKS_H

