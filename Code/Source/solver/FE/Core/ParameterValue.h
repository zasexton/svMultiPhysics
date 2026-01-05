#ifndef SVMP_FE_CORE_PARAMETER_VALUE_H
#define SVMP_FE_CORE_PARAMETER_VALUE_H

#include "Core/Types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace svmp {
namespace FE {
namespace params {

using Vector3 = std::array<Real, 3>;
using Matrix3x3 = std::array<std::array<Real, 3>, 3>;

struct DenseMatrix {
    std::size_t rows{0};
    std::size_t cols{0};
    std::vector<Real> data{};

    [[nodiscard]] bool empty() const noexcept { return rows == 0u || cols == 0u; }
};

[[nodiscard]] inline bool operator==(const DenseMatrix& a, const DenseMatrix& b) noexcept
{
    return a.rows == b.rows && a.cols == b.cols && a.data == b.data;
}

[[nodiscard]] inline bool operator!=(const DenseMatrix& a, const DenseMatrix& b) noexcept
{
    return !(a == b);
}

struct Object {
    virtual ~Object() = default;
};

using ObjectPtr = std::shared_ptr<const Object>;

using Value = std::variant<Real, int, bool, Vector3, Matrix3x3, std::string, std::vector<Real>, DenseMatrix, ObjectPtr>;

enum class ValueType : std::uint8_t {
    Any,
    Real,
    Int,
    Bool,
    Vec3,
    Mat3x3,
    String,
    RealVector,
    DenseMatrix,
    Object,
};

[[nodiscard]] inline ValueType typeOf(const Value& v) noexcept
{
    if (std::holds_alternative<Real>(v)) return ValueType::Real;
    if (std::holds_alternative<int>(v)) return ValueType::Int;
    if (std::holds_alternative<bool>(v)) return ValueType::Bool;
    if (std::holds_alternative<Vector3>(v)) return ValueType::Vec3;
    if (std::holds_alternative<Matrix3x3>(v)) return ValueType::Mat3x3;
    if (std::holds_alternative<std::string>(v)) return ValueType::String;
    if (std::holds_alternative<std::vector<Real>>(v)) return ValueType::RealVector;
    if (std::holds_alternative<DenseMatrix>(v)) return ValueType::DenseMatrix;
    if (std::holds_alternative<ObjectPtr>(v)) return ValueType::Object;
    return ValueType::Any;
}

[[nodiscard]] inline std::string_view typeName(ValueType t) noexcept
{
    switch (t) {
        case ValueType::Any:
            return "Any";
        case ValueType::Real:
            return "Real";
        case ValueType::Int:
            return "Int";
        case ValueType::Bool:
            return "Bool";
        case ValueType::Vec3:
            return "Vector3";
        case ValueType::Mat3x3:
            return "Matrix3x3";
        case ValueType::String:
            return "String";
        case ValueType::RealVector:
            return "RealVector";
        case ValueType::DenseMatrix:
            return "DenseMatrix";
        case ValueType::Object:
            return "Object";
        default:
            return "Unknown";
    }
}

struct Spec {
    std::string key{};
    ValueType type{ValueType::Any};
    bool required{true};
    std::optional<Value> default_value{};
    std::string doc{};
};

template <class T>
[[nodiscard]] inline std::optional<T> get(const Value& v)
{
    if (const auto* p = std::get_if<T>(&v)) return *p;
    return std::nullopt;
}

} // namespace params
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CORE_PARAMETER_VALUE_H
