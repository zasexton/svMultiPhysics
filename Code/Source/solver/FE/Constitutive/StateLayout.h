#ifndef SVMP_FE_CONSTITUTIVE_STATE_LAYOUT_H
#define SVMP_FE_CONSTITUTIVE_STATE_LAYOUT_H

#include "Core/FEException.h"

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace constitutive {

namespace detail {

[[nodiscard]] inline bool isPowerOfTwo(std::size_t x) noexcept
{
    return x != 0u && (x & (x - 1u)) == 0u;
}

[[nodiscard]] inline std::size_t alignUp(std::size_t value, std::size_t alignment)
{
    FE_THROW_IF(alignment == 0u, InvalidArgumentException, "alignUp: alignment must be non-zero");
    FE_THROW_IF(!isPowerOfTwo(alignment), InvalidArgumentException, "alignUp: alignment must be a power of two");
    return (value + alignment - 1u) & ~(alignment - 1u);
}

} // namespace detail

struct StateField {
    std::string name{};
    std::size_t offset_bytes{0};
    std::size_t size_bytes{0};
    std::size_t alignment{1};
};

class StateLayout {
public:
    StateLayout() = default;

    [[nodiscard]] static StateLayout raw(std::size_t bytes_per_point,
                                         std::size_t alignment = alignof(std::max_align_t),
                                         std::string name = {})
    {
        FE_THROW_IF(bytes_per_point == 0u, InvalidArgumentException, "StateLayout::raw: bytes_per_point must be > 0");
        FE_THROW_IF(alignment == 0u, InvalidArgumentException, "StateLayout::raw: alignment must be > 0");
        FE_THROW_IF(!detail::isPowerOfTwo(alignment), InvalidArgumentException,
                    "StateLayout::raw: alignment must be a power of two");

        StateLayout layout;
        layout.name_ = std::move(name);
        layout.bytes_per_point_ = bytes_per_point;
        layout.alignment_ = alignment;
        layout.stride_bytes_ = detail::alignUp(bytes_per_point, alignment);
        return layout;
    }

    [[nodiscard]] std::string_view name() const noexcept { return name_; }
    [[nodiscard]] std::size_t bytesPerPoint() const noexcept { return bytes_per_point_; }
    [[nodiscard]] std::size_t alignment() const noexcept { return alignment_; }
    [[nodiscard]] std::size_t strideBytes() const noexcept { return stride_bytes_; }
    [[nodiscard]] bool empty() const noexcept { return bytes_per_point_ == 0u; }

    [[nodiscard]] const std::vector<StateField>& fields() const noexcept { return fields_; }

private:
    friend class StateLayoutBuilder;

    std::string name_{};
    std::size_t bytes_per_point_{0};
    std::size_t alignment_{1};
    std::size_t stride_bytes_{0};
    std::vector<StateField> fields_{};
};

class StateLayoutBuilder {
public:
    explicit StateLayoutBuilder(std::string name = {}) : name_(std::move(name)) {}

    template <class T>
    StateLayoutBuilder& add(std::string field_name, std::size_t count = 1)
    {
        FE_THROW_IF(count == 0u, InvalidArgumentException, "StateLayoutBuilder::add: count must be > 0");
        constexpr std::size_t a = alignof(T);
        FE_THROW_IF(!detail::isPowerOfTwo(a), InvalidArgumentException, "StateLayoutBuilder::add: invalid alignment");

        offset_bytes_ = detail::alignUp(offset_bytes_, a);
        fields_.push_back(StateField{std::move(field_name),
                                     offset_bytes_,
                                     sizeof(T) * count,
                                     a});

        offset_bytes_ += sizeof(T) * count;
        alignment_ = std::max(alignment_, a);
        return *this;
    }

    [[nodiscard]] StateLayout build() const
    {
        FE_THROW_IF(offset_bytes_ == 0u, InvalidArgumentException, "StateLayoutBuilder::build: empty layout");

        StateLayout layout;
        layout.name_ = name_;
        layout.bytes_per_point_ = offset_bytes_;
        layout.alignment_ = alignment_;
        layout.stride_bytes_ = detail::alignUp(layout.bytes_per_point_, layout.alignment_);
        layout.fields_ = fields_;
        return layout;
    }

private:
    std::string name_{};
    std::size_t offset_bytes_{0};
    std::size_t alignment_{1};
    std::vector<StateField> fields_{};
};

} // namespace constitutive
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTITUTIVE_STATE_LAYOUT_H
