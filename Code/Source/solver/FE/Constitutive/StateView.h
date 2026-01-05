#ifndef SVMP_FE_CONSTITUTIVE_STATE_VIEW_H
#define SVMP_FE_CONSTITUTIVE_STATE_VIEW_H

#include "Core/FEException.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace svmp {
namespace FE {
namespace constitutive {

class StateView {
public:
    StateView() = default;
    explicit StateView(std::span<std::byte> bytes) : bytes_(bytes) {}

    [[nodiscard]] std::span<std::byte> bytes() const noexcept { return bytes_; }
    [[nodiscard]] bool empty() const noexcept { return bytes_.empty(); }
    [[nodiscard]] std::size_t sizeBytes() const noexcept { return bytes_.size(); }

    template <class T>
    [[nodiscard]] T& get(std::size_t offset_bytes = 0) const
    {
        static_assert(!std::is_reference_v<T>, "StateView::get<T>: T must not be a reference type");
        static_assert(!std::is_const_v<T>, "StateView::get<T>: T must not be const-qualified");

        FE_THROW_IF(offset_bytes + sizeof(T) > bytes_.size(), InvalidArgumentException,
                    "StateView::get: out of bounds");

        auto* ptr = bytes_.data() + offset_bytes;
        const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        FE_THROW_IF((addr % alignof(T)) != 0u, InvalidArgumentException,
                    "StateView::get: misaligned access");

        return *reinterpret_cast<T*>(ptr);
    }

    template <class T>
    [[nodiscard]] std::span<T> span(std::size_t offset_bytes, std::size_t count) const
    {
        static_assert(!std::is_reference_v<T>, "StateView::span<T>: T must not be a reference type");
        static_assert(!std::is_const_v<T>, "StateView::span<T>: T must not be const-qualified");

        const std::size_t nbytes = sizeof(T) * count;
        FE_THROW_IF(offset_bytes + nbytes > bytes_.size(), InvalidArgumentException,
                    "StateView::span: out of bounds");

        auto* ptr = bytes_.data() + offset_bytes;
        const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        FE_THROW_IF((addr % alignof(T)) != 0u, InvalidArgumentException,
                    "StateView::span: misaligned access");

        return {reinterpret_cast<T*>(ptr), count};
    }

private:
    std::span<std::byte> bytes_{};
};

class ConstStateView {
public:
    ConstStateView() = default;
    explicit ConstStateView(std::span<const std::byte> bytes) : bytes_(bytes) {}

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return bytes_; }
    [[nodiscard]] bool empty() const noexcept { return bytes_.empty(); }
    [[nodiscard]] std::size_t sizeBytes() const noexcept { return bytes_.size(); }

    template <class T>
    [[nodiscard]] const T& get(std::size_t offset_bytes = 0) const
    {
        static_assert(!std::is_reference_v<T>, "ConstStateView::get<T>: T must not be a reference type");

        FE_THROW_IF(offset_bytes + sizeof(T) > bytes_.size(), InvalidArgumentException,
                    "ConstStateView::get: out of bounds");

        auto* ptr = bytes_.data() + offset_bytes;
        const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        FE_THROW_IF((addr % alignof(T)) != 0u, InvalidArgumentException,
                    "ConstStateView::get: misaligned access");

        return *reinterpret_cast<const T*>(ptr);
    }

    template <class T>
    [[nodiscard]] std::span<const T> span(std::size_t offset_bytes, std::size_t count) const
    {
        static_assert(!std::is_reference_v<T>, "ConstStateView::span<T>: T must not be a reference type");

        const std::size_t nbytes = sizeof(T) * count;
        FE_THROW_IF(offset_bytes + nbytes > bytes_.size(), InvalidArgumentException,
                    "ConstStateView::span: out of bounds");

        auto* ptr = bytes_.data() + offset_bytes;
        const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        FE_THROW_IF((addr % alignof(T)) != 0u, InvalidArgumentException,
                    "ConstStateView::span: misaligned access");

        return {reinterpret_cast<const T*>(ptr), count};
    }

private:
    std::span<const std::byte> bytes_{};
};

} // namespace constitutive
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTITUTIVE_STATE_VIEW_H

