/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_COPYONWRITEVECTOR_H
#define SVMP_FE_INTERFACES_COPYONWRITEVECTOR_H

/**
 * @file CopyOnWriteVector.h
 * @brief Immutable-by-default vector whose copies share their elements.
 */

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace svmp::FE::interfaces {

/**
 * Vector storage shared between copies until one of them is modified.
 *
 * Generated cut geometry is copied between the lifecycle caches, the
 * returned domain, and every snapshot built from it, while the per-region
 * quadrature points never change after construction.  Copies of this type
 * share one element array; reads see the same elements as a std::vector
 * would, and mutation goes through mutate() (or the vector-like modifiers),
 * which first gives this copy its own array if the current one is shared.
 * Const iteration therefore never copies, and an accidental write through a
 * const iterator is a compile error.
 */
template <typename T>
class CopyOnWriteVector {
public:
    using value_type = T;
    using size_type = std::size_t;
    using const_iterator = typename std::vector<T>::const_iterator;
    using iterator = const_iterator;
    using const_reference = const T&;
    using reference = const T&;

    CopyOnWriteVector() = default;

    // NOLINTNEXTLINE(google-explicit-constructor): vector-compatible.
    CopyOnWriteVector(std::vector<T> values)
        : data_(values.empty()
                    ? nullptr
                    : std::make_shared<std::vector<T>>(std::move(values)))
    {
    }

    CopyOnWriteVector(std::initializer_list<T> values)
        : CopyOnWriteVector(std::vector<T>(values))
    {
    }

    CopyOnWriteVector& operator=(std::vector<T> values)
    {
        *this = CopyOnWriteVector(std::move(values));
        return *this;
    }

    CopyOnWriteVector& operator=(std::initializer_list<T> values)
    {
        *this = CopyOnWriteVector(values);
        return *this;
    }

    [[nodiscard]] size_type size() const noexcept
    {
        return data_ ? data_->size() : 0u;
    }
    [[nodiscard]] bool empty() const noexcept { return size() == 0u; }
    [[nodiscard]] size_type capacity() const noexcept
    {
        return data_ ? data_->capacity() : 0u;
    }

    [[nodiscard]] const_iterator begin() const noexcept
    {
        return values().cbegin();
    }
    [[nodiscard]] const_iterator end() const noexcept
    {
        return values().cend();
    }
    [[nodiscard]] const_iterator cbegin() const noexcept { return begin(); }
    [[nodiscard]] const_iterator cend() const noexcept { return end(); }

    [[nodiscard]] const T& operator[](size_type index) const
    {
        return (*data_)[index];
    }
    [[nodiscard]] const T& front() const { return data_->front(); }
    [[nodiscard]] const T& back() const { return data_->back(); }
    [[nodiscard]] const T* data() const noexcept
    {
        return data_ ? data_->data() : nullptr;
    }

    /** The elements as a std::vector (no copy). */
    [[nodiscard]] const std::vector<T>& values() const noexcept
    {
        return data_ ? *data_ : emptyValues();
    }
    // NOLINTNEXTLINE(google-explicit-constructor): vector-compatible reads.
    operator const std::vector<T>&() const noexcept { return values(); }

    /** Number of copies sharing the element array (0 when empty). */
    [[nodiscard]] long shareCount() const noexcept
    {
        return data_ ? data_.use_count() : 0;
    }

    /** Own the element array (copying it if shared) and return it. */
    [[nodiscard]] std::vector<T>& mutate()
    {
        if (!data_) {
            data_ = std::make_shared<std::vector<T>>();
        } else if (data_.use_count() > 1) {
            data_ = std::make_shared<std::vector<T>>(*data_);
        }
        return *data_;
    }

    void clear() noexcept { data_.reset(); }
    void reserve(size_type count) { mutate().reserve(count); }
    void resize(size_type count) { mutate().resize(count); }
    void push_back(const T& value) { mutate().push_back(value); }
    void push_back(T&& value) { mutate().push_back(std::move(value)); }
    template <typename... Args>
    T& emplace_back(Args&&... args)
    {
        return mutate().emplace_back(std::forward<Args>(args)...);
    }

private:
    [[nodiscard]] static const std::vector<T>& emptyValues() noexcept
    {
        static const std::vector<T> empty{};
        return empty;
    }

    std::shared_ptr<std::vector<T>> data_{};
};

} // namespace svmp::FE::interfaces

#endif // SVMP_FE_INTERFACES_COPYONWRITEVECTOR_H
