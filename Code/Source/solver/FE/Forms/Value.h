#ifndef SVMP_FE_FORMS_VALUE_H
#define SVMP_FE_FORMS_VALUE_H

/**
 * @file Value.h
 * @brief Common scalar/vector/matrix value container used by FE/Forms evaluation
 */

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

template<typename T>
struct Value {
    enum class Kind : std::uint8_t {
        Scalar,
        Vector,
        Matrix,
        SymmetricMatrix,
        SkewMatrix,
        Tensor3,
        Tensor4
    };

    Kind kind{Kind::Scalar};
    T s{};
    std::array<T, 3> v{};
    std::array<std::array<T, 3>, 3> m{};
    std::array<T, 27> t3{};
    std::array<T, 81> t4{};

    // Optional dynamic payloads for shapes that exceed the inline storage.
    std::vector<T> v_dyn{};
    std::vector<T> m_dyn{};
    std::vector<T> t3_dyn{};

    // Optional shape metadata. When unset (== 0), callers may assume the legacy
    // 3-component / 3×3 conventions.
    int vector_size{0};
    int matrix_rows{0};
    int matrix_cols{0};
    int tensor3_dim0{0};
    int tensor3_dim1{0};
    int tensor3_dim2{0};

    [[nodiscard]] std::size_t vectorSize() const noexcept
    {
        if (kind != Kind::Vector) return 0u;
        if (vector_size > 0) return static_cast<std::size_t>(vector_size);
        return v_dyn.empty() ? 3u : v_dyn.size();
    }

    [[nodiscard]] std::size_t matrixRows() const noexcept
    {
        if (kind != Kind::Matrix && kind != Kind::SymmetricMatrix && kind != Kind::SkewMatrix) return 0u;
        if (matrix_rows > 0) return static_cast<std::size_t>(matrix_rows);
        return 3u;
    }

    [[nodiscard]] std::size_t matrixCols() const noexcept
    {
        if (kind != Kind::Matrix && kind != Kind::SymmetricMatrix && kind != Kind::SkewMatrix) return 0u;
        if (matrix_cols > 0) return static_cast<std::size_t>(matrix_cols);
        return 3u;
    }

    [[nodiscard]] std::size_t tensor3Dim0() const noexcept
    {
        if (kind != Kind::Tensor3) return 0u;
        if (tensor3_dim0 > 0) return static_cast<std::size_t>(tensor3_dim0);
        return 3u;
    }

    [[nodiscard]] std::size_t tensor3Dim1() const noexcept
    {
        if (kind != Kind::Tensor3) return 0u;
        if (tensor3_dim1 > 0) return static_cast<std::size_t>(tensor3_dim1);
        return 3u;
    }

    [[nodiscard]] std::size_t tensor3Dim2() const noexcept
    {
        if (kind != Kind::Tensor3) return 0u;
        if (tensor3_dim2 > 0) return static_cast<std::size_t>(tensor3_dim2);
        return 3u;
    }

    [[nodiscard]] std::size_t tensor3Size() const noexcept
    {
        if (kind != Kind::Tensor3) return 0u;
        return tensor3Dim0() * tensor3Dim1() * tensor3Dim2();
    }

    void resizeVector(std::size_t n)
    {
        vector_size = static_cast<int>(n);
        if (n <= 3u) {
            v_dyn.clear();
            v = {};
            return;
        }
        v_dyn.assign(n, T{});
        v = {};
    }

    void resizeMatrix(std::size_t rows, std::size_t cols)
    {
        matrix_rows = static_cast<int>(rows);
        matrix_cols = static_cast<int>(cols);
        if (rows <= 3u && cols <= 3u) {
            m_dyn.clear();
            m = {};
            return;
        }
        m_dyn.assign(rows * cols, T{});
        m = {};
    }

    void resizeTensor3(std::size_t d0, std::size_t d1, std::size_t d2)
    {
        tensor3_dim0 = static_cast<int>(d0);
        tensor3_dim1 = static_cast<int>(d1);
        tensor3_dim2 = static_cast<int>(d2);

        if (d0 <= 3u && d1 <= 3u && d2 <= 3u) {
            t3_dyn.clear();
            t3 = {};
            return;
        }
        t3_dyn.assign(d0 * d1 * d2, T{});
        t3 = {};
    }

    [[nodiscard]] std::span<T> vectorSpan()
    {
        const auto n = vectorSize();
        if (n <= 3u) {
            return {v.data(), n};
        }
        return {v_dyn.data(), n};
    }

    [[nodiscard]] std::span<const T> vectorSpan() const
    {
        const auto n = vectorSize();
        if (n <= 3u) {
            return {v.data(), n};
        }
        return {v_dyn.data(), n};
    }

    [[nodiscard]] T& vectorAt(std::size_t i)
    {
        return vectorSpan()[i];
    }

    [[nodiscard]] const T& vectorAt(std::size_t i) const
    {
        return vectorSpan()[i];
    }

    [[nodiscard]] T& matrixAt(std::size_t r, std::size_t c)
    {
        if (matrix_rows <= 3 && matrix_cols <= 3 && m_dyn.empty()) {
            return m[r][c];
        }
        return m_dyn[r * static_cast<std::size_t>(matrix_cols) + c];
    }

    [[nodiscard]] const T& matrixAt(std::size_t r, std::size_t c) const
    {
        if (matrix_rows <= 3 && matrix_cols <= 3 && m_dyn.empty()) {
            return m[r][c];
        }
        return m_dyn[r * static_cast<std::size_t>(matrix_cols) + c];
    }

    [[nodiscard]] T& tensor3At(std::size_t i, std::size_t j, std::size_t k)
    {
        if (tensor3_dim0 <= 3 && tensor3_dim1 <= 3 && tensor3_dim2 <= 3 && t3_dyn.empty()) {
            return t3[(i * 3u + j) * 3u + k];
        }
        return t3_dyn[(i * static_cast<std::size_t>(tensor3_dim1) + j) * static_cast<std::size_t>(tensor3_dim2) + k];
    }

    [[nodiscard]] const T& tensor3At(std::size_t i, std::size_t j, std::size_t k) const
    {
        if (tensor3_dim0 <= 3 && tensor3_dim1 <= 3 && tensor3_dim2 <= 3 && t3_dyn.empty()) {
            return t3[(i * 3u + j) * 3u + k];
        }
        return t3_dyn[(i * static_cast<std::size_t>(tensor3_dim1) + j) * static_cast<std::size_t>(tensor3_dim2) + k];
    }
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_VALUE_H
