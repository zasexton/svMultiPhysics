#ifndef SVMP_FE_MATH_ROTATIONS_H
#define SVMP_FE_MATH_ROTATIONS_H

/**
 * @file Rotations.h
 * @brief Rotation matrices and coordinate transformations for FE computations
 *
 * This header provides 2D and 3D rotation representations and transformations
 * essential for anisotropic materials, coordinate system changes, and tensor
 * rotations. Supports multiple representations: rotation matrices, Euler angles,
 * axis-angle, and quaternions.
 */

#include "Vector.h"
#include "Matrix.h"
#include "Tensor.h"
#include "MathConstants.h"
#include "MathUtils.h"
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace svmp {
namespace FE {
namespace math {

/**
 * @brief 2D rotation representation and operations
 * @tparam T Floating-point type
 *
 * Provides efficient 2D rotation operations for vectors and tensors.
 * Stores rotation as sine and cosine to avoid repeated trigonometric computations.
 */
template<typename T>
class Rotation2D {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

private:
    T cos_theta_;  // Cosine of rotation angle
    T sin_theta_;  // Sine of rotation angle

public:
    /**
     * @brief Default constructor - identity rotation
     */
    Rotation2D() : cos_theta_(T(1)), sin_theta_(T(0)) {}

    /**
     * @brief Constructor from angle in radians
     * @param angle Rotation angle in radians (positive = counterclockwise)
     */
    explicit Rotation2D(T angle)
        : cos_theta_(std::cos(angle)), sin_theta_(std::sin(angle)) {}

    /**
     * @brief Constructor from cosine and sine
     * @param cos_theta Cosine of rotation angle
     * @param sin_theta Sine of rotation angle
     */
    Rotation2D(T cos_theta, T sin_theta)
        : cos_theta_(cos_theta), sin_theta_(sin_theta) {
        // Normalize to ensure cos^2 + sin^2 = 1
        T norm = std::sqrt(cos_theta * cos_theta + sin_theta * sin_theta);
        if (norm > epsilon<T>) {
            cos_theta_ /= norm;
            sin_theta_ /= norm;
        }
    }

    /**
     * @brief Create rotation from vector direction
     * @param v Direction vector (will be normalized)
     * @return Rotation that aligns x-axis with v
     */
    static Rotation2D from_vector(const Vector<T, 2>& v) {
        T norm = v.norm();
        if (norm < epsilon<T>) {
            return Rotation2D();  // Identity if vector is zero
        }
        return Rotation2D(v[0] / norm, v[1] / norm);
    }

    /**
     * @brief Get rotation angle in radians
     * @return Angle in range [-π, π]
     */
    T angle() const {
        return std::atan2(sin_theta_, cos_theta_);
    }

    /**
     * @brief Get cosine of rotation angle
     * @return cos(θ)
     */
    T cos() const { return cos_theta_; }

    /**
     * @brief Get sine of rotation angle
     * @return sin(θ)
     */
    T sin() const { return sin_theta_; }

    /**
     * @brief Convert to 2x2 rotation matrix
     * @return Rotation matrix R = [cos -sin; sin cos]
     */
    Matrix<T, 2, 2> matrix() const {
        Matrix<T, 2, 2> R;
        R(0, 0) = cos_theta_;
        R(0, 1) = -sin_theta_;
        R(1, 0) = sin_theta_;
        R(1, 1) = cos_theta_;
        return R;
    }

    /**
     * @brief Apply rotation to a vector
     * @param v Input vector
     * @return Rotated vector
     */
    Vector<T, 2> rotate(const Vector<T, 2>& v) const {
        Vector<T, 2> result;
        result[0] = cos_theta_ * v[0] - sin_theta_ * v[1];
        result[1] = sin_theta_ * v[0] + cos_theta_ * v[1];
        return result;
    }

    /**
     * @brief Apply inverse rotation to a vector
     * @param v Input vector
     * @return Inversely rotated vector
     */
    Vector<T, 2> rotate_inverse(const Vector<T, 2>& v) const {
        Vector<T, 2> result;
        result[0] = cos_theta_ * v[0] + sin_theta_ * v[1];
        result[1] = -sin_theta_ * v[0] + cos_theta_ * v[1];
        return result;
    }

    /**
     * @brief Rotate a rank-2 tensor: T' = R * T * R^T
     * @param tensor Input tensor
     * @return Rotated tensor
     */
    Tensor2<T, 2> rotate_tensor(const Tensor2<T, 2>& tensor) const {
        // T' = R * T * R^T
        Tensor2<T, 2> result;
        const T c = cos_theta_;
        const T s = sin_theta_;
        const T R[2][2] = {{c, -s}, {s, c}};

        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                T sum = T(0);
                for (std::size_t k = 0; k < 2; ++k) {
                    for (std::size_t l = 0; l < 2; ++l) {
                        sum += R[i][k] * tensor(k, l) * R[j][l];
                    }
                }
                result(i, j) = sum;
            }
        }

        return result;
    }

    /**
     * @brief Get inverse rotation
     * @return Inverse rotation (negative angle)
     */
    Rotation2D inverse() const {
        return Rotation2D(cos_theta_, -sin_theta_);
    }

    /**
     * @brief Compose two rotations
     * @param other Second rotation to apply
     * @return Combined rotation (this * other)
     */
    Rotation2D compose(const Rotation2D& other) const {
        T c = cos_theta_ * other.cos_theta_ - sin_theta_ * other.sin_theta_;
        T s = sin_theta_ * other.cos_theta_ + cos_theta_ * other.sin_theta_;
        return Rotation2D(c, s);
    }
};

/**
 * @brief 3D rotation representations and operations
 * @tparam T Floating-point type
 *
 * Provides multiple representations for 3D rotations: matrix, Euler angles,
 * axis-angle, and quaternions. Includes tensor rotation operations for
 * material property transformations.
 */
template<typename T>
class Rotation3D {
    static_assert(std::is_floating_point_v<T>, "T must be floating-point type");

private:
    Matrix<T, 3, 3> R_;  // Internal storage as rotation matrix

    /**
     * @brief Normalize rotation matrix to ensure orthogonality
     */
    void normalize_matrix() {
        // Use Gram-Schmidt to orthogonalize
        Vector<T, 3> x{R_(0, 0), R_(1, 0), R_(2, 0)};
        Vector<T, 3> y{R_(0, 1), R_(1, 1), R_(2, 1)};
        Vector<T, 3> z{R_(0, 2), R_(1, 2), R_(2, 2)};

        // Normalize x
        x = x.normalized();

        // Make y orthogonal to x and normalize
        y = y - x.dot(y) * x;
        y = y.normalized();

        // z = x × y
        z = x.cross(y);

        // Update matrix
        for (std::size_t i = 0; i < 3; ++i) {
            R_(i, 0) = x[i];
            R_(i, 1) = y[i];
            R_(i, 2) = z[i];
        }
    }

public:
    /**
     * @brief Default constructor - identity rotation
     */
    Rotation3D() : R_(Matrix<T, 3, 3>::identity()) {}

    /**
     * @brief Constructor from rotation matrix
     * @param R Rotation matrix (will be normalized)
     */
    explicit Rotation3D(const Matrix<T, 3, 3>& R) : R_(R) {
        normalize_matrix();
    }

    /**
     * @brief Create rotation from Euler angles (XYZ / roll-pitch-yaw)
     * @param phi Rotation about x-axis (roll)
     * @param theta Rotation about y-axis (pitch)
     * @param psi Rotation about z-axis (yaw)
     * @return Rotation object
     *
     * Applies rotations in order: R = Rx(phi) * Ry(theta) * Rz(psi)
     */
    static Rotation3D from_euler(T phi, T theta, T psi) {
        T c1 = std::cos(phi), s1 = std::sin(phi);
        T c2 = std::cos(theta), s2 = std::sin(theta);
        T c3 = std::cos(psi), s3 = std::sin(psi);

        Matrix<T, 3, 3> R;
        R(0, 0) = c2 * c3;
        R(0, 1) = -c2 * s3;
        R(0, 2) = s2;
        R(1, 0) = s1 * s2 * c3 + c1 * s3;
        R(1, 1) = -s1 * s2 * s3 + c1 * c3;
        R(1, 2) = -s1 * c2;
        R(2, 0) = -c1 * s2 * c3 + s1 * s3;
        R(2, 1) = c1 * s2 * s3 + s1 * c3;
        R(2, 2) = c1 * c2;

        return Rotation3D(R);
    }

    /**
     * @brief Create rotation from axis-angle representation
     * @param axis Rotation axis (will be normalized)
     * @param angle Rotation angle in radians
     * @return Rotation object
     *
     * Uses Rodrigues' rotation formula
     */
    static Rotation3D from_axis_angle(const Vector<T, 3>& axis, T angle) {
        Vector<T, 3> n = axis.normalized();
        T c = std::cos(angle);
        T s = std::sin(angle);
        T c1 = T(1) - c;

        Matrix<T, 3, 3> R;
        R(0, 0) = c + n[0] * n[0] * c1;
        R(0, 1) = n[0] * n[1] * c1 - n[2] * s;
        R(0, 2) = n[0] * n[2] * c1 + n[1] * s;
        R(1, 0) = n[1] * n[0] * c1 + n[2] * s;
        R(1, 1) = c + n[1] * n[1] * c1;
        R(1, 2) = n[1] * n[2] * c1 - n[0] * s;
        R(2, 0) = n[2] * n[0] * c1 - n[1] * s;
        R(2, 1) = n[2] * n[1] * c1 + n[0] * s;
        R(2, 2) = c + n[2] * n[2] * c1;

        return Rotation3D(R);
    }

    /**
     * @brief Create rotation from quaternion
     * @param q Quaternion [w, x, y, z] (will be normalized)
     * @return Rotation object
     */
    static Rotation3D from_quaternion(const Vector<T, 4>& q) {
        // Normalize quaternion
        Vector<T, 4> qn = q.normalized();
        T w = qn[0], x = qn[1], y = qn[2], z = qn[3];

        Matrix<T, 3, 3> R;
        R(0, 0) = T(1) - T(2)*(y*y + z*z);
        R(0, 1) = T(2)*(x*y - w*z);
        R(0, 2) = T(2)*(x*z + w*y);
        R(1, 0) = T(2)*(x*y + w*z);
        R(1, 1) = T(1) - T(2)*(x*x + z*z);
        R(1, 2) = T(2)*(y*z - w*x);
        R(2, 0) = T(2)*(x*z - w*y);
        R(2, 1) = T(2)*(y*z + w*x);
        R(2, 2) = T(1) - T(2)*(x*x + y*y);

        return Rotation3D(R);
    }

    /**
     * @brief Create rotation from two vectors
     * @param from Source vector
     * @param to Target vector
     * @return Rotation that rotates 'from' to 'to'
     */
    static Rotation3D from_two_vectors(const Vector<T, 3>& from, const Vector<T, 3>& to) {
        Vector<T, 3> f = from.normalized();
        Vector<T, 3> t = to.normalized();

        // Check for parallel vectors
        T dot = f.dot(t);
        if (dot > T(1) - epsilon<T>) {
            return Rotation3D();  // Identity
        }

        if (dot < -T(1) + epsilon<T>) {
            // Vectors are opposite - rotate 180 degrees about any perpendicular axis
            Vector<T, 3> axis = Vector<T, 3>{T(1), T(0), T(0)}.cross(f);
            if (axis.norm_squared() < epsilon<T>) {
                axis = Vector<T, 3>{T(0), T(1), T(0)}.cross(f);
            }
            return from_axis_angle(axis, pi<T>);
        }

        // General case - use cross product to find axis
        Vector<T, 3> axis = f.cross(t);
        T angle = std::acos(clamp(dot, T(-1), T(1)));
        return from_axis_angle(axis, angle);
    }

    /**
     * @brief Convert to rotation matrix
     * @return 3x3 rotation matrix
     */
    const Matrix<T, 3, 3>& matrix() const { return R_; }

    /**
     * @brief Convert to Euler angles (XYZ / roll-pitch-yaw)
     * @return Vector [phi (roll), theta (pitch), psi (yaw)] of Euler angles
     */
    Vector<T, 3> to_euler() const {
        Vector<T, 3> euler;

        // theta = asin(R(0,2))
        T sin_theta = R_(0, 2);
        if (std::abs(sin_theta) >= T(1) - epsilon<T>) {
            // Gimbal lock when cos(theta) ~ 0
            euler[1] = sin_theta > T(0) ? Constants<T>::half_pi : -Constants<T>::half_pi;
            euler[0] = std::atan2(R_(2, 0), R_(1, 0));
            euler[2] = T(0);
        } else {
            euler[1] = std::asin(sin_theta);
            euler[0] = std::atan2(-R_(1, 2), R_(2, 2));
            euler[2] = std::atan2(-R_(0, 1), R_(0, 0));
        }

        return euler;
    }

    /**
     * @brief Convert to axis-angle representation
     * @return Pair of (axis, angle)
     */
    std::pair<Vector<T, 3>, T> to_axis_angle() const {
        // Use the fact that R - R^T gives the axis direction
        T trace = R_(0, 0) + R_(1, 1) + R_(2, 2);
        T angle = std::acos(clamp((trace - T(1)) / T(2), T(-1), T(1)));

        if (angle < sqrt_epsilon<T>) {
            // Near identity - arbitrary axis
            return {Vector<T, 3>{T(1), T(0), T(0)}, T(0)};
        }

        if (std::abs(angle - pi<T>) < sqrt_epsilon<T>) {
            // 180 degree rotation: R = -I + 2*u*u^T. Use the diagonals to recover |u|.
            Vector<T, 3> axis;
            axis[0] = std::sqrt(std::max(T(0), (R_(0, 0) + T(1)) / T(2)));
            axis[1] = std::sqrt(std::max(T(0), (R_(1, 1) + T(1)) / T(2)));
            axis[2] = std::sqrt(std::max(T(0), (R_(2, 2) + T(1)) / T(2)));

            // Choose the largest component for stable reconstruction of the others.
            std::size_t i = 0;
            if (axis[1] > axis[0]) i = 1;
            if (axis[2] > axis[i]) i = 2;

            if (axis[i] > sqrt_epsilon<T>) {
                const T denom = T(4) * axis[i];
                if (i == 0) {
                    axis[1] = (R_(0, 1) + R_(1, 0)) / denom;
                    axis[2] = (R_(0, 2) + R_(2, 0)) / denom;
                } else if (i == 1) {
                    axis[0] = (R_(0, 1) + R_(1, 0)) / denom;
                    axis[2] = (R_(1, 2) + R_(2, 1)) / denom;
                } else {
                    axis[0] = (R_(0, 2) + R_(2, 0)) / denom;
                    axis[1] = (R_(1, 2) + R_(2, 1)) / denom;
                }
            } else {
                // Degenerate fallback (should not occur for a valid rotation matrix).
                axis = Vector<T, 3>{T(1), T(0), T(0)};
            }

            // Normalize to defend against small numerical drift.
            const T n = axis.norm();
            if (n > T(0)) {
                axis /= n;
            } else {
                axis = Vector<T, 3>{T(1), T(0), T(0)};
            }

            return {axis, angle};
        }

        // General case
        Vector<T, 3> axis;
        T factor = T(1) / (T(2) * std::sin(angle));
        axis[0] = (R_(2, 1) - R_(1, 2)) * factor;
        axis[1] = (R_(0, 2) - R_(2, 0)) * factor;
        axis[2] = (R_(1, 0) - R_(0, 1)) * factor;
        const T n = axis.norm();
        if (n > T(0)) {
            axis /= n;
        } else {
            axis = Vector<T, 3>{T(1), T(0), T(0)};
        }

        return {axis, angle};
    }

    /**
     * @brief Convert to quaternion
     * @return Quaternion [w, x, y, z]
     */
    Vector<T, 4> to_quaternion() const {
        Vector<T, 4> q;
        T trace = R_(0, 0) + R_(1, 1) + R_(2, 2);

        if (trace > T(0)) {
            T s = T(0.5) / std::sqrt(trace + T(1));
            q[0] = T(0.25) / s;
            q[1] = (R_(2, 1) - R_(1, 2)) * s;
            q[2] = (R_(0, 2) - R_(2, 0)) * s;
            q[3] = (R_(1, 0) - R_(0, 1)) * s;
        } else {
            if (R_(0, 0) > R_(1, 1) && R_(0, 0) > R_(2, 2)) {
                T s = T(2) * std::sqrt(T(1) + R_(0, 0) - R_(1, 1) - R_(2, 2));
                q[0] = (R_(2, 1) - R_(1, 2)) / s;
                q[1] = T(0.25) * s;
                q[2] = (R_(0, 1) + R_(1, 0)) / s;
                q[3] = (R_(0, 2) + R_(2, 0)) / s;
            } else if (R_(1, 1) > R_(2, 2)) {
                T s = T(2) * std::sqrt(T(1) + R_(1, 1) - R_(0, 0) - R_(2, 2));
                q[0] = (R_(0, 2) - R_(2, 0)) / s;
                q[1] = (R_(0, 1) + R_(1, 0)) / s;
                q[2] = T(0.25) * s;
                q[3] = (R_(1, 2) + R_(2, 1)) / s;
            } else {
                T s = T(2) * std::sqrt(T(1) + R_(2, 2) - R_(0, 0) - R_(1, 1));
                q[0] = (R_(1, 0) - R_(0, 1)) / s;
                q[1] = (R_(0, 2) + R_(2, 0)) / s;
                q[2] = (R_(1, 2) + R_(2, 1)) / s;
                q[3] = T(0.25) * s;
            }
        }

        return q;
    }

    /**
     * @brief Apply rotation to a vector
     * @param v Input vector
     * @return Rotated vector
     */
    Vector<T, 3> rotate(const Vector<T, 3>& v) const {
        return R_ * v;
    }

    /**
     * @brief Apply inverse rotation to a vector
     * @param v Input vector
     * @return Inversely rotated vector
     */
    Vector<T, 3> rotate_inverse(const Vector<T, 3>& v) const {
        // R^T * v (transpose = inverse for rotation matrices)
        Vector<T, 3> result;
        for (std::size_t i = 0; i < 3; ++i) {
            result[i] = R_(0, i) * v[0] + R_(1, i) * v[1] + R_(2, i) * v[2];
        }
        return result;
    }

    /**
     * @brief Rotate a rank-2 tensor: T' = R * T * R^T
     * @param tensor Input tensor
     * @return Rotated tensor
     */
    Tensor2<T, 3> rotate_tensor(const Tensor2<T, 3>& tensor) const {
        // T' = R * T * R^T
        Tensor2<T, 3> result;
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                T sum = T(0);
                for (std::size_t k = 0; k < 3; ++k) {
                    for (std::size_t l = 0; l < 3; ++l) {
                        sum += R_(i, k) * tensor(k, l) * R_(j, l);
                    }
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    /**
     * @brief Rotate a rank-4 tensor: C'_ijkl = R_im * R_jn * R_kp * R_lq * C_mnpq
     * @param tensor Input elasticity tensor
     * @return Rotated tensor
     */
    Tensor4<T, 3> rotate_tensor(const Tensor4<T, 3>& tensor) const {
        Tensor4<T, 3> result;
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t k = 0; k < 3; ++k) {
                    for (std::size_t l = 0; l < 3; ++l) {
                        T sum = T(0);
                        for (std::size_t m = 0; m < 3; ++m) {
                            for (std::size_t n = 0; n < 3; ++n) {
                                for (std::size_t p = 0; p < 3; ++p) {
                                    for (std::size_t q = 0; q < 3; ++q) {
                                        sum += R_(i, m) * R_(j, n) * R_(k, p) * R_(l, q) * tensor(m, n, p, q);
                                    }
                                }
                            }
                        }
                        result(i, j, k, l) = sum;
                    }
                }
            }
        }
        return result;
    }

    /**
     * @brief Get inverse rotation
     * @return Inverse rotation
     */
    Rotation3D inverse() const {
        return Rotation3D(R_.transpose());
    }

    /**
     * @brief Compose two rotations
     * @param other Second rotation to apply
     * @return Combined rotation (this * other)
     */
    Rotation3D compose(const Rotation3D& other) const {
        return Rotation3D(R_ * other.R_);
    }

    /**
     * @brief Spherical linear interpolation between two rotations
     * @param other Target rotation
     * @param t Interpolation parameter [0, 1]
     * @return Interpolated rotation
     */
    Rotation3D slerp(const Rotation3D& other, T t) const {
        // Convert to quaternions for interpolation
        Vector<T, 4> q1 = to_quaternion();
        Vector<T, 4> q2 = other.to_quaternion();

        // Ensure shortest path
        T dot = q1.dot(q2);
        if (dot < T(0)) {
            q2 = -q2;
            dot = -dot;
        }

        // Use linear interpolation for small angles
        if (dot > T(0.9995)) {
            Vector<T, 4> q = q1 * (T(1) - t) + q2 * t;
            return from_quaternion(q.normalized());
        }

        // Spherical interpolation
        T angle = std::acos(clamp(dot, T(-1), T(1)));
        T sin_angle = std::sin(angle);
        T w1 = std::sin((T(1) - t) * angle) / sin_angle;
        T w2 = std::sin(t * angle) / sin_angle;

        Vector<T, 4> q = q1 * w1 + q2 * w2;
        return from_quaternion(q);
    }
};

/**
 * @brief Build rotation matrix that aligns one vector to another
 * @tparam T Floating-point type
 * @param from Source vector
 * @param to Target vector
 * @return 3x3 rotation matrix
 */
template<typename T>
inline Matrix<T, 3, 3> rotation_between_vectors(const Vector<T, 3>& from,
                                                const Vector<T, 3>& to) {
    Rotation3D<T> rot = Rotation3D<T>::from_two_vectors(from, to);
    return rot.matrix();
}

/**
 * @brief Create rotation matrix for coordinate system alignment
 * @tparam T Floating-point type
 * @param x_new New x-axis direction
 * @param y_new New y-axis direction
 * @return 3x3 rotation matrix
 */
template<typename T>
inline Matrix<T, 3, 3> coordinate_system_rotation(const Vector<T, 3>& x_new,
                                                  const Vector<T, 3>& y_new) {
    Vector<T, 3> x = x_new.normalized();
    Vector<T, 3> y = y_new.normalized();
    Vector<T, 3> z = x.cross(y);

    // Re-orthogonalize y
    y = z.cross(x);

    Matrix<T, 3, 3> R;
    for (std::size_t i = 0; i < 3; ++i) {
        R(i, 0) = x[i];
        R(i, 1) = y[i];
        R(i, 2) = z[i];
    }

    return R;
}

// Type aliases for common usage
template<typename T> using Rot2D = Rotation2D<T>;
template<typename T> using Rot3D = Rotation3D<T>;

} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_ROTATIONS_H
