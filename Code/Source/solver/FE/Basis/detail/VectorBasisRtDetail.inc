// Private implementation detail for VectorBasis.cpp.
// Included inside the basis anonymous namespace after the direct wedge/pyramid
// seed evaluators are declared.

// Flag to indicate direct construction is used (bypass modal approach)
constexpr bool USE_DIRECT_WEDGE_PYRAMID_CONSTRUCTION = true;

inline Real integral_monomial_1d(int p) {
    FE_CHECK_ARG(p >= 0, "integral_monomial_1d: negative power");
    if (p % 2 != 0) {
        return Real(0);
    }
    return Real(2) / static_cast<Real>(p + 1);
}

inline Real factorial_int(int n) {
    FE_CHECK_ARG(n >= 0, "factorial_int: negative argument");
    return std::tgamma(static_cast<Real>(n) + Real(1));
}

inline Real integral_triangle_monomial(int px, int py) {
    // Reference triangle: (0,0), (1,0), (0,1)
    FE_CHECK_ARG(px >= 0 && py >= 0, "integral_triangle_monomial: negative power");
    return factorial_int(px) * factorial_int(py) / factorial_int(px + py + 2);
}

inline Real integral_tetra_monomial(int px, int py, int pz) {
    // Reference tetrahedron: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    FE_CHECK_ARG(px >= 0 && py >= 0 && pz >= 0, "integral_tetra_monomial: negative power");
    return factorial_int(px) * factorial_int(py) * factorial_int(pz) / factorial_int(px + py + pz + 3);
}

inline Vec3 lerp(const Vec3& a, const Vec3& b, Real s) {
    // s in [-1,1], map to t in [0,1]
    const Real t = (s + Real(1)) * Real(0.5);
    return a * (Real(1) - t) + b * t;
}

inline Vec3 bilinear(const std::array<Vec3, 4>& v, Real u, Real w) {
    const Real N0 = Real(0.25) * (Real(1) - u) * (Real(1) - w);
    const Real N1 = Real(0.25) * (Real(1) + u) * (Real(1) - w);
    const Real N2 = Real(0.25) * (Real(1) + u) * (Real(1) + w);
    const Real N3 = Real(0.25) * (Real(1) - u) * (Real(1) + w);
    return v[0] * N0 + v[1] * N1 + v[2] * N2 + v[3] * N3;
}

inline Vec3 bilinear_du(const std::array<Vec3, 4>& v, Real u, Real w) {
    (void)u;
    const Real dN0 = -Real(0.25) * (Real(1) - w);
    const Real dN1 =  Real(0.25) * (Real(1) - w);
    const Real dN2 =  Real(0.25) * (Real(1) + w);
    const Real dN3 = -Real(0.25) * (Real(1) + w);
    return v[0] * dN0 + v[1] * dN1 + v[2] * dN2 + v[3] * dN3;
}

inline Vec3 bilinear_dw(const std::array<Vec3, 4>& v, Real u, Real w) {
    (void)w;
    const Real dN0 = -Real(0.25) * (Real(1) - u);
    const Real dN1 = -Real(0.25) * (Real(1) + u);
    const Real dN2 =  Real(0.25) * (Real(1) + u);
    const Real dN3 =  Real(0.25) * (Real(1) - u);
    return v[0] * dN0 + v[1] * dN1 + v[2] * dN2 + v[3] * dN3;
}

inline Vec3 cross3(const Vec3& a, const Vec3& b) {
    return Vec3{a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]};
}

inline Vec3 normalize3(const Vec3& v) {
    const Real n = v.norm();
    FE_CHECK_ARG(n > std::numeric_limits<Real>::epsilon(), "normalize3: zero-length vector");
    return v / n;
}

inline std::vector<Real> invert_dense_matrix(std::vector<Real> A, std::size_t n) {
    FE_CHECK_ARG(A.size() == n * n, "invert_dense_matrix: size mismatch");

    std::vector<Real> inv(n * n, Real(0));
    for (std::size_t i = 0; i < n; ++i) {
        inv[i * n + i] = Real(1);
    }

    const Real eps = std::numeric_limits<Real>::epsilon();

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t piv = k;
        Real piv_val = std::abs(A[k * n + k]);
        for (std::size_t r = k + 1; r < n; ++r) {
            const Real v = std::abs(A[r * n + k]);
            if (v > piv_val) {
                piv_val = v;
                piv = r;
            }
        }
        FE_CHECK_ARG(piv_val > eps, "invert_dense_matrix: singular matrix");

        if (piv != k) {
            for (std::size_t c = 0; c < n; ++c) {
                std::swap(A[k * n + c], A[piv * n + c]);
                std::swap(inv[k * n + c], inv[piv * n + c]);
            }
        }

        const Real diag = A[k * n + k];
        FE_CHECK_ARG(std::abs(diag) > eps, "invert_dense_matrix: zero pivot");
        const Real inv_diag = Real(1) / diag;

        for (std::size_t c = 0; c < n; ++c) {
            A[k * n + c] *= inv_diag;
            inv[k * n + c] *= inv_diag;
        }

        for (std::size_t r = 0; r < n; ++r) {
            if (r == k) {
                continue;
            }
            const Real factor = A[r * n + k];
            if (std::abs(factor) <= eps) {
                continue;
            }
            A[r * n + k] = Real(0);
            for (std::size_t c = 0; c < n; ++c) {
                A[r * n + c] -= factor * A[k * n + c];
                inv[r * n + c] -= factor * inv[k * n + c];
            }
        }
    }

    return inv;
}

inline std::vector<Real> powers(Real x, int max_p) {
    FE_CHECK_ARG(max_p >= 0, "powers: negative max_p");
    std::vector<Real> out(static_cast<std::size_t>(max_p + 1), Real(1));
    for (int i = 1; i <= max_p; ++i) {
        out[static_cast<std::size_t>(i)] = out[static_cast<std::size_t>(i - 1)] * x;
    }
    return out;
}

// Compute rank of matrix A (n x n) via pivoted Gaussian elimination
inline int compute_rank(std::vector<Real> A, std::size_t n) {
    const Real eps = Real(1e-12);
    int rank = 0;
    std::size_t row = 0;
    for (std::size_t col = 0; col < n && row < n; ++col) {
        // Find pivot
        std::size_t piv = row;
        Real piv_val = std::abs(A[row * n + col]);
        for (std::size_t r = row + 1; r < n; ++r) {
            const Real v = std::abs(A[r * n + col]);
            if (v > piv_val) {
                piv_val = v;
                piv = r;
            }
        }
        if (piv_val < eps) {
            continue; // This column is zero (in remaining rows)
        }
        // Swap rows
        if (piv != row) {
            for (std::size_t c = col; c < n; ++c) {
                std::swap(A[row * n + c], A[piv * n + c]);
            }
        }
        // Eliminate below
        for (std::size_t r = row + 1; r < n; ++r) {
            const Real factor = A[r * n + col] / A[row * n + col];
            for (std::size_t c = col; c < n; ++c) {
                A[r * n + c] -= factor * A[row * n + c];
            }
        }
        ++rank;
        ++row;
    }
    return rank;
}

// QR decomposition via modified Gram-Schmidt, returns orthogonalized columns
// and the R matrix for diagnostics. Used to orthogonalize the moment matrix rows.
inline void gram_schmidt_columns(std::vector<Real>& A, std::size_t n, std::vector<Real>& norms) {
    norms.resize(n);
    const Real eps = std::numeric_limits<Real>::epsilon() * Real(100);

    for (std::size_t j = 0; j < n; ++j) {
        // Orthogonalize column j against columns 0..j-1
        for (std::size_t i = 0; i < j; ++i) {
            if (norms[i] < eps) continue;
            // Compute dot product of column j with column i
            Real dot = Real(0);
            for (std::size_t r = 0; r < n; ++r) {
                dot += A[r * n + j] * A[r * n + i];
            }
            // Subtract projection
            for (std::size_t r = 0; r < n; ++r) {
                A[r * n + j] -= dot * A[r * n + i];
            }
        }
        // Normalize column j
        Real norm = Real(0);
        for (std::size_t r = 0; r < n; ++r) {
            norm += A[r * n + j] * A[r * n + j];
        }
        norm = std::sqrt(norm);
        norms[j] = norm;
        if (norm > eps) {
            for (std::size_t r = 0; r < n; ++r) {
                A[r * n + j] /= norm;
            }
        }
    }
}

// SVD-based pseudo-inverse for singular or near-singular matrices
// Uses a simple iterative power method approach for small matrices
inline std::vector<Real> pseudo_inverse_dense_matrix(const std::vector<Real>& A, std::size_t n) {
    FE_CHECK_ARG(A.size() == n * n, "pseudo_inverse: size mismatch");

    const Real eps = std::numeric_limits<Real>::epsilon() * Real(1000);

    // First try regular inversion
    std::vector<Real> Acopy = A;
    std::vector<Real> inv(n * n, Real(0));
    for (std::size_t i = 0; i < n; ++i) {
        inv[i * n + i] = Real(1);
    }

    bool singular = false;
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t piv = k;
        Real piv_val = std::abs(Acopy[k * n + k]);
        for (std::size_t r = k + 1; r < n; ++r) {
            const Real v = std::abs(Acopy[r * n + k]);
            if (v > piv_val) {
                piv_val = v;
                piv = r;
            }
        }

        if (piv_val < eps) {
            singular = true;
            break;
        }

        if (piv != k) {
            for (std::size_t c = 0; c < n; ++c) {
                std::swap(Acopy[k * n + c], Acopy[piv * n + c]);
                std::swap(inv[k * n + c], inv[piv * n + c]);
            }
        }

        const Real diag = Acopy[k * n + k];
        const Real inv_diag = Real(1) / diag;

        for (std::size_t c = 0; c < n; ++c) {
            Acopy[k * n + c] *= inv_diag;
            inv[k * n + c] *= inv_diag;
        }

        for (std::size_t r = 0; r < n; ++r) {
            if (r == k) continue;
            const Real factor = Acopy[r * n + k];
            if (std::abs(factor) <= eps) continue;
            Acopy[r * n + k] = Real(0);
            for (std::size_t c = 0; c < n; ++c) {
                Acopy[r * n + c] -= factor * Acopy[k * n + c];
                inv[r * n + c] -= factor * inv[k * n + c];
            }
        }
    }

    if (!singular) {
        return inv;
    }

    // For singular matrices, use Moore-Penrose pseudo-inverse via A^+ = (A^T A)^{-1} A^T
    // with Tikhonov regularization: A^+ = (A^T A + lambda*I)^{-1} A^T
    const Real lambda = eps * Real(n);

    // Compute A^T * A + lambda * I
    std::vector<Real> ATA(n * n, Real(0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            Real sum = Real(0);
            for (std::size_t k = 0; k < n; ++k) {
                sum += A[k * n + i] * A[k * n + j]; // A^T * A
            }
            ATA[i * n + j] = sum;
        }
        ATA[i * n + i] += lambda; // Tikhonov regularization
    }

    // Invert (A^T A + lambda*I)
    std::vector<Real> ATA_inv = invert_dense_matrix(std::move(ATA), n);

    // Compute (A^T A + lambda*I)^{-1} * A^T
    std::vector<Real> result(n * n, Real(0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            Real sum = Real(0);
            for (std::size_t k = 0; k < n; ++k) {
                sum += ATA_inv[i * n + k] * A[j * n + k]; // * A^T
            }
            result[i * n + j] = sum;
        }
    }

    return result;
}

inline void eval_rt1_seed_values(ElementType type, const Vec3& xi, std::vector<Vec3>& values) {
    if (is_wedge(type)) {
        eval_wedge_rt1_direct(xi, values);
    } else {
        eval_pyramid_rt1_direct(xi, values);
    }
}

inline void eval_rt1_seed_divergence(ElementType type, const Vec3& xi, std::vector<Real>& divergence) {
    if (is_wedge(type)) {
        eval_wedge_rt1_divergence_direct(xi, divergence);
    } else {
        eval_pyramid_rt1_divergence_direct(xi, divergence);
    }
}

inline std::vector<Real> build_rt1_direct_transform(ElementType type, std::size_t n) {
    FE_CHECK_ARG(is_wedge(type) || is_pyramid(type),
                 "build_rt1_direct_transform: only wedge/pyramid RT(1) is supported");

    constexpr int k = 1;
    const elements::ReferenceElement ref = elements::ReferenceElement::create(type);
    std::vector<Real> A(n * n, Real(0));
    std::size_t row = 0;

    const LagrangeBasis tri_face_basis(ElementType::Triangle3, k);
    const LagrangeBasis quad_face_basis(ElementType::Quad4, k);
    const auto tri_quad = quadrature::QuadratureFactory::create(
        ElementType::Triangle3, 8, QuadratureType::GaussLegendre, /*use_cache=*/false);
    const auto quad_quad = quadrature::QuadratureFactory::create(
        ElementType::Quad4, 8, QuadratureType::GaussLegendre, /*use_cache=*/false);

    for (std::size_t f = 0; f < ref.num_faces(); ++f) {
        const auto& fn = ref.face_nodes(f);
        if (fn.size() == 3u) {
            const Vec3 v0 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[0]));
            const Vec3 v1 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[1]));
            const Vec3 v2 = NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[2]));
            const Vec3 e01 = v1 - v0;
            const Vec3 e02 = v2 - v0;
            const Vec3 cross = cross3(e01, e02);
            const std::size_t nface = tri_face_basis.size();

            FE_CHECK_ARG(row + nface <= n, "build_rt1_direct_transform: triangular face row overflow");
            for (std::size_t q = 0; q < tri_quad->num_points(); ++q) {
                const auto uv = tri_quad->point(q);
                const Real u = uv[0];
                const Real v = uv[1];
                const Vec3 xi = v0 + e01 * u + e02 * v;

                std::vector<Real> face_vals;
                tri_face_basis.evaluate_values(Vec3{u, v, Real(0)}, face_vals);
                std::vector<Vec3> seed_values;
                eval_rt1_seed_values(type, xi, seed_values);
                FE_CHECK_ARG(seed_values.size() == n,
                             "build_rt1_direct_transform: triangular face seed basis size mismatch");

                const Real wt = tri_quad->weight(q);
                for (std::size_t p = 0; p < n; ++p) {
                    const Real flux =
                        seed_values[p][0] * cross[0] +
                        seed_values[p][1] * cross[1] +
                        seed_values[p][2] * cross[2];
                    for (std::size_t a = 0; a < nface; ++a) {
                        A[(row + a) * n + p] += wt * face_vals[a] * flux;
                    }
                }
            }
            row += nface;
            continue;
        }

        const std::array<Vec3, 4> fv{
            NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[0])),
            NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[1])),
            NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[2])),
            NodeOrdering::get_node_coords(type, static_cast<std::size_t>(fn[3]))
        };
        const std::size_t nface = quad_face_basis.size();

        FE_CHECK_ARG(row + nface <= n, "build_rt1_direct_transform: quadrilateral face row overflow");
        for (std::size_t q = 0; q < quad_quad->num_points(); ++q) {
            const auto uv = quad_quad->point(q);
            const Real u = uv[0];
            const Real w = uv[1];
            const Vec3 xi = bilinear(fv, u, w);
            const Vec3 du = bilinear_du(fv, u, w);
            const Vec3 dw = bilinear_dw(fv, u, w);
            const Vec3 cross = cross3(du, dw);

            std::vector<Real> face_vals;
            quad_face_basis.evaluate_values(Vec3{u, w, Real(0)}, face_vals);
            std::vector<Vec3> seed_values;
            eval_rt1_seed_values(type, xi, seed_values);
            FE_CHECK_ARG(seed_values.size() == n,
                         "build_rt1_direct_transform: quadrilateral face seed basis size mismatch");

            const Real wt = quad_quad->weight(q);
            for (std::size_t p = 0; p < n; ++p) {
                const Real flux =
                    seed_values[p][0] * cross[0] +
                    seed_values[p][1] * cross[1] +
                    seed_values[p][2] * cross[2];
                for (std::size_t a = 0; a < nface; ++a) {
                    A[(row + a) * n + p] += wt * face_vals[a] * flux;
                }
            }
        }
        row += nface;
    }

    const auto vol_quad = quadrature::QuadratureFactory::create(
        type,
        is_wedge(type) ? 8 : 10,
        QuadratureType::GaussLegendre,
        /*use_cache=*/false);

    if (is_wedge(type)) {
        FE_CHECK_ARG(n == 24u, "build_rt1_direct_transform: unexpected wedge RT(1) size");
        for (int c = 0; c < 3; ++c) {
            for (std::size_t l = 0; l < 2u; ++l) {
                FE_CHECK_ARG(row < n, "build_rt1_direct_transform: wedge interior row overflow");
                for (std::size_t q = 0; q < vol_quad->num_points(); ++q) {
                    const Vec3 xi = vol_quad->point(q);
                    const Real wt = vol_quad->weight(q);

                    std::vector<Vec3> seed_values;
                    eval_rt1_seed_values(type, xi, seed_values);
                    FE_CHECK_ARG(seed_values.size() == n,
                                 "build_rt1_direct_transform: wedge interior seed basis size mismatch");

                    const Real z_test = (l == 0u) ? Real(1) : xi[2];
                    for (std::size_t p = 0; p < n; ++p) {
                        A[row * n + p] +=
                            wt * seed_values[p][static_cast<std::size_t>(c)] * z_test;
                    }
                }
                ++row;
            }
        }
    } else {
        FE_CHECK_ARG(n == 19u, "build_rt1_direct_transform: unexpected pyramid RT(1) size");
        for (std::size_t q = 0; q < vol_quad->num_points(); ++q) {
            const Vec3 xi = vol_quad->point(q);
            const Real wt = vol_quad->weight(q);

            std::vector<Vec3> seed_values;
            eval_rt1_seed_values(type, xi, seed_values);
            FE_CHECK_ARG(seed_values.size() == n,
                         "build_rt1_direct_transform: pyramid interior seed basis size mismatch");

            for (int c = 0; c < 3; ++c) {
                FE_CHECK_ARG(row + static_cast<std::size_t>(c) < n,
                             "build_rt1_direct_transform: pyramid interior row overflow");
                for (std::size_t p = 0; p < n; ++p) {
                    A[(row + static_cast<std::size_t>(c)) * n + p] +=
                        wt * seed_values[p][static_cast<std::size_t>(c)];
                }
            }
        }
        row += 3u;
    }

    FE_CHECK_ARG(row == n, "build_rt1_direct_transform: DOF assembly did not fill matrix");
    return invert_dense_matrix(std::move(A), n);
}

inline Real integral_pyramid_z(int power) {
    // Integral of z^power over [0,1] (pyramid apex direction)
    // For pyramid with apex at z=1, the z-integral factor is 1/(power+1)
    FE_CHECK_ARG(power >= 0, "integral_pyramid_z: negative power");
    return Real(1) / static_cast<Real>(power + 1);
}

inline Real integral_wedge_monomial(int px, int py, int pz) {
    // Integral of x^px * y^py * z^pz over wedge: triangle(x,y) x [-1,1](z)
    // = integral_triangle(px,py) * integral_1d(pz)
    return integral_triangle_monomial(px, py) * integral_monomial_1d(pz);
}
