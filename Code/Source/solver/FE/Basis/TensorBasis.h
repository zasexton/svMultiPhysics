/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_TENSORBASIS_H
#define SVMP_FE_BASIS_TENSORBASIS_H

/**
 * @file TensorBasis.h
 * @brief Tensor-product basis wrapper for quadrilateral and hexahedral elements
 */

#include "BasisFunction.h"
#include "NodeOrderingConventions.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <type_traits>
#include <utility>

namespace svmp {
namespace FE {
namespace basis {

namespace detail {

template<typename T, typename = void>
struct has_nodes_method : std::false_type {};

template<typename T>
struct has_nodes_method<T, std::void_t<decltype(std::declval<const T&>().nodes())>> : std::true_type {};

inline bool coords_close(const math::Vector<Real, 3>& a,
                         const math::Vector<Real, 3>& b,
                         Real tol = Real(1e-12)) {
    return (std::abs(a[0] - b[0]) <= tol) &&
           (std::abs(a[1] - b[1]) <= tol) &&
           (std::abs(a[2] - b[2]) <= tol);
}

} // namespace detail

/**
 * @brief Generic tensor-product basis composed from 1D bases
 *
 * Basis1D must satisfy the BasisFunction interface on a line element.
 */
template<typename Basis1D>
class TensorProductBasis : public BasisFunction {
public:
    /// Construct isotropic tensor-product basis from a single 1D prototype
    explicit TensorProductBasis(const Basis1D& basis_1d, int dimension = 2)
        : bases_{basis_1d, basis_1d, basis_1d}, dimension_(dimension) {
        if (dimension_ != 1 && dimension_ != 2 && dimension_ != 3) {
            throw BasisConfigurationException("TensorProductBasis dimension must be 1, 2, or 3",
                                              __FILE__, __LINE__, __func__);
        }
        normalize_orders();
        build_indices();
        finalize_node_ordering();
    }

    /// Anisotropic 2D tensor product
    TensorProductBasis(const Basis1D& bx, const Basis1D& by)
        : bases_{bx, by, bx}, dimension_(2) {
        normalize_orders();
        build_indices();
        finalize_node_ordering();
    }

    /// Anisotropic 3D tensor product
    TensorProductBasis(const Basis1D& bx, const Basis1D& by, const Basis1D& bz)
        : bases_{bx, by, bz}, dimension_(3) {
        normalize_orders();
        build_indices();
        finalize_node_ordering();
    }

    BasisType basis_type() const noexcept override { return bases_[0].basis_type(); }
    ElementType element_type() const noexcept override {
        if (dimension_ == 1) return ElementType::Line2;
        if (dimension_ == 2) return ElementType::Quad4;
        return ElementType::Hex8;
    }
    int dimension() const noexcept override { return dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return indices_.size(); }
    std::string cache_identity() const override {
        std::ostringstream oss;
        oss << BasisFunction::cache_identity() << "|axes=" << dimension_;
        for (int axis = 0; axis < dimension_; ++axis) {
            oss << "|axis" << axis << '=' << bases_[static_cast<std::size_t>(axis)].cache_identity();
        }
        return oss.str();
    }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override {
        values.assign(size(), Real(0));

        std::vector<Real> vx, vy, vz;
        bases_[0].evaluate_values(math::Vector<Real,3>{xi[0], Real(0), Real(0)}, vx);
        if (dimension_ >= 2) {
            bases_[1].evaluate_values(math::Vector<Real,3>{xi[1], Real(0), Real(0)}, vy);
        }
        if (dimension_ == 3) {
            bases_[2].evaluate_values(math::Vector<Real,3>{xi[2], Real(0), Real(0)}, vz);
        }

        for (std::size_t idx = 0; idx < indices_.size(); ++idx) {
            const auto& id = indices_[idx];
            Real val = vx[static_cast<std::size_t>(id[0])];
            if (dimension_ >= 2) {
                val *= vy[static_cast<std::size_t>(id[1])];
            }
            if (dimension_ == 3) {
                val *= vz[static_cast<std::size_t>(id[2])];
            }
            values[idx] = val;
        }
    }

    void evaluate_gradients(const math::Vector<Real, 3>& xi,
                            std::vector<Gradient>& gradients) const override {
        gradients.assign(size(), Gradient{});

        std::vector<Real> vx, vy, vz;
        std::vector<Gradient> gx, gy, gz;
        bases_[0].evaluate_values(math::Vector<Real,3>{xi[0], Real(0), Real(0)}, vx);
        bases_[0].evaluate_gradients(math::Vector<Real,3>{xi[0], Real(0), Real(0)}, gx);
        if (dimension_ >= 2) {
            bases_[1].evaluate_values(math::Vector<Real,3>{xi[1], Real(0), Real(0)}, vy);
            bases_[1].evaluate_gradients(math::Vector<Real,3>{xi[1], Real(0), Real(0)}, gy);
        }
        if (dimension_ == 3) {
            bases_[2].evaluate_values(math::Vector<Real,3>{xi[2], Real(0), Real(0)}, vz);
            bases_[2].evaluate_gradients(math::Vector<Real,3>{xi[2], Real(0), Real(0)}, gz);
        }

        for (std::size_t idx = 0; idx < indices_.size(); ++idx) {
            const auto& id = indices_[idx];
            Gradient g{};
            if (dimension_ == 1) {
                g[0] = gx[static_cast<std::size_t>(id[0])][0];
            } else if (dimension_ == 2) {
                g[0] = gx[static_cast<std::size_t>(id[0])][0] * vy[static_cast<std::size_t>(id[1])];
                g[1] = vx[static_cast<std::size_t>(id[0])] * gy[static_cast<std::size_t>(id[1])][0];
            } else {
                g[0] = gx[static_cast<std::size_t>(id[0])][0] *
                       vy[static_cast<std::size_t>(id[1])] *
                       vz[static_cast<std::size_t>(id[2])];
                g[1] = vx[static_cast<std::size_t>(id[0])] *
                       gy[static_cast<std::size_t>(id[1])][0] *
                       vz[static_cast<std::size_t>(id[2])];
                g[2] = vx[static_cast<std::size_t>(id[0])] *
                       vy[static_cast<std::size_t>(id[1])] *
                       gz[static_cast<std::size_t>(id[2])][0];
            }
            gradients[idx] = g;
        }
    }

    void evaluate_hessians(const math::Vector<Real, 3>& xi,
                           std::vector<Hessian>& hessians) const override {
        hessians.assign(size(), Hessian{});

        std::vector<Real> vx, vy, vz;
        std::vector<Gradient> gx, gy, gz;
        std::vector<Hessian> hx, hy, hz;
        bases_[0].evaluate_values(math::Vector<Real,3>{xi[0], Real(0), Real(0)}, vx);
        bases_[0].evaluate_gradients(math::Vector<Real,3>{xi[0], Real(0), Real(0)}, gx);
        bases_[0].evaluate_hessians(math::Vector<Real,3>{xi[0], Real(0), Real(0)}, hx);
        if (dimension_ >= 2) {
            bases_[1].evaluate_values(math::Vector<Real,3>{xi[1], Real(0), Real(0)}, vy);
            bases_[1].evaluate_gradients(math::Vector<Real,3>{xi[1], Real(0), Real(0)}, gy);
            bases_[1].evaluate_hessians(math::Vector<Real,3>{xi[1], Real(0), Real(0)}, hy);
        }
        if (dimension_ == 3) {
            bases_[2].evaluate_values(math::Vector<Real,3>{xi[2], Real(0), Real(0)}, vz);
            bases_[2].evaluate_gradients(math::Vector<Real,3>{xi[2], Real(0), Real(0)}, gz);
            bases_[2].evaluate_hessians(math::Vector<Real,3>{xi[2], Real(0), Real(0)}, hz);
        }

        for (std::size_t idx = 0; idx < indices_.size(); ++idx) {
            const auto& id = indices_[idx];
            const auto ix = static_cast<std::size_t>(id[0]);
            Hessian H{};
            if (dimension_ == 1) {
                H(0, 0) = hx[ix](0, 0);
            } else if (dimension_ == 2) {
                const auto iy = static_cast<std::size_t>(id[1]);
                const Real dx = gx[ix][0];
                const Real dy = gy[iy][0];
                H(0, 0) = hx[ix](0, 0) * vy[iy];
                H(1, 1) = vx[ix] * hy[iy](0, 0);
                H(0, 1) = dx * dy;
                H(1, 0) = H(0, 1);
            } else {
                const auto iy = static_cast<std::size_t>(id[1]);
                const auto iz = static_cast<std::size_t>(id[2]);
                const Real dx = gx[ix][0];
                const Real dy = gy[iy][0];
                const Real dz = gz[iz][0];
                H(0, 0) = hx[ix](0, 0) * vy[iy] * vz[iz];
                H(1, 1) = vx[ix] * hy[iy](0, 0) * vz[iz];
                H(2, 2) = vx[ix] * vy[iy] * hz[iz](0, 0);
                H(0, 1) = dx * dy * vz[iz];
                H(1, 0) = H(0, 1);
                H(0, 2) = dx * vy[iy] * dz;
                H(2, 0) = H(0, 2);
                H(1, 2) = vx[ix] * dy * dz;
                H(2, 1) = H(1, 2);
            }
            hessians[idx] = H;
        }
    }

private:
    std::array<Basis1D, 3> bases_;
    int dimension_;
    int order_{0};
    std::vector<std::array<int, 3>> indices_;

    void normalize_orders() {
        order_ = bases_[0].order();
        if (dimension_ >= 2) {
            order_ = std::max(order_, bases_[1].order());
        }
        if (dimension_ == 3) {
            order_ = std::max(order_, bases_[2].order());
        }
    }

    void build_indices() {
        indices_.clear();
        const int nx = static_cast<int>(bases_[0].size());
        const int ny = (dimension_ >= 2) ? static_cast<int>(bases_[1].size()) : 1;
        const int nz = (dimension_ == 3) ? static_cast<int>(bases_[2].size()) : 1;
        if (dimension_ == 1) {
            for (int i = 0; i < nx; ++i) {
                indices_.push_back({i, 0, 0});
            }
            return;
        }
        if (dimension_ == 2) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    indices_.push_back({i, j, 0});
                }
            }
            return;
        }
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    indices_.push_back({i, j, k});
                }
            }
        }
    }

    void finalize_node_ordering() {
        if (dimension_ == 1) {
            return;
        }
        if (bases_[0].basis_type() != BasisType::Lagrange) {
            return;
        }
        if constexpr (!detail::has_nodes_method<Basis1D>::value) {
            return;
        } else {
            const int ox = bases_[0].order();
            const int oy = (dimension_ >= 2) ? bases_[1].order() : ox;
            const int oz = (dimension_ == 3) ? bases_[2].order() : ox;
            if ((dimension_ == 2 && ox != oy) || (dimension_ == 3 && (ox != oy || ox != oz))) {
                return;
            }

            ElementType ordering_type = ElementType::Unknown;
            if (dimension_ == 2) {
                if (ox == 1) ordering_type = ElementType::Quad4;
                if (ox == 2) ordering_type = ElementType::Quad9;
            } else if (dimension_ == 3) {
                if (ox == 1) ordering_type = ElementType::Hex8;
                if (ox == 2) ordering_type = ElementType::Hex27;
            }
            if (ordering_type == ElementType::Unknown) {
                return;
            }
            if (indices_.size() != NodeOrdering::num_nodes(ordering_type)) {
                return;
            }

            const auto& nx = bases_[0].nodes();
            const auto& ny = bases_[1].nodes();
            const auto& nz = bases_[2].nodes();
            if (nx.size() != bases_[0].size()) {
                return;
            }
            if (dimension_ >= 2 && ny.size() != bases_[1].size()) {
                return;
            }
            if (dimension_ == 3 && nz.size() != bases_[2].size()) {
                return;
            }

            std::vector<math::Vector<Real, 3>> internal_nodes;
            internal_nodes.reserve(indices_.size());
            for (const auto& id : indices_) {
                math::Vector<Real, 3> p{Real(0), Real(0), Real(0)};
                p[0] = nx[static_cast<std::size_t>(id[0])][0];
                if (dimension_ >= 2) {
                    p[1] = ny[static_cast<std::size_t>(id[1])][0];
                }
                if (dimension_ == 3) {
                    p[2] = nz[static_cast<std::size_t>(id[2])][0];
                }
                internal_nodes.push_back(p);
            }

            std::vector<std::array<int, 3>> reordered(indices_.size());
            std::vector<bool> used(indices_.size(), false);
            for (std::size_t ext = 0; ext < indices_.size(); ++ext) {
                const auto target = NodeOrdering::get_node_coords(ordering_type, ext);
                bool found = false;
                for (std::size_t in = 0; in < indices_.size(); ++in) {
                    if (used[in]) {
                        continue;
                    }
                    if (detail::coords_close(internal_nodes[in], target)) {
                        reordered[ext] = indices_[in];
                        used[in] = true;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw BasisNodeOrderingException("TensorProductBasis: failed to align tensor-product nodes with NodeOrderingConventions",
                                                     __FILE__, __LINE__, __func__);
                }
            }

            indices_ = std::move(reordered);
        }
    }
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_TENSORBASIS_H
