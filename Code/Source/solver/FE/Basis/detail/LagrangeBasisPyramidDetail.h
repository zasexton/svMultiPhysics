// Private helper for LagrangeBasis.cpp.
// This header is only intended to be included from LagrangeBasis.cpp after
// the FE basis type aliases are already available.

#include <cmath>
#include <map>
#include <mutex>
#include <vector>
#include "LagrangeBasisUtilityDetail.h"

namespace svmp {
namespace FE {
namespace basis {
namespace detail {

class PyramidLagrangeCache {
public:
    struct ModalTerm {
        int px{0};
        int py{0};
        int pz{0};
        int denom_power{0};
    };

    struct OrderData {
        int order{0};
        std::vector<math::Vector<Real, 3>> nodes;
        std::vector<ModalTerm> modal_terms;
        std::vector<std::vector<Real>> modal_to_nodal;
    };

    static math::Vector<Real, 3>
    regularize_eval_point(const math::Vector<Real, 3>& xi) {
        return std::abs(Real(1) - xi[2]) <= Real(1e-12)
                   ? math::Vector<Real, 3>{Real(0), Real(0), Real(1) - Real(1e-8)}
                   : xi;
    }

    static void evaluate_modal_term(const ModalTerm& term,
                                    const math::Vector<Real, 3>& xi,
                                    Real& value,
                                    Gradient* gradient = nullptr,
                                    Hessian* hessian = nullptr) {
        const Real x = xi[0];
        const Real y = xi[1];
        const Real z = xi[2];
        const Real t = Real(1) - z;
        const Real eps = Real(1e-14);

        if (std::abs(t) <= eps) {
            if (term.px == 0 && term.py == 0) {
                value = std::pow(z, term.pz);
            } else {
                value = Real(0);
            }
            if (gradient != nullptr) {
                *gradient = Gradient{};
                if (term.px == 0 && term.py == 0 && term.pz > 0) {
                    (*gradient)[2] = static_cast<Real>(term.pz) * std::pow(z, term.pz - 1);
                }
            }
            if (hessian != nullptr) {
                *hessian = Hessian{};
                if (term.px == 0 && term.py == 0 && term.pz > 1) {
                    (*hessian)(2, 2) =
                        static_cast<Real>(term.pz * (term.pz - 1)) * std::pow(z, term.pz - 2);
                }
            }
            return;
        }

        const auto pow_nonneg = [](Real base, int p) -> Real {
            return (p <= 0) ? Real(1) : std::pow(base, p);
        };

        const Real base = pow_nonneg(x, term.px) *
                          pow_nonneg(y, term.py) *
                          pow_nonneg(z, term.pz);
        const Real denom = pow_nonneg(t, term.denom_power);
        value = base / denom;

        if (gradient != nullptr) {
            *gradient = Gradient{};
            if (term.px > 0) {
                (*gradient)[0] =
                    static_cast<Real>(term.px) * pow_nonneg(x, term.px - 1) *
                    pow_nonneg(y, term.py) * pow_nonneg(z, term.pz) / denom;
            }
            if (term.py > 0) {
                (*gradient)[1] =
                    static_cast<Real>(term.py) * pow_nonneg(x, term.px) *
                    pow_nonneg(y, term.py - 1) * pow_nonneg(z, term.pz) / denom;
            }

            Real gz = Real(0);
            if (term.pz > 0) {
                gz += static_cast<Real>(term.pz) * pow_nonneg(x, term.px) *
                      pow_nonneg(y, term.py) * pow_nonneg(z, term.pz - 1) / denom;
            }
            if (term.denom_power > 0) {
                gz += static_cast<Real>(term.denom_power) * base / (denom * t);
            }
            (*gradient)[2] = gz;
        }

        if (hessian != nullptr) {
            *hessian = Hessian{};

            if (term.px > 1) {
                (*hessian)(0, 0) =
                    static_cast<Real>(term.px * (term.px - 1)) *
                    pow_nonneg(x, term.px - 2) * pow_nonneg(y, term.py) *
                    pow_nonneg(z, term.pz) / denom;
            }
            if (term.py > 1) {
                (*hessian)(1, 1) =
                    static_cast<Real>(term.py * (term.py - 1)) *
                    pow_nonneg(x, term.px) * pow_nonneg(y, term.py - 2) *
                    pow_nonneg(z, term.pz) / denom;
            }
            if (term.px > 0 && term.py > 0) {
                const Real hxy =
                    static_cast<Real>(term.px * term.py) *
                    pow_nonneg(x, term.px - 1) * pow_nonneg(y, term.py - 1) *
                    pow_nonneg(z, term.pz) / denom;
                (*hessian)(0, 1) = hxy;
                (*hessian)(1, 0) = hxy;
            }

            if (term.px > 0) {
                Real hxz =
                    static_cast<Real>(term.px) * pow_nonneg(x, term.px - 1) *
                    pow_nonneg(y, term.py) / denom;
                if (term.pz > 0) {
                    hxz *= static_cast<Real>(term.pz) * pow_nonneg(z, term.pz - 1);
                } else {
                    hxz = Real(0);
                }
                if (term.denom_power > 0) {
                    hxz += static_cast<Real>(term.px * term.denom_power) *
                           pow_nonneg(x, term.px - 1) * pow_nonneg(y, term.py) *
                           pow_nonneg(z, term.pz) / (denom * t);
                }
                (*hessian)(0, 2) = hxz;
                (*hessian)(2, 0) = hxz;
            }

            if (term.py > 0) {
                Real hyz =
                    static_cast<Real>(term.py) * pow_nonneg(x, term.px) *
                    pow_nonneg(y, term.py - 1) / denom;
                if (term.pz > 0) {
                    hyz *= static_cast<Real>(term.pz) * pow_nonneg(z, term.pz - 1);
                } else {
                    hyz = Real(0);
                }
                if (term.denom_power > 0) {
                    hyz += static_cast<Real>(term.py * term.denom_power) *
                           pow_nonneg(x, term.px) * pow_nonneg(y, term.py - 1) *
                           pow_nonneg(z, term.pz) / (denom * t);
                }
                (*hessian)(1, 2) = hyz;
                (*hessian)(2, 1) = hyz;
            }

            Real hzz = Real(0);
            if (term.pz > 1) {
                hzz += static_cast<Real>(term.pz * (term.pz - 1)) *
                       pow_nonneg(x, term.px) * pow_nonneg(y, term.py) *
                       pow_nonneg(z, term.pz - 2) / denom;
            }
            if (term.pz > 0 && term.denom_power > 0) {
                hzz += static_cast<Real>(2 * term.pz * term.denom_power) *
                       pow_nonneg(x, term.px) * pow_nonneg(y, term.py) *
                       pow_nonneg(z, term.pz - 1) / (denom * t);
            }
            if (term.denom_power > 0) {
                hzz += static_cast<Real>(term.denom_power * (term.denom_power + 1)) *
                       base / (denom * t * t);
            }
            (*hessian)(2, 2) = hzz;
        }
    }

    static const OrderData& get(int order) {
        static std::mutex mutex;
        static std::map<int, OrderData> cache;

        std::lock_guard<std::mutex> lock(mutex);
        const auto found = cache.find(order);
        if (found != cache.end()) {
            return found->second;
        }

        OrderData data;
        data.order = order;

        data.nodes = build_public_nodes(order);
        data.modal_terms = build_modal_terms(order);

        const std::size_t n = data.nodes.size();
        if (data.modal_terms.size() != n) {
            throw FEException("LagrangeBasis pyramid modal basis size mismatch",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
        }

        std::vector<std::vector<Real>> vandermonde(n, std::vector<Real>(n, Real(0)));
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                Real value = Real(0);
                evaluate_modal_term(data.modal_terms[col], data.nodes[row], value);
                vandermonde[row][col] = value;
            }
        }

        std::vector<std::vector<Real>> inverse;
        invert_dense_matrix(vandermonde, inverse);

        data.modal_to_nodal.assign(n, std::vector<Real>(n, Real(0)));
        for (std::size_t basis_i = 0; basis_i < n; ++basis_i) {
            for (std::size_t modal_j = 0; modal_j < n; ++modal_j) {
                data.modal_to_nodal[basis_i][modal_j] = inverse[modal_j][basis_i];
            }
        }

        const auto [it, inserted] = cache.emplace(order, std::move(data));
        (void)inserted;
        return it->second;
    }

    static void evaluate_values(const OrderData& data,
                                const math::Vector<Real, 3>& xi,
                                std::vector<Real>& values) {
        std::vector<Real> modal(data.modal_terms.size(), Real(0));
        for (std::size_t m = 0; m < data.modal_terms.size(); ++m) {
            evaluate_modal_term(data.modal_terms[m], xi, modal[m]);
        }
        apply_modal_to_nodal(data.modal_to_nodal, modal, values);
    }

    static void evaluate_gradients(const OrderData& data,
                                   const math::Vector<Real, 3>& xi,
                                   std::vector<Gradient>& gradients) {
        const math::Vector<Real, 3> eval_xi = regularize_eval_point(xi);
        std::vector<Gradient> modal_gradients(data.modal_terms.size(), Gradient{});
        for (std::size_t m = 0; m < data.modal_terms.size(); ++m) {
            Real value = Real(0);
            evaluate_modal_term(data.modal_terms[m], eval_xi, value, &modal_gradients[m]);
        }
        apply_modal_to_nodal(data.modal_to_nodal, modal_gradients, gradients);
    }

    static void evaluate_hessians(const OrderData& data,
                                  const math::Vector<Real, 3>& xi,
                                  std::vector<Hessian>& hessians) {
        const math::Vector<Real, 3> eval_xi = regularize_eval_point(xi);
        std::vector<Hessian> modal_hessians(data.modal_terms.size(), Hessian{});
        for (std::size_t m = 0; m < data.modal_terms.size(); ++m) {
            Real value = Real(0);
            evaluate_modal_term(data.modal_terms[m], eval_xi, value, nullptr, &modal_hessians[m]);
        }
        apply_modal_to_nodal(data.modal_to_nodal, modal_hessians, hessians);
    }

private:
    static std::vector<math::Vector<Real, 3>> build_public_nodes(int order) {
        if (order == 0) {
            return {math::Vector<Real, 3>{Real(0), Real(0), Real(0.25)}};
        }

        std::vector<math::Vector<Real, 3>> nodes;
        nodes.reserve(static_cast<std::size_t>((order + 1) * (order + 2) * (2 * order + 3) / 6));

        nodes.push_back(math::Vector<Real, 3>{Real(-1), Real(-1), Real(0)});
        nodes.push_back(math::Vector<Real, 3>{Real(1), Real(-1), Real(0)});
        nodes.push_back(math::Vector<Real, 3>{Real(1), Real(1), Real(0)});
        nodes.push_back(math::Vector<Real, 3>{Real(-1), Real(1), Real(0)});
        nodes.push_back(math::Vector<Real, 3>{Real(0), Real(0), Real(1)});

        for (int m = 1; m < order; ++m) {
            nodes.push_back(math::Vector<Real, 3>{equispaced_pm_one_coord(m, order), Real(-1), Real(0)});
        }
        for (int m = 1; m < order; ++m) {
            nodes.push_back(math::Vector<Real, 3>{Real(1), equispaced_pm_one_coord(m, order), Real(0)});
        }
        for (int m = order - 1; m >= 1; --m) {
            nodes.push_back(math::Vector<Real, 3>{equispaced_pm_one_coord(m, order), Real(1), Real(0)});
        }
        for (int m = order - 1; m >= 1; --m) {
            nodes.push_back(math::Vector<Real, 3>{Real(-1), equispaced_pm_one_coord(m, order), Real(0)});
        }

        for (int level = 1; level < order; ++level) {
            const Real z = static_cast<Real>(level) / static_cast<Real>(order);
            const Real scale = Real(1) - z;
            nodes.push_back(math::Vector<Real, 3>{-scale, -scale, z});
            nodes.push_back(math::Vector<Real, 3>{scale, -scale, z});
            nodes.push_back(math::Vector<Real, 3>{scale, scale, z});
            nodes.push_back(math::Vector<Real, 3>{-scale, scale, z});
        }

        for (int j = 1; j < order; ++j) {
            for (int i = 1; i < order; ++i) {
                nodes.push_back(math::Vector<Real, 3>{equispaced_pm_one_coord(i, order),
                                                      equispaced_pm_one_coord(j, order),
                                                      Real(0)});
            }
        }

        for (int level = 1; level < order - 1; ++level) {
            const int n = order - level;
            const Real z = static_cast<Real>(level) / static_cast<Real>(order);
            const Real scale = Real(1) - z;

            for (int m = 1; m < n; ++m) {
                const Real s = equispaced_pm_one_coord(m, n) * scale;
                nodes.push_back(math::Vector<Real, 3>{s, -scale, z});
            }
            for (int m = 1; m < n; ++m) {
                const Real s = equispaced_pm_one_coord(m, n) * scale;
                nodes.push_back(math::Vector<Real, 3>{scale, s, z});
            }
            for (int m = n - 1; m >= 1; --m) {
                const Real s = equispaced_pm_one_coord(m, n) * scale;
                nodes.push_back(math::Vector<Real, 3>{s, scale, z});
            }
            for (int m = n - 1; m >= 1; --m) {
                const Real s = equispaced_pm_one_coord(m, n) * scale;
                nodes.push_back(math::Vector<Real, 3>{-scale, s, z});
            }
        }

        for (int level = 1; level < order - 1; ++level) {
            const int n = order - level;
            const Real z = static_cast<Real>(level) / static_cast<Real>(order);
            const Real scale = Real(1) - z;
            for (int j = 1; j < n; ++j) {
                for (int i = 1; i < n; ++i) {
                    nodes.push_back(math::Vector<Real, 3>{equispaced_pm_one_coord(i, n) * scale,
                                                          equispaced_pm_one_coord(j, n) * scale,
                                                          z});
                }
            }
        }

        return nodes;
    }

    static std::vector<ModalTerm> build_modal_terms(int order) {
        std::vector<ModalTerm> terms;
        terms.reserve(static_cast<std::size_t>((order + 1) * (order + 2) * (2 * order + 3) / 6));
        for (int pz = 0; pz <= order; ++pz) {
            const int n = order - pz;
            for (int py = 0; py <= n; ++py) {
                for (int px = 0; px <= n; ++px) {
                    terms.push_back(ModalTerm{px, py, pz, std::min(px, py)});
                }
            }
        }
        return terms;
    }

    static void apply_modal_to_nodal(const std::vector<std::vector<Real>>& modal_to_nodal,
                                     const std::vector<Real>& modal_values,
                                     std::vector<Real>& nodal_values) {
        nodal_values.assign(modal_to_nodal.size(), Real(0));
        for (std::size_t i = 0; i < modal_to_nodal.size(); ++i) {
            Real sum = Real(0);
            for (std::size_t m = 0; m < modal_values.size(); ++m) {
                sum += modal_to_nodal[i][m] * modal_values[m];
            }
            nodal_values[i] = sum;
        }
    }

    static void apply_modal_to_nodal(const std::vector<std::vector<Real>>& modal_to_nodal,
                                     const std::vector<Gradient>& modal_gradients,
                                     std::vector<Gradient>& nodal_gradients) {
        nodal_gradients.assign(modal_to_nodal.size(), Gradient{});
        for (std::size_t i = 0; i < modal_to_nodal.size(); ++i) {
            Gradient sum{};
            for (std::size_t m = 0; m < modal_gradients.size(); ++m) {
                sum[0] += modal_to_nodal[i][m] * modal_gradients[m][0];
                sum[1] += modal_to_nodal[i][m] * modal_gradients[m][1];
                sum[2] += modal_to_nodal[i][m] * modal_gradients[m][2];
            }
            nodal_gradients[i] = sum;
        }
    }

    static void apply_modal_to_nodal(const std::vector<std::vector<Real>>& modal_to_nodal,
                                     const std::vector<Hessian>& modal_hessians,
                                     std::vector<Hessian>& nodal_hessians) {
        nodal_hessians.assign(modal_to_nodal.size(), Hessian{});
        for (std::size_t i = 0; i < modal_to_nodal.size(); ++i) {
            Hessian sum{};
            for (std::size_t m = 0; m < modal_hessians.size(); ++m) {
                const Real coeff = modal_to_nodal[i][m];
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        sum(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) +=
                            coeff * modal_hessians[m](static_cast<std::size_t>(r),
                                                      static_cast<std::size_t>(c));
                    }
                }
            }
            nodal_hessians[i] = sum;
        }
    }

    static void invert_dense_matrix(std::vector<std::vector<Real>>& matrix,
                                    std::vector<std::vector<Real>>& inverse) {
        const std::size_t n = matrix.size();
        inverse.assign(n, std::vector<Real>(n, Real(0)));
        for (std::size_t i = 0; i < n; ++i) {
            inverse[i][i] = Real(1);
        }

        for (std::size_t col = 0; col < n; ++col) {
            std::size_t pivot_row = col;
            Real pivot_abs = std::abs(matrix[col][col]);
            for (std::size_t row = col + 1; row < n; ++row) {
                const Real candidate = std::abs(matrix[row][col]);
                if (candidate > pivot_abs) {
                    pivot_abs = candidate;
                    pivot_row = row;
                }
            }
            if (pivot_abs <= Real(1e-14)) {
                throw FEException("LagrangeBasis pyramid Vandermonde is singular",
                                  __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
            }

            if (pivot_row != col) {
                std::swap(matrix[pivot_row], matrix[col]);
                std::swap(inverse[pivot_row], inverse[col]);
            }

            const Real pivot = matrix[col][col];
            for (std::size_t j = 0; j < n; ++j) {
                matrix[col][j] /= pivot;
                inverse[col][j] /= pivot;
            }

            for (std::size_t row = 0; row < n; ++row) {
                if (row == col) {
                    continue;
                }
                const Real factor = matrix[row][col];
                if (std::abs(factor) <= Real(0)) {
                    continue;
                }
                for (std::size_t j = 0; j < n; ++j) {
                    matrix[row][j] -= factor * matrix[col][j];
                    inverse[row][j] -= factor * inverse[col][j];
                }
            }
        }
    }
};

} // namespace detail
} // namespace basis
} // namespace FE
} // namespace svmp
