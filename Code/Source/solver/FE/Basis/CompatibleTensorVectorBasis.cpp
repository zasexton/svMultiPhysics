/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Basis/CompatibleTensorVectorBasis.h"

#include "VectorBasisEvaluationHelpers.h"

#include <sstream>
#include <vector>

namespace svmp {
namespace FE {
namespace basis {

namespace {

struct CompatibleTensorSinglePointScratch {
    std::vector<Real> scalar_values;
    std::vector<Real> scalar_gradients;
};

CompatibleTensorSinglePointScratch& single_point_scratch() {
    static thread_local CompatibleTensorSinglePointScratch scratch;
    return scratch;
}

} // namespace

CompatibleTensorVectorBasis::CompatibleTensorVectorBasis(
    Family family,
    BasisType semantic_basis_type,
    std::shared_ptr<BasisFunction> first_component_basis,
    std::shared_ptr<BasisFunction> second_component_basis,
    std::vector<DofAssociation> associations,
    int order,
    ElementType element_type)
    : CompatibleTensorVectorBasis(family,
                                  semantic_basis_type,
                                  std::vector<std::shared_ptr<BasisFunction>>{
                                      std::move(first_component_basis),
                                      std::move(second_component_basis)},
                                  std::move(associations),
                                  order,
                                  element_type) {}

CompatibleTensorVectorBasis::CompatibleTensorVectorBasis(
    Family family,
    BasisType semantic_basis_type,
    std::vector<std::shared_ptr<BasisFunction>> component_bases,
    std::vector<DofAssociation> associations,
    int order,
    ElementType element_type)
    : family_(family)
    , semantic_basis_type_(semantic_basis_type)
    , element_type_(element_type)
    , order_(order)
    , component_bases_(std::move(component_bases))
    , associations_(std::move(associations)) {
    BASIS_CHECK_CONFIG(element_type_ == ElementType::Quad4 || element_type_ == ElementType::Hex8,
                       "CompatibleTensorVectorBasis: only Quad4 and Hex8 tensor-product cells are supported");

    dimension_ = (element_type_ == ElementType::Quad4) ? 2 : 3;
    BASIS_CHECK_CONFIG(component_bases_.size() == static_cast<std::size_t>(dimension_),
                       "CompatibleTensorVectorBasis: component count must match tensor dimension");

    size_ = 0;
    for (const auto& basis : component_bases_) {
        BASIS_CHECK_CONFIG(basis.get() != nullptr,
                           "CompatibleTensorVectorBasis: null component basis");
        BASIS_CHECK_CONFIG(!basis->is_vector_valued(),
                           "CompatibleTensorVectorBasis: component bases must be scalar");
        BASIS_CHECK_CONFIG(basis->dimension() == dimension_,
                           "CompatibleTensorVectorBasis: component basis dimension mismatch");
        BASIS_CHECK_CONFIG(basis->element_type() == element_type_,
                           "CompatibleTensorVectorBasis: component basis element mismatch");
        size_ += basis->size();
    }

    BASIS_CHECK_CONFIG(associations_.size() == size_,
                       "CompatibleTensorVectorBasis: DOF association size mismatch");
    std::ostringstream oss;
    oss << "CompatibleTensorVectorBasis"
        << "|family=" << static_cast<int>(family_)
        << "|semantic=" << static_cast<int>(semantic_basis_type_)
        << "|elem=" << static_cast<int>(element_type_)
        << "|order=" << order_;
    for (std::size_t c = 0; c < component_bases_.size(); ++c) {
        oss << "|component" << c << '=' << component_bases_[c]->cache_identity();
    }
    cache_identity_ = oss.str();

    cache_identity_words_.clear();
    cache_identity_words_.reserve(16u * component_bases_.size());
    cache_identity_words_.push_back(0x637476623031ULL);
    cache_identity_words_.push_back(static_cast<std::uint64_t>(family_));
    cache_identity_words_.push_back(static_cast<std::uint64_t>(semantic_basis_type_));
    cache_identity_words_.push_back(static_cast<std::uint64_t>(element_type_));
    cache_identity_words_.push_back(static_cast<std::uint64_t>(dimension_));
    cache_identity_words_.push_back(static_cast<std::uint64_t>(order_));
    cache_identity_words_.push_back(static_cast<std::uint64_t>(size_));
    cache_identity_words_.push_back(static_cast<std::uint64_t>(component_bases_.size()));

    for (std::size_t component = 0; component < component_bases_.size(); ++component) {
        const auto& basis = *component_bases_[component];
        cache_identity_words_.push_back(static_cast<std::uint64_t>(component));
        cache_identity_words_.push_back(static_cast<std::uint64_t>(basis.basis_type()));
        cache_identity_words_.push_back(static_cast<std::uint64_t>(basis.element_type()));
        cache_identity_words_.push_back(static_cast<std::uint64_t>(basis.dimension()));
        cache_identity_words_.push_back(static_cast<std::uint64_t>(basis.order()));
        cache_identity_words_.push_back(static_cast<std::uint64_t>(basis.size()));
        cache_identity_words_.push_back(basis.is_vector_valued() ? 1u : 0u);

        const bool needs_identity = !basis.cache_identity_is_structural();
        cache_identity_words_.push_back(needs_identity ? 1u : 0u);
        if (!needs_identity) {
            continue;
        }

        const auto count_position = cache_identity_words_.size();
        cache_identity_words_.push_back(0u);
        const auto component_start = cache_identity_words_.size();
        if (!basis.cache_identity_words(cache_identity_words_)) {
            cache_identity_words_.clear();
            cache_identity_hash_a_ = 0;
            cache_identity_hash_b_ = 0;
            return;
        }
        cache_identity_words_[count_position] =
            static_cast<std::uint64_t>(cache_identity_words_.size() - component_start);
    }

    const auto fingerprint = compute_basis_identity_fingerprint(cache_identity_words_);
    cache_identity_hash_a_ = fingerprint.hash_a;
    cache_identity_hash_b_ = fingerprint.hash_b;
}

std::string CompatibleTensorVectorBasis::cache_identity() const {
    return cache_identity_;
}

bool CompatibleTensorVectorBasis::cache_identity_words(std::vector<std::uint64_t>& words) const {
    if (cache_identity_words_.empty()) {
        return false;
    }
    words.insert(words.end(), cache_identity_words_.begin(), cache_identity_words_.end());
    return true;
}

bool CompatibleTensorVectorBasis::cache_identity_fingerprint(std::uint64_t& hash_a,
                                                             std::uint64_t& hash_b) const {
    if (cache_identity_words_.empty()) {
        return false;
    }
    hash_a = cache_identity_hash_a_;
    hash_b = cache_identity_hash_b_;
    return true;
}

void CompatibleTensorVectorBasis::evaluate_vector_values(
    const math::Vector<Real, 3>& xi,
    std::vector<math::Vector<Real, 3>>& values) const {
    values.assign(size_, math::Vector<Real, 3>{});
    auto& scratch = single_point_scratch();
    std::size_t offset = 0;
    for (std::size_t component = 0; component < component_bases_.size(); ++component) {
        const auto& scalar_basis = *component_bases_[component];
        const std::size_t component_size = scalar_basis.size();
        scratch.scalar_values.resize(component_size);
        scalar_basis.evaluate_values_to(xi, scratch.scalar_values.data());
        for (std::size_t i = 0; i < component_size; ++i) {
            values[offset + i][component] = scratch.scalar_values[i];
        }
        offset += component_size;
    }
}

void CompatibleTensorVectorBasis::evaluate_vector_jacobians(
    const math::Vector<Real, 3>& xi,
    std::vector<VectorJacobian>& jacobians) const {
    jacobians.assign(size_, VectorJacobian{});
    auto& scratch = single_point_scratch();
    std::size_t offset = 0;
    for (std::size_t component = 0; component < component_bases_.size(); ++component) {
        const auto& scalar_basis = *component_bases_[component];
        const std::size_t component_size = scalar_basis.size();
        scratch.scalar_gradients.resize(component_size * 3u);
        scalar_basis.evaluate_gradients_to(xi, scratch.scalar_gradients.data());
        for (std::size_t i = 0; i < component_size; ++i) {
            for (int d = 0; d < dimension_; ++d) {
                jacobians[offset + i](component, static_cast<std::size_t>(d)) =
                    scratch.scalar_gradients[i * 3u + static_cast<std::size_t>(d)];
            }
        }
        offset += component_size;
    }
}

void CompatibleTensorVectorBasis::evaluate_divergence(
    const math::Vector<Real, 3>& xi,
    std::vector<Real>& divergence) const {
    BASIS_CHECK_CONFIG(family_ == Family::HDiv,
                       "CompatibleTensorVectorBasis::evaluate_divergence is only valid for H(div)");

    divergence.assign(size_, Real(0));
    auto& scratch = single_point_scratch();
    std::size_t offset = 0;
    for (std::size_t component = 0; component < component_bases_.size(); ++component) {
        const auto& scalar_basis = *component_bases_[component];
        const std::size_t component_size = scalar_basis.size();
        scratch.scalar_gradients.resize(component_size * 3u);
        scalar_basis.evaluate_gradients_to(xi, scratch.scalar_gradients.data());
        for (std::size_t i = 0; i < component_size; ++i) {
            divergence[offset + i] = scratch.scalar_gradients[i * 3u + component];
        }
        offset += component_size;
    }
}

void CompatibleTensorVectorBasis::evaluate_curl(
    const math::Vector<Real, 3>& xi,
    std::vector<math::Vector<Real, 3>>& curl) const {
    BASIS_CHECK_CONFIG(family_ == Family::HCurl,
                       "CompatibleTensorVectorBasis::evaluate_curl is only valid for H(curl)");

    curl.assign(size_, math::Vector<Real, 3>{});
    auto& scratch = single_point_scratch();
    std::size_t offset = 0;
    for (std::size_t component = 0; component < component_bases_.size(); ++component) {
        const auto& scalar_basis = *component_bases_[component];
        const std::size_t component_size = scalar_basis.size();
        scratch.scalar_gradients.resize(component_size * 3u);
        scalar_basis.evaluate_gradients_to(xi, scratch.scalar_gradients.data());
        for (std::size_t i = 0; i < component_size; ++i) {
            const Real* gradient = scratch.scalar_gradients.data() + i * 3u;
            auto& value = curl[offset + i];
            if (dimension_ == 2) {
                if (component == 0u) {
                    value[2] = -gradient[1];
                } else {
                    value[2] = gradient[0];
                }
            } else {
                if (component == 0u) {
                    value[1] = gradient[2];
                    value[2] = -gradient[1];
                } else if (component == 1u) {
                    value[0] = -gradient[2];
                    value[2] = gradient[0];
                } else {
                    value[0] = gradient[1];
                    value[1] = -gradient[0];
                }
            }
        }
        offset += component_size;
    }
}

void CompatibleTensorVectorBasis::evaluate_vector_at_quadrature_points_strided(
    const std::vector<math::Vector<Real, 3>>& points,
    std::size_t output_stride,
    Real* SVMP_RESTRICT values_out,
    Real* SVMP_RESTRICT jacobians_out,
    Real* SVMP_RESTRICT curls_out,
    Real* SVMP_RESTRICT divergence_out) const {
    if (output_stride < points.size()) {
        throw BasisConfigurationException(
            "CompatibleTensorVectorBasis strided evaluation requires output_stride >= points.size()",
            __FILE__, __LINE__, __func__);
    }
    if (curls_out) {
        BASIS_CHECK_CONFIG(family_ == Family::HCurl,
                           "CompatibleTensorVectorBasis curl output is only valid for H(curl)");
    }
    if (divergence_out) {
        BASIS_CHECK_CONFIG(family_ == Family::HDiv,
                           "CompatibleTensorVectorBasis divergence output is only valid for H(div)");
    }

    struct Scratch {
        std::vector<Real> values;
        std::vector<Real> gradients;
    };
    static thread_local Scratch scratch;

    const bool need_gradients = jacobians_out || curls_out || divergence_out;
    const std::size_t num_qpts = points.size();
    const std::size_t num_dofs = size_;
    if (values_out) {
        detail::vector_common::zero_active_strided_rows(
            values_out, num_dofs * 3u, output_stride, num_qpts);
    }
    if (jacobians_out) {
        detail::vector_common::zero_active_strided_rows(
            jacobians_out, num_dofs * 9u, output_stride, num_qpts);
    }
    if (curls_out) {
        detail::vector_common::zero_active_strided_rows(
            curls_out, num_dofs * 3u, output_stride, num_qpts);
    }
    if (divergence_out) {
        detail::vector_common::zero_active_strided_rows(
            divergence_out, num_dofs, output_stride, num_qpts);
    }

    std::size_t offset = 0;
    for (std::size_t component = 0; component < component_bases_.size(); ++component) {
        const auto& scalar_basis = *component_bases_[component];
        const std::size_t component_size = scalar_basis.size();
        if (values_out) {
            scratch.values.resize(component_size * output_stride);
        } else {
            scratch.values.clear();
        }
        if (need_gradients) {
            scratch.gradients.resize(component_size * 3u * output_stride);
        } else {
            scratch.gradients.clear();
        }

        scalar_basis.evaluate_at_quadrature_points_strided(
            points,
            output_stride,
            values_out ? scratch.values.data() : nullptr,
            need_gradients ? scratch.gradients.data() : nullptr,
            nullptr);

        for (std::size_t i = 0; i < component_size; ++i) {
            const std::size_t dof = offset + i;
            for (std::size_t q = 0; q < num_qpts; ++q) {
                if (values_out) {
                    values_out[(dof * 3u + component) * output_stride + q] =
                        scratch.values[i * output_stride + q];
                }

                if (!need_gradients) {
                    continue;
                }

                Real gradient[3] = {Real(0), Real(0), Real(0)};
                for (int d = 0; d < dimension_; ++d) {
                    const auto sd = static_cast<std::size_t>(d);
                    gradient[sd] = scratch.gradients[(i * 3u + sd) * output_stride + q];
                    if (jacobians_out) {
                        jacobians_out[(dof * 9u + component * 3u + sd) * output_stride + q] =
                            gradient[sd];
                    }
                }

                if (divergence_out) {
                    divergence_out[dof * output_stride + q] =
                        gradient[component];
                }

                if (curls_out) {
                    if (dimension_ == 2) {
                        curls_out[(dof * 3u + 2u) * output_stride + q] =
                            (component == 0u) ? -gradient[1] : gradient[0];
                    } else if (component == 0u) {
                        curls_out[(dof * 3u + 1u) * output_stride + q] = gradient[2];
                        curls_out[(dof * 3u + 2u) * output_stride + q] = -gradient[1];
                    } else if (component == 1u) {
                        curls_out[(dof * 3u + 0u) * output_stride + q] = -gradient[2];
                        curls_out[(dof * 3u + 2u) * output_stride + q] = gradient[0];
                    } else {
                        curls_out[(dof * 3u + 0u) * output_stride + q] = gradient[1];
                        curls_out[(dof * 3u + 1u) * output_stride + q] = -gradient[0];
                    }
                }
            }
        }
        offset += component_size;
    }
}

} // namespace basis
} // namespace FE
} // namespace svmp
