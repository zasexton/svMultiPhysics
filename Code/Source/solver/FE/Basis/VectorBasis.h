/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_VECTORBASIS_H
#define SVMP_FE_BASIS_VECTORBASIS_H

/**
 * @file VectorBasis.h
 * @brief Vector-valued bases for H(div) and H(curl) conforming spaces
 */

#include "BasisFunction.h"
#include <array>

namespace svmp {
namespace FE {
namespace basis {

/**
 * @brief DOF entity type for vector-valued basis functions
 */
enum class DofEntity {
    Vertex,   ///< DOF associated with a vertex
    Edge,     ///< DOF associated with an edge (tangential moments for H(curl))
    Face,     ///< DOF associated with a face (normal moments for H(div), tangential for H(curl))
    Interior  ///< DOF associated with element interior
};

/**
 * @brief DOF association metadata for a single DOF
 */
struct DofAssociation {
    DofEntity entity_type{DofEntity::Interior};
    int entity_id{-1};      ///< Local index of the entity (edge/face/vertex)
    int moment_index{0};    ///< Index within the entity's moment space
};

class VectorBasisFunction : public BasisFunction {
public:
    bool is_vector_valued() const noexcept override { return true; }
    void evaluate_values(const math::Vector<Real, 3>&,
                         std::vector<Real>&) const override {
        throw FEException("Vector basis uses evaluate_vector_values",
                          __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    /**
     * @brief Get DOF association metadata for all basis functions
     *
     * Returns a vector of size(), where each entry describes which
     * geometric entity (vertex/edge/face/interior) the corresponding
     * DOF is associated with. This is essential for orientation-aware
     * assembly of H(div) and H(curl) spaces.
     */
    virtual std::vector<DofAssociation> dof_associations() const {
        // Default: all interior DOFs (subclasses should override)
        std::vector<DofAssociation> result(size());
        for (std::size_t i = 0; i < size(); ++i) {
            result[i].entity_type = DofEntity::Interior;
            result[i].entity_id = 0;
            result[i].moment_index = static_cast<int>(i);
        }
        return result;
    }
};

/**
 * @brief Raviart-Thomas H(div) basis on supported element families
 */
class RaviartThomasBasis : public VectorBasisFunction {
public:
    RaviartThomasBasis(ElementType type, int order = 0);

    BasisType basis_type() const noexcept override { return BasisType::RaviartThomas; }
    ElementType element_type() const noexcept override { return element_type_; }
    int dimension() const noexcept override { return dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return size_; }

    void evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                std::vector<math::Vector<Real, 3>>& values) const override;
    void evaluate_divergence(const math::Vector<Real, 3>& xi,
                             std::vector<Real>& divergence) const override;

    /// Get DOF associations (face/edge DOFs for 2D, face DOFs for 3D H(div))
    std::vector<DofAssociation> dof_associations() const override;

private:
    struct ModalTerm {
        int component{0}; // 0=x, 1=y, 2=z
        int px{0};
        int py{0};
        int pz{0};
        Real coefficient{Real(1)};
    };

    struct ModalPolynomial {
        std::array<ModalTerm, 3> terms{};
        int num_terms{0};
    };

    ElementType element_type_;
    int dimension_;
    int order_;
    std::size_t size_{0};

    bool nodal_generated_{false};
    bool use_direct_construction_{false};  ///< True for wedge/pyramid k>=1 (uses explicit formulas)
    std::vector<ModalPolynomial> monomials_;
    // Coefficients for nodal basis in modal monomial basis:
    //   phi_j = sum_p coeffs_[p * size_ + j] * modal_p
    // where p indexes monomials_ (modal basis) and j indexes nodal basis functions/DOFs.
    std::vector<Real> coeffs_;
};

/**
 * @brief First-kind Nedelec H(curl) basis on supported element families
 */
class NedelecBasis : public VectorBasisFunction {
public:
    NedelecBasis(ElementType type, int order = 0);

    BasisType basis_type() const noexcept override { return BasisType::Nedelec; }
    ElementType element_type() const noexcept override { return element_type_; }
    int dimension() const noexcept override { return dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return size_; }

    void evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                std::vector<math::Vector<Real, 3>>& values) const override;
    void evaluate_curl(const math::Vector<Real, 3>& xi,
                       std::vector<math::Vector<Real, 3>>& curl) const override;

    /// Get DOF associations (edge DOFs for H(curl), face DOFs for 3D interior)
    std::vector<DofAssociation> dof_associations() const override;

private:
    struct ModalTerm {
        int component{0}; // 0=x, 1=y, 2=z
        int px{0};
        int py{0};
        int pz{0};
        Real coefficient{Real(1)};
    };

    struct ModalPolynomial {
        std::array<ModalTerm, 3> terms{};
        int num_terms{0};
    };

    ElementType element_type_;
    int dimension_;
    int order_;
    std::size_t size_{0};

    bool nodal_generated_{false};
    bool use_direct_construction_{false};  ///< True for wedge/pyramid k>=1 (uses explicit formulas)
    std::vector<ModalPolynomial> monomials_;
    std::vector<Real> coeffs_;
};

/**
 * @brief Brezzi-Douglas-Marini basis (simple linear variant)
 */
class BDMBasis : public VectorBasisFunction {
public:
    BDMBasis(ElementType type, int order = 1);

    BasisType basis_type() const noexcept override { return BasisType::BDM; }
    ElementType element_type() const noexcept override { return element_type_; }
    int dimension() const noexcept override { return dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return size_; }

    void evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                std::vector<math::Vector<Real, 3>>& values) const override;
    void evaluate_divergence(const math::Vector<Real, 3>& xi,
                             std::vector<Real>& divergence) const override;

    /// Get DOF associations (face/edge DOFs for H(div))
    std::vector<DofAssociation> dof_associations() const override;

private:
    ElementType element_type_;
    int dimension_;
    int order_;
    std::size_t size_{0};
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_VECTORBASIS_H
