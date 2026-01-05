/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_EIGEN_VECTOR_H
#define SVMP_FE_BACKENDS_EIGEN_VECTOR_H

#include "Backends/Interfaces/GenericVector.h"

#if defined(FE_HAS_EIGEN)
#include <Eigen/Dense>
#endif

namespace svmp {
namespace FE {
namespace backends {

#if defined(FE_HAS_EIGEN)

class EigenVector final : public GenericVector {
public:
    explicit EigenVector(GlobalIndex size);

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::Eigen; }
    [[nodiscard]] GlobalIndex size() const noexcept override { return static_cast<GlobalIndex>(vec_.size()); }

    void zero() override;
    void set(Real value) override;
    void add(Real value) override;
    void scale(Real alpha) override;

    [[nodiscard]] Real dot(const GenericVector& other) const override;
    [[nodiscard]] Real norm() const override;

    void updateGhosts() override;

    [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() override;

    [[nodiscard]] std::span<Real> localSpan() override;
    [[nodiscard]] std::span<const Real> localSpan() const override;

    [[nodiscard]] Eigen::VectorXd& eigen() noexcept { return vec_; }
    [[nodiscard]] const Eigen::VectorXd& eigen() const noexcept { return vec_; }

private:
    Eigen::VectorXd vec_;
};

#endif // FE_HAS_EIGEN

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_EIGEN_VECTOR_H

