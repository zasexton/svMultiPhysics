/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Eigen/EigenVector.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace backends {

#if defined(FE_HAS_EIGEN)

namespace {

class EigenVectorView final : public assembly::GlobalSystemView {
public:
    explicit EigenVectorView(EigenVector& vec) : vec_(&vec) {}

    // Matrix operations (no-op for vector-only view)
    void addMatrixEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
    void addMatrixEntries(std::span<const GlobalIndex>, std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
    void addMatrixEntry(GlobalIndex, GlobalIndex, Real, assembly::AddMode) override {}
    void setDiagonal(std::span<const GlobalIndex>, std::span<const Real>) override {}
    void setDiagonal(GlobalIndex, Real) override {}
    void zeroRows(std::span<const GlobalIndex>, bool) override {}

    // Vector operations
    void addVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_vector,
                          assembly::AddMode mode) override
    {
        if (dofs.size() != local_vector.size()) {
            FE_THROW(InvalidArgumentException, "EigenVectorView::addVectorEntries: size mismatch");
        }
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            addVectorEntry(dofs[i], local_vector[i], mode);
        }
    }

    void addVectorEntry(GlobalIndex dof, Real value, assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(vec_, "EigenVectorView::vec");
        if (dof < 0 || dof >= vec_->size()) {
            return;
        }

        auto& data = vec_->eigen();
        const auto i = static_cast<Eigen::Index>(dof);

        switch (mode) {
            case assembly::AddMode::Add:
                data[i] += value;
                break;
            case assembly::AddMode::Insert:
                data[i] = value;
                break;
            case assembly::AddMode::Max:
                data[i] = std::max(data[i], value);
                break;
            case assembly::AddMode::Min:
                data[i] = std::min(data[i], value);
                break;
        }
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> values) override
    {
        if (dofs.size() != values.size()) {
            FE_THROW(InvalidArgumentException, "EigenVectorView::setVectorEntries: size mismatch");
        }
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            addVectorEntry(dofs[i], values[i], assembly::AddMode::Insert);
        }
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
    {
        for (const auto dof : dofs) {
            addVectorEntry(dof, 0.0, assembly::AddMode::Insert);
        }
    }

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override
    {
        FE_CHECK_NOT_NULL(vec_, "EigenVectorView::vec");
        if (dof < 0 || dof >= vec_->size()) {
            return 0.0;
        }
        return vec_->eigen()[static_cast<Eigen::Index>(dof)];
    }

    // Assembly lifecycle
    void beginAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Building; }
    void endAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Flushing; }
    void finalizeAssembly() override { phase_ = assembly::AssemblyPhase::Finalized; }
    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return phase_; }

    // Properties
    [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
    [[nodiscard]] bool hasVector() const noexcept override { return true; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return vec_ ? vec_->size() : 0; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return 1; }
    [[nodiscard]] std::string backendName() const override { return "EigenVector"; }

    void zero() override
    {
        FE_CHECK_NOT_NULL(vec_, "EigenVectorView::vec");
        vec_->zero();
    }

private:
    EigenVector* vec_{nullptr};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

} // namespace

EigenVector::EigenVector(GlobalIndex size)
{
    FE_THROW_IF(size < 0, InvalidArgumentException, "EigenVector: negative size");
    FE_THROW_IF(size > static_cast<GlobalIndex>(std::numeric_limits<Eigen::Index>::max()),
                InvalidArgumentException, "EigenVector: size exceeds Eigen::Index range");
    vec_.resize(static_cast<Eigen::Index>(size));
    vec_.setZero();
}

void EigenVector::zero()
{
    vec_.setZero();
}

void EigenVector::set(Real value)
{
    vec_.setConstant(value);
}

void EigenVector::add(Real value)
{
    vec_.array() += value;
}

void EigenVector::scale(Real alpha)
{
    vec_ *= alpha;
}

Real EigenVector::dot(const GenericVector& other) const
{
    const auto* o = dynamic_cast<const EigenVector*>(&other);
    FE_THROW_IF(!o, InvalidArgumentException, "EigenVector::dot: backend mismatch");
    FE_THROW_IF(o->vec_.size() != vec_.size(), InvalidArgumentException, "EigenVector::dot: size mismatch");
    return vec_.dot(o->vec_);
}

Real EigenVector::norm() const
{
    return vec_.norm();
}

void EigenVector::updateGhosts()
{
    // Serial backend: no ghost state.
}

std::unique_ptr<assembly::GlobalSystemView> EigenVector::createAssemblyView()
{
    return std::make_unique<EigenVectorView>(*this);
}

std::span<Real> EigenVector::localSpan()
{
    return std::span<Real>(vec_.data(), static_cast<std::size_t>(vec_.size()));
}

std::span<const Real> EigenVector::localSpan() const
{
    return std::span<const Real>(vec_.data(), static_cast<std::size_t>(vec_.size()));
}

#endif // FE_HAS_EIGEN

} // namespace backends
} // namespace FE
} // namespace svmp
