/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_SCHEDULED_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_SCHEDULED_ASSEMBLER_H

/**
 * @file ScheduledAssembler.h
 * @brief Decorator that reorders mesh traversal for cache/locality
 */

#include "Assembly/DecoratorAssembler.h"
#include "Assembly/AssemblyScheduler.h"

#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {

/**
 * @brief Decorator that applies a deterministic element ordering during assembly
 *
 * This wrapper does not change the underlying assembly algorithm; it only
 * reorders `IMeshAccess::forEachCell` / `forEachOwnedCell` traversal.
 */
class ScheduledAssembler final : public DecoratorAssembler {
public:
    enum class Strategy : int {
        Natural = 0,
        Hilbert = 1,
        Morton = 2,
        RCM = 3,
        ComplexityBased = 4,
        CacheBlocked = 5,
    };

    explicit ScheduledAssembler(std::unique_ptr<Assembler> base,
                               Strategy strategy = Strategy::Natural)
        : DecoratorAssembler(std::move(base)), strategy_(strategy)
    {
    }

    void setStrategy(Strategy strategy) noexcept { strategy_ = strategy; }
    [[nodiscard]] Strategy strategy() const noexcept { return strategy_; }

    [[nodiscard]] std::string name() const override
    {
        return "Scheduled(" + base().name() + ")";
    }

    [[nodiscard]] AssemblyResult assembleMatrix(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view) override
    {
        return base().assembleMatrix(makeOrdered(mesh), test_space, trial_space, kernel, matrix_view);
    }

    [[nodiscard]] AssemblyResult assembleVector(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView& vector_view) override
    {
        return base().assembleVector(makeOrdered(mesh), space, kernel, vector_view);
    }

    [[nodiscard]] AssemblyResult assembleBoth(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView& vector_view) override
    {
        return base().assembleBoth(makeOrdered(mesh), test_space, trial_space, kernel, matrix_view, vector_view);
    }

private:
    class OrderedMeshAccess final : public IMeshAccess {
    public:
        OrderedMeshAccess(const IMeshAccess& base, std::span<const GlobalIndex> ordering)
            : base_(base), ordering_(ordering)
        {
        }

        [[nodiscard]] GlobalIndex numCells() const override { return base_.numCells(); }
        [[nodiscard]] GlobalIndex numOwnedCells() const override { return base_.numOwnedCells(); }
        [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return base_.numBoundaryFaces(); }
        [[nodiscard]] GlobalIndex numInteriorFaces() const override { return base_.numInteriorFaces(); }
        [[nodiscard]] int dimension() const override { return base_.dimension(); }

        [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override { return base_.isOwnedCell(cell_id); }
        [[nodiscard]] ElementType getCellType(GlobalIndex cell_id) const override { return base_.getCellType(cell_id); }
        [[nodiscard]] int getCellDomainId(GlobalIndex cell_id) const override { return base_.getCellDomainId(cell_id); }

        void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
        {
            base_.getCellNodes(cell_id, nodes);
        }

        [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
        {
            return base_.getNodeCoordinates(node_id);
        }

        void getCellCoordinates(GlobalIndex cell_id, std::vector<std::array<Real, 3>>& coords) const override
        {
            base_.getCellCoordinates(cell_id, coords);
        }

        [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id, GlobalIndex cell_id) const override
        {
            return base_.getLocalFaceIndex(face_id, cell_id);
        }

        [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
        {
            return base_.getBoundaryFaceMarker(face_id);
        }

        [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex face_id) const override
        {
            return base_.getInteriorFaceCells(face_id);
        }

        void forEachCell(std::function<void(GlobalIndex)> callback) const override
        {
            for (GlobalIndex cell_id : ordering_) {
                callback(cell_id);
            }
        }

        void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
        {
            for (GlobalIndex cell_id : ordering_) {
                if (base_.isOwnedCell(cell_id)) {
                    callback(cell_id);
                }
            }
        }

        void forEachBoundaryFace(int marker,
                                 std::function<void(GlobalIndex, GlobalIndex)> callback) const override
        {
            base_.forEachBoundaryFace(marker, std::move(callback));
        }

        void forEachInteriorFace(
            std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
        {
            base_.forEachInteriorFace(std::move(callback));
        }

    private:
        const IMeshAccess& base_;
        std::span<const GlobalIndex> ordering_;
    };

    [[nodiscard]] const OrderedMeshAccess makeOrdered(const IMeshAccess& mesh) const
    {
        cached_ordering_.clear();

        AssemblyScheduler scheduler;
        scheduler.setMesh(mesh);

        SchedulerOptions opts;
        opts.ordering = toSchedulerStrategy(strategy_);
        scheduler.setOptions(opts);

        auto schedule = scheduler.computeSchedule();
        cached_ordering_ = std::move(schedule.ordering);
        return OrderedMeshAccess(mesh, cached_ordering_);
    }

    static OrderingStrategy toSchedulerStrategy(Strategy s) noexcept
    {
        switch (s) {
            case Strategy::Hilbert:
                return OrderingStrategy::Hilbert;
            case Strategy::Morton:
                return OrderingStrategy::Morton;
            case Strategy::RCM:
                return OrderingStrategy::RCM;
            case Strategy::ComplexityBased:
                return OrderingStrategy::ComplexityBased;
            case Strategy::CacheBlocked:
                return OrderingStrategy::CacheBlocked;
            case Strategy::Natural:
            default:
                return OrderingStrategy::Natural;
        }
    }

    mutable std::vector<GlobalIndex> cached_ordering_{};
    Strategy strategy_{Strategy::Natural};
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_SCHEDULED_ASSEMBLER_H

