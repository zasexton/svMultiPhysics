/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "PeriodicBC.h"
#include "ConstraintTools.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// Construction - Direct pair specification
// ============================================================================

PeriodicBC::PeriodicBC(std::vector<GlobalIndex> slave_dofs,
                        std::vector<GlobalIndex> master_dofs)
{
    if (slave_dofs.size() != master_dofs.size()) {
        CONSTRAINT_THROW("Slave and master DOF arrays must have same size");
    }

    pairs_.reserve(slave_dofs.size());
    for (std::size_t i = 0; i < slave_dofs.size(); ++i) {
        pairs_.push_back({slave_dofs[i], master_dofs[i], 1.0});
    }
}

PeriodicBC::PeriodicBC(std::vector<GlobalIndex> slave_dofs,
                        std::vector<GlobalIndex> master_dofs,
                        const PeriodicBCOptions& options)
    : options_(options)
{
    if (slave_dofs.size() != master_dofs.size()) {
        CONSTRAINT_THROW("Slave and master DOF arrays must have same size");
    }

    double weight = options.anti_periodic ? -1.0 : 1.0;

    pairs_.reserve(slave_dofs.size());
    for (std::size_t i = 0; i < slave_dofs.size(); ++i) {
        pairs_.push_back({slave_dofs[i], master_dofs[i], weight});
    }
}

PeriodicBC::PeriodicBC(std::vector<PeriodicPair> pairs)
    : pairs_(std::move(pairs)) {}

// ============================================================================
// Construction - Coordinate-based matching
// ============================================================================

PeriodicBC::PeriodicBC(std::vector<GlobalIndex> slave_dofs,
                        std::vector<std::array<double, 3>> slave_coords,
                        std::vector<GlobalIndex> master_dofs,
                        std::vector<std::array<double, 3>> master_coords,
                        std::array<double, 3> translation,
                        const PeriodicBCOptions& options)
    : options_(options)
{
    auto transform = [translation](std::array<double, 3> p) -> std::array<double, 3> {
        return {{p[0] + translation[0], p[1] + translation[1], p[2] + translation[2]}};
    };

    matchCoordinates(slave_dofs, slave_coords, master_dofs, master_coords, transform);
}

PeriodicBC::PeriodicBC(std::vector<GlobalIndex> slave_dofs,
                        std::vector<std::array<double, 3>> slave_coords,
                        std::vector<GlobalIndex> master_dofs,
                        std::vector<std::array<double, 3>> master_coords,
                        std::function<std::array<double, 3>(std::array<double, 3>)> transform,
                        const PeriodicBCOptions& options)
    : options_(options)
{
    matchCoordinates(slave_dofs, slave_coords, master_dofs, master_coords, transform);
}

// ============================================================================
// Copy/Move
// ============================================================================

PeriodicBC::PeriodicBC(const PeriodicBC& other) = default;
PeriodicBC::PeriodicBC(PeriodicBC&& other) noexcept = default;
PeriodicBC& PeriodicBC::operator=(const PeriodicBC& other) = default;
PeriodicBC& PeriodicBC::operator=(PeriodicBC&& other) noexcept = default;

// ============================================================================
// Constraint interface
// ============================================================================

void PeriodicBC::apply(AffineConstraints& constraints) const {
    for (const auto& pair : pairs_) {
        constraints.addLine(pair.slave_dof);
        constraints.addEntry(pair.slave_dof, pair.master_dof, pair.weight);
    }
}

ConstraintInfo PeriodicBC::getInfo() const {
    ConstraintInfo info;
    info.name = "PeriodicBC";
    info.type = ConstraintType::Periodic;
    info.num_constrained_dofs = pairs_.size();
    info.is_time_dependent = false;
    info.is_homogeneous = true;  // Periodic constraints have no inhomogeneity
    return info;
}

// ============================================================================
// Modification
// ============================================================================

void PeriodicBC::addPair(GlobalIndex slave_dof, GlobalIndex master_dof, double weight) {
    pairs_.push_back({slave_dof, master_dof, weight});
}

void PeriodicBC::addPairs(std::span<const GlobalIndex> slave_dofs,
                          std::span<const GlobalIndex> master_dofs) {
    if (slave_dofs.size() != master_dofs.size()) {
        CONSTRAINT_THROW("Slave and master DOF arrays must have same size");
    }

    double weight = options_.anti_periodic ? -1.0 : 1.0;
    for (std::size_t i = 0; i < slave_dofs.size(); ++i) {
        pairs_.push_back({slave_dofs[i], master_dofs[i], weight});
    }
}

// ============================================================================
// Factory methods
// ============================================================================

PeriodicBC PeriodicBC::xPeriodic(
    std::vector<GlobalIndex> left_dofs,
    std::vector<std::array<double, 3>> left_coords,
    std::vector<GlobalIndex> right_dofs,
    std::vector<std::array<double, 3>> right_coords,
    double domain_length)
{
    std::array<double, 3> translation = {{domain_length, 0.0, 0.0}};
    return PeriodicBC(std::move(left_dofs), std::move(left_coords),
                      std::move(right_dofs), std::move(right_coords),
                      translation);
}

PeriodicBC PeriodicBC::yPeriodic(
    std::vector<GlobalIndex> bottom_dofs,
    std::vector<std::array<double, 3>> bottom_coords,
    std::vector<GlobalIndex> top_dofs,
    std::vector<std::array<double, 3>> top_coords,
    double domain_length)
{
    std::array<double, 3> translation = {{0.0, domain_length, 0.0}};
    return PeriodicBC(std::move(bottom_dofs), std::move(bottom_coords),
                      std::move(top_dofs), std::move(top_coords),
                      translation);
}

PeriodicBC PeriodicBC::zPeriodic(
    std::vector<GlobalIndex> back_dofs,
    std::vector<std::array<double, 3>> back_coords,
    std::vector<GlobalIndex> front_dofs,
    std::vector<std::array<double, 3>> front_coords,
    double domain_length)
{
    std::array<double, 3> translation = {{0.0, 0.0, domain_length}};
    return PeriodicBC(std::move(back_dofs), std::move(back_coords),
                      std::move(front_dofs), std::move(front_coords),
                      translation);
}

// ============================================================================
// Internal helpers
// ============================================================================

void PeriodicBC::matchCoordinates(
    std::span<const GlobalIndex> slave_dofs,
    std::span<const std::array<double, 3>> slave_coords,
    std::span<const GlobalIndex> master_dofs,
    std::span<const std::array<double, 3>> master_coords,
    const std::function<std::array<double, 3>(std::array<double, 3>)>& transform)
{
    if (slave_dofs.size() != slave_coords.size()) {
        CONSTRAINT_THROW("Slave DOFs and coordinates must have same size");
    }
    if (master_dofs.size() != master_coords.size()) {
        CONSTRAINT_THROW("Master DOFs and coordinates must have same size");
    }

    double weight = options_.anti_periodic ? -1.0 : 1.0;
    double tol_sq = options_.matching_tolerance * options_.matching_tolerance;

    for (std::size_t i = 0; i < slave_dofs.size(); ++i) {
        std::array<double, 3> transformed = transform(slave_coords[i]);

        // Find matching master
        double best_dist_sq = std::numeric_limits<double>::max();
        std::size_t best_j = master_coords.size();

        for (std::size_t j = 0; j < master_coords.size(); ++j) {
            double dx = transformed[0] - master_coords[j][0];
            double dy = transformed[1] - master_coords[j][1];
            double dz = transformed[2] - master_coords[j][2];
            double dist_sq = dx*dx + dy*dy + dz*dz;

            if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_j = j;
            }
        }

        if (best_j < master_coords.size() && best_dist_sq < tol_sq) {
            pairs_.push_back({slave_dofs[i], master_dofs[best_j], weight});
        } else {
            // No match found - could be an error or the user intentionally
            // provided non-matching boundaries
        }
    }
}

} // namespace constraints
} // namespace FE
} // namespace svmp
