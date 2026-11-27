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
#include <map>

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
    coordinate_matched_ = true;
    auto transform = [translation](std::array<double, 3> p) -> std::array<double, 3> {
        return {{p[0] + translation[0], p[1] + translation[1], p[2] + translation[2]}};
    };

    pairs_ = matchNodalPeriodicDofs(slave_dofs, slave_coords, master_dofs, master_coords, transform, options_);
}

PeriodicBC::PeriodicBC(std::vector<GlobalIndex> slave_dofs,
                        std::vector<std::array<double, 3>> slave_coords,
                        std::vector<GlobalIndex> master_dofs,
                        std::vector<std::array<double, 3>> master_coords,
                        std::function<std::array<double, 3>(std::array<double, 3>)> transform,
                        const PeriodicBCOptions& options)
    : options_(options)
{
    coordinate_matched_ = true;
    pairs_ = matchNodalPeriodicDofs(slave_dofs, slave_coords, master_dofs, master_coords, transform, options_);
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
        if (pair.slave_dof == pair.master_dof) {
            if (pair.weight != 1.0) constraints.addDirichlet(pair.slave_dof, 0.0);
        } else {
            constraints.addLine(pair.slave_dof);
            constraints.addEntry(pair.slave_dof, pair.master_dof, pair.weight);
        }
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

ConstraintDependencyDeclaration PeriodicBC::dependencyDeclaration() const {
    ConstraintDependencyDeclaration out = Constraint::dependencyDeclaration();
    if (coordinate_matched_) {
        merge_into(out.structural, ConstraintDependencyMask::meshGeometry());
        merge_into(out.structural, ConstraintDependencyMask::meshBoundaryTopology());
        out.structural.fe_dof_layout = true;
    }
    return out;
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

namespace {
using Coordinate = std::array<double, 3>;
struct NodalBoundary {
    std::vector<Coordinate> coordinates;
    std::vector<std::vector<GlobalIndex>> dofs;
};

NodalBoundary groupNodalDofs(std::span<const GlobalIndex> dofs, std::span<const Coordinate> coords) {
    NodalBoundary result;
    if (dofs.empty() && coords.empty()) return result;
    if (coords.empty() || dofs.empty() || dofs.size() % coords.size() != 0) {
        CONSTRAINT_THROW("PeriodicBC: each point must supply the same number of component DOFs");
    }
    const auto stride = dofs.size() / coords.size();
    std::map<Coordinate, std::size_t> locations;
    for (std::size_t i = 0; i < coords.size(); ++i) {
        if (!std::all_of(coords[i].begin(), coords[i].end(), [](double x) { return std::isfinite(x); })) {
            CONSTRAINT_THROW("PeriodicBC: nonfinite nodal coordinate");
        }
        const auto [it, inserted] = locations.emplace(coords[i], result.coordinates.size());
        if (inserted) {
            result.coordinates.push_back(coords[i]);
            result.dofs.emplace_back();
        }
        const auto components = dofs.subspan(i * stride, stride);
        auto& point = result.dofs[it->second];
        point.insert(point.end(), components.begin(), components.end());
    }
    return result;
}
} // namespace

std::vector<PeriodicPair> matchNodalPeriodicDofs(
    std::span<const GlobalIndex> slave_dofs, std::span<const Coordinate> slave_coords,
    std::span<const GlobalIndex> master_dofs, std::span<const Coordinate> master_coords,
    const std::function<Coordinate(Coordinate)>& transform, const PeriodicBCOptions& options)
{
    const auto slave = groupNodalDofs(slave_dofs, slave_coords);
    const auto master = groupNodalDofs(master_dofs, master_coords);
    const auto matches = findMatchingPoints(slave.coordinates, master.coordinates, transform, options.matching_tolerance);
    if (matches.size() != slave.coordinates.size() || matches.size() != master.coordinates.size()) {
        CONSTRAINT_THROW("PeriodicBC: the transformed boundaries do not have a one-to-one nodal match");
    }
    std::vector<bool> used(master.coordinates.size(), false);
    std::vector<PeriodicPair> pairs;
    pairs.reserve(slave_dofs.size());
    for (const auto& [i, j] : matches) {
        if (used[j]) CONSTRAINT_THROW("PeriodicBC: two distinct slave points matched the same master");
        used[j] = true;
        const auto& components = slave.dofs[i];
        if (components.size() != master.dofs[j].size() ||
            components.size() != slave.dofs.front().size() ||
            (!options.component_mask.allSelected() && options.component_mask.size() != components.size())) {
            CONSTRAINT_THROW("PeriodicBC: component counts on the periodic field must agree");
        }
        for (std::size_t c = 0; c < components.size(); ++c) {
            if (options.component_mask[c]) {
                pairs.push_back({components[c], master.dofs[j][c], options.anti_periodic ? -1.0 : 1.0});
            }
        }
    }
    return pairs;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
