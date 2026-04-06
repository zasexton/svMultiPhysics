#include "Auxiliary/AuxiliaryMultirateScheduler.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace systems {

void AuxiliaryMultirateScheduler::addBlockSchedule(MultirateBlockSchedule schedule)
{
    schedules_.push_back(std::move(schedule));
}

const MultirateBlockSchedule* AuxiliaryMultirateScheduler::getSchedule(
    std::string_view block_name) const
{
    for (const auto& s : schedules_) {
        if (s.block_name == block_name) return &s;
    }
    return nullptr;
}

std::vector<std::string> AuxiliaryMultirateScheduler::scheduledBlocks() const
{
    std::vector<std::string> names;
    names.reserve(schedules_.size());
    for (const auto& s : schedules_) {
        names.push_back(s.block_name);
    }
    return names;
}

std::vector<AuxiliaryMultirateScheduler::Substep>
AuxiliaryMultirateScheduler::planSubsteps(Real t, Real dt) const
{
    std::vector<Substep> all_substeps;

    for (const auto& sched : schedules_) {
        const int n = std::max(sched.rate_ratio, 1);
        const Real dt_sub = dt / static_cast<Real>(n);

        for (int k = 0; k < n; ++k) {
            Substep ss;
            ss.block_name = sched.block_name;
            ss.t_start = t + static_cast<Real>(k) * dt_sub;
            ss.dt_sub = dt_sub;
            ss.substep_index = k;
            ss.total_substeps = n;
            all_substeps.push_back(std::move(ss));
        }
    }

    // Sort by time, then by block name for determinism.
    std::sort(all_substeps.begin(), all_substeps.end(),
              [](const Substep& a, const Substep& b) {
                  if (a.t_start != b.t_start) return a.t_start < b.t_start;
                  return a.block_name < b.block_name;
              });

    return all_substeps;
}

std::vector<std::string> AuxiliaryMultirateScheduler::validate(
    std::span<const std::string> monolithic_blocks) const
{
    std::vector<std::string> errors;

    for (const auto& sched : schedules_) {
        if (sched.rate_ratio < 1) {
            errors.push_back("Block '" + sched.block_name +
                             "': rate_ratio must be >= 1, got " +
                             std::to_string(sched.rate_ratio));
        }

        // Monolithic blocks must have rate_ratio == 1.
        for (const auto& mono : monolithic_blocks) {
            if (sched.block_name == mono && sched.rate_ratio != 1) {
                errors.push_back("Block '" + sched.block_name +
                                 "' is Monolithic but has rate_ratio=" +
                                 std::to_string(sched.rate_ratio) +
                                 " (must be 1)");
            }
        }
    }

    return errors;
}

void AuxiliaryMultirateScheduler::clear()
{
    schedules_.clear();
}

} // namespace systems
} // namespace FE
} // namespace svmp
