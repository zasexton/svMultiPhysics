#ifndef SVMP_FE_FORMS_BOUNDARY_FUNCTIONAL_H
#define SVMP_FE_FORMS_BOUNDARY_FUNCTIONAL_H

/**
 * @file BoundaryFunctional.h
 * @brief Boundary-integrated scalar quantities for coupled boundary conditions
 *
 * Boundary functionals are scalar quantities obtained by integrating a
 * scalar-valued FormExpr over a boundary marker. They are designed as the
 * "non-local" building block for coupled boundary conditions (e.g. RCR-style
 * 0D-3D coupling), where a boundary condition depends on global boundary
 * integrals (such as total flow rate).
 */

#include "Core/Types.h"
#include "Core/Alignment.h"
#include "Core/AlignedAllocator.h"
#include "Core/FEException.h"
#include "Forms/FormExpr.h"

#include <span>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <memory>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {

namespace assembly {
class FunctionalKernel;
}

namespace forms {

/**
 * @brief A scalar quantity computed by integrating over a boundary marker
 */
struct BoundaryFunctional {
    enum class Reduction : std::uint8_t {
        Sum,
        Average,
        Max,
        Min
    };

    FormExpr integrand{};
    int boundary_marker{-1};
    std::string name{};
    Reduction reduction{Reduction::Sum};
};

/**
 * @brief Evaluated boundary functional values (named scalar lookup)
 */
class BoundaryFunctionalResults {
public:
    BoundaryFunctionalResults() = default;

    [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }
    [[nodiscard]] std::span<const Real> all() const noexcept { return values_; }

    [[nodiscard]] bool has(std::string_view name) const noexcept
    {
        return name_to_index_.find(std::string(name)) != name_to_index_.end();
    }

    [[nodiscard]] std::optional<std::size_t> tryIndexOf(std::string_view name) const noexcept
    {
        auto it = name_to_index_.find(std::string(name));
        if (it == name_to_index_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    [[nodiscard]] std::size_t indexOf(std::string_view name) const
    {
        auto it = name_to_index_.find(std::string(name));
        FE_THROW_IF(it == name_to_index_.end(), InvalidArgumentException,
                    "BoundaryFunctionalResults::indexOf: unknown functional '" + std::string(name) + "'");
        return it->second;
    }

    void clear()
    {
        values_.clear();
        names_.clear();
        name_to_index_.clear();
    }

    void set(std::string name, Real value)
    {
        FE_THROW_IF(name.empty(), InvalidArgumentException,
                    "BoundaryFunctionalResults::set: empty name");
        auto it = name_to_index_.find(name);
        if (it == name_to_index_.end()) {
            const auto idx = values_.size();
            values_.push_back(value);
            names_.push_back(name);
            name_to_index_.emplace(std::move(name), idx);
            return;
        }
        values_.at(it->second) = value;
    }

    [[nodiscard]] Real get(std::string_view name) const
    {
        auto it = name_to_index_.find(std::string(name));
        FE_THROW_IF(it == name_to_index_.end(), InvalidArgumentException,
                    "BoundaryFunctionalResults::get: unknown functional '" + std::string(name) + "'");
        return values_.at(it->second);
    }

    [[nodiscard]] Real get(std::size_t index) const
    {
        FE_THROW_IF(index >= values_.size(), InvalidArgumentException,
                    "BoundaryFunctionalResults::get: index out of range");
        return values_.at(index);
    }

    [[nodiscard]] std::string_view name(std::size_t index) const
    {
        FE_THROW_IF(index >= names_.size(), InvalidArgumentException,
                    "BoundaryFunctionalResults::name: index out of range");
        return names_.at(index);
    }

private:
    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> values_{};
    std::vector<std::string> names_{};
    std::unordered_map<std::string, std::size_t> name_to_index_{};
};

/**
 * @brief Compile a boundary functional integrand to an Assembly/FunctionalKernel
 *
 * The returned kernel is suitable for `systems::FESystem::evaluateBoundaryFunctional`.
 * The integrand must be scalar-valued and must not contain TestFunction/TrialFunction.
 */
std::shared_ptr<assembly::FunctionalKernel>
compileBoundaryFunctionalKernel(const FormExpr& integrand, int boundary_marker);

std::shared_ptr<assembly::FunctionalKernel>
compileBoundaryFunctionalKernel(const FormExpr& integrand,
                                int boundary_marker,
                                const SymbolicOptions& options);

std::shared_ptr<assembly::FunctionalKernel>
compileBoundaryFunctionalKernel(const BoundaryFunctional& functional);

std::shared_ptr<assembly::FunctionalKernel>
compileBoundaryFunctionalKernel(const BoundaryFunctional& functional,
                                const SymbolicOptions& options);

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_BOUNDARY_FUNCTIONAL_H
