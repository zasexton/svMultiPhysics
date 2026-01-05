#ifndef SVMP_FE_CONSTITUTIVE_PARAMETERS_H
#define SVMP_FE_CONSTITUTIVE_PARAMETERS_H

#include "Core/FEException.h"
#include "Core/Types.h"

#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace svmp {
namespace FE {
namespace constitutive {

class Parameters {
public:
    using RealGetter = std::function<std::optional<Real>(std::string_view)>;

    Parameters() = default;
    explicit Parameters(RealGetter get_real) : get_real_(std::move(get_real)) {}

    [[nodiscard]] bool empty() const noexcept { return !static_cast<bool>(get_real_); }

    [[nodiscard]] std::optional<Real> getReal(std::string_view key) const
    {
        if (!get_real_) return std::nullopt;
        return get_real_(key);
    }

    [[nodiscard]] Real getRealOr(std::string_view key, Real default_value) const
    {
        const auto v = getReal(key);
        return v.has_value() ? *v : default_value;
    }

    [[nodiscard]] Real requireReal(std::string_view key) const
    {
        const auto v = getReal(key);
        FE_THROW_IF(!v.has_value(), InvalidArgumentException,
                    std::string("Parameters: missing required Real parameter '") + std::string(key) + "'");
        return *v;
    }

private:
    RealGetter get_real_{};
};

} // namespace constitutive
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTITUTIVE_PARAMETERS_H

