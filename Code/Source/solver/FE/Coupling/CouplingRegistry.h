/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGREGISTRY_H
#define SVMP_FE_COUPLING_COUPLINGREGISTRY_H

/**
 * @file CouplingRegistry.h
 * @brief Explicit registry for reusable coupling contract types.
 */

#include "Coupling/CouplingContract.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

class CouplingRegistry {
public:
    using Factory = std::function<std::unique_ptr<CouplingContract>()>;

    void registerContract(std::string name, Factory factory);

    [[nodiscard]] bool contains(std::string_view name) const noexcept;
    [[nodiscard]] bool supportsMode(std::string_view name, CouplingMode mode) const;
    [[nodiscard]] std::unique_ptr<CouplingContract> create(std::string_view name) const;
    [[nodiscard]] std::vector<std::string> names() const;
    [[nodiscard]] std::vector<std::string> namesSupporting(CouplingMode mode) const;

private:
    struct Entry {
        std::string name;
        Factory factory;
    };

    std::vector<Entry> entries_{};
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGREGISTRY_H
