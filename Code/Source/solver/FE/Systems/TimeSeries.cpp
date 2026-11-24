/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/TimeSeries.h"

#include "Core/FEException.h"
#include "Dofs/DofHandler.h"
#include "Systems/FESystem.h"
#include "Systems/FieldBinding.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <iterator>
#include <limits>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Mesh.h"
#endif

namespace svmp {
namespace FE {
namespace systems {

TimeSeries::TimeSeries(std::vector<Real> times, std::vector<Real> values, TimeSeriesEnd end, int components)
    : times_(std::move(times))
    , values_(std::move(values))
    , end_(end)
    , components_(components)
{
    FE_THROW_IF(components_ < 1, InvalidArgumentException, "TimeSeries: at least one component is required");
    FE_THROW_IF(times_.size() < 2u, InvalidArgumentException, "TimeSeries: at least two times are required");
    FE_THROW_IF(values_.size() != times_.size() * static_cast<std::size_t>(components_), InvalidArgumentException,
                "TimeSeries: expected " + std::to_string(times_.size() * static_cast<std::size_t>(components_)) +
                    " values, got " + std::to_string(values_.size()));
    for (std::size_t i = 0; i < times_.size(); ++i) {
        FE_THROW_IF(!std::isfinite(times_[i]), InvalidArgumentException, "TimeSeries: a time is not finite");
        FE_THROW_IF(i > 0 && !(times_[i] > times_[i - 1]), InvalidArgumentException,
                    "TimeSeries: the times must increase strictly");
    }
    FE_THROW_IF(std::any_of(values_.begin(), values_.end(), [](Real v) { return !std::isfinite(v); }),
                InvalidArgumentException, "TimeSeries: a value is not finite");
    FE_THROW_IF(end_ != TimeSeriesEnd::Clamp && end_ != TimeSeriesEnd::Periodic, InvalidArgumentException,
                "TimeSeries: invalid end behavior");
}

Real TimeSeries::sample(std::size_t time_index, int component) const
{
    FE_THROW_IF(time_index >= times_.size(), InvalidArgumentException, "TimeSeries::sample: time index out of range");
    FE_THROW_IF(component < 0 || component >= components_, InvalidArgumentException,
                "TimeSeries::sample: component out of range");
    return values_[time_index * static_cast<std::size_t>(components_) + static_cast<std::size_t>(component)];
}

Real TimeSeries::value(Real time, int component) const
{
    FE_THROW_IF(component < 0 || component >= components_, InvalidArgumentException,
                "TimeSeries::value: component out of range");

    // The interpolation of the temporal-values files, operation by operation.
    const std::size_t last = times_.size() - 1u;
    const Real t0 = times_.front();
    const Real t1 = times_.back();
    const Real span = t1 - t0;
    Real tt = time;
    if (end_ == TimeSeriesEnd::Periodic && span > 0.0 && std::isfinite(span)) {
        tt = std::fmod(time - t0, span);
        if (tt < 0.0) {
            tt += span;
        }
        tt += t0;
    }
    if (tt <= t0) {
        return sample(0, component);
    }
    if (tt >= t1) {
        return sample(last, component);
    }

    const auto upper = std::upper_bound(times_.begin(), times_.end(), tt);
    const auto i1 = static_cast<std::size_t>(std::distance(times_.begin(), upper));
    const std::size_t i0 = i1 > 0u ? i1 - 1u : 0u;
    const Real ta = times_[i0];
    const Real tb = times_[i1];
    const Real dt = tb - ta;
    const Real alpha = (dt > 0.0) ? ((tt - ta) / dt) : 0.0;
    const Real va = sample(i0, component);
    const Real vb = sample(i1, component);
    return (1.0 - alpha) * va + alpha * vb;
}

TimeSeries readTimeSeries(const std::string& path, TimeSeriesEnd end, int components)
{
    const std::string where = "FE::systems::readTimeSeries('" + path + "'): ";
    FE_THROW_IF(components < 1, InvalidArgumentException, where + "at least one component is required");
    std::ifstream in(path);
    FE_THROW_IF(!in.is_open(), InvalidArgumentException, where + "cannot open the file");

    long long count = 0;
    long long legacy_count = 0;
    in >> count >> legacy_count;
    FE_THROW_IF(in.fail() || count < 2, InvalidArgumentException,
                where + "the file must start with '<number of times> <count>' and hold at least two times");

    std::vector<Real> times(static_cast<std::size_t>(count));
    std::vector<Real> values(static_cast<std::size_t>(count) * static_cast<std::size_t>(components));
    for (long long i = 0; i < count; ++i) {
        const auto row = static_cast<std::size_t>(i);
        in >> times[row];
        FE_THROW_IF(in.fail(), InvalidArgumentException, where + "cannot read the time of row " + std::to_string(i));
        for (int c = 0; c < components; ++c) {
            in >> values[row * static_cast<std::size_t>(components) + static_cast<std::size_t>(c)];
            FE_THROW_IF(in.fail(), InvalidArgumentException,
                        where + "cannot read component " + std::to_string(c) + " of row " + std::to_string(i));
        }
    }
    return TimeSeries(std::move(times), std::move(values), end, components);
}

forms::FormExpr timeFunction(FESystem& system, std::string name, std::function<Real(Real)> f)
{
    const std::string where = "FE::systems::timeFunction('" + name + "'): ";
    FE_THROW_IF(system.isSetup(), InvalidArgumentException, where + "time functions cannot be declared after setup");
    FE_THROW_IF(name.empty(), InvalidArgumentException, where + "the name must be nonempty");
    FE_THROW_IF(!f, InvalidArgumentException, where + "the function is empty");
    auto& store = system.attachment<TimeFunctionStore>();
    FE_THROW_IF(std::any_of(store.declarations.begin(), store.declarations.end(),
                            [&](const TimeFunctionDeclaration& existing) { return existing.name == name; }),
                InvalidArgumentException, where + "a time function of this name is already declared");
    store.declarations.push_back(TimeFunctionDeclaration{name, std::move(f)});
    return forms::FormExpr::parameter(std::move(name));
}

std::span<const TimeFunctionDeclaration> timeFunctions(const FESystem& system) noexcept
{
    const auto* store = system.findAttachment<TimeFunctionStore>();
    if (store == nullptr) {
        return {};
    }
    return store->declarations;
}

const std::function<Real(Real)>* findTimeFunction(const FESystem& system, std::string_view name)
{
    for (const auto& declaration : timeFunctions(system)) {
        if (declaration.name == name) {
            return &declaration.function;
        }
    }
    return nullptr;
}

namespace {

/// The fields of meshVertexDataField on a system (its attachment), with the
/// time of their last fill.
struct MeshVertexDataFieldStore {
    struct Entry {
        FieldId field{INVALID_FIELD_ID};
        std::string name{};
        MeshVertexValuesAtTime values{};
        bool filled{false};
        Real filled_time{0.0};
    };
    std::vector<Entry> entries{};
};

} // namespace

forms::FormExpr meshVertexDataField(FESystem& system, std::string name, MeshVertexData data)
{
    const std::string where = "FE::systems::meshVertexDataField('" + name + "'): ";
    FE_THROW_IF(system.isSetup(), InvalidArgumentException, where + "vertex data cannot be declared after setup");
    FE_THROW_IF(name.empty(), InvalidArgumentException, where + "the name must be nonempty");
    FE_THROW_IF(!data.space, InvalidArgumentException, where + "the space is empty");
    FE_THROW_IF(!data.values, InvalidArgumentException, where + "the function is empty");
    FE_THROW_IF(data.components < 1, InvalidArgumentException, where + "at least one component is required");
    auto& store = system.attachment<MeshVertexDataFieldStore>();
    FE_THROW_IF(std::any_of(store.entries.begin(), store.entries.end(),
                            [&](const MeshVertexDataFieldStore::Entry& entry) { return entry.name == name; }),
                InvalidArgumentException, where + "vertex data of this name are already declared");

    FieldBinding binding(system, "FE::systems::meshVertexDataField");
    const auto space = data.space;
    auto bound = binding.prescribedSymbol(FieldSpec{.name = name,
                                                    .space = space,
                                                    .components = data.components,
                                                    .source_kind = FieldSourceKind::PrescribedData},
                                          /*auto_register=*/true, "vertex data", name);
    store.entries.push_back(MeshVertexDataFieldStore::Entry{
        .field = bound.id, .name = std::move(name), .values = std::move(data.values)});
    return bound.value;
}

void refreshMeshVertexDataFields(FESystem& system, Real time)
{
    auto* store = system.findAttachment<MeshVertexDataFieldStore>();
    if (store == nullptr) {
        return;
    }
    for (auto& entry : store->entries) {
        if (entry.filled && entry.filled_time == time) {
            continue;
        }
        const std::string where = "FE::systems::refreshMeshVertexDataFields('" + entry.name + "'): ";
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
        const auto components = static_cast<std::size_t>(std::max(1, system.fieldRecord(entry.field).components));
        const auto n_vertices = static_cast<std::size_t>(system.singleMesh(where).n_vertices());
        std::vector<Real> values(n_vertices * components, std::numeric_limits<Real>::quiet_NaN());
        entry.values(time, std::span<Real>(values));

        const auto n_dofs = static_cast<std::size_t>(system.fieldDofHandler(entry.field).getNumDofs());
        std::vector<Real> coefficients(n_dofs, Real{0.0});
        std::vector<std::uint8_t> assigned(n_dofs, std::uint8_t{0});
        const auto projection = system.projectMeshVertexValuesToFieldCoefficients(
            entry.field, std::span<const Real>(values), components, std::span<Real>(coefficients),
            std::span<std::uint8_t>(assigned), where);
        FE_THROW_IF(projection.unassigned_dofs != 0u, InvalidArgumentException,
                    where + std::to_string(projection.unassigned_dofs) +
                        " DOF(s) of the field have no node at a mesh vertex");
        for (std::size_t dof = 0; dof < n_dofs; ++dof) {
            FE_THROW_IF(assigned[dof] != 0u && !std::isfinite(coefficients[dof]), InvalidArgumentException,
                        where + "the data give DOF " + std::to_string(dof) + " of the field no finite value at time " +
                            std::to_string(time));
        }
        system.setPrescribedFieldCoefficients(entry.field, std::span<const Real>(coefficients));
        entry.filled = true;
        entry.filled_time = time;
#else
        FE_THROW(InvalidArgumentException, where + "vertex data require FE built with the mesh");
#endif
    }
}

} // namespace systems
} // namespace FE
} // namespace svmp
