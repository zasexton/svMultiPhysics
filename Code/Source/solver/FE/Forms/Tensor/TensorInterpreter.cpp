/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorInterpreter.h"

#include "Forms/Tensor/SymmetryOptimizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

namespace {

constexpr std::size_t idx4(int i, int j, int k, int l) noexcept
{
    return static_cast<std::size_t>((((i * 3) + j) * 3 + k) * 3 + l);
}

[[nodiscard]] std::size_t denseOffsetRowMajor(const TensorSpec& spec,
                                              const std::vector<int>& assignment)
{
    if (spec.rank <= 0) {
        return 0u;
    }
    std::size_t off = 0u;
    for (int ax = 0; ax < spec.rank; ++ax) {
        const int id = spec.axes[static_cast<std::size_t>(ax)];
        const int extent = spec.extents[static_cast<std::size_t>(ax)];
        const int v = (id >= 0 && static_cast<std::size_t>(id) < assignment.size()) ? assignment[static_cast<std::size_t>(id)] : 0;
        off = off * static_cast<std::size_t>(extent) + static_cast<std::size_t>(v);
    }
    return off;
}

[[nodiscard]] Real readTensorElement(const TensorSpec& spec,
                                     std::span<const Real> data,
                                     const std::vector<int>& assignment)
{
    if (spec.rank == 0) {
        return data.empty() ? 0.0 : data[0];
    }

    if (spec.storage == TensorStorageKind::KroneckerDelta) {
        if (spec.rank != 2 || spec.axes.size() != 2u) {
            throw std::runtime_error("TensorInterpreter: KroneckerDelta expects rank-2 tensor");
        }
        const int a_id = spec.axes[0];
        const int b_id = spec.axes[1];
        const int a = (a_id >= 0 && static_cast<std::size_t>(a_id) < assignment.size()) ? assignment[static_cast<std::size_t>(a_id)] : 0;
        const int b = (b_id >= 0 && static_cast<std::size_t>(b_id) < assignment.size()) ? assignment[static_cast<std::size_t>(b_id)] : 0;
        return (a == b) ? 1.0 : 0.0;
    }

    if (spec.storage == TensorStorageKind::Symmetric2) {
        if (spec.rank != 2 || spec.axes.size() != 2u || spec.extents.size() != 2u || spec.extents[0] != spec.extents[1]) {
            throw std::runtime_error("TensorInterpreter: Symmetric2 expects square rank-2 tensor");
        }
        const int dim = spec.extents[0];
        const int i_id = spec.axes[0];
        const int j_id = spec.axes[1];
        const int i = (i_id >= 0 && static_cast<std::size_t>(i_id) < assignment.size()) ? assignment[static_cast<std::size_t>(i_id)] : 0;
        const int j = (j_id >= 0 && static_cast<std::size_t>(j_id) < assignment.size()) ? assignment[static_cast<std::size_t>(j_id)] : 0;
        const int a = std::min(i, j);
        const int b = std::max(i, j);
        const int off = packedIndexSymmetricPair(a, b, dim);
        return data[static_cast<std::size_t>(off)];
    }

    if (spec.storage == TensorStorageKind::Antisymmetric2) {
        if (spec.rank != 2 || spec.axes.size() != 2u || spec.extents.size() != 2u || spec.extents[0] != spec.extents[1]) {
            throw std::runtime_error("TensorInterpreter: Antisymmetric2 expects square rank-2 tensor");
        }
        const int dim = spec.extents[0];
        const int i_id = spec.axes[0];
        const int j_id = spec.axes[1];
        const int i = (i_id >= 0 && static_cast<std::size_t>(i_id) < assignment.size()) ? assignment[static_cast<std::size_t>(i_id)] : 0;
        const int j = (j_id >= 0 && static_cast<std::size_t>(j_id) < assignment.size()) ? assignment[static_cast<std::size_t>(j_id)] : 0;
        if (i == j) {
            return 0.0;
        }
        const bool swap = (j < i);
        const int a = swap ? j : i;
        const int b = swap ? i : j;
        const int off = packedIndexAntisymmetricPair(a, b, dim);
        const Real v = data[static_cast<std::size_t>(off)];
        return swap ? -v : v;
    }

    if (spec.storage == TensorStorageKind::ElasticityVoigt) {
        if (spec.rank != 4 || spec.axes.size() != 4u || spec.extents.size() != 4u ||
            spec.extents[0] != spec.extents[1] || spec.extents[0] != spec.extents[2] || spec.extents[0] != spec.extents[3]) {
            throw std::runtime_error("TensorInterpreter: ElasticityVoigt expects square rank-4 tensor");
        }
        const int dim = spec.extents[0];
        const int i = assignment[static_cast<std::size_t>(spec.axes[0])];
        const int j = assignment[static_cast<std::size_t>(spec.axes[1])];
        const int k = assignment[static_cast<std::size_t>(spec.axes[2])];
        const int l = assignment[static_cast<std::size_t>(spec.axes[3])];
        const int off = packedIndexElasticityVoigt(i, j, k, l, dim);
        return data[static_cast<std::size_t>(off)];
    }

    // Dense fallback.
    const auto off = denseOffsetRowMajor(spec, assignment);
    return data[off];
}

[[nodiscard]] Real baseComponent(const Value<Real>& base,
                                 int base_rank,
                                 const std::array<int, 4>& idx)
{
    if (base_rank == 1) {
        return base.vectorAt(static_cast<std::size_t>(idx[0]));
    }
    if (base_rank == 2) {
        return base.matrixAt(static_cast<std::size_t>(idx[0]), static_cast<std::size_t>(idx[1]));
    }
    if (base_rank == 3) {
        return base.tensor3At(static_cast<std::size_t>(idx[0]),
                              static_cast<std::size_t>(idx[1]),
                              static_cast<std::size_t>(idx[2]));
    }
    if (base_rank == 4) {
        return base.t4[idx4(idx[0], idx[1], idx[2], idx[3])];
    }
    return 0.0;
}

[[nodiscard]] bool baseShapeCompatible(const Value<Real>& base,
                                       int base_rank,
                                       const std::vector<int>& extents)
{
    if (base_rank <= 0) {
        return false;
    }
    if (base_rank == 1) {
        const auto n = static_cast<int>(base.vectorSize());
        return static_cast<int>(extents.size()) == 1 && extents[0] <= n;
    }
    if (base_rank == 2) {
        const auto r = static_cast<int>(base.matrixRows());
        const auto c = static_cast<int>(base.matrixCols());
        return static_cast<int>(extents.size()) == 2 && extents[0] <= r && extents[1] <= c;
    }
    if (base_rank == 3) {
        const auto d0 = static_cast<int>(base.tensor3Dim0());
        const auto d1 = static_cast<int>(base.tensor3Dim1());
        const auto d2 = static_cast<int>(base.tensor3Dim2());
        return static_cast<int>(extents.size()) == 3 && extents[0] <= d0 && extents[1] <= d1 && extents[2] <= d2;
    }
    if (base_rank == 4) {
        // tensor4 is fixed-size 3x3x3x3 in Value<Real>
        return static_cast<int>(extents.size()) == 4 &&
               extents[0] <= 3 && extents[1] <= 3 && extents[2] <= 3 && extents[3] <= 3;
    }
    return false;
}

void fillInputTensor(const TensorSpec& spec,
                     std::vector<Real>& out,
                     const TensorInterpreterCallbacks& cb)
{
    if (!spec.base.isValid() || spec.base.node() == nullptr) {
        throw std::runtime_error("TensorInterpreter: input tensor spec has null base expression");
    }
    if (spec.storage == TensorStorageKind::KroneckerDelta) {
        out.clear();
        return;
    }

    const int base_rank = spec.base_rank;
    if (base_rank <= 0 || base_rank > 4) {
        throw std::runtime_error("TensorInterpreter: base_rank out of supported range (1..4)");
    }
    const auto base_value = cb.eval_value(spec.base);

    // Build base extents (in base axis order) from the tensor extents and axis mapping.
    std::vector<int> base_extents;
    base_extents.reserve(static_cast<std::size_t>(base_rank));
    for (int ax = 0; ax < base_rank; ++ax) {
        const int tpos = spec.base_axis_to_tensor_axis[static_cast<std::size_t>(ax)];
        if (tpos < 0 || tpos >= spec.rank) {
            throw std::runtime_error("TensorInterpreter: invalid base_axis_to_tensor_axis mapping");
        }
        base_extents.push_back(spec.extents[static_cast<std::size_t>(tpos)]);
    }

    if (!baseShapeCompatible(base_value, base_rank, base_extents)) {
        throw std::runtime_error("TensorInterpreter: base tensor value has incompatible shape");
    }

    const auto rank = spec.rank;
    if (rank < 0 || rank > 4) {
        throw std::runtime_error("TensorInterpreter: tensor rank out of supported range (0..4)");
    }

    if (spec.storage == TensorStorageKind::Dense) {
        out.assign(spec.size, 0.0);
        std::array<int, 4> t_idx{};
        std::array<int, 4> b_idx{};

        const auto loop = [&](const auto& self, int depth) -> void {
            if (depth == rank) {
                for (int ax = 0; ax < base_rank; ++ax) {
                    const int tpos = spec.base_axis_to_tensor_axis[static_cast<std::size_t>(ax)];
                    b_idx[static_cast<std::size_t>(ax)] = t_idx[static_cast<std::size_t>(tpos)];
                }
                const Real v = baseComponent(base_value, base_rank, b_idx);
                std::size_t off = 0u;
                for (int k = 0; k < rank; ++k) {
                    off = off * static_cast<std::size_t>(spec.extents[static_cast<std::size_t>(k)]) +
                          static_cast<std::size_t>(t_idx[static_cast<std::size_t>(k)]);
                }
                out[off] = v;
                return;
            }
            const int extent = spec.extents[static_cast<std::size_t>(depth)];
            for (int i = 0; i < extent; ++i) {
                t_idx[static_cast<std::size_t>(depth)] = i;
                self(self, depth + 1);
            }
        };

        loop(loop, 0);
        return;
    }

    if (spec.storage == TensorStorageKind::Symmetric2 || spec.storage == TensorStorageKind::Antisymmetric2) {
        if (rank != 2 || spec.extents.size() != 2u || spec.extents[0] != spec.extents[1]) {
            throw std::runtime_error("TensorInterpreter: symmetric/antisymmetric storage expects square rank-2 tensor");
        }
        const int dim = spec.extents[0];
        out.assign(spec.size, 0.0);

        std::array<int, 4> b_idx{};
        for (int i = 0; i < dim; ++i) {
            const int j0 = (spec.storage == TensorStorageKind::Symmetric2) ? i : (i + 1);
            for (int j = j0; j < dim; ++j) {
                // Tensor indices are (i,j) in storage order.
                const std::array<int, 4> t_idx{{i, j, 0, 0}};
                for (int ax = 0; ax < base_rank; ++ax) {
                    const int tpos = spec.base_axis_to_tensor_axis[static_cast<std::size_t>(ax)];
                    b_idx[static_cast<std::size_t>(ax)] = t_idx[static_cast<std::size_t>(tpos)];
                }
                const Real v = baseComponent(base_value, base_rank, b_idx);
                const int off = (spec.storage == TensorStorageKind::Symmetric2)
                                    ? packedIndexSymmetricPair(i, j, dim)
                                    : packedIndexAntisymmetricPair(i, j, dim);
                out[static_cast<std::size_t>(off)] = v;
            }
        }
        return;
    }

    if (spec.storage == TensorStorageKind::ElasticityVoigt) {
        if (rank != 4 || spec.extents.size() != 4u || spec.extents[0] != spec.extents[1] ||
            spec.extents[0] != spec.extents[2] || spec.extents[0] != spec.extents[3]) {
            throw std::runtime_error("TensorInterpreter: ElasticityVoigt expects square rank-4 tensor");
        }
        const int dim = spec.extents[0];
        out.assign(spec.size, 0.0);
        std::array<int, 4> b_idx{};
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                for (int k = 0; k < dim; ++k) {
                    for (int l = 0; l < dim; ++l) {
                        const std::array<int, 4> t_idx{{i, j, k, l}};
                        for (int ax = 0; ax < base_rank; ++ax) {
                            const int tpos = spec.base_axis_to_tensor_axis[static_cast<std::size_t>(ax)];
                            b_idx[static_cast<std::size_t>(ax)] = t_idx[static_cast<std::size_t>(tpos)];
                        }
                        const Real v = baseComponent(base_value, base_rank, b_idx);
                        const int off = packedIndexElasticityVoigt(i, j, k, l, dim);
                        out[static_cast<std::size_t>(off)] = v;
                    }
                }
            }
        }
        return;
    }

    throw std::runtime_error("TensorInterpreter: unsupported tensor storage kind in input fill");
}

} // namespace

TensorInterpreterResult evalTensorIRScalar(const TensorIR& ir,
                                          const TensorInterpreterCallbacks& cb)
{
    TensorInterpreterResult out;

    if (!ir.program.ok) {
        out.ok = false;
        out.message = ir.program.message.empty() ? "TensorInterpreter: program is invalid" : ir.program.message;
        return out;
    }
    if (!ir.allocation.ok) {
        out.ok = false;
        out.message = ir.allocation.message.empty() ? "TensorInterpreter: allocation plan is invalid" : ir.allocation.message;
        return out;
    }
    if (ir.program.output.rank != 0) {
        out.ok = false;
        out.message = "TensorInterpreter: only scalar outputs are supported";
        return out;
    }
    if (!cb.eval_value || !cb.eval_scalar) {
        out.ok = false;
        out.message = "TensorInterpreter: missing callbacks";
        return out;
    }

    // Thread-local scratch to avoid repeated allocations in interpreter mode.
    thread_local TensorTempWorkspace temps;
    thread_local std::vector<std::vector<Real>> inputs;
    thread_local std::vector<int> assignment;

    temps.reset(ir.allocation, /*batch_size=*/1);

    const auto& p = ir.program;
    const std::size_t ntensors = p.tensors.size();
    if (inputs.size() < ntensors) {
        inputs.resize(ntensors);
    }

    // Determine max index id for assignment storage.
    int max_id = -1;
    for (const auto& t : p.tensors) {
        for (const int id : t.axes) {
            max_id = std::max(max_id, id);
        }
    }
    for (const auto& op : p.ops) {
        for (const auto& li : op.loops) {
            max_id = std::max(max_id, li.id);
            max_id = std::max(max_id, li.lower_bound_id);
        }
    }
    for (const auto& li : p.output_loops) {
        max_id = std::max(max_id, li.id);
        max_id = std::max(max_id, li.lower_bound_id);
    }
    if (max_id < 0) {
        assignment.clear();
    } else {
        assignment.assign(static_cast<std::size_t>(max_id) + 1u, -1);
    }

    // Fill input tensor buffers.
    for (std::size_t tid = 0; tid < ntensors; ++tid) {
        const auto& spec = p.tensors[tid];
        if (!spec.base.isValid() || spec.base.node() == nullptr) {
            continue; // temporary
        }
        if (spec.storage == TensorStorageKind::KroneckerDelta) {
            inputs[tid].clear();
            continue;
        }
        fillInputTensor(spec, inputs[tid], cb);
        if (inputs[tid].size() != spec.size) {
            throw std::runtime_error("TensorInterpreter: input tensor stored size mismatch");
        }
    }

    const auto tensorData = [&](int tensor_id) -> std::span<const Real> {
        if (tensor_id < 0) {
            return {};
        }
        const std::size_t tid = static_cast<std::size_t>(tensor_id);
        if (tid >= ntensors) {
            return {};
        }
        const auto tmp = temps.spanForTensor(tensor_id);
        if (!tmp.empty()) {
            return tmp;
        }
        return std::span<const Real>(inputs[tid].data(), inputs[tid].size());
    };

    const auto tensorDataMutable = [&](int tensor_id) -> std::span<Real> {
        if (tensor_id < 0) {
            return {};
        }
        return temps.spanForTensor(tensor_id);
    };

    // Execute contraction ops in order.
    for (const auto& op : p.ops) {
        if (op.kind != ContractionOp::Kind::Contraction && op.kind != ContractionOp::Kind::Reduction) {
            throw std::runtime_error("TensorInterpreter: unsupported op kind in interpreter (expected Contraction/Reduction)");
        }

        if (op.out < 0 || static_cast<std::size_t>(op.out) >= ntensors) {
            throw std::runtime_error("TensorInterpreter: op.out out of range");
        }
        const auto& out_spec = p.tensors[static_cast<std::size_t>(op.out)];
        auto out_buf = tensorDataMutable(op.out);
        if (out_spec.storage != TensorStorageKind::Dense) {
            throw std::runtime_error("TensorInterpreter: temporaries are expected to be dense");
        }
        if (out_buf.size() != out_spec.size) {
            throw std::runtime_error("TensorInterpreter: output buffer size mismatch");
        }

        const std::size_t out_rank = op.out_axes.size();
        if (op.loops.size() < out_rank) {
            throw std::runtime_error("TensorInterpreter: op loop metadata is inconsistent");
        }
        const std::span<const LoopIndex> out_loops(op.loops.data(), out_rank);
        const std::span<const LoopIndex> sum_loops(op.loops.data() + out_rank, op.loops.size() - out_rank);

        const auto evalSum = [&](const auto& self, std::size_t depth) -> Real {
            if (depth == sum_loops.size()) {
                const auto& lhs_spec = p.tensors[static_cast<std::size_t>(op.lhs)];
                const auto lhs = readTensorElement(lhs_spec, tensorData(op.lhs), assignment);
                if (op.kind == ContractionOp::Kind::Reduction) {
                    return lhs;
                }
                const auto& rhs_spec = p.tensors[static_cast<std::size_t>(op.rhs)];
                const auto rhs = readTensorElement(rhs_spec, tensorData(op.rhs), assignment);
                return lhs * rhs;
            }

            const auto& li = sum_loops[depth];
            int start = 0;
            if (li.lower_bound_id >= 0 &&
                static_cast<std::size_t>(li.lower_bound_id) < assignment.size() &&
                assignment[static_cast<std::size_t>(li.lower_bound_id)] >= 0) {
                start = assignment[static_cast<std::size_t>(li.lower_bound_id)] + li.lower_bound_offset;
            }
            start = std::max(0, start);
            const int end = li.extent;

            Real acc = 0.0;
            for (int v = start; v < end; ++v) {
                assignment[static_cast<std::size_t>(li.id)] = v;
                acc += self(self, depth + 1u);
            }
            return acc;
        };

        const auto evalOut = [&](const auto& self, std::size_t depth) -> void {
            if (depth == out_loops.size()) {
                const Real s = evalSum(evalSum, 0u);
                const std::size_t off = denseOffsetRowMajor(out_spec, assignment);
                out_buf[off] = s;
                return;
            }
            const auto& li = out_loops[depth];
            int start = 0;
            if (li.lower_bound_id >= 0 &&
                static_cast<std::size_t>(li.lower_bound_id) < assignment.size() &&
                assignment[static_cast<std::size_t>(li.lower_bound_id)] >= 0) {
                start = assignment[static_cast<std::size_t>(li.lower_bound_id)] + li.lower_bound_offset;
            }
            start = std::max(0, start);
            const int end = li.extent;
            for (int v = start; v < end; ++v) {
                assignment[static_cast<std::size_t>(li.id)] = v;
                self(self, depth + 1u);
            }
        };

        evalOut(evalOut, 0u);
    }

    // Accumulate contributions to scalar output.
    Real result = 0.0;
    for (const auto& c : p.contributions) {
        const Real pref = cb.eval_scalar(c.scalar);
        Real t = 1.0;
        if (c.tensor_id >= 0) {
            const auto& spec = p.tensors[static_cast<std::size_t>(c.tensor_id)];
            if (spec.rank != 0) {
                throw std::runtime_error("TensorInterpreter: scalar output expects rank-0 contribution tensor");
            }
            const auto data = tensorData(c.tensor_id);
            t = data.empty() ? 0.0 : data[0];
        }
        result += pref * t;
    }

    out.ok = true;
    out.value = result;
    return out;
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

