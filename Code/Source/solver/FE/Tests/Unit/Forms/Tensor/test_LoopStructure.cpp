/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Forms/Index.h"
#include "Forms/Tensor/LoopStructure.h"
#include "Forms/Tensor/SymmetryOptimizer.h"

#include <array>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace svmp::FE::forms::tensor {

namespace {

using Mat3 = std::array<std::array<Real, 3>, 3>;

[[nodiscard]] Mat3 makeMat(Real scale)
{
    Mat3 m{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                scale * (1.0 + static_cast<Real>(i) + 0.1 * static_cast<Real>(j));
        }
    }
    return m;
}

[[nodiscard]] std::size_t offsetRowMajor(const std::vector<int>& extents,
                                         const std::vector<int>& idx)
{
    std::size_t off = 0;
    std::size_t stride = 1;
    for (std::size_t k = extents.size(); k-- > 0;) {
        off += static_cast<std::size_t>(idx[k]) * stride;
        stride *= static_cast<std::size_t>(extents[k]);
    }
    return off;
}

struct EvalState {
    const LoopNestProgram& p;
    std::vector<std::vector<Real>> data; // per tensor_id
    std::vector<int> assignment;         // id -> value
};

[[nodiscard]] std::vector<Real> loadDenseBase(const TensorSpec& spec)
{
    // Supports matrix coefficients and AsTensor constants for tests.
    const auto* n = spec.base.node();
    if (n == nullptr) {
        throw std::runtime_error("LoopStructure test: base node is null");
    }

    const int rank = spec.base_rank;
    const auto extents = [&]() {
        std::vector<int> e;
        e.reserve(static_cast<std::size_t>(rank));
        for (int ax = 0; ax < rank; ++ax) {
            const int tpos = spec.base_axis_to_tensor_axis[static_cast<std::size_t>(ax)];
            e.push_back(spec.extents[static_cast<std::size_t>(tpos)]);
        }
        return e;
    }();

    if (n->type() == FormExprType::Coefficient) {
        if (rank == 2) {
            const auto* f = n->matrixCoefficient();
            if (f == nullptr) {
                throw std::runtime_error("LoopStructure test: expected matrix coefficient");
            }
            const auto m = (*f)(0.0, 0.0, 0.0);
            std::vector<Real> out(9, 0.0);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    out[static_cast<std::size_t>(i * 3 + j)] =
                        m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                }
            }
            return out;
        }
    }

    if (n->type() == FormExprType::AsTensor) {
        const int rows = n->tensorRows().value_or(0);
        const int cols = n->tensorCols().value_or(0);
        if (rank != 2) {
            throw std::runtime_error("LoopStructure test: AsTensor base_rank must be 2");
        }
        if (rows != extents[0] || cols != extents[1]) {
            throw std::runtime_error("LoopStructure test: AsTensor extents mismatch");
        }
        std::vector<Real> out(static_cast<std::size_t>(rows * cols), 0.0);
        const auto kids = n->childrenShared();
        if (kids.size() != out.size()) {
            throw std::runtime_error("LoopStructure test: AsTensor child count mismatch");
        }
        for (std::size_t k = 0; k < kids.size(); ++k) {
            if (!kids[k] || kids[k]->type() != FormExprType::Constant) {
                throw std::runtime_error("LoopStructure test: AsTensor entries must be constants");
            }
            out[k] = kids[k]->constantValue().value_or(0.0);
        }
        return out;
    }

    if (n->type() == FormExprType::SymmetricPart || n->type() == FormExprType::SkewPart) {
        const auto kids = n->childrenShared();
        if (kids.size() != 1u || !kids[0]) {
            throw std::runtime_error("LoopStructure test: sym/skew base missing child");
        }
        TensorSpec child = spec;
        child.base = FormExpr(kids[0]);
        const auto a = loadDenseBase(child);
        if (rank != 2 || extents.size() != 2u) {
            throw std::runtime_error("LoopStructure test: sym/skew expects rank-2 base");
        }
        const int rows = extents[0];
        const int cols = extents[1];
        std::vector<Real> out(static_cast<std::size_t>(rows * cols), 0.0);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const std::size_t rc = static_cast<std::size_t>(r * cols + c);
                const std::size_t cr = static_cast<std::size_t>(c * cols + r);
                if (n->type() == FormExprType::SymmetricPart) {
                    out[rc] = 0.5 * (a[rc] + a[cr]);
                } else {
                    out[rc] = 0.5 * (a[rc] - a[cr]);
                }
            }
        }
        return out;
    }

    throw std::runtime_error("LoopStructure test: unsupported base tensor type");
}

[[nodiscard]] std::vector<Real> loadTensorData(const TensorSpec& spec)
{
    if (spec.storage == TensorStorageKind::KroneckerDelta) {
        return {};
    }
    if (spec.storage == TensorStorageKind::Dense) {
        return loadDenseBase(spec);
    }
    if (spec.storage == TensorStorageKind::Symmetric2) {
        const auto a = loadDenseBase(spec);
        if (spec.rank != 2 || spec.extents.size() != 2u || spec.extents[0] != spec.extents[1]) {
            throw std::runtime_error("LoopStructure test: Symmetric2 must be square rank-2");
        }
        const int dim = spec.extents[0];
        std::vector<Real> out(spec.size, 0.0);
        for (int i = 0; i < dim; ++i) {
            for (int j = i; j < dim; ++j) {
                const int off = packedIndexSymmetricPair(i, j, dim);
                out[static_cast<std::size_t>(off)] = a[static_cast<std::size_t>(i * dim + j)];
            }
        }
        return out;
    }
    if (spec.storage == TensorStorageKind::Antisymmetric2) {
        const auto a = loadDenseBase(spec);
        if (spec.rank != 2 || spec.extents.size() != 2u || spec.extents[0] != spec.extents[1]) {
            throw std::runtime_error("LoopStructure test: Antisymmetric2 must be square rank-2");
        }
        const int dim = spec.extents[0];
        std::vector<Real> out(spec.size, 0.0);
        for (int i = 0; i < dim; ++i) {
            for (int j = i + 1; j < dim; ++j) {
                const int off = packedIndexAntisymmetricPair(i, j, dim);
                out[static_cast<std::size_t>(off)] = a[static_cast<std::size_t>(i * dim + j)];
            }
        }
        return out;
    }
    throw std::runtime_error("LoopStructure test: unsupported tensor storage kind");
}

[[nodiscard]] int extentOfId(const LoopNestProgram& p, int id)
{
    for (const auto& t : p.tensors) {
        for (std::size_t k = 0; k < t.axes.size(); ++k) {
            if (t.axes[k] == id) return t.extents[k];
        }
    }
    for (std::size_t k = 0; k < p.output.axes.size(); ++k) {
        if (p.output.axes[k] == id) return p.output.extents[k];
    }
    return 0;
}

Real readTensor(EvalState& st, int tensor_id)
{
    const auto& spec = st.p.tensors[static_cast<std::size_t>(tensor_id)];
    if (spec.rank == 0) {
        return st.data[static_cast<std::size_t>(tensor_id)][0];
    }

    if (spec.storage == TensorStorageKind::KroneckerDelta) {
        if (spec.rank != 2) {
            throw std::runtime_error("LoopStructure test: KroneckerDelta must have rank 2");
        }
        const int a = st.assignment[static_cast<std::size_t>(spec.axes[0])];
        const int b = st.assignment[static_cast<std::size_t>(spec.axes[1])];
        return (a == b) ? 1.0 : 0.0;
    }

    if (spec.storage == TensorStorageKind::Symmetric2) {
        if (spec.rank != 2 || spec.extents.size() != 2u || spec.extents[0] != spec.extents[1]) {
            throw std::runtime_error("LoopStructure test: Symmetric2 must be square rank-2");
        }
        const int dim = spec.extents[0];
        const int i = st.assignment[static_cast<std::size_t>(spec.axes[0])];
        const int j = st.assignment[static_cast<std::size_t>(spec.axes[1])];
        const int a = std::min(i, j);
        const int b = std::max(i, j);
        const int off = packedIndexSymmetricPair(a, b, dim);
        return st.data[static_cast<std::size_t>(tensor_id)][static_cast<std::size_t>(off)];
    }

    if (spec.storage == TensorStorageKind::Antisymmetric2) {
        if (spec.rank != 2 || spec.extents.size() != 2u || spec.extents[0] != spec.extents[1]) {
            throw std::runtime_error("LoopStructure test: Antisymmetric2 must be square rank-2");
        }
        const int dim = spec.extents[0];
        const int i = st.assignment[static_cast<std::size_t>(spec.axes[0])];
        const int j = st.assignment[static_cast<std::size_t>(spec.axes[1])];
        if (i == j) return 0.0;
        const int a = std::min(i, j);
        const int b = std::max(i, j);
        const int off = packedIndexAntisymmetricPair(a, b, dim);
        const Real v = st.data[static_cast<std::size_t>(tensor_id)][static_cast<std::size_t>(off)];
        return (i < j) ? v : -v;
    }

    // Inputs are stored in base layout (base_rank); temporaries are stored in tensor layout (rank).
    if (spec.base.isValid() && spec.base.node() != nullptr) {
        const int rank = spec.base_rank;
        std::vector<int> base_extents;
        std::vector<int> base_idx;
        base_extents.reserve(static_cast<std::size_t>(rank));
        base_idx.reserve(static_cast<std::size_t>(rank));
        for (int ax = 0; ax < rank; ++ax) {
            const int tpos = spec.base_axis_to_tensor_axis[static_cast<std::size_t>(ax)];
            const int id = spec.axes[static_cast<std::size_t>(tpos)];
            base_extents.push_back(spec.extents[static_cast<std::size_t>(tpos)]);
            base_idx.push_back(st.assignment[static_cast<std::size_t>(id)]);
        }
        const std::size_t off = offsetRowMajor(base_extents, base_idx);
        return st.data[static_cast<std::size_t>(tensor_id)][off];
    } else {
        std::vector<int> tensor_idx;
        tensor_idx.reserve(spec.axes.size());
        for (const int id : spec.axes) {
            tensor_idx.push_back(st.assignment[static_cast<std::size_t>(id)]);
        }
        const std::size_t off = offsetRowMajor(spec.extents, tensor_idx);
        return st.data[static_cast<std::size_t>(tensor_id)][off];
    }
}

void writeTensor(EvalState& st, int tensor_id, Real value)
{
    const auto& spec = st.p.tensors[static_cast<std::size_t>(tensor_id)];
    if (spec.rank == 0) {
        st.data[static_cast<std::size_t>(tensor_id)][0] = value;
        return;
    }
    if (spec.storage == TensorStorageKind::Symmetric2) {
        if (spec.rank != 2 || spec.extents.size() != 2u || spec.extents[0] != spec.extents[1]) {
            throw std::runtime_error("LoopStructure test: Symmetric2 must be square rank-2");
        }
        const int dim = spec.extents[0];
        const int i = st.assignment[static_cast<std::size_t>(spec.axes[0])];
        const int j = st.assignment[static_cast<std::size_t>(spec.axes[1])];
        const int a = std::min(i, j);
        const int b = std::max(i, j);
        const int off = packedIndexSymmetricPair(a, b, dim);
        st.data[static_cast<std::size_t>(tensor_id)][static_cast<std::size_t>(off)] = value;
        return;
    }
    if (spec.storage == TensorStorageKind::Antisymmetric2) {
        if (spec.rank != 2 || spec.extents.size() != 2u || spec.extents[0] != spec.extents[1]) {
            throw std::runtime_error("LoopStructure test: Antisymmetric2 must be square rank-2");
        }
        const int dim = spec.extents[0];
        const int i = st.assignment[static_cast<std::size_t>(spec.axes[0])];
        const int j = st.assignment[static_cast<std::size_t>(spec.axes[1])];
        if (i == j) return;
        const int a = std::min(i, j);
        const int b = std::max(i, j);
        const int off = packedIndexAntisymmetricPair(a, b, dim);
        const Real stored = (i < j) ? value : -value;
        st.data[static_cast<std::size_t>(tensor_id)][static_cast<std::size_t>(off)] = stored;
        return;
    }
    std::vector<int> tensor_idx;
    tensor_idx.reserve(spec.axes.size());
    for (const int id : spec.axes) {
        tensor_idx.push_back(st.assignment[static_cast<std::size_t>(id)]);
    }
    const std::size_t off = offsetRowMajor(spec.extents, tensor_idx);
    st.data[static_cast<std::size_t>(tensor_id)][off] = value;
}

Real evalScalarPrefactor(const FormExpr& expr)
{
    if (!expr.isValid() || expr.node() == nullptr) {
        throw std::runtime_error("LoopStructure test: invalid scalar prefactor expression");
    }
    const auto* n = expr.node();
    if (n->type() == FormExprType::Constant) {
        return n->constantValue().value_or(0.0);
    }
    if (n->type() == FormExprType::Negate) {
        const auto kids = n->childrenShared();
        if (kids.size() != 1u || !kids[0]) {
            throw std::runtime_error("LoopStructure test: negate child missing");
        }
        return -evalScalarPrefactor(FormExpr(kids[0]));
    }
    if (n->type() == FormExprType::Multiply) {
        const auto kids = n->childrenShared();
        if (kids.size() != 2u || !kids[0] || !kids[1]) {
            throw std::runtime_error("LoopStructure test: multiply children missing");
        }
        return evalScalarPrefactor(FormExpr(kids[0])) * evalScalarPrefactor(FormExpr(kids[1]));
    }
    throw std::runtime_error("LoopStructure test: unsupported scalar prefactor: " + expr.toString());
}

Real evaluateProgramScalar(const LoopNestProgram& p)
{
    if (!p.ok || !p.isScalar()) {
        throw std::runtime_error("LoopStructure test: invalid program");
    }

    // Find max index id.
    int max_id = -1;
    for (const auto& t : p.tensors) {
        for (const int id : t.axes) max_id = std::max(max_id, id);
    }
    for (const int id : p.output.axes) max_id = std::max(max_id, id);

    EvalState st{p, {}, {}};
    st.assignment.assign(static_cast<std::size_t>(max_id + 1), 0);
    st.data.resize(p.tensors.size());

    // Load inputs.
    for (std::size_t tid = 0; tid < p.tensors.size(); ++tid) {
        const auto& spec = p.tensors[tid];
        if (!spec.base.isValid() || spec.base.node() == nullptr) {
            // Temporary (or scalar); allocate.
            const std::size_t sz = (spec.rank == 0) ? 1u : spec.size;
            st.data[tid].assign(sz, 0.0);
            continue;
        }
        st.data[tid] = loadTensorData(spec);
    }

    // Execute ops.
    for (const auto& op : p.ops) {
        const auto loop_out = [&](const auto& self, std::size_t k) -> void {
            if (k == op.out_axes.size()) {
                if (op.kind == ContractionOp::Kind::Contraction) {
                    Real acc = 0.0;
                    const auto loop_sum = [&](const auto& self_sum, std::size_t sidx) -> void {
                        if (sidx == op.sum_axes.size()) {
                            acc += readTensor(st, op.lhs) * readTensor(st, op.rhs);
                            return;
                        }
                        const int id = op.sum_axes[sidx];
                        const int e = extentOfId(p, id);
                        if (e <= 0) throw std::runtime_error("LoopStructure test: missing extent for index id");
                        for (int v = 0; v < e; ++v) {
                            st.assignment[static_cast<std::size_t>(id)] = v;
                            self_sum(self_sum, sidx + 1);
                        }
                    };
                    loop_sum(loop_sum, 0);
                    writeTensor(st, op.out, acc);
                    return;
                }
                if (op.kind == ContractionOp::Kind::Reduction) {
                    Real acc = 0.0;
                    const auto loop_sum = [&](const auto& self_sum, std::size_t sidx) -> void {
                        if (sidx == op.sum_axes.size()) {
                            acc += readTensor(st, op.lhs);
                            return;
                        }
                        const int id = op.sum_axes[sidx];
                        const int e = extentOfId(p, id);
                        if (e <= 0) throw std::runtime_error("LoopStructure test: missing extent for index id");
                        for (int v = 0; v < e; ++v) {
                            st.assignment[static_cast<std::size_t>(id)] = v;
                            self_sum(self_sum, sidx + 1);
                        }
                    };
                    loop_sum(loop_sum, 0);
                    writeTensor(st, op.out, acc);
                    return;
                }
                throw std::runtime_error("LoopStructure test: unsupported op kind");
                return;
            }

            const int id = op.out_axes[k];
            const int e = extentOfId(p, id);
            if (e <= 0) throw std::runtime_error("LoopStructure test: missing extent for index id");
            for (int v = 0; v < e; ++v) {
                st.assignment[static_cast<std::size_t>(id)] = v;
                self(self, k + 1);
            }
        };

        loop_out(loop_out, 0);
    }

    // Accumulate scalar output from contributions.
    Real out = 0.0;
    for (const auto& c : p.contributions) {
        const Real s = evalScalarPrefactor(c.scalar);
        if (c.tensor_id < 0) {
            out += s;
            continue;
        }
        out += s * st.data[static_cast<std::size_t>(c.tensor_id)][0];
    }
    return out;
}

std::vector<Real> evaluateProgramTensor(const LoopNestProgram& p)
{
    if (!p.ok || p.isScalar()) {
        throw std::runtime_error("LoopStructure test: invalid tensor program");
    }
    if (p.output.storage == TensorStorageKind::KroneckerDelta) {
        // No stored entries; delta is evaluated procedurally.
        return {};
    }
    if (p.output_loops.empty()) {
        throw std::runtime_error("LoopStructure test: tensor output requires output_loops metadata");
    }

    // Find max index id.
    int max_id = -1;
    for (const auto& t : p.tensors) {
        for (const int id : t.axes) max_id = std::max(max_id, id);
    }
    for (const int id : p.output.axes) max_id = std::max(max_id, id);

    EvalState st{p, {}, {}};
    st.assignment.assign(static_cast<std::size_t>(max_id + 1), 0);
    st.data.resize(p.tensors.size());

    // Load inputs.
    for (std::size_t tid = 0; tid < p.tensors.size(); ++tid) {
        const auto& spec = p.tensors[tid];
        if (!spec.base.isValid() || spec.base.node() == nullptr) {
            const std::size_t sz = (spec.rank == 0) ? 1u : spec.size;
            st.data[tid].assign(sz, 0.0);
            continue;
        }
        st.data[tid] = loadTensorData(spec);
    }

    // Execute ops (rectangular loop nests).
    for (const auto& op : p.ops) {
        const auto loop_out = [&](const auto& self, std::size_t k) -> void {
            if (k == op.out_axes.size()) {
                if (op.kind == ContractionOp::Kind::Contraction) {
                    Real acc = 0.0;
                    const auto loop_sum = [&](const auto& self_sum, std::size_t sidx) -> void {
                        if (sidx == op.sum_axes.size()) {
                            acc += readTensor(st, op.lhs) * readTensor(st, op.rhs);
                            return;
                        }
                        const int id = op.sum_axes[sidx];
                        const int e = extentOfId(p, id);
                        if (e <= 0) throw std::runtime_error("LoopStructure test: missing extent for index id");
                        for (int v = 0; v < e; ++v) {
                            st.assignment[static_cast<std::size_t>(id)] = v;
                            self_sum(self_sum, sidx + 1);
                        }
                    };
                    loop_sum(loop_sum, 0);
                    writeTensor(st, op.out, acc);
                    return;
                }
                if (op.kind == ContractionOp::Kind::Reduction) {
                    Real acc = 0.0;
                    const auto loop_sum = [&](const auto& self_sum, std::size_t sidx) -> void {
                        if (sidx == op.sum_axes.size()) {
                            acc += readTensor(st, op.lhs);
                            return;
                        }
                        const int id = op.sum_axes[sidx];
                        const int e = extentOfId(p, id);
                        if (e <= 0) throw std::runtime_error("LoopStructure test: missing extent for index id");
                        for (int v = 0; v < e; ++v) {
                            st.assignment[static_cast<std::size_t>(id)] = v;
                            self_sum(self_sum, sidx + 1);
                        }
                    };
                    loop_sum(loop_sum, 0);
                    writeTensor(st, op.out, acc);
                    return;
                }
                throw std::runtime_error("LoopStructure test: unsupported op kind");
            }

            const int id = op.out_axes[k];
            const int e = extentOfId(p, id);
            if (e <= 0) throw std::runtime_error("LoopStructure test: missing extent for index id");
            for (int v = 0; v < e; ++v) {
                st.assignment[static_cast<std::size_t>(id)] = v;
                self(self, k + 1);
            }
        };

        loop_out(loop_out, 0);
    }

    // Materialize output in its stored (possibly packed) representation.
    std::vector<Real> out(p.output.size, 0.0);
    const auto write_out = [&](Real value) -> void {
        if (p.output.rank != 2 || p.output.axes.size() != 2u || p.output.extents.size() != 2u) {
            throw std::runtime_error("LoopStructure test: only rank-2 tensor output supported");
        }
        const int i = st.assignment[static_cast<std::size_t>(p.output.axes[0])];
        const int j = st.assignment[static_cast<std::size_t>(p.output.axes[1])];
        if (p.output.storage == TensorStorageKind::Dense) {
            const std::size_t off = offsetRowMajor(p.output.extents, {i, j});
            out[off] = value;
            return;
        }
        if (p.output.storage == TensorStorageKind::Symmetric2) {
            const int dim = p.output.extents[0];
            const int off = packedIndexSymmetricPair(i, j, dim);
            out[static_cast<std::size_t>(off)] = value;
            return;
        }
        if (p.output.storage == TensorStorageKind::Antisymmetric2) {
            const int dim = p.output.extents[0];
            const int off = packedIndexAntisymmetricPair(i, j, dim);
            out[static_cast<std::size_t>(off)] = value;
            return;
        }
        throw std::runtime_error("LoopStructure test: unsupported output storage kind");
    };

    const auto loop_out = [&](const auto& self, std::size_t k) -> void {
        if (k == p.output_loops.size()) {
            Real acc = 0.0;
            for (const auto& c : p.contributions) {
                if (c.tensor_id < 0) {
                    throw std::runtime_error("LoopStructure test: tensor output cannot have scalar-only contributions");
                }
                acc += evalScalarPrefactor(c.scalar) * readTensor(st, c.tensor_id);
            }
            write_out(acc);
            return;
        }

        const auto& loop = p.output_loops[k];
        const int id = loop.id;
        const int e = loop.extent;
        if (e <= 0) throw std::runtime_error("LoopStructure test: invalid loop extent");
        int start = 0;
        if (loop.lower_bound_id >= 0) {
            start = st.assignment[static_cast<std::size_t>(loop.lower_bound_id)] + loop.lower_bound_offset;
        }
        for (int v = start; v < e; ++v) {
            st.assignment[static_cast<std::size_t>(id)] = v;
            self(self, k + 1);
        }
    };
    loop_out(loop_out, 0);

    return out;
}

} // namespace

TEST(LoopStructure, GeneratesOptimizedChainContraction)
{
    const auto A = FormExpr::coefficient("A", [=](Real, Real, Real) { return makeMat(1.0); });
    const auto B = FormExpr::coefficient("B", [=](Real, Real, Real) { return makeMat(2.0); });
    const auto C = FormExpr::coefficient("C", [=](Real, Real, Real) { return makeMat(3.0); });
    const auto D = FormExpr::coefficient("D", [=](Real, Real, Real) { return makeMat(4.0); });

    forms::Index i("i");
    forms::Index j("j");
    forms::Index k("k");
    forms::Index l("l");

    const auto expr = A(i, j) * B(j, k) * C(k, l) * D(l, i);

    LoopStructureOptions opts;
    opts.enable_symmetry_lowering = true;
    opts.enable_optimal_contraction_order = true;
    const auto p = generateLoopNest(expr, opts);
    ASSERT_TRUE(p.ok) << p.message;
    EXPECT_TRUE(p.isScalar());

    // 4 operands -> 3 pairwise contractions.
    EXPECT_EQ(p.ops.size(), 3u);

    // For n=3, optimal order costs 27 + 27 + 9 = 63 multiplications.
    EXPECT_EQ(p.estimated_flops, 63u);

    // Verify result against brute-force 4-loop expansion.
    const Mat3 a = makeMat(1.0);
    const Mat3 b = makeMat(2.0);
    const Mat3 c = makeMat(3.0);
    const Mat3 d = makeMat(4.0);
    Real ref = 0.0;
    for (int ii = 0; ii < 3; ++ii) {
        for (int jj = 0; jj < 3; ++jj) {
            for (int kk = 0; kk < 3; ++kk) {
                for (int ll = 0; ll < 3; ++ll) {
                    ref += a[static_cast<std::size_t>(ii)][static_cast<std::size_t>(jj)] *
                           b[static_cast<std::size_t>(jj)][static_cast<std::size_t>(kk)] *
                           c[static_cast<std::size_t>(kk)][static_cast<std::size_t>(ll)] *
                           d[static_cast<std::size_t>(ll)][static_cast<std::size_t>(ii)];
                }
            }
        }
    }

    const Real got = evaluateProgramScalar(p);
    EXPECT_NEAR(got, ref, 1e-12);
}

TEST(LoopStructure, HandlesTraceViaInternalRepeatedIndex)
{
    const auto A = FormExpr::coefficient("A", [=](Real, Real, Real) { return makeMat(1.0); });
    forms::Index i("i");

    const auto expr = A(i, i); // trace(A)

    const auto p = generateLoopNest(expr);
    ASSERT_TRUE(p.ok) << p.message;
    EXPECT_TRUE(p.isScalar());

    const Mat3 a = makeMat(1.0);
    const Real ref = a[0][0] + a[1][1] + a[2][2];
    const Real got = evaluateProgramScalar(p);
    EXPECT_NEAR(got, ref, 1e-12);
}

TEST(LoopStructure, PreservesAntisymmetricOutputStorageForSkew)
{
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(0.0), FormExpr::constant(1.0), FormExpr::constant(2.0)},
        {FormExpr::constant(-1.0), FormExpr::constant(0.0), FormExpr::constant(3.0)},
        {FormExpr::constant(-2.0), FormExpr::constant(-3.0), FormExpr::constant(0.0)},
    });

    forms::Index i("i");
    forms::Index j("j");
    const auto expr = A.skew()(i, j);

    const auto p = generateLoopNest(expr);
    ASSERT_TRUE(p.ok) << p.message;
    EXPECT_EQ(p.output.rank, 2);
    EXPECT_EQ(p.output.storage, TensorStorageKind::Antisymmetric2);
    EXPECT_EQ(p.output.size, 3u); // 3D skew has 3 independent components.
}

TEST(LoopStructure, EvaluatesAntisymmetricOutputInPackedForm)
{
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(0.0), FormExpr::constant(1.0), FormExpr::constant(2.0)},
        {FormExpr::constant(-1.0), FormExpr::constant(0.0), FormExpr::constant(3.0)},
        {FormExpr::constant(-2.0), FormExpr::constant(-3.0), FormExpr::constant(0.0)},
    });

    forms::Index i("i");
    forms::Index j("j");

    const auto expr = A.skew()(i, j) + A.skew()(i, j);
    const auto p = generateLoopNest(expr);
    ASSERT_TRUE(p.ok) << p.message;
    EXPECT_EQ(p.output.rank, 2);
    EXPECT_EQ(p.output.storage, TensorStorageKind::Antisymmetric2);
    EXPECT_EQ(p.output.size, 3u);
    ASSERT_EQ(p.output_loops.size(), 2u);
    EXPECT_EQ(p.output_loops[1].lower_bound_id, p.output_loops[0].id);
    EXPECT_EQ(p.output_loops[1].lower_bound_offset, 1);

    const auto out = evaluateProgramTensor(p);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_NEAR(out[0], 2.0 * 1.0, 1e-12); // (0,1)
    EXPECT_NEAR(out[1], 2.0 * 2.0, 1e-12); // (0,2)
    EXPECT_NEAR(out[2], 2.0 * 3.0, 1e-12); // (1,2)
}

TEST(LoopStructure, EvaluatesSymmetricOutputInPackedForm)
{
    const auto A = FormExpr::asTensor({
        {FormExpr::constant(1.0), FormExpr::constant(2.0), FormExpr::constant(3.0)},
        {FormExpr::constant(4.0), FormExpr::constant(5.0), FormExpr::constant(6.0)},
        {FormExpr::constant(7.0), FormExpr::constant(8.0), FormExpr::constant(9.0)},
    });

    forms::Index i("i");
    forms::Index j("j");

    const auto expr = A.sym()(i, j);
    const auto p = generateLoopNest(expr);
    ASSERT_TRUE(p.ok) << p.message;
    EXPECT_EQ(p.output.rank, 2);
    EXPECT_EQ(p.output.storage, TensorStorageKind::Symmetric2);
    EXPECT_EQ(p.output.size, 6u);
    ASSERT_EQ(p.output_loops.size(), 2u);
    EXPECT_EQ(p.output_loops[1].lower_bound_id, p.output_loops[0].id);
    EXPECT_EQ(p.output_loops[1].lower_bound_offset, 0);

    const auto out = evaluateProgramTensor(p);
    ASSERT_EQ(out.size(), 6u);
    EXPECT_NEAR(out[0], 1.0, 1e-12); // (0,0)
    EXPECT_NEAR(out[1], 3.0, 1e-12); // (0,1): 0.5*(2+4)
    EXPECT_NEAR(out[2], 5.0, 1e-12); // (0,2): 0.5*(3+7)
    EXPECT_NEAR(out[3], 5.0, 1e-12); // (1,1)
    EXPECT_NEAR(out[4], 7.0, 1e-12); // (1,2): 0.5*(6+8)
    EXPECT_NEAR(out[5], 9.0, 1e-12); // (2,2)
}

} // namespace svmp::FE::forms::tensor
