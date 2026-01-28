/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/ExternalCalls.h"

#include "Assembly/JIT/KernelArgs.h"
#include "Forms/ConstitutiveModel.h"
#include "Forms/FormExpr.h"
#include "Forms/Value.h"
#include "Math/Eigensolvers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace {

using SideArgs = assembly::jit::KernelSideArgsV4;

[[nodiscard]] const external::ExternalCallTableV1* tryGetExternalCallTable(const SideArgs* side) noexcept
{
    if (side == nullptr) {
        return nullptr;
    }

    const auto* table = static_cast<const external::ExternalCallTableV1*>(side->user_data);
    if (table == nullptr) {
        return nullptr;
    }
    if (table->magic != external::kExternalCallsMagicV1) {
        return nullptr;
    }
    if (table->abi_version != external::kExternalCallsABIVersionV1) {
        return nullptr;
    }
    return table;
}

[[nodiscard]] std::array<Real, 3> physicalPoint(const SideArgs& side, std::uint32_t q) noexcept
{
    if (side.physical_points_xyz == nullptr) {
        return {Real(0.0), Real(0.0), Real(0.0)};
    }
    const std::size_t base = static_cast<std::size_t>(q) * 3u;
    return {side.physical_points_xyz[base + 0u],
            side.physical_points_xyz[base + 1u],
            side.physical_points_xyz[base + 2u]};
}

[[nodiscard]] std::array<Real, 3> physicalPointAt(const SideArgs& side, LocalIndex qpt) noexcept
{
    if (side.physical_points_xyz == nullptr) {
        return {Real(0.0), Real(0.0), Real(0.0)};
    }
    const std::size_t base = static_cast<std::size_t>(qpt) * 3u;
    return {side.physical_points_xyz[base + 0u],
            side.physical_points_xyz[base + 1u],
            side.physical_points_xyz[base + 2u]};
}

[[nodiscard]] Real integrationWeightAt(const SideArgs& side, LocalIndex qpt) noexcept
{
    if (side.integration_weights == nullptr) {
        return Real(0.0);
    }
    return side.integration_weights[static_cast<std::size_t>(qpt)];
}

struct NonlocalV1 {
    const SideArgs* side{nullptr};
};

const std::byte* stateOldPtrAt(const SideArgs& side, LocalIndex qpt) noexcept
{
    if (side.material_state_old_base == nullptr) {
        return nullptr;
    }
    if (side.material_state_stride_bytes == 0u) {
        return nullptr;
    }
    const std::size_t byte_off =
        static_cast<std::size_t>(qpt) * static_cast<std::size_t>(side.material_state_stride_bytes);
    return side.material_state_old_base + byte_off;
}

std::byte* stateWorkPtrAt(const SideArgs& side, LocalIndex qpt) noexcept
{
    if (side.material_state_work_base == nullptr) {
        return nullptr;
    }
    if (side.material_state_stride_bytes == 0u) {
        return nullptr;
    }
    const std::size_t byte_off =
        static_cast<std::size_t>(qpt) * static_cast<std::size_t>(side.material_state_stride_bytes);
    return side.material_state_work_base + byte_off;
}

[[nodiscard]] std::span<const std::byte> stateOldSpanAt(const SideArgs& side, LocalIndex qpt) noexcept
{
    const auto* p = stateOldPtrAt(side, qpt);
    if (p == nullptr) return {};
    return {p, static_cast<std::size_t>(side.material_state_bytes_per_qpt)};
}

[[nodiscard]] std::span<std::byte> stateWorkSpanAt(const SideArgs& side, LocalIndex qpt) noexcept
{
    auto* p = stateWorkPtrAt(side, qpt);
    if (p == nullptr) return {};
    return {p, static_cast<std::size_t>(side.material_state_bytes_per_qpt)};
}

[[nodiscard]] forms::Value<Real> toFormsValue(const external::ValueViewV1& view) noexcept
{
    forms::Value<Real> v;

    const auto kind = static_cast<external::ValueKindV1>(view.kind);
    const auto* data = view.data;
    const auto n = view.length;

    switch (kind) {
        case external::ValueKindV1::Scalar:
            v.kind = forms::Value<Real>::Kind::Scalar;
            v.s = (data != nullptr && n >= 1u) ? static_cast<Real>(data[0]) : Real(0.0);
            return v;

        case external::ValueKindV1::Vector:
            v.kind = forms::Value<Real>::Kind::Vector;
            v.vector_size = static_cast<int>(std::min<std::uint32_t>(n, 3u));
            v.v = {};
            if (data != nullptr) {
                for (std::uint32_t i = 0; i < std::min<std::uint32_t>(n, 3u); ++i) {
                    v.v[static_cast<std::size_t>(i)] = static_cast<Real>(data[i]);
                }
            }
            return v;

        case external::ValueKindV1::Matrix:
            v.kind = forms::Value<Real>::Kind::Matrix;
            v.matrix_rows = 3;
            v.matrix_cols = 3;
            v.m = {};
            if (data != nullptr) {
                for (std::uint32_t i = 0; i < std::min<std::uint32_t>(n, 9u); ++i) {
                    v.m[static_cast<std::size_t>(i / 3u)][static_cast<std::size_t>(i % 3u)] = static_cast<Real>(data[i]);
                }
            }
            return v;

        case external::ValueKindV1::Tensor3:
            v.kind = forms::Value<Real>::Kind::Tensor3;
            v.tensor3_dim0 = 3;
            v.tensor3_dim1 = 3;
            v.tensor3_dim2 = 3;
            v.t3 = {};
            if (data != nullptr) {
                for (std::uint32_t i = 0; i < std::min<std::uint32_t>(n, 27u); ++i) {
                    v.t3[static_cast<std::size_t>(i)] = static_cast<Real>(data[i]);
                }
            }
            return v;

        case external::ValueKindV1::Tensor4:
            v.kind = forms::Value<Real>::Kind::Tensor4;
            v.t4 = {};
            if (data != nullptr) {
                for (std::uint32_t i = 0; i < std::min<std::uint32_t>(n, 81u); ++i) {
                    v.t4[static_cast<std::size_t>(i)] = static_cast<Real>(data[i]);
                }
            }
            return v;
    }

    v.kind = forms::Value<Real>::Kind::Scalar;
    v.s = Real(0.0);
    return v;
}

[[nodiscard]] bool isMatrixKind(forms::Value<Real>::Kind k) noexcept
{
    return k == forms::Value<Real>::Kind::Matrix ||
           k == forms::Value<Real>::Kind::SymmetricMatrix ||
           k == forms::Value<Real>::Kind::SkewMatrix;
}

void zeroOut(double* out_values, std::uint32_t out_len) noexcept
{
    if (out_values == nullptr) return;
    for (std::uint32_t i = 0; i < out_len; ++i) {
        out_values[i] = 0.0;
    }
}

void writeFormsValueToFlat(const forms::Value<Real>& v,
                           std::uint32_t out_kind_u32,
                           double* out_values,
                           std::uint32_t out_len) noexcept
{
    zeroOut(out_values, out_len);
    if (out_values == nullptr || out_len == 0u) {
        return;
    }

    const auto out_kind = static_cast<external::ValueKindV1>(out_kind_u32);

    switch (out_kind) {
        case external::ValueKindV1::Scalar:
            if (v.kind == forms::Value<Real>::Kind::Scalar) {
                out_values[0] = static_cast<double>(v.s);
            }
            return;

        case external::ValueKindV1::Vector:
            if (v.kind != forms::Value<Real>::Kind::Vector) {
                return;
            }
            for (std::uint32_t i = 0; i < std::min<std::uint32_t>(out_len, 3u); ++i) {
                out_values[i] = static_cast<double>(v.v[static_cast<std::size_t>(i)]);
            }
            return;

        case external::ValueKindV1::Matrix:
            if (!isMatrixKind(v.kind)) {
                return;
            }
            for (std::uint32_t i = 0; i < std::min<std::uint32_t>(out_len, 9u); ++i) {
                out_values[i] = static_cast<double>(v.m[static_cast<std::size_t>(i / 3u)][static_cast<std::size_t>(i % 3u)]);
            }
            return;

        case external::ValueKindV1::Tensor3:
            if (v.kind != forms::Value<Real>::Kind::Tensor3) {
                return;
            }
            for (std::uint32_t i = 0; i < std::min<std::uint32_t>(out_len, 27u); ++i) {
                out_values[i] = static_cast<double>(v.t3[static_cast<std::size_t>(i)]);
            }
            return;

        case external::ValueKindV1::Tensor4:
            if (v.kind != forms::Value<Real>::Kind::Tensor4) {
                return;
            }
            for (std::uint32_t i = 0; i < std::min<std::uint32_t>(out_len, 81u); ++i) {
                out_values[i] = static_cast<double>(v.t4[static_cast<std::size_t>(i)]);
            }
            return;
    }
}

} // namespace

extern "C" double svmp_fe_jit_coeff_eval_scalar_v1(const void* side_ptr,
                                                   std::uint32_t q,
                                                   const void* coeff_node_ptr) noexcept
{
    try {
        const auto* side = static_cast<const SideArgs*>(side_ptr);
        const auto* node = static_cast<const FormExprNode*>(coeff_node_ptr);
        if (side == nullptr || node == nullptr) {
            return 0.0;
        }

        if (const auto* table = tryGetExternalCallTable(side); table && table->eval_coefficient) {
            double out = 0.0;
            const bool ok = table->eval_coefficient(table->context,
                                                    side_ptr,
                                                    q,
                                                    coeff_node_ptr,
                                                    static_cast<std::uint32_t>(external::ValueKindV1::Scalar),
                                                    &out,
                                                    1u);
            if (ok) {
                return out;
            }
        }

        const auto x = physicalPoint(*side, q);

        if (const auto* f = node->timeScalarCoefficient(); f != nullptr) {
            return static_cast<double>((*f)(x[0], x[1], x[2], side->time));
        }
        if (const auto* f = node->scalarCoefficient(); f != nullptr) {
            return static_cast<double>((*f)(x[0], x[1], x[2]));
        }
        return 0.0;
    } catch (...) {
        return 0.0;
    }
}

extern "C" void svmp_fe_jit_coeff_eval_vector_v1(const void* side_ptr,
                                                 std::uint32_t q,
                                                 const void* coeff_node_ptr,
                                                 double* out3) noexcept
{
    if (out3 == nullptr) return;
    out3[0] = out3[1] = out3[2] = 0.0;

    try {
        const auto* side = static_cast<const SideArgs*>(side_ptr);
        const auto* node = static_cast<const FormExprNode*>(coeff_node_ptr);
        if (side == nullptr || node == nullptr) {
            return;
        }

        if (const auto* table = tryGetExternalCallTable(side); table && table->eval_coefficient) {
            const bool ok = table->eval_coefficient(table->context,
                                                    side_ptr,
                                                    q,
                                                    coeff_node_ptr,
                                                    static_cast<std::uint32_t>(external::ValueKindV1::Vector),
                                                    out3,
                                                    3u);
            if (ok) {
                return;
            }
        }

        const auto x = physicalPoint(*side, q);
        if (const auto* f = node->vectorCoefficient(); f != nullptr) {
            const auto v = (*f)(x[0], x[1], x[2]);
            out3[0] = static_cast<double>(v[0]);
            out3[1] = static_cast<double>(v[1]);
            out3[2] = static_cast<double>(v[2]);
        }
    } catch (...) {
        out3[0] = out3[1] = out3[2] = 0.0;
    }
}

extern "C" void svmp_fe_jit_coeff_eval_matrix_v1(const void* side_ptr,
                                                 std::uint32_t q,
                                                 const void* coeff_node_ptr,
                                                 double* out9) noexcept
{
    if (out9 == nullptr) return;
    for (int i = 0; i < 9; ++i) out9[i] = 0.0;

    try {
        const auto* side = static_cast<const SideArgs*>(side_ptr);
        const auto* node = static_cast<const FormExprNode*>(coeff_node_ptr);
        if (side == nullptr || node == nullptr) {
            return;
        }

        if (const auto* table = tryGetExternalCallTable(side); table && table->eval_coefficient) {
            const bool ok = table->eval_coefficient(table->context,
                                                    side_ptr,
                                                    q,
                                                    coeff_node_ptr,
                                                    static_cast<std::uint32_t>(external::ValueKindV1::Matrix),
                                                    out9,
                                                    9u);
            if (ok) {
                return;
            }
        }

        const auto x = physicalPoint(*side, q);
        if (const auto* f = node->matrixCoefficient(); f != nullptr) {
            const auto m = (*f)(x[0], x[1], x[2]);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    out9[r * 3 + c] = static_cast<double>(m[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]);
                }
            }
        }
    } catch (...) {
        for (int i = 0; i < 9; ++i) out9[i] = 0.0;
    }
}

extern "C" void svmp_fe_jit_coeff_eval_tensor3_v1(const void* side_ptr,
                                                  std::uint32_t q,
                                                  const void* coeff_node_ptr,
                                                  double* out27) noexcept
{
    if (out27 == nullptr) return;
    for (int i = 0; i < 27; ++i) out27[i] = 0.0;

    try {
        const auto* side = static_cast<const SideArgs*>(side_ptr);
        const auto* node = static_cast<const FormExprNode*>(coeff_node_ptr);
        if (side == nullptr || node == nullptr) {
            return;
        }

        if (const auto* table = tryGetExternalCallTable(side); table && table->eval_coefficient) {
            const bool ok = table->eval_coefficient(table->context,
                                                    side_ptr,
                                                    q,
                                                    coeff_node_ptr,
                                                    static_cast<std::uint32_t>(external::ValueKindV1::Tensor3),
                                                    out27,
                                                    27u);
            if (ok) {
                return;
            }
        }

        const auto x = physicalPoint(*side, q);
        if (const auto* f = node->tensor3Coefficient(); f != nullptr) {
            const auto t3 = (*f)(x[0], x[1], x[2]);
            for (int i = 0; i < 27; ++i) {
                out27[i] = static_cast<double>(t3[static_cast<std::size_t>(i)]);
            }
        }
    } catch (...) {
        for (int i = 0; i < 27; ++i) out27[i] = 0.0;
    }
}

extern "C" void svmp_fe_jit_coeff_eval_tensor4_v1(const void* side_ptr,
                                                  std::uint32_t q,
                                                  const void* coeff_node_ptr,
                                                  double* out81) noexcept
{
    if (out81 == nullptr) return;
    for (int i = 0; i < 81; ++i) out81[i] = 0.0;

    try {
        const auto* side = static_cast<const SideArgs*>(side_ptr);
        const auto* node = static_cast<const FormExprNode*>(coeff_node_ptr);
        if (side == nullptr || node == nullptr) {
            return;
        }

        if (const auto* table = tryGetExternalCallTable(side); table && table->eval_coefficient) {
            const bool ok = table->eval_coefficient(table->context,
                                                    side_ptr,
                                                    q,
                                                    coeff_node_ptr,
                                                    static_cast<std::uint32_t>(external::ValueKindV1::Tensor4),
                                                    out81,
                                                    81u);
            if (ok) {
                return;
            }
        }

        const auto x = physicalPoint(*side, q);
        if (const auto* f = node->tensor4Coefficient(); f != nullptr) {
            const auto t4 = (*f)(x[0], x[1], x[2]);
            for (int i = 0; i < 81; ++i) {
                out81[i] = static_cast<double>(t4[static_cast<std::size_t>(i)]);
            }
        }
    } catch (...) {
        for (int i = 0; i < 81; ++i) out81[i] = 0.0;
    }
}

extern "C" void svmp_fe_jit_constitutive_eval_v1(const void* side_ptr,
                                                 std::uint32_t q,
                                                 std::uint32_t trace_side,
                                                 const void* constitutive_node_ptr,
                                                 std::uint32_t output_index,
                                                 const external::ValueViewV1* inputs,
                                                 std::uint32_t num_inputs,
                                                 std::uint32_t out_kind,
                                                 double* out_values,
                                                 std::uint32_t out_len) noexcept
{
    zeroOut(out_values, out_len);

    try {
        const auto* side = static_cast<const SideArgs*>(side_ptr);
        const auto* node = static_cast<const FormExprNode*>(constitutive_node_ptr);
        if (side == nullptr || node == nullptr) {
            return;
        }

        if (const auto* table = tryGetExternalCallTable(side); table && table->eval_constitutive) {
            const bool ok = table->eval_constitutive(table->context,
                                                     side_ptr,
                                                     q,
                                                     trace_side,
                                                     constitutive_node_ptr,
                                                     output_index,
                                                     inputs,
                                                     num_inputs,
                                                     out_kind,
                                                     out_values,
                                                     out_len);
            if (ok) {
                return;
            }
        }

        const auto* model = node->constitutiveModel();
        if (model == nullptr) {
            return;
        }

        ConstitutiveEvalContext ctx;
        ctx.domain = [&]() {
            const auto ct = static_cast<assembly::ContextType>(side->context_type);
            switch (ct) {
                case assembly::ContextType::Cell:
                    return ConstitutiveEvalContext::Domain::Cell;
                case assembly::ContextType::BoundaryFace:
                    return ConstitutiveEvalContext::Domain::BoundaryFace;
                case assembly::ContextType::InteriorFace:
                default:
                    return ConstitutiveEvalContext::Domain::InteriorFace;
            }
        }();
        ctx.side = (trace_side == static_cast<std::uint32_t>(external::TraceSideV1::Plus))
                       ? ConstitutiveEvalContext::TraceSide::Plus
                       : ConstitutiveEvalContext::TraceSide::Minus;
        ctx.dim = static_cast<int>(side->dim);
        ctx.x = physicalPoint(*side, q);
        ctx.time = side->time;
        ctx.dt = side->dt;
        ctx.cell_id = side->cell_id;
        ctx.face_id = side->face_id;
        ctx.local_face_id = static_cast<LocalIndex>(side->local_face_id);
        ctx.boundary_marker = static_cast<int>(side->boundary_marker);
        ctx.q = static_cast<LocalIndex>(q);
        ctx.num_qpts = static_cast<LocalIndex>(side->n_qpts);
        ctx.user_data = side->user_data;

        NonlocalV1 nonlocal_state{side};
        ConstitutiveEvalContext::NonlocalAccess nonlocal;
        nonlocal.self = &nonlocal_state;
        nonlocal.state_old = +[](const void* self, LocalIndex qpt) -> std::span<const std::byte> {
            const auto& s = *static_cast<const NonlocalV1*>(self);
            if (s.side == nullptr) return {};
            return stateOldSpanAt(*s.side, qpt);
        };
        nonlocal.state_work = +[](const void* self, LocalIndex qpt) -> std::span<std::byte> {
            const auto& s = *static_cast<const NonlocalV1*>(self);
            if (s.side == nullptr) return {};
            return stateWorkSpanAt(*s.side, qpt);
        };
        nonlocal.physical_point = +[](const void* self, LocalIndex qpt) -> std::array<Real, 3> {
            const auto& s = *static_cast<const NonlocalV1*>(self);
            if (s.side == nullptr) return {Real(0.0), Real(0.0), Real(0.0)};
            return physicalPointAt(*s.side, qpt);
        };
        nonlocal.integration_weight = +[](const void* self, LocalIndex qpt) -> Real {
            const auto& s = *static_cast<const NonlocalV1*>(self);
            if (s.side == nullptr) return Real(0.0);
            return integrationWeightAt(*s.side, qpt);
        };
        ctx.nonlocal = &nonlocal;
        ctx.state_old = stateOldSpanAt(*side, static_cast<LocalIndex>(q));
        ctx.state_work = stateWorkSpanAt(*side, static_cast<LocalIndex>(q));

        std::vector<forms::Value<Real>> input_values;
        input_values.reserve(num_inputs);
        for (std::uint32_t i = 0; i < num_inputs; ++i) {
            input_values.push_back(toFormsValue(inputs[i]));
        }

        const auto n_outputs = model->outputCount();
        if (output_index >= n_outputs) {
            return;
        }

        std::vector<forms::Value<Real>> output_values;
        output_values.resize(n_outputs);

        model->evaluateNaryOutputs({input_values.data(), input_values.size()},
                                   ctx,
                                   {output_values.data(), output_values.size()});

        writeFormsValueToFlat(output_values[output_index], out_kind, out_values, out_len);
    } catch (...) {
        zeroOut(out_values, out_len);
    }
}

extern "C" void svmp_fe_jit_constitutive_eval_batch_v1(const void* side_ptr,
                                                       const std::uint32_t* q_indices,
                                                       std::uint32_t num_q,
                                                       std::uint32_t trace_side,
                                                       const void* constitutive_node_ptr,
                                                       std::uint32_t output_index,
                                                       const external::ValueBatchViewV1* inputs,
                                                       std::uint32_t num_inputs,
                                                       std::uint32_t out_kind,
                                                       double* out_values,
                                                       std::uint32_t out_len,
                                                       std::uint32_t out_stride) noexcept
{
    if (out_values == nullptr) {
        return;
    }

    try {
        std::vector<external::ValueViewV1> views;
        views.resize(num_inputs);

        for (std::uint32_t qi = 0; qi < num_q; ++qi) {
            const std::uint32_t q = (q_indices != nullptr) ? q_indices[qi] : qi;

            for (std::uint32_t i = 0; i < num_inputs; ++i) {
                external::ValueViewV1 v;
                v.kind = (inputs != nullptr) ? inputs[i].kind : 0u;
                v.length = (inputs != nullptr) ? inputs[i].length : 0u;
                v.data = nullptr;
                if (inputs != nullptr && inputs[i].data != nullptr) {
                    const auto stride = static_cast<std::size_t>(inputs[i].stride);
                    v.data = inputs[i].data + stride * static_cast<std::size_t>(qi);
                }
                views[i] = v;
            }

            auto* out_q = out_values + static_cast<std::size_t>(out_stride) * static_cast<std::size_t>(qi);
            svmp_fe_jit_constitutive_eval_v1(side_ptr,
                                             q,
                                             trace_side,
                                             constitutive_node_ptr,
                                             output_index,
                                             views.data(),
                                             num_inputs,
                                             out_kind,
                                             out_q,
                                             out_len);
        }
    } catch (...) {
        // Best-effort: leave outputs as-is.
    }
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

// ---------------------------------------------------------------------------
// New-physics numeric helpers (matrix functions + eigendecomposition)
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] std::array<double, 4> sym2(const double* A) noexcept
{
    if (A == nullptr) return {0.0, 0.0, 0.0, 0.0};
    const double a00 = A[0];
    const double a01 = 0.5 * (A[1] + A[2]);
    const double a11 = A[3];
    return {a00, a01, a01, a11};
}

[[nodiscard]] std::array<double, 9> sym3(const double* A) noexcept
{
    std::array<double, 9> S{};
    if (A == nullptr) return S;
    for (int r = 0; r < 3; ++r) {
        S[static_cast<std::size_t>(r * 3 + r)] = A[static_cast<std::size_t>(r * 3 + r)];
    }
    for (int r = 0; r < 3; ++r) {
        for (int c = r + 1; c < 3; ++c) {
            const double v = 0.5 * (A[static_cast<std::size_t>(r * 3 + c)] + A[static_cast<std::size_t>(c * 3 + r)]);
            S[static_cast<std::size_t>(r * 3 + c)] = v;
            S[static_cast<std::size_t>(c * 3 + r)] = v;
        }
    }
    return S;
}

template <std::size_t N>
void fillNaN(double* out) noexcept
{
    if (out == nullptr) return;
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = qnan;
    }
}

template <std::size_t N>
void fillZero(double* out) noexcept
{
    if (out == nullptr) return;
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = 0.0;
    }
}

template <std::size_t N>
void fillIdentity(double* out) noexcept
{
    fillZero<N * N>(out);
    if (out == nullptr) return;
    for (std::size_t i = 0; i < N; ++i) {
        out[i * N + i] = 1.0;
    }
}

} // namespace

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_eig_sym_2x2_v1(const double* A,
                                                                 double* eigvals2,
                                                                 double* eigvecs4) noexcept
{
    if (eigvals2 == nullptr || eigvecs4 == nullptr) {
        return;
    }

    svmp::FE::math::Matrix2x2<double> M;
    const auto S = sym2(A);
    M(0, 0) = S[0];
    M(0, 1) = S[1];
    M(1, 0) = S[2];
    M(1, 1) = S[3];

    const auto [evals, evecs] = svmp::FE::math::eigen_2x2_symmetric(M); // descending
    eigvals2[0] = static_cast<double>(evals[0]);
    eigvals2[1] = static_cast<double>(evals[1]);

    // Row-major eigenvector matrix with columns as eigenvectors.
    eigvecs4[0] = static_cast<double>(evecs(0, 0));
    eigvecs4[1] = static_cast<double>(evecs(0, 1));
    eigvecs4[2] = static_cast<double>(evecs(1, 0));
    eigvecs4[3] = static_cast<double>(evecs(1, 1));
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_eig_sym_3x3_v1(const double* A,
                                                                 double* eigvals3,
                                                                 double* eigvecs9) noexcept
{
    if (eigvals3 == nullptr || eigvecs9 == nullptr) {
        return;
    }

    svmp::FE::math::Matrix3x3<double> M;
    const auto S = sym3(A);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            M(r, c) = S[static_cast<std::size_t>(r * 3 + c)];
        }
    }

    // eigen_3x3_symmetric returns ascending eigenvalues; reorder to descending.
    const auto [evals_asc, evecs_asc] = svmp::FE::math::eigen_3x3_symmetric(M);

    eigvals3[0] = static_cast<double>(evals_asc[2]);
    eigvals3[1] = static_cast<double>(evals_asc[1]);
    eigvals3[2] = static_cast<double>(evals_asc[0]);

    // Row-major eigenvector matrix with columns as eigenvectors (descending order).
    for (int r = 0; r < 3; ++r) {
        eigvecs9[static_cast<std::size_t>(r * 3 + 0)] = static_cast<double>(evecs_asc(r, 2));
        eigvecs9[static_cast<std::size_t>(r * 3 + 1)] = static_cast<double>(evecs_asc(r, 1));
        eigvecs9[static_cast<std::size_t>(r * 3 + 2)] = static_cast<double>(evecs_asc(r, 0));
    }
}

namespace {

template <std::size_t N, class F>
void spectralMapSPD(const double* A, double* out, const F& f) noexcept
{
    if (out == nullptr) {
        return;
    }

    std::array<double, N> evals{};
    std::array<double, N * N> evecs{};

    if constexpr (N == 2) {
        double eigvals2[2]{};
        double eigvecs4[4]{};
        svmp::FE::forms::jit::svmp_fe_jit_eig_sym_2x2_v1(A, eigvals2, eigvecs4);
        evals = {eigvals2[0], eigvals2[1]};
        evecs = {eigvecs4[0], eigvecs4[1], eigvecs4[2], eigvecs4[3]};
    } else if constexpr (N == 3) {
        double eigvals3[3]{};
        double eigvecs9[9]{};
        svmp::FE::forms::jit::svmp_fe_jit_eig_sym_3x3_v1(A, eigvals3, eigvecs9);
        evals = {eigvals3[0], eigvals3[1], eigvals3[2]};
        for (std::size_t i = 0; i < 9; ++i) evecs[i] = eigvecs9[i];
    }

    for (std::size_t i = 0; i < N; ++i) {
        if (!(evals[i] > 0.0) || !std::isfinite(evals[i])) {
            fillNaN<N * N>(out);
            return;
        }
    }

    std::array<double, N> fe{};
    for (std::size_t i = 0; i < N; ++i) {
        fe[i] = f(evals[i]);
    }

    // out = Q diag(fe) Q^T (Q columns are eigenvectors), where evecs is row-major.
    fillZero<N * N>(out);
    for (std::size_t i = 0; i < N; ++i) {
        const double fi = fe[i];
        for (std::size_t r = 0; r < N; ++r) {
            const double qri = evecs[r * N + i];
            for (std::size_t c = 0; c < N; ++c) {
                const double qci = evecs[c * N + i];
                out[r * N + c] += fi * qri * qci;
            }
        }
    }
}

template <std::size_t N, class F>
void spectralMapSymmetric(const double* A, double* out, const F& f) noexcept
{
    if (out == nullptr) {
        return;
    }

    std::array<double, N> evals{};
    std::array<double, N * N> evecs{};

    if constexpr (N == 2) {
        double eigvals2[2]{};
        double eigvecs4[4]{};
        svmp::FE::forms::jit::svmp_fe_jit_eig_sym_2x2_v1(A, eigvals2, eigvecs4);
        evals = {eigvals2[0], eigvals2[1]};
        evecs = {eigvecs4[0], eigvecs4[1], eigvecs4[2], eigvecs4[3]};
    } else if constexpr (N == 3) {
        double eigvals3[3]{};
        double eigvecs9[9]{};
        svmp::FE::forms::jit::svmp_fe_jit_eig_sym_3x3_v1(A, eigvals3, eigvecs9);
        evals = {eigvals3[0], eigvals3[1], eigvals3[2]};
        for (std::size_t i = 0; i < 9; ++i) evecs[i] = eigvecs9[i];
    }

    std::array<double, N> fe{};
    for (std::size_t i = 0; i < N; ++i) {
        fe[i] = f(evals[i]);
    }

    fillZero<N * N>(out);
    for (std::size_t i = 0; i < N; ++i) {
        const double fi = fe[i];
        for (std::size_t r = 0; r < N; ++r) {
            const double qri = evecs[r * N + i];
            for (std::size_t c = 0; c < N; ++c) {
                const double qci = evecs[c * N + i];
                out[r * N + c] += fi * qri * qci;
            }
        }
    }
}

} // namespace

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_exp_2x2_v1(const double* A, double* expA4) noexcept
{
    spectralMapSymmetric<2>(A, expA4, [](double x) { return std::exp(x); });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_exp_3x3_v1(const double* A, double* expA9) noexcept
{
    spectralMapSymmetric<3>(A, expA9, [](double x) { return std::exp(x); });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_log_2x2_v1(const double* A, double* logA4) noexcept
{
    spectralMapSPD<2>(A, logA4, [](double x) { return std::log(x); });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_log_3x3_v1(const double* A, double* logA9) noexcept
{
    spectralMapSPD<3>(A, logA9, [](double x) { return std::log(x); });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_sqrt_2x2_v1(const double* A, double* sqrtA4) noexcept
{
    spectralMapSPD<2>(A, sqrtA4, [](double x) { return std::sqrt(x); });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_sqrt_3x3_v1(const double* A, double* sqrtA9) noexcept
{
    spectralMapSPD<3>(A, sqrtA9, [](double x) { return std::sqrt(x); });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_pow_2x2_v1(const double* A, double p, double* Ap4) noexcept
{
    if (!std::isfinite(p)) {
        fillNaN<4>(Ap4);
        return;
    }
    if (p == 0.0) {
        fillIdentity<2>(Ap4);
        return;
    }
    spectralMapSPD<2>(A, Ap4, [&](double x) { return std::pow(x, p); });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_pow_3x3_v1(const double* A, double p, double* Ap9) noexcept
{
    if (!std::isfinite(p)) {
        fillNaN<9>(Ap9);
        return;
    }
    if (p == 0.0) {
        fillIdentity<3>(Ap9);
        return;
    }
    spectralMapSPD<3>(A, Ap9, [&](double x) { return std::pow(x, p); });
}

namespace {

[[nodiscard]] std::int32_t clampWhich(std::int32_t which, std::int32_t n) noexcept
{
    if (n <= 1) return 0;
    if (which < 0) return 0;
    if (which >= n) return n - 1;
    return which;
}

[[nodiscard]] double safeDenom(double denom, double scale) noexcept
{
    const double eps = 1e-12 * std::max(1.0, scale);
    if (std::abs(denom) < eps) {
        return (denom < 0.0) ? -eps : eps;
    }
    return denom;
}

template <std::size_t N>
struct EigSymData {
    std::array<double, N> evals{};
    std::array<double, N * N> evecs{}; // row-major, columns are eigenvectors (descending)
};

template <std::size_t N>
[[nodiscard]] EigSymData<N> eigSym(const double* A) noexcept
{
    EigSymData<N> out{};
    if constexpr (N == 2) {
        double eigvals2[2]{};
        double eigvecs4[4]{};
        svmp::FE::forms::jit::svmp_fe_jit_eig_sym_2x2_v1(A, eigvals2, eigvecs4);
        out.evals = {eigvals2[0], eigvals2[1]};
        out.evecs = {eigvecs4[0], eigvecs4[1], eigvecs4[2], eigvecs4[3]};
    } else if constexpr (N == 3) {
        double eigvals3[3]{};
        double eigvecs9[9]{};
        svmp::FE::forms::jit::svmp_fe_jit_eig_sym_3x3_v1(A, eigvals3, eigvecs9);
        out.evals = {eigvals3[0], eigvals3[1], eigvals3[2]};
        for (std::size_t i = 0; i < 9; ++i) out.evecs[i] = eigvecs9[i];
    }
    return out;
}

template <std::size_t N, class F, class FP>
void spectralFrechetSymmetric(const double* A,
                              const double* dA,
                              double* out,
                              const F& f,
                              const FP& fp) noexcept
{
    if (out == nullptr) return;

    const auto ed = eigSym<N>(A);

    std::array<double, N * N> Sd{};
    if constexpr (N == 2) {
        const auto S = sym2(dA);
        Sd = {S[0], S[1], S[2], S[3]};
    } else if constexpr (N == 3) {
        const auto S = sym3(dA);
        for (std::size_t i = 0; i < 9; ++i) Sd[i] = S[i];
    }

    // B = Q^T dA Q (row-major, with Q columns as eigenvectors).
    std::array<double, N * N> B{};
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (std::size_t r = 0; r < N; ++r) {
                const double qri = ed.evecs[r * N + i];
                for (std::size_t c = 0; c < N; ++c) {
                    const double qcj = ed.evecs[c * N + j];
                    sum += qri * Sd[r * N + c] * qcj;
                }
            }
            B[i * N + j] = sum;
        }
    }

    std::array<double, N> fe{};
    std::array<double, N> fpe{};
    double scale = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double lam = ed.evals[i];
        scale = std::max(scale, std::abs(lam));
        fe[i] = f(lam);
        fpe[i] = fp(lam);
    }
    scale = std::max(1.0, scale);

    std::array<double, N * N> C{};
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double g = 0.0;
            if (i == j) {
                g = fpe[i];
            } else {
                const double denom_raw = ed.evals[i] - ed.evals[j];
                const double denom = safeDenom(denom_raw, scale);
                if (std::abs(denom_raw) < 1e-12 * scale) {
                    g = fp(0.5 * (ed.evals[i] + ed.evals[j]));
                } else {
                    g = (fe[i] - fe[j]) / denom;
                }
            }
            C[i * N + j] = g * B[i * N + j];
        }
    }

    // out = Q C Q^T
    fillZero<N * N>(out);
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            double sum = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                const double qri = ed.evecs[r * N + i];
                for (std::size_t j = 0; j < N; ++j) {
                    const double qcj = ed.evecs[c * N + j];
                    sum += qri * C[i * N + j] * qcj;
                }
            }
            out[r * N + c] = sum;
        }
    }
}

template <std::size_t N, class F, class FP>
void spectralFrechetSPD(const double* A,
                        const double* dA,
                        double* out,
                        const F& f,
                        const FP& fp) noexcept
{
    if (out == nullptr) return;

    const auto ed = eigSym<N>(A);
    for (std::size_t i = 0; i < N; ++i) {
        if (!(ed.evals[i] > 0.0) || !std::isfinite(ed.evals[i])) {
            fillNaN<N * N>(out);
            return;
        }
    }

    spectralFrechetSymmetric<N>(A, dA, out, f, fp);
}

} // namespace

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_exp_dd_2x2_v1(const double* A, const double* dA, double* dExpA4) noexcept
{
    spectralFrechetSymmetric<2>(A, dA, dExpA4, [](double x) { return std::exp(x); }, [](double x) { return std::exp(x); });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_exp_dd_3x3_v1(const double* A, const double* dA, double* dExpA9) noexcept
{
    spectralFrechetSymmetric<3>(A, dA, dExpA9, [](double x) { return std::exp(x); }, [](double x) { return std::exp(x); });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_log_dd_2x2_v1(const double* A, const double* dA, double* dLogA4) noexcept
{
    spectralFrechetSPD<2>(A, dA, dLogA4, [](double x) { return std::log(x); }, [](double x) { return 1.0 / x; });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_log_dd_3x3_v1(const double* A, const double* dA, double* dLogA9) noexcept
{
    spectralFrechetSPD<3>(A, dA, dLogA9, [](double x) { return std::log(x); }, [](double x) { return 1.0 / x; });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_sqrt_dd_2x2_v1(const double* A, const double* dA, double* dSqrtA4) noexcept
{
    spectralFrechetSPD<2>(A, dA, dSqrtA4, [](double x) { return std::sqrt(x); }, [](double x) { return 0.5 / std::sqrt(x); });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_sqrt_dd_3x3_v1(const double* A, const double* dA, double* dSqrtA9) noexcept
{
    spectralFrechetSPD<3>(A, dA, dSqrtA9, [](double x) { return std::sqrt(x); }, [](double x) { return 0.5 / std::sqrt(x); });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_pow_dd_2x2_v1(const double* A, const double* dA, double p, double* dAp4) noexcept
{
    if (dAp4 == nullptr) return;
    if (!std::isfinite(p)) {
        fillNaN<4>(dAp4);
        return;
    }
    if (p == 0.0) {
        fillZero<4>(dAp4);
        return;
    }
    spectralFrechetSPD<2>(A, dA, dAp4,
                          [&](double x) { return std::pow(x, p); },
                          [&](double x) { return p * std::pow(x, p - 1.0); });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_matrix_pow_dd_3x3_v1(const double* A, const double* dA, double p, double* dAp9) noexcept
{
    if (dAp9 == nullptr) return;
    if (!std::isfinite(p)) {
        fillNaN<9>(dAp9);
        return;
    }
    if (p == 0.0) {
        fillZero<9>(dAp9);
        return;
    }
    spectralFrechetSPD<3>(A, dA, dAp9,
                          [&](double x) { return std::pow(x, p); },
                          [&](double x) { return p * std::pow(x, p - 1.0); });
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_eigvec_sym_dd_2x2_v1(const double* A,
                                                                       const double* dA,
                                                                       std::int32_t which,
                                                                       double* dv2) noexcept
{
    if (dv2 == nullptr) return;
    which = clampWhich(which, 2);

    double eigvals2[2]{};
    double evecs4[4]{};
    svmp::FE::forms::jit::svmp_fe_jit_eig_sym_2x2_v1(A, eigvals2, evecs4);

    const int i = static_cast<int>(which);
    const int j = 1 - i;

    const std::array<double, 2> vi{evecs4[0 * 2 + i], evecs4[1 * 2 + i]};
    const std::array<double, 2> vj{evecs4[0 * 2 + j], evecs4[1 * 2 + j]};

    const auto Sd = sym2(dA);
    const std::array<double, 2> Sd_vi{
        Sd[0] * vi[0] + Sd[1] * vi[1],
        Sd[2] * vi[0] + Sd[3] * vi[1],
    };
    const double vjDvi = vj[0] * Sd_vi[0] + vj[1] * Sd_vi[1];

    const double denom_raw = eigvals2[static_cast<std::size_t>(i)] - eigvals2[static_cast<std::size_t>(j)];
    const double scale = std::max(std::abs(eigvals2[0]), std::abs(eigvals2[1]));
    const double denom = safeDenom(denom_raw, scale);

    const double factor = vjDvi / denom;
    dv2[0] = vj[0] * factor;
    dv2[1] = vj[1] * factor;
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_eigvec_sym_dd_3x3_v1(const double* A,
                                                                       const double* dA,
                                                                       std::int32_t which,
                                                                       double* dv3) noexcept
{
    if (dv3 == nullptr) return;
    which = clampWhich(which, 3);

    double eigvals3[3]{};
    double evecs9[9]{};
    svmp::FE::forms::jit::svmp_fe_jit_eig_sym_3x3_v1(A, eigvals3, evecs9);

    const int i = static_cast<int>(which);

    std::array<double, 3> vi{
        evecs9[0 * 3 + i],
        evecs9[1 * 3 + i],
        evecs9[2 * 3 + i],
    };

    const auto Sd = sym3(dA);
    const double scale = std::max({std::abs(eigvals3[0]), std::abs(eigvals3[1]), std::abs(eigvals3[2])});

    std::array<double, 3> dv{0.0, 0.0, 0.0};
    for (int j = 0; j < 3; ++j) {
        if (j == i) continue;

        std::array<double, 3> vj{
            evecs9[0 * 3 + j],
            evecs9[1 * 3 + j],
            evecs9[2 * 3 + j],
        };

        std::array<double, 3> Sd_vi{
            Sd[0 * 3 + 0] * vi[0] + Sd[0 * 3 + 1] * vi[1] + Sd[0 * 3 + 2] * vi[2],
            Sd[1 * 3 + 0] * vi[0] + Sd[1 * 3 + 1] * vi[1] + Sd[1 * 3 + 2] * vi[2],
            Sd[2 * 3 + 0] * vi[0] + Sd[2 * 3 + 1] * vi[1] + Sd[2 * 3 + 2] * vi[2],
        };
        const double vjDvi = vj[0] * Sd_vi[0] + vj[1] * Sd_vi[1] + vj[2] * Sd_vi[2];

        const double denom_raw = eigvals3[static_cast<std::size_t>(i)] - eigvals3[static_cast<std::size_t>(j)];
        const double denom = safeDenom(denom_raw, scale);

        const double factor = vjDvi / denom;
        for (int r = 0; r < 3; ++r) {
            dv[static_cast<std::size_t>(r)] += vj[static_cast<std::size_t>(r)] * factor;
        }
    }

    dv3[0] = dv[0];
    dv3[1] = dv[1];
    dv3[2] = dv[2];
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_spectral_decomp_dd_2x2_v1(const double* A,
                                                                            const double* dA,
                                                                            double* dQ4) noexcept
{
    if (dQ4 == nullptr) return;

    double eigvals2[2]{};
    double evecs4[4]{};
    svmp::FE::forms::jit::svmp_fe_jit_eig_sym_2x2_v1(A, eigvals2, evecs4);
    const auto Sd = sym2(dA);

    const double scale = std::max(std::abs(eigvals2[0]), std::abs(eigvals2[1]));

    for (int i = 0; i < 2; ++i) {
        const int j = 1 - i;
        const std::array<double, 2> vi{evecs4[0 * 2 + i], evecs4[1 * 2 + i]};
        const std::array<double, 2> vj{evecs4[0 * 2 + j], evecs4[1 * 2 + j]};

        const std::array<double, 2> Sd_vi{
            Sd[0] * vi[0] + Sd[1] * vi[1],
            Sd[2] * vi[0] + Sd[3] * vi[1],
        };
        const double vjDvi = vj[0] * Sd_vi[0] + vj[1] * Sd_vi[1];
        const double denom_raw = eigvals2[static_cast<std::size_t>(i)] - eigvals2[static_cast<std::size_t>(j)];
        const double denom = safeDenom(denom_raw, scale);
        const double factor = vjDvi / denom;

        // Column i of dQ is dv_i.
        dQ4[0 * 2 + i] = vj[0] * factor;
        dQ4[1 * 2 + i] = vj[1] * factor;
    }
}

extern "C" void svmp::FE::forms::jit::svmp_fe_jit_spectral_decomp_dd_3x3_v1(const double* A,
                                                                            const double* dA,
                                                                            double* dQ9) noexcept
{
    if (dQ9 == nullptr) return;

    double eigvals3[3]{};
    double evecs9[9]{};
    svmp::FE::forms::jit::svmp_fe_jit_eig_sym_3x3_v1(A, eigvals3, evecs9);
    const auto Sd = sym3(dA);

    const double scale = std::max({std::abs(eigvals3[0]), std::abs(eigvals3[1]), std::abs(eigvals3[2])});

    for (int i = 0; i < 3; ++i) {
        std::array<double, 3> vi{
            evecs9[0 * 3 + i],
            evecs9[1 * 3 + i],
            evecs9[2 * 3 + i],
        };

        std::array<double, 3> dv{0.0, 0.0, 0.0};
        for (int j = 0; j < 3; ++j) {
            if (j == i) continue;

            std::array<double, 3> vj{
                evecs9[0 * 3 + j],
                evecs9[1 * 3 + j],
                evecs9[2 * 3 + j],
            };

            std::array<double, 3> Sd_vi{
                Sd[0 * 3 + 0] * vi[0] + Sd[0 * 3 + 1] * vi[1] + Sd[0 * 3 + 2] * vi[2],
                Sd[1 * 3 + 0] * vi[0] + Sd[1 * 3 + 1] * vi[1] + Sd[1 * 3 + 2] * vi[2],
                Sd[2 * 3 + 0] * vi[0] + Sd[2 * 3 + 1] * vi[1] + Sd[2 * 3 + 2] * vi[2],
            };
            const double vjDvi = vj[0] * Sd_vi[0] + vj[1] * Sd_vi[1] + vj[2] * Sd_vi[2];

            const double denom_raw = eigvals3[static_cast<std::size_t>(i)] - eigvals3[static_cast<std::size_t>(j)];
            const double denom = safeDenom(denom_raw, scale);
            const double factor = vjDvi / denom;
            for (int r = 0; r < 3; ++r) {
                dv[static_cast<std::size_t>(r)] += vj[static_cast<std::size_t>(r)] * factor;
            }
        }

        // Column i of dQ is dv_i.
        for (int r = 0; r < 3; ++r) {
            dQ9[static_cast<std::size_t>(r * 3 + i)] = dv[static_cast<std::size_t>(r)];
        }
    }
}
