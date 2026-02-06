/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/LLVMGen.h"

#include "Core/FEException.h"
#include "Core/Logger.h"

#include "Assembly/JIT/KernelArgs.h"
#include "Forms/ConstitutiveModel.h"
#include "Forms/JIT/ExternalCalls.h"
#include "Forms/JIT/LLVMTensorGen.h"
#include "Forms/JIT/KernelIR.h"
#include "Forms/Tensor/TensorIR.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

#if SVMP_FE_ENABLE_LLVM_JIT
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#endif

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace {

#if SVMP_FE_ENABLE_LLVM_JIT
[[nodiscard]] std::string sanitizeFilename(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for (const char ch : s) {
        const bool ok =
            (ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') ||
            (ch == '_' || ch == '-' || ch == '.');
        out.push_back(ok ? ch : '_');
    }
    return out;
}

void writeTextFile(const std::filesystem::path& path, std::string_view contents) noexcept
{
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        return;
    }

    try {
        std::ofstream os(path, std::ios::trunc);
        if (!os.good()) {
            return;
        }
        os.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        os.flush();
    } catch (...) {
    }
}

[[nodiscard]] std::filesystem::path dumpPath(const JITOptions& options,
                                            std::string_view symbol,
                                            std::string_view suffix)
{
    const std::filesystem::path dir =
        options.dump_directory.empty() ? std::filesystem::path("svmp_fe_jit_dumps")
                                       : std::filesystem::path(options.dump_directory);
    return dir / (sanitizeFilename(symbol) + std::string(suffix));
}

[[nodiscard]] bool hasCpuFeature(std::string_view features, std::string_view needle) noexcept
{
    std::size_t pos = 0;
    while (pos <= features.size()) {
        const auto end = features.find(',', pos);
        std::string_view tok =
            (end == std::string_view::npos) ? features.substr(pos) : features.substr(pos, end - pos);
        while (!tok.empty() && tok.front() == ' ') {
            tok.remove_prefix(1);
        }
        while (!tok.empty() && tok.back() == ' ') {
            tok.remove_suffix(1);
        }
        if (!tok.empty() && (tok.front() == '+' || tok.front() == '-')) {
            tok.remove_prefix(1);
        }
        if (tok == needle) {
            return true;
        }
        if (end == std::string_view::npos) {
            break;
        }
        pos = end + 1;
    }
    return false;
}

[[nodiscard]] int preferredVectorWidthFromCpuFeatures(std::string_view cpu_features) noexcept
{
    if (hasCpuFeature(cpu_features, "avx512f")) {
        return 8;
    }
    if (hasCpuFeature(cpu_features, "avx2") || hasCpuFeature(cpu_features, "avx")) {
        return 4;
    }
    if (hasCpuFeature(cpu_features, "sse2") ||
        hasCpuFeature(cpu_features, "neon") ||
        hasCpuFeature(cpu_features, "sve") ||
        hasCpuFeature(cpu_features, "sve2")) {
        return 2;
    }
    return 1;
}

struct Shape {
    enum class Kind : std::uint8_t {
        Scalar,
        Vector,
        Matrix,
        Tensor3,
        Tensor4
    };

    Kind kind{Kind::Scalar};
    std::array<std::uint32_t, 4> dims{1u, 1u, 1u, 1u};

    [[nodiscard]] std::uint32_t rank() const noexcept
    {
        switch (kind) {
            case Kind::Scalar: return 0u;
            case Kind::Vector: return 1u;
            case Kind::Matrix: return 2u;
            case Kind::Tensor3: return 3u;
            case Kind::Tensor4: return 4u;
        }
        return 0u;
    }

    [[nodiscard]] bool isScalar() const noexcept { return kind == Kind::Scalar; }
};

constexpr std::uint32_t kMaxVectorDim = 3u;
constexpr std::uint32_t kMaxMatrixDim = 3u;
constexpr std::uint32_t kMaxTensor3Dim = 3u;
constexpr std::uint32_t kMaxTensor4Dim = 3u;

struct ABIV3 {
    static constexpr std::size_t cell_side_off = offsetof(assembly::jit::CellKernelArgsV6, side);
    static constexpr std::size_t cell_out_off = offsetof(assembly::jit::CellKernelArgsV6, output);

    static constexpr std::size_t bdry_side_off = offsetof(assembly::jit::BoundaryFaceKernelArgsV6, side);
    static constexpr std::size_t bdry_out_off = offsetof(assembly::jit::BoundaryFaceKernelArgsV6, output);

    static constexpr std::size_t face_minus_side_off = offsetof(assembly::jit::InteriorFaceKernelArgsV6, minus);
    static constexpr std::size_t face_plus_side_off = offsetof(assembly::jit::InteriorFaceKernelArgsV6, plus);

    static constexpr std::size_t face_out_minus_off = offsetof(assembly::jit::InteriorFaceKernelArgsV6, output_minus);
    static constexpr std::size_t face_out_plus_off = offsetof(assembly::jit::InteriorFaceKernelArgsV6, output_plus);
    static constexpr std::size_t face_coupling_minus_plus_off =
        offsetof(assembly::jit::InteriorFaceKernelArgsV6, coupling_minus_plus);
    static constexpr std::size_t face_coupling_plus_minus_off =
        offsetof(assembly::jit::InteriorFaceKernelArgsV6, coupling_plus_minus);

    static constexpr std::size_t side_dim_off = offsetof(assembly::jit::KernelSideArgsV6, dim);
    static constexpr std::size_t side_n_qpts_off = offsetof(assembly::jit::KernelSideArgsV6, n_qpts);
    static constexpr std::size_t side_n_test_dofs_off = offsetof(assembly::jit::KernelSideArgsV6, n_test_dofs);
    static constexpr std::size_t side_n_trial_dofs_off = offsetof(assembly::jit::KernelSideArgsV6, n_trial_dofs);

    static constexpr std::size_t side_test_field_type_off = offsetof(assembly::jit::KernelSideArgsV6, test_field_type);
    static constexpr std::size_t side_trial_field_type_off = offsetof(assembly::jit::KernelSideArgsV6, trial_field_type);
    static constexpr std::size_t side_test_value_dim_off = offsetof(assembly::jit::KernelSideArgsV6, test_value_dim);
    static constexpr std::size_t side_trial_value_dim_off = offsetof(assembly::jit::KernelSideArgsV6, trial_value_dim);
    static constexpr std::size_t side_test_uses_vector_basis_off =
        offsetof(assembly::jit::KernelSideArgsV6, test_uses_vector_basis);
    static constexpr std::size_t side_trial_uses_vector_basis_off =
        offsetof(assembly::jit::KernelSideArgsV6, trial_uses_vector_basis);

    static constexpr std::size_t side_integration_weights_off =
        offsetof(assembly::jit::KernelSideArgsV6, integration_weights);
    static constexpr std::size_t side_quad_points_xyz_off = offsetof(assembly::jit::KernelSideArgsV6, quad_points_xyz);
    static constexpr std::size_t side_physical_points_xyz_off =
        offsetof(assembly::jit::KernelSideArgsV6, physical_points_xyz);

    static constexpr std::size_t side_jacobians_off = offsetof(assembly::jit::KernelSideArgsV6, jacobians);
    static constexpr std::size_t side_inverse_jacobians_off = offsetof(assembly::jit::KernelSideArgsV6, inverse_jacobians);
    static constexpr std::size_t side_jacobian_dets_off = offsetof(assembly::jit::KernelSideArgsV6, jacobian_dets);
    static constexpr std::size_t side_normals_xyz_off = offsetof(assembly::jit::KernelSideArgsV6, normals_xyz);
    static constexpr std::size_t side_interleaved_geom_off =
        offsetof(assembly::jit::KernelSideArgsV6, interleaved_qpoint_geometry);
    static constexpr std::size_t side_interleaved_geom_stride_off =
        offsetof(assembly::jit::KernelSideArgsV6, interleaved_qpoint_geometry_stride_reals);
    static constexpr std::size_t side_interleaved_geom_phys_off =
        offsetof(assembly::jit::KernelSideArgsV6, interleaved_qpoint_geometry_physical_offset);
    static constexpr std::size_t side_interleaved_geom_jac_off =
        offsetof(assembly::jit::KernelSideArgsV6, interleaved_qpoint_geometry_jacobian_offset);
    static constexpr std::size_t side_interleaved_geom_jinv_off =
        offsetof(assembly::jit::KernelSideArgsV6, interleaved_qpoint_geometry_inverse_jacobian_offset);
    static constexpr std::size_t side_interleaved_geom_det_off =
        offsetof(assembly::jit::KernelSideArgsV6, interleaved_qpoint_geometry_det_offset);
    static constexpr std::size_t side_interleaved_geom_normal_off =
        offsetof(assembly::jit::KernelSideArgsV6, interleaved_qpoint_geometry_normal_offset);

    static constexpr std::size_t side_test_basis_values_off = offsetof(assembly::jit::KernelSideArgsV6, test_basis_values);
    static constexpr std::size_t side_trial_basis_values_off =
        offsetof(assembly::jit::KernelSideArgsV6, trial_basis_values);
    static constexpr std::size_t side_test_phys_grads_off =
        offsetof(assembly::jit::KernelSideArgsV6, test_phys_gradients_xyz);
    static constexpr std::size_t side_trial_phys_grads_off =
        offsetof(assembly::jit::KernelSideArgsV6, trial_phys_gradients_xyz);

    static constexpr std::size_t side_test_phys_hess_off = offsetof(assembly::jit::KernelSideArgsV6, test_phys_hessians);
    static constexpr std::size_t side_trial_phys_hess_off =
        offsetof(assembly::jit::KernelSideArgsV6, trial_phys_hessians);

    static constexpr std::size_t side_test_vector_basis_values_xyz_off =
        offsetof(assembly::jit::KernelSideArgsV6, test_basis_vector_values_xyz);
    static constexpr std::size_t side_test_vector_basis_curls_xyz_off =
        offsetof(assembly::jit::KernelSideArgsV6, test_basis_curls_xyz);
    static constexpr std::size_t side_test_vector_basis_divs_off =
        offsetof(assembly::jit::KernelSideArgsV6, test_basis_divergences);

    static constexpr std::size_t side_trial_vector_basis_values_xyz_off =
        offsetof(assembly::jit::KernelSideArgsV6, trial_basis_vector_values_xyz);
    static constexpr std::size_t side_trial_vector_basis_curls_xyz_off =
        offsetof(assembly::jit::KernelSideArgsV6, trial_basis_curls_xyz);
    static constexpr std::size_t side_trial_vector_basis_divs_off =
        offsetof(assembly::jit::KernelSideArgsV6, trial_basis_divergences);

    static constexpr std::size_t side_solution_coefficients_off =
        offsetof(assembly::jit::KernelSideArgsV6, solution_coefficients);
    static constexpr std::size_t side_num_previous_solutions_off =
        offsetof(assembly::jit::KernelSideArgsV6, num_previous_solutions);
    static constexpr std::size_t side_previous_solution_coefficients_off =
        offsetof(assembly::jit::KernelSideArgsV6, previous_solution_coefficients);

    static constexpr std::size_t side_num_history_steps_off =
        offsetof(assembly::jit::KernelSideArgsV6, num_history_steps);
    static constexpr std::size_t side_history_weights_off =
        offsetof(assembly::jit::KernelSideArgsV6, history_weights);
    static constexpr std::size_t side_history_solution_coefficients_off =
        offsetof(assembly::jit::KernelSideArgsV6, history_solution_coefficients);

    static constexpr std::size_t side_field_solutions_off = offsetof(assembly::jit::KernelSideArgsV6, field_solutions);
    static constexpr std::size_t side_num_field_solutions_off =
        offsetof(assembly::jit::KernelSideArgsV6, num_field_solutions);

    static constexpr std::size_t side_jit_constants_off = offsetof(assembly::jit::KernelSideArgsV6, jit_constants);
    static constexpr std::size_t side_coupled_integrals_off = offsetof(assembly::jit::KernelSideArgsV6, coupled_integrals);
    static constexpr std::size_t side_coupled_aux_off = offsetof(assembly::jit::KernelSideArgsV6, coupled_aux);

    static constexpr std::size_t side_time_off = offsetof(assembly::jit::KernelSideArgsV6, time);
    static constexpr std::size_t side_dt_off = offsetof(assembly::jit::KernelSideArgsV6, dt);
    static constexpr std::size_t side_cell_domain_id_off = offsetof(assembly::jit::KernelSideArgsV6, cell_domain_id);
    static constexpr std::size_t side_cell_diameter_off = offsetof(assembly::jit::KernelSideArgsV6, cell_diameter);
    static constexpr std::size_t side_cell_volume_off = offsetof(assembly::jit::KernelSideArgsV6, cell_volume);
    static constexpr std::size_t side_facet_area_off = offsetof(assembly::jit::KernelSideArgsV6, facet_area);

    static constexpr std::size_t side_time_derivative_term_weight_off =
        offsetof(assembly::jit::KernelSideArgsV6, time_derivative_term_weight);
    static constexpr std::size_t side_non_time_derivative_term_weight_off =
        offsetof(assembly::jit::KernelSideArgsV6, non_time_derivative_term_weight);
    static constexpr std::size_t side_dt_stencil_coeffs_off =
        offsetof(assembly::jit::KernelSideArgsV6, dt_stencil_coeffs);
    static constexpr std::size_t side_dt_term_weights_off =
        offsetof(assembly::jit::KernelSideArgsV6, dt_term_weights);
    static constexpr std::size_t side_max_time_derivative_order_off =
        offsetof(assembly::jit::KernelSideArgsV6, max_time_derivative_order);

    static constexpr std::size_t side_material_state_old_base_off =
        offsetof(assembly::jit::KernelSideArgsV6, material_state_old_base);
    static constexpr std::size_t side_material_state_work_base_off =
        offsetof(assembly::jit::KernelSideArgsV6, material_state_work_base);
    static constexpr std::size_t side_material_state_stride_bytes_off =
        offsetof(assembly::jit::KernelSideArgsV6, material_state_stride_bytes);

    static constexpr std::size_t out_element_matrix_off = offsetof(assembly::jit::KernelOutputViewV6, element_matrix);
    static constexpr std::size_t out_element_vector_off = offsetof(assembly::jit::KernelOutputViewV6, element_vector);
    static constexpr std::size_t out_n_test_dofs_off = offsetof(assembly::jit::KernelOutputViewV6, n_test_dofs);
    static constexpr std::size_t out_n_trial_dofs_off = offsetof(assembly::jit::KernelOutputViewV6, n_trial_dofs);

    static constexpr std::size_t field_entry_field_id_off = offsetof(assembly::jit::FieldSolutionEntryV1, field_id);
    static constexpr std::size_t field_entry_field_type_off = offsetof(assembly::jit::FieldSolutionEntryV1, field_type);
    static constexpr std::size_t field_entry_value_dim_off = offsetof(assembly::jit::FieldSolutionEntryV1, value_dim);

    static constexpr std::size_t field_entry_values_off = offsetof(assembly::jit::FieldSolutionEntryV1, values);
    static constexpr std::size_t field_entry_gradients_xyz_off = offsetof(assembly::jit::FieldSolutionEntryV1, gradients_xyz);
    static constexpr std::size_t field_entry_hessians_off = offsetof(assembly::jit::FieldSolutionEntryV1, hessians);
    static constexpr std::size_t field_entry_laplacians_off = offsetof(assembly::jit::FieldSolutionEntryV1, laplacians);

    static constexpr std::size_t field_entry_vector_values_xyz_off =
        offsetof(assembly::jit::FieldSolutionEntryV1, vector_values_xyz);
    static constexpr std::size_t field_entry_jacobians_off = offsetof(assembly::jit::FieldSolutionEntryV1, jacobians);
    static constexpr std::size_t field_entry_component_hessians_off =
        offsetof(assembly::jit::FieldSolutionEntryV1, component_hessians);
    static constexpr std::size_t field_entry_component_laplacians_off =
        offsetof(assembly::jit::FieldSolutionEntryV1, component_laplacians);

    static constexpr std::size_t field_entry_history_count_off = offsetof(assembly::jit::FieldSolutionEntryV1, history_count);
    static constexpr std::size_t field_entry_history_values_off = offsetof(assembly::jit::FieldSolutionEntryV1, history_values);
    static constexpr std::size_t field_entry_history_vector_values_xyz_off =
        offsetof(assembly::jit::FieldSolutionEntryV1, history_vector_values_xyz);
};

[[nodiscard]] Shape scalarShape() noexcept { return Shape{.kind = Shape::Kind::Scalar, .dims = {1u, 1u, 1u, 1u}}; }

[[nodiscard]] Shape vectorShape(std::uint32_t n) noexcept
{
    Shape s;
    s.kind = Shape::Kind::Vector;
    s.dims = {n, 1u, 1u, 1u};
    return s;
}

[[nodiscard]] std::uint32_t unpackValueDim(std::uint64_t imm0) noexcept
{
    return static_cast<std::uint32_t>((imm0 >> 24) & 0xffffULL);
}

[[nodiscard]] Shape matrixShape(std::uint32_t rows, std::uint32_t cols) noexcept
{
    Shape s;
    s.kind = Shape::Kind::Matrix;
    s.dims = {rows, cols, 1u, 1u};
    return s;
}

[[nodiscard]] Shape tensor3Shape(std::uint32_t d0, std::uint32_t d1, std::uint32_t d2) noexcept
{
    Shape s;
    s.kind = Shape::Kind::Tensor3;
    s.dims = {d0, d1, d2, 1u};
    return s;
}

[[nodiscard]] Shape tensor4Shape(std::uint32_t d0, std::uint32_t d1, std::uint32_t d2, std::uint32_t d3) noexcept
{
    Shape s;
    s.kind = Shape::Kind::Tensor4;
    s.dims = {d0, d1, d2, d3};
    return s;
}

[[nodiscard]] std::optional<Shape> shapeFromValueKind(forms::Value<Real>::Kind kind) noexcept
{
    switch (kind) {
        case forms::Value<Real>::Kind::Scalar:
            return scalarShape();
        case forms::Value<Real>::Kind::Vector:
            return vectorShape(3u);
        case forms::Value<Real>::Kind::Matrix:
        case forms::Value<Real>::Kind::SymmetricMatrix:
        case forms::Value<Real>::Kind::SkewMatrix:
            return matrixShape(3u, 3u);
        case forms::Value<Real>::Kind::Tensor3:
            return tensor3Shape(3u, 3u, 3u);
        case forms::Value<Real>::Kind::Tensor4:
            return tensor4Shape(3u, 3u, 3u, 3u);
    }
    return std::nullopt;
}

[[nodiscard]] std::uint32_t unpackSpaceFieldType(std::uint64_t imm0) noexcept
{
    return static_cast<std::uint32_t>((imm0 >> 8) & 0xffULL);
}

[[nodiscard]] std::uint32_t unpackSpaceValueDim(std::uint64_t imm0) noexcept
{
    return unpackValueDim(imm0);
}

[[nodiscard]] std::uint32_t unpackU32Lo(std::uint64_t x) noexcept
{
    return static_cast<std::uint32_t>(x & 0xffffffffULL);
}

[[nodiscard]] std::uint32_t unpackU32Hi(std::uint64_t x) noexcept
{
    return static_cast<std::uint32_t>((x >> 32) & 0xffffffffULL);
}

[[nodiscard]] std::uint8_t unpackIndexedRank(std::uint64_t imm1) noexcept
{
    return static_cast<std::uint8_t>((imm1 >> 32) & 0xffULL);
}

[[nodiscard]] std::uint16_t unpackIndexedId(std::uint64_t imm0, int k) noexcept
{
    return static_cast<std::uint16_t>((imm0 >> (16u * static_cast<std::uint32_t>(k))) & 0xffffULL);
}

[[nodiscard]] std::uint8_t unpackIndexedExtent(std::uint64_t imm1, int k) noexcept
{
    return static_cast<std::uint8_t>((imm1 >> (8u * static_cast<std::uint32_t>(k))) & 0xffULL);
}

struct ShapeInferenceResult {
    bool ok{false};
    std::string message{};
    std::vector<Shape> shapes{};
};

[[nodiscard]] ShapeInferenceResult inferShapes(const KernelIR& ir,
                                               const std::optional<FormExprNode::SpaceSignature>& test_sig,
                                               const std::optional<FormExprNode::SpaceSignature>& trial_sig,
                                               bool require_scalar_root = true)
{
    ShapeInferenceResult out;
    out.ok = false;
    out.shapes.resize(ir.ops.size(), scalarShape());

    const auto dim_from_sig = [](const std::optional<FormExprNode::SpaceSignature>& sig) -> std::optional<std::uint32_t> {
        if (!sig.has_value()) {
            return std::nullopt;
        }
        const int td = sig->topological_dimension;
        if (td <= 0) {
            return std::nullopt;
        }
        return static_cast<std::uint32_t>(td);
    };

    // Spatial dimension used for geometric quantities and derivatives.
    //
    // The kernel ABI always stores xyz/jacobians in size-3/3x3 buffers, but the
    // active dimension is provided by the function-space signature.
    std::uint32_t dim = 3u;
    if (const auto d = dim_from_sig(test_sig)) {
        dim = *d;
    } else if (const auto d = dim_from_sig(trial_sig)) {
        dim = *d;
    }
    dim = std::max<std::uint32_t>(1u, std::min<std::uint32_t>(dim, kMaxVectorDim));

    auto fail = [&](std::string msg) -> ShapeInferenceResult {
        out.ok = false;
        out.message = std::move(msg);
        return out;
    };

    for (std::size_t idx = 0; idx < ir.ops.size(); ++idx) {
        const auto& op = ir.ops[idx];
        auto childAt = [&](std::size_t i) -> const Shape& {
            const auto c = ir.children[static_cast<std::size_t>(op.first_child) + i];
            return out.shapes[c];
        };

        switch (op.type) {
            case FormExprType::Constant:
            case FormExprType::ParameterRef:
            case FormExprType::BoundaryIntegralRef:
            case FormExprType::AuxiliaryStateRef:
            case FormExprType::MaterialStateOldRef:
            case FormExprType::MaterialStateWorkRef:
            case FormExprType::Time:
            case FormExprType::TimeStep:
            case FormExprType::EffectiveTimeStep:
            case FormExprType::CellDiameter:
            case FormExprType::CellVolume:
            case FormExprType::FacetArea:
            case FormExprType::CellDomainId:
            case FormExprType::JacobianDeterminant:
                out.shapes[idx] = scalarShape();
                break;

            case FormExprType::PreviousSolutionRef: {
                if (trial_sig && trial_sig->field_type == FieldType::Vector) {
                    const auto vd = static_cast<std::uint32_t>(std::max(1, trial_sig->value_dimension));
                    out.shapes[idx] = vectorShape(vd);
                } else {
                    out.shapes[idx] = scalarShape();
                }
                break;
            }

	            case FormExprType::Coefficient: {
	                const auto* node =
	                    reinterpret_cast<const FormExprNode*>(static_cast<std::uintptr_t>(op.imm0));
	                if (node == nullptr) {
                    return fail("LLVMGen: Coefficient node missing pointer identity (imm0)");
                }

                if (node->scalarCoefficient() != nullptr || node->timeScalarCoefficient() != nullptr) {
                    out.shapes[idx] = scalarShape();
                    break;
                }
                if (node->vectorCoefficient() != nullptr) {
                    out.shapes[idx] = vectorShape(dim);
                    break;
                }
                if (node->matrixCoefficient() != nullptr) {
                    out.shapes[idx] = matrixShape(dim, dim);
                    break;
                }
                if (node->tensor3Coefficient() != nullptr) {
                    out.shapes[idx] = tensor3Shape(dim, dim, dim);
                    break;
                }
                if (node->tensor4Coefficient() != nullptr) {
                    out.shapes[idx] = tensor4Shape(dim, dim, dim, dim);
                    break;
                }

                return fail("LLVMGen: Coefficient node has no callable");
            }

            case FormExprType::Constitutive: {
                const auto* node =
                    reinterpret_cast<const FormExprNode*>(static_cast<std::uintptr_t>(op.imm0));
                if (node == nullptr) {
                    return fail("LLVMGen: Constitutive node missing pointer identity (imm0)");
                }
                const auto* model = node->constitutiveModel();
                if (model == nullptr) {
                    return fail("LLVMGen: Constitutive node missing model");
                }

                try {
                    const auto spec = model->outputSpec(0u);
                    if (!spec.kind.has_value()) {
                        return fail("LLVMGen: Constitutive model output kind not specified (override outputSpec or inline model)");
                    }
                    const auto sh = shapeFromValueKind(*spec.kind);
                    if (!sh.has_value()) {
                        return fail("LLVMGen: Constitutive model output kind not supported in JIT");
                    }
                    out.shapes[idx] = *sh;
                } catch (const std::exception& e) {
                    return fail(std::string("LLVMGen: Constitutive outputSpec failed: ") + e.what());
                }
                break;
            }

            case FormExprType::ConstitutiveOutput: {
                if (op.child_count != 1u) {
                    return fail("LLVMGen: ConstitutiveOutput expects exactly 1 child");
                }
                const auto child_idx =
                    ir.children[static_cast<std::size_t>(op.first_child)];
                if (child_idx >= ir.ops.size()) {
                    return fail("LLVMGen: ConstitutiveOutput has invalid child index");
                }
                const auto& call_op = ir.ops[child_idx];
                if (call_op.type != FormExprType::Constitutive) {
                    return fail("LLVMGen: ConstitutiveOutput child must be a Constitutive call");
                }

                const auto* node =
                    reinterpret_cast<const FormExprNode*>(static_cast<std::uintptr_t>(call_op.imm0));
                if (node == nullptr) {
                    return fail("LLVMGen: ConstitutiveOutput call missing pointer identity");
                }
                const auto* model = node->constitutiveModel();
                if (model == nullptr) {
                    return fail("LLVMGen: ConstitutiveOutput call missing model");
                }

                const auto out_idx_i64 = static_cast<std::int64_t>(op.imm0);
                if (out_idx_i64 < 0) {
                    return fail("LLVMGen: ConstitutiveOutput has negative output index");
                }
                const auto out_idx = static_cast<std::size_t>(out_idx_i64);

                try {
                    const auto spec = model->outputSpec(out_idx);
                    if (!spec.kind.has_value()) {
                        return fail("LLVMGen: ConstitutiveOutput kind not specified (override outputSpec or inline model)");
                    }
                    const auto sh = shapeFromValueKind(*spec.kind);
                    if (!sh.has_value()) {
                        return fail("LLVMGen: ConstitutiveOutput kind not supported in JIT");
                    }
                    out.shapes[idx] = *sh;
                } catch (const std::exception& e) {
                    return fail(std::string("LLVMGen: ConstitutiveOutput outputSpec failed: ") + e.what());
                }
                break;
            }

            case FormExprType::Coordinate:
            case FormExprType::ReferenceCoordinate:
            case FormExprType::Normal:
                out.shapes[idx] = vectorShape(dim);
                break;

            case FormExprType::Identity: {
                const auto idim = static_cast<std::int64_t>(op.imm0);
                const auto n = (idim > 0) ? static_cast<std::uint32_t>(idim) : dim;
                out.shapes[idx] = matrixShape(n, n);
                break;
            }

            case FormExprType::Jacobian:
            case FormExprType::JacobianInverse:
                out.shapes[idx] = matrixShape(dim, dim);
                break;

            case FormExprType::TestFunction:
            case FormExprType::TrialFunction:
            case FormExprType::DiscreteField:
            case FormExprType::StateField: {
                const auto ft = unpackSpaceFieldType(op.imm0);
                const auto vd = unpackSpaceValueDim(op.imm0);
                if (ft == static_cast<std::uint32_t>(FieldType::Vector)) {
                    out.shapes[idx] = vectorShape(std::max<std::uint32_t>(1u, vd));
                } else {
                    out.shapes[idx] = scalarShape();
                }
                break;
            }

            case FormExprType::Gradient: {
                const auto& a = childAt(0);
                if (a.isScalar()) {
                    out.shapes[idx] = vectorShape(dim);
                } else if (a.kind == Shape::Kind::Vector) {
                    out.shapes[idx] = matrixShape(a.dims[0], dim);
                } else {
                    return fail("LLVMGen: Gradient expects scalar or vector operand");
                }
                break;
            }

            case FormExprType::Divergence: {
                const auto& a = childAt(0);
                if (a.kind == Shape::Kind::Vector) {
                    out.shapes[idx] = scalarShape();
                    break;
                }
                if (a.kind == Shape::Kind::Matrix) {
                    out.shapes[idx] = vectorShape(a.dims[0]);
                    break;
                }
                return fail("LLVMGen: Divergence expects vector or matrix operand");
            }

            case FormExprType::Curl: {
                const auto& a = childAt(0);
                if (a.kind != Shape::Kind::Vector) {
                    return fail("LLVMGen: Curl expects vector operand");
                }
                out.shapes[idx] = vectorShape(3u);
                break;
            }

            case FormExprType::Hessian: {
                const auto& a = childAt(0);
                if (!a.isScalar()) {
                    return fail("LLVMGen: Hessian expects scalar operand");
                }
                out.shapes[idx] = matrixShape(dim, dim);
                break;
            }

            case FormExprType::TimeDerivative:
            case FormExprType::RestrictMinus:
            case FormExprType::RestrictPlus:
                out.shapes[idx] = childAt(0);
                break;

            case FormExprType::Jump:
            case FormExprType::Average:
                out.shapes[idx] = childAt(0);
                break;

            case FormExprType::Negate:
                out.shapes[idx] = childAt(0);
                break;

            case FormExprType::Add:
            case FormExprType::Subtract: {
                const auto& a = childAt(0);
                const auto& b = childAt(1);
                if (a.kind != b.kind || a.dims != b.dims) {
                    return fail("LLVMGen: Add/Subtract requires matching shapes");
                }
                out.shapes[idx] = a;
                break;
            }

            case FormExprType::Multiply: {
                const auto& a = childAt(0);
                const auto& b = childAt(1);
                if (a.isScalar() && b.isScalar()) {
                    out.shapes[idx] = scalarShape();
                    break;
                }
                if (a.isScalar()) {
                    out.shapes[idx] = b;
                    break;
                }
                if (b.isScalar()) {
                    out.shapes[idx] = a;
                    break;
                }
                if (a.kind == Shape::Kind::Matrix && b.kind == Shape::Kind::Vector && a.dims[1] == b.dims[0]) {
                    out.shapes[idx] = vectorShape(a.dims[0]);
                    break;
                }
                if (a.kind == Shape::Kind::Vector && b.kind == Shape::Kind::Matrix && a.dims[0] == b.dims[0]) {
                    out.shapes[idx] = vectorShape(b.dims[1]);
                    break;
                }
                if (a.kind == Shape::Kind::Matrix && b.kind == Shape::Kind::Matrix && a.dims[1] == b.dims[0]) {
                    out.shapes[idx] = matrixShape(a.dims[0], b.dims[1]);
                    break;
                }
                return fail("LLVMGen: Multiply operand shapes not supported");
            }

            case FormExprType::Divide: {
                const auto& a = childAt(0);
                const auto& b = childAt(1);
                if (!b.isScalar()) {
                    return fail("LLVMGen: Divide requires scalar denominator");
                }
                out.shapes[idx] = a;
                break;
            }

            case FormExprType::InnerProduct: {
                const auto& a = childAt(0);
                const auto& b = childAt(1);
                if (a.kind != b.kind || a.dims != b.dims) {
                    return fail("LLVMGen: InnerProduct requires matching operand shapes");
                }
                out.shapes[idx] = scalarShape();
                break;
            }

            case FormExprType::DoubleContraction: {
                const auto& a = childAt(0);
                const auto& b = childAt(1);
                if (a.kind == Shape::Kind::Tensor4 && b.kind == Shape::Kind::Matrix) {
                    out.shapes[idx] = matrixShape(3u, 3u);
                    break;
                }
                if (a.kind == Shape::Kind::Matrix && b.kind == Shape::Kind::Tensor4) {
                    out.shapes[idx] = matrixShape(3u, 3u);
                    break;
                }
                if (a.kind != b.kind || a.dims != b.dims) {
                    return fail("LLVMGen: DoubleContraction requires matching operand shapes");
                }
                out.shapes[idx] = scalarShape();
                break;
            }

            case FormExprType::OuterProduct: {
                const auto& a = childAt(0);
                const auto& b = childAt(1);
                if (a.kind != Shape::Kind::Vector || b.kind != Shape::Kind::Vector) {
                    return fail("LLVMGen: OuterProduct expects vector-vector");
                }
                out.shapes[idx] = matrixShape(a.dims[0], b.dims[0]);
                break;
            }

            case FormExprType::CrossProduct: {
                const auto& a = childAt(0);
                const auto& b = childAt(1);
                if (a.kind != Shape::Kind::Vector || b.kind != Shape::Kind::Vector) {
                    return fail("LLVMGen: CrossProduct expects vector-vector");
                }
                out.shapes[idx] = vectorShape(3u);
                break;
            }

            case FormExprType::Power:
            case FormExprType::Minimum:
            case FormExprType::Maximum:
            case FormExprType::Less:
            case FormExprType::LessEqual:
            case FormExprType::Greater:
            case FormExprType::GreaterEqual:
            case FormExprType::Equal:
            case FormExprType::NotEqual: {
                if (!childAt(0).isScalar() || !childAt(1).isScalar()) {
                    return fail("LLVMGen: scalar op expects scalar operands");
                }
                out.shapes[idx] = scalarShape();
                break;
            }

            case FormExprType::Conditional: {
                if (!childAt(0).isScalar()) {
                    return fail("LLVMGen: Conditional condition must be scalar");
                }
                const auto& a = childAt(1);
                const auto& b = childAt(2);
                if (a.kind != b.kind || a.dims != b.dims) {
                    return fail("LLVMGen: Conditional branches must match");
                }
                out.shapes[idx] = a;
                break;
            }

            case FormExprType::AsVector: {
                if (op.child_count == 0u) {
                    return fail("LLVMGen: AsVector expects at least 1 component");
                }
                for (std::size_t k = 0; k < op.child_count; ++k) {
                    if (!childAt(k).isScalar()) {
                        return fail("LLVMGen: AsVector components must be scalar");
                    }
                }
                out.shapes[idx] = vectorShape(op.child_count);
                break;
            }

            case FormExprType::AsTensor: {
                const auto rows = unpackU32Lo(op.imm0);
                const auto cols = unpackU32Hi(op.imm0);
                if (rows == 0u || cols == 0u) {
                    return fail("LLVMGen: AsTensor requires non-zero rows/cols");
                }
                if (op.child_count != static_cast<std::uint32_t>(rows * cols)) {
                    return fail("LLVMGen: AsTensor child count mismatch");
                }
                for (std::size_t k = 0; k < op.child_count; ++k) {
                    if (!childAt(k).isScalar()) {
                        return fail("LLVMGen: AsTensor entries must be scalar");
                    }
                }
                out.shapes[idx] = matrixShape(rows, cols);
                break;
            }

            case FormExprType::Component:
                out.shapes[idx] = scalarShape();
                break;

            case FormExprType::Transpose: {
                const auto& a = childAt(0);
                if (a.kind != Shape::Kind::Matrix) {
                    return fail("LLVMGen: Transpose expects matrix");
                }
                out.shapes[idx] = matrixShape(a.dims[1], a.dims[0]);
                break;
            }

            case FormExprType::Trace:
            case FormExprType::Determinant: {
                const auto& a = childAt(0);
                if (a.kind != Shape::Kind::Matrix || a.dims[0] != a.dims[1] || a.dims[0] == 0u) {
                    return fail("LLVMGen: Trace/Determinant expects square matrix");
                }
                out.shapes[idx] = scalarShape();
                break;
            }

            case FormExprType::Inverse:
            case FormExprType::Cofactor:
            case FormExprType::Deviator:
            case FormExprType::SymmetricPart:
            case FormExprType::SkewPart: {
                const auto& a = childAt(0);
                if (a.kind != Shape::Kind::Matrix || a.dims[0] != a.dims[1]) {
                    return fail("LLVMGen: matrix op expects square matrix");
                }
                out.shapes[idx] = a;
                break;
            }

            case FormExprType::Norm:
                out.shapes[idx] = scalarShape();
                break;

            case FormExprType::Normalize: {
                const auto& a = childAt(0);
                if (a.kind != Shape::Kind::Vector) {
                    return fail("LLVMGen: Normalize expects vector");
                }
                out.shapes[idx] = a;
                break;
            }

            case FormExprType::AbsoluteValue:
            case FormExprType::Sign:
            case FormExprType::Sqrt:
            case FormExprType::Exp:
            case FormExprType::Log: {
                const auto& a = childAt(0);
                if (!a.isScalar()) {
                    return fail("LLVMGen: scalar function expects scalar operand");
                }
                out.shapes[idx] = scalarShape();
                break;
            }

            case FormExprType::MatrixExponential:
            case FormExprType::MatrixLogarithm:
            case FormExprType::MatrixSqrt: {
                const auto& a = childAt(0);
                if (a.kind != Shape::Kind::Matrix || a.dims[0] != a.dims[1] || (a.dims[0] != 2u && a.dims[0] != 3u)) {
                    return fail("LLVMGen: matrix function expects a 2x2 or 3x3 square matrix");
                }
                out.shapes[idx] = a;
                break;
            }

            case FormExprType::MatrixPower: {
                const auto& a = childAt(0);
                const auto& p = childAt(1);
                if (a.kind != Shape::Kind::Matrix || a.dims[0] != a.dims[1] || (a.dims[0] != 2u && a.dims[0] != 3u)) {
                    return fail("LLVMGen: powm expects a 2x2 or 3x3 square matrix");
                }
                if (!p.isScalar()) {
                    return fail("LLVMGen: powm expects a scalar exponent");
                }
                out.shapes[idx] = a;
                break;
            }

            case FormExprType::MatrixExponentialDirectionalDerivative:
            case FormExprType::MatrixLogarithmDirectionalDerivative:
            case FormExprType::MatrixSqrtDirectionalDerivative: {
                const auto& a = childAt(0);
                const auto& da = childAt(1);
                if (a.kind != Shape::Kind::Matrix || da.kind != Shape::Kind::Matrix ||
                    a.dims[0] != a.dims[1] || da.dims[0] != da.dims[1] ||
                    a.dims[0] != da.dims[0] || (a.dims[0] != 2u && a.dims[0] != 3u)) {
                    return fail("LLVMGen: matrix function directional derivative expects two matching 2x2 or 3x3 square matrices");
                }
                out.shapes[idx] = a;
                break;
            }

            case FormExprType::MatrixPowerDirectionalDerivative: {
                const auto& a = childAt(0);
                const auto& da = childAt(1);
                const auto& p = childAt(2);
                if (a.kind != Shape::Kind::Matrix || da.kind != Shape::Kind::Matrix ||
                    a.dims[0] != a.dims[1] || da.dims[0] != da.dims[1] ||
                    a.dims[0] != da.dims[0] || (a.dims[0] != 2u && a.dims[0] != 3u)) {
                    return fail("LLVMGen: powm_dd expects two matching 2x2 or 3x3 square matrices");
                }
                if (!p.isScalar()) {
                    return fail("LLVMGen: powm_dd expects a scalar exponent");
                }
                out.shapes[idx] = a;
                break;
            }

            case FormExprType::SmoothAbsoluteValue:
            case FormExprType::SmoothSign:
            case FormExprType::SmoothHeaviside: {
                const auto& a = childAt(0);
                const auto& eps = childAt(1);
                if (!a.isScalar() || !eps.isScalar()) {
                    return fail("LLVMGen: smooth function expects scalar operands");
                }
                out.shapes[idx] = scalarShape();
                break;
            }

            case FormExprType::SmoothMin:
            case FormExprType::SmoothMax: {
                const auto& a = childAt(0);
                const auto& b = childAt(1);
                const auto& eps = childAt(2);
                if (!a.isScalar() || !b.isScalar() || !eps.isScalar()) {
                    return fail("LLVMGen: smooth min/max expects scalar operands");
                }
                out.shapes[idx] = scalarShape();
                break;
            }

            case FormExprType::SymmetricEigenvalue: {
                const auto& a = childAt(0);
                if (a.kind != Shape::Kind::Matrix || a.dims[0] != a.dims[1] || (a.dims[0] != 2u && a.dims[0] != 3u)) {
                    return fail("LLVMGen: eig_sym expects a 2x2 or 3x3 square matrix");
                }
                out.shapes[idx] = scalarShape();
                break;
            }

            case FormExprType::Eigenvalue: {
                const auto& a = childAt(0);
                if (a.kind != Shape::Kind::Matrix || a.dims[0] != a.dims[1] || (a.dims[0] != 2u && a.dims[0] != 3u)) {
                    return fail("LLVMGen: eig expects a 2x2 or 3x3 square matrix");
                }
                out.shapes[idx] = scalarShape();
                break;
            }

            case FormExprType::SymmetricEigenvector: {
                const auto& a = childAt(0);
                if (a.kind != Shape::Kind::Matrix || a.dims[0] != a.dims[1] || (a.dims[0] != 2u && a.dims[0] != 3u)) {
                    return fail("LLVMGen: eigvec_sym expects a 2x2 or 3x3 square matrix");
                }
                out.shapes[idx] = vectorShape(a.dims[0]);
                break;
            }

            case FormExprType::SpectralDecomposition: {
                const auto& a = childAt(0);
                if (a.kind != Shape::Kind::Matrix || a.dims[0] != a.dims[1] || (a.dims[0] != 2u && a.dims[0] != 3u)) {
                    return fail("LLVMGen: spectral_decomp expects a 2x2 or 3x3 square matrix");
                }
                out.shapes[idx] = a;
                break;
            }

            case FormExprType::SymmetricEigenvectorDirectionalDerivative: {
                const auto& a = childAt(0);
                const auto& da = childAt(1);
                if (a.kind != Shape::Kind::Matrix || da.kind != Shape::Kind::Matrix ||
                    a.dims[0] != a.dims[1] || da.dims[0] != da.dims[1] ||
                    a.dims[0] != da.dims[0] || (a.dims[0] != 2u && a.dims[0] != 3u)) {
                    return fail("LLVMGen: eigvec_sym_dd expects two matching 2x2 or 3x3 square matrices");
                }
                out.shapes[idx] = vectorShape(a.dims[0]);
                break;
            }

            case FormExprType::SpectralDecompositionDirectionalDerivative: {
                const auto& a = childAt(0);
                const auto& da = childAt(1);
                if (a.kind != Shape::Kind::Matrix || da.kind != Shape::Kind::Matrix ||
                    a.dims[0] != a.dims[1] || da.dims[0] != da.dims[1] ||
                    a.dims[0] != da.dims[0] || (a.dims[0] != 2u && a.dims[0] != 3u)) {
                    return fail("LLVMGen: spectral_decomp_dd expects two matching 2x2 or 3x3 square matrices");
                }
                out.shapes[idx] = a;
                break;
            }

            case FormExprType::HistoryWeightedSum:
            case FormExprType::HistoryConvolution: {
                for (std::size_t k = 0; k < static_cast<std::size_t>(op.child_count); ++k) {
                    if (!childAt(k).isScalar()) {
                        return fail("LLVMGen: history operator expects scalar weights");
                    }
                }
                if (trial_sig && trial_sig->field_type == FieldType::Vector) {
                    const auto vd = static_cast<std::uint32_t>(std::max(1, trial_sig->value_dimension));
                    out.shapes[idx] = vectorShape(vd);
                } else {
                    out.shapes[idx] = scalarShape();
                }
                break;
            }

            case FormExprType::SymmetricEigenvalueDirectionalDerivative: {
                const auto& a = childAt(0);
                const auto& b = childAt(1);
                if (a.kind != Shape::Kind::Matrix || b.kind != Shape::Kind::Matrix ||
                    a.dims[0] != a.dims[1] || b.dims[0] != b.dims[1] ||
                    a.dims[0] != b.dims[0] || (a.dims[0] != 2u && a.dims[0] != 3u)) {
                    return fail("LLVMGen: eig_sym_dd expects two 2x2 or 3x3 square matrices with matching dims");
                }
                out.shapes[idx] = scalarShape();
                break;
            }

            case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA: {
                const auto& a = childAt(0);
                const auto& b = childAt(1);
                const auto& c = childAt(2);
                if (a.kind != Shape::Kind::Matrix || b.kind != Shape::Kind::Matrix || c.kind != Shape::Kind::Matrix ||
                    a.dims[0] != a.dims[1] || b.dims[0] != b.dims[1] || c.dims[0] != c.dims[1] ||
                    a.dims[0] != b.dims[0] || a.dims[0] != c.dims[0] || (a.dims[0] != 2u && a.dims[0] != 3u)) {
                    return fail("LLVMGen: eig_sym_ddA expects three 2x2 or 3x3 square matrices with matching dims");
                }
                out.shapes[idx] = scalarShape();
                break;
            }

            case FormExprType::IndexedAccess: {
                if (op.child_count != 1u) {
                    return fail("LLVMGen: IndexedAccess expects exactly 1 child");
                }

                const auto& a = childAt(0);
                if (a.kind == Shape::Kind::Scalar) {
                    out.shapes[idx] = scalarShape();
                    break;
                }

                const auto rank = static_cast<std::uint32_t>(unpackIndexedRank(op.imm1));
                if (rank == 0u || rank > 4u) {
                    return fail("LLVMGen: IndexedAccess has invalid rank");
                }

                std::uint32_t expected_rank = 0u;
                switch (a.kind) {
                    case Shape::Kind::Scalar:
                        expected_rank = 0u;
                        break;
                    case Shape::Kind::Vector:
                        expected_rank = 1u;
                        break;
                    case Shape::Kind::Matrix:
                        expected_rank = 2u;
                        break;
                    case Shape::Kind::Tensor3:
                        expected_rank = 3u;
                        break;
                    case Shape::Kind::Tensor4:
                        expected_rank = 4u;
                        break;
                }

                if (rank != expected_rank) {
                    return fail("LLVMGen: IndexedAccess rank/operand shape mismatch");
                }

                for (std::uint32_t k = 0u; k < rank; ++k) {
                    const auto ext = static_cast<std::uint32_t>(unpackIndexedExtent(op.imm1, static_cast<int>(k)));
                    if (ext == 0u) {
                        return fail("LLVMGen: IndexedAccess has invalid (zero) extent");
                    }
                    const auto dimk = static_cast<std::uint32_t>(a.dims[k]);
                    if (dimk == 0u) {
                        return fail("LLVMGen: IndexedAccess operand has invalid dimension metadata");
                    }
                    if (ext > dimk) {
                        return fail("LLVMGen: IndexedAccess extent exceeds operand dimension");
                    }
                }

                out.shapes[idx] = scalarShape();
                break;
            }

            default:
                return fail("LLVMGen: unsupported op in shape inference (FormExprType=" +
                            std::to_string(static_cast<std::uint16_t>(op.type)) + ")");
        }
    }

    if (ir.root >= out.shapes.size()) {
        return fail("LLVMGen: invalid KernelIR root index");
    }

    if (require_scalar_root) {
        const auto root_shape = out.shapes[ir.root];
        if (!root_shape.isScalar()) {
            return fail("LLVMGen: integrand must lower to a scalar for AssemblyKernel evaluation");
        }
    }

    for (const auto& s : out.shapes) {
        switch (s.kind) {
            case Shape::Kind::Scalar:
                break;
            case Shape::Kind::Vector:
                if (s.dims[0] > kMaxVectorDim) {
                    return fail("LLVMGen: vector dimension exceeds JIT-fast-mode limit");
                }
                break;
            case Shape::Kind::Matrix:
                if (s.dims[0] > kMaxMatrixDim || s.dims[1] > kMaxMatrixDim) {
                    return fail("LLVMGen: matrix dimension exceeds JIT-fast-mode limit");
                }
                break;
            case Shape::Kind::Tensor3:
                if (s.dims[0] > kMaxTensor3Dim || s.dims[1] > kMaxTensor3Dim || s.dims[2] > kMaxTensor3Dim) {
                    return fail("LLVMGen: tensor3 dimension exceeds JIT-fast-mode limit");
                }
                break;
            case Shape::Kind::Tensor4:
                if (s.dims[0] > kMaxTensor4Dim || s.dims[1] > kMaxTensor4Dim || s.dims[2] > kMaxTensor4Dim ||
                    s.dims[3] > kMaxTensor4Dim) {
                    return fail("LLVMGen: tensor4 dimension exceeds JIT-fast-mode limit");
                }
                break;
        }
    }

    out.ok = true;
    return out;
}

[[nodiscard]] std::string toString(IntegralDomain d)
{
    switch (d) {
        case IntegralDomain::Cell: return "Cell";
        case IntegralDomain::Boundary: return "Boundary";
        case IntegralDomain::InteriorFace: return "InteriorFace";
        case IntegralDomain::InterfaceFace: return "InterfaceFace";
    }
    return "<unknown>";
}

#endif
} // namespace

LLVMGen::LLVMGen(JITOptions options)
    : options_(std::move(options))
{
}

LLVMGenResult LLVMGen::compileAndAddKernel(JITEngine& engine,
                                           const FormIR& ir,
                                           std::span<const std::size_t> term_indices,
                                           IntegralDomain domain,
                                           int boundary_marker,
                                           int interface_marker,
                                           std::string_view symbol,
                                           std::uintptr_t& out_address,
                                           const JITCompileSpecialization* specialization) const
{
    (void)engine;
    (void)ir;
    (void)term_indices;
    (void)domain;
    (void)boundary_marker;
    (void)interface_marker;
    (void)symbol;
    (void)specialization;
    out_address = 0;

#if !SVMP_FE_ENABLE_LLVM_JIT
    return LLVMGenResult{.ok = false, .message = "LLVMGen: FE was built without LLVM JIT support"};
#else
    (void)options_;

    try {
        if (!ir.isCompiled()) {
            return LLVMGenResult{.ok = false, .message = "LLVMGen: FormIR is not compiled"};
        }

        if (domain != IntegralDomain::Cell &&
            domain != IntegralDomain::Boundary &&
            domain != IntegralDomain::InteriorFace &&
            domain != IntegralDomain::InterfaceFace) {
            return LLVMGenResult{.ok = false,
                                 .message = "LLVMGen: domain not supported for codegen: " + toString(domain)};
        }

        if (term_indices.empty()) {
            return LLVMGenResult{.ok = false, .message = "LLVMGen: empty term set for kernel"};
        }

        // Pre-lower terms to deterministic KernelIR and validate shapes.
        struct LoweredTerm {
            KernelIR ir{};
            std::vector<Shape> shapes{};
            std::vector<std::uint8_t> dep_mask{};
            int time_derivative_order{0};

            std::optional<tensor::TensorIR> tensor_ir{};

            bool has_indexed_access{false};
            // Canonical (KernelIR-level) index ids and extents for Einstein-summation loops.
            std::vector<std::pair<std::uint16_t, std::uint8_t>> bound_indices{};
        };

        const int preferred_vector_width =
            preferredVectorWidthFromCpuFeatures(engine.cpuFeaturesString());

        std::vector<LoweredTerm> terms;
        terms.reserve(term_indices.size());

        for (const auto tidx : term_indices) {
            if (tidx >= ir.terms().size()) {
                return LLVMGenResult{.ok = false, .message = "LLVMGen: term index out of range"};
            }
            const auto& term = ir.terms()[tidx];
            if (term.domain != domain) {
                return LLVMGenResult{.ok = false, .message = "LLVMGen: term domain does not match kernel domain"};
            }
            FormExpr effective_integrand = term.integrand;
            std::optional<tensor::TensorIR> tensor_ir;

            if (options_.tensor.mode != TensorLoweringMode::Off) {
                try {
                    tensor::TensorIRLoweringOptions tensor_opts;
                    tensor_opts.enable_cache = false;
                    tensor_opts.force_loop_nest =
                        (options_.tensor.mode == TensorLoweringMode::On) || options_.tensor.force_loop_nest;
                    tensor_opts.log_decisions = options_.tensor.log_decisions;

                    tensor_opts.loop.enable_symmetry_lowering = options_.tensor.enable_symmetry_lowering;
                    tensor_opts.loop.enable_optimal_contraction_order = options_.tensor.enable_optimal_contraction_order;
                    tensor_opts.loop.enable_vectorization_hints = options_.vectorize;
                    tensor_opts.loop.preferred_vector_width = preferred_vector_width;
                    tensor_opts.loop.enable_delta_shortcuts = options_.tensor.enable_delta_shortcuts;
                    tensor_opts.loop.scalar_expansion_term_threshold = options_.tensor.scalar_expansion_term_threshold;

                    tensor_opts.alloc.stack_max_entries = options_.tensor.temp_stack_max_entries;
                    tensor_opts.alloc.alignment_bytes = options_.tensor.temp_alignment_bytes;
                    tensor_opts.alloc.enable_reuse = options_.tensor.temp_enable_reuse;

                    const auto tl = tensor::lowerToTensorIR(term.integrand, tensor_opts);
                    if (!tl.ok) {
                        return LLVMGenResult{
                            .ok = false,
                            .message = tl.message.empty() ? "LLVMGen: tensor lowering failed" : tl.message,
                        };
                    }
                    if (tl.used_loop_nest) {
                        tensor_ir = tl.ir;
                    } else if (tl.fallback_expr.isValid()) {
                        effective_integrand = tl.fallback_expr;
                    }
                } catch (const std::exception& e) {
                    return LLVMGenResult{
                        .ok = false,
                        .message = std::string("LLVMGen: tensor lowering threw exception: ") + e.what(),
                    };
                }
            }

            auto lowered = lowerToKernelIR(effective_integrand);
            if (lowered.ir.empty()) {
                return LLVMGenResult{.ok = false, .message = "LLVMGen: failed to lower term to KernelIR"};
            }

            if (options_.dump_kernel_ir) {
                std::ostringstream oss;
                oss << "Symbol: " << symbol << "\n"
                    << "Term index: " << tidx << "\n"
                    << "Domain: " << toString(domain) << "\n"
                    << "Boundary marker: " << boundary_marker << "\n"
                    << "Interface marker: " << interface_marker << "\n"
                    << "Integrand: " << term.debug_string << "\n\n"
                    << lowered.ir.dump();
                const auto path = dumpPath(options_, symbol, "_term" + std::to_string(tidx) + ".kernelir.txt");
                writeTextFile(path, oss.str());
            }

            auto shapes = inferShapes(lowered.ir, ir.testSpace(), ir.trialSpace());
            if (!shapes.ok) {
                return LLVMGenResult{.ok = false, .message = shapes.message};
            }

            for (const auto& op : lowered.ir.ops) {
                if (domain == IntegralDomain::Cell && op.type == FormExprType::Normal) {
                    return LLVMGenResult{.ok = false, .message = "LLVMGen: Normal is not supported for Cell kernels"};
                }
                if ((domain == IntegralDomain::Cell || domain == IntegralDomain::Boundary) &&
                    (op.type == FormExprType::RestrictMinus || op.type == FormExprType::RestrictPlus ||
                     op.type == FormExprType::Jump || op.type == FormExprType::Average)) {
                    return LLVMGenResult{.ok = false,
                                         .message = "LLVMGen: DG restriction/jump/avg operators are not supported outside face kernels"};
                }
                if (op.type == FormExprType::TimeDerivative) {
                    const int order = static_cast<int>(static_cast<std::int64_t>(op.imm0));
                    if (order < 1 || order > static_cast<int>(assembly::jit::kMaxTimeDerivativeOrderV6)) {
                        return LLVMGenResult{.ok = false, .message = "LLVMGen: dt(·,k) requires 1 <= k <= " +
                                                                       std::to_string(assembly::jit::kMaxTimeDerivativeOrderV6)};
                    }
                }
            }

            std::vector<std::uint8_t> dep;
            dep.resize(lowered.ir.ops.size(), 0u);
            for (std::size_t op_idx = 0; op_idx < lowered.ir.ops.size(); ++op_idx) {
                const auto& op = lowered.ir.ops[op_idx];
                std::uint8_t d = 0u;
                for (std::size_t k = 0; k < op.child_count; ++k) {
                    const auto c = lowered.ir.children[static_cast<std::size_t>(op.first_child) + k];
                    d = static_cast<std::uint8_t>(d | dep[c]);
                }
                if (op.type == FormExprType::TestFunction) {
                    d = static_cast<std::uint8_t>(d | 0x1u);
                } else if (op.type == FormExprType::TrialFunction && ir.kind() == FormKind::Bilinear) {
                    d = static_cast<std::uint8_t>(d | 0x2u);
                }
                dep[op_idx] = d;
            }

            bool has_indexed_access = false;
            std::vector<std::pair<std::uint16_t, std::uint8_t>> bound_indices;
            {
                struct Use {
                    std::uint8_t extent{0};
                    int count{0};
                };
                std::unordered_map<std::uint16_t, Use> uses;
                uses.reserve(8);

                for (const auto& op : lowered.ir.ops) {
                    if (op.type != FormExprType::IndexedAccess) {
                        continue;
                    }
                    has_indexed_access = true;
                    const int rank = static_cast<int>(unpackIndexedRank(op.imm1));
                    if (rank <= 0 || rank > 4) {
                        return LLVMGenResult{.ok = false, .message = "LLVMGen: IndexedAccess has invalid rank"};
                    }
                    for (int k = 0; k < rank; ++k) {
                        const auto id = unpackIndexedId(op.imm0, k);
                        const auto ext = unpackIndexedExtent(op.imm1, k);
                        if (ext == 0) {
                            return LLVMGenResult{.ok = false, .message = "LLVMGen: IndexedAccess has zero extent"};
                        }
                        auto& u = uses[id];
                        if (u.count == 0) {
                            u.extent = ext;
                        } else if (u.extent != ext) {
                            return LLVMGenResult{.ok = false, .message = "LLVMGen: IndexedAccess uses inconsistent extents for the same index id"};
                        }
                        u.count += 1;
                    }
                }

                if (has_indexed_access) {
                    for (const auto& [id, u] : uses) {
                        if (u.count != 2) {
                            return LLVMGenResult{
                                .ok = false,
                                .message = "LLVMGen: IndexedAccess currently requires fully-contracted Einstein sums (each index must appear exactly twice)",
                            };
                        }
                        bound_indices.emplace_back(id, u.extent);
                    }
                    std::sort(bound_indices.begin(), bound_indices.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                }
            }

            terms.push_back(LoweredTerm{
                .ir = std::move(lowered.ir),
                .shapes = std::move(shapes.shapes),
                .dep_mask = std::move(dep),
                .time_derivative_order = term.time_derivative_order,
                .tensor_ir = std::move(tensor_ir),
                .has_indexed_access = has_indexed_access,
                .bound_indices = std::move(bound_indices),
            });
        }

        auto ctx = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>(std::string(symbol), *ctx);
        module->setModuleIdentifier(std::string(symbol));

        std::unique_ptr<llvm::DIBuilder> di_builder;
        llvm::DICompileUnit* di_cu = nullptr;
        llvm::DIFile* di_file = nullptr;

        if (options_.debug_info) {
            module->setSourceFileName("svmp_fe_jit");
            di_builder = std::make_unique<llvm::DIBuilder>(*module);
            di_file = di_builder->createFile("svmp_fe_jit", ".");
            di_cu = di_builder->createCompileUnit(llvm::dwarf::DW_LANG_C_plus_plus,
                                                  di_file,
                                                  "svmp_fe_jit",
                                                  /*isOptimized=*/options_.optimization_level > 0,
                                                  /*Flags=*/"",
                                                  /*RV=*/0);
        }

        const std::string target_triple = engine.targetTriple();
        const std::string data_layout = engine.dataLayoutString();
        if (!target_triple.empty()) {
            module->setTargetTriple(target_triple);
        }
        if (!data_layout.empty()) {
            module->setDataLayout(data_layout);
        }

        llvm::IRBuilder<> builder(*ctx);
        auto* i8_ptr = builder.getInt8PtrTy();
        auto* fn_ty = llvm::FunctionType::get(builder.getVoidTy(),
                                              {i8_ptr},
                                              /*isVarArg=*/false);

        auto* fn = llvm::Function::Create(fn_ty,
                                          llvm::GlobalValue::ExternalLinkage,
                                          std::string(symbol),
                                          module.get());

        if (di_builder && di_cu && di_file) {
            auto* sub_type =
                di_builder->createSubroutineType(di_builder->getOrCreateTypeArray({}));
            auto* sp = di_builder->createFunction(di_cu,
                                                  std::string(symbol),
                                                  std::string(symbol),
                                                  di_file,
                                                  /*LineNo=*/1,
                                                  sub_type,
                                                  /*ScopeLine=*/1,
                                                  llvm::DINode::FlagZero,
                                                  llvm::DISubprogram::SPFlagDefinition);
            fn->setSubprogram(sp);
            builder.SetCurrentDebugLocation(llvm::DILocation::get(*ctx, 1, 1, sp));
        }

        auto* entry = llvm::BasicBlock::Create(*ctx, "entry", fn);
        builder.SetInsertPoint(entry);

        auto* args_ptr = fn->getArg(0);
        args_ptr->setName("args");

#if 0
        struct ABIV3 {
            static constexpr std::size_t cell_side_off = offsetof(assembly::jit::CellKernelArgsV3, side);
            static constexpr std::size_t cell_out_off = offsetof(assembly::jit::CellKernelArgsV3, output);

            static constexpr std::size_t bdry_side_off = offsetof(assembly::jit::BoundaryFaceKernelArgsV3, side);
            static constexpr std::size_t bdry_out_off = offsetof(assembly::jit::BoundaryFaceKernelArgsV3, output);

            static constexpr std::size_t face_minus_side_off = offsetof(assembly::jit::InteriorFaceKernelArgsV3, minus);
            static constexpr std::size_t face_plus_side_off = offsetof(assembly::jit::InteriorFaceKernelArgsV3, plus);

            static constexpr std::size_t face_out_minus_off =
                offsetof(assembly::jit::InteriorFaceKernelArgsV3, output_minus);
            static constexpr std::size_t face_out_plus_off =
                offsetof(assembly::jit::InteriorFaceKernelArgsV3, output_plus);
            static constexpr std::size_t face_coupling_minus_plus_off =
                offsetof(assembly::jit::InteriorFaceKernelArgsV3, coupling_minus_plus);
            static constexpr std::size_t face_coupling_plus_minus_off =
                offsetof(assembly::jit::InteriorFaceKernelArgsV3, coupling_plus_minus);

            static constexpr std::size_t side_dim_off = offsetof(assembly::jit::KernelSideArgsV3, dim);
            static constexpr std::size_t side_n_qpts_off = offsetof(assembly::jit::KernelSideArgsV3, n_qpts);
            static constexpr std::size_t side_n_test_dofs_off = offsetof(assembly::jit::KernelSideArgsV3, n_test_dofs);
            static constexpr std::size_t side_n_trial_dofs_off = offsetof(assembly::jit::KernelSideArgsV3, n_trial_dofs);

            static constexpr std::size_t side_test_field_type_off =
                offsetof(assembly::jit::KernelSideArgsV3, test_field_type);
            static constexpr std::size_t side_trial_field_type_off =
                offsetof(assembly::jit::KernelSideArgsV3, trial_field_type);
            static constexpr std::size_t side_test_value_dim_off =
                offsetof(assembly::jit::KernelSideArgsV3, test_value_dim);
            static constexpr std::size_t side_trial_value_dim_off =
                offsetof(assembly::jit::KernelSideArgsV3, trial_value_dim);
            static constexpr std::size_t side_test_uses_vector_basis_off =
                offsetof(assembly::jit::KernelSideArgsV3, test_uses_vector_basis);
            static constexpr std::size_t side_trial_uses_vector_basis_off =
                offsetof(assembly::jit::KernelSideArgsV3, trial_uses_vector_basis);

            static constexpr std::size_t side_integration_weights_off =
                offsetof(assembly::jit::KernelSideArgsV3, integration_weights);

            static constexpr std::size_t side_quad_points_xyz_off =
                offsetof(assembly::jit::KernelSideArgsV3, quad_points_xyz);
            static constexpr std::size_t side_physical_points_xyz_off =
                offsetof(assembly::jit::KernelSideArgsV3, physical_points_xyz);

            static constexpr std::size_t side_jacobians_off =
                offsetof(assembly::jit::KernelSideArgsV3, jacobians);
            static constexpr std::size_t side_inverse_jacobians_off =
                offsetof(assembly::jit::KernelSideArgsV3, inverse_jacobians);
            static constexpr std::size_t side_jacobian_dets_off =
                offsetof(assembly::jit::KernelSideArgsV3, jacobian_dets);
            static constexpr std::size_t side_normals_xyz_off = offsetof(assembly::jit::KernelSideArgsV3, normals_xyz);

            static constexpr std::size_t side_test_basis_values_off =
                offsetof(assembly::jit::KernelSideArgsV3, test_basis_values);
            static constexpr std::size_t side_trial_basis_values_off =
                offsetof(assembly::jit::KernelSideArgsV3, trial_basis_values);
            static constexpr std::size_t side_test_phys_grads_off =
                offsetof(assembly::jit::KernelSideArgsV3, test_phys_gradients_xyz);
            static constexpr std::size_t side_trial_phys_grads_off =
                offsetof(assembly::jit::KernelSideArgsV3, trial_phys_gradients_xyz);
            static constexpr std::size_t side_test_phys_hess_off =
                offsetof(assembly::jit::KernelSideArgsV3, test_phys_hessians);
            static constexpr std::size_t side_trial_phys_hess_off =
                offsetof(assembly::jit::KernelSideArgsV3, trial_phys_hessians);

            static constexpr std::size_t side_test_vector_basis_values_xyz_off =
                offsetof(assembly::jit::KernelSideArgsV3, test_basis_vector_values_xyz);
            static constexpr std::size_t side_test_vector_basis_curls_xyz_off =
                offsetof(assembly::jit::KernelSideArgsV3, test_basis_curls_xyz);
            static constexpr std::size_t side_test_vector_basis_divs_off =
                offsetof(assembly::jit::KernelSideArgsV3, test_basis_divergences);

            static constexpr std::size_t side_trial_vector_basis_values_xyz_off =
                offsetof(assembly::jit::KernelSideArgsV3, trial_basis_vector_values_xyz);
            static constexpr std::size_t side_trial_vector_basis_curls_xyz_off =
                offsetof(assembly::jit::KernelSideArgsV3, trial_basis_curls_xyz);
            static constexpr std::size_t side_trial_vector_basis_divs_off =
                offsetof(assembly::jit::KernelSideArgsV3, trial_basis_divergences);

            static constexpr std::size_t side_solution_coefficients_off =
                offsetof(assembly::jit::KernelSideArgsV3, solution_coefficients);
            static constexpr std::size_t side_num_previous_solutions_off =
                offsetof(assembly::jit::KernelSideArgsV3, num_previous_solutions);
            static constexpr std::size_t side_previous_solution_coefficients_off =
                offsetof(assembly::jit::KernelSideArgsV3, previous_solution_coefficients);

            static constexpr std::size_t side_field_solutions_off =
                offsetof(assembly::jit::KernelSideArgsV3, field_solutions);
            static constexpr std::size_t side_num_field_solutions_off =
                offsetof(assembly::jit::KernelSideArgsV3, num_field_solutions);

            static constexpr std::size_t side_jit_constants_off = offsetof(assembly::jit::KernelSideArgsV3, jit_constants);
            static constexpr std::size_t side_coupled_integrals_off =
                offsetof(assembly::jit::KernelSideArgsV3, coupled_integrals);
            static constexpr std::size_t side_coupled_aux_off = offsetof(assembly::jit::KernelSideArgsV3, coupled_aux);

            static constexpr std::size_t side_time_off = offsetof(assembly::jit::KernelSideArgsV3, time);
            static constexpr std::size_t side_dt_off = offsetof(assembly::jit::KernelSideArgsV3, dt);
            static constexpr std::size_t side_cell_domain_id_off = offsetof(assembly::jit::KernelSideArgsV3, cell_domain_id);
            static constexpr std::size_t side_cell_diameter_off = offsetof(assembly::jit::KernelSideArgsV3, cell_diameter);
            static constexpr std::size_t side_cell_volume_off = offsetof(assembly::jit::KernelSideArgsV3, cell_volume);
            static constexpr std::size_t side_facet_area_off = offsetof(assembly::jit::KernelSideArgsV3, facet_area);

            static constexpr std::size_t side_time_derivative_term_weight_off =
                offsetof(assembly::jit::KernelSideArgsV3, time_derivative_term_weight);
            static constexpr std::size_t side_non_time_derivative_term_weight_off =
                offsetof(assembly::jit::KernelSideArgsV3, non_time_derivative_term_weight);
            static constexpr std::size_t side_dt1_term_weight_off =
                offsetof(assembly::jit::KernelSideArgsV3, dt1_term_weight);
            static constexpr std::size_t side_dt2_term_weight_off =
                offsetof(assembly::jit::KernelSideArgsV3, dt2_term_weight);

            static constexpr std::size_t side_dt1_coeff0_off = offsetof(assembly::jit::KernelSideArgsV3, dt1_coeff0);
            static constexpr std::size_t side_dt2_coeff0_off = offsetof(assembly::jit::KernelSideArgsV3, dt2_coeff0);

            static constexpr std::size_t side_material_state_old_base_off =
                offsetof(assembly::jit::KernelSideArgsV3, material_state_old_base);
            static constexpr std::size_t side_material_state_work_base_off =
                offsetof(assembly::jit::KernelSideArgsV3, material_state_work_base);
            static constexpr std::size_t side_material_state_stride_bytes_off =
                offsetof(assembly::jit::KernelSideArgsV3, material_state_stride_bytes);

            static constexpr std::size_t out_element_matrix_off = offsetof(assembly::jit::KernelOutputViewV3, element_matrix);
            static constexpr std::size_t out_element_vector_off = offsetof(assembly::jit::KernelOutputViewV3, element_vector);
            static constexpr std::size_t out_n_test_dofs_off = offsetof(assembly::jit::KernelOutputViewV3, n_test_dofs);
            static constexpr std::size_t out_n_trial_dofs_off = offsetof(assembly::jit::KernelOutputViewV3, n_trial_dofs);

            static constexpr std::size_t field_entry_field_id_off = offsetof(assembly::jit::FieldSolutionEntryV1, field_id);
            static constexpr std::size_t field_entry_field_type_off = offsetof(assembly::jit::FieldSolutionEntryV1, field_type);
            static constexpr std::size_t field_entry_value_dim_off = offsetof(assembly::jit::FieldSolutionEntryV1, value_dim);

            static constexpr std::size_t field_entry_values_off = offsetof(assembly::jit::FieldSolutionEntryV1, values);
            static constexpr std::size_t field_entry_gradients_xyz_off =
                offsetof(assembly::jit::FieldSolutionEntryV1, gradients_xyz);
            static constexpr std::size_t field_entry_hessians_off =
                offsetof(assembly::jit::FieldSolutionEntryV1, hessians);
            static constexpr std::size_t field_entry_laplacians_off = offsetof(assembly::jit::FieldSolutionEntryV1, laplacians);

            static constexpr std::size_t field_entry_vector_values_xyz_off =
                offsetof(assembly::jit::FieldSolutionEntryV1, vector_values_xyz);
            static constexpr std::size_t field_entry_jacobians_off = offsetof(assembly::jit::FieldSolutionEntryV1, jacobians);
            static constexpr std::size_t field_entry_component_hessians_off =
                offsetof(assembly::jit::FieldSolutionEntryV1, component_hessians);
            static constexpr std::size_t field_entry_component_laplacians_off =
                offsetof(assembly::jit::FieldSolutionEntryV1, component_laplacians);

            static constexpr std::size_t field_entry_history_count_off =
                offsetof(assembly::jit::FieldSolutionEntryV1, history_count);
            static constexpr std::size_t field_entry_history_values_off =
                offsetof(assembly::jit::FieldSolutionEntryV1, history_values);
            static constexpr std::size_t field_entry_history_vector_values_xyz_off =
                offsetof(assembly::jit::FieldSolutionEntryV1, history_vector_values_xyz);
        };
#endif

        auto gepBytes = [&](llvm::Value* base, std::size_t off) -> llvm::Value* {
            return builder.CreateGEP(builder.getInt8Ty(),
                                     base,
                                     llvm::ConstantInt::get(builder.getInt64Ty(), static_cast<std::uint64_t>(off)));
        };

        auto loadU32 = [&](llvm::Value* base, std::size_t off) -> llvm::Value* {
            auto* addr = gepBytes(base, off);
            return builder.CreateLoad(builder.getInt32Ty(), addr);
        };

        auto loadI32 = [&](llvm::Value* base, std::size_t off) -> llvm::Value* {
            auto* addr = gepBytes(base, off);
            return builder.CreateLoad(builder.getInt32Ty(), addr);
        };

        auto loadU64 = [&](llvm::Value* base, std::size_t off) -> llvm::Value* {
            auto* addr = gepBytes(base, off);
            return builder.CreateLoad(builder.getInt64Ty(), addr);
        };

        auto loadF64 = [&](llvm::Value* base, std::size_t off) -> llvm::Value* {
            auto* addr = gepBytes(base, off);
            return builder.CreateLoad(builder.getDoubleTy(), addr);
        };

        auto loadPtr = [&](llvm::Value* base, std::size_t off) -> llvm::Value* {
            auto* addr = gepBytes(base, off);
            return builder.CreateLoad(i8_ptr, addr);
        };

        auto* f64 = builder.getDoubleTy();
        auto* i32 = builder.getInt32Ty();
        auto* i64 = builder.getInt64Ty();
        auto* f64_ptr = llvm::PointerType::getUnqual(f64);

        // External-call trampolines (relaxed-mode).
        auto coeff_eval_scalar_fn =
            module->getOrInsertFunction("svmp_fe_jit_coeff_eval_scalar_v1",
                                        f64,
                                        i8_ptr,
                                        i32,
                                        i8_ptr);
        auto coeff_eval_vector_fn =
            module->getOrInsertFunction("svmp_fe_jit_coeff_eval_vector_v1",
                                        builder.getVoidTy(),
                                        i8_ptr,
                                        i32,
                                        i8_ptr,
                                        f64_ptr);
        auto coeff_eval_matrix_fn =
            module->getOrInsertFunction("svmp_fe_jit_coeff_eval_matrix_v1",
                                        builder.getVoidTy(),
                                        i8_ptr,
                                        i32,
                                        i8_ptr,
                                        f64_ptr);
        auto coeff_eval_tensor3_fn =
            module->getOrInsertFunction("svmp_fe_jit_coeff_eval_tensor3_v1",
                                        builder.getVoidTy(),
                                        i8_ptr,
                                        i32,
                                        i8_ptr,
                                        f64_ptr);
        auto coeff_eval_tensor4_fn =
            module->getOrInsertFunction("svmp_fe_jit_coeff_eval_tensor4_v1",
                                        builder.getVoidTy(),
                                        i8_ptr,
                                        i32,
                                        i8_ptr,
                                        f64_ptr);

        auto* value_view_ty = llvm::StructType::create(*ctx, "svmp_fe_jit.ValueViewV1");
        value_view_ty->setBody({i32, i32, f64_ptr}, /*isPacked=*/false);
        auto* value_view_ptr_ty = llvm::PointerType::getUnqual(value_view_ty);

        auto constitutive_eval_fn =
            module->getOrInsertFunction("svmp_fe_jit_constitutive_eval_v1",
                                        builder.getVoidTy(),
                                        i8_ptr,            // side_ptr
                                        i32,               // q
                                        i32,               // trace_side
                                        i8_ptr,            // constitutive_node_ptr
                                        i32,               // output_index
                                        value_view_ptr_ty, // inputs
                                        i32,               // num_inputs
                                        i32,               // out_kind
                                        f64_ptr,           // out_values
                                        i32);              // out_len

        // Spectral helpers (symmetric 2x2/3x3 eigenvalues; strict-mode, cacheable).
        auto eig_sym_2x2_fn =
            module->getOrInsertFunction("svmp_fe_sym_eigenvalue_2x2_v1", f64, f64_ptr, i32);
        auto eig_sym_3x3_fn =
            module->getOrInsertFunction("svmp_fe_sym_eigenvalue_3x3_v1", f64, f64_ptr, i32);
        auto eig_sym_dd_2x2_fn =
            module->getOrInsertFunction("svmp_fe_sym_eigenvalue_dd_2x2_v1", f64, f64_ptr, f64_ptr, i32);
        auto eig_sym_dd_3x3_fn =
            module->getOrInsertFunction("svmp_fe_sym_eigenvalue_dd_3x3_v1", f64, f64_ptr, f64_ptr, i32);
        auto eig_sym_ddA_2x2_fn =
            module->getOrInsertFunction("svmp_fe_sym_eigenvalue_ddA_2x2_v1", f64, f64_ptr, f64_ptr, f64_ptr, i32);
        auto eig_sym_ddA_3x3_fn =
            module->getOrInsertFunction("svmp_fe_sym_eigenvalue_ddA_3x3_v1", f64, f64_ptr, f64_ptr, f64_ptr, i32);

        // Symmetric eigendecomposition helpers (eigenvalues + eigenvectors; cacheable).
        auto eig_sym_full_2x2_fn =
            module->getOrInsertFunction("svmp_fe_jit_eig_sym_2x2_v1", builder.getVoidTy(), f64_ptr, f64_ptr, f64_ptr);
        auto eig_sym_full_3x3_fn =
            module->getOrInsertFunction("svmp_fe_jit_eig_sym_3x3_v1", builder.getVoidTy(), f64_ptr, f64_ptr, f64_ptr);

        // Matrix function helpers (small 2x2/3x3; versioned symbols).
        auto mat_exp_2x2_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_exp_2x2_v1", builder.getVoidTy(), f64_ptr, f64_ptr);
        auto mat_exp_3x3_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_exp_3x3_v1", builder.getVoidTy(), f64_ptr, f64_ptr);
        auto mat_log_2x2_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_log_2x2_v1", builder.getVoidTy(), f64_ptr, f64_ptr);
        auto mat_log_3x3_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_log_3x3_v1", builder.getVoidTy(), f64_ptr, f64_ptr);
        auto mat_sqrt_2x2_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_sqrt_2x2_v1", builder.getVoidTy(), f64_ptr, f64_ptr);
        auto mat_sqrt_3x3_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_sqrt_3x3_v1", builder.getVoidTy(), f64_ptr, f64_ptr);
        auto mat_pow_2x2_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_pow_2x2_v1", builder.getVoidTy(), f64_ptr, f64, f64_ptr);
        auto mat_pow_3x3_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_pow_3x3_v1", builder.getVoidTy(), f64_ptr, f64, f64_ptr);

        // Matrix function directional derivatives (Fréchet derivatives).
        auto mat_exp_dd_2x2_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_exp_dd_2x2_v1", builder.getVoidTy(), f64_ptr, f64_ptr, f64_ptr);
        auto mat_exp_dd_3x3_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_exp_dd_3x3_v1", builder.getVoidTy(), f64_ptr, f64_ptr, f64_ptr);
        auto mat_log_dd_2x2_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_log_dd_2x2_v1", builder.getVoidTy(), f64_ptr, f64_ptr, f64_ptr);
        auto mat_log_dd_3x3_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_log_dd_3x3_v1", builder.getVoidTy(), f64_ptr, f64_ptr, f64_ptr);
        auto mat_sqrt_dd_2x2_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_sqrt_dd_2x2_v1", builder.getVoidTy(), f64_ptr, f64_ptr, f64_ptr);
        auto mat_sqrt_dd_3x3_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_sqrt_dd_3x3_v1", builder.getVoidTy(), f64_ptr, f64_ptr, f64_ptr);
        auto mat_pow_dd_2x2_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_pow_dd_2x2_v1", builder.getVoidTy(), f64_ptr, f64_ptr, f64, f64_ptr);
        auto mat_pow_dd_3x3_fn =
            module->getOrInsertFunction("svmp_fe_jit_matrix_pow_dd_3x3_v1", builder.getVoidTy(), f64_ptr, f64_ptr, f64, f64_ptr);

        // Symmetric eigen operator directional derivatives.
        auto eigvec_sym_dd_2x2_fn =
            module->getOrInsertFunction("svmp_fe_jit_eigvec_sym_dd_2x2_v1", builder.getVoidTy(), f64_ptr, f64_ptr, i32, f64_ptr);
        auto eigvec_sym_dd_3x3_fn =
            module->getOrInsertFunction("svmp_fe_jit_eigvec_sym_dd_3x3_v1", builder.getVoidTy(), f64_ptr, f64_ptr, i32, f64_ptr);
        auto spectral_decomp_dd_2x2_fn =
            module->getOrInsertFunction("svmp_fe_jit_spectral_decomp_dd_2x2_v1", builder.getVoidTy(), f64_ptr, f64_ptr, f64_ptr);
        auto spectral_decomp_dd_3x3_fn =
            module->getOrInsertFunction("svmp_fe_jit_spectral_decomp_dd_3x3_v1", builder.getVoidTy(), f64_ptr, f64_ptr, f64_ptr);

        auto f64c = [&](double v) -> llvm::Constant* {
            return llvm::ConstantFP::get(f64, v);
        };

        auto loadRealPtrAt = [&](llvm::Value* base_ptr,
                                 llvm::Value* index64) -> llvm::Value* {
            auto* gep = builder.CreateGEP(f64, base_ptr, index64);
            return builder.CreateLoad(f64, gep);
        };

        auto storeRealPtrAt = [&](llvm::Value* base_ptr,
                                  llvm::Value* index64,
                                  llvm::Value* value) -> void {
            auto* gep = builder.CreateGEP(f64, base_ptr, index64);
            builder.CreateStore(value, gep);
        };

        struct SideView {
            llvm::Value* side_ptr{nullptr};

            llvm::Value* dim{nullptr};
            llvm::Value* n_qpts{nullptr};
            llvm::Value* n_test_dofs{nullptr};
            llvm::Value* n_trial_dofs{nullptr};

            llvm::Value* test_field_type{nullptr};
            llvm::Value* trial_field_type{nullptr};
            llvm::Value* test_value_dim{nullptr};
            llvm::Value* trial_value_dim{nullptr};
            llvm::Value* test_uses_vector_basis{nullptr};
            llvm::Value* trial_uses_vector_basis{nullptr};

            llvm::Value* cell_domain_id{nullptr}; // i32
            llvm::Value* cell_diameter{nullptr};
            llvm::Value* cell_volume{nullptr};
            llvm::Value* facet_area{nullptr};

            llvm::Value* integration_weights{nullptr};
            llvm::Value* quad_points_xyz{nullptr};
            llvm::Value* physical_points_xyz{nullptr};
            llvm::Value* jacobians{nullptr};
            llvm::Value* inverse_jacobians{nullptr};
            llvm::Value* jacobian_dets{nullptr};
            llvm::Value* normals_xyz{nullptr};
            llvm::Value* interleaved_qpoint_geometry{nullptr};
            llvm::Value* interleaved_qpoint_geometry_stride_reals{nullptr}; // i32
            llvm::Value* interleaved_qpoint_geometry_physical_offset{nullptr}; // i32
            llvm::Value* interleaved_qpoint_geometry_jacobian_offset{nullptr}; // i32
            llvm::Value* interleaved_qpoint_geometry_inverse_jacobian_offset{nullptr}; // i32
            llvm::Value* interleaved_qpoint_geometry_det_offset{nullptr}; // i32
            llvm::Value* interleaved_qpoint_geometry_normal_offset{nullptr}; // i32

            llvm::Value* test_basis_values{nullptr};
            llvm::Value* trial_basis_values{nullptr};
            llvm::Value* test_phys_grads_xyz{nullptr};
            llvm::Value* trial_phys_grads_xyz{nullptr};
            llvm::Value* test_phys_hessians{nullptr};
            llvm::Value* trial_phys_hessians{nullptr};

            llvm::Value* test_basis_vector_values_xyz{nullptr};
            llvm::Value* test_basis_curls_xyz{nullptr};
            llvm::Value* test_basis_divs{nullptr};

            llvm::Value* trial_basis_vector_values_xyz{nullptr};
            llvm::Value* trial_basis_curls_xyz{nullptr};
            llvm::Value* trial_basis_divs{nullptr};

            llvm::Value* solution_coefficients{nullptr};
            llvm::Value* num_previous_solutions{nullptr};
            llvm::Value* previous_solution_coefficients_base{nullptr}; // i8*

            llvm::Value* num_history_steps{nullptr};
            llvm::Value* history_weights{nullptr};
            llvm::Value* history_solution_coefficients_base{nullptr}; // i8*

            llvm::Value* field_solutions{nullptr};
            llvm::Value* num_field_solutions{nullptr};

            llvm::Value* jit_constants{nullptr};
            llvm::Value* coupled_integrals{nullptr};
            llvm::Value* coupled_aux{nullptr};

            llvm::Value* time{nullptr};
            llvm::Value* dt{nullptr};

            llvm::Value* time_derivative_term_weight{nullptr};
            llvm::Value* non_time_derivative_term_weight{nullptr};
            llvm::Value* dt_stencil_coeffs_base{nullptr}; // f64*
            llvm::Value* dt_term_weights_base{nullptr};   // f64*
            llvm::Value* max_time_derivative_order{nullptr}; // i32

            llvm::Value* material_state_old_base{nullptr};
            llvm::Value* material_state_work_base{nullptr};
            llvm::Value* material_state_stride_bytes{nullptr};
        };

        auto loadSideView = [&](llvm::Value* side_ptr,
                                const std::optional<std::uint32_t>& fixed_n_qpts,
                                const std::optional<std::uint32_t>& fixed_n_test_dofs,
                                const std::optional<std::uint32_t>& fixed_n_trial_dofs) -> SideView {
            SideView s;
            s.side_ptr = side_ptr;

            s.dim = loadU32(side_ptr, ABIV3::side_dim_off);
            s.n_qpts = fixed_n_qpts ? builder.getInt32(*fixed_n_qpts)
                                    : loadU32(side_ptr, ABIV3::side_n_qpts_off);
            s.n_test_dofs = fixed_n_test_dofs ? builder.getInt32(*fixed_n_test_dofs)
                                              : loadU32(side_ptr, ABIV3::side_n_test_dofs_off);
            s.n_trial_dofs = fixed_n_trial_dofs ? builder.getInt32(*fixed_n_trial_dofs)
                                                : loadU32(side_ptr, ABIV3::side_n_trial_dofs_off);

            s.test_field_type = loadU32(side_ptr, ABIV3::side_test_field_type_off);
            s.trial_field_type = loadU32(side_ptr, ABIV3::side_trial_field_type_off);
            s.test_value_dim = loadU32(side_ptr, ABIV3::side_test_value_dim_off);
            s.trial_value_dim = loadU32(side_ptr, ABIV3::side_trial_value_dim_off);
            s.test_uses_vector_basis = loadU32(side_ptr, ABIV3::side_test_uses_vector_basis_off);
            s.trial_uses_vector_basis = loadU32(side_ptr, ABIV3::side_trial_uses_vector_basis_off);

            s.cell_domain_id = loadI32(side_ptr, ABIV3::side_cell_domain_id_off);
            s.cell_diameter = loadF64(side_ptr, ABIV3::side_cell_diameter_off);
            s.cell_volume = loadF64(side_ptr, ABIV3::side_cell_volume_off);
            s.facet_area = loadF64(side_ptr, ABIV3::side_facet_area_off);

            s.integration_weights = loadPtr(side_ptr, ABIV3::side_integration_weights_off);
            s.quad_points_xyz = loadPtr(side_ptr, ABIV3::side_quad_points_xyz_off);
            s.physical_points_xyz = loadPtr(side_ptr, ABIV3::side_physical_points_xyz_off);
            s.jacobians = loadPtr(side_ptr, ABIV3::side_jacobians_off);
            s.inverse_jacobians = loadPtr(side_ptr, ABIV3::side_inverse_jacobians_off);
            s.jacobian_dets = loadPtr(side_ptr, ABIV3::side_jacobian_dets_off);
            s.normals_xyz = loadPtr(side_ptr, ABIV3::side_normals_xyz_off);
            s.interleaved_qpoint_geometry = loadPtr(side_ptr, ABIV3::side_interleaved_geom_off);
            s.interleaved_qpoint_geometry_stride_reals = loadU32(side_ptr, ABIV3::side_interleaved_geom_stride_off);
            s.interleaved_qpoint_geometry_physical_offset = loadU32(side_ptr, ABIV3::side_interleaved_geom_phys_off);
            s.interleaved_qpoint_geometry_jacobian_offset = loadU32(side_ptr, ABIV3::side_interleaved_geom_jac_off);
            s.interleaved_qpoint_geometry_inverse_jacobian_offset =
                loadU32(side_ptr, ABIV3::side_interleaved_geom_jinv_off);
            s.interleaved_qpoint_geometry_det_offset = loadU32(side_ptr, ABIV3::side_interleaved_geom_det_off);
            s.interleaved_qpoint_geometry_normal_offset = loadU32(side_ptr, ABIV3::side_interleaved_geom_normal_off);

            s.test_basis_values = loadPtr(side_ptr, ABIV3::side_test_basis_values_off);
            s.trial_basis_values = loadPtr(side_ptr, ABIV3::side_trial_basis_values_off);
            s.test_phys_grads_xyz = loadPtr(side_ptr, ABIV3::side_test_phys_grads_off);
            s.trial_phys_grads_xyz = loadPtr(side_ptr, ABIV3::side_trial_phys_grads_off);
            s.test_phys_hessians = loadPtr(side_ptr, ABIV3::side_test_phys_hess_off);
            s.trial_phys_hessians = loadPtr(side_ptr, ABIV3::side_trial_phys_hess_off);

            s.test_basis_vector_values_xyz = loadPtr(side_ptr, ABIV3::side_test_vector_basis_values_xyz_off);
            s.test_basis_curls_xyz = loadPtr(side_ptr, ABIV3::side_test_vector_basis_curls_xyz_off);
            s.test_basis_divs = loadPtr(side_ptr, ABIV3::side_test_vector_basis_divs_off);

            s.trial_basis_vector_values_xyz = loadPtr(side_ptr, ABIV3::side_trial_vector_basis_values_xyz_off);
            s.trial_basis_curls_xyz = loadPtr(side_ptr, ABIV3::side_trial_vector_basis_curls_xyz_off);
            s.trial_basis_divs = loadPtr(side_ptr, ABIV3::side_trial_vector_basis_divs_off);

            s.solution_coefficients = loadPtr(side_ptr, ABIV3::side_solution_coefficients_off);
            s.num_previous_solutions = loadU32(side_ptr, ABIV3::side_num_previous_solutions_off);
            s.previous_solution_coefficients_base = gepBytes(side_ptr, ABIV3::side_previous_solution_coefficients_off);

            s.num_history_steps = loadU32(side_ptr, ABIV3::side_num_history_steps_off);
            s.history_weights = loadPtr(side_ptr, ABIV3::side_history_weights_off);
            s.history_solution_coefficients_base = gepBytes(side_ptr, ABIV3::side_history_solution_coefficients_off);

            s.field_solutions = loadPtr(side_ptr, ABIV3::side_field_solutions_off);
            s.num_field_solutions = loadU32(side_ptr, ABIV3::side_num_field_solutions_off);

            s.jit_constants = loadPtr(side_ptr, ABIV3::side_jit_constants_off);
            s.coupled_integrals = loadPtr(side_ptr, ABIV3::side_coupled_integrals_off);
            s.coupled_aux = loadPtr(side_ptr, ABIV3::side_coupled_aux_off);

            s.time = loadF64(side_ptr, ABIV3::side_time_off);
            s.dt = loadF64(side_ptr, ABIV3::side_dt_off);

            s.time_derivative_term_weight = loadF64(side_ptr, ABIV3::side_time_derivative_term_weight_off);
            s.non_time_derivative_term_weight = loadF64(side_ptr, ABIV3::side_non_time_derivative_term_weight_off);
            s.dt_stencil_coeffs_base =
                builder.CreatePointerCast(gepBytes(side_ptr, ABIV3::side_dt_stencil_coeffs_off), f64_ptr);
            s.dt_term_weights_base =
                builder.CreatePointerCast(gepBytes(side_ptr, ABIV3::side_dt_term_weights_off), f64_ptr);
            s.max_time_derivative_order = loadU32(side_ptr, ABIV3::side_max_time_derivative_order_off);

            s.material_state_old_base = loadPtr(side_ptr, ABIV3::side_material_state_old_base_off);
            s.material_state_work_base = loadPtr(side_ptr, ABIV3::side_material_state_work_base_off);
            s.material_state_stride_bytes = loadU64(side_ptr, ABIV3::side_material_state_stride_bytes_off);
            return s;
        };

	        const bool is_face_domain = (domain == IntegralDomain::InteriorFace || domain == IntegralDomain::InterfaceFace);
	        const std::optional<std::uint32_t> fixed_n_qpts_minus =
	            (specialization != nullptr) ? specialization->n_qpts_minus : std::optional<std::uint32_t>{};
	        const std::optional<std::uint32_t> fixed_n_test_dofs_minus =
	            (specialization != nullptr) ? specialization->n_test_dofs_minus : std::optional<std::uint32_t>{};
	        const std::optional<std::uint32_t> fixed_n_trial_dofs_minus =
	            (specialization != nullptr) ? specialization->n_trial_dofs_minus : std::optional<std::uint32_t>{};

	        const std::optional<std::uint32_t> fixed_n_qpts_plus =
	            (specialization != nullptr) ? specialization->n_qpts_plus : std::optional<std::uint32_t>{};
	        const std::optional<std::uint32_t> fixed_n_test_dofs_plus =
	            (specialization != nullptr) ? specialization->n_test_dofs_plus : std::optional<std::uint32_t>{};
	        const std::optional<std::uint32_t> fixed_n_trial_dofs_plus =
	            (specialization != nullptr) ? specialization->n_trial_dofs_plus : std::optional<std::uint32_t>{};

	        SideView side_single{};
	        SideView side_minus{};
	        SideView side_plus{};

        llvm::Value* out_single_ptr = nullptr;
        llvm::Value* out_minus_ptr = nullptr;
        llvm::Value* out_plus_ptr = nullptr;
        llvm::Value* out_coupling_mp_ptr = nullptr;
        llvm::Value* out_coupling_pm_ptr = nullptr;

        llvm::Value* element_matrix_single = nullptr;
        llvm::Value* element_vector_single = nullptr;

        if (!is_face_domain) {
            const std::size_t side_off = (domain == IntegralDomain::Cell) ? ABIV3::cell_side_off : ABIV3::bdry_side_off;
	            const std::size_t out_off = (domain == IntegralDomain::Cell) ? ABIV3::cell_out_off : ABIV3::bdry_out_off;

	            auto* side_ptr = gepBytes(args_ptr, side_off);
	            out_single_ptr = gepBytes(args_ptr, out_off);
	            side_single = loadSideView(side_ptr, fixed_n_qpts_minus, fixed_n_test_dofs_minus, fixed_n_trial_dofs_minus);

	            element_matrix_single = loadPtr(out_single_ptr, ABIV3::out_element_matrix_off);
	            element_vector_single = loadPtr(out_single_ptr, ABIV3::out_element_vector_off);
	        } else {
            auto* minus_ptr = gepBytes(args_ptr, ABIV3::face_minus_side_off);
            auto* plus_ptr = gepBytes(args_ptr, ABIV3::face_plus_side_off);

	            side_minus =
	                loadSideView(minus_ptr, fixed_n_qpts_minus, fixed_n_test_dofs_minus, fixed_n_trial_dofs_minus);
	            side_plus =
	                loadSideView(plus_ptr, fixed_n_qpts_plus, fixed_n_test_dofs_plus, fixed_n_trial_dofs_plus);

            out_minus_ptr = gepBytes(args_ptr, ABIV3::face_out_minus_off);
            out_plus_ptr = gepBytes(args_ptr, ABIV3::face_out_plus_off);
            out_coupling_mp_ptr = gepBytes(args_ptr, ABIV3::face_coupling_minus_plus_off);
            out_coupling_pm_ptr = gepBytes(args_ptr, ABIV3::face_coupling_plus_minus_off);
        }

        auto attachLoopUnrollMetadata = [&](llvm::BranchInst* backedge,
                                            llvm::Value* end) -> void {
            if (!options_.specialization.enable_loop_unroll_metadata) {
                return;
            }
            auto* c = llvm::dyn_cast<llvm::ConstantInt>(end);
            if (c == nullptr) {
                return;
            }

            const auto trip = c->getZExtValue();
            if (trip == 0u) {
                return;
            }

            const auto max_full = static_cast<std::uint64_t>(options_.specialization.max_unroll_trip_count);
            const auto max_partial = (max_full <= (std::numeric_limits<std::uint64_t>::max)() / 4u)
                                         ? (max_full * 4u)
                                         : max_full;

            llvm::SmallVector<llvm::Metadata*, 4> md_args;
            llvm::TempMDNode tmp = llvm::MDNode::getTemporary(*ctx, {});
            md_args.push_back(tmp.get());

            if (trip <= max_full) {
                md_args.push_back(llvm::MDNode::get(
                    *ctx, {llvm::MDString::get(*ctx, "llvm.loop.unroll.full")}));
            } else if (trip <= max_partial) {
                std::uint64_t count = (trip <= 64u) ? 4u : 8u;
                count = std::min(count, trip);
                if (count <= 1u) {
                    return;
                }
                md_args.push_back(llvm::MDNode::get(
                    *ctx,
                    {llvm::MDString::get(*ctx, "llvm.loop.unroll.count"),
                     llvm::ConstantAsMetadata::get(builder.getInt32(static_cast<std::uint32_t>(count)))}));
            } else {
                return;
            }

            auto* loop_id = llvm::MDNode::get(*ctx, md_args);
            loop_id->replaceOperandWith(0, loop_id);
            backedge->setMetadata("llvm.loop", loop_id);
        };

	        auto emitForLoop = [&](llvm::Value* end,
	                               std::string_view name,
	                               const auto& body_fn) -> void {
	            auto* pre = builder.GetInsertBlock();
	            auto* header = llvm::BasicBlock::Create(*ctx, std::string(name) + ".hdr", fn);
	            auto* body = llvm::BasicBlock::Create(*ctx, std::string(name) + ".body", fn);
	            auto* exit = llvm::BasicBlock::Create(*ctx, std::string(name) + ".exit", fn);

            builder.CreateBr(header);
            builder.SetInsertPoint(header);

            auto* idx_phi = builder.CreatePHI(i32, 2, std::string(name) + ".i");
            idx_phi->addIncoming(builder.getInt32(0), pre);

            auto* cond = builder.CreateICmpULT(idx_phi, end);
            builder.CreateCondBr(cond, body, exit);

	            builder.SetInsertPoint(body);
	            body_fn(idx_phi);

	            auto* body_end = builder.GetInsertBlock();
	            auto* next = builder.CreateAdd(idx_phi, builder.getInt32(1));
	            auto* backedge = builder.CreateBr(header);
	            attachLoopUnrollMetadata(backedge, end);
	            idx_phi->addIncoming(next, body_end);

	            builder.SetInsertPoint(exit);
	        };

        auto emitReduceSum = [&](llvm::Value* end,
                                 std::string_view name,
                                 std::size_t n_acc,
                                 const auto& term_fn) -> std::vector<llvm::Value*> {
            if (n_acc == 0u) {
                throw std::invalid_argument("LLVMGen: emitReduceSum requires n_acc > 0");
            }

            auto* preheader = builder.GetInsertBlock();
            auto* header = llvm::BasicBlock::Create(*ctx, std::string(name) + ".hdr", fn);
            auto* body = llvm::BasicBlock::Create(*ctx, std::string(name) + ".body", fn);
            auto* latch = llvm::BasicBlock::Create(*ctx, std::string(name) + ".latch", fn);
            auto* exit = llvm::BasicBlock::Create(*ctx, std::string(name) + ".exit", fn);

            builder.CreateBr(header);
            builder.SetInsertPoint(header);

            auto* idx_phi = builder.CreatePHI(i32, 2, std::string(name) + ".i");
            idx_phi->addIncoming(builder.getInt32(0), preheader);

            std::vector<llvm::PHINode*> acc_phi;
            acc_phi.reserve(n_acc);
            for (std::size_t k = 0; k < n_acc; ++k) {
                auto* a = builder.CreatePHI(f64, 2, std::string(name) + ".acc" + std::to_string(k));
                a->addIncoming(f64c(0.0), preheader);
                acc_phi.push_back(a);
            }

            auto* cond = builder.CreateICmpULT(idx_phi, end);
            builder.CreateCondBr(cond, body, exit);

            builder.SetInsertPoint(body);
            const auto terms = term_fn(idx_phi);
            if (terms.size() != n_acc) {
                throw std::invalid_argument("LLVMGen: emitReduceSum term_fn returned wrong arity");
            }
            builder.CreateBr(latch);

            builder.SetInsertPoint(latch);
            auto* idx_next = builder.CreateAdd(idx_phi, builder.getInt32(1));
            idx_phi->addIncoming(idx_next, latch);
	            for (std::size_t k = 0; k < n_acc; ++k) {
	                auto* next = builder.CreateFAdd(acc_phi[k], terms[k]);
	                acc_phi[k]->addIncoming(next, latch);
	            }
	            auto* backedge = builder.CreateBr(header);
	            attachLoopUnrollMetadata(backedge, end);

	            builder.SetInsertPoint(exit);
	            std::vector<llvm::Value*> out;
            out.reserve(n_acc);
            for (std::size_t k = 0; k < n_acc; ++k) {
                out.push_back(acc_phi[k]);
            }
            return out;
        };

        auto emitReduceSumScalar = [&](llvm::Value* end,
                                       std::string_view name,
                                       const auto& term_fn) -> llvm::Value* {
            return emitReduceSum(end, name, 1u, [&](llvm::Value* idx) {
                std::vector<llvm::Value*> terms;
                terms.reserve(1u);
                terms.push_back(term_fn(idx));
                return terms;
            })[0];
        };

        auto emitMatrixAccum = [&](llvm::Value* element_matrix_ptr,
                                   llvm::Value* n_trial_dofs,
                                   llvm::Value* i_idx,
                                   llvm::Value* j_idx,
                                   llvm::Value* contrib) -> void {
            auto* off_i = builder.CreateMul(i_idx, n_trial_dofs);
            auto* off = builder.CreateAdd(off_i, j_idx);
            auto* off64 = builder.CreateZExt(off, i64);
            auto* ptr = builder.CreateGEP(f64, element_matrix_ptr, off64);
            auto* old = builder.CreateLoad(f64, ptr);
            builder.CreateStore(builder.CreateFAdd(old, contrib), ptr);
        };

        auto emitVectorAccum = [&](llvm::Value* element_vector_ptr,
                                   llvm::Value* i_idx,
                                   llvm::Value* contrib) -> void {
            auto* off64 = builder.CreateZExt(i_idx, i64);
            auto* ptr = builder.CreateGEP(f64, element_vector_ptr, off64);
            auto* old = builder.CreateLoad(f64, ptr);
            builder.CreateStore(builder.CreateFAdd(old, contrib), ptr);
        };

        auto termWeight = [&](const SideView& side, int time_derivative_order) -> llvm::Value* {
            auto* td = side.time_derivative_term_weight;
            auto* non_td = side.non_time_derivative_term_weight;

            if (time_derivative_order > 0) {
                const int idx = time_derivative_order - 1;
                if (idx >= 0 && idx < static_cast<int>(assembly::jit::kMaxTimeDerivativeOrderV6)) {
                    auto* w = loadRealPtrAt(side.dt_term_weights_base, builder.getInt64(static_cast<std::uint64_t>(idx)));
                    return builder.CreateFMul(td, w);
                }
                return td;
            }

            return non_td;
        };

	        auto* fabs_fn = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::fabs, {f64});
	        auto* sqrt_fn = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::sqrt, {f64});
	        auto* exp_fn = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::exp, {f64});
	        auto* log_fn = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::log, {f64});
	        auto* pow_fn = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::pow, {f64});
	        auto* fmuladd_fn = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::fmuladd, {f64});

	        auto f_fabs = [&](llvm::Value* v) -> llvm::Value* { return builder.CreateCall(fabs_fn, {v}); };
	        auto f_sqrt = [&](llvm::Value* v) -> llvm::Value* { return builder.CreateCall(sqrt_fn, {v}); };
	        auto f_exp = [&](llvm::Value* v) -> llvm::Value* { return builder.CreateCall(exp_fn, {v}); };
	        auto f_log = [&](llvm::Value* v) -> llvm::Value* { return builder.CreateCall(log_fn, {v}); };
	        auto f_pow = [&](llvm::Value* a, llvm::Value* b) -> llvm::Value* { return builder.CreateCall(pow_fn, {a, b}); };
	        auto f_fmuladd = [&](llvm::Value* a, llvm::Value* b, llvm::Value* c) -> llvm::Value* {
	            return builder.CreateCall(fmuladd_fn, {a, b, c});
	        };

        auto f_min = [&](llvm::Value* a, llvm::Value* b) -> llvm::Value* {
            // Match std::min semantics: (b < a) ? b : a
            auto* pick_b = builder.CreateFCmpOLT(b, a);
            return builder.CreateSelect(pick_b, b, a);
        };
        auto f_max = [&](llvm::Value* a, llvm::Value* b) -> llvm::Value* {
            // Match std::max semantics: (a < b) ? b : a
            auto* pick_b = builder.CreateFCmpOLT(a, b);
            return builder.CreateSelect(pick_b, b, a);
        };

        struct CodeValue {
            Shape shape{};
            std::array<llvm::Value*, 81> elems{};
        };

	        auto elemCount = [&](const Shape& s) -> std::size_t {
	            switch (s.kind) {
	                case Shape::Kind::Scalar: return 1u;
	                case Shape::Kind::Vector: return static_cast<std::size_t>(s.dims[0]);
	                case Shape::Kind::Matrix: return static_cast<std::size_t>(s.dims[0] * s.dims[1]);
	                case Shape::Kind::Tensor3: return static_cast<std::size_t>(s.dims[0] * s.dims[1] * s.dims[2]);
	                case Shape::Kind::Tensor4:
	                    return static_cast<std::size_t>(s.dims[0] * s.dims[1] * s.dims[2] * s.dims[3]);
	            }
	            return 1u;
	        };

	        auto kernelIRUseCounts = [&](const KernelIR& kernel_ir) -> std::vector<std::uint32_t> {
	            std::vector<std::uint32_t> counts(kernel_ir.ops.size(), 0u);
	            for (std::size_t parent = 0; parent < kernel_ir.ops.size(); ++parent) {
	                const auto& op = kernel_ir.ops[parent];
	                for (std::size_t k = 0; k < static_cast<std::size_t>(op.child_count); ++k) {
	                    const auto child = static_cast<std::size_t>(
	                        kernel_ir.children[static_cast<std::size_t>(op.first_child) + k]);
	                    if (child < counts.size()) {
	                        counts[child] += 1u;
	                    }
	                }
	            }
	            return counts;
	        };

	        auto tryFuseMulAddScalar = [&](FormExprType parent_type,
	                                      const LoweredTerm& term,
	                                      const std::vector<std::uint32_t>& use_counts,
	                                      const KernelIROp& op,
	                                      const std::vector<CodeValue>& values) -> llvm::Value* {
	            if (options_.optimization_level < 2) {
	                return nullptr;
	            }
	            if (parent_type != FormExprType::Add && parent_type != FormExprType::Subtract) {
	                return nullptr;
	            }
	            if (op.child_count != 2u) {
	                return nullptr;
	            }

	            const auto lhs_idx = static_cast<std::size_t>(
	                term.ir.children[static_cast<std::size_t>(op.first_child)]);
	            const auto rhs_idx = static_cast<std::size_t>(
	                term.ir.children[static_cast<std::size_t>(op.first_child) + 1u]);
	            if (lhs_idx >= values.size() || rhs_idx >= values.size()) {
	                return nullptr;
	            }
	            if (term.shapes[lhs_idx].kind != Shape::Kind::Scalar || term.shapes[rhs_idx].kind != Shape::Kind::Scalar) {
	                return nullptr;
	            }

	            auto* lhs_val = values[lhs_idx].elems[0];
	            auto* rhs_val = values[rhs_idx].elems[0];
	            if (lhs_val == nullptr || rhs_val == nullptr) {
	                return nullptr;
	            }

	            auto mulOperands = [&](std::size_t mul_idx) -> std::optional<std::pair<llvm::Value*, llvm::Value*>> {
	                if (mul_idx >= term.ir.ops.size() || mul_idx >= use_counts.size()) {
	                    return std::nullopt;
	                }
	                if (term.ir.ops[mul_idx].type != FormExprType::Multiply) {
	                    return std::nullopt;
	                }
	                if (term.shapes[mul_idx].kind != Shape::Kind::Scalar) {
	                    return std::nullopt;
	                }
	                if (use_counts[mul_idx] != 1u) {
	                    return std::nullopt;
	                }

	                const auto& mop = term.ir.ops[mul_idx];
	                if (mop.child_count != 2u) {
	                    return std::nullopt;
	                }
	                const auto a_idx = static_cast<std::size_t>(
	                    term.ir.children[static_cast<std::size_t>(mop.first_child)]);
	                const auto b_idx = static_cast<std::size_t>(
	                    term.ir.children[static_cast<std::size_t>(mop.first_child) + 1u]);
	                if (a_idx >= values.size() || b_idx >= values.size()) {
	                    return std::nullopt;
	                }
	                if (term.shapes[a_idx].kind != Shape::Kind::Scalar || term.shapes[b_idx].kind != Shape::Kind::Scalar) {
	                    return std::nullopt;
	                }
	                auto* a = values[a_idx].elems[0];
	                auto* b = values[b_idx].elems[0];
	                if (a == nullptr || b == nullptr) {
	                    return std::nullopt;
	                }
	                return std::pair{a, b};
	            };

	            if (const auto mo = mulOperands(lhs_idx)) {
	                auto* c = (parent_type == FormExprType::Subtract) ? builder.CreateFNeg(rhs_val) : rhs_val;
	                return f_fmuladd(mo->first, mo->second, c);
	            }
	            if (const auto mo = mulOperands(rhs_idx)) {
	                if (parent_type == FormExprType::Add) {
	                    return f_fmuladd(mo->first, mo->second, lhs_val);
	                }
	                auto* bneg = builder.CreateFNeg(mo->second);
	                return f_fmuladd(mo->first, bneg, lhs_val);
	            }

	            return nullptr;
	        };

        auto externalKindU32 = [&](const Shape& s) -> std::uint32_t {
            switch (s.kind) {
                case Shape::Kind::Scalar:
                    return static_cast<std::uint32_t>(external::ValueKindV1::Scalar);
                case Shape::Kind::Vector:
                    return static_cast<std::uint32_t>(external::ValueKindV1::Vector);
                case Shape::Kind::Matrix:
                    return static_cast<std::uint32_t>(external::ValueKindV1::Matrix);
                case Shape::Kind::Tensor3:
                    return static_cast<std::uint32_t>(external::ValueKindV1::Tensor3);
                case Shape::Kind::Tensor4:
                    return static_cast<std::uint32_t>(external::ValueKindV1::Tensor4);
            }
            return static_cast<std::uint32_t>(external::ValueKindV1::Scalar);
        };

        auto makeZero = [&](const Shape& s) -> CodeValue {
            CodeValue out;
            out.shape = s;
            out.elems.fill(nullptr);
            const auto n = elemCount(s);
            for (std::size_t i = 0; i < n; ++i) {
                out.elems[i] = f64c(0.0);
            }
            return out;
        };

        auto makeScalar = [&](llvm::Value* v) -> CodeValue {
            CodeValue out;
            out.shape = scalarShape();
            out.elems.fill(nullptr);
            out.elems[0] = v;
            return out;
        };

        auto makeVector = [&](std::uint32_t n, llvm::Value* x, llvm::Value* y, llvm::Value* z) -> CodeValue {
            CodeValue out;
            out.shape = vectorShape(n);
            out.elems.fill(nullptr);
            out.elems[0] = x;
            out.elems[1] = (n > 1u) ? y : f64c(0.0);
            out.elems[2] = (n > 2u) ? z : f64c(0.0);
            return out;
        };

        auto makeMatrix = [&](std::uint32_t rows, std::uint32_t cols) -> CodeValue {
            return makeZero(matrixShape(rows, cols));
        };

        auto allocaInEntry = [&](llvm::Type* ty, llvm::Value* count, std::string_view name) -> llvm::AllocaInst* {
            llvm::IRBuilder<> ab(&fn->getEntryBlock(), fn->getEntryBlock().begin());
            return ab.CreateAlloca(ty, count, std::string(name));
        };

        auto nodePtrFromImm0 = [&](std::uint64_t imm0) -> llvm::Value* {
            auto* addr = llvm::ConstantInt::get(i64, imm0);
            return builder.CreateIntToPtr(addr, i8_ptr);
        };

        auto evalExternalCoefficient = [&](const SideView& side,
                                           llvm::Value* q_index,
                                           const Shape& out_shape,
                                           std::uint64_t coeff_node_ptr) -> CodeValue {
            const auto out_kind = out_shape.kind;
            auto* node_ptr = nodePtrFromImm0(coeff_node_ptr);

            if (out_kind == Shape::Kind::Scalar) {
                auto* v = builder.CreateCall(coeff_eval_scalar_fn, {side.side_ptr, q_index, node_ptr});
                return makeScalar(v);
            }

            const auto n = elemCount(out_shape);
            const std::size_t fixed_len = [&]() -> std::size_t {
                switch (out_kind) {
                    case Shape::Kind::Vector: return 3u;
                    case Shape::Kind::Matrix: return 9u;
                    case Shape::Kind::Tensor3: return 27u;
                    case Shape::Kind::Tensor4: return 81u;
                    case Shape::Kind::Scalar: return 1u;
                }
                return 1u;
            }();

            auto* buf = allocaInEntry(f64,
                                      builder.getInt32(static_cast<std::uint32_t>(fixed_len)),
                                      "coeff.buf");

            llvm::FunctionCallee callee = [&]() {
                switch (out_kind) {
                    case Shape::Kind::Vector:
                        return coeff_eval_vector_fn;
                    case Shape::Kind::Matrix:
                        return coeff_eval_matrix_fn;
                    case Shape::Kind::Tensor3:
                        return coeff_eval_tensor3_fn;
                    case Shape::Kind::Tensor4:
                        return coeff_eval_tensor4_fn;
                    case Shape::Kind::Scalar:
                        break;
                }
                return coeff_eval_vector_fn;
            }();

            builder.CreateCall(callee, {side.side_ptr, q_index, node_ptr, buf});

            CodeValue out = makeZero(out_shape);
            for (std::size_t i = 0; i < n; ++i) {
                auto* p = builder.CreateGEP(f64, buf, builder.getInt64(i));
                out.elems[i] = builder.CreateLoad(f64, p);
            }
            return out;
        };

        auto evalExternalConstitutiveOutput = [&](const SideView& side,
                                                  llvm::Value* q_index,
                                                  llvm::Value* trace_side,
                                                  std::uint64_t constitutive_node_ptr,
                                                  std::uint32_t output_index,
                                                  const KernelIR& ir,
                                                  const KernelIROp& call_op,
                                                  const std::vector<CodeValue>& values,
                                                  const Shape& out_shape) -> CodeValue {
            const std::size_t num_inputs = static_cast<std::size_t>(call_op.child_count);
            auto* node_ptr = nodePtrFromImm0(constitutive_node_ptr);

            auto* views = allocaInEntry(value_view_ty,
                                        builder.getInt32(static_cast<std::uint32_t>(num_inputs)),
                                        "c.inputs");

            for (std::size_t i = 0; i < num_inputs; ++i) {
                const auto child_idx =
                    ir.children[static_cast<std::size_t>(call_op.first_child) + i];
                const auto& in = values[child_idx];
                const auto in_len = static_cast<std::uint32_t>(elemCount(in.shape));

                auto* in_buf = allocaInEntry(f64,
                                             builder.getInt32(in_len),
                                             "c.inbuf");
                for (std::size_t k = 0; k < static_cast<std::size_t>(in_len); ++k) {
                    auto* p = builder.CreateGEP(f64, in_buf, builder.getInt64(k));
                    builder.CreateStore(in.elems[k], p);
                }

                auto* view_ptr = builder.CreateGEP(value_view_ty, views, builder.getInt64(i));
                builder.CreateStore(builder.getInt32(externalKindU32(in.shape)),
                                    builder.CreateStructGEP(value_view_ty, view_ptr, 0));
                builder.CreateStore(builder.getInt32(in_len),
                                    builder.CreateStructGEP(value_view_ty, view_ptr, 1));
                builder.CreateStore(in_buf,
                                    builder.CreateStructGEP(value_view_ty, view_ptr, 2));
            }

            const auto out_len = static_cast<std::uint32_t>(elemCount(out_shape));
            auto* out_buf = allocaInEntry(f64,
                                          builder.getInt32(out_len),
                                          "c.outbuf");

            builder.CreateCall(constitutive_eval_fn,
                               {side.side_ptr,
                                q_index,
                                trace_side,
                                node_ptr,
                                builder.getInt32(output_index),
                                views,
                                builder.getInt32(static_cast<std::uint32_t>(num_inputs)),
                                builder.getInt32(externalKindU32(out_shape)),
                                out_buf,
                                builder.getInt32(out_len)});

            CodeValue out = makeZero(out_shape);
            for (std::size_t k = 0; k < static_cast<std::size_t>(out_len); ++k) {
                auto* p = builder.CreateGEP(f64, out_buf, builder.getInt64(k));
                out.elems[k] = builder.CreateLoad(f64, p);
            }
            return out;
        };

        auto getVecComp = [&](const CodeValue& v, std::size_t i) -> llvm::Value* {
            if (v.shape.kind != Shape::Kind::Vector) {
                throw std::runtime_error("LLVMGen: expected vector");
            }
            if (i >= static_cast<std::size_t>(v.shape.dims[0])) {
                return f64c(0.0);
            }
            return v.elems[i];
        };

        auto matIndex = [&](const CodeValue& m, std::size_t r, std::size_t c) -> std::size_t {
            if (m.shape.kind != Shape::Kind::Matrix) {
                throw std::runtime_error("LLVMGen: expected matrix");
            }
            return r * static_cast<std::size_t>(m.shape.dims[1]) + c;
        };

        auto add = [&](const CodeValue& a, const CodeValue& b) -> CodeValue {
            if (a.shape.kind != b.shape.kind || a.shape.dims != b.shape.dims) {
                throw std::runtime_error("LLVMGen: Add shape mismatch");
            }
            CodeValue out;
            out.shape = a.shape;
            out.elems.fill(nullptr);
            const auto n = elemCount(a.shape);
            for (std::size_t i = 0; i < n; ++i) {
                out.elems[i] = builder.CreateFAdd(a.elems[i], b.elems[i]);
            }
            return out;
        };

        auto sub = [&](const CodeValue& a, const CodeValue& b) -> CodeValue {
            if (a.shape.kind != b.shape.kind || a.shape.dims != b.shape.dims) {
                throw std::runtime_error("LLVMGen: Subtract shape mismatch");
            }
            CodeValue out;
            out.shape = a.shape;
            out.elems.fill(nullptr);
            const auto n = elemCount(a.shape);
            for (std::size_t i = 0; i < n; ++i) {
                out.elems[i] = builder.CreateFSub(a.elems[i], b.elems[i]);
            }
            return out;
        };

        auto neg = [&](const CodeValue& a) -> CodeValue {
            CodeValue out;
            out.shape = a.shape;
            out.elems.fill(nullptr);
            const auto n = elemCount(a.shape);
            for (std::size_t i = 0; i < n; ++i) {
                out.elems[i] = builder.CreateFNeg(a.elems[i]);
            }
            return out;
        };

        auto mul = [&](const CodeValue& a, const CodeValue& b) -> CodeValue {
            if (a.shape.kind == Shape::Kind::Scalar && b.shape.kind == Shape::Kind::Scalar) {
                return makeScalar(builder.CreateFMul(a.elems[0], b.elems[0]));
            }
            if (a.shape.kind == Shape::Kind::Scalar) {
                CodeValue out;
                out.shape = b.shape;
                out.elems.fill(nullptr);
                const auto n = elemCount(b.shape);
                for (std::size_t i = 0; i < n; ++i) {
                    out.elems[i] = builder.CreateFMul(a.elems[0], b.elems[i]);
                }
                return out;
            }
            if (b.shape.kind == Shape::Kind::Scalar) {
                CodeValue out;
                out.shape = a.shape;
                out.elems.fill(nullptr);
                const auto n = elemCount(a.shape);
                for (std::size_t i = 0; i < n; ++i) {
                    out.elems[i] = builder.CreateFMul(a.elems[i], b.elems[0]);
                }
                return out;
            }

            if (a.shape.kind == Shape::Kind::Matrix && b.shape.kind == Shape::Kind::Vector) {
                const auto rows = static_cast<std::size_t>(a.shape.dims[0]);
                const auto cols = static_cast<std::size_t>(a.shape.dims[1]);
                if (cols != static_cast<std::size_t>(b.shape.dims[0])) {
                    throw std::runtime_error("LLVMGen: matrix-vector Multiply shape mismatch");
                }
                CodeValue out;
                out.shape = vectorShape(static_cast<std::uint32_t>(rows));
                out.elems.fill(nullptr);
                for (std::size_t r = 0; r < rows; ++r) {
                    llvm::Value* sumv = f64c(0.0);
                    for (std::size_t c = 0; c < cols; ++c) {
                        const auto aidx = matIndex(a, r, c);
                        sumv = builder.CreateFAdd(sumv, builder.CreateFMul(a.elems[aidx], getVecComp(b, c)));
                    }
                    out.elems[r] = sumv;
                }
                return out;
            }

            if (a.shape.kind == Shape::Kind::Vector && b.shape.kind == Shape::Kind::Matrix) {
                const auto rows = static_cast<std::size_t>(b.shape.dims[0]);
                const auto cols = static_cast<std::size_t>(b.shape.dims[1]);
                if (static_cast<std::size_t>(a.shape.dims[0]) != rows) {
                    throw std::runtime_error("LLVMGen: vector-matrix Multiply shape mismatch");
                }
                CodeValue out;
                out.shape = vectorShape(static_cast<std::uint32_t>(cols));
                out.elems.fill(nullptr);
                for (std::size_t c = 0; c < cols; ++c) {
                    llvm::Value* sumv = f64c(0.0);
                    for (std::size_t r = 0; r < rows; ++r) {
                        const auto bidx = matIndex(b, r, c);
                        sumv = builder.CreateFAdd(sumv, builder.CreateFMul(getVecComp(a, r), b.elems[bidx]));
                    }
                    out.elems[c] = sumv;
                }
                return out;
            }

            if (a.shape.kind == Shape::Kind::Matrix && b.shape.kind == Shape::Kind::Matrix) {
                const auto a_rows = static_cast<std::size_t>(a.shape.dims[0]);
                const auto a_cols = static_cast<std::size_t>(a.shape.dims[1]);
                const auto b_rows = static_cast<std::size_t>(b.shape.dims[0]);
                const auto b_cols = static_cast<std::size_t>(b.shape.dims[1]);
                if (a_cols != b_rows) {
                    throw std::runtime_error("LLVMGen: matrix-matrix Multiply shape mismatch");
                }
                CodeValue out;
                out.shape = matrixShape(static_cast<std::uint32_t>(a_rows), static_cast<std::uint32_t>(b_cols));
                out.elems.fill(nullptr);
                for (std::size_t r = 0; r < a_rows; ++r) {
                    for (std::size_t c = 0; c < b_cols; ++c) {
                        llvm::Value* sumv = f64c(0.0);
                        for (std::size_t k = 0; k < a_cols; ++k) {
                            const auto aidx = matIndex(a, r, k);
                            const auto bidx = matIndex(b, k, c);
                            sumv = builder.CreateFAdd(sumv, builder.CreateFMul(a.elems[aidx], b.elems[bidx]));
                        }
                        out.elems[r * b_cols + c] = sumv;
                    }
                }
                return out;
            }

            throw std::runtime_error("LLVMGen: Multiply operand shapes not supported");
        };

        auto div = [&](const CodeValue& a, const CodeValue& b) -> CodeValue {
            if (b.shape.kind != Shape::Kind::Scalar) {
                throw std::runtime_error("LLVMGen: Divide requires scalar denominator");
            }
            if (a.shape.kind == Shape::Kind::Scalar) {
                return makeScalar(builder.CreateFDiv(a.elems[0], b.elems[0]));
            }
            CodeValue out;
            out.shape = a.shape;
            out.elems.fill(nullptr);
            const auto n = elemCount(a.shape);
            for (std::size_t i = 0; i < n; ++i) {
                out.elems[i] = builder.CreateFDiv(a.elems[i], b.elems[0]);
            }
            return out;
        };

        auto inner = [&](const CodeValue& a, const CodeValue& b) -> CodeValue {
            if (a.shape.kind != b.shape.kind || a.shape.dims != b.shape.dims) {
                throw std::runtime_error("LLVMGen: InnerProduct requires matching shapes");
            }
            llvm::Value* sumv = f64c(0.0);
            const auto n = elemCount(a.shape);
            for (std::size_t i = 0; i < n; ++i) {
                sumv = builder.CreateFAdd(sumv, builder.CreateFMul(a.elems[i], b.elems[i]));
            }
            return makeScalar(sumv);
        };

        auto doubleContraction = [&](const CodeValue& a, const CodeValue& b) -> CodeValue {
            if (a.shape.kind == Shape::Kind::Tensor4 && b.shape.kind == Shape::Kind::Matrix) {
                const auto dim = static_cast<std::size_t>(b.shape.dims[0]);
                CodeValue out;
                out.shape = matrixShape(static_cast<std::uint32_t>(dim), static_cast<std::uint32_t>(dim));
                out.elems.fill(nullptr);
                for (std::size_t i = 0; i < dim; ++i) {
                    for (std::size_t j = 0; j < dim; ++j) {
                        llvm::Value* sumv = f64c(0.0);
                        for (std::size_t k = 0; k < dim; ++k) {
                            for (std::size_t l = 0; l < dim; ++l) {
                                const std::size_t idx4 = ((i * 3 + j) * 3 + k) * 3 + l;
                                const auto midx = matIndex(b, k, l);
                                sumv = builder.CreateFAdd(sumv, builder.CreateFMul(a.elems[idx4], b.elems[midx]));
                            }
                        }
                        out.elems[i * dim + j] = sumv;
                    }
                }
                return out;
            }
            if (a.shape.kind == Shape::Kind::Matrix && b.shape.kind == Shape::Kind::Tensor4) {
                const auto dim = static_cast<std::size_t>(a.shape.dims[0]);
                CodeValue out;
                out.shape = matrixShape(static_cast<std::uint32_t>(dim), static_cast<std::uint32_t>(dim));
                out.elems.fill(nullptr);
                for (std::size_t k = 0; k < dim; ++k) {
                    for (std::size_t l = 0; l < dim; ++l) {
                        llvm::Value* sumv = f64c(0.0);
                        for (std::size_t i = 0; i < dim; ++i) {
                            for (std::size_t j = 0; j < dim; ++j) {
                                const auto midx = matIndex(a, i, j);
                                const std::size_t idx4 = ((i * 3 + j) * 3 + k) * 3 + l;
                                sumv = builder.CreateFAdd(sumv, builder.CreateFMul(a.elems[midx], b.elems[idx4]));
                            }
                        }
                        out.elems[k * dim + l] = sumv;
                    }
                }
                return out;
            }
            return inner(a, b);
        };

        auto outer = [&](const CodeValue& a, const CodeValue& b) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Vector || b.shape.kind != Shape::Kind::Vector) {
                throw std::runtime_error("LLVMGen: OuterProduct expects vector-vector");
            }
            const auto rows = static_cast<std::size_t>(a.shape.dims[0]);
            const auto cols = static_cast<std::size_t>(b.shape.dims[0]);
            CodeValue out;
            out.shape = matrixShape(static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols));
            out.elems.fill(nullptr);
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    out.elems[r * cols + c] = builder.CreateFMul(getVecComp(a, r), getVecComp(b, c));
                }
            }
            return out;
        };

        auto cross = [&](const CodeValue& a, const CodeValue& b) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Vector || b.shape.kind != Shape::Kind::Vector) {
                throw std::runtime_error("LLVMGen: CrossProduct expects vector arguments");
            }
            const auto ax = getVecComp(a, 0);
            const auto ay = getVecComp(a, 1);
            const auto az = getVecComp(a, 2);
            const auto bx = getVecComp(b, 0);
            const auto by = getVecComp(b, 1);
            const auto bz = getVecComp(b, 2);
            auto* cx = builder.CreateFSub(builder.CreateFMul(ay, bz), builder.CreateFMul(az, by));
            auto* cy = builder.CreateFSub(builder.CreateFMul(az, bx), builder.CreateFMul(ax, bz));
            auto* cz = builder.CreateFSub(builder.CreateFMul(ax, by), builder.CreateFMul(ay, bx));
            return makeVector(3u, cx, cy, cz);
        };

        auto powv = [&](const CodeValue& a, const CodeValue& b) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Scalar || b.shape.kind != Shape::Kind::Scalar) {
                throw std::runtime_error("LLVMGen: Power expects scalar arguments");
            }
            return makeScalar(f_pow(a.elems[0], b.elems[0]));
        };

        auto cmpToScalar01 = [&](llvm::Value* cmp) -> CodeValue {
            return makeScalar(builder.CreateSelect(cmp, f64c(1.0), f64c(0.0)));
        };

        auto cmp = [&](FormExprType t, const CodeValue& a, const CodeValue& b) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Scalar || b.shape.kind != Shape::Kind::Scalar) {
                throw std::runtime_error("LLVMGen: comparisons expect scalar operands");
            }
            llvm::Value* pred = nullptr;
            switch (t) {
                case FormExprType::Less: pred = builder.CreateFCmpOLT(a.elems[0], b.elems[0]); break;
                case FormExprType::LessEqual: pred = builder.CreateFCmpOLE(a.elems[0], b.elems[0]); break;
                case FormExprType::Greater: pred = builder.CreateFCmpOGT(a.elems[0], b.elems[0]); break;
                case FormExprType::GreaterEqual: pred = builder.CreateFCmpOGE(a.elems[0], b.elems[0]); break;
                case FormExprType::Equal: pred = builder.CreateFCmpOEQ(a.elems[0], b.elems[0]); break;
                case FormExprType::NotEqual: pred = builder.CreateFCmpUNE(a.elems[0], b.elems[0]); break;
                default: throw std::runtime_error("LLVMGen: unexpected comparison op");
            }
            return cmpToScalar01(pred);
        };

        auto conditional = [&](const CodeValue& cond, const CodeValue& a, const CodeValue& b) -> CodeValue {
            if (cond.shape.kind != Shape::Kind::Scalar) {
                throw std::runtime_error("LLVMGen: Conditional condition must be scalar");
            }
            if (a.shape.kind != b.shape.kind || a.shape.dims != b.shape.dims) {
                throw std::runtime_error("LLVMGen: Conditional branch shape mismatch");
            }
            auto* pick_a = builder.CreateFCmpOGT(cond.elems[0], f64c(0.0));
            CodeValue out;
            out.shape = a.shape;
            out.elems.fill(nullptr);
            const auto n = elemCount(a.shape);
            for (std::size_t i = 0; i < n; ++i) {
                out.elems[i] = builder.CreateSelect(pick_a, a.elems[i], b.elems[i]);
            }
            return out;
        };

        auto transpose = [&](const CodeValue& a) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix) {
                throw std::runtime_error("LLVMGen: Transpose expects matrix");
            }
            const auto rows = static_cast<std::size_t>(a.shape.dims[0]);
            const auto cols = static_cast<std::size_t>(a.shape.dims[1]);
            CodeValue out;
            out.shape = matrixShape(static_cast<std::uint32_t>(cols), static_cast<std::uint32_t>(rows));
            out.elems.fill(nullptr);
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    const auto aidx = matIndex(a, r, c);
                    out.elems[c * rows + r] = a.elems[aidx];
                }
            }
            return out;
        };

        auto trace = [&](const CodeValue& a) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || a.shape.dims[0] != a.shape.dims[1] || a.shape.dims[0] == 0u) {
                throw std::runtime_error("LLVMGen: Trace expects square matrix");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            llvm::Value* tr = f64c(0.0);
            for (std::size_t d = 0; d < n; ++d) {
                tr = builder.CreateFAdd(tr, a.elems[matIndex(a, d, d)]);
            }
            return makeScalar(tr);
        };

        auto det = [&](const CodeValue& a) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || a.shape.dims[0] != a.shape.dims[1] || a.shape.dims[0] == 0u) {
                throw std::runtime_error("LLVMGen: Determinant expects square matrix");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            if (n == 1u) {
                return makeScalar(a.elems[0]);
            }
            if (n == 2u) {
                auto* a00 = a.elems[matIndex(a, 0, 0)];
                auto* a01 = a.elems[matIndex(a, 0, 1)];
                auto* a10 = a.elems[matIndex(a, 1, 0)];
                auto* a11 = a.elems[matIndex(a, 1, 1)];
                return makeScalar(builder.CreateFSub(builder.CreateFMul(a00, a11), builder.CreateFMul(a01, a10)));
            }
            if (n == 3u) {
                auto* a00 = a.elems[matIndex(a, 0, 0)];
                auto* a01 = a.elems[matIndex(a, 0, 1)];
                auto* a02 = a.elems[matIndex(a, 0, 2)];
                auto* a10 = a.elems[matIndex(a, 1, 0)];
                auto* a11 = a.elems[matIndex(a, 1, 1)];
                auto* a12 = a.elems[matIndex(a, 1, 2)];
                auto* a20 = a.elems[matIndex(a, 2, 0)];
                auto* a21 = a.elems[matIndex(a, 2, 1)];
                auto* a22 = a.elems[matIndex(a, 2, 2)];

                auto* t0 = builder.CreateFSub(builder.CreateFMul(a11, a22), builder.CreateFMul(a12, a21));
                auto* t1 = builder.CreateFSub(builder.CreateFMul(a10, a22), builder.CreateFMul(a12, a20));
                auto* t2 = builder.CreateFSub(builder.CreateFMul(a10, a21), builder.CreateFMul(a11, a20));

                auto* p0 = builder.CreateFMul(a00, t0);
                auto* p1 = builder.CreateFMul(a01, t1);
                auto* p2 = builder.CreateFMul(a02, t2);
                return makeScalar(builder.CreateFAdd(builder.CreateFSub(p0, p1), p2));
            }
            throw std::runtime_error("LLVMGen: Determinant supports up to 3x3");
        };

        auto cofactor = [&](const CodeValue& a) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || a.shape.dims[0] != a.shape.dims[1] || a.shape.dims[0] == 0u) {
                throw std::runtime_error("LLVMGen: Cofactor expects square matrix");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            if (n == 1u) {
                CodeValue out = makeMatrix(1u, 1u);
                out.elems[0] = f64c(1.0);
                return out;
            }
            if (n == 2u) {
                CodeValue cof = makeMatrix(2u, 2u);
                cof.elems[matIndex(cof, 0, 0)] = a.elems[matIndex(a, 1, 1)];
                cof.elems[matIndex(cof, 0, 1)] = builder.CreateFNeg(a.elems[matIndex(a, 1, 0)]);
                cof.elems[matIndex(cof, 1, 0)] = builder.CreateFNeg(a.elems[matIndex(a, 0, 1)]);
                cof.elems[matIndex(cof, 1, 1)] = a.elems[matIndex(a, 0, 0)];
                return cof;
            }
            if (n == 3u) {
                CodeValue cof = makeMatrix(3u, 3u);
                auto A = [&](std::size_t r, std::size_t c) -> llvm::Value* { return a.elems[matIndex(a, r, c)]; };

                cof.elems[matIndex(cof, 0, 0)] = builder.CreateFSub(builder.CreateFMul(A(1, 1), A(2, 2)), builder.CreateFMul(A(1, 2), A(2, 1)));
                cof.elems[matIndex(cof, 0, 1)] = builder.CreateFNeg(builder.CreateFSub(builder.CreateFMul(A(1, 0), A(2, 2)), builder.CreateFMul(A(1, 2), A(2, 0))));
                cof.elems[matIndex(cof, 0, 2)] = builder.CreateFSub(builder.CreateFMul(A(1, 0), A(2, 1)), builder.CreateFMul(A(1, 1), A(2, 0)));

                cof.elems[matIndex(cof, 1, 0)] = builder.CreateFNeg(builder.CreateFSub(builder.CreateFMul(A(0, 1), A(2, 2)), builder.CreateFMul(A(0, 2), A(2, 1))));
                cof.elems[matIndex(cof, 1, 1)] = builder.CreateFSub(builder.CreateFMul(A(0, 0), A(2, 2)), builder.CreateFMul(A(0, 2), A(2, 0)));
                cof.elems[matIndex(cof, 1, 2)] = builder.CreateFNeg(builder.CreateFSub(builder.CreateFMul(A(0, 0), A(2, 1)), builder.CreateFMul(A(0, 1), A(2, 0))));

                cof.elems[matIndex(cof, 2, 0)] = builder.CreateFSub(builder.CreateFMul(A(0, 1), A(1, 2)), builder.CreateFMul(A(0, 2), A(1, 1)));
                cof.elems[matIndex(cof, 2, 1)] = builder.CreateFNeg(builder.CreateFSub(builder.CreateFMul(A(0, 0), A(1, 2)), builder.CreateFMul(A(0, 2), A(1, 0))));
                cof.elems[matIndex(cof, 2, 2)] = builder.CreateFSub(builder.CreateFMul(A(0, 0), A(1, 1)), builder.CreateFMul(A(0, 1), A(1, 0)));
                return cof;
            }
            throw std::runtime_error("LLVMGen: Cofactor supports up to 3x3");
        };

        auto inv = [&](const CodeValue& a) -> CodeValue {
            const auto detA = det(a).elems[0];
            const auto cof = cofactor(a);
            const auto cofT = transpose(cof);
            CodeValue out;
            out.shape = a.shape;
            out.elems.fill(nullptr);
            const auto n = elemCount(a.shape);
            for (std::size_t i = 0; i < n; ++i) {
                out.elems[i] = builder.CreateFDiv(cofT.elems[i], detA);
            }
            return out;
        };

        auto storeMatrixToStack = [&](const CodeValue& a) -> llvm::Value* {
            const auto n = elemCount(a.shape);
            auto* arr_ty = llvm::ArrayType::get(f64, n);
            auto* alloca = builder.CreateAlloca(arr_ty);
            for (std::size_t i = 0; i < n; ++i) {
                auto* gep = builder.CreateInBoundsGEP(arr_ty,
                                                     alloca,
                                                     {builder.getInt32(0), builder.getInt32(static_cast<std::uint32_t>(i))});
                builder.CreateStore(a.elems[i], gep);
            }
            return builder.CreatePointerCast(alloca, f64_ptr);
        };

        auto callMatrixUnary = [&](const CodeValue& a,
                                   llvm::FunctionCallee fn2x2,
                                   llvm::FunctionCallee fn3x3) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || a.shape.dims[0] != a.shape.dims[1]) {
                throw std::runtime_error("LLVMGen: matrix function expects square matrix");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            if (n != 2u && n != 3u) {
                throw std::runtime_error("LLVMGen: matrix function supports only 2x2 and 3x3 matrices");
            }

            auto* aptr = storeMatrixToStack(a);
            auto* out_arr_ty = llvm::ArrayType::get(f64, static_cast<std::uint32_t>(n * n));
            auto* out_alloca = builder.CreateAlloca(out_arr_ty);
            auto* out_ptr = builder.CreatePointerCast(out_alloca, f64_ptr);

            auto callee = (n == 2u) ? fn2x2 : fn3x3;
            builder.CreateCall(callee, {aptr, out_ptr});

            CodeValue out = makeMatrix(static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(n));
            for (std::size_t i = 0; i < n * n; ++i) {
                out.elems[i] = loadRealPtrAt(out_ptr, builder.getInt64(i));
            }
            return out;
        };

	        auto callMatrixPow = [&](const CodeValue& a,
	                                 llvm::Value* p,
	                                 llvm::FunctionCallee fn2x2,
	                                 llvm::FunctionCallee fn3x3) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || a.shape.dims[0] != a.shape.dims[1]) {
                throw std::runtime_error("LLVMGen: matrix pow expects square matrix");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            if (n != 2u && n != 3u) {
                throw std::runtime_error("LLVMGen: matrix pow supports only 2x2 and 3x3 matrices");
            }

            auto* aptr = storeMatrixToStack(a);
            auto* out_arr_ty = llvm::ArrayType::get(f64, static_cast<std::uint32_t>(n * n));
            auto* out_alloca = builder.CreateAlloca(out_arr_ty);
            auto* out_ptr = builder.CreatePointerCast(out_alloca, f64_ptr);

            auto callee = (n == 2u) ? fn2x2 : fn3x3;
            builder.CreateCall(callee, {aptr, p, out_ptr});

            CodeValue out = makeMatrix(static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(n));
            for (std::size_t i = 0; i < n * n; ++i) {
                out.elems[i] = loadRealPtrAt(out_ptr, builder.getInt64(i));
            }
	            return out;
	        };

	        auto emitMatrixExp = [&](const CodeValue& a) -> CodeValue {
	            return callMatrixUnary(a, mat_exp_2x2_fn, mat_exp_3x3_fn);
	        };

	        auto emitMatrixLog = [&](const CodeValue& a) -> CodeValue {
	            return callMatrixUnary(a, mat_log_2x2_fn, mat_log_3x3_fn);
	        };

	        auto emitMatrixSqrt = [&](const CodeValue& a) -> CodeValue {
	            return callMatrixUnary(a, mat_sqrt_2x2_fn, mat_sqrt_3x3_fn);
	        };

	        auto emitMatrixPow = [&](const CodeValue& a, llvm::Value* p) -> CodeValue {
	            return callMatrixPow(a, p, mat_pow_2x2_fn, mat_pow_3x3_fn);
	        };

	        auto callMatrixUnaryDD = [&](const CodeValue& a,
	                                     const CodeValue& da,
	                                     llvm::FunctionCallee fn2x2,
	                                     llvm::FunctionCallee fn3x3) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || da.shape.kind != Shape::Kind::Matrix ||
                a.shape.dims[0] != a.shape.dims[1] || da.shape.dims[0] != da.shape.dims[1] ||
                a.shape.dims[0] != da.shape.dims[0]) {
                throw std::runtime_error("LLVMGen: matrix function directional derivative expects matching square matrices");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            if (n != 2u && n != 3u) {
                throw std::runtime_error("LLVMGen: matrix function directional derivative supports only 2x2 and 3x3 matrices");
            }

            auto* aptr = storeMatrixToStack(a);
            auto* daptr = storeMatrixToStack(da);
            auto* out_arr_ty = llvm::ArrayType::get(f64, static_cast<std::uint32_t>(n * n));
            auto* out_alloca = builder.CreateAlloca(out_arr_ty);
            auto* out_ptr = builder.CreatePointerCast(out_alloca, f64_ptr);

            auto callee = (n == 2u) ? fn2x2 : fn3x3;
            builder.CreateCall(callee, {aptr, daptr, out_ptr});

            CodeValue out = makeMatrix(static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(n));
            for (std::size_t i = 0; i < n * n; ++i) {
                out.elems[i] = loadRealPtrAt(out_ptr, builder.getInt64(i));
            }
            return out;
        };

        auto callMatrixPowDD = [&](const CodeValue& a,
                                   const CodeValue& da,
                                   llvm::Value* p,
                                   llvm::FunctionCallee fn2x2,
                                   llvm::FunctionCallee fn3x3) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || da.shape.kind != Shape::Kind::Matrix ||
                a.shape.dims[0] != a.shape.dims[1] || da.shape.dims[0] != da.shape.dims[1] ||
                a.shape.dims[0] != da.shape.dims[0]) {
                throw std::runtime_error("LLVMGen: matrix pow directional derivative expects matching square matrices");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            if (n != 2u && n != 3u) {
                throw std::runtime_error("LLVMGen: matrix pow directional derivative supports only 2x2 and 3x3 matrices");
            }

            auto* aptr = storeMatrixToStack(a);
            auto* daptr = storeMatrixToStack(da);
            auto* out_arr_ty = llvm::ArrayType::get(f64, static_cast<std::uint32_t>(n * n));
            auto* out_alloca = builder.CreateAlloca(out_arr_ty);
            auto* out_ptr = builder.CreatePointerCast(out_alloca, f64_ptr);

            auto callee = (n == 2u) ? fn2x2 : fn3x3;
            builder.CreateCall(callee, {aptr, daptr, p, out_ptr});

            CodeValue out = makeMatrix(static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(n));
            for (std::size_t i = 0; i < n * n; ++i) {
                out.elems[i] = loadRealPtrAt(out_ptr, builder.getInt64(i));
            }
            return out;
        };

        auto eigSymVecDD = [&](const CodeValue& a, const CodeValue& da, std::uint32_t which) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || da.shape.kind != Shape::Kind::Matrix ||
                a.shape.dims[0] != a.shape.dims[1] || da.shape.dims[0] != da.shape.dims[1] ||
                a.shape.dims[0] != da.shape.dims[0]) {
                throw std::runtime_error("LLVMGen: eigvec_sym_dd expects matching square matrices");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            if (n != 2u && n != 3u) {
                throw std::runtime_error("LLVMGen: eigvec_sym_dd supports only 2x2 and 3x3 matrices");
            }
            auto* aptr = storeMatrixToStack(a);
            auto* daptr = storeMatrixToStack(da);

            auto* out_arr_ty = llvm::ArrayType::get(f64, static_cast<std::uint32_t>(n));
            auto* out_alloca = builder.CreateAlloca(out_arr_ty);
            auto* out_ptr = builder.CreatePointerCast(out_alloca, f64_ptr);

            auto callee = (n == 2u) ? eigvec_sym_dd_2x2_fn : eigvec_sym_dd_3x3_fn;
            builder.CreateCall(callee, {aptr, daptr, builder.getInt32(which), out_ptr});

            CodeValue out = makeZero(vectorShape(static_cast<std::uint32_t>(n)));
            for (std::size_t r = 0; r < n; ++r) {
                out.elems[r] = loadRealPtrAt(out_ptr, builder.getInt64(r));
            }
            return out;
        };

        auto spectralDecompDD = [&](const CodeValue& a, const CodeValue& da) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || da.shape.kind != Shape::Kind::Matrix ||
                a.shape.dims[0] != a.shape.dims[1] || da.shape.dims[0] != da.shape.dims[1] ||
                a.shape.dims[0] != da.shape.dims[0]) {
                throw std::runtime_error("LLVMGen: spectral_decomp_dd expects matching square matrices");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            if (n != 2u && n != 3u) {
                throw std::runtime_error("LLVMGen: spectral_decomp_dd supports only 2x2 and 3x3 matrices");
            }

            auto* aptr = storeMatrixToStack(a);
            auto* daptr = storeMatrixToStack(da);
            auto* out_arr_ty = llvm::ArrayType::get(f64, static_cast<std::uint32_t>(n * n));
            auto* out_alloca = builder.CreateAlloca(out_arr_ty);
            auto* out_ptr = builder.CreatePointerCast(out_alloca, f64_ptr);

            auto callee = (n == 2u) ? spectral_decomp_dd_2x2_fn : spectral_decomp_dd_3x3_fn;
            builder.CreateCall(callee, {aptr, daptr, out_ptr});

            CodeValue out = makeMatrix(static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(n));
            for (std::size_t i = 0; i < n * n; ++i) {
                out.elems[i] = loadRealPtrAt(out_ptr, builder.getInt64(i));
            }
            return out;
        };

        auto eigSymVec = [&](const CodeValue& a, std::uint32_t which) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || a.shape.dims[0] != a.shape.dims[1]) {
                throw std::runtime_error("LLVMGen: eigvec_sym expects square matrix");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            if (n != 2u && n != 3u) {
                throw std::runtime_error("LLVMGen: eigvec_sym supports only 2x2 and 3x3 matrices");
            }
            auto* aptr = storeMatrixToStack(a);
            auto* evals_arr_ty = llvm::ArrayType::get(f64, static_cast<std::uint32_t>(n));
            auto* evecs_arr_ty = llvm::ArrayType::get(f64, static_cast<std::uint32_t>(n * n));
            auto* evals_alloca = builder.CreateAlloca(evals_arr_ty);
            auto* evecs_alloca = builder.CreateAlloca(evecs_arr_ty);
            auto* evals_ptr = builder.CreatePointerCast(evals_alloca, f64_ptr);
            auto* evecs_ptr = builder.CreatePointerCast(evecs_alloca, f64_ptr);
            auto callee = (n == 2u) ? eig_sym_full_2x2_fn : eig_sym_full_3x3_fn;
            builder.CreateCall(callee, {aptr, evals_ptr, evecs_ptr});

            const auto ww = std::min<std::uint32_t>(which, static_cast<std::uint32_t>(n > 0u ? (n - 1u) : 0u));
            CodeValue out = makeZero(vectorShape(static_cast<std::uint32_t>(n)));
            for (std::size_t r = 0; r < n; ++r) {
                const auto idx = r * n + static_cast<std::size_t>(ww);
                out.elems[r] = loadRealPtrAt(evecs_ptr, builder.getInt64(idx));
            }
            return out;
        };

        auto spectralDecomp = [&](const CodeValue& a) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || a.shape.dims[0] != a.shape.dims[1]) {
                throw std::runtime_error("LLVMGen: spectral_decomp expects square matrix");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            if (n != 2u && n != 3u) {
                throw std::runtime_error("LLVMGen: spectral_decomp supports only 2x2 and 3x3 matrices");
            }
            auto* aptr = storeMatrixToStack(a);
            auto* evals_arr_ty = llvm::ArrayType::get(f64, static_cast<std::uint32_t>(n));
            auto* evecs_arr_ty = llvm::ArrayType::get(f64, static_cast<std::uint32_t>(n * n));
            auto* evals_alloca = builder.CreateAlloca(evals_arr_ty);
            auto* evecs_alloca = builder.CreateAlloca(evecs_arr_ty);
            auto* evals_ptr = builder.CreatePointerCast(evals_alloca, f64_ptr);
            auto* evecs_ptr = builder.CreatePointerCast(evecs_alloca, f64_ptr);
            auto callee = (n == 2u) ? eig_sym_full_2x2_fn : eig_sym_full_3x3_fn;
            builder.CreateCall(callee, {aptr, evals_ptr, evecs_ptr});

            CodeValue out = makeMatrix(static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(n));
            for (std::size_t i = 0; i < n * n; ++i) {
                out.elems[i] = loadRealPtrAt(evecs_ptr, builder.getInt64(i));
            }
            return out;
        };

        auto smoothAbs = [&](const CodeValue& x, const CodeValue& eps) -> CodeValue {
            if (x.shape.kind != Shape::Kind::Scalar || eps.shape.kind != Shape::Kind::Scalar) {
                throw std::runtime_error("LLVMGen: smooth_abs expects scalars");
            }
            auto* xx = builder.CreateFMul(x.elems[0], x.elems[0]);
            auto* ee = builder.CreateFMul(eps.elems[0], eps.elems[0]);
            return makeScalar(f_sqrt(builder.CreateFAdd(xx, ee)));
        };

        auto smoothSign = [&](const CodeValue& x, const CodeValue& eps) -> CodeValue {
            const auto denom = smoothAbs(x, eps);
            return makeScalar(builder.CreateFDiv(x.elems[0], denom.elems[0]));
        };

        auto smoothHeaviside = [&](const CodeValue& x, const CodeValue& eps) -> CodeValue {
            const auto s = smoothSign(x, eps);
            return makeScalar(builder.CreateFMul(f64c(0.5), builder.CreateFAdd(f64c(1.0), s.elems[0])));
        };

        auto smoothMin = [&](const CodeValue& a, const CodeValue& b, const CodeValue& eps) -> CodeValue {
            auto* diff = builder.CreateFSub(a.elems[0], b.elems[0]);
            const auto ad = smoothAbs(makeScalar(diff), eps);
            auto* sum = builder.CreateFAdd(a.elems[0], b.elems[0]);
            return makeScalar(builder.CreateFMul(f64c(0.5), builder.CreateFSub(sum, ad.elems[0])));
        };

        auto smoothMax = [&](const CodeValue& a, const CodeValue& b, const CodeValue& eps) -> CodeValue {
            auto* diff = builder.CreateFSub(a.elems[0], b.elems[0]);
            const auto ad = smoothAbs(makeScalar(diff), eps);
            auto* sum = builder.CreateFAdd(a.elems[0], b.elems[0]);
            return makeScalar(builder.CreateFMul(f64c(0.5), builder.CreateFAdd(sum, ad.elems[0])));
        };

        auto eigSym = [&](const CodeValue& a, std::uint32_t which) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || a.shape.dims[0] != a.shape.dims[1]) {
                throw std::runtime_error("LLVMGen: eig_sym expects square matrix");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            if (n != 2u && n != 3u) {
                throw std::runtime_error("LLVMGen: eig_sym supports only 2x2 and 3x3 matrices");
            }
            auto* aptr = storeMatrixToStack(a);
            auto callee = (n == 2u) ? eig_sym_2x2_fn : eig_sym_3x3_fn;
            auto* val = builder.CreateCall(callee, {aptr, builder.getInt32(which)});
            return makeScalar(val);
        };

        auto eigSymDD = [&](const CodeValue& a, const CodeValue& da, std::uint32_t which) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || da.shape.kind != Shape::Kind::Matrix ||
                a.shape.dims[0] != a.shape.dims[1] || da.shape.dims[0] != da.shape.dims[1] ||
                a.shape.dims[0] != da.shape.dims[0]) {
                throw std::runtime_error("LLVMGen: eig_sym_dd expects matching square matrices");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            if (n != 2u && n != 3u) {
                throw std::runtime_error("LLVMGen: eig_sym_dd supports only 2x2 and 3x3 matrices");
            }
            auto* aptr = storeMatrixToStack(a);
            auto* dptr = storeMatrixToStack(da);
            auto callee = (n == 2u) ? eig_sym_dd_2x2_fn : eig_sym_dd_3x3_fn;
            auto* val = builder.CreateCall(callee, {aptr, dptr, builder.getInt32(which)});
            return makeScalar(val);
        };

        auto eigSymDDWrtA = [&](const CodeValue& a,
                                const CodeValue& b,
                                const CodeValue& da,
                                std::uint32_t which) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || b.shape.kind != Shape::Kind::Matrix || da.shape.kind != Shape::Kind::Matrix ||
                a.shape.dims[0] != a.shape.dims[1] || b.shape.dims[0] != b.shape.dims[1] || da.shape.dims[0] != da.shape.dims[1] ||
                a.shape.dims[0] != b.shape.dims[0] || a.shape.dims[0] != da.shape.dims[0]) {
                throw std::runtime_error("LLVMGen: eig_sym_ddA expects matching square matrices");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            if (n != 2u && n != 3u) {
                throw std::runtime_error("LLVMGen: eig_sym_ddA supports only 2x2 and 3x3 matrices");
            }
            auto* aptr = storeMatrixToStack(a);
            auto* bptr = storeMatrixToStack(b);
            auto* dptr = storeMatrixToStack(da);
            auto callee = (n == 2u) ? eig_sym_ddA_2x2_fn : eig_sym_ddA_3x3_fn;
            auto* val = builder.CreateCall(callee, {aptr, bptr, dptr, builder.getInt32(which)});
            return makeScalar(val);
        };

        auto symOrSkew = [&](bool symmetric, const CodeValue& a) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || a.shape.dims[0] != a.shape.dims[1]) {
                throw std::runtime_error("LLVMGen: sym/skew expects square matrix");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            CodeValue out;
            out.shape = a.shape;
            out.elems.fill(nullptr);
            for (std::size_t r = 0; r < n; ++r) {
                for (std::size_t c = 0; c < n; ++c) {
                    auto* art = a.elems[matIndex(a, r, c)];
                    auto* atr = a.elems[matIndex(a, c, r)];
                    auto* sum = symmetric ? builder.CreateFAdd(art, atr) : builder.CreateFSub(art, atr);
                    out.elems[matIndex(out, r, c)] = builder.CreateFMul(f64c(0.5), sum);
                }
            }
            return out;
        };

        auto deviator = [&](const CodeValue& a) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Matrix || a.shape.dims[0] != a.shape.dims[1] || a.shape.dims[0] == 0u) {
                throw std::runtime_error("LLVMGen: dev expects square matrix");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            auto tr = trace(a).elems[0];
            auto mean = builder.CreateFDiv(tr, f64c(static_cast<double>(n)));
            CodeValue out = a;
            for (std::size_t d = 0; d < n; ++d) {
                const auto idx = matIndex(out, d, d);
                out.elems[idx] = builder.CreateFSub(out.elems[idx], mean);
            }
            return out;
        };

        auto norm = [&](const CodeValue& a) -> CodeValue {
            if (a.shape.kind == Shape::Kind::Scalar) {
                return makeScalar(f_fabs(a.elems[0]));
            }
            llvm::Value* sumv = f64c(0.0);
            const auto n = elemCount(a.shape);
            for (std::size_t i = 0; i < n; ++i) {
                auto* p = builder.CreateFMul(a.elems[i], a.elems[i]);
                sumv = builder.CreateFAdd(sumv, p);
            }
            return makeScalar(f_sqrt(sumv));
        };

        auto normalize = [&](const CodeValue& a) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Vector) {
                throw std::runtime_error("LLVMGen: normalize expects vector");
            }
            const auto n = static_cast<std::size_t>(a.shape.dims[0]);
            llvm::Value* sumv = f64c(0.0);
            for (std::size_t i = 0; i < n; ++i) {
                auto* p = builder.CreateFMul(a.elems[i], a.elems[i]);
                sumv = builder.CreateFAdd(sumv, p);
            }
            auto* nrm = f_sqrt(sumv);
            auto* is_zero = builder.CreateFCmpOEQ(nrm, f64c(0.0));
            CodeValue out;
            out.shape = a.shape;
            out.elems.fill(nullptr);
            for (std::size_t i = 0; i < n; ++i) {
                auto* divv = builder.CreateFDiv(a.elems[i], nrm);
                out.elems[i] = builder.CreateSelect(is_zero, f64c(0.0), divv);
            }
            return out;
        };

        auto absScalar = [&](const CodeValue& a) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Scalar) {
                throw std::runtime_error("LLVMGen: abs expects scalar");
            }
            return makeScalar(f_fabs(a.elems[0]));
        };

        auto signScalar = [&](const CodeValue& a) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Scalar) {
                throw std::runtime_error("LLVMGen: sign expects scalar");
            }
            auto* gt0 = builder.CreateFCmpOGT(a.elems[0], f64c(0.0));
            auto* lt0 = builder.CreateFCmpOLT(a.elems[0], f64c(0.0));
            auto* one = f64c(1.0);
            auto* neg_one = f64c(-1.0);
            auto* pos_or_zero = builder.CreateSelect(gt0, one, f64c(0.0));
            auto* neg_or_zero = builder.CreateSelect(lt0, neg_one, f64c(0.0));
            return makeScalar(builder.CreateFAdd(pos_or_zero, neg_or_zero));
        };

        auto evalSqrt = [&](const CodeValue& a) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Scalar) {
                throw std::runtime_error("LLVMGen: sqrt expects scalar");
            }
            return makeScalar(f_sqrt(a.elems[0]));
        };

        auto evalExp = [&](const CodeValue& a) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Scalar) {
                throw std::runtime_error("LLVMGen: exp expects scalar");
            }
            return makeScalar(f_exp(a.elems[0]));
        };

        auto evalLog = [&](const CodeValue& a) -> CodeValue {
            if (a.shape.kind != Shape::Kind::Scalar) {
                throw std::runtime_error("LLVMGen: log expects scalar");
            }
            return makeScalar(f_log(a.elems[0]));
        };

        auto loadBasisScalar = [&](const SideView& side,
                                   llvm::Value* basis_base,
                                   llvm::Value* dof_index,
                                   llvm::Value* q_index) -> llvm::Value* {
            auto* stride = side.n_qpts;
            auto* offset = builder.CreateAdd(builder.CreateMul(dof_index, stride), q_index);
            auto* offset64 = builder.CreateZExt(offset, i64);
            return loadRealPtrAt(basis_base, offset64);
        };

        auto loadVec3FromTable = [&](llvm::Value* base_ptr,
                                     llvm::Value* n_qpts,
                                     llvm::Value* dof_index,
                                     llvm::Value* q_index) -> std::array<llvm::Value*, 3> {
            auto* offset = builder.CreateAdd(builder.CreateMul(dof_index, n_qpts), q_index);
            auto* base3 = builder.CreateMul(offset, builder.getInt32(3));
            auto* base3_64 = builder.CreateZExt(base3, i64);
            auto* x = loadRealPtrAt(base_ptr, base3_64);
            auto* y = loadRealPtrAt(base_ptr, builder.CreateAdd(base3_64, builder.getInt64(1)));
            auto* z = loadRealPtrAt(base_ptr, builder.CreateAdd(base3_64, builder.getInt64(2)));
            return {x, y, z};
        };

        auto loadVec3FromTableQMajor = [&](llvm::Value* base_ptr,
                                           llvm::Value* n_dofs,
                                           llvm::Value* dof_index,
                                           llvm::Value* q_index) -> std::array<llvm::Value*, 3> {
            auto* offset = builder.CreateAdd(builder.CreateMul(q_index, n_dofs), dof_index);
            auto* base3 = builder.CreateMul(offset, builder.getInt32(3));
            auto* base3_64 = builder.CreateZExt(base3, i64);
            auto* x = loadRealPtrAt(base_ptr, base3_64);
            auto* y = loadRealPtrAt(base_ptr, builder.CreateAdd(base3_64, builder.getInt64(1)));
            auto* z = loadRealPtrAt(base_ptr, builder.CreateAdd(base3_64, builder.getInt64(2)));
            return {x, y, z};
        };

        auto loadMat3FromTable = [&](llvm::Value* base_ptr,
                                     llvm::Value* n_qpts,
                                     llvm::Value* dof_index,
                                     llvm::Value* q_index) -> CodeValue {
            CodeValue out = makeMatrix(3u, 3u);
            auto* offset = builder.CreateAdd(builder.CreateMul(dof_index, n_qpts), q_index);
            auto* base9 = builder.CreateMul(offset, builder.getInt32(9));
            auto* base9_64 = builder.CreateZExt(base9, i64);
            for (std::size_t i = 0; i < 9; ++i) {
                out.elems[i] = loadRealPtrAt(base_ptr, builder.CreateAdd(base9_64, builder.getInt64(i)));
            }
            return out;
        };

        auto loadMatDimFromTable = [&](llvm::Value* base_ptr,
                                       llvm::Value* n_qpts,
                                       llvm::Value* dof_index,
                                       llvm::Value* q_index,
                                       std::uint32_t dim) -> CodeValue {
            CodeValue out = makeMatrix(dim, dim);
            auto* offset = builder.CreateAdd(builder.CreateMul(dof_index, n_qpts), q_index);
            auto* base9 = builder.CreateMul(offset, builder.getInt32(9));
            auto* base9_64 = builder.CreateZExt(base9, i64);
            for (std::uint32_t r = 0; r < dim; ++r) {
                for (std::uint32_t c = 0; c < dim; ++c) {
                    const std::uint32_t idx = r * 3u + c;
                    out.elems[static_cast<std::size_t>(r * dim + c)] =
                        loadRealPtrAt(base_ptr, builder.CreateAdd(base9_64, builder.getInt64(idx)));
                }
            }
            return out;
        };

	        auto loadXYZ = [&](llvm::Value* xyz_base,
	                           llvm::Value* q_index) -> CodeValue {
	            auto* base3 = builder.CreateMul(q_index, builder.getInt32(3));
	            auto* base3_64 = builder.CreateZExt(base3, i64);
	            auto* x = loadRealPtrAt(xyz_base, base3_64);
	            auto* y = loadRealPtrAt(xyz_base, builder.CreateAdd(base3_64, builder.getInt64(1)));
	            auto* z = loadRealPtrAt(xyz_base, builder.CreateAdd(base3_64, builder.getInt64(2)));
	            return makeVector(3u, x, y, z);
	        };

	        auto loadXYZDim = [&](llvm::Value* xyz_base,
	                              llvm::Value* q_index,
	                              std::uint32_t dim) -> CodeValue {
	            auto* base3 = builder.CreateMul(q_index, builder.getInt32(3));
	            auto* base3_64 = builder.CreateZExt(base3, i64);
	            auto* x = loadRealPtrAt(xyz_base, base3_64);
	            auto* y = loadRealPtrAt(xyz_base, builder.CreateAdd(base3_64, builder.getInt64(1)));
	            auto* z = loadRealPtrAt(xyz_base, builder.CreateAdd(base3_64, builder.getInt64(2)));
	            return makeVector(dim, x, y, z);
	        };

	        auto loadVec3FromQ = [&](llvm::Value* vec_base,
	                                 llvm::Value* q_index) -> std::array<llvm::Value*, 3> {
	            auto* base3 = builder.CreateMul(q_index, builder.getInt32(3));
	            auto* base3_64 = builder.CreateZExt(base3, i64);
	            auto* x = loadRealPtrAt(vec_base, base3_64);
	            auto* y = loadRealPtrAt(vec_base, builder.CreateAdd(base3_64, builder.getInt64(1)));
	            auto* z = loadRealPtrAt(vec_base, builder.CreateAdd(base3_64, builder.getInt64(2)));
	            return {x, y, z};
	        };

	        auto loadMat3FromQ = [&](llvm::Value* mat_base,
	                                 llvm::Value* q_index) -> CodeValue {
	            CodeValue out = makeMatrix(3u, 3u);
	            auto* base9 = builder.CreateMul(q_index, builder.getInt32(9));
	            auto* base9_64 = builder.CreateZExt(base9, i64);
            for (std::size_t i = 0; i < 9; ++i) {
                out.elems[i] = loadRealPtrAt(mat_base, builder.CreateAdd(base9_64, builder.getInt64(i)));
            }
            return out;
        };

	        auto loadMatDimFromQ = [&](llvm::Value* mat_base,
	                                  llvm::Value* q_index,
	                                  std::uint32_t dim) -> CodeValue {
	            CodeValue out = makeMatrix(dim, dim);
	            auto* base9 = builder.CreateMul(q_index, builder.getInt32(9));
	            auto* base9_64 = builder.CreateZExt(base9, i64);
	            for (std::uint32_t r = 0; r < dim; ++r) {
	                for (std::uint32_t c = 0; c < dim; ++c) {
	                    const std::uint32_t idx = r * 3u + c;
	                    out.elems[static_cast<std::size_t>(r * dim + c)] =
	                        loadRealPtrAt(mat_base, builder.CreateAdd(base9_64, builder.getInt64(idx)));
	                }
	            }
	            return out;
	        };

        auto loadInterleavedReal = [&](const SideView& side,
                                       llvm::Value* q_index,
                                       llvm::Value* field_offset,
                                       llvm::Value* component_offset) -> llvm::Value* {
            auto* q_stride = builder.CreateMul(q_index, side.interleaved_qpoint_geometry_stride_reals);
            auto* idx = builder.CreateAdd(q_stride, builder.CreateAdd(field_offset, component_offset));
            return loadRealPtrAt(side.interleaved_qpoint_geometry, builder.CreateZExt(idx, i64));
        };

        auto useInterleavedGeometry = [&](const SideView& side) -> llvm::Value* {
            auto* has_ptr = builder.CreateIsNotNull(side.interleaved_qpoint_geometry);
            auto* has_stride =
                builder.CreateICmpUGT(side.interleaved_qpoint_geometry_stride_reals, builder.getInt32(0));
            return builder.CreateAnd(has_ptr, has_stride);
        };

        auto loadXYZDimFromSide = [&](const SideView& side,
                                      llvm::Value* legacy_xyz_base,
                                      llvm::Value* q_index,
                                      std::uint32_t dim,
                                      llvm::Value* interleaved_offset) -> CodeValue {
            const auto legacy = loadXYZDim(legacy_xyz_base, q_index, dim);
            auto inter = makeVector(dim, f64c(0.0), f64c(0.0), f64c(0.0));
            inter.elems[0] = loadInterleavedReal(side, q_index, interleaved_offset, builder.getInt32(0));
            inter.elems[1] = loadInterleavedReal(side, q_index, interleaved_offset, builder.getInt32(1));
            inter.elems[2] = loadInterleavedReal(side, q_index, interleaved_offset, builder.getInt32(2));
            auto out = makeVector(dim, f64c(0.0), f64c(0.0), f64c(0.0));
            auto* use_interleaved = useInterleavedGeometry(side);
            for (std::uint32_t d = 0; d < dim; ++d) {
                out.elems[d] = builder.CreateSelect(use_interleaved, inter.elems[d], legacy.elems[d]);
            }
            return out;
        };

        auto loadMatDimFromSide = [&](const SideView& side,
                                      llvm::Value* legacy_mat_base,
                                      llvm::Value* q_index,
                                      std::uint32_t dim,
                                      llvm::Value* interleaved_offset) -> CodeValue {
            const auto legacy = loadMatDimFromQ(legacy_mat_base, q_index, dim);
            auto inter = makeMatrix(dim, dim);
            for (std::uint32_t r = 0; r < dim; ++r) {
                for (std::uint32_t c = 0; c < dim; ++c) {
                    const std::uint32_t idx = r * 3u + c;
                    inter.elems[static_cast<std::size_t>(r * dim + c)] =
                        loadInterleavedReal(side, q_index, interleaved_offset, builder.getInt32(idx));
                }
            }
            auto out = makeMatrix(dim, dim);
            auto* use_interleaved = useInterleavedGeometry(side);
            for (std::uint32_t r = 0; r < dim; ++r) {
                for (std::uint32_t c = 0; c < dim; ++c) {
                    const std::size_t idx = static_cast<std::size_t>(r * dim + c);
                    out.elems[idx] = builder.CreateSelect(use_interleaved, inter.elems[idx], legacy.elems[idx]);
                }
            }
            return out;
        };

        auto loadScalarFromSide = [&](const SideView& side,
                                      llvm::Value* legacy_base,
                                      llvm::Value* q_index,
                                      llvm::Value* interleaved_offset) -> llvm::Value* {
            auto* legacy = loadRealPtrAt(legacy_base, builder.CreateZExt(q_index, i64));
            auto* inter =
                loadInterleavedReal(side, q_index, interleaved_offset, builder.getInt32(0));
            auto* use_interleaved = useInterleavedGeometry(side);
            return builder.CreateSelect(use_interleaved, inter, legacy);
        };

        auto loadMaterialStateReal = [&](llvm::Value* base_ptr,
                                         llvm::Value* stride_bytes,
                                         llvm::Value* q_index,
                                         std::uint64_t offset_bytes) -> llvm::Value* {
            auto* q64 = builder.CreateZExt(q_index, i64);
            auto* q_off = builder.CreateMul(q64, stride_bytes);
            auto* byte_off = builder.CreateAdd(q_off, builder.getInt64(offset_bytes));
            auto* addr = builder.CreateGEP(builder.getInt8Ty(), base_ptr, byte_off);
            return builder.CreateLoad(f64, addr);
        };

        auto storeMaterialStateReal = [&](llvm::Value* base_ptr,
                                          llvm::Value* stride_bytes,
                                          llvm::Value* q_index,
                                          std::uint64_t offset_bytes,
                                          llvm::Value* value) -> void {
            auto* q64 = builder.CreateZExt(q_index, i64);
            auto* q_off = builder.CreateMul(q64, stride_bytes);
            auto* byte_off = builder.CreateAdd(q_off, builder.getInt64(offset_bytes));
            auto* addr = builder.CreateGEP(builder.getInt8Ty(), base_ptr, byte_off);
            builder.CreateStore(value, addr);
        };

        constexpr std::uint64_t kDtCoeffStride =
            static_cast<std::uint64_t>(assembly::jit::kMaxPreviousSolutionsV6 + 1u);

        auto loadDtCoeff = [&](const SideView& side, int order, int history_index) -> llvm::Value* {
            if (order < 1 || order > static_cast<int>(assembly::jit::kMaxTimeDerivativeOrderV6)) {
                return f64c(0.0);
            }
            if (history_index < 0 || history_index > static_cast<int>(assembly::jit::kMaxPreviousSolutionsV6)) {
                return f64c(0.0);
            }
            const std::uint64_t idx =
                static_cast<std::uint64_t>(static_cast<std::int64_t>(order - 1)) * kDtCoeffStride +
                static_cast<std::uint64_t>(static_cast<std::int64_t>(history_index));
            return loadRealPtrAt(side.dt_stencil_coeffs_base, builder.getInt64(idx));
        };

        auto loadEffectiveDt = [&](const SideView& side) -> llvm::Value* {
            auto* a0 = loadDtCoeff(side, /*order=*/1, /*history_index=*/0);
            auto* is_zero = builder.CreateFCmpOEQ(a0, f64c(0.0));
            auto* inv_abs = builder.CreateFDiv(f64c(1.0), f_fabs(a0));
            return builder.CreateSelect(is_zero, side.dt, inv_abs);
        };

        auto loadDtCoeff0 = [&](const SideView& side, int order) -> llvm::Value* {
            return loadDtCoeff(side, order, /*history_index=*/0);
        };

        auto loadPrevSolutionCoeffsPtr = [&](const SideView& side, int k) -> llvm::Value* {
            const auto idx = static_cast<std::uint64_t>(static_cast<std::int64_t>(k - 1));
            auto* addr = builder.CreateGEP(builder.getInt8Ty(),
                                           side.previous_solution_coefficients_base,
                                           builder.getInt64(idx * sizeof(void*)));
            return builder.CreateLoad(i8_ptr, addr);
        };

        auto loadHistorySolutionCoeffsPtr = [&](const SideView& side, int k) -> llvm::Value* {
            const auto idx = static_cast<std::uint64_t>(static_cast<std::int64_t>(k - 1));
            auto* addr = builder.CreateGEP(builder.getInt8Ty(),
                                           side.history_solution_coefficients_base,
                                           builder.getInt64(idx * sizeof(void*)));
            return builder.CreateLoad(i8_ptr, addr);
        };

        auto unpackFieldIdImm1 = [&](std::uint64_t imm1) -> int {
            return static_cast<int>((imm1 >> 16) & 0xffffULL);
        };

        std::vector<int> used_field_ids;
        used_field_ids.reserve(8);
        for (const auto& t : terms) {
            for (const auto& op : t.ir.ops) {
                if (op.type == FormExprType::DiscreteField || op.type == FormExprType::StateField) {
                    const int fid = unpackFieldIdImm1(op.imm1);
                    if (op.type == FormExprType::StateField && fid == 0xffff) {
                        // StateField(INVALID_FIELD_ID) represents the current solution state u, not a field table entry.
                        continue;
                    }
                    used_field_ids.push_back(fid);
                }
            }
        }
        std::sort(used_field_ids.begin(), used_field_ids.end());
        used_field_ids.erase(std::unique(used_field_ids.begin(), used_field_ids.end()), used_field_ids.end());

        struct FieldEntryPtrs {
            int field_id{-1};
            llvm::Value* single{nullptr};
            llvm::Value* minus{nullptr};
            llvm::Value* plus{nullptr};
        };

        auto emitFindFieldEntry = [&](const SideView& side,
                                      int field_id,
                                      std::string_view tag) -> llvm::Value* {
            const auto entry_bytes = static_cast<std::uint64_t>(sizeof(assembly::jit::FieldSolutionEntryV1));
            auto* found_ptr = builder.CreateAlloca(i8_ptr, nullptr, std::string("field_entry_") + std::string(tag));
            builder.CreateStore(llvm::ConstantPointerNull::get(i8_ptr), found_ptr);

            auto* fs_is_null = builder.CreateICmpEQ(side.field_solutions, llvm::ConstantPointerNull::get(i8_ptr));
            auto* end = builder.CreateSelect(fs_is_null, builder.getInt32(0), side.num_field_solutions);

            emitForLoop(end, std::string("find_field_") + std::string(tag), [&](llvm::Value* idx) {
                auto* idx64 = builder.CreateZExt(idx, i64);
                auto* byte_off = builder.CreateMul(idx64, builder.getInt64(entry_bytes));
                auto* entry_ptr = builder.CreateGEP(builder.getInt8Ty(), side.field_solutions, byte_off);
                auto* id = loadI32(entry_ptr, ABIV3::field_entry_field_id_off);
                auto* hit = builder.CreateICmpEQ(id, builder.getInt32(field_id));
                auto* old = builder.CreateLoad(i8_ptr, found_ptr);
                auto* neu = builder.CreateSelect(hit, entry_ptr, old);
                builder.CreateStore(neu, found_ptr);
            });

            return builder.CreateLoad(i8_ptr, found_ptr);
        };

        std::vector<FieldEntryPtrs> field_entries;
        field_entries.reserve(used_field_ids.size());
        for (const auto fid : used_field_ids) {
            FieldEntryPtrs e;
            e.field_id = fid;
            if (!is_face_domain) {
                e.single = emitFindFieldEntry(side_single, fid, "s" + std::to_string(fid));
            } else {
                e.minus = emitFindFieldEntry(side_minus, fid, "m" + std::to_string(fid));
                e.plus = emitFindFieldEntry(side_plus, fid, "p" + std::to_string(fid));
            }
            field_entries.push_back(e);
        }

        auto fieldEntryPtrFor = [&](bool plus_side, int fid) -> llvm::Value* {
            for (const auto& e : field_entries) {
                if (e.field_id == fid) {
                    if (!is_face_domain) return e.single;
                    return plus_side ? e.plus : e.minus;
                }
            }
            return llvm::ConstantPointerNull::get(i8_ptr);
        };

        const bool is_residual = (ir.kind() == FormKind::Residual);
        const bool want_matrix = (ir.kind() == FormKind::Bilinear);
        const bool want_vector = (ir.kind() == FormKind::Linear) || is_residual;

        if (is_face_domain) {
            if (ir.kind() == FormKind::Linear) {
                return LLVMGenResult{.ok = false, .message = "LLVMGen: face kernels currently support bilinear/residual forms only"};
            }
        }

        if (!want_matrix && !want_vector) {
            return LLVMGenResult{.ok = false, .message = "LLVMGen: unsupported FormKind"};
        }

        if (ir.kind() == FormKind::Linear) {
            for (const auto& t : terms) {
                for (const auto& op : t.ir.ops) {
                    if (op.type == FormExprType::TrialFunction) {
                        return LLVMGenResult{.ok = false,
                                             .message = "LLVMGen: TrialFunction in Linear form is not supported"};
                    }
                }
            }
        }

	        auto evalPreviousSolution = [&](const SideView& side,
	                                        const Shape& out_shape,
	                                        int k,
	                                        llvm::Value* q_index) -> CodeValue {
	            auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
	            auto* coeffs_null = builder.CreateICmpEQ(coeffs, llvm::ConstantPointerNull::get(i8_ptr));

	            const std::string tag = "prev_u" + std::to_string(k);
	            auto* ok = llvm::BasicBlock::Create(*ctx, tag + ".ok", fn);
	            auto* zero = llvm::BasicBlock::Create(*ctx, tag + ".zero", fn);
	            auto* merge = llvm::BasicBlock::Create(*ctx, tag + ".merge", fn);

	            builder.CreateCondBr(coeffs_null, zero, ok);

	            if (out_shape.kind == Shape::Kind::Scalar) {
	                builder.SetInsertPoint(ok);
	                auto* sum = emitReduceSumScalar(side.n_trial_dofs, tag + ".sum", [&](llvm::Value* j) -> llvm::Value* {
	                    auto* j64 = builder.CreateZExt(j, i64);
	                    auto* cj = loadRealPtrAt(coeffs, j64);
	                    auto* phi = loadBasisScalar(side, side.trial_basis_values, j, q_index);
	                    return builder.CreateFMul(cj, phi);
	                });
	                builder.CreateBr(merge);
	                auto* ok_block = builder.GetInsertBlock();

	                builder.SetInsertPoint(zero);
	                builder.CreateBr(merge);
	                auto* zero_block = builder.GetInsertBlock();

	                builder.SetInsertPoint(merge);
	                auto* phi = builder.CreatePHI(f64, 2, tag + ".val");
	                phi->addIncoming(f64c(0.0), zero_block);
	                phi->addIncoming(sum, ok_block);
	                return makeScalar(phi);
	            }

	            if (out_shape.kind == Shape::Kind::Vector) {
	                const auto vd = static_cast<std::size_t>(out_shape.dims[0]);
	                CodeValue out = makeZero(out_shape);

	                builder.SetInsertPoint(ok);
	                auto* uses_vec_basis = builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
	                auto* vb = llvm::BasicBlock::Create(*ctx, tag + ".vb", fn);
	                auto* sb = llvm::BasicBlock::Create(*ctx, tag + ".sb", fn);
	                auto* ok_merge = llvm::BasicBlock::Create(*ctx, tag + ".ok.merge", fn);
	                builder.CreateCondBr(uses_vec_basis, vb, sb);

	                builder.SetInsertPoint(vb);
	                auto vb_sums = emitReduceSum(side.n_trial_dofs, tag + ".vb.sum", vd, [&](llvm::Value* j) {
	                    auto* j64 = builder.CreateZExt(j, i64);
	                    auto* cj = loadRealPtrAt(coeffs, j64);
	                    const auto phi = loadVec3FromTable(side.trial_basis_vector_values_xyz, side.n_qpts, j, q_index);
	                    std::vector<llvm::Value*> terms;
	                    terms.reserve(vd);
	                    for (std::size_t c = 0; c < vd; ++c) {
	                        terms.push_back(builder.CreateFMul(cj, phi[c]));
	                    }
	                    return terms;
	                });
	                builder.CreateBr(ok_merge);
	                auto* vb_block = builder.GetInsertBlock();

	                builder.SetInsertPoint(sb);
	                std::vector<llvm::Value*> sb_sums;
	                sb_sums.reserve(vd);
	                auto* dofs_per_comp =
	                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                for (std::size_t comp = 0; comp < vd; ++comp) {
	                    auto* sum = emitReduceSumScalar(dofs_per_comp,
	                                                    tag + ".sb.sum_c" + std::to_string(comp),
	                                                    [&](llvm::Value* jj) -> llvm::Value* {
	                        auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                        auto* j = builder.CreateAdd(base, jj);
	                        auto* j64 = builder.CreateZExt(j, i64);
	                        auto* cj = loadRealPtrAt(coeffs, j64);
	                        auto* phi = loadBasisScalar(side, side.trial_basis_values, j, q_index);
	                        return builder.CreateFMul(cj, phi);
	                    });
	                    sb_sums.push_back(sum);
	                }
	                builder.CreateBr(ok_merge);
	                auto* sb_block = builder.GetInsertBlock();

	                builder.SetInsertPoint(ok_merge);
	                std::vector<llvm::Value*> ok_vals;
	                ok_vals.reserve(vd);
	                for (std::size_t c = 0; c < vd; ++c) {
	                    auto* phi = builder.CreatePHI(f64, 2, tag + ".ok.phi" + std::to_string(c));
	                    phi->addIncoming(vb_sums[c], vb_block);
	                    phi->addIncoming(sb_sums[c], sb_block);
	                    ok_vals.push_back(phi);
	                }
	                builder.CreateBr(merge);
	                auto* ok_block = builder.GetInsertBlock();

	                builder.SetInsertPoint(zero);
	                builder.CreateBr(merge);
	                auto* zero_block = builder.GetInsertBlock();

	                builder.SetInsertPoint(merge);
	                for (std::size_t c = 0; c < vd; ++c) {
	                    auto* phi = builder.CreatePHI(f64, 2, tag + ".phi" + std::to_string(c));
	                    phi->addIncoming(f64c(0.0), zero_block);
	                    phi->addIncoming(ok_vals[c], ok_block);
	                    out.elems[c] = phi;
	                }
	                return out;
	            }

		            throw std::runtime_error("LLVMGen: PreviousSolutionRef unsupported shape");
		        };

		        auto evalHistorySolution = [&](const SideView& side,
		                                       const Shape& out_shape,
		                                       int k,
		                                       llvm::Value* q_index) -> CodeValue {
		            auto* coeffs = loadHistorySolutionCoeffsPtr(side, k);
		            auto* coeffs_null = builder.CreateICmpEQ(coeffs, llvm::ConstantPointerNull::get(i8_ptr));

		            const std::string tag = "hist_u" + std::to_string(k);
		            auto* ok = llvm::BasicBlock::Create(*ctx, tag + ".ok", fn);
		            auto* zero = llvm::BasicBlock::Create(*ctx, tag + ".zero", fn);
		            auto* merge = llvm::BasicBlock::Create(*ctx, tag + ".merge", fn);

		            builder.CreateCondBr(coeffs_null, zero, ok);

		            if (out_shape.kind == Shape::Kind::Scalar) {
		                builder.SetInsertPoint(ok);
		                auto* sum =
		                    emitReduceSumScalar(side.n_trial_dofs, tag + ".sum", [&](llvm::Value* j) -> llvm::Value* {
		                        auto* j64 = builder.CreateZExt(j, i64);
		                        auto* cj = loadRealPtrAt(coeffs, j64);
		                        auto* phi = loadBasisScalar(side, side.trial_basis_values, j, q_index);
		                        return builder.CreateFMul(cj, phi);
		                    });
		                builder.CreateBr(merge);
		                auto* ok_block = builder.GetInsertBlock();

		                builder.SetInsertPoint(zero);
		                builder.CreateBr(merge);
		                auto* zero_block = builder.GetInsertBlock();

		                builder.SetInsertPoint(merge);
		                auto* phi = builder.CreatePHI(f64, 2, tag + ".val");
		                phi->addIncoming(f64c(0.0), zero_block);
		                phi->addIncoming(sum, ok_block);
		                return makeScalar(phi);
		            }

		            if (out_shape.kind == Shape::Kind::Vector) {
		                const auto vd = static_cast<std::size_t>(out_shape.dims[0]);
		                CodeValue out = makeZero(out_shape);

		                builder.SetInsertPoint(ok);
		                auto* uses_vec_basis =
		                    builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
		                auto* vb = llvm::BasicBlock::Create(*ctx, tag + ".vb", fn);
		                auto* sb = llvm::BasicBlock::Create(*ctx, tag + ".sb", fn);
		                auto* ok_merge = llvm::BasicBlock::Create(*ctx, tag + ".ok.merge", fn);
		                builder.CreateCondBr(uses_vec_basis, vb, sb);

		                builder.SetInsertPoint(vb);
		                auto vb_sums = emitReduceSum(side.n_trial_dofs, tag + ".vb.sum", vd, [&](llvm::Value* j) {
		                    auto* j64 = builder.CreateZExt(j, i64);
		                    auto* cj = loadRealPtrAt(coeffs, j64);
		                    const auto phi =
		                        loadVec3FromTable(side.trial_basis_vector_values_xyz, side.n_qpts, j, q_index);
		                    std::vector<llvm::Value*> terms;
		                    terms.reserve(vd);
		                    for (std::size_t c = 0; c < vd; ++c) {
		                        terms.push_back(builder.CreateFMul(cj, phi[c]));
		                    }
		                    return terms;
		                });
		                builder.CreateBr(ok_merge);
		                auto* vb_block = builder.GetInsertBlock();

		                builder.SetInsertPoint(sb);
		                std::vector<llvm::Value*> sb_sums;
		                sb_sums.reserve(vd);
		                auto* dofs_per_comp =
		                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
		                for (std::size_t comp = 0; comp < vd; ++comp) {
		                    auto* sum = emitReduceSumScalar(
		                        dofs_per_comp,
		                        tag + ".sb.sum_c" + std::to_string(comp),
		                        [&](llvm::Value* jj) -> llvm::Value* {
		                            auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)),
		                                                           dofs_per_comp);
		                            auto* j = builder.CreateAdd(base, jj);
		                            auto* j64 = builder.CreateZExt(j, i64);
		                            auto* cj = loadRealPtrAt(coeffs, j64);
		                            auto* phi = loadBasisScalar(side, side.trial_basis_values, j, q_index);
		                            return builder.CreateFMul(cj, phi);
		                        });
		                    sb_sums.push_back(sum);
		                }
		                builder.CreateBr(ok_merge);
		                auto* sb_block = builder.GetInsertBlock();

		                builder.SetInsertPoint(ok_merge);
		                std::vector<llvm::Value*> ok_vals;
		                ok_vals.reserve(vd);
		                for (std::size_t c = 0; c < vd; ++c) {
		                    auto* phi = builder.CreatePHI(f64, 2, tag + ".ok.phi" + std::to_string(c));
		                    phi->addIncoming(vb_sums[c], vb_block);
		                    phi->addIncoming(sb_sums[c], sb_block);
		                    ok_vals.push_back(phi);
		                }
		                builder.CreateBr(merge);
		                auto* ok_block = builder.GetInsertBlock();

		                builder.SetInsertPoint(zero);
		                builder.CreateBr(merge);
		                auto* zero_block = builder.GetInsertBlock();

		                builder.SetInsertPoint(merge);
		                for (std::size_t c = 0; c < vd; ++c) {
		                    auto* phi = builder.CreatePHI(f64, 2, tag + ".phi" + std::to_string(c));
		                    phi->addIncoming(f64c(0.0), zero_block);
		                    phi->addIncoming(ok_vals[c], ok_block);
		                    out.elems[c] = phi;
		                }
		                return out;
		            }

		            throw std::runtime_error("LLVMGen: history solution unsupported shape");
		        };

		        auto loadHistoryWeightOrZero = [&](const SideView& side, int k) -> llvm::Value* {
		            auto* weights = side.history_weights;
		            auto* weights_null = builder.CreateICmpEQ(weights, llvm::ConstantPointerNull::get(i8_ptr));
		            auto* have_k = builder.CreateICmpUGE(side.num_history_steps,
		                                                 builder.getInt32(static_cast<std::uint32_t>(k)));
		            auto* cond = builder.CreateAnd(builder.CreateNot(weights_null), have_k);

		            const std::string tag = "hist_w" + std::to_string(k);
		            auto* ok = llvm::BasicBlock::Create(*ctx, tag + ".ok", fn);
		            auto* zero = llvm::BasicBlock::Create(*ctx, tag + ".zero", fn);
		            auto* merge = llvm::BasicBlock::Create(*ctx, tag + ".merge", fn);

		            builder.CreateCondBr(cond, ok, zero);

		            builder.SetInsertPoint(ok);
		            auto* w = loadRealPtrAt(weights, builder.getInt64(static_cast<std::uint64_t>(k - 1)));
		            builder.CreateBr(merge);
		            auto* ok_block = builder.GetInsertBlock();

		            builder.SetInsertPoint(zero);
		            builder.CreateBr(merge);
		            auto* zero_block = builder.GetInsertBlock();

		            builder.SetInsertPoint(merge);
		            auto* phi = builder.CreatePHI(f64, 2, tag + ".val");
		            phi->addIncoming(f64c(0.0), zero_block);
		            phi->addIncoming(w, ok_block);
		            return phi;
		        };

		        auto evalCurrentSolution = [&](const SideView& side,
		                                       const Shape& out_shape,
		                                       llvm::Value* q_index) -> CodeValue {
		            auto* coeffs = side.solution_coefficients;
		            auto* coeffs_null = builder.CreateICmpEQ(coeffs, llvm::ConstantPointerNull::get(i8_ptr));

	            constexpr std::string_view tag = "u";
	            auto* ok = llvm::BasicBlock::Create(*ctx, std::string(tag) + ".ok", fn);
	            auto* zero = llvm::BasicBlock::Create(*ctx, std::string(tag) + ".zero", fn);
	            auto* merge = llvm::BasicBlock::Create(*ctx, std::string(tag) + ".merge", fn);

	            builder.CreateCondBr(coeffs_null, zero, ok);

	            if (out_shape.kind == Shape::Kind::Scalar) {
	                builder.SetInsertPoint(ok);
	                auto* sum = emitReduceSumScalar(side.n_trial_dofs, std::string(tag) + ".sum", [&](llvm::Value* j) -> llvm::Value* {
	                    auto* j64 = builder.CreateZExt(j, i64);
	                    auto* cj = loadRealPtrAt(coeffs, j64);
	                    auto* phi = loadBasisScalar(side, side.trial_basis_values, j, q_index);
	                    return builder.CreateFMul(cj, phi);
	                });
	                builder.CreateBr(merge);
	                auto* ok_block = builder.GetInsertBlock();

	                builder.SetInsertPoint(zero);
	                builder.CreateBr(merge);
	                auto* zero_block = builder.GetInsertBlock();

	                builder.SetInsertPoint(merge);
	                auto* phi = builder.CreatePHI(f64, 2, std::string(tag) + ".val");
	                phi->addIncoming(f64c(0.0), zero_block);
	                phi->addIncoming(sum, ok_block);
	                return makeScalar(phi);
	            }

	            if (out_shape.kind == Shape::Kind::Vector) {
	                const auto vd = static_cast<std::size_t>(out_shape.dims[0]);
	                CodeValue out = makeZero(out_shape);

	                builder.SetInsertPoint(ok);
	                auto* uses_vec_basis = builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
	                auto* vb = llvm::BasicBlock::Create(*ctx, std::string(tag) + ".vb", fn);
	                auto* sb = llvm::BasicBlock::Create(*ctx, std::string(tag) + ".sb", fn);
	                auto* ok_merge = llvm::BasicBlock::Create(*ctx, std::string(tag) + ".ok.merge", fn);
	                builder.CreateCondBr(uses_vec_basis, vb, sb);

	                builder.SetInsertPoint(vb);
	                auto vb_sums = emitReduceSum(side.n_trial_dofs, std::string(tag) + ".vb.sum", vd, [&](llvm::Value* j) {
	                    auto* j64 = builder.CreateZExt(j, i64);
	                    auto* cj = loadRealPtrAt(coeffs, j64);
	                    const auto phi = loadVec3FromTable(side.trial_basis_vector_values_xyz, side.n_qpts, j, q_index);
	                    std::vector<llvm::Value*> terms;
	                    terms.reserve(vd);
	                    for (std::size_t c = 0; c < vd; ++c) {
	                        terms.push_back(builder.CreateFMul(cj, phi[c]));
	                    }
	                    return terms;
	                });
	                builder.CreateBr(ok_merge);
	                auto* vb_block = builder.GetInsertBlock();

	                builder.SetInsertPoint(sb);
	                std::vector<llvm::Value*> sb_sums;
	                sb_sums.reserve(vd);
	                auto* dofs_per_comp =
	                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                for (std::size_t comp = 0; comp < vd; ++comp) {
	                    auto* sum = emitReduceSumScalar(dofs_per_comp,
	                                                    std::string(tag) + ".sb.sum_c" + std::to_string(comp),
	                                                    [&](llvm::Value* jj) -> llvm::Value* {
	                        auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                        auto* j = builder.CreateAdd(base, jj);
	                        auto* j64 = builder.CreateZExt(j, i64);
	                        auto* cj = loadRealPtrAt(coeffs, j64);
	                        auto* phi = loadBasisScalar(side, side.trial_basis_values, j, q_index);
	                        return builder.CreateFMul(cj, phi);
	                    });
	                    sb_sums.push_back(sum);
	                }
	                builder.CreateBr(ok_merge);
	                auto* sb_block = builder.GetInsertBlock();

	                builder.SetInsertPoint(ok_merge);
	                std::vector<llvm::Value*> ok_vals;
	                ok_vals.reserve(vd);
	                for (std::size_t c = 0; c < vd; ++c) {
	                    auto* phi = builder.CreatePHI(f64, 2, std::string(tag) + ".ok.phi" + std::to_string(c));
	                    phi->addIncoming(vb_sums[c], vb_block);
	                    phi->addIncoming(sb_sums[c], sb_block);
	                    ok_vals.push_back(phi);
	                }
	                builder.CreateBr(merge);
	                auto* ok_block = builder.GetInsertBlock();

	                builder.SetInsertPoint(zero);
	                builder.CreateBr(merge);
	                auto* zero_block = builder.GetInsertBlock();

	                builder.SetInsertPoint(merge);
	                for (std::size_t c = 0; c < vd; ++c) {
	                    auto* phi = builder.CreatePHI(f64, 2, std::string(tag) + ".phi" + std::to_string(c));
	                    phi->addIncoming(f64c(0.0), zero_block);
	                    phi->addIncoming(ok_vals[c], ok_block);
	                    out.elems[c] = phi;
	                }
	                return out;
	            }

	            throw std::runtime_error("LLVMGen: current solution unsupported shape");
	        };

        auto evalDiscreteOrStateField = [&](bool plus_side,
                                            const Shape& out_shape,
                                            int fid,
                                            llvm::Value* q_index) -> CodeValue {
            auto* entry = fieldEntryPtrFor(plus_side, fid);
            auto* is_null = builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));

            auto* merge = llvm::BasicBlock::Create(*ctx, "field.merge", fn);
            auto* ok = llvm::BasicBlock::Create(*ctx, "field.ok", fn);
            auto* zero = llvm::BasicBlock::Create(*ctx, "field.zero", fn);

            builder.CreateCondBr(is_null, zero, ok);

            if (out_shape.kind == Shape::Kind::Scalar) {
                llvm::Value* loaded = f64c(0.0);

                builder.SetInsertPoint(ok);
                auto* base = loadPtr(entry, ABIV3::field_entry_values_off);
                auto* base_is_null = builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field.values.ok", fn);
                builder.CreateCondBr(base_is_null, zero, ok2);

                builder.SetInsertPoint(ok2);
                auto* q64 = builder.CreateZExt(q_index, i64);
                loaded = loadRealPtrAt(base, q64);
                builder.CreateBr(merge);
                auto* ok2_block = builder.GetInsertBlock();

                builder.SetInsertPoint(zero);
                builder.CreateBr(merge);
                auto* zero_block = builder.GetInsertBlock();

                builder.SetInsertPoint(merge);
                auto* phi = builder.CreatePHI(f64, 2, "field.s");
                phi->addIncoming(f64c(0.0), zero_block);
                phi->addIncoming(loaded, ok2_block);
                return makeScalar(phi);
            }

            if (out_shape.kind == Shape::Kind::Vector) {
                const auto vd = static_cast<std::size_t>(out_shape.dims[0]);
                std::array<llvm::Value*, 3> loaded{f64c(0.0), f64c(0.0), f64c(0.0)};

                builder.SetInsertPoint(ok);
                auto* base = loadPtr(entry, ABIV3::field_entry_vector_values_xyz_off);
                auto* base_is_null = builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field.vec.ok", fn);
                builder.CreateCondBr(base_is_null, zero, ok2);

                builder.SetInsertPoint(ok2);
                const auto v = loadXYZ(base, q_index);
                for (std::size_t c = 0; c < vd; ++c) {
                    loaded[c] = v.elems[c];
                }
                builder.CreateBr(merge);
                auto* ok2_block = builder.GetInsertBlock();

                builder.SetInsertPoint(zero);
                builder.CreateBr(merge);
                auto* zero_block = builder.GetInsertBlock();

                builder.SetInsertPoint(merge);
                CodeValue out = makeZero(out_shape);
                for (std::size_t c = 0; c < vd; ++c) {
                    auto* phi = builder.CreatePHI(f64, 2, "field.v" + std::to_string(c));
                    phi->addIncoming(f64c(0.0), zero_block);
                    phi->addIncoming(loaded[c], ok2_block);
                    out.elems[c] = phi;
                }
                return out;
            }

            throw std::runtime_error("LLVMGen: DiscreteField/StateField unsupported shape");
        };

        auto evalDiscreteOrStateFieldHistoryK = [&](const SideView& side,
                                                    bool plus_side,
                                                    const Shape& out_shape,
                                                    int fid,
                                                    int k,
                                                    llvm::Value* q_index) -> CodeValue {
            auto* entry = fieldEntryPtrFor(plus_side, fid);
            auto* entry_is_null = builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));

            const std::string tag =
                std::string("field_hist") + (plus_side ? "_p" : "_m") + "_f" + std::to_string(fid) + "_k" + std::to_string(k);

            auto* ok_entry = llvm::BasicBlock::Create(*ctx, tag + ".entry.ok", fn);
            auto* zero = llvm::BasicBlock::Create(*ctx, tag + ".zero", fn);
            auto* merge = llvm::BasicBlock::Create(*ctx, tag + ".merge", fn);

            builder.CreateCondBr(entry_is_null, zero, ok_entry);

            builder.SetInsertPoint(ok_entry);
            auto* history_count = loadU32(entry, ABIV3::field_entry_history_count_off);
            auto* has_k = builder.CreateICmpUGE(history_count, builder.getInt32(static_cast<std::uint32_t>(k)));
            auto* ok_k = llvm::BasicBlock::Create(*ctx, tag + ".k.ok", fn);
            builder.CreateCondBr(has_k, ok_k, zero);

            if (out_shape.kind == Shape::Kind::Scalar) {
                llvm::Value* loaded = f64c(0.0);

                builder.SetInsertPoint(ok_k);
                auto* base = loadPtr(entry, ABIV3::field_entry_history_values_off);
                auto* base_is_null = builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                auto* ok_base = llvm::BasicBlock::Create(*ctx, tag + ".base.ok", fn);
                builder.CreateCondBr(base_is_null, zero, ok_base);

                builder.SetInsertPoint(ok_base);
                auto* k_minus1 = builder.getInt32(static_cast<std::uint32_t>(k - 1));
                auto* off = builder.CreateMul(k_minus1, side.n_qpts);
                auto* idx = builder.CreateAdd(off, q_index);
                auto* idx64 = builder.CreateZExt(idx, i64);
                loaded = loadRealPtrAt(base, idx64);
                builder.CreateBr(merge);
                auto* ok_block = builder.GetInsertBlock();

                builder.SetInsertPoint(zero);
                builder.CreateBr(merge);
                auto* zero_block = builder.GetInsertBlock();

                builder.SetInsertPoint(merge);
                auto* phi = builder.CreatePHI(f64, 2, tag + ".phi");
                phi->addIncoming(f64c(0.0), zero_block);
                phi->addIncoming(loaded, ok_block);
                return makeScalar(phi);
            }

            if (out_shape.kind == Shape::Kind::Vector) {
                const auto vd = static_cast<std::size_t>(out_shape.dims[0]);
                std::array<llvm::Value*, 3> loaded{f64c(0.0), f64c(0.0), f64c(0.0)};

                builder.SetInsertPoint(ok_k);
                auto* base = loadPtr(entry, ABIV3::field_entry_history_vector_values_xyz_off);
                auto* base_is_null = builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                auto* ok_base = llvm::BasicBlock::Create(*ctx, tag + ".base.ok", fn);
                builder.CreateCondBr(base_is_null, zero, ok_base);

                builder.SetInsertPoint(ok_base);
                auto* n_qpts64 = builder.CreateZExt(side.n_qpts, i64);
                auto* scale = builder.getInt64(static_cast<std::uint64_t>(3u * static_cast<std::uint32_t>(k - 1)));
                auto* off64 = builder.CreateMul(n_qpts64, scale);
                auto* base_k = builder.CreateGEP(f64, base, off64);
                const auto v = loadVec3FromQ(base_k, q_index);
                for (std::size_t c = 0; c < vd; ++c) {
                    loaded[c] = v[c];
                }
                builder.CreateBr(merge);
                auto* ok_block = builder.GetInsertBlock();

                builder.SetInsertPoint(zero);
                builder.CreateBr(merge);
                auto* zero_block = builder.GetInsertBlock();

                builder.SetInsertPoint(merge);
                CodeValue out = makeZero(out_shape);
                for (std::size_t c = 0; c < vd; ++c) {
                    auto* phi = builder.CreatePHI(f64, 2, tag + ".phi" + std::to_string(c));
                    phi->addIncoming(f64c(0.0), zero_block);
                    phi->addIncoming(loaded[c], ok_block);
                    out.elems[c] = phi;
                }
                return out;
            }

            throw std::runtime_error("LLVMGen: DiscreteField/StateField history unsupported shape");
        };

	        auto evalKernelIRSingleValue = [&](const LoweredTerm& term,
	                                           llvm::Value* q_index,
	                                           llvm::Value* i_index,
	                                           llvm::Value* j_index,
	                                           const SideView& side,
	                                           const std::vector<llvm::Value*>* indexed_env,
	                                           const std::vector<CodeValue>* cached) -> CodeValue {
	            std::vector<CodeValue> values;
	            values.resize(term.ir.ops.size());
	            const auto use_counts = kernelIRUseCounts(term.ir);

            auto getChild = [&](const KernelIROp& op, std::size_t k) -> const CodeValue& {
                const auto c = term.ir.children[static_cast<std::size_t>(op.first_child) + k];
                return values[c];
            };

            for (std::size_t op_idx = 0; op_idx < term.ir.ops.size(); ++op_idx) {
                if (cached != nullptr && term.dep_mask[op_idx] == 0u) {
                    values[op_idx] = (*cached)[op_idx];
                    continue;
                }

                const auto& op = term.ir.ops[op_idx];
                const auto& shape = term.shapes[op_idx];

                switch (op.type) {
                    case FormExprType::Constant: {
                        const double v = std::bit_cast<double>(op.imm0);
                        values[op_idx] = makeScalar(f64c(v));
                        break;
                    }

                    case FormExprType::ParameterRef: {
                        auto* idx64 = builder.getInt64(op.imm0);
                        values[op_idx] = makeScalar(loadRealPtrAt(side.jit_constants, idx64));
                        break;
                    }

                    case FormExprType::BoundaryIntegralRef: {
                        auto* idx64 = builder.getInt64(op.imm0);
                        values[op_idx] = makeScalar(loadRealPtrAt(side.coupled_integrals, idx64));
                        break;
                    }

                    case FormExprType::AuxiliaryStateRef: {
                        auto* idx64 = builder.getInt64(op.imm0);
                        values[op_idx] = makeScalar(loadRealPtrAt(side.coupled_aux, idx64));
                        break;
                    }

                    case FormExprType::DiscreteField:
                    case FormExprType::StateField: {
                        const int fid = unpackFieldIdImm1(op.imm1);
                        if (op.type == FormExprType::StateField && fid == 0xffff) {
                            values[op_idx] = evalCurrentSolution(side, shape, q_index);
                        } else {
                            values[op_idx] = evalDiscreteOrStateField(/*plus_side=*/false, shape, fid, q_index);
                        }
                        break;
                    }

                    case FormExprType::Coefficient: {
                        values[op_idx] = evalExternalCoefficient(side, q_index, shape, op.imm0);
                        break;
                    }

                    case FormExprType::Constitutive: {
                        values[op_idx] = evalExternalConstitutiveOutput(
                            side,
                            q_index,
                            builder.getInt32(static_cast<std::uint32_t>(external::TraceSideV1::Minus)),
                            op.imm0,
                            /*output_index=*/0u,
                            term.ir,
                            op,
                            values,
                            shape);
                        break;
                    }

                    case FormExprType::ConstitutiveOutput: {
                        if (op.child_count != 1u) {
                            throw std::runtime_error("LLVMGen: ConstitutiveOutput expects exactly 1 child");
                        }

                        const auto out_idx_i64 = static_cast<std::int64_t>(op.imm0);
                        if (out_idx_i64 < 0) {
                            throw std::runtime_error("LLVMGen: ConstitutiveOutput has negative output index");
                        }

                        const auto child_call_idx =
                            term.ir.children[static_cast<std::size_t>(op.first_child)];

                        // Fast path: output(0) is equivalent to the call result.
                        if (out_idx_i64 == 0) {
                            values[op_idx] = values[child_call_idx];
                            break;
                        }

                        const auto& call_op = term.ir.ops[child_call_idx];
                        if (call_op.type != FormExprType::Constitutive) {
                            throw std::runtime_error("LLVMGen: ConstitutiveOutput child must be Constitutive");
                        }

                        values[op_idx] = evalExternalConstitutiveOutput(
                            side,
                            q_index,
                            builder.getInt32(static_cast<std::uint32_t>(external::TraceSideV1::Minus)),
                            call_op.imm0,
                            static_cast<std::uint32_t>(out_idx_i64),
                            term.ir,
                            call_op,
                            values,
                            shape);
                        break;
                    }

                    case FormExprType::Time:
                        values[op_idx] = makeScalar(side.time);
                        break;

                    case FormExprType::TimeStep:
                        values[op_idx] = makeScalar(side.dt);
                        break;

                    case FormExprType::EffectiveTimeStep:
                        values[op_idx] = makeScalar(loadEffectiveDt(side));
                        break;

                    case FormExprType::CellDiameter:
                        values[op_idx] = makeScalar(side.cell_diameter);
                        break;

                    case FormExprType::CellVolume:
                        values[op_idx] = makeScalar(side.cell_volume);
                        break;

                    case FormExprType::FacetArea:
                        values[op_idx] = makeScalar(side.facet_area);
                        break;

                    case FormExprType::CellDomainId:
                        values[op_idx] = makeScalar(builder.CreateSIToFP(side.cell_domain_id, f64));
                        break;

                    case FormExprType::Coordinate:
                        values[op_idx] =
                            loadXYZDimFromSide(side, side.physical_points_xyz, q_index, shape.dims[0],
                                               side.interleaved_qpoint_geometry_physical_offset);
                        break;

                    case FormExprType::ReferenceCoordinate:
                        values[op_idx] = loadXYZDim(side.quad_points_xyz, q_index, shape.dims[0]);
                        break;

                    case FormExprType::Normal:
                        values[op_idx] =
                            loadXYZDimFromSide(side, side.normals_xyz, q_index, shape.dims[0],
                                               side.interleaved_qpoint_geometry_normal_offset);
                        break;

                    case FormExprType::Jacobian:
                        values[op_idx] =
                            loadMatDimFromSide(side, side.jacobians, q_index, shape.dims[0],
                                               side.interleaved_qpoint_geometry_jacobian_offset);
                        break;

                    case FormExprType::JacobianInverse:
                        values[op_idx] =
                            loadMatDimFromSide(side, side.inverse_jacobians, q_index, shape.dims[0],
                                               side.interleaved_qpoint_geometry_inverse_jacobian_offset);
                        break;

                    case FormExprType::Identity: {
                        CodeValue out = makeZero(shape);
                        const auto n = static_cast<std::size_t>(shape.dims[0]);
                        for (std::size_t d = 0; d < n; ++d) {
                            out.elems[d * n + d] = f64c(1.0);
                        }
                        values[op_idx] = out;
                        break;
                    }

                    case FormExprType::JacobianDeterminant: {
                        auto* q64 = builder.CreateZExt(q_index, i64);
                        values[op_idx] =
                            makeScalar(loadScalarFromSide(side, side.jacobian_dets, q_index,
                                                          side.interleaved_qpoint_geometry_det_offset));
                        break;
                    }

                    case FormExprType::TestFunction: {
                        if (shape.kind == Shape::Kind::Scalar) {
                            values[op_idx] = makeScalar(loadBasisScalar(side, side.test_basis_values, i_index, q_index));
                            break;
                        }
                        if (shape.kind != Shape::Kind::Vector) {
                            throw std::runtime_error("LLVMGen: TestFunction unsupported shape");
                        }

                        const auto vd = static_cast<std::size_t>(shape.dims[0]);
                        auto* uses_vec_basis = builder.CreateICmpNE(side.test_uses_vector_basis, builder.getInt32(0));
                        auto* vb = llvm::BasicBlock::Create(*ctx, "test.vec_basis", fn);
                        auto* sb = llvm::BasicBlock::Create(*ctx, "test.scalar_basis", fn);
                        auto* merge = llvm::BasicBlock::Create(*ctx, "test.merge", fn);

                        builder.CreateCondBr(uses_vec_basis, vb, sb);

                        std::array<llvm::Value*, 3> vb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
                        std::array<llvm::Value*, 3> sb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};

                        builder.SetInsertPoint(vb);
                        {
                            const auto v = loadVec3FromTable(side.test_basis_vector_values_xyz, side.n_qpts, i_index, q_index);
                            for (std::size_t c = 0; c < vd; ++c) {
                                vb_vals[c] = v[c];
                            }
                            builder.CreateBr(merge);
                        }
                        auto* vb_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(sb);
                        {
                            const auto dofs_per_comp = builder.CreateUDiv(side.n_test_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
                            const auto comp = builder.CreateUDiv(i_index, dofs_per_comp);
                            const auto phi = loadBasisScalar(side, side.test_basis_values, i_index, q_index);
                            for (std::size_t c = 0; c < vd; ++c) {
                                auto* is_c = builder.CreateICmpEQ(comp, builder.getInt32(static_cast<std::uint32_t>(c)));
                                sb_vals[c] = builder.CreateSelect(is_c, phi, f64c(0.0));
                            }
                            builder.CreateBr(merge);
                        }
                        auto* sb_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(merge);
                        CodeValue out = makeZero(shape);
                        for (std::size_t c = 0; c < vd; ++c) {
                            auto* phi = builder.CreatePHI(f64, 2, "test.phi" + std::to_string(c));
                            phi->addIncoming(vb_vals[c], vb_block);
                            phi->addIncoming(sb_vals[c], sb_block);
                            out.elems[c] = phi;
                        }
                        values[op_idx] = out;
                        break;
                    }

                    case FormExprType::TrialFunction: {
                        if (is_residual) {
                            // Residual: TrialFunction represents the current solution u_h(x_q).
                            values[op_idx] = evalCurrentSolution(side, shape, q_index);
                            break;
                        }
                        if (shape.kind == Shape::Kind::Scalar) {
                            values[op_idx] = makeScalar(loadBasisScalar(side, side.trial_basis_values, j_index, q_index));
                            break;
                        }
                        if (shape.kind != Shape::Kind::Vector) {
                            throw std::runtime_error("LLVMGen: TrialFunction unsupported shape");
                        }

                        const auto vd = static_cast<std::size_t>(shape.dims[0]);
                        auto* uses_vec_basis = builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
                        auto* vb = llvm::BasicBlock::Create(*ctx, "trial.vec_basis", fn);
                        auto* sb = llvm::BasicBlock::Create(*ctx, "trial.scalar_basis", fn);
                        auto* merge = llvm::BasicBlock::Create(*ctx, "trial.merge", fn);

                        builder.CreateCondBr(uses_vec_basis, vb, sb);

                        std::array<llvm::Value*, 3> vb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
                        std::array<llvm::Value*, 3> sb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};

                        builder.SetInsertPoint(vb);
                        {
                            const auto v = loadVec3FromTable(side.trial_basis_vector_values_xyz, side.n_qpts, j_index, q_index);
                            for (std::size_t c = 0; c < vd; ++c) {
                                vb_vals[c] = v[c];
                            }
                            builder.CreateBr(merge);
                        }
                        auto* vb_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(sb);
                        {
                            const auto dofs_per_comp = builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
                            const auto comp = builder.CreateUDiv(j_index, dofs_per_comp);
                            const auto phi = loadBasisScalar(side, side.trial_basis_values, j_index, q_index);
                            for (std::size_t c = 0; c < vd; ++c) {
                                auto* is_c = builder.CreateICmpEQ(comp, builder.getInt32(static_cast<std::uint32_t>(c)));
                                sb_vals[c] = builder.CreateSelect(is_c, phi, f64c(0.0));
                            }
                            builder.CreateBr(merge);
                        }
                        auto* sb_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(merge);
                        CodeValue out = makeZero(shape);
                        for (std::size_t c = 0; c < vd; ++c) {
                            auto* phi = builder.CreatePHI(f64, 2, "trial.phi" + std::to_string(c));
                            phi->addIncoming(vb_vals[c], vb_block);
                            phi->addIncoming(sb_vals[c], sb_block);
                            out.elems[c] = phi;
                        }
                        values[op_idx] = out;
                        break;
                    }

                    case FormExprType::PreviousSolutionRef: {
                        const int k = static_cast<int>(static_cast<std::int64_t>(op.imm0));
                        values[op_idx] = evalPreviousSolution(side, shape, k, q_index);
                        break;
                    }

                    case FormExprType::Gradient: {
                        const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                        const auto& kid = term.ir.ops[child_idx];
                        if (kid.type == FormExprType::TestFunction) {
                            if (shape.kind == Shape::Kind::Vector) {
                                const auto v = loadVec3FromTableQMajor(side.test_phys_grads_xyz, side.n_test_dofs, i_index, q_index);
                                values[op_idx] = makeVector(shape.dims[0], v[0], v[1], v[2]);
                                break;
                            }
                            if (shape.kind == Shape::Kind::Matrix) {
                                const auto vd = static_cast<std::size_t>(shape.dims[0]);
                                const auto dim = static_cast<std::size_t>(shape.dims[1]);
                                CodeValue out = makeZero(shape);
                                const auto dofs_per_comp = builder.CreateUDiv(side.n_test_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
                                const auto comp = builder.CreateUDiv(i_index, dofs_per_comp);
                                const auto g = loadVec3FromTableQMajor(side.test_phys_grads_xyz, side.n_test_dofs, i_index, q_index);
                                for (std::size_t r = 0; r < vd; ++r) {
                                    auto* is_r = builder.CreateICmpEQ(comp, builder.getInt32(static_cast<std::uint32_t>(r)));
                                    for (std::size_t d = 0; d < dim; ++d) {
                                        out.elems[r * dim + d] = builder.CreateSelect(is_r, g[d], f64c(0.0));
                                    }
                                }
                                values[op_idx] = out;
                                break;
                            }
                            throw std::runtime_error("LLVMGen: grad(TestFunction) unsupported shape");
                        }
	                        if (kid.type == FormExprType::TrialFunction) {
	                            if (is_residual) {
	                                auto* coeffs = side.solution_coefficients;
	                                if (shape.kind == Shape::Kind::Vector) {
	                                    const auto dim = static_cast<std::size_t>(shape.dims[0]);
	                                    const auto sums = emitReduceSum(side.n_trial_dofs, "grad_u", dim, [&](llvm::Value* j) {
	                                        auto* j64 = builder.CreateZExt(j, i64);
	                                        auto* cj = loadRealPtrAt(coeffs, j64);
	                                        const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                        std::vector<llvm::Value*> terms;
	                                        terms.reserve(dim);
	                                        for (std::size_t d = 0; d < dim; ++d) {
	                                            terms.push_back(builder.CreateFMul(cj, g[d]));
	                                        }
	                                        return terms;
	                                    });
	                                    auto* x = sums[0];
	                                    auto* y = (dim > 1u) ? sums[1] : f64c(0.0);
	                                    auto* z = (dim > 2u) ? sums[2] : f64c(0.0);
	                                    values[op_idx] = makeVector(static_cast<std::uint32_t>(dim), x, y, z);
	                                    break;
	                                }
	                                if (shape.kind == Shape::Kind::Matrix) {
	                                    const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                                    const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                                    CodeValue out = makeZero(shape);
	                                    auto* dofs_per_comp =
	                                        builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                                    for (std::size_t comp = 0; comp < vd; ++comp) {
	                                        const auto acc = emitReduceSum(dofs_per_comp, "grad_u_c" + std::to_string(comp), dim, [&](llvm::Value* jj) {
	                                            auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                            auto* j = builder.CreateAdd(base, jj);
	                                            auto* j64 = builder.CreateZExt(j, i64);
	                                            auto* cj = loadRealPtrAt(coeffs, j64);
	                                            const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                            std::vector<llvm::Value*> terms;
	                                            terms.reserve(dim);
	                                            for (std::size_t d = 0; d < dim; ++d) {
	                                                terms.push_back(builder.CreateFMul(cj, g[d]));
	                                            }
	                                            return terms;
	                                        });
	                                        for (std::size_t d = 0; d < dim; ++d) {
	                                            out.elems[comp * dim + d] = acc[d];
	                                        }
	                                    }
	                                    values[op_idx] = out;
                                    break;
                                }
                                throw std::runtime_error("LLVMGen: grad(u) unsupported shape");
                            }
                            if (shape.kind == Shape::Kind::Vector) {
                                const auto v = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j_index, q_index);
                                values[op_idx] = makeVector(shape.dims[0], v[0], v[1], v[2]);
                                break;
                            }
                            if (shape.kind == Shape::Kind::Matrix) {
                                const auto vd = static_cast<std::size_t>(shape.dims[0]);
                                const auto dim = static_cast<std::size_t>(shape.dims[1]);
                                CodeValue out = makeZero(shape);
                                const auto dofs_per_comp = builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
                                const auto comp = builder.CreateUDiv(j_index, dofs_per_comp);
                                const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j_index, q_index);
                                for (std::size_t r = 0; r < vd; ++r) {
                                    auto* is_r = builder.CreateICmpEQ(comp, builder.getInt32(static_cast<std::uint32_t>(r)));
                                    for (std::size_t d = 0; d < dim; ++d) {
                                        out.elems[r * dim + d] = builder.CreateSelect(is_r, g[d], f64c(0.0));
                                    }
                                }
                                values[op_idx] = out;
                                break;
                            }
                            throw std::runtime_error("LLVMGen: grad(TrialFunction) unsupported shape");
                        }
	                        if (kid.type == FormExprType::PreviousSolutionRef) {
	                            const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
	                            if (shape.kind == Shape::Kind::Vector) {
	                                const auto dim = static_cast<std::size_t>(shape.dims[0]);
	                                auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
	                                const auto sums =
	                                    emitReduceSum(side.n_trial_dofs, "prev_grad" + std::to_string(k), dim, [&](llvm::Value* j) {
	                                        auto* j64 = builder.CreateZExt(j, i64);
	                                        auto* cj = loadRealPtrAt(coeffs, j64);
	                                        const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                        std::vector<llvm::Value*> terms;
	                                        terms.reserve(dim);
	                                        for (std::size_t d = 0; d < dim; ++d) {
	                                            terms.push_back(builder.CreateFMul(cj, g[d]));
	                                        }
	                                        return terms;
	                                    });
	                                auto* x = sums[0];
	                                auto* y = (dim > 1u) ? sums[1] : f64c(0.0);
	                                auto* z = (dim > 2u) ? sums[2] : f64c(0.0);
	                                values[op_idx] = makeVector(static_cast<std::uint32_t>(dim), x, y, z);
	                                break;
	                            }
	                            if (shape.kind == Shape::Kind::Matrix) {
	                                const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                                const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                                CodeValue out = makeZero(shape);
	                                auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
	                                auto* dofs_per_comp =
	                                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                                for (std::size_t comp = 0; comp < vd; ++comp) {
	                                    const auto acc = emitReduceSum(dofs_per_comp,
	                                                                   "prev_grad" + std::to_string(k) + "_c" + std::to_string(comp),
	                                                                   dim,
	                                                                   [&](llvm::Value* jj) {
	                                        auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                        auto* j = builder.CreateAdd(base, jj);
	                                        auto* j64 = builder.CreateZExt(j, i64);
	                                        auto* cj = loadRealPtrAt(coeffs, j64);
	                                        const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                        std::vector<llvm::Value*> terms;
	                                        terms.reserve(dim);
	                                        for (std::size_t d = 0; d < dim; ++d) {
	                                            terms.push_back(builder.CreateFMul(cj, g[d]));
	                                        }
	                                        return terms;
	                                    });
	                                    for (std::size_t d = 0; d < dim; ++d) {
	                                        out.elems[comp * dim + d] = acc[d];
	                                    }
	                                }
                                values[op_idx] = out;
                                break;
                            }
                            throw std::runtime_error("LLVMGen: grad(PreviousSolutionRef) unsupported shape");
                        }
	                        if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
	                            const int fid = unpackFieldIdImm1(kid.imm1);
	                            if (kid.type == FormExprType::StateField && fid == 0xffff) {
	                                auto* coeffs = side.solution_coefficients;
	                                if (shape.kind == Shape::Kind::Vector) {
	                                    const auto dim = static_cast<std::size_t>(shape.dims[0]);
	                                    const auto sums =
	                                        emitReduceSum(side.n_trial_dofs, "grad_state_u", dim, [&](llvm::Value* j) {
	                                            auto* j64 = builder.CreateZExt(j, i64);
	                                            auto* cj = loadRealPtrAt(coeffs, j64);
	                                            const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                            std::vector<llvm::Value*> terms;
	                                            terms.reserve(dim);
	                                            for (std::size_t d = 0; d < dim; ++d) {
	                                                terms.push_back(builder.CreateFMul(cj, g[d]));
	                                            }
	                                            return terms;
	                                        });
	                                    auto* x = sums[0];
	                                    auto* y = (dim > 1u) ? sums[1] : f64c(0.0);
	                                    auto* z = (dim > 2u) ? sums[2] : f64c(0.0);
	                                    values[op_idx] = makeVector(static_cast<std::uint32_t>(dim), x, y, z);
	                                    break;
	                                }
	                                if (shape.kind == Shape::Kind::Matrix) {
	                                    const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                                    const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                                    CodeValue out = makeZero(shape);
	                                    auto* dofs_per_comp =
	                                        builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                                    for (std::size_t comp = 0; comp < vd; ++comp) {
	                                        const auto acc = emitReduceSum(
	                                            dofs_per_comp, "grad_state_u_c" + std::to_string(comp), dim, [&](llvm::Value* jj) {
	                                            auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                            auto* j = builder.CreateAdd(base, jj);
	                                            auto* j64 = builder.CreateZExt(j, i64);
	                                            auto* cj = loadRealPtrAt(coeffs, j64);
	                                            const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                            std::vector<llvm::Value*> terms;
	                                            terms.reserve(dim);
	                                            for (std::size_t d = 0; d < dim; ++d) {
	                                                terms.push_back(builder.CreateFMul(cj, g[d]));
	                                            }
	                                            return terms;
	                                        });
	                                        for (std::size_t d = 0; d < dim; ++d) {
	                                            out.elems[comp * dim + d] = acc[d];
	                                        }
	                                    }
	                                    values[op_idx] = out;
                                    break;
                                }
                                throw std::runtime_error("LLVMGen: grad(StateField(INVALID)) unsupported shape");
                            }
                            if (shape.kind == Shape::Kind::Vector) {
                                const auto dim = static_cast<std::size_t>(shape.dims[0]);
                                auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
                                auto* entry_is_null =
                                    builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok = llvm::BasicBlock::Create(*ctx, "field_grad.ok", fn);
                                auto* zero = llvm::BasicBlock::Create(*ctx, "field_grad.zero", fn);
                                auto* merge = llvm::BasicBlock::Create(*ctx, "field_grad.merge", fn);

                                builder.CreateCondBr(entry_is_null, zero, ok);

                                std::array<llvm::Value*, 3> loaded{f64c(0.0), f64c(0.0), f64c(0.0)};
                                builder.SetInsertPoint(ok);
                                auto* base = loadPtr(entry, ABIV3::field_entry_gradients_xyz_off);
                                auto* base_is_null =
                                    builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_grad.xyz.ok", fn);
                                builder.CreateCondBr(base_is_null, zero, ok2);

                                builder.SetInsertPoint(ok2);
                                const auto v = loadXYZ(base, q_index);
                                loaded = {v.elems[0], v.elems[1], v.elems[2]};
                                builder.CreateBr(merge);
                                auto* ok2_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(zero);
                                builder.CreateBr(merge);
                                auto* zero_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(merge);
                                std::array<llvm::Value*, 3> outv{f64c(0.0), f64c(0.0), f64c(0.0)};
                                for (std::size_t d = 0; d < dim; ++d) {
                                    auto* phi = builder.CreatePHI(f64, 2, "field_grad.v" + std::to_string(d));
                                    phi->addIncoming(f64c(0.0), zero_block);
                                    phi->addIncoming(loaded[d], ok2_block);
                                    outv[d] = phi;
                                }
                                values[op_idx] = makeVector(static_cast<std::uint32_t>(dim), outv[0], outv[1], outv[2]);
                                break;
                            }
                            if (shape.kind == Shape::Kind::Matrix) {
                                const auto vd = static_cast<std::size_t>(shape.dims[0]);
                                const auto dim = static_cast<std::size_t>(shape.dims[1]);
                                auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
                                auto* entry_is_null =
                                    builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok = llvm::BasicBlock::Create(*ctx, "field_jac.ok", fn);
                                auto* zero = llvm::BasicBlock::Create(*ctx, "field_jac.zero", fn);
                                auto* merge = llvm::BasicBlock::Create(*ctx, "field_jac.merge", fn);

                                builder.CreateCondBr(entry_is_null, zero, ok);

                                CodeValue loaded = makeZero(matrixShape(3u, 3u));
                                builder.SetInsertPoint(ok);
                                auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                                auto* base_is_null =
                                    builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_jac.mat.ok", fn);
                                builder.CreateCondBr(base_is_null, zero, ok2);

                                builder.SetInsertPoint(ok2);
                                loaded = loadMat3FromQ(base, q_index);
                                builder.CreateBr(merge);
                                auto* ok2_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(zero);
                                builder.CreateBr(merge);
                                auto* zero_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(merge);
                                CodeValue out = makeZero(shape);
                                for (std::size_t r = 0; r < vd; ++r) {
                                    for (std::size_t c = 0; c < dim; ++c) {
                                        auto* phi = builder.CreatePHI(f64, 2, "field_jac.a");
                                        phi->addIncoming(f64c(0.0), zero_block);
                                        phi->addIncoming(loaded.elems[r * 3 + c], ok2_block);
                                        out.elems[r * dim + c] = phi;
                                    }
                                }
                                values[op_idx] = out;
                                break;
                            }
                            throw std::runtime_error("LLVMGen: grad(DiscreteField) unsupported shape");
                        }
                        if (kid.type == FormExprType::TimeDerivative) {
                            const int order = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
                            auto* coeff0 = loadDtCoeff0(side, order);
                            const auto dt_child_idx =
                                term.ir.children[static_cast<std::size_t>(kid.first_child)];
                            const auto& dt_child = term.ir.ops[dt_child_idx];

	                            auto gradTrialBasis = [&]() -> CodeValue {
	                                CodeValue g = makeZero(shape);
	                                if (shape.kind == Shape::Kind::Vector) {
	                                    const auto v = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j_index, q_index);
	                                    g = makeVector(shape.dims[0], v[0], v[1], v[2]);
	                                    return g;
	                                }
	                                if (shape.kind == Shape::Kind::Matrix) {
	                                    const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                                    const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                                    const auto dofs_per_comp = builder.CreateUDiv(
	                                        side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                                    const auto comp = builder.CreateUDiv(j_index, dofs_per_comp);
	                                    const auto gg = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j_index, q_index);
	                                    for (std::size_t r = 0; r < vd; ++r) {
	                                        auto* is_r = builder.CreateICmpEQ(comp, builder.getInt32(static_cast<std::uint32_t>(r)));
	                                        for (std::size_t d = 0; d < dim; ++d) {
	                                            g.elems[r * dim + d] = builder.CreateSelect(is_r, gg[d], f64c(0.0));
	                                        }
	                                    }
	                                    return g;
	                                }
	                                throw std::runtime_error("LLVMGen: grad(dt(TrialFunction)) unsupported shape");
	                            };

		                            auto gradCurrentSolution = [&]() -> CodeValue {
		                                auto* coeffs = side.solution_coefficients;
		                                if (shape.kind == Shape::Kind::Vector) {
		                                    const auto dim = static_cast<std::size_t>(shape.dims[0]);
		                                    const auto sums =
		                                        emitReduceSum(side.n_trial_dofs, "grad_dt_u", dim, [&](llvm::Value* j) {
		                                            auto* j64 = builder.CreateZExt(j, i64);
		                                            auto* cj = loadRealPtrAt(coeffs, j64);
		                                            const auto g =
		                                                loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
		                                            std::vector<llvm::Value*> terms;
		                                            terms.reserve(dim);
		                                            for (std::size_t d = 0; d < dim; ++d) {
		                                                terms.push_back(builder.CreateFMul(cj, g[d]));
		                                            }
		                                            return terms;
		                                        });
		                                    auto* x = sums[0];
		                                    auto* y = (dim > 1u) ? sums[1] : f64c(0.0);
		                                    auto* z = (dim > 2u) ? sums[2] : f64c(0.0);
		                                    return makeVector(static_cast<std::uint32_t>(dim), x, y, z);
		                                }
		                                if (shape.kind == Shape::Kind::Matrix) {
		                                    const auto vd = static_cast<std::size_t>(shape.dims[0]);
		                                    const auto dim = static_cast<std::size_t>(shape.dims[1]);
		                                    CodeValue out = makeZero(shape);
		                                    auto* dofs_per_comp =
		                                        builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
		                                    for (std::size_t comp = 0; comp < vd; ++comp) {
		                                        const auto acc =
		                                            emitReduceSum(dofs_per_comp, "grad_dt_u_c" + std::to_string(comp), dim, [&](llvm::Value* jj) {
		                                            auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
		                                            auto* j = builder.CreateAdd(base, jj);
		                                            auto* j64 = builder.CreateZExt(j, i64);
		                                            auto* cj = loadRealPtrAt(coeffs, j64);
		                                            const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
		                                            std::vector<llvm::Value*> terms;
		                                            terms.reserve(dim);
		                                            for (std::size_t d = 0; d < dim; ++d) {
		                                                terms.push_back(builder.CreateFMul(cj, g[d]));
		                                            }
		                                            return terms;
		                                        });
		                                        for (std::size_t d = 0; d < dim; ++d) {
		                                            out.elems[comp * dim + d] = acc[d];
		                                        }
		                                    }
	                                    return out;
	                                }
	                                throw std::runtime_error("LLVMGen: grad(dt(u)) unsupported shape");
	                            };

	                            auto gradDiscreteOrStateField = [&](int fid) -> CodeValue {
	                                if (shape.kind == Shape::Kind::Vector) {
	                                    const auto dim = static_cast<std::size_t>(shape.dims[0]);
	                                    auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
	                                    auto* entry_is_null =
	                                        builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                    auto* ok = llvm::BasicBlock::Create(*ctx, "field_grad.ok", fn);
                                    auto* zero = llvm::BasicBlock::Create(*ctx, "field_grad.zero", fn);
                                    auto* merge = llvm::BasicBlock::Create(*ctx, "field_grad.merge", fn);

                                    builder.CreateCondBr(entry_is_null, zero, ok);

                                    std::array<llvm::Value*, 3> loaded{f64c(0.0), f64c(0.0), f64c(0.0)};
                                    builder.SetInsertPoint(ok);
	                                    auto* base = loadPtr(entry, ABIV3::field_entry_gradients_xyz_off);
                                    auto* base_is_null =
                                        builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                    auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_grad.ok2", fn);
                                    builder.CreateCondBr(base_is_null, zero, ok2);

                                    builder.SetInsertPoint(ok2);
                                    loaded = loadVec3FromQ(base, q_index);
                                    builder.CreateBr(merge);
                                    auto* ok2_block = builder.GetInsertBlock();

                                    builder.SetInsertPoint(zero);
                                    builder.CreateBr(merge);
                                    auto* zero_block = builder.GetInsertBlock();

	                                    builder.SetInsertPoint(merge);
	                                    std::array<llvm::Value*, 3> outv{f64c(0.0), f64c(0.0), f64c(0.0)};
	                                    for (std::size_t d = 0; d < dim; ++d) {
	                                        auto* phi = builder.CreatePHI(f64, 2, "field_grad.g");
	                                        phi->addIncoming(f64c(0.0), zero_block);
	                                        phi->addIncoming(loaded[d], ok2_block);
	                                        outv[d] = phi;
	                                    }
	                                    return makeVector(static_cast<std::uint32_t>(dim), outv[0], outv[1], outv[2]);
	                                }
	                                if (shape.kind == Shape::Kind::Matrix) {
	                                    const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                                    const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                                    auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
	                                    auto* entry_is_null =
	                                        builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                    auto* ok = llvm::BasicBlock::Create(*ctx, "field_jac.ok", fn);
                                    auto* zero = llvm::BasicBlock::Create(*ctx, "field_jac.zero", fn);
                                    auto* merge = llvm::BasicBlock::Create(*ctx, "field_jac.merge", fn);

                                    builder.CreateCondBr(entry_is_null, zero, ok);

                                    CodeValue loaded = makeZero(matrixShape(3u, 3u));
                                    builder.SetInsertPoint(ok);
                                    auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                                    auto* base_is_null =
                                        builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                    auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_jac.mat.ok", fn);
                                    builder.CreateCondBr(base_is_null, zero, ok2);

                                    builder.SetInsertPoint(ok2);
                                    loaded = loadMat3FromQ(base, q_index);
                                    builder.CreateBr(merge);
                                    auto* ok2_block = builder.GetInsertBlock();

                                    builder.SetInsertPoint(zero);
                                    builder.CreateBr(merge);
                                    auto* zero_block = builder.GetInsertBlock();

	                                    builder.SetInsertPoint(merge);
	                                    CodeValue out = makeZero(shape);
	                                    for (std::size_t r = 0; r < vd; ++r) {
	                                        for (std::size_t c = 0; c < dim; ++c) {
	                                            auto* phi = builder.CreatePHI(f64, 2, "field_jac.a");
	                                            phi->addIncoming(f64c(0.0), zero_block);
	                                            phi->addIncoming(loaded.elems[r * 3 + c], ok2_block);
	                                            out.elems[r * dim + c] = phi;
	                                        }
	                                    }
	                                    return out;
	                                }
	                                throw std::runtime_error("LLVMGen: grad(dt(field)) unsupported shape");
	                            };

                            if (dt_child.type == FormExprType::TrialFunction) {
                                const auto g = is_residual ? gradCurrentSolution() : gradTrialBasis();
                                values[op_idx] = mul(makeScalar(coeff0), g);
                                break;
                            }

                            if (dt_child.type == FormExprType::DiscreteField || dt_child.type == FormExprType::StateField) {
                                const int fid = unpackFieldIdImm1(dt_child.imm1);
                                const auto g = (dt_child.type == FormExprType::StateField && fid == 0xffff)
                                                   ? gradCurrentSolution()
                                                   : gradDiscreteOrStateField(fid);
                                values[op_idx] = mul(makeScalar(coeff0), g);
                                break;
                            }

                            if (dt_child.type == FormExprType::Constant) {
                                values[op_idx] = makeZero(shape);
                                break;
                            }

	                            throw std::runtime_error("LLVMGen: grad(dt(...)) operand not supported");
	                        }

                        // Gradient is linear; handle the common simplified pattern grad(0 * X) -> 0.
                        if (kid.type == FormExprType::Multiply && kid.child_count == 2u) {
                            const auto a_idx = term.ir.children[static_cast<std::size_t>(kid.first_child)];
                            const auto b_idx = term.ir.children[static_cast<std::size_t>(kid.first_child) + 1u];
                            const auto& aop = term.ir.ops[a_idx];
                            const auto& bop = term.ir.ops[b_idx];

                            auto isZeroConstant = [](const KernelIROp& op2) -> bool {
                                if (op2.type != FormExprType::Constant) {
                                    return false;
                                }
                                const double v = std::bit_cast<double>(op2.imm0);
                                return v == 0.0;
                            };

                            if (isZeroConstant(aop) || isZeroConstant(bop)) {
                                values[op_idx] = makeZero(shape);
                                break;
                            }
                        }
	                        throw std::runtime_error("LLVMGen: Gradient operand not supported");
	                    }

	                    case FormExprType::Divergence: {
	                        const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
	                        const auto& kid = term.ir.ops[child_idx];
	                        const auto vd = static_cast<std::size_t>(term.shapes[child_idx].dims[0]);

	                        // Matrix divergence (returns a vector with length = number of matrix rows).
	                        //
	                        // Currently implemented for the Navier–Stokes/Stokes VMS strong residual pattern
	                        //   div( scalar * sym(grad(u)) )
	                        // where the scalar factor is spatially constant and u is one of:
	                        //   - TrialFunction  (residual: current solution; tangent: trial basis)
	                        //   - StateField(INVALID_FIELD_ID) (current solution)
	                        //   - PreviousSolutionRef
	                        if (shape.kind == Shape::Kind::Vector) {
	                            auto isSpatiallyConstantScalar = [&](auto&& self, std::size_t idx) -> bool {
	                                const auto& op2 = term.ir.ops[idx];
	                                switch (op2.type) {
	                                    case FormExprType::Constant:
	                                    case FormExprType::ParameterRef:
	                                    case FormExprType::Time:
	                                    case FormExprType::TimeStep:
	                                    case FormExprType::EffectiveTimeStep:
	                                        return true;
	                                    case FormExprType::Negate:
	                                    case FormExprType::AbsoluteValue:
	                                    case FormExprType::Sign:
	                                    case FormExprType::Sqrt:
	                                    case FormExprType::Exp:
	                                    case FormExprType::Log:
	                                        break;
	                                    case FormExprType::Add:
	                                    case FormExprType::Subtract:
	                                    case FormExprType::Multiply:
	                                    case FormExprType::Divide:
	                                    case FormExprType::Power:
	                                    case FormExprType::Minimum:
	                                    case FormExprType::Maximum:
	                                        break;
	                                    default:
	                                        return false;
	                                }

	                                for (std::size_t k = 0; k < static_cast<std::size_t>(op2.child_count); ++k) {
	                                    const auto c =
	                                        term.ir.children[static_cast<std::size_t>(op2.first_child) + k];
	                                    if (!self(self, c)) {
	                                        return false;
	                                    }
	                                }
	                                return true;
	                            };

	                            auto traceHessBasis = [&](const CodeValue& H, std::size_t dim) -> llvm::Value* {
	                                llvm::Value* tr = H.elems[0];
	                                if (dim >= 2) tr = builder.CreateFAdd(tr, H.elems[4]);
	                                if (dim >= 3) tr = builder.CreateFAdd(tr, H.elems[8]);
	                                return tr;
	                            };

	                            auto divSymGradFromCoeffs = [&](llvm::Value* coeffs_ptr,
	                                                           std::size_t rows,
	                                                           std::size_t dim) -> CodeValue {
	                                CodeValue out = makeZero(vectorShape(static_cast<std::uint32_t>(rows)));
	                                auto* dofs_per_comp =
	                                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(rows)));

	                                const std::size_t n = std::min(rows, dim);
		                                for (std::size_t r = 0; r < rows; ++r) {
		                                    if (r >= dim) {
		                                        out.elems[r] = f64c(0.0);
		                                        continue;
		                                    }

		                                    const auto loop_name = "lap_u_c" + std::to_string(r);
		                                    auto* lap = emitReduceSumScalar(dofs_per_comp, loop_name, [&](llvm::Value* jj) -> llvm::Value* {
		                                        auto* base =
		                                            builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(r)), dofs_per_comp);
		                                        auto* j = builder.CreateAdd(base, jj);
		                                        auto* j64 = builder.CreateZExt(j, i64);
		                                        auto* cj = loadRealPtrAt(coeffs_ptr, j64);
		                                        const auto H =
		                                            loadMat3FromTable(side.trial_phys_hessians, side.n_qpts, j, q_index);
		                                        auto* tr = traceHessBasis(H, dim);
		                                        return builder.CreateFMul(cj, tr);
		                                    });

		                                    llvm::Value* gd = f64c(0.0);
		                                    for (std::size_t d = 0; d < n; ++d) {
		                                        const auto loop_name =
		                                            "gd_u_r" + std::to_string(r) + "_d" + std::to_string(d);
		                                        auto* acc = emitReduceSumScalar(dofs_per_comp, loop_name, [&](llvm::Value* jj) -> llvm::Value* {
		                                            auto* base =
		                                                builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(d)), dofs_per_comp);
		                                            auto* j = builder.CreateAdd(base, jj);
		                                            auto* j64 = builder.CreateZExt(j, i64);
		                                            auto* cj = loadRealPtrAt(coeffs_ptr, j64);
		                                            const auto H =
		                                                loadMat3FromTable(side.trial_phys_hessians, side.n_qpts, j, q_index);
		                                            const auto idxH = static_cast<std::size_t>(r * 3u + d);
		                                            return builder.CreateFMul(cj, H.elems[idxH]);
		                                        });
		                                        gd = builder.CreateFAdd(gd, acc);
		                                    }

	                                    out.elems[r] = builder.CreateFMul(f64c(0.5), builder.CreateFAdd(lap, gd));
	                                }

	                                return out;
	                            };

	                            auto divSymGradTrialBasis = [&](std::size_t rows, std::size_t dim) -> CodeValue {
	                                CodeValue out = makeZero(vectorShape(static_cast<std::uint32_t>(rows)));
	                                auto* dofs_per_comp =
	                                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(rows)));
	                                auto* comp = builder.CreateUDiv(j_index, dofs_per_comp);
	                                const auto H = loadMat3FromTable(side.trial_phys_hessians, side.n_qpts, j_index, q_index);
	                                auto* tr = traceHessBasis(H, dim);

	                                for (std::size_t r = 0; r < rows; ++r) {
	                                    if (r >= dim) {
	                                        out.elems[r] = f64c(0.0);
	                                        continue;
	                                    }

	                                    auto* is_r = builder.CreateICmpEQ(comp, builder.getInt32(static_cast<std::uint32_t>(r)));
	                                    auto* lap = builder.CreateSelect(is_r, tr, f64c(0.0));

	                                    llvm::Value* h_rc = f64c(0.0);
	                                    if (dim >= 1) {
	                                        auto* is0 = builder.CreateICmpEQ(comp, builder.getInt32(0));
	                                        h_rc = builder.CreateSelect(is0, H.elems[r * 3u + 0u], h_rc);
	                                    }
	                                    if (dim >= 2) {
	                                        auto* is1 = builder.CreateICmpEQ(comp, builder.getInt32(1));
	                                        h_rc = builder.CreateSelect(is1, H.elems[r * 3u + 1u], h_rc);
	                                    }
	                                    if (dim >= 3) {
	                                        auto* is2 = builder.CreateICmpEQ(comp, builder.getInt32(2));
	                                        h_rc = builder.CreateSelect(is2, H.elems[r * 3u + 2u], h_rc);
	                                    }

	                                    out.elems[r] = builder.CreateFMul(f64c(0.5), builder.CreateFAdd(lap, h_rc));
	                                }
	                                return out;
	                            };

	                            auto divSymGradForU = [&](std::size_t u_idx,
	                                                      std::size_t rows,
	                                                      std::size_t dim) -> CodeValue {
	                                auto divSymGradFromField = [&](int fid, std::size_t rows2, std::size_t dim2) -> CodeValue {
	                                    CodeValue out = makeZero(vectorShape(static_cast<std::uint32_t>(rows2)));

	                                    auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
	                                    auto* entry_is_null = builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
	                                    auto* ok = llvm::BasicBlock::Create(*ctx, "field_divsym.ok", fn);
	                                    auto* zero = llvm::BasicBlock::Create(*ctx, "field_divsym.zero", fn);
	                                    auto* merge = llvm::BasicBlock::Create(*ctx, "field_divsym.merge", fn);
	                                    builder.CreateCondBr(entry_is_null, zero, ok);

	                                    std::array<llvm::Value*, 3> computed{f64c(0.0), f64c(0.0), f64c(0.0)};
	                                    builder.SetInsertPoint(ok);
	                                    auto* base = loadPtr(entry, ABIV3::field_entry_component_hessians_off);
	                                    auto* base_is_null = builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
	                                    auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_divsym.ok2", fn);
	                                    builder.CreateCondBr(base_is_null, zero, ok2);

	                                    builder.SetInsertPoint(ok2);
	                                    auto* vd = loadU32(entry, ABIV3::field_entry_value_dim_off);
	                                    const std::size_t n2 = std::min(rows2, dim2);
	                                    auto* vd_ok = builder.CreateICmpUGE(vd, builder.getInt32(static_cast<std::uint32_t>(n2)));
	                                    auto* ok3 = llvm::BasicBlock::Create(*ctx, "field_divsym.ok3", fn);
	                                    builder.CreateCondBr(vd_ok, ok3, zero);

	                                    builder.SetInsertPoint(ok3);
	                                    auto loadHessElem = [&](std::size_t comp,
	                                                            std::size_t d0,
	                                                            std::size_t d1) -> llvm::Value* {
	                                        auto* q64 = builder.CreateZExt(q_index, i64);
	                                        auto* vd64 = builder.CreateZExt(vd, i64);
	                                        auto* idx = builder.CreateAdd(builder.CreateMul(q64, vd64),
	                                                                      builder.getInt64(static_cast<std::uint64_t>(comp)));
	                                        auto* base9 = builder.CreateMul(idx, builder.getInt64(9));
	                                        const auto elem = static_cast<std::uint64_t>(d0 * 3u + d1);
	                                        return loadRealPtrAt(base, builder.CreateAdd(base9, builder.getInt64(elem)));
	                                    };

	                                    for (std::size_t r = 0; r < rows2; ++r) {
	                                        if (r >= dim2) {
	                                            computed[r] = f64c(0.0);
	                                            continue;
	                                        }

	                                        llvm::Value* lap = loadHessElem(r, 0u, 0u);
	                                        if (dim2 >= 2u) lap = builder.CreateFAdd(lap, loadHessElem(r, 1u, 1u));
	                                        if (dim2 >= 3u) lap = builder.CreateFAdd(lap, loadHessElem(r, 2u, 2u));

	                                        llvm::Value* gd = f64c(0.0);
	                                        for (std::size_t d = 0; d < n2; ++d) {
	                                            gd = builder.CreateFAdd(gd, loadHessElem(d, r, d));
	                                        }

	                                        computed[r] = builder.CreateFMul(f64c(0.5), builder.CreateFAdd(lap, gd));
	                                    }

	                                    builder.CreateBr(merge);
	                                    auto* ok3_block = builder.GetInsertBlock();

	                                    builder.SetInsertPoint(zero);
	                                    builder.CreateBr(merge);
	                                    auto* zero_block = builder.GetInsertBlock();

	                                    builder.SetInsertPoint(merge);
	                                    for (std::size_t r = 0; r < rows2; ++r) {
	                                        auto* phi = builder.CreatePHI(f64, 2, "field_divsym." + std::to_string(r));
	                                        phi->addIncoming(f64c(0.0), zero_block);
	                                        phi->addIncoming(computed[r], ok3_block);
	                                        out.elems[r] = phi;
	                                    }
	                                    return out;
	                                };

	                                auto divSymGradForURec = [&](auto&& self,
	                                                             std::size_t u_idx2,
	                                                             std::size_t rows2,
	                                                             std::size_t dim2) -> CodeValue {
	                                    const auto& uop = term.ir.ops[u_idx2];
	                                    if (uop.type == FormExprType::TrialFunction) {
	                                        return is_residual ? divSymGradFromCoeffs(side.solution_coefficients, rows2, dim2)
	                                                          : divSymGradTrialBasis(rows2, dim2);
	                                    }
	                                    if (uop.type == FormExprType::DiscreteField) {
	                                        const int fid = unpackFieldIdImm1(uop.imm1);
	                                        return divSymGradFromField(fid, rows2, dim2);
	                                    }
	                                    if (uop.type == FormExprType::StateField) {
	                                        const int fid = unpackFieldIdImm1(uop.imm1);
	                                        if (fid == 0xffff) {
	                                            return divSymGradFromCoeffs(side.solution_coefficients, rows2, dim2);
	                                        }
	                                        return divSymGradFromField(fid, rows2, dim2);
	                                    }
	                                    if (uop.type == FormExprType::PreviousSolutionRef) {
	                                        const int k = static_cast<int>(static_cast<std::int64_t>(uop.imm0));
	                                        auto* coeffs_ptr = loadPrevSolutionCoeffsPtr(side, k);
	                                        return divSymGradFromCoeffs(coeffs_ptr, rows2, dim2);
	                                    }
	                                    if (uop.type == FormExprType::Negate) {
	                                        const auto a_idx = term.ir.children[static_cast<std::size_t>(uop.first_child)];
	                                        return neg(self(self, a_idx, rows2, dim2));
	                                    }
	                                    if (uop.type == FormExprType::Add || uop.type == FormExprType::Subtract) {
	                                        if (uop.child_count != 2u) {
	                                            throw std::runtime_error("LLVMGen: div(sym(grad(Add/Subtract))) expects 2 children");
	                                        }
	                                        const auto a_idx = term.ir.children[static_cast<std::size_t>(uop.first_child)];
	                                        const auto b_idx = term.ir.children[static_cast<std::size_t>(uop.first_child) + 1u];
	                                        return (uop.type == FormExprType::Add) ? add(self(self, a_idx, rows2, dim2), self(self, b_idx, rows2, dim2))
	                                                                             : sub(self(self, a_idx, rows2, dim2), self(self, b_idx, rows2, dim2));
	                                    }
	                                    if (uop.type == FormExprType::Multiply) {
	                                        if (uop.child_count != 2u) {
	                                            throw std::runtime_error("LLVMGen: div(sym(grad(Multiply))) expects 2 children");
	                                        }
	                                        const auto a_idx = term.ir.children[static_cast<std::size_t>(uop.first_child)];
	                                        const auto b_idx = term.ir.children[static_cast<std::size_t>(uop.first_child) + 1u];
	                                        const auto& ash = term.shapes[a_idx];
	                                        const auto& bsh = term.shapes[b_idx];
	                                        const bool a_scalar = (ash.kind == Shape::Kind::Scalar);
	                                        const bool b_scalar = (bsh.kind == Shape::Kind::Scalar);
	                                        const bool a_vec = (ash.kind == Shape::Kind::Vector);
	                                        const bool b_vec = (bsh.kind == Shape::Kind::Vector);
	                                        if (a_scalar && b_vec) {
	                                            if (!isSpatiallyConstantScalar(isSpatiallyConstantScalar, a_idx)) {
	                                                throw std::runtime_error("LLVMGen: div(sym(grad(s*u))) requires spatially-constant scalar s in JIT");
	                                            }
	                                            return mul(values[a_idx], self(self, b_idx, rows2, dim2));
	                                        }
	                                        if (b_scalar && a_vec) {
	                                            if (!isSpatiallyConstantScalar(isSpatiallyConstantScalar, b_idx)) {
	                                                throw std::runtime_error("LLVMGen: div(sym(grad(u*s))) requires spatially-constant scalar s in JIT");
	                                            }
	                                            return mul(values[b_idx], self(self, a_idx, rows2, dim2));
	                                        }
	                                    }
	                                    throw std::runtime_error(
	                                        "LLVMGen: div(sym(grad(u))) expects u = TrialFunction, DiscreteField, StateField(INVALID_FIELD_ID), PreviousSolutionRef, or linear scalar*vector ops");
	                                };
	                                return divSymGradForURec(divSymGradForURec, u_idx, rows, dim);
	                            };

	                            auto evalDivMatrix = [&](auto&& self, std::size_t m_idx) -> CodeValue {
	                                const auto& mop = term.ir.ops[m_idx];
	                                const auto& msh = term.shapes[m_idx];
	                                if (msh.kind != Shape::Kind::Matrix) {
	                                    throw std::runtime_error("LLVMGen: internal error: div(matrix) expected Matrix operand");
	                                }
	                                const std::size_t rows = static_cast<std::size_t>(msh.dims[0]);
	                                const std::size_t dim = static_cast<std::size_t>(msh.dims[1]);

	                                switch (mop.type) {
	                                    case FormExprType::SymmetricPart: {
	                                        const auto a_idx = term.ir.children[static_cast<std::size_t>(mop.first_child)];
	                                        const auto& aop = term.ir.ops[a_idx];
	                                        if (aop.type != FormExprType::Gradient) {
	                                            throw std::runtime_error("LLVMGen: div(sym(A)) currently only supports A=grad(u)");
	                                        }
	                                        const auto u_idx = term.ir.children[static_cast<std::size_t>(aop.first_child)];
	                                        return divSymGradForU(u_idx, rows, dim);
	                                    }
	                                    case FormExprType::Multiply: {
	                                        if (mop.child_count != 2u) {
	                                            throw std::runtime_error("LLVMGen: div(Multiply) expects 2 children");
	                                        }
	                                        const auto a_idx = term.ir.children[static_cast<std::size_t>(mop.first_child)];
	                                        const auto b_idx = term.ir.children[static_cast<std::size_t>(mop.first_child) + 1u];
	                                        const auto& ash = term.shapes[a_idx];
	                                        const auto& bsh = term.shapes[b_idx];

	                                        const bool a_scalar = (ash.kind == Shape::Kind::Scalar);
	                                        const bool b_scalar = (bsh.kind == Shape::Kind::Scalar);
	                                        const bool a_mat = (ash.kind == Shape::Kind::Matrix);
	                                        const bool b_mat = (bsh.kind == Shape::Kind::Matrix);

	                                        if (a_scalar && b_mat) {
	                                            if (!isSpatiallyConstantScalar(isSpatiallyConstantScalar, a_idx)) {
	                                                throw std::runtime_error("LLVMGen: div(s*A) requires spatially-constant scalar s in JIT");
	                                            }
	                                            return mul(values[a_idx], self(self, b_idx));
	                                        }
	                                        if (b_scalar && a_mat) {
	                                            if (!isSpatiallyConstantScalar(isSpatiallyConstantScalar, b_idx)) {
	                                                throw std::runtime_error("LLVMGen: div(A*s) requires spatially-constant scalar s in JIT");
	                                            }
	                                            return mul(values[b_idx], self(self, a_idx));
	                                        }
	                                        throw std::runtime_error("LLVMGen: div(Multiply) currently supports scalar-matrix products only");
	                                    }
	                                    case FormExprType::Add: {
	                                        const auto a_idx = term.ir.children[static_cast<std::size_t>(mop.first_child)];
	                                        const auto b_idx = term.ir.children[static_cast<std::size_t>(mop.first_child) + 1u];
	                                        return add(self(self, a_idx), self(self, b_idx));
	                                    }
	                                    case FormExprType::Subtract: {
	                                        const auto a_idx = term.ir.children[static_cast<std::size_t>(mop.first_child)];
	                                        const auto b_idx = term.ir.children[static_cast<std::size_t>(mop.first_child) + 1u];
	                                        return sub(self(self, a_idx), self(self, b_idx));
	                                    }
	                                    default:
	                                        break;
	                                }

	                                throw std::runtime_error("LLVMGen: div(matrix) operand not supported");
	                            };

	                            values[op_idx] = evalDivMatrix(evalDivMatrix, child_idx);
	                            break;
	                        }

	                        auto divScalarBasis = [&](llvm::Value* n_dofs,
	                                                  llvm::Value* dof_index,
	                                                  llvm::Value* grads_xyz) -> llvm::Value* {
	                            auto* dofs_per_comp = builder.CreateUDiv(n_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                            auto* comp = builder.CreateUDiv(dof_index, dofs_per_comp);
	                            const auto g = loadVec3FromTableQMajor(grads_xyz, n_dofs, dof_index, q_index);
	                            llvm::Value* out = f64c(0.0);
	                            if (vd >= 1) {
	                                out = builder.CreateSelect(builder.CreateICmpEQ(comp, builder.getInt32(0)), g[0], out);
	                            }
	                            if (vd >= 2) {
	                                out = builder.CreateSelect(builder.CreateICmpEQ(comp, builder.getInt32(1)), g[1], out);
                            }
                            if (vd >= 3) {
                                out = builder.CreateSelect(builder.CreateICmpEQ(comp, builder.getInt32(2)), g[2], out);
                            }
                            return out;
                        };

                        auto divTest = [&]() -> llvm::Value* {
                            auto* uses_vec_basis = builder.CreateICmpNE(side.test_uses_vector_basis, builder.getInt32(0));
                            auto* vb = llvm::BasicBlock::Create(*ctx, "div.test.vec_basis", fn);
                            auto* sb = llvm::BasicBlock::Create(*ctx, "div.test.scalar_basis", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "div.test.merge", fn);

                            builder.CreateCondBr(uses_vec_basis, vb, sb);

                            llvm::Value* v_vb = f64c(0.0);
                            builder.SetInsertPoint(vb);
                            v_vb = loadBasisScalar(side, side.test_basis_divs, i_index, q_index);
                            builder.CreateBr(merge);
                            auto* vb_block = builder.GetInsertBlock();

                            llvm::Value* v_sb = f64c(0.0);
                            builder.SetInsertPoint(sb);
                            v_sb = divScalarBasis(side.n_test_dofs, i_index, side.test_phys_grads_xyz);
                            builder.CreateBr(merge);
                            auto* sb_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(merge);
                            auto* phi = builder.CreatePHI(f64, 2, "div.test");
                            phi->addIncoming(v_vb, vb_block);
                            phi->addIncoming(v_sb, sb_block);
                            return phi;
                        };

                        auto divTrial = [&]() -> llvm::Value* {
                            auto* uses_vec_basis = builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
                            auto* vb = llvm::BasicBlock::Create(*ctx, "div.trial.vec_basis", fn);
                            auto* sb = llvm::BasicBlock::Create(*ctx, "div.trial.scalar_basis", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "div.trial.merge", fn);

                            builder.CreateCondBr(uses_vec_basis, vb, sb);

                            llvm::Value* v_vb = f64c(0.0);
                            builder.SetInsertPoint(vb);
                            v_vb = loadBasisScalar(side, side.trial_basis_divs, j_index, q_index);
                            builder.CreateBr(merge);
                            auto* vb_block = builder.GetInsertBlock();

                            llvm::Value* v_sb = f64c(0.0);
                            builder.SetInsertPoint(sb);
                            v_sb = divScalarBasis(side.n_trial_dofs, j_index, side.trial_phys_grads_xyz);
                            builder.CreateBr(merge);
                            auto* sb_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(merge);
                            auto* phi = builder.CreatePHI(f64, 2, "div.trial");
                            phi->addIncoming(v_vb, vb_block);
                            phi->addIncoming(v_sb, sb_block);
                            return phi;
                        };

                        auto divCurrentSolution = [&]() -> llvm::Value* {
                            auto* coeffs = side.solution_coefficients;
                            auto* uses_vec_basis = builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
                            auto* vb = llvm::BasicBlock::Create(*ctx, "div.u.vec_basis", fn);
                            auto* sb = llvm::BasicBlock::Create(*ctx, "div.u.scalar_basis", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "div.u.merge", fn);

	                            builder.CreateCondBr(uses_vec_basis, vb, sb);

		                            llvm::Value* div_vb = f64c(0.0);
		                            builder.SetInsertPoint(vb);
		                            {
		                                div_vb = emitReduceSumScalar(side.n_trial_dofs, "div_u_vb", [&](llvm::Value* j) -> llvm::Value* {
		                                    auto* j64 = builder.CreateZExt(j, i64);
		                                    auto* cj = loadRealPtrAt(coeffs, j64);
		                                    auto* div_phi = loadBasisScalar(side, side.trial_basis_divs, j, q_index);
		                                    return builder.CreateFMul(cj, div_phi);
		                                });
	                                builder.CreateBr(merge);
		                            }
		                            auto* vb_block = builder.GetInsertBlock();

		                            llvm::Value* div_sb = f64c(0.0);
		                            builder.SetInsertPoint(sb);
		                            {
		                                div_sb = emitReduceSumScalar(
		                                    side.n_trial_dofs, "div_u_sb", [&](llvm::Value* j) -> llvm::Value* {
		                                        auto* j64 = builder.CreateZExt(j, i64);
		                                        auto* cj = loadRealPtrAt(coeffs, j64);
		                                        auto* div_phi = divScalarBasis(side.n_trial_dofs, j, side.trial_phys_grads_xyz);
		                                        return builder.CreateFMul(cj, div_phi);
		                                    });
	                                builder.CreateBr(merge);
		                            }
		                            auto* sb_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(merge);
                            auto* phi = builder.CreatePHI(f64, 2, "div.u");
                            phi->addIncoming(div_vb, vb_block);
                            phi->addIncoming(div_sb, sb_block);
                            return phi;
                        };

                        if (kid.type == FormExprType::TestFunction) {
                            values[op_idx] = makeScalar(divTest());
                            break;
                        }
                        if (kid.type == FormExprType::TrialFunction) {
                            values[op_idx] = makeScalar(is_residual ? divCurrentSolution() : divTrial());
                            break;
                        }
	                        if (kid.type == FormExprType::PreviousSolutionRef) {
	                            const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
	                            auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
	                            auto* dofs_per_comp = builder.CreateUDiv(
	                                side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
		                            llvm::Value* div = f64c(0.0);
		                            for (std::size_t comp = 0; comp < vd; ++comp) {
		                                auto* acc = emitReduceSumScalar(
		                                    dofs_per_comp,
		                                    "prev_div" + std::to_string(k) + "_c" + std::to_string(comp),
		                                    [&](llvm::Value* jj) -> llvm::Value* {
		                                        auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
		                                        auto* j = builder.CreateAdd(base, jj);
		                                        auto* j64 = builder.CreateZExt(j, i64);
		                                        auto* cj = loadRealPtrAt(coeffs, j64);
		                                        const auto g =
		                                            loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
		                                        return builder.CreateFMul(cj, g[comp]);
		                                    });
		                                div = builder.CreateFAdd(div, acc);
		                            }
                            values[op_idx] = makeScalar(div);
                            break;
                        }
                        if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                            const int fid = unpackFieldIdImm1(kid.imm1);
                            if (kid.type == FormExprType::StateField && fid == 0xffff) {
                                values[op_idx] = makeScalar(divCurrentSolution());
                                break;
                            }
                            auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
                            auto* entry_is_null =
                                builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok = llvm::BasicBlock::Create(*ctx, "field_div.ok", fn);
                            auto* zero = llvm::BasicBlock::Create(*ctx, "field_div.zero", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "field_div.merge", fn);

                            builder.CreateCondBr(entry_is_null, zero, ok);

                            llvm::Value* div = f64c(0.0);
                            builder.SetInsertPoint(ok);
                            auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                            auto* base_is_null =
                                builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_div.mat.ok", fn);
                            builder.CreateCondBr(base_is_null, zero, ok2);

                            builder.SetInsertPoint(ok2);
                            const auto J = loadMat3FromQ(base, q_index);
                            div = J.elems[0];
                            if (vd >= 2) div = builder.CreateFAdd(div, J.elems[4]);
                            if (vd >= 3) div = builder.CreateFAdd(div, J.elems[8]);
                            builder.CreateBr(merge);
                            auto* ok2_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(zero);
                            builder.CreateBr(merge);
                            auto* zero_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(merge);
                            auto* phi = builder.CreatePHI(f64, 2, "field_div");
                            phi->addIncoming(f64c(0.0), zero_block);
                            phi->addIncoming(div, ok2_block);
                            values[op_idx] = makeScalar(phi);
                            break;
                        }
                        if (kid.type == FormExprType::TimeDerivative) {
                            const int order = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
                            auto* coeff0 = loadDtCoeff0(side, order);
                            const auto dt_child_idx =
                                term.ir.children[static_cast<std::size_t>(kid.first_child)];
                            const auto& dt_child = term.ir.ops[dt_child_idx];

                            if (dt_child.type == FormExprType::TrialFunction) {
                                values[op_idx] = makeScalar(builder.CreateFMul(coeff0, is_residual ? divCurrentSolution() : divTrial()));
                                break;
                            }

                            if (dt_child.type == FormExprType::DiscreteField || dt_child.type == FormExprType::StateField) {
                                const int fid = unpackFieldIdImm1(dt_child.imm1);
                                if (dt_child.type == FormExprType::StateField && fid == 0xffff) {
                                    values[op_idx] = makeScalar(builder.CreateFMul(coeff0, divCurrentSolution()));
                                    break;
                                }

                                auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
                                auto* entry_is_null =
                                    builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok = llvm::BasicBlock::Create(*ctx, "field_dt_div.ok", fn);
                                auto* zero = llvm::BasicBlock::Create(*ctx, "field_dt_div.zero", fn);
                                auto* merge = llvm::BasicBlock::Create(*ctx, "field_dt_div.merge", fn);

                                builder.CreateCondBr(entry_is_null, zero, ok);

                                llvm::Value* div = f64c(0.0);
                                builder.SetInsertPoint(ok);
                                auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                                auto* base_is_null =
                                    builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_dt_div.mat.ok", fn);
                                builder.CreateCondBr(base_is_null, zero, ok2);

                                builder.SetInsertPoint(ok2);
                                const auto J = loadMat3FromQ(base, q_index);
                                div = J.elems[0];
                                if (vd >= 2) div = builder.CreateFAdd(div, J.elems[4]);
                                if (vd >= 3) div = builder.CreateFAdd(div, J.elems[8]);
                                div = builder.CreateFMul(coeff0, div);
                                builder.CreateBr(merge);
                                auto* ok2_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(zero);
                                builder.CreateBr(merge);
                                auto* zero_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(merge);
                                auto* phi = builder.CreatePHI(f64, 2, "field_dt_div");
                                phi->addIncoming(f64c(0.0), zero_block);
                                phi->addIncoming(div, ok2_block);
                                values[op_idx] = makeScalar(phi);
                                break;
                            }

                            if (dt_child.type == FormExprType::Constant) {
                                values[op_idx] = makeScalar(f64c(0.0));
                                break;
                            }

                            throw std::runtime_error("LLVMGen: div(dt(...)) operand not supported");
                            break;
                        }
	                        if (kid.type == FormExprType::Constant) {
	                            values[op_idx] = makeScalar(f64c(0.0));
	                            break;
	                        }

                        // Divergence is linear; handle the common simplified pattern div(0 * X) -> 0.
                        if (kid.type == FormExprType::Multiply && kid.child_count == 2u) {
                            const auto a_idx = term.ir.children[static_cast<std::size_t>(kid.first_child)];
                            const auto b_idx = term.ir.children[static_cast<std::size_t>(kid.first_child) + 1u];
                            const auto& aop = term.ir.ops[a_idx];
                            const auto& bop = term.ir.ops[b_idx];

                            auto isZeroConstant = [](const KernelIROp& op2) -> bool {
                                if (op2.type != FormExprType::Constant) {
                                    return false;
                                }
                                const double v = std::bit_cast<double>(op2.imm0);
                                return v == 0.0;
                            };

                            if (isZeroConstant(aop) || isZeroConstant(bop)) {
                                values[op_idx] = makeScalar(f64c(0.0));
                                break;
                            }
                        }
	                        throw std::runtime_error("LLVMGen: Divergence operand not supported");
	                    }

                    case FormExprType::Curl: {
                        const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                        const auto& kid = term.ir.ops[child_idx];
                        const auto vd = static_cast<std::size_t>(term.shapes[child_idx].dims[0]);

                        auto curlScalarBasis = [&](llvm::Value* n_dofs,
                                                   llvm::Value* dof_index,
                                                   llvm::Value* grads_xyz) -> std::array<llvm::Value*, 3> {
	                        auto* dofs_per_comp = builder.CreateUDiv(n_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                        auto* comp = builder.CreateUDiv(dof_index, dofs_per_comp);
	                        const auto g = loadVec3FromTableQMajor(grads_xyz, n_dofs, dof_index, q_index);
	                        llvm::Value* x = f64c(0.0);
	                        llvm::Value* y = f64c(0.0);
	                        llvm::Value* z = f64c(0.0);
	                        if (vd >= 1) {
	                            auto* is0 = builder.CreateICmpEQ(comp, builder.getInt32(0));
	                            y = builder.CreateSelect(is0, g[2], y);
	                            z = builder.CreateSelect(is0, builder.CreateFNeg(g[1]), z);
	                        }
                            if (vd >= 2) {
                                auto* is1 = builder.CreateICmpEQ(comp, builder.getInt32(1));
                                x = builder.CreateSelect(is1, builder.CreateFNeg(g[2]), x);
                                z = builder.CreateSelect(is1, g[0], z);
                            }
                            if (vd >= 3) {
                                auto* is2 = builder.CreateICmpEQ(comp, builder.getInt32(2));
                                x = builder.CreateSelect(is2, g[1], x);
                                y = builder.CreateSelect(is2, builder.CreateFNeg(g[0]), y);
                            }
                            return {x, y, z};
                        };

                        auto curlTest = [&]() -> CodeValue {
                            auto* uses_vec_basis = builder.CreateICmpNE(side.test_uses_vector_basis, builder.getInt32(0));
                            auto* vb = llvm::BasicBlock::Create(*ctx, "curl.test.vec_basis", fn);
                            auto* sb = llvm::BasicBlock::Create(*ctx, "curl.test.scalar_basis", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "curl.test.merge", fn);

                            builder.CreateCondBr(uses_vec_basis, vb, sb);

                            std::array<llvm::Value*, 3> vb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
                            builder.SetInsertPoint(vb);
                            vb_vals = loadVec3FromTable(side.test_basis_curls_xyz, side.n_qpts, i_index, q_index);
                            builder.CreateBr(merge);
                            auto* vb_block = builder.GetInsertBlock();

                            std::array<llvm::Value*, 3> sb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
                            builder.SetInsertPoint(sb);
                            sb_vals = curlScalarBasis(side.n_test_dofs, i_index, side.test_phys_grads_xyz);
                            builder.CreateBr(merge);
                            auto* sb_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(merge);
                            std::array<llvm::Value*, 3> out{f64c(0.0), f64c(0.0), f64c(0.0)};
                            for (std::size_t d = 0; d < 3; ++d) {
                                auto* phi = builder.CreatePHI(f64, 2, "curl.test." + std::to_string(d));
                                phi->addIncoming(vb_vals[d], vb_block);
                                phi->addIncoming(sb_vals[d], sb_block);
                                out[d] = phi;
                            }
                            return makeVector(3u, out[0], out[1], out[2]);
                        };

                        auto curlTrial = [&]() -> CodeValue {
                            auto* uses_vec_basis = builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
                            auto* vb = llvm::BasicBlock::Create(*ctx, "curl.trial.vec_basis", fn);
                            auto* sb = llvm::BasicBlock::Create(*ctx, "curl.trial.scalar_basis", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "curl.trial.merge", fn);

                            builder.CreateCondBr(uses_vec_basis, vb, sb);

                            std::array<llvm::Value*, 3> vb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
                            builder.SetInsertPoint(vb);
                            vb_vals = loadVec3FromTable(side.trial_basis_curls_xyz, side.n_qpts, j_index, q_index);
                            builder.CreateBr(merge);
                            auto* vb_block = builder.GetInsertBlock();

                            std::array<llvm::Value*, 3> sb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
                            builder.SetInsertPoint(sb);
                            sb_vals = curlScalarBasis(side.n_trial_dofs, j_index, side.trial_phys_grads_xyz);
                            builder.CreateBr(merge);
                            auto* sb_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(merge);
                            std::array<llvm::Value*, 3> out{f64c(0.0), f64c(0.0), f64c(0.0)};
                            for (std::size_t d = 0; d < 3; ++d) {
                                auto* phi = builder.CreatePHI(f64, 2, "curl.trial." + std::to_string(d));
                                phi->addIncoming(vb_vals[d], vb_block);
                                phi->addIncoming(sb_vals[d], sb_block);
                                out[d] = phi;
                            }
                            return makeVector(3u, out[0], out[1], out[2]);
                        };

	                        auto curlCurrentSolution = [&]() -> CodeValue {
	                            auto* coeffs = side.solution_coefficients;
	                            auto* uses_vec_basis = builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
	                            auto* vb = llvm::BasicBlock::Create(*ctx, "curl.u.vec_basis", fn);
	                            auto* sb = llvm::BasicBlock::Create(*ctx, "curl.u.scalar_basis", fn);
	                            auto* merge = llvm::BasicBlock::Create(*ctx, "curl.u.merge", fn);

	                            builder.CreateCondBr(uses_vec_basis, vb, sb);

	                            std::array<llvm::Value*, 3> vb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
	                            builder.SetInsertPoint(vb);
	                            {
	                                const auto sums = emitReduceSum(side.n_trial_dofs, "curl_u_vb", 3u, [&](llvm::Value* j) {
	                                    auto* j64 = builder.CreateZExt(j, i64);
	                                    auto* cj = loadRealPtrAt(coeffs, j64);
	                                    const auto phi =
	                                        loadVec3FromTable(side.trial_basis_curls_xyz, side.n_qpts, j, q_index);
	                                    std::vector<llvm::Value*> terms;
	                                    terms.reserve(3u);
	                                    for (std::size_t d = 0; d < 3u; ++d) {
	                                        terms.push_back(builder.CreateFMul(cj, phi[d]));
	                                    }
	                                    return terms;
	                                });
	                                vb_vals = {sums[0], sums[1], sums[2]};
	                                builder.CreateBr(merge);
	                            }
	                            auto* vb_block = builder.GetInsertBlock();

	                            std::array<llvm::Value*, 3> sb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
	                            builder.SetInsertPoint(sb);
	                            {
	                                const auto sums = emitReduceSum(side.n_trial_dofs, "curl_u_sb", 3u, [&](llvm::Value* j) {
	                                    auto* j64 = builder.CreateZExt(j, i64);
	                                    auto* cj = loadRealPtrAt(coeffs, j64);
	                                    const auto phi = curlScalarBasis(side.n_trial_dofs, j, side.trial_phys_grads_xyz);
	                                    std::vector<llvm::Value*> terms;
	                                    terms.reserve(3u);
	                                    for (std::size_t d = 0; d < 3u; ++d) {
	                                        terms.push_back(builder.CreateFMul(cj, phi[d]));
	                                    }
	                                    return terms;
	                                });
	                                sb_vals = {sums[0], sums[1], sums[2]};
	                                builder.CreateBr(merge);
	                            }
	                            auto* sb_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(merge);
                            std::array<llvm::Value*, 3> out{f64c(0.0), f64c(0.0), f64c(0.0)};
                            for (std::size_t d = 0; d < 3; ++d) {
                                auto* phi = builder.CreatePHI(f64, 2, "curl.u." + std::to_string(d));
                                phi->addIncoming(vb_vals[d], vb_block);
                                phi->addIncoming(sb_vals[d], sb_block);
                                out[d] = phi;
                            }
                            return makeVector(3u, out[0], out[1], out[2]);
                        };

                        if (kid.type == FormExprType::TestFunction) {
                            values[op_idx] = curlTest();
                            break;
                        }
                        if (kid.type == FormExprType::TrialFunction) {
                            values[op_idx] = is_residual ? curlCurrentSolution() : curlTrial();
                            break;
                        }
                        if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                            const int fid = unpackFieldIdImm1(kid.imm1);
                            if (kid.type == FormExprType::StateField && fid == 0xffff) {
                                values[op_idx] = curlCurrentSolution();
                                break;
                            }
                            auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
                            auto* entry_is_null =
                                builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok = llvm::BasicBlock::Create(*ctx, "field_curl.ok", fn);
                            auto* zero = llvm::BasicBlock::Create(*ctx, "field_curl.zero", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "field_curl.merge", fn);

                            builder.CreateCondBr(entry_is_null, zero, ok);

                            std::array<llvm::Value*, 3> c{f64c(0.0), f64c(0.0), f64c(0.0)};
                            builder.SetInsertPoint(ok);
                            auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                            auto* base_is_null =
                                builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_curl.mat.ok", fn);
                            builder.CreateCondBr(base_is_null, zero, ok2);

                            builder.SetInsertPoint(ok2);
                            const auto J = loadMat3FromQ(base, q_index);
                            auto d = [&](std::size_t comp, std::size_t wrt) -> llvm::Value* {
                                if (comp >= vd || comp >= 3u || wrt >= 3u) return f64c(0.0);
                                return J.elems[comp * 3 + wrt];
                            };
                            c[0] = builder.CreateFSub(d(2, 1), d(1, 2));
                            c[1] = builder.CreateFSub(d(0, 2), d(2, 0));
                            c[2] = builder.CreateFSub(d(1, 0), d(0, 1));
                            builder.CreateBr(merge);
                            auto* ok2_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(zero);
                            builder.CreateBr(merge);
                            auto* zero_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(merge);
                            std::array<llvm::Value*, 3> out{f64c(0.0), f64c(0.0), f64c(0.0)};
                            for (std::size_t d0 = 0; d0 < 3; ++d0) {
                                auto* phi = builder.CreatePHI(f64, 2, "field_curl." + std::to_string(d0));
                                phi->addIncoming(f64c(0.0), zero_block);
                                phi->addIncoming(c[d0], ok2_block);
                                out[d0] = phi;
                            }
                            values[op_idx] = makeVector(3u, out[0], out[1], out[2]);
                            break;
                        }
	                        if (kid.type == FormExprType::PreviousSolutionRef) {
	                            const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
	                            auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
	                            auto* dofs_per_comp = builder.CreateUDiv(
	                                side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));

                            std::array<std::array<llvm::Value*, 3>, 3> J{};
                            for (auto& row : J) {
                                row = {f64c(0.0), f64c(0.0), f64c(0.0)};
                            }

	                            for (std::size_t comp = 0; comp < std::min<std::size_t>(vd, 3u); ++comp) {
	                                const auto acc = emitReduceSum(
	                                    dofs_per_comp,
	                                    "prev_curl" + std::to_string(k) + "_c" + std::to_string(comp),
	                                    3u,
	                                    [&](llvm::Value* jj) {
	                                        auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                        auto* j = builder.CreateAdd(base, jj);
	                                        auto* j64 = builder.CreateZExt(j, i64);
	                                        auto* cj = loadRealPtrAt(coeffs, j64);
	                                        const auto g =
	                                            loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                        std::vector<llvm::Value*> terms;
	                                        terms.reserve(3u);
	                                        for (std::size_t d = 0; d < 3u; ++d) {
	                                            terms.push_back(builder.CreateFMul(cj, g[d]));
	                                        }
	                                        return terms;
	                                    });
	                                J[comp] = {acc[0], acc[1], acc[2]};
	                            }

                            auto* cx = builder.CreateFSub(J[2][1], J[1][2]);
                            auto* cy = builder.CreateFSub(J[0][2], J[2][0]);
                            auto* cz = builder.CreateFSub(J[1][0], J[0][1]);
                            values[op_idx] = makeVector(3u, cx, cy, cz);
                            break;
                        }
                        if (kid.type == FormExprType::TimeDerivative) {
                            const int order = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
                            auto* coeff0 = loadDtCoeff0(side, order);
                            const auto dt_child_idx =
                                term.ir.children[static_cast<std::size_t>(kid.first_child)];
                            const auto& dt_child = term.ir.ops[dt_child_idx];

                            if (dt_child.type == FormExprType::TrialFunction) {
                                const auto v = is_residual ? curlCurrentSolution() : curlTrial();
                                values[op_idx] = mul(makeScalar(coeff0), v);
                                break;
                            }

                            if (dt_child.type == FormExprType::DiscreteField || dt_child.type == FormExprType::StateField) {
                                const int fid = unpackFieldIdImm1(dt_child.imm1);
                                const auto v =
                                    (dt_child.type == FormExprType::StateField && fid == 0xffff)
                                        ? curlCurrentSolution()
                                        : [&]() -> CodeValue {
                                              auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
                                              auto* entry_is_null =
                                                  builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                              auto* ok = llvm::BasicBlock::Create(*ctx, "field_dt_curl.ok", fn);
                                              auto* zero = llvm::BasicBlock::Create(*ctx, "field_dt_curl.zero", fn);
                                              auto* merge = llvm::BasicBlock::Create(*ctx, "field_dt_curl.merge", fn);

                                              builder.CreateCondBr(entry_is_null, zero, ok);

                                              std::array<llvm::Value*, 3> c{f64c(0.0), f64c(0.0), f64c(0.0)};
                                              builder.SetInsertPoint(ok);
                                              auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                                              auto* base_is_null =
                                                  builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                              auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_dt_curl.mat.ok", fn);
                                              builder.CreateCondBr(base_is_null, zero, ok2);

                                              builder.SetInsertPoint(ok2);
                                              const auto J = loadMat3FromQ(base, q_index);
                                              auto d = [&](std::size_t comp, std::size_t wrt) -> llvm::Value* {
                                                  if (comp >= vd || comp >= 3u || wrt >= 3u) return f64c(0.0);
                                                  return J.elems[comp * 3 + wrt];
                                              };
                                              c[0] = builder.CreateFSub(d(2, 1), d(1, 2));
                                              c[1] = builder.CreateFSub(d(0, 2), d(2, 0));
                                              c[2] = builder.CreateFSub(d(1, 0), d(0, 1));
                                              builder.CreateBr(merge);
                                              auto* ok2_block = builder.GetInsertBlock();

                                              builder.SetInsertPoint(zero);
                                              builder.CreateBr(merge);
                                              auto* zero_block = builder.GetInsertBlock();

                                              builder.SetInsertPoint(merge);
                                              std::array<llvm::Value*, 3> out{f64c(0.0), f64c(0.0), f64c(0.0)};
                                              for (std::size_t d0 = 0; d0 < 3; ++d0) {
                                                  auto* phi = builder.CreatePHI(f64, 2, "field_dt_curl." + std::to_string(d0));
                                                  phi->addIncoming(f64c(0.0), zero_block);
                                                  phi->addIncoming(c[d0], ok2_block);
                                                  out[d0] = phi;
                                              }
                                              return makeVector(3u, out[0], out[1], out[2]);
                                          }();
                                values[op_idx] = mul(makeScalar(coeff0), v);
                                break;
                            }

                            if (dt_child.type == FormExprType::Constant) {
                                values[op_idx] = makeZero(vectorShape(3u));
                                break;
                            }

                            throw std::runtime_error("LLVMGen: curl(dt(...)) operand not supported");
                            break;
                        }
                        if (kid.type == FormExprType::Constant) {
                            values[op_idx] = makeZero(vectorShape(3u));
                            break;
                        }
                        throw std::runtime_error("LLVMGen: Curl operand not supported");
                    }

                    case FormExprType::Hessian: {
                        const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                        const auto& kid = term.ir.ops[child_idx];

                        auto* zero = f64c(0.0);
                        const auto dim_u32 = shape.dims[0];
                        const auto mat_len = elemCount(shape);
	                        auto matZero = [&]() -> CodeValue { return makeZero(shape); };

	                        auto hessFromCoeffs = [&](llvm::Value* coeffs_ptr, const std::string& loop_name) -> CodeValue {
	                            CodeValue out = makeZero(shape);
	                            const auto sums = emitReduceSum(side.n_trial_dofs, loop_name, mat_len, [&](llvm::Value* j) {
	                                auto* j64 = builder.CreateZExt(j, i64);
	                                auto* cj = loadRealPtrAt(coeffs_ptr, j64);
	                                const auto H =
	                                    loadMatDimFromTable(side.trial_phys_hessians, side.n_qpts, j, q_index, dim_u32);
	                                std::vector<llvm::Value*> terms;
	                                terms.reserve(mat_len);
	                                for (std::size_t i = 0; i < mat_len; ++i) {
	                                    terms.push_back(builder.CreateFMul(cj, H.elems[i]));
	                                }
	                                return terms;
	                            });
	                            for (std::size_t i = 0; i < mat_len; ++i) {
	                                out.elems[i] = sums[i];
	                            }
	                            return out;
	                        };

	                        auto hessCurrentSolution = [&]() -> CodeValue {
	                            auto* coeffs = side.solution_coefficients;
	                            return hessFromCoeffs(coeffs, "hess_u");
	                        };

                        if (kid.type == FormExprType::TestFunction) {
                            values[op_idx] = loadMatDimFromTable(side.test_phys_hessians, side.n_qpts, i_index, q_index, dim_u32);
                            break;
                        }
                        if (kid.type == FormExprType::TrialFunction) {
                            values[op_idx] =
                                is_residual ? hessCurrentSolution()
                                            : loadMatDimFromTable(side.trial_phys_hessians, side.n_qpts, j_index, q_index, dim_u32);
                            break;
                        }
	                        if (kid.type == FormExprType::PreviousSolutionRef) {
	                            const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
	                            auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
	                            values[op_idx] = hessFromCoeffs(coeffs, "prev_hess" + std::to_string(k));
	                            break;
	                        }
                        if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                            const int fid = unpackFieldIdImm1(kid.imm1);
                            if (kid.type == FormExprType::StateField && fid == 0xffff) {
                                values[op_idx] = hessCurrentSolution();
                                break;
                            }
                            auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
                            auto* entry_is_null =
                                builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok = llvm::BasicBlock::Create(*ctx, "field_hess.ok", fn);
                            auto* z = llvm::BasicBlock::Create(*ctx, "field_hess.zero", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "field_hess.merge", fn);

                            builder.CreateCondBr(entry_is_null, z, ok);

                            CodeValue loaded = matZero();
                            builder.SetInsertPoint(ok);
                            auto* base = loadPtr(entry, ABIV3::field_entry_hessians_off);
                            auto* base_is_null =
                                builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_hess.mat.ok", fn);
                            builder.CreateCondBr(base_is_null, z, ok2);

                            builder.SetInsertPoint(ok2);
                            loaded = loadMatDimFromQ(base, q_index, dim_u32);
                            builder.CreateBr(merge);
                            auto* ok2_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(z);
                            builder.CreateBr(merge);
                            auto* zero_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(merge);
                            CodeValue out = makeZero(shape);
                            for (std::size_t i = 0; i < mat_len; ++i) {
                                auto* phi = builder.CreatePHI(f64, 2, "field_hess." + std::to_string(i));
                                phi->addIncoming(zero, zero_block);
                                phi->addIncoming(loaded.elems[i], ok2_block);
                                out.elems[i] = phi;
                            }
                            values[op_idx] = out;
                            break;
                        }
                        if (kid.type == FormExprType::TimeDerivative) {
                            const int order = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
                            auto* coeff0 = loadDtCoeff0(side, order);
                            const auto dt_child_idx =
                                term.ir.children[static_cast<std::size_t>(kid.first_child)];
                            const auto& dt_child = term.ir.ops[dt_child_idx];

                            if (dt_child.type == FormExprType::TrialFunction) {
                                const auto H =
                                    is_residual ? hessCurrentSolution()
                                                : loadMatDimFromTable(side.trial_phys_hessians, side.n_qpts, j_index, q_index, dim_u32);
                                values[op_idx] = mul(makeScalar(coeff0), H);
                                break;
                            }

                            if (dt_child.type == FormExprType::DiscreteField || dt_child.type == FormExprType::StateField) {
                                const int fid = unpackFieldIdImm1(dt_child.imm1);
                                if (dt_child.type == FormExprType::StateField && fid == 0xffff) {
                                    values[op_idx] = mul(makeScalar(coeff0), hessCurrentSolution());
                                    break;
                                }

                                auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
                                auto* entry_is_null =
                                    builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok = llvm::BasicBlock::Create(*ctx, "field_dt_hess.ok", fn);
                                auto* z = llvm::BasicBlock::Create(*ctx, "field_dt_hess.zero", fn);
                                auto* merge = llvm::BasicBlock::Create(*ctx, "field_dt_hess.merge", fn);

                                builder.CreateCondBr(entry_is_null, z, ok);

                                CodeValue loaded = matZero();
                                builder.SetInsertPoint(ok);
                                auto* base = loadPtr(entry, ABIV3::field_entry_hessians_off);
                                auto* base_is_null =
                                    builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_dt_hess.mat.ok", fn);
                                builder.CreateCondBr(base_is_null, z, ok2);

                                builder.SetInsertPoint(ok2);
                                loaded = loadMatDimFromQ(base, q_index, dim_u32);
                                builder.CreateBr(merge);
                                auto* ok2_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(z);
                                builder.CreateBr(merge);
                                auto* zero_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(merge);
                                CodeValue out = matZero();
                                for (std::size_t i = 0; i < mat_len; ++i) {
                                    auto* phi = builder.CreatePHI(f64, 2, "field_dt_hess." + std::to_string(i));
                                    phi->addIncoming(zero, zero_block);
                                    phi->addIncoming(loaded.elems[i], ok2_block);
                                    out.elems[i] = phi;
                                }
                                values[op_idx] = mul(makeScalar(coeff0), out);
                                break;
                            }

                            if (dt_child.type == FormExprType::Constant) {
                                values[op_idx] = matZero();
                                break;
                            }

                            throw std::runtime_error("LLVMGen: H(dt(...)) operand not supported");
                            break;
                        }
                        if (kid.type == FormExprType::Constant) {
                            values[op_idx] = matZero();
                            break;
                        }
                        throw std::runtime_error("LLVMGen: Hessian operand not supported");
                    }

                    case FormExprType::TimeDerivative: {
                        const int order = static_cast<int>(static_cast<std::int64_t>(op.imm0));
                        CodeValue acc = mul(makeScalar(loadDtCoeff(side, order, 0)), getChild(op, 0));

                        const auto child_op_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                        const auto& kid = term.ir.ops[child_op_idx];
                        if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                            const int fid = unpackFieldIdImm1(kid.imm1);
                            if (kid.type == FormExprType::StateField && fid == 0xffff) {
                                for (int k = 1; k <= static_cast<int>(assembly::jit::kMaxPreviousSolutionsV6); ++k) {
                                    acc = add(acc,
                                              mul(makeScalar(loadDtCoeff(side, order, k)),
                                                  evalPreviousSolution(side, shape, k, q_index)));
                                }
                            } else {
                                for (int k = 1; k <= static_cast<int>(assembly::jit::kMaxPreviousSolutionsV6); ++k) {
                                    acc = add(acc,
                                              mul(makeScalar(loadDtCoeff(side, order, k)),
                                                  evalDiscreteOrStateFieldHistoryK(side, /*plus_side=*/false, shape, fid, k, q_index)));
                                }
                            }
                        }

                        values[op_idx] = acc;
                        break;
                    }

                    case FormExprType::MaterialStateOldRef:
                        values[op_idx] = makeScalar(loadMaterialStateReal(side.material_state_old_base,
                                                                         side.material_state_stride_bytes,
                                                                         q_index,
                                                                         static_cast<std::uint64_t>(op.imm0)));
                        break;

                    case FormExprType::MaterialStateWorkRef:
                        values[op_idx] = makeScalar(loadMaterialStateReal(side.material_state_work_base,
                                                                         side.material_state_stride_bytes,
                                                                         q_index,
                                                                         static_cast<std::uint64_t>(op.imm0)));
                        break;

                    case FormExprType::Negate:
                        values[op_idx] = neg(getChild(op, 0));
                        break;

	                    case FormExprType::Add: {
	                        if (shape.kind == Shape::Kind::Scalar) {
	                            if (auto* fused = tryFuseMulAddScalar(op.type, term, use_counts, op, values)) {
	                                values[op_idx] = makeScalar(fused);
	                                break;
	                            }
	                        }
	                        values[op_idx] = add(getChild(op, 0), getChild(op, 1));
	                        break;
	                    }

	                    case FormExprType::Subtract: {
	                        if (shape.kind == Shape::Kind::Scalar) {
	                            if (auto* fused = tryFuseMulAddScalar(op.type, term, use_counts, op, values)) {
	                                values[op_idx] = makeScalar(fused);
	                                break;
	                            }
	                        }
	                        values[op_idx] = sub(getChild(op, 0), getChild(op, 1));
	                        break;
	                    }

                    case FormExprType::Multiply:
                        values[op_idx] = mul(getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::Divide:
                        values[op_idx] = div(getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::InnerProduct:
                        values[op_idx] = inner(getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::DoubleContraction:
                        values[op_idx] = doubleContraction(getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::OuterProduct:
                        values[op_idx] = outer(getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::CrossProduct:
                        values[op_idx] = cross(getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::Power:
                        values[op_idx] = powv(getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::Minimum:
                        values[op_idx] = makeScalar(f_min(getChild(op, 0).elems[0], getChild(op, 1).elems[0]));
                        break;

                    case FormExprType::Maximum:
                        values[op_idx] = makeScalar(f_max(getChild(op, 0).elems[0], getChild(op, 1).elems[0]));
                        break;

                    case FormExprType::Less:
                    case FormExprType::LessEqual:
                    case FormExprType::Greater:
                    case FormExprType::GreaterEqual:
                    case FormExprType::Equal:
                    case FormExprType::NotEqual:
                        values[op_idx] = cmp(op.type, getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::Conditional:
                        values[op_idx] = conditional(getChild(op, 0), getChild(op, 1), getChild(op, 2));
                        break;

                    case FormExprType::AsVector: {
                        CodeValue out = makeZero(shape);
                        for (std::size_t k = 0; k < op.child_count; ++k) {
                            out.elems[k] = getChild(op, k).elems[0];
                        }
                        values[op_idx] = out;
                        break;
                    }

                    case FormExprType::AsTensor: {
                        const auto rows = unpackU32Lo(op.imm0);
                        const auto cols = unpackU32Hi(op.imm0);
                        CodeValue out = makeMatrix(rows, cols);
                        for (std::size_t r = 0; r < rows; ++r) {
                            for (std::size_t c = 0; c < cols; ++c) {
                                const auto k = r * cols + c;
                                out.elems[k] = getChild(op, k).elems[0];
                            }
                        }
                        values[op_idx] = out;
                        break;
                    }

                    case FormExprType::Component: {
                        const auto i = static_cast<std::int32_t>(unpackU32Lo(op.imm0));
                        const auto j = static_cast<std::int32_t>(unpackU32Hi(op.imm0));
                        const auto& a = getChild(op, 0);
                        if (a.shape.kind == Shape::Kind::Scalar) {
                            values[op_idx] = a;
                            break;
                        }
                        if (a.shape.kind == Shape::Kind::Vector) {
                            if (j >= 0) throw std::runtime_error("LLVMGen: component(v,i,j) invalid for vector");
                            const auto idxv = static_cast<std::size_t>(i);
                            values[op_idx] = makeScalar(getVecComp(a, idxv));
                            break;
                        }
                        if (a.shape.kind == Shape::Kind::Matrix) {
                            if (j < 0) throw std::runtime_error("LLVMGen: component(A,i) missing column index");
                            const auto rows = static_cast<std::size_t>(a.shape.dims[0]);
                            const auto cols = static_cast<std::size_t>(a.shape.dims[1]);
                            const auto rr = static_cast<std::size_t>(i);
                            const auto cc = static_cast<std::size_t>(j);
                            if (rr >= rows || cc >= cols) throw std::runtime_error("LLVMGen: component(A,i,j) index out of range");
                            values[op_idx] = makeScalar(a.elems[rr * cols + cc]);
                            break;
                        }
                        throw std::runtime_error("LLVMGen: component() unsupported operand kind");
                    }

                    case FormExprType::IndexedAccess: {
                        if (op.child_count != 1u) {
                            throw std::runtime_error("LLVMGen: IndexedAccess expects exactly 1 child");
                        }
                        if (indexed_env == nullptr) {
                            throw std::runtime_error("LLVMGen: IndexedAccess requires an index environment (Einstein-sum lowering)");
                        }

                        const auto& a = getChild(op, 0);
                        if (a.shape.kind == Shape::Kind::Scalar) {
                            values[op_idx] = a;
                            break;
                        }

                        const int rank = static_cast<int>(unpackIndexedRank(op.imm1));
                        if (rank <= 0 || rank > 4) {
                            throw std::runtime_error("LLVMGen: IndexedAccess has invalid rank");
                        }

                        auto loadIndex = [&](int k) -> llvm::Value* {
                            const std::size_t id = static_cast<std::size_t>(unpackIndexedId(op.imm0, k));
                            if (id >= indexed_env->size() || (*indexed_env)[id] == nullptr) {
                                throw std::runtime_error("LLVMGen: IndexedAccess missing index assignment");
                            }
                            return (*indexed_env)[id];
                        };

                        auto selectLinear = [&](llvm::Value* idx) -> llvm::Value* {
                            const auto n = elemCount(a.shape);
                            if (n == 0u) {
                                throw std::runtime_error("LLVMGen: IndexedAccess operand has no elements");
                            }
                            llvm::Value* out = a.elems[0];
                            for (std::size_t t = 1; t < n; ++t) {
                                auto* is_t = builder.CreateICmpEQ(idx, builder.getInt32(static_cast<std::uint32_t>(t)));
                                out = builder.CreateSelect(is_t, a.elems[t], out);
                            }
                            return out;
                        };

                        if (rank == 1 && a.shape.kind == Shape::Kind::Vector) {
                            values[op_idx] = makeScalar(selectLinear(loadIndex(0)));
                            break;
                        }
                        if (rank == 2 && a.shape.kind == Shape::Kind::Matrix) {
                            auto* ii = loadIndex(0);
                            auto* jj = loadIndex(1);
                            const auto cols = static_cast<std::uint32_t>(a.shape.dims[1]);
                            auto* lin = builder.CreateAdd(builder.CreateMul(ii, builder.getInt32(cols)), jj);
                            values[op_idx] = makeScalar(selectLinear(lin));
                            break;
                        }
                        if (rank == 3 && a.shape.kind == Shape::Kind::Tensor3) {
                            auto* ii = loadIndex(0);
                            auto* jj = loadIndex(1);
                            auto* kk = loadIndex(2);
                            const auto d1 = static_cast<std::uint32_t>(a.shape.dims[1]);
                            const auto d2 = static_cast<std::uint32_t>(a.shape.dims[2]);
                            auto* lin = builder.CreateAdd(builder.CreateMul(ii, builder.getInt32(d1 * d2)),
                                                          builder.CreateAdd(builder.CreateMul(jj, builder.getInt32(d2)), kk));
                            values[op_idx] = makeScalar(selectLinear(lin));
                            break;
                        }
                        if (rank == 4 && a.shape.kind == Shape::Kind::Tensor4) {
                            auto* ii = loadIndex(0);
                            auto* jj = loadIndex(1);
                            auto* kk = loadIndex(2);
                            auto* ll = loadIndex(3);
                            const auto d1 = static_cast<std::uint32_t>(a.shape.dims[1]);
                            const auto d2 = static_cast<std::uint32_t>(a.shape.dims[2]);
                            const auto d3 = static_cast<std::uint32_t>(a.shape.dims[3]);
                            const auto stride0 = d1 * d2 * d3;
                            const auto stride1 = d2 * d3;
                            const auto stride2 = d3;
                            auto* lin = builder.CreateAdd(builder.CreateMul(ii, builder.getInt32(stride0)),
                                                          builder.CreateAdd(builder.CreateMul(jj, builder.getInt32(stride1)),
                                                                            builder.CreateAdd(builder.CreateMul(kk, builder.getInt32(stride2)), ll)));
                            values[op_idx] = makeScalar(selectLinear(lin));
                            break;
                        }

                        throw std::runtime_error("LLVMGen: IndexedAccess rank/operand shape mismatch");
                    }

                    case FormExprType::Transpose:
                        values[op_idx] = transpose(getChild(op, 0));
                        break;

                    case FormExprType::Trace:
                        values[op_idx] = trace(getChild(op, 0));
                        break;

                    case FormExprType::Determinant:
                        values[op_idx] = det(getChild(op, 0));
                        break;

                    case FormExprType::Cofactor:
                        values[op_idx] = cofactor(getChild(op, 0));
                        break;

                    case FormExprType::Inverse:
                        values[op_idx] = inv(getChild(op, 0));
                        break;

                    case FormExprType::SymmetricPart:
                        values[op_idx] = symOrSkew(true, getChild(op, 0));
                        break;

                    case FormExprType::SkewPart:
                        values[op_idx] = symOrSkew(false, getChild(op, 0));
                        break;

                    case FormExprType::Deviator:
                        values[op_idx] = deviator(getChild(op, 0));
                        break;

                    case FormExprType::Norm:
                        values[op_idx] = norm(getChild(op, 0));
                        break;

                    case FormExprType::Normalize:
                        values[op_idx] = normalize(getChild(op, 0));
                        break;

                    case FormExprType::AbsoluteValue:
                        values[op_idx] = absScalar(getChild(op, 0));
                        break;

                    case FormExprType::Sign:
                        values[op_idx] = signScalar(getChild(op, 0));
                        break;

                    case FormExprType::Sqrt:
                        values[op_idx] = evalSqrt(getChild(op, 0));
                        break;

                    case FormExprType::Exp:
                        values[op_idx] = evalExp(getChild(op, 0));
                        break;

                    case FormExprType::Log:
                        values[op_idx] = evalLog(getChild(op, 0));
                        break;

	                    case FormExprType::MatrixExponential:
	                        values[op_idx] = emitMatrixExp(getChild(op, 0));
	                        break;

	                    case FormExprType::MatrixLogarithm:
	                        values[op_idx] = emitMatrixLog(getChild(op, 0));
	                        break;

	                    case FormExprType::MatrixSqrt:
	                        values[op_idx] = emitMatrixSqrt(getChild(op, 0));
	                        break;

	                    case FormExprType::MatrixPower:
	                        values[op_idx] =
	                            emitMatrixPow(getChild(op, 0), getChild(op, 1).elems[0]);
	                        break;

                    case FormExprType::MatrixExponentialDirectionalDerivative:
                        values[op_idx] = callMatrixUnaryDD(getChild(op, 0), getChild(op, 1), mat_exp_dd_2x2_fn, mat_exp_dd_3x3_fn);
                        break;

                    case FormExprType::MatrixLogarithmDirectionalDerivative:
                        values[op_idx] = callMatrixUnaryDD(getChild(op, 0), getChild(op, 1), mat_log_dd_2x2_fn, mat_log_dd_3x3_fn);
                        break;

                    case FormExprType::MatrixSqrtDirectionalDerivative:
                        values[op_idx] = callMatrixUnaryDD(getChild(op, 0), getChild(op, 1), mat_sqrt_dd_2x2_fn, mat_sqrt_dd_3x3_fn);
                        break;

                    case FormExprType::MatrixPowerDirectionalDerivative:
                        values[op_idx] = callMatrixPowDD(getChild(op, 0), getChild(op, 1), getChild(op, 2).elems[0], mat_pow_dd_2x2_fn, mat_pow_dd_3x3_fn);
                        break;

                    case FormExprType::SmoothAbsoluteValue:
                        values[op_idx] = smoothAbs(getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::SmoothSign:
                        values[op_idx] = smoothSign(getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::SmoothHeaviside:
                        values[op_idx] = smoothHeaviside(getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::SmoothMin:
                        values[op_idx] = smoothMin(getChild(op, 0), getChild(op, 1), getChild(op, 2));
                        break;

                    case FormExprType::SmoothMax:
                        values[op_idx] = smoothMax(getChild(op, 0), getChild(op, 1), getChild(op, 2));
                        break;

                    case FormExprType::SymmetricEigenvalue: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values[op_idx] = eigSym(getChild(op, 0), which);
                        break;
                    }

                    case FormExprType::Eigenvalue: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values[op_idx] = eigSym(getChild(op, 0), which);
                        break;
                    }

                    case FormExprType::SymmetricEigenvector: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values[op_idx] = eigSymVec(getChild(op, 0), which);
                        break;
                    }

                    case FormExprType::SpectralDecomposition:
                        values[op_idx] = spectralDecomp(getChild(op, 0));
                        break;

                    case FormExprType::SymmetricEigenvectorDirectionalDerivative: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values[op_idx] = eigSymVecDD(getChild(op, 0), getChild(op, 1), which);
                        break;
                    }

                    case FormExprType::SpectralDecompositionDirectionalDerivative:
                        values[op_idx] = spectralDecompDD(getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::HistoryWeightedSum:
                    case FormExprType::HistoryConvolution: {
                        CodeValue acc = makeZero(shape);
                        if (op.child_count != 0u) {
                            for (std::size_t kk = 0; kk < static_cast<std::size_t>(op.child_count); ++kk) {
                                const int k = static_cast<int>(kk + 1u);
                                acc = add(acc, mul(getChild(op, kk), evalHistorySolution(side, shape, k, q_index)));
                            }
                        } else {
                            for (int k = 1; k <= static_cast<int>(assembly::jit::kMaxPreviousSolutionsV6); ++k) {
                                acc = add(acc,
                                          mul(makeScalar(loadHistoryWeightOrZero(side, k)),
                                              evalHistorySolution(side, shape, k, q_index)));
                            }
                        }
                        values[op_idx] = acc;
                        break;
                    }

                    case FormExprType::SymmetricEigenvalueDirectionalDerivative: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values[op_idx] = eigSymDD(getChild(op, 0), getChild(op, 1), which);
                        break;
                    }

                    case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values[op_idx] = eigSymDDWrtA(getChild(op, 0), getChild(op, 1), getChild(op, 2), which);
                        break;
                    }

                    default:
                        throw std::runtime_error("LLVMGen: unsupported FormExprType in codegen (FormExprType=" +
                                                 std::to_string(static_cast<std::uint16_t>(op.type)) + ")");
                }
            }

            if (term.ir.root >= values.size()) {
                throw std::runtime_error("LLVMGen: invalid KernelIR root index");
            }
            return values[term.ir.root];
        };

        auto evalKernelIRSingleScalar = [&](const LoweredTerm& term,
                                            llvm::Value* q_index,
                                            llvm::Value* i_index,
                                            llvm::Value* j_index,
                                            const SideView& side,
                                            const std::vector<llvm::Value*>* indexed_env,
                                            const std::vector<CodeValue>* cached) -> llvm::Value* {
            const auto root = evalKernelIRSingleValue(term, q_index, i_index, j_index, side, indexed_env, cached);
            if (root.shape.kind != Shape::Kind::Scalar) {
                throw std::runtime_error("LLVMGen: integrand did not lower to scalar");
            }
            return root.elems[0];
        };

        auto evalKernelIRSingle = [&](const LoweredTerm& term,
                                      llvm::Value* q_index,
                                      llvm::Value* i_index,
                                      llvm::Value* j_index,
                                      const SideView& side,
                                      const std::vector<CodeValue>* cached) -> llvm::Value* {
            if (term.tensor_ir.has_value()) {
                LLVMTensorGen tensor_gen(*ctx,
                                         builder,
                                         *fn,
                                         LLVMTensorGenOptions{
                                             .vectorize = options_.vectorize,
                                             .enable_polly = options_.tensor.enable_polly,
                                             .enable_tiling = options_.tensor.enable_loop_tiling,
                                             .tile_size = static_cast<int>(options_.tensor.tile_size),
                                             .min_tiling_extent = static_cast<int>(options_.tensor.min_tiling_extent),
                                         });

                const auto eval_scalar = [&](const FormExpr& scalar_expr) -> llvm::Value* {
                    if (!scalar_expr.isValid() || scalar_expr.node() == nullptr) {
                        throw std::runtime_error("LLVMGen: TensorIR scalar expression is invalid");
                    }

                    auto lowered_scalar = lowerToKernelIR(scalar_expr);
                    auto scalar_shapes = inferShapes(lowered_scalar.ir, ir.testSpace(), ir.trialSpace(), /*require_scalar_root=*/false);
                    if (!scalar_shapes.ok) {
                        throw std::runtime_error(scalar_shapes.message.empty() ? "LLVMGen: failed to infer scalar shapes" : scalar_shapes.message);
                    }

                    LoweredTerm scalar_term;
                    scalar_term.ir = std::move(lowered_scalar.ir);
                    scalar_term.shapes = std::move(scalar_shapes.shapes);

                    const auto v = evalKernelIRSingleValue(scalar_term,
                                                          q_index,
                                                          i_index,
                                                          j_index,
                                                          side,
                                                          /*indexed_env=*/nullptr,
                                                          /*cached=*/nullptr);
                    if (v.shape.kind != Shape::Kind::Scalar) {
                        throw std::runtime_error("LLVMGen: TensorIR scalar expression did not lower to scalar");
                    }
                    return v.elems[0];
                };

                const auto& tir = *term.tensor_ir;
                std::vector<CodeValue> base_values;
                base_values.resize(tir.program.tensors.size());
                std::vector<bool> has_base;
                has_base.resize(tir.program.tensors.size(), false);

                for (std::size_t tid = 0; tid < tir.program.tensors.size(); ++tid) {
                    const auto& spec = tir.program.tensors[tid];
                    if (!spec.base.isValid() || spec.base.node() == nullptr) {
                        continue;
                    }

                    auto lowered_base = lowerToKernelIR(spec.base);
                    auto base_shapes = inferShapes(lowered_base.ir, ir.testSpace(), ir.trialSpace(), /*require_scalar_root=*/false);
                    if (!base_shapes.ok) {
                        throw std::runtime_error(base_shapes.message.empty() ? "LLVMGen: failed to infer base-tensor shapes" : base_shapes.message);
                    }

                    LoweredTerm base_term;
                    base_term.ir = std::move(lowered_base.ir);
                    base_term.shapes = std::move(base_shapes.shapes);

                    base_values[tid] = evalKernelIRSingleValue(base_term,
                                                              q_index,
                                                              i_index,
                                                              j_index,
                                                              side,
                                                              /*indexed_env=*/nullptr,
                                                              /*cached=*/nullptr);
                    has_base[tid] = true;
                }

                const auto load_input = [&](int tensor_id,
                                            const tensor::TensorSpec& spec,
                                            const std::vector<llvm::Value*>& index_env) -> llvm::Value* {
                    if (tensor_id < 0 || static_cast<std::size_t>(tensor_id) >= base_values.size()) {
                        throw std::runtime_error("LLVMGen: TensorIR input tensor id out of range");
                    }
                    if (!has_base[static_cast<std::size_t>(tensor_id)]) {
                        throw std::runtime_error("LLVMGen: TensorIR input tensor missing evaluated base value");
                    }

                    const auto& base = base_values[static_cast<std::size_t>(tensor_id)];
                    if (base.shape.kind == Shape::Kind::Scalar) {
                        return base.elems[0];
                    }

                    const int base_rank = spec.base_rank;
                    if (base_rank < 0 || static_cast<std::size_t>(base_rank) > spec.base_axis_to_tensor_axis.size()) {
                        throw std::runtime_error("LLVMGen: TensorIR input tensor has invalid base-rank metadata");
                    }

                    auto loadIndex = [&](int axis) -> llvm::Value* {
                        if (axis < 0 || axis >= base_rank) {
                            throw std::runtime_error("LLVMGen: TensorIR input tensor base-axis out of range");
                        }
                        const int tpos = spec.base_axis_to_tensor_axis[static_cast<std::size_t>(axis)];
                        if (tpos < 0 || static_cast<std::size_t>(tpos) >= spec.axes.size()) {
                            throw std::runtime_error("LLVMGen: TensorIR input tensor has invalid axis mapping");
                        }
                        const int id = spec.axes[static_cast<std::size_t>(tpos)];
                        if (id < 0 || static_cast<std::size_t>(id) >= index_env.size() || index_env[static_cast<std::size_t>(id)] == nullptr) {
                            throw std::runtime_error("LLVMGen: TensorIR input tensor missing index assignment");
                        }
                        return index_env[static_cast<std::size_t>(id)];
                    };

                    auto selectLinear = [&](llvm::Value* idx) -> llvm::Value* {
                        const auto n = elemCount(base.shape);
                        if (n == 0u) {
                            throw std::runtime_error("LLVMGen: TensorIR input tensor has empty base");
                        }
                        llvm::Value* out = base.elems[0];
                        for (std::size_t t = 1; t < n; ++t) {
                            auto* is_t = builder.CreateICmpEQ(idx, builder.getInt32(static_cast<std::uint32_t>(t)));
                            out = builder.CreateSelect(is_t, base.elems[t], out);
                        }
                        return out;
                    };

                    if (base.shape.kind == Shape::Kind::Vector) {
                        if (base_rank != 1) throw std::runtime_error("LLVMGen: TensorIR input vector rank mismatch");
                        return selectLinear(loadIndex(0));
                    }
                    if (base.shape.kind == Shape::Kind::Matrix) {
                        if (base_rank != 2) throw std::runtime_error("LLVMGen: TensorIR input matrix rank mismatch");
                        auto* ii = loadIndex(0);
                        auto* jj = loadIndex(1);
                        const auto cols = static_cast<std::uint32_t>(base.shape.dims[1]);
                        auto* lin = builder.CreateAdd(builder.CreateMul(ii, builder.getInt32(cols)), jj);
                        return selectLinear(lin);
                    }
                    if (base.shape.kind == Shape::Kind::Tensor3) {
                        if (base_rank != 3) throw std::runtime_error("LLVMGen: TensorIR input tensor3 rank mismatch");
                        auto* ii = loadIndex(0);
                        auto* jj = loadIndex(1);
                        auto* kk = loadIndex(2);
                        const auto d1 = static_cast<std::uint32_t>(base.shape.dims[1]);
                        const auto d2 = static_cast<std::uint32_t>(base.shape.dims[2]);
                        auto* lin = builder.CreateAdd(builder.CreateMul(ii, builder.getInt32(d1 * d2)),
                                                      builder.CreateAdd(builder.CreateMul(jj, builder.getInt32(d2)), kk));
                        return selectLinear(lin);
                    }
                    if (base.shape.kind == Shape::Kind::Tensor4) {
                        if (base_rank != 4) throw std::runtime_error("LLVMGen: TensorIR input tensor4 rank mismatch");
                        auto* ii = loadIndex(0);
                        auto* jj = loadIndex(1);
                        auto* kk = loadIndex(2);
                        auto* ll = loadIndex(3);
                        const auto d1 = static_cast<std::uint32_t>(base.shape.dims[1]);
                        const auto d2 = static_cast<std::uint32_t>(base.shape.dims[2]);
                        const auto d3 = static_cast<std::uint32_t>(base.shape.dims[3]);
                        const auto stride0 = d1 * d2 * d3;
                        const auto stride1 = d2 * d3;
                        const auto stride2 = d3;
                        auto* lin = builder.CreateAdd(builder.CreateMul(ii, builder.getInt32(stride0)),
                                                      builder.CreateAdd(builder.CreateMul(jj, builder.getInt32(stride1)),
                                                                        builder.CreateAdd(builder.CreateMul(kk, builder.getInt32(stride2)), ll)));
                        return selectLinear(lin);
                    }

                    throw std::runtime_error("LLVMGen: TensorIR input tensor has unsupported shape kind");
                };

                (void)cached;
                return tensor_gen.emitScalar(tir, eval_scalar, load_input);
            }

            if (!term.has_indexed_access) {
                return evalKernelIRSingleScalar(term, q_index, i_index, j_index, side, /*indexed_env=*/nullptr, cached);
            }

            if (term.bound_indices.empty()) {
                throw std::runtime_error("LLVMGen: IndexedAccess term missing bound-indices metadata");
            }

            std::size_t max_id = 0;
            for (const auto& [id, ext] : term.bound_indices) {
                (void)ext;
                max_id = std::max(max_id, static_cast<std::size_t>(id));
            }
            std::vector<llvm::Value*> idx_env(max_id + 1u, nullptr);

            const auto emitSum = [&](const auto& self, std::size_t level) -> llvm::Value* {
                if (level >= term.bound_indices.size()) {
                    return evalKernelIRSingleScalar(term, q_index, i_index, j_index, side, &idx_env, cached);
                }

                const auto [id_u16, extent_u8] = term.bound_indices[level];
                const std::size_t id = static_cast<std::size_t>(id_u16);
                const std::uint32_t extent = static_cast<std::uint32_t>(extent_u8);
                if (extent == 0u) {
                    throw std::runtime_error("LLVMGen: IndexedAccess has zero loop extent");
                }

                auto* preheader = builder.GetInsertBlock();
                auto* header = llvm::BasicBlock::Create(*ctx, "idx" + std::to_string(level) + ".h", fn);
                auto* body = llvm::BasicBlock::Create(*ctx, "idx" + std::to_string(level) + ".b", fn);
                auto* latch = llvm::BasicBlock::Create(*ctx, "idx" + std::to_string(level) + ".l", fn);
                auto* exit = llvm::BasicBlock::Create(*ctx, "idx" + std::to_string(level) + ".x", fn);

                builder.CreateBr(header);
                builder.SetInsertPoint(header);
                auto* idx_phi = builder.CreatePHI(i32, 2, "idx" + std::to_string(level));
                idx_phi->addIncoming(builder.getInt32(0), preheader);
                auto* acc_phi = builder.CreatePHI(f64, 2, "idx.acc" + std::to_string(level));
                acc_phi->addIncoming(f64c(0.0), preheader);
                auto* cond = builder.CreateICmpULT(idx_phi, builder.getInt32(extent));
                builder.CreateCondBr(cond, body, exit);

                builder.SetInsertPoint(body);
                llvm::Value* prev = nullptr;
                if (id < idx_env.size()) {
                    prev = idx_env[id];
                } else {
                    idx_env.resize(id + 1u, nullptr);
                }
                idx_env[id] = idx_phi;

                auto* inner = self(self, level + 1u);

                idx_env[id] = prev;
                builder.CreateBr(latch);

                builder.SetInsertPoint(latch);
                auto* acc_next = builder.CreateFAdd(acc_phi, inner);
                auto* idx_next = builder.CreateAdd(idx_phi, builder.getInt32(1));
                builder.CreateBr(header);
                idx_phi->addIncoming(idx_next, latch);
                acc_phi->addIncoming(acc_next, latch);

                builder.SetInsertPoint(exit);
                return acc_phi;
            };

            return emitSum(emitSum, 0u);
        };

	        auto computeCachedSingle = [&](const LoweredTerm& term,
	                                       llvm::Value* q_index,
	                                       const SideView& side) -> std::vector<CodeValue> {
	            std::vector<CodeValue> cached;
	            cached.resize(term.ir.ops.size());
	            std::vector<CodeValue> values;
	            values.resize(term.ir.ops.size());
	            const auto use_counts = kernelIRUseCounts(term.ir);

            auto getChild = [&](const KernelIROp& op, std::size_t k) -> const CodeValue& {
                const auto c = term.ir.children[static_cast<std::size_t>(op.first_child) + k];
                return values[c];
            };

            for (std::size_t op_idx = 0; op_idx < term.ir.ops.size(); ++op_idx) {
                if (term.dep_mask[op_idx] != 0u) {
                    cached[op_idx] = makeZero(term.shapes[op_idx]);
                    continue;
                }
                const auto& op = term.ir.ops[op_idx];
                const auto& shape = term.shapes[op_idx];
                switch (op.type) {
                    case FormExprType::Constant: {
                        const double v = std::bit_cast<double>(op.imm0);
                        values[op_idx] = makeScalar(f64c(v));
                        break;
                    }
                    case FormExprType::ParameterRef: {
                        auto* idx64 = builder.getInt64(op.imm0);
                        values[op_idx] = makeScalar(loadRealPtrAt(side.jit_constants, idx64));
                        break;
                    }
                    case FormExprType::BoundaryIntegralRef: {
                        auto* idx64 = builder.getInt64(op.imm0);
                        values[op_idx] = makeScalar(loadRealPtrAt(side.coupled_integrals, idx64));
                        break;
                    }
                    case FormExprType::AuxiliaryStateRef: {
                        auto* idx64 = builder.getInt64(op.imm0);
                        values[op_idx] = makeScalar(loadRealPtrAt(side.coupled_aux, idx64));
                        break;
                    }
                    case FormExprType::DiscreteField:
                    case FormExprType::StateField: {
                        const int fid = unpackFieldIdImm1(op.imm1);
                        if (op.type == FormExprType::StateField && fid == 0xffff) {
                            values[op_idx] = evalCurrentSolution(side, shape, q_index);
                        } else {
                            values[op_idx] = evalDiscreteOrStateField(/*plus_side=*/false, shape, fid, q_index);
                        }
                        break;
                    }
                    case FormExprType::Coefficient: {
                        values[op_idx] = evalExternalCoefficient(side, q_index, shape, op.imm0);
                        break;
                    }
                    case FormExprType::Constitutive: {
                        values[op_idx] = evalExternalConstitutiveOutput(
                            side,
                            q_index,
                            builder.getInt32(static_cast<std::uint32_t>(external::TraceSideV1::Minus)),
                            op.imm0,
                            /*output_index=*/0u,
                            term.ir,
                            op,
                            values,
                            shape);
                        break;
                    }
                    case FormExprType::ConstitutiveOutput: {
                        if (op.child_count != 1u) {
                            throw std::runtime_error("LLVMGen: cached ConstitutiveOutput expects exactly 1 child");
                        }

                        const auto out_idx_i64 = static_cast<std::int64_t>(op.imm0);
                        if (out_idx_i64 < 0) {
                            throw std::runtime_error("LLVMGen: cached ConstitutiveOutput has negative output index");
                        }

                        const auto child_call_idx =
                            term.ir.children[static_cast<std::size_t>(op.first_child)];

                        if (out_idx_i64 == 0) {
                            values[op_idx] = values[child_call_idx];
                            break;
                        }

                        const auto& call_op = term.ir.ops[child_call_idx];
                        if (call_op.type != FormExprType::Constitutive) {
                            throw std::runtime_error("LLVMGen: cached ConstitutiveOutput child must be Constitutive");
                        }

                        values[op_idx] = evalExternalConstitutiveOutput(
                            side,
                            q_index,
                            builder.getInt32(static_cast<std::uint32_t>(external::TraceSideV1::Minus)),
                            call_op.imm0,
                            static_cast<std::uint32_t>(out_idx_i64),
                            term.ir,
                            call_op,
                            values,
                            shape);
                        break;
                    }
                    case FormExprType::Time:
                        values[op_idx] = makeScalar(side.time);
                        break;
                    case FormExprType::TimeStep:
                        values[op_idx] = makeScalar(side.dt);
                        break;
                    case FormExprType::EffectiveTimeStep:
                        values[op_idx] = makeScalar(loadEffectiveDt(side));
                        break;
                    case FormExprType::CellDiameter:
                        values[op_idx] = makeScalar(side.cell_diameter);
                        break;
                    case FormExprType::CellVolume:
                        values[op_idx] = makeScalar(side.cell_volume);
                        break;
                    case FormExprType::FacetArea:
                        values[op_idx] = makeScalar(side.facet_area);
                        break;
                    case FormExprType::CellDomainId:
                        values[op_idx] = makeScalar(builder.CreateSIToFP(side.cell_domain_id, f64));
                        break;
                    case FormExprType::Coordinate:
                        values[op_idx] =
                            loadXYZDimFromSide(side, side.physical_points_xyz, q_index, shape.dims[0],
                                               side.interleaved_qpoint_geometry_physical_offset);
                        break;
                    case FormExprType::ReferenceCoordinate:
                        values[op_idx] = loadXYZDim(side.quad_points_xyz, q_index, shape.dims[0]);
                        break;
                    case FormExprType::Normal:
                        values[op_idx] =
                            loadXYZDimFromSide(side, side.normals_xyz, q_index, shape.dims[0],
                                               side.interleaved_qpoint_geometry_normal_offset);
                        break;
                    case FormExprType::Jacobian:
                        values[op_idx] =
                            loadMatDimFromSide(side, side.jacobians, q_index, shape.dims[0],
                                               side.interleaved_qpoint_geometry_jacobian_offset);
                        break;
                    case FormExprType::JacobianInverse:
                        values[op_idx] =
                            loadMatDimFromSide(side, side.inverse_jacobians, q_index, shape.dims[0],
                                               side.interleaved_qpoint_geometry_inverse_jacobian_offset);
                        break;
                    case FormExprType::Identity: {
                        CodeValue out = makeZero(shape);
                        const auto n = static_cast<std::size_t>(shape.dims[0]);
                        for (std::size_t d = 0; d < n; ++d) {
                            out.elems[d * n + d] = f64c(1.0);
                        }
                        values[op_idx] = out;
                        break;
                    }
                    case FormExprType::JacobianDeterminant: {
                        values[op_idx] =
                            makeScalar(loadScalarFromSide(side, side.jacobian_dets, q_index,
                                                          side.interleaved_qpoint_geometry_det_offset));
                        break;
                    }
                    case FormExprType::PreviousSolutionRef: {
                        const int k = static_cast<int>(static_cast<std::int64_t>(op.imm0));
                        values[op_idx] = evalPreviousSolution(side, shape, k, q_index);
                        break;
                    }
	                    case FormExprType::Gradient: {
	                        const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
	                        const auto& kid = term.ir.ops[child_idx];

	                        auto gradFromCoeffs = [&](llvm::Value* coeffs_ptr, const std::string& loop_base) -> CodeValue {
	                            if (shape.kind == Shape::Kind::Vector) {
	                                const auto dim = static_cast<std::size_t>(shape.dims[0]);
	                                const auto sums = emitReduceSum(side.n_trial_dofs, loop_base, dim, [&](llvm::Value* j) {
	                                    auto* j64 = builder.CreateZExt(j, i64);
	                                    auto* cj = loadRealPtrAt(coeffs_ptr, j64);
	                                    const auto g =
	                                        loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                    std::vector<llvm::Value*> terms;
	                                    terms.reserve(dim);
	                                    for (std::size_t d = 0; d < dim; ++d) {
	                                        terms.push_back(builder.CreateFMul(cj, g[d]));
	                                    }
	                                    return terms;
	                                });
	                                auto* x = sums[0];
	                                auto* y = (dim > 1u) ? sums[1] : f64c(0.0);
	                                auto* z = (dim > 2u) ? sums[2] : f64c(0.0);
	                                return makeVector(static_cast<std::uint32_t>(dim), x, y, z);
	                            }
	                            if (shape.kind == Shape::Kind::Matrix) {
	                                const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                                const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                                CodeValue out = makeZero(shape);
	                                auto* dofs_per_comp =
	                                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                                for (std::size_t comp = 0; comp < vd; ++comp) {
	                                    const auto acc =
	                                        emitReduceSum(dofs_per_comp, loop_base + "_c" + std::to_string(comp), dim, [&](llvm::Value* jj) {
	                                            auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                            auto* j = builder.CreateAdd(base, jj);
	                                            auto* j64 = builder.CreateZExt(j, i64);
	                                            auto* cj = loadRealPtrAt(coeffs_ptr, j64);
	                                            const auto g =
	                                                loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                            std::vector<llvm::Value*> terms;
	                                            terms.reserve(dim);
	                                            for (std::size_t d = 0; d < dim; ++d) {
	                                                terms.push_back(builder.CreateFMul(cj, g[d]));
	                                            }
	                                            return terms;
	                                        });
	                                    for (std::size_t d = 0; d < dim; ++d) {
	                                        out.elems[comp * dim + d] = acc[d];
	                                    }
	                                }
	                                return out;
	                            }
	                            throw std::runtime_error("LLVMGen: cached grad(u) unsupported shape");
	                        };

	                        if (kid.type == FormExprType::TrialFunction) {
	                            if (!is_residual) {
	                                throw std::runtime_error(
	                                    "LLVMGen: cached grad(TrialFunction) only supports residual (current solution)");
	                            }
	                            values[op_idx] = gradFromCoeffs(side.solution_coefficients, "grad_u");
	                            break;
	                        }

	                        if (kid.type == FormExprType::TimeDerivative) {
	                            const int order = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
	                            auto* coeff0 = loadDtCoeff0(side, order);
	                            const auto dt_child_idx =
	                                term.ir.children[static_cast<std::size_t>(kid.first_child)];
	                            const auto& dt_child = term.ir.ops[dt_child_idx];

	                            auto gradCurrentSolution = [&]() -> CodeValue {
	                                return gradFromCoeffs(side.solution_coefficients, "grad_dt_u");
	                            };

	                            if (dt_child.type == FormExprType::TrialFunction) {
	                                if (!is_residual) {
	                                    throw std::runtime_error(
	                                        "LLVMGen: cached grad(dt(TrialFunction)) only supports residual (current solution)");
	                                }
	                                values[op_idx] = mul(makeScalar(coeff0), gradCurrentSolution());
	                                break;
	                            }
	                            if (dt_child.type == FormExprType::StateField) {
	                                const int fid = unpackFieldIdImm1(dt_child.imm1);
	                                if (fid == 0xffff) {
	                                    values[op_idx] = mul(makeScalar(coeff0), gradCurrentSolution());
	                                    break;
	                                }
	                            }
	                            if (dt_child.type == FormExprType::PreviousSolutionRef) {
	                                const int k = static_cast<int>(static_cast<std::int64_t>(dt_child.imm0));
	                                auto* coeffs_ptr = loadPrevSolutionCoeffsPtr(side, k);
	                                values[op_idx] =
	                                    mul(makeScalar(coeff0), gradFromCoeffs(coeffs_ptr, "grad_dt_prev" + std::to_string(k)));
	                                break;
	                            }
	                            if (dt_child.type == FormExprType::Constant) {
	                                values[op_idx] = makeZero(shape);
	                                break;
	                            }
	                            throw std::runtime_error("LLVMGen: cached grad(dt(...)) operand not supported");
	                        }

	                        if (kid.type == FormExprType::PreviousSolutionRef) {
	                            const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
	                            auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
	                            if (shape.kind == Shape::Kind::Vector) {
	                                const auto dim = static_cast<std::size_t>(shape.dims[0]);
	                                const auto sums =
	                                    emitReduceSum(side.n_trial_dofs, "prev_grad" + std::to_string(k), dim, [&](llvm::Value* j) {
	                                        auto* j64 = builder.CreateZExt(j, i64);
	                                        auto* cj = loadRealPtrAt(coeffs, j64);
	                                        const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                        std::vector<llvm::Value*> terms;
	                                        terms.reserve(dim);
	                                        for (std::size_t d = 0; d < dim; ++d) {
	                                            terms.push_back(builder.CreateFMul(cj, g[d]));
	                                        }
	                                        return terms;
	                                    });
	                                auto* x = sums[0];
	                                auto* y = (dim > 1u) ? sums[1] : f64c(0.0);
	                                auto* z = (dim > 2u) ? sums[2] : f64c(0.0);
	                                values[op_idx] = makeVector(static_cast<std::uint32_t>(dim), x, y, z);
	                                break;
	                            }
	                            if (shape.kind == Shape::Kind::Matrix) {
	                                const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                                const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                                CodeValue out = makeZero(shape);
	                                auto* dofs_per_comp =
	                                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                                for (std::size_t comp = 0; comp < vd; ++comp) {
	                                    const auto acc = emitReduceSum(dofs_per_comp,
	                                                                   "prev_grad" + std::to_string(k) + "_c" +
	                                                                       std::to_string(comp),
	                                                                   dim,
	                                                                   [&](llvm::Value* jj) {
	                                        auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                        auto* j = builder.CreateAdd(base, jj);
	                                        auto* j64 = builder.CreateZExt(j, i64);
	                                        auto* cj = loadRealPtrAt(coeffs, j64);
	                                        const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                        std::vector<llvm::Value*> terms;
	                                        terms.reserve(dim);
	                                        for (std::size_t d = 0; d < dim; ++d) {
	                                            terms.push_back(builder.CreateFMul(cj, g[d]));
	                                        }
	                                        return terms;
	                                    });
	                                    for (std::size_t d = 0; d < dim; ++d) {
	                                        out.elems[comp * dim + d] = acc[d];
	                                    }
	                                }
                                values[op_idx] = out;
                                break;
                            }
                            throw std::runtime_error("LLVMGen: cached grad(PreviousSolutionRef) unsupported shape");
                        }

                        if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                            const int fid = unpackFieldIdImm1(kid.imm1);
                            if (kid.type == FormExprType::StateField && fid == 0xffff) {
                                values[op_idx] = gradFromCoeffs(side.solution_coefficients, "grad_state_u");
                                break;
                            }
                            if (shape.kind == Shape::Kind::Vector) {
                                auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
                                auto* entry_is_null =
                                    builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok = llvm::BasicBlock::Create(*ctx, "field_grad.ok", fn);
                                auto* zero = llvm::BasicBlock::Create(*ctx, "field_grad.zero", fn);
                                auto* merge = llvm::BasicBlock::Create(*ctx, "field_grad.merge", fn);

                                builder.CreateCondBr(entry_is_null, zero, ok);

                                std::array<llvm::Value*, 3> loaded{f64c(0.0), f64c(0.0), f64c(0.0)};
                                builder.SetInsertPoint(ok);
                                auto* base = loadPtr(entry, ABIV3::field_entry_gradients_xyz_off);
                                auto* base_is_null =
                                    builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_grad.xyz.ok", fn);
                                builder.CreateCondBr(base_is_null, zero, ok2);

                                builder.SetInsertPoint(ok2);
                                const auto v = loadXYZ(base, q_index);
                                loaded = {v.elems[0], v.elems[1], v.elems[2]};
                                builder.CreateBr(merge);
                                auto* ok2_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(zero);
                                builder.CreateBr(merge);
                                auto* zero_block = builder.GetInsertBlock();

	                                builder.SetInsertPoint(merge);
	                                std::array<llvm::Value*, 3> outv{f64c(0.0), f64c(0.0), f64c(0.0)};
	                                const auto dim = static_cast<std::size_t>(shape.dims[0]);
	                                for (std::size_t d = 0; d < dim; ++d) {
	                                    auto* phi = builder.CreatePHI(f64, 2, "field_grad.v" + std::to_string(d));
	                                    phi->addIncoming(f64c(0.0), zero_block);
	                                    phi->addIncoming(loaded[d], ok2_block);
	                                    outv[d] = phi;
	                                }
	                                values[op_idx] = makeVector(shape.dims[0], outv[0], outv[1], outv[2]);
	                                break;
	                            }
	                            if (shape.kind == Shape::Kind::Matrix) {
	                                const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                                const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                                auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
	                                auto* entry_is_null =
	                                    builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok = llvm::BasicBlock::Create(*ctx, "field_jac.ok", fn);
                                auto* zero = llvm::BasicBlock::Create(*ctx, "field_jac.zero", fn);
                                auto* merge = llvm::BasicBlock::Create(*ctx, "field_jac.merge", fn);

                                builder.CreateCondBr(entry_is_null, zero, ok);

                                CodeValue loaded = makeZero(matrixShape(3u, 3u));
                                builder.SetInsertPoint(ok);
                                auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                                auto* base_is_null =
                                    builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_jac.mat.ok", fn);
                                builder.CreateCondBr(base_is_null, zero, ok2);

                                builder.SetInsertPoint(ok2);
                                loaded = loadMat3FromQ(base, q_index);
                                builder.CreateBr(merge);
                                auto* ok2_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(zero);
                                builder.CreateBr(merge);
                                auto* zero_block = builder.GetInsertBlock();

	                                builder.SetInsertPoint(merge);
	                                CodeValue out = makeZero(shape);
	                                for (std::size_t r = 0; r < vd; ++r) {
	                                    for (std::size_t c = 0; c < dim; ++c) {
	                                        auto* phi = builder.CreatePHI(f64, 2, "field_jac.a");
	                                        phi->addIncoming(f64c(0.0), zero_block);
	                                        auto* l = (r < 3u && c < 3u) ? loaded.elems[r * 3u + c] : f64c(0.0);
	                                        phi->addIncoming(l, ok2_block);
	                                        out.elems[r * dim + c] = phi;
	                                    }
	                                }
	                                values[op_idx] = out;
	                                break;
	                            }
                            throw std::runtime_error("LLVMGen: cached grad(DiscreteField) unsupported shape");
                        }

	                        if (kid.type == FormExprType::Constant) {
	                            values[op_idx] = makeZero(shape);
	                            break;
	                        }

                        // Gradient is linear; handle the common simplified pattern grad(0 * X) -> 0.
                        if (kid.type == FormExprType::Multiply && kid.child_count == 2u) {
                            const auto a_idx = term.ir.children[static_cast<std::size_t>(kid.first_child)];
                            const auto b_idx = term.ir.children[static_cast<std::size_t>(kid.first_child) + 1u];
                            const auto& aop = term.ir.ops[a_idx];
                            const auto& bop = term.ir.ops[b_idx];

                            auto isZeroConstant = [](const KernelIROp& op2) -> bool {
                                if (op2.type != FormExprType::Constant) {
                                    return false;
                                }
                                const double v = std::bit_cast<double>(op2.imm0);
                                return v == 0.0;
                            };

                            if (isZeroConstant(aop) || isZeroConstant(bop)) {
                                values[op_idx] = makeZero(shape);
                                break;
                            }
                        }

	                        throw std::runtime_error("LLVMGen: cached Gradient operand not supported");
	                    }

                    case FormExprType::Divergence: {
                        const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                        const auto& kid = term.ir.ops[child_idx];
                        const auto vd = static_cast<std::size_t>(term.shapes[child_idx].dims[0]);

                        // Matrix divergence (returns a vector with length = number of matrix rows).
                        //
                        // Currently implemented for the Navier–Stokes/Stokes VMS strong residual pattern
                        //   div( scalar * sym(grad(u)) )
                        // where the scalar factor is spatially constant and u is the current solution
                        // (StateField(INVALID_FIELD_ID)) or a PreviousSolutionRef.
                        if (shape.kind == Shape::Kind::Vector) {
                            auto isSpatiallyConstantScalar = [&](auto&& self, std::size_t idx) -> bool {
                                const auto& op2 = term.ir.ops[idx];
                                switch (op2.type) {
                                    case FormExprType::Constant:
                                    case FormExprType::ParameterRef:
                                    case FormExprType::Time:
                                    case FormExprType::TimeStep:
                                    case FormExprType::EffectiveTimeStep:
                                        return true;
                                    case FormExprType::Negate:
                                    case FormExprType::AbsoluteValue:
                                    case FormExprType::Sign:
                                    case FormExprType::Sqrt:
                                    case FormExprType::Exp:
                                    case FormExprType::Log:
                                        break;
                                    case FormExprType::Add:
                                    case FormExprType::Subtract:
                                    case FormExprType::Multiply:
                                    case FormExprType::Divide:
                                    case FormExprType::Power:
                                    case FormExprType::Minimum:
                                    case FormExprType::Maximum:
                                        break;
                                    default:
                                        return false;
                                }

                                for (std::size_t k = 0; k < static_cast<std::size_t>(op2.child_count); ++k) {
                                    const auto c = term.ir.children[static_cast<std::size_t>(op2.first_child) + k];
                                    if (!self(self, c)) {
                                        return false;
                                    }
                                }
                                return true;
                            };

                            auto traceHessBasis = [&](const CodeValue& H, std::size_t dim) -> llvm::Value* {
                                llvm::Value* tr = H.elems[0];
                                if (dim >= 2) tr = builder.CreateFAdd(tr, H.elems[4]);
                                if (dim >= 3) tr = builder.CreateFAdd(tr, H.elems[8]);
                                return tr;
                            };

                            auto divSymGradCurrentSolution = [&](llvm::Value* coeffs_ptr,
                                                                 std::size_t rows,
                                                                 std::size_t dim) -> CodeValue {
                                CodeValue out = makeZero(vectorShape(static_cast<std::uint32_t>(rows)));
                                auto* dofs_per_comp =
                                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(rows)));

                                const std::size_t n = std::min(rows, dim);
	                                for (std::size_t r = 0; r < rows; ++r) {
	                                    if (r >= dim) {
	                                        out.elems[r] = f64c(0.0);
	                                        continue;
	                                    }

	                                    const auto loop_name = "lap_u_c" + std::to_string(r);
	                                    auto* lap = emitReduceSumScalar(dofs_per_comp, loop_name, [&](llvm::Value* jj) -> llvm::Value* {
	                                        auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(r)), dofs_per_comp);
	                                        auto* j = builder.CreateAdd(base, jj);
	                                        auto* j64 = builder.CreateZExt(j, i64);
	                                        auto* cj = loadRealPtrAt(coeffs_ptr, j64);
	                                        const auto H = loadMat3FromTable(side.trial_phys_hessians, side.n_qpts, j, q_index);
	                                        auto* tr = traceHessBasis(H, dim);
	                                        return builder.CreateFMul(cj, tr);
	                                    });

	                                    llvm::Value* gd = f64c(0.0);
	                                    for (std::size_t d = 0; d < n; ++d) {
	                                        const auto loop_name = "gd_u_r" + std::to_string(r) + "_d" + std::to_string(d);
	                                        auto* acc = emitReduceSumScalar(dofs_per_comp, loop_name, [&](llvm::Value* jj) -> llvm::Value* {
	                                            auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(d)), dofs_per_comp);
	                                            auto* j = builder.CreateAdd(base, jj);
	                                            auto* j64 = builder.CreateZExt(j, i64);
	                                            auto* cj = loadRealPtrAt(coeffs_ptr, j64);
	                                            const auto H = loadMat3FromTable(side.trial_phys_hessians, side.n_qpts, j, q_index);
	                                            const auto idxH = static_cast<std::size_t>(r * 3u + d);
	                                            return builder.CreateFMul(cj, H.elems[idxH]);
	                                        });
	                                        gd = builder.CreateFAdd(gd, acc);
	                                    }

                                    out.elems[r] = builder.CreateFMul(f64c(0.5), builder.CreateFAdd(lap, gd));
                                }

                                return out;
                            };

                            auto divSymGradFromField = [&](int fid, std::size_t rows2, std::size_t dim2) -> CodeValue {
                                CodeValue out = makeZero(vectorShape(static_cast<std::uint32_t>(rows2)));

                                auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
                                auto* entry_is_null = builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok = llvm::BasicBlock::Create(*ctx, "field_divsym.ok", fn);
                                auto* zero = llvm::BasicBlock::Create(*ctx, "field_divsym.zero", fn);
                                auto* merge = llvm::BasicBlock::Create(*ctx, "field_divsym.merge", fn);
                                builder.CreateCondBr(entry_is_null, zero, ok);

                                std::array<llvm::Value*, 3> computed{f64c(0.0), f64c(0.0), f64c(0.0)};
                                builder.SetInsertPoint(ok);
                                auto* base = loadPtr(entry, ABIV3::field_entry_component_hessians_off);
                                auto* base_is_null = builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_divsym.ok2", fn);
                                builder.CreateCondBr(base_is_null, zero, ok2);

                                builder.SetInsertPoint(ok2);
                                auto* vd = loadU32(entry, ABIV3::field_entry_value_dim_off);
                                const std::size_t n2 = std::min(rows2, dim2);
                                auto* vd_ok = builder.CreateICmpUGE(vd, builder.getInt32(static_cast<std::uint32_t>(n2)));
                                auto* ok3 = llvm::BasicBlock::Create(*ctx, "field_divsym.ok3", fn);
                                builder.CreateCondBr(vd_ok, ok3, zero);

                                builder.SetInsertPoint(ok3);
                                auto loadHessElem = [&](std::size_t comp,
                                                        std::size_t d0,
                                                        std::size_t d1) -> llvm::Value* {
                                    auto* q64 = builder.CreateZExt(q_index, i64);
                                    auto* vd64 = builder.CreateZExt(vd, i64);
                                    auto* idx = builder.CreateAdd(builder.CreateMul(q64, vd64),
                                                                  builder.getInt64(static_cast<std::uint64_t>(comp)));
                                    auto* base9 = builder.CreateMul(idx, builder.getInt64(9));
                                    const auto elem = static_cast<std::uint64_t>(d0 * 3u + d1);
                                    return loadRealPtrAt(base, builder.CreateAdd(base9, builder.getInt64(elem)));
                                };

                                for (std::size_t r = 0; r < rows2; ++r) {
                                    if (r >= dim2) {
                                        computed[r] = f64c(0.0);
                                        continue;
                                    }

                                    llvm::Value* lap = loadHessElem(r, 0u, 0u);
                                    if (dim2 >= 2u) lap = builder.CreateFAdd(lap, loadHessElem(r, 1u, 1u));
                                    if (dim2 >= 3u) lap = builder.CreateFAdd(lap, loadHessElem(r, 2u, 2u));

                                    llvm::Value* gd = f64c(0.0);
                                    for (std::size_t d = 0; d < n2; ++d) {
                                        gd = builder.CreateFAdd(gd, loadHessElem(d, r, d));
                                    }

                                    computed[r] = builder.CreateFMul(f64c(0.5), builder.CreateFAdd(lap, gd));
                                }

                                builder.CreateBr(merge);
                                auto* ok3_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(zero);
                                builder.CreateBr(merge);
                                auto* zero_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(merge);
                                for (std::size_t r = 0; r < rows2; ++r) {
                                    auto* phi = builder.CreatePHI(f64, 2, "field_divsym." + std::to_string(r));
                                    phi->addIncoming(f64c(0.0), zero_block);
                                    phi->addIncoming(computed[r], ok3_block);
                                    out.elems[r] = phi;
                                }
                                return out;
                            };

                            auto divSymGradForU = [&](auto&& self,
                                                      std::size_t u_idx,
                                                      std::size_t rows,
                                                      std::size_t dim) -> CodeValue {
                                const auto& uop = term.ir.ops[u_idx];
                                if (uop.type == FormExprType::TrialFunction) {
                                    if (!is_residual) {
                                        throw std::runtime_error(
                                            "LLVMGen: cached div(sym(grad(TrialFunction))) only supports residual (current solution)");
                                    }
                                    return divSymGradCurrentSolution(side.solution_coefficients, rows, dim);
                                }
                                if (uop.type == FormExprType::DiscreteField) {
                                    const int fid = unpackFieldIdImm1(uop.imm1);
                                    return divSymGradFromField(fid, rows, dim);
                                }
                                if (uop.type == FormExprType::StateField) {
                                    const int fid = unpackFieldIdImm1(uop.imm1);
                                    if (fid == 0xffff) {
                                        return divSymGradCurrentSolution(side.solution_coefficients, rows, dim);
                                    }
                                    return divSymGradFromField(fid, rows, dim);
                                }
                                if (uop.type == FormExprType::PreviousSolutionRef) {
                                    const int k = static_cast<int>(static_cast<std::int64_t>(uop.imm0));
                                    auto* coeffs_ptr = loadPrevSolutionCoeffsPtr(side, k);
                                    return divSymGradCurrentSolution(coeffs_ptr, rows, dim);
                                }
                                if (uop.type == FormExprType::Negate) {
                                    const auto a_idx = term.ir.children[static_cast<std::size_t>(uop.first_child)];
                                    return neg(self(self, a_idx, rows, dim));
                                }
                                if (uop.type == FormExprType::Add || uop.type == FormExprType::Subtract) {
                                    if (uop.child_count != 2u) {
                                        throw std::runtime_error("LLVMGen: cached div(sym(grad(Add/Subtract))) expects 2 children");
                                    }
                                    const auto a_idx = term.ir.children[static_cast<std::size_t>(uop.first_child)];
                                    const auto b_idx = term.ir.children[static_cast<std::size_t>(uop.first_child) + 1u];
                                    return (uop.type == FormExprType::Add) ? add(self(self, a_idx, rows, dim), self(self, b_idx, rows, dim))
                                                                         : sub(self(self, a_idx, rows, dim), self(self, b_idx, rows, dim));
                                }
                                if (uop.type == FormExprType::Multiply) {
                                    if (uop.child_count != 2u) {
                                        throw std::runtime_error("LLVMGen: cached div(sym(grad(Multiply))) expects 2 children");
                                    }
                                    const auto a_idx = term.ir.children[static_cast<std::size_t>(uop.first_child)];
                                    const auto b_idx = term.ir.children[static_cast<std::size_t>(uop.first_child) + 1u];
                                    const auto& ash = term.shapes[a_idx];
                                    const auto& bsh = term.shapes[b_idx];
                                    const bool a_scalar = (ash.kind == Shape::Kind::Scalar);
                                    const bool b_scalar = (bsh.kind == Shape::Kind::Scalar);
                                    const bool a_vec = (ash.kind == Shape::Kind::Vector);
                                    const bool b_vec = (bsh.kind == Shape::Kind::Vector);
                                    if (a_scalar && b_vec) {
                                        if (!isSpatiallyConstantScalar(isSpatiallyConstantScalar, a_idx)) {
                                            throw std::runtime_error(
                                                "LLVMGen: cached div(sym(grad(s*u))) requires spatially-constant scalar s in JIT");
                                        }
                                        return mul(values[a_idx], self(self, b_idx, rows, dim));
                                    }
                                    if (b_scalar && a_vec) {
                                        if (!isSpatiallyConstantScalar(isSpatiallyConstantScalar, b_idx)) {
                                            throw std::runtime_error(
                                                "LLVMGen: cached div(sym(grad(u*s))) requires spatially-constant scalar s in JIT");
                                        }
                                        return mul(values[b_idx], self(self, a_idx, rows, dim));
                                    }
                                }
                                throw std::runtime_error(
                                    "LLVMGen: cached div(sym(grad(u))) expects u = TrialFunction, DiscreteField, StateField, PreviousSolutionRef, or linear scalar*vector ops");
                            };

                            auto evalDivMatrix = [&](auto&& self, std::size_t m_idx) -> CodeValue {
                                const auto& mop = term.ir.ops[m_idx];
                                const auto& msh = term.shapes[m_idx];
                                if (msh.kind != Shape::Kind::Matrix) {
                                    throw std::runtime_error("LLVMGen: internal error: div(matrix) expected Matrix operand");
                                }
                                const std::size_t rows = static_cast<std::size_t>(msh.dims[0]);
                                const std::size_t dim = static_cast<std::size_t>(msh.dims[1]);

                                switch (mop.type) {
                                    case FormExprType::SymmetricPart: {
                                        const auto a_idx = term.ir.children[static_cast<std::size_t>(mop.first_child)];
                                        const auto& aop = term.ir.ops[a_idx];
                                        if (aop.type != FormExprType::Gradient) {
                                            throw std::runtime_error("LLVMGen: div(sym(A)) currently only supports A=grad(u)");
                                        }
                                        const auto u_idx = term.ir.children[static_cast<std::size_t>(aop.first_child)];
                                        return divSymGradForU(divSymGradForU, u_idx, rows, dim);
                                    }
                                    case FormExprType::Multiply: {
                                        if (mop.child_count != 2u) {
                                            throw std::runtime_error("LLVMGen: div(Multiply) expects 2 children");
                                        }
                                        const auto a_idx = term.ir.children[static_cast<std::size_t>(mop.first_child)];
                                        const auto b_idx = term.ir.children[static_cast<std::size_t>(mop.first_child) + 1u];
                                        const auto& ash = term.shapes[a_idx];
                                        const auto& bsh = term.shapes[b_idx];

                                        const bool a_scalar = (ash.kind == Shape::Kind::Scalar);
                                        const bool b_scalar = (bsh.kind == Shape::Kind::Scalar);
                                        const bool a_mat = (ash.kind == Shape::Kind::Matrix);
                                        const bool b_mat = (bsh.kind == Shape::Kind::Matrix);

                                        if (a_scalar && b_mat) {
                                            if (!isSpatiallyConstantScalar(isSpatiallyConstantScalar, a_idx)) {
                                                throw std::runtime_error("LLVMGen: div(s*A) requires spatially-constant scalar s in JIT");
                                            }
                                            return mul(values[a_idx], self(self, b_idx));
                                        }
                                        if (b_scalar && a_mat) {
                                            if (!isSpatiallyConstantScalar(isSpatiallyConstantScalar, b_idx)) {
                                                throw std::runtime_error("LLVMGen: div(A*s) requires spatially-constant scalar s in JIT");
                                            }
                                            return mul(values[b_idx], self(self, a_idx));
                                        }
                                        throw std::runtime_error("LLVMGen: div(Multiply) currently supports scalar-matrix products only");
                                    }
                                    case FormExprType::Add: {
                                        const auto a_idx = term.ir.children[static_cast<std::size_t>(mop.first_child)];
                                        const auto b_idx = term.ir.children[static_cast<std::size_t>(mop.first_child) + 1u];
                                        return add(self(self, a_idx), self(self, b_idx));
                                    }
                                    case FormExprType::Subtract: {
                                        const auto a_idx = term.ir.children[static_cast<std::size_t>(mop.first_child)];
                                        const auto b_idx = term.ir.children[static_cast<std::size_t>(mop.first_child) + 1u];
                                        return sub(self(self, a_idx), self(self, b_idx));
                                    }
                                    default:
                                        break;
                                }

                                throw std::runtime_error("LLVMGen: cached div(matrix) operand not supported");
                            };

                            values[op_idx] = evalDivMatrix(evalDivMatrix, child_idx);
                            break;
                        }

	                        if (kid.type == FormExprType::PreviousSolutionRef) {
	                            const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
	                            auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
	                            auto* dofs_per_comp =
	                                builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
		                            llvm::Value* div = f64c(0.0);
		                            for (std::size_t comp = 0; comp < vd; ++comp) {
		                                auto* acc = emitReduceSumScalar(
		                                    dofs_per_comp,
		                                    "prev_div" + std::to_string(k) + "_c" + std::to_string(comp),
		                                    [&](llvm::Value* jj) -> llvm::Value* {
		                                        auto* base =
		                                            builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
		                                        auto* j = builder.CreateAdd(base, jj);
		                                        auto* j64 = builder.CreateZExt(j, i64);
		                                        auto* cj = loadRealPtrAt(coeffs, j64);
		                                        const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
		                                        return builder.CreateFMul(cj, g[comp]);
		                                    });
		                                div = builder.CreateFAdd(div, acc);
		                            }
		                            values[op_idx] = makeScalar(div);
		                            break;
	                        }

	                        if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
	                            const int fid = unpackFieldIdImm1(kid.imm1);
	                            if (kid.type == FormExprType::StateField && fid == 0xffff) {
	                                // StateField(INVALID_FIELD_ID) represents the current solution state u.
	                                // Compute div(u_h) directly from current solution coefficients.
	                                auto* coeffs = side.solution_coefficients;
	                                auto* uses_vec_basis =
	                                    builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
	                                auto* vb = llvm::BasicBlock::Create(*ctx, "div.state_u.vec_basis", fn);
	                                auto* sb = llvm::BasicBlock::Create(*ctx, "div.state_u.scalar_basis", fn);
	                                auto* merge = llvm::BasicBlock::Create(*ctx, "div.state_u.merge", fn);

	                                builder.CreateCondBr(uses_vec_basis, vb, sb);

	                                llvm::Value* div_vb = f64c(0.0);
	                                builder.SetInsertPoint(vb);
	                                div_vb = emitReduceSumScalar(
	                                    side.n_trial_dofs, "div_state_u_vb", [&](llvm::Value* j) -> llvm::Value* {
	                                        auto* j64 = builder.CreateZExt(j, i64);
	                                        auto* cj = loadRealPtrAt(coeffs, j64);
	                                        auto* div_phi = loadBasisScalar(side, side.trial_basis_divs, j, q_index);
	                                        return builder.CreateFMul(cj, div_phi);
	                                    });
	                                builder.CreateBr(merge);
	                                auto* vb_block = builder.GetInsertBlock();

	                                llvm::Value* div_sb = f64c(0.0);
	                                builder.SetInsertPoint(sb);
	                                auto* dofs_per_comp =
	                                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                                llvm::Value* div_sum = f64c(0.0);
	                                for (std::size_t comp = 0; comp < vd; ++comp) {
	                                    auto* acc = emitReduceSumScalar(
	                                        dofs_per_comp,
	                                        "div_state_u_c" + std::to_string(comp),
	                                        [&](llvm::Value* jj) -> llvm::Value* {
	                                            auto* base =
	                                                builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                            auto* j = builder.CreateAdd(base, jj);
	                                            auto* j64 = builder.CreateZExt(j, i64);
	                                            auto* cj = loadRealPtrAt(coeffs, j64);
	                                            const auto g =
	                                                loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                            return builder.CreateFMul(cj, g[comp]);
	                                        });
	                                    div_sum = builder.CreateFAdd(div_sum, acc);
	                                }
	                                div_sb = div_sum;
	                                builder.CreateBr(merge);
	                                auto* sb_block = builder.GetInsertBlock();

	                                builder.SetInsertPoint(merge);
	                                auto* phi = builder.CreatePHI(f64, 2, "div.state_u");
	                                phi->addIncoming(div_vb, vb_block);
	                                phi->addIncoming(div_sb, sb_block);
	                                values[op_idx] = makeScalar(phi);
	                                break;
	                            }
	                            auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
                            auto* entry_is_null =
                                builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok = llvm::BasicBlock::Create(*ctx, "field_div.ok", fn);
                            auto* zero = llvm::BasicBlock::Create(*ctx, "field_div.zero", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "field_div.merge", fn);

                            builder.CreateCondBr(entry_is_null, zero, ok);

                            llvm::Value* div = f64c(0.0);
                            builder.SetInsertPoint(ok);
                            auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                            auto* base_is_null =
                                builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_div.mat.ok", fn);
                            builder.CreateCondBr(base_is_null, zero, ok2);

                            builder.SetInsertPoint(ok2);
                            const auto J = loadMat3FromQ(base, q_index);
                            div = J.elems[0];
                            if (vd >= 2) div = builder.CreateFAdd(div, J.elems[4]);
                            if (vd >= 3) div = builder.CreateFAdd(div, J.elems[8]);
                            builder.CreateBr(merge);
                            auto* ok2_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(zero);
                            builder.CreateBr(merge);
                            auto* zero_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(merge);
                            auto* phi = builder.CreatePHI(f64, 2, "field_div");
                            phi->addIncoming(f64c(0.0), zero_block);
                            phi->addIncoming(div, ok2_block);
	                            values[op_idx] = makeScalar(phi);
	                            break;
	                        }

                        // Divergence is linear; handle the common simplified pattern div(0 * X) -> 0.
                        if (kid.type == FormExprType::Multiply && kid.child_count == 2u) {
                            const auto a_idx = term.ir.children[static_cast<std::size_t>(kid.first_child)];
                            const auto b_idx = term.ir.children[static_cast<std::size_t>(kid.first_child) + 1u];
                            const auto& aop = term.ir.ops[a_idx];
                            const auto& bop = term.ir.ops[b_idx];

                            auto isZeroConstant = [](const KernelIROp& op2) -> bool {
                                if (op2.type != FormExprType::Constant) {
                                    return false;
                                }
                                const double v = std::bit_cast<double>(op2.imm0);
                                return v == 0.0;
                            };

                            if (isZeroConstant(aop) || isZeroConstant(bop)) {
                                values[op_idx] = makeScalar(f64c(0.0));
                                break;
                            }
                        }

	                        if (kid.type == FormExprType::Constant) {
	                            values[op_idx] = makeScalar(f64c(0.0));
	                            break;
	                        }

                        throw std::runtime_error("LLVMGen: cached Divergence operand not supported");
                    }

                    case FormExprType::Curl: {
                        const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                        const auto& kid = term.ir.ops[child_idx];
                        const auto vd = static_cast<std::size_t>(term.shapes[child_idx].dims[0]);

                        if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                            const int fid = unpackFieldIdImm1(kid.imm1);
                            auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
                            auto* entry_is_null =
                                builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok = llvm::BasicBlock::Create(*ctx, "field_curl.ok", fn);
                            auto* zero = llvm::BasicBlock::Create(*ctx, "field_curl.zero", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "field_curl.merge", fn);

                            builder.CreateCondBr(entry_is_null, zero, ok);

                            std::array<llvm::Value*, 3> c{f64c(0.0), f64c(0.0), f64c(0.0)};
                            builder.SetInsertPoint(ok);
                            auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                            auto* base_is_null =
                                builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_curl.mat.ok", fn);
                            builder.CreateCondBr(base_is_null, zero, ok2);

                            builder.SetInsertPoint(ok2);
                            const auto J = loadMat3FromQ(base, q_index);
                            auto d = [&](std::size_t comp, std::size_t wrt) -> llvm::Value* {
                                if (comp >= vd || comp >= 3u || wrt >= 3u) return f64c(0.0);
                                return J.elems[comp * 3 + wrt];
                            };
                            c[0] = builder.CreateFSub(d(2, 1), d(1, 2));
                            c[1] = builder.CreateFSub(d(0, 2), d(2, 0));
                            c[2] = builder.CreateFSub(d(1, 0), d(0, 1));
                            builder.CreateBr(merge);
                            auto* ok2_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(zero);
                            builder.CreateBr(merge);
                            auto* zero_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(merge);
                            std::array<llvm::Value*, 3> out{f64c(0.0), f64c(0.0), f64c(0.0)};
                            for (std::size_t d0 = 0; d0 < 3; ++d0) {
                                auto* phi = builder.CreatePHI(f64, 2, "field_curl." + std::to_string(d0));
                                phi->addIncoming(f64c(0.0), zero_block);
                                phi->addIncoming(c[d0], ok2_block);
                                out[d0] = phi;
                            }
                            values[op_idx] = makeVector(3u, out[0], out[1], out[2]);
                            break;
                        }

                        if (kid.type == FormExprType::PreviousSolutionRef) {
                            const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
                            auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
                            auto* dofs_per_comp =
                                builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));

                            std::array<std::array<llvm::Value*, 3>, 3> J{};
                            for (auto& row : J) {
                                row = {f64c(0.0), f64c(0.0), f64c(0.0)};
                            }

	                            for (std::size_t comp = 0; comp < std::min<std::size_t>(vd, 3u); ++comp) {
	                                const auto acc =
	                                    emitReduceSum(dofs_per_comp,
	                                                 "prev_curl" + std::to_string(k) + "_c" + std::to_string(comp),
	                                                 3u,
	                                                 [&](llvm::Value* jj) {
	                                                     auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                                     auto* j = builder.CreateAdd(base, jj);
	                                                     auto* j64 = builder.CreateZExt(j, i64);
	                                                     auto* cj = loadRealPtrAt(coeffs, j64);
	                                                     const auto g =
	                                                         loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                                     std::vector<llvm::Value*> terms;
	                                                     terms.reserve(3u);
	                                                     for (std::size_t d = 0; d < 3u; ++d) {
	                                                         terms.push_back(builder.CreateFMul(cj, g[d]));
	                                                     }
	                                                     return terms;
	                                                 });
	                                J[comp] = {acc[0], acc[1], acc[2]};
	                            }

                            auto* cx = builder.CreateFSub(J[2][1], J[1][2]);
                            auto* cy = builder.CreateFSub(J[0][2], J[2][0]);
                            auto* cz = builder.CreateFSub(J[1][0], J[0][1]);
                            values[op_idx] = makeVector(3u, cx, cy, cz);
                            break;
                        }

                        if (kid.type == FormExprType::Constant) {
                            values[op_idx] = makeZero(vectorShape(3u));
                            break;
                        }

                        throw std::runtime_error("LLVMGen: cached Curl operand not supported");
                    }

	                    case FormExprType::Hessian: {
	                        const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
	                        const auto& kid = term.ir.ops[child_idx];
	                        const auto dim_u32 = shape.dims[0];
	                        const auto mat_len = elemCount(shape);

		                        if (kid.type == FormExprType::PreviousSolutionRef) {
		                            const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
		                            auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
		                            CodeValue out = makeZero(shape);
		                            const auto sums =
		                                emitReduceSum(side.n_trial_dofs, "prev_hess" + std::to_string(k), mat_len, [&](llvm::Value* j) {
		                                    auto* j64 = builder.CreateZExt(j, i64);
		                                    auto* cj = loadRealPtrAt(coeffs, j64);
		                                    const auto H = loadMatDimFromTable(
		                                        side.trial_phys_hessians, side.n_qpts, j, q_index, dim_u32);
		                                    std::vector<llvm::Value*> terms;
		                                    terms.reserve(mat_len);
		                                    for (std::size_t i = 0; i < mat_len; ++i) {
		                                        terms.push_back(builder.CreateFMul(cj, H.elems[i]));
		                                    }
		                                    return terms;
		                                });
		                            for (std::size_t i = 0; i < mat_len; ++i) {
		                                out.elems[i] = sums[i];
		                            }
		                            values[op_idx] = out;
		                            break;
		                        }

                        if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                            const int fid = unpackFieldIdImm1(kid.imm1);
                            auto* entry = fieldEntryPtrFor(/*plus_side=*/false, fid);
                            auto* entry_is_null =
                                builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok = llvm::BasicBlock::Create(*ctx, "field_hess.ok", fn);
                            auto* z = llvm::BasicBlock::Create(*ctx, "field_hess.zero", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "field_hess.merge", fn);

	                            builder.CreateCondBr(entry_is_null, z, ok);

	                            CodeValue loaded = makeZero(shape);
	                            builder.SetInsertPoint(ok);
	                            auto* base = loadPtr(entry, ABIV3::field_entry_hessians_off);
	                            auto* base_is_null =
	                                builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_hess.mat.ok", fn);
                            builder.CreateCondBr(base_is_null, z, ok2);

	                            builder.SetInsertPoint(ok2);
	                            loaded = loadMatDimFromQ(base, q_index, dim_u32);
	                            builder.CreateBr(merge);
	                            auto* ok2_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(z);
                            builder.CreateBr(merge);
                            auto* zero_block = builder.GetInsertBlock();

	                            builder.SetInsertPoint(merge);
	                            CodeValue out = makeZero(shape);
	                            for (std::size_t i = 0; i < mat_len; ++i) {
	                                auto* phi = builder.CreatePHI(f64, 2, "field_hess." + std::to_string(i));
	                                phi->addIncoming(f64c(0.0), zero_block);
	                                phi->addIncoming(loaded.elems[i], ok2_block);
	                                out.elems[i] = phi;
	                            }
	                            values[op_idx] = out;
	                            break;
	                        }

	                        if (kid.type == FormExprType::Constant) {
	                            values[op_idx] = makeZero(shape);
	                            break;
	                        }

                        throw std::runtime_error("LLVMGen: cached Hessian operand not supported");
                    }

                    case FormExprType::TimeDerivative: {
                        const int order = static_cast<int>(static_cast<std::int64_t>(op.imm0));
                        CodeValue acc = mul(makeScalar(loadDtCoeff(side, order, 0)), getChild(op, 0));

                        const auto child_op_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                        const auto& kid = term.ir.ops[child_op_idx];
                        if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                            const int fid = unpackFieldIdImm1(kid.imm1);
                            if (kid.type == FormExprType::StateField && fid == 0xffff) {
                                for (int k = 1; k <= static_cast<int>(assembly::jit::kMaxPreviousSolutionsV6); ++k) {
                                    acc = add(acc,
                                              mul(makeScalar(loadDtCoeff(side, order, k)),
                                                  evalPreviousSolution(side, shape, k, q_index)));
                                }
                            } else {
                                for (int k = 1; k <= static_cast<int>(assembly::jit::kMaxPreviousSolutionsV6); ++k) {
                                    acc = add(acc,
                                              mul(makeScalar(loadDtCoeff(side, order, k)),
                                                  evalDiscreteOrStateFieldHistoryK(side, /*plus_side=*/false, shape, fid, k, q_index)));
                                }
                            }
                        }

                        values[op_idx] = acc;
                        break;
                    }
                    case FormExprType::MaterialStateOldRef:
                        values[op_idx] = makeScalar(loadMaterialStateReal(side.material_state_old_base,
                                                                         side.material_state_stride_bytes,
                                                                         q_index,
                                                                         static_cast<std::uint64_t>(op.imm0)));
                        break;
                    case FormExprType::MaterialStateWorkRef:
                        values[op_idx] = makeScalar(loadMaterialStateReal(side.material_state_work_base,
                                                                         side.material_state_stride_bytes,
                                                                         q_index,
                                                                         static_cast<std::uint64_t>(op.imm0)));
                        break;
                    case FormExprType::Negate:
                        values[op_idx] = neg(getChild(op, 0));
                        break;
	                    case FormExprType::Add: {
	                        if (shape.kind == Shape::Kind::Scalar) {
	                            if (auto* fused = tryFuseMulAddScalar(op.type, term, use_counts, op, values)) {
	                                values[op_idx] = makeScalar(fused);
	                                break;
	                            }
	                        }
	                        values[op_idx] = add(getChild(op, 0), getChild(op, 1));
	                        break;
	                    }
	                    case FormExprType::Subtract: {
	                        if (shape.kind == Shape::Kind::Scalar) {
	                            if (auto* fused = tryFuseMulAddScalar(op.type, term, use_counts, op, values)) {
	                                values[op_idx] = makeScalar(fused);
	                                break;
	                            }
	                        }
	                        values[op_idx] = sub(getChild(op, 0), getChild(op, 1));
	                        break;
	                    }
                    case FormExprType::Multiply:
                        values[op_idx] = mul(getChild(op, 0), getChild(op, 1));
                        break;
                    case FormExprType::Divide:
                        values[op_idx] = div(getChild(op, 0), getChild(op, 1));
                        break;
                    case FormExprType::InnerProduct:
                        values[op_idx] = inner(getChild(op, 0), getChild(op, 1));
                        break;
                    case FormExprType::DoubleContraction:
                        values[op_idx] = doubleContraction(getChild(op, 0), getChild(op, 1));
                        break;
                    case FormExprType::OuterProduct:
                        values[op_idx] = outer(getChild(op, 0), getChild(op, 1));
                        break;
                    case FormExprType::CrossProduct:
                        values[op_idx] = cross(getChild(op, 0), getChild(op, 1));
                        break;
                    case FormExprType::Power:
                        values[op_idx] = powv(getChild(op, 0), getChild(op, 1));
                        break;
                    case FormExprType::Minimum:
                        values[op_idx] = makeScalar(f_min(getChild(op, 0).elems[0], getChild(op, 1).elems[0]));
                        break;
                    case FormExprType::Maximum:
                        values[op_idx] = makeScalar(f_max(getChild(op, 0).elems[0], getChild(op, 1).elems[0]));
                        break;
                    case FormExprType::Less:
                    case FormExprType::LessEqual:
                    case FormExprType::Greater:
                    case FormExprType::GreaterEqual:
                    case FormExprType::Equal:
                    case FormExprType::NotEqual:
                        values[op_idx] = cmp(op.type, getChild(op, 0), getChild(op, 1));
                        break;
                    case FormExprType::Conditional:
                        values[op_idx] = conditional(getChild(op, 0), getChild(op, 1), getChild(op, 2));
                        break;
                    case FormExprType::AsVector: {
                        CodeValue out = makeZero(shape);
                        for (std::size_t k = 0; k < op.child_count; ++k) {
                            out.elems[k] = getChild(op, k).elems[0];
                        }
                        values[op_idx] = out;
                        break;
                    }
                    case FormExprType::AsTensor: {
                        const auto rows = unpackU32Lo(op.imm0);
                        const auto cols = unpackU32Hi(op.imm0);
                        CodeValue out = makeMatrix(rows, cols);
                        for (std::size_t r = 0; r < rows; ++r) {
                            for (std::size_t c = 0; c < cols; ++c) {
                                const auto k = r * cols + c;
                                out.elems[k] = getChild(op, k).elems[0];
                            }
                        }
                        values[op_idx] = out;
                        break;
                    }
                    case FormExprType::Component: {
                        const auto i = static_cast<std::int32_t>(unpackU32Lo(op.imm0));
                        const auto j = static_cast<std::int32_t>(unpackU32Hi(op.imm0));
                        const auto& a = getChild(op, 0);
                        if (a.shape.kind == Shape::Kind::Scalar) {
                            values[op_idx] = a;
                            break;
                        }
                        if (a.shape.kind == Shape::Kind::Vector) {
                            if (j >= 0) throw std::runtime_error("LLVMGen: component(v,i,j) invalid for vector");
                            values[op_idx] = makeScalar(getVecComp(a, static_cast<std::size_t>(i)));
                            break;
                        }
                        if (a.shape.kind == Shape::Kind::Matrix) {
                            if (j < 0) throw std::runtime_error("LLVMGen: component(A,i) missing column index");
                            const auto rows = static_cast<std::size_t>(a.shape.dims[0]);
                            const auto cols = static_cast<std::size_t>(a.shape.dims[1]);
                            const auto rr = static_cast<std::size_t>(i);
                            const auto cc = static_cast<std::size_t>(j);
                            if (rr >= rows || cc >= cols) throw std::runtime_error("LLVMGen: component(A,i,j) index out of range");
                            values[op_idx] = makeScalar(a.elems[rr * cols + cc]);
                            break;
                        }
                        throw std::runtime_error("LLVMGen: component() unsupported operand kind");
                    }
                    case FormExprType::Transpose:
                        values[op_idx] = transpose(getChild(op, 0));
                        break;
                    case FormExprType::Trace:
                        values[op_idx] = trace(getChild(op, 0));
                        break;
                    case FormExprType::Determinant:
                        values[op_idx] = det(getChild(op, 0));
                        break;
                    case FormExprType::Cofactor:
                        values[op_idx] = cofactor(getChild(op, 0));
                        break;
                    case FormExprType::Inverse:
                        values[op_idx] = inv(getChild(op, 0));
                        break;
                    case FormExprType::SymmetricPart:
                        values[op_idx] = symOrSkew(true, getChild(op, 0));
                        break;
                    case FormExprType::SkewPart:
                        values[op_idx] = symOrSkew(false, getChild(op, 0));
                        break;
                    case FormExprType::Deviator:
                        values[op_idx] = deviator(getChild(op, 0));
                        break;
                    case FormExprType::Norm:
                        values[op_idx] = norm(getChild(op, 0));
                        break;
                    case FormExprType::Normalize:
                        values[op_idx] = normalize(getChild(op, 0));
                        break;
                    case FormExprType::AbsoluteValue:
                        values[op_idx] = absScalar(getChild(op, 0));
                        break;
                    case FormExprType::Sign:
                        values[op_idx] = signScalar(getChild(op, 0));
                        break;
                    case FormExprType::Sqrt:
                        values[op_idx] = evalSqrt(getChild(op, 0));
                        break;
                    case FormExprType::Exp:
                        values[op_idx] = evalExp(getChild(op, 0));
                        break;
                    case FormExprType::Log:
                        values[op_idx] = evalLog(getChild(op, 0));
                        break;

	                    case FormExprType::MatrixExponential:
	                        values[op_idx] = emitMatrixExp(getChild(op, 0));
	                        break;

	                    case FormExprType::MatrixLogarithm:
	                        values[op_idx] = emitMatrixLog(getChild(op, 0));
	                        break;

	                    case FormExprType::MatrixSqrt:
	                        values[op_idx] = emitMatrixSqrt(getChild(op, 0));
	                        break;

		                    case FormExprType::MatrixPower:
		                        values[op_idx] =
		                            emitMatrixPow(getChild(op, 0), getChild(op, 1).elems[0]);
		                        break;

		                    case FormExprType::MatrixExponentialDirectionalDerivative:
		                        values[op_idx] = callMatrixUnaryDD(getChild(op, 0),
		                                                          getChild(op, 1),
		                                                          mat_exp_dd_2x2_fn,
		                                                          mat_exp_dd_3x3_fn);
		                        break;

		                    case FormExprType::MatrixLogarithmDirectionalDerivative:
		                        values[op_idx] = callMatrixUnaryDD(getChild(op, 0),
		                                                          getChild(op, 1),
		                                                          mat_log_dd_2x2_fn,
		                                                          mat_log_dd_3x3_fn);
		                        break;

		                    case FormExprType::MatrixSqrtDirectionalDerivative:
		                        values[op_idx] = callMatrixUnaryDD(getChild(op, 0),
		                                                          getChild(op, 1),
		                                                          mat_sqrt_dd_2x2_fn,
		                                                          mat_sqrt_dd_3x3_fn);
		                        break;

		                    case FormExprType::MatrixPowerDirectionalDerivative:
		                        values[op_idx] = callMatrixPowDD(getChild(op, 0),
		                                                        getChild(op, 1),
		                                                        getChild(op, 2).elems[0],
		                                                        mat_pow_dd_2x2_fn,
		                                                        mat_pow_dd_3x3_fn);
		                        break;

	                    case FormExprType::SmoothAbsoluteValue:
	                        values[op_idx] = smoothAbs(getChild(op, 0), getChild(op, 1));
	                        break;

                    case FormExprType::SmoothSign:
                        values[op_idx] = smoothSign(getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::SmoothHeaviside:
                        values[op_idx] = smoothHeaviside(getChild(op, 0), getChild(op, 1));
                        break;

                    case FormExprType::SmoothMin:
                        values[op_idx] = smoothMin(getChild(op, 0), getChild(op, 1), getChild(op, 2));
                        break;

                    case FormExprType::SmoothMax:
                        values[op_idx] = smoothMax(getChild(op, 0), getChild(op, 1), getChild(op, 2));
                        break;

                    case FormExprType::SymmetricEigenvalue: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values[op_idx] = eigSym(getChild(op, 0), which);
                        break;
                    }

                    case FormExprType::Eigenvalue: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values[op_idx] = eigSym(getChild(op, 0), which);
                        break;
                    }

                    case FormExprType::SymmetricEigenvector: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values[op_idx] = eigSymVec(getChild(op, 0), which);
                        break;
                    }

                    case FormExprType::SpectralDecomposition:
                        values[op_idx] = spectralDecomp(getChild(op, 0));
                        break;

                    case FormExprType::HistoryWeightedSum:
                    case FormExprType::HistoryConvolution: {
                        CodeValue acc = makeZero(shape);
                        if (op.child_count != 0u) {
                            for (std::size_t kk = 0; kk < static_cast<std::size_t>(op.child_count); ++kk) {
                                const int k = static_cast<int>(kk + 1u);
                                acc = add(acc, mul(getChild(op, kk), evalHistorySolution(side, shape, k, q_index)));
                            }
                        } else {
                            for (int k = 1; k <= static_cast<int>(assembly::jit::kMaxPreviousSolutionsV6); ++k) {
                                acc = add(acc,
                                          mul(makeScalar(loadHistoryWeightOrZero(side, k)),
                                              evalHistorySolution(side, shape, k, q_index)));
                            }
                        }
                        values[op_idx] = acc;
                        break;
                    }

                    case FormExprType::SymmetricEigenvalueDirectionalDerivative: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values[op_idx] = eigSymDD(getChild(op, 0), getChild(op, 1), which);
                        break;
                    }

                    case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values[op_idx] = eigSymDDWrtA(getChild(op, 0), getChild(op, 1), getChild(op, 2), which);
                        break;
                    }

                    default:
                        throw std::runtime_error("LLVMGen: unsupported op in cached eval");
                }
                cached[op_idx] = values[op_idx];
            }

            return cached;
        };

        struct FaceCached {
            std::vector<CodeValue> minus;
            std::vector<CodeValue> plus;
        };

	        auto computeCachedFace = [&](const LoweredTerm& term,
	                                     llvm::Value* q_index) -> FaceCached {
	            FaceCached cached;
	            cached.minus.resize(term.ir.ops.size());
	            cached.plus.resize(term.ir.ops.size());

	            std::vector<CodeValue> values_minus;
	            std::vector<CodeValue> values_plus;
	            values_minus.resize(term.ir.ops.size());
	            values_plus.resize(term.ir.ops.size());
	            const auto use_counts = kernelIRUseCounts(term.ir);

            auto childIndex = [&](const KernelIROp& op, std::size_t k) -> std::size_t {
                return static_cast<std::size_t>(
                    term.ir.children[static_cast<std::size_t>(op.first_child) + k]);
            };
            auto childMinus = [&](const KernelIROp& op, std::size_t k) -> const CodeValue& {
                return values_minus[childIndex(op, k)];
            };
            auto childPlus = [&](const KernelIROp& op, std::size_t k) -> const CodeValue& {
                return values_plus[childIndex(op, k)];
            };

            for (std::size_t op_idx = 0; op_idx < term.ir.ops.size(); ++op_idx) {
                if (term.dep_mask[op_idx] != 0u) {
                    cached.minus[op_idx] = makeZero(term.shapes[op_idx]);
                    cached.plus[op_idx] = makeZero(term.shapes[op_idx]);
                    continue;
                }

                const auto& op = term.ir.ops[op_idx];
                const auto& shape = term.shapes[op_idx];

                switch (op.type) {
                    case FormExprType::Constant: {
                        const double v = std::bit_cast<double>(op.imm0);
                        const auto cv = makeScalar(f64c(v));
                        values_minus[op_idx] = cv;
                        values_plus[op_idx] = cv;
                        break;
                    }

                    case FormExprType::ParameterRef: {
                        auto* idx64 = builder.getInt64(op.imm0);
                        values_minus[op_idx] = makeScalar(loadRealPtrAt(side_minus.jit_constants, idx64));
                        values_plus[op_idx] = makeScalar(loadRealPtrAt(side_plus.jit_constants, idx64));
                        break;
                    }

                    case FormExprType::BoundaryIntegralRef: {
                        auto* idx64 = builder.getInt64(op.imm0);
                        values_minus[op_idx] = makeScalar(loadRealPtrAt(side_minus.coupled_integrals, idx64));
                        values_plus[op_idx] = makeScalar(loadRealPtrAt(side_plus.coupled_integrals, idx64));
                        break;
                    }

                    case FormExprType::AuxiliaryStateRef: {
                        auto* idx64 = builder.getInt64(op.imm0);
                        values_minus[op_idx] = makeScalar(loadRealPtrAt(side_minus.coupled_aux, idx64));
                        values_plus[op_idx] = makeScalar(loadRealPtrAt(side_plus.coupled_aux, idx64));
                        break;
                    }

                    case FormExprType::DiscreteField:
                    case FormExprType::StateField: {
                        const int fid = unpackFieldIdImm1(op.imm1);
                        if (op.type == FormExprType::StateField && fid == 0xffff) {
                            values_minus[op_idx] = evalCurrentSolution(side_minus, shape, q_index);
                            values_plus[op_idx] = evalCurrentSolution(side_plus, shape, q_index);
                        } else {
                            values_minus[op_idx] = evalDiscreteOrStateField(/*plus_side=*/false, shape, fid, q_index);
                            values_plus[op_idx] = evalDiscreteOrStateField(/*plus_side=*/true, shape, fid, q_index);
                        }
                        break;
                    }

                    case FormExprType::Coefficient: {
                        values_minus[op_idx] = evalExternalCoefficient(side_minus, q_index, shape, op.imm0);
                        values_plus[op_idx] = evalExternalCoefficient(side_plus, q_index, shape, op.imm0);
                        break;
                    }

                    case FormExprType::Constitutive: {
                        values_minus[op_idx] = evalExternalConstitutiveOutput(
                            side_minus,
                            q_index,
                            builder.getInt32(static_cast<std::uint32_t>(external::TraceSideV1::Minus)),
                            op.imm0,
                            /*output_index=*/0u,
                            term.ir,
                            op,
                            values_minus,
                            shape);
                        values_plus[op_idx] = evalExternalConstitutiveOutput(
                            side_plus,
                            q_index,
                            builder.getInt32(static_cast<std::uint32_t>(external::TraceSideV1::Plus)),
                            op.imm0,
                            /*output_index=*/0u,
                            term.ir,
                            op,
                            values_plus,
                            shape);
                        break;
                    }

                    case FormExprType::ConstitutiveOutput: {
                        if (op.child_count != 1u) {
                            throw std::runtime_error("LLVMGen: cached face ConstitutiveOutput expects exactly 1 child");
                        }

                        const auto out_idx_i64 = static_cast<std::int64_t>(op.imm0);
                        if (out_idx_i64 < 0) {
                            throw std::runtime_error("LLVMGen: cached face ConstitutiveOutput has negative output index");
                        }

                        const auto child_call_idx =
                            term.ir.children[static_cast<std::size_t>(op.first_child)];

                        if (out_idx_i64 == 0) {
                            values_minus[op_idx] = values_minus[child_call_idx];
                            values_plus[op_idx] = values_plus[child_call_idx];
                            break;
                        }

                        const auto& call_op = term.ir.ops[child_call_idx];
                        if (call_op.type != FormExprType::Constitutive) {
                            throw std::runtime_error("LLVMGen: cached face ConstitutiveOutput child must be Constitutive");
                        }

                        values_minus[op_idx] = evalExternalConstitutiveOutput(
                            side_minus,
                            q_index,
                            builder.getInt32(static_cast<std::uint32_t>(external::TraceSideV1::Minus)),
                            call_op.imm0,
                            static_cast<std::uint32_t>(out_idx_i64),
                            term.ir,
                            call_op,
                            values_minus,
                            shape);
                        values_plus[op_idx] = evalExternalConstitutiveOutput(
                            side_plus,
                            q_index,
                            builder.getInt32(static_cast<std::uint32_t>(external::TraceSideV1::Plus)),
                            call_op.imm0,
                            static_cast<std::uint32_t>(out_idx_i64),
                            term.ir,
                            call_op,
                            values_plus,
                            shape);
                        break;
                    }

                    case FormExprType::Time:
                        values_minus[op_idx] = makeScalar(side_minus.time);
                        values_plus[op_idx] = makeScalar(side_plus.time);
                        break;

                    case FormExprType::TimeStep:
                        values_minus[op_idx] = makeScalar(side_minus.dt);
                        values_plus[op_idx] = makeScalar(side_plus.dt);
                        break;

                    case FormExprType::EffectiveTimeStep:
                        values_minus[op_idx] = makeScalar(loadEffectiveDt(side_minus));
                        values_plus[op_idx] = makeScalar(loadEffectiveDt(side_plus));
                        break;

                    case FormExprType::CellDiameter:
                        values_minus[op_idx] = makeScalar(side_minus.cell_diameter);
                        values_plus[op_idx] = makeScalar(side_plus.cell_diameter);
                        break;

                    case FormExprType::CellVolume:
                        values_minus[op_idx] = makeScalar(side_minus.cell_volume);
                        values_plus[op_idx] = makeScalar(side_plus.cell_volume);
                        break;

                    case FormExprType::FacetArea:
                        values_minus[op_idx] = makeScalar(side_minus.facet_area);
                        values_plus[op_idx] = makeScalar(side_plus.facet_area);
                        break;

                    case FormExprType::CellDomainId:
                        values_minus[op_idx] = makeScalar(builder.CreateSIToFP(side_minus.cell_domain_id, f64));
                        values_plus[op_idx] = makeScalar(builder.CreateSIToFP(side_plus.cell_domain_id, f64));
                        break;

	                    case FormExprType::Coordinate:
	                        values_minus[op_idx] =
	                            loadXYZDimFromSide(side_minus, side_minus.physical_points_xyz, q_index, shape.dims[0],
	                                               side_minus.interleaved_qpoint_geometry_physical_offset);
	                        values_plus[op_idx] =
	                            loadXYZDimFromSide(side_plus, side_plus.physical_points_xyz, q_index, shape.dims[0],
	                                               side_plus.interleaved_qpoint_geometry_physical_offset);
	                        break;

	                    case FormExprType::ReferenceCoordinate:
	                        values_minus[op_idx] = loadXYZDim(side_minus.quad_points_xyz, q_index, shape.dims[0]);
	                        values_plus[op_idx] = loadXYZDim(side_plus.quad_points_xyz, q_index, shape.dims[0]);
	                        break;

	                    case FormExprType::Normal:
	                        values_minus[op_idx] =
	                            loadXYZDimFromSide(side_minus, side_minus.normals_xyz, q_index, shape.dims[0],
	                                               side_minus.interleaved_qpoint_geometry_normal_offset);
	                        values_plus[op_idx] =
	                            loadXYZDimFromSide(side_plus, side_plus.normals_xyz, q_index, shape.dims[0],
	                                               side_plus.interleaved_qpoint_geometry_normal_offset);
	                        break;

	                    case FormExprType::Jacobian:
	                        values_minus[op_idx] =
	                            loadMatDimFromSide(side_minus, side_minus.jacobians, q_index, shape.dims[0],
	                                               side_minus.interleaved_qpoint_geometry_jacobian_offset);
	                        values_plus[op_idx] =
	                            loadMatDimFromSide(side_plus, side_plus.jacobians, q_index, shape.dims[0],
	                                               side_plus.interleaved_qpoint_geometry_jacobian_offset);
	                        break;

	                    case FormExprType::JacobianInverse:
	                        values_minus[op_idx] =
	                            loadMatDimFromSide(side_minus, side_minus.inverse_jacobians, q_index, shape.dims[0],
	                                               side_minus.interleaved_qpoint_geometry_inverse_jacobian_offset);
	                        values_plus[op_idx] =
	                            loadMatDimFromSide(side_plus, side_plus.inverse_jacobians, q_index, shape.dims[0],
	                                               side_plus.interleaved_qpoint_geometry_inverse_jacobian_offset);
	                        break;

                    case FormExprType::Identity: {
                        CodeValue out = makeZero(shape);
                        const auto n = static_cast<std::size_t>(shape.dims[0]);
                        for (std::size_t d = 0; d < n; ++d) {
                            out.elems[d * n + d] = f64c(1.0);
                        }
                        values_minus[op_idx] = out;
                        values_plus[op_idx] = out;
                        break;
                    }

                    case FormExprType::JacobianDeterminant: {
                        values_minus[op_idx] =
                            makeScalar(loadScalarFromSide(side_minus, side_minus.jacobian_dets, q_index,
                                                          side_minus.interleaved_qpoint_geometry_det_offset));
                        values_plus[op_idx] =
                            makeScalar(loadScalarFromSide(side_plus, side_plus.jacobian_dets, q_index,
                                                          side_plus.interleaved_qpoint_geometry_det_offset));
                        break;
                    }

                    case FormExprType::PreviousSolutionRef: {
                        const int k = static_cast<int>(static_cast<std::int64_t>(op.imm0));
                        values_minus[op_idx] = evalPreviousSolution(side_minus, shape, k, q_index);
                        values_plus[op_idx] = evalPreviousSolution(side_plus, shape, k, q_index);
                        break;
                    }

                    case FormExprType::Gradient: {
                        const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                        const auto& kid = term.ir.ops[child_idx];

                        auto evalSide = [&](const SideView& side, bool plus_side) -> CodeValue {
		                            if (kid.type == FormExprType::PreviousSolutionRef) {
		                                const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
		                                auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
		                                if (shape.kind == Shape::Kind::Vector) {
		                                    const auto dim = static_cast<std::size_t>(shape.dims[0]);
		                                    const auto sums =
		                                        emitReduceSum(side.n_trial_dofs, "prev_grad" + std::to_string(k), dim, [&](llvm::Value* j) {
		                                            auto* j64 = builder.CreateZExt(j, i64);
		                                            auto* cj = loadRealPtrAt(coeffs, j64);
		                                            const auto g =
		                                                loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
		                                            std::vector<llvm::Value*> terms;
		                                            terms.reserve(dim);
		                                            for (std::size_t d = 0; d < dim; ++d) {
		                                                terms.push_back(builder.CreateFMul(cj, g[d]));
		                                            }
		                                            return terms;
		                                        });
		                                    auto* x = sums[0];
		                                    auto* y = (dim > 1u) ? sums[1] : f64c(0.0);
		                                    auto* z = (dim > 2u) ? sums[2] : f64c(0.0);
		                                    return makeVector(shape.dims[0], x, y, z);
		                                }
		                                if (shape.kind == Shape::Kind::Matrix) {
		                                    const auto vd = static_cast<std::size_t>(shape.dims[0]);
		                                    const auto dim = static_cast<std::size_t>(shape.dims[1]);
		                                    CodeValue out = makeZero(shape);
		                                    auto* dofs_per_comp =
		                                        builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
		                                    for (std::size_t comp = 0; comp < vd; ++comp) {
	                                        const auto acc = emitReduceSum(dofs_per_comp,
	                                                                       "prev_grad" + std::to_string(k) + "_c" +
	                                                                           std::to_string(comp),
	                                                                       dim,
	                                                                       [&](llvm::Value* jj) {
	                                            auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                            auto* j = builder.CreateAdd(base, jj);
		                                            auto* j64 = builder.CreateZExt(j, i64);
		                                            auto* cj = loadRealPtrAt(coeffs, j64);
		                                            const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                            std::vector<llvm::Value*> terms;
	                                            terms.reserve(dim);
		                                            for (std::size_t d = 0; d < dim; ++d) {
		                                                terms.push_back(builder.CreateFMul(cj, g[d]));
		                                            }
	                                            return terms;
		                                        });
		                                        for (std::size_t d = 0; d < dim; ++d) {
		                                            out.elems[comp * dim + d] = acc[d];
		                                        }
		                                    }
	                                    return out;
	                                }
	                                throw std::runtime_error("LLVMGen: cached grad(PreviousSolutionRef) unsupported shape");
	                            }

	                            if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
	                                const int fid = unpackFieldIdImm1(kid.imm1);
	                                if (shape.kind == Shape::Kind::Vector) {
	                                    const auto dim = static_cast<std::size_t>(shape.dims[0]);
	                                    auto* entry = fieldEntryPtrFor(plus_side, fid);
	                                    auto* entry_is_null =
	                                        builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                    auto* ok = llvm::BasicBlock::Create(*ctx, "field_grad.ok", fn);
                                    auto* zero = llvm::BasicBlock::Create(*ctx, "field_grad.zero", fn);
                                    auto* merge = llvm::BasicBlock::Create(*ctx, "field_grad.merge", fn);

                                    builder.CreateCondBr(entry_is_null, zero, ok);

                                    std::array<llvm::Value*, 3> loaded{f64c(0.0), f64c(0.0), f64c(0.0)};
                                    builder.SetInsertPoint(ok);
                                    auto* base = loadPtr(entry, ABIV3::field_entry_gradients_xyz_off);
                                    auto* base_is_null =
                                        builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                    auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_grad.xyz.ok", fn);
                                    builder.CreateCondBr(base_is_null, zero, ok2);

                                    builder.SetInsertPoint(ok2);
                                    const auto v = loadXYZ(base, q_index);
                                    loaded = {v.elems[0], v.elems[1], v.elems[2]};
                                    builder.CreateBr(merge);
                                    auto* ok2_block = builder.GetInsertBlock();

                                    builder.SetInsertPoint(zero);
                                    builder.CreateBr(merge);
                                    auto* zero_block = builder.GetInsertBlock();

	                                    builder.SetInsertPoint(merge);
	                                    std::array<llvm::Value*, 3> outv{f64c(0.0), f64c(0.0), f64c(0.0)};
	                                    for (std::size_t d = 0; d < dim; ++d) {
	                                        auto* phi = builder.CreatePHI(f64, 2, "field_grad.v" + std::to_string(d));
	                                        phi->addIncoming(f64c(0.0), zero_block);
	                                        phi->addIncoming(loaded[d], ok2_block);
	                                        outv[d] = phi;
	                                    }
	                                    return makeVector(shape.dims[0], outv[0], outv[1], outv[2]);
	                                }
	                                if (shape.kind == Shape::Kind::Matrix) {
	                                    const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                                    const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                                    auto* entry = fieldEntryPtrFor(plus_side, fid);
	                                    auto* entry_is_null =
	                                        builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                    auto* ok = llvm::BasicBlock::Create(*ctx, "field_jac.ok", fn);
                                    auto* zero = llvm::BasicBlock::Create(*ctx, "field_jac.zero", fn);
                                    auto* merge = llvm::BasicBlock::Create(*ctx, "field_jac.merge", fn);

                                    builder.CreateCondBr(entry_is_null, zero, ok);

                                    CodeValue loaded = makeZero(matrixShape(3u, 3u));
                                    builder.SetInsertPoint(ok);
                                    auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                                    auto* base_is_null =
                                        builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                    auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_jac.mat.ok", fn);
                                    builder.CreateCondBr(base_is_null, zero, ok2);

                                    builder.SetInsertPoint(ok2);
                                    loaded = loadMat3FromQ(base, q_index);
                                    builder.CreateBr(merge);
                                    auto* ok2_block = builder.GetInsertBlock();

                                    builder.SetInsertPoint(zero);
                                    builder.CreateBr(merge);
                                    auto* zero_block = builder.GetInsertBlock();

	                                    builder.SetInsertPoint(merge);
	                                    CodeValue out = makeZero(shape);
	                                    for (std::size_t r = 0; r < vd; ++r) {
	                                        for (std::size_t c = 0; c < dim; ++c) {
	                                            auto* phi = builder.CreatePHI(f64, 2, "field_jac.a");
	                                            phi->addIncoming(f64c(0.0), zero_block);
	                                            auto* l =
	                                                (r < 3u && c < 3u) ? loaded.elems[r * 3u + c] : f64c(0.0);
	                                            phi->addIncoming(l, ok2_block);
	                                            out.elems[r * dim + c] = phi;
	                                        }
	                                    }
	                                    return out;
	                                }
                                throw std::runtime_error("LLVMGen: cached grad(DiscreteField) unsupported shape");
                            }

                            if (kid.type == FormExprType::Constant) {
                                return makeZero(shape);
                            }

                            throw std::runtime_error("LLVMGen: cached Gradient operand not supported");
                        };

                        values_minus[op_idx] = evalSide(side_minus, /*plus_side=*/false);
                        values_plus[op_idx] = evalSide(side_plus, /*plus_side=*/true);
                        break;
                    }

                    case FormExprType::Divergence: {
                        const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                        const auto& kid = term.ir.ops[child_idx];
                        const auto vd = static_cast<std::size_t>(term.shapes[child_idx].dims[0]);

                        auto evalSide = [&](const SideView& side, bool plus_side) -> CodeValue {
                            if (kid.type == FormExprType::PreviousSolutionRef) {
                                const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
	                                auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
		                                auto* dofs_per_comp =
		                                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
		                                llvm::Value* div = f64c(0.0);
		                                for (std::size_t comp = 0; comp < vd; ++comp) {
		                                    auto* acc = emitReduceSumScalar(
		                                        dofs_per_comp,
		                                        "prev_div" + std::to_string(k) + "_c" + std::to_string(comp),
		                                        [&](llvm::Value* jj) -> llvm::Value* {
		                                            auto* base =
		                                                builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
		                                            auto* j = builder.CreateAdd(base, jj);
		                                            auto* j64 = builder.CreateZExt(j, i64);
		                                            auto* cj = loadRealPtrAt(coeffs, j64);
		                                            const auto g =
		                                                loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
		                                            return builder.CreateFMul(cj, g[comp]);
		                                        });
		                                    div = builder.CreateFAdd(div, acc);
		                                }
		                                return makeScalar(div);
		                            }

                            if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                                const int fid = unpackFieldIdImm1(kid.imm1);
                                auto* entry = fieldEntryPtrFor(plus_side, fid);
                                auto* entry_is_null =
                                    builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok = llvm::BasicBlock::Create(*ctx, "field_div.ok", fn);
                                auto* zero = llvm::BasicBlock::Create(*ctx, "field_div.zero", fn);
                                auto* merge = llvm::BasicBlock::Create(*ctx, "field_div.merge", fn);

                                builder.CreateCondBr(entry_is_null, zero, ok);

                                llvm::Value* div = f64c(0.0);
                                builder.SetInsertPoint(ok);
                                auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                                auto* base_is_null =
                                    builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_div.mat.ok", fn);
                                builder.CreateCondBr(base_is_null, zero, ok2);

                                builder.SetInsertPoint(ok2);
                                const auto J = loadMat3FromQ(base, q_index);
                                div = J.elems[0];
                                if (vd >= 2) div = builder.CreateFAdd(div, J.elems[4]);
                                if (vd >= 3) div = builder.CreateFAdd(div, J.elems[8]);
                                builder.CreateBr(merge);
                                auto* ok2_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(zero);
                                builder.CreateBr(merge);
                                auto* zero_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(merge);
                                auto* phi = builder.CreatePHI(f64, 2, "field_div");
                                phi->addIncoming(f64c(0.0), zero_block);
                                phi->addIncoming(div, ok2_block);
                                return makeScalar(phi);
                            }

                            if (kid.type == FormExprType::Constant) {
                                return makeScalar(f64c(0.0));
                            }

                            throw std::runtime_error("LLVMGen: cached Divergence operand not supported");
                        };

                        values_minus[op_idx] = evalSide(side_minus, /*plus_side=*/false);
                        values_plus[op_idx] = evalSide(side_plus, /*plus_side=*/true);
                        break;
                    }

                    case FormExprType::Curl: {
                        const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                        const auto& kid = term.ir.ops[child_idx];
                        const auto vd = static_cast<std::size_t>(term.shapes[child_idx].dims[0]);

                        auto evalSide = [&](const SideView& side, bool plus_side) -> CodeValue {
                            if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                                const int fid = unpackFieldIdImm1(kid.imm1);
                                auto* entry = fieldEntryPtrFor(plus_side, fid);
                                auto* entry_is_null =
                                    builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok = llvm::BasicBlock::Create(*ctx, "field_curl.ok", fn);
                                auto* zero = llvm::BasicBlock::Create(*ctx, "field_curl.zero", fn);
                                auto* merge = llvm::BasicBlock::Create(*ctx, "field_curl.merge", fn);

                                builder.CreateCondBr(entry_is_null, zero, ok);

                                std::array<llvm::Value*, 3> c{f64c(0.0), f64c(0.0), f64c(0.0)};
                                builder.SetInsertPoint(ok);
                                auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                                auto* base_is_null =
                                    builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_curl.mat.ok", fn);
                                builder.CreateCondBr(base_is_null, zero, ok2);

                                builder.SetInsertPoint(ok2);
                                const auto J = loadMat3FromQ(base, q_index);
                                auto d = [&](std::size_t comp, std::size_t wrt) -> llvm::Value* {
                                    if (comp >= vd || comp >= 3u || wrt >= 3u) return f64c(0.0);
                                    return J.elems[comp * 3 + wrt];
                                };
                                c[0] = builder.CreateFSub(d(2, 1), d(1, 2));
                                c[1] = builder.CreateFSub(d(0, 2), d(2, 0));
                                c[2] = builder.CreateFSub(d(1, 0), d(0, 1));
                                builder.CreateBr(merge);
                                auto* ok2_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(zero);
                                builder.CreateBr(merge);
                                auto* zero_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(merge);
                                std::array<llvm::Value*, 3> out{f64c(0.0), f64c(0.0), f64c(0.0)};
                                for (std::size_t d0 = 0; d0 < 3; ++d0) {
                                    auto* phi = builder.CreatePHI(f64, 2, "field_curl." + std::to_string(d0));
                                    phi->addIncoming(f64c(0.0), zero_block);
                                    phi->addIncoming(c[d0], ok2_block);
                                    out[d0] = phi;
                                }
                                return makeVector(3u, out[0], out[1], out[2]);
                            }

                            if (kid.type == FormExprType::PreviousSolutionRef) {
                                const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
                                auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
                                auto* dofs_per_comp =
                                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));

                                std::array<std::array<llvm::Value*, 3>, 3> J{};
                                for (auto& row : J) {
                                    row = {f64c(0.0), f64c(0.0), f64c(0.0)};
                                }

	                                for (std::size_t comp = 0; comp < std::min<std::size_t>(vd, 3u); ++comp) {
	                                    const auto acc =
	                                        emitReduceSum(dofs_per_comp,
	                                                     "prev_curl" + std::to_string(k) + "_c" + std::to_string(comp),
	                                                     3u,
	                                                     [&](llvm::Value* jj) {
	                                                         auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                                         auto* j = builder.CreateAdd(base, jj);
	                                                         auto* j64 = builder.CreateZExt(j, i64);
	                                                         auto* cj = loadRealPtrAt(coeffs, j64);
	                                                         const auto g =
	                                                             loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                                         std::vector<llvm::Value*> terms;
	                                                         terms.reserve(3u);
	                                                         for (std::size_t d = 0; d < 3u; ++d) {
	                                                             terms.push_back(builder.CreateFMul(cj, g[d]));
	                                                         }
	                                                         return terms;
	                                                     });
	                                    J[comp] = {acc[0], acc[1], acc[2]};
	                                }

                                auto* cx = builder.CreateFSub(J[2][1], J[1][2]);
                                auto* cy = builder.CreateFSub(J[0][2], J[2][0]);
                                auto* cz = builder.CreateFSub(J[1][0], J[0][1]);
                                return makeVector(3u, cx, cy, cz);
                            }

                            if (kid.type == FormExprType::Constant) {
                                return makeZero(vectorShape(3u));
                            }

                            throw std::runtime_error("LLVMGen: cached Curl operand not supported");
                        };

                        values_minus[op_idx] = evalSide(side_minus, /*plus_side=*/false);
                        values_plus[op_idx] = evalSide(side_plus, /*plus_side=*/true);
                        break;
                    }

	                    case FormExprType::Hessian: {
	                        const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
	                        const auto& kid = term.ir.ops[child_idx];
	                        const auto dim_u32 = shape.dims[0];
	                        const auto mat_len = elemCount(shape);

		                        auto evalSide = [&](const SideView& side, bool plus_side) -> CodeValue {
		                            if (kid.type == FormExprType::PreviousSolutionRef) {
		                                const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
		                                auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
		                                CodeValue out = makeZero(shape);
		                                const auto sums =
		                                    emitReduceSum(side.n_trial_dofs, "prev_hess" + std::to_string(k), mat_len, [&](llvm::Value* j) {
		                                        auto* j64 = builder.CreateZExt(j, i64);
		                                        auto* cj = loadRealPtrAt(coeffs, j64);
		                                        const auto H = loadMatDimFromTable(
		                                            side.trial_phys_hessians, side.n_qpts, j, q_index, dim_u32);
		                                        std::vector<llvm::Value*> terms;
		                                        terms.reserve(mat_len);
		                                        for (std::size_t i = 0; i < mat_len; ++i) {
		                                            terms.push_back(builder.CreateFMul(cj, H.elems[i]));
		                                        }
		                                        return terms;
		                                    });
		                                for (std::size_t i = 0; i < mat_len; ++i) {
		                                    out.elems[i] = sums[i];
		                                }
		                                return out;
		                            }

                            if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                                const int fid = unpackFieldIdImm1(kid.imm1);
                                auto* entry = fieldEntryPtrFor(plus_side, fid);
                                auto* entry_is_null =
                                    builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok = llvm::BasicBlock::Create(*ctx, "field_hess.ok", fn);
                                auto* z = llvm::BasicBlock::Create(*ctx, "field_hess.zero", fn);
	                                auto* merge = llvm::BasicBlock::Create(*ctx, "field_hess.merge", fn);

	                                builder.CreateCondBr(entry_is_null, z, ok);

	                                CodeValue loaded = makeZero(shape);
	                                builder.SetInsertPoint(ok);
	                                auto* base = loadPtr(entry, ABIV3::field_entry_hessians_off);
	                                auto* base_is_null =
	                                    builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_hess.mat.ok", fn);
                                builder.CreateCondBr(base_is_null, z, ok2);

	                                builder.SetInsertPoint(ok2);
	                                loaded = loadMatDimFromQ(base, q_index, dim_u32);
	                                builder.CreateBr(merge);
	                                auto* ok2_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(z);
                                builder.CreateBr(merge);
	                                auto* zero_block = builder.GetInsertBlock();

	                                builder.SetInsertPoint(merge);
	                                CodeValue out = makeZero(shape);
	                                for (std::size_t i = 0; i < mat_len; ++i) {
	                                    auto* phi = builder.CreatePHI(f64, 2, "field_hess." + std::to_string(i));
	                                    phi->addIncoming(f64c(0.0), zero_block);
	                                    phi->addIncoming(loaded.elems[i], ok2_block);
	                                    out.elems[i] = phi;
	                                }
	                                return out;
	                            }

	                            if (kid.type == FormExprType::Constant) {
	                                return makeZero(shape);
	                            }

                            throw std::runtime_error("LLVMGen: cached Hessian operand not supported");
                        };

                        values_minus[op_idx] = evalSide(side_minus, /*plus_side=*/false);
                        values_plus[op_idx] = evalSide(side_plus, /*plus_side=*/true);
                        break;
                    }

                    case FormExprType::TimeDerivative: {
                        const int order = static_cast<int>(static_cast<std::int64_t>(op.imm0));
                        CodeValue acc_minus = mul(makeScalar(loadDtCoeff(side_minus, order, 0)), childMinus(op, 0));
                        CodeValue acc_plus = mul(makeScalar(loadDtCoeff(side_plus, order, 0)), childPlus(op, 0));

                        const auto child_op_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                        const auto& kid = term.ir.ops[child_op_idx];
                        if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                            const int fid = unpackFieldIdImm1(kid.imm1);
                            if (kid.type == FormExprType::StateField && fid == 0xffff) {
                                for (int k = 1; k <= static_cast<int>(assembly::jit::kMaxPreviousSolutionsV6); ++k) {
                                    acc_minus = add(acc_minus,
                                                    mul(makeScalar(loadDtCoeff(side_minus, order, k)),
                                                        evalPreviousSolution(side_minus, shape, k, q_index)));
                                    acc_plus = add(acc_plus,
                                                   mul(makeScalar(loadDtCoeff(side_plus, order, k)),
                                                       evalPreviousSolution(side_plus, shape, k, q_index)));
                                }
                            } else {
                                for (int k = 1; k <= static_cast<int>(assembly::jit::kMaxPreviousSolutionsV6); ++k) {
                                    acc_minus = add(acc_minus,
                                                    mul(makeScalar(loadDtCoeff(side_minus, order, k)),
                                                        evalDiscreteOrStateFieldHistoryK(side_minus, /*plus_side=*/false, shape, fid, k, q_index)));
                                    acc_plus = add(acc_plus,
                                                   mul(makeScalar(loadDtCoeff(side_plus, order, k)),
                                                       evalDiscreteOrStateFieldHistoryK(side_plus, /*plus_side=*/true, shape, fid, k, q_index)));
                                }
                            }
                        }

                        values_minus[op_idx] = acc_minus;
                        values_plus[op_idx] = acc_plus;
                        break;
                    }

                    case FormExprType::RestrictMinus: {
                        const auto child = childIndex(op, 0);
                        values_minus[op_idx] = values_minus[child];
                        values_plus[op_idx] = values_minus[child];
                        break;
                    }

                    case FormExprType::RestrictPlus: {
                        const auto child = childIndex(op, 0);
                        values_minus[op_idx] = values_plus[child];
                        values_plus[op_idx] = values_plus[child];
                        break;
                    }

                    case FormExprType::Jump: {
                        const auto child = childIndex(op, 0);
                        const auto out = sub(values_minus[child], values_plus[child]);
                        values_minus[op_idx] = out;
                        values_plus[op_idx] = out;
                        break;
                    }

                    case FormExprType::Average: {
                        const auto child = childIndex(op, 0);
                        const auto out = mul(makeScalar(f64c(0.5)), add(values_minus[child], values_plus[child]));
                        values_minus[op_idx] = out;
                        values_plus[op_idx] = out;
                        break;
                    }

                    case FormExprType::MaterialStateOldRef:
                        values_minus[op_idx] = makeScalar(loadMaterialStateReal(side_minus.material_state_old_base,
                                                                               side_minus.material_state_stride_bytes,
                                                                               q_index,
                                                                               static_cast<std::uint64_t>(op.imm0)));
                        values_plus[op_idx] = makeScalar(loadMaterialStateReal(side_plus.material_state_old_base,
                                                                              side_plus.material_state_stride_bytes,
                                                                              q_index,
                                                                              static_cast<std::uint64_t>(op.imm0)));
                        break;

                    case FormExprType::MaterialStateWorkRef:
                        values_minus[op_idx] = makeScalar(loadMaterialStateReal(side_minus.material_state_work_base,
                                                                               side_minus.material_state_stride_bytes,
                                                                               q_index,
                                                                               static_cast<std::uint64_t>(op.imm0)));
                        values_plus[op_idx] = makeScalar(loadMaterialStateReal(side_plus.material_state_work_base,
                                                                              side_plus.material_state_stride_bytes,
                                                                              q_index,
                                                                              static_cast<std::uint64_t>(op.imm0)));
                        break;

                    case FormExprType::Negate:
                        values_minus[op_idx] = neg(childMinus(op, 0));
                        values_plus[op_idx] = neg(childPlus(op, 0));
                        break;

	                    case FormExprType::Add: {
	                        if (shape.kind == Shape::Kind::Scalar) {
	                            if (auto* fused = tryFuseMulAddScalar(op.type, term, use_counts, op, values_minus)) {
	                                values_minus[op_idx] = makeScalar(fused);
	                            } else {
	                                values_minus[op_idx] = add(childMinus(op, 0), childMinus(op, 1));
	                            }
	                            if (auto* fused = tryFuseMulAddScalar(op.type, term, use_counts, op, values_plus)) {
	                                values_plus[op_idx] = makeScalar(fused);
	                            } else {
	                                values_plus[op_idx] = add(childPlus(op, 0), childPlus(op, 1));
	                            }
	                            break;
	                        }
	                        values_minus[op_idx] = add(childMinus(op, 0), childMinus(op, 1));
	                        values_plus[op_idx] = add(childPlus(op, 0), childPlus(op, 1));
	                        break;
	                    }

	                    case FormExprType::Subtract: {
	                        if (shape.kind == Shape::Kind::Scalar) {
	                            if (auto* fused = tryFuseMulAddScalar(op.type, term, use_counts, op, values_minus)) {
	                                values_minus[op_idx] = makeScalar(fused);
	                            } else {
	                                values_minus[op_idx] = sub(childMinus(op, 0), childMinus(op, 1));
	                            }
	                            if (auto* fused = tryFuseMulAddScalar(op.type, term, use_counts, op, values_plus)) {
	                                values_plus[op_idx] = makeScalar(fused);
	                            } else {
	                                values_plus[op_idx] = sub(childPlus(op, 0), childPlus(op, 1));
	                            }
	                            break;
	                        }
	                        values_minus[op_idx] = sub(childMinus(op, 0), childMinus(op, 1));
	                        values_plus[op_idx] = sub(childPlus(op, 0), childPlus(op, 1));
	                        break;
	                    }

                    case FormExprType::Multiply:
                        values_minus[op_idx] = mul(childMinus(op, 0), childMinus(op, 1));
                        values_plus[op_idx] = mul(childPlus(op, 0), childPlus(op, 1));
                        break;

                    case FormExprType::Divide:
                        values_minus[op_idx] = div(childMinus(op, 0), childMinus(op, 1));
                        values_plus[op_idx] = div(childPlus(op, 0), childPlus(op, 1));
                        break;

                    case FormExprType::InnerProduct:
                        values_minus[op_idx] = inner(childMinus(op, 0), childMinus(op, 1));
                        values_plus[op_idx] = inner(childPlus(op, 0), childPlus(op, 1));
                        break;

                    case FormExprType::DoubleContraction:
                        values_minus[op_idx] = doubleContraction(childMinus(op, 0), childMinus(op, 1));
                        values_plus[op_idx] = doubleContraction(childPlus(op, 0), childPlus(op, 1));
                        break;

                    case FormExprType::OuterProduct:
                        values_minus[op_idx] = outer(childMinus(op, 0), childMinus(op, 1));
                        values_plus[op_idx] = outer(childPlus(op, 0), childPlus(op, 1));
                        break;

                    case FormExprType::CrossProduct:
                        values_minus[op_idx] = cross(childMinus(op, 0), childMinus(op, 1));
                        values_plus[op_idx] = cross(childPlus(op, 0), childPlus(op, 1));
                        break;

                    case FormExprType::Power:
                        values_minus[op_idx] = powv(childMinus(op, 0), childMinus(op, 1));
                        values_plus[op_idx] = powv(childPlus(op, 0), childPlus(op, 1));
                        break;

                    case FormExprType::Minimum:
                        values_minus[op_idx] = makeScalar(f_min(childMinus(op, 0).elems[0], childMinus(op, 1).elems[0]));
                        values_plus[op_idx] = makeScalar(f_min(childPlus(op, 0).elems[0], childPlus(op, 1).elems[0]));
                        break;

                    case FormExprType::Maximum:
                        values_minus[op_idx] = makeScalar(f_max(childMinus(op, 0).elems[0], childMinus(op, 1).elems[0]));
                        values_plus[op_idx] = makeScalar(f_max(childPlus(op, 0).elems[0], childPlus(op, 1).elems[0]));
                        break;

                    case FormExprType::Less:
                    case FormExprType::LessEqual:
                    case FormExprType::Greater:
                    case FormExprType::GreaterEqual:
                    case FormExprType::Equal:
                    case FormExprType::NotEqual:
                        values_minus[op_idx] = cmp(op.type, childMinus(op, 0), childMinus(op, 1));
                        values_plus[op_idx] = cmp(op.type, childPlus(op, 0), childPlus(op, 1));
                        break;

                    case FormExprType::Conditional:
                        values_minus[op_idx] = conditional(childMinus(op, 0), childMinus(op, 1), childMinus(op, 2));
                        values_plus[op_idx] = conditional(childPlus(op, 0), childPlus(op, 1), childPlus(op, 2));
                        break;

                    case FormExprType::AsVector: {
                        CodeValue out_m = makeZero(shape);
                        CodeValue out_p = makeZero(shape);
                        for (std::size_t k = 0; k < op.child_count; ++k) {
                            out_m.elems[k] = childMinus(op, k).elems[0];
                            out_p.elems[k] = childPlus(op, k).elems[0];
                        }
                        values_minus[op_idx] = out_m;
                        values_plus[op_idx] = out_p;
                        break;
                    }

                    case FormExprType::AsTensor: {
                        const auto rows = unpackU32Lo(op.imm0);
                        const auto cols = unpackU32Hi(op.imm0);
                        CodeValue out_m = makeMatrix(rows, cols);
                        CodeValue out_p = makeMatrix(rows, cols);
                        for (std::size_t r = 0; r < rows; ++r) {
                            for (std::size_t c = 0; c < cols; ++c) {
                                const auto k = r * cols + c;
                                out_m.elems[k] = childMinus(op, k).elems[0];
                                out_p.elems[k] = childPlus(op, k).elems[0];
                            }
                        }
                        values_minus[op_idx] = out_m;
                        values_plus[op_idx] = out_p;
                        break;
                    }

                    case FormExprType::Component: {
                        const auto i = static_cast<std::int32_t>(unpackU32Lo(op.imm0));
                        const auto j = static_cast<std::int32_t>(unpackU32Hi(op.imm0));

                        auto comp = [&](const CodeValue& a) -> CodeValue {
                            if (a.shape.kind == Shape::Kind::Scalar) {
                                return a;
                            }
                            if (a.shape.kind == Shape::Kind::Vector) {
                                if (j >= 0) throw std::runtime_error("LLVMGen: component(v,i,j) invalid for vector");
                                return makeScalar(getVecComp(a, static_cast<std::size_t>(i)));
                            }
                            if (a.shape.kind == Shape::Kind::Matrix) {
                                if (j < 0) throw std::runtime_error("LLVMGen: component(A,i) missing column index");
                                const auto rows = static_cast<std::size_t>(a.shape.dims[0]);
                                const auto cols = static_cast<std::size_t>(a.shape.dims[1]);
                                const auto rr = static_cast<std::size_t>(i);
                                const auto cc = static_cast<std::size_t>(j);
                                if (rr >= rows || cc >= cols) throw std::runtime_error("LLVMGen: component(A,i,j) index out of range");
                                return makeScalar(a.elems[rr * cols + cc]);
                            }
                            throw std::runtime_error("LLVMGen: component() unsupported operand kind");
                        };

                        values_minus[op_idx] = comp(childMinus(op, 0));
                        values_plus[op_idx] = comp(childPlus(op, 0));
                        break;
                    }

                    case FormExprType::Transpose:
                        values_minus[op_idx] = transpose(childMinus(op, 0));
                        values_plus[op_idx] = transpose(childPlus(op, 0));
                        break;

                    case FormExprType::Trace:
                        values_minus[op_idx] = trace(childMinus(op, 0));
                        values_plus[op_idx] = trace(childPlus(op, 0));
                        break;

                    case FormExprType::Determinant:
                        values_minus[op_idx] = det(childMinus(op, 0));
                        values_plus[op_idx] = det(childPlus(op, 0));
                        break;

                    case FormExprType::Cofactor:
                        values_minus[op_idx] = cofactor(childMinus(op, 0));
                        values_plus[op_idx] = cofactor(childPlus(op, 0));
                        break;

                    case FormExprType::Inverse:
                        values_minus[op_idx] = inv(childMinus(op, 0));
                        values_plus[op_idx] = inv(childPlus(op, 0));
                        break;

                    case FormExprType::SymmetricPart:
                        values_minus[op_idx] = symOrSkew(true, childMinus(op, 0));
                        values_plus[op_idx] = symOrSkew(true, childPlus(op, 0));
                        break;

                    case FormExprType::SkewPart:
                        values_minus[op_idx] = symOrSkew(false, childMinus(op, 0));
                        values_plus[op_idx] = symOrSkew(false, childPlus(op, 0));
                        break;

	                    case FormExprType::Deviator:
	                        values_minus[op_idx] = deviator(childMinus(op, 0));
	                        values_plus[op_idx] = deviator(childPlus(op, 0));
	                        break;

                    case FormExprType::Norm:
                        values_minus[op_idx] = norm(childMinus(op, 0));
                        values_plus[op_idx] = norm(childPlus(op, 0));
                        break;

                    case FormExprType::Normalize:
                        values_minus[op_idx] = normalize(childMinus(op, 0));
                        values_plus[op_idx] = normalize(childPlus(op, 0));
                        break;

                    case FormExprType::AbsoluteValue:
                        values_minus[op_idx] = absScalar(childMinus(op, 0));
                        values_plus[op_idx] = absScalar(childPlus(op, 0));
                        break;

                    case FormExprType::Sign:
                        values_minus[op_idx] = signScalar(childMinus(op, 0));
                        values_plus[op_idx] = signScalar(childPlus(op, 0));
                        break;

                    case FormExprType::Sqrt:
                        values_minus[op_idx] = evalSqrt(childMinus(op, 0));
                        values_plus[op_idx] = evalSqrt(childPlus(op, 0));
                        break;

                    case FormExprType::Exp:
                        values_minus[op_idx] = evalExp(childMinus(op, 0));
                        values_plus[op_idx] = evalExp(childPlus(op, 0));
                        break;

                    case FormExprType::Log:
                        values_minus[op_idx] = evalLog(childMinus(op, 0));
                        values_plus[op_idx] = evalLog(childPlus(op, 0));
                        break;

	                    case FormExprType::MatrixExponential:
	                        values_minus[op_idx] = emitMatrixExp(childMinus(op, 0));
	                        values_plus[op_idx] = emitMatrixExp(childPlus(op, 0));
	                        break;

	                    case FormExprType::MatrixLogarithm:
	                        values_minus[op_idx] = emitMatrixLog(childMinus(op, 0));
	                        values_plus[op_idx] = emitMatrixLog(childPlus(op, 0));
	                        break;

	                    case FormExprType::MatrixSqrt:
	                        values_minus[op_idx] = emitMatrixSqrt(childMinus(op, 0));
	                        values_plus[op_idx] = emitMatrixSqrt(childPlus(op, 0));
	                        break;

	                    case FormExprType::MatrixPower:
	                        values_minus[op_idx] = emitMatrixPow(childMinus(op, 0), childMinus(op, 1).elems[0]);
	                        values_plus[op_idx] = emitMatrixPow(childPlus(op, 0), childPlus(op, 1).elems[0]);
	                        break;

                    case FormExprType::MatrixExponentialDirectionalDerivative:
                        values_minus[op_idx] = callMatrixUnaryDD(childMinus(op, 0), childMinus(op, 1), mat_exp_dd_2x2_fn, mat_exp_dd_3x3_fn);
                        values_plus[op_idx] = callMatrixUnaryDD(childPlus(op, 0), childPlus(op, 1), mat_exp_dd_2x2_fn, mat_exp_dd_3x3_fn);
                        break;

                    case FormExprType::MatrixLogarithmDirectionalDerivative:
                        values_minus[op_idx] = callMatrixUnaryDD(childMinus(op, 0), childMinus(op, 1), mat_log_dd_2x2_fn, mat_log_dd_3x3_fn);
                        values_plus[op_idx] = callMatrixUnaryDD(childPlus(op, 0), childPlus(op, 1), mat_log_dd_2x2_fn, mat_log_dd_3x3_fn);
                        break;

                    case FormExprType::MatrixSqrtDirectionalDerivative:
                        values_minus[op_idx] = callMatrixUnaryDD(childMinus(op, 0), childMinus(op, 1), mat_sqrt_dd_2x2_fn, mat_sqrt_dd_3x3_fn);
                        values_plus[op_idx] = callMatrixUnaryDD(childPlus(op, 0), childPlus(op, 1), mat_sqrt_dd_2x2_fn, mat_sqrt_dd_3x3_fn);
                        break;

                    case FormExprType::MatrixPowerDirectionalDerivative:
                        values_minus[op_idx] = callMatrixPowDD(childMinus(op, 0), childMinus(op, 1), childMinus(op, 2).elems[0], mat_pow_dd_2x2_fn, mat_pow_dd_3x3_fn);
                        values_plus[op_idx] = callMatrixPowDD(childPlus(op, 0), childPlus(op, 1), childPlus(op, 2).elems[0], mat_pow_dd_2x2_fn, mat_pow_dd_3x3_fn);
                        break;

                    case FormExprType::SmoothAbsoluteValue:
                        values_minus[op_idx] = smoothAbs(childMinus(op, 0), childMinus(op, 1));
                        values_plus[op_idx] = smoothAbs(childPlus(op, 0), childPlus(op, 1));
                        break;

                    case FormExprType::SmoothSign:
                        values_minus[op_idx] = smoothSign(childMinus(op, 0), childMinus(op, 1));
                        values_plus[op_idx] = smoothSign(childPlus(op, 0), childPlus(op, 1));
                        break;

                    case FormExprType::SmoothHeaviside:
                        values_minus[op_idx] = smoothHeaviside(childMinus(op, 0), childMinus(op, 1));
                        values_plus[op_idx] = smoothHeaviside(childPlus(op, 0), childPlus(op, 1));
                        break;

                    case FormExprType::SmoothMin:
                        values_minus[op_idx] = smoothMin(childMinus(op, 0), childMinus(op, 1), childMinus(op, 2));
                        values_plus[op_idx] = smoothMin(childPlus(op, 0), childPlus(op, 1), childPlus(op, 2));
                        break;

                    case FormExprType::SmoothMax:
                        values_minus[op_idx] = smoothMax(childMinus(op, 0), childMinus(op, 1), childMinus(op, 2));
                        values_plus[op_idx] = smoothMax(childPlus(op, 0), childPlus(op, 1), childPlus(op, 2));
                        break;

                    case FormExprType::SymmetricEigenvalue: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values_minus[op_idx] = eigSym(childMinus(op, 0), which);
                        values_plus[op_idx] = eigSym(childPlus(op, 0), which);
                        break;
                    }

                    case FormExprType::Eigenvalue: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values_minus[op_idx] = eigSym(childMinus(op, 0), which);
                        values_plus[op_idx] = eigSym(childPlus(op, 0), which);
                        break;
                    }

                    case FormExprType::SymmetricEigenvector: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values_minus[op_idx] = eigSymVec(childMinus(op, 0), which);
                        values_plus[op_idx] = eigSymVec(childPlus(op, 0), which);
                        break;
                    }

                    case FormExprType::SpectralDecomposition:
                        values_minus[op_idx] = spectralDecomp(childMinus(op, 0));
                        values_plus[op_idx] = spectralDecomp(childPlus(op, 0));
                        break;

                    case FormExprType::SymmetricEigenvectorDirectionalDerivative: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values_minus[op_idx] = eigSymVecDD(childMinus(op, 0), childMinus(op, 1), which);
                        values_plus[op_idx] = eigSymVecDD(childPlus(op, 0), childPlus(op, 1), which);
                        break;
                    }

                    case FormExprType::SpectralDecompositionDirectionalDerivative:
                        values_minus[op_idx] = spectralDecompDD(childMinus(op, 0), childMinus(op, 1));
                        values_plus[op_idx] = spectralDecompDD(childPlus(op, 0), childPlus(op, 1));
                        break;

                    case FormExprType::HistoryWeightedSum:
                    case FormExprType::HistoryConvolution: {
                        CodeValue acc_m = makeZero(shape);
                        CodeValue acc_p = makeZero(shape);
                        if (op.child_count != 0u) {
                            for (std::size_t kk = 0; kk < static_cast<std::size_t>(op.child_count); ++kk) {
                                const int k = static_cast<int>(kk + 1u);
                                acc_m = add(acc_m, mul(childMinus(op, kk), evalHistorySolution(side_minus, shape, k, q_index)));
                                acc_p = add(acc_p, mul(childPlus(op, kk), evalHistorySolution(side_plus, shape, k, q_index)));
                            }
                        } else {
                            for (int k = 1; k <= static_cast<int>(assembly::jit::kMaxPreviousSolutionsV6); ++k) {
                                acc_m = add(acc_m,
                                            mul(makeScalar(loadHistoryWeightOrZero(side_minus, k)),
                                                evalHistorySolution(side_minus, shape, k, q_index)));
                                acc_p = add(acc_p,
                                            mul(makeScalar(loadHistoryWeightOrZero(side_plus, k)),
                                                evalHistorySolution(side_plus, shape, k, q_index)));
                            }
                        }
                        values_minus[op_idx] = acc_m;
                        values_plus[op_idx] = acc_p;
                        break;
                    }

                    case FormExprType::SymmetricEigenvalueDirectionalDerivative: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values_minus[op_idx] = eigSymDD(childMinus(op, 0), childMinus(op, 1), which);
                        values_plus[op_idx] = eigSymDD(childPlus(op, 0), childPlus(op, 1), which);
                        break;
                    }

                    case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA: {
                        const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                        const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                        values_minus[op_idx] = eigSymDDWrtA(childMinus(op, 0), childMinus(op, 1), childMinus(op, 2), which);
                        values_plus[op_idx] = eigSymDDWrtA(childPlus(op, 0), childPlus(op, 1), childPlus(op, 2), which);
                        break;
                    }

                    default:
                        throw std::runtime_error("LLVMGen: unsupported op in cached face eval");
                }

                cached.minus[op_idx] = values_minus[op_idx];
                cached.plus[op_idx] = values_plus[op_idx];
            }

            return cached;
        };

        if (!is_face_domain) {
            for (std::size_t t = 0; t < terms.size(); ++t) {
                auto* term_entry = llvm::BasicBlock::Create(*ctx, "term" + std::to_string(t) + ".entry", fn);
                auto* term_body = llvm::BasicBlock::Create(*ctx, "term" + std::to_string(t) + ".body", fn);
                auto* term_end = llvm::BasicBlock::Create(*ctx, "term" + std::to_string(t) + ".end", fn);

                builder.CreateBr(term_entry);
                builder.SetInsertPoint(term_entry);
                auto* tw = termWeight(side_single, terms[t].time_derivative_order);
                auto* is_zero = builder.CreateFCmpOEQ(tw, f64c(0.0));
                builder.CreateCondBr(is_zero, term_end, term_body);

                builder.SetInsertPoint(term_body);
                emitForLoop(side_single.n_qpts, "q" + std::to_string(t), [&](llvm::Value* q) {
                    auto* q64 = builder.CreateZExt(q, i64);
                    auto* w = loadRealPtrAt(side_single.integration_weights, q64);
                    auto* scaled_w = builder.CreateFMul(tw, w);

                    std::vector<CodeValue> cached;
                    const std::vector<CodeValue>* cached_ptr = nullptr;
                    if (!terms[t].has_indexed_access) {
                        cached = computeCachedSingle(terms[t], q, side_single);
                        cached_ptr = &cached;
                    }

                    if (want_matrix) {
                        emitForLoop(side_single.n_test_dofs, "i" + std::to_string(t), [&](llvm::Value* i) {
                            emitForLoop(side_single.n_trial_dofs, "j" + std::to_string(t), [&](llvm::Value* j) {
                                auto* val = evalKernelIRSingle(terms[t], q, i, j, side_single, cached_ptr);
                                auto* contrib = builder.CreateFMul(scaled_w, val);
                                emitMatrixAccum(element_matrix_single, side_single.n_trial_dofs, i, j, contrib);
                            });
                        });
                    } else if (want_vector) {
                        emitForLoop(side_single.n_test_dofs, "i" + std::to_string(t), [&](llvm::Value* i) {
                            auto* val = evalKernelIRSingle(terms[t], q, i, builder.getInt32(0), side_single, cached_ptr);
                            auto* contrib = builder.CreateFMul(scaled_w, val);
                            emitVectorAccum(element_vector_single, i, contrib);
                        });
                    }
                });
                builder.CreateBr(term_end);
                builder.SetInsertPoint(term_end);
            }
        } else {
	            auto evalKernelIRFaceValue = [&](const LoweredTerm& term,
	                                             llvm::Value* q_index,
	                                             llvm::Value* i_index,
	                                             llvm::Value* j_index,
	                                             bool eval_plus,
	                                             bool test_active_plus,
	                                             bool trial_active_plus,
	                                             const std::vector<llvm::Value*>* indexed_env,
	                                             const FaceCached* cached) -> CodeValue {
	                std::vector<CodeValue> values_minus;
	                std::vector<CodeValue> values_plus;
	                values_minus.resize(term.ir.ops.size());
	                values_plus.resize(term.ir.ops.size());
	                const auto use_counts = kernelIRUseCounts(term.ir);

                auto childIndex = [&](const KernelIROp& op, std::size_t k) -> std::size_t {
                    return term.ir.children[static_cast<std::size_t>(op.first_child) + k];
                };

                auto childMinus = [&](const KernelIROp& op, std::size_t k) -> const CodeValue& {
                    return values_minus[childIndex(op, k)];
                };
                auto childPlus = [&](const KernelIROp& op, std::size_t k) -> const CodeValue& {
                    return values_plus[childIndex(op, k)];
                };

                auto evalTestFunction = [&](const SideView& side, bool is_plus, const Shape& shape) -> CodeValue {
                    if (is_plus != test_active_plus) {
                        return makeZero(shape);
                    }

                    if (shape.kind == Shape::Kind::Scalar) {
                        return makeScalar(loadBasisScalar(side, side.test_basis_values, i_index, q_index));
                    }

                    if (shape.kind != Shape::Kind::Vector) {
                        throw std::runtime_error("LLVMGen: TestFunction unsupported shape");
                    }

                    const auto vd = static_cast<std::size_t>(shape.dims[0]);
                    auto* uses_vec_basis = builder.CreateICmpNE(side.test_uses_vector_basis, builder.getInt32(0));
                    auto* vb = llvm::BasicBlock::Create(*ctx, "test.vec_basis", fn);
                    auto* sb = llvm::BasicBlock::Create(*ctx, "test.scalar_basis", fn);
                    auto* merge = llvm::BasicBlock::Create(*ctx, "test.merge", fn);

                    builder.CreateCondBr(uses_vec_basis, vb, sb);

                    std::array<llvm::Value*, 3> vb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
                    std::array<llvm::Value*, 3> sb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};

                    builder.SetInsertPoint(vb);
                    {
                        const auto v = loadVec3FromTable(side.test_basis_vector_values_xyz, side.n_qpts, i_index, q_index);
                        for (std::size_t c = 0; c < vd; ++c) {
                            vb_vals[c] = v[c];
                        }
                        builder.CreateBr(merge);
                    }
                    auto* vb_block = builder.GetInsertBlock();

                    builder.SetInsertPoint(sb);
                    {
                        const auto dofs_per_comp =
                            builder.CreateUDiv(side.n_test_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
                        const auto comp = builder.CreateUDiv(i_index, dofs_per_comp);
                        const auto phi = loadBasisScalar(side, side.test_basis_values, i_index, q_index);
                        for (std::size_t c = 0; c < vd; ++c) {
                            auto* is_c = builder.CreateICmpEQ(comp, builder.getInt32(static_cast<std::uint32_t>(c)));
                            sb_vals[c] = builder.CreateSelect(is_c, phi, f64c(0.0));
                        }
                        builder.CreateBr(merge);
                    }
                    auto* sb_block = builder.GetInsertBlock();

                    builder.SetInsertPoint(merge);
                    CodeValue out = makeZero(shape);
                    for (std::size_t c = 0; c < vd; ++c) {
                        auto* phi = builder.CreatePHI(f64, 2, "test.phi" + std::to_string(c));
                        phi->addIncoming(vb_vals[c], vb_block);
                        phi->addIncoming(sb_vals[c], sb_block);
                        out.elems[c] = phi;
                    }
                    return out;
                };

                auto evalTrialFunction = [&](const SideView& side, bool is_plus, const Shape& shape) -> CodeValue {
                    if (is_residual) {
                        // Residual: TrialFunction represents the current solution u_h(x_q).
                        return evalCurrentSolution(side, shape, q_index);
                    }

                    if (is_plus != trial_active_plus) {
                        return makeZero(shape);
                    }

                    if (shape.kind == Shape::Kind::Scalar) {
                        return makeScalar(loadBasisScalar(side, side.trial_basis_values, j_index, q_index));
                    }

                    if (shape.kind != Shape::Kind::Vector) {
                        throw std::runtime_error("LLVMGen: TrialFunction unsupported shape");
                    }

                    const auto vd = static_cast<std::size_t>(shape.dims[0]);
                    auto* uses_vec_basis = builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
                    auto* vb = llvm::BasicBlock::Create(*ctx, "trial.vec_basis", fn);
                    auto* sb = llvm::BasicBlock::Create(*ctx, "trial.scalar_basis", fn);
                    auto* merge = llvm::BasicBlock::Create(*ctx, "trial.merge", fn);

                    builder.CreateCondBr(uses_vec_basis, vb, sb);

                    std::array<llvm::Value*, 3> vb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
                    std::array<llvm::Value*, 3> sb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};

                    builder.SetInsertPoint(vb);
                    {
                        const auto v = loadVec3FromTable(side.trial_basis_vector_values_xyz, side.n_qpts, j_index, q_index);
                        for (std::size_t c = 0; c < vd; ++c) {
                            vb_vals[c] = v[c];
                        }
                        builder.CreateBr(merge);
                    }
                    auto* vb_block = builder.GetInsertBlock();

                    builder.SetInsertPoint(sb);
                    {
                        const auto dofs_per_comp =
                            builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
                        const auto comp = builder.CreateUDiv(j_index, dofs_per_comp);
                        const auto phi = loadBasisScalar(side, side.trial_basis_values, j_index, q_index);
                        for (std::size_t c = 0; c < vd; ++c) {
                            auto* is_c = builder.CreateICmpEQ(comp, builder.getInt32(static_cast<std::uint32_t>(c)));
                            sb_vals[c] = builder.CreateSelect(is_c, phi, f64c(0.0));
                        }
                        builder.CreateBr(merge);
                    }
                    auto* sb_block = builder.GetInsertBlock();

                    builder.SetInsertPoint(merge);
                    CodeValue out = makeZero(shape);
                    for (std::size_t c = 0; c < vd; ++c) {
                        auto* phi = builder.CreatePHI(f64, 2, "trial.phi" + std::to_string(c));
                        phi->addIncoming(vb_vals[c], vb_block);
                        phi->addIncoming(sb_vals[c], sb_block);
                        out.elems[c] = phi;
                    }
                    return out;
                };

                auto evalGradient = [&](const SideView& side, bool is_plus, const KernelIROp& op, const Shape& shape) -> CodeValue {
                    const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                    const auto& kid = term.ir.ops[child_idx];

	                    if (kid.type == FormExprType::TestFunction) {
	                        if (is_plus != test_active_plus) return makeZero(shape);

	                        if (shape.kind == Shape::Kind::Vector) {
	                            const auto v = loadVec3FromTableQMajor(side.test_phys_grads_xyz, side.n_test_dofs, i_index, q_index);
	                            return makeVector(shape.dims[0], v[0], v[1], v[2]);
	                        }
	                        if (shape.kind == Shape::Kind::Matrix) {
	                            const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                            const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                            CodeValue out = makeZero(shape);
	                            const auto dofs_per_comp = builder.CreateUDiv(
	                                side.n_test_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                            const auto comp = builder.CreateUDiv(i_index, dofs_per_comp);
	                            const auto g = loadVec3FromTableQMajor(side.test_phys_grads_xyz, side.n_test_dofs, i_index, q_index);
	                            for (std::size_t r = 0; r < vd; ++r) {
	                                auto* is_r = builder.CreateICmpEQ(comp, builder.getInt32(static_cast<std::uint32_t>(r)));
	                                for (std::size_t d = 0; d < dim; ++d) {
	                                    out.elems[r * dim + d] = builder.CreateSelect(is_r, g[d], f64c(0.0));
	                                }
	                            }
	                            return out;
	                        }
                        throw std::runtime_error("LLVMGen: grad(TestFunction) unsupported shape");
                    }

                    if (kid.type == FormExprType::TrialFunction) {
	                        if (is_residual) {
	                            // Residual: grad(u_h) computed from current solution coefficients.
		                            auto* coeffs = side.solution_coefficients;
		                            if (shape.kind == Shape::Kind::Vector) {
		                                const auto dim = static_cast<std::size_t>(shape.dims[0]);
		                                const auto sums =
		                                    emitReduceSum(side.n_trial_dofs, "grad_u", dim, [&](llvm::Value* j) {
		                                        auto* j64 = builder.CreateZExt(j, i64);
		                                        auto* cj = loadRealPtrAt(coeffs, j64);
		                                        const auto g =
		                                            loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
		                                        std::vector<llvm::Value*> terms;
		                                        terms.reserve(dim);
		                                        for (std::size_t d = 0; d < dim; ++d) {
		                                            terms.push_back(builder.CreateFMul(cj, g[d]));
		                                        }
		                                        return terms;
		                                    });
		                                auto* x = sums[0];
		                                auto* y = (dim > 1u) ? sums[1] : f64c(0.0);
		                                auto* z = (dim > 2u) ? sums[2] : f64c(0.0);
		                                return makeVector(shape.dims[0], x, y, z);
		                            }
		                            if (shape.kind == Shape::Kind::Matrix) {
		                                const auto vd = static_cast<std::size_t>(shape.dims[0]);
		                                const auto dim = static_cast<std::size_t>(shape.dims[1]);
		                                CodeValue out = makeZero(shape);
		                                auto* dofs_per_comp =
		                                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
		                                for (std::size_t comp = 0; comp < vd; ++comp) {
	                                    const auto acc = emitReduceSum(dofs_per_comp, "grad_u_c" + std::to_string(comp), dim, [&](llvm::Value* jj) {
	                                        auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                        auto* j = builder.CreateAdd(base, jj);
	                                        auto* j64 = builder.CreateZExt(j, i64);
		                                        auto* cj = loadRealPtrAt(coeffs, j64);
		                                        const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                        std::vector<llvm::Value*> terms;
	                                        terms.reserve(dim);
		                                        for (std::size_t d = 0; d < dim; ++d) {
		                                            terms.push_back(builder.CreateFMul(cj, g[d]));
		                                        }
	                                        return terms;
		                                    });
		                                    for (std::size_t d = 0; d < dim; ++d) {
		                                        out.elems[comp * dim + d] = acc[d];
		                                    }
		                                }
	                                return out;
	                            }
                            throw std::runtime_error("LLVMGen: grad(u) unsupported shape (residual)");
                        }

	                        if (is_plus != trial_active_plus) return makeZero(shape);

	                        if (shape.kind == Shape::Kind::Vector) {
	                            const auto v = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j_index, q_index);
	                            return makeVector(shape.dims[0], v[0], v[1], v[2]);
	                        }
	                        if (shape.kind == Shape::Kind::Matrix) {
	                            const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                            const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                            CodeValue out = makeZero(shape);
	                            const auto dofs_per_comp = builder.CreateUDiv(
	                                side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                            const auto comp = builder.CreateUDiv(j_index, dofs_per_comp);
	                            const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j_index, q_index);
	                            for (std::size_t r = 0; r < vd; ++r) {
	                                auto* is_r = builder.CreateICmpEQ(comp, builder.getInt32(static_cast<std::uint32_t>(r)));
	                                for (std::size_t d = 0; d < dim; ++d) {
	                                    out.elems[r * dim + d] = builder.CreateSelect(is_r, g[d], f64c(0.0));
	                                }
	                            }
	                            return out;
	                        }
                        throw std::runtime_error("LLVMGen: grad(TrialFunction) unsupported shape");
                    }

	                    if (kid.type == FormExprType::PreviousSolutionRef) {
	                        const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
		                        auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
		                        if (shape.kind == Shape::Kind::Vector) {
		                            const auto dim = static_cast<std::size_t>(shape.dims[0]);
		                            const auto sums =
		                                emitReduceSum(side.n_trial_dofs, "prev_grad" + std::to_string(k), dim, [&](llvm::Value* j) {
		                                    auto* j64 = builder.CreateZExt(j, i64);
		                                    auto* cj = loadRealPtrAt(coeffs, j64);
		                                    const auto g =
		                                        loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
		                                    std::vector<llvm::Value*> terms;
		                                    terms.reserve(dim);
		                                    for (std::size_t d = 0; d < dim; ++d) {
		                                        terms.push_back(builder.CreateFMul(cj, g[d]));
		                                    }
		                                    return terms;
		                                });
		                            auto* x = sums[0];
		                            auto* y = (dim > 1u) ? sums[1] : f64c(0.0);
		                            auto* z = (dim > 2u) ? sums[2] : f64c(0.0);
		                            return makeVector(shape.dims[0], x, y, z);
		                        }
		                        if (shape.kind == Shape::Kind::Matrix) {
		                            const auto vd = static_cast<std::size_t>(shape.dims[0]);
		                            const auto dim = static_cast<std::size_t>(shape.dims[1]);
		                            CodeValue out = makeZero(shape);
		                            auto* dofs_per_comp =
		                                builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
		                            for (std::size_t comp = 0; comp < vd; ++comp) {
	                                const auto acc = emitReduceSum(dofs_per_comp,
	                                                               "prev_grad" + std::to_string(k) + "_c" + std::to_string(comp),
	                                                               dim,
	                                                               [&](llvm::Value* jj) {
	                                    auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                    auto* j = builder.CreateAdd(base, jj);
	                                    auto* j64 = builder.CreateZExt(j, i64);
		                                    auto* cj = loadRealPtrAt(coeffs, j64);
		                                    const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                    std::vector<llvm::Value*> terms;
	                                    terms.reserve(dim);
		                                    for (std::size_t d = 0; d < dim; ++d) {
		                                        terms.push_back(builder.CreateFMul(cj, g[d]));
		                                    }
	                                    return terms;
		                                });
		                                for (std::size_t d = 0; d < dim; ++d) {
		                                    out.elems[comp * dim + d] = acc[d];
		                                }
		                            }
	                            return out;
	                        }
                        throw std::runtime_error("LLVMGen: grad(PreviousSolutionRef) unsupported shape");
                    }

	                    if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
	                        const int fid = unpackFieldIdImm1(kid.imm1);
	                        if (kid.type == FormExprType::StateField && fid == 0xffff) {
	                            // Current solution state u.
		                            auto* coeffs = side.solution_coefficients;
		                            if (shape.kind == Shape::Kind::Vector) {
		                                const auto dim = static_cast<std::size_t>(shape.dims[0]);
		                                const auto sums =
		                                    emitReduceSum(side.n_trial_dofs, "grad_state_u", dim, [&](llvm::Value* j) {
		                                        auto* j64 = builder.CreateZExt(j, i64);
		                                        auto* cj = loadRealPtrAt(coeffs, j64);
		                                        const auto g =
		                                            loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
		                                        std::vector<llvm::Value*> terms;
		                                        terms.reserve(dim);
		                                        for (std::size_t d = 0; d < dim; ++d) {
		                                            terms.push_back(builder.CreateFMul(cj, g[d]));
		                                        }
		                                        return terms;
		                                    });
		                                auto* x = sums[0];
		                                auto* y = (dim > 1u) ? sums[1] : f64c(0.0);
		                                auto* z = (dim > 2u) ? sums[2] : f64c(0.0);
		                                return makeVector(shape.dims[0], x, y, z);
		                            }
		                            if (shape.kind == Shape::Kind::Matrix) {
		                                const auto vd = static_cast<std::size_t>(shape.dims[0]);
		                                const auto dim = static_cast<std::size_t>(shape.dims[1]);
		                                CodeValue out = makeZero(shape);
		                                auto* dofs_per_comp =
		                                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
		                                for (std::size_t comp = 0; comp < vd; ++comp) {
	                                    const auto acc =
	                                        emitReduceSum(dofs_per_comp, "grad_state_u_c" + std::to_string(comp), dim, [&](llvm::Value* jj) {
	                                        auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                        auto* j = builder.CreateAdd(base, jj);
		                                        auto* j64 = builder.CreateZExt(j, i64);
		                                        auto* cj = loadRealPtrAt(coeffs, j64);
		                                        const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                        std::vector<llvm::Value*> terms;
	                                        terms.reserve(dim);
		                                        for (std::size_t d = 0; d < dim; ++d) {
		                                            terms.push_back(builder.CreateFMul(cj, g[d]));
		                                        }
	                                        return terms;
		                                    });
		                                    for (std::size_t d = 0; d < dim; ++d) {
		                                        out.elems[comp * dim + d] = acc[d];
		                                    }
		                                }
	                                return out;
	                            }
	                            throw std::runtime_error("LLVMGen: grad(StateField(INVALID)) unsupported shape");
	                        }
	                        if (shape.kind == Shape::Kind::Vector) {
                            auto* entry = fieldEntryPtrFor(is_plus, fid);
                            auto* entry_is_null =
                                builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok = llvm::BasicBlock::Create(*ctx, "field_grad.ok", fn);
                            auto* zero = llvm::BasicBlock::Create(*ctx, "field_grad.zero", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "field_grad.merge", fn);

                            builder.CreateCondBr(entry_is_null, zero, ok);

                            std::array<llvm::Value*, 3> loaded{f64c(0.0), f64c(0.0), f64c(0.0)};
                            builder.SetInsertPoint(ok);
                            auto* base = loadPtr(entry, ABIV3::field_entry_gradients_xyz_off);
                            auto* base_is_null =
                                builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_grad.xyz.ok", fn);
                            builder.CreateCondBr(base_is_null, zero, ok2);

                            builder.SetInsertPoint(ok2);
                            const auto v = loadXYZ(base, q_index);
                            loaded = {v.elems[0], v.elems[1], v.elems[2]};
                            builder.CreateBr(merge);
                            auto* ok2_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(zero);
                            builder.CreateBr(merge);
                            auto* zero_block = builder.GetInsertBlock();

	                            builder.SetInsertPoint(merge);
	                            std::array<llvm::Value*, 3> outv{f64c(0.0), f64c(0.0), f64c(0.0)};
	                            const auto dim = static_cast<std::size_t>(shape.dims[0]);
	                            for (std::size_t d = 0; d < dim; ++d) {
	                                auto* phi = builder.CreatePHI(f64, 2, "field_grad.v" + std::to_string(d));
	                                phi->addIncoming(f64c(0.0), zero_block);
	                                phi->addIncoming(loaded[d], ok2_block);
	                                outv[d] = phi;
	                            }
	                            return makeVector(shape.dims[0], outv[0], outv[1], outv[2]);
	                        }
	                        if (shape.kind == Shape::Kind::Matrix) {
	                            const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                            const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                            auto* entry = fieldEntryPtrFor(is_plus, fid);
	                            auto* entry_is_null =
	                                builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok = llvm::BasicBlock::Create(*ctx, "field_jac.ok", fn);
                            auto* zero = llvm::BasicBlock::Create(*ctx, "field_jac.zero", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "field_jac.merge", fn);

                            builder.CreateCondBr(entry_is_null, zero, ok);

                            CodeValue loaded = makeZero(matrixShape(3u, 3u));
                            builder.SetInsertPoint(ok);
                            auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                            auto* base_is_null =
                                builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_jac.mat.ok", fn);
                            builder.CreateCondBr(base_is_null, zero, ok2);

                            builder.SetInsertPoint(ok2);
                            loaded = loadMat3FromQ(base, q_index);
                            builder.CreateBr(merge);
                            auto* ok2_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(zero);
                            builder.CreateBr(merge);
                            auto* zero_block = builder.GetInsertBlock();

	                            builder.SetInsertPoint(merge);
	                            CodeValue out = makeZero(shape);
	                            for (std::size_t r = 0; r < vd; ++r) {
	                                for (std::size_t c = 0; c < dim; ++c) {
	                                    auto* phi = builder.CreatePHI(f64, 2, "field_jac.a");
	                                    phi->addIncoming(f64c(0.0), zero_block);
	                                    auto* l =
	                                        (r < 3u && c < 3u) ? loaded.elems[r * 3u + c] : f64c(0.0);
	                                    phi->addIncoming(l, ok2_block);
	                                    out.elems[r * dim + c] = phi;
	                                }
	                            }
	                            return out;
	                        }
                        throw std::runtime_error("LLVMGen: grad(DiscreteField) unsupported shape");
                    }

                    if (kid.type == FormExprType::TimeDerivative) {
                        const int order = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
                        auto* coeff0 = loadDtCoeff0(side, order);
                        const auto dt_child_idx =
                            term.ir.children[static_cast<std::size_t>(kid.first_child)];
                        const auto& dt_child = term.ir.ops[dt_child_idx];

	                        auto gradTrialBasis = [&]() -> CodeValue {
	                            if (is_plus != trial_active_plus) {
	                                return makeZero(shape);
	                            }
	                            CodeValue g = makeZero(shape);
	                            if (shape.kind == Shape::Kind::Vector) {
	                                const auto v = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j_index, q_index);
	                                return makeVector(shape.dims[0], v[0], v[1], v[2]);
	                            }
	                            if (shape.kind == Shape::Kind::Matrix) {
	                                const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                                const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                                const auto dofs_per_comp = builder.CreateUDiv(
	                                    side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                                const auto comp = builder.CreateUDiv(j_index, dofs_per_comp);
	                                const auto gg = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j_index, q_index);
	                                for (std::size_t r = 0; r < vd; ++r) {
	                                    auto* is_r = builder.CreateICmpEQ(comp, builder.getInt32(static_cast<std::uint32_t>(r)));
	                                    for (std::size_t d = 0; d < dim; ++d) {
	                                        g.elems[r * dim + d] = builder.CreateSelect(is_r, gg[d], f64c(0.0));
	                                    }
	                                }
	                                return g;
	                            }
                            throw std::runtime_error("LLVMGen: grad(dt(TrialFunction)) unsupported shape");
                        };

		                        auto gradCurrentSolution = [&]() -> CodeValue {
		                            auto* coeffs = side.solution_coefficients;
		                            if (shape.kind == Shape::Kind::Vector) {
		                                const auto dim = static_cast<std::size_t>(shape.dims[0]);
		                                const auto sums =
		                                    emitReduceSum(side.n_trial_dofs, "grad_dt_u", dim, [&](llvm::Value* j) {
		                                        auto* j64 = builder.CreateZExt(j, i64);
		                                        auto* cj = loadRealPtrAt(coeffs, j64);
		                                        const auto g =
		                                            loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
		                                        std::vector<llvm::Value*> terms;
		                                        terms.reserve(dim);
		                                        for (std::size_t d = 0; d < dim; ++d) {
		                                            terms.push_back(builder.CreateFMul(cj, g[d]));
		                                        }
		                                        return terms;
		                                    });
		                                auto* x = sums[0];
		                                auto* y = (dim > 1u) ? sums[1] : f64c(0.0);
		                                auto* z = (dim > 2u) ? sums[2] : f64c(0.0);
		                                return makeVector(shape.dims[0], x, y, z);
		                            }
		                            if (shape.kind == Shape::Kind::Matrix) {
		                                const auto vd = static_cast<std::size_t>(shape.dims[0]);
		                                const auto dim = static_cast<std::size_t>(shape.dims[1]);
		                                CodeValue out = makeZero(shape);
		                                auto* dofs_per_comp =
		                                    builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
		                                for (std::size_t comp = 0; comp < vd; ++comp) {
	                                    const auto acc = emitReduceSum(
	                                        dofs_per_comp, "grad_dt_u_c" + std::to_string(comp), dim, [&](llvm::Value* jj) {
	                                        auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                        auto* j = builder.CreateAdd(base, jj);
		                                        auto* j64 = builder.CreateZExt(j, i64);
		                                        auto* cj = loadRealPtrAt(coeffs, j64);
		                                        const auto g = loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                        std::vector<llvm::Value*> terms;
	                                        terms.reserve(dim);
		                                        for (std::size_t d = 0; d < dim; ++d) {
		                                            terms.push_back(builder.CreateFMul(cj, g[d]));
		                                        }
	                                        return terms;
		                                    });
		                                    for (std::size_t d = 0; d < dim; ++d) {
		                                        out.elems[comp * dim + d] = acc[d];
		                                    }
		                                }
	                                return out;
	                            }
                            throw std::runtime_error("LLVMGen: grad(dt(u)) unsupported shape");
                        };

                        auto gradDiscreteOrStateField = [&](int fid) -> CodeValue {
                            if (shape.kind == Shape::Kind::Vector) {
                                auto* entry = fieldEntryPtrFor(is_plus, fid);
                                auto* entry_is_null =
                                    builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok = llvm::BasicBlock::Create(*ctx, "field_dt_grad.ok", fn);
                                auto* zero = llvm::BasicBlock::Create(*ctx, "field_dt_grad.zero", fn);
                                auto* merge = llvm::BasicBlock::Create(*ctx, "field_dt_grad.merge", fn);

                                builder.CreateCondBr(entry_is_null, zero, ok);

                                std::array<llvm::Value*, 3> loaded{f64c(0.0), f64c(0.0), f64c(0.0)};
                                builder.SetInsertPoint(ok);
                                auto* base = loadPtr(entry, ABIV3::field_entry_gradients_xyz_off);
                                auto* base_is_null =
                                    builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_dt_grad.xyz.ok", fn);
                                builder.CreateCondBr(base_is_null, zero, ok2);

                                builder.SetInsertPoint(ok2);
                                const auto v = loadXYZ(base, q_index);
                                loaded = {v.elems[0], v.elems[1], v.elems[2]};
                                builder.CreateBr(merge);
                                auto* ok2_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(zero);
                                builder.CreateBr(merge);
                                auto* zero_block = builder.GetInsertBlock();

	                                builder.SetInsertPoint(merge);
	                                std::array<llvm::Value*, 3> outv{f64c(0.0), f64c(0.0), f64c(0.0)};
	                                const auto dim = static_cast<std::size_t>(shape.dims[0]);
	                                for (std::size_t d = 0; d < dim; ++d) {
	                                    auto* phi = builder.CreatePHI(f64, 2, "field_dt_grad.v" + std::to_string(d));
	                                    phi->addIncoming(f64c(0.0), zero_block);
	                                    phi->addIncoming(loaded[d], ok2_block);
	                                    outv[d] = phi;
	                                }
	                                return makeVector(shape.dims[0], outv[0], outv[1], outv[2]);
	                            }
	                            if (shape.kind == Shape::Kind::Matrix) {
	                                const auto vd = static_cast<std::size_t>(shape.dims[0]);
	                                const auto dim = static_cast<std::size_t>(shape.dims[1]);
	                                auto* entry = fieldEntryPtrFor(is_plus, fid);
	                                auto* entry_is_null =
	                                    builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok = llvm::BasicBlock::Create(*ctx, "field_dt_jac.ok", fn);
                                auto* zero = llvm::BasicBlock::Create(*ctx, "field_dt_jac.zero", fn);
                                auto* merge = llvm::BasicBlock::Create(*ctx, "field_dt_jac.merge", fn);

                                builder.CreateCondBr(entry_is_null, zero, ok);

                                CodeValue loaded = makeZero(matrixShape(3u, 3u));
                                builder.SetInsertPoint(ok);
                                auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                                auto* base_is_null =
                                    builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                                auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_dt_jac.mat.ok", fn);
                                builder.CreateCondBr(base_is_null, zero, ok2);

                                builder.SetInsertPoint(ok2);
                                loaded = loadMat3FromQ(base, q_index);
                                builder.CreateBr(merge);
                                auto* ok2_block = builder.GetInsertBlock();

                                builder.SetInsertPoint(zero);
                                builder.CreateBr(merge);
                                auto* zero_block = builder.GetInsertBlock();

	                                builder.SetInsertPoint(merge);
	                                CodeValue out = makeZero(shape);
	                                for (std::size_t r = 0; r < vd; ++r) {
	                                    for (std::size_t c = 0; c < dim; ++c) {
	                                        auto* phi = builder.CreatePHI(f64, 2, "field_dt_jac.a");
	                                        phi->addIncoming(f64c(0.0), zero_block);
	                                        auto* l =
	                                            (r < 3u && c < 3u) ? loaded.elems[r * 3u + c] : f64c(0.0);
	                                        phi->addIncoming(l, ok2_block);
	                                        out.elems[r * dim + c] = phi;
	                                    }
	                                }
	                                return out;
	                            }
                            throw std::runtime_error("LLVMGen: grad(dt(field)) unsupported shape");
                        };

                        if (dt_child.type == FormExprType::TrialFunction) {
                            const auto g = is_residual ? gradCurrentSolution() : gradTrialBasis();
                            return mul(makeScalar(coeff0), g);
                        }

                        if (dt_child.type == FormExprType::DiscreteField || dt_child.type == FormExprType::StateField) {
                            const int fid = unpackFieldIdImm1(dt_child.imm1);
                            const auto g = (dt_child.type == FormExprType::StateField && fid == 0xffff)
                                               ? gradCurrentSolution()
                                               : gradDiscreteOrStateField(fid);
                            return mul(makeScalar(coeff0), g);
                        }

                        if (dt_child.type == FormExprType::Constant) {
                            return makeZero(shape);
                        }

                        throw std::runtime_error("LLVMGen: grad(dt(...)) operand not supported");
                    }

                    throw std::runtime_error("LLVMGen: Gradient operand not supported");
                };

                auto evalDivergence = [&](const SideView& side, bool is_plus, const KernelIROp& op) -> CodeValue {
                    const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                    const auto& kid = term.ir.ops[child_idx];
                    const auto vd = static_cast<std::size_t>(term.shapes[child_idx].dims[0]);

	                    auto divScalarBasis = [&](llvm::Value* n_dofs,
	                                              llvm::Value* dof_index,
	                                              llvm::Value* grads_xyz) -> llvm::Value* {
	                        auto* dofs_per_comp = builder.CreateUDiv(n_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                        auto* comp = builder.CreateUDiv(dof_index, dofs_per_comp);
	                        const auto g = loadVec3FromTableQMajor(grads_xyz, n_dofs, dof_index, q_index);
	                        llvm::Value* out = f64c(0.0);
	                        if (vd >= 1) {
	                            out = builder.CreateSelect(builder.CreateICmpEQ(comp, builder.getInt32(0)), g[0], out);
	                        }
	                        if (vd >= 2) {
	                            out = builder.CreateSelect(builder.CreateICmpEQ(comp, builder.getInt32(1)), g[1], out);
                        }
                        if (vd >= 3) {
                            out = builder.CreateSelect(builder.CreateICmpEQ(comp, builder.getInt32(2)), g[2], out);
                        }
                        return out;
                    };

                    auto divTest = [&]() -> llvm::Value* {
                        if (is_plus != test_active_plus) return f64c(0.0);

                        auto* uses_vec_basis = builder.CreateICmpNE(side.test_uses_vector_basis, builder.getInt32(0));
                        auto* vb = llvm::BasicBlock::Create(*ctx, "div.test.vec_basis", fn);
                        auto* sb = llvm::BasicBlock::Create(*ctx, "div.test.scalar_basis", fn);
                        auto* merge = llvm::BasicBlock::Create(*ctx, "div.test.merge", fn);

                        builder.CreateCondBr(uses_vec_basis, vb, sb);

                        llvm::Value* v_vb = f64c(0.0);
                        builder.SetInsertPoint(vb);
                        v_vb = loadBasisScalar(side, side.test_basis_divs, i_index, q_index);
                        builder.CreateBr(merge);
                        auto* vb_block = builder.GetInsertBlock();

                        llvm::Value* v_sb = f64c(0.0);
                        builder.SetInsertPoint(sb);
                        v_sb = divScalarBasis(side.n_test_dofs, i_index, side.test_phys_grads_xyz);
                        builder.CreateBr(merge);
                        auto* sb_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(merge);
                        auto* phi = builder.CreatePHI(f64, 2, "div.test");
                        phi->addIncoming(v_vb, vb_block);
                        phi->addIncoming(v_sb, sb_block);
                        return phi;
                    };

                    auto divTrial = [&]() -> llvm::Value* {
                        if (is_plus != trial_active_plus) return f64c(0.0);

                        auto* uses_vec_basis = builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
                        auto* vb = llvm::BasicBlock::Create(*ctx, "div.trial.vec_basis", fn);
                        auto* sb = llvm::BasicBlock::Create(*ctx, "div.trial.scalar_basis", fn);
                        auto* merge = llvm::BasicBlock::Create(*ctx, "div.trial.merge", fn);

                        builder.CreateCondBr(uses_vec_basis, vb, sb);

                        llvm::Value* v_vb = f64c(0.0);
                        builder.SetInsertPoint(vb);
                        v_vb = loadBasisScalar(side, side.trial_basis_divs, j_index, q_index);
                        builder.CreateBr(merge);
                        auto* vb_block = builder.GetInsertBlock();

                        llvm::Value* v_sb = f64c(0.0);
                        builder.SetInsertPoint(sb);
                        v_sb = divScalarBasis(side.n_trial_dofs, j_index, side.trial_phys_grads_xyz);
                        builder.CreateBr(merge);
                        auto* sb_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(merge);
                        auto* phi = builder.CreatePHI(f64, 2, "div.trial");
                        phi->addIncoming(v_vb, vb_block);
                        phi->addIncoming(v_sb, sb_block);
                        return phi;
                    };

                    auto divCurrentSolution = [&]() -> llvm::Value* {
                        auto* coeffs = side.solution_coefficients;
                        auto* uses_vec_basis = builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
                        auto* vb = llvm::BasicBlock::Create(*ctx, "div.u.vec_basis", fn);
                        auto* sb = llvm::BasicBlock::Create(*ctx, "div.u.scalar_basis", fn);
                        auto* merge = llvm::BasicBlock::Create(*ctx, "div.u.merge", fn);

                        builder.CreateCondBr(uses_vec_basis, vb, sb);

		                        llvm::Value* div_vb = f64c(0.0);
		                        builder.SetInsertPoint(vb);
		                        {
		                            div_vb = emitReduceSumScalar(side.n_trial_dofs, "div_u_vb", [&](llvm::Value* j) -> llvm::Value* {
		                                auto* j64 = builder.CreateZExt(j, i64);
		                                auto* cj = loadRealPtrAt(coeffs, j64);
		                                auto* div_phi = loadBasisScalar(side, side.trial_basis_divs, j, q_index);
		                                return builder.CreateFMul(cj, div_phi);
		                            });
	                            builder.CreateBr(merge);
	                        }
	                        auto* vb_block = builder.GetInsertBlock();

		                        llvm::Value* div_sb = f64c(0.0);
		                        builder.SetInsertPoint(sb);
		                        {
		                            div_sb = emitReduceSumScalar(
		                                side.n_trial_dofs, "div_u_sb", [&](llvm::Value* j) -> llvm::Value* {
		                                    auto* j64 = builder.CreateZExt(j, i64);
		                                    auto* cj = loadRealPtrAt(coeffs, j64);
		                                    auto* div_phi = divScalarBasis(side.n_trial_dofs, j, side.trial_phys_grads_xyz);
		                                    return builder.CreateFMul(cj, div_phi);
		                                });
	                            builder.CreateBr(merge);
	                        }
	                        auto* sb_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(merge);
                        auto* phi = builder.CreatePHI(f64, 2, "div.u");
                        phi->addIncoming(div_vb, vb_block);
                        phi->addIncoming(div_sb, sb_block);
                        return phi;
                    };

                    if (kid.type == FormExprType::TestFunction) {
                        return makeScalar(divTest());
                    }
                    if (kid.type == FormExprType::TrialFunction) {
                        return makeScalar(is_residual ? divCurrentSolution() : divTrial());
                    }
	                    if (kid.type == FormExprType::PreviousSolutionRef) {
	                        const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
	                        auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
		                        auto* dofs_per_comp =
		                            builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
		                        llvm::Value* div = f64c(0.0);
		                        for (std::size_t comp = 0; comp < vd; ++comp) {
		                            auto* acc = emitReduceSumScalar(
		                                dofs_per_comp,
		                                "prev_div" + std::to_string(k) + "_c" + std::to_string(comp),
		                                [&](llvm::Value* jj) -> llvm::Value* {
		                                    auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
		                                    auto* j = builder.CreateAdd(base, jj);
		                                    auto* j64 = builder.CreateZExt(j, i64);
		                                    auto* cj = loadRealPtrAt(coeffs, j64);
		                                    const auto g =
		                                        loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
		                                    return builder.CreateFMul(cj, g[comp]);
		                                });
		                            div = builder.CreateFAdd(div, acc);
		                        }
		                        return makeScalar(div);
		                    }
                    if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                        const int fid = unpackFieldIdImm1(kid.imm1);
                        if (kid.type == FormExprType::StateField && fid == 0xffff) {
                            return makeScalar(divCurrentSolution());
                        }
                        auto* entry = fieldEntryPtrFor(is_plus, fid);
                        auto* entry_is_null =
                            builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                        auto* ok = llvm::BasicBlock::Create(*ctx, "field_div.ok", fn);
                        auto* zero = llvm::BasicBlock::Create(*ctx, "field_div.zero", fn);
                        auto* merge = llvm::BasicBlock::Create(*ctx, "field_div.merge", fn);

                        builder.CreateCondBr(entry_is_null, zero, ok);

                        llvm::Value* div = f64c(0.0);
                        builder.SetInsertPoint(ok);
                        auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                        auto* base_is_null =
                            builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                        auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_div.mat.ok", fn);
                        builder.CreateCondBr(base_is_null, zero, ok2);

                        builder.SetInsertPoint(ok2);
                        const auto J = loadMat3FromQ(base, q_index);
                        div = J.elems[0];
                        if (vd >= 2) div = builder.CreateFAdd(div, J.elems[4]);
                        if (vd >= 3) div = builder.CreateFAdd(div, J.elems[8]);
                        builder.CreateBr(merge);
                        auto* ok2_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(zero);
                        builder.CreateBr(merge);
                        auto* zero_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(merge);
                        auto* phi = builder.CreatePHI(f64, 2, "field_div");
                        phi->addIncoming(f64c(0.0), zero_block);
                        phi->addIncoming(div, ok2_block);
                        return makeScalar(phi);
                    }
                    if (kid.type == FormExprType::TimeDerivative) {
                        const int order = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
                        auto* coeff0 = loadDtCoeff0(side, order);
                        const auto dt_child_idx =
                            term.ir.children[static_cast<std::size_t>(kid.first_child)];
                        const auto& dt_child = term.ir.ops[dt_child_idx];

                        if (dt_child.type == FormExprType::TrialFunction) {
                            return makeScalar(builder.CreateFMul(coeff0, is_residual ? divCurrentSolution() : divTrial()));
                        }

                        if (dt_child.type == FormExprType::DiscreteField || dt_child.type == FormExprType::StateField) {
                            const int fid = unpackFieldIdImm1(dt_child.imm1);
                            if (dt_child.type == FormExprType::StateField && fid == 0xffff) {
                                return makeScalar(builder.CreateFMul(coeff0, divCurrentSolution()));
                            }

                            auto* entry = fieldEntryPtrFor(is_plus, fid);
                            auto* entry_is_null =
                                builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok = llvm::BasicBlock::Create(*ctx, "field_dt_div.ok", fn);
                            auto* zero = llvm::BasicBlock::Create(*ctx, "field_dt_div.zero", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "field_dt_div.merge", fn);

                            builder.CreateCondBr(entry_is_null, zero, ok);

                            llvm::Value* div = f64c(0.0);
                            builder.SetInsertPoint(ok);
                            auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                            auto* base_is_null =
                                builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_dt_div.mat.ok", fn);
                            builder.CreateCondBr(base_is_null, zero, ok2);

                            builder.SetInsertPoint(ok2);
                            const auto J = loadMat3FromQ(base, q_index);
                            div = J.elems[0];
                            if (vd >= 2) div = builder.CreateFAdd(div, J.elems[4]);
                            if (vd >= 3) div = builder.CreateFAdd(div, J.elems[8]);
                            div = builder.CreateFMul(coeff0, div);
                            builder.CreateBr(merge);
                            auto* ok2_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(zero);
                            builder.CreateBr(merge);
                            auto* zero_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(merge);
                            auto* phi = builder.CreatePHI(f64, 2, "field_dt_div");
                            phi->addIncoming(f64c(0.0), zero_block);
                            phi->addIncoming(div, ok2_block);
                            return makeScalar(phi);
                        }

                        if (dt_child.type == FormExprType::Constant) {
                            return makeScalar(f64c(0.0));
                        }

                        throw std::runtime_error("LLVMGen: div(dt(...)) operand not supported");
                    }
                    if (kid.type == FormExprType::Constant) {
                        return makeScalar(f64c(0.0));
                    }
                    throw std::runtime_error("LLVMGen: Divergence operand not supported");
                };

                auto evalCurl = [&](const SideView& side, bool is_plus, const KernelIROp& op) -> CodeValue {
                    const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                    const auto& kid = term.ir.ops[child_idx];
                    const auto vd = static_cast<std::size_t>(term.shapes[child_idx].dims[0]);

	                    auto curlScalarBasis = [&](llvm::Value* n_dofs,
	                                               llvm::Value* dof_index,
	                                               llvm::Value* grads_xyz) -> std::array<llvm::Value*, 3> {
	                        auto* dofs_per_comp = builder.CreateUDiv(n_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));
	                        auto* comp = builder.CreateUDiv(dof_index, dofs_per_comp);
	                        const auto g = loadVec3FromTableQMajor(grads_xyz, n_dofs, dof_index, q_index);
	                        llvm::Value* x = f64c(0.0);
	                        llvm::Value* y = f64c(0.0);
	                        llvm::Value* z = f64c(0.0);
	                        if (vd >= 1) {
	                            auto* is0 = builder.CreateICmpEQ(comp, builder.getInt32(0));
	                            y = builder.CreateSelect(is0, g[2], y);
	                            z = builder.CreateSelect(is0, builder.CreateFNeg(g[1]), z);
	                        }
                        if (vd >= 2) {
                            auto* is1 = builder.CreateICmpEQ(comp, builder.getInt32(1));
                            x = builder.CreateSelect(is1, builder.CreateFNeg(g[2]), x);
                            z = builder.CreateSelect(is1, g[0], z);
                        }
                        if (vd >= 3) {
                            auto* is2 = builder.CreateICmpEQ(comp, builder.getInt32(2));
                            x = builder.CreateSelect(is2, g[1], x);
                            y = builder.CreateSelect(is2, builder.CreateFNeg(g[0]), y);
                        }
                        return {x, y, z};
                    };

                    auto curlTest = [&]() -> CodeValue {
                        if (is_plus != test_active_plus) return makeZero(vectorShape(3u));

                        auto* uses_vec_basis = builder.CreateICmpNE(side.test_uses_vector_basis, builder.getInt32(0));
                        auto* vb = llvm::BasicBlock::Create(*ctx, "curl.test.vec_basis", fn);
                        auto* sb = llvm::BasicBlock::Create(*ctx, "curl.test.scalar_basis", fn);
                        auto* merge = llvm::BasicBlock::Create(*ctx, "curl.test.merge", fn);

                        builder.CreateCondBr(uses_vec_basis, vb, sb);

                        std::array<llvm::Value*, 3> vb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
                        builder.SetInsertPoint(vb);
                        vb_vals = loadVec3FromTable(side.test_basis_curls_xyz, side.n_qpts, i_index, q_index);
                        builder.CreateBr(merge);
                        auto* vb_block = builder.GetInsertBlock();

                        std::array<llvm::Value*, 3> sb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
                        builder.SetInsertPoint(sb);
                        sb_vals = curlScalarBasis(side.n_test_dofs, i_index, side.test_phys_grads_xyz);
                        builder.CreateBr(merge);
                        auto* sb_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(merge);
                        std::array<llvm::Value*, 3> out{f64c(0.0), f64c(0.0), f64c(0.0)};
                        for (std::size_t d = 0; d < 3; ++d) {
                            auto* phi = builder.CreatePHI(f64, 2, "curl.test." + std::to_string(d));
                            phi->addIncoming(vb_vals[d], vb_block);
                            phi->addIncoming(sb_vals[d], sb_block);
                            out[d] = phi;
                        }
                        return makeVector(3u, out[0], out[1], out[2]);
                    };

                    auto curlTrial = [&]() -> CodeValue {
                        if (is_plus != trial_active_plus) return makeZero(vectorShape(3u));

                        auto* uses_vec_basis = builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
                        auto* vb = llvm::BasicBlock::Create(*ctx, "curl.trial.vec_basis", fn);
                        auto* sb = llvm::BasicBlock::Create(*ctx, "curl.trial.scalar_basis", fn);
                        auto* merge = llvm::BasicBlock::Create(*ctx, "curl.trial.merge", fn);

                        builder.CreateCondBr(uses_vec_basis, vb, sb);

                        std::array<llvm::Value*, 3> vb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
                        builder.SetInsertPoint(vb);
                        vb_vals = loadVec3FromTable(side.trial_basis_curls_xyz, side.n_qpts, j_index, q_index);
                        builder.CreateBr(merge);
                        auto* vb_block = builder.GetInsertBlock();

                        std::array<llvm::Value*, 3> sb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
                        builder.SetInsertPoint(sb);
                        sb_vals = curlScalarBasis(side.n_trial_dofs, j_index, side.trial_phys_grads_xyz);
                        builder.CreateBr(merge);
                        auto* sb_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(merge);
                        std::array<llvm::Value*, 3> out{f64c(0.0), f64c(0.0), f64c(0.0)};
                        for (std::size_t d = 0; d < 3; ++d) {
                            auto* phi = builder.CreatePHI(f64, 2, "curl.trial." + std::to_string(d));
                            phi->addIncoming(vb_vals[d], vb_block);
                            phi->addIncoming(sb_vals[d], sb_block);
                            out[d] = phi;
                        }
                        return makeVector(3u, out[0], out[1], out[2]);
                    };

                    auto curlCurrentSolution = [&]() -> CodeValue {
                        auto* coeffs = side.solution_coefficients;
                        auto* uses_vec_basis = builder.CreateICmpNE(side.trial_uses_vector_basis, builder.getInt32(0));
                        auto* vb = llvm::BasicBlock::Create(*ctx, "curl.u.vec_basis", fn);
                        auto* sb = llvm::BasicBlock::Create(*ctx, "curl.u.scalar_basis", fn);
                        auto* merge = llvm::BasicBlock::Create(*ctx, "curl.u.merge", fn);

                        builder.CreateCondBr(uses_vec_basis, vb, sb);

	                        std::array<llvm::Value*, 3> vb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
	                        builder.SetInsertPoint(vb);
	                        {
	                            const auto sums = emitReduceSum(side.n_trial_dofs, "curl_u_vb", 3u, [&](llvm::Value* j) {
	                                auto* j64 = builder.CreateZExt(j, i64);
	                                auto* cj = loadRealPtrAt(coeffs, j64);
	                                const auto phi =
	                                    loadVec3FromTable(side.trial_basis_curls_xyz, side.n_qpts, j, q_index);
	                                std::vector<llvm::Value*> terms;
	                                terms.reserve(3u);
	                                for (std::size_t d = 0; d < 3u; ++d) {
	                                    terms.push_back(builder.CreateFMul(cj, phi[d]));
	                                }
	                                return terms;
	                            });
	                            vb_vals = {sums[0], sums[1], sums[2]};
	                            builder.CreateBr(merge);
	                        }
	                        auto* vb_block = builder.GetInsertBlock();

	                        std::array<llvm::Value*, 3> sb_vals{f64c(0.0), f64c(0.0), f64c(0.0)};
	                        builder.SetInsertPoint(sb);
	                        {
	                            const auto sums = emitReduceSum(side.n_trial_dofs, "curl_u_sb", 3u, [&](llvm::Value* j) {
	                                auto* j64 = builder.CreateZExt(j, i64);
	                                auto* cj = loadRealPtrAt(coeffs, j64);
	                                const auto phi = curlScalarBasis(side.n_trial_dofs, j, side.trial_phys_grads_xyz);
	                                std::vector<llvm::Value*> terms;
	                                terms.reserve(3u);
	                                for (std::size_t d = 0; d < 3u; ++d) {
	                                    terms.push_back(builder.CreateFMul(cj, phi[d]));
	                                }
	                                return terms;
	                            });
	                            sb_vals = {sums[0], sums[1], sums[2]};
	                            builder.CreateBr(merge);
	                        }
	                        auto* sb_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(merge);
                        std::array<llvm::Value*, 3> out{f64c(0.0), f64c(0.0), f64c(0.0)};
                        for (std::size_t d = 0; d < 3; ++d) {
                            auto* phi = builder.CreatePHI(f64, 2, "curl.u." + std::to_string(d));
                            phi->addIncoming(vb_vals[d], vb_block);
                            phi->addIncoming(sb_vals[d], sb_block);
                            out[d] = phi;
                        }
                        return makeVector(3u, out[0], out[1], out[2]);
                    };

                    if (kid.type == FormExprType::TestFunction) {
                        return curlTest();
                    }
                    if (kid.type == FormExprType::TrialFunction) {
                        return is_residual ? curlCurrentSolution() : curlTrial();
                    }
                    if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                        const int fid = unpackFieldIdImm1(kid.imm1);
                        if (kid.type == FormExprType::StateField && fid == 0xffff) {
                            return curlCurrentSolution();
                        }
                        auto* entry = fieldEntryPtrFor(is_plus, fid);
                        auto* entry_is_null =
                            builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                        auto* ok = llvm::BasicBlock::Create(*ctx, "field_curl.ok", fn);
                        auto* zero = llvm::BasicBlock::Create(*ctx, "field_curl.zero", fn);
                        auto* merge = llvm::BasicBlock::Create(*ctx, "field_curl.merge", fn);

                        builder.CreateCondBr(entry_is_null, zero, ok);

                        std::array<llvm::Value*, 3> c{f64c(0.0), f64c(0.0), f64c(0.0)};
                        builder.SetInsertPoint(ok);
                        auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                        auto* base_is_null =
                            builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                        auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_curl.mat.ok", fn);
                        builder.CreateCondBr(base_is_null, zero, ok2);

                        builder.SetInsertPoint(ok2);
                        const auto J = loadMat3FromQ(base, q_index);
                        auto d = [&](std::size_t comp, std::size_t wrt) -> llvm::Value* {
                            if (comp >= vd || comp >= 3u || wrt >= 3u) return f64c(0.0);
                            return J.elems[comp * 3 + wrt];
                        };
                        c[0] = builder.CreateFSub(d(2, 1), d(1, 2));
                        c[1] = builder.CreateFSub(d(0, 2), d(2, 0));
                        c[2] = builder.CreateFSub(d(1, 0), d(0, 1));
                        builder.CreateBr(merge);
                        auto* ok2_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(zero);
                        builder.CreateBr(merge);
                        auto* zero_block = builder.GetInsertBlock();

                        builder.SetInsertPoint(merge);
                        std::array<llvm::Value*, 3> out{f64c(0.0), f64c(0.0), f64c(0.0)};
                        for (std::size_t d0 = 0; d0 < 3; ++d0) {
                            auto* phi = builder.CreatePHI(f64, 2, "field_curl." + std::to_string(d0));
                            phi->addIncoming(f64c(0.0), zero_block);
                            phi->addIncoming(c[d0], ok2_block);
                            out[d0] = phi;
                        }
                        return makeVector(3u, out[0], out[1], out[2]);
                    }
                    if (kid.type == FormExprType::PreviousSolutionRef) {
                        const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
                        auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
                        auto* dofs_per_comp =
                            builder.CreateUDiv(side.n_trial_dofs, builder.getInt32(static_cast<std::uint32_t>(vd)));

                        std::array<std::array<llvm::Value*, 3>, 3> J{};
                        for (auto& row : J) {
                            row = {f64c(0.0), f64c(0.0), f64c(0.0)};
                        }

	                        for (std::size_t comp = 0; comp < std::min<std::size_t>(vd, 3u); ++comp) {
	                            const auto acc =
	                                emitReduceSum(dofs_per_comp,
	                                             "prev_curl" + std::to_string(k) + "_c" + std::to_string(comp),
	                                             3u,
	                                             [&](llvm::Value* jj) {
	                                                 auto* base = builder.CreateMul(builder.getInt32(static_cast<std::uint32_t>(comp)), dofs_per_comp);
	                                                 auto* j = builder.CreateAdd(base, jj);
	                                                 auto* j64 = builder.CreateZExt(j, i64);
	                                                 auto* cj = loadRealPtrAt(coeffs, j64);
	                                                 const auto g =
	                                                     loadVec3FromTableQMajor(side.trial_phys_grads_xyz, side.n_trial_dofs, j, q_index);
	                                                 std::vector<llvm::Value*> terms;
	                                                 terms.reserve(3u);
	                                                 for (std::size_t d = 0; d < 3u; ++d) {
	                                                     terms.push_back(builder.CreateFMul(cj, g[d]));
	                                                 }
	                                                 return terms;
	                                             });
	                            J[comp] = {acc[0], acc[1], acc[2]};
	                        }

                        auto* cx = builder.CreateFSub(J[2][1], J[1][2]);
                        auto* cy = builder.CreateFSub(J[0][2], J[2][0]);
                        auto* cz = builder.CreateFSub(J[1][0], J[0][1]);
                        return makeVector(3u, cx, cy, cz);
                    }
                    if (kid.type == FormExprType::TimeDerivative) {
                        const int order = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
                        auto* coeff0 = loadDtCoeff0(side, order);
                        const auto dt_child_idx =
                            term.ir.children[static_cast<std::size_t>(kid.first_child)];
                        const auto& dt_child = term.ir.ops[dt_child_idx];
                        if (dt_child.type == FormExprType::TrialFunction) {
                            return mul(makeScalar(coeff0), is_residual ? curlCurrentSolution() : curlTrial());
                        }

                        if (dt_child.type == FormExprType::DiscreteField || dt_child.type == FormExprType::StateField) {
                            const int fid = unpackFieldIdImm1(dt_child.imm1);
                            if (dt_child.type == FormExprType::StateField && fid == 0xffff) {
                                return mul(makeScalar(coeff0), curlCurrentSolution());
                            }

                            // Reuse the standard curl(field) lowering and scale by coeff0.
                            auto* entry = fieldEntryPtrFor(is_plus, fid);
                            auto* entry_is_null =
                                builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok = llvm::BasicBlock::Create(*ctx, "field_dt_curl.ok", fn);
                            auto* zero = llvm::BasicBlock::Create(*ctx, "field_dt_curl.zero", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "field_dt_curl.merge", fn);

                            builder.CreateCondBr(entry_is_null, zero, ok);

                            std::array<llvm::Value*, 3> c{f64c(0.0), f64c(0.0), f64c(0.0)};
                            builder.SetInsertPoint(ok);
                            auto* base = loadPtr(entry, ABIV3::field_entry_jacobians_off);
                            auto* base_is_null =
                                builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_dt_curl.mat.ok", fn);
                            builder.CreateCondBr(base_is_null, zero, ok2);

                            builder.SetInsertPoint(ok2);
                            const auto J = loadMat3FromQ(base, q_index);
                            auto d = [&](std::size_t comp, std::size_t wrt) -> llvm::Value* {
                                if (comp >= vd || comp >= 3u || wrt >= 3u) return f64c(0.0);
                                return J.elems[comp * 3 + wrt];
                            };
                            c[0] = builder.CreateFSub(d(2, 1), d(1, 2));
                            c[1] = builder.CreateFSub(d(0, 2), d(2, 0));
                            c[2] = builder.CreateFSub(d(1, 0), d(0, 1));
                            for (auto& e : c) {
                                e = builder.CreateFMul(coeff0, e);
                            }
                            builder.CreateBr(merge);
                            auto* ok2_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(zero);
                            builder.CreateBr(merge);
                            auto* zero_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(merge);
                            std::array<llvm::Value*, 3> out{f64c(0.0), f64c(0.0), f64c(0.0)};
                            for (std::size_t d0 = 0; d0 < 3; ++d0) {
                                auto* phi = builder.CreatePHI(f64, 2, "field_dt_curl." + std::to_string(d0));
                                phi->addIncoming(f64c(0.0), zero_block);
                                phi->addIncoming(c[d0], ok2_block);
                                out[d0] = phi;
                            }
                            return makeVector(3u, out[0], out[1], out[2]);
                        }

                        if (dt_child.type == FormExprType::Constant) {
                            return makeZero(vectorShape(3u));
                        }

                        throw std::runtime_error("LLVMGen: curl(dt(...)) operand not supported");
                    }
                    if (kid.type == FormExprType::Constant) {
                        return makeZero(vectorShape(3u));
                    }
                    throw std::runtime_error("LLVMGen: Curl operand not supported");
                };

	                auto evalHessian = [&](const SideView& side, bool is_plus, const KernelIROp& op, const Shape& shape) -> CodeValue {
	                    const auto child_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
	                    const auto& kid = term.ir.ops[child_idx];
		                    const auto dim_u32 = shape.dims[0];
		                    const auto mat_len = elemCount(shape);
		                    auto matZero = [&]() -> CodeValue { return makeZero(shape); };

		                    auto hessFromCoeffs = [&](llvm::Value* coeffs_ptr, const std::string& loop_name) -> CodeValue {
		                        CodeValue out = makeZero(shape);
		                        const auto sums =
		                            emitReduceSum(side.n_trial_dofs, loop_name, mat_len, [&](llvm::Value* j) {
		                                auto* j64 = builder.CreateZExt(j, i64);
		                                auto* cj = loadRealPtrAt(coeffs_ptr, j64);
		                                const auto H =
		                                    loadMatDimFromTable(side.trial_phys_hessians, side.n_qpts, j, q_index, dim_u32);
		                                std::vector<llvm::Value*> terms;
		                                terms.reserve(mat_len);
		                                for (std::size_t i = 0; i < mat_len; ++i) {
		                                    terms.push_back(builder.CreateFMul(cj, H.elems[i]));
		                                }
		                                return terms;
		                            });
		                        for (std::size_t i = 0; i < mat_len; ++i) {
		                            out.elems[i] = sums[i];
		                        }
		                        return out;
		                    };

		                    auto hessCurrentSolution = [&]() -> CodeValue {
		                        auto* coeffs = side.solution_coefficients;
		                        return hessFromCoeffs(coeffs, "hess_u");
		                    };

	                    if (kid.type == FormExprType::TestFunction) {
	                        if (is_plus != test_active_plus) return matZero();
	                        return loadMatDimFromTable(side.test_phys_hessians, side.n_qpts, i_index, q_index, dim_u32);
	                    }
	                    if (kid.type == FormExprType::TrialFunction) {
	                        if (is_residual) {
	                            return hessCurrentSolution();
	                        }
	                        if (is_plus != trial_active_plus) return matZero();
	                        return loadMatDimFromTable(side.trial_phys_hessians, side.n_qpts, j_index, q_index, dim_u32);
	                    }
		                    if (kid.type == FormExprType::PreviousSolutionRef) {
		                        const int k = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
		                        auto* coeffs = loadPrevSolutionCoeffsPtr(side, k);
		                        return hessFromCoeffs(coeffs, "prev_hess" + std::to_string(k));
		                    }
	                    if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
	                        const int fid = unpackFieldIdImm1(kid.imm1);
	                        if (kid.type == FormExprType::StateField && fid == 0xffff) {
	                            return hessCurrentSolution();
	                        }
	                        auto* entry = fieldEntryPtrFor(is_plus, fid);
	                        auto* entry_is_null =
	                            builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
	                        auto* ok = llvm::BasicBlock::Create(*ctx, "field_hess.ok", fn);
	                        auto* z = llvm::BasicBlock::Create(*ctx, "field_hess.zero", fn);
	                        auto* merge = llvm::BasicBlock::Create(*ctx, "field_hess.merge", fn);

	                        builder.CreateCondBr(entry_is_null, z, ok);

	                        CodeValue loaded = matZero();
	                        builder.SetInsertPoint(ok);
	                        auto* base = loadPtr(entry, ABIV3::field_entry_hessians_off);
	                        auto* base_is_null =
	                            builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
	                        auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_hess.mat.ok", fn);
	                        builder.CreateCondBr(base_is_null, z, ok2);

	                        builder.SetInsertPoint(ok2);
	                        loaded = loadMatDimFromQ(base, q_index, dim_u32);
	                        builder.CreateBr(merge);
	                        auto* ok2_block = builder.GetInsertBlock();

	                        builder.SetInsertPoint(z);
	                        builder.CreateBr(merge);
	                        auto* zero_block = builder.GetInsertBlock();

	                        builder.SetInsertPoint(merge);
	                        CodeValue out = matZero();
	                        for (std::size_t i = 0; i < mat_len; ++i) {
	                            auto* phi = builder.CreatePHI(f64, 2, "field_hess." + std::to_string(i));
	                            phi->addIncoming(f64c(0.0), zero_block);
	                            phi->addIncoming(loaded.elems[i], ok2_block);
	                            out.elems[i] = phi;
	                        }
	                        return out;
	                    }
	                    if (kid.type == FormExprType::TimeDerivative) {
	                        const int order = static_cast<int>(static_cast<std::int64_t>(kid.imm0));
	                        auto* coeff0 = loadDtCoeff0(side, order);
	                        const auto dt_child_idx =
	                            term.ir.children[static_cast<std::size_t>(kid.first_child)];
	                        const auto& dt_child = term.ir.ops[dt_child_idx];
	                        if (dt_child.type == FormExprType::TrialFunction) {
	                            if (is_residual) {
	                                return mul(makeScalar(coeff0), hessCurrentSolution());
	                            }
	                            if (is_plus != trial_active_plus) return matZero();
	                            return mul(makeScalar(coeff0),
	                                       loadMatDimFromTable(side.trial_phys_hessians, side.n_qpts, j_index, q_index, dim_u32));
	                        }

                        if (dt_child.type == FormExprType::DiscreteField || dt_child.type == FormExprType::StateField) {
                            const int fid = unpackFieldIdImm1(dt_child.imm1);
                            if (dt_child.type == FormExprType::StateField && fid == 0xffff) {
                                return mul(makeScalar(coeff0), hessCurrentSolution());
                            }

                            auto* entry = fieldEntryPtrFor(is_plus, fid);
                            auto* entry_is_null =
                                builder.CreateICmpEQ(entry, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok = llvm::BasicBlock::Create(*ctx, "field_dt_hess.ok", fn);
                            auto* z = llvm::BasicBlock::Create(*ctx, "field_dt_hess.zero", fn);
                            auto* merge = llvm::BasicBlock::Create(*ctx, "field_dt_hess.merge", fn);

	                            builder.CreateCondBr(entry_is_null, z, ok);

	                            CodeValue loaded = matZero();
	                            builder.SetInsertPoint(ok);
	                            auto* base = loadPtr(entry, ABIV3::field_entry_hessians_off);
	                            auto* base_is_null =
	                                builder.CreateICmpEQ(base, llvm::ConstantPointerNull::get(i8_ptr));
                            auto* ok2 = llvm::BasicBlock::Create(*ctx, "field_dt_hess.mat.ok", fn);
                            builder.CreateCondBr(base_is_null, z, ok2);

	                            builder.SetInsertPoint(ok2);
	                            loaded = loadMatDimFromQ(base, q_index, dim_u32);
	                            builder.CreateBr(merge);
	                            auto* ok2_block = builder.GetInsertBlock();

                            builder.SetInsertPoint(z);
                            builder.CreateBr(merge);
                            auto* zero_block = builder.GetInsertBlock();

	                            builder.SetInsertPoint(merge);
	                            CodeValue out = matZero();
	                            for (std::size_t i = 0; i < mat_len; ++i) {
	                                auto* phi = builder.CreatePHI(f64, 2, "field_dt_hess." + std::to_string(i));
	                                phi->addIncoming(f64c(0.0), zero_block);
	                                phi->addIncoming(loaded.elems[i], ok2_block);
	                                out.elems[i] = phi;
	                            }
	                            return mul(makeScalar(coeff0), out);
	                        }

	                        if (dt_child.type == FormExprType::Constant) {
	                            return matZero();
	                        }

                        throw std::runtime_error("LLVMGen: H(dt(...)) operand not supported");
	                    }
	                    if (kid.type == FormExprType::Constant) {
	                        return matZero();
	                    }
	                    throw std::runtime_error("LLVMGen: Hessian operand not supported");
	                };

                for (std::size_t op_idx = 0; op_idx < term.ir.ops.size(); ++op_idx) {
                    if (cached != nullptr && term.dep_mask[op_idx] == 0u) {
                        values_minus[op_idx] = cached->minus[op_idx];
                        values_plus[op_idx] = cached->plus[op_idx];
                        continue;
                    }
                    const auto& op = term.ir.ops[op_idx];
                    const auto& shape = term.shapes[op_idx];

                    switch (op.type) {
                        case FormExprType::Constant: {
                            const double v = std::bit_cast<double>(op.imm0);
                            const auto cv = makeScalar(f64c(v));
                            values_minus[op_idx] = cv;
                            values_plus[op_idx] = cv;
                            break;
                        }

                        case FormExprType::ParameterRef: {
                            auto* idx64 = builder.getInt64(op.imm0);
                            values_minus[op_idx] = makeScalar(loadRealPtrAt(side_minus.jit_constants, idx64));
                            values_plus[op_idx] = makeScalar(loadRealPtrAt(side_plus.jit_constants, idx64));
                            break;
                        }

                        case FormExprType::BoundaryIntegralRef: {
                            auto* idx64 = builder.getInt64(op.imm0);
                            values_minus[op_idx] = makeScalar(loadRealPtrAt(side_minus.coupled_integrals, idx64));
                            values_plus[op_idx] = makeScalar(loadRealPtrAt(side_plus.coupled_integrals, idx64));
                            break;
                        }

                        case FormExprType::AuxiliaryStateRef: {
                            auto* idx64 = builder.getInt64(op.imm0);
                            values_minus[op_idx] = makeScalar(loadRealPtrAt(side_minus.coupled_aux, idx64));
                            values_plus[op_idx] = makeScalar(loadRealPtrAt(side_plus.coupled_aux, idx64));
                            break;
                        }

                        case FormExprType::DiscreteField:
                        case FormExprType::StateField: {
                            const int fid = unpackFieldIdImm1(op.imm1);
                            if (op.type == FormExprType::StateField && fid == 0xffff) {
                                values_minus[op_idx] = evalCurrentSolution(side_minus, shape, q_index);
                                values_plus[op_idx] = evalCurrentSolution(side_plus, shape, q_index);
                            } else {
                                values_minus[op_idx] = evalDiscreteOrStateField(/*plus_side=*/false, shape, fid, q_index);
                                values_plus[op_idx] = evalDiscreteOrStateField(/*plus_side=*/true, shape, fid, q_index);
                            }
                            break;
                        }

                        case FormExprType::Coefficient: {
                            values_minus[op_idx] = evalExternalCoefficient(side_minus, q_index, shape, op.imm0);
                            values_plus[op_idx] = evalExternalCoefficient(side_plus, q_index, shape, op.imm0);
                            break;
                        }

                        case FormExprType::Constitutive: {
                            values_minus[op_idx] = evalExternalConstitutiveOutput(
                                side_minus,
                                q_index,
                                builder.getInt32(static_cast<std::uint32_t>(external::TraceSideV1::Minus)),
                                op.imm0,
                                /*output_index=*/0u,
                                term.ir,
                                op,
                                values_minus,
                                shape);
                            values_plus[op_idx] = evalExternalConstitutiveOutput(
                                side_plus,
                                q_index,
                                builder.getInt32(static_cast<std::uint32_t>(external::TraceSideV1::Plus)),
                                op.imm0,
                                /*output_index=*/0u,
                                term.ir,
                                op,
                                values_plus,
                                shape);
                            break;
                        }

                        case FormExprType::ConstitutiveOutput: {
                            if (op.child_count != 1u) {
                                throw std::runtime_error("LLVMGen: face ConstitutiveOutput expects exactly 1 child");
                            }

                            const auto out_idx_i64 = static_cast<std::int64_t>(op.imm0);
                            if (out_idx_i64 < 0) {
                                throw std::runtime_error("LLVMGen: face ConstitutiveOutput has negative output index");
                            }

                            const auto child_call_idx =
                                term.ir.children[static_cast<std::size_t>(op.first_child)];

                            if (out_idx_i64 == 0) {
                                values_minus[op_idx] = values_minus[child_call_idx];
                                values_plus[op_idx] = values_plus[child_call_idx];
                                break;
                            }

                            const auto& call_op = term.ir.ops[child_call_idx];
                            if (call_op.type != FormExprType::Constitutive) {
                                throw std::runtime_error("LLVMGen: face ConstitutiveOutput child must be Constitutive");
                            }

                            values_minus[op_idx] = evalExternalConstitutiveOutput(
                                side_minus,
                                q_index,
                                builder.getInt32(static_cast<std::uint32_t>(external::TraceSideV1::Minus)),
                                call_op.imm0,
                                static_cast<std::uint32_t>(out_idx_i64),
                                term.ir,
                                call_op,
                                values_minus,
                                shape);
                            values_plus[op_idx] = evalExternalConstitutiveOutput(
                                side_plus,
                                q_index,
                                builder.getInt32(static_cast<std::uint32_t>(external::TraceSideV1::Plus)),
                                call_op.imm0,
                                static_cast<std::uint32_t>(out_idx_i64),
                                term.ir,
                                call_op,
                                values_plus,
                                shape);
                            break;
                        }

                        case FormExprType::Time:
                            values_minus[op_idx] = makeScalar(side_minus.time);
                            values_plus[op_idx] = makeScalar(side_plus.time);
                            break;

                        case FormExprType::TimeStep:
                            values_minus[op_idx] = makeScalar(side_minus.dt);
                            values_plus[op_idx] = makeScalar(side_plus.dt);
                            break;

                        case FormExprType::EffectiveTimeStep:
                            values_minus[op_idx] = makeScalar(loadEffectiveDt(side_minus));
                            values_plus[op_idx] = makeScalar(loadEffectiveDt(side_plus));
                            break;

                        case FormExprType::CellDiameter:
                            values_minus[op_idx] = makeScalar(side_minus.cell_diameter);
                            values_plus[op_idx] = makeScalar(side_plus.cell_diameter);
                            break;

                        case FormExprType::CellVolume:
                            values_minus[op_idx] = makeScalar(side_minus.cell_volume);
                            values_plus[op_idx] = makeScalar(side_plus.cell_volume);
                            break;

                        case FormExprType::FacetArea:
                            values_minus[op_idx] = makeScalar(side_minus.facet_area);
                            values_plus[op_idx] = makeScalar(side_plus.facet_area);
                            break;

                        case FormExprType::CellDomainId:
                            values_minus[op_idx] = makeScalar(builder.CreateSIToFP(side_minus.cell_domain_id, f64));
                            values_plus[op_idx] = makeScalar(builder.CreateSIToFP(side_plus.cell_domain_id, f64));
                            break;

	                        case FormExprType::Coordinate:
	                            values_minus[op_idx] =
	                                loadXYZDimFromSide(side_minus, side_minus.physical_points_xyz, q_index, shape.dims[0],
	                                                   side_minus.interleaved_qpoint_geometry_physical_offset);
	                            values_plus[op_idx] =
	                                loadXYZDimFromSide(side_plus, side_plus.physical_points_xyz, q_index, shape.dims[0],
	                                                   side_plus.interleaved_qpoint_geometry_physical_offset);
	                            break;

	                        case FormExprType::ReferenceCoordinate:
	                            values_minus[op_idx] = loadXYZDim(side_minus.quad_points_xyz, q_index, shape.dims[0]);
	                            values_plus[op_idx] = loadXYZDim(side_plus.quad_points_xyz, q_index, shape.dims[0]);
	                            break;

	                        case FormExprType::Normal:
	                            values_minus[op_idx] =
	                                loadXYZDimFromSide(side_minus, side_minus.normals_xyz, q_index, shape.dims[0],
	                                                   side_minus.interleaved_qpoint_geometry_normal_offset);
	                            values_plus[op_idx] =
	                                loadXYZDimFromSide(side_plus, side_plus.normals_xyz, q_index, shape.dims[0],
	                                                   side_plus.interleaved_qpoint_geometry_normal_offset);
	                            break;

	                        case FormExprType::Jacobian:
	                            values_minus[op_idx] =
	                                loadMatDimFromSide(side_minus, side_minus.jacobians, q_index, shape.dims[0],
	                                                   side_minus.interleaved_qpoint_geometry_jacobian_offset);
	                            values_plus[op_idx] =
	                                loadMatDimFromSide(side_plus, side_plus.jacobians, q_index, shape.dims[0],
	                                                   side_plus.interleaved_qpoint_geometry_jacobian_offset);
	                            break;

	                        case FormExprType::JacobianInverse:
	                            values_minus[op_idx] =
	                                loadMatDimFromSide(side_minus, side_minus.inverse_jacobians, q_index, shape.dims[0],
	                                                   side_minus.interleaved_qpoint_geometry_inverse_jacobian_offset);
	                            values_plus[op_idx] =
	                                loadMatDimFromSide(side_plus, side_plus.inverse_jacobians, q_index, shape.dims[0],
	                                                   side_plus.interleaved_qpoint_geometry_inverse_jacobian_offset);
	                            break;

                        case FormExprType::Identity: {
                            CodeValue out = makeZero(shape);
                            const auto n = static_cast<std::size_t>(shape.dims[0]);
                            for (std::size_t d = 0; d < n; ++d) {
                                out.elems[d * n + d] = f64c(1.0);
                            }
                            values_minus[op_idx] = out;
                            values_plus[op_idx] = out;
                            break;
                        }

                        case FormExprType::JacobianDeterminant: {
                            values_minus[op_idx] =
                                makeScalar(loadScalarFromSide(side_minus, side_minus.jacobian_dets, q_index,
                                                              side_minus.interleaved_qpoint_geometry_det_offset));
                            values_plus[op_idx] =
                                makeScalar(loadScalarFromSide(side_plus, side_plus.jacobian_dets, q_index,
                                                              side_plus.interleaved_qpoint_geometry_det_offset));
                            break;
                        }

                        case FormExprType::TestFunction:
                            values_minus[op_idx] = evalTestFunction(side_minus, /*is_plus=*/false, shape);
                            values_plus[op_idx] = evalTestFunction(side_plus, /*is_plus=*/true, shape);
                            break;

                        case FormExprType::TrialFunction:
                            values_minus[op_idx] = evalTrialFunction(side_minus, /*is_plus=*/false, shape);
                            values_plus[op_idx] = evalTrialFunction(side_plus, /*is_plus=*/true, shape);
                            break;

                        case FormExprType::PreviousSolutionRef: {
                            const int k = static_cast<int>(static_cast<std::int64_t>(op.imm0));
                            values_minus[op_idx] = evalPreviousSolution(side_minus, shape, k, q_index);
                            values_plus[op_idx] = evalPreviousSolution(side_plus, shape, k, q_index);
                            break;
                        }

                        case FormExprType::Gradient:
                            values_minus[op_idx] = evalGradient(side_minus, /*is_plus=*/false, op, shape);
                            values_plus[op_idx] = evalGradient(side_plus, /*is_plus=*/true, op, shape);
                            break;

                        case FormExprType::Divergence:
                            values_minus[op_idx] = evalDivergence(side_minus, /*is_plus=*/false, op);
                            values_plus[op_idx] = evalDivergence(side_plus, /*is_plus=*/true, op);
                            break;

                        case FormExprType::Curl:
                            values_minus[op_idx] = evalCurl(side_minus, /*is_plus=*/false, op);
                            values_plus[op_idx] = evalCurl(side_plus, /*is_plus=*/true, op);
                            break;

	                        case FormExprType::Hessian:
	                            values_minus[op_idx] = evalHessian(side_minus, /*is_plus=*/false, op, shape);
	                            values_plus[op_idx] = evalHessian(side_plus, /*is_plus=*/true, op, shape);
	                            break;

                        case FormExprType::TimeDerivative: {
                            const int order = static_cast<int>(static_cast<std::int64_t>(op.imm0));
                            CodeValue acc_minus = mul(makeScalar(loadDtCoeff(side_minus, order, 0)), childMinus(op, 0));
                            CodeValue acc_plus = mul(makeScalar(loadDtCoeff(side_plus, order, 0)), childPlus(op, 0));

                            const auto child_op_idx = term.ir.children[static_cast<std::size_t>(op.first_child)];
                            const auto& kid = term.ir.ops[child_op_idx];
                            if (kid.type == FormExprType::DiscreteField || kid.type == FormExprType::StateField) {
                                const int fid = unpackFieldIdImm1(kid.imm1);
                                if (kid.type == FormExprType::StateField && fid == 0xffff) {
                                    for (int k = 1; k <= static_cast<int>(assembly::jit::kMaxPreviousSolutionsV6); ++k) {
                                        acc_minus = add(acc_minus,
                                                        mul(makeScalar(loadDtCoeff(side_minus, order, k)),
                                                            evalPreviousSolution(side_minus, shape, k, q_index)));
                                        acc_plus = add(acc_plus,
                                                       mul(makeScalar(loadDtCoeff(side_plus, order, k)),
                                                           evalPreviousSolution(side_plus, shape, k, q_index)));
                                    }
                                } else {
                                    for (int k = 1; k <= static_cast<int>(assembly::jit::kMaxPreviousSolutionsV6); ++k) {
                                        acc_minus = add(acc_minus,
                                                        mul(makeScalar(loadDtCoeff(side_minus, order, k)),
                                                            evalDiscreteOrStateFieldHistoryK(side_minus, /*plus_side=*/false, shape, fid, k, q_index)));
                                        acc_plus = add(acc_plus,
                                                       mul(makeScalar(loadDtCoeff(side_plus, order, k)),
                                                           evalDiscreteOrStateFieldHistoryK(side_plus, /*plus_side=*/true, shape, fid, k, q_index)));
                                    }
                                }
                            }

                            values_minus[op_idx] = acc_minus;
                            values_plus[op_idx] = acc_plus;
                            break;
                        }

                        case FormExprType::RestrictMinus: {
                            const auto child = childIndex(op, 0);
                            values_minus[op_idx] = values_minus[child];
                            values_plus[op_idx] = values_minus[child];
                            break;
                        }

                        case FormExprType::RestrictPlus: {
                            const auto child = childIndex(op, 0);
                            values_minus[op_idx] = values_plus[child];
                            values_plus[op_idx] = values_plus[child];
                            break;
                        }

                        case FormExprType::Jump: {
                            const auto child = childIndex(op, 0);
                            const auto out = sub(values_minus[child], values_plus[child]);
                            values_minus[op_idx] = out;
                            values_plus[op_idx] = out;
                            break;
                        }

                        case FormExprType::Average: {
                            const auto child = childIndex(op, 0);
                            const auto out = mul(makeScalar(f64c(0.5)), add(values_minus[child], values_plus[child]));
                            values_minus[op_idx] = out;
                            values_plus[op_idx] = out;
                            break;
                        }

                        case FormExprType::MaterialStateOldRef:
                            values_minus[op_idx] = makeScalar(loadMaterialStateReal(side_minus.material_state_old_base,
                                                                                   side_minus.material_state_stride_bytes,
                                                                                   q_index,
                                                                                   static_cast<std::uint64_t>(op.imm0)));
                            values_plus[op_idx] = makeScalar(loadMaterialStateReal(side_plus.material_state_old_base,
                                                                                  side_plus.material_state_stride_bytes,
                                                                                  q_index,
                                                                                  static_cast<std::uint64_t>(op.imm0)));
                            break;

                        case FormExprType::MaterialStateWorkRef:
                            values_minus[op_idx] = makeScalar(loadMaterialStateReal(side_minus.material_state_work_base,
                                                                                   side_minus.material_state_stride_bytes,
                                                                                   q_index,
                                                                                   static_cast<std::uint64_t>(op.imm0)));
                            values_plus[op_idx] = makeScalar(loadMaterialStateReal(side_plus.material_state_work_base,
                                                                                  side_plus.material_state_stride_bytes,
                                                                                  q_index,
                                                                                  static_cast<std::uint64_t>(op.imm0)));
                            break;

                        case FormExprType::Negate:
                            values_minus[op_idx] = neg(childMinus(op, 0));
                            values_plus[op_idx] = neg(childPlus(op, 0));
                            break;

	                        case FormExprType::Add: {
	                            if (shape.kind == Shape::Kind::Scalar) {
	                                if (auto* fused = tryFuseMulAddScalar(op.type, term, use_counts, op, values_minus)) {
	                                    values_minus[op_idx] = makeScalar(fused);
	                                } else {
	                                    values_minus[op_idx] = add(childMinus(op, 0), childMinus(op, 1));
	                                }
	                                if (auto* fused = tryFuseMulAddScalar(op.type, term, use_counts, op, values_plus)) {
	                                    values_plus[op_idx] = makeScalar(fused);
	                                } else {
	                                    values_plus[op_idx] = add(childPlus(op, 0), childPlus(op, 1));
	                                }
	                                break;
	                            }
	                            values_minus[op_idx] = add(childMinus(op, 0), childMinus(op, 1));
	                            values_plus[op_idx] = add(childPlus(op, 0), childPlus(op, 1));
	                            break;
	                        }

	                        case FormExprType::Subtract: {
	                            if (shape.kind == Shape::Kind::Scalar) {
	                                if (auto* fused = tryFuseMulAddScalar(op.type, term, use_counts, op, values_minus)) {
	                                    values_minus[op_idx] = makeScalar(fused);
	                                } else {
	                                    values_minus[op_idx] = sub(childMinus(op, 0), childMinus(op, 1));
	                                }
	                                if (auto* fused = tryFuseMulAddScalar(op.type, term, use_counts, op, values_plus)) {
	                                    values_plus[op_idx] = makeScalar(fused);
	                                } else {
	                                    values_plus[op_idx] = sub(childPlus(op, 0), childPlus(op, 1));
	                                }
	                                break;
	                            }
	                            values_minus[op_idx] = sub(childMinus(op, 0), childMinus(op, 1));
	                            values_plus[op_idx] = sub(childPlus(op, 0), childPlus(op, 1));
	                            break;
	                        }

                        case FormExprType::Multiply:
                            values_minus[op_idx] = mul(childMinus(op, 0), childMinus(op, 1));
                            values_plus[op_idx] = mul(childPlus(op, 0), childPlus(op, 1));
                            break;

                        case FormExprType::Divide:
                            values_minus[op_idx] = div(childMinus(op, 0), childMinus(op, 1));
                            values_plus[op_idx] = div(childPlus(op, 0), childPlus(op, 1));
                            break;

                        case FormExprType::InnerProduct:
                            values_minus[op_idx] = inner(childMinus(op, 0), childMinus(op, 1));
                            values_plus[op_idx] = inner(childPlus(op, 0), childPlus(op, 1));
                            break;

                        case FormExprType::DoubleContraction:
                            values_minus[op_idx] = doubleContraction(childMinus(op, 0), childMinus(op, 1));
                            values_plus[op_idx] = doubleContraction(childPlus(op, 0), childPlus(op, 1));
                            break;

                        case FormExprType::OuterProduct:
                            values_minus[op_idx] = outer(childMinus(op, 0), childMinus(op, 1));
                            values_plus[op_idx] = outer(childPlus(op, 0), childPlus(op, 1));
                            break;

                        case FormExprType::CrossProduct:
                            values_minus[op_idx] = cross(childMinus(op, 0), childMinus(op, 1));
                            values_plus[op_idx] = cross(childPlus(op, 0), childPlus(op, 1));
                            break;

                        case FormExprType::Power:
                            values_minus[op_idx] = powv(childMinus(op, 0), childMinus(op, 1));
                            values_plus[op_idx] = powv(childPlus(op, 0), childPlus(op, 1));
                            break;

                        case FormExprType::Minimum:
                            values_minus[op_idx] = makeScalar(f_min(childMinus(op, 0).elems[0], childMinus(op, 1).elems[0]));
                            values_plus[op_idx] = makeScalar(f_min(childPlus(op, 0).elems[0], childPlus(op, 1).elems[0]));
                            break;

                        case FormExprType::Maximum:
                            values_minus[op_idx] = makeScalar(f_max(childMinus(op, 0).elems[0], childMinus(op, 1).elems[0]));
                            values_plus[op_idx] = makeScalar(f_max(childPlus(op, 0).elems[0], childPlus(op, 1).elems[0]));
                            break;

                        case FormExprType::Less:
                        case FormExprType::LessEqual:
                        case FormExprType::Greater:
                        case FormExprType::GreaterEqual:
                        case FormExprType::Equal:
                        case FormExprType::NotEqual:
                            values_minus[op_idx] = cmp(op.type, childMinus(op, 0), childMinus(op, 1));
                            values_plus[op_idx] = cmp(op.type, childPlus(op, 0), childPlus(op, 1));
                            break;

                        case FormExprType::Conditional:
                            values_minus[op_idx] = conditional(childMinus(op, 0), childMinus(op, 1), childMinus(op, 2));
                            values_plus[op_idx] = conditional(childPlus(op, 0), childPlus(op, 1), childPlus(op, 2));
                            break;

                        case FormExprType::AsVector: {
                            CodeValue out_m = makeZero(shape);
                            CodeValue out_p = makeZero(shape);
                            for (std::size_t k = 0; k < op.child_count; ++k) {
                                out_m.elems[k] = childMinus(op, k).elems[0];
                                out_p.elems[k] = childPlus(op, k).elems[0];
                            }
                            values_minus[op_idx] = out_m;
                            values_plus[op_idx] = out_p;
                            break;
                        }

                        case FormExprType::AsTensor: {
                            const auto rows = unpackU32Lo(op.imm0);
                            const auto cols = unpackU32Hi(op.imm0);
                            CodeValue out_m = makeMatrix(rows, cols);
                            CodeValue out_p = makeMatrix(rows, cols);
                            for (std::size_t r = 0; r < rows; ++r) {
                                for (std::size_t c = 0; c < cols; ++c) {
                                    const auto k = r * cols + c;
                                    out_m.elems[k] = childMinus(op, k).elems[0];
                                    out_p.elems[k] = childPlus(op, k).elems[0];
                                }
                            }
                            values_minus[op_idx] = out_m;
                            values_plus[op_idx] = out_p;
                            break;
                        }

                        case FormExprType::IndexedAccess: {
                            if (op.child_count != 1u) {
                                throw std::runtime_error("LLVMGen: IndexedAccess expects exactly 1 child");
                            }
                            if (indexed_env == nullptr) {
                                throw std::runtime_error("LLVMGen: IndexedAccess requires an index environment (Einstein-sum lowering)");
                            }

                            const auto& am = childMinus(op, 0);
                            const auto& ap = childPlus(op, 0);
                            if (am.shape.kind == Shape::Kind::Scalar) {
                                values_minus[op_idx] = am;
                                values_plus[op_idx] = ap;
                                break;
                            }

                            const int rank = static_cast<int>(unpackIndexedRank(op.imm1));
                            if (rank <= 0 || rank > 4) {
                                throw std::runtime_error("LLVMGen: IndexedAccess has invalid rank");
                            }

                            auto loadIndex = [&](int k) -> llvm::Value* {
                                const std::size_t id = static_cast<std::size_t>(unpackIndexedId(op.imm0, k));
                                if (id >= indexed_env->size() || (*indexed_env)[id] == nullptr) {
                                    throw std::runtime_error("LLVMGen: IndexedAccess missing index assignment");
                                }
                                return (*indexed_env)[id];
                            };

	                            auto selectLinear = [&](const CodeValue& a, llvm::Value* idx) -> llvm::Value* {
	                                const auto n = elemCount(a.shape);
	                                if (n == 0u) {
	                                    throw std::runtime_error("LLVMGen: IndexedAccess operand has no elements");
	                                }
	                                llvm::Value* out = a.elems[0];
	                                for (std::size_t t = 1; t < n; ++t) {
	                                    auto* is_t = builder.CreateICmpEQ(idx, builder.getInt32(static_cast<std::uint32_t>(t)));
	                                    out = builder.CreateSelect(is_t, a.elems[t], out);
	                                }
	                                return out;
	                            };

                            if (rank == 1 && am.shape.kind == Shape::Kind::Vector) {
                                auto* ii = loadIndex(0);
                                values_minus[op_idx] = makeScalar(selectLinear(am, ii));
                                values_plus[op_idx] = makeScalar(selectLinear(ap, ii));
                                break;
                            }
                            if (rank == 2 && am.shape.kind == Shape::Kind::Matrix) {
                                auto* ii = loadIndex(0);
                                auto* jj = loadIndex(1);
                                const auto cols = static_cast<std::uint32_t>(am.shape.dims[1]);
                                auto* lin = builder.CreateAdd(builder.CreateMul(ii, builder.getInt32(cols)), jj);
                                values_minus[op_idx] = makeScalar(selectLinear(am, lin));
                                values_plus[op_idx] = makeScalar(selectLinear(ap, lin));
                                break;
                            }
                            if (rank == 3 && am.shape.kind == Shape::Kind::Tensor3) {
                                auto* ii = loadIndex(0);
                                auto* jj = loadIndex(1);
                                auto* kk = loadIndex(2);
                                const auto d1 = static_cast<std::uint32_t>(am.shape.dims[1]);
                                const auto d2 = static_cast<std::uint32_t>(am.shape.dims[2]);
                                auto* lin = builder.CreateAdd(builder.CreateMul(ii, builder.getInt32(d1 * d2)),
                                                              builder.CreateAdd(builder.CreateMul(jj, builder.getInt32(d2)), kk));
                                values_minus[op_idx] = makeScalar(selectLinear(am, lin));
                                values_plus[op_idx] = makeScalar(selectLinear(ap, lin));
                                break;
                            }
                            if (rank == 4 && am.shape.kind == Shape::Kind::Tensor4) {
                                auto* ii = loadIndex(0);
                                auto* jj = loadIndex(1);
                                auto* kk = loadIndex(2);
                                auto* ll = loadIndex(3);
                                const auto d1 = static_cast<std::uint32_t>(am.shape.dims[1]);
                                const auto d2 = static_cast<std::uint32_t>(am.shape.dims[2]);
                                const auto d3 = static_cast<std::uint32_t>(am.shape.dims[3]);
                                const auto stride0 = d1 * d2 * d3;
                                const auto stride1 = d2 * d3;
                                const auto stride2 = d3;
                                auto* lin = builder.CreateAdd(builder.CreateMul(ii, builder.getInt32(stride0)),
                                                              builder.CreateAdd(builder.CreateMul(jj, builder.getInt32(stride1)),
                                                                                builder.CreateAdd(builder.CreateMul(kk, builder.getInt32(stride2)), ll)));
                                values_minus[op_idx] = makeScalar(selectLinear(am, lin));
                                values_plus[op_idx] = makeScalar(selectLinear(ap, lin));
                                break;
                            }

                            throw std::runtime_error("LLVMGen: IndexedAccess rank/operand shape mismatch");
                        }

                        case FormExprType::Component: {
                            const auto i = static_cast<std::int32_t>(unpackU32Lo(op.imm0));
                            const auto j = static_cast<std::int32_t>(unpackU32Hi(op.imm0));
                            const auto& am = childMinus(op, 0);
                            const auto& ap = childPlus(op, 0);

                            if (am.shape.kind == Shape::Kind::Scalar) {
                                values_minus[op_idx] = am;
                            } else if (am.shape.kind == Shape::Kind::Vector) {
                                if (j >= 0) throw std::runtime_error("LLVMGen: component(v,i,j) invalid for vector");
                                values_minus[op_idx] = makeScalar(getVecComp(am, static_cast<std::size_t>(i)));
                            } else if (am.shape.kind == Shape::Kind::Matrix) {
                                if (j < 0) throw std::runtime_error("LLVMGen: component(A,i) missing column index");
                                const auto cols = static_cast<std::size_t>(am.shape.dims[1]);
                                values_minus[op_idx] = makeScalar(am.elems[static_cast<std::size_t>(i) * cols + static_cast<std::size_t>(j)]);
                            } else {
                                throw std::runtime_error("LLVMGen: component() unsupported operand kind");
                            }

                            if (ap.shape.kind == Shape::Kind::Scalar) {
                                values_plus[op_idx] = ap;
                            } else if (ap.shape.kind == Shape::Kind::Vector) {
                                if (j >= 0) throw std::runtime_error("LLVMGen: component(v,i,j) invalid for vector");
                                values_plus[op_idx] = makeScalar(getVecComp(ap, static_cast<std::size_t>(i)));
                            } else if (ap.shape.kind == Shape::Kind::Matrix) {
                                if (j < 0) throw std::runtime_error("LLVMGen: component(A,i) missing column index");
                                const auto cols = static_cast<std::size_t>(ap.shape.dims[1]);
                                values_plus[op_idx] = makeScalar(ap.elems[static_cast<std::size_t>(i) * cols + static_cast<std::size_t>(j)]);
                            } else {
                                throw std::runtime_error("LLVMGen: component() unsupported operand kind");
                            }
                            break;
                        }

                        case FormExprType::Transpose:
                            values_minus[op_idx] = transpose(childMinus(op, 0));
                            values_plus[op_idx] = transpose(childPlus(op, 0));
                            break;

                        case FormExprType::Trace:
                            values_minus[op_idx] = trace(childMinus(op, 0));
                            values_plus[op_idx] = trace(childPlus(op, 0));
                            break;

                        case FormExprType::Determinant:
                            values_minus[op_idx] = det(childMinus(op, 0));
                            values_plus[op_idx] = det(childPlus(op, 0));
                            break;

                        case FormExprType::Cofactor:
                            values_minus[op_idx] = cofactor(childMinus(op, 0));
                            values_plus[op_idx] = cofactor(childPlus(op, 0));
                            break;

                        case FormExprType::Inverse:
                            values_minus[op_idx] = inv(childMinus(op, 0));
                            values_plus[op_idx] = inv(childPlus(op, 0));
                            break;

                        case FormExprType::SymmetricPart:
                            values_minus[op_idx] = symOrSkew(true, childMinus(op, 0));
                            values_plus[op_idx] = symOrSkew(true, childPlus(op, 0));
                            break;

                        case FormExprType::SkewPart:
                            values_minus[op_idx] = symOrSkew(false, childMinus(op, 0));
                            values_plus[op_idx] = symOrSkew(false, childPlus(op, 0));
                            break;

                        case FormExprType::Deviator:
                            values_minus[op_idx] = deviator(childMinus(op, 0));
                            values_plus[op_idx] = deviator(childPlus(op, 0));
                            break;

                        case FormExprType::Norm:
                            values_minus[op_idx] = norm(childMinus(op, 0));
                            values_plus[op_idx] = norm(childPlus(op, 0));
                            break;

                        case FormExprType::Normalize:
                            values_minus[op_idx] = normalize(childMinus(op, 0));
                            values_plus[op_idx] = normalize(childPlus(op, 0));
                            break;

                        case FormExprType::AbsoluteValue:
                            values_minus[op_idx] = absScalar(childMinus(op, 0));
                            values_plus[op_idx] = absScalar(childPlus(op, 0));
                            break;

                        case FormExprType::Sign:
                            values_minus[op_idx] = signScalar(childMinus(op, 0));
                            values_plus[op_idx] = signScalar(childPlus(op, 0));
                            break;

                        case FormExprType::Sqrt:
                            values_minus[op_idx] = evalSqrt(childMinus(op, 0));
                            values_plus[op_idx] = evalSqrt(childPlus(op, 0));
                            break;

                        case FormExprType::Exp:
                            values_minus[op_idx] = evalExp(childMinus(op, 0));
                            values_plus[op_idx] = evalExp(childPlus(op, 0));
                            break;

                        case FormExprType::Log:
                            values_minus[op_idx] = evalLog(childMinus(op, 0));
                            values_plus[op_idx] = evalLog(childPlus(op, 0));
                            break;

	                        case FormExprType::MatrixExponential:
	                            values_minus[op_idx] = emitMatrixExp(childMinus(op, 0));
	                            values_plus[op_idx] = emitMatrixExp(childPlus(op, 0));
	                            break;

	                        case FormExprType::MatrixLogarithm:
	                            values_minus[op_idx] = emitMatrixLog(childMinus(op, 0));
	                            values_plus[op_idx] = emitMatrixLog(childPlus(op, 0));
	                            break;

	                        case FormExprType::MatrixSqrt:
	                            values_minus[op_idx] = emitMatrixSqrt(childMinus(op, 0));
	                            values_plus[op_idx] = emitMatrixSqrt(childPlus(op, 0));
	                            break;

	                        case FormExprType::MatrixPower:
	                            values_minus[op_idx] = emitMatrixPow(childMinus(op, 0), childMinus(op, 1).elems[0]);
	                            values_plus[op_idx] = emitMatrixPow(childPlus(op, 0), childPlus(op, 1).elems[0]);
	                            break;

                        case FormExprType::MatrixExponentialDirectionalDerivative:
                            values_minus[op_idx] = callMatrixUnaryDD(childMinus(op, 0), childMinus(op, 1), mat_exp_dd_2x2_fn, mat_exp_dd_3x3_fn);
                            values_plus[op_idx] = callMatrixUnaryDD(childPlus(op, 0), childPlus(op, 1), mat_exp_dd_2x2_fn, mat_exp_dd_3x3_fn);
                            break;

                        case FormExprType::MatrixLogarithmDirectionalDerivative:
                            values_minus[op_idx] = callMatrixUnaryDD(childMinus(op, 0), childMinus(op, 1), mat_log_dd_2x2_fn, mat_log_dd_3x3_fn);
                            values_plus[op_idx] = callMatrixUnaryDD(childPlus(op, 0), childPlus(op, 1), mat_log_dd_2x2_fn, mat_log_dd_3x3_fn);
                            break;

                        case FormExprType::MatrixSqrtDirectionalDerivative:
                            values_minus[op_idx] = callMatrixUnaryDD(childMinus(op, 0), childMinus(op, 1), mat_sqrt_dd_2x2_fn, mat_sqrt_dd_3x3_fn);
                            values_plus[op_idx] = callMatrixUnaryDD(childPlus(op, 0), childPlus(op, 1), mat_sqrt_dd_2x2_fn, mat_sqrt_dd_3x3_fn);
                            break;

                        case FormExprType::MatrixPowerDirectionalDerivative:
                            values_minus[op_idx] = callMatrixPowDD(childMinus(op, 0), childMinus(op, 1), childMinus(op, 2).elems[0], mat_pow_dd_2x2_fn, mat_pow_dd_3x3_fn);
                            values_plus[op_idx] = callMatrixPowDD(childPlus(op, 0), childPlus(op, 1), childPlus(op, 2).elems[0], mat_pow_dd_2x2_fn, mat_pow_dd_3x3_fn);
                            break;

                        case FormExprType::SmoothAbsoluteValue:
                            values_minus[op_idx] = smoothAbs(childMinus(op, 0), childMinus(op, 1));
                            values_plus[op_idx] = smoothAbs(childPlus(op, 0), childPlus(op, 1));
                            break;

                        case FormExprType::SmoothSign:
                            values_minus[op_idx] = smoothSign(childMinus(op, 0), childMinus(op, 1));
                            values_plus[op_idx] = smoothSign(childPlus(op, 0), childPlus(op, 1));
                            break;

                        case FormExprType::SmoothHeaviside:
                            values_minus[op_idx] = smoothHeaviside(childMinus(op, 0), childMinus(op, 1));
                            values_plus[op_idx] = smoothHeaviside(childPlus(op, 0), childPlus(op, 1));
                            break;

                        case FormExprType::SmoothMin:
                            values_minus[op_idx] = smoothMin(childMinus(op, 0), childMinus(op, 1), childMinus(op, 2));
                            values_plus[op_idx] = smoothMin(childPlus(op, 0), childPlus(op, 1), childPlus(op, 2));
                            break;

                        case FormExprType::SmoothMax:
                            values_minus[op_idx] = smoothMax(childMinus(op, 0), childMinus(op, 1), childMinus(op, 2));
                            values_plus[op_idx] = smoothMax(childPlus(op, 0), childPlus(op, 1), childPlus(op, 2));
                            break;

                        case FormExprType::SymmetricEigenvalue: {
                            const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                            const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                            values_minus[op_idx] = eigSym(childMinus(op, 0), which);
                            values_plus[op_idx] = eigSym(childPlus(op, 0), which);
                            break;
                        }

                        case FormExprType::Eigenvalue: {
                            const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                            const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                            values_minus[op_idx] = eigSym(childMinus(op, 0), which);
                            values_plus[op_idx] = eigSym(childPlus(op, 0), which);
                            break;
                        }

                        case FormExprType::SymmetricEigenvector: {
                            const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                            const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                            values_minus[op_idx] = eigSymVec(childMinus(op, 0), which);
                            values_plus[op_idx] = eigSymVec(childPlus(op, 0), which);
                            break;
                        }

                        case FormExprType::SpectralDecomposition:
                            values_minus[op_idx] = spectralDecomp(childMinus(op, 0));
                            values_plus[op_idx] = spectralDecomp(childPlus(op, 0));
                            break;

                        case FormExprType::SymmetricEigenvectorDirectionalDerivative: {
                            const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                            const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                            values_minus[op_idx] = eigSymVecDD(childMinus(op, 0), childMinus(op, 1), which);
                            values_plus[op_idx] = eigSymVecDD(childPlus(op, 0), childPlus(op, 1), which);
                            break;
                        }

                        case FormExprType::SpectralDecompositionDirectionalDerivative:
                            values_minus[op_idx] = spectralDecompDD(childMinus(op, 0), childMinus(op, 1));
                            values_plus[op_idx] = spectralDecompDD(childPlus(op, 0), childPlus(op, 1));
                            break;

	                        case FormExprType::HistoryWeightedSum:
	                        case FormExprType::HistoryConvolution: {
	                            CodeValue acc_m = makeZero(shape);
	                            CodeValue acc_p = makeZero(shape);
	                            if (op.child_count != 0u) {
	                                for (std::size_t kk = 0; kk < static_cast<std::size_t>(op.child_count); ++kk) {
	                                    const int k = static_cast<int>(kk + 1u);
	                                    acc_m = add(acc_m,
	                                                mul(childMinus(op, kk),
	                                                    evalHistorySolution(side_minus, shape, k, q_index)));
	                                    acc_p = add(acc_p,
	                                                mul(childPlus(op, kk),
	                                                    evalHistorySolution(side_plus, shape, k, q_index)));
	                                }
	                            } else {
	                                for (int k = 1; k <= static_cast<int>(assembly::jit::kMaxPreviousSolutionsV6); ++k) {
	                                    acc_m = add(acc_m,
	                                                mul(makeScalar(loadHistoryWeightOrZero(side_minus, k)),
	                                                    evalHistorySolution(side_minus, shape, k, q_index)));
	                                    acc_p = add(acc_p,
	                                                mul(makeScalar(loadHistoryWeightOrZero(side_plus, k)),
	                                                    evalHistorySolution(side_plus, shape, k, q_index)));
	                                }
	                            }
	                            values_minus[op_idx] = acc_m;
	                            values_plus[op_idx] = acc_p;
	                            break;
	                        }

                        case FormExprType::SymmetricEigenvalueDirectionalDerivative: {
                            const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                            const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                            values_minus[op_idx] = eigSymDD(childMinus(op, 0), childMinus(op, 1), which);
                            values_plus[op_idx] = eigSymDD(childPlus(op, 0), childPlus(op, 1), which);
                            break;
                        }

                        case FormExprType::SymmetricEigenvalueDirectionalDerivativeWrtA: {
                            const auto which_i32 = static_cast<std::int32_t>(static_cast<std::int64_t>(op.imm0));
                            const auto which = static_cast<std::uint32_t>(std::max<std::int32_t>(0, which_i32));
                            values_minus[op_idx] = eigSymDDWrtA(childMinus(op, 0), childMinus(op, 1), childMinus(op, 2), which);
                            values_plus[op_idx] = eigSymDDWrtA(childPlus(op, 0), childPlus(op, 1), childPlus(op, 2), which);
                            break;
                        }

                        default:
                            throw std::runtime_error("LLVMGen: unsupported FormExprType in face codegen (FormExprType=" +
                                                     std::to_string(static_cast<std::uint16_t>(op.type)) + ")");
                    }
                }

                if (term.ir.root >= values_minus.size() || term.ir.root >= values_plus.size()) {
                    throw std::runtime_error("LLVMGen: invalid KernelIR root index");
                }
                return eval_plus ? values_plus[term.ir.root] : values_minus[term.ir.root];
            };

            auto evalKernelIRFaceScalar = [&](const LoweredTerm& term,
                                              llvm::Value* q_index,
                                              llvm::Value* i_index,
                                              llvm::Value* j_index,
                                              bool eval_plus,
                                              bool test_active_plus,
                                              bool trial_active_plus,
                                              const std::vector<llvm::Value*>* indexed_env,
                                              const FaceCached* cached) -> llvm::Value* {
                const auto root = evalKernelIRFaceValue(term,
                                                        q_index,
                                                        i_index,
                                                        j_index,
                                                        eval_plus,
                                                        test_active_plus,
                                                        trial_active_plus,
                                                        indexed_env,
                                                        cached);
                if (root.shape.kind != Shape::Kind::Scalar) {
                    throw std::runtime_error("LLVMGen: integrand did not lower to scalar");
                }
                return root.elems[0];
            };

            auto evalKernelIRFace = [&](const LoweredTerm& term,
                                        llvm::Value* q_index,
                                        llvm::Value* i_index,
                                        llvm::Value* j_index,
                                        bool eval_plus,
                                        bool test_active_plus,
                                        bool trial_active_plus,
                                        const FaceCached* cached) -> llvm::Value* {
                if (term.tensor_ir.has_value()) {
                    LLVMTensorGen tensor_gen(*ctx,
                                             builder,
                                             *fn,
                                             LLVMTensorGenOptions{
                                                 .vectorize = options_.vectorize,
                                                 .enable_polly = options_.tensor.enable_polly,
                                                 .enable_tiling = options_.tensor.enable_loop_tiling,
                                                 .tile_size = static_cast<int>(options_.tensor.tile_size),
                                                 .min_tiling_extent = static_cast<int>(options_.tensor.min_tiling_extent),
                                             });

                    const auto eval_scalar = [&](const FormExpr& scalar_expr) -> llvm::Value* {
                        if (!scalar_expr.isValid() || scalar_expr.node() == nullptr) {
                            throw std::runtime_error("LLVMGen: TensorIR scalar expression is invalid");
                        }

                        auto lowered_scalar = lowerToKernelIR(scalar_expr);
                        auto scalar_shapes = inferShapes(lowered_scalar.ir, ir.testSpace(), ir.trialSpace(), /*require_scalar_root=*/false);
                        if (!scalar_shapes.ok) {
                            throw std::runtime_error(scalar_shapes.message.empty() ? "LLVMGen: failed to infer scalar shapes" : scalar_shapes.message);
                        }

                        LoweredTerm scalar_term;
                        scalar_term.ir = std::move(lowered_scalar.ir);
                        scalar_term.shapes = std::move(scalar_shapes.shapes);

                        const auto v = evalKernelIRFaceValue(scalar_term,
                                                             q_index,
                                                             i_index,
                                                             j_index,
                                                             eval_plus,
                                                             test_active_plus,
                                                             trial_active_plus,
                                                             /*indexed_env=*/nullptr,
                                                             /*cached=*/nullptr);
                        if (v.shape.kind != Shape::Kind::Scalar) {
                            throw std::runtime_error("LLVMGen: TensorIR scalar expression did not lower to scalar");
                        }
                        return v.elems[0];
                    };

                    const auto& tir = *term.tensor_ir;
                    std::vector<CodeValue> base_values;
                    base_values.resize(tir.program.tensors.size());
                    std::vector<bool> has_base;
                    has_base.resize(tir.program.tensors.size(), false);

                    for (std::size_t tid = 0; tid < tir.program.tensors.size(); ++tid) {
                        const auto& spec = tir.program.tensors[tid];
                        if (!spec.base.isValid() || spec.base.node() == nullptr) {
                            continue;
                        }

                        auto lowered_base = lowerToKernelIR(spec.base);
                        auto base_shapes = inferShapes(lowered_base.ir, ir.testSpace(), ir.trialSpace(), /*require_scalar_root=*/false);
                        if (!base_shapes.ok) {
                            throw std::runtime_error(base_shapes.message.empty() ? "LLVMGen: failed to infer base-tensor shapes" : base_shapes.message);
                        }

                        LoweredTerm base_term;
                        base_term.ir = std::move(lowered_base.ir);
                        base_term.shapes = std::move(base_shapes.shapes);

                        base_values[tid] = evalKernelIRFaceValue(base_term,
                                                                 q_index,
                                                                 i_index,
                                                                 j_index,
                                                                 eval_plus,
                                                                 test_active_plus,
                                                                 trial_active_plus,
                                                                 /*indexed_env=*/nullptr,
                                                                 /*cached=*/nullptr);
                        has_base[tid] = true;
                    }

                    const auto load_input = [&](int tensor_id,
                                                const tensor::TensorSpec& spec,
                                                const std::vector<llvm::Value*>& index_env) -> llvm::Value* {
                        if (tensor_id < 0 || static_cast<std::size_t>(tensor_id) >= base_values.size()) {
                            throw std::runtime_error("LLVMGen: TensorIR input tensor id out of range");
                        }
                        if (!has_base[static_cast<std::size_t>(tensor_id)]) {
                            throw std::runtime_error("LLVMGen: TensorIR input tensor missing evaluated base value");
                        }

                        const auto& base = base_values[static_cast<std::size_t>(tensor_id)];
                        if (base.shape.kind == Shape::Kind::Scalar) {
                            return base.elems[0];
                        }

                        const int base_rank = spec.base_rank;
                        if (base_rank < 0 || static_cast<std::size_t>(base_rank) > spec.base_axis_to_tensor_axis.size()) {
                            throw std::runtime_error("LLVMGen: TensorIR input tensor has invalid base-rank metadata");
                        }

                        auto loadIndex = [&](int axis) -> llvm::Value* {
                            if (axis < 0 || axis >= base_rank) {
                                throw std::runtime_error("LLVMGen: TensorIR input tensor base-axis out of range");
                            }
                            const int tpos = spec.base_axis_to_tensor_axis[static_cast<std::size_t>(axis)];
                            if (tpos < 0 || static_cast<std::size_t>(tpos) >= spec.axes.size()) {
                                throw std::runtime_error("LLVMGen: TensorIR input tensor has invalid axis mapping");
                            }
                            const int id = spec.axes[static_cast<std::size_t>(tpos)];
                            if (id < 0 || static_cast<std::size_t>(id) >= index_env.size() || index_env[static_cast<std::size_t>(id)] == nullptr) {
                                throw std::runtime_error("LLVMGen: TensorIR input tensor missing index assignment");
                            }
                            return index_env[static_cast<std::size_t>(id)];
                        };

                        auto selectLinear = [&](llvm::Value* idx) -> llvm::Value* {
                            const auto n = elemCount(base.shape);
                            if (n == 0u) {
                                throw std::runtime_error("LLVMGen: TensorIR input tensor has empty base");
                            }
                            llvm::Value* out = base.elems[0];
                            for (std::size_t t = 1; t < n; ++t) {
                                auto* is_t = builder.CreateICmpEQ(idx, builder.getInt32(static_cast<std::uint32_t>(t)));
                                out = builder.CreateSelect(is_t, base.elems[t], out);
                            }
                            return out;
                        };

                        if (base.shape.kind == Shape::Kind::Vector) {
                            if (base_rank != 1) throw std::runtime_error("LLVMGen: TensorIR input vector rank mismatch");
                            return selectLinear(loadIndex(0));
                        }
                        if (base.shape.kind == Shape::Kind::Matrix) {
                            if (base_rank != 2) throw std::runtime_error("LLVMGen: TensorIR input matrix rank mismatch");
                            auto* ii = loadIndex(0);
                            auto* jj = loadIndex(1);
                            const auto cols = static_cast<std::uint32_t>(base.shape.dims[1]);
                            auto* lin = builder.CreateAdd(builder.CreateMul(ii, builder.getInt32(cols)), jj);
                            return selectLinear(lin);
                        }
                        if (base.shape.kind == Shape::Kind::Tensor3) {
                            if (base_rank != 3) throw std::runtime_error("LLVMGen: TensorIR input tensor3 rank mismatch");
                            auto* ii = loadIndex(0);
                            auto* jj = loadIndex(1);
                            auto* kk = loadIndex(2);
                            const auto d1 = static_cast<std::uint32_t>(base.shape.dims[1]);
                            const auto d2 = static_cast<std::uint32_t>(base.shape.dims[2]);
                            auto* lin = builder.CreateAdd(builder.CreateMul(ii, builder.getInt32(d1 * d2)),
                                                          builder.CreateAdd(builder.CreateMul(jj, builder.getInt32(d2)), kk));
                            return selectLinear(lin);
                        }
                        if (base.shape.kind == Shape::Kind::Tensor4) {
                            if (base_rank != 4) throw std::runtime_error("LLVMGen: TensorIR input tensor4 rank mismatch");
                            auto* ii = loadIndex(0);
                            auto* jj = loadIndex(1);
                            auto* kk = loadIndex(2);
                            auto* ll = loadIndex(3);
                            const auto d1 = static_cast<std::uint32_t>(base.shape.dims[1]);
                            const auto d2 = static_cast<std::uint32_t>(base.shape.dims[2]);
                            const auto d3 = static_cast<std::uint32_t>(base.shape.dims[3]);
                            const auto stride0 = d1 * d2 * d3;
                            const auto stride1 = d2 * d3;
                            const auto stride2 = d3;
                            auto* lin = builder.CreateAdd(builder.CreateMul(ii, builder.getInt32(stride0)),
                                                          builder.CreateAdd(builder.CreateMul(jj, builder.getInt32(stride1)),
                                                                            builder.CreateAdd(builder.CreateMul(kk, builder.getInt32(stride2)), ll)));
                            return selectLinear(lin);
                        }

                        throw std::runtime_error("LLVMGen: TensorIR input tensor has unsupported shape kind");
                    };

                    (void)cached;
                    return tensor_gen.emitScalar(tir, eval_scalar, load_input);
                }

                if (!term.has_indexed_access) {
                    return evalKernelIRFaceScalar(term, q_index, i_index, j_index,
                                                  eval_plus, test_active_plus, trial_active_plus,
                                                  /*indexed_env=*/nullptr, cached);
                }

                if (term.bound_indices.empty()) {
                    throw std::runtime_error("LLVMGen: IndexedAccess term missing bound-indices metadata");
                }

                std::size_t max_id = 0;
                for (const auto& [id, ext] : term.bound_indices) {
                    (void)ext;
                    max_id = std::max(max_id, static_cast<std::size_t>(id));
                }
                std::vector<llvm::Value*> idx_env(max_id + 1u, nullptr);

                const auto emitSum = [&](const auto& self, std::size_t level) -> llvm::Value* {
                    if (level >= term.bound_indices.size()) {
                        return evalKernelIRFaceScalar(term, q_index, i_index, j_index,
                                                      eval_plus, test_active_plus, trial_active_plus,
                                                      &idx_env, cached);
                    }

                    const auto [id_u16, extent_u8] = term.bound_indices[level];
                    const std::size_t id = static_cast<std::size_t>(id_u16);
                    const std::uint32_t extent = static_cast<std::uint32_t>(extent_u8);
                    if (extent == 0u) {
                        throw std::runtime_error("LLVMGen: IndexedAccess has zero loop extent");
                    }

                    auto* preheader = builder.GetInsertBlock();
                    auto* header = llvm::BasicBlock::Create(*ctx, "fidx" + std::to_string(level) + ".h", fn);
                    auto* body = llvm::BasicBlock::Create(*ctx, "fidx" + std::to_string(level) + ".b", fn);
                    auto* latch = llvm::BasicBlock::Create(*ctx, "fidx" + std::to_string(level) + ".l", fn);
                    auto* exit = llvm::BasicBlock::Create(*ctx, "fidx" + std::to_string(level) + ".x", fn);

                    builder.CreateBr(header);
                    builder.SetInsertPoint(header);
                    auto* idx_phi = builder.CreatePHI(i32, 2, "fidx" + std::to_string(level));
                    idx_phi->addIncoming(builder.getInt32(0), preheader);
                    auto* acc_phi = builder.CreatePHI(f64, 2, "fidx.acc" + std::to_string(level));
                    acc_phi->addIncoming(f64c(0.0), preheader);
                    auto* cond = builder.CreateICmpULT(idx_phi, builder.getInt32(extent));
                    builder.CreateCondBr(cond, body, exit);

                    builder.SetInsertPoint(body);
                    llvm::Value* prev = nullptr;
                    if (id < idx_env.size()) {
                        prev = idx_env[id];
                    } else {
                        idx_env.resize(id + 1u, nullptr);
                    }
                    idx_env[id] = idx_phi;

                    auto* inner = self(self, level + 1u);

                    idx_env[id] = prev;
                    builder.CreateBr(latch);

                    builder.SetInsertPoint(latch);
                    auto* acc_next = builder.CreateFAdd(acc_phi, inner);
                    auto* idx_next = builder.CreateAdd(idx_phi, builder.getInt32(1));
                    builder.CreateBr(header);
                    idx_phi->addIncoming(idx_next, latch);
                    acc_phi->addIncoming(acc_next, latch);

                    builder.SetInsertPoint(exit);
                    return acc_phi;
                };

                return emitSum(emitSum, 0u);
            };

            if (want_matrix) {
	                auto emitFaceBlock = [&](bool eval_plus,
	                                         bool test_active_plus,
	                                         bool trial_active_plus,
	                                         llvm::Value* out_view_ptr,
	                                         std::string_view label) -> void {
	                    const SideView& side_eval = eval_plus ? side_plus : side_minus;
	                    auto* element_matrix = loadPtr(out_view_ptr, ABIV3::out_element_matrix_off);
	                    auto* n_test = loadU32(out_view_ptr, ABIV3::out_n_test_dofs_off);
	                    auto* n_trial = loadU32(out_view_ptr, ABIV3::out_n_trial_dofs_off);
	                    if (specialization != nullptr) {
	                        const auto& fixed_test =
	                            test_active_plus ? specialization->n_test_dofs_plus : specialization->n_test_dofs_minus;
	                        const auto& fixed_trial =
	                            trial_active_plus ? specialization->n_trial_dofs_plus : specialization->n_trial_dofs_minus;
	                        if (fixed_test) {
	                            n_test = builder.getInt32(*fixed_test);
	                        }
	                        if (fixed_trial) {
	                            n_trial = builder.getInt32(*fixed_trial);
	                        }
	                    }

                    for (std::size_t t = 0; t < terms.size(); ++t) {
                        auto* term_entry = llvm::BasicBlock::Create(*ctx, std::string(label) + "_term" + std::to_string(t) + ".entry", fn);
                        auto* term_body = llvm::BasicBlock::Create(*ctx, std::string(label) + "_term" + std::to_string(t) + ".body", fn);
                        auto* term_end = llvm::BasicBlock::Create(*ctx, std::string(label) + "_term" + std::to_string(t) + ".end", fn);

                        builder.CreateBr(term_entry);
                        builder.SetInsertPoint(term_entry);
                        auto* tw = termWeight(side_eval, terms[t].time_derivative_order);
                        auto* is_zero = builder.CreateFCmpOEQ(tw, f64c(0.0));
                        builder.CreateCondBr(is_zero, term_end, term_body);

                        builder.SetInsertPoint(term_body);
                        emitForLoop(side_eval.n_qpts, std::string(label) + "_q" + std::to_string(t), [&](llvm::Value* q) {
                            auto* q64 = builder.CreateZExt(q, i64);
                            auto* w = loadRealPtrAt(side_eval.integration_weights, q64);
                            auto* scaled_w = builder.CreateFMul(tw, w);

                            FaceCached cached;
                            const FaceCached* cached_ptr = nullptr;
                            if (!terms[t].has_indexed_access) {
                                cached = computeCachedFace(terms[t], q);
                                cached_ptr = &cached;
                            }

                            emitForLoop(n_test, std::string(label) + "_i" + std::to_string(t), [&](llvm::Value* i) {
                                emitForLoop(n_trial, std::string(label) + "_j" + std::to_string(t), [&](llvm::Value* j) {
                                    auto* val = evalKernelIRFace(terms[t], q, i, j,
                                                                 eval_plus,
                                                                 test_active_plus,
                                                                 trial_active_plus,
                                                                 cached_ptr);
                                    auto* contrib = builder.CreateFMul(scaled_w, val);
                                    emitMatrixAccum(element_matrix, n_trial, i, j, contrib);
                                });
                            });
                        });

                        builder.CreateBr(term_end);
                        builder.SetInsertPoint(term_end);
                    }
                };

                emitFaceBlock(/*eval_plus=*/false,
                              /*test_active_plus=*/false,
                              /*trial_active_plus=*/false,
                              out_minus_ptr,
                              "mm");
                emitFaceBlock(/*eval_plus=*/true,
                              /*test_active_plus=*/true,
                              /*trial_active_plus=*/true,
                              out_plus_ptr,
                              "pp");
                emitFaceBlock(/*eval_plus=*/false,
                              /*test_active_plus=*/false,
                              /*trial_active_plus=*/true,
                              out_coupling_mp_ptr,
                              "mp");
                emitFaceBlock(/*eval_plus=*/true,
                              /*test_active_plus=*/true,
                              /*trial_active_plus=*/false,
                              out_coupling_pm_ptr,
                              "pm");
            } else if (want_vector) {
	                auto emitFaceVector = [&](bool eval_plus,
	                                          bool test_active_plus,
	                                          llvm::Value* out_view_ptr,
	                                          std::string_view label) -> void {
	                    const SideView& side_eval = eval_plus ? side_plus : side_minus;
	                    auto* element_vector = loadPtr(out_view_ptr, ABIV3::out_element_vector_off);
	                    auto* n_test = loadU32(out_view_ptr, ABIV3::out_n_test_dofs_off);
	                    if (specialization != nullptr) {
	                        const auto& fixed_test =
	                            test_active_plus ? specialization->n_test_dofs_plus : specialization->n_test_dofs_minus;
	                        if (fixed_test) {
	                            n_test = builder.getInt32(*fixed_test);
	                        }
	                    }

                    for (std::size_t t = 0; t < terms.size(); ++t) {
                        auto* term_entry = llvm::BasicBlock::Create(*ctx, std::string(label) + "_term" + std::to_string(t) + ".entry", fn);
                        auto* term_body = llvm::BasicBlock::Create(*ctx, std::string(label) + "_term" + std::to_string(t) + ".body", fn);
                        auto* term_end = llvm::BasicBlock::Create(*ctx, std::string(label) + "_term" + std::to_string(t) + ".end", fn);

                        builder.CreateBr(term_entry);
                        builder.SetInsertPoint(term_entry);
                        auto* tw = termWeight(side_eval, terms[t].time_derivative_order);
                        auto* is_zero = builder.CreateFCmpOEQ(tw, f64c(0.0));
                        builder.CreateCondBr(is_zero, term_end, term_body);

                        builder.SetInsertPoint(term_body);
                        emitForLoop(side_eval.n_qpts, std::string(label) + "_q" + std::to_string(t), [&](llvm::Value* q) {
                            auto* q64 = builder.CreateZExt(q, i64);
                            auto* w = loadRealPtrAt(side_eval.integration_weights, q64);
                            auto* scaled_w = builder.CreateFMul(tw, w);

                            FaceCached cached;
                            const FaceCached* cached_ptr = nullptr;
                            if (!terms[t].has_indexed_access) {
                                cached = computeCachedFace(terms[t], q);
                                cached_ptr = &cached;
                            }

                            emitForLoop(n_test, std::string(label) + "_i" + std::to_string(t), [&](llvm::Value* i) {
                                auto* val = evalKernelIRFace(terms[t], q, i, builder.getInt32(0),
                                                             eval_plus,
                                                             test_active_plus,
                                                             /*trial_active_plus=*/false,
                                                             cached_ptr);
                                auto* contrib = builder.CreateFMul(scaled_w, val);
                                emitVectorAccum(element_vector, i, contrib);
                            });
                        });

                        builder.CreateBr(term_end);
                        builder.SetInsertPoint(term_end);
                    }
                };

                emitFaceVector(/*eval_plus=*/false,
                               /*test_active_plus=*/false,
                               out_minus_ptr,
                               "rm");
                emitFaceVector(/*eval_plus=*/true,
                               /*test_active_plus=*/true,
                               out_plus_ptr,
                               "rp");
            }
        }

        builder.CreateRetVoid();

        if (di_builder) {
            di_builder->finalize();
        }

        if (llvm::verifyModule(*module, &llvm::errs())) {
            return LLVMGenResult{.ok = false, .message = "LLVMGen: generated module failed verification"};
        }

        if (options_.dump_llvm_ir) {
            std::string ir_text;
            llvm::raw_string_ostream os(ir_text);
            module->print(os, nullptr);
            os.flush();
            writeTextFile(dumpPath(options_, symbol, "_before.ll"), ir_text);
        }
        if (options_.dump_llvm_ir_optimized && options_.optimization_level <= 0) {
            std::string ir_text;
            llvm::raw_string_ostream os(ir_text);
            module->print(os, nullptr);
            os.flush();
            writeTextFile(dumpPath(options_, symbol, "_after.ll"), ir_text);
        }

        llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(ctx));
        engine.addModule(std::move(tsm));
        out_address = engine.lookup(symbol);
        return LLVMGenResult{.ok = true, .message = {}};
    } catch (const std::exception& e) {
        return LLVMGenResult{.ok = false, .message = std::string("LLVMGen: exception: ") + e.what()};
    }
#endif
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp
