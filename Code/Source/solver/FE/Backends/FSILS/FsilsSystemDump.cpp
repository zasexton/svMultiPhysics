/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/FSILS/FsilsSystemDump.h"

#include "Backends/FSILS/FsilsMatrix.h"
#include "Backends/FSILS/FsilsShared.h"
#include "Backends/FSILS/FsilsVector.h"
#include "Core/FEException.h"
#include "Core/Logger.h"

#include "Backends/FSILS/liner_solver/fils_struct.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

namespace {

constexpr char kMagic[8] = {'S', 'V', 'M', 'P', 'L', 'S', '0', '1'};
constexpr std::int32_t kVersion = 1;

[[nodiscard]] const char* dumpPrefix() noexcept
{
    const char* env = std::getenv("SVMP_FSILS_DUMP_SYSTEM_PREFIX");
    if (env == nullptr || env[0] == '\0') {
        return nullptr;
    }
    return env;
}

[[nodiscard]] long long envInteger(const char* name, long long fallback) noexcept
{
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return fallback;
    }
    try {
        return std::stoll(std::string(env));
    } catch (...) {
        return fallback;
    }
}

template <typename T>
void writeValue(std::ofstream& out, const T& value)
{
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void writeArray(std::ofstream& out, const std::vector<T>& values)
{
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(T)));
    }
}

template <typename T>
void readValue(std::ifstream& in, T& value, const std::string& path)
{
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    FE_THROW_IF(!in, FEException, "readFsilsSystemSnapshot: truncated file '" + path + "'");
}

template <typename T>
void readArray(std::ifstream& in, std::vector<T>& values, std::size_t count, const std::string& path)
{
    values.resize(count);
    if (count > 0) {
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(count * sizeof(T)));
        FE_THROW_IF(!in, FEException, "readFsilsSystemSnapshot: truncated file '" + path + "'");
    }
}

} // namespace

bool fsilsSystemDumpRequested() noexcept
{
    return dumpPrefix() != nullptr;
}

void writeFsilsSystemSnapshot(const FsilsSystemSnapshot& s, const std::string& path)
{
    std::ofstream out(path, std::ios::binary);
    FE_THROW_IF(!out, FEException, "writeFsilsSystemSnapshot: cannot open '" + path + "'");
    out.write(kMagic, sizeof(kMagic));
    writeValue(out, kVersion);
    writeValue(out, static_cast<std::int32_t>(s.dof));
    writeValue(out, static_cast<std::int32_t>(s.n_nodes));
    writeValue(out, static_cast<std::int64_t>(s.nnzBlocks()));
    writeValue(out, static_cast<std::int64_t>(s.dirichlet_dofs.size()));
    const std::int32_t ints[] = {s.method, s.preconditioner, s.use_rcs, s.max_iter, s.krylov_dim,
                                 s.iterations, s.converged, s.native_update_count};
    for (const auto v : ints) {
        writeValue(out, v);
    }
    const double reals[] = {s.rel_tol, s.abs_tol, s.initial_residual_norm, s.final_residual_norm,
                            s.solve_seconds};
    for (const auto v : reals) {
        writeValue(out, v);
    }
    writeValue(out, static_cast<std::int32_t>(s.blocks.size()));
    for (const auto& block : s.blocks) {
        writeValue(out, static_cast<std::int32_t>(block.start_component));
        writeValue(out, static_cast<std::int32_t>(block.n_components));
        writeValue(out, static_cast<std::int32_t>(block.role));
        writeValue(out, static_cast<std::int32_t>(block.name.size()));
        out.write(block.name.data(), static_cast<std::streamsize>(block.name.size()));
    }
    writeArray(out, s.row_ptr);
    writeArray(out, s.cols);
    writeArray(out, s.values);
    writeArray(out, s.rhs);
    writeArray(out, s.dirichlet_dofs);
    writeArray(out, s.solution);
    FE_THROW_IF(!out, FEException, "writeFsilsSystemSnapshot: write failed for '" + path + "'");
}

FsilsSystemSnapshot readFsilsSystemSnapshot(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    FE_THROW_IF(!in, FEException, "readFsilsSystemSnapshot: cannot open '" + path + "'");
    char magic[sizeof(kMagic)] = {};
    in.read(magic, sizeof(magic));
    FE_THROW_IF(!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0, FEException,
                "readFsilsSystemSnapshot: bad header in '" + path + "'");
    std::int32_t version = 0;
    readValue(in, version, path);
    FE_THROW_IF(version != kVersion, FEException,
                "readFsilsSystemSnapshot: unsupported version in '" + path + "'");

    FsilsSystemSnapshot s;
    std::int32_t dof = 0;
    std::int32_t n_nodes = 0;
    std::int64_t nnz = 0;
    std::int64_t n_dir = 0;
    readValue(in, dof, path);
    readValue(in, n_nodes, path);
    readValue(in, nnz, path);
    readValue(in, n_dir, path);
    FE_THROW_IF(dof <= 0 || n_nodes <= 0 || nnz <= 0 || n_dir < 0, FEException,
                "readFsilsSystemSnapshot: invalid sizes in '" + path + "'");
    s.dof = dof;
    s.n_nodes = n_nodes;
    std::int32_t ints[8] = {};
    for (auto& v : ints) {
        readValue(in, v, path);
    }
    s.method = ints[0];
    s.preconditioner = ints[1];
    s.use_rcs = ints[2];
    s.max_iter = ints[3];
    s.krylov_dim = ints[4];
    s.iterations = ints[5];
    s.converged = ints[6];
    s.native_update_count = ints[7];
    double reals[5] = {};
    for (auto& v : reals) {
        readValue(in, v, path);
    }
    s.rel_tol = reals[0];
    s.abs_tol = reals[1];
    s.initial_residual_norm = reals[2];
    s.final_residual_norm = reals[3];
    s.solve_seconds = reals[4];
    std::int32_t n_blocks = 0;
    readValue(in, n_blocks, path);
    FE_THROW_IF(n_blocks < 0 || n_blocks > dof, FEException,
                "readFsilsSystemSnapshot: invalid block count in '" + path + "'");
    for (int i = 0; i < n_blocks; ++i) {
        std::int32_t start = 0;
        std::int32_t count = 0;
        std::int32_t role = 0;
        std::int32_t name_len = 0;
        readValue(in, start, path);
        readValue(in, count, path);
        readValue(in, role, path);
        readValue(in, name_len, path);
        FE_THROW_IF(name_len < 0 || name_len > 4096, FEException,
                    "readFsilsSystemSnapshot: invalid block name in '" + path + "'");
        std::string name(static_cast<std::size_t>(name_len), '\0');
        if (name_len > 0) {
            in.read(name.data(), name_len);
        }
        BlockDescriptor block;
        block.name = name;
        block.start_component = start;
        block.n_components = count;
        block.role = static_cast<BlockRole>(role);
        s.blocks.push_back(block);
    }
    const auto nd = static_cast<std::size_t>(n_nodes) * static_cast<std::size_t>(dof);
    readArray(in, s.row_ptr, static_cast<std::size_t>(n_nodes) + 1u, path);
    readArray(in, s.cols, static_cast<std::size_t>(nnz), path);
    readArray(in, s.values, static_cast<std::size_t>(nnz) * static_cast<std::size_t>(dof * dof), path);
    readArray(in, s.rhs, nd, path);
    readArray(in, s.dirichlet_dofs, static_cast<std::size_t>(n_dir), path);
    readArray(in, s.solution, nd, path);
    FE_THROW_IF(s.row_ptr.front() != 0 || s.row_ptr.back() != nnz, FEException,
                "readFsilsSystemSnapshot: inconsistent row pointer in '" + path + "'");
    return s;
}

void maybeDumpFsilsSystem(const FsilsMatrix& A,
                          const FsilsVector& b,
                          const FsilsVector& x,
                          std::span<const GlobalIndex> dirichlet_fe_dofs,
                          const SolverOptions& options,
                          const SolverReport& report,
                          double solve_seconds,
                          int native_update_count)
{
    const char* prefix = dumpPrefix();
    if (prefix == nullptr) {
        return;
    }
    static std::atomic<long long> attempts{0};
    const long long attempt = attempts.fetch_add(1);
    const long long skip = std::max(0LL, envInteger("SVMP_FSILS_DUMP_SYSTEM_SKIP", 0));
    const long long max_dumps = std::max(0LL, envInteger("SVMP_FSILS_DUMP_SYSTEM_MAX", 16));
    if (attempt < skip || attempt >= skip + max_dumps) {
        return;
    }

    const auto shared = A.shared();
    if (!shared) {
        return;
    }
    const auto& lhs = shared->lhs;
    if (lhs.commu.nTasks != 1) {
        return;
    }
    const int dof = A.fsilsDof();
    const int nNo = lhs.nNo;
    if (dof <= 0 || nNo <= 0) {
        return;
    }

    // Internal (FSILS) node order -> backend global node id.
    std::vector<int> internal_to_global(static_cast<std::size_t>(nNo), -1);
    std::vector<int> global_to_old(static_cast<std::size_t>(nNo), -1);
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        const int global = shared->oldToGlobalNode(old);
        if (internal < 0 || internal >= nNo || global < 0 || global >= nNo) {
            FE_LOG_INFO("maybeDumpFsilsSystem: skipped (non-contiguous node numbering)");
            return;
        }
        internal_to_global[static_cast<std::size_t>(internal)] = global;
        global_to_old[static_cast<std::size_t>(global)] = old;
    }
    std::vector<int> global_to_internal(static_cast<std::size_t>(nNo), -1);
    for (int internal = 0; internal < nNo; ++internal) {
        global_to_internal[static_cast<std::size_t>(internal_to_global[static_cast<std::size_t>(internal)])] =
            internal;
    }

    FsilsSystemSnapshot s;
    s.dof = dof;
    s.n_nodes = nNo;
    const std::size_t block = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    const Real* values = A.fsilsValuesPtr();
    s.row_ptr.assign(static_cast<std::size_t>(nNo) + 1u, 0);
    std::vector<std::pair<int, int>> row_entries;
    for (int g = 0; g < nNo; ++g) {
        const int r = global_to_internal[static_cast<std::size_t>(g)];
        row_entries.clear();
        for (int nz = lhs.rowPtr(0, r); nz <= lhs.rowPtr(1, r); ++nz) {
            const int c = lhs.colPtr(nz);
            if (c < 0 || c >= nNo) {
                continue;
            }
            row_entries.emplace_back(internal_to_global[static_cast<std::size_t>(c)], nz);
        }
        std::sort(row_entries.begin(), row_entries.end());
        for (const auto& [gc, nz] : row_entries) {
            s.cols.push_back(gc);
            const Real* src = values + static_cast<std::size_t>(nz) * block;
            s.values.insert(s.values.end(), src, src + block);
        }
        s.row_ptr[static_cast<std::size_t>(g) + 1u] = static_cast<std::int64_t>(s.cols.size());
    }

    const auto& b_data = b.data();
    const auto& x_data = x.data();
    s.rhs.assign(static_cast<std::size_t>(nNo) * static_cast<std::size_t>(dof), 0.0);
    s.solution.assign(s.rhs.size(), 0.0);
    for (int g = 0; g < nNo; ++g) {
        const int old = global_to_old[static_cast<std::size_t>(g)];
        for (int c = 0; c < dof; ++c) {
            const auto dst = static_cast<std::size_t>(g) * static_cast<std::size_t>(dof) +
                             static_cast<std::size_t>(c);
            const auto src = static_cast<std::size_t>(old) * static_cast<std::size_t>(dof) +
                             static_cast<std::size_t>(c);
            s.rhs[dst] = b_data[src];
            s.solution[dst] = x_data[src];
        }
    }

    for (const auto fe_dof : dirichlet_fe_dofs) {
        GlobalIndex backend_dof = fe_dof;
        if (shared->dof_permutation) {
            const auto idx = static_cast<std::size_t>(fe_dof);
            if (idx < shared->dof_permutation->forward.size()) {
                backend_dof = shared->dof_permutation->forward[idx];
            }
        }
        if (backend_dof >= 0) {
            s.dirichlet_dofs.push_back(static_cast<std::int64_t>(backend_dof));
        }
    }
    std::sort(s.dirichlet_dofs.begin(), s.dirichlet_dofs.end());

    s.method = static_cast<int>(options.method);
    s.preconditioner = static_cast<int>(options.preconditioner);
    s.use_rcs = options.fsils_use_rcs ? 1 : 0;
    s.max_iter = options.max_iter;
    s.krylov_dim = options.krylov_dim;
    s.rel_tol = options.rel_tol;
    s.abs_tol = options.abs_tol;
    s.iterations = report.iterations;
    s.converged = report.converged ? 1 : 0;
    s.native_update_count = native_update_count;
    s.initial_residual_norm = report.initial_residual_norm;
    s.final_residual_norm = report.final_residual_norm;
    s.solve_seconds = solve_seconds;
    if (options.block_layout.has_value()) {
        s.blocks = options.block_layout->blocks;
    }

    std::ostringstream path;
    path << prefix << "." << attempt << ".bin";
    try {
        writeFsilsSystemSnapshot(s, path.str());
        std::ostringstream oss;
        oss << "maybeDumpFsilsSystem: wrote '" << path.str() << "' nodes=" << nNo << " dof=" << dof
            << " nnz_blocks=" << s.nnzBlocks() << " dirichlet=" << s.dirichlet_dofs.size()
            << " iterations=" << report.iterations;
        FE_LOG_INFO(oss.str());
    } catch (const std::exception& e) {
        FE_LOG_INFO(std::string("maybeDumpFsilsSystem: ") + e.what());
    }
}

} // namespace backends
} // namespace FE
} // namespace svmp
