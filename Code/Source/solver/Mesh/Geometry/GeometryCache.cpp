/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "GeometryCache.h"
#include "../Core/MeshBase.h"
#include "MeshGeometry.h"
#include <type_traits>

namespace svmp {

//=============================================================================
// GeometryCache Implementation
//=============================================================================

GeometryCache::GeometryCache(const MeshBase& mesh)
    : GeometryCache(mesh, CacheConfig{}) {}

GeometryCache::GeometryCache(const MeshBase& mesh, const CacheConfig& config)
    : mesh_(mesh), config_(config) {
    ensure_cache_capacity();
    subscription_ = ScopedSubscription(&mesh_.event_bus(), this);
}

void GeometryCache::reset_stats() {
    const size_t bytes = cache_memory_bytes();
    stats_ = CacheStats{};
    stats_.memory_bytes = bytes;
}

void GeometryCache::on_mesh_event(MeshEvent event) {
    // Invalidate on topology or geometry changes
    if (event == MeshEvent::TopologyChanged) {
        invalidate_all();
        ensure_cache_capacity(); // Resize caches if topology changed
    } else if (event == MeshEvent::GeometryChanged) {
        // Without configuration-specific events, conservatively invalidate both.
        invalidate_all();
    } else if (event == MeshEvent::AdaptivityApplied) {
        // Adaptivity changes both topology and geometry
        invalidate_all();
        ensure_cache_capacity();
    }
}

//=============================================================================
// Cell Queries
//=============================================================================

std::array<real_t, 3> GeometryCache::cell_center(index_t cell, Configuration cfg) {
    if (!config_.enable_cell_centers) {
        return MeshGeometry::cell_center(mesh_, cell, cfg);
    }

    if (cfg == Configuration::Reference && config_.cache_reference) {
        return get_or_compute(ref_cache_.cell_centers, cell,
                              stats_.cell_center_hits, stats_.cell_center_misses,
                              [&]() { return MeshGeometry::cell_center(mesh_, cell, cfg); });
    } else if ((cfg == Configuration::Current || cfg == Configuration::Deformed) &&
               config_.cache_current) {
        return get_or_compute(cur_cache_.cell_centers, cell,
                              stats_.cell_center_hits, stats_.cell_center_misses,
                              [&]() { return MeshGeometry::cell_center(mesh_, cell, cfg); });
    }

    return MeshGeometry::cell_center(mesh_, cell, cfg);
}

real_t GeometryCache::cell_measure(index_t cell, Configuration cfg) {
    if (!config_.enable_cell_measures) {
        return MeshGeometry::cell_measure(mesh_, cell, cfg);
    }

    if (cfg == Configuration::Reference && config_.cache_reference) {
        return get_or_compute(ref_cache_.cell_measures, cell,
                              stats_.cell_measure_hits, stats_.cell_measure_misses,
                              [&]() { return MeshGeometry::cell_measure(mesh_, cell, cfg); });
    } else if ((cfg == Configuration::Current || cfg == Configuration::Deformed) &&
               config_.cache_current) {
        return get_or_compute(cur_cache_.cell_measures, cell,
                              stats_.cell_measure_hits, stats_.cell_measure_misses,
                              [&]() { return MeshGeometry::cell_measure(mesh_, cell, cfg); });
    }

    return MeshGeometry::cell_measure(mesh_, cell, cfg);
}

AABB GeometryCache::cell_bounding_box(index_t cell, Configuration cfg) {
    auto to_aabb = [](const BoundingBox& bb) -> AABB { return AABB(bb.min, bb.max); };

    if (!config_.enable_cell_bboxes) {
        return to_aabb(MeshGeometry::cell_bounding_box(mesh_, cell, cfg));
    }

    if (cfg == Configuration::Reference && config_.cache_reference) {
        return get_or_compute(ref_cache_.cell_bboxes, cell,
                              stats_.cell_bbox_hits, stats_.cell_bbox_misses,
                              [&]() { return to_aabb(MeshGeometry::cell_bounding_box(mesh_, cell, cfg)); });
    } else if ((cfg == Configuration::Current || cfg == Configuration::Deformed) &&
               config_.cache_current) {
        return get_or_compute(cur_cache_.cell_bboxes, cell,
                              stats_.cell_bbox_hits, stats_.cell_bbox_misses,
                              [&]() { return to_aabb(MeshGeometry::cell_bounding_box(mesh_, cell, cfg)); });
    }

    return to_aabb(MeshGeometry::cell_bounding_box(mesh_, cell, cfg));
}

//=============================================================================
// Face Queries
//=============================================================================

std::array<real_t, 3> GeometryCache::face_center(index_t face, Configuration cfg) {
    if (!config_.enable_face_centers) {
        return MeshGeometry::face_center(mesh_, face, cfg);
    }

    if (cfg == Configuration::Reference && config_.cache_reference) {
        return get_or_compute(ref_cache_.face_centers, face,
                              stats_.face_center_hits, stats_.face_center_misses,
                              [&]() { return MeshGeometry::face_center(mesh_, face, cfg); });
    } else if ((cfg == Configuration::Current || cfg == Configuration::Deformed) &&
               config_.cache_current) {
        return get_or_compute(cur_cache_.face_centers, face,
                              stats_.face_center_hits, stats_.face_center_misses,
                              [&]() { return MeshGeometry::face_center(mesh_, face, cfg); });
    }

    return MeshGeometry::face_center(mesh_, face, cfg);
}

std::array<real_t, 3> GeometryCache::face_normal(index_t face, Configuration cfg) {
    if (!config_.enable_face_normals) {
        return MeshGeometry::face_normal(mesh_, face, cfg);
    }

    if (cfg == Configuration::Reference && config_.cache_reference) {
        return get_or_compute(ref_cache_.face_normals, face,
                              stats_.face_normal_hits, stats_.face_normal_misses,
                              [&]() { return MeshGeometry::face_normal(mesh_, face, cfg); });
    } else if ((cfg == Configuration::Current || cfg == Configuration::Deformed) &&
               config_.cache_current) {
        return get_or_compute(cur_cache_.face_normals, face,
                              stats_.face_normal_hits, stats_.face_normal_misses,
                              [&]() { return MeshGeometry::face_normal(mesh_, face, cfg); });
    }

    return MeshGeometry::face_normal(mesh_, face, cfg);
}

real_t GeometryCache::face_area(index_t face, Configuration cfg) {
    if (!config_.enable_face_areas) {
        return MeshGeometry::face_area(mesh_, face, cfg);
    }

    if (cfg == Configuration::Reference && config_.cache_reference) {
        return get_or_compute(ref_cache_.face_areas, face,
                              stats_.face_area_hits, stats_.face_area_misses,
                              [&]() { return MeshGeometry::face_area(mesh_, face, cfg); });
    } else if ((cfg == Configuration::Current || cfg == Configuration::Deformed) &&
               config_.cache_current) {
        return get_or_compute(cur_cache_.face_areas, face,
                              stats_.face_area_hits, stats_.face_area_misses,
                              [&]() { return MeshGeometry::face_area(mesh_, face, cfg); });
    }

    return MeshGeometry::face_area(mesh_, face, cfg);
}

//=============================================================================
// Edge Queries
//=============================================================================

std::array<real_t, 3> GeometryCache::edge_center(index_t edge, Configuration cfg) {
    if (!config_.enable_edge_centers) {
        return MeshGeometry::edge_center(mesh_, edge, cfg);
    }

    if (cfg == Configuration::Reference && config_.cache_reference) {
        return get_or_compute(ref_cache_.edge_centers, edge,
                              stats_.edge_center_hits, stats_.edge_center_misses,
                              [&]() { return MeshGeometry::edge_center(mesh_, edge, cfg); });
    } else if ((cfg == Configuration::Current || cfg == Configuration::Deformed) &&
               config_.cache_current) {
        return get_or_compute(cur_cache_.edge_centers, edge,
                              stats_.edge_center_hits, stats_.edge_center_misses,
                              [&]() { return MeshGeometry::edge_center(mesh_, edge, cfg); });
    }

    return MeshGeometry::edge_center(mesh_, edge, cfg);
}

//=============================================================================
// Mesh-wide Queries
//=============================================================================

AABB GeometryCache::mesh_bounding_box(Configuration cfg) {
    auto to_aabb = [](const BoundingBox& bb) -> AABB { return AABB(bb.min, bb.max); };

    if (!config_.enable_mesh_bbox) {
        return to_aabb(MeshGeometry::bounding_box(mesh_, cfg));
    }

    if (cfg == Configuration::Reference && config_.cache_reference) {
        if (!ref_cache_.mesh_bbox.has_value()) {
            ++stats_.mesh_bbox_misses;
            ref_cache_.mesh_bbox = to_aabb(MeshGeometry::bounding_box(mesh_, cfg));
        } else {
            ++stats_.mesh_bbox_hits;
        }
        return ref_cache_.mesh_bbox.value();
    } else if ((cfg == Configuration::Current || cfg == Configuration::Deformed) &&
               config_.cache_current) {
        if (!cur_cache_.mesh_bbox.has_value()) {
            ++stats_.mesh_bbox_misses;
            cur_cache_.mesh_bbox = to_aabb(MeshGeometry::bounding_box(mesh_, cfg));
        } else {
            ++stats_.mesh_bbox_hits;
        }
        return cur_cache_.mesh_bbox.value();
    }

    return to_aabb(MeshGeometry::bounding_box(mesh_, cfg));
}

real_t GeometryCache::total_volume(Configuration cfg) {
    // Sum all cell measures
    real_t volume = 0.0;
    const index_t n_cells = mesh_.n_cells();

    for (index_t cell = 0; cell < n_cells; ++cell) {
        volume += cell_measure(cell, cfg);
    }

    return volume;
}

//=============================================================================
// Cache Management
//=============================================================================

void GeometryCache::invalidate_all() {
    invalidate_ref_cache();
    invalidate_cur_cache();
}

void GeometryCache::invalidate_configuration(Configuration cfg) {
    if (cfg == Configuration::Reference) {
        invalidate_ref_cache();
    } else {
        invalidate_cur_cache();
    }
}

void GeometryCache::warm_cache(Configuration cfg) {
    if (cfg == Configuration::Reference && config_.cache_reference) {
        const index_t n_cells = mesh_.n_cells();
        const index_t n_faces = mesh_.n_faces();
        const index_t n_edges = mesh_.n_edges();

        if (config_.enable_cell_centers) {
            for (index_t i = 0; i < n_cells; ++i) {
                cell_center(i, cfg);
            }
        }

        if (config_.enable_cell_measures) {
            for (index_t i = 0; i < n_cells; ++i) {
                cell_measure(i, cfg);
            }
        }

        if (config_.enable_cell_bboxes) {
            for (index_t i = 0; i < n_cells; ++i) {
                cell_bounding_box(i, cfg);
            }
        }

        if (config_.enable_face_centers) {
            for (index_t i = 0; i < n_faces; ++i) {
                face_center(i, cfg);
            }
        }

        if (config_.enable_face_normals) {
            for (index_t i = 0; i < n_faces; ++i) {
                face_normal(i, cfg);
            }
        }

        if (config_.enable_face_areas) {
            for (index_t i = 0; i < n_faces; ++i) {
                face_area(i, cfg);
            }
        }

        if (config_.enable_edge_centers) {
            for (index_t i = 0; i < n_edges; ++i) {
                edge_center(i, cfg);
            }
        }

        if (config_.enable_mesh_bbox) {
            mesh_bounding_box(cfg);
        }

        ref_cache_.valid = true;

    } else if ((cfg == Configuration::Current || cfg == Configuration::Deformed) &&
               config_.cache_current) {
        const index_t n_cells = mesh_.n_cells();
        const index_t n_faces = mesh_.n_faces();
        const index_t n_edges = mesh_.n_edges();

        if (config_.enable_cell_centers) {
            for (index_t i = 0; i < n_cells; ++i) {
                cell_center(i, cfg);
            }
        }

        if (config_.enable_cell_measures) {
            for (index_t i = 0; i < n_cells; ++i) {
                cell_measure(i, cfg);
            }
        }

        if (config_.enable_cell_bboxes) {
            for (index_t i = 0; i < n_cells; ++i) {
                cell_bounding_box(i, cfg);
            }
        }

        if (config_.enable_face_centers) {
            for (index_t i = 0; i < n_faces; ++i) {
                face_center(i, cfg);
            }
        }

        if (config_.enable_face_normals) {
            for (index_t i = 0; i < n_faces; ++i) {
                face_normal(i, cfg);
            }
        }

        if (config_.enable_face_areas) {
            for (index_t i = 0; i < n_faces; ++i) {
                face_area(i, cfg);
            }
        }

        if (config_.enable_edge_centers) {
            for (index_t i = 0; i < n_edges; ++i) {
                edge_center(i, cfg);
            }
        }

        if (config_.enable_mesh_bbox) {
            mesh_bounding_box(cfg);
        }

        cur_cache_.valid = true;
    }
}

//=============================================================================
// Private Helpers
//=============================================================================

void GeometryCache::ensure_cache_capacity() {
    const index_t n_cells = mesh_.n_cells();
    const index_t n_faces = mesh_.n_faces();
    const index_t n_edges = mesh_.n_edges();

    // Resize reference caches
    if (config_.cache_reference) {
        if (config_.enable_cell_centers) ref_cache_.cell_centers.resize(n_cells);
        if (config_.enable_cell_measures) ref_cache_.cell_measures.resize(n_cells);
        if (config_.enable_cell_bboxes) ref_cache_.cell_bboxes.resize(n_cells);
        if (config_.enable_face_centers) ref_cache_.face_centers.resize(n_faces);
        if (config_.enable_face_normals) ref_cache_.face_normals.resize(n_faces);
        if (config_.enable_face_areas) ref_cache_.face_areas.resize(n_faces);
        if (config_.enable_edge_centers) ref_cache_.edge_centers.resize(n_edges);
    }

    // Resize current caches
    if (config_.cache_current) {
        if (config_.enable_cell_centers) cur_cache_.cell_centers.resize(n_cells);
        if (config_.enable_cell_measures) cur_cache_.cell_measures.resize(n_cells);
        if (config_.enable_cell_bboxes) cur_cache_.cell_bboxes.resize(n_cells);
        if (config_.enable_face_centers) cur_cache_.face_centers.resize(n_faces);
        if (config_.enable_face_normals) cur_cache_.face_normals.resize(n_faces);
        if (config_.enable_face_areas) cur_cache_.face_areas.resize(n_faces);
        if (config_.enable_edge_centers) cur_cache_.edge_centers.resize(n_edges);
    }

    stats_.memory_bytes = cache_memory_bytes();
}

size_t GeometryCache::cache_memory_bytes() const {
    size_t bytes = 0;
    auto add = [&bytes](const auto& v) {
        bytes += v.capacity() * sizeof(typename std::decay_t<decltype(v)>::value_type);
    };

    add(ref_cache_.cell_centers);
    add(ref_cache_.cell_measures);
    add(ref_cache_.cell_bboxes);
    add(ref_cache_.face_centers);
    add(ref_cache_.face_normals);
    add(ref_cache_.face_areas);
    add(ref_cache_.edge_centers);

    add(cur_cache_.cell_centers);
    add(cur_cache_.cell_measures);
    add(cur_cache_.cell_bboxes);
    add(cur_cache_.face_centers);
    add(cur_cache_.face_normals);
    add(cur_cache_.face_areas);
    add(cur_cache_.edge_centers);

    return bytes;
}

void GeometryCache::invalidate_ref_cache() {
    // Clear all optional values
    for (auto& opt : ref_cache_.cell_centers) opt.reset();
    for (auto& opt : ref_cache_.cell_measures) opt.reset();
    for (auto& opt : ref_cache_.cell_bboxes) opt.reset();
    for (auto& opt : ref_cache_.face_centers) opt.reset();
    for (auto& opt : ref_cache_.face_normals) opt.reset();
    for (auto& opt : ref_cache_.face_areas) opt.reset();
    for (auto& opt : ref_cache_.edge_centers) opt.reset();
    ref_cache_.mesh_bbox.reset();
    ref_cache_.valid = false;
}

void GeometryCache::invalidate_cur_cache() {
    // Clear all optional values
    for (auto& opt : cur_cache_.cell_centers) opt.reset();
    for (auto& opt : cur_cache_.cell_measures) opt.reset();
    for (auto& opt : cur_cache_.cell_bboxes) opt.reset();
    for (auto& opt : cur_cache_.face_centers) opt.reset();
    for (auto& opt : cur_cache_.face_normals) opt.reset();
    for (auto& opt : cur_cache_.face_areas) opt.reset();
    for (auto& opt : cur_cache_.edge_centers) opt.reset();
    cur_cache_.mesh_bbox.reset();
    cur_cache_.valid = false;
}

template<typename T, typename ComputeFn>
const T& GeometryCache::get_or_compute(
    std::vector<std::optional<T>>& cache,
    index_t index,
    size_t& hits,
    size_t& misses,
    ComputeFn&& compute_fn) const {

    if (index < 0 || index >= static_cast<index_t>(cache.size())) {
        static T default_value{};
        return default_value;
    }

    if (!cache[index].has_value()) {
        ++misses;
        cache[index] = compute_fn();
    } else {
        ++hits;
    }

    return cache[index].value();
}

} // namespace svmp
