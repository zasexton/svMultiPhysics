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

#ifndef SVMP_GEOMETRY_CACHE_H
#define SVMP_GEOMETRY_CACHE_H

#include "../Core/MeshTypes.h"
#include "../Observer/MeshObserver.h"
#include "BoundingVolume.h"
#include <vector>
#include <array>
#include <memory>
#include <optional>

namespace svmp {

// Forward declarations
class MeshBase;

/**
 * @brief Lazy cache for geometric quantities (centers, measures, bounding boxes)
 *
 * This cache stores commonly-accessed geometric quantities to avoid recomputation.
 * It automatically invalidates itself when the mesh topology or geometry changes
 * by subscribing to mesh events via the Observer bus.
 *
 * **Design Philosophy:**
 * - Lazy evaluation: quantities are computed on-demand and cached
 * - Automatic invalidation: subscribes to MeshEvent::TopologyChanged and
 *   MeshEvent::GeometryChanged via the Observer pattern
 * - Configuration-aware: caches for both Reference and Current configurations
 * - Optional: users can bypass caching by calling MeshGeometry directly
 *
 * **Thread Safety:**
 * - Not thread-safe: use per-thread instances or external synchronization
 *
 * **Memory vs Performance Trade-off:**
 * - Caching trades memory for speed on repeated queries
 * - For one-time queries, MeshGeometry is more efficient
 * - For iterative algorithms (adaptivity, smoothing), caching wins
 */
class GeometryCache : public MeshObserver {
public:
    /**
     * @brief Cache configuration flags
     */
    struct CacheConfig {
        bool enable_cell_centers{true};
        bool enable_cell_measures{true};
        bool enable_cell_bboxes{true};
        bool enable_face_centers{false};   // Less commonly used
        bool enable_face_normals{false};
        bool enable_face_areas{false};
        bool enable_edge_centers{false};
        bool enable_mesh_bbox{true};

        // Configuration to cache (Reference, Current, or both)
        bool cache_reference{true};
        bool cache_current{false};
    };

    /**
     * @brief Construct cache for a mesh with configuration
     * @param mesh The mesh to cache geometry for
     * @param config Cache configuration
     */
    explicit GeometryCache(const MeshBase& mesh, const CacheConfig& config = CacheConfig{});

    ~GeometryCache() override = default;

    // ---- MeshObserver interface ----

    void on_mesh_event(MeshEvent event) override;
    const char* observer_name() const override { return "GeometryCache"; }

    // ---- Cell queries ----

    /**
     * @brief Get cached cell center (lazy evaluation)
     * @param cell Cell index
     * @param cfg Configuration
     * @return Cell center coordinates
     */
    std::array<real_t, 3> cell_center(index_t cell, Configuration cfg = Configuration::Reference);

    /**
     * @brief Get cached cell measure (lazy evaluation)
     * @param cell Cell index
     * @param cfg Configuration
     * @return Cell measure (length/area/volume)
     */
    real_t cell_measure(index_t cell, Configuration cfg = Configuration::Reference);

    /**
     * @brief Get cached cell bounding box (lazy evaluation)
     * @param cell Cell index
     * @param cfg Configuration
     * @return Cell bounding box
     */
    AABB cell_bounding_box(index_t cell, Configuration cfg = Configuration::Reference);

    // ---- Face queries ----

    /**
     * @brief Get cached face center (lazy evaluation)
     * @param face Face index
     * @param cfg Configuration
     * @return Face center coordinates
     */
    std::array<real_t, 3> face_center(index_t face, Configuration cfg = Configuration::Reference);

    /**
     * @brief Get cached face normal (lazy evaluation)
     * @param face Face index
     * @param cfg Configuration
     * @return Unit normal vector
     */
    std::array<real_t, 3> face_normal(index_t face, Configuration cfg = Configuration::Reference);

    /**
     * @brief Get cached face area (lazy evaluation)
     * @param face Face index
     * @param cfg Configuration
     * @return Face area
     */
    real_t face_area(index_t face, Configuration cfg = Configuration::Reference);

    // ---- Edge queries ----

    /**
     * @brief Get cached edge center (lazy evaluation)
     * @param edge Edge index
     * @param cfg Configuration
     * @return Edge midpoint coordinates
     */
    std::array<real_t, 3> edge_center(index_t edge, Configuration cfg = Configuration::Reference);

    // ---- Mesh-wide queries ----

    /**
     * @brief Get cached mesh bounding box
     * @param cfg Configuration
     * @return Mesh bounding box
     */
    AABB mesh_bounding_box(Configuration cfg = Configuration::Reference);

    /**
     * @brief Compute total mesh volume (sum of cell measures)
     * @param cfg Configuration
     * @return Total volume
     */
    real_t total_volume(Configuration cfg = Configuration::Reference);

    // ---- Cache management ----

    /**
     * @brief Explicitly invalidate all caches
     */
    void invalidate_all();

    /**
     * @brief Invalidate caches for a specific configuration
     * @param cfg Configuration to invalidate
     */
    void invalidate_configuration(Configuration cfg);

    /**
     * @brief Pre-populate (warm) all enabled caches
     * @param cfg Configuration to warm
     *
     * Useful before performance-critical loops to avoid lazy evaluation overhead.
     */
    void warm_cache(Configuration cfg = Configuration::Reference);

    /**
     * @brief Get cache statistics (hit rate, memory usage)
     */
    struct CacheStats {
        size_t cell_center_hits{0};
        size_t cell_center_misses{0};
        size_t cell_measure_hits{0};
        size_t cell_measure_misses{0};
        size_t memory_bytes{0};
    };

    CacheStats get_stats() const { return stats_; }

    /**
     * @brief Clear cache statistics
     */
    void reset_stats() { stats_ = CacheStats{}; }

    /**
     * @brief Get cache configuration
     */
    const CacheConfig& config() const { return config_; }

private:
    const MeshBase& mesh_;
    CacheConfig config_;
    mutable CacheStats stats_;

    // ---- Reference configuration caches ----
    struct RefCaches {
        std::vector<std::optional<std::array<real_t, 3>>> cell_centers;
        std::vector<std::optional<real_t>> cell_measures;
        std::vector<std::optional<AABB>> cell_bboxes;
        std::vector<std::optional<std::array<real_t, 3>>> face_centers;
        std::vector<std::optional<std::array<real_t, 3>>> face_normals;
        std::vector<std::optional<real_t>> face_areas;
        std::vector<std::optional<std::array<real_t, 3>>> edge_centers;
        std::optional<AABB> mesh_bbox;
        bool valid{false};
    };

    // ---- Current configuration caches ----
    struct CurCaches {
        std::vector<std::optional<std::array<real_t, 3>>> cell_centers;
        std::vector<std::optional<real_t>> cell_measures;
        std::vector<std::optional<AABB>> cell_bboxes;
        std::vector<std::optional<std::array<real_t, 3>>> face_centers;
        std::vector<std::optional<std::array<real_t, 3>>> face_normals;
        std::vector<std::optional<real_t>> face_areas;
        std::vector<std::optional<std::array<real_t, 3>>> edge_centers;
        std::optional<AABB> mesh_bbox;
        bool valid{false};
    };

    mutable RefCaches ref_cache_;
    mutable CurCaches cur_cache_;

    // ---- Helper methods ----

    void ensure_cache_capacity();
    void invalidate_ref_cache();
    void invalidate_cur_cache();

    template<typename T>
    const T& get_or_compute(
        std::vector<std::optional<T>>& cache,
        index_t index,
        std::function<T()> compute_fn) const;
};

/**
 * @brief Cache invalidator observer (simplified standalone version)
 *
 * Standalone observer that invalidates a GeometryCache on relevant events.
 * Can be used to wire multiple caches to a mesh.
 */
class CacheInvalidator : public MeshObserver {
public:
    explicit CacheInvalidator(GeometryCache& cache) : cache_(cache) {}

    void on_mesh_event(MeshEvent event) override {
        if (event == MeshEvent::TopologyChanged ||
            event == MeshEvent::GeometryChanged ||
            event == MeshEvent::AdaptivityApplied) {
            cache_.invalidate_all();
        }
    }

    const char* observer_name() const override { return "CacheInvalidator"; }

private:
    GeometryCache& cache_;
};

} // namespace svmp

#endif // SVMP_GEOMETRY_CACHE_H
