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

#ifndef SVMP_MESH_FIELD_DESCRIPTOR_H
#define SVMP_MESH_FIELD_DESCRIPTOR_H

#include "../Core/MeshTypes.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace svmp {

// Forward declaration
class MeshBase;

// ====================
// P0 #3: Field Semantics & Metadata
// ====================
// Extends the basic field attachment system with rich metadata for multiphysics.
// Allows each physics module to attach self-describing data without touching Mesh internals.

enum class FieldIntent {
  ReadOnly,    // Input data (e.g., material properties, boundary conditions)
  ReadWrite,   // Solution or auxiliary fields modified during solve
  Temporary    // Scratch space, not saved to disk
};

enum class FieldGhostPolicy {
  None,        // No ghost exchange needed (purely local data)
  Exchange,    // Direct ghost exchange (copy values to neighbors)
  Accumulate   // Accumulate contributions from neighbors (e.g., gradients, fluxes)
};

// Self-describing field metadata
struct FieldDescriptor {
  EntityKind location = EntityKind::Vertex;        // Where the field lives
  size_t components = 1;                            // Number of components (1=scalar, 3=vector, 9=tensor, etc.)
  std::vector<std::string> component_names;         // Optional: {"x", "y", "z"} or {"xx", "xy", "xz", ...}
  std::string units;                                // Physical units: "m/s", "Pa", "K", "J/kg", etc.
  real_t unit_scale = 1.0;                          // Scaling factor (e.g., 1e6 for MPa -> Pa)
  bool time_dependent = false;                      // Does this field evolve in time?
  FieldIntent intent = FieldIntent::ReadWrite;      // How the field is used
  FieldGhostPolicy ghost_policy = FieldGhostPolicy::None; // MPI ghost exchange behavior
  std::string description;                          // Human-readable description

  // Factory methods for common field types
  static FieldDescriptor scalar(EntityKind loc, const std::string& units = "", bool time_dep = false) {
    FieldDescriptor d;
    d.location = loc;
    d.components = 1;
    d.units = units;
    d.time_dependent = time_dep;
    return d;
  }

  static FieldDescriptor vector(EntityKind loc, int dim, const std::string& units = "", bool time_dep = false) {
    FieldDescriptor d;
    d.location = loc;
    d.components = static_cast<size_t>(dim);
    d.units = units;
    d.time_dependent = time_dep;
    if (dim == 2) d.component_names = {"x", "y"};
    else if (dim == 3) d.component_names = {"x", "y", "z"};
    return d;
  }

  static FieldDescriptor tensor(EntityKind loc, int dim, const std::string& units = "", bool time_dep = false) {
    FieldDescriptor d;
    d.location = loc;
    d.components = static_cast<size_t>(dim * dim);
    d.units = units;
    d.time_dependent = time_dep;
    if (dim == 2) {
      d.component_names = {"xx", "xy", "yx", "yy"};
    } else if (dim == 3) {
      d.component_names = {"xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"};
    }
    return d;
  }

  static FieldDescriptor symtensor(EntityKind loc, int dim, const std::string& units = "", bool time_dep = false) {
    FieldDescriptor d;
    d.location = loc;
    d.components = static_cast<size_t>(dim * (dim + 1) / 2); // Voigt notation
    d.units = units;
    d.time_dependent = time_dep;
    if (dim == 2) {
      d.component_names = {"xx", "yy", "xy"};
    } else if (dim == 3) {
      d.component_names = {"xx", "yy", "zz", "yz", "xz", "xy"};
    }
    return d;
  }
};

// ====================
// Extended Field Manager (optional helper)
// ====================
// Wraps MeshBase field attachment with metadata tracking.
// Physics codes can use this to maintain a registry of their fields with full semantics.

class FieldManager {
public:
  explicit FieldManager(MeshBase& mesh) : mesh_(mesh) {}

  // Attach a field with full metadata
  FieldHandle attach(const std::string& name, FieldScalarType type, const FieldDescriptor& descriptor);

  // Query metadata
  bool has_descriptor(const FieldHandle& h) const {
    return descriptors_.find(h.id) != descriptors_.end();
  }

  const FieldDescriptor& descriptor(const FieldHandle& h) const {
    auto it = descriptors_.find(h.id);
    if (it == descriptors_.end()) {
      throw std::runtime_error("FieldManager: no descriptor for field " + h.name);
    }
    return it->second;
  }

  // Query by intent
  std::vector<FieldHandle> fields_with_intent(FieldIntent intent) const {
    std::vector<FieldHandle> result;
    for (const auto& [id, desc] : descriptors_) {
      if (desc.intent == intent) {
        // Reconstruct handle (needs name lookup from mesh)
        // This is a simplified version; in practice, you'd cache handles
        result.push_back({id, desc.location, ""});
      }
    }
    return result;
  }

  // Query time-dependent fields (for checkpointing, output)
  std::vector<FieldHandle> time_dependent_fields() const {
    std::vector<FieldHandle> result;
    for (const auto& [id, desc] : descriptors_) {
      if (desc.time_dependent) {
        result.push_back({id, desc.location, ""});
      }
    }
    return result;
  }

  // Query fields requiring ghost exchange
  std::vector<FieldHandle> fields_requiring_exchange() const {
    std::vector<FieldHandle> result;
    for (const auto& [id, desc] : descriptors_) {
      if (desc.ghost_policy != FieldGhostPolicy::None) {
        result.push_back({id, desc.location, ""});
      }
    }
    return result;
  }

private:
  MeshBase& mesh_;
  std::unordered_map<uint32_t, FieldDescriptor> descriptors_;
};

} // namespace svmp

#endif // SVMP_MESH_FIELD_DESCRIPTOR_H
