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

#include "MeshFieldDescriptor.h"
#include "../Core/MeshBase.h"

#include <algorithm>

namespace svmp {

namespace {

bool handle_less(const FieldHandle& a, const FieldHandle& b) {
  const int ak = static_cast<int>(a.kind);
  const int bk = static_cast<int>(b.kind);
  if (ak != bk) return ak < bk;
  return a.name < b.name;
}

template <typename Pred>
std::vector<FieldHandle> scan_fields_with_descriptor(const MeshBase& mesh, Pred pred) {
  std::vector<FieldHandle> result;
  for (int k = 0; k < 4; ++k) {
    const auto kind = static_cast<EntityKind>(k);
    for (const auto& name : mesh.field_names(kind)) {
      const FieldHandle h = mesh.field_handle(kind, name);
      if (h.id == 0) continue;
      const FieldDescriptor* desc = mesh.field_descriptor(h);
      if (!desc) continue;
      if (pred(*desc)) {
        result.push_back(h);
      }
    }
  }
  std::sort(result.begin(), result.end(), handle_less);
  return result;
}

} // namespace

FieldHandle FieldManager::attach(const std::string& name,
                                 FieldScalarType type,
                                 const FieldDescriptor& descriptor) {
  return mesh_.attach_field_with_descriptor(descriptor.location, name, type, descriptor);
}

bool FieldManager::has_descriptor(const FieldHandle& h) const {
  const FieldHandle current = mesh_.field_handle(h.kind, h.name);
  if (current.id == 0 || current.id != h.id) {
    return false;
  }
  return mesh_.field_descriptor(current) != nullptr;
}

const FieldDescriptor& FieldManager::descriptor(const FieldHandle& h) const {
  const FieldHandle current = mesh_.field_handle(h.kind, h.name);
  if (current.id == 0 || current.id != h.id) {
    throw std::runtime_error("FieldManager: stale or invalid field handle for field '" + h.name + "'");
  }
  const FieldDescriptor* desc = mesh_.field_descriptor(current);
  if (!desc) {
    throw std::runtime_error("FieldManager: no descriptor for field " + h.name);
  }
  return *desc;
}

std::vector<FieldHandle> FieldManager::fields_with_intent(FieldIntent intent) const {
  return scan_fields_with_descriptor(mesh_, [&](const FieldDescriptor& d) { return d.intent == intent; });
}

std::vector<FieldHandle> FieldManager::time_dependent_fields() const {
  return scan_fields_with_descriptor(mesh_, [&](const FieldDescriptor& d) { return d.time_dependent; });
}

std::vector<FieldHandle> FieldManager::fields_requiring_exchange() const {
  return scan_fields_with_descriptor(mesh_, [&](const FieldDescriptor& d) { return d.ghost_policy != FieldGhostPolicy::None; });
}

} // namespace svmp
