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

#include "Mesh.h"

#include <stdexcept>

#include "ComMod.h"    // mshType
#include "vtk_xml.h"   // vtk_xml::read_vtu

namespace svmp {

Mesh::Mesh()
  : owned_(new mshType()), view_(nullptr), owns_(true)
{}

Mesh Mesh::Wrap(const mshType& legacy)
{
  Mesh m;
  m.owned_.reset();
  m.view_ = &legacy;
  m.owns_ = false;
  return m;
}

Mesh Mesh::FromVtu(const std::string& file_path)
{
  Mesh m;
  // Read mesh geometry/topology into the owned legacy container.
  vtk_xml::read_vtu(file_path, m.ref_());
  // Note: element props (Gauss, etc.) may be set later by callers using existing utilities.
  // This keeps Mesh minimally coupled.
  return m;
}

std::string Mesh::name() const
{
  return ref_().name;
}

void Mesh::set_name(const std::string& n)
{
  ref_().name = n;
}

int Mesh::spatial_dimension() const
{
  const auto& x = ref_().x;
  return x.size() == 0 ? 0 : x.rows();
}

int Mesh::nodes_per_element() const
{
  return ref_().eNoN;
}

int Mesh::num_nodes() const
{
  // Prefer local nNo if set; otherwise fallback to global gnNo.
  const auto& r = ref_();
  return (r.nNo > 0) ? r.nNo : r.gnNo;
}

int Mesh::num_elements() const
{
  const auto& r = ref_();
  return (r.nEl > 0) ? r.nEl : r.gnEl;
}

consts::ElementType Mesh::element_type() const
{
  return ref_().eType;
}

const Array<double>& Mesh::coordinates() const
{
  return ref_().x;
}

const Array<int>& Mesh::connectivity() const
{
  return ref_().IEN;
}

const mshType& Mesh::legacy() const
{
  return ref_();
}

mshType& Mesh::legacy_mut()
{
  return ref_();
}

const mshType& Mesh::ref_() const
{
  if (owns_) {
    return *owned_;
  }
  return *view_;
}

mshType& Mesh::ref_()
{
  if (!owns_) {
    // This is a view; we cannot mutate non-owned data safely here.
    // If mutation is absolutely required, callers should copy or construct owning Mesh.
    throw std::runtime_error("Attempted to mutate a non-owning Mesh view.");
  }
  return *owned_;
}

} // namespace svmp

