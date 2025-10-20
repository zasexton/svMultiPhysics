Topology Ordering Conventions
=============================

This module centralizes canonical connectivity and ordering for standard finite element cell families.

Key Conventions
---------------
- Canonical faces: Within-face vertex indices are sorted (ascending). These are used for topology detection and stable keys.
- Oriented faces: Vertices are ordered to follow the right-hand rule so that the face normal points outward from the cell interior.
- Edges: Edge lists use local corner indices, ordered consistently with the oriented face loops.

Reference Ordering (VTK)
------------------------
Oriented face and edge definitions follow VTK’s cell documentation for standard families:
- Tetrahedron: vtkTetra
- Hexahedron: vtkHexahedron
- Wedge (Triangular Prism): vtkWedge
- Pyramid: vtkPyramid
- Triangle: vtkTriangle
- Quadrilateral: vtkQuad

High‑Order Nodes (Lagrange)
---------------------------
- Edges: For order p, each topology edge has p-1 interior edge nodes in increasing parametric order.
- Faces: Triangular faces enumerate barycentric interior nodes; quadrilateral faces enumerate a tensor grid.
- Pyramid volume nodes: Enumerated in layers in the vertical index k=1..p-2. At layer k, the in‑layer grid has 
  size n=(p+1-k) with (n-2)^2 strict interior points. The total volume nodes sum to (p-2)(p-1)(2p-3)/6.
- Total node count for a Lagrange pyramid (order p) is sum_{m=1}^{p+1} m^2 = (p+1)(p+2)(2p+3)/6.

Why This Matters
----------------
All mesh algorithms depending on face traversal, normal orientation, and high‑order connectivity 
(VTK I/O, geometry, BC application) rely on these conventions. Keeping definitions in one place provides 
consistency and simplifies validation.
