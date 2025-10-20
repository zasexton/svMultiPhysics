CellTopology Test Plan (Future Additions)
========================================

Overview
--------
Current tests cover counts, canonical sorting, oriented edge-cancellation, face→edge consistency, variable-arity basic properties, and high‑order node count formulas for several families. This plan lists additional tests to thoroughly validate vertex, edge, face, and high‑order orderings against the intended conventions and VTK expectations.

Gaps And Proposed Tests
-----------------------
1) Oriented Faces: exact sequences per family
- Validate get_oriented_boundary_faces_view() returns exactly the documented vertex loops (right‑hand rule), not just sets.
- Families: Triangle, Quad, Tetra, Hex, Wedge, Pyramid.

2) Edges View: exact edge pair sequences per family
- Validate get_edges_view() equals the canonical edge tables (flat [v0,v1] sequence), not just derived mapping.
- Families: Triangle, Quad, Tetra, Hex, Wedge, Pyramid.

3) Face List Order
- Verify the ordering of faces in the view matches the documented family order:
  - Hex: bottom, top, then four sides
  - Wedge: bottom tri, top tri, then three quads
  - Pyramid: base quad, then four tri sides
- Check alignment between oriented and canonical views for the same face indices (canonical == sorted oriented vertices face‑wise).

4) Face→Edge Loop Order
- Strengthen current mapping test by asserting edge indices listed per face follow the loop around the face (cyclic order), not just set equality.
- Already partially covered via derived order; make explicit.

5) 2D Orientation (CCW)
- Validate 2D (Triangle, Quad) oriented boundary edges are CCW for the reference element (by explicit sequence checks).

6) High‑Order Pattern: role grouping and per‑face ordering
- Validate role sequence ordering, not just counts:
  - Corners first, by ascending corner id.
  - Edges next, grouped by edge index, each with k=1..p−1.
  - Faces next, grouped by face index; for tri faces lexicographic barycentric (i, j), for quad faces row‑major (i then j).
  - Volumes last, with expected layered/barycentric patterns.
- Suggested families/p: Tetra p=3 (tri faces only), Hex p=3 (quad faces), Wedge p=3 (mixture of tri/quad faces), Pyramid p=3 (mixture + layered volume).

7) Variable‑Arity: stronger coverage
- Prism(m) and Pyramid(m): add checks for m=3 (equivalence with Wedge/Pyramid fixed) and additional m (e.g., m=4,5) for face list order and face→edge loop order.

8) Thread‑Safety Note (Non‑unit)
- Variable‑arity caches are not thread‑safe on first touch; document expected single‑thread construction or add a small stress test harness if desired (not a gtest).

References
----------
- VTK cell documentation (vtkTetra, vtkHexahedron, vtkWedge, vtkPyramid, vtkTriangle, vtkQuad).
- Comments and static tables in CellTopology.cpp oriented/canonical definitions.
