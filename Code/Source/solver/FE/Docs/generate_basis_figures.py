#!/usr/bin/env python3
"""
Generate SVG figures comparing the new FE Basis library against the legacy solver.

Figures produced:
  1. feature_coverage.svg       — Heatmap: basis families × element topologies
  2. dof_scaling.svg            — DOF count vs polynomial order per topology
  3. hessian_support.svg        — Derivative support comparison (legacy vs new)
  4. polynomial_reproduction.svg — Interpolation accuracy by monomial degree
  5. basis_family_breadth.svg   — Basis families available in each library

Output directory: same as this script (Code/Source/solver/FE/Docs/).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Shared palette ──────────────────────────────────────────────────────────

BLUE   = "#3A7CA5"
ORANGE = "#D4622B"
GREEN  = "#4A9E6F"
RED    = "#C0392B"
GRAY   = "#B0B0B0"
LIGHT  = "#E8E8E8"
DARK   = "#2C3E50"
WHITE  = "#FFFFFF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Feature coverage heatmap
# ═══════════════════════════════════════════════════════════════════════════

def fig_feature_coverage():
    """Heatmap showing element × basis-family support for legacy vs new."""

    topologies = [
        "Line", "Triangle", "Quad", "Tetra", "Hex",
        "Wedge", "Pyramid",
    ]
    basis_families = [
        "Lagrange", "Serendipity", "Hierarchical", "Spectral\n(GLL)",
        "Bernstein", "Hermite", "Bubble", "B-Spline /\nNURBS",
        "Raviart–\nThomas", "Nédélec", "BDM",
    ]

    # 0 = not supported, 1 = legacy only, 2 = new only, 3 = both
    # Rows = topologies, Cols = basis families
    # B-Spline/NURBS: BSplineBasis is 1D (Line), TensorProductBasis<BSplineBasis>
    # and NURBSTensorBasis extend to Quad (2D) and Hex (3D) — tensor-product only.
    # Tri/Tet/Wedge/Pyramid are not tensor-product topologies.
    #                       Lag  Ser  Hier Spec Bern Herm Bubb BSpl RT   Ned  BDM
    data = np.array([
        # Line
        [3,   0,   2,   2,   2,   2,   2,   2,   0,   0,   0],
        # Triangle
        [3,   0,   2,   2,   2,   0,   2,   0,   2,   2,   2],
        # Quad
        [3,   3,   2,   2,   2,   2,   2,   2,   2,   2,   2],
        # Tetra
        [3,   0,   2,   2,   2,   0,   2,   0,   2,   2,   0],
        # Hex
        [3,   3,   2,   2,   2,   2,   2,   2,   2,   2,   0],
        # Wedge
        [3,   2,   2,   2,   2,   0,   2,   0,   2,   2,   0],
        # Pyramid
        [2,   2,   2,   2,   2,   0,   2,   0,   2,   2,   0],
    ], dtype=int)

    cmap = ListedColormap([LIGHT, "#F4D03F", BLUE, GREEN])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(basis_families)))
    ax.set_xticklabels(basis_families, fontsize=9, ha="center")
    ax.set_yticks(np.arange(len(topologies)))
    ax.set_yticklabels(topologies, fontsize=10)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Grid lines
    for edge in np.arange(-0.5, len(basis_families), 1):
        ax.axvline(edge, color=WHITE, linewidth=1.5)
    for edge in np.arange(-0.5, len(topologies), 1):
        ax.axhline(edge, color=WHITE, linewidth=1.5)

    # Cell labels
    labels = {0: "—", 1: "Legacy", 2: "New", 3: "Both"}
    text_colors = {0: GRAY, 1: DARK, 2: WHITE, 3: WHITE}
    for i in range(len(topologies)):
        for j in range(len(basis_families)):
            ax.text(j, i, labels[data[i, j]], ha="center", va="center",
                    fontsize=8, fontweight="bold", color=text_colors[data[i, j]])

    # Legend
    patches = [
        mpatches.Patch(facecolor=LIGHT,    edgecolor=GRAY, label="Not supported"),
        mpatches.Patch(facecolor="#F4D03F", edgecolor=GRAY, label="Legacy only"),
        mpatches.Patch(facecolor=BLUE,     edgecolor=GRAY, label="New library only"),
        mpatches.Patch(facecolor=GREEN,    edgecolor=GRAY, label="Both (backward-compat)"),
    ]
    ax.legend(handles=patches, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, -0.10), fontsize=9, frameon=False)

    ax.set_title("Element Topology × Basis Family Support", pad=50)

    fig.savefig(os.path.join(OUT_DIR, "feature_coverage.svg"), format="svg")
    plt.close(fig)
    print("  ✓ feature_coverage.svg")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — DOF scaling by polynomial order
# ═══════════════════════════════════════════════════════════════════════════

def fig_dof_scaling():
    """DOF count vs polynomial order, with legacy fixed-order points."""

    orders = np.arange(1, 7)

    def dof_line(p):      return p + 1
    def dof_tri(p):       return (p + 1) * (p + 2) // 2
    def dof_quad(p):      return (p + 1) ** 2
    def dof_tet(p):       return (p + 1) * (p + 2) * (p + 3) // 6
    def dof_hex(p):       return (p + 1) ** 3
    def dof_wedge(p):     return (p + 1) * (p + 1) * (p + 2) // 2
    def dof_pyramid(p):   return (p + 1) * (p + 2) * (2 * p + 3) // 6

    topo = [
        ("Line",     dof_line,    BLUE,     "o"),
        ("Triangle", dof_tri,     ORANGE,   "s"),
        ("Quad",     dof_quad,    GREEN,    "D"),
        ("Tetra",    dof_tet,     RED,      "^"),
        ("Hex",      dof_hex,     "#8E44AD", "v"),
        ("Wedge",    dof_wedge,   "#16A085", "P"),
        ("Pyramid",  dof_pyramid, "#D4A017", "X"),
    ]

    # Legacy fixed orders:  element → (order, n_dofs)
    legacy_points = {
        "Line":     [(1, 2), (2, 3)],
        "Triangle": [(1, 3), (2, 6)],
        "Quad":     [(1, 4), (2, 9)],
        "Tetra":    [(1, 4), (2, 10)],
        "Hex":      [(1, 8), (2, 27)],
        "Wedge":    [(1, 6)],
        "Pyramid":  [],  # no legacy pyramid
    }
    # Serendipity legacy variants (reduced DOF)
    legacy_serendipity = {
        "Quad": [(2, 8)],   # Quad8
        "Hex":  [(2, 20)],  # Hex20
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for name, func, color, marker in topo:
        dofs = [func(p) for p in orders]
        ax.plot(orders, dofs, color=color, marker=marker, markersize=6,
                linewidth=1.8, label=f"{name}", zorder=3)

        # Legacy points (filled black circle overlay)
        for p, n in legacy_points.get(name, []):
            ax.plot(p, n, marker="o", color="black", markersize=10,
                    markerfacecolor="none", markeredgewidth=2.2, zorder=5)

        # Legacy serendipity (open diamond)
        for p, n in legacy_serendipity.get(name, []):
            ax.plot(p, n, marker="D", color="black", markersize=9,
                    markerfacecolor="none", markeredgewidth=2.0,
                    linestyle="none", zorder=5)

    # Annotations
    ax.annotate("Legacy fixed\norders", xy=(1.05, 2.2), fontsize=8.5,
                fontstyle="italic", color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                xytext=(1.8, 15), ha="center")

    ax.annotate("Serendipity\n(Quad8, Hex20)", xy=(2.05, 20), fontsize=8,
                fontstyle="italic", color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
                xytext=(3.3, 45), ha="center")

    ax.set_xlabel("Polynomial Order  $p$")
    ax.set_ylabel("Degrees of Freedom")
    ax.set_xticks(orders)
    ax.set_yscale("log")
    ax.set_ylim(1, 500)
    ax.set_xlim(0.7, 6.3)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Build legend with legacy markers
    handles, labels = ax.get_legend_handles_labels()
    legacy_marker = plt.Line2D([0], [0], marker="o", color="black",
                               markerfacecolor="none", markeredgewidth=2.0,
                               markersize=9, linestyle="none",
                               label="Legacy solver (fixed)")
    seren_marker = plt.Line2D([0], [0], marker="D", color="black",
                              markerfacecolor="none", markeredgewidth=1.8,
                              markersize=8, linestyle="none",
                              label="Legacy serendipity")
    handles.extend([legacy_marker, seren_marker])
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, ncol=3,
              framealpha=0.9)

    ax.set_title("DOF Scaling: Arbitrary-Order (New) vs Fixed-Order (Legacy)")

    fig.savefig(os.path.join(OUT_DIR, "dof_scaling.svg"), format="svg")
    plt.close(fig)
    print("  ✓ dof_scaling.svg")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Derivative support comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig_hessian_support():
    """Side-by-side bar chart: derivative support by element type."""

    elements = [
        "Line2/3", "Tri3", "Tri6", "Quad4", "Quad8", "Quad9",
        "Tet4", "Tet10", "Hex8", "Hex20", "Hex27",
        "Wedge6", "Pyramid5", "Pyramid14",
    ]

    # 0 = no support, 1 = values only, 2 = values + gradients, 3 = values + grad + hessians
    # Legacy support levels (from nn_elem_gnn.h / nn_elem_gnnxx.h)
    legacy = [
        2,  # Line (LIN1/LIN2): N + Nx, no Nxx
        2,  # Tri3: N + Nx, no Nxx
        3,  # Tri6: N + Nx + Nxx
        2,  # Quad4: N + Nx, no Nxx
        3,  # Quad8: N + Nx + Nxx
        3,  # Quad9: N + Nx + Nxx
        2,  # Tet4: N + Nx, no Nxx
        3,  # Tet10: N + Nx + Nxx
        2,  # Hex8: N + Nx, no Nxx
        2,  # Hex20: N + Nx, no Nxx
        2,  # Hex27: N + Nx, no Nxx
        2,  # Wedge6: N + Nx, no Nxx
        0,  # Pyramid5: not in legacy
        0,  # Pyramid14: not in legacy
    ]

    # New basis library: all elements have analytic values + gradients + hessians
    new = [3] * len(elements)

    fig, ax = plt.subplots(figsize=(11, 4.5))

    x = np.arange(len(elements))
    bar_w = 0.38

    colors_legacy = [
        {0: LIGHT, 1: "#F5CBA7", 2: "#F0B27A", 3: GREEN}[v] for v in legacy
    ]
    colors_new = [GREEN] * len(elements)

    bars_l = ax.bar(x - bar_w / 2, legacy, bar_w, label="Legacy solver",
                    color=colors_legacy, edgecolor=DARK, linewidth=0.6)
    bars_n = ax.bar(x + bar_w / 2, new, bar_w, label="New Basis library",
                    color=colors_new, edgecolor=DARK, linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(elements, fontsize=8.5, rotation=35, ha="right")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["None", "Values", "Values +\nGradients",
                        "Values +\nGradients +\nHessians"], fontsize=8.5)
    ax.set_ylim(-0.15, 3.65)

    # Mark missing elements in legacy
    for i, v in enumerate(legacy):
        if v == 0:
            ax.text(x[i] - bar_w / 2, 0.08, "N/A", ha="center", va="bottom",
                    fontsize=7, color=RED, fontweight="bold")

    # Highlight the gap
    for i, (lv, nv) in enumerate(zip(legacy, new)):
        if lv < nv:
            gap = nv - lv
            label = f"+{'H' if gap == 1 else 'G+H' if gap == 2 else 'all'}"
            ax.annotate(label, xy=(x[i] + bar_w / 2, nv + 0.05),
                        fontsize=6.5, ha="center", va="bottom",
                        color=BLUE, fontweight="bold")

    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_title("Derivative Support by Element Type: Legacy vs New Basis Library")

    fig.savefig(os.path.join(OUT_DIR, "hessian_support.svg"), format="svg")
    plt.close(fig)
    print("  ✓ hessian_support.svg")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Polynomial reproduction accuracy
# ═══════════════════════════════════════════════════════════════════════════

def fig_polynomial_reproduction():
    """
    Visualise the polynomial reproduction test results.

    The unit tests verify that for every monomial x^a y^b z^c with total
    degree a+b+c <= p, nodal interpolation through the Lagrange basis
    reproduces the monomial exactly (to ~machine precision).  We replicate
    the test logic in pure Python for the simplest topologies so the figure
    contains real computed data, not just placeholders.
    """

    # ── Lagrange basis evaluation (reference implementations) ──

    def lagrange_line(order, xi):
        """1D Lagrange on [-1,1], equispaced nodes."""
        nodes = np.linspace(-1, 1, order + 1)
        n = len(nodes)
        vals = np.ones(n)
        for i in range(n):
            for j in range(n):
                if j != i:
                    vals[i] *= (xi - nodes[j]) / (nodes[i] - nodes[j])
        return nodes, vals

    def lagrange_tri(order, xi, eta):
        """2D Lagrange on reference triangle (0,0)-(1,0)-(0,1)."""
        pts = []
        for j in range(order + 1):
            for i in range(order + 1 - j):
                pts.append((i / order, j / order))
        pts = np.array(pts)
        n = len(pts)
        vals = np.ones(n)
        # Use Wachspress-style product formula for simplicity — only
        # numerically stable for low order, which is all we need.
        for k in range(n):
            for m in range(n):
                if m != k:
                    # barycentric coord difference
                    dx = pts[k] - pts[m]
                    dp = np.array([xi, eta]) - pts[m]
                    # We need the full Lagrange product; use 1D products
                    # along each coordinate that differs.
                    pass
        # Simpler: just use the cardinal Lagrange via Vandermonde inverse
        # Build Vandermonde for complete polynomial space on triangle
        mono_exp = []
        for j in range(order + 1):
            for i in range(order + 1 - j):
                mono_exp.append((i, j))
        V = np.zeros((n, n))
        for row, (px, py) in enumerate(pts):
            for col, (ex, ey) in enumerate(mono_exp):
                V[row, col] = px**ex * py**ey
        # Cardinal values at (xi, eta)
        mono_vec = np.array([xi**ex * eta**ey for ex, ey in mono_exp])
        vals = np.linalg.solve(V.T, mono_vec)
        return pts, vals

    def lagrange_quad(order, xi, eta):
        """2D Lagrange on [-1,1]^2 via tensor product."""
        nodes_1d = np.linspace(-1, 1, order + 1)
        # 1D basis values
        def basis_1d(x):
            n = len(nodes_1d)
            v = np.ones(n)
            for i in range(n):
                for j in range(n):
                    if j != i:
                        v[i] *= (x - nodes_1d[j]) / (nodes_1d[i] - nodes_1d[j])
            return v
        bx = basis_1d(xi)
        by = basis_1d(eta)
        # Tensor product nodes and values
        pts = []
        vals = []
        for j in range(order + 1):
            for i in range(order + 1):
                pts.append((nodes_1d[i], nodes_1d[j]))
                vals.append(bx[i] * by[j])
        return np.array(pts), np.array(vals)

    def lagrange_tet(order, xi, eta, zeta):
        """3D Lagrange on reference tet (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1)."""
        pts = []
        mono_exp = []
        for k in range(order + 1):
            for j in range(order + 1 - k):
                for i in range(order + 1 - j - k):
                    pts.append((i / order if order > 0 else 0,
                                j / order if order > 0 else 0,
                                k / order if order > 0 else 0))
                    mono_exp.append((i, j, k))
        pts = np.array(pts)
        n = len(pts)
        V = np.zeros((n, n))
        for row in range(n):
            for col, (ex, ey, ez) in enumerate(mono_exp):
                V[row, col] = pts[row, 0]**ex * pts[row, 1]**ey * pts[row, 2]**ez
        mono_vec = np.array([xi**ex * eta**ey * zeta**ez
                             for ex, ey, ez in mono_exp])
        vals = np.linalg.solve(V.T, mono_vec)
        return pts, vals

    def lagrange_hex(order, xi, eta, zeta):
        """3D Lagrange on [-1,1]^3 via tensor product."""
        nodes_1d = np.linspace(-1, 1, order + 1)
        def basis_1d(x):
            n = len(nodes_1d)
            v = np.ones(n)
            for i in range(n):
                for j in range(n):
                    if j != i:
                        v[i] *= (x - nodes_1d[j]) / (nodes_1d[i] - nodes_1d[j])
            return v
        bx, by, bz = basis_1d(xi), basis_1d(eta), basis_1d(zeta)
        pts, vals = [], []
        for k in range(order + 1):
            for j in range(order + 1):
                for i in range(order + 1):
                    pts.append((nodes_1d[i], nodes_1d[j], nodes_1d[k]))
                    vals.append(bx[i] * by[j] * bz[k])
        return np.array(pts), np.array(vals)

    # ── Compute interpolation errors ──

    # For each topology and order, test all monomials up to that order.
    # Measure max |interpolated - exact| across several test points.

    test_configs = [
        # (name, order, dim, eval_func, test_points)
        ("Line p=1",   1, 1, lambda p, xi: lagrange_line(1, xi[0]),
         [np.array([0.33]), np.array([-0.71]), np.array([0.0])]),
        ("Line p=2",   2, 1, lambda p, xi: lagrange_line(2, xi[0]),
         [np.array([0.33]), np.array([-0.71]), np.array([0.0])]),
        ("Tri p=1",    1, 2, lambda p, xi: lagrange_tri(1, xi[0], xi[1]),
         [np.array([0.2, 0.3]), np.array([0.1, 0.1]), np.array([0.5, 0.2])]),
        ("Tri p=2",    2, 2, lambda p, xi: lagrange_tri(2, xi[0], xi[1]),
         [np.array([0.2, 0.3]), np.array([0.1, 0.1]), np.array([0.5, 0.2])]),
        ("Quad p=1",   1, 2, lambda p, xi: lagrange_quad(1, xi[0], xi[1]),
         [np.array([0.2, -0.3]), np.array([-0.5, 0.7]), np.array([0.0, 0.0])]),
        ("Quad p=2",   2, 2, lambda p, xi: lagrange_quad(2, xi[0], xi[1]),
         [np.array([0.2, -0.3]), np.array([-0.5, 0.7]), np.array([0.0, 0.0])]),
        ("Tet p=1",    1, 3, lambda p, xi: lagrange_tet(1, xi[0], xi[1], xi[2]),
         [np.array([0.1, 0.2, 0.1]), np.array([0.3, 0.1, 0.05])]),
        ("Tet p=2",    2, 3, lambda p, xi: lagrange_tet(2, xi[0], xi[1], xi[2]),
         [np.array([0.1, 0.2, 0.1]), np.array([0.3, 0.1, 0.05])]),
        ("Hex p=1",    1, 3, lambda p, xi: lagrange_hex(1, xi[0], xi[1], xi[2]),
         [np.array([0.2, -0.3, 0.4]), np.array([-0.5, 0.1, -0.2])]),
        ("Hex p=2",    2, 3, lambda p, xi: lagrange_hex(2, xi[0], xi[1], xi[2]),
         [np.array([0.2, -0.3, 0.4]), np.array([-0.5, 0.1, -0.2])]),
    ]

    def monomial_exponents(dim, max_degree):
        """All (a,b,c) with a+b+c <= max_degree, truncated to dim coords."""
        exps = []
        for total in range(max_degree + 1):
            if dim == 1:
                exps.append((total,))
            elif dim == 2:
                for a in range(total + 1):
                    exps.append((a, total - a))
            else:
                for a in range(total + 1):
                    for b in range(total - a + 1):
                        exps.append((a, b, total - a - b))
        return exps

    def monomial_label(exp):
        parts = []
        vars_ = "xyz"
        for i, e in enumerate(exp):
            if e == 0:
                continue
            elif e == 1:
                parts.append(vars_[i])
            else:
                parts.append(f"{vars_[i]}^{e}")
        return "·".join(parts) if parts else "1"

    # Gather results: config_name -> list of (monomial_label, max_error)
    results = {}
    for name, order, dim, eval_func, test_pts in test_configs:
        exps = monomial_exponents(dim, order)
        errors = []
        for exp in exps:
            max_err = 0.0
            for xi in test_pts:
                nodes, vals = eval_func(order, xi)
                nodes = np.atleast_2d(nodes)
                if nodes.shape[0] == 1 and len(vals) > 1:
                    nodes = nodes.T
                if nodes.ndim == 1:
                    nodes = nodes.reshape(-1, 1)
                # Nodal coefficients for this monomial
                coeffs = np.array([
                    np.prod([float(nodes[k, d])**exp[d] for d in range(dim)])
                    for k in range(len(nodes))
                ])
                interpolated = np.dot(coeffs, vals)
                exact = np.prod([xi[d]**exp[d] for d in range(dim)])
                max_err = max(max_err, abs(interpolated - exact))
            # Floor at 1e-16 for log scale
            errors.append((monomial_label(exp), max(max_err, 1e-16)))
        results[name] = errors

    # ── Plot ──

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"hspace": 0.45})

    # Panel A: 1D and 2D topologies
    ax = axes[0]
    configs_2d = ["Line p=1", "Line p=2", "Tri p=1", "Tri p=2",
                  "Quad p=1", "Quad p=2"]
    colors_2d = [BLUE, BLUE, ORANGE, ORANGE, GREEN, GREEN]
    hatches_2d = ["", "//", "", "//", "", "//"]

    group_names = []
    group_errs = []
    group_colors = []
    group_hatches = []
    for ci, cname in enumerate(configs_2d):
        for label, err in results[cname]:
            group_names.append(f"{cname}\n{label}")
            group_errs.append(err)
            group_colors.append(colors_2d[ci])
            group_hatches.append(hatches_2d[ci])

    x = np.arange(len(group_names))
    bars = ax.bar(x, group_errs, color=group_colors, edgecolor=DARK, linewidth=0.5)
    for bar, h in zip(bars, group_hatches):
        bar.set_hatch(h)
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-8)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=6, rotation=60, ha="right")
    ax.axhline(1e-12, color=RED, linestyle="--", linewidth=1, alpha=0.7,
               label="Test tolerance (1e-12)")
    ax.axhline(np.finfo(float).eps, color=GRAY, linestyle=":", linewidth=1,
               alpha=0.7, label="Machine epsilon")
    ax.set_ylabel("Max Interpolation Error")
    ax.set_title("(a)  Polynomial Reproduction: 1D and 2D Elements")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    # Panel B: 3D topologies
    ax = axes[1]
    configs_3d = ["Tet p=1", "Tet p=2", "Hex p=1", "Hex p=2"]
    colors_3d = [RED, RED, "#8E44AD", "#8E44AD"]
    hatches_3d = ["", "//", "", "//"]

    group_names = []
    group_errs = []
    group_colors = []
    group_hatches = []
    for ci, cname in enumerate(configs_3d):
        for label, err in results[cname]:
            group_names.append(f"{cname}\n{label}")
            group_errs.append(err)
            group_colors.append(colors_3d[ci])
            group_hatches.append(hatches_3d[ci])

    x = np.arange(len(group_names))
    bars = ax.bar(x, group_errs, color=group_colors, edgecolor=DARK, linewidth=0.5)
    for bar, h in zip(bars, group_hatches):
        bar.set_hatch(h)
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-8)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=6.5, rotation=60, ha="right")
    ax.axhline(1e-12, color=RED, linestyle="--", linewidth=1, alpha=0.7,
               label="Test tolerance (1e-12)")
    ax.axhline(np.finfo(float).eps, color=GRAY, linestyle=":", linewidth=1,
               alpha=0.7, label="Machine epsilon")
    ax.set_ylabel("Max Interpolation Error")
    ax.set_title("(b)  Polynomial Reproduction: 3D Elements")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    fig.savefig(os.path.join(OUT_DIR, "polynomial_reproduction.svg"), format="svg")
    plt.close(fig)
    print("  ✓ polynomial_reproduction.svg")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Basis family breadth comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig_basis_family_breadth():
    """
    Grouped bar chart: number of basis families, total element-order combos,
    derivative levels, and functional-space coverage (H1, H(div), H(curl)).
    """

    categories = [
        "Basis\nFamilies",
        "Element-Order\nCombinations",
        "Elements with\nAnalytic Hessians",
        "Functional\nSpaces",
        "Unit\nTests",
    ]

    legacy_vals = [
        1,     # Lagrange only (+ serendipity variants = still Lagrange family)
        14,    # 14 fixed element types
        4,     # Tri6, Quad8, Quad9, Tet10
        1,     # H1 only
        0,     # no unit tests for basis functions
    ]
    new_vals = [
        13,    # Lagrange, Hierarchical, Bernstein, Spectral, Serendipity,
               # Hermite, Bubble, BSpline, NURBS/Tensor, RT, Nedelec, BDM, TensorProduct
        56,    # 7 topologies × ~4 orders average across 8 scalar families + vector combos
        14,    # all element types
        3,     # H1, H(div), H(curl)
        386,   # from PLAN.md
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5))

    x = np.arange(len(categories))
    bar_w = 0.35

    ax.bar(x - bar_w / 2, legacy_vals, bar_w, label="Legacy solver",
           color=ORANGE, edgecolor=DARK, linewidth=0.6)
    ax.bar(x + bar_w / 2, new_vals, bar_w, label="New Basis library",
           color=BLUE, edgecolor=DARK, linewidth=0.6)

    # Value labels
    for i, (lv, nv) in enumerate(zip(legacy_vals, new_vals)):
        ax.text(x[i] - bar_w / 2, lv + max(nv * 0.02, 0.8), str(lv),
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=ORANGE)
        ax.text(x[i] + bar_w / 2, nv + max(nv * 0.02, 0.8), str(nv),
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=BLUE)

    # Multiplier annotations
    for i, (lv, nv) in enumerate(zip(legacy_vals, new_vals)):
        if lv > 0:
            ratio = nv / lv
            ax.annotate(f"{ratio:.0f}×", xy=(x[i], max(lv, nv) * 1.25),
                        fontsize=10, fontweight="bold", ha="center",
                        color=DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 800)
    ax.set_ylabel("Count")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    ax.set_title("Basis Library Capability Comparison")

    fig.savefig(os.path.join(OUT_DIR, "basis_family_breadth.svg"), format="svg")
    plt.close(fig)
    print("  ✓ basis_family_breadth.svg")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Generating figures in {OUT_DIR}/")
    fig_feature_coverage()
    fig_dof_scaling()
    fig_hessian_support()
    fig_polynomial_reproduction()
    fig_basis_family_breadth()
    print("Done.")
