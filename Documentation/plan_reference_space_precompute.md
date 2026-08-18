# Plan: Reference-Space Stiffness Precomputation for Affine Elements

## Summary

For affine elements (Tet4, Tri3), terms that are linear in the basis functions can have their
element matrix computed as a matrix product in reference space rather than QP-by-QP accumulation.
This eliminates the QP loop entirely for those terms.

## Theory

For the diffusion bilinear form on an affine element:

```
K_ij = ∫ ∇φ_i · ∇φ_j dΩ
     = |det(J)| Σ_q w_q (J^-T ∇̂φ_i(ξ_q)) · (J^-T ∇̂φ_j(ξ_q))
     = |det(J)| Σ_q w_q ∇̂φ_i(ξ_q)^T (J^-1 J^-T) ∇̂φ_j(ξ_q)
```

Since J is constant for affine elements, the metric tensor G = J^-1 J^-T is constant:

```
K_ij = |det(J)| * (Σ_q w_q ∇̂φ_i(ξ_q)^T) G (Σ_q w_q ... ∇̂φ_j(ξ_q))
```

Wait — that factorization is incorrect because each q has its own basis values. The correct
approach is:

```
K_ij = |det(J)| * Σ_q w_q * Σ_{d1,d2} G_{d1,d2} * ∂φ̂_i/∂ξ_{d1}(ξ_q) * ∂φ̂_j/∂ξ_{d2}(ξ_q)
```

Define the reference stiffness tensor:
```
K̂_{ij}^{d1,d2} = Σ_q w_q * ∂φ̂_i/∂ξ_{d1}(ξ_q) * ∂φ̂_j/∂ξ_{d2}(ξ_q)
```

This is precomputed ONCE per element type (depends only on reference basis and quadrature rule).
Then per element:
```
K_ij = |det(J)| * Σ_{d1,d2} G_{d1,d2} * K̂_{ij}^{d1,d2}
```

For 3D: 9 terms in the sum (or 6 by symmetry). For 2D: 4 terms (or 3 by symmetry).
Per-element cost: 9 multiplies + 8 adds per matrix entry = 17 FLOPs/entry.
vs QP loop: 4 QPs × (3 muls + 2 adds) = 20 FLOPs/entry for simple Laplacian.

## Applicability to NS-VMS

NS-VMS has these term types:
1. **Viscous diffusion** `ν ∇u : ∇v` — bilinear, constant ν → precomputable
2. **Pressure gradient** `p ∇·v` — bilinear → precomputable
3. **Continuity** `∇·u q` — bilinear → precomputable
4. **Convection** `(u·∇)u · v` — trilinear (depends on u) → NOT precomputable
5. **SUPG/PSPG stabilization** — depends on solution → NOT precomputable
6. **Mass matrix** `u · v` — bilinear → precomputable

Items 1-3, 6 can use precomputed reference stiffness. Items 4-5 must use QP loops.
The fraction of total work that's precomputable depends on the NS-VMS formulation weights.

## Implementation Approach

### KernelIR Level
- Tag each term as "reference-precomputable" based on dependency analysis
- A term is precomputable if: (a) element is affine, (b) all non-basis factors are element-constant
  (no solution dependence, no spatial variation within element)

### LLVMGen Level
- For precomputable terms: emit direct matrix accumulation from precomputed tables + G tensor
- For non-precomputable terms: emit standard QP loop

### Assembler Level
- Precompute reference stiffness tables per element type at setup
- Pass as JIT constants alongside existing parameters

## Expected Impact

- For pure diffusion: eliminate QP loop entirely → ~4x fewer FLOPs per element
- For NS-VMS: ~20-30% of terms precomputable, ~10-15% kernel reduction
- Most beneficial for simple formulations (heat equation, Stokes)

## Risk

- Complexity of detecting precomputable terms at the FormIR/KernelIR level
- Limited impact for fully nonlinear formulations like NS-VMS
- Interaction with trial-only caching (may be redundant for precomputable terms)
