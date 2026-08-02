# Shape statistics of a 2D field.
#
# Conservation tests say nothing about direction: an operator that swapped bR and
# bZ would conserve exactly as well as the correct one. Verifying that transport
# follows b̂ means solving and then asking where the material went, which is what
# these moments are for.

"""
    density_second_moments(G, n) -> (Σ_RR, Σ_RZ, Σ_ZZ)

Density-weighted covariance of position over the whole grid.

```
    R̄    = Σ n·R / Σ n
    Σ_RR = Σ n·(R − R̄)² / Σ n          (and likewise Σ_RZ, Σ_ZZ)
```

The eigenvectors of `[Σ_RR Σ_RZ; Σ_RZ Σ_ZZ]` are the principal axes of the
distribution and its eigenvalue ratio is how elongated it is — so a blob released
in an anisotropic tensor should come back with its major axis along `b̂` and an
axis ratio approaching `D∥/D⊥`.

**Weighted by density, not by particle count.** In cylindrical geometry the two
differ: `Σ J·n` is the conserved particle measure, and its centroid moves *outward*
while the density centroid moves *inward*, both at `D/R`. What is wanted here is
the shape of the density field as plotted, so `n` alone is the weight. Use the
Jacobian-weighted form if the question is about particles rather than about shape.

`Σ_RZ` is the discriminating component: it vanishes identically whenever the
distribution is aligned with the grid, so it is zero for an axis-aligned field and
non-zero exactly when the transport has been tilted.
"""
function density_second_moments(G::GridGeometry{FT}, n::AbstractMatrix{FT}) where {FT <: AbstractFloat}
    w = sum(n)
    iszero(w) && throw(ArgumentError("cannot take moments of an empty field"))
    R̄ = sum(n .* G.R2D) / w
    Z̄ = sum(n .* G.Z2D) / w
    dR = G.R2D .- R̄
    dZ = G.Z2D .- Z̄
    return (
        sum(n .* dR .^ 2) / w,
        sum(n .* dR .* dZ) / w,
        sum(n .* dZ .^ 2) / w,
    )
end
