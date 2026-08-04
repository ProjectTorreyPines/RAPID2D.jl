# Wall face geometry, and the one-sided flux that bombards it.
#
# Every wall process — absorption, recycling, sputtering, secondary emission —
# is an exchange across a FACE, not a property of a cell. A staircase corner has
# two outward faces and must be debited on both; the R- and Z-faces of one cell
# carry different spacings and different areas. Naming the faces once, here, is
# what lets every species share one wall treatment instead of inventing its own.
#
# Nothing in this file solves anything. It answers two questions the solvers will
# ask: *which faces are there and how big are they*, and *how hard is the gas
# hitting them*.

"""
    WallFace{FT}

One outward face of an in-wall cell — the interface across which that cell
exchanges particles with the wall.

- `rid`, `zid`, `nid`  the **in-wall** cell that owns the face (grid indices and
  linear index). A face always belongs to the cell on the plasma side.
- `outward`  index step `(ΔR, ΔZ)` from that cell across the face, one of
  `(±1, 0)` or `(0, ±1)`. The cell it points at is on or outside the wall, and
  may be off-grid entirely when the wall coincides with the grid frame.
- `area`  `A_f` [m²], the true area of the surface of revolution.
- `area_per_volume`  `A_f/V_i` [1/m], the factor a boundary flux is multiplied by
  to become a rate in the owning cell: `∂n_i/∂t = −(A_f/V_i)·Γ_f`.

Both `area` and `area_per_volume` are stored because they answer different
questions. `area` converts a flux density into particles per second (a
diagnostic, and the wall ledger); `area_per_volume` is the coefficient a Robin
condition subtracts from the diagonal. Deriving one from the other at each call
site is how they drift apart, and they must not — absorption and re-emission are
only exactly reciprocal across a face if both use the same pair.
"""
struct WallFace{FT <: AbstractFloat}
    rid::Int
    zid::Int
    nid::Int
    outward::Tuple{Int, Int}
    area::FT
    area_per_volume::FT
end

"""
    is_in_wall(G, i, j) -> Bool

Whether node `(i, j)` is strictly inside the wall and therefore part of the
problem being solved.

`nodes.state` is 1 inside, 0 **on** the wall, −1 outside; only `> 0.5` counts, so
an on-wall node is treated as outside. Off-grid indices are false, which is what
makes a wall coinciding with the grid frame work — there is simply no outside
region to reach.

This is the single predicate that decides where the boundary is. `wall_faces`
uses it to decide which cardinal arms are wall faces, and any wall-aware operator
must use the same one, or the boundary term would land on faces the flux terms did
not omit.
"""
function is_in_wall(G::GridGeometry{FT}, i::Integer, j::Integer) where {FT <: AbstractFloat}
    return 1 <= i <= G.NR && 1 <= j <= G.NZ && G.nodes.state[i, j] > FT(0.5)
end

"""
    wall_faces(G) -> Vector{WallFace}

Every outward face of every in-wall cell of `G`.

A face exists wherever an in-wall cell has a cardinal neighbour that is *not*
in-wall — on the wall, outside it, or off the grid. That is deliberately the same
predicate `build_reflective_diffusion_matrix` uses to decide which stencil arm to
omit, so the faces returned here are exactly the arms that operator drops: the
places where the flux is currently forced to zero and where a Robin condition
will instead put `v_absorb·n_w`.

**Areas in cylindrical geometry.** With `Jacob = R` the cell volume is
`V_i = 2π·R_i·ΔR·ΔZ`, and

```
    R-face at i±½ :  A_f = 2π·R_{i±½}·ΔZ    A_f/V_i = R_{i±½}/(R_i·ΔR)
    Z-face at j±½ :  A_f = 2π·R_i·ΔR        A_f/V_i = 1/ΔZ
```

The Z-face factor is exactly `1/ΔZ`; the R-face one is **not** `1/ΔR`. The extra
`R_{i±½}/R_i = 1 ± ΔR/(2R_i)` is load-bearing twice over: the diffusion operator
already carries it in `invJ·½(CT_out + CT_in)`, so a boundary term without it
would contradict the flux terms in its own row; and the conserved measure
`Σ J_k·n_k` closes against the true face area, so dropping it leaks by `ΔR/(2R)`
— small enough to look like round-off, large enough to matter.

**Staircase walls over-count area on inclined segments** by `√2` at 45°, and
unlike the half-cell offset of an axis-aligned wall it does not shrink under grid
refinement. Measured and pinned in `wall_test.jl` rather than corrected; the fix
changes what `area` means and belongs with the machinery that consumes it.
"""
function wall_faces(G::GridGeometry{FT}) where {FT <: AbstractFloat}
    NR, NZ = G.NR, G.NZ

    faces = WallFace{FT}[]
    for j in 1:NZ, i in 1:NR
        is_in_wall(G, i, j) || continue
        R = G.R2D[i, j]
        vol = 2 * FT(π) * R * G.dR * G.dZ
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            is_in_wall(G, i + di, j + dj) && continue
            # R_face = R for a Z-face (di = 0), R ± ΔR/2 for an R-face
            R_face = R + di * G.dR / 2
            span = dj == 0 ? G.dZ : G.dR
            area = 2 * FT(π) * R_face * span
            push!(
                faces,
                WallFace{FT}(i, j, G.nodes.nid[i, j], (di, dj), area, area / vol)
            )
        end
    end
    return faces
end

"""
    v_incident(T_eV, m)

One-sided impingement speed `¼v̄` [m/s] of a Maxwellian at temperature `T_eV` [eV]
and particle mass `m` [kg], with `v̄ = √(8kT/πm)` the **mean** speed.

The velocity factor of the Hertz-Knudsen gross flux `Γ = v_incident·n_w` — the
rate at which particles arrive regardless of what the surface then does with them.
It does **not** depend on the boundary condition, so diagnostics built on it
(sputtering, bombardment rates) are unaffected by the wall treatment: a reflective
wall still has `Γ_in = Γ_out = v_incident·n_w`, with only the *net* flux zero.

**`v̄`, not `v_th`.** `neutral_gas_thermal_speed` is `√(T/m)`, the convention
`D = ½·v_th·λ` is written in; the two differ by `√(8/π) = 1.596`, so `¼v_th` here
would under-count every impact by 37 %. Both go as `√(T/m)`, so no scaling test
would reveal the swap — hence the ratio is pinned to `0.3989 = ¼√(8/π)`.

Shares [`maxwellian_mean_speed`](@ref) with the channel ceiling rather than
restating the formula. It was stated twice, and the two copies disagreed: this one
took `(T, m)` and was right, while the ceiling scaled a channel's declared `v` by
one shared ratio and was √2 high for every collisional channel.
"""
v_incident(T_eV, m) = maxwellian_mean_speed(T_eV, m) / 4

"""
    gross_impingement(n_w, T_eV, m)

Gross one-sided particle flux `Γ = ¼n_w·v̄` [m⁻²s⁻¹] onto a surface in contact
with density `n_w` [m⁻³] at temperature `T_eV` [eV] (Hertz-Knudsen).

For the standard `10⁻² Pa` H₂ fill at `T_gas = 0.026 eV` this is
`1.07×10²¹ m⁻²s⁻¹` — a monolayer every 9 ms, which is why surface processes
matter at breakdown densities even with small yields.

The result is a flux **density**. Multiply by `WallFace.area` to get particles per
second onto a given face; multiplying by `area_per_volume` instead gives the rate
of change of density in the cell that owns it.
"""
gross_impingement(n_w, T_eV, m) = v_incident(T_eV, m) * n_w
