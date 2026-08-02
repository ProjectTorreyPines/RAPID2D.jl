# Per-face inventory of what crossed the wall.
#
# Conservation has so far been a GLOBAL statement — Σ J·n does not drift. That
# catches a leak but cannot say where it came from, and it cannot answer the
# question every surface process reduces to: how much did *this* tile absorb?
#
# Today electrons and ions are booked per NODE (`cum2D_Ne_loss[on_out_wall_nids]`,
# `physics.jl:731`), recording what an outside cell held at the moment it was
# zeroed. Those are cells the Robin form removes entirely, and a node is not a
# face: it cannot say which way the material went, nor which wall segment it
# belongs to.

"""
    WallLedger{FT}(n_faces)

Cumulative particle counts across each wall face: `absorbed[f]` is what has left
the plasma through face `f`, `emitted[f]` what has come back.

Both are **particles**, not densities and not fluxes, so they can be summed over
faces, over species and over time without carrying a volume factor around. Index
order matches the `Vector{WallFace}` they were built from.
"""
struct WallLedger{FT <: AbstractFloat}
    absorbed::Vector{FT}
    emitted::Vector{FT}

    WallLedger{FT}(n_faces::Integer) where {FT <: AbstractFloat} =
        new{FT}(zeros(FT, n_faces), zeros(FT, n_faces))
end

"Total particles absorbed across all faces so far."
total_absorbed(led::WallLedger) = sum(led.absorbed)

"Total particles re-emitted across all faces so far."
total_emitted(led::WallLedger) = sum(led.emitted)

"""
    accumulate_wall_absorption!(ledger, faces, v_absorb, n, dt)

Add this step's absorption to `ledger.absorbed`, face by face:

```
    absorbed_f += A_f · v_absorb_f · n[owner(f)] · Δt
```

**`n` must be the POST-step density.** The Robin term is carried implicitly, so
backward Euler evaluates the wall flux at `n^{n+1}`; booking `n^n` instead leaves
the ledger short by `O(Δt)` and the conservation identity below stops closing.

The identity this buys is local rather than global:

```
    Δ(Σ J·n)_interior  +  Σ_f absorbed_f / (2π·ΔR·ΔZ)  =  0
```

It holds to machine precision because `A_f · v_absorb_f` is exactly the same
product the operator subtracted from the diagonal, scaled by `V_i`: absorption and
accounting cannot drift apart because they are the same arithmetic. A discrepancy
therefore points at a *face*, not at the whole domain.

A staircase corner owns two faces and books each on its own area — the R-face and
the Z-face sweep different areas and generally carry different `v_absorb`, since
`g = (b̂·n̂)²` is a face property.
"""
function accumulate_wall_absorption!(
        ledger::WallLedger{FT},
        faces::AbstractVector{WallFace{FT}},
        v_absorb::AbstractVector{FT},
        n::AbstractArray{FT},
        dt::Real,
    ) where {FT <: AbstractFloat}
    length(faces) == length(v_absorb) == length(ledger.absorbed) ||
        throw(DimensionMismatch("faces, v_absorb and the ledger must agree in length"))
    @inbounds for (k, f) in enumerate(faces)
        ledger.absorbed[k] += f.area * v_absorb[k] * n[f.nid] * FT(dt)
    end
    return ledger
end

"""
    wall_emission_source(G, faces, emitted, dt) -> Vector

Volumetric source [m⁻³s⁻¹] that returns `emitted[f]` particles through face `f`
into the **wall-adjacent interior cell** that owns it:

```
    S_i = Σ_f  emitted_f / (V_i · Δt)
        = Σ_f  (A_f/V_i) · Γ_f
```

The factor `A_f/V_i` is the very one the Robin diagonal uses, so absorption and
re-emission are exactly reciprocal across the same face. That gives the invariant
which catches the entire "created at the wall and silently lost" class:

> re-emitting everything absorbed (`Y = 1`, same species) must reproduce the
> reflective result to machine precision.

**Never deposit outside the wall.** The existing secondary-electron path does the
opposite — `treat_ion_outside_wall!` adds `γ_2nd·n_i` to cells *outside* and relies
on diffusion to carry them back, but `treat_electron_outside_wall!` books that band
as loss and zeroes it at the top of the next step. The measured yield reaching the
interior is ≈ 0 at production settings and varies by 1800× with Δt and ΔR, because
the controlling group is `D⊥Δt/Δx² ≈ 5e-5` rather than `γ_2nd`. Returning material
to the interior cell is that fix.

For cross-species return (recycling, sputtering, secondary electrons) scale
`emitted` by the yield before calling; this function only moves particles back
through the geometry.
"""
function wall_emission_source(
        G::GridGeometry{FT},
        faces::AbstractVector{WallFace{FT}},
        emitted::AbstractVector{FT},
        dt::Real,
    ) where {FT <: AbstractFloat}
    length(faces) == length(emitted) ||
        throw(DimensionMismatch("faces and emitted must agree in length"))
    src = zeros(FT, G.NR * G.NZ)
    @inbounds for (k, f) in enumerate(faces)
        vol = f.area / f.area_per_volume          # V_i, from the face's own pair
        src[f.nid] += emitted[k] / (vol * FT(dt))
    end
    return src
end
