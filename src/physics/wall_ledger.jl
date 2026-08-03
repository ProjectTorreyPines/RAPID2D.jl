# Per-face inventory of what crossed the wall.
#
# Conservation has so far been a GLOBAL statement — Σ J·n does not drift. That
# catches a leak but cannot say where it came from, and it cannot answer the
# question every surface process reduces to: how much did *this* tile absorb?
#
# Today electrons and ions are booked per NODE (`cum2D_Ne_loss[on_out_wall_nids]`
# in `treat_electron_outside_wall!`), recording what an outside cell held at the
# moment it was zeroed. Those are cells the Robin form removes entirely, and a
# node is not a face: it cannot say which way the material went, nor which wall
# segment it belongs to.

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
    accumulate_wall_absorption!(ledger, faces, v_absorb, n_new, dt; n_prev, θ = 1)

Add this step's absorption to `ledger.absorbed`, face by face:

```
    absorbed_f += A_f · v_absorb_f · n̄[owner(f)] · Δt        n̄ = (1−θ)·nⁿ + θ·nⁿ⁺¹
```

**`n̄` must match the θ the solve used**, because that is the density the operator
itself charged. `θ = 1` (backward Euler) is the default and needs only `n_new`;
any other θ requires `n_prev` as well and throws without it, since silently
booking `nⁿ⁺¹` under a θ-scheme is exactly the error this signature exists to
prevent. `Implicit_weight` defaults to `0.5` in this code, where booking `nⁿ⁺¹`
alone leaves a 1.3e-4 relative hole in the identity below against 8e-17 when the
weighting is right.

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
that depends on `b̂·n̂`, a face property.
"""
function accumulate_wall_absorption!(
        ledger::WallLedger{FT},
        faces::AbstractVector{WallFace{FT}},
        v_absorb::AbstractVector{FT},
        n_new::AbstractArray{FT},
        dt::Real;
        n_prev::Union{Nothing, AbstractArray{FT}} = nothing,
        θ::Real = 1,
    ) where {FT <: AbstractFloat}
    length(faces) == length(v_absorb) == length(ledger.absorbed) ||
        throw(DimensionMismatch("faces, v_absorb and the ledger must agree in length"))
    θ == 1 || !isnothing(n_prev) ||
        throw(ArgumentError("θ = $θ books (1−θ)·nⁿ + θ·nⁿ⁺¹, so n_prev is required"))
    w_new = FT(θ)
    w_prev = FT(1 - θ)
    @inbounds for (k, f) in enumerate(faces)
        n̄ = isnothing(n_prev) ? n_new[f.nid] :
            w_prev * n_prev[f.nid] + w_new * n_new[f.nid]
        ledger.absorbed[k] += f.area * v_absorb[k] * n̄ * FT(dt)
    end
    return ledger
end

"""
    net_wall_speed(v_absorb, Y)

Wall speed to hand the operator when a fraction `Y` of what a face absorbs comes
straight back **as the same species, through the same face**: `(1−Y)·v_absorb`.

This is the exact route, and the only one that reproduces the reflective solution.
Absorbing implicitly and then adding the absorbed count back as an explicit source
restores the total inventory but not the field: with `L` the interior operator and
`B` the Robin diagonal,

```
    (I + ΔtB)(I − ΔtL + ΔtB)⁻¹  −  (I − ΔtL)⁻¹  =  −Δt²·M⁻¹LB(M + ΔtB)⁻¹
```

a splitting error of `O(Δt²)` per step — measured at 1.7 % for `Δt = 8e-5` and
converging at the expected second-order rate. Folding `Y` into the coefficient has
no splitting at all: at `Y = 1` the assembled matrix is *bit-identical* to the
reflective one, so the reflective result is reproduced exactly rather than nearly.

Book the ledger with the **gross** `v_absorb`, not this net value: gross is what
bombards the surface and what a sputtering or recycling yield multiplies, while
net is only what the transport equation loses.

`wall_emission_source` remains the route for **cross-species** return (H⁺→H⁰,
H⁺→C⁰), where the material lands in a different equation and cannot be folded
into any one operator's coefficient.
"""
net_wall_speed(v_absorb, Y) = @. (1 - Y) * v_absorb

"""
    wall_emission_source(G, faces, emitted, dt) -> Vector

Volumetric source [m⁻³s⁻¹] that returns `emitted[f]` particles through face `f`
into the **wall-adjacent interior cell** that owns it:

```
    S_i = Σ_f  emitted_f / (V_i · Δt)
        = Σ_f  (A_f/V_i) · Γ_f
```

The factor `A_f/V_i` is the very one the Robin diagonal uses, so absorption and
re-emission are exactly reciprocal across the same face, and the total inventory
closes to machine precision.

**The density field does not.** Absorption is implicit and this source is applied
after the solve, which leaves an `O(Δt²)` splitting error — 1.7 % of the field at
`Δt = 8e-5`. For same-species return use [`net_wall_speed`](@ref) instead, which
has no splitting error at all. This function is for **cross-species** return,
where the material lands in another species' equation and folding is impossible;
there the lag is unavoidable, and the receiving species should read the source
species' `nⁿ⁺¹` so only the coupling is lagged.

Restoring the total is a weak check on its own: sending every face's return to
the *wrong* face still conserves `Σ J·n` exactly, still writes nothing outside the
wall, still keeps the density positive and still balances the ledger, while
changing the field by 13 %. Any test of this function has to compare the field.

**Never deposit outside the wall.** `treat_ion_outside_wall!` does the opposite —
it adds `γ_2nd·n_i` to cells *outside* and relies on diffusion to carry them back,
but `treat_electron_outside_wall!` books that band as loss and zeroes it at the top
of the next step. The measured yield reaching the interior is ≈ 0, controlled by
`D⊥Δt/Δx²` rather than by `γ_2nd` (`secondary_electron_test.jl`). Returning
material to the interior cell is that fix.

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
