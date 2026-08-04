# Transport channels: the (v∥, λ∥, v⊥, λ⊥) basis.
#
# A diffusivity cannot state a wall condition, because
#
#     D = ½·v·λ
#
# is one equation in two unknowns. The PDE sees only the PRODUCT — a long slow
# step and a short fast one are interchangeable in a gradient-driven net flux.
# The wall sees only the SPEED — a one-sided flux across a surface does not care
# how far a particle would have travelled next. The same D = 669 m²/s comes from
# (v = 1115 m/s, λ = 1.2 m) and from (v = 1e6 m/s, λ = 1.3 mm), whose kinetic
# ceilings differ by a factor of 900.
#
# Nothing here is a new physical model: every channel in this code already
# computes v and λ and then destroys them by multiplying (`neutral_gas_diffusivity`,
# the `Dpol_turb` assignment in `fields.jl`). The wall coefficient this replaces,
# D/(2Δx), is what happens when the discarded λ is silently replaced by the grid
# spacing.

"""
Ratio of the mean speed to the diffusivity-convention speed, `v̄/v = √(8/π)`.

A channel declares `v` in the convention `D = ½·v·λ`, but a one-sided flux needs
the **mean** speed `v̄ = √(8T/πm)`. Both scale as `√(T/m)`, so no scaling argument
separates them — only this absolute ratio of 1.596 does, which is why it is a
named constant with a test rather than a `0.4` buried in an expression.
"""
const MEAN_SPEED_FACTOR = sqrt(8 / π)

"""
    TransportChannel{FT}

One physical transport mechanism, described by the characteristic quantities that
generate everything else about it.

The defining property of the type is that instances **add**: separate mechanisms
are separate arrival paths, so their contributions superpose. Sub-processes that
share one characteristic speed are *not* separate channels — they combine inside
a single channel (by Matthiessen, for a mean free path) and their one-sided flux
is counted once.

`DiffusionChannel` is the only subtype today; a drift or convection mechanism
would be another.
"""
abstract type TransportChannel{FT <: AbstractFloat} end

"""
    DiffusionChannel{FT}

A diffusive channel, described by four per-node fields:

- `v_para`, `λ_para`  speed and step **along** `b̂`, giving `D∥ = ½v∥λ∥`
- `v_perp`, `λ_perp`  speed and step **across** `b̂`, giving `D⊥ = ½v⊥λ⊥`

All four are full `NR×NZ` fields: a channel's speed and step vary with the local
plasma state, and the wall condition is evaluated cell by cell.

`v_para = 0` marks a channel with no parallel transport — Bohm and other
cross-field mechanisms. Such a channel contributes nothing at a wall the field
points straight into, which is right: reaching that wall requires motion along
`b̂`, which this channel does not have.

A useful corollary of the combination rule in [`TransportChannel`](@ref): if a
channel's `D∥` and `D⊥` do not yield the same `v`, it is not one anisotropic
channel but two.
"""
struct DiffusionChannel{FT <: AbstractFloat} <: TransportChannel{FT}
    v_para::Matrix{FT}
    λ_para::Matrix{FT}
    v_perp::Matrix{FT}
    λ_perp::Matrix{FT}

    function DiffusionChannel{FT}(
            v_para::Matrix{FT}, λ_para::Matrix{FT},
            v_perp::Matrix{FT}, λ_perp::Matrix{FT}
        ) where {FT <: AbstractFloat}
        sz = size(v_para)
        all(==(sz), (size(λ_para), size(v_perp), size(λ_perp))) ||
            throw(DimensionMismatch("all four channel fields must share a size"))
        return new{FT}(v_para, λ_para, v_perp, λ_perp)
    end
end

"""
    DiffusionChannel(v_para, λ_para, v_perp, λ_perp)

Build a channel from any mix of fields and scalars, expanded to their common
broadcast shape.

Mixing is the normal case rather than a convenience: a channel's speed is a field
(`v_E = E_pol/B_tot`) while its step length is usually a configuration scalar
(`L_mixing`), and a cross-field channel's parallel quantities are identically
zero. Broadcasting decides the shape, so a genuine mismatch raises
`DimensionMismatch` from Julia itself rather than being silently reshaped.

At least one argument must be a matrix — a channel lives on the `NR×NZ` grid, and
an all-scalar call is more likely a mistake than a request for a 0-dimensional
channel.
"""
function DiffusionChannel(v_para, λ_para, v_perp, λ_perp)
    ax = Broadcast.combine_axes(v_para, λ_para, v_perp, λ_perp)
    length(ax) == 2 || throw(
        ArgumentError(
            "a channel lives on the NR×NZ grid, but these arguments broadcast to " *
                "$(length(ax)) dimension(s); pass the field quantities as matrices"
        )
    )
    FT = float(promote_type(map(eltype, (v_para, λ_para, v_perp, λ_perp))...))
    z = zeros(FT, map(length, ax))
    return DiffusionChannel{FT}(z .+ v_para, z .+ λ_para, z .+ v_perp, z .+ λ_perp)
end

"Parallel diffusivity `D∥ = ½·v∥·λ∥` [m²/s], per node."
channel_D_para(ch::DiffusionChannel{FT}) where {FT} = @. FT(0.5) * ch.v_para * ch.λ_para

"Perpendicular diffusivity `D⊥ = ½·v⊥·λ⊥` [m²/s], per node."
channel_D_perp(ch::DiffusionChannel{FT}) where {FT} = @. FT(0.5) * ch.v_perp * ch.λ_perp

"""
    diffusion_tensor(ch, bR, bZ) -> (D_RR, D_RZ, D_ZZ)

Components of `𝐃 = D⊥(𝐈 − b̂b̂) + D∥b̂b̂` in the `(R,Z)` plane:

```
    D_RR = D⊥ + (D∥ − D⊥)·b_R²
    D_RZ =      (D∥ − D⊥)·b_R·b_Z
    D_ZZ = D⊥ + (D∥ − D⊥)·b_Z²
```

`bR`, `bZ` may be fields or scalars. The result is symmetric positive
semi-definite with eigenvalues `D∥` and `D⊥` and `b̂` as the `D∥` eigenvector.

**`D_RZ` vanishes whenever `b̂` lies on a grid axis**, so an axis-aligned test
passes even with the cross term deleted. Only an oblique field exercises it — and
the cross term is what makes the operator a 9-point stencil reaching diagonal
neighbours, which is where a wall boundary becomes hard.
"""
function diffusion_tensor(ch::DiffusionChannel, bR, bZ)
    D_para = channel_D_para(ch)
    D_perp = channel_D_perp(ch)
    D_RR = @. D_perp + (D_para - D_perp) * bR^2
    D_RZ = @. (D_para - D_perp) * bR * bZ
    D_ZZ = @. D_perp + (D_para - D_perp) * bZ^2
    return D_RR, D_RZ, D_ZZ
end

"""
    channel_ceiling(ch, bR, bZ, outward) -> Matrix

Kinetic ceiling `¼v̄_n` [m/s] this channel imposes on a face whose outward normal
is the index step `outward` (one of `(±1,0)`, `(0,±1)` — a `WallFace.outward`).

With `b_n = b̂·n̂` the field component normal to the wall, running from 0 when the
field lies *in* the wall to ±1 when it meets the wall head-on,

```
    v̄_n = √( v̄⊥² + (v̄∥² − v̄⊥²)·b_n² )        v̄ = MEAN_SPEED_FACTOR·v
```

**The ceiling is a speed, not a tensor projection.** The supply side of the Robin
condition already carries the anisotropy through `𝐃`; putting a directional factor
on the ceiling as well would double-count it. Direction enters only through what
the channel's own motion can deliver: `n̂·𝐃·n̂` interpolates linearly in `b_n²`,
`v̄_n` in quadrature.

Two limits fall out rather than being modelled: a cross-field channel (`v∥ = 0`)
contributes exactly zero at a head-on wall, and `v⊥ = 0` at a grazing wall gives
`¼v̄∥·sin α` — the magnetic projection a presheath model is usually invoked for.

`b_n` is squared, so the sign of `outward` is irrelevant; and because it is a
*face* property, the two faces of one staircase corner cell generally carry
different ceilings.
"""
function channel_ceiling(ch::DiffusionChannel{FT}, bR, bZ, outward::Tuple{Int, Int}) where {FT}
    nR, nZ = outward
    b_n_sq = @. (bR * nR + bZ * nZ)^2
    f = FT(MEAN_SPEED_FACTOR)
    v̄_para = @. f * ch.v_para
    v̄_perp = @. f * ch.v_perp
    # max(0, ·) guards a b̂ that is not exactly normalised; b_n² ≤ 1 analytically
    return @. FT(0.25) * sqrt(max(zero(FT), v̄_perp^2 + (v̄_para^2 - v̄_perp^2) * b_n_sq))
end

"""
    total_tensor(channels, bR, bZ) -> (D_RR, D_RZ, D_ZZ)

Sum of `diffusion_tensor` over independent channels sharing one `b̂`. Fluxes add,
so tensors add.
"""
function total_tensor(channels, bR, bZ)
    parts = [diffusion_tensor(ch, bR, bZ) for ch in channels]
    return (sum(p[1] for p in parts), sum(p[2] for p in parts), sum(p[3] for p in parts))
end

"""
    total_tensor(channels_with_directions) -> (D_RR, D_RZ, D_ZZ)

Sum over channels aligned with **different** axes, given as
`((ch₁, bR₁, bZ₁), (ch₂, bR₂, bZ₂), …)`.

`update_diffusion_tensor!` builds its base tensor from the **full** field
`F.bR/bZ` and its turbulent tensor from the **poloidal** field `F.bpol_R/bpol_Z`,
so the single-`b̂` method above does not describe it. A channel's anisotropy is
defined relative to its own axis, and the direction has to travel with it.
"""
function total_tensor(channels_with_directions)
    parts = [diffusion_tensor(ch, bR, bZ) for (ch, bR, bZ) in channels_with_directions]
    return (sum(p[1] for p in parts), sum(p[2] for p in parts), sum(p[3] for p in parts))
end

"""
    mixture_channel(channels, weights) -> DiffusionChannel

Population-weighted mean of channels describing the **same** mechanism carried by
**different** species, so that one operator can stand for all of them.

Contrast [`total_tensor`](@ref), which *sums*. Summing is for independent arrival
paths of one population; averaging is for one path shared by several populations,
where the result must reproduce the mixture's total flux, not several times it.

Two quantities are averaged, each because a flux is linear in it:

```
    v = Σ wₛvₛ / Σ wₛ          a one-sided wall flux is Σ ¼v̄ₛnₛ
    D = Σ wₛDₛ / Σ wₛ          a diffusive flux is Σ Dₛ∇nₛ
```

and the step follows as `λ = 2D/v`. Averaging `λ` directly instead would conserve
neither. `weights` are per-node densities; only their **ratio** matters, so the
mixture does not drift as a discharge climbs nine orders in density.

Where every weight vanishes the unweighted mean is used. Such a node is empty
*now* and is exactly where material is about to arrive, so a `0/0` — or a
`D = 0`, which would be an artificial barrier at the plasma edge — must not be
what the operator sees there.

Channels that are the same object are returned as that object: a mass- and
charge-free mechanism is genuinely shared, and this makes it free and exact
rather than merely cheap and close.

## What is exact and what is not

`D` is reproduced exactly whenever the species agree on it — which covers Bohm
(`D_B = T_e/16B`, mass cancels) and ExB mixing (`v_E = E_pol/B_tot`), leaving
`D∥` as the only mechanism the mixture actually approximates.

The wall ceiling is exact only at the two limits. `¼v̄_n` interpolates in
quadrature, which is linear in `v` at `b_n = 0` and `b_n = 1` but concave in
between, so an oblique wall is under-supplied by the mixture — 0.2 % at 45° for a
2:1 speed ratio.
"""
function mixture_channel(
        channels::AbstractVector{<:DiffusionChannel},
        weights::AbstractVector{<:AbstractMatrix}
    )
    isempty(channels) && throw(ArgumentError("a mixture needs at least one channel"))
    length(channels) == length(weights) ||
        throw(ArgumentError("got $(length(channels)) channels but $(length(weights)) weights"))

    sz = size(first(channels).v_para)
    all(ch -> size(ch.v_para) == sz, channels) ||
        throw(DimensionMismatch("all channels in a mixture must share a size"))
    all(w -> size(w) == sz, weights) ||
        throw(DimensionMismatch("each weight must be an $(sz) field, one per node"))

    all(ch -> ch === first(channels), channels) && return first(channels)

    FT = eltype(first(channels).v_para)
    ns = FT(length(channels))
    Σw = sum(weights)
    empty = Σw .<= zero(FT)
    norm = @. ifelse(empty, ns, Σw)

    v_para = zeros(FT, sz)
    v_perp = zeros(FT, sz)
    D_para = zeros(FT, sz)
    D_perp = zeros(FT, sz)
    for (ch, w) in zip(channels, weights)
        w̃ = @. ifelse(empty, one(FT), FT(w))
        @. v_para += w̃ * ch.v_para
        @. v_perp += w̃ * ch.v_perp
        @. D_para += w̃ * FT(0.5) * ch.v_para * ch.λ_para
        @. D_perp += w̃ * FT(0.5) * ch.v_perp * ch.λ_perp
    end
    @. v_para /= norm
    @. v_perp /= norm
    @. D_para /= norm
    @. D_perp /= norm

    # v = 0 forces D = 0, since every vₛ ≥ 0 makes the mean vanish only when all
    # do, and then every Dₛ = ½vₛλₛ vanishes too. λ = 0 is the consistent value.
    λ_para = @. ifelse(v_para > zero(FT), 2 * D_para / v_para, zero(FT))
    λ_perp = @. ifelse(v_perp > zero(FT), 2 * D_perp / v_perp, zero(FT))
    return DiffusionChannel{FT}(v_para, λ_para, v_perp, λ_perp)
end

"""
    total_ceiling(channels, bR, bZ, outward) -> Matrix

Sum of `channel_ceiling` over independent channels — separate arrival mechanisms
each deliver their own one-sided flux.

Do **not** use this for sub-processes that share a speed: those belong in one
channel's `λ`, and summing them would count the same particles arriving twice.
"""
function total_ceiling(channels, bR, bZ, outward::Tuple{Int, Int})
    return sum(channel_ceiling(ch, bR, bZ, outward) for ch in channels)
end

"""
    total_ceiling(channels_with_directions, outward) -> Matrix

Sum of `channel_ceiling` over channels aligned with **different** axes, given as
`((ch₁, bR₁, bZ₁), (ch₂, bR₂, bZ₂), …)` — the ceiling counterpart of the
same-shaped [`total_tensor`](@ref) method, and the form to use whenever a
collisional channel (along `b̂`) sits beside a turbulent one (along `b̂_pol`).
"""
function total_ceiling(channels_with_directions, outward::Tuple{Int, Int})
    return sum(channel_ceiling(ch, bR, bZ, outward) for (ch, bR, bZ) in channels_with_directions)
end

# ── the physical channels ───────────────────────────────────────────────────
#
# Adapters, not models. Each already existed in the code as a `v × λ` product
# collapsed into a single D; these stop the return from being lossy, and each is
# pinned by a round-trip test against the diffusivity the solver already computes.
#
# All arithmetic here is elementwise. These take fields, and `/` between two
# matrices is right-division, not division — `E_pol / B_tot` on a square grid
# throws `SingularException`, and on a non-square one it silently returns an
# array of the wrong shape. Every expression below is written under `@.` for that
# reason, and the non-square case is a regression test.

"""
    turbulent_ExB_channel(E_pol, B_tot, L_mixing, f_para, f_perp)

Anomalous ExB mixing: `D_pol = ½·v_E·L_mixing` with `v_E = E_pol/B_tot`, split
`f_para : f_perp` along the **poloidal** field.

One speed, two step lengths — `λ∥ = f∥·L_mixing`, `λ⊥ = f⊥·L_mixing` — so at the
default split the eddy is 9× longer along `b̂_pol` than across it. A field-aligned
eddy, which is what an ExB mixing model should produce.

Aligned with `b̂_pol`, **not** the full `b̂`: use the per-channel-direction
`total_tensor` when combining it with a collisional channel.
"""
function turbulent_ExB_channel(E_pol, B_tot, L_mixing, f_para, f_perp)
    v_E = @. E_pol / B_tot
    return DiffusionChannel(v_E, (@. f_para * L_mixing), v_E, (@. f_perp * L_mixing))
end

"""
    parallel_collisional_channel(v_p, D_para)

Collisional transport along `B`: `D∥ = ½·v_p²/ν`, so `λ∥ = 2D∥/v_p = v_p/ν`.

`v_perp = 0` — streaming along the field contributes nothing across it, and so
nothing at a wall the field points straight into. That is also why a parallel and
a cross-field mechanism are two channels rather than one anisotropic channel: they
do not share a speed, so their ceilings add.
"""
function parallel_collisional_channel(v_p, D_para)
    return DiffusionChannel(v_p, (@. 2 * D_para / v_p), zero.(v_p), zero.(v_p))
end

"""
    bohm_channel(Te_eV, B, m_i)

Bohm diffusion `D_B = T_e/(16B)` in the channel basis — **the one adapter whose
split is assumed rather than derived.**

`D_B` is an empirical scaling, so it fixes only the product `v⊥·λ⊥`. Reading it as
a random walk, `ρ_s²ω_ci = T_e/eB` gives `D_B = ¹⁄₁₆ρ_s²ω_ci`, and adopting
`λ⊥ = ρ_s` under the `D = ½vλ` convention yields

```
    v⊥ = 2D_B/ρ_s = ⅛·ρ_s·ω_ci = c_s/8       τ = λ⊥/v⊥ = 8/ω_ci
```

— a step of one sound gyroradius per ≈1.3 gyro-periods, reproducing `D_B` exactly
by construction. At `T_e = 5` eV, `B = 0.63` T, H⁺: `ρ_s = 0.36` mm, `v⊥ = 2.7` km/s.

**Mass-free, but not charge-free.** `ρ_s²ω_ci = T_e/(ZeB)`, so `D_B = T_e/(16ZB)`:
the mass cancels exactly and every ion species shares `D⊥` only at equal charge.
A six-times-charged ion diffuses across the field six times more slowly.

The alternatives are not absurd (`λ⊥ = ρ_i` gives `v⊥` 2.2× larger, a turbulent
correlation length gives less), so the choice is recorded rather than buried. It is
also not load-bearing: this channel contributes ~15 % of an ion's kinetic ceiling
and ~0.2 % of an electron's. A `@test_broken` fires if Bohm and the ExB channel are
ever unified.

`v_para = 0`: a cross-field channel cannot reach a wall the field points into.
"""
function bohm_channel(Te_eV, B, m_i, Z::Integer = 1)
    # max(0, ·) because a temperature equation may land microscopically below
    # zero, where `sqrt` raises rather than returning NaN
    c_s = @. sqrt(max(zero(eltype(Te_eV)), Te_eV * EE_GAS / m_i))   # EE_GAS is the elementary charge
    # `abs(B)`: `Bϕ = R0B0/R` carries the sign of the user's R0B0, and a signed
    # ω_ci would hand back a negative ρ_s — hence a negative λ⊥ and a negative
    # D⊥ = ½v⊥λ⊥, i.e. ANTI-diffusion, for a field that merely points the other
    # way. `D_B` is a magnitude; the gyration sense does not enter it. The
    # electron path states the same thing as `abs(Te/Bϕ)` in `transport.jl`.
    ω_ci = @. Z * EE_GAS * abs(B) / m_i
    ρ_s = @. c_s / ω_ci
    return DiffusionChannel(zero.(c_s), zero.(c_s), (@. c_s / 8), ρ_s)
end
