# Transport channels: the (v∥, λ∥, v⊥, λ⊥) basis.
#
# A diffusivity cannot state a wall condition, and the reason is arithmetic:
#
#     D = ½·v·λ
#
# is one equation in two unknowns. The PDE sees only the PRODUCT — a long slow
# step and a short fast one are interchangeable in a gradient-driven net flux.
# The wall sees only the SPEED — a one-sided flux across a surface does not care
# how far a particle would have travelled next. The same D = 669 m²/s is produced
# by (v = 1115 m/s, λ = 1.2 m) and by (v = 1e6, λ = 1.3 mm), whose kinetic
# ceilings differ by a factor of 900.
#
# So a channel declares four numbers per node and everything else is derived:
#
#     D∥ = ½·v∥·λ∥          D⊥ = ½·v⊥·λ⊥
#     𝐃  = D⊥·(𝐈 − b̂b̂) + D∥·b̂b̂
#     ¼v̄_n(n̂)               the kinetic ceiling on a face with normal n̂
#
# **Nothing here is a new physical model.** Every channel in this code already
# computes its four numbers and then destroys two of them by multiplying:
#
#     neutral_gas.jl:121-122   inv_λ = ... ; return 0.5 * v_th / inv_λ
#     fields.jl:366            Dpol_turb = 0.5 * (Epol_self/Btot) * L_mixing
#                                          ½  ·      v_ExB        ·  λ
#
# The wall term this design replaces, D/(2Δx), is what happens when the discarded
# λ is silently replaced by the grid spacing. The defect is that one substitution.

"""
Ratio of the mean speed to the diffusivity-convention speed, `v̄/v = √(8/π)`.

A channel declares `v` in the convention `D = ½·v·λ`. A one-sided flux needs the
**mean** speed `v̄ = √(8T/πm)` instead, and the two differ by 1.596. Both scale as
`√(T/m)`, so no scaling argument separates them — only the absolute ratio does,
which is why it is a named constant with a test on its value rather than a `0.4`
buried in an expression.
"""
const MEAN_SPEED_FACTOR = sqrt(8 / π)

"""
    TransportChannel{FT}

One transport mechanism, described by the four per-node quantities that generate
both its diffusivity tensor and its kinetic ceiling.

- `v_para`, `λ_para`  speed and step **along** `b̂`, giving `D∥ = ½v∥λ∥`
- `v_perp`, `λ_perp`  speed and step **across** `b̂`, giving `D⊥ = ½v⊥λ⊥`

All four are full `NR×NZ` fields, not scalars: a channel's speed and step vary
with the local plasma state, and the wall condition is evaluated cell by cell.

Setting `v_para = 0` marks a channel with no parallel transport at all — Bohm and
other cross-field mechanisms. Such a channel contributes nothing to a wall the
field points straight into, which is physically right: reaching that wall requires
motion along `b̂`, which this channel does not have.

Channels with **different** characteristic speeds are independent arrival
mechanisms: both their diffusivities and their ceilings add (`total_tensor`,
`total_ceiling`). Sub-processes that share **one** speed — the gas's elastic,
ionization and wall terms, all traversed at `v_th` — combine by Matthiessen
*inside* a single channel's `λ`, and their ceiling is counted once. A useful
corollary: if a channel's `D∥` and `D⊥` do not yield the same `v`, it is not one
anisotropic channel but two.
"""
struct TransportChannel{FT <: AbstractFloat}
    v_para::Matrix{FT}
    λ_para::Matrix{FT}
    v_perp::Matrix{FT}
    λ_perp::Matrix{FT}

    function TransportChannel(
            v_para::Matrix{FT}, λ_para::Matrix{FT},
            v_perp::Matrix{FT}, λ_perp::Matrix{FT}
        ) where {FT <: AbstractFloat}
        sz = size(v_para)
        all(==(sz), (size(λ_para), size(v_perp), size(λ_perp))) ||
            throw(DimensionMismatch("all four channel fields must share a size"))
        return new{FT}(v_para, λ_para, v_perp, λ_perp)
    end
end

"Parallel diffusivity `D∥ = ½·v∥·λ∥` [m²/s], per node."
channel_D_para(ch::TransportChannel{FT}) where {FT} = @. FT(0.5) * ch.v_para * ch.λ_para

"Perpendicular diffusivity `D⊥ = ½·v⊥·λ⊥` [m²/s], per node."
channel_D_perp(ch::TransportChannel{FT}) where {FT} = @. FT(0.5) * ch.v_perp * ch.λ_perp

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
the cross term is what makes the operator a 9-point stencil that reaches diagonal
neighbours, which is where a wall boundary becomes hard.
"""
function diffusion_tensor(ch::TransportChannel, bR, bZ)
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

With `g = (b̂·n̂)²`, running from 0 when the field lies *in* the wall to 1 when it
meets the wall head-on,

```
    v̄_n(n̂) = √( v̄⊥² + (v̄∥² − v̄⊥²)·g )        v̄ = MEAN_SPEED_FACTOR·v
```

**The ceiling is a speed, not a tensor projection.** The supply side of the Robin
condition already carries the anisotropy through `𝐃`; putting a directional factor
on the ceiling *as well* double-counts it. What direction does change here is only
what a channel's own motion can deliver: `D_nn` interpolates linearly in `g`,
`v̄_n` in quadrature, and the two are different contractions of the same basis.

Two limits worth knowing, both of which fall out rather than being modelled:

- `v∥ = 0` (a cross-field channel) contributes **exactly zero** at `g = 1`.
- `v⊥ = 0` at a grazing wall, `g = sin²α`, gives `¼v̄∥·sin α` — the magnetic
  projection a presheath model is usually invoked for, here with no sheath.

`g` is squared, so the sign of `outward` is irrelevant; and because `g` is a
*face* property, the two faces of one staircase corner cell generally carry
different ceilings.
"""
function channel_ceiling(ch::TransportChannel{FT}, bR, bZ, outward::Tuple{Int, Int}) where {FT}
    nR, nZ = outward
    g = @. (bR * nR + bZ * nZ)^2
    f = FT(MEAN_SPEED_FACTOR)
    v̄_para = @. f * ch.v_para
    v̄_perp = @. f * ch.v_perp
    # max(0, ·) guards a b̂ that is not exactly normalised; g ≤ 1 analytically
    return @. FT(0.25) * sqrt(max(zero(FT), v̄_perp^2 + (v̄_para^2 - v̄_perp^2) * g))
end

"""
    total_tensor(channels, bR, bZ) -> (D_RR, D_RZ, D_ZZ)

Sum of `diffusion_tensor` over independent channels. Fluxes add, so tensors add.
"""
function total_tensor(channels, bR, bZ)
    parts = [diffusion_tensor(ch, bR, bZ) for ch in channels]
    return (sum(p[1] for p in parts), sum(p[2] for p in parts), sum(p[3] for p in parts))
end

"""
    total_tensor(channels_with_directions) -> (D_RR, D_RZ, D_ZZ)

Sum over channels that are aligned with **different** axes, given as
`((ch₁, bR₁, bZ₁), (ch₂, bR₂, bZ₂), …)`.

The three-argument method assumes every channel shares one `b̂`, which is not what
this code does: `update_diffusion_tensor!` builds its base tensor from the **full**
field `F.bR/bZ` and its turbulent tensor from the **poloidal** field
`F.bpol_R/bpol_Z`. A channel's anisotropy is defined relative to its own axis, so
combining them needs the direction to travel with the channel.
"""
function total_tensor(channels_with_directions)
    parts = [diffusion_tensor(ch, bR, bZ) for (ch, bR, bZ) in channels_with_directions]
    return (sum(p[1] for p in parts), sum(p[2] for p in parts), sum(p[3] for p in parts))
end

# ── the physical channels ───────────────────────────────────────────────────
#
# Adapters, not models. Each of these already existed inside the code as a
# `v × λ` product that was collapsed into a single D; all they do is stop the
# return from being lossy. Every one is pinned by a round-trip test against the
# diffusivity the solver already computes.

"""
    turbulent_ExB_channel(E_pol, B_tot, L_mixing, f_para, f_perp)

Anomalous ExB mixing: `D_pol = ½·v_E·L_mixing` with `v_E = E_pol/B_tot`
(`fields.jl:366`), split `f_para : f_perp` along the **poloidal** field.

One speed, two step lengths — `λ∥ = f∥·L_mixing`, `λ⊥ = f⊥·L_mixing` — so the
eddy is `f∥/f⊥ = 9` times longer along `b̂_pol` than across it at the default
split. A field-aligned eddy, which is what an ExB mixing model should produce, and
it satisfies the §2.2 consistency condition `D∥/λ∥ = D⊥/λ⊥` by construction.

Aligned with `b̂_pol`, **not** the full `b̂` — use the per-channel-direction
`total_tensor` when combining it with a collisional channel.
"""
function turbulent_ExB_channel(E_pol, B_tot, L_mixing, f_para, f_perp)
    v_E = E_pol / B_tot
    return (
        v_para = v_E, λ_para = f_para * L_mixing,
        v_perp = v_E, λ_perp = f_perp * L_mixing,
    )
end

"""
    parallel_collisional_channel(v_p, D_para)

Collisional transport along `B`: `D∥ = ½·v_p²/ν`, so `λ∥ = 2D∥/v_p = v_p/ν`.

`v_perp = 0` — streaming along the field contributes nothing across it, and
therefore nothing at a wall the field points straight into (`g = 1`). That is not
a modelling shortcut but the reason a cross-field channel and a parallel one are
*different channels*: they do not share a speed, so §2.2 says their ceilings add
rather than combining by Matthiessen.
"""
function parallel_collisional_channel(v_p, D_para)
    return (
        v_para = v_p, λ_para = 2 * D_para / v_p,
        v_perp = zero(v_p), λ_perp = zero(v_p),
    )
end

"""
    bohm_channel(Te_eV, B, m_i)

Bohm diffusion `D_B = T_e/(16B)` in the channel basis — **the one adapter whose
split is assumed rather than derived.**

`D_B` is an empirical scaling, so it fixes only the product `v⊥·λ⊥`. Reading it as
a random walk, the identity `ρ_s²ω_ci = T_e/eB` gives
`D_B = ¹⁄₁₆ρ_s²ω_ci`, and adopting `λ⊥ = ρ_s` under this design's `D = ½vλ`
convention yields

```
    v⊥ = 2D_B/ρ_s = ⅛·ρ_s·ω_ci = c_s/8       τ = λ⊥/v⊥ = 8/ω_ci
```

— a step of one sound gyroradius per ≈1.3 gyro-periods. Physically sensible, and
it reproduces `D_B` exactly by construction. At `T_e = 5` eV, `B = 0.63` T, H⁺:
`ρ_s = 0.36` mm, `v⊥ = 2.7` km/s.

**The alternatives are not absurd** — `λ⊥ = ρ_i` gives `v⊥` 2.2× larger, a
turbulent correlation length gives less — so the choice is recorded rather than
buried. It is also not load-bearing: this channel contributes ~15 % of an ion's
kinetic ceiling and ~0.2 % of an electron's, so a factor-two error in the split
barely moves either. See the design note §2.3.1, and the `@test_broken` that fires
if Bohm and the ExB channel are ever unified.

`v_para = 0`: a cross-field channel cannot reach a wall the field points into.
"""
function bohm_channel(Te_eV, B, m_i)
    c_s = sqrt(Te_eV * EE_GAS / m_i)      # EE_GAS is the elementary charge
    ω_ci = EE_GAS * B / m_i
    ρ_s = c_s / ω_ci
    return (
        v_para = zero(c_s), λ_para = zero(c_s),
        v_perp = c_s / 8, λ_perp = ρ_s,
    )
end

"""
    total_ceiling(channels, bR, bZ, outward) -> Matrix

Sum of `channel_ceiling` over independent channels — separate arrival mechanisms
each deliver their own one-sided flux, so the ceilings add.

Do **not** use this to combine sub-processes that share a speed: those belong in
one channel's `λ` by Matthiessen, and summing them would count the same particles
arriving more than once.
"""
function total_ceiling(channels, bR, bZ, outward::Tuple{Int, Int})
    return sum(channel_ceiling(ch, bR, bZ, outward) for ch in channels)
end
