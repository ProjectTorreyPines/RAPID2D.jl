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

# ── Speed conventions ───────────────────────────────────────────────────────
#
# Four speeds get called "the thermal speed" in the literature; three are used
# here. Each has ONE definition, ONE function that computes it, and ONE variable
# stem that carries it — so a call site names the moment and the arithmetic under
# it stays readable:
#
#   function                        var    formula    ×√(T/m)  what it is
#   ──────────────────────────────  ─────  ────────   ───────  ───────────────────
#   maxwellian_thermal_speed        vth_*  √(T/m)      1.000   √⟨v_z²⟩, ONE compt.
#   maxwellian_most_probable_speed  vp_*   √(2T/m)     1.414   peak of f(|v|)
#   maxwellian_mean_speed           vm_*   √(8T/πm)    1.596   ⟨|v|⟩, mean SPEED
#   (rms √⟨v²⟩ — not used here)     —      √(3T/m)     1.732
#
# WHICH ONE IS RIGHT is not a matter of taste. It follows from what the mechanism
# supplies:
#
#   vth   λ was back-derived from a MEASURED D under D = ½vλ.  NO CALLER TODAY —
#         the neutral gas used it until the D-level composition removed the
#         back-derivation entirely. Kept as the reference scale the other two are
#         quoted against, and as the yardstick `wall_test.jl` pins ¼√(8/π) to.
#   vp    D is built from a KNOWN RATE — ½vp² = T/m exactly, so
#         D = ½vp²/ν = T/(mν) reproduces Einstein
#   vm    anything CROSSING A SURFACE: ⅓vm isotropic diffusion, ½vm = ⟨|v_∥|⟩
#         along one axis, ¼vm = ⟨v_nθ(v_n)⟩ one-sided at a wall.
#         Every thermal channel declares vm — gas and collisional alike.
#
# A factor √2 between vth and vp was a live bug in this file's history. The rule
# that settles it is derived in
# `internal/docs/src/details/speed-and-composition.md`.
#
# All three take `(T_eV, m)` in eV and kg, return m/s, and guard `max(0, T)`: the
# temperature equations are free to land microscopically below zero (−1.3e-61 was
# observed), where `sqrt` raises a DomainError rather than returning NaN.

"""
    maxwellian_thermal_speed(T_eV, m) -> Float

`√(T/m)` [m/s] — the rms of a **single** velocity component, `√⟨v_z²⟩`, and the
scale appearing in `exp(−v_z²/2v_th²)`.

The smallest of the three and the one most often meant by an unqualified "thermal
speed", so the function name pins which moment is meant.

**No production caller.** The neutral gas channel used it while its `λ` was
back-derived from a measured diffusivity under `D = ½·v·λ`; composing at `D`
removed that step, and every thermal channel now declares `vm`. It is retained as
the reference scale the other two moments are quoted against (`vp = √2·vth`,
`vm = √(8/π)·vth`) and as the yardstick the wall ratio test pins `¼√(8/π)` to —
a test that would still pass if both sides drifted together, which is why the
independent definition is worth keeping.
"""
maxwellian_thermal_speed(T_eV, m) = sqrt(max(zero(T_eV), T_eV) * EE / m)

"""
    maxwellian_most_probable_speed(T_eV, m) -> Float

`√(2T/m)` [m/s] — the **most probable** speed, the peak of `f(|v|) ∝ v²exp(−mv²/2T)`.

Forced, not chosen, wherever a diffusivity is built from a **known collision
rate**: `½·vp² = T/m` exactly, so `D = ½·vp²/ν = T/(mν)` reproduces Einstein.
Any other moment here would need a different numerical prefactor, so the pair
`(½, vp)` must be read and changed together.
"""
maxwellian_most_probable_speed(T_eV, m) = sqrt(2 * max(zero(T_eV), T_eV) * EE / m)

"""
    maxwellian_mean_speed(T_eV, m) -> Float

`⟨|v|⟩ = √(8T/πm)` [m/s] — the **mean speed**, and the only one of the three that
is an observable rather than a convention.

**Anything crossing a surface is a projection of `vm`,** with the coefficient
saying which projection: `⅓vm` for isotropic 3-D diffusion, `½vm = ⟨|v_∥|⟩` along
one axis, `¼vm = ⟨v_nθ(v_n)⟩` one-sided through a wall (Hertz–Knudsen).

A channel's own `v` is bookkeeping married to its `λ` so that `½vλ` reproduces
`D`; which split was chosen is a property of how the code obtained `D`. `vm` is
not — it is fixed by `(T, m)` alone, so it is taken from `(T, m)` and never
scaled from a channel's `v`. No single `vm/v` ratio could serve every channel,
which is why the ratio is not the thing to store.

The `¼` is not a magnetization correction and never was. The one-sided flux
`∫f(v)·v_z·θ(v_z)d³v` factorizes to a 1-D integral over the wall-normal component
alone, and `∫₀^∞ v·f₁D(v)dv = ¼vm` exactly. Magnetizing the plasma changes *which
axis* is free — `ẑ` becomes `b̂` — so the `¼` survives and the field enters only
through the projection `|b̂·n̂|` in [`channel_ceiling`](@ref).
"""
maxwellian_mean_speed(T_eV, m) = sqrt(8 * max(zero(T_eV), T_eV) * EE / (π * m))

# ─── The random-step vocabulary ──────────────────────────────────────────────
#
# Every geometric bound on a diffusivity here is `D = v̄·L/d`; only `d` differs, so
# it is a type rather than a literal scattered across call sites. `d` is an angular
# average of the same `v̄` (internal/docs/src/details/speed-and-composition.md):
#
#     d = 3   isotropic,   ⟨v_z²/|v|⟩ = v̄/3   — a neutral molecule
#     d = 2   along-axis,  ⟨|v_x|⟩    = v̄/2   — a magnetized particle on b̂
#
# The wall ceiling's `¼` is not a third member: it is the `½` at the arithmetic
# mean of two distances (`wall_step_ceiling`).

"""
    RandomStepModel

Which directions a species can take its random step in. Selects the denominator
`d` in `D = v̄·L/d`; see [`geometric_diffusivity`](@ref).
"""
abstract type RandomStepModel end

"Free in all three directions: a neutral molecule. `D = v̄L/3`."
struct IsotropicStep <: RandomStepModel end

"Confined to one axis, `b̂`: a magnetized particle. `D = v̄L/2`."
struct AlongAxisStep <: RandomStepModel end

step_denominator(::IsotropicStep) = 3
step_denominator(::AlongAxisStep) = 2

# A step model is scalar config, not a container — the same declaration `Base`
# makes for `AbstractString`, so broadcasts take it as-is instead of iterating it.
Base.broadcastable(model::RandomStepModel) = Ref(model)

# Negative or NaN inputs are programming errors, and a negative product would pass
# through `min` silently as a very tight ceiling — reject at the door.
@inline function _check_step_inputs(vm, L)
    (isnan(vm) || isnan(L)) && throw(DomainError((vm, L), "speed and length must not be NaN"))
    vm < zero(vm) && throw(DomainError(vm, "mean speed must be non-negative"))
    L < zero(L) && throw(DomainError(L, "step length must be non-negative"))
    return nothing
end

# Result type from the argument TYPES, not their product: `0 * Inf` is a legitimate
# input here, and `zero(vm * L)` would be asking `zero` about a `NaN`.
@inline _step_float(vm, L) = float(promote_type(typeof(vm), typeof(L)))

"""
    geometric_diffusivity(vm, L, model::RandomStepModel)

The diffusivity a random walk of step `L` at speed `vm` can produce: `vm·L/d`.

**Zero wins over infinity**: a dead speed or a zero length means no diffusion, and
the branch is taken before the multiply so `0 * Inf` never becomes `NaN`.
"""
function geometric_diffusivity(vm, L, model::RandomStepModel)
    _check_step_inputs(vm, L)
    T = _step_float(vm, L)
    (iszero(vm) || iszero(L)) && return zero(T)
    return T(vm * L / step_denominator(model))
end

"""
    inv_geometric_diffusivity(vm, L, model::RandomStepModel)

`d/(vm·L)`, for reciprocal sums of competing mechanisms: one that cannot act
contributes `0`, one that acts instantly contributes `Inf`. Computed directly
rather than as `inv(geometric_diffusivity(...))` — the round trip costs an ulp,
and the neutral-gas refactor is required to be bit-identical.
"""
function inv_geometric_diffusivity(vm, L, model::RandomStepModel)
    _check_step_inputs(vm, L)
    T = _step_float(vm, L)
    (iszero(vm) || iszero(L)) && return T(Inf)
    return T(step_denominator(model)) / (vm * L)
end

"""
    wall_step_ceiling(vm, Lc_forward, Lc_backward)

The largest parallel diffusivity the geometry can carry: `vm·(Lf + Lb)/4`.

The two distances bound **disjoint halves of velocity space**, so they enter as
an arithmetic mean under the along-axis `½` — not harmonically; they are not
competing events. Symmetric lengths recover `½v̄L`. Either length `Inf` ⟹ that
half is unbounded and the ceiling is absent; `vm = 0` or both lengths `0` ⟹ `0`.
"""
function wall_step_ceiling(vm, Lc_forward, Lc_backward)
    _check_step_inputs(vm, Lc_forward)
    _check_step_inputs(vm, Lc_backward)
    return geometric_diffusivity(vm, (Lc_forward + Lc_backward) / 2, AlongAxisStep())
end

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

A diffusive channel, described by six per-node fields:

- `v_para`, `λ_para`  speed and step **along** `b̂`, giving `D∥ = ½v∥λ∥`
- `v_perp`, `λ_perp`  speed and step **across** `b̂`, giving `D⊥ = ½v⊥λ⊥`
- `vm_para`, `vm_perp`  mean speeds — what this mechanism **delivers to a wall**

All six are full `NR×NZ` fields: a channel's speed and step vary with the local
plasma state, and the wall condition is evaluated cell by cell.

**`v_para`/`v_perp` and `vm_para`/`vm_perp` are different kinds of
quantity, which is why they are named differently.** The first pair is bookkeeping:
only the product `½vλ = D` is physical, so a channel may split it however its
derivation forced — rescale `v` and `λ` follows, leaving `D` untouched. The second
pair is an observable, `⟨|v|⟩` of the actual distribution, fixed by `(T, m)` and
not free to be chosen. Calling both of them `v` is what once let a `√(8/π)` be
applied to the wrong one.

`vm_*` is carried rather than derived from `v_*` because it is not a function
of it — see [`maxwellian_mean_speed`](@ref). For a thermal channel it is that
function of `(T, m)`; for a channel that is not a Maxwellian at all — Bohm, ExB
mixing — there is no theorem to appeal to and the value is a modelling choice,
which is why every construction site has to state it rather than inherit one.

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
    vm_para::Matrix{FT}
    vm_perp::Matrix{FT}

    function DiffusionChannel{FT}(
            v_para::Matrix{FT}, λ_para::Matrix{FT},
            v_perp::Matrix{FT}, λ_perp::Matrix{FT},
            vm_para::Matrix{FT}, vm_perp::Matrix{FT}
        ) where {FT <: AbstractFloat}
        sz = size(v_para)
        all(==(sz), (size(λ_para), size(v_perp), size(λ_perp), size(vm_para), size(vm_perp))) ||
            throw(DimensionMismatch("all six channel fields must share a size"))
        return new{FT}(v_para, λ_para, v_perp, λ_perp, vm_para, vm_perp)
    end
end

"""
    DiffusionChannel(v_para, λ_para, v_perp, λ_perp; vm_para, vm_perp)

Build a channel from any mix of fields and scalars, expanded to their common
broadcast shape.

`vm_para` and `vm_perp` are **required keywords**, not defaulted, because there is
no default that is right for more than one channel. A default would have to be
`v̄ = c·v`, and the whole reason this field exists is that no such `c` exists.

Mixing is the normal case rather than a convenience: a channel's speed is a field
(`v_E = E_pol/B_tot`) while its step length is usually a configuration scalar
(`L_mixing`), and a cross-field channel's parallel quantities are identically
zero. Broadcasting decides the shape, so a genuine mismatch raises
`DimensionMismatch` from Julia itself rather than being silently reshaped.

At least one argument must be a matrix — a channel lives on the `NR×NZ` grid, and
an all-scalar call is more likely a mistake than a request for a 0-dimensional
channel.
"""
function DiffusionChannel(v_para, λ_para, v_perp, λ_perp; vm_para, vm_perp)
    args = (v_para, λ_para, v_perp, λ_perp, vm_para, vm_perp)
    ax = Broadcast.combine_axes(args...)
    length(ax) == 2 || throw(
        ArgumentError(
            "a channel lives on the NR×NZ grid, but these arguments broadcast to " *
                "$(length(ax)) dimension(s); pass the field quantities as matrices"
        )
    )
    FT = float(promote_type(map(eltype, args)...))
    z = zeros(FT, map(length, ax))
    return DiffusionChannel{FT}(map(a -> z .+ a, args)...)
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
    v̄_n = √( v̄⊥² + (v̄∥² − v̄⊥²)·b_n² )
```

taking the channel's own `vm_para`/`vm_perp` — see [`maxwellian_mean_speed`](@ref)
for why those are carried rather than scaled from `v_para`/`v_perp`.

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
    # max(0, ·) guards a b̂ that is not exactly normalised; b_n² ≤ 1 analytically
    return @. FT(0.25) * sqrt(
        max(zero(FT), ch.vm_perp^2 + (ch.vm_para^2 - ch.vm_perp^2) * b_n_sq)
    )
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
    # v̄ is averaged on its own rather than scaled from the averaged v: the species
    # can carry different v̄/v ratios, and the wall flux Σₛ nₛ·¼v̄ₛ is linear in v̄,
    # so the density-weighted mean is what reproduces it — the same argument that
    # makes the mixture exact at b_n = 0 and 1.
    vm_para = zeros(FT, sz)
    vm_perp = zeros(FT, sz)
    for (ch, w) in zip(channels, weights)
        w̃ = @. ifelse(empty, one(FT), FT(w))
        @. v_para += w̃ * ch.v_para
        @. v_perp += w̃ * ch.v_perp
        @. vm_para += w̃ * ch.vm_para
        @. vm_perp += w̃ * ch.vm_perp
        @. D_para += w̃ * FT(0.5) * ch.v_para * ch.λ_para
        @. D_perp += w̃ * FT(0.5) * ch.v_perp * ch.λ_perp
    end
    @. v_para /= norm
    @. v_perp /= norm
    @. vm_para /= norm
    @. vm_perp /= norm
    @. D_para /= norm
    @. D_perp /= norm

    # v = 0 forces D = 0, since every vₛ ≥ 0 makes the mean vanish only when all
    # do, and then every Dₛ = ½vₛλₛ vanishes too. λ = 0 is the consistent value.
    λ_para = @. ifelse(v_para > zero(FT), 2 * D_para / v_para, zero(FT))
    λ_perp = @. ifelse(v_perp > zero(FT), 2 * D_perp / v_perp, zero(FT))
    return DiffusionChannel{FT}(v_para, λ_para, v_perp, λ_perp, vm_para, vm_perp)
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
    # An ExB eddy is not a Maxwellian, so `maxwellian_mean_speed` has nothing to say
    # about how fast it delivers to a wall and `v̄ = √(8/π)·v_E` is a modelling
    # choice, not a theorem. It is written out here rather than inherited from a
    # shared constant so that the choice is visible where it is made. A cross-field
    # delivery model would replace it; see internal/docs/src/notes/TODO/wall-boundary-conditions.md.
    vm_E = @. sqrt(8 / π) * v_E
    return DiffusionChannel(
        v_E, (@. f_para * L_mixing), v_E, (@. f_perp * L_mixing);
        vm_para = vm_E, vm_perp = vm_E
    )
end

"""
    parallel_collisional_channel(vp_s, D_para)

Collisional transport along `B`: `D∥ = ½·v²/ν`, so `λ∥ = 2D∥/v = v/ν`.

`vp_s` is contracted to be [`maxwellian_most_probable_speed`](@ref),
`√(2T/m)` — that is the only moment for which `½v²/ν` equals `T/(mν)`, so it is
forced by the derivation rather than chosen. The parameter is named for the moment
so that a caller passing `√(T/m)` is making a visible mistake rather than a silent
factor of `√2`.

`v_perp = 0` — streaming along the field contributes nothing across it, and so
nothing at a wall the field points straight into. That is also why a parallel and
a cross-field mechanism are two channels rather than one anisotropic channel: they
do not share a speed, so their ceilings add.
"""
function parallel_collisional_channel(vp_s, D_para)
    # ⟨|v|⟩/√(2T/m) = √(8/π)/√2 = √(4/π) = 1.128 — NOT the √(8/π) = 1.596 that
    # applies to a channel declaring √(T/m). Both are Maxwellian; they differ only
    # in which moment the channel chose to name.
    vm_s = @. sqrt(4 / π) * vp_s
    return DiffusionChannel(
        vp_s, (@. 2 * D_para / vp_s),
        zero.(vp_s), zero.(vp_s);
        vm_para = vm_s, vm_perp = zero.(vp_s)
    )
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
    c_s = @. sqrt(max(zero(eltype(Te_eV)), Te_eV * EE / m_i))   # EE is the elementary charge
    # `abs(B)`: `Bϕ = R0B0/R` carries the sign of the user's R0B0, and a signed
    # ω_ci would hand back a negative ρ_s — hence a negative λ⊥ and a negative
    # D⊥ = ½v⊥λ⊥, i.e. ANTI-diffusion, for a field that merely points the other
    # way. `D_B` is a magnitude; the gyration sense does not enter it. The
    # electron path states the same thing as `abs(Te/Bϕ)` in `transport.jl`.
    ω_ci = @. Z * EE * abs(B) / m_i
    ρ_s = @. c_s / ω_ci
    # Bohm is an empirical scaling, not a Maxwellian, so nothing derives its wall
    # delivery speed. `√(8/π)·(c_s/8)` preserves the value this channel has always
    # contributed and is a modelling choice held pending a cross-field delivery
    # model; the Bohm sheath criterion would argue for `c_s` itself, an 8× move.
    return DiffusionChannel(
        zero.(c_s), zero.(c_s), (@. c_s / 8), ρ_s;
        vm_para = zero.(c_s), vm_perp = (@. sqrt(8 / π) * c_s / 8)
    )
end

# ── the free-streaming ceiling ──────────────────────────────────────────────
#
# Not a channel. Everything above this line is a transport MECHANISM, and
# mechanisms compose harmonically because they are competing termination events
# (`speed-and-composition.md` §4). A density gradient terminates nothing — a
# particle in a steep profile does not collide more often — so the ceiling below
# is a different kind of object: a bound on the TOTAL flux, applied after the
# mechanisms have composed and after any base diffusivity has been added.
#
# Putting it in the harmonic sum instead is what let `Dpara0` escape it.

"""
    flux_limited_diffusivity(D, ∇n, n, vm, flux_limit_factor) -> Float

`D` capped so that the diffusive flux cannot exceed what a Maxwellian can supply,

```
    Γ = D·|∇n| ≤ flux_limit_factor·n·v̄        flux_limit_factor = ¼ is Hertz–Knudsen
```

composed as the **`n = 2`** member of Larsen's flux-limited-diffusion family,

```
    1/D_limited = √( (1/D)² + (|∇n| / (flux_limit_factor·vm·n))² )
```

`flux_limit_factor` is `RP.flags.limit_flux.factor`, spelled out rather than `α` because
`α` is this codebase's Townsend first-ionization coefficient
(`reaction_rate_coefficients.jl`).

`vm` is contracted to be [`maxwellian_mean_speed`](@ref): the ceiling is a one-way
flux across a surface, `⟨v_n θ(v_n)⟩ = ¼v̄`, and only that moment has the meaning.
The same `¼v̄` the Robin wall condition uses, now multiplied by the gradient scale
rather than a geometric one.

**Why `n = 2`.** For `x ≡ D/D_max` the family expands as
`D_n/D = 1 − xⁿ/n + O(x²ⁿ)`, so the exponent *is* the order of the leading
correction. This code builds `D∥ = ½vp²/ν = T/(mν)` from a single
velocity-independent rate — the BGK closure — whose exact linear response is
`1 − 1.910R²` with `R = λ/Lₙ`: second order, because parity plus analyticity of the
kernel forbid an odd term. `n = 1` manufactures a first-order term the kinetics do
not have (0.6 % low against a true 0.002 % at this device's operating point, 15 %
low at `R = 0.1` — figures from
`internal/docs/src/notes/design/flux-limiter-review.md` §1.A and Appendix A);
`n = ∞` has none at all and errs the other way. `n = 2` gives `1 − 2.000R²`, and
minimax over `R ∈ [0.003, 0.3]` picks 2.04, so the integer is a rounding rather
than a fit. `n = 1` is not smooth at `∇n = 0`, and `n = 2` is: `n = 1`'s
`D ∝ (a + b|g|)⁻¹` has unequal one-sided derivatives there, while `n = 2`'s
`D ∝ (a² + b²g²)^(-1/2)` depends on `g²` and does not — and `∇n ≈ 0` is where most
of the grid sits most of the time. Derived in
`internal/docs/src/notes/design/flux-limiter.md` §1.

**`Lₙ = n/|∇n|` is never formed**, which is the whole structural point: the guard
that used to patch its non-finite branch had the sign backwards, mapping a flat
profile (`Lₙ = ∞`, no limit needed) to `D = 0` (maximum limit). What replaces it is
not a set of tamed special cases but the same arithmetic falling through every one
of them without a branch — a claim about *how* each case is handled, not that every
one lands on a finite number:

| state | returns | why |
|---|---|---|
| `n > 0`, `∇n = 0` | `D` unchanged, bit for bit | flat profile, Fick at its most valid |
| `n ≤ 0` | `0`, whatever `∇n` does | an empty cell has nothing to supply. **The supply test runs before the flat-profile short-circuit, and the order is load-bearing:** the bound is `D·\\|∇n\\| ≤ ¼n v̄`, whose right-hand side is exactly `0` at `n = 0`. A flat profile then admits any *finite* `D` but not `D = Inf`, since `Inf·0` is `NaN` rather than `≤ 0`. Checking supply first is what makes a vacuum — no gas, no plasma, so `½vp²/ν = Inf` natively — come out at `D = 0` instead of poisoning `∇𝐃∇`. |
| `n < 0` | `0` | transient, before `negative_n_correction`; `max(n,0)` folds it into the row above rather than letting it behave like `\\|n\\|` |
| `vm = 0` | `0` | `T = 0`, nothing crosses any surface |
| `D = 0` | `0` | already immobile |
| `n > 0`, `∇n = 0`, `D = Inf` | `Inf`, unchanged | **not tamed, and correctly so.** A uniform collisionless plasma really does have no diffusive description; the ceiling has no flux to bound and declines to act. Reachable only with `ν = 0` at live density — no fixture in the suite reaches it. Bounding it needs a *geometric* length (the mfp-truncation ceiling the gas and ion channels already carry as `L_char`/`L_mixing`), which is separate work — see `flux-limiter.md` §1. |

**`NaN` is not one of the branches above, and is not caught.** None of the
following is reachable in production today; recorded because the four fail in
inconsistent directions, not to guard against something that cannot currently
happen. Measured: `∇n = NaN` returns `D` unchanged — the short-circuit's `g > 0`
compares `false` for `NaN`, so this fails **open**, silently disabling the limiter.
`n = NaN` or `vm = NaN` returns `0.0` — `NaN` poisons `supply = flux_limit_factor·vm·max(n,0)`, and
`supply > 0` is again `false` for `NaN`, so this fails **closed**, the opposite
direction. `D = NaN` returns `NaN`, propagated through `hypot` rather than caught
either way.

Two more edges, both outside anything a caller is expected to hit: `D` at or below
`floatmax()`'s reciprocal (`≈ 5.56e-309`) returns exactly `0.0`, because `inv(D)`
itself overflows to `Inf` there before `hypot` ever runs. And a negative `D`
returns a *positive* value of the same magnitude `-D` would — `hypot` only ever
sees `inv(D)` squared, so the sign is lost rather than propagated. Unreachable
given `Dpara0 ≥ 0` and `Dpara_e_eff ≥ 0` upstream, but worth naming as a sign flip
rather than assumed to be graceful degradation.

`hypot` rather than `sqrt(a^2 + b^2)`: what actually arrives here is `D = Inf`, not
a subnormal-adjacent value — `Dpara_e_coll` produces it natively whenever the
collision frequency vanishes, and `inv(Inf) === 0.0` exactly, so that arrival needs
no special handling. The regime `hypot` defends is `D = floatmax()`, the worst
finite input the test suite pins rather than one this code's own arithmetic
produces: there `inv(D)` is genuinely subnormal, and the naive
`sqrt(inv(D)^2 + (g/supply)^2)` would underflow `inv(D)^2` to exactly `0.0` before
the `sqrt` runs — underflow, not the overflow an earlier version of this note
claimed. `hypot` carries the small term through without that loss.
"""
function flux_limited_diffusivity(D, ∇n, n, vm, flux_limit_factor)
    # Supply first, gradient second. An empty cell has nothing to carry a flux with
    # whatever the profile does, and taking that branch before the flat-profile
    # short-circuit is what keeps `D·∇n` from being `Inf·0 = NaN` in a vacuum.
    supply = flux_limit_factor * vm * max(n, zero(n))  # ¼·n·v̄, the one-way flux ceiling
    supply > zero(supply) || return zero(D)
    g = abs(∇n)
    # short-circuit, not `ifelse`: this must return D untouched rather than round
    # trip through inv(hypot(inv(D), 0)) and lose an ulp
    g > zero(g) || return D
    return inv(hypot(inv(D), g / supply))
end
