export reset_reaction_counts!,
    update_reaction_counts!,
    check_reaction_counts,
    reaction_θ,
    net_electron_count,
    net_ion_count,
    net_H2_gas_count

# Contracting reaction EVENT counts into per-species particle changes.
#
# The struct and the stoichiometry table live in `types.jl`; this is the arithmetic
# `ΔNₛ = Σₖ νₖ,ₛ Nₖ` that turns one into the other, plus the freshness check.
#
# Written from `REACTION_STOICHIOMETRY` and kept honest against it by
# `reactions_test.jl`, which walks the table and re-derives every accessor.
# With one channel these are one-liners; with four they are four-term sums, which
# is still both readable and type-stable, so no contraction engine is warranted.

"""
    reset_reaction_counts!(RP) -> RP

Void the previous advance's reaction counts. **Call at the start of a step.**

This is the lifecycle, and it is what makes producer-before-consumer a structural
property rather than an ordering convention:

```
    reset_reaction_counts!   →   update_reaction_counts!  →   net_*_count
    (advance_timestep!)          (electron continuity)        (ion source,
     unpublishes                  publishes                     ledgers, gas sink)
```

Deliberately **not** keyed on `RP.step`. A step number looks like a validity
token and is not one: `run_simulation!` increments `RP.step` between
`advance_timestep!` and the wall passes, so those consumers — which belong to the
step just advanced — would compare against the next step's number and fail. An
explicit reset has no such coincidence to get right.

Counts are zeroed as well as unpublished, so nothing that bypasses
[`check_reaction_counts`](@ref) (a snapshot, a debugger) can read last step's
numbers and believe them.
"""
function reset_reaction_counts!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    rx = RP.reactions
    empty!(rx.published)
    fill!(rx.counts.iz, zero(FT))
    return RP
end

"""
    reaction_θ(flags, channel) -> FT

The θ channel `channel`'s quadrature uses, from `REACTION_STOICHIOMETRY`'s `θ`
family and [`ImplicitWeights`](@ref). Zero for an explicit run.

θ is a statement about **accuracy, not about time**. What is stored is the
definite integral over the step,

```
    Nₖ = ∫ₜⁿ^ₜⁿ⁺¹ … dt ≈ Δt·[(1−θ)·(…)ⁿ + θ·(…)ⁿ⁺¹],
```

so "how many events happened between `tⁿ` and `tⁿ⁺¹`" is unambiguous whatever θ
is, and θ only says how good the quadrature was — trapezoid at ½, one-sided
rectangle at 0 or 1. Storing a *rate* instead would have left a genuine question
("centred when?") whose answer varied per channel: `:growth` uses ½ while
`:decay` will use 1.

Exposed because a consumer may still care how the integral was formed — an
energy term pairing `E(Tₑⁿ⁺¹)` with a channel evaluated at ½ is inconsistent at
`O(Δt)` even though the particle count is not.
"""
function reaction_θ(flags::SimulationFlags{FT}, channel::Symbol) where {FT <: AbstractFloat}
    haskey(REACTION_STOICHIOMETRY, channel) ||
        throw(ArgumentError("no reaction channel called $channel"))
    flags.Implicit || return zero(FT)
    return getproperty(flags.θ_imp, REACTION_STOICHIOMETRY[channel].θ)
end

"""
    update_reaction_counts!(RP)

Write this step's reaction event counts. **The single producer.**

Called at the end of [`solve_electron_continuity_equation!`](@ref), which is the
only place that knows `n* = (1−θ)nⁿ + θnⁿ⁺¹` — the density the electron equation
actually ionized at — because it holds `θ`, `prev_n` and the just-solved `ne` in
one scope. Anywhere else it would have to be reconstructed from state that may
have moved, which is the failure this replaces.

`N_iz = Δt·ν_en_iz·n*` with `ν_en_iz` as `update_RRCs!` materialized it at the
step-entry state; the tables are not re-queried here. Δt is baked in on purpose —
a count is what it is, and a consumer cannot accidentally scale it by a step
length other than the one it was formed with.

With `flags.src` off the counts are zeroed rather than left stale, so a run that
switches the source off stops creating particles on the same step.
"""
function update_reaction_counts!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    rx, pla = RP.reactions, RP.plasma
    N = rx.counts

    # ── e + H₂ → 2e + H₂⁺ ───────────────────────────────────────────────────
    if RP.flags.src
        # θ comes from the table, not from this line: a `:decay` channel added
        # later then picks up backward Euler by existing.
        θ = reaction_θ(RP.flags, :iz)
        dt = RP.dt
        @. N.iz = dt * ((one(FT) - θ) * RP.prev_n + θ * pla.ne) * pla.ν_en_iz
    else
        fill!(N.iz, zero(FT))
    end
    push!(rx.published, :iz)

    return RP
end

"""
    check_reaction_counts(RP)

Throw unless [`update_reaction_counts!`](@ref) has run for the step in progress.

The producer must precede its consumers — nothing can know how many events
happened before the equation that determines them is solved. That constraint
cannot be designed away, so it is checked instead of assumed: a stale read used
to halve the ion source silently at `θ = ½`.

Consumers are free to run in **any order among themselves**; they read one array
and get bit-identical answers.
"""
function check_reaction_counts(RP::RAPID)
    missing_channels = filter(k -> k ∉ RP.reactions.published, keys(REACTION_STOICHIOMETRY))
    isempty(missing_channels) || throw(
        ArgumentError(
            "reaction channel(s) $(join(missing_channels, ", ")) have no count for the " *
                "advance in progress. `advance_timestep!` clears them with " *
                "`reset_reaction_counts!`, and `solve_electron_continuity_equation!` " *
                "publishes them again with `update_reaction_counts!` — that solve must " *
                "precede every consumer: the ion source, the particle ledgers and the " *
                "neutral-gas sink"
        )
    )
    return RP.reactions.counts
end

"""
    net_electron_count(counts) -> Matrix

`Σₖ νₖ,ₑ Nₖ` — electrons created per unit volume during this step, `[m⁻³]`.

Aliases the channel array when only one channel contributes, so it allocates
nothing; do not write through the result.
"""
net_electron_count(N::ReactionCounts) = N.iz
# + N.diz − N.rec_H2 − N.rec_H3

"""
    net_H2_gas_count(counts)    -> Matrix
    net_H2_gas_count(counts, k) -> scalar

`Σₖ νₖ,H₂ Nₖ` — molecules created (negative: destroyed) per unit volume during
this step. The neutral-gas sink is this, not a second estimate of it: one
molecule is destroyed for each electron born, so `−net_electron_count` today and
a genuinely different combination once dissociative channels land.

Unlike the other two this combination cannot alias a channel array, so the
whole-field form allocates. The indexed form exists because the sink is an
elementwise sweep over in-wall nodes and has no reason to pay for that.
"""
net_H2_gas_count(N::ReactionCounts) = -N.iz
@inline net_H2_gas_count(N::ReactionCounts, k::Integer) = -N.iz[k]
# − N.diz + N.rec_H3

"""
    net_ion_count(counts, name) -> Matrix or nothing

`Σₖ νₖ,ₛ Nₖ` for the ion species called `name`, or `nothing` when no channel
touches it — a species nothing creates or destroys has no source term rather
than a zero one, so the caller can skip the work entirely.
"""
function net_ion_count(N::ReactionCounts, name::Symbol)
    name === :H2⁺ && return N.iz
    # :H⁺  → N.diz
    # :H3⁺ → −N.rec_H3
    return nothing
end
