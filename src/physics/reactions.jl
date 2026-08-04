export reset_reaction_rates!,
    update_reaction_rates!,
    check_reaction_rates,
    net_electron_rate,
    net_ion_rate,
    net_H2_gas_rate

# Contracting reaction EVENT rates into per-species particle sources.
#
# The struct and the stoichiometry table live in `types.jl`; this is the arithmetic
# `Ṡₛ = Σₖ νₖ,ₛ Rₖ` that turns one into the other, plus the freshness check.
#
# Written from `REACTION_STOICHIOMETRY` and kept honest against it by
# `reaction_sources_test.jl`, which walks the table and re-derives every accessor.
# With one channel these are one-liners; with four they are four-term sums, which
# is still both readable and type-stable, so no contraction engine is warranted.

"""
    reset_reaction_rates!(RP) -> RP

Void the previous advance's reaction rates. **Call at the start of a step.**

This is the lifecycle, and it is what makes producer-before-consumer a structural
property rather than an ordering convention:

```
    reset_reaction_rates!    →   update_reaction_rates!   →   net_*_rate
    (advance_timestep!)          (electron continuity)        (ion source,
     invalidates                  publishes                     ledgers, gas sink)
```

Deliberately **not** keyed on `RP.step`. A step number looks like a validity
token and is not one: `run_simulation!` increments `RP.step` between
`advance_timestep!` and the wall passes, so those consumers — which belong to the
step just advanced — would compare against the next step's number and fail. An
explicit reset has no such coincidence to get right.

Rates are zeroed as well as marked invalid, so nothing that bypasses
[`check_reaction_rates`](@ref) (a snapshot, a debugger) can read last step's
numbers and believe them.
"""
function reset_reaction_rates!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    rx = RP.reactions
    rx.valid = false
    fill!(rx.rates.iz, zero(FT))
    return RP
end

"""
    update_reaction_rates!(RP)

Write this step's reaction event rates. **The single producer.**

Called at the end of [`solve_electron_continuity_equation!`](@ref), which is the
only place that knows `n* = (1−θ)nⁿ + θnⁿ⁺¹` — the density the electron equation
actually ionized at — because it holds `θ`, `prev_n` and the just-solved `ne` in
one scope. Anywhere else it would have to be reconstructed from state that may
have moved, which is the failure this replaces.

`R_iz = ν_en_iz · n*` with `ν_en_iz` as `update_RRCs!` materialized it at the
step-entry state; the tables are not re-queried here.

With `flags.src` off the rates are zeroed rather than left stale, so a run that
switches the source off stops creating particles on the same step.
"""
function update_reaction_rates!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    rx, pla = RP.reactions, RP.plasma
    rx.valid = true
    S = rx.rates
    if !RP.flags.src
        fill!(S.iz, zero(FT))
        return RP
    end
    # θ_imp.growth, because ionization is a λ > 0 term — see `ImplicitWeights`.
    # Explicit runs weight the source at nⁿ, which is θ = 0.
    θ = RP.flags.Implicit ? RP.flags.θ_imp.growth : zero(FT)
    @. S.iz = ((one(FT) - θ) * RP.prev_n + θ * pla.ne) * pla.ν_en_iz
    return RP
end

"""
    check_reaction_rates(RP)

Throw unless [`update_reaction_rates!`](@ref) has run for the step in progress.

The producer must precede its consumers — nothing can know how many events
happened before the equation that determines them is solved. That constraint
cannot be designed away, so it is checked instead of assumed: a stale read used
to halve the ion source silently at `θ = ½`.

Consumers are free to run in **any order among themselves**; they read one array
and get bit-identical answers.
"""
function check_reaction_rates(RP::RAPID)
    RP.reactions.valid || throw(
        ArgumentError(
            "no reaction rates for the advance in progress. `advance_timestep!` clears " *
                "them with `reset_reaction_rates!`, and `solve_electron_continuity_equation!` " *
                "publishes them again with `update_reaction_rates!` — that solve must precede " *
                "every consumer of the ionization rate: the ion source, the particle ledgers " *
                "and the neutral-gas sink"
        )
    )
    return RP.reactions.rates
end

"""
    net_electron_rate(rates) -> Matrix

`Σₖ νₖ,ₑ Rₖ` — electrons created per unit volume per second, `[m⁻³ s⁻¹]`.

Aliases the channel array when only one channel contributes, so it allocates
nothing; do not write through the result.
"""
net_electron_rate(S::ReactionRates) = S.iz
# + S.diz − S.rec_H2 − S.rec_H3

"""
    net_H2_gas_rate(rates)    -> Matrix
    net_H2_gas_rate(rates, k) -> scalar

`Σₖ νₖ,H₂ Rₖ` — molecules created (negative: destroyed) per unit volume per
second. The neutral-gas sink is this, not a second estimate of it: one molecule
is destroyed for each electron born, so `−net_electron_rate` today and a
genuinely different combination once dissociative channels land.

Unlike the other two this combination cannot alias a channel array, so the
whole-field form allocates. The indexed form exists because the sink is an
elementwise sweep over in-wall nodes and has no reason to pay for that.
"""
net_H2_gas_rate(S::ReactionRates) = -S.iz
@inline net_H2_gas_rate(S::ReactionRates, k::Integer) = -S.iz[k]
# − S.diz + S.rec_H3

"""
    net_ion_rate(rates, name) -> Matrix or nothing

`Σₖ νₖ,ₛ Rₖ` for the ion species called `name`, or `nothing` when no channel
touches it — a species nothing creates or destroys has no source term rather
than a zero one, so the caller can skip the work entirely.
"""
function net_ion_rate(S::ReactionRates, name::Symbol)
    name === :H2⁺ && return S.iz
    # :H⁺  → S.diz
    # :H3⁺ → −S.rec_H3
    return nothing
end
