# Contracting reaction EVENT counts into per-species particle changes.
#
# Nothing here is exported: this is the wiring of one step, driven by
# `advance_timestep!` and the species equations. Reach for it by name.
#
# The struct and the stoichiometry table live in `types.jl`; this is the arithmetic
# `ΔNₛ = Σₖ νₖ,ₛ Nₖ` that turns one into the other, plus the freshness check.
#
# The accessors below are HAND-WRITTEN from `REACTION_STOICHIOMETRY`; nothing
# derives them from it and nothing checks that they still agree. Only `.θ` and
# `keys()` are actually read at runtime (`reaction_θ`, `check_reaction_counts`),
# so the particle columns are documentation until a second channel makes the sums
# worth generating. Adding a channel means editing the table AND every accessor —
# that is the cost being deferred, and it is the moment to reconsider.

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
    fill!(rx.counts.diz, zero(FT))
    return RP
end

"""
    reaction_θ(flags, channel) -> FT

The θ channel `channel`'s quadrature uses, from `REACTION_STOICHIOMETRY`'s `θ`
family and [`ImplicitWeights`](@ref). Zero for an explicit run.

θ is a statement about **accuracy, not about time**. What is stored is the definite
integral `Nₖ = ∫ … dt ≈ Δt[(1−θ)(…)ⁿ + θ(…)ⁿ⁺¹]`, so the event count is unambiguous
whatever θ is; θ only says how good the quadrature was. It is exposed because a
consumer may still care — pairing `E(Tₑⁿ⁺¹)` with a channel evaluated at ½ is
inconsistent at `O(Δt)` even where the particle count is not.

Answers from the flags alone, which is enough while every family runs a θ-scheme.
Under `scheme.<family> == ExpRB` the weight differs per cell and per step; use
[`reaction_θ(RP, channel)`](@ref), which covers both and reduces to this one.
"""
function reaction_θ(flags::SimulationFlags{FT}, channel::Symbol) where {FT <: AbstractFloat}
    haskey(REACTION_STOICHIOMETRY, channel) ||
        throw(ArgumentError("no reaction channel called $channel"))
    flags.Implicit || return zero(FT)
    return getproperty(flags.θ_imp, REACTION_STOICHIOMETRY[channel].θ)
end

"""
    reaction_θ(RP, channel) -> FT | Matrix{FT}

The same weight, able to answer when it is no longer a constant.

Under `scheme.<family> == ExpRB` the quadrature is [`exprb_theta`](@ref) at the
family's own `z = λΔt` — per cell, per step. Still in `(0, 1)`, so
`Nₖ = Δt[(1−θ)(…)ⁿ + θ(…)ⁿ⁺¹]` reads unchanged downstream.

**`ExpRB` is checked before `Implicit`, and the order is load-bearing.** For a
θ-scheme the two coincide — no matrix, no weight. `ExpRB` separates them: its
explicit branch applies the same `bern(z)`, and a ledger formed at `θ = 0` there
under-reports every event (3.6 % at `z = 0.15`, worse as `z` grows).

`z` comes from `plasma.exprb.z_growth` rather than being re-derived, so a capped step
is weighted at the `z` that ran, and the ExpRB branch is taken on
`plasma.exprb.growth_fitted` rather than on `scheme.growth` — both are written by the
solve, so a flag flipped between the solve and this call cannot answer for a step that
never happened. Before the first solve `growth_fitted` is false and the answer is the
θ constant, with no events to weight. Throws rather than guessing if a family is switched to `ExpRB`
without its rate wired here.
"""
function reaction_θ(RP::RAPID{FT}, channel::Symbol) where {FT <: AbstractFloat}
    haskey(REACTION_STOICHIOMETRY, channel) ||
        throw(ArgumentError("no reaction channel called $channel"))

    family = REACTION_STOICHIOMETRY[channel].θ

    # A family set to ExpRB with no rate wired here is a CONFIGURATION error, so it
    # reads the live flag: it is wrong the moment it is written, not one solve later.
    if getproperty(RP.flags.scheme, family) === ExpRB && family !== :growth
        throw(
            ArgumentError(
                "scheme.$family = ExpRB, but the rate behind channel :$channel is not " *
                    "wired into reaction_θ. Wire it, or the event count would be formed " *
                    "at a weight the solve did not use."
            )
        )
    end

    # The weight is a property of the SOLVE, not of the flag as it stands now — the
    # ledger it has to match was formed at `z_growth`, by whatever scheme wrote it.
    if family === :growth && RP.plasma.exprb.growth_fitted
        return exprb_theta.(RP.plasma.exprb.z_growth)
    end
    return RP.flags.Implicit ? reaction_θ(RP.flags, channel) : zero(FT)
end

"""
    update_reaction_counts!(RP)

Write this step's reaction event counts. **The single producer.**

Called at the end of [`solve_electron_continuity_equation!`](@ref), which is the
only place that knows `n* = (1−θ)nⁿ + θnⁿ⁺¹` — the density the electron equation
actually ionized at — because it holds `θ`, `prev_n` and the just-solved `ne` in
one scope. Anywhere else it would have to be reconstructed from state that may
have moved, which is the failure this replaces.

`N_iz = Δt·ν_en_iz·n*` and `N_diz = Δt·ν_en_diss_iz·n*`, with the rates as
`update_RRCs!` materialized them at the step-entry state; the tables are not
re-queried here. Δt is baked in on purpose — a count is what it is, and a
consumer cannot accidentally scale it by a step length other than the one it
was formed with.

Under `scheme.growth === ExpRB` the two channels share one capped exponent
(`exprb.z_growth`, built from `ν_en_iz_tot`) and are split by their instantaneous
rate share rather than each recomputed from `Δt·ν` — see the branch below.

With `flags.src` off the counts are zeroed rather than left stale, so a run that
switches the source off stops creating particles on the same step.
"""
function update_reaction_counts!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    rx, pla = RP.reactions, RP.plasma
    N = rx.counts

    # ── e + H₂ → 2e + H₂⁺ / e + H₂ → 2e + H⁺ + H⁰ ──────────────────────────
    if RP.flags.src
        # The θ FAMILY comes from the table, not from this line: a `:decay` channel
        # added later then picks up backward Euler by existing. The CHANNEL asked
        # for here is `:iz`, not `:diz`, but that is not a second lookup skipped —
        # `:diz` is in the same `:growth` family as `:iz` (REACTION_STOICHIOMETRY),
        # so `reaction_θ(RP, :iz)` and `reaction_θ(RP, :diz)` would answer identically;
        # asking once and reusing it below is just not paying for that twice.
        # From RP, not from flags alone: under ExpRB the weight is the fitted θ(z)
        # of the step just taken, per cell. Broadcasting below covers both forms.
        θ = reaction_θ(RP, :iz)
        if RP.flags.scheme.growth === ExpRB
            # `z_growth` is capped and built from `ν_en_iz_tot` (Task C1), so it books
            # the electrons the solve ACTUALLY created, both channels together. Split
            # it by channel share rather than recomputing from Δt·ν: recomputing would
            # undo the cap and book (z/z_cap)× the true number — 4/3 at z = 40.
            frac_iz = @. ifelse(pla.ν_en_iz_tot > zero(FT), pla.ν_en_iz / pla.ν_en_iz_tot, one(FT))
            born = @. pla.exprb.z_growth * ((one(FT) - θ) * RP.prev_n + θ * pla.ne)
            @. N.iz = born * frac_iz
            @. N.diz = born * (one(FT) - frac_iz)
        else
            dt = RP.dt
            n_star = @. (one(FT) - θ) * RP.prev_n + θ * pla.ne
            @. N.iz = dt * n_star * pla.ν_en_iz
            @. N.diz = dt * n_star * pla.ν_en_diss_iz
        end
    else
        fill!(N.iz, zero(FT))
        fill!(N.diz, zero(FT))
    end
    push!(rx.published, :iz)
    push!(rx.published, :diz)

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

Both channels make exactly one electron per event (`REACTION_STOICHIOMETRY`), so
this is `N.iz .+ N.diz`. Allocates: with two channels contributing there is no
single array left to alias.
"""
net_electron_count(N::ReactionCounts) = N.iz .+ N.diz
# − N.rec_H2 − N.rec_H3

"""
    net_H2_gas_count(counts)    -> Matrix
    net_H2_gas_count(counts, k) -> scalar

`Σₖ νₖ,H₂ Nₖ` — molecules created (negative: destroyed) per unit volume during
this step. The neutral-gas sink is this, not a second estimate of it: today
`ReactionCounts` only tracks `iz` and `diz`, and both destroy one H₂ per
electron born, so the sum reduces to `−net_electron_count` — a consequence of
which channels are booked, not a conservation identity.

**It undercounts.** Dissociative excitation (`e + H₂ → e + H + H*`) destroys an
H₂ and makes no electron, so it is invisible to `−net_electron_count` by
construction. Its particle rate `K_diss_exc` is already loaded by
`update_RRCs!` (`reaction_rate_coefficients.jl:319`) — only its energy sibling
`Kerg_diss_exc` is consumed (into `P_en_diss_exc`); the channel has no
`ReactionCounts` field and never reaches this sum. Measured on the shipped
table, `K_diss_exc/(K_iz + K_diss_iz)` is 182× at Ē = 3 eV, 2.06× at 10 eV,
0.79× at 20 eV, 0.40× at 50 eV — this sink under-consumes H₂ by roughly 1.4–3×
through burn-through and by two orders of magnitude in the few-eV band.
Booking it would need a `diss_exc` row in `REACTION_STOICHIOMETRY` (H₂ → −1,
no electron) and a matching `ReactionCounts` field; deliberately out of scope
here.

The indexed form exists because the sink is an elementwise sweep over in-wall
nodes and has no reason to pay for allocating the whole field.
"""
net_H2_gas_count(N::ReactionCounts) = -(N.iz .+ N.diz)
@inline net_H2_gas_count(N::ReactionCounts, k::Integer) = -(N.iz[k] + N.diz[k])
# + N.rec_H3

"""
    net_ion_count(counts, name) -> Matrix or nothing

`Σₖ νₖ,ₛ Nₖ` for the ion species called `name`, or `nothing` when no channel
touches it — a species nothing creates or destroys has no source term rather
than a zero one, so the caller can skip the work entirely.

`:H2⁺` is `N.iz .+ N.diz`, not `N.iz` alone: under the INTERIM
(`REACTION_STOICHIOMETRY.diz`) DI's ion is booked here too, because H⁺ is not yet
a transportable species. That makes this identically `net_electron_count` for
now — not a conservation check, just the consequence of the interim.
"""
function net_ion_count(N::ReactionCounts, name::Symbol)
    name === :H2⁺ && return N.iz .+ N.diz
    # :H⁺  → N.diz alone, once REACTION_STOICHIOMETRY.diz.ions changes to :H⁺ => 1
    #        and H⁺ is a declared, transportable species — see the docstring above.
    # :H3⁺ → −N.rec_H3
    return nothing
end
