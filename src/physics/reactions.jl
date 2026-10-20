# Contracting reaction EVENT counts into per-species particle changes.
#
# Nothing here is exported: this is the wiring of one step, driven by
# `advance_timestep!` and the species equations. Reach for it by name.
#
# The struct and the stoichiometry table live in `types.jl`; this is the arithmetic
# `ΔNₛ = Σₖ νₖ,ₛ Nₖ` that turns one into the other, plus the freshness check.
#
# The contractions are generated from `REACTION_STOICHIOMETRY`'s columns (bottom of this
# file). The producer, `update_reaction_counts!`, still names its two channels: it is the
# one place that knows how each count is integrated.

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
    for ch in keys(REACTION_STOICHIOMETRY)
        fill!(getproperty(rx.counts, ch)::Matrix{FT}, zero(FT))
    end
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
            # `z_growth` is capped and built from `ν_en_iz_tot`, so it books
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
    union!(rx.published, keys(REACTION_STOICHIOMETRY))

    return RP
end

"""
    check_reaction_counts(RP, channels = keys(REACTION_STOICHIOMETRY))

Throw unless every channel in `channels` has a count for the step in progress
([`update_reaction_counts!`](@ref) has run). Returns the counts.

The producer must precede its consumers — nothing can know how many events
happened before the equation that determines them is solved. That constraint
cannot be designed away, so it is checked instead of assumed: a stale read used
to halve the ion source silently at `θ = ½`.

Consumers are free to run in **any order among themselves**; they read one array
and get bit-identical answers. A consumer that needs only some channels names them.
"""
function check_reaction_counts(RP::RAPID, channels = keys(REACTION_STOICHIOMETRY))
    missing_channels = filter(k -> k ∉ RP.reactions.published, collect(channels))
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

# ── contractions generated from the table ────────────────────────────────────────────────
#
# A channel contributes to a column with its coefficient. The channels are summed in table
# order and a common coefficient is applied once, `c·(N₁ + N₂)`: the expression the
# hand-written accessors always had, so the rounding and the signed zeros are unchanged.
# A column whose channels carry different coefficients is refused here until a producer
# needs one (the reaction catalog generalizes this).

function _column_channels(column)
    chans = [ch for ch in keys(REACTION_STOICHIOMETRY) if column(REACTION_STOICHIOMETRY[ch]) != 0]
    coefs = unique(column(REACTION_STOICHIOMETRY[ch]) for ch in chans)
    length(coefs) <= 1 || error("a column with mixed coefficients needs the general contraction")
    return chans, isempty(coefs) ? 0 : only(coefs)
end

function _contraction(chans, c, ref)
    terms = [ref(ch) for ch in chans]
    sum_expr = length(terms) == 1 ? terms[1] : foldl((a, b) -> :($a .+ $b), terms)
    length(terms) == 1 && (sum_expr = :(copy($sum_expr)))
    return c == 1 ? sum_expr : c == -1 ? :(-($sum_expr)) : :($c .* $sum_expr)
end

function _contraction_at(chans, c, ref)
    terms = [ref(ch) for ch in chans]
    sum_expr = length(terms) == 1 ? terms[1] : foldl((a, b) -> :($a + $b), terms)
    return c == 1 ? sum_expr : c == -1 ? :(-($sum_expr)) : :($c * $sum_expr)
end

let (chans, c) = _column_channels(row -> row.electron)
    @eval net_electron_count(N::ReactionCounts) = $(_contraction(chans, c, ch -> :(N.$ch)))
end

let (chans, c) = _column_channels(row -> row.H2_gas)
    @eval net_H2_gas_count(N::ReactionCounts) = $(_contraction(chans, c, ch -> :(N.$ch)))
    @eval @inline net_H2_gas_count(N::ReactionCounts, k::Integer) =
        $(_contraction_at(chans, c, ch -> :(N.$ch[k])))
end

"""
    net_electron_count(counts) -> Matrix

`Σₖ νₖ,ₑ Nₖ` — electrons created per unit volume during this step, `[m⁻³]`.
"""
net_electron_count

"""
    net_H2_gas_count(counts)    -> Matrix
    net_H2_gas_count(counts, k) -> scalar

`Σₖ νₖ,H₂ Nₖ` — molecules created (negative: destroyed) per unit volume during this step.
The neutral-gas sink is this, not a second estimate of it.

**It undercounts.** Dissociative excitation (`e + H₂ → e + H + H*`) destroys an
H₂ and makes no electron, and the channel has no row in `REACTION_STOICHIOMETRY`.
Its particle rate `K_diss_exc` is already loaded by `update_RRCs!` — only its energy
sibling `Kerg_diss_exc` is consumed (into `P_en_diss_exc`). Measured on the shipped
table, the median of `K_diss_exc/(K_iz + K_diss_iz)` over E/p is 18 at
Ē = 5 eV, 2.1 at 10, 1.1 at 15, 0.79 at 20, 0.40 at 50 and 0.19 at 300 — so
this sink under-consumes H₂ by 1.2–3× from burn-through up, and by more than an
order of magnitude in the few-eV band, where ionization has barely switched on
and dissociative excitation is nearly the whole H₂ loss. (Quote the median, not
the range: below Ē ≈ 5 eV the denominator is near zero on most E/p rows, so the
ratio there is numerically unbounded and says nothing.)

The indexed form exists because the sink is an elementwise sweep over in-wall
nodes and has no reason to pay for allocating the whole field.
"""
net_H2_gas_count

const _ION_PRODUCERS = let
    producers = Dict{SpeciesKey, Vector{Pair{Symbol, Int}}}()
    for ch in keys(REACTION_STOICHIOMETRY), (key, ν) in REACTION_STOICHIOMETRY[ch].ions
        push!(get!(producers, key, Pair{Symbol, Int}[]), ch => ν)
    end
    producers
end

"""
    net_ion_count(counts, key::SpeciesKey) -> Matrix or nothing

`Σₖ νₖ,ₛ Nₖ` for the ion species with identity `key`, or `nothing` when no channel
touches it — a species nothing creates or destroys has no source term rather
than a zero one, so the caller can skip the work entirely.

`H2[1]` is `N.iz .+ N.diz`, not `N.iz` alone: under the INTERIM(diz-ion-species)
(`REACTION_STOICHIOMETRY.diz`) DI's ion is booked here too, because H⁺ is not yet
a transportable species.
"""
function net_ion_count(N::ReactionCounts{FT}, key::SpeciesKey) where {FT <: AbstractFloat}
    producers = get(_ION_PRODUCERS, key, nothing)
    isnothing(producers) && return nothing
    coefs = unique(last.(producers))
    length(coefs) == 1 || error("an ion with mixed coefficients needs the general contraction")
    acc = getproperty(N, first(producers[1]))::Matrix{FT}
    acc = length(producers) == 1 ? copy(acc) : acc
    for (ch, _) in producers[2:end]
        acc = acc .+ (getproperty(N, ch)::Matrix{FT})
    end
    c = only(coefs)
    return c == 1 ? acc : c .* acc
end
