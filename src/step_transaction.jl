# A time step as a transaction: the state a step writes is copied out before it, an attempt the
# verdict rejects is put back, and the step is taken again in halves.

# ── verdicts ────────────────────────────────────────────────────────────────────────────

"""
    StepVerdict

What becomes of an attempt at a step: [`Accept`](@ref) or [`Reject`](@ref). Judged after the
solve, on the state it left, before the negative-density floor and the ion-source ledger.
"""
abstract type StepVerdict end

"The attempt stands."
struct Accept <: StepVerdict end

"The attempt is put back and the step taken in halves; `reason` is counted in `diagnostics.step_retries`."
struct Reject <: StepVerdict
    reason::Symbol
end

"""
    step_verdict(RP) -> StepVerdict

The verdict `run_simulation!` gives every attempt at a step: [`Accept`](@ref).
"""
step_verdict(::RAPID) = Accept()

# ── the state a step writes ─────────────────────────────────────────────────────────────

# The records a step writes, copied whole, and the fields of the other owners it writes (the
# coils' for each coil). The owners' other fields are constant over a run, rebuilt where they
# are used, or refreshed from the state after an accepted step.
const STEP_STATE_RECORDS = (:plasma, :fields, :vertical_field_control)
const STEP_STATE_FIELDS = (
    transport = (),        # its ion columns are gathered from the species store (plasma) at each solve
    reactions = (:counts, :published),
    diagnostics = (:Ntracker, :ampere_picard, :near_vacuum, :ambipolar_exchange),
    closed_surface = (:solve, :ledger),
    # the circuit matrices with the step and θ they were built for: the coils' last step, over
    # which their induced field is read
    coil_system = (:time_s, :ΔI_step, :Δt, :θimp, :A_LR_circuit, :inv_A_LR_circuit),
    coil = (:current, :ψ_pla, :voltage_applied),
    rapid = (:prev_n, :step, :time_s),
)

# A checkpoint's copies of the coil system's step state and of a coil's
mutable struct SavedCoil{FT <: AbstractFloat}
    current::FT
    ψ_pla::FT
    voltage_applied::Union{FT, Function}
end

mutable struct SavedCoilSystem{FT <: AbstractFloat}
    time_s::FT
    ΔI_step::Vector{FT}
    Δt::FT
    θimp::FT
    A_LR_circuit::Matrix{FT}
    inv_A_LR_circuit::Matrix{FT}
    coils::Vector{SavedCoil{FT}}
end

"""
    StepState(RP)

A copy of what a step writes in `RP` (`STEP_STATE_RECORDS` whole, `STEP_STATE_FIELDS` by
owner), taken by [`save_step_state!`](@ref) and put back by [`restore_step_state!`](@ref). Both
go through each owner's fields by name, so an array the step rebinds is read and written where
it now is; the checkpoint holds no reference into `RP`.
"""
mutable struct StepState{FT <: AbstractFloat}
    plasma::PlasmaState{FT}
    fields::Fields{FT}
    vertical_field_control::VerticalFieldControlState{FT}
    reactions::ReactionState{FT}
    diagnostics::Diagnostics{FT}
    closed_surface::ClosedSurfaceState{FT}
    coil_system::SavedCoilSystem{FT}
    prev_n::Matrix{FT}
    step::Int
    time_s::FT
end

function StepState(RP::RAPID{FT}) where {FT <: AbstractFloat}
    NR, NZ = RP.G.NR, RP.G.NZ
    none = Matrix{FT}(undef, 0, 0)
    ck = StepState{FT}(
        deepcopy(RP.plasma), deepcopy(RP.fields), deepcopy(RP.vertical_field_control),
        ReactionState{FT}(dims = (NR, NZ)),
        Diagnostics{FT}(NR, NZ), ClosedSurfaceState{FT}(NR, NZ),
        SavedCoilSystem{FT}(zero(FT), FT[], zero(FT), zero(FT), copy(none), copy(none), SavedCoil{FT}[]),
        zeros(FT, NR, NZ), 0, zero(FT),
    )
    return save_step_state!(ck, RP)
end

"""
    save_step_state!(ck, RP) -> ck

Copy what a step writes from `RP` into the checkpoint `ck`.
"""
function save_step_state!(ck::StepState{FT}, RP::RAPID{FT}) where {FT <: AbstractFloat}
    coils = ck.coil_system.coils
    n = length(RP.coil_system.coils)
    while length(coils) < n
        push!(coils, SavedCoil{FT}(zero(FT), zero(FT), zero(FT)))
    end
    resize!(coils, n)
    check_store_layout(ck.plasma, RP.plasma)   # before anything is copied
    return copy_step_state!(ck, RP)
end

"""
    restore_step_state!(RP, ck) -> RP

Put the checkpoint `ck` back into `RP`: what a step writes is again, bit for bit, what
[`save_step_state!`](@ref) copied.
"""
function restore_step_state!(RP::RAPID{FT}, ck::StepState{FT}) where {FT <: AbstractFloat}
    length(RP.coil_system.coils) == length(ck.coil_system.coils) ||
        error("the coil set changed during the step; its state cannot be put back")
    check_store_layout(RP.plasma, ck.plasma)   # before anything is copied
    return copy_step_state!(RP, ck)
end

# `dst` takes the step state of `src`: one is the RAPID object, the other a checkpoint
function copy_step_state!(dst, src)
    for name in STEP_STATE_RECORDS
        copy_state!(getfield(dst, name), getfield(src, name))
    end
    for owner in (:reactions, :diagnostics, :closed_surface, :coil_system)
        copy_fields!(getfield(dst, owner), getfield(src, owner), STEP_STATE_FIELDS[owner])
    end
    for (d, s) in zip(getfield(dst, :coil_system).coils, getfield(src, :coil_system).coils)
        copy_fields!(d, s, STEP_STATE_FIELDS.coil)
    end
    copy_fields!(dst, src, STEP_STATE_FIELDS.rapid)
    return dst
end

# `dst` takes the value of `src`: every field (`copy_state!`), or the named ones
copy_state!(dst::T, src::T) where {T} = copy_fields!(dst, src, fieldnames(T))

function copy_fields!(dst, src, names)
    for name in names
        copy_field!(dst, name, getfield(dst, name), getfield(src, name))
    end
    return dst
end

# One field of `dst` takes the value `s`, in place where its present value `d` allows: an array
# of numbers by its contents (a vector resized to `s`), an array of such arrays of the same
# shapes array by array (so that an array shared with another field stays shared: the species
# store), a set by its elements, a mutable record of this package field by field. Anything else,
# an array of another shape included, is assigned, a mutable value as a copy.
function copy_field!(dst, name::Symbol, d, s)
    if d isa Array && typeof(d) === typeof(s) && isbitstype(eltype(s)) && (ndims(s) == 1 || size(d) == size(s))
        ndims(s) == 1 && resize!(d, length(s))
        copyto!(d, s)
    elseif d isa Array{<:Array} && typeof(d) === typeof(s) && isbitstype(eltype(eltype(s))) &&
            size(d) == size(s) && all(k -> size(d[k]) == size(s[k]), eachindex(s))
        foreach(copyto!, d, s)
    elseif d isa AbstractSet && typeof(d) === typeof(s)
        union!(empty!(d), s)
    elseif typeof(d) === typeof(s) && ismutabletype(typeof(s)) && parentmodule(typeof(s)) === @__MODULE__
        copy_state!(d, s)
    else
        setfield!(dst, name, ismutable(s) ? deepcopy(s) : s)
    end
    return nothing
end

# ── the step ────────────────────────────────────────────────────────────────────────────

# A rejected half is halved again down to Δt/2^STEP_MAX_HALVINGS; positions in a step are counted
# in units of that shortest part
const STEP_MAX_HALVINGS = 6
const STEP_UNITS = 1 << STEP_MAX_HALVINGS

"""
    StepTransaction(RP)

The checkpoints of [`take_step!`](@ref): the state at the start of the step, and at the start of
the part being taken once a step is taken in parts (made on the first such step); `refreshed`
when an accepted part of the step in progress has refreshed the caches from its state.
"""
mutable struct StepTransaction{FT <: AbstractFloat}
    entry::StepState{FT}
    part::Union{Nothing, StepState{FT}}
    refreshed::Bool
end

StepTransaction(RP::RAPID{FT}) where {FT <: AbstractFloat} = StepTransaction{FT}(StepState(RP), nothing, false)

"""
    take_step!(RP, tx = StepTransaction(RP); verdict = step_verdict, callback_before_step = nothing) -> RP

The step tⁿ → tⁿ + Δt, Δt = `RP.dt`, as a transaction. An attempt runs [`prepare_timestep!`](@ref),
`callback_before_step`, [`solve_timestep!`](@ref) and `verdict(RP)`; accepted, it sets the clock
and books what belongs to the step (the ion-source ledger, the negative-density floor). A rejected
attempt is put back ([`restore_step_state!`](@ref)) and the step taken in two halves, each a
transaction of its own; a rejected half is halved again, down to Δt/2⁶. A rejection there puts the
state back as the step found it and throws. `RP.dt` is Δt again on every exit, and an accepted
step lands `RP.time_s` on tⁿ + Δt exactly. An exception inside an attempt is not rolled back:
the state stays where the attempt failed, to be inspected.

Between two accepted parts, the electrostatic estimate and `update_transport_quantities!` run as
after a step; the field-line analysis waits for the end of the step. After a step taken in parts,
what describes the last step (the reaction counts, `ΔI_step` with the coils' `Δt`, the induced
field) is its last part's; the ledgers hold every accepted part.
"""
function take_step!(
        RP::RAPID{FT}, tx::StepTransaction{FT} = StepTransaction(RP);
        verdict = step_verdict, callback_before_step = nothing,
    ) where {FT <: AbstractFloat}
    dt, t0 = RP.dt, RP.time_s
    save_step_state!(tx.entry, RP)
    tx.refreshed = false
    try
        v = attempt_step!(RP, t0 + dt, true, verdict, callback_before_step)
        v isa Accept && return RP
        restore_step_state!(RP, tx.entry)
        count_rejection!(RP, v)
        RP.diagnostics.step_retries.split_steps += 1
        isnothing(tx.part) && (tx.part = StepState(RP))
        take_part!(RP, tx, t0, dt, 0, 1, verdict, callback_before_step)
        take_part!(RP, tx, t0, dt, STEP_UNITS ÷ 2, 1, verdict, callback_before_step)
    finally
        RP.dt = dt
    end
    return RP
end

# The part of the step from t0 + q Δt/2⁶, Δt/2^level long, as a transaction: rejected, it is
# taken as two halves.
function take_part!(RP::RAPID{FT}, tx::StepTransaction{FT}, t0::FT, dt::FT, q::Int, level::Int, verdict, callback_before_step) where {FT <: AbstractFloat}
    len = STEP_UNITS >> level
    completes = q + len == STEP_UNITS
    t_end = completes ? t0 + dt : t0 + (q + len) * (dt / STEP_UNITS)
    RP.dt = dt / (1 << level)
    save_step_state!(tx.part, RP)
    v = attempt_step!(RP, t_end, completes, verdict, callback_before_step)
    if v isa Accept
        if !completes
            refresh_between_parts!(RP)
            tx.refreshed = true
        end
        return RP
    end
    restore_step_state!(RP, tx.part)
    count_rejection!(RP, v)
    if level == STEP_MAX_HALVINGS
        h = RP.dt
        restore_step_state!(RP, tx.entry)
        RP.dt = dt
        if tx.refreshed
            # the caches an accepted part refreshed, rebuilt from the state put back: the charge
            # sheets' turbulent diffusivity from the field and weight the step found (not estimated
            # again: a run's first step has no field), then the transport's; what that writes into
            # the plasma, put back as the step found it
            update_turbulent_diffusivity!(RP)
            update_transport_quantities!(RP)
            restore_step_state!(RP, tx.entry)
        end
        error(
            "the step from t = $t0 s was rejected ($(v.reason)) down to Δt/2^$level = $h s; " *
                "the state is back at the start of the step"
        )
    end
    take_part!(RP, tx, t0, dt, q, level + 1, verdict, callback_before_step)
    take_part!(RP, tx, t0, dt, q + len ÷ 2, level + 1, verdict, callback_before_step)
    return RP
end

# One attempt from the present state to `t_end`, `RP.dt` long, `completes` when it ends the step.
# Accepted, it sets the clock and books what belongs to the step.
function attempt_step!(RP::RAPID{FT}, t_end::FT, completes::Bool, verdict, callback_before_step) where {FT <: AbstractFloat}
    prepare_timestep!(RP)
    isnothing(callback_before_step) || callback_before_step(RP)
    solve_timestep!(RP)
    v = verdict(RP)::StepVerdict
    v isa Accept || return v
    RP.time_s = t_end
    completes && (RP.step += 1)
    # Ledgers and floors after the step. Nothing zeroes the band outside the wall: no operator
    # writes there.
    book_ionization_sources!(RP)
    correct_negative_densities!(RP)
    return v
end

# Between two accepted parts of a step, what the end of a step refreshes from the state; the
# field-line analysis keeps to the steps.
function refresh_between_parts!(RP::RAPID)
    @timeit RAPID_TIMER "electrostatic_field" estimate_electrostatic_field_effects!(RP)
    @timeit RAPID_TIMER "transport_quantities" update_transport_quantities!(RP)
    return RP
end

function count_rejection!(RP::RAPID, v::Reject)
    rejected = RP.diagnostics.step_retries.rejected
    rejected[v.reason] = get(rejected, v.reason, 0) + 1
    return RP
end
