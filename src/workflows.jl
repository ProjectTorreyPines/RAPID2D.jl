"""
Workflows module for RAPID2D.

Contains high-level simulation workflows, including:
- Time stepping algorithms
- Simulation advancement functions
- Multi-physics coupling strategies
"""

using TimerOutputs
using Printf
using Dates

# Use the global timer from the main module
# (RAPID_TIMER is already defined in the main RAPID2D module)


"""
    is_snap0D_time(time)

Check if the given time corresponds to a 0D snapshot time.

Returns `true` if the time matches a 0D snapshot timing, `false` otherwise.
"""
function is_snap0D_time(RP)
    RP_Δt_s = RP.time_s - RP.t_start_s
    return abs(RP_Δt_s - round(RP_Δt_s / RP.config.snap0D_Δt_s) * RP.config.snap0D_Δt_s) < 0.1 * RP.dt
end

"""
    is_snap2D_time(time)

Check if the given time corresponds to a 2D snapshot time.

Returns `true` if the time matches a 2D snapshot timing, `false` otherwise.
"""
function is_snap2D_time(RP)
    RP_Δt_s = RP.time_s - RP.t_start_s
    return abs(RP_Δt_s - round(RP_Δt_s / RP.config.snap2D_Δt_s) * RP.config.snap2D_Δt_s) < 0.1 * RP.dt
end


"""
    update_Jϕ!(RP)

`Jϕ` of the present state: (qₑ nₑ uₑ∥ + Z e nᵢ uᵢ∥) bϕ, electrons and ions.
"""
function update_Jϕ!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    F = RP.fields
    @unpack qe, ee = RP.config.constants
    # `ni·Z` is the ion CHARGE density. One species, so it is a product and not a sum; `Z`
    # comes from the species itself and cannot lag behind it.
    Z_i = FT(bulk_ion_charge(RP))
    @. pla.Jϕ = (pla.ne * qe * pla.ue_para + pla.ni * (ee * Z_i) * pla.ui_para) * F.bϕ
    return RP
end

"""
    initialize_coupled_fields!(RP)

Make the coils' memory and `ψ_self` consistent with the initial state, both from the same `Jϕ`:
- every coil without a memory (`ψ_pla` unset) takes the plasma flux of `Jϕ`;
- if the run starts with currents, Ampère is on and `ψ_self` is zero, `ψ_self` becomes the
  Grad–Shafranov solution of those currents, with no induced field.

`run_simulation!` calls it before the first step; a loop over `advance_timestep!` calls it
once before the first. A coil added later with a current enters `ψ_self` at the next solve;
call `solve_Ampere_equation!(RP; update_Eϕ_self = false)` after adding it.
"""
function initialize_coupled_fields!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    update_Jϕ!(RP)
    csys = RP.coil_system
    csys.n_total > 0 && init_unset_coil_plasma_flux!(csys, RP.G, RP.plasma.Jϕ)
    has_current = any(!iszero, RP.plasma.Jϕ) || (csys.n_total > 0 && any(!iszero, csys.coils.current))
    if RP.flags.Ampere && has_current && all(iszero, RP.fields.ψ_self)
        solve_Ampere_equation!(RP; update_Eϕ_self = false)
        fill!(RP.fields.Eϕ_self, zero(FT))
        fill!(RP.fields.Eϕ_self_prev, zero(FT))
        combine_external_and_self_fields!(RP)
        update_Jϕ!(RP)   # bϕ has moved with the self field
    end
    return RP
end

"""
    prepare_timestep!(RP)

Set the inputs of the next step from the current state and time tⁿ, leaving the state itself
unchanged: reset the per-step reaction counts, evaluate the external fields at `RP.time_s`,
and compute `Jϕ` from the densities and velocities.
"""
function prepare_timestep!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "prepare_timestep!" begin
        # Last step's reaction counts are void from here on. Reset at the START of
        # the advance rather than the end, because the wall passes that book the
        # ionization run outside `advance_timestep!` and must still see it.
        reset_reaction_counts!(RP)

        # Update vacuum fields from external sources
        @timeit RAPID_TIMER "external_fields" begin
            update_external_fields!(RP)
        end

        # Current calculations
        @timeit RAPID_TIMER "current_calculation" update_Jϕ!(RP)
    end
    return RP
end

# What a step below the Ampère gate writes before it knows whether its current crosses the gate,
# put back when the step is solved again by the coupled solve: u∥, Jϕ, the induced field and
# its parallel projection, and the coils' currents, memory and clock. The rest it writes is
# assigned afresh by that solve before use (Rue_ei), or is a cache rebuilt from the same state
# (the u∥ operators and LU, the circuit matrices, the Grad–Shafranov factorization).
function below_gate_entry_state(RP::RAPID)
    F, pla, csys = RP.fields, RP.plasma, RP.coil_system
    return (;
        ue_para = copy(pla.ue_para), Jϕ = copy(pla.Jϕ),
        Eϕ_self = copy(F.Eϕ_self), Eϕ_self_prev = copy(F.Eϕ_self_prev),
        E_para_self_EM = copy(F.E_para_self_EM), E_para_tot = copy(F.E_para_tot),
        coil_current = csys.n_total > 0 ? copy(get_all_currents(csys)) : nothing,
        coil_ψ_pla = csys.n_total > 0 ? copy(csys.coils.ψ_pla) : nothing, coil_time = csys.time_s,
    )
end

function restore_below_gate_entry_state!(RP::RAPID, entry)
    F, pla, csys = RP.fields, RP.plasma, RP.coil_system
    pla.ue_para .= entry.ue_para
    pla.Jϕ .= entry.Jϕ
    F.Eϕ_self .= entry.Eϕ_self
    F.Eϕ_self_prev .= entry.Eϕ_self_prev
    F.E_para_self_EM .= entry.E_para_self_EM
    F.E_para_tot .= entry.E_para_tot
    if csys.n_total > 0
        set_all_currents!(csys, entry.coil_current)
        csys.coils.ψ_pla = entry.coil_ψ_pla
    end
    csys.time_s = entry.coil_time
    return RP
end

"""
    solve_timestep!(RP, dt = RP.dt)

Advance the state from tⁿ to tⁿ⁺¹ on the inputs `prepare_timestep!` set: the momentum
equation and the coil circuits (coupled with Ampère above the current threshold; below it, and
with Ampère off, the coils advance as vacuum circuits and Ampère only keeps `ψ_self`), the
densities, the ion velocity and the temperatures, the global J×B force and the neutral gas.
A step below the threshold whose u∥ update carries the current across it is solved again by
the coupled solve, from the state it started at, when that solve can run (Ampère, the
inductive E∥ and u∥ evolution on).
"""
function solve_timestep!(RP::RAPID{FT}, dt::FT = RP.dt) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "solve_timestep!" begin
        I_tor = sum(RP.plasma.Jϕ * RP.G.dR * RP.G.dZ)  # Total toroidal current

        above_gate = RP.flags.Ampere && abs(I_tor) >= RP.flags.Ampere_Itor_threshold
        if above_gate && RP.flags.E_para_self_EM && RP.flags.ud_evolve
            # u∥, ψ_self and the coil currents together
            solve_combined_momentum_Ampere_equations_with_coils!(RP; NamedTuple(RP.flags.ampere_picard)...)
        elseif above_gate
            # The coils advance on their own circuits, the plasma entering through the flux
            # each coil remembers
            advance_coils!(RP)
            if RP.flags.ud_evolve
                update_ue_para!(RP)
            end
            update_Jϕ!(RP)
            @timeit RAPID_TIMER "solve_Ampere_equation!" solve_Ampere_equation!(RP)
        else
            # Below the gate the plasma current is not a source of induction: the coils
            # advance as in vacuum and the induced field is theirs alone (see
            # set_Eϕ_self_from_coils!). ψ_self stays the field of the present currents, so the
            # coupled solve starts from it when the gate opens. A step whose u∥ update carries
            # the current across the gate would have accelerated it without its
            # self-inductance; it is solved again by the coupled solve, from where it started.
            coupled = RP.flags.Ampere && RP.flags.E_para_self_EM && RP.flags.ud_evolve
            entry = coupled ? below_gate_entry_state(RP) : nothing
            ΔI_coils = advance_coils!(RP; plasma = false)
            RP.flags.Ampere && RP.flags.E_para_self_EM && set_Eϕ_self_from_coils!(RP, ΔI_coils)
            if RP.flags.ud_evolve
                update_ue_para!(RP)
            end
            if RP.flags.Ampere
                update_Jϕ!(RP)
                if coupled && abs(sum(RP.plasma.Jϕ) * RP.G.dR * RP.G.dZ) >= RP.flags.Ampere_Itor_threshold
                    restore_below_gate_entry_state!(RP, entry)
                    solve_combined_momentum_Ampere_equations_with_coils!(RP; NamedTuple(RP.flags.ampere_picard)...)
                else
                    @timeit RAPID_TIMER "solve_Ampere_equation!" solve_Ampere_equation!(RP; update_Eϕ_self = false)
                end
            end
        end

        combine_external_and_self_fields!(RP)

        # Update electron density
        solve_electron_continuity_equation!(RP)

        # Ion dynamics
        if RP.flags.update_ni_independently
            # Solve ion continuity equation
            solve_ion_continuity_equation!(RP)
        else
            # Set ion density from electron density with charge neutrality
            slave_ions_to_electrons!(RP)
        end

        if RP.flags.ud_evolve
            update_ui_para!(RP)
        end

        if RP.flags.Ti_evolve
            update_Ti!(RP)
        end

        if RP.flags.Global_JxB_Force
            update_uMHD_by_global_JxB_force!(RP)
        end

        # Electron temperature evolution if enabled
        if RP.flags.Te_evolve
            update_Te!(RP)
        end

        # Neutral gas density evolution if enabled
        if RP.flags.Gas_evolve
            update_neutral_H2_gas_density!(RP)
        end

    end
    return RP
end

"""
    advance_timestep!(RP, dt = RP.dt)

One whole time step: [`prepare_timestep!`](@ref), then [`solve_timestep!`](@ref).
`run_simulation!` calls the two itself, with `callback_before_step` in between.
"""
function advance_timestep!(RP::RAPID{FT}, dt::FT = RP.dt) where {FT <: AbstractFloat}
    prepare_timestep!(RP)
    solve_timestep!(RP, dt)
    return RP
end

"""
    run_simulation!(RP::RAPID{FT}) where FT<:AbstractFloat

Run a full simulation from current time to the end time specified in the RAPID object.
Handles time stepping, diagnostics output, and snapshot generation.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing all simulation state
- `controller`: optional `Controller` updated at the end of every step
- `callback_before_step`: optional `f(RP)`, called on every step between
  `prepare_timestep!` and `solve_timestep!`: the state is at tⁿ and the step's inputs (the
  reaction counts, the external fields at tⁿ, `Jϕ`) have just been set from it. Nothing
  resets them before the step solves, so what the callback writes to them is what the step
  uses. The step recomputes `Jϕ` once it has updated u∥.
- `callback_after_step`: optional `f(RP)`, called at the end of every completed step, at
  tⁿ⁺¹, after that step's snapshots and the controller update.

The two bracket one step: diagnose the change a step makes, adjust a step's inputs, or
record what the snapshots do not carry, without re-implementing this loop. A callback that
changes the state itself (densities, velocities, temperatures) should refresh what depends
on it (`update_transport_quantities!`); in `callback_before_step`, this step's `Jϕ` is
already set.

# Returns
- `RP`: The updated RAPID object after completion of the simulation
"""
function run_simulation!(
        RP::RAPID{FT};
        controller::Union{Nothing, Controller{FT}} = nothing,
        callback_before_step = nothing,
        callback_after_step = nothing,
    ) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "run_simulation!" begin
        # Simulation parameters
        dt = RP.dt
        t_end = RP.t_end_s

        if RP.step == 0
            # The band outside the wall is not part of the problem: no operator has rows
            # there and nothing books it, so an initial condition written over the whole
            # grid is cleared there once, here, and never touched again.
            RP.plasma.ne[RP.G.nodes.on_out_wall_nids] .= zero(FT)
            RP.plasma.ni[RP.G.nodes.on_out_wall_nids] .= zero(FT)
            RP.flags.secondary_electron && RP.flags.update_ni_independently &&
                @warn "secondary_electron is inert until secondaries are emitted through the wall faces from the ion ledger" maxlog = 1

            # The coils' memory and the self field start from the initial currents. A resumed
            # run keeps both: they are state.
            initialize_coupled_fields!(RP)
        end

        # Establish the invariant the loop only maintains: its refresh runs at the END of
        # the body, so the first step would otherwise consume the rates and the operator
        # cache of the state the last refresh saw — `initialize!`'s for a fresh run, the
        # predecessor's final step for a resumed one — rather than the state and flags
        # handed over since (an initial condition, a changed `flags.upwind`). The refresh
        # rebuilds everything from the current state, so on an untouched resume it changes
        # nothing and splitting a run in two stays bit for bit. Ahead of the snapshots,
        # which report those rates too. notes/issues/stale-rrcs-on-first-step.md
        update_transport_quantities!(RP)

        # Initial snapshots at t_start_s
        @timeit RAPID_TIMER "initial_snapshots" begin
            update_snaps0D!(RP)
            update_snaps2D!(RP)

            # Save initial snapshots at t_start_s
            write_latest_snap0D!(RP)
            write_latest_snap2D!(RP)
        end

        # Main time loop
        @timeit RAPID_TIMER "main_time_loop" begin
            while RP.time_s < t_end - 0.1 * dt

                # One time step: set its inputs from the state at tⁿ, let the caller see or
                # adjust them, then solve to tⁿ⁺¹
                prepare_timestep!(RP)
                if !isnothing(callback_before_step)
                    callback_before_step(RP)
                end
                solve_timestep!(RP, dt)

                # Increment time
                RP.time_s += dt
                RP.step += 1

                # Ledgers and floors after the step. Nothing zeroes the band outside the
                # wall: no operator writes there.
                book_ionization_sources!(RP)
                correct_negative_densities!(RP)

                if RP.step == 1 || mod(RP.step, RP.flags.FLF_nstep) == 0
                    @timeit RAPID_TIMER "field_line_following" begin
                        # `strict = false`: a mid-run refresh warns and substitutes a
                        # conservative length instead of killing the run; setup
                        # (`initialize!`) keeps the strict default.
                        flf_analysis_field_lines_rz_plane!(RP; strict = false)
                        # if !isempty(RP.flf.closed_surface_nids)
                        #     RP.flags.FLF_nstep=1;
                        # end
                    end
                end

                # Calculate self-consistent electrostatic field if enabled
                if RP.flags.E_para_self_ES
                    @timeit RAPID_TIMER "electrostatic_field" begin
                        estimate_electrostatic_field_effects!(RP)
                    end
                end

                # Update transport coefficients after all state variables are updated
                @timeit RAPID_TIMER "transport_quantities" begin
                    update_transport_quantities!(RP)
                end

                # Print progress
                if RP.step % 100 == 0
                    @printf("Time: %.6e s, Step: %d\n", RP.time_s, RP.step)
                end

                # Handle snapshots and file outputs if needed
                if is_snap0D_time(RP)
                    @timeit RAPID_TIMER "snapshot 0D" begin
                        update_snaps0D!(RP)
                        write_latest_snap0D!(RP)
                    end
                end

                if is_snap2D_time(RP)
                    @timeit RAPID_TIMER "snapshot 2D" begin
                        update_snaps2D!(RP)
                        write_latest_snap2D!(RP)
                    end
                end


                if !isnothing(controller)
                    update_controller!(RP, controller)
                end

                if !isnothing(callback_after_step)
                    callback_after_step(RP)
                end
            end
        end

        println("Simulation completed")

        return RP
    end
end

# Export workflow functions
export advance_timestep!, prepare_timestep!, solve_timestep!, run_simulation!, initialize_coupled_fields!

# Export timer utilities
export RAPID_TIMER, print_timer_results, save_timer_results

"""
    print_timer_results()

Print detailed timing results from the RAPID simulation.
"""
function print_timer_results()
    println("")
    show(RAPID_TIMER; title = "RAPID2D Timing Results", allocations = true, sortby = :time, linechars = :unicode, compact = false)
    return println("")
end


"""
    save_timer_results(filename::String)

Save timing results to a file.

# Arguments
- `filename::String`: Output filename (will be saved in current directory)
"""
function save_timer_results(filename::String = "rapid_timing_results.txt")
    open(filename, "w") do io
        println(io, "RAPID2D Performance Timing Results")
        println(io, "Generated at: $(now())")
        println(io, "="^60)
        print_timer(io, RAPID_TIMER)
    end
    return println("Timing results saved to: $filename")
end
