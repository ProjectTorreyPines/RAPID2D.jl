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
    return abs(RP_Δt_s - round(RP_Δt_s/RP.config.snap0D_Δt_s) * RP.config.snap0D_Δt_s) < 0.1 * RP.dt
end

"""
    is_snap2D_time(time)

Check if the given time corresponds to a 2D snapshot time.

Returns `true` if the time matches a 2D snapshot timing, `false` otherwise.
"""
function is_snap2D_time(RP)
    RP_Δt_s = RP.time_s - RP.t_start_s
    return abs(RP_Δt_s- round(RP_Δt_s/RP.config.snap2D_Δt_s) * RP.config.snap2D_Δt_s) < 0.1 * RP.dt
end


"""
    advance_timestep!(RP::RAPID{FT}, dt::FT) where FT<:AbstractFloat

Advance the simulation by one time step, coupling all physics processes.
This function represents the core time-stepping algorithm of RAPID2D.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing all simulation state
- `dt::FT`: Time step size in seconds

# Returns
- `RP`: The updated RAPID object after advancement

# Main process flow:
1. Field calculation (vacuum + self-consistent)
2. Particle transport (density continuity equations)
3. Energy transport (temperature evolution)
4. Transport coefficient updates
"""
function advance_timestep!(RP::RAPID{FT}, dt::FT=RP.dt) where FT<:AbstractFloat
    @timeit RAPID_TIMER "advance_timestep!" begin

        # Update vacuum fields from external sources
        @timeit RAPID_TIMER "external_fields" begin
            update_external_fields!(RP)
        end

        # Current calculations
        @timeit RAPID_TIMER "current_calculation" begin
            # Calculate the current density based on electron drift velocity
            RP.plasma.Jϕ .= (-RP.config.ee) .* RP.plasma.ne .* RP.plasma.ue_para .* RP.fields.bϕ
            I_tor = sum(RP.plasma.Jϕ * RP.G.dR * RP.G.dZ)  # Total toroidal current
        end

        # For high current: update electromagnetic fields using Ampere's law
        if RP.flags.Ampere && abs(I_tor) >= RP.flags.Ampere_Itor_threshold
            if RP.flags.E_para_self_EM && RP.flags.ud_evolve && RP.flags.ud_method == "Xsec"
                # Solve the coupled drift velocity and magnetic field equations
                # @timeit RAPID_TIMER "solve_coupled_momentum_Ampere_equations_with_coils!" solve_coupled_momentum_Ampere_equations_with_coils!(RP)
                solve_combined_momentum_Ampere_equations_with_coils!(RP)
            else
                # Update drift velocity separately
                if RP.flags.ud_evolve
                    update_ue_para!(RP)
                end

                # Solve the Grad-Shafranov equation for the magnetic field
                @timeit RAPID_TIMER "solve_Ampere_equation!" solve_Ampere_equation!(RP)
            end
        else
            # For low current: only update drift velocity
            if RP.flags.ud_evolve
                update_ue_para!(RP)
            end
        end

        combine_external_and_self_fields!(RP)

        # Update electron density
        solve_electron_continuity_equation!(RP)

        # Ion dynamics
        if RP.flags.update_ni_independently
            # Update ion drift velocity
            update_ui_para!(RP)

            # Solve ion continuity equation
            solve_ion_continuity_equation!(RP)

            # Update ion temperature if needed
            update_Ti!(RP)
        else
            # Update ion drift velocity
            update_ui_para!(RP)

            # Set ion density from electron density with charge neutrality
            RP.plasma.ni .= RP.plasma.ne ./ RP.plasma.Zeff

            # Update ion temperature if needed
            update_Ti!(RP)
        end


        if RP.flags.Global_Force_Balance
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

        return RP
    end
end

"""
    run_simulation!(RP::RAPID{FT}) where FT<:AbstractFloat

Run a full simulation from current time to the end time specified in the RAPID object.
Handles time stepping, diagnostics output, and snapshot generation.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing all simulation state

# Returns
- `RP`: The updated RAPID object after completion of the simulation
"""
function run_simulation!(RP::RAPID{FT}) where FT<:AbstractFloat
    @timeit RAPID_TIMER "run_simulation!" begin
        # Simulation parameters
        dt = RP.dt
        t_end = RP.t_end_s

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
            while RP.time_s < t_end - 0.1*dt

                # Advance simulation one time step
                advance_timestep!(RP, dt)

                # Increment time
                RP.time_s += dt
                RP.step += 1

                treat_electron_outside_wall!(RP)
                if RP.flags.update_ni_independently
                    treat_ion_outside_wall!(RP)
                end

                if mod(RP.step, RP.flags.FLF_nstep)==0
                    @timeit RAPID_TIMER "field_line_following" begin
                        flf_analysis_field_lines_rz_plane!(RP)
                        if !isempty(RP.flf.closed_surface_nids)
                            RP.flags.FLF_nstep=1;
                        end
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

            end
        end

        println("Simulation completed")

        return RP
    end
end

# Export workflow functions
export advance_timestep!, run_simulation!

# Export timer utilities
export RAPID_TIMER, print_timer_results, save_timer_results

"""
    print_timer_results()

Print detailed timing results from the RAPID simulation.
"""
function print_timer_results()
    println("")
    show(RAPID_TIMER; title="RAPID2D Timing Results", allocations=true, sortby=:time, linechars=:unicode, compact=false)
    println("")
end



"""
    save_timer_results(filename::String)

Save timing results to a file.

# Arguments
- `filename::String`: Output filename (will be saved in current directory)
"""
function save_timer_results(filename::String="rapid_timing_results.txt")
    open(filename, "w") do io
        println(io, "RAPID2D Performance Timing Results")
        println(io, "Generated at: $(now())")
        println(io, "="^60)
        print_timer(io, RAPID_TIMER)
    end
    println("Timing results saved to: $filename")
end