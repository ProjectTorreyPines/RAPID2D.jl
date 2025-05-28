"""
Workflows module for RAPID2D.

Contains high-level simulation workflows, including:
- Time stepping algorithms
- Simulation advancement functions
- Multi-physics coupling strategies
"""


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
    # Update vacuum fields from external sources
    update_external_fields!(RP)

    # Current calculations
    # Calculate the current density based on electron drift velocity
    RP.plasma.Jϕ .= (-RP.config.ee) .* RP.plasma.ne .* RP.plasma.ue_para .* RP.fields.bϕ

    # For high current: update electromagnetic fields using Ampere's law
    if RP.flags.Ampere && sum(abs.(RP.plasma.Jϕ) .* RP.G.inVol2D) * RP.G.dR * RP.G.dZ >= RP.flags.Ampere_Itor_threshold
        if RP.flags.E_para_self_EM && RP.flags.ud_evolve && RP.flags.ud_method == "Xsec" && RP.step % RP.flags.Ampere_nstep == 0
            # Solve the coupled drift velocity and magnetic field equations
            solve_coupled_ud_GS_equations!(RP)
        else
            # Update drift velocity separately
            if RP.flags.ud_evolve
                update_ue_para!(RP)
            end

            # Solve the Grad-Shafranov equation for the magnetic field
            solve_grad_shafranov_equation!(RP)
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
    # Simulation parameters
    dt = RP.dt
    t_end = RP.t_end_s

    # Initial snapshots at t=0
    measure_snap0D!(RP)
    measure_snap2D!(RP)

    # Main time loop
    while RP.time_s < t_end - 0.1*dt

        # Advance simulation one time step
        advance_timestep!(RP, dt)

        # Increment time
        RP.time_s += dt
        RP.step += 1

        # Apply boundary conditions and handle negative values
        apply_electron_density_boundary_conditions!(RP)

        # Calculate self-consistent electrostatic field if enabled
        if RP.flags.E_para_self_ES
            estimate_electrostatic_field_effects!(RP)
        end
        # Update transport coefficients after all state variables are updated
        update_transport_quantities!(RP)


        # Print progress
        if RP.step % 100 == 0
            @printf("Time: %.6e s, Step: %d\n", RP.time_s, RP.step)
        end


        # Handle snapshots and file outputs if needed
        if is_snap0D_time(RP)
            measure_snap0D!(RP)
        end
        if is_snap2D_time(RP)
            measure_snap2D!(RP)
        end

        # Handle snapshots and file outputs if needed
        # if hasfield(typeof(RP), :snap2D_Δt_s) && abs(RP.time_s - round(RP.time_s/RP.snap2D_Δt_s)*RP.snap2D_Δt_s) < 0.1*dt
        #     # Take snapshot of 2D data
        #     save_snapshot2D(RP)
        # end

        # if(obj.Flag.Adapt_dt)
        #     obj.Check_adaptive_dt();
        # end

        # obj.tElap.Main = toc(tMainLoopStart);
        # obj.print_output()

        # obj.write_output_file()
    end

    # if(obj.Flag.vis1D)
    #     obj.vis_snap1D(obj.snap1D);
    # end
    # if(obj.Flag.vis2D)
    #     obj.vis_snap2D(obj.snap2D);
    # end
    println("Simulation completed")
    return RP
end

# Export workflow functions
export advance_timestep!, run_simulation!