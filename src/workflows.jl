"""
Workflows module for RAPID2D.

Contains high-level simulation workflows, including:
- Time stepping algorithms
- Simulation advancement functions
- Multi-physics coupling strategies
"""

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
function advance_timestep!(RP::RAPID{FT}, dt::FT) where FT<:AbstractFloat
    # Update vacuum fields from external sources
    update_vacuum_fields!(RP)

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

    # Combine vacuum and self-generated fields
    RP.fields.BR .= RP.fields.BR_ext .+ RP.fields.BR_self
    RP.fields.BZ .= RP.fields.BZ_ext .+ RP.fields.BZ_self
    RP.fields.Eϕ .= RP.fields.Eϕ_ext .+ RP.fields.Eϕ_self

    # Update derived magnetic field quantities
    calculate_derived_fields!(RP)

    # Update parallel electric field components
    RP.fields.E_para_ext .= RP.fields.Eϕ_ext .* RP.fields.bϕ
    RP.fields.E_para_ext .*= RP.damping_func  # Apply damping outside wall

    if RP.flags.E_para_self_EM
        RP.fields.E_para_self_EM .= RP.fields.Eϕ_self .* RP.fields.bϕ
    else
        RP.fields.E_para_self_EM .= 0.0
    end

    # Calculate self-consistent electrostatic field if enabled
    if RP.flags.E_para_self_ES
        estimate_electrostatic_field_effects!(RP)
    end

    # Update total parallel electric field
    RP.fields.E_para_tot .= RP.fields.E_para_ext .+
                          (RP.flags.E_para_self_ES ? RP.fields.E_para_self_ES : 0.0) .+
                          (RP.flags.E_para_self_EM ? RP.fields.E_para_self_EM : 0.0)

    # Calculate RHS terms for density equation
    if RP.flags.src
        calculate_density_source_terms!(RP)
    end

    if RP.flags.diffu
        calculate_density_diffusion_terms!(RP)
    end

    if RP.flags.convec
        calculate_density_convection_terms!(RP)
    end

    # Update electron density
    solve_electron_continuity_equation!(RP)

    # Apply boundary conditions and handle negative values
    apply_electron_density_boundary_conditions!(RP)

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

    # Update transport coefficients after all state variables are updated
    update_transport_coefficients!(RP)
    
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

    # Main time loop
    while RP.time_s < t_end - 0.1*dt

        # Advance simulation one time step
        advance_timestep!(RP, dt)

        # Increment time
        RP.time_s += dt
        RP.step += 1

        # Print progress
        if RP.step % 100 == 0
            @printf("Time: %.6e s, Step: %d\n", RP.time_s, RP.step)
        end

        # Handle snapshots and file outputs if needed
        if hasfield(typeof(RP), :snap2D_Interval_s) && abs(RP.time_s - round(RP.time_s/RP.snap2D_Interval_s)*RP.snap2D_Interval_s) < 0.1*dt
            # Take snapshot of 2D data
            save_snapshot2D(RP)
        end
    end

    println("Simulation completed")
    return RP
end

# Export workflow functions
export advance_timestep!, run_simulation!