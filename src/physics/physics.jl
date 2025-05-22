"""
Physics module for RAPID2D.

Contains functions related to plasma physics models, including:
- Electron and ion dynamics
- Collisions
- Reaction rates
- Power balance
"""

# Export public functions
export update_ue_para!,
       update_ui_para!,
       update_Te!,
       update_Ti!,
       update_coulomb_collision_parameters!,
       update_electron_heating_powers!,
       calculate_density_source_terms!,
       calculate_density_diffusion_terms!,
       calculate_density_convection_terms!,
       solve_electron_continuity_equation!,
       apply_electron_density_boundary_conditions!,
       calculate_para_grad_of_scalar_F,
       calculate_grad_of_scalar_F,
       calculate_electron_acceleration_by_pressure,
       calculate_electron_acceleration_by_convection

"""
    update_ue_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the parallel electron velocity.
"""
function update_ue_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Define constants at function start for type stability
    one_FT = one(FT)
    zero_FT = zero(FT)

    # Update method depends on flag
    if RP.flags.ud_method == "Lloyd_fit"
        # Simple fit for drift velocity
        @. RP.plasma.ue_para = 5719.0 * (-RP.fields.E_para_tot / RP.config.prefilled_gas_pressure)

    elseif RP.flags.ud_method == "Xsec_fit"
        # Cross section fit with simplified collisions
        qe = -RP.config.ee

        # Get momentum transfer reaction rate coefficient
        RRC_mom = get_electron_RRC(RP, RP.eRRCs, :Momentum)

        # Calculate collision frequency
        tot_coll_freq = @. RP.plasma.n_H2_gas * RRC_mom

        # Add Coulomb collisions if enabled
        if RP.flags.Coulomb_Collision
            if RP.flags.Spitzer_Resistivity
                @. tot_coll_freq += RP.plasma.sptz_fac * RP.plasma.ν_ei
            else
                @. tot_coll_freq += RP.plasma.ν_ei
            end
        end

        # Calculate drift velocity from balance of electric field and collisions
        @. RP.plasma.ue_para = qe * RP.fields.E_para_tot / (RP.config.me * tot_coll_freq)

    elseif RP.flags.ud_method == "Xsec"
        # Full cross section model with collisions
        qe = -RP.config.constants.ee
        me = RP.config.constants.me
        dt = RP.dt

        pla = RP.plasma
        F = RP.fields

        # allocate arrays
        tot_coll_freq = zeros(FT, size(pla.ue_para))
        mom_eff_nu_ei = zeros(FT, size(pla.ue_para))

        if RP.flags.Atomic_Collision
            # Get reaction rate coefficients for both ionization and momentum transfer
            RRC_iz = get_electron_RRC(RP, RP.eRRCs, :Ionization)
            RRC_mom = get_electron_RRC(RP, RP.eRRCs, :Momentum)

            # Calculate collision frequency from neutrals
            @. tot_coll_freq += pla.n_H2_gas * (RRC_mom + RRC_iz)
        end

        # Add Coulomb collisions if enabled
        if RP.flags.Coulomb_Collision
            if RP.flags.Spitzer_Resistivity
                @. mom_eff_nu_ei = pla.sptz_fac * pla.ν_ei
            else
                @. mom_eff_nu_ei = pla.ν_ei
            end
            @. tot_coll_freq += mom_eff_nu_ei
        end

        # Ensure no NaN values in collision frequency
        if(any(isnan.(tot_coll_freq)))
            @warn "NaN values in collision frequency detected. Replacing with zero."
            # Replace NaN values with zero
            tot_coll_freq[isnan.(tot_coll_freq)] .= zero_FT
            # replace!(tot_coll_freq, NaN => zero_FT)
        end


        # Always use backward Euler for ue_para (θu=1.0) for better saturation
        # but keep the formula structure compatible with variable θ_imp
        # θu = RP.flags.Implicit_weight
        θu = one_FT

        # Calculate Rue_ei (electron-ion momentum exchange rate) - first part (n-th step)
        if RP.flags.Coulomb_Collision
            @. pla.Rue_ei = mom_eff_nu_ei * (pla.ui_para - (one_FT - θu) * pla.ue_para)
        end

        # Advance ue_para using implicit or explicit method
        if RP.flags.Implicit
            # Implicit scheme implementation
            OP = RP.operators
            @. OP.A_LHS = OP.II

            # #1: Electric acceleartion term [qe*E_para_tot/me]
            accel_para_tilde = qe * F.E_para_tot / me

            # #2: Advection term (1-θ)*[-(𝐮⋅∇)*ue_para]
            if RP.flags.Include_ud_convec_term
                update_𝐮∇_operator!(RP)
                @views accel_para_tilde[:] .+= (one_FT - θu) * (-OP.𝐮∇ * pla.ue_para[:])
                @. OP.A_LHS += θu * dt * OP.𝐮∇
            end

            # #3: Pressure term [-∇∥(ne*Te)/(me*ne)]
            accel_para_tilde .+= calculate_electron_acceleration_by_pressure(RP)

            # #4: collision drag force  (1-θ)*[-(ν_tot)*ue_para]
            @. accel_para_tilde += (one_FT - θu) * (-tot_coll_freq * pla.ue_para)

            diag_indices = diagind(OP.A_LHS)
            @. @views OP.A_LHS[diag_indices] += θu * dt * tot_coll_freq[:]

            # #5: momentum source from electron-ion collision [+sptz_fac*νei*ui_para]
            @. accel_para_tilde += (mom_eff_nu_ei * pla.ui_para)

            # #6: turbulent Diffusive term by ExB mixing
            if RP.flags.Include_ud_diffu_term
                @warn "Turbulent diffusion term not implemented yet" maxlog=1
                @. accel_para_tilde += (one_FT - θu) * (OP.A_𝐮_diffu * pla.ue_para[:])
                @. OP.A_LHS += θu * dt * OP.A_𝐮_diffu
            end

            # Set-up the RHS
            @. OP.RHS = pla.ue_para + dt * accel_para_tilde

            # Solve the momentum equation
            @views pla.ue_para[:] .= OP.A_LHS \ OP.RHS[:]
        else

            inv_factor = @. one_FT / (one_FT + θu * tot_coll_freq * dt)
            @. pla.ue_para = inv_factor*(
                                    pla.ue_para * (one_FT-(one_FT-θu)*dt*tot_coll_freq)
                                    + dt * (qe * F.E_para_tot / me + mom_eff_nu_ei * pla.ui_para)
                                )

            # Add pressure and convection terms in the same way as MATLAB
            if RP.flags.Include_ud_convec_term
                accel_by_pressure = calculate_electron_acceleration_by_pressure(RP)
                accel_by_grad_ud = calculate_electron_acceleration_by_convection(RP)
                @. pla.ue_para +=  inv_factor * dt * (accel_by_pressure + accel_by_grad_ud)
            end
        end

        # Complete the Rue_ei calculation with second part (n+1 step contribution)
        if RP.flags.Coulomb_Collision
            @. pla.Rue_ei += mom_eff_nu_ei * (-θu * pla.ue_para)
        end
    else
        error("Unknown electron drift velocity method: $(RP.flags.ud_method)")
    end

    return RP
end

"""
    update_ui_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the parallel ion velocity.
"""
function update_ui_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    if RP.flags.ud_method == "Xsec"
        # Use cross sections for ion velocity update
        # Alias
        cnst = RP.config.constants
        pla = RP.plasma

        eff_atomic_coll_freq = zeros(FT, size(pla.ui_para))
        if RP.flags.Atomic_Collision
            # Get ion reaction rate coefficients
            iRRC_elastic = get_H2_ion_RRC(RP, RP.iRRCs, :Elastic)
            iRRC_cx = get_H2_ion_RRC(RP, RP.iRRCs, :Charge_Exchange)

            # Add ionization contribution if source terms are enabled
            if RP.flags.src
                eRRC_iz = get_electron_RRC(RP, RP.eRRCs, :Ionization)
            else
                eRRC_iz = 0.0
            end

            # Calculate effective atomic collision frequency
            # Note: 0.5 factor for elastic collisions because they only lose half of momentum
            eff_atomic_coll_freq = @. pla.n_H2_gas * (
                FT(0.5) * iRRC_elastic + iRRC_cx + pla.Zeff * eRRC_iz
            )

            # NOTE: convection and pressure contribution are ignored for ions
            # TODO: Add pressure and convection terms for ions if needed, check Zeff effects

            # Fix any NaN values
            replace!(eff_atomic_coll_freq, NaN => 0.0)
        end

        # Calculate acceleration from electric field
        qi = cnst.ee

        # Apply backward Euler time integration
        one_FT = one(FT)
        θ = one_FT  # Backward Euler
        @. pla.ui_para = (pla.ui_para * (one_FT - (one_FT-θ) * RP.dt * eff_atomic_coll_freq) +
                              RP.dt * qi * RP.fields.E_para_tot / cnst.mi) /
                              (one_FT + θ * RP.dt * eff_atomic_coll_freq)

        # Add electron-ion momentum transfer effect
        if RP.flags.Coulomb_Collision
            if RP.flags.Spitzer_Resistivity
                @. Rui_ei = pla.sptz_fac*(cnst.me/cnst.mi)* pla.ν_ei * (pla.ue_para - pla.ui_para)
            else
                @. Rui_ei = (cnst.me/cnst.mi)*pla.ν_ei * (pla.ue_para - pla.ui_para)
            end
            pla.ui_para .+= RP.dt * Rui_ei
        end
    else
        error("Ion velocity update only implemented for ud_method = Xsec")
    end
end

"""
    update_Te!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the electron temperature based on energy balance equation.

This function evolves the electron temperature by solving the electron energy equation:
3/2 n_e ∂T_e/∂t = P_heat - P_loss + ∇·(κ_e ∇T_e) - 3/2 n_e u_e·∇T_e

where:
- P_heat includes ohmic heating and other power sources
- P_loss includes ionization, excitation, radiation, and equilibration losses
- κ_e is the thermal conductivity
- The last term represents convective transport

The implementation supports both explicit and implicit time integration schemes.
"""
function update_Te!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Skip temperature evolution if flag is disabled
    if !RP.flags.Te_evolve
        return RP
    end

    # Skip temperature evolution outside wall if flag is set
    if RP.flags.evolve_Te_inWall_only
        # Save old temperature for later restoration
        Te_old = copy(RP.plasma.Te_eV)
    end

    # Update power terms for energy equation
    update_electron_heating_powers!(RP)

    # Define constants for type stability
    one_FT = one(FT)
    zero_FT = zero(FT)
    dt = RP.dt

    # Minimum density to avoid division by zero
    n_min = FT(1.0e10)

    # Apply time integration method based on flag
    if RP.flags.Implicit && (RP.flags.Include_Te_diffu_term || RP.flags.Include_Te_convec_term)
        # Implicit method
        # Initialize matrices
        OP = RP.operators
        @. OP.A_LHS = OP.II
        NR, NZ = RP.G.NR, RP.G.NZ

        # Add total power density to RHS
        # Scale by (2/3)/n_e to get dT/dt
        n_safe = copy(RP.plasma.ne)
        n_safe[n_safe .< n_min] .= n_min
        P_total = copy(RP.plasma.ePowers.tot)

        # Apply density safety factor
        dTdt = @. (FT(2.0)/FT(3.0)) * P_total / n_safe

        # Set-up RHS vector with explicit contribution: T_e^n + dt*(1-θ)*dTe/dt
        θ = RP.flags.Implicit_weight
        @. OP.RHS = RP.plasma.Te_eV + dt * (one_FT - θ) * dTdt

        # Add temperature diffusion term if enabled
        if RP.flags.Include_Te_diffu_term
            # Add diffusive operator on LHS: A_LHS = I - θ*dt*D_T
            # For now, using a simplified diffusion operator
            # In a complete implementation, need to construct a proper diffusion operator
            # 1. Set diffusion coefficients based on plasma parameters
            # Parallel thermal conductivity κ_∥ = κ_0 n_e T_e^(5/2) / (m_e^(1/2) Z)
            # Cross-field thermal conductivity κ_⊥ is much smaller
            # For now, using constant diffusivities scaled by temperature
            D_para_T = RP.transport.Dpara0 * FT(100.0) # Enhanced thermal diffusivity
            D_perp_T = RP.transport.Dperp0 * FT(10.0)  # Enhanced thermal diffusivity

            # 2. Build diffusion operator - this is a simplified placeholder
            # In a real implementation, we would build the full diffusion operator
            # that properly accounts for magnetic geometry, similar to ∇𝐃∇
            # But for now, we'll assume a simplified form
            A_T_diffu = spzeros(FT, NR*NZ, NR*NZ)

            # 3. Add diagonal contribution (center node)
            # This is just a simplified placeholder - the real implementation would
            # need a properly constructed diffusion operator
            diag_indices = diagind(A_T_diffu)
            @. @views A_T_diffu[diag_indices] = -4.0 * (D_para_T * RP.plasma.Te_eV[:] / FT(RP.G.dR^2))

            # 4. Add off-diagonal contributions (neighbor nodes) - placeholder
            # Note: this is a simplified example that doesn't account for field geometry
            # A proper implementation would need to account for the magnetic field direction

            # 5. Add diffusion operator to LHS matrix with implicit weighting
            @. OP.A_LHS += θ * dt * A_T_diffu

            # 6. Add explicit diffusion contribution to RHS
            # For complete implementation, would compute: (1-θ)*dt*∇·(κ_e ∇T_e)
            # Simplified place-holder would be: (1-θ)*dt*A_T_diffu*T_e
            @. OP.RHS += (one_FT - θ) * dt * (A_T_diffu * RP.plasma.Te_eV[:])
        end

        # Add temperature convection term if enabled
        if RP.flags.Include_Te_convec_term
            # Add convective operator on LHS: A_LHS = I - θ*dt*C_T
            # For a complete implementation, need to construct a proper convection operator
            # similar to An_convec, but for temperature

            # 1. Build convection operator for temperature using upwind method
            # This is a simplified placeholder
            A_T_convec = spzeros(FT, NR*NZ, NR*NZ)

            # 2. Add convection operator to LHS matrix with implicit weighting
            @. OP.A_LHS += θ * dt * A_T_convec

            # 3. Add explicit convection contribution to RHS
            # For complete implementation, would compute: (1-θ)*dt*(-u_e·∇T_e)
            # Simplified place-holder would be: (1-θ)*dt*A_T_convec*T_e
            @. OP.RHS += (one_FT - θ) * dt * (A_T_convec * RP.plasma.Te_eV[:])
        end

        # Solve the linear system
        @views RP.plasma.Te_eV[:] = OP.A_LHS \ OP.RHS[:]

    else
        # Explicit method (forward Euler)

        # Energy equation: 3/2 n ∂T/∂t = P_total
        # dT/dt = (2/3) * P_total / n
        # T_new = T_old + dt * (2/3) * P_total / n

        # Avoid division by zero
        n_safe = copy(RP.plasma.ne)
        n_safe[n_safe .< n_min] .= n_min

        # Temperature change rate
        dTdt = @. (FT(2.0)/FT(3.0)) * RP.plasma.ePowers.tot / n_safe

        # Update temperature
        @. RP.plasma.Te_eV += dTdt * dt

        # Add diffusion and convection terms in explicit step if needed
        # This would require calculating ∇·(κ_e ∇T_e) and -u_e·∇T_e
        # Placeholder for future implementation
    end

    # Apply temperature limits
    @. RP.plasma.Te_eV = max(RP.plasma.Te_eV, RP.config.min_Te)
    @. RP.plasma.Te_eV = min(RP.plasma.Te_eV, RP.config.max_Te)

    # Restore old temperature outside wall if evolving only inside wall
    if RP.flags.evolve_Te_inWall_only
        # Set temperature to old values at out-wall nodes
        RP.plasma.Te_eV[RP.G.nodes.out_wall_nids] .= Te_old[RP.G.nodes.out_wall_nids]
    else
        # Set temperature to minimum outside wall
        RP.plasma.Te_eV[RP.G.nodes.out_wall_nids] .= RP.config.min_Te
    end

    return RP
end

"""
    update_Ti!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the ion temperature.
"""
function update_Ti!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    if !RP.flags.Ti_evolve
        # If ion temperature evolution is disabled, just match electron temperature
        RP.plasma.Ti_eV .= copy(RP.plasma.Te_eV)
        return RP
    end

    @warn "update_Ti! not fully implemented yet"

    # Update ion temperature based on power balance
    # Similar to update_Te! but with ion power terms

    # Energy equation: 3/2 n ∂T/∂t = P_total
    # Simplified implementation: forward Euler
    # dT/dt = (2/3) * P_total / n

    # Avoid division by zero
    n_min = FT(1.0e6)
    n_safe = copy(RP.plasma.ni)
    n_safe[n_safe .< n_min] .= n_min

    # Temperature change rate
    dTdt = (FT(2.0)/FT(3.0)) * RP.plasma.iPowers.tot ./ n_safe

    # Update temperature
    RP.plasma.Ti_eV .+= dTdt * RP.dt

    # Apply temperature limits
    RP.plasma.Ti_eV .= max.(RP.plasma.Ti_eV, RP.config.min_Te) # Using same min as electrons

    # # Zero temperature outside wall
    # RP.plasma.Ti_eV[RP.G.nodes.out_wall_nids] .= RP.config.min_Te

    return RP
end

"""
    update_electron_heating_powers!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update electron heating power components for electron energy equation.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- `RP`: The updated RAPID object

# Notes
- Calculates all electron power sources and sinks:
  - Diffusion and convection powers (if enabled)
  - Collision drag power
  - Heat generation from density gradients
  - Ionization, excitation, and dilution powers
  - Temperature equilibration power with ions
- All powers stored in the RP.plasma.ePowers struct
"""
function update_electron_heating_powers!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Ionization energy for H2 molecule (H2->H2+ + e-) in eV
    iz_erg_eV = FT(15.46)

    # TODO: Replace with actual average excitation energy
    avg_exc_erg_eV = FT(12.0)  # Average excitation energy in eV (assumption)

    # Extract physical constants
    @unpack ee, qe, me, mi = RP.config.constants

    # Alias common objects for readability
    pla = RP.plasma
    ePowers = RP.plasma.ePowers

    zero_FT = zero(FT)

    # Reset all power arrays to zero (precaution to avoid accumulation)
    ePowers.diffu .= zero_FT
    ePowers.conv .= zero_FT
    ePowers.heat .= zero_FT
    ePowers.drag .= zero_FT
    ePowers.iz .= zero_FT
    ePowers.exc .= zero_FT
    ePowers.dilution .= zero_FT
    ePowers.equi .= zero_FT

    # If diffusion term is included in temperature equation
    if RP.flags.Include_Te_diffu_term
        @warn "Include_Te_diffu_term not implemented yet" maxlog=1
        # Calculate diffusive power transfer (this would be calculated elsewhere and stored)
        # In a full implementation, this would use the diffusion operator on Te
    end

    # If convection term is included in temperature equation
    if RP.flags.Include_Te_convec_term
        @warn "Include_Te_convec_term not implemented yet" maxlog=1
        # Calculate convective power transfer (also calculated elsewhere)
        # In a full implementation, this would use the convection operator on Te
    end

    if RP.flags.Include_heat_flux_term
        # NOTE: Assumption: 𝐪 ≈ p*𝐮 (heat flux is about in the order of pressure*velocity)
        # Calculate heating from density gradients
        # Smooth density field to avoid numerical issues with gradients
        # n_SM = smooth_data_2D(pla.ne; num_SM=2)
        # n_SM[n_SM .< zero_FT] .= zero_FT

        # Calculate gradient of log(n)
        ∇log_n_R, ∇log_n_Z = calculate_gradient(RP, log.(pla.ne))

        # Heat flux from density gradient
        @. ePowers.heat = -ee * pla.Te_eV * (pla.ueR * ∇log_n_R + pla.ueZ * ∇log_n_Z)


        @unpack ∇𝐮, 𝐮∇ = RP.operators

        -ee*pla.Te_eV.*(𝐮∇*log.(pla.ne))

        # # Handle NaN values
        # replace!(ePowers.heat, NaN => zero_FT)
    end


    if RP.flags.Atomic_Collision
        # Get reaction rate coefficients for momentum transfer
        RRC_mom = get_electron_RRC(RP, :Momentum)

        # Calculate velocity magnitudes for drag forces
        ue_mag_sq = @. pla.ueR^2 .+ pla.ueϕ^2 .+ pla.ueZ^2
        ue_dot_ui = @. pla.ueR * pla.uiR + pla.ueϕ * pla.uiϕ + pla.ueZ * pla.uiZ

        @. ePowers.drag = me * (
            ue_mag_sq * pla.n_H2_gas * RRC_mom
            + (ue_mag_sq - ue_dot_ui) * pla.sptz_fac * pla.ν_ei
        )

        # Get excitation rate coefficient
        RRC_exc = get_electron_RRC(RP, :Total_Excitation)
        # Excitation power (energy lost to excite particles)
        @. ePowers.exc = ee * avg_exc_erg_eV * pla.n_H2_gas * RRC_exc

        # For ionization
        if RP.flags.src
            # Get ionization rate coefficient
            RRC_iz = get_electron_RRC(RP, :Ionization)
            freq_iz = @. pla.n_H2_gas * RRC_iz

            # Ionization power (energy lost to ionize particles)
            @. ePowers.iz = freq_iz * iz_erg_eV * ee

            # Dilution power (energy change due to density increase)
            @. ePowers.dilution = freq_iz * (
                FT(1.5) * pla.Te_eV * ee
                - FT(0.5) * me * ue_mag_sq
            )
        end
    end


    # Equilibration power with ions (energy exchange from temperature differences)
    if RP.flags.Coulomb_Collision
        # Factor for energy transfer rate between electrons and ions
        @. ePowers.equi = (FT(2.0)*(mi*me/(mi+me)^2)) * FT(1.5) * ee * (pla.Te_eV - pla.Ti_eV) * pla.ν_ei
    end

    # Calculate total power (sum of all components)
    @. ePowers.tot = (
            ePowers.drag + ePowers.conv + ePowers.heat + ePowers.diffu
            - ePowers.dilution - ePowers.iz - ePowers.exc - ePowers.equi
        )

    # # Zero out power values outside the wall
    # out_wall_nids = RP.G.nodes.out_wall_nids
    # if !isempty(out_wall_nids)
    #     @views ePowers.tot[out_wall_nids] .= zero_FT
    #     @views ePowers.diffu[out_wall_nids] .= zero_FT
    #     @views ePowers.conv[out_wall_nids] .= zero_FT
    #     @views ePowers.drag[out_wall_nids] .= zero_FT
    #     @views ePowers.dilution[out_wall_nids] .= zero_FT
    #     @views ePowers.iz[out_wall_nids] .= zero_FT
    #     @views ePowers.exc[out_wall_nids] .= zero_FT
    #     @views ePowers.equi[out_wall_nids] .= zero_FT
    #     @views ePowers.heat[out_wall_nids] .= zero_FT
    # end

    return RP
end

"""
    get_avg_RRC_Te_ud_gFac(RP::RAPID{FT}, reaction::String, Te_eV::Matrix{FT}, ue_para::Matrix{FT}, gFac::Matrix{FT}) where {FT<:AbstractFloat}

Calculate the average reaction rate coefficient for a specified reaction, accounting for electron temperature, drift velocity, and distribution function deformation (g-factor).
"""
function get_avg_RRC_Te_ud_gFac(RP::RAPID{FT}, reaction::String, Te_eV::Matrix{FT}, ue_para::Matrix{FT}, gFac::Matrix{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "get_avg_RRC_Te_ud_gFac not fully implemented yet"

    # In a real implementation, this would interpolate in the reaction rate tables
    # For now, return a simple approximation

    # Simple Arrhenius form: A * exp(-E_a/T)
    if reaction == "Ionization"
        A = FT(2.0e-14)
        E_a = FT(15.0)
    elseif reaction == "Momentum"
        A = FT(5.0e-15)
        E_a = FT(5.0)
    elseif reaction == "tot_Excitation"
        A = FT(1.0e-14)
        E_a = FT(10.0)
    else
        A = FT(1.0e-15)
        E_a = FT(5.0)
    end

    # Calculate rate coefficient
    rrc = A * exp.(-E_a ./ Te_eV)

    # Apply drift velocity enhancement
    u_thermal = sqrt.(Te_eV * RP.config.ee / RP.config.me)
    u_ratio = abs.(ue_para) ./ u_thermal

    # Enhancement factor (simplified)
    enhancement = 1.0 .+ u_ratio.^2 .* gFac

    return rrc .* enhancement
end

"""
    calculate_density_source_terms!(RP::RAPID{FT}) where FT<:AbstractFloat

Calculate the source terms for electron density evolution, including ionization processes.
"""
function calculate_density_source_terms!(RP::RAPID{FT}) where FT<:AbstractFloat
    # Calculate ionization rate based on the method specified in flags
    if RP.flags.Ionz_method == "Townsend_coeff"
        # Compute source by electron avalanche using Townsend coefficient
        # α = 3.88 * p * exp(-95 * p / |E_para|)
        α = @. 3.88 * RP.config.prefilled_gas_pressure *
               exp(-95 * RP.config.prefilled_gas_pressure / abs(RP.fields.E_para_tot))

        # Electron growth rate
        RP.plasma.eGrowth_rate = @. α * abs(RP.plasma.ue_para)
    elseif RP.flags.Ionz_method == "Xsec"
        # Method based on temperature, drift velocity and distribution function
        RRC_iz = get_electron_RRC(RP, RP.eRRCs, :Ionization)

        # Growth rate = density * reaction rate
        RP.plasma.eGrowth_rate = @. RP.plasma.n_H2_gas * RRC_iz
    else
        error("Unknown ionization method: $(RP.flags.Ionz_method)")
    end

    # # Zero out the growth rate outside the wall
    # eGrowth_rate[RP.G.nodes.out_wall_nids] .= 0.0
    # Store the right-hand side source term
    RP.operators.neRHS_src .= RP.plasma.ne .* RP.plasma.eGrowth_rate

    # Update sparse matrix operator for implicit methods if needed
    if RP.flags.Implicit
        # Create diagonal matrix with electron growth rate
        diagnoal_indices = diagind(RP.operators.An_src)
        @. @views RP.operators.An_src[diagnoal_indices] = RP.plasma.eGrowth_rate[:]
    end
end

"""
    calculate_density_diffusion_terms!(RP::RAPID{FT}) where FT<:AbstractFloat

Calculate the diffusion terms for electron density evolution, including constructing
the diffusion operator matrix for implicit time stepping or directly calculating the
diffusion term for explicit time stepping.
"""
function calculate_density_diffusion_terms!(RP::RAPID{FT}) where FT<:AbstractFloat
    if RP.flags.Implicit
        update_∇𝐃∇_operator!(RP)
        RP.operators.neRHS_diffu[:] = RP.operators.∇𝐃∇ * RP.plasma.ne[:]
    else
        # For explicit method, calculate diffusion term directly
        calculate_ne_diffusion_explicit_RHS!(RP)
    end
    return RP
end

"""
    calculate_density_convection_terms!(RP::RAPID{FT}) where FT<:AbstractFloat

Calculate the convection terms for electron density evolution, including constructing
the convection operator matrix for implicit time stepping or directly calculating the
convection term for explicit time stepping.
"""
function calculate_density_convection_terms!(RP::RAPID{FT}) where FT<:AbstractFloat
    if RP.flags.Implicit
        update_Ane_convection_operator!(RP)
        RP.operators.neRHS_convec[:] = RP.operators.An_convec * RP.plasma.ne[:]
    else
        # For explicit method, calculate convection term directly
        calculate_ne_convection_explicit_RHS!(RP)
    end
    return RP
end

"""
    solve_electron_continuity_equation!(RP::RAPID{FT}) where FT<:AbstractFloat

Solve the electron continuity equation to update electron density.
Uses either explicit or implicit time integration based on RP.flags.Implicit.
"""
function solve_electron_continuity_equation!(RP::RAPID{FT}) where FT<:AbstractFloat
    # Alias for readability
    OP = RP.operators
    # Get time step from RP
    dt = RP.dt

    # Store previous density state for transport calculations
    RP.prev_n .= RP.plasma.ne

    if RP.flags.Implicit
        # Implicit method implementation
        # Weight for implicit method (0.0 = fully explicit, 1.0 = fully implicit)
        θ = RP.flags.Implicit_weight

        # Build full RHS with explicit contribution
        @. OP.RHS = RP.plasma.ne + dt * (one(FT) - θ) * (OP.neRHS_diffu + OP.neRHS_convec + OP.neRHS_src)
        # Build LHS operator
        @. OP.A_LHS = OP.II - θ*dt* (OP.∇𝐃∇ + OP.An_convec + OP.An_src)

        # Solve the linear system
        @views RP.plasma.ne[:] = OP.A_LHS \ OP.RHS[:]
    else
        # Explicit method
        @. RP.plasma.ne += dt* (OP.neRHS_diffu + OP.neRHS_convec + OP.neRHS_src)
    end

    return RP
end

"""
    apply_electron_density_boundary_conditions!(RP::RAPID{FT}) where FT<:AbstractFloat

Apply boundary conditions to electron density, including setting density to zero outside the wall
and handling negative densities.
"""
function apply_electron_density_boundary_conditions!(RP::RAPID{FT}) where FT<:AbstractFloat
    # Set density to zero outside wall
    RP.plasma.ne[RP.G.nodes.out_wall_nids] .= 0.0

    # Correct negative densities if enabled
    if RP.flags.negative_n_correction
        RP.plasma.ne[RP.plasma.ne .< 0] .= 0.0
    end

    return RP
end

"""
    calculate_para_grad_of_scalar_F(RP::RAPID{FT}, F::Matrix{FT}; upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Calculate the parallel gradient [∇∥ ≡ b⋅∇] of a scalar field F in the direction of the magnetic field.
Uses either upwind scheme (based on flow velocity) or central differences.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `F::Matrix{FT}`: The scalar field whose parallel gradient is to be calculated
- `upwind::Bool=RP.flags.upwind`: whether to use flow direction to choose appropriate differencing

# Returns
- `Matrix{FT}`: The calculated parallel gradient field

# Notes
- When upwind=true, uses flow direction to choose appropriate differencing
- When upwind=false, uses central differencing for interior points
- Provides better numerical stability for advection-dominated problems when upwind=true
- Matrix indexing is F[i,j] where i is R-index and j is Z-index
"""
function calculate_para_grad_of_scalar_F(RP::RAPID{FT}, F::Matrix{FT}; upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}
    NR, NZ = size(F)
    @assert NR > 1 && NZ > 1 "Grid size must be at least 2x2"

    # Define constants for type stability
    zero_FT = zero(FT)
    half = FT(0.5)
    eps_val = eps(FT)

    # Pre-compute inverse values for faster calculation
    inv_dR = one(FT) / RP.G.dR
    inv_dZ = one(FT) / RP.G.dZ

    # Initialize output array
    para_∇F = zeros(FT, NR, NZ)

    # Calculate parallel gradient for interior points
    if upwind
        # Upwind scheme based on flow velocity direction
        @inbounds for j in 2:NZ-1, i in 2:NR-1

            # R-direction contribution
            if abs(RP.plasma.ueR[i,j]) < eps_val
                # Zero velocity: use central differencing for stability
                para_∇F[i,j] += RP.fields.bR[i,j] * (F[i+1,j] - F[i-1,j]) * (inv_dR * half)
            elseif RP.plasma.ueR[i,j] > zero_FT
                # Positive flow: backward difference (upwind)
                para_∇F[i,j] += RP.fields.bR[i,j] * (F[i,j] - F[i-1,j]) * inv_dR
            else
                # Negative flow: forward difference (upwind)
                para_∇F[i,j] += RP.fields.bR[i,j] * (F[i+1,j] - F[i,j]) * inv_dR
            end

            # Z-direction contribution
            if abs(RP.plasma.ueZ[i,j]) < eps_val
                # Zero velocity: use central differencing for stability
                para_∇F[i,j] += RP.fields.bZ[i,j] * (F[i,j+1] - F[i,j-1]) * (inv_dZ * half)
            elseif RP.plasma.ueZ[i,j] > zero_FT
                # Positive flow: backward difference (upwind)
                para_∇F[i,j] += RP.fields.bZ[i,j] * (F[i,j] - F[i,j-1]) * inv_dZ
            else
                # Negative flow: forward difference (upwind)
                para_∇F[i,j] += RP.fields.bZ[i,j] * (F[i,j+1] - F[i,j]) * inv_dZ
            end
        end
    else
        # Central difference scheme for interior points
        # This is more accurate for smooth solutions but may have stability issues for advection-dominated flows
        @inbounds for j in 2:NZ-1, i in 2:NR-1
            para_∇F[i,j] = RP.fields.bR[i,j] * (F[i+1,j] - F[i-1,j]) * (inv_dR * half) +
                             RP.fields.bZ[i,j] * (F[i,j+1] - F[i,j-1]) * (inv_dZ * half)
        end
    end

    # Handle boundaries with one-sided differences
    # Calculate R derivative contributions
    @inbounds for j in 1:NZ
        # Left boundary: forward difference
        i = 1
        para_∇F[i,j] += RP.fields.bR[i,j] * (F[i+1,j] - F[1,j]) * inv_dR
        i = NR
        # Right boundary: backward difference
        para_∇F[i,j] += RP.fields.bR[i,j] * (F[i,j] - F[i-1,j]) * inv_dR
    end
    # Bottom and Top boundary: central difference
    @inbounds for j in [1, NZ]
        for i in 2:NR-1
            para_∇F[i,j] += RP.fields.bR[i,j] * (F[i+1,j] - F[i-1,j]) * (inv_dR * half)
        end
    end

    # Calculate Z derivative contributions
    @inbounds for i in 1:NR
        # Bottom boundary: forward difference
        j = 1
        para_∇F[i,j] += RP.fields.bZ[i,j] * (F[i,j+1] - F[i,j]) * inv_dZ
        # Top boundary: backward difference
        j = NZ
        para_∇F[i,j] += RP.fields.bZ[i,j] * (F[i,j] - F[i,j-1]) * inv_dZ
    end
    # Left and Right boundary: central difference
    @inbounds for i in [1, NR]
        for j in 2:NZ-1
            para_∇F[i,j] += RP.fields.bZ[i,j] * (F[i,j+1] - F[i,j-1]) * (inv_dZ * half)
        end
    end

    return para_∇F
end


"""
    calculate_grad_of_scalar_F(RP::RAPID{FT}, F::Matrix{FT}; upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}

Calculate the gradient components of a scalar field F in R and Z directions.
Returns gradF_R and gradF_Z as separate matrices.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `F::Matrix{FT}`: The scalar field whose gradient is to be calculated
- `upwind::Bool=RP.flags.upwind`: Whether to use upwind differencing based on velocity field

# Returns
- Tuple of two matrices (gradF_R, gradF_Z): Components of the gradient in R and Z directions

# Notes
- When upwind=true, uses flow velocity direction to choose appropriate differencing scheme
- When upwind=false, uses standard central differencing for interior points with one-sided differences at boundaries
- Provides better numerical stability for advection-dominated problems when upwind=true
- Matrix indexing is F[i,j] where i is R-index and j is Z-index (Julia convention)
  which differs from MATLAB's (j,i) convention
"""
function calculate_grad_of_scalar_F(RP::RAPID{FT}, F::Matrix{FT}; upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}
    NR, NZ = size(F)
    @assert NR > 1 && NZ > 1 "Grid size must be at least 2x2"

    # Define constants for type stability
    zero_FT = zero(FT)
    half = FT(0.5)
    eps_val = eps(FT)

    # Pre-compute inverse values for faster calculation
    inv_dR = one(FT) /  RP.G.dR
    inv_dZ = one(FT) / RP.G.dZ

    # Initialize output arrays
    ∇F_R = zeros(FT, NR, NZ)
    ∇F_Z = zeros(FT, NR, NZ)


    # Calculate gradients for interior points
    if upwind
        # Upwind differencing scheme based on flow velocity
        @inbounds for j in 2:NZ-1, i in 2:NR-1
            # R-direction gradient
            if abs(RP.plasma.ueR[i,j]) < eps_val
                # Zero velocity: use central differencing for stability
                ∇F_R[i,j] = (F[i+1,j] - F[i-1,j]) * (inv_dR * half)
            elseif RP.plasma.ueR[i,j] > zero_FT
                # Positive velocity: backward difference (upwind)
                ∇F_R[i,j] = (F[i,j] - F[i-1,j]) * inv_dR
            else
                # Negative velocity: forward difference (upwind)
                ∇F_R[i,j] = (F[i+1,j] - F[i,j]) * inv_dR
            end

            # Z-direction gradient
            if abs(RP.plasma.ueZ[i,j]) < eps_val
                # Zero velocity: use central differencing for stability
                ∇F_Z[i,j] = (F[i,j+1] - F[i,j-1]) * (inv_dZ * half)
            elseif RP.plasma.ueZ[i,j] > zero_FT
                # Positive velocity: backward difference (upwind)
                ∇F_Z[i,j] = (F[i,j] - F[i,j-1]) * inv_dZ
            else
                # Negative velocity: forward difference (upwind)
                ∇F_Z[i,j] = (F[i,j+1] - F[i,j]) * inv_dZ
            end
        end
    else
        # Standard central differencing scheme
        @inbounds for j in 2:NZ-1, i in 2:NR-1
            # R-direction gradient
            ∇F_R[i,j] = (F[i+1,j] - F[i-1,j]) * (inv_dR * half)
            # Z-direction gradient
            ∇F_Z[i,j] = (F[i,j+1] - F[i,j-1]) * (inv_dZ * half)
        end
    end

    # Handle boundaries with one-sided differences
    # Calculate R derivative contributions
    @inbounds for j in 1:NZ
        # Left boundary: forward difference
        ∇F_R[1,j] = (F[2,j] - F[1,j]) * inv_dR
        # Right boundary: backward difference
        ∇F_R[NR,j] = (F[NR,j] - F[NR-1,j]) * inv_dR
    end
    # Bottom and Top boundary: central difference
    @inbounds for j in [1, NZ]
        for i in 2:NR-1
            ∇F_R[i,j] = (F[i+1,j] - F[i-1,j]) * (inv_dR * half)
        end
    end

    # Calculate Z derivative contributions
    @inbounds for i in 1:NR
        # Bottom boundary: forward difference
        ∇F_Z[i,1] = (F[i,2] - F[i,1]) * inv_dZ
        # Top boundary: backward difference
        ∇F_Z[i,NZ] = (F[i,NZ] - F[i,NZ-1]) * inv_dZ
    end
    # Left and Right boundary: central difference
    @inbounds for i in [1, NR]
        for j in 2:NZ-1
            ∇F_Z[i,j] = (F[i,j+1] - F[i,j-1]) * (inv_dZ * half)
        end
    end

    return ∇F_R, ∇F_Z
end


"""
    calculate_electron_acceleration_by_pressure(RP::RAPID{FT}; num_SM::Int=2) where {FT<:AbstractFloat}

Calculate the electron pressure gradient force acceleration along the magnetic field.
This uses a smoothed density field to improve numerical stability.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `num_SM::Int=2`: Number of smoothing iterations (if 0, no smoothing is applied)

# Returns
- `Matrix{FT}`: The pressure gradient acceleration field (m/s²)

# Notes
- Uses smoothed density field to avoid numerical issues with very low density regions
- Calculates both density gradient and temperature gradient contributions
- Limits the maximum acceleration to maintain numerical stability
- Setting `num_SM=0` bypasses smoothing, which may be desirable for specific use cases
"""
function calculate_electron_acceleration_by_pressure(RP::RAPID{FT}; num_SM::Int=2) where {FT<:AbstractFloat}
    # alias
    cnst = RP.config.constants

    # Smooth the density field to reduce numerical noise (skip if num_SM is 0)
    n_SM = smooth_data_2D(RP.plasma.ne; num_SM, weighting=RP.G.Jacob)
    n_SM[n_SM .< 0] .= zero(FT)

    # Calculate ln(n) gradients along B to avoid division by zero issues with low density
    # Calculate temperature gradients along B
    para_grad_ln_n = calculate_para_grad_of_scalar_F(RP, log.(n_SM))
    para_grad_Te_eV = calculate_para_grad_of_scalar_F(RP, RP.plasma.Te_eV )

    # Combine both terms for total pressure gradient acceleration
    accel_by_pressure = @. (- para_grad_ln_n * RP.plasma.Te_eV * cnst.ee / cnst.me
                            - para_grad_Te_eV * cnst.ee / cnst.me)

    # Handle any NaN or Inf values that might arise
    accel_by_pressure[.!isfinite.(accel_by_pressure)] .= zero(FT)

    return accel_by_pressure
end


"""
    calculate_electron_acceleration_by_convection(RP::RAPID{FT}; num_SM::Int=2) where {FT<:AbstractFloat}

Calculate the electron acceleration due to convection

This function computes the convection term [-(ud⋅∇)ud] in the electron momentum equation.
It uses a smoothed parallel electron velocity field to improve numerical stability.

# Arguments
- `RP::RAPID{FT}`: RAPID simulation object containing plasma and grid data
- `num_SM::Int=0`: Number of smoothing iterations to apply to the velocity field
- `flag_upwind::Bool=RP.flags.upwind`: Whether to use upwind differencing for gradient calculation

# Returns
- Electron acceleration due to convection term

# Implementation Details
1. Smooths the parallel electron velocity field using the Jacobian-weighted smoothing
2. Calculates the gradient of the smoothed velocity field
3. Computes the convection term as -(ueR*∇ud_R + ueZ*∇ud_Z)
"""
function calculate_electron_acceleration_by_convection(RP::RAPID{FT}; num_SM::Int=0, flag_upwind::Bool=RP.flags.upwind) where {FT<:AbstractFloat}
    # alias
    cnst = RP.config.constants

    # Smooth the density field to reduce numerical noise (skip if num_SM is 0)
    ue_para_SM = smooth_data_2D(RP.plasma.ue_para; num_SM, weighting=RP.G.Jacob)

    ∇ud_R, ∇ud_Z = calculate_grad_of_scalar_F(RP, ue_para_SM; upwind=flag_upwind)
    accel_by_convection = @. -(RP.plasma.ueR*∇ud_R + RP.plasma.ueZ*∇ud_Z)

    return accel_by_convection
end
