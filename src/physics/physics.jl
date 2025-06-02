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
       update_ion_heating_powers!,
       calculate_ν_iz!,
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
                accel_para_tilde .+= (one_FT - θu) * (-OP.𝐮∇ * pla.ue_para)
                @. OP.A_LHS += θu * dt * OP.𝐮∇
            end

            # #3: Pressure term [-∇∥(ne*Te)/(me*ne)]
            accel_para_tilde .+= calculate_electron_acceleration_by_pressure(RP)

            # #4: collision drag force  (1-θ)*[-(ν_tot)*ue_para]
            @. accel_para_tilde += (one_FT - θu) * (-tot_coll_freq * pla.ue_para)

            # Add collision frequency to diagonal elements using spdiagm
            OP.A_LHS += @views spdiagm(θu * dt * tot_coll_freq[:])

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
            pla.ue_para .= OP.A_LHS \ OP.RHS
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


function solve_ion_continuity_equation!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Solve the ion continuity equation
    # This function is a placeholder and should be implemented based on the specific model
    # For now, we just return the RAPID object unchanged
    @warn "Ion continuity equation solver not implemented yet" maxlog=1
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
                Rui_ei = @. pla.sptz_fac*(cnst.me/cnst.mi)* pla.ν_ei * (pla.ue_para - pla.ui_para)
            else
                Rui_ei = @. (cnst.me/cnst.mi)*pla.ν_ei * (pla.ue_para - pla.ui_para)
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
    # Update power terms for energy equation
    update_electron_heating_powers!(RP)

    ee = RP.config.constants.ee
    dt = RP.dt
    pla = RP.plasma
    OP = RP.operators
    θimp = RP.flags.Implicit_weight

    # Apply time integration method based on flag
    if RP.flags.Implicit
        if RP.flags.evolve_Te_inWall_only
            @warn "Implicit method for Te_evolve_inWall_only not implemented yet" maxlog=1
        else
            # Calculate RHS
            # ePowers_tilde = ePowers already known at crruent time t
            ePowers_tilde = pla.ePowers.tot - θimp * (pla.ePowers.diffu + pla.ePowers.conv)
            OP.RHS .= pla.Te_eV + (FT(2.0)/FT(3.0)*dt*ePowers_tilde / ee)

            # Calculate LHS
            div_u = calculate_divergence(RP.G, pla.ueR, pla.ueZ)
            OP.A_LHS = OP.II - dt*θimp*(OP.∇𝐃∇ - OP.𝐮∇ + spdiagm(div_u[:]/FT(3.0)))

            # Solve the linear system
            pla.Te_eV .= OP.A_LHS \ OP.RHS
        end
    else
        # Explicit method (forward Euler)
        @. pla.Te_eV += (FT(2.0)/FT(3.0)) * pla.ePowers.tot * dt / ee;
    end

    # Apply temperature limits
    @. pla.Te_eV = max(pla.Te_eV, RP.config.min_Te)
    @. pla.Te_eV = min(pla.Te_eV, RP.config.max_Te)

    return RP
end

"""
    update_Ti!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the ion temperature based on energy balance equation.

This function evolves the ion temperature by solving the ion energy equation:
(3/2)*∂Ti/∂t = P_ion_heating

where P_ion_heating includes atomic collision heating and equilibration with electrons.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- `RP`: The updated RAPID object

# Notes
- Uses forward Euler time integration
- Calls update_ion_heating_powers! to compute heating terms
- Applies temperature limits after update
- Sets ion temperature equal to electron temperature if Ti_evolve is disabled
"""
function update_Ti!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Update ion heating power terms
    update_ion_heating_powers!(RP)

    ee = RP.config.constants.ee
    dt = RP.dt
    pla = RP.plasma

    # Update ion temperature using forward Euler
    @. pla.Ti_eV += (FT(2.0)/FT(3.0)) * pla.iPowers.tot * dt / (ee)

    # Apply temperature limits (same as electrons for simplicity)
    @. pla.Ti_eV = max(pla.Ti_eV, RP.config.min_Te)
    @. pla.Ti_eV = min(pla.Ti_eV, RP.config.max_Te)

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
    OP = RP.operators

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
        # P_diffu = 1.5*∇𝐃∇*Te
        if RP.flags.evolve_Te_inWall_only
            @warn "Include_Te_diffu_term not implemented for evolve_Te_inWall_only" maxlog=1
            # TODO:Need to implement A_Dturb_diffu_reflective instead of using ∇𝐃∇
            # obj.ePowers.diffu(obj.in_Wall_idx) =  obj.ee*(1.5*obj.A_Dturb_diffu_reflective)*obj.Te_eV(obj.in_Wall_idx);
        else
            ePowers.diffu .= ee*FT(1.5)*compute_∇𝐃∇f_directly(RP, RP.plasma.Te_eV)
        end
    end

    # If convection term is included in temperature equation
    if RP.flags.Include_Te_convec_term
        if RP.flags.evolve_Te_inWall_only
            @warn "Include_Te_convec_term not implemented for evolve_Te_inWall_only" maxlog=1
            # obj.A_Te_conv = obj.Construct_A_Te_convec_only_IN_nodes(obj.Jacob,obj.ueR,obj.ueZ,obj.Flag);
            # obj.ePowers.conv(obj.in_Wall_idx) = obj.ee*obj.A_Te_conv*obj.Te_eV(obj.in_Wall_idx);
        else
            ePowers.conv .= ee*(
                    -FT(1.5)*compute_∇f𝐮_directly(RP, pla.Te_eV)
                    .+FT(0.5)* pla.Te_eV .* calculate_divergence(RP.G, pla.ueR, pla.ueZ)
            )
        end
    end

    if RP.flags.Include_heat_flux_term
        # NOTE: Assumption: 𝐪 ≈ p*𝐮 (heat flux is about in the order of pressure*velocity)
        # Pcond = Pheat = -∇⋅(Te * 𝐮) - Te*𝐮⋅∇(ln_n)
        ePowers.heat .= ee*(
            - compute_∇f𝐮_directly(RP, pla.Te_eV)
            - pla.Te_eV * compute_𝐮∇f_directly(RP, log.(pla.ne))
        )
    end


    if RP.flags.Atomic_Collision
        # Get reaction rate coefficients for momentum transfer
        RRC_mom = get_electron_RRC(RP, :Momentum)

        # Calculate velocity magnitudes for drag forces
        ue_mag_sq = @. pla.ueR^FT(2.0) .+ pla.ueϕ^FT(2.0) .+ pla.ueZ^FT(2.0)
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
    out_wall_nids = RP.G.nodes.out_wall_nids
    if !isempty(out_wall_nids)
        @views ePowers.tot[out_wall_nids] .= zero_FT
        @views ePowers.diffu[out_wall_nids] .= zero_FT
        @views ePowers.conv[out_wall_nids] .= zero_FT
        @views ePowers.drag[out_wall_nids] .= zero_FT
        @views ePowers.dilution[out_wall_nids] .= zero_FT
        @views ePowers.iz[out_wall_nids] .= zero_FT
        @views ePowers.exc[out_wall_nids] .= zero_FT
        @views ePowers.equi[out_wall_nids] .= zero_FT
        @views ePowers.heat[out_wall_nids] .= zero_FT
    end

    return RP
end

"""
    update_ion_heating_powers!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update ion heating power components for ion energy equation.

This function calculates the ion power sources and sinks based on the MATLAB
`Cal_Ion_Heating_Powers` function, including:
- Atomic collision power (from elastic, charge exchange, and ionization)
- Equilibration power with electrons (if Coulomb collisions enabled)

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state

# Returns
- `RP`: The updated RAPID object

# Notes
The power calculation includes:
- Energy change from atomic collisions: 0.5*mi*ui_mag_sq - 1.5*(Ti-T_gas)*ee
- Effective collision frequency: n_H2_gas*(0.5*elastic + charge_exchange + Zeff*ionization)
- Atomic power: collision_frequency * energy_change
- Equilibration power: matches electron equilibration power if Coulomb collisions enabled
- Total power: atomic + equilibration
- Sets power to zero outside wall boundaries
"""
function update_ion_heating_powers!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Define type-stable constants
    zero_FT = zero(FT)

    # TODO: add convection and diffusion terms for ions if needed

    # Extract physical constants
    @unpack ee, mi, me = RP.config.constants

    # Alias common objects for readability
    pla = RP.plasma
    iPowers = RP.plasma.iPowers

    # Reset ion power arrays to zero (precaution against accumulation)
    iPowers.atomic .= zero_FT
    iPowers.equi .= zero_FT

    if RP.flags.Atomic_Collision
        # Get reaction rate coefficients for ion reactions
        if RP.flags.src
            # Get electron ionization RRC (note: electron RRC is used for ionization)
            eRRC_iz = get_electron_RRC(RP, :Ionization)
        else
            eRRC_iz = zero_FT
        end

        # Get ion reaction rate coefficients
        iRRC_cx = get_H2_ion_RRC(RP, :Charge_Exchange)
        iRRC_elastic = get_H2_ion_RRC(RP, :Elastic)

        # Calculate ion velocity magnitude squared
        ui_mag_sq = @. pla.uiR^FT(2.0) + pla.uiϕ^FT(2.0) + pla.uiZ^FT(2.0)

        # Calculate average energy change from atomic collisions
        # Energy balance: kinetic energy loss minus thermal energy change
        avg_erg_change_by_atomic_collision = @. (
            FT(0.5) * mi * ui_mag_sq - FT(1.5) * (pla.Ti_eV - pla.T_gas_eV) * ee
        )

        # Calculate effective atomic collision frequency
        # Note: 0.5 factor for elastic collisions (momentum transfer efficiency)
        eff_atomic_coll_freq = @. pla.n_H2_gas * (
            FT(0.5) * iRRC_elastic + iRRC_cx + pla.Zeff * eRRC_iz
        )

        # Calculate atomic power: collision frequency times energy change
        @. iPowers.atomic = eff_atomic_coll_freq * avg_erg_change_by_atomic_collision
    end

    # Handle equilibration power with electrons
    if RP.flags.Coulomb_Collision
        @. iPowers.equi = (FT(2.0)*(mi*me/(mi+me)^2)) * FT(1.5) * ee * (pla.Te_eV - pla.Ti_eV) * pla.ν_ei
    end

    # Calculate total ion heating power
    @. iPowers.tot = iPowers.atomic + iPowers.equi

    # Set power to zero outside wall boundaries
    out_wall_nids = RP.G.nodes.out_wall_nids
    if !isempty(out_wall_nids)
        @views iPowers.tot[out_wall_nids] .= zero_FT
        @views iPowers.atomic[out_wall_nids] .= zero_FT
        @views iPowers.equi[out_wall_nids] .= zero_FT
    end

    return RP
end

"""
    calculate_ν_iz!(RP::RAPID{FT}) where FT<:AbstractFloat

Calculate the ionization frequency ν_iz = n_H2_gas * <σ_iz * v_e>
"""
function calculate_ν_iz!(RP::RAPID{FT}) where FT<:AbstractFloat
    # Calculate ionization rate based on the method specified in flags
    if RP.flags.Ionz_method == "Townsend_coeff"
        # Compute source by electron avalanche using Townsend coefficient
        # α = 3.88 * p * exp(-95 * p / |E_para|)
        α = @. 3.88 * RP.config.prefilled_gas_pressure *
               exp(-95 * RP.config.prefilled_gas_pressure / abs(RP.fields.E_para_tot))

        # Electron ionization frequency
        RP.plasma.ν_iz = @. α * abs(RP.plasma.ue_para)
    elseif RP.flags.Ionz_method == "Xsec"
        # Ionization frequency = (gas density) * (<σ_iz*v>)
        eRRC_iz = get_electron_RRC(RP, RP.eRRCs, :Ionization)
        @. RP.plasma.ν_iz = RP.plasma.n_H2_gas * eRRC_iz
    else
        error("Unknown ionization method: $(RP.flags.Ionz_method)")
    end

    # Zero out the ionization frequency outside the wall
    RP.plasma.ν_iz[RP.G.nodes.out_wall_nids] .= 0.0

    # Update sparse matrix operator for implicit methods if needed
    if RP.flags.Implicit
        RP.operators.ν_iz .= @views spdiagm(RP.plasma.ν_iz[:])
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
    op = RP.operators
    # Get time step from RP
    dt = RP.dt
    pla = RP.plasma

    # Store previous density state for transport calculations
    RP.prev_n .= RP.plasma.ne

    # Calculate source terms for electron density
    fill!(op.RHS, zero(FT))  # Reset RHS to zero
    if RP.flags.src
        # ne * ν_iz = ne * nH2_gas * eRRC_iz
        calculate_ν_iz!(RP)
        op.RHS += pla.ne .* pla.ν_iz
    end

    if RP.flags.diffu
        # ∇⋅𝐃⋅∇n
        update_∇𝐃∇_operator!(RP)
        op.RHS .+= compute_∇𝐃∇f_directly(RP, pla.ne)
    end
    if RP.flags.convec
        # -∇⋅(n 𝐮)
        update_∇𝐮_operator!(RP)
        op.RHS .+= -compute_∇f𝐮_directly(RP, pla.ne)
    end

    # update electron density
    if RP.flags.Implicit
        # Implicit method implementation
        # Weight for implicit method (0.0 = fully explicit, 1.0 = fully implicit)
        θn = RP.flags.Implicit_weight

        # Build full RHS with explicit contribution
        @. op.RHS = pla.ne + dt * (one(FT) - θn) * op.RHS

        # Build LHS operator
        @. op.A_LHS = op.II - θn*dt* (op.∇𝐃∇ - op.∇𝐮 + op.ν_iz)

        # Solve the linear system
        @views pla.ne[:] = op.A_LHS \ op.RHS[:]
    else
        @. RP.plasma.ne += dt * op.RHS
    end

    return RP
end

"""
    treat_electron_outside_wall!(RP::RAPID{FT}) where FT<:AbstractFloat

Apply boundary conditions for electrons outside the wall and track particle sources/losses.

This function performs three main operations:
1. **Track ionization sources**: Calculates electrons generated by ionization and adds to cumulative tracking
2. **Apply wall boundary conditions**:
- Sets electron density to zero outside the wall
- Sets electron temperature to room temperature outside the wall
3. **Correct negative densities**: Optionally removes negative densities and counts them as losses

# Arguments
- `RP::RAPID{FT}`: RAPID simulation object containing plasma state and geometry

# Details
- Uses implicit weighting when `RP.flags.Implicit = true` for ionization source calculation
- Tracks cumulative particle numbers in both 1D (volume-integrated) and 2D (spatially-resolved) formats
- Applies negative density correction when `RP.flags.negative_n_correction = true`
- Updates particle number tracker (`Ntracker`) for diagnostic purposes

# Returns
- `RP`: Modified RAPID object with updated electron density and particle tracking
"""
function treat_electron_outside_wall!(RP::RAPID{FT}) where FT<:AbstractFloat
    # Estimate the number of electrons generated by ionization
    if RP.flags.Implicit
        θimp = RP.flags.Implicit_weight
        effective_ne = @. (FT(1.0) - θimp) * RP.prev_n + θimp * RP.plasma.ne
        Ne_iz = @. RP.plasma.ν_iz * effective_ne * RP.G.inVol2D * RP.dt
    else
        Ne_iz = @. RP.plasma.ν_iz * RP.plasma.ne * RP.G.inVol2D * RP.dt
    end

    # Estimate electron loss outside the wall
    # TODO: How to accurately define the volume outside/on the wall?
    out_wall_nids = RP.G.nodes.out_wall_nids
    Ne_loss = @. RP.plasma.ne[out_wall_nids] * FT(2.0 * pi) * RP.G.Jacob[out_wall_nids] * RP.G.dR * RP.G.dZ

    # Track changes in number of electrons
    Ntracker = RP.diagnostics.Ntracker

    Ntracker.cum0D_Ne_src += sum(Ne_iz)
    @. Ntracker.cum2D_Ne_src += Ne_iz

    Ntracker.cum0D_Ne_loss += sum(Ne_loss)
    @. Ntracker.cum2D_Ne_loss[out_wall_nids] += Ne_loss

    # Set electron density to zero outside the wall
    RP.plasma.ne[out_wall_nids] .= 0.0

    # Set electron temperature to room temperature outside the wall
    RP.plasma.Te_eV[out_wall_nids] .= RP.config.constants.room_T_eV

    # Correct negative densities if enabled
    if RP.flags.negative_n_correction
        neg_n_idx = findall(RP.plasma.ne .< 0)
        if !isempty(neg_n_idx)
            Ne_loss = @. RP.plasma.ne[neg_n_idx] * FT(2.0 * pi) * RP.G.Jacob[neg_n_idx] * RP.G.dR * RP.G.dZ
            Ntracker.cum0D_Ne_loss += sum(Ne_loss)
            @. @views Ntracker.cum2D_Ne_loss[neg_n_idx] += Ne_loss
            RP.plasma.ne[neg_n_idx] .= 0.0
        end
    end

    return RP
end



"""
    treat_ion_outside_wall!(RP::RAPID{FT}) where FT<:AbstractFloat

Applies boundary conditions for ions outside the computational domain wall. Tracks ionization
source/loss, sets boundary conditions (zero density, room temperature), generates secondary
electrons from ion wall impacts, and optionally corrects negative densities.

# Arguments
- `RP`: RAPID simulation object containing plasma state and configuration

# Returns
- Modified `RP` object with updated ion density and temperature boundary conditions
"""
function treat_ion_outside_wall!(RP::RAPID{FT}) where FT<:AbstractFloat
    # Estimate the number of ions generated by ionization
    # NOTE: ν_iz must be multiplied by the electron density to get the ionization rate
    # In other words, Ne_iz = Ni_iz
    Ni_iz = @. RP.plasma.ν_iz * RP.plasma.ne * RP.G.inVol2D * RP.dt

    # Estimate electron loss outside the wall
    # TODO: How to accurately define the volume outside/on the wall?
    out_wall_nids = RP.G.nodes.out_wall_nids
    Ni_loss = @. RP.plasma.ni[out_wall_nids] * FT(2.0 * pi) * RP.G.Jacob[out_wall_nids] * RP.G.dR * RP.G.dZ

    # Track changes in number of electrons
    Ntracker = RP.diagnostics.Ntracker

    Ntracker.cum0D_Ni_src += sum(Ni_iz)
    @. Ntracker.cum2D_Ni_src += Ni_iz

    Ntracker.cum0D_Ni_loss += sum(Ni_loss)
    @. Ntracker.cum2D_Ni_loss[out_wall_nids] += Ni_loss

    # Secondary electron generated by ion impacts on wall
    # TODO: needs to improve this part (somehow this should generate them inside wall)
    if RP.flags.secondary_electron
        RP.plasma.ne[out_wall_nids] .+= RP.flags.γ_2nd_electron * RP.plasma.ni[out_wall_nids]
    end

    # Set ion density to zero outside the wall
    RP.plasma.ni[out_wall_nids] .= 0.0

    # Set electron temperature to room temperature outside the wall
    RP.plasma.Ti_eV[out_wall_nids] .= RP.config.constants.room_T_eV

    # Correct negative densities if enabled
    if RP.flags.negative_n_correction
        neg_n_idx = findall(RP.plasma.ni .< 0)
        if !isempty(neg_n_idx)
            Ni_loss = @. RP.plasma.ni[neg_n_idx] * FT(2.0 * pi) * RP.G.Jacob[neg_n_idx] * RP.G.dR * RP.G.dZ
            Ntracker.cum0D_Ni_loss += sum(Ni_loss)
            @. @views Ntracker.cum2D_Ni_loss[neg_n_idx] += Ni_loss
            RP.plasma.ni[neg_n_idx] .= 0.0
        end
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
