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
       update_power_terms!,
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
                @views accel_para_tilde[:] .+= (one_FT - θu) * (-OP.A_𝐮∇ * pla.ue_para[:])
                @. OP.A_LHS += θu * dt * OP.A_𝐮∇
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

Update the electron temperature.
"""
function update_Te!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "update_Te! not fully implemented yet"

    # Update electron temperature based on power balance
    update_power_terms!(RP)

    # Energy equation: 3/2 n ∂T/∂t = P_total
    # Simplified implementation: forward Euler
    # dT/dt = (2/3) * P_total / n

    # Avoid division by zero
    n_min = FT(1.0e6)
    n_safe = copy(RP.plasma.ne)
    n_safe[n_safe .< n_min] .= n_min

    # Temperature change rate
    dTdt = (FT(2.0)/FT(3.0)) * RP.plasma.ePowers.tot ./ n_safe

    # Update temperature
    RP.plasma.Te_eV .+= dTdt * RP.dt

    # Apply temperature limits
    RP.plasma.Te_eV .= max.(RP.plasma.Te_eV, RP.config.min_Te)
    RP.plasma.Te_eV .= min.(RP.plasma.Te_eV, RP.config.max_Te)

    # # Zero temperature outside wall
    # RP.plasma.Te_eV[RP.G.nodes.out_wall_nids] .= RP.config.min_Te

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
    update_power_terms!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update power terms for electron and ion energy equations.
"""
function update_power_terms!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "update_power_terms! not fully implemented yet"

    # Initialize all power terms to zero
    RP.plasma.ePowers.diffu .= zeros(FT, RP.G.NR, RP.G.NZ)
    RP.plasma.ePowers.conv .= zeros(FT, RP.G.NR, RP.G.NZ)
    RP.plasma.ePowers.heat .= zeros(FT, RP.G.NR, RP.G.NZ)
    RP.plasma.ePowers.drag .= zeros(FT, RP.G.NR, RP.G.NZ)
    RP.plasma.ePowers.equi .= zeros(FT, RP.G.NR, RP.G.NZ)
    RP.plasma.ePowers.iz .= zeros(FT, RP.G.NR, RP.G.NZ)
    RP.plasma.ePowers.exc .= zeros(FT, RP.G.NR, RP.G.NZ)
    RP.plasma.ePowers.dilution .= zeros(FT, RP.G.NR, RP.G.NZ)

    RP.plasma.iPowers.atomic .= zeros(FT, RP.G.NR, RP.G.NZ)
    RP.plasma.iPowers.equi .= zeros(FT, RP.G.NR, RP.G.NZ)

    # Calculate ohmic heating (simplified)
    # P_ohmic = j·E = σ·E²
    qe = RP.config.ee
    ohmic_power = qe * RP.plasma.ne .* RP.plasma.ue_para .* RP.fields.E_para_tot

    # Add to heating power
    RP.plasma.ePowers.heat .= ohmic_power

    # Add electron-ion equilibration if Coulomb collisions are enabled
    if RP.flags.Coulomb_Collision
        # Electron-ion energy equilibration rate (simplified)
        # P_ei = 3 m_e/m_i · n_e · ν_ei · (T_i - T_e)
        mass_ratio = RP.config.me / RP.config.mi
        equi_power = 3.0 * mass_ratio * RP.plasma.ne .* RP.plasma.ν_ei .*
                    (RP.plasma.Ti_eV .- RP.plasma.Te_eV) * RP.config.ee

        RP.plasma.ePowers.equi .= equi_power
        RP.plasma.iPowers.equi .= -equi_power
    end

    # Calculate energy loss due to excitation and ionization
    if RP.flags.src
        # In real implementation, would use reaction rate data
        # Simplified placeholder
        iz_cost = FT(15.0) # Ionization cost in eV
        exc_cost = FT(5.0) # Excitation cost in eV

        # Estimate ionization and excitation rates
        # Simplified implementation
        iz_rate = RP.plasma.eGrowth_rate
        exc_rate = iz_rate * FT(3.0) # Assume excitation rate is 3x ionization rate

        # Calculate power loss
        RP.plasma.ePowers.iz .= -iz_cost * iz_rate * RP.config.ee
        RP.plasma.ePowers.exc .= -exc_cost * exc_rate * RP.config.ee
    end

    # Calculate total power
    RP.plasma.ePowers.tot .= (
        RP.plasma.ePowers.diffu .+
        RP.plasma.ePowers.conv .+
        RP.plasma.ePowers.heat .+
        RP.plasma.ePowers.drag .+
        RP.plasma.ePowers.equi .+
        RP.plasma.ePowers.iz .+
        RP.plasma.ePowers.exc .+
        RP.plasma.ePowers.dilution
    )

    RP.plasma.iPowers.tot .= RP.plasma.iPowers.atomic .+ RP.plasma.iPowers.equi

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
        RP.operators.neRHS_diffu[:] = RP.operators.An_diffu * RP.plasma.ne[:]
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
        calculate_ne_convection_explicit_RHS(RP)
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
        @. OP.A_LHS = OP.II - θ*dt* (OP.An_diffu + OP.An_convec + OP.An_src)

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

Calculate the electron acceleration due to convection along the magnetic field.

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

    # Calculate ln(n) gradients along B to avoid division by zero issues with low density
    # Calculate temperature gradients along B
    ∇ud_R, ∇ud_Z = calculate_grad_of_scalar_F(RP, ue_para_SM; upwind=flag_upwind)
    accel_by_convection = @. -(RP.plasma.ueR*∇ud_R + RP.plasma.ueZ*∇ud_Z)

    return accel_by_convection
end
