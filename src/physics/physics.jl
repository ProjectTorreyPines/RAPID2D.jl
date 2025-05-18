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
       apply_electron_density_boundary_conditions!

"""
    update_ue_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the parallel electron velocity.
"""
function update_ue_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}
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
        coll_freq = @. RP.plasma.n_H2_gas * RRC_mom

        # Calculate drift velocity from balance of electric field and collisions
        @. RP.plasma.ue_para = qe * RP.fields.E_para_tot / (RP.config.me * coll_freq)

    elseif RP.flags.ud_method == "Xsec"
        # Full cross section model with collisions

        # Get reaction rate coefficients for both ionization and momentum transfer
        RRC_iz = get_electron_RRC(RP, RP.eRRCs, :Ionization)
        RRC_mom = get_electron_RRC(RP, RP.eRRCs, :Momentum)

        # Calculate collision frequency from neutrals
        coll_freq = @. RP.plasma.n_H2_gas * (RRC_mom + RRC_iz)

        # Add Coulomb collisions if enabled
        if RP.flags.Coulomb_Collision
            if RP.flags.Spitzer_Resistivity
                mom_eff_nu_ei = @. RP.plasma.sptz_fac * RP.plasma.ν_ei
            else
                mom_eff_nu_ei = RP.plasma.ν_ei
            end
            coll_freq .+= mom_eff_nu_ei
        end

        # Ensure no NaN values in collision frequency
        replace!(coll_freq, NaN => 0.0)

        # Consider pressure gradient, convection, and electric field
        accel_para_tilde = zeros(FT, size(RP.plasma.ue_para))

        # Add pressure gradient acceleration if needed
        if RP.flags.Include_ud_convec_term
            # Calculate pressure gradient acceleration (simplified)
            accel_para_tilde .+= calculate_pressure_acceleration(RP)

            # Add convection acceleration
            accel_para_tilde .+= calculate_convection_acceleration(RP)
        end

        # Add electric field acceleration
        qe = -RP.config.ee
        @. accel_para_tilde += qe * RP.fields.E_para_tot / RP.config.me

        # Add Coulomb collision effects with ions
        if RP.flags.Coulomb_Collision
            @. accel_para_tilde += mom_eff_nu_ei * RP.plasma.ui_para
        end

        # Apply implicit time integration
        impFac = 1.0  # Backward Euler
        @. RP.plasma.ue_para = (RP.plasma.ue_para * (1 - (1-impFac) * coll_freq * RP.dt) +
                               accel_para_tilde * RP.dt) / (1 + impFac * coll_freq * RP.dt)

        # Zero out velocity outside wall
        RP.plasma.ue_para[RP.G.nodes.out_wall_nids] .= 0.0
    else
        error("Unknown electron drift velocity method: $(RP.flags.ud_method)")
    end
end

"""
    update_ui_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the parallel ion velocity.
"""
function update_ui_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    if RP.flags.ud_method == "Xsec"
        # Use cross sections for ion velocity update

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
        eff_atomic_coll_freq = @. RP.plasma.n_H2_gas * (
            0.5 * iRRC_elastic + iRRC_cx + RP.plasma.Zeff * eRRC_iz
        )

        # Fix any NaN values
        replace!(eff_atomic_coll_freq, NaN => 0.0)

        # Calculate acceleration from electric field
        qi = RP.config.ee

        # Apply backward Euler time integration
        th = 1.0  # Backward Euler
        @. RP.plasma.ui_para = (RP.plasma.ui_para * (1 - (1-th) * RP.dt * eff_atomic_coll_freq) +
                              RP.dt * qi * RP.fields.E_para_tot / RP.config.mi) /
                              (1 + th * RP.dt * eff_atomic_coll_freq)

        # Add electron-ion momentum transfer effect
        if RP.flags.Coulomb_Collision
            Rui_ei = @. -(RP.config.me / RP.config.mi) * RP.plasma.Zeff * RP.plasma.ν_ei *
                     (RP.plasma.ue_para - RP.plasma.ui_para)
            RP.plasma.ui_para .+= RP.dt * Rui_ei
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

    # Zero temperature outside wall
    RP.plasma.Te_eV[RP.G.nodes.out_wall_nids] .= RP.config.min_Te

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

    # Zero temperature outside wall
    RP.plasma.Ti_eV[RP.G.nodes.out_wall_nids] .= RP.config.min_Te

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
        eGrowth_rate = @. α * abs(RP.plasma.ue_para)
    elseif RP.flags.Ionz_method == "Xsec"
        # Method based on temperature, drift velocity and distribution function
        RRC_iz = get_electron_RRC(RP, RP.eRRCs, :Ionization)

        # Growth rate = density * reaction rate
        eGrowth_rate = @. RP.plasma.n_H2_gas * RRC_iz
    else
        error("Unknown ionization method: $(RP.flags.Ionz_method)")
    end

    # Zero out the growth rate outside the wall
    eGrowth_rate[RP.G.nodes.out_wall_nids] .= 0.0

    # Store the right-hand side source term
    RP.operators.neRHS_src .= RP.plasma.ne .* eGrowth_rate

    # Update sparse matrix operator for implicit methods if needed
    if RP.flags.Implicit
        # Create diagonal matrix with electron growth rate
        RP.operators.An_src = spdiagm(0 => eGrowth_rate[:])
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
        update_diffusion_operator!(RP)
        RP.operators.neRHS_diffu[:] = RP.operators.An_diffu * RP.plasma.ne[:]
    else
        # For explicit method, calculate diffusion term directly
        calculate_diffusion_term!(RP)
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
        update_convection_operator!(RP)
        RP.operators.neRHS_convec[:] = RP.operators.An_convec * RP.plasma.ne[:]
    else
        # For explicit method, calculate convection term directly
        calculate_convection_term!(RP)
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
    if RP.flags.neg_n_correction
        RP.plasma.ne[RP.plasma.ne .< 0] .= 0.0
    end

    return RP
end

"""
    calculate_para_grad_of_scalar_F(RP::RAPID{FT}, F::Matrix{FT}) where {FT<:AbstractFloat}

Calculate the parallel gradient of a scalar field F in the direction of the magnetic field.
Uses either upwind scheme (based on flow velocity) or central differences.

# Arguments
- `RP::RAPID{FT}`: The RAPID object containing simulation state
- `F::Matrix{FT}`: The scalar field whose parallel gradient is to be calculated

# Returns
- `Matrix{FT}`: The calculated parallel gradient field

# Notes
- When upwind=true, uses flow direction to choose appropriate differencing
- When upwind=false, uses central differencing for interior points
- Provides better numerical stability for advection-dominated problems when upwind=true
- Matrix indexing is F[i,j] where i is R-index and j is Z-index
"""
function calculate_para_grad_of_scalar_F(RP::RAPID{FT}, F::Matrix{FT}) where {FT<:AbstractFloat}
    # Define constants for type stability
    zero_FT = zero(FT)
    half = FT(0.5)
    eps_val = eps(FT)

    # Get grid dimensions and spacing
    dR = RP.G.dR
    dZ = RP.G.dZ
    NR, NZ = size(F)

    # Pre-compute inverse values for faster calculation
    inv_dR = one(FT) / dR
    inv_dZ = one(FT) / dZ

    # Initialize output array
    para_grad_F = zeros(FT, NR, NZ)

    if RP.flags.upwind
        # Upwind scheme based on flow velocity direction
        @inbounds for j in 2:NZ-1, i in 2:NR-1

            # R-direction contribution
            if abs(RP.plasma.ueR[i,j]) < eps_val
                # Zero velocity: use central differencing for stability
                para_grad_F[i,j] += RP.fields.bR[i,j] * (F[i+1,j] - F[i-1,j]) * (inv_dR * half)
            elseif RP.plasma.ueR[i,j] > zero_FT
                # Positive flow: backward difference (upwind)
                para_grad_F[i,j] += RP.fields.bR[i,j] * (F[i,j] - F[i-1,j]) * inv_dR
            else
                # Negative flow: forward difference (upwind)
                para_grad_F[i,j] += RP.fields.bR[i,j] * (F[i+1,j] - F[i,j]) * inv_dR
            end

            # Z-direction contribution
            if abs(RP.plasma.ueZ[i,j]) < eps_val
                # Zero velocity: use central differencing for stability
                para_grad_F[i,j] += RP.fields.bZ[i,j] * (F[i,j+1] - F[i,j-1]) * (inv_dZ * half)
            elseif RP.plasma.ueZ[i,j] > zero_FT
                # Positive flow: backward difference (upwind)
                para_grad_F[i,j] += RP.fields.bZ[i,j] * (F[i,j] - F[i,j-1]) * inv_dZ
            else
                # Negative flow: forward difference (upwind)
                para_grad_F[i,j] += RP.fields.bZ[i,j] * (F[i,j+1] - F[i,j]) * inv_dZ
            end
        end
    else
        # Central difference scheme for interior points
        # This is more accurate for smooth solutions but may have stability issues for advection-dominated flows
        @inbounds for j in 2:NZ-1, i in 2:NR-1
            para_grad_F[i,j] = RP.fields.bR[i,j] * (F[i+1,j] - F[i-1,j]) * (inv_dR * half) +
                             RP.fields.bZ[i,j] * (F[i,j+1] - F[i,j-1]) * (inv_dZ * half)
        end
    end

    return para_grad_F
end
