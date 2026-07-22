"""
Physics module for RAPID2D.

Contains functions related to plasma physics models, including:
- Electron and ion dynamics
- Collisions
- Reaction rates
- Power balance
"""

using TimerOutputs

# Use the global timer from the main module

# Export public functions
export update_ue_para!,
    update_ui_para!,
    update_Te!,
    update_Ti!,
    update_coulomb_collision_parameters!,
    update_electron_heating_powers!,
    update_ion_heating_powers!,
    calculate_ν_en_iz!,
    solve_electron_continuity_equation!,
    apply_electron_density_boundary_conditions!,
    calculate_para_grad_of_scalar_F,
    calculate_grad_of_scalar_F,
    calculate_electron_acceleration_by_pressure,
    calculate_electron_acceleration_by_convection,
    update_uMHD_by_global_JxB_force!,
    combine_Au_and_ΔGS_sparse_matrices,
    solve_combined_momentum_Ampere_equations_with_coils!

"""
    update_ue_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the parallel electron velocity.
"""
function update_ue_para!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_ue_para!" begin
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
            @unpack qe, me = RP.config.constants
            dt = RP.dt

            pla = RP.plasma
            F = RP.fields

            # Define sum of collision frequencies of ionization and momentum reactions
            ν_iz_mom = @. pla.ν_en_iz + pla.ν_en_mom

            # Always use backward Euler for ue_para (θu=1.0) for better saturation
            # but keep the formula structure compatible with variable θ_imp
            # θu = RP.flags.Implicit_weight
            θu = one_FT

            # Calculate Rue_ei (electron-ion momentum exchange rate) - first part (n-th step)
            if RP.flags.Coulomb_Collision
                @. pla.Rue_ei = pla.ν_ei_eff * (pla.ui_para - (one_FT - θu) * pla.ue_para)
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
                    accel_para_tilde .+= (one_FT - θu) * (-OP.𝐮∇ * pla.ue_para)
                    @. OP.A_LHS += θu * dt * OP.𝐮∇
                end

                # #3: Pressure term [-∇∥(ne*Te)/(me*ne)]
                if RP.flags.Include_ud_pressure_term
                    accel_para_tilde .+= calculate_electron_acceleration_by_pressure(RP)
                end


                # #4: collision drag force  (1-θ)*[-(ν_iz + ν_mom)*ue_para]
                @. accel_para_tilde += (one_FT - θu) * (-ν_iz_mom * pla.ue_para)

                # Add collision frequency to diagonal elements using spdiagm
                OP.A_LHS += @views spdiagm(θu * dt * ν_iz_mom[:])

                # #5: momentum source from electron-ion collision [+sptz_fac*νei*ui_para]
                @. accel_para_tilde += (pla.ν_ei_eff * pla.ui_para)

                # #6: turbulent Diffusive term by ExB mixing
                if RP.flags.Include_ud_diffu_term
                    accel_para_tilde .+= (one_FT - θu) * (OP.∇𝐃∇ * pla.ue_para)
                    @. OP.A_LHS -= θu * dt * OP.∇𝐃∇
                end

                # Set-up the RHS
                @. OP.RHS = pla.ue_para + dt * accel_para_tilde

                # Solve the momentum equation
                @timeit RAPID_TIMER "ue_para LinearSolve" begin
                    pla.ue_para .= OP.A_LHS \ OP.RHS
                end
            else

                inv_factor = @. one_FT / (one_FT + θu * ν_iz_mom * dt)
                @. pla.ue_para = inv_factor * (
                    pla.ue_para * (one_FT - (one_FT - θu) * dt * ν_iz_mom)
                        + dt * (qe * F.E_para_tot / me + pla.ν_ei_eff * pla.ui_para)
                )

                # Add pressure and convection terms in the same way as MATLAB
                if RP.flags.Include_ud_pressure_term
                    accel_by_pressure = calculate_electron_acceleration_by_pressure(RP)
                    @. pla.ue_para += inv_factor * dt * (accel_by_pressure)
                end

                if RP.flags.Include_ud_convec_term
                    accel_by_grad_ud = calculate_electron_acceleration_by_convection(RP)
                    @. pla.ue_para += inv_factor * dt * (accel_by_grad_ud)
                end
            end

            # Complete the Rue_ei calculation with second part (n+1 step contribution)
            if RP.flags.Coulomb_Collision
                @. pla.Rue_ei += pla.ν_ei_eff * (-θu * pla.ue_para)
            end
        else
            error("Unknown electron drift velocity method: $(RP.flags.ud_method)")
        end

        return RP
    end # @timeit
end


function solve_ion_continuity_equation!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    # Solve the ion continuity equation
    # This function is a placeholder and should be implemented based on the specific model
    # For now, we just return the RAPID object unchanged
    @warn "Ion continuity equation solver not implemented yet" maxlog = 1
    return RP
end

"""
    update_ui_para!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the parallel ion velocity.
"""
function update_ui_para!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    return @timeit RAPID_TIMER "update_ui_para!" begin
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
            @. pla.ui_para = (
                pla.ui_para * (one_FT - (one_FT - θ) * RP.dt * eff_atomic_coll_freq) +
                    RP.dt * qi * RP.fields.E_para_tot / cnst.mi
            ) /
                (one_FT + θ * RP.dt * eff_atomic_coll_freq)

            # Add electron-ion momentum transfer effect
            if RP.flags.Coulomb_Collision
                if RP.flags.Spitzer_Resistivity
                    Rui_ei = @. pla.sptz_fac * (cnst.me / cnst.mi) * pla.ν_ei * (pla.ue_para - pla.ui_para)
                else
                    Rui_ei = @. (cnst.me / cnst.mi) * pla.ν_ei * (pla.ue_para - pla.ui_para)
                end
                pla.ui_para .+= RP.dt * Rui_ei
            end
        else
            error("Ion velocity update only implemented for ud_method = Xsec")
        end
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
function update_Te!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_Te!" begin
        # Update power terms for energy equation
        update_electron_heating_powers!(RP)

        ee = RP.config.constants.ee
        dt = RP.dt
        pla = RP.plasma
        OP = RP.operators
        θimp = RP.flags.Implicit_weight

        two_thirds_FT = FT(2.0) / FT(3.0)

        # Apply time integration method based on flag
        if RP.flags.Implicit
            if RP.flags.evolve_Te_inWall_only
                @warn "Implicit method for Te_evolve_inWall_only not implemented yet" maxlog = 1
            else
                @. OP.A_LHS = OP.II

                ePowers_tilde = copy(pla.ePowers.tot)

                # Calculate RHS
                # ePowers_tilde = ePowers already known at crruent time t
                # Note: diffu and conv will have (1-θimp) contribution
                # ePowers_tilde = pla.ePowers.tot - θimp * (pla.ePowers.diffu + pla.ePowers.conv)

                # Calculate LHS
                if RP.flags.Include_Te_diffu_term
                    # P_diffu = 1.5*∇𝐃∇*Te
                    @. ePowers_tilde -= θimp * pla.ePowers.diffu
                    @. OP.A_LHS -= two_thirds_FT * FT(1.5) * (dt * θimp * OP.∇𝐃∇)
                end

                if RP.flags.Include_Te_convec_term
                    # P_conv = -1.5*∇⋅(𝐮 Te) + 0.5*Te*(∇⋅𝐮)
                    @. ePowers_tilde -= θimp * pla.ePowers.conv
                    div_u = calculate_divergence(RP.G, pla.ueR, pla.ueZ)
                    OP.A_LHS .-= two_thirds_FT * (@views dt * θimp * (-FT(1.5) * OP.∇𝐮 + spdiagm(FT(0.5) * div_u[:])))
                end

                OP.RHS .= pla.Te_eV + two_thirds_FT * (dt * ePowers_tilde / ee)

                # Solve the linear system
                @timeit RAPID_TIMER "Te_eV LinearSolve" begin
                    pla.Te_eV .= OP.A_LHS \ OP.RHS
                end
            end
        else
            # Explicit method (forward Euler)
            @. pla.Te_eV += two_thirds_FT * pla.ePowers.tot * dt / ee
        end

        # Apply temperature limits
        @. pla.Te_eV = max(pla.Te_eV, RP.config.min_Te)
        @. pla.Te_eV = min(pla.Te_eV, RP.config.max_Te)

        return RP
    end # @timeit
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
function update_Ti!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_Ti!" begin
        # Update ion heating power terms
        update_ion_heating_powers!(RP)

        ee = RP.config.constants.ee
        dt = RP.dt
        pla = RP.plasma

        # Update ion temperature using forward Euler
        @. pla.Ti_eV += (FT(2.0) / FT(3.0)) * pla.iPowers.tot * dt / (ee)

        # Apply temperature limits (same as electrons for simplicity)
        @. pla.Ti_eV = max(pla.Ti_eV, RP.config.min_Te)
        @. pla.Ti_eV = min(pla.Ti_eV, RP.config.max_Te)

        return RP
    end # @timeit
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
  - Elastic energy loss to neutrals (2me/M per momentum-transfer collision)
  - Heat generation from density gradients
  - Ionization, excitation, and dilution powers
  - Temperature equilibration power with ions
- All powers stored in the RP.plasma.ePowers struct
"""
function update_electron_heating_powers!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_electron_heating_powers!" begin
        # Extract physical constants + reaction energies (all in RP.config.constants).
        # exc_erg_eV normalizes the Total_Excitation surface and is validated at load
        # against the table's characteristic_exc_erg_eV (Electron_RRCs), so P_exc
        # reproduces the kinetic loss exactly.
        @unpack ee, qe, me, mi, exc_erg_eV, iz_erg_eV = RP.config.constants
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
        ePowers.ela .= zero_FT
        ePowers.iz .= zero_FT
        ePowers.exc .= zero_FT
        ePowers.dilution .= zero_FT
        ePowers.equi .= zero_FT

        # If diffusion term is included in temperature equation
        if RP.flags.Include_Te_diffu_term
            # P_diffu = 1.5*∇𝐃∇*Te
            if RP.flags.Implicit
                ePowers.diffu .= ee * FT(1.5) * (OP.∇𝐃∇ * pla.Te_eV)
            else
                ePowers.diffu .= ee * FT(1.5) * compute_∇𝐃∇f_directly(RP, RP.plasma.Te_eV)
            end
        end

        # If convection term is included in temperature equation
        if RP.flags.Include_Te_convec_term
            # P_conv = -1.5*∇⋅(𝐮 Te) + 0.5*Te*(∇⋅𝐮)
            if RP.flags.Implicit
                ePowers.conv .= ee * (
                    -FT(1.5) * (OP.∇𝐮 * pla.Te_eV)
                        .+ FT(0.5) * pla.Te_eV .* calculate_divergence(RP.G, pla.ueR, pla.ueZ)
                )
            else
                ePowers.conv .= ee * (
                    -FT(1.5) * compute_∇f𝐮_directly(RP, pla.Te_eV)
                        .+ FT(0.5) * pla.Te_eV .* calculate_divergence(RP.G, pla.ueR, pla.ueZ)
                )
            end
        end

        if RP.flags.Include_heat_flux_term
            # NOTE: Assumption: 𝐪 ≈ p*𝐮 (heat flux is about in the order of pressure*velocity)
            # Pcond = Pheat = -∇⋅(Te * 𝐮) - Te*𝐮⋅∇(ln_n)
            ePowers.heat .= ee * (
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

            # Elastic energy loss to neutrals: each momentum-transfer collision hands
            # a fraction ~2me/M of the electron energy to the gas molecule. This is
            # the dominant electron cooling channel below the ~9 eV excitation
            # threshold; without it Te saturates ~20% above the kinetic (BD) value
            # at low E/p and cool-downs stall above the true saturation point.
            @. ePowers.ela = (FT(2.0) * me / mi) * pla.n_H2_gas * RRC_mom *
                FT(1.5) * (pla.Te_eV - pla.T_gas_eV) * ee

            # Get excitation rate coefficient
            RRC_exc = get_electron_RRC(RP, :Total_Excitation)
            # Excitation power (energy lost to excite particles)
            @. ePowers.exc = ee * exc_erg_eV * pla.n_H2_gas * RRC_exc

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
            @. ePowers.equi = (FT(2.0) * (mi * me / (mi + me)^2)) * FT(1.5) * ee * (pla.Te_eV - pla.Ti_eV) * pla.ν_ei
        end

        # Calculate total power (sum of all components)
        @. ePowers.tot = (
            ePowers.drag + ePowers.conv + ePowers.heat + ePowers.diffu
                - ePowers.ela - ePowers.dilution - ePowers.iz - ePowers.exc - ePowers.equi
        )

        # # Zero out power values outside the wall
        on_out_wall_nids = RP.G.nodes.on_out_wall_nids
        if !isempty(on_out_wall_nids)
            @views ePowers.tot[on_out_wall_nids] .= zero_FT
            @views ePowers.diffu[on_out_wall_nids] .= zero_FT
            @views ePowers.conv[on_out_wall_nids] .= zero_FT
            @views ePowers.drag[on_out_wall_nids] .= zero_FT
            @views ePowers.ela[on_out_wall_nids] .= zero_FT
            @views ePowers.dilution[on_out_wall_nids] .= zero_FT
            @views ePowers.iz[on_out_wall_nids] .= zero_FT
            @views ePowers.exc[on_out_wall_nids] .= zero_FT
            @views ePowers.equi[on_out_wall_nids] .= zero_FT
            @views ePowers.heat[on_out_wall_nids] .= zero_FT
        end

        return RP
    end # @timeit
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
function update_ion_heating_powers!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_ion_heating_powers!" begin
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
            @. iPowers.equi = (FT(2.0) * (mi * me / (mi + me)^2)) * FT(1.5) * ee * (pla.Te_eV - pla.Ti_eV) * pla.ν_ei
        end

        # Calculate total ion heating power
        @. iPowers.tot = iPowers.atomic + iPowers.equi

        # Set power to zero outside wall boundaries
        on_out_wall_nids = RP.G.nodes.on_out_wall_nids
        if !isempty(on_out_wall_nids)
            @views iPowers.tot[on_out_wall_nids] .= zero_FT
            @views iPowers.atomic[on_out_wall_nids] .= zero_FT
            @views iPowers.equi[on_out_wall_nids] .= zero_FT
        end

        return RP
    end # @timeit
end

"""
    calculate_ν_en_iz!(RP::RAPID{FT}) where FT<:AbstractFloat

Calculate the ionization frequency ν_en_iz = n_H2_gas * <σ_iz * v_e>
"""
function calculate_ν_en_iz!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    # Calculate ionization rate based on the method specified in flags
    if RP.flags.Ionz_method == "Townsend_coeff"
        # Compute source by electron avalanche using Townsend coefficient
        # α = 3.88 * p * exp(-95 * p / |E_para|)
        α = @. 3.88 * RP.config.prefilled_gas_pressure *
            exp(-95 * RP.config.prefilled_gas_pressure / abs(RP.fields.E_para_tot))

        # Electron ionization frequency
        RP.plasma.ν_en_iz = @. α * abs(RP.plasma.ue_para)
    elseif RP.flags.Ionz_method == "Xsec"
        # Ionization frequency = (gas density) * (<σ_iz*v>)
        eRRC_iz = get_electron_RRC(RP, RP.eRRCs, :Ionization)
        @. RP.plasma.ν_en_iz = RP.plasma.n_H2_gas * eRRC_iz
    else
        error("Unknown ionization method: $(RP.flags.Ionz_method)")
    end

    # Zero out the ionization frequency outside the wall
    RP.plasma.ν_en_iz[RP.G.nodes.on_out_wall_nids] .= 0.0

    # Update sparse matrix operator for implicit methods if needed
    if RP.flags.Implicit
        RP.operators.ν_en_iz .= @views spdiagm(RP.plasma.ν_en_iz[:])
    end

    return RP
end

"""
    solve_electron_continuity_equation!(RP::RAPID{FT}) where FT<:AbstractFloat

Solve the electron continuity equation to update electron density.
Uses either explicit or implicit time integration based on RP.flags.Implicit.
"""
function solve_electron_continuity_equation!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "solve_electron_continuity_equation!" begin
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
            # ne * ν_en_iz = ne * nH2_gas * eRRC_iz
            calculate_ν_en_iz!(RP)
            op.RHS += pla.ne .* pla.ν_en_iz
        end

        if RP.flags.diffu
            # ∇⋅𝐃⋅∇n
            op.RHS .+= compute_∇𝐃∇f_directly(RP, pla.ne)
        end
        if RP.flags.convec
            # -∇⋅(n 𝐮)
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
            @. op.A_LHS = op.II - θn * dt * (op.∇𝐃∇ - op.∇𝐮 + op.ν_en_iz)

            # Solve the linear system
            @timeit RAPID_TIMER "ne LinearSolve`" begin
                pla.ne .= op.A_LHS \ op.RHS
            end
        else
            @. RP.plasma.ne += dt * op.RHS
        end
        return RP
    end # @timeit
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
function treat_electron_outside_wall!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "treat_electron_outside_wall!" begin
        # Estimate the number of electrons generated by ionization
        if RP.flags.Implicit
            θimp = RP.flags.Implicit_weight
            effective_ne = @. (FT(1.0) - θimp) * RP.prev_n + θimp * RP.plasma.ne
            Ne_iz = @. RP.plasma.ν_en_iz * effective_ne * RP.G.inVol2D * RP.dt
        else
            Ne_iz = @. RP.plasma.ν_en_iz * RP.plasma.ne * RP.G.inVol2D * RP.dt
        end

        # Estimate electron loss outside the wall
        # TODO: How to accurately define the volume outside/on the wall?
        on_out_wall_nids = RP.G.nodes.on_out_wall_nids
        Ne_loss = @. RP.plasma.ne[on_out_wall_nids] * FT(2.0 * pi) * RP.G.Jacob[on_out_wall_nids] * RP.G.dR * RP.G.dZ

        # Track changes in number of electrons
        Ntracker = RP.diagnostics.Ntracker

        Ntracker.cum0D_Ne_src += sum(Ne_iz)
        @. Ntracker.cum2D_Ne_src += Ne_iz

        Ntracker.cum0D_Ne_loss += sum(Ne_loss)
        @. Ntracker.cum2D_Ne_loss[on_out_wall_nids] += Ne_loss

        # Set electron density to zero outside the wall
        RP.plasma.ne[on_out_wall_nids] .= 0.0

        # Damp out electron temperature outside the wall
        out_wall_nids = RP.G.nodes.out_wall_nids
        @. RP.plasma.Te_eV[out_wall_nids] *= RP.damping_func[out_wall_nids]

        # Correct negative densities if enabled
        if RP.flags.negative_n_correction
            neg_n_idx = findall(RP.plasma.ne .< 0)
            if !isempty(neg_n_idx)
                ori_ne = copy(RP.plasma.ne)
                @inbounds for nid in neg_n_idx
                    rid = RP.G.nodes.rid[nid]
                    zid = RP.G.nodes.zid[nid]

                    ngh_rids = max(1, rid - 1):min(RP.G.NR, rid + 1)
                    ngh_zids = max(1, zid - 1):min(RP.G.NZ, zid + 1)

                    RP.plasma.ne[nid] = min(0.1 * abs(ori_ne[nid]), 0.01 * mean(abs.(ori_ne[ngh_rids, ngh_zids])))

                    Ne_loss = (RP.plasma.ne[nid] - ori_ne[nid]) * FT(2.0 * pi) * RP.G.Jacob[nid] * RP.G.dR * RP.G.dZ
                    Ntracker.cum0D_Ne_loss += Ne_loss
                    Ntracker.cum2D_Ne_loss[nid] += Ne_loss
                end
            end
        end

        if !RP.flags.update_ni_independently
            RP.plasma.ni .= RP.plasma.ne
        end

        return RP
    end # @timeit
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
function treat_ion_outside_wall!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "treat_ion_outside_wall!" begin
        # Estimate the number of ions generated by ionization
        # NOTE: ν_en_iz must be multiplied by the electron density to get the ionization rate
        # In other words, Ne_iz = Ni_iz
        Ni_iz = @. RP.plasma.ν_en_iz * RP.plasma.ne * RP.G.inVol2D * RP.dt

        # Estimate electron loss outside the wall
        # TODO: How to accurately define the volume outside/on the wall?
        on_out_wall_nids = RP.G.nodes.on_out_wall_nids
        Ni_loss = @. RP.plasma.ni[on_out_wall_nids] * FT(2.0 * pi) * RP.G.Jacob[on_out_wall_nids] * RP.G.dR * RP.G.dZ

        # Track changes in number of electrons
        Ntracker = RP.diagnostics.Ntracker

        Ntracker.cum0D_Ni_src += sum(Ni_iz)
        @. Ntracker.cum2D_Ni_src += Ni_iz

        Ntracker.cum0D_Ni_loss += sum(Ni_loss)
        @. Ntracker.cum2D_Ni_loss[on_out_wall_nids] += Ni_loss

        # Secondary electron generated by ion impacts on wall
        # TODO: needs to improve this part (somehow this should generate them inside wall)
        if RP.flags.secondary_electron
            RP.plasma.ne[on_out_wall_nids] .+= RP.flags.γ_2nd_electron * RP.plasma.ni[on_out_wall_nids]
        end

        # Set ion density to zero outside the wall
        RP.plasma.ni[on_out_wall_nids] .= 0.0

        # Set electron temperature to room temperature outside the wall
        out_wall_nids = RP.G.nodes.out_wall_nids
        @. RP.plasma.Ti_eV[out_wall_nids] *= RP.damping_func[out_wall_nids]

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
    end # @timeit
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
function calculate_para_grad_of_scalar_F(RP::RAPID{FT}, F::Matrix{FT}; upwind::Bool = RP.flags.upwind) where {FT <: AbstractFloat}
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
        @inbounds for j in 2:(NZ - 1), i in 2:(NR - 1)

            # R-direction contribution
            if abs(RP.plasma.ueR[i, j]) < eps_val
                # Zero velocity: use central differencing for stability
                para_∇F[i, j] += RP.fields.bR[i, j] * (F[i + 1, j] - F[i - 1, j]) * (inv_dR * half)
            elseif RP.plasma.ueR[i, j] > zero_FT
                # Positive flow: backward difference (upwind)
                para_∇F[i, j] += RP.fields.bR[i, j] * (F[i, j] - F[i - 1, j]) * inv_dR
            else
                # Negative flow: forward difference (upwind)
                para_∇F[i, j] += RP.fields.bR[i, j] * (F[i + 1, j] - F[i, j]) * inv_dR
            end

            # Z-direction contribution
            if abs(RP.plasma.ueZ[i, j]) < eps_val
                # Zero velocity: use central differencing for stability
                para_∇F[i, j] += RP.fields.bZ[i, j] * (F[i, j + 1] - F[i, j - 1]) * (inv_dZ * half)
            elseif RP.plasma.ueZ[i, j] > zero_FT
                # Positive flow: backward difference (upwind)
                para_∇F[i, j] += RP.fields.bZ[i, j] * (F[i, j] - F[i, j - 1]) * inv_dZ
            else
                # Negative flow: forward difference (upwind)
                para_∇F[i, j] += RP.fields.bZ[i, j] * (F[i, j + 1] - F[i, j]) * inv_dZ
            end
        end
    else
        # Central difference scheme for interior points
        # This is more accurate for smooth solutions but may have stability issues for advection-dominated flows
        @inbounds for j in 2:(NZ - 1), i in 2:(NR - 1)
            para_∇F[i, j] = RP.fields.bR[i, j] * (F[i + 1, j] - F[i - 1, j]) * (inv_dR * half) +
                RP.fields.bZ[i, j] * (F[i, j + 1] - F[i, j - 1]) * (inv_dZ * half)
        end
    end

    # Handle boundaries with one-sided differences
    # Calculate R derivative contributions
    @inbounds for j in 1:NZ
        # Left boundary: forward difference
        i = 1
        para_∇F[i, j] += RP.fields.bR[i, j] * (F[i + 1, j] - F[1, j]) * inv_dR
        i = NR
        # Right boundary: backward difference
        para_∇F[i, j] += RP.fields.bR[i, j] * (F[i, j] - F[i - 1, j]) * inv_dR
    end
    # Bottom and Top boundary: central difference
    @inbounds for j in [1, NZ]
        for i in 2:(NR - 1)
            para_∇F[i, j] += RP.fields.bR[i, j] * (F[i + 1, j] - F[i - 1, j]) * (inv_dR * half)
        end
    end

    # Calculate Z derivative contributions
    @inbounds for i in 1:NR
        # Bottom boundary: forward difference
        j = 1
        para_∇F[i, j] += RP.fields.bZ[i, j] * (F[i, j + 1] - F[i, j]) * inv_dZ
        # Top boundary: backward difference
        j = NZ
        para_∇F[i, j] += RP.fields.bZ[i, j] * (F[i, j] - F[i, j - 1]) * inv_dZ
    end
    # Left and Right boundary: central difference
    @inbounds for i in [1, NR]
        for j in 2:(NZ - 1)
            para_∇F[i, j] += RP.fields.bZ[i, j] * (F[i, j + 1] - F[i, j - 1]) * (inv_dZ * half)
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
function calculate_grad_of_scalar_F(RP::RAPID{FT}, F::Matrix{FT}; upwind::Bool = RP.flags.upwind) where {FT <: AbstractFloat}
    NR, NZ = size(F)
    @assert NR > 1 && NZ > 1 "Grid size must be at least 2x2"

    # Define constants for type stability
    zero_FT = zero(FT)
    half = FT(0.5)
    eps_val = eps(FT)

    # Pre-compute inverse values for faster calculation
    inv_dR = one(FT) / RP.G.dR
    inv_dZ = one(FT) / RP.G.dZ

    # Initialize output arrays
    ∇F_R = zeros(FT, NR, NZ)
    ∇F_Z = zeros(FT, NR, NZ)


    # Calculate gradients for interior points
    if upwind
        # Upwind differencing scheme based on flow velocity
        @inbounds for j in 2:(NZ - 1), i in 2:(NR - 1)
            # R-direction gradient
            if abs(RP.plasma.ueR[i, j]) < eps_val
                # Zero velocity: use central differencing for stability
                ∇F_R[i, j] = (F[i + 1, j] - F[i - 1, j]) * (inv_dR * half)
            elseif RP.plasma.ueR[i, j] > zero_FT
                # Positive velocity: backward difference (upwind)
                ∇F_R[i, j] = (F[i, j] - F[i - 1, j]) * inv_dR
            else
                # Negative velocity: forward difference (upwind)
                ∇F_R[i, j] = (F[i + 1, j] - F[i, j]) * inv_dR
            end

            # Z-direction gradient
            if abs(RP.plasma.ueZ[i, j]) < eps_val
                # Zero velocity: use central differencing for stability
                ∇F_Z[i, j] = (F[i, j + 1] - F[i, j - 1]) * (inv_dZ * half)
            elseif RP.plasma.ueZ[i, j] > zero_FT
                # Positive velocity: backward difference (upwind)
                ∇F_Z[i, j] = (F[i, j] - F[i, j - 1]) * inv_dZ
            else
                # Negative velocity: forward difference (upwind)
                ∇F_Z[i, j] = (F[i, j + 1] - F[i, j]) * inv_dZ
            end
        end
    else
        # Standard central differencing scheme
        @inbounds for j in 2:(NZ - 1), i in 2:(NR - 1)
            # R-direction gradient
            ∇F_R[i, j] = (F[i + 1, j] - F[i - 1, j]) * (inv_dR * half)
            # Z-direction gradient
            ∇F_Z[i, j] = (F[i, j + 1] - F[i, j - 1]) * (inv_dZ * half)
        end
    end

    # Handle boundaries with one-sided differences
    # Calculate R derivative contributions
    @inbounds for j in 1:NZ
        # Left boundary: forward difference
        ∇F_R[1, j] = (F[2, j] - F[1, j]) * inv_dR
        # Right boundary: backward difference
        ∇F_R[NR, j] = (F[NR, j] - F[NR - 1, j]) * inv_dR
    end
    # Bottom and Top boundary: central difference
    @inbounds for j in [1, NZ]
        for i in 2:(NR - 1)
            ∇F_R[i, j] = (F[i + 1, j] - F[i - 1, j]) * (inv_dR * half)
        end
    end

    # Calculate Z derivative contributions
    @inbounds for i in 1:NR
        # Bottom boundary: forward difference
        ∇F_Z[i, 1] = (F[i, 2] - F[i, 1]) * inv_dZ
        # Top boundary: backward difference
        ∇F_Z[i, NZ] = (F[i, NZ] - F[i, NZ - 1]) * inv_dZ
    end
    # Left and Right boundary: central difference
    @inbounds for i in [1, NR]
        for j in 2:(NZ - 1)
            ∇F_Z[i, j] = (F[i, j + 1] - F[i, j - 1]) * (inv_dZ * half)
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
function calculate_electron_acceleration_by_pressure(RP::RAPID{FT}; num_SM::Int = 2) where {FT <: AbstractFloat}
    # alias
    @unpack ee, me = RP.config.constants

    # Smooth the density field to reduce numerical noise (skip if num_SM is 0)
    n_SM = smooth_data_2D(RP.plasma.ne; num_SM, weighting = RP.G.Jacob)
    n_SM[n_SM .< 0] .= zero(FT)

    # Calculate ln(n) gradients along B to avoid division by zero issues with low density
    # Calculate temperature gradients along B
    para_grad_ln_n = calculate_para_grad_of_scalar_F(RP, log.(n_SM))
    para_grad_Te_eV = calculate_para_grad_of_scalar_F(RP, RP.plasma.Te_eV)

    # Combine both terms for total pressure gradient acceleration
    accel_by_pressure = @. (
        - para_grad_ln_n * RP.plasma.Te_eV * ee / me
            - para_grad_Te_eV * ee / me
    )

    if RP.flags.limit_acceleration.state
        factor = RP.flags.limit_acceleration.factor
        max_abs_accel = factor .* maximum(abs.(RP.plasma.ue_para[RP.G.nodes.in_wall_nids])) ./ RP.dt
        clamp!(accel_by_pressure, -max_abs_accel, +max_abs_accel)
    end

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
function calculate_electron_acceleration_by_convection(RP::RAPID{FT}; num_SM::Int = 2, flag_upwind::Bool = RP.flags.upwind) where {FT <: AbstractFloat}

    # Smooth the density field to reduce numerical noise (skip if num_SM is 0)
    ue_para_SM = smooth_data_2D(RP.plasma.ue_para; num_SM, weighting = RP.G.Jacob)

    ∇ud_R, ∇ud_Z = calculate_grad_of_scalar_F(RP, ue_para_SM; upwind = flag_upwind)
    accel_by_convection = @. -(RP.plasma.ueR * ∇ud_R + RP.plasma.ueZ * ∇ud_Z)

    if RP.flags.limit_acceleration.state
        factor = RP.flags.limit_acceleration.factor
        min_max_accel = factor .* extrema(RP.plasma.ue_para[RP.G.nodes.in_wall_nids]) ./ RP.dt
        clamp!(accel_by_convection, min_max_accel...)
    end

    return accel_by_convection
end

"""
    solve_Ampere_equation!(RP::RAPID{FT}, F::Fields{FT}=RP.fields; plasma::Bool=true, coils::Bool=true, update_Eϕ_self::Bool=true) where {FT<:AbstractFloat}

Solve Ampère's equation (Grad-Shafranov equation) to update self-consistent magnetic fields.

Solves: ΔGS * ψ = - μ₀R²Jϕ with boundary conditions from Green's function.
Updates ψ_self, and optionally Eϕ_self.

# Arguments
- `RP::RAPID{FT}`: RAPID simulation object containing plasma, grid, and operators
- `F::Fields{FT}`: Fields object to update (defaults to `RP.fields`)

# Keyword Arguments
- `plasma::Bool=true`: Include plasma current density (Jϕ) in the source term
- `coils::Bool=true`: Include external coil current contributions to the source term
"""
function solve_Ampere_equation!(RP::RAPID{FT}, F::Fields{FT} = RP.fields; plasma::Bool = true, coils::Bool = true, update_Eϕ_self::Bool = true) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "solve_Ampere_equation" begin
        # Alias for readability
        pla = RP.plasma
        OP = RP.operators
        μ0 = RP.config.constants.μ0
        csys = RP.coil_system

        if plasma
            @. OP.RHS = - μ0 * RP.G.R2D * pla.Jϕ
        else
            @. OP.RHS = zero(FT)
        end

        if coils && csys.n_total > 0
            inside_Jϕ_coil_k = distribute_coil_currents_to_Jϕ(csys, RP.G)
            @. OP.RHS .+= -μ0 * RP.G.R2D * inside_Jϕ_coil_k
        end

        # Boudnary condition: calculate psi values at boundaries using the Green_inWall2bdy
        if plasma
            @views OP.RHS[RP.G.BDY_idx] .= (RP.G.Green_inWall2bdy * pla.Jϕ[RP.G.nodes.in_wall_nids]) * RP.G.dR * RP.G.dZ
        else
            OP.RHS[RP.G.BDY_idx] .= zero(FT)
        end

        if coils && csys.n_total > 0
            OP.RHS[RP.G.BDY_idx] .+= csys.Green_coils2bdy * csys.coils.current
        end

        old_ψ_self = copy(F.ψ_self)

        # solve Ampere's equation
        F.ψ_self = OP.ΔGS \ OP.RHS

        # % calculate the magnetic field from the self-consistent ψ
        calculate_B_from_ψ!(RP.G, F.ψ_self, F.BR_self, F.BZ_self)

        if update_Eϕ_self && RP.flags.E_para_self_EM
            @. F.Eϕ_self = - (F.ψ_self - old_ψ_self) / (RP.G.R2D * RP.flags.Ampere_nstep * RP.dt)
        end

        return RP
    end # @timeit
end


"""
    solve_Ampere_equation(RP::RAPID{FT}; plasma::Bool=true, coils::Bool=true) where {FT<:AbstractFloat}

Non-mutating version of `solve_Ampere_equation!` that returns a new Fields object.

Creates a new `Fields{FT}` object and solves Ampère's equation without modifying
the original fields in the RAPID object.

# See Also
- [`solve_Ampere_equation!`](@ref): In-place version that modifies the RAPID object
"""
function solve_Ampere_equation(RP::RAPID{FT}; plasma::Bool = true, coils::Bool = true) where {FT <: AbstractFloat}
    F = Fields{FT}(RP.G.NR, RP.G.NZ)
    solve_Ampere_equation!(RP, F; plasma, coils)
    return F
end

"""
    solve_coupled_momentum_Ampere_equations_with_coils!(RP::RAPID{FT};
                                                        tolerance=1e-3,
                                                        max_iter=10,
                                                        relaxation_w=0.5) where {FT}

Solve coupled electron momentum and Ampère equations with coil interactions using Picard iteration.

Solves the coupled system:
- Electron parallel momentum:
    - Au ≡ [ 𝐈 + Δt*θimp*(νe_eff + 𝐮⋅∇)]
    - Au * ue∥⁽ⁿ⁺¹⁾ = ue∥⁽ⁿ⁾ + Δt*ã∥⁽ⁿ⁾ - (qe*bϕ²/me*R)*ψ_self⁽ⁿ⁺¹⁾
- Implicit Ampere's equation:
    - [Au*ΔGS - μ₀*ne*qe²*bϕ²/me]*ψ_self⁽ⁿ⁺¹⁾ = -μ₀R² J̃ϕ⁽ⁿ⁾
- Coil circuit equations: V = Ic*Rc + Lc(dI/dt) + mutual coupling (coils+plasma)

The electromagnetic induction coupling creates strong nonlinearity requiring iterative solution.

# Arguments
- `tolerance`: Convergence tolerance for Picard iteration (default: 1e-3)
- `max_iter`: Maximum iterations (default: 10)
- `relaxation_w`: Boundary relaxation weight (default: 0.5)

# Updates
Modifies `RP.plasma.ue_para`, `RP.fields.ψ_self`, `RP.fields.Eϕ_self`, and magnetic fields.
"""
function solve_coupled_momentum_Ampere_equations_with_coils!(
        RP::RAPID{FT};
        tolerance::FT = 1.0e-6,
        max_iter::Int = 10,
        relaxation_w::FT = 0.5
    ) where {FT <: AbstractFloat}

    # Aliases for readability
    pla = RP.plasma
    F = RP.fields
    OP = RP.operators
    G = RP.G
    flags = RP.flags
    dt = RP.dt
    csys = RP.coil_system

    # Physical constants
    @unpack ee, me, μ0, qe = RP.config.constants

    θimp = FT(1.0)  # Explicit(=0), Crank-Nicholson(=0.5), Backward Euler(=1)

    # Factor for EM drive contribution
    # derived from [E_para_EM = -qe/me * (ψ^(n+1) - ψ^(n))/(R*Δt)  = -facEM * ((ψ^(n+1) - ψ^(n)))/Δt]
    facEM = (qe / me) * (F.bϕ ./ G.R2D)

    # 1. calculate accel_para_tilde using the information at the current time step
    accel_para_tilde = zeros(FT, G.NR, G.NZ) # Initialize acceleration field

    # pressure gradient contribution: [-∇∥(ne*Te)/(me*ne)]
    if RP.flags.Include_ud_pressure_term
        accel_para_tilde .+= calculate_electron_acceleration_by_pressure(RP)
    end

    # convection contribution:  (1-θimp)*[-(𝐮⋅∇)u∥]
    if RP.flags.Include_ud_convec_term
        accel_para_tilde .+= (one(FT) - θimp) * calculate_electron_acceleration_by_convection(RP)
    end

    # Electric field contributions: [(qe/me)* (E∥_ext + E∥_self_ES)]
    if flags.E_para_self_ES
        @. accel_para_tilde += qe / me * (F.E_para_ext + F.E_para_self_ES)
    else
        @. accel_para_tilde += qe / me * (F.E_para_ext)
    end

    # Effective electron collision frequency
    νe_eff = pla.ν_en_mom + pla.ν_en_iz + pla.ν_ei_eff

    @. accel_para_tilde += (
        facEM / dt * F.ψ_self
            - (one(FT) - θimp) * νe_eff * pla.ue_para
            + pla.ν_ei_eff * pla.ui_para
    )


    # 2. Define Au matrix for the electron parallel momentum equation
    # Au ≡ [ 𝐈 + Δt*θimp*(νe_eff + 𝐮⋅∇)]
    Au = DiscretizedOperator{FT}(dims_rz = (G.NR, G.NZ))
    Au .= OP.II + spdiagm(@views dt * θimp * νe_eff[:])
    if flags.Include_ud_convec_term
        Au .+= dt * θimp * OP.𝐮∇
    end
    Au_X_ui_para = Au * pla.ui_para

    # Jϕ_tilde is the part of prediction of Jϕ at the next time step, using the current information
    Jϕ_tilde = @. (
        pla.ne * qe * (pla.ue_para + dt * accel_para_tilde)
            + pla.ni * (ee * pla.Zeff) * Au_X_ui_para
    ) * F.bϕ


    # Calculate Rue_ei (electron-ion momentum exchange rate) - first part (n-th step)
    if RP.flags.Coulomb_Collision
        @. pla.Rue_ei = pla.ν_ei_eff * (pla.ui_para - (one(FT) - θimp) * pla.ue_para)
    end

    # Toroidal current density Jϕ @ t=(n-th step)
    Jϕ_pla_0 = @. (qe * pla.ne * pla.ue_para + pla.ni * (ee * pla.Zeff) * pla.ui_para) * F.bϕ

    # 6. Initial guess for ψ_self using θ-implicit scheme with extrapolated Eϕ_self
    # Predict Eϕ_self(n+1) by linear extrapolation: 2*E(n) - E(n-1)
    Eϕ_self_np1_pred = @. FT(2.0) * F.Eϕ_self - FT(1.0) * F.Eϕ_self_prev
    # Apply θ-weighting: (1-θ)*E(n) + θ*E(n+1_predicted)
    new_ψ_self_k = @. F.ψ_self - dt * G.R2D * ((one(FT) - θimp) * F.Eϕ_self + θimp * Eϕ_self_np1_pred)

    # Prepare Picard iteration for coupled system
    F.Eϕ_self_prev .= F.Eϕ_self # Store previous Eϕ_self for self-consistency
    old_ψ_self = copy(F.ψ_self) # Store old ψ_self for convergence checking


    ue_para_k = zeros(FT, G.NR, G.NZ) # Initialize ue_para for iterationo
    new_ψ_self_kp1 = zeros(FT, G.NR, G.NZ) # Initialize next ψ_self for iteration
    RHS = zeros(FT, G.NR, G.NZ) # preallocate reusable RHS for efficiency
    Jϕ_pla_k = zeros(FT, G.NR, G.NZ) # Initialize Jϕ for iteration

    # Prepare coil_system for current calculation
    if (RP.dt != csys.Δt || θimp != csys.θimp)
        # If the time step or implicit factor have changed, recalculate the coil system matrices
        csys.Δt = RP.dt
        csys.θimp = θimp
        calculate_circuit_matrices!(csys)
    end

    # Define implicit LHS matrix for the coupled Ampere equation
    # A_imp_ampere ≡ [Au*ΔGS - μ₀*ne*qe²*bϕ²/me]
    induc_shielding_term = @. μ0 * pla.ne * qe^2 * F.bϕ^2 / me
    induc_shielding_term[G.BDY_idx] .= 0.0 # For dirichlet condition of A_imp_ampere
    A_imp_ampere = (Au * OP.ΔGS) - spdiagm(@views induc_shielding_term[:])

    # TODO: need to make it more efficient.. direct indexing is not efficient
    for nid in G.BDY_idx
        A_imp_ampere.matrix[nid, nid] = one(FT) # Dirichlet condition at boundary nodes
    end

    new_coils_I_k = zeros(FT, csys.n_total) # Initialize coil currents for iteration

    iter = 1
    converged = false
    while (true)
        # Step #1: Calculate ue_para, Jphi, coils according to new_psi_self_k
        @. RHS = pla.ue_para + dt * accel_para_tilde - facEM * new_ψ_self_k
        ue_para_k .= Au \ RHS # Solve for ue_para at (k)-th step
        @. Jϕ_pla_k = (qe * pla.ne * ue_para_k + pla.ni * (ee * pla.Zeff) * pla.ui_para) * F.bϕ


        if csys.n_total > 0
            Mcp_dIpla = 2π * csys.Green_grid2coils * (Jϕ_pla_k[:] .- Jϕ_pla_0[:]) * G.dR * G.dZ

            if flags.convec
                # TODO: Is this part needed? Grid is not moving, so plasma movement should not affect coil currents?
                Ipla = @. (θimp * Jϕ_pla_k + (1 - θimp) * Jϕ_pla_0) * G.dR * G.dZ
                pla_displacement_R = pla.ueR * dt + 0.5 * pla.mean_aR_by_JxB * dt^2
                pla_displacement_Z = pla.ueZ * dt + 0.5 * pla.mean_aZ_by_JxB * dt^2
                # change rate of Mcp (mutual inductance between coils and plasma) due to plasma movement
                Ipla_dMcp = 2π * (
                    csys.dGreen_dRg_grid2coils * (Ipla[:] .* pla_displacement_R[:]) +
                        csys.dGreen_dZg_grid2coils * (Ipla[:] .* pla_displacement_Z[:])
                )

                # grad_Ipla_R, grad_Ipla_Z = Cal_grad_of_scalar_F(reshape(Ipla, size(R2D)))
                # dIpla_by_conv = pla_displacement_R .* grad_Ipla_R + pla_displacement_Z .* grad_Ipla_Z
                # Mcp_dIpla_by_conv = 2π * coils.G_grid2coil * dIpla_by_conv[:]
                Mcp_dIpla_by_conv = 0
            else
                Ipla_dMcp = 0
                Mcp_dIpla_by_conv = 0
            end

            coil_flux_change_by_plasma = @. Mcp_dIpla + Ipla_dMcp + Mcp_dIpla_by_conv

            circuit_rhs = calculate_LR_circuit_rhs_by_coils(csys, RP.time_s) - coil_flux_change_by_plasma
            new_coils_I_k = csys.inv_A_LR_circuit * circuit_rhs  # valid if "dt" is constant
        end

        # Step #2: Update Boundary psi by both plasma and coils currents using Green's function
        new_ψ_self_kp1_at_BDY = (G.Green_inWall2bdy * Jϕ_pla_k[G.nodes.in_wall_nids]) * G.dR * G.dZ
        if csys.n_total > 0
            new_ψ_self_kp1_at_BDY .+= csys.Green_coils2bdy * new_coils_I_k
        end

        #  update (k+1)-th boudary psi with some relaxation
        @views new_ψ_self_kp1_at_BDY .= (
            relaxation_w * new_ψ_self_kp1_at_BDY
                + (one(FT) - relaxation_w) * new_ψ_self_k[G.BDY_idx]
        )

        # Step #3: Set RHS of the implicit Ampere equation
        @. RHS = -μ0 * G.R2D * Jϕ_tilde
        if csys.n_total > 0
            inside_Jϕ_coil_k = distribute_coil_currents_to_Jϕ(csys, RP.G; currents = new_coils_I_k)
            @. RHS .+= -μ0 * G.R2D * inside_Jϕ_coil_k
        end
        RHS[G.BDY_idx] .= new_ψ_self_kp1_at_BDY # Set RHS at boundary nodes

        # Step #4: Solve the implicit Ampere equation
        new_ψ_self_kp1 = A_imp_ampere \ RHS

        # Step #5: Check if ψ solution is converged
        convergence_rate = norm(new_ψ_self_kp1 - new_ψ_self_k) / norm(new_ψ_self_k)
        if (convergence_rate < tolerance)
            # println("  ψ_self converged after $iter iterations! convergence_rate: $convergence_rate")
            converged = true
            break
        elseif iter >= max_iter
            converged = false
            println("  Warning: Picard iteration did not converge after $max_iter iterations at step=$(RP.step)")
            println("  Final change: $(norm(new_ψ_self_kp1 - new_ψ_self_k) / norm(new_ψ_self_k))")
            break
        else
            new_ψ_self_k .= new_ψ_self_kp1 # Update for next iteration
            iter += 1
        end
    end

    # 7. Final updates of electromagnetic fields
    @. F.ψ_self = new_ψ_self_kp1

    # Update self-consistent electric field: Eϕ = -∂ψ/∂t/R
    @. F.Eϕ_self = -(F.ψ_self - old_ψ_self) / (G.R2D * dt)

    # Update parallel electron velocity: ue_para = (ψ_self - ψ_self_old)/(R*Δt) + ue_para_k
    @. RHS = pla.ue_para + dt * accel_para_tilde - facEM * F.ψ_self
    pla.ue_para = Au \ RHS # Solve for ue_para at (k)-th step

    # Complete the Rue_ei calculation with second part (n+1 step contribution)
    if RP.flags.Coulomb_Collision
        @. pla.Rue_ei += pla.ν_ei_eff * (-θimp * pla.ue_para)
    end

    @. pla.Jϕ = pla.ne * qe * pla.ue_para * F.bϕ

    # Update coil currents
    if RP.coil_system.n_total > 0
        csys.time_s += csys.Δt
        set_all_currents!(csys, new_coils_I_k)
    end

    # Update magnetic fields from ψ_self
    calculate_B_from_ψ!(G, F.ψ_self, F.BR_self, F.BZ_self)

    return RP
end


"""
    combine_Au_and_ΔGS_sparse_matrices(RP::RAPID{FT}, Au::SparseMatrixCSC{FT,Int}, A_GS::SparseMatrixCSC{FT,Int}) where {FT<:AbstractFloat}

Combine the electron parallel momentum operator and Grad–Shafranov operator into a single block sparse matrix for coupled solves.

Constructs a 2×2 block matrix of size (2N×2N):

    [ Au         diag(inductive_term);
      diag(current_term)   A_GS        ]

where:
- `Au` is the electron parallel momentum operator.
- `A_GS` is the Grad–Shafranov operator.
- `inductive_term = (qe/me) * (bϕ ./ R2D)` couples poloidal flux changes into the momentum equation.
- `current_term = μ0 * R2D * ne * qe * bϕ` couples plasma current into Ampère's equation.

# Arguments
- `RP::RAPID{FT}`: Simulation state, providing grid geometry and physical constants.
- `Au::SparseMatrixCSC{FT,Int}`: Momentum operator matrix.
- `A_GS::SparseMatrixCSC{FT,Int}`: Grad–Shafranov operator matrix.

# Returns
- `SparseMatrixCSC{FT,Int}`: Combined block sparse matrix of size (2N×2N), where N = RP.G.NR * RP.G.NZ.

# Notes
- Boundary conditions are enforced by zeroing coupling terms at boundary nodes.
"""
function combine_Au_and_ΔGS_sparse_matrices(RP::RAPID{FT}, Au::SparseMatrixCSC{FT}, A_GS::SparseMatrixCSC{FT}) where {FT <: AbstractFloat}
    # Get dimensions
    N = RP.G.NR * RP.G.NZ

    # Physical constants
    @unpack qe, me, μ0 = RP.config.constants

    # Calculate coupling terms
    inductive_term = @. qe * RP.fields.bϕ / (me * RP.G.R2D)
    electron_current_term = @. μ0 * RP.G.R2D * RP.plasma.ne * qe * RP.fields.bϕ
    # Zero out boundary terms for proper boundary conditions
    electron_current_term[RP.G.BDY_idx] .= zero(FT)

    # Count non-zero entries for efficient allocation
    # Upper left: A_upara entries
    # Lower right: A_GS entries
    # Upper right: N diagonal entries (inductive coupling)
    # Lower left: N diagonal entries (current coupling)
    nnz_A_upara = nnz(Au)
    nnz_A_GS = nnz(A_GS)
    nnz_coupling = 2 * N  # Two diagonal blocks
    total_nnz = nnz_A_upara + nnz_A_GS + nnz_coupling

    # Pre-allocate arrays for sparse matrix construction
    I_combined = zeros(Int, total_nnz)
    J_combined = zeros(Int, total_nnz)
    V_combined = zeros(FT, total_nnz)

    idx = 1

    # Upper left block: A_upara (rows 1:N, cols 1:N)
    I_up, J_up, V_up = findnz(Au)
    len_upara = length(I_up)
    I_combined[idx:(idx + len_upara - 1)] = I_up
    J_combined[idx:(idx + len_upara - 1)] = J_up
    V_combined[idx:(idx + len_upara - 1)] = V_up
    idx += len_upara

    # Lower right block: A_GS (rows N+1:2N, cols N+1:2N)
    I_gs, J_gs, V_gs = findnz(A_GS)
    len_gs = length(I_gs)
    I_combined[idx:(idx + len_gs - 1)] = I_gs .+ N  # Shift row indices
    J_combined[idx:(idx + len_gs - 1)] = J_gs .+ N  # Shift column indices
    V_combined[idx:(idx + len_gs - 1)] = V_gs
    idx += len_gs

    # Upper right block: inductive coupling (rows 1:N, cols N+1:2N)
    # Diagonal matrix: (i,i) -> value inductive_term[i]
    for i in 1:N
        I_combined[idx] = i      # Row index
        J_combined[idx] = i + N  # Column index (shifted to upper right block)
        V_combined[idx] = inductive_term[i]
        idx += 1
    end

    # Lower left block: current coupling (rows N+1:2N, cols 1:N)
    # Diagonal matrix: (i,i) -> value electron_current_term[i]
    for i in 1:N
        I_combined[idx] = i + N  # Row index (shifted to lower left block)
        J_combined[idx] = i      # Column index
        V_combined[idx] = electron_current_term[i]
        idx += 1
    end

    return sparse(I_combined, J_combined, V_combined, 2 * N, 2 * N)
end

"""
    solve_combined_momentum_Ampere_equations_with_coils!(RP::RAPID{FT};
                                                         tolerance::FT=1e-3,
                                                         max_iter::Int=10,
                                                         relaxation_w::FT=0.5) where {FT<:AbstractFloat}

Solve the coupled electron momentum and Ampère equations with coil interactions using a single block-sparse solver.

This method:
1. Assembles the combined block matrix via `combine_Au_and_ΔGS_sparse_matrices`.
2. Constructs a unified linear system for `ue_para` and `ψ_self`.
3. Performs Picard iteration to update:
   - `RP.plasma.ue_para`
   - `RP.fields.ψ_self`
   - `RP.fields.Eϕ_self`
   - External coil currents and resulting magnetic fields.

# Arguments
- `RP::RAPID{FT}`: Simulation state object, modified in place.
- `tolerance::FT=1e-3`: Convergence tolerance for the Picard iteration.
- `max_iter::Int=10`: Maximum number of Picard iterations.
- `relaxation_w::FT=0.5`: Relaxation weight for boundary ψ updates.

# Returns
- `RP::RAPID{FT}`: The updated simulation object with new plasma and field values.
"""
function solve_combined_momentum_Ampere_equations_with_coils!(
        RP::RAPID{FT};
        tolerance::FT = 1.0e-6,
        max_iter::Int = 10,
        relaxation_w::FT = 0.5
    ) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "solve_combined_momentum_Ampere_equations_with_coils!" begin
        # Aliases for readability
        pla = RP.plasma
        F = RP.fields
        OP = RP.operators
        G = RP.G
        flags = RP.flags
        dt = RP.dt
        csys = RP.coil_system


        # TODO: diffusion term

        # Physical constants
        @unpack ee, me, μ0, qe = RP.config.constants

        θimp = FT(1.0)  # Explicit(=0), Crank-Nicholson(=0.5), Backward Euler(=1)

        # Factor for EM drive contribution
        # derived from [E_para_EM = -qe/me * (ψ^(n+1) - ψ^(n))/(R*Δt)  = -facEM * ((ψ^(n+1) - ψ^(n)))/Δt]
        facEM = (qe / me) * (F.bϕ ./ G.R2D)

        # 1. calculate accel_para_tilde using the information at the current time step
        accel_para_tilde = zeros(FT, G.NR, G.NZ) # Initialize acceleration field

        # pressure gradient contribution: [-∇∥(ne*Te)/(me*ne)]
        if RP.flags.Include_ud_pressure_term
            accel_para_tilde .+= calculate_electron_acceleration_by_pressure(RP)
        end

        # convection contribution:  (1-θ)*[-(𝐮⋅∇)u∥]
        if RP.flags.Include_ud_convec_term
            accel_para_tilde .+= (one(FT) - θimp) * calculate_electron_acceleration_by_convection(RP)
        end

        # Electric field contributions: [(qe/me)* (E∥_ext + E∥_self_ES)]
        if flags.E_para_self_ES
            @. accel_para_tilde += qe / me * (F.E_para_ext + F.E_para_self_ES)
        else
            @. accel_para_tilde += qe / me * (F.E_para_ext)
        end

        # Effective electron collision frequency
        νe_eff = pla.ν_en_mom + pla.ν_en_iz + pla.ν_ei_eff

        @. accel_para_tilde += (
            facEM / dt * F.ψ_self
                - (one(FT) - θimp) * νe_eff * pla.ue_para
                + pla.ν_ei_eff * pla.ui_para
        )

        A_u = OP.II + spdiagm(@views dt * θimp * νe_eff[:])
        if flags.Include_ud_convec_term
            A_u += dt * θimp * (OP.𝐮∇.matrix)
        end

        # Calculate Rue_ei (electron-ion momentum exchange rate) - first part (n-th step)
        if RP.flags.Coulomb_Collision
            @. pla.Rue_ei = pla.ν_ei_eff * (pla.ui_para - (one(FT) - θimp) * pla.ue_para)
        end


        # Toroidal current density Jϕ @ t=(n-th step)
        Jϕ_pla_0 = @. (qe * pla.ne * pla.ue_para + pla.ni * (ee * pla.Zeff) * pla.ui_para) * F.bϕ


        # 6. Initial guess for ψ_self using θ-implicit scheme with extrapolated Eϕ_self
        # Predict Eϕ_self(n+1) by linear extrapolation: 2*E(n) - E(n-1)
        Eϕ_self_np1_pred = @. FT(2.0) * F.Eϕ_self - FT(1.0) * F.Eϕ_self_prev
        # Apply θ-weighting: (1-θ)*E(n) + θ*E(n+1_predicted)
        new_ψ_self_k = @. F.ψ_self - dt * G.R2D * ((one(FT) - θimp) * F.Eϕ_self + θimp * Eϕ_self_np1_pred)

        # Prepare Picard iteration for coupled system
        F.Eϕ_self_prev .= F.Eϕ_self # Store previous Eϕ_self for self-consistency
        old_ψ_self = copy(F.ψ_self) # Store old ψ_self for convergence checking


        ue_para_k = zeros(FT, G.NR, G.NZ) # Initialize ue_para for iterationo
        ue_para_kp1 = zeros(FT, G.NR, G.NZ) # Initialize ue_para for iterationo
        new_ψ_self_kp1 = zeros(FT, G.NR, G.NZ) # Initialize next ψ_self for iteration

        new_coils_I_k = zeros(FT, csys.n_total) # Initialize coil currents for iteration
        coil_flux_change_by_plasma = zeros(FT, csys.n_total)

        RHS_u = zeros(FT, G.NR, G.NZ) # preallocate reusable RHS related to u
        RHS_ψ = zeros(FT, G.NR, G.NZ) # preallocate reusable RHS relatedl to ψ
        Jϕ_pla_k = zeros(FT, G.NR, G.NZ) # Initialize Jϕ for iteration

        # Prepare coil_system for current calculation
        if (RP.dt != csys.Δt || θimp != csys.θimp)
            # If the time step or implicit factor have changed, recalculate the coil system matrices
            csys.Δt = RP.dt
            csys.θimp = θimp
            calculate_circuit_matrices!(csys)
        end

        A_u_ψ = combine_Au_and_ΔGS_sparse_matrices(RP, A_u, OP.ΔGS.matrix)
        @. RHS_u = pla.ue_para + dt * accel_para_tilde - facEM * new_ψ_self_k
        @views ue_para_k[:] .= A_u \ RHS_u[:] # Solve for ue_para at (k)-th step

        iter = 1
        converged = false
        while true
            # Step #1: Calculate ue_para, Jphi, coils according to new_psi_self_k
            @. Jϕ_pla_k = (qe * pla.ne * ue_para_k + pla.ni * (ee * pla.Zeff) * pla.ui_para) * F.bϕ


            if csys.n_total > 0
                Mcp_dIpla = 2π * csys.Green_grid2coils * (Jϕ_pla_k[:] .- Jϕ_pla_0[:]) * G.dR * G.dZ

                if flags.convec
                    # TODO: Is this part needed? Grid is not moving, so plasma movement should not affect coil currents?
                    Ipla = @. (θimp * Jϕ_pla_k + (1 - θimp) * Jϕ_pla_0) * G.dR * G.dZ
                    pla_displacement_R = pla.ueR * dt + 0.5 * pla.mean_aR_by_JxB * dt^2
                    pla_displacement_Z = pla.ueZ * dt + 0.5 * pla.mean_aZ_by_JxB * dt^2
                    # change rate of Mcp (mutual inductance between coils and plasma) due to plasma movement
                    Ipla_dMcp = 2π * (
                        csys.dGreen_dRg_grid2coils * (Ipla[:] .* pla_displacement_R[:]) +
                            csys.dGreen_dZg_grid2coils * (Ipla[:] .* pla_displacement_Z[:])
                    )

                    # grad_Ipla_R, grad_Ipla_Z = Cal_grad_of_scalar_F(reshape(Ipla, size(R2D)))
                    # dIpla_by_conv = pla_displacement_R .* grad_Ipla_R + pla_displacement_Z .* grad_Ipla_Z
                    # Mcp_dIpla_by_conv = 2π * coils.G_grid2coil * dIpla_by_conv[:]
                    Mcp_dIpla_by_conv = 0
                else
                    Ipla_dMcp = 0
                    Mcp_dIpla_by_conv = 0
                end

                @. coil_flux_change_by_plasma = Mcp_dIpla + Ipla_dMcp + Mcp_dIpla_by_conv

                circuit_rhs = calculate_LR_circuit_rhs_by_coils(csys, RP.time_s) - coil_flux_change_by_plasma
                new_coils_I_k = csys.inv_A_LR_circuit * circuit_rhs  # valid if "dt" is constant
            end

            # Step #2: Update Boundary psi by both plasma and coils currents using Green's function
            new_ψ_self_kp1_at_BDY = (G.Green_inWall2bdy * Jϕ_pla_k[G.nodes.in_wall_nids]) * G.dR * G.dZ
            if csys.n_total > 0
                new_ψ_self_kp1_at_BDY .+= csys.Green_coils2bdy * new_coils_I_k
            end

            #  update (k+1)-th boudary psi with some relaxation
            @views new_ψ_self_kp1_at_BDY .= (
                relaxation_w * new_ψ_self_kp1_at_BDY
                    + (one(FT) - relaxation_w) * new_ψ_self_k[G.BDY_idx]
            )

            # Step #3: Set RHS of the implicit Ampere equation
            @. RHS_u = pla.ue_para + dt * accel_para_tilde

            @. RHS_ψ = -μ0 * G.R2D * pla.ni * ee * pla.Zeff * pla.ui_para * F.bϕ
            if csys.n_total > 0
                inside_Jϕ_coil_k = distribute_coil_currents_to_Jϕ(csys, RP.G; currents = new_coils_I_k)
                @. RHS_ψ += -μ0 * G.R2D * inside_Jϕ_coil_k
            end
            RHS_ψ[G.BDY_idx] .= new_ψ_self_kp1_at_BDY

            # Step #4: Solve the implicit Ampere equation
            @views RHS_u_ψ = vcat(RHS_u[:], RHS_ψ[:])
            sol = A_u_ψ \ RHS_u_ψ

            @views ue_para_kp1[:] .= sol[1:(G.NR * G.NZ)]
            @views new_ψ_self_kp1[:] .= sol[(G.NR * G.NZ + 1):end]


            # Step #5: Check if ψ solution is converged
            convergence_rate = norm(new_ψ_self_kp1 - new_ψ_self_k) / norm(new_ψ_self_k)
            if (convergence_rate < tolerance)
                # println("  ψ_self converged after $iter iterations! convergence_rate: $convergence_rate")
                converged = true
                break
            elseif iter >= max_iter
                converged = false
                println("  Warning: Picard iteration did not converge after $max_iter iterations at step=$(RP.step)")
                println("  Final change: $(norm(new_ψ_self_kp1 - new_ψ_self_k) / norm(new_ψ_self_k))")
                break
            else
                new_ψ_self_k .= new_ψ_self_kp1 # Update for next iteration
                ue_para_k .= ue_para_kp1
                iter += 1
            end
        end

        # 7. Final updates of electromagnetic fields
        F.ψ_self .= new_ψ_self_kp1
        pla.ue_para .= ue_para_kp1

        # Update self-consistent electric field: Eϕ = -∂ψ/∂t/R
        @. F.Eϕ_self = -(F.ψ_self - old_ψ_self) / (G.R2D * dt)

        # Complete the Rue_ei calculation with second part (n+1 step contribution)
        if RP.flags.Coulomb_Collision
            @. pla.Rue_ei += pla.ν_ei_eff * (-θimp * pla.ue_para)
        end

        @. pla.Jϕ = (pla.ne * qe * pla.ue_para + pla.ni * (ee * pla.Zeff) * pla.ui_para) * F.bϕ

        # Update coil currents
        if RP.coil_system.n_total > 0
            csys.time_s += csys.Δt
            set_all_currents!(csys, new_coils_I_k)
        end

        # Update magnetic fields from ψ_self
        calculate_B_from_ψ!(G, F.ψ_self, F.BR_self, F.BZ_self)

        return RP
    end # @timeit
end


"""
    update_uMHD_by_global_JxB_force!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update plasma MHD velocities using global JxB force balance.

This function computes the global toroidal force balance for closed flux surfaces
and updates the plasma mean velocities by applying JxB accelerations. The method
enforces momentum conservation across the entire plasma volume when magnetic
confinement creates closed flux surfaces.

# Physics
- Calculates global JxB forces on the plasma
- Applies momentum conservation constraints
- Updates plasma mean velocities (uMHD) based on electromagnetic forces
- Only active when closed flux surfaces are present

# Arguments
- `RP::RAPID{FT}`: RAPID simulation object containing plasma state and fields
"""
function update_uMHD_by_global_JxB_force!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    # Check if we have closed flux surfaces
    if !isempty(RP.flf.closed_surface_nids)

        @unpack mi, me = RP.config.constants
        nids = vcat(RP.G.nodes.on_wall_nids, RP.G.nodes.in_wall_nids)
        pla = RP.plasma
        F = RP.fields

        # Calculate total plasma mass by integrating over volume
        sum_plasma_mass = sum(@. (mi * pla.ni[nids] + me * pla.ne[nids]) * RP.G.inVol2D[nids])
        sum_JxB_R = sum(@. pla.Jϕ[nids] * F.BZ[nids] * RP.G.inVol2D[nids])
        sum_JxB_Z = sum(@. -pla.Jϕ[nids] * F.BR[nids] * RP.G.inVol2D[nids])

        # Calculate mean accelerations
        if sum_plasma_mass > zero(FT)
            pla.mean_aR_by_JxB[nids] .= sum_JxB_R / sum_plasma_mass
            pla.mean_aZ_by_JxB[nids] .= sum_JxB_Z / sum_plasma_mass

            # TODO: Is this explicit way stable? Should we use the mean_aR and mean_aZ implicitly in some functions?
            pla.uMHD_R[nids] .+= pla.mean_aR_by_JxB[nids] * RP.dt
            pla.uMHD_Z[nids] .+= pla.mean_aZ_by_JxB[nids] * RP.dt
        end
    end

    return RP
end
