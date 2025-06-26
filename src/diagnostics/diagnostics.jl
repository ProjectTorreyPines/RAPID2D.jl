using SimpleUnPack

"""
    measure_snap0D!(RP::RAPID{FT}, snap0D::Snapshot0D{FT}) where {FT<:AbstractFloat}

In-place measurement of 0D (scalar) diagnostics into an existing Snapshot0D object.
Updates the provided snap0D with current simulation state and volume-averaged quantities.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to measure
- `snap0D::Snapshot0D{FT}`: Pre-allocated snapshot object to fill with measurements
"""
function measure_snap0D!(RP::RAPID{FT}, snap0D::Snapshot0D{FT}) where {FT<:AbstractFloat}

    pla = RP.plasma
    F = RP.fields
    inVol2D = RP.G.inVol2D  # alias for convenience

    @unpack ee, me, mi = RP.config.constants  # Unpack constants for convenience

    # Store metadata
    snap0D.time_s = RP.time_s
    snap0D.dt = RP.dt
    snap0D.step = RP.step

    # total number of electrons and ions in the device volume
    Ne2D = pla.ne .* inVol2D
    Ni2D = pla.ni .* inVol2D

    total_Ne = sum(Ne2D)
    total_Ni = sum(Ni2D)


    # Basic electron quantities
    snap0D.ne = total_Ne / RP.G.device_inVolume
    snap0D.ne_max = maximum(pla.ne)
    snap0D.ue_para = sum(@. pla.ue_para * Ne2D) / total_Ne
    snap0D.Te_eV = sum(@. pla.Te_eV * Ne2D) / total_Ne
    snap0D.Ke_eV = sum(@. (1.5 * pla.Te_eV + (0.5 * me * pla.ue_para^2 )/ ee) * Ne2D) / total_Ne

    # Ion quantities
    snap0D.ni = total_Ni / RP.G.device_inVolume
    snap0D.ni_max = maximum(pla.ni)
    snap0D.ui_para = sum(@. pla.ui_para * Ni2D) / total_Ni
    snap0D.Ti_eV = sum(@. pla.Ti_eV * Ni2D) / total_Ni
    snap0D.Ki_eV = sum(@. (1.5 * pla.Ti_eV + (0.5 * mi * pla.ui_para^2 ) / ee) * Ni2D) / total_Ni

    # Toroidal current
    snap0D.I_tor = sum( pla.Jϕ * RP.G.dR * RP.G.dZ)

    # Electric fields (density-weighted averages)
    snap0D.Epara_tot = sum(@. F.E_para_tot * Ne2D) / total_Ne
    snap0D.Epara_ext = sum(@. F.E_para_ext * Ne2D) / total_Ne
    snap0D.Epara_self_ES = sum(@. F.E_para_self_ES * Ne2D) / total_Ne
    snap0D.Epara_self_EM = sum(@. F.E_para_self_EM * Ne2D) / total_Ne

    # Transport quantities
    snap0D.abs_ue_para_RZ = sum(@. abs(pla.ue_para) * F.Bpol / F.Btot * Ne2D) / total_Ne
    snap0D.D_RZ = sum(@. sqrt(RP.transport.DRR^2 + RP.transport.DZZ^2) * Ne2D) / total_Ne

    # Gas quantities
    snap0D.n_H2_gas = sum(@. pla.n_H2_gas * inVol2D) / RP.G.device_inVolume
    snap0D.n_H2_gas_min = minimum(pla.n_H2_gas)

    # Electron collision frequencies
    if RP.flags.Atomic_Collision
        eRRC_iz = get_electron_RRC(RP, :Ionization)
        eRRC_mom = get_electron_RRC(RP, :Momentum)
        eRRC_Hα = get_electron_RRC(RP, :Halpha)

        snap0D.ν_en_iz = sum(@. pla.n_H2_gas * eRRC_iz  * Ne2D) / total_Ne
        snap0D.ν_en_mom = sum(@. pla.n_H2_gas * eRRC_mom  * Ne2D) / total_Ne
        snap0D.ν_en_Hα = sum(@. pla.n_H2_gas * eRRC_Hα * Ne2D) / total_Ne
    end

    if RP.flags.Coulomb_Collision
        # Coulomb collision frequency
        snap0D.ν_ei = sum(@. pla.ν_ei * Ne2D) / total_Ne
        snap0D.ν_ei_eff = sum(@. pla.ν_ei_eff * Ne2D) / total_Ne
        snap0D.ν_ii = sum(@. pla.ν_ii * Ni2D) / total_Ni
    end


    # CFL conditions (if adaptive timestepping is not used)
    if hasfield(typeof(RP), :Flag) && hasfield(typeof(RP.Flag), :Adapt_dt) && !RP.Flag.Adapt_dt
        # Placeholder for CFL calculation
        @warn "CFL condition calculation not yet implemented" maxlog=10
        # Would need: Cal_CFL_conditions() equivalent
    end

    # Source/loss rates
    Ntracker = RP.diagnostics.Ntracker
    snap0D.Ne_src_rate = Ntracker.cum0D_Ne_src / RP.config.snap0D_Δt_s
    snap0D.Ne_loss_rate = Ntracker.cum0D_Ne_loss / RP.config.snap0D_Δt_s
    snap0D.Ni_src_rate = Ntracker.cum0D_Ni_src / RP.config.snap0D_Δt_s
    snap0D.Ni_loss_rate = Ntracker.cum0D_Ni_loss / RP.config.snap0D_Δt_s

    # Growth rates
    # TODO: Check if this is the correct way to calculate growth rates
    prev_snap0D = RP.diagnostics.snaps0D[max(1, RP.diagnostics.tid_0D-1)]
    prev_total_Ne = prev_snap0D.ne * RP.G.device_inVolume
    snap0D.eGrowth_rate = log(FT(1) + Ntracker.cum0D_Ne_src / prev_total_Ne) / RP.config.snap0D_Δt_s
    try
        snap0D.eLoss_rate = -log(FT(1) - Ntracker.cum0D_Ne_loss / prev_total_Ne) / RP.config.snap0D_Δt_s
    catch
        snap0D.eLoss_rate = -Inf
    end

    # Alternative growth rates
    snap0D.growth_rate2 = snap0D.Ne_src_rate / total_Ne
    snap0D.loss_rate2 = snap0D.Ne_loss_rate / total_Ne
    # Power balance calculations

    ePowers, iPowers = RP.plasma.ePowers, RP.plasma.iPowers
    # Calculate density-weighted averages for electron power components
    if total_Ne > 0
        snap0D.Pe_diffu = sum(@. ePowers.diffu * Ne2D) / total_Ne
        snap0D.Pe_conv = sum(@. ePowers.conv * Ne2D) / total_Ne
        snap0D.Pe_drag = sum(@. ePowers.drag * Ne2D) / total_Ne
        snap0D.Pe_iz = sum(@. ePowers.iz * Ne2D) / total_Ne
        snap0D.Pe_exc = sum(@. ePowers.exc * Ne2D) / total_Ne
        snap0D.Pe_dilution = sum(@. ePowers.dilution * Ne2D) / total_Ne
        snap0D.Pe_equi = sum(@. ePowers.equi * Ne2D) / total_Ne
        snap0D.Pe_heat = sum(@. ePowers.heat * Ne2D) / total_Ne
        snap0D.Pe_tot = sum(@. ePowers.tot * Ne2D) / total_Ne
    end

    # Calculate density-weighted averages for ion power components
    if total_Ni > 0
        snap0D.Pi_tot = sum(@. iPowers.tot * Ni2D) / total_Ni
        snap0D.Pi_atomic = sum(@. iPowers.atomic * Ni2D) / total_Ni
        snap0D.Pi_equi = sum(@. iPowers.equi * Ni2D) / total_Ni
    end

    # Plasma center tracking
    snap0D.ne_cen_R = sum(@. RP.G.R2D * Ne2D) / total_Ne
    snap0D.ne_cen_Z = sum(@. RP.G.Z2D * Ne2D) / total_Ne

    snap0D.J_cen_R = sum(@. RP.G.R2D * pla.Jϕ * inVol2D) / sum(pla.Jϕ .* inVol2D)
    snap0D.J_cen_Z = sum(@. RP.G.Z2D * pla.Jϕ * inVol2D) / sum(pla.Jϕ .* inVol2D)

    snap0D.ueR = sum(@. pla.ueR * Ne2D) / total_Ne
    snap0D.ueZ = sum(@. pla.ueZ * Ne2D) / total_Ne

    snap0D.aR_by_JxB = sum(@. pla.mean_aR_by_JxB * Ne2D) / total_Ne
    snap0D.aZ_by_JxB = sum(@. pla.mean_aZ_by_JxB * Ne2D) / total_Ne

    # Control system (if enabled)
    if hasfield(typeof(RP), :Flag) && hasfield(typeof(RP.Flag), :Control) && hasfield(typeof(RP.Flag.Control), :state) && RP.Flag.Control.state
        @warn "Control system diagnostics not yet implemented" maxlog=10
        # Would need PID controller and control field calculations
    end

    # measure magnetic energies by toroidal currents
    snap0D.tot_W_mag = zero(FT)
    if RP.flags.Ampere
        F_plasma = solve_Ampere_equation(RP; plasma=true, coils=false)
        W_mag_by_plasma_without_coil =  0.5 * sum(@. RP.plasma.Jϕ * F_plasma.ψ_self / RP.G.R2D * RP.G.inVol2D)
        snap0D.self_inductance_plasma = 2.0 * W_mag_by_plasma_without_coil / (snap0D.I_tor^2 + eps(FT))

        # plasma's contribution to magnetic energy [J]
        # NOTE: must use total ψ_self by both plasma and coils
        # tot_W_mag =(1/2)*∫Jϕ⋅Aϕ dV
        snap0D.tot_W_mag += 0.5 * sum(@. RP.plasma.Jϕ * RP.fields.ψ_self / RP.G.R2D * RP.G.inVol2D)

        # Ohmic dissipation [W]
        ν_eff = @. pla.ν_ei_eff + pla.ν_en_mom + pla.ν_en_iz
        @unpack me, ee = RP.config.constants

        σ_conductivity = @. pla.ne * ee^2 / (me * ν_eff)

        valid_idx = findall(pla.ne .> 1e3 .&& ν_eff .> 0.0)

        if isempty(valid_idx)
            snap0D.η_resistivity = Inf
            snap0D.resistance_plasma = Inf
            snap0D.tot_P_ohm_plasma = 0.0
        else
            snap0D.η_resistivity = 1.0 / (sum(@views @. σ_conductivity[valid_idx] * Ne2D[valid_idx]) / total_Ne)
            snap0D.resistance_plasma = 1.0/sum(@views @. σ_conductivity[valid_idx]*RP.G.dR * RP.G.dZ/ (2π * RP.G.R2D[valid_idx]))
            snap0D.tot_P_ohm_plasma = snap0D.I_tor^2 * snap0D.resistance_plasma # P_ohm = I^2*R
        end
    end

    if RP.coil_system.n_total > 0
        csys = RP.coil_system
        coils = csys.coils

        snap0D.coils_I = coils.current
        snap0D.coils_V_ext = get_all_voltages_at_time(csys)

        # magnetic energy by coils [J]
        Φ_coils = ( csys.mutual_inductance*coils.current
                    +2π*(csys.Green_grid2coils *pla.Jϕ[:])*RP.G.dR*RP.G.dZ
                )
        snap0D.tot_W_mag += 0.5*coils.current'*Φ_coils

        # Ohmic coil dissipation [W]
        snap0D.tot_P_input_coils = sum(coils.current .* get_all_voltages_at_time(csys))
        snap0D.tot_P_ohm_coils = sum(@. coils.current^2 * coils.resistance)
    end

    return RP
end

"""
    measure_snap0D(RP::RAPID{FT}) where {FT<:AbstractFloat}

Measure 0D (scalar) diagnostics and return a new Snapshot0D object.
Creates a new snapshot and fills it with volume-averaged quantities from the simulation.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to measure

# Returns
- `Snapshot0D{FT}`: New snapshot object containing measured diagnostics
"""
function measure_snap0D(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Create a new Snapshot0D object
    snap0D = Snapshot0D{FT}()

    # Measure and fill the snapshot
    measure_snap0D!(RP, snap0D)

    return snap0D
end


"""
    measure_snap2D!(RP::RAPID{FT}, snap2D::Snapshot2D{FT}) where {FT<:AbstractFloat}

In-place measurement of 2D diagnostics into an existing Snapshot2D object.
Updates the provided snap2D with current simulation state and 2D field distributions.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to measure
- `snap2D::Snapshot2D{FT}`: Pre-allocated snapshot object to fill with measurements
"""
function measure_snap2D!(RP::RAPID{FT}, snap2D::Snapshot2D{FT}) where {FT<:AbstractFloat}

    @assert snap2D.dims_RZ == (RP.G.NR, RP.G.NZ) "Snapshot dimensions must match RAPID grid dimensions"

    # Get current index
    pla = RP.plasma                  # alias for convenience
    F = RP.fields                    # alias for convenience
    tp = RP.transport                 # alias for convenience

    @unpack ee, me, mi = RP.config.constants  # Unpack constants for convenience

    # Store metadata
    snap2D.step = RP.step
    snap2D.dt = RP.dt
    snap2D.time_s = RP.time_s

    # Basic plasma quantities
    snap2D.ne .= pla.ne
    snap2D.Te_eV .= pla.Te_eV

    # Transport coefficients
    snap2D.Dpara .= tp.Dpara
    snap2D.ue_para .= pla.ue_para

    # Calculate derived quantities
    @. snap2D.u_pol = sqrt(pla.ueR^2 + pla.ueZ^2)
    @. snap2D.D_pol = sqrt(tp.DRR^2 + tp.DZZ^2)

    # Magnetic fields
    snap2D.BR .= F.BR
    snap2D.BZ .= F.BZ
    snap2D.B_pol .= F.Bpol
    snap2D.BR_self .= F.BR_self
    snap2D.BZ_self .= F.BZ_self

    # Electric fields
    snap2D.E_para_tot .= F.E_para_tot
    snap2D.E_para_ext .= F.E_para_ext
    snap2D.Epol_self .= F.Epol_self
    snap2D.Eϕ_self .= F.Eϕ_self

    # Loop voltages
    snap2D.LV_ext .= F.LV_ext
    @. snap2D.LV_tot = F.LV_ext + 2π*RP.G.R2D * F.Eϕ_self

    # Calculate ExB drift magnitude
    @. snap2D.mean_ExB_pol = sqrt(pla.mean_ExB_R^2 + pla.mean_ExB_Z^2)

    # Source/loss rates from cumulative trackers
    Ntracker = RP.diagnostics.Ntracker
    @. snap2D.Ne_src_rate = Ntracker.cum2D_Ne_src / RP.config.snap2D_Δt_s
    @. snap2D.Ne_loss_rate = Ntracker.cum2D_Ne_loss / RP.config.snap2D_Δt_s
    @. snap2D.Ni_src_rate = Ntracker.cum2D_Ni_src / RP.config.snap2D_Δt_s
    @. snap2D.Ni_loss_rate = Ntracker.cum2D_Ni_loss / RP.config.snap2D_Δt_s

    # Current densities (parallel current components)
    snap2D.Jϕ .= pla.Jϕ
    @. snap2D.J_para = ee * (pla.ni * pla.ui_para - pla.ne * pla.ue_para)

    # Poloidal flux
    snap2D.ψ .= F.ψ
    snap2D.ψ_ext .= F.ψ_ext
    snap2D.ψ_self .= F.ψ_self

    # Electron velocity components
    snap2D.ueR .= pla.ueR
    snap2D.ueϕ .= pla.ueϕ
    snap2D.ueZ .= pla.ueZ

    # Physics parameters
    tp = RP.transport
    snap2D.L_mixing .= tp.L_mixing
    snap2D.nc_para .= pla.nc_para
    snap2D.nc_perp .= pla.nc_perp

    snap2D.γ_shape_fac .= pla.γ_shape_fac

    # Ion quantities
    snap2D.ni .= pla.ni
    snap2D.ui_para .= pla.ui_para
    snap2D.uiR .= pla.uiR
    snap2D.uiϕ .= pla.uiϕ
    snap2D.uiZ .= pla.uiZ
    snap2D.Ti_eV .= pla.Ti_eV

    # MHD accelerations
    snap2D.mean_aR_by_JxB .= pla.mean_aR_by_JxB
    snap2D.mean_aZ_by_JxB .= pla.mean_aZ_by_JxB

    # Coulomb logarithm
    snap2D.lnΛ .= pla.lnΛ

    # Neutral gas
    snap2D.n_H2_gas .= pla.n_H2_gas

    # Calculate mean energies
    @. snap2D.Ke_eV = 1.5 * pla.Te_eV + 0.5 * me * pla.ue_para^2 / ee
    @. snap2D.Ki_eV = 1.5 * pla.Ti_eV + 0.5 * mi * pla.ui_para^2 / ee

    # Collision frequencies using RRC methods
    if RP.flags.Atomic_Collision
        eRRC_mom = get_electron_RRC(RP, :Momentum)
        eRRC_iz = get_electron_RRC(RP, :Ionization)
        eRRC_Halpha = get_electron_RRC(RP, :Halpha)

        @. snap2D.ν_en_mom = pla.n_H2_gas * eRRC_mom
        @. snap2D.ν_en_iz = pla.n_H2_gas * eRRC_iz
        @. snap2D.ν_en_Hα = pla.n_H2_gas * eRRC_Halpha
    end

    if RP.flags.Coulomb_Collision
        snap2D.ν_ei .= pla.ν_ei
        snap2D.ν_ei_eff .= pla.ν_ei_eff
        snap2D.ν_ii .= pla.ν_ii
    end

    # Store electron power components
    snap2D.Pe_tot .= pla.ePowers.tot
    snap2D.Pe_diffu .= pla.ePowers.diffu
    snap2D.Pe_conv .= pla.ePowers.conv
    snap2D.Pe_drag .= pla.ePowers.drag
    snap2D.Pe_dilution .= pla.ePowers.dilution
    snap2D.Pe_iz .= pla.ePowers.iz
    snap2D.Pe_exc .= pla.ePowers.exc

    # Store ion power components
    snap2D.Pi_tot .= pla.iPowers.tot
    snap2D.Pi_atomic .= pla.iPowers.atomic
    snap2D.Pi_equi .= pla.iPowers.equi

    ν_eff = @. pla.ν_ei_eff + pla.ν_en_mom + pla.ν_en_iz
    @. snap2D.η_resistivity = (me * ν_eff) / (pla.ne * ee^2)

    # Handle near-zero density regions
    # near_zero_density_mask = pla.ne .< 1.0  # Find indices where density is effectively zero
    # snap2D.Te_eV[near_zero_density_mask] .= NaN
    # snap2D.ue_para[near_zero_density_mask] .= NaN
    # snap2D.Ke_eV[near_zero_density_mask] .= NaN


    return RP
end

"""
    measure_snap2D(RP::RAPID{FT}) where {FT<:AbstractFloat}

Measure 2D diagnostics and return a new Snapshot2D object.
Creates a new snapshot with correct grid dimensions and fills it with 2D field distributions.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to measure

# Returns
- `Snapshot2D{FT}`: New snapshot object containing measured 2D diagnostics
"""
function measure_snap2D(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Create a new Snapshot2D object with the correct dimensions
    snap2D = Snapshot2D{FT}(dims_RZ = (RP.G.NR, RP.G.NZ) )

    # Measure and fill the snapshot
    measure_snap2D!(RP, snap2D)

    return snap2D
end


"""
    reset_Ntracker_cumulative_0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Reset cumulative 0D particle number trackers to zero.
Called after each 0D snapshot to restart accumulation for the next diagnostic interval.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance with trackers to reset
"""
function reset_Ntracker_cumulative_0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Reset cumulative trackers
    RP.diagnostics.Ntracker.cum0D_Ne_src = zero(FT)
    RP.diagnostics.Ntracker.cum0D_Ne_loss = zero(FT)
    RP.diagnostics.Ntracker.cum0D_Ni_src = zero(FT)
    RP.diagnostics.Ntracker.cum0D_Ni_loss = zero(FT)

    return RP
end

"""
    reset_Ntracker_cumulative_2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Reset cumulative 2D particle number trackers to zero.
Called after each 2D snapshot to restart accumulation for the next diagnostic interval.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance with trackers to reset
"""
function reset_Ntracker_cumulative_2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Reset cumulative trackers
    fill!(RP.diagnostics.Ntracker.cum2D_Ne_src, zero(FT))
    fill!(RP.diagnostics.Ntracker.cum2D_Ne_loss, zero(FT))
    fill!(RP.diagnostics.Ntracker.cum2D_Ni_src, zero(FT))
    fill!(RP.diagnostics.Ntracker.cum2D_Ni_loss, zero(FT))

    return RP
end


"""
    update_snaps0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the 0D snapshots vector with a new measurement.
Increments the time index and either fills a pre-allocated snapshot or creates a new one.
Automatically resets cumulative trackers after measurement.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to update diagnostics for
"""
function update_snaps0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # update time index for 0D snapshots
    RP.diagnostics.tid_0D += 1

    tid_0D = RP.diagnostics.tid_0D
    if tid_0D <= length(RP.diagnostics.snaps0D)
        # If we already assigned a preallocated snapshots, update it
        snap0D = RP.diagnostics.snaps0D[tid_0D]
        measure_snap0D!(RP, snap0D)
    else
        # Otherwise, create and measure a new snapshot
        snap0D = measure_snap0D(RP)
        # Append the new snapshot to the list
        push!(RP.diagnostics.snaps0D, snap0D)
    end

    # Reset cumulative trackers after snapshot
    reset_Ntracker_cumulative_0D!(RP)

    return RP
end


"""
    update_snaps2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the 2D snapshots vector with a new measurement.
Increments the time index and either fills a pre-allocated snapshot or creates a new one.
Automatically resets cumulative trackers after measurement.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to update diagnostics for
"""
function update_snaps2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # update time index for 2D snapshots
    RP.diagnostics.tid_2D += 1

    tid_2D = RP.diagnostics.tid_2D
    if tid_2D <= length(RP.diagnostics.snaps2D)
        # If we already assigned a preallocated snapshots, update it
        snap2D = RP.diagnostics.snaps2D[tid_2D]
        measure_snap2D!(RP, snap2D)
    else
        # Otherwise, create and measure a new snapshot
        snap2D = measure_snap2D(RP)
        # Append the new snapshot to the list
        push!(RP.diagnostics.snaps2D, snap2D)
    end

    # Reset cumulative trackers after snapshot
    reset_Ntracker_cumulative_2D!(RP)

    return RP
end

function check_energy_conservation(snaps0D::Vector{Snapshot0D{FT}}) where FT
    times = snaps0D.time_s

    trapz = (P, t) -> sum(0.5*(P[2:end]+P[1:end-1]) .* diff(t))

    W_input = trapz(snaps0D.tot_P_input_coils, times)
    W_ohm_coil = trapz(snaps0D.tot_P_ohm_coils, times)
    W_ohm_plasma = trapz(snaps0D.tot_P_ohm_plasma, times)

    ΔW_mag = snaps0D.tot_W_mag[end] - snaps0D.tot_W_mag[1]

    # Input = Losses + Stored Energy
    energy_balance = W_input - (W_ohm_coil + W_ohm_plasma + ΔW_mag )

    @printf("Energy Conservation Analysis:\n")
    @printf("=================================\n")
    @printf("Input energy (coils):    %12.7f J\n", W_input)
    @printf("Magnetic energy change:  %12.7f J\n", ΔW_mag)
    @printf("Coil ohmic losses:       %12.7f J\n", W_ohm_coil)
    @printf("Plasma ohmic losses:     %12.7f J\n", W_ohm_plasma)
    @printf("---------------------------------\n")
    @printf("Energy balance:          %12.7f J (should be ≈ 0)\n", energy_balance)
    @printf("Relative error:          %12.2f%%\n", abs(energy_balance/W_input)*100)

    return (;
         W_input,
         W_ohm_coil,
         W_ohm_plasma,
         ΔW_mag,
         energy_balance,
         rel_error = abs(energy_balance/W_input)*100
    )
end