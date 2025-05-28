using SimpleUnPack

"""
    measure_snap0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Measure snap0D diagnostics in the RAPID object, which are 0D (scalar) quantities averaged over the volume.
This function is analogous to MATLAB's Measure_snap1D().

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to measure
"""
function measure_snap0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Check timing interval (like MATLAB's nearestMultiple logic)
    nearest_multiple = round(RP.time_s / RP.config.snap0D_Δt_s) * RP.config.snap0D_Δt_s
    if abs(RP.time_s - nearest_multiple) >= 0.1 * RP.dt
        return nothing  # Skip if not at the right interval
    end

    snap0D = RP.diagnostics.snap0D  # alias for convenience
    pla = RP.plasma
    inVol2D = RP.G.inVol2D  # alias for convenience

    # Get current index
    idx = snap0D.idx
    # Check if we've reached the end of the array
    if idx > length(snap0D.time_s)
        @warn "Reached end of pre-allocated 0D diagnostic arrays at index $idx"
        return nothing
    end

    @unpack ee, me, mi = RP.config.constants  # Unpack constants for convenience

    # Store metadata
    snap0D.time_s[idx] = RP.time_s
    snap0D.dt[idx] = RP.dt
    snap0D.step[idx] = RP.step

    # total number of electrons and ions in the device volume
    total_Ne = sum(@. pla.ne * inVol2D)
    total_Ni = sum(@. pla.ni * inVol2D)

    # Basic electron quantities
    snap0D.ne[idx] = total_Ne / RP.G.device_inVolume
    snap0D.ne_max[idx] = maximum(pla.ne)
    snap0D.ue_para[idx] = sum(@. pla.ue_para * pla.ne * inVol2D) / total_Ne
    snap0D.Te_eV[idx] = sum(@. pla.Te_eV * pla.ne * inVol2D) / total_Ne
    snap0D.mean_eErg_eV[idx] = sum(@. (1.5 * pla.Te_eV + 0.5 * me * pla.ue_para.^2 )/ ee * pla.ne * inVol2D) / total_Ne

    # Ion quantities
    snap0D.ni[idx] = total_Ni / RP.G.device_inVolume
    snap0D.ni_max[idx] = maximum(pla.ni)
    snap0D.ui_para[idx] = sum(@. pla.ui_para * pla.ni * inVol2D) / total_Ni
    snap0D.Ti_eV[idx] = sum(@. pla.Ti_eV * pla.ni * inVol2D) / total_Ni
    snap0D.mean_iErg_eV[idx] = sum(@. (1.5 * pla.Ti_eV + 0.5 * mi * pla.ui_para.^2 ) / ee * pla.ni * inVol2D) / total_Ni

    # Toroidal current
    snap0D.I_tor[idx] = sum( pla.Jϕ * RP.G.dR * RP.G.dZ)

    # Electric fields (density-weighted averages)
    snap0D.Epara_tot[idx] = sum(@. RP.fields.E_para_tot * pla.ne * inVol2D) / total_Ne
    snap0D.Epara_ext[idx] = sum(@. RP.fields.E_para_ext * pla.ne * inVol2D) / total_Ne
    snap0D.Epara_self_ES[idx] = sum(@. RP.fields.E_para_self_ES * pla.ne * inVol2D) / total_Ne
    snap0D.Epara_self_EM[idx] = sum(@. RP.fields.E_para_self_EM * pla.ne * inVol2D) / total_Ne

    # Transport quantities
    snap0D.abs_ue_para_RZ[idx] = sum(@. abs(pla.ue_para) * RP.fields.Bpol / RP.fields.Btot * pla.ne * inVol2D) / total_Ne
    snap0D.D_RZ[idx] = sum(@. sqrt.(RP.transport.DRR.^2 + RP.transport.DZZ.^2) * pla.ne * inVol2D) / total_Ne

    # Gas quantities
    snap0D.n_H2_gas[idx] = sum(@. pla.n_H2_gas * inVol2D) / RP.G.device_inVolume
    snap0D.n_H2_gas_min[idx] = minimum(pla.n_H2_gas)

    # Electron collision frequencies
    eRRC_iz = get_electron_RRC(RP, :Ionization)
    eRRC_mom = get_electron_RRC(RP, :Momentum)
    eRRC_Hα = get_electron_RRC(RP, :Halpha)

    snap0D.ν_iz[idx] = sum(@. pla.n_H2_gas * eRRC_iz  * pla.ne * inVol2D) / total_Ne
    snap0D.ν_mom[idx] = sum(@. pla.n_H2_gas * eRRC_mom  * pla.ne * inVol2D) / total_Ne
    snap0D.ν_Hα[idx] = sum(@. pla.n_H2_gas * eRRC_Hα * pla.ne * inVol2D) / total_Ne
    snap0D.ν_ei[idx] = sum(@. pla.ν_ei * pla.ne *  inVol2D) / total_Ne


    # CFL conditions (if adaptive timestepping is not used)
    if hasfield(typeof(RP), :Flag) && hasfield(typeof(RP.Flag), :Adapt_dt) && !RP.Flag.Adapt_dt
        # Placeholder for CFL calculation
        @warn "CFL condition calculation not yet implemented" maxlog=10
        # Would need: Cal_CFL_conditions() equivalent
    end

    # Source/loss rates
    if hasfield(typeof(RP.diagnostics.tracker), :cum1D_Ne_src)
        snap0D.Ne_src_rate[idx] = RP.diagnostics.tracker.cum1D_Ne_src / RP.config.snap0D_Δt_s
        snap0D.Ne_loss_rate[idx] = RP.diagnostics.tracker.cum1D_Ne_loss / RP.config.snap0D_Δt_s

        # Growth rates
        prev_N = idx > 1 ? snap0D.ne[idx-1] * RP.G.device_inVolume : snap0D.ne[idx] * RP.G.device_inVolume
        snap0D.eGrowth_rate[idx] = log(1 + RP.diagnostics.tracker.cum1D_Ne_src / prev_N) / RP.config.snap0D_Δt_s
        snap0D.eLoss_rate[idx] = -log(1 - RP.diagnostics.tracker.cum1D_Ne_loss / prev_N) / RP.config.snap0D_Δt_s

        # Alternative growth rates
        snap0D.growth_rate2[idx] = snap0D.Ne_src_rate[idx] / (snap0D.ne[idx] * RP.G.device_inVolume)
        snap0D.loss_rate2[idx] = snap0D.Ne_loss_rate[idx] / (snap0D.ne[idx] * RP.G.device_inVolume)
    end

    if hasfield(typeof(RP.diagnostics.tracker), :cum1D_Ni_src)
        snap0D.Ni_src_rate[idx] = RP.diagnostics.tracker.cum1D_Ni_src / RP.config.snap0D_Δt_s
        snap0D.Ni_loss_rate[idx] = RP.diagnostics.tracker.cum1D_Ni_loss / RP.config.snap0D_Δt_s
    end

    # Power balance calculations
    # Update power calculations first
    update_electron_heating_powers!(RP)
    update_ion_heating_powers!(RP)

    ePowers, iPowers = RP.plasma.ePowers, RP.plasma.iPowers

    # Calculate density-weighted averages for electron power components
    if total_Ne > 0
        snap0D.P_diffu[idx] = sum(@. ePowers.diffu * pla.ne * inVol2D) / total_Ne
        snap0D.P_conv[idx] = sum(@. ePowers.conv * pla.ne * inVol2D) / total_Ne
        snap0D.P_drag[idx] = sum(@. ePowers.drag * pla.ne * inVol2D) / total_Ne
        snap0D.P_iz[idx] = sum(@. ePowers.iz * pla.ne * inVol2D) / total_Ne
        snap0D.P_exc[idx] = sum(@. ePowers.exc * pla.ne * inVol2D) / total_Ne
        snap0D.P_dilution[idx] = sum(@. ePowers.dilution * pla.ne * inVol2D) / total_Ne
        snap0D.P_equi[idx] = sum(@. ePowers.equi * pla.ne * inVol2D) / total_Ne
        snap0D.P_heat[idx] = sum(@. ePowers.heat * pla.ne * inVol2D) / total_Ne
        snap0D.P_tot[idx] = sum(@. ePowers.tot * pla.ne * inVol2D) / total_Ne
    end

    # Calculate density-weighted averages for ion power components
    if total_Ni > 0
        snap0D.Pi_tot[idx] = sum(@. iPowers.tot * pla.ni * inVol2D) / total_Ni
        snap0D.Pi_atomic[idx] = sum(@. iPowers.atomic * pla.ni * inVol2D) / total_Ni
        snap0D.Pi_equi[idx] = sum(@. iPowers.equi * pla.ni * inVol2D) / total_Ni
    end

    # Plasma center tracking
    snap0D.ne_cen_R[idx] = sum(@. pla.ne * RP.G.R2D) / sum(pla.ne)
    snap0D.ne_cen_Z[idx] = sum(@. pla.ne * RP.G.Z2D) / sum(pla.ne)

    snap0D.J_cen_R[idx] = sum(@. pla.Jϕ * RP.G.R2D) / sum(pla.Jϕ)
    snap0D.J_cen_Z[idx] = sum(@. pla.Jϕ * RP.G.Z2D) / sum(pla.Jϕ)

    # Control system (if enabled)
    if hasfield(typeof(RP), :Flag) && hasfield(typeof(RP.Flag), :Control) && hasfield(typeof(RP.Flag.Control), :state) && RP.Flag.Control.state
        @warn "Control system diagnostics not yet implemented" maxlog=10
        # Would need PID controller and control field calculations
    end

    # Coil currents (if present)
    if hasfield(typeof(RP), :coils) && hasfield(typeof(RP.coils), :N) && RP.coils.N > 0
        if hasfield(typeof(snap0D), :I_coils) && snap0D.I_coils !== nothing
            snap0D.I_coils[:, idx] = RP.coils.I
        else
            @warn "Coil current storage not properly initialized" maxlog=10
        end
    end

    # Increment index
    snap0D.idx += 1

    # Reset cumulative trackers (like MATLAB version)
    if hasfield(typeof(RP.diagnostics.tracker), :cum1D_Ne_src)
        RP.diagnostics.tracker.cum1D_Ne_src = zero(RP.diagnostics.tracker.cum1D_Ne_src)
        RP.diagnostics.tracker.cum1D_Ne_loss = zero(RP.diagnostics.tracker.cum1D_Ne_loss)
    end

    if hasfield(typeof(RP.diagnostics.tracker), :cum1D_Ni_src)
        RP.diagnostics.tracker.cum1D_Ni_src = zero(RP.diagnostics.tracker.cum1D_Ni_src)
        RP.diagnostics.tracker.cum1D_Ni_loss = zero(RP.diagnostics.tracker.cum1D_Ni_loss)
    end

    return nothing
end

"""
    measure_snap2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Measure 2D diagnostic snapshots in the RAPID object.
This function is analogous to MATLAB's Measure_snap2D().

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to measure
"""
function measure_snap2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Get current index
    idx = RP.diagnostics.snap2D.idx

    # Check if we've reached the end of the array
    if idx > length(RP.diagnostics.snap2D.time_s)
        @warn "Reached end of pre-allocated 2D diagnostic arrays at index $idx"
        return nothing
    end

    # Store current time
    RP.diagnostics.snap2D.time_s[idx] = RP.time_s

    # Store 2D field snapshots using pre-allocated 3D arrays
    # The arrays are already allocated with proper dimensions (NR, NZ, dim_tt_2D)
    RP.diagnostics.snap2D.ne[:, :, idx] = RP.plasma.ne
    RP.diagnostics.snap2D.Te_eV[:, :, idx] = RP.plasma.Te_eV
    RP.diagnostics.snap2D.ue_para[:, :, idx] = RP.plasma.ue_para
    RP.diagnostics.snap2D.E_para_tot[:, :, idx] = RP.fields.E_para_tot

    # Increment index
    RP.diagnostics.snap2D.idx += 1

    return nothing
end
