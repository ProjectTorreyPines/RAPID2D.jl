using SimpleUnPack

"""
    measure_snap0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Measure snap0D diagnostics in the RAPID object, which are 0D (scalar) quantities averaged over the volume.
This function is analogous to MATLAB's Measure_snap1D().

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to measure
"""
function measure_snap0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
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
    snap0D.𝒲e_eV[idx] = sum(@. (1.5 * pla.Te_eV + (0.5 * me * pla.ue_para.^2 )/ ee) * pla.ne * inVol2D) / total_Ne

    # Ion quantities
    snap0D.ni[idx] = total_Ni / RP.G.device_inVolume
    snap0D.ni_max[idx] = maximum(pla.ni)
    snap0D.ui_para[idx] = sum(@. pla.ui_para * pla.ni * inVol2D) / total_Ni
    snap0D.Ti_eV[idx] = sum(@. pla.Ti_eV * pla.ni * inVol2D) / total_Ni
    snap0D.𝒲i_eV[idx] = sum(@. (1.5 * pla.Ti_eV + (0.5 * mi * pla.ui_para.^2 ) / ee) * pla.ni * inVol2D) / total_Ni

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
    Ntracker = RP.diagnostics.Ntracker
    snap0D.Ne_src_rate[idx] = Ntracker.cum0D_Ne_src / RP.config.snap0D_Δt_s
    snap0D.Ne_loss_rate[idx] = Ntracker.cum0D_Ne_loss / RP.config.snap0D_Δt_s
    snap0D.Ni_src_rate[idx] = Ntracker.cum0D_Ni_src / RP.config.snap0D_Δt_s
    snap0D.Ni_loss_rate[idx] = Ntracker.cum0D_Ni_loss / RP.config.snap0D_Δt_s

    # Growth rates
    prev_N = idx > 1 ? snap0D.ne[idx-1] * RP.G.device_inVolume : snap0D.ne[idx] * RP.G.device_inVolume
    snap0D.eGrowth_rate[idx] = log(FT(1) + Ntracker.cum0D_Ne_src / prev_N) / RP.config.snap0D_Δt_s
    snap0D.eLoss_rate[idx] = -log(FT(1) - Ntracker.cum0D_Ne_loss / prev_N) / RP.config.snap0D_Δt_s

    # Alternative growth rates
    snap0D.growth_rate2[idx] = snap0D.Ne_src_rate[idx] / (snap0D.ne[idx] * RP.G.device_inVolume)
    snap0D.loss_rate2[idx] = snap0D.Ne_loss_rate[idx] / (snap0D.ne[idx] * RP.G.device_inVolume)

    # Power balance calculations
    # Update power calculations first
    update_electron_heating_powers!(RP)
    update_ion_heating_powers!(RP)

    ePowers, iPowers = RP.plasma.ePowers, RP.plasma.iPowers

    # Calculate density-weighted averages for electron power components
    if total_Ne > 0
        snap0D.Pe.diffu[idx] = sum(@. ePowers.diffu * pla.ne * inVol2D) / total_Ne
        snap0D.Pe.conv[idx] = sum(@. ePowers.conv * pla.ne * inVol2D) / total_Ne
        snap0D.Pe.drag[idx] = sum(@. ePowers.drag * pla.ne * inVol2D) / total_Ne
        snap0D.Pe.iz[idx] = sum(@. ePowers.iz * pla.ne * inVol2D) / total_Ne
        snap0D.Pe.exc[idx] = sum(@. ePowers.exc * pla.ne * inVol2D) / total_Ne
        snap0D.Pe.dilution[idx] = sum(@. ePowers.dilution * pla.ne * inVol2D) / total_Ne
        snap0D.Pe.equi[idx] = sum(@. ePowers.equi * pla.ne * inVol2D) / total_Ne
        snap0D.Pe.heat[idx] = sum(@. ePowers.heat * pla.ne * inVol2D) / total_Ne
        snap0D.Pe.tot[idx] = sum(@. ePowers.tot * pla.ne * inVol2D) / total_Ne
    end

    # Calculate density-weighted averages for ion power components
    if total_Ni > 0
        snap0D.Pi.tot[idx] = sum(@. iPowers.tot * pla.ni * inVol2D) / total_Ni
        snap0D.Pi.atomic[idx] = sum(@. iPowers.atomic * pla.ni * inVol2D) / total_Ni
        snap0D.Pi.equi[idx] = sum(@. iPowers.equi * pla.ni * inVol2D) / total_Ni
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

    # Reset cumulative trackers
    RP.diagnostics.Ntracker.cum0D_Ne_src = zero(FT)
    RP.diagnostics.Ntracker.cum0D_Ne_loss = zero(FT)
    RP.diagnostics.Ntracker.cum0D_Ni_src = zero(FT)
    RP.diagnostics.Ntracker.cum0D_Ni_loss = zero(FT)

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

    snap2D = RP.diagnostics.snap2D  # alias for convenience
    pla = RP.plasma                  # alias for convenience
    F = RP.fields                    # alias for convenience
    T = RP.transport                 # alias for convenience

    @unpack ee, me, mi = RP.config.constants  # Unpack constants for convenience

    # Store metadata
    snap2D.step[idx] = RP.step
    snap2D.dt[idx] = RP.dt
    snap2D.time_s[idx] = RP.time_s

    # Basic plasma quantities
    snap2D.ne[:, :, idx] = pla.ne
    snap2D.Te_eV[:, :, idx] = pla.Te_eV

    # Transport coefficients
    snap2D.Dpara[:, :, idx] = T.Dpara
    snap2D.ue_para[:, :, idx] = pla.ue_para

    # Calculate derived quantities
    snap2D.u_pol[:, :, idx] = @. sqrt(pla.ueR^2 + pla.ueZ^2)
    snap2D.D_pol[:, :, idx] = @. sqrt(T.DRR^2 + T.DZZ^2)

    # Magnetic fields
    snap2D.BR[:, :, idx] = F.BR
    snap2D.BZ[:, :, idx] = F.BZ
    snap2D.B_pol[:, :, idx] = F.Bpol
    snap2D.BR_self[:, :, idx] = F.BR_self
    snap2D.BZ_self[:, :, idx] = F.BZ_self

    # Electric fields
    snap2D.E_para_tot[:, :, idx] = F.E_para_tot
    snap2D.E_para_ext[:, :, idx] = F.E_para_ext
    snap2D.Epol_self[:, :, idx] = F.Epol_self
    snap2D.Eϕ_self[:, :, idx] = F.Eϕ_self

    # Calculate ExB drift magnitude
    if hasfield(typeof(F), :mean_ExB_R) && hasfield(typeof(F), :mean_ExB_Z)
        snap2D.mean_ExB_pol[:, :, idx] = @. sqrt(F.mean_ExB_R^2 + F.mean_ExB_Z^2)
    end

    # Source/loss rates from cumulative trackers
    Ntracker = RP.diagnostics.Ntracker
    snap2D.Ne_src_rate[:, :, idx] = Ntracker.cum2D_Ne_src / RP.config.snap2D_Δt_s
    snap2D.Ne_loss_rate[:, :, idx] = Ntracker.cum2D_Ne_loss / RP.config.snap2D_Δt_s
    snap2D.Ni_src_rate[:, :, idx] = Ntracker.cum2D_Ni_src / RP.config.snap2D_Δt_s
    snap2D.Ni_loss_rate[:, :, idx] = Ntracker.cum2D_Ni_loss / RP.config.snap2D_Δt_s

    # Current densities (parallel current components)
    snap2D.Jϕ[:, :, idx] = pla.Jϕ
    snap2D.J_para[:, :, idx] = @. ee * (pla.ni * pla.ui_para - pla.ne * pla.ue_para)

    # Poloidal flux
    snap2D.psi_ext[:, :, idx] = F.psi_ext
    snap2D.psi_self[:, :, idx] = F.psi_self

    # Electron velocity components
    snap2D.ueR[:, :, idx] = pla.ueR
    snap2D.ueϕ[:, :, idx] = pla.ueϕ
    snap2D.ueZ[:, :, idx] = pla.ueZ

    # Physics parameters
    if hasfield(typeof(pla), :L_mixing)
        snap2D.L_mixing[:, :, idx] = pla.L_mixing
    end
    if hasfield(typeof(pla), :nc_para)
        snap2D.nc_para[:, :, idx] = pla.nc_para
        snap2D.nc_perp[:, :, idx] = pla.nc_perp
    end

    snap2D.γ_shape_fac[:, :, idx] = pla.γ_shape_fac

    # Ion quantities
    snap2D.ni[:, :, idx] = pla.ni
    snap2D.ui_para[:, :, idx] = pla.ui_para
    snap2D.uiR[:, :, idx] = pla.uiR
    snap2D.uiϕ[:, :, idx] = pla.uiϕ
    snap2D.uiZ[:, :, idx] = pla.uiZ
    snap2D.Ti_eV[:, :, idx] = pla.Ti_eV

    # MHD accelerations
    if hasfield(typeof(pla), :mean_aR_by_JxB)
        snap2D.mean_aR_by_JxB[:, :, idx] = pla.mean_aR_by_JxB
        snap2D.mean_aZ_by_JxB[:, :, idx] = pla.mean_aZ_by_JxB
    end

    # Coulomb logarithm
    if hasfield(typeof(pla), :lnA)
        snap2D.lnΛ[:, :, idx] = pla.lnA
    end

    # Neutral gas
    snap2D.n_H2_gas[:, :, idx] = pla.n_H2_gas

    # Calculate mean energies
    snap2D.𝒲e_eV[:, :, idx] = @. 1.5 * pla.Te_eV + 0.5 * me * pla.ue_para^2 / ee
    snap2D.𝒲i_eV[:, :, idx] = @. 1.5 * pla.Ti_eV + 0.5 * mi * pla.ui_para^2 / ee

    # Collision frequencies using RRC methods
    if RP.flags.Atomic_Collision
        RRC_mom = get_electron_RRC(RP, :Momentum)
        snap2D.coll_freq_en_mom[:, :, idx] = @. pla.n_H2_gas * RRC_mom
    end

    if RP.flags.Coulomb_Collision
        snap2D.coll_freq_ei[:, :, idx] = pla.ν_ei
    end

    # H-alpha emission
    if RP.flags.Atomic_Collision
        RRC_Halpha = get_electron_RRC(RP, :Halpha)
        snap2D.Halpha[:, :, idx] = @. pla.n_H2_gas * RRC_Halpha * pla.ne
    end

    # Update power calculations before storing
    update_electron_heating_powers!(RP)
    update_ion_heating_powers!(RP)

    # Store electron power components
    ePowers = pla.ePowers
    snap2D.Pe.tot[:, :, idx] = ePowers.tot
    snap2D.Pe.diffu[:, :, idx] = ePowers.diffu
    snap2D.Pe.conv[:, :, idx] = ePowers.conv
    snap2D.Pe.drag[:, :, idx] = ePowers.drag
    snap2D.Pe.dilution[:, :, idx] = ePowers.dilution
    snap2D.Pe.iz[:, :, idx] = ePowers.iz
    snap2D.Pe.exc[:, :, idx] = ePowers.exc

    # Store ion power components
    iPowers = pla.iPowers
    snap2D.Pi.tot[:, :, idx] = iPowers.tot
    snap2D.Pi.atomic[:, :, idx] = iPowers.atomic
    snap2D.Pi.equi[:, :, idx] = iPowers.equi

    # Control fields (if enabled)
    if hasfield(typeof(RP), :flags) && hasfield(typeof(RP.flags), :Control) && hasfield(typeof(RP.flags.Control), :state) && RP.flags.Control.state
        if snap2D.BR_ctrl !== nothing && snap2D.BZ_ctrl !== nothing
            snap2D.BR_ctrl[:, :, idx] = F.BR_ctrl
            snap2D.BZ_ctrl[:, :, idx] = F.BZ_ctrl
        end
    end

    # Increment index
    snap2D.idx += 1

    # Reset cumulative trackers
    fill!(RP.diagnostics.Ntracker.cum2D_Ne_src, zero(FT))
    fill!(RP.diagnostics.Ntracker.cum2D_Ne_loss, zero(FT))
    fill!(RP.diagnostics.Ntracker.cum2D_Ni_src, zero(FT))
    fill!(RP.diagnostics.Ntracker.cum2D_Ni_loss, zero(FT))

    return nothing
end
