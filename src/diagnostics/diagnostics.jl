using SimpleUnPack

"""
    measure_snap0D!(RP::RAPID{FT}, snap0D::Snapshot0D{FT}) where {FT<:AbstractFloat}

In-place measurement of 0D (scalar) diagnostics into an existing Snapshot0D object.
Updates the provided snap0D with current simulation state and volume-averaged quantities.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to measure
- `snap0D::Snapshot0D{FT}`: Pre-allocated snapshot object to fill with measurements
"""
function measure_snap0D!(RP::RAPID{FT}, snap0D::Snapshot0D{FT}) where {FT <: AbstractFloat}

    update_charge_states!(RP)   # the ion sums of the densities as they are now
    pla = RP.plasma
    F = RP.fields
    inVol2D = RP.G.inVol2D  # alias for convenience

    @unpack ee, me = RP.config.constants  # Unpack constants for convenience

    # Store metadata
    snap0D.time_s = RP.time_s
    snap0D.dt = RP.dt
    snap0D.step = RP.step

    # total number of electrons and ions (all species) in the device volume
    Ne2D = pla.ne .* inVol2D
    Ni2D = pla.ni_tot .* inVol2D

    total_Ne = sum(Ne2D)
    total_Ni = sum(Ni2D)


    # Basic electron quantities
    snap0D.ne = total_Ne / RP.G.device_inVolume
    snap0D.ne_max = maximum(pla.ne)
    snap0D.ue_para = sum(@. pla.ue_para * Ne2D) / total_Ne
    snap0D.Te_eV = sum(@. pla.Te_eV * Ne2D) / total_Ne
    snap0D.Ke_eV = sum(@. (1.5 * pla.Te_eV + (0.5 * me * pla.ue_para^2) / ee) * Ne2D) / total_Ne

    # Ion quantities
    snap0D.ni_tot = total_Ni / RP.G.device_inVolume
    snap0D.ni_tot_max = maximum(pla.ni_tot)
    snap0D.ui_para = sum(@. pla.ui_para * Ni2D) / total_Ni
    snap0D.Ti_eV = sum(@. pla.Ti_eV * Ni2D) / total_Ni
    snap0D.Ki_eV = sum(@. (1.5 * pla.Ti_eV + (0.5 * pla.mi_mean * pla.ui_para^2) / ee) * Ni2D) / total_Ni

    # Toroidal current: the stored Jϕ, and the one of the step's end state (free runs do not
    # refresh Jϕ after the density and the ions move)
    snap0D.I_tor = sum(pla.Jϕ * RP.G.dR * RP.G.dZ)
    snap0D.I_tor_endpoint = sum(
        @. (
            pla.ne * RP.config.constants.qe * pla.ue_para +
                pla.Zni_tot * RP.config.constants.ee * pla.ui_para
        ) * F.bϕ
    ) * RP.G.dR * RP.G.dZ

    # Electron kinetic energy, and the closed-surface flux constraint's domain and ledger
    snap0D.Ke_tot = electron_kinetic_energy(RP.G, pla.ne, pla.ue_para, RP.G.nodes.in_wall_nids; me)
    cs = RP.closed_surface
    if cs.basis === nothing
        snap0D.cs_nodes, snap0D.cs_knots, snap0D.cs_current_out = 0, 0, zero(FT)
    else
        closed = [k for k in RP.flf.closed_surface_nids if RP.G.nodes.state[k] > 0.5]
        snap0D.cs_nodes, snap0D.cs_knots = length(cs.basis.nodes), length(cs.basis.knots)
        snap0D.cs_current_out = current_outside_constraint(cs.basis, closed, pla.Jϕ)
    end
    led = cs.ledger
    snap0D.cs_E_activation, snap0D.cs_E_basis_change = led.activation, led.basis_change
    snap0D.cs_E_projection, snap0D.cs_E_remap = led.projection, led.remap

    # The near-vacuum treatments: where the relaxation acts now, and the run's ledger
    relax_weight = vacuum_relaxation_weights(RP, pla.ne)
    relaxed = relax_weight .> zero(FT)
    snap0D.vac_relax_nodes = count(relaxed)
    snap0D.vac_relax_Ne_share = total_Ne > zero(FT) ? sum(Ne2D[relaxed]) / total_Ne : zero(FT)
    nv = RP.diagnostics.near_vacuum
    snap0D.vac_relax_momentum, snap0D.vac_relax_energy = nv.relax_momentum, nv.relax_energy
    snap0D.limited_pairs = nv.limited_pairs
    snap0D.floor_nodes, snap0D.floor_Ne = nv.floor_nodes, nv.floor_Ne

    # The current column's force balance, and the vertical-field control that cancels it
    for (name, value) in pairs(column_force_balance(RP))
        setfield!(snap0D, name, value)
    end
    measure_vertical_field_control!(snap0D, RP)

    # Electric fields (density-weighted averages)
    snap0D.E_para_tot = sum(@. F.E_para_tot * Ne2D) / total_Ne
    snap0D.Eϕ_para_presc = sum(@. F.Eϕ_para_presc * Ne2D) / total_Ne
    snap0D.Eϕ_para_ctrl = sum(@. F.Eϕ_para_ctrl * Ne2D) / total_Ne
    snap0D.ERZ_para_pla = sum(@. F.ERZ_para_pla * Ne2D) / total_Ne
    snap0D.Eϕ_para_coils_pla = sum(@. F.Eϕ_para_coils_pla * Ne2D) / total_Ne

    # Transport quantities
    snap0D.abs_ue_para_RZ = sum(@. abs(pla.ue_para) * F.Bpol / F.Btot * Ne2D) / total_Ne
    snap0D.D_RZ = sum(@. sqrt(RP.transport.DRR^2 + RP.transport.DZZ^2) * Ne2D) / total_Ne
    snap0D.turbulence_weight = sum(@. pla.turbulence_weight * Ne2D) / total_Ne

    # Each parent's species, volume-averaged: its ions by charge state, its neutral gas (the H₂
    # gas among them)
    layout = RP.store_layout
    snap0D.ion_parents = copy(layout.ion_parents)
    snap0D.ni_parents = [[sum(view(A, :, :, z) .* inVol2D) / RP.G.device_inVolume for z in axes(A, 3)] for A in pla.n_ions]
    snap0D.gas_parents = copy(layout.gas_parents)
    snap0D.ng_parents = [sum(@. n * inVol2D) / RP.G.device_inVolume for n in pla.n_gas]
    snap0D.n_H2_gas_min = minimum(pla.n_H2_gas)

    # Electron collision frequencies
    if RP.flags.Atomic_Collision
        # ν_en_iz / ν_en_mom_tot are materialized by update_RRCs! and already carry the
        # n_H2_gas factor, so a snapshot reports exactly what the step ran on. Halpha has
        # no materialized field: it is diagnostic-only and lives on the (Te, u∥) surfaces
        # rather than (E/p, Ē), so it stays a live query.
        eRRC_Hα = get_electron_RRC(RP, :Halpha)

        snap0D.ν_en_iz = sum(@. pla.ν_en_iz * Ne2D) / total_Ne
        snap0D.ν_en_mom_tot = sum(@. pla.ν_en_mom_tot * Ne2D) / total_Ne
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
        @warn "CFL condition calculation not yet implemented" maxlog = 10
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
    #
    # `snaps0D` holds only COMPLETED snapshots here — `update_snaps0D!` pushes the one
    # being measured afterwards — so `end` is the previous one. The first snapshot has no
    # predecessor; zero reproduces what the preallocated slot supplied (NaN growth rates).
    snaps0D = RP.diagnostics.snaps0D
    prev_total_Ne = isempty(snaps0D) ? zero(FT) : snaps0D[end].ne * RP.G.device_inVolume
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
        snap0D.Pe_visc_heat = sum(@. ePowers.visc_heat * Ne2D) / total_Ne
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
    snap0D.R_cen_by_ne = sum(@. RP.G.R2D * Ne2D) / total_Ne
    snap0D.Z_cen_by_ne = sum(@. RP.G.Z2D * Ne2D) / total_Ne

    snap0D.R_cen_by_Jϕ_Vol = sum(@. RP.G.R2D * pla.Jϕ * inVol2D) / sum(pla.Jϕ .* inVol2D)
    snap0D.Z_cen_by_Jϕ_Vol = sum(@. RP.G.Z2D * pla.Jϕ * inVol2D) / sum(pla.Jϕ .* inVol2D)

    snap0D.ueR = sum(@. pla.ueR * Ne2D) / total_Ne
    snap0D.ueZ = sum(@. pla.ueZ * Ne2D) / total_Ne

    snap0D.aR_by_JxB = sum(@. pla.mean_aR_by_JxB * Ne2D) / total_Ne
    snap0D.aZ_by_JxB = sum(@. pla.mean_aZ_by_JxB * Ne2D) / total_Ne

    # Control system (if enabled)
    if hasfield(typeof(RP), :Flag) && hasfield(typeof(RP.Flag), :Control) && hasfield(typeof(RP.Flag.Control), :state) && RP.Flag.Control.state
        @warn "Control system diagnostics not yet implemented" maxlog = 10
        # Would need PID controller and control field calculations
    end

    # measure magnetic energies by toroidal currents
    snap0D.tot_W_mag = zero(FT)
    if RP.flags.Ampere
        # the plasma's own flux, for its self-inductance (solved with coils = false, so the array
        # named for the coils' and the plasma's field holds the plasma's alone)
        F_plasma = solve_Ampere_equation(RP; plasma = true, coils = false)
        W_mag_by_plasma_without_coil = 0.5 * sum(@. RP.plasma.Jϕ * F_plasma.ψ_coils_pla / RP.G.R2D * RP.G.inVol2D)
        snap0D.self_inductance_plasma = 2.0 * W_mag_by_plasma_without_coil / (snap0D.I_tor^2 + eps(FT))

        # the plasma current's magnetic energy in the field of the coils and the plasma [J]:
        # (1/2)∫Jϕ Aϕ dV with Aϕ = ψ_coils_pla/R
        snap0D.tot_W_mag += 0.5 * sum(@. RP.plasma.Jϕ * RP.fields.ψ_coils_pla / RP.G.R2D * RP.G.inVol2D)

        # Ohmic dissipation [W]. ν_en_iz_tot, not ν_en_iz alone: this mirrors the SAME
        # momentum-randomizing sum update_ue_para! and the momentum-Ampere solvers use
        # (physics.jl), and both ionization channels dilute the drift identically.
        ν_eff = @. pla.ν_ei_eff + pla.ν_en_mom_tot + pla.ν_en_iz_tot
        @unpack me, ee = RP.config.constants

        σ_conductivity = @. pla.ne * ee^2 / (me * ν_eff)

        valid_idx = findall(pla.ne .> 1.0e3 .&& ν_eff .> 0.0)

        if isempty(valid_idx)
            snap0D.η_resistivity = Inf
            snap0D.resistance_plasma = Inf
            snap0D.tot_P_ohm_plasma = 0.0
        else
            snap0D.η_resistivity = 1.0 / (sum(@views @. σ_conductivity[valid_idx] * Ne2D[valid_idx]) / total_Ne)
            snap0D.resistance_plasma = 1.0 / sum(@views @. σ_conductivity[valid_idx] * RP.G.dR * RP.G.dZ / (2π * RP.G.R2D[valid_idx]))
            snap0D.tot_P_ohm_plasma = snap0D.I_tor^2 * snap0D.resistance_plasma # P_ohm = I^2*R
        end
    end

    if RP.coil_system.n_total > 0
        csys = RP.coil_system
        coils = csys.coils

        snap0D.coils_I = coils.current
        snap0D.coils_V_applied = get_all_voltages_at_time(csys)

        # magnetic energy by coils [J]
        Φ_coils = (
            csys.mutual_inductance * coils.current
                + 2π * (csys.Green_grid2coils * pla.Jϕ[:]) * RP.G.dR * RP.G.dZ
        )
        snap0D.tot_W_mag += 0.5 * coils.current' * Φ_coils

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
function measure_snap0D(RP::RAPID{FT}) where {FT <: AbstractFloat}
    # Create a new Snapshot0D object
    snap0D = Snapshot0D{FT}()

    # Measure and fill the snapshot
    measure_snap0D!(RP, snap0D)

    return snap0D
end


# a flux below which the closure is measured against this floor, not the fields [Wb/rad]
const PSI_CLOSURE_FLOOR = 1.0e-12

"""
    measure_field_sources!(RP, snap2D)

The coils' (walls included) and the plasma's fields by source, for a 2D snapshot:
- ψ, B_R, B_Z of the plasma from its stored `Jϕ` ([`plasma_flux_on_grid`](@ref)) and of the coils
  from their currents ([`coil_flux_on_grid`](@ref)), each computed here;
- the closure ‖ψ_coils_pla − ψ_pla − ψ_coils‖∞ against the solved sum, absolute and over
  max(`PSI_CLOSURE_FLOOR`, ‖ψ_coils_pla‖∞, ‖ψ_pla‖∞ + ‖ψ_coils‖∞): what the coupled solve's outer
  iteration and the updates after it (PostProjection, continuity, ions) leave, not a solver error;
- the induced field of the last step split as the coils' part [`coil_induced_Eϕ`](@ref) of the
  step's change `ΔI_step` and the rest, `Eϕ_coils_pla − Eϕ_coils`: the field the model applied less the
  coils' part, not the change of the plasma's own flux (below the Ampère gate it is zero).
NaN where the run does not solve the field (Ampère off), and the split NaN without induction
(`E_para_induction` off).
"""
function measure_field_sources!(RP::RAPID{FT}, snap2D::Snapshot2D{FT}) where {FT <: AbstractFloat}
    G, F, csys = RP.G, RP.fields, RP.coil_system
    sources = (snap2D.ψ_pla, snap2D.BR_pla, snap2D.BZ_pla, snap2D.ψ_coils, snap2D.BR_coils, snap2D.BZ_coils)
    if !_solves_coils_pla_field(RP)
        foreach(a -> fill!(a, FT(NaN)), (sources..., snap2D.Eϕ_coils, snap2D.Eϕ_pla))
        snap2D.ψ_closure_abs = snap2D.ψ_closure_rel = FT(NaN)
        return snap2D
    end
    snap2D.ψ_pla .= plasma_flux_on_grid(RP, RP.plasma.Jϕ)
    calculate_B_from_ψ!(G, snap2D.ψ_pla, snap2D.BR_pla, snap2D.BZ_pla)
    if csys.n_total > 0
        snap2D.ψ_coils .= coil_flux_on_grid(RP, get_all_currents(csys))
        calculate_B_from_ψ!(G, snap2D.ψ_coils, snap2D.BR_coils, snap2D.BZ_coils)
    else
        foreach(a -> fill!(a, zero(FT)), (snap2D.ψ_coils, snap2D.BR_coils, snap2D.BZ_coils))
    end
    snap2D.ψ_closure_abs = maximum(abs, F.ψ_coils_pla .- snap2D.ψ_pla .- snap2D.ψ_coils)
    scale = max(FT(PSI_CLOSURE_FLOOR), maximum(abs, F.ψ_coils_pla), maximum(abs, snap2D.ψ_pla) + maximum(abs, snap2D.ψ_coils))
    snap2D.ψ_closure_rel = snap2D.ψ_closure_abs / scale

    if !RP.flags.E_para_induction
        fill!(snap2D.Eϕ_coils, FT(NaN))
        fill!(snap2D.Eϕ_pla, FT(NaN))
        return snap2D
    end
    if length(csys.ΔI_step) != csys.n_total       # the coil set changed outside add_coil!: unknown
        fill!(snap2D.Eϕ_coils, FT(NaN))
        fill!(snap2D.Eϕ_pla, FT(NaN))
        return snap2D
    end
    snap2D.Eϕ_coils .= coil_induced_Eϕ(RP, csys.ΔI_step)
    @. snap2D.Eϕ_pla = F.Eϕ_coils_pla - snap2D.Eϕ_coils
    return snap2D
end

"""
    measure_snap2D!(RP::RAPID{FT}, snap2D::Snapshot2D{FT}) where {FT<:AbstractFloat}

In-place measurement of 2D diagnostics into an existing Snapshot2D object.
Updates the provided snap2D with current simulation state and 2D field distributions.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to measure
- `snap2D::Snapshot2D{FT}`: Pre-allocated snapshot object to fill with measurements
"""
function measure_snap2D!(RP::RAPID{FT}, snap2D::Snapshot2D{FT}) where {FT <: AbstractFloat}

    @assert snap2D.dims_RZ == (RP.G.NR, RP.G.NZ) "Snapshot dimensions must match RAPID grid dimensions"
    update_charge_states!(RP)   # the ion sums of the densities as they are now

    # Get current index
    pla = RP.plasma                  # alias for convenience
    F = RP.fields                    # alias for convenience
    tp = RP.transport                 # alias for convenience

    @unpack ee, me = RP.config.constants  # Unpack constants for convenience

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
    snap2D.BR_coils_pla .= F.BR_coils_pla
    snap2D.BZ_coils_pla .= F.BZ_coils_pla

    # Electric fields
    snap2D.E_para_tot .= F.E_para_tot
    snap2D.Eϕ_para_presc .= F.Eϕ_para_presc
    snap2D.ERZ_pla_mag .= F.ERZ_pla_mag
    snap2D.Eϕ_coils_pla .= F.Eϕ_coils_pla

    # Loop voltages
    snap2D.LV_presc .= F.LV_presc
    @. snap2D.LV_tot = F.LV_presc + 2π * RP.G.R2D * (F.Eϕ_coils_pla + F.Eϕ_ctrl)

    # Calculate ExB drift magnitude
    @. snap2D.mean_ExB_pol = sqrt(pla.mean_ExB_R^2 + pla.mean_ExB_Z^2)

    # Source/loss rates from cumulative trackers
    Ntracker = RP.diagnostics.Ntracker
    @. snap2D.Ne_src_rate = Ntracker.cum2D_Ne_src / RP.config.snap2D_Δt_s
    @. snap2D.Ne_loss_rate = Ntracker.cum2D_Ne_loss / RP.config.snap2D_Δt_s
    @. snap2D.Ni_src_rate = Ntracker.cum2D_Ni_src / RP.config.snap2D_Δt_s
    @. snap2D.Ni_loss_rate = Ntracker.cum2D_Ni_loss / RP.config.snap2D_Δt_s

    # Current densities (parallel current components). The ion term is the ion CHARGE
    # density, Σ zₛ nₛ, not the ion density.
    snap2D.Jϕ .= pla.Jϕ
    @. snap2D.J_para = ee * (pla.Zni_tot * pla.ui_para - pla.ne * pla.ue_para)

    # Poloidal flux
    snap2D.ψ .= F.ψ
    snap2D.ψ_presc .= F.ψ_presc
    snap2D.ψ_coils_pla .= F.ψ_coils_pla
    snap2D.ψ_ctrl .= F.ψ_ctrl
    measure_field_sources!(RP, snap2D)

    # the closed region, its axis and LCFS
    fill!(snap2D.closed_lines, zero(FT))
    for k in RP.flf.closed_surface_nids
        RP.G.nodes.state[k] > 0.5 && (snap2D.closed_lines[k] = one(FT))
    end
    lcfs = closed_region_lcfs(RP)
    snap2D.ψ_axis, snap2D.ψ_lcfs, snap2D.R_axis, snap2D.Z_axis = lcfs === nothing ? (FT(NaN), FT(NaN), FT(NaN), FT(NaN)) :
        (lcfs.ψ_axis, lcfs.ψ_lcfs, lcfs.R, lcfs.Z)
    measure_magnetic_lcfs!(snap2D, RP, lcfs)

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
    snap2D.turbulence_weight .= pla.turbulence_weight

    # Ion quantities (all species: their sum, sharing one velocity and temperature)
    snap2D.ni_tot .= pla.ni_tot
    snap2D.ui_para .= pla.ui_para
    snap2D.uiR .= pla.uiR
    snap2D.uiϕ .= pla.uiϕ
    snap2D.uiZ .= pla.uiZ
    snap2D.Ti_eV .= pla.Ti_eV

    # MHD accelerations
    snap2D.mean_aR_by_JxB .= pla.mean_aR_by_JxB
    snap2D.mean_aZ_by_JxB .= pla.mean_aZ_by_JxB

    # Coulomb logarithms — the electron-ion one and the ion-ion one, which are
    # different formulas (NRL p.34b vs p.34c) and differ by tens of percent
    # whenever Te ≠ Ti.
    snap2D.lnΛ .= pla.lnΛ
    snap2D.lnΛ_ii .= pla.lnΛ_ii

    # Each parent's species: its ions by charge state, its neutral gas
    snap2D.ion_parents = copy(RP.store_layout.ion_parents)
    snap2D.ni_parents = [copy(A) for A in pla.n_ions]
    snap2D.gas_parents = copy(RP.store_layout.gas_parents)
    snap2D.ng_parents = [copy(n) for n in pla.n_gas]

    # Calculate mean energies
    @. snap2D.Ke_eV = 1.5 * pla.Te_eV + 0.5 * me * pla.ue_para^2 / ee
    @. snap2D.Ki_eV = 1.5 * pla.Ti_eV + 0.5 * pla.mi_mean * pla.ui_para^2 / ee

    # Collision frequencies using RRC methods
    if RP.flags.Atomic_Collision
        # Materialized frequencies (update_RRCs!); Halpha stays a live query, see the
        # 0D snapshot above.
        eRRC_Halpha = get_electron_RRC(RP, :Halpha)

        snap2D.ν_en_mom_tot .= pla.ν_en_mom_tot
        snap2D.ν_en_iz .= pla.ν_en_iz
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
    snap2D.Pe_visc_heat .= pla.ePowers.visc_heat
    snap2D.Pe_conv .= pla.ePowers.conv
    snap2D.Pe_drag .= pla.ePowers.drag
    snap2D.Pe_dilution .= pla.ePowers.dilution
    snap2D.Pe_iz .= pla.ePowers.iz
    snap2D.Pe_exc .= pla.ePowers.exc

    # Store ion power components
    snap2D.Pi_tot .= pla.iPowers.tot
    snap2D.Pi_atomic .= pla.iPowers.atomic
    snap2D.Pi_equi .= pla.iPowers.equi

    # ν_en_iz_tot, not ν_en_iz alone: mirrors the momentum-randomizing sum the
    # momentum solve itself uses (see the 0D Ohmic-dissipation comment above).
    ν_eff = @. pla.ν_ei_eff + pla.ν_en_mom_tot + pla.ν_en_iz_tot
    @. snap2D.η_resistivity = (me * ν_eff) / (pla.ne * ee^2)

    # The band outside the wall holds no plasma and no solve reaches it: the copies above
    # carry whatever the initial condition left there (zero density, room temperature, a
    # velocity nothing evolves). Masked, so a plot or a moment never reads the band as
    # plasma. Fields, fluxes and rates are not plasma state and stay as they are.
    band = RP.G.nodes.on_out_wall_nids
    for field in (
            snap2D.ne, snap2D.ni_tot, snap2D.Te_eV, snap2D.Ti_eV, snap2D.ue_para, snap2D.ui_para,
            snap2D.Ke_eV, snap2D.Ki_eV,
        )
        field[band] .= FT(NaN)
    end
    for A in snap2D.ni_parents, z in axes(A, 3)          # every ion species, plane by plane
        view(A, :, :, z)[band] .= FT(NaN)
    end

    return RP
end

# The magnetic LCFS around the closed region's axis (`lcfs`, from `closed_region_lcfs`), as the
# 2D snapshot records it: NaN and configuration 0 without an axis or without a closed level.
function measure_magnetic_lcfs!(snap2D::Snapshot2D{FT}, RP::RAPID{FT}, lcfs) where {FT <: AbstractFloat}
    nan = FT(NaN)
    snap2D.ψ_lcfs_magnetic, snap2D.lcfs_configuration = nan, 0
    snap2D.R_geo, snap2D.Z_geo, snap2D.R_contact, snap2D.Z_contact = nan, nan, nan, nan
    snap2D.R_xpoint1, snap2D.Z_xpoint1, snap2D.R_xpoint2, snap2D.Z_xpoint2 = nan, nan, nan, nan
    lcfs === nothing && return snap2D
    G, ψ = RP.G, RP.fields.ψ
    in_wall = falses(G.NR, G.NZ)
    in_wall[G.nodes.in_wall_nids] .= true
    g = FieldLineGeometry(G, ψ; inside = G.cell_state .>= 0)
    m = magnetic_lcfs(G, ψ, (; R = lcfs.R, Z = lcfs.Z, ψ = lcfs.ψ_axis), in_wall; g)
    m === nothing && return snap2D
    snap2D.ψ_lcfs_magnetic = m.ψ_lcfs
    snap2D.lcfs_configuration = findfirst(==(m.configuration), LCFS_CONFIGURATIONS)
    snap2D.R_geo, snap2D.Z_geo = m.geometric_centre.R, m.geometric_centre.Z
    snap2D.R_contact, snap2D.Z_contact = m.contact.R, m.contact.Z
    length(m.xpoints) >= 1 && ((snap2D.R_xpoint1, snap2D.Z_xpoint1) = (m.xpoints[1].R, m.xpoints[1].Z))
    length(m.xpoints) >= 2 && ((snap2D.R_xpoint2, snap2D.Z_xpoint2) = (m.xpoints[2].R, m.xpoints[2].Z))
    return snap2D
end

"""
    finite_extrema(arrays) -> (lo, hi)

Extrema over every finite entry of every array in `arrays`, for colour limits: the 2D
snapshots carry NaN on the band outside the wall, and a plain `minimum`/`maximum` over
them returns NaN. `(NaN, NaN)` when nothing is finite.
"""
function finite_extrema(arrays)
    lo, hi = Inf, -Inf
    for a in arrays, x in a
        isfinite(x) || continue
        lo = min(lo, x)
        hi = max(hi, x)
    end
    return lo <= hi ? (lo, hi) : (NaN, NaN)
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
function measure_snap2D(RP::RAPID{FT}) where {FT <: AbstractFloat}
    # Create a new Snapshot2D object with the correct dimensions
    snap2D = Snapshot2D{FT}(dims_RZ = (RP.G.NR, RP.G.NZ))

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
function reset_Ntracker_cumulative_0D!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    # Reset cumulative trackers
    RP.diagnostics.Ntracker.cum0D_Ne_src = zero(FT)
    RP.diagnostics.Ntracker.cum0D_Ne_loss = zero(FT)
    RP.diagnostics.Ntracker.cum0D_Ni_src = zero(FT)
    RP.diagnostics.Ntracker.cum0D_Ni_loss = zero(FT)
    fill!(RP.diagnostics.Ntracker.cum0D_Ni_src_s, zero(FT))
    fill!(RP.diagnostics.Ntracker.cum0D_Ni_loss_s, zero(FT))

    return RP
end

"""
    reset_Ntracker_cumulative_2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Reset cumulative 2D particle number trackers to zero.
Called after each 2D snapshot to restart accumulation for the next diagnostic interval.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance with trackers to reset
"""
function reset_Ntracker_cumulative_2D!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    # Reset cumulative trackers
    fill!(RP.diagnostics.Ntracker.cum2D_Ne_src, zero(FT))
    fill!(RP.diagnostics.Ntracker.cum2D_Ne_loss, zero(FT))
    fill!(RP.diagnostics.Ntracker.cum2D_Ni_src, zero(FT))
    fill!(RP.diagnostics.Ntracker.cum2D_Ni_loss, zero(FT))

    return RP
end


"""
    update_snaps0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Measure the current state into a new `Snapshot0D` and append it to `snaps0D`.
Automatically resets cumulative trackers after measurement.

Measure then append, not append then fill: `measure_snap0D!` reads `snaps0D[end]` for the
growth rate, and that must be the PREVIOUS snapshot.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to update diagnostics for
"""
function update_snaps0D!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    push!(RP.diagnostics.snaps0D, measure_snap0D(RP))

    # Reset cumulative trackers after snapshot
    reset_Ntracker_cumulative_0D!(RP)

    return RP
end


"""
    update_snaps2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Measure the current state into a new `Snapshot2D` and append it to `snaps2D`.
Automatically resets cumulative trackers after measurement.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to update diagnostics for
"""
function update_snaps2D!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    push!(RP.diagnostics.snaps2D, measure_snap2D(RP))

    # Reset cumulative trackers after snapshot
    reset_Ntracker_cumulative_2D!(RP)

    return RP
end

function check_energy_conservation(snaps0D::Vector{Snapshot0D{FT}}) where {FT}
    times = snaps0D.time_s

    trapz = (P, t) -> sum(0.5 * (P[2:end] + P[1:(end - 1)]) .* diff(t))

    W_input = trapz(snaps0D.tot_P_input_coils, times)
    W_ohm_coil = trapz(snaps0D.tot_P_ohm_coils, times)
    W_ohm_plasma = trapz(snaps0D.tot_P_ohm_plasma, times)

    ΔW_mag = snaps0D.tot_W_mag[end] - snaps0D.tot_W_mag[1]

    # Input = Losses + Stored Energy
    energy_balance = W_input - (W_ohm_coil + W_ohm_plasma + ΔW_mag)

    @printf("Energy Conservation Analysis:\n")
    @printf("=================================\n")
    @printf("Input energy (coils):    %12.7f J\n", W_input)
    @printf("Magnetic energy change:  %12.7f J\n", ΔW_mag)
    @printf("Coil ohmic losses:       %12.7f J\n", W_ohm_coil)
    @printf("Plasma ohmic losses:     %12.7f J\n", W_ohm_plasma)
    @printf("---------------------------------\n")
    @printf("Energy balance:          %12.7f J (should be ≈ 0)\n", energy_balance)
    @printf("Relative error:          %12.2f%%\n", abs(energy_balance / W_input) * 100)

    return (;
        W_input,
        W_ohm_coil,
        W_ohm_plasma,
        ΔW_mag,
        energy_balance,
        rel_error = abs(energy_balance / W_input) * 100,
    )
end
