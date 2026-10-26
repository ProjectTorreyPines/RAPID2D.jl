# The open-line charge sheets' field (`estimate_electrostatic_field_effects!`): γ is the field's
# shape alone, a closed line carries no sheets, and every case of the field and its thresholds is
# finite. A current disc's self field gives closed lines around the disc and open ones out to the
# wall; plasma is put on the open lines so the sheets have charge to hold.

@testitem "charge-sheet field: γ keeps its shape on closed lines, which carry no sheets" setup = [CurrentDisc] begin
    using RAPID2D: estimate_electrostatic_field_effects!
    using RAPID2D.Statistics: mean
    RP, _ = current_disc()
    trace_closed_region!(RP)
    pla, F, G = RP.plasma, RP.fields, RP.G
    closed = RP.flf.closed_surface_nids
    open_in = setdiff(G.nodes.in_wall_nids, closed)
    pla.ne[open_in] .= 1.0e17
    estimate_electrostatic_field_effects!(RP)

    @test !isempty(closed) && !isempty(open_in)
    @test pla.γ_shape_fac ≈ min.(1.0, 0.65 .* F.Bpol ./ mean(F.Bpol[G.nodes.in_wall_nids]))
    @test all(>(0), pla.γ_shape_fac[closed])
    # the sheets are on open lines only, boundary nodes included
    @test all(iszero, F.ERZ_pla_mag[closed]) && all(iszero, F.ERZ_para_pla[closed])
    @test any(>(0), F.ERZ_pla_mag[open_in])
    # an open-line threshold is never reached on a closed line
    @test all(isinf, pla.nc_para[closed]) && all(isinf, pla.nc_perp[closed])
    @test all(isfinite, pla.nc_perp[open_in])
end

@testitem "charge-sheet field: finite where Bpol vanishes, and none without a poloidal field" setup = [CurrentDisc] begin
    using RAPID2D: estimate_electrostatic_field_effects!
    using RAPID2D.Statistics: mean
    RP, _ = current_disc()
    trace_closed_region!(RP)
    pla, F, G, tp = RP.plasma, RP.fields, RP.G, RP.transport
    open_in = setdiff(G.nodes.in_wall_nids, RP.flf.closed_surface_nids)
    pla.ne[open_in] .= 1.0e17
    all_finite() = all(
        a -> all(isfinite, a),
        (F.ERZ_pla_mag, F.ERZ_para_pla, F.E_para_tot, pla.mean_ExB_R, pla.mean_ExB_Z, F.ER, F.EZ, tp.Dpol_turb),
    )

    # an open node where Bpol vanishes: γ ERZ_req at its limit 0.65 |E∥| B/⟨Bpol⟩ (γ → 0, ERZ_req → ∞),
    # below the Debye field here
    k = open_in[length(open_in) ÷ 2]
    F.Bpol[k] = F.bpol_R[k] = F.bpol_Z[k] = 0.0
    E_drive = F.Eϕ_para_presc[k] + F.Eϕ_para_coils_pla[k]
    estimate_electrostatic_field_effects!(RP)
    @test all_finite()
    @test F.ERZ_pla_mag[k] ≈ 0.65 * abs(E_drive) * F.Btot[k] / mean(F.Bpol[G.nodes.in_wall_nids])
    @test isfinite(pla.nc_perp[k])

    # no poloidal field at all: no shape, no sheets
    foreach(a -> fill!(a, 0.0), (F.Bpol, F.bpol_R, F.bpol_Z))
    estimate_electrostatic_field_effects!(RP)
    @test all(iszero, pla.γ_shape_fac) && all(iszero, F.ERZ_pla_mag)
    @test all(isinf, pla.nc_perp)
    @test all_finite()
end

@testitem "turbulence weight: one onset for the four consumers; the mean field's effects take none" setup = [CurrentDisc] begin
    using RAPID2D: estimate_electrostatic_field_effects!, shared_turbulent_channel, electron_wall_channels,
        ion_transport_channels, smooth_data_2D
    RP, _ = current_disc()
    trace_closed_region!(RP)
    pla, F, G, tp = RP.plasma, RP.fields, RP.G, RP.transport
    closed = RP.flf.closed_surface_nids
    open_in = setdiff(G.nodes.in_wall_nids, closed)
    pla.ne[open_in] .= [G.R2D[k] < R0 ? 1.0e10 : 1.0e17 for k in open_in]   # below and above nc_perp
    estimate_electrostatic_field_effects!(RP)

    w = pla.turbulence_weight
    n_SM = smooth_data_2D(pla.ne; num_SM = 2, weighting = G.inVol2D)
    @test w ≈ min.(1, sqrt.(max.(n_SM, 0) ./ pla.nc_perp))
    @test any(x -> 0 < x < 1, w[open_in]) && any(==(1), w[open_in])
    @test all(iszero, w[closed])
    # one speed for the electrons' tensor (`Dpol_turb`), the electron wall, the ions' channels and
    # their wall
    v = w .* F.ERZ_pla_mag ./ F.Btot
    @test tp.Dpol_turb ≈ 0.5 .* v .* tp.L_mixing
    turb = shared_turbulent_channel(RP)
    @test turb.v_para ≈ v && turb.v_perp ≈ v
    @test electron_wall_channels(RP)[end][1].v_para ≈ v
    @test ion_transport_channels(RP, tp.ion_species[1], turb)[end] === turb
    # the mean field's effects take no weight
    @test pla.mean_ExB_R ≈ F.ERZ_pla_mag ./ F.Btot .* sign.(F.Eϕ) .* F.bpol_Z
    @test F.ERZ_para_pla ≈ sign.(-F.Eϕ_presc .* F.Bϕ) .* F.ERZ_pla_mag .* F.Bpol ./ F.Btot

    # without the mixing, no weight and no turbulent diffusivity
    RP.flags.turb_ExB_mixing = false
    estimate_electrostatic_field_effects!(RP)
    @test all(iszero, pla.turbulence_weight) && all(iszero, tp.Dpol_turb)
end

@testitem "ambipolar weight: 1 on closed lines, the sheets' share of the cancelling field on open ones" setup = [CurrentDisc] begin
    using RAPID2D: update_space_charge_weights!, estimate_electrostatic_field_effects!, smooth_data_2D
    RP, _ = current_disc()
    trace_closed_region!(RP)
    pla, F, G = RP.plasma, RP.fields, RP.G
    closed = RP.flf.closed_surface_nids
    open_in = setdiff(G.nodes.in_wall_nids, closed)
    pla.ne[open_in] .= [G.R2D[k] < R0 ? 1.0e10 : 1.0e17 for k in open_in]   # below and above nc_para
    ERZ_target, E_Debye = update_space_charge_weights!(RP)
    w = pla.ambipolar_weight
    n_SM = smooth_data_2D(pla.ne; num_SM = 2, weighting = G.inVol2D)

    @test all(==(1), w[closed])
    # on open lines the share of the cancelling field the sheets hold, as the threshold gives it too
    @test w[open_in] ≈ min.(1, E_Debye[open_in] ./ ERZ_target[open_in])
    @test w[open_in] ≈ min.(1, sqrt.(n_SM[open_in] ./ pla.nc_para[open_in]))
    @test any(x -> 0 < x < 1, w[open_in]) && any(==(1), w[open_in])
    # and the sheets' field is that share of what they would need
    estimate_electrostatic_field_effects!(RP)
    @test F.ERZ_pla_mag[open_in] ≈ w[open_in] .* ERZ_target[open_in]
    # the snapshots carry both weights, the 0D ones as density-weighted means
    s2, s0 = RAPID2D.measure_snap2D(RP), RAPID2D.measure_snap0D(RP)
    @test s2.ambipolar_weight == w && s2.turbulence_weight == pla.turbulence_weight
    Ne = pla.ne .* G.inVol2D
    @test s0.ambipolar_weight ≈ sum(w .* Ne) / sum(Ne) && s0.turbulence_weight ≈ sum(pla.turbulence_weight .* Ne) / sum(Ne)

    # nothing to cancel: any plasma holds it
    foreach(a -> fill!(a, 0.0), (F.Eϕ_para_presc, F.Eϕ_para_coils_pla))
    update_space_charge_weights!(RP)
    @test all(==(1), w[open_in]) && all(iszero, pla.nc_para[open_in])
    # a cold plasma holds no charge
    fill!(pla.Te_eV, 0.0)
    update_space_charge_weights!(RP)
    @test all(iszero, w[open_in]) && all(==(1), w[closed])
    @test all(isinf, pla.nc_para)
    # without the coupling, no weight
    RP.flags.ambipolar_coupling = false
    update_space_charge_weights!(RP)
    @test all(iszero, w)
end

@testitem "space-charge weights: set by initialize!, kept without the sheets' field, which leaves nothing behind" setup = [CurrentDisc] begin
    using RAPID2D: update_space_charge_weights!, estimate_electrostatic_field_effects!
    # initialize! sets the weights from its state; the sheets' field waits for a step's end
    config = SimulationConfig{Float64}(;
        device_Name = "manual", manual = ManualSetup{Float64}(BR = 0.0, BZ = 1.0e-3, Eϕ = 0.5),
        NR = 17, NZ = 17, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(),
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    w0 = (copy(RP.plasma.turbulence_weight), copy(RP.plasma.ambipolar_weight))
    @test all(a -> any(>(0), a), w0)
    @test all(iszero, RP.fields.ERZ_pla_mag)
    update_space_charge_weights!(RP)
    @test (RP.plasma.turbulence_weight, RP.plasma.ambipolar_weight) == w0

    # the sheets' field switched off, its cancellation left on: no field, parallel part,
    # turbulence weight or diffusivity left from before, and E∥ without them at once (the
    # transport's refresh reads it before the next step); the coupling's weight is still set
    RP, _ = current_disc()
    trace_closed_region!(RP)
    pla, F, tp = RP.plasma, RP.fields, RP.transport
    closed = RP.flf.closed_surface_nids
    pla.ne[setdiff(RP.G.nodes.in_wall_nids, closed)] .= 1.0e17
    estimate_electrostatic_field_effects!(RP)
    @test any(>(0), F.ERZ_pla_mag) && any(>(0), tp.Dpol_turb) && any(!iszero, F.ERZ_para_pla)
    RP.flags.charge_sheet_field = false
    estimate_electrostatic_field_effects!(RP)
    @test all(iszero, F.ERZ_pla_mag) && all(iszero, F.ERZ_para_pla)
    @test F.E_para_tot ≈ F.Eϕ_para_presc .+ F.Eϕ_para_coils_pla
    @test all(iszero, pla.turbulence_weight) && all(iszero, tp.Dpol_turb)
    @test all(==(1), pla.ambipolar_weight[closed])
end
