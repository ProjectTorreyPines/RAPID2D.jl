# Electron wall: the Robin path for the electron continuity equation.
#
# PR1 of internal/docs/src/notes/plans/PLAN_wall-flux-channels.md. The flag keeps the
# legacy path (`ne[on/out_wall] = 0` each step, loss booked from the zeroed band) as the
# default until the Robin path is validated, so every existing result is bit-identical
# while the new one is built beside it.

@testitem "electron_wall flag: two spellings, default keeps the legacy path" begin
    f = SimulationFlags{Float64}()
    @test f.electron_wall === :zeroing
    @test SimulationFlags{Float64}(electron_wall = :robin).electron_wall === :robin
    c = SimulationConfig{Float64}(NR = 6, NZ = 6)
    @test c.electron_wall_albedo == 0.0
end

@testitem "electron wall channels: R = 1 is reflective, ceilings are non-negative, tensor matches legacy" begin
    using RAPID2D: electron_wall_channels, electron_wall_absorption_speeds, electron_transport_operator,
        wall_faces, total_tensor, build_wall_diffusion_matrix
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Dperp0 = 0.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    faces = wall_faces(RP.G)
    v = electron_wall_absorption_speeds(RP, faces)
    @test length(v) == length(faces)
    @test all(>=(0), v)
    # R_e = 1: every channel's ceiling is multiplied by (1 − R) = 0 — a reflective wall
    RP.config.electron_wall_albedo = 1.0
    @test all(==(0), electron_wall_absorption_speeds(RP, faces))
    RP.config.electron_wall_albedo = 0.0
    # tensor consistency: with Dperp0 = 0 the channel sum reproduces the legacy tensor on
    # every in-wall node. Outside the wall the legacy path damps Dpara/Dperp
    # (`Damp_Transp_outWall`) and zeroes the grid frame; the wall-aware operator never reads
    # the tensor there, so those nodes are not part of the contract.
    D_RR, D_RZ, D_ZZ = total_tensor(electron_wall_channels(RP))
    tp = RP.transport
    inw = RP.G.nodes.in_wall_nids
    scale = maximum(abs, tp.DZZ[inw])
    @test D_RR[inw] ≈ tp.DRR[inw] atol = 1.0e-10 * scale
    @test D_RZ[inw] ≈ tp.DRZ[inw] atol = 1.0e-10 * scale
    @test D_ZZ[inw] ≈ tp.DZZ[inw] atol = 1.0e-10 * scale
    A, v2 = electron_transport_operator(RP, faces)
    @test v2 == v
    @test A == build_wall_diffusion_matrix(RP.G, tp.DRR, tp.DRZ, tp.DZZ; faces = faces, v_absorb = v)
end

@testitem "electron Robin wall: what the wall took plus what remains is what there was" begin
    function one_step(θ)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = 2.0e-6, t_end_s = 2.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,   # no mid-run snapshot: `update_snaps0D!` resets the tracker
        )
        config.Output_path = mktempdir()
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(
            electron_wall = :robin, diffu = true, convec = false, src = false,
            Atomic_Collision = false, Coulomb_Collision = false, mean_ExB = false,
            turb_ExB_mixing = false, E_para_self_ES = false, E_para_self_EM = false, Ampere = false,
            Te_evolve = false, ud_evolve = false, Ti_evolve = false, Gas_evolve = false,
            update_ni_independently = false, secondary_electron = false, negative_n_correction = false,
        )
        w = RP.flags.θ_imp
        RP.flags.θ_imp = ImplicitWeights{Float64}(transport = θ, growth = θ, decay = w.decay, gas = w.gas)
        initialize!(RP)
        G = RP.G
        inw = G.nodes.in_wall_nids
        RP.plasma.ne .= 0
        RP.plasma.ne[inw] .= 1.0e14 .* (1 .+ 0.3 .* sin.(3 .* G.R2D[inw]))   # in-wall only; on/out stay 0
        N0 = sum(G.Jacob[inw] .* RP.plasma.ne[inw])
        loss0 = RP.diagnostics.Ntracker.cum0D_Ne_loss
        run_simulation!(RP)
        N1 = sum(G.Jacob[inw] .* RP.plasma.ne[inw])
        absorbed = (RP.diagnostics.Ntracker.cum0D_Ne_loss - loss0) / (2π * G.dR * G.dZ)
        return N0, N1, absorbed, RP
    end
    for θ in (1.0, 0.5)
        N0, N1, absorbed, RP = one_step(θ)
        @test absorbed > 0
        @test (N0 - N1) ≈ absorbed rtol = 1.0e-10
        @test all(==(0), RP.plasma.ne[RP.G.nodes.on_out_wall_nids])
        @test sum(RP.diagnostics.Ntracker.cum2D_Ne_loss) ≈ RP.diagnostics.Ntracker.cum0D_Ne_loss rtol = 1.0e-12
    end
end

@testitem "electron Robin wall: R_e = 1 conserves Σ J·ne exactly" begin
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 2.0e-6, t_end_s = 1.0e-5, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, electron_wall_albedo = 1.0,
    )
    config.Output_path = mktempdir()
    RP = RAPID{Float64}(config)
    RP.flags = SimulationFlags{Float64}(
        electron_wall = :robin, diffu = true, convec = false, src = false,
        Atomic_Collision = false, Coulomb_Collision = false, mean_ExB = false, turb_ExB_mixing = false,
        E_para_self_ES = false, E_para_self_EM = false, Ampere = false, Te_evolve = false,
        ud_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
        secondary_electron = false, negative_n_correction = false,
    )
    initialize!(RP)
    inw = RP.G.nodes.in_wall_nids
    RP.plasma.ne .= 0
    RP.plasma.ne[inw] .= 1.0e14
    N0 = sum(RP.G.Jacob[inw] .* RP.plasma.ne[inw])
    run_simulation!(RP)
    @test sum(RP.G.Jacob[inw] .* RP.plasma.ne[inw]) ≈ N0 rtol = 1.0e-12
    @test RP.diagnostics.Ntracker.cum0D_Ne_loss == 0
end

@testitem "electron Robin wall + face-flux convection: the ledger closes with both channels on" begin
    # Under `:robin` convection is the face-flux operator: its wall-face outflow is a
    # diagonal debit like the Robin one, so one ledger coefficient per face (diffusive +
    # convective speed) books exactly what the two operators removed. The nodal upwind
    # operator could not do this (no rows on the grid frame; a central-difference branch
    # at |u| < eps), which is why the electron half of the face-flux work rides in this PR.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 2.0e-6, t_end_s = 2.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    config.Output_path = mktempdir()
    RP = RAPID{Float64}(config)
    RP.flags = SimulationFlags{Float64}(
        electron_wall = :robin, diffu = true, convec = true, src = false,
        Atomic_Collision = false, Coulomb_Collision = false, mean_ExB = true,
        turb_ExB_mixing = false, E_para_self_ES = false, E_para_self_EM = false, Ampere = false,
        Te_evolve = false, ud_evolve = false, Ti_evolve = false, Gas_evolve = false,
        update_ni_independently = false, secondary_electron = false, negative_n_correction = false,
    )
    initialize!(RP)
    G = RP.G
    inw = G.nodes.in_wall_nids
    RP.plasma.ne .= 0
    RP.plasma.ne[inw] .= 1.0e14
    # a poloidal drift toward the outer wall; E_para_self_ES is off so nothing rebuilds it
    RP.plasma.mean_ExB_R .= 2.0e4
    RP.plasma.mean_ExB_Z .= 0.0
    N0 = sum(G.Jacob[inw] .* RP.plasma.ne[inw])
    loss0 = RP.diagnostics.Ntracker.cum0D_Ne_loss
    run_simulation!(RP)
    N1 = sum(G.Jacob[inw] .* RP.plasma.ne[inw])
    booked = (RP.diagnostics.Ntracker.cum0D_Ne_loss - loss0) / (2π * G.dR * G.dZ)
    @test booked > 0
    @test (N0 - N1) ≈ booked rtol = 1.0e-10
end
