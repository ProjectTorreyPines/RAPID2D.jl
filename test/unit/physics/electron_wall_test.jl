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
