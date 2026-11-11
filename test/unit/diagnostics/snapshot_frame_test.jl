# The diagnostics by the poloidal frame: the electrons' diffusivity along b̂_pol (`diffusivity_along_bpol`),
# the in-plane magnitudes `X_RZ_mag`, and `Bpol` under the state's name.

@testitem "diffusivity along b̂_pol: the tensor's frame component, the same at any tilt of the lines" begin
    using RAPID2D: diffusivity_along_bpol
    Dpara, Dperp, Dturb = 1.0e4, 2.0, 30.0
    bpol_frac = 0.1                                       # B_pol/B
    θ = collect(range(0, π; length = 13))                 # the lines' direction in the RZ plane
    bpol_R, bpol_Z = cos.(θ), sin.(θ)
    bR, bZ = bpol_frac .* bpol_R, bpol_frac .* bpol_Z     # the RZ part of the full b̂
    # the model's tensor: collisional along b̂, turbulent along b̂_pol
    DRR = @. Dperp + (Dpara - Dperp) * bR^2 + Dturb * bpol_R^2
    DRZ = @. (Dpara - Dperp) * bR * bZ + Dturb * bpol_R * bpol_Z
    DZZ = @. Dperp + (Dpara - Dperp) * bZ^2 + Dturb * bpol_Z^2
    along = Dpara * bpol_frac^2 + Dperp * (1 - bpol_frac^2) + Dturb
    @test all(isapprox.(diffusivity_along_bpol(DRR, DRZ, DZZ, bpol_R, bpol_Z), along; rtol = 1.0e-12))
    # the same quadratic form across the lines (φ̂ × b̂_pol) gives Dperp alone
    @test all(isapprox.(diffusivity_along_bpol(DRR, DRZ, DZZ, -bpol_Z, bpol_R), Dperp; rtol = 1.0e-12))
    # a poloidal null, b̂_pol stored as zero: half the RZ trace, the isotropic Dperp there
    @test diffusivity_along_bpol([Dperp], [0.0], [Dperp], [0.0], [0.0]) == [Dperp]
end

@testitem "snapshots by the poloidal frame: Bpol, the in-plane magnitudes, Dpol along b̂_pol" begin
    using RAPID2D: measure_snap2D, measure_snap0D, update_diffusion_tensor!
    config = SimulationConfig{Float64}(
        device_Name = "manual", manual = ManualSetup{Float64}(BR = 0.01, BZ = 0.02), NR = 25, NZ = 25,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    F, tp, pla = RP.fields, RP.transport, RP.plasma
    pla.ne[RP.G.nodes.in_wall_nids] .= 1.0e15
    # an oblique field (DRZ ≠ 0) and the model's tensor on it
    RP.flags.turb_ExB_mixing = true
    fill!(tp.Dpara, 1.0e4)
    fill!(tp.Dperp, 2.0)
    fill!(tp.Dpol_turb, 30.0)
    update_diffusion_tensor!(RP)
    @test maximum(abs, tp.DRZ) > 0
    # a drift and a flow of known magnitude, 5 in the plane
    fill!(pla.mean_ExB_R, 3.0)
    fill!(pla.mean_ExB_Z, -4.0)
    fill!(pla.ueR, -3.0)
    fill!(pla.ueZ, 4.0)
    fill!(pla.ue_para, 1.0e5)

    s2 = measure_snap2D(RP)
    @test s2.Bpol == F.Bpol
    @test all(≈(5.0), s2.mean_ExB_RZ_mag) && all(≈(5.0), s2.ue_RZ_mag)
    bpol_frac = F.Bpol ./ F.Btot
    along = @. 1.0e4 * bpol_frac^2 + 2.0 * (1 - bpol_frac^2) + 30.0
    @test all(isapprox.(s2.Dpol, along; rtol = 1.0e-10))

    s0 = measure_snap0D(RP)
    w = pla.ne .* RP.G.inVol2D
    @test s0.Dpol ≈ sum(along .* w) / sum(w) rtol = 1.0e-10
    @test s0.ue_para_RZ_mag ≈ sum(@. 1.0e5 * bpol_frac * w) / sum(w) rtol = 1.0e-10
end
