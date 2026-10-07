# The fixtures the mixing tests stand on: an analytic X-point poloidal field for the manual
# setup, and a prescribed electron diffusion tensor that replaces the plasma's own.
#
# Pure mixing along field lines cannot be isolated from the plasma tensor: Bohm cannot be
# switched off and D∥ is never zero. So the tensor is a POLICY, resolved where the tensor is
# assembled, and the default policy is the plasma tensor the code always had. The X-point
# field is a policy of the manual setup for the same reason: the uniform field stays the
# default, and the hyperbolic one is a choice, not a different code path.
# internal/docs/src/notes/design/turbulent-mixing-u-T.md §6; PLAN_turbulent-mixing-u-T.md Phase 2a.

@testsnippet MixingFixtureConfig begin
    "A walled box on a 41 × 47 grid, 1 ≤ R ≤ 2, |Z| ≤ 0.5, writing its config into a temp dir."
    function box_config(; kw...)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = 41, NZ = 47,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.2, 1.8, 1.8, 1.2], wall_Z = [-0.3, -0.3, 0.3, 0.3],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = 1.0e-7, t_end_s = 1.0e-7, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            kw...,
        )
        config.Output_path = mktempdir()
        return config
    end

    "In-wall nodes whose 3 × 3 neighbourhood is in-wall: where a central difference is central."
    function deep_nodes(G)
        return [
            nid for nid in G.nodes.in_wall_nids if all(
                    RAPID2D.is_in_wall(G, G.nodes.rid[nid] + di, G.nodes.zid[nid] + dj) for di in -1:1, dj in -1:1
                )
        ]
    end
end

@testitem "X-point manual field: R B_R = B' y, R B_Z = B' x, divergence-free, ψ analytic and tangent, FLF completes" setup = [MixingFixtureConfig] begin
    using RAPID2D: XPointPoloidal, UniformPoloidal, calculate_divergence, calculate_B_from_ψ, wall_gradient
    Bprime, R0, Z0 = 0.02, 1.5, 0.0
    config = box_config()
    config.manual.poloidal = XPointPoloidal(R0 = R0, Z0 = Z0, Bprime = Bprime)
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G, F = RP.G, RP.fields
    x = G.R2D .- R0
    y = G.Z2D .- Z0
    # the field, on every node
    @test G.R2D .* F.BR ≈ Bprime .* y
    @test G.R2D .* F.BZ ≈ Bprime .* x
    @test F.BR_ext == F.BR && F.BZ_ext == F.BZ
    # the null sits on a grid node, and B_pol vanishes there
    i0, j0 = argmin(abs.(G.R1D .- R0)), argmin(abs.(G.Z1D .- Z0))
    @test F.Bpol[i0, j0] <= 1.0e-12 * Bprime
    @test F.bpol_R[i0, j0] == 0 && F.bpol_Z[i0, j0] == 0
    # ∇·B = 0, exactly for central differences since R B_R is constant in R and B_Z in Z
    div = calculate_divergence(G, F.BR, F.BZ)
    @test maximum(abs, div[2:(end - 1), 2:(end - 1)]) <= 1.0e-12 * Bprime / minimum(G.R1D) / G.dR
    # ψ_ext is the analytic flux of this field, in the code's sign convention
    @test F.ψ_ext ≈ Bprime / 2 .* (x .^ 2 .- y .^ 2)
    BRψ, BZψ = calculate_B_from_ψ(G, F.ψ_ext)
    @test BRψ[2:(end - 1), 2:(end - 1)] ≈ F.BR[2:(end - 1), 2:(end - 1)] rtol = 1.0e-10
    @test BZψ[2:(end - 1), 2:(end - 1)] ≈ F.BZ[2:(end - 1), 2:(end - 1)] rtol = 1.0e-10
    # field lines follow the contours of ψ: B·∇ψ = 0 where the gradient is central
    gR, gZ = wall_gradient(G, F.ψ_ext)
    deep = deep_nodes(G)
    tangent = (F.BR .* gR .+ F.BZ .* gZ)[deep]
    scale = (F.Bpol .* hypot.(gR, gZ))[deep]
    @test maximum(abs, tangent) <= 1.0e-12 * maximum(scale)
    # the field-line analysis completes on the hyperbolic field (D∥'s ceiling needs it)
    inw = G.nodes.in_wall_nids
    @test all(isfinite, RP.flf.Lpol_tot[inw])
    @test all(isfinite, RP.transport.Dpara[inw])
    # the default is the uniform field of the setup, with ψ_ext = 0 as before
    RP_u = RAPID{Float64}(box_config())
    initialize!(RP_u)
    @test RP_u.config.manual.poloidal isa UniformPoloidal
    @test RP_u.fields.BR == fill(RP_u.config.manual.BR, G.NR, G.NZ)
    @test RP_u.fields.BZ == fill(RP_u.config.manual.BZ, G.NR, G.NZ)
    @test iszero(RP_u.fields.ψ_ext)
end

@testitem "prescribed tensor: aligned with the poloidal field line, no Bohm or D∥ in it, kept through a step" setup = [MixingFixtureConfig] begin
    using RAPID2D: PrescribedTensor, PlasmaTensor, XPointPoloidal
    D_along, D_across = 50.0, 0.5
    # a straight vertical field: D_ZZ = D_along, D_RR = D_across, no cross term
    RP = RAPID{Float64}(box_config())
    RP.flags.Ampere = false
    RP.flags.diffusion_tensor = PrescribedTensor(D_along = D_along, D_across = D_across)
    initialize!(RP)
    G, tp = RP.G, RP.transport
    check_straight(tp) = all(==(D_along), tp.DZZ) && all(==(D_across), tp.DRR) && all(iszero, tp.DRZ)
    @test check_straight(tp)
    @test tp.CTZZ ≈ G.Jacob .* D_along ./ G.dZ^2
    @test tp.CTRR ≈ G.Jacob .* D_across ./ G.dR^2
    # the plasma's own D⊥ (Bohm) and D∥ are computed and differ from the prescription; they
    # are not applied
    inw = G.nodes.in_wall_nids
    @test !all(==(D_across), tp.Dperp[inw])
    @test !all(==(D_along), tp.Dpara[inw])
    @test maximum(tp.Dpara[inw]) > D_along
    # the step's end rebuilds the tensor; the prescription survives it
    run_simulation!(RP)
    @test RP.step == 1
    @test check_straight(RP.transport)
    # on the X-point field the tensor is D_across 𝟙 + (D_along − D_across) b_pol b_polᵀ, isotropic at the null
    config = box_config()
    config.manual.poloidal = XPointPoloidal(R0 = 1.5, Z0 = 0.0, Bprime = 0.02)
    RPx = RAPID{Float64}(config)
    RPx.flags.Ampere = false
    RPx.flags.diffusion_tensor = PrescribedTensor(D_along = D_along, D_across = D_across)
    initialize!(RPx)
    Fx, tpx = RPx.fields, RPx.transport
    @test tpx.DRR ≈ D_across .+ (D_along - D_across) .* Fx.bpol_R .^ 2
    @test tpx.DRZ ≈ (D_along - D_across) .* Fx.bpol_R .* Fx.bpol_Z
    @test tpx.DZZ ≈ D_across .+ (D_along - D_across) .* Fx.bpol_Z .^ 2
    i0, j0 = argmin(abs.(G.R1D .- 1.5)), argmin(abs.(G.Z1D .- 0.0))
    @test tpx.DRR[i0, j0] == D_across && tpx.DZZ[i0, j0] == D_across && tpx.DRZ[i0, j0] == 0
    # the default policy is the plasma tensor, and naming it changes nothing
    @test SimulationFlags{Float64}().diffusion_tensor isa PlasmaTensor
    RP_d = RAPID{Float64}(box_config())
    RP_d.flags.Ampere = false
    initialize!(RP_d)
    RP_p = RAPID{Float64}(box_config())
    RP_p.flags.Ampere = false
    RP_p.flags.diffusion_tensor = PlasmaTensor()
    initialize!(RP_p)
    @test RP_p.transport.DRR == RP_d.transport.DRR
    @test RP_p.transport.DRZ == RP_d.transport.DRZ
    @test RP_p.transport.DZZ == RP_d.transport.DZZ
    @test !check_straight(RP_d.transport)
end

@testitem "mixing fixtures: the policies are exported and refuse non-physical parameters" begin
    # exported: reachable unqualified after `using RAPID2D`
    @test PrescribedTensor === RAPID2D.PrescribedTensor
    @test PlasmaTensor === RAPID2D.PlasmaTensor
    @test DiffusionTensorModel === RAPID2D.DiffusionTensorModel
    @test XPointPoloidal === RAPID2D.XPointPoloidal
    @test UniformPoloidal === RAPID2D.UniformPoloidal
    @test ManualPoloidalField === RAPID2D.ManualPoloidalField
    @test PrescribedTensor <: DiffusionTensorModel && PlasmaTensor <: DiffusionTensorModel
    @test XPointPoloidal <: ManualPoloidalField && UniformPoloidal <: ManualPoloidalField
    # a diffusivity is finite and non-negative; D_across defaults to zero
    t = PrescribedTensor(D_along = 50.0)
    @test t.D_along == 50.0 && t.D_across == 0.0
    @test_throws ArgumentError PrescribedTensor(D_along = -1.0)
    @test_throws ArgumentError PrescribedTensor(D_along = 1.0, D_across = -0.1)
    @test_throws ArgumentError PrescribedTensor(D_along = NaN)
    @test_throws ArgumentError PrescribedTensor(D_along = Inf)
    # an X-point has a finite position and a non-zero gradient
    p = XPointPoloidal(R0 = 1.5, Z0 = 0.0, Bprime = 0.02)
    @test (p.R0, p.Z0, p.Bprime) == (1.5, 0.0, 0.02)
    @test_throws ArgumentError XPointPoloidal(R0 = 1.5, Z0 = 0.0, Bprime = 0.0)
    @test_throws ArgumentError XPointPoloidal(R0 = NaN, Z0 = 0.0, Bprime = 0.02)
    @test_throws ArgumentError XPointPoloidal(R0 = 1.5, Z0 = Inf, Bprime = 0.02)
    @test_throws ArgumentError XPointPoloidal(R0 = 0.0, Z0 = 0.0, Bprime = 0.02)
end

@testitem "config.bp keeps the manual field policy: its concrete type and its parameters" setup = [MixingFixtureConfig] begin
    using RAPID2D: XPointPoloidal, adios_load
    if !Sys.iswindows()
        config = box_config()
        config.manual.poloidal = XPointPoloidal(R0 = 1.5, Z0 = 0.1, Bprime = 0.02)
        RP = RAPID{Float64}(config)
        initialize!(RP)
        # global scalars come back as one-element arrays
        saved = adios_load(joinpath(config.Output_path, config.Output_prefix * "config.bp"))
        @test saved["manual/poloidal/type"] == ["XPointPoloidal"]
        @test saved["manual/poloidal/R0"] == [1.5]
        @test saved["manual/poloidal/Z0"] == [0.1]
        @test saved["manual/poloidal/Bprime"] == [0.02]
        # a policy without parameters still leaves its name
        RP_u = RAPID{Float64}(box_config())
        initialize!(RP_u)
        saved_u = adios_load(joinpath(RP_u.config.Output_path, RP_u.config.Output_prefix * "config.bp"))
        @test saved_u["manual/poloidal/type"] == ["UniformPoloidal"]
        @test saved_u["manual/BZ"] == [RP_u.config.manual.BZ]
    end
end
