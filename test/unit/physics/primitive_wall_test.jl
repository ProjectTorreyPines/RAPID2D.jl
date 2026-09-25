# u∥ and Te at the wall: advected by the mass flux, diffused by a reflective in-wall operator,
# never damped through an out-wall band. A uniform state with no drive must stay uniform
# through a step, wall cells included. internal/docs/src/notes/design/wall-flux-channels.md §2.5–2.6.

@testsnippet PrimitiveWallDriver begin
    function primitive_one_step(; Implicit = true, upwind = true, heat_flux = false, albedo = 0.0, convec = true)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = 2.0e-6, t_end_s = 4.0e-6, snap0D_Δt_s = 4.0e-6, snap2D_Δt_s = 4.0e-6,
            electron_wall_albedo = albedo,
        )
        config.Output_path = mktempdir()
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(
            Implicit = Implicit, upwind = upwind,
            diffu = true, convec = convec, src = false, Atomic_Collision = false, Coulomb_Collision = false,
            mean_ExB = false, turb_ExB_mixing = false, E_para_self_ES = false, E_para_self_EM = false,
            Ampere = false, Te_evolve = true, ud_evolve = true, Ti_evolve = false, Gas_evolve = false,
            update_ni_independently = false, secondary_electron = false, negative_n_correction = false,
            Include_ud_pressure_term = false, Include_heat_flux_term = heat_flux,
        )
        initialize!(RP)
        G = RP.G
        inw = G.nodes.in_wall_nids
        RP.plasma.ne .= 0
        RP.plasma.ne[inw] .= 1.0e14
        # no drive: nothing should change u∥ or Te. The derived fields were built at
        # initialize!, so zero them too, not just the loop voltage they came from.
        RP.fields.LV_ext .= 0.0
        RP.fields.Eϕ_ext .= 0.0
        RP.fields.E_para_ext .= 0.0
        RP.fields.E_para_tot .= 0.0
        RP.plasma.ue_para .= -1.0e6
        RP.plasma.Te_eV .= 12.0
        run_simulation!(RP)
        return RP
    end
end

@testitem "primitive transport: one path, no flag; upwind = false keeps the interior central and the wall faces upwind" setup = [PrimitiveWallDriver] begin
    @test !(:primitive_advection in fieldnames(SimulationFlags{Float64}))
    @test_throws MethodError SimulationFlags{Float64}(primitive_advection = :nodal)
    RP = primitive_one_step(; upwind = false)
    @test all(isfinite, RP.plasma.ue_para) && all(isfinite, RP.plasma.Te_eV)
    # a uniform state stays uniform at the wall under the central interior scheme too
    inw = RP.G.nodes.in_wall_nids
    Te = RP.plasma.Te_eV[inw]
    @test maximum(Te) - minimum(Te) < 1.0e-9 * maximum(Te)
    u = RP.plasma.ue_para[inw]
    @test (maximum(u) - minimum(u)) < 1.0e-9 * abs(minimum(u))
end

@testitem "u∥ and Te stay uniform through a step at the wall" setup = [PrimitiveWallDriver] begin
    RP = primitive_one_step()
    inw = RP.G.nodes.in_wall_nids
    @test all(iszero, RP.fields.E_para_tot[inw])
    # u∥ carries a uniform 2.5e-8 relative decay from the residual collision frequencies the
    # flags leave on; what the wall must not do is make the state NON-uniform.
    u = RP.plasma.ue_para[inw]
    @test all(x -> isapprox(x, -1.0e6; rtol = 1.0e-6), u)
    @test (maximum(u) - minimum(u)) < 1.0e-9 * abs(minimum(u))
    @test all(x -> isapprox(x, 12.0; rtol = 1.0e-10), RP.plasma.Te_eV[inw])
    @test all(isfinite, RP.plasma.Te_eV) && all(isfinite, RP.plasma.ue_para)
end

@testitem "Te diffusion is reflective: Σ vol·(∇·D∇Te) over in-wall nodes vanishes" setup = [PrimitiveWallDriver] begin
    using RAPID2D: electron_primitive_operators
    RP = primitive_one_step()
    G = RP.G
    inw = G.nodes.in_wall_nids
    # a non-uniform Te so the operator actually does something
    Te = 10.0 .+ 5.0 .* sin.(3 .* G.R2D) .* cos.(2 .* G.Z2D)
    ops = electron_primitive_operators(RP)
    LTe = ops.D_op * vec(Te)
    vol = vec(G.inVol2D)
    @test sum(abs.(LTe[inw]) .* vol[inw]) > 0
    @test abs(sum(LTe[inw] .* vol[inw])) < 1.0e-10 * sum(abs.(LTe[inw]) .* vol[inw])
    @test all(iszero, LTe[G.nodes.on_out_wall_nids])
end

@testitem "the heat-flux term reads nothing outside the wall" setup = [PrimitiveWallDriver] begin
    # The retired form differentiated log(ne) on the whole grid, and ne is zero on the excluded
    # band. Freeze the density (convec = false, mirror wall) so it stays uniform: then every
    # piece of −∇·(T u) − T u·∇ln n vanishes on the in-wall operators and Te must stay uniform.
    # Anything else is the band being read.
    RP = primitive_one_step(; heat_flux = true, albedo = 1.0, convec = false)
    inw = RP.G.nodes.in_wall_nids
    @test all(isfinite, RP.plasma.Te_eV[inw])
    @test all(x -> isapprox(x, 12.0; rtol = 1.0e-10), RP.plasma.Te_eV[inw])
    # With convection and an absorbing wall the density develops a gradient at the wall and the
    # term responds to it: a finite, small, physical response, not a NaN
    RP0 = primitive_one_step(; heat_flux = true)
    Te = RP0.plasma.Te_eV[RP0.G.nodes.in_wall_nids]
    @test all(isfinite, Te)
    @test all(x -> abs(x - 12.0) < 0.5, Te)
end

@testitem "the electron in-wall operators are cached once per step and equal a fresh build" setup = [PrimitiveWallDriver] begin
    using RAPID2D: build_face_flux_divergence, build_wall_diffusion_matrix, wall_divergence, wall_faces
    RP = primitive_one_step()
    tp, G, pla = RP.transport, RP.G, RP.plasma
    @test tp.wall_faces == wall_faces(G)
    @test tp.C_e == build_face_flux_divergence(G, pla.ueR, pla.ueZ; upwind = RP.flags.upwind)
    @test tp.D_op_e == build_wall_diffusion_matrix(G, tp.DRR, tp.DRZ, tp.DZZ; cross_terms = :drop)
    @test tp.div_ue == wall_divergence(G, pla.ueR, pla.ueZ)
end
