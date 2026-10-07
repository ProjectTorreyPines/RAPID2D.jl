# u∥ and Te at the wall: advected by the mass flux, diffused by a reflective in-wall operator,
# never damped through an out-wall band. A uniform state with no drive must stay uniform
# through a step, wall cells included. internal/docs/src/notes/design/wall-flux-channels.md §2.5–2.6.

@testsnippet UeTeWallDriver begin
    function ue_Te_one_step(; Implicit = true, upwind = true, heat_flux = false, albedo = 0.0, convec = true)
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

@testitem "u∥ and Te transport: one path, no flag; upwind = false keeps the interior central and the wall faces upwind" setup = [UeTeWallDriver] begin
    @test !(:primitive_advection in fieldnames(SimulationFlags{Float64}))
    @test_throws MethodError SimulationFlags{Float64}(primitive_advection = :nodal)
    # u∥ and Te are solved as themselves — that is RAPID2D's default, not a variant that
    # needs a label — so no `primitive_*` name survives either
    for name in (:electron_primitive_operators, :primitive_advection_operator, :apply_primitive_advection)
        @test !isdefined(RAPID2D, name)
    end
    using RAPID2D: ue_Te_operators, advection_operator, apply_advection
    RP = ue_Te_one_step(; upwind = false)
    @test all(isfinite, RP.plasma.ue_para) && all(isfinite, RP.plasma.Te_eV)
    # a uniform state stays uniform at the wall under the central interior scheme too
    inw = RP.G.nodes.in_wall_nids
    Te = RP.plasma.Te_eV[inw]
    @test maximum(Te) - minimum(Te) < 1.0e-9 * maximum(Te)
    u = RP.plasma.ue_para[inw]
    @test (maximum(u) - minimum(u)) < 1.0e-9 * abs(minimum(u))
end

@testitem "u∥ and Te stay uniform through a step at the wall" setup = [UeTeWallDriver] begin
    RP = ue_Te_one_step()
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

@testitem "Te diffusion is reflective: Σ vol·(∇·D∇Te) over in-wall nodes vanishes" setup = [UeTeWallDriver] begin
    using RAPID2D: ue_Te_operators
    RP = ue_Te_one_step()
    G = RP.G
    inw = G.nodes.in_wall_nids
    # a non-uniform Te so the operator actually does something; a uniform density first, where
    # the particle-weighted operator is the density operator itself and Σ vol·(A Te) = 0 says
    # the wall is reflective (the run left a gradient at the absorbing wall)
    Te = 10.0 .+ 5.0 .* sin.(3 .* G.R2D) .* cos.(2 .* G.Z2D)
    pla = RP.plasma
    pla.ne .= 0.0
    pla.ne[inw] .= 1.0e14
    ops = ue_Te_operators(RP)
    LTe = ops.A_diffu * vec(Te)
    vol = vec(G.inVol2D)
    @test sum(abs.(LTe[inw]) .* vol[inw]) > 0
    @test abs(sum(LTe[inw] .* vol[inw])) < 1.0e-10 * sum(abs.(LTe[inw]) .* vol[inw])
    @test all(iszero, LTe[G.nodes.on_out_wall_nids])
    # with a non-uniform density the operator is the particle-weighted one: what it conserves
    # is Σ vol·n·Te, so Σ vol·n·(M Te) is what the density's own diffusion moves, −Σ vol·Te·(A n)
    pla.ne[inw] .*= 1 .+ 0.3 .* sin.(2 .* G.R2D[inw]) .* cos.(3 .* G.Z2D[inw])
    n = vec(pla.ne)
    MTe = ue_Te_operators(RP).A_diffu * vec(Te)
    An = RP.operators.A_diffu_e * n
    carried = sum(vol[inw] .* n[inw] .* MTe[inw])
    @test abs(carried + sum(vol[inw] .* vec(Te)[inw] .* An[inw])) < 1.0e-10 * sum(vol[inw] .* n[inw] .* abs.(MTe[inw]))
    @test abs(carried) > 1.0e-6 * sum(vol[inw] .* n[inw] .* abs.(MTe[inw]))   # and it is not zero itself
end

@testitem "the heat-flux term reads nothing outside the wall" setup = [UeTeWallDriver] begin
    # The retired form differentiated log(ne) on the whole grid, and ne is zero on the excluded
    # band. Freeze the density (convec = false, mirror wall) so it stays uniform: then every
    # piece of −∇·(T u) − T u·∇ln n vanishes on the in-wall operators and Te must stay uniform.
    # Anything else is the band being read.
    RP = ue_Te_one_step(; heat_flux = true, albedo = 1.0, convec = false)
    inw = RP.G.nodes.in_wall_nids
    @test all(isfinite, RP.plasma.Te_eV[inw])
    @test all(x -> isapprox(x, 12.0; rtol = 1.0e-10), RP.plasma.Te_eV[inw])
    # With convection and an absorbing wall the density develops a gradient at the wall and the
    # term responds to it: a finite, small, physical response, not a NaN
    RP0 = ue_Te_one_step(; heat_flux = true)
    Te = RP0.plasma.Te_eV[RP0.G.nodes.in_wall_nids]
    @test all(isfinite, Te)
    @test all(x -> abs(x - 12.0) < 0.5, Te)
end

@testitem "the electron in-wall operators live on the wall pattern in RP.operators and equal a fresh build" setup = [UeTeWallDriver] begin
    using RAPID2D: build_face_flux_divergence, build_wall_diffusion_matrix, wall_divergence, wall_faces,
        build_wall_pattern
    RP = ue_Te_one_step()
    op, tp, G, pla = RP.operators, RP.transport, RP.G, RP.plasma
    @test tp.wall_faces == wall_faces(G)
    @test op.A_conv_e.matrix == build_face_flux_divergence(G, pla.ueR, pla.ueZ; upwind = RP.flags.upwind)
    @test op.A_diffu_e.matrix == build_wall_diffusion_matrix(G, tp.DRR, tp.DRZ, tp.DZZ; cross_terms = :drop)
    @test op.div_ue == wall_divergence(G, pla.ueR, pla.ueZ)
    # every reused operator, and the LHS buffer, sits on the one pattern: values only change
    P = build_wall_pattern(G)
    for A in (op.A_conv_e, op.A_diffu_e, op.A_adv_e, op.A_visc_drift_e, op.A_LHS)
        @test A.matrix.colptr == P.matrix.colptr && A.matrix.rowval == P.matrix.rowval && A.k2csc == P.k2csc
    end
end

@testitem "in-wall operators: one symbolic analysis per electron solver over a run with reversing flow" begin
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 25, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-7, t_end_s = 2.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    config.Output_path = mktempdir()
    RP = RAPID{Float64}(config)
    RP.flags = SimulationFlags{Float64}(
        src = true, diffu = true, convec = true, Atomic_Collision = true, Coulomb_Collision = true,
        ud_evolve = true, Te_evolve = true, Ti_evolve = true, update_ni_independently = true,
        Gas_evolve = false, E_para_self_ES = true, mean_ExB = true, turb_ExB_mixing = true,
        negative_n_correction = true, Ampere = false, E_para_self_EM = false,
    )
    initialize!(RP)
    G = RP.G
    n = @. 1.0e15 * exp(-((G.R2D - 1.5)^2 + G.Z2D^2) / (2 * 0.15^2))
    n[G.nodes.on_out_wall_nids] .= 0.0
    RP.plasma.ne .= n
    RP.plasma.ni .= n
    RP.plasma.Te_eV .= 5.0
    run_simulation!(RP)
    # reverse the drift: every face's upwind side flips, and the pattern must not care
    RP.plasma.ue_para .*= -1
    RP.t_end_s = 4.0e-6
    run_simulation!(RP)
    op = RP.operators
    nsteps = RP.step
    @test nsteps == 40
    for s in (op.ne_solver, op.Te_solver, op.ue_solver)
        @test s.nsymbolic == 1                  # analysed once for the whole run
        @test s.nfactor == nsteps               # refactorized numerically every step
    end
    @test all(iszero, RP.plasma.ne[G.nodes.on_out_wall_nids])
    @test all(isfinite, RP.plasma.ne) && all(isfinite, RP.plasma.Te_eV) && all(isfinite, RP.plasma.ue_para)
end

@testitem "operators not allocated on the wall pattern are refused before any update" begin
    using RAPID2D: cache_electron_operators!, ue_Te_operators, Operators
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    RP.operators = Operators{Float64}(RP.G.NR, RP.G.NZ)       # built by hand, not by initialize!
    @test_throws ArgumentError cache_electron_operators!(RP)
    @test_throws ArgumentError ue_Te_operators(RP)
    @test_throws ArgumentError solve_electron_continuity_equation!(RP)
end

@testitem "continuity with the wall as diagonal terms equals the fresh Robin and albedo operators" begin
    # The step folds the Robin debit and the convective albedo into the LHS and RHS as diagonal
    # vectors on the cached A_diffu_e and A_conv_e. The reference assembles the fresh wall
    # operators (the Robin 9-point matrix, the albedo-folded face flux) and solves with `\`.
    using RAPID2D: electron_transport_operator, convective_wall_operator, electron_wall_albedo,
        cache_electron_operators!
    using RAPID2D.LinearAlgebra, RAPID2D.SparseArrays
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 25, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        electron_wall_albedo = 0.3,
    )
    RP = RAPID{Float64}(config)
    RP.flags = SimulationFlags{Float64}(src = false, diffu = true, convec = true, Atomic_Collision = false)
    initialize!(RP)
    G, pla = RP.G, RP.plasma
    n0 = @. 1.0e15 * exp(-((G.R2D - 1.55)^2 + (G.Z2D - 0.05)^2) / (2 * 0.12^2))
    n0[G.nodes.on_out_wall_nids] .= 0.0
    pla.ne .= n0
    pla.ueR .= @. 2.0e4 * (G.R2D - 1.5)
    pla.ueZ .= @. -1.5e4 * G.Z2D
    update_transport_quantities!(RP)
    pla.ueR .= @. 2.0e4 * (G.R2D - 1.5)                  # prescribed after the transport update
    pla.ueZ .= @. -1.5e4 * G.Z2D
    cache_electron_operators!(RP)
    faces = RP.transport.wall_faces
    A_d, _ = electron_transport_operator(RP, faces)
    A_c, _ = convective_wall_operator(G, faces, pla.ueR, pla.ueZ, electron_wall_albedo(RP); upwind = RP.flags.upwind)
    θ, dt = RP.flags.θ_imp.transport, RP.dt
    n = vec(copy(pla.ne))
    L = sparse(1.0I, length(n), length(n)) - dt * θ * (A_d - A_c)
    n_ref = L \ (n + dt * (1 - θ) * (A_d * n - A_c * n))
    solve_electron_continuity_equation!(RP)
    @test vec(pla.ne) ≈ n_ref rtol = 1.0e-12
    @test maximum(abs, vec(pla.ne) .- n) > 1.0e-6 * maximum(n)   # the step did move the density
end

@testitem "electron operators cached for the other interior scheme are refused at the point of use" begin
    # A consumer cannot tell a central `A_conv_e` from an upwind one by looking at it, so the
    # cache records the `flags.upwind` it was built with and every consumer checks it: a
    # flag changed since the refresh is refused, not silently applied to the old operators.
    using RAPID2D: ue_Te_operators, solve_electron_continuity_equation!,
        update_electron_heating_powers!, cache_electron_operators!
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 2.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    @test RP.operators.A_conv_e_upwind == RP.flags.upwind
    RP.flags.upwind = !RP.flags.upwind                  # changed after the cache was built
    @test_throws ArgumentError ue_Te_operators(RP)
    @test_throws ArgumentError solve_electron_continuity_equation!(RP)
    @test_throws ArgumentError update_electron_heating_powers!(RP)
    cache_electron_operators!(RP)                       # the refresh records the scheme it used
    @test RP.operators.A_conv_e_upwind == RP.flags.upwind
    ue_Te_operators(RP)
    solve_electron_continuity_equation!(RP)
    update_electron_heating_powers!(RP)
    @test all(isfinite, RP.plasma.ne)
end

@testitem "a never-refreshed convective operator is refused while the wall faces see outflow" begin
    # An operator allocated on the pattern but never written applies nothing, while the
    # ledger books the outflow the faces see: refused, as master refused an empty cache.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    RP.flags.convec = true
    RP.plasma.ueR .= 1.0e5                           # outflow through the outer R-faces
    RAPID2D.cache_electron_operators!(RP)
    solve_electron_continuity_equation!(RP)          # refreshed: fine
    RAPID2D.initialize_operators!(RP)                # fresh operators: allocated, never written
    @test_throws ArgumentError solve_electron_continuity_equation!(RP)
end
