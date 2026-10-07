# Mixing of u∥ and Te along field lines: the acceptance tests.
#
# A diffusive particle flux carries momentum and energy. Mixed by the particles that move,
# each field line settles to the n-weighted means of what it started with (design (6.1)),
# and the kinetic energy of the sheared flow it erased reappears as heat (6.2). The operator
# RAPID2D used until this work diffused u∥ and Te as if every node held the same number of
# electrons, so it settles to volume means and heats nothing: the checks marked broken here
# are the ones Phase 3 (u∥) and Phase 4 (Te) turn on.
# internal/docs/src/reference/electron-diffusive-transport.md; notes/design/turbulent-mixing-u-T.md §6.

@testsnippet PureMixingRun begin
    using RAPID2D: PrescribedTensor, UniformPoloidal

    """
    A run in which nothing but mixing acts on u∥ and Te: E = 0, no sources, no convection, no
    pressure, a reflective wall (albedo 1), the ions and the gas frozen, and the bulk tensor
    prescribed along the poloidal field line. The stored collision rates are zeroed on every
    step by `enforce_pure_mixing!`, because the momentum equation reads them whatever the
    flags say. The caller sets the initial state after this returns.
    """
    function pure_mixing_RP(;
            NR = 21, NZ = 47, dt = 1.0e-6, t_end_s, D_along, D_across = 0.0,
            poloidal = UniformPoloidal(), Implicit = true, θ_transport = 0.5,
        )
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = NR, NZ = NZ,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.2, 1.8, 1.8, 1.2], wall_Z = [-0.3, -0.3, 0.3, 0.3],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = dt, t_end_s = t_end_s, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            electron_wall_albedo = 1.0,
        )
        config.manual.Eϕ = 0.0
        config.manual.poloidal = poloidal
        config.Output_path = mktempdir()
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(
            Implicit = Implicit, diffu = true, convec = false, src = false,
            Atomic_Collision = false, Coulomb_Collision = false,
            mean_ExB = false, turb_ExB_mixing = false, E_para_self_ES = false, E_para_self_EM = false,
            Ampere = false, Te_evolve = true, ud_evolve = true, Ti_evolve = false, Gas_evolve = false,
            update_ni_independently = false, secondary_electron = false, negative_n_correction = false,
            Include_ud_pressure_term = false, Include_ud_convec_term = false, Include_ud_diffu_term = true,
            Include_Te_convec_term = false, Include_Te_diffu_term = true, Include_heat_flux_term = false,
            diffusion_tensor = PrescribedTensor(D_along = D_along, D_across = D_across),
        )
        RP.flags.θ_imp.transport = θ_transport
        RP.flags.θ_imp.decay = θ_transport
        initialize!(RP)
        return RP
    end

    "Before each step: no friction, no drive. The rates are refreshed at every step's end."
    function enforce_pure_mixing!(RP)
        pla = RP.plasma
        fill!(pla.ν_en_mom_tot, 0.0)
        fill!(pla.ν_en_iz_tot, 0.0)
        fill!(pla.ν_ei_eff, 0.0)
        return all(iszero, RP.fields.E_para_tot)
    end

    "After each step: plasma above the floor everywhere in-wall, Te never clipped, no power but diffusion."
    function pure_mixing_holds(RP)
        pla, inw, cfg = RP.plasma, RP.G.nodes.in_wall_nids, RP.config
        return all(>(1.0), pla.ne[inw]) &&
            all(T -> cfg.min_Te < T < cfg.max_Te, pla.Te_eV[inw]) &&
            pla.ePowers.tot[inw] ≈ pla.ePowers.diffu[inw]
    end

    "Run to `t_end_s` under the contract; returns whether it held on every step."
    function run_pure_mixing!(RP)
        held = Ref(true)
        run_simulation!(
            RP;
            callback_before_step = rp -> (held[] &= enforce_pure_mixing!(rp)),
            callback_after_step = rp -> (held[] &= pure_mixing_holds(rp)),
        )
        return held[]
    end

    "The in-wall nodes of each in-wall column, keyed by the R index."
    function wall_columns(G)
        cols = Dict{Int, Vector{Int}}()
        for nid in G.nodes.in_wall_nids
            push!(get!(cols, G.nodes.rid[nid], Int[]), nid)
        end
        return cols
    end

    """
    What a set of nodes mixed by its own particles settles to: the n-weighted mean velocity
    (6.1) and the temperature that conserves the total energy per particle (6.2),
    `(3/2) e T∞ = ⟨(3/2) e T + ½ m u²⟩ₙ − ½ m u∞²`, in eV.
    """
    function mixed_state(nodes, V, n, u, T_eV; me = 9.1093837015e-31, ee = 1.602176634e-19)
        w = V[nodes] .* n[nodes]
        W = sum(w)
        u_inf = sum(w .* u[nodes]) / W
        ε = sum(w .* (1.5 .* ee .* T_eV[nodes] .+ 0.5 .* me .* u[nodes] .^ 2)) / W
        T_inf = (ε - 0.5 * me * u_inf^2) / (1.5 * ee)
        return (u = u_inf, T = T_inf)
    end
end

@testitem "mixing along straight field lines: each column settles to its own particle-weighted means" setup = [PureMixingRun] begin
    # B_R = 0, B_Z uniform: every R column is one field line, and with D_ZZ alone the columns
    # are independent and the stencil has no cross term.
    D, L = 500.0, 0.6                       # m²/s; the column length between the walls
    τ = L^2 / (π^2 * D)
    u0, T0 = 2.0e6, 10.0
    RP = pure_mixing_RP(; t_end_s = 6τ, D_along = D)
    G, pla = RP.G, RP.plasma
    inw = G.nodes.in_wall_nids
    # n and u correlated along the line, one full period across the column (what separates
    # the particle-weighted mean from the volume mean by ~10 %); T uniform, so any change in
    # T is heating, ≈ 0.8 eV from (6.2) at this shear
    shape = @. 1 + 0.5 * cos(2π * G.Z2D / L)
    pla.ne .= 0.0
    pla.ne[inw] .= 1.0e14 .* vec(shape)[inw]
    pla.ue_para .= u0 .* shape
    pla.Te_eV .= T0
    V = vec(G.inVol2D)
    n_i, u_i, T_i = vec(copy(pla.ne)), vec(copy(pla.ue_para)), vec(copy(pla.Te_eV))
    cols = wall_columns(G)
    @test length(cols) > 5
    expected = Dict(i => mixed_state(nodes, V, n_i, u_i, T_i) for (i, nodes) in cols)
    held = run_pure_mixing!(RP)
    @test held
    n1, u1, T1 = vec(pla.ne), vec(pla.ue_para), vec(pla.Te_eV)
    @test all(isfinite, u1[inw]) && all(isfinite, T1[inw])
    @test u1[inw] != u_i[inw]                                    # it evolved
    for (i, nodes) in cols
        # no exchange between columns: each keeps its particles
        @test sum(V[nodes] .* n1[nodes]) ≈ sum(V[nodes] .* n_i[nodes]) rtol = 1.0e-12
        # homogenized along the line (6τ: the first mode is down to e⁻⁶)
        @test maximum(u1[nodes]) - minimum(u1[nodes]) < 1.0e-2 * u0
        @test maximum(T1[nodes]) - minimum(T1[nodes]) < 1.0e-2 * T0
        # the column's particle-weighted momentum is what it started with, up to the O(Δt)
        # of the split between the continuity and the momentum solve, and the line settled
        # to it (6.1); the volume-weighted operator lost ~10 % here
        @test sum(V[nodes] .* n1[nodes] .* u1[nodes]) / sum(V[nodes] .* n1[nodes]) ≈ expected[i].u rtol = 1.0e-2
        @test all(x -> isapprox(x, expected[i].u; rtol = 1.0e-2), u1[nodes])
        # the sheared flow's kinetic energy became heat (6.2); without the heating T stays T0
        @test_broken all(x -> isapprox(x, expected[i].T; rtol = 1.0e-3), T1[nodes])
        @test expected[i].T > T0 + 0.5
    end
end

@testitem "ue_Te_operators: A_diffu is the particle-weighted mixing operator by policy, the density operator by the reference policy" setup = [PureMixingRun] begin
    using RAPID2D: ue_Te_operators, is_on_wall_pattern
    @test SimulationFlags{Float64}().mixing_policy isa ParticleMixing
    @test ParticleMixing <: MixingPolicy && VelocityDiffusion <: MixingPolicy
    RP = pure_mixing_RP(; t_end_s = 1.0e-6, D_along = 50.0, D_across = 5.0)
    G, pla, op = RP.G, RP.plasma, RP.operators
    inw = G.nodes.in_wall_nids
    pla.ne .= 0.0
    pla.ne[inw] .= 1.0e14 .* (1 .+ 0.4 .* sin.(3 .* vec(G.R2D)[inw]) .* cos.(2 .* vec(G.Z2D)[inw]))
    n = vec(pla.ne)
    f = 1.0e5 .* (1 .+ 0.5 .* sin.(2 .* vec(G.R2D)) .* cos.(3 .* vec(G.Z2D)))
    A = op.A_diffu_e
    # the default: M = N⁻¹(A N − diag(A n)) on the cached density operator and the CURRENT ne,
    # in its own buffer on the wall pattern; n (M f) + f (A n) = A (n f) row by row
    pops = ue_Te_operators(RP)
    @test pops.A_diffu === op.A_mix_e
    @test is_on_wall_pattern(op.A_mix_e)
    @test n .* (pops.A_diffu * f) .+ f .* (A * n) ≈ A * (n .* f) rtol = 1.0e-12
    @test pops.A_diffu.matrix.nzval != A.matrix.nzval
    # rebuilt from the density of the call: a changed ne gives a changed M
    pla.ne[inw] .*= 2 .+ sin.(vec(G.Z2D)[inw])
    n2 = vec(pla.ne)
    @test n2 .* (ue_Te_operators(RP).A_diffu * f) .+ f .* (A * n2) ≈ A * (n2 .* f) rtol = 1.0e-12
    # the reference: the density operator itself, as before this work
    RP.flags.mixing_policy = VelocityDiffusion()
    @test ue_Te_operators(RP).A_diffu === op.A_diffu_e
end

@testitem "mixing along straight field lines: the column momentum converges to first order in Δt" setup = [PureMixingRun] begin
    # The continuity and the momentum solve are split, so Σ V n u drifts by O(Δt) over a run;
    # halving Δt halves it. The volume-weighted operator's defect −2∫∇n·D∇u does not shrink.
    D, L = 500.0, 0.6
    t_end = 1.0e-4                            # ≈ 1.4 τ: the mixing is well under way
    errs = Float64[]
    for dt in (4.0e-6, 2.0e-6, 1.0e-6)
        RP = pure_mixing_RP(; dt = dt, t_end_s = t_end, D_along = D)
        G, pla = RP.G, RP.plasma
        inw = G.nodes.in_wall_nids
        shape = @. 1 + 0.5 * cos(2π * G.Z2D / L)
        pla.ne .= 0.0
        pla.ne[inw] .= 1.0e14 .* vec(shape)[inw]
        pla.ue_para .= 2.0e6 .* shape
        pla.Te_eV .= 10.0
        V = vec(G.inVol2D)
        P0 = sum(V[inw] .* vec(pla.ne)[inw] .* vec(pla.ue_para)[inw])
        @test run_pure_mixing!(RP)
        P1 = sum(V[inw] .* vec(pla.ne)[inw] .* vec(pla.ue_para)[inw])
        push!(errs, abs(P1 - P0) / abs(P0))
    end
    @test errs[end] < 1.0e-2
    @test 0.35 < errs[2] / errs[1] < 0.65
    @test 0.35 < errs[3] / errs[2] < 0.65
end
