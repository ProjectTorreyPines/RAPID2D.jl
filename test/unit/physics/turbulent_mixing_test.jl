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
        # the column's particle-weighted momentum is what it started with, and the line
        # settled to it (6.1); the volume-weighted operator loses ~10 % here
        @test_broken sum(V[nodes] .* n1[nodes] .* u1[nodes]) / sum(V[nodes] .* n1[nodes]) ≈ expected[i].u rtol = 1.0e-3
        @test_broken all(x -> isapprox(x, expected[i].u; rtol = 1.0e-2), u1[nodes])
        # the sheared flow's kinetic energy became heat (6.2); without the heating T stays T0
        @test_broken all(x -> isapprox(x, expected[i].T; rtol = 1.0e-3), T1[nodes])
        @test expected[i].T > T0 + 0.5
    end
end
