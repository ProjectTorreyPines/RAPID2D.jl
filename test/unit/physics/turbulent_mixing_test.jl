# Mixing of u∥ and Te along field lines: the acceptance tests.
#
# A diffusive particle flux carries momentum and energy. Mixed by the particles that move,
# each field line settles to the n-weighted means of what it started with (design (6.1)),
# and the kinetic energy of the sheared flow it erased reappears as heat (6.2). The operator
# RAPID2D used until this work diffused u∥ and Te as if every node held the same number of
# electrons, so it settled to volume means and heated nothing.
# internal/docs/src/reference/electron-diffusive-transport.md; notes/design/turbulent-mixing-u-T.md §6.

@testsnippet PureMixingRun begin
    """
    The bulk electron tensor `D_across 𝟙 + (D_along − D_across) b_pol b_polᵀ` on every node,
    isotropic where B_pol = 0, frozen for the run (`flags.freeze_diffusion_tensor`): the per-step
    refresh then recomputes only `CT*` from it, so neither Bohm nor D∥ enters. The transport
    refresh is redone at once, so the electron operators carry the tensor from here on.
    """
    function prescribe_aligned_tensor!(RP; D_along, D_across = 0.0)
        F, tp = RP.fields, RP.transport
        @. tp.DRR = D_across + (D_along - D_across) * F.bpol_R^2
        @. tp.DRZ = (D_along - D_across) * F.bpol_R * F.bpol_Z
        @. tp.DZZ = D_across + (D_along - D_across) * F.bpol_Z^2
        RP.flags.freeze_diffusion_tensor = true
        RAPID2D.update_transport_quantities!(RP)
        return RP
    end

    """
    A run in which nothing but mixing acts on u∥ and Te: E = 0, no sources, no convection, no
    pressure, a reflective wall (albedo 1), the ions and the gas frozen, and the bulk tensor
    prescribed along the poloidal field line, which is the setup's uniform vertical field. The
    stored collision rates are zeroed on every step by `enforce_pure_mixing!`, because the
    momentum equation reads them whatever the flags say. The caller sets the initial state after
    this returns.
    """
    function pure_mixing_RP(;
            NR = 21, NZ = 47, dt = 1.0e-6, t_end_s, D_along, D_across = 0.0,
            Implicit = true, θ_transport = 0.5,
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
        )
        RP.flags.θ_imp.transport = θ_transport
        RP.flags.θ_imp.decay = θ_transport
        initialize!(RP)
        prescribe_aligned_tensor!(RP; D_along, D_across)
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

    "After each step: plasma above the floor everywhere in-wall, Te never clipped, no power but the mixing's own."
    function pure_mixing_holds(RP)
        pla, inw, cfg = RP.plasma, RP.G.nodes.in_wall_nids, RP.config
        return all(>(1.0), pla.ne[inw]) &&
            all(T -> cfg.min_Te < T < cfg.max_Te, pla.Te_eV[inw]) &&
            pla.ePowers.tot[inw] ≈ pla.ePowers.diffu[inw] .+ pla.ePowers.visc_heat[inw]
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
        # the sheared flow's kinetic energy became heat (6.2), up to the O(Δt) of the split
        @test all(x -> isapprox(x, expected[i].T; rtol = 1.0e-2), T1[nodes])
        @test expected[i].T > T0 + 0.5
    end
end

@testitem "ue_Te_operators: A_diffu is the particle-weighted operator of the density operator and the current ne" setup = [PureMixingRun] begin
    using RAPID2D: ue_Te_operators, is_on_wall_pattern
    RP = pure_mixing_RP(; t_end_s = 1.0e-6, D_along = 50.0, D_across = 5.0)
    G, pla, op = RP.G, RP.plasma, RP.operators
    inw = G.nodes.in_wall_nids
    pla.ne .= 0.0
    pla.ne[inw] .= 1.0e14 .* (1 .+ 0.4 .* sin.(3 .* vec(G.R2D)[inw]) .* cos.(2 .* vec(G.Z2D)[inw]))
    n = vec(pla.ne)
    f = 1.0e5 .* (1 .+ 0.5 .* sin.(2 .* vec(G.R2D)) .* cos.(3 .* vec(G.Z2D)))
    A = op.A_diffu_e
    # M = N⁻¹(A N − diag(A n)) on the cached density operator and the CURRENT ne,
    # in its own buffer on the wall pattern; n (M f) + f (A n) = A (n f) row by row
    pops = ue_Te_operators(RP)
    @test pops.A_diffu === op.A_visc_drift_e
    @test is_on_wall_pattern(op.A_visc_drift_e)
    @test n .* (pops.A_diffu * f) .+ f .* (A * n) ≈ A * (n .* f) rtol = 1.0e-12
    @test pops.A_diffu.matrix.nzval != A.matrix.nzval
    # rebuilt from the density of the call: a changed ne gives a changed M
    pla.ne[inw] .*= 2 .+ sin.(vec(G.Z2D)[inw])
    n2 = vec(pla.ne)
    @test n2 .* (ue_Te_operators(RP).A_diffu * f) .+ f .* (A * n2) ≈ A * (n2 .* f) rtol = 1.0e-12
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

@testitem "mixing along straight field lines: the column energy converges to first order in Δt" setup = [PureMixingRun] begin
    # With the heating P_mix = mₑ Γ_M(u∥) credited to Te, Σ V n (3/2 e T + ½ mₑ u²) drifts by
    # the split's O(Δt) over a run, so halving Δt at least halves it (measured: the energy
    # drift falls faster, by 0.29–0.32 per halving, the momentum drift by 0.4–0.6). Without
    # the heating the shear's kinetic energy is lost, ~3 % here, at every Δt.
    D, L = 500.0, 0.6
    t_end = 1.0e-4
    me, ee = 9.1093837015e-31, 1.602176634e-19
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
        energy() = sum(V[inw] .* vec(pla.ne)[inw] .* (1.5 .* ee .* vec(pla.Te_eV)[inw] .+ 0.5 .* me .* vec(pla.ue_para)[inw] .^ 2))
        E0 = energy()
        @test run_pure_mixing!(RP)
        push!(errs, abs(energy() - E0) / E0)
    end
    @test errs[end] < 1.0e-2
    @test errs[2] / errs[1] < 0.6
    @test errs[3] / errs[2] < 0.6
end

@testitem "viscous heating: mₑ Γ_M(u∥) per electron, in the ledger, the total and the snapshots; zero at uniform u, off by flag" setup = [PureMixingRun] begin
    using RAPID2D: ue_Te_operators, dissipation_rate!, update_electron_heating_powers!
    me = 9.1093837015e-31
    RP = pure_mixing_RP(; t_end_s = 1.0e-6, D_along = 50.0, D_across = 5.0)
    G, pla = RP.G, RP.plasma
    inw = G.nodes.in_wall_nids
    out = G.nodes.on_out_wall_nids
    pla.ne .= 0.0
    pla.ne[inw] .= 1.0e14 .* (1 .+ 0.4 .* sin.(3 .* vec(G.R2D)[inw]) .* cos.(2 .* vec(G.Z2D)[inw]))
    pla.ue_para .= 2.0e6 .* (1 .+ 0.5 .* sin.(2 .* G.R2D) .* cos.(3 .* G.Z2D))
    enforce_pure_mixing!(RP)
    @test RP.flags.Include_Te_visc_heat_term            # on by default
    update_electron_heating_powers!(RP)
    P = pla.ePowers.visc_heat
    # the formula: the dissipation rate of the operator Te diffuses with, times the mass
    Γ = dissipation_rate!(zeros(length(pla.ne)), ue_Te_operators(RP).A_diffu, vec(pla.ue_para))
    @test vec(P)[inw] ≈ me .* Γ[inw] rtol = 1.0e-12
    @test sum(vec(P)[inw]) > 0
    @test all(iszero, vec(P)[out])                      # masked like every other power
    @test pla.ePowers.tot ≈ pla.ePowers.diffu .+ P      # the only other power on this fixture
    # the snapshots carry it
    snap2D = RAPID2D.measure_snap2D(RP)
    @test snap2D.Pe_visc_heat == P
    snap0D = RAPID2D.measure_snap0D(RP)
    Ne = vec(pla.ne .* G.inVol2D)
    @test snap0D.Pe_visc_heat ≈ sum(vec(P) .* Ne) / sum(Ne)
    # uniform u: nothing to dissipate
    pla.ue_para .= 1.5e6
    update_electron_heating_powers!(RP)
    @test all(iszero, pla.ePowers.visc_heat)
    # off by flag: zero, and the total is diffusion alone
    pla.ue_para .= 2.0e6 .* (1 .+ 0.5 .* sin.(2 .* G.R2D) .* cos.(3 .* G.Z2D))
    RP.flags.Include_Te_visc_heat_term = false
    update_electron_heating_powers!(RP)
    @test all(iszero, pla.ePowers.visc_heat)
    @test pla.ePowers.tot ≈ pla.ePowers.diffu
end

@testitem "the prescribed tensor: aligned with the field line, no Bohm or D∥ in it, kept through a step" setup = [PureMixingRun] begin
    D_along, D_across = 50.0, 0.5
    # a straight vertical field: D_ZZ = D_along, D_RR = D_across, no cross term
    RP = pure_mixing_RP(; t_end_s = 1.0e-6, D_along, D_across)
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
    # the step's end rebuilds the transport; the prescription survives it
    run_simulation!(RP)
    @test RP.step == 1
    @test check_straight(RP.transport)
    # and the electron operator carries it: a uniform density diffuses nothing, a Z-ramp does
    pla = RP.plasma
    pla.ne .= 0.0
    pla.ne[inw] .= 1.0e14
    @test maximum(abs, (RP.operators.A_diffu_e * vec(pla.ne))[inw]) <= 1.0e-6 * 1.0e14 * D_along / G.dZ^2
    # thawed, the refresh puts the plasma tensor back
    RP.flags.freeze_diffusion_tensor = false
    RAPID2D.update_transport_quantities!(RP)
    @test !check_straight(RP.transport)
end

@testitem "the X-point field: R B_R = B' y, R B_Z = B' x, divergence-free, ψ analytic and tangent, FLF completes" setup = [PureMixingRun, XPointMixing] begin
    using RAPID2D: calculate_divergence, calculate_B_from_ψ, wall_gradient
    RP = xpoint_RP(; N = 41, t_end_s = 1.0e-6, D_along = 50.0)
    G, F = RP.G, RP.fields
    x = G.R2D .- XP.R0
    y = G.Z2D .- XP.Z0
    # the field, on every node, as the external field the step recombines from
    @test G.R2D .* F.BR ≈ XP.Bprime .* y
    @test G.R2D .* F.BZ ≈ XP.Bprime .* x
    @test F.BR_ext == F.BR && F.BZ_ext == F.BZ
    # the null sits on a grid node, and B_pol vanishes there
    i0, j0 = argmin(abs.(G.R1D .- XP.R0)), argmin(abs.(G.Z1D .- XP.Z0))
    @test F.Bpol[i0, j0] <= 1.0e-12 * XP.Bprime
    @test F.bpol_R[i0, j0] == 0 && F.bpol_Z[i0, j0] == 0
    # ∇·B = 0, exactly for central differences since R B_R is constant in R and B_Z in Z
    div = calculate_divergence(G, F.BR, F.BZ)
    @test maximum(abs, div[2:(end - 1), 2:(end - 1)]) <= 1.0e-12 * XP.Bprime / minimum(G.R1D) / G.dR
    # ψ_ext is the analytic flux of this field, in the code's sign convention
    @test F.ψ_ext ≈ XP.Bprime / 2 .* (x .^ 2 .- y .^ 2)
    BRψ, BZψ = calculate_B_from_ψ(G, F.ψ_ext)
    @test BRψ[2:(end - 1), 2:(end - 1)] ≈ F.BR[2:(end - 1), 2:(end - 1)] rtol = 1.0e-10
    @test BZψ[2:(end - 1), 2:(end - 1)] ≈ F.BZ[2:(end - 1), 2:(end - 1)] rtol = 1.0e-10
    # field lines follow the contours of ψ: B·∇ψ = 0 where the gradient is central
    gR, gZ = wall_gradient(G, F.ψ_ext)
    deep = [
        nid for nid in G.nodes.in_wall_nids if all(
                RAPID2D.is_in_wall(G, G.nodes.rid[nid] + di, G.nodes.zid[nid] + dj) for di in -1:1, dj in -1:1
            )
    ]
    tangent = (F.BR .* gR .+ F.BZ .* gZ)[deep]
    scale = (F.Bpol .* hypot.(gR, gZ))[deep]
    @test maximum(abs, tangent) <= 1.0e-12 * maximum(scale)
    # the field-line analysis was redone on the hyperbolic field, and the tensor follows it:
    # isotropic at the null, D_along along b_pol elsewhere
    inw = G.nodes.in_wall_nids
    @test all(isfinite, RP.flf.Lpol_tot[inw])
    tp = RP.transport
    @test tp.DRR[i0, j0] == 0 && tp.DZZ[i0, j0] == 0 && tp.DRZ[i0, j0] == 0
    @test tp.DRR ≈ 50.0 .* F.bpol_R .^ 2
    @test tp.DRZ ≈ 50.0 .* F.bpol_R .* F.bpol_Z
    # the step keeps the field: Ampère is off, and the external field is what it recombines
    run_simulation!(RP)
    @test G.R2D .* RP.fields.BR ≈ XP.Bprime .* y
end
