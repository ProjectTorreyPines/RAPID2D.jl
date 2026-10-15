# The closed-surface flux constraint in a run: u∥ is solved with n u∥/B a surface function on
# the closed region inside the LCFS, then remapped to the density the step left, and the
# kinetic energy the constraint removes or the remap changes is booked.

@testsnippet ClosedSurfaceRun begin
    using RAPID2D: SimulationConfig, SimulationFlags, RAPID, initialize!, flf_analysis_field_lines_rz_plane!,
        combine_external_and_self_fields!, calculate_B_from_ψ!, update_transport_quantities!,
        SurfaceFluxConstraint, FreeParallelFlow, update_closed_surface_basis!

    """
    A box wall with prescribed closed surfaces: an elliptic ψ_ext about (1.5113, 0.0071), the external
    B taken from that ψ (both, so the tracer and the field agree), Ampère off; a smooth density,
    10 eV and a drift that varies along the surfaces.
    """
    function closed_RP(; policy = SurfaceFluxConstraint(), NR = 31, NZ = 41, flags...)
        config = SimulationConfig{Float64}(
            NR = NR, NZ = NZ, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-6, t_end_s = 1.0e-5, snap0D_Δt_s = 1.0e-6, R0B0 = 3.0, prefilled_gas_pressure = 5.0e-3,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(; Ampere = false, Gas_evolve = false, closed_surface_flow = policy, flags...)
        initialize!(RP)
        G, F, pla = RP.G, RP.fields, RP.plasma
        # the axis off the nodes: a B_pol null on a node makes the transport operators NaN today
        @. F.ψ_ext = 2.0e-3 * ((G.R2D - 1.5113)^2 + ((G.Z2D - 0.0071) / 1.3)^2)
        calculate_B_from_ψ!(G, F.ψ_ext, F.BR_ext, F.BZ_ext)
        fill!(F.ψ_self, 0.0); fill!(F.BR_self, 0.0); fill!(F.BZ_self, 0.0)
        combine_external_and_self_fields!(RP)
        inwall = G.nodes.state .> 0.5
        @. pla.ne = ifelse(inwall, 1.0e17 * (1 + 0.4 * (G.R2D - 1.5)), 0.0)
        pla.ni .= pla.ne
        fill!(pla.Te_eV, 10.0)
        @. pla.ue_para = ifelse(inwall, 1.0e4 * (1 + 0.5 * G.Z2D), 0.0)
        update_transport_quantities!(RP)
        flf_analysis_field_lines_rz_plane!(RP)
        update_closed_surface_basis!(RP)
        return RP
    end

    "n u∥/B on the constrained nodes against its surface function at the knots."
    function surface_flux_error(RP)
        b, pla, F = RP.closed_surface.basis, RP.plasma, RP.fields
        sel = findall(<=(1), b.ψN)
        k = b.nodes[sel]
        f = pla.ne[k] .* pla.ue_para[k] ./ F.Btot[k]
        return maximum(abs, f .- b.knot_weights[sel, :] * RP.closed_surface.solve.Γe_para_over_B) / maximum(abs, f)
    end
end

@testitem "Closed-surface run: below the gate u∥ is a surface flux, and the removed energy is booked" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: update_ue_para!
    RP = closed_RP()
    s = RP.closed_surface
    @test s.basis !== nothing && length(s.basis.nodes) > 20
    update_ue_para!(RP)
    @test length(s.solve.Γe_para_over_B) == length(s.basis.knots)
    @test surface_flux_error(RP) < 1.0e-10
    # the first constrained step removes the non-surface part of the drift
    @test s.ledger.activation > 0 && s.ledger.projection == 0
    update_ue_para!(RP)
    @test surface_flux_error(RP) < 1.0e-10
    @test s.ledger.projection >= 0
    # the second solve on the same basis refactorizes the constrained system by value
    @test s.u_system.solver.nfactor == 2 && s.u_system.solver.nsymbolic == 1
end

@testitem "Closed-surface run: above the gate the coupled step is projected onto a surface flux" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: initialize_coupled_fields!, prepare_timestep!, solve_combined_momentum_Ampere_equations_with_coils!,
        run_simulation!
    RP = closed_RP(; Ampere = true)
    initialize_coupled_fields!(RP)                     # what run_simulation! does at step 0
    prepare_timestep!(RP)
    # the drift's current is above the gate, so solve_timestep! takes the coupled route
    @test abs(sum(RP.plasma.Jϕ) * RP.G.dR * RP.G.dZ) >= RP.flags.Ampere_Itor_threshold
    ψ0 = copy(RP.fields.ψ_self)
    solve_combined_momentum_Ampere_equations_with_coils!(RP; NamedTuple(RP.flags.ampere_picard)...)
    s = RP.closed_surface
    @test RP.fields.ψ_self != ψ0                      # the coupled solve's ψ is kept
    @test length(s.solve.Γe_para_over_B) == length(s.basis.knots) && all(isfinite, s.solve.Γe_para_over_B)
    @test surface_flux_error(RP) < 1.0e-10
    @test s.ledger.activation > 0

    # and through the run loop, step after step
    RP = closed_RP(; Ampere = true)
    RP.config.t_end_s = RP.t_end_s = RP.time_s + 5 * RP.dt
    run_simulation!(RP)
    @test !isempty(RP.closed_surface.solve.Γe_para_over_B) && all(isfinite, RP.plasma.ue_para) && all(isfinite, RP.fields.ψ_self)
    snap = RP.diagnostics.snaps0D[end]
    @test isfinite(snap.cs_E_projection) && snap.cs_E_projection >= 0
end

@testitem "Closed-surface block: Ampère's law holds for the constrained current" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: initialize_coupled_fields!, prepare_timestep!, solve_combined_momentum_Ampere_equations_with_coils!,
        uψ_coupling_terms, bulk_ion_charge, BlockConstraint, PostProjection
    # the interior rows of the coupled block's ψ equation, ΔGS ψ + c_ψu u∥ = −μ0 R n_i e Z u_i∥ bϕ
    # (no coils in the grid), with the u∥ the step keeps
    function ampere_residual(RP)
        G, pla, F = RP.G, RP.plasma, RP.fields
        ee, μ0 = RP.config.constants.ee, RP.config.constants.μ0
        Zi = bulk_ion_charge(RP)
        _, c_ψu = uψ_coupling_terms(RP)
        ions = vec(@. -μ0 * G.R2D * pla.ni * ee * Zi * pla.ui_para * F.bϕ)
        res = RP.operators.ΔGS.matrix * vec(F.ψ_self) .+ c_ψu .* vec(pla.ue_para) .- ions
        res[G.BDY_idx] .= 0
        return maximum(abs, res) / maximum(abs, c_ψu .* vec(pla.ue_para))
    end
    res = Dict()
    for coupling in (BlockConstraint(), PostProjection())
        RP = closed_RP(; Ampere = true, policy = SurfaceFluxConstraint(; coupling))
        initialize_coupled_fields!(RP)
        prepare_timestep!(RP)
        solve_combined_momentum_Ampere_equations_with_coils!(RP; NamedTuple(RP.flags.ampere_picard)...)
        @test surface_flux_error(RP) < 1.0e-10
        res[coupling] = ampere_residual(RP)
    end
    # the block solves ψ with the constrained drift; the post-projection with the free one
    @test res[BlockConstraint()] < 1.0e-10
    @test res[PostProjection()] > 1.0e-3
end

@testitem "Closed-surface block: its u∥, n u∥/B and Rue_ei are the below-gate solve's at the field it induces" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: initialize_coupled_fields!, prepare_timestep!, solve_combined_momentum_Ampere_equations_with_coils!,
        update_ue_para!
    # The coupled block and the below-gate u∥ system hold the same constrained equation; given the
    # step's induced field bϕ Eϕ_self, the two give the same u∥, surface flux and friction ledger.
    a, b = (closed_RP(; Ampere = true, E_para_self_ES = false) for _ in 1:2)
    for RP in (a, b)
        initialize_coupled_fields!(RP)
        prepare_timestep!(RP)
    end
    u0 = copy(a.plasma.ue_para)
    solve_combined_momentum_Ampere_equations_with_coils!(a; NamedTuple(a.flags.ampere_picard)...)
    @. b.fields.E_para_tot = b.fields.E_para_ext + b.fields.bϕ * a.fields.Eϕ_self
    update_ue_para!(b)
    Δu = a.plasma.ue_para .- u0
    @test maximum(abs, Δu) > 1.0e-3 * maximum(abs, u0)
    @test maximum(abs, a.plasma.ue_para .- b.plasma.ue_para) <= 1.0e-10 * maximum(abs, Δu)
    Γa, Γb = a.closed_surface.solve.Γe_para_over_B, b.closed_surface.solve.Γe_para_over_B
    @test maximum(abs, Γa .- Γb) <= 1.0e-10 * maximum(abs, Γb)
    @test maximum(abs, a.plasma.Rue_ei .- b.plasma.Rue_ei) <= 1.0e-10 * maximum(abs, b.plasma.Rue_ei)
end

@testitem "Closed-surface block: the direct solve agrees, and one symbolic analysis serves a basis" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: initialize_coupled_fields!, prepare_timestep!, solve_combined_momentum_Ampere_equations_with_coils!,
        AndersonOuterSolve, DirectOuterSolve
    tight = (tolerance = 1.0e-12, max_iter = 200, E_floor = 0.0, I_floor = 0.0)
    a, b = (closed_RP(; Ampere = true) for _ in 1:2)
    for RP in (a, b)
        initialize_coupled_fields!(RP)
        prepare_timestep!(RP)
    end
    solve_combined_momentum_Ampere_equations_with_coils!(a; tight..., method = AndersonOuterSolve())
    solve_combined_momentum_Ampere_equations_with_coils!(b; tight..., method = DirectOuterSolve())
    agree(x, y) = maximum(abs, x .- y) <= 1.0e-10 * maximum(abs, y)
    @test a.diagnostics.ampere_picard.nunconverged == 0 && b.diagnostics.ampere_picard.nunconverged == 0
    @test agree(b.plasma.ue_para, a.plasma.ue_para)
    @test agree(b.fields.ψ_self, a.fields.ψ_self)
    @test agree(b.closed_surface.solve.Γe_para_over_B, a.closed_surface.solve.Γe_para_over_B)
    # a second step on the same basis refactorizes the block by value; the plain block is not used
    prepare_timestep!(a)
    solve_combined_momentum_Ampere_equations_with_coils!(a; tight...)
    sys = a.closed_surface.block_system
    @test sys.solver.nfactor == 2 && sys.solver.nsymbolic == 1
    @test a.operators.uψ_solver.nfactor == 0
end

@testitem "Closed-surface block: a solve that does not converge keeps the surface flux of its best evaluation" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: initialize_coupled_fields!, prepare_timestep!, solve_combined_momentum_Ampere_equations_with_coils!,
        AndersonOuterSolve
    # Relaxed (m = 0) with a weight of 4, every iterate after the first is worse; cut at
    # max_iter = 3 the solve accepts its first evaluation, as the solve cut at max_iter = 1 does.
    # The surface flux it keeps must be that evaluation's, with its u∥ and ψ.
    function first_step(; relaxation_w = 0.5, picard...)
        RP = closed_RP(; Ampere = true)
        initialize_coupled_fields!(RP)
        prepare_timestep!(RP)
        redirect_stderr(devnull) do
            solve_combined_momentum_Ampere_equations_with_coils!(
                RP; method = AndersonOuterSolve(; memory = 0, relaxation_w), E_floor = 0.0, I_floor = 0.0, picard...,
            )
        end
        return RP
    end
    ref = first_step(; max_iter = 1)
    cut = first_step(; relaxation_w = 4.0, max_iter = 3)
    @test cut.diagnostics.ampere_picard.last_niter == 3 && cut.diagnostics.ampere_picard.nunconverged == 1
    @test cut.closed_surface.solve.Γe_para_over_B == ref.closed_surface.solve.Γe_para_over_B
    @test cut.plasma.ue_para == ref.plasma.ue_para && cut.fields.ψ_self == ref.fields.ψ_self
    @test surface_flux_error(cut) < 1.0e-10
end

@testitem "Closed-surface run: the gate puts the constraint's last solve back" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: update_ue_para!, below_gate_entry_state, restore_below_gate_entry_state!
    RP = closed_RP()
    update_ue_para!(RP)                                # the previous step's solve
    s = RP.closed_surface
    saved() = (
        copy(s.solve.Γe_para_over_B), s.solve.signature, copy(s.solve.ΓR), copy(s.solve.ΓZ), copy(s.solve.compression),
        s.ledger.activation, s.ledger.basis_change, s.ledger.projection, s.ledger.remap,
    )
    before = saved()
    entry = below_gate_entry_state(RP)
    RP.plasma.ue_para .*= 1.5                          # a below-gate attempt, discarded at the gate
    update_ue_para!(RP)
    @test s.solve.ΓR != before[3]
    restore_below_gate_entry_state!(RP, entry)
    @test saved() == before
end

@testitem "Closed-surface run: free flow builds no basis" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: update_ue_para!
    RP = closed_RP(; policy = FreeParallelFlow())
    @test RP.closed_surface.basis === nothing
    update_ue_para!(RP)
    @test isempty(RP.closed_surface.solve.Γe_para_over_B) && RP.closed_surface.ledger.activation == 0
end

@testitem "Closed-surface run: the remap keeps the surface flux on the new density" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: update_ue_para!, remap_surface_flux!
    RP = closed_RP()
    update_ue_para!(RP)
    s, pla, G = RP.closed_surface, RP.plasma, RP.G
    J0 = copy(pla.Jϕ)
    k = s.basis.nodes[s.basis.ψN .<= 1]
    @. pla.ne[k] *= 1 + 0.2 * sin(3 * G.R2D[k])
    remap_surface_flux!(RP)
    @test surface_flux_error(RP) < 1.0e-12
    # the current follows the surfaces, not the density: n u∥ is unchanged where u∥ is constrained
    @test pla.Jϕ[k] ≈ (pla.ne[k] .* pla.ue_para[k] .* RP.config.constants.qe .* RP.fields.bϕ[k])
    @test s.ledger.remap != 0
end

@testitem "Closed-surface run: ionization dilutes the drift once" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: update_ue_para!, remap_surface_flux!
    RP = closed_RP(; Include_ud_convec_term = false, Include_ud_pressure_term = false, Include_ud_diffu_term = false)
    s, pla, F = RP.closed_surface, RP.plasma, RP.fields
    b = s.basis
    sel = findall(<=(1), b.ψN)
    k = b.nodes[sel]
    # a drift that is already a surface flux, no field, ionization only
    Γ_over_B0 = [1.0e21 * (1 + x) for x in b.knots]
    fill!(pla.ue_para, 0.0)
    pla.ue_para[k] .= (F.Btot[k] ./ pla.ne[k]) .* (b.knot_weights[sel, :] * Γ_over_B0)
    fill!(F.E_para_tot, 0.0)
    fill!(pla.ν_en_mom_tot, 0.0); fill!(pla.ν_ei_eff, 0.0); fill!(pla.ν_en_iz_tot, 1.0e6)
    nu0 = pla.ne[k] .* pla.ue_para[k]
    update_ue_para!(RP)
    @test s.solve.Γe_para_over_B ≈ Γ_over_B0                  # momentum density untouched by newborn electrons
    pla.ne[k] .*= 2.0                      # the step's ionization, as continuity would leave it
    remap_surface_flux!(RP)
    @test pla.ne[k] .* pla.ue_para[k] ≈ nu0   # diluted once: n u∥ kept, u∥ halved
end

@testitem "Closed-surface run: ExpRB with the constraint is refused" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: update_ue_para!, ExpRB, validate_scheme_flags, SimulationFlags, SurfaceFluxConstraint
    # at setup, before closed surfaces form partway through a run
    flags = SimulationFlags{Float64}(; closed_surface_flow = SurfaceFluxConstraint())
    flags.scheme.decay = ExpRB
    @test_throws ArgumentError validate_scheme_flags(flags)
    # and at the u∥ solve, for flags changed after initialize!
    RP = closed_RP()
    RP.flags.scheme.decay = ExpRB
    @test_throws ArgumentError update_ue_para!(RP)
end

@testitem "Closed-surface run: a few steps of the run loop with the constraint" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: run_simulation!
    RP = closed_RP()
    RP.config.t_end_s = RP.time_s + 5 * RP.dt
    RP.t_end_s = RP.config.t_end_s
    run_simulation!(RP)
    @test RP.closed_surface.basis !== nothing
    snap = RP.diagnostics.snaps0D[end]
    @test snap.step == RP.step
    @test snap.cs_nodes == length(RP.closed_surface.basis.nodes)
    @test snap.Ke_tot > 0
    @test isfinite(snap.cs_E_activation) && snap.cs_E_activation > 0
end

@testitem "Closed-surface density: the parallel flow leaves Ω_c unmoved, the perpendicular drift stays" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: update_ue_para!, solve_electron_continuity_equation!, cache_electron_operators!
    flags = (; diffu = false, src = false, mean_ExB = false, turb_ExB_mixing = false)
    RPc = closed_RP(; flags...)
    RPf = closed_RP(; policy = FreeParallelFlow(), flags...)
    Ω = RPc.closed_surface.basis.nodes
    inΩ = falses(RPc.G.NR, RPc.G.NZ)
    inΩ[Ω] .= true
    for (RP, upar) in ((RPc, 1.0e5), (RPf, 0.0))
        pla = RP.plasma
        # a parallel drift on Ω_c only (none in the free run), the same perpendicular drift in both
        @. pla.ue_para = ifelse(inΩ, upar * (1 + 0.5 * RP.G.Z2D), 0.0)
        @. pla.ueR = pla.ue_para * RP.fields.bR + 300.0
        @. pla.ueZ = pla.ue_para * RP.fields.bZ - 200.0
        cache_electron_operators!(RP)
    end
    n0 = copy(RPc.plasma.ne)
    solve_electron_continuity_equation!(RPc)
    solve_electron_continuity_equation!(RPf)
    @test RPc.plasma.ne ≈ RPf.plasma.ne
    @test RPc.plasma.ne != n0                       # the perpendicular drift did move it
end

@testitem "Closed-surface transport: Te and u∥ ride the stream-function flux" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: update_ue_para!, ue_Te_operators, cache_electron_operators!, pressure_operator
    RP = closed_RP(; mean_ExB = false, turb_ExB_mixing = false)
    update_ue_para!(RP)                               # n u∥/B and the stream-function fluxes of the step
    s, pla, G = RP.closed_surface, RP.plasma, RP.G
    @. pla.ueR = pla.ue_para * RP.fields.bR          # no perpendicular drift
    @. pla.ueZ = pla.ue_para * RP.fields.bZ
    cache_electron_operators!(RP)
    ops = ue_Te_operators(RP)
    Ω = s.basis.nodes
    V = 2π .* G.R2D .* G.dR .* G.dZ
    # a constant is carried unchanged, and Σ V n (u·∇)f vanishes: the flux is divergence-free
    @test maximum(abs, (ops.A_adv * ones(length(pla.ne)))[Ω]) < 1.0e-6 * maximum(abs, ops.A_adv.matrix)
    f = sin.(3 .* vec(G.R2D)) .+ vec(G.Z2D)
    Af = ops.A_adv * f
    @test abs(sum(vec(V)[Ω] .* vec(pla.ne)[Ω] .* Af[Ω])) < 1.0e-8 * sum(abs.(vec(V)[Ω] .* vec(pla.ne)[Ω] .* Af[Ω]))
    # on Ω_c the compression is the pressure operator's adjoint: −(B_pᵀ Γ_over_B)/(n V)
    Bp = pressure_operator(s.basis, log.(max.(pla.ne, 1.0)))
    c = Bp' * s.solve.Γe_para_over_B
    @test ops.div_u[Ω] ≈ -c ./ (pla.ne[Ω] .* V[Ω])
end

@testitem "Closed-surface transport: Te's explicit convection takes the carriage and compression of its implicit half" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: update_ue_para!, ue_Te_operators, cache_electron_operators!, update_electron_heating_powers!
    # The θ-scheme for Te puts (1 − θ) of P_conv = −1.5 (u·∇)Te − Te ∇·u on the right-hand side,
    # from ePowers.conv, and θ of it on the left, from ue_Te_operators. On the closed region both
    # halves must carry Te by the stream-function flux and compress it with the pressure
    # operator's adjoint, or the parallel carriage and the compression act with weight θ only.
    RP = closed_RP(; mean_ExB = false, turb_ExB_mixing = false)
    update_ue_para!(RP)                               # n u∥/B and the stream-function fluxes of the step
    pla, G = RP.plasma, RP.G
    @. pla.ueR = pla.ue_para * RP.fields.bR          # no perpendicular drift: all carriage is parallel
    @. pla.ueZ = pla.ue_para * RP.fields.bZ
    cache_electron_operators!(RP)
    @. pla.Te_eV = 10.0 + 4.0 * sin(3 * G.R2D) * cos(2 * G.Z2D)   # varies along the surfaces
    update_electron_heating_powers!(RP)
    ops = ue_Te_operators(RP)
    ee = RP.config.constants.ee
    implicit_half = ee .* (-1.5 .* (ops.A_adv * pla.Te_eV) .- pla.Te_eV .* ops.div_u)
    Ω = RP.closed_surface.basis.nodes
    @test maximum(abs, implicit_half[Ω]) > 0           # the parallel carriage does act on Ω_c
    @test pla.ePowers.conv[Ω] ≈ implicit_half[Ω]
end

@testitem "DEFECT: a depleted sector of the closed surfaces overheats under the constraint" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: run_simulation!
    # The constraint holds J∥/B constant on a surface, so a sector a thousand times thinner carries the
    # same current density with u = (n u∥/B) B/n: the drift passes v_te and the friction and viscous heating
    # per electron grow as 1/n. Spitzer friction is outside its range there (kinetic current
    # limiting, runaway), and nothing along the surface (parallel conduction, pressure balance)
    # evens it out. The peak is a transient: the hot sector ionizes and refills (178 eV at step 87,
    # 45 eV at step 300). Free flow stays near the bulk. As thermal energy the sector stays small.
    # internal/docs/src/notes/issues/closed-surface-depleted-sector-overheats.md
    function depleted_run(policy; nsteps = 100, contrast = 1.0e-3)
        RP = closed_RP(; policy)
        G, pla = RP.G, RP.plasma
        inwall = G.nodes.state .> 0.5
        @. pla.ne = ifelse(inwall, 1.0e17 * ifelse(G.R2D > 1.62, contrast, 1.0), 0.0)
        pla.ni .= pla.ne
        fill!(pla.ue_para, 0.0)
        update_transport_quantities!(RP)
        flf_analysis_field_lines_rz_plane!(RP)
        update_closed_surface_basis!(RP)
        closed = [k for k in RP.flf.closed_surface_nids if inwall[k]]
        sector = filter(k -> G.R2D[k] > 1.62, closed)
        vte(k) = sqrt(pla.Te_eV[k] * RP.config.constants.ee / RP.config.constants.me)
        peak = (Te = 0.0, drift = 0.0)
        record(_) = (
            peak = (
                Te = max(peak.Te, maximum(pla.Te_eV[sector])),
                drift = max(peak.drift, maximum(k -> abs(pla.ue_para[k]) / vte(k), sector)),
            )
        )
        RP.config.t_end_s = RP.t_end_s = RP.time_s + nsteps * RP.dt
        run_simulation!(RP; callback_after_step = record)
        V = 2π .* G.R2D
        Eth(ks) = sum(k -> pla.ne[k] * pla.Te_eV[k] * V[k], ks)
        return (; peak.Te, peak.drift, share = Eth(sector) / Eth(closed))
    end
    c = depleted_run(SurfaceFluxConstraint())
    f = depleted_run(FreeParallelFlow())
    @test_broken c.Te < 50                       # 178 eV today, bulk ~6 eV
    @test_broken c.drift <= 1                    # 2.2 today, at the start
    @test f.Te < 20 && f.drift < 1               # the overheating is the constraint's
    @test c.share < 0.1                          # 2 % today: local, not an energy sink
end
