# The closed-surface flux constraint in a run: u∥ is solved with n u∥/B a surface function on
# the closed region inside the cut, then remapped to the density the step left, and the
# kinetic energy the constraint removes or the remap changes is booked.
# internal/docs/src/notes/plans/PLAN_closed-surface-flux.md (D5, D8, D9, D13, D17, D19).

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

    "n u∥/B on the constrained nodes against P K."
    function surface_flux_error(RP)
        b, pla, F = RP.closed_surface.basis, RP.plasma, RP.fields
        sel = findall(<=(1), b.ψN)
        k = b.nodes[sel]
        f = pla.ne[k] .* pla.ue_para[k] ./ F.Btot[k]
        return maximum(abs, f .- b.P[sel, :] * RP.closed_surface.K) / maximum(abs, f)
    end
end

@testitem "Closed-surface run: below the gate u∥ is a surface flux, and the removed energy is booked" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: update_ue_para!
    RP = closed_RP()
    s = RP.closed_surface
    @test s.basis !== nothing && length(s.basis.nodes) > 20
    update_ue_para!(RP)
    @test length(s.K) == length(s.basis.knots)
    @test surface_flux_error(RP) < 1.0e-10
    # the first constrained step removes the non-surface part of the drift
    @test s.E_activation > 0 && s.E_projection == 0
    update_ue_para!(RP)
    @test surface_flux_error(RP) < 1.0e-10
    @test s.E_projection >= 0
end

@testitem "Closed-surface run: free flow builds no basis" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: update_ue_para!
    RP = closed_RP(; policy = FreeParallelFlow())
    @test RP.closed_surface.basis === nothing
    update_ue_para!(RP)
    @test isempty(RP.closed_surface.K) && RP.closed_surface.E_activation == 0
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
    @test s.E_remap != 0
end

@testitem "Closed-surface run: ionization dilutes the drift once" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: update_ue_para!, remap_surface_flux!
    RP = closed_RP(; Include_ud_convec_term = false, Include_ud_pressure_term = false, Include_ud_diffu_term = false)
    s, pla, F = RP.closed_surface, RP.plasma, RP.fields
    b = s.basis
    sel = findall(<=(1), b.ψN)
    k = b.nodes[sel]
    # a drift that is already a surface flux, no field, ionization only
    K0 = [1.0e21 * (1 + x) for x in b.knots]
    fill!(pla.ue_para, 0.0)
    pla.ue_para[k] .= (F.Btot[k] ./ pla.ne[k]) .* (b.P[sel, :] * K0)
    fill!(F.E_para_tot, 0.0)
    fill!(pla.ν_en_mom_tot, 0.0); fill!(pla.ν_ei_eff, 0.0); fill!(pla.ν_en_iz_tot, 1.0e6)
    nu0 = pla.ne[k] .* pla.ue_para[k]
    update_ue_para!(RP)
    @test s.K ≈ K0                        # momentum density untouched by newborn electrons
    pla.ne[k] .*= 2.0                      # the step's ionization, as continuity would leave it
    remap_surface_flux!(RP)
    @test pla.ne[k] .* pla.ue_para[k] ≈ nu0   # diluted once: n u∥ kept, u∥ halved
end

@testitem "Closed-surface run: ExpRB with the constraint is refused" setup = [ClosedSurfaceRun] tags = [:physics] begin
    using RAPID2D: update_ue_para!, ExpRB
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
