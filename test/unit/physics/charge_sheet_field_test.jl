# The open-line charge sheets' field (`estimate_electrostatic_field_effects!`): γ is the field's
# shape alone, a closed line carries no sheets, and every case of the field and its thresholds is
# finite. A current disc's self field gives closed lines around the disc and open ones out to the
# wall; plasma is put on the open lines so the sheets have charge to hold.

@testitem "charge-sheet field: γ keeps its shape on closed lines, which carry no sheets" setup = [CurrentDisc] begin
    using RAPID2D: estimate_electrostatic_field_effects!
    using RAPID2D.Statistics: mean
    RP, _ = current_disc()
    trace_closed_region!(RP)
    pla, F, G = RP.plasma, RP.fields, RP.G
    closed = RP.flf.closed_surface_nids
    open_in = setdiff(G.nodes.in_wall_nids, closed)
    pla.ne[open_in] .= 1.0e17
    estimate_electrostatic_field_effects!(RP)

    @test !isempty(closed) && !isempty(open_in)
    @test pla.γ_shape_fac ≈ min.(1.0, 0.65 .* F.Bpol ./ mean(F.Bpol[G.nodes.in_wall_nids]))
    @test all(>(0), pla.γ_shape_fac[closed])
    # the sheets are on open lines only, boundary nodes included
    @test all(iszero, F.ERZ_pla_mag[closed]) && all(iszero, F.ERZ_para_pla[closed])
    @test any(>(0), F.ERZ_pla_mag[open_in])
    # an open-line threshold is never reached on a closed line
    @test all(isinf, pla.nc_para[closed]) && all(isinf, pla.nc_perp[closed])
    @test all(isfinite, pla.nc_perp[open_in])
end

@testitem "charge-sheet field: finite where Bpol vanishes, and none without a poloidal field" setup = [CurrentDisc] begin
    using RAPID2D: estimate_electrostatic_field_effects!
    using RAPID2D.Statistics: mean
    RP, _ = current_disc()
    trace_closed_region!(RP)
    pla, F, G, tp = RP.plasma, RP.fields, RP.G, RP.transport
    open_in = setdiff(G.nodes.in_wall_nids, RP.flf.closed_surface_nids)
    pla.ne[open_in] .= 1.0e17
    all_finite() = all(
        a -> all(isfinite, a),
        (F.ERZ_pla_mag, F.ERZ_para_pla, F.E_para_tot, pla.mean_ExB_R, pla.mean_ExB_Z, F.ER, F.EZ, tp.Dpol_turb),
    )

    # an open node where Bpol vanishes: γ E_req at its limit 0.65 |E∥| B/⟨Bpol⟩ (γ → 0, E_req → ∞),
    # below the Debye field here
    k = open_in[length(open_in) ÷ 2]
    F.Bpol[k] = F.bpol_R[k] = F.bpol_Z[k] = 0.0
    E_drive = F.Eϕ_para_presc[k] + F.Eϕ_para_coils_pla[k]
    estimate_electrostatic_field_effects!(RP)
    @test all_finite()
    @test F.ERZ_pla_mag[k] ≈ 0.65 * abs(E_drive) * F.Btot[k] / mean(F.Bpol[G.nodes.in_wall_nids])
    @test isfinite(pla.nc_perp[k])

    # no poloidal field at all: no shape, no sheets
    foreach(a -> fill!(a, 0.0), (F.Bpol, F.bpol_R, F.bpol_Z))
    estimate_electrostatic_field_effects!(RP)
    @test all(iszero, pla.γ_shape_fac) && all(iszero, F.ERZ_pla_mag)
    @test all(isinf, pla.nc_perp)
    @test all_finite()
end
