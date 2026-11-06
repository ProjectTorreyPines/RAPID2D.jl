# The open lines' circuit: on straight lines from wall to wall the line EMF is E L_c and the resistance
# B η L_c, so the ends set the current: conducting ends pass the whole EMF (no shielding), a cap leaves
# γ = Φ/V, and an ion-supply end passes at most its wall's ion charge flux per flux-tube cross-section.

@testsnippet StraightCircuit begin
    using RAPID2D
    using RAPID2D: ManualSetup, SimulationFlags, SimulationConfig, RAPID, initialize!, LineCircuit, ConductingEnds,
        CappedEnds, IonSupplyEnds, ChargeSheetHeuristic, open_line_circuit!, flf_analysis_field_lines_rz_plane!,
        driven_parallel_field, parallel_resistivity
    const EE = 1.602176634e-19

    "A uniform plasma on straight open lines (B_Z, Bϕ = R0B0/R, box wall), Tₑ and nₑ held, under Eϕ."
    function straight_RP(policy; Eϕ = 0.3, Bpol = 0.1, ne = 1.0e17, Te = 5.0)
        config = SimulationConfig{Float64}(;
            device_Name = "manual", manual = ManualSetup{Float64}(; R = (1.2, 2.0), BR = 0.0, BZ = Bpol, Eϕ),
            NR = 17, NZ = 33, R0B0 = 3.0, prefilled_gas_pressure = 2.0e-3,
            dt = 1.0e-6, t_end_s = 1.0e-5, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(),
        )
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(;
            src = false, Te_evolve = false, Ti_evolve = false, Gas_evolve = false, Ampere = false,
            open_line_shielding = policy,
        )
        redirect_stderr(devnull) do
            initialize!(RP)
        end
        pla = RP.plasma
        inw = RP.G.nodes.in_wall_nids
        fill!(pla.ne, 0.0); pla.ne[inw] .= ne
        fill!(pla.ni, 0.0); pla.ni[inw] .= ne
        fill!(pla.Te_eV, Te); fill!(pla.Ti_eV, 1.0)
        RAPID2D.update_transport_quantities!(RP)
        flf_analysis_field_lines_rz_plane!(RP)
        return RP
    end

    applied(RP) = findall(vec(RP.open_line.applies))
end

@testitem "Open-line circuit: conducting ends pass the whole EMF, so nothing is shielded" setup = [StraightCircuit] begin
    RP = straight_RP(LineCircuit(ConductingEnds()))
    open_line_circuit!(RP)
    st, flf = RP.open_line, RP.flf
    k = applied(RP)
    @test length(k) > 100
    E = vec(driven_parallel_field(RP)); η = vec(parallel_resistivity(RP)); B = vec(RP.fields.Btot)
    # uniform line over its plasma part L: V = E L, R = B η L (the half cells at the walls are the gap),
    # so K = E/(B η) and E_res = E
    @test maximum(abs.(vec(st.line_emf)[k] ./ (E[k] .* flf.Lc_tot[k]) .- 1)) < 0.15
    @test maximum(abs.(vec(st.line_resistance)[k] ./ ((B .* η)[k] .* flf.Lc_tot[k]) .- 1)) < 0.15
    @test maximum(abs.(vec(st.line_emf)[k] ./ vec(st.line_resistance)[k] .- E[k] ./ (B .* η)[k]) ./ abs.(E[k] ./ (B .* η)[k])) < 1.0e-9
    @test maximum(abs.(vec(st.E_para_res)[k] .- E[k]) ./ abs.(E[k])) < 1.0e-9
    @test maximum(abs.(vec(st.gamma_local)[k])) < 1.0e-9
    @test all(isinf, vec(st.Jpara_over_B_limit)[k])
    # the heuristic policy computes nothing
    RP2 = straight_RP(ChargeSheetHeuristic())
    open_line_circuit!(RP2)
    @test !any(RP2.open_line.applies)
end

@testitem "Open-line circuit: a voltage cap leaves γ = Φ/V, and an ion-supply end passes its wall's ions" setup = [StraightCircuit] begin
    Φ = 5.0
    RP = straight_RP(LineCircuit(CappedEnds(Φ)))
    open_line_circuit!(RP)
    st = RP.open_line
    k = applied(RP)
    V = vec(st.line_emf)[k]
    @test all(abs.(V) .> Φ)                    # the lines' EMF is above the cap
    γ_line = 1 .- vec(st.Jpara_over_B)[k] .* vec(st.line_resistance)[k] ./ V
    @test maximum(abs.(γ_line .- Φ ./ abs.(V))) < 1.0e-9

    RP = straight_RP(LineCircuit(IonSupplyEnds()))
    # the ions' last step delivered a charge flux j_i to every wall face: the current per flux tube is
    # at most j_i/(μ B) at the ion-collecting end, μ = |b̂·n̂| the incidence on the face
    j_i = 2.0
    fill!(RP.open_line.ion_wall_charge_flux, j_i)
    open_line_circuit!(RP)
    st = RP.open_line
    k = applied(RP)
    F = RP.fields
    faces = RP.transport.wall_faces
    μB_top = maximum(abs(F.bZ[f.rid, f.zid]) * F.Btot[f.rid, f.zid] for f in faces if f.outward == (0, 1))
    @test all(isfinite, vec(st.Jpara_over_B_limit)[k])
    # the limit binds: |K| = limit < |V|/R, and it is the top (or bottom) face's supply per flux
    @test all(abs.(vec(st.Jpara_over_B)[k]) .≈ vec(st.Jpara_over_B_limit)[k])
    @test all(vec(st.Jpara_over_B_limit)[k] .< abs.(vec(st.line_emf)[k]) ./ vec(st.line_resistance)[k])
    @test minimum(vec(st.Jpara_over_B_limit)[k]) ≈ j_i / μB_top rtol = 0.05
    # the residual field is then the limited current's, and the shielding near one
    @test all(vec(st.gamma_local)[k] .> 0.9)
    # no supply at all: no current, full shielding
    fill!(RP.open_line.ion_wall_charge_flux, 0.0)
    open_line_circuit!(RP)
    @test all(iszero, vec(RP.open_line.Jpara_over_B)[k])
    @test all(vec(RP.open_line.gamma_local)[k] .≈ 1)
end
