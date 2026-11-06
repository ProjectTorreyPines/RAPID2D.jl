# The open lines' shielding in the fields: with `LineCircuit` the total E∥ on an open line is the
# circuit's residual field, the sheets' RZ field is what leaves it, and the drifts turn with its sign;
# with `ChargeSheetHeuristic` nothing changes.

@testsnippet LineShielding begin
    using RAPID2D
    using RAPID2D: ManualSetup, SimulationFlags, SimulationConfig, RAPID, initialize!, LineCircuit, ConductingEnds,
        IonSupplyEnds, ChargeSheetHeuristic, flf_analysis_field_lines_rz_plane!, estimate_electrostatic_field_effects!,
        driven_parallel_field, solve_timestep!

    "A uniform plasma on straight open lines under Eϕ, with a thin stretch (n/10) near the top wall."
    function shielded_RP(policy; Eϕ = 0.3, Bpol = 0.1, ne = 1.0e17, thin = true)
        config = SimulationConfig{Float64}(;
            device_Name = "manual", manual = ManualSetup{Float64}(; R = (1.2, 2.0), BR = 0.0, BZ = Bpol, Eϕ),
            NR = 17, NZ = 33, R0B0 = 3.0, prefilled_gas_pressure = 2.0e-3,
            dt = 1.0e-6, t_end_s = 5.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(),
        )
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(;
            src = false, Te_evolve = false, Ti_evolve = false, Gas_evolve = false, Ampere = false,
            open_line_shielding = policy,
        )
        redirect_stderr(devnull) do
            initialize!(RP)
        end
        G, pla = RP.G, RP.plasma
        inw = G.nodes.in_wall_nids
        fill!(pla.ne, 0.0); pla.ne[inw] .= ne
        thin && (pla.ne[G.Z2D .> 0.6] ./= 10)
        pla.ni .= pla.ne
        fill!(pla.Te_eV, 5.0); fill!(pla.Ti_eV, 1.0)
        RAPID2D.update_transport_quantities!(RP)
        flf_analysis_field_lines_rz_plane!(RP)
        estimate_electrostatic_field_effects!(RP)
        return RP
    end
end

@testitem "Open-line shielding: the circuit's residual is the total E∥, piled on the thin stretch" setup = [LineShielding] begin
    RP = shielded_RP(LineCircuit(ConductingEnds()))
    F, st, G = RP.fields, RP.open_line, RP.G
    k = findall(vec(st.applies))
    @test length(k) > 100
    # the total parallel field is the circuit's residual
    @test maximum(abs.(vec(F.E_para_tot)[k] .- vec(st.E_para_res)[k]) ./ abs.(vec(st.E_para_res)[k])) < 1.0e-9
    # conducting ends: the EMF of the whole line drives the current, so the thin stretch (η a few
    # times higher: the e–n part scales as 1/n, the Spitzer part not) takes more than the vacuum field
    # and the dense part less: γ < 0 there, > 0 here
    E = vec(driven_parallel_field(RP))
    thin = k[vec(G.Z2D)[k] .> 0.6]; dense = k[vec(G.Z2D)[k] .< 0.5]
    @test all(vec(st.gamma_local)[thin] .< -0.5)
    @test all(0.1 .< vec(st.gamma_local)[dense] .< 1)
    # the sheets' field sign follows: against the drive on the dense part, with it on the thin one
    @test all(vec(F.ERZ_pla_sign)[dense] .== 1)
    @test all(vec(F.ERZ_pla_sign)[thin] .== -1)
    @test all(vec(F.ERZ_pla_mag)[k] .>= 0)
    # the drift turns with the sign
    sgn_dense = sign(sum(vec(RP.plasma.mean_ExB_R)[dense]))
    @test all(sign.(vec(RP.plasma.mean_ExB_R)[dense]) .== sgn_dense)
    @test all(sign.(vec(RP.plasma.mean_ExB_R)[thin]) .== -sgn_dense)
    # a few steps run
    redirect_stdout(devnull) do
        for _ in 1:3
            solve_timestep!(RP)
        end
    end
    @test all(isfinite, RP.plasma.ne) && all(isfinite, RP.plasma.ue_para)
end

@testitem "Open-line shielding: the heuristic is untouched, and an ion-supply end shields the dense line" setup = [LineShielding] begin
    RP = shielded_RP(ChargeSheetHeuristic())
    @test !any(RP.open_line.applies)
    @test all(RP.fields.ERZ_pla_sign .== 1)
    # the heuristic's field: γ₀ B_pol/⟨B_pol⟩ of the cancelling field, capped by E_Debye
    F = RP.fields
    k = RP.G.nodes.in_wall_nids
    @test all(F.ERZ_pla_mag[k] .> 0)
    @test maximum(F.ERZ_pla_mag[k] .* F.Bpol[k] ./ F.Btot[k] ./ abs.(driven_parallel_field(RP)[k])) <= 1 + 1.0e-12

    RP = shielded_RP(LineCircuit(IonSupplyEnds()); thin = false)
    # before any ion step the walls supplied nothing: no current, the lines fully shielded
    st = RP.open_line
    kk = findall(vec(st.applies))
    @test all(vec(st.Jpara_over_B)[kk] .== 0)
    @test all(abs.(vec(RP.fields.E_para_tot)[kk]) .< 1.0e-12)
    # after a step the ions' wall flux is recorded and a current flows, below the vacuum drive
    redirect_stdout(devnull) do
        solve_timestep!(RP)
    end
    estimate_electrostatic_field_effects!(RP)
    @test any(RP.open_line.ion_wall_charge_flux .> 0)
    kk = findall(vec(RP.open_line.applies))
    @test all(0 .< vec(RP.open_line.gamma_local)[kk] .<= 1)
end
