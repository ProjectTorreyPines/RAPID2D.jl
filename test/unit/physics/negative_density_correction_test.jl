# The post-step ledgers and floors, without a band pass.
#
# Nothing zeroes the band outside the wall any more: no transport operator has rows there
# and the ionization rates are zero there, so the band keeps what the initial condition put
# there (zero). What is left after a step is bookkeeping — the ionization count goes to the
# source ledgers, a negative density is floored and what the floor ADDED is booked as a
# negative loss, so Δ(Σ V·n) = src − loss keeps holding for both species.

@testitem "a negative-density correction adds particles and books a NEGATIVE loss, for both species" begin
    using RAPID2D: correct_negative_densities!
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 25,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8,
        t_end_s = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    RP.flags.update_ni_independently = true
    RP.flags.negative_n_correction = true
    G = RP.G
    inw = G.nodes.in_wall_nids
    V = vec(2π .* G.Jacob .* G.dR .* G.dZ)
    RP.plasma.ne .= 0.0
    RP.plasma.ni .= 0.0
    RP.plasma.ne[inw] .= 1.0e15
    RP.plasma.ni[inw] .= 1.0e15
    k = inw[length(inw) ÷ 2]                     # an interior cell
    RP.plasma.ne[k] = -1.0e12
    RP.plasma.ni[k] = -2.0e12
    Ne0 = sum(vec(RP.plasma.ne) .* V)
    Ni0 = sum(vec(RP.plasma.ni) .* V)
    T = RP.diagnostics.Ntracker
    le, li = T.cum0D_Ne_loss, T.cum0D_Ni_loss

    correct_negative_densities!(RP)

    Ne1 = sum(vec(RP.plasma.ne) .* V)
    Ni1 = sum(vec(RP.plasma.ni) .* V)
    @test RP.plasma.ne[k] >= 0
    @test RP.plasma.ni[k] == 0
    @test Ne1 > Ne0
    @test Ni1 > Ni0
    # what the floor added is a NEGATIVE loss, booked from the corrected cell itself…
    @test (T.cum0D_Ne_loss - le) ≈ -(RP.plasma.ne[k] - (-1.0e12)) * V[k] rtol = 1.0e-12
    @test (T.cum0D_Ni_loss - li) ≈ -(0.0 - (-2.0e12)) * V[k] rtol = 1.0e-12
    # …so Δ(Σ V·n) = src − loss with src = 0 (the global sums cancel to ~1e-11 relative)
    @test (T.cum0D_Ne_loss - le) ≈ -(Ne1 - Ne0) rtol = 1.0e-8
    @test (T.cum0D_Ni_loss - li) ≈ -(Ni1 - Ni0) rtol = 1.0e-8
    @test sum(T.cum2D_Ne_loss) ≈ T.cum0D_Ne_loss rtol = 1.0e-12
    @test sum(T.cum2D_Ni_loss) ≈ T.cum0D_Ni_loss rtol = 1.0e-12
    # nothing else moved
    rest = setdiff(inw, [k])
    @test all(==(1.0e15), vec(RP.plasma.ne)[rest])
    @test all(==(1.0e15), vec(RP.plasma.ni)[rest])
    @test all(==(0.0), vec(RP.plasma.ne)[G.nodes.on_out_wall_nids])
end

@testitem "slaved ions follow the corrected electrons" begin
    using RAPID2D: correct_negative_densities!
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 25,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8,
        t_end_s = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    RP.flags.update_ni_independently = false
    RP.flags.negative_n_correction = true
    inw = RP.G.nodes.in_wall_nids
    RP.plasma.ne .= 0.0
    RP.plasma.ne[inw] .= 1.0e15
    k = inw[length(inw) ÷ 2]
    RP.plasma.ne[k] = -1.0e12
    RP.plasma.ni .= 7.0                           # stale on purpose
    correct_negative_densities!(RP)
    @test RP.plasma.ne[k] >= 0
    @test RP.plasma.ni == RP.plasma.ne            # Z = 1 for the default species
    @test RP.diagnostics.Ntracker.cum0D_Ni_loss == 0
end
