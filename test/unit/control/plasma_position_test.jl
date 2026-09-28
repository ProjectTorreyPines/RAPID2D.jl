# The current centroid used by position control, and its fallback when there is no current.
# A controller calls it after every step, including the first, when Jϕ is still zero.

@testitem "extract_plasma_position: the current centroid, and the geometric centre at zero current" begin
    config = SimulationConfig{Float64}(
        NR = 11, NZ = 13, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    fill!(RP.plasma.Jϕ, 0.0)
    @test extract_plasma_position(RP) ≈ sum(G.R1D) / length(G.R1D)     # no current: no 0/0
    RP.plasma.Jϕ[4, 7] = 2.0
    RP.plasma.Jϕ[8, 7] = 1.0
    @test extract_plasma_position(RP) ≈ (2 * G.R1D[4] + G.R1D[8]) / 3
end
