# Electron wall: the Robin path for the electron continuity equation.
#
# PR1 of internal/docs/src/notes/plans/PLAN_wall-flux-channels.md. The flag keeps the
# legacy path (`ne[on/out_wall] = 0` each step, loss booked from the zeroed band) as the
# default until the Robin path is validated, so every existing result is bit-identical
# while the new one is built beside it.

@testitem "electron_wall flag: two spellings, default keeps the legacy path" begin
    f = SimulationFlags{Float64}()
    @test f.electron_wall === :zeroing
    @test SimulationFlags{Float64}(electron_wall = :robin).electron_wall === :robin
    c = SimulationConfig{Float64}(NR = 6, NZ = 6)
    @test c.electron_wall_albedo == 0.0
end
