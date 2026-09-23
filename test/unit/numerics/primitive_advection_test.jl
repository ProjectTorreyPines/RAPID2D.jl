# (u·∇)f for a primitive variable (u∥, Te), derived from the SAME mass-flux divergence the
# continuity equation uses:  u·∇f ≡ [∇·(n u f) − f ∇·(n u)] / n.
# Constants are annihilated exactly, the interior matches the nodal upwind operator when n and
# u are uniform, and at a wall face the two terms cancel so nothing outside the plasma is read.
# internal/docs/src/notes/design/wall-flux-channels.md §2.5–2.6; PLAN_wall-flux-channels.md PR2b.

@testitem "primitive advection: annihilates constants, equals nodal upwind at uniform n, no wall gradient" begin
    using RAPID2D: primitive_advection_operator, build_face_flux_divergence, update_𝐮∇_operator!
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    inw = G.nodes.in_wall_nids
    uR = @. 1.0e5 * (1 + 0.3 * sin(G.Z2D))
    uZ = @. -4.0e4 * cos(G.R2D)
    C = build_face_flux_divergence(G, uR, uZ)
    n = zeros(G.NR, G.NZ)
    n[inw] .= 1.0e14 .* (1 .+ 0.5 .* sin.(4 .* G.R2D[inw]))
    U = primitive_advection_operator(C, vec(n); n_floor = 1.0)
    # 1. (u·∇)const = 0 exactly (to rounding of the two cancelling terms)
    @test all(x -> abs(x) < 1.0e-9 * 1.0e5 / G.dR, (U * fill(3.0, G.NR * G.NZ))[inw])
    # 2. uniform n and uniform u: identical to the nodal upwind u·∇ deep inside
    n_u = zeros(G.NR, G.NZ)
    n_u[inw] .= 1.0e14
    uRu = fill(1.5e5, G.NR, G.NZ)
    uZu = fill(-6.0e4, G.NR, G.NZ)
    Uu = primitive_advection_operator(build_face_flux_divergence(G, uRu, uZu), vec(n_u); n_floor = 1.0)
    update_𝐮∇_operator!(RP, uRu, uZu)
    f = @. sin(3 * G.R2D) * cos(2 * G.Z2D)
    # deep: the whole 5×5 stencil footprint is in-wall (two cells of margin)
    deep = [
        nid for nid in inw if all(
                RAPID2D.is_in_wall(G, G.nodes.rid[nid] + di, G.nodes.zid[nid] + dj) for di in -2:2, dj in -2:2
            )
    ]
    @test !isempty(deep)
    # The face form weights the upwind difference by R_face/R_node, the nodal form by 1: they
    # agree to O(ΔR/R) (same bound face_flux_test.jl pins), exactly in Z.
    @test (Uu * vec(f))[deep] ≈ (RP.operators.𝐮∇.matrix * vec(f))[deep] rtol = G.dR / minimum(G.R1D)
    # 3. rows outside the plasma are empty
    @test all(iszero, (U * vec(f))[G.nodes.on_out_wall_nids])
    # 4. a uniform f stays uniform after one explicit step even at the wall: no wall-induced gradient
    f0 = vec(fill(7.0, G.NR, G.NZ))
    f1 = f0 .- 1.0e-6 .* (U * f0)
    @test all(x -> x ≈ 7.0, f1[inw])
end

@testitem "wall divergence: one-sided at the wall, zero outside, matches central inside" begin
    using RAPID2D: wall_divergence, calculate_divergence
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    inw = G.nodes.in_wall_nids
    uR = @. 2.0e4 / G.R2D          # R·u_R constant: the R term vanishes exactly
    uZ = @. 5.0e3 * G.Z2D          # linear: exact for one-sided differences too
    d = wall_divergence(G, uR, uZ)
    @test all(iszero, d[G.nodes.on_out_wall_nids])
    @test all(isfinite, d[inw])
    # deep: the whole 5×5 stencil footprint is in-wall (two cells of margin)
    deep = [
        nid for nid in inw if all(
                RAPID2D.is_in_wall(G, G.nodes.rid[nid] + di, G.nodes.zid[nid] + dj) for di in -2:2, dj in -2:2
            )
    ]
    @test d[deep] ≈ calculate_divergence(G, uR, uZ)[deep] rtol = 1.0e-10
    @test all(x -> isapprox(x, 5.0e3; rtol = 1.0e-8), d[inw])
end
