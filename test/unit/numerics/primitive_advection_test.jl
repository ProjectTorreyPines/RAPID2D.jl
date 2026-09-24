# (u·∇)f for a primitive variable (u∥, Te), derived from the SAME mass-flux divergence the
# continuity equation uses:  u·∇f ≡ [∇·(n u f) − f ∇·(n u)] / n.
# Constants are annihilated exactly, the interior is the donor-cell difference with the face
# metric when n and u are uniform, and at a wall face the two terms cancel so nothing outside
# the plasma is read.
# internal/docs/src/notes/design/wall-flux-channels.md §2.5–2.6; PLAN_wall-flux-channels.md PR2b.

@testitem "primitive advection: annihilates constants, donor-cell difference at uniform n, no wall gradient" begin
    using RAPID2D: primitive_advection_operator, build_face_flux_divergence
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
    # 2. uniform n and uniform u: the donor-cell difference, weighted on the R faces by the
    #    face metric R_{i−½}/R_i (uR > 0) and exactly (f_{j+1} − f_j)/ΔZ along Z (uZ < 0)
    n_u = zeros(G.NR, G.NZ)
    n_u[inw] .= 1.0e14
    uRu, uZu = 1.5e5, -6.0e4
    Cu = build_face_flux_divergence(G, fill(uRu, G.NR, G.NZ), fill(uZu, G.NR, G.NZ))
    Uu = primitive_advection_operator(Cu, vec(n_u); n_floor = 1.0)
    f = @. sin(3 * G.R2D) * cos(2 * G.Z2D)
    # deep: the whole 5×5 stencil footprint is in-wall (two cells of margin)
    deep = [
        nid for nid in inw if all(
                RAPID2D.is_in_wall(G, G.nodes.rid[nid] + di, G.nodes.zid[nid] + dj) for di in -2:2, dj in -2:2
            )
    ]
    @test !isempty(deep)
    Uf = Uu * vec(f)
    for nid in deep
        i, j = G.nodes.rid[nid], G.nodes.zid[nid]
        Rm = G.R2D[i, j] - G.dR / 2
        expected = uRu * Rm / G.R2D[i, j] * (f[i, j] - f[i - 1, j]) / G.dR +
            uZu * (f[i, j + 1] - f[i, j]) / G.dZ
        @test Uf[nid] ≈ expected rtol = 1.0e-10
    end
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

@testitem "wall gradient: one-sided at the wall, reads nothing outside, exact for a linear field" begin
    using RAPID2D: wall_gradient
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    inw = G.nodes.in_wall_nids
    # linear inside the wall, poison outside: a stencil that touched the band would show it
    f = @. 3.0 + 2.0e3 * G.R2D - 7.0e2 * G.Z2D
    f[G.nodes.on_out_wall_nids] .= NaN
    gR, gZ = wall_gradient(G, f)
    @test all(iszero, gR[G.nodes.on_out_wall_nids])
    @test all(iszero, gZ[G.nodes.on_out_wall_nids])
    @test all(x -> isapprox(x, 2.0e3; rtol = 1.0e-10), gR[inw])
    @test all(x -> isapprox(x, -7.0e2; rtol = 1.0e-10), gZ[inw])
end
