# Averages along the field line through each node, weighted by dl/B_pol: from wall to wall on an
# open line, once around on a closed one. Straight lines give the averages in closed form; on a
# closed line the average is the flux-surface average at the node's level.
# internal/docs/src/notes/plans/PLAN_flf-contours.md.

@testitem "Field-line averages: straight open lines, from wall to wall" tags = [:numerics] begin
    using RAPID2D: initialize_grid_geometry, FieldLineGeometry, field_line_average, line_average, to_grid,
        MarchingSquaresTracer, CubicTracer
    # ψ = Z: horizontal lines with B_pol = 1/R, so the weight is R dl. Between walls at Ra and Rb,
    # ⟨R⟩ = (2/3)(Rb³ − Ra³)/(Rb² − Ra²) on every line, and a field constant along a line averages
    # to itself.
    G = initialize_grid_geometry(41, 41, (1.0, 2.0), (-0.5, 0.5))
    inside = [1.2 < G.R1D[i] + G.dR / 2 < 1.8 for i in 1:G.NR, _ in 1:G.NZ]
    Ra = G.R1D[findfirst(i -> inside[i, 1], 1:G.NR)]
    Rb = G.R1D[findlast(i -> inside[i, 1], 1:G.NR) + 1]
    g = FieldLineGeometry(G, copy(G.Z2D); inside)
    nodes = [LinearIndices(G.R2D)[i, j] for i in 1:G.NR, j in 2:(G.NZ - 1) if Ra < G.R1D[i] < Rb]
    for tracer in (MarchingSquaresTracer(), CubicTracer())
        fla = field_line_average(g, nodes; tracer)
        @test all(fla.valid) && !any(fla.closed)
        @test line_average(fla, ones(G.NR, G.NZ)) ≈ ones(length(nodes))
        @test all(isapprox.(line_average(fla, G.R2D), (2 / 3) * (Rb^3 - Ra^3) / (Rb^2 - Ra^2); rtol = 2.0e-4))
        @test line_average(fla, G.Z2D) ≈ G.Z2D[nodes]
        @test to_grid(fla, line_average(fla, G.Z2D))[nodes] ≈ G.Z2D[nodes]
    end
end

@testitem "Field-line averages: a closed line is the flux-surface average at its level" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: FieldLineGeometry, field_line_average, line_average, CubicTracer, MarchingSquaresTracer
    G = test_grid(61)
    κ = 1.6
    ψ = ellipse_ψ(G; κ)
    nodes = [k for k in inside(ψ; a = 0.3) if ψ[k] > 0.05^2]
    g = FieldLineGeometry(G, ψ)
    for (tracer, tol) in ((CubicTracer(), 1.0e-3), (MarchingSquaresTracer(), 5.0e-3))
        fla = field_line_average(g, nodes; tracer)
        @test all(fla.valid) && all(fla.closed)
        avg = line_average(fla, 1 ./ G.R2D .^ 2)
        err = [abs(avg[m] / reference_average((R, Z) -> 1 / R^2, sqrt(ψ[k]); κ)[1] - 1) for (m, k) in pairs(nodes)]
        @test maximum(err) < tol
        @test line_average(fla, ones(size(ψ))) ≈ ones(length(nodes))
    end
end

@testitem "Field-line averages: no average without a poloidal field, and the run's lines" setup = [FLFStartup] tags = [:numerics] begin
    using RAPID2D: initialize_grid_geometry, FieldLineGeometry, field_line_average, line_average
    G = initialize_grid_geometry(21, 21, (1.0, 2.0), (-0.5, 0.5))
    fla = field_line_average(FieldLineGeometry(G, zeros(21, 21)), [LinearIndices((21, 21))[11, 11]])
    @test !any(fla.valid)
    @test all(isnan, line_average(fla, G.R2D))
    # The manual field is uniform and vertical: each in-wall node's line runs between the walls of
    # its column with a uniform weight, so ⟨Z⟩ is their midpoint.
    RP = startup_RP()
    fla = field_line_average(RP)
    @test all(fla.valid)
    Zmid = line_average(fla, RP.G.Z2D)
    for (m, n) in pairs(fla.nodes)
        i, j = RP.G.nodes.rid[n], RP.G.nodes.zid[n]
        @test Zmid[m] ≈ (RP.G.Z1D[j] + RP.flf.Lpol_forward[n] + RP.G.Z1D[j] - RP.flf.Lpol_backward[n]) / 2 atol = 1.0e-12
    end
end

@testitem "Field-line averages: a node handed to the fallback starts from clean weights" tags = [:numerics] begin
    using RAPID2D: initialize_grid_geometry, FieldLineGeometry, field_line_average, line_average, CubicTracer
    # Next to an O-point marching squares sees a point and hands the node to the bicubic tracer;
    # what its own attempt marked must not hide the fallback's weights.
    G = initialize_grid_geometry(17, 17, (1.0, 2.0), (-0.5, 0.5))
    g = FieldLineGeometry(G, @. -((G.R2D - 1.51)^2 + (G.Z2D - 0.01)^2))
    node = [LinearIndices((17, 17))[9, 9]]
    fla, ref = field_line_average(g, node), field_line_average(g, node; tracer = CubicTracer())
    @test all(fla.valid)
    @test line_average(fla, ones(17, 17)) ≈ [1.0]
    @test line_average(fla, G.R2D) ≈ line_average(ref, G.R2D)
end
