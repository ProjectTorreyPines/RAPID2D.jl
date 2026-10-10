# Following field lines in the poloidal plane: the tracers.
#
# In axisymmetry the poloidal projection of a field line is a ψ contour. A tracer follows one
# from a node along ±B_pol and reports every vertex to a visitor; the fixtures here have contours
# known in closed form (straight lines, circles, hyperbolas), so the paths and their lengths can be
# checked directly. internal/docs/src/notes/plans/PLAN_flf-contours.md.

@testsnippet TracingFixtures begin
    using RAPID2D: initialize_grid_geometry, FieldLineGeometry, MarchingSquaresTracer, PathLengths,
        trace_from_node!

    "A uniform grid on R ∈ [1, 2]."
    test_grid(NR, NZ = NR; Z = (-0.5, 0.5)) = initialize_grid_geometry(NR, NZ, (1.0, 2.0), Z)

    "Records every vertex a tracer visits."
    struct Path
        R::Vector{Float64}
        Z::Vector{Float64}
        Bpol::Vector{Float64}
    end
    Path() = Path(Float64[], Float64[], Float64[])
    function RAPID2D.visit!(p::Path, R, Z, Bpol, Bϕ)
        push!(p.R, R)
        push!(p.Z, Z)
        push!(p.Bpol, Bpol)
        return p
    end
    poloidal_length(p::Path) = sum(hypot(p.R[k + 1] - p.R[k], p.Z[k + 1] - p.Z[k]) for k in 1:(length(p.R) - 1); init = 0.0)

    "The node of `G` nearest to (R, Z)."
    nearest(G, R, Z) = (argmin(abs.(G.R1D .- R)), argmin(abs.(G.Z1D .- Z)))
end

@testitem "Marching squares: a straight contour is followed exactly to the wall" setup = [TracingFixtures] tags = [:numerics] begin
    # ψ = R: vertical contours, B_pol = (∂ψ/∂R)/R ẑ = ẑ/R upward. With Bϕ = F/R the ratio
    # B/B_pol = √(1 + F²) is the same everywhere, so Lc = √(1 + F²) Lpol exactly.
    G = test_grid(21, 41; Z = (-1.0, 1.0))
    F = 3.0
    inside = [abs(G.Z1D[j] + G.dZ / 2) < 0.6 for _ in 1:G.NR, j in 1:G.NZ]   # wall on Z = ±0.6
    g = FieldLineGeometry(G, copy(G.R2D); Bϕ = F ./ G.R2D, inside)
    i, j = nearest(G, 1.5, 0.0)
    for (sgn, Zend) in ((1, 0.6), (-1, -0.6))
        p, L = Path(), PathLengths{Float64}()
        @test trace_from_node!(p, MarchingSquaresTracer(), g, i, j, sgn) === :wall
        @test trace_from_node!(L, MarchingSquaresTracer(), g, i, j, sgn) === :wall
        @test (p.R[1], p.Z[1]) == (G.R1D[i], G.Z1D[j])         # the node first
        @test all(==(G.R1D[i]), p.R)
        @test p.Z[end] ≈ Zend atol = 1.0e-12                   # stops on the wall edge
        @test L.Lpol ≈ 0.6 atol = 1.0e-12
        @test L.Lc ≈ 0.6 * sqrt(1 + F^2) rtol = 1.0e-12
        @test L.min_Bpol ≈ 1 / G.R1D[i] rtol = 1.0e-12
    end
end

@testitem "Marching squares: a closed contour returns to its node, to second order" setup = [TracingFixtures] tags = [:numerics] begin
    # ψ = r² about an axis off the nodes: circles, B_pol = 2r/R. With Bϕ = F/R the ratio
    # B/B_pol = √(1 + F²/4r²) is constant on a circle.
    R0, Z0, F = 1.5113, 0.0071, 3.0
    errors = Float64[]
    for N in (41, 81, 161)
        G = test_grid(N)
        ψ = @. (G.R2D - R0)^2 + (G.Z2D - Z0)^2
        g = FieldLineGeometry(G, ψ; Bϕ = F ./ G.R2D)
        i, j = nearest(G, R0 + 0.3, Z0)
        r = sqrt(ψ[i, j])
        p, L = Path(), PathLengths{Float64}()
        @test trace_from_node!(p, MarchingSquaresTracer(), g, i, j, 1) === :closed
        @test trace_from_node!(L, MarchingSquaresTracer(), g, i, j, 1) === :closed
        @test (p.R[end], p.Z[end]) == (p.R[1], p.Z[1])        # back at the node
        @test L.Lc / L.Lpol ≈ sqrt(1 + F^2 / (4r^2)) rtol = 1.0e-2
        push!(errors, abs(L.Lpol / (2π * r) - 1))
    end
    @test errors[1] < 2.0e-3
    @test errors[1] / errors[2] > 3 && errors[2] / errors[3] > 3     # second order
end

@testitem "Marching squares: forward is along B_pol" setup = [TracingFixtures] tags = [:numerics] begin
    # On the outer midplane of ψ = ±r², B_pol = (−∂ψ/∂Z, ∂ψ/∂R)/R points up for +r², down for −r².
    G = test_grid(41)
    R0, Z0 = 1.5113, 0.0071
    i, j = nearest(G, R0 + 0.3, Z0)
    for (s, up) in ((1, true), (-1, false))
        g = FieldLineGeometry(G, @. s * ((G.R2D - R0)^2 + (G.Z2D - Z0)^2))
        p = Path()
        trace_from_node!(p, MarchingSquaresTracer(), g, i, j, 1)
        k = findfirst(k -> hypot(p.R[k] - p.R[1], p.Z[k] - p.Z[1]) > 0, eachindex(p.R))
        @test (p.Z[k] > p.Z[1]) == up
        q = Path()
        trace_from_node!(q, MarchingSquaresTracer(), g, i, j, -1)
        k = findfirst(k -> hypot(q.R[k] - q.R[1], q.Z[k] - q.Z[1]) > 0, eachindex(q.R))
        @test (q.Z[k] > q.Z[1]) == !up
    end
end

@testitem "Marching squares: a wall cell ends the trace on its edge" setup = [TracingFixtures] tags = [:numerics] begin
    # Circles of ψ = r² cut by a wall on the row Z = Zw: both directions end on that row, and
    # together they cover the arc below it.
    G = test_grid(81)
    R0, Z0 = 1.5113, 0.0071
    ψ = @. (G.R2D - R0)^2 + (G.Z2D - Z0)^2
    jw = nearest(G, R0, 0.2)[2]
    Zw = G.Z1D[jw]
    inside = [j < jw for _ in 1:G.NR, j in 1:G.NZ]
    g = FieldLineGeometry(G, ψ; inside)
    i, j = nearest(G, R0 + 0.3, Z0)
    r = sqrt(ψ[i, j])
    paths = (Path(), Path())
    for (p, sgn) in zip(paths, (1, -1))
        @test trace_from_node!(p, MarchingSquaresTracer(), g, i, j, sgn) === :wall
        @test p.Z[end] ≈ Zw atol = 1.0e-12
    end
    @test sum(poloidal_length, paths) ≈ r * (2π - 2acos((Zw - Z0) / r)) rtol = 2.0e-3
end

@testitem "Marching squares: a node at an extremum of the grid flux is a point" setup = [TracingFixtures] tags = [:numerics] begin
    G = test_grid(41)
    i, j = 20, 22
    for s in (1, -1)                             # a minimum and a maximum on the node
        g = FieldLineGeometry(G, @. s * ((G.R2D - G.R1D[i])^2 + (G.Z2D - G.Z1D[j])^2))
        @test trace_from_node!(Path(), MarchingSquaresTracer(), g, i, j, 1) === :degenerate
        @test trace_from_node!(Path(), MarchingSquaresTracer(), g, i, j, -1) === :degenerate
    end
end

@testitem "Marching squares: a saddle cell connects its edges by the centre value" setup = [TracingFixtures] tags = [:numerics] begin
    using RAPID2D: ms_exit_edge
    # corners counter-clockwise from (1, 1): above, below, above, below at level 0
    ψ = [1.0 -1.0; -1.0 1.0]                     # ψ[i, j]: (1,1) = 1, (2,1) = −1, (2,2) = 1, (1,2) = −1
    # centre 0 ≥ 0 is above: the below corners are cut off
    @test ms_exit_edge(ψ, 0.0, 1, 1, 1) == 2     # entering the bottom, corner 2 (below) cut off
    @test ms_exit_edge(ψ, 0.0, 1, 1, 3) == 4     # entering the top, corner 4 (below) cut off
    @test ms_exit_edge(ψ, 0.0, 1, 1, 2) == 1
    # centre below: the above corners are cut off
    ψ = [1.0 -1.2; -1.2 1.0]
    @test ms_exit_edge(ψ, 0.0, 1, 1, 1) == 4     # corner 1 (above) cut off
    @test ms_exit_edge(ψ, 0.0, 1, 1, 2) == 3     # corner 3 (above) cut off

    # A hyperbola beside an X-point at a cell centre: the level ψ = L > 0 of ψ = x² − y² is
    # two branches, and the one through a node on the right stays on the right to the wall.
    G = test_grid(21)                            # square cells
    Rs, Zs = G.R1D[10] + G.dR / 2, G.Z1D[10] + G.dZ / 2
    g = FieldLineGeometry(G, @. (G.R2D - Rs)^2 - (G.Z2D - Zs)^2)
    for i in 11:13, sgn in (1, -1)
        p = Path()
        @test trace_from_node!(p, MarchingSquaresTracer(), g, i, 10, sgn) === :wall
        @test minimum(p.R) > Rs
    end
end

@testitem "Marching squares: the length budget, and no allocation per trace" setup = [TracingFixtures] tags = [:numerics] begin
    G = test_grid(41)
    g = FieldLineGeometry(G, copy(G.R2D))
    i, j = nearest(G, 1.5, 0.0)
    L = PathLengths{Float64}()
    @test trace_from_node!(L, MarchingSquaresTracer(), g, i, j, 1; max_length = 0.1) === :trace_limit
    @test 0.1 < L.Lpol <= 0.1 + G.dZ
    allocations(L, g, i, j) = @allocated trace_from_node!(L, MarchingSquaresTracer(), g, i, j, 1)
    allocations(L, g, i, j)
    @test allocations(PathLengths{Float64}(), g, i, j) == 0
end
