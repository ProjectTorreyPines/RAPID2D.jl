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
        @test trace_from_node!(p, MarchingSquaresTracer(), g, i, j, sgn).termination === :wall
        @test trace_from_node!(L, MarchingSquaresTracer(), g, i, j, sgn).termination === :wall
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
        @test trace_from_node!(p, MarchingSquaresTracer(), g, i, j, 1).termination === :closed
        @test trace_from_node!(L, MarchingSquaresTracer(), g, i, j, 1).termination === :closed
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
        @test trace_from_node!(p, MarchingSquaresTracer(), g, i, j, sgn).termination === :wall
        @test p.Z[end] ≈ Zw atol = 1.0e-12
    end
    @test sum(poloidal_length, paths) ≈ r * (2π - 2acos((Zw - Z0) / r)) rtol = 2.0e-3
end

@testitem "Marching squares: a node at an extremum of the grid flux is a point" setup = [TracingFixtures] tags = [:numerics] begin
    G = test_grid(41)
    i, j = 20, 22
    for s in (1, -1)                             # a minimum and a maximum on the node
        g = FieldLineGeometry(G, @. s * ((G.R2D - G.R1D[i])^2 + (G.Z2D - G.Z1D[j])^2))
        @test trace_from_node!(Path(), MarchingSquaresTracer(), g, i, j, 1).termination === :degenerate
        @test trace_from_node!(Path(), MarchingSquaresTracer(), g, i, j, -1).termination === :degenerate
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
        @test trace_from_node!(p, MarchingSquaresTracer(), g, i, 10, sgn).termination === :wall
        @test minimum(p.R) > Rs
    end
end

@testitem "Marching squares: the length budget, and no allocation per trace" setup = [TracingFixtures] tags = [:numerics] begin
    G = test_grid(41)
    g = FieldLineGeometry(G, copy(G.R2D))
    i, j = nearest(G, 1.5, 0.0)
    L = PathLengths{Float64}()
    @test trace_from_node!(L, MarchingSquaresTracer(), g, i, j, 1; max_length = 0.1).termination === :trace_limit
    @test 0.1 < L.Lpol <= 0.1 + G.dZ
    allocations(L, g, i, j) = @allocated trace_from_node!(L, MarchingSquaresTracer(), g, i, j, 1)
    allocations(L, g, i, j)
    @test allocations(PathLengths{Float64}(), g, i, j) == 0
end

@testitem "Bicubic tracer: a line to the wall, a circle to its chords, along B_pol" setup = [TracingFixtures] tags = [:numerics] begin
    using RAPID2D: CubicTracer
    # ψ = R is reproduced exactly by the bicubic ψ: the path is the line, cut exactly at the wall.
    G = test_grid(21, 41; Z = (-1.0, 1.0))
    F = 3.0
    inside = [abs(G.Z1D[j] + G.dZ / 2) < 0.6 for _ in 1:G.NR, j in 1:G.NZ]
    g = FieldLineGeometry(G, copy(G.R2D); Bϕ = F ./ G.R2D, inside)
    i, j = nearest(G, 1.5, 0.0)
    for (sgn, Zend) in ((1, 0.6), (-1, -0.6))
        L = PathLengths{Float64}()
        e = trace_from_node!(L, CubicTracer(), g, i, j, sgn)
        @test e.termination === :wall
        @test e.Z ≈ Zend atol = 1.0e-12
        @test L.Lpol ≈ 0.6 atol = 1.0e-12
        @test L.Lc ≈ 0.6 * sqrt(1 + F^2) rtol = 1.0e-10
    end
    # ψ = r² is reproduced exactly too, so a closed path is a polygon inscribed in the circle:
    # its length falls short by about θ²/24 for a turn θ per step (here θ ≈ 2.4°, 7e-5).
    R0, Z0 = 1.5113, 0.0071
    G = test_grid(41)
    g = FieldLineGeometry(G, @. (G.R2D - R0)^2 + (G.Z2D - Z0)^2; Bϕ = F ./ G.R2D)
    i, j = nearest(G, R0 + 0.3, Z0)
    r = hypot(G.R1D[i] - R0, G.Z1D[j] - Z0)
    p, L = Path(), PathLengths{Float64}()
    @test trace_from_node!(p, CubicTracer(), g, i, j, 1).termination === :closed
    @test trace_from_node!(L, CubicTracer(), g, i, j, 1).termination === :closed
    @test (p.R[end], p.Z[end]) == (p.R[1], p.Z[1])
    @test maximum(abs.(hypot.(p.R .- R0, p.Z .- Z0) .- r)) < 1.0e-9     # every vertex on the level
    @test 0 < 1 - L.Lpol / (2π * r) < 1.0e-4
    @test L.Lc / L.Lpol ≈ sqrt(1 + F^2 / (4r^2)) rtol = 1.0e-3
    @test p.Z[2] > p.Z[1]                                              # along B_pol: up on the outer side
end

@testitem "Bicubic tracer: no corner of the wall is stepped over" setup = [TracingFixtures] tags = [:numerics] begin
    using RAPID2D: CubicTracer
    # ψ = R + βZ: straight contours going down to the right against B_pol. The one through node
    # (i, j) passes the corner (i+1, j−3) on its left by δ, so it cuts the corner of cell (i, j−4)
    # for about 3δ, a thousandth of a step. Marking that cell outside must end the trace there.
    G = test_grid(41)
    h = G.dR
    δ = 1.0e-3 * h
    β = (h - δ) / (3 * G.dZ)
    i, j = 20, 25
    inside = trues(G.NR, G.NZ)
    inside[i, j - 4] = false
    g = FieldLineGeometry(G, @. G.R2D + β * G.Z2D; inside)
    e = trace_from_node!(Path(), CubicTracer(), g, i, j, -1)
    @test e.termination === :wall
    @test e.Z ≈ G.Z1D[j - 3] atol = 1.0e-9
    @test e.R ≈ G.R1D[i + 1] - δ atol = 1.0e-9
    # without the cell the line runs on to the edge of the grid
    g = FieldLineGeometry(G, @. G.R2D + β * G.Z2D)
    @test trace_from_node!(Path(), CubicTracer(), g, i, j, -1).Z < G.Z1D[j - 4]
end

@testitem "Bicubic tracer: a null at the node, and the length budget" setup = [TracingFixtures] tags = [:numerics] begin
    using RAPID2D: CubicTracer
    G = test_grid(41)
    i, j = 20, 22
    g = FieldLineGeometry(G, @. (G.R2D - G.R1D[i])^2 + (G.Z2D - G.Z1D[j])^2)
    @test trace_from_node!(Path(), CubicTracer(), g, i, j, 1).termination === :null
    g = FieldLineGeometry(G, copy(G.R2D))
    L = PathLengths{Float64}()
    @test trace_from_node!(L, CubicTracer(), g, 20, 21, 1; max_length = 0.1).termination === :trace_limit
    @test 0.1 < L.Lpol <= 0.1 + G.dZ
end

@testitem "RK4 tracer: the same trace as trace_single_field_line" setup = [TracingFixtures] tags = [:numerics] begin
    using RAPID2D: RK4Tracer, StepLengths, trace_single_field_line, is_in_wall_by_cell_state, default_max_length
    G = test_grid(41)
    R0, Z0 = 1.5113, 0.0071
    ψ = @. (G.R2D - R0)^2 + (G.Z2D - Z0)^2
    inside = [G.R1D[i] + G.dR / 2 < 1.85 for i in 1:G.NR, _ in 1:G.NZ]
    @test_throws ArgumentError trace_from_node!(StepLengths{Float64}(), RK4Tracer(), FieldLineGeometry(G, ψ; inside), 30, 21, 1)
    g = FieldLineGeometry(G, ψ; Bϕ = 3.0 ./ G.R2D, inside, nodal_B = true)
    B = g.nodal_B
    wall = (R, Z) -> is_in_wall_by_cell_state(R, Z, g.R1D[1], g.Z1D[1], G.NR, 1 / g.dR, 1 / g.dZ, inside)
    step = 0.5 * min(g.dR, g.dZ)
    Lmax = default_max_length(g)
    for (i, j) in ((30, 21), (25, 30), (35, 21)), sgn in (1, -1)     # closed, closed, cut by the wall
        L = StepLengths{Float64}()
        e = trace_from_node!(L, RK4Tracer(), g, i, j, sgn)
        ref = trace_single_field_line(g.R1D[i], g.Z1D[j], sgn, B.BR, B.BZ, B.Bϕ, step, floor(Int, Lmax / step), Lmax, wall)
        @test e.termination === ref.termination
        @test (L.Lpol, L.Lc, L.min_Bpol, e.R, e.Z) === (ref.Lpol, ref.Lc, ref.min_Bpol, ref.final_R, ref.final_Z)
    end
end

@testitem "Bicubic tracer: a level is found to the flux's own rounding" tags = [:numerics] begin
    using RAPID2D: project_to_level
    # Near an O-point on a large flux, |∇ψ| is tiny against ψ, and an interpolant evaluates the level
    # only to a few ulps of ψ. A tolerance relative to the step alone then cannot be met: the
    # projection failed, the step halved, and the trace stopped as if at a null. This flux is a plane
    # of slope 1e-5 on ψ ≈ 1, evaluated two ulps off with alternating sign, so it never reads exact.
    mutable struct RoundedPlane
        sign::Int
    end
    (p::RoundedPlane)(x) = (p.sign = -p.sign; 1.0 + 1.0e-5 * (x[1] - 1.5) + 2 * p.sign * eps(1.0))
    RAPID2D.FastInterpolations.gradient(::RoundedPlane, x) = (1.0e-5, 0.0)
    level = 1.0 + 1.0e-5 * 0.1
    x = project_to_level(RoundedPlane(1), level, (1.601, 0.0), 0.01)
    @test x !== nothing
    @test abs(x[1] - 1.6) < 1.0e-9
end

@testitem "Field lines: a contour that closes one way is closed" setup = [TracingFixtures] tags = [:numerics] begin
    using RAPID2D: FieldLineTracer, CubicTracer, FieldLineFollowingResult, field_line_analysis!
    import RAPID2D: trace_from_node!, step_budget
    # A tracer that fails forward (as a stalled projection did) but follows the contour backward
    # back to its node: returning to the start proves the line closed, so both directions publish
    # the circuit.
    struct ForwardFails <: FieldLineTracer end
    trace_from_node!(v, ::ForwardFails, g, i, j, sgn; max_length = Inf) =
        sgn > 0 ? (; termination = :null, R = g.R1D[i], Z = g.Z1D[j]) : trace_from_node!(v, CubicTracer(), g, i, j, sgn; max_length)
    step_budget(::ForwardFails, g, max_length) = 1
    G = test_grid(21)
    g = FieldLineGeometry(G, @. (G.R2D - 1.5113)^2 + (G.Z2D - 0.0071)^2)
    flf = field_line_analysis!(FieldLineFollowingResult{Float64}(G.NR, G.NZ), g, ForwardFails())
    k = LinearIndices(g.ψ)[14, 11]
    @test flf.termination_forward[k] === flf.termination_backward[k] === :closed
    @test isfinite(flf.Lpol_tot[k]) && flf.Lpol_forward[k] == flf.Lpol_backward[k] > 0
    @test k in flf.closed_surface_nids
end
