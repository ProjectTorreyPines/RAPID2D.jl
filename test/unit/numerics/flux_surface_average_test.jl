# Flux-surface averages on the (R, Z) grid.
#
# A connected component of a ψ contour, revolved toroidally, is a flux surface. Its average
# carries the volume measure dl/B_pol:
#
#     ⟨f⟩ = ∮ f dl/B_pol / ∮ dl/B_pol,        dV/dψ = 2π ∮ dl/B_pol   (ψ per radian).
#
# The fixtures are elliptic surfaces ψ = (R − R0)² + ((Z − Z0)/κ)² about an axis that sits off
# the grid nodes, whose averages are known by direct quadrature along the ellipse.
# internal/docs/src/notes/plans/PLAN_flux-surface-average.md.

@testsnippet FluxSurfaceFixtures begin
    using RAPID2D: initialize_grid_geometry

    const R0, Z0 = 1.5113, 0.0071

    "A uniform N × N grid on R ∈ [1, 2], Z ∈ [−0.5, 0.5], with toroidal cell volumes."
    function test_grid(N)
        G = initialize_grid_geometry(N, N, (1.0, 2.0), (-0.5, 0.5))
        G.inVol2D .= 2π .* G.R2D .* G.dR .* G.dZ
        return G
    end

    "ψ = (R − R0)² + ((Z − Z0)/κ)²: elliptic surfaces of elongation κ, circles at κ = 1."
    ellipse_ψ(G; κ = 1.0) = @. (G.R2D - R0)^2 + ((G.Z2D - Z0) / κ)^2

    "The nodes inside the surface of minor radius a."
    inside(ψ; a = 0.3) = findall(<(a^2), vec(ψ))

    """
    ⟨f⟩ and dV/dψ on the surface ψ = a², by quadrature along R = R0 + a cos t,
    Z = Z0 + κ a sin t.
    """
    function reference_average(f, a; κ = 1.0, n = 20_000)
        num = den = 0.0
        for k in 0:(n - 1)
            t = 2π * k / n
            R = R0 + a * cos(t)
            Z = Z0 + κ * a * sin(t)
            dl = a * hypot(sin(t), κ * cos(t)) * (2π / n)
            Bpol = hypot(2 * (R - R0), 2 * (Z - Z0) / κ^2) / R
            num += f(R, Z) * dl / Bpol
            den += dl / Bpol
        end
        return num / den, 2π * den
    end
end

@testitem "Flux surfaces: the O-point off the grid nodes" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: find_o_point
    for N in (31, 61), κ in (1.0, 1.6), s in (1, -1)
        G = test_grid(N)
        ψ = s .* ellipse_ψ(G; κ)
        o = find_o_point(G, ψ, inside(s .* ψ))
        @test o.converged
        @test abs(o.R - R0) < 1.0e-6 * G.dR
        @test abs(o.Z - Z0) < 1.0e-6 * G.dZ
        @test abs(o.ψ) < 1.0e-10
    end
end

@testitem "Flux surfaces: levels from the axis to the region's edge" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: find_o_point, surface_levels
    G = test_grid(61)
    ψ = ellipse_ψ(G)
    region = inside(ψ)
    o = find_o_point(G, ψ, region)
    lv = surface_levels(o, ψ, region; nsurf = 8)
    @test length(lv.ψ) == 8
    @test lv.ψ_axis == o.ψ
    @test lv.ψ_edge ≈ maximum(ψ[region])
    @test issorted(lv.ψN) && 0 < first(lv.ψN) && last(lv.ψN) < 1
    @test lv.ψ ≈ lv.ψ_axis .+ lv.ψN .* (lv.ψ_edge - lv.ψ_axis)
    # one surface per grid spacing across the region by default
    lv_default = surface_levels(o, ψ, region)
    @test 0.5 * 0.3 / G.dR <= length(lv_default.ψ) <= 1.5 * 0.3 / G.dR
end

@testitem "Flux surfaces: no closed region" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: find_o_point
    G = test_grid(31)
    @test find_o_point(G, ellipse_ψ(G), Int[]) === nothing
end

@testitem "Flux surfaces: hat binning keeps constants and is close to the ellipse" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: flux_surface_average, surface_average, HatBinningAverage
    G = test_grid(61)
    for κ in (1.0, 1.6)
        ψ = ellipse_ψ(G; κ)
        fsa = flux_surface_average(G, ψ, inside(ψ); policy = HatBinningAverage(), nsurf = 10)
        @test fsa.policy isa HatBinningAverage
        @test all(fsa.valid)
        @test maximum(abs, surface_average(fsa, ones(G.NR, G.NZ)) .- 1) < 1.0e-13
        # ⟨R⟩ − R0 = O(a²/R0) is the part that tests the weighting; compare that part.
        # Binning averages f over a band one level spacing wide, so it keeps an O(Δψ²) bias
        # that does not fall with the grid (measured at 61²: 0.14, 4.9e-3, 0.05 at most).
        meanR = surface_average(fsa, G.R2D)
        invR2 = surface_average(fsa, 1 ./ G.R2D .^ 2)
        for (s, ψs) in pairs(fsa.ψ)
            a = sqrt(ψs)
            refR, refV = reference_average((R, Z) -> R, a; κ)
            refI, _ = reference_average((R, Z) -> 1 / R^2, a; κ)
            @test abs((meanR[s] - R0) / (refR - R0) - 1) < 0.2
            @test abs(invR2[s] / refI - 1) < 7.0e-3
            @test abs(fsa.dVdψ[s] / refV - 1) < 0.08
        end
    end
end

@testitem "Flux surfaces: weighted averages, the way back to the grid, and the region" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: flux_surface_average, surface_average, to_grid, HatBinningAverage
    G = test_grid(61)
    ψ = ellipse_ψ(G; κ = 1.3)
    region = inside(ψ)
    fsa = flux_surface_average(G, ψ, region; policy = HatBinningAverage())
    f = @. G.R2D^2 * (1 + G.Z2D)
    ω = @. 1 + G.R2D
    @test surface_average(fsa, f, ω) ≈ surface_average(fsa, ω .* f) ./ surface_average(fsa, ω)
    # a grid array and its vector give the same averages
    @test surface_average(fsa, f) == surface_average(fsa, vec(f))
    # a flux function comes back to itself between the first and the last surface
    g = to_grid(fsa, surface_average(fsa, ψ))
    @test size(g) == size(ψ)
    between = [k for k in region if first(fsa.ψ) <= ψ[k] <= last(fsa.ψ)]
    @test maximum(abs.(g[between] .- ψ[between])) < 0.05 * maximum(ψ[region])
    # nodes outside the region get nothing
    outside = setdiff(1:(G.NR * G.NZ), region)
    @test all(iszero, g[outside])
    # an empty region gives no averager
    @test flux_surface_average(G, ψ, Int[]; policy = HatBinningAverage()) === nothing
end

@testitem "Flux surfaces: marching squares matches the ellipse" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: flux_surface_average, surface_average, MarchingSquaresAverage, CubicContourAverage
    G = test_grid(61)
    @test flux_surface_average(G, ellipse_ψ(G), inside(ellipse_ψ(G))).policy isa CubicContourAverage   # the default
    for κ in (1.0, 1.6)
        ψ = ellipse_ψ(G; κ)
        fsa = flux_surface_average(G, ψ, inside(ψ); policy = MarchingSquaresAverage(), nsurf = 10)
        @test fsa.policy isa MarchingSquaresAverage
        @test all(fsa.valid)
        @test maximum(abs, surface_average(fsa, ones(G.NR, G.NZ)) .- 1) < 1.0e-13
        # measured at 61²: 0.024, 4.7e-5, 2.3e-3 at most (the smallest surface is the worst)
        meanR = surface_average(fsa, G.R2D)
        invR2 = surface_average(fsa, 1 ./ G.R2D .^ 2)
        for (s, ψs) in pairs(fsa.ψ)
            a = sqrt(ψs)
            refR, refV = reference_average((R, Z) -> R, a; κ)
            refI, _ = reference_average((R, Z) -> 1 / R^2, a; κ)
            @test abs((meanR[s] - R0) / (refR - R0) - 1) < 0.04
            @test abs(invR2[s] / refI - 1) < 1.0e-4
            @test abs(fsa.dVdψ[s] / refV - 1) < 5.0e-3
        end
    end
end

@testitem "Flux surfaces: marching squares converges with the grid, binning does not" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: flux_surface_average, surface_average, HatBinningAverage, MarchingSquaresAverage
    κ = 1.6
    err(policy, N) = begin
        G = test_grid(N)
        ψ = ellipse_ψ(G; κ)
        fsa = flux_surface_average(G, ψ, inside(ψ); policy, nsurf = 6)
        avg = surface_average(fsa, 1 ./ G.R2D .^ 2)
        maximum(abs(avg[s] / reference_average((R, Z) -> 1 / R^2, sqrt(ψs); κ)[1] - 1) for (s, ψs) in pairs(fsa.ψ))
    end
    e31, e61, e121 = (err(MarchingSquaresAverage(), N) for N in (31, 61, 121))
    @test e61 < e31 / 2.5 && e121 < e61 / 2.5        # about second order
    b61, b121 = (err(HatBinningAverage(), N) for N in (61, 121))
    @test e121 < b121 / 10
end

@testitem "Flux surfaces: a level too close to the axis is skipped, not an error" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: flux_surface_average, surface_average, to_grid, MarchingSquaresAverage
    G = test_grid(31)
    ψ = ellipse_ψ(G)
    region = inside(ψ)
    # The nearest node is 0.013 m from the axis, so only levels with ψN below about 0.002 lie
    # inside the axis cell, where marching squares sees no contour: 400 levels put one there.
    fsa = flux_surface_average(G, ψ, region; policy = MarchingSquaresAverage(), nsurf = 400)
    @test !all(fsa.valid) && any(fsa.valid)
    avg = surface_average(fsa, G.R2D)
    @test all(isnan, avg[.!fsa.valid]) && all(isfinite, avg[fsa.valid])
    @test all(isfinite, to_grid(fsa, avg)[region])
end

@testitem "Flux surfaces: back to the grid is linear in ψN beyond the end levels too" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: flux_surface_average, surface_average, to_grid
    G = test_grid(61)
    ψ = ellipse_ψ(G)
    region = inside(ψ)
    fsa = flux_surface_average(G, ψ, region)
    # A field regular at the axis is a function of r² ∝ ψN, so near the axis it is linear in ψN.
    # Linear in ψN is the case the way back must keep everywhere, the ends included.
    g = ψ ./ 0.09
    back = to_grid(fsa, surface_average(fsa, g))
    ψN = g[region]
    ends = (ψN .< first(fsa.ψN)) .| (ψN .> last(fsa.ψN))
    @test any(ends)
    @test maximum(abs.(back[region][ends] .- ψN[ends])) < 2.0e-3
    @test maximum(abs.(back[region] .- ψN)) < 2.0e-3
end

@testitem "Flux surfaces: beside an X-point, the surfaces stay around their own O-point" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: flux_surface_average, surface_average, MarchingSquaresAverage, HatBinningAverage
    # A doublet: two O-points at Z = ±Zc and an X-point between them at Z = 0.
    G = test_grid(61)
    Zc, w = 0.2, 0.15
    ψ = @. -exp(-((G.R2D - R0)^2 + (G.Z2D - Zc)^2) / w^2) - exp(-((G.R2D - R0)^2 + (G.Z2D + Zc)^2) / w^2)
    ψX = -2 * exp(-Zc^2 / w^2)
    upper = [k for k in eachindex(ψ) if ψ[k] < ψX && G.Z2D[k] > 0]
    for policy in (MarchingSquaresAverage(), HatBinningAverage())
        fsa = flux_surface_average(G, ψ, upper; policy)
        @test fsa.axis.converged
        # a Gaussian is not cubic, so the axis carries interpolation error (5e-4 dR at 61²)
        @test abs(fsa.axis.R - R0) < 0.01 * G.dR && abs(fsa.axis.Z - Zc) < 0.01
        @test all(fsa.valid)
        @test all(>(0), surface_average(fsa, G.Z2D))       # never across the X-point
        @test issorted(fsa.dVdψ)                             # dV/dψ grows toward the separatrix
    end
end

@testitem "Flux surfaces: a region without an O-point is reported, not averaged" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: flux_surface_average, surface_average, to_grid
    G = test_grid(31)
    ψ = copy(G.R2D)                     # no extremum anywhere: open, parallel contours
    region = findall(k -> abs(G.R2D[k] - 1.5) < 0.2 && abs(G.Z2D[k]) < 0.2, eachindex(ψ))
    fsa = flux_surface_average(G, ψ, region)
    @test !fsa.axis.converged
    @test !any(fsa.valid)
    @test all(isnan, surface_average(fsa, G.R2D))
    @test all(iszero, to_grid(fsa, surface_average(fsa, G.R2D)))
end

@testitem "Flux surfaces: contour tracing on the bicubic flux keeps small coarse surfaces accurate" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: initialize_grid_geometry, flux_surface_average, surface_average, CubicContourAverage, MarchingSquaresAverage
    # The start-up grid: 30 × 50 over R 1–2 m, Z ±1 m (dR 3.4 cm, dZ 4.1 cm). A region 2–5 cells
    # across: marching squares sees a polygon of a handful of vertices, so its dV/dψ is off by
    # 4–11 %; tracing the bicubic level set is not limited by the cells.
    G = initialize_grid_geometry(30, 50, (1.0, 2.0), (-1.0, 1.0))
    G.inVol2D .= 2π .* G.R2D .* G.dR .* G.dZ
    κ = 1.4
    ψ = ellipse_ψ(G; κ)
    for a in (0.08, 0.15)
        region = inside(ψ; a)
        fsa = flux_surface_average(G, ψ, region; policy = CubicContourAverage(), nsurf = 6)
        @test fsa.policy isa CubicContourAverage
        @test all(fsa.valid)
        @test maximum(abs, surface_average(fsa, ones(G.NR, G.NZ)) .- 1) < 1.0e-13
        invR2 = surface_average(fsa, 1 ./ G.R2D .^ 2)
        for (s, ψs) in pairs(fsa.ψ)
            refI, refV = reference_average((R, Z) -> 1 / R^2, sqrt(ψs); κ)
            @test abs(fsa.dVdψ[s] / refV - 1) < 1.0e-3
            @test abs(invR2[s] / refI - 1) < 5.0e-3      # f itself is still bilinear between nodes
        end
        # and better than marching squares where marching squares is weakest
        ms = flux_surface_average(G, ψ, region; policy = MarchingSquaresAverage(), nsurf = 6)
        refV1 = reference_average((R, Z) -> 1.0, sqrt(ms.ψ[1]); κ)[2]
        @test abs(fsa.dVdψ[1] / refV1 - 1) < abs(ms.dVdψ[1] / refV1 - 1) / 10
    end
end

@testitem "Flux surfaces: contour tracing beside an X-point and without an O-point" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: flux_surface_average, surface_average, CubicContourAverage
    G = test_grid(61)
    Zc, w = 0.2, 0.15
    ψ = @. -exp(-((G.R2D - R0)^2 + (G.Z2D - Zc)^2) / w^2) - exp(-((G.R2D - R0)^2 + (G.Z2D + Zc)^2) / w^2)
    ψX = -2 * exp(-Zc^2 / w^2)
    upper = [k for k in eachindex(ψ) if ψ[k] < ψX && G.Z2D[k] > 0]
    fsa = flux_surface_average(G, ψ, upper; policy = CubicContourAverage())
    @test all(fsa.valid)
    @test all(>(0), surface_average(fsa, G.Z2D))
    @test issorted(fsa.dVdψ)
    # no extremum: nothing to trace
    flat = flux_surface_average(G, copy(G.R2D), upper; policy = CubicContourAverage())
    @test !flat.axis.converged && !any(flat.valid)
end

@testitem "Flux surfaces: tracing stays accurate up to a separatrix (exact cubic saddle)" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: find_o_point, psi_interpolant, surface_weights, CubicContourAverage
    # ψ = x² + y² − y³, x = (R − 1.5)/0.4, y = Z/0.4: an O-point at the origin and an X-point at
    # y = 2/3, ψX = 4/27. Exactly cubic, so the bicubic interpolant is exact. Inside the level c,
    # c − y² + y³ = (y − y₁)(y₂ − y)(y₃ − y), and dV/dψ = 0.48π ∫ dy/√(…) = 0.48π · 2K(m)/√(y₃ − y₁),
    # m = (y₂ − y₁)/(y₃ − y₁): an elliptic integral, exact up to the separatrix (m → 1).
    G = test_grid(31)
    x, y = (G.R2D .- 1.5) ./ 0.4, G.Z2D ./ 0.4
    ψ = @. x^2 + y^2 - y^3
    ψX = 4 / 27
    region = findall(k -> ψ[k] < ψX && y[k] < 2 / 3, eachindex(ψ))
    function root(f, a, b)
        for _ in 1:200
            c = (a + b) / 2
            sign(f(c)) == sign(f(a)) ? (a = c) : (b = c)
        end
        return (a + b) / 2
    end
    agm(a, b) = (
        for _ in 1:60
            a, b = (a + b) / 2, sqrt(a * b)
        end; a
    )
    function exact_dVdψ(c)
        p(t) = t^3 - t^2 + c
        y1, y2, y3 = root(p, -1.0, 0.0), root(p, 0.0, 2 / 3), root(p, 2 / 3, 1.5)
        m = (y2 - y1) / (y3 - y1)
        return 0.48π * 2 * (π / (2 * agm(1.0, sqrt(1 - m)))) / sqrt(y3 - y1)
    end
    itp = psi_interpolant(G, ψ)
    o = find_o_point(G, ψ, region; itp)
    fractions = [0.5, 0.9, 0.99, 0.999, 0.9999]
    levels = fractions .* ψX
    lv = (; ψ_axis = o.ψ, ψ_edge = ψX, ψN = fractions, ψ = levels)
    _, dVdψ, valid = surface_weights(CubicContourAverage(), G, ψ, region, o, lv, itp)
    @test all(valid)
    for (s, c) in pairs(levels)
        @test abs(dVdψ[s] / exact_dVdψ(c) - 1) < 2.0e-3
    end
end

@testitem "Flux surfaces: edge cases of the contour start and of the O-point" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: initialize_grid_geometry, initialize_grid_geometry!, GridGeometry, closed_contour!, find_o_point,
        flux_surface_average, CubicContourAverage, IMASutils
    # A level equal to the last radial node on the axis row: marching squares must not start
    # its trace in the cell past the grid.
    G5 = initialize_grid_geometry(5, 5, (1.0, 2.0), (-0.5, 0.5))
    ψ5 = @. (G5.R2D - 1.5)^2 + G5.Z2D^2
    Rc, Zc = IMASutils.contour_cache(G5.R1D, G5.Z1D)
    @test closed_contour!(Rc, Zc, ψ5, G5, 0.25, (; R = 1.5, Z = 0.0, ψ = 0.0)) == (nothing, nothing)
    # A region whose outermost surfaces reach within a quarter cell of the grid edge.
    G = test_grid(31)
    ψ = @. (G.R2D - R0)^2 + G.Z2D^2
    fsa = flux_surface_average(G, ψ, findall(<(0.488^2), vec(ψ)); nsurf = 200)
    @test all(fsa.valid)
    # A region that does not contain the extremum has no O-point of its own.
    ψc = @. (G.R2D - 1.5)^2 + G.Z2D^2
    away = findall(k -> 1.6 < G.R2D[k] < 1.7 && abs(G.Z2D[k]) < 0.1, eachindex(ψc))
    @test !find_o_point(G, ψc, away).converged
    # Float32 grids, with default and with Float32 tracing parameters.
    G32 = initialize_grid_geometry!(GridGeometry{Float32}(31, 31), (1.0f0, 2.0f0), (-0.5f0, 0.5f0))
    G32.inVol2D .= 2.0f0 * Float32(π) .* G32.R2D .* G32.dR .* G32.dZ
    ψ32 = @. (G32.R2D - Float32(R0))^2 + (G32.Z2D - Float32(Z0))^2
    region32 = findall(<(0.09f0), vec(ψ32))
    for policy in (CubicContourAverage(), CubicContourAverage(max_turn = Float32(deg2rad(5)), max_step_cells = 0.5f0))
        @test all(flux_surface_average(G32, ψ32, region32; policy).valid)
    end
end
