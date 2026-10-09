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

@testitem "Flux surfaces: marching squares is the default and matches the ellipse" setup = [FluxSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: flux_surface_average, surface_average, MarchingSquaresAverage
    G = test_grid(61)
    for κ in (1.0, 1.6)
        ψ = ellipse_ψ(G; κ)
        fsa = flux_surface_average(G, ψ, inside(ψ); nsurf = 10)
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
    using RAPID2D: flux_surface_average, surface_average, to_grid
    G = test_grid(31)
    ψ = ellipse_ψ(G)
    region = inside(ψ)
    # The nearest node is 0.013 m from the axis, so only levels with ψN below about 0.002 lie
    # inside the axis cell, where marching squares sees no contour: 400 levels put one there.
    fsa = flux_surface_average(G, ψ, region; nsurf = 400)
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
