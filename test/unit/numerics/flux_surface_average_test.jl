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
