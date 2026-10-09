# Flux-surface averages on the (R, Z) grid.
#
# A connected component of a ψ contour, revolved toroidally, is a flux surface made of many
# field lines: closed around an O-point, or open from wall to wall (not handled yet). Its
# average carries the volume measure dl/B_pol,
#
#     ⟨f⟩ = ∮ f dl/B_pol / ∮ dl/B_pol,        dV/dψ = 2π ∮ dl/B_pol   (ψ per radian),
#
# which on a closed surface is the flux-surface average. The plain arclength average along a
# field line is ⟨B f⟩/⟨B⟩.
# internal/docs/src/notes/plans/PLAN_flux-surface-average.md;
# internal/docs/src/notes/design/closed-surface-quasineutrality.md §9.4.

"""
    psi_interpolant(G, ψ)

Bicubic interpolant of the grid flux `ψ`, with analytic gradient and Hessian. It reproduces
cubic polynomials exactly.
"""
function psi_interpolant(G::GridGeometry, ψ::AbstractMatrix)
    Rg = range(first(G.R1D), last(G.R1D); length = G.NR)
    Zg = range(first(G.Z1D), last(G.Z1D); length = G.NZ)
    return cubic_interp((Rg, Zg), ψ; bc = CubicFit(), extrap = ExtendExtrap())
end

"""
    find_o_point(G, ψ, region; tol = 1e-10, maxit = 50) -> (; R, Z, ψ, converged) or nothing

The O-point of a closed region (`region` holds linear node indices): the extremum of the
bicubic interpolant of `ψ`, found by damped Newton on ∇ψ = 0 from the region's node farthest
in ψ from the region's edge. `converged` requires a definite Hessian (an extremum, not an
X-point) and a step below `tol` grid spacings. `nothing` when `region` is empty.
"""
function find_o_point(
        G::GridGeometry{FT}, ψ::AbstractMatrix{FT}, region::AbstractVector{<:Integer};
        tol::Real = 1.0e-10, maxit::Int = 50,
    ) where {FT <: AbstractFloat}
    isempty(region) && return nothing
    itp = psi_interpolant(G, ψ)
    k0 = o_point_seed(G, ψ, region)
    x = (G.R2D[k0], G.Z2D[k0])
    h = min(G.dR, G.dZ)
    converged = false
    for _ in 1:maxit
        g = gradient(itp, x)
        H = hessian(itp, x)
        det = H[1, 1] * H[2, 2] - H[1, 2] * H[2, 1]
        det == 0 && break
        δ = ((H[2, 2] * g[1] - H[1, 2] * g[2]) / det, (H[1, 1] * g[2] - H[2, 1] * g[1]) / det)
        # Backtrack until |∇ψ| decreases: a full Newton step can overshoot on a coarse grid.
        g2 = g[1]^2 + g[2]^2
        α = one(FT)
        xn = (x[1] - α * δ[1], x[2] - α * δ[2])
        while α > FT(1.0e-3)
            gn = gradient(itp, xn)
            gn[1]^2 + gn[2]^2 < g2 && break
            α /= 2
            xn = (x[1] - α * δ[1], x[2] - α * δ[2])
        end
        step = hypot(xn[1] - x[1], xn[2] - x[2])
        x = xn
        if step < tol * h
            Hn = hessian(itp, x)
            converged = Hn[1, 1] * Hn[2, 2] - Hn[1, 2] * Hn[2, 1] > 0
            break
        end
    end
    inside = first(G.R1D) <= x[1] <= last(G.R1D) && first(G.Z1D) <= x[2] <= last(G.Z1D)
    return (; R = x[1], Z = x[2], ψ = itp(x), converged = converged && inside)
end

# The region's node farthest in ψ from the mean ψ of its edge nodes (those with a neighbour
# outside the region): the node closest to the O-point of a closed region.
function o_point_seed(G::GridGeometry, ψ::AbstractMatrix, region::AbstractVector{<:Integer})
    NR, NZ = G.NR, G.NZ
    in_region = falses(NR * NZ)
    in_region[region] .= true
    edge_sum, edge_count = zero(eltype(ψ)), 0
    for k in region
        i, j = mod1(k, NR), cld(k, NR)
        neighbours = ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
        on_edge = any(n -> !(1 <= n[1] <= NR && 1 <= n[2] <= NZ) || !in_region[(n[2] - 1) * NR + n[1]], neighbours)
        if on_edge
            edge_sum += ψ[k]
            edge_count += 1
        end
    end
    ψ_edge = edge_count > 0 ? edge_sum / edge_count : sum(ψ[region]) / length(region)
    return region[argmax(abs.(ψ[region] .- ψ_edge))]
end

"""
    surface_levels(o, ψ, region; nsurf = default) -> (; ψ_axis, ψ_edge, ψN, ψ)

ψ levels of `nsurf` closed surfaces between the O-point `o` and the region's edge, the region
node farthest in ψ from the axis. Levels sit at the centres of `nsurf` equal steps of the
normalized flux ψN = (ψ − ψ_axis)/(ψ_edge − ψ_axis), so none is the axis or the edge itself.
The default gives about one surface per grid spacing across the region.
"""
function surface_levels(
        o, ψ::AbstractMatrix{FT}, region::AbstractVector{<:Integer};
        nsurf::Int = default_surface_count(region),
    ) where {FT <: AbstractFloat}
    nsurf >= 1 || throw(ArgumentError("nsurf must be positive, got $nsurf"))
    ψ_axis = FT(o.ψ)
    ψ_edge = ψ[region[argmax(abs.(ψ[region] .- ψ_axis))]]
    ψN = (FT.(1:nsurf) .- FT(0.5)) ./ nsurf
    return (; ψ_axis, ψ_edge, ψN, ψ = ψ_axis .+ ψN .* (ψ_edge - ψ_axis))
end

# About one surface per grid spacing across a roughly round region.
default_surface_count(region) = max(2, round(Int, sqrt(length(region) / π)))
