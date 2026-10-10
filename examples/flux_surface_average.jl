# Flux-surface averages on the (R, Z) grid.
#
# A closed ψ contour, revolved toroidally, is a flux surface. Its average carries the volume
# measure dl/B_pol,
#
#     ⟨f⟩ = ∮ f dl/B_pol / ∮ dl/B_pol,        |dV/dψ| = 2π ∮ dl/B_pol   (ψ per radian),
#
# and the safety factor follows as |q| = F ⟨1/R²⟩ |dV/dψ| / (4π²), F = R Bϕ. Three ways to
# build the average on the grid are compared:
#   CubicContourAverage (the default): the level set of the bicubic ψ, traced by predictor–corrector steps;
#   MarchingSquaresAverage: the closed contour of each level by marching squares on the grid ψ;
#   HatBinningAverage: share each node between the two nearest levels, weighted by volume.
#
# Part 1 uses elliptic surfaces whose averages are known by quadrature along the ellipse.
# Part 2 follows the closed region of the full start-up run (single-quadrupole field, every
# module on except Global_JxB_Force) as it forms and grows.
#
#   julia --project=examples examples/flux_surface_average.jl

include("common.jl")
using RAPID2D: initialize_grid_geometry, flux_surface_average, surface_average, MarchingSquaresAverage, HatBinningAverage, CubicContourAverage

name = "flux_surface_average"
out = output_dir(name)
policies = (MarchingSquaresAverage(), CubicContourAverage(), HatBinningAverage())
label(p) = p isa MarchingSquaresAverage ? "marching squares" : p isa CubicContourAverage ? "bicubic tracing" : "hat binning"
linestyle(p) = p isa MarchingSquaresAverage ? :solid : p isa CubicContourAverage ? :dot : :dash

# ── Part 1: elliptic surfaces ψ = (R − R0)² + ((Z − Z0)/κ)², axis off the nodes ──────────────
const R0, Z0, κ = 1.5113, 0.0071, 1.6

function ellipse_grid(N)
    G = initialize_grid_geometry(N, N, (1.0, 2.0), (-0.5, 0.5))
    G.inVol2D .= 2π .* G.R2D .* G.dR .* G.dZ
    ψ = @. (G.R2D - R0)^2 + ((G.Z2D - Z0) / κ)^2
    return G, ψ, findall(<(0.09), vec(ψ))
end

function ellipse_reference(f, a; n = 20_000)
    num = den = 0.0
    for k in 0:(n - 1)
        t = 2π * k / n
        R, Z = R0 + a * cos(t), Z0 + κ * a * sin(t)
        dl = a * hypot(sin(t), κ * cos(t)) * (2π / n)
        Bpol = hypot(2 * (R - R0), 2 * (Z - Z0) / κ^2) / R
        num += f(R, Z) * dl / Bpol
        den += dl / Bpol
    end
    return num / den, 2π * den
end

println("Part 1: elliptic surfaces (κ = $κ), 10 surfaces")
pl_err = plot(xlabel = "ψN", ylabel = "|error| of ⟨1/R²⟩", yscale = :log10, legend = :bottomright, title = "elliptic surfaces")
pl_dV = plot(xlabel = "ψN", ylabel = "|error| of dV/dψ", yscale = :log10, legend = false, title = "elliptic surfaces")
for N in (31, 61, 121), pol in policies
    G, ψ, region = ellipse_grid(N)
    fsa = flux_surface_average(G, ψ, region; policy = pol, nsurf = 10)
    t_build = minimum(@elapsed(flux_surface_average(G, ψ, region; policy = pol, nsurf = 10)) for _ in 1:5)
    avg = surface_average(fsa, 1 ./ G.R2D .^ 2)
    err = [abs(avg[s] / ellipse_reference((R, Z) -> 1 / R^2, sqrt(ψs))[1] - 1) for (s, ψs) in pairs(fsa.ψ)]
    errV = [abs(fsa.dVdψ[s] / ellipse_reference((R, Z) -> 1.0, sqrt(ψs))[2] - 1) for (s, ψs) in pairs(fsa.ψ)]
    @printf("  %-17s %3d² | ⟨1/R²⟩ error max %.1e | dV/dψ error max %.1e | build %.2f ms\n", label(pol), N, maximum(err), maximum(errV), 1.0e3 * t_build)
    style = linestyle(pol)
    plot!(pl_err, fsa.ψN, err; lw = 2, ls = style, marker = :circle, ms = 3, label = "$(label(pol)) $(N)²")
    plot!(pl_dV, fsa.ψN, errV; lw = 2, ls = style, marker = :circle, ms = 3)
end

# ── Part 2: the closed region of the full start-up run ─────────────────────────────────────
config = SimulationConfig{Float64}(
    inputs = InputPaths(field = SINGLE_QUAD, wall = BOX_WALL),
    NR = 30, NZ = 50, R0B0 = 3.0, prefilled_gas_pressure = 2.0e-3,
    dt = 2.0e-6, t_end_s = 20.0e-3, snap0D_Δt_s = 20.0e-6, snap2D_Δt_s = 1.0e-3,
    Output_path = out,
)
RP = setup(
    config;
    src = true, Atomic_Collision = true, Coulomb_Collision = true, Spitzer_Resistivity = true,
    diffu = true, convec = true, ud_evolve = true, Te_evolve = true, Ti_evolve = true,
    update_ni_independently = true, Gas_evolve = true,
    mean_ExB = true, turb_ExB_mixing = true, E_para_self_ES = true,
    Ampere = true, Ampere_Itor_threshold = 0.1, Ampere_nstep = 1, E_para_self_EM = true,
    Global_JxB_Force = false, FLF_nstep = 50,
)

times_ms = (4.0, 6.0, 8.0, 10.0, 14.0, 20.0)
marks = Dict(round(Int, t * 1.0e-3 / config.dt) => t for t in times_ms)
records = Dict{Tuple{Float64, Symbol}, Any}()
run!(
    RP; callback_after_step = rp -> begin
        haskey(marks, rp.step) || return
        isempty(rp.flf.closed_surface_nids) && return
        t = marks[rp.step]
        for pol in policies
            fsa = flux_surface_average(rp; policy = pol)
            t_build = @elapsed flux_surface_average(rp; policy = pol)
            F = rp.config.R0B0
            q = abs.(F .* surface_average(fsa, 1 ./ rp.G.R2D .^ 2) .* fsa.dVdψ ./ (4π^2))
            records[(t, nameof(typeof(pol)))] = (;
                nclosed = length(rp.flf.closed_surface_nids), nvalid = count(fsa.valid), nsurf = length(fsa.ψ),
                ψN = fsa.ψN, Te = surface_average(fsa, rp.plasma.Te_eV, rp.plasma.ne),
                ne = surface_average(fsa, rp.plasma.ne), q, t_build,
            )
        end
    end
)

println("Part 2: closed region of full_startup")
pl_Te = plot(xlabel = "ψN", ylabel = "n-weighted ⟨Te⟩ (eV)", legend = :topright, title = "full_startup")
pl_q = plot(xlabel = "ψN", ylabel = "q", yscale = :log10, legend = false, title = "full_startup")
for t in times_ms
    haskey(records, (t, :MarchingSquaresAverage)) || continue
    for pol in policies
        r = records[(t, nameof(typeof(pol)))]
        @printf(
            "  t %4.1f ms | %-17s | closed nodes %3d | surfaces valid %d of %d | q %s | build %.2f ms\n", t, label(pol), r.nclosed, r.nvalid, r.nsurf,
            join((@sprintf("%.0f", x) for x in r.q), " "), 1.0e3 * r.t_build
        )
        style = linestyle(pol)
        plot!(pl_Te, r.ψN, r.Te; lw = 2, ls = style, marker = :circle, ms = 3, label = "$(t) ms, $(label(pol))")
        plot!(pl_q, r.ψN, r.q; lw = 2, ls = style, marker = :circle, ms = 3)
    end
end
savefig(plot(pl_err, pl_dV, pl_Te, pl_q; layout = (2, 2), size = (1200, 900), margin = 5Plots.mm), joinpath(out, "flux_surface_average.png"))
println("outputs in ", out)
