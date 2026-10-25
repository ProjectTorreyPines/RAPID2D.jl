# Flux-surface averages on the (R, Z) grid.
#
# A closed ψ contour, revolved toroidally, is a flux surface. Its average carries the volume
# measure dl/B_pol,
#
#     ⟨f⟩ = ∮ f dl/B_pol / ∮ dl/B_pol,        |dV/dψ| = 2π ∮ dl/B_pol   (ψ per radian),
#
# and the safety factor follows as |q| = F ⟨1/R²⟩ |dV/dψ| / (4π²), F = R Bϕ. Three ways to build
# the average on the grid are compared:
#   ContourAverage(CubicTracer()) (the default): each level followed on the bicubic ψ;
#   ContourAverage(MarchingSquaresTracer()): each level's contour by marching squares on the grid ψ;
#   HatBinningAverage(): each node shared between the two nearest levels, weighted by volume.
#
# Part 1 is the analytic start-up field of examples/field_line_tracing.jl, a quadrupole null plus a
# current (an O-point between two X-points), on three grids, averaging over the closed nodes the
# field-line analysis finds. It draws
#   - the surfaces each contour policy integrates over, against the exact surfaces;
#   - dV/dψ, q and ⟨1/R²⟩ against the normalized flux ψN (1 at the separatrix), against a fine-grid
#     reference, with the average along the field line through each closed node.
# Part 2 follows the closed region of the full start-up run (single-quadrupole field, every module
# on except Global_JxB_Force) as it forms and grows: the n-weighted ⟨Te⟩ and q against ψN, and the
# surfaces at the last time.
#
#   julia --project=examples examples/flux_surface_average.jl

include("common.jl")
using RAPID2D: initialize_grid_geometry, FieldLineGeometry, FieldLineFollowingResult, field_line_analysis!,
    flux_surface_average, surface_average, field_line_average, line_average, trace_around!,
    ContourAverage, CubicTracer, MarchingSquaresTracer, HatBinningAverage

name = "flux_surface_average"
out = output_dir(name)

policies = (ContourAverage(MarchingSquaresTracer()), ContourAverage(CubicTracer()), HatBinningAverage())
label(p) = p isa HatBinningAverage ? "hat binning" : p.tracer isa MarchingSquaresTracer ? "marching squares" : "bicubic tracing"
color(p) = p isa HatBinningAverage ? :purple : p.tracer isa MarchingSquaresTracer ? :royalblue : :darkorange
style(p) = p isa HatBinningAverage ? :dashdot : p.tracer isa MarchingSquaresTracer ? :solid : :dash

"Every vertex a tracer visits."
struct Path
    R::Vector{Float64}
    Z::Vector{Float64}
end
Path() = Path(Float64[], Float64[])
RAPID2D.visit!(p::Path, R, Z, Bpol, Bϕ) = (push!(p.R, R); push!(p.Z, Z); p)
surface(tracer, g, level, o) = (p = Path(); trace_around!(p, tracer, g, level, o); p)

# ── Part 1: the analytic start-up field ────────────────────────────────────────────────────────
const F = 3.0                                   # R Bϕ
const RC, ZC, W = 1.5113, 0.0071, 0.15           # the current's centre (off the nodes) and width
ψe(R, Z) = 0.1 * ((R - 1.5)^2 - Z^2) - 0.02 * exp(-((R - RC)^2 + (Z - ZC)^2) / W^2)
∇ψe(R, Z) = (
    0.2 * (R - 1.5) + 0.04 * exp(-((R - RC)^2 + (Z - ZC)^2) / W^2) * (R - RC) / W^2,
    -0.2 * Z + 0.04 * exp(-((R - RC)^2 + (Z - ZC)^2) / W^2) * (Z - ZC) / W^2,
)
function stationary(x)
    h = 1.0e-6
    for _ in 1:50
        g = ∇ψe(x...)
        gR = (∇ψe(x[1] + h, x[2]) .- ∇ψe(x[1] - h, x[2])) ./ 2h
        gZ = (∇ψe(x[1], x[2] + h) .- ∇ψe(x[1], x[2] - h)) ./ 2h
        δ = [gR[1] gZ[1]; gR[2] gZ[2]] \ collect(g)
        x = (x[1] - δ[1], x[2] - δ[2])
    end
    return x
end
O, Xu, Xl = stationary((1.51, 0.0)), stationary((1.5, 0.2)), stationary((1.5, -0.2))
ψO = ψe(O...)
ψX = min(ψe(Xu...), ψe(Xl...))
ψN(ψ) = (ψ - ψO) / (ψX - ψO)

function field(NR, NZ)
    G = initialize_grid_geometry(NR, NZ, (1.0, 2.0), (-1.0, 1.0))
    G.inVol2D .= 2π .* G.R2D .* G.dR .* G.dZ
    ψ = ψe.(G.R2D, G.Z2D)
    return G, ψ, FieldLineGeometry(G, ψ)
end
profiles(fsa, G) = (
    ψN = ψN.(fsa.ψ), dV = fsa.dVdψ, invR2 = surface_average(fsa, 1 ./ G.R2D .^ 2),
    q = F .* surface_average(fsa, 1 ./ G.R2D .^ 2) .* fsa.dVdψ ./ (4π^2),
)

# the reference: bicubic tracing on a fine grid, over the region inside the separatrix
Gf, ψf, gf = field(201, 401)
core = findall(k -> ψf[k] < ψX && Xl[2] < Gf.Z2D[k] < Xu[2], eachindex(ψf))
ref = profiles(flux_surface_average(Gf, ψf, core; nsurf = 60), Gf)
o_exact = (; R = O[1], Z = O[2], ψ = ψO)

Rf, Zf = range(1.0, 2.0; length = 301), range(-1.0, 1.0; length = 601)
ψgrid = [ψe(R, Z) for Z in Zf, R in Rf]
grids = ((16, 26), (30, 50), (60, 100))
println("Part 1: the analytic start-up field, averages over the closed nodes")
shapes, rows = Any[], [Any[], Any[], Any[]]
for (NR, NZ) in grids
    G, ψ, g = field(NR, NZ)
    region = field_line_analysis!(FieldLineFollowingResult{Float64}(NR, NZ), g, MarchingSquaresTracer()).closed_surface_nids
    built = [(pol, flux_surface_average(G, ψ, region; policy = pol)) for pol in policies]
    for (pol, fsa) in built
        t = minimum(@elapsed(flux_surface_average(G, ψ, region; policy = pol)) for _ in 1:5)
        r = profiles(fsa, G)
        @printf(
            "  %3d × %-3d %-17s | %3d closed nodes, %2d of %2d surfaces valid, ψN %.2f–%.2f | build %.3f ms\n",
            NR, NZ, label(pol), length(region), count(fsa.valid), length(fsa.valid), extrema(r.ψN)..., 1.0e3 * t
        )
    end

    # the surfaces each contour policy integrates over, against the exact ones
    p = plot(;
        aspect_ratio = :equal, xlims = (1.25, 1.8), ylims = (-0.3, 0.3), xlabel = "R (m)", ylabel = "Z (m)",
        title = "$NR × $NZ: $(length(region)) closed nodes", legend = NR == first(grids)[1] ? :bottomleft : false
    )
    contour!(p, Rf, Zf, ψgrid; levels = [ψX], color = :black, lw = 1.5, colorbar = false)
    scatter!(p, G.R2D[region], G.Z2D[region]; ms = 3, color = :gray60, markerstrokewidth = 0, label = "closed nodes")
    fsa = built[2][2]
    for (s, level) in pairs(fsa.ψ)
        ex = surface(CubicTracer(), gf, level, o_exact)
        plot!(p, ex.R, ex.Z; color = :black, lw = 3, alpha = 0.25, label = s == 1 ? "exact" : "")
        for (pol, f) in built[1:2]
            l = surface(pol.tracer, g, level, f.axis)
            plot!(p, l.R, l.Z; color = color(pol), ls = style(pol), lw = 1.5, label = s == 1 ? label(pol) : "")
        end
    end
    push!(shapes, p)

    # profiles against ψN, with the field-line averages through the closed nodes
    fla = field_line_average(g, region; tracer = CubicTracer())
    for (row, (key, ylabel)) in enumerate(((:dV, "|dV/dψ| (m³ per Wb/rad)"), (:q, "q"), (:invR2, "⟨1/R²⟩ (m⁻²)")))
        p = plot(; xlabel = "ψN", ylabel, xlims = (0, 1), title = "$NR × $NZ", legend = (row == 1 && NR == first(grids)[1]) ? :topleft : false)
        plot!(p, ref.ψN, getfield(ref, key); color = :black, lw = 3, alpha = 0.4, label = "fine-grid reference")
        if key === :invR2
            scatter!(p, ψN.(ψ[region]), line_average(fla, 1 ./ G.R2D .^ 2); ms = 2.5, color = :gray40, markerstrokewidth = 0, label = "field-line average")
        end
        for (pol, fsa) in built
            r = profiles(fsa, G)
            plot!(
                p, r.ψN[fsa.valid], getfield(r, key)[fsa.valid]; color = color(pol), ls = style(pol), lw = 1.5,
                marker = :circle, ms = 3.5, markercolor = color(pol), markerstrokewidth = 0, label = label(pol)
            )
        end
        push!(rows[row], p)
    end
end
savefig(plot(shapes...; layout = (1, 3), size = (1500, 600), margin = 4Plots.mm), joinpath(out, "surfaces.png"))
savefig(plot(vcat(rows...)...; layout = (3, 3), size = (1500, 1200), margin = 5Plots.mm), joinpath(out, "profiles.png"))

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
    mean_ExB = true, turb_ExB_mixing = true, charge_sheet_field = true, E_para_cancellation = true,
    Ampere = true, Ampere_Itor_threshold = 0.1, Ampere_nstep = 1, E_para_induction = true,
    Global_JxB_Force = false, FLF_nstep = 50,
)

times_ms = (4.0, 6.0, 8.0, 10.0, 14.0, 20.0)
marks = Dict(round(Int, t * 1.0e-3 / config.dt) => t for t in times_ms)
records = Dict{Tuple{Float64, String}, Any}()
last_field = Ref{Any}(nothing)
run!(
    RP; callback_after_step = rp -> begin
        haskey(marks, rp.step) || return
        isempty(rp.flf.closed_surface_nids) && return
        t = marks[rp.step]
        for pol in policies
            fsa = flux_surface_average(rp; policy = pol)
            t_build = @elapsed flux_surface_average(rp; policy = pol)
            q = abs.(rp.config.R0B0 .* surface_average(fsa, 1 ./ rp.G.R2D .^ 2) .* fsa.dVdψ ./ (4π^2))
            records[(t, label(pol))] = (;
                nclosed = length(rp.flf.closed_surface_nids), nvalid = count(fsa.valid), nsurf = length(fsa.ψ),
                ψN = fsa.ψN, Te = surface_average(fsa, rp.plasma.Te_eV, rp.plasma.ne), q, t_build, fsa,
            )
        end
        last_field[] = (; t, G = rp.G, g = FieldLineGeometry(rp), closed = copy(rp.flf.closed_surface_nids), Te = copy(rp.plasma.Te_eV))
    end
)

println("Part 2: closed region of full_startup")
# times by colour, policies by line style
tcolor(t) = cgrad(:viridis)[(t - first(times_ms)) / (last(times_ms) - first(times_ms))]
pl_Te = plot(xlabel = "ψN", ylabel = "n-weighted ⟨Te⟩ (eV)", legend = :outertopright, title = "full_startup: ⟨Te⟩ as the region grows")
pl_q = plot(xlabel = "ψN", ylabel = "q", legend = false, title = "full_startup: q")
for pol in policies
    plot!(pl_Te, [NaN], [NaN]; color = :black, ls = style(pol), lw = 2, label = label(pol))
end
for t in times_ms
    haskey(records, (t, "marching squares")) || continue
    for pol in policies
        r = records[(t, label(pol))]
        @printf(
            "  t %4.1f ms | %-17s | closed nodes %3d | surfaces valid %d of %d | q %s | build %.2f ms\n", t, label(pol), r.nclosed, r.nvalid, r.nsurf,
            join((@sprintf("%.0f", x) for x in r.q), " "), 1.0e3 * r.t_build
        )
        plot!(pl_Te, r.ψN, r.Te; lw = 2, ls = style(pol), color = tcolor(t), marker = :circle, ms = 3, markercolor = tcolor(t), markerstrokewidth = 0, label = pol isa HatBinningAverage ? "$(t) ms" : "")
        plot!(pl_q, r.ψN, r.q; lw = 2, ls = style(pol), color = tcolor(t), marker = :circle, ms = 3, markercolor = tcolor(t), markerstrokewidth = 0)
    end
end
# the surfaces at the last time, over the electron temperature
lf = last_field[]
G = lf.G
pl_shape = heatmap(
    G.R1D, G.Z1D, transpose(lf.Te); c = :inferno, aspect_ratio = :equal, xlims = (1.0, 2.0), ylims = (-0.6, 0.6),
    title = "full_startup $(lf.t) ms: Te (eV)\nsurfaces: marching —, bicubic --", titlefontsize = 10, xlabel = "R (m)", ylabel = "Z (m)"
)
scatter!(pl_shape, G.R2D[lf.closed], G.Z2D[lf.closed]; ms = 2.5, color = :white, markerstrokewidth = 0, label = "")
for pol in policies[1:2]
    fsa = records[(lf.t, label(pol))].fsa
    for level in fsa.ψ[fsa.valid]
        l = surface(pol.tracer, lf.g, level, fsa.axis)
        plot!(pl_shape, l.R, l.Z; color = color(pol), ls = style(pol), lw = 1.5, label = "")
    end
end
savefig(plot(pl_shape, pl_Te, pl_q; layout = (1, 3), size = (1500, 550), margin = 5Plots.mm), joinpath(out, "full_startup.png"))
println("outputs in ", out)
