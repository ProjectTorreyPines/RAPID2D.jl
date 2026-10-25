# Field-line following on the (R, Z) grid: three ways to trace a field line, and what they find.
#
# In axisymmetry the poloidal projection of a field line is a ψ contour. Following it from a node to
# the wall gives the lengths transport reads, the poloidal length Lpol = ∫ dl and the length along B,
# Lc = ∫ (B/B_pol) dl, and tells which nodes lie on closed surfaces. Three tracers:
#   MarchingSquaresTracer (the default): the bilinear contour, cell by cell;
#   CubicTracer: the bicubic level set, by predictor–corrector steps;
#   RK4Tracer: RK4 along the interpolated nodal B.
#
# Part 1 is an analytic start-up field, a quadrupole null plus a current: an O-point between two
# X-points, inside a box. On three grids it draws
#   - the lines each tracer follows from the same starting nodes, against the exact lines;
#   - the last closed surface each tracer finds, against the exact separatrix, and the closed nodes;
#   - Lc against the normalized flux ψN, which diverges at the separatrix (ψN = 1).
# The exact lines are the bicubic tracer on a fine grid of the same field.
# Part 2 follows the full start-up run (single-quadrupole field, every module on except
# Global_JxB_Force) as its closed region forms: time, closed nodes and length errors of each tracer
# against a fine-step bicubic trace, and maps of Lc.
#
#   julia --project=examples examples/field_line_tracing.jl

include("common.jl")
using RAPID2D: initialize_grid_geometry, FieldLineGeometry, FieldLineFollowingResult, field_line_analysis!,
    trace_from_node!, trace_around!, cubic_trace!, find_o_point, published_lengths, default_max_length,
    MarchingSquaresTracer, CubicTracer, RK4Tracer, PathLengths, needs_nodal_B

name = "field_line_tracing"
out = output_dir(name)

tracers = (MarchingSquaresTracer(), CubicTracer(), RK4Tracer())
label(t) = t isa MarchingSquaresTracer ? "marching squares" : t isa CubicTracer ? "bicubic" : "RK4"
color(t) = t isa MarchingSquaresTracer ? :royalblue : t isa CubicTracer ? :darkorange : :forestgreen
style(t) = t isa MarchingSquaresTracer ? :solid : t isa CubicTracer ? :dash : :dot

"Every vertex a tracer visits."
struct Path
    R::Vector{Float64}
    Z::Vector{Float64}
end
Path() = Path(Float64[], Float64[])
RAPID2D.visit!(p::Path, R, Z, Bpol, Bϕ) = (push!(p.R, R); push!(p.Z, Z); p)

"The whole line through a node: backward reversed, then forward (forward alone when it closed)."
function line_through(trace)
    fwd = Path()
    e = trace(fwd, 1)
    e.termination === :closed && return fwd
    bwd = Path()
    trace(bwd, -1)
    return Path(vcat(reverse(bwd.R), fwd.R), vcat(reverse(bwd.Z), fwd.Z))
end

# ── Part 1: the analytic start-up field ────────────────────────────────────────────────────────
const F = 3.0                                   # R Bϕ
const RC, ZC, W = 1.5113, 0.0071, 0.15           # the current's centre (off the nodes) and width
ψe(R, Z) = 0.1 * ((R - 1.5)^2 - Z^2) - 0.02 * exp(-((R - RC)^2 + (Z - ZC)^2) / W^2)
∇ψe(R, Z) = (
    0.2 * (R - 1.5) + 0.04 * exp(-((R - RC)^2 + (Z - ZC)^2) / W^2) * (R - RC) / W^2,
    -0.2 * Z + 0.04 * exp(-((R - RC)^2 + (Z - ZC)^2) / W^2) * (Z - ZC) / W^2,
)
# stationary points of the exact ψ by Newton
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
ψX = min(ψe(Xu...), ψe(Xl...))                   # the separatrix: the first X-point reached
ψN(R, Z) = (ψe(R, Z) - ψO) / (ψX - ψO)
@printf("Part 1: O-point (%.4f, %.4f), X-points (%.4f, %.4f) and (%.4f, %.4f)\n", O..., Xu..., Xl...)

function field(NR, NZ)
    G = initialize_grid_geometry(NR, NZ, (1.0, 2.0), (-1.0, 1.0))
    ψ = ψe.(G.R2D, G.Z2D)
    return G, ψ, FieldLineGeometry(G, ψ; Bϕ = F ./ G.R2D, nodal_B = true)
end
Gf, _, gf = field(201, 401)                      # the exact lines: bicubic on a fine grid
exact_line(x0) = line_through((p, s) -> cubic_trace!(p, CubicTracer(), gf, x0, gf.itp(x0), s, default_max_length(gf)))

grids = ((16, 26), (30, 50), (60, 100))
# closed; closed near the separatrix; open just outside it, past the X-points; open far out
targets = ((O[1] + 0.06, ZC), (O[1] + 0.11, ZC), (O[1] + 0.16, ZC), (1.86, 0.0), (1.75, 0.6))
nearest(G, x) = (argmin(abs.(G.R1D .- x[1])), argmin(abs.(G.Z1D .- x[2])))
frame(; zoom = false, kw...) = plot(;
    aspect_ratio = :equal, xlabel = "R (m)", ylabel = "Z (m)", legend = false,
    xlims = zoom ? (1.25, 1.8) : (1.0, 2.0), ylims = zoom ? (-0.35, 0.35) : (-1.0, 1.0), kw...
)
Rf, Zf = range(1.0, 2.0; length = 301), range(-1.0, 1.0; length = 601)
ψgrid = [ψe(R, Z) for Z in Zf, R in Rf]
separatrix!(p) = contour!(p, Rf, Zf, ψgrid; levels = [ψX], color = :black, lw = 1.5, colorbar = false)

paths, lcfs, lengths, errors = Any[], Any[], Any[], Any[]
for (NR, NZ) in grids
    G, ψ, g = field(NR, NZ)
    nodes = [LinearIndices(ψ)[i, j] for i in 2:(NR - 1), j in 2:(NZ - 1)]
    flf = Dict(t => field_line_analysis!(FieldLineFollowingResult{Float64}(NR, NZ), g, t) for t in tracers)
    # the exact line through every node: closed or not, and its Lc (one circuit, or wall to wall)
    exact = Dict(
        map(nodes) do k
            x0 = (G.R2D[k], G.Z2D[k])
            Ls = map((1, -1)) do s
                L = PathLengths{Float64}()
                e = cubic_trace!(L, CubicTracer(), gf, x0, gf.itp(x0), s, default_max_length(gf))
                published_lengths(L, e.termination, e.R), e.termination === :closed
            end
            k => (Ls[1][2] ? Ls[1][1].Lc : Ls[1][1].Lc + Ls[2][1].Lc, Ls[1][2])
        end
    )

    # 1. lines from the same nodes
    for zoom in (false, true)
        first_panel = !zoom && NR == first(grids)[1]
        p = frame(; zoom, title = "$NR × $NZ" * (zoom ? ", closed region" : ""), legend = first_panel ? :bottomleft : false)
        scatter!(p, vec(G.R2D), vec(G.Z2D); ms = 1, color = :gray70, markerstrokewidth = 0, label = "")
        separatrix!(p)
        for (n, x) in enumerate(targets)
            i, j = nearest(G, x)
            ex = exact_line((G.R1D[i], G.Z1D[j]))
            plot!(p, ex.R, ex.Z; color = :black, lw = 3, alpha = 0.25, label = n == 1 ? "exact" : "")
            for t in tracers
                l = line_through((v, s) -> trace_from_node!(v, t, g, i, j, s))
                plot!(p, l.R, l.Z; color = color(t), ls = style(t), lw = 1.5, label = n == 1 ? label(t) : "")
            end
            scatter!(p, [G.R1D[i]], [G.Z1D[j]]; ms = 4, color = :red, markerstrokewidth = 0, label = n == 1 ? "starting node" : "")
        end
        push!(paths, p)
    end

    # 2. the last closed surface: bisection in ψN on whether the level closes around the O-point;
    #    gray dots are the nodes whose exact line closes, colored marks the nodes a tracer reads otherwise
    o = find_o_point(G, ψ, flf[MarchingSquaresTracer()].closed_surface_nids; itp = g.itp)
    p = frame(; zoom = true, title = "$NR × $NZ: last closed surface", legend = :bottomleft)
    separatrix!(p)
    exact_closed = [k for k in nodes if exact[k][2]]
    scatter!(p, G.R2D[exact_closed], G.Z2D[exact_closed]; ms = 3, color = :gray60, markerstrokewidth = 0, label = "closed (exact)")
    for t in tracers
        closes(x) = trace_around!(PathLengths{Float64}(), t, g, ψO + x * (ψX - ψO), o).termination === :closed
        lo, hi = 0.5, 1.5
        for _ in 1:30
            mid = (lo + hi) / 2
            closes(mid) ? (lo = mid) : (hi = mid)
        end
        l = Path()
        trace_around!(l, t, g, ψO + (lo - 1.0e-4) * (ψX - ψO), o)
        wrong = symdiff(intersect(flf[t].closed_surface_nids, nodes), exact_closed)
        plot!(p, l.R, l.Z; color = color(t), ls = style(t), lw = 2, label = @sprintf("%s: ψN = %.3f, %d node%s read otherwise", label(t), lo, length(wrong), length(wrong) == 1 ? "" : "s"))
        scatter!(p, G.R2D[wrong], G.Z2D[wrong]; ms = 7, marker = :xcross, color = color(t), markerstrokewidth = 2, label = "")
    end
    push!(lcfs, p)

    # 3. Lc against ψN: closed lines (one circuit) and lines outside the separatrix (wall to wall)
    ks = [k for k in nodes if exact[k][2] || ψN(G.R2D[k], G.Z2D[k]) > 1]          # leave out the private flux
    x = [ψN(G.R2D[k], G.Z2D[k]) for k in ks]
    first_panel = NR == first(grids)[1]
    p = plot(; xlabel = "ψN", ylabel = "Lc (m)", yscale = :log10, xlims = (0, 2), ylims = (5, 300), title = "$NR × $NZ: Lc", legend = first_panel ? :bottomright : false)
    e = plot(; xlabel = "ψN", ylabel = "|Lc / Lc exact − 1|", yscale = :log10, xlims = (0, 2), ylims = (1.0e-6, 3), title = "$NR × $NZ: error of Lc", legend = false)
    vline!(p, [1.0]; color = :black, ls = :dash, label = "separatrix")
    vline!(e, [1.0]; color = :black, ls = :dash)
    scatter!(p, x, [exact[k][1] for k in ks]; ms = 4, color = :black, markerstrokewidth = 0, label = "exact")
    for t in tracers
        Lc = [flf[t].termination_forward[k] === :closed ? flf[t].Lc_forward[k] : flf[t].Lc_tot[k] for k in ks]
        scatter!(p, x, Lc; ms = 2.5, color = color(t), markerstrokewidth = 0, label = label(t))
        scatter!(e, x, max.(abs.(Lc ./ [exact[k][1] for k in ks] .- 1), 1.0e-6); ms = 2.5, color = color(t), markerstrokewidth = 0)
    end
    push!(lengths, p)
    push!(errors, e)
end
order = [paths[1], paths[3], paths[5], paths[2], paths[4], paths[6]]
savefig(plot(order...; layout = (2, 3), size = (1500, 1500), margin = 4Plots.mm), joinpath(out, "lines.png"))
savefig(plot(lcfs...; layout = (1, 3), size = (1500, 650), margin = 4Plots.mm), joinpath(out, "last_closed_surface.png"))
savefig(plot(lengths..., errors...; layout = (2, 3), size = (1500, 900), margin = 5Plots.mm), joinpath(out, "connection_length.png"))
println("  figures: lines.png (the lines each tracer follows), last_closed_surface.png, connection_length.png")

# ── Part 2: the full start-up run ─────────────────────────────────────────────────────────────
reference = CubicTracer(max_step_cells = 0.02, max_turn = deg2rad(1), max_weight_change = 0.002)

function analysis(f, t)
    g = FieldLineGeometry(f.G, f.ψ; f.Bϕ, f.inside, nodal_B = needs_nodal_B(t), f.BR, f.BZ)
    run() = field_line_analysis!(FieldLineFollowingResult{Float64}(f.G.NR, f.G.NZ), g, t)
    flf = run()
    return flf, minimum(@elapsed(run()) for _ in 1:3)
end

function compare(f)
    ref, _ = analysis(f, reference)
    closed_ref = intersect(ref.closed_surface_nids, f.nodes)
    results = map(tracers) do t
        flf, time = analysis(f, t)
        closed = intersect(flf.closed_surface_nids, f.nodes)
        open = [n for n in f.nodes if ref.termination_forward[n] === :wall && flf.termination_forward[n] === :wall]
        eL = abs.(flf.Lpol_forward[open] .- ref.Lpol_forward[open])
        eC = abs.(flf.Lc_forward[open] ./ ref.Lc_forward[open] .- 1)
        @printf(
            "  %-16s %6.2f ms | closed nodes %3d (reference %3d, %d differ) | open lines: Lpol off by %.1e m (median), Lc by %.1e (median), %.1e (95th pct)\n",
            label(t), 1.0e3 * time, length(closed), length(closed_ref), length(symdiff(closed, closed_ref)), median(eL), median(eC), quantile(eC, 0.95)
        )
        flf
    end
    return ref, results
end

function maps(f, ref, results, title)
    G = f.G
    masked(A) = (M = fill(NaN, size(A)); M[f.nodes] .= A[f.nodes]; M)
    map2d(A; kw...) = heatmap(G.R1D, G.Z1D, transpose(A); aspect_ratio = :equal, xlabel = "R (m)", ylabel = "Z (m)", titlefontsize = 10, kw...)
    panels = [map2d(masked(log10.(ref.Lc_tot)); c = :viridis, title = "$title: log10 Lc (m)\nfine-step bicubic")]
    for (t, flf) in zip(tracers, results)
        t isa CubicTracer && continue
        err = masked(log10.(abs.(flf.Lc_tot ./ ref.Lc_tot .- 1) .+ 1.0e-6))
        push!(panels, map2d(err; clims = (-6, 0), c = :magma, title = "log10 |Lc / Lc fine − 1|\n$(label(t))"))
    end
    return panels
end

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
times_ms = (2.0, 4.0, 10.0, 20.0)
marks = Dict(round(Int, t * 1.0e-3 / config.dt) => t for t in times_ms)
fields = Dict{Float64, Any}()
run!(
    RP; callback_after_step = rp -> begin
        haskey(marks, rp.step) || return
        fields[marks[rp.step]] = (;
            G = rp.G, ψ = copy(rp.fields.ψ), Bϕ = copy(rp.fields.Bϕ), inside = rp.G.cell_state .>= 0,
            BR = copy(rp.fields.BR), BZ = copy(rp.fields.BZ), nodes = copy(rp.G.nodes.in_wall_nids),
        )
    end
)
panels = Any[]
for t in times_ms
    println("Part 2: full_startup at $t ms, $(length(fields[t].nodes)) in-wall nodes")
    ref, results = compare(fields[t])
    t == 10.0 && append!(panels, maps(fields[t], ref, results, "full_startup 10 ms"))
end
savefig(plot(panels...; layout = (1, 3), size = (1500, 650), margin = 5Plots.mm), joinpath(out, "full_startup_Lc.png"))
println("outputs in ", out)
