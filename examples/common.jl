# Shared plumbing for the examples: input/output paths, one setup call, and the plots
# every example makes. `include("common.jl")` from an example script.

using RAPID2D
using Plots
using Printf
using Statistics

ENV["GKSwstype"] = "100"   # headless GR: no plot windows while rendering frames

# Field and wall files shipped with the examples (BREAK format).
const DATA = joinpath(@__DIR__, "data")
const SINGLE_QUAD = joinpath(DATA, "SingleQuad_LV=+5.dat")   # static single-quadrupole null, 5 V loop voltage
const BOX_WALL = joinpath(DATA, "box_wall.dat")              # rectangular wall, 1–2 m × −1–1 m

# Device inputs not shipped here (<device>/<shot>/*.dat + <device>_First_Wall.dat), e.g. KSTAR.
const INPUT_PATH = get(ENV, "RAPID_INPUT_PATH", "")

# examples/output/<name>/, created on demand (git-ignored).
function output_dir(name::AbstractString)
    dir = joinpath(@__DIR__, "output", name)
    mkpath(dir)
    return dir
end

# config → flags → initialize!. `flags` are SimulationFlags keyword overrides.
function setup(config::SimulationConfig{Float64}; flags...)
    mkpath(config.Output_path)
    RP = RAPID{Float64}(config)
    RP.flags = SimulationFlags{Float64}(; flags...)
    initialize!(RP)
    return RP
end

# ── pure toroidal field and initial columns (current_diffusion, force_balance_control) ──
# The manual setup without its vertical field: Bϕ = R0B0/R, no poloidal field, and
# Eϕ = E0·R̄/R (E0 [V/m] at the mean R), so one loop voltage everywhere.
# Pass it as `SimulationConfig(manual = pure_toroidal(E0))`.
pure_toroidal(E0::Real) = ManualSetup{Float64}(BR = 0.0, BZ = 0.0, Eϕ = E0)

# Uniform column of radius `radius` centred at (cenR, cenZ).
function tophat(G; cenR, cenZ = 0.0, radius, n0)
    r = @. sqrt((G.R2D - cenR)^2 + (G.Z2D - cenZ)^2)
    return @. ifelse(r < radius, n0, 0.0)
end

# Gaussian of 1σ width `radius`, cut off at r = radius.
function masked_gaussian(G; cenR, cenZ = 0.0, radius, n0)
    r2 = @. (G.R2D - cenR)^2 + (G.Z2D - cenZ)^2
    return @. ifelse(r2 < radius^2, n0 * exp(-r2 / (2 * radius^2)), 0.0)
end

# Column density/temperature at rest; ions follow electrons.
function set_column!(RP::RAPID, n; Te_eV = 10.0, Ti_eV = 0.03)
    RP.plasma.ne .= n
    RP.plasma.ni .= n
    fill!(RP.plasma.Te_eV, Te_eV)
    fill!(RP.plasma.Ti_eV, Ti_eV)
    fill!(RP.plasma.ue_para, 0.0)
    fill!(RP.plasma.ui_para, 0.0)
    return RP
end

# ── per-step record of what the snapshots do not carry ─────────────────────────────
Base.@kwdef struct StepRecord
    t::Vector{Float64} = Float64[]
    n_closed::Vector{Int} = Int[]        # nodes on closed field lines (FLF cadence)
    I_tor::Vector{Float64} = Float64[]
end

# `run_simulation!(RP; callback_after_step = recorder(rec))`
recorder(rec::StepRecord) = RP -> begin
    push!(rec.t, RP.time_s)
    push!(rec.n_closed, length(RP.flf.closed_surface_nids))
    push!(rec.I_tor, sum(RP.plasma.Jϕ) * RP.G.dR * RP.G.dZ)
    return nothing
end

# Run to t_end, or as far as it gets: a failure mid-run is reported, and the snapshots
# recorded up to that point stay available for the plots that follow.
function run!(RP::RAPID; kw...)
    try
        run_simulation!(RP; kw...)
    catch err
        @printf("\n*** run stopped at t = %.6e s (step %d): %s\n", RP.time_s, RP.step, sprint(showerror, err))
        bt = stacktrace(catch_backtrace())
        foreach(f -> println("    ", f), bt[1:min(12, length(bt))])
    end
    return RP
end

function summarize(RP::RAPID; extra = "")
    s = RP.diagnostics.snaps0D
    @printf(
        "t = %.3f ms | ⟨ne⟩ %.3g → %.3g m⁻³ | ⟨Te⟩ %.2f eV | I_tor %.3g A | closed-line nodes %d %s\n",
        RP.time_s * 1.0e3, s[1].ne, s[end].ne, s[end].Te_eV, s[end].I_tor,
        length(RP.flf.closed_surface_nids), extra
    )
    return nothing
end

# ── 0D traces ──────────────────────────────────────────────────────────────────────
# ⟨ne⟩, |⟨ue∥⟩|, ⟨Ke⟩ and I_tor against time; several runs overlay.
function plot_traces(runs::Vector{<:Pair}; file)
    p1 = plot(ylabel = "⟨ne⟩ (m⁻³)", yscale = :log10, legend = :bottomright)
    p2 = plot(ylabel = "|⟨ue∥⟩| (m/s)", legend = false)
    p3 = plot(ylabel = "⟨Ke⟩ (eV)", legend = false)
    p4 = plot(ylabel = "I_tor (A)", xlabel = "t (ms)", legend = false)
    for (label, RP) in runs
        s = RP.diagnostics.snaps0D
        t = s.time_s .* 1.0e3
        plot!(p1, t, max.(s.ne, 1.0); lw = 2, label)
        plot!(p2, t, abs.(s.ue_para); lw = 2)
        plot!(p3, t, s.Ke_eV; lw = 2)
        plot!(p4, t, s.I_tor; lw = 2)
    end
    fig = plot(p1, p2, p3, p4; layout = (4, 1), size = (700, 1000), left_margin = 8Plots.mm)
    savefig(fig, file)
    return fig
end
plot_traces(RP::RAPID; file) = plot_traces(["" => RP]; file)

# The 8-panel dashboard from RAPID2DPlotsExt (E∥ components, ν_iz, loss rate, ...).
plot_dashboard(RP::RAPID; file) = savefig(plot(RP.diagnostics.snaps0D), file)

# ── 2D snapshots ───────────────────────────────────────────────────────────────────
# Color range of one field over the whole run; log10 for densities.
function field_clims(RP::RAPID, field::Symbol)
    vals = [RAPID2D.snapshot_matrix(s, field) for s in RP.diagnostics.snaps2D]
    if RAPID2D.is_density_field(field)
        top = log10(max(maximum(maximum, vals), 10.0))
        return (max(top - 8, 0.0), top)
    end
    lo, hi = minimum(minimum, vals), maximum(maximum, vals)
    return hi > lo ? (lo, hi) : (lo, lo + 1.0)
end

# One field of one snapshot: heatmap, wall outline, ψ contours (= field lines) when ψ varies.
function panel2D(RP::RAPID, snap, field::Symbol; clims, title = string(field))
    G = RP.G
    v = RAPID2D.snapshot_matrix(snap, field)
    if RAPID2D.is_density_field(field)
        v = log10.(max.(v, 1.0))
        title = "log10 " * title
    end
    p = heatmap(
        G.R1D, G.Z1D, permutedims(v);
        aspect_ratio = :equal, c = :turbo, clims,
        xlims = extrema(G.R1D), ylims = extrema(G.Z1D),
        xticks = range(extrema(G.R1D)...; length = 4), xformatter = x -> @sprintf("%.1f", x),
        title, titlefontsize = 9, xlabel = "R (m)", ylabel = "Z (m)", framestyle = :box
    )
    ψ = snap.ψ
    if maximum(ψ) - minimum(ψ) > 0
        contour!(p, G.R1D, G.Z1D, permutedims(ψ); levels = 15, c = :white, lw = 0.6, colorbar_entry = false)
    end
    plot!(p, vcat(RP.wall.R, RP.wall.R[1]), vcat(RP.wall.Z, RP.wall.Z[1]); c = :red, lw = 1.5, label = "")
    return p
end

# One field at the snapshots nearest to `times_ms`, side by side.
function plot_snapshots2D(RP::RAPID, field::Symbol, times_ms; file)
    s2 = RP.diagnostics.snaps2D
    t_ms = [s.time_s for s in s2] .* 1.0e3
    idx = unique(argmin(abs.(t_ms .- t)) for t in times_ms)
    cl = field_clims(RP, field)
    panels = [
        panel2D(RP, s2[k], field; clims = cl, title = @sprintf("%s  %.2f ms", field, t_ms[k]))
            for k in idx
    ]
    ncol = min(length(idx), 4)
    nrow = cld(length(idx), ncol)
    fig = plot(panels...; layout = (nrow, ncol), size = (360 * ncol, 520 * nrow), margin = 4Plots.mm)
    savefig(fig, file)
    return fig
end

# mp4 over the 2D snapshots: rows = runs, columns = fields (runs share the cadence).
function animate2D(runs::Vector{<:Pair}, fields::Vector{Symbol}; file, fps = 10)
    clims = Dict((lbl, f) => field_clims(RP, f) for (lbl, RP) in runs for f in fields)
    nsnap = minimum(length(RP.diagnostics.snaps2D) for (_, RP) in runs)
    anim = @animate for k in 1:nsnap
        panels = [
            panel2D(
                RP, RP.diagnostics.snaps2D[k], f;
                clims = clims[(lbl, f)], title = strip("$lbl $f")
            )
                for (lbl, RP) in runs for f in fields
        ]
        t_ms = first(runs)[2].diagnostics.snaps2D[k].time_s * 1.0e3
        plot(
            panels...; layout = (length(runs), length(fields)),
            size = (380 * length(fields), 520 * length(runs)),
            plot_title = @sprintf("t = %.3f ms", t_ms)
        )
    end
    mp4(anim, file; fps, loop = 0)
    return file
end
animate2D(RP::RAPID, fields::Vector{Symbol}; kw...) = animate2D(["" => RP], fields; kw...)

# ── flux-surface frames: nₑ, Tₑ, u∥ and Jϕ side by side ──────────────────────────────
# Five panels: log10 nₑ (seed level to 1e18 m⁻³, the whole run on one scale), then nₑ, Tₑ,
# u∥ and Jϕ on linear scales of the frame itself: [0, top] for nₑ and Tₑ, ±max for u∥ and
# Jϕ, over the nodes that hold plasma. `top` is the maximum unless it stands out by more
# than half above the 99th percentile; a spike then saturates its pixel and the title
# keeps the maximum. Each panel carries faint ψ contours, the magnetic LCFS (thick cyan:
# through an X-point or touching the wall) with ψ_N = 0.2 … 0.8 inside it and its geometric
# centre (cyan ◆, the reference of the Shafranov shift), the closed-surface basis's LCFS
# (orange dashed: where closed lines with density end), the O-point (magenta ○), the X-points
# inside the wall (magenta ×), the current centroid (lime +) and the wall.
const FRAME_PANELS = (:log_ne, :ne, :Te, :ue, :J)
const MARKER_KEY = "(— magnetic LCFS, ┄ basis LCFS; + current centroid, ○ O-point, ◆ LCFS centre, × X-points)"

# Marching squares: the polyline(s) of F = c on the grid (R, Z), NaN-separated.
function isoline(R, Z, F, c)
    xs, ys = Float64[], Float64[]
    for j in 1:(length(Z) - 1), i in 1:(length(R) - 1)
        q = ((R[i], Z[j], F[i, j]), (R[i + 1], Z[j], F[i + 1, j]), (R[i + 1], Z[j + 1], F[i + 1, j + 1]), (R[i], Z[j + 1], F[i, j + 1]))
        pts = Tuple{Float64, Float64}[]
        for e in 1:4
            (x1, y1, f1), (x2, y2, f2) = q[e], q[mod1(e + 1, 4)]
            if (f1 - c) * (f2 - c) < 0
                s = (c - f1) / (f2 - f1)
                push!(pts, (x1 + s * (x2 - x1), y1 + s * (y2 - y1)))
            end
        end
        for k in 1:2:(length(pts) - 1)
            push!(xs, pts[k][1], pts[k + 1][1], NaN); push!(ys, pts[k][2], pts[k + 1][2], NaN)
        end
    end
    return xs, ys
end

# Break a polyline where it leaves the box (r1, r2) × (z1, z2).
function clip_to_box!(x, y, (r1, r2), (z1, z2))
    for k in eachindex(x)
        isnan(x[k]) || (r1 <= x[k] <= r2 && z1 <= y[k] <= z2) || (x[k] = NaN; y[k] = NaN)
    end
    return x, y
end

function in_wall_mask(RP::RAPID)
    inside = falses(RP.G.NR, RP.G.NZ)
    inside[RP.G.nodes.in_wall_nids] .= true
    return inside
end

# The X-points of ψ inside the wall, as (R, Z) vectors (`RAPID2D.find_critical_points`).
function find_x_points(RP::RAPID, ψ::AbstractMatrix; inside = in_wall_mask(RP))
    pts = RAPID2D.find_critical_points(RP.G, ψ; kind = RAPID2D.XPoint(), inside)
    return [p.R for p in pts], [p.Z for p in pts]
end

# A closed contour the bicubic tracer follows around the O-point `o` at `level`, as a polygon
# with two centres: the geometric centre ((R_max + R_min)/2, (Z_max + Z_min)/2), the reference of
# the Shafranov shift, and the area centroid (∫R dA / A, ∫Z dA / A). `nothing` if it does not close.
struct ContourVertices
    R::Vector{Float64}
    Z::Vector{Float64}
end
RAPID2D.visit!(c::ContourVertices, R, Z, Bpol, Bϕ) = (push!(c.R, R); push!(c.Z, Z); c)

function trace_closed_contour(g, level, o)
    c = ContourVertices(Float64[], Float64[])
    RAPID2D.trace_around!(c, RAPID2D.CubicTracer(), g, level, o).termination === :closed || return nothing
    R, Z = c.R, c.Z
    (r1, r2), (z1, z2) = extrema(R), extrema(Z)
    A = CR = CZ = 0.0
    for k in eachindex(R)
        n = mod1(k + 1, length(R))
        w = R[k] * Z[n] - R[n] * Z[k]
        A += w / 2
        CR += (R[k] + R[n]) * w / 6
        CZ += (Z[k] + Z[n]) * w / 6
    end
    return (; R, Z, R_geo = (r1 + r2) / 2, Z_geo = (z1 + z2) / 2, R_area = CR / A, Z_area = CZ / A)
end

# The two LCFS of a 2D snapshot, as traced contours: `basis`, the closed-surface basis's (the
# snapshot's `ψ_lcfs`, which stops at the admissible region of closed lines and density), and
# `magnetic`, set by the field and the wall alone (`RAPID2D.magnetic_lcfs`: through an X-point or
# touching the wall, `limited_by`). Each `nothing` when there is none.
function snapshot_lcfs(RP::RAPID, snap; inside = in_wall_mask(RP))
    isnan(snap.ψ_lcfs) && return (; basis = nothing, magnetic = nothing)
    g = RAPID2D.FieldLineGeometry(RP.G, snap.ψ; inside = RP.G.cell_state .>= 0)
    o = (; R = snap.R_axis, Z = snap.Z_axis, ψ = snap.ψ_axis)
    basis = trace_closed_contour(g, snap.ψ_lcfs, o)
    m = RAPID2D.magnetic_lcfs(RP.G, snap.ψ, o, inside; g)
    c = m === nothing ? nothing : trace_closed_contour(g, m.ψ_trace, o)
    magnetic = c === nothing ? nothing : (; c..., m.ψ_lcfs, m.limited_by, R_limit = m.R, Z_limit = m.Z)
    return (; basis, magnetic)
end

# The top of a linear colour range: the maximum, or the 99th percentile when the maximum
# is a spike (more than 1.5 times it).
function robust_top(vals)
    hi, p99 = maximum(vals), quantile(vals, 0.99)
    return hi <= 1.5p99 ? hi : p99
end
range_title(name, unit, top, hi) = top < hi ? @sprintf("%s [0, %.3g] %s, max %.3g", name, top, unit, hi) : @sprintf("%s [0, %.3g] %s", name, top, unit)

# One panel of one 2D snapshot, with the flux surfaces, the LCFS and its centre, the X-points
# and the 0D snapshot's centroid.
function flux_panel(
        RP::RAPID, snap, snap0, kind::Symbol; inside, ne_range, prefix = "",
        xpoints = find_x_points(RP, snap.ψ; inside), lcfs = snapshot_lcfs(RP, snap; inside),
    )
    R, Z = RP.G.R1D, RP.G.Z1D
    ne = snap.ne
    plasma = inside .& isfinite.(ne) .& (ne .> 1.0e-3 * maximum(ne[inside .& isfinite.(ne)]))
    if kind === :log_ne
        v, cl, cm, ttl = log10.(max.(ne, 1.0)), ne_range, :inferno, @sprintf("log10 nₑ [%.0f, %.0f]", ne_range...)
    elseif kind in (:ne, :Te)
        v = Float64.(kind === :ne ? ne : snap.Te_eV)
        top = max(robust_top(v[plasma]), kind === :ne ? 1.0 : 1.0e-3)
        v, cl, cm = v, (0.0, top), :inferno
        ttl = kind === :ne ? range_title("nₑ", "m⁻³", top, maximum(v[plasma])) : range_title("Tₑ", "eV", top, maximum(v[plasma]))
    else
        v = Float64.(kind === :J ? snap.Jϕ : snap.ue_para)
        s = max(maximum(abs, v[plasma]), kind === :J ? 1.0e-3 : 1.0)
        cl, cm = (-s, s), :RdBu
        ttl = kind === :J ? @sprintf("Jϕ [±%.2g kA/m²]", 1.0e-3s) : @sprintf("u∥ [±%.2g m/s]", s)
    end
    v[.!inside] .= NaN
    BOX = (extrema(RP.wall.R), extrema(RP.wall.Z))
    p = heatmap(
        R, Z, permutedims(v); c = cm, clims = cl, aspect_ratio = :equal, xlims = BOX[1], ylims = BOX[2],
        title = prefix * ttl, titlefontsize = 8, framestyle = :box, colorbar = false, tickfontsize = 6,
        xticks = round.(range(BOX[1]...; length = 4)[2:3]; digits = 1),
    )
    light = kind in (:log_ne, :ne, :Te)
    faint, dense = light ? (RGBA(1, 1, 1, 0.3), RGBA(1, 1, 1, 0.75)) : (RGBA(0, 0, 0, 0.3), RGBA(0, 0, 0, 0.75))
    ψ = snap.ψ
    lo, hi = extrema(ψ[inside])
    if hi > lo
        for c in range(lo, hi; length = 18)[2:(end - 1)]
            plot!(p, isoline(R, Z, ψ, c)...; color = faint, lw = 0.5, label = "")
        end
    end
    if lcfs.basis !== nothing
        b = lcfs.basis
        plot!(p, vcat(b.R, b.R[1]), vcat(b.Z, b.Z[1]); color = :orange, lw = 1.5, ls = :dash, label = "")
    end
    if lcfs.magnetic !== nothing
        m = lcfs.magnetic
        ψa, ψl = snap.ψ_axis, m.ψ_lcfs
        h = 2 * RP.G.dR
        box = (extrema(m.R) .+ (-h, h), extrema(m.Z) .+ (-h, h))
        for ψN in 0.2:0.2:0.8
            plot!(p, clip_to_box!(isoline(R, Z, ψ, ψa + ψN * (ψl - ψa))..., box...)...; color = dense, lw = 0.6, label = "")
        end
        plot!(p, vcat(m.R, m.R[1]), vcat(m.Z, m.Z[1]); color = :cyan, lw = 2.0, label = "")
        scatter!(p, [m.R_geo], [m.Z_geo]; marker = :diamond, ms = 6, color = :cyan, msc = :black, msw = 1, label = "")
    end
    if !isnan(snap.ψ_lcfs)
        # the O-point, an open circle drawn as a line: GR fills a transparent marker black
        r, θ = 0.75 * min(RP.G.dR, RP.G.dZ), range(0, 2π; length = 33)
        plot!(p, snap.R_axis .+ r .* cos.(θ), snap.Z_axis .+ r .* sin.(θ); color = :magenta, lw = 2, label = "")
    end
    isempty(xpoints[1]) || scatter!(p, xpoints...; marker = :xcross, ms = 6, msw = 2, color = :magenta, label = "")
    isnan(snap0.R_cen_by_Jϕ) || scatter!(p, [snap0.R_cen_by_Jϕ], [snap0.Z_cen_by_Jϕ]; marker = :+, ms = 7, msw = 2, color = :lime, label = "")
    plot!(p, vcat(RP.wall.R, RP.wall.R[1]), vcat(RP.wall.Z, RP.wall.Z[1]); color = :red, lw = 1.2, label = "")
    return p
end

# The colour range of log10 nₑ: from the seed level (the first snapshot's peak) to 1e18 m⁻³.
function frame_ne_range(RP::RAPID)
    ne0 = RP.diagnostics.snaps2D[1].ne
    return (floor(log10(max(maximum(x for x in ne0 if isfinite(x)), 1.0))), 18.0)
end
nearest_snap0D(RP::RAPID, t) = (s0 = RP.diagnostics.snaps0D; s0[argmin(abs.(s0.time_s .- t))])

# One frame: the panels of 2D snapshot k under a two-line heading (title; currents).
function fields_frame(RP::RAPID, k::Integer; title, inside = in_wall_mask(RP), ne_range = frame_ne_range(RP))
    snap = RP.diagnostics.snaps2D[k]
    snap0 = nearest_snap0D(RP, snap.time_s)
    amps(I) = abs(I) < 1.0e3 ? @sprintf("%.3g A", abs(I)) : @sprintf("%.2f kA", 1.0e-3abs(I))
    info = "|I_tor| " * amps(snap0.I_tor) * ", on closed lines " * (isnan(snap0.I_closed) ? "none" : amps(snap0.I_closed))
    RP.flags.vertical_field_control isa RAPID2D.NoVerticalFieldControl || (info *= @sprintf(", control a₁ %.2f mT", 1.0e3snap0.ctrl_a1))
    head = plot(; framestyle = :none, xlims = (0, 1), ylims = (0, 1), legend = false)
    annotate!(head, 0.5, 0.68, text(@sprintf("%s, t = %.2f ms", title, 1.0e3snap.time_s), 11))
    annotate!(head, 0.5, 0.2, text(info * "   " * MARKER_KEY, 10))
    xpoints, lcfs = find_x_points(RP, snap.ψ; inside), snapshot_lcfs(RP, snap; inside)
    panels = [flux_panel(RP, snap, snap0, kind; inside, ne_range, xpoints, lcfs) for kind in FRAME_PANELS]
    n = length(FRAME_PANELS)
    return plot(head, panels...; layout = @layout([a{0.07h}; grid(1, n)]), size = (300n, 760))
end

# mp4 of every 2D snapshot as a `fields_frame`.
function animate_fields(RP::RAPID; file, title, fps = 15)
    inside, ne_range = in_wall_mask(RP), frame_ne_range(RP)
    anim = @animate for k in eachindex(RP.diagnostics.snaps2D)
        fields_frame(RP, k; title, inside, ne_range)
    end
    mp4(anim, file; fps, loop = 0)
    return file
end

# The panels (rows) at the 2D snapshots nearest to `times_ms` (columns).
function plot_fields_snapshots(RP::RAPID, times_ms; file, title)
    s2 = RP.diagnostics.snaps2D
    t_ms = [s.time_s for s in s2] .* 1.0e3
    idx = unique(argmin(abs.(t_ms .- t)) for t in times_ms)
    inside, ne_range = in_wall_mask(RP), frame_ne_range(RP)
    panels = [
        flux_panel(RP, s2[k], nearest_snap0D(RP, s2[k].time_s), kind; inside, ne_range, prefix = @sprintf("t = %.2f ms\n", t_ms[k]))
            for kind in FRAME_PANELS for k in idx
    ]
    n = length(FRAME_PANELS)
    fig = plot(
        panels...; layout = (n, length(idx)), size = (250 * length(idx), 470n), plot_titlefontsize = 11,
        plot_title = title * "   " * MARKER_KEY,
    )
    savefig(fig, file)
    return fig
end

# ── verdicts ───────────────────────────────────────────────────────────────────────
# An example that checks itself against its prediction saves its figure as <name>__PASS.png
# or <name>__FAIL.png, and removes the figure of the other verdict (and an unlabelled one), so
# the output folder shows the last run's verdict. The verdict and `detail` also head the
# figure and are printed.
function save_with_verdict(fig, dir::AbstractString, name::AbstractString, pass::Bool, detail::AbstractString)
    verdict = pass ? "PASS" : "FAIL"
    for old in ("$name.png", "$(name)__PASS.png", "$(name)__FAIL.png")
        isfile(joinpath(dir, old)) && rm(joinpath(dir, old))
    end
    # the heading wrapped to the figure's width (about 9 px per character at this size)
    width = fig[:size][1]
    words, lines = split("$verdict: $detail"), [""]
    for w in words
        if isempty(lines[end]) || length(lines[end]) + length(w) + 1 <= width / 9
            lines[end] = isempty(lines[end]) ? String(w) : lines[end] * " " * w
        else
            push!(lines, String(w))
        end
    end
    plot!(
        fig; plot_title = join(lines, "\n"), plot_titlefontsize = 11, plot_titlefontcolor = pass ? :darkgreen : :red3,
        plot_titlevspan = min(0.05 + 0.03 * (length(lines) - 1), 0.2),
    )
    file = joinpath(dir, "$(name)__$(verdict).png")
    savefig(fig, file)
    println(verdict, "  ", name, ": ", detail)
    return file
end
