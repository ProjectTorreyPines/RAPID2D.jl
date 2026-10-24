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
# The 0D snapshots of a run in memory (`RAPID`) or read from its files (`load_run`).
snaps0D(RP::RAPID) = RP.diagnostics.snaps0D
snaps0D(run::NamedTuple) = run.snaps0D

# ⟨ne⟩, |⟨ue∥⟩|, ⟨Ke⟩ and I_tor against time; several runs overlay.
function plot_traces(runs::Vector{<:Pair}; file)
    p1 = plot(ylabel = "⟨ne⟩ (m⁻³)", yscale = :log10, legend = :bottomright)
    p2 = plot(ylabel = "|⟨ue∥⟩| (m/s)", legend = false)
    p3 = plot(ylabel = "⟨Ke⟩ (eV)", legend = false)
    p4 = plot(ylabel = "I_tor (A)", xlabel = "t (ms)", legend = false)
    for (label, RP) in runs
        s = snaps0D(RP)
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
plot_traces(run::Union{RAPID, NamedTuple}; file) = plot_traces(["" => run]; file)

# The 8-panel dashboard from RAPID2DPlotsExt (E∥ components, ν_iz, loss rate, ...).
plot_dashboard(run::Union{RAPID, NamedTuple}; file) = savefig(plot(snaps0D(run)), file)

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

# ── a run's files ──────────────────────────────────────────────────────────────────
# What a finished run wrote to its output folder, read back for plotting without running: the grid
# with its wall masks (grid.bp), the wall polygon (wall.bp) and the snapshots (snap0D.bp,
# snap2D.bp). `controlled`: the ideal vertical-field control acted (a nonzero a₁).
function load_run(dir::AbstractString; prefix::AbstractString = "")
    path(name) = joinpath(dir, prefix * name)
    grid, wall = RAPID2D.ADIOS2.adios_load(path("grid.bp")), RAPID2D.ADIOS2.adios_load(path("wall.bp"))
    # a one-step file keeps its step axis: drop it
    step(a) = ndims(a) > 1 && size(a, ndims(a)) == 1 ? dropdims(a; dims = ndims(a)) : a
    R1D, Z1D = vec(grid["R1D"]), vec(grid["Z1D"])
    G = RAPID2D.initialize_grid_geometry(length(R1D), length(Z1D), extrema(R1D), extrema(Z1D))
    G.R1D .= R1D
    G.Z1D .= Z1D
    G.cell_state = Int.(step(grid["cell_state"]))
    G.nodes.state .= step(grid["nodes/state"])
    G.nodes.in_wall_nids = Int.(vec(grid["nodes/in_wall_nids"]))
    s0 = RAPID2D.adiosBP_to_snap0D(path("snap0D.bp"))
    s2 = RAPID2D.adiosBP_to_snap2D(path("snap2D.bp"))
    return (; dir, G, wall = (; R = vec(wall["R"]), Z = vec(wall["Z"])), snaps0D = s0, snaps2D = s2, controlled = any(x -> x.ctrl_a1 != 0, s0))
end

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

function in_wall_mask(run)
    inside = falses(run.G.NR, run.G.NZ)
    inside[run.G.nodes.in_wall_nids] .= true
    return inside
end

# The X-points of ψ inside the wall, as (R, Z) vectors (`RAPID2D.find_critical_points`).
function find_x_points(run, ψ::AbstractMatrix; inside = in_wall_mask(run))
    pts = RAPID2D.find_critical_points(run.G, ψ; kind = RAPID2D.XPoint(), inside)
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

# The two LCFS of a 2D snapshot as its file holds them, traced for drawing: `basis`, the
# closed-surface basis's (`ψ_lcfs`, which stops at the admissible region of closed lines and
# density), and `magnetic`, set by the field and the wall alone (`ψ_lcfs_magnetic`, with the
# configuration, X-points and geometric centre the snapshot recorded). A diverted LCFS is a
# separatrix through a null, so its contour is traced just inside it. Each `nothing` when there
# is none.
function snapshot_lcfs(run, snap)
    isnan(snap.ψ_lcfs) && return (; basis = nothing, magnetic = nothing)
    g = RAPID2D.FieldLineGeometry(run.G, snap.ψ; inside = run.G.cell_state .>= 0)
    o = (; R = snap.R_axis, Z = snap.Z_axis, ψ = snap.ψ_axis)
    basis = trace_closed_contour(g, snap.ψ_lcfs, o)
    snap.lcfs_configuration == 0 && return (; basis, magnetic = nothing)
    configuration = RAPID2D.LCFS_CONFIGURATIONS[snap.lcfs_configuration]
    s, Δψ = sign(snap.ψ_lcfs_magnetic - snap.ψ_axis), abs(snap.ψ_lcfs_magnetic - snap.ψ_axis)
    Δψ_trace = configuration === :limited ? Δψ : RAPID2D.level_inside(g, o, s, Δψ)
    c = Δψ_trace === nothing ? nothing : trace_closed_contour(g, snap.ψ_axis + s * Δψ_trace, o)
    c === nothing && return (; basis, magnetic = nothing)
    xpoints = filter(x -> !isnan(x.R), [(; R = snap.R_xpoint1, Z = snap.Z_xpoint1), (; R = snap.R_xpoint2, Z = snap.Z_xpoint2)])
    magnetic = (; c..., R_geo = snap.R_geo, Z_geo = snap.Z_geo, ψ_lcfs = snap.ψ_lcfs_magnetic, configuration, xpoints)
    return (; basis, magnetic)
end

# The top of a linear colour range: the maximum, or the 99th percentile when the maximum
# is a spike (more than 1.5 times it).
function robust_top(vals)
    hi, p99 = maximum(vals), quantile(vals, 0.99)
    return hi <= 1.5p99 ? hi : p99
end
range_title(name, unit, top, hi) = top < hi ? @sprintf("%s [0, %.3g] %s, max %.3g", name, top, unit, hi) : @sprintf("%s [0, %.3g] %s", name, top, unit)

# The values, colour range, colour map and titles of one panel of a 2D snapshot: `full`
# names the range, `short` only the quantity in units of its colour bar (nₑ and u∥ scaled by
# their range's power of ten, `factor`) and the maximum when a spike is clipped.
superscript(n::Integer) = map(c -> c == '-' ? '⁻' : "⁰¹²³⁴⁵⁶⁷⁸⁹"[nextind("⁰¹²³⁴⁵⁶⁷⁸⁹", 0, c - '0' + 1)], string(n))
decade(x) = x > 0 ? floor(Int, log10(x)) : 0
function panel_scale(snap, kind::Symbol; inside, ne_range)
    ne = snap.ne
    plasma = inside .& isfinite.(ne) .& (ne .> 1.0e-3 * maximum(ne[inside .& isfinite.(ne)]))
    if kind === :log_ne
        v, cl, cm, factor = log10.(max.(ne, 1.0)), ne_range, :inferno, 1.0
        full, short = @sprintf("log10 nₑ [%.0f, %.0f]", ne_range...), "log10 nₑ [m⁻³]"
    elseif kind in (:ne, :Te)
        v = Float64.(kind === :ne ? ne : snap.Te_eV)
        hi = maximum(v[plasma])
        top = max(robust_top(v[plasma]), kind === :ne ? 1.0 : 1.0e-3)
        v, cl, cm = v, (0.0, top), :inferno
        e = kind === :ne ? decade(top) : 0
        factor = 10.0^e
        name, unit = kind === :ne ? ("nₑ", "10" * superscript(e) * " m⁻³") : ("Tₑ", "eV")
        full = range_title(name, kind === :ne ? "m⁻³" : "eV", top, hi)
        short = top < hi ? @sprintf("%s [%s], max %.3g", name, unit, hi / factor) : @sprintf("%s [%s]", name, unit)
    else
        v = Float64.(kind === :J ? snap.Jϕ : snap.ue_para)
        sm = max(maximum(abs, v[plasma]), kind === :J ? 1.0e-3 : 1.0)
        cl, cm = (-sm, sm), :RdBu
        e = kind === :J && 3 <= decade(sm) <= 5 ? 3 : decade(sm)      # Jϕ in kA/m² in its usual range
        factor = 10.0^e
        full = kind === :J ? @sprintf("Jϕ [±%.2g kA/m²]", 1.0e-3sm) : @sprintf("u∥ [±%.2g m/s]", sm)
        short = kind === :J ? (e == 3 ? "Jϕ [kA/m²]" : "Jϕ [10" * superscript(e) * " A/m²]") : "u∥ [10" * superscript(e) * " m/s]"
    end
    return (; v, cl, cm, full, short, factor)
end

# A thin horizontal colour bar for a panel's scale, three ticks in the units of its short title.
function colorbar_strip(sc, kind::Symbol)
    x = range(sc.cl...; length = 256)
    ticks = kind === :log_ne ? collect(ceil(sc.cl[1]):3:sc.cl[2]) : collect(range(sc.cl...; length = 3))
    label(t) = kind === :log_ne ? @sprintf("%.0f", t) : @sprintf("%.3g", t / sc.factor)
    return heatmap(
        x, [0.0, 1.0], repeat(collect(x)', 2); c = sc.cm, clims = sc.cl, colorbar = false, xlims = sc.cl,
        yticks = false, xticks = (ticks, label.(ticks)), tickfontsize = 7, framestyle = :box,
    )
end

# One panel of one 2D snapshot, with the flux surfaces, the LCFS and its centre, the X-points
# and the 0D snapshot's centroid; titled with its range, or (`short`) for a colour bar below it.
function flux_panel(
        run, snap, snap0, kind::Symbol; inside, ne_range, prefix = "", short = false,
        scale = panel_scale(snap, kind; inside, ne_range),
        xpoints = find_x_points(run, snap.ψ; inside), lcfs = snapshot_lcfs(run, snap),
    )
    R, Z = run.G.R1D, run.G.Z1D
    v, cl, cm = copy(scale.v), scale.cl, scale.cm
    ttl = short ? scale.short : scale.full
    v[.!inside] .= NaN
    BOX = (extrema(run.wall.R), extrema(run.wall.Z))
    p = heatmap(
        R, Z, permutedims(v); c = cm, clims = cl, aspect_ratio = :equal, xlims = BOX[1], ylims = BOX[2],
        title = prefix * ttl, titlefontsize = 8, framestyle = :box, colorbar = false, tickfontsize = 6,
        xticks = round.(range(BOX[1]...; length = 4)[2:3]; digits = 1),
    )
    light = kind in (:log_ne, :ne, :Te)
    faint, dense = light ? (RGBA(1, 1, 1, 0.45), RGBA(1, 1, 1, 0.85)) : (RGBA(0, 0, 0, 0.4), RGBA(0, 0, 0, 0.8))
    ψ = snap.ψ
    lo, hi = extrema(ψ[inside])
    if hi > lo
        for c in range(lo, hi; length = 18)[2:(end - 1)]
            plot!(p, isoline(R, Z, ψ, c)...; color = faint, lw = 1.0, label = "")
        end
    end
    if lcfs.basis !== nothing
        b = lcfs.basis
        plot!(p, vcat(b.R, b.R[1]), vcat(b.Z, b.Z[1]); color = :orange, lw = 2.0, ls = :dash, label = "")
    end
    if lcfs.magnetic !== nothing
        m = lcfs.magnetic
        ψa, ψl = snap.ψ_axis, m.ψ_lcfs
        h = 2 * run.G.dR
        box = (extrema(m.R) .+ (-h, h), extrema(m.Z) .+ (-h, h))
        for ψN in 0.2:0.2:0.8
            plot!(p, clip_to_box!(isoline(R, Z, ψ, ψa + ψN * (ψl - ψa))..., box...)...; color = dense, lw = 1.2, label = "")
        end
        plot!(p, vcat(m.R, m.R[1]), vcat(m.Z, m.Z[1]); color = :cyan, lw = 2.5, label = "")
        scatter!(p, [m.R_geo], [m.Z_geo]; marker = :diamond, ms = 6, color = :cyan, msc = :black, msw = 1, label = "")
    end
    if !isnan(snap.ψ_lcfs)
        # the O-point, an open circle drawn as a line: GR fills a transparent marker black
        r, θ = 0.75 * min(run.G.dR, run.G.dZ), range(0, 2π; length = 33)
        plot!(p, snap.R_axis .+ r .* cos.(θ), snap.Z_axis .+ r .* sin.(θ); color = :magenta, lw = 2, label = "")
    end
    isempty(xpoints[1]) || scatter!(p, xpoints...; marker = :xcross, ms = 6, msw = 2, color = :magenta, label = "")
    isnan(snap0.R_cen_by_Jϕ) || scatter!(p, [snap0.R_cen_by_Jϕ], [snap0.Z_cen_by_Jϕ]; marker = :+, ms = 7, msw = 2, color = :lime, label = "")
    plot!(p, vcat(run.wall.R, run.wall.R[1]), vcat(run.wall.Z, run.wall.Z[1]); color = :red, lw = 1.2, label = "")
    return p
end

# The colour range of log10 nₑ: from the seed level (the first snapshot's peak) to 1e18 m⁻³.
function frame_ne_range(run)
    ne0 = run.snaps2D[1].ne
    return (floor(log10(max(maximum(x for x in ne0 if isfinite(x)), 1.0))), 18.0)
end
nearest_snap0D(run, t) = (s0 = run.snaps0D; s0[argmin(abs.(s0.time_s .- t))])

# One frame: the panels of 2D snapshot k under a two-line heading (title; currents), each
# above its colour bar.
function fields_frame(run, k::Integer; title, inside = in_wall_mask(run), ne_range = frame_ne_range(run))
    snap = run.snaps2D[k]
    snap0 = nearest_snap0D(run, snap.time_s)
    amps(I) = abs(I) < 1.0e3 ? @sprintf("%.3g A", abs(I)) : @sprintf("%.2f kA", 1.0e-3abs(I))
    info = "|I_tor| " * amps(snap0.I_tor) * ", on closed lines " * (isnan(snap0.I_closed) ? "none" : amps(snap0.I_closed))
    run.controlled && (info *= @sprintf(", control a₁ %.2f mT", 1.0e3snap0.ctrl_a1))
    head = plot(; framestyle = :none, xlims = (0, 1), ylims = (0, 1), legend = false)
    annotate!(head, 0.5, 0.68, text(@sprintf("%s, t = %.2f ms", title, 1.0e3snap.time_s), 11))
    annotate!(head, 0.5, 0.2, text(info * "   " * MARKER_KEY, 10))
    xpoints, lcfs = find_x_points(run, snap.ψ; inside), snapshot_lcfs(run, snap)
    scales = [panel_scale(snap, kind; inside, ne_range) for kind in FRAME_PANELS]
    panels = [flux_panel(run, snap, snap0, kind; inside, ne_range, xpoints, lcfs, scale, short = true) for (kind, scale) in zip(FRAME_PANELS, scales)]
    bars = [colorbar_strip(scale, kind) for (kind, scale) in zip(FRAME_PANELS, scales)]
    n = length(FRAME_PANELS)
    # panels and bars in one grid, so that each bar spans its panel's plot area
    return plot(head, panels..., bars...; layout = @layout([a{0.08h}; grid(2, n, heights = [0.955, 0.045])]), size = (300n, 720))
end

# mp4 of every 2D snapshot as a `fields_frame`.
function animate_fields(run; file, title, fps = 15)
    inside, ne_range = in_wall_mask(run), frame_ne_range(run)
    anim = @animate for k in eachindex(run.snaps2D)
        fields_frame(run, k; title, inside, ne_range)
    end
    mp4(anim, file; fps, loop = 0)
    return file
end

# The panels (rows) at the 2D snapshots nearest to `times_ms` (columns).
function plot_fields_snapshots(run, times_ms; file, title)
    s2 = run.snaps2D
    t_ms = [s.time_s for s in s2] .* 1.0e3
    idx = unique(argmin(abs.(t_ms .- t)) for t in times_ms)
    inside, ne_range = in_wall_mask(run), frame_ne_range(run)
    panels = [
        flux_panel(run, s2[k], nearest_snap0D(run, s2[k].time_s), kind; inside, ne_range, prefix = @sprintf("t = %.2f ms\n", t_ms[k]))
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

# ── a run's figures ─────────────────────────────────────────────────────────────────
configuration_name(code) = code == 0 ? "–" : string(RAPID2D.LCFS_CONFIGURATIONS[code])
first_closed_time(run) = (k = findfirst(x -> !isnan(x.ψ_lcfs), run.snaps2D); k === nothing ? NaN : run.snaps2D[k].time_s)

# The current column at `times_ms`, with the magnetic LCFS the 2D snapshots record: its
# configuration and geometric centre R_geo (the Shafranov shift's reference), and its area
# centroid R_area.
function print_column_table(run; times_ms)
    t2 = [x.time_s for x in run.snaps2D]
    println("  t     |I_tor|   closed   R_cen   R_axis   R_geo  R_area  R_axis−R_geo  LCFS          a₁      ⟨Tₑ⟩")
    for tms in times_ms
        x, y = nearest_snap0D(run, tms * 1.0e-3), run.snaps2D[argmin(abs.(t2 .- tms * 1.0e-3))]
        m = snapshot_lcfs(run, y).magnetic
        R_area = m === nothing ? NaN : m.R_area
        @printf(
            "%5.1f  %6.2f kA %5.2f kA  %.3f m  %.3f m  %.3f  %.3f   %+6.3f m   %-12s  %5.2f mT  %5.2f eV\n", 1.0e3x.time_s,
            1.0e-3abs(x.I_tor), 1.0e-3abs(x.I_closed), x.R_cen_by_Jϕ, x.R_axis, y.R_geo, R_area, x.R_axis - y.R_geo,
            configuration_name(y.lcfs_configuration), 1.0e3x.ctrl_a1, x.Te_eV
        )
    end
    return nothing
end

# |I| and the current on closed lines, the control's a₁, the centroid, axis and LCFS centre, ⟨Tₑ⟩.
function plot_column(run; file, title)
    s0, s2 = run.snaps0D, run.snaps2D
    t, t2 = s0.time_s .* 1.0e3, [x.time_s for x in s2] .* 1.0e3
    p1 = plot(t, 1.0e-3abs.(s0.I_tor); lw = 2, label = "I_tor", ylabel = "|I| (kA)", legend = :topleft)
    plot!(p1, t, 1.0e-3abs.(s0.I_closed); lw = 2, label = "on closed lines")
    p2 = plot(t, 1.0e3s0.ctrl_a1; lw = 2, ylabel = "control a₁ = B_Z (mT)", legend = false)
    p3 = plot(t, s0.R_cen_by_Jϕ; lw = 2, label = "current centroid", ylabel = "R (m)", legend = :bottomright)
    plot!(p3, t, s0.R_axis; lw = 2, label = "magnetic axis")
    plot!(p3, t2, [x.R_geo for x in s2]; lw = 2, label = "magnetic LCFS, geometric centre")
    p4 = plot(t, s0.Te_eV; lw = 2, ylabel = "⟨Tₑ⟩ (eV)", xlabel = "t (ms)", legend = false)
    fig = plot(p1, p2, p3, p4; layout = (4, 1), size = (700, 1000), left_margin = 8Plots.mm, plot_title = title, plot_titlefontsize = 10)
    savefig(fig, file)
    return fig
end

# Every figure of a run, from its files, into its folder: the column table (printed) and figure,
# the 0D traces and dashboard, the snapshot grid and the movie. By default the table runs over
# twelve times and the grid over seven, from the first closed region (or the start) to the end.
function plot_run_figures(run; title, table_times_ms = nothing, snapshot_times_ms = nothing, fps = 15)
    t_end = 1.0e3 * run.snaps2D[end].time_s
    t_from = isnan(first_closed_time(run)) ? 0.0 : 1.0e3 * first_closed_time(run)
    print_column_table(run; times_ms = something(table_times_ms, range(t_from, t_end; length = 12)))
    plot_column(run; file = joinpath(run.dir, "column.png"), title)
    plot_traces(run; file = joinpath(run.dir, "traces.png"))
    plot_dashboard(run; file = joinpath(run.dir, "dashboard.png"))
    plot_fields_snapshots(run, something(snapshot_times_ms, range(t_from, t_end; length = 7)); file = joinpath(run.dir, "fields_snapshots.png"), title)
    animate_fields(run; file = joinpath(run.dir, "fields.mp4"), title, fps)
    println("figures in ", run.dir)
    return nothing
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
