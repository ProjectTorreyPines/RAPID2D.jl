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
        @time run_simulation!(RP; kw...)
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
    vals = [getfield(s, field) for s in RP.diagnostics.snaps2D]
    if field in (:ne, :ni)
        top = log10(max(maximum(maximum, vals), 10.0))
        return (max(top - 8, 0.0), top)
    end
    lo, hi = minimum(minimum, vals), maximum(maximum, vals)
    return hi > lo ? (lo, hi) : (lo, lo + 1.0)
end

# One field of one snapshot: heatmap, wall outline, ψ contours (= field lines) when ψ varies.
function panel2D(RP::RAPID, snap, field::Symbol; clims, title = string(field))
    G = RP.G
    v = getfield(snap, field)
    if field in (:ne, :ni)
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
