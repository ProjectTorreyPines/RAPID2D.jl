# Field-line following on the (R, Z) grid: three ways to trace a field line.
#
# In axisymmetry the poloidal projection of a field line is a ψ contour. Following it from each node
# to the wall gives the lengths transport reads, the poloidal length Lpol = ∫ dl and the length along
# B, Lc = ∫ (B/B_pol) dl, and tells which nodes lie on closed surfaces. Three tracers:
#   MarchingSquaresTracer (the default): the bilinear contour, cell by cell;
#   CubicTracer: the bicubic level set, by predictor–corrector steps;
#   RK4Tracer: RK4 along the interpolated nodal B.
# Each is compared with the bicubic tracer at a fiftieth of a cell per step, on the same wall.
#
# Part 1 is a start-up-like field (a quadrupole null plus a current: an O-point between two
# X-points) inside a box. Part 2 follows the full start-up run (single-quadrupole field, every module
# on except Global_JxB_Force) as its closed region forms and grows.
#
#   julia --project=examples examples/field_line_tracing.jl

include("common.jl")
using RAPID2D: initialize_grid_geometry, FieldLineGeometry, FieldLineFollowingResult, field_line_analysis!,
    MarchingSquaresTracer, CubicTracer, RK4Tracer, needs_nodal_B

name = "field_line_tracing"
out = output_dir(name)
tracers = (MarchingSquaresTracer(), CubicTracer(), RK4Tracer())
label(t) = t isa MarchingSquaresTracer ? "marching squares" : t isa CubicTracer ? "bicubic" : "RK4"
reference = CubicTracer(max_step_cells = 0.02, max_turn = deg2rad(1), max_weight_change = 0.002)

function analysis(f, t)
    g = FieldLineGeometry(f.G, f.ψ; f.Bϕ, f.inside, nodal_B = needs_nodal_B(t), f.BR, f.BZ)
    run() = field_line_analysis!(FieldLineFollowingResult{Float64}(f.G.NR, f.G.NZ), g, t)
    flf = run()
    return flf, minimum(@elapsed(run()) for _ in 1:3)
end

# Each tracer against the reference on the nodes `f.nodes`; returns the reference and the analyses.
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
    masked(A) = [k in f.nodes ? A[k] : NaN for k in CartesianIndices(A)]
    panels = [heatmap(G.R1D, G.Z1D, transpose(masked(log10.(ref.Lc_tot))); aspect_ratio = :equal, title = "$title: log10 Lc_tot, reference", c = :viridis)]
    for (t, flf) in zip(tracers, results)
        t isa CubicTracer && continue
        err = masked(log10.(abs.(flf.Lc_tot ./ ref.Lc_tot .- 1) .+ 1.0e-6))
        push!(panels, heatmap(G.R1D, G.Z1D, transpose(err); aspect_ratio = :equal, clims = (-6, 0), title = "log10 |Lc_tot error|, $(label(t))", c = :magma))
    end
    return panels
end

# ── Part 1: a quadrupole null with a current inside a box ─────────────────────────────────
println("Part 1: quadrupole null plus a current (two X-points), 30 × 50")
G = initialize_grid_geometry(30, 50, (1.0, 2.0), (-1.0, 1.0))
ψ = @. 0.1 * ((G.R2D - 1.5)^2 - G.Z2D^2) - 0.02 * exp(-((G.R2D - 1.5113)^2 + (G.Z2D - 0.0071)^2) / 0.15^2)
part1 = (;
    G, ψ, Bϕ = 3.0 ./ G.R2D, inside = nothing, BR = nothing, BZ = nothing,
    nodes = [LinearIndices(ψ)[i, j] for i in 2:(G.NR - 1), j in 2:(G.NZ - 1)],
)
ref, results = compare(part1)
panels = maps(part1, ref, results, "analytic")

# ── Part 2: the full start-up run ─────────────────────────────────────────────────────────
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
for t in times_ms
    println("Part 2: full_startup at $t ms, $(length(fields[t].nodes)) in-wall nodes")
    ref, results = compare(fields[t])
    t == 10.0 && append!(panels, maps(fields[t], ref, results, "full_startup 10 ms"))
end
savefig(plot(panels...; layout = (2, 3), size = (1500, 1100), margin = 5Plots.mm), joinpath(out, "field_line_tracing.png"))
println("outputs in ", out)
