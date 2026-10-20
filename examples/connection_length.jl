# Connection lengths of start-up fields with a null.
#
# Before breakdown the poloidal field of a start-up scenario has a null, where B_pol vanishes, and
# the field lines near it run many times around the torus before they reach the wall. The
# connection length Lc = ∫ (B/B_pol) dl from wall to wall is the path an electron avalanche grows
# along. This draws Lc on the grid for the vacuum fields of several devices and scenarios, traced
# three ways: marching squares (the default), the bicubic tracer, and RK4 along the interpolated B.
# For each field: |B_pol| with the flux and its nulls, Lc by each tracer, and how far marching
# squares and RK4 are from the bicubic tracer. It prints Lc near each null (the median within
# 10 cm), the median Lc inside the wall, and the time of one field-line analysis.
#
# The field inputs are not shipped: set RAPID_INPUT_PATH to the directory holding the device
# folders (KSTAR/Reference_201x201/, PPCF2/SingleQuad_LV=+5/, …) and their *_First_Wall.dat.
#
#   RAPID_INPUT_PATH=/path/to/Input julia --project=examples examples/connection_length.jl

include("common.jl")
isempty(INPUT_PATH) && error("set RAPID_INPUT_PATH to the directory holding the device folders and *_First_Wall.dat")
using RAPID2D: FieldLineGeometry, FieldLineFollowingResult, field_line_analysis!, update_prescribed_fields!,
    combine_field_sources!, MarchingSquaresTracer, CubicTracer, RK4Tracer, gradient, hessian

name = "connection_length"
out = output_dir(name)

tracers = (MarchingSquaresTracer(), CubicTracer(), RK4Tracer())
label(t) = t isa MarchingSquaresTracer ? "marching squares" : t isa CubicTracer ? "bicubic" : "RK4"

cases = (
    (device = "KSTAR", shot = "Reference_201x201", R0B0 = -2.7 * 1.8, times_ms = (0.0, 20.0, 40.0)),
    (device = "KSTAR", shot = "FNC_12393", R0B0 = -2.7 * 1.8, times_ms = (0.0,)),
    (device = "PPCF2", shot = "SingleQuad_LV=+5", R0B0 = 3.0, times_ms = (0.0,)),
    (device = "PPCF2", shot = "SingleHexa_LV=+5", R0B0 = 3.0, times_ms = (0.0,)),
)

# The nulls: Newton on ∇ψ = 0 for the bicubic ψ from every in-wall node where |B_pol| is least among
# its neighbours and well below its median, merged within two cells.
function find_nulls(RP, g)
    G, Bp = RP.G, RP.fields.Bpol
    nids = G.nodes.in_wall_nids
    level = 0.3 * median(Bp[nids])
    nulls = NTuple{2, Float64}[]
    for k in nids
        i, j = G.nodes.rid[k], G.nodes.zid[k]
        Bp[k] < level && all(Bp[k] <= Bp[i + a, j + b] for a in -1:1, b in -1:1 if 1 <= i + a <= G.NR && 1 <= j + b <= G.NZ) || continue
        x = (G.R2D[k], G.Z2D[k])
        for _ in 1:30
            gr, H = gradient(g.itp, x), hessian(g.itp, x)
            δ = [H[1, 1] H[1, 2]; H[2, 1] H[2, 2]] \ [gr[1], gr[2]]
            x = (x[1] - δ[1], x[2] - δ[2])
        end
        hypot(x[1] - G.R2D[k], x[2] - G.Z2D[k]) < 2 * G.dR || continue        # Newton stayed near the node
        any(y -> hypot(x[1] - y[1], x[2] - y[2]) < 2 * G.dR, nulls) || push!(nulls, x)
    end
    return nulls
end

maps = Any[]
for c in cases
    config = SimulationConfig{Float64}(
        Input_path = INPUT_PATH, device_Name = c.device, shot_Name = c.shot, NR = 30, NZ = 50, R0B0 = c.R0B0,
        prefilled_gas_pressure = 2.0e-3, dt = 1.0e-5, t_end_s = 60.0e-3, Output_path = joinpath(out, "run"),
    )
    RP = setup(config)
    G = RP.G
    nids = G.nodes.in_wall_nids
    for t_ms in c.times_ms
        update_prescribed_fields!(RP, 1.0e-3 * t_ms)
        combine_field_sources!(RP)
        g = FieldLineGeometry(RP; nodal_B = true)
        flf = map(tracers) do t
            f = field_line_analysis!(FieldLineFollowingResult{Float64}(G.NR, G.NZ), g, t)
            time = minimum(@elapsed(field_line_analysis!(FieldLineFollowingResult{Float64}(G.NR, G.NZ), g, t)) for _ in 1:3)
            f, time
        end
        nulls = find_nulls(RP, g)
        title = "$(c.device) $(c.shot), $(t_ms) ms"
        @printf(
            "%s: %d in-wall nodes, %d null%s at %s\n", title, length(nids), length(nulls), length(nulls) == 1 ? "" : "s",
            join((@sprintf("(%.2f, %.2f)", x...) for x in nulls), ", ")
        )
        ref = flf[2][1]
        # Lc near a null: the median over the nodes within 10 cm of it (a single node on a separatrix
        # leg is ill-conditioned)
        near(x) = [n for n in nids if hypot(G.R2D[n] - x[1], G.Z2D[n] - x[2]) < 0.1]
        for (t, (f, time)) in zip(tracers, flf)
            rel = [abs(f.Lc_tot[n] / ref.Lc_tot[n] - 1) for n in nids if isfinite(ref.Lc_tot[n])]
            @printf(
                "  %-16s %6.2f ms | Lc near the null%s %s m | median %6.0f m, max %7.0f m | closed nodes %d | vs bicubic: median %.1e, 95th pct %.1e\n",
                label(t), 1.0e3 * time, length(nulls) == 1 ? "" : "s", join((@sprintf("%6.0f", median(f.Lc_tot[near(x)])) for x in nulls), " /"),
                median(f.Lc_tot[nids]), maximum(f.Lc_tot[nids]), length(intersect(f.closed_surface_nids, nids)), median(rel), quantile(rel, 0.95)
            )
        end

        inside(A) = (M = fill(NaN, size(A)); M[nids] .= A[nids]; M)
        frame(t; kw...) = heatmap(G.R1D, G.Z1D, transpose(t); aspect_ratio = :equal, xlabel = "R (m)", ylabel = "Z (m)", xlims = extrema(G.R1D), ylims = extrema(G.Z1D), kw...)
        wall!(p) = (plot!(p, RP.wall.R, RP.wall.Z; color = :white, lw = 1.5, label = ""); scatter!(p, first.(nulls), last.(nulls); marker = :star5, ms = 8, color = :cyan, label = ""))
        panels = Any[]
        p = frame(inside(log10.(RP.fields.Bpol .* 1.0e4)); c = :viridis, title = "$title\nlog10 |B_pol| (G), ψ, nulls")
        contour!(p, G.R1D, G.Z1D, transpose(RP.fields.ψ); levels = 20, color = :white, lw = 0.6, colorbar_entry = false)
        wall!(p)
        push!(panels, p)
        Lc_max = log10(maximum(filter(isfinite, ref.Lc_tot[nids])))
        for (t, (f, _)) in zip(tracers, flf)
            p = frame(inside(log10.(f.Lc_tot)); c = :inferno, clims = (0, Lc_max), title = "log10 Lc (m), $(label(t))")
            wall!(p)
            push!(panels, p)
        end
        for i in (1, 3)
            err = inside(log10.(abs.(flf[i][1].Lc_tot ./ ref.Lc_tot .- 1) .+ 1.0e-6))
            p = frame(err; c = :magma, clims = (-6, 0), title = "log10 |Lc / Lc bicubic − 1|, $(label(tracers[i]))")
            wall!(p)
            push!(panels, p)
        end
        file = joinpath(out, "$(c.device)_$(c.shot)_$(round(Int, t_ms))ms.png")
        savefig(plot(panels...; layout = (2, 3), size = (1500, 1150), margin = 4Plots.mm), file)
        p = frame(inside(log10.(flf[1][1].Lc_tot)); c = :inferno, clims = (0, 4), title = "$title\nlog10 Lc (m)")
        wall!(p)
        push!(maps, p)
    end
end
savefig(plot(maps...; layout = (2, cld(length(maps), 2)), size = (1500, 1150), margin = 4Plots.mm), joinpath(out, "connection_length_all.png"))
println("outputs in ", out)
