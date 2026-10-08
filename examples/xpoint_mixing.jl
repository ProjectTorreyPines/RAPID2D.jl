# Mixing of u∥ and Tₑ along field lines, isolated: no E field, no collisions, no sources, a
# reflective wall, and the bulk electron tensor prescribed along the poloidal field line. A
# diffusive particle flux carries momentum and energy, so each line settles to the n-weighted
# means of what its particles started with, and the kinetic energy of the sheared flow it
# erased reappears as heat. The operator RAPID2D used before this work diffused u∥ and Tₑ as
# fields and settled to volume means, losing both; its numbers are recorded in
# internal/docs notes/examples-results.md (2026-10-07).
#
# Three scenarios: a straight vertical field, where each R column is one line and the
# prediction is exact per column; an analytic X-point, where the prediction is per flux tube
# and the 9-point stencil also leaks across the lines; and a blob of plasma beside the X-point,
# which should fill the lines through it and nothing else.
#
# Outputs (examples/output/xpoint_mixing/): the summary figure with the verdict, the time
# traces of the particles' momentum and energy, 2-D frames of u∥, Tₑ and n in time, and mp4s.
# reference/electron-diffusive-transport.md; notes/design/turbulent-mixing-u-T.md §6.

include("common.jl")

const me, ee = 9.1093837015e-31, 1.602176634e-19
const XP = (R0 = 1.5, Z0 = 0.0, Bprime = 0.02)
const D, u0, T0 = 500.0, 2.0e6, 10.0

# ── the fixtures: the X-point field and the aligned tensor ─────────────────────────────────
# The hyperbolic field ψ = (B′/2)(x² − y²), R B_R = B′ y, R B_Z = B′ x, written as the external
# field of a set-up run (Ampère off): the step recombines B from it. The derived quantities and
# the field-line analysis are redone, as the setup did them on its uniform field.
function xpoint_field!(RP; R0, Z0, Bprime)
    G, F = RP.G, RP.fields
    x = G.R2D .- R0
    y = G.Z2D .- Z0
    F.BR_ext .= Bprime .* y ./ G.R2D
    F.BZ_ext .= Bprime .* x ./ G.R2D
    F.ψ_ext .= Bprime / 2 .* (x .^ 2 .- y .^ 2)
    RAPID2D.combine_external_and_self_fields!(RP)
    RAPID2D.flf_analysis_field_lines_rz_plane!(RP)
    return RP
end

# D_across 𝟙 + (D_along − D_across) b_pol b_polᵀ on every node, frozen for the run: the per-step
# refresh keeps it and recomputes only its coefficient tensor, so neither Bohm nor D∥ enters.
function prescribe_aligned_tensor!(RP; D_along, D_across = 0.0)
    F, tp = RP.fields, RP.transport
    @. tp.DRR = D_across + (D_along - D_across) * F.bpol_R^2
    @. tp.DRZ = (D_along - D_across) * F.bpol_R * F.bpol_Z
    @. tp.DZZ = D_across + (D_along - D_across) * F.bpol_Z^2
    RP.flags.freeze_diffusion_tensor = true
    RAPID2D.update_transport_quantities!(RP)
    return RP
end

# ── the pure-mixing run ────────────────────────────────────────────────────────────────
function mixing_RP(name; N, t_end_s, xpoint::Bool, dt = 1.0e-6, nsnap = 10)
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = N, NZ = N, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.2, 1.8, 1.8, 1.2], wall_Z = [-0.3, -0.3, 0.3, 0.3],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = dt, t_end_s = t_end_s,
        snap0D_Δt_s = t_end_s / nsnap, snap2D_Δt_s = t_end_s / nsnap, electron_wall_albedo = 1.0,
        Output_path = output_dir("xpoint_mixing/$name"),
    )
    config.manual.Eϕ = 0.0
    RP = setup(
        config;
        Implicit = true, diffu = true, convec = false, src = false,
        Atomic_Collision = false, Coulomb_Collision = false,
        mean_ExB = false, turb_ExB_mixing = false, E_para_self_ES = false, E_para_self_EM = false,
        Ampere = false, Te_evolve = true, ud_evolve = true, Ti_evolve = false, Gas_evolve = false,
        update_ni_independently = false, secondary_electron = false, negative_n_correction = false,
        Include_ud_pressure_term = false, Include_ud_convec_term = false,
        Include_Te_convec_term = false, Include_heat_flux_term = false,
    )
    xpoint && xpoint_field!(RP; XP...)
    prescribe_aligned_tensor!(RP; D_along = D)
    return RP
end

# the momentum equation reads the stored collision rates whatever the flags say: zero them
function no_friction!(RP)
    fill!(RP.plasma.ν_en_mom_tot, 0.0)
    fill!(RP.plasma.ν_en_iz_tot, 0.0)
    fill!(RP.plasma.ν_ei_eff, 0.0)
    return nothing
end

# ── what the traces record: the particles' momentum and energy, ⟨Tₑ⟩, the spread along lines ──
Base.@kwdef struct MixingRecord
    t::Vector{Float64} = Float64[]
    P::Vector{Float64} = Float64[]        # Σ V n u∥
    E::Vector{Float64} = Float64[]        # Σ V n (3/2 e Tₑ + ½ mₑ u∥²)
    T::Vector{Float64} = Float64[]        # particle-weighted ⟨Tₑ⟩
    spread::Vector{Float64} = Float64[]   # rms deviation of u∥ from each line's own mean, averaged over lines
end

wmean(nodes, V, n, f) = sum(V[nodes] .* n[nodes] .* f[nodes]) / sum(V[nodes] .* n[nodes])
function line_spread(lines, V, n, u)
    s = 0.0
    for nodes in lines
        m = wmean(nodes, V, n, u)
        w = V[nodes] .* n[nodes]
        s += sqrt(sum(w .* (u[nodes] .- m) .^ 2) / sum(w))
    end
    return s / length(lines)
end
function record!(rec::MixingRecord, RP, lines)
    inw = RP.G.nodes.in_wall_nids
    V, n, u, T = vec(RP.G.inVol2D), vec(RP.plasma.ne), vec(RP.plasma.ue_para), vec(RP.plasma.Te_eV)
    push!(rec.t, RP.time_s)
    push!(rec.P, sum(V[inw] .* n[inw] .* u[inw]))
    push!(rec.E, sum(V[inw] .* n[inw] .* (1.5 .* ee .* T[inw] .+ 0.5 .* me .* u[inw] .^ 2)))
    push!(rec.T, wmean(inw, V, n, T))
    push!(rec.spread, line_spread(lines, V, n, u))
    return nothing
end

function run_mixing!(RP, n, u, lines)
    inw = RP.G.nodes.in_wall_nids
    RP.plasma.ne .= 0.0
    RP.plasma.ne[inw] .= n[inw]
    RP.plasma.ue_para .= u
    RP.plasma.Te_eV .= T0
    rec = MixingRecord()
    record!(rec, RP, lines)
    run!(RP; callback_before_step = no_friction!, callback_after_step = rp -> record!(rec, rp, lines))
    return rec
end

# what a set of nodes mixed by its own particles settles to: the particle-weighted mean
# velocity and the temperature that conserves the energy per particle
function mixed_state(nodes, V, n, u)
    w = V[nodes] .* n[nodes]
    W = sum(w)
    u∞ = sum(w .* u[nodes]) / W
    ε = sum(w .* (1.5 .* ee .* T0 .+ 0.5 .* me .* u[nodes] .^ 2)) / W
    return (u = u∞, T = (ε - 0.5 * me * u∞^2) / (1.5 * ee))
end

out = output_dir("xpoint_mixing")

# ── 1. straight lines: each R column is one line ───────────────────────────────────────
L = 0.6                                       # the column between the walls
τ = L^2 / (π^2 * D)                           # the slowest mode's decay time
function wall_columns(G)
    cols = Dict{Int, Vector{Int}}()
    for nid in G.nodes.in_wall_nids
        push!(get!(cols, G.nodes.rid[nid], Int[]), nid)
    end
    return cols
end
RPs = mixing_RP("straight"; N = 41, t_end_s = 6τ, xpoint = false)
G = RPs.G
shape = @. 1 + 0.5 * cos(2π * G.Z2D / L)       # n and u correlated along the line, one period
n_s0 = 1.0e14 .* vec(shape)
u_s0 = vec(u0 .* shape)
rec_s = run_mixing!(RPs, n_s0, u0 .* shape, collect(values(wall_columns(G))))
V = vec(G.inVol2D)
i_mid = G.nodes.rid[G.nodes.in_wall_nids[length(G.nodes.in_wall_nids) ÷ 2]]
col = wall_columns(G)[i_mid]
Z = vec(G.Z2D)[col]
pred = mixed_state(col, V, n_s0, u_s0)
u_s = vec(RPs.plasma.ue_para)[col]
T_s = vec(RPs.plasma.Te_eV)[col]
straight_ok = all(x -> isapprox(x, pred.u; rtol = 1.0e-2), u_s) && all(x -> isapprox(x, pred.T; rtol = 1.0e-2), T_s)
@printf("straight column: predicted u∞ %.4g m/s, T∞ %.3f eV; the run ends at %.4g m/s, %.3f eV\n", pred.u, pred.T, mean(u_s), mean(T_s))

# ── 2. the X-point: flux tubes over the inner half of ψ ─────────────────────────────────
function tubes_of(G; n_bins = 3)
    x = vec(G.R2D) .- XP.R0
    y = vec(G.Z2D) .- XP.Z0
    ψ = XP.Bprime / 2 .* (x .^ 2 .- y .^ 2)
    ψ_cut = XP.Bprime / 2 * (3 * G.dR)^2
    edges = range(ψ_cut, XP.Bprime / 2 * 0.3^2 / 2; length = n_bins + 1)
    tubes = Dict{Tuple{Int, Int}, Vector{Int}}()
    for nid in G.nodes.in_wall_nids
        a = abs(ψ[nid])
        edges[1] <= a < edges[end] || continue
        branch = ψ[nid] > 0 ? sign(x[nid]) : sign(y[nid])
        branch == 0 && continue
        push!(get!(tubes, (searchsortedlast(edges, a), Int(branch)), Int[]), nid)
    end
    return tubes, x .* y
end
t_x = 2.5e-4                                  # ≈ two crossing times of the longest inner line
RPx = mixing_RP("xpoint"; N = 49, t_end_s = t_x, xpoint = true)
Gx = RPx.G
tubes, χ = tubes_of(Gx)
along = @. 1 + 0.5 * cos(2π * χ / 0.09)       # two oscillations along each line
n_x0 = 1.0e14 .* along
u_x0 = u0 .* along
rec_x = run_mixing!(RPx, n_x0, reshape(u_x0, Gx.NR, Gx.NZ), collect(values(tubes)))
Vx = vec(Gx.inVol2D)
dP = rec_x.P[end] / rec_x.P[1] - 1
dE = rec_x.E[end] / rec_x.E[1] - 1
@printf("X-point: particle momentum %+.2f %%, energy %+.2f %%\n", 100 * dP, 100 * dE)
xpoint_ok = abs(dP) < 1.0e-2 && abs(dE) < 1.0e-2

# ── 3. a blob beside the X-point: the particles fill the lines through it ──────────────────
# n is a Gaussian at (x, y) = (0.17, 0) m from the null, σ = 6 cm (three cells on the 49² grid:
# a blob the 9-point cross-term stencil cannot resolve undershoots below zero at its edge), on
# a 1e12 background, two σ from the separatrix and from the wall; it carries u∥ = u0 (the
# background is at rest) and a uniform Tₑ. The lines through the blob are the ψ > 0, x > 0
# branch within |ψ − ψ_blob| of two σ; what ends up outside that band and branch is the
# stencil's leakage across the lines. At the blob's edge the same cross terms cool a few
# low-density nodes to the Tₑ floor for a few steps (the dissipation rate's wrong-signed
# entries, a hundred times amplified by the density contrast) before the mixing restores them.
const BLOB = (x = 0.17, y = 0.0, σ = 0.06, n_bg = 1.0e12)
function blob_band(G)
    x = vec(G.R2D) .- XP.R0
    y = vec(G.Z2D) .- XP.Z0
    ψ = XP.Bprime / 2 .* (x .^ 2 .- y .^ 2)
    ψ_b = XP.Bprime / 2 * (BLOB.x^2 - BLOB.y^2)
    # the ψ half-width the blob covers: dψ = B′ (x dx − y dy) over ±2σ in both directions
    half = 2 * XP.Bprime * BLOB.σ * (abs(BLOB.x) + abs(BLOB.y) + BLOB.σ)
    inband = [abs(ψ[k] - ψ_b) <= half && x[k] > 0 && ψ[k] > 0 for k in eachindex(ψ)]
    return inband, ψ_b, half
end
t_b = 2.5e-4
RPb = mixing_RP("blob"; N = 49, t_end_s = t_b, xpoint = true)
Gb = RPb.G
inw_b = Gb.nodes.in_wall_nids
xb = vec(Gb.R2D) .- XP.R0
yb = vec(Gb.Z2D) .- XP.Z0
gauss = @. exp(-((xb - BLOB.x)^2 + (yb - BLOB.y)^2) / (2 * BLOB.σ^2))
n_b0 = BLOB.n_bg .+ 1.0e14 .* gauss
u_b0 = reshape(u0 .* (1.0e14 .* gauss) ./ n_b0, Gb.NR, Gb.NZ)     # the blob moves, the background does not
inband, ψ_b, half = blob_band(Gb)
Vb = vec(Gb.inVol2D)
kept = Float64[]
record_band!(rp) = push!(kept, sum(Vb[inw_b] .* vec(rp.plasma.ne)[inw_b] .* inband[inw_b]) / sum(Vb[inw_b] .* vec(rp.plasma.ne)[inw_b]))
RPb.plasma.ne .= 0.0
RPb.plasma.ne[inw_b] .= n_b0[inw_b]
RPb.plasma.ue_para .= u_b0
RPb.plasma.Te_eV .= T0
record_band!(RPb)
rec_b = MixingRecord()
record!(rec_b, RPb, [inw_b])
run!(RPb; callback_before_step = no_friction!, callback_after_step = rp -> (record!(rec_b, rp, [inw_b]); record_band!(rp)))
@printf(
    "blob: particles kept in the blob's band and branch %.1f %% → %.1f %%; particle momentum %+.2f %%\n",
    100 * kept[1], 100 * kept[end], 100 * (rec_b.P[end] / rec_b.P[1] - 1)
)

# ── figures ────────────────────────────────────────────────────────────────────────────
# the summary, with the verdict: one column, the X-point maps, the tubes
p1 = plot(xlabel = "Z (m)", ylabel = "u∥ (m/s)", title = @sprintf("one column, R = %.2f m, t = 6τ", G.R1D[i_mid]), legend = :topright)
plot!(p1, Z, u_s0[col]; lw = 2, c = :gray, label = "initial")
plot!(p1, Z, u_s; lw = 2, label = "final")
hline!(p1, [pred.u]; ls = :dash, c = :black, label = "particle-weighted mean")
p2 = plot(xlabel = "Z (m)", ylabel = "Tₑ (eV)", legend = :topright)
plot!(p2, Z, T_s; lw = 2, label = "final")
hline!(p2, [pred.T]; ls = :dash, c = :black, label = "energy-conserving Tₑ")
ψx = XP.Bprime / 2 .* ((Gx.R2D .- XP.R0) .^ 2 .- (Gx.Z2D .- XP.Z0) .^ 2)
function umap(RP, u, title)
    p = heatmap(RP.G.R1D, RP.G.Z1D, u' ./ 1.0e6; xlabel = "R (m)", ylabel = "Z (m)", title, c = :viridis, clims = (0.8, 3.2), aspect_ratio = :equal, colorbar_title = "u∥ (10⁶ m/s)")
    contour!(p, RP.G.R1D, RP.G.Z1D, ψx'; levels = 12, c = :white, lw = 0.5, colorbar_entry = false)
    plot!(p, RP.wall.R, RP.wall.Z; c = :black, lw = 2, label = false)
    return p
end
m1 = umap(RPx, reshape(u_x0, Gx.NR, Gx.NZ), "u∥ initial")
m2 = umap(RPx, RPx.plasma.ue_para, @sprintf("u∥ at t = %.2f ms", 1.0e3 * t_x))
keys_sorted = sort(collect(keys(tubes)))
p4 = plot(xlabel = "flux tube (ψ bin, branch)", ylabel = "u∥ (10⁶ m/s)", legend = :topright, xticks = (1:length(keys_sorted), string.(keys_sorted)), xrotation = 30)
vals = [wmean(tubes[k], Vx, vec(RPx.plasma.ne), vec(RPx.plasma.ue_para)) for k in keys_sorted] ./ 1.0e6
scatter!(p4, 1:length(keys_sorted), vals; ms = 6, label = "final tube mean")
preds = [mixed_state(tubes[k], Vx, n_x0, u_x0).u for k in keys_sorted] ./ 1.0e6
scatter!(p4, 1:length(keys_sorted), preds; m = :x, ms = 8, c = :black, label = "particle-weighted prediction")
fig = plot(p1, p2, m1, m2, p4; layout = (3, 2), size = (1200, 1500), left_margin = 8Plots.mm)
save_with_verdict(
    fig, out, "xpoint_mixing", straight_ok && xpoint_ok,
    @sprintf(
        "a straight column should settle within 1 %% of its particle-weighted mean and energy-conserving temperature (it ends at %.4g m/s, %.3f eV against %.4g m/s, %.3f eV), and on the X-point the particle momentum and energy should be kept within 1 %% (they move by %+.2f %% and %+.2f %%)",
        mean(u_s), mean(T_s), pred.u, pred.T, 100 * dP, 100 * dE
    ),
)

# the traces: what the particles carry, and how fast the lines homogenize
q1 = plot(ylabel = "Σ V n u∥ / initial − 1 (%)", title = "X-point: the particles' momentum", legend = :bottomleft)
q2 = plot(ylabel = "Σ V n ε̄ / initial − 1 (%)", title = "X-point: the particles' energy", legend = :bottomleft)
q3 = plot(ylabel = "⟨Tₑ⟩ (eV)", xlabel = "t (ms)", title = "heating by the erased shear", legend = :bottomright)
q4 = plot(ylabel = "spread of u∥ along a line (m/s)", xlabel = "t (ms)", title = "homogenization along the lines", yscale = :log10, legend = :topright)
t_xm = rec_x.t .* 1.0e3
plot!(q1, t_xm, 100 .* (rec_x.P ./ rec_x.P[1] .- 1); lw = 2, label = "X-point")
plot!(q2, t_xm, 100 .* (rec_x.E ./ rec_x.E[1] .- 1); lw = 2, label = "X-point")
plot!(q3, t_xm, rec_x.T; lw = 2, label = "X-point")
plot!(q4, t_xm, max.(rec_x.spread, 1.0); lw = 2, label = "X-point")
t_sm = rec_s.t .* 1.0e3
plot!(q3, t_sm, rec_s.T; lw = 2, ls = :dash, label = "straight")
plot!(q4, t_sm, max.(rec_s.spread, 1.0); lw = 2, ls = :dash, label = "straight")
savefig(plot(q1, q2, q3, q4; layout = (2, 2), size = (1100, 800), left_margin = 8Plots.mm), joinpath(out, "traces.png"))

# 2-D frames in time: one row per run, columns = times, one colour range per field, with the
# field lines (ψ contours) drawn on the X-point frames
function frame(RP, snap, field, title; clims, contours, linear = false)
    F = getfield(snap, field)
    Fp = field === :ue_para ? F ./ 1.0e6 : field === :ne ? (linear ? F ./ 1.0e13 : log10.(max.(F, 1.0))) : F
    p = heatmap(RP.G.R1D, RP.G.Z1D, Fp'; xlabel = "R (m)", ylabel = "Z (m)", title, c = :viridis, clims, aspect_ratio = :equal, colorbar = false, titlefontsize = 9)
    contours && contour!(p, RP.G.R1D, RP.G.Z1D, ψx'; levels = 16, c = :white, lw = 0.8, alpha = 0.6, colorbar_entry = false)
    plot!(p, RP.wall.R, RP.wall.Z; c = :black, lw = 1.5, label = false)
    return p
end
function frames_figure(runs, field, times_ms; clims, contours, file, unit, linear = false)
    panels = []
    for (label, RP) in runs
        s2 = RP.diagnostics.snaps2D
        t_ms = [s.time_s for s in s2] .* 1.0e3
        for (j, t) in enumerate(times_ms)
            k = argmin(abs.(t_ms .- t))
            title = j == 1 ? @sprintf("%s\n%s  %.2f ms", label, unit, t_ms[k]) : @sprintf("%.2f ms", t_ms[k])
            push!(panels, frame(RP, s2[k], field, title; clims, contours, linear))
        end
    end
    n = length(times_ms)
    savefig(plot(panels...; layout = (length(runs), n), size = (300 * n, 330 * length(runs)), margin = 3Plots.mm), file)
    return nothing
end
xrun = ["X-point" => RPx]
srun = ["straight" => RPs]
brun = ["blob" => RPb]
frames_figure(xrun, :ue_para, collect(range(0, 1.0e3 * t_x; length = 5)); clims = (0.8, 3.2), contours = true, file = joinpath(out, "xpoint_ue_para_frames.png"), unit = "u∥ (10⁶ m/s)")
frames_figure(xrun, :Te_eV, collect(range(0, 1.0e3 * t_x; length = 5)); clims = (9.9, 11.0), contours = true, file = joinpath(out, "xpoint_Te_frames.png"), unit = "Tₑ (eV)")
frames_figure(srun, :ue_para, collect(range(0, 1.0e3 * 6τ; length = 5)); clims = (0.8, 3.2), contours = false, file = joinpath(out, "straight_ue_para_frames.png"), unit = "u∥ (10⁶ m/s)")
frames_figure(srun, :Te_eV, collect(range(0, 1.0e3 * 6τ; length = 5)); clims = (9.9, 11.0), contours = false, file = joinpath(out, "straight_Te_frames.png"), unit = "Tₑ (eV)")
frames_figure(brun, :ne, collect(range(0, 1.0e3 * t_b; length = 5)); clims = (10.0, 14.0), contours = true, file = joinpath(out, "blob_ne_frames.png"), unit = "log10 n")
frames_figure(brun, :ne, collect(range(0, 1.0e3 * t_b; length = 5)); clims = (0.0, 3.0), contours = true, file = joinpath(out, "blob_ne_linear_frames.png"), unit = "n (10¹³ m⁻³; the initial peak, 10, saturates)", linear = true)
frames_figure(brun, :ue_para, collect(range(0, 1.0e3 * t_b; length = 5)); clims = (0.0, 2.0), contours = true, file = joinpath(out, "blob_ue_para_frames.png"), unit = "u∥ (10⁶ m/s)")
# the blob: what stays on its own lines, and the momentum it carries
b1 = plot(xlabel = "t (ms)", ylabel = "particles in the blob's band and branch (%)", legend = :topright, title = "filling its own lines; the rest is leakage across them")
b2 = plot(xlabel = "t (ms)", ylabel = "Σ V n u∥ / initial − 1 (%)", legend = :bottomleft, title = "the blob's momentum")
t_bm = rec_b.t .* 1.0e3
plot!(b1, t_bm, 100 .* kept; lw = 2, label = "blob")
plot!(b2, t_bm, 100 .* (rec_b.P ./ rec_b.P[1] .- 1); lw = 2, label = "blob")
savefig(plot(b1, b2; layout = (1, 2), size = (1100, 420), left_margin = 8Plots.mm), joinpath(out, "blob_traces.png"))
animate2D(["X-point" => RPx], [:ue_para, :Te_eV]; file = joinpath(out, "xpoint_snaps2D.mp4"), fps = 4)
animate2D(["blob" => RPb], [:ne, :ue_para]; file = joinpath(out, "blob_snaps2D.mp4"), fps = 4)
println("outputs in ", out)
