# Mixing of u∥ and Tₑ along field lines, with nothing else acting: no field, no sources, no
# collisions, a reflective wall, and the electron tensor prescribed along the poloidal field
# line (PrescribedTensor). A diffusive particle flux carries momentum and energy, so each
# field line settles to the particle-weighted means of what it started with, and the kinetic
# energy of the sheared flow it erased reappears as heat (ParticleMixing, the default). The
# reference policy VelocityDiffusion diffuses u∥ and Tₑ as fields and settles to volume means
# without heating.
#
# Two fields: straight vertical lines, where every R column is one line and the prediction is
# exact per column; and an analytic X-point (XPointPoloidal), where the prediction is per flux
# tube and the 9-point stencil also spreads along-line structure across the lines.
#
# The initial state correlates n and u along each line (what separates the particle-weighted
# mean from the volume mean) and keeps Tₑ uniform, so any change in Tₑ is the heating.
#
#   julia --project=examples examples/xpoint_mixing.jl
#
# Outputs, in examples/output/xpoint_mixing/: the verdict figure (xpoint_mixing__PASS/FAIL.png),
# the time traces (traces.png), 2-D frames of u∥ and Tₑ in time under both policies
# (*_frames.png), an mp4 of the X-point runs, and each run's snapshot files in its own folder.

include("common.jl")
using RAPID2D: PrescribedTensor, ParticleMixing, VelocityDiffusion, XPointPoloidal, UniformPoloidal

const me, ee = 9.1093837015e-31, 1.602176634e-19
const XP = (R0 = 1.5, Z0 = 0.0, Bprime = 0.02)
const D, u0, T0 = 500.0, 2.0e6, 10.0
const POLICIES = ["particle mixing" => ParticleMixing(), "velocity diffusion" => VelocityDiffusion()]

# ── the pure-mixing run ────────────────────────────────────────────────────────────────
function mixing_RP(name; N, t_end_s, poloidal, policy, dt = 1.0e-6, nsnap = 10)
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = N, NZ = N, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.2, 1.8, 1.8, 1.2], wall_Z = [-0.3, -0.3, 0.3, 0.3],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = dt, t_end_s = t_end_s,
        snap0D_Δt_s = t_end_s / nsnap, snap2D_Δt_s = t_end_s / nsnap, electron_wall_albedo = 1.0,
        Output_path = output_dir("xpoint_mixing/$name"),
    )
    config.manual.Eϕ = 0.0
    config.manual.poloidal = poloidal
    return setup(
        config;
        Implicit = true, diffu = true, convec = false, src = false,
        Atomic_Collision = false, Coulomb_Collision = false,
        mean_ExB = false, turb_ExB_mixing = false, E_para_self_ES = false, E_para_self_EM = false,
        Ampere = false, Te_evolve = true, ud_evolve = true, Ti_evolve = false, Gas_evolve = false,
        update_ni_independently = false, secondary_electron = false, negative_n_correction = false,
        Include_ud_pressure_term = false, Include_ud_convec_term = false,
        Include_Te_convec_term = false, Include_heat_flux_term = false,
        diffusion_tensor = PrescribedTensor(D_along = D, D_across = 0.0),
        mixing_policy = policy,
    )
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
straight = Dict{String, Any}()
for (label, policy) in POLICIES
    RP = mixing_RP("straight_" * replace(label, " " => "_"); N = 41, t_end_s = 6τ, poloidal = UniformPoloidal(), policy)
    G = RP.G
    shape = @. 1 + 0.5 * cos(2π * G.Z2D / L)   # n and u correlated along the line, one period
    n = 1.0e14 .* vec(shape)
    u = u0 .* shape
    rec = run_mixing!(RP, n, u, collect(values(wall_columns(G))))
    straight[label] = (RP = RP, n0 = n, u0 = vec(u), rec = rec)
end
RPs = straight["particle mixing"].RP
G = RPs.G
V = vec(G.inVol2D)
i_mid = G.nodes.rid[G.nodes.in_wall_nids[length(G.nodes.in_wall_nids) ÷ 2]]
col = wall_columns(G)[i_mid]
Z = vec(G.Z2D)[col]
pred = mixed_state(col, V, straight["particle mixing"].n0, straight["particle mixing"].u0)
u_s = vec(RPs.plasma.ue_para)[col]
T_s = vec(RPs.plasma.Te_eV)[col]
straight_ok = all(x -> isapprox(x, pred.u; rtol = 1.0e-2), u_s) && all(x -> isapprox(x, pred.T; rtol = 1.0e-2), T_s)
@printf(
    "straight column: predicted u∞ %.4g m/s, T∞ %.3f eV; particle mixing ends at %.4g m/s, %.3f eV; velocity diffusion at %.4g m/s, %.3f eV\n",
    pred.u, pred.T, mean(u_s), mean(T_s),
    mean(vec(straight["velocity diffusion"].RP.plasma.ue_para)[col]), mean(vec(straight["velocity diffusion"].RP.plasma.Te_eV)[col])
)

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
xpoint = Dict{String, Any}()
for (label, policy) in POLICIES
    RP = mixing_RP("xpoint_" * replace(label, " " => "_"); N = 49, t_end_s = t_x, poloidal = XPointPoloidal(; XP...), policy)
    G = RP.G
    tubes, χ = tubes_of(G)
    along = @. 1 + 0.5 * cos(2π * χ / 0.09)       # two oscillations along each line
    n = 1.0e14 .* along
    u = reshape(u0 .* along, G.NR, G.NZ)
    rec = run_mixing!(RP, n, u, collect(values(tubes)))
    xpoint[label] = (RP = RP, tubes = tubes, n0 = n, u0 = vec(u), rec = rec)
end
RPx = xpoint["particle mixing"].RP
Gx = RPx.G
Vx = vec(Gx.inVol2D)
dP = Dict(label => r.rec.P[end] / r.rec.P[1] - 1 for (label, r) in xpoint)
dE = Dict(label => r.rec.E[end] / r.rec.E[1] - 1 for (label, r) in xpoint)
for (label, r) in xpoint
    @printf("X-point, %s: particle momentum %+.2f %%, energy %+.2f %%\n", label, 100 * dP[label], 100 * dE[label])
end
xpoint_ok = abs(dP["particle mixing"]) < 1.0e-2 && abs(dE["particle mixing"]) < 1.0e-2

# ── figures ────────────────────────────────────────────────────────────────────────────
# the summary, with the verdict: one column, the X-point maps, the tubes
p1 = plot(xlabel = "Z (m)", ylabel = "u∥ (m/s)", title = @sprintf("one column, R = %.2f m, t = 6τ", G.R1D[i_mid]), legend = :topright)
plot!(p1, Z, straight["particle mixing"].u0[col]; lw = 2, c = :gray, label = "initial")
for (label, run) in straight
    plot!(p1, Z, vec(run.RP.plasma.ue_para)[col]; lw = 2, label)
end
hline!(p1, [pred.u]; ls = :dash, c = :black, label = "particle-weighted mean")
p2 = plot(xlabel = "Z (m)", ylabel = "Tₑ (eV)", legend = :topright)
for (label, run) in straight
    plot!(p2, Z, vec(run.RP.plasma.Te_eV)[col]; lw = 2, label)
end
hline!(p2, [pred.T]; ls = :dash, c = :black, label = "energy-conserving Tₑ")
ψx = XP.Bprime / 2 .* ((Gx.R2D .- XP.R0) .^ 2 .- (Gx.Z2D .- XP.Z0) .^ 2)
function umap(RP, u, title)
    p = heatmap(RP.G.R1D, RP.G.Z1D, u' ./ 1.0e6; xlabel = "R (m)", ylabel = "Z (m)", title, c = :viridis, clims = (0.8, 3.2), aspect_ratio = :equal, colorbar_title = "u∥ (10⁶ m/s)")
    contour!(p, RP.G.R1D, RP.G.Z1D, ψx'; levels = 12, c = :white, lw = 0.5, colorbar_entry = false)
    plot!(p, RP.wall.R, RP.wall.Z; c = :black, lw = 2, label = false)
    return p
end
m1 = umap(RPx, reshape(xpoint["particle mixing"].u0, Gx.NR, Gx.NZ), "u∥ initial")
m2 = umap(RPx, RPx.plasma.ue_para, @sprintf("particle mixing, t = %.2f ms", 1.0e3 * t_x))
m3 = umap(xpoint["velocity diffusion"].RP, xpoint["velocity diffusion"].RP.plasma.ue_para, @sprintf("velocity diffusion, t = %.2f ms", 1.0e3 * t_x))
keys_sorted = sort(collect(keys(xpoint["particle mixing"].tubes)))
p4 = plot(xlabel = "flux tube (ψ bin, branch)", ylabel = "u∥ (10⁶ m/s)", legend = :topright, xticks = (1:length(keys_sorted), string.(keys_sorted)), xrotation = 30)
for (label, r) in xpoint
    vals = [wmean(r.tubes[k], Vx, vec(r.RP.plasma.ne), vec(r.RP.plasma.ue_para)) for k in keys_sorted] ./ 1.0e6
    scatter!(p4, 1:length(keys_sorted), vals; ms = 6, label)
end
preds = [mixed_state(xpoint["particle mixing"].tubes[k], Vx, xpoint["particle mixing"].n0, xpoint["particle mixing"].u0).u for k in keys_sorted] ./ 1.0e6
scatter!(p4, 1:length(keys_sorted), preds; m = :x, ms = 8, c = :black, label = "particle-weighted prediction")
fig = plot(p1, p2, m1, m2, m3, p4; layout = (3, 2), size = (1200, 1500), left_margin = 8Plots.mm)
save_with_verdict(
    fig, out, "xpoint_mixing", straight_ok && xpoint_ok,
    @sprintf(
        "with particle mixing a straight column should settle within 1 %% of its particle-weighted mean and energy-conserving temperature (it ends at %.4g m/s, %.3f eV against %.4g m/s, %.3f eV), and on the X-point the particle momentum and energy should be kept within 1 %% (they move by %+.2f %% and %+.2f %%; velocity diffusion moves them by %+.2f %% and %+.2f %%)",
        mean(u_s), mean(T_s), pred.u, pred.T, 100 * dP["particle mixing"], 100 * dE["particle mixing"], 100 * dP["velocity diffusion"], 100 * dE["velocity diffusion"]
    ),
)

# the traces: what the particles carry, and how fast the lines homogenize
q1 = plot(ylabel = "Σ V n u∥ / initial − 1 (%)", title = "X-point: the particles' momentum", legend = :bottomleft)
q2 = plot(ylabel = "Σ V n ε̄ / initial − 1 (%)", title = "X-point: the particles' energy", legend = :bottomleft)
q3 = plot(ylabel = "⟨Tₑ⟩ (eV)", xlabel = "t (ms)", title = "heating by the erased shear", legend = :bottomright)
q4 = plot(ylabel = "spread of u∥ along a line (m/s)", xlabel = "t (ms)", title = "homogenization along the lines", yscale = :log10, legend = :topright)
for (label, r) in xpoint
    t = r.rec.t .* 1.0e3
    plot!(q1, t, 100 .* (r.rec.P ./ r.rec.P[1] .- 1); lw = 2, label)
    plot!(q2, t, 100 .* (r.rec.E ./ r.rec.E[1] .- 1); lw = 2, label)
    plot!(q3, t, r.rec.T; lw = 2, label = "X-point, " * label)
    plot!(q4, t, max.(r.rec.spread, 1.0); lw = 2, label = "X-point, " * label)
end
for (label, r) in straight
    t = r.rec.t .* 1.0e3
    plot!(q3, t, r.rec.T; lw = 2, ls = :dash, label = "straight, " * label)
    plot!(q4, t, max.(r.rec.spread, 1.0); lw = 2, ls = :dash, label = "straight, " * label)
end
savefig(plot(q1, q2, q3, q4; layout = (2, 2), size = (1100, 800), left_margin = 8Plots.mm), joinpath(out, "traces.png"))

# 2-D frames in time: rows = policies, columns = times, one colour range per field, with the
# field lines (ψ contours) drawn on the X-point frames
function frame(RP, snap, field, title; clims, contours)
    F = getfield(snap, field)
    scale = field === :ue_para ? 1.0e6 : 1.0
    p = heatmap(RP.G.R1D, RP.G.Z1D, F' ./ scale; xlabel = "R (m)", ylabel = "Z (m)", title, c = :viridis, clims, aspect_ratio = :equal, colorbar = false, titlefontsize = 9)
    contours && contour!(p, RP.G.R1D, RP.G.Z1D, ψx'; levels = 12, c = :white, lw = 0.4, colorbar_entry = false)
    plot!(p, RP.wall.R, RP.wall.Z; c = :black, lw = 1.5, label = false)
    return p
end
function frames_figure(runs, field, times_ms; clims, contours, file, unit)
    panels = []
    for (label, RP) in runs
        s2 = RP.diagnostics.snaps2D
        t_ms = [s.time_s for s in s2] .* 1.0e3
        for (j, t) in enumerate(times_ms)
            k = argmin(abs.(t_ms .- t))
            title = j == 1 ? @sprintf("%s\n%s  %.2f ms", label, unit, t_ms[k]) : @sprintf("%.2f ms", t_ms[k])
            push!(panels, frame(RP, s2[k], field, title; clims, contours))
        end
    end
    n = length(times_ms)
    savefig(plot(panels...; layout = (length(runs), n), size = (300 * n, 330 * length(runs)), margin = 3Plots.mm), file)
    return nothing
end
xruns = [label => xpoint[label].RP for (label, _) in POLICIES]
sruns = [label => straight[label].RP for (label, _) in POLICIES]
frames_figure(xruns, :ue_para, collect(range(0, 1.0e3 * t_x; length = 5)); clims = (0.8, 3.2), contours = true, file = joinpath(out, "xpoint_ue_para_frames.png"), unit = "u∥ (10⁶ m/s)")
frames_figure(xruns, :Te_eV, collect(range(0, 1.0e3 * t_x; length = 5)); clims = (9.9, 11.0), contours = true, file = joinpath(out, "xpoint_Te_frames.png"), unit = "Tₑ (eV)")
frames_figure(sruns, :ue_para, collect(range(0, 1.0e3 * 6τ; length = 5)); clims = (0.8, 3.2), contours = false, file = joinpath(out, "straight_ue_para_frames.png"), unit = "u∥ (10⁶ m/s)")
frames_figure(sruns, :Te_eV, collect(range(0, 1.0e3 * 6τ; length = 5)); clims = (9.9, 11.0), contours = false, file = joinpath(out, "straight_Te_frames.png"), unit = "Tₑ (eV)")
animate2D(["particle mixing" => RPx, "velocity diffusion" => xpoint["velocity diffusion"].RP], [:ue_para, :Te_eV]; file = joinpath(out, "xpoint_snaps2D.mp4"), fps = 4)
println("outputs in ", out)
