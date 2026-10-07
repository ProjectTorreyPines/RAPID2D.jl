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
#   julia --project=examples examples/xpoint_mixing.jl

include("common.jl")
using RAPID2D: PrescribedTensor, ParticleMixing, VelocityDiffusion, XPointPoloidal, UniformPoloidal

const me, ee = 9.1093837015e-31, 1.602176634e-19
const XP = (R0 = 1.5, Z0 = 0.0, Bprime = 0.02)

# ── the pure-mixing run ────────────────────────────────────────────────────────────────
function mixing_RP(; N, t_end_s, D_along, poloidal, policy, dt = 1.0e-6)
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = N, NZ = N, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.2, 1.8, 1.8, 1.2], wall_Z = [-0.3, -0.3, 0.3, 0.3],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = dt, t_end_s = t_end_s,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, electron_wall_albedo = 1.0,
        Output_path = mktempdir(),
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
        diffusion_tensor = PrescribedTensor(D_along = D_along, D_across = 0.0),
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

function run_mixing!(RP, n, u, T)
    inw = RP.G.nodes.in_wall_nids
    RP.plasma.ne .= 0.0
    RP.plasma.ne[inw] .= n[inw]
    RP.plasma.ue_para .= u
    RP.plasma.Te_eV .= T
    run_simulation!(RP; callback_before_step = no_friction!)
    return RP
end

# what a set of nodes mixed by its own particles settles to: the particle-weighted mean
# velocity and the temperature that conserves the energy per particle
function mixed_state(nodes, V, n, u, T)
    w = V[nodes] .* n[nodes]
    W = sum(w)
    u∞ = sum(w .* u[nodes]) / W
    ε = sum(w .* (1.5 .* ee .* T[nodes] .+ 0.5 .* me .* u[nodes] .^ 2)) / W
    return (u = u∞, T = (ε - 0.5 * me * u∞^2) / (1.5 * ee))
end
wmean(nodes, V, n, f) = sum(V[nodes] .* n[nodes] .* f[nodes]) / sum(V[nodes] .* n[nodes])
momentum(RP) = (inw = RP.G.nodes.in_wall_nids; sum(vec(RP.G.inVol2D)[inw] .* vec(RP.plasma.ne)[inw] .* vec(RP.plasma.ue_para)[inw]))
function energy(RP)
    inw = RP.G.nodes.in_wall_nids
    V, n, u, T = vec(RP.G.inVol2D), vec(RP.plasma.ne), vec(RP.plasma.ue_para), vec(RP.plasma.Te_eV)
    return sum(V[inw] .* n[inw] .* (1.5 .* ee .* T[inw] .+ 0.5 .* me .* u[inw] .^ 2))
end

out = output_dir("xpoint_mixing")
D, u0, T0 = 500.0, 2.0e6, 10.0
policies = ["particle mixing" => ParticleMixing(), "velocity diffusion" => VelocityDiffusion()]

# ── 1. straight lines: each R column is one line ───────────────────────────────────────
L = 0.6                                       # the column between the walls
τ = L^2 / (π^2 * D)
straight = Dict{String, Any}()
for (label, policy) in policies
    RP = mixing_RP(; N = 41, t_end_s = 6τ, D_along = D, poloidal = UniformPoloidal(), policy)
    G = RP.G
    shape = @. 1 + 0.5 * cos(2π * G.Z2D / L)   # n and u correlated along the line; T uniform
    n = 1.0e14 .* vec(shape)
    u = u0 .* shape
    run_mixing!(RP, n, u, fill(T0, G.NR, G.NZ))
    straight[label] = (RP = RP, n0 = n, u0 = u)
end
RPs = straight["particle mixing"].RP
G = RPs.G
i_mid = G.nodes.rid[G.nodes.in_wall_nids[length(G.nodes.in_wall_nids) ÷ 2]]
col = [nid for nid in G.nodes.in_wall_nids if G.nodes.rid[nid] == i_mid]
Z = vec(G.Z2D)[col]
V = vec(G.inVol2D)
pred = mixed_state(col, V, straight["particle mixing"].n0, straight["particle mixing"].u0, fill(T0, length(V)))
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
u_s = vec(RPs.plasma.ue_para)[col]
T_s = vec(RPs.plasma.Te_eV)[col]
straight_ok = all(x -> isapprox(x, pred.u; rtol = 1.0e-2), u_s) && all(x -> isapprox(x, pred.T; rtol = 1.0e-2), T_s)
@printf(
    "straight column: predicted u∞ %.4g m/s, T∞ %.3f eV; particle mixing ends at %.4g m/s, %.3f eV; velocity diffusion at %.4g m/s, %.3f eV\n",
    pred.u, pred.T, mean(u_s), mean(T_s), mean(vec(straight["velocity diffusion"].RP.plasma.ue_para)[col]), mean(vec(straight["velocity diffusion"].RP.plasma.Te_eV)[col])
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
xpoint = Dict{String, Any}()
for (label, policy) in policies
    RP = mixing_RP(; N = 49, t_end_s = 2.5e-4, D_along = D, poloidal = XPointPoloidal(; XP...), policy)
    G = RP.G
    tubes, χ = tubes_of(G)
    along = @. 1 + 0.5 * cos(2π * χ / 0.09)       # two oscillations along each line
    n = 1.0e14 .* along
    u = reshape(u0 .* along, G.NR, G.NZ)
    P0, E0 = (sum(vec(G.inVol2D)[G.nodes.in_wall_nids] .* n[G.nodes.in_wall_nids] .* vec(u)[G.nodes.in_wall_nids]), 0.0)
    RP.plasma.ne .= 0.0
    RP.plasma.ne[G.nodes.in_wall_nids] .= n[G.nodes.in_wall_nids]
    RP.plasma.ue_para .= u
    RP.plasma.Te_eV .= T0
    E0 = energy(RP)
    u_init = copy(RP.plasma.ue_para)
    run_simulation!(RP; callback_before_step = no_friction!)
    xpoint[label] = (RP = RP, tubes = tubes, n0 = n, u0 = vec(u_init), P0 = P0, E0 = E0)
end
RPx = xpoint["particle mixing"].RP
Gx = RPx.G
Vx = vec(Gx.inVol2D)
dP = Dict(label => (momentum(r.RP) - r.P0) / r.P0 for (label, r) in xpoint)
dE = Dict(label => (energy(r.RP) - r.E0) / r.E0 for (label, r) in xpoint)
for (label, r) in xpoint
    @printf("X-point, %s: particle momentum %+.2f %%, energy %+.2f %%\n", label, 100 * dP[label], 100 * dE[label])
end
# maps of u∥ with the field lines
ψx = XP.Bprime / 2 .* ((Gx.R2D .- XP.R0) .^ 2 .- (Gx.Z2D .- XP.Z0) .^ 2)
function umap(RP, u, title)
    p = heatmap(RP.G.R1D, RP.G.Z1D, u' ./ 1.0e6; xlabel = "R (m)", ylabel = "Z (m)", title, c = :viridis, clims = (0.8, 3.2), aspect_ratio = :equal, colorbar_title = "u∥ (10⁶ m/s)")
    contour!(p, RP.G.R1D, RP.G.Z1D, ψx'; levels = 12, c = :white, lw = 0.5, colorbar_entry = false)
    plot!(p, RP.wall.R, RP.wall.Z; c = :black, lw = 2, label = false)
    return p
end
m1 = umap(RPx, reshape(xpoint["particle mixing"].u0, Gx.NR, Gx.NZ), "u∥ initial")
m2 = umap(RPx, RPx.plasma.ue_para, "particle mixing, t = 0.25 ms")
m3 = umap(xpoint["velocity diffusion"].RP, xpoint["velocity diffusion"].RP.plasma.ue_para, "velocity diffusion, t = 0.25 ms")
# per tube: what each policy settles to against the particle-weighted prediction
keys_sorted = sort(collect(keys(xpoint["particle mixing"].tubes)))
p4 = plot(xlabel = "flux tube (ψ bin, branch)", ylabel = "u∥ (10⁶ m/s)", legend = :topright, xticks = (1:length(keys_sorted), string.(keys_sorted)), xrotation = 30)
for (label, r) in xpoint
    vals = [wmean(r.tubes[k], Vx, vec(r.RP.plasma.ne), vec(r.RP.plasma.ue_para)) for k in keys_sorted] ./ 1.0e6
    scatter!(p4, 1:length(keys_sorted), vals; ms = 6, label)
end
preds = [mixed_state(xpoint["particle mixing"].tubes[k], Vx, xpoint["particle mixing"].n0, xpoint["particle mixing"].u0, fill(T0, length(Vx))).u for k in keys_sorted] ./ 1.0e6
scatter!(p4, 1:length(keys_sorted), preds; m = :x, ms = 8, c = :black, label = "particle-weighted prediction")

xpoint_ok = abs(dP["particle mixing"]) < 1.0e-2 && abs(dE["particle mixing"]) < 1.0e-2
fig = plot(p1, p2, m1, m2, m3, p4; layout = (3, 2), size = (1200, 1500), left_margin = 8Plots.mm)
save_with_verdict(
    fig, out, "xpoint_mixing", straight_ok && xpoint_ok,
    @sprintf(
        "with particle mixing a straight column should settle within 1 %% of its particle-weighted mean and energy-conserving temperature (it ends at %.4g m/s, %.3f eV against %.4g m/s, %.3f eV), and on the X-point the particle momentum and energy should be kept within 1 %% (they move by %+.2f %% and %+.2f %%; velocity diffusion moves them by %+.2f %% and %+.2f %%)",
        mean(u_s), mean(T_s), pred.u, pred.T, 100 * dP["particle mixing"], 100 * dE["particle mixing"], 100 * dP["velocity diffusion"], 100 * dE["velocity diffusion"]
    ),
)
println("outputs in ", out)
