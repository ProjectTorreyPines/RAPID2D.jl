# Current on closed flux surfaces whose conductivity varies along each surface. A column sits on
# prescribed elliptic surfaces and is driven by a loop voltage. Friction is on a neutral background
# at a fixed rate ν, so σ = n e²/(mₑ ν) follows the density, which is either a surface function
# (α = 0) or nine times higher outboard than inboard on every surface (α = 0.8). The density and Te
# are held, and the ions are at rest.
#
# On a closed surface the parallel current returns along the field line, and quasineutrality,
# ∇·J = 0, makes J∥/B a surface function: each surface is a series circuit, with
# J∥/B = ⟨E∥B⟩/⟨B²/σ⟩ in steady state, and the surfaces are coupled inductively while the current
# rises. Three treatments of the electron parallel flow:
# - free flow (`FreeParallelFlow`): u∥ node by node, so each node carries its local Ohmic current;
# - post-projection (`SurfaceFluxConstraint(; coupling = PostProjection())`): n u∥/B a function of ψ on the
#   closed region, with ψ and the induced field solved for the unconstrained current;
# - block constraint (`SurfaceFluxConstraint()`): n u∥/B solved inside the coupled u∥–ψ step.
# The reference is a 1-D model of the same physics: nested elliptic shells, each with its
# shell-averaged Ohm's law and electron inertia, coupled by free-space Green's functions, solved
# exactly in time.
#
#   julia --project=examples examples/closed_surface_current.jl

include("common.jl")
using LinearAlgebra
using RAPID2D: SurfaceFluxConstraint, FreeParallelFlow, BlockConstraint, PostProjection, ManualSetup,
    flf_analysis_field_lines_rz_plane!, update_closed_surface_basis!, combine_external_and_self_fields!,
    calculate_B_from_ψ!, update_transport_quantities!, calculate_ψ_by_green_function

const R0, Z0, κ = 1.5113, 0.0071, 1.3      # axis of the elliptic surfaces (off the nodes), elongation
const C_PSI = 3.0e-2                        # ψ_ext = C ((R − R0)² + ((Z − Z0)/κ)²) [Wb/rad/m²]
const RHO_N = 0.4                           # the column's edge, as the ellipse's minor radius [m]
const N0, NU0, E0, TE = 1.0e17, 1.0e6, 0.3, 10.0
const R0B0, RBAR = 3.0, 1.5                 # Bϕ = R0B0/R; Eϕ = E0 R̄/R
const NR, NZ, DT, T_END = 61, 81, 2.0e-6, 1.0e-3
const QE, ME = -1.602176634e-19, 9.1093837015e-31

minor_radius(R, Z) = sqrt((R - R0)^2 + ((Z - Z0) / κ)^2)
poloidal_angle(R, Z) = atan((Z - Z0) / κ, R - R0)                       # 0 outboard
density(R, Z, α) = (ρ = minor_radius(R, Z); ρ < RHO_N ? N0 * (1 - (ρ / RHO_N)^2) * (1 + α * cos(poloidal_angle(R, Z))) : 0.0)

const POLICIES = (
    free = ("free flow", FreeParallelFlow()),
    post = ("post-projection", SurfaceFluxConstraint(; coupling = PostProjection())),
    block = ("block constraint", SurfaceFluxConstraint(; coupling = BlockConstraint())),
)
const COLORS = (free = RGB(0.85, 0.37, 0.0), post = RGB(0.0, 0.45, 0.7), block = RGB(0.0, 0.62, 0.45))

function column(policy; α, ampere, t_end)
    config = SimulationConfig{Float64}(
        manual = ManualSetup{Float64}(BR = 0.0, BZ = 0.0, Eϕ = E0), NR = NR, NZ = NZ,
        R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2, R0B0 = R0B0, prefilled_gas_pressure = 0.0,
        wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        dt = DT, t_end_s = t_end, snap0D_Δt_s = t_end, snap2D_Δt_s = t_end, Output_path = mktempdir(),
    )
    RP = setup(
        config;
        Ampere = ampere, Ampere_Itor_threshold = 0.0, E_para_self_EM = ampere, ud_evolve = true,
        Atomic_Collision = false, Coulomb_Collision = false, src = false, convec = false, diffu = false,
        Te_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
        Include_ud_convec_term = false, Include_ud_diffu_term = false, Include_ud_pressure_term = false,
        Include_Te_convec_term = false, E_para_self_ES = false, mean_ExB = false, turb_ExB_mixing = false,
        FLF_nstep = 100, closed_surface_flow = policy,
    )
    G, F, pla = RP.G, RP.fields, RP.plasma
    @. F.ψ_ext = C_PSI * ((G.R2D - R0)^2 + ((G.Z2D - Z0) / κ)^2)
    calculate_B_from_ψ!(G, F.ψ_ext, F.BR_ext, F.BZ_ext)
    fill!(F.ψ_self, 0.0); fill!(F.BR_self, 0.0); fill!(F.BZ_self, 0.0)
    combine_external_and_self_fields!(RP)
    pla.ne .= density.(G.R2D, G.Z2D, α) .* (G.nodes.state .> 0.5)
    pla.ni .= pla.ne
    fill!(pla.Te_eV, TE)
    fill!(pla.ue_para, 0.0); fill!(pla.ui_para, 0.0)
    update_transport_quantities!(RP)
    flf_analysis_field_lines_rz_plane!(RP)
    update_closed_surface_basis!(RP)
    return RP
end

# before every step: friction on the neutral background at ν, ions at rest
function hold!(RP)
    pla = RP.plasma
    pla.ν_en_mom_tot .= NU0
    pla.ν_en_iz_tot .= 0.0
    pla.ν_ei_eff .= 0.0
    pla.ui_para .= 0.0
    return nothing
end

electron_current(RP) = sum(QE .* RP.plasma.ne .* RP.plasma.ue_para .* RP.fields.bϕ) * RP.G.dR * RP.G.dZ
flux_per_B(RP) = [n > 0 ? n * u / B : 0.0 for (n, u, B) in zip(RP.plasma.ne, RP.plasma.ue_para, RP.fields.Btot)]

# ── the reference: shells of the ellipse, the momentum equation averaged over each with the
# weight B, J∥/B = q n u∥/B on each, coupled by the free-space flux of every shell's current ───
const GL2 = ([-1 / sqrt(3), 1 / sqrt(3)], [1.0, 1.0])
const GL4 = (
    [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
    [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538],
)

function shell_points(edges, rule, nθ, θ_offset)
    R, Z, dA, shell = Float64[], Float64[], Float64[], Int[]
    for j in 1:(length(edges) - 1), (x, w) in zip(rule...), i in 1:nθ
        a, b = edges[j], edges[j + 1]
        ρ, θ = (a + b) / 2 + (b - a) / 2 * x, 2π * (i - θ_offset) / nθ
        push!(R, R0 + ρ * cos(θ)); push!(Z, Z0 + κ * ρ * sin(θ))
        push!(dA, κ * ρ * (b - a) / 2 * w * 2π / nθ); push!(shell, j)
    end
    return (; R, Z, dA, shell)
end

# m A (dg/dt + ν g) = q (F − G dg/dt) per shell, g = n u∥/B: A = ∫ B²/n dV, F = ∫ Bϕ Eϕ dV, and
# G dg/dt the induced part, G_jk = ∫_j (Bϕ/R) ψ_k dV with ψ_k the flux of shell k's current per unit g.
function shell_reference(α; nshell = 40)
    edges = collect(range(0.0, RHO_N; length = nshell + 1))
    d, s = shell_points(edges, GL2, 64, 0.5), shell_points(edges, GL4, 128, 0.25)
    Bϕ(R) = R0B0 / R
    B(R, Z) = hypot(Bϕ(R), C_PSI * hypot(2 * (R - R0), 2 * (Z - Z0) / κ^2) / R)
    A, F, C, G = zeros(nshell), zeros(nshell), zeros(nshell), zeros(nshell, nshell)
    for i in eachindex(d.R)
        j, R, dV = d.shell[i], d.R[i], 2π * d.R[i] * d.dA[i]
        A[j] += B(R, d.Z[i])^2 / density(R, d.Z[i], α) * dV
        F[j] += Bϕ(R) * E0 * RBAR / R * dV
        C[j] += Bϕ(R) * d.dA[i]
    end
    for k in 1:nshell
        src = findall(==(k), s.shell)
        ψ = vec(sum(calculate_ψ_by_green_function(d.R, d.Z, s.R[src], s.Z[src], [QE * Bϕ(s.R[i]) * s.dA[i] for i in src]); dims = 2))
        for i in eachindex(d.R)
            G[d.shell[i], k] += 2π * Bϕ(d.R[i]) * d.dA[i] * ψ[i]
        end
    end
    Γ_over_B∞ = QE .* F ./ (ME * NU0 .* A)
    W = (ME .* Diagonal(A) .+ QE .* G) \ (ME * NU0 .* Matrix(Diagonal(A)))    # dg/dt = W (g∞ − g)
    Γ_over_B(t) = isinf(t) ? copy(Γ_over_B∞) : Γ_over_B∞ .- exp(-t .* W) * Γ_over_B∞
    ρ = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
    at(Kv, x) = (j = clamp(searchsortedlast(ρ, x), 1, nshell - 1); t = clamp((x - ρ[j]) / (ρ[j + 1] - ρ[j]), 0, 1); (1 - t) * Kv[j] + t * Kv[j + 1])
    return (; ρ, Γ_over_B, I = t -> sum(QE .* Γ_over_B(t) .* C), at)
end

# ── runs ────────────────────────────────────────────────────────────────────────────────────
function rise(key, α)
    name, policy = POLICIES[key]
    RP = column(policy; α, ampere = true, t_end = T_END)
    t, I, profiles = Float64[], Float64[], Dict{Int, Matrix{Float64}}()
    snaps = round.(Int, (50.0e-6, 200.0e-6, 1.0e-3) ./ DT)
    @time "$name, α = $α" redirect_stdout(devnull) do
        run_simulation!(
            RP; callback_before_step = hold!, callback_after_step = r -> begin
                push!(t, r.time_s); push!(I, electron_current(r))
                r.step in snaps && (profiles[r.step] = flux_per_B(r))
                nothing
            end
        )
    end
    return (; t, I, profiles, RP)
end

function steady(key, α)
    RP = column(POLICIES[key][2]; α, ampere = false, t_end = 20 * DT)
    redirect_stdout(() -> run_simulation!(RP; callback_before_step = hold!), devnull)
    return RP
end

out = output_dir("closed_surface_current")
refs = Dict(α => shell_reference(α) for α in (0.0, 0.8))
runs = Dict((key, α) => rise(key, α) for key in keys(POLICIES), α in (0.0, 0.8))
stills = Dict(key => steady(key, 0.8) for key in (:free, :block))

# ── steady state, α = 0.8: J∥/B on the grid and along the surfaces ──────────────────────────
let ref = refs[0.8], G = stills[:block].G
    Γ_over_B∞ = ref.Γ_over_B(Inf)
    scale = Γ_over_B∞[argmax(abs.(Γ_over_B∞))]
    ρ = minor_radius.(G.R2D, G.Z2D)
    inside = stills[:block].plasma.ne .> 0
    map_of(v) = (M = fill(NaN, G.NR, G.NZ); M[inside] .= v[inside]; M')
    panels = [
        heatmap(G.R1D, G.Z1D, map_of(flux_per_B(stills[:free]) ./ scale); title = "J∥/B, free flow"),
        heatmap(G.R1D, G.Z1D, map_of(flux_per_B(stills[:block]) ./ scale); title = "J∥/B, surface-flux constraint"),
        heatmap(G.R1D, G.Z1D, map_of([ref.at(Γ_over_B∞, x) / scale for x in ρ]); title = "J∥/B, shell reference"),
    ]
    foreach(p -> plot!(p; aspect_ratio = :equal, xlims = (1.05, 1.95), ylims = (-0.6, 0.6), c = :inferno, clims = (0, 2.5), xlabel = "R (m)"), panels)
    band = findall(x -> 0.15 < x < 0.25, ρ) ∩ findall(inside)
    θ = poloidal_angle.(G.R2D[band], G.Z2D[band])
    p4 = plot(; xlabel = "poloidal angle θ (0 outboard)", ylabel = "(J∥/B) / reference", title = "surfaces 0.15 < ρ < 0.25 m", ylims = (0, 3.4))
    for (key, label) in ((:free, "free flow"), (:block, "surface-flux constraint"))
        f = flux_per_B(stills[key])
        scatter!(p4, θ, [f[k] / ref.at(Γ_over_B∞, ρ[k]) for k in band]; ms = 2, msw = 0, color = COLORS[key], label)
    end
    hline!(p4, [1]; color = :black, ls = :dash, label = "reference")
    I_ref, I_free, I_block = (abs(x) for x in (ref.I(Inf), electron_current(stills[:free]), electron_current(stills[:block])))
    gap = abs(I_block / I_ref - 1)
    fig = plot(panels..., p4; layout = (1, 4), size = (1700, 450), left_margin = 3Plots.mm, bottom_margin = 9Plots.mm, top_margin = 8Plots.mm)
    save_with_verdict(
        fig, out, "steady_series_resistor", gap < 0.01,
        @sprintf("steady I: reference %.1f A, constraint %.1f A (%.2g %% off, needs 1 %%), free flow %.1f A", I_ref, I_block, 100gap, I_free),
    )
end

# ── the current rise ───────────────────────────────────────────────────────────────────────────
let panels = []
    gaps = Dict{Tuple{Symbol, Float64}, Float64}()
    for α in (0.0, 0.8)
        ref = refs[α]
        ts = range(0, T_END; length = 300)
        pI = plot(
            1.0e6 .* ts, abs.(ref.I.(ts)); color = :black, lw = 2.5, label = "shell reference", xlabel = "t (μs)",
            ylabel = "|I_tor| (A)", title = @sprintf("current rise, α = %.1f", α), legend = :bottomright
        )
        pe = plot(; xlabel = "t (μs)", ylabel = "(I − I_ref) / I_ref(∞)", title = "against the reference", legend = :topleft)
        I∞ = abs(ref.I(Inf))
        for key in keys(POLICIES)
            r = runs[(key, α)]
            err = (abs.(r.I) .- abs.(ref.I.(r.t))) ./ I∞
            gaps[(key, α)] = maximum(abs, err)
            plot!(pI, 1.0e6 .* r.t, abs.(r.I); color = COLORS[key], lw = 1.5, ls = key === :post ? :dash : :solid, label = POLICIES[key][1])
            plot!(pe, 1.0e6 .* r.t, err; color = COLORS[key], lw = 1.5, ls = key === :post ? :dash : :solid, label = POLICIES[key][1])
        end
        hline!(pe, [0]; color = :black, lw = 0.5, label = "")
        push!(panels, pI, pe)
    end
    fig = plot(panels...; layout = (2, 2), size = (1150, 800), left_margin = 4Plots.mm, top_margin = 8Plots.mm)
    detail = join((@sprintf("%s %.2g %%", POLICIES[k][1], 100gaps[(k, 0.8)]) for k in keys(POLICIES)), ", ")
    save_with_verdict(
        fig, out, "current_rise", gaps[(:block, 0.8)] < 0.01,
        "α = 0.8, largest gap to the reference over the rise: $detail (the block constraint needs 1 %)",
    )
end

# ── the profile during the rise, α = 0.8 ────────────────────────────────────────────────────
let ref = refs[0.8], G = runs[(:block, 0.8)].RP.G
    ρ = vec(minor_radius.(G.R2D, G.Z2D))
    Γ_over_B∞ = ref.Γ_over_B(Inf)
    scale = Γ_over_B∞[argmax(abs.(Γ_over_B∞))]
    b = runs[(:block, 0.8)].RP.closed_surface.basis
    constrained = b.nodes[b.ψN .<= 1]
    panels = map(sort(collect(keys(runs[(:block, 0.8)].profiles)))) do step
        t = step * DT
        p = plot(
            ref.ρ, ref.Γ_over_B(t) ./ scale; color = :black, lw = 2, label = "shell reference", xlabel = "ρ (m)", ylabel = "J∥/B (normalized)",
            title = @sprintf("t = %.0f μs", 1.0e6 * t), ylims = (-0.05, 1.15), legend = step == 25 ? :topright : false
        )
        for key in (:post, :block)
            f = runs[(key, 0.8)].profiles[step]
            scatter!(
                p, ρ[constrained], f[constrained] ./ scale; ms = 2.2, msw = key === :post ? 0.8 : 0, msc = COLORS[key],
                color = COLORS[key], marker = key === :post ? :xcross : :circle, label = POLICIES[key][1]
            )
        end
        p
    end
    savefig(plot(panels...; layout = (1, 3), size = (1350, 420), left_margin = 4Plots.mm, bottom_margin = 5Plots.mm), joinpath(out, "profiles_during_rise.png"))
end
# ── J∥/B on the grid during the rise: the reference and each treatment ─────────────────────
# α = 0 is current diffusion: the induced field drives a skin current that the surfaces carry
# whole while it soaks in; α = 0.8 adds the series circuits.
for α in (0.0, 0.8)
    ref = refs[α]
    G = runs[(:block, α)].RP.G
    ρ = minor_radius.(G.R2D, G.Z2D)
    has_density = runs[(:block, α)].RP.plasma.ne .> 0
    b = runs[(:block, α)].RP.closed_surface.basis
    constrained = b.nodes[b.ψN .<= 1]
    Γ_over_B∞ = ref.Γ_over_B(Inf)
    scale = Γ_over_B∞[argmax(abs.(Γ_over_B∞))]
    map_of(v) = (M = fill(NaN, G.NR, G.NZ); M[has_density] .= v[has_density]; M')
    panels = []
    for step in sort(collect(keys(runs[(:block, α)].profiles)))
        t = step * DT
        f_ref = [ref.at(ref.Γ_over_B(t), x) for x in ρ] ./ scale
        peak = maximum(abs, ref.Γ_over_B(t)) / abs(scale)
        push!(panels, heatmap(G.R1D, G.Z1D, map_of(f_ref); title = @sprintf("shell reference, t = %.0f μs", 1.0e6t), clims = (0, 1.05peak)))
        for key in keys(POLICIES)
            f = runs[(key, α)].profiles[step] ./ scale
            gap = maximum(abs, f[constrained] .- f_ref[constrained]) / peak
            push!(
                panels, heatmap(
                    G.R1D, G.Z1D, map_of(f); clims = (0, 1.05peak),
                    title = @sprintf("%s (inside the cut ≤ %.0f %% off)", POLICIES[key][1], 100gap)
                )
            )
        end
    end
    foreach(p -> plot!(p; aspect_ratio = :equal, xlims = (1.08, 1.92), ylims = (-0.55, 0.55), c = :inferno, titlefontsize = 9), panels)
    savefig(plot(panels...; layout = (3, 4), size = (1500, 1150)), joinpath(out, @sprintf("maps_during_rise_alpha%.1f.png", α)))
end
println("outputs in ", out)
