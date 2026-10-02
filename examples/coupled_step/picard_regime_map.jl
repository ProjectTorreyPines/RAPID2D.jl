# Where the coupled solve's default iteration stops reproducing the step iterated to
# convergence, in a KSTAR-like domain (the grid of the KSTAR field files with the KSTAR first
# wall). Columns of radius 0.25–0.45 m stand with their inboard edge at R = 1.30 m, 4 cm from
# the inboard wall, as an inboard-limited start-up plasma does, at states from a late avalanche
# (1e16 m⁻³, 2 eV) to a formed plasma (1e19 m⁻³, 20 eV). Each runs 5 steps from rest with the
# default and with the converged iteration; the map shows the largest gap between their plasma
# currents, which should stay under 1 %.
#
#   julia --project=examples examples/coupled_step/picard_regime_map.jl

include("common.jl")

const RADII = [0.25, 0.3, 0.35, 0.4, 0.45]
const STATES = [(1.0e16, 2.0), (1.0e17, 5.0), (1.0e18, 10.0), (1.0e19, 20.0)]
const NSTEPS = 5

kstar_column(n0, Te, a) = column(
    "coupled_step/picard_regime_map"; n0, Te, cenR = 1.3 + a, radius = a,
    manual = kstar_like(0.3), wall_R = KSTAR_WALL_R, wall_Z = KSTAR_WALL_Z,
)

gaps = fill(NaN, length(STATES), length(RADII))
for (j, a) in enumerate(RADII), (i, (n0, Te)) in enumerate(STATES)
    def, conv = default_vs_converged(() -> kstar_column(n0, Te, a); nsteps = NSTEPS)
    gaps[i, j] = maximum(abs.(def.I .- conv.I)) / maximum(abs, conv.I)
    @printf(
        "a = %.2f m, n = %.0e m⁻³, Te = %4.1f eV: gap %.2e; default %.1f solves/step, %d unconverged; converged %.0f solves/step\n",
        a, n0, Te, gaps[i, j], def.stats.niter / def.stats.nsolve, def.stats.nunconverged, conv.stats.niter / conv.stats.nsolve
    )
end

state_labels = [@sprintf("%.0e m⁻³, %g eV", n0, Te) for (n0, Te) in STATES]
RP_big = kstar_column(1.0e18, 10.0, RADII[end])
p1 = plot!(
    plot_layout(RP_big; J = RP_big.plasma.ne, title = "largest column (density), inboard edge at R = 1.30 m");
    colorbar = false, titlefontsize = 9,
)
logg = log10.(max.(gaps, 1.0e-8))
p2 = heatmap(
    RADII, 1:length(STATES), logg; c = cgrad([:seagreen, :khaki, :orange, :red3], [0.0, 0.5, 0.75, 1.0]),
    clims = (-6, 2), yticks = (1:length(STATES), state_labels), xlabel = "column radius a (m)",
    title = "log₁₀ of the current gap, default vs converged (fail above −2)", colorbar_title = "log₁₀ gap",
)
for (j, a) in enumerate(RADII), i in eachindex(STATES)
    g = gaps[i, j]
    annotate!(p2, a, i, text(g > 1.0e-2 ? @sprintf("FAIL\n%.3g%%", 100g) : @sprintf("%.2g%%", 100g), 8, g > 1.0e-2 ? :white : :black))
end
fig = plot(p1, p2; layout = @layout([a{0.32w} b]), size = (1400, 560), margin = 6Plots.mm, top_margin = 10Plots.mm)
nfail = count(>(1.0e-2), gaps)
save_with_verdict(
    fig, output_dir("coupled_step"), "picard_regime_map", nfail == 0,
    "$nfail of $(length(gaps)) inboard-limited KSTAR-like columns step more than 1 % off the converged solve",
)
