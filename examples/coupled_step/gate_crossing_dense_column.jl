# A column at rest that is already dense when the loop voltage comes on, as a pre-ionized
# plasma or a formed column at I = 0 is, under the default Ampère gate of 1 A. Below the gate
# the plasma is not a source of induction, so the step that carries the current across the gate
# accelerates the electrons without their self-inductance; with the gate at 0 every step has it.
# The two runs should step alike: the induced field holds the rise to about V/(L_p + L_kin) per
# unit time, a few amperes per step here.
#
#   julia --project=examples examples/coupled_step/gate_crossing_dense_column.jl

include("common.jl")

const CASES = [(1.0e16, 1.0), (1.0e17, 5.0), (1.0e18, 5.0)]
const NSTEPS = 10

runs = map(CASES) do (n0, Te)
    map((1.0, 0.0)) do threshold
        RP = column("coupled_step/gate_crossing_dense_column"; n0, Te, threshold, t_end = NSTEPS * 5.0e-6)
        I = Float64[]
        quiet(() -> run_simulation!(RP; callback_after_step = rp -> push!(I, plasma_current(rp, current_density(rp)))))
        I
    end
end

t = (1:NSTEPS) .* 5.0
gaps = [maximum(abs.(I1 .- I0)) / maximum(abs, I0) for (I1, I0) in runs]
panels = map(zip(CASES, runs)) do ((n0, Te), (I1, I0))
    p = plot(
        t, I0; c = :black, lw = 3, label = "gate 0 (expected)", yscale = :log10, xlabel = "t (µs)", ylabel = "I_p (A)",
        title = @sprintf("n = %.0e m⁻³, Te = %g eV", n0, Te), legend = :right,
    )
    plot!(p, t, I1; c = :red3, ls = :dash, lw = 2, m = :circle, ms = 3, label = "gate 1 A (default)")
    p
end
fig = plot(panels...; layout = (1, length(CASES)), size = (1500, 480), margin = 6Plots.mm, top_margin = 10Plots.mm)
worst = argmax(gaps)
save_with_verdict(
    fig, output_dir("coupled_step"), "gate_crossing_dense_column", all(<=(1.0e-2), gaps),
    @sprintf(
        "the 1 A gate run stays within 1 %% of the gate-0 run in %d of %d columns; worst: first step %.3g A vs %.3g A",
        count(<=(1.0e-2), gaps), length(gaps), runs[worst][1][1], runs[worst][2][1]
    ),
)
