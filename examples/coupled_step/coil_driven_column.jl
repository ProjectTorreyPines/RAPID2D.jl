# No loop voltage: a coil at R = 0.6 m, inboard of the grid like a central solenoid, with
# 10 V applied, is the only drive. The column is the secondary of a transformer. The
# prediction is two coupled circuits, the coil and the column as one plasma loop carrying a
# uniform current (lumped_column in common.jl),
#   [L_c  M; M  L_p + L_kin] d/dt [I_c; I_p] = [V − R_c I_c; −R_p I_p],
# with L_kin = mₑ 2πR / (n e² πa²) the electrons' inertia, R_p the Coulomb resistance of the
# column, and M the coil's flux per ampere of that uniform current. Two runs, as in
# two_coils.jl: Ampère from the first step (threshold 0), and the default threshold (1 A).
#
#   julia --project=examples examples/coupled_step/coil_driven_column.jl

include("common.jl")

const V_COIL, R_COIL = 10.0, 1.0e-4   # applied voltage [V], coil resistance [Ω]
const R0, A0 = 1.5, 0.3               # column centre and radius [m]

function driven(threshold)
    RP = column("coupled_step/coil_driven_column/threshold_$threshold"; E0 = 0.0, cenR = R0, radius = A0, threshold)
    Lc = add_loop!(RP, 0.6, 0.0; a = 0.1, R = R_COIL, V = V_COIL, name = "OH")
    initialize_coil_system!(RP)
    rec = (t = [0.0], Ip = [0.0], Ic = [0.0])
    run_simulation!(
        RP; callback_after_step = rp -> begin
            push!(rec.t, rp.time_s)
            push!(rec.Ip, plasma_current(rp, current_density(rp)))
            push!(rec.Ic, rp.coil_system.coils[1].current)
        end
    )
    return RP, Lc, rec
end

RP, Lc, open = driven(0.0)
_, _, default = driven(1.0)

# The lumped model, read after the run: n and Te are fixed, so the resistance is constant.
m = lumped_column(RP; R0)
L = [Lc m.M[1]; m.M[1] (m.L_p + m.L_kin)]
R = [R_COIL 0; 0 column_resistance(RP)]
model = circuits(_ -> L, _ -> R, [V_COIL, 0.0], [0.0, 0.0], open.t)
@printf(
    "at %.2f ms: I_p = %.1f A (model %.1f A), I_coil = %.0f A (model %.0f A); threshold 1 A: I_p = %.2f A\n",
    open.t[end] * 1.0e3, open.Ip[end], model[end][2], open.Ic[end], model[end][1], default.Ip[end]
)

out = output_dir("coupled_step")
t = open.t .* 1.0e3
p1 = plot(
    t, last.(model); c = :green, lw = 5, alpha = 0.35, label = "two-circuit model", ylabel = "I_plasma (A)",
    title = "a 10 V coil drives the column; no loop voltage", legend = :bottomleft,
)
plot!(p1, t, open.Ip; c = :royalblue, lw = 2, label = "RAPID2D, Ampère threshold 0")
plot!(p1, t, default.Ip; c = :crimson, ls = :dot, lw = 2, label = "RAPID2D, threshold 1 A (default)")
p2 = plot(
    t, first.(model) ./ 1.0e3; c = :green, lw = 5, alpha = 0.35, label = "two-circuit model",
    ylabel = "I_coil (kA)", xlabel = "t (ms)", legend = :topleft,
)
plot!(p2, t, open.Ic ./ 1.0e3; c = :royalblue, lw = 2, label = "threshold 0")
plot!(p2, t, default.Ic ./ 1.0e3; c = :crimson, ls = :dot, lw = 2, label = "threshold 1 A")
fig = plot(
    plot_layout(RP; title = "column and driving coil"), p1, p2;
    layout = @layout([a{0.38w} grid(2, 1)]), size = (1200, 620), margin = 4Plots.mm,
)
savefig(fig, joinpath(out, "coil_driven_column.png"))
println("outputs in ", out)
