# Two coupled toroidal loops and no plasma. Loop A starts at 1 kA and decays through its
# resistance; loop B, linked to A by their mutual inductance, picks up the current that A's
# decay induces. The coil circuits advance by backward Euler,
#   (M + Δt R) Iⁿ⁺¹ = M Iⁿ,
# so that recursion is the answer for the discrete system, step for step. Two runs: Ampère
# from the first step (threshold 0), and the default threshold (1 A). With no plasma the
# threshold on the plasma current has nothing to act on, so both runs should follow the
# same curve.
#
#   julia --project=examples examples/coupled_step/two_coils.jl

include("common.jl")

const DT, T_END, I_START = 20.0e-6, 3.0e-3, 1000.0

function two_coils(threshold)
    RP = column("coupled_step/two_coils/threshold_$threshold"; E0 = 0.0, n0 = 0.0, threshold, dt = DT, t_end = T_END)
    add_loop!(RP, 1.0, 0.6; R = 5.0e-3, I0 = I_START, name = "A")
    add_loop!(RP, 1.2, 0.8; R = 5.0e-3, name = "B")
    initialize_coil_system!(RP)
    I = [copy(get_all_currents(RP.coil_system))]
    run_simulation!(RP; callback_after_step = rp -> push!(I, copy(get_all_currents(rp.coil_system))))
    return RP, I
end

RP, I_open = two_coils(0.0)
_, I_default = two_coils(1.0)

# the backward-Euler recursion, from the same mutual-inductance matrix
M, r = RP.coil_system.mutual_inductance, get_all_resistances(RP.coil_system)
step = (M + DT * [r[1] 0; 0 r[2]]) \ M
exact = accumulate((I, _) -> step * I, 1:(length(I_open) - 1); init = [I_START, 0.0])
pushfirst!(exact, [I_START, 0.0])
dev(run) = maximum(maximum(abs.(run[k] .- exact[k])) for k in eachindex(exact)) / I_START
@printf("largest deviation from the recursion, over I_start: %.2e (threshold 0), %.2e (threshold 1 A)\n", dev(I_open), dev(I_default))

out = output_dir("coupled_step")
t = (0:(length(exact) - 1)) .* DT .* 1.0e3
p1 = plot(
    t, first.(exact); c = :green, lw = 5, alpha = 0.35, label = "backward Euler, exact",
    ylabel = "I_A (A)", title = "two loops, no plasma: A decays, B is induced",
)
plot!(p1, t, first.(I_open); c = :royalblue, lw = 2, label = "RAPID2D, Ampère threshold 0")
plot!(p1, t, first.(I_default); c = :crimson, ls = :dot, lw = 2, label = "RAPID2D, threshold 1 A (default)")
p2 = plot(t, last.(exact); c = :green, lw = 5, alpha = 0.35, label = "exact", ylabel = "I_B (A)", xlabel = "t (ms)")
plot!(p2, t, last.(I_open); c = :royalblue, lw = 2, label = "threshold 0")
plot!(p2, t, last.(I_default); c = :crimson, ls = :dot, lw = 2, label = "threshold 1 A")
savefig(plot(p1, p2; layout = (2, 1), size = (720, 620)), joinpath(out, "two_coils.png"))
println("outputs in ", out)
