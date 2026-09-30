# A column driven by the loop voltage, with a superconducting loop beside it. A
# superconducting loop keeps the flux it links, L_c I_c + Φ_p = 0 with Φ_p the plasma's flux
# through it, so its current is −Φ_p/L_c at all times. At t1 every electron and ion is
# cloned with its own velocity: n → 2n at a fixed drift. Magnetic flux cannot jump, so at the
# next step the drift drops to about half and the plasma current stays close to where it
# was; Coulomb resistivity does not depend on n, so it stays there. The loop current, tied
# to the plasma flux, should not move either.
#
#   julia --project=examples examples/coupled_step/density_doubling.jl

include("common.jl")

const T1 = 0.75e-3   # when the density doubles [s]

RP = column("coupled_step/density_doubling"; t_end = 1.5e-3)
const Lc = add_loop!(RP, 1.2, 0.8)
initialize_coil_system!(RP)

# After every step: the plasma current, the loop current, and the current that holds the
# loop's flux, −Φ_p/L_c. At T1, after recording, the density doubles.
const rec = (t = Float64[], Ip = Float64[], Ic = Float64[], Ic_flux = Float64[])
const doubled = Ref(false)
function record_then_double(rp)
    J = current_density(rp)
    push!(rec.t, rp.time_s)
    push!(rec.Ip, plasma_current(rp, J))
    push!(rec.Ic, rp.coil_system.coils[1].current)
    push!(rec.Ic_flux, -flux_at_coils(rp, J)[1] / Lc)
    if !doubled[] && rp.time_s >= T1 - 1.0e-12
        rp.plasma.ne .*= 2
        rp.plasma.ni .*= 2
        doubled[] = true
    end
    return nothing
end
run_simulation!(RP; callback_after_step = record_then_double)

k1 = findlast(<(T1 + 1.0e-12), rec.t)
@printf(
    "plasma current %.1f A before the doubling, %.1f A two steps after; loop current at the end %.2f A (flux held: %.2f A)\n",
    rec.Ip[k1], rec.Ip[k1 + 2], rec.Ic[end], rec.Ic_flux[end]
)

out = output_dir("coupled_step")
t = rec.t .* 1.0e3
p1 = plot(
    t, rec.Ip; c = :royalblue, lw = 2, label = "RAPID2D", ylabel = "I_plasma (A)",
    title = "n → 2n at t = $(T1 * 1.0e3) ms, drift unchanged", legend = :bottomright,
)
vline!(p1, [T1 * 1.0e3]; c = :gray, lw = 1, label = "n → 2n")
p2 = plot(
    t, rec.Ic_flux; c = :green, lw = 5, alpha = 0.35, label = "flux held: −Φ_p/L_c",
    ylabel = "loop current (A)", xlabel = "t (ms)", legend = :right,
)
plot!(p2, t, rec.Ic; c = :royalblue, lw = 2, ls = :dash, label = "RAPID2D")
vline!(p2, [T1 * 1.0e3]; c = :gray, lw = 1, label = "")
fig = plot(
    plot_layout(RP; title = "column and superconducting loop"), p1, p2;
    layout = @layout([a{0.38w} grid(2, 1)]), size = (1200, 620), margin = 4Plots.mm,
)
savefig(fig, joinpath(out, "density_doubling.png"))
println("outputs in ", out)
