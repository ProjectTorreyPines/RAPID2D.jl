# A column (R = 1.4 m, a = 0.2 m) is driven to its saturated current, then pushed outward at
# 200 m/s, a uniform radial drift that carries n and u∥ along, toward a superconducting loop
# at R = 2.3 m, just outside the wall. The loop keeps its flux, L_c I_c + Φ_p = 0, so it
# carries a current opposite to the plasma's that grows as the column approaches. Opposite
# currents repel: the loop's force on the plasma, F_R = ∫ Jϕ B_Z dV, points back inward
# (F_R < 0). This is how eddy currents in a conductor resist the plasma's motion toward it.
# The column does not move as a rigid body: its current spreads as it goes.
#
#   julia --project=examples examples/coupled_step/column_pushed_toward_loop.jl

include("common.jl")

const T_PUSH, V_PUSH, R_LOOP = 0.6e-3, 200.0, 2.3   # push start [s], speed [m/s], loop radius [m]

RP = column("coupled_step/column_pushed_toward_loop"; cenR = 1.4, radius = 0.2, t_end = 2.0e-3, moving = true)
const Lc = add_loop!(RP, R_LOOP, 0.0)
initialize_coil_system!(RP)
const G = RP.G

# The loop's B_Z per ampere on the grid, B_Z = (1/R) ∂ψ/∂R, from the Green function's flux.
const ψ1 = reshape(calculate_ψ_by_green_function(vec(G.R2D), vec(G.Z2D), [R_LOOP], [0.0], 1.0), size(G.R2D))
const BZ1 = zeros(size(ψ1))
for i in 2:(G.NR - 1), j in 1:G.NZ
    BZ1[i, j] = (ψ1[i + 1, j] - ψ1[i - 1, j]) / (2G.dR) / G.R1D[i]
end

const rec = (t = Float64[], Rc = Float64[], Ic = Float64[], Ic_flux = Float64[], FR = Float64[], FR_flux = Float64[])
const J_at_push = zeros(size(G.R2D))
const pushed = Ref(false)
function push_and_record(rp)
    if !pushed[] && rp.time_s >= T_PUSH - 1.0e-12
        J_at_push .= current_density(rp)
        fill!(rp.plasma.mean_ExB_R, V_PUSH)   # the radial drift of electrons and ions
        pushed[] = true
    end
    J = current_density(rp)
    push!(rec.t, rp.time_s)
    push!(rec.Rc, sum(J .* G.R2D) / sum(J))
    push!(rec.Ic, rp.coil_system.coils[1].current)
    push!(rec.Ic_flux, -flux_at_coils(rp, J)[1] / Lc)
    F1 = sum(@. J * BZ1 * 2π * G.R2D) * G.dR * G.dZ   # ∫ Jϕ B_Z dV per ampere in the loop
    push!(rec.FR, F1 * rec.Ic[end])
    push!(rec.FR_flux, F1 * rec.Ic_flux[end])
    return nothing
end
run_simulation!(RP; callback_after_step = push_and_record)
@printf(
    "column at R = %.3f m: loop current %.2f A (flux held: %.2f A), radial force %.3g N (flux held: %.3g N)\n",
    rec.Rc[end], rec.Ic[end], rec.Ic_flux[end], rec.FR[end], rec.FR_flux[end]
)

out = output_dir("coupled_step")
t = rec.t .* 1.0e3
p1 = plot(
    t, rec.Rc; c = :black, lw = 2, label = "current centroid", ylabel = "R (m)", legend = :topleft,
    title = "column pushed at 200 m/s toward a superconducting loop",
)
hline!(p1, [R_LOOP]; c = :orange, lw = 3, label = "loop at R = $(R_LOOP) m")
p2 = plot(
    t, rec.Ic_flux; c = :green, lw = 5, alpha = 0.35, label = "flux held: −Φ_p/L_c",
    ylabel = "loop current (A)", legend = :bottomleft,
)
plot!(p2, t, rec.Ic; c = :royalblue, ls = :dash, lw = 2, label = "RAPID2D")
p3 = plot(
    t, rec.FR_flux; c = :green, lw = 5, alpha = 0.35, label = "flux held",
    ylabel = "loop force on plasma, F_R (N)", xlabel = "t (ms)", legend = :bottomleft,
)
plot!(p3, t, rec.FR; c = :royalblue, ls = :dash, lw = 2, label = "RAPID2D")
hline!(p3, [0]; c = :gray, lw = 1, label = "")
l1 = plot_layout(RP; J = J_at_push, title = "t = $(T_PUSH * 1.0e3) ms, push starts")
l2 = plot_layout(RP; title = "t = $(round(t[end]; digits = 2)) ms")
fig = plot(
    l1, l2, p1, p2, p3;
    layout = @layout([grid(2, 1){0.34w} grid(3, 1)]), size = (1200, 860), margin = 4Plots.mm,
)
savefig(fig, joinpath(out, "column_pushed_toward_loop.png"))
println("outputs in ", out)
