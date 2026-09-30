# A column driven by the loop voltage, with a superconducting loop beside it. At t1 every
# electron and ion is cloned with its own velocity: n → 2n at a fixed drift, which by itself
# would double the current. It does not double. Magnetic flux cannot jump, and the inductive
# E that holds it slows every electron, old and new, by the same amount: each keeps its
# canonical momentum mₑu − eA. For the circuits this means that the fluxes
#   (L_p + L_kin) I_p + M I_c    and    L_c I_c + M I_p
# are continuous, where L_kin = mₑ 2πR / (n e² πa²) is the electrons' inertia, which the
# doubling halves. So I_p rises by a factor (L_p + L_kin) / (L_p + L_kin/2) ≈ 1 + L_kin/(2L_p),
# under 2 % here; the drift drops to about half; and the loop current, −Φ_p/L_c, barely
# moves. Coulomb resistivity does not depend on n, so the current then rises to the same
# saturation as before. The plots add this lumped model, the column as one loop of uniform
# current:
#   d/dt [(L_p + L_kin) I_p + M I_c] = V − R_p I_p,    d/dt [L_c I_c + M I_p] = 0.
#
#   julia --project=examples examples/coupled_step/density_doubling.jl

include("common.jl")

const T1 = 0.75e-3   # when the density doubles [s]

RP = column("coupled_step/density_doubling"; t_end = 1.5e-3)
const Lc = add_loop!(RP, 1.2, 0.8)
initialize_coil_system!(RP)
const model = lumped_column(RP)    # at the initial density
const col = findall(>(0), RP.plasma.ne)

# After every step: the plasma current, the loop current, the current that holds the loop's
# flux (−Φ_p/L_c), and the column's mean drift. At T1, after recording, the density doubles.
const rec = (t = Float64[], Ip = Float64[], Ic = Float64[], Ic_flux = Float64[], u = Float64[])
const doubling = (t = Ref(NaN), R_p = Ref(NaN))
function record_then_double(rp)
    J = current_density(rp)
    push!(rec.t, rp.time_s)
    push!(rec.Ip, plasma_current(rp, J))
    push!(rec.Ic, rp.coil_system.coils[1].current)
    push!(rec.Ic_flux, -flux_at_coils(rp, J)[1] / Lc)
    push!(rec.u, -mean(rp.plasma.ue_para[col]))
    if isnan(doubling.t[]) && rp.time_s >= T1 - 1.0e-12
        doubling.t[], doubling.R_p[] = rp.time_s, column_resistance(rp)
        rp.plasma.ne .*= 2
        rp.plasma.ni .*= 2
        RAPID2D.update_transport_quantities!(rp)   # ν ∝ n doubles too
    end
    return nothing
end
run_simulation!(RP; callback_after_step = record_then_double)

# The lumped model from t = 0, with L_kin halved after the doubling. R_p is the same on both
# sides up to the Coulomb logarithm; each side takes its own.
const V = mean(RP.fields.LV_ext[col])
const M = model.M[1]
const R_after = column_resistance(RP)
t = vcat(0.0, rec.t)
before(τ) = τ <= doubling.t[]
L(τ) = [model.L_p + (before(τ) ? model.L_kin : model.L_kin / 2) M; M Lc]
R(τ) = [(before(τ) ? doubling.R_p[] : R_after) 0; 0 1.0e-12]
I_model = circuits(L, R, [V, 0.0], [0.0, 0.0], t)
n_model = [before(τ) ? 1 : 2 for τ in t] .* mean(RP.plasma.ne[col]) ./ 2
u_model = first.(I_model) ./ (RP.config.constants.ee .* n_model .* model.S)

k = findfirst(==(doubling.t[]), rec.t)
@printf(
    "across the doubling: I_p ×%.4f (model ×%.4f), drift ×%.3f (model ×%.3f); loop current %.2f → %.2f A (model %.2f → %.2f A)\n",
    rec.Ip[k + 1] / rec.Ip[k], I_model[k + 2][1] / I_model[k + 1][1], rec.u[k + 1] / rec.u[k], u_model[k + 2] / u_model[k + 1],
    rec.Ic[k], rec.Ic[k + 1], I_model[k + 1][2], I_model[k + 2][2]
)

out = output_dir("coupled_step")
tms = t .* 1.0e3
green = (c = :green, lw = 5, alpha = 0.35, label = "lumped model")
p1 = plot(
    tms, first.(I_model); green..., ylabel = "I_plasma (A)", legend = :bottomright,
    title = "n → 2n at t = $(T1 * 1.0e3) ms, drift unchanged",
)
plot!(p1, tms[2:end], rec.Ip; c = :royalblue, lw = 2, ls = :dash, label = "RAPID2D")
zoom = findall(τ -> 0.7e-3 <= τ <= 0.9e-3, t)
p2 = plot(
    tms[zoom], first.(I_model[zoom]); green..., ylabel = "I_plasma (A)", legend = :bottomright,
    title = "zoom: the current does not double",
)
plot!(p2, tms[zoom], rec.Ip[zoom .- 1]; c = :royalblue, lw = 2, ls = :dash, m = :circle, ms = 2, label = "RAPID2D")
p3 = plot(tms, u_model ./ 1.0e3; green..., ylabel = "electron drift −⟨u∥⟩ (km/s)", xlabel = "t (ms)", legend = :topright)
plot!(p3, tms[2:end], rec.u ./ 1.0e3; c = :royalblue, lw = 2, ls = :dash, label = "RAPID2D")
p4 = plot(tms, last.(I_model); green..., ylabel = "loop current (A)", xlabel = "t (ms)", legend = :right)
plot!(p4, tms[2:end], rec.Ic_flux; c = :gray30, lw = 1.5, ls = :dot, label = "flux held, −Φ_p/L_c")
plot!(p4, tms[2:end], rec.Ic; c = :royalblue, lw = 2, ls = :dash, label = "RAPID2D")
fig = plot(
    plot_layout(RP; title = "column and superconducting loop"), p1, p2, p3, p4;
    layout = @layout([a{0.28w} grid(2, 2)]), size = (1450, 760), margin = 5Plots.mm,
)
savefig(fig, joinpath(out, "density_doubling.png"))
println("outputs in ", out)
