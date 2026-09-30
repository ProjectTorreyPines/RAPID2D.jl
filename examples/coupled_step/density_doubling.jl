# A column driven by the loop voltage. At t1 every electron and ion is cloned with its own
# velocity: n → 2n at a fixed drift, which by itself would double the current. It does not
# double: the column's self-inductance holds its flux, so the drift drops to about half
# instead. The electrons' inertia adds a kinetic inductance L_kin = mₑ 2πR / (n e² πa²) to the
# magnetic L_p, and the doubling halves it. Each electron keeps its canonical momentum
# mₑu − eA through the jump, so what stays continuous is the flux (L_p + L_kin) I_p:
#   I_p⁺ / I_p⁻ = (L_p + L_kin) / (L_p + L_kin/2) ≈ 1 + L_kin/(2L_p),
# under 2 % here and zero for massless electrons. Coulomb resistivity does not depend on n,
# so the current then rises to the same saturation as before.
#
# Two runs: the column alone, and with a superconducting loop beside it. The loop keeps its
# flux, L_c I_c + M I_p, so its current barely moves either; the jump keeps both fluxes. The
# plots add the lumped model, the column as one loop of uniform current,
#   d/dt [(L_p + L_kin) I_p + M I_c] = V − R_p I_p,    d/dt [L_c I_c + M I_p] = 0,
# with M = 0 and no second equation for the column alone.
#
#   julia --project=examples examples/coupled_step/density_doubling.jl

include("common.jl")

const T1 = 0.75e-3   # when the density doubles [s]

# One run, with or without the loop. After every step it records the plasma current, the
# column's mean drift and, with the loop, the loop current and the current that holds the
# loop's flux (−Φ_p/L_c). At T1, after recording, the density doubles.
function doubling_run(name; loop::Bool)
    RP = column("coupled_step/density_doubling/$name"; t_end = 1.5e-3)
    Lc = loop ? add_loop!(RP, 1.2, 0.8) : NaN
    loop && initialize_coil_system!(RP)
    model = lumped_column(RP)   # at the initial density
    col = findall(>(0), RP.plasma.ne)
    rec = (t = Float64[], Ip = Float64[], u = Float64[], Ic = Float64[], Ic_flux = Float64[])
    doubling = (t = Ref(NaN), R_p = Ref(NaN))
    run_simulation!(
        RP; callback_after_step = rp -> begin
            J = current_density(rp)
            push!(rec.t, rp.time_s)
            push!(rec.Ip, plasma_current(rp, J))
            push!(rec.u, -mean(rp.plasma.ue_para[col]))
            if loop
                push!(rec.Ic, rp.coil_system.coils[1].current)
                push!(rec.Ic_flux, -flux_at_coils(rp, J)[1] / Lc)
            end
            if isnan(doubling.t[]) && rp.time_s >= T1 - 1.0e-12
                doubling.t[], doubling.R_p[] = rp.time_s, column_resistance(rp)
                rp.plasma.ne .*= 2
                rp.plasma.ni .*= 2
                RAPID2D.update_transport_quantities!(rp)   # ν ∝ n doubles too
            end
        end
    )
    return (;
        RP, rec, model, Lc, t_d = doubling.t[], R_before = doubling.R_p[], R_after = column_resistance(RP),
        V = mean(RP.fields.LV_ext[col]), n0 = mean(RP.plasma.ne[col]) / 2,
    )
end

# The lumped model from t = 0 over a run's times: L_kin halves at the doubling, and R_p is the
# column's on each side (equal up to the Coulomb logarithm).
function lumped_doubling(r)
    before(τ) = τ <= r.t_d
    L_col(τ) = r.model.L_p + (before(τ) ? r.model.L_kin : r.model.L_kin / 2)
    R_col(τ) = before(τ) ? r.R_before : r.R_after
    t = vcat(0.0, r.rec.t)
    if isempty(r.model.M)   # the column alone
        I = circuits(τ -> fill(L_col(τ), 1, 1), τ -> fill(R_col(τ), 1, 1), [r.V], [0.0], t)
        Ip, Ic = first.(I), Float64[]
    else
        M = r.model.M[1]
        I = circuits(τ -> [L_col(τ) M; M r.Lc], τ -> [R_col(τ) 0; 0 1.0e-12], [r.V, 0.0], [0.0, 0.0], t)
        Ip, Ic = first.(I), last.(I)
    end
    u = Ip ./ (r.RP.config.constants.ee * r.model.S .* [before(τ) ? r.n0 : 2r.n0 for τ in t])
    return (; t, Ip, Ic, u)
end

alone = doubling_run("alone"; loop = false)
looped = doubling_run("with_loop"; loop = true)
models = map(lumped_doubling, (alone, looped))

for (label, r, m) in zip(("column alone", "with the loop"), (alone, looped), models)
    k = findfirst(==(r.t_d), r.rec.t)   # rec index of the doubling; the model's is k + 1
    @printf(
        "%-14s across the doubling: I_p ×%.4f (model ×%.4f), drift ×%.3f (model ×%.3f)",
        label, r.rec.Ip[k + 1] / r.rec.Ip[k], m.Ip[k + 2] / m.Ip[k + 1], r.rec.u[k + 1] / r.rec.u[k], m.u[k + 2] / m.u[k + 1]
    )
    isempty(m.Ic) || @printf("; loop current %.2f → %.2f A (model %.2f → %.2f A)", r.rec.Ic[k], r.rec.Ic[k + 1], m.Ic[k + 1], m.Ic[k + 2])
    println()
end

out = output_dir("coupled_step")
cases = (
    (alone, models[1], :darkorange, "column alone"),
    (looped, models[2], :royalblue, "with the loop"),
)
model_line(c) = (; c, lw = 5, alpha = 0.35)
sim_line(c) = (; c, lw = 2, ls = :dash)
p1 = plot(ylabel = "I_plasma (A)", legend = :bottomright, title = "n → 2n at t = $(T1 * 1.0e3) ms, drift unchanged")
p2 = plot(ylabel = "I_plasma (A)", legend = :bottomright, ylims = (138, 157), title = "zoom: the current does not double")
p3 = plot(ylabel = "electron drift −⟨u∥⟩ (km/s)", xlabel = "t (ms)", legend = :bottomright)
zoom = findall(τ -> 0.7e-3 <= τ <= 0.9e-3, models[1].t)
for (r, m, c, label) in cases
    tms = m.t .* 1.0e3
    plot!(p1, tms, m.Ip; model_line(c)..., label = "$label, lumped model")
    plot!(p1, tms[2:end], r.rec.Ip; sim_line(c)..., label = "$label, RAPID2D")
    plot!(p2, tms[zoom], m.Ip[zoom]; model_line(c)..., label = "$label, lumped model")
    plot!(p2, tms[zoom], r.rec.Ip[zoom .- 1]; sim_line(c)..., m = :circle, ms = 2, label = "$label, RAPID2D")
    plot!(p3, tms, m.u ./ 1.0e3; model_line(c)..., label = "$label, lumped model")
    plot!(p3, tms[2:end], r.rec.u ./ 1.0e3; sim_line(c)..., label = "$label, RAPID2D")
end
m = models[2]
tms = m.t .* 1.0e3
p4 = plot(
    tms, m.Ic; model_line(:royalblue)..., label = "lumped model",
    ylabel = "loop current (A)", xlabel = "t (ms)", legend = :right, title = "the loop (second run)",
)
plot!(p4, tms[2:end], looped.rec.Ic_flux; c = :gray30, lw = 1.5, ls = :dot, label = "flux held, −Φ_p/L_c")
plot!(p4, tms[2:end], looped.rec.Ic; sim_line(:royalblue)..., label = "RAPID2D")
fig = plot(
    plot_layout(looped.RP; title = "column; the loop is in the second run only"), p1, p2, p3, p4;
    layout = @layout([a{0.25w} grid(2, 2)]), size = (1500, 780), margin = 5Plots.mm,
)
savefig(fig, joinpath(out, "density_doubling.png"))
println("outputs in ", out)
