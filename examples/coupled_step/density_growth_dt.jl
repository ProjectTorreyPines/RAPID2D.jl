# The column and superconducting loop of density_doubling.jl, with the density growing
# smoothly instead: after every step n → n(1 + γΔt), γ = 200/s, the same history for any
# Δt (+22 % in 1 ms). The loop must keep L_c I_c + Φ_p = 0. Measured on the state a step
# hands to the next, i.e. after that step's growth, which the loop can answer only in the
# next step, a consistent scheme leaves one step of growth: an error of γΔt, which halves
# when Δt does. Two runs, Δt = 5 µs and 2.5 µs. The loop-current plot adds the lumped model
# of density_doubling.jl, with the electrons' inertia L_kin ∝ 1/n falling as e^(−γt):
#   d/dt [(L_p + L_kin) I_p + M I_c] = V − R_p I_p,    d/dt [L_c I_c + M I_p] = 0.
#
#   julia --project=examples examples/coupled_step/density_growth_dt.jl

include("common.jl")

const γ = 200.0   # density growth rate [1/s]

function growing(dt)
    RP = column("coupled_step/density_growth_dt/dt_$(dt)"; dt)
    Lc = add_loop!(RP, 1.2, 0.8)
    initialize_coil_system!(RP)
    model = lumped_column(RP)   # at the initial density
    col = findall(>(0), RP.plasma.ne)
    rec = (t = Float64[], Ic = Float64[], Ic_flux = Float64[], err = Float64[], R_p = Float64[])
    run_simulation!(
        RP; callback_after_step = rp -> begin
            push!(rec.R_p, column_resistance(rp))   # of the step just taken
            rp.plasma.ne .*= 1 + γ * rp.dt
            rp.plasma.ni .*= 1 + γ * rp.dt
            RAPID2D.update_transport_quantities!(rp)
            Φ = flux_at_coils(rp, current_density(rp))[1]
            push!(rec.t, rp.time_s)
            push!(rec.Ic, rp.coil_system.coils[1].current)
            push!(rec.Ic_flux, -Φ / Lc)
            push!(rec.err, abs(Lc * rec.Ic[end] + Φ) / abs(Φ))
        end
    )
    return (; rec, model, Lc, V = mean(RP.fields.LV_ext[col]))
end

# the lumped model over one run's times, with R_p as that run's column had it
function lumped(run)
    (; rec, model, Lc, V) = run
    M = model.M[1]
    R_p(τ) = rec.R_p[clamp(searchsortedfirst(rec.t, τ), 1, length(rec.t))]
    L(τ) = [model.L_p + model.L_kin * exp(-γ * τ) M; M Lc]
    R(τ) = [R_p(τ) 0; 0 1.0e-12]
    t = vcat(0.0, rec.t)
    return t, circuits(L, R, [V, 0.0], [0.0, 0.0], t)
end

runs = [dt => growing(dt) for dt in (5.0e-6, 2.5e-6)]
t_model, I_model = lumped(last(runs[1]))
for (dt, run) in runs
    @printf(
        "Δt = %.1f µs: loop flux error at %.1f ms %.3e (one step of growth: %.1e); loop current %.2f A (lumped model %.2f A)\n",
        dt * 1.0e6, run.rec.t[end] * 1.0e3, run.rec.err[end], γ * dt, run.rec.Ic[end], I_model[end][2]
    )
end

out = output_dir("coupled_step")
colors = [:royalblue, :darkorange]
p1 = plot(
    t_model .* 1.0e3, last.(I_model); c = :black, lw = 1.5, ls = :dot, label = "lumped model",
    ylabel = "loop current (A)", title = "density grows 200/s; superconducting loop", legend = :topright,
)
p2 = plot(
    ylabel = "|L_c I_c + Φ_p| / |Φ_p|", xlabel = "t (ms)", yscale = :log10, legend = :bottomright,
    ylims = (1.0e-5, 1),
)
for ((dt, run), c) in zip(runs, colors)
    t = run.rec.t .* 1.0e3
    lbl = @sprintf("Δt = %.1f µs", dt * 1.0e6)
    plot!(p1, t, run.rec.Ic_flux; c, lw = 5, alpha = 0.3, label = "flux held, " * lbl)
    plot!(p1, t, run.rec.Ic; c, lw = 2, ls = :dash, label = "RAPID2D, " * lbl)
    plot!(p2, t, run.rec.err; c, lw = 2, label = "RAPID2D, " * lbl)
    hline!(p2, [γ * dt]; c, ls = :dot, lw = 1.5, label = "one step of growth, γΔt")
end
savefig(plot(p1, p2; layout = (2, 1), size = (760, 680), left_margin = 4Plots.mm), joinpath(out, "density_growth_dt.png"))
println("outputs in ", out)
