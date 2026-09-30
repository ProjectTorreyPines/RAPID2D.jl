# The column and superconducting loop of density_doubling.jl, with the density growing
# smoothly instead: after every step n → n(1 + γΔt), γ = 200/s, the same history for any
# Δt (+22 % in 1 ms). The loop must keep L_c I_c + Φ_p = 0. Measured on the state that a
# step hands to the next, i.e. after that step's growth, which the loop can answer only in
# the next step, a consistent scheme leaves one step of growth: an error of γΔt, which halves
# when Δt does. Two runs, Δt = 5 µs and 2.5 µs.
#
#   julia --project=examples examples/coupled_step/density_growth_dt.jl

include("common.jl")

const γ = 200.0   # density growth rate [1/s]

function growing(dt)
    RP = column("coupled_step/density_growth_dt/dt_$(dt)"; dt)
    Lc = add_loop!(RP, 1.2, 0.8)
    initialize_coil_system!(RP)
    rec = (t = Float64[], Ic = Float64[], Ic_flux = Float64[], err = Float64[])
    run_simulation!(
        RP; callback_after_step = rp -> begin
            rp.plasma.ne .*= 1 + γ * rp.dt
            rp.plasma.ni .*= 1 + γ * rp.dt
            Φ = flux_at_coils(rp, current_density(rp))[1]
            push!(rec.t, rp.time_s)
            push!(rec.Ic, rp.coil_system.coils[1].current)
            push!(rec.Ic_flux, -Φ / Lc)
            push!(rec.err, abs(Lc * rec.Ic[end] + Φ) / abs(Φ))
        end
    )
    return rec
end

runs = [dt => growing(dt) for dt in (5.0e-6, 2.5e-6)]
for (dt, rec) in runs
    @printf("Δt = %.1f µs: loop flux error at %.1f ms %.3e (one step of growth: %.1e)\n", dt * 1.0e6, rec.t[end] * 1.0e3, rec.err[end], γ * dt)
end

out = output_dir("coupled_step")
colors = [:royalblue, :darkorange]
p1 = plot(ylabel = "loop current (A)", title = "density grows 200/s; superconducting loop", legend = :topright)
p2 = plot(
    ylabel = "|L_c I_c + Φ_p| / |Φ_p|", xlabel = "t (ms)", yscale = :log10, legend = :bottomright,
    ylims = (1.0e-5, 1),
)
for ((dt, rec), c) in zip(runs, colors)
    t = rec.t .* 1.0e3
    lbl = @sprintf("Δt = %.1f µs", dt * 1.0e6)
    plot!(p1, t, rec.Ic_flux; c, lw = 5, alpha = 0.3, label = "flux held, " * lbl)
    plot!(p1, t, rec.Ic; c, lw = 2, ls = :dash, label = "RAPID2D, " * lbl)
    plot!(p2, t, rec.err; c, lw = 2, label = "RAPID2D, " * lbl)
    hline!(p2, [γ * dt]; c, ls = :dot, lw = 1.5, label = "one step of growth, γΔt")
end
savefig(plot(p1, p2; layout = (2, 1), size = (760, 680), left_margin = 4Plots.mm), joinpath(out, "density_growth_dt.png"))
println("outputs in ", out)
