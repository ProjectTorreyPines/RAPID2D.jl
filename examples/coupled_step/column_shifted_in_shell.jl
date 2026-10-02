# An ideal conducting shell: 24 superconducting filaments on a circle of radius 0.42 m around
# the column, 0.12 m outside it, up-down symmetric. Once the column carries its saturated
# current, the whole plasma state (n, u∥) moves up by one grid cell, an exact rigid shift of
# J. A flux-conserving shell answers with the currents
#   ΔI = −M⁻¹ (Φ(J_shifted) − Φ(J_before)),
# M the filaments' inductance matrix and Φ the plasma flux through each, and those currents
# push the column back down: F_Z = −∫ Jϕ B_R dV < 0. This restoring force is how a
# conducting wall holds a vertical displacement until its currents decay.
#
#   julia --project=examples examples/coupled_step/column_shifted_in_shell.jl

include("common.jl")

const T_SHIFT, T_AFTER = 0.8e-3, 0.4e-3   # shift time, run after it [s]
const CEN_R, R_SHELL, N_FIL = 1.5, 0.42, 24
const θ = [2π * (k - 0.5) / N_FIL for k in 1:N_FIL]
const R_fil, Z_fil = CEN_R .+ R_SHELL .* cos.(θ), R_SHELL .* sin.(θ)

RP = column("coupled_step/column_shifted_in_shell"; cenR = CEN_R, t_end = T_SHIFT + T_AFTER)
for k in 1:N_FIL   # filaments whose cross-sections tile the shell
    add_loop!(RP, R_fil[k], Z_fil[k]; a = 2R_SHELL / N_FIL * sqrt(π), name = "shell_$k")
end
initialize_coil_system!(RP)
const G, pla = RP.G, RP.plasma

# Vertical force of shell currents I on a current density J, with B_R = −(1/R) ∂ψ/∂Z from
# the Green function's derivative.
const dGdZ = calculate_ψ_by_green_function(vec(G.R2D), vec(G.Z2D), R_fil, Z_fil, 1.0; compute_derivatives = true)[2].dψ_dZdest
function force_Z(J, I)
    BR = reshape(-(dGdZ * I), size(J)) ./ G.R2D
    return -sum(@. J * BR * 2π * G.R2D) * G.dR * G.dZ
end

const rec = (t = Float64[], F = Float64[], I = Vector{Float64}[])
const pred = (F = Ref(NaN), I_before = zeros(N_FIL), ΔI = zeros(N_FIL))
const shifted = Ref(false)
function shift_and_record(rp)
    J = current_density(rp)
    if shifted[]
        I = copy(get_all_currents(rp.coil_system))
        push!(rec.t, rp.time_s - T_SHIFT)
        push!(rec.F, force_Z(J, I))
        push!(rec.I, I)
    elseif rp.time_s >= T_SHIFT - 1.0e-12
        shifted[] = true
        pred.I_before .= get_all_currents(rp.coil_system)
        for A in (pla.ne, pla.ni, pla.ue_para, pla.ui_para)
            A .= circshift(A, (0, 1))   # up by one cell
        end
        RAPID2D.update_transport_quantities!(rp)   # collision rates follow the plasma
        J_shifted = current_density(rp)
        pred.ΔI .= -(rp.coil_system.mutual_inductance \ (flux_at_coils(rp, J_shifted) .- flux_at_coils(rp, J)))
        pred.F[] = force_Z(J_shifted, pred.I_before .+ pred.ΔI)
    end
    return nothing
end
run_simulation!(RP; callback_after_step = shift_and_record)
@printf(
    "shell force on the plasma: %.3e N one step after the shift, %.3e N at +%.1f ms; flux-conserving shell %.3e N\n",
    rec.F[1], rec.F[end], rec.t[end] * 1.0e3, pred.F[]
)

out = output_dir("coupled_step")
p1 = plot(
    rec.t .* 1.0e3, rec.F; c = :royalblue, ls = :dash, lw = 2, label = "RAPID2D",
    xlabel = "time after the shift (ms)", ylabel = "F_Z on the plasma (N)",
    title = "column moved up $(round(100G.dZ; digits = 1)) cm inside an ideal shell", legend = :right,
)
hline!(p1, [pred.F[]]; c = :green, lw = 5, alpha = 0.35, label = "flux-conserving shell")
hline!(p1, [0]; c = :gray, lw = 1, label = "")
deg = rad2deg.(θ)
p2 = plot(
    deg, pred.ΔI; c = :green, lw = 5, alpha = 0.35, label = "flux-conserving shell",
    xlabel = "filament angle θ (deg), 0 = outboard midplane", ylabel = "shell current change ΔI (A)",
    xticks = 0:90:360, legend = :bottomright,
)
plot!(p2, deg, rec.I[1] .- pred.I_before; c = :royalblue, lw = 2, ls = :dash, m = :circle, ms = 3, label = "RAPID2D, one step after")
plot!(p2, deg, rec.I[end] .- pred.I_before; c = :navy, lw = 1, label = "RAPID2D, +$(round(rec.t[end] * 1.0e3; digits = 1)) ms")
fig = plot(
    plot_layout(RP; title = "column and shell filaments"), p1, p2;
    layout = @layout([a{0.38w} grid(2, 1)]), size = (1200, 680), margin = 4Plots.mm,
)
savefig(fig, joinpath(out, "column_shifted_in_shell.png"))
println("outputs in ", out)
