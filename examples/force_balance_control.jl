# Hoop force and vertical-field position control. A current-carrying blob in a pure toroidal
# field is pushed outward by its own J×B force (Global_JxB_Force). The second run imposes a
# curved vertical field, scaled to the blob's self-field and PID-modulated on the current
# centroid, through the external field between steps (callback_after_step of run_simulation!).
#
#   julia --project=examples examples/force_balance_control.jl

include("common.jl")

const R_BLOB, A_BLOB, N_BLOB = 1.3, 0.2, 1.0e16   # centre [m], radius [m], peak density [m⁻³]

# BZ_ctrl = −ctrl_mag, BR_ctrl = ctrl_mag·sin(atan(Z/2, R − 0.01)); ctrl_mag = |Jϕ|-weighted
# self-field at the column, times (1 − PID(R_cen)). Applied via BR_ext/BZ_ext, which the
# manual device leaves untouched, so the next step sees it in the total field.
Base.@kwdef mutable struct CurvedBzControl
    target::Float64
    dt::Float64
    Kp::Float64 = 1.0
    Ki::Float64 = 0.1
    Kd::Float64 = 0.0
    int_err::Float64 = 0.0
    prev_err::Float64 = 0.0
    t::Vector{Float64} = Float64[]
    Bz_ctrl::Vector{Float64} = Float64[]
    R_cen::Vector{Float64} = Float64[]
end

function (c::CurvedBzControl)(RP::RAPID)
    G, F, pla = RP.G, RP.fields, RP.plasma
    w = abs.(pla.Jϕ) .* G.inVol2D
    sw = sum(w)
    ctrl_mag = sw > 0 ? max(sum(w .* F.BZ_self) / sw, sum(w .* F.BR_self) / sw) : 1.0e-8
    # Jϕ-weighted centroid in R; with no current there is no centroid, and nothing to correct
    I_sum = sum(pla.Jϕ)
    R_cen = iszero(I_sum) ? NaN : sum(pla.Jϕ .* G.R2D) / I_sum
    if RP.step >= 5 && !isnan(R_cen)
        err = c.target - R_cen
        c.int_err += err * c.dt
        pid = c.Kp * err + c.Ki * c.int_err + c.Kd * (err - c.prev_err) / c.dt
        c.prev_err = err
        ctrl_mag *= (1 - pid)
    end
    @. F.BZ_ext = -ctrl_mag
    @. F.BR_ext = ctrl_mag * sin(atan(G.Z2D / 2, G.R2D - 0.01))
    push!(c.t, RP.time_s)
    push!(c.Bz_ctrl, -ctrl_mag)
    push!(c.R_cen, R_cen)
    return nothing
end

function blob_run(name; control::Union{Nothing, CurvedBzControl})
    config = SimulationConfig{Float64}(
        device_Name = "manual", manual = pure_toroidal(0.3), NR = 30, NZ = 50, R0B0 = 3.0,
        prefilled_gas_pressure = 1.0e-3,
        dt = 5.0e-6, t_end_s = 2.0e-3, snap0D_Δt_s = 10.0e-6, snap2D_Δt_s = 20.0e-6,
        Output_path = output_dir(name),
    )
    RP = setup(
        config;
        Global_JxB_Force = true,                # the hoop force
        # Benchmark setting: the blob starts formed at 0 A, so Ampère runs from the first step;
        # under a gate its first step would accelerate freely, without its self-inductance.
        Ampere = true, Ampere_Itor_threshold = 0.0, E_para_self_EM = true,
        ud_evolve = true, convec = true, Coulomb_Collision = true, Atomic_Collision = true,
        src = false, diffu = false, Te_evolve = false, Ti_evolve = false, Gas_evolve = false,
        update_ni_independently = true, Include_Te_convec_term = true,
        E_para_self_ES = false, mean_ExB = false, turb_ExB_mixing = false,
        FLF_nstep = 10,
    )
    set_column!(RP, masked_gaussian(RP.G; cenR = R_BLOB, radius = A_BLOB, n0 = N_BLOB))
    isnothing(control) || (control.dt = RP.dt)
    run!(RP; callback_after_step = control)
    summarize(RP)
    return RP
end

RP_free = blob_run("force_balance_control/no_ctrl"; control = nothing)
ctrl = CurvedBzControl(target = 1.5, dt = 5.0e-6)
RP_ctrl = blob_run("force_balance_control/with_ctrl"; control = ctrl)

out = output_dir("force_balance_control")
wall_R = extrema(RP_free.wall.R)

# current-centroid trajectory vs target and wall
s_free, s_ctrl = RP_free.diagnostics.snaps0D, RP_ctrl.diagnostics.snaps0D
p1 = plot(xlabel = "t (ms)", ylabel = "J-centroid R (m)", legend = :bottomright, ylims = wall_R .+ (-0.05, 0.05))
plot!(p1, s_free.time_s .* 1.0e3, s_free.J_cen_R; lw = 2, label = "no ctrl")
plot!(p1, s_ctrl.time_s .* 1.0e3, s_ctrl.J_cen_R; lw = 2, label = "with ctrl")
hline!(p1, [ctrl.target]; ls = :dash, c = :black, label = "target")
hline!(p1, collect(wall_R); lw = 4, c = :gray, alpha = 0.5, label = "wall")

# controller vertical field vs the analytic B_v of a uniform-current loop (li = 1)
μ0 = RP_ctrl.config.constants.μ0
Bv = @. -(μ0 * s_ctrl.I_tor / (4π * s_ctrl.J_cen_R)) * (log(8 * s_ctrl.J_cen_R / A_BLOB) - 1)
p2 = plot(xlabel = "t (ms)", ylabel = "B_Z (G)", legend = :topright)
plot!(p2, ctrl.t .* 1.0e3, ctrl.Bz_ctrl .* 1.0e4; lw = 2, label = "Bz_ctrl")
plot!(p2, s_ctrl.time_s .* 1.0e3, Bv .* 1.0e4; lw = 2, ls = :dash, c = :black, label = "analytic B_v (li = 1)")
savefig(plot(p1, p2; layout = (2, 1), size = (700, 800)), joinpath(out, "position_control.png"))

animate2D(["no ctrl" => RP_free, "with ctrl" => RP_ctrl], [:ne, :Jϕ]; file = joinpath(out, "snaps2D.mp4"))
println("outputs in ", out)
