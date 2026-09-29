# Verification problems for the coupled step: electron parallel momentum, Ampère (the
# free-boundary GS solve) and the circuits of coils and vessel filaments, from the simplest up.
# Each item isolates one mechanism and checks it against a prediction that does not come
# from the code: an exact circuit solution, a lumped circuit model, the flux a
# superconducting loop must keep, or time-step refinement.
#
#   two coils without plasma            circuit integrator, mutual inductance
#   loop beside a driven column         plasma → coil coupling (loop flux conserved)
#   coil-driven column                  coil → plasma coupling (two-circuit model)
#   density doubled at fixed drift      a density change between steps must reach the coils
#   growing density, Δt halved          the same, as consistency: the error must shrink with Δt
#   column pushed toward a loop         motion → eddy current; the loop pushes back
#   column shifted inside a shell       an ideal wall's restoring force (the VDE mechanism)
#
# @test_broken marks what the current scheme gets wrong. Two defects, both inherited from the
# MATLAB version (internal notes, issues/):
#   - coil-flux-misses-between-step-current-change: the circuit sees only the change of the
#     electron velocity within the coupled solve, never the current change made after it
#     (density, ions, field direction);
#   - coils-advance-only-in-coupled-solve: below the Ampère gate (and with ud_evolve or
#     E_para_self_EM off) the coil currents do not advance.
# They flip to failures the day the defect is fixed. Design and first measurements:
# internal notes, design/coupled-step-test-problems.md.
#
# RAPID_VISUALIZE=true writes one figure per item to a fresh temp directory (reported by @info).

@testsnippet CoupledStepSetup begin
    using RAPID2D.LinearAlgebra
    using RAPID2D.Statistics
    using Printf
    using Plots

    # A stationary column in a pure toroidal field (no poloidal field; Eϕ = E0·R̄/R), Coulomb
    # drag only. Te = 1 eV gives an L/R time near 0.23 ms, so 1 ms reaches saturation.
    function column_rp(;
            E0 = 0.3, Te = 1.0, n0 = 1.0e16, cenR = 1.5, cenZ = 0.0, radius = 0.3, threshold = 0.0,
            dt = 5.0e-6, t_end = 1.0e-3, convec = false, u_adv = false, mean_ExB = false,
        )
        config = regression_config(;
            manual = ManualSetup{Float64}(BR = 0.0, BZ = 0.0, Eϕ = E0),
            NR = 30, NZ = 50, R0B0 = 3.0, prefilled_gas_pressure = 0.0,
            dt, t_end_s = t_end, snap0D_Δt_s = 50.0e-6, snap2D_Δt_s = t_end,
        )
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(;
            Ampere = true, Ampere_Itor_threshold = threshold, E_para_self_EM = true,
            ud_evolve = true, Coulomb_Collision = true, Atomic_Collision = false, src = false,
            convec, diffu = false, Te_evolve = false, Ti_evolve = false, Gas_evolve = false,
            update_ni_independently = false, Include_ud_convec_term = u_adv, Include_ud_diffu_term = false,
            Include_ud_pressure_term = false, Include_Te_convec_term = false, E_para_self_ES = false,
            mean_ExB, turb_ExB_mixing = false, FLF_nstep = 100_000,
        )
        initialize!(RP)
        n = tophat_blob(RP.G; cenR, cenZ, radius, n0)
        RP.plasma.ne .= n
        RP.plasma.ni .= n
        fill!(RP.plasma.Te_eV, Te)
        fill!(RP.plasma.Ti_eV, 0.03)
        fill!(RP.plasma.ue_para, 0.0)
        fill!(RP.plasma.ui_para, 0.0)
        return RP
    end

    # A toroidal loop of minor radius `a`: L = μ0 R (ln(8R/a) − 7/4). R = 1e-12 Ω makes it
    # superconducting on these time scales. Call initialize_coil_system! after the last one.
    function add_loop!(RP, r, z; a = 0.05, R = 1.0e-12, V = 0.0, I0 = 0.0, name = "loop")
        L = RP.config.constants.μ0 * r * (log(8r / a) - 2 + 0.25)
        add_coil!(RP.coil_system, Coil{Float64}((r = r, z = z), π * a^2, R, L, V != 0, false, name, nothing, nothing, I0, V))
        return L
    end

    # Toroidal current density of the present state, electrons and ions: what the next
    # step starts from.
    function J_state(RP)
        p, c = RP.plasma, RP.config.constants
        Zi = Float64(RAPID2D.bulk_ion_charge(RP))
        return @. (c.qe * p.ne * p.ue_para + p.ni * (c.ee * Zi) * p.ui_para) * RP.fields.bϕ
    end
    plasma_current(RP, J) = sum(J) * RP.G.dR * RP.G.dZ

    # Plasma flux linked by each coil, 2π Σ G(r_c; r_g) J_g dA
    flux_at_coils(RP, J) = 2π .* (RP.coil_system.Green_grid2coils * vec(J)) .* (RP.G.dR * RP.G.dZ)

    # Run to t_end with a callback after every step; stdout (progress lines, Picard
    # warnings at zero current) is dropped.
    function run_quiet!(RP; after = nothing)
        redirect_stdout(devnull) do
            run_simulation!(RP; callback_after_step = after)
        end
        return RP
    end

    function save_figure(fig, name)
        outdir = mktempdir(; cleanup = false)
        path = joinpath(outdir, name)
        savefig(fig, path)
        @info "Coupled-step figure saved" path
        return path
    end
    ms(t) = t .* 1.0e3
end

@testitem "Coupled step: two coils without plasma" tags = [:regression] setup = [RegressionCommon, CoupledStepSetup] begin
    using RAPID2D.LinearAlgebra
    using Printf
    using Plots
    visualize = get(ENV, "RAPID_VISUALIZE", "false") == "true"

    # Loop A starts at 1 kA and decays; loop B, coupled to it, picks up the induced current.
    # No plasma. The coil circuits are backward Euler, so the prediction is that recursion,
    # step for step: (M + Δt R) Iⁿ⁺¹ = M Iⁿ.
    dt, t_end, I0 = 20.0e-6, 3.0e-3, 1000.0
    # (a function, not a loop body: the testitem runs at module top level, where a loop
    # cannot assign the variables around it)
    function two_coils(threshold)
        RP = column_rp(; E0 = 0.0, n0 = 0.0, threshold, dt, t_end)
        add_loop!(RP, 1.0, 0.6; R = 5.0e-3, I0, name = "A")
        add_loop!(RP, 1.2, 0.8; R = 5.0e-3, name = "B")
        initialize_coil_system!(RP)
        M, R = copy(RP.coil_system.mutual_inductance), diagm(get_all_resistances(RP.coil_system))
        I = Vector{Float64}[]
        run_quiet!(RP; after = rp -> push!(I, copy(get_all_currents(rp.coil_system))))
        return (; I, M, R)
    end
    runs = Dict(threshold => two_coils(threshold) for threshold in (0.0, 1.0))
    M, R = runs[0.0].M, runs[0.0].R
    step = (M + dt * R) \ M
    exact = accumulate((Ik, _) -> step * Ik, 1:length(runs[0.0].I); init = [I0, 0.0])
    err0 = maximum(maximum(abs.(runs[0.0].I[k] .- exact[k])) for k in eachindex(exact)) / I0

    # Ampère gate open (threshold 0): the coupled solve runs and carries the circuits.
    @test err0 < 1.0e-12
    # Default gate (1 A): no plasma current, so the coupled solve never runs and the coils
    # stay frozen (coils-advance-only-in-coupled-solve).
    @test_broken runs[1.0].I[end][1] < 0.1 * I0

    if visualize
        t = ms((1:length(exact)) .* dt)
        p1 = plot(
            t, first.(exact); c = :black, ls = :dash, label = "backward Euler, exact", ylabel = "I_A (A)",
            title = "two coils, no plasma: A decays, B is induced", lw = 2
        )
        plot!(p1, t, first.(runs[0.0].I); c = :royalblue, lw = 2, label = "RAPID2D, Ampère gate 0")
        plot!(p1, t, first.(runs[1.0].I); c = :crimson, ls = :dot, lw = 2, label = "RAPID2D, gate 1 A (default)")
        p2 = plot(t, last.(exact); c = :black, ls = :dash, lw = 2, label = "exact", ylabel = "I_B (A)", xlabel = "t (ms)")
        plot!(p2, t, last.(runs[0.0].I); c = :royalblue, lw = 2, label = "gate 0")
        plot!(p2, t, last.(runs[1.0].I); c = :crimson, ls = :dot, lw = 2, label = "gate 1 A")
        save_figure(plot(p1, p2; layout = (2, 1), size = (720, 620)), "two_coils.png")
    end
end

@testitem "Coupled step: loop flux with the density doubled" tags = [:regression] setup = [RegressionCommon, CoupledStepSetup] begin
    using RAPID2D.LinearAlgebra
    using Printf
    using Plots
    visualize = get(ENV, "RAPID_VISUALIZE", "false") == "true"
    verbose = get(ENV, "RAPID_VERBOSE", "false") == "true"

    # A column driven by the loop voltage, with a superconducting loop beside it. The loop
    # must keep its flux: L_c I_c + Φ_plasma(r_c) = 0 at every step.
    # At t1 every electron and ion is cloned with its own velocity (n → 2n). The column's own
    # flux is conserved, so the electrons slow to half and the plasma current barely
    # changes; Spitzer resistivity does not depend on n, so it stays there. The loop current
    # must not change either.
    t1 = 0.75e-3
    RP = column_rp(; t_end = 1.5e-3)
    Lc = add_loop!(RP, 1.2, 0.8)
    initialize_coil_system!(RP)
    t, Ip, Ic, Ic_pred = Float64[], Float64[], Float64[], Float64[]
    doubled = Ref(false)
    run_quiet!(
        RP; after = rp -> begin
            J = J_state(rp)
            push!(t, rp.time_s)
            push!(Ip, plasma_current(rp, J))
            push!(Ic, rp.coil_system.coils[1].current)
            push!(Ic_pred, -flux_at_coils(rp, J)[1] / Lc)
            if !doubled[] && rp.time_s >= t1 - 1.0e-12
                rp.plasma.ne .*= 2
                rp.plasma.ni .*= 2
                doubled[] = true
            end
        end
    )
    before = findall(<(t1 - 1.0e-12), t)
    later = findall(>=(t1 + 50.0e-6), t)
    err_before = maximum(abs.(Ic[before] .- Ic_pred[before])) / maximum(abs.(Ic_pred[before]))
    err_after = maximum(abs.(Ic[later] .- Ic_pred[later])) / maximum(abs.(Ic_pred[later]))
    verbose && @printf(
        "loop flux error: %.2e before the doubling, %.2e after; loop current %.2f A (expected %.2f A)\n",
        err_before, err_after, Ic[end], Ic_pred[end]
    )

    # Fixed density: the coupled solve conserves the loop's flux.
    @test err_before < 1.0e-3
    # The plasma's own flux is conserved through the doubling (its current moves < 10 %).
    k1 = before[end]
    @test abs(Ip[k1 + 2] - Ip[k1]) / Ip[k1] < 0.1
    # After the doubling the loop must still hold its flux; today it sees only the halved
    # electron velocity and drops to about zero (coil-flux-misses-between-step-current-change).
    @test_broken err_after < 1.0e-2

    if visualize
        p1 = plot(
            ms(t), Ip; c = :royalblue, lw = 2, label = "RAPID2D", ylabel = "I_plasma (A)",
            title = "density doubled at t = 0.75 ms, drift unchanged", legend = :bottomright
        )
        vline!(p1, [ms(t1)]; c = :gray, lw = 1, label = "doubling")
        p2 = plot(
            ms(t), Ic_pred; c = :green, lw = 5, alpha = 0.35, label = "expected: −(plasma flux)/L_c",
            ylabel = "loop current (A)", xlabel = "t (ms)", legend = :right
        )
        plot!(p2, ms(t), Ic; c = :royalblue, lw = 2, ls = :dash, label = "RAPID2D")
        vline!(p2, [ms(t1)]; c = :gray, lw = 1, label = "")
        save_figure(plot(p1, p2; layout = (2, 1), size = (720, 620)), "density_doubling.png")
    end
end

@testitem "Coupled step: coil-driven column" tags = [:regression] setup = [RegressionCommon, CoupledStepSetup] begin
    using RAPID2D.LinearAlgebra
    using RAPID2D.Statistics
    using Printf
    using Plots
    visualize = get(ENV, "RAPID_VISUALIZE", "false") == "true"
    verbose = get(ENV, "RAPID_VERBOSE", "false") == "true"

    # No loop voltage. A coil at (0.6, 0), outside the grid, with 10 V applied is the only
    # drive: a transformer. Prediction: two coupled circuits, the coil and a single-filament
    # plasma loop with its kinetic inductance,
    #   [L_c M; M L_p + L_kin] d/dt [I_c; I_p] = [V − R_c I_c; −R_p I_p].
    V, Rcoil = 10.0, 1.0e-4
    function driven(threshold)
        RP = column_rp(; E0 = 0.0, threshold)
        Lc = add_loop!(RP, 0.6, 0.0; a = 0.1, R = Rcoil, V, name = "OH")
        initialize_coil_system!(RP)
        t, Ip, Ic = Float64[], Float64[], Float64[]
        run_quiet!(
            RP; after = rp -> begin
                push!(t, rp.time_s)
                push!(Ip, plasma_current(rp, J_state(rp)))
                push!(Ic, rp.coil_system.coils[1].current)
            end
        )
        # The lumped plasma loop (uniform current over the column), read after the run:
        # the collision frequencies follow the initial condition only from the run's entry
        # refresh on, and with n and Te fixed they stay constant through it.
        pla, c = RP.plasma, RP.config.constants
        col = findall(>(0), pla.ne)
        R0, a = 1.5, 0.3
        Lk = c.me / (mean(pla.ne[col]) * c.ee^2) * (2π * R0 / (π * a^2))
        Lp = c.μ0 * R0 * (log(8R0 / a) - 2 + 0.25)
        Rp = mean(pla.ν_ei_eff[col]) * Lk
        Juni = zeros(size(pla.ne))
        Juni[col] .= 1.0
        model = (; Lc, M = flux_at_coils(RP, Juni)[1] / plasma_current(RP, Juni), L = Lp + Lk, Rp)
        return (; t, Ip, Ic, model)
    end
    runs = Dict(threshold => driven(threshold) for threshold in (0.0, 1.0))
    # the two-circuit model at the last output time, RK4
    function two_circuit(tend, nsteps, m)
        A = [m.Lc m.M; m.M m.L]
        f(x) = A \ [V - Rcoil * x[1], -m.Rp * x[2]]
        x = [0.0, 0.0]
        h = tend / nsteps
        for _ in 1:nsteps
            k1 = f(x); k2 = f(x .+ h / 2 .* k1); k3 = f(x .+ h / 2 .* k2); k4 = f(x .+ h .* k3)
            x = x .+ h / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        end
        return x
    end
    t = runs[0.0].t
    I_model = two_circuit(t[end], 50 * length(t), runs[0.0].model)
    verbose && @printf(
        "coil-driven column at %.2f ms: I_p = %.1f A (model %.1f A), I_coil = %.0f A (model %.0f A); gate 1 A: I_p = %.2f A\n",
        ms(t[end]), runs[0.0].Ip[end], I_model[2], runs[0.0].Ic[end], I_model[1], runs[1.0].Ip[end]
    )

    # Gate open: the plasma follows the two-circuit model (the lumped loop assumes a uniform
    # current; the coil's field falls off as 1/R across the column).
    @test abs(runs[0.0].Ip[end] - I_model[2]) < 0.05 * abs(I_model[2])
    @test abs(runs[0.0].Ic[end] - I_model[1]) < 0.02 * abs(I_model[1])
    # Default gate (1 A): the column starts at zero current, so the coupled solve, the only
    # place the coil advances, never runs: neither current moves.
    @test_broken abs(runs[1.0].Ip[end]) > 0.5 * abs(I_model[2])

    if visualize
        p1 = plot(
            ms(t), runs[0.0].Ip; c = :royalblue, lw = 2, label = "RAPID2D, Ampère gate 0", ylabel = "I_plasma (A)",
            title = "a coil (10 V) drives the column; no loop voltage", legend = :bottomleft
        )
        plot!(p1, ms(t), runs[1.0].Ip; c = :crimson, ls = :dot, lw = 2, label = "RAPID2D, gate 1 A (default)")
        hline!(p1, [I_model[2]]; c = :black, ls = :dash, label = "two-circuit model at $(round(ms(t[end]); digits = 1)) ms")
        p2 = plot(
            ms(t), runs[0.0].Ic ./ 1.0e3; c = :royalblue, lw = 2, label = "gate 0", ylabel = "I_coil (kA)",
            xlabel = "t (ms)", legend = :topleft
        )
        plot!(p2, ms(t), runs[1.0].Ic ./ 1.0e3; c = :crimson, ls = :dot, lw = 2, label = "gate 1 A")
        save_figure(plot(p1, p2; layout = (2, 1), size = (720, 620)), "coil_driven_column.png")
    end
end

@testitem "Coupled step: growing density, time step halved" tags = [:regression] setup = [RegressionCommon, CoupledStepSetup] begin
    using Printf
    verbose = get(ENV, "RAPID_VERBOSE", "false") == "true"

    # After every step the density grows by (1 + γΔt), γ = 200/s: +22 % in 1 ms, the same
    # physical history for any Δt. A superconducting loop must keep L_c I_c + Φ = 0. A
    # consistent scheme has an error that shrinks with Δt (here one step of growth, γΔt).
    γ = 200.0
    errs = Float64[]
    for dt in (5.0e-6, 2.5e-6)
        RP = column_rp(; dt)
        Lc = add_loop!(RP, 1.2, 0.8)
        initialize_coil_system!(RP)
        run_quiet!(
            RP; after = rp -> begin
                rp.plasma.ne .*= 1 + γ * rp.dt
                rp.plasma.ni .*= 1 + γ * rp.dt
            end
        )
        Φ = flux_at_coils(RP, J_state(RP))[1]
        push!(errs, abs(Lc * RP.coil_system.coils[1].current + Φ) / abs(Φ))
    end
    verbose && @printf("loop flux error at 1 ms: %.3e (Δt = 5 µs), %.3e (Δt = 2.5 µs)\n", errs...)

    # Today the error is 16.5 % at both steps: the missing flux does not depend on Δt
    # (coil-flux-misses-between-step-current-change).
    @test_broken errs[1] < 0.01 && 0.4 < errs[2] / errs[1] < 0.6
end

@testitem "Coupled step: column pushed toward a loop" tags = [:regression] setup = [RegressionCommon, CoupledStepSetup] begin
    using RAPID2D.LinearAlgebra
    using Printf
    using Plots
    visualize = get(ENV, "RAPID_VISUALIZE", "false") == "true"
    verbose = get(ENV, "RAPID_VERBOSE", "false") == "true"

    # A column (R = 1.4 m, a = 0.2 m) is driven to its saturated current, then pushed outward
    # at 200 m/s (mean_ExB_R, with u∥ advected along) toward a superconducting loop at
    # R = 1.95 m. The loop keeps its flux, so it carries a current opposite to the plasma's,
    # growing as the column approaches, and opposite currents repel: its force on the plasma
    # points back inward (F_R < 0).
    t0, v, rl = 0.6e-3, 200.0, 1.95
    RP = column_rp(; cenR = 1.4, radius = 0.2, t_end = 2.0e-3, convec = true, u_adv = true, mean_ExB = true)
    Lc = add_loop!(RP, rl, 0.0)
    initialize_coil_system!(RP)
    G = RP.G
    # the loop's B_Z per ampere on the grid: finite difference of its flux, which the Green
    # function gives exactly
    ψ1 = reshape(calculate_ψ_by_green_function(vec(G.R2D), vec(G.Z2D), [rl], [0.0], 1.0), size(G.R2D))
    BZ1 = zeros(size(ψ1))
    for i in 2:(G.NR - 1), j in 1:G.NZ
        BZ1[i, j] = (ψ1[i + 1, j] - ψ1[i - 1, j]) / (2G.dR) / G.R1D[i]
    end
    t, Rc, Ic, Ic_pred, FR = Float64[], Float64[], Float64[], Float64[], Float64[]
    pushed = Ref(false)
    run_quiet!(
        RP; after = rp -> begin
            if !pushed[] && rp.time_s >= t0 - 1.0e-12
                fill!(rp.plasma.mean_ExB_R, v)
                pushed[] = true
            end
            J = J_state(rp)
            push!(t, rp.time_s)
            push!(Rc, sum(J .* G.R2D) / sum(J))
            push!(Ic, rp.coil_system.coils[1].current)
            push!(Ic_pred, -flux_at_coils(rp, J)[1] / Lc)
            push!(FR, sum(@. J * BZ1 * 2π * G.R2D) * G.dR * G.dZ * Ic[end])
        end
    )
    err = maximum(abs.(Ic .- Ic_pred)) / maximum(abs.(Ic_pred))
    verbose && @printf(
        "column at R = %.3f m: loop current %.2f A (expected %.2f A, max flux error %.1f %%), radial force %.3g N\n",
        Rc[end], Ic[end], Ic_pred[end], 100err, FR[end]
    )

    @test Rc[end] > 1.6   # the push moved the column
    # Today the loop sees the motion only through the rigid-filament displacement term,
    # which misses the current carried into new cells: its current ends with the wrong
    # sign and it pulls the column in (coil-flux-misses-between-step-current-change).
    @test_broken err < 0.02
    @test_broken FR[end] < 0

    if visualize
        p1 = plot(
            ms(t), Rc; c = :black, lw = 2, label = "current centroid", ylabel = "R (m)", legend = :topleft,
            title = "column pushed at 200 m/s toward a superconducting loop"
        )
        hline!(p1, [rl]; c = :orange, lw = 3, label = "loop at R = $(rl) m")
        p2 = plot(ms(t), Ic_pred; c = :green, lw = 5, alpha = 0.35, label = "expected: −(plasma flux)/L_c", ylabel = "loop current (A)", legend = :bottomleft)
        plot!(p2, ms(t), Ic; c = :royalblue, ls = :dash, lw = 2, label = "RAPID2D")
        p3 = plot(ms(t), FR; c = :royalblue, ls = :dash, lw = 2, label = "RAPID2D", ylabel = "loop force F_R (N)", xlabel = "t (ms)", legend = :bottomleft)
        hline!(p3, [0]; c = :gray, lw = 1, label = "")
        save_figure(plot(p1, p2, p3; layout = (3, 1), size = (720, 860)), "column_pushed_toward_loop.png")
    end
end

@testitem "Coupled step: column shifted inside a shell" tags = [:regression] setup = [RegressionCommon, CoupledStepSetup] begin
    using RAPID2D.LinearAlgebra
    using Printf
    using Plots
    visualize = get(ENV, "RAPID_VISUALIZE", "false") == "true"
    verbose = get(ENV, "RAPID_VERBOSE", "false") == "true"

    # An ideal conducting shell (24 superconducting filaments 0.12 m outside the column,
    # up-down symmetric). At t_on the whole plasma state (n, u∥) moves up by one grid cell,
    # an exact rigid shift of J. The shell must keep its flux,
    #   ΔI = −M⁻¹ (Φ(J_shifted) − Φ(J_before)),
    # and those currents push the column back down. This restoring force is what lets a wall
    # hold a vertical displacement.
    t_on, t_after = 0.8e-3, 0.4e-3
    cenR, rs, Ns = 1.5, 0.42, 24
    θ = [2π * (k - 0.5) / Ns for k in 1:Ns]
    Rs, Zs = cenR .+ rs .* cos.(θ), rs .* sin.(θ)
    area = (2π * rs / Ns)^2
    RP = column_rp(; cenR, t_end = t_on + t_after)
    for k in 1:Ns
        add_loop!(RP, Rs[k], Zs[k]; a = sqrt(area / π), name = "shell_$k")
    end
    initialize_coil_system!(RP)
    G, pla = RP.G, RP.plasma
    dGdZ = calculate_ψ_by_green_function(vec(G.R2D), vec(G.Z2D), Rs, Zs, 1.0; compute_derivatives = true)[2].dψ_dZdest
    # vertical force of the shell currents I on the current density J: −∫ J B_R dV
    force_Z(J, I) = (BR = reshape(-(dGdZ * I), size(J)) ./ G.R2D; -sum(@. J * BR * 2π * G.R2D) * G.dR * G.dZ)
    t, F = Float64[], Float64[]
    F_pred = Ref(NaN)
    shifted = Ref(false)
    run_quiet!(
        RP; after = rp -> begin
            J = J_state(rp)
            if shifted[]
                push!(t, rp.time_s - t_on)
                push!(F, force_Z(J, get_all_currents(rp.coil_system)))
            elseif rp.time_s >= t_on - 1.0e-12
                shifted[] = true
                I_before = copy(get_all_currents(rp.coil_system))
                for A in (pla.ne, pla.ni, pla.ue_para, pla.ui_para)
                    A .= circshift(A, (0, 1))
                end
                J_shift = J_state(rp)
                ΔI = -(rp.coil_system.mutual_inductance \ (flux_at_coils(rp, J_shift) .- flux_at_coils(rp, J)))
                F_pred[] = force_Z(J_shift, I_before .+ ΔI)
            end
        end
    )
    verbose && @printf(
        "shell force: %.3e N one step after the shift, %.3e N at +%.1f ms; flux-conserving shell %.3e N\n",
        F[1], F[end], ms(t[end]), F_pred[]
    )

    @test F_pred[] < 0   # the prediction itself: a flux-conserving shell pushes back down
    # Today the shell never sees the shift (it happened between steps) and answers only the
    # plasma's velocity response: first a push the wrong way, then nothing
    # (coil-flux-misses-between-step-current-change).
    @test_broken F[1] < 0
    @test_broken abs(F[end] - F_pred[]) < 0.02 * abs(F_pred[])

    if visualize
        p = plot(
            ms(t), F; c = :royalblue, ls = :dash, lw = 2, label = "RAPID2D", xlabel = "time after the shift (ms)",
            ylabel = "shell force on the plasma, F_Z (N)", title = "column moved up 4.9 cm inside an ideal shell", legend = :right
        )
        hline!(p, [F_pred[]]; c = :green, lw = 5, alpha = 0.35, label = "flux-conserving shell")
        hline!(p, [0]; c = :gray, lw = 1, label = "")
        save_figure(plot(p; size = (720, 440)), "column_shifted_in_shell.png")
    end
end
