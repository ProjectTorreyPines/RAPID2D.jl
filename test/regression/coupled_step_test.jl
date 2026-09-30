# The coupled step (electron parallel momentum, Ampère's free-boundary GS solve, and the
# circuits of coils and vessel filaments) against predictions that do not come from the
# code: an exact circuit solution, a lumped circuit model, the flux a superconducting loop
# must keep, and time-step refinement. examples/coupled_step/ has the same problems with
# their physics written out and plotted.
#
# @test_broken marks two defects of the current scheme, both inherited from MATLAB (internal
# notes, issues/); they flip to failures the day the defect is fixed.
#   - coil-flux-misses-between-step-current-change: the circuits see the change of the
#     electron velocity inside the coupled solve, never a current change made between solves
#     (density, ions, motion).
#   - coils-advance-only-in-coupled-solve: below the Ampère threshold the coil currents do
#     not advance.

@testsnippet CoupledStepSetup begin
    # A uniform column at rest in a pure toroidal field (Eϕ = E0·R̄/R), Coulomb drag only;
    # Te = 1 eV puts its L/R time near 0.2 ms. `moving` turns on the mean E×B drift and the
    # advection of n and u∥ by it.
    function column(;
            E0 = 0.3, Te = 1.0, n0 = 1.0e16, cenR = 1.5, cenZ = 0.0, radius = 0.3, threshold = 0.0,
            dt = 5.0e-6, t_end = 1.0e-3, moving = false,
        )
        config = regression_config(;
            manual = ManualSetup{Float64}(BR = 0.0, BZ = 0.0, Eϕ = E0),
            NR = 30, NZ = 50, R0B0 = 3.0, prefilled_gas_pressure = 0.0,
            dt, t_end_s = t_end, snap0D_Δt_s = 10dt, snap2D_Δt_s = t_end,
        )
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(;
            Ampere = true, Ampere_Itor_threshold = threshold, E_para_self_EM = true,
            ud_evolve = true, Coulomb_Collision = true, Atomic_Collision = false, src = false,
            convec = moving, diffu = false, Te_evolve = false, Ti_evolve = false, Gas_evolve = false,
            update_ni_independently = false, Include_ud_convec_term = moving, Include_ud_diffu_term = false,
            Include_ud_pressure_term = false, Include_Te_convec_term = false, E_para_self_ES = false,
            mean_ExB = moving, turb_ExB_mixing = false, FLF_nstep = 100_000,
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

    # A toroidal loop of minor radius `a`, L = μ0 r (ln(8r/a) − 7/4); returns L. R = 1e-12 Ω
    # is superconducting on these time scales. Call initialize_coil_system! after the last one.
    function add_loop!(RP, r, z; a = 0.05, R = 1.0e-12, V = 0.0, I0 = 0.0, name = "loop")
        L = RP.config.constants.μ0 * r * (log(8r / a) - 7 / 4)
        coil = Coil{Float64}(;
            location = (r = r, z = z), area = π * a^2, resistance = R, self_inductance = L,
            is_powered = V != 0, is_controllable = false, name, current = I0, voltage_ext = V,
        )
        add_coil!(RP.coil_system, coil)
        return L
    end

    # Toroidal current density of the present state, electrons and ions.
    function current_density(RP)
        p, c = RP.plasma, RP.config.constants
        Zi = Float64(RAPID2D.bulk_ion_charge(RP))
        return @. (c.qe * p.ne * p.ue_para + p.ni * (c.ee * Zi) * p.ui_para) * RP.fields.bϕ
    end
    plasma_current(RP, J) = sum(J) * RP.G.dR * RP.G.dZ

    # Plasma flux through each loop, 2π Σ G(r_loop; r) J dA.
    flux_at_coils(RP, J) = 2π .* (RP.coil_system.Green_grid2coils * vec(J)) .* (RP.G.dR * RP.G.dZ)

    # Run to t_end with a callback after every step; progress lines are dropped.
    function run_quiet!(RP; after = nothing)
        redirect_stdout(devnull) do
            run_simulation!(RP; callback_after_step = after)
        end
        return RP
    end
end

@testitem "Coupled step: two coils without plasma" tags = [:regression] setup = [RegressionCommon, CoupledStepSetup] begin
    # Loop A (1 kA) decays and induces a current in loop B. The coil circuits are backward
    # Euler, so the answer is that recursion, step for step: (M + Δt R) Iⁿ⁺¹ = M Iⁿ.
    dt, I0 = 20.0e-6, 1000.0
    function two_coils(threshold)
        RP = column(; E0 = 0.0, n0 = 0.0, threshold, dt, t_end = 3.0e-3)
        add_loop!(RP, 1.0, 0.6; R = 5.0e-3, I0, name = "A")
        add_loop!(RP, 1.2, 0.8; R = 5.0e-3, name = "B")
        initialize_coil_system!(RP)
        I = Vector{Float64}[]
        run_quiet!(RP; after = rp -> push!(I, copy(get_all_currents(rp.coil_system))))
        return RP, I
    end
    RP, I_open = two_coils(0.0)
    _, I_default = two_coils(1.0)
    M, r = RP.coil_system.mutual_inductance, get_all_resistances(RP.coil_system)
    step = (M + dt * [r[1] 0; 0 r[2]]) \ M
    exact = accumulate((I, _) -> step * I, eachindex(I_open); init = [I0, 0.0])

    @test maximum(maximum(abs.(I_open[k] .- exact[k])) for k in eachindex(exact)) < 1.0e-12 * I0
    # Without plasma the 1 A threshold has nothing to act on, yet the coils stay frozen.
    @test_broken I_default[end][1] < 0.1 * I0
end

@testitem "Coupled step: loop flux with the density doubled" tags = [:regression] setup = [RegressionCommon, CoupledStepSetup] begin
    # A driven column beside a superconducting loop, which must keep L_c I_c + Φ_p = 0. At t1
    # every electron and ion is cloned with its own velocity (n → 2n): the column keeps its own
    # flux, so its current barely moves, and the loop current must not move either.
    t1 = 0.75e-3
    RP = column(; t_end = 1.5e-3)
    Lc = add_loop!(RP, 1.2, 0.8)
    initialize_coil_system!(RP)
    t, Ip, Ic, Ic_flux = Float64[], Float64[], Float64[], Float64[]
    doubled = Ref(false)
    run_quiet!(
        RP; after = rp -> begin
            J = current_density(rp)
            push!(t, rp.time_s)
            push!(Ip, plasma_current(rp, J))
            push!(Ic, rp.coil_system.coils[1].current)
            push!(Ic_flux, -flux_at_coils(rp, J)[1] / Lc)
            if !doubled[] && rp.time_s >= t1 - 1.0e-12
                rp.plasma.ne .*= 2
                rp.plasma.ni .*= 2
                doubled[] = true
            end
        end
    )
    flux_error(ks) = maximum(abs.(Ic[ks] .- Ic_flux[ks])) / maximum(abs.(Ic_flux[ks]))
    before, later = findall(<(t1 - 1.0e-12), t), findall(>=(t1 + 50.0e-6), t)
    k1 = before[end]

    @test flux_error(before) < 1.0e-3
    @test abs(Ip[k1 + 2] - Ip[k1]) / Ip[k1] < 0.1
    # The loop sees only the halved drift and loses its current.
    @test_broken flux_error(later) < 1.0e-2
end

@testitem "Coupled step: coil-driven column" tags = [:regression] setup = [RegressionCommon, CoupledStepSetup] begin
    using RAPID2D.Statistics
    # No loop voltage; a 10 V coil at R = 0.6 m drives the column. Prediction: the coil and a
    # single-filament plasma loop (uniform current, with the electrons' kinetic inductance),
    #   [L_c M; M L_p + L_kin] d/dt [I_c; I_p] = [V − R_c I_c; −R_p I_p],   R_p = ν L_kin.
    V, Rcoil, R0, a = 10.0, 1.0e-4, 1.5, 0.3
    function driven(threshold)
        RP = column(; E0 = 0.0, cenR = R0, radius = a, threshold)
        Lc = add_loop!(RP, 0.6, 0.0; a = 0.1, R = Rcoil, V, name = "OH")
        initialize_coil_system!(RP)
        run_quiet!(RP)
        return RP, Lc
    end
    # the model at t_end, RK4; read after the run (n and Te are fixed, so ν is constant)
    function two_circuit(RP, Lc; nsteps = 10_000)
        pla, c = RP.plasma, RP.config.constants
        col = findall(>(0), pla.ne)
        Lk = c.me / (mean(pla.ne[col]) * c.ee^2) * (2π * R0 / (π * a^2))
        Juni = zeros(size(pla.ne))
        Juni[col] .= 1.0
        M = flux_at_coils(RP, Juni)[1] / plasma_current(RP, Juni)
        A = [Lc M; M (c.μ0 * R0 * (log(8R0 / a) - 7 / 4) + Lk)]
        Rp = mean(pla.ν_ei_eff[col]) * Lk
        f(x) = A \ [V - Rcoil * x[1], -Rp * x[2]]
        x, h = [0.0, 0.0], RP.time_s / nsteps
        for _ in 1:nsteps
            k1 = f(x)
            k2 = f(x .+ h / 2 .* k1)
            k3 = f(x .+ h / 2 .* k2)
            k4 = f(x .+ h .* k3)
            x = x .+ h / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        end
        return x
    end
    RP, Lc = driven(0.0)
    RP_default, _ = driven(1.0)
    Ic_model, Ip_model = two_circuit(RP, Lc)
    Ip = plasma_current(RP, current_density(RP))

    # The lumped loop assumes a uniform current; the coil's field falls off across the column.
    @test abs(Ip - Ip_model) < 0.05 * abs(Ip_model)
    @test abs(RP.coil_system.coils[1].current - Ic_model) < 0.02 * abs(Ic_model)
    # The column starts at zero current, under the 1 A threshold: neither current moves.
    @test_broken abs(plasma_current(RP_default, current_density(RP_default))) > 0.5 * abs(Ip_model)
end

@testitem "Coupled step: growing density, time step halved" tags = [:regression] setup = [RegressionCommon, CoupledStepSetup] begin
    # After every step n → n(1 + γΔt), γ = 200/s: the same history for any Δt. The
    # superconducting loop must keep L_c I_c + Φ_p = 0. On the state a step hands to the next,
    # a consistent scheme is off by that step's growth, γΔt, which halves with Δt.
    γ = 200.0
    function flux_error(dt)
        RP = column(; dt)
        Lc = add_loop!(RP, 1.2, 0.8)
        initialize_coil_system!(RP)
        run_quiet!(
            RP; after = rp -> begin
                rp.plasma.ne .*= 1 + γ * rp.dt
                rp.plasma.ni .*= 1 + γ * rp.dt
            end
        )
        Φ = flux_at_coils(RP, current_density(RP))[1]
        return abs(Lc * RP.coil_system.coils[1].current + Φ) / abs(Φ)
    end
    errs = flux_error.((5.0e-6, 2.5e-6))

    # The missing flux does not shrink with Δt.
    @test_broken errs[1] < 0.01 && 0.4 < errs[2] / errs[1] < 0.6
end

@testitem "Coupled step: column pushed toward a loop" tags = [:regression] setup = [RegressionCommon, CoupledStepSetup] begin
    # A column (R = 1.4 m, a = 0.2 m) at its saturated current is pushed outward at 200 m/s
    # toward a superconducting loop at R = 2.3 m, outside the wall. The loop keeps its flux, so
    # it carries a current opposite to the plasma's and pushes the column back (F_R < 0).
    t0, v, rl = 0.6e-3, 200.0, 2.3
    RP = column(; cenR = 1.4, radius = 0.2, t_end = 2.0e-3, moving = true)
    Lc = add_loop!(RP, rl, 0.0)
    initialize_coil_system!(RP)
    G = RP.G
    # the loop's B_Z per ampere, (1/R) ∂ψ/∂R of its Green-function flux
    ψ1 = reshape(calculate_ψ_by_green_function(vec(G.R2D), vec(G.Z2D), [rl], [0.0], 1.0), size(G.R2D))
    BZ1 = zeros(size(ψ1))
    BZ1[2:(end - 1), :] .= (ψ1[3:end, :] .- ψ1[1:(end - 2), :]) ./ (2G.dR) ./ G.R1D[2:(end - 1)]
    Rc, Ic, Ic_flux, F1 = Float64[], Float64[], Float64[], Float64[]
    pushed = Ref(false)
    run_quiet!(
        RP; after = rp -> begin
            if !pushed[] && rp.time_s >= t0 - 1.0e-12
                fill!(rp.plasma.mean_ExB_R, v)
                pushed[] = true
            end
            J = current_density(rp)
            push!(Rc, sum(J .* G.R2D) / sum(J))
            push!(Ic, rp.coil_system.coils[1].current)
            push!(Ic_flux, -flux_at_coils(rp, J)[1] / Lc)
            push!(F1, sum(@. J * BZ1 * 2π * G.R2D) * G.dR * G.dZ)   # F_R per ampere in the loop
        end
    )

    @test Rc[end] > 1.6                # the push moved the column
    @test F1[end] * Ic_flux[end] < 0   # a flux-conserving loop pushes it back
    # The loop sees the motion only through the rigid-filament term, which misses the current
    # carried into new cells: it keeps almost none of its flux.
    @test_broken maximum(abs.(Ic .- Ic_flux)) < 0.02 * maximum(abs.(Ic_flux))
end

@testitem "Coupled step: column shifted inside a shell" tags = [:regression] setup = [RegressionCommon, CoupledStepSetup] begin
    # An ideal shell of 24 superconducting filaments, 0.12 m outside the column. At t_on the
    # plasma state moves up one cell, a rigid shift of J. A flux-conserving shell answers with
    # ΔI = −M⁻¹ (Φ(J_shifted) − Φ(J_before)), which pushes the column back down.
    t_on, cenR, rs, Ns = 0.8e-3, 1.5, 0.42, 24
    θ = [2π * (k - 0.5) / Ns for k in 1:Ns]
    Rs, Zs = cenR .+ rs .* cos.(θ), rs .* sin.(θ)
    RP = column(; cenR, t_end = t_on + 0.4e-3)
    for k in 1:Ns
        add_loop!(RP, Rs[k], Zs[k]; a = 2rs / Ns * sqrt(π), name = "shell_$k")
    end
    initialize_coil_system!(RP)
    G, pla = RP.G, RP.plasma
    dGdZ = calculate_ψ_by_green_function(vec(G.R2D), vec(G.Z2D), Rs, Zs, 1.0; compute_derivatives = true)[2].dψ_dZdest
    # vertical force of shell currents I on J: −∫ Jϕ B_R dV, B_R = −(1/R) ∂ψ/∂Z
    function force_Z(J, I)
        BR = reshape(-(dGdZ * I), size(J)) ./ G.R2D
        return -sum(@. J * BR * 2π * G.R2D) * G.dR * G.dZ
    end
    F = Float64[]
    F_pred = Ref(NaN)
    shifted = Ref(false)
    run_quiet!(
        RP; after = rp -> begin
            J = current_density(rp)
            if shifted[]
                push!(F, force_Z(J, get_all_currents(rp.coil_system)))
            elseif rp.time_s >= t_on - 1.0e-12
                shifted[] = true
                I_before = copy(get_all_currents(rp.coil_system))
                for A in (pla.ne, pla.ni, pla.ue_para, pla.ui_para)
                    A .= circshift(A, (0, 1))
                end
                J_shifted = current_density(rp)
                ΔI = -(rp.coil_system.mutual_inductance \ (flux_at_coils(rp, J_shifted) .- flux_at_coils(rp, J)))
                F_pred[] = force_Z(J_shifted, I_before .+ ΔI)
            end
        end
    )

    @test F_pred[] < 0
    # The shift happened between steps, so the shell never sees it: it answers only the
    # plasma's velocity response, first pushing the wrong way, then not at all.
    @test_broken F[1] < 0
    @test_broken abs(F[end] - F_pred[]) < 0.02 * abs(F_pred[])
end
