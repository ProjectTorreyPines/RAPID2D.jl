# The coupled step (electron parallel momentum, Ampère's free-boundary GS solve, and the
# circuits of coils and vessel filaments) against predictions that do not come from the
# code: an exact circuit solution, a lumped circuit model, the flux a superconducting loop
# must keep, and time-step refinement. examples/coupled_step/ has the same problems with
# their physics written out and plotted.
#

@testsnippet CoupledStepSetup begin
    # A uniform column at rest in a pure toroidal field (Eϕ = E0·R̄/R), Coulomb drag only;
    # Te = 1 eV puts its L/R time near 0.2 ms. `moving` turns on the mean E×B drift and the
    # advection of n and u∥ by it.
    function column(;
            E0 = 0.3, Te = 1.0, n0 = 1.0e16, cenR = 1.5, cenZ = 0.0, radius = 0.3, threshold = 0.0,
            dt = 5.0e-6, t_end = 1.0e-3, moving = false,
        )
        # Built here, not with RegressionCommon's helpers: under ReTestItems a snippet is its
        # own module and cannot call another snippet's functions. cleanup = false, because the
        # RAPID constructor opens ADIOS handles there that a finalizer closes later.
        config = SimulationConfig{Float64}(;
            device_Name = "manual", Output_path = mktempdir(; cleanup = false),
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
        r = @. sqrt((RP.G.R2D - cenR)^2 + (RP.G.Z2D - cenZ)^2)
        n = @. ifelse(r < radius, n0, 0.0)
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

    # The column as one loop of uniform current over the cells it fills (area S): L_p, the
    # electrons' inertia L_kin at the present density, and M, each coil's flux per ampere.
    function lumped_column(RP; R0 = 1.5)
        G, pla, c = RP.G, RP.plasma, RP.config.constants
        col = findall(>(0), pla.ne)
        S = length(col) * G.dR * G.dZ
        J = zeros(size(pla.ne))
        J[col] .= 1.0
        return (;
            L_p = c.μ0 * R0 * (log(8R0 / sqrt(S / π)) - 7 / 4),
            L_kin = c.me * 2π * R0 / (sum(pla.ne[col]) / length(col) * c.ee^2 * S),
            M = RP.coil_system.n_total > 0 ? flux_at_coils(RP, J) ./ plasma_current(RP, J) : Float64[],
        )
    end

    # Resistance of the column to a uniform loop voltage, 1/R_p = Σ σ dA / (2πR), σ = n e²/(mₑ ν).
    function column_resistance(RP)
        G, pla, c = RP.G, RP.plasma, RP.config.constants
        col = findall(>(0), pla.ne)
        return 1 / (sum(@. c.ee^2 * pla.ne[col] / (c.me * pla.ν_ei_eff[col] * 2π * G.R2D[col])) * G.dR * G.dZ)
    end

    # Run to t_end with a callback after every step; progress lines are dropped. A callback
    # that changes the plasma state calls RAPID2D.update_transport_quantities!: the step
    # refreshes the collision rates before the callback, not after it.
    function run_quiet!(RP; before = nothing, after = nothing)
        redirect_stdout(devnull) do
            run_simulation!(RP; callback_before_step = before, callback_after_step = after)
        end
        return RP
    end
end

@testitem "Coupled step: two coils without plasma" tags = [:regression] setup = [CoupledStepSetup] begin
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
    # without plasma the 1 A threshold has nothing to act on: the same recursion
    @test maximum(maximum(abs.(I_default[k] .- exact[k])) for k in eachindex(exact)) < 1.0e-12 * I0
end

@testitem "Coupled step: a pre-charged loop induces nothing" tags = [:regression] setup = [CoupledStepSetup] begin
    # A column at rest, no loop voltage, beside a superconducting loop that already carries
    # 1 kA. Nothing changes, so nothing is induced. The loop's flux must be in ψ_self from the
    # start; otherwise the first step reads its appearance as a sudden flux change and the
    # column screens it.
    RP = column(; E0 = 0.0, t_end = 0.1e-3)
    add_loop!(RP, 1.2, 0.8; I0 = 1000.0)
    initialize_coil_system!(RP)
    Ip = Float64[]
    run_quiet!(RP; after = rp -> push!(Ip, plasma_current(rp, current_density(rp))))

    @test maximum(abs, Ip) < 1.0e-3 * 1000.0
end

@testitem "Coupled step: density doubled, column alone" tags = [:regression] setup = [CoupledStepSetup] begin
    # The column alone; at t1 every electron and ion is cloned with its own velocity (n → 2n).
    # Its self-inductance holds the flux (L_p + L_kin) I_p, and the electrons' inertia L_kin
    # halves: the current rises by (L_p + L_kin)/(L_p + L_kin/2) and the drift halves. The
    # coupled step gets this right today; the check guards it.
    t1 = 0.75e-3
    RP = column(; t_end = t1 + 10.0e-6)
    m = lumped_column(RP)
    t, Ip = Float64[], Float64[]
    doubled = Ref(false)
    run_quiet!(
        RP; after = rp -> begin
            push!(t, rp.time_s)
            push!(Ip, plasma_current(rp, current_density(rp)))
            if !doubled[] && rp.time_s >= t1 - 1.0e-12
                rp.plasma.ne .*= 2
                rp.plasma.ni .*= 2
                RAPID2D.update_transport_quantities!(rp)
                doubled[] = true
            end
        end
    )
    kd = findfirst(>=(t1 - 1.0e-12), t)   # the doubling step, recorded just before it
    jump = (m.L_p + m.L_kin) / (m.L_p + m.L_kin / 2)

    @test abs(Ip[kd + 1] / Ip[kd] - jump) < 0.005
end

@testitem "Coupled step: loop flux with the density doubled" tags = [:regression] setup = [CoupledStepSetup] begin
    # A driven column beside a superconducting loop, which must keep L_c I_c + Φ_p = 0. At t1
    # every electron and ion is cloned with its own velocity (n → 2n). The fluxes
    # (L_p + L_kin) I_p + M I_c and L_c I_c + M I_p cannot jump, and L_kin halves: the current
    # rises by under 2 % (it does not double), the drift halves, the loop current holds.
    t1 = 0.75e-3
    RP = column(; t_end = 1.5e-3)
    Lc = add_loop!(RP, 1.2, 0.8)
    initialize_coil_system!(RP)
    m = lumped_column(RP)
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
                RAPID2D.update_transport_quantities!(rp)
                doubled[] = true
            end
        end
    )
    flux_error(ks) = maximum(abs.(Ic[ks] .- Ic_flux[ks])) / maximum(abs.(Ic_flux[ks]))
    before, later = findall(<(t1 - 1.0e-12), t), findall(>=(t1 + 50.0e-6), t)
    kd = before[end] + 1   # the doubling step, recorded just before it
    L_before = [m.L_p + m.L_kin m.M[1]; m.M[1] Lc]
    L_after = [m.L_p + m.L_kin / 2 m.M[1]; m.M[1] Lc]
    jump = (L_after \ (L_before * [Ip[kd], Ic[kd]]))[1] / Ip[kd]
    rise = Ip[kd + 1] / Ip[kd]

    @test flux_error(before) < 1.0e-3
    @test abs(rise - 1) < 0.1   # the current does not double
    @test abs(rise - jump) < 0.005   # the jump of the canonical fluxes, loop included
    @test flux_error(later) < 1.0e-2  # and the loop keeps its flux through it
end

@testitem "Coupled step: coil-driven column" tags = [:regression] setup = [CoupledStepSetup] begin
    # No loop voltage; a 10 V coil at R = 0.6 m drives the column. Prediction: the coil and the
    # column as one loop of uniform current, with the electrons' inertia L_kin,
    #   [L_c M; M L_p + L_kin] d/dt [I_c; I_p] = [V − R_c I_c; −R_p I_p].
    V, Rcoil, R0, a = 10.0, 1.0e-4, 1.5, 0.3
    function driven(threshold)
        RP = column(; E0 = 0.0, cenR = R0, radius = a, threshold)
        Lc = add_loop!(RP, 0.6, 0.0; a = 0.1, R = Rcoil, V, name = "OH")
        initialize_coil_system!(RP)
        run_quiet!(RP)
        return RP, Lc
    end
    # the model at t_end, RK4; read after the run (n and Te are fixed, so R_p is constant)
    function two_circuit(RP, Lc; nsteps = 10_000)
        m = lumped_column(RP; R0)
        A = [Lc m.M[1]; m.M[1] (m.L_p + m.L_kin)]
        Rp = column_resistance(RP)
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
    # Under the default 1 A threshold the coil still drives the column from zero current.
    @test abs(plasma_current(RP_default, current_density(RP_default)) - Ip) < 0.02 * abs(Ip)
end

@testitem "Coupled step: a decaying current passes the gate and dies" tags = [:regression] setup = [CoupledStepSetup] begin
    # A column driven for 1 ms, then left without loop voltage: its current decays to zero.
    # Under the default 1 A threshold it must decay through the threshold the same way;
    # nothing should keep pushing once the coupled solve stops.
    function decay(threshold)
        RP = column(; threshold, t_end = 3.5e-3)
        off = rp -> if rp.time_s >= 1.0e-3 - 1.0e-12
            fill!(rp.fields.LV_ext, 0.0)
            fill!(rp.fields.Eϕ_ext, 0.0)
            fill!(rp.fields.E_para_ext, 0.0)
        end
        Ip = Float64[]
        run_quiet!(RP; before = off, after = rp -> push!(Ip, plasma_current(rp, current_density(rp))))
        return Ip
    end
    Ip_open, Ip_default = decay(0.0), decay(1.0)

    @test abs(Ip_open[end]) < 1.0e-2
    @test abs(Ip_default[end]) < 1.0e-2
end

@testitem "Coupled step: below the gate a loop does not drive the column" tags = [:regression] setup = [CoupledStepSetup] begin
    # The decay through the 1 A gate with a superconducting loop beside the column. Below the
    # gate the column is not a source of induction: its own inductance is left out, and so is
    # its flux in the loop's circuit. Were the loop's reaction fed back without L_p, it would
    # act as a negative inductance (L_kin − M²/L_c < 0 here), and the current would ring at
    # the gate instead of dying.
    RP = column(; threshold = 1.0, t_end = 3.0e-3)
    add_loop!(RP, 1.2, 0.8)
    initialize_coil_system!(RP)
    off = rp -> if rp.time_s >= 1.0e-3 - 1.0e-12
        fill!(rp.fields.LV_ext, 0.0)
        fill!(rp.fields.Eϕ_ext, 0.0)
        fill!(rp.fields.E_para_ext, 0.0)
    end
    Ip = Float64[]
    run_quiet!(RP; before = off, after = rp -> push!(Ip, plasma_current(rp, current_density(rp))))
    k = findfirst(i -> i > 200 && abs(Ip[i]) < 1.0, eachindex(Ip))   # first below the gate after 1 ms

    @test abs(Ip[end]) < 1.0e-2
    @test count(i -> sign(Ip[i]) != sign(Ip[i - 1]), (k + 1):length(Ip)) <= 1
end

@testitem "Coupled step: the current does not jump back when the gate opens" tags = [:regression] setup = [CoupledStepSetup] begin
    # A thin column (1e14 m⁻³) under a weak loop voltage: its current grows through the
    # default 1 A threshold over many steps, as in an avalanche. When the coupled solve takes
    # over it must start from the flux of the current already there, not from zero.
    RP = column(; E0 = 0.003, n0 = 1.0e14, threshold = 1.0, t_end = 1.0e-3)
    Ip = Float64[]
    run_quiet!(RP; after = rp -> push!(Ip, plasma_current(rp, current_density(rp))))
    k = findfirst(>=(1.0), Ip)

    @test k !== nothing && 1 < k < length(Ip)   # the gate opens inside the run
    @test all(diff(Ip) .>= -1.0e-9 * maximum(Ip))
end

@testitem "Coupled step: growing density, time step halved" tags = [:regression] setup = [CoupledStepSetup] begin
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
                RAPID2D.update_transport_quantities!(rp)
            end
        )
        Φ = flux_at_coils(RP, current_density(RP))[1]
        return abs(Lc * RP.coil_system.coils[1].current + Φ) / abs(Φ)
    end
    errs = flux_error.((5.0e-6, 2.5e-6))

    @test errs[1] < 0.01 && 0.4 < errs[2] / errs[1] < 0.6
end

@testitem "Coupled step: column pushed toward a loop" tags = [:regression] setup = [CoupledStepSetup] begin
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
    # the loop keeps its flux while the current moves into new cells (one step behind)
    @test maximum(abs.(Ic .- Ic_flux)) < 0.02 * maximum(abs.(Ic_flux))
end

@testitem "Coupled step: column shifted inside a shell" tags = [:regression] setup = [CoupledStepSetup] begin
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
                RAPID2D.update_transport_quantities!(rp)
                J_shifted = current_density(rp)
                ΔI = -(rp.coil_system.mutual_inductance \ (flux_at_coils(rp, J_shifted) .- flux_at_coils(rp, J)))
                F_pred[] = force_Z(J_shifted, I_before .+ ΔI)
            end
        end
    )

    @test F_pred[] < 0
    # The shift happens between steps; the shell answers it at the next one and then holds.
    @test F[1] < 0
    @test abs(F[end] - F_pred[]) < 0.02 * abs(F_pred[])
end

@testitem "Coupled step: a dense column at rest under the 1 A gate steps as with the gate at 0" tags = [:regression] setup = [CoupledStepSetup] begin
    # A column already dense (1e17 m⁻³, 5 eV) when the loop voltage comes on. Below the gate it
    # is not a source of induction, so the step that carries its current across the gate would
    # accelerate it without its self-inductance, to several hundred amperes. That step is solved
    # again by the coupled solve, from the same state. The run then steps exactly as the run with
    # the gate at 0, whose every step is coupled.
    function steps(threshold)
        RP = column(; n0 = 1.0e17, Te = 5.0, threshold, t_end = 20.0e-6)
        rec = (I = Float64[], u = Matrix{Float64}[], ψ = Matrix{Float64}[], E = Matrix{Float64}[])
        run_quiet!(
            RP; after = rp -> begin
                push!(rec.I, plasma_current(rp, current_density(rp)))
                push!(rec.u, copy(rp.plasma.ue_para))
                push!(rec.ψ, copy(rp.fields.ψ_self))
                push!(rec.E, copy(rp.fields.Eϕ_self))
            end
        )
        return rec, RP
    end
    (gated, RPg), (open, RPo) = steps(1.0), steps(0.0)

    @test length(gated.I) == RPg.step == 4   # one callback per step
    @test gated.I == open.I
    @test gated.u == open.u
    @test gated.ψ == open.ψ
    @test gated.E == open.E
    @test RPg.diagnostics.ampere_picard.nsolve == RPo.diagnostics.ampere_picard.nsolve
    @test RPg.diagnostics.ampere_picard.niter == RPo.diagnostics.ampere_picard.niter
end

@testitem "Coupled step: the re-solved crossing leaves the coils as the gate at 0 does" tags = [:regression] setup = [CoupledStepSetup] begin
    # The same crossing with a coil driven by a voltage that changes in time, a resistive loop,
    # and circuit matrices built for another time step (the step rebuilds them). The step that
    # crossed and was solved again leaves the coil currents, their memory and their clock as the
    # coupled step does.
    function steps(threshold)
        RP = column(; n0 = 1.0e17, Te = 5.0, threshold, t_end = 20.0e-6)
        add_loop!(RP, 0.6, 0.0; a = 0.1, R = 1.0e-4, V = t -> 10.0 + 1.0e6 * t, name = "OH")
        add_loop!(RP, 1.2, 0.8; R = 1.0e-3, name = "loop")
        initialize_coil_system!(RP)
        RP.coil_system.Δt = 2 * RP.dt
        rec = (I = Float64[], Ic = Vector{Float64}[], ψc = Vector{Float64}[], tc = Float64[])
        run_quiet!(
            RP; after = rp -> begin
                push!(rec.I, plasma_current(rp, current_density(rp)))
                push!(rec.Ic, copy(rp.coil_system.coils.current))
                push!(rec.ψc, copy(rp.coil_system.coils.ψ_pla))
                push!(rec.tc, rp.coil_system.time_s)
            end
        )
        return rec, RP
    end
    (gated, RPg), (open, RPo) = steps(1.0), steps(0.0)

    @test abs(gated.I[1]) > 1.0   # the first step crossed the gate
    @test gated.I == open.I
    @test gated.Ic == open.Ic
    @test gated.ψc == open.ψc
    @test gated.tc == open.tc
    @test RPg.fields.ψ_self == RPo.fields.ψ_self
end

@testitem "Coupled step: a crossing taken outside run_simulation! keeps the coils' memory unset until the coupled solve" tags = [:regression] setup = [CoupledStepSetup] begin
    # Steps taken with advance_timestep! leave Coil.ψ_pla unset until a circuit first uses it.
    # The trial below the gate sets it; when the crossing is solved again, the coupled solve must
    # start from the memory as it was before the trial.
    function first_step(threshold)
        RP = column(; n0 = 1.0e17, Te = 5.0, threshold)
        add_loop!(RP, 1.2, 0.8; R = 1.0e-3)
        initialize_coil_system!(RP)
        RAPID2D.update_transport_quantities!(RP)
        @assert all(isnan, RP.coil_system.coils.ψ_pla)
        redirect_stdout(() -> advance_timestep!(RP), devnull)
        return RP
    end
    gated, open = first_step(1.0), first_step(0.0)

    @test gated.coil_system.coils.ψ_pla == open.coil_system.coils.ψ_pla
    @test gated.coil_system.coils.current == open.coil_system.coils.current
    @test gated.plasma.ue_para == open.plasma.ue_para
end

@testitem "Coupled step: a run split just before the gate crossing is bit-identical" tags = [:regression] setup = [CoupledStepSetup] begin
    # The loop voltage comes on at 15 µs: the column sits below the gate until then, and the
    # step after crosses it. A run stopped at 10 µs and resumed steps as the whole run.
    function switched_on(t_end)
        RP = column(; n0 = 1.0e17, Te = 5.0, threshold = 1.0, t_end)
        field = (LV = copy(RP.fields.LV_ext), Eϕ = copy(RP.fields.Eϕ_ext), Epara = copy(RP.fields.E_para_ext))
        return RP, field
    end
    function drive(field)
        return rp -> if rp.time_s < 15.0e-6 - 1.0e-12
            fill!(rp.fields.LV_ext, 0.0)
            fill!(rp.fields.Eϕ_ext, 0.0)
            fill!(rp.fields.E_para_ext, 0.0)
        else
            rp.fields.LV_ext .= field.LV
            rp.fields.Eϕ_ext .= field.Eϕ
            rp.fields.E_para_ext .= field.Epara
        end
    end
    whole, fw = switched_on(30.0e-6)
    run_quiet!(whole; before = drive(fw))
    split, fs = switched_on(10.0e-6)
    run_quiet!(split; before = drive(fs))
    split.t_end_s = 30.0e-6
    run_quiet!(split; before = drive(fs))

    @test abs(plasma_current(whole, current_density(whole))) > 1.0   # it crossed the gate
    @test split.step == whole.step
    @test split.plasma.ue_para == whole.plasma.ue_para
    @test split.fields.ψ_self == whole.fields.ψ_self
    @test split.fields.Eϕ_self == whole.fields.Eϕ_self
end
