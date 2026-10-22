# Coulomb collisions in 0D: a uniform box without transport. The code's temperature exchange and its
# collision rates are held against a reference that gives every species its own temperature (NRL 2023
# p.34 energy-exchange rates, per-pair Coulomb logarithms) and against the slowing-down rate of a trace
# species on the bulk (NRL p.31). The DEFECT items pin what the 0D audit found
# (internal/docs/src/notes/coulomb-collision-0d-audit.md): the trace species' friction is not the
# pairwise rate, and the common mode's diffusivities carry the internal friction.

@testsnippet CoulombReference begin
    using RAPID2D
    using RAPID2D: ion_ion_coulomb_log, update_coulomb_collision_parameters!, update_transport_quantities!,
        IonRecord, layout_ion_species!, ManualSetup, SimulationFlags, SimulationConfig, RAPID, initialize!,
        run_simulation!, ion_transport_channels, shared_turbulent_channel, channel_D_para

    const EE = 1.602176634e-19
    const ME = 9.1093837015e-31
    const MP = 1.67262192369e-27
    const EPS0 = 8.8541878128e-12
    const MH = MP + ME
    const MH2 = 2MP + ME
    const MD = 2MP + ME
    const MC = 12MP

    # NRL p.34: ν_ε^{α|β} = 1.8e-19 (m_α m_β)^{1/2} Z_α² Z_β² n_β λ / (m_α T_β + m_β T_α)^{3/2}  [g, cm⁻³, eV]
    function ν_ε(m_a, Z_a, T_a, m_b, Z_b, n_b, T_b, λ)
        ma, mb = m_a * 1.0e3, m_b * 1.0e3
        return 1.8e-19 * sqrt(ma * mb) * Z_a^2 * Z_b^2 * (n_b * 1.0e-6) * λ / (ma * T_b + mb * T_a)^1.5
    end
    # NRL p.34(b), the electron–ion logarithm with the ion species' own charge
    function λ_ei(ne, Te, ni, Ti, Z, μ)
        n_cm, ni_cm = ne * 1.0e-6, ni * 1.0e-6
        Te < Ti * ME / (μ * MP) && return 16 - log(sqrt(ni_cm) * Ti^(-1.5) * Z^2 * μ)
        Te > 10 * Z^2 && return 24 - log(sqrt(n_cm) / Te)
        return 23 - log(sqrt(n_cm) * Z * Te^(-1.5))
    end

    """
    Every species at its own temperature, dT_α/dt = Σ_β ν_ε^{α|β} (T_β − T_α), RK4 with the step kept at
    (fastest rate) × step ≤ `safety`, recorded on `0:dt:t_end`. `species` are `(label, m, Z, n, T0)`,
    the electrons first (Z = −1). `hist[i, a]` is species a at `ts[i]`.
    """
    function reference_temperatures(species, t_end, dt; safety = 0.05)
        N = length(species)
        T = [float(s[5]) for s in species]
        ne = species[1][4]
        rate_sum = zeros(N)
        function rates!(dT, T)
            fill!(dT, 0.0)
            fill!(rate_sum, 0.0)
            for a in 1:N, b in 1:N
                a == b && continue
                ma, Za, na = species[a][2], species[a][3], species[a][4]
                mb, Zb, nb = species[b][2], species[b][3], species[b][4]
                nb > 0 || continue
                Ta, Tb = max(T[a], 1.0e-6), max(T[b], 1.0e-6)
                λ = if a == 1 || b == 1
                    i = a == 1 ? b : a
                    λ_ei(ne, max(T[1], 1.0e-6), species[i][4], max(T[i], 1.0e-6), abs(species[i][3]), species[i][2] / MP)
                else
                    ion_ion_coulomb_log(na, Ta, Za, ma / MP, nb, Tb, Zb, mb / MP)
                end
                ν = ν_ε(ma, Za, Ta, mb, Zb, nb, Tb, λ)
                dT[a] += ν * (T[b] - T[a])
                rate_sum[a] += ν
            end
            return maximum(rate_sum)
        end
        ts = collect(0.0:dt:t_end)
        hist = zeros(length(ts), N)
        hist[1, :] .= T
        k1, k2, k3, k4 = (zeros(N) for _ in 1:4)
        for i in 2:length(ts)
            t, t_next = ts[i - 1], ts[i]
            while t < t_next * (1 - 1.0e-12)
                r = rates!(k1, T)
                h = min(t_next - t, safety / max(r, 1.0e-300))
                rates!(k2, T .+ h / 2 .* k1)
                rates!(k3, T .+ h / 2 .* k2)
                rates!(k4, T .+ h .* k3)
                T .+= h / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
                t += h
            end
            hist[i, :] .= T
        end
        return ts, hist
    end
    "The reference's number-weighted ion temperature: the code's one Tᵢ."
    ion_mean_temperature(ions, hist) =
        [sum(ions[k][4] * hist[i, k + 1] for k in eachindex(ions)) / sum(s[4] for s in ions) for i in axes(hist, 1)]

    "Only the Coulomb exchange acts over the step: both temperatures evolve, the ions keep their own densities, everything else is off."
    exchange_only_flags() = SimulationFlags{Float64}(;
        src = false, diffu = false, convec = false, ud_evolve = false, Gas_evolve = false,
        Atomic_Collision = false, Coulomb_Collision = true, Te_evolve = true, Ti_evolve = true,
        update_ni_independently = true, mean_ExB = false, turb_ExB_mixing = false,
        E_para_self_ES = false, E_para_self_EM = false, Ampere = false, Global_JxB_Force = false,
        Include_Te_convec_term = false, Include_Te_diffu_term = false, Include_heat_flux_term = false,
        vacuum_relaxation = (state = false, factor = 100.0),
    )
    "A uniform 7×7 box; `ions` are `(label, m, Z, n)`, the first the bulk. `flags` replace the defaults before `initialize!`."
    function uniform_box(ions; ne, Te, Ti, dt, p_gas = 2.0e-3, flags = nothing)
        config = SimulationConfig{Float64}(
            device_Name = "manual", manual = ManualSetup{Float64}(BR = 0.0, BZ = 5.0e-3, Eϕ = 0.0),
            NR = 7, NZ = 7, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = p_gas, R0B0 = 1.0, dt = dt, t_end_s = 1.0,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(; cleanup = false),
        )
        RP = RAPID{Float64}(config)
        flags === nothing || (RP.flags = flags)
        initialize!(RP)
        pla = RP.plasma
        layout_ion_species!(RP, [IonRecord(Symbol(s[1]), s[2], s[3]) for s in ions])
        pla.ni .= ions[1][4]
        for (k, s) in enumerate(ions[2:end])
            pla.n_ions[k + 1] .= s[4]
        end
        pla.ne .= ne
        pla.Te_eV .= Te
        pla.Ti_eV .= Ti
        return RP
    end
    "The exchange substep alone, as the step calls it: the collision parameters at the step's temperatures, then the exchange."
    function substep_temperatures(ions; ne, Te0, Ti0, dt, t_end, flags = nothing)
        RP = uniform_box(ions; ne, Te = Te0, Ti = Ti0, dt, flags)
        nsteps = round(Int, t_end / dt)
        ts, Te, Ti = zeros(nsteps + 1), zeros(nsteps + 1), zeros(nsteps + 1)
        Te[1], Ti[1] = Te0, Ti0
        for k in 1:nsteps
            update_coulomb_collision_parameters!(RP)
            RAPID2D.apply_ei_exchange!(RP)
            ts[k + 1] = k * dt
            Te[k + 1], Ti[k + 1] = RP.plasma.Te_eV[4, 4], RP.plasma.Ti_eV[4, 4]
        end
        return ts, Te, Ti
    end
    "The whole step with `exchange_only_flags`, the temperatures at the start of every step and at the end."
    function whole_step_temperatures(ions; ne, Te0, Ti0, dt, t_end)
        RP = uniform_box(ions; ne, Te = Te0, Ti = Ti0, dt, flags = exchange_only_flags())
        ts, Te, Ti = Float64[], Float64[], Float64[]
        record(RP) = (push!(ts, RP.time_s); push!(Te, RP.plasma.Te_eV[4, 4]); push!(Ti, RP.plasma.Ti_eV[4, 4]))
        RP.t_end_s = t_end
        redirect_stdout(devnull) do
            run_simulation!(RP; callback_before_step = record)
        end
        record(RP)
        return ts, Te, Ti
    end
    sample_at(tr, v, tc) = [v[clamp(searchsortedfirst(tr, t), 1, length(v))] for t in tc]
    """
    The code's Tₑ and Tᵢ against the reference's Tₑ and number-weighted T̄ᵢ: the max relative
    differences, the equilibrium temperature, and whether the code ever passes it.
    """
    function exchange_errors(ions, tr, hist, tc, Te_c, Ti_c)
        Te_r = sample_at(tr, hist[:, 1], tc)
        Ti_r = sample_at(tr, ion_mean_temperature(ions, hist), tc)
        ne = sum(s[3] * s[4] for s in ions)
        n_I = sum(s[4] for s in ions)
        T_eq = (ne * Te_c[1] + n_I * Ti_c[1]) / (ne + n_I)
        overshoot = Te_c[1] >= Ti_c[1] ? (minimum(Te_c) < T_eq * (1 - 1.0e-9) || maximum(Ti_c) > T_eq * (1 + 1.0e-9)) :
            (maximum(Te_c) > T_eq * (1 + 1.0e-9) || minimum(Ti_c) < T_eq * (1 - 1.0e-9))
        return (; rel_e = maximum(abs.(Te_c .- Te_r) ./ Te_r), rel_i = maximum(abs.(Ti_c .- Ti_r) ./ Ti_r), T_eq, overshoot)
    end

    # ψ(x) = (2/√π) ∫₀ˣ √t e^{-t} dt, Simpson
    function ψ(x)
        x <= 0 && return 0.0
        N = 4000
        h = x / N
        f(t) = sqrt(t) * exp(-t)
        s = f(0.0) + f(x)
        for k in 1:(N - 1)
            s += (isodd(k) ? 4 : 2) * f(k * h)
        end
        return 2 / sqrt(π) * s * h / 3
    end
    "NRL p.31: the slowing-down rate of species s at its most probable speed on a Maxwellian field t at the same T [eV]."
    function slowing_rate(m_s, Z_s, n_t, m_t, Z_t, T, λ)
        v = sqrt(2 * EE * T / m_s)
        ν0 = 4π * (Z_s * Z_t * EE^2 / (4π * EPS0))^2 * λ * n_t / (m_s^2 * v^3)
        return (1 + m_s / m_t) * ψ(m_t / m_s) * ν0
    end
    "The friction of trace species `k` on the other ions, the bath held fixed: what D∥ = T/(m ν) stands for."
    function trace_friction_reference(ions, k; Ti)
        m_s, Z_s, n_s = ions[k][2], ions[k][3], ions[k][4]
        ν = 0.0
        for (t, sp) in enumerate(ions)
            t == k && continue
            m_t, Z_t, n_t = sp[2], sp[3], sp[4]
            λ = ion_ion_coulomb_log(n_s, Ti, Z_s, m_s / MP, n_t, Ti, Z_t, m_t / MP)
            # the slow-test-particle form holds for the heavier species; the lighter one's rate follows
            # from momentum conservation, m_s n_s ν_st = m_t n_t ν_ts
            ν += m_s >= m_t ? slowing_rate(m_s, Z_s, n_t, m_t, Z_t, Ti, λ) :
                slowing_rate(m_t, Z_t, n_s, m_s, Z_s, Ti, λ) * (m_t * n_t) / (m_s * n_s)
        end
        return ν
    end
    "The friction rate the code's collisional channel of species `k` implies, read back from its D∥ = T/(m ν)."
    function code_trace_friction(ions, k; Ti)
        RP = uniform_box(ions; ne = sum(s[3] * s[4] for s in ions), Te = 10.0, Ti, dt = 1.0e-6)
        RP.flags.Atomic_Collision = false
        update_transport_quantities!(RP)
        rec = RP.transport.ion_species[k]
        ch = ion_transport_channels(RP, rec, shared_turbulent_channel(RP))[1]
        return Ti * EE / (rec.mass * channel_D_para(ch)[4, 4])
    end
end

@testitem "Coulomb exchange: one ion relaxes as the per-species reference, from no difference to a thousandfold" setup = [CoulombReference] tags = [:physics] begin
    ions = [("H2", MH2, 1, 1.0e18)]
    # no temperature difference: the exchange is exactly zero, and both temperatures stay bit for bit
    _, Te_same, Ti_same = substep_temperatures(ions; ne = 1.0e18, Te0 = 1.0, Ti0 = 1.0, dt = 2.0e-6, t_end = 2.0e-4)
    @test all(==(1.0), Te_same) && all(==(1.0), Ti_same)
    # the electrons tenfold and a thousandfold hotter than the ions, and tenfold colder: cold electrons
    # exchange faster (ν_ei ∝ Tₑ^{-3/2}), so that case takes half the step over the same 4 ms
    for (Te0, Ti0, dt) in ((10.0, 1.0, 2.0e-6), (1000.0, 1.0, 2.0e-6), (1.0, 10.0, 1.0e-6))
        tc, Te, Ti = substep_temperatures(ions; ne = 1.0e18, Te0, Ti0, dt, t_end = 4.0e-3)
        tr, hist = reference_temperatures([("e", ME, -1, 1.0e18, Te0), ("H2", MH2, 1, 1.0e18, Ti0)], 4.0e-3, dt)
        err = exchange_errors(ions, tr, hist, tc, Te, Ti)
        @test err.rel_e < 0.02
        @test err.rel_i < 0.03
        @test !err.overshoot
        # the exchange conserves the thermal energy exactly, whatever it moves
        @test all(≈(1.0e18 * (Te0 + Ti0); rtol = 1.0e-12), 1.0e18 .* Te .+ 1.0e18 .* Ti)
    end
end

@testitem "Coulomb exchange: three ions at different temperatures relax as the reference's number-weighted mean" setup = [CoulombReference] tags = [:physics] begin
    # H⁺ bulk, H₂⁺ and C⁶⁺ beside it, the ions 300× apart at the start, the electrons at 10 eV
    for temps in ((1.0, 10.0, 0.03), (0.03, 0.03, 10.0))
        ions = [("H", MH, 1, 1.0e18), ("H2", MH2, 1, 1.0e17), ("C6", MC, 6, 1.0e16)]
        ne = 1.0e18 + 1.0e17 + 6 * 1.0e16
        n_I = 1.0e18 + 1.0e17 + 1.0e16
        Ti0 = (1.0e18 * temps[1] + 1.0e17 * temps[2] + 1.0e16 * temps[3]) / n_I   # the code's one Tᵢ: number-weighted
        species = vcat([("e", ME, -1, ne, 10.0)], [(ions[k]..., temps[k]) for k in 1:3])
        tr, hist = reference_temperatures(species, 2.0e-3, 2.0e-6)
        tc, Te, Ti = substep_temperatures(ions; ne, Te0 = 10.0, Ti0, dt = 2.0e-6, t_end = 2.0e-3)
        err = exchange_errors(ions, tr, hist, tc, Te, Ti)
        @test err.rel_e < 0.02
        @test err.rel_i < 0.05
        @test !err.overshoot
        # the ions equalize among themselves (to 5 % of their spread) within a fifth of the time the
        # electrons take to give them half of the gap: one ion temperature is a sound reduction
        spread = [maximum(hist[i, 2:end]) - minimum(hist[i, 2:end]) for i in axes(hist, 1)]
        Tbar = ion_mean_temperature(ions, hist)
        gap = abs.(hist[:, 1] .- Tbar)
        t_ions = tr[findfirst(<(0.05 * spread[1]), spread)]
        t_half = tr[findfirst(<(0.5 * gap[1]), gap)]
        @test t_ions < 0.2 * t_half
    end
end

@testitem "Coulomb exchange: the substep is stable and exact at equilibrium however stiff" setup = [CoulombReference] tags = [:physics] begin
    h2_ions(n) = [("H2", MH2, 1, n)]
    # cold dense plasma and a long step: the exchange rate × step is in the thousands, from either side
    for (Te0, Ti0) in ((2.0, 0.1), (0.1, 2.0))
        RP = uniform_box(h2_ions(1.0e21); ne = 1.0e21, Te = Te0, Ti = Ti0, dt = 2.0e-4)
        update_coulomb_collision_parameters!(RP)
        a0 = RAPID2D.ei_relaxation_rate!(similar(RP.plasma.ni), RP)[4, 4]
        @test a0 * 2 * 2.0e-4 > 1000
        tc, Te, Ti = substep_temperatures(h2_ions(1.0e21); ne = 1.0e21, Te0, Ti0, dt = 2.0e-4, t_end = 20 * 2.0e-4)
        T_eq = (Te0 + Ti0) / 2
        @test Te[2] ≈ T_eq rtol = 1.0e-12
        @test Ti[2] ≈ T_eq rtol = 1.0e-12
        @test all(≈(T_eq; rtol = 1.0e-12), Te[2:end]) && all(≈(T_eq; rtol = 1.0e-12), Ti[2:end])
        # the hot one never falls below the equilibrium, the cold one never rises above it
        hot, cold = Te0 > Ti0 ? (Te, Ti) : (Ti, Te)
        @test minimum(hot) >= T_eq * (1 - 1.0e-12) && maximum(cold) <= T_eq * (1 + 1.0e-12)
    end
    # one temperature a bath (its flag off): it stays bit for bit, and the other lands on it, never past it
    for (Te_evolve, Ti_evolve) in ((true, false), (false, true))
        _, Te_b, Ti_b = substep_temperatures(
            h2_ions(1.0e21); ne = 1.0e21, Te0 = 2.0, Ti0 = 0.1, dt = 2.0e-4, t_end = 20 * 2.0e-4,
            flags = SimulationFlags{Float64}(; Te_evolve, Ti_evolve),
        )
        bath, mover = Te_evolve ? (Ti_b, Te_b) : (Te_b, Ti_b)
        @test all(==(bath[1]), bath)
        @test all(≈(bath[1]; rtol = 1.0e-12), mover[2:end])
        @test Te_evolve ? minimum(mover) >= bath[1] * (1 - 1.0e-12) : maximum(mover) <= bath[1] * (1 + 1.0e-12)
    end
    # and the resolved regime is accurate: the same plasma at 1e18 and 2 µs
    tc_r, Te_r, Ti_r = substep_temperatures(h2_ions(1.0e18); ne = 1.0e18, Te0 = 2.0, Ti0 = 0.1, dt = 2.0e-6, t_end = 4.0e-5)
    tr, hist = reference_temperatures([("e", ME, -1, 1.0e18, 2.0), ("H2", MH2, 1, 1.0e18, 0.1)], 4.0e-5, 2.0e-6)
    err = exchange_errors(h2_ions(1.0e18), tr, hist, tc_r, Te_r, Ti_r)
    @test err.rel_e < 0.01 && err.rel_i < 0.03 && !err.overshoot
end

@testitem "Coulomb exchange: the whole step is the substep alone when everything else is off" setup = [CoulombReference] tags = [:physics] begin
    for (ions, Te0, Ti0) in (
            ([("H2", MH2, 1, 1.0e18)], 10.0, 1.0),
            ([("H", MH, 1, 1.0e18), ("H2", MH2, 1, 1.0e17), ("C6", MC, 6, 1.0e16)], 10.0, 1.8),
        )
        ne = sum(s[3] * s[4] for s in ions)
        tc, Te_s, Ti_s = substep_temperatures(ions; ne, Te0, Ti0, dt = 2.0e-6, t_end = 3.0e-4)
        tf, Te_f, Ti_f = whole_step_temperatures(ions; ne, Te0, Ti0, dt = 2.0e-6, t_end = 3.0e-4)
        @test length(tf) == length(tc)
        # the same function on the same numbers: bit for bit
        @test Te_f == Te_s && Ti_f == Ti_s
    end
end

@testitem "DEFECT: a trace species' Coulomb friction is not the pairwise slowing-down rate on the other species" setup = [CoulombReference] tags = [:physics] begin
    # The channel's rate is the mean ion's ν_ii scaled by a test-particle ratio with a √ mass factor
    # (`ion_transport_channels`). It is right where the species are alike …
    code_over_reference(ions) = code_trace_friction(ions, 2; Ti = 1.0) / trace_friction_reference(ions, 2; Ti = 1.0)
    @test code_over_reference([("H2", MH2, 1, 1.0e18), ("H", MH, 1, 1.0e17)]) ≈ 1 rtol = 0.1
    # … and about twice the rate for a heavy trace, two thirds of it for an isotope trace: a single
    # anchor cannot follow the pairwise rates (internal/docs/src/notes/coulomb-collision-0d-audit.md §2.2)
    heavy = code_over_reference([("H", MH, 1, 1.0e18), ("C6", MC, 6, 1.0e16)])
    @test_broken heavy ≈ 1 rtol = 0.2
    @test 1.9 < heavy < 2.2                   # the size of the defect, so a partial fix is noticed
    isotope = code_over_reference([("H", MH, 1, 1.0e18), ("D", MD, 1, 5.0e16)])
    @test_broken isotope ≈ 1 rtol = 0.2
    @test 0.6 < isotope < 0.75
end

@testitem "DEFECT: the common mode's parallel diffusivities carry the internal friction" setup = [CoulombReference] tags = [:physics] begin
    # Electrons and ions moving together feel no e–i friction, and ion–ion collisions cannot slow the
    # ion fluid: only the neutrals drag the common motion. The ion momentum equation drags at that rate …
    ions = [("H2", MH2, 1, 1.0e18)]
    RP = uniform_box(ions; ne = 1.0e18, Te = 5.0, Ti = 1.0, dt = 1.0e-6)
    RP.flags.src = false                      # no births to dilute the drift: the drag alone acts
    pla, tp = RP.plasma, RP.transport
    fill!(pla.ui_para, 1.0e3)                 # the common drift, set before the rates are evaluated:
    fill!(pla.ue_para, 1.0e3)                 # the ion–neutral rate reads the ions' drift energy
    update_transport_quantities!(RP)
    ν_in = tp.νi_neutral[4, 4]
    RAPID2D.update_ui_para!(RP)
    @test (1.0e3 / pla.ui_para[4, 4] - 1) / RP.dt ≈ ν_in rtol = 1.0e-9
    # … but D_a and the bulk ion channel are written with ν_ei and ν_ii in their denominators.
    # D_a and the bulk ion channel's D∥ at density n, beside what the neutral drag alone gives
    function common_mode(n)
        box = uniform_box([("H2", MH2, 1, n)]; ne = n, Te = 5.0, Ti = 1.0, dt = 1.0e-6)
        update_transport_quantities!(box)
        Te, Ti = box.plasma.Te_eV[4, 4], box.plasma.Ti_eV[4, 4]
        drag_e = ME * (box.plasma.ν_en_mom_tot[4, 4] + box.plasma.ν_en_iz_tot[4, 4])   # m ν on the neutrals, per particle
        drag_i = MH2 * box.transport.νi_neutral[4, 4]
        return (
            Da = box.transport.Dpara_amb[4, 4], Di = box.transport.Dpara_i_coll[4, 4],
            Da_drag = EE * (Te + Ti) / (drag_e + drag_i), Di_drag = EE * Ti / drag_i,
        )
    end
    # Where the plasma is tenuous the Coulomb rates are negligible and the two agree …
    tenuous = common_mode(1.0e12)
    @test tenuous.Da ≈ tenuous.Da_drag rtol = 0.02
    @test tenuous.Di ≈ tenuous.Di_drag rtol = 0.02
    # … and once ionized they fall far below the neutral-drag values, a thousandfold at 1e18
    # (internal/docs/src/notes/coulomb-collision-0d-audit.md §3)
    ionized = common_mode(1.0e18)
    @test_broken ionized.Da ≈ ionized.Da_drag rtol = 0.1
    @test_broken ionized.Di ≈ ionized.Di_drag rtol = 0.1
    @test ionized.Da < 1.0e-2 * ionized.Da_drag   # a hundredfold at least: the size of the defect, so a partial fix is noticed
end
