# The plasma flux each coil remembers, ψ_pla(r_c): the value of ψ_pla at the coil that the
# circuit used in its last update. The circuit's d/dt[2π ψ_pla(r_c)] is the difference from
# it, so whatever changes the plasma current between two updates (ionization, transport,
# losses, motion) reaches the coil at the next one.

@testitem "Coil.ψ_pla starts unset and takes vector assignment" setup = [CoilFactories] begin
    c1, c2 = pf_coil("PF1"), wall_coil("W1")
    @test isnan(c1.ψ_pla) && isnan(c2.ψ_pla)
    csys = CoilSystem([c1, c2])
    csys.coils.ψ_pla = [1.0e-3, -2.0e-3]
    @test csys.coils.ψ_pla == [1.0e-3, -2.0e-3]
end

@testitem "Coupled step: the circuit's flux balance closes every step" begin
    # M (Iⁿ⁺¹ − Iⁿ) + Δt R Iⁿ⁺¹ + 2π (ψ_plaⁿ⁺¹ − ψ_plaⁿ) = Δt V, to round-off, whatever the
    # plasma does between steps: here its density grows by 1 % after every step.
    FT = Float64
    config = SimulationConfig{FT}(;
        device_Name = "manual", manual = ManualSetup{FT}(BR = 0.0, BZ = 0.0, Eϕ = 0.3),
        NR = 20, NZ = 30, R0B0 = 3.0, prefilled_gas_pressure = 0.0,
        dt = 5.0e-6, t_end_s = 100.0e-6, snap0D_Δt_s = 50.0e-6, snap2D_Δt_s = 100.0e-6,
        Output_path = mktempdir(; cleanup = false),
    )
    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(;
        Ampere = true, Ampere_Itor_threshold = 0.0, E_para_self_EM = true, ud_evolve = true,
        Coulomb_Collision = true, Atomic_Collision = false, src = false, convec = false, diffu = false,
        Te_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
        Include_ud_convec_term = false, Include_ud_diffu_term = false, Include_ud_pressure_term = false,
        E_para_self_ES = false, mean_ExB = false, turb_ExB_mixing = false, FLF_nstep = 100_000,
    )
    initialize!(RP)
    r = @. sqrt((RP.G.R2D - 1.5)^2 + RP.G.Z2D^2)
    RP.plasma.ne .= ifelse.(r .< 0.3, 1.0e16, 0.0)
    RP.plasma.ni .= RP.plasma.ne
    fill!(RP.plasma.Te_eV, 1.0)
    # a resistive loop beside the column and a 10 V coil inboard of it
    μ0 = RP.config.constants.μ0
    for (rc, zc, R, V, name) in ((1.2, 0.8, 1.0e-3, 0.0, "loop"), (0.6, 0.0, 1.0e-4, 10.0, "OH"))
        coil = Coil{FT}(;
            location = (r = rc, z = zc), area = π * 0.05^2, resistance = R,
            self_inductance = μ0 * rc * (log(8rc / 0.05) - 7 / 4), is_powered = V != 0,
            is_controllable = false, name, voltage_ext = V,
        )
        add_coil!(RP.coil_system, coil)
    end
    initialize_coil_system!(RP)
    rec = (I = Vector{FT}[], ψ = Vector{FT}[])
    redirect_stdout(devnull) do
        run_simulation!(
            RP; callback_after_step = rp -> begin
                push!(rec.I, copy(rp.coil_system.coils.current))
                push!(rec.ψ, copy(rp.coil_system.coils.ψ_pla))
                rp.plasma.ne .*= 1.01
                rp.plasma.ni .*= 1.01
                RAPID2D.update_transport_quantities!(rp)
            end
        )
    end
    csys = RP.coil_system
    M, r_c, V, dt = csys.mutual_inductance, get_all_resistances(csys), get_all_voltages_at_time(csys), RP.dt
    residual(k) = M * (rec.I[k] - rec.I[k - 1]) + dt * r_c .* rec.I[k] + 2π * (rec.ψ[k] - rec.ψ[k - 1]) - dt * V
    scale(k) = maximum(abs, M * rec.I[k]) + dt * maximum(abs, V)

    @test length(rec.I) == 20
    @test maximum(maximum(abs, residual(k)) / scale(k) for k in 2:length(rec.I)) < 1.0e-10
end
