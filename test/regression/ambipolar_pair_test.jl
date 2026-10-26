# The ambipolar pair of the parallel transport in motion: the electrons and the ions leaving open
# field lines together at the sound speed, and evening out the density over closed surfaces.
#
# Open lines: a blob with no loop voltage (Eϕ = 0: no charge sheets, no turbulence, w_amb = 1)
# loses itself to the walls as the pair, c_s² = e(Z̄Tₑ + Tᵢ)/m̄ with m̄ the density-weighted mass and
# Z̄ = nₑ/Σ nₛ. The late decay time is ≈ 0.8 L_c/c_s for the isothermal pair with sonic outflow
# (internal/probes/transport/ambipolar_parallel_loss.jl); it falls with c_s whatever makes it.
# Closed surfaces: a density nine times higher outboard evens out over each surface, the
# null space of the parallel transport being ∇∥n = 0.
# Under a loop voltage, the open-line test sets the sheets' shape factor to 1, so that they shield
# E∥ wherever the plasma holds the charge and the parallel physics is all there is; the code's
# 0.65 stands for an EMF the global potential cannot shield. One fingerprint is red: the ions' own
# parallel flow escapes the surface-flux constraint on closed surfaces.
# internal/docs/src/notes/ambipolar-parallel-loss.md.

@testsnippet OpenLineBlob begin
    using RAPID2D
    using RAPID2D: ManualSetup, SimulationFlags, SimulationConfig, RAPID, initialize!, run_simulation!,
        IonRecord, layout_ion_species!
    using RAPID2D.Particles: H, H2, H3
    using RAPID2D.Statistics: mean

    const U = 1.6605390666e-27
    const QE = 1.602176634e-19
    const N0, SIGMA, TE, R_BLOB = 1.0e17, 0.1, 5.0, 1.6
    const H1P = IonRecord(H[1]; mass = 1.00794U)
    const H2P = IonRecord(H2[1]; mass = 2 * 1.00794U)
    const H3P = IonRecord(H3[1]; mass = 3 * 1.00794U)
    const HE2P = IonRecord(:He2, 4.002602U, 2)

    """
    A blob on straight open field lines (B_Z = B_pol, Bϕ = R0B0/R) of `species` in the given
    fractions, Tₑ and Tᵢ held, no sources, nearly no gas. `field` scales B_pol and Bϕ together:
    the lines stay, Bohm falls as 1/B. `γ0` is the charge sheets' shape factor. Steps of 1/200 of
    L_c/c_s for `ntau` of them, c_s of the pair. Returns the run and the species' density planes.
    """
    function open_line_blob(species, fractions; Ti = 1.0, Bpol = 0.3, Eϕ = 0.0, field = 1.0, γ0 = 0.65, ntau = 3.0, t_end = nothing, dt = nothing)
        m̄ = sum(f * s.mass for (s, f) in zip(species, fractions))
        Z̄ = sum(f * s.charge for (s, f) in zip(species, fractions))
        τ = 2 * 1.05 * hypot(Bpol, 3.0 / R_BLOB) / Bpol / sqrt(QE * (Z̄ * TE + Ti) / m̄)
        config = SimulationConfig{Float64}(;
            device_Name = "manual", manual = ManualSetup{Float64}(; R = (1.2, 2.0), BR = 0.0, BZ = field * Bpol, Eϕ),
            NR = 17, NZ = 49, R0B0 = field * 3.0, prefilled_gas_pressure = 2.0e-5, charge_sheet_shape_factor = γ0,
            dt = something(dt, τ / 200), t_end_s = something(t_end, ntau * τ),
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(),
        )
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(;
            src = false, Te_evolve = false, Ti_evolve = false, Gas_evolve = false,
            update_ni_independently = true, Ampere = false,
        )
        redirect_stderr(devnull) do
            initialize!(RP)
        end
        layout_ion_species!(RP, species)
        G, pla = RP.G, RP.plasma
        inw = G.nodes.in_wall_nids
        blob = @. N0 * exp(-((G.R2D - R_BLOB)^2 + G.Z2D^2) / (2 * SIGMA^2)) + 1.0e8
        planes = [pla.ni, (pla.n_ions[k] for k in 2:length(species))...]
        for (n, f) in zip(planes, fractions)
            fill!(n, 0.0)
            n[inw] .= f .* blob[inw]
        end
        fill!(pla.ne, 0.0)
        pla.ne[inw] .= Z̄ .* blob[inw]
        fill!(pla.Te_eV, TE)
        fill!(pla.Ti_eV, Ti)
        fill!(pla.ue_para, 0.0)
        fill!(pla.ui_para, 0.0)
        return RP, planes
    end

    "L_c/c_s on the blob's centre line, c_s of the pair at the start."
    function sound_time(RP, species, fractions)
        G, pla = RP.G, RP.plasma
        i, j = argmin(abs.(G.R1D .- R_BLOB)), argmin(abs.(G.Z1D))
        m̄ = sum(f * s.mass for (s, f) in zip(species, fractions))
        Z̄ = sum(f * s.charge for (s, f) in zip(species, fractions))
        return (RP.flf.Lc_forward[i, j] + RP.flf.Lc_backward[i, j]) / sqrt(QE * (Z̄ * TE + pla.Ti_eV[i, j]) / m̄)
    end

    "Run to the end; the late decay time of the electrons' inventory, ln N fitted over the second half."
    function late_decay_time(RP)
        G = RP.G
        inw = G.nodes.in_wall_nids
        inventory(n) = sum(n[inw] .* G.inVol2D[inw])
        N0e = inventory(RP.plasma.ne)
        t, N = [0.0], [1.0]
        redirect_stdout(devnull) do
            run_simulation!(RP; callback_after_step = rp -> (push!(t, rp.time_s); push!(N, inventory(rp.plasma.ne) / N0e); nothing))
        end
        late = t .>= t[end] / 2
        x, y = t[late], log.(N[late])
        return -sum((x .- mean(x)) .^ 2) / sum((x .- mean(x)) .* (y .- mean(y)))
    end

    "max |nₑ − Σ zₛnₛ|/nₑ where the plasma holds at least 10⁻³ of its peak."
    function charge_imbalance(RP, planes)
        pla = RP.plasma
        held = pla.ne .> 1.0e-3 * maximum(pla.ne)
        balanced = sum(s.charge .* n for (s, n) in zip(RP.transport.ion_species, planes))
        return maximum(abs.(pla.ne[held] .- balanced[held]) ./ pla.ne[held])
    end
end

@testsnippet ClosedSurfaceDensity begin
    using RAPID2D
    using RAPID2D: ManualSetup, SimulationFlags, SimulationConfig, RAPID, initialize!, run_simulation!,
        FreeParallelFlow, SurfaceFluxConstraint

    const R0, Z0, κ, C_PSI = 1.5113, 0.0071, 1.3, 0.25
    const RHO_N, N0, ALPHA = 0.4, 1.0e17, 0.8
    const SHELLS = range(0.0, 0.4; length = 9)

    minor_radius(R, Z) = sqrt((R - R0)^2 + ((Z - Z0) / κ)^2)
    density(R, Z) = (ρ = minor_radius(R, Z); ρ < RHO_N ? N0 * (1 - (ρ / RHO_N)^2) * (1 + ALPHA * (R - R0) / max(ρ, eps())) : 0.0)

    """
    Prescribed elliptic surfaces ψ = C((R − R0)² + ((Z − Z0)/κ)²), B_pol ≈ 0.1 T at ρ = 0.3, with a
    density nine times higher outboard than inboard on every surface, Tₑ 10 eV and Tᵢ 1 eV held.
    """
    function closed_surfaces(; Eϕ, policy, t_end = 2.0e-3)
        config = SimulationConfig{Float64}(;
            manual = ManualSetup{Float64}(BR = 0.0, BZ = 0.0, Eϕ = Eϕ), NR = 29, NZ = 41,
            R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2, R0B0 = 3.0, prefilled_gas_pressure = 2.0e-3,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
            dt = 5.0e-6, t_end_s = t_end, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(),
        )
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(;
            src = false, Te_evolve = false, Ti_evolve = false, Gas_evolve = false,
            update_ni_independently = true, Ampere = false, FLF_nstep = 100, closed_surface_flow = policy,
        )
        redirect_stderr(devnull) do
            initialize!(RP)
        end
        G, F, pla = RP.G, RP.fields, RP.plasma
        @. F.ψ_presc = C_PSI * ((G.R2D - R0)^2 + ((G.Z2D - Z0) / κ)^2)
        RAPID2D.calculate_B_from_ψ!(G, F.ψ_presc, F.BR_presc, F.BZ_presc)
        fill!(F.ψ_coils_pla, 0.0); fill!(F.BR_coils_pla, 0.0); fill!(F.BZ_coils_pla, 0.0)
        RAPID2D.combine_field_sources!(RP)
        pla.ne .= density.(G.R2D, G.Z2D) .* (G.nodes.state .> 0.5) .+ 1.0e8
        pla.ni .= pla.ne
        fill!(pla.Te_eV, 10.0)
        fill!(pla.Ti_eV, 1.0)
        fill!(pla.ue_para, 0.0); fill!(pla.ui_para, 0.0)
        RAPID2D.flf_analysis_field_lines_rz_plane!(RP)
        RAPID2D.update_closed_surface_basis!(RP)
        return RP
    end

    """
    How far n is from uniform over the closed surfaces: the rms of n over its surface's mean,
    minus one, weighted by n dV. The surface's mean is taken per shell of ρ and interpolated in ρ
    between the shells' centres, so a shell's radial gradient is not read as spread.
    """
    function surface_spread(RP, n)
        G = RP.G
        ρ = minor_radius.(G.R2D, G.Z2D)
        V, closed = G.inVol2D, RP.flf.is_closed
        centres, means = Float64[], Float64[]
        for k in 1:(length(SHELLS) - 1)
            m = (ρ .>= SHELLS[k]) .& (ρ .< SHELLS[k + 1]) .& closed
            any(m) || continue
            push!(centres, sum(V[m] .* ρ[m]) / sum(V[m]))
            push!(means, sum(V[m] .* n[m]) / sum(V[m]))
        end
        num = den = 0.0
        for i in eachindex(n)
            (closed[i] && SHELLS[2] <= ρ[i] < SHELLS[end]) || continue
            k = clamp(searchsortedlast(centres, ρ[i]), 1, length(centres) - 1)
            a = clamp((ρ[i] - centres[k]) / (centres[k + 1] - centres[k]), 0.0, 1.0)
            r = n[i] / ((1 - a) * means[k] + a * means[k + 1])
            num += V[i] * n[i] * (r - 1)^2
            den += V[i] * n[i]
        end
        return sqrt(num / den)
    end

    "∫|nₑ − nᵢ| over ∫nᵢ on the closed surfaces."
    function species_apart(RP)
        pla, V, closed = RP.plasma, RP.G.inVol2D, RP.flf.is_closed
        return sum(abs.(pla.ne[closed] .- pla.ni[closed]) .* V[closed]) / sum(pla.ni[closed] .* V[closed])
    end
end

@testitem "ambipolar pair on open lines: the pair leaves at the sound speed of its mass, mixture, charge and ion temperature" tags = [:regression] setup = [OpenLineBlob] begin
    cases = (
        H2 = ([H2P], [1.0], (;)),
        H1 = ([H1P], [1.0], (;)),
        H3 = ([H3P], [1.0], (;)),
        mix = ([H2P, H3P], [0.5, 0.5], (;)),
        He2 = ([HE2P], [1.0], (;)),
        hot = ([H2P], [1.0], (; Ti = 100.0)),
        strong = ([H2P], [1.0], (; field = 10.0)),     # Bohm a tenth: the loss is along B
    )
    runs = map(cases) do (species, fractions, kw)
        RP, planes = open_line_blob(species, fractions; kw...)
        τ_ref = sound_time(RP, species, fractions)
        τ = late_decay_time(RP)
        (; τ, scaled = τ / τ_ref, imbalance = charge_imbalance(RP, planes), planes, RP)
    end
    # every case decays on its own L_c/c_s, near the isothermal pair's ≈ 0.8
    for r in runs
        @test 0.75 < r.scaled < 1.0
        @test abs(r.scaled / runs.H2.scaled - 1) < 0.06
        # the electrons follow the charge of the ions they leave with
        @test r.imbalance < 1.0e-8
    end
    # the masses, Z = 1: τ ∝ √m̄, m̄ the density-weighted mass
    m(s) = s.mass
    @test runs.H1.τ / runs.H2.τ ≈ sqrt(m(H1P) / m(H2P)) rtol = 0.01
    @test runs.H3.τ / runs.H2.τ ≈ sqrt(m(H3P) / m(H2P)) rtol = 0.01
    @test runs.mix.τ / runs.H2.τ ≈ sqrt((m(H2P) + m(H3P)) / 2 / m(H2P)) rtol = 0.01
    # the charge: c_s² = e(Z̄Tₑ + Tᵢ)/m̄
    @test runs.He2.τ / runs.H2.τ ≈ sqrt(m(HE2P) / (2TE + 1) / (m(H2P) / (TE + 1))) rtol = 0.03
    # the ion temperature
    @test runs.hot.τ / runs.H2.τ ≈ sqrt((TE + 1) / (TE + 100)) rtol = 0.06
    # Bohm has no part in it
    @test runs.strong.τ / runs.H2.τ ≈ 1 rtol = 0.01
    # the mixture moves as one fluid: its fractions stay where there is plasma
    mix = runs.mix
    held = mix.RP.plasma.ne .> 1.0e-3 * maximum(mix.RP.plasma.ne)
    @test maximum(abs.(mix.planes[2][held] ./ mix.planes[1][held] .- 1)) < 1.0e-8
end

@testitem "ambipolar pair on closed surfaces: the density evens out over each surface, the electrons and the ions together" tags = [:regression] setup = [ClosedSurfaceDensity] begin
    RP = closed_surfaces(; Eϕ = 0.0, policy = FreeParallelFlow())
    spread0 = surface_spread(RP, RP.plasma.ne)
    redirect_stdout(devnull) do
        run_simulation!(RP)
    end
    @test spread0 > 0.5
    @test surface_spread(RP, RP.plasma.ne) < 0.05
    @test species_apart(RP) < 1.0e-3

    # [gap] Under a loop voltage the surface-flux constraint holds the electrons' parallel flux
    # divergence-free, but the ions' own E∥-driven parallel flow (~10⁴ m/s on the gas drag) is
    # outside it: the density still evens out, and the species part (0.17 in 2 ms). The
    # constraint belongs on the total current.
    RP = closed_surfaces(; Eϕ = 0.3, policy = SurfaceFluxConstraint())
    redirect_stdout(devnull) do
        run_simulation!(RP)
    end
    @test surface_spread(RP, RP.plasma.ne) < 0.05
    @test_broken species_apart(RP) < 0.02
end

@testitem "open lines under a loop voltage, the sheets shielding E∥ fully: the pair stays together and leaves as without it" tags = [:regression] setup = [OpenLineBlob] begin
    # γ = 1: the sheets' field reaches ERZ_req wherever E_Debye does, and E∥ drives no current there
    function loss(Eϕ)
        RP, planes = open_line_blob([H2P], [1.0]; Bpol = 0.1, Eϕ, γ0 = 1.0, t_end = 2.0e-3, dt = 1.0e-6)
        G, pla = RP.G, RP.plasma
        inw = G.nodes.in_wall_nids
        inventory(n) = sum(n[inw] .* G.inVol2D[inw])
        N0 = inventory(pla.ne)
        redirect_stdout(devnull) do
            run_simulation!(RP)
        end
        apart = sum(abs.(pla.ne[inw] .- pla.ni[inw]) .* G.inVol2D[inw]) / sum(pla.ni[inw] .* G.inVol2D[inw])
        return (; Ne = inventory(pla.ne) / N0, Ni = inventory(pla.ni) / N0, apart)
    end
    driven, free = loss(0.3), loss(0.0)
    @test driven.Ne < 0.8                       # well into the loss to the walls
    @test driven.Ne ≈ driven.Ni rtol = 0.03
    @test driven.apart < 0.05                   # the rest: the blob's edge, below the density that holds the charge
    @test driven.Ne ≈ free.Ne rtol = 0.05
end
