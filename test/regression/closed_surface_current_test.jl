# Current on closed flux surfaces whose conductivity varies along each surface, against a 1-D
# shell reference (examples/closed_surface_current.jl draws the figures). A loop voltage drives a
# column on prescribed elliptic surfaces; friction is on a neutral background at a fixed rate, so
# σ ∝ n, and the density is nine times higher outboard than inboard on every surface. With the
# surface-flux constraint each surface is a series circuit and the surfaces rise together on their
# mutual L/R; the reference holds the same physics in nested shells coupled by free-space Green's
# functions, solved exactly in time.

@testsnippet ClosedSurfaceColumn begin
    using RAPID2D.LinearAlgebra: Diagonal

    const R0, Z0, κ = 1.5113, 0.0071, 1.3      # axis of the elliptic surfaces (off the nodes), elongation
    const C_PSI = 3.0e-2                        # ψ_ext = C ((R − R0)² + ((Z − Z0)/κ)²)
    const RHO_N, N0, NU0, E0, TE = 0.4, 1.0e17, 1.0e6, 0.3, 10.0
    const R0B0, RBAR = 3.0, 1.5
    const QE, ME = -1.602176634e-19, 9.1093837015e-31

    minor_radius(R, Z) = sqrt((R - R0)^2 + ((Z - Z0) / κ)^2)
    density(R, Z, α) = (ρ = minor_radius(R, Z); ρ < RHO_N ? N0 * (1 - (ρ / RHO_N)^2) * (1 + α * (R - R0) / max(ρ, eps())) : 0.0)

    function closed_column(policy; α, ampere, NR = 31, NZ = 41, dt = 2.0e-6, t_end)
        config = RAPID2D.SimulationConfig{Float64}(
            manual = RAPID2D.ManualSetup{Float64}(BR = 0.0, BZ = 0.0, Eϕ = E0), NR = NR, NZ = NZ,
            R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2, R0B0 = R0B0, prefilled_gas_pressure = 0.0,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
            dt = dt, t_end_s = t_end, snap0D_Δt_s = t_end, snap2D_Δt_s = t_end, Output_path = mktempdir(; cleanup = false),
        )
        RP = RAPID2D.RAPID{Float64}(config)
        RP.flags = RAPID2D.SimulationFlags{Float64}(;
            Ampere = ampere, Ampere_Itor_threshold = 0.0, E_para_self_EM = ampere, ud_evolve = true,
            Atomic_Collision = false, Coulomb_Collision = false, src = false, convec = false, diffu = false,
            Te_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
            Include_ud_convec_term = false, Include_ud_diffu_term = false, Include_ud_pressure_term = false,
            Include_Te_convec_term = false, E_para_self_ES = false, mean_ExB = false, turb_ExB_mixing = false,
            FLF_nstep = 100, closed_surface_flow = policy,
        )
        RAPID2D.initialize!(RP)
        G, F, pla = RP.G, RP.fields, RP.plasma
        @. F.ψ_ext = C_PSI * ((G.R2D - R0)^2 + ((G.Z2D - Z0) / κ)^2)
        RAPID2D.calculate_B_from_ψ!(G, F.ψ_ext, F.BR_ext, F.BZ_ext)
        fill!(F.ψ_self, 0.0); fill!(F.BR_self, 0.0); fill!(F.BZ_self, 0.0)
        RAPID2D.combine_external_and_self_fields!(RP)
        pla.ne .= density.(G.R2D, G.Z2D, α) .* (G.nodes.state .> 0.5)
        pla.ni .= pla.ne
        fill!(pla.Te_eV, TE)
        fill!(pla.ue_para, 0.0); fill!(pla.ui_para, 0.0)
        RAPID2D.update_transport_quantities!(RP)
        RAPID2D.flf_analysis_field_lines_rz_plane!(RP)
        RAPID2D.update_closed_surface_basis!(RP)
        return RP
    end

    "Before every step: friction on the neutral background at ν, ions at rest."
    function hold!(RP)
        pla = RP.plasma
        pla.ν_en_mom_tot .= NU0
        pla.ν_en_iz_tot .= 0.0
        pla.ν_ei_eff .= 0.0
        pla.ui_para .= 0.0
        return nothing
    end

    electron_current(RP) = sum(QE .* RP.plasma.ne .* RP.plasma.ue_para .* RP.fields.bϕ) * RP.G.dR * RP.G.dZ

    """
    The shell reference: m A (dg/dt + ν g) = q (F − G dg/dt) per shell of the ellipse, g = n u∥/B on
    the shell, with A = ∫ B²/n dV, F = ∫ Bϕ Eϕ dV and G_jk = ∫_j (Bϕ/R) ψ_k dV, ψ_k the free-space flux
    of shell k's current per unit g. Returns the total current I(t).
    """
    function shell_current(α; nshell = 24, nθ = 48)
        edges = collect(range(0.0, RHO_N; length = nshell + 1))
        points(gauss, nθ, offset) = begin
            R, Z, dA, shell = Float64[], Float64[], Float64[], Int[]
            for j in 1:nshell, (x, w) in zip(gauss...), i in 1:nθ
                a, b = edges[j], edges[j + 1]
                ρ, θ = (a + b) / 2 + (b - a) / 2 * x, 2π * (i - offset) / nθ
                push!(R, R0 + ρ * cos(θ)); push!(Z, Z0 + κ * ρ * sin(θ))
                push!(dA, κ * ρ * (b - a) / 2 * w * 2π / nθ); push!(shell, j)
            end
            (; R, Z, dA, shell)
        end
        d = points(([-1 / sqrt(3), 1 / sqrt(3)], [1.0, 1.0]), nθ, 0.5)
        s = points(
            (
                [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
                [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538],
            ), 2nθ, 0.25
        )
        Bϕ(R) = R0B0 / R
        B(R, Z) = hypot(Bϕ(R), C_PSI * hypot(2 * (R - R0), 2 * (Z - Z0) / κ^2) / R)
        A, F, C, G = zeros(nshell), zeros(nshell), zeros(nshell), zeros(nshell, nshell)
        for i in eachindex(d.R)
            j, R, dV = d.shell[i], d.R[i], 2π * d.R[i] * d.dA[i]
            A[j] += B(R, d.Z[i])^2 / density(R, d.Z[i], α) * dV
            F[j] += Bϕ(R) * E0 * RBAR / R * dV
            C[j] += Bϕ(R) * d.dA[i]
        end
        for k in 1:nshell
            src = findall(==(k), s.shell)
            ψ = vec(sum(RAPID2D.calculate_ψ_by_green_function(d.R, d.Z, s.R[src], s.Z[src], [QE * Bϕ(s.R[i]) * s.dA[i] for i in src]); dims = 2))
            for i in eachindex(d.R)
                G[d.shell[i], k] += 2π * Bϕ(d.R[i]) * d.dA[i] * ψ[i]
            end
        end
        Γ_over_B∞ = QE .* F ./ (ME * NU0 .* A)
        W = (ME .* Diagonal(A) .+ QE .* G) \ (ME * NU0 .* Matrix(Diagonal(A)))
        return t -> sum(QE .* (isinf(t) ? Γ_over_B∞ : Γ_over_B∞ .- exp(-t .* W) * Γ_over_B∞) .* C)
    end
end

@testitem "Closed-surface current: series circuits and their L/R rise against the shell reference" tags = [:regression] setup = [ClosedSurfaceColumn] begin
    using RAPID2D: FreeParallelFlow, SurfaceFluxConstraint
    α = 0.8
    I_ref = shell_current(α)
    I∞ = abs(I_ref(Inf))

    # steady state without induction: each surface carries ⟨E∥B⟩/⟨B²/σ⟩; free flow carries the
    # local Ohmic current, the mean σ on each surface rather than its series combination
    steady(policy) = (
        RP = closed_column(policy; α, ampere = false, t_end = 20.0e-6);
        redirect_stdout(() -> run_simulation!(RP; callback_before_step = hold!), devnull); abs(electron_current(RP))
    )
    @test abs(steady(SurfaceFluxConstraint()) / I∞ - 1) < 0.03
    @test steady(FreeParallelFlow()) > 1.5 * I∞

    # the rise with Ampère on: the surfaces and the induced field solved together
    RP = closed_column(SurfaceFluxConstraint(); α, ampere = true, t_end = 300.0e-6)
    gap = Ref(0.0)
    redirect_stdout(devnull) do
        run_simulation!(
            RP; callback_before_step = hold!, callback_after_step = r -> begin
                gap[] = max(gap[], abs(abs(electron_current(r)) - abs(I_ref(r.time_s))) / I∞)
                nothing
            end
        )
    end
    @test RP.step == 150
    @test gap[] < 0.02
end
