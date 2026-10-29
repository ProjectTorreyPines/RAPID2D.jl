# The electrons' conduction near the axis of the plasma's own closed surfaces, against an exact solution.
#
# A uniform current around an axis makes b_pol ∝ r, so u∥ and Tₑ conduct along circles with
#     D = D⊥ I + κ [[y², −xy], [−xy, x²]],   κ = D∥ b_pol²/r²  (the m-mode's parallel decay rate is m² κ),
# a tensor smooth through the axis. On a slab (R ≈ 1000 m, the Jacobian nearly constant) the profile
#     T0 = 31 + 40 (1 − e^{−r²/w0²}) + (10/w1) x e^{−r²/w1²}
# evolves exactly as
#     T  = 31 + 40 (1 − (w0/W0)² e^{−r²/W0²}) + (10/w1) x (w1/W1)⁴ e^{−r²/W1²} e^{−κ t},   W² = w² + 4 D⊥ t
# (the 2D heat kernel on a Gaussian and on x times one; κ∂θ² commutes with D⊥∇²), and is flat at the wall.
# Its minimum only rises. internal/docs/src/notes/plans/PLAN_monotone-conduction.md §7.

@testsnippet ConductionAxis begin
    using RAPID2D: cache_electron_operators!, ue_Te_operators, build_wall_diffusion_matrix!, build_wall_pattern
    using RAPID2D.SparseArrays: sparse
    using RAPID2D.LinearAlgebra: I, lu

    const AXIS = (Dperp = 1.0, κ = 4000.0, w0 = 0.15, w1 = 0.12)

    "An N × N slab box about R = 1000 m, its walls at ±0.49 m."
    function slab_RP(N)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = N, NZ = N, R_min = 999.5, R_max = 1000.5, Z_min = -0.5, Z_max = 0.5,
            wall_R = [999.51, 1000.49, 1000.49, 999.51], wall_Z = [-0.49, -0.49, 0.49, 0.49],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 25.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        config.Output_path = mktempdir()
        RP = RAPID{Float64}(config)
        initialize!(RP)
        return RP
    end

    function exact_T(x, y, t)
        (; Dperp, κ, w0, w1) = AXIS
        W0², W1² = w0^2 + 4Dperp * t, w1^2 + 4Dperp * t
        r² = x^2 + y^2
        return 31 + 40 * (1 - w0^2 / W0² * exp(-r² / W0²)) + 10 / w1 * x * (w1^2 / W1²)^2 * exp(-r² / W1²) * exp(-κ * t)
    end

    """
    The electrons on the axis at (Rc, Zc): particles diffuse with D⊥ alone, u∥ and Tₑ conduct with D⊥ and κ r²
    along the circles, a uniform density. Returns the operator they are carried with and the offsets (x, y).
    """
    function axis_carriage(RP, Rc, Zc)
        G, tp, F, pla = RP.G, RP.transport, RP.fields, RP.plasma
        x, y = G.R2D .- Rc, G.Z2D .- Zc
        ρ = hypot.(x, y)
        F.bR .= ifelse.(ρ .> 0, -y ./ ρ, 0.0)
        F.bZ .= ifelse.(ρ .> 0, x ./ ρ, 0.0)
        tp.DRR .= AXIS.Dperp
        tp.DRZ .= 0.0
        tp.DZZ .= AXIS.Dperp
        tp.Dpara .= 0.0
        tp.Dpara_cond .= AXIS.κ .* ρ .^ 2
        pla.ne .= 0.0
        pla.ne[G.nodes.in_wall_nids] .= 1.0e18
        cache_electron_operators!(RP)
        return sparse(ue_Te_operators(RP).A_diffu.matrix), vec(x), vec(y)
    end

    "Backward Euler with `M` from the exact profile: the errors at t_end and the largest fall of the inner minimum."
    function evolve_against_exact(RP, M, x, y; dt = 25.0e-6, nsteps = 80)
        G = RP.G
        inw = falses(G.NR * G.NZ)
        inw[G.nodes.in_wall_nids] .= true
        r = hypot.(x, y)
        inner = inw .& (r .< 0.25)
        axis = argmin(r .+ .!inw .* 1.0e9)
        T = exact_T.(x, y, 0.0) .* inw
        F = lu(I - dt * M)
        lowest, worst_fall = minimum(T[inner]), 0.0
        for _ in 1:nsteps
            T = F \ T
            m = minimum(T[inner])
            worst_fall = max(worst_fall, lowest - m)
            lowest = m
        end
        e = T .- exact_T.(x, y, dt * nsteps)
        return (; axis = abs(e[axis]), Linf = maximum(abs, e[inner]), rms = sqrt(sum(abs2, e[inner]) / count(inner)), worst_fall)
    end
end

@testitem "conduction near an axis: at uniform density the electrons conduct with the full tensor's own operator" setup = [ConductionAxis] begin
    RP = slab_RP(21)
    G = RP.G
    M, x, y = axis_carriage(RP, 1000.0, 0.0)
    (; Dperp, κ) = AXIS
    X, Y = reshape(x, G.NR, G.NZ), reshape(y, G.NR, G.NZ)
    full = build_wall_diffusion_matrix!(build_wall_pattern(G), G, Dperp .+ κ .* Y .^ 2, -κ .* X .* Y, Dperp .+ κ .* X .^ 2)
    A = sparse(full.matrix)
    # the carried variable's operator is the conduction's: the particles' diffusion plus the density-weighted
    # excess, assembled face by face like it, so the same face fluxes and the same cross terms
    @test maximum(abs, M .- A) <= 1.0e-12 * maximum(abs, A)
end

@testitem "conduction near an axis: a Tₑ minimum on the axis of a uniform current's surfaces is never deepened, and the profile follows the exact solution" setup = [ConductionAxis] begin
    # h = 0.05 m resolves the axis region like the filament of examples/ohmic_heating.jl; the axis on a node
    # and between nodes
    errs = Dict{Tuple, Any}()
    for shift in ((0.0, 0.0), (0.37, 0.21)), N in (21, 41)
        RP = slab_RP(N)
        h = RP.G.dR
        M, x, y = axis_carriage(RP, 1000.0 + shift[1] * h, shift[2] * h)
        e = evolve_against_exact(RP, M, x, y)
        errs[(N, shift)] = e
        @test e.worst_fall <= 1.0e-8
        N == 21 && @test e.axis <= 3.5 && e.Linf <= 4.0 && e.rms <= 2.0
        if N == 41
            # refining reduces every error
            c = errs[(21, shift)]
            @test e.axis <= 0.45 * c.axis && e.Linf <= 0.45 * c.Linf && e.rms <= 0.45 * c.rms
        end
    end
end
