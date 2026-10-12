# Surface functions on the closed region, and the electron parallel flux they carry.
#
# On closed flux surfaces the electron parallel particle flux per B is constrained to a surface
# function, n u∥ / B = K_e(ψ). Its poloidal part K_e ∇ψ × ∇ϕ = ∇H × ∇ϕ with H′ = K_e, so the flux
# through a face of a control volume is 2π (H(ψ_end) − H(ψ_start)) at the face's two corners and
# the divergence of every control volume telescopes to zero. The constraint domain is the support
# of H on the admissible nodes, which puts every face between it and the rest at H = const.
# internal/docs/src/notes/plans/PLAN_closed-surface-flux.md (D1, D2, D6, D17).

@testsnippet ClosedSurfaceFixtures begin
    using RAPID2D: initialize_grid_geometry

    const R0, Z0, κ, c = 1.5113, 0.0071, 1.4, 0.01   # c scales ψ to a few mT of B_pol

    "A uniform N × N grid on R ∈ [1, 2], Z ∈ [−0.5, 0.5]."
    grid(N) = initialize_grid_geometry(N, N, (1.0, 2.0), (-0.5, 0.5))

    "ψ = s c ((R − R0)² + ((Z − Z0)/κ)²): elliptic surfaces about an axis off the nodes."
    ellipse(R, Z; s = 1.0) = s * c * ((R - R0)^2 + ((Z - Z0) / κ)^2)
    ellipse_ψR(R, Z; s = 1.0) = s * c * 2 * (R - R0)
    ellipse_ψZ(R, Z; s = 1.0) = s * c * 2 * (Z - Z0) / κ^2
    ellipse(G::RAPID2D.GridGeometry; s = 1.0) = ellipse.(G.R2D, G.Z2D; s)

    "Nodes inside the surface of minor radius a, and inside R < R_wall."
    admissible_ellipse(G; a = 0.3, R_wall = Inf) =
        Matrix{Bool}(@. (G.R2D - R0)^2 + ((G.Z2D - Z0) / κ)^2 < a^2 && G.R2D < R_wall)

    "The cells left of R_wall, as `inside` for the tracer: (NR, NZ) like `cell_state`, cell (i, j)
    spanning nodes i..i+1 × j..j+1."
    cells_left_of(G, R_wall) = [i < G.NR && j < G.NZ && G.R1D[i] + G.dR / 2 < R_wall for i in 1:G.NR, j in 1:G.NZ]

    "The piecewise-linear function of the knot values `v` at x."
    function hat_interp(knots, v, x)
        length(knots) == 1 && return v[1]
        x >= knots[end] && return x > knots[end] ? zero(x) : v[end]
        s = searchsortedlast(knots, x)
        t = (x - knots[s]) / (knots[s + 1] - knots[s])
        return (1 - t) * v[s] + t * v[s + 1]
    end
end

@testitem "Closed-surface basis: the cut, the support and the hats" setup = [ClosedSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: closed_surface_basis
    for N in (31, 61), s in (1.0, -1.0)
        G = grid(N)
        ψ = ellipse(G; s)
        A = admissible_ellipse(G)
        b = closed_surface_basis(G, ψ, A)
        @test b !== nothing
        @test abs(b.axis.R - R0) < 1.0e-6 * G.dR && abs(b.axis.Z - Z0) < 1.0e-6 * G.dZ
        @test sign(b.ψ_cut - b.axis.ψ) == s                     # ψ_N′ grows outward either way
        @test all(k -> A[k], b.nodes)                           # Ω_c ⊆ admissible
        @test length(b.nodes) > 0.8 * count(A)
        # hats: one per knot from the axis (0) to the cut (1), a partition of unity inside the cut
        # and zero beyond it, where K_e ends
        @test b.knots[1] == 0 && b.knots[end] == 1 && issorted(b.knots)
        rows = vec(sum(b.P; dims = 2))
        @test all(r -> r ≈ 1, rows[b.ψN .<= 1]) && all(iszero, rows[b.ψN .> 1])
        # the τ basis continues the hats as constants beyond the cut
        @test all(≈(1), vec(sum(b.Pτ; dims = 2)))
    end
end

@testitem "Closed-surface flux: the stream-function flux telescopes and matches K_e B_pol" setup = [ClosedSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: closed_surface_basis, stream_function_fluxes, face_outflow
    using RAPID2D.Statistics: median
    errs = Float64[]
    for N in (31, 61)
        G = grid(N)
        ψ = ellipse(G)
        b = closed_surface_basis(G, ψ, admissible_ellipse(G))
        K = [1 + x - 2x^2 for x in b.knots]
        ΓR, ΓZ = stream_function_fluxes(b, K)
        # every control volume of the grid: zero divergence
        out = face_outflow(ΓR, ΓZ)
        scale = maximum(abs, ΓR) + maximum(abs, ΓZ)
        @test maximum(abs, out) <= 1.0e-13 * scale
        # the faces between Ω_c and the rest carry nothing: both corners lie beyond the cut
        inΩ = falses(G.NR, G.NZ)
        inΩ[b.nodes] .= true
        @test all(iszero(ΓR[i, j]) for i in 1:(G.NR - 1), j in 1:G.NZ if inΩ[i, j] != inΩ[i + 1, j])
        @test all(iszero(ΓZ[i, j]) for i in 1:G.NR, j in 1:(G.NZ - 1) if inΩ[i, j] != inΩ[i, j + 1])
        # against the analytic flux of K_e B_pol through each face (midpoint rule), sign included
        xN(R, Z) = (ellipse(R, Z) - b.axis.ψ) / (b.ψ_cut - b.axis.ψ)
        rel, mismatch = Float64[], 0
        for j in 2:(G.NZ - 1), i in 1:(G.NR - 1)
            R, Z = G.R1D[i] + G.dR / 2, G.Z1D[j]
            x = xN(R, Z)
            0 < x < 0.95 || continue
            ref = hat_interp(b.knots, K, x) * (-ellipse_ψZ(R, Z) / R) * 2π * R * G.dZ
            abs(ref) > 1.0e-2 * scale || continue
            push!(rel, abs(ΓR[i, j] / ref - 1))
            mismatch += sign(ΓR[i, j]) != sign(ref)
        end
        for j in 1:(G.NZ - 1), i in 2:(G.NR - 1)
            R, Z = G.R1D[i], G.Z1D[j] + G.dZ / 2
            x = xN(R, Z)
            0 < x < 0.95 || continue
            ref = hat_interp(b.knots, K, x) * (ellipse_ψR(R, Z) / R) * 2π * R * G.dR
            abs(ref) > 1.0e-2 * scale || continue
            push!(rel, abs(ΓZ[i, j] / ref - 1))
            mismatch += sign(ΓZ[i, j]) != sign(ref)
        end
        @test mismatch == 0
        push!(errs, median(rel))
    end
    @test errs[2] < errs[1] / 2      # second order in the grid
    @test errs[2] < 1.0e-2
    # ψ → −ψ reverses B_pol, and so the flux of the same K_e
    G = grid(31)
    bp = closed_surface_basis(G, ellipse(G), admissible_ellipse(G))
    bm = closed_surface_basis(G, ellipse(G; s = -1.0), admissible_ellipse(G))
    K = [1 + x for x in bp.knots]
    @test all(stream_function_fluxes(bm, K)[1] .≈ .-stream_function_fluxes(bp, K)[1])
end

@testitem "Closed-surface basis: a region against the wall keeps its flux inside" setup = [ClosedSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: closed_surface_basis, stream_function_fluxes, face_outflow
    G = grid(41)
    ψ = ellipse(G)
    R_wall = 1.72                                        # cuts the a = 0.3 surface on its outboard side
    A = admissible_ellipse(G; R_wall)
    b = closed_surface_basis(G, ψ, A; inside = cells_left_of(G, R_wall))
    @test b !== nothing
    @test all(k -> G.R2D[k] < R_wall, b.nodes)
    # the cut sits inside the surface that touches the wall
    @test (b.ψ_cut - b.axis.ψ) < ellipse(R_wall, Z0) - b.axis.ψ
    ΓR, ΓZ = stream_function_fluxes(b, ones(length(b.knots)))
    @test maximum(abs, face_outflow(ΓR, ΓZ)) <= 1.0e-13 * maximum(abs, ΓR)
    inΩ = falses(G.NR, G.NZ)
    inΩ[b.nodes] .= true
    @test all(iszero(ΓR[i, j]) for i in 1:(G.NR - 1), j in 1:G.NZ if inΩ[i, j] != inΩ[i + 1, j])
end

@testitem "Closed-surface basis: no O-point, or two, gives no basis; one level is one constant" setup = [ClosedSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: closed_surface_basis
    G = grid(31)
    # no closed surface: ψ = R has no O-point
    @test closed_surface_basis(G, Matrix(G.R2D), fill(true, G.NR, G.NZ)) === nothing
    # two islands in one region
    ψ2 = @. ((G.R2D - 1.5)^2 - 0.25^2)^2 + (G.Z2D - 0.03125)^2
    @test closed_surface_basis(G, ψ2, Matrix{Bool}(ψ2 .< 0.002)) === nothing
    # a few nodes around the axis keep several levels
    b = closed_surface_basis(G, ellipse(G), admissible_ellipse(G; a = 0.045))
    @test b !== nothing && length(b.knots) >= 2
    # four nodes on one level (circles about a dual corner): one constant surface function
    Rc, Zc = G.R1D[16] + G.dR / 2, G.Z1D[16] + G.dZ / 2
    ψc = @. c * ((G.R2D - Rc)^2 + (G.Z2D - Zc)^2)
    A4 = fill(false, G.NR, G.NZ)
    A4[16:17, 16:17] .= true
    b = closed_surface_basis(G, ψc, A4)
    @test b !== nothing
    @test length(b.knots) == 1
    @test all(isone, b.P[b.ψN .<= 1, 1]) && all(iszero, b.P[b.ψN .> 1, 1])
end

@testitem "Closed-surface basis: hats merge deterministically under a large weight contrast" setup = [ClosedSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: closed_surface_basis
    using LinearAlgebra: cond, Diagonal
    G = grid(31)
    ψ = ellipse(G)
    A = admissible_ellipse(G)
    w = @. exp(12 * (G.R2D - R0))                       # weight contrast ≈ e^6 across the region
    b1 = closed_surface_basis(G, ψ, A; weights = w)
    b2 = closed_surface_basis(G, ψ, A; weights = w)
    @test b1.knots == b2.knots
    @test cond(Matrix(b1.P' * Diagonal(w[b1.nodes]) * b1.P)) < 1.0e8
end

@testitem "Closed-surface pressure: zero for surface T, the continuous force otherwise" setup = [ClosedSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: closed_surface_basis, pressure_operator, stream_function_fluxes
    using Random: MersenneTwister
    rng = MersenneTwister(7)
    errs = Float64[]
    for N in (31, 61)
        G = grid(N)
        ψ = ellipse(G)
        b = closed_surface_basis(G, ψ, admissible_ellipse(G))
        R, Z = G.R2D[b.nodes], G.Z2D[b.nodes]
        # g = ln n single-valued and arbitrary; τ a surface function, or a constant: zero force
        g = randn(rng, G.NR, G.NZ)
        Bp = pressure_operator(b, g)
        for τ in (b.Pτ * randn(rng, size(b.Pτ, 2)), fill(3.0, length(b.nodes)))
            F = Bp * τ
            @test maximum(abs, F) <= 1.0e-12 * maximum(abs, abs.(Bp) * abs.(τ))
        end
        # the pressure work on the flux and the compression power are one sum
        K = randn(rng, length(b.knots))
        τ = randn(rng, length(b.nodes))
        @test K' * (Bp * τ) ≈ sum(τ .* (Bp' * K))
        # a temperature that varies along the surfaces: ∫ h_s τ B·∇g dV, by midpoint quadrature
        gfun(R, Z) = 0.5 * (R - R0) / 0.3 + 0.2 * ((Z - Z0) / 0.3)^2
        τfun(R, Z) = 1 + 0.3 * (Z - Z0) / 0.3
        Bps = pressure_operator(b, gfun.(G.R2D, G.Z2D))
        F = Bps * τfun.(R, Z)
        nq = 401
        Rq = range(1.0, 2.0; length = nq + 1)
        Zq = range(-0.5, 0.5; length = nq + 1)
        ref = zeros(length(b.knots))
        for jq in 1:nq, iq in 1:nq
            Rm, Zm = (Rq[iq] + Rq[iq + 1]) / 2, (Zq[jq] + Zq[jq + 1]) / 2
            x = (ellipse(Rm, Zm) - b.axis.ψ) / (b.ψ_cut - b.axis.ψ)
            0 <= x <= 1 || continue
            BgradG = (-ellipse_ψZ(Rm, Zm) / Rm) * (0.5 / 0.3) + (ellipse_ψR(Rm, Zm) / Rm) * (0.4 * (Zm - Z0) / 0.3^2)
            dV = 2π * Rm * step(Rq) * step(Zq)
            for s in eachindex(b.knots)
                h = hat_interp(b.knots, [t == s ? 1.0 : 0.0 for t in eachindex(b.knots)], x)
                ref[s] += h * τfun(Rm, Zm) * BgradG * dV
            end
        end
        push!(errs, maximum(abs, F .- ref) / maximum(abs, ref))
    end
    @test errs[2] < errs[1]
    @test errs[2] < 0.1
end

@testitem "Closed-surface diagnostics: the flux of a velocity, and how far it is from one" setup = [ClosedSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: closed_surface_basis, galerkin_flux, surface_residual
    using Random: MersenneTwister
    rng = MersenneTwister(5)
    G = grid(31)
    b = closed_surface_basis(G, ellipse(G), admissible_ellipse(G))
    B = @. sqrt((3.0 / G.R2D)^2 + 1.0e-6)
    n = @. 1.0e17 * (1 + 0.5 * (G.R2D - R0) / 0.3)
    # a velocity that is a surface flux (zero beyond the cut) gives back its K_e, with no residual
    K = [1 + x - 2x^2 for x in b.knots]
    u = zeros(G.NR, G.NZ)
    u[b.nodes] .= (B[b.nodes] ./ n[b.nodes]) .* (b.P * K)
    @test galerkin_flux(b, n, u, B) ≈ K
    @test surface_residual(b, n, u, B) < 1.0e-12
    # any other velocity: its kinetic-metric projection, and a residual between 0 and 1
    u2 = copy(u)
    u2[b.nodes] .*= 1 .+ 0.3 .* (2 .* rand(rng, length(b.nodes)) .- 1)
    K2 = galerkin_flux(b, n, u2, B)
    P, V = b.P, b.V
    Bk, nk = B[b.nodes], n[b.nodes]
    @test P' * (V .* Bk .* (Bk ./ nk .* (P * K2))) ≈ P' * (V .* Bk .* u2[b.nodes])
    @test 0 < surface_residual(b, n, u2, B) < 1
end

@testitem "Closed-surface diagnostics: kinetic energy, surface statistics, current beyond the cut" setup = [ClosedSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: closed_surface_basis, electron_kinetic_energy, surface_statistics, current_outside_constraint
    me, ee = 9.1093837e-31, 1.602176634e-19
    G = grid(31)
    b = closed_surface_basis(G, ellipse(G), admissible_ellipse(G))
    # kinetic energy: ½ mₑ n u² over the toroidal control volumes of the nodes given
    n = fill(1.0e17, G.NR, G.NZ)
    u = fill(2.0e5, G.NR, G.NZ)
    E = electron_kinetic_energy(G, n, u, b.nodes; me)
    @test E ≈ sum(0.5 * me * 1.0e17 * 4.0e10 * 2π * G.R2D[k] * G.dR * G.dZ for k in b.nodes)
    # per hat: the smallest density, the largest drift against the thermal speed and the largest
    # relative imbalance of the nodes it covers
    ne = @. 1.0e17 * (1 + 0.5 * (G.R2D - R0) / 0.3)
    ni = 1.02 .* ne
    Te = fill(10.0, G.NR, G.NZ)
    ue, ui = fill(1.0e5, G.NR, G.NZ), zeros(G.NR, G.NZ)
    st = surface_statistics(b, ne, ni, ue, ui, Te; me, ee)
    @test length(st.n_min) == length(b.knots)
    @test all(st.drift .≈ 1.0e5 / sqrt(10.0 * ee / me))
    @test all(st.imbalance .≈ 0.02)
    cov = [findall(>(0), b.P[:, s]) for s in eachindex(b.knots)]
    @test st.n_min ≈ [minimum(ne[b.nodes[r]]) for r in cov]
    # current carried by closed nodes the constraint leaves out (outside Ω_c or beyond the cut)
    closed = findall(vec(admissible_ellipse(G)))
    J = ones(G.NR, G.NZ)
    f = current_outside_constraint(b, closed, J)
    out = count(k -> !(k in b.nodes) || b.ψN[searchsortedfirst(b.nodes, k)] > 1, closed)
    @test f ≈ out / length(closed)
    @test 0 < f < 0.5
end

@testitem "Closed-surface policy: free flow by default, the constraint by choice" tags = [:numerics] begin
    using RAPID2D: FreeParallelFlow, SurfaceFluxConstraint, PostProjection, ClosedSurfaceState
    @test SimulationFlags{Float64}().closed_surface_flow isa FreeParallelFlow
    @test SurfaceFluxConstraint().coupling isa PostProjection
    s = ClosedSurfaceState{Float64}(5, 7)
    @test s.basis === nothing && isempty(s.K)
    @test size(s.ΓR) == (4, 7) && size(s.ΓZ) == (5, 6)
end

@testitem "Constrained momentum: surface flux inside the cut, ordinary rows elsewhere" setup = [ClosedSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: closed_surface_basis, constrained_momentum_solve
    using RAPID2D.SparseArrays: sparse, spdiagm, SparseMatrixCSC
    using LinearAlgebra: Diagonal, I
    using Random: MersenneTwister
    rng = MersenneTwister(11)
    G = grid(31)
    b = closed_surface_basis(G, ellipse(G), admissible_ellipse(G))
    N = G.NR * G.NZ
    B = vec(@. sqrt((3.0 / G.R2D)^2 + 1.0e-6))
    n = vec(@. 1.0e17 * (1 + 0.5 * (G.R2D - R0) / 0.3))
    ν = vec(@. 1.0e6 * (1 + 0.6 * (G.Z2D - Z0) / 0.3))
    dt = 1.0e-6
    # a diagonal step operator: friction only, so open rows decouple
    A = spdiagm(0 => 1 .+ dt .* ν)
    r = randn(rng, N) .* 1.0e4
    u, K = constrained_momentum_solve(A, r, b, n, B)
    inside = [b.nodes[k] for k in eachindex(b.nodes) if b.ψN[k] <= 1]
    rest = setdiff(1:N, inside)
    # the constraint, the ordinary rows, and the Galerkin rows
    @test n[inside] .* u[inside] ./ B[inside] ≈ b.P[b.ψN .<= 1, :] * K
    @test A[rest, :] * u ≈ r[rest]
    Pin = b.P[b.ψN .<= 1, :]
    Vin = b.V[b.ψN .<= 1]
    Q = Pin' * Diagonal(Vin .* B[inside])
    f = (A * u .- r)[inside]
    @test maximum(abs, Q * f) <= 1.0e-10 * maximum(abs, Q * abs.(r[inside]))
    # (P.5): the constraint force does no work on the constrained flow
    @test abs(sum(Vin .* n[inside] .* u[inside] .* f)) <= 1.0e-10 * sum(abs.(Vin .* n[inside] .* u[inside] .* f))
    # with friction only, the series formula: (Pᵀ V B a B/n P) K = Pᵀ V B r
    a = 1 .+ dt .* ν[inside]
    @test K ≈ (Pin' * Diagonal(Vin .* B[inside] .^ 2 .* a ./ n[inside]) * Pin) \ (Pin' * (Vin .* B[inside] .* r[inside]))
    # nodes of Ω_c beyond the cut keep their ordinary rows: their current is not removed
    beyond = [b.nodes[k] for k in eachindex(b.nodes) if b.ψN[k] > 1]
    @test !isempty(beyond) && u[beyond] ≈ r[beyond] ./ (1 .+ dt .* ν[beyond])
    # a coupling across the cut (a transport-like stencil) is honoured by both kinds of rows
    L = spdiagm(1 => fill(-0.1, N - 1), -1 => fill(-0.1, N - 1), 0 => fill(0.2, N))
    A2 = A + L
    u2, K2 = constrained_momentum_solve(A2, r, b, n, B)
    @test A2[rest, :] * u2 ≈ r[rest]
    @test maximum(abs, Q * (A2 * u2 .- r)[inside]) <= 1.0e-10 * maximum(abs, Q * abs.(r[inside]))
    # no basis: the ordinary solve
    u0, K0 = constrained_momentum_solve(A2, r, nothing, n, B)
    @test u0 ≈ A2 \ r && isempty(K0)
end

@testitem "Closed-surface basis: a hole inside the region (the axis node) does not set the cut" setup = [ClosedSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: closed_surface_basis
    G = grid(31)
    ψ = ellipse(G)
    A = admissible_ellipse(G)
    b0 = closed_surface_basis(G, ψ, A)
    # field-line following ends at the null of the node nearest the O-point and leaves it out
    k0 = argmin(vec(ψ))
    A[k0] = false
    b = closed_surface_basis(G, ψ, A)
    @test b !== nothing
    @test b.ψ_cut == b0.ψ_cut
    @test k0 in b.nodes
end

@testitem "Closed-surface flux: no flux outside Ω_c where ψ leaves the region's range" setup = [ClosedSurfaceFixtures] tags = [:numerics] begin
    using RAPID2D: closed_surface_basis, stream_function_fluxes, face_outflow
    G = grid(41)
    # a deeper well away from the region, outside the admissible nodes: ψ there runs below the
    # axis level, which a flux function of ψ alone would turn into flux
    ψ = @. ellipse(G.R2D, G.Z2D) - 0.3 * c * exp(-((G.R2D - 1.88)^2 + (G.Z2D - 0.38)^2) / 0.004)
    A = admissible_ellipse(G; a = 0.25)
    b = closed_surface_basis(G, ψ, A)
    @test b !== nothing
    @test minimum(b.corner_ψN) < 0                     # the case this test is about
    ΓR, ΓZ = stream_function_fluxes(b, [1 + x for x in b.knots])
    inΩ = falses(G.NR, G.NZ)
    inΩ[b.nodes] .= true
    @test all(iszero(ΓR[i, j]) for i in 1:(G.NR - 1), j in 1:G.NZ if !(inΩ[i, j] && inΩ[i + 1, j]))
    @test all(iszero(ΓZ[i, j]) for i in 1:G.NR, j in 1:(G.NZ - 1) if !(inΩ[i, j] && inΩ[i, j + 1]))
    @test maximum(abs, face_outflow(ΓR, ΓZ)) <= 1.0e-13 * maximum(abs, ΓR)
end
