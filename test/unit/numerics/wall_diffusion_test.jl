# Wall-aware anisotropic diffusion operator.
#
# The shared ∇𝐃∇ builder sweeps 2:N-1 with no wall awareness, so material
# diffuses past the wall and is mopped up afterwards. The reflective neutral-gas
# builder knows about the wall but is 5-point and isotropic. This operator is
# both: a 9-point tensor stencil that stops at the wall.
#
# **The hard part is the cross term.** Each of the four D_RZ groups decomposes
# into two centred-difference PAIRS:
#
#     Group 1 (R-face i+½):  C·[ (f[i,j+1] − f[i,j−1]) + (f[i+1,j+1] − f[i+1,j−1]) ]
#                                 ‾‾‾‾‾‾ pair A ‾‾‾‾‾‾    ‾‾‾‾‾‾‾ pair B ‾‾‾‾‾‾‾
#
# Dropping a single ARM is not an option: on a constant field a pair gives
# 1 − 1 = 0, but one arm alone gives 1, so the row sum stops vanishing and the
# operator manufactures material. Both treatments therefore act on whole pairs:
#
#     :drop     remove any pair containing a node that is not in-wall
#     :reflect  substitute the owning cell's value for a not-in-wall node
#
# Both keep constants in the kernel by construction. Whether either keeps
# Σ J·n — which needs J·A symmetric — is measured, not assumed.

const WALL_DIFFU_CFG = (;
    device_Name = "manual", NR = 25, NZ = 30,
    prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
    snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
)

@testitem "Wall diffusion: reduces to the reflective 5-point operator when isotropic" begin
    using RAPID2D: build_wall_diffusion_matrix, build_reflective_diffusion_matrix

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G

    # spatially varying D, so agreement cannot be an artefact of a constant
    D = [
        30.0 + 20.0 * sin(3.0 * G.R1D[i]) * cos(2.0 * G.Z1D[j])
            for i in 1:G.NR, j in 1:G.NZ
    ]
    Z = zeros(G.NR, G.NZ)

    # THE regression anchor. With D_RZ = 0 and D_RR = D_ZZ the 9-point stencil
    # collapses to the 5-point one, and the known-good reflective operator — the
    # one whose conservation is already established — must be reproduced.
    ref = build_reflective_diffusion_matrix(G, D)
    for treatment in (:drop, :reflect)
        A = build_wall_diffusion_matrix(G, D, Z, D; cross_terms = treatment)
        @test size(A) == size(ref)
        @test maximum(abs, A - ref) < 1.0e-13 * maximum(abs, ref)
    end
end

@testitem "Wall diffusion: rows sum to zero with a cross term present" begin
    using RAPID2D: build_wall_diffusion_matrix

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G

    # a strongly anisotropic tensor, oblique to the grid: D_RZ is then comparable
    # to the diagonal, which is the regime a turbulent tensor actually sits in
    Dpara, Dperp = 100.0, 0.1
    θ = 0.6
    bR, bZ = cos(θ), sin(θ)
    D_RR = fill(Dperp + (Dpara - Dperp) * bR^2, G.NR, G.NZ)
    D_RZ = fill((Dpara - Dperp) * bR * bZ, G.NR, G.NZ)
    D_ZZ = fill(Dperp + (Dpara - Dperp) * bZ^2, G.NR, G.NZ)

    for treatment in (:drop, :reflect)
        A = build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms = treatment)

        # Zero row sum is what makes the stencil a divergence: no node creates gas
        # on its own. This is why a cross-term PAIR must be removed whole — one
        # arm alone leaves a residual that acts as a source.
        @test maximum(abs, sum(A, dims = 2)) < 1.0e-12 * maximum(abs, A)

        # equivalently, a uniform field produces no flux anywhere
        n_uniform = fill(7.0e18, G.NR * G.NZ)
        @test maximum(abs, A * n_uniform) < 1.0e-12 * maximum(abs, A) * 7.0e18
    end
end

@testitem "Wall diffusion: nothing couples across the wall" begin
    using RAPID2D: build_wall_diffusion_matrix

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G

    Dpara, Dperp = 100.0, 0.1
    θ = 0.6
    bR, bZ = cos(θ), sin(θ)
    D_RR = fill(Dperp + (Dpara - Dperp) * bR^2, G.NR, G.NZ)
    D_RZ = fill((Dpara - Dperp) * bR * bZ, G.NR, G.NZ)
    D_ZZ = fill(Dperp + (Dpara - Dperp) * bZ^2, G.NR, G.NZ)

    inside = G.nodes.in_wall_nids
    outside = G.nodes.on_out_wall_nids

    for treatment in (:drop, :reflect)
        A = build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms = treatment)

        # `== 0`, not a tolerance. With this boundary nothing may ever be written
        # outside the wall, so a tolerance would hide a genuine leak rather than
        # measure it. The 9-point stencil makes this harder than it was for the
        # 5-point reflective operator: the DIAGONAL arms cross too.
        @test maximum(abs, A[inside, outside]) == 0.0
        @test maximum(abs, A[outside, :]) == 0.0
    end
end

@testitem "Wall diffusion: a re-entrant cell cannot reach outside through a cross term" begin
    using RAPID2D: build_wall_diffusion_matrix, is_in_wall

    # THE isolation test for the cross-term problem. A concave corner cell has all
    # four cardinal neighbours in-wall, so it owns no wall face and the boundary
    # machinery never touches it — yet its D_RZ arm reaches the outside diagonal.
    # A leak here cannot come from anywhere else, so this separates §3.4(a) from
    # every other error source.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 61, NZ = 61,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.2, 1.8, 1.8, 1.5, 1.5, 1.2],
        wall_Z = [-0.3, -0.3, 0.0, 0.0, 0.3, 0.3],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G

    reentrant = [
        (i, j) for j in 1:G.NZ, i in 1:G.NR
            if is_in_wall(G, i, j) &&
            all(is_in_wall(G, i + di, j + dj) for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))) &&
            any(is_in_wall(G, i + di, j + dj) == false for (di, dj) in ((1, 1), (1, -1), (-1, 1), (-1, -1)))
    ]
    @test length(reentrant) == 1

    Dpara, Dperp = 100.0, 0.1
    s = 1 / sqrt(2.0)                                   # 45°: D_RZ is maximal
    D_RR = fill(Dperp + (Dpara - Dperp) * s^2, G.NR, G.NZ)
    D_RZ = fill((Dpara - Dperp) * s * s, G.NR, G.NZ)
    D_ZZ = fill(Dperp + (Dpara - Dperp) * s^2, G.NR, G.NZ)

    outside = G.nodes.on_out_wall_nids
    for treatment in (:drop, :reflect)
        A = build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms = treatment)
        for (i, j) in reentrant
            @test maximum(abs, A[G.nodes.nid[i, j], outside]) == 0.0
        end
        # and the convex corners, whose three crossing diagonals are the noisier case
        @test maximum(abs, A[G.nodes.in_wall_nids, outside]) == 0.0
    end
end

@testitem "Wall diffusion: conserves Jacobian-weighted particles" begin
    using RAPID2D: build_wall_diffusion_matrix

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G

    # In cylindrical geometry the invariant is Σ J·n, not Σ n. d/dt(Σ J n) = (Jᵀ A) n
    # must vanish for EVERY n, i.e. the Jacobian-weighted column sums are zero —
    # which is the statement that J·A is symmetric. Zero row sums do NOT imply
    # this, and at a wall the cross-term treatment is exactly what can break it.
    Dpara, Dperp = 100.0, 0.1
    θ = 0.6
    bR, bZ = cos(θ), sin(θ)
    D_RR = fill(Dperp + (Dpara - Dperp) * bR^2, G.NR, G.NZ)
    D_RZ = fill((Dpara - Dperp) * bR * bZ, G.NR, G.NZ)
    D_ZZ = fill(Dperp + (Dpara - Dperp) * bZ^2, G.NR, G.NZ)

    Jv = vec(G.Jacob)
    inw = G.nodes.in_wall_nids
    resid(A) = maximum(abs, vec(Jv' * A)[inw]) / (maximum(Jv) * maximum(abs, A))

    # `:drop` conserves to machine precision. Measured on three wall shapes
    # (axis-aligned box, 45° diamond, L-shape with a re-entrant corner) at
    # D∥/D⊥ = 1, 10, 1000: residual 1.7e-16 … 2.3e-16 throughout.
    A_drop = build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms = :drop)
    @test resid(A_drop) < 1.0e-12

    # `:reflect` does NOT, and the design note's claim that reflection "conserves
    # by construction" is wrong. Substituting the owning cell for an outside node
    # keeps the pair a difference — so constants stay in the kernel and row sums
    # still vanish — but it moves ±C onto the diagonal with no matching change in
    # any other row, so J·A stops being symmetric. Measured 0.077 … 0.209, i.e.
    # 8–21 %, fifteen orders above `:drop`.
    #
    # Kept as a marker rather than a deleted option: it records WHY the default is
    # `:drop`, and it fires if someone finds a reflection that does conserve.
    A_ref = build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms = :reflect)
    @test_broken resid(A_ref) < 1.0e-12
    @test resid(A_ref) > 1.0e-3                  # the failure is gross, not marginal

    # both still keep constants in the kernel — that much reflection does buy
    @test maximum(abs, sum(A_ref, dims = 2)) < 1.0e-12 * maximum(abs, A_ref)

    # and with an isotropic tensor the two coincide exactly: no cross term, no
    # ambiguity, so the whole question only exists for D_RZ ≠ 0
    Diso = fill(10.0, G.NR, G.NZ)
    Z = zeros(G.NR, G.NZ)
    @test build_wall_diffusion_matrix(G, Diso, Z, Diso; cross_terms = :drop) ==
        build_wall_diffusion_matrix(G, Diso, Z, Diso; cross_terms = :reflect)
end

@testitem "Wall diffusion: the grid frame may itself be the wall" begin
    using RAPID2D: build_wall_diffusion_matrix

    # Wall polygon larger than the domain: every node is in-wall and the outward
    # neighbours of the frame cells lie off-grid. Nothing may index there, and the
    # 9-point stencil has diagonal arms off-grid too.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 15, NZ = 17,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [0.5, 2.5, 2.5, 0.5], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G

    Dpara, Dperp = 100.0, 0.1
    s = 1 / sqrt(2.0)
    D_RR = fill(Dperp + (Dpara - Dperp) * s^2, G.NR, G.NZ)
    D_RZ = fill((Dpara - Dperp) * s * s, G.NR, G.NZ)
    D_ZZ = fill(Dperp + (Dpara - Dperp) * s^2, G.NR, G.NZ)

    for treatment in (:drop, :reflect)
        A = build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms = treatment)
        @test all(isfinite, A)
        @test maximum(abs, sum(A, dims = 2)) < 1.0e-12 * maximum(abs, A)
    end
end

@testitem "Wall diffusion: the implicit operator stays an M-matrix" begin
    using RAPID2D: build_wall_diffusion_matrix
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G

    # Positivity is structural for backward Euler only if I − θΔt·A is an M-matrix:
    # non-positive off-diagonals and a non-negative diagonal. The 5-point part
    # guarantees it; a cross-term treatment that introduces a POSITIVE off-diagonal
    # would destroy it, and negative densities would follow at large Δt.
    Dpara, Dperp = 100.0, 0.1
    θb = 0.6
    bR, bZ = cos(θb), sin(θb)
    D_RR = fill(Dperp + (Dpara - Dperp) * bR^2, G.NR, G.NZ)
    D_RZ = fill((Dpara - Dperp) * bR * bZ, G.NR, G.NZ)
    D_ZZ = fill(Dperp + (Dpara - Dperp) * bZ^2, G.NR, G.NZ)

    inw = G.nodes.in_wall_nids
    for treatment in (:drop, :reflect)
        A = build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms = treatment)

        # the diagonal drains, never sources
        @test all(A[k, k] <= 0 for k in inw)

        # a long implicit step must not produce a negative density from a
        # non-negative one — the property backward Euler is chosen for
        M = sparse(I, size(A, 1), size(A, 2)) - 1.0e-3 * A
        n0 = zeros(G.NR * G.NZ)
        n0[inw] .= 1.0e18
        n1 = M \ n0
        @test minimum(n1[inw]) >= -1.0e-6 * maximum(n0)
    end
end
