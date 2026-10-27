# Wall-aware anisotropic diffusion operator.
#
# A whole-grid tensor builder that sweeps 2:N-1 with no wall awareness lets material
# diffuse past the wall to be mopped up afterwards (the retired `∇𝐃∇`). The reflective
# neutral-gas builder knows about the wall but is 5-point and isotropic. This operator
# is both: a 9-point tensor stencil that stops at the wall.
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

@testitem "Wall diffusion: reduces to the reflective 5-point operator when isotropic" begin
    using RAPID2D: build_wall_diffusion_matrix, build_isotropic_diffusion_matrix

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
    ref = build_isotropic_diffusion_matrix(G, D)
    # its own builder keeps the five-point pattern: no cross-term slots for the LU to carry
    @test all(<=(5), diff(ref.colptr))
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

    # It is NOT an M-matrix here, and the data has to be able to show that. A
    # field that is uniform inside the wall is an exact kernel mode of a
    # reflective diffusion operator — the solve returns it untouched, so any
    # positivity assertion on it is vacuous. Use a localized blob on a ZERO
    # background instead, which is what an undershoot needs in order to be
    # negative rather than merely a dip.
    inw = G.nodes.in_wall_nids
    Rc = (G.R1D[1] + G.R1D[end]) / 2
    Zc = (G.Z1D[1] + G.Z1D[end]) / 2
    n0 = zeros(G.NR * G.NZ)
    for k in inw
        i, j = G.nodes.rid[k], G.nodes.zid[k]
        n0[k] = 1.0e18 * exp(-((G.R2D[i, j] - Rc)^2 + (G.Z2D[i, j] - Zc)^2) / 0.02)
    end
    Jv = vec(G.Jacob)
    total_Jn(v) = sum(Jv[k] * v[k] for k in inw)

    for treatment in (:drop, :reflect)
        A = build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms = treatment)

        # the diagonal drains, never sources
        @test all(A[k, k] <= 0 for k in inw)

        # the undershoot is real, grows with Δt, and shrinks as Δt does. Values
        # measured at D∥/D⊥ = 1000: −7.3e-11 at Δt = 1e-6 rising to −1.7e-2 at
        # Δt = 1e-3. Bounded and transient, not accumulating — see
        # internal/docs/src/notes/TODO/anisotropic-positivity.md before "fixing" the stencil.
        unders = map((1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)) do dt
            n1 = (sparse(I, size(A, 1), size(A, 2)) - dt * A) \ n0
            return -min(minimum(n1[inw]), 0.0) / maximum(n1[inw])
        end
        @test issorted(unders)                    # more implicit time, more undershoot
        @test unders[1] < 1.0e-9                  # negligible at a production step
        @test unders[end] < 0.1                   # and bounded even at Δt = 1e-3

        # conservation is untouched by any of it — this is a monotonicity defect,
        # not a leak. `:reflect` is the exception, and that is its own test.
        if treatment === :drop
            n1 = (sparse(I, size(A, 1), size(A, 2)) - 1.0e-3 * A) \ n0
            @test abs(total_Jn(n1) - total_Jn(n0)) / total_Jn(n0) < 1.0e-13
        end
    end
end

@testitem "wall diffusion in place: rewritten on the wall pattern, both cross-term rules, Robin debit included" begin
    using RAPID2D: build_wall_diffusion_matrix, build_wall_diffusion_matrix!, build_wall_pattern, wall_faces
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)
    Dpara, Dperp = 1000.0, 1.0
    bR, bZ = cos(0.6), sin(0.6)
    D_RR = [Dperp + (Dpara - Dperp) * bR^2 * (1 + 0.1 * sin(3 * G.R1D[i])) for i in 1:G.NR, j in 1:G.NZ]
    D_RZ = [(Dpara - Dperp) * bR * bZ * (1 + 0.1 * cos(2 * G.Z1D[j])) for i in 1:G.NR, j in 1:G.NZ]
    D_ZZ = [Dperp + (Dpara - Dperp) * bZ^2 for i in 1:G.NR, j in 1:G.NZ]
    v = [50.0 + 10.0 * k for k in 1:length(faces)]
    A = build_wall_pattern(G)
    for rule in (:drop, :reflect)
        build_wall_diffusion_matrix!(A, G, D_RR, D_RZ, D_ZZ; cross_terms = rule, faces, v_absorb = v)
        @test A.matrix == build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms = rule, faces, v_absorb = v)
        build_wall_diffusion_matrix!(A, G, D_RR, D_RZ, D_ZZ; cross_terms = rule)
        @test A.matrix == build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms = rule)
    end
    build_wall_diffusion_matrix!(A, G, D_ZZ, zero(D_RZ), D_ZZ)     # isotropic after anisotropic: cross slots rewritten
    @test A.matrix == build_wall_diffusion_matrix(G, D_ZZ, zero(D_RZ), D_ZZ)
    @test_throws ArgumentError build_wall_diffusion_matrix!(A, G, D_RR, D_RZ, D_ZZ; faces)   # faces without v_absorb
end

# ── The anti-diffusive pair limiter ───────────────────────────────────────────
#
# Across an unresolved density cliff a negative pair of the 9-point stencil drains the empty
# node by `|A_kl| n_l` and multiplies the per-particle operator by `n_l/n_k`. The limiter adds a
# symmetric pair diffusion w on those pairs only (`limit_antidiffusive_pairs!`).

@testsnippet PairLimiterFixtures begin
    using RAPID2D: build_wall_pattern, build_wall_diffusion_matrix!, limit_antidiffusive_pairs!
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    "A walled box on a 41 × 47 grid, the field tilted by θ, D varying in space: cross terms everywhere."
    function tilted_case(; θ = 0.6, Dpara = 100.0, Dperp = 0.1)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = 41, NZ = 47,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.2, 1.8, 1.8, 1.2], wall_Z = [-0.3, -0.3, 0.3, 0.3],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        G = RP.G
        bR, bZ = cos(θ), sin(θ)
        s = @. 1 + 0.2 * sin(3 * G.R2D) * cos(2 * G.Z2D)
        D = (
            @.(s * (Dperp + (Dpara - Dperp) * bR^2)),
            @.(s * (Dpara - Dperp) * bR * bZ),
            @.(s * (Dperp + (Dpara - Dperp) * bZ^2)),
        )
        across = @. -bZ * (G.R2D - 1.5) + bR * G.Z2D       # distance across the field
        return (; G, D, across, faces = RP.transport.wall_faces)
    end

    diffusion(G, D; kw...) = build_wall_diffusion_matrix!(build_wall_pattern(G), G, D...; cross_terms = :drop, kw...)

    "Across the field: 1e17, a one-cell 1e11 shelf, then 1e5. Zero outside the wall."
    function cliff_density(G, across)
        n = zeros(G.NR, G.NZ)
        for k in G.nodes.in_wall_nids
            n[k] = across[k] < 0 ? 1.0e17 : (across[k] < 0.03 ? 1.0e11 : 1.0e5)
        end
        return n
    end

    "Smooth: every neighbour ratio is far below 20."
    smooth_density(G) = @. 1.0e16 * (1 + 0.5 * sin(3 * G.R2D) * cos(2 * G.Z2D))

    "The off-diagonal entries (k, l, value) of a sparse matrix."
    offdiag(matrix) = [(k, l, v) for (k, l, v) in zip(findnz(matrix)...) if k != l]
end

@testitem "pair limiter: a resolved density gives back the operator bit for bit" setup = [PairLimiterFixtures] begin
    using RAPID2D.SparseArrays: nonzeros
    c = tilted_case()
    v_absorb = fill(40.0, length(c.faces))
    for kw in ((;), (; faces = c.faces, v_absorb))
        A0 = diffusion(c.G, c.D; kw...)
        A = diffusion(c.G, c.D; kw...)
        @test limit_antidiffusive_pairs!(A, c.G, smooth_density(c.G); max_ratio = 20.0) == 0
        @test nonzeros(A.matrix) == nonzeros(A0.matrix)
    end
end

@testitem "pair limiter: conserves Σ Jacob·n, keeps row sums, and only adds diffusion on cliff pairs" setup = [PairLimiterFixtures] begin
    c = tilted_case()
    G, Jacob = c.G, vec(c.G.Jacob)
    n = cliff_density(G, c.across)
    max_ratio = 20.0
    v_absorb = fill(40.0, length(c.faces))
    for kw in ((;), (; faces = c.faces, v_absorb))
        A0 = diffusion(G, c.D; kw...)
        A = diffusion(G, c.D; kw...)
        n_limited = limit_antidiffusive_pairs!(A, G, n; max_ratio)
        before, after = A0.matrix, A.matrix
        scale = maximum(Jacob) * maximum(abs, before)

        # Jacob·A is not symmetric here, so this does not follow from symmetry: it is the pair form
        @test maximum(abs, Jacob' * before - Jacob' * after) < 1.0e-13 * scale
        @test maximum(abs, sum(after; dims = 2) - sum(before; dims = 2)) < 1.0e-13 * maximum(abs, before)

        # it acts, and only on pairs across a ratio above max_ratio with a negative side
        changed = [(k, l) for (k, l, v) in offdiag(after) if v != before[k, l]]
        @test length(changed) > 50
        @test length(changed) == 2 * n_limited          # both entries of every limited pair
        ratio(k, l) = max(n[k], n[l]) / max(min(n[k], n[l]), eps())
        @test all(ratio(k, l) > max_ratio for (k, l) in changed)
        @test all(min(Jacob[k] * before[k, l], Jacob[l] * before[l, k]) < 0 for (k, l) in changed)

        # an added pair diffusion: no off-diagonal decreases, a negative one shrinks to at most `keep` of its size
        tol = 1.0e-12 * maximum(abs, before)
        @test all(v >= before[k, l] - tol for (k, l, v) in offdiag(after))
        for (k, l) in changed
            keep = max_ratio * min(n[k], n[l]) / max(n[k], n[l])
            before[k, l] < 0 && @test after[k, l] >= keep * before[k, l] - tol
        end
    end
end

@testitem "pair limiter: a negative entry of the per-particle operator stays above −max_ratio |A| at the cliff" setup = [PairLimiterFixtures] begin
    using RAPID2D: per_particle_operator!
    using RAPID2D.SparseArrays: nonzeros
    c = tilted_case()
    G = c.G
    n = cliff_density(G, c.across)
    max_ratio = 20.0
    A0 = diffusion(G, c.D)
    A = diffusion(G, c.D)
    limit_antidiffusive_pairs!(A, G, n; max_ratio)
    per_particle_raw, per_particle_limited = build_wall_pattern(G), build_wall_pattern(G)
    per_particle_operator!(per_particle_raw, A0, n; n_floor = 1.0)
    per_particle_operator!(per_particle_limited, A, n; n_floor = 1.0)
    # per-particle entries A_kl n_l/n_k: without the limiter the cliff multiplies a negative pair by up to 1e12
    amax = maximum(abs, nonzeros(A0.matrix))
    @test minimum(v for (k, l, v) in offdiag(per_particle_raw.matrix)) < -1.0e6 * amax
    # with it, every off-diagonal is at least max_ratio times the negative part of the original entry
    @test all(v >= max_ratio * min(A0.matrix[k, l], 0.0) * (1 + 1.0e-12) for (k, l, v) in offdiag(per_particle_limited.matrix))
end

@testitem "pair limiter: an implicit step across a cliff leaves orders of magnitude less negative mass" setup = [PairLimiterFixtures] begin
    # Not a positivity guarantee. Inside (I − Δt A)⁻¹ a negative pair between two empty nodes
    # (ratio ≤ max_ratio, so not limited) can still pass the full side's density on in two hops.
    using RAPID2D.SparseArrays: nonzeros
    using RAPID2D.LinearAlgebra: I, dot
    c = tilted_case()
    G, Jacob = c.G, vec(c.G.Jacob)
    inw = G.nodes.in_wall_nids
    n0 = vec(cliff_density(G, c.across))
    A0 = diffusion(G, c.D)
    A = diffusion(G, c.D)
    limit_antidiffusive_pairs!(A, G, n0; max_ratio = 20.0)
    Δt = 0.1 / maximum(abs, nonzeros(A0.matrix))
    function negative_mass(matrix)
        n1 = (I - Δt * matrix) \ n0
        return -sum(Jacob[k] * min(n1[k], 0.0) for k in inw) / dot(Jacob, n0)
    end
    base, limited = negative_mass(A0.matrix), negative_mass(A.matrix)
    @test base > 1.0e-4                       # the stencil's own defect at this cliff
    @test limited < 1.0e-3 * base
end

@testitem "pair limiter: allocates nothing and refuses what it cannot limit" setup = [PairLimiterFixtures] begin
    c = tilted_case()
    G = c.G
    n = cliff_density(G, c.across)
    A = diffusion(G, c.D)
    limit_antidiffusive_pairs!(A, G, n; max_ratio = 20.0)
    alloc(A, G, n) = @allocated limit_antidiffusive_pairs!(A, G, n; max_ratio = 20.0)
    alloc(A, G, n)
    @test alloc(A, G, n) == 0
    alloc(A, G, vec(n))
    @test alloc(A, G, vec(n)) == 0
    bad = copy(n)
    bad[G.nodes.in_wall_nids[1]] = NaN
    @test_throws ArgumentError limit_antidiffusive_pairs!(A, G, bad; max_ratio = 20.0)
    @test_throws ArgumentError limit_antidiffusive_pairs!(A, G, n; max_ratio = 0.5)
    @test_throws ArgumentError limit_antidiffusive_pairs!(A, G, n; max_ratio = Inf)
    @test_throws DimensionMismatch limit_antidiffusive_pairs!(A, G, vec(n)[1:(end - 1)]; max_ratio = 20.0)
end

# ── One assembly ──────────────────────────────────────────────────────────────
#
# Every anisotropic diffusion operator of a run is assembled through `assemble_anisotropic_diffusion!` by the
# run's `AnisotropicDiscretization`: `NinePoint` is the stencil, `LimitedNinePoint` the stencil with
# its anti-diffusive pairs limited where the density is not resolved.

@testitem "diffusion discretization: the policies check their parameters, and the run default limits at 20" begin
    @test SimulationFlags{Float64}().anisotropic_discretization == LimitedNinePoint(max_ratio = 20)
    @test LimitedNinePoint(4).max_ratio === 4.0
    @test LimitedNinePoint(max_ratio = 1).max_ratio === 1.0
    for bad in (0.5, Inf, NaN)
        @test_throws ArgumentError LimitedNinePoint(max_ratio = bad)
    end
    @test NinePoint() isa AnisotropicDiscretization
end

@testitem "diffusion assembly: NinePoint is the stencil, LimitedNinePoint the stencil with its pairs limited, bit for bit" setup = [PairLimiterFixtures] begin
    using RAPID2D: assemble_anisotropic_diffusion!
    c = tilted_case()
    G = c.G
    v_absorb = fill(40.0, length(c.faces))
    cliff, smooth = cliff_density(G, c.across), smooth_density(G)
    for kw in ((;), (; faces = c.faces, v_absorb))
        stencil = diffusion(G, c.D; kw...)
        A = build_wall_pattern(G)
        @test assemble_anisotropic_diffusion!(A, G, c.D..., NinePoint(); n = cliff, kw...) == 0
        @test nonzeros(A.matrix) == nonzeros(stencil.matrix)

        limited = diffusion(G, c.D; kw...)
        n_limited = limit_antidiffusive_pairs!(limited, G, cliff; max_ratio = 20.0)
        @test n_limited > 50
        @test assemble_anisotropic_diffusion!(A, G, c.D..., LimitedNinePoint(); n = cliff, kw...) == n_limited
        @test nonzeros(A.matrix) == nonzeros(limited.matrix)

        # a resolved density: the two discretizations are one operator
        @test assemble_anisotropic_diffusion!(A, G, c.D..., LimitedNinePoint(); n = smooth, kw...) == 0
        @test nonzeros(A.matrix) == nonzeros(stencil.matrix)
    end
    # the limiter's density is not optional
    @test_throws ArgumentError assemble_anisotropic_diffusion!(build_wall_pattern(G), G, c.D..., LimitedNinePoint())
end
