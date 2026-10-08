# The per-particle operators: what a density operator does to the variable its particles carry.
#
# A conservative operator A advances a density, dn/dt = A n. A variable f carried by those
# particles (u∥, Tₑ, the energy per electron) then obeys df/dt = M f with
#
#     M = N⁻¹ (A N − diag(A n)),        N = diag(n),
#
# the one operator whose primitive form reproduces d(n f)/dt = A (n f) row by row. Advection
# (A = the face-flux divergence) and mixing (A = the density diffusion) are the same
# construction. The dissipation rate Γ_M(u) = ½ M u² − u ⊙ M u is the kinetic energy the
# exchange of momentum turns into heat.
# internal/docs/src/reference/electron-diffusive-transport.md §5;
# internal/docs/src/notes/design/turbulent-mixing-u-T.md §4.

@testsnippet PerParticleFixtures begin
    using RAPID2D: build_wall_pattern, build_wall_diffusion_matrix!

    "A walled box, or a staircase wall with a re-entrant corner, on a 41 × 47 grid."
    function walled_RP(wall::Symbol)
        wall_R, wall_Z = if wall === :box
            [1.2, 1.8, 1.8, 1.2], [-0.3, -0.3, 0.3, 0.3]
        else
            [1.2, 1.8, 1.8, 1.5, 1.5, 1.2], [-0.3, -0.3, 0.0, 0.0, 0.3, 0.3]
        end
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = 41, NZ = 47,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = wall_R, wall_Z = wall_Z,
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        return RP
    end

    "D = s(R, Z)·(D⊥ 𝟙 + (D∥ − D⊥) b bᵀ) along one tilted direction b: cross terms everywhere."
    function tilted_tensor(G; θ = 0.6, Dpara = 100.0, Dperp = 0.1)
        bR, bZ = cos(θ), sin(θ)
        s = @. 1 + 0.2 * sin(3 * G.R2D) * cos(2 * G.Z2D)
        DRR = @. s * (Dperp + (Dpara - Dperp) * bR^2)
        DRZ = @. s * (Dpara - Dperp) * bR * bZ
        DZZ = @. s * (Dperp + (Dpara - Dperp) * bZ^2)
        return (DRR, DRZ, DZZ)
    end

    "The tensor aligned with the poloidal field of an X-point at (R0, Z0): R B_R = y, R B_Z = x."
    function xpoint_tensor(G; R0 = 1.5, Z0 = 0.0, Dpara = 100.0, Dperp = 0.1)
        DRR = zeros(G.NR, G.NZ)
        DRZ = zeros(G.NR, G.NZ)
        DZZ = zeros(G.NR, G.NZ)
        for j in 1:G.NZ, i in 1:G.NR
            x, y = G.R2D[i, j] - R0, G.Z2D[i, j] - Z0
            BR, BZ = y / G.R2D[i, j], x / G.R2D[i, j]
            Bp = hypot(BR, BZ)
            bR, bZ = Bp > 0 ? (BR / Bp, BZ / Bp) : (0.0, 0.0)   # isotropic at the null
            DRR[i, j] = Dperp + (Dpara - Dperp) * bR^2
            DRZ[i, j] = (Dpara - Dperp) * bR * bZ
            DZZ[i, j] = Dperp + (Dpara - Dperp) * bZ^2
        end
        return (DRR, DRZ, DZZ)
    end

    "∇·(D∇) on the wall pattern; with `faces` and `v_absorb`, the Robin debit on the diagonal."
    function diffusion_operator(G, D; faces = nothing, v_absorb = nothing)
        A = build_wall_pattern(G)
        return build_wall_diffusion_matrix!(A, G, D...; cross_terms = :drop, faces, v_absorb)
    end

    "Above the floor on every in-wall node, zero outside; the first `floor_nodes` in-wall nodes below it."
    function plasma_density(G; floor_nodes = 0)
        n = zeros(G.NR * G.NZ)
        inw = G.nodes.in_wall_nids
        R, Z = vec(G.R2D), vec(G.Z2D)
        n[inw] .= 1.0e14 .* (1 .+ 0.4 .* sin.(3 .* R[inw] .+ 2 .* Z[inw]))
        n[inw[1:floor_nodes]] .= 0.5
        return n
    end

    function smooth_field(G; a = 1.0e5, k = (3, 2))
        return a .* (1 .+ 0.5 .* sin.(k[1] .* vec(G.R2D)) .* cos.(k[2] .* vec(G.Z2D)))
    end

    "Cell volumes up to 2π: the weight of every conservation statement."
    cell_volume(G) = vec(G.Jacob) .* (G.dR * G.dZ)

    "The Robin debit per node, read off the diagonals."
    wall_debit(A, A_robin) = [A.matrix[k, k] - A_robin.matrix[k, k] for k in 1:size(A.matrix, 1)]

    """
    Three operators with cross terms: a tilted tensor on the box, the X-point tensor on the box,
    the tilted tensor on the staircase. Each with its reflective and its Robin form.
    """
    function mixing_cases()
        box, stair = walled_RP(:box), walled_RP(:lshape)
        specs = (
            ("tilted tensor, box", box, tilted_tensor(box.G)),
            ("X-point tensor, box", box, xpoint_tensor(box.G)),
            ("tilted tensor, staircase", stair, tilted_tensor(stair.G; θ = 0.9)),
        )
        return map(specs) do (name, RP, D)
            G = RP.G
            faces = RP.transport.wall_faces
            v_absorb = fill(40.0, length(faces))
            (
                name = name, G = G,
                A = diffusion_operator(G, D),
                A_robin = diffusion_operator(G, D; faces, v_absorb),
                V = cell_volume(G), n = plasma_density(G),
            )
        end
    end
end

@testitem "per-particle operator: annihilates constants" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!
    for c in mixing_cases()
        M = per_particle_operator!(similar(c.A), c.A, c.n; n_floor = 1.0)
        r = M * ones(length(c.n))
        @test maximum(abs, r) <= 1.0e-12 * maximum(abs, M.matrix.nzval)
    end
end

@testitem "per-particle operator: n (M f) + f (A n) = A (n f) on every row above the floor" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!
    for c in mixing_cases()
        n = plasma_density(c.G; floor_nodes = 6)
        M = per_particle_operator!(similar(c.A), c.A, n; n_floor = 1.0)
        f = smooth_field(c.G)
        lhs = n .* (M * f) .+ f .* (c.A * n)
        rhs = c.A * (n .* f)
        above = findall(>(1.0), n)
        @test length(above) == length(c.G.nodes.in_wall_nids) - 6
        @test lhs[above] ≈ rhs[above] rtol = 1.0e-12
    end
end

@testitem "per-particle operator: semi-discrete conservation on a reflective and on a Robin wall" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!
    for c in mixing_cases()
        (; G, A, A_robin, V, n) = c
        inw = G.nodes.in_wall_nids
        # the V-weighted column sums of A vanish: that, not zero row sums, is what conserves Σ V n
        colsum = vec(V' * A.matrix)
        @test maximum(abs, colsum[inw]) <= 1.0e-12 * maximum(V) * maximum(abs, A.matrix.nzval)
        f = smooth_field(G)
        M = per_particle_operator!(similar(A), A, n; n_floor = 1.0)
        carried = sum(V .* n .* (M * f))
        scale = sum(V .* n .* abs.(M * f))
        @test abs(carried + sum(V .* f .* (A * n))) <= 1.0e-12 * scale
        # a Robin wall drains n f at the rate it drains n: −Σ V r n f, with r the debit per node
        r = wall_debit(A, A_robin)
        drained = -sum(V .* r .* n .* f)
        @test drained != 0
        M_r = per_particle_operator!(similar(A_robin), A_robin, n; n_floor = 1.0)
        @test sum(V .* n .* (M_r * f)) + sum(V .* f .* (A_robin * n)) ≈ drained rtol = 1.0e-10
    end
end

@testitem "per-particle operator: the Robin debit cancels; uniform n gives M = A" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!
    for c in mixing_cases()
        (; G, A, A_robin, n) = c
        # M reads only the off-diagonals of A: a per-particle variable does not see the wall drain
        M = per_particle_operator!(similar(A), A, n; n_floor = 1.0)
        M_r = per_particle_operator!(similar(A_robin), A_robin, n; n_floor = 1.0)
        @test M_r.matrix.nzval ≈ M.matrix.nzval rtol = 1.0e-12
        # in a uniform plasma the carried variable diffuses exactly as the density does
        n_u = zeros(length(n))
        n_u[G.nodes.in_wall_nids] .= 3.0e13
        M_u = per_particle_operator!(similar(A), A, n_u; n_floor = 1.0)
        @test M_u.matrix.nzval ≈ A.matrix.nzval rtol = 1.0e-12
        M_ru = per_particle_operator!(similar(A_robin), A_robin, n_u; n_floor = 1.0)
        @test M_ru.matrix.nzval ≈ A.matrix.nzval rtol = 1.0e-12
    end
end

@testitem "per-particle operator: rows below the floor are empty, and the defect they leave is accounted" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!, floor_defect
    for c in mixing_cases()
        (; G, A, V) = c
        n = plasma_density(G; floor_nodes = 6)
        M = per_particle_operator!(similar(A), A, n; n_floor = 1.0)
        floor = findall(<=(1.0), n)
        @test all(iszero, M.matrix[floor, :])
        # a frozen f on a floor node whose neighbours hold plasma: n f there does not follow A (n f)
        f = smooth_field(G)
        balance = sum(V .* n .* (M * f)) + sum(V .* f .* (A * n))
        defect = floor_defect(A, n, f, V; n_floor = 1.0)
        @test defect != 0
        @test balance ≈ -defect rtol = 1.0e-10
        @test floor_defect(A, c.n, f, V; n_floor = 1.0) == 0
    end
end

@testitem "per-particle operator: keeps the pattern and its stored zeros, allocates nothing, refuses aliasing and non-finite n" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!
    (; G, A, n) = mixing_cases()[1]
    M = similar(A)
    per_particle_operator!(M, A, n; n_floor = 1.0)
    @test M.matrix.colptr == A.matrix.colptr
    @test M.matrix.rowval == A.matrix.rowval
    stored_zeros = findall(iszero, A.matrix.nzval)
    @test !isempty(stored_zeros)
    @test all(iszero, M.matrix.nzval[stored_zeros])
    # the scratch A n comes from the task's pool: nothing after the first call
    alloc(M, A, n) = @allocated per_particle_operator!(M, A, n; n_floor = 1.0)
    alloc(M, A, n)
    @test alloc(M, A, n) == 0
    # the (NR, NZ) grid itself, read in node order, gives the same operator, also without allocating
    Mg = per_particle_operator!(similar(A), A, reshape(n, G.NR, G.NZ); n_floor = 1.0)
    @test Mg.matrix.nzval == M.matrix.nzval
    ng = reshape(copy(n), G.NR, G.NZ)
    alloc_grid(M, A, ng) = @allocated per_particle_operator!(M, A, ng; n_floor = 1.0)
    alloc_grid(M, A, ng)
    @test alloc_grid(M, A, ng) == 0
    # rewritten, not accumulated
    M2 = per_particle_operator!(similar(A), A, 2 .* n; n_floor = 1.0)
    per_particle_operator!(M, A, 2 .* n; n_floor = 1.0)
    @test M.matrix.nzval == M2.matrix.nzval
    @test_throws ArgumentError per_particle_operator!(A, A, n; n_floor = 1.0)
    for bad in (NaN, Inf)
        n_bad = copy(n)
        n_bad[G.nodes.in_wall_nids[3]] = bad
        @test_throws ArgumentError per_particle_operator!(M, A, n_bad; n_floor = 1.0)
    end
end

@testitem "dissipation rate: the kinetic energy the exchange removes, node by node and in total" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!, dissipation_rate!
    negative_seen = Ref(false)
    for c in mixing_cases()
        (; G, A, V, n) = c
        M = per_particle_operator!(similar(A), A, n; n_floor = 1.0)
        u = smooth_field(G; a = 2.0e5, k = (2, 3))
        Γ = dissipation_rate!(similar(u), M, u)
        # Γ = ½ M u² − u ⊙ M u, evaluated as an edge sum so large u does not cancel
        @test Γ ≈ 0.5 .* (M * (u .^ 2)) .- u .* (M * u) rtol = 1.0e-10
        # in total: what Σ V ½ n u² loses under dn/dt = A n, du/dt = M u is what the heating gains
        dK = sum(V .* (0.5 .* u .^ 2 .* (A * n) .+ n .* u .* (M * u)))
        @test sum(V .* n .* Γ) ≈ -dK rtol = 1.0e-10
        # node by node: with P = Γ (the identity is linear in the mass, so m = 1 and T sized to
        # compete with ½ u²), the energy per particle 3/2 T + ½ u² obeys dε̄/dt = M ε̄
        T = smooth_field(G; a = 1.0e10, k = (1, 4))
        lhs = 1.5 .* (M * T) .+ Γ .+ u .* (M * u)
        rhs = M * (1.5 .* T .+ 0.5 .* u .^ 2)
        @test lhs ≈ rhs rtol = 1.0e-12
        # uniform u: nothing to dissipate, exactly
        @test all(iszero, dissipation_rate!(similar(u), M, fill(3.0e5, length(u))))
        @test all(isfinite, Γ)
        negative_seen[] |= any(<(0), Γ)
        # each row reads its neighbours' u, so the output must not overlap it
        @test_throws ArgumentError dissipation_rate!(u, M, u)
    end
    # with cross terms the discrete rate goes negative on some nodes, and the edge-sum identity
    # above holds there too: it is not clipped
    @test negative_seen[]
    # no cross term: the off-diagonals of M are ≥ 0, so Γ ≥ 0 on every node
    RP = walled_RP(:box)
    G = RP.G
    D = (fill(5.0, G.NR, G.NZ), zeros(G.NR, G.NZ), fill(50.0, G.NR, G.NZ))
    A = diffusion_operator(G, D)
    n = plasma_density(G)
    M = per_particle_operator!(similar(A), A, n; n_floor = 1.0)
    u = smooth_field(G; a = 2.0e5, k = (2, 3))
    Γ = dissipation_rate!(similar(u), M, u)
    @test all(>=(0), Γ)
    @test sum(cell_volume(G) .* n .* Γ) > 0
end

@testitem "advection operator: the per-particle construction on the face-flux divergence, bit for bit with its former assembly" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!, advection_operator!, build_face_flux_divergence!, build_wall_pattern,
        slot_position, SLOT_C
    using RAPID2D.SparseArrays: nonzeros, rowvals, nzrange
    using RAPID2D.LinearAlgebra: mul!
    # `advection_operator!` as it was assembled before it delegated: A n by `mul!`, then the
    # reciprocal of the row's density on every slot
    function former_assembly(C, n; n_floor)
        A = C.matrix
        nzU = zeros(length(nonzeros(A)))
        An = similar(n)
        mul!(An, A, n)
        nzA, rows = nonzeros(A), rowvals(A)
        for j in 1:size(A, 2), s in nzrange(A, j)
            i = rows[s]
            r = n[i] > n_floor ? 1 / n[i] : 0.0
            nzU[s] = r * (nzA[s] * n[j])
        end
        for i in eachindex(n)
            r = n[i] > n_floor ? 1 / n[i] : 0.0
            s = slot_position(C.k2csc, i, SLOT_C)
            nzU[s] = r * ((nzA[s] * n[i]) - An[i])
        end
        return nzU
    end
    for RP in (walled_RP(:box), walled_RP(:lshape))
        G = RP.G
        uR = @. 1.0e5 * sin(5 * G.Z2D) * (1 + 0.2 * G.R2D)
        uZ = @. -3.0e4 * cos(4 * G.R2D)
        C = build_wall_pattern(G)
        build_face_flux_divergence!(C, G, uR, uZ)
        n = plasma_density(G; floor_nodes = 4)
        U = advection_operator!(similar(C), C, n; n_floor = 1.0)
        @test U.matrix.nzval == former_assembly(C, n; n_floor = 1.0)
    end
end
