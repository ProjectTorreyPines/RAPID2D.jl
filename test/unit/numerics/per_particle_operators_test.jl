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
    using RAPID2D: build_wall_pattern, build_wall_diffusion_matrix!, limit_antidiffusive_pairs!,
        per_particle_operator!, add_conduction_correction!

    "A walled box, or a staircase wall with a re-entrant corner, on a 41 × 47 grid by default."
    function walled_RP(wall::Symbol; NR = 41, NZ = 47)
        wall_R, wall_Z = if wall === :box
            [1.2, 1.8, 1.8, 1.2], [-0.3, -0.3, 0.3, 0.3]
        else
            [1.2, 1.8, 1.8, 1.5, 1.5, 1.2], [-0.3, -0.3, 0.0, 0.0, 0.3, 0.3]
        end
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = NR, NZ = NZ,
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

    """
    A_part and A_cond along the same tilted or X-point field, A_cond with a larger parallel
    coefficient, both limited at a density with floor nodes (so the pair limiter acts).
    """
    function conduction_cases()
        box, stair = walled_RP(:box), walled_RP(:lshape)
        specs = (
            ("tilted tensor, box", box, G -> tilted_tensor(G; Dpara = 100.0), G -> tilted_tensor(G; Dpara = 400.0)),
            ("X-point tensor, box", box, G -> xpoint_tensor(G; Dpara = 100.0), G -> xpoint_tensor(G; Dpara = 250.0)),
            ("tilted tensor, staircase", stair, G -> tilted_tensor(G; θ = 0.9, Dpara = 100.0), G -> tilted_tensor(G; θ = 0.9, Dpara = 300.0)),
        )
        return map(specs) do (name, RP, part, cond)
            G = RP.G
            n = plasma_density(G; floor_nodes = 6)
            A_part, A_cond = diffusion_operator(G, part(G)), diffusion_operator(G, cond(G))
            limited = limit_antidiffusive_pairs!(A_part, G, n; max_ratio = 4.0)
            limit_antidiffusive_pairs!(A_cond, G, n; max_ratio = 4.0)
            (; name, G, A_part, A_cond, n, V = cell_volume(G), limited)
        end
    end

    "The per-particle operator of A_part, and the same with the conduction correction."
    function with_correction(c; n = c.n)
        M_part = per_particle_operator!(similar(c.A_part), c.A_part, n; n_floor = 1.0)
        M = per_particle_operator!(similar(c.A_part), c.A_part, n; n_floor = 1.0)
        add_conduction_correction!(M, c.A_part, c.A_cond, n, c.G; n_floor = 1.0)
        return M_part, M
    end

    """
    Two poloidal fields with a profile `f` constant along `b`, and `∇·(n̂ n̂·∇f)` across it (n̂ ⊥ b
    in the poloidal plane, cylindrical, Jacobian R): a diagonal field and circles about (1.5, 0),
    the centre excluded.
    """
    function parallel_fields()
        s = sin(π / 4)
        diagonal = (
            name = "diagonal",
            b = (R, Z) -> (s, s),
            f = (R, Z) -> (Z - (R - 1.5))^2 / 2,
            perp_div = (R, Z) -> 2 - (Z - (R - 1.5)) / R,
            excluded = (R, Z) -> false,
        )
        circular = (
            name = "circular",
            b = (R, Z) -> (r = hypot(R - 1.5, Z); (-Z / r, (R - 1.5) / r)),
            f = (R, Z) -> (R - 1.5)^2 + Z^2,
            perp_div = (R, Z) -> 6 - 3 / R,
            excluded = (R, Z) -> hypot(R - 1.5, Z) < 0.1,
        )
        return (diagonal, circular)
    end

    "D∥ b bᵀ along the field, no cross-field part."
    function parallel_tensor(G, field; Dpara)
        DRR, DRZ, DZZ = zeros(G.NR, G.NZ), zeros(G.NR, G.NZ), zeros(G.NR, G.NZ)
        for j in 1:G.NZ, i in 1:G.NR
            field.excluded(G.R2D[i, j], G.Z2D[i, j]) && continue
            bR, bZ = field.b(G.R2D[i, j], G.Z2D[i, j])
            DRR[i, j], DRZ[i, j], DZZ[i, j] = Dpara * bR^2, Dpara * bR * bZ, Dpara * bZ^2
        end
        return (DRR, DRZ, DZZ)
    end

    """
    The largest cross-field diffusivity `Op` adds, `|(Op f)_k| / ∇·(n̂ n̂·∇f)_k`, over the nodes two
    cells inside the wall and away from the excluded centre: zero for an operator that carries
    nothing across the field.
    """
    function artificial_perp_diffusivity(Op, G, field)
        R, Z = vec(G.R2D), vec(G.Z2D)
        Of = Op * field.f.(R, Z)
        inw = Set(G.nodes.in_wall_nids)
        inner(k) = all(((k + di + dj * G.NR) in inw) for di in -2:2, dj in -2:2) &&
            !any(field.excluded(R[k + di + dj * G.NR], Z[k + di + dj * G.NR]) for di in -1:1, dj in -1:1)
        return maximum(abs(Of[k]) / field.perp_div(R[k], Z[k]) for k in G.nodes.in_wall_nids if inner(k))
    end

    "Two operators on one grid, the second's density a front absent over half the box, and a still third in the total."
    function carrier_mixture(c)
        G = c.G
        R, Z = vec(G.R2D), vec(G.Z2D)
        inw = G.nodes.in_wall_nids
        w1 = plasma_density(G)
        w2 = zeros(length(w1))
        w2[inw] .= 3.0e13 .* max.(0.0, tanh.(8 .* (Z[inw] .- 0.05)))   # zero below Z ≈ 0.05
        w_still = zeros(length(w1))
        w_still[inw] .= 2.0e12 .* (1 .+ 0.3 .* cos.(4 .* R[inw]))
        A1 = c.A
        A2 = diffusion_operator(G, tilted_tensor(G; θ = -0.4, Dpara = 30.0, Dperp = 2.0))
        return (; A1, A2, w1, w2, w_still, w_I = w1 .+ w2 .+ w_still)
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

# The conduction correction: u∥ and Te ride the particles the density diffusion A_part moves, but
# conduct (and diffuse momentum) with their own A_cond. C adds the difference as symmetric pair
# transfers S_kl = ½(R_k ΔA_kl + R_l ΔA_lk), ΔA = A_cond − A_part, weighted by the pair's mean
# density: (C f)_k = Σ_l S_kl (n_k + n_l)/2 (f_l − f_k) / (R_k n_k). The production matrices are
# not volume-symmetric (the cross terms), so this, not a split of M by face densities, is what
# keeps Σ V n f. internal/docs/src/notes/design/ambipolar-gate-criterion-v3.md §3.5.

@testitem "conduction correction: keeps Σ V n f and constants; no difference leaves M bit for bit" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!, add_conduction_correction!
    for c in conduction_cases()
        @test c.limited > 0
        M_part, M = with_correction(c)
        C = M.matrix - M_part.matrix
        @test maximum(abs, C) > 0
        # constants stay in the kernel
        @test maximum(abs, M * ones(length(c.n))) <= 1.0e-12 * maximum(abs, M.matrix.nzval)
        # what the correction moves between particles adds up to nothing
        f = smooth_field(c.G)
        Cf = C * f
        @test abs(sum(c.V .* c.n .* Cf)) <= 1.0e-12 * sum(c.V .* c.n .* abs.(Cf))
        # A_cond = A_part: nothing added, bit for bit
        M_same = per_particle_operator!(similar(c.A_part), c.A_part, c.n; n_floor = 1.0)
        add_conduction_correction!(M_same, c.A_part, deepcopy(c.A_part), c.n, c.G; n_floor = 1.0)
        @test M_same.matrix.nzval == M_part.matrix.nzval
    end
end

@testitem "conduction correction: carries the cross terms, adds dissipation in total, and the heating identity holds for the whole operator" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!, add_conduction_correction!, dissipation_rate!
    using RAPID2D.SparseArrays: findnz
    for c in conduction_cases()
        M_part, M = with_correction(c)
        # the negative transfers come with it, as in the operators it is built from: they are what
        # cancels the diffusion across b (the artificial cross-field diffusivity test)
        C = M.matrix - M_part.matrix
        off = [C[k, l] for (k, l, _) in zip(findnz(C)...) if k != l]
        @test any(<(0), off) && any(>(0), off)
        u = smooth_field(c.G; a = 2.0e5, k = (2, 3))
        Γ, Γ_part = dissipation_rate!(similar(u), M, u), dissipation_rate!(similar(u), M_part, u)
        # the heating it adds is what it takes from Σ V ½ n u², and positive in total
        heat = sum(c.V .* c.n .* (Γ .- Γ_part))
        @test heat > 0
        @test heat ≈ -sum(c.V .* c.n .* u .* (C * u)) rtol = 1.0e-10
        # above the floor everywhere, what Σ V ½ n u² loses under dn/dt = A_part n, du/dt = M u is
        # what the heating of the whole operator gains (floor rows break it for M_part alone)
        n = plasma_density(c.G)
        _, M = with_correction(c; n)
        dK = sum(c.V .* (0.5 .* u .^ 2 .* (c.A_part * n) .+ n .* u .* (M * u)))
        @test sum(c.V .* n .* dissipation_rate!(similar(u), M, u)) ≈ -dK rtol = 1.0e-10
    end
end

@testitem "conduction correction: a pair with a node on the floor is skipped, both ways" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!, add_conduction_correction!
    for c in conduction_cases()
        M_part, M = with_correction(c)
        C = M.matrix - M_part.matrix
        floor = findall(<=(1.0), c.n)
        @test count(in(floor), c.G.nodes.in_wall_nids) == 6
        @test all(iszero, C[floor, :]) && all(iszero, C[:, floor])
    end
end

@testitem "conduction correction: in a uniform plasma with no cross terms, the variable conducts with A_cond" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!, add_conduction_correction!
    RP = walled_RP(:box)
    G = RP.G
    A_part = diffusion_operator(G, (fill(5.0, G.NR, G.NZ), zeros(G.NR, G.NZ), fill(50.0, G.NR, G.NZ)))
    A_cond = diffusion_operator(G, (fill(5.0, G.NR, G.NZ), zeros(G.NR, G.NZ), fill(80.0, G.NR, G.NZ)))
    n = zeros(G.NR * G.NZ)
    n[G.nodes.in_wall_nids] .= 3.0e13
    M = per_particle_operator!(similar(A_part), A_part, n; n_floor = 1.0)
    add_conduction_correction!(M, A_part, A_cond, n, G; n_floor = 1.0)
    @test M.matrix.nzval ≈ A_cond.matrix.nzval rtol = 1.0e-12
end

# A profile constant along b must not diffuse: what an operator moves across the field is its
# truncation error alone, second order. The conduction correction carries the cross terms of
# A_cond − A_part as its operators do; dropping its negative transfers left ≈ ½ΔD across a
# diagonal field at every resolution. The loop over discretizations is where a new
# `AnisotropicDiscretization` joins; at this resolved density `LimitedNinePoint` limits nothing and
# is `NinePoint`.
@testitem "artificial cross-field diffusivity: truncation only, in the diffusion, per-particle and conduction operators" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!, add_conduction_correction!, assemble_anisotropic_diffusion!
    # D_art / D at 41 × 47, about ten cells across the box: the diagonal field is exact for the
    # diffusion operator and second order through the density's variation; the circles, second order
    bound = (diagonal = 1.0e-3, circular = 2.0e-2)
    function assembled(G, D, discretization, n)
        A = build_wall_pattern(G)
        @test assemble_anisotropic_diffusion!(A, G, D..., discretization; n) == 0   # resolved: nothing limited
        return A
    end
    grids = Dict(r => walled_RP(:box; NR = r[1], NZ = r[2]).G for r in ((41, 47), (81, 93)))
    for discretization in (LimitedNinePoint(), NinePoint()), field in parallel_fields()
        D_art = map(((41, 47), (81, 93))) do (NR, NZ)
            G = grids[(NR, NZ)]
            n = plasma_density(G)
            A_part = assembled(G, parallel_tensor(G, field; Dpara = 1.0), discretization, n)
            A_cond = assembled(G, parallel_tensor(G, field; Dpara = 4.0), discretization, n)
            M_part = per_particle_operator!(similar(A_part), A_part, n; n_floor = 1.0)
            M = per_particle_operator!(similar(A_part), A_part, n; n_floor = 1.0)
            add_conduction_correction!(M, A_part, A_cond, n, G; n_floor = 1.0)
            # each per its own coefficient along b: 4 for A_cond and for M + C, 1 for M
            (
                diffusion = artificial_perp_diffusivity(A_cond, G, field) / 4,
                per_particle = artificial_perp_diffusivity(M_part, G, field),
                conduction = artificial_perp_diffusivity(M, G, field) / 4,
            )
        end
        for op in (:diffusion, :per_particle, :conduction)
            coarse, fine = D_art[1][op], D_art[2][op]
            @test coarse <= bound[Symbol(field.name)]
            @test fine <= max(coarse / 3, 1.0e-10)
        end
    end
end

# ── a mixture of carriers ───────────────────────────────────────────────────────────────
# Several groups of particles, each moved by its own operator A_g with density w_g, share one
# carried variable f, counted over all of them, w_I ≥ Σ w_g (a species that does not move is in
# w_I and in no group). The numerator is the groups' sum and the denominator the total:
#     M_kl = Σ_g A_g,kl w_g,l / w_I,k  (l ≠ k),   zero row sums,
# so a node where one group is absent still takes what that group brings in. The floor is on the
# total. internal/docs/src/notes/plans/PLAN_ion-momentum-energy-transport.md §2.3.

@testitem "mixture per-particle operator: one group whose density is the total is the per-particle operator, bit for bit" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!
    for c in mixing_cases()
        n = plasma_density(c.G; floor_nodes = 6)
        M = per_particle_operator!(similar(c.A), c.A, n; n_floor = 1.0)
        Mx = per_particle_operator!(similar(c.A), (c.A,), (n,), n; n_floor = 1.0)
        @test Mx.matrix.nzval == M.matrix.nzval
        Mr = per_particle_operator!(similar(c.A_robin), (c.A_robin,), (n,), n; n_floor = 1.0)
        @test Mr.matrix.nzval == per_particle_operator!(similar(c.A_robin), c.A_robin, n; n_floor = 1.0).matrix.nzval
    end
end

@testitem "mixture per-particle operator: Σ_g A_g (w_g f) = w_I (M f) + f Σ_g A_g w_g on every row above the total's floor, where a group is absent too" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!
    for c in mixing_cases()
        (; A1, A2, w1, w2, w_still, w_I) = carrier_mixture(c)
        M = per_particle_operator!(similar(A1), (A1, A2), (w1, w2), w_I; n_floor = 1.0)
        f = smooth_field(c.G)
        lhs = w_I .* (M * f) .+ f .* (A1 * w1 .+ A2 * w2)
        rhs = A1 * (w1 .* f) .+ A2 * (w2 .* f)
        above = findall(>(1.0), w_I)
        @test lhs[above] ≈ rhs[above] rtol = 1.0e-12
        # the front: rows where the second group is absent but brings particles in from a neighbour
        front = [k for k in above if w2[k] == 0 && abs((A2 * w2)[k]) > 0]
        @test !isempty(front)
        @test lhs[front] ≈ rhs[front] rtol = 1.0e-12
        # constants are annihilated, and the rows of nodes the total leaves empty are empty
        @test maximum(abs, M * ones(length(f))) <= 1.0e-12 * maximum(abs, M.matrix.nzval)
        w_I_floor = copy(w_I)
        floor = c.G.nodes.in_wall_nids[1:4]
        w_I_floor[floor] .= 0.5
        Mf = per_particle_operator!(similar(A1), (A1, A2), (w1, w2), w_I_floor; n_floor = 1.0)
        @test all(iszero, Mf.matrix[floor, :])
    end
end

@testitem "mixture per-particle operator: Σ V w_I f is kept by conservative operators, the still part included" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!
    for c in mixing_cases()
        (; A1, A2, w1, w2, w_I) = carrier_mixture(c)
        V = c.V
        M = per_particle_operator!(similar(A1), (A1, A2), (w1, w2), w_I; n_floor = 1.0)
        f = smooth_field(c.G)
        carried = sum(V .* w_I .* (M * f))
        moved = sum(V .* f .* (A1 * w1 .+ A2 * w2))
        @test abs(carried + moved) <= 1.0e-12 * sum(V .* w_I .* abs.(M * f))
    end
end

@testitem "conduction correction over a total: n_total = n is bit for bit; with a group's density it keeps Σ V w_I f and reaches where that group is absent" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!, add_conduction_correction!
    for c in conduction_cases()
        M_part, M = with_correction(c)
        Mn = per_particle_operator!(similar(c.A_part), c.A_part, c.n; n_floor = 1.0)
        add_conduction_correction!(Mn, c.A_part, c.A_cond, c.n, c.G; n_floor = 1.0, n_total = c.n)
        @test Mn.matrix.nzval == M.matrix.nzval
        # one group of a mixture: its pair transfers over the total's density
        G, V = c.G, c.V
        inw = G.nodes.in_wall_nids
        Z = vec(G.Z2D)
        w_g = zeros(length(c.n))
        w_g[inw] .= 3.0e13 .* max.(0.0, tanh.(8 .* (Z[inw] .- 0.05)))
        w_I = c.n .+ w_g
        Mg = per_particle_operator!(similar(c.A_part), (c.A_part,), (w_g,), w_I; n_floor = 1.0)
        before = copy(Mg.matrix.nzval)
        add_conduction_correction!(Mg, c.A_part, c.A_cond, w_g, G; n_floor = 1.0, n_total = w_I)
        C = similar(Mg)
        C.matrix.nzval .= Mg.matrix.nzval .- before
        f = smooth_field(G)
        @test abs(sum(V .* w_I .* (C * f))) <= 1.0e-12 * sum(V .* w_I .* abs.(C * f))
        @test maximum(abs, C * ones(length(f))) <= 1.0e-12 * maximum(abs, C.matrix.nzval)
        # a row where the group is absent takes the correction from a neighbour where it is not
        Cf = C * f
        @test any(k -> w_g[k] == 0 && w_I[k] > 1 && Cf[k] != 0, inw)
    end
end

@testitem "mixture per-particle operator: allocates nothing after the first call, refuses mismatched groups" setup = [PerParticleFixtures] begin
    using RAPID2D: per_particle_operator!
    c = mixing_cases()[1]
    (; A1, A2, w1, w2, w_I) = carrier_mixture(c)
    M = similar(A1)
    ops, ws = (A1, A2), (w1, w2)
    alloc(M, ops, ws, w_I) = @allocated per_particle_operator!(M, ops, ws, w_I; n_floor = 1.0)
    alloc(M, ops, ws, w_I)
    @test alloc(M, ops, ws, w_I) == 0
    @test_throws ArgumentError per_particle_operator!(M, (A1, A2), (w1,), w_I; n_floor = 1.0)
    @test_throws ArgumentError per_particle_operator!(A1, (A1, A2), ws, w_I; n_floor = 1.0)
end
