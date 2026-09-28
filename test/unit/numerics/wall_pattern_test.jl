# One sparsity pattern for every in-wall operator, allocated once from the geometry: the
# 9-point stencil restricted to in-wall neighbours, plus the diagonal on every node so that a
# θ-matrix `I − θΔt·A` fits the same pattern. Both upwind sides of every face are in it, so the
# pattern is the same for any velocity field. A `DiscretizedOperator` on it maps (row, stencil
# slot) to its `nonzeros` position through `k2csc`, nine slots per row; operators only ever
# write values, so the structure — and every cached symbolic LU — never changes.

# Testitems import what they call in their own bodies: under ReTestItems (the parallel suite)
# an import inside the snippet reaches the snippet module only.
@testsnippet PatternBox begin
    function pattern_box()
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        return RP
    end
end

@testitem "wall pattern: every (row, slot) names its own entry, and nothing else is stored" setup = [PatternBox] begin
    using RAPID2D: build_wall_pattern, is_in_wall, STENCIL_OFFSETS
    using RAPID2D.SparseArrays
    G = pattern_box().G
    P = build_wall_pattern(G)
    Ng = G.NR * G.NZ
    @test size(P.matrix) == (Ng, Ng)
    @test length(P.k2csc) == 9 * Ng
    Q = similar(P)
    nz = nonzeros(Q.matrix)
    for j in 1:G.NZ, i in 1:G.NR
        r = G.nodes.nid[i, j]
        for (s, (di, dj)) in enumerate(STENCIL_OFFSETS)
            present = s == 1 || (is_in_wall(G, i, j) && is_in_wall(G, i + di, j + dj))
            c = P.k2csc[9 * (r - 1) + s]
            @test (c > 0) == present
            present && (nz[c] = s)
        end
    end
    for j in 1:G.NZ, i in 1:G.NR
        r = G.nodes.nid[i, j]
        row = Q.matrix[r, :]
        if is_in_wall(G, i, j)
            for (s, (di, dj)) in enumerate(STENCIL_OFFSETS)
                is_in_wall(G, i + di, j + dj) && @test row[G.nodes.nid[i + di, j + dj]] == s
            end
            @test nnz(row) == count(((di, dj),) -> is_in_wall(G, i + di, j + dj), STENCIL_OFFSETS)
        else
            @test nnz(row) == 1 && row[r] == 1        # an identity row outside the wall
        end
    end
end

@testitem "wall pattern: the fresh builders' patterns are subsets of it" setup = [PatternBox] begin
    using RAPID2D: build_face_flux_divergence, build_wall_diffusion_matrix, build_wall_pattern
    using RAPID2D.SparseArrays
    G = pattern_box().G
    P = Set(zip(findnz(build_wall_pattern(G).matrix)[1:2]...))
    Rc = 0.5 * (G.R1D[1] + G.R1D[end])
    uR = @. 1.0e5 * sign(Rc - G.R2D)
    uZ = @. 3.0e4 * sin(G.Z2D)
    bR, bZ = cos(0.6), sin(0.6)
    D_RR = fill(1.0 + 999.0 * bR^2, G.NR, G.NZ)
    D_RZ = fill(999.0 * bR * bZ, G.NR, G.NZ)
    D_ZZ = fill(1.0 + 999.0 * bZ^2, G.NR, G.NZ)
    for A in (
            build_face_flux_divergence(G, uR, uZ),
            build_face_flux_divergence(G, uR, uZ; upwind = false),
            build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ),
            build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; cross_terms = :reflect),
        )
        @test issubset(Set(zip(findnz(A)[1:2]...)), P)
    end
end

@testitem "wall pattern: a valid CSC, and similar gives independent values on the same structure" setup = [PatternBox] begin
    using RAPID2D: build_wall_pattern
    using RAPID2D.SparseArrays
    G = pattern_box().G
    P = build_wall_pattern(G)
    M = P.matrix
    @test all(issorted(rowvals(M)[nzrange(M, c)]) for c in 1:size(M, 2))   # rows sorted within each column
    @test M.colptr[end] == nnz(M) + 1
    stored = Set(zip(findnz(M)[1:2]...))
    @test stored == Set((c, r) for (r, c) in stored)                      # structurally symmetric
    A = similar(P)
    B = similar(P)
    nonzeros(A.matrix) .= 1.0
    @test all(iszero, nonzeros(B.matrix)) && all(iszero, nonzeros(P.matrix))
    @test A.matrix.colptr !== P.matrix.colptr && A.matrix.rowval !== P.matrix.rowval
    @test A.matrix.colptr == P.matrix.colptr && A.matrix.rowval == P.matrix.rowval
    @test A.k2csc == P.k2csc && A.k2csc !== P.k2csc
end

@testitem "wall pattern: identity, scaled add and diagonal add are vector arithmetic on it" setup = [PatternBox] begin
    using RAPID2D: set_identity!, add_scaled!, add_diagonal!, build_wall_diffusion_matrix, DiscretizedOperator,
        build_wall_pattern
    using RAPID2D.LinearAlgebra, RAPID2D.SparseArrays
    G = pattern_box().G
    Ng = G.NR * G.NZ
    A = build_wall_pattern(G)
    B = build_wall_pattern(G)
    nonzeros(B.matrix) .= [sin(1.3 * k) + 0.2 * cos(0.31 * k) for k in 1:nnz(B.matrix)]   # every entry distinct, none zero
    v = [0.5 + 0.4 * cos(0.7 * k) for k in 1:Ng]
    set_identity!(A)
    @test A.matrix == I(Ng)
    add_scaled!(A, -0.25, B)
    add_diagonal!(A, v; scale = 2.0)
    @test A.matrix ≈ I(Ng) - 0.25 * B.matrix + 2.0 * Diagonal(v) rtol = 1.0e-15
    @test nnz(A.matrix) == nnz(B.matrix)                          # nothing dropped, structure untouched
    D = build_wall_diffusion_matrix(G, ones(G.NR, G.NZ), zeros(G.NR, G.NZ), ones(G.NR, G.NZ))
    off = DiscretizedOperator((G.NR, G.NZ), findnz(D)...)        # the same operator, but not on the pattern
    @test_throws ArgumentError add_scaled!(A, 1.0, off)
    @test_throws ArgumentError set_identity!(off)
    @test_throws ArgumentError add_diagonal!(off, v)
end

@testitem "wall pattern: is_on_wall_pattern tells a pattern operator from any other" setup = [PatternBox] begin
    # Callers ask the predicate, never the storage: the stride of k2csc is numerics' business.
    using RAPID2D: build_wall_pattern, is_on_wall_pattern, check_wall_pattern, build_wall_diffusion_matrix,
        DiscretizedOperator
    using RAPID2D.SparseArrays
    G = pattern_box().G
    P = build_wall_pattern(G)
    @test is_on_wall_pattern(P) && is_on_wall_pattern(similar(P))
    D = build_wall_diffusion_matrix(G, ones(G.NR, G.NZ), zeros(G.NR, G.NZ), ones(G.NR, G.NZ))
    for off in (DiscretizedOperator{Float64}(G.NR, G.NZ), DiscretizedOperator((G.NR, G.NZ), findnz(D)...))
        @test !is_on_wall_pattern(off)
        @test_throws ArgumentError check_wall_pattern(off)
    end
end

@testitem "wall pattern: a broadcast that changes the structure drops k2csc, so in-place writes refuse" setup = [PatternBox] begin
    # The generic DiscretizedOperator broadcast materializes a new sparse matrix; if its
    # structure differs from the pattern, a copied k2csc would point at wrong positions and
    # the next in-place write would land there silently. The map must be dropped instead.
    using RAPID2D: build_wall_pattern, set_identity!, build_wall_diffusion_matrix!
    using RAPID2D.SparseArrays
    G = pattern_box().G
    A = build_wall_pattern(G)
    B = build_wall_pattern(G)
    build_wall_diffusion_matrix!(B, G, ones(G.NR, G.NZ), zeros(G.NR, G.NZ), ones(G.NR, G.NZ))
    @. A = B - B                                     # an all-zero result: sparse broadcast stores nothing
    @test isempty(A.k2csc)
    isempty(A.k2csc) && @test_throws ArgumentError set_identity!(A)
end
