# The open lines' quadrature of field-line following: the row of `line_weights` of a line that ends
# on the wall both ways integrates along B, ∫ f ds ≈ Σ w f, and the ends are where the line met the wall.

@testsnippet LineQuadrature begin
    using RAPID2D: initialize_grid_geometry, FieldLineGeometry, FieldLineFollowingResult, field_line_analysis!,
        MarchingSquaresTracer, CubicTracer, RK4Tracer
    using RAPID2D.SparseArrays: nnz
    using RAPID2D.LinearAlgebra: norm

    "Straight vertical lines, B_Z = BZ uniform (ψ = BZ R²/2 per radian), Bϕ = R0B0/R, inside a box of cells."
    function straight_lines(; NR = 25, NZ = 41, BZ = 0.1, R0B0 = 3.0, box = (1.1, 1.9, -0.8, 0.8))
        G = initialize_grid_geometry(NR, NZ, (1.0, 2.0), (-1.0, 1.0))
        ψ = @. BZ * G.R2D^2 / 2
        inside = [box[1] < G.R1D[i] + G.dR / 2 < box[2] && box[3] < G.Z1D[j] + G.dZ / 2 < box[4] for i in 1:NR, j in 1:NZ]
        g = FieldLineGeometry(G, ψ; Bϕ = R0B0 ./ G.R2D, inside, nodal_B = true)
        return G, g, inside
    end

    "The in-wall nodes: every cell around them inside."
    in_wall_nodes(G, inside) = [LinearIndices((G.NR, G.NZ))[i, j] for i in 2:(G.NR - 1), j in 2:(G.NZ - 1) if all(inside[(i - 1):i, (j - 1):j])]
end

@testitem "Line quadrature: the row integrates along B, and the ends are on the wall" setup = [LineQuadrature] begin
    G, g, inside = straight_lines()
    BZ, R0B0 = 0.1, 3.0
    for tracer in (MarchingSquaresTracer(), CubicTracer(), RK4Tracer())
        flf = field_line_analysis!(FieldLineFollowingResult{Float64}(G.NR, G.NZ), g, tracer; quadrature = true)
        W = flf.line_weights
        nodes = in_wall_nodes(G, inside)
        @test all(flf.termination_forward[nodes] .=== :wall)
        @test nnz(W) > 0
        # the row's weights sum to the length along B the lengths measured
        rowsum = W * ones(G.NR * G.NZ)
        @test maximum(abs.(rowsum[nodes] .- flf.Lc_tot[nodes]) ./ flf.Lc_tot[nodes]) < 1.0e-6
        # ∫ Z² ds on a vertical line from Z_a to Z_b is (B/B_Z)(Z_b³ − Z_a³)/3: the quadrature of a smooth function
        # (the line is symmetric about Z = 0, so ∫ Z ds would be 0 and prove nothing)
        ratio = @. hypot(BZ, R0B0 / G.R2D) / BZ
        ∫Z² = W * vec(G.Z2D .^ 2)
        Za, Zb = flf.end_Z_backward, flf.end_Z_forward
        exact = @. ratio * (Zb^3 - Za^3) / 3
        @test maximum(abs.(∫Z²[nodes] .- vec(exact)[nodes]) ./ vec(exact)[nodes]) < 5.0e-3
        # the ends: on the box's top and bottom (within a cell), at the node's R
        @test all(abs.(vec(flf.end_Z_forward)[nodes] .- 0.8) .< G.dZ)
        @test all(abs.(vec(flf.end_Z_backward)[nodes] .+ 0.8) .< G.dZ)
        @test all(abs.(vec(flf.end_R_forward)[nodes] .- vec(G.R2D)[nodes]) .< G.dR)
        # without the quadrature nothing is kept, and the lengths are the same
        plain = field_line_analysis!(FieldLineFollowingResult{Float64}(G.NR, G.NZ), g, tracer)
        @test nnz(plain.line_weights) == 0
        @test plain.Lc_tot == flf.Lc_tot
        @test all(isnan, plain.end_Z_forward)
    end
end

@testitem "Line quadrature: closed lines keep no row, open ones do" setup = [LineQuadrature] begin
    G = initialize_grid_geometry(30, 50, (1.0, 2.0), (-1.0, 1.0))
    R0, Z0, κ = 1.5113, 0.0071, 1.6
    ψ = @. (G.R2D - R0)^2 + ((G.Z2D - Z0) / κ)^2
    inside = [1.1 < G.R1D[i] + G.dR / 2 < 1.9 && abs(G.Z1D[j] + G.dZ / 2) < 0.9 for i in 1:G.NR, j in 1:G.NZ]
    g = FieldLineGeometry(G, ψ; Bϕ = 3.0 ./ G.R2D, inside, nodal_B = true)
    flf = field_line_analysis!(FieldLineFollowingResult{Float64}(G.NR, G.NZ), g, MarchingSquaresTracer(); quadrature = true)
    W = flf.line_weights
    rowsum = W * ones(G.NR * G.NZ)
    nodes = in_wall_nodes(G, inside)
    closed = [k for k in nodes if flf.is_closed[k]]
    open = [k for k in nodes if flf.termination_forward[k] === :wall && flf.termination_backward[k] === :wall]
    @test !isempty(closed) && !isempty(open)
    @test all(iszero, rowsum[closed])
    @test all(isnan, vec(flf.end_Z_forward)[closed])
    @test maximum(abs.(rowsum[open] .- flf.Lc_tot[open]) ./ flf.Lc_tot[open]) < 1.0e-6
end
