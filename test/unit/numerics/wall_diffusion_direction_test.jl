# Does the anisotropic operator actually transport along b̂?
#
# Every Phase 2 assertion was structural — row sums, column sums, exact zeros
# outside the wall. Those hold even if the tensor points the wrong way: swapping
# bR and bZ conserves just as well. Direction can only be checked by SOLVING, so
# this file marches a blob in a reflective box and measures the second-moment
# tensor of where it went.
#
# The diagonal and circular cases are the ones that matter. With b̂ on a grid axis
# D_RZ = 0, so an axis-aligned test passes with the cross term deleted from the
# code entirely.

@testitem "Transport direction: a blob elongates along an axis-aligned field" begin
    using RAPID2D: build_wall_diffusion_matrix, density_second_moments
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    # deliberately non-square: dR = 0.025, dZ = 0.05. A swapped dR/dZ inside the
    # operator is invisible on a square grid.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 41, NZ = 41,
        R_min = 1.0, R_max = 2.0, Z_min = -1.0, Z_max = 1.0,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    @test !(G.dR ≈ G.dZ)

    Dperp, ratio = 0.05, 20.0
    Dpara = Dperp * ratio

    function spread(bR_val, bZ_val)
        D_RR = fill(Dperp + (Dpara - Dperp) * bR_val^2, G.NR, G.NZ)
        D_RZ = fill((Dpara - Dperp) * bR_val * bZ_val, G.NR, G.NZ)
        D_ZZ = fill(Dperp + (Dpara - Dperp) * bZ_val^2, G.NR, G.NZ)
        A = build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ)

        Rc = (G.R1D[1] + G.R1D[end]) / 2
        Zc = (G.Z1D[1] + G.Z1D[end]) / 2
        n = [
            exp(-((G.R2D[i, j] - Rc)^2 + (G.Z2D[i, j] - Zc)^2) / 2.0e-4)
                for i in 1:G.NR, j in 1:G.NZ
        ]
        n[G.nodes.on_out_wall_nids] .= 0.0

        dt = 2.0e-4
        M = sparse(I, size(A, 1), size(A, 2)) - dt * A
        v = vec(n)
        for _ in 1:20
            v = M \ v
        end
        return density_second_moments(G, reshape(v, G.NR, G.NZ))
    end

    # b̂ = R̂ — the blob must stretch in R, and the variance ratio must approach the
    # imposed D∥/D⊥ rather than merely being "bigger".
    #
    # It approaches rather than equals it because the blob starts with a finite
    # isotropic width: Var∥/Var⊥ = (σ₀² + 2D∥t)/(σ₀² + 2D⊥t), which tends to D∥/D⊥
    # only as the spread outgrows σ₀. Measured at three ratios on a 61² grid —
    # 5 → 4.47, 20 → 17.48, 100 → 86.9 — all three fit that expression with the
    # SAME σ₀²/(2D⊥t) = 0.153, so the deficit is the initial width and not an
    # error in the operator. Measured axis error: 0.0° at every angle and ratio.
    ΣRR, ΣRZ, ΣZZ = spread(1.0, 0.0)
    @test ΣRR / ΣZZ ≈ ratio rtol = 0.2
    @test abs(ΣRZ) < 1.0e-3 * ΣRR                 # no cross correlation on an axis

    # b̂ = Ẑ — everything inverts. This is what a transposed b̂ or a swapped
    # dR/dZ fails, and the non-square grid is what makes it visible.
    ΣRR, ΣRZ, ΣZZ = spread(0.0, 1.0)
    @test ΣZZ / ΣRR ≈ ratio rtol = 0.2
    @test abs(ΣRZ) < 1.0e-3 * ΣZZ
end

@testitem "Transport direction: an oblique field tilts the blob to match" begin
    using RAPID2D: build_wall_diffusion_matrix, density_second_moments
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    # THE test the cross term exists for. With b̂ on an axis D_RZ vanishes and the
    # previous testitem passes with the cross-term code removed; only an oblique
    # field can tell the difference. Square grid here so an angle means an angle.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 61, NZ = 61,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    @test G.dR ≈ G.dZ

    Dperp, ratio = 0.05, 20.0
    Dpara = Dperp * ratio

    function principal_axis(θ)
        bR, bZ = cos(θ), sin(θ)
        D_RR = fill(Dperp + (Dpara - Dperp) * bR^2, G.NR, G.NZ)
        D_RZ = fill((Dpara - Dperp) * bR * bZ, G.NR, G.NZ)
        D_ZZ = fill(Dperp + (Dpara - Dperp) * bZ^2, G.NR, G.NZ)
        A = build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ)

        Rc = (G.R1D[1] + G.R1D[end]) / 2
        Zc = (G.Z1D[1] + G.Z1D[end]) / 2
        n = [
            exp(-((G.R2D[i, j] - Rc)^2 + (G.Z2D[i, j] - Zc)^2) / 1.0e-4)
                for i in 1:G.NR, j in 1:G.NZ
        ]
        n[G.nodes.on_out_wall_nids] .= 0.0

        dt = 1.0e-4
        M = sparse(I, size(A, 1), size(A, 2)) - dt * A
        v = vec(n)
        for _ in 1:20
            v = M \ v
        end
        ΣRR, ΣRZ, ΣZZ = density_second_moments(G, reshape(v, G.NR, G.NZ))
        vals, vecs = eigen(Symmetric([ΣRR ΣRZ; ΣRZ ΣZZ]))
        major = vecs[:, argmax(vals)]
        return atan(major[2], major[1]), maximum(vals) / minimum(vals)
    end

    # not only 45°: a coincidence there would not survive 30° and 60°
    for θ in (π / 6, π / 4, π / 3)
        axis, spread_ratio = principal_axis(θ)
        # principal axes are defined mod π, so compare directions not vectors
        Δ = abs(rem(axis - θ, π, RoundNearest))
        @test Δ < deg2rad(4.0)
        @test spread_ratio > 3.0            # genuinely elongated, not round
    end

    # the anti-diagonal tilts the other way — pins the SIGN of D_RZ, which a
    # magnitude-only bug would sail through.
    #
    # An eigenvector is only defined up to sign, so `eigen` may hand back either
    # end of the axis and atan then differs by π: +45° can arrive as −135°. Fold
    # into (−π/2, π/2] before asking which way it leans.
    canonical(θ) = rem(θ, π, RoundNearest)
    axis_plus, _ = principal_axis(π / 4)
    axis_minus, _ = principal_axis(-π / 4)
    @test canonical(axis_plus) > 0
    @test canonical(axis_minus) < 0
    @test canonical(axis_plus) ≈ -canonical(axis_minus) rtol = 0.05
end

@testitem "Transport direction: a circular field spreads into an annulus" begin
    using RAPID2D: build_wall_diffusion_matrix
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    # b̂ tangent to circles about a centre: material must run AROUND the centre,
    # not outwards. This is the one case where b̂ turns under the blob, so it tests
    # something no uniform field can — and it is where an axis-aligned grid leaks
    # numerically, since a curved field is never aligned with the stencil.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 81, NZ = 81,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G

    R0, Z0 = 1.5, 0.0                      # centre of the circular field
    Dperp, ratio = 0.02, 50.0
    Dpara = Dperp * ratio

    # b̂ = ϕ̂ about (R0, Z0): tangent to circles
    dR2 = @. G.R2D - R0
    dZ2 = @. G.Z2D - Z0
    rr = @. sqrt(dR2^2 + dZ2^2) + 1.0e-30
    bR = @. -dZ2 / rr
    bZ = @. dR2 / rr

    D_RR = @. Dperp + (Dpara - Dperp) * bR^2
    D_RZ = @. (Dpara - Dperp) * bR * bZ
    D_ZZ = @. Dperp + (Dpara - Dperp) * bZ^2
    A = build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ)

    # blob on the ring r = 0.2, at angle 0
    r_ring = 0.2
    n = [
        exp(-((G.R2D[i, j] - (R0 + r_ring))^2 + (G.Z2D[i, j] - Z0)^2) / 4.0e-4)
            for i in 1:G.NR, j in 1:G.NZ
    ]
    n[G.nodes.on_out_wall_nids] .= 0.0

    # a function, not a bare loop: a testitem body is module top level, where
    # reassigning inside `for` would create a fresh local each iteration
    function march(M, v0, nsteps)
        v = copy(v0)
        for _ in 1:nsteps
            v = M \ v
        end
        return v
    end

    dt = 5.0e-4
    M = sparse(I, size(A, 1), size(A, 2)) - dt * A
    nf = reshape(march(M, vec(n), 20), G.NR, G.NZ)
    nf[G.nodes.on_out_wall_nids] .= 0.0

    # measure spread in POLAR coordinates about the field centre: the angular
    # spread is transport along b̂, the radial spread is everything that leaked
    # across it — physical D⊥ plus whatever the grid added.
    w = sum(nf)
    r_of = @. sqrt((G.R2D - R0)^2 + (G.Z2D - Z0)^2)
    θ_of = @. atan(G.Z2D - Z0, G.R2D - R0)
    r̄ = sum(nf .* r_of) / w
    θ̄ = sum(nf .* θ_of) / w
    var_r = sum(nf .* (r_of .- r̄) .^ 2) / w
    var_t = sum(nf .* (r_of .* (θ_of .- θ̄)) .^ 2) / w   # arc length, comparable to r

    # material follows the curved field: it runs around, not out
    @test var_t / var_r > 10.0                    # measured 34–35
    # and it stays on its ring rather than collapsing to the centre or escaping
    @test r̄ ≈ r_ring rtol = 0.25

    # An effective perpendicular diffusivity read back from the radial spread,
    # with the blob's initial width removed so what remains is transport. A curved
    # field is never aligned with an axis-aligned stencil, so this is where a
    # numerical cross-field leak would appear — and it turns out to be small:
    #
    #     N       ΔR        var_t/var_r     D_eff⊥/D⊥
    #     61      0.0167    34.4            1.02
    #     81      0.0125    34.7            1.01
    #     121     0.0083    34.9            1.00
    #     161     0.0063    35.0            1.00
    #
    # 2 % at the coarsest grid, vanishing under refinement, at D∥/D⊥ = 50. The
    # 9-point cross-derivative stencil follows a curved field faithfully; the
    # artificial perpendicular diffusion that an axis-aligned grid might have
    # added is not a limitation here.
    var_r0 = sum(n .* (r_of .- sum(n .* r_of) / sum(n)) .^ 2) / sum(n)
    D_eff_perp = max(var_r - var_r0, 0.0) / (2 * dt * 20)
    @test D_eff_perp ≈ Dperp rtol = 0.3
end

@testitem "Transport direction: an isotropic tensor prefers no direction" begin
    using RAPID2D: build_wall_diffusion_matrix, density_second_moments
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    # The control. If the anisotropic cases above passed for some reason unrelated
    # to b̂, this would show it: with D∥ = D⊥ the blob must stay round whatever b̂ is.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 61, NZ = 61,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G

    D = fill(1.0, G.NR, G.NZ)
    Z = zeros(G.NR, G.NZ)
    A = build_wall_diffusion_matrix(G, D, Z, D)

    Rc = (G.R1D[1] + G.R1D[end]) / 2
    Zc = (G.Z1D[1] + G.Z1D[end]) / 2
    n = [
        exp(-((G.R2D[i, j] - Rc)^2 + (G.Z2D[i, j] - Zc)^2) / 1.0e-4)
            for i in 1:G.NR, j in 1:G.NZ
    ]
    n[G.nodes.on_out_wall_nids] .= 0.0

    function march(M, v0, nsteps)
        v = copy(v0)
        for _ in 1:nsteps
            v = M \ v
        end
        return v
    end

    dt = 1.0e-5
    M = sparse(I, size(A, 1), size(A, 2)) - dt * A
    ΣRR, ΣRZ, ΣZZ = density_second_moments(G, reshape(march(M, vec(n), 20), G.NR, G.NZ))
    @test ΣRR / ΣZZ ≈ 1.0 rtol = 0.05
    @test abs(ΣRZ) < 0.05 * sqrt(ΣRR * ΣZZ)
end
