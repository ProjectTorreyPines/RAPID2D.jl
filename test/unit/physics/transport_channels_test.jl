# Transport channels: the (v∥, λ∥, v⊥, λ⊥) basis.
#
# A diffusivity cannot state a wall condition. D = ½vλ is one equation in two
# unknowns: the PDE sees only the product, the wall sees only the speed. So every
# channel declares four numbers per node, and both the tensor and the kinetic
# ceiling are derived from them.
#
# Almost everything here is pure algebra on small hand-built arrays — no grid, no
# RAPID object. That is the point of the basis: the numerics can be driven to
# their limits long before any physical channel model exists.

# ── the tensor ──────────────────────────────────────────────────────────────

@testitem "Transport channel: an isotropic channel has no preferred direction" begin
    using RAPID2D: DiffusionChannel, diffusion_tensor

    # v∥ = v⊥ and λ∥ = λ⊥ must erase b̂ entirely, for EVERY b̂ — not just the
    # axis-aligned ones where D_RZ happens to vanish anyway.
    v, λ = fill(800.0, 2, 3), fill(0.05, 2, 3)
    ch = DiffusionChannel(v, λ, v, λ)
    D = 0.5 * 800.0 * 0.05

    for θ in (0.0, 0.3, π / 4, 1.1, π / 2, 2.7)
        bR, bZ = fill(cos(θ), 2, 3), fill(sin(θ), 2, 3)
        DRR, DRZ, DZZ = diffusion_tensor(ch, bR, bZ)
        @test all(≈(D; rtol = 1.0e-14), DRR)
        @test all(≈(D; rtol = 1.0e-14), DZZ)
        # exactly zero, not approximately: D∥ − D⊥ is identically 0, so the
        # product is 0 whatever b̂ is. A tolerance here would hide a stray term.
        @test all(==(0.0), DRZ)
    end
end

@testitem "Transport channel: an axis-aligned field puts D_para on that axis" begin
    using RAPID2D: DiffusionChannel, diffusion_tensor

    ch = DiffusionChannel(
        fill(1000.0, 1, 1), fill(0.2, 1, 1),   # D∥ = 100
        fill(10.0, 1, 1), fill(0.02, 1, 1)
    )    # D⊥ = 0.1
    Dpara, Dperp = 100.0, 0.1

    # b̂ = R̂
    DRR, DRZ, DZZ = diffusion_tensor(ch, fill(1.0, 1, 1), fill(0.0, 1, 1))
    @test DRR[1] ≈ Dpara rtol = 1.0e-14
    @test DZZ[1] ≈ Dperp rtol = 1.0e-14
    @test DRZ[1] == 0.0

    # b̂ = Ẑ — the two swap, which is what catches a transposed b̂
    DRR, DRZ, DZZ = diffusion_tensor(ch, fill(0.0, 1, 1), fill(1.0, 1, 1))
    @test DRR[1] ≈ Dperp rtol = 1.0e-14
    @test DZZ[1] ≈ Dpara rtol = 1.0e-14
    @test DRZ[1] == 0.0
end

@testitem "Transport channel: only an oblique field produces a cross term" begin
    using RAPID2D: DiffusionChannel, diffusion_tensor

    # THE discriminating case. With b̂ on an axis D_RZ vanishes, so every
    # axis-aligned test would still pass with the cross term deleted from the
    # code. Only an oblique field exercises it.
    ch = DiffusionChannel(
        fill(1000.0, 1, 1), fill(0.2, 1, 1),
        fill(10.0, 1, 1), fill(0.02, 1, 1)
    )
    Dpara, Dperp = 100.0, 0.1
    s = 1 / sqrt(2.0)

    DRR, DRZ, DZZ = diffusion_tensor(ch, fill(s, 1, 1), fill(s, 1, 1))
    @test DRR[1] ≈ (Dpara + Dperp) / 2 rtol = 1.0e-14
    @test DZZ[1] ≈ (Dpara + Dperp) / 2 rtol = 1.0e-14
    @test DRZ[1] ≈ (Dpara - Dperp) / 2 rtol = 1.0e-14
    @test DRZ[1] != 0.0

    # the anti-diagonal flips its sign — pins the b_R·b_Z product, not |b|
    _, DRZ_anti, _ = diffusion_tensor(ch, fill(s, 1, 1), fill(-s, 1, 1))
    @test DRZ_anti[1] ≈ -(Dpara - Dperp) / 2 rtol = 1.0e-14

    # a general angle, against the closed form
    θ = 0.7
    bR, bZ = fill(cos(θ), 1, 1), fill(sin(θ), 1, 1)
    DRR, DRZ, DZZ = diffusion_tensor(ch, bR, bZ)
    @test DRR[1] ≈ Dperp + (Dpara - Dperp) * cos(θ)^2 rtol = 1.0e-14
    @test DRZ[1] ≈ (Dpara - Dperp) * cos(θ) * sin(θ) rtol = 1.0e-14
    @test DZZ[1] ≈ Dperp + (Dpara - Dperp) * sin(θ)^2 rtol = 1.0e-14
end

@testitem "Transport channel: the tensor's eigenvectors are the field direction" begin
    using RAPID2D: DiffusionChannel, diffusion_tensor
    using RAPID2D.LinearAlgebra

    # The strongest single statement of "the tensor means what we think": its
    # principal axes ARE b̂ and its normal, with eigenvalues D∥ and D⊥.
    ch = DiffusionChannel(
        fill(1000.0, 1, 1), fill(0.2, 1, 1),
        fill(10.0, 1, 1), fill(0.02, 1, 1)
    )
    Dpara, Dperp = 100.0, 0.1

    θ = 0.6
    bR, bZ = cos(θ), sin(θ)
    DRR, DRZ, DZZ = diffusion_tensor(ch, fill(bR, 1, 1), fill(bZ, 1, 1))
    𝐃 = [DRR[1] DRZ[1]; DRZ[1] DZZ[1]]

    @test issymmetric(𝐃)
    vals, vecs = eigen(Symmetric(𝐃))
    @test minimum(vals) ≈ Dperp rtol = 1.0e-12      # positive semi-definite
    @test maximum(vals) ≈ Dpara rtol = 1.0e-12

    # the D∥ eigenvector is b̂ (up to sign)
    v_para = vecs[:, argmax(vals)]
    @test abs(v_para ⋅ [bR, bZ]) ≈ 1.0 rtol = 1.0e-12
end

@testitem "Transport channel: the basis varies node by node" begin
    using RAPID2D: DiffusionChannel, diffusion_tensor, channel_D_para, channel_D_perp

    # Not a global constant. A scalar-broadcast bug reproduces every uniform test
    # above and fails only here, so this is the guard for the whole design.
    NR, NZ = 4, 5
    v_para = [100.0 * i + j for i in 1:NR, j in 1:NZ]
    λ_para = [0.01 * j for i in 1:NR, j in 1:NZ]
    v_perp = [3.0 * i for i in 1:NR, j in 1:NZ]
    λ_perp = fill(0.002, NR, NZ)
    ch = DiffusionChannel(v_para, λ_para, v_perp, λ_perp)

    @test channel_D_para(ch) ≈ 0.5 .* v_para .* λ_para rtol = 1.0e-14
    @test channel_D_perp(ch) ≈ 0.5 .* v_perp .* λ_perp rtol = 1.0e-14

    # b̂ varies too, so a node cannot borrow its neighbour's direction either
    bR = [cos(0.1 * i + 0.2 * j) for i in 1:NR, j in 1:NZ]
    bZ = sqrt.(1 .- bR .^ 2)
    DRR, DRZ, DZZ = diffusion_tensor(ch, bR, bZ)

    # parentheses are load-bearing: `@.` would otherwise swallow the `rtol` argument
    Dpa, Dpe = channel_D_para(ch), channel_D_perp(ch)
    @test DRR ≈ (@. Dpe + (Dpa - Dpe) * bR^2) rtol = 1.0e-14
    @test DRZ ≈ (@. (Dpa - Dpe) * bR * bZ) rtol = 1.0e-14
    @test DZZ ≈ (@. Dpe + (Dpa - Dpe) * bZ^2) rtol = 1.0e-14

    # and the result genuinely varies — a constant answer would pass the formulas
    # above only if the inputs were constant, which they are not
    @test length(unique(round.(DRR; digits = 9))) > 1
    @test length(unique(round.(DRZ; digits = 9))) > 1
end

# ── the kinetic ceiling ─────────────────────────────────────────────────────

@testitem "Channel ceiling: uses the MEAN speed, not the D-convention speed" begin
    using RAPID2D: DiffusionChannel, channel_ceiling, MEAN_SPEED_FACTOR

    # The trap. A channel declares v in the convention D = ½·v·λ, but a one-sided
    # flux needs the MEAN speed v̄ = √(8/π)·v — a factor 1.596. Both scale as
    # √(T/m), so no scaling test can separate them; only the absolute ratio can.
    @test MEAN_SPEED_FACTOR ≈ sqrt(8 / π) rtol = 1.0e-14
    @test MEAN_SPEED_FACTOR ≈ 1.5958 rtol = 1.0e-4

    v = 1000.0
    ch = DiffusionChannel(fill(v, 1, 1), fill(0.2, 1, 1), fill(v, 1, 1), fill(0.2, 1, 1))
    # isotropic, so g is irrelevant; ceiling = ¼·v̄ = ¼·√(8/π)·v
    @test channel_ceiling(ch, 1.0, 0.0, (1, 0))[1] ≈ 0.25 * sqrt(8 / π) * v rtol = 1.0e-14
    @test channel_ceiling(ch, 1.0, 0.0, (1, 0))[1] ≈ 0.3989 * v rtol = 1.0e-3
end

@testitem "Channel ceiling: an isotropic channel ignores the face orientation" begin
    using RAPID2D: DiffusionChannel, channel_ceiling

    # When v∥ = v⊥ the ceiling must not depend on n̂ at all. This is the guard
    # against double-counting the anisotropy: the supply side already carries it
    # through D_nn, so putting a directional factor on the ceiling too is wrong.
    v = 1000.0
    ch = DiffusionChannel(fill(v, 1, 1), fill(0.2, 1, 1), fill(v, 1, 1), fill(0.05, 1, 1))
    bR, bZ = cos(0.4), sin(0.4)

    c = [channel_ceiling(ch, bR, bZ, o)[1] for o in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    @test all(≈(c[1]; rtol = 1.0e-14), c)
end

@testitem "Channel ceiling: a perpendicular-only channel cannot reach a head-on wall" begin
    using RAPID2D: DiffusionChannel, channel_ceiling

    # Bohm is the case: it has no parallel transport at all. A wall the field
    # points straight into (g = 1) can only be reached along B, so a cross-field
    # channel must contribute exactly nothing there.
    ch = DiffusionChannel(
        fill(0.0, 1, 1), fill(0.0, 1, 1),      # v∥ = 0
        fill(2500.0, 1, 1), fill(4.0e-4, 1, 1)
    )

    # b̂ = R̂ and an R-face → g = 1
    @test channel_ceiling(ch, 1.0, 0.0, (1, 0))[1] == 0.0
    # the same field on a Z-face → g = 0, full perpendicular ceiling
    @test channel_ceiling(ch, 1.0, 0.0, (0, 1))[1] ≈ 0.25 * sqrt(8 / π) * 2500.0 rtol = 1.0e-14
end

@testitem "Channel ceiling: a grazing field recovers the sin(alpha) projection" begin
    using RAPID2D: DiffusionChannel, channel_ceiling

    # The magnetic projection that a presheath model is usually invoked for falls
    # straight out of the basis. A parallel-only channel meeting the wall at α
    # delivers ¼·v̄∥·sin α, and at RAPID2D's B_pol/B_φ ≈ 8e-3 that is a 125×
    # suppression — obtained with no sheath model anywhere.
    v_para = 1.5e6
    ch = DiffusionChannel(
        fill(v_para, 1, 1), fill(1.0, 1, 1),
        fill(0.0, 1, 1), fill(0.0, 1, 1)
    )       # v⊥ = 0

    α = 8.0e-3                       # ≈ 0.46°, the manual configuration's pitch
    bR, bZ = sin(α), cos(α)          # b̂ nearly in the wall; R-face → g = sin²α
    expected = 0.25 * sqrt(8 / π) * v_para * sin(α)
    @test channel_ceiling(ch, bR, bZ, (1, 0))[1] ≈ expected rtol = 1.0e-12

    # and it really is a suppression, not a rounding artefact
    head_on = channel_ceiling(ch, 1.0, 0.0, (1, 0))[1]
    @test head_on / channel_ceiling(ch, bR, bZ, (1, 0))[1] ≈ 1 / sin(α) rtol = 1.0e-12
end

@testitem "Channel ceiling: the two faces of one cell differ under an oblique field" begin
    using RAPID2D: DiffusionChannel, channel_ceiling

    # g = (b̂·n̂)² is a FACE property, not a cell property. A staircase corner cell
    # owns one R-face and one Z-face, and they must not share a ceiling unless the
    # field happens to bisect them.
    ch = DiffusionChannel(
        fill(1.0e5, 1, 1), fill(1.0, 1, 1),
        fill(100.0, 1, 1), fill(0.01, 1, 1)
    )

    θ = 0.3                                     # oblique
    bR, bZ = cos(θ), sin(θ)
    cR = channel_ceiling(ch, bR, bZ, (1, 0))[1]
    cZ = channel_ceiling(ch, bR, bZ, (0, 1))[1]
    @test !(cR ≈ cZ)
    @test cR > cZ                               # b̂ leans toward R̂

    # at 45° they coincide — the symmetry that makes the oblique case meaningful
    s = 1 / sqrt(2.0)
    @test channel_ceiling(ch, s, s, (1, 0))[1] ≈ channel_ceiling(ch, s, s, (0, 1))[1] rtol = 1.0e-14

    # the sign of the outward direction cannot matter: g is squared
    @test channel_ceiling(ch, bR, bZ, (-1, 0))[1] ≈ cR rtol = 1.0e-14
    @test channel_ceiling(ch, bR, bZ, (0, -1))[1] ≈ cZ rtol = 1.0e-14
end

@testitem "Channel ceiling: independent channels add their diffusivities and ceilings" begin
    using RAPID2D: DiffusionChannel, channel_ceiling, diffusion_tensor,
        channel_D_para, channel_D_perp, total_ceiling, total_tensor

    # §2.2: channels with DIFFERENT characteristic speeds are independent arrival
    # mechanisms, so both fluxes and ceilings add. (Sub-processes that share one
    # speed combine by Matthiessen *inside* a channel and are counted once — that
    # is the gas's three-way λ, already handled by neutral_gas.jl.)
    fast = DiffusionChannel(
        fill(1.0e6, 1, 1), fill(1.0, 1, 1),
        fill(0.0, 1, 1), fill(0.0, 1, 1)
    )
    slow = DiffusionChannel(
        fill(0.0, 1, 1), fill(0.0, 1, 1),
        fill(60.0, 1, 1), fill(1.0, 1, 1)
    )
    chs = (fast, slow)
    bR, bZ = cos(0.4), sin(0.4)

    @test total_ceiling(chs, bR, bZ, (1, 0))[1] ≈
        channel_ceiling(fast, bR, bZ, (1, 0))[1] +
        channel_ceiling(slow, bR, bZ, (1, 0))[1] rtol = 1.0e-14

    DRR, DRZ, DZZ = total_tensor(chs, fill(bR, 1, 1), fill(bZ, 1, 1))
    f = diffusion_tensor(fast, fill(bR, 1, 1), fill(bZ, 1, 1))
    s = diffusion_tensor(slow, fill(bR, 1, 1), fill(bZ, 1, 1))
    @test DRR[1] ≈ f[1][1] + s[1][1] rtol = 1.0e-14
    @test DRZ[1] ≈ f[2][1] + s[2][1] rtol = 1.0e-14
    @test DZZ[1] ≈ f[3][1] + s[3][1] rtol = 1.0e-14
end

# ── the Phase 0 geometry gap ────────────────────────────────────────────────

@testitem "Wall faces: concave and convex staircase corners" begin
    using RAPID2D: wall_faces

    # The last unasserted row of the design's wall-geometry matrix. An L-shaped
    # wall carries both orientations:
    #
    #        i-1   i   i+1              i-1   i   i+1
    #  j+1    ·    ·    ·          j+1   ■    ■    ·
    #  j      ·    C    ■          j     ■    C    ■
    #  j-1    ·    ■    ■          j-1   ■    ■    ■
    #         convex: 2 faces           concave: 0 faces
    #
    # Zero is the right answer for the re-entrant cell — no flux crosses a corner
    # in a five-point face set — but it must be asserted, because an
    # implementation that consulted diagonal neighbours would invent a face on a
    # cell that has no wall-adjacent boundary.
    #
    # It also marks where the 9-point cross terms will leak: that cell owns no
    # face, so the Robin condition never touches it, yet its CTRZ arm reaches the
    # outside diagonal. Face-based accounting cannot see that by construction.
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
    faces = wall_faces(G)
    n_faces = zeros(Int, G.NR, G.NZ)
    for f in faces
        n_faces[f.rid, f.zid] += 1
    end

    state = G.nodes.state
    is_in(i, j) = 1 <= i <= G.NR && 1 <= j <= G.NZ && state[i, j] > 0.5

    # the polygon has 5 convex vertices and 1 re-entrant one
    @test count(==(2), n_faces) == 5
    @test maximum(n_faces) == 2                  # a corner never takes three

    # re-entrant cells: every cardinal neighbour in-wall, some diagonal outside
    reentrant = [
        (i, j) for j in 1:G.NZ, i in 1:G.NR
            if is_in(i, j) &&
            all(is_in(i + di, j + dj) for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))) &&
            any(!is_in(i + di, j + dj) for (di, dj) in ((1, 1), (1, -1), (-1, 1), (-1, -1)))
    ]
    @test length(reentrant) == 1
    @test all(n_faces[i, j] == 0 for (i, j) in reentrant)

    # both orientations still respect the invariant that owns this design: a face
    # exists exactly where an in-wall cell meets a non-in-wall cardinal neighbour
    for f in faces
        @test is_in(f.rid, f.zid)
        @test !is_in(f.rid + f.outward[1], f.zid + f.outward[2])
    end
end
