# Wall face geometry, and the one-sided flux that bombards it.
#
# Step 1 of the Robin wall-boundary sequence: geometry and diagnostics only, no
# solver change. What must be nailed down here is (a) which faces exist and which
# way they point, (b) how much area each carries, and (c) the kinetic ceiling
# ¼n·v̄ that any absorption rate must eventually respect.

# ── which faces exist ───────────────────────────────────────────────────────

@testitem "Wall faces: one face for every arm the reflective stencil omits" begin
    using RAPID2D: wall_faces, build_reflective_diffusion_matrix
    using RAPID2D.SparseArrays

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    G = RP.G
    faces = wall_faces(G)
    @test !isempty(faces)

    # The reflective operator is an INDEPENDENT statement of the same geometry: it
    # omits exactly the arms that cross the wall. If the two ever disagree, the
    # Robin debit of step 2 would land on faces the flux terms do not.
    A = build_reflective_diffusion_matrix(G, fill(50.0, G.NR, G.NZ))

    for f in faces
        # the owner is in-wall and the cell across the face is not — direction, not
        # just count. An inward-pointing face would still balance the tally below.
        @test G.nodes.state[f.rid, f.zid] > 0.5
        ii, jj = f.rid + f.outward[1], f.zid + f.outward[2]
        if 1 <= ii <= G.NR && 1 <= jj <= G.NZ
            @test G.nodes.state[ii, jj] <= 0.5
            @test A[f.nid, G.nodes.nid[ii, jj]] == 0.0
        end
    end

    # Each of the four cardinal directions is either an in-wall neighbour (one
    # off-diagonal entry) or a wall face. So the two counts must be complementary.
    offdiag = copy(A)
    for k in axes(offdiag, 1)
        offdiag[k, k] = 0.0
    end
    dropzeros!(offdiag)
    n_offdiag = vec(sum(offdiag .!= 0, dims = 2))

    n_faces = zeros(Int, G.NR * G.NZ)
    for f in faces
        n_faces[f.nid] += 1
    end
    for k in G.nodes.in_wall_nids
        @test n_faces[k] == 4 - n_offdiag[k]
    end

    # and nothing outside the wall owns a face
    @test all(n_faces[k] == 0 for k in G.nodes.on_out_wall_nids)
end

@testitem "Wall faces: a staircase corner takes both of its faces" begin
    using RAPID2D: wall_faces

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
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

    # The default manual wall is an axis-aligned rectangle, so the in-wall region
    # is a rectangle too. Its four corner cells each have an outward R-face AND an
    # outward Z-face; crediting only one is the "corner debited once" bug.
    rids = [f.rid for f in faces]
    zids = [f.zid for f in faces]
    i0, i1 = minimum(rids), maximum(rids)
    j0, j1 = minimum(zids), maximum(zids)

    for (i, j) in ((i0, j0), (i0, j1), (i1, j0), (i1, j1))
        @test G.nodes.state[i, j] > 0.5
        @test n_faces[i, j] == 2
    end
    @test count(==(2), n_faces) == 4          # only the corners
    @test maximum(n_faces) == 2               # nothing takes three
end

# ── how much area each face carries ─────────────────────────────────────────

@testitem "Wall faces: R- and Z-faces carry their own spacing" begin
    using RAPID2D: wall_faces

    # dR != dZ deliberately. On a square grid a swapped dR/dZ in the face factor is
    # invisible; here it is a factor of two.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 21, NZ = 21,
        R_min = 1.0, R_max = 2.0, Z_min = -1.0, Z_max = 1.0,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    G = RP.G
    @test G.dR ≈ 0.05
    @test G.dZ ≈ 0.1                          # genuinely non-square
    faces = wall_faces(G)

    n_R = count(f -> f.outward[1] != 0, faces)
    n_Z = count(f -> f.outward[2] != 0, faces)
    @test n_R > 0 && n_Z > 0

    for f in faces
        R = G.R2D[f.rid, f.zid]
        V = 2π * R * G.dR * G.dZ               # cylindrical cell volume
        if f.outward[2] == 0
            # R-face: an annulus-edged cylinder at R ± dR/2. Its radius is NOT the
            # cell-centre radius, which is what makes the two sides differ below.
            R_face = R + f.outward[1] * G.dR / 2
            @test f.area ≈ 2π * R_face * G.dZ rtol = 1.0e-14
            @test f.area_per_volume ≈ R_face / (R * G.dR) rtol = 1.0e-14
        else
            # Z-face: a flat annulus of width dR at the cell radius. Here — and only
            # here — the A_f/V_i factor is exactly 1/dZ.
            @test f.area ≈ 2π * R * G.dR rtol = 1.0e-14
            @test f.area_per_volume ≈ 1 / G.dZ rtol = 1.0e-14
        end
        # area and area_per_volume must describe the SAME face of the SAME cell
        @test f.area / f.area_per_volume ≈ V rtol = 1.0e-14
    end
end

@testitem "Wall faces: the R-face factor is not 1/dR — inboard and outboard differ" begin
    using RAPID2D: wall_faces

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 21, NZ = 21,
        R_min = 1.0, R_max = 2.0, Z_min = -1.0, Z_max = 1.0,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    G = RP.G
    faces = wall_faces(G)

    # The design note writes A_f/V_i = 1/dR for an R-face, which drops the factor
    # R_{i±½}/R_i. That factor is not decoration: build_reflective_diffusion_matrix
    # already carries it (invJ·½(CT_out + CT_in) is (1/J_i)·J_face·D/dR², and
    # ½(R_i + R_{i±1}) = R_{i±½} exactly because Jacob = R is linear), and the
    # conserved measure Σ J·n only closes against the true face area. Dropping it
    # would make the two sides of a given row equal — hence this test.
    j_mid = (minimum(f.zid for f in faces) + maximum(f.zid for f in faces)) ÷ 2
    row = [f for f in faces if f.zid == j_mid && f.outward[2] == 0]
    @test length(row) == 2                     # one inboard, one outboard

    inboard = only(f for f in row if f.outward[1] == -1)
    outboard = only(f for f in row if f.outward[1] == +1)

    @test inboard.area_per_volume < 1 / G.dR   # face is nearer the axis than the cell
    @test outboard.area_per_volume > 1 / G.dR  # face is further out
    @test !(inboard.area_per_volume ≈ outboard.area_per_volume)

    for f in (inboard, outboard)
        R = G.R2D[f.rid, f.zid]
        @test f.area_per_volume ≈ (1 + f.outward[1] * G.dR / (2R)) / G.dR rtol = 1.0e-14
    end
end

@testitem "Wall faces: the grid frame may itself be the wall" begin
    using RAPID2D: wall_faces

    # Wall polygon larger than the domain, so every node is in-wall and the outward
    # neighbours of the frame cells lie off-grid entirely. Nothing may index there.
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
    @test all(G.nodes.state[i, j] > 0.5 for i in 1:G.NR, j in 1:G.NZ)

    faces = wall_faces(G)

    # exactly the frame: 2 R-columns × NZ  +  2 Z-rows × NR, corners counted twice
    @test count(f -> f.outward[1] != 0, faces) == 2 * G.NZ
    @test count(f -> f.outward[2] != 0, faces) == 2 * G.NR
    @test all(1 <= f.rid <= G.NR && 1 <= f.zid <= G.NZ for f in faces)
end

# ── the staircase area over-count (design note §3.4b) ───────────────────────

@testitem "Wall faces: an axis-aligned wall carries no area penalty" begin
    using RAPID2D: wall_faces

    # Lateral area of the surface of revolution, edge by edge (Pappus, frustum).
    revolved_area(Rw, Zw) = sum(
        π * (Rw[k] + Rw[k + 1]) * hypot(Rw[k + 1] - Rw[k], Zw[k + 1] - Zw[k])
            for k in 1:(length(Rw) - 1)
    )
    box(Ra, Rb, Za, Zb) = revolved_area(
        [Ra, Rb, Rb, Ra, Ra], [Za, Za, Zb, Zb, Za]
    )
    Ra, Rb, Za, Zb = 1.21, 1.79, -0.31, 0.31

    function staircase(NR, NZ)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = NR, NZ = NZ,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [Ra, Rb, Rb, Ra], wall_Z = [Za, Za, Zb, Zb],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        return sum(f.area for f in wall_faces(RP.G)), RP.G.dR, RP.G.dZ
    end

    # A wall normal to a grid axis is represented by faces of exactly the right
    # shape; the only error is WHERE they sit. A node is in-wall only if it lies
    # strictly inside, so every face lands within half a cell of the polygon, and
    # the swept area is bracketed by the polygon shrunk and grown by that much.
    #
    # Bracketing rather than a tolerance on purpose: which nodes fall inside jumps
    # discretely with the grid, so the error is a SAWTOOTH in dx, not a monotone
    # sequence — a `finer is closer` assertion would be flaky. The bracket is pure
    # geometry and holds at every resolution.
    #
    # This is the control for the 45° case below: the penalty here is a placement
    # error that halves when the grid does, not an area penalty that stays.
    A_true = box(Ra, Rb, Za, Zb)
    widths = Float64[]
    for (NR, NZ) in ((45, 52), (89, 103))
        A_stair, dR, dZ = staircase(NR, NZ)
        A_lo = box(Ra + dR / 2, Rb - dR / 2, Za + dZ / 2, Zb - dZ / 2)
        A_hi = box(Ra - dR / 2, Rb + dR / 2, Za - dZ / 2, Zb + dZ / 2)
        @test A_lo <= A_stair <= A_hi
        @test A_stair ≈ A_true rtol = 0.01     # already within 1 %, vs 41 % at 45°
        push!(widths, (A_hi - A_lo) / A_true)
    end
    @test widths[2] < 0.55 * widths[1]         # first order: halving dx halves it
end

@testitem "Wall faces: a 45° wall over-counts area by √2 at every resolution" begin
    using RAPID2D: wall_faces

    revolved_area(Rw, Zw) = sum(
        π * (Rw[k] + Rw[k + 1]) * hypot(Rw[k + 1] - Rw[k], Zw[k + 1] - Zw[k])
            for k in 1:(length(Rw) - 1)
    )

    function staircase_ratio(NR, NZ)
        # A diamond: all four edges at 45°, so no edge needs isolating. dR = dZ is
        # required for the staircase to actually be at 45°.
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = NR, NZ = NZ,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.8, 1.5, 1.2, 1.5], wall_Z = [0.0, 0.3, 0.0, -0.3],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        @test RP.G.dR ≈ RP.G.dZ
        return sum(f.area for f in wall_faces(RP.G)) /
            revolved_area(RP.wall.R, RP.wall.Z)
    end

    # Measured rather than assumed. A staircase presents dR + dZ of face per cell
    # where the true wall is √(dR² + dZ²) — a factor √2 at 45°, and unlike the
    # axis-aligned case above it does NOT shrink as the grid refines. Every absorbed
    # flux, and so every recycling and sputtering source, inherits it on inclined
    # segments. Recorded here so a later step cannot quietly assume 1.
    #
    #     grid      dR        ΣA_f/A_poly    excess over √2
    #     41²       0.0250    1.4731         +4.2 %
    #     61²       0.0167    1.4535         +2.8 %
    #     81²       0.0125    1.4437         +2.1 %
    #     121²      0.0083    1.4339         +1.4 %
    #     161²      0.0063    1.4289         +1.0 %
    #
    # √2 is the ASYMPTOTE and it is approached from above: the excess is O(dx) and
    # halves when the grid does, because the diamond's four corners are locally
    # axis-aligned and contribute the vanishing error of the previous test. So a
    # production grid pays somewhat MORE than √2, never less.
    coarse = staircase_ratio(41, 41)
    fine = staircase_ratio(81, 81)
    @test coarse ≈ sqrt(2) rtol = 0.06
    @test fine ≈ sqrt(2) rtol = 0.06
    @test fine > sqrt(2)                       # approached from above
    @test fine < coarse                        # ... and it is an approach
    @test fine - sqrt(2) < 0.6 * (coarse - sqrt(2))   # the excess is first order
    @test fine > 1.35                          # refinement does not remove the √2
end

# ── the one-sided flux that bombards the wall ───────────────────────────────

@testitem "Wall impingement: v_incident is a quarter of the MEAN speed" begin
    using RAPID2D: v_incident, neutral_gas_thermal_speed, M_H2_GAS

    # Two different speeds live in this codebase and they differ by √(8/π) = 1.596.
    # neutral_gas.jl's v_th = √(T/m) is the convention D = ½·v_th·λ needs; the
    # Hertz-Knudsen flux needs the MEAN speed v̄ = √(8T/πm). Taking ¼v_th instead of
    # ¼v̄ under-counts every impact by 37 %, and no scaling law would expose it —
    # both go as √(T/m).
    T = 0.026
    @test v_incident(T, M_H2_GAS) ≈
        0.25 * sqrt(8 / π) * neutral_gas_thermal_speed(T) rtol = 1.0e-14
    @test v_incident(T, M_H2_GAS) / neutral_gas_thermal_speed(T) ≈ 0.3989 rtol = 1.0e-3

    # v̄ ∝ √(T/m): four times the temperature doubles it, four times the mass halves it
    @test v_incident(4T, M_H2_GAS) ≈ 2 * v_incident(T, M_H2_GAS) rtol = 1.0e-14
    @test v_incident(T, 4 * M_H2_GAS) ≈ v_incident(T, M_H2_GAS) / 2 rtol = 1.0e-14
end

@testitem "Wall impingement: reproduces the Hertz-Knudsen rate of the 1e-2 Pa fill" begin
    using RAPID2D: gross_impingement, v_incident, M_H2_GAS

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 15, NZ = 17,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    # The absolute number both design notes quote for the standard fill. Pinning it
    # against the initialised state, rather than against a hand-substituted n, is
    # what ties the documented 1.07e21 m⁻²s⁻¹ to the code that produces it.
    k = RP.G.nodes.in_wall_nids[1]
    Γ = gross_impingement(RP.plasma.n_H2_gas[k], RP.plasma.T_gas_eV, M_H2_GAS)
    @test Γ ≈ 1.07e21 rtol = 0.01

    # linear in density, and the velocity factor is exactly v_incident
    @test gross_impingement(2.0e18, 0.026, M_H2_GAS) ≈
        2 * gross_impingement(1.0e18, 0.026, M_H2_GAS) rtol = 1.0e-14
    @test gross_impingement(1.0e18, 0.026, M_H2_GAS) ≈
        1.0e18 * v_incident(0.026, M_H2_GAS) rtol = 1.0e-14
    @test gross_impingement(0.0, 0.026, M_H2_GAS) == 0.0
end

@testitem "Wall impingement: the total bombardment rate is an area-weighted sum" begin
    using RAPID2D: wall_faces, gross_impingement, M_H2_GAS

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    # Γ is a flux DENSITY [m⁻²s⁻¹]; turning it into particles per second is what the
    # face area is for. A uniform fill makes the sum exactly Γ·ΣA_f, so a face
    # carrying the wrong area — or being counted twice — shows up here.
    faces = wall_faces(RP.G)
    n_w = 2.4e18
    RP.plasma.n_H2_gas .= n_w

    Γ = gross_impingement(n_w, RP.plasma.T_gas_eV, M_H2_GAS)
    rate = sum(
        f.area * gross_impingement(
                RP.plasma.n_H2_gas[f.rid, f.zid], RP.plasma.T_gas_eV, M_H2_GAS
            ) for f in faces
    )
    @test rate ≈ Γ * sum(f.area for f in faces) rtol = 1.0e-14
    @test rate > 0
end
