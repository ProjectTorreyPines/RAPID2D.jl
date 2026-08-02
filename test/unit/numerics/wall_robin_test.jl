# The Robin debit: what the wall actually removes.
#
# Phase 2 gave a reflective wall — nothing crosses, everything is conserved. This
# adds the one term that lets the wall absorb:
#
#     diag_i −= (A_f/V_i)·v_absorb_f      per wall face, not per cell
#
# Since v_absorb ≥ 0 the diagonal only becomes more negative, so the operator
# drains and never sources. The three familiar wall conditions become one formula:
# v_absorb = 0 is reflective, v_absorb → ∞ is Dirichlet, and ¼v̄(1−R) is the
# physical case in between.
#
# The defect this replaces is that today's schemes are ALSO Robin conditions, with
# v_absorb = D/(2Δx) — a discretisation artefact rather than a surface property,
# which is why the absorbed rate does not converge and why no albedo can be
# expressed.

@testitem "Robin: a fully reflective wall is bit-identical to no wall term at all" begin
    using RAPID2D: build_wall_diffusion_matrix, wall_faces

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)

    Dpara, Dperp = 100.0, 0.1
    θ = 0.6
    bR, bZ = cos(θ), sin(θ)
    D_RR = fill(Dperp + (Dpara - Dperp) * bR^2, G.NR, G.NZ)
    D_RZ = fill((Dpara - Dperp) * bR * bZ, G.NR, G.NZ)
    D_ZZ = fill(Dperp + (Dpara - Dperp) * bZ^2, G.NR, G.NZ)

    # `==`, not `≈`. Everything established in Phase 2 — exact conservation, exact
    # zeros outside — must survive the new term untouched at R = 1, or none of it
    # can be relied on afterwards.
    plain = build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ)
    robin = build_wall_diffusion_matrix(
        G, D_RR, D_RZ, D_ZZ;
        faces = faces, v_absorb = zeros(length(faces))
    )
    @test robin == plain
end

@testitem "Robin: absorption is monotone in the albedo and maximal at R = 0" begin
    using RAPID2D: build_wall_diffusion_matrix, wall_faces
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 31, NZ = 31,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)

    D = fill(5.0, G.NR, G.NZ)
    Z = zeros(G.NR, G.NZ)
    v_incident_face = 400.0                # a stand-in ¼v̄; the physics comes later
    Jv = vec(G.Jacob)
    inw = G.nodes.in_wall_nids

    function absorbed_fraction(R_albedo)
        v_abs = fill(v_incident_face * (1 - R_albedo), length(faces))
        A = build_wall_diffusion_matrix(G, D, Z, D; faces = faces, v_absorb = v_abs)
        n0 = zeros(G.NR * G.NZ)
        n0[inw] .= 1.0e18
        dt = 1.0e-5
        M = sparse(I, size(A, 1), size(A, 2)) - dt * A
        v = copy(n0)
        for _ in 1:40
            v = M \ v
        end
        before = sum(Jv[k] * n0[k] for k in inw)
        after = sum(Jv[k] * v[k] for k in inw)
        return (before - after) / before
    end

    fracs = [absorbed_fraction(R) for R in (0.0, 0.25, 0.5, 0.9, 0.99, 1.0)]

    # R = 1 removes exactly nothing — the reflective invariant survives
    @test fracs[end] ≈ 0.0 atol = 1.0e-12
    # strictly decreasing in R: more reflection, less loss
    @test all(fracs[k] > fracs[k + 1] for k in 1:(length(fracs) - 1))
    # R = 0 is the maximum: no albedo removes more
    @test fracs[1] == maximum(fracs)
    @test fracs[1] > 0.0
end

@testitem "Robin: the wall never absorbs faster than particles can arrive" begin
    using RAPID2D: build_wall_diffusion_matrix, wall_faces
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    # The defect this whole design exists to remove. With v_absorb = D/(2Δx) the
    # face coefficient carries the grid, and measured against the kinetic ceiling
    # it runs 15–21× too fast at production resolution and grows without bound as
    # Δx → 0 — the wall removing more than can physically arrive. With a Robin
    # condition the flux is v_absorb·n_w by construction, so the ceiling holds at
    # every grid. This asserts that, and contrasts it with what D/(2Δx) would do.
    v_inc = 400.0
    D_val = 5.0

    function check(N)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = N, NZ = N,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        G = RP.G
        faces = wall_faces(G)
        D = fill(D_val, G.NR, G.NZ)
        Z = zeros(G.NR, G.NZ)

        plain = build_wall_diffusion_matrix(G, D, Z, D)
        robin = build_wall_diffusion_matrix(
            G, D, Z, D;
            faces = faces, v_absorb = fill(v_inc, length(faces))
        )

        # Read the absorption velocity back OUT of the assembled matrix rather
        # than trusting what was put in: the debit on a cell is Σ_f (A_f/V_i)·v,
        # so dividing the diagonal change by Σ_f (A_f/V_i) must recover v exactly,
        # on every cell, at every grid. That is the whole claim — the coefficient
        # is a surface property and carries no Δx.
        worst = 0.0
        for k in unique(f.nid for f in faces)
            debit = plain[k, k] - robin[k, k]
            aov = sum(f.area_per_volume for f in faces if f.nid == k)
            worst = max(worst, abs(debit / aov - v_inc) / v_inc)
        end

        # what the scheme being replaced would have used, in the same units
        return worst, (D_val / (2 * G.dR)) / v_inc
    end

    old_ratios = Float64[]
    for N in (31, 61, 121, 181)
        recovered_err, old_over_ceiling = check(N)
        # the Robin coefficient is exactly v_absorb on every grid — no Δx in it
        @test recovered_err < 1.0e-12
        push!(old_ratios, old_over_ceiling)
    end

    # The contrast. Whether D/(2Δx) exceeds a given ceiling at a given resolution
    # depends on D and v̄ — with these values it crosses only between N = 121 and
    # 181 (0.19, 0.38, 0.75, 1.13). What is universal is that it grows like 1/Δx,
    # so it crosses ANY ceiling once the grid is fine enough: measured here as an
    # exact doubling per halving of Δx.
    @test old_ratios == sort(old_ratios)
    for k in 1:(length(old_ratios) - 2)
        @test old_ratios[k + 1] / old_ratios[k] ≈ 2.0 rtol = 0.05
    end
    @test old_ratios[end] > 1.0            # and by N = 181 it has crossed
    @test old_ratios[1] < 1.0              # while at N = 31 it had not
end

@testitem "Robin: the absorbed fraction converges under grid refinement" begin
    using RAPID2D: build_wall_diffusion_matrix, wall_faces
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    # The other half of the §1.1 defect: with a grid-set coefficient the absorbed
    # fraction over a fixed physical time varies 4× across N_R = 31 → 181 and has
    # not converged. A surface property must give the same answer on any grid.
    function absorbed(N)
        # The wall polygon must be PINNED. Without wall_R/wall_Z the manual device
        # places it at `3·dR` from the frame, so the vessel would shrink as the
        # grid refines and the scan would compare different geometries rather than
        # different resolutions of one.
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = N, NZ = N,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.21, 1.79, 1.79, 1.21], wall_Z = [-0.31, -0.31, 0.31, 0.31],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        G = RP.G
        faces = wall_faces(G)
        D = fill(5.0, G.NR, G.NZ)
        Z = zeros(G.NR, G.NZ)
        A = build_wall_diffusion_matrix(
            G, D, Z, D;
            faces = faces, v_absorb = fill(400.0, length(faces))
        )
        Jv = vec(G.Jacob)
        inw = G.nodes.in_wall_nids
        n0 = zeros(G.NR * G.NZ)
        n0[inw] .= 1.0e18
        dt = 1.0e-5
        M = sparse(I, size(A, 1), size(A, 2)) - dt * A
        v = copy(n0)
        for _ in 1:20
            v = M \ v
        end
        before = sum(Jv[k] * n0[k] for k in inw)
        after = sum(Jv[k] * v[k] for k in inw)
        return (before - after) / before
    end

    f = [absorbed(N) for N in (31, 61, 121)]
    @test all(>(0), f)

    # Two different grid dependencies, and only one of them was the defect.
    #
    # The COEFFICIENT is now exactly grid-independent — the previous testitem
    # recovers v_absorb from the assembled matrix to 1e-12 at every resolution.
    # That is what D/(2Δx) got wrong, and it is fixed outright.
    #
    # The OUTCOME still converges at first order, because the staircase wall sits
    # within half a cell of the polygon, so the enclosed volume and the wall area
    # are themselves O(Δx) wrong. Measured with the wall pinned, N = 31→181:
    #
    #     0.2317   0.1958   0.1793   0.1738
    #     differences:  −0.0359   −0.0165   −0.0055
    #
    # Ratios of successive differences ≈ 0.46, 0.33 — first order, as expected for
    # a staircase boundary. Removing that needs cut cells, not a boundary
    # condition. Against the scheme this replaces, which spread 4× over the same
    # scan and was not converging at all, this is the difference between a
    # discretisation error and a broken coefficient.
    @test abs(f[3] - f[2]) < 0.6 * abs(f[2] - f[1])
    @test maximum(f) / minimum(f) < 1.35
end

@testitem "Robin: nothing is ever written outside the wall, at any albedo" begin
    using RAPID2D: build_wall_diffusion_matrix, wall_faces

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)

    Dpara, Dperp = 100.0, 0.1
    θ = 0.6
    bR, bZ = cos(θ), sin(θ)
    D_RR = fill(Dperp + (Dpara - Dperp) * bR^2, G.NR, G.NZ)
    D_RZ = fill((Dpara - Dperp) * bR * bZ, G.NR, G.NZ)
    D_ZZ = fill(Dperp + (Dpara - Dperp) * bZ^2, G.NR, G.NZ)

    inside = G.nodes.in_wall_nids
    outside = G.nodes.on_out_wall_nids

    # This is the structural pay-off of the Robin form: the absorbed material is a
    # boundary term on the interior cell, so nothing is transported into cells
    # nothing solves for. It must hold at EVERY albedo, exactly.
    for R_albedo in (0.0, 0.5, 1.0)
        A = build_wall_diffusion_matrix(
            G, D_RR, D_RZ, D_ZZ;
            faces = faces, v_absorb = fill(400.0 * (1 - R_albedo), length(faces))
        )
        @test maximum(abs, A[inside, outside]) == 0.0
        @test maximum(abs, A[outside, :]) == 0.0
        # the debit drains, never sources
        @test all(A[k, k] <= 0 for k in inside)
    end
end

@testitem "Robin: nothing unphysical accumulates over a long run" begin
    using RAPID2D: build_wall_diffusion_matrix, wall_faces
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    # Accuracy is the lesser question — a few per cent of discretisation error is
    # tolerable. What is not tolerable is error that ACCUMULATES into something
    # unphysical: density going negative, material appearing from nowhere, or the
    # total creeping up. Every other testitem here checks a single step; this one
    # checks that a thousand of them compose.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)

    # strongly anisotropic and oblique: the cross terms are active throughout, so
    # any asymmetry they introduce has a thousand steps to compound
    Dpara, Dperp = 100.0, 0.1
    θ = 0.6
    bR, bZ = cos(θ), sin(θ)
    D_RR = fill(Dperp + (Dpara - Dperp) * bR^2, G.NR, G.NZ)
    D_RZ = fill((Dpara - Dperp) * bR * bZ, G.NR, G.NZ)
    D_ZZ = fill(Dperp + (Dpara - Dperp) * bZ^2, G.NR, G.NZ)

    Jv = vec(G.Jacob)
    inw = G.nodes.in_wall_nids
    outw = G.nodes.on_out_wall_nids
    total(v) = sum(Jv[k] * v[k] for k in inw)

    Rc = (G.R1D[1] + G.R1D[end]) / 2
    Zc = (G.Z1D[1] + G.Z1D[end]) / 2
    n0 = zeros(G.NR * G.NZ)
    for k in inw
        i, j = G.nodes.rid[k], G.nodes.zid[k]
        n0[k] = 1.0e18 * (1 + 3exp(-((G.R2D[i, j] - Rc)^2 + (G.Z2D[i, j] - Zc)^2) / 0.02))
    end

    function run(v_abs_val, dt, nsteps)
        A = build_wall_diffusion_matrix(
            G, D_RR, D_RZ, D_ZZ;
            faces = faces, v_absorb = fill(v_abs_val, length(faces))
        )
        F = lu(sparse(I, size(A, 1), size(A, 2)) - dt * A)
        v = copy(n0)
        mins = Float64[]
        max_outside = 0.0
        totals = Float64[total(v)]
        for _ in 1:nsteps
            v = F \ v
            push!(mins, minimum(v[inw]))
            max_outside = max(max_outside, maximum(abs, v[outw]))
            push!(totals, total(v))
        end
        return totals, mins, max_outside
    end

    # ── reflective, 1000 steps: the invariant must not drift ────────────────
    totals, mins, max_out = run(0.0, 1.0e-6, 1000)
    drift = abs(totals[end] - totals[1]) / totals[1]
    @test drift < 1.0e-12                  # not per step — after a thousand of them
    @test minimum(mins) >= 0.0             # strictly non-negative with no absorption
    @test max_out == 0.0                   # outside is never written, ever

    # ── absorbing, long and stiff ───────────────────────────────────────────
    # Δt is ~100× the explicit limit. Here a small undershoot DOES appear, and
    # what matters is that it is bounded and transient rather than compounding.
    #
    # Cause, and it is not the Robin term: the absorption boundary layer is
    # D⊥/v_absorb = 2.5e-4 m against ΔR = 0.067 m — unresolved by ~270×. An
    # unresolved layer under a NON-MONOTONE stencil undershoots, and the stencil is
    # non-monotone only because the cross terms carry the sign of D_RZ (Phase 2:
    # ~1054 of 4312 off-diagonals negative). With an isotropic tensor, or with a
    # v_absorb ten times smaller, the density stays strictly non-negative.
    #
    # Measured over 3000 steps: the undershoot deepens to −3.9e-4 of the initial
    # peak at step 206, then RECOVERS — −2.8e-5 by step 1000, −4e-9 by step 3000 —
    # while the absolute negative mass peaks and decays with it. It does not
    # accumulate.
    totals, mins, max_out = run(400.0, 1.0e-4, 1000)
    peak = maximum(n0)

    @test max_out == 0.0
    @test minimum(mins) > -1.0e-3 * peak           # bounded, and small
    # and it heals: the tail undershoot is a small fraction of the worst
    @test abs(min(mins[end], 0.0)) < 0.2 * abs(minimum(mins))

    # the total may only DECREASE: with v_absorb ≥ 0 and no sources, a step that
    # increased it would be material created out of arithmetic error
    @test all(totals[k + 1] <= totals[k] * (1 + 1.0e-14) for k in 1:(length(totals) - 1))
    @test totals[end] < totals[1]          # and it really is absorbing
    @test totals[end] > 0.0                # without overshooting into nothing

    # the decay is smooth, not oscillatory: successive decrements never grow,
    # which is what ringing would look like
    drops = [totals[k] - totals[k + 1] for k in 1:(length(totals) - 1)]
    @test all(drops[k + 1] <= drops[k] * (1 + 1.0e-9) for k in 1:(length(drops) - 1))

    # ── the undershoot is the anisotropy's, not the wall's ──────────────────
    # Same wall, same v_absorb, same Δt, isotropic tensor: strictly non-negative.
    Diso = fill(Dperp, G.NR, G.NZ)
    Zt = zeros(G.NR, G.NZ)
    A_iso = build_wall_diffusion_matrix(
        G, Diso, Zt, Diso;
        faces = faces, v_absorb = fill(400.0, length(faces))
    )
    # a function, not a bare loop: a testitem body is module top level, where
    # reassigning inside `for` creates a fresh local each iteration
    function worst_over(F, v0, nsteps)
        v = copy(v0)
        worst = Inf
        for _ in 1:nsteps
            v = F \ v
            worst = min(worst, minimum(v[inw]))
        end
        return worst
    end

    F_iso = lu(sparse(I, size(A_iso, 1), size(A_iso, 2)) - 1.0e-4 * A_iso)
    @test worst_over(F_iso, n0, 1000) >= 0.0
end

@testitem "Robin: the albedo is a face property, not a cell or a global" begin
    using RAPID2D: build_wall_diffusion_matrix, wall_faces

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)
    D = fill(5.0, G.NR, G.NZ)
    Z = zeros(G.NR, G.NZ)

    plain = build_wall_diffusion_matrix(G, D, Z, D)

    # absorb on the outboard faces only — a limiter that pumps on one segment
    v_abs = [f.outward == (1, 0) ? 400.0 : 0.0 for f in faces]
    @test count(>(0), v_abs) > 0
    @test count(==(0.0), v_abs) > 0
    A = build_wall_diffusion_matrix(G, D, Z, D; faces = faces, v_absorb = v_abs)

    # every row whose faces all have v_absorb = 0 is untouched, bit for bit
    touched = Set(f.nid for (f, v) in zip(faces, v_abs) if v > 0)
    for k in G.nodes.in_wall_nids
        k in touched && continue
        @test A[k, k] == plain[k, k]
    end
    # and the touched ones are strictly more negative
    for k in touched
        @test A[k, k] < plain[k, k]
    end

    # a staircase corner owning two faces takes BOTH debits, not one
    n_faces = Dict{Int, Int}()
    for f in faces
        n_faces[f.nid] = get(n_faces, f.nid, 0) + 1
    end
    corner = first(k for (k, c) in n_faces if c == 2)
    both = build_wall_diffusion_matrix(
        G, D, Z, D;
        faces = faces, v_absorb = fill(400.0, length(faces))
    )
    one_face = 400.0 * first(f.area_per_volume for f in faces if f.nid == corner)
    @test plain[corner, corner] - both[corner, corner] > 1.5 * one_face
end
