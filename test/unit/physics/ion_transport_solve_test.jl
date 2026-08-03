# From a group of ion species to an advanced density.
#
# A group names the species that share an operator; this layer builds that
# operator and steps every member of the group through it with one
# factorization. The tensor and the wall ceiling must come from the SAME channel
# list — assembling the matrix from one set and the boundary condition from
# another is the failure this layer exists to make impossible.

@testsnippet IonGrid begin
    "A box wall inside the grid, so there are faces on all four sides."
    function boxed_grid(; NR = 31, NZ = 31)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = NR, NZ = NZ,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        return RP.G
    end

    "A field direction that varies over the grid, so no two faces agree by accident."
    function oblique_b(G)
        α = @. 0.4 + 0.3 * G.R2D + 0.7 * G.Z2D
        return (cos.(α), sin.(α))
    end

    "A channel with both a parallel and a cross-field leg, on the grid's shape."
    function both_legs(G; v_para = 3.0e4, λ_para = 0.02, v_perp = 2.0e3, λ_perp = 0.3)
        sz = (G.NR, G.NZ)
        return RAPID2D.DiffusionChannel(
            fill(v_para, sz), fill(λ_para, sz), fill(v_perp, sz), fill(λ_perp, sz)
        )
    end
end

@testitem "Absorption speed is a face property, sampled per face" setup = [IonGrid] begin
    using RAPID2D: DiffusionChannel, wall_faces, wall_absorption_speeds, channel_ceiling

    G = boxed_grid()
    faces = wall_faces(G)
    bR, bZ = oblique_b(G)

    # two mechanisms on two different axes, as a real ion has: collisional along
    # b̂, turbulent along b̂_pol
    ch1 = DiffusionChannel(fill(3.0e4, G.NR, G.NZ), fill(0.02, G.NR, G.NZ), zeros(G.NR, G.NZ), zeros(G.NR, G.NZ))
    ch2 = DiffusionChannel(zeros(G.NR, G.NZ), zeros(G.NR, G.NZ), fill(2.0e3, G.NR, G.NZ), fill(0.3, G.NR, G.NZ))
    cwd = [(ch1, bR, bZ), (ch2, ones(G.NR, G.NZ), zeros(G.NR, G.NZ))]

    v = wall_absorption_speeds(cwd, faces, 0.0)

    # against the definition, face by face — this is what the direction-cached
    # implementation must reproduce, including the face ↔ entry ordering
    ref = [
        sum(channel_ceiling(ch, br, bz, f.outward)[f.nid] for (ch, br, bz) in cwd)
            for f in faces
    ]
    @test v ≈ ref

    # and the values are genuinely distinct, so the comparison above is not
    # satisfied by any permutation of a constant
    @test length(unique(round.(v; digits = 6))) > 10
    @test all(>(0), v)
end

@testitem "A corner cell absorbs differently through each of its two faces" setup = [IonGrid] begin
    using RAPID2D: DiffusionChannel, wall_faces, wall_absorption_speeds

    G = boxed_grid()
    faces = wall_faces(G)
    bR, bZ = oblique_b(G)
    ch = both_legs(G)
    v = wall_absorption_speeds([(ch, bR, bZ)], faces, 0.0)

    # a staircase corner owns an R-face and a Z-face; b̂·n̂ differs between them,
    # so a per-CELL absorption speed would be wrong on at least one
    owners = Dict{Int, Vector{Int}}()
    for (k, f) in enumerate(faces)
        push!(get!(owners, f.nid, Int[]), k)
    end
    corners = [ks for ks in values(owners) if length(ks) == 2]
    @test !isempty(corners)
    @test all(ks -> v[ks[1]] != v[ks[2]], corners)
end

@testitem "The albedo scales the speed, and R = 1 restores a reflective wall" setup = [IonGrid] begin
    using RAPID2D: DiffusionChannel, wall_faces, wall_absorption_speeds

    G = boxed_grid()
    faces = wall_faces(G)
    bR, bZ = oblique_b(G)
    ch = both_legs(G)
    cwd = [(ch, bR, bZ)]

    gross = wall_absorption_speeds(cwd, faces, 0.0)
    @test wall_absorption_speeds(cwd, faces, 1.0) == zeros(length(faces))
    @test wall_absorption_speeds(cwd, faces, 0.5) ≈ 0.5 .* gross

    # a per-face albedo is the point of the vector form: a divertor tile and a
    # main-chamber tile do not recycle alike
    R_face = range(0.0, 1.0; length = length(faces))
    @test wall_absorption_speeds(cwd, faces, collect(R_face)) ≈ (1 .- R_face) .* gross

    @test_throws ArgumentError wall_absorption_speeds(cwd, faces, 1.5)
    @test_throws DimensionMismatch wall_absorption_speeds(cwd, faces, zeros(3))
end

@testitem "One operator is built from one channel list" setup = [IonGrid] begin
    using RAPID2D: DiffusionChannel, IonTransportGroup, wall_faces,
        ion_transport_operator, build_wall_diffusion_matrix, total_tensor,
        wall_absorption_speeds

    G = boxed_grid()
    faces = wall_faces(G)
    bR, bZ = oblique_b(G)
    ch1 = DiffusionChannel(fill(3.0e4, G.NR, G.NZ), fill(0.02, G.NR, G.NZ), zeros(G.NR, G.NZ), zeros(G.NR, G.NZ))
    ch2 = DiffusionChannel(zeros(G.NR, G.NZ), zeros(G.NR, G.NZ), fill(2.0e3, G.NR, G.NZ), fill(0.3, G.NR, G.NZ))
    group = IonTransportGroup([1, 2], [ch1, ch2])
    dirs = [(bR, bZ), (ones(G.NR, G.NZ), zeros(G.NR, G.NZ))]

    A, v_abs = ion_transport_operator(G, group, dirs; faces = faces, albedo = 0.3)

    cwd = [(group.channels[m], dirs[m]...) for m in 1:2]
    D_RR, D_RZ, D_ZZ = total_tensor(cwd)
    @test A == build_wall_diffusion_matrix(
        G, D_RR, D_RZ, D_ZZ;
        faces = faces, v_absorb = wall_absorption_speeds(cwd, faces, 0.3)
    )
    @test v_abs ≈ wall_absorption_speeds(cwd, faces, 0.3)

    # with no wall given, the operator is the reflective one
    A_refl, v_none = ion_transport_operator(G, group, dirs)
    @test A_refl == build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ)
    @test isempty(v_none)

    @test_throws ArgumentError ion_transport_operator(G, group, dirs[1:1])
end

@testitem "A batch solve is the sequential solve, one factorization apart" setup = [IonGrid] begin
    using RAPID2D: DiffusionChannel, IonTransportGroup, wall_faces,
        ion_transport_operator, solve_ion_group!, SparseLUSolver, factorize!, solve!
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    G = boxed_grid()
    faces = wall_faces(G)
    bR, bZ = oblique_b(G)
    ch = both_legs(G)
    group = IonTransportGroup([1, 2, 3], [ch])
    A, v_abs = ion_transport_operator(G, group, [(bR, bZ)]; faces = faces, albedo = 0.4)

    Ng = G.NR * G.NZ
    inw = G.nodes.in_wall_nids
    N = zeros(Ng, 3)
    N[inw, 1] .= 1.0e16
    N[inw, 2] .= 3.0e15 .* (1 .+ vec(G.Z2D)[inw])
    N[inw, 3] .= 7.0e14
    S = zeros(Ng, 3)
    S[inw, 2] .= 1.0e18
    dt, θ = 1.0e-5, 0.7

    batched = copy(N)
    solve_ion_group!(batched, group, A, SparseLUSolver{Float64}(), dt; θ = θ, S = S)

    # the same θ-scheme, one species at a time, with no batching anywhere
    M = sparse(I, Ng, Ng) - (θ * dt) * A
    one_at_a_time = similar(N)
    for s in 1:3
        rhs = N[:, s] .+ ((1 - θ) * dt) .* (A * N[:, s]) .+ dt .* S[:, s]
        one_at_a_time[:, s] = M \ rhs
    end

    @test batched ≈ one_at_a_time

    # Species did not leak into one another. Columns 1 and 3 start uniform with
    # no source, so linearity forces them to stay proportional at their initial
    # ratio — any cross-column contamination breaks that immediately. Column 2
    # carries a gradient and a source and must not be proportional to either.
    @test batched[inw, 1] ≈ (1.0e16 / 7.0e14) .* batched[inw, 3]
    @test !(batched[inw, 2] ≈ (batched[inw[1], 2] / batched[inw[1], 1]) .* batched[inw, 1])
end

@testitem "A group advances only its own species" setup = [IonGrid] begin
    using RAPID2D: DiffusionChannel, IonTransportGroup, ion_transport_operator,
        solve_ion_group!, SparseLUSolver

    G = boxed_grid(; NR = 21, NZ = 21)
    bR, bZ = oblique_b(G)
    ch = both_legs(G)
    group = IonTransportGroup([1, 3], [ch])          # species 2 belongs elsewhere
    A, _ = ion_transport_operator(G, group, [(bR, bZ)])

    Ng = G.NR * G.NZ
    N = zeros(Ng, 3)
    N[G.nodes.in_wall_nids, :] .= 1.0e16
    untouched = copy(N[:, 2])

    solve_ion_group!(N, group, A, SparseLUSolver{Float64}(), 1.0e-5)

    @test N[:, 2] == untouched
    @test N[:, 1] != untouched
    @test N[:, 1] == N[:, 3]                          # same operator, same start
end

@testitem "A reflective wall conserves each ion species separately" setup = [IonGrid] begin
    using RAPID2D: DiffusionChannel, IonTransportGroup, wall_faces,
        ion_transport_operator, solve_ion_group!, SparseLUSolver

    # The Robin condition's first use on ions. Two species with different
    # profiles share one operator; at R = 1 each must keep its own Σ J·n, and
    # they must not converge to a common profile.
    G = boxed_grid()
    faces = wall_faces(G)
    bR, bZ = oblique_b(G)
    ch = both_legs(G)
    group = IonTransportGroup([1, 2], [ch])
    A, _ = ion_transport_operator(G, group, [(bR, bZ)]; faces = faces, albedo = 1.0)

    Ng = G.NR * G.NZ
    inw = G.nodes.in_wall_nids
    Jv = vec(G.Jacob)
    N = zeros(Ng, 2)
    N[inw, 1] .= 1.0e16
    N[inw, 2] .= 1.0e16 .* (1 .+ 0.5 .* sin.(10 .* vec(G.Z2D)[inw]))
    inventory(N) = (sum(Jv .* N[:, 1]), sum(Jv .* N[:, 2]))
    before = inventory(N)

    solver = SparseLUSolver{Float64}()
    for _ in 1:20
        solve_ion_group!(N, group, A, solver, 1.0e-5)
    end
    after = inventory(N)

    @test after[1] ≈ before[1] rtol = 1.0e-12
    @test after[2] ≈ before[2] rtol = 1.0e-12
    @test all(==(0.0), N[setdiff(1:Ng, inw), :])
    # species 2 relaxed but has not become species 1 in 20 steps
    @test !(N[inw, 2] ≈ N[inw, 1])
end

@testitem "Absorption is monotone in the albedo, for ions on a real wall" setup = [IonGrid] begin
    using RAPID2D: DiffusionChannel, IonTransportGroup, wall_faces,
        ion_transport_operator, solve_ion_group!, SparseLUSolver

    G = boxed_grid()
    faces = wall_faces(G)
    bR, bZ = oblique_b(G)
    ch = both_legs(G)
    group = IonTransportGroup([1], [ch])
    Ng = G.NR * G.NZ
    inw = G.nodes.in_wall_nids
    Jv = vec(G.Jacob)

    function remaining(albedo)
        A, _ = ion_transport_operator(G, group, [(bR, bZ)]; faces = faces, albedo = albedo)
        N = zeros(Ng, 1)
        N[inw, 1] .= 1.0e16
        start = sum(Jv .* N[:, 1])
        solver = SparseLUSolver{Float64}()
        for _ in 1:20
            solve_ion_group!(N, group, A, solver, 1.0e-6)
        end
        return sum(Jv .* N[:, 1]) / start
    end

    fracs = remaining.((1.0, 0.9, 0.5, 0.0))
    @test fracs[1] ≈ 1.0 rtol = 1.0e-12
    @test issorted(fracs; rev = true)
    @test fracs[end] < fracs[1]
    @test all(0 .<= fracs .<= 1)
end

@testitem "Species that differ only in mass separate under per-species transport" setup = [IonGrid] begin
    using RAPID2D: parallel_collisional_channel, bohm_channel, wall_faces,
        SharedEffectiveTransport, PerSpeciesTransport, ion_transport_groups,
        ion_transport_operator, solve_ion_group!, SparseLUSolver

    # The whole reason the policy exists. H₂⁺ and a 6× heavier carbon ion have
    # the same D⊥ (Bohm's mass cancels) but D∥ ∝ 1/m, so only the per-species
    # policy lets them move apart. If the shared policy reproduced that, there
    # would be nothing to choose between them.
    G = boxed_grid(; NR = 25, NZ = 25)
    faces = wall_faces(G)
    bR, bZ = oblique_b(G)
    sz = (G.NR, G.NZ)
    m_p = 1.6726e-27
    Te = fill(5.0, sz)
    Ti = fill(1.0, sz)
    Bt = fill(0.6, sz)
    ν = fill(662.0, sz)

    function channels(m)
        v_p = @. sqrt(2 * Ti * 1.602176634e-19 / m)
        return [parallel_collisional_channel(v_p, (@. 0.5 * v_p^2 / ν)), bohm_channel(Te, Bt, m)]
    end
    per_species = [channels(2m_p), channels(12m_p)]
    dirs = [(bR, bZ), (bR, bZ)]

    Ng = G.NR * G.NZ
    inw = G.nodes.in_wall_nids
    Jv = vec(G.Jacob)
    n0 = zeros(Ng, 2)
    n0[inw, :] .= 1.0e16
    w = [reshape(n0[:, 1], sz), reshape(n0[:, 2], sz)]

    function evolve(policy)
        N = copy(n0)
        for g in ion_transport_groups(policy, per_species, w)
            A, _ = ion_transport_operator(G, g, dirs; faces = faces, albedo = 0.0)
            solver = SparseLUSolver{Float64}()
            for _ in 1:20
                solve_ion_group!(N, g, A, solver, 1.0e-6)
            end
        end
        return (sum(Jv .* N[:, 1]), sum(Jv .* N[:, 2]))
    end

    per = evolve(PerSpeciesTransport())
    shared = evolve(SharedEffectiveTransport())

    # heavier ion is slower along B, so more of it survives
    @test per[2] > per[1]
    # the shared policy gives both species one answer, between the two
    @test shared[1] ≈ shared[2] rtol = 1.0e-12
    @test per[1] < shared[1] < per[2]
end
