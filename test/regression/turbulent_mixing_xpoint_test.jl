# Mixing along the field lines of an X-point: u∥ and Te homogenize along each line and keep
# what the line's particles carried.
#
# The hyperbolic field ψ = (B′/2)(x² − y²) has four families of lines, each leaving the box
# through the wall. A prescribed tensor D b_pol b_polᵀ mixes along them only in the continuum;
# on the grid the 9-point stencil also diffuses along-line structure across the lines, an
# error that falls with refinement. So the checks are: the global particle momentum (and
# energy) is kept, the tubes' deviation from their own means falls with the grid, and every
# tube heats by the shear it erased. The energy checks are broken until Phase 4 lands.
# notes/design/turbulent-mixing-u-T.md §6.2; PLAN_turbulent-mixing-u-T.md Phase 2b.

@testsnippet XPointMixing begin
    using RAPID2D: XPointPoloidal

    const XP = (R0 = 1.5, Z0 = 0.0, Bprime = 0.02)

    "ψ of the X-point field, and χ = x y, which runs along each line, on every node."
    function xpoint_coordinates(G)
        x = vec(G.R2D) .- XP.R0
        y = vec(G.Z2D) .- XP.Z0
        return (ψ = XP.Bprime / 2 .* (x .^ 2 .- y .^ 2), χ = x .* y, x = x, y = y)
    end

    """
    In-wall nodes grouped into tubes: a ψ interval on one branch (sign of x for ψ > 0, of y
    for ψ < 0), over the inner half of the ψ range. Lines within `ψ_cut` of the null are left
    out, since the tensor vanishes there; the outer half is left out because its short,
    strongly curved corner lines leak across ψ several times more than the long ones
    (measured 2–4 % of u0 against 0.1–0.9 %). Returns `Dict((bin, branch) => nodes)`.
    """
    function flux_tubes(G, co; ψ_cut, n_bins = 3)
        inw = G.nodes.in_wall_nids
        ψ_max = XP.Bprime / 2 * 0.3^2         # the box half-width: the same tubes on every grid
        edges = range(ψ_cut, ψ_max / 2; length = n_bins + 1)
        tubes = Dict{Tuple{Int, Int}, Vector{Int}}()
        for nid in inw
            a = abs(co.ψ[nid])
            edges[1] <= a < edges[end] || continue
            bin = searchsortedlast(edges, a)
            branch = co.ψ[nid] > 0 ? sign(co.x[nid]) : sign(co.y[nid])
            branch == 0 && continue
            push!(get!(tubes, (bin, Int(branch)), Int[]), nid)
        end
        return tubes
    end

    "Particle-weighted mean of f over `nodes`."
    tube_mean(nodes, V, n, f) = sum(V[nodes] .* n[nodes] .* f[nodes]) / sum(V[nodes] .* n[nodes])

    "Particle-weighted spread of f over `nodes`: how far from uniform along the tube."
    function tube_spread(nodes, V, n, f)
        m = tube_mean(nodes, V, n, f)
        w = V[nodes] .* n[nodes]
        return sqrt(sum(w .* (f[nodes] .- m) .^ 2) / sum(w))
    end
end

@testitem "mixing along X-point field lines: the particle momentum is kept, tube deviations fall with the grid, the shear heats" tags = [:regression] setup = [PureMixingRun, XPointMixing] begin
    # Measured (33² and 49², 250 steps). The volume-weighted operator loses 9.2 % of the
    # global particle momentum on both grids; the particle-weighted one keeps it to 0.25 %,
    # the O(Δt) of the split between the continuity and the momentum solve, on both grids.
    # Tube by tube, the particle-weighted operator's deviation from (6.1) falls with
    # refinement (3 → 0.3 %, 10 → 8 %, 5 → 1.5 % of u0): what remains is the 9-point stencil's
    # cross-line diffusion of along-line structure, ∝ D (k h)². The volume-weighted one's
    # deviation grows with refinement (8 → 12 %, 1 → 4 %, 7 → 10 %), since it converges to the
    # wrong limit. (6.2) predicts 0.65–0.9 eV of heating in every tube.
    D = 500.0
    t_end = 2.5e-4                            # ≈ 2 crossing times of the longest inner line
    u0, T0 = 2.0e6, 10.0
    me, ee = 9.1093837015e-31, 1.602176634e-19
    deviation = Dict{Int, Dict{Tuple{Int, Int}, Float64}}()
    heats = Dict{Int, Vector{Bool}}()
    energy_kept = Dict{Int, Bool}()
    for N in (33, 49)
        # n and u correlated along the line (two oscillations in χ across the box), uniform
        # across it; T uniform
        RP = pure_mixing_RP(; NR = N, NZ = N, t_end_s = t_end, D_along = D, poloidal = XPointPoloidal(; XP...))
        G, pla = RP.G, RP.plasma
        inw = G.nodes.in_wall_nids
        co = xpoint_coordinates(G)
        k = 2π / 0.09
        along = @. 1 + 0.5 * cos(k * co.χ)
        pla.ne .= 0.0
        pla.ne[inw] .= 1.0e14 .* along[inw]
        pla.ue_para .= reshape(u0 .* along, G.NR, G.NZ)
        pla.Te_eV .= T0
        V = vec(G.inVol2D)
        n_i, u_i, T_i = vec(copy(pla.ne)), vec(copy(pla.ue_para)), vec(copy(pla.Te_eV))
        ψ_cut = XP.Bprime / 2 * (3 / 32)^2    # three cells of the coarse grid from the null
        tubes = flux_tubes(G, co; ψ_cut)
        @test length(tubes) == 6
        expected = Dict(key => mixed_state(nodes, V, n_i, u_i, T_i) for (key, nodes) in tubes)
        energy(n, u, T) = sum(V[inw] .* n[inw] .* (1.5 .* ee .* T[inw] .+ 0.5 .* me .* u[inw] .^ 2))
        P_i, E_i = sum(V[inw] .* n_i[inw] .* u_i[inw]), energy(n_i, u_i, T_i)
        @test run_pure_mixing!(RP)
        n_f, u_f, T_f = vec(pla.ne), vec(pla.ue_para), vec(pla.Te_eV)
        @test all(isfinite, u_f[inw]) && all(isfinite, T_f[inw])
        # the particles' momentum is kept, up to the O(Δt) of the split; the volume-weighted
        # operator loses 9 %
        @test abs(sum(V[inw] .* n_f[inw] .* u_f[inw]) - P_i) <= 1.0e-2 * abs(P_i)
        for (key, nodes) in tubes
            # homogenized along the line
            @test tube_spread(nodes, V, n_f, u_f) < 0.3 * tube_spread(nodes, V, n_i, u_i)
            @test expected[key].T > T0 + 0.5
        end
        deviation[N] = Dict(key => abs(tube_mean(nodes, V, n_f, u_f) - expected[key].u) for (key, nodes) in tubes)
        # the erased shear heats every tube (6.2): 0.65–0.9 eV, against none today
        heats[N] = [abs(tube_mean(nodes, V, n_f, T_f) - expected[key].T) <= 1.0e-1 for (key, nodes) in tubes]
        # and the particles' energy is kept once the heating credits what the shear lost
        energy_kept[N] = abs(energy(n_f, u_f, T_f) - E_i) <= 1.0e-2 * E_i
    end
    # the tubes' deviation from (6.1) is a discretization error: it falls with refinement,
    # tube by tube (the volume-weighted operator's grows)
    for key in keys(deviation[33])
        @test deviation[49][key] < deviation[33][key]
    end
    @test_broken all(heats[33]) && all(heats[49])
    @test_broken energy_kept[33] && energy_kept[49]
end
