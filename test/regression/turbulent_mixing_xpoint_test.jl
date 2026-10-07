# Mixing along the field lines of an X-point: u∥ and Te homogenize along each line and keep
# what the line's particles carried.
#
# The hyperbolic field ψ = (B′/2)(x² − y²) has four families of lines, each leaving the box
# through the wall. A prescribed tensor D b_pol b_polᵀ mixes along them only; what crosses
# the lines is the 9-point stencil's leakage, measured by a control run whose initial state
# is uniform along every line, and it shrinks with the grid. The line-wise checks are
# marked broken until Phase 3 (u∥) and Phase 4 (Te) land.
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

@testitem "mixing along X-point field lines: tubes keep their particle-weighted momentum and energy, leakage shrinks with the grid" tags = [:regression] setup = [PureMixingRun, XPointMixing] begin
    # Measured on the volume-weighted operator (33² and 49², 250 steps): the tubes' deviation
    # from (6.1) is 6–13 % of u0, the leakage 0.1–0.9 %, the along-line spread falls to ≤ 2.5 %
    # of its start, and (6.2) predicts 0.55–0.9 eV of heating where the operator gives none.
    D = 500.0
    t_end = 2.5e-4                            # ≈ 2 crossing times of the longest inner line
    u0, T0 = 2.0e6, 10.0
    results = Dict{Int, Any}()
    for N in (33, 49)
        # the main run: n and u correlated along the line (two oscillations in χ across the
        # box), uniform across it; T uniform
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
        @test run_pure_mixing!(RP)
        n_f, u_f, T_f = vec(pla.ne), vec(pla.ue_para), vec(pla.Te_eV)
        @test all(isfinite, u_f[inw]) && all(isfinite, T_f[inw])
        # the control: u varies across the lines only, so any change in a tube's momentum
        # is what the stencil leaks across ψ
        RPc = pure_mixing_RP(; NR = N, NZ = N, t_end_s = t_end, D_along = D, poloidal = XPointPoloidal(; XP...))
        plac = RPc.plasma
        plac.ne .= 0.0
        plac.ne[inw] .= 1.0e14 .* along[inw]
        plac.ue_para .= reshape(u0 .* (1 .+ 0.5 .* co.ψ ./ maximum(abs, co.ψ[inw])), G.NR, G.NZ)
        plac.Te_eV .= T0
        n_ci, u_ci = vec(copy(plac.ne)), vec(copy(plac.ue_para))
        @test run_pure_mixing!(RPc)
        n_cf, u_cf = vec(plac.ne), vec(plac.ue_para)
        leak = Dict(key => abs(tube_mean(nodes, V, n_cf, u_cf) - tube_mean(nodes, V, n_ci, u_ci)) for (key, nodes) in tubes)
        results[N] = (leak = leak, tubes = tubes)
        for (key, nodes) in tubes
            # homogenized along the line
            @test tube_spread(nodes, V, n_f, u_f) < 0.3 * tube_spread(nodes, V, n_i, u_i)
            @test expected[key].T > T0 + 0.5
        end
        # every tube keeps its particle-weighted momentum, up to what leaks across the lines
        # (one claim over the tubes: the volume-weighted operator loses 1–12 % of u0 depending on
        # how n and u happen to correlate along each line, the leakage is 0.1–1 %)
        keeps_u = [abs(tube_mean(nodes, V, n_f, u_f) - expected[key].u) <= 3 * leak[key] + 1.0e-3 * u0 for (key, nodes) in tubes]
        @test_broken all(keeps_u)
        # and the erased shear heated every tube (6.2): 0.65–0.9 eV, against none today
        heats = [abs(tube_mean(nodes, V, n_f, T_f) - expected[key].T) <= 1.0e-2 * T0 for (key, nodes) in tubes]
        @test_broken all(heats)
    end
    # the leakage is a discretization error: it falls with refinement, tube by tube
    coarse, fine = results[33], results[49]
    for key in keys(coarse.tubes)
        haskey(fine.tubes, key) || continue
        @test fine.leak[key] < coarse.leak[key]
    end
end
