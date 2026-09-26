# Ions convect with the ION velocity.
#
# This is the term that makes `ni` a different field from `ne` rather than a
# scaled copy: under an applied E the electron and ion parallel drifts point
# opposite ways, so the two densities separate before any diffusion has had time
# to act. Everything about quasi-neutrality later — that it is *broken* early and
# recovered once the turbulent D switches on — depends on this term existing and
# on it reading `uiR`/`uiZ` rather than `ueR`/`ueZ`.
#
# The wall is the one the electron continuity equation has under `:robin`:
# convection is the face-flux operator (`build_face_flux_divergence`, rows on
# in-wall nodes only), so nothing is ever written outside the wall, and the
# outflow through a wall face is a diagonal debit booked on the per-face ledger
# next to the Robin diffusive speed. There is no band to zero and no second
# bookkeeping pass — diffusion and convection reach the surface by different
# mechanisms (`¼v̄n` and `n𝐮·n̂`) and add into one coefficient per face.

@testsnippet IonDrift begin
    # Testitems import what they call in their own bodies: a name imported here reaches
    # the SNIPPET module only (`using ..IonDrift` re-exports what the snippet DEFINES).
    # Under TestItemRunner the two scopes happen to coincide; under ReTestItems (CI) not.

    "A case with prescribed, uniform parallel velocities and no field solve."
    function drift_case(; ui = 3.0e4, ue = -3.0e4, NR = 25, NZ = 25, kw...)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = NR, NZ = NZ,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8,
            t_end_s = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0;
            kw...
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        RP.flags.update_ni_independently = true
        # A blob, not a slab: ∇·(n𝐮) vanishes identically for uniform n and uniform
        # 𝐮, so a filled box would sit still no matter which velocity was used.
        blob = @. 1.0e15 * exp(-(RP.G.Z2D / 0.12)^2 - ((RP.G.R2D - 1.5) / 0.25)^2)
        RP.plasma.ne .= blob
        RP.plasma.ni .= blob
        RP.plasma.Te_eV .= 5.0
        RP.plasma.Ti_eV .= 1.0
        update_transport_quantities!(RP)

        # prescribe the drifts AFTER the transport update, so nothing overwrites them
        RP.plasma.uiR .= 0.0
        RP.plasma.uiZ .= ui
        RP.plasma.ueR .= 0.0
        RP.plasma.ueZ .= ue
        # the per-step cache was built from the velocities the transport update projected;
        # a drift prescribed by hand needs the operators rebuilt from it
        RAPID2D.cache_electron_operators!(RP)
        return RP
    end

    "Density-weighted Z of the in-wall plasma."
    function centroid_Z(RP, n)
        inw = RP.G.nodes.in_wall_nids
        w = vec(RP.G.Jacob)[inw] .* vec(n)[inw]
        return sum(w .* vec(RP.G.Z2D)[inw]) / sum(w)
    end

    "Particles inside the wall: Σ 2π·J·ΔR·ΔZ·n over in-wall nodes."
    function particles_inside(RP, n)
        inw = RP.G.nodes.in_wall_nids
        vol = vec(RP.G.Jacob) .* (2π * RP.G.dR * RP.G.dZ)
        return sum(vol[inw] .* vec(n)[inw])
    end
end

@testitem "Ions convect along their own velocity" setup = [IonDrift] begin
    RP = drift_case(; ui = 3.0e4)
    RP.flags.src = false
    RP.flags.diffu = false

    before = centroid_Z(RP, RP.plasma.ni)
    for _ in 1:20
        solve_ion_continuity_equation!(RP)
    end
    @test centroid_Z(RP, RP.plasma.ni) > before

    # and the other way round for the opposite drift
    RP2 = drift_case(; ui = -3.0e4)
    RP2.flags.src = false
    RP2.flags.diffu = false
    before2 = centroid_Z(RP2, RP2.plasma.ni)
    for _ in 1:20
        solve_ion_continuity_equation!(RP2)
    end
    @test centroid_Z(RP2, RP2.plasma.ni) < before2
end

@testitem "Ions do not convect along the ELECTRON velocity" setup = [IonDrift] begin
    # The bug this exists to catch: a face-flux operator built from the default
    # (electron) velocities compiles, runs, and silently drifts the ions the wrong way.
    RP = drift_case(; ui = 3.0e4, ue = -3.0e4)
    RP.flags.src = false
    RP.flags.diffu = false

    ni0 = centroid_Z(RP, RP.plasma.ni)
    ne0 = centroid_Z(RP, RP.plasma.ne)
    for _ in 1:20
        solve_ion_continuity_equation!(RP)
        solve_electron_continuity_equation!(RP)
    end

    # opposite drifts, opposite displacements — the signature the low-density
    # discharge validation is built on
    @test centroid_Z(RP, RP.plasma.ni) > ni0
    @test centroid_Z(RP, RP.plasma.ne) < ne0
end

@testitem "Turning convection off removes the ion convective term" setup = [IonDrift] begin
    RP = drift_case(; ui = 3.0e4)
    RP.flags.src = false
    RP.flags.diffu = false
    RP.flags.convec = false

    before = copy(RP.plasma.ni)
    for _ in 1:5
        solve_ion_continuity_equation!(RP)
    end
    @test RP.plasma.ni == before
end

@testitem "The ion operator is the diffusion operator minus the ion face-flux divergence" setup = [IonDrift] begin
    using RAPID2D: wall_faces, ion_transport_operator, ion_transport_groups,
        ion_transport_channels, ion_channel_directions, shared_turbulent_channel,
        solve_ion_group!, SparseLUSolver, build_face_flux_divergence
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    RP = drift_case(; ui = 3.0e4)
    RP.flags.src = false
    G = RP.G

    # reproduce the step by hand from the documented pieces
    turb = shared_turbulent_channel(RP)
    per_species = [ion_transport_channels(RP, sp, turb) for sp in RP.transport.ion_species]
    weights = [copy(RP.plasma.ni)]
    groups = ion_transport_groups(RP.flags.ion_transport_policy, per_species, weights)
    @test length(groups) == 1

    A_diff, _ = ion_transport_operator(
        G, groups[1], ion_channel_directions(RP);
        faces = wall_faces(G), albedo = RP.config.ion_wall_albedo
    )
    # the same face-flux divergence the electron equation uses, from the ION velocities
    A = A_diff - build_face_flux_divergence(G, RP.plasma.uiR, RP.plasma.uiZ; upwind = RP.flags.upwind)
    N = reshape(copy(vec(RP.plasma.ni)), :, 1)
    solve_ion_group!(N, groups[1], A, SparseLUSolver{Float64}(), RP.dt; θ = RP.flags.θ_imp.transport)

    solve_ion_continuity_equation!(RP)
    @test vec(RP.plasma.ni) ≈ N[:, 1]
end

@testitem "Ion convection leaves through the wall faces: nothing lands outside, the ledger books what left" setup = [IonDrift] begin
    # Convection alone, twenty implicit steps, no boundary pass in between. The
    # face-flux rows stop at the wall, so the band outside stays exactly zero, and the
    # diagonal outflow debit is the very number the ledger books — even with diffusion
    # off, when there is no Robin coefficient for it to ride on.
    RP = drift_case(; ui = 3.0e4)
    RP.flags.src = false
    RP.flags.diffu = false
    G = RP.G
    outside = setdiff(1:(G.NR * G.NZ), G.nodes.in_wall_nids)
    vec(RP.plasma.ni)[outside] .= 0.0
    N0 = particles_inside(RP, RP.plasma.ni)
    RP.diagnostics.Ntracker.cum0D_Ni_loss = 0.0

    for _ in 1:20
        solve_ion_continuity_equation!(RP)
    end

    @test all(==(0.0), vec(RP.plasma.ni)[outside])
    booked = RP.diagnostics.Ntracker.cum0D_Ni_loss
    @test booked > 0
    @test N0 - particles_inside(RP, RP.plasma.ni) ≈ booked rtol = 1.0e-10
end

@testitem "ion_wall_albedo scales the convective outflow, and R = 1 keeps every convected ion" setup = [IonDrift] begin
    using RAPID2D: wall_faces, face_outflow_speeds

    # A uniform slab under a uniform flow, one explicit step: interior faces carry the
    # same flux on both sides, so Σ J·n changes only by what crosses the downstream wall
    # faces — the gross outflow n₀·A_f·max(u·n̂, 0) per face, of which the wall keeps 1 − R.
    function one_explicit_step(albedo)
        RP = drift_case(; ui = 3.0e4, ion_wall_albedo = albedo)
        RP.flags.src = false
        RP.flags.diffu = false
        RP.flags.Implicit = false
        G = RP.G
        inw = G.nodes.in_wall_nids
        RP.plasma.ni .= 0.0
        RP.plasma.ni[inw] .= 1.0e15
        N0 = particles_inside(RP, RP.plasma.ni)
        faces = wall_faces(G)
        v_out = face_outflow_speeds(G, faces, RP.plasma.uiR, RP.plasma.uiZ)
        gross = RP.dt * sum(f.area * v_out[k] * 1.0e15 for (k, f) in enumerate(faces))
        RP.diagnostics.Ntracker.cum0D_Ni_loss = 0.0
        solve_ion_continuity_equation!(RP)
        lost = N0 - particles_inside(RP, RP.plasma.ni)
        return (; N0, lost, booked = RP.diagnostics.Ntracker.cum0D_Ni_loss, gross)
    end

    r0 = one_explicit_step(0.0)
    @test r0.gross > 0
    @test r0.lost ≈ r0.gross rtol = 1.0e-10
    @test r0.booked ≈ r0.lost rtol = 1.0e-10

    r5 = one_explicit_step(0.5)
    @test r5.lost ≈ 0.5 * r5.gross rtol = 1.0e-10
    @test r5.booked ≈ r5.lost rtol = 1.0e-10

    # R = 1: what reaches a wall face comes straight back through it — no net flux, the
    # slab piles up against the downstream wall instead of draining
    r1 = one_explicit_step(1.0)
    @test abs(r1.lost) <= 1.0e-12 * r1.N0
    @test r1.booked == 0.0
end

@testitem "an ion albedo outside [0, 1] is rejected even when only convection reaches the wall" setup = [IonDrift] begin
    # The diffusive builder validates the albedo, but it does not run with `diffu = false`;
    # an out-of-range value would otherwise turn the convective wall debit into a source.
    for bad in (1.5, -0.1)
        RP = drift_case(; ui = 3.0e4, ion_wall_albedo = bad)
        RP.flags.src = false
        RP.flags.diffu = false
        @test_throws ArgumentError solve_ion_continuity_equation!(RP)
    end
end
