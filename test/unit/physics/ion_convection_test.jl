# Ions convect with the ION velocity.
#
# This is the term that makes `ni` a different field from `ne` rather than a
# scaled copy: under an applied E the electron and ion parallel drifts point
# opposite ways, so the two densities separate before any diffusion has had time
# to act. Everything about quasi-neutrality later — that it is *broken* early and
# recovered once the turbulent D switches on — depends on this term existing and
# on it reading `uiR`/`uiZ` rather than `ueR`/`ueZ`.
#
# The wall treatment here is the one electrons already have: `∇·(n𝐮)` sweeps the
# interior with no wall awareness, material convects onto out-of-wall nodes, and
# `treat_ion_outside_wall!` zeroes and books it. A wall-aware convective flux is
# its own piece of work; mixing a Robin diffusive wall with an outflow convective
# wall is not an inconsistency, since the two channels deliver to a surface by
# genuinely different mechanisms.

@testsnippet IonDrift begin
    using RAPID2D: update_transport_related_operators!, treat_ion_outside_wall!

    "A case with prescribed, uniform parallel velocities and no field solve."
    function drift_case(; ui = 3.0e4, ue = -3.0e4, NR = 25, NZ = 25)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = NR, NZ = NZ,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8,
            t_end_s = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
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
        update_transport_related_operators!(RP)
        return RP
    end

    "Density-weighted Z of the in-wall plasma."
    function centroid_Z(RP, n)
        inw = RP.G.nodes.in_wall_nids
        w = vec(RP.G.Jacob)[inw] .* vec(n)[inw]
        return sum(w .* vec(RP.G.Z2D)[inw]) / sum(w)
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
    # The bug this exists to catch: `update_∇𝐮_operator!` defaults to `ueR`/`ueZ`,
    # so an ion operator built without explicit velocities compiles, runs, and
    # silently drifts the ions the wrong way.
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

@testitem "The ion operator is the diffusion operator minus the ion flux divergence" setup = [IonDrift] begin
    using RAPID2D: wall_faces, ion_transport_operator, ion_transport_groups,
        ion_transport_channels, ion_channel_directions, shared_turbulent_channel,
        solve_ion_group!, SparseLUSolver
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
    A = A_diff - RP.operators.∇𝐮_i.matrix
    N = reshape(copy(vec(RP.plasma.ni)), :, 1)
    solve_ion_group!(N, groups[1], A, SparseLUSolver{Float64}(), RP.dt; θ = RP.flags.Implicit_weight)

    solve_ion_continuity_equation!(RP)
    @test vec(RP.plasma.ni) ≈ N[:, 1]

    # and the ion flux-divergence operator is genuinely a different matrix from
    # the electron one, because the velocities differ
    @test RP.operators.∇𝐮_i.matrix != RP.operators.∇𝐮.matrix
end

@testitem "Convected ions leaving the wall are booked, not lost silently" setup = [IonDrift] begin
    # The Robin debit books what DIFFUSION takes. Convection puts material on
    # out-of-wall nodes instead, where `treat_ion_outside_wall!` books it. The two
    # paths must not overlap and must not leave a gap.
    RP = drift_case(; ui = 3.0e4)
    RP.flags.src = false
    RP.flags.diffu = false
    inw = RP.G.nodes.in_wall_nids
    outside = setdiff(1:(RP.G.NR * RP.G.NZ), inw)

    for _ in 1:20
        solve_ion_continuity_equation!(RP)
    end
    # convection alone does deposit outside — that is what the boundary pass is for
    @test any(>(0), vec(RP.plasma.ni)[outside])

    RP.diagnostics.Ntracker.cum0D_Ni_loss = 0.0
    treat_ion_outside_wall!(RP)
    @test RP.diagnostics.Ntracker.cum0D_Ni_loss > 0
    @test all(==(0.0), vec(RP.plasma.ni)[outside])
end
