# The ion continuity equation, as the workflow actually calls it.
#
# On master `solve_ion_continuity_equation!` warned once and returned, so `ni`
# was frozen whenever `update_ni_independently` was true and slaved to `ne/Zeff`
# otherwise. Nothing else in the code knows that: `ν_ei` is built from `ni`
# (`initialization.jl`), so a frozen `ni` quietly corrupts the Coulomb collision
# frequency as soon as `ne` moves.
#
# These tests are about the wiring — that the species axis, the policy and the
# wall reach the real solver. The mixing physics is pinned in
# `ion_transport_test.jl`, the operator and batch solve in
# `ion_transport_solve_test.jl`.

@testsnippet IonRun begin
    "A discharge-shaped setup with a box wall and ions free to evolve."
    function ion_case(; NR = 25, NZ = 25, ne = 1.0e15, Te = 5.0, kw...)
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
        RP.plasma.ne .= ne
        RP.plasma.ni .= ne
        RP.plasma.Te_eV .= Te
        RP.plasma.Ti_eV .= 1.0
        return RP
    end

    "Σ J·n over in-wall nodes — the measure the operator conserves."
    inventory(RP, n) = sum(vec(RP.G.Jacob)[RP.G.nodes.in_wall_nids] .* vec(n)[RP.G.nodes.in_wall_nids])
end

@testitem "The ion continuity equation moves ni" setup = [IonRun] begin
    RP = ion_case()
    update_transport_quantities!(RP)
    before = copy(RP.plasma.ni)

    solve_ion_continuity_equation!(RP)

    @test RP.plasma.ni != before
    @test all(isfinite, RP.plasma.ni)
    @test all(>=(0), RP.plasma.ni)
end

@testitem "Slaving ions to electrons is untouched" setup = [IonRun] begin
    # The default path in every existing test. Whatever the ion solver does, this
    # branch must keep behaving exactly as it did, or the whole suite moves.
    #
    # What it does is `ni = ne`, NOT `ne/Zeff`. Two places slave ions to
    # electrons and they disagree: `workflows.jl` writes `ni .= ne ./ Zeff`
    # inside the step, then `treat_electron_outside_wall!` overwrites it with
    # `ni .= ne` after the step. The last write wins, so the Zeff-aware one is
    # dead code whenever the step is followed by the boundary pass — i.e. always,
    # in `run_simulation!`. Pinned here rather than fixed: changing which one
    # wins is a physics decision about what Zeff means, and it moves every
    # existing result.
    RP = ion_case()
    RP.flags.update_ni_independently = false
    RP.plasma.Zeff .= 1.3
    run_simulation!(RP)

    @test RP.plasma.ni == RP.plasma.ne
    @test !(RP.plasma.ni ≈ RP.plasma.ne ./ RP.plasma.Zeff)
end

@testitem "Ionization enters the ion equation at the electron rate" setup = [IonRun] begin
    # Ni_iz = Ne_iz: each ionization makes one ion and one electron, and the rate
    # is set by the ELECTRON density, so for ions it is a pure explicit source.
    RP = ion_case()
    RP.flags.diffu = false
    RP.flags.convec = false
    RP.flags.src = true
    update_transport_quantities!(RP)

    before = copy(RP.plasma.ni)
    expected = @. before + RP.dt * RP.plasma.ne * RP.plasma.ν_en_iz
    solve_ion_continuity_equation!(RP)

    inw = RP.G.nodes.in_wall_nids
    @test RP.plasma.ni[inw] ≈ expected[inw] rtol = 1.0e-12
    @test any(>(0), RP.plasma.ν_en_iz[inw])      # the source is not silently zero
end

@testitem "Turning the source off leaves the ion source out" setup = [IonRun] begin
    RP = ion_case()
    RP.flags.diffu = false
    RP.flags.convec = false
    RP.flags.src = false
    update_transport_quantities!(RP)

    before = copy(RP.plasma.ni)
    solve_ion_continuity_equation!(RP)
    @test RP.plasma.ni == before
end

@testitem "The transport policy is a type, and one species makes them agree" setup = [IonRun] begin
    using RAPID2D: SharedEffectiveTransport, PerSpeciesTransport, IonTransportPolicy

    # H2+ is the only ion species today, so the two policies must be
    # indistinguishable. That is what allows the default to be chosen on physical
    # grounds now and revisited when a second species is appended.
    @test RAPID2D.SimulationFlags{Float64}().ion_transport_policy isa IonTransportPolicy

    function final_ni(policy)
        RP = ion_case()
        RP.flags.ion_transport_policy = policy
        run_simulation!(RP)
        return copy(RP.plasma.ni)
    end

    @test final_ni(SharedEffectiveTransport()) == final_ni(PerSpeciesTransport())
end

@testitem "Ions never appear outside the wall" setup = [IonRun] begin
    RP = ion_case()
    run_simulation!(RP)

    outside = setdiff(1:(RP.G.NR * RP.G.NZ), RP.G.nodes.in_wall_nids)
    @test all(==(0.0), vec(RP.plasma.ni)[outside])
end

@testitem "A reflecting ion wall conserves ions; an absorbing one does not" setup = [IonRun] begin
    # The Robin condition's first appearance in a production solve. With the
    # source off, ion number is exactly conserved at albedo 1 and only decreases
    # at albedo 0.
    function surviving(albedo)
        RP = ion_case(; ion_wall_albedo = albedo)
        RP.flags.src = false
        RP.flags.convec = false
        update_transport_quantities!(RP)
        start = inventory(RP, RP.plasma.ni)
        for _ in 1:50
            solve_ion_continuity_equation!(RP)
        end
        return inventory(RP, RP.plasma.ni) / start
    end

    fracs = surviving.((1.0, 0.9, 0.0))
    @test fracs[1] ≈ 1.0 rtol = 1.0e-11
    @test issorted(fracs; rev = true)
    @test fracs[end] < 1.0
end

@testitem "Ions absorbed at the wall are booked as ion loss" setup = [IonRun] begin
    # The wall-aware operator never writes outside the wall, so the old
    # `treat_ion_outside_wall!` accounting -- which reads the density it finds on
    # out-of-wall nodes -- would silently report zero loss. The Robin debit has
    # to be booked where it is taken: at the face.
    RP = ion_case(; ion_wall_albedo = 0.0)
    RP.flags.src = false
    RP.flags.convec = false
    update_transport_quantities!(RP)

    start = inventory(RP, RP.plasma.ni)
    RP.diagnostics.Ntracker.cum0D_Ni_loss = 0.0
    for _ in 1:50
        solve_ion_continuity_equation!(RP)
    end
    lost = start - inventory(RP, RP.plasma.ni)

    # Σ J·n is 1/2π of a particle count; the ledger books true particles
    @test RP.diagnostics.Ntracker.cum0D_Ni_loss > 0
    @test RP.diagnostics.Ntracker.cum0D_Ni_loss ≈ 2π * lost * RP.G.dR * RP.G.dZ rtol = 1.0e-6
end

@testitem "Diffusion off means the ion operator is not assembled" setup = [IonRun] begin
    # `flags.diffu` has to reach the ion equation too, or a run configured for
    # source-only physics silently gains transport.
    RP = ion_case()
    RP.flags.diffu = false
    RP.flags.convec = false
    RP.flags.src = false
    update_transport_quantities!(RP)

    RP.plasma.ni[RP.G.nodes.in_wall_nids] .= 1.0e15 .* rand(length(RP.G.nodes.in_wall_nids))
    before = copy(RP.plasma.ni)
    for _ in 1:5
        solve_ion_continuity_equation!(RP)
    end
    @test RP.plasma.ni == before
end
