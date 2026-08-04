# Does the existing secondary-electron source actually put electrons in the plasma?
#
# `treat_ion_outside_wall!` adds `γ_2nd · n_i` to the cells the wall mask calls
# OUTSIDE, and hopes diffusion carries them back in. The source already flags the
# intent: *"needs to improve this part (somehow this should generate them inside
# wall)"*. These tests measure what it does instead, so the defect is pinned in
# the suite rather than only in a design note, and so the replacement path
# (`wall_emission_source`) has something concrete to beat.
#
# Every `@test_broken` here states the INTENDED behaviour. Julia turns an
# unexpected pass into an error, so whoever fixes the source is told to come back
# and delete the marker rather than discovering it silently drifted.

@testitem "Secondary electrons are deposited where the plasma is not" begin
    using RAPID2D: treat_ion_outside_wall!, is_in_wall

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 41, NZ = 41,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-7,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    RP.flags.update_ni_independently = true
    RP.flags.secondary_electron = true
    RP.flags.γ_2nd_electron = 0.1

    G = RP.G
    band = G.nodes.on_out_wall_nids
    V = vec(2π .* G.Jacob .* G.dR .* G.dZ)
    inw = [is_in_wall(G, G.nodes.rid[k], G.nodes.zid[k]) for k in 1:(G.NR * G.NZ)]

    RP.plasma.ne .= 0.0
    RP.plasma.ni .= 0.0
    RP.plasma.ni[band] .= 1.0e16
    N_impact = sum(1.0e16 .* V[band])

    treat_ion_outside_wall!(RP)
    ne = vec(RP.plasma.ne)
    N_made = sum(ne .* V)
    N_inside = sum(ne[inw] .* V[inw])

    # the count is right — γ · (what hit the wall) electrons are created
    @test N_made ≈ 0.1 * N_impact rtol = 1.0e-12

    # …but not one of them is in the plasma. A source that lands entirely on
    # nodes the transport operator does not own is not a source.
    @test N_inside == 0.0
    @test_broken N_inside / N_made > 0.5          # INTENDED: they belong inside

    # every single one sits on a node the wall mask excludes
    @test sum(ne[.!inw] .* V[.!inw]) ≈ N_made rtol = 1.0e-12
end

@testitem "Secondary electrons are booked as electron loss on the next step" begin
    using RAPID2D: treat_ion_outside_wall!, treat_electron_outside_wall!

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 41, NZ = 41,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-7,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    RP.flags.update_ni_independently = true
    RP.flags.secondary_electron = true
    RP.flags.γ_2nd_electron = 0.1

    G = RP.G
    V = vec(2π .* G.Jacob .* G.dR .* G.dZ)
    RP.plasma.ne .= 0.0
    RP.plasma.ni .= 0.0
    RP.plasma.ni[G.nodes.on_out_wall_nids] .= 1.0e16

    treat_ion_outside_wall!(RP)
    N_made = sum(vec(RP.plasma.ne) .* V)
    loss_before = RP.diagnostics.Ntracker.cum0D_Ne_loss

    # the top of the very next step, with no transport in between. The pass books
    # ionization from the published rates, so stand the producer up — nothing is
    # ionizing here, which is what this test wants.
    RAPID2D.update_reaction_counts!(RP)
    treat_electron_outside_wall!(RP)
    Δloss = RP.diagnostics.Ntracker.cum0D_Ne_loss - loss_before

    # they are gone, and — worse than a no-op — the particle balance now records
    # them as electrons the wall TOOK. Creating a particle inflates the loss.
    @test sum(vec(RP.plasma.ne) .* V) == 0.0
    @test Δloss ≈ N_made rtol = 1.0e-12
    @test_broken Δloss < 0.01 * N_made            # INTENDED: creation is not loss
end

@testitem "Turning secondary electrons on moves the loss diagnostic, not the plasma" begin
    using RAPID2D: is_in_wall

    # The integration statement. Two identical runs differing only in
    # `secondary_electron`; whatever γ does must show up in one of these.
    function run_with(sec::Bool; γ = 0.5)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = 31, NZ = 31,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-7,
            t_end_s = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        RP.flags.update_ni_independently = true
        RP.flags.secondary_electron = sec
        RP.flags.γ_2nd_electron = γ
        RP.plasma.ne .= 1.0e15
        RP.plasma.ni .= 1.0e15
        RP.plasma.Te_eV .= 5.0
        run_simulation!(RP)

        G = RP.G
        V = vec(2π .* G.Jacob .* G.dR .* G.dZ)
        inw = [is_in_wall(G, G.nodes.rid[k], G.nodes.zid[k]) for k in 1:(G.NR * G.NZ)]
        return (
            ne_loss = RP.diagnostics.Ntracker.cum0D_Ne_loss,
            ni_loss = RP.diagnostics.Ntracker.cum0D_Ni_loss,
            inside = sum(vec(RP.plasma.ne)[inw] .* V[inw]),
        )
    end

    γ = 0.5
    off = run_with(false; γ = γ)
    on = run_with(true; γ = γ)
    intended = γ * on.ni_loss              # electrons γ was asked to return

    # `intended` is essentially the same target in both runs. Not to machine
    # precision any more: the ion continuity equation takes `ne·ν_iz` as its
    # source, so a flag that changes `ne` now reaches `ni` too — 1.7e-8 of it.
    # When `ni` was frozen this was exactly zero, which is what the old rtol of
    # 1e-12 was really measuring.
    @test on.ni_loss ≈ off.ni_loss rtol = 1.0e-6

    # what γ actually bought: the electron LOSS grew by essentially the whole
    # intended amount. Measured shortfall 0.39 %, which is the defect's own
    # bookkeeping and not the yield — `treat_ion_outside_wall!` deposits on
    # out-of-wall nodes while `treat_electron_outside_wall!` books the on-or-out
    # set, and the two do not coincide.
    @test (on.ne_loss - off.ne_loss) ≈ intended rtol = 0.01

    # and the plasma gained 0.014 % of it — the effective yield is not γ = 0.5
    # but ≈ 7e-5, set by D⊥Δt/Δx² rather than by any surface property
    @test (on.inside - off.inside) / intended < 1.0e-3
    @test_broken (on.inside - off.inside) ≈ intended rtol = 0.5     # INTENDED
end

@testitem "Secondary electrons are unreachable when ions are slaved to electrons" begin
    # `workflows.jl` gates the ONLY secondary-electron code behind an unrelated
    # flag: `if RP.flags.update_ni_independently  treat_ion_outside_wall!(RP)`.
    # So `secondary_electron = true` is silently inert for slaved ions, and the
    # ion wall loss is not booked either.
    function run_slaved(sec::Bool)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = 31, NZ = 31,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-7,
            t_end_s = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        RP.flags.update_ni_independently = false
        RP.flags.secondary_electron = sec
        RP.flags.γ_2nd_electron = 0.5
        RP.plasma.ne .= 1.0e15
        RP.plasma.ni .= 1.0e15
        RP.plasma.Te_eV .= 5.0
        run_simulation!(RP)
        return RP.diagnostics.Ntracker
    end

    off = run_slaved(false)
    on = run_slaved(true)

    # bit-identical: the flag has no effect at all in this configuration
    @test on.cum0D_Ne_loss == off.cum0D_Ne_loss
    @test on.cum0D_Ne_src == off.cum0D_Ne_src
    # and nothing books the ions that reached the wall
    @test on.cum0D_Ni_loss == 0.0
    @test_broken on.cum0D_Ne_loss != off.cum0D_Ne_loss     # INTENDED: γ should act
end

@testitem "The wall-emission path returns particles the secondary path loses" begin
    using RAPID2D: wall_faces, WallLedger, accumulate_wall_absorption!,
        wall_emission_source, treat_electron_outside_wall!, is_in_wall

    # Positive control. Same geometry, same γ, same particles crossing the wall —
    # routed through `wall_emission_source` instead of a deposit outside.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 41, NZ = 41,
        R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
        wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-7,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    RP.flags.update_ni_independently = true

    G = RP.G
    dt = 1.0e-7
    γ = 0.1
    faces = wall_faces(G)
    V = vec(2π .* G.Jacob .* G.dR .* G.dZ)
    inw = [is_in_wall(G, G.nodes.rid[k], G.nodes.zid[k]) for k in 1:(G.NR * G.NZ)]

    # ions absorbed at the wall this step
    ni = fill(1.0e16, G.NR, G.NZ)
    v_absorb = fill(300.0, length(faces))
    led = WallLedger{Float64}(length(faces))
    accumulate_wall_absorption!(led, faces, v_absorb, ni, dt)
    N_absorbed = sum(led.absorbed)
    @test N_absorbed > 0

    src = wall_emission_source(G, faces, γ .* led.absorbed, dt)
    RP.plasma.ne .= 0.0
    RP.plasma.ne .+= reshape(src, G.NR, G.NZ) .* dt

    ne = vec(RP.plasma.ne)
    N_returned = sum(ne .* V)

    # the count matches the secondary path's…
    @test N_returned ≈ γ * N_absorbed rtol = 1.0e-12
    # …but every particle is on an interior node
    @test sum(ne[inw] .* V[inw]) ≈ N_returned rtol = 1.0e-12
    @test sum(ne[.!inw] .* V[.!inw]) == 0.0

    # and the next step's boundary pass leaves them alone instead of booking them
    loss_before = RP.diagnostics.Ntracker.cum0D_Ne_loss
    RAPID2D.update_reaction_counts!(RP)
    treat_electron_outside_wall!(RP)
    @test RP.diagnostics.Ntracker.cum0D_Ne_loss == loss_before
    @test sum(vec(RP.plasma.ne) .* V) ≈ N_returned rtol = 1.0e-12
end
