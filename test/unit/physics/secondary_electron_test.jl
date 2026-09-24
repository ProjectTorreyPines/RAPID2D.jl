# Secondary electrons: what the flag does today, and what it is meant to do.
#
# The legacy source added `γ_2nd · n_i` to the cells OUTSIDE the wall and hoped
# diffusion carried them back in; the electron band pass booked and zeroed them
# first, so the yield that reached the plasma was ≈ 0 (set by D⊥Δt/Δx², not by γ).
# Since ion transport stopped writing outside the wall — Robin diffusion and
# face-flux convection are both diagonal debits on in-wall rows — that band holds
# nothing to multiply and the injection has been removed. `secondary_electron`
# is inert under BOTH electron wall modes until secondaries are emitted through
# the wall faces from the ion ledger (`wall_emission_source`, plan PR3).
#
# Every `@test_broken` here states the INTENDED behaviour. Julia turns an
# unexpected pass into an error, so whoever lands the source is told to come
# back and delete the marker rather than discovering it silently drifted.

@testitem "Secondary electrons are inert under either electron wall until the wall-face source lands" begin
    using RAPID2D: is_in_wall

    # Two identical runs differing only in `secondary_electron`, both wall channels on,
    # under each electron wall. Ions reach the wall and are booked on the face ledger;
    # nothing turns that into electrons yet, so the runs are bit-identical.
    function run_with(sec::Bool, wall::Symbol; γ = 0.5)
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
        RP.flags.electron_wall = wall
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
    for wall in (:zeroing, :robin)
        off = run_with(false, wall; γ = γ)
        on = run_with(true, wall; γ = γ)

        # the premise: ions really did reach the wall and were booked
        @test on.ni_loss > 0.0
        # …and γ changed nothing, bit for bit
        @test on.ni_loss == off.ni_loss
        @test on.inside == off.inside
        @test on.ne_loss == off.ne_loss

        # INTENDED: γ·(what hit the wall) electrons, returned to the wall-adjacent
        # INTERIOR cells through `wall_emission_source` from the ion face ledger
        @test_broken (on.inside - off.inside) ≈ γ * on.ni_loss rtol = 0.5
    end
end

@testitem "Secondary electrons are unreachable when ions are slaved to electrons" begin
    # With slaved ions there is no ion solve, so nothing reaches the ion face
    # ledger and nothing is booked as ion loss — the source PR3 adds, fed by that
    # ledger, will have nothing to emit in this configuration either.
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
