# Electron wall: the Robin path for the electron continuity equation — the only path.
# Rows on in-wall nodes, a Robin debit on the diagonal, the loss booked per wall face.
# internal/docs/src/notes/design/wall-flux-channels.md §2.1, §3.

@testitem "electron wall: one path, no flag" begin
    @test !(:electron_wall in fieldnames(SimulationFlags{Float64}))
    @test_throws MethodError SimulationFlags{Float64}(electron_wall = :zeroing)
    c = SimulationConfig{Float64}(NR = 6, NZ = 6)
    @test c.electron_wall_albedo == 0.0
end

@testitem "electron wall channels: R = 1 is reflective, ceilings are non-negative, tensor matches legacy" begin
    using RAPID2D: electron_wall_channels, electron_wall_absorption_speeds, electron_transport_operator,
        wall_faces, total_tensor, build_wall_diffusion_matrix
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Dperp0 = 0.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    faces = wall_faces(RP.G)
    v = electron_wall_absorption_speeds(RP, faces)
    @test length(v) == length(faces)
    @test all(>=(0), v)
    # R_e = 1: every channel's ceiling is multiplied by (1 − R) = 0 — a reflective wall
    RP.config.electron_wall_albedo = 1.0
    @test all(==(0), electron_wall_absorption_speeds(RP, faces))
    RP.config.electron_wall_albedo = 0.0
    # tensor consistency: with Dperp0 = 0 the channel sum reproduces the transport tensor on
    # every in-wall node. The wall-aware operator never reads the tensor outside the wall,
    # so those nodes are not part of the contract.
    D_RR, D_RZ, D_ZZ = total_tensor(electron_wall_channels(RP))
    tp = RP.transport
    inw = RP.G.nodes.in_wall_nids
    scale = maximum(abs, tp.DZZ[inw])
    @test D_RR[inw] ≈ tp.DRR[inw] atol = 1.0e-10 * scale
    @test D_RZ[inw] ≈ tp.DRZ[inw] atol = 1.0e-10 * scale
    @test D_ZZ[inw] ≈ tp.DZZ[inw] atol = 1.0e-10 * scale
    A, v2 = electron_transport_operator(RP, faces)
    @test v2 == v
    @test A == build_wall_diffusion_matrix(RP.G, tp.DRR, tp.DRZ, tp.DZZ; faces = faces, v_absorb = v)
end

@testitem "electron Robin wall: what the wall took plus what remains is what there was" begin
    function one_step(θ)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = 2.0e-6, t_end_s = 2.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,   # no mid-run snapshot: `update_snaps0D!` resets the tracker
        )
        config.Output_path = mktempdir()
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(
            diffu = true, convec = false, src = false,
            Atomic_Collision = false, Coulomb_Collision = false, mean_ExB = false,
            turb_ExB_mixing = false, E_para_self_ES = false, E_para_self_EM = false, Ampere = false,
            Te_evolve = false, ud_evolve = false, Ti_evolve = false, Gas_evolve = false,
            update_ni_independently = false, secondary_electron = false, negative_n_correction = false,
        )
        w = RP.flags.θ_imp
        RP.flags.θ_imp = ImplicitWeights{Float64}(transport = θ, growth = θ, decay = w.decay, gas = w.gas)
        initialize!(RP)
        G = RP.G
        inw = G.nodes.in_wall_nids
        RP.plasma.ne .= 0
        RP.plasma.ne[inw] .= 1.0e14 .* (1 .+ 0.3 .* sin.(3 .* G.R2D[inw]))   # in-wall only; on/out stay 0
        N0 = sum(G.Jacob[inw] .* RP.plasma.ne[inw])
        loss0 = RP.diagnostics.Ntracker.cum0D_Ne_loss
        run_simulation!(RP)
        N1 = sum(G.Jacob[inw] .* RP.plasma.ne[inw])
        absorbed = (RP.diagnostics.Ntracker.cum0D_Ne_loss - loss0) / (2π * G.dR * G.dZ)
        return N0, N1, absorbed, RP
    end
    for θ in (1.0, 0.5)
        N0, N1, absorbed, RP = one_step(θ)
        @test absorbed > 0
        @test (N0 - N1) ≈ absorbed rtol = 1.0e-10
        @test all(==(0), RP.plasma.ne[RP.G.nodes.on_out_wall_nids])
        @test sum(RP.diagnostics.Ntracker.cum2D_Ne_loss) ≈ RP.diagnostics.Ntracker.cum0D_Ne_loss rtol = 1.0e-12
    end
end

@testitem "electron Robin wall: R_e = 1 conserves Σ J·ne exactly" begin
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 2.0e-6, t_end_s = 1.0e-5, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, electron_wall_albedo = 1.0,
    )
    config.Output_path = mktempdir()
    RP = RAPID{Float64}(config)
    RP.flags = SimulationFlags{Float64}(
        diffu = true, convec = false, src = false,
        Atomic_Collision = false, Coulomb_Collision = false, mean_ExB = false, turb_ExB_mixing = false,
        E_para_self_ES = false, E_para_self_EM = false, Ampere = false, Te_evolve = false,
        ud_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
        secondary_electron = false, negative_n_correction = false,
    )
    initialize!(RP)
    inw = RP.G.nodes.in_wall_nids
    RP.plasma.ne .= 0
    RP.plasma.ne[inw] .= 1.0e14
    N0 = sum(RP.G.Jacob[inw] .* RP.plasma.ne[inw])
    run_simulation!(RP)
    @test sum(RP.G.Jacob[inw] .* RP.plasma.ne[inw]) ≈ N0 rtol = 1.0e-12
    @test RP.diagnostics.Ntracker.cum0D_Ne_loss == 0
end

@testitem "electron Robin wall + face-flux convection: the ledger closes with both channels on" begin
    # Under `:robin` convection is the face-flux operator: its wall-face outflow is a
    # diagonal debit like the Robin one, so one ledger coefficient per face (diffusive +
    # convective speed) books exactly what the two operators removed. The nodal upwind
    # operator could not do this (no rows on the grid frame; a central-difference branch
    # at |u| < eps), which is why the electron half of the face-flux work rides in this PR.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 2.0e-6, t_end_s = 2.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    config.Output_path = mktempdir()
    RP = RAPID{Float64}(config)
    RP.flags = SimulationFlags{Float64}(
        diffu = true, convec = true, src = false,
        Atomic_Collision = false, Coulomb_Collision = false, mean_ExB = true,
        turb_ExB_mixing = false, E_para_self_ES = false, E_para_self_EM = false, Ampere = false,
        Te_evolve = false, ud_evolve = false, Ti_evolve = false, Gas_evolve = false,
        update_ni_independently = false, secondary_electron = false, negative_n_correction = false,
    )
    initialize!(RP)
    G = RP.G
    inw = G.nodes.in_wall_nids
    RP.plasma.ne .= 0
    RP.plasma.ne[inw] .= 1.0e14
    # a poloidal drift toward the outer wall; E_para_self_ES is off so nothing rebuilds it
    RP.plasma.mean_ExB_R .= 2.0e4
    RP.plasma.mean_ExB_Z .= 0.0
    N0 = sum(G.Jacob[inw] .* RP.plasma.ne[inw])
    loss0 = RP.diagnostics.Ntracker.cum0D_Ne_loss
    run_simulation!(RP)
    N1 = sum(G.Jacob[inw] .* RP.plasma.ne[inw])
    booked = (RP.diagnostics.Ntracker.cum0D_Ne_loss - loss0) / (2π * G.dR * G.dZ)
    @test booked > 0
    @test (N0 - N1) ≈ booked rtol = 1.0e-10
    # the convective channel really is on: some wall face sees an outflow
    using RAPID2D: face_outflow_speeds, wall_faces
    @test any(>(0), face_outflow_speeds(G, wall_faces(G), RP.plasma.ueR, RP.plasma.ueZ))
end

# One driver for the flag combinations below: in-wall uniform density, an imposed
# poloidal drift when convection is on, one step, and the ledger identity.
@testsnippet RobinLedgerDriver begin
    using RAPID2D: ImplicitWeights
    function robin_one_step(;
            diffu, convec, implicit = true, θ = 0.5, secondary = false,
            independent_ions = false, drift = convec ? 2.0e4 : 0.0
        )
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = 2.0e-6, t_end_s = 2.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        config.Output_path = mktempdir()
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(
            diffu = diffu, convec = convec, Implicit = implicit, src = false,
            Atomic_Collision = false, Coulomb_Collision = false, mean_ExB = convec,
            turb_ExB_mixing = false, E_para_self_ES = false, E_para_self_EM = false, Ampere = false,
            Te_evolve = false, ud_evolve = false, Ti_evolve = false, Gas_evolve = false,
            update_ni_independently = independent_ions, secondary_electron = secondary,
            negative_n_correction = false,
        )
        w = RP.flags.θ_imp
        RP.flags.θ_imp = ImplicitWeights{Float64}(transport = θ, growth = w.growth, decay = w.decay, gas = w.gas)
        initialize!(RP)
        G = RP.G
        inw = G.nodes.in_wall_nids
        RP.plasma.ne .= 0
        RP.plasma.ne[inw] .= 1.0e14
        RP.plasma.ni .= RP.plasma.ne
        RP.plasma.mean_ExB_R .= drift
        RP.plasma.mean_ExB_Z .= 0.0
        N0 = sum(G.Jacob[inw] .* RP.plasma.ne[inw])
        loss0 = RP.diagnostics.Ntracker.cum0D_Ne_loss
        run_simulation!(RP)
        N1 = sum(G.Jacob[inw] .* RP.plasma.ne[inw])
        booked = (RP.diagnostics.Ntracker.cum0D_Ne_loss - loss0) / (2π * G.dR * G.dZ)
        return (; N0, N1, booked, RP)
    end
end

@testitem "electron Robin wall: no transport, no loss booked" setup = [RobinLedgerDriver] begin
    r = robin_one_step(diffu = false, convec = false)
    @test r.N1 == r.N0
    @test r.booked == 0
end

@testitem "electron Robin wall: convection only, θ = 0.5 — booked with the θ the solve used" setup = [RobinLedgerDriver] begin
    r = robin_one_step(diffu = false, convec = true, θ = 0.5)
    @test r.booked > 0
    @test (r.N0 - r.N1) ≈ r.booked rtol = 1.0e-10
end

@testitem "electron Robin wall: explicit scheme books at nⁿ and the ledger closes" setup = [RobinLedgerDriver] begin
    r = robin_one_step(diffu = true, convec = true, implicit = false)
    @test r.booked > 0
    @test (r.N0 - r.N1) ≈ r.booked rtol = 1.0e-10
end

@testitem "electron Robin wall: secondary electrons cannot enter the loss ledger without a source" setup = [RobinLedgerDriver] begin
    # The retired injection put γ·ni on out-wall nodes, rows the operator never solves.
    # Until secondaries are emitted through the wall faces from the ion ledger the flag is
    # inert, and the ledger must still close.
    r = robin_one_step(diffu = true, convec = true, secondary = true, independent_ions = true)
    @test r.booked > 0
    @test (r.N0 - r.N1) ≈ r.booked rtol = 1.0e-10
    @test all(==(0), r.RP.plasma.ne[r.RP.G.nodes.on_out_wall_nids])
end

@testitem "electron Robin wall: the late-time wall loss rate converges with the grid" tags = [:regression] begin
    # The legacy rule v_absorb = D/(2Δx) moved the absorbed fraction ~4× over NR = 31 → 181
    # (internal/docs/src/notes/TODO/wall-boundary-conditions.md §1.1). A kinetic Robin
    # coefficient is a surface property, so once the profile has relaxed to its lowest
    # eigenmode the decay rate must settle as the grid refines. The early transient is a
    # boundary layer of thickness √(D⊥t) far below any of these cells and is not the claim.
    using RAPID2D: ImplicitWeights
    function decay_rate(NR, NZ)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = NR, NZ = NZ, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = 1.0e-4, t_end_s = 2.0e-2, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        config.Output_path = mktempdir()
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(
            diffu = true, convec = false, src = false,
            Atomic_Collision = true, Coulomb_Collision = false, mean_ExB = false, turb_ExB_mixing = false,
            E_para_self_ES = false, E_para_self_EM = false, Ampere = false, Te_evolve = false,
            ud_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
            secondary_electron = false, negative_n_correction = false,
        )
        w = RP.flags.θ_imp
        RP.flags.θ_imp = ImplicitWeights{Float64}(transport = 1.0, growth = w.growth, decay = w.decay, gas = w.gas)
        initialize!(RP)
        inw = RP.G.nodes.in_wall_nids
        RP.plasma.ne .= 0
        RP.plasma.ne[inw] .= 1.0e14
        total() = sum(RP.G.Jacob[inw] .* RP.plasma.ne[inw])
        run_simulation!(RP)                 # 0 → 20 ms: relax onto the eigenmode
        N1, t1 = total(), RP.time_s
        RP.t_end_s = 4.0e-2
        run_simulation!(RP)                 # 20 → 40 ms: measure the decay
        N2, t2 = total(), RP.time_s
        return log(N1 / N2) / (t2 - t1)
    end
    γ31, γ61, γ121 = decay_rate(31, 31), decay_rate(61, 61), decay_rate(121, 121)
    d1, d2 = abs(γ61 - γ31) / γ31, abs(γ121 - γ61) / γ61
    @info "electron Robin wall late-time loss rate" γ_31 = γ31 γ_61 = γ61 γ_121 = γ121 step_31_61 = d1 step_61_121 = d2
    @test γ31 > 0
    # first-order convergence: each refinement halves the change (measured 0.152 → 0.069)
    @test d2 < 0.1
    @test d2 < 0.6 * d1
end

# ── Simple, readable checks on a uniform density: what the wall takes and at what rate ──────
@testsnippet UniformWallDriver begin
    using RAPID2D: ImplicitWeights, wall_faces, electron_wall_absorption_speeds, face_outflow_speeds
    # uniform n0 in-wall, transport only, θ = 1; convection driven by a FIXED u∥ along the manual
    # device's vertical field (ud_evolve = false, mean_ExB = false), so nothing recomputes u.
    function uniform_wall_run(; diffu, convec, albedo, nsteps, upara = 1.0e6)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = 2.0e-6, t_end_s = nsteps * 2.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            electron_wall_albedo = albedo,
        )
        config.Output_path = mktempdir()
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(
            diffu = diffu, convec = convec, Implicit = true, src = false,
            Atomic_Collision = false, Coulomb_Collision = false, mean_ExB = false,
            turb_ExB_mixing = false, E_para_self_ES = false, E_para_self_EM = false, Ampere = false,
            Te_evolve = false, ud_evolve = false, Ti_evolve = false, Gas_evolve = false,
            update_ni_independently = false, secondary_electron = false, negative_n_correction = false,
        )
        w = RP.flags.θ_imp
        RP.flags.θ_imp = ImplicitWeights{Float64}(transport = 1.0, growth = w.growth, decay = w.decay, gas = w.gas)
        initialize!(RP)
        G = RP.G
        inw = G.nodes.in_wall_nids
        RP.plasma.ne .= 0
        RP.plasma.ne[inw] .= 1.0e14
        RP.plasma.ni .= RP.plasma.ne
        RP.plasma.ue_para .= convec ? upara : 0.0
        RP.plasma.ueR .= RP.plasma.ue_para .* RP.fields.bR
        RP.plasma.ueZ .= RP.plasma.ue_para .* RP.fields.bZ
        faces = wall_faces(G)
        v_diff = diffu ? electron_wall_absorption_speeds(RP, faces) : zeros(length(faces))
        v_conv = convec ? face_outflow_speeds(G, faces, RP.plasma.ueR, RP.plasma.ueZ) : zeros(length(faces))
        rate0 = sum(f.area * (v_diff[k] + v_conv[k]) for (k, f) in enumerate(faces)) * 1.0e14   # /s at n = n0
        Ntot() = sum(G.Jacob[inw] .* RP.plasma.ne[inw]) * 2π * G.dR * G.dZ
        N0 = Ntot()
        run_simulation!(RP)
        Nend = Ntot()
        return (; N0, Nend, booked = RP.diagnostics.Ntracker.cum0D_Ne_loss, rate0, RP, inw, faces, v_conv)
    end
end

@testitem "uniform density, diffusion only: first-step loss is the wall's own Σ A·v_absorb·n, booked exactly" setup = [UniformWallDriver] begin
    r = uniform_wall_run(diffu = true, convec = false, albedo = 0.0, nsteps = 1)
    loss = r.N0 - r.Nend
    @test loss > 0
    @test r.booked ≈ loss rtol = 1.0e-10
    # θ = 1 books at n¹ ≤ n0, so the rate sits just below the n0 estimate
    @test 0.98 < loss / r.RP.dt / r.rate0 <= 1.0
end

@testitem "uniform density: albedo 0.5 halves the diffusive loss, albedo 1 conserves and stays uniform over 100 steps" setup = [UniformWallDriver] begin
    r0 = uniform_wall_run(diffu = true, convec = false, albedo = 0.0, nsteps = 1)
    rh = uniform_wall_run(diffu = true, convec = false, albedo = 0.5, nsteps = 1)
    @test rh.booked / r0.booked ≈ 0.5 rtol = 5.0e-3          # v_absorb ∝ (1 − R); the 0.3 % is n¹ ≠ n0
    r1 = uniform_wall_run(diffu = true, convec = false, albedo = 1.0, nsteps = 100)
    @test r1.booked == 0
    @test r1.Nend ≈ r1.N0 rtol = 1.0e-12
    ne = r1.RP.plasma.ne[r1.inw]
    @test (maximum(ne) - minimum(ne)) / (sum(ne) / length(ne)) < 1.0e-12
end

@testitem "uniform density, convection only: the loss is the downstream-face outflow n·A·max(u·n̂,0), booked exactly" setup = [UniformWallDriver] begin
    r = uniform_wall_run(diffu = false, convec = true, albedo = 0.0, nsteps = 1)
    loss = r.N0 - r.Nend
    @test loss > 0
    @test r.booked ≈ loss rtol = 1.0e-10
    @test loss / r.RP.dt ≈ r.rate0 rtol = 1.0e-6              # n¹ = n0 on the (upwind) owner cell at step 1
    # the same over 100 steps: what left is what was booked
    r100 = uniform_wall_run(diffu = false, convec = true, albedo = 0.0, nsteps = 100)
    @test r100.Nend < 0.5 * r100.N0
    @test r100.booked ≈ r100.N0 - r100.Nend rtol = 1.0e-10
end

@testitem "uniform density, convection: albedo 0.5 halves the convective loss, albedo 1 keeps every electron (pile-up, no net flux)" setup = [UniformWallDriver] begin
    # θ = 1 books at the step-end wall density, which the albedo itself changes, so the
    # halving is exact in the COEFFICIENT: booked = (1 − R)·Σ_f A_f·v_out,f·n¹_w·dt.
    for (R, r) in (
            (0.0, uniform_wall_run(diffu = false, convec = true, albedo = 0.0, nsteps = 1)),
            (0.5, uniform_wall_run(diffu = false, convec = true, albedo = 0.5, nsteps = 1)),
        )
        ne = r.RP.plasma.ne
        expected = (1 - R) * sum(f.area * r.v_conv[k] * ne[f.nid] for (k, f) in enumerate(r.faces)) * r.RP.dt
        @test r.booked ≈ expected rtol = 1.0e-10
        @test (r.N0 - r.Nend) ≈ r.booked rtol = 1.0e-10
    end
    r1 = uniform_wall_run(diffu = false, convec = true, albedo = 1.0, nsteps = 100)
    @test r1.booked == 0
    @test r1.Nend ≈ r1.N0 rtol = 1.0e-10                 # nothing leaves a zero-net-flux wall
    ne = r1.RP.plasma.ne[r1.inw]
    @test maximum(ne) > 1.5 * minimum(ne)                # …so the flow piles up in the downstream wall cells
    @test all(>=(0), ne)
end

@testitem "an electron albedo outside [0, 1] is rejected even when only convection reaches the wall" setup = [UniformWallDriver] begin
    # With diffu = false the diffusive builder (which validates the albedo) never runs, so the
    # convective path must validate it itself: 1.5 would turn the wall into a source.
    @test_throws ArgumentError uniform_wall_run(diffu = false, convec = true, albedo = 1.5, nsteps = 1)
    @test_throws ArgumentError uniform_wall_run(diffu = false, convec = true, albedo = -0.1, nsteps = 1)
end

@testitem "an empty wall-face cache is refused rather than dropping the wall silently" begin
    # The faces and the electron operators live on `Transport` from `initialize!`. A Transport
    # that has not been through it (or was replaced afterwards) has no faces and an empty
    # divergence: solving with it would apply no wall loss and no convection while still
    # booking the outflow speeds, so operator and ledger would diverge without a word.
    using RAPID2D: convective_wall_operator, wall_faces
    using RAPID2D.SparseArrays: spzeros
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)
    uR = fill(1.0e5, G.NR, G.NZ)
    uZ = zeros(G.NR, G.NZ)
    # a supplied divergence that is empty while faces see outflow is an inconsistent cache
    @test_throws ArgumentError convective_wall_operator(G, faces, uR, uZ, 0.0; C = spzeros(G.NR * G.NZ, G.NR * G.NZ))
    RP.transport = RAPID2D.Transport{Float64}(G.NR, G.NZ)
    @test isempty(RP.transport.wall_faces)
    @test_throws ArgumentError solve_electron_continuity_equation!(RP)
    @test_throws ArgumentError RAPID2D.ion_step_operators(RP)
end
