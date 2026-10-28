# The ions' u∥ and Tᵢ carried by the particles that move them, as the electrons' are.
#
# The per-particle operator of the operator that moved the ion densities this step: the ion
# continuity's own (diffusion per transport group, convection by the ion velocity), or the
# electrons' where the ions are slaved to them. Published by the density move, retracted at the
# start of every step: nothing moved, nothing carried.
# internal/docs/src/reference/governing-equations.md §4.1–4.2;
# internal/docs/src/notes/plans/PLAN_ion-momentum-energy-transport.md §2.

@testsnippet IonCarriageCase begin
    using RAPID2D: NotCarried, CarriedByIonOperators, CarriedByElectronOperators, per_particle_operator!

    """
    A peaked plasma inside a box wall, ions free to evolve, ready for a standalone ion continuity
    solve: rates and reaction counts refreshed, `prev_n` the electrons' density.
    """
    function carriage_case(; NR = 25, NZ = 25, n0 = 1.0e17, Te = 5.0, Ti = 1.0, kw...)
        config = SimulationConfig{Float64}(;
            device_Name = "manual", NR = NR, NZ = NZ,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-7,
            t_end_s = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, kw...
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        RP.flags.update_ni_independently = true
        G, pla = RP.G, RP.plasma
        inw = G.nodes.in_wall_nids
        shape = @. exp(-((G.R2D - 1.5)^2 + G.Z2D^2) / 0.05)
        pla.ne .= 0.0
        vec(pla.ne)[inw] .= n0 .* vec(shape)[inw] .+ 1.0e12
        pla.ni .= pla.ne
        pla.Te_eV .= Te
        pla.Ti_eV .= Ti
        update_transport_quantities!(RP)
        RAPID2D.update_reaction_counts!(RP)
        RP.prev_n .= pla.ne
        return RP
    end

    "A smooth field on the grid, for the operator identities."
    smooth_grid(G; a = 1.0e4) = a .* (1 .+ 0.5 .* sin.(3 .* G.R2D) .* cos.(2 .* G.Z2D))
end

@testitem "the ion continuity publishes the operators it moved the densities with" setup = [IonCarriageCase] begin
    using RAPID2D: build_face_flux_divergence!, build_wall_pattern, ion_step_operators
    RP = carriage_case()
    pla = RP.plasma
    # the ions and the electrons drift apart along B
    pla.ue_para .= 2.0e5
    pla.ui_para .= 3.0e3 .* (1 .+ RP.G.Z2D)
    update_transport_quantities!(RP)
    car = RP.operators.ion_carriage
    @test car.source isa NotCarried                     # nothing moved yet
    groups, _, _ = ion_step_operators(RP)
    A_step = groups[1][2]
    solve_ion_continuity_equation!(RP)
    @test car.source isa CarriedByIonOperators
    @test length(car.parts) == 1
    part = car.parts[1]
    @test part.sids == [1]
    # diffusion minus convection, on the wall pattern, is the density solve's operator
    @test Matrix(part.A_part.matrix - car.A_conv.matrix) == Matrix(A_step)
    # the convection is the ions' velocity, not the electrons'
    @test pla.uiR != pla.ueR || pla.uiZ != pla.ueZ
    A_ion = build_face_flux_divergence!(build_wall_pattern(RP.G), RP.G, pla.uiR, pla.uiZ; upwind = RP.flags.upwind)
    @test car.A_conv.matrix.nzval == A_ion.matrix.nzval
    @test car.div_u == RAPID2D.wall_divergence(RP.G, pla.uiR, pla.uiZ)
end

@testitem "nothing moved, nothing carried: no publication without transport, retracted at the start of a step and on a new layout" setup = [IonCarriageCase] begin
    RP = carriage_case()
    car = RP.operators.ion_carriage
    RP.flags.diffu = false
    RP.flags.convec = false
    solve_ion_continuity_equation!(RP)
    @test car.source isa NotCarried
    RP.flags.diffu = true
    RP.flags.convec = true
    solve_ion_continuity_equation!(RP)
    @test car.source isa CarriedByIonOperators
    RAPID2D.retract_ion_carriage!(RP)
    @test car.source isa NotCarried
    solve_ion_continuity_equation!(RP)
    RAPID2D.layout_ion_species!(RP, RP.transport.ion_species)
    @test RP.operators.ion_carriage.source isa NotCarried
    # a step starts by retracting what the last one published: with nothing moving the ions, none
    car = RP.operators.ion_carriage
    solve_ion_continuity_equation!(RP)
    @test car.source isa CarriedByIonOperators
    RP.flags.diffu = false
    RP.flags.convec = false
    RAPID2D.advance_timestep!(RP)
    @test car.source isa NotCarried
    # slaved ions: the step's own density move, the electrons', is published
    RP.flags.diffu = true
    RP.flags.convec = true
    RP.flags.update_ni_independently = false
    RAPID2D.advance_timestep!(RP)
    @test car.source isa CarriedByElectronOperators
end

@testitem "the ions' own parallel conduction: Tᵢ/(m_I ν) with the gas drag and ν_ii, under the free-flight ceiling" setup = [IonCarriageCase] begin
    RP = carriage_case(; Ti = 2.0)
    tp, pla = RP.transport, RP.plasma
    EE = 1.602176634e-19
    inw = RP.G.nodes.in_wall_nids
    ν = tp.νi_neutral .+ tp.νi_birth .+ tp.νi_coulomb
    vm = RAPID2D.maxwellian_mean_speed.(pla.Ti_eV, pla.mi_mean)
    ceiling = RAPID2D.wall_step_ceiling.(vm, RP.flf.Lc_tot)
    expected = @. min(tp.Dpara0 + EE * pla.Ti_eV / (pla.mi_mean * ν), ceiling)
    @test tp.Dpara_cond_i[inw] ≈ expected[inw] rtol = 1.0e-12
    @test any(tp.νi_coulomb[inw] .> 0)                  # ν_ii is in it
    # without Coulomb collisions the self-collisions leave it, and it rises
    RP.flags.Coulomb_Collision = false
    update_transport_quantities!(RP)
    @test all(tp.Dpara_cond_i[inw] .>= expected[inw])
    # cold ions do not conduct; without any collision the ceiling binds
    pla.Ti_eV .= 0.0
    update_transport_quantities!(RP)
    @test all(iszero, tp.Dpara_cond_i[inw] .- tp.Dpara0)
    RP.flags.Atomic_Collision = false
    pla.Ti_eV .= 2.0
    update_transport_quantities!(RP)
    vm = RAPID2D.maxwellian_mean_speed.(pla.Ti_eV, pla.mi_mean)
    @test tp.Dpara_cond_i[inw] ≈ RAPID2D.wall_step_ceiling.(vm, RP.flf.Lc_tot)[inw] rtol = 1.0e-12
end

@testitem "one species: the momentum and the heat are carried by the per-particle operator of the published one, bit for bit" setup = [IonCarriageCase] begin
    using RAPID2D: ion_momentum_carriage, ion_heat_carriage, add_conduction_correction!
    RP = carriage_case()
    solve_ion_continuity_equation!(RP)
    car, pla = RP.operators.ion_carriage, RP.plasma
    part = car.parts[1]
    M = per_particle_operator!(similar(part.A_part), part.A_part, pla.ni; n_floor = 1.0)
    isnothing(part.A_cond) || add_conduction_correction!(M, part.A_part, part.A_cond, pla.ni, RP.G; n_floor = 1.0)
    A_adv = per_particle_operator!(similar(car.A_conv), car.A_conv, pla.ni; n_floor = 1.0)
    mom, heat = ion_momentum_carriage(RP), ion_heat_carriage(RP)
    @test mom.M.matrix.nzval == M.matrix.nzval
    @test heat.M.matrix.nzval == M.matrix.nzval
    @test mom.A_adv.matrix.nzval == A_adv.matrix.nzval
    @test heat.A_adv.matrix.nzval == A_adv.matrix.nzval
    @test heat.div_u == car.div_u                        # every ion moves: the whole compression
    # none published: nothing
    RAPID2D.retract_ion_carriage!(RP)
    @test isnothing(ion_momentum_carriage(RP))
    @test isnothing(ion_heat_carriage(RP))
end

@testitem "a mixture with a species that does not move: momentum by mass, heat by number, over the whole fluid" setup = [IonCarriageCase] begin
    using RAPID2D: ion_momentum_carriage, ion_heat_carriage, ion_plane
    using RAPID2D.Particles: H, H2, H3
    RP = carriage_case(; species = (H[0], H[1], H2[0], H2[1], H3[1]))
    pla, G = RP.plasma, RP.G
    inw = G.nodes.in_wall_nids
    # H₂⁺ is the transported species; H⁺ and H₃⁺ sit in the store, H₃⁺ only on one side
    @test length(RP.transport.ion_species) == 1
    still = [(k, z) for (k, recs) in enumerate(RP.store_layout.ion_records) for (z, _) in enumerate(recs) if (k, z) != RP.store_layout.transport_slots[1]]
    @test length(still) == 2
    for (i, slot) in enumerate(still)
        plane = ion_plane(pla, slot)
        plane .= 0.0
        vec(plane)[inw] .= i == 1 ? 0.2 .* vec(pla.ni)[inw] : 0.3 .* vec(pla.ni)[inw] .* (vec(G.Z2D)[inw] .> 0)
    end
    RAPID2D.update_charge_states!(RP)
    solve_ion_continuity_equation!(RP)
    RAPID2D.update_charge_states!(RP)
    car = RP.operators.ion_carriage
    part = car.parts[1]
    m_bulk = RAPID2D.bulk_ion_mass(RP)
    n_T = vec(copy(pla.ni))
    ρ_T, ρ_I, n_I = n_T .* (m_bulk / m_bulk), vec(pla.mni_tot) ./ m_bulk, vec(pla.ni_tot)
    f = vec(smooth_grid(G))
    V = vec(G.inVol2D)
    for (op, w_I) in ((ion_momentum_carriage(RP), ρ_I), (ion_heat_carriage(RP), n_I))
        # the moved part over the whole: what moves carries, what does not dilutes
        Mf = op.M * f
        moved = part.A_part * (n_T .* f) .- f .* (part.A_part * n_T)
        above = findall(>(1.0), w_I)
        isnothing(part.A_cond) && @test (w_I .* Mf)[above] ≈ moved[above] rtol = 1.0e-10
        # in total, with the conduction's pair transfers too: what the moving part's operator does to
        # n f beyond carrying f with the particles it moves (here the absorbing wall's drain)
        @test sum(V .* w_I .* Mf) ≈ sum(V .* moved) rtol = 1.0e-10
        convected = car.A_conv * (n_T .* f) .- f .* (car.A_conv * n_T)
        @test sum(V .* w_I .* (op.A_adv * f)) ≈ sum(V .* convected) rtol = 1.0e-10
    end
    # the compression is the moving part's
    heat = ion_heat_carriage(RP)
    @test vec(heat.div_u)[inw] ≈ (vec(car.div_u) .* n_T ./ n_I)[inw] rtol = 1.0e-12
end

@testitem "two transport groups: a part each, the conduction raised in one only, the momentum and the heat moved as each group moved its density" setup = [IonCarriageCase] begin
    using RAPID2D: IonRecord, PerSpeciesTransport, layout_ion_species!, ion_momentum_carriage, ion_heat_carriage,
        ion_plane, update_ui_para!, ion_transport_channels, shared_turbulent_channel, channel_D_para
    RP = carriage_case()
    RP.flags.ion_transport_policy = PerSpeciesTransport()
    tp, pla, G = RP.transport, RP.plasma, RP.G
    inw = G.nodes.in_wall_nids
    layout_ion_species!(RP, [tp.ion_species[1], IonRecord(:C6, 6 * tp.ion_species[1].mass, 6)])
    slots = RP.store_layout.transport_slots
    n_C = ion_plane(pla, slots[2])
    n_C .= 0.0
    vec(n_C)[inw] .= 0.05 .* vec(pla.ni)[inw] .* (1 .+ 0.5 .* vec(G.Z2D)[inw])
    update_transport_quantities!(RP)
    # the ions conduct with the bulk's own particle coefficient: raised for the heavier species where
    # it diffuses less, nowhere for the bulk
    turb = shared_turbulent_channel(RP)
    D_bulk, D_C = (channel_D_para(ion_transport_channels(RP, sp, turb)[1]) for sp in tp.ion_species)
    @test any(D_bulk[inw] .> D_C[inw])
    tp.Dpara_cond_i .= D_bulk
    solve_ion_continuity_equation!(RP)
    car = RP.operators.ion_carriage
    @test car.source isa CarriedByIonOperators
    @test [p.sids for p in car.parts] == [[1], [2]]
    @test isnothing(car.parts[1].A_cond) && !isnothing(car.parts[2].A_cond)
    RAPID2D.update_charge_states!(RP)
    m_bulk = RAPID2D.bulk_ion_mass(RP)
    n_s = [vec(copy(ion_plane(pla, slot))) for slot in slots]
    f = vec(smooth_grid(G))
    V = vec(G.inVol2D)
    for (op, weight, w_I) in (
            (ion_momentum_carriage(RP), sp -> sp.mass / m_bulk, vec(pla.mni_tot) ./ m_bulk),
            (ion_heat_carriage(RP), _ -> 1.0, vec(pla.ni_tot)),
        )
        # each group's operator moves its own carriers, the variable counted over the whole fluid,
        # the conduction's pair transfers in the second group conserving the total
        ws = [weight(sp) .* n for (sp, n) in zip(tp.ion_species, n_s)]
        moved = sum(p.A_part * (w .* f) .- f .* (p.A_part * w) for (p, w) in zip(car.parts, ws))
        @test sum(V .* w_I .* (op.M * f)) ≈ sum(V .* moved) rtol = 1.0e-10
    end
    # a uniform drift stays uniform through the momentum update
    RP.flags.Atomic_Collision = false
    RP.flags.Coulomb_Collision = false
    RP.flags.src = false
    RP.fields.E_para_tot .= 0.0
    pla.ui_para .= 4.0e3
    update_ui_para!(RP)
    @test maximum(abs, vec(pla.ui_para)[inw] .- 4.0e3) <= 1.0e-9 * 4.0e3
end

@testitem "slaved ions: the electrons' convection carries them, values copied onto the ions' own buffer; a uniform drift stays uniform" setup = [IonCarriageCase] begin
    using RAPID2D: ion_momentum_carriage, ion_heat_carriage, publish_electron_carriage!, ue_Te_operators, update_ui_para!
    RP = carriage_case()
    RP.flags.update_ni_independently = false
    RP.flags.Atomic_Collision = false
    RP.flags.Coulomb_Collision = false
    RP.flags.src = false
    RP.fields.E_para_tot .= 0.0
    pla = RP.plasma
    pla.ue_para .= 2.0e5                                  # the electrons' flow along b̂ moves the density
    pla.ui_para .= 4.0e3
    update_transport_quantities!(RP)
    RAPID2D.cache_electron_operators!(RP)
    publish_electron_carriage!(RP)
    car = RP.operators.ion_carriage
    @test car.source isa CarriedByElectronOperators && car.convec
    pops = ue_Te_operators(RP)
    @test any(!iszero, pops.A_adv.matrix.nzval)
    mom = ion_momentum_carriage(RP)
    @test mom.A_adv.matrix == pops.A_adv.matrix
    @test mom.A_adv.matrix.nzval !== pops.A_adv.matrix.nzval
    heat = ion_heat_carriage(RP)
    @test heat.A_adv.matrix == pops.A_adv.matrix
    @test heat.div_u == pops.div_u
    update_ui_para!(RP)
    inw = RP.G.nodes.in_wall_nids
    @test maximum(abs, vec(pla.ui_para)[inw] .- 4.0e3) <= 1.0e-9 * 4.0e3
end

# ── u_i∥ ─────────────────────────────────────────────────────────────────────────────────

@testitem "u_i∥ with nothing carried: births, field, drag and friction as before, bit for bit" setup = [IonCarriageCase] begin
    using RAPID2D: born_ion_mass, get_H2_ion_RRC, ion_mean!, update_charge_states!, update_ui_para!
    RP = carriage_case()
    pla = RP.plasma
    pla.ui_para .= 2.0e3 .* (1 .+ RP.G.R2D .- 1.5)
    RAPID2D.retract_ion_carriage!(RP)
    RAPID2D.update_ue_para!(RP)                          # records the e–i impulse the ions take
    update_charge_states!(RP)
    dt, ee = RP.dt, RP.config.constants.ee
    ρ = copy(pla.mni_tot)
    u = copy(pla.ui_para)
    Δρ = born_ion_mass(RP)
    @. u *= ifelse(ρ > 0, max(0.0, 1 - Δρ / ρ), 1.0)
    # the drag's rates at the diluted drift, where the update reads them
    u_entry = copy(pla.ui_para)
    pla.ui_para .= u
    K_ela, K_cx = get_H2_ion_RRC(RP, RP.iRRCs, :Elastic), get_H2_ion_RRC(RP, RP.iRRCs, :Charge_Exchange)
    ν_a = @. pla.n_H2_gas * (0.5 * K_ela + K_cx)
    replace!(ν_a, NaN => 0.0)
    pla.ui_para .= u_entry
    Z_mean = ion_mean!(similar(ρ), RP, (z, m) -> z)
    @. u = (u + dt * ee * Z_mean * RP.fields.E_para_tot / pla.mi_mean) / (1 + dt * ν_a)
    @. u -= ifelse(ρ > 0, pla.I_ei / ρ, 0.0)
    @test any(!iszero, pla.I_ei) && any(!iszero, ν_a)
    update_ui_para!(RP)
    @test pla.ui_para == u
end

@testitem "u_i∥ carried: a uniform drift stays uniform through the continuity and the momentum update, at an absorbing wall" setup = [IonCarriageCase] begin
    using RAPID2D: update_ui_para!
    for upwind in (true, false), albedo in (0.0, 0.6)
        RP = carriage_case(; ion_wall_albedo = albedo)
        RP.flags.upwind = upwind
        RP.flags.Atomic_Collision = false
        RP.flags.Coulomb_Collision = false
        RP.flags.src = false
        RP.fields.E_para_tot .= 0.0
        pla = RP.plasma
        pla.ui_para .= 4.0e3                              # a flow along b̂ that reaches the wall
        update_transport_quantities!(RP)
        RAPID2D.cache_electron_operators!(RP)
        RP.flags.Coulomb_Collision = false
        solve_ion_continuity_equation!(RP)
        @test RP.operators.ion_carriage.source isa CarriedByIonOperators
        update_ui_para!(RP)
        inw = RP.G.nodes.in_wall_nids
        @test maximum(abs, vec(pla.ui_para)[inw] .- 4.0e3) <= 1.0e-9 * 4.0e3
    end
end

@testitem "u_i∥ mixing along straight field lines: each column settles to its own mass-weighted mean, the momentum to first order in Δt" setup = [PureMixingRun] begin
    # slaved ions ride the electrons' prescribed tensor; nothing but the mixing acts on u_i∥
    D, L = 500.0, 0.6
    τ = L^2 / (π^2 * D)
    function ion_mixing_run!(RP)
        keep! = tensor_keeper(RP)
        held = Ref(true)
        run_simulation!(
            RP;
            callback_before_step = rp -> (keep!(rp); rp.transport.Dpara_cond_i .= rp.transport.Dpara; held[] &= enforce_pure_mixing!(rp)),
            callback_after_step = rp -> (held[] &= pure_mixing_holds(rp)),
        )
        return held[]
    end
    function set_state!(RP)
        G, pla = RP.G, RP.plasma
        inw = G.nodes.in_wall_nids
        shape = @. 1 + 0.5 * cos(2π * G.Z2D / L)
        pla.ne .= 0.0
        pla.ne[inw] .= 1.0e14 .* vec(shape)[inw]
        pla.ni .= pla.ne
        pla.ue_para .= 1.0e6
        pla.ui_para .= 3.0e4 .* shape
        pla.Te_eV .= 10.0
        return RP
    end
    RP = set_state!(pure_mixing_RP(; t_end_s = 6τ, D_along = D))
    G, pla = RP.G, RP.plasma
    V = vec(G.inVol2D)
    n0, u0 = vec(copy(pla.ni)), vec(copy(pla.ui_para))
    cols = wall_columns(G)
    expected = Dict(i => sum(V[nodes] .* n0[nodes] .* u0[nodes]) / sum(V[nodes] .* n0[nodes]) for (i, nodes) in cols)
    @test ion_mixing_run!(RP)
    u1 = vec(pla.ui_para)
    for (i, nodes) in cols
        @test maximum(u1[nodes]) - minimum(u1[nodes]) < 1.0e-2 * 3.0e4
        @test all(x -> isapprox(x, expected[i]; rtol = 1.0e-2), u1[nodes])
    end
    # Σ V nᵢ u drifts by the split's O(Δt)
    errs = Float64[]
    for dt in (4.0e-6, 2.0e-6, 1.0e-6)
        RP = set_state!(pure_mixing_RP(; dt = dt, t_end_s = 1.0e-4, D_along = D))
        inw = RP.G.nodes.in_wall_nids
        P(RP) = sum(V[inw] .* vec(RP.plasma.ni)[inw] .* vec(RP.plasma.ui_para)[inw])
        P0 = P(RP)
        @test ion_mixing_run!(RP)
        push!(errs, abs(P(RP) - P0) / abs(P0))
    end
    @test errs[end] < 1.0e-2
    @test 0.35 < errs[2] / errs[1] < 0.65
    @test 0.35 < errs[3] / errs[2] < 0.65
end

@testitem "u_i∥ carried by the ions' own operators: the momentum of independent ions converges to first order in Δt at a reflecting wall" setup = [IonCarriageCase] begin
    using RAPID2D: update_ui_para!
    # Bohm and the collisional channel as they come, no convection, a reflecting wall: what the
    # diffusion moves keeps Σ V ρ u up to the split between the continuity and the momentum update
    errs = Float64[]
    for dt in (4.0e-7, 2.0e-7, 1.0e-7)
        RP = carriage_case(; ion_wall_albedo = 1.0, dt = dt)
        RP.transport.Dperp0 = 200.0                       # a mixing that moves the profile within the run
        RP.flags.convec = false
        RP.flags.Atomic_Collision = false
        RP.flags.Coulomb_Collision = false
        RP.flags.src = false
        pla, G = RP.plasma, RP.G
        V = vec(G.inVol2D)
        inw = G.nodes.in_wall_nids
        pla.ui_para .= 5.0e3 .* (1 .+ 0.8 .* sin.(4 .* G.R2D) .* cos.(3 .* G.Z2D))
        update_transport_quantities!(RP)
        RP.fields.E_para_tot .= 0.0
        P() = sum(V[inw] .* vec(pla.mni_tot)[inw] .* vec(pla.ui_para)[inw])
        RAPID2D.update_charge_states!(RP)
        P0, u0 = P(), copy(pla.ui_para)
        for _ in 1:round(Int, 2.0e-5 / dt)
            RP.prev_n .= pla.ne
            solve_ion_continuity_equation!(RP)
            update_ui_para!(RP)
        end
        @test pla.ui_para != u0                           # it was carried
        push!(errs, abs(P() - P0) / abs(P0))
    end
    @test errs[end] < 1.0e-2
    @test 0.35 < errs[2] / errs[1] < 0.65
    @test 0.35 < errs[3] / errs[2] < 0.65
end

# ── Tᵢ ───────────────────────────────────────────────────────────────────────────────────

@testitem "Tᵢ with nothing carried: the atomic power alone, as before, bit for bit" setup = [IonCarriageCase] begin
    using RAPID2D: update_ion_heating_powers!, update_Ti!
    RP = carriage_case()
    pla = RP.plasma
    pla.Ti_eV .= 0.5 .+ 0.3 .* RP.G.Z2D .^ 2
    RAPID2D.retract_ion_carriage!(RP)
    update_ion_heating_powers!(RP)
    P = copy(pla.iPowers.atomic)
    @test any(!iszero, P)
    T = @. clamp(pla.Ti_eV + (2 / 3) * P * RP.dt / RP.config.constants.ee, RP.config.min_Te, RP.config.max_Te)
    update_Ti!(RP)
    @test pla.Ti_eV == T
    @test pla.iPowers.tot == P
    @test all(iszero, pla.iPowers.conv) && all(iszero, pla.iPowers.diffu) && all(iszero, pla.iPowers.visc_heat)
end

@testitem "Tᵢ mixing along straight field lines: each column settles to its energy mean, the sheared flow's kinetic energy turned to heat; the energy converges to first order in Δt" setup = [PureMixingRun] begin
    D, L = 500.0, 0.6
    τ = L^2 / (π^2 * D)
    mi, ee = 2 * 1.67262192369e-27 + 9.1093837015e-31, 1.602176634e-19
    function ion_mixing_run!(RP)
        keep! = tensor_keeper(RP)
        held = Ref(true)
        run_simulation!(
            RP;
            callback_before_step = rp -> (keep!(rp); rp.transport.Dpara_cond_i .= rp.transport.Dpara; held[] &= enforce_pure_mixing!(rp)),
            callback_after_step = rp -> (held[] &= pure_mixing_holds(rp)),
        )
        return held[]
    end
    function set_state!(RP)
        RP.flags.Ti_evolve = true
        G, pla = RP.G, RP.plasma
        inw = G.nodes.in_wall_nids
        shape = @. 1 + 0.5 * cos(2π * G.Z2D / L)
        pla.ne .= 0.0
        pla.ne[inw] .= 1.0e14 .* vec(shape)[inw]
        pla.ni .= pla.ne
        pla.ue_para .= 1.0e6
        pla.ui_para .= 3.0e4 .* shape
        pla.Te_eV .= 10.0
        pla.Ti_eV .= 1.0 .+ 0.5 .* sin.(2π .* G.Z2D ./ L)
        return RP
    end
    # The flow's kinetic energy (≈ 12 eV per ion) dwarfs the heat (≈ 1.7 eV), so the split's O(Δt)
    # energy drift shows in Tᵢ several times over (measured: −2.3 %, −1.1 %, −0.6 % at Δt = 1, 0.5,
    # 0.25 µs): the final state is checked to converge to the mixed one, first order
    T_errs = Float64[]
    for dt in (1.0e-6, 5.0e-7)
        RP = set_state!(pure_mixing_RP(; dt = dt, t_end_s = 6τ, D_along = D))
        m_i = RAPID2D.bulk_ion_mass(RP)
        G, pla = RP.G, RP.plasma
        V = vec(G.inVol2D)
        n0, u0, T0 = vec(copy(pla.ni)), vec(copy(pla.ui_para)), vec(copy(pla.Ti_eV))
        cols = wall_columns(G)
        expected = Dict(i => mixed_state(nodes, V, n0, u0, T0; me = m_i) for (i, nodes) in cols)
        @test ion_mixing_run!(RP)
        T1 = vec(pla.Ti_eV)
        err = 0.0
        for (i, nodes) in cols
            @test maximum(T1[nodes]) - minimum(T1[nodes]) < 1.0e-2           # homogenized
            @test expected[i].T > sum(V[nodes] .* n0[nodes] .* T0[nodes]) / sum(V[nodes] .* n0[nodes]) + 0.5   # heated
            err = max(err, maximum(abs, T1[nodes] ./ expected[i].T .- 1))
        end
        push!(T_errs, err)
    end
    @test T_errs[end] < 1.5e-2
    @test T_errs[2] / T_errs[1] < 0.6
    V = vec(pure_mixing_RP(; t_end_s = 1.0e-6, D_along = D).G.inVol2D)
    # Σ V n (3/2 e T + ½ m u²): the split's O(Δt), halving Δt at least halves it
    errs = Float64[]
    for dt in (4.0e-6, 2.0e-6, 1.0e-6)
        RP = set_state!(pure_mixing_RP(; dt = dt, t_end_s = 1.0e-4, D_along = D))
        inw = RP.G.nodes.in_wall_nids
        energy(RP) = sum(V[inw] .* vec(RP.plasma.ni)[inw] .* (1.5 .* ee .* vec(RP.plasma.Ti_eV)[inw] .+ 0.5 .* RAPID2D.bulk_ion_mass(RP) .* vec(RP.plasma.ui_para)[inw] .^ 2))
        E0 = energy(RP)
        @test ion_mixing_run!(RP)
        push!(errs, abs(energy(RP) - E0) / E0)
    end
    @test errs[end] < 1.0e-2
    @test errs[2] / errs[1] < 0.6
    @test errs[3] / errs[2] < 0.6
end

@testitem "Tᵢ compressed by a converging flow: Tᵢ ∝ n_I^(2/3) where the flow is uniform, the time error second order" setup = [IonCarriageCase] begin
    using RAPID2D: update_Ti!
    # u_Z = −α Z on every node (u_R = 0): ∇·u = −α, the density grows as e^{αt}, the temperature
    # as its 2/3 power. The walls the flow leaves behind start a rarefaction the upwind divergence
    # spreads inward, a spatial error of the density alone; the centre rows stay uniform.
    α, t_end, T0 = 1.0e4, 4.0e-5, 2.0
    errs = Float64[]
    for dt in (2.0e-6, 1.0e-6, 5.0e-7)
        RP = carriage_case(; dt = dt)
        RP.flags.diffu = false
        RP.flags.Atomic_Collision = false
        RP.flags.Coulomb_Collision = false
        RP.flags.src = false
        RP.flags.mean_ExB = false                        # the flow is the prescribed u∥ alone
        G, pla = RP.G, RP.plasma
        inw = G.nodes.in_wall_nids
        pla.ne .= 0.0
        vec(pla.ne)[inw] .= 1.0e17
        pla.ni .= pla.ne
        pla.Ti_eV .= T0
        pla.ui_para .= -α .* G.Z2D ./ RP.fields.bZ
        update_transport_quantities!(RP)
        @test maximum(abs, vec(pla.uiZ)[inw] .+ α .* vec(G.Z2D)[inw]) < 1.0e-6 * α
        for _ in 1:round(Int, t_end / dt)
            RP.prev_n .= pla.ne
            solve_ion_continuity_equation!(RP)
            update_Ti!(RP)
        end
        centre = [k for k in inw if abs(G.Z2D[k]) < 0.02 && abs(G.R2D[k] - 1.5) < 0.2]   # the Z = 0 row
        n, T = vec(pla.ni)[centre] ./ 1.0e17, vec(pla.Ti_eV)[centre] ./ T0
        @test all(n .> 1.45)                              # compressed, e^{0.4} = 1.49
        @test maximum(abs, T ./ n .^ (2 / 3) .- 1) < 1.0e-4
        push!(errs, maximum(abs, T .- exp(2 * α * t_end / 3)))
    end
    @test errs[end] < 1.0e-5
    @test errs[2] / errs[1] < 0.35                       # θ = ½ on both: second order
    @test errs[3] / errs[2] < 0.35
end

@testitem "the ions' viscous heating: m_I Γ_M(u_i∥) per ion, in the total and the snapshots; zero at uniform u and where u_i∥ is not diffused" setup = [IonCarriageCase] begin
    using RAPID2D: ion_momentum_carriage, dissipation_rate!, update_ion_heating_powers!
    RP = carriage_case()
    pla, G = RP.plasma, RP.G
    RP.transport.Dperp0 = 200.0
    pla.ui_para .= 5.0e3 .* (1 .+ 0.8 .* sin.(4 .* G.R2D) .* cos.(3 .* G.Z2D))
    update_transport_quantities!(RP)
    solve_ion_continuity_equation!(RP)
    update_ion_heating_powers!(RP)
    mom = ion_momentum_carriage(RP)
    Γ = dissipation_rate!(similar(pla.ui_para), mom.M, pla.ui_para)
    inw = G.nodes.in_wall_nids
    @test vec(pla.iPowers.visc_heat)[inw] ≈ vec(pla.mi_mean .* Γ)[inw] rtol = 1.0e-12
    @test sum(vec(G.inVol2D .* pla.ni .* pla.iPowers.visc_heat)) > 0
    iP = pla.iPowers
    @test iP.tot == iP.atomic .+ iP.conv .+ iP.diffu .+ iP.visc_heat
    RAPID2D.update_snaps2D!(RP)
    @test RP.diagnostics.snaps2D[end].Pi_visc_heat == iP.visc_heat
    # uniform u: nothing to dissipate
    pla.ui_para .= 5.0e3
    update_ion_heating_powers!(RP)
    @test maximum(abs, iP.visc_heat) <= 1.0e-10 * maximum(abs, vec(pla.mi_mean .* Γ))
    # u_i∥ not diffused: no heating
    pla.ui_para .= 5.0e3 .* (1 .+ 0.8 .* sin.(4 .* G.R2D))
    RP.flags.ud_evolve = false
    update_ion_heating_powers!(RP)
    @test all(iszero, iP.visc_heat)
end

# ── schemes, retries, near vacuum ──────────────────────────────────────────────────────

@testitem "carried ions under ExpRB, and with Implicit = false: the Theta answer at a small Δt" setup = [IonCarriageCase] begin
    using RAPID2D: ExpRB, update_ui_para!, update_Ti!
    function marched(; decay = nothing, atomic = nothing, implicit = true, dt = 2.0e-9, steps = 50)
        RP = carriage_case(; dt = dt)
        RP.transport.Dperp0 = 200.0
        RP.flags.Coulomb_Collision = false
        RP.flags.Implicit = implicit
        isnothing(decay) || (RP.flags.scheme.decay = decay)
        isnothing(atomic) || (RP.flags.scheme.atomic = atomic)
        G, pla = RP.G, RP.plasma
        pla.ui_para .= 5.0e3 .* (1 .+ 0.8 .* sin.(4 .* G.R2D) .* cos.(3 .* G.Z2D))
        pla.Ti_eV .= 1 .+ 0.5 .* cos.(3 .* G.Z2D)
        update_transport_quantities!(RP)
        RAPID2D.update_reaction_counts!(RP)
        u0, T0 = copy(pla.ui_para), copy(pla.Ti_eV)
        for _ in 1:steps
            RP.prev_n .= pla.ne
            solve_ion_continuity_equation!(RP)
            update_ui_para!(RP)
            update_Ti!(RP)
        end
        inw = G.nodes.in_wall_nids
        return (u = vec(pla.ui_para)[inw], T = vec(pla.Ti_eV)[inw], Δu = vec(pla.ui_para .- u0)[inw], ΔT = vec(pla.Ti_eV .- T0)[inw])
    end
    ref = marched()
    @test maximum(abs, ref.Δu) > 0 && maximum(abs, ref.ΔT) > 0
    for other in (marched(; decay = ExpRB), marched(; atomic = ExpRB), marched(; implicit = false))
        @test maximum(abs, other.Δu .- ref.Δu) <= 1.0e-2 * maximum(abs, ref.Δu)
        @test maximum(abs, other.ΔT .- ref.ΔT) <= 1.0e-2 * maximum(abs, ref.ΔT)
    end
end

@testitem "a step taken again from its saved state carries the ions alike" setup = [IonCarriageCase] begin
    using RAPID2D: StepState, save_step_state!, restore_step_state!
    RP = carriage_case()
    RP.transport.Dperp0 = 200.0
    G, pla = RP.G, RP.plasma
    pla.ui_para .= 5.0e3 .* (1 .+ 0.8 .* sin.(4 .* G.R2D) .* cos.(3 .* G.Z2D))
    pla.Ti_eV .= 1 .+ 0.5 .* cos.(3 .* G.Z2D)
    update_transport_quantities!(RP)
    ck = StepState(RP)
    save_step_state!(ck, RP)
    RAPID2D.advance_timestep!(RP)
    @test RP.operators.ion_carriage.source isa CarriedByIonOperators
    u1, T1 = copy(pla.ui_para), copy(pla.Ti_eV)
    restore_step_state!(RP, ck)
    RAPID2D.advance_timestep!(RP)
    @test pla.ui_para ≈ u1 rtol = 1.0e-12
    @test pla.Ti_eV ≈ T1 rtol = 1.0e-12
end

@testitem "near vacuum: ions carried out of a dense blob into almost empty nodes stay within what they started with" setup = [IonCarriageCase] begin
    using RAPID2D: update_ui_para!, update_Ti!
    RP = carriage_case()
    RP.transport.Dperp0 = 200.0
    RP.flags.Atomic_Collision = false
    RP.flags.Coulomb_Collision = false
    RP.flags.src = false
    RP.fields.E_para_tot .= 0.0
    G, pla = RP.G, RP.plasma
    inw = G.nodes.in_wall_nids
    blob = @. exp(-((G.R2D - 1.5)^2 + G.Z2D^2) / 0.01)
    pla.ne .= 0.0
    vec(pla.ne)[inw] .= ifelse.(vec(blob)[inw] .> 1.0e-3, 1.0e18 .* vec(blob)[inw], 1.0e2)   # below 100 n_vac outside
    pla.ni .= pla.ne
    pla.ui_para .= 2.0e3 .* blob
    pla.Ti_eV .= 0.5 .+ 1.5 .* blob
    update_transport_quantities!(RP)
    RP.fields.E_para_tot .= 0.0
    u_lo, u_hi = extrema(vec(pla.ui_para)[inw])
    T_lo, T_hi = extrema(vec(pla.Ti_eV)[inw])
    for _ in 1:20
        RP.prev_n .= pla.ne
        solve_ion_continuity_equation!(RP)
        update_ui_para!(RP)
        update_Ti!(RP)
    end
    u, T = vec(pla.ui_para)[inw], vec(pla.Ti_eV)[inw]
    @test all(isfinite, u) && all(isfinite, T)
    @test u_lo - 0.02 * u_hi <= minimum(u) && maximum(u) <= 1.02 * u_hi
    @test 0.98 * T_lo <= minimum(T) && maximum(T) <= 1.02 * T_hi
    nv = RP.diagnostics.near_vacuum
    @test nv.relax_energy_i != 0 && nv.relax_momentum_i != 0
end
