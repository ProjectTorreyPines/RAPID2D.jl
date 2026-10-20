# A step as a transaction (`take_step!`): the state a step writes is copied out before it; an
# attempt the verdict rejects is put back, and the step taken in halves, down to Δt/2⁶.

@testsnippet StepTransactionRun begin
    # Ions free to evolve, with ionization, gas, ion energy, the electrostatic estimate, a
    # powered coil and a passive one, Ampère on: the state of every owner moves in a step.
    function transaction_rp(; dt = 2.0e-8)
        config = SimulationConfig{Float64}(;
            device_Name = "manual", manual = ManualSetup{Float64}(BR = 0.0, BZ = 1.0e-3, Eϕ = 0.5),
            NR = 17, NZ = 17, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = dt, t_end_s = 100dt,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(; cleanup = false),
        )
        RP = RAPID{Float64}(config)
        RP.flags.Ampere = true             # before initialize!, which builds the Grad–Shafranov operator
        initialize!(RP)
        f = RP.flags
        f.update_ni_independently = true
        f.Ti_evolve = true
        f.Gas_evolve = true
        G, pla = RP.G, RP.plasma
        inwall = G.nodes.state .> 0.5
        @. pla.ne = ifelse(inwall, 1.0e15 * (0.2 + exp(-((G.R2D - 1.45)^2 + (G.Z2D - 0.05)^2) / 0.02)), 0.0)
        pla.ni .= pla.ne
        fill!(pla.Te_eV, 5.0)
        fill!(pla.Ti_eV, 1.0)
        μ0 = RP.config.constants.μ0
        for (rc, zc, R, V, name) in ((2.3, 0.6, 1.0e-3, 50.0, "PF"), (2.3, -0.6, 2.0e-3, 0.0, "passive"))
            add_coil!(
                RP.coil_system, Coil{Float64}(;
                    location = (r = rc, z = zc), area = π * 0.05^2, resistance = R,
                    self_inductance = μ0 * rc * (log(8rc / 0.05) - 7 / 4), is_powered = V != 0,
                    is_controllable = false, name, voltage_applied = V,
                )
            )
        end
        initialize_coil_system!(RP)
        initialize_coupled_fields!(RP)
        RAPID2D.update_transport_quantities!(RP)   # what run_simulation! establishes before its loop
        return RP
    end

    # Equal value, bit for bit where it is a number (isequal: NaN equals NaN, −0.0 differs from 0.0)
    same(a::Number, b::Number) = isequal(a, b)
    same(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(same.(a, b))
    same(a::AbstractSet, b::AbstractSet) = a == b
    same(a::AbstractDict, b::AbstractDict) = a == b
    same(a::Tuple, b::Tuple) = length(a) == length(b) && all(same.(a, b))
    same(a::Union{Symbol, AbstractString, Function, Nothing}, b) = isequal(a, b)
    function same(a, b)
        typeof(a) === typeof(b) || return false
        return all(n -> same(getfield(a, n), getfield(b, n)), fieldnames(typeof(a)))
    end

    # The step state of a run, as a value to compare
    step_state(RP) = RAPID2D.StepState(RP)

    # A step as run_simulation! takes it, with the refresh that ends it (the field-line analysis
    # and the snapshots left out): the caches the next step starts from follow its state
    function step!(RP)
        RAPID2D.take_step!(RP)
        RP.flags.E_para_self_ES && RAPID2D.estimate_electrostatic_field_effects!(RP)
        RAPID2D.update_transport_quantities!(RP)
        return RP
    end
end

@testitem "Every field of the step's owners is put back or is classified as not written by a step" setup = [StepTransactionRun] begin
    using RAPID2D: StepState, save_step_state!, restore_step_state!, STEP_STATE_RECORDS, STEP_STATE_FIELDS
    RP = transaction_rp()

    # What a step does not write, by owner: constants over a run, caches rebuilt where they are
    # used, and what is refreshed from the state after an accepted step. A new field fails here
    # until it is put in the step state or in these lists.
    not_written = Dict(
        :rapid => (
            :G, :wall, :fitted_wall, :external_field, :eRRCs, :iRRCs, :config, :flags, :operators,
            :t_start_s, :t_end_s, :dt, :tElap, :flf, :snap0D_path, :snap2D_path,
        ),
        :transport => (
            :dims, :Dpara0, :Dperp0, :Dpara_e_coll, :Dpara_i_coll, :Dpara_amb, :Dpara_e_eff, :νi_eff,
            :νi_neutral, :νi_coulomb, :ion_species, :ion_solvers, :Dpara, :Dperp, :L_mixing, :Dpol_turb,
            :DRR_turb, :DRZ_turb, :DZZ_turb, :DRR, :DRZ, :DZZ, :CTRR, :CTRZ, :CTZZ, :wall_faces,
        ),
        :reactions => (:dims,),
        :diagnostics => (:dims_RZ, :snaps0D, :snaps2D, :step_retries),
        :closed_surface => (:basis, :u_system, :block_system),
        :coil_system => (
            :coils, :n_total, :n_powered, :n_controllable, :powered_indices, :controllable_indices,
            :passive_indices, :mutual_inductance, :Green_coils2bdy, :Green_grid2coils,
            :dGreen_dRg_grid2coils, :dGreen_dZg_grid2coils, :inside_domain_indices, :μ0, :cu_resistivity,
        ),
        :coil => (:location, :area, :resistance, :self_inductance, :is_powered, :is_controllable, :name, :max_voltage, :max_current),
    )
    owners = Dict(
        :rapid => RP, :transport => RP.transport, :reactions => RP.reactions, :diagnostics => RP.diagnostics,
        :closed_surface => RP.closed_surface, :coil_system => RP.coil_system, :coil => RP.coil_system.coils[1],
    )
    for (owner, obj) in owners
        written = owner === :rapid ?
            (STEP_STATE_RECORDS..., STEP_STATE_FIELDS.rapid..., :transport, :reactions, :diagnostics, :closed_surface, :coil_system) :
            STEP_STATE_FIELDS[owner]
        @test isempty(intersect(written, not_written[owner]))
        @test Set(fieldnames(typeof(obj))) == Set((written..., not_written[owner]...))
    end

    # every field the step state holds, of every kind (arrays, vectors that change length,
    # sets, nested records, numbers, flags, a function), comes back bit for bit
    function perturb!(obj, name)
        v = getfield(obj, name)
        # a coil voltage changes kind both ways, a number for a function and back
        name === :voltage_applied && return setfield!(obj, name, v isa Function ? 1.0 : sin)
        if v isa Vector{<:Real}
            push!(v, 7.0)
        elseif v isa Array{<:Real}
            v .= v .* 2 .+ 1
        elseif v isa AbstractSet
            push!(v, :perturbed)
        elseif v isa Bool
            setfield!(obj, name, !v)
        elseif v isa Real
            setfield!(obj, name, v + one(v))
        elseif ismutable(v)
            foreach(n -> perturb!(v, n), fieldnames(typeof(v)))
        end
        return obj
    end
    RP.coil_system.coils[1].voltage_applied = t -> 50.0   # a waveform, as a function
    for _ in 1:3                       # some state to put back, not only the initial zeros
        step!(RP)
    end
    before = deepcopy(StepState(RP))
    ck = StepState(RP)
    for name in STEP_STATE_RECORDS
        foreach(n -> perturb!(getfield(RP, name), n), fieldnames(typeof(getfield(RP, name))))
    end
    for owner in (:transport, :reactions, :diagnostics, :closed_surface, :coil_system)
        foreach(n -> perturb!(getfield(RP, owner), n), STEP_STATE_FIELDS[owner])
    end
    foreach(c -> foreach(n -> perturb!(c, n), STEP_STATE_FIELDS.coil), RP.coil_system.coils)
    foreach(n -> perturb!(RP, n), STEP_STATE_FIELDS.rapid)
    # arrays and records the step rebinds are put back where they now are
    RP.fields.Bϕ = RP.fields.Bϕ .+ 1
    RP.plasma.mean_ExB_R = RP.plasma.mean_ExB_R .- 1
    RP.closed_surface.solve = deepcopy(RP.closed_surface.solve)
    RP.diagnostics.near_vacuum = deepcopy(RP.diagnostics.near_vacuum)
    RP.transport.ion_N = zeros(size(RP.transport.ion_N, 1), 2)   # another shape: replaced back
    rebound = RP.fields.Bϕ
    @test !same(StepState(RP), before)
    restore_step_state!(RP, ck)
    @test same(StepState(RP), before)
    @test RP.fields.Bϕ === rebound      # written into, not replaced
    # the checkpoint holds no reference into RP: changing RP after the save leaves it whole
    save_step_state!(ck, RP)
    RP.plasma.ne .= -1.0
    push!(RP.diagnostics.Ntracker.cum0D_Ni_loss_s, 1.0)
    restore_step_state!(RP, ck)
    @test same(StepState(RP), before)
end

@testitem "A rejected attempt leaves the caches it does not put back as it found them" setup = [StepTransactionRun] begin
    using RAPID2D: StepState, save_step_state!, restore_step_state!, attempt_step!, Accept
    RP = transaction_rp()
    for _ in 1:3
        step!(RP)
    end
    OP = RP.operators
    refreshed() = deepcopy(
        (
            RP.transport.Dpara, RP.transport.Dperp, RP.transport.DRR, RP.transport.DRZ, RP.transport.DZZ,
            RP.transport.CTRR, RP.transport.CTRZ, RP.transport.CTZZ, RP.transport.νi_eff, RP.transport.Dpara_e_eff,
            RP.transport.L_mixing, OP.A_conv_e, OP.A_diffu_e, OP.div_ue, OP.ΔGS, OP.∂R, OP.∂Z,
        )
    )
    caches = refreshed()
    ck = StepState(RP)
    attempt_step!(RP, RP.time_s + RP.dt, true, rp -> Accept(), nothing)   # the whole attempt, floors included
    restore_step_state!(RP, ck)
    @test same(refreshed(), caches)
end

@testitem "An attempt put back and taken again is the step taken once, bit for bit" setup = [StepTransactionRun] begin
    using RAPID2D: StepState, restore_step_state!, attempt_step!, Accept
    once, twice = transaction_rp(), transaction_rp()
    for RP in (once, twice), _ in 1:3
        step!(RP)
    end
    ck = StepState(twice)
    attempt_step!(twice, twice.time_s + twice.dt, true, rp -> Accept(), nothing)
    restore_step_state!(twice, ck)
    for RP in (once, twice), _ in 1:3
        step!(RP)
    end
    @test same(step_state(twice), step_state(once))
    @test same(twice.transport.Dpara, once.transport.Dpara)
end

@testitem "A rejected step is taken in two halves that land on the step's end" setup = [StepTransactionRun] begin
    using RAPID2D: take_step!, StepTransaction, Accept, Reject, book_ionization_sources!, correct_negative_densities!,
        estimate_electrostatic_field_effects!, update_transport_quantities!
    RP, ref = transaction_rp(), transaction_rp()
    for r in (RP, ref), _ in 1:3
        step!(r)
    end
    dt, t0, n0 = RP.dt, RP.time_s, RP.step

    # rejects the whole step once; records the length and start of every attempt it judges, and
    # the callback records the attempts it is called on
    attempts = Tuple{Float64, Float64}[]
    called = Tuple{Float64, Float64}[]
    verdict = rp -> (push!(attempts, (rp.dt, rp.time_s)); length(attempts) == 1 ? Reject(:test) : Accept())
    callback_before_step = rp -> push!(called, (rp.dt, rp.time_s))
    take_step!(RP, StepTransaction(RP); verdict, callback_before_step)
    @test attempts == [(dt, t0), (dt / 2, t0), (dt / 2, t0 + 32 * (dt / 64))]
    @test called == attempts            # once on every attempt, the rejected one included
    @test RP.time_s === t0 + dt          # on the step's end exactly, as an unsplit step
    @test RP.step == n0 + 1              # one step, however many parts
    @test RP.dt === dt
    @test RP.diagnostics.step_retries.rejected == Dict(:test => 1)
    @test RP.diagnostics.step_retries.split_steps == 1

    # the same two halves by hand, with the refresh between them and none at the end
    ref.dt = dt / 2
    for (k, t_end) in enumerate((t0 + 32 * (dt / 64), t0 + dt))
        prepare_timestep!(ref)
        solve_timestep!(ref)
        ref.time_s = t_end
        k == 2 && (ref.step += 1)
        book_ionization_sources!(ref)
        correct_negative_densities!(ref)
        if k == 1
            estimate_electrostatic_field_effects!(ref)
            update_transport_quantities!(ref)
        end
    end
    ref.dt = dt
    a, b = step_state(RP), step_state(ref)
    for name in (:ne, :ni, :Te_eV, :Ti_eV, :ue_para, :ui_para, :n_H2_gas)
        @test getfield(a.plasma, name) ≈ getfield(b.plasma, name) rtol = 1.0e-12
    end
    @test a.diagnostics.Ntracker.cum0D_Ne_src ≈ b.diagnostics.Ntracker.cum0D_Ne_src rtol = 1.0e-12
    @test a.coil_system.ΔI_step ≈ b.coil_system.ΔI_step rtol = 1.0e-12
    @test RP.coil_system.Δt === dt / 2   # the last part: the coils' last step, as ΔI_step's
end

@testitem "A rejected second half is halved again, and the accepted first half stands" setup = [StepTransactionRun] begin
    using RAPID2D: take_step!, StepTransaction, Accept, Reject
    RP = transaction_rp()
    step!(RP)
    dt, t0 = RP.dt, RP.time_s
    attempts = Tuple{Float64, Float64}[]
    verdict = rp -> begin
        push!(attempts, (rp.dt, rp.time_s))
        length(attempts) in (1, 3) ? Reject(length(attempts) == 1 ? :whole : :second) : Accept()
    end
    take_step!(RP, StepTransaction(RP); verdict)
    u = dt / 64
    @test attempts == [(dt, t0), (dt / 2, t0), (dt / 2, t0 + 32u), (dt / 4, t0 + 32u), (dt / 4, t0 + 48u)]
    @test RP.time_s === t0 + dt
    @test RP.diagnostics.step_retries.rejected == Dict(:whole => 1, :second => 1)
    @test RP.diagnostics.step_retries.split_steps == 1
end

@testitem "A step rejected down to Δt/2⁶ fails with the state put back as the step found it" setup = [StepTransactionRun] begin
    using RAPID2D: take_step!, StepTransaction, StepState, Reject
    RP = transaction_rp()
    step!(RP)
    dt = RP.dt
    entry = StepState(RP)
    lengths = Float64[]
    @test_throws ErrorException take_step!(RP, StepTransaction(RP); verdict = rp -> (push!(lengths, rp.dt); Reject(:always)))
    @test lengths == dt ./ 2 .^ (0:6)    # the first part at every halving
    @test same(StepState(RP), entry)
    @test RP.dt === dt
    @test RP.diagnostics.step_retries.rejected == Dict(:always => 7)
end

@testitem "A step taken in parts after an accepted part fails back to the step's start" setup = [StepTransactionRun] begin
    # the first half is accepted (and its refresh run) before the second fails all the way down
    using RAPID2D: take_step!, StepTransaction, StepState, Accept, Reject
    RP = transaction_rp()
    step!(RP)
    entry = StepState(RP)
    Dpara = copy(RP.transport.Dpara)
    calls = Ref(0)
    verdict = rp -> (calls[] += 1; calls[] == 2 ? Accept() : Reject(:late))
    @test_throws ErrorException take_step!(RP, StepTransaction(RP); verdict)
    @test same(StepState(RP), entry)
    @test same(RP.transport.Dpara, Dpara)   # refreshed again from the state put back
end

@testitem "A rejected attempt leaves no trace in the ledgers or the reaction counts" setup = [StepTransactionRun] begin
    using RAPID2D: StepState, restore_step_state!, attempt_step!, Accept
    RP = transaction_rp()
    for _ in 1:3
        step!(RP)
    end
    ledgers(rp) = deepcopy(
        (
            rp.diagnostics.Ntracker, rp.diagnostics.near_vacuum, rp.diagnostics.ampere_picard,
            rp.closed_surface.ledger, rp.vertical_field_control.energy, rp.reactions.counts, rp.reactions.published,
        )
    )
    seen = ledgers(RP)
    ck = StepState(RP)
    attempt_step!(RP, RP.time_s + RP.dt, true, rp -> Accept(), nothing)
    @test !same(ledgers(RP), seen)       # the attempt booked something
    restore_step_state!(RP, ck)
    @test same(ledgers(RP), seen)
end

@testitem "solve_timestep! and advance_timestep! refuse a step other than RP.dt" setup = [StepTransactionRun] begin
    RP = transaction_rp()
    @test_throws ArgumentError solve_timestep!(RP, RP.dt / 2)
    @test_throws ArgumentError advance_timestep!(RP, 2 * RP.dt)
    t0 = RP.time_s
    advance_timestep!(RP, RP.dt)          # the step itself is accepted, as before
    @test RP.time_s == t0                 # (advance_timestep! leaves the clock to its caller)
end
