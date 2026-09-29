# `run_simulation!` takes two optional callbacks that bracket every step:
# `callback_before_step(RP)` once prepare_timestep! has set the step's inputs from the state
# at tⁿ, before solve_timestep!, and
# `callback_after_step(RP)` at tⁿ⁺¹, after the step's snapshots and controller update.
# Drivers use them to record what the snapshots do not carry, to impose a field between
# steps, or to diagnose the change one step makes, without re-implementing the loop.

@testsnippet StepCallbackRun begin
    function step_callback_rp()
        config = SimulationConfig{Float64}(
            NR = 6, NZ = 6, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-6, t_end_s = 3.0e-6, R0B0 = 1.0, prefilled_gas_pressure = 2.0e-3,
            snap0D_Δt_s = 1.0e-6, snap2D_Δt_s = 3.0e-6,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(
            diffu = false, convec = false, mean_ExB = false, turb_ExB_mixing = false,
            Ampere = false, E_para_self_ES = false, E_para_self_EM = false,
            Coulomb_Collision = false, Gas_evolve = false, Ti_evolve = false,
            update_ni_independently = false,
        )
        initialize!(RP)
        return RP
    end
end

@testitem "run_simulation! calls callback_after_step once per step, after the state advanced" setup = [StepCallbackRun] begin
    RP = step_callback_rp()
    seen_steps = Int[]
    seen_times = Float64[]
    n_snaps0D = Int[]
    run_simulation!(
        RP; callback_after_step = rp -> begin
            push!(seen_steps, rp.step)
            push!(seen_times, rp.time_s)
            push!(n_snaps0D, length(rp.diagnostics.snaps0D))
        end
    )
    @test seen_steps == [1, 2, 3]
    @test seen_times ≈ [1.0e-6, 2.0e-6, 3.0e-6]
    # The 0D snapshot of the step is already recorded when the callback runs.
    @test n_snaps0D == [2, 3, 4]
    # The default is no callback, and the loop still runs.
    RP2 = step_callback_rp()
    run_simulation!(RP2)
    @test RP2.step == 3
end

@testitem "callback_before_step sees tⁿ and callback_after_step tⁿ⁺¹, so the pair brackets each step" setup = [StepCallbackRun] begin
    RP = step_callback_rp()
    events = Tuple{Symbol, Int, Float64}[]
    run_simulation!(
        RP;
        callback_before_step = rp -> push!(events, (:before, rp.step, rp.time_s)),
        callback_after_step = rp -> push!(events, (:after, rp.step, rp.time_s)),
    )
    @test first.(events) == [:before, :after, :before, :after, :before, :after]
    @test getindex.(events, 2) == [0, 1, 1, 2, 2, 3]
    t = getindex.(events, 3)
    @test t[2:2:end] .- t[1:2:end] ≈ fill(RP.dt, 3)   # each pair spans exactly one step
end

@testitem "callback_before_step sees the step's inputs set from the state at tⁿ" setup = [StepCallbackRun] begin
    # Jϕ is set from the state at the start of each step (prepare_timestep!); the callback runs
    # after that, so it sees the current state's Jϕ, not the one of the step before.
    RP = step_callback_rp()
    RP.plasma.ne .= 1.0e14
    RP.plasma.ni .= 1.0e14
    mismatch = Float64[]
    run_simulation!(
        RP; callback_before_step = rp -> begin
            pla, F, c = rp.plasma, rp.fields, rp.config.constants
            Z = RAPID2D.bulk_ion_charge(rp)
            J = @. (pla.ne * c.qe * pla.ue_para + pla.ni * (c.ee * Z) * pla.ui_para) * F.bϕ
            push!(mismatch, maximum(abs, J .- pla.Jϕ))
        end
    )
    @test length(mismatch) == 3 && all(==(0.0), mismatch)
    @test maximum(abs, RP.plasma.Jϕ) > 0                     # the current did change over the run
end

@testitem "an external field set in callback_before_step is not overwritten before the step solves" begin
    # With a field file the step refreshes the external field from it; the callback runs after
    # that refresh, so what it writes is the field the step goes on with.
    mktempdir() do dir
        file = joinpath(dir, "static.dat")
        R, Z = range(0.8, 2.2, length = 8), range(-1.2, 1.2, length = 9)
        open(file, "w") do io
            println(io, "Time=\t0.0")
            println(io, "R_NUM=\t", length(R), "\tR_MIN=\t", first(R), "\tR_MAX=\t", last(R))
            println(io, "Z_NUM=\t", length(Z), "\tZ_MIN=\t", first(Z), "\tZ_MAX=\t", last(Z))
            println(io, "R Z B_R B_Z psi LoopVoltage")
            for r in R, z in Z
                println(io, r, " ", z, " 0.0 0.005 0.0 1.0")
            end
        end
        config = SimulationConfig{Float64}(
            inputs = InputPaths(field = file), NR = 8, NZ = 9, R0B0 = 1.0, prefilled_gas_pressure = 2.0e-3,
            dt = 1.0e-6, t_end_s = 3.0e-6, snap0D_Δt_s = 1.0e-6, snap2D_Δt_s = 3.0e-6,
        )
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(
            diffu = false, convec = false, mean_ExB = false, turb_ExB_mixing = false,
            Ampere = false, E_para_self_ES = false, E_para_self_EM = false,
            Coulomb_Collision = false, Gas_evolve = false, Ti_evolve = false, update_ni_independently = false,
        )
        initialize!(RP)
        off_by = Float64[]
        run_simulation!(
            RP;
            callback_before_step = rp -> fill!(rp.fields.BZ_ext, 0.123),
            callback_after_step = rp -> push!(off_by, maximum(abs, rp.fields.BZ_ext .- 0.123)),
        )
        @test off_by == zeros(3)
    end
end
