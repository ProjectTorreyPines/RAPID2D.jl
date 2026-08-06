# How snapshot storage is allocated, and when a snapshot is due — the two guards
# 201b1f8 named as belonging in their own change.
#
#   1. Buffers were sized from `t_end_s / snap*_Δt_s` and allocated EAGERLY, so one
#      config asked for 1.5 GiB from a run that recorded nothing. Storage now grows
#      with `push!`.
#   2. `snap*_Δt_s` was neither required to be ≥ `dt` nor a multiple of it, so a
#      non-integer ratio silently dropped snapshots. Settled in `validate_config!`.

@testsnippet SnapAlloc begin
    # Snapshot writers resolve Output_path relative to the process cwd, and
    # TestItemRunner cd's into each test file's directory. cleanup=false is REQUIRED:
    # the RAPID constructor opens ADIOS handles closed by a FINALIZER at a
    # GC-determined time, so the directory must outlive the RAPID object.
    scratch_output_dir() = mktempdir(; cleanup = false)

    "A config on a box wall. Everything the snapshot machinery reads is a keyword."
    function snap_config(; kw...)
        return SimulationConfig{Float64}(
            device_Name = "manual", NR = 15, NZ = 15,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            Output_path = scratch_output_dir(),
            dt = 1.0e-8, t_end_s = 1.0e-7,
            snap0D_Δt_s = 2.0e-8, snap2D_Δt_s = 5.0e-8;
            kw...
        )
    end

    "An initialized RAPID with a seed plasma, ready to step."
    function snap_case(; kw...)
        RP = RAPID{Float64}(snap_config(; kw...))
        initialize!(RP)
        RP.plasma.ne .= 1.0e15
        RP.plasma.ni .= 1.0e15
        RP.plasma.Te_eV .= 5.0
        return RP
    end
end

# ── Allocation ───────────────────────────────────────────────────────────────────────

@testitem "Snapshot storage is empty until a snapshot is taken" setup = [SnapAlloc] begin
    # The 201b1f8 case: t_end_s outruns the interval by 5000×, which used to mean 1.5 GiB
    # before a single step ran. Nothing may be allocated on `t_end_s` alone.
    RP = RAPID{Float64}(snap_config(t_end_s = 0.5, snap2D_Δt_s = 1.0e-4, snap0D_Δt_s = 2.0e-5))

    @test isempty(RP.diagnostics.snaps0D)
    @test isempty(RP.diagnostics.snaps2D)

    # initialize! rebuilds diagnostics; it must not reintroduce the preallocation.
    initialize!(RP)
    @test isempty(RP.diagnostics.snaps0D)
    @test isempty(RP.diagnostics.snaps2D)
end

@testitem "The first snapshot records against no predecessor" setup = [SnapAlloc] begin
    # `measure_snap0D!` reads the previous snapshot for the growth rate, and indexed
    # `snaps0D[max(1, tid_0D - 1)]` — on an empty vector that is `snaps0D[1]`, a
    # BoundsError. Preallocation always handed it a zeroed slot, so the push! path that
    # already existed in `update_snaps0D!` had never been reached.
    RP = snap_case()
    @test isempty(RP.diagnostics.snaps0D)

    RAPID2D.update_snaps0D!(RP)

    @test length(RP.diagnostics.snaps0D) == 1
    @test RP.diagnostics.snaps0D[1].time_s == RP.time_s
    @test RP.diagnostics.snaps0D[1].ne > 0
end

@testitem "Snapshots accumulate in order without trailing blanks" setup = [SnapAlloc] begin
    # Preallocation left the unused tail zero-filled, and `snaps0D.time_s` (the Vector
    # getproperty overload) returned those zeros as if they were samples.
    #
    # The run must fall SHORT of config.t_end_s for the gap to show — RAPID keeps its own
    # `t_end_s`, and tests that chain run_simulation! calls move it. Sized from
    # config.t_end_s = 1 µs the buffers held 51 and 21 slots for a run that fills 6 and 3.
    RP = snap_case(t_end_s = 1.0e-6)
    RP.t_end_s = 1.0e-7

    run_simulation!(RP)

    @test length(RP.diagnostics.snaps0D) == 6   # t = 0, 20, 40, 60, 80, 100 ns
    @test length(RP.diagnostics.snaps2D) == 3   # t = 0, 50, 100 ns

    t0D = RP.diagnostics.snaps0D.time_s
    @test issorted(t0D)
    @test t0D[1] == 0.0
    @test all(>(0), t0D[2:end])          # no zero-filled tail
    @test t0D[end] ≈ RP.t_end_s rtol = 1.0e-9
end

# ── Cadence guards ───────────────────────────────────────────────────────────────────

@testitem "A snapshot interval below dt is rejected" setup = [SnapAlloc] begin
    # Recording more often than the solver steps is not something it can do, and it is
    # what inflated the preallocated count past the step count.
    @test_throws ArgumentError validate_config!(snap_config(dt = 1.0e-6, snap0D_Δt_s = 1.0e-7))
    @test_throws ArgumentError validate_config!(snap_config(dt = 1.0e-6, snap2D_Δt_s = 1.0e-7))

    # Exactly dt is the boundary and is legal — record every step.
    @test validate_config!(snap_config(dt = 1.0e-6, snap0D_Δt_s = 1.0e-6, snap2D_Δt_s = 1.0e-6)) === nothing
end

@testitem "A non-integer snapshot interval is snapped onto the dt grid" setup = [SnapAlloc] begin
    # 2.4·dt can never be hit — `is_snap*_time` is only evaluated at multiples of dt, so
    # the samples at 2.4, 7.2, 12.0 … fall between steps and vanish.
    config = snap_config(dt = 1.0e-8, snap0D_Δt_s = 2.4e-8, snap2D_Δt_s = 5.0e-8)

    @test_logs (:warn,) match_mode = :any validate_config!(config)

    @test config.snap0D_Δt_s ≈ 2.0e-8
    @test config.snap2D_Δt_s ≈ 5.0e-8   # already a multiple; left alone
end

@testitem "An interval already on the dt grid passes silently" setup = [SnapAlloc] begin
    # A non-regression guard: warning on the package defaults would train everyone to
    # ignore the warning that matters.
    @test_logs validate_config!(snap_config(dt = 1.0e-5, snap0D_Δt_s = 2.0e-5, snap2D_Δt_s = 1.0e-4))
    @test_logs validate_config!(snap_config(dt = 1.0e-9, snap0D_Δt_s = 2.0e-5, snap2D_Δt_s = 1.0e-4))

    # Why the check is `isapprox` and not `==`: 9e-8 IS 3·dt at dt = 3e-8, but the round
    # trip lands on 8.999999999999999e-8. Equality would warn and perturb a good interval,
    # which is the rate denominator downstream — so it must come out untouched, to the bit.
    config = snap_config(dt = 3.0e-8, snap0D_Δt_s = 9.0e-8, snap2D_Δt_s = 9.0e-8)
    @test_logs validate_config!(config)
    @test config.snap0D_Δt_s === 9.0e-8
    @test config.snap2D_Δt_s === 9.0e-8
end

@testitem "A snapped interval actually restores the dropped snapshots" setup = [SnapAlloc] begin
    # End to end: unsnapped, only t = 0, 72, 96 ns land within 0.1·dt of a 2.4e-8
    # multiple. Snapped to 2e-8 the series is complete — 0, 20, 40, 60, 80, 100 ns.
    RP = snap_case(snap0D_Δt_s = 2.4e-8)

    @test RP.config.snap0D_Δt_s ≈ 2.0e-8
    run_simulation!(RP)
    @test length(RP.diagnostics.snaps0D) == 6
end
