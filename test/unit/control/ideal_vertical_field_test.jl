# The ideal vertical-field control on the uniform current disc of `CurrentDisc`
# (diagnostics/column_force_balance_test.jl): its vacuum bases, the target it shares with the
# force estimate, signs with a negative current, the update's timing and states, the induced
# field, the compensated flux, the default path, the refusals, and a few steps of a run.

@testitem "ideal vertical field: the bases are vacuum fields whose B and flux agree" setup = [CurrentDisc] begin
    using RAPID2D: IdealVerticalFieldControl, calculate_B_from_ψ, write_vertical_control_field!
    RP, _ = current_disc(; control = IdealVerticalFieldControl(t_on_s = 0.0))
    G, F, st = RP.G, RP.fields, RP.vertical_field_control
    st.R_ref = 1.6
    st.dt_applied = 1.0e-6
    interior = (2:(G.NR - 1), 2:(G.NZ - 1))
    for a in ([1.0e-3, 0.0], [0.0, 1.0e-3], [-4.0e-4, 7.0e-4])
        st.a .= a
        st.a_prev .= a
        write_vertical_control_field!(RP)
        BR, BZ = calculate_B_from_ψ(G, F.ψ_ctrl)
        # central differences are exact for these polynomials inside the grid
        @test BZ[interior...] ≈ F.BZ_ctrl[interior...] rtol = 1.0e-10
        @test BR[interior...] ≈ F.BR_ctrl[interior...] atol = 1.0e-12
        # the flux between two radii is ∫ B_Z 2πR dR (the integrand is linear in R)
        j, i1, i2 = 70, 10, 60
        flux = sum(π * (G.R1D[i] * F.BZ_ctrl[i, j] + G.R1D[i + 1] * F.BZ_ctrl[i + 1, j]) * G.dR for i in i1:(i2 - 1))
        @test flux ≈ 2π * (F.ψ_ctrl[i2, j] - F.ψ_ctrl[i1, j]) rtol = 1.0e-10
        @test all(iszero, F.Eϕ_ctrl)                            # no change, no induced field
    end
end

@testitem "ideal vertical field: the target is the force estimate's field to add" setup = [CurrentDisc] begin
    using RAPID2D: IdealVerticalFieldControl, vertical_field_control_target, write_vertical_control_field!,
        ReducedForceTarget, VerticalAndRadialField
    # the reduced-force target with the uniform field: a₁ is column_force_balance's Bv_to_add
    RP, _ = current_disc(; B0 = 2.0e-3, control = IdealVerticalFieldControl(t_on_s = 0.0, target = ReducedForceTarget()))
    target = vertical_field_control_target(RP)
    fb = column_force_balance(RP)
    @test target.solvable
    @test target.a[1] ≈ fb.Bv_to_add rtol = 1.0e-12
    @test target.a[2] == 0
    # the magnetic target (the default): for a one-signed current, minus the |Jϕ|dV-weighted mean
    # B_Z of the field without control
    RP, _ = current_disc(; B0 = 2.0e-3, control = IdealVerticalFieldControl(t_on_s = 0.0))
    w = abs.(RP.plasma.Jϕ) .* RP.G.inVol2D
    BZ_mean = sum(w .* (RP.fields.BZ_ext .+ RP.fields.BZ_self)) / sum(w)
    @test vertical_field_control_target(RP).a[1] ≈ -BZ_mean rtol = 1.0e-8
    # both bases (the column sits at Z0 = 0.1, off the midplane): A a = −F₀, and with the control
    # field added to the external one, nothing is left to cancel
    RP, _ = current_disc(; B0 = 2.0e-3, control = IdealVerticalFieldControl(t_on_s = 0.0, target = ReducedForceTarget(), field = VerticalAndRadialField()))
    RP.fields.BR_ext .= 3.0e-4
    target = vertical_field_control_target(RP)
    @test target.A[1, 2] != 0
    @test target.A * target.a ≈ -target.F rtol = 1.0e-12
    st = RP.vertical_field_control
    st.R_ref = (RP.G.R1D[1] + RP.G.R1D[end]) / 2
    st.a .= target.a
    st.a_prev .= target.a
    write_vertical_control_field!(RP)
    RP.fields.BR_ext .+= RP.fields.BR_ctrl
    RP.fields.BZ_ext .+= RP.fields.BZ_ctrl
    fb = column_force_balance(RP)
    @test abs(fb.Bv_to_add) < 1.0e-12 * abs(target.a[1])
    @test abs(fb.Br_to_add) < 1.0e-12 * abs(target.a[1])
end

@testitem "ideal vertical field: signs with a negative current" setup = [CurrentDisc] begin
    using RAPID2D: IdealVerticalFieldControl, vertical_field_control_target, write_vertical_control_field!
    RP0, d = current_disc()                                     # I < 0
    B_need = column_force_balance(RP0).Bv_req
    RP, _ = current_disc(; B0 = B_need, control = IdealVerticalFieldControl(t_on_s = 0.0))
    target = vertical_field_control_target(RP)
    # a uniform B_Z > 0 pushes a negative current inward
    @test d.I < 0 && target.A[1, 1] < 0
    # raising a₁ induces a negative loop voltage, which drives the negative current: power in
    st, F, G = RP.vertical_field_control, RP.fields, RP.G
    st.R_ref = 1.6
    st.dt_applied = 1.0e-6
    ΔB = 1.0e-3
    st.a .= [ΔB, 0.0]
    write_vertical_control_field!(RP)
    @test all(<(0), F.Eϕ_ctrl)
    @test sum(RP.plasma.Jϕ .* F.Eϕ_ctrl .* G.inVol2D) > 0
    # the control field reaches the total ψ the field lines follow: the axis moves inward by a ΔB/B_pa
    on = column_force_balance(trace_closed_region!(current_disc(; B0 = B_need)[1]))
    st.a_prev .= st.a                                           # the field without its induced E
    write_vertical_control_field!(RP)
    off = column_force_balance(trace_closed_region!(RP))
    shift = off.R_axis - on.R_axis
    @test shift < 0
    @test shift ≈ on.a_eff_by_Jϕ * ΔB / on.Bpol_edge rtol = 0.05
end

@testitem "ideal vertical field: off before t_on, the ramp, relaxation in physical time, saturation" setup = [CurrentDisc] begin
    using RAPID2D: IdealVerticalFieldControl, update_vertical_field_control!, vertical_field_control_target,
        FirstOrderResponse, InstantaneousResponse
    τ = 5.0e-5
    # a ramp shorter than a step: from turning on, the coefficient relaxes to a fixed target as
    # 1 − exp(−t/τ), whatever the step
    a_at(dt) = begin
        RP, _ = current_disc(; B0 = 2.0e-3, control = IdealVerticalFieldControl(t_on_s = 0.0, ramp_s = 1.0e-12, response = FirstOrderResponse(τ)))
        RP.dt = dt
        target = vertical_field_control_target(RP).a[1]
        for _ in 1:round(Int, 2τ / dt)
            update_vertical_field_control!(RP)
            RP.time_s += dt
        end
        RP.vertical_field_control.a[1], target, RP.vertical_field_control.status
    end
    a_fine, target, status = a_at(1.0e-6)
    a_coarse, _, _ = a_at(5.0e-6)
    @test a_fine ≈ target * (1 - exp(-2)) rtol = 1.0e-9
    @test a_coarse ≈ a_fine rtol = 1.0e-9
    @test status == 3
    # before t_on: off, nothing applied
    RP, _ = current_disc(; B0 = 2.0e-3, control = IdealVerticalFieldControl(t_on_s = 1.0e-3))
    update_vertical_field_control!(RP)
    @test RP.vertical_field_control.status == 0 && all(iszero, RP.vertical_field_control.a)
    # a ramp: the status says so, and the coefficient trails the ramped target
    RP, _ = current_disc(; B0 = 2.0e-3, control = IdealVerticalFieldControl(t_on_s = 0.0, ramp_s = 1.0e-4, response = FirstOrderResponse(1.0e-5)))
    RP.dt = 1.0e-6
    for _ in 1:20
        update_vertical_field_control!(RP)
        RP.time_s += RP.dt
    end
    st = RP.vertical_field_control
    @test st.status == 2
    @test 0 < st.a[1] / st.a_target[1] < 0.2
    # beyond coefficient_max the target is clipped and the status saturated
    RP, _ = current_disc(; B0 = 2.0e-3, control = IdealVerticalFieldControl(t_on_s = 0.0, ramp_s = 1.0e-9, response = FirstOrderResponse(1.0e-6), coefficient_max = 1.0e-5))
    RP.dt = 1.0e-6
    for _ in 1:50
        update_vertical_field_control!(RP)
        RP.time_s += RP.dt
    end
    st = RP.vertical_field_control
    @test st.status == 4
    @test abs(st.a[1]) ≈ 1.0e-5 rtol = 1.0e-6
    @test abs(st.a_target[1]) > 1.0e-5
    # the instantaneous response applies the (ramped, clipped) target of the step itself
    RP, _ = current_disc(; B0 = 2.0e-3, control = IdealVerticalFieldControl(t_on_s = 0.0, ramp_s = 1.0e-12, response = InstantaneousResponse()))
    RP.dt = 1.0e-6
    update_vertical_field_control!(RP)
    @test RP.vertical_field_control.a == RP.vertical_field_control.a_target
end

@testitem "ideal vertical field: hysteresis, turning off, and the ramp again on turning on" setup = [CurrentDisc] begin
    using RAPID2D: IdealVerticalFieldControl, update_vertical_field_control!
    # |I| = 10 kA sits between I_off and I_on
    RP, d = current_disc(; B0 = 2.0e-3, control = IdealVerticalFieldControl(t_on_s = 0.0, ramp_s = 2.0e-5, response = RAPID2D.FirstOrderResponse(1.0e-5), I_on = 2.0e4, I_off = 5.0e3))
    RP.dt = 1.0e-6
    st, J0 = RP.vertical_field_control, copy(RP.plasma.Jϕ)
    steps!(n) = for _ in 1:n
        update_vertical_field_control!(RP)
        RP.time_s += RP.dt
    end
    steps!(5)
    @test st.status == 1 && all(iszero, st.a)                   # waiting: below I_on
    RP.plasma.Jϕ .= 3 .* J0                                     # 30 kA: on
    steps!(1)
    t_first = st.t_turned_on
    @test st.qualifies && st.status == 2
    steps!(60)
    RP.plasma.Jϕ .= J0                                          # back to 10 kA: still on
    steps!(5)
    @test st.qualifies && st.status == 3
    RP.plasma.Jϕ .= 0.4 .* J0                                   # 4 kA: below I_off, turning off
    steps!(1)
    @test !st.qualifies && st.status == 5
    steps!(300)
    @test st.status == 1 && all(iszero, st.a)                   # off, coefficients at zero
    RP.plasma.Jϕ .= 3 .* J0                                     # on again, with a new ramp
    steps!(1)
    @test st.status == 2 && st.t_turned_on > t_first
    # a current that cancels over the volume cannot be held: turning off
    steps!(60)
    G, J = RP.G, RP.plasma.Jϕ
    fill!(J, 0.0)
    node(R) = CartesianIndex(round(Int, (R - G.R1D[1]) / G.dR) + 1, round(Int, (0.1 - G.Z1D[1]) / G.dZ) + 1)
    J[node(1.0)] = 2.0 * 3.0e4 / (G.dR * G.dZ)
    J[node(2.0)] = -1.0 * 3.0e4 / (G.dR * G.dZ)
    steps!(1)
    @test !st.qualifies && st.status == 5
end

@testitem "ideal vertical field: the induced field in E∥, Eϕ and the sign of the electrostatic estimate" setup = [CurrentDisc] begin
    using RAPID2D: IdealVerticalFieldControl, write_vertical_control_field!, combine_external_and_self_fields!
    RP, _ = current_disc(; control = IdealVerticalFieldControl(t_on_s = 0.0))
    st, F = RP.vertical_field_control, RP.fields
    RP.flags.E_para_self_ES = true
    RP.fields.Epol_self .= 1.0                                  # a poloidal self field to project
    st.R_ref = 1.6
    st.dt_applied = 1.0e-6
    st.a .= [1.0e-3, 0.0]                                       # a fast rise: Eϕ,ctrl ≈ −500 R V/m reverses Eϕ,ext
    write_vertical_control_field!(RP)
    combine_external_and_self_fields!(RP)
    applied = F.Eϕ_ext .+ F.Eϕ_ctrl
    @test all(<(0), applied) && all(>(0), F.Eϕ_ext)
    @test F.Eϕ ≈ applied .+ F.Eϕ_self
    @test F.E_para_ext ≈ applied .* F.bϕ
    @test F.ψ ≈ F.ψ_ext .+ F.ψ_self .+ F.ψ_ctrl
    @test F.BZ ≈ F.BZ_ext .+ F.BZ_self .+ F.BZ_ctrl
    nonzero = F.E_para_self_ES .!= 0
    @test any(nonzero)
    @test sign.(F.E_para_self_ES[nonzero]) == sign.(-applied[nonzero] .* F.Bϕ[nonzero])
end

@testitem "ideal vertical field: a compensated flux holds the loop voltage at zero at the centroid" setup = [CurrentDisc] begin
    using RAPID2D: IdealVerticalFieldControl, update_vertical_field_control!
    run_control(compensate) = begin
        induction = compensate ? RAPID2D.CSCompensatedInduction() : RAPID2D.FaradayInduction()
        RP, _ = current_disc(; B0 = 2.0e-3, control = IdealVerticalFieldControl(; t_on_s = 0.0, ramp_s = 2.0e-5, response = RAPID2D.FirstOrderResponse(1.0e-5), induction))
        RP.dt = 1.0e-6
        for _ in 1:10
            update_vertical_field_control!(RP)
            RP.time_s += RP.dt
        end
        RP
    end
    plain, compensated = run_control(false), run_control(true)
    st, G, F = compensated.vertical_field_control, compensated.G, compensated.fields
    i = round(Int, (st.R_comp - G.R1D[1]) / G.dR) + 1
    j = round(Int, (st.Z_comp - G.Z1D[1]) / G.dZ) + 1
    @test st.R_comp ≈ R0 atol = 1.0e-12                          # the centroid, on a node
    @test st.Z_comp ≈ Z0 atol = 1.0e-12
    @test abs(F.Eϕ_ctrl[i, j]) < 1.0e-9 * maximum(abs, plain.fields.Eϕ_ctrl)
    @test abs(plain.fields.Eϕ_ctrl[i, j]) > 0
    @test F.BZ_ctrl == plain.fields.BZ_ctrl && F.BR_ctrl == plain.fields.BR_ctrl
end

@testitem "ideal vertical field: the default leaves the code path of a run without control" setup = [CurrentDisc] begin
    using RAPID2D: IdealVerticalFieldControl, update_vertical_field_control!, combine_external_and_self_fields!
    RP, _ = current_disc()
    F = RP.fields
    @test RP.flags.vertical_field_control isa RAPID2D.NoVerticalFieldControl
    update_vertical_field_control!(RP)
    @test all(iszero, F.BZ_ctrl) && all(iszero, F.Eϕ_ctrl)
    # control arrays left with values are not added
    F.BZ_ctrl .= 1.0
    F.Eϕ_ctrl .= 1.0
    combine_external_and_self_fields!(RP)
    @test F.BZ == F.BZ_ext .+ F.BZ_self
    @test F.E_para_ext == F.Eϕ_ext .* F.bϕ
end

@testitem "ideal vertical field: refused with the global J×B kick, without Ampère, and with coils" setup = [CurrentDisc] begin
    using RAPID2D: IdealVerticalFieldControl, update_vertical_field_control!
    control = IdealVerticalFieldControl(t_on_s = 0.0)
    config = SimulationConfig{Float64}(;
        device_Name = "manual", manual = ManualSetup{Float64}(BR = 0.0, BZ = 0.0), NR = 21, NZ = 31,
        R0B0 = 3.0, prefilled_gas_pressure = 0.0, dt = 1.0e-6, snap0D_Δt_s = 1.0e-6,
        snap2D_Δt_s = 1.0e-6, Output_path = mktempdir(),
    )
    RP = RAPID{Float64}(config)
    RP.flags.vertical_field_control = control
    RP.flags.Global_JxB_Force = true
    @test_throws ArgumentError initialize!(RP)
    RP = RAPID{Float64}(config)
    RP.flags.vertical_field_control = control
    RP.flags.Ampere = false
    @test_throws ArgumentError initialize!(RP)
    RP_coil, _ = current_disc(; control, coil_current = 1.0e3)
    @test_throws ArgumentError update_vertical_field_control!(RP_coil)
    @test_throws ArgumentError IdealVerticalFieldControl(t_on_s = 0.0, I_on = 10, I_off = 20)
    @test_throws ArgumentError RAPID2D.FirstOrderResponse(0.0)
    @test_throws ArgumentError IdealVerticalFieldControl(t_on_s = 0.0, coefficient_max = 0.0)
end

@testitem "ideal vertical field: a few steps of a run, and a split run the same bit for bit" setup = [CurrentDisc] begin
    using RAPID2D: IdealVerticalFieldControl
    # the disc with u∥ that gives back its current, controlled from the first step
    make(t_end) = begin
        RP, _ = current_disc(; B0 = 2.0e-3, n = 1.0e17, T = 5.0, control = IdealVerticalFieldControl(t_on_s = 0.0, ramp_s = 3.0e-6, response = RAPID2D.FirstOrderResponse(2.0e-6)))
        pla, F = RP.plasma, RP.fields
        on = pla.ne .> 0
        pla.ue_para[on] .= pla.Jϕ[on] ./ (RP.config.constants.qe .* pla.ne[on] .* F.bϕ[on])
        RP.t_end_s = t_end
        RP
    end
    RP = make(6.0e-6)
    redirect_stdout(devnull) do
        run_simulation!(RP)
    end
    s = RP.diagnostics.snaps0D
    @test s[1].ctrl_status == 0 && s[end].ctrl_status == 3
    residual = [abs(x.ctrl_force_R_residual) for x in s[2:end]]
    @test all(isfinite, residual) && residual[end] < 0.5 * residual[1]
    @test isfinite(s[end].ctrl_energy) && s[end].ctrl_energy != 0
    # each snapshot books the step it follows: power = I × current-weighted loop voltage on the same
    # current and field, and the energy is the sum of power × Δt up to it (a snapshot every step)
    @test all(x -> x.ctrl_power ≈ x.I_tor * x.ctrl_loop_voltage, s[2:end])
    @test s[end].ctrl_energy ≈ sum(x.ctrl_power for x in s[2:end]) * RP.dt
    @test all(x -> isfinite(x.I_tor_endpoint), s)
    # three steps, then three more: the same as six
    split = make(3.0e-6)
    redirect_stdout(devnull) do
        run_simulation!(split)
        split.t_end_s = 6.0e-6
        run_simulation!(split)
    end
    @test split.step == RP.step
    @test split.plasma.ue_para == RP.plasma.ue_para && split.plasma.ne == RP.plasma.ne
    @test split.vertical_field_control.a == RP.vertical_field_control.a
    @test split.vertical_field_control.energy == RP.vertical_field_control.energy
end

@testitem "ideal vertical field: the whole external field's decay index, and the closed region in the 2D snapshot" setup = [CurrentDisc] begin
    using RAPID2D: IdealVerticalFieldControl, update_vertical_field_control!, measure_snap0D!, measure_snap2D!
    # a curved prescribed field and a uniform control: n = −R ∂_R B_Z,ext / (B_Z,ext + a₁)
    RP0, _ = current_disc()
    B0 = column_force_balance(RP0).Bv_req
    RP, _ = current_disc(; B0, control = IdealVerticalFieldControl(t_on_s = 0.0, ramp_s = 1.0e-12, response = RAPID2D.InstantaneousResponse()))
    G, F = RP.G, RP.fields
    n_decay = 0.5
    @. F.BZ_ext = B0 * (R0 / G.R2D)^n_decay
    @. F.BR_ext = -n_decay * B0 * (R0 / G.R2D)^n_decay * G.Z2D / G.R2D
    RP.dt = 1.0e-6
    update_vertical_field_control!(RP)
    snap = Snapshot0D{Float64}()
    measure_snap0D!(RP, snap)
    a₁ = RP.vertical_field_control.a[1]
    @test snap.ctrl_BZ_total_at_cen ≈ snap.BZ_ext_at_cen_by_Jϕ + a₁ rtol = 1.0e-12
    @test snap.ctrl_decay_index_total ≈ snap.decay_index * snap.BZ_ext_at_cen_by_Jϕ / snap.ctrl_BZ_total_at_cen rtol = 1.0e-10
    # the closed region of the disc in its field: the field-line closed nodes, its axis and LCFS
    RP, _ = current_disc(; B0)
    trace_closed_region!(RP)
    snap2 = Snapshot2D{Float64}(; dims_RZ = (RP.G.NR, RP.G.NZ))
    measure_snap2D!(RP, snap2)
    closed = count(k -> RP.G.nodes.state[k] > 0.5, RP.flf.closed_surface_nids)
    @test closed > 0 && count(==(1.0), snap2.closed_lines) == closed
    @test isfinite(snap2.ψ_lcfs) && snap2.ψ_lcfs != snap2.ψ_axis
    @test snap2.R_axis ≈ column_force_balance(RP).R_axis atol = 1.0e-9
    # the wall polygon is written next to the grid
    @test ispath(joinpath(RP.config.Output_path, RP.config.Output_prefix * "wall.bp"))
end
