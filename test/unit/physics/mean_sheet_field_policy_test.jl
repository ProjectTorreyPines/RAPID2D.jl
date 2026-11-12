# The policies of the open-line charge sheets' mean field (`flags.mean_sheet_field`): the local pitch
# (the default), the line's least pitch, and the line-averaged saturation ceiling. A policy sets the mean field
# alone; the critical densities, the closures' weights and the mixing amplitude do not follow it.

@testitem "mean sheet field: the default is the local pitch, and a policy moves the mean field alone" setup = [CurrentDisc] begin
    using RAPID2D: estimate_electrostatic_field_effects!, LocalPitch, LineMinPitch, LineAvgSaturation, smooth_data_2D, debye_field
    using RAPID2D.Statistics: mean
    RP, _ = current_disc()
    trace_closed_region!(RP)
    pla, F, G, flf = RP.plasma, RP.fields, RP.G, RP.flf
    closed = flf.closed_surface_nids
    open_in = setdiff(G.nodes.in_wall_nids, closed)
    pla.ne[open_in] .= 1.0e17
    on_wall = Set(G.nodes.on_wall_nids)
    interior = [k for k in open_in if !(k in on_wall)]
    Bpol_avg = mean(F.Bpol[G.nodes.in_wall_nids])
    kept() = map(copy, (pla.nc_para, pla.nc_perp, pla.ambipolar_weight, pla.turbulence_weight, F.Erad_fluc, RP.transport.Dpol_turb))

    @test RP.flags.mean_sheet_field isa LocalPitch
    estimate_electrostatic_field_effects!(RP)
    mean0, kept0 = copy(F.Epol_mean), kept()

    E_Debye = debye_field(RP, smooth_data_2D(pla.ne; num_SM = 2, weighting = G.inVol2D))
    E_req = abs.(F.Eϕ_para_presc .+ F.Eϕ_para_coils_pla) .* F.Btot ./ F.Bpol
    for policy in (LineMinPitch(), LineAvgSaturation(), LineAvgSaturation(p = 4.0), LineAvgSaturation(coefficient = 3.0))
        RP.flags.mean_sheet_field = policy
        estimate_electrostatic_field_effects!(RP)
        @test all(isfinite, F.Epol_mean) && all(>=(0), F.Epol_mean)
        # never above the requirement nor above what the plasma holds; none on a closed line
        @test all(k -> F.Epol_mean[k] <= min(E_req[k], E_Debye[k]) * (1 + 1.0e-12), interior)
        @test all(iszero, F.Epol_mean[closed])
        @test any(k -> F.Epol_mean[k] != mean0[k], interior)
        # nothing else follows the policy
        @test all(kept() .== kept0)
    end

    # the line's least pitch: the local closure with the smallest B_pol met along the node's line
    RP.flags.mean_sheet_field = LineMinPitch()
    estimate_electrostatic_field_effects!(RP)
    @test F.Epol_mean[interior] ≈ min.(min.(1.0, 0.65 .* flf.min_Bpol[interior] ./ Bpol_avg) .* E_req[interior], E_Debye[interior])
    @test all(k -> F.Epol_mean[k] <= mean0[k] * (1 + 1.0e-12), interior)
    @test any(k -> F.Epol_mean[k] < 0.9 * mean0[k], interior)

    # back to the default: bit for bit what it was
    RP.flags.mean_sheet_field = LocalPitch()
    estimate_electrostatic_field_effects!(RP)
    @test F.Epol_mean == mean0 && all(kept() .== kept0)
end

@testitem "mean sheet field: the saturation ceiling is the requirement at the mean B_pol times the line's factor" setup = [CurrentDisc] begin
    using RAPID2D: estimate_electrostatic_field_effects!, LineAvgSaturation, flf_analysis_field_lines_rz_plane!,
        sheet_mixing_integrand, mean_sheet_line_integrand
    using RAPID2D.Statistics: mean
    RP, _ = current_disc()
    trace_closed_region!(RP)
    pla, F, G, flf = RP.plasma, RP.fields, RP.G, RP.flf
    open_in = setdiff(G.nodes.in_wall_nids, flf.closed_surface_nids)
    pla.ne[open_in] .= 1.0e22                   # the Debye cap out of the way
    on_wall = Set(G.nodes.on_wall_nids)
    measured = [k for k in open_in if !(k in on_wall) && isfinite(flf.Lpol_tot[k]) && flf.Lpol_tot[k] > 0 && isfinite(flf.Lc_tot[k])]
    Bpol_avg = mean(F.Bpol[G.nodes.in_wall_nids])
    E_drive() = abs.(F.Eϕ_para_presc .+ F.Eϕ_para_coils_pla)
    E_req() = E_drive() .* F.Btot ./ F.Bpol
    E0() = E_drive() .* F.Btot ./ Bpol_avg
    follow!() = flf_analysis_field_lines_rz_plane!(RP; integrand = sheet_mixing_integrand(RP), second_integrand = mean_sheet_line_integrand(RP))
    mean_of = Dict{Float64, Matrix{Float64}}()
    for p in (2.0, 4.0)
        RP.flags.mean_sheet_field = LineAvgSaturation(; p)
        # the following integrates (⟨B_pol⟩/B_pol)^p along every line, beside the mixing's field
        follow!()
        @test flf.has_second_integral
        estimate_electrostatic_field_effects!(RP)
        B_eff = Bpol_avg .* (flf.second_integral_tot ./ flf.Lpol_tot) .^ (-1 / p)
        ceiling = E0() .* B_eff ./ (B_eff .+ Bpol_avg)
        @test F.Epol_mean[measured] ≈ min.(E_req(), ceiling)[measured]
        # a mean of 1/B_pol weighs the weak stretches: never below the line's least B_pol
        @test all(k -> flf.min_Bpol[k] * (1 - 1.0e-9) <= B_eff[k], measured)
        # the ceiling binds somewhere and the requirement elsewhere
        @test any(k -> ceiling[k] < E_req()[k], measured)
        mean_of[p] = copy(F.Epol_mean)
    end
    # a larger p weighs the line's weakest stretch more: a lower ceiling, toward the least pitch
    @test all(k -> mean_of[4.0][k] <= mean_of[2.0][k] * (1 + 1.0e-9), measured)
    @test any(k -> mean_of[4.0][k] < 0.99 * mean_of[2.0][k], measured)

    # no spike where B_pol is weakest: the field there is bounded by the requirement at the mean B_pol, which
    # the requirement itself far exceeds
    RP.flags.mean_sheet_field = LineAvgSaturation()
    follow!()
    estimate_electrostatic_field_effects!(RP)
    weakest = measured[argmin(F.Bpol[measured])]
    @test all(k -> F.Epol_mean[k] <= E0()[k] * (1 + 1.0e-12), measured)
    @test F.Epol_mean[weakest] < E_req()[weakest]

    # the coefficient scales the ceiling, and the requirement still bounds the field
    mean_one = copy(F.Epol_mean)
    for coefficient in (0.5, 3.0)
        RP.flags.mean_sheet_field = LineAvgSaturation(; coefficient)
        estimate_electrostatic_field_effects!(RP)
        @test F.Epol_mean[measured] ≈ min.(E_req(), coefficient .* E0() .* (Bpol_avg .* (flf.second_integral_tot ./ flf.Lpol_tot) .^ (-1 / 2)) ./ ((Bpol_avg .* (flf.second_integral_tot ./ flf.Lpol_tot) .^ (-1 / 2)) .+ Bpol_avg))[measured]
        monotone = coefficient < 1 ? (<=) : (>=)
        @test all(k -> monotone(F.Epol_mean[k], mean_one[k] * (coefficient < 1 ? 1 + 1.0e-12 : 1 - 1.0e-12)), measured)
    end

    # the ceiling is the field's geometry times the present drive: the mean field scales with the drive
    RP.flags.mean_sheet_field = LineAvgSaturation()
    estimate_electrostatic_field_effects!(RP)
    F.Eϕ_para_presc .*= 2
    estimate_electrostatic_field_effects!(RP)
    @test F.Epol_mean[measured] ≈ 2 .* mean_one[measured]

    # a thin plasma: the Debye field caps it
    pla.ne[open_in] .= 1.0e10
    estimate_electrostatic_field_effects!(RP)
    @test any(k -> F.Epol_mean[k] < 0.5 * 2 * mean_one[k], measured)
    pla.ne[open_in] .= 1.0e22

    # a line whose integral diverged passes a null: a measured zero, no ceiling to hold, not the node's own B_pol
    pla.ne[open_in] .= 1.0e22
    kept_integral = flf.second_integral_tot[weakest]
    flf.second_integral_tot[weakest] = Inf
    estimate_electrostatic_field_effects!(RP)
    @test iszero(F.Epol_mean[weakest])
    flf.second_integral_tot[weakest] = kept_integral

    # the integrand is finite and scaled where B_pol vanishes or is tiny, in single precision too
    @test all(isfinite, mean_sheet_line_integrand(RP))
    Bpol_kept = F.Bpol[weakest]
    F.Bpol[weakest] = 0.0
    @test all(isfinite, mean_sheet_line_integrand(RP)) && mean_sheet_line_integrand(RP)[weakest] ≈ 1 / eps(Float64)
    F.Bpol[weakest] = Bpol_kept

    # a following without the second field: B/⟨B/B_pol⟩ from the lengths, an approximation until the next one
    flf_analysis_field_lines_rz_plane!(RP; integrand = sheet_mixing_integrand(RP))
    @test !flf.has_second_integral && all(iszero, flf.second_integral_tot)
    estimate_electrostatic_field_effects!(RP)
    B_len = F.Btot .* flf.Lpol_tot ./ flf.Lc_tot
    @test F.Epol_mean[measured] ≈ min.(E_req(), E0() .* B_len ./ (B_len .+ Bpol_avg))[measured]
end

@testitem "mean sheet field: in a uniform poloidal field the least pitch is the local one, and the ceiling cancels half the coefficient" begin
    using RAPID2D: estimate_electrostatic_field_effects!, run_simulation!, LocalPitch, LineMinPitch, LineAvgSaturation
    function vertical(policy)
        config = SimulationConfig{Float64}(;
            device_Name = "manual", manual = ManualSetup{Float64}(BR = 0.0, BZ = 1.0e-3, Eϕ = 0.5),
            NR = 17, NZ = 17, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(),
        )
        RP = RAPID{Float64}(config)
        RP.flags.mean_sheet_field = policy
        initialize!(RP)
        RP.plasma.ne .= 1.0e17
        return RP
    end
    interior(RP) = setdiff(RP.G.nodes.in_wall_nids, RP.G.nodes.on_wall_nids)
    ref = vertical(LocalPitch())
    estimate_electrostatic_field_effects!(ref)
    k = interior(ref)
    # every line sees one B_pol: its least pitch is the node's
    line = vertical(LineMinPitch())
    estimate_electrostatic_field_effects!(line)
    @test line.fields.Epol_mean[k] ≈ ref.fields.Epol_mean[k]
    # every line's B_pol is the device's mean: the ceiling is coefficient/2 of the requirement, the requirement
    # itself from coefficient 2 on; at the setup's following and after a step's
    cancelled(RP) = (F = RP.fields; (F.Epol_mean .* F.Bpol ./ (abs.(F.Eϕ_para_presc) .* F.Btot))[k])
    for p in (2.0, 4.0), coefficient in (0.5, 1.0, 3.0)
        soft = vertical(LineAvgSaturation(; p, coefficient))
        @test soft.flf.has_second_integral
        soft.plasma.ne .= 1.0e22                   # the Debye cap out of the way
        estimate_electrostatic_field_effects!(soft)
        @test all(x -> isapprox(x, min(1, coefficient / 2); rtol = 1.0e-6), cancelled(soft))
        soft.t_end_s = soft.time_s + soft.dt
        run_simulation!(soft)
        @test soft.flf.has_second_integral
        soft.plasma.ne .= 1.0e22
        estimate_electrostatic_field_effects!(soft)
        @test all(x -> isapprox(x, min(1, coefficient / 2); rtol = 1.0e-6), cancelled(soft))
    end
    # the local pitch asks for no second field
    @test !ref.flf.has_second_integral
end

@testitem "mean sheet field: finite and empty where there is no poloidal field or no drive, and the parameters are checked" begin
    using RAPID2D: estimate_electrostatic_field_effects!, LineMinPitch, LineAvgSaturation
    @test_throws ArgumentError LineAvgSaturation(p = 0.5)
    @test_throws ArgumentError LineAvgSaturation(p = Inf)
    @test_throws ArgumentError LineAvgSaturation(p = big"1e400")
    @test_throws ArgumentError LineAvgSaturation(coefficient = 0.0)
    @test_throws ArgumentError LineAvgSaturation(coefficient = -1.0)
    @test_throws ArgumentError LineAvgSaturation(coefficient = NaN)
    @test LineAvgSaturation() == LineAvgSaturation(2, 1) && LineAvgSaturation().p === 2.0
    for policy in (LineMinPitch(), LineAvgSaturation()), (BZ, Eϕ) in ((0.0, 0.5), (1.0e-3, 0.0))
        config = SimulationConfig{Float64}(;
            device_Name = "manual", manual = ManualSetup{Float64}(; BR = 0.0, BZ, Eϕ),
            NR = 11, NZ = 11, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, Output_path = mktempdir(),
        )
        RP = RAPID{Float64}(config)
        RP.flags.mean_sheet_field = policy
        initialize!(RP)
        RP.plasma.ne .= 1.0e17
        estimate_electrostatic_field_effects!(RP)
        F, pla = RP.fields, RP.plasma
        @test all(iszero, F.Epol_mean)
        @test all(a -> all(isfinite, a), (F.Epol_mean, F.Epol_para_pla, F.E_para_tot, pla.mean_ExB_R, pla.mean_ExB_Z))
    end
end
