# The policies of the open-line charge sheets' mean field (`flags.mean_sheet_field`): the local pitch
# (the default), the line's least pitch, and the soft line average. A policy sets the shape γ of the mean
# field alone; the critical densities, the closures' weights and the mixing amplitude do not follow it.

@testitem "mean sheet field: the default is the local pitch, and a policy moves the mean field alone" setup = [CurrentDisc] begin
    using RAPID2D: estimate_electrostatic_field_effects!, LocalPitch, LineMinPitch, LineAvgPitch, LineAvgRequirement, smooth_data_2D, debye_field
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
    mean0, γ0_map, kept0 = copy(F.Epol_mean), copy(pla.γ_shape_fac), kept()
    @test γ0_map ≈ min.(1.0, 0.65 .* F.Bpol ./ Bpol_avg)

    E_Debye = debye_field(RP, smooth_data_2D(pla.ne; num_SM = 2, weighting = G.inVol2D))
    E_req = abs.(F.Eϕ_para_presc .+ F.Eϕ_para_coils_pla) .* F.Btot ./ F.Bpol
    for policy in (LineMinPitch(), LineAvgPitch(), LineAvgPitch(p = 4.0), LineAvgRequirement(), LineAvgRequirement(p = 4.0))
        RP.flags.mean_sheet_field = policy
        estimate_electrostatic_field_effects!(RP)
        γ = pla.γ_shape_fac
        @test all(x -> 0 <= x <= 1, γ) && all(isfinite, F.Epol_mean)
        # the mean field is the shaped requirement, capped at what the plasma holds; none on a closed line
        @test F.Epol_mean[interior] ≈ min.(γ[interior] .* E_req[interior], E_Debye[interior])
        @test all(iszero, F.Epol_mean[closed])
        @test any(k -> F.Epol_mean[k] != mean0[k], interior)
        # nothing else follows the policy
        @test all(kept() .== kept0)
    end

    # the line's least pitch: the local closure with the smallest B_pol met along the node's line
    RP.flags.mean_sheet_field = LineMinPitch()
    estimate_electrostatic_field_effects!(RP)
    @test pla.γ_shape_fac[open_in] ≈ min.(1.0, 0.65 .* flf.min_Bpol[open_in] ./ Bpol_avg)
    @test all(k -> pla.γ_shape_fac[k] <= γ0_map[k] * (1 + 1.0e-12), open_in)
    @test any(k -> pla.γ_shape_fac[k] < 0.9 * γ0_map[k], open_in)

    # back to the default: bit for bit what it was
    RP.flags.mean_sheet_field = LocalPitch()
    estimate_electrostatic_field_effects!(RP)
    @test F.Epol_mean == mean0 && pla.γ_shape_fac == γ0_map && all(kept() .== kept0)
end

@testitem "mean sheet field: the averaged pitch is the line's effective B_pol against the device's mean" setup = [CurrentDisc] begin
    using RAPID2D: estimate_electrostatic_field_effects!, LineAvgPitch, flf_analysis_field_lines_rz_plane!,
        sheet_mixing_integrand, mean_sheet_line_integrand
    using RAPID2D.Statistics: mean
    RP, _ = current_disc()
    trace_closed_region!(RP)
    pla, F, G, flf = RP.plasma, RP.fields, RP.G, RP.flf
    open_in = setdiff(G.nodes.in_wall_nids, flf.closed_surface_nids)
    pla.ne[open_in] .= 1.0e17
    on_wall = Set(G.nodes.on_wall_nids)
    measured = [k for k in open_in if !(k in on_wall) && isfinite(flf.Lpol_tot[k]) && flf.Lpol_tot[k] > 0 && isfinite(flf.Lc_tot[k])]
    Bpol_avg = mean(F.Bpol[G.nodes.in_wall_nids])
    γ_of = Dict{Float64, Matrix{Float64}}()
    for p in (2.0, 4.0)
        RP.flags.mean_sheet_field = LineAvgPitch(; p)
        # the following integrates B_pol^-p along every line, beside the mixing's field
        flf_analysis_field_lines_rz_plane!(RP; integrand = sheet_mixing_integrand(RP), second_integrand = mean_sheet_line_integrand(RP))
        @test flf.has_second_integral
        estimate_electrostatic_field_effects!(RP)
        B_eff = (flf.second_integral_tot ./ flf.Lpol_tot) .^ (-1 / p)
        @test pla.γ_shape_fac[measured] ≈ (B_eff ./ (B_eff .+ Bpol_avg))[measured]
        # a mean of 1/B_pol weighs the weak stretches: never above the line's largest B_pol, never below its least
        @test all(k -> flf.min_Bpol[k] * (1 - 1.0e-9) <= B_eff[k], measured)
        γ_of[p] = copy(pla.γ_shape_fac)
    end
    # a larger p weighs the line's weakest stretch more: less cancelled, toward the least pitch
    @test all(k -> γ_of[4.0][k] <= γ_of[2.0][k] * (1 + 1.0e-9), measured)
    @test any(k -> γ_of[4.0][k] < 0.99 * γ_of[2.0][k], measured)

    # the shape is the field's geometry: the drive's size does not enter it, the mean field scales with it
    γ_before = copy(pla.γ_shape_fac)
    pla.ne[open_in] .= 1.0e22                   # the Debye cap out of the way
    estimate_electrostatic_field_effects!(RP)
    mean_before = copy(F.Epol_mean)
    F.Eϕ_para_presc .*= 2
    estimate_electrostatic_field_effects!(RP)
    @test pla.γ_shape_fac == γ_before
    @test F.Epol_mean[measured] ≈ 2 .* mean_before[measured]

    # a following without the second field: the mean of B/B_pol the lengths hold (p = 1)
    flf_analysis_field_lines_rz_plane!(RP; integrand = sheet_mixing_integrand(RP))
    @test !flf.has_second_integral && all(iszero, flf.second_integral_tot)
    estimate_electrostatic_field_effects!(RP)
    B_len = F.Btot .* flf.Lpol_tot ./ flf.Lc_tot
    @test pla.γ_shape_fac[measured] ≈ (B_len ./ (B_len .+ Bpol_avg))[measured]
end

@testitem "mean sheet field: the averaged requirement is the node's at the mean B_pol against the line's" setup = [CurrentDisc] begin
    using RAPID2D: estimate_electrostatic_field_effects!, LineAvgRequirement, flf_analysis_field_lines_rz_plane!,
        sheet_mixing_integrand, mean_sheet_line_integrand
    using RAPID2D.Statistics: mean
    RP, _ = current_disc()
    trace_closed_region!(RP)
    pla, F, G, flf = RP.plasma, RP.fields, RP.G, RP.flf
    open_in = setdiff(G.nodes.in_wall_nids, flf.closed_surface_nids)
    pla.ne[open_in] .= 1.0e17
    on_wall = Set(G.nodes.on_wall_nids)
    measured = [k for k in open_in if !(k in on_wall) && isfinite(flf.Lpol_tot[k]) && flf.Lpol_tot[k] > 0 && isfinite(flf.Lc_tot[k])]
    Bpol_avg = mean(F.Bpol[G.nodes.in_wall_nids])
    follow!() = flf_analysis_field_lines_rz_plane!(RP; integrand = sheet_mixing_integrand(RP), second_integrand = mean_sheet_line_integrand(RP))
    E_drive() = abs.(F.Eϕ_para_presc .+ F.Eϕ_para_coils_pla)
    γ_of = Dict{Float64, Matrix{Float64}}()
    for p in (2.0, 4.0)
        RP.flags.mean_sheet_field = LineAvgRequirement(; p)
        # the following integrates the uncapped requirement to the power p along every line, beside the mixing's field
        follow!()
        @test flf.has_second_integral
        estimate_electrostatic_field_effects!(RP)
        M = (flf.second_integral_tot ./ flf.Lpol_tot) .^ (1 / p)
        E0 = E_drive() .* F.Btot ./ Bpol_avg
        @test pla.γ_shape_fac[measured] ≈ (E0 ./ (E0 .+ M))[measured]
        @test all(k -> 0 < pla.γ_shape_fac[k] < 1, measured)
        γ_of[p] = copy(pla.γ_shape_fac)
    end
    # a larger p weighs the line's weakest stretch more: less cancelled, toward the least pitch
    @test all(k -> γ_of[4.0][k] <= γ_of[2.0][k] * (1 + 1.0e-9), measured)
    @test any(k -> γ_of[4.0][k] < 0.99 * γ_of[2.0][k], measured)

    # the second integral is the requirement's own, not the mixing's capped one: a thin plasma lowers the mixing's
    # integral and leaves this one and the shape as they were
    second, mixing, γ_dense = copy(flf.second_integral_tot), copy(flf.integral_tot), copy(pla.γ_shape_fac)
    pla.ne[open_in] .= 1.0e12
    follow!()
    estimate_electrostatic_field_effects!(RP)
    @test flf.second_integral_tot == second && pla.γ_shape_fac == γ_dense
    @test any(k -> flf.integral_tot[k] < 0.99 * mixing[k], measured)

    # the drive's size does not enter the shape once the following has seen it; the mean field scales with it
    pla.ne[open_in] .= 1.0e22                   # the Debye cap out of the way
    estimate_electrostatic_field_effects!(RP)
    mean_before = copy(F.Epol_mean)
    F.Eϕ_para_presc .*= 2
    follow!()
    estimate_electrostatic_field_effects!(RP)
    @test pla.γ_shape_fac[measured] ≈ γ_dense[measured]
    @test F.Epol_mean[measured] ≈ 2 .* mean_before[measured]

    # a following without the second field: the line's EMF over its RZ length with the node's drive (p = 1)
    flf_analysis_field_lines_rz_plane!(RP; integrand = sheet_mixing_integrand(RP))
    @test !flf.has_second_integral && all(iszero, flf.second_integral_tot)
    estimate_electrostatic_field_effects!(RP)
    E0 = E_drive() .* F.Btot ./ Bpol_avg
    M1 = E_drive() .* flf.Lc_tot ./ flf.Lpol_tot
    @test pla.γ_shape_fac[measured] ≈ (E0 ./ (E0 .+ M1))[measured]
end

@testitem "mean sheet field: in a uniform poloidal field the least pitch is the local one, and both averages give one half" begin
    using RAPID2D: estimate_electrostatic_field_effects!, run_simulation!, LocalPitch, LineMinPitch, LineAvgPitch, LineAvgRequirement
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
    @test line.plasma.γ_shape_fac[k] ≈ ref.plasma.γ_shape_fac[k]
    @test line.fields.Epol_mean[k] ≈ ref.fields.Epol_mean[k]
    # every line's requirement is its nodes' own, and B_pol the device's mean: half the requirement, at the setup's
    # following and after a step's
    for policy in (LineAvgPitch, LineAvgRequirement), p in (2.0, 4.0)
        soft = vertical(policy(; p))
        @test soft.flf.has_second_integral
        estimate_electrostatic_field_effects!(soft)
        @test all(x -> isapprox(x, 0.5; rtol = 1.0e-6), soft.plasma.γ_shape_fac[k])
        soft.t_end_s = soft.time_s + soft.dt
        run_simulation!(soft)
        @test soft.flf.has_second_integral
        @test all(x -> isapprox(x, 0.5; rtol = 1.0e-6), soft.plasma.γ_shape_fac[k])
    end
    # the local pitch asks for no second field
    @test !ref.flf.has_second_integral
end

@testitem "mean sheet field: finite and empty where there is no poloidal field, and p must be at least one" begin
    using RAPID2D: estimate_electrostatic_field_effects!, LineMinPitch, LineAvgPitch, LineAvgRequirement
    @test_throws ArgumentError LineAvgPitch(p = 0.5)
    @test_throws ArgumentError LineAvgPitch(p = Inf)
    @test_throws ArgumentError LineAvgRequirement(p = 0.5)
    for policy in (LineMinPitch(), LineAvgPitch(), LineAvgRequirement())
        config = SimulationConfig{Float64}(;
            device_Name = "manual", manual = ManualSetup{Float64}(BR = 0.0, BZ = 0.0, Eϕ = 0.5),
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
        @test all(iszero, pla.γ_shape_fac) && all(iszero, F.Epol_mean)
        @test all(a -> all(isfinite, a), (F.Epol_mean, F.Epol_para_pla, F.E_para_tot, pla.mean_ExB_R, pla.mean_ExB_Z))
    end
end
