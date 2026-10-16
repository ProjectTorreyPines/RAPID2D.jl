# The current column's 0D force balance against a uniform current disc (and ellipse): its geometry,
# l_i = 1/2, the hoop and tire-tube forces, the Shafranov field it needs and where it then sits,
# the decay index of a curved external field, a coil's field kept apart from the plasma's, and the
# magnetic axis's response to a mismatched field.

@testsnippet CurrentDisc begin
    using RAPID2D: column_force_balance, solve_Ampere_equation!
    const μ0 = 4π * 1.0e-7
    const R0, Z0 = 1.6, 0.1             # on the nodes of the 81 × 121 grid (ΔR = ΔZ = 0.02 m)

    "A uniform disc of current `I` (radius `a` = 0.205 m, no node on its edge; an ellipse of
    elongation `κ`, its long axis `tilt` from Z) at (R0, Z0), with nₑ = nᵢ = `n` and Tₑ = Tᵢ = `T` on it and `n_out` in a ring
    out to 1.6 a, in the uniform external B_Z = `B0`; the self field solved from the current.
    `coil_current`: one passive loop outside the grid at R = 3 m carrying it. `ampere = false`
    leaves the run without the boundary Green function and the self field."
    function current_disc(;
            I = -1.0e4, B0 = 0.0, n = 1.0e19, T = 10.0, a = 0.205, κ = 1.0, tilt = 0.0, n_out = 0.0,
            coil_current = nothing, ampere = true,
        )
        config = SimulationConfig{Float64}(;
            device_Name = "manual", manual = ManualSetup{Float64}(BR = 0.0, BZ = B0),
            NR = 81, NZ = 121, R0B0 = 3.0, prefilled_gas_pressure = 0.0, dt = 1.0e-6,
            snap0D_Δt_s = 1.0e-6, snap2D_Δt_s = 1.0e-6, Output_path = mktempdir(),
        )
        RP = RAPID{Float64}(config)
        RP.flags.Ampere = ampere                # builds the boundary Green function the self field needs
        initialize!(RP)
        if coil_current !== nothing
            add_coil!(
                RP.coil_system, Coil{Float64}(;
                    location = (r = 3.0, z = 0.0), area = π * 0.05^2, resistance = 1.0e-4,
                    self_inductance = μ0 * 3.0 * (log(8 * 3.0 / 0.05) - 7 / 4),
                    is_powered = false, is_controllable = false, name = "loop",
                )
            )
            initialize_coil_system!(RP)
            RAPID2D.set_all_currents!(RP.coil_system, [coil_current])
        end
        G, pla = RP.G, RP.plasma
        across(k) = (G.R2D[k] - R0) * cos(tilt) - (G.Z2D[k] - Z0) * sin(tilt)
        along(k) = (G.R2D[k] - R0) * sin(tilt) + (G.Z2D[k] - Z0) * cos(tilt)
        ρ²(k) = (across(k) / a)^2 + (along(k) / (κ * a))^2
        disc = [k for k in G.nodes.in_wall_nids if ρ²(k) <= 1]
        ring = [k for k in G.nodes.in_wall_nids if 1 < ρ²(k) <= 1.6^2]
        foreach(fill!, (pla.Jϕ, pla.ne, pla.ni), 0.0)
        pla.Jϕ[disc] .= I / (length(disc) * G.dR * G.dZ)
        pla.ne[disc] .= n
        pla.ni[disc] .= n
        pla.ne[ring] .= n_out
        pla.ni[ring] .= n_out
        fill!(pla.Te_eV, T)
        fill!(pla.Ti_eV, T)
        ampere && solve_Ampere_equation!(RP; update_Eϕ_self = false)
        area = length(disc) * G.dR * G.dZ
        return RP, (; I, area, a = sqrt(area / π), p0 = RP.config.constants.ee * n * 2T)
    end

    "The field-line analysis of the disc in its external field, so the closed region is known."
    function trace_closed_region!(RP)
        RAPID2D.combine_external_and_self_fields!(RP)
        RAPID2D.flf_analysis_field_lines_rz_plane!(RP)
        return RP
    end
end

@testitem "column force balance: a uniform disc's geometry, l_i, β_p, hoop and tire-tube forces" setup = [CurrentDisc] begin
    RP, d = current_disc()
    fb = column_force_balance(RP)
    @test fb.current_coherence ≈ 1
    @test fb.R_cen_by_Jϕ ≈ R0 atol = 1.0e-9
    @test fb.Z_cen_by_Jϕ ≈ Z0 atol = 1.0e-9
    @test fb.a_eff_by_Jϕ ≈ d.a rtol = 0.03
    @test fb.Bpol_edge ≈ μ0 * d.I / (2π * fb.a_eff_by_Jϕ)
    @test fb.li ≈ 0.5 rtol = 0.05                                  # uniform current (0.519 here, 0.498 at twice the resolution)
    @test fb.beta_p ≈ 4 * d.p0 * 2π * R0 * d.area / (μ0 * d.I^2 * R0) rtol = 0.05
    @test fb.pressure_column_share ≈ 1
    # the plasma's own field gives the hoop force of a ring, (μ0 I²/2)(ln(8R/a) − 1 + l_i/2)
    hoop = μ0 * d.I^2 / 2 * (log(8R0 / d.a) - 1 + fb.li / 2)
    @test fb.force_R_self ≈ hoop rtol = 0.02
    @test abs(fb.force_Z_self) < 1.0e-2 * abs(fb.force_R_self)     # an isolated ring: no vertical self-force
    @test fb.force_R_pressure ≈ 2π * d.p0 * d.area rtol = 1.0e-10
    @test fb.force_R_toroidal ≈ μ0 * d.I^2 / 4 * (fb.beta_p - 1)
    @test fb.force_R_coils == 0 && fb.force_Z_coils == 0          # no coils
    # no external field: both estimates of the vertical field to add agree; no equilibrium radius,
    # no decay index, and no closed region traced yet
    @test fb.force_R_ext == 0 && fb.BZ_ext_at_cen_by_Jϕ == 0
    @test fb.Bv_to_add ≈ fb.Bv_req rtol = 0.02
    @test isnan(fb.decay_index) && isnan(fb.R_eq) && isnan(fb.R_eq_stiffness)
    @test isnan(fb.R_axis) && isnan(fb.Z_axis)
end

@testitem "column force balance: in the vertical field it needs, the column sits at its centroid" setup = [CurrentDisc] begin
    RP0, _ = current_disc()
    B0 = column_force_balance(RP0).Bv_req
    RP, d = current_disc(; B0)
    fb = column_force_balance(RP)
    Λ = -4π * R0 * B0 / (μ0 * d.I)
    @test sign(B0) == -sign(d.I)                                   # it pushes the column inward
    @test fb.Bv_req ≈ B0 rtol = 1.0e-12
    @test fb.BZ_ext_at_cen_by_Jϕ ≈ B0 rtol = 1.0e-12
    @test fb.decay_index == 0                                      # a uniform field
    @test fb.decay_index_radial_limit ≈ 1 - 1 / Λ
    @test fb.force_R_ext ≈ 2π * B0 * d.I * R0 rtol = 1.0e-10
    @test fb.force_Z_ext == 0
    @test fb.R_eq ≈ R0 atol = 1.0e-3
    @test fb.R_eq_stiffness ≈ μ0 * d.I^2 / (2R0) * (1 - Λ) rtol = 0.02   # dF_R/dR at n = 0
    @test fb.R_eq_stiffness < 0                                    # restoring
    @test abs(fb.Bv_to_add) < 0.02 * abs(B0)                       # the reduced force estimate agrees
end

@testitem "column force balance: the decay index of a curved field, and its vertical restoring force" setup = [CurrentDisc] begin
    RP0, _ = current_disc()
    B0 = column_force_balance(RP0).Bv_req
    RP, d = current_disc(; B0)
    G, F = RP.G, RP.fields
    n_decay = 0.5
    @. F.BZ_ext = B0 * (R0 / G.R2D)^n_decay
    @. F.BR_ext = -n_decay * B0 * (R0 / G.R2D)^n_decay * G.Z2D / G.R2D   # curl-free to first order in Z
    fb = column_force_balance(RP)
    @test fb.decay_index ≈ n_decay rtol = 0.02
    @test fb.force_Z_ext * Z0 < 0                                  # above the midplane, pushed back down
end

@testitem "column force balance: a coil's field stays out of l_i and the hoop force, as its own force" setup = [CurrentDisc] begin
    using RAPID2D: coil_flux_on_grid, calculate_B_from_ψ!, get_all_currents, _bilinear
    RP0, d = current_disc()
    RP, _ = current_disc(; coil_current = 5.0e3)
    without, with = column_force_balance(RP0), column_force_balance(RP)
    @test with.li == without.li
    @test with.force_R_self == without.force_R_self
    # a loop 1.4 m away is nearly uniform over the disc: 2πR I B_Z,coil at the centroid
    G = RP.G
    BR, BZ = zeros(G.NR, G.NZ), zeros(G.NR, G.NZ)
    calculate_B_from_ψ!(G, coil_flux_on_grid(RP, get_all_currents(RP.coil_system)), BR, BZ)
    @test with.force_R_coils ≈ 2π * R0 * d.I * _bilinear(G, BZ, R0, Z0) rtol = 0.02
    @test with.force_R_coils != 0
    # and it enters the vertical field to add as any other radial force
    ∫J = sum(RP.plasma.Jϕ .* G.inVol2D)
    @test with.Bv_to_add - without.Bv_to_add ≈ -with.force_R_coils / ∫J rtol = 1.0e-10
end

@testitem "column force balance: the magnetic axis moves by a ΔB_Z/B_pa when the field is off by ΔB_Z" setup = [CurrentDisc] begin
    RP0, _ = current_disc()
    B0 = column_force_balance(RP0).Bv_req
    ΔB = 1.0e-3                                                    # a tenth of the column's edge field
    on, d = current_disc(; B0)
    off, _ = current_disc(; B0 = B0 + ΔB)
    fb_on = column_force_balance(trace_closed_region!(on))
    fb_off = column_force_balance(trace_closed_region!(off))
    @test isfinite(fb_on.R_axis) && isfinite(fb_off.R_axis)
    @test fb_on.Z_axis ≈ Z0 atol = 1.0e-3
    # B_Z in excess of what the column needs, with I < 0: the axis moves inward
    shift = fb_off.R_axis - fb_on.R_axis
    @test shift < 0
    @test shift ≈ fb_on.a_eff_by_Jϕ * ΔB / fb_on.Bpol_edge rtol = 0.05     # 0.98 of it on this grid, either sign
end

@testitem "column force balance: elongation, the vertical field to add, and the pressure's support" setup = [CurrentDisc] begin
    RP, _ = current_disc()
    @test column_force_balance(RP).kappa_by_Jϕ ≈ 1 rtol = 0.02
    # an ellipse of κ = 2 (half-widths a, b): its moment ellipse is itself, l_i is its internal
    # inductance 2ab/(a + b)², and the formula on ā = (a + b)/2 follows the force estimate
    a, b = 0.15, 0.3
    RP_ell, _ = current_disc(; a, κ = b / a, n = 0.0)
    fb_ell = column_force_balance(RP_ell)
    @test fb_ell.kappa_by_Jϕ ≈ 2 rtol = 0.05
    @test fb_ell.half_width_minor_by_Jϕ ≈ a rtol = 0.03
    @test fb_ell.half_width_major_by_Jϕ ≈ b rtol = 0.03
    @test fb_ell.li ≈ 2a * b / (a + b)^2 rtol = 0.03
    @test fb_ell.Bv_req ≈ fb_ell.Bv_to_add rtol = 0.01
    # tilted, the column's region still holds all of its pressure; at 45° σ_RR = σ_ZZ
    for tilt in (π / 6, π / 4)
        fb_tilt = column_force_balance(current_disc(; a, κ = b / a, tilt)[1])
        @test fb_tilt.pressure_column_share ≈ 1 atol = 0.02
        @test fb_tilt.half_width_major_by_Jϕ ≈ b rtol = 0.03
        @test fb_tilt.li ≈ fb_ell.li rtol = 0.03
    end
    @test column_force_balance(current_disc(; a, κ = b / a, tilt = π / 4)[1]).kappa_by_Jϕ ≈ 1 rtol = 0.02
    # a uniform B_R: the B_R to add is its opposite
    RP.fields.BR_ext .= 1.0e-4
    @test column_force_balance(RP).Br_to_add ≈ -1.0e-4 rtol = 0.01
    # pressure on the disc alone: the grid's tire tube is the disc's; with pressure around it, more
    @test column_force_balance(current_disc()[1]).force_R_pressure_all ≈ column_force_balance(current_disc()[1]).force_R_pressure
    fb_out = column_force_balance(current_disc(; n_out = 5.0e18)[1])
    @test fb_out.force_R_pressure_all > fb_out.force_R_pressure
    @test fb_out.pressure_column_share < 1
end

@testitem "column force balance: the closed-line current, and the time to the wall if the column could move" setup = [CurrentDisc] begin
    using RAPID2D: _wall_gap_R, bulk_ion_mass
    RP0, _ = current_disc()
    B0 = column_force_balance(RP0).Bv_req
    RP, d = current_disc(; B0)
    fb = column_force_balance(trace_closed_region!(RP))
    # all the current is on closed lines: the closed-line estimate is the column's
    @test fb.I_closed ≈ d.I
    @test fb.Bv_to_add_closed ≈ fb.Bv_to_add rtol = 1.0e-10
    @test fb.force_Z_on_closed ≈ fb.force_Z_ext + fb.force_Z_coils + fb.force_Z_self rtol = 1.0e-10
    # the ion mass on the disc, and the times to close the gap at constant force
    G = RP.G
    F_R = fb.force_R_ext + fb.force_R_coils + fb.force_R_self + fb.force_R_pressure + fb.force_R_toroidal
    @test fb.column_mass ≈ bulk_ion_mass(RP) * 1.0e19 * 2π * R0 * d.area rtol = 1.0e-10
    @test fb.t_wall_inertial ≈ sqrt(2 * fb.column_mass * fb.wall_gap_R / abs(F_R))
    @test fb.nu_in_column == 0 && fb.t_wall_drag == 0                 # no gas, no drag
    # the gap runs to the wall the net force points at: outboard for F > 0, inboard for F < 0
    j = round(Int, (Z0 - G.Z1D[1]) / G.dZ) + 1
    row = [G.R1D[i] for i in 1:G.NR if G.nodes.state[i, j] > 0.5]
    @test _wall_gap_R(G, 1.2, Z0, 1.0) ≈ maximum(row) - 1.2
    @test _wall_gap_R(G, 1.2, Z0, -1.0) ≈ 1.2 - minimum(row)
end

@testitem "column force balance: NaN where nothing exists, the snapshot carries it, edges held" setup = [CurrentDisc] begin
    using RAPID2D: measure_snap0D!, COLUMN_FORCE_BALANCE_FIELDS, _bilinear
    RP, _ = current_disc()
    snap = Snapshot0D{Float64}()
    measure_snap0D!(RP, snap)
    fb = column_force_balance(RP)
    @test all(isequal(getfield(snap, name), fb[name]) for name in COLUMN_FORCE_BALANCE_FIELDS)
    @test isfinite(snap.Bv_req)
    # Ampère off: no self field and no Grad–Shafranov operator, so what needs the plasma's or the
    # coils' field is NaN (and nothing tries to solve for it)
    RP_noamp, _ = current_disc(; ampere = false, coil_current = 5.0e3)
    fb_noamp = column_force_balance(RP_noamp)
    @test isnan(fb_noamp.li) && isnan(fb_noamp.force_R_self) && isnan(fb_noamp.Bv_to_add)
    @test isnan(fb_noamp.force_R_coils)
    @test isfinite(fb_noamp.R_cen_by_Jϕ) && fb_noamp.force_R_ext == 0
    # below the current threshold, nothing
    fill!(RP.plasma.Jϕ, 0.0)
    @test all(isnan, column_force_balance(RP))
    # the bilinear value: exact for a linear field, held at the edge outside the grid
    G = RP.G
    @test _bilinear(G, G.R2D, 1.234, 0.1) ≈ 1.234
    @test _bilinear(G, G.R2D, G.R1D[end] + 1.0, 0.1) == G.R1D[end]
end
