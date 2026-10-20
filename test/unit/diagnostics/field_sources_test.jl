# The coils' and the plasma's fields by source in the 2D snapshots (`measure_field_sources!`):
# each from its own present current, their closure against the solved ψ_coils_pla, the induced
# field split into the coils' part and the rest, the step's coil-current change it uses, and old
# snapshot names read into the new ones.

@testitem "field sources: after a Grad–Shafranov solve the sources add up to ψ_coils_pla" setup = [CurrentDisc] begin
    using RAPID2D: measure_field_sources!, plasma_flux_on_grid
    # the disc and a loop outside the grid, ψ_coils_pla solved from both (the fixture's solve)
    RP, _ = current_disc(; coil_current = 5.0e3)
    F = RP.fields
    snap = Snapshot2D{Float64}(; dims_RZ = (RP.G.NR, RP.G.NZ))
    measure_field_sources!(RP, snap)
    @test maximum(abs, snap.ψ_coils) > 0 && maximum(abs, snap.ψ_pla) > 0
    # the same source currents: only the rounding of separate solves is left
    @test snap.ψ_closure_rel < 1.0e-10
    @test snap.ψ_closure_abs ≈ maximum(abs, F.ψ_coils_pla .- snap.ψ_pla .- snap.ψ_coils)
    scale = maximum(abs, F.BZ_coils_pla)
    @test maximum(abs, snap.BZ_pla .+ snap.BZ_coils .- F.BZ_coils_pla) < 1.0e-10 * scale
    @test maximum(abs, snap.BR_pla .+ snap.BR_coils .- F.BR_coils_pla) < 1.0e-10 * scale
    @test snap.ψ_pla == plasma_flux_on_grid(RP, RP.plasma.Jϕ)
    # before any step: the coil added after initialize! has no change, nothing is split off
    @test RP.coil_system.ΔI_step == [0.0]
    @test all(iszero, snap.Eϕ_coils) && snap.Eϕ_pla == F.Eϕ_coils_pla
    # no coils: their fields are zero, the plasma's is the whole
    RP0, _ = current_disc()
    measure_field_sources!(RP0, snap)
    @test all(iszero, snap.ψ_coils) && snap.ψ_closure_rel < 1.0e-10
    # without Ampère nothing is solved: NaN
    RP_off, _ = current_disc(; ampere = false)
    measure_field_sources!(RP_off, snap)
    @test all(isnan, snap.ψ_pla) && all(isnan, snap.ψ_coils) && all(isnan, snap.Eϕ_pla)
    @test isnan(snap.ψ_closure_rel)
end

@testitem "field sources: below the gate the induced field is the coils', the plasma's part zero" setup = [CoilFluxColumn] begin
    using RAPID2D: measure_field_sources!
    # a 10 V coil changes below the gate: the induced field is its alone
    RP = column_with_loops([(0.6, 0.0, 1.0e-4, 10.0, "OH")]; t_end = 25.0e-6)
    RP.flags.Ampere_Itor_threshold = 1.0e6
    quiet(() -> run_simulation!(RP))
    snap = Snapshot2D{Float64}(; dims_RZ = (RP.G.NR, RP.G.NZ))
    measure_field_sources!(RP, snap)
    @test any(!iszero, RP.coil_system.ΔI_step)
    @test snap.Eϕ_coils == RP.fields.Eϕ_coils_pla
    @test all(iszero, snap.Eϕ_pla)
    # without the induced field (E_para_self_EM off) the split is not defined
    RP = column_with_loops([(0.6, 0.0, 1.0e-4, 10.0, "OH")]; t_end = 25.0e-6)
    RP.flags.Ampere_Itor_threshold = 1.0e6
    RP.flags.E_para_self_EM = false
    quiet(() -> run_simulation!(RP))
    measure_field_sources!(RP, snap)
    @test all(isnan, snap.Eϕ_coils) && all(isnan, snap.Eϕ_pla)
    @test all(isfinite, snap.ψ_pla)
end

@testitem "field sources: above the gate, the step's coil-current change, a split run, the closure" setup = [CoilFluxColumn] begin
    using RAPID2D: measure_field_sources!, get_all_currents, coil_flux_on_grid, add_coil!
    # a column carrying current beside a loop inside the grid, coupled from the first step
    RP = column_with_loops([(1.2, 0.8, 1.0e-3, 0.0, "loop")]; t_end = 20.0e-6)
    in_column = RP.plasma.ne .> 0
    RP.plasma.ne[in_column] .= 1.0e18
    RP.plasma.ni .= RP.plasma.ne
    RP.plasma.ue_para[in_column] .= -2.5e5
    quiet(() -> run_simulation!(RP))
    # one more step: ΔI_step is its change, and it outlives the run
    I_before = get_all_currents(RP.coil_system)
    RP.t_end_s += RP.dt
    quiet(() -> run_simulation!(RP))
    @test RP.coil_system.ΔI_step == get_all_currents(RP.coil_system) .- I_before
    @test any(!iszero, RP.coil_system.ΔI_step)
    snap = Snapshot2D{Float64}(; dims_RZ = (RP.G.NR, RP.G.NZ))
    measure_field_sources!(RP, snap)
    # the coils' part is the field of the step's current change, computed here on its own; the
    # plasma's is the rest of the field the model applied, and not zero above the gate
    E_coils = -coil_flux_on_grid(RP, RP.coil_system.ΔI_step) ./ (RP.G.R2D .* RP.dt)
    @test snap.Eϕ_coils == E_coils
    @test snap.Eϕ_pla == RP.fields.Eϕ_coils_pla .- E_coils
    @test maximum(abs, snap.Eϕ_pla) > 0 && maximum(abs, E_coils) > 0
    # the solved sum and the present currents differ by what the outer iteration and the
    # updates after it leave: small, not zero
    @test isfinite(snap.ψ_closure_rel) && snap.ψ_closure_rel < 1.0e-2
    # the 2D snapshots of the run carry them
    s = RP.diagnostics.snaps2D[end]
    @test all(isfinite, s.ψ_pla) && all(isfinite, s.ψ_coils) && isfinite(s.ψ_closure_abs)
    # a coil added after the run had no change over the last step: the others' split stands
    ΔI_last = copy(RP.coil_system.ΔI_step)
    loop2 = deepcopy(RP.coil_system.coils[1])
    loop2.location, loop2.name, loop2.current = (r = 1.2, z = -0.8), "loop2", 0.0
    add_coil!(RP.coil_system, loop2)
    initialize_coil_system!(RP)
    @test RP.coil_system.ΔI_step == [ΔI_last; 0.0]
    measure_field_sources!(RP, snap)
    @test snap.Eϕ_coils == E_coils
    # a coil set changed without add_coil!: the split is unknown, NaN
    pop!(RP.coil_system.ΔI_step)
    measure_field_sources!(RP, snap)
    @test all(isnan, snap.Eϕ_coils) && all(isnan, snap.Eϕ_pla) && all(isfinite, snap.ψ_pla)
end

@testitem "field sources: old snapshot names are read into the new ones" begin
    using RAPID2D: create_symbol_path_mapping, process_symbol_based_snapshots!
    dims_RZ = (4, 5)
    old = reshape(collect(1.0:20.0), dims_RZ..., 1)
    Adict = Dict{String, Any}("time_s" => [0.1], "ψ_self" => old, "BZ_self" => 2 .* old)
    snaps = [Snapshot2D{Float64}(; dims_RZ)]
    paths = create_symbol_path_mapping(Snapshot2D{Float64}, Adict)
    @test paths["ψ_self"] == [:ψ_coils_pla]
    process_symbol_based_snapshots!(snaps, Adict, paths, 1)
    @test snaps[1].ψ_coils_pla == old[:, :, 1] && snaps[1].BZ_coils_pla == 2 .* old[:, :, 1]
    # a file with the new name as well keeps the new one
    Adict["ψ_coils_pla"] = 3 .* old
    @test !haskey(create_symbol_path_mapping(Snapshot2D{Float64}, Adict), "ψ_self")
    # the 0D force names
    paths0 = create_symbol_path_mapping(Snapshot0D{Float64}, Dict{String, Any}("time_s" => [0.1], "force_R_self" => [1.0]))
    @test paths0["force_R_self"] == [:force_R_pla]
end

@testitem "field sources: old prescribed-field names are read into the new ones" begin
    using RAPID2D: create_symbol_path_mapping, process_symbol_based_snapshots!, RENAMED_SNAPSHOT_FIELDS
    # every name a field was renamed to is a snapshot field
    @test all(values(RENAMED_SNAPSHOT_FIELDS)) do n
        hasfield(Snapshot0D{Float64}, Symbol(n)) || hasfield(Snapshot2D{Float64}, Symbol(n))
    end
    dims_RZ = (4, 5)
    old = reshape(collect(1.0:20.0), dims_RZ..., 1)
    Adict = Dict{String, Any}("time_s" => [0.1], "ψ_ext" => old, "LV_ext" => 2 .* old, "E_para_ext" => 3 .* old)
    snaps = [Snapshot2D{Float64}(; dims_RZ)]
    process_symbol_based_snapshots!(snaps, Adict, create_symbol_path_mapping(Snapshot2D{Float64}, Adict), 1)
    @test snaps[1].ψ_presc == old[:, :, 1] && snaps[1].LV_presc == 2 .* old[:, :, 1]
    @test snaps[1].E_para_presc == 3 .* old[:, :, 1]
    # the 0D names
    old0D = (
        "Epara_ext" => :Epara_presc, "force_R_ext" => :force_R_presc, "force_Z_ext" => :force_Z_presc,
        "BZ_ext_at_cen_by_Jϕ" => :BZ_presc_at_cen_by_Jϕ, "BR_ext_at_cen_by_Jϕ" => :BR_presc_at_cen_by_Jϕ,
        "coils_V_ext" => :coils_V_applied,
    )
    Adict0 = Dict{String, Any}("time_s" => [0.1], (k => [1.0] for (k, _) in old0D)...)
    paths0 = create_symbol_path_mapping(Snapshot0D{Float64}, Adict0)
    @test all(paths0[k] == [v] for (k, v) in old0D)
end
