# The species in the snapshots: each parent's ions by charge state (`ni_<parent>`) and its neutral
# gas (`ng_<parent>`), read by name, written to the files under those names and read back; the
# total ion density `ni_tot`; files of the first layout read into the second.

@testsnippet SnapshotSpeciesCase begin
    using RAPID2D: LadderRates, ConstantRate
    using RAPID2D.Particles: H, H2

    # A toy element X[0:3]: three ion charge states and a neutral
    struct SnapLadder <: RAPID2D.Element end
    RAPID2D.allowed_charges(::SnapLadder) = 0:3
    RAPID2D.composition(::SnapLadder) = (:X => 1,)
    RAPID2D.parent_name(::SnapLadder) = "X"
    RAPID2D.neutral_mass(::SnapLadder, ::RAPID2D.AbstractHydrogenIsotope, constants::RAPID2D.PlasmaConstants) = 12 * constants.mp
    RAPID2D.ladder_rates(::SnapLadder, ::RAPID2D.RateClosure, ::Type{FT}) where {FT} = LadderRates(
        (ConstantRate(FT(4.0e-14)), ConstantRate(FT(2.0e-14)), ConstantRate(FT(1.0e-14))),
        (ConstantRate(FT(1.0e-14)), ConstantRate(FT(3.0e-14)), ConstantRate(FT(5.0e-14))),
        (FT(11.3), FT(24.4), FT(47.9)), ("toy", "toy"),
    )
    const X = SnapLadder()

    "A run of `species` in `hydrogen`, `nsteps` steps long with a snapshot at every step."
    function species_case(; nsteps = 3, species, hydrogen = Protium())
        config = SimulationConfig{Float64}(;
            device_Name = "manual", NR = 9, NZ = 9, R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8, t_end_s = nsteps * 1.0e-8,
            snap0D_Δt_s = 1.0e-8, snap2D_Δt_s = 1.0e-8, Output_path = mktempdir(; cleanup = false),
            species, hydrogen,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        return RP
    end

    "A run of H₂, H and X with X[2] seeded."
    function species_run(; nsteps = 3)
        RP = species_case(; nsteps, species = (H2[0], H2[1], H[0], H[1], X[0], X[1], X[2], X[3]))
        RP.plasma.n_ions[3][:, :, 2] .= 1.0e15
        return RP
    end
end

@testitem "A snapshot holds each parent's species by name, its ions by charge state" setup = [SnapshotSpeciesCase] begin
    using RAPID2D: measure_snap0D, measure_snap2D, snapshot_matrix, is_density_field

    RP = species_run()
    G, pla = RP.G, RP.plasma
    inside, band = G.nodes.in_wall_nids, G.nodes.on_out_wall_nids
    s2, s0 = measure_snap2D(RP), measure_snap0D(RP)

    # 2D: the ions of each parent as planes, masked outside the wall as the total is; the gas as is
    @test s2.ion_parents == ["H2", "H", "X"] && s2.gas_parents == ["H2", "H", "X"]
    @test size(s2.ni_H2) == (9, 9, 1) && size(s2.ni_X) == (9, 9, 3) && size(s2.ng_H2) == (9, 9)
    @test s2.ni_X[:, :, 2][inside] == pla.n_ions[3][:, :, 2][inside] && all(isnan, s2.ni_X[:, :, 2][band])
    @test s2.ni_tot[inside] == pla.ni_tot[inside]
    @test s2.ng_H2 == pla.n_H2_gas && s2.n_H2_gas === s2.ng_H2
    @test :ni_X in propertynames(s2) && :ng_H in propertynames(s2) && :n_H2_gas in propertynames(s2)
    @test_throws ArgumentError s2.ni_Y

    # 0D: volume averages, the ions' by charge state
    V = G.inVol2D                                    # the volume the averages take, wall cells in part
    @test s0.ni_X ≈ [sum(pla.n_ions[3][:, :, z] .* V) / G.device_inVolume for z in 1:3] rtol = 1.0e-12
    @test s0.ng_H2 ≈ sum(pla.n_H2_gas .* V) / G.device_inVolume rtol = 1.0e-12
    @test s0.n_H2_gas === s0.ng_H2
    @test s0.ni_tot ≈ sum(pla.ni_tot .* V) / G.device_inVolume rtol = 1.0e-12

    # for a plot: one charge state at a time
    @test snapshot_matrix(s2, :ng_H2) === s2.ng_H2
    @test isequal(snapshot_matrix(s2, :ni_X, 2), s2.ni_X[:, :, 2])        # NaN outside the wall
    @test_throws ArgumentError snapshot_matrix(s2, :ni_X)
    @test is_density_field(:ni_X) && is_density_field(:ng_H2) && is_density_field(:ne) && !is_density_field(:Te_eV)
end

@testitem "The species round-trip through the files, one step or many, and stack in time" setup = [SnapshotSpeciesCase] begin
    using RAPID2D: adiosBP_to_snap0D, adiosBP_to_snap2D, write_to_adiosBP!

    RP = species_run(nsteps = 3)
    run_simulation!(RP)
    d = RP.diagnostics
    r2, r0 = adiosBP_to_snap2D(RP.snap2D_path), adiosBP_to_snap0D(RP.snap0D_path)
    @test length(r2) == length(d.snaps2D) >= 3 && length(r0) == length(d.snaps0D)
    @test all(isequal(a, b) for (a, b) in zip(r2, d.snaps2D)) && all(isequal(a, b) for (a, b) in zip(r0, d.snaps0D))
    @test r2[end].ion_parents == ["H2", "H", "X"] && r2[end].schema == RAPID2D.SNAPSHOT_SCHEMA

    # in time: the species gain a last axis, as the fields do
    Nt = length(r2)
    @test size(r2.ni_X) == (9, 9, 3, Nt) && size(r2.ni_H2) == (9, 9, 1, Nt) && size(r2.ng_H2) == (9, 9, Nt)
    @test size(r0.ni_X) == (3, Nt) && size(r0.ng_H2) == (Nt,)
    @test isequal(r2.ni_X[:, :, :, end], d.snaps2D[end].ni_X)

    # a file of one step, whose step axis the reader does not see
    dir = mktempdir()
    write_to_adiosBP!(joinpath(dir, "one2D.bp"), d.snaps2D[end])
    write_to_adiosBP!(joinpath(dir, "one0D.bp"), d.snaps0D[end])
    @test isequal(only(adiosBP_to_snap2D(joinpath(dir, "one2D.bp"))), d.snaps2D[end])
    @test isequal(only(adiosBP_to_snap0D(joinpath(dir, "one0D.bp"))), d.snaps0D[end])
end

@testitem "In deuterium `n_H2_gas` is the D₂ gas, measured, read back and in time" setup = [SnapshotSpeciesCase] begin
    using RAPID2D: measure_snap0D, measure_snap2D, adiosBP_to_snap0D, adiosBP_to_snap2D

    RP = species_case(species = (H2[0], H2[1]), hydrogen = Deuterium())
    s2, s0 = measure_snap2D(RP), measure_snap0D(RP)
    @test s2.gas_parents == ["D2"] && s2.n_H2_gas === s2.ng_D2 && s0.n_H2_gas === s0.ng_D2
    @test s2.n_H2_gas == RP.plasma.n_H2_gas
    @test :n_H2_gas in propertynames(s2) && :n_H2_gas in propertynames(s0)

    run_simulation!(RP)
    r2, r0 = adiosBP_to_snap2D(RP.snap2D_path), adiosBP_to_snap0D(RP.snap0D_path)
    @test r2[end].n_H2_gas === r2[end].ng_D2 && r0[end].n_H2_gas === r0[end].ng_D2
    @test r2.n_H2_gas == r2.ng_D2 && size(r2.n_H2_gas) == (9, 9, length(r2)) && r0.n_H2_gas == r0.ng_D2
end

@testitem "A file of the first layout reads its ion density as the total and its H₂ gas as the H2 parent's" begin
    using RAPID2D: Snapshot0D, Snapshot2D, create_symbol_path_mapping, process_symbol_based_snapshots!, read_snapshot_species!

    # what such a file holds, as adios_load returns it: two steps
    A0 = Dict{String, Any}("time_s" => [0.0, 1.0], "step" => [0, 1], "ni" => [1.0, 2.0], "ni_max" => [3.0, 4.0], "n_H2_gas" => [5.0, 6.0])
    snaps = [Snapshot0D{Float64}() for _ in 1:2]
    process_symbol_based_snapshots!(snaps, A0, create_symbol_path_mapping(Snapshot0D{Float64}, A0), 2)
    read_snapshot_species!(snaps, A0)
    @test snaps[2].ni_tot == 2.0 && snaps[2].ni_tot_max == 4.0
    @test snaps[2].ng_H2 == 6.0 && snaps[2].n_H2_gas == 6.0 && snaps[2].schema == 1
    @test isempty(snaps[2].ion_parents)          # which ions made the total, the file does not say

    gas = reshape(collect(1.0:(3 * 3 * 2)), 3, 3, 2)
    A2 = Dict{String, Any}("time_s" => [0.0, 1.0], "ni" => 2 .* gas, "n_H2_gas" => gas, "dims_RZ/1" => [3], "dims_RZ/2" => [3])
    snaps = [Snapshot2D{Float64}(dims_RZ = (3, 3)) for _ in 1:2]
    process_symbol_based_snapshots!(snaps, A2, create_symbol_path_mapping(Snapshot2D{Float64}, A2), 2)
    read_snapshot_species!(snaps, A2)
    @test snaps[2].ni_tot == 2 .* gas[:, :, 2] && snaps[2].ng_H2 == gas[:, :, 2] && snaps[2].schema == 1
end
