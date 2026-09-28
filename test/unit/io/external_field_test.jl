# A BREAK-format external field: one file per time slice, read from a directory, or a single
# file given by its path. One slice is a static field, applied at every time.

@testsnippet BreakFieldFile begin
    # B_R = s·R, B_Z = s·Z, ψ = s·R·Z and a 5 V loop voltage on a small (R, Z) grid
    function write_break_field(path; time_s = 0.0, s = 1.0)
        R, Z = range(1.0, 2.0, length = 5), range(-1.0, 1.0, length = 4)
        open(path, "w") do io
            println(io, "Time=\t", time_s)
            println(io, "R_NUM=\t", length(R), "\tR_MIN=\t", first(R), "\tR_MAX=\t", last(R))
            println(io, "Z_NUM=\t", length(Z), "\tZ_MIN=\t", first(Z), "\tZ_MAX=\t", last(Z))
            println(io, "R Z B_R B_Z psi LoopVoltage")
            for r in R, z in Z
                println(io, r, " ", z, " ", s * r, " ", s * z, " ", s * r * z, " ", 5.0)
            end
        end
        return path
    end
end

@testitem "external field: one file is a static field, from its path or its directory" setup = [BreakFieldFile] begin
    using RAPID2D: read_external_field_time_series, calculate_external_fields_at_time
    mktempdir() do dir
        file = write_break_field(joinpath(dir, "static.dat"))
        for src in (dir, file)
            extF = read_external_field_time_series(src)
            @test length(extF.time_s) == 1
            for t in (-1.0, 0.0, 0.3, 10.0)
                f = calculate_external_fields_at_time(extF, t)
                @test f.BR == extF.BR[:, :, 1] && f.BZ == extF.BZ[:, :, 1]
                @test f.psi == extF.psi[:, :, 1] && f.LV == extF.LV[:, :, 1]
            end
        end
    end
end

@testitem "external field: InputPaths names the field and the wall, no device name needed" setup = [BreakFieldFile] begin
    mktempdir() do dir
        field = write_break_field(joinpath(dir, "quad.dat"))
        wall = joinpath(dir, "box_wall.dat")
        write(wall, "WALL_NUM\t\t4\n\n1.2 -0.7\n1.2 0.7\n1.8 0.7\n1.8 -0.7\n")
        config = SimulationConfig{Float64}(
            inputs = InputPaths(field = field, wall = wall), NR = 11, NZ = 13, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        G = RP.G
        @test (G.R1D[1], G.R1D[end], G.Z1D[1], G.Z1D[end]) == (1.0, 2.0, -1.0, 1.0)   # the field file's box
        @test RP.wall.R[1:4] == [1.2, 1.2, 1.8, 1.8] && RP.wall.Z[1:4] == [-0.7, 0.7, 0.7, -0.7]
        # B_R = R and B_Z = Z are linear, so interpolation onto the grid is exact
        @test RP.fields.BR_ext ≈ G.R2D && RP.fields.BZ_ext ≈ G.Z2D && all(==(5.0), RP.fields.LV_ext)
        BR0 = copy(RP.fields.BR_ext)
        update_external_fields!(RP, 1.0)                    # static: the same field later on
        @test RP.fields.BR_ext == BR0

        # the manual fields with the wall from its file
        config.inputs.field = ""
        RP = RAPID{Float64}(config)
        initialize!(RP)
        @test RP.wall.R[1:4] == [1.2, 1.2, 1.8, 1.8] && RP.wall.Z[1:4] == [-0.7, 0.7, 0.7, -0.7]

        # a field file and no wall: the same default box as the manual fields, three cells inside
        config.inputs.field, config.inputs.wall = field, ""
        RP = RAPID{Float64}(config)
        initialize!(RP)
        G = RP.G
        @test extrema(RP.wall.R) == (G.R1D[1] + 3 * G.dR, G.R1D[end] - 3 * G.dR)
        @test extrema(RP.wall.Z) == (G.Z1D[1] + 3 * G.dZ, G.Z1D[end] - 3 * G.dZ)
    end
end

@testitem "external field: initialize! reports which field and which wall it chose" setup = [BreakFieldFile] begin
    # Each source is a separate path through the lookup; the report names the one taken, so a
    # run that silently fell back to something else is visible in its log.
    mktempdir() do dir
        field = write_break_field(joinpath(dir, "quad.dat"))
        wall = joinpath(dir, "box_wall.dat")
        write(wall, "WALL_NUM\t\t4\n\n1.2 -0.7\n1.2 0.7\n1.8 0.7\n1.8 -0.7\n")
        mkpath(joinpath(dir, "dev", "shot"))
        write_break_field(joinpath(dir, "dev", "shot", "slice.dat"))
        cp(wall, joinpath(dir, "dev_First_Wall.dat"))
        rp(; kw...) = RAPID{Float64}(
            SimulationConfig{Float64}(;
                NR = 11, NZ = 13, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
                dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0, kw...,
            )
        )

        @test_logs (:info, r"^External field from inputs\.field: static \(1 slice\)") (:info, r"^Wall from inputs\.wall: 4 points") match_mode = :any initialize!(
            rp(inputs = InputPaths(field = field, wall = wall))
        )
        @test_logs (:info, r"^External field from Input_path/device_Name/shot_Name: static") (:info, r"^Wall from the device file: 4 points") match_mode = :any initialize!(
            rp(Input_path = dir, device_Name = "dev", shot_Name = "shot")
        )
        @test_logs (:info, r"^External field from the manual setup") (:info, r"^Wall from config\.wall_R/wall_Z: 4 points") match_mode = :any initialize!(
            rp(wall_R = [1.2, 1.8, 1.8, 1.2], wall_Z = [-0.7, -0.7, 0.7, 0.7])
        )
        @test_logs (:info, r"^Wall from a default box three cells inside the domain") match_mode = :any initialize!(
            rp(inputs = InputPaths(field = field))
        )
    end
end

@testitem "external field: two slices are interpolated linearly in time and held outside them" setup = [BreakFieldFile] begin
    using RAPID2D: read_external_field_time_series, calculate_external_fields_at_time
    mktempdir() do dir
        write_break_field(joinpath(dir, "a.dat"); time_s = 0.0, s = 1.0)
        write_break_field(joinpath(dir, "b.dat"); time_s = 1.0, s = 3.0)
        extF = read_external_field_time_series(dir)
        @test extF.time_s == [0.0, 1.0]
        @test calculate_external_fields_at_time(extF, 0.5).BR ≈ 2 * extF.BR[:, :, 1]
        @test calculate_external_fields_at_time(extF, -1.0).BR == extF.BR[:, :, 1]
        @test calculate_external_fields_at_time(extF, 2.0).BR == extF.BR[:, :, 2]
    end
end
