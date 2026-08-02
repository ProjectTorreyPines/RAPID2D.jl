# Per-face absorbed inventory.
#
# Conservation has so far been a GLOBAL statement: Σ J·n does not drift. That
# catches a leak but cannot say where it came from, and it cannot answer the
# question every surface process reduces to — how much did *this* tile absorb.
#
# Booking per face makes conservation a LOCAL identity:
#
#     Δ(Σ J·n)_interior  +  Σ_f absorbed_f / (2π·ΔR·ΔZ)  =  0
#
# so a discrepancy points at a face instead of at the whole domain. Today
# electrons and ions are booked per NODE (`cum2D_Ne_loss[on_out_wall_nids]`),
# recording what an outside cell held when it was zeroed — cells the Robin form
# removes entirely.

@testitem "Wall ledger: absorbed plus remaining accounts for everything" begin
    using RAPID2D: build_wall_diffusion_matrix, wall_faces, WallLedger,
        accumulate_wall_absorption!, total_absorbed
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)

    # anisotropic and oblique, so the cross terms are active while the books close
    Dpara, Dperp = 100.0, 0.1
    θ = 0.6
    bR, bZ = cos(θ), sin(θ)
    D_RR = fill(Dperp + (Dpara - Dperp) * bR^2, G.NR, G.NZ)
    D_RZ = fill((Dpara - Dperp) * bR * bZ, G.NR, G.NZ)
    D_ZZ = fill(Dperp + (Dpara - Dperp) * bZ^2, G.NR, G.NZ)

    v_abs = fill(300.0, length(faces))
    A = build_wall_diffusion_matrix(G, D_RR, D_RZ, D_ZZ; faces = faces, v_absorb = v_abs)

    Jv = vec(G.Jacob)
    inw = G.nodes.in_wall_nids
    total_Jn(v) = sum(Jv[k] * v[k] for k in inw)

    n0 = zeros(G.NR * G.NZ)
    n0[inw] .= 1.0e18
    dt = 1.0e-5
    F = lu(sparse(I, size(A, 1), size(A, 2)) - dt * A)

    ledger = WallLedger{Float64}(length(faces))
    function step!(v)
        v_new = F \ v
        # backward Euler evaluates the flux at the NEW state, so the ledger must
        # read the post-step density or the books miss by O(Δt)
        accumulate_wall_absorption!(ledger, faces, v_abs, v_new, dt)
        return v_new
    end

    function march(v0, nsteps)
        v = copy(v0)
        for _ in 1:nsteps
            v = step!(v)
        end
        return v
    end

    before = total_Jn(n0)
    v = march(n0, 50)
    after = total_Jn(v)

    # THE local identity. Every particle that left the interior went through a
    # named face; nothing vanished into the discretisation.
    cell_volume_factor = 2π * G.dR * G.dZ
    @test (before - after) ≈ total_absorbed(ledger) / cell_volume_factor rtol = 1.0e-10

    @test total_absorbed(ledger) > 0
    @test all(>=(0), ledger.absorbed)         # a wall cannot un-absorb
end

@testitem "Wall ledger: a reflective wall books exactly nothing" begin
    using RAPID2D: build_wall_diffusion_matrix, wall_faces, WallLedger,
        accumulate_wall_absorption!, total_absorbed
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)

    D = fill(10.0, G.NR, G.NZ)
    Z = zeros(G.NR, G.NZ)
    v_abs = zeros(length(faces))
    A = build_wall_diffusion_matrix(G, D, Z, D; faces = faces, v_absorb = v_abs)

    inw = G.nodes.in_wall_nids
    n0 = zeros(G.NR * G.NZ)
    n0[inw] .= 1.0e18
    dt = 1.0e-5
    F = lu(sparse(I, size(A, 1), size(A, 2)) - dt * A)

    ledger = WallLedger{Float64}(length(faces))
    function march(v0, nsteps)
        v = copy(v0)
        for _ in 1:nsteps
            v = F \ v
            accumulate_wall_absorption!(ledger, faces, v_abs, v, dt)
        end
        return v
    end
    march(n0, 20)

    # `== 0`, exactly. R = 1 is the case every existing conservation test rests on,
    # and a ledger that booked round-off there would make the invariant untestable.
    @test all(==(0.0), ledger.absorbed)
    @test total_absorbed(ledger) == 0.0
end

@testitem "Wall ledger: absorption is attributed to the face it crossed" begin
    using RAPID2D: build_wall_diffusion_matrix, wall_faces, WallLedger,
        accumulate_wall_absorption!
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)

    D = fill(10.0, G.NR, G.NZ)
    Z = zeros(G.NR, G.NZ)

    # a limiter that pumps on the outboard segment only
    pumped = [f.outward == (1, 0) for f in faces]
    v_abs = [p ? 300.0 : 0.0 for p in pumped]
    @test any(pumped) && !all(pumped)

    A = build_wall_diffusion_matrix(G, D, Z, D; faces = faces, v_absorb = v_abs)
    inw = G.nodes.in_wall_nids
    n0 = zeros(G.NR * G.NZ)
    n0[inw] .= 1.0e18
    dt = 1.0e-5
    F = lu(sparse(I, size(A, 1), size(A, 2)) - dt * A)

    ledger = WallLedger{Float64}(length(faces))
    function march(v0, nsteps)
        v = copy(v0)
        for _ in 1:nsteps
            v = F \ v
            accumulate_wall_absorption!(ledger, faces, v_abs, v, dt)
        end
        return v
    end
    march(n0, 20)

    # every un-pumped face books EXACTLY zero — this is what "per face" buys over
    # a global inventory, and what makes "which tile did this" answerable
    @test all(ledger.absorbed[k] == 0.0 for k in eachindex(faces) if !pumped[k])
    @test all(ledger.absorbed[k] > 0.0 for k in eachindex(faces) if pumped[k])
end

@testitem "Wall ledger: a corner cell books each of its two faces separately" begin
    using RAPID2D: build_wall_diffusion_matrix, wall_faces, WallLedger,
        accumulate_wall_absorption!
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)

    D = fill(10.0, G.NR, G.NZ)
    Z = zeros(G.NR, G.NZ)
    v_abs = fill(300.0, length(faces))
    A = build_wall_diffusion_matrix(G, D, Z, D; faces = faces, v_absorb = v_abs)

    inw = G.nodes.in_wall_nids
    n0 = zeros(G.NR * G.NZ)
    n0[inw] .= 1.0e18
    dt = 1.0e-5
    F = lu(sparse(I, size(A, 1), size(A, 2)) - dt * A)
    n1 = F \ n0

    ledger = WallLedger{Float64}(length(faces))
    accumulate_wall_absorption!(ledger, faces, v_abs, n1, dt)

    # a staircase corner owns an R-face and a Z-face; each is booked on its own
    # area, and together they account for the whole cell's loss
    owners = Dict{Int, Vector{Int}}()
    for (k, f) in enumerate(faces)
        push!(get!(owners, f.nid, Int[]), k)
    end
    corner_nid = first(nid for (nid, ks) in owners if length(ks) == 2)
    ks = owners[corner_nid]

    @test length(ks) == 2
    @test all(ledger.absorbed[k] > 0 for k in ks)
    # they differ, because an R-face and a Z-face sweep different areas
    @test !(ledger.absorbed[ks[1]] ≈ ledger.absorbed[ks[2]])

    # and each equals A_f·v·n·Δt on its own face
    for k in ks
        @test ledger.absorbed[k] ≈ faces[k].area * v_abs[k] * n1[corner_nid] * dt rtol = 1.0e-12
    end
end

@testitem "Wall ledger: re-emitting everything reproduces a reflective wall" begin
    using RAPID2D: build_wall_diffusion_matrix, wall_faces, WallLedger,
        accumulate_wall_absorption!, wall_emission_source,
        total_absorbed, total_emitted
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    # The invariant that catches the whole "created at the wall and silently lost"
    # class — the one the existing secondary-electron path fails, by depositing
    # outside the wall where the next step zeroes it. If everything absorbed is
    # returned to the wall-adjacent INTERIOR cell, the result must be the
    # reflective one to machine precision.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)

    D = fill(10.0, G.NR, G.NZ)
    Z = zeros(G.NR, G.NZ)
    Jv = vec(G.Jacob)
    inw = G.nodes.in_wall_nids
    outw = G.nodes.on_out_wall_nids
    total_Jn(v) = sum(Jv[k] * v[k] for k in inw)

    n0 = zeros(G.NR * G.NZ)
    n0[inw] .= 1.0e18
    dt = 1.0e-5

    # absorbing wall, with every absorbed particle returned as a source
    v_abs = fill(300.0, length(faces))
    A_abs = build_wall_diffusion_matrix(G, D, Z, D; faces = faces, v_absorb = v_abs)
    F_abs = lu(sparse(I, size(A_abs, 1), size(A_abs, 2)) - dt * A_abs)

    function march_recycled(v0, nsteps)
        v = copy(v0)
        led = WallLedger{Float64}(length(faces))
        step_abs = zeros(length(faces))
        for _ in 1:nsteps
            v = F_abs \ v
            fill!(step_abs, 0.0)
            before = copy(led.absorbed)
            accumulate_wall_absorption!(led, faces, v_abs, v, dt)
            step_abs .= led.absorbed .- before
            # Y = 1, same species: everything comes straight back into the
            # wall-adjacent interior cell across the same face, and the return is
            # booked so the ledger describes both directions
            v .+= wall_emission_source(G, faces, step_abs, dt) .* dt
            led.emitted .+= step_abs
        end
        return v, led
    end

    recycled, led = march_recycled(n0, 20)

    # nothing leaves: Y = 1 must reproduce the reflective invariant exactly
    @test total_Jn(recycled) ≈ total_Jn(n0) rtol = 1.0e-10
    @test maximum(abs, recycled[outw]) == 0.0
    @test minimum(recycled[inw]) >= 0.0

    # and the ledger says so too: every absorbed particle is accounted for as
    # emitted, face by face, which is what makes the closure auditable
    @test total_emitted(led) ≈ total_absorbed(led) rtol = 1.0e-14
    @test total_absorbed(led) > 0
    @test led.emitted ≈ led.absorbed rtol = 1.0e-14
end
