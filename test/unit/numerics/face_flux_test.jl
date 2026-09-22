# ∇·(u f) as a flux across cell faces, rows on in-wall nodes only.
#
# The nodal upwind `∇𝐮` cannot close a wall ledger: it has no rows on the grid frame, so
# whatever the out-wall band convects into the frame leaves unbooked, and where |u| < eps it
# switches to central differencing and receives half of what an upwind neighbour sent. The
# face form telescopes exactly over interior faces, so the only flux left in Σ V·∇·(u f) is
# the outflow through wall faces — which is what the ledger books.
# internal/docs/src/notes/design/wall-flux-channels.md §2.6, §3.

@testitem "face flux: divergence sums to the wall outflow, including where u changes sign" begin
    using RAPID2D: build_face_flux_divergence, face_outflow_speeds, wall_faces
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)
    inw = G.nodes.in_wall_nids
    n = zeros(G.NR, G.NZ)
    n[inw] .= 1.0e14 .* (1 .+ 0.2 .* cos.(2 .* G.Z2D[inw]))
    # converging flow: u_R > 0 on the left half, < 0 on the right half
    Rc = 0.5 * (G.R1D[1] + G.R1D[end])
    uR = @. 1.0e5 * sign(Rc - G.R2D)
    uZ = @. 3.0e4 * sin(G.Z2D)
    A = build_face_flux_divergence(G, uR, uZ)
    d = A * vec(n)
    total = sum(G.Jacob[inw] .* d[inw]) * G.dR * G.dZ           # Σ V_i (∇·(u n))_i / 2π
    v_out = face_outflow_speeds(G, faces, uR, uZ)
    boundary = sum(f.area * v_out[k] * n[f.nid] for (k, f) in enumerate(faces)) / (2π)
    @test boundary > 0
    @test total ≈ boundary rtol = 1.0e-12       # interior faces telescope; only wall faces remain
    @test all(iszero, d[G.nodes.on_out_wall_nids])
end

@testitem "face flux: inflow faces read nothing from outside, outflow faces are the Robin debit" begin
    using RAPID2D: build_face_flux_divergence, face_outflow_speeds, wall_faces
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)
    n = fill(NaN, G.NR, G.NZ)
    n[G.nodes.in_wall_nids] .= 1.0                       # NaN outside: any read from there shows
    uR = fill(-1.0e5, G.NR, G.NZ)
    uZ = zeros(G.NR, G.NZ)                                # uniform flow toward −R
    A = build_face_flux_divergence(G, uR, uZ)
    d = A * vec(n)
    @test all(isfinite, d[G.nodes.in_wall_nids])
    v_out = face_outflow_speeds(G, faces, uR, uZ)
    for (k, f) in enumerate(faces)
        expected = f.outward == (-1, 0) ? 1.0e5 : 0.0
        @test v_out[k] == expected
    end
end

@testitem "face flux: uniform u reproduces the nodal upwind operator deep inside the plasma" begin
    using RAPID2D: build_face_flux_divergence, update_∇𝐮_operator!, is_in_wall
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    # nodes two cells away from anything that is not in-wall: both stencils see only
    # in-wall values there
    deep = [
        (j - 1) * G.NR + i for j in 1:G.NZ, i in 1:G.NR
            if all(is_in_wall(G, i + di, j + dj) for di in -2:2, dj in -2:2)
    ]
    @test !isempty(deep)
    n = @. 1.0e14 * (1 + 0.1 * sin(3 * G.R2D) * cos(2 * G.Z2D))
    # Z direction carries no Jacobian: with uniform u the face velocity equals the node
    # velocity and the two operators are the same arithmetic.
    uR = zeros(G.NR, G.NZ)
    uZ = fill(-7.0e4, G.NR, G.NZ)
    update_∇𝐮_operator!(RP, uR, uZ)
    dZ_old = RP.operators.∇𝐮.matrix * vec(n)
    dZ_new = build_face_flux_divergence(G, uR, uZ) * vec(n)
    @test dZ_new[deep] ≈ dZ_old[deep] rtol = 1.0e-12
    # R direction: the nodal form puts the donor cell's J on the face, the face form puts
    # ½(J_i + J_{i+1}) there. Both telescope; they differ by O(ΔR/R).
    uR = fill(2.0e5, G.NR, G.NZ)
    uZ = zeros(G.NR, G.NZ)
    update_∇𝐮_operator!(RP, uR, uZ)
    dR_old = RP.operators.∇𝐮.matrix * vec(n)
    dR_new = build_face_flux_divergence(G, uR, uZ) * vec(n)
    @test dR_new[deep] ≈ dR_old[deep] rtol = G.dR / minimum(G.R1D)
    @test !isapprox(dR_new[deep], dR_old[deep]; rtol = 1.0e-12)   # and they are not the same arithmetic
end

@testitem "face flux: central interior faces (upwind = false) still telescope to the wall outflow" begin
    using RAPID2D: build_face_flux_divergence, face_outflow_speeds, wall_faces
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
        dt = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    G = RP.G
    faces = wall_faces(G)
    inw = G.nodes.in_wall_nids
    n = zeros(G.NR, G.NZ)
    n[inw] .= 1.0e14 .* (1 .+ 0.2 .* cos.(2 .* G.Z2D[inw]))
    Rc = 0.5 * (G.R1D[1] + G.R1D[end])
    uR = @. 1.0e5 * sign(Rc - G.R2D)
    uZ = @. 3.0e4 * sin(G.Z2D)
    A = build_face_flux_divergence(G, uR, uZ; upwind = false)
    d = A * vec(n)
    total = sum(G.Jacob[inw] .* d[inw]) * G.dR * G.dZ
    v_out = face_outflow_speeds(G, faces, uR, uZ)
    boundary = sum(f.area * v_out[k] * n[f.nid] for (k, f) in enumerate(faces)) / (2π)
    @test total ≈ boundary rtol = 1.0e-12         # wall faces stay upwind; interior central faces cancel
    @test A != build_face_flux_divergence(G, uR, uZ; upwind = true)
end
