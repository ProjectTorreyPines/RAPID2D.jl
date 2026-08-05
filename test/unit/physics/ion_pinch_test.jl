# Impurity pinch — the one term that cannot share the transport operator.
#
# NRL p.37's momentum balance, with the ion–ion friction R_zi kept rather than
# dropped, gives the trace-species flux
#
#     Γ_z = −D[ ∇n_z − (Z_z/Z_i)(n_z/n_i) ∇n_i − H (n_z/T_i) ∇T_i ]
#
# The middle term is the pinch: it drives species z UP the bulk gradient, with
# strength Z_z/Z_i, and it is why highly charged impurities accumulate on axis.
#
# It is proportional to ANOTHER species' gradient, so no averaging of D can
# reproduce it and it cannot go into the shared operator A. That is also why it
# does not need to: written as a convective velocity
#
#     V_pinch,z = 𝐃 (Z_z/Z_i) ∇n_i/n_i = Z_z · 𝐖,    𝐖 ≡ 𝐃 ∇n_i/(Z_i n_i)
#
# 𝐖 is species-INDEPENDENT, so one divergence operator serves every species and
# the term lands in the right-hand side. The factorization count does not move.
#
# `z ≠ i` is a PREMISE of that derivation, not a convention: the term exists
# because z rubs against i. Applying it to the bulk itself makes Z_z/Z_i = 1 and
# n_z = n_i, so `n_i V_pinch,i = 𝐃∇n_i` — the diffusive flux exactly, pointing the
# other way. The bulk would stop diffusing. So the source skips it, and with one
# ion species the pinch is identically zero: nothing to rub against.
#
# The genuinely multi-species assertions (a trace peaking on the bulk, the exact
# Z_z linearity, per-species groups each carrying their own 𝐃) need a second
# species, which `set_ion_species!` does not take yet. They are written out
# ready-to-run in internal_docs/src/notes/TODO/ion-inventory-multi-species.md.
#
# Default off: the term is real but unvalidated here, and turning it on would
# move every existing result once a second species exists.

@testsnippet PinchRun begin
    "A box-wall case with a peaked bulk ion."
    function pinch_case(; pinch::Bool, NR = 21, NZ = 21, policy = RAPID2D.SharedEffectiveTransport())
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = NR, NZ = NZ,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8,
            t_end_s = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        RP.flags.update_ni_independently = true
        RP.flags.ion_pinch = pinch
        RP.flags.convec = false            # isolate diffusion + pinch
        RP.flags.src = false
        RP.flags.ion_transport_policy = policy

        # Peaked at the box centre, so diffusion has something to flatten.
        R0, Z0 = 1.5, 0.0
        @. RP.plasma.ni = 1.0e18 * exp(-((RP.G.R2D - R0)^2 + (RP.G.Z2D - Z0)^2) / 0.02)
        RP.plasma.ne .= RP.plasma.ni
        RP.plasma.Te_eV .= 10.0
        RP.plasma.Ti_eV .= 1.0
        update_transport_quantities!(RP)
        return RP
    end

    "Peak of the bulk column."
    bulk_peak(RP) = maximum(view(RP.transport.ion_N, :, 1))
end

@testitem "The bulk ion does not pinch against itself" setup = [PinchRun] begin
    # With one species the only candidate partner is the bulk itself, and the
    # self-term is `−∇⋅(𝐃∇n)` — its own diffusion with the sign flipped. Left in,
    # it cancels the diffusion it accompanies and a peaked profile stops relaxing.
    on = pinch_case(pinch = true)
    off = pinch_case(pinch = false)
    p0 = maximum(off.plasma.ni)   # `ion_N` is only synced once a solve starts

    for _ in 1:20
        solve_ion_continuity_equation!(on)
        solve_ion_continuity_equation!(off)
    end

    # Over these 20 steps diffusion alone takes the peak down by 3.54e-5 of itself.
    # With the self-term left in, only 6.69e-6 of it survived — the pinch cancelled
    # 81 % of the relaxation, and `solve_ion_continuity_equation!` returned negative
    # densities where the profile was steepest.
    @test bulk_peak(off) < p0                          # diffusion really is flattening it
    @test bulk_peak(on) ≈ bulk_peak(off) rtol = 1.0e-12
    @test all(≥(0.0), on.transport.ion_N)
end

@testitem "The pinch is off by default, and is a no-op at one species" setup = [PinchRun] begin
    @test !RAPID{Float64}(SimulationConfig{Float64}()).flags.ion_pinch

    on = pinch_case(pinch = true)
    off = pinch_case(pinch = false)
    before = copy(off.transport.ion_N)
    solve_ion_continuity_equation!(on)
    solve_ion_continuity_equation!(off)

    @test off.transport.ion_N != before                # diffusion still ran
    @test on.transport.ion_N == off.transport.ion_N    # …and the flag changed nothing
end

@testitem "The pinch velocity is 𝐃∇n_i/(Z_i n_i), species-independent" setup = [PinchRun] begin
    using RAPID2D: ion_pinch_velocity

    # 𝐖 is built whether or not anyone consumes it, and it is a property of the
    # BULK profile and the tensor alone — no species enters until `Z_z` multiplies
    # it downstream. That is what lets one operator serve every species.
    RP = pinch_case(pinch = true)
    NR, NZ = RP.G.NR, RP.G.NZ

    # A bulk ramping linearly in R, a diagonal tensor: W_R = D_RR (dn/dR)/(Z_i n)
    slope = 1.0e18
    @. RP.plasma.ni = 1.0e18 + slope * (RP.G.R2D - 1.0)
    D_RR = fill(7.0, NR, NZ)
    D_RZ = zeros(NR, NZ)
    D_ZZ = fill(3.0, NR, NZ)

    W_R, W_Z = ion_pinch_velocity(RP, D_RR, D_RZ, D_ZZ)
    i, j = NR ÷ 2 + 1, NZ ÷ 2 + 1
    @test W_R[i, j] ≈ 7.0 * slope / RP.plasma.ni[i, j] rtol = 1.0e-10
    @test W_Z[i, j] ≈ 0.0 atol = 1.0e-12

    # The off-diagonal must be carried: 𝐃 is a tensor, not two scalars
    D_RZ2 = fill(2.0, NR, NZ)
    W_R2, W_Z2 = ion_pinch_velocity(RP, D_RR, D_RZ2, D_ZZ)
    @test W_R2[i, j] ≈ W_R[i, j] rtol = 1.0e-10          # ∇n has no Z component
    @test W_Z2[i, j] ≈ 2.0 * slope / RP.plasma.ni[i, j] rtol = 1.0e-10
end

@testitem "The pinch velocity is bounded where the gradient is unresolved" setup = [PinchRun] begin
    using RAPID2D: ion_pinch_velocity

    RP = pinch_case(pinch = true)
    NR, NZ = RP.G.NR, RP.G.NZ
    D = fill(5.0, NR, NZ)
    Z = zeros(NR, NZ)

    # A gradient scale shorter than one cell is not resolved, so |∇n/n| is capped
    # at 1/Δx. Without the cap |V| is unbounded and the explicit term's CFL goes
    # with it; with it, |V| ≤ D/Δx and the pinch is never stiffer than explicit
    # diffusion would have been.
    RP.plasma.ni .= 1.0
    RP.plasma.ni[NR ÷ 2 + 1, NZ ÷ 2 + 1] = 1.0e30       # one-cell spike
    W_R, W_Z = ion_pinch_velocity(RP, D, Z, D)
    @test all(isfinite, W_R) && all(isfinite, W_Z)
    @test maximum(abs, W_R) ≤ 5.0 / RP.G.dR * (1 + 1.0e-10)
    @test maximum(abs, W_Z) ≤ 5.0 / RP.G.dZ * (1 + 1.0e-10)

    # An empty cell has nothing to pinch, and 0/0 must not reach the operator
    RP.plasma.ni .= 0.0
    W_R0, W_Z0 = ion_pinch_velocity(RP, D, Z, D)
    @test all(iszero, W_R0) && all(iszero, W_Z0)
end

@testitem "The pinch stays out of the factorized operator" setup = [PinchRun] begin
    using RAPID2D: ion_step_operators

    # EXPLICIT means the matrix that gets factorized does not know the flag
    # exists. That is what keeps one factorization covering every species —
    # putting the pinch on the left would make the operator carry Z_z and cost a
    # factorization PER SPECIES.
    on = pinch_case(pinch = true)
    off = pinch_case(pinch = false)
    @test ion_step_operators(on)[1][1][2] == ion_step_operators(off)[1][1][2]

    solve_ion_continuity_equation!(on)
    @test length(on.transport.ion_solvers) == 1
end

@testitem "A full run survives the pinch being on" setup = [PinchRun] begin
    # The other tests drive `solve_ion_continuity_equation!` directly. This one
    # goes through `run_simulation!`, because that is where the operator's lazy
    # allocation, the per-step rebuild and the wall bookkeeping actually meet —
    # all of which happen whether or not the source it feeds is zero.
    RP = pinch_case(pinch = true)
    RP.flags.convec = true                     # both convective operators live
    RP.flags.src = true
    run_simulation!(RP)

    @test all(isfinite, RP.transport.ion_N)
    @test all(isfinite, RP.plasma.ni)
    @test all(≥(0.0), RP.transport.ion_N)      # upwinding must keep it positive
    @test length(RP.transport.ion_solvers) == 1
end
