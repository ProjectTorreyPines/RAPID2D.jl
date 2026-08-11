# `validate_scheme_flags` runs once, at `initialize!`. Every flag it inspects stays
# mutable afterwards, and configuring a scheme after initialization is a routine idiom
# in this repo's own tests. So each refusal it makes needs a partner at the point of
# use, or the refused configuration is reachable by writing the same two lines in the
# other order — and reachable silently, which is the part that matters.

@testsnippet FlagMutationFixtures begin
    using RAPID2D: ExpRB, Theta, ForwardEuler, PartialLinearResponse, FullLinearResponse,
        update_ue_para!, update_electron_power_jacobian!, update_transport_quantities!

    # 0-D-ish and deliberately minimal: these tests are about which configurations a
    # solver accepts, not about what it computes.
    function mutable_RAPID(; decay = Theta, atomic = ForwardEuler)
        config = SimulationConfig{Float64}(
            NR = 6, NZ = 6, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-8, t_end_s = 1.0e-6, R0B0 = 1.0,
            prefilled_gas_pressure = 5.0e-3,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{Float64}(config)
        RP.flags.Atomic_Collision = true
        RP.flags.src = true
        RP.flags.Coulomb_Collision = false
        RP.flags.ud_evolve = true
        RP.flags.Ampere = false            # keep the coupled-solver refusal out of it
        RP.flags.scheme.decay = decay
        RP.flags.scheme.atomic = atomic
        initialize!(RP)
        RP.plasma.Te_eV .= 5.0
        RP.plasma.ne .= 1.0e16
        RP.plasma.ni .= 1.0e16
        RP.plasma.ue_para .= -1.0e5
        RP.plasma.sptz_fac .= 0.0
        RP.plasma.ν_ei .= 0.0
        return RP
    end
end

@testitem "flag mutation: the momentum solver refuses a response depth it cannot honour" setup = [FlagMutationFixtures] begin
    using RAPID2D: ExpRB, PartialLinearResponse, FullLinearResponse, update_ue_para!

    # `update_ue_para!` fits λ = −(ν_mom + ν_iz + ν_ei_eff), the stated rate, and
    # nothing computes the −(mₑu∥²/e)·∂ν/∂Ē that would complete it.
    # `validate_scheme_flags` refuses the pairing; setting the depth AFTER
    # `initialize!` walks past that refusal into a step that quietly delivers
    # PartialLinearResponse under a flag that says otherwise.
    RP = mutable_RAPID(; decay = ExpRB)
    @test RP.flags.exprb_eigenvalue === PartialLinearResponse
    update_ue_para!(RP)                                  # the supported pairing runs

    RP.flags.exprb_eigenvalue = FullLinearResponse
    err = try
        update_ue_para!(RP)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("FullLinearResponse", sprint(showerror, err))

    # One-directional, like the refusal it backs up: θ never claimed to read the
    # depth, so nothing about it becomes newly illegal.
    θ = mutable_RAPID()
    θ.flags.exprb_eigenvalue = FullLinearResponse
    @test update_ue_para!(θ) === θ
end

@testitem "flag mutation: the Tₑ Jacobian refuses derivatives its rate step never took" setup = [FlagMutationFixtures] begin
    using RAPID2D: ExpRB, PartialLinearResponse, FullLinearResponse,
        update_electron_power_jacobian!, update_transport_quantities!

    # `dν_dTe` is materialized by `update_RRCs!` only when FullLinearResponse was
    # already in force — that is the optimization that keeps the default path free.
    # Switching depth afterwards therefore hands `_eig_Te_from_linear_response!`
    # surfaces that are zero, or evaluated at a state one or more steps old. Both
    # produce a λ, and neither announces itself.
    RP = mutable_RAPID(; atomic = ExpRB)
    @test RP.flags.exprb_eigenvalue === PartialLinearResponse
    update_electron_power_jacobian!(RP)                  # the default path is unaffected
    partial = copy(RP.plasma.exprb.eig_Te)
    @test !all(iszero, partial)

    RP.flags.exprb_eigenvalue = FullLinearResponse
    err = try
        update_electron_power_jacobian!(RP)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("update_transport_quantities!", sprint(showerror, err))

    # And the message names the cure, so it has to work: one rate step under the new
    # depth, and the same call goes through.
    update_transport_quantities!(RP)
    update_electron_power_jacobian!(RP)
    @test !all(iszero, RP.plasma.dν_dTe.iz[RP.G.nodes.in_wall_nids])
    @test RP.plasma.exprb.eig_Te != partial

    # Switching back is symmetric: the partial branch reads no derivative, so a rate
    # step that stopped materializing them cannot make it stale.
    RP.flags.exprb_eigenvalue = PartialLinearResponse
    update_transport_quantities!(RP)
    @test update_electron_power_jacobian!(RP) === RP
end

@testitem "flag mutation: the ledger weight follows the solve, not the flag" setup = [FlagMutationFixtures] begin
    using RAPID2D: ExpRB, Theta, reaction_θ, solve_electron_continuity_equation!

    # `reaction_θ` must report the quadrature the LAST continuity solve used, because
    # that is the weight `update_reaction_counts!` formed its ledger at. Reading the
    # live flag instead means a flag flipped between the solve and the query answers
    # for a step that never happened: `exprb_theta` of a stale (or zero) `z_growth`
    # after a θ solve, or the θ constant after an ExpRB solve.
    RP = mutable_RAPID()
    @test RP.flags.scheme.growth === Theta
    solve_electron_continuity_equation!(RP)
    θ_solved = reaction_θ(RP, :iz)
    @test θ_solved isa Float64                       # a θ solve gives one constant

    RP.flags.scheme.growth = ExpRB                   # nothing has been solved with it
    @test reaction_θ(RP, :iz) == θ_solved

    # After a solve under the new scheme it is a field, per cell.
    solve_electron_continuity_equation!(RP)
    @test reaction_θ(RP, :iz) isa AbstractMatrix

    # And back again.
    RP.flags.scheme.growth = Theta
    @test reaction_θ(RP, :iz) isa AbstractMatrix     # still the ExpRB solve's weight
    solve_electron_continuity_equation!(RP)
    @test reaction_θ(RP, :iz) isa Float64
end
