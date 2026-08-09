@testitem "scheme: one algorithm per family, defaulting to today's behaviour" begin
    using RAPID2D: TimeSchemes, TimeScheme, ForwardEuler, Theta, ExpRB, SimulationFlags

    s = TimeSchemes()

    # The families are `ImplicitWeights`' families — the split is by the character
    # of the operator, and that is exactly the split B(z) can and cannot cross.
    # Four of them are θ-weighted today.
    @test s.transport === Theta
    @test s.growth === Theta
    @test s.decay === Theta
    @test s.gas === Theta

    # Tₑ's atomic power is the asymmetry this whole change exists for: it is not
    # θ-weighted, it is *unweighted*. `ForwardEuler` names that state rather than
    # letting a boolean "off" mean two different things in different families.
    @test s.atomic === ForwardEuler

    # Defaults must reproduce current behaviour, so a fresh flags object is inert.
    @test SimulationFlags{Float64}().scheme.atomic === ForwardEuler
end

@testitem "scheme: rejected combinations fail where they are written" begin
    using RAPID2D: TimeSchemes, ForwardEuler, Theta, ExpRB

    # `ImplicitWeights` validates on assignment so a bad weight "fails where it was
    # written rather than as a growing mode ten thousand steps later". Its sibling
    # follows suit, and in doing so turns the design note's scope table into a
    # contract the code enforces instead of a paragraph someone has to have read.
    s = TimeSchemes()

    # B(z) fits the LOCAL (diagonal) eigenvalue. A diffusion operator's stiff mode
    # ~4D/h² belongs to the mesh and the operator, not to any one cell, so no
    # per-cell fit can see it. Transport and neutral-gas diffusion are nonlocal.
    @test_throws ArgumentError s.transport = ExpRB
    @test_throws ArgumentError s.gas = ExpRB

    # θ on the linearised atomic power is design note §3.1: same Jacobian, one
    # expm1 less, and measured at FIRST order against ExpRB's second. θ_imp has no
    # `atomic` member precisely so this cannot be configured by accident.
    @test_throws ArgumentError s.atomic = Theta

    # The rejections carry their reason, not just a type name.
    err = try
        s.transport = ExpRB
    catch e
        sprint(showerror, e)
    end
    @test occursin("nonlocal", err)

    # A rejected assignment leaves the field untouched.
    @test s.transport === Theta
    @test s.atomic === ForwardEuler
end

@testitem "scheme: the transitions each phase makes are allowed" begin
    using RAPID2D: TimeSchemes, ForwardEuler, Theta, ExpRB

    s = TimeSchemes()

    s.atomic = ExpRB            # Phase 1
    @test s.atomic === ExpRB
    s.decay = ExpRB             # Phase 2
    @test s.decay === ExpRB
    s.growth = ExpRB            # Phase 3
    @test s.growth === ExpRB

    # And every one of them can be walked back — the manuscript's figures are
    # pinned to the current scheme, so reproducing them must stay one assignment
    # away rather than a git checkout.
    s.atomic = ForwardEuler
    s.decay = Theta
    s.growth = Theta
    @test (s.atomic, s.decay, s.growth) === (ForwardEuler, Theta, Theta)
end

@testitem "scheme: θ_imp is the weight store, scheme is the algorithm" begin
    using RAPID2D: TimeSchemes, ImplicitWeights, ExpRB, Theta

    # The two structs are orthogonal by design: θ_imp holds the *value* of θ and
    # is read only where scheme == Theta. Nothing about ImplicitWeights changes,
    # including its validation, so no existing configuration is reinterpreted.
    w = ImplicitWeights{Float64}()
    @test w.decay == 1.0
    @test !hasproperty(w, :atomic)          # §3.1 is not configurable, see above

    s = TimeSchemes()
    s.decay = ExpRB
    @test w.decay == 1.0                    # switching the algorithm does not touch the weight
    s.decay = Theta
    @test w.decay == 1.0
end

@testitem "scheme: ExpRB on the atomic power needs a differentiable rate" begin
    using RAPID2D: SimulationFlags, ExpRB, validate_scheme_flags

    # Legacy comparison paths have no rate to differentiate. Rather than carry a
    # dν/dTe = 0 branch for each of them forever, the combination is refused —
    # these paths are on their way out and should not shape the new code.
    for (field, bad) in ((:Ionz_method, "Townsend_coeff"), (:ud_method, "Lloyd_fit"))
        flags = SimulationFlags{Float64}()
        flags.scheme.atomic = ExpRB
        setproperty!(flags, field, bad)
        err = try
            validate_scheme_flags(flags)
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("Xsec", err)
        @test occursin(bad, err)
    end

    # The supported combination passes, and so does every combination with
    # ExpRB off — validation must not narrow what already works.
    ok = SimulationFlags{Float64}()
    ok.scheme.atomic = ExpRB
    @test validate_scheme_flags(ok) === ok

    legacy = SimulationFlags{Float64}()
    legacy.Ionz_method = "Townsend_coeff"
    @test validate_scheme_flags(legacy) === legacy
end

@testitem "scheme: initialize! is where the refusal actually lands" begin
    using RAPID2D: ExpRB

    function small_RAPID()
        config = SimulationConfig{Float64}(
            NR = 10, NZ = 10, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = 1.0e-8, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        config.Output_path = mktempdir(; cleanup = false)
        return RAPID{Float64}(config)
    end

    # A validator nothing calls is not a validator. The check runs before any
    # state is built, so a bad pairing fails at setup rather than as a silently
    # wrong Jacobian several thousand steps in.
    RP = small_RAPID()
    RP.flags.scheme.atomic = ExpRB
    RP.flags.Ionz_method = "Townsend_coeff"
    @test_throws ArgumentError initialize!(RP)

    # And the default configuration still initializes — validation refuses in one
    # direction only, it must not narrow what already works.
    @test initialize!(small_RAPID()) isa RAPID
end
