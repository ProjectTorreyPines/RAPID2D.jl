# Type-stability pins for the RRC accessors.
#
# `RAPID.eRRCs` and `RAPID.iRRCs` are declared as the ABSTRACT `AbstractSpeciesRRCs{FT}`
# (`types.jl`), so a direct field read off them infers `Any`. One non-concrete SCALAR
# inside a `@.` stops `Broadcast.combine_eltypes` from producing a concrete eltype and
# the fused kernel falls back to per-element dynamic dispatch — a cost proportional to
# the grid, plus a boxed allocation per call.
#
# Every accessor that pulls a scalar off those fields and hands it to array code must
# therefore be a FUNCTION BARRIER with a concretising `::FT`. These tests are what stop
# one from silently losing it.
#
# The `@test @inferred(f(x)) isa T` form is load-bearing and the parentheses are not
# optional. `a isa T` lowers to `Expr(:call, :isa, a, T)`, so writing
#
#     @inferred f(x) isa T          # WRONG
#
# hands `@inferred` the `isa` call — which returns `Bool` whatever `f` does — and the
# assertion passes on a function inferring `Any`. Measured: on a deliberately unstable
# `bad(h) = h.a.x`, the un-parenthesised form returns `true` while `(@inferred bad(h))`
# correctly throws "return type Float64 does not match inferred return type Any".

@testsnippet TypeStabilityFixtures begin
    using RAPID2D: mean_energy_floor, erg_axis_bounds, _eRRC_query_point, update_RRCs!

    # Small, cheap, and with a plasma state real enough that the query point lands
    # inside the tables rather than on a clamp for every node.
    function ts_RAPID(; Te_eV = 5.0, ne = 1.0e16)
        config = SimulationConfig{Float64}(
            NR = 8, NZ = 8, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-8, t_end_s = 1.0e-6, R0B0 = 1.0,
            prefilled_gas_pressure = 5.0e-3,
            snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{Float64}(config)
        RP.flags.Atomic_Collision = true
        RP.flags.src = true
        initialize!(RP)
        RP.plasma.Te_eV .= Te_eV
        RP.plasma.ne .= ne
        RP.plasma.ni .= ne
        RP.plasma.ue_para .= -1.0e5
        RP.fields.E_para_tot .= -50.0
        update_RRCs!(RP)
        return RP
    end
end

@testitem "RRC accessors: scalars pulled off the abstract eRRCs field are concrete" setup = [TypeStabilityFixtures] begin
    RP = ts_RAPID()

    # Bare calls first, so a failure names the accessor rather than a comparison.
    # Wrapped in a closure taking the object as an ARGUMENT: `@inferred` on a
    # captured global would see `Any` for the object itself and pin nothing.
    floor_of(rp) = mean_energy_floor(rp)
    bounds_of(rp, sym) = erg_axis_bounds(rp, sym)

    @test @inferred(floor_of(RP)) isa Float64
    @test @inferred(bounds_of(RP, :K_iz)) isa Tuple{Float64, Float64}

    # Every `(E/p, Ē)` surface `update_rate_jacobian!` is called with, so a new ledger
    # column cannot be added through a path that skips the barrier.
    for sym in (:K_iz, :K_diss_iz, :K_exc, :K_mom, :Kerg_ela, :Kerg_exc, :Kerg_diss_exc)
        @test @inferred(bounds_of(RP, sym)) isa Tuple{Float64, Float64}
    end

    # The barrier must not merely be inferrable — it must return the right numbers.
    # All `RRC_EoverP_Erg` surfaces share the one `Erg_eV` vector the constructor read,
    # so every column has the same pair, and it is the pair the floor is taken from.
    lo, hi = erg_axis_bounds(RP, :Kerg_ela)
    @test lo == mean_energy_floor(RP)
    @test lo < hi
    @test (lo, hi) == erg_axis_bounds(RP, :K_iz)

    # A non-`(E/p, Ē)` surface is a caller error, not a silent zero.
    @test_throws ArgumentError erg_axis_bounds(RP, :Dissoc_Ionz_legacy)
end

@testitem "RRC accessors: the Ē-axis clamp broadcast is not run on abstract scalars" setup = [TypeStabilityFixtures] begin

    # `update_rate_jacobian!` clamps its query onto the Ē axis with
    # `clamp.(mean_Ke_eV, Ē_lo, Ē_hi)`. That is a whole-grid broadcast with two scalar
    # operands, and it is the specific place the abstract field read used to land: an
    # `::AbstractFloat` scalar there costs the kernel its specialization.
    #
    # Pinned as the exact expression the production path builds, rather than by
    # inspecting `update_rate_jacobian!` itself — that function returns its `out`
    # argument, so its own return type is inferrable whatever happens inside it and
    # `@inferred` on it would pass while the interior stayed dynamic.
    RP = ts_RAPID()

    clamp_query(rp, sym) = begin
        _, mean_Ke_eV = _eRRC_query_point(rp)
        lo, hi = erg_axis_bounds(rp, sym)
        clamp.(mean_Ke_eV, lo, hi)
    end

    @test @inferred(clamp_query(RP, :Kerg_ela)) isa Matrix{Float64}
    @test @inferred(clamp_query(RP, :K_iz)) isa Matrix{Float64}

    # And the values are a real clamp, not an accidental identity — otherwise the
    # assertion above would hold for a broadcast that does nothing.
    lo, hi = erg_axis_bounds(RP, :Kerg_ela)
    q = clamp_query(RP, :Kerg_ela)
    @test all(lo .<= q .<= hi)
end
