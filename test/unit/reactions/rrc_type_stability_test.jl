# Type-stability pins for the RRC accessors.
#
# `RAPID.eRRCs` and `RAPID.iRRCs` are declared as the ABSTRACT `AbstractSpeciesRRCs{FT}`
# (`types.jl`), so a direct field read off them infers `Any`, and a non-concrete scalar
# handed to a whole-grid broadcast makes the CALL SITE unspecialisable.
#
# **What that costs, measured — not what it is often claimed to cost.** The common
# telling is that `Broadcast.combine_eltypes` fails and the loop drops to per-element
# dynamic dispatch, so the penalty grows with the grid. That is wrong, and it had been
# repeated through three comments and a PR description here before anyone measured it.
# A `Broadcasted` is built from VALUES, so at run time its arg tuple is
# `Tuple{Matrix{Float64}, Float64, Float64}` and `combine_eltypes` returns a concrete
# `Float64` — the kernel that actually executes IS specialised. Only the compiler's
# static view is lost. Measured on the pattern in isolation:
#
#     N          abstract    concrete   ratio   extra alloc
#     2 500        872 ns      569 ns    1.53        32 B
#     40 000      9916 ns     6666 ns    1.49        32 B
#     160 000    38625 ns    26458 ns    1.46        32 B
#
# The extra allocation is CONSTANT and the ratio SHRINKS toward ~1.46. That is one
# dynamic dispatch plus a fixed box plus lost inlining — a constant-factor tax on a
# memory-bound loop, not a per-element blowup. Worth fixing; not worth overstating, and
# never worth asserting as a ratio in a test.
#
# So the reason to keep these barriers is TYPE STABILITY as a contract: an `Any` that
# escapes a function propagates into its callers' broadcasts too (see the ion case
# below, whose `Any` return reaches two more grid broadcasts in
# `update_ion_power_jacobian!`). Every accessor that pulls a scalar off those fields
# must carry a concretising `::FT`. These tests are what stop one from silently losing
# it.
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
    # `Ti_eV` defaults to 1 eV, not the 0.026 eV `initialize!` leaves: the ion elastic
    # `∂K/∂T` is genuinely flat at room temperature (measured 0.0 across the grid, and
    # 6.1e-16 at 1 eV), so a fixture left at the default would let a type assertion pass
    # on a function returning all zeros.
    function ts_RAPID(; Te_eV = 5.0, ne = 1.0e16, Ti_eV = 1.0)
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
        RP.plasma.Ti_eV .= Ti_eV
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

@testitem "RRC accessors: ion_rate_jacobian returns a concrete Matrix" setup = [TypeStabilityFixtures] begin
    using RAPID2D: ion_rate_jacobian, t_axis_bounds

    # The twin of the `update_rate_jacobian!` case, 430 lines earlier in the same file
    # and reached through `RP.iRRCs` instead of `RP.eRRCs`. Found by sweeping for the
    # pattern rather than by reading the diff, which is why it outlived the first fix.
    #
    # Two separate leaks have to be closed here, and closing only the obvious one
    # leaves the return type `Any`:
    #
    #   1. `first(rrc.T_eV)` off the abstract field — the same `::FT` as elsewhere.
    #   2. `RRC_T_ud` declares `itp` and `dK_dT` with NO TYPE AT ALL, so `rrc.dK_dT(…)`
    #      is `Any` however well the scalars are pinned.
    #
    # Unlike `update_rate_jacobian!` — which returns its `out` ARGUMENT and so has an
    # inferrable return type no matter how dynamic its interior — this function builds
    # its result, so `@inferred` on the function itself is the honest pin.
    RP = ts_RAPID()

    jac_of(rp, sym) = ion_rate_jacobian(rp, sym)
    tbounds_of(rp, sym) = t_axis_bounds(rp, sym)

    # The scalar layer needs its OWN assertion. `ion_rate_jacobian`'s `::Matrix{FT}`
    # makes the return type concrete whether or not the T-axis scalars are pinned, so
    # `@inferred` on the function alone cannot see this leak — verified by mutation:
    # dropping the `::FT` still leaves the function inferring `Matrix{Float64}`.
    for sym in (
            :Elastic, :Charge_Exchange, :Target_Ionization,
            :Projectile_Dissociation, :Particle_Exchange,
        )
        @test @inferred(tbounds_of(RP, sym)) isa Tuple{Float64, Float64}
    end
    lo, hi = t_axis_bounds(RP, :Elastic)
    @test lo < hi
    @test (lo, hi) == t_axis_bounds(RP, :Charge_Exchange)   # one shared T_eV vector

    for sym in (:Elastic, :Charge_Exchange)
        @test @inferred(jac_of(RP, sym)) isa Matrix{Float64}
        @test size(jac_of(RP, sym)) == size(RP.plasma.Ti_eV)
    end

    # Anti-vacuity: the derivative must not be identically zero here, or "it inferred a
    # Matrix{Float64}" would hold for a function that had stopped computing anything.
    @test any(!iszero, ion_rate_jacobian(RP, :Elastic))

    # Outside the T axis the derivative is deliberately zeroed, so the in-range mask is
    # doing work rather than passing everything through.
    RP_cold = ts_RAPID(; Ti_eV = 1.0e-8)    # far below the table's bottom row (1 meV)
    @test @inferred(jac_of(RP_cold, :Elastic)) isa Matrix{Float64}
    @test all(iszero, ion_rate_jacobian(RP_cold, :Elastic))

    # Every field of `H2_Ion_RRCs` is an `RRC_T_ud`, so the `isa` guard inside is
    # unreachable for any symbol that names one — an unknown symbol fails earlier, in
    # `getfield`. Asserted as it behaves rather than as the guard suggests.
    @test_throws FieldError ion_rate_jacobian(RP, :not_a_surface)
end
