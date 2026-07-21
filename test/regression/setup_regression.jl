# Shared fixtures for the regression scenarios in this directory.
#
# WHAT BELONGS HERE (and what does not)
# -------------------------------------
# Only plumbing that is common to two or more scenarios: baseline test configuration,
# the shared field setup, and generic density-profile builders.
#
# Scenario-defining settings — which physics flags are on, grid size, timestep, the
# plasma parameters, the pass/fail thresholds — deliberately live in each @testitem
# body instead. A reader must be able to see what a test is actually testing without
# following a call into this file. "Everything off except Ampère" IS the inductance
# test; hiding it here would make the test unreadable.
#
# A @testsnippet body is include_string'd into each @testitem's private module, so
# nothing here leaks between files and no method can be overwritten.

@testsnippet RegressionCommon begin
    using RAPID2D.Statistics

    # Baseline configuration for a regression scenario.
    #
    # Supplies ONLY the two infrastructure defaults every scenario shares:
    #   device_Name = "manual"  — no device file lookup
    #   Output_path             — a per-run temp dir, so snapshot writers never touch
    #                             the repo. cleanup=false is REQUIRED: the RAPID
    #                             constructor opens ADIOS handles here (src/types.jl)
    #                             that are closed by a FINALIZER at a GC-determined
    #                             time, so the directory must outlive the object. A
    #                             self-deleting tempdir aborts the process instead.
    # Everything else — NR, NZ, dt, t_end_s, R0B0, pressures, snapshot intervals — is
    # scenario physics and is passed explicitly at the call site.
    function regression_config(; kwargs...)
        return SimulationConfig{Float64}(;
            device_Name = "manual",
            Output_path = mktempdir(; cleanup=false),
            kwargs...
        )
    end

    # Pure toroidal magnetic field, shared by all three regression scenarios.
    #
    # `E0` is the applied toroidal electric field referenced to the mean major radius
    # [V/m]. It is passed explicitly by each scenario rather than defaulted silently,
    # because whether the plasma is loop-voltage driven (E0 > 0) or coil driven
    # (E0 = 0) is a defining property of the scenario.
    #
    # With E0 = 0 the general form below is identically 0.0 everywhere — Jacob = R2D is
    # strictly positive and finite — so it reproduces a bespoke `fill!(Eϕ, 0)` variant
    # bit-for-bit.
    function setup_toroidal_field!(RP::RAPID; E0::Real, verbose::Bool=false)
        # Zero poloidal field components (pure toroidal field)
        fill!(RP.fields.BR, 0.0)
        fill!(RP.fields.BZ, 0.0)
        fill!(RP.fields.BR_ext, 0.0)
        fill!(RP.fields.BZ_ext, 0.0)

        # Set Jacobian for toroidal geometry
        @. RP.G.Jacob = RP.G.R2D

        # Set toroidal field: Bφ = R₀B₀/R
        @. RP.fields.Bϕ = RP.config.R0B0 / RP.G.Jacob

        # Update total field
        @. RP.fields.Bpol = sqrt(RP.fields.BR^2 + RP.fields.BZ^2)
        @. RP.fields.Btot = abs(RP.fields.Bϕ)

        # Update unit vectors
        @. RP.fields.bR = RP.fields.BR / RP.fields.Btot
        @. RP.fields.bZ = RP.fields.BZ / RP.fields.Btot
        @. RP.fields.bϕ = RP.fields.Bϕ / RP.fields.Btot

        # Applied toroidal electric field
        @. RP.fields.Eϕ = E0 * mean(RP.G.R1D) / RP.G.Jacob
        @. RP.fields.Eϕ_ext = RP.fields.Eϕ
        @. RP.fields.E_para_ext = RP.fields.Eϕ * RP.fields.bϕ

        if verbose
            println("  ✓ Toroidal field set: R₀B₀ = $(RP.config.R0B0) T⋅m, E0 = $E0 V/m")
        end

        return RP
    end

    # ── Density profile builders ─────────────────────────────────────────────────
    # Pure geometry: they return an array and touch no RP state, so the SCENARIO
    # (centre, minor radius, peak density, background floor) stays visible at the
    # call site where it matters.

    # Uniform column of radius `radius`, `background` outside it.
    function tophat_blob(G; cenR, cenZ=0.0, radius, n0, background=0.0)
        r = @. sqrt((G.R2D - cenR)^2 + (G.Z2D - cenZ)^2)
        return @. ifelse(r < radius, n0, background)
    end

    # Gaussian blob with 1-sigma width `radius`.
    function gaussian_blob(G; cenR, cenZ=0.0, radius, n0)
        r2 = @. (G.R2D - cenR)^2 + (G.Z2D - cenZ)^2
        return @. n0 * exp(-r2 / (2 * radius^2))
    end
end
