# Shared fixtures for the regression scenarios in this directory: baseline config, the
# toroidal field setup, and density-profile builders.
#
# RULE: only plumbing common to two or more scenarios belongs here. Scenario-defining
# settings — physics flags, grid, timestep, plasma parameters, thresholds — stay in the
# @testitem body, where a reader can see what the test actually tests.

@testsnippet RegressionCommon begin
    using RAPID2D.Statistics

    # Baseline config: the infrastructure defaults every scenario shares.
    # `device_Name = "manual"` skips the device-file lookup; `Output_path` is a per-run
    # temp dir so snapshot writers never touch the repo. cleanup=false is REQUIRED —
    # the RAPID constructor opens ADIOS handles here (src/types.jl) that a finalizer
    # closes at a GC-determined time, so the directory must outlive the object.
    function regression_config(; kwargs...)
        return SimulationConfig{Float64}(;
            device_Name = "manual",
            Output_path = mktempdir(; cleanup = false),
            kwargs...
        )
    end

    # Pure toroidal magnetic field, shared by all three regression scenarios.
    #
    # `E0` [V/m] is the applied toroidal electric field referenced to the mean major
    # radius. Each scenario passes it explicitly: loop-voltage driven (E0 > 0) vs coil
    # driven (E0 = 0) is a defining property of the scenario, not a default.
    function setup_toroidal_field!(RP::RAPID; E0::Real, verbose::Bool = false)
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
    # Pure geometry: they return an array and touch no RP state, so the scenario
    # (centre, radius, peak density, background floor) stays at the call site.

    # Uniform column of radius `radius`, `background` outside it.
    function tophat_blob(G; cenR, cenZ = 0.0, radius, n0, background = 0.0)
        r = @. sqrt((G.R2D - cenR)^2 + (G.Z2D - cenZ)^2)
        return @. ifelse(r < radius, n0, background)
    end

    # Gaussian blob with 1-sigma width `radius`.
    function gaussian_blob(G; cenR, cenZ = 0.0, radius, n0)
        r2 = @. (G.R2D - cenR)^2 + (G.Z2D - cenZ)^2
        return @. n0 * exp(-r2 / (2 * radius^2))
    end
end
