# Shared pieces of the coupled-step examples: a current-carrying column in a pure toroidal
# field, toroidal loops around it, the plasma flux each loop links, and the lumped circuit
# model the plots compare against. `include("common.jl")` from a script in this directory.
#
# A step refreshes the collision rates at its end, before `callback_after_step` runs. A
# callback that changes n, T or u must refresh them itself, or the next step uses the rates
# of the old state: `RAPID2D.update_transport_quantities!(RP)`.

include(joinpath(@__DIR__, "..", "common.jl"))

# A uniform column in a pure toroidal field (Eϕ = E0·R̄/R) with Coulomb drag as its only
# friction: the column of current_diffusion.jl, at Te = 1 eV so that its L/R time is about
# 0.2 ms. Ampère and the inductive E∥ are on; Ampère runs once |I_p| ≥ `threshold`
# (0: from the first step). No sources and fixed temperatures. `moving = true` turns on the
# mean E×B drift and the advection of n and u∥ by it, so that a script can push the column.
function column(
        name; E0 = 0.3, Te = 1.0, n0 = 1.0e16, cenR = 1.5, cenZ = 0.0, radius = 0.3,
        threshold = 0.0, dt = 5.0e-6, t_end = 1.0e-3, moving = false,
    )
    config = SimulationConfig{Float64}(;
        device_Name = "manual", manual = pure_toroidal(E0), NR = 30, NZ = 50, R0B0 = 3.0,
        prefilled_gas_pressure = 0.0,          # vacuum: no neutrals
        dt, t_end_s = t_end, snap0D_Δt_s = 10dt, snap2D_Δt_s = t_end,
        Output_path = output_dir(name),
    )
    RP = setup(
        config;
        Ampere = true, Ampere_Itor_threshold = threshold, E_para_self_EM = true,
        ud_evolve = true, Coulomb_Collision = true,
        Atomic_Collision = false, src = false, convec = moving, diffu = false,
        Te_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
        Include_ud_convec_term = moving, Include_ud_diffu_term = false, Include_ud_pressure_term = false,
        Include_Te_convec_term = false,
        E_para_self_ES = false, mean_ExB = moving, turb_ExB_mixing = false,
        FLF_nstep = 100_000,                   # no field-line tracing
    )
    set_column!(RP, tophat(RP.G; cenR, cenZ, radius, n0); Te_eV = Te)
    return RP
end

# A toroidal loop of minor radius `a` at (r, z), L = μ0 r (ln(8r/a) − 7/4); returns L.
# R = 1e-12 Ω makes it superconducting on these time scales. A nonzero V powers it. Call
# initialize_coil_system! after the last one.
function add_loop!(RP, r, z; a = 0.05, R = 1.0e-12, V = 0.0, I0 = 0.0, name = "loop")
    L = RP.config.constants.μ0 * r * (log(8r / a) - 7 / 4)
    coil = Coil{Float64}(;
        location = (r = r, z = z), area = π * a^2, resistance = R, self_inductance = L,
        is_powered = V != 0, is_controllable = false, name, current = I0, voltage_ext = V,
    )
    add_coil!(RP.coil_system, coil)
    return L
end

# Toroidal current density of the present state, electrons and ions.
function current_density(RP)
    p, c = RP.plasma, RP.config.constants
    Zi = Float64(RAPID2D.bulk_ion_charge(RP))
    return @. (c.qe * p.ne * p.ue_para + p.ni * (c.ee * Zi) * p.ui_para) * RP.fields.bϕ
end
plasma_current(RP, J) = sum(J) * RP.G.dR * RP.G.dZ

# Plasma flux through each loop, 2π Σ G(r_loop; r) J dA.
flux_at_coils(RP, J) = 2π .* (RP.coil_system.Green_grid2coils * vec(J)) .* (RP.G.dR * RP.G.dZ)

# ── the lumped model: the column as one plasma loop ────────────────────────────────
# One loop carrying a uniform current over the cells the column fills (area S, a = √(S/π)):
# L_p = μ0 R (ln(8R/a) − 7/4); the electrons' inertia L_kin = mₑ 2πR / (n e² S) at the present
# density; and M, the flux through each coil per ampere of that current.
function lumped_column(RP; R0 = 1.5)
    G, pla, c = RP.G, RP.plasma, RP.config.constants
    col = findall(>(0), pla.ne)
    S = length(col) * G.dR * G.dZ
    J = zeros(size(pla.ne))
    J[col] .= 1.0
    return (;
        L_p = c.μ0 * R0 * (log(8R0 / sqrt(S / π)) - 7 / 4),
        L_kin = c.me * 2π * R0 / (mean(pla.ne[col]) * c.ee^2 * S),
        M = RP.coil_system.n_total > 0 ? flux_at_coils(RP, J) ./ plasma_current(RP, J) : Float64[],
        S,
    )
end

# Resistance of the column to a uniform loop voltage, 1/R_p = Σ σ dA / (2πR), with the Coulomb
# conductivity σ = n e² / (mₑ ν), which does not depend on n (ν ∝ n).
function column_resistance(RP)
    G, pla, c = RP.G, RP.plasma, RP.config.constants
    col = findall(>(0), pla.ne)
    return 1 / (sum(@. c.ee^2 * pla.ne[col] / (c.me * pla.ν_ei_eff[col] * 2π * G.R2D[col])) * G.dR * G.dZ)
end

# Circuits in flux form, dΨ/dt = V − R I with Ψ = L I, by RK4 over the times `t` from
# I(t[1]) = I0; L(t) and R(t) are matrices given as functions of time. Where L jumps, Ψ stays
# continuous, as the circuit equations require.
function circuits(L, R, V, I0, t; substeps = 20)
    f(τ, Ψ) = V .- R(τ) * (L(τ) \ Ψ)
    Ψ = L(t[1]) * I0
    I = [I0]
    for k in 2:length(t)
        h = (t[k] - t[k - 1]) / substeps
        for j in 1:substeps
            τ = t[k - 1] + (j - 1) * h
            k1 = f(τ, Ψ)
            k2 = f(τ + h / 2, Ψ .+ h / 2 .* k1)
            k3 = f(τ + h / 2, Ψ .+ h / 2 .* k2)
            k4 = f(τ + h, Ψ .+ h .* k3)
            Ψ = Ψ .+ h / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        end
        push!(I, L(t[k]) \ Ψ)
    end
    return I
end

# Where things sit: the current density (scaled to its peak), the wall, and the loops.
function plot_layout(RP; J = current_density(RP), title = "layout")
    G = RP.G
    rc = [c.location.r for c in RP.coil_system.coils]
    zc = [c.location.z for c in RP.coil_system.coils]
    Jmax = maximum(abs, J)
    p = heatmap(
        G.R1D, G.Z1D, permutedims(Jmax > 0 ? J ./ Jmax : J);
        c = :balance, clims = (-1, 1), aspect_ratio = :equal, colorbar_title = "J / max|J|",
        xlims = (min(G.R1D[1], minimum(rc) - 0.1), max(G.R1D[end], maximum(rc) + 0.1)),
        ylims = extrema(G.Z1D), xlabel = "R (m)", ylabel = "Z (m)", title, titlefontsize = 10,
        framestyle = :box,
    )
    plot!(p, vcat(RP.wall.R, RP.wall.R[1]), vcat(RP.wall.Z, RP.wall.Z[1]); c = :gray40, lw = 1, label = "wall")
    scatter!(p, rc, zc; c = :orange, ms = 4, msw = 0, label = "loops")
    return p
end
