# Audit a (T, u_d) reaction-rate table against the cross sections it claims to average.
#
# Recomputes ⟨σ v⟩(T, u_d) from BREAK's published σ(E) under a shifted Maxwellian and
# compares, point by point, with what the file holds. It needs no BREAK binary and no
# knowledge of the generator — only the cross sections and the distribution.
#
# WHY IT EXISTS. The shipped files stored axis 1 = drift while `RRC_T_ud` indexed axis 1 =
# temperature, and both axes had the same length, so nothing raised for as long as the
# files existed. `internal/docs/src/notes/issues/ion-rrc-table-transposed.md` has the
# audit. This script is what turned "the orientation looks wrong" into a number.
#
# WHEN TO RUN IT. Before shipping any regenerated `iRRCs_T_ud.h5`. The intended path for
# the ion cross-section overhaul is: fix the guards in BREAK, regenerate, run this, and
# expect the CHANNELS WITH LIFTED GUARDS TO DISAGREE — that is the change. Everything
# else must still match to a few tenths of a percent, which is what says the regeneration
# did not also move something it was not meant to.
#
#   julia --project=. scripts/check_ion_rrc_tables.jl [path/to/iRRCs_T_ud.h5]
#
# Exits 1 if the axis-order contract is broken or an unguarded channel has drifted.

using HDF5, Printf, Statistics, LinearAlgebra

const ee = 1.602176634e-19
const m_i = 2 * 1.6726219e-27        # H₂⁺
# Two tolerances, because the disagreement is not uniform. The BULK must match to a few
# tenths of a percent — that is quadrature against quadrature. A handful of points near a
# cross section's break energy can be worse, because how the generator handled the elastic
# fit's NaN below 0.767 eV is guesswork (see σ_elastic); measured 3.3 % at worst there
# against a 0.07 % median.
const TOL_MEDIAN = 0.005
const TOL_MAX = 0.05
const TOL_PEAK = 0.01                # max |Δ| normalised by the channel's own peak

# ── BREAK's cross sections, ported from ~/Prog/BREAK/src/c_MCC.cpp (m², LAB eV) ───────
# Keep these in step with that file. Each `return 0.0` below reproduces a guard there;
# the ones marked FIT-DOMAIN are boundaries of a fitted region written as "this reaction
# does not occur", which is the defect the cross-section overhaul is meant to remove.

# FIT-DOMAIN: sqrt(log(E/0.76729)) is NaN below 0.767 eV in C++ (`pow(negative, 0.5)`).
# The generator evidently took it as zero — of three conventions tried, only this one
# reproduces the table (`abs(log(...))` is 225× off at T = 0.17 eV).
function σ_elastic(E)
    E < 0.76729 && return 0.0
    E <= 12.2 && return 1.0 / (5.2368e14 * E^0.84772) * sqrt(log(E / 0.76729)) * 1.0e-4
    E <= 90.9 && return 1.0 / (1.0589e14 * E^1.26118) * sqrt(log(E / 5.17621)) * 1.0e-4
    E <= 676.0 && return 1.0 / (4.8091e12 * E^1.80284) * sqrt(log(E / 43.34855)) * 1.0e-4
    E <= 3159.0 && return 1.0 / (4.4641e12 * E^2.01848) * sqrt(log(E / 8.2799e-12)) * 1.0e-4
    return 0.0
end

# FIT-DOMAIN at 4.9 eV: symmetric resonant charge exchange has no threshold and is
# LARGEST at low velocity. The Rapp–Francis fit is finite down to 0.0796 eV.
σ_cx(E) = E < 4.9 ? 0.0 :
    E <= 10400.0 ? 1.0 / (1.3244e15 * E^0.18467) * sqrt(log(E / 0.07955)) * 1.0e-4 :
    E <= 199000.0 ? (2.673e-18 + 5.8871e-16 * exp(-(E + 273.98867) / 39085.72374)) * 1.0e-4 :
    0.0

σ_cx_unguarded(E) = (E < 0.0796 || E > 199000.0) ? 0.0 :
    E <= 10400.0 ? 1.0 / (1.3244e15 * E^0.18467) * sqrt(log(E / 0.07955)) * 1.0e-4 :
    (2.673e-18 + 5.8871e-16 * exp(-(E + 273.98867) / 39085.72374)) * 1.0e-4

# FIT-DOMAIN at 1.64 eV: measured to proceed at the Langevin rate 3800× below it
# (Allmendinger 2016). Langevin continuation: σ(E) = σ(1.64)·sqrt(1.64/E).
σ_px(E) = (1.64 <= E < 20.1) ? 3.15e-19 * exp(-4.26e-1 * E) : 0.0
σ_px_unguarded(E) = E >= 20.1 ? 0.0 :
    E >= 1.64 ? 3.15e-19 * exp(-4.26e-1 * E) :
    3.15e-19 * exp(-4.26e-1 * 1.64) * sqrt(1.64 / max(E, 1.0e-12))

const GUARDED = (
    Elastic = (σ = σ_elastic, unguarded = nothing),
    Charge_Exchange = (σ = σ_cx, unguarded = σ_cx_unguarded),
    Particle_Exchange = (σ = σ_px, unguarded = σ_px_unguarded),
)

# ── ⟨σv⟩ under a shifted Maxwellian against a target at rest ─────────────────────────
# Ions drift at u with a Maxwellian spread at T. The speed density of a drifting 3-D
# Maxwellian is
#
#   F(v) = (v/u)·√(a/π)·[exp(−a(v−u)²) − exp(−a(v+u)²)],   a = m/2kT,   ∫F dv = 1
#
# and ⟨σv⟩ = ∫₀^∞ σ(E(v))·v·F(v) dv with E = ½mv²/e — the projectile's LAB energy, which
# is what BREAK passes its cross sections.
function gauss_legendre(n)
    β = [k / sqrt(4k^2 - 1) for k in 1:(n - 1)]
    vals, vecs = eigen(diagm(1 => β, -1 => β))
    return vals, 2 .* vec(vecs[1, :]) .^ 2
end
const GLX, GLW = gauss_legendre(200)

function panel_integrate(f, edges)
    s = 0.0
    for i in 1:(length(edges) - 1)
        a, b = edges[i], edges[i + 1]
        b <= a && continue
        h, c = (b - a) / 2, (b + a) / 2
        s += h * sum(GLW[k] * f(c + h * GLX[k]) for k in eachindex(GLX))
    end
    return s
end

"""
`⟨σv⟩` for one `(T, u)`. Panel edges sit ON the cross sections' break energies and on
multiples of the thermal speed either side of the drift, so no panel straddles a
discontinuity and none is wasted on the far tail.
"""
function rate_coefficient(σ, T_eV, u)
    a = m_i / (2 * T_eV * ee)
    vth = sqrt(T_eV * ee / m_i)
    F0(v) = 4π * (a / π)^1.5 * v^2 * exp(-a * v^2)
    Fu(v) = (v / u) * sqrt(a / π) * (exp(-a * (v - u)^2) - exp(-a * (v + u)^2))
    f = (u < 1.0e-6 * vth) ? F0 : Fu

    v_of(E) = sqrt(2 * E * ee / m_i)
    hi = max(10vth + 5u, v_of(4000.0))
    edges = sort!(
        unique!(
            vcat(
                0.0, hi,
                [v_of(E) for E in (0.0796, 0.76729, 1.64, 4.9, 12.2, 20.1, 90.9, 676.0, 3159.0)],
                [max(u - k * vth, 0.0) for k in 0:6], [u + k * vth for k in 0:6],
            )
        )
    )
    filter!(v -> 0.0 <= v <= hi, edges)
    return panel_integrate(v -> σ(0.5 * m_i * v^2 / ee) * v * f(v), edges)
end

# ── the audit ─────────────────────────────────────────────────────────────────────────
path = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(dirname(@__DIR__), "RRC_data", "iRRCs_T_ud.h5")
failures = String[]

# Same rule the loader uses: the file declares its layout, and no declaration means the
# legacy one. Reading it any other way would make this script agree with a bug.
T, u, SURF, stamped = h5open(path) do f
    order = haskey(attrs(f), "axis_order") ? read_attribute(f, "axis_order") : nothing
    flip = order != "T_eV,ud_para"
    (
        read(f, "T_eV"), read(f, "ud_para"),
        Dict(
            k => (flip ? permutedims(read(f, k)) : read(f, k))
                for k in keys(f) if k ∉ ("T_eV", "ud_para")
        ),
        order,
    )
end

@printf(
    "\n%s\n  T_eV %d pts %.3g … %.3g\n  ud_para %d pts %.3g … %.3g\n",
    path, length(T), T[1], T[end], length(u), u[1], u[end]
)
@printf(
    "  axis_order = %s%s\n", stamped === nothing ? "ABSENT" : "\"$stamped\"",
    stamped == "T_eV,ud_para" ? "" : "  → transposed on read (legacy layout)"
)
stamped === nothing || stamped in ("T_eV,ud_para", "ud_para,T_eV") ||
    push!(failures, "unknown axis_order \"$stamped\"")
for (k, v) in SURF
    size(v) == (length(T), length(u)) ||
        push!(failures, "$k is $(size(v)); axis 1 must be T_eV, axis 2 ud_para")
end

# A coarse sweep of both axes: enough points to catch a transposition or a resampling,
# few enough to run in seconds.
iTs = round.(Int, range(1, length(T), 9))
ius = round.(Int, range(1, length(u), 5))

# Two numbers, because one cannot carry both questions. A RELATIVE deviation is the
# right measure where the rate matters and meaningless in an exponentially small tail
# (two values that are both 1e-50 can differ by 100 % and agree perfectly for every
# purpose). So: relative deviation restricted to the points carrying the surface, and
# absolute deviation normalised by the channel's own peak everywhere else.
@printf("\n╔═ table vs ⟨σv⟩ recomputed from BREAK's σ(E) ═══════════════════════════════\n")
@printf(
    "║ %-18s %7s %11s %11s %11s %8s\n",
    "channel", "pts", "med rel*", "max rel*", "max |Δ|/pk", "verdict"
)
@printf("║ %s\n", "* over points holding ≥ 0.1 % of the channel's peak rate")
for (name, spec) in pairs(GUARDED)
    haskey(SURF, String(name)) || continue
    A = SURF[String(name)]
    peak = maximum(A)
    rel, absn, worst = Float64[], 0.0, (0.0, 0.0, 0.0)
    for iT in iTs, iu in ius
        tbl = A[iT, iu]
        calc = rate_coefficient(spec.σ, T[iT], u[iu])
        d = abs(calc - tbl)
        if d / peak > absn
            absn = d / peak
            worst = (T[iT], u[iu], tbl == 0 ? Inf : calc / tbl)
        end
        tbl > 1.0e-3 * peak && push!(rel, d / tbl)
    end
    med = isempty(rel) ? 0.0 : median(rel)
    ok = med < TOL_MEDIAN && maximum(rel; init = 0.0) < TOL_MAX && absn < TOL_PEAK
    ok || push!(
        failures,
        "$name: median relative $(round(100med, digits = 3)) %, " *
            "max relative $(round(100maximum(rel; init = 0.0), digits = 2)) %, " *
            "max |Δ|/peak $(round(100absn, digits = 2)) % " *
            "(worst near T = $(round(worst[1], sigdigits = 3)) eV, u = $(round(worst[2], sigdigits = 3)) m/s)"
    )
    @printf(
        "║ %-18s %7d %11.2e %11.2e %11.2e %8s\n",
        name, length(rel), med, maximum(rel; init = 0.0), absn,
        ok ? "ok" : "DRIFTED"
    )
end
println("╚", "─"^78)

# ── what the fit-domain guards cost, for the overhaul to aim at ──────────────────────
@printf("\n╔═ what removing each FIT-DOMAIN guard would change ═════════════════════════\n")
@printf("║ These are not failures. They are the size of the physics the guards suppress,\n")
@printf("║ recomputed at zero drift so the numbers are read against T_i directly.\n")
for (name, spec) in pairs(GUARDED)
    spec.unguarded === nothing && continue
    @printf("║\n║ %s\n║ %10s %14s %14s %12s\n", name, "T_i [eV]", "as shipped", "guard lifted", "factor")
    for iT in round.(Int, range(1, length(T), 12))
        g = rate_coefficient(spec.σ, T[iT], 0.0)
        n = rate_coefficient(spec.unguarded, T[iT], 0.0)
        (g < 1.0e-30 && n < 1.0e-30) && continue
        @printf(
            "║ %10.4g %14.4e %14.4e %12s\n", T[iT], g, n,
            g > 1.0e-30 ? @sprintf("%.3g×", n / g) : "∞"
        )
    end
end
println("╚", "─"^78)

if isempty(failures)
    @printf("\n✅ %s is consistent with its cross sections.\n\n", basename(path))
else
    @printf("\n❌ %s:\n", basename(path))
    foreach(f -> println("   • ", f), failures)
    println()
    exit(1)
end
