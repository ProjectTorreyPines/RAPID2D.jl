# How the open-line charge sheets' mean field is shaped, as types only. They sit here, before
# `types.jl`, so the flags can hold them; the methods are in `fields.jl` (`mean_sheet_target`).

"""
    MeanSheetFieldPolicy

The closure of the open-line charge sheets' mean poloidal field `Epol_mean` (Ē_pol), held by
`flags.mean_sheet_field`. The field is the least of three: the requirement that would cancel the
driven E∥, `Epol_req = |E∥,drive| B/B_pol`; what the plasma holds, `E_Debye`; and the policy's
saturation ceiling, what the field's geometry lets the sheets sustain in the mean. One field sets both
the residual E∥ (the heating) and the mean E×B drift across the lines; the cancelled share
γ = Ē_pol B_pol/(|E∥,drive| B) is a result, not an input.

The policy shapes the mean field alone: the critical densities `nc_para`, `nc_perp`, the closures'
weights, the mixing amplitude `Erad_fluc` and `γ_shape_fac` keep the local pitch's reference under
every policy.
"""
abstract type MeanSheetFieldPolicy end

"""
    LocalPitch()

**The default.** Ceiling `γ₀ E₀`, uniform in the field's geometry: `E₀ = |E∥,drive| B/⟨B_pol⟩` is the
node's requirement at the in-wall mean ⟨B_pol⟩ and γ₀ `config.charge_sheet_shape_factor`. Written as
`γ Epol_req` with γ = min(1, γ₀ B_pol/⟨B_pol⟩): the cancellation fails in a round pocket around a
poloidal null and is complete wherever B_pol exceeds ⟨B_pol⟩/γ₀.
"""
struct LocalPitch <: MeanSheetFieldPolicy end

"""
    LineMinPitch()

The local pitch's shape with the least poloidal field met along the node's whole line
(`flf.min_Bpol`): `min(1, γ₀ min B_pol/⟨B_pol⟩) Epol_req`, never above the local pitch's ceiling
`γ₀ E₀`. One share along a line, so a line that passes a null is poorly cancelled all the way to the
wall, a band along the separatrix. Sensitive to how close a traced line comes to the null.
"""
struct LineMinPitch <: MeanSheetFieldPolicy end

"""
    LineAvgSaturation(; p = 2, coefficient = 1)

Ceiling `coefficient E₀ B_eff/(B_eff + ⟨B_pol⟩)`: the node's requirement at the in-wall mean,
`E₀ = |E∥,drive| B/⟨B_pol⟩`, times a factor of the line's geometry alone, with
`B_eff = ⟨B_pol⁻ᵖ⟩^(-1/p)` the line's effective poloidal field averaged over its RZ length. No shape
factor γ₀. A line that passes a weak poloidal field saturates low all the way to the wall (a smooth band along
the separatrix), and the field stays finite at a null, where it is the ceiling itself.

- `p ≥ 1` weighs the line's weakest stretch: 1 is the mean of B/B_pol the connection length holds, a
  large `p` approaches the least pitch.
- `coefficient > 0` scales the ceiling: with 1 a uniform poloidal field cancels one half of a uniform
  drive, by construction; the requirement and the Debye field still bound the field above.

Both are choices of the model, not constants of the physics. The field-line following integrates
`B_pol⁻ᵖ` (`mean_sheet_line_integrand`); a following that did not carry it gives `p = 1` from the
lengths.
"""
struct LineAvgSaturation <: MeanSheetFieldPolicy
    p::Float64
    coefficient::Float64
    function LineAvgSaturation(p::Real, coefficient::Real)
        p, coefficient = Float64(p), Float64(coefficient)
        (isfinite(p) && p >= 1) || throw(ArgumentError("LineAvgSaturation: p = $p; it must be finite and at least 1"))
        (isfinite(coefficient) && coefficient > 0) ||
            throw(ArgumentError("LineAvgSaturation: coefficient = $coefficient; it must be finite and positive"))
        return new(p, coefficient)
    end
end
LineAvgSaturation(; p::Real = 2.0, coefficient::Real = 1.0) = LineAvgSaturation(p, coefficient)
