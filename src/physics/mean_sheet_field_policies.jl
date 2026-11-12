# How the open-line charge sheets' mean field is shaped, as types only. They sit here, before
# `types.jl`, so the flags can hold them; the methods are in `fields.jl` (`mean_sheet_target`).

"""
    MeanSheetFieldPolicy

The closure of the open-line charge sheets' mean poloidal field `Epol_mean` (Ē_pol), held by
`flags.mean_sheet_field`: which share γ of the field that would cancel the driven E∥,
`Epol_req = |E∥,drive| B/B_pol`, the sheets sustain in the mean, before the Debye cap. One field
sets both the residual E∥ (the heating) and the mean E×B drift across the lines.

The policy shapes the mean field alone: the critical densities `nc_para`, `nc_perp`, the closures'
weights and the mixing amplitude `Erad_fluc` keep the local pitch's reference under every policy.
"""
abstract type MeanSheetFieldPolicy end

"""
    LocalPitch()

**The default.** γ = min(1, γ₀ B_pol/⟨B_pol⟩) with the node's own poloidal field, ⟨B_pol⟩ the in-wall
mean and γ₀ `config.charge_sheet_shape_factor`: the cancellation fails in a round pocket around a
poloidal null and is complete wherever B_pol exceeds ⟨B_pol⟩/γ₀.
"""
struct LocalPitch <: MeanSheetFieldPolicy end

"""
    LineMinPitch()

γ = min(1, γ₀ min B_pol/⟨B_pol⟩) with the least poloidal field met along the node's whole line
(`flf.min_Bpol`): one value along a line, so a line that passes a null is poorly cancelled all the way
to the wall, a band along the separatrix. Sensitive to how close a traced line comes to the null.
"""
struct LineMinPitch <: MeanSheetFieldPolicy end

"""
    LineAvgPitch(; p = 2)

γ = B_eff/(B_eff + ⟨B_pol⟩), with the line's effective poloidal field `B_eff = ⟨B_pol⁻ᵖ⟩^(-1/p)`
averaged over the line's RZ length: the line's requirement against the device's mean one,
γ = 1/(1 + M_p/E₀), with no shape factor. Smooth along the separatrix, never complete (a uniform
poloidal field gives one half), and one value along a line. `p ≥ 1` weighs the line's weakest
stretch: 1 is the mean of B/B_pol the connection length holds, a large `p` approaches the least
pitch. A choice of the model, not a constant of the physics.

The field-line following integrates `B_pol⁻ᵖ` (`mean_sheet_line_integrand`); a following that did not
carry it gives `p = 1` from the lengths.
"""
struct LineAvgPitch <: MeanSheetFieldPolicy
    p::Float64
    function LineAvgPitch(p::Real)
        (isfinite(p) && p >= 1) || throw(ArgumentError("LineAvgPitch: p = $p; it must be finite and at least 1"))
        return new(Float64(p))
    end
end
LineAvgPitch(; p::Real = 2.0) = LineAvgPitch(p)
