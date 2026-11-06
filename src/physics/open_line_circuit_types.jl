# The open-line shielding closure's policies, as types only. They sit here, before `types.jl`, so
# the flags can hold them; the methods are in `open_line_circuit.jl`.

"""
    OpenLineShielding

How the open field lines' charge sheets shield the driven E∥ (`notes/plans/PLAN_open-line-circuit.md`).
"""
abstract type OpenLineShielding end

"""
    ChargeSheetHeuristic()

**The default.** The sheets' field is a fraction γ = γ₀ B_pol/⟨B_pol⟩ (`config.charge_sheet_shape_factor`)
of the field that would cancel the driven E∥, capped by the field the local plasma holds, E_Debye
(`update_space_charge_weights!`).
"""
struct ChargeSheetHeuristic <: OpenLineShielding end

"""
    LineEnds

What the walls at the two ends of an open line do with its current: the end condition of the
line's circuit ([`LineCircuit`](@ref)).
"""
abstract type LineEnds end

"""
    IonSupplyEnds()

The ion-collecting end passes at most the current its wall's ions supply, by every channel: the
sheath absorbs the ions and turns the electrons back, so `J_para/B ≤ e Σ Z_s Γ_s,wall / (B μ)`, μ the
line's incidence on the face. No parameter.
"""
struct IonSupplyEnds <: LineEnds end

"""
    ConductingEnds()

The ends drop no voltage: the whole line EMF drives current, `J_para/B = V_line / R_line`. The limit
of a wall that passes any current, as the Robin walls do without a circuit. For comparison.
"""
struct ConductingEnds <: LineEnds end

"""
    CappedEnds(Φ)

The ends together drop at most the voltage `Φ` [V]: `J_para/B = (V_line − sgn(V) min(|V_line|, Φ)) / R_line`.
The one-parameter form the BREAK data suggested (`design/open-line-shielding-potential.md` §4g); for comparison.
"""
struct CappedEnds{FT <: AbstractFloat} <: LineEnds
    Φ::FT
end

"""
    LineCircuit(; ends = IonSupplyEnds())

Each open line is a circuit: `J_para/B` is one value along it, the residual field is `J_para η`, the
line EMF `∫ E_para,drive dl` against the line resistance `∫ B η dl` and the end condition `ends`.
The shielding fraction is a result. Field-line following keeps the line → node quadrature it needs.
"""
struct LineCircuit{E <: LineEnds} <: OpenLineShielding
    ends::E
end
LineCircuit(; ends::LineEnds = IonSupplyEnds()) = LineCircuit(ends)
