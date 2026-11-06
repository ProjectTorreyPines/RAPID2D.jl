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
    LineCircuit(ends = IonSupplyEnds(); n_cut = 1e-3)

Each open line is a circuit: `J_para/B` is one value along it, the residual field is `J_para η`, the
line EMF `∫ E_para,drive dl` against the line resistance `∫ B η dl` and the end condition `ends`.
The shielding fraction is a result. Field-line following keeps the line → node quadrature it needs.
Both integrals run over the line's plasma part, the nodes above `n_cut` of the densest in-wall node:
the gap between the plasma and the wall carries neither (a declared approximation).
"""
struct LineCircuit{E <: LineEnds} <: OpenLineShielding
    ends::E
    n_cut::Float64
end
LineCircuit(ends::LineEnds = IonSupplyEnds(); n_cut::Real = 1.0e-3) = LineCircuit{typeof(ends)}(ends, Float64(n_cut))

"""
    OpenLineCircuitState{FT}(NR, NZ, n_faces)

The line circuit in a run ([`open_line_circuit!`](@ref)): on every node of an open line that ends
on the wall both ways, the line's EMF `line_emf` [V], resistance `line_resistance` [Ω m²], current
per flux `Jpara_over_B` [A/(m² T)], its end limit `Jpara_over_B_limit` (Inf without one), the
residual field `E_para_res` [V/m] and the shielding `gamma_local = 1 − E_res/E_drive`; NaN
elsewhere. `applies` marks those nodes. `ion_wall_charge_flux[f]` [A/m²] is what the ions' last
continuity step carried into wall face `f`, Σ_s Z_s Γ_s, which the ion-supply end reads.
"""
mutable struct OpenLineCircuitState{FT <: AbstractFloat}
    applies::Matrix{Bool}
    line_emf::Matrix{FT}
    line_resistance::Matrix{FT}
    Jpara_over_B::Matrix{FT}
    Jpara_over_B_limit::Matrix{FT}
    E_para_res::Matrix{FT}
    gamma_local::Matrix{FT}
    ion_wall_charge_flux::Vector{FT}
end
OpenLineCircuitState{FT}(NR::Int, NZ::Int, n_faces::Int) where {FT <: AbstractFloat} = OpenLineCircuitState{FT}(
    falses(NR, NZ), fill(FT(NaN), NR, NZ), fill(FT(NaN), NR, NZ), fill(FT(NaN), NR, NZ), fill(FT(NaN), NR, NZ),
    fill(FT(NaN), NR, NZ), fill(FT(NaN), NR, NZ), zeros(FT, n_faces),
)
