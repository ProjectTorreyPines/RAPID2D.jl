# The closed-surface flux constraint in a run.
# At the end of each step the basis is rebuilt from the field-line analysis; the next u∥ solve
# takes n u∥/B a function of ψ on the closed region inside the LCFS, below the Ampère gate in the u∥ step
# and above it inside the coupled u∥–ψ block (`BlockConstraint`) or after it (`PostProjection`);
# after continuity u∥ is remapped to the density the step left. The kinetic energy the constraint
# removes and the remap changes is booked.

# the density below which a node is not admitted to the constraint (the per-particle operators' floor)
const CLOSED_SURFACE_N_FLOOR = 1.0

closed_surface_active(RP::RAPID) =
    RP.flags.closed_surface_flow isa SurfaceFluxConstraint && RP.closed_surface.basis !== nothing

"""
    update_closed_surface_basis!(RP) -> RP

Rebuild the constraint's basis from the last field-line analysis: the closed nodes inside the
wall whose density is above the per-particle floor are admissible, the wall cells bound the LCFS,
and the basis functions are merged on the Gram weights B²/n. Without a basis (no single O-point, no closed
level) the step runs free, and the surface flux of the previous solve is forgotten.
`run_simulation!` calls it after every field-line analysis; a caller stepping by hand calls it
after its own.
"""
function update_closed_surface_basis!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    RP.flags.closed_surface_flow isa SurfaceFluxConstraint || return RP
    G, pla, F, s = RP.G, RP.plasma, RP.fields, RP.closed_surface
    A = fill(false, G.NR, G.NZ)
    for k in RP.flf.closed_surface_nids
        G.nodes.state[k] > 0.5 && pla.ne[k] > CLOSED_SURFACE_N_FLOOR && (A[k] = true)
    end
    s.basis = any(A) ? closed_surface_basis(
            G, F.ψ, A; inside = G.cell_state .>= 0, gram_weights = F.Btot .^ 2 ./ max.(pla.ne, CLOSED_SURFACE_N_FLOOR),
        ) : nothing
    s.basis === nothing && empty!(s.solve.Γe_para_over_B)
    return RP
end

"""
    surface_constrained_ue!(RP, A, rhs, θu) -> RP

Solve the u∥ step `A u = rhs` (`A` sparse over all nodes) with the closed-surface constraint
inside the LCFS ([`constrained_momentum_solve`](@ref), on the run's [`ConstrainedSystem`](@ref),
kept while its pattern holds), and write u∥ and the surface flux
([`surface_constraint_terms`](@ref), [`commit_surface_flux!`](@ref)).
"""
function surface_constrained_ue!(RP::RAPID{FT}, A::SparseMatrixCSC{FT}, rhs::AbstractVector, θu::Real) where {FT <: AbstractFloat}
    s, pla = RP.closed_surface, RP.plasma
    t = surface_constraint_terms(RP, θu)
    rhs_c = Vector{FT}(rhs)
    rhs_c[t.cu] .+= t.ionization
    s.u_system = constrained_system(s.u_system, A, s.basis)
    sys = update_constrained_system!(s.u_system, A, s.basis, pla.ne, RP.fields.Btot; t.shift)
    y = constrained_rhs(sys, rhs_c; t.extra)
    x = solve!(similar(y), sys.solver, y)
    commit_surface_flux!(RP, x[(sys.L + 1):end] ./ sys.scale, t.Bp)
    copyto!(pla.ue_para, view(x, 1:sys.L))
    return RP
end

"""
    coupled_block_solver!(RP, A_u_ψ, θu) -> (solver, sys, terms)

The coupled u∥–ψ block's factorized solver for the step. With `SurfaceFluxConstraint(;
coupling = BlockConstraint())` on a closed region, the block with the surface flux as M more
unknowns: the run's [`ConstrainedSystem`](@ref) `sys`, kept while its pattern holds, and the
terms its constrained rows take ([`surface_constraint_terms`](@ref)). Otherwise the plain block's
LU, with `sys` and `terms` `nothing`.
"""
function coupled_block_solver!(RP::RAPID{FT}, A_u_ψ::SparseMatrixCSC{FT}, θu::Real) where {FT <: AbstractFloat}
    if !(closed_surface_active(RP) && RP.flags.closed_surface_flow.coupling isa BlockConstraint)
        factorize!(RP.operators.uψ_solver, A_u_ψ)
        return RP.operators.uψ_solver, nothing, nothing
    end
    cs = RP.closed_surface
    t = surface_constraint_terms(RP, θu)
    cs.block_system = constrained_system(cs.block_system, A_u_ψ, cs.basis)
    sys = update_constrained_system!(cs.block_system, A_u_ψ, cs.basis, RP.plasma.ne, RP.fields.Btot; t.shift)
    return sys.solver, sys, t
end

"""
    surface_constraint_terms(RP, θu) -> (; cu, shift, ionization, extra, Bp)

What the constrained u∥ rows take that the plain step does not, on the nodes `cu` inside the LCFS:
- ionization leaves their decay, `shift` on the diagonal and `ionization` on the right-hand
  side: the newborn electrons' dilution is applied once, by the remap after continuity;
- the pressure enters the Galerkin rows as its generalized force on the stream-function fluxes
  ([`pressure_operator`](@ref) `Bp`), `extra`, in place of the nodal force of the plain rows.
"""
function surface_constraint_terms(RP::RAPID{FT}, θu::Real) where {FT <: AbstractFloat}
    pla, b, dt = RP.plasma, RP.closed_surface.basis, RP.dt
    @unpack me, ee = RP.config.constants
    sel, cu, knot_w_in = inside_lcfs(b)
    ν_iz = pla.ν_en_iz_tot[cu]
    Bp = pressure_operator(b, log.(max.(pla.ne, CLOSED_SURFACE_N_FLOOR)))
    extra = nothing
    if RP.flags.Include_ud_pressure_term
        a_p = calculate_electron_acceleration_by_pressure(RP)
        F_p = -(Bp * (ee .* pla.Te_eV[b.nodes])) ./ me
        extra = dt .* (F_p .- knot_w_in' * (b.V[sel] .* RP.fields.Btot[cu] .* a_p[cu]))
    end
    return (; cu, shift = -θu * dt .* ν_iz, ionization = (1 - θu) * dt .* ν_iz .* pla.ue_para[cu], extra, Bp)
end

"""
    commit_surface_flux!(RP, Γ_over_B, Bp) -> RP

Keep the step's n u∥/B at the knots, `Γ_over_B`, and what the step's transport reads from it: the
stream-function fluxes and the compression `Bp`ᵀK that pairs with the pressure force. Called
while u∥ is still the step's starting drift: the kinetic energy of its part that is not a
surface flux, which the constraint removes, is booked as activation (no surface flux before),
basis change, or projection.
"""
function commit_surface_flux!(RP::RAPID{FT}, Γ_over_B::Vector{FT}, Bp::AbstractMatrix) where {FT <: AbstractFloat}
    s, pla = RP.closed_surface, RP.plasma
    b, sol, led = s.basis, s.solve, s.ledger
    n, B, u = pla.ne, RP.fields.Btot, pla.ue_para
    sel, cu, knot_w_in = inside_lcfs(b)
    Γ_over_B0 = galerkin_flux(b, n, u, B)
    removed = RP.config.constants.me / 2 * sum(b.V[sel] .* n[cu] .* (u[cu] .- (B[cu] ./ n[cu]) .* (knot_w_in * Γ_over_B0)) .^ 2)
    signature = hash((b.nodes, b.knots))
    if isempty(sol.Γe_para_over_B)
        led.activation += removed
    elseif signature != sol.signature
        led.basis_change += removed
    else
        led.projection += removed
    end
    sol.Γe_para_over_B, sol.signature = Γ_over_B, signature
    sol.ΓR, sol.ΓZ = stream_function_fluxes(b, Γ_over_B)
    fill!(sol.compression, zero(FT))
    sol.compression[b.nodes] .= Bp' * Γ_over_B
    return RP
end

"""
    remap_surface_flux!(RP) -> RP

After continuity, put u∥ on the constrained nodes back on the surface flux of the step,
u∥ = (n u∥/B) B / n with the density the step left, so the current follows the surfaces and not
the density; book the kinetic-energy change and update Jϕ.
"""
function remap_surface_flux!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    (closed_surface_active(RP) && !isempty(RP.closed_surface.solve.Γe_para_over_B)) || return RP
    s, pla, F = RP.closed_surface, RP.plasma, RP.fields
    b = s.basis
    me = RP.config.constants.me
    sel, cu, knot_w_in = inside_lcfs(b)
    u_new = (F.Btot[cu] ./ pla.ne[cu]) .* (knot_w_in * s.solve.Γe_para_over_B)
    s.ledger.remap += me / 2 * sum(b.V[sel] .* pla.ne[cu] .* (u_new .^ 2 .- pla.ue_para[cu] .^ 2))
    pla.ue_para[cu] .= u_new
    update_Jϕ!(RP)
    return RP
end

"""
    add_stream_function_transport!(A_adv, RP) -> A_adv

Add the carriage of a per-particle variable by the stream-function fluxes of the last solve to
the advection operator `A_adv` (df/dt = −A_adv f): each face's flux takes the value upwind of it
into the node it enters, M_lk = −Γ/(V_l n_l), M_ll = +Γ/(V_l n_l), with the density of this call,
so the flux carried is Γ whatever the density has become. Nodes at the per-particle floor are
skipped, like the per-particle operators' empty rows.
"""
function add_stream_function_transport!(A_adv::DiscretizedOperator{FT}, RP::RAPID{FT}) where {FT <: AbstractFloat}
    sol, G, n = RP.closed_surface.solve, RP.G, RP.plasma.ne
    nz, k2c, nid = nonzeros(A_adv.matrix), A_adv.k2csc, G.nodes.nid
    function enter!(i, j, Γ, slot)              # Γ > 0 enters node (i, j) from its neighbour at `slot`
        n[i, j] > CLOSED_SURFACE_N_FLOOR || return
        w = Γ / (G.inVol2D[i, j] * n[i, j])
        r = nid[i, j]
        nz[slot_position(k2c, r, SLOT_C)] += w
        nz[slot_position(k2c, r, slot)] -= w
        return
    end
    NR, NZ = G.NR, G.NZ
    for j in 1:NZ, i in 1:(NR - 1)
        Γ = sol.ΓR[i, j]
        iszero(Γ) && continue
        Γ > 0 ? enter!(i + 1, j, Γ, SLOT_W) : enter!(i, j, -Γ, SLOT_E)
    end
    for j in 1:(NZ - 1), i in 1:NR
        Γ = sol.ΓZ[i, j]
        iszero(Γ) && continue
        Γ > 0 ? enter!(i, j + 1, Γ, SLOT_S) : enter!(i, j, -Γ, SLOT_N)
    end
    return A_adv
end
