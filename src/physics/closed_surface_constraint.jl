# The closed-surface flux constraint in a run.
# At the end of each step the basis is rebuilt from the field-line analysis; the next u∥ solve
# takes n u∥/B = K_e(ψ) on the closed region inside the cut, below the Ampère gate directly and
# above it after the coupled solve (`PostProjection`); after continuity u∥ is remapped to the
# density the step left. The kinetic energy the constraint removes and the remap changes is booked.

# the density below which a node is not admitted to the constraint (the per-particle operators' floor)
const CLOSED_SURFACE_N_FLOOR = 1.0

closed_surface_active(RP::RAPID) =
    RP.flags.closed_surface_flow isa SurfaceFluxConstraint && RP.closed_surface.basis !== nothing

"""
    update_closed_surface_basis!(RP) -> RP

Rebuild the constraint's basis from the last field-line analysis: the closed nodes inside the
wall whose density is above the per-particle floor are admissible, the wall cells bound the cut,
and the hats are merged on the weights B²/n. Without a basis (no single O-point, no closed
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
            G, F.ψ, A; inside = G.cell_state .>= 0, weights = F.Btot .^ 2 ./ max.(pla.ne, CLOSED_SURFACE_N_FLOOR),
        ) : nothing
    s.basis === nothing && empty!(s.solve.K)
    return RP
end

"""
    surface_constrained_ue!(RP, A, rhs, θu) -> RP

Solve the u∥ step `A u = rhs` (`A` sparse over all nodes) with the closed-surface constraint
inside the cut ([`constrained_momentum_solve`](@ref)), and write u∥ and the surface flux.
- The constrained rows leave ionization out of the decay: the newborn electrons' dilution is
  applied once, by the remap after continuity.
- The pressure enters the Galerkin rows as its generalized force on the stream-function fluxes
  ([`pressure_operator`](@ref)), in place of the nodal force the rows of `rhs` carry.
- The kinetic energy the constraint removes from the step's starting drift is booked as
  activation (no surface flux before), basis change, or projection.
"""
function surface_constrained_ue!(RP::RAPID{FT}, A::SparseMatrixCSC{FT}, rhs::AbstractVector, θu::Real) where {FT <: AbstractFloat}
    s, pla, F = RP.closed_surface, RP.plasma, RP.fields
    b = s.basis
    @unpack me, ee = RP.config.constants
    dt = RP.dt
    n, B, u0 = vec(pla.ne), vec(F.Btot), vec(pla.ue_para)
    sel = findall(<=(1), b.ψN)
    cu = b.nodes[sel]
    Pin = b.P[sel, :]
    A2, rhs2 = copy(A), Vector{FT}(rhs)
    for k in cu
        A2[k, k] -= θu * dt * pla.ν_en_iz_tot[k]
        rhs2[k] += (1 - θu) * dt * pla.ν_en_iz_tot[k] * u0[k]
    end
    Bp = pressure_operator(b, log.(max.(pla.ne, CLOSED_SURFACE_N_FLOOR)))
    extra = nothing
    if RP.flags.Include_ud_pressure_term
        a_p = vec(calculate_electron_acceleration_by_pressure(RP))
        F_p = -(Bp * (ee .* vec(pla.Te_eV)[b.nodes])) ./ me
        extra = dt .* (F_p .- Pin' * (b.V[sel] .* B[cu] .* a_p[cu]))
    end
    signature = hash((b.nodes, b.knots))
    K0 = galerkin_flux(b, n, u0, B)
    removed = me / 2 * sum(b.V[sel] .* n[cu] .* (u0[cu] .- (B[cu] ./ n[cu]) .* (Pin * K0)) .^ 2)
    sol, led = s.solve, s.ledger
    if isempty(sol.K)
        led.activation += removed
    elseif signature != sol.signature
        led.basis_change += removed
    else
        led.projection += removed
    end
    u, K = constrained_momentum_solve(A2, rhs2, b, n, B; extra)
    vec(pla.ue_para) .= u
    sol.K, sol.signature = K, signature
    # what the density, Te and u∥ transport of the step read: the stream-function fluxes of K and
    # the compression that pairs with the pressure force
    sol.ΓR, sol.ΓZ = stream_function_fluxes(b, K)
    fill!(sol.compression, zero(FT))
    sol.compression[b.nodes] .= Bp' * K
    return RP
end

"""
    remap_surface_flux!(RP) -> RP

After continuity, put u∥ on the constrained nodes back on the surface flux of the step,
u∥ = K_e B / n with the density the step left, so the current follows the surfaces and not
the density; book the kinetic-energy change and update Jϕ.
"""
function remap_surface_flux!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    (closed_surface_active(RP) && !isempty(RP.closed_surface.solve.K)) || return RP
    s, pla, F = RP.closed_surface, RP.plasma, RP.fields
    b = s.basis
    me = RP.config.constants.me
    sel = findall(<=(1), b.ψN)
    cu = b.nodes[sel]
    u_new = (F.Btot[cu] ./ pla.ne[cu]) .* (b.P[sel, :] * s.solve.K)
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
