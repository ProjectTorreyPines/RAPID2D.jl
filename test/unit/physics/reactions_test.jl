@testitem "Reaction counts: both ionization channels make an electron and burn a molecule" begin
    using RAPID2D
    using RAPID2D: ReactionCounts, net_electron_count, net_H2_gas_count, net_ion_count,
        REACTION_STOICHIOMETRY

    N = ReactionCounts{Float64}(dims = (3, 3))
    N.iz .= 2.0
    N.diz .= 0.5

    # One electron and one destroyed H₂ per event, through either channel.
    @test net_electron_count(N) ≈ N.iz .+ N.diz
    @test net_H2_gas_count(N) ≈ -(N.iz .+ N.diz)
    @test net_H2_gas_count(N) ≈ -net_electron_count(N)

    # The freshness check iterates the stoichiometry, so the row must exist.
    @test haskey(REACTION_STOICHIOMETRY, :diz)
    @test REACTION_STOICHIOMETRY.diz.electron == 1
    @test REACTION_STOICHIOMETRY.diz.H2_gas == -1
end

@testitem "Reaction counts: DI's charge rides the H2+ column until H+ can be transported" begin
    using RAPID2D
    using RAPID2D: ReactionCounts, net_ion_count, net_electron_count, REACTION_STOICHIOMETRY, species_key
    using RAPID2D.Particles: H, H2

    # INTERIM, pinned deliberately. DI produces H⁺, but `set_ion_species!` refuses a
    # second species (its docstring names the blockers). Carrying its charge on the
    # H₂⁺ column keeps quasi-neutrality exact and gets the ion MASS wrong for the DI
    # fraction — the smallest reversible error available. When multi-species lands,
    # change the row to `H[1]` and this test with it.
    N = ReactionCounts{Float64}(dims = (3, 3))
    N.iz .= 2.0
    N.diz .= 0.5

    @test net_ion_count(N, species_key(H2[1])) ≈ N.iz .+ N.diz
    @test net_ion_count(N, species_key(H[1])) === nothing      # not a declared species yet
    # NOT a conservation check. Under this interim `net_ion_count(N, H2[1])` and
    # `net_electron_count(N)` are literally the same expression, `N.iz .+ N.diz`
    # — they are equal here because they are the same code, not because anything
    # was proven about charge balance. This pins that fact so it is not read as
    # a proof once it stops being an identity (when DI's ion moves to `H[1]`).
    @test net_ion_count(N, species_key(H2[1])) ≈ net_electron_count(N)
    @test REACTION_STOICHIOMETRY.diz.ions == (species_key(H2[1]) => 1,)
end
