# Main test entry point. TestItemRunner auto-discovers every @testitem under the
# package root — discovery is by MARKER, so a new test directory can never go unrun.
#
# Patterns match the @testitem NAME or its FILENAME; plain patterns are substrings,
# a `re:` prefix switches to a Julia regex.
#     cc-julia-test-runner .                       # all but :broken/:regression
#     cc-julia-test-runner . physics               # name or filename contains "physics"
#     cc-julia-test-runner . "re:^Coils .* Vector" # regex over name or filename
#
# Tags: :regression → RAPID_RUN_REGRESSION=true, :broken → RAPID_RUN_BROKEN=true
#       (:broken tests are blocked on src/control/ defects).
# RETESTITEMS_NWORKERS=N (N>0) runs the suite through ReTestItems on N workers
# instead; see test/runtests_parallel.jl.

if parse(Int, get(ENV, "RETESTITEMS_NWORKERS", "0")) > 0
    include("runtests_parallel.jl")
else
    # Loaded HERE, not at the top: TestItemRunner and ReTestItems both export
    # `@testitem`, so only one runner may be imported per process. Hence the FUNCTION
    # form (run_tests) below — a macro would expand before this `using` ever runs.
    using TestItemRunner

    const PATTERNS = filter(a -> !isempty(a) && a != ".", ARGS)
    const WANT_REGRESSION = get(ENV, "RAPID_RUN_REGRESSION", "false") == "true"
    const WANT_BROKEN = get(ENV, "RAPID_RUN_BROKEN", "false") == "true"

    matches(ti) = any(PATTERNS) do arg
        p = startswith(arg, "re:") ? Regex(chopprefix(arg, "re:")) : arg
        return occursin(p, ti.name) || occursin(p, ti.filename)
    end

    if isempty(PATTERNS)
        WANT_REGRESSION || @info "Skipping :regression testitems. Set RAPID_RUN_REGRESSION=true to run them."
        WANT_BROKEN || @info "Skipping :broken testitems (blocked on src/control/ defects). Set RAPID_RUN_BROKEN=true to run them."
    end

    # NB: the PACKAGE ROOT, not test/.
    TestItemRunner.run_tests(
        joinpath(@__DIR__, "..");
        verbose = true,
        filter = ti -> begin
            # An explicit pattern is an explicit request: it wins over tag gating.
            isempty(PATTERNS) || return matches(ti)
            :broken in ti.tags && return WANT_BROKEN
            :regression in ti.tags && return WANT_REGRESSION
            return true
        end,
    )
end
