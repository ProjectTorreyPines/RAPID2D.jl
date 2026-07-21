# Main test entry point — TestItemRunner auto-discovers every @testitem under the
# package root. Discovery is by MARKER, not by directory list: adding a new test
# directory can never silently go unrun (which is exactly how test/control/ rotted
# for months under the previous walkdir("unit") + include() scheme).
#
# ── Selecting what runs ──────────────────────────────────────────────────────
# Patterns match the @testitem NAME or its FILENAME. Plain patterns are
# substrings; a `re:` prefix switches to a Julia regex.
#     cc-julia-test-runner .                       # everything except :broken/:regression
#     cc-julia-test-runner . physics               # name or filename contains "physics"
#     cc-julia-test-runner . "Pure Diffusion"      # by testitem name
#     cc-julia-test-runner . "re:^Coils .* Vector" # regex over name or filename
#
# ── Tags ─────────────────────────────────────────────────────────────────────
# :regression  slow physics regression scenarios — opt in with RAPID_RUN_REGRESSION=true
#              (same env var as the previous runner, so existing muscle memory holds)
# :broken      tests whose PRODUCTION code is currently non-functional. Opt in with
#              RAPID_RUN_BROKEN=true. See test/control/control_basic_test.jl for the
#              src/control/ defects these are blocked on.
# An explicit ARGS pattern OVERRIDES tag gating — asking for something by name always
# runs it, otherwise `cc-julia-test-runner . control` would surprise you with silence.
#
# ── Parallel opt-in (ReTestItems) ────────────────────────────────────────────
# RETESTITEMS_NWORKERS=N (N>0) routes the same suite through ReTestItems across N
# worker processes. See test/runtests_parallel.jl — it runs a transient shadow copy
# rewriting @testsnippet into @testsetup (ReTestItems has no @testsnippet); nothing
# under test/ changes and the TestItemRunner/VS Code path below is untouched.
#     RETESTITEMS_NWORKERS=4 cc-julia-test-runner .

if parse(Int, get(ENV, "RETESTITEMS_NWORKERS", "0")) > 0
    include("runtests_parallel.jl")
else
    # Loaded HERE, not at the top: TestItemRunner and ReTestItems both export
    # `@testitem`, so only one runner may be imported per process. Consequence: the
    # suite is invoked through the FUNCTION form (run_tests) rather than
    # @run_package_tests — a macro in this branch would be expanded when the whole
    # `if` lowers, i.e. before the `using` above has ever run.
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

    # NB: the PACKAGE ROOT — run_tests reads the package name from the root
    # Project.toml to auto-inject `using RAPID2D` into every testitem.
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
