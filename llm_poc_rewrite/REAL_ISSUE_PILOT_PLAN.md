# Real-issue pilot design note: SciPy/Gensim/Flair dependency path

> **Status: design only; not a runnable real-issue benchmark.** The repo has an offline mechanism smoke trace, not real task prompts, coding-capability vectors, repository snapshots, patch generation, or hidden regression tests. The two historical issues below are candidate task identities, not an implemented cohort. Do not interpret the `--real-issue-plan-only` output as a pilot result.

## Purpose, present gap, and candidate

The primary goal is **transfer to real software issues**: use authentic issue statements, repository code, patch generation and executable tests to see whether the orchestration approach remains useful outside the synthetic task bank. A perfect historical replay or a graph edge that changes between two cutoffs is not a prerequisite for this first transfer pilot. The Gensim #3525 / Flair #3441 pair below remains a small historical mechanism candidate, not the proposed main cohort.

This is a package **dependency-path change**, not proof of a prerequisite between the two patches or positive co-assignment synergy. The current smoke trace does not operationalize even the package-path change across its episodes: it reuses the same synthetic two-task placeholders, slot vectors and task inputs in both episodes, and sends neither a changed task dependency nor `G1` to the controller. `G1` is only an evaluator target. Therefore the trace checks that the adapter can update `Theta` from assignments; it cannot test whether adaptation benefits performance under changed dependencies. The candidate is too small and too weakly connected for the intended substantive orchestration claim. Do not pad it with an unrelated third issue.

A larger-cohort search has not yet produced an acceptable replacement. The Eclipse Update Manager retirement issues have more nodes, but at the suggested cutoff two of five were already closed and the observed rewire coincides with creation/linking of a new coordination bug; this is not clean evidence of a technical prerequisite change. See [311590](https://bugs.eclipse.org/bugs/show_bug.cgi?format=multiple&id=311590), [518351 history](https://bugs.eclipse.org/bugs/rest/bug/518351/history), [527378 history](https://bugs.eclipse.org/bugs/rest/bug/527378/history), [527379 history](https://bugs.eclipse.org/bugs/rest/bug/527379/history), and [530357 history](https://bugs.eclipse.org/bugs/rest/bug/530357/history). TypeScript's decorator migration has several real fixable issues but little issue-to-issue dependency evidence; a large Mozilla meta-issue is not yet verified as independent, patchable tasks. These are leads, not approved cohorts.

A recent intake pass examined Pydantic issues [#12197](https://github.com/pydantic/pydantic/issues/12197), [#10694](https://github.com/pydantic/pydantic/issues/10694), [#10477](https://github.com/pydantic/pydantic/issues/10477), and [#14004](https://github.com/pydantic/pydantic/issues/14004). They look like possible reproductions, but the first three lack confirmed fixes and #12197/#10477 have overlapping cached-property/assignment scope; #14004 already discloses a proposed fix commit. None has been verified here as fail-on-frozen-base/pass-on-trusted-fix. Accordingly Pydantic remains a **provisional intake target only**, not an eligible cohort; do not populate the study ledger or begin model runs from this web review alone.

The baseline solver implementation is reused from `controlled_benchmark` (the same `ConventionalGreedySolver`), but that alone does not make a real-issue baseline complete or comparable: current task and slot features remain synthetic, and the real patch-evaluation path does not exist.

## Proposed prospective real-issue transfer pilot

This is the recommended next study, distinct from the historical two-node mechanism trace. It tests transfer on real, reproducible software tasks; it does not require claiming that the task graph itself changed over time. Record the graph and task context that are available at intake, then evaluate real patches against frozen repository snapshots. If links evolve during collection, record the change as useful contextual evidence, not as a required inclusion criterion or causal result.

### Cohort and intake protocol

1. **Select a coherent but manageable cohort:** target 4–8 actionable issues, preferably from one actively maintained repository/subsystem so that repository setup and test harness are shared. Use a fixed intake window (for example, 4–8 weeks) and freeze the eligible issue list at its end; do not select issues based on whether the eventual fix is convenient or successful. A small pilot can proceed with 4 tasks, but fewer than 4 is a pipeline/mechanism check only. Avoid meta-issues, duplicates, vague feature requests, documentation-only changes, security-sensitive issues, and tasks whose acceptance criteria cannot be independently tested.
2. **Capture at intake, before resolution where possible:** issue URL and stable tracker ID; UTC capture time; raw title/body and their hashes; labels/state/author-visible acceptance criteria; linked issues and relation types; repository URL and exact base commit SHA; runtime/container or lockfile digest; relevant test commands; and a note about whether a public fix is already known. If an issue closes before an experimental run, keep its intake snapshot and use the pre-fix base SHA; do not substitute post-fix text or code.
3. **Freeze the task packet:** convert each issue into a bounded task brief that preserves its actual requirements but removes only irrelevant tracker noise. Reviewers document each edit and retain both source text and normalized prompt. Freeze prompts, tool permissions, model IDs, decoding settings, seed policy, timeout, retry policy, and per-task resource limits before comparing conditions. Do not expose reference patches, maintainer solutions, hidden tests, post-intake discussion, or later graph states to the worker.
4. **Audit eligibility and leakage:** two reviewers independently label each candidate as actionable, patchable, and testable from its intake snapshot; record disagreements and resolve them before model execution. Exclude any task whose solution is substantially disclosed in the captured prompt/history or whose regression test cannot be shown to fail on the frozen base and pass on a trusted fix. Keep the trusted fix and hidden regression tests evaluator-only.

### Relation schema and graph interpretation

Store issue-to-issue relations separately from package/runtime dependencies and from the controller's learned `Theta`. Normalize direction as `prerequisite -> dependent` only when evidence supports execution order. Use these relation labels:

- `tracker_blocks`: a tracker-declared blocks/blocked-by link, stored with its original direction and source; it is **not automatically a verified prerequisite**.
- `verified_prerequisite`: reviewers find contemporaneous evidence that the dependent task requires an output/interface/change from the prerequisite task.
- `shared_context`: tasks touch a shared module, interface, test fixture, or schema; symmetric context evidence, not a blocking edge.
- `related_only`: thematic, parent/child, duplicate, or administrative relation; retain for audit, exclude from prerequisite adjacency.
- `package_dependency`: software/runtime dependency path; never silently reinterpret as task blocking or same-agent synergy.

Each relation record stores `source_task_id`, `target_task_id`, `relation_type`, `tracker_relation`, `observed_at_utc` or a bounded observation interval, `evidence_url`, a short evidence quote/rationale, reviewer IDs/decision, and confidence (`high|medium|low`). Maintain separate tracker-declared and adjudicated graphs. If only one snapshot is available, call it a static contextual graph snapshot: do not claim a temporal graph shift. For prospective change tracking, snapshot links at intake and at least weekly, preserve raw API payloads and receipt timestamps, and record edge additions/removals as observed intervals unless the source gives a trustworthy action timestamp.

### Data package and run record

An initial, dependency-free package and strict validator now exist at [`data/real_issue_transfer/v1/`](data/real_issue_transfer/v1/study.json) and [`transfer_data.py`](transfer_data.py). Validate the package from the repository root with `python -m llm_poc_rewrite.transfer_data`; it exits nonzero on malformed records and separately reports `valid` from `ready`. The default package is intentionally empty/draft and should report blocking readiness items rather than fabricate a cohort. The opt-in `--export-worker-packets` output is limited to included task packet fields; evaluator-only records are stored in a separate ledger and are never exported. Keep the package immutable/versioned after preregistration, and use a new version for schema changes.

Keep an immutable, versioned study directory (for example `data/real_issue_transfer/v1/`) with these machine-readable files:

- `study.json`: question/estimand, cohort window, inclusion rules, conditions, model/prompt/tool versions, seed/repetition plan, API budget, evaluator policy, claim limits, and preregistration timestamp.
- `issues.jsonl`: one record per task with tracker/repository identifiers, `captured_at_utc`, source-text hash, worker-visible normalized brief/acceptance criteria and canonical `prompt_sha256`, base SHA, environment digest, test commands, and eligibility/adjudication. Evaluator references/results do **not** belong in this file.
- `evaluator.jsonl`: separate task-keyed trusted-fix and regression-test references plus base/fix test outcomes, never part of the worker export.
- `relations.jsonl`: tracker links and adjudicated task relations with the fields above, plus snapshot/event provenance.
- `runs.jsonl`: one record per condition × task × repetition, with assigned slot, model and prompt hash, seed, start/end times, request/retry counts, patch artifact hash, application status, structured test results, token/cost/runtime totals, and failure category.
- `artifacts/`: isolated patch diffs, logs, test outputs and environment/build manifests; keep hidden tests and reference fixes in a separately access-controlled evaluator area. Never place API secrets in the study package.

Use stable IDs such as `github:<owner>/<repo>#<number>`; all timestamps UTC ISO 8601. Save raw tracker snapshots alongside normalized JSONL so another reviewer can reconstruct the normalization. Include missing-data and API pagination/access logs. Before publication, check issue/repository licenses and avoid republishing personal data or restricted issue content. The validator checks duplicate IDs, referential integrity, hashes/timestamps, finite non-negative cost/runtime data, secret-like field names, and worker/evaluator separation; package validity is not a test-sandbox or eligibility proof.

### Evaluation protocol and claims

Run a paired comparison: every eligible issue is attempted under the same pre-registered conditions from the same base SHA and environment; each attempt uses a clean, isolated worktree/container, so patches cannot leak across conditions. Keep worker-facing facts, worker model family, tool access, timeout, test budget, and retry policy matched. Evaluate the patch with public tests plus private regression tests that are verified to fail on base and pass on the trusted fix. Score patch application, regression pass rate, acceptance-criteria coverage, regressions, runtime/token/API cost, and human rubric judgments separately; issue closure or MR merge alone is not a correctness label.

The first run should be framed as a **feasibility and transfer pilot**, not a powered superiority study. A practical starting scale is 4–8 issues × 3 conditions (`baseline`, `static_energy`, `adaptive_energy`) × 2 repetitions = **24–48 proposed patch generations**, excluding transport retries. Preflight the sandbox and test harness locally; do not spend API quota until each task passes the base/fix test gate. Keep the adaptive-vs-static comparison primary and baseline descriptive. Use issue-level paired outcomes and report the full per-task results; with this cohort size, uncertainty and task heterogeneity matter more than a single aggregate score.

If the chosen graph/assignment mechanism relies on slot competencies, use code-capable slots and freeze a named, reviewed feature schema plus evidence for each slot/task value. If defensible skill vectors are not available, do **not** claim skill-aware routing: either (a) use one fixed code-capable worker policy in a pipeline-only transfer check, or (b) stop the allocation-effect comparison until distinguishable, documented slot capabilities exist. Likewise, if tasks have prerequisites, execute/evaluate them in isolated DAG-consistent worktrees and pass only the prerequisite artifacts allowed by the frozen protocol; do not infer dependency benefit from co-assignment alone.

### Minimal prospective collection sequence

1. Choose repository, issue intake window, inclusion rules and a safety/license policy; write `study.json` before selecting outcomes.
2. Capture issue text, status, tracker links, repository base SHAs and setup/test metadata weekly; freeze 4–8 eligible tasks and adjudicate the relations.
3. Build each task's base/fix test fixture; prove the hidden regression test fails on base and passes on a trusted fix in the sandbox.
4. Freeze conditions and prompts; run mock/offline validation and a no-API patch-application test, then a small quota-capped preflight.
5. Run paired attempts, preserve raw artifacts and per-task outcomes, and publish limitations. If no suitable task cohort arrives within the intake window, extend the window rather than weakening task/test criteria.

## Historical two-issue candidate (not the transfer cohort)

Keep the sources and graph definition below as a **mechanism candidate only**, not as the prospective transfer cohort. Before claiming a task-level graph change, require evidence that each selected node is a distinct actionable task and that the edge semantics predict a real change to task order, shared context, or joint ownership.

## Source-grounded nodes and graph snapshots

| Node | Stable issue-derived task | Primary evidence / fix |
|---|---|---|
| A | **Gensim #3525**: make Gensim's `triu` import compatible with SciPy 1.13 | [Issue #3525](https://github.com/piskvorky/gensim/issues/3525), opened 2024-04-03, reports `ImportError` after SciPy removed `scipy.linalg.triu`; [PR #3524](https://github.com/piskvorky/gensim/pull/3524) switches to NumPy's `triu` (with SciPy fallback), merged 2024-04-05; [Gensim 4.3.3](https://github.com/piskvorky/gensim/releases/tag/4.3.3) was tagged 2024-07-19. |
| B | **Flair #3441**: keep core `flair.data.Sentence` import working when the transitive Gensim/SciPy path fails | [Issue #3441](https://github.com/flairNLP/flair/issues/3441), opened 2024-04-04, gives the failing import traceback through Flair → Gensim → removed SciPy symbol; [Flair PR #3493](https://github.com/flairNLP/flair/pull/3493) makes Gensim optional in the core package; [Flair 0.14.0](https://github.com/flairNLP/flair/releases/tag/v0.14.0), released 2024-07-25, documents Gensim as optional for word-embedding features. The issue was later closed as fixed since 0.14.0. |

Define the graph edge narrowly: **“Flair's core import path requires the Gensim package whose import is broken by the SciPy 1.13 API removal.”** `G0` is the documented core dependency path; `G1` is the post-release core-import path. The Gensim optional extra remains a dependency for some Flair functionality, so `G1` means “edge removed for core import,” not “Flair is independent of Gensim in every feature.” Use fixed node order `[Gensim #3525, Flair #3441]`, symmetric zero-diagonal `Theta` because the existing interaction-energy term is undirected:

```text
G0 = [[0, 1],
      [1, 0]]

G1 = [[0, 0],
      [0, 0]]
```

Snapshot cutoffs: `G0` is 2024-04-04, after both issues were filed and before PR #3524 merged on April 5. `G1` is 2024-07-25, after Gensim 4.3.3 was tagged and Flair 0.14.0 released. The evidence supports an actual change in the package dependency path; it does **not** show that the two task patches share source files, that one issue formally blocks the other, or that an edge has positive co-assignment synergy. The graph is an operational, source-audited dependency proxy.

### Why not use a third node?

The searched near-matches fail a useful criterion: SpeechBrain's contemporaneous SciPy pin is a PR rather than an issue and is a workaround, while GraphRAG #132 / Graspologic #1059 were filed after Gensim #3525 had already been closed and have different resolutions. Flair #3438 concerns a separate Python 3.12 install failure; counting it as another SciPy/Gensim task would conflate issues. Keep the graph at two stable issue identities instead of padding the cohort.

## Task/test protocol and historical nature

Freeze each task as an issue-derived patch benchmark with its own repository, exact pre-fix base SHA, bounded prompt, one-shot diff generation, and private regression check. The tests should reproduce the reported failures: (A) import/use the relevant Gensim `triu` path with SciPy >=1.13; (B) install/use Flair core without the optional Gensim extra and import `Sentence`. Verify the tests fail on their pinned task bases and pass on the historical fixes before any model call. PR #3524's displayed change did not add a regression test, so test A must be authored and audited from the issue's reproduction rather than falsely attributed to that PR. Pin the exact PR parent commits and compatible Python/SciPy/Gensim/Flair environments in a manifest; the dates above alone are not sufficient to reproduce a test environment.

The issue-derived tasks are **replayed historical work items** in both graph episodes; by `G1` the public fixes already exist. This is a controlled transfer experiment, not a claim that those GitHub issues were still open in July 2024. Keep patches, PR diffs, package release notes, G1 evidence, and hidden tests out of model-visible prompts. Public-fix contamination remains a serious limitation. If the intended claim requires simultaneously open, unresolved issues at both cutoffs, this candidate does not qualify and the live pilot must wait for a prospectively captured cohort.

## Adaptation: implemented smoke path vs. required experiment

The ordinary rewrite remains the default mode: its `make_initial_state()` still uses zero `Theta`, and `orchestrate()` defaults to the existing `full` adaptation. Pilot-specific helpers add a validated initial `Theta` and `orchestrate_real_issue()` requests `theta-only`; these are implemented, but they are exercised only by synthetic diagnostic `TaskSpec`s, not real issues.

The offline command runs two episodes with the same placeholder tasks and synthetic analyst/extractor/synthesizer slots. It logs assignment, `Theta` before/after, co-assignment observation, and evaluator distance. The controller receives `G0` as its prior and never receives `G1`; the evaluator can compare to `G1` after assignment. There is no issue-derived change in prompts, prerequisite outputs, task acceptance criteria, source snapshots, or tests at episode 1. Thus no real graph-shift response or task-success benefit is measured.

The previous six-instance capacity allocator in `real_issue_design.py` has been removed because its issue set did not match this candidate. The remaining offline gate checks only whether future issue/slot vectors have compatible named dimensions and a preregistered rationale; it does not allocate tasks. The two-node candidate is an experiment limitation, not a claim that two tasks are sufficient.

Keep the existing solver, `Landscape`, energy equations, `EpisodeAdaptationManager` and `eta_theta` unchanged. The two-node pilot initializes both static and adaptive trajectories with `kappa=0`, `Theta=G0`; its adaptive wrapper fixes `theta-only`. `C=0` and `W_risk=0`, so no real cost/risk claims are made. Solve episode 0, then update only from its selected assignment `X0`; episode 1 uses the resulting `Theta_1`, while the static arm retains `G0`. Neither controller receives `G1`, package release information, patch success, or hidden-test results; `G1` is available only to the post-assignment diagnostic/evaluator, never to the controller. The offline pilot-plan command also records assignments, pre/post Theta, the co-assignment observation, and graph distance, making clear that adaptation is from assignment history—not direct detection of the dependency release. Since `X_t` is chosen using current `Theta_t`, the update is endogenous and may reinforce the initial edge; tracking is a mechanism diagnostic, not a promised benefit.

Use the two issue identities as the two task nodes (`M=2`) and keep the existing three worker slots (`N=3`) in this mechanism-only smoke trace. The current baseline solver can assign both tasks to the same slot; no fixed-capacity allocator is implied. Before any coding-allocation claim, replace the synthetic analyze/extract/synthesize vectors with reviewed issue/slot vectors in one shared named feature schema. Current model/role embeddings do not represent validated coding competencies. The design gate must stay closed until vectors, rationale, test manifests, and task-to-vector mapping are frozen. If no defensible model capability vectors can be obtained, run a clearly labelled structural-mechanism test only or stop; do not claim meaningful coding-task allocation performance.

A future real-task run may compare `baseline`, `static_energy`, and `adaptive_energy` with matched task inputs, slot models, seeds and solver budgets. The main contrast would be adaptive vs. static; baseline descriptive. This is **not implemented**: the normal runner still builds and evaluates the synthetic ten-task bank, uses its deterministic checks/LLM judge, and has no real-issue patch-application or hidden-test adapter. The 36-call count is only arithmetic for a proposed design, not a live runner schedule.

## Implementation status and next gates

The ordinary synthetic run plans 540 worker + 108 semantic judge + one fresh-run preflight request (649 total). The offline real-issue design calculation is **3 conditions × 3 repetitions × 2 candidate issues × 2 episodes = 36 proposed patch generations**, with no judge/preflight. This is not a runnable real-patch call budget yet: no real-issue live path exists, and future issue count/design may change after cohort selection.

Implemented in the repo:

- configuration constants for two candidate IDs, `G0`/`G1`, and `theta-only`;
- optional symmetric zero-diagonal `Theta_0` initialization and a pilot wrapper;
- a zero-API mechanism trace over synthetic placeholder tasks, with post-hoc `G1` distance;
- a feature-schema gate which currently reports `ready=false`.

Not implemented / still blocking an empirical pilot:

1. Select and freeze a prospective cohort of 4–8 real, independently actionable issues, ideally from one repository/subsystem; a graph rewire is **not** a gate for the transfer claim. The two-issue package-path candidate and rejected Eclipse cohort remain insufficient as primary cohorts.
2. Select a repository and freeze an intake window, then capture issue/repository snapshots, base SHAs, environment and relation provenance in the versioned data package. `python -m llm_poc_rewrite.transfer_data` validates its format, not cohort eligibility; keep tracker links, verified prerequisites, shared context and package dependencies distinct. Current Pydantic candidates are provisional and fail the reproduce-on-base/pass-on-fix gate.
3. Freeze task prompts and evaluator-only tests; demonstrate each regression test fails on its pinned base and passes on a trusted fix. Record public-fix contamination and exclude materially disclosed tasks.
4. Implement a separate code-capable patch/test runner that applies each attempt in an isolated worktree, runs tests, and records patch, outcome, retries and cost. Do not route synthetic task specs through it or alter the ordinary synthetic runner implicitly.
5. If comparing skill-aware allocation, provide a defensible named feature schema and evidence for slot capability; otherwise narrow the study to pipeline/feasibility transfer and do not claim routing gains.
6. Offline-test collection completeness, graph normalization, sandbox boundaries, paired conditions, no evaluator leakage, patch/test plumbing, resume/logging and request arithmetic before any quota-bearing live run.

Until these gates pass, `--real-issue-plan-only` should be described as a synthetic mechanism smoke diagnostic, not a real-issue pilot or evidence of benefits under graph shift.

## Interpretation limits

Even a successful prospective pilot would be exploratory: 4–8 issues cannot establish broad generalization, and tracker links do not prove execution necessity. The historical Gensim/Flair candidate remains a tiny replay with public-fix contamination; its edge is package-path evidence, not formal issue blocking or shared-patch synergy. The current smoke trace cannot demonstrate adaptation to changed dependencies because its task inputs stay fixed. Report transfer as performance on this frozen real-issue sample under the tested harness, not as general real-world coding allocation. A temporal graph-adaptation claim would require a separate prospective graph-change study; do not imply one from a static relation snapshot.

---

## Earlier proposal (superseded; retained for audit trail)

This section records the original two-node proposal and its planned implementation. It is **not the current plan**; the status and gates above supersede its prospective phrasing.

The earlier candidate treated Gensim #3525 and Flair #3441 as a historical two-task cohort. The proposed edge represented the documented Flair core-import dependency on Gensim, which was affected by SciPy 1.13 at `G0` and removed for Flair core imports by 0.14.0 at `G1`. The interpretation was explicitly a package-dependency-path proxy, not proof of issue blocking or positive patch synergy. The graph was:

```text
G0 = [[0, 1],
      [1, 0]]
G1 = [[0, 0],
      [0, 0]]
```

The earlier protocol proposed two repeated historical patch tasks with private regression checks, three conditions, three repetitions and two graph episodes, yielding 36 planned generations. It also proposed `Theta=G0`, assignment-history `theta-only` adaptation, and post-hoc distance-to-`G1` diagnostics. Those tasks, test fixtures, feature vectors, live patch loop, and budget are not implemented. The candidate has public-fix contamination; Gensim PR #3524 did not add its own regression test, so its reproduction would need to be independently authored and audited. The contemporaneous open-task requirement is not met at `G1`.

Near-matches considered in that proposal were rejected: SpeechBrain's SciPy change was a PR/workaround rather than an issue; GraphRAG #132 and Graspologic #1059 were filed after Gensim #3525 had closed and had different resolutions; Flair #3438 concerned a separate Python 3.12 installation failure. No third node was substantiated under the same criteria.

This historical candidate can remain a mechanism fixture, but must not be presented as a sufficiently large or validated real-issue cohort. The live experiment stays blocked until a larger, independently actionable cohort has an evidenced technical edge change and reproducible tests.
