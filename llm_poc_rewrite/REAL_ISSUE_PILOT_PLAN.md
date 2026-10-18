# Real-issue text-transfer pilot and historical mechanism note

> **Selected pragmatic protocol: draft only; no live model calls have been run.** The text-transfer pilot gives each of the three existing configured agent roles an authentic issue brief plus bounded documentation context independently, then evaluates each response with a blinded human rubric. No full source checkout, patch run, test execution, or agent-framework comparison is required. Two public issue records are provisional candidates with explicit leakage notes; eligibility review remains pending, so the study is intentionally blocked from running/export.

## Pragmatic text-only transfer protocol

The research question is whether the three existing configured agents—analyst, extractor and synthesist—can produce useful, evidence-grounded diagnoses and test plans for authentic software issue reports when given the same concise issue packet and official-document context. Reuse the exact role-to-model IDs from `llm_poc_rewrite/config.py::AGENT_MODELS` and the role-specific system prompts from `tasks.py`, but run each role independently on the full issue prompt/context. This is not an orchestrated analyst→extractor→synthesist pipeline: no role sees another agent's output. Each task/role has three repeated requests and two blinded human ratings per response on six 0–4 dimensions (issue understanding, evidence grounding, diagnostic reasoning, proposed change quality, test strategy, uncertainty/scope), for 18 responses and 36 ratings total.

The configured bundles are analyst → `gemini-3.5-flash-lite`, extractor → `gemma-4-26b-a4b-it`, and synthesist → `gemini-3.1-flash-lite`. They use the existing worker temperature 0.0 and 512-token limit. Their role-specific system prompt hashes are frozen into `study.json`; validation checks those, the model mapping, temperature, and token limit against `llm_poc_rewrite/config.py`. Repetitions are repeated requests under deterministic decoding configuration, not planned temperature-based sampling diversity. Any observed role differences conflate model identity and system prompt/specialization; do not claim a clean model-to-model comparison or an agent-framework effect.

No repository is cloned. The worker gets the reported versions, reproduction symptom, requested outcome and a short excerpt/paraphrase of relevant official documentation. The task packet explicitly prohibits claims of code execution or checkout inspection. Workers have no web-browsing or repository tools; they receive only the issue brief and attached context. This holds the information set fixed across repetitions and keeps scoring focused on reasoning from the same evidence. Keep issue comments, fixes, workarounds and evaluator notes out of the worker packet. The evaluator separately classifies prior public exposure and leakage; do not call public historical issue tasks blind discovery.

The current two provisional tasks are Pydantic #10694 (nested union serialization omission) and #10477 (cached-property export after assignment): distinct technical families. Their issue bodies provide concrete symptoms, but both are public and comments leak useful hints—#10694 includes maintainer workarounds; #10477 includes a causal hint. Therefore the default status is `pending`, not eligible. Independent named eligibility review is still required. If the study proceeds with those historical records, results are explicitly leakage-aware descriptive transfer/answer-quality observations, not evidence of unaided discovery. A prospective collection of new issue reports before solution discussion would reduce this contamination.

The v2 data package and fail-closed validator live at [`data/real_issue_transfer/v2/`](data/real_issue_transfer/v2/study.json), with the canonical v2 record schema at [`schemas/transfer_pilot_v2.schema.json`](schemas/transfer_pilot_v2.schema.json). The legacy v1 study metadata uses [`schemas/transfer_study.schema.json`](schemas/transfer_study.schema.json), while [`transfer_data.schema.json`](transfer_data.schema.json) describes only v1 ledger records. Run `python -m llm_poc_rewrite.transfer_data` from the repository root; a structurally valid `ready: false` report is expected until actual reviewers adjudicate eligibility. The package pins every configured agent and validator checks it against `config.AGENT_MODELS`; `--export-worker-packets` emits only the packet content and is blocked until the study is ready. The separate v1 package remains for compatibility with the earlier proposed patch/test schema; it is not the selected v2 protocol.

## Scope limits

- The study scores text responses only; it does not verify a diagnosis against a code checkout, run regression tests, or prove a fix.
- Three repetitions on two issue families do not establish broad transfer, model superiority, or production coding ability. The existing config uses temperature 0.0, so repetitions are repeat requests under that fixed decoding setting, not planned stochastic variation.
- Public exposure, memorization and comments/fix leakage are confounds; report them, stratify/flag rather than hide them.
- The current configured units differ in both model and role-specific system prompt; results compare those bundles, not isolated models. A clean model comparison would need to hold system prompts identical. Do not add a framework/agent-pipeline factor unless that becomes a distinct research question.

## Historical graph-adaptation mechanism note (separate, not the text-transfer cohort)

The historical mechanism candidate below is only a separate graph-adaptation smoke design. It is not the primary v2 text-response question or an implemented real-issue benchmark.

## Purpose, present gap, and candidate

This is a package **dependency-path change**, not proof of a prerequisite between the two patches or positive co-assignment synergy. The current smoke trace does not operationalize even the package-path change across its episodes: it reuses the same synthetic two-task placeholders, slot vectors and task inputs in both episodes, and sends neither a changed task dependency nor `G1` to the controller. `G1` is only an evaluator target. Therefore the trace checks that the adapter can update `Theta` from assignments; it cannot test whether adaptation benefits performance under changed dependencies. The candidate is too small and too weakly connected for the intended substantive orchestration claim. Do not pad it with an unrelated third issue.

A larger-cohort search has not yet produced an acceptable replacement. The Eclipse Update Manager retirement issues have more nodes, but at the suggested cutoff two of five were already closed and the observed rewire coincides with creation/linking of a new coordination bug; this is not clean evidence of a technical prerequisite change. See [311590](https://bugs.eclipse.org/bugs/show_bug.cgi?format=multiple&id=311590), [518351 history](https://bugs.eclipse.org/bugs/rest/bug/518351/history), [527378 history](https://bugs.eclipse.org/bugs/rest/bug/527378/history), [527379 history](https://bugs.eclipse.org/bugs/rest/bug/527379/history), and [530357 history](https://bugs.eclipse.org/bugs/rest/bug/530357/history). TypeScript's decorator migration has several real fixable issues but little issue-to-issue dependency evidence; a large Mozilla meta-issue is not yet verified as independent, patchable tasks. These are leads, not approved cohorts.

An earlier patch-oriented intake pass looked at Pydantic #12197, #10694, #10477 and #14004. Those records are not being used as a patch/test cohort. The selected v2 text tasks are #10694 and #10477, which have distinct issue families but public comment leakage; both remain pending eligibility review, and no model calls should start until reviewers and an exact model ID are recorded.

## Superseded patch-oriented proposal (historical notes only)

The sections below retain the earlier Gensim/Flair graph-mechanism and patch/test design for audit context. It is not the current v2 study and none of those candidate task counts or run estimates describes a live LLM real-issue experiment.

## Proposed prospective real-issue transfer pilot (superseded)

This earlier proposal evaluated issue tasks against frozen repository snapshots; it is not implemented and is separate from v2's text-response question.

### Cohort and intake protocol

1. **Historical patch proposal only:** target 4–8 actionable issues, preferably from one maintained repository/subsystem. This cohort rule does not apply to the text-only v2 answer-quality sample.
2. **Capture at intake, before resolution where possible:** issue URL and stable tracker ID; UTC capture time; raw title/body and their hashes; labels/state/author-visible acceptance criteria; linked issues and relation types; repository URL and exact base commit SHA; runtime/container or lockfile digest; relevant test commands; and a note about whether a public fix is already known. If an issue closes before an experimental run, keep its intake snapshot and use the pre-fix base SHA; do not substitute post-fix text or code.
3. **Freeze the task packet:** convert each issue into a bounded task brief that preserves its actual requirements but removes only irrelevant tracker noise. Reviewers document each edit and retain both source text and normalized prompt. Freeze prompts, tool permissions, model IDs, decoding settings, seed policy, timeout, retry policy, and per-task resource limits before comparing conditions. Do not expose reference patches, maintainer solutions, hidden tests, post-intake discussion, or later graph states to the worker.
4. **Historical patch proposal only:** the earlier design required patchability/testability and base/fix regression checks. Those requirements are superseded for v2 and must not be used to claim patch correctness from text answers.

### Relation schema and graph interpretation

Store issue-to-issue relations separately from package/runtime dependencies and from the controller's learned `Theta`. Normalize direction as `prerequisite -> dependent` only when evidence supports execution order. Use these relation labels:

- `tracker_blocks`: a tracker-declared blocks/blocked-by link, stored with its original direction and source; it is **not automatically a verified prerequisite**.
- `verified_prerequisite`: reviewers find contemporaneous evidence that the dependent task requires an output/interface/change from the prerequisite task.
- `shared_context`: tasks touch a shared module, interface, test fixture, or schema; symmetric context evidence, not a blocking edge.
- `related_only`: thematic, parent/child, duplicate, or administrative relation; retain for audit, exclude from prerequisite adjacency.
- `package_dependency`: software/runtime dependency path; never silently reinterpret as task blocking or same-agent synergy.

Each relation record stores `source_task_id`, `target_task_id`, `relation_type`, `tracker_relation`, `observed_at_utc` or a bounded observation interval, `evidence_url`, a short evidence quote/rationale, reviewer IDs/decision, and confidence (`high|medium|low`). Maintain separate tracker-declared and adjudicated graphs. If only one snapshot is available, call it a static contextual graph snapshot: do not claim a temporal graph shift. For prospective change tracking, snapshot links at intake and at least weekly, preserve raw API payloads and receipt timestamps, and record edge additions/removals as observed intervals unless the source gives a trustworthy action timestamp.

### Legacy patch-oriented v1 package (superseded)

The legacy v1 schemas and empty data directory remain solely for backwards-compatible validation of the superseded patch/test proposal. They are not inputs to the v2 protocol. Validate them explicitly with `python -m llm_poc_rewrite.transfer_data --version 1`; use the default command for v2. Do not use v1's base-fail/fix-pass readiness gate to judge v2 text answers.

For reference, the old v1 format separated `study.json`, task records in `issues.jsonl`, evaluator-only `evaluator.jsonl`, task relations in `relations.jsonl`, and patch/test outcomes in `runs.jsonl`. Its empty ledgers are archival scaffolding, not a runnable cohort. V2 instead uses `study.json`, `tasks.jsonl`, `traces.jsonl`, and `rubrics.jsonl`; keep fixes, hidden tests, comments and evaluator-only hints out of worker exports.

Both validators check package structure and recorded hashes, but validity is not proof of issue eligibility, reproduction, patch success, or study readiness. The v1 schema is retained for compatibility only.

### Historical patch-evaluation proposal (superseded)

The former v1 proposal compared patches across three allocation conditions and required sandbox/test evaluation. It is preserved as a separate possible future research direction only. It is not the v2 protocol and no such generation count should be attributed to the text-only study.

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

The ordinary rewrite remains the default mode: its `make_initial_state()` still uses zero `Theta`, and `orchestrate()` defaults to the existing `full` adaptation. Pilot-specific helpers add a validated initial `Theta` and `orchestrate_real_issue()` requests `theta-only`; these are exercised only by synthetic diagnostic `TaskSpec`s, not by real issue runs.

The offline command runs two episodes with the same placeholder tasks and synthetic analyst/extractor/synthesizer slots. It logs assignment, `Theta` before/after, co-assignment observation, and evaluator distance. The controller receives `G0` as its prior and never receives `G1`; the evaluator can compare to `G1` after assignment. There is no issue-derived change in prompts, prerequisite outputs, task acceptance criteria, source snapshots, or tests at episode 1. Thus no real graph-shift response or task-success benefit is measured.

The offline gate in `real_issue_design.py` only checks whether future issue/slot vectors have compatible named dimensions and a preregistered rationale; it does not allocate tasks. The two-node candidate is an experiment limitation, not a claim that two tasks are sufficient.

Keep the existing solver, `Landscape`, energy equations, `EpisodeAdaptationManager` and `eta_theta` unchanged. The two-node pilot initializes both static and adaptive trajectories with `kappa=0`, `Theta=G0`; its adaptive wrapper fixes `theta-only`. `C=0` and `W_risk=0`, so no real cost/risk claims are made. Solve episode 0, then update only from its selected assignment `X0`; episode 1 uses the resulting `Theta_1`, while the static arm retains `G0`. Neither controller receives `G1`, package release information, patch success, or hidden-test results; `G1` is available only to the post-assignment diagnostic/evaluator, never to the controller. The offline pilot-plan command also records assignments, pre/post Theta, the co-assignment observation, and graph distance, making clear that adaptation is from assignment history—not direct detection of the dependency release. Since `X_t` is chosen using current `Theta_t`, the update is endogenous and may reinforce the initial edge; tracking is a mechanism diagnostic, not a promised benefit.

Use the two issue identities as the two task nodes (`M=2`) and keep the existing three worker slots (`N=3`) in this mechanism-only smoke trace. The current baseline solver can assign both tasks to the same slot; no fixed-capacity allocator is implied. Before any coding-allocation claim, replace the synthetic analyze/extract/synthesize vectors with reviewed issue/slot vectors in one shared named feature schema. Current model/role embeddings do not represent validated coding competencies. The design gate must stay closed until vectors, rationale, test manifests, and task-to-vector mapping are frozen. If no defensible model capability vectors can be obtained, run a clearly labelled structural-mechanism test only or stop; do not claim meaningful coding-task allocation performance.

A future real-task orchestration study may compare `baseline`, `static_energy`, and `adaptive_energy` on matched issue inputs and verified task outcomes. This is **not implemented**: the normal runner still operates on the synthetic ten-task bank, while v2 is a separate text-response protocol with no real-issue patch run. The displayed 36-call figure belongs only to a historical arithmetic sketch and is not a live schedule.

## Implementation status and next gates

The active synthetic benchmark plans 540 worker + 108 semantic judge + one fresh-run preflight request (649 total). The historical mechanism's 36-call arithmetic is unrelated to both that benchmark and v2. The v2 draft contains two issue packets and empty trace/rubric ledgers; it sends no API calls.

Implemented in the repo:

- configuration constants for two candidate IDs, `G0`/`G1`, and `theta-only`;
- optional symmetric zero-diagonal `Theta_0` initialization and a pilot wrapper;
- a zero-API mechanism trace over synthetic placeholder tasks, with post-hoc `G1` distance;
- a feature-schema gate which currently reports `ready=false`.

Not implemented / still blocking an empirical pilot:

1. Obtain two independent, real reviewer decisions for each issue and record the true reviewer IDs; keep the task `pending` until then. Decide whether this public historical sample is acceptable as explicitly leakage-aware exploratory transfer or whether to replace it with prospectively captured issue reports.
2. `study.json` pins the current `AGENT_MODELS` mapping and its role names. If `config.AGENT_MODELS` changes, reconcile and freeze all model IDs, role prompts and decoding/API limits before any call; write a versioned run manifest for source citations, API requests and returned model versions.
3. Check the task packet hashes, source identity and schema; inspect the worker export to confirm it excludes issue numbers, comments, disclosure notes and evaluator metadata.
4. Run a small quota-capped set of independent analyst, extractor and synthesist responses for every included task/repetition, record model, prompt hash, request/token/cost/latency traces, blind outputs, and collect two independent rubric ratings per completed response.
5. Analyze scores per task and repetition, summarize rater disagreement and leakage status, and keep claims limited to answer quality on these tasks. No patch sandbox, repo clone, fix gate or agent-framework comparison is needed for this protocol.
6. Keep the graph-shift/slot-allocation mechanism as a separate future study with its own clear question; do not mix its synthetic traces into this LLM response sample.

Until the v2 independent-review gates pass, the text-transfer study remains a draft and no model run is authorized by the package. The configured agent models are already pinned from `AGENT_MODELS`; do not replace them with a new model selection. `--real-issue-plan-only` still describes only a synthetic mechanism smoke diagnostic, not results for this text-transfer study.

## Interpretation limits

The v2 text pilot is exploratory: two historical public issues cannot establish broad generalization, and comment-derived hints are leakage confounds. The historical Gensim/Flair mechanism sketch is a separate synthetic diagnostic and cannot demonstrate adaptation benefits on real issues. Report any v2 result only as human-rated response quality on the frozen issue packets—not as coding success, allocation performance, or general real-world ability. A temporal graph-adaptation claim requires its own prospective graph-change study.

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

The old proposal counted two repeated historical patch tasks across three conditions, three repetitions and two graph episodes (36 proposed generations). That is arithmetic from a superseded design, not an executable schedule; its test fixtures, feature vectors and patch loop are not implemented. The candidate also has public-fix contamination, and its contemporaneous open-task requirement is not met at `G1`.

Near-matches considered in that proposal were rejected: SpeechBrain's SciPy change was a PR/workaround rather than an issue; GraphRAG #132 and Graspologic #1059 were filed after Gensim #3525 had closed and had different resolutions; Flair #3438 concerned a separate Python 3.12 installation failure. No third node was substantiated under the same criteria.

This historical candidate can remain a mechanism fixture, but must not be presented as a sufficiently large or validated real-issue cohort. The live experiment stays blocked until a larger, independently actionable cohort has an evidenced technical edge change and reproducible tests.
