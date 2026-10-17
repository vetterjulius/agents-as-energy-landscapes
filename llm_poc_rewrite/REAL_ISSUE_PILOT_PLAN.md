# Real-issue transfer pilot: dependency change in the SciPy/Gensim/Flair path

## Purpose and decision

A pilot without a graph change would not test the intended transfer question. The best compact, source-supported candidate found so far is a **two-task historical cohort** around the SciPy 1.13 removal of `scipy.linalg.triu`: Gensim issue #3525 and the downstream Flair import failure #3441. Both task identities exist before the initial snapshot; the documented dependency of Flair core imports on Gensim is present at `G0` and removed from the Flair core path by the Flair 0.14.0 release at `G1`.

This is a genuine, fixed-node **dependency-path change**, not proof that task interactions or positive patch synergy changed. Use that exact label. The corpus is small (two real issue-derived patch tasks), but preferable to inventing a three-node graph or keeping the earlier unsupported Flask design. Do not add a third task just to fill the rewrite's three slots: research did not verify one that meets the same timeline and patchability criteria.

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

## Adaptation: the minimal necessary correction

The ordinary rewrite remains the default mode: its `make_initial_state()` still uses zero `Theta`, and `orchestrate()` defaults to the existing `full` adaptation. The pilot-specific helpers add an optional validated `Theta_0` and use `theta-only` without changing the default path.

The previous six-instance capacity allocator in `real_issue_design.py` has been removed because its issue set did not match this pilot and the current orchestration has no validated coding-capability schema to support that allocator. The remaining offline gate only checks that any future issue/slot vectors have the same named dimensions and a preregistered rationale; it does not allocate tasks. The change in scope to two selected issue nodes is an experiment limitation, not a claim that two tasks are sufficient.

Keep the existing solver, `Landscape`, energy equations, `EpisodeAdaptationManager` and `eta_theta` unchanged. The two-node pilot initializes both static and adaptive trajectories with `kappa=0`, `Theta=G0`; its adaptive wrapper fixes `theta-only`. `C=0` and `W_risk=0`, so no real cost/risk claims are made. Solve episode 0, then update only from its selected assignment `X0`; episode 1 uses the resulting `Theta_1`, while the static arm retains `G0`. Neither controller receives `G1`, package release information, patch success, or hidden-test results; `G1` is available only to the post-assignment diagnostic/evaluator, never to the controller. The offline pilot-plan command also records assignments, pre/post Theta, the co-assignment observation, and graph distance, making clear that adaptation is from assignment history—not direct detection of the dependency release. Since `X_t` is chosen using current `Theta_t`, the update is endogenous and may reinforce the initial edge; tracking is a mechanism diagnostic, not a promised benefit.

Use the two issue identities as the two task nodes (`M=2`) and keep the existing three worker slots (`N=3`) in this mechanism-only smoke trace. The current baseline solver can assign both tasks to the same slot; no fixed-capacity allocator is implied. Before any coding-allocation claim, replace the synthetic analyze/extract/synthesize vectors with reviewed issue/slot vectors in one shared named feature schema. Current model/role embeddings do not represent validated coding competencies. The design gate must stay closed until vectors, rationale, test manifests, and task-to-vector mapping are frozen. If no defensible model capability vectors can be obtained, run a clearly labelled structural-mechanism test only or stop; do not claim meaningful coding-task allocation performance.

Compare all three existing conditions—`baseline`, `static_energy`, and `adaptive_energy`—with the same task inputs, slot models, neutral task instructions, seeds and solver budget. Main contrast is adaptive versus static; baseline is descriptive. Each assignment triggers one patch-generation request using the selected slot's existing model and the existing API client. No coding agent, shell/tools, LLM judge, or new dependency. Hidden tests score patch outcomes.

## Call budget and implementation sequence

Use the current PoC's scale as reference rather than an arbitrary 12-call cap. For the first preregistered run, retain the rewrite's three conditions and three repetitions, with two issue tasks and two graph episodes: **3 × 3 × 2 × 2 = 36 planned patch-generation requests**, zero judge/preflight requests. This is a planned generation count; existing transport retries can increase billed HTTP attempts, so log attempts and successes separately and do not change retry behavior without a demonstrated need. For comparison, the ordinary synthetic run currently plans 540 worker + 108 semantic judge + 1 fresh-run preflight request (649 total). Do not report that synthetic count as the real-issue run's count.

1. Freeze task snapshots, regression tests, runtime/container manifests, source citations, graph edge definition/cutoffs, feature schema, prompts, models, seeds, repetitions, and 36-call design. Verify the historical fixes pass the tests; keep fixes/tests hidden from workers.
2. Add the optional `Theta_0`, pilot-only `theta-only` wrapper, and offline state/assignment diagnostics. Do not refactor the regular runner or alter ordinary retries/resume.
3. Test offline: graph dimensions/symmetry and `Theta_0=G0`; both arms start identically; static state stays at `G0`; adaptive update occurs after `X0` and affects only episode 1; no `G1` or test outcome reaches the controller; exact request planning is 36 before retries.
4. Run mock adapter tests, the zero-API design check, and existing relevant rewrite tests. Execute the live 36-request pilot only when ready to spend the existing Google API quota.

## Interpretation limits

This candidate provides a dated, fixed-node package-dependency-path change, but not a direct coding-synergy graph. It uses two issue-derived tasks, not three; one historical issue was classified as a question; task prompts recur after their public fixes; and issue #3525's PR did not add its own regression test. The adaptive state sees only assignments and cannot know that the graph changed unless assignment history is informative. Results are feasibility/transfer evidence only, conditional on passing the feature and reproduction gates—not proof of real-world dependency learning or improved software engineering. If those caveats do not match the intended scientific claim, capture a prospective issue cohort rather than weakening the graph definition.