---
id: F_20261123_1006_gui-self-evolve
title: GUI agent self-evolution rails
type: F
author: kwangyyinc
author_email: chia.kwang.yang@h-partners.com
modules: [openjiuwen/harness]
files:
  - openjiuwen/harness/rails/evolution/gui_self_evolve.py
  - openjiuwen/harness/rails/evolution/__init__.py
  - tests/unit_tests/harness/rails/evolution/test_gui_self_evolve.py
  - docs/en/2.Development Guide/API Docs/openjiuwen.harness/subagents/cua_agent.md
  - docs/zh/2.开发指南/API文档/openjiuwen.harness/subagents/桌面子智能体.md
  - .agents/agent-docs/specs/harness/S_04_rails-contract.md
designs: [D-20261123-0925-gui-self-evolve]
---

# GUI agent self-evolution rails

## Metadata

| Item | Value |
| --- | --- |
| Date | 2026-11-23 |
| Scope | Committed part of phase A: `GuiOutcomeRail`, `VerifiedSuccessDetector`, `GuiSelfEvolveConfig`, `build_gui_self_evolve_rails`; lazy exports in `rails/evolution/__init__.py`; usage docs (en/zh); S_04 revision |
| Specs | S_04 |
| Test baseline | 168 passed: `test_gui_self_evolve.py` (12), `test_ttse_rail.py`, `test_cua_agent.py`, `test_create_browser_agent.py`, `test_deep_agent_rail_event_routing.py` (2026-11-23) |
| Not done here | Phase B script and the bench hook itself (local, not committed) |

## What was built

`build_gui_self_evolve_rails(GuiSelfEvolveConfig)` returns `[TTSERail]`, or `[GuiOutcomeRail, TTSERail]` when a verifier is given. A caller passes them through a GUI factory's existing `rails=`. No factory, default or permission rule changed.

- `GuiOutcomeRail.after_invoke` awaits the caller's verifier on the invoke result. It queues the score (or `None`) per session in `VerifiedSuccessDetector`, and runs at `EvolutionRail.priority + 1` so the score is queued before TTSE prepares or starts evolution.
- `VerifiedSuccessDetector.detect` pops the queued score for the trajectory's `session_id` and hands it to the fallback detector as `ttse_score`. The success threshold and outcome mapping therefore stay TTSE's own. With no score, the fallback (`SignalBasedSuccessDetector`) decides as usual.
- `store_path` is required and is refused inside the `openjiuwen` package (`DEEPAGENT_CONFIG_PARAM_ERROR`). A caller's `TTSEConfig` is copied, never mutated.

## Deviations from the design

1. **Score delivery (A-1 did not hold).** The design assumed the score could be set on `ctx`. In reality, `TTSERail.run_evolution` calls the detector with `ctx=None`, and its snapshot is built from a prepared input that has no score field. Setting `ctx.ttse_score` would never be seen. Fix: the detector wrapper plus a per-session FIFO queue. It works because invokes within a session run in order, and so do their evolutions (`max_concurrent_evolution=1`). TTSE itself is unchanged.
2. **`trajectory_span_processor` is a required config field.** `EvolutionRail` requires it, and every existing caller passes the process-wide one (`get_trajectory_span_processor()`). Spans only exist when observability is acquired, the agent carries an `AgentObservabilityRail`, and the host wraps each run in a run root span (`open_agent_run_span` / `close_agent_run_span`). Without the root span, `AgentObservabilityRail` opens no invoke span and the LLM/tool spans are skipped. The first live run showed this: zero spans and an empty bank. The builder does not add that rail, because a host that already has one would record every span twice. The docs state the requirement.
3. **Consult tools.** With `inject_enabled=True`, `TTSERail` registers its read-only consult tools at init. The design's "tool list unchanged" (V-05) holds only at construction. The tests check permissions and the constructed tool list, not the post-init tool set.

## Verification

| Design ID | Test | Result |
| --- | --- | --- |
| V-01 | `test_cua_agent_gains_only_the_self_evolve_rails_and_keeps_its_permissions`, `test_browser_agent_receives_the_self_evolve_rails_through_rails` | Pass. Mobile is not covered: `create_mobile_gui_agent` connects to a device during construction (`uiautomator2` missing here) |
| V-02 | `test_verifier_score_reaches_detector_for_that_session_and_result_is_unchanged`, `test_verifier_score_uses_ttse_own_success_threshold_without_calling_the_judge` | Pass |
| V-03 | `test_failing_verifier_leaves_run_alone_and_falls_back_to_ttse_detection` | Pass |
| FIFO pairing | `test_scores_pair_with_their_own_runs_in_order_and_per_session` | Pass |
| Ordering | `test_outcome_rail_runs_before_the_evolution_rail_after_invoke` | Pass |
| V-05 | Permissions and constructed tools equal with and without the builder (cua) | Pass, with deviation 3 |
| V-06 | `test_screenshots_reach_the_induction_prompt_only_as_a_bounded_fragment` | **Partial.** Each tool result is truncated, but about 560 base64 characters of a data URL still reach the induce prompt. The test pins the bound (< 1000); it does not show stripping |
| V-08 | `test_builder_wires_learn_and_inject_flags_to_the_instance_bank` | Pass (wiring only; live learning is V-04) |
| V-09 | `test_builder_requires_an_explicit_store_path`, `test_builder_refuses_a_bank_inside_the_openjiuwen_package` | Pass |
| V-04 | Live `cua_bench.py run --self-evolve` on `calc_multiply` (qwen/qwen3.7-plus, 2026-11-24) | Pass: verify passed, 19 trajectory increments drained, the bank went from 0 to 2 tips, stored under `%LOCALAPPDATA%`, outside the repo |
| V-02 (live) | `calc_multiply` with the verifier score flipped before it reaches TTSE (2026-11-24) | Pass: the run failed and verify scored 0.0; TTSE received 1.0 and reported `success` with reason `explicit-score`, so the verifier score, not TTSE's own judgement, decides the outcome |
| V-07 | `cua_bench.py run --self-evolve --inject` on a bank learned from two `calc_multiply` runs (2 facts, 1 tip; 2026-11-24) | Pass: the experience catalog and the `ttse_consult` guidance are in the LLM request from the second model call on. In one of three runs the agent called `ttse_consult` and then used the learned tip (`type_text`) |

## Remaining

- Strip image data URLs before induction (V-06). This needs a change in `agent_evolving` (TTSE message projection), which the design kept out of scope.
- The catalog is missing from the first model call of a run, where the agent plans. This comes from TTSE's attachment timing.
- Whether the agent consults the catalog is up to the model (one of three runs). The bank's `inject_hits` stays 0 and TTSE's "trailed catalog" log line does not appear, although the catalog reaches the prompt; not investigated.
- Mobile factory coverage needs a device-free construction path.
