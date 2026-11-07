# Runtime limits and evaluation

`timing_budgets` in the run configuration limits generation, each active session,
cumulative active run time, and the age of a decision screenshot before input.
Defaults are 600 seconds per generation, 3600 per session, 86400 per run, and
600 for screenshot age. `scene_age_seconds` can set stricter ages for individual
scenes. Timing policy is fixed on first use in a run.

Active time survives resume. An unclosed session after a crash is conservatively
charged elapsed wall time up to its session limit, which can include downtime.
Limits are checked before input and while waiting for generation. Native emulator
calls cannot be preempted. A timed-out generation may continue on the server;
its late response cannot commit a decision or send input.

To continue a stopped run after its cumulative allowance is exhausted, record an
explicit extension without discarding memory or resetting existing usage:

```powershell
pokemon-harness extend-runtime runs/example --seconds 3600 --reason "continue supervised evaluation"
```

Extensions are recorded with timestamps and reasons. Each adds at most seven days;
session, generation, freshness, review, and recovery limits remain in force.

For a legitimate repeated task, stop the harness, inspect its current subgoal with
`pokemon-harness state`, and grant a narrow allowance:

```powershell
pokemon-harness allow-repeat runs/example --subgoal walk-to-exit --strategy move --button UP --moves 8 --seconds 120 --reason "visible corridor" --until "doorway reached"
```

Allowances bind the exact working-state revision, strategy, and buttons. They permit
single moves in overworld, dialogue, or menus and expire on any state revision, time
limit, or reserved move count. A reservation is consumed even if execution is
interrupted. An allowance cannot be renewed for the same subgoal. Completing or
abandoning the subgoal ends it. The visible termination condition is interpreted
by the model and is not independently verified.

Only input-repetition and related similarity signals can be suppressed. Exact
visual cycles, no-progress limits, outcome reviews, existing recovery episodes,
and blocking cases remain in force. Allowances do not authorize repeated battle
choices, captures, or fleeing.

## Replay archived requests

```powershell
pokemon-harness evaluate runs/example --mode replay --limit 20 --output-dir evaluations/replay-01
```

Replay reconstructs the original prompts and verified screenshot bytes. It checks
the stored request fingerprint, decision schema, supplied visual evidence, and
review/recovery gates visible in the archived context. It performs no inference.

## Shadow inference

```powershell
pokemon-harness evaluate runs/example --mode shadow --config configs/emerald.json --limit 10 --repeat 3 --output-dir evaluations/shadow-01
```

Shadow mode asks LM Studio for new proposals using the original archived requests.
It compares input plans across repetitions and against the archived proposal.
It never constructs an emulator, applies proposals to source memory, or simulates
their consequences. Each request is independent; later cases use their original
context rather than the preceding shadow proposal. Original prompts are retained,
so this measures model behavior on that prompt version.

Use `--offset` for another page, `--generation-seconds` for each inference deadline,
and `--total-seconds` for the evaluation deadline. Generation expiry abandons the
response and ends evaluation. Reports written before interruption remain available.
The source run must be stopped. The output directory must be new and separate from
the source. Database evaluation occurs in a private copy, including any saved WAL.

Optional human criteria can be supplied with `--labels cases.json`:

```json
{
  "cases": [
    {
      "request_id": 12,
      "allowed_action_kinds": ["search_memory", "stop"],
      "forbidden_claim_fragments": ["I own Mudkip"],
      "note": "Example manually reviewed ambiguous ownership scene"
    }
  ]
}
```

Criteria use action kinds and literal fragments in memory claims/event details.
They are not semantic verification. Select labels from manually inspected scenes
and keep held-out cases separate. Reports include unselected labels, contract
errors, label failures, proposal variation, timing, returned token usage, and
source loop/recovery/case counts. Schema validity and agreement with an archived
action do not establish that an action is correct. Database write legality,
semantic progress, actual emulator outcomes, and full-playthrough reliability
still require supervised live evaluation.
