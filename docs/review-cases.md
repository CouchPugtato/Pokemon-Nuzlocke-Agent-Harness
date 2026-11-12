# Reviewing saved cases

Inspect a stopped run with `pokemon-harness cases --case 1`. Consequential
Nuzlocke cases still require a linked ledger correction for an evidence resolution.
A manual assertion or unverifiable continuation remains recorded as unverifiable.

Live captures wait `capture_settling.after_input_delay_ms` after a controller input
before normal screenshot settling. The default is 750 ms. This gives delayed game
responses time to become visible before their first review; it does not pause the
emulator or guarantee that an animation has finished. Initial and read-only captures
do not receive this extra delay. Stop and focus checks remain active during it.

Pending review screenshots take priority over redundant early captures and optional
history. When required review images cannot fit, the harness stops before another
model request or controller plan. It does not discard the comparison needed to
review an attempted input. Visual similarity checks include local changes across
the screen, including upper name fields as well as dialogue text.

A false progress stall can instead be resolved from a revised outcome review:

```powershell
pokemon-harness resolve-case 1 --outcome evidence --reason "Reviewed advancement to a new dialogue page" --review-file review.json
```

The review file contains the observation used for review and exactly one review:

```json
{
  "observation_id": 29,
  "outcome_reviews": [{
    "turn_id": 11,
    "expected_revision": 1,
    "steps": [{
      "step_index": 1,
      "before_observation_id": 27,
      "after_observation_id": 29,
      "assessment": "progress_observed",
      "explanation": "The supplied screenshots show a new dialogue page after the confirmation press"
    }]
  }]
}
```

IDs and revision must match that run. Inspect the linked screenshots first. A later
endpoint is allowed only for the final captured completed move within ten minutes,
without an intervening plan, session change or handoff. Missing captures remain
uncertain. The review must cover every attempted step and use valid recorded or
eligible delayed evidence.

The command requires progress from the latest attempted plan, a recovery attempt
belonging to the stopped episode, changed screenshot hashes, no remaining visual
cycle and no review backlog. It appends the revised review and evidence resolution
in one transaction. Failed validation rolls both back. Original reviews, inputs,
recovery attempts and ledger events remain recorded.

A successful evidence resolution marks the existing episode recovered without
creating an override or replenishing its recovery budget. The validity label stays
supported unless earlier assertions or invalidity already changed it. This label
records evidence provenance; it does not independently verify the interpretation
of screenshots or prove full-playthrough reliability.

The command does not control the emulator. Resume explicitly after reviewing the
resolution output.
