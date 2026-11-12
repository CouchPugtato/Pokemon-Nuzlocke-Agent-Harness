# Scene reassessment

Live emulator runs perform a read-only visual reassessment at startup and on resume.
Further checks are triggered by scene changes, ineffective inputs, a new recovery
episode or a periodic turn limit. `scene_reassessment` in the configuration controls
the thresholds and minimum interval. Replay runs default to disabled; tests can
enable it explicitly.

Reassessment uses two separate model requests. The first receives only the current
screenshot and its ID, without the old goal, scene memories, objective or dialogue
history. It describes visible objects, readable text, possible exits and game mode.
The second receives that observation, the screenshot, fixed objective and rules,
the old subgoal as a claim to test, and bounded scene-specific memory candidates.
Its schema locks the original visual fields, including game mode and candidate
exits. It can assess subgoal support and suggest a next check, but cannot rewrite
the visual observation to fit the old story. Missing completion proof alone does
not contradict a goal. Candidate exits and all interpretations remain fallible.

Each request has its own archived context and model trace and uses the existing
generation, session and stop checks. Nothing is committed to scene memory until
both validated requests succeed. Failure in either stage stops before game input.

Reports and traces are saved separately from controller decisions. Scene reports
survive restart and compaction. Contradicted scene facts are marked stale, while
older unconfirmed scene facts need refreshing. Both remain searchable and archived
but are excluded from automatic recall. Party facts, encounter eligibility, death
records and other consequential ledger history are not invalidated by this process.

A contradicted subgoal is withheld until explicitly revised or abandoned using its
existing ID. A later uncertain report cannot restore it. A merely uncertain subgoal
is withheld while that uncertainty remains. Source working revisions remain intact.
Older scene interpretations are withheld from automatic conversation recall using
the report's turn boundary, including old decisions about the same screenshot.

After reassessment, the first controller decision permits one safe probe or stop.
History search, queued moves and state/ledger writes are unavailable for that probe.
Movement strategies require directional buttons; interaction buttons cannot stand
in for walking to a distant target. This checks control consistency, not path safety.
Pending outcome reviews still block controller input. Existing recovery episode,
scene restrictions, signatures, case gates, stop handling and runtime budgets remain
in force. No recovery allowance is replenished by reassessment.

Repeated interaction transitions with identical before/after images and input are
not counted as fresh progress. This prevents repeatedly opening and closing the same
object text from continually resetting the stagnation counter. It is an exact-image
heuristic, not a guarantee of semantic progress detection.

This mechanism improves evidence handling and reassessment discipline. It cannot
guarantee that a local vision model recognizes a room, exit or safe movement. Shadow
tests use archived screenshots and private database copies; they do not establish
that the model can navigate successfully in the live emulator.
