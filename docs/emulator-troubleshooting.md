# Verify controller input before model play

The run log distinguishes two independent steps: sending controller input and
requesting the next LM Studio decision. A completed input attempt does not prove
that the emulator registered it.

## Test Start without LM Studio

1. Open Emerald in VBA-M and close the input configuration dialog and all menus.
2. Disable Emulation > Pause when inactive. Close all emulator dialogs/menus.
3. The VBA-M backend reads the frontend Pause checkbox and synchronizes the actual
   state, so it can start from either paused or running. The shared mGBA backend
   still uses the configured `initial_state` assumption.
4. Run this from the repository terminal:

   ```powershell
   pokemon-harness input-test --config configs/emerald.json --button START
   ```

The command focuses VBA-M's game panel, resumes, holds the configured Start key (Enter) for
about 200 ms, releases it, waits about 750 ms, and pauses again. It does not call
LM Studio. A report and before/after screenshots are saved under
`runs/emerald-001/input-tests/<test-id>/`. It does not modify agent memory.

Verify that the New Game/Continue menu appears. A title-screen animation changing
is not sufficient evidence of a successful Start press. If input still fails,
check whether a physical Enter press works when the game is running. The binding
must match Player 1, and that player must be the selected/default controller.

The supplied controls are WASD movement, E/Q for A/B, R/T for L/R, Caps Lock for
Select, and Enter for Start. Start was changed from a modifier-only Shift binding
in the live emulator and config. Ctrl+P remains the manual pause shortcut.

VBA-M pause/resume uses its native frontend menu command and checks the resulting
Pause checkbox instead of blindly pressing a toggle. Reading this emulator UI state
does not access game RAM. If the menu state cannot be read or confirmed, execution
stops before further game input. Keep the menu bar enabled; `focus_game_area: true`
clicks the game panel to ensure the rendering child has focus and restore an
auto-hidden menu bar. This moves the mouse pointer into the game display.

## Input settings

```json
{
  "input_mode": "scancode",
  "min_hold_frames": 6,
  "fps": 60
}
```

These settings belong inside `emulator` in `configs/emerald.json`. Scan-code
injection is the default. `virtual_key` is available for comparison. Presses are
extended to at least 6 approximate frames (100 ms); waits are not extended.
One-frame presses can be missed while the frontend is processing frames/events.
VBA-M's [keyboard handler](https://github.com/visualboyadvance-m/visualboyadvance-m/blob/v2.1.5/src/wx/panel.cpp)
also checks the current key state when handling a key event. This makes very short
presses a plausible failure cause; it does not establish the cause on every setup.

The requested action remains in the model trace. Actual input duration and mode
appear in the execution report and recent-turn memory. All timing is wall-clock
approximation, including frontend overhead, rather than exact emulator frames.

Keep the emulator visible and focused during actions. If focus is lost, cleanup
may be unable to pause; check its actual state before restarting. Do not manually
toggle pause while the harness is running. VBA-M verifies pause through its frontend
menu; mGBA still tracks pause from the configured initial state.

## LM Studio timeouts

`model.timeout_seconds` is 600 in the example config. Slow generation, model loading,
or server problems can exceed the previous 180-second timeout. A longer timeout
allows more inference time; it does not fix input delivery.

The console reports which observation is waiting for inference. A model timeout
sends no controller input for that observation and does not retry previous inputs.
Check LM Studio's server/generation status if it persists. Once the input-only
test succeeds, try one model turn:

```powershell
pokemon-harness run --config configs/emerald.json --turns 1
```
