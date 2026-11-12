# Switching local models

Use `--model` with a model key or loaded instance identifier from LM Studio:

```powershell
pokemon-harness resume --model qwen/qwen3.8-27b --context-length 32768 --thinking off --turns 15
```

The same option works with `run`. A successful start saves the selected model for
future resumes. Switching models preserves the run's fixed objective, rules,
ledger, archived turns and memory. It does not reset the emulator. A failed model
load leaves the saved model configuration unchanged.

Load the larger model in LM Studio first when using a custom GPU offload profile.
Keep the loaded context at least as large as the harness requests. The harness
reuses a suitable loaded instance, but may reload an undersized instance to obtain
the requested context. Load settings such as GPU placement and parallel slots are
managed in LM Studio, separately from the generation token budget.

Unload unused models to leave memory for the larger model and its context cache.
Start with one prediction slot and partial GPU offload if the model cannot fit
entirely in GPU memory. Keep the existing output reserve and safety margin while
checking resource use. CPU inference is a diagnostic fallback and can be much slower.

First compare archived screenshots without game input. Then try a short live run
and inspect actual movement, scene descriptions and memory updates before increasing
the turn limit. A correct proposal on one screenshot does not establish reliable
navigation or full-playthrough capability.
