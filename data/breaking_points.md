# Breaking Points Ladder

| Dimension | Method | Level Where It Broke | What Failed First |
| --- | --- | --- | --- |
| `furniture_count` | `engine_top1` | 15 items (clean rate 0%) | `capacity_exhausted (<50% placed)` |
| `furniture_count` | `gemini_baseline` | 15 items (clean rate 0%) | `outside_room` |
| `scale_gap` | `engine_top1` | 0.4x area (clean rate 0%) | `capacity_exhausted (<50% placed)` |
| `scale_gap` | `gemini_baseline` | 0.8x area (clean rate 0%) | `outside_room` |
| `mirror` | `engine_top1` | held through max level | `none` |
| `mirror` | `gemini_baseline` | mirrored_x (clean rate 33%) | `outside_room` |
| `dimension_noise` | `engine_top1` | held through max level | `none` |
| `dimension_noise` | `gemini_baseline` | 5% jitter (clean rate 0%) | `outside_room` |
