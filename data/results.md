# Homeweave Evaluation Results

| Method | Zero-Violation Rate (95% CI) | Mean Violations (95% CI) | Placement Rate (95% CI) | Relations Kept (95% CI) | Runtime s (95% CI) | JSON Failures |
| --- | --- | --- | --- | --- | --- | --- |
| `engine_top1` | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.89 [0.83, 1.00] | 1.00 [1.00, 1.00] | 0.087 [0.051, 0.106] | 0 |
| `engine_judge_top5` | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.89 [0.83, 1.00] | 1.00 [1.00, 1.00] | 0.121 [0.082, 0.145] | 0 |
| `gemini_baseline` | 0.00 [0.00, 0.00] | 6.00 [2.00, 10.00] | 1.00 [1.00, 1.00] | 0.78 [0.33, 1.00] | 0.000 [0.000, 0.000] | 0 |

### Paired Bootstrap (`engine_top1 - gemini_baseline`)
- **Zero-violation delta:** `+1.00` (95% CI `[+1.00, +1.00]`)
- **Mean violations delta:** `-6.00` (95% CI `[-10.00, -2.00]`)
- **Relations kept delta:** `+0.22` (95% CI `[+0.00, +0.67]`)
