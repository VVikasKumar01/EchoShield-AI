# DeepFilterNet Defence – Tri-Model Comparison

*Generated: 2026-09-04 17:23:31*


## Stationary Noise Performance

| Model | ΔSNR (dB) | SI-SDR Gain (dB) |
| :---- | --------: | ---------------: |
| Model A – DFN3 Core Only (Baseline) | -6.23 | -12.09 |
| Model B – DFN3 + Noise Classifier + Adaptive DSP | -5.24 | -32.73 |
| Model C – FULL DefenseDfNet (Proposed) | -7.60 | -38.15 |

## Non-Stationary Noise Performance

| Model | STOI | SI-SDR Gain (dB) |
| :---- | ---: | ---------------: |
| Model A – DFN3 Core Only (Baseline) | 0.184 | -13.22 |
| Model B – DFN3 + Noise Classifier + Adaptive DSP | 0.054 | -35.64 |
| Model C – FULL DefenseDfNet (Proposed) | 0.013 | -35.47 |

## Impulsive (Gunfire / Artillery) Performance

| Model | Peak Atten. (dB) | Recovery Frames | Swallow Ratio |
| :---- | ---------------: | --------------: | ------------: |
| Model A – DFN3 Core Only (Baseline) | [X] 3.29 | 2.8 | 0.953 |
| Model B – DFN3 + Noise Classifier + Adaptive DSP | [X] 2.71 | 0.4 | 0.816 |
| Model C – FULL DefenseDfNet (Proposed) | [X] -0.60 | 0.2 | 1.070 |

## Computational Performance

| Model | Params | RTF | Frame Time (ms) | Peak Mem (MB) |
| :---- | -----: | --: | --------------: | ------------: |
| Model A – DFN3 Core Only (Baseline) | 2,364,520 | [OK] 0.924 | 9.24 | 2.6 |
| Model B – DFN3 + Noise Classifier + Adaptive DSP | 2,369,323 | [X] 1.565 | 15.65 | 2.2 |
| Model C – FULL DefenseDfNet (Proposed) | 2,369,323 | [X] 19.300 | 193.00 | 2.2 |