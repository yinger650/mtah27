# Rule-based team possession labeling

Code that regenerates the Phase 1 possession file for [More than a Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027): Barcelona vs Athletic Club, LaLiga, 24 August 2024.

The method is rule-based and trains no model. It reads only the organizer's machine-generated tactical Automatic Tracking Data and automatic Smart Tagging. Tactical and broadcast video are not inputs. The output is one label per tactical-video frame: `H` (Barcelona, live), `A` (Athletic Club, live), or `D` (dead ball). The file contains no `X` abstentions.

The rules and the public-window scores are in [`technical_report_k5.md`](technical_report_k5.md).

## Requirements

- Python 3.13.5 (the version used to generate the submitted file)
- `numpy==2.5.4`

## Installation

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The match assets are gated and are not in this repository. Enrol, accept the terms, and download the bundle as described in the official `DATA.md`. Unpack it, keeping the delivered filenames:

```
data/
└─ tactical/
   ├─ BAR-ATH_tactical_atd.xml
   ├─ BAR-ATH_tactical_atd.txt
   └─ BAR-ATH_tactical_smart_tagging.xml
```

The pipeline uses only those three files. You may unpack into `./data` in a clone of this repository, or keep the bundle elsewhere.

## Run

From the repository root:

```bash
python -m mtah27.run --data /path/to/data --out submission.csv
```

`/path/to/data` is the folder that contains `tactical/`. A parent directory that contains `data/tactical/`, or the `tactical/` folder itself, is also accepted. If `--data` is omitted, the command looks at `MTAH_DATA_DIR` and then at `./data`.

The command writes a CSV with header `frame,label` and 199,319 rows, frames `0` through `199318`. On a successful reproduction the SHA-256 of that file is:

```
d2f7649ad9ee44924e794e88d29bad4114fca06c3aa7b0b1d3dc1702247d2232
```

A run typically takes a few minutes: it streams the tactical ATD three times and then applies the rules.

## Validation

This repository does not bundle the official evaluator. After generating the file, check it with the organizer repository:

```bash
python evaluation/validate_submission.py submission.csv
```

## Rules, in brief

The pipeline runs in this order. ATD record numbers are converted from one-based records to zero-based tactical frames. Normalized positions are scaled by the pitch dimensions. Team membership comes from the metadata. Track slots are not stable player identities.

1. For each visible ball, the closest visible player within 6 m supplies team evidence.
2. Automatic phase tags override that evidence for selected codes: `BUILD UP`, `CREATING CHANCES`, and `LONG BALL` assign possession to the tagged team; `DEFENDING IN DEFENSIVE THIRD` and `DEFENDING IN ATTACKING THIRD` assign it to the opposite team.
3. A frame is a dead-ball seed if the tracked ball is more than 0.5 m outside the pitch, or if it belongs to a missing-ball run of at least 30 frames. Each `SET PIECES` start adds a dead seed from 50 frames before the start until a restart kick, or until 100 frames after the start if no kick is found. A kick is declared when, after the first visible ball in the first 100 frames after the tag start, the ball is more than 1.0 m from that reference on three consecutive visible frames. The dead window then ends at the tag start plus the kick offset, clipped to 10–250 frames.
4. A dead seed emits `D` and clears the current team. Otherwise the reducer keeps the current team until contrary evidence has accumulated eight confirmations. With no team evidence, the current team persists for up to 75 frames, after which the reducer emits `D`.
5. A centred majority vote over `H`, `A`, and `D` follows (`smooth_w=125`, a half-width). Ties favor `H`, then `A`, then `D`.
6. After smoothing, the method writes `D` wherever the ball is missing and low collective activity persists for at least 25 frames, and it reapplies the out-of-pitch mask.
7. Predicted `D` is released when live evidence holds for 12 consecutive frames, relabeled with the preceding `H` or `A` carrier.
8. Up to three passes replace `H` or `A` runs shorter than 30 frames.
9. A single pass fills a pure `D` interval of at most 30 frames when the immediately adjacent runs belong to the same team.

Constants are in `mtah27/config.py`.

## License

MIT License, Copyright (c) 2026 Min Wang. See [LICENSE](LICENSE). Portions are adapted from the official More than a Hack 2027 starter code (Copyright (c) 2026-2027 Metrica Sports, MIT). The official licence does not cover the match data; do not redistribute that data or any reconstructive derivative of it.
