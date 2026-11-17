# Rule-based team possession labeling

More than a Hack 2027 · Phase 1 technical report

Barcelona vs Athletic Club, LaLiga, 24 August 2024

## Task and inputs

The system assigns one label to each of the 199,319 tactical-video frames at 25 frames per second. `H` denotes Barcelona in live possession, `A` denotes Athletic Club in live possession, and `D` denotes dead ball. The submitted file contains no `X` abstentions.

The method is rule-based and trains no model. It uses only the organizer's machine-generated tactical Automatic Tracking Data and automatic Smart Tagging. Tactical and broadcast video are not inputs to the label generator.

## Rules

The pipeline runs in the order below. ATD record numbers are converted from one-based records to zero-based tactical frames. Normalized positions are scaled by the pitch dimensions. Team membership comes from the metadata. Track slots are not stable player identities.

### Evidence and dead-ball seeds

1. For each visible ball, the closest visible player within 6 m supplies team evidence.

2. Automatic phase tags override that evidence when a selected phase supplies a known team. `BUILD UP`, `CREATING CHANCES`, and `LONG BALL` assign possession to the tagged team. `DEFENDING IN DEFENSIVE THIRD` and `DEFENDING IN ATTACKING THIRD` assign possession to the opposite team. Overlapping accepted priors are written in event iteration order. Other phase codes are not used as priors.

3. A frame is a dead-ball seed if the tracked ball is more than 0.5 m outside the pitch, or if it belongs to a missing-ball run of at least 30 frames. The missing-run rule marks the whole qualifying run, not only its tail. Each `SET PIECES` start adds a dead seed from 50 frames before the start until a restart kick, or until 100 frames after the start if no kick is found. The upper bound is exclusive. The rule does not mark the entire tag interval.

Restart-kick detection works as follows. In the first 100 frames after the `SET PIECES` start, the first visible ball position is the reference. From the next frame through 250 frames after the start, a kick is declared when the ball is more than 1.0 m from the reference on three consecutive visible frames. The kick time is the first of those three frames. The dead window then ends at the tag start plus the kick offset, clipped to the range 10 to 250 frames.

### Sticky assignment, smoothing, and activity overlay

4. A dead seed emits `D` and clears the current team and any pending switch. Otherwise the reducer keeps the current team until contrary team evidence has accumulated eight pending confirmations. A same-team observation resets the pending switch. A no-evidence observation does not. With no team evidence, the current team persists for up to 75 frames, after which the reducer emits `D` and clears the team.

5. A centered majority vote over `H`, `A`, and `D` follows. The parameter `smooth_w=125` is a half-width, so the interior window has 251 frames. Windows clip at match boundaries. Ties favor `H`, then `A`, then `D`, in array order. The vote uses the full match, so it is not causal.

6. After smoothing, the method writes `D` wherever the ball is missing and low collective activity persists for at least 25 frames, and it reapplies the out-of-pitch mask. Low activity requires a finite median player speed below 1.25 m/s and a slow-player fraction above 0.55. The slow fraction counts ATD speeds below 1.2 m/s. Coverage requires at least eight valid speeds and at least five tracked positions for each team. Player activity uses the ATD speed field, and only speeds in the range 0 to 12 m/s.

### Temporal correction

These stages run in order on the labels above.

7. Predicted `D` is released when live evidence holds for 12 consecutive frames. The streak, including its start, is relabeled with the preceding `H` or `A` carrier, not with the nearest player's team at release. The carrier defaults to `H` before any team label appears. Live evidence is either a ball inside the pitch (0.5 m tolerance) with a known nearest team within 2 m and an accepted alignment check, or a ball inside the pitch with the same coverage gate as above, a median speed of at least 1.8 m/s, and a moving-player fraction of at least 0.45. The moving-player fraction is the share of valid ATD speeds that are at least 1.5 m/s. The alignment check uses adjacent-frame position differences for the ball and the nearest player slot. It accepts a missing velocity estimate, or either speed below 0.8 m/s. Otherwise the velocity angle must be at most 45 degrees. The supplied ball speed field is not used by this check.

8. Up to three passes replace `H` or `A` runs shorter than 30 frames. If both neighbors share the same valid label, that label replaces the short run. Otherwise a neighboring `D` takes precedence. Remaining cases use the available team neighbor, or `D` if neither exists.

9. A single pass fills a pure `D` interval of at most 30 frames when the immediately adjacent runs belong to the same team. Opposite-team runs are not bridged. A configured dead-run protection of 75 frames does not trigger, because the bridge limit is 30.

## Public-window scores

Scores come from the official function `evaluation/possession_metrics.evaluate` on the public window, frames 11,185 to 18,684 inclusive (7,500 graded frames, 17 reference possessions). The previous submission uses the same rules, except that every `SET PIECES` dead window ends 100 frames after the tag start.

| Metric | This submission | Previous submission |
|---|---:|---:|
| Spell F1 (mean over IoU 0.3, 0.4, 0.5, 0.6, 0.7) | 0.5161 | 0.5032 |
| Frame accuracy | 0.7735 | 0.7707 |
| Macro F1 | 0.7126 | 0.7119 |
| Cohen's kappa | 0.6330 | 0.6297 |
| Alive or dead F1 | 0.8990 | 0.8968 |
| Spell F1 at IoU 0.5 | 0.5806 | 0.5161 |
| Predicted possessions | 14 | 14 |

Per-class scores for this submission:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| H | 3454 | 0.8292 | 0.9189 | 0.8717 |
| A | 2577 | 0.7611 | 0.7776 | 0.7693 |
| D | 1469 | 0.5996 | 0.4241 | 0.4968 |

Per-class scores for the previous submission:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| H | 3454 | 0.8359 | 0.9100 | 0.8714 |
| A | 2577 | 0.7549 | 0.7745 | 0.7646 |
| D | 1469 | 0.5849 | 0.4364 | 0.4998 |

Spell F1 by IoU for this submission is 0.6452 at 0.3 and 0.4, 0.5806 at 0.5, 0.3871 at 0.6, and 0.3226 at 0.7.

## Predicted possession share

On the two scored halves (frames 11,185 to 90,651 and 110,808 to 190,197), dead-ball frames are excluded, and the live-frame split is Barcelona 63.58% and Athletic Club 36.42%. The [LaLiga official match page](https://www.laliga.com/en-GB/match/temporada-2024-2025-laliga-ea-sports-fc-barcelona-athletic-club-2) reports 65.4% and 34.6%. This comparison is a post-hoc reasonableness check, not a scoring criterion.
