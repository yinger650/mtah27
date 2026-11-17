"""Feature channels from tactical ATD and Smart Tagging. No ground truth."""

from __future__ import annotations

import numpy as np

from mtah27.config import (
    ALIGN_MAX_ANGLE_DEG,
    ALIGN_MIN_SPEED_MPS,
    AWAY,
    BALL_CLUSTER_M,
    FIELD_SEP,
    FPS,
    GROUP_SEP,
    HOME,
    MIN_POS_SPREAD,
    MIN_TEAM_SPREAD,
    MISSING,
    MOVE_MPS,
    NEAR_BOUND_M,
    NOBODY,
    RUN_MPS,
    SLOW_MPS,
    SPEED_HI,
    SPEED_LO,
    STRONG_CODES,
    STRONG_DEF_CODES,
)
from mtah27.data import parse_xy_speed, team_side


def extract_atd_channels(raw_path, team_of_track, n_frames, pitch):
    """Nearest team (-1/0/1), nearest distance (m), ball xy (m, nan if missing)."""
    length, width = pitch
    nearest = np.full(n_frames, NOBODY, dtype=np.int8)
    nearest_d = np.full(n_frames, np.nan, dtype=np.float64)
    ball_x = np.full(n_frames, np.nan, dtype=np.float64)
    ball_y = np.full(n_frames, np.nan, dtype=np.float64)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            ball = parts[2]
            if ball[0] == MISSING:
                continue
            comma = ball.index(",")
            bx = float(ball[:comma]) * length
            rest = ball[comma + 1:]
            comma2 = rest.index(",")
            by = float(rest[:comma2]) * width
            ball_x[frame] = bx
            ball_y[frame] = by

            best, best_team = None, NOBODY
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if not group or group[0] == MISSING:
                    continue
                comma = group.index(",")
                dx = float(group[:comma]) * length - bx
                dy = float(group[comma + 1:group.index(",", comma + 1)]) * width - by
                d2 = dx * dx + dy * dy
                if best is None or d2 < best:
                    best, best_team = d2, int(team_of_track[slot])
            nearest[frame] = best_team
            if best is not None:
                nearest_d[frame] = best ** 0.5
    return nearest, nearest_d, ball_x, ball_y


def smart_tag_channels(events, n_frames):
    """Strong team prior (-1/0/1) and list of SET PIECES (start, end, side)."""
    strong = np.full(n_frames, NOBODY, dtype=np.int8)
    set_pieces = []
    for event in events:
        code = (event["code"] or "").strip().upper()
        side = team_side(event["tags"].get("Team"))
        start = max(0, event["start_frame"])
        end = min(n_frames, event["end_frame"])
        if start >= end:
            continue
        if code == "SET PIECES":
            set_pieces.append((start, end, side))
            continue
        if side == NOBODY:
            continue
        if code in STRONG_CODES:
            strong[start:end] = side
        elif code in STRONG_DEF_CODES:
            strong[start:end] = HOME if side == AWAY else AWAY
    return strong, set_pieces


def set_pieces_mask(events, n_frames: int) -> np.ndarray:
    """True on frames inside any SET PIECES [start, end)."""
    mask = np.zeros(n_frames, dtype=bool)
    for event in events:
        if (event["code"] or "").strip().upper() != "SET PIECES":
            continue
        start = max(0, event["start_frame"])
        end = min(n_frames, event["end_frame"])
        if start < end:
            mask[start:end] = True
    return mask


def build_player_caches(raw_path, team, n, pitch):
    """One ATD pass: point-cloud and activity features."""
    L, W = pitch
    n_slots = len(team)
    med_spd = np.full(n, np.nan)
    pct_move = np.full(n, np.nan)
    pct_run = np.full(n, np.nan)
    pct_slow = np.full(n, np.nan)
    n_valid = np.zeros(n, dtype=np.int16)
    n_home = np.zeros(n, dtype=np.int16)
    n_away = np.zeros(n, dtype=np.int16)
    med_home = np.full(n, np.nan)
    med_away = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    spread_H = np.full(n, np.nan)
    spread_A = np.full(n, np.nan)
    area_proxy = np.full(n, np.nan)
    cent_shift = np.full(n, np.nan)
    near_bound_frac = np.full(n, np.nan)
    ball_cluster = np.full(n, np.nan)

    prev_cent = None
    prev_f = -2

    with open(raw_path) as h:
        for line in h:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= n:
                continue
            groups = parts[1].split(GROUP_SEP)
            xs, ys, spds = [], [], []
            spds_H, spds_A = [], []
            pos_H, pos_A = [], []
            nh = na = 0
            for slot, g in enumerate(groups):
                if slot >= n_slots:
                    break
                p = parse_xy_speed(g, L, W)
                if p is None:
                    continue
                x, y, spd = p
                tm = int(team[slot])
                if tm == HOME:
                    nh += 1
                    pos_H.append((x, y))
                elif tm == AWAY:
                    na += 1
                    pos_A.append((x, y))
                xs.append(x)
                ys.append(y)
                if np.isfinite(spd) and SPEED_LO <= spd <= SPEED_HI:
                    spds.append(spd)
                    if tm == HOME:
                        spds_H.append(spd)
                    elif tm == AWAY:
                        spds_A.append(spd)
            n_valid[frame] = len(spds)
            n_home[frame] = nh
            n_away[frame] = na
            if spds:
                med_spd[frame] = float(np.median(spds))
                pct_move[frame] = float(np.mean([s >= MOVE_MPS for s in spds]))
                pct_run[frame] = float(np.mean([s >= RUN_MPS for s in spds]))
                pct_slow[frame] = float(np.mean([s < SLOW_MPS for s in spds]))
            if spds_H:
                med_home[frame] = float(np.median(spds_H))
            if spds_A:
                med_away[frame] = float(np.median(spds_A))
            if len(xs) >= MIN_POS_SPREAD:
                X = np.array(xs)
                Y = np.array(ys)
                cx, cy = X.mean(), Y.mean()
                spread[frame] = float(np.mean(np.hypot(X - cx, Y - cy)))
                area_proxy[frame] = float((X.max() - X.min()) * (Y.max() - Y.min()))
                near_bound_frac[frame] = float(np.mean(
                    (X < NEAR_BOUND_M) | (X > L - NEAR_BOUND_M) |
                    (Y < NEAR_BOUND_M) | (Y > W - NEAR_BOUND_M)
                ))
                if prev_cent is not None and prev_f == frame - 1:
                    cent_shift[frame] = float(np.hypot(cx - prev_cent[0], cy - prev_cent[1]) * 25.0)
                prev_cent = (cx, cy)
                prev_f = frame
                if len(pos_H) >= MIN_TEAM_SPREAD:
                    Xm = np.array([p[0] for p in pos_H])
                    Ym = np.array([p[1] for p in pos_H])
                    c = (Xm.mean(), Ym.mean())
                    spread_H[frame] = float(np.mean(np.hypot(Xm - c[0], Ym - c[1])))
                if len(pos_A) >= MIN_TEAM_SPREAD:
                    Xm = np.array([p[0] for p in pos_A])
                    Ym = np.array([p[1] for p in pos_A])
                    c = (Xm.mean(), Ym.mean())
                    spread_A[frame] = float(np.mean(np.hypot(Xm - c[0], Ym - c[1])))
                ball = parse_xy_speed(parts[2], L, W)
                if ball is not None:
                    bx, by, _ = ball
                    if np.isfinite(bx) and np.isfinite(by):
                        ball_cluster[frame] = float(np.mean(np.hypot(X - bx, Y - by) <= BALL_CLUSTER_M))
            else:
                prev_cent = None

    pointcloud = dict(
        med_spd=med_spd, pct_move=pct_move, pct_run=pct_run, n_valid=n_valid,
        spread=spread, spread_H=spread_H, spread_A=spread_A, area_proxy=area_proxy,
        cent_shift=cent_shift, near_bound_frac=near_bound_frac, ball_cluster=ball_cluster,
    )
    activity = dict(
        med_spd=med_spd, pct_slow=pct_slow, n_valid=n_valid,
        n_home=n_home, n_away=n_away, med_home=med_home, med_away=med_away,
        SLOW=np.float64(SLOW_MPS),
    )
    return pointcloud, activity


def scan_extended(raw_path, team_of_track, n_frames, pitch, set_mask):
    """Per-frame nearest player, alignment, and ball position from ATD."""
    length, width = pitch
    fps = float(FPS)
    n_slots = len(team_of_track)

    nearest_team = np.full(n_frames, NOBODY, dtype=np.int8)
    nearest_dist = np.full(n_frames, np.nan, dtype=np.float64)
    align_ok = np.zeros(n_frames, dtype=bool)
    ball_present = np.zeros(n_frames, dtype=bool)
    ball_x = np.full(n_frames, np.nan, dtype=np.float64)
    ball_y = np.full(n_frames, np.nan, dtype=np.float64)
    min_dist_h = np.full(n_frames, np.nan, dtype=np.float64)
    min_dist_a = np.full(n_frames, np.nan, dtype=np.float64)

    prev_ball = None
    prev_players = np.full((n_slots, 2), np.nan, dtype=np.float64)
    prev_players_frame = -2
    cos_min = float(np.cos(np.deg2rad(ALIGN_MAX_ANGLE_DEG)))

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= n_frames:
                continue

            cur_xy = np.full((n_slots, 2), np.nan, dtype=np.float64)
            groups = parts[1].split(GROUP_SEP)
            for slot, group in enumerate(groups):
                if slot >= n_slots:
                    break
                parsed = parse_xy_speed(group, length, width)
                if parsed is None:
                    continue
                cur_xy[slot, 0] = parsed[0]
                cur_xy[slot, 1] = parsed[1]

            ball = parse_xy_speed(parts[2], length, width)
            if ball is None:
                prev_ball = None
                prev_players = cur_xy
                prev_players_frame = frame
                continue

            bx, by, _bs = ball
            ball_present[frame] = True
            ball_x[frame] = bx
            ball_y[frame] = by

            best_d2 = None
            best_slot = -1
            h_best = None
            a_best = None
            for slot in range(n_slots):
                if np.isnan(cur_xy[slot, 0]):
                    continue
                dx = cur_xy[slot, 0] - bx
                dy = cur_xy[slot, 1] - by
                d2 = dx * dx + dy * dy
                team = int(team_of_track[slot])
                if team == HOME:
                    if h_best is None or d2 < h_best:
                        h_best = d2
                elif team == AWAY:
                    if a_best is None or d2 < a_best:
                        a_best = d2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_slot = slot

            if h_best is not None:
                min_dist_h[frame] = h_best ** 0.5
            if a_best is not None:
                min_dist_a[frame] = a_best ** 0.5

            if best_slot < 0:
                prev_ball = (frame, bx, by)
                prev_players = cur_xy
                prev_players_frame = frame
                continue

            nearest_dist[frame] = best_d2 ** 0.5
            nearest_team[frame] = int(team_of_track[best_slot])

            ball_vx = ball_vy = player_vx = player_vy = float("nan")
            if prev_ball is not None and prev_ball[0] == frame - 1:
                ball_vx = (bx - prev_ball[1]) * fps
                ball_vy = (by - prev_ball[2]) * fps
            if (
                prev_players_frame == frame - 1
                and not np.isnan(prev_players[best_slot, 0])
                and not np.isnan(cur_xy[best_slot, 0])
            ):
                player_vx = (cur_xy[best_slot, 0] - prev_players[best_slot, 0]) * fps
                player_vy = (cur_xy[best_slot, 1] - prev_players[best_slot, 1]) * fps

            b_spd = float(np.hypot(ball_vx, ball_vy)) if not np.isnan(ball_vx) else float("nan")
            p_spd = float(np.hypot(player_vx, player_vy)) if not np.isnan(player_vx) else float("nan")
            ok = False
            if np.isnan(b_spd) or np.isnan(p_spd):
                ok = True
            elif b_spd < ALIGN_MIN_SPEED_MPS or p_spd < ALIGN_MIN_SPEED_MPS:
                ok = True
            else:
                denom = b_spd * p_spd
                cos = (ball_vx * player_vx + ball_vy * player_vy) / denom
                ok = cos >= cos_min
            align_ok[frame] = ok
            prev_ball = (frame, bx, by)
            prev_players = cur_xy
            prev_players_frame = frame

    return {
        "nearest_team": nearest_team,
        "nearest_dist": nearest_dist,
        "align_ok": align_ok,
        "ball_present": ball_present,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "set_mask": set_mask,
        "min_dist_h": min_dist_h,
        "min_dist_a": min_dist_a,
    }


def build_rb_channels(raw_path, team, events, n, pitch):
    nearest, nearest_d, bx, by = extract_atd_channels(raw_path, team, n, pitch)
    strong, set_pieces = smart_tag_channels(events, n)
    sp_start = np.array([s for s, e, side in set_pieces], dtype=np.int32)
    sp_end = np.array([e for s, e, side in set_pieces], dtype=np.int32)
    sp_side = np.array([side for s, e, side in set_pieces], dtype=np.int8)
    return dict(
        nearest=nearest, nearest_d=nearest_d, ball_x=bx, ball_y=by,
        strong=strong, sp_start=sp_start, sp_end=sp_end, sp_side=sp_side,
    ), set_pieces
