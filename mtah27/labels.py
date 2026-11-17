"""Rule-based possession assignment, activity overlay, and temporal correction."""

from __future__ import annotations

import numpy as np

from mtah27.config import (
    BRIDGE_ACROSS,
    BRIDGE_GAP_FRAMES,
    C0_MED_LOW,
    C0_MIN_TEAM,
    C0_MIN_VALID,
    C0_SLOW_FRAC,
    C0_SUSTAIN,
    CLOSE_M,
    FALLBACK_PREFIX,
    GAP_FILL,
    GAP_KILL,
    GEO_LIVE_DIST_M,
    GEO_STREAK,
    KICK_CONSEC,
    KICK_DISP_M,
    MAX_PREFIX,
    MED_THR,
    MIN_HA_FRAMES,
    MIN_PREFIX,
    MISS_D_RUN,
    MOVE_FRAC,
    NOBODY,
    OUT_MARGIN_M,
    PROTECT_D_MIN,
    REF_SEARCH,
    RELEASE_FALSE_D,
    SET_LEAD,
    SMOOTH_W,
    SWITCH_NEED,
)


def run_length_true(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(len(mask), dtype=np.int32)
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        out[i:j] = j - i
        i = j
    return out


def sustain_true(mask, min_len):
    out = np.zeros(len(mask), dtype=bool)
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_len:
            out[i:j] = True
        i = j
    return out


def dead_mask(ball_x, ball_y, pitch, set_pieces,
              miss_d_run=MISS_D_RUN, set_lead=SET_LEAD, set_dead_prefix=FALLBACK_PREFIX):
    length, width = pitch
    miss = np.isnan(ball_x)
    out = (~miss) & (
        (ball_x < -OUT_MARGIN_M) | (ball_x > length + OUT_MARGIN_M) |
        (ball_y < -OUT_MARGIN_M) | (ball_y > width + OUT_MARGIN_M)
    )
    miss_run = run_length_true(miss)
    dead = out | (miss & (miss_run >= miss_d_run))
    n = len(ball_x)
    for start, end, _side in set_pieces:
        d0 = max(0, start - set_lead)
        d1 = min(n, start + set_dead_prefix)
        dead[d0:d1] = True
    return dead


def sticky_reduce(nearest, nearest_d, strong, dead,
                  close_m=CLOSE_M, switch_need=SWITCH_NEED,
                  gap_fill=GAP_FILL, gap_kill=GAP_KILL):
    """Per-frame H/A/D codes as int8 (0/1/2)."""
    n = len(nearest)
    sig = nearest.copy()
    far = np.isnan(nearest_d) | (nearest_d > close_m)
    sig[far] = NOBODY
    sig = np.where(strong != NOBODY, strong, sig)

    lab = np.full(n, 2, dtype=np.int8)  # D
    cur = NOBODY
    pending = NOBODY
    pend_count = 0
    gap = 0
    for i in range(n):
        if dead[i]:
            lab[i] = 2
            cur = NOBODY
            pending = NOBODY
            pend_count = 0
            gap = 0
            continue
        s = int(sig[i])
        if s == NOBODY:
            gap += 1
            if cur != NOBODY and gap <= gap_kill:
                lab[i] = cur
            else:
                lab[i] = 2
                cur = NOBODY
            continue
        gap = 0
        if cur == NOBODY:
            cur = s
            lab[i] = cur
            pending = NOBODY
            pend_count = 0
            continue
        if s == cur:
            lab[i] = cur
            pending = NOBODY
            pend_count = 0
            continue
        if pending == s:
            pend_count += 1
        else:
            pending = s
            pend_count = 1
        if pend_count >= switch_need:
            cur = s
            pending = NOBODY
            pend_count = 0
        lab[i] = cur
    return lab


def smooth_had(codes: np.ndarray, window: int) -> np.ndarray:
    """Majority vote over {H,A,D} with half-width `window`. codes are 0/1/2."""
    n = len(codes)
    if window <= 0:
        return np.array(["H", "A", "D"])[codes]
    ch = np.concatenate(([0], np.cumsum(codes == 0)))
    ca = np.concatenate(([0], np.cumsum(codes == 1)))
    cd = np.concatenate(([0], np.cumsum(codes == 2)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - window)
    hi = np.minimum(n, idx + window + 1)
    choice = np.argmax(np.stack([ch[hi] - ch[lo], ca[hi] - ca[lo], cd[hi] - cd[lo]], 0), 0)
    return np.array(["H", "A", "D"])[choice]


def kick_offsets(bx, by, set_pieces,
                 ref_search=REF_SEARCH, max_prefix=MAX_PREFIX,
                 kick_disp_m=KICK_DISP_M, kick_consec=KICK_CONSEC):
    """Per set piece: kick offset from the tag start, or None."""
    n = len(bx)
    out = []
    for start, _e, _s in set_pieces:
        start = int(start)
        ref = None
        f0 = None
        for f in range(start, min(n, start + ref_search + 1)):
            if np.isfinite(bx[f]):
                ref = (bx[f], by[f])
                f0 = f
                break
        k = None
        if ref is not None:
            run = 0
            first = None
            for g in range(f0 + 1, min(n, start + max_prefix + 1)):
                if not np.isfinite(bx[g]):
                    continue
                if np.hypot(bx[g] - ref[0], by[g] - ref[1]) > kick_disp_m:
                    run += 1
                    first = g if run == 1 else first
                    if run >= kick_consec:
                        k = first - start
                        break
                else:
                    run = 0
                    first = None
        out.append(k)
    return out


def set_ends(set_pieces, kicks, fallback_prefix=FALLBACK_PREFIX,
             min_prefix=MIN_PREFIX, max_prefix=MAX_PREFIX):
    ends = []
    for (start, _e, _s), k in zip(set_pieces, kicks):
        if k is None:
            ends.append(int(start) + fallback_prefix)
        else:
            ends.append(int(start) + int(np.clip(k, min_prefix, max_prefix)))
    return ends


def kick_dead_mask(bx, by, pitch, set_pieces, ends,
                   miss_d_run=MISS_D_RUN, set_lead=SET_LEAD):
    dead = dead_mask(bx, by, pitch, [], miss_d_run=miss_d_run)
    n = len(bx)
    for (start, _e, _s), e in zip(set_pieces, ends):
        dead[max(0, int(start) - set_lead):min(n, e)] = True
    return dead


def c0_activity_masks(ch, act, pitch):
    med, pct, nv = act["med_spd"], act["pct_slow"], act["n_valid"]
    nh, na = act["n_home"], act["n_away"]
    gated = np.isfinite(med) & (nv >= C0_MIN_VALID) & (nh >= C0_MIN_TEAM) & (na >= C0_MIN_TEAM)
    low = gated & (med < C0_MED_LOW) & (pct > C0_SLOW_FRAC)
    low_sust = sustain_true(low, C0_SUSTAIN)
    L, W = pitch
    bx, by = ch["ball_x"], ch["ball_y"]
    miss = np.isnan(bx)
    op = ((bx < -OUT_MARGIN_M) | (bx > L + OUT_MARGIN_M) |
          (by < -OUT_MARGIN_M) | (by > W + OUT_MARGIN_M)) & ~miss
    return dict(gated=gated, low=low, low_sust=low_sust, miss=miss, op=op)


def post_motion_miss_D(labels, miss, low_sust, op):
    """Write D where sustained low activity meets a missing ball, and on out-of-pitch frames."""
    out = labels.copy()
    dead = (low_sust & miss) | op
    out[dead] = "D"
    return out, dead


def build_evidence(scan, pc, act, pitch):
    L, W = pitch
    bx, by = scan["ball_x"], scan["ball_y"]
    miss = ~np.isfinite(bx)
    outb = ((bx < -OUT_MARGIN_M) | (bx > L + OUT_MARGIN_M) |
            (by < -OUT_MARGIN_M) | (by > W + OUT_MARGIN_M)) & ~miss
    geo_live = (
        scan["ball_present"] & (~outb)
        & np.isfinite(scan["nearest_dist"]) & (scan["nearest_dist"] <= GEO_LIVE_DIST_M)
        & scan["align_ok"] & (scan["nearest_team"] != NOBODY)
    )
    med, move, nv = pc["med_spd"], pc["pct_move"], pc["n_valid"]
    nh, na = act["n_home"], act["n_away"]
    gated = np.isfinite(med) & (nv >= C0_MIN_VALID) & (nh >= C0_MIN_TEAM) & (na >= C0_MIN_TEAM)
    ball_in = (~miss) & (~outb)
    return geo_live, med, move, gated, ball_in


def runs_of(labs: np.ndarray):
    """[(start, end_exclusive, label)]"""
    out = []
    n = len(labs)
    i = 0
    while i < n:
        j = i + 1
        while j < n and labs[j] == labs[i]:
            j += 1
        out.append((i, j, labs[i]))
        i = j
    return out


def release_false_D(labs, geo_live, med, move, gated, ball_in, *,
                    geo_streak: int, med_thr: float, move_frac: float = MOVE_FRAC):
    """Convert D to the preceding H/A carrier where live evidence persists."""
    out = labs.copy()
    n = len(out)
    carrier = out.copy()
    last = "H"
    for i in range(n):
        if carrier[i] in ("H", "A"):
            last = carrier[i]
        elif carrier[i] == "D":
            carrier[i] = last
        else:
            carrier[i] = last
    last = "H"
    for i in range(n - 1, -1, -1):
        if labs[i] in ("H", "A"):
            last = labs[i]
        elif out[i] == "D" and carrier[i] not in ("H", "A"):
            carrier[i] = last

    streak = 0
    for i in range(n):
        if out[i] != "D":
            streak = 0
            continue
        live = bool(geo_live[i]) or (
            bool(gated[i]) and float(med[i]) >= med_thr
            and float(move[i]) >= move_frac and bool(ball_in[i])
        )
        streak = streak + 1 if live else 0
        if streak >= geo_streak:
            for j in range(i - streak + 1, i + 1):
                if out[j] == "D":
                    out[j] = carrier[j]
    return out


def absorb_short_HA(labs, min_ha: int):
    if min_ha <= 0:
        return labs.copy()
    out = labs.copy()
    for _ in range(3):
        rr = runs_of(out)
        changed = False
        for idx, (a, b, lab) in enumerate(rr):
            if lab not in ("H", "A") or (b - a) >= min_ha:
                continue
            left = rr[idx - 1][2] if idx > 0 else None
            right = rr[idx + 1][2] if idx + 1 < len(rr) else None
            repl = None
            if left in ("H", "A", "D") and left == right:
                repl = left
            elif left == "D" or right == "D":
                repl = "D"
            elif left in ("H", "A") and right in ("H", "A"):
                repl = "D"
            elif left in ("H", "A"):
                repl = left
            elif right in ("H", "A"):
                repl = right
            else:
                repl = "D"
            if repl is not None and repl != lab:
                out[a:b] = repl
                changed = True
        if not changed:
            break
    return out


def bridge_same_team(labs, gap_max: int, across: str, protect_D_min: int):
    """Fill short interruptions between same-team H/A spells."""
    if gap_max <= 0:
        return labs.copy()
    out = labs.copy()
    rr = runs_of(out)
    for idx in range(1, len(rr) - 1):
        a0, b0, l0 = rr[idx - 1]
        a1, b1, l1 = rr[idx]
        a2, b2, l2 = rr[idx + 1]
        if l0 not in ("H", "A") or l0 != l2:
            continue
        gap = b1 - a1
        if gap <= 0 or gap > gap_max:
            continue
        mid = out[a1:b1]
        if across == "D_only":
            if not np.all(mid == "D"):
                continue
            if gap >= protect_D_min:
                continue
        elif across == "D_or_opp":
            opp = "A" if l0 == "H" else "H"
            ok = np.isin(mid, ["D", opp]).all()
            if not ok:
                continue
            if np.all(mid == "D") and gap >= protect_D_min:
                continue
            if np.all(mid == opp) and gap >= min(gap_max, max(protect_D_min // 2, 20)):
                if gap > gap_max // 2 and gap >= 20:
                    continue
        else:
            raise ValueError(across)
        out[a1:b1] = l0
    return out


def apply_correction(base, ev, *, release, geo_streak, med_thr, min_ha, bridge, across, protect_D):
    geo_live, med, move, gated, ball_in = ev
    labs = base.copy()
    if release:
        labs = release_false_D(
            labs, geo_live, med, move, gated, ball_in,
            geo_streak=geo_streak, med_thr=med_thr,
        )
    if min_ha > 0:
        labs = absorb_short_HA(labs, min_ha)
    if bridge > 0:
        labs = bridge_same_team(labs, bridge, across, protect_D)
    labs = labs.copy()
    labs[~np.isin(labs, list("HAD"))] = "D"
    return labs


def assign_labels(ch, set_pieces, pitch, act, ev):
    """Full submitted pipeline: kick-anchored dead windows, sticky+smooth, overlay, correction."""
    bx, by = ch["ball_x"], ch["ball_y"]
    kicks = kick_offsets(bx, by, set_pieces)
    ends = set_ends(set_pieces, kicks)
    dead = kick_dead_mask(bx, by, pitch, set_pieces, ends)
    codes = sticky_reduce(
        ch["nearest"], ch["nearest_d"], ch["strong"], dead,
        close_m=CLOSE_M, switch_need=SWITCH_NEED, gap_kill=GAP_KILL,
    )
    labs0 = smooth_had(codes, SMOOTH_W)
    masks = c0_activity_masks(ch, act, pitch)
    labs_c, _ = post_motion_miss_D(labs0, masks["miss"], masks["low_sust"], masks["op"])
    final = apply_correction(
        labs_c, ev,
        release=RELEASE_FALSE_D, geo_streak=GEO_STREAK, med_thr=MED_THR,
        min_ha=MIN_HA_FRAMES, bridge=BRIDGE_GAP_FRAMES,
        across=BRIDGE_ACROSS, protect_D=PROTECT_D_MIN,
    )
    return final
