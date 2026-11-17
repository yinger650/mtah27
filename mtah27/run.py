"""Build the possession submission CSV from the official data bundle."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from mtah27.config import EXPECTED_SHA256, N_FRAMES, PITCH
from mtah27.data import load_smart_tagging, require_assets, resolve_data_dir, team_of_each_track
from mtah27.features import build_player_caches, build_rb_channels, scan_extended, set_pieces_mask
from mtah27.labels import assign_labels, build_evidence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_labels(path: Path, labs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        for i, lab in enumerate(labs):
            handle.write(f"{i},{lab}\n")


def generate(data_dir: Path):
    assets = require_assets(data_dir)
    n = N_FRAMES
    pitch = PITCH
    print(f"loading tactical ATD and Smart Tagging from {data_dir}", flush=True)
    team = team_of_each_track(assets["atd_meta"])
    events = load_smart_tagging(assets["smart_tagging"])
    raw = assets["atd_raw"]

    print("building player activity features", flush=True)
    pointcloud, activity = build_player_caches(raw, team, n, pitch)

    print("scanning ball–player geometry", flush=True)
    set_mask = set_pieces_mask(events, n)
    scan = scan_extended(raw, team, n, pitch, set_mask)

    print("extracting nearest-player and phase channels", flush=True)
    channels, set_pieces = build_rb_channels(raw, team, events, n, pitch)

    ev = build_evidence(scan, pointcloud, activity, pitch)
    print("assigning labels", flush=True)
    labels = assign_labels(channels, set_pieces, pitch, activity, ev)
    if len(labels) != n:
        raise RuntimeError(f"expected {n} labels, got {len(labels)}")
    return labels


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate the Phase 1 possession submission from the official data bundle.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to the unpacked data bundle (the folder that contains tactical/). "
             "May also be a parent with data/tactical/, or the tactical/ folder itself. "
             "Defaults to MTAH_DATA_DIR or ./data.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to the output CSV (frame,label).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = resolve_data_dir(args.data)
    labels = generate(data_dir)
    out = Path(args.out)
    write_labels(out, labels)
    digest = sha256_file(out)
    print(f"wrote {out} ({len(labels)} frames)")
    print(f"sha256 {digest}")
    if digest == EXPECTED_SHA256:
        print("sha256 matches the submitted file")
    else:
        print(
            f"warning: sha256 differs from the submitted file "
            f"({EXPECTED_SHA256})",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
