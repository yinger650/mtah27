"""Locate the official data bundle and load ATD metadata and Smart Tagging."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from mtah27.config import (
    ASSET_FILES,
    AWAY,
    AWAY_NAMES,
    FPS,
    HOME,
    HOME_NAMES,
    MISSING,
    NOBODY,
    SOURCE,
)


class DataBundleError(FileNotFoundError):
    """The provided path does not contain the expected tactical assets."""


def resolve_data_dir(data: str | os.PathLike[str] | None = None) -> Path:
    """Return the bundle root that contains ``tactical/``.

    Accepts any of:

    - the unpacked bundle itself (a directory containing ``tactical/``)
    - a parent that contains ``data/tactical/`` (official-repo layout)
    - the ``tactical/`` folder directly
    - ``None``, in which case ``MTAH_DATA_DIR`` or ``./data`` is used
    """
    candidates: list[Path] = []
    if data is not None:
        candidates.append(Path(data))
    env = os.environ.get("MTAH_DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(Path.cwd() / "data")

    tried: list[str] = []
    for raw in candidates:
        path = raw.expanduser()
        try:
            path = path.resolve()
        except OSError:
            tried.append(str(raw))
            continue
        for bundle in _bundle_options(path):
            if _is_bundle(bundle):
                return bundle
        tried.append(str(path))

    expected = ", ".join(ASSET_FILES.values())
    raise DataBundleError(
        "match data not found. Expected a directory containing "
        f"tactical/{{{expected}}}. Looked at: {', '.join(tried)}. "
        "Unpack the official bundle as described in DATA.md, or pass --data "
        "to the bundle root (the folder that contains tactical/), or set "
        "MTAH_DATA_DIR."
    )


def _bundle_options(path: Path) -> list[Path]:
    options = [path, path / "data"]
    if path.name == SOURCE:
        options.append(path.parent)
    return options


def _is_bundle(path: Path) -> bool:
    return all((path / SOURCE / name).is_file() for name in ASSET_FILES.values())


def asset_paths(data_dir: Path) -> dict[str, Path]:
    """Map asset keys to files under ``data_dir/tactical/``, without existence checks."""
    folder = data_dir / SOURCE
    return {key: folder / name for key, name in ASSET_FILES.items()}


def require_assets(data_dir: Path) -> dict[str, Path]:
    paths = asset_paths(data_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        listed = "\n  ".join(missing)
        raise DataBundleError(
            f"match data not found:\n  {listed}\n\n"
            f"Unpack the official bundle into {data_dir} "
            "(see DATA.md) or point --data / MTAH_DATA_DIR at that folder."
        )
    return paths


def team_of_each_track(meta_path: str | os.PathLike[str]) -> np.ndarray:
    """Array over track slots: 0 if the slot belongs to the home team, 1 if away.

    The first team in the metadata is the home side (FIFA/EPTS convention).
    Track slots are not stable player identities.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_ids)}")
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"{meta_path}: no Player entries")
    return np.array(
        [HOME if team_id == home_id else AWAY for team_id in slots],
        dtype=np.int8,
    )


def team_side(name: str | None) -> int:
    name = (name or "").strip()
    if name in HOME_NAMES:
        return HOME
    if name in AWAY_NAMES:
        return AWAY
    return NOBODY


def load_smart_tagging(path: str | os.PathLike[str], fps: int = FPS) -> list[dict]:
    """[{code, start_frame, end_frame, start_s, end_s, tags}] in file order.

    Frames are half-open on the right. Times in the XML are seconds from the
    first frame of the tactical video; they are converted with seconds x fps,
    rounded to the nearest frame.
    """
    out = []
    for instance in ET.parse(path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        out.append({
            "code": instance.findtext("code"),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "tags": {
                label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")
            },
        })
    return out


def parse_xy_speed(group: str, length: float, width: float):
    """Return (x_m, y_m, speed_or_nan) or None if the observation is missing."""
    if not group or group[0] == MISSING:
        return None
    comma = group.index(",")
    x = float(group[:comma]) * length
    rest = group[comma + 1:]
    comma2 = rest.index(",")
    y = float(rest[:comma2]) * width
    spd_s = rest[comma2 + 1:]
    spd = float("nan") if (not spd_s or spd_s[0] == MISSING) else float(spd_s)
    return x, y, spd
