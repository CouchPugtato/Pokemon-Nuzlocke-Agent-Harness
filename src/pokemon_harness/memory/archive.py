"""bounded verified image retrieval from local observation records"""

import hashlib
import io
from pathlib import Path

from PIL import Image


def archive_settings(config):
    result = {"max_images": 4, "max_bytes": 8388608, "max_image_pixels": 4194304}
    value = config.get("archive_retrieval", {})
    if not isinstance(value, dict) or set(value) - set(result):
        raise ValueError("Invalid archive_retrieval settings")
    result.update(value)
    for key, lower, upper in (("max_images", 1, 4), ("max_bytes", 1024, 67108864),
                              ("max_image_pixels", 1, 16777216)):
        if type(result[key]) is not int or not lower <= result[key] <= upper:
            raise ValueError(f"archive_retrieval.{key} must be an integer between {lower} and {upper}")
    return result


def retrieve_images(memory, run_dir, observation_ids, current_id, settings, previous=()):
    frames = {frame["observation_id"]: frame for frame in previous}
    if len(set(observation_ids) | set(frames)) > settings["max_images"]:
        raise ValueError("Archived image retrieval exceeds the per decision image limit")
    root = Path(run_dir).resolve()
    records = []
    for source_id in observation_ids:
        if type(source_id) is not int or not 1 <= source_id <= current_id:
            raise ValueError("Archived observation id must belong to this run and not be in the future")
        row = memory.connection.execute("SELECT * FROM observations WHERE id=?", (source_id,)).fetchone()
        if row is None:
            raise ValueError("Archived observation id does not belong to this run")
        record = {"observation_id": source_id, "captured_at": row["created_at"],
                  "image_hash": row["image_hash"], "historical_only": True,
                  "age_observations_at_retrieval": current_id - source_id, "capture_timing": row["timing"]}
        path = (root / row["image_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Archived observation image path escapes the run directory")
        if source_id in frames:
            records.append(dict(record, read_status="verified"))
            continue
        remaining = settings["max_bytes"] - sum(len(frame["image"]) for frame in frames.values())
        try:
            with path.open("rb") as stream:
                data = stream.read(remaining + 1)
        except FileNotFoundError:
            records.append(dict(record, read_status="missing"))
            continue
        if len(data) > remaining:
            records.append(dict(record, read_status="byte_limit"))
            continue
        if hashlib.sha256(data).hexdigest() != row["image_hash"]:
            raise ValueError("Archived observation image fingerprint mismatch")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.width * image.height > settings["max_image_pixels"]:
                raise ValueError("Archived observation image has invalid format or dimensions")
            image.verify()
        frames[source_id] = {"observation_id": source_id, "image": data, "purpose": "archive"}
        records.append(dict(record, read_status="verified"))
    return list(frames.values()), records
