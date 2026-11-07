"""bounded screenshot sampling without semantic completion claims"""

import io
import hashlib
import math
import time
from datetime import datetime, timezone

from PIL import Image, ImageChops, ImageDraw, ImageStat


SCENES = {"overworld", "dialogue", "menu", "battle", "transition", "unknown"}


def capture_settings(config):
    result = {"enabled": config.get("emulator", {}).get("backend") in {"vbam", "mgba"},
              "min_delay_ms": 50, "interval_ms": 75, "max_ms": 750, "max_samples": 8,
              "stable_comparisons": 2, "difference_threshold": 1.0, "masks": [],
              "scene_max_ms": {"battle": 250, "dialogue": 250, "menu": 250, "overworld": 350, "unknown": 350, "transition": 600}}
    value = config.get("capture_settling", {})
    if not isinstance(value, dict) or set(value) - set(result):
        raise ValueError("Invalid capture_settling settings")
    result.update(value)
    if type(result["enabled"]) is not bool:
        raise ValueError("capture_settling.enabled must be a boolean")
    for key, lower, upper in (("min_delay_ms", 0, 2000), ("interval_ms", 1, 1000), ("max_ms", 1, 5000),
                              ("max_samples", 1, 32), ("stable_comparisons", 1, 8)):
        if type(result[key]) is not int or not lower <= result[key] <= upper:
            raise ValueError(f"capture_settling.{key} must be an integer between {lower} and {upper}")
    if result["min_delay_ms"] > result["max_ms"]:
        raise ValueError("capture_settling.min_delay_ms must not exceed max_ms")
    threshold = result["difference_threshold"]
    if type(threshold) not in {int, float} or not math.isfinite(threshold) or not 0 <= threshold <= 255:
        raise ValueError("capture_settling.difference_threshold must be finite between zero and two hundred fifty five")
    profiles = result["scene_max_ms"]
    if not isinstance(profiles, dict) or set(profiles) - SCENES or any(
            type(limit) is not int or not result["min_delay_ms"] <= limit <= 5000 or limit < 1 for limit in profiles.values()):
        raise ValueError("Invalid capture_settling.scene_max_ms limits")
    masks = result["masks"]
    if not isinstance(masks, list) or len(masks) > 8 or any(
            not isinstance(rect, list) or len(rect) != 4 or any(type(number) is not int for number in rect)
            or min(rect[:2]) < 0 or min(rect[2:]) < 1 for rect in masks):
        raise ValueError("capture_settling.masks must contain bounded pixel rectangles")
    return result


def comparison_image(png, masks):
    with Image.open(io.BytesIO(png)) as source:
        image = source.convert("RGB")
    coverage = Image.new("1", image.size, 1) if masks else None
    for left, top, width, height in masks:
        if left + width > image.width or top + height > image.height:
            raise ValueError("Capture settling mask lies outside the screenshot")
        ImageDraw.Draw(image).rectangle((left, top, left + width - 1, top + height - 1), fill=(0, 0, 0))
        ImageDraw.Draw(coverage).rectangle((left, top, left + width - 1, top + height - 1), fill=0)
    if coverage is not None and coverage.getbbox() is None:
        raise ValueError("Capture settling masks cannot hide the entire screenshot")
    return image


def settled_capture(emulator, settings, control=None, scene="unknown", clock=time.monotonic, sleep=time.sleep):
    started = clock()
    budget_ms = min(settings["max_ms"], settings["scene_max_ms"].get(scene, settings["max_ms"]))
    deadline = started + budget_ms / 1000

    def check():
        if control:
            control.check()
        guard = getattr(emulator, "check_control", None)
        if guard:
            guard()
        if getattr(emulator, "held", ()):
            raise RuntimeError("Cannot settle screenshots while controller keys are held")

    def wait(seconds):
        end = min(clock() + seconds, deadline)
        while clock() < end:
            check()
            sleep(min(0.025, end - clock()))
        check()

    enabled = settings["enabled"]
    if enabled:
        wait(settings["min_delay_ms"] / 1000)
    previous = None
    stable = 0
    differences = []
    sample_hashes = []
    samples = 0

    def finish(status):
        finished = clock()
        return capture, {"schema_version": 1, "status": status, "samples": samples,
            "captured_at": captured_at, "elapsed_ms": round((finished - started) * 1000, 3),
            "last_capture_ms": round((after - before) * 1000, 3), "budget_ms": budget_ms,
            "deadline_exceeded": enabled and finished > deadline,
            "scene_hint": scene, "policy": settings, "differences": differences, "sample_hashes": sample_hashes,
            "comparison": "full resolution rgb mean absolute difference with configured masks",
            "semantic_completion_verified": False,
            "timing_limit": "Sampling stops between synchronous captures; an in progress capture cannot be preempted"}

    while True:
        check()
        captured_at = datetime.now(timezone.utc).isoformat()
        before = clock()
        capture = emulator.capture()
        after = clock()
        check()
        samples += 1
        sample_hashes.append(hashlib.sha256(capture.image).hexdigest())
        status = "disabled"
        if enabled:
            image = comparison_image(capture.image, settings["masks"])
            difference = None
            if previous is not None and image.size == previous.size:
                difference = sum(ImageStat.Stat(ImageChops.difference(previous, image)).mean) / 3
            differences.append(difference)
            stable = stable + 1 if difference is not None and difference <= settings["difference_threshold"] else 0
            previous = image
            status = "stable" if stable >= settings["stable_comparisons"] else "sampling"
            if status == "sampling" and clock() >= deadline:
                status = "time_limit"
            if status == "sampling" and samples >= settings["max_samples"]:
                status = "sample_limit"
        if status != "sampling":
            return finish(status)
        wait(settings["interval_ms"] / 1000)
        if clock() >= deadline:
            return finish("time_limit")


def capture_age(metadata):
    return max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(metadata["captured_at"])).total_seconds()) if metadata else None
