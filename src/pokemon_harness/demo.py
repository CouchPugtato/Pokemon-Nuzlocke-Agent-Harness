"""Synthetic integration demo. This does not exercise model intelligence."""

import json


def create_images(directory):
    from PIL import Image, ImageDraw

    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, caption in enumerate(["DEMO: near a doorway", "DEMO: doorway approached", "DEMO: inspect result"]):
        image = Image.new("RGB", (480, 320), (24, 48 + index * 25, 50))
        draw = ImageDraw.Draw(image)
        draw.rectangle((200, 50, 260, 130), fill=(110, 70, 30))
        draw.rectangle((220, 220 - index * 30, 240, 240 - index * 30), fill=(240, 240, 180))
        draw.text((20, 280), caption, fill="white")
        path = directory / f"{index:02d}.png"
        image.save(path)
        paths.append(path)
    return paths


class DemoAgent:
    def __init__(self):
        self.step = 0

    def decide(self, messages):
        context = json.loads(messages[1]["content"][0]["text"])
        self.step += 1
        kind = "stop" if self.step >= 3 else "press"
        value = {
            "observation_id": context["observation_id"], "scene": "overworld", "location": "Synthetic demo",
            "observation": "Synthetic fixture; this decision is scripted, not inferred by a model.",
            "action": {"kind": kind, "buttons": ["UP"] if kind == "press" else [],
                       "hold_frames": 8, "settle_frames": 4},
            "expected_result": "Inspect the next demo image" if kind == "press" else "Demo completed",
            "memory_updates": [{"category": "world", "key": "demo:doorway", "claim": "A doorway is visible in the synthetic scene.",
                                "location": "Synthetic demo", "basis": "observed", "confidence": "high", "status": "active"}],
            "events": [],
            "memory_search": None,
        }
        return value, {"backend": "scripted demo", "decision": value}
