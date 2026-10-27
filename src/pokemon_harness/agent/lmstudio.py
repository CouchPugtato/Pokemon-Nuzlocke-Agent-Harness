"""OpenAI-compatible HTTP client, with no cloud fallback or automatic retries."""

import json
import os
from urllib.request import Request, urlopen

from .schemas import DECISION_SCHEMA


class LMStudio:
    def __init__(self, config):
        self.config = config
        self.base_url = config.get("base_url", "http://127.0.0.1:1234/v1").rstrip("/")
        if not config.get("name"):
            raise ValueError("model.name must be the loaded vision model identifier")

    def decide(self, messages):
        payload = {
            "model": self.config["name"], "messages": messages,
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": self.config.get("max_tokens", 2000), "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "game_decision", "strict": True, "schema": DECISION_SCHEMA}},
        }
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.config.get("api_key_env", "LM_STUDIO_API_KEY"))
        if token:
            headers["Authorization"] = "Bearer " + token
        request = Request(self.base_url + "/chat/completions",
                          data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urlopen(request, timeout=self.config.get("timeout_seconds", 180)) as response:
            raw = response.read().decode("utf-8")
        try:
            envelope = json.loads(raw)
            choice = envelope["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Model did not finish a complete decision")
            value = json.loads(choice["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid LM Studio response: {exc}") from exc
        # Screenshots already live in the observation archive; do not duplicate
        # base64 images in every trace file.
        settings = {key: value for key, value in payload.items() if key != "messages"}
        return value, {"response": envelope, "request_settings": settings}
