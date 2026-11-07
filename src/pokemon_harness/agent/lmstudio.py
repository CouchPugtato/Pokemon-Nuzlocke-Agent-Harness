"""OpenAI-compatible HTTP client, with no cloud fallback or automatic retries."""

import json
import os
import socket
from urllib.error import URLError
from urllib.request import Request, urlopen

from .schemas import DECISION_SCHEMA


class InvalidModelResponse(ValueError):
    def __init__(self, message, trace):
        super().__init__(message)
        self.trace = trace


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
            "max_tokens": self.config.get("max_tokens", 8192), "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "game_decision", "strict": True, "schema": DECISION_SCHEMA}},
        }
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.config.get("api_key_env", "LM_STUDIO_API_KEY"))
        if token:
            headers["Authorization"] = "Bearer " + token
        request = Request(self.base_url + "/chat/completions",
                          data=json.dumps(payload).encode("utf-8"), headers=headers)
        timeout = self.config.get("timeout_seconds", 180)
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except (TimeoutError, socket.timeout) as exc:
            raise TimeoutError(f"LM Studio did not return a decision within {timeout}s. "
                               "No controller input was sent for this observation. "
                               "Check LM Studio generation/load status or increase model.timeout_seconds.") from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise TimeoutError(f"LM Studio connection timed out after {timeout}s. "
                                   "No controller input was sent for this observation.") from exc
            raise
        settings = {key: value for key, value in payload.items() if key != "messages"}
        trace = {"raw_response": raw, "request_settings": settings}
        try:
            envelope = json.loads(raw)
            trace = {"response": envelope, "request_settings": settings}
            choice = envelope["choices"][0]
            finish_reason = choice.get("finish_reason")
            if finish_reason != "stop":
                detail = f"Model did not finish a complete decision (finish_reason={finish_reason!r})"
                if finish_reason == "length":
                    detail += (f". Output reached the {payload['max_tokens']}-token limit; "
                               "reasoning and decision JSON share this budget. Increase model.max_tokens")
                raise ValueError(detail)
            value = json.loads(choice["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise InvalidModelResponse(f"Invalid LM Studio response: {exc}", trace) from exc
        # Screenshots already live in the observation archive; do not duplicate
        # base64 images in every trace file.
        return value, trace
