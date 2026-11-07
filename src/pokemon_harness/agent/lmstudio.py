"""local structured generation without cloud fallback or automatic retries"""

import json
import os
import socket
from urllib.error import URLError
from urllib.request import Request, urlopen

from .schemas import request_schema
from jsonschema import Draft202012Validator, ValidationError
from .context_budget import assess_request, budget_settings, calibrate_usage


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
        self.inference_model = config["name"]
        self.context_length = config.get("context_length", 32768)
        self.loaded_context_length = None
        self.budget_factor = 1.0
        budget_settings(config)
        if type(self.context_length) is not int or self.context_length < 1024:
            raise ValueError("model.context_length must be an integer of at least 1024")
        if type(config.get("max_tokens", 32768)) is not int or config.get("max_tokens", 32768) < 1:
            raise ValueError("model.max_tokens must be a positive integer")
        if "thinking" in config and type(config["thinking"]) is not bool:
            raise ValueError("model.thinking must be true or false")

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.config.get("api_key_env", "LM_STUDIO_API_KEY"))
        if token:
            headers["Authorization"] = "Bearer " + token
        return headers

    def _native(self, endpoint, payload=None):
        if not self.base_url.endswith("/v1"):
            raise ValueError("LM Studio base_url must end in /v1 for managed model loading")
        url = self.base_url[:-3] + "/api/v1/models" + endpoint
        request = Request(url, headers=self._headers(),
                          data=json.dumps(payload).encode("utf-8") if payload is not None else None)
        with urlopen(request, timeout=self.config.get("timeout_seconds", 180)) as response:
            return json.loads(response.read().decode("utf-8"))

    def prepare(self):
        """loading and verifying context independently of generation limits"""
        models = self._native("")["models"]
        name = self.config["name"]
        matches = [model for model in models if name == model["key"]
                   or name in model.get("variants", [])
                   or any(instance["id"] == name for instance in model.get("loaded_instances", []))]
        if not matches:
            matches = [model for model in models if model["key"].rsplit("/", 1)[-1] == name]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely resolve model {name!r} in LM Studio; use its full model key")
        model = matches[0]
        maximum = model.get("max_context_length")
        if maximum and self.context_length > maximum:
            raise ValueError(f"Requested context {self.context_length} exceeds model maximum {maximum}")
        instances = model.get("loaded_instances", [])
        for instance in instances:
            actual = instance.get("config", {}).get("context_length", 0)
            if actual >= self.context_length:
                self.inference_model = instance["id"]
                self.loaded_context_length = actual
                print(f"LM Studio: {self.inference_model}, verified context {actual} tokens", flush=True)
                return
        # reloading only an instance of the selected model
        if instances:
            self._native("/unload", {"instance_id": instances[0]["id"]})
        print(f"Loading {model['key']} with context {self.context_length} tokens...", flush=True)
        payload = {"model": name if name in model.get("variants", []) else model["key"],
                   "context_length": self.context_length, "echo_load_config": True}
        loaded = self._native("/load", payload)
        actual = loaded.get("load_config", {}).get("context_length", 0)
        # checking reused instances against the requested context
        if loaded.get("instance_id") and actual < self.context_length:
            self._native("/unload", {"instance_id": loaded["instance_id"]})
            loaded = self._native("/load", payload)
            actual = loaded.get("load_config", {}).get("context_length", 0)
        if loaded.get("status") != "loaded" or not loaded.get("instance_id") or actual < self.context_length:
            raise ValueError(f"LM Studio did not confirm context {self.context_length}; reported {actual}. "
                             "Stopping before game input")
        self.inference_model = loaded["instance_id"]
        self.loaded_context_length = actual
        print(f"LM Studio: {self.inference_model}, verified context {actual} tokens", flush=True)

    def decide(self, messages):
        schema = request_schema(messages)
        budget = assess_request(messages, self.config, self.loaded_context_length, self.budget_factor)
        if budget["effective_output_tokens"] != budget["requested_output_tokens"]:
            print(f"Context budget: estimated input {budget['estimated_input_tokens']}, "
                  f"output {budget['requested_output_tokens']} -> {budget['effective_output_tokens']}, "
                  f"safety {budget['safety_margin_tokens']} tokens", flush=True)
        payload = {
            "model": self.inference_model, "messages": messages,
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": budget["effective_output_tokens"], "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "game_decision", "strict": True, "schema": schema}},
        }
        if "thinking" in self.config:
            payload["reasoning_effort"] = "medium" if self.config["thinking"] else "none"
        request = Request(self.base_url + "/chat/completions",
                          data=json.dumps(payload).encode("utf-8"), headers=self._headers())
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
        trace = {"raw_response": raw, "request_settings": settings, "context_budget": budget}
        try:
            envelope = json.loads(raw)
            budget["calibration_suggestion"] = calibrate_usage(budget, envelope.get("usage"))
            self.budget_factor = budget["calibration_suggestion"]
            trace = {"response": envelope, "request_settings": settings, "context_budget": budget}
            choice = envelope["choices"][0]
            finish_reason = choice.get("finish_reason")
            if finish_reason != "stop":
                detail = f"Model did not finish a complete decision (finish_reason={finish_reason!r})"
                if finish_reason == "length":
                    usage = envelope.get("usage") or {}
                    detail += (f". Token usage: prompt={usage.get('prompt_tokens', 'unknown')}, "
                               f"completion={usage.get('completion_tokens', 'unknown')}, "
                               f"total={usage.get('total_tokens', 'unknown')}; "
                               f"requested output limit={payload['max_tokens']}. "
                               "Reasoning and decision JSON share the output budget. ")
                    if usage.get("completion_tokens", 0) >= payload["max_tokens"]:
                        detail += ("The output budget was exhausted. Use --max-tokens to adjust it; "
                                   "if reasoning is repetitive, try --thinking off instead")
                    else:
                        detail += ("The loaded context window may be full: increase --context-length "
                                   "or reduce prompt/history")
                raise ValueError(detail)
            value = json.loads(choice["message"]["content"])
            if "anyOf" in schema:
                Draft202012Validator(schema).validate(value)
        except ValidationError as exc:
            raise InvalidModelResponse(f"Invalid LM Studio phase response: {exc.message[:600]}", trace) from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise InvalidModelResponse(f"Invalid LM Studio response: {exc}", trace) from exc
        # screenshots staying in the observation archive
        # avoiding duplicate image bytes in traces
        return value, trace
