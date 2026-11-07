import json
import unittest
from unittest.mock import patch

from pokemon_harness.agent.lmstudio import InvalidModelResponse, LMStudio


def catalog(instances=None):
    return {"models": [{"key": "qwen/vision", "max_context_length": 131072,
                        "loaded_instances": instances or []}]}


def loaded(context=32768, identifier="vision-instance"):
    return {"status": "loaded", "instance_id": identifier,
            "load_config": {"context_length": context}}


class ModelLoadingTests(unittest.TestCase):
    def test_thinking_toggle_is_sent_without_removing_structured_output(self):
        response = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
        for thinking in (True, False):
            with self.subTest(thinking=thinking), patch("pokemon_harness.agent.lmstudio.urlopen") as request:
                request.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
                LMStudio({"name": "test", "thinking": thinking, "max_tokens": 8192}).decide([])
                payload = json.loads(request.call_args.args[0].data)
                self.assertEqual(payload["reasoning_effort"], "medium" if thinking else "none")
                self.assertEqual(payload["response_format"]["type"], "json_schema")

    def test_invalid_output_or_thinking_options_are_rejected(self):
        for settings in ({"max_tokens": 0}, {"max_tokens": True}, {"thinking": "off"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                LMStudio(dict(name="test", **settings))

    def test_output_exhaustion_does_not_claim_context_is_full(self):
        response = {"usage": {"prompt_tokens": 5345, "completion_tokens": 8192, "total_tokens": 13537},
                    "choices": [{"finish_reason": "length", "message": {"content": ""}}]}
        with patch("pokemon_harness.agent.lmstudio.urlopen") as request:
            request.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            with self.assertRaises(InvalidModelResponse) as error:
                LMStudio({"name": "test", "max_tokens": 8192}).decide([])
        self.assertIn("output budget was exhausted", str(error.exception))
        self.assertIn("--thinking off", str(error.exception))
        self.assertNotIn("context window may be full", str(error.exception))

    def test_unloaded_model_is_explicitly_loaded_and_used_for_inference(self):
        agent = LMStudio({"name": "qwen/vision", "max_tokens": 2000, "context_length": 32768})
        with patch.object(agent, "_native", side_effect=[catalog(), loaded()]) as native:
            agent.prepare()
        self.assertEqual(native.call_args_list[1].args,
                         ("/load", {"model": "qwen/vision", "context_length": 32768, "echo_load_config": True}))
        self.assertEqual(agent.inference_model, "vision-instance")
        response = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
        with patch("pokemon_harness.agent.lmstudio.urlopen") as request:
            request.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            agent.decide([])
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["model"], "vision-instance")
        self.assertEqual(payload["max_tokens"], 2000)

    def test_sufficient_loaded_context_is_reused_without_reload(self):
        instances = [{"id": "existing", "config": {"context_length": 65536}}]
        agent = LMStudio({"name": "qwen/vision"})
        with patch.object(agent, "_native", return_value=catalog(instances)) as native:
            agent.prepare()
        self.assertEqual(native.call_count, 1)
        self.assertEqual(agent.inference_model, "existing")

    def test_undersized_instance_is_unloaded_before_loading_requested_context(self):
        agent = LMStudio({"name": "qwen/vision"})
        instances = [{"id": "small", "config": {"context_length": 8192}}]
        with patch.object(agent, "_native", side_effect=[catalog(instances), {}, loaded()]) as native:
            agent.prepare()
        self.assertEqual(native.call_args_list[1].args, ("/unload", {"instance_id": "small"}))
        self.assertEqual(agent.inference_model, "vision-instance")

    def test_legacy_short_model_name_resolves_to_unique_full_key(self):
        agent = LMStudio({"name": "vision"})
        with patch.object(agent, "_native", side_effect=[catalog(), loaded()]) as native:
            agent.prepare()
        self.assertEqual(native.call_args_list[1].args[1]["model"], "qwen/vision")

    def test_load_reusing_small_context_is_retried_after_explicit_unload(self):
        agent = LMStudio({"name": "qwen/vision"})
        with patch.object(agent, "_native", side_effect=[catalog(), loaded(8192), {}, loaded()]) as native:
            agent.prepare()
        self.assertEqual(native.call_args_list[2].args, ("/unload", {"instance_id": "vision-instance"}))
        self.assertEqual(agent.inference_model, "vision-instance")

    def test_ignored_context_setting_stops_instead_of_using_server_default(self):
        agent = LMStudio({"name": "qwen/vision"})
        with patch.object(agent, "_native", side_effect=[catalog(), loaded(8192), {}, loaded(8192)]):
            with self.assertRaisesRegex(ValueError, "reported 8192.*Stopping before game input"):
                agent.prepare()

    def test_context_beyond_model_limit_is_rejected_before_load(self):
        agent = LMStudio({"name": "qwen/vision", "context_length": 262144})
        with patch.object(agent, "_native", return_value=catalog()) as native:
            with self.assertRaisesRegex(ValueError, "exceeds model maximum"):
                agent.prepare()
        self.assertEqual(native.call_count, 1)

    def test_native_load_uses_same_server_and_authentication(self):
        agent = LMStudio({"name": "qwen/vision", "base_url": "http://localhost:1234/v1",
                          "api_key_env": "HARNESS_TEST_KEY"})
        with patch.dict("os.environ", {"HARNESS_TEST_KEY": "test-secret"}), \
                patch("pokemon_harness.agent.lmstudio.urlopen") as request:
            request.return_value.__enter__.return_value.read.return_value = b'{}'
            agent._native("/load", {"context_length": 32768})
        sent = request.call_args.args[0]
        self.assertEqual(sent.full_url, "http://localhost:1234/api/v1/models/load")
        self.assertEqual(sent.get_header("Authorization"), "Bearer test-secret")
