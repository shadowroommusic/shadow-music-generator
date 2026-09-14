from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from shadow_music_generator import mcp_server
from shadow_music_generator.jobs import GenerationRequest, JobStore, model_license_for
from shadow_music_generator.pipeline import (
    PipelineContractError,
    PipelineNotConfigured,
    load_object,
    load_pipeline,
    resolve_spec,
    run_stages,
)

SRC = Path(__file__).resolve().parents[1] / "src"

FAKE_YUE2 = """
class FakePlan:
    def __init__(self, request):
        self.request = request
        self.saved = None

    def save(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        self.saved = path


class FakePipeline:
    instances = []

    def __init__(self):
        self.calls = []
        self.request = None

    @classmethod
    def from_pretrained(cls, model, device=None, vae=None):
        instance = cls()
        instance.model = model
        instance.device = device
        instance.vae = vae
        cls.instances.append(instance)
        return instance

    def __enter__(self):
        self.calls.append("enter")
        return self

    def __exit__(self, *exc):
        self.calls.append("exit")

    def plan(self, **request):
        self.calls.append("plan")
        self.request = request
        return FakePlan(request)

    def generate_semantic(self, plan):
        self.calls.append("generate_semantic")
        return {"tokens": [1, 2, 3]}

    def synthesize(self, semantic):
        self.calls.append("synthesize")
        return {"latents": True}

    def decode(self, latents):
        self.calls.append("decode")
        return [[0.25, -0.25]] * 16
"""

FAKE_PIPELINE = """
import os


class FakePipeline:
    def plan(self, context):
        return {"plan": {"sections": [{"bars": 8}]}}

    def generate_semantic(self, context):
        return {"tokens": [1, 2, 3]}

    def synthesize(self, context):
        return {"vocoder_output": "vocoder.wav"}

    def decode(self, context):
        path = os.path.join(context["output_dir"], "take-1.wav")
        with open(path, "wb") as handle:
            handle.write(b"RIFF")
        context["artifacts"].append(path)
        return {"artifacts": list(context["artifacts"])}


def make_pipeline():
    return FakePipeline()
"""

BROKEN_PIPELINE = """
class Broken:
    def plan(self, context):
        return {"plan": "ok"}

    def generate_semantic(self, context):
        return {"semantic": "ok"}

    def synthesize(self, context):
        raise RuntimeError("no vocoder weights")

    def decode(self, context):
        return {}


def make_pipeline():
    return Broken()
"""

INCOMPLETE_PIPELINE = """
class Incomplete:
    def plan(self, context):
        return {}

    def generate_semantic(self, context):
        return {}

    def synthesize(self, context):
        return {}


def make_pipeline():
    return Incomplete()
"""


class PipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="shadow-generator-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.store = JobStore(self.root / "jobs")

    def write_adapter(self, name: str, source: str) -> Path:
        path = self.root / name
        path.write_text(source, encoding="utf-8")
        return path

    def module_env(self, spec: str):
        return mock.patch.dict(os.environ, {"SHADOW_PIPELINE_FACTORY": spec}), mock.patch.object(
            sys, "path", [str(self.root), *sys.path]
        )


class DryRunTests(PipelineTestCase):
    def test_dry_run_completes_without_any_adapter(self) -> None:
        job = self.store.submit(GenerationRequest("dark melodic techno, 128 bpm"))
        result = self.store.run(job.id)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.pipeline, None)
        self.assertEqual([stage.name for stage in result.stages], ["dry-run"])
        self.assertEqual(result.outputs, ())
        self.assertIn("dry-run", result.output or "")
        self.assertIsNone(result.error)

    def test_invalid_requests_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.submit(GenerationRequest("   "))
        with self.assertRaises(ValueError):
            self.store.submit(GenerationRequest("ok", mode="turbo"))

    def test_missing_job_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.store.get("does-not-exist")

    def test_model_license_tracks_the_checkpoint(self) -> None:
        self.assertIn("CC-BY-NC-4.0", model_license_for("YuE2-7B"))
        self.assertIn("Apache-2.0", model_license_for("yue-stage1"))
        self.assertIn("unknown", model_license_for(None))


class LocalModeTests(PipelineTestCase):
    def test_local_mode_without_a_factory_reports_a_clear_error(self) -> None:
        job = self.store.submit(GenerationRequest("techno", mode="local"))
        with mock.patch.dict(os.environ, {}, clear=True):
            result = self.store.run(job.id)
        self.assertEqual(result.status, "failed")
        self.assertIn("SHADOW_PIPELINE_FACTORY", result.error or "")
        self.assertEqual(result.stages, ())

    def test_module_adapter_runs_every_stage_and_records_outputs(self) -> None:
        self.write_adapter("fake_pipeline.py", FAKE_PIPELINE)
        job = self.store.submit(GenerationRequest("acid house", mode="local"))
        env_patch, path_patch = self.module_env("fake_pipeline:make_pipeline")
        with env_patch, path_patch:
            result = self.store.run(job.id)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.pipeline, "fake_pipeline:make_pipeline")
        self.assertEqual([stage.name for stage in result.stages], ["plan", "generate_semantic", "synthesize", "decode"])
        self.assertTrue(all(stage.status == "completed" for stage in result.stages))
        self.assertTrue(all(stage.duration_ms >= 0 for stage in result.stages))
        self.assertEqual(len(result.outputs), 1)
        self.assertTrue(result.outputs[0].endswith("take-1.wav"))
        self.assertTrue(Path(result.outputs[0]).is_file())
        self.assertEqual(result.output, result.outputs[0])

    def test_file_path_adapter_spec_is_supported(self) -> None:
        adapter = self.write_adapter("standalone_adapter.py", FAKE_PIPELINE)
        job = self.store.submit(GenerationRequest("ambient", mode="local"))
        with mock.patch.dict(os.environ, {}, clear=True):
            result = self.store.run(job.id, spec=f"{adapter}:make_pipeline")
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.outputs), 1)

    def test_missing_stage_fails_the_contract_clearly(self) -> None:
        self.write_adapter("incomplete_pipeline.py", INCOMPLETE_PIPELINE)
        job = self.store.submit(GenerationRequest("techno", mode="local"))
        env_patch, path_patch = self.module_env("incomplete_pipeline:make_pipeline")
        with env_patch, path_patch:
            result = self.store.run(job.id)
        self.assertEqual(result.status, "failed")
        self.assertIn("decode", result.error or "")
        self.assertEqual(result.stages, ())

    def test_stage_failure_keeps_the_progress_it_made(self) -> None:
        self.write_adapter("broken_pipeline.py", BROKEN_PIPELINE)
        job = self.store.submit(GenerationRequest("techno", mode="local"))
        env_patch, path_patch = self.module_env("broken_pipeline:make_pipeline")
        with env_patch, path_patch:
            result = self.store.run(job.id)
        self.assertEqual(result.status, "failed")
        self.assertIn("synthesize", result.error or "")
        self.assertIn("no vocoder weights", result.error or "")
        self.assertEqual([stage.name for stage in result.stages], ["plan", "generate_semantic", "synthesize"])
        self.assertEqual([stage.status for stage in result.stages], ["completed", "completed", "failed"])

    def test_jobs_persist_across_store_instances(self) -> None:
        self.write_adapter("fake_pipeline.py", FAKE_PIPELINE)
        job = self.store.submit(GenerationRequest("techno", mode="local"))
        env_patch, path_patch = self.module_env("fake_pipeline:make_pipeline")
        with env_patch, path_patch:
            self.store.run(job.id)
        reloaded = JobStore(self.root / "jobs").get(job.id)
        self.assertEqual(reloaded.status, "completed")
        self.assertEqual([stage.name for stage in reloaded.stages], ["plan", "generate_semantic", "synthesize", "decode"])
        self.assertEqual(len(reloaded.outputs), 1)
        self.assertTrue(Path(reloaded.outputs[0]).is_file())

    def test_job_directory_can_come_from_the_environment(self) -> None:
        target = self.root / "env-jobs"
        with mock.patch.dict(os.environ, {"SHADOW_JOB_DIR": str(target)}):
            store = JobStore()
            self.assertEqual(store.root, target)
            job = store.submit(GenerationRequest("techno"))
        self.assertTrue((target / f"{job.id}.json").is_file())
        self.assertTrue(store.jobs())


class PipelineLoadingTests(PipelineTestCase):
    def test_resolve_spec_prefers_the_argument(self) -> None:
        self.assertEqual(resolve_spec("explicit:factory", {"SHADOW_PIPELINE_FACTORY": "env:factory"}), "explicit:factory")
        self.assertEqual(resolve_spec(None, {"SHADOW_PIPELINE_FACTORY": " env:factory "}), "env:factory")
        self.assertIsNone(resolve_spec("  ", {}))

    def test_missing_configuration_raises_with_guidance(self) -> None:
        with self.assertRaises(PipelineNotConfigured) as caught:
            load_pipeline(None, {})
        self.assertIn("SHADOW_PIPELINE_FACTORY", str(caught.exception))

    def test_invalid_specs_raise_contract_errors(self) -> None:
        with self.assertRaises(PipelineContractError):
            load_object("just_a_module")
        with self.assertRaises(PipelineContractError):
            load_object("no_such_module:make_pipeline")
        with self.assertRaises(PipelineContractError):
            load_object("json:not_a_real_attribute")

    def test_dotted_spec_form_is_accepted(self) -> None:
        self.write_adapter("fake_pipeline.py", FAKE_PIPELINE)
        with mock.patch.object(sys, "path", [str(self.root), *sys.path]):
            pipeline, spec = load_pipeline("fake_pipeline.make_pipeline")
        self.assertEqual(spec, "fake_pipeline.make_pipeline")
        self.assertTrue(callable(pipeline.decode))


class InterfaceTests(PipelineTestCase):
    def test_ping_and_negotiation_methods_return_empty_results(self) -> None:
        for method, expected in (
            ("ping", {}),
            ("resources/list", {"resources": []}),
            ("resources/templates/list", {"resourceTemplates": []}),
            ("prompts/list", {"prompts": []}),
            ("logging/setLevel", {}),
        ):
            reply = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": method, "params": {}})
            self.assertEqual(reply["result"], expected, method)
            self.assertNotIn("error", reply)

    def test_tool_failure_is_reported_with_is_error(self) -> None:
        reply = mcp_server.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "job_status", "arguments": {"job_id": "nope", "job_dir": str(self.root / "mcp-jobs")}},
            }
        )
        self.assertTrue(reply["result"]["isError"])
        self.assertIn("no such job", reply["result"]["content"][0]["text"])

    def test_mcp_server_submits_and_reports_a_dry_run(self) -> None:
        reply = mcp_server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "submit_generation",
                    "arguments": {"prompt": "melodic techno", "job_dir": str(self.root / "mcp-jobs")},
                },
            }
        )
        self.assertIsNotNone(reply)
        payload = json.loads(reply["result"]["content"][0]["text"])
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["mode"], "dry-run")

    def test_mcp_server_lists_three_tools(self) -> None:
        reply = mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        self.assertEqual(
            [tool["name"] for tool in reply["result"]["tools"]],
            [
                "submit_generation",
                "run_job",
                "job_status",
                "render_part",
                "render_song",
                "mix_arrangement",
            ],
        )

    def test_cli_dry_run_and_failed_local_run(self) -> None:
        job_dir = self.root / "cli-jobs"
        environment = {**os.environ, "PYTHONPATH": str(SRC), "SHADOW_JOB_DIR": ""}
        completed = subprocess.run(
            [sys.executable, "-m", "shadow_music_generator.cli", "submit", "--prompt", "techno", "--job-dir", str(job_dir)],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["model_license"], "unknown; confirm the checkpoint license before any commercial use")

        failed = subprocess.run(
            [
                sys.executable,
                "-m",
                "shadow_music_generator.cli",
                "submit",
                "--prompt",
                "techno",
                "--mode",
                "local",
                "--run",
                "--job-dir",
                str(job_dir),
            ],
            capture_output=True,
            text=True,
            env={**environment, "SHADOW_PIPELINE_FACTORY": ""},
            check=False,
        )
        self.assertEqual(failed.returncode, 1)
        failure = json.loads(failed.stdout)
        self.assertEqual(failure["status"], "failed")
        self.assertIn("SHADOW_PIPELINE_FACTORY", failure["error"])


class YuE2AdapterTests(unittest.TestCase):
    """The YuE2 adapter maps our four stages 1:1 onto YuE2's staged API — and never touches audio."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.request = GenerationRequest(prompt="dark techno, 128 bpm", mode="local", output_dir=str(self.root / "out"))

    def _fake_yue2(self):
        module = types.ModuleType("yue2")
        namespace: "dict[str, object]" = {}
        exec(FAKE_YUE2, namespace)  # noqa: S102 - test fixture
        module.YuE2Pipeline = namespace["FakePipeline"]
        module.FakePipeline = namespace["FakePipeline"]
        return module

    def test_pipeline_mode_runs_yue2_stages_in_order(self) -> None:
        from shadow_music_generator.adapters import yue2_adapter

        fake = self._fake_yue2()
        with mock.patch.dict(sys.modules, {"yue2": fake}):
            pipeline = yue2_adapter.YuE2Adapter(model="fake/model", device="cpu")
            stages, outputs = run_stages(pipeline, self.request, self.root / "out")

        self.assertEqual([stage.name for stage in stages], ["plan", "generate_semantic", "synthesize", "decode"])
        self.assertEqual([stage.status for stage in stages], ["completed"] * 4)
        instance = fake.FakePipeline.instances[-1]
        self.assertEqual(instance.model, "fake/model")
        self.assertEqual(instance.device, "cpu")
        self.assertEqual(instance.calls, ["enter", "plan", "generate_semantic", "synthesize", "decode", "exit"])
        self.assertEqual(instance.request["style"], "dark techno, 128 bpm")
        self.assertEqual(instance.request["cot"], "full")
        self.assertTrue(outputs and Path(outputs[0]).exists())
        self.assertEqual(Path(outputs[0]).name, "audio.wav")  # no soundfile in the test env → stdlib WAV fallback
        self.assertTrue((self.root / "out" / "request.json").is_file())

    def test_command_mode_collects_outputs_without_modifying_them(self) -> None:
        from shadow_music_generator.adapters import yue2_adapter

        script = self.root / "fake_yue.py"
        script.write_text(
            "import hashlib, pathlib, sys\n"
            "out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
            "target = out / 'song.flac'\n"
            "payload = b'fLaC' + b'\\x00' * 64\n"
            "target.write_bytes(payload)\n"
            "(out / 'song.sha256').write_text(hashlib.sha256(payload).hexdigest())\n",
            encoding="utf-8",
        )
        command = f"{sys.executable} {script} {{output_dir}}"
        with mock.patch.dict(os.environ, {"YUE_COMMAND": command}):
            pipeline = yue2_adapter.YuE2Adapter()
            stages, outputs = run_stages(pipeline, self.request, self.root / "out2")

        self.assertEqual([stage.status for stage in stages], ["completed"] * 4)
        self.assertEqual(len(outputs), 1)
        produced = Path(outputs[0])
        recorded = (self.root / "out2" / "song.sha256").read_text()
        self.assertEqual(hashlib.sha256(produced.read_bytes()).hexdigest(), recorded)

    def test_command_mode_reports_missing_outputs(self) -> None:
        from shadow_music_generator.adapters import yue2_adapter
        from shadow_music_generator.pipeline import PipelineStageError

        with mock.patch.dict(os.environ, {"YUE_COMMAND": f"{sys.executable} -c pass {{output_dir}}"}):
            pipeline = yue2_adapter.YuE2Adapter()
            with self.assertRaises(PipelineStageError) as caught:
                run_stages(pipeline, self.request, self.root / "out3")
        self.assertEqual(caught.exception.stage, "synthesize")
        self.assertIn("no audio", str(caught.exception))

    def test_command_mode_reports_failing_command(self) -> None:
        from shadow_music_generator.adapters import yue2_adapter
        from shadow_music_generator.pipeline import PipelineStageError

        with mock.patch.dict(os.environ, {"YUE_COMMAND": f"{sys.executable} -c 'import sys; sys.exit(3)' {{output_dir}}"}):
            pipeline = yue2_adapter.YuE2Adapter()
            with self.assertRaises(PipelineStageError) as caught:
                run_stages(pipeline, self.request, self.root / "out4")
        self.assertEqual(caught.exception.stage, "generate_semantic")
        self.assertIn("YUE_COMMAND failed", str(caught.exception))

    def test_missing_yue2_environment_is_explained(self) -> None:
        from shadow_music_generator.adapters import yue2_adapter
        from shadow_music_generator.pipeline import PipelineStageError

        with mock.patch.dict(sys.modules, {"yue2": None}), mock.patch.dict(os.environ, {"YUE_COMMAND": ""}):
            pipeline = yue2_adapter.YuE2Adapter()
            with self.assertRaises(PipelineStageError) as caught:
                run_stages(pipeline, self.request, self.root / "out5")
        self.assertIn("yue2", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
