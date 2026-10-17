from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.ai.music_model_backends import ACEStep15Backend, ACEStepBackend, AudioXBackend, HeartMuLaBackend, MelodyFlowBackend, MiniMaxMusicBackend, MusicGenRequest, YuE2Backend, get_backend_registry
from tools.ai.run_minimax_backend import compose_caption, compose_lyrics, require_minimax_model_dir
from tools.ai.run_yue2_backend import compose_lyrics as compose_yue2_lyrics
from tools.ai.run_yue2_backend import compose_style, require_yue2_model_dir, require_yue2_vae_dir, resolve_yue2_dirs


class MusicModelBackendsTests(unittest.TestCase):
    def test_audiox_backend_reports_missing_modules(self) -> None:
        with patch.dict(os.environ, {"AUDIOX_PYTHON": "C:/fake/python.exe"}, clear=False):
            with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=["audiox", "einops", "soundfile"]):
                    result = AudioXBackend().run(MusicGenRequest(prompt="test", output_dir=".tmp/test-audiox"))
        self.assertFalse(result.success)
        self.assertIn("audiox, einops, soundfile", result.error)

    def test_audiox_backend_uses_configured_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            request = MusicGenRequest(prompt="test", output_dir=str(output_dir), duration_seconds=1.0)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "audiox_output.wav").write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            with patch.dict(os.environ, {"AUDIOX_PYTHON": "C:/custom/python.exe"}, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = AudioXBackend().run(request)

        self.assertTrue(result.success)
        self.assertEqual("C:/custom/python.exe", recorded["command"][0])

    def test_heartmula_backend_uses_configured_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            heart_root = root / "heart"
            ckpt_dir = root / "ckpt"
            heart_root.mkdir()
            ckpt_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(prompt="test", output_dir=str(output_dir), duration_seconds=1.0)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "heartmula_output.wav").write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            env = {
                "HEARTMULA_ROOT": str(heart_root),
                "HEARTMULA_CKPT_DIR": str(ckpt_dir),
                "HEARTMULA_PYTHON": "C:/heart/python.exe",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                        result = HeartMuLaBackend().run(request)

        self.assertTrue(result.success)
        self.assertEqual("C:/heart/python.exe", recorded["command"][0])
        self.assertTrue(recorded["command"][1].endswith("run_heartmula_backend.py"))
        self.assertIn("--cfg-scale", recorded["command"])
        self.assertIn("1.5", recorded["command"])

    def test_heartmula_variant_backend_uses_variant_checkpoint_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            heart_root = root / "heart"
            base_ckpt_dir = root / "ckpt-base"
            heart_root.mkdir()
            base_ckpt_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(prompt="test", output_dir=str(output_dir), duration_seconds=1.0)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "heartmula_base_output.wav").write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            env = {
                "HEARTMULA_ROOT": str(heart_root),
                "HEARTMULA_BASE_CKPT_DIR": str(base_ckpt_dir),
                "HEARTMULA_PYTHON": "C:/heart/python.exe",
            }
            backend = HeartMuLaBackend(name="heartmula_base", checkpoint_env="HEARTMULA_BASE_CKPT_DIR")
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                        result = backend.run(request)

        self.assertTrue(result.success)
        self.assertIn(str(base_ckpt_dir), recorded["command"])

    def test_heartmula_backend_reads_latency_env_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            heart_root = root / "heart"
            ckpt_dir = root / "ckpt"
            heart_root.mkdir()
            ckpt_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(prompt="test", output_dir=str(output_dir), duration_seconds=1.0)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "heartmula_output.wav").write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            env = {
                "HEARTMULA_ROOT": str(heart_root),
                "HEARTMULA_CKPT_DIR": str(ckpt_dir),
                "HEARTMULA_PYTHON": "C:/heart/python.exe",
                "HEARTMULA_CFG_SCALE": "1.0",
                "HEARTMULA_LAZY_LOAD": "false",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                        result = HeartMuLaBackend().run(request)

        self.assertTrue(result.success)
        cfg_scale_index = recorded["command"].index("--cfg-scale") + 1
        self.assertEqual("1.0", recorded["command"][cfg_scale_index])
        lazy_load_index = recorded["command"].index("--lazy-load") + 1
        self.assertEqual("false", recorded["command"][lazy_load_index])

    def test_heartmula_backend_writes_descriptor_and_lyrics_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            heart_root = root / "heart"
            ckpt_dir = root / "ckpt"
            heart_root.mkdir()
            ckpt_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(
                prompt="night drive",
                output_dir=str(output_dir),
                duration_seconds=1.0,
                tags="synthwave, neon bass",
                lyrics=" line one\nline two\n",
            )
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                tags_path = Path(command[command.index("--tags") + 1])
                lyrics_path = Path(command[command.index("--lyrics") + 1])
                recorded["tags_text"] = tags_path.read_text(encoding="utf-8")
                recorded["lyrics_text"] = lyrics_path.read_text(encoding="utf-8")
                (output_dir / "heartmula_output.wav").write_bytes(b"RIFFtest")

                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return Completed()

            env = {
                "HEARTMULA_ROOT": str(heart_root),
                "HEARTMULA_CKPT_DIR": str(ckpt_dir),
                "HEARTMULA_PYTHON": "C:/heart/python.exe",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                        result = HeartMuLaBackend().run(request)

        self.assertTrue(result.success)
        self.assertEqual("night drive. synthwave, neon bass\n", recorded["tags_text"])
        self.assertEqual("line one\nline two", recorded["lyrics_text"])
        self.assertEqual("night drive. synthwave, neon bass", result.metadata["effective_descriptor_text"])
        self.assertEqual("synthwave, neon bass", result.metadata["tags"])
        self.assertEqual(" line one\nline two\n", result.metadata["lyrics"])

    def test_acestep_backend_uses_configured_python_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ckpt_dir = root / "ace-step"
            ckpt_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(prompt="test", output_dir=str(output_dir), duration_seconds=1.0)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "ace_step_output.wav").write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            env = {
                "ACESTEP_CKPT_DIR": str(ckpt_dir),
                "ACESTEP_PYTHON": "C:/ace/python.exe",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = ACEStepBackend().run(request)

        self.assertTrue(result.success)
        self.assertEqual("C:/ace/python.exe", recorded["command"][0])
        self.assertTrue(recorded["command"][1].endswith("run_acestep_backend.py"))
        self.assertIn(str(ckpt_dir.resolve()), recorded["command"])

    def test_acestep15_turbo_backend_passes_tags_lyrics_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "ace-step-1.5"
            checkpoints_dir = root / "checkpoints"
            lm_dir = checkpoints_dir / "acestep-5Hz-lm-1.7B"
            source_root.mkdir()
            checkpoints_dir.mkdir()
            lm_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(
                prompt="cinematic rise",
                output_dir=str(output_dir),
                duration_seconds=2.0,
                tags="orchestral, trailer",
                lyrics="hold the line",
                seed=1234,
            )
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "ace_step_v15_turbo_output.wav").write_bytes(b"RIFFtest")

                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return Completed()

            env = {
                "ACESTEP15_ROOT": str(source_root),
                "ACESTEP15_CKPT_DIR": str(checkpoints_dir),
                "ACESTEP15_PYTHON": "C:/ace15/python.exe",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = ACEStep15Backend(name="ace_step_v15_turbo", output_stem="ace_step_v15_turbo").run(request)

        self.assertTrue(result.success)
        self.assertEqual("C:/ace15/python.exe", recorded["command"][0])
        self.assertTrue(recorded["command"][1].endswith("run_acestep15_backend.py"))
        self.assertIn("cinematic rise", recorded["command"])
        self.assertIn("orchestral, trailer", recorded["command"])
        self.assertIn("hold the line", recorded["command"])
        self.assertIn("1234", recorded["command"])
        self.assertIn("--init-llm", recorded["command"])
        self.assertIn("true", recorded["command"])
        self.assertIn("--inference-steps", recorded["command"])
        self.assertIn("--guidance-scale", recorded["command"])
        self.assertEqual("orchestral, trailer", result.metadata["tags"])
        self.assertEqual("hold the line", result.metadata["lyrics"])
        self.assertTrue(result.metadata["init_llm"])
        self.assertEqual(8, result.metadata["inference_steps"])
        self.assertEqual(1.0, result.metadata["guidance_scale"])
        self.assertEqual("turbo", result.metadata["model_kind"])

    def test_acestep15_sft_backend_uses_cover_mode_with_reference_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "ace-step-1.5"
            checkpoints_dir = root / "checkpoints"
            reference_audio = root / "input.wav"
            source_root.mkdir()
            checkpoints_dir.mkdir()
            reference_audio.write_bytes(b"RIFFtest")
            output_dir = root / "out"
            request = MusicGenRequest(
                prompt="lush strings",
                output_dir=str(output_dir),
                duration_seconds=2.0,
                reference_audio=str(reference_audio),
            )
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "ace_step_v15_sft_output.wav").write_bytes(b"RIFFtest")

                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return Completed()

            env = {
                "ACESTEP15_ROOT": str(source_root),
                "ACESTEP15_CKPT_DIR": str(checkpoints_dir),
                "ACESTEP15_PYTHON": "C:/ace15/python.exe",
                "ACESTEP15_TASK_TYPE": "auto",
                "ACESTEP15_SFT_CONFIG_PATH": "acestep-v15-sft",
                "ACESTEP15_COVER_STRENGTH": "0.35",
            }
            backend = ACEStep15Backend(
                name="ace_step_v15_sft",
                output_stem="ace_step_v15_sft",
                config_env="ACESTEP15_SFT_CONFIG_PATH",
                default_config_path="acestep-v15-sft",
            )
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = backend.run(request)

        self.assertTrue(result.success)
        self.assertIn("--task-type", recorded["command"])
        self.assertIn("cover", recorded["command"])
        self.assertIn("--source-audio", recorded["command"])
        self.assertIn(str(reference_audio), recorded["command"])
        self.assertIn("--audio-cover-strength", recorded["command"])
        self.assertEqual("cover", result.metadata["task_type"])

    def test_acestep15_sft_backend_defaults_to_50_steps_without_auto_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "ace-step-1.5"
            checkpoints_dir = root / "checkpoints"
            source_root.mkdir()
            checkpoints_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(prompt="detailed chamber pop", output_dir=str(output_dir), duration_seconds=2.0)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "ace_step_v15_sft_output.wav").write_bytes(b"RIFFtest")

                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return Completed()

            env = {
                "ACESTEP15_ROOT": str(source_root),
                "ACESTEP15_CKPT_DIR": str(checkpoints_dir),
                "ACESTEP15_PYTHON": "C:/ace15/python.exe",
                "ACESTEP15_SFT_CONFIG_PATH": "acestep-v15-sft",
            }
            backend = ACEStep15Backend(
                name="ace_step_v15_sft",
                output_stem="ace_step_v15_sft",
                config_env="ACESTEP15_SFT_CONFIG_PATH",
                default_config_path="acestep-v15-sft",
            )
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = backend.run(request)

        self.assertTrue(result.success)
        self.assertIn("--inference-steps", recorded["command"])
        steps_value = recorded["command"][recorded["command"].index("--inference-steps") + 1]
        guidance_value = recorded["command"][recorded["command"].index("--guidance-scale") + 1]
        init_llm_value = recorded["command"][recorded["command"].index("--init-llm") + 1]
        self.assertEqual("50", steps_value)
        self.assertEqual("7.0", guidance_value)
        self.assertEqual("false", init_llm_value)
        self.assertEqual("sft", result.metadata["model_kind"])
        self.assertEqual(50, result.metadata["inference_steps"])
        self.assertEqual(7.0, result.metadata["guidance_scale"])
        self.assertFalse(result.metadata["init_llm"])

    def test_acestep15_sft_backend_auto_enables_lm_when_shared_lm_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "ace-step-1.5"
            checkpoints_dir = root / "checkpoints"
            lm_dir = checkpoints_dir / "acestep-5Hz-lm-1.7B"
            source_root.mkdir()
            checkpoints_dir.mkdir()
            lm_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(prompt="detailed chamber pop", output_dir=str(output_dir), duration_seconds=2.0)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "ace_step_v15_sft_output.wav").write_bytes(b"RIFFtest")

                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return Completed()

            env = {
                "ACESTEP15_ROOT": str(source_root),
                "ACESTEP15_CKPT_DIR": str(checkpoints_dir),
                "ACESTEP15_PYTHON": "C:/ace15/python.exe",
                "ACESTEP15_SFT_CONFIG_PATH": "acestep-v15-sft",
            }
            backend = ACEStep15Backend(
                name="ace_step_v15_sft",
                output_stem="ace_step_v15_sft",
                config_env="ACESTEP15_SFT_CONFIG_PATH",
                default_config_path="acestep-v15-sft",
            )
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = backend.run(request)

        self.assertTrue(result.success)
        init_llm_value = recorded["command"][recorded["command"].index("--init-llm") + 1]
        self.assertEqual("true", init_llm_value)
        self.assertTrue(result.metadata["init_llm"])
        self.assertEqual("acestep-5Hz-lm-1.7B", result.metadata["lm_model_path"])

    def test_melodyflow_backend_uses_configured_python_and_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "melodyflow"
            model_dir.mkdir()
            output_dir = root / "out"
            request = MusicGenRequest(prompt="test", output_dir=str(output_dir), duration_seconds=1.0, seed=7)
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "melodyflow_output.wav").write_bytes(b"RIFFtest")
                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""
                return Completed()

            env = {
                "MELODYFLOW_MODEL_DIR": str(model_dir),
                "MELODYFLOW_PYTHON": "C:/melody/python.exe",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = MelodyFlowBackend().run(request)

        self.assertTrue(result.success)
        self.assertEqual("C:/melody/python.exe", recorded["command"][0])
        self.assertTrue(recorded["command"][1].endswith("run_melodyflow_backend.py"))
        self.assertIn(str(model_dir.resolve()), recorded["command"])
        self.assertIn("--seed", recorded["command"])

    def test_registry_includes_minimax_music_3(self) -> None:
        self.assertIn("minimax_music_3", get_backend_registry())

    def test_minimax_backend_reports_missing_python(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MINIMAX_MODEL_DIR": "C:/missing/minimax-music-3",
                "MINIMAX_PYTHON": "C:/missing/python.exe",
            },
            clear=False,
        ):
            with patch("tools.ai.music_model_backends.python_exists", return_value=False):
                result = MiniMaxMusicBackend().run(MusicGenRequest(prompt="test", output_dir=".tmp/test-minimax"))
        self.assertFalse(result.success)
        self.assertIn("MINIMAX_PYTHON", result.error)

    def test_minimax_backend_launches_diffusers_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "minimax-music-3"
            model_dir.mkdir()
            (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
            output_dir = root / "out"
            request = MusicGenRequest(
                prompt="late night r&b",
                output_dir=str(output_dir),
                duration_seconds=30.0,
                tags="autotune, 808",
                lyrics="[verse]\nhold the line",
                seed=9,
            )
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "minimax_music_3_output.wav").write_bytes(b"RIFFtest")

                class Completed:
                    returncode = 0
                    stdout = '{"generated_seconds": 28.4, "backend": "diffusers"}'
                    stderr = ""

                return Completed()

            env = {
                "MINIMAX_MODEL_DIR": str(model_dir),
                "MINIMAX_PYTHON": "C:/minimax/python.exe",
                "MINIMAX_CPU_OFFLOAD": "true",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = MiniMaxMusicBackend().run(request)

        self.assertTrue(result.success)
        self.assertEqual("C:/minimax/python.exe", recorded["command"][0])
        self.assertTrue(recorded["command"][1].endswith("run_minimax_backend.py"))
        self.assertIn("late night r&b", recorded["command"])
        self.assertIn("autotune, 808", recorded["command"])
        self.assertIn("[verse]\nhold the line", recorded["command"])
        self.assertIn(str(model_dir.resolve()), recorded["command"])
        self.assertIn("--cpu-offload", recorded["command"])
        self.assertNotIn("--comfy-root", recorded["command"])
        self.assertTrue(result.metadata["duration_honored"])
        self.assertEqual(30.0, result.metadata["requested_duration_seconds"])
        self.assertEqual(28.4, result.metadata["generated_seconds"])

    def test_compose_caption_merges_prompt_and_tags(self) -> None:
        self.assertEqual("dark r&b. autotune, 808", compose_caption("dark r&b", "autotune, 808"))

    def test_compose_lyrics_uses_instrumental_placeholder(self) -> None:
        self.assertEqual("[Instrumental]", compose_lyrics("", instrumental=True))
        self.assertEqual("", compose_lyrics("", instrumental=False))
        self.assertEqual("[verse]\nhold", compose_lyrics("[verse]\nhold", instrumental=True))

    def test_require_minimax_model_dir_needs_model_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            with self.assertRaises(FileNotFoundError):
                require_minimax_model_dir(model_dir)
            (model_dir / "modular_model_index.json").write_text("{}", encoding="utf-8")
            self.assertEqual(model_dir.resolve(), require_minimax_model_dir(model_dir))

    def test_remap_local_component_paths_rewrites_hub_id(self) -> None:
        from tools.ai.run_minimax_backend import remap_local_component_paths

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            index = {
                "transformer": [
                    "diffusers",
                    "MiniMaxMusic3Transformer1DModel",
                    {"pretrained_model_name_or_path": "MiniMaxAI/MiniMax-Music3", "subfolder": "transformer"},
                ]
            }
            (model_dir / "modular_model_index.json").write_text(json.dumps(index), encoding="utf-8")
            remap_local_component_paths(model_dir)
            payload = json.loads((model_dir / "modular_model_index.json").read_text(encoding="utf-8"))
            self.assertEqual(str(model_dir.resolve()), payload["transformer"][2]["pretrained_model_name_or_path"])

    def test_registry_includes_yue2(self) -> None:
        self.assertIn("yue2", get_backend_registry())

    def test_yue2_backend_reports_missing_python(self) -> None:
        with patch.dict(
            os.environ,
            {
                "YUE2_MODEL_DIR": "C:/missing/YuE2-3B",
                "YUE2_VAE_DIR": "C:/missing/YuE2-Vae",
                "YUE2_PYTHON": "C:/missing/python.exe",
                "YUE2_COT": "full",
                "YUE2_ABC_PATH": "",
            },
            clear=False,
        ):
            with patch("tools.ai.music_model_backends.python_exists", return_value=False):
                result = YuE2Backend().run(MusicGenRequest(prompt="piano pop", output_dir=".tmp/test-yue2"))
        self.assertFalse(result.success)
        self.assertIn("YUE2_PYTHON", result.error or "")

    def test_yue2_backend_rejects_abc_when_planning_is_off(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            abc_path = Path(temp_dir) / "score.abc"
            abc_path.write_text("X:1\n", encoding="utf-8")
            with patch.dict(os.environ, {"YUE2_COT": "off", "YUE2_ABC_PATH": str(abc_path)}, clear=False):
                result = YuE2Backend().run(MusicGenRequest(prompt="piano pop", output_dir=temp_dir))
        self.assertFalse(result.success)
        self.assertIn("cot=full", result.error or "")

    def test_yue2_backend_launches_official_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "YuE2-3B"
            vae_dir = root / "YuE2-Vae"
            model_dir.mkdir()
            vae_dir.mkdir()
            (model_dir / "qwen.tiktoken").write_bytes(b"token")
            output_dir = root / "out"
            request = MusicGenRequest(
                prompt="warm piano pop",
                output_dir=str(output_dir),
                duration_seconds=30.0,
                tags="female vocal, 88 BPM",
                lyrics="[Verse]\nNeon fades",
                seed=7,
            )
            recorded = {}

            def fake_run_subprocess(command, **_kwargs):
                recorded["command"] = command
                (output_dir / "yue2_output.wav").write_bytes(b"RIFFtest")

                class Completed:
                    returncode = 0
                    stdout = '{"audio_seconds": 42.5, "sample_rate": 48000, "instrumental": false, "backend": "yue2"}'
                    stderr = ""

                return Completed()

            env = {
                "YUE2_MODEL_DIR": str(model_dir),
                "YUE2_VAE_DIR": str(vae_dir),
                "YUE2_PYTHON": "C:/yue2/python.exe",
                "YUE2_COT": "full",
                "YUE2_RUNTIME": "torch-eager",
                "YUE2_MEMORY_BUDGET_GIB": "24",
                "YUE2_OFFLOAD_AR": "true",
                "YUE2_ABC_PATH": "",
                "YUE2_CFG_SCALE": "",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("tools.ai.music_model_backends.python_exists", return_value=True):
                    with patch("tools.ai.music_model_backends.find_missing_python_modules", return_value=[]):
                        with patch("tools.ai.music_model_backends._run_subprocess", side_effect=fake_run_subprocess):
                            result = YuE2Backend().run(request)

        self.assertTrue(result.success)
        self.assertEqual("C:/yue2/python.exe", recorded["command"][0])
        self.assertTrue(recorded["command"][1].endswith("run_yue2_backend.py"))
        self.assertIn("warm piano pop", recorded["command"])
        self.assertIn("female vocal, 88 BPM", recorded["command"])
        self.assertIn("[Verse]\nNeon fades", recorded["command"])
        self.assertIn("--runtime-backend", recorded["command"])
        self.assertIn("torch-eager", recorded["command"])
        self.assertFalse(result.metadata["duration_honored"])
        self.assertEqual(42.5, result.metadata["generated_seconds"])
        self.assertTrue(any("Duration" in note for note in result.notes))

    def test_yue2_style_and_empty_lyrics(self) -> None:
        self.assertEqual("warm piano pop. 88 BPM", compose_style("warm piano pop", "88 BPM"))
        lyrics, instrumental = compose_yue2_lyrics("")
        self.assertEqual("[Instrumental]", lyrics)
        self.assertTrue(instrumental)
        lyrics, instrumental = compose_yue2_lyrics("[Chorus]\nSing")
        self.assertEqual("[Chorus]\nSing", lyrics)
        self.assertFalse(instrumental)

    def test_require_yue2_dirs_accept_snapshot_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "YuE2-3B"
            vae_dir = root / "YuE2-Vae"
            model_dir.mkdir()
            vae_dir.mkdir()
            (model_dir / "qwen.tiktoken").write_bytes(b"token")
            (model_dir / "model.safetensors").write_bytes(b"weights")
            (vae_dir / "config.json").write_text("{}", encoding="utf-8")
            (vae_dir / "model.safetensors").write_bytes(b"weights")
            nested_model, nested_vae = resolve_yue2_dirs(root, None)
            self.assertEqual(model_dir.resolve(), nested_model.resolve())
            self.assertEqual(vae_dir.resolve(), nested_vae.resolve())
            self.assertEqual(model_dir.resolve(), require_yue2_model_dir(model_dir))
            self.assertEqual(vae_dir.resolve(), require_yue2_vae_dir(vae_dir))


if __name__ == "__main__":
    unittest.main()
