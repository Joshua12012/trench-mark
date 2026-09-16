"""
test_mock_pipeline.py
=====================
Tests the mock simulation pipeline end-to-end without requiring GPU hardware.
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Add src to pythonpath
src_dir = Path(__file__).resolve().parent.parent / "src"
if not src_dir.exists():
    src_dir = Path(__file__).resolve().parent.parent / "trench-mark" / "src"
sys.path.insert(0, str(src_dir))

from trench_mark.mock_runner import MockTRTExec, MockTegraStats


class TestMockPipeline(unittest.TestCase):

    def test_mock_build_command(self):
        """Tests that MockTRTExec creates a placeholder engine file on build commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine_path = Path(tmpdir) / "test.engine"
            cmd = [
                "trtexec",
                "--onnx=dummy.onnx",
                f"--saveEngine={engine_path}",
                "--minShapes=images:1x3x640x640",
                "--optShapes=images:1x3x640x640",
                "--maxShapes=images:8x3x640x640",
                "--fp16",
                "--memPoolSize=workspace:2048",
            ]
            stdout, code = MockTRTExec.execute(cmd)
            self.assertEqual(code, 0)
            self.assertIn("PASSED", stdout)
            self.assertTrue(engine_path.exists())
            self.assertGreater(engine_path.stat().st_size, 0)

    def test_mock_infer_command(self):
        """Tests that MockTRTExec generates realistic throughput and latency stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            times_json = Path(tmpdir) / "times.json"
            cmd = [
                "trtexec",
                "--loadEngine=test_fp16.engine",
                "--shapes=images:4x3x640x640",
                "--iterations=50",
                "--warmUp=10",
                f"--exportTimes={times_json}",
            ]
            stdout, code = MockTRTExec.execute(cmd)
            self.assertEqual(code, 0)
            self.assertIn("Throughput:", stdout)
            self.assertTrue(times_json.exists())
            self.assertGreater(times_json.stat().st_size, 0)

    def test_mock_tegrastats(self):
        """Tests that MockTegraStats returns a valid Jetson tegrastats telemetry line."""
        line = MockTegraStats.sample_line(batch_size=4, precision="fp16")
        self.assertIn("RAM", line)
        self.assertIn("VDD_IN", line)

    def test_system_detection_helpers(self):
        """Tests system detection and trtexec lookup helpers."""
        from trench_mark.cli import find_trtexec, is_jetson_system
        self.assertIsInstance(is_jetson_system(), bool)
        trtexec = find_trtexec()
        self.assertTrue(trtexec is None or isinstance(trtexec, str))

    def test_trtexec_version_detection(self):
        """Tests that get_trtexec_version correctly parses TensorRT 8, 10, and 11 version banners."""
        from unittest.mock import MagicMock, patch
        from trench_mark.cli import get_trtexec_version

        # Test TensorRT 11.3 banner (e.g. from Google Colab / CUDA 12.8)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="&&&& RUNNING TensorRT.trtexec [TensorRT v110300] [b99]\n")
            ver = get_trtexec_version("/dummy/trtexec")
            self.assertEqual(ver, (11, 3, 0))

        # Test TensorRT 8.6 banner (e.g. from Jetson JetPack 5 / 6)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="&&&& RUNNING TensorRT.trtexec [TensorRT v8602]\n")
            ver = get_trtexec_version("/dummy/trtexec")
            self.assertEqual(ver, (8, 6, 2))

        # Test TensorRT 10.1 banner (e.g. from JetPack 6.1)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="&&&& RUNNING TensorRT.trtexec [TensorRT v100100]\n")
            ver = get_trtexec_version("/dummy/trtexec")
            self.assertEqual(ver, (10, 1, 0))

        # Test dotted banner (e.g. [TensorRT v8.6.1])
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="&&&& RUNNING TensorRT.trtexec [TensorRT v8.6.1]\n")
            ver = get_trtexec_version("/dummy/trtexec")
            self.assertEqual(ver, (8, 6, 1))

        # Test fallback when trtexec is absent
        with patch("trench_mark.cli.find_trtexec", return_value=None):
            ver = get_trtexec_version(None)
            self.assertEqual(ver, (8, 6, 0))

    def test_cli_argument_parsing(self):
        """Tests that --output-dir and --install-trt arguments parse correctly."""
        from trench_mark.cli import parse_arguments
        import sys
        orig_argv = sys.argv
        try:
            sys.argv = ["trench-mark", "-m", "yolov8n", "-o", "custom_out", "--install-trt"]
            args = parse_arguments()
            self.assertEqual(args.output_dir, "custom_out")
            self.assertTrue(args.install_trt)
        finally:
            sys.argv = orig_argv


if __name__ == "__main__":
    unittest.main()
