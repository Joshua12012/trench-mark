"""
test_telemetry.py
=================
Unit tests for telemetry parsers, latency statistics, hardware profiler, and CSV exporter.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

# Add src to pythonpath
src_dir = Path(__file__).resolve().parent.parent / "src"
if not src_dir.exists():
    src_dir = Path(__file__).resolve().parent.parent / "trench-mark" / "src"
sys.path.insert(0, str(src_dir))

from trench_mark.telemetry import (
    SystemHardwareProfiler,
    compute_energy_metrics,
    compute_latency_statistics,
    export_to_csv,
    parse_trtexec_stdout,
)


class TestTelemetry(unittest.TestCase):

    def test_latency_statistics(self):
        """Tests that compute_latency_statistics computes valid percentiles and jitter."""
        # 100 sample latencies: 10.0, 10.1, 10.2, ... 19.9
        samples = [10.0 + (i * 0.1) for i in range(100)]
        stats = compute_latency_statistics(samples)

        self.assertEqual(stats.count, 100)
        self.assertAlmostEqual(stats.min_ms, 10.0, places=2)
        self.assertAlmostEqual(stats.max_ms, 19.9, places=2)
        self.assertAlmostEqual(stats.p50_ms, 14.95, places=1)
        self.assertGreater(stats.p99_ms, stats.p95_ms)
        self.assertGreater(stats.std_dev_ms, 0)
        self.assertGreater(stats.jitter_ms, 0)

    def test_energy_metrics(self):
        """Tests energy consumption and efficiency calculation."""
        energy = compute_energy_metrics(
            avg_power_w=12.5,
            peak_power_w=15.0,
            duration_sec=10.0,
            qps=50.0,
            batch_size=4,
        )
        self.assertEqual(energy["avg_power_w"], 12.5)
        self.assertEqual(energy["total_energy_j"], 125.0)
        # fps = 200, energy_mj_per_frame = 12.5 / 200 * 1000 = 62.5
        self.assertAlmostEqual(energy["energy_mj_per_frame"], 62.5, places=1)
        # efficiency = 200 / 12.5 = 16.0
        self.assertAlmostEqual(energy["efficiency_fps_per_w"], 16.0, places=1)

    def test_parse_trtexec_stdout(self):
        """Tests extraction of throughput, latency percentiles, and memory from trtexec output."""
        sample_log = """
[I] === Performance summary ===
[I] Throughput: 147.20 qps
[I] Latency: min = 6.12 ms, max = 8.45 ms, mean = 6.79 ms, median = 6.75 ms, percentile(90%) = 7.02 ms, percentile(95%) = 7.15 ms, percentile(99%) = 7.35 ms
[I] Enqueue Time: min = 0.12 ms, max = 0.35 ms, mean = 0.18 ms, median = 0.17 ms
[I] GPU Compute Time: min = 5.95 ms, max = 8.10 ms, mean = 6.55 ms, median = 6.50 ms
[I] Total Host Persistent Memory: 42.15 MiB
[I] Total Device Persistent Memory: 85.32 MiB
[I] Total Scratch Memory: 112.50 MiB
&&&& PASSED TensorRT.trtexec
"""
        parsed = parse_trtexec_stdout(sample_log, batch_size=4)
        self.assertEqual(parsed["throughput"]["qps"], 147.20)
        self.assertEqual(parsed["throughput"]["fps"], 147.20 * 4)
        self.assertEqual(parsed["latencies"]["mean_ms"], 6.79)
        self.assertEqual(parsed["latencies"]["p50_ms"], 6.75)
        self.assertEqual(parsed["latencies"]["p95_ms"], 7.15)
        self.assertEqual(parsed["latencies"]["p99_ms"], 7.35)
        self.assertEqual(parsed["latencies"]["gpu_compute"]["mean"], 6.55)
        self.assertEqual(parsed["memory_engine"]["host_persistent"]["value"], 42.15)

    def test_system_profiler(self):
        """Tests the background memory profiler in mock mode."""
        profiler = SystemHardwareProfiler(mock=True)
        profiler.start(batch_size=4, precision="fp16")
        metrics = profiler.stop(qps=150.0, batch_size=4)
        self.assertIn("peak_ram_mb", metrics)
        self.assertGreater(metrics["peak_ram_mb"], 0)

    def test_export_to_csv(self):
        """Tests saving benchmark rows to CSV and verifies format and headers."""
        results = [
            {
                "model": "yolov8n",
                "precision": "FP16",
                "batch_size": 1,
                "throughput_qps": 147.2,
                "throughput_fps": 147.2,
                "latency_mean_ms": 6.79,
                "latency_min_ms": 6.12,
                "latency_max_ms": 8.45,
                "latency_p50_ms": 6.75,
                "latency_p90_ms": 7.02,
                "latency_p95_ms": 7.15,
                "latency_p99_ms": 7.35,
                "peak_ram_mb": 3340,
            },
            {
                "model": "yolov8n",
                "precision": "FP16",
                "batch_size": 4,
                "throughput_qps": 85.0,
                "throughput_fps": 340.0,
                "latency_mean_ms": 11.76,
                "latency_min_ms": 10.50,
                "latency_max_ms": 13.20,
                "latency_p50_ms": 11.70,
                "latency_p90_ms": 12.10,
                "latency_p95_ms": 12.40,
                "latency_p99_ms": 12.80,
                "peak_ram_mb": 3760,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "test_benchmark.csv"
            export_to_csv(results, str(csv_file))

            self.assertTrue(csv_file.exists())
            self.assertGreater(csv_file.stat().st_size, 0)

            # Read back and verify CSV content
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = list(csv.reader(f))
                self.assertEqual(len(reader), 3)  # Header + 2 data rows
                headers = reader[0]
                self.assertIn("model", headers)
                self.assertIn("throughput_qps", headers)
                self.assertIn("throughput_fps", headers)
                self.assertIn("latency_p99_ms", headers)
                self.assertIn("peak_ram_mb", headers)


if __name__ == "__main__":
    unittest.main()
