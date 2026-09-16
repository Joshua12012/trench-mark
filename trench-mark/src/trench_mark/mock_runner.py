"""
mock_runner.py
==============
This module provides high-fidelity simulation (Mocking) of NVIDIA TensorRT tools:
1. MockTRTExec: Simulates TensorRT engine compilation and inference benchmarking.
   - Generates realistic latency percentiles, throughput (QPS and FPS), and memory output.
   - Allows users to test and develop trench-mark on any Linux machine (x86_64 or aarch64)
     without requiring an NVIDIA Jetson or dedicated GPU.
2. MockTegraStats: Simulates the NVIDIA Jetson tegrastats hardware telemetry stream
   (RAM usage, VDD_IN power consumption, CPU/GPU utilization).
"""

import json
import math
from pathlib import Path
import re
from typing import List, Tuple


class MockTRTExec:
    """
    Simulates NVIDIA 'trtexec' command-line behavior.
    
    When building an engine (--saveEngine):
      Creates a dummy engine file on disk and outputs TensorRT compilation logs.
      
    When benchmarking an engine (--loadEngine):
      Extracts batch size and precision from the arguments, calculates realistic
      edge-device latency numbers, and prints the standard trtexec performance summary.
    """

    @classmethod
    def execute(cls, cmd: List[str]) -> Tuple[str, int]:
        """
        Executes a simulated trtexec command.
        
        Returns:
          (stdout_string, exit_code)
        """
        cmd_str = " ".join(cmd)

        # ----------------------------------------------------------------------
        # Phase 1: Engine Build Simulation (--saveEngine)
        # ----------------------------------------------------------------------
        if "--saveEngine" in cmd_str:
            engine_path = None
            for arg in cmd:
                if arg.startswith("--saveEngine="):
                    engine_path = arg.split("=", 1)[1]
                elif arg == "--saveEngine" and cmd.index(arg) + 1 < len(cmd):
                    engine_path = cmd[cmd.index(arg) + 1]

            if engine_path:
                p = Path(engine_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                # Write a placeholder binary header so the file exists and is valid on disk
                with open(p, "wb") as f:
                    f.write(b"TRT_MOCK_ENGINE_DATA_V86" + b"\x00" * 4096)

            stdout = (
                "[I] === Model Options ===\n"
                "[I] Format: ONNX\n"
                "[I] === Build Options ===\n"
                "[I] Dynamic dimensions configured.\n"
                "[I] Memory Pool: 2048 MiB allocated for tactic evaluation.\n"
                "[I] Engine generation completed successfully.\n"
                "[I] Engine built in 1.85 seconds.\n"
                "&&&& PASSED TensorRT.trtexec [TensorRT v8.6.1]\n"
            )
            return stdout, 0

        # ----------------------------------------------------------------------
        # Phase 2: Inference Benchmarking Simulation (--loadEngine)
        # ----------------------------------------------------------------------
        # Extract batch size from --shapes (e.g. --shapes=images:4x3x640x640)
        batch_size = 1
        shapes_match = re.search(r"--shapes=[^:]*:(\d+)x", cmd_str)
        if shapes_match:
            try:
                batch_size = int(shapes_match.group(1))
            except ValueError:
                batch_size = 1

        # Extract precision from engine filename or command flags
        precision = "fp32"
        if "int4" in cmd_str.lower():
            precision = "int4"
        elif "int8" in cmd_str.lower():
            precision = "int8"
        elif "fp16" in cmd_str.lower():
            precision = "fp16"

        # Realistic baseline latencies for an edge device (e.g. Jetson Orin Nano):
        # FP32 is slowest, FP16 is ~1.8x faster, INT8 is ~2.8x faster, INT4 is ~4.5x faster (weight-only)
        base_latencies = {
            "fp32": 10.50,  # ~10.5 ms for batch=1
            "fp16": 5.80,   # ~5.8 ms for batch=1
            "int8": 3.40,   # ~3.4 ms for batch=1
            "int4": 2.20,   # ~2.2 ms for batch=1 (INT4 weight-only / W4A16 acceleration)
        }
        base_ms = base_latencies.get(precision, 8.0)

        # As batch size increases, latency scales sub-linearly (approx bs^0.55)
        mean_ms = base_ms * math.pow(batch_size, 0.55)
        min_ms = round(mean_ms * 0.94, 2)
        max_ms = round(mean_ms * 1.18, 2)
        median_ms = round(mean_ms * 0.99, 2)  # p50
        p90_ms = round(mean_ms * 1.04, 2)
        p95_ms = round(mean_ms * 1.08, 2)
        p99_ms = round(mean_ms * 1.14, 2)
        mean_ms = round(mean_ms, 2)

        # In TensorRT, QPS (Queries Per Second) = batches per second
        # Throughput QPS = 1000.0 / mean_ms
        qps = round(1000.0 / mean_ms, 2)

        # Enqueue and GPU compute timing
        enqueue_mean = 0.18
        gpu_compute_mean = round(mean_ms * 0.96, 2)

        # Check if caller requested exported times json (--exportTimes)
        export_times_path = None
        for arg in cmd:
            if arg.startswith("--exportTimes="):
                export_times_path = arg.split("=", 1)[1]
        if export_times_path:
            times_p = Path(export_times_path)
            times_p.parent.mkdir(parents=True, exist_ok=True)
            # Create a sample list of individual execution times
            mock_times = [
                {"computeMs": gpu_compute_mean, "latencyMs": mean_ms}
                for _ in range(50)
            ]
            with open(times_p, "w", encoding="utf-8") as f:
                json.dump(mock_times, f)

        # Format trtexec output exactly as the real binary prints it
        stdout = f"""
[I] === Performance summary ===
[I] Throughput: {qps:.2f} qps
[I] Latency: min = {min_ms:.2f} ms, max = {max_ms:.2f} ms, mean = {mean_ms:.2f} ms, median = {median_ms:.2f} ms, percentile(90%) = {p90_ms:.2f} ms, percentile(95%) = {p95_ms:.2f} ms, percentile(99%) = {p99_ms:.2f} ms
[I] Enqueue Time: min = 0.12 ms, max = 0.35 ms, mean = {enqueue_mean:.2f} ms, median = 0.17 ms
[I] GPU Compute Time: min = {min_ms * 0.95:.2f} ms, max = {max_ms * 0.95:.2f} ms, mean = {gpu_compute_mean:.2f} ms, median = {median_ms * 0.95:.2f} ms
[I] Total Host Persistent Memory: 42.15 MiB
[I] Total Device Persistent Memory: 85.32 MiB
[I] Total Scratch Memory: 112.50 MiB
&&&& PASSED TensorRT.trtexec [TensorRT v8.6.1]
"""
        return stdout, 0


class MockTegraStats:
    """
    Generates realistic NVIDIA Jetson 'tegrastats' telemetry lines.
    
    On Jetson devices, tegrastats reports:
    - RAM usage (Unified memory shared between CPU and GPU)
    - Power consumption (VDD_IN in milliwatts)
    - CPU core loads and GPU frequencies
    """

    @classmethod
    def sample_line(cls, batch_size: int = 1, precision: str = "fp16") -> str:
        """
        Returns a sample tegrastats output line corresponding to the workload.
        """
        # RAM increases with batch size
        ram_used = 3200 + (batch_size * 140)
        # Power increases under load (around 11.5W - 14.5W on Orin Nano)
        power_mw = 11500 + (batch_size * 350)

        return (
            f"RAM {ram_used}/7620MB (lfb 420x4MB) SWAP 0/3810MB (cached 0MB) "
            f"CPU [25%@1420,28%@1420,20%@1420,18%@1420,22%@1420,24%@1420] "
            f"EMC_FREQ 0%@1600 GR3D_FREQ 72%@624 VDD_IN {power_mw}mW"
        )

    @classmethod
    def get_mock_ram_mb(cls, batch_size: int = 1) -> int:
        """Convenience method returning peak RAM in Megabytes for mock runs."""
        return 3200 + (batch_size * 140)
