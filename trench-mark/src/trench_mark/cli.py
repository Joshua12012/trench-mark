"""
cli.py
======
Trench-Mark: Automated, Memory-Safe TensorRT Performance Profiling, Universal Model Ingestion,
and Edge Hardware Telemetry for NVIDIA Jetson and Edge Devices.

How it works:
1. Takes a model name (e.g. 'yolov8n', 'resnet50') or local file (.onnx, .pt, .tflite).
2. Automatically downloads the model weights and exports to dynamic-batch ONNX.
3. Automatically inspects the ONNX graph for input node name and dimensions.
4. Compiles memory-safe TensorRT engines for requested precisions (FP32, FP16, INT8).
5. Benchmarks latency percentiles (min, max, mean, p50, p90, p95, p99), QPS, True FPS, and Peak RAM.
6. Prints a clean results table in the terminal and saves all data to a CSV file.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# Import our simplified modules
from .mock_runner import MockTRTExec
from .model_resolver import convert_onnx_to_fp16, get_model_zoo_help, resolve_model
from .telemetry import SystemHardwareProfiler, export_to_csv, parse_trtexec_stdout


# ANSI Color codes for clean terminal output
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class AuditLogger:
    """
    Logs all executed commands and tool outputs to a persistent file in logs/.
    Useful for debugging and recording exact reproducibility details.
    """
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = self.log_dir / f"benchmark_{timestamp}.log"
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\nTRENCH-MARK EXECUTION LOG\n" + "=" * 80 + f"\nStarted: {datetime.now()}\n\n")

    def log(self, message: str):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(message + ("\n" if not message.endswith("\n") else ""))
        except Exception:
            pass

    def command(self, cmd: List[str], output: str, code: int):
        self.log(
            f"\n{'='*80}\nCOMMAND: {' '.join(cmd)}\nEXIT CODE: {code}\nOUTPUT:\n{output}\n{'='*80}\n"
        )


logger: Optional[AuditLogger] = None


def find_trtexec() -> Optional[str]:
    """
    Finds the 'trtexec' binary on the system.
    Checks system $PATH first, then common NVIDIA JetPack installation paths.
    """
    found = shutil.which("trtexec")
    if found:
        return found

    # Standard JetPack and CUDA paths
    candidate_paths = [
        Path("/usr/src/tensorrt/bin/trtexec"),
        Path("/usr/local/cuda/bin/trtexec"),
        Path("/usr/bin/trtexec"),
    ]
    for candidate in candidate_paths:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            # Prepend to PATH so any child invocations and trt libs resolve
            os.environ["PATH"] = f"{candidate.parent}:{os.environ.get('PATH', '')}"
            return str(candidate)

    return None


def get_trtexec_version(trtexec_path: Optional[str] = None) -> Tuple[int, int, int]:
    """
    Determines the installed TensorRT major, minor, and patch version.
    Queries trtexec --help and parses the header banner (e.g. '[TensorRT v110300]' or '[TensorRT v8.6.1]').
    Returns (major, minor, patch), e.g. (11, 3, 0) or (8, 6, 2). Defaults to (8, 6, 0) if unknown.
    """
    bin_path = trtexec_path or find_trtexec()
    if not bin_path:
        try:
            import tensorrt as trt
            parts = [int(p) for p in trt.__version__.split(".") if p.isdigit()]
            while len(parts) < 3:
                parts.append(0)
            return (parts[0], parts[1], parts[2])
        except Exception:
            return (8, 6, 0)

    try:
        proc = subprocess.run([bin_path, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=5)
        out = proc.stdout
        match = re.search(r"TensorRT v?([0-9\.]+)", out)
        if match:
            ver_str = match.group(1)
            if "." in ver_str:
                parts = [int(p) for p in ver_str.split(".") if p.isdigit()]
                while len(parts) < 3:
                    parts.append(0)
                return (parts[0], parts[1], parts[2])
            else:
                num = int(ver_str)
                if num >= 100000:
                    return (num // 10000, (num % 10000) // 100, num % 100)
                elif num >= 10000:
                    return (num // 1000, (num % 1000) // 100, num % 100)
                elif num >= 1000:
                    return (num // 1000, (num % 1000) // 100, num % 100)
                else:
                    return (num, 0, 0)
    except Exception:
        pass

    return (8, 6, 0)


def is_jetson_system() -> bool:
    """Detects if running on an NVIDIA Jetson Linux platform (L4T / Tegra)."""
    return Path("/etc/nv_tegra_release").exists() or Path("/sys/devices/soc0/family").exists()


def install_tensorrt() -> bool:
    """
    Attempts to install NVIDIA TensorRT ('trtexec') on Linux via system package manager.
    Returns True if successfully installed and detected, False otherwise.
    """
    print(f"\n{Colors.CYAN}🔧 Attempting to install NVIDIA TensorRT and 'trtexec'...{Colors.RESET}")

    if sys.platform != "linux" or not shutil.which("apt-get"):
        print(f"{Colors.RED}❌ Automated TensorRT installation requires an apt-based Linux environment (Ubuntu / Debian / JetPack).{Colors.RESET}")
        return False

    is_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
    sudo_prefix = [] if is_root else ["sudo"]

    if not is_root and not shutil.which("sudo"):
        print(f"{Colors.RED}❌ Root privileges or 'sudo' required to install system packages.{Colors.RESET}")
        return False

    is_jetson = is_jetson_system()
    platform_name = "NVIDIA Jetson (Tegra)" if is_jetson else "Ubuntu/Debian Linux"
    print(f"  Platform detected: {Colors.BOLD}{platform_name}{Colors.RESET}")

    packages_to_install = ["tensorrt", "tensorrt-dev"]
    if is_jetson:
        packages_to_install.extend(["libnvinfer-bin", "libnvinfer-samples"])

    try:
        print(f"📦 Updating package index (apt-get update)...")
        subprocess.run(sudo_prefix + ["apt-get", "update", "-y"], check=False)

        print(f"📦 Installing TensorRT packages: {', '.join(packages_to_install)}...")
        install_cmd = sudo_prefix + ["apt-get", "install", "-y"] + packages_to_install
        subprocess.run(install_cmd, check=False)

        # Post-install check for trtexec binary
        found = find_trtexec()
        if not found and Path("/usr/src/tensorrt/bin/trtexec").is_file():
            try:
                subprocess.run(sudo_prefix + ["ln", "-sf", "/usr/src/tensorrt/bin/trtexec", "/usr/local/bin/trtexec"], check=False)
            except Exception:
                pass
            found = find_trtexec()

        if found:
            print(f"{Colors.GREEN}✅ NVIDIA TensorRT ('trtexec') successfully installed: {found}{Colors.RESET}\n")
            return True
        else:
            print(f"{Colors.YELLOW}⚠️  Apt installation finished, but 'trtexec' binary was not located in standard paths.{Colors.RESET}")
            return False

    except Exception as e:
        print(f"{Colors.RED}❌ Error during TensorRT installation: {e}{Colors.RESET}")
        return False


def run_command(cmd: List[str], is_mock: bool = False) -> str:
    """
    Runs a shell command (or simulated mock command) and logs the output.
    """
    global logger
    if logger:
        logger.log(f"Executing: {' '.join(cmd)}")

    if is_mock:
        stdout, code = MockTRTExec.execute(cmd)
        if logger:
            logger.command(cmd, stdout, code)
        return stdout

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if logger:
            logger.command(cmd, res.stdout, res.returncode)
        return res.stdout
    except Exception as e:
        err = str(e)
        if logger:
            logger.command(cmd, err, -1)
        return err


def print_banner(model_name: str, onnx_path: str, is_mock: bool, trt_version: Tuple[int, int, int]):
    """Prints a friendly startup banner."""
    mode_str = f"{Colors.YELLOW}[SIMULATION / MOCK MODE]{Colors.RESET}" if is_mock else f"{Colors.GREEN}[HARDWARE GPU MODE]{Colors.RESET}"
    trt_str = f"v{trt_version[0]}.{trt_version[1]}.{trt_version[2]}"
    print(f"\n{Colors.CYAN}{Colors.BOLD}========================================================================{Colors.RESET}")
    print(f"{Colors.BOLD} TRENCH-MARK: Edge Benchmarking & Telemetry Suite {mode_str}")
    print(f"{Colors.CYAN}========================================================================{Colors.RESET}")
    print(f"  Model Name : {Colors.BOLD}{model_name}{Colors.RESET}")
    print(f"  ONNX File  : {onnx_path}")
    print(f"  TensorRT   : {Colors.BOLD}{trt_str}{Colors.RESET}")


def print_table_header():
    """Prints the table header for benchmark metrics."""
    print("\n" + "=" * 115)
    print(
        f"{'PRECISION':<10} {'BATCH':<7} {'THROUGHPUT':<13} {'TRUE FPS':<11} "
        f"{'MEAN (ms)':<11} {'P50 (ms)':<10} {'P95 (ms)':<10} {'P99 (ms)':<10} {'PEAK RAM (MB)':<14}"
    )
    print("=" * 115)


def print_table_row(
    precision: str,
    batch_size: int,
    qps: float,
    fps: float,
    mean_ms: float,
    p50_ms: float,
    p95_ms: float,
    p99_ms: float,
    peak_ram: int,
):
    """Prints a single formatted result row in the terminal."""
    print(
        f"{precision:<10} {batch_size:<7} {f'{qps:.1f} qps':<13} {f'{fps:.1f}':<11} "
        f"{f'{mean_ms:.2f}':<11} {f'{p50_ms:.2f}':<10} {f'{p95_ms:.2f}':<10} {f'{p99_ms:.2f}':<10} {peak_ram:<14}"
    )


def parse_arguments() -> argparse.Namespace:
    """Configures command line arguments and appends the Model Zoo catalog to --help."""
    description = (
        "Automated, memory-safe TensorRT performance profiling and model ingestion for edge devices."
    )
    epilog = get_model_zoo_help() + """

Examples:
  # Benchmark YOLOv8 Nano across dynamic batches 1, 4, 8:
  trench-mark -m yolov8n -b 1 4 8 -p fp16 int8

  # Benchmark a Torchvision ResNet-50 backbone:
  trench-mark -m resnet50 -b 1 4 8 -p fp16

  # Run on CPU in simulation mode (no GPU or trtexec required):
  trench-mark -m yolov8n -b 1 4 8 -p fp16 int8 --mock

  # Save results to a custom CSV file:
  trench-mark -m yolov8n --export-csv my_results.csv
"""
    parser = argparse.ArgumentParser(
        prog="trench-mark",
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Target model identifier (e.g. 'yolov8n', 'resnet50') or local file (.onnx, .pt, .tflite).",
    )
    parser.add_argument(
        "-b", "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8],
        help="Space-separated list of batch sizes to benchmark (default: 1 4 8).",
    )
    parser.add_argument(
        "-p", "--precisions",
        nargs="+",
        choices=["fp32", "fp16", "int8", "int4"],
        default=["fp32", "fp16", "int8"],
        help="Precisions to compile and evaluate (default: fp32 fp16 int8, options: fp32 fp16 int8 int4).",
    )
    parser.add_argument(
        "-s", "--input-shape",
        type=str,
        default=None,
        help="Input tensor shape in CHW format (e.g. '3x640x640'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default=None,
        help="Input node binding name (e.g. 'images'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=200,
        help="Number of inference cycles per test (default: 200).",
    )
    parser.add_argument(
        "-w", "--warmup",
        type=int,
        default=50,
        help="Warmup period in milliseconds (default: 50).",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=2048,
        help="Maximum memory workspace for TensorRT compilation in MB (default: 2048).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory where models and compiled engines are cached (default: 'models').",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run simulation mode on CPU (no NVIDIA GPU or trtexec required).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Parent directory to consolidate all generated artifacts (logs/, models/, reports/).",
    )
    parser.add_argument(
        "--install-trt",
        action="store_true",
        help="Automatically attempt to install NVIDIA TensorRT ('trtexec') via apt if not detected.",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Filepath where CSV benchmark dataset will be saved.",
    )

    return parser.parse_args()


def main():
    """Main execution entry point."""
    global logger
    args = parse_arguments()

    # Configure unified directory structure if --output-dir is provided
    if args.output_dir:
        base_dir = Path(args.output_dir)
        logs_dir = base_dir / "logs"
        models_dir = Path(args.models_dir) if args.models_dir != "models" else (base_dir / "models")
        reports_dir = base_dir / "reports"
    else:
        logs_dir = Path("logs")
        models_dir = Path(args.models_dir)
        reports_dir = Path("reports")

    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger = AuditLogger(log_dir=str(logs_dir))

    # Determine if system has native trtexec or if simulation mode is needed
    trtexec_bin = find_trtexec()

    # If trtexec is missing, attempt or prompt installation
    if not trtexec_bin and (args.install_trt or (not args.mock and sys.stdin.isatty())):
        should_install = args.install_trt
        if not should_install:
            print(f"\n{Colors.YELLOW}⚠️  NVIDIA TensorRT ('trtexec') was not found on this Linux system.{Colors.RESET}")
            try:
                prompt_str = "Would you like Trench-Mark to attempt installing TensorRT via apt? [y/N]: "
                resp = input(prompt_str).strip().lower()
                should_install = resp in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                should_install = False

        if should_install:
            if install_tensorrt():
                trtexec_bin = find_trtexec()

    has_trtexec = bool(trtexec_bin)
    is_mock = args.mock or not has_trtexec

    if not has_trtexec and not args.mock:
        print(
            f"{Colors.YELLOW}⚠️  'trtexec' not found in system $PATH or standard JetPack directories. "
            f"Automatically activating simulation mode (--mock).{Colors.RESET}"
        )

    # --------------------------------------------------------------------------
    # 1. Model Resolution & Ingestion
    # --------------------------------------------------------------------------
    print(f"\n{Colors.CYAN}🔍 Ingesting & Preparing Model:{Colors.RESET} '{args.model}'...")
    try:
        model_name, onnx_path, input_name, input_shape = resolve_model(
            model_input=args.model,
            models_dir=str(models_dir),
            user_shape=args.input_shape,
            user_input_name=args.input_name,
        )
    except Exception as e:
        print(f"{Colors.RED}{Colors.BOLD}Error during model ingestion:{Colors.RESET} {e}")
        sys.exit(1)

    trt_version = (8, 6, 0) if is_mock else get_trtexec_version(trtexec_bin)
    trt_major, trt_minor, _ = trt_version

    print_banner(model_name, onnx_path, is_mock, trt_version)
    print(f"  Input Node : {input_name} | Input Shape: {input_shape}")

    if trt_major >= 11 and not is_mock:
        print(f"\n{Colors.CYAN}ℹ️  TensorRT {trt_major}.x detected (Strict Strongly Typed Mode).{Colors.RESET}")
        print(f"{Colors.CYAN}   Adapting compilation commands to modern TensorRT 11 standards.{Colors.RESET}")

    # Configure dynamic shape dimensions
    max_batch = max(args.batch_sizes)
    min_shape = f"{input_name}:1x{input_shape}"
    opt_shape = f"{input_name}:1x{input_shape}"
    max_shape = f"{input_name}:{max_batch}x{input_shape}"

    benchmark_rows: List[Dict[str, Any]] = []

    print_table_header()

    # --------------------------------------------------------------------------
    # 2. Precision & Batch Evaluation Loop
    # --------------------------------------------------------------------------
    for precision in args.precisions:
        active_onnx = onnx_path
        if precision == "fp16" and trt_major >= 11 and not is_mock:
            fp16_onnx = str(models_dir / f"{model_name}_fp16.onnx")
            print(f"⚙️  {Colors.CYAN}Preparing FP16-typed ONNX graph for TensorRT 11 strong typing...{Colors.RESET}")
            active_onnx = convert_onnx_to_fp16(onnx_path, fp16_onnx)

        engine_file = models_dir / f"{model_name}_{precision}_maxb{max_batch}.engine"

        # Check if engine already exists or needs compiling
        if not engine_file.exists() or engine_file.stat().st_size == 0:
            print(f"\n⚙️  {Colors.YELLOW}Compiling {precision.upper()} TensorRT Engine...{Colors.RESET}")
            build_cmd = [
                "trtexec",
                f"--onnx={active_onnx}",
                f"--saveEngine={str(engine_file)}",
                f"--minShapes={min_shape}",
                f"--optShapes={opt_shape}",
                f"--maxShapes={max_shape}",
                f"--tempdir={str(models_dir)}",
                "--tempfileControls=in_memory:deny,temporary:allow",
                f"--memPoolSize=workspace:{args.workspace}",
                "--profilingVerbosity=detailed",
            ]
            if trt_major < 11:
                # TensorRT 8.x / 10.x (NVIDIA Jetson JetPack 5 & 6)
                if precision == "fp16":
                    build_cmd.append("--fp16")
                elif precision == "int8":
                    build_cmd.extend(["--int8", "--fp16"])
                elif precision == "int4":
                    build_cmd.append("--int4")
            else:
                # TensorRT 11.x+ (Strict strongly-typed execution)
                if precision == "int8":
                    print(f"{Colors.YELLOW}⚠️  Note: In TensorRT 11+, --int8 flag was removed in favor of offline ModelOpt QDQ quantization.{Colors.RESET}")
                    print(f"{Colors.YELLOW}   Skipping runtime --int8 flag.{Colors.RESET}")

            run_command(build_cmd, is_mock=is_mock)

            if not engine_file.exists() or engine_file.stat().st_size == 0:
                print(f"{Colors.RED}❌ Failed to build {precision.upper()} engine. Check log: {logger.log_file}{Colors.RESET}")
                continue

        # Benchmark inference across each batch size
        for bs in sorted(args.batch_sizes):
            shape_arg = f"{input_name}:{bs}x{input_shape}"

            infer_cmd = [
                "trtexec",
                f"--loadEngine={str(engine_file)}",
                f"--shapes={shape_arg}",
                f"--iterations={args.iterations}",
                f"--warmUp={args.warmup}",
            ]
            if trt_major < 11:
                # --noDataTransfers is valid in TRT 8/10, but deprecated in TRT 11 (where transfers are disabled by default)
                infer_cmd.append("--noDataTransfers")

            # Start background memory monitoring
            profiler = SystemHardwareProfiler(mock=is_mock)
            profiler.start(batch_size=bs, precision=precision)

            infer_log = run_command(infer_cmd, is_mock=is_mock)

            # Extract metrics from trtexec output
            parsed = parse_trtexec_stdout(infer_log, batch_size=bs)
            qps_val = parsed["throughput"]["qps"]
            fps_val = parsed["throughput"]["fps"]

            # Stop memory monitor
            hw_metrics = profiler.stop(qps=qps_val, batch_size=bs)
            peak_ram = hw_metrics["peak_ram_mb"]

            lats = parsed["latencies"]
            mean_ms = lats["mean_ms"]
            p50_ms = lats["p50_ms"]
            p95_ms = lats["p95_ms"]
            p99_ms = lats["p99_ms"]

            # Record row for CSV export
            row_data = {
                "model": model_name,
                "precision": precision.upper(),
                "batch_size": bs,
                "throughput_qps": qps_val,
                "throughput_fps": fps_val,
                "latency_mean_ms": mean_ms,
                "latency_min_ms": lats["min_ms"],
                "latency_max_ms": lats["max_ms"],
                "latency_p50_ms": p50_ms,
                "latency_p90_ms": lats["p90_ms"],
                "latency_p95_ms": p95_ms,
                "latency_p99_ms": p99_ms,
                "peak_ram_mb": peak_ram,
            }
            benchmark_rows.append(row_data)

            # Print formatted row in terminal table
            print_table_row(
                precision=precision.upper(),
                batch_size=bs,
                qps=qps_val,
                fps=fps_val,
                mean_ms=mean_ms,
                p50_ms=p50_ms,
                p95_ms=p95_ms,
                p99_ms=p99_ms,
                peak_ram=peak_ram,
            )

    print("=" * 115)

    # --------------------------------------------------------------------------
    # 3. Save Benchmark Results to CSV
    # --------------------------------------------------------------------------
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_filename = f"benchmark_{model_name}_{ts}.csv"

    # Intelligently determine if user passed a directory or a specific filename
    if args.export_csv:
        target_path = Path(args.export_csv)
        if target_path.is_dir() or str(args.export_csv).endswith(("/", "\\")) or not target_path.suffix:
            target_path.mkdir(parents=True, exist_ok=True)
            csv_file = str(target_path / default_filename)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = str(target_path)
    else:
        csv_file = str(reports_dir / default_filename)

    saved_csv = export_to_csv(benchmark_rows, csv_file)
    print(f"\n📈 {Colors.GREEN}{Colors.BOLD}Benchmark results successfully saved to CSV:{Colors.RESET} {saved_csv}")
    print(f"📝 {Colors.CYAN}Execution audit log:{Colors.RESET} {logger.log_file}\n")


if __name__ == "__main__":
    main()