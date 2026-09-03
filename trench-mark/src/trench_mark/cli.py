import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
import yaml

# ANSI Color Codes for Premium Terminal UI
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    DIM = '\033[2m'

class MemoryProfiler:
    def __init__(self):
        self.running = False
        self.max_ram = 0
        self.thread = None
        self.process = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        if not shutil.which("tegrastats"):
            return
        try:
            self.process = subprocess.Popen(
                ["tegrastats"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            while self.running:
                line = self.process.stdout.readline()
                if not line: break
                match = re.search(r"RAM\s+(\d+)/\d+MB", line)
                if match:
                    self.max_ram = max(self.max_ram, int(match.group(1)))
        except Exception:
            pass

    def stop(self):
        self.running = False
        if self.process:
            try: self.process.terminate()
            except Exception: pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        return self.max_ram

class Logger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(self.log_dir, f"benchmark_{timestamp}.log")
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\nTRENCH-MARK LOG\n" + "=" * 90 + f"\nStarted: {datetime.now()}\n\n")

    def write(self, message):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(message + ("\n" if not message.endswith("\n") else ""))
        except Exception: pass

    def command(self, cmd, output, returncode):
        self.write(f"\n{'='*90}\nCOMMAND: {' '.join(cmd)}\nRETURN CODE: {returncode}\nOUTPUT:\n{output if output else '[No output]'}\n{'='*90}\n")

logger = None

def run_command(cmd):
    global logger
    if logger: logger.write(f"\nExecuting: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if logger: logger.command(cmd, result.stdout, result.returncode)
        return result.stdout
    except Exception as e:
        if logger: logger.command(cmd, str(e), -1)
        return str(e)

def extract_metrics(log):
    lat = re.search(r"mean\s*[=:]\s*([\d\.]+)\s*ms", log, re.IGNORECASE)
    fps = re.search(r"Throughput\s*[=:]\s*([\d\.]+)\s*qps", log, re.IGNORECASE)
    return {
        "latency_ms": float(lat.group(1)) if lat else None,
        "fps": float(fps.group(1)) if fps else None
    }

def parse_arguments():
    # Formats the help menu beautifully
    description = f"""
{Colors.CYAN}{Colors.BOLD}╔╦╗┬─┐┌─┐┌┐┌┌─┐┬ ┬   ╔╦╗┌─┐┬─┐┬┌─
 ║ ├┬┘├┤ ││││  ├─┤───║║║├─┤├┬┘├┴┐
 ╩ ┴└─└─┘┘└┘└─┘┴ ┴   ╩ ╩┴ ┴┴└─┴ ┴{Colors.RESET}
Automated, memory-safe TensorRT performance profiling for Jetson edge devices.
"""
    parser = argparse.ArgumentParser(
        prog="trench-mark",
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"{Colors.YELLOW}Example:{Colors.RESET} sudo trench-mark -m yolov8n.onnx -b 1 4 8 -p fp16 int8"
    )

    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the target ONNX model file (Required).")
    parser.add_argument("-s", "--input-shape", type=str, default="3x640x640", help="Input tensor shape in CHW format. Default: 3x640x640.")
    parser.add_argument("--input-name", type=str, default="images", help="Input binding name in the ONNX model. Default: 'images'.")
    parser.add_argument("-p", "--precisions", nargs="+", choices=["fp32", "fp16", "int8"], default=["fp32", "fp16", "int8"], help="List of precisions to evaluate. Default: fp32 fp16 int8")
    parser.add_argument("-b", "--batch-sizes", nargs="+", type=int, default=[1, 4, 8], help="List of dynamic batch sizes. Default: 1 4 8")
    parser.add_argument("-i", "--iterations", type=int, default=500, help="Inference cycles per run. Default: 500")
    parser.add_argument("-w", "--warmup", type=int, default=50, help="Warmup duration in ms. Default: 50")
    parser.add_argument("--workspace", type=int, default=2048, help="Max workspace memory in MB. Default: 2048")
    
    return parser.parse_args()

def main():
    global logger
    args = parse_arguments()
    logger = Logger()

    if not os.path.exists(args.model):
        print(f"{Colors.RED}{Colors.BOLD}Error:{Colors.RESET} ONNX file '{args.model}' does not exist.")
        sys.exit(1)

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    input_shape = args.input_shape[2:] if args.input_shape.startswith("1x") else args.input_shape
    
    min_shape = f"{args.input_name}:1x{input_shape}"
    opt_shape = f"{args.input_name}:1x{input_shape}"
    max_shape = f"{args.input_name}:{max(args.batch_sizes)}x{input_shape}"

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 TRENCH-MARK EDGE BENCHMARK{Colors.RESET}")
    print(f"{Colors.DIM}Target Model:{Colors.RESET} {args.model}")
    print(f"{Colors.DIM}Output Dir:{Colors.RESET} ./{models_dir}/\n")
    
    # Beautiful Terminal Table Header
    print(f"{Colors.BOLD}{'Precision':<10} | {'Batch':<7} | {'Latency (ms)':<15} | {'Throughput (FPS)':<18} | {'Peak RAM (MB)'}{Colors.RESET}")
    print("-" * 75)

    for precision in args.precisions:
        engine_path = os.path.join(models_dir, f"{model_name}_{precision}.engine")
        
        is_corrupted = os.path.exists(engine_path) and os.path.getsize(engine_path) == 0

        if not os.path.exists(engine_path) or is_corrupted:
            print(f"{Colors.YELLOW}⚙️  Building {precision.upper()} engine (This may take a few minutes)...{Colors.RESET}")
            build_cmd = [
                "trtexec", f"--onnx={args.model}", f"--saveEngine={engine_path}",
                f"--minShapes={min_shape}", f"--optShapes={opt_shape}", f"--maxShapes={max_shape}",
                "--tempdir=.", "--tempfileControls=in_memory:deny,temporary:allow",
                f"--memPoolSize=workspace:{args.workspace}", 
                "--timingCacheFile=trt.cache" # KEEP the cache, but REMOVE the optimization level
            ]
            if precision == "fp16": build_cmd.append("--fp16")
            elif precision == "int8": build_cmd.extend(["--int8", "--fp16"])
            
            run_command(build_cmd)

            if not os.path.exists(engine_path) or os.path.getsize(engine_path) == 0:
                print(f"{Colors.RED}❌ {precision.upper()} engine build failed. Check logs.{Colors.RESET}")
                continue

        for bs in sorted(args.batch_sizes):
            shape = f"{args.input_name}:{bs}x{input_shape}"
            infer_cmd = [
                "trtexec", f"--loadEngine={engine_path}", f"--shapes={shape}",
                f"--iterations={args.iterations}", f"--warmUp={args.warmup}", "--noDataTransfers"
            ]

            profiler = MemoryProfiler()
            profiler.start()
            infer_log = run_command(infer_cmd)
            peak_ram = profiler.stop()
            metrics = extract_metrics(infer_log)

            if metrics["latency_ms"] and metrics["fps"]:
                print(f"{Colors.GREEN}{precision.upper():<10}{Colors.RESET} | {bs:<7} | {metrics['latency_ms']:<15.2f} | {metrics['fps']:<18.2f} | {peak_ram}")
            else:
                print(f"{Colors.RED}{precision.upper():<10} | {bs:<7} | {'FAILED':<15} | {'FAILED':<18} | {peak_ram}{Colors.RESET}")

    print("-" * 75)
    print(f"📄 Full execution log saved to: {Colors.CYAN}{logger.log_file}{Colors.RESET}\n")

if __name__ == "__main__":
    main()