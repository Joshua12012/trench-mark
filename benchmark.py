import subprocess
import re
import yaml
import threading
import sys
import os
from datetime import datetime

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
        try:
            self.process = subprocess.Popen(
                ["tegrastats"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            while self.running:
                line = self.process.stdout.readline()
                if not line:
                    break
                match = re.search(r"RAM\s+(\d+)/\d+MB", line)
                if match:
                    ram = int(match.group(1))
                    self.max_ram = max(self.max_ram, ram)
        except Exception as e:
            if logger:
                logger.write(f"Memory profiler error: {e}")

    def stop(self):
        self.running = False
        if self.process:
            try:
                self.process.terminate()
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=2)
        return self.max_ram

class Logger:
    def __init__(self):
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(
            self.log_dir,
            f"benchmark_{timestamp}.log"
        )
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n")
            f.write("YOLOv8 TENSORRT BENCHMARK LOG\n")
            f.write("=" * 90 + "\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write(f"Working Directory: {os.getcwd()}\n")
            f.write("=" * 90 + "\n\n")

    def write(self, message):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(message)
                if not message.endswith("\n"):
                    f.write("\n")
        except Exception:
            pass

    def command(self, cmd, output, returncode):
        self.write("\n" + "=" * 90)
        self.write("COMMAND")
        self.write("=" * 90)
        self.write(" ".join(cmd))
        self.write("\nRETURN CODE")
        self.write("=" * 90)
        self.write(str(returncode))
        self.write("\nOUTPUT")
        self.write("=" * 90)
        self.write(output if output else "[No output]")
        self.write("\n" + "=" * 90 + "\n")

logger = None

def run_command(cmd):
    global logger
    if logger:
        logger.write(f"\nExecuting command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        output = result.stdout
        if logger:
            logger.command(cmd, output, result.returncode)
        return output
    except Exception as e:
        error = f"Command execution failed: {e}"
        if logger:
            logger.command(cmd, error, -1)
        return error

def extract_metrics(log):
    latency_match = re.search(
        r"mean\s*[=:]\s*([\d\.]+)\s*ms",
        log,
        re.IGNORECASE
    )
    throughput_match = re.search(
        r"Throughput\s*[=:]\s*([\d\.]+)\s*qps",
        log,
        re.IGNORECASE
    )
    latency = float(latency_match.group(1)) if latency_match else None
    throughput = float(throughput_match.group(1)) if throughput_match else None
    return {
        "latency_ms": latency,
        "fps": throughput
    }

def log_system_information():
    global logger
    if not logger:
        return
    try:
        logger.write("\n" + "=" * 90)
        logger.write("SYSTEM INFORMATION")
        logger.write("=" * 90)
        commands = [
            ["uname", "-a"],
            ["cat", "/etc/nv_tegra_release"],
            ["python3", "--version"],
            ["trtexec", "--help"]
        ]
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                logger.command(cmd, result.stdout, result.returncode)
            except Exception as e:
                logger.write(f"Failed to execute {' '.join(cmd)}: {e}")
    except Exception as e:
        logger.write(f"System information logging failed: {e}")

def main():
    global logger
    logger = Logger()
    logger.write("Loading configuration...")
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.write(f"Failed to load config.yaml: {e}")
        print(f"Failed to load config.yaml: {e}")
        return

    model_name = config["model"]["name"]
    onnx_path = config["model"]["onnx_path"]
    batch_sizes = config["benchmarks"]["batch_sizes"]
    precisions = config["benchmarks"]["precisions"]
    iterations = config["settings"]["iterations"]
    warmup = config["settings"]["warmup"]
    input_shape = config["model"]["input_shape"]

    logger.write("\nCONFIGURATION")
    logger.write("=" * 90)
    logger.write(yaml.dump(config, sort_keys=False))
    logger.write("=" * 90)
    
    log_system_information()

    input_name = "images"
    min_shape = f"{input_name}:1x{input_shape}"
    opt_shape = f"{input_name}:1x{input_shape}"
    max_batch = max(batch_sizes)
    max_shape = f"{input_name}:{max_batch}x{input_shape}"

    print("\nYOLOv8 TENSORRT BENCHMARK")
    print("=" * 75)
    print(
        f"{'Precision':<10} | "
        f"{'Batch':<7} | "
        f"{'Latency (ms)':<15} | "
        f"{'Throughput (FPS)':<18} | "
        f"{'Peak RAM (MB)'}"
    )
    print("-" * 75)

    logger.write("\nBenchmark started.")
    logger.write(f"Input name: {input_name}")
    logger.write(f"Input shape: {input_shape}")
    logger.write(f"Min shape: {min_shape}")
    logger.write(f"Opt shape: {opt_shape}")
    logger.write(f"Max shape: {max_shape}")

    for precision in precisions:
        engine_path = f"{model_name}_{precision}.engine"
        
        # Check if engine already exists to save build time
        if not os.path.exists(engine_path):
            logger.write("\n" + "#" * 90)
            logger.write(f"BUILDING {precision.upper()} ENGINE")
            logger.write("#" * 90)
            
            print(f"Building {precision.upper()} engine... (This can take 15+ mins)")
            
            # Jetson Memory Safety Build Command
            build_cmd = [
                "trtexec",
                f"--onnx={onnx_path}",
                f"--saveEngine={engine_path}",
                f"--minShapes={min_shape}",
                f"--optShapes={opt_shape}",
                f"--maxShapes={max_shape}",
                "--tempdir=.", 
                "--tempfileControls=in_memory:deny,temporary:allow",
                "--memPoolSize=workspace:2048"
            ]
            
            if precision == "fp16":
                build_cmd.append("--fp16")
            elif precision == "int8":
                build_cmd.extend(["--int8", "--fp16"])
                
            run_command(build_cmd)

            if not os.path.exists(engine_path):
                print(f"{precision.upper()} engine build failed.")
                logger.write(f"\n{precision.upper()} ENGINE BUILD FAILED.")
                continue
                
            logger.write(f"\n{precision.upper()} engine successfully created: {engine_path}")
        else:
            logger.write(f"\nFound existing {precision.upper()} engine. Skipping build.")

        for bs in batch_sizes:
            logger.write("\n" + "-" * 90)
            logger.write(f"RUNNING INFERENCE | Precision={precision.upper()} | Batch={bs}")
            logger.write("-" * 90)
            
            shape = f"{input_name}:{bs}x{input_shape}"
            infer_cmd = [
                "trtexec",
                f"--loadEngine={engine_path}",
                f"--shapes={shape}",
                f"--iterations={iterations}",
                f"--warmUp={warmup}",
                "--noDataTransfers"
            ]

            profiler = MemoryProfiler()
            profiler.start()
            
            infer_log = run_command(infer_cmd)
            
            peak_ram = profiler.stop()
            metrics = extract_metrics(infer_log)

            logger.write(f"\nExtracted metrics for {precision.upper()} Batch {bs}:")
            logger.write(f"Latency: {metrics['latency_ms']}")
            logger.write(f"Throughput: {metrics['fps']}")
            logger.write(f"Peak RAM: {peak_ram} MB")

            if metrics["latency_ms"] is not None and metrics["fps"] is not None:
                print(
                    f"{precision.upper():<10} | "
                    f"{bs:<7} | "
                    f"{metrics['latency_ms']:<15.2f} | "
                    f"{metrics['fps']:<18.2f} | "
                    f"{peak_ram}"
                )
            else:
                print(
                    f"{precision.upper():<10} | "
                    f"{bs:<7} | "
                    f"{'FAILED':<15} | "
                    f"{'FAILED':<18} | "
                    f"{peak_ram}"
                )
                logger.write("\nMetric extraction failed. Full trtexec output has been saved above.")

    logger.write("\n" + "=" * 90)
    logger.write("BENCHMARK COMPLETED")
    logger.write(f"Completed: {datetime.now()}")
    logger.write("=" * 90)
    print("=" * 75)
    print(f"Full log saved to: {logger.log_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if logger:
            logger.write("\n" + "=" * 90)
            logger.write("UNHANDLED EXCEPTION")
            logger.write("=" * 90)
            logger.write(str(e))
        print(f"Benchmark failed: {e}")