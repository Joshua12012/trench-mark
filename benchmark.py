import subprocess
import re
import yaml
import threading
import sys
import os

class MemoryProfiler:
    def __init__(self):
        self.running = False
        self.max_ram = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._poll_tegrastats)
        self.thread.start()

    def _poll_tegrastats(self):
        process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, text=True)
        while self.running:
            line = process.stdout.readline()
            if "RAM" in line:
                match = re.search(r'RAM (\d+)/\d+MB', line)
                if match:
                    current_ram = int(match.group(1))
                    self.max_ram = max(self.max_ram, current_ram)
        process.terminate()

    def stop(self):
        self.running = False
        self.thread.join()
        return self.max_ram

def run_command(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return result.stdout

def extract_metrics(log):
    # Updated Regex to handle both ':' and '=' for TensorRT 10 compatibility
    latency = re.search(r'mean\s*[=:]\s*([\d\.]+)\s*ms', log)
    throughput = re.search(r'Throughput\s*[=:]\s*([\d\.]+)\s*qps', log, re.IGNORECASE)
    return {
        "latency_ms": float(latency.group(1)) if latency else None,
        "fps": float(throughput.group(1)) if throughput else None
    }

def main():
    if not os.path.exists("config.yaml"):
        print("Error: config.yaml not found.")
        sys.exit(1)
        
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    onnx_path = config["model"]["onnx_path"]
    input_name = "images" 
    c_h_w = "3x640x640" 
    
    batches = config["benchmarks"]["batch_sizes"]
    min_b, opt_b, max_b = min(batches), batches[len(batches)//2], max(batches)
    
    shape_profile = f"--minShapes={input_name}:{min_b}x{c_h_w} --optShapes={input_name}:{opt_b}x{c_h_w} --maxShapes={input_name}:{max_b}x{c_h_w}"
    
    print(f"Starting Benchmark Pipeline for {config['model']['name']} on TensorRT 10\n" + "="*60)
    print(f"{'Precision':<10} | {'Batch':<7} | {'Latency (ms)':<15} | {'Throughput (FPS)':<18} | {'Peak RAM (MB)'}")
    print("-" * 75)

    for precision in config["benchmarks"]["precisions"]:
        engine_path = f"{config['model']['name']}_{precision}.engine"
        
        # 1. Build Engine
        print(f"-> Compiling {precision.upper()} engine (This can take 5-15 minutes...)")
        build_cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
        build_cmd.extend(shape_profile.split())
        
        if precision == "fp16":
            build_cmd.append("--fp16")
        elif precision == "int8":
            build_cmd.extend(["--int8", "--fp16"])
            
        build_log = run_command(build_cmd)

        # Safety Check: Did the engine actually build?
        if not os.path.exists(engine_path):
            print(f"!!! Failed to compile {precision.upper()} engine. Check ONNX file.")
            continue # Skip inference if there is no engine to run

        for bs in config["benchmarks"]["batch_sizes"]:
            profiler = MemoryProfiler()
            profiler.start()

            # 3. Run Inference
            run_cmd = [
                "trtexec",
                f"--loadEngine={engine_path}",
                f"--shapes={input_name}:{bs}x{c_h_w}", 
                f"--iterations={config['settings']['iterations']}",
                f"--warmUp={config['settings']['warmup']}",
                "--noDataTransfers"
            ]
            log = run_command(run_cmd)

            peak_ram = profiler.stop()
            metrics = extract_metrics(log)

            if metrics['latency_ms'] is None:
                print(f"{precision.upper():<10} | {bs:<7} | {'FAILED':<15} | {'FAILED':<18} | {peak_ram}")
                print("\n" + "!"*50)
                print("TRTEXEC FAILED. Here is the end of the error log:")
                print(log[-1500:]) 
                print("!"*50 + "\n")
            else:
                print(f"{precision.upper():<10} | {bs:<7} | {metrics['latency_ms']:<15.2f} | {metrics['fps']:<18.2f} | {peak_ram}")

if __name__ == "__main__":
    main()