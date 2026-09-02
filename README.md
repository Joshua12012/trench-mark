# TensorRT Edge Benchmarker

An automated, memory-safe performance profiling framework for Jetson edge devices. This pipeline evaluates Neural Network latency, throughput, and hardware memory consumption across FP32, FP16, and INT8 TensorRT optimization levels.

## Features
* **Jetson Memory Safety:** Hardcoded `trtexec` constraints (`--tempdir`, `--memPoolSize`) to prevent `tmpfs` RAM exhaustion during intensive INT8 graph compilation.
* **TensorRT 10 Compatible:** Implements the modern `--shapes` flag and dynamic batching profiles, replacing legacy static batch arguments.
* **Hardware Telemetry:** Spawns an asynchronous thread to poll NVIDIA `tegrastats`, capturing true peak Unified Memory (VRAM + RAM) consumption.

## Prerequisites
* NVIDIA Jetson Orin Nano (or equivalent Tegra device)
* JetPack SDK (TensorRT 10.x + CUDA)
* Python 3.8+

## Usage
1. Define your evaluation matrix in `config.yaml`.
2. Compile the standalone binary:
   ```bash
   pyinstaller build.spec


3. Execute with root privileges (required for tegrastats hardware access):

```Bash
sudo ./dist/trt-benchmarker
```

### Implementation Commands
To build and execute this framework on a fresh Jetson Orin Nano, run these terminal commands sequentially. 

```bash
# 1. Create the project directory
mkdir -p ~/trt_benchmark && cd ~/trt_benchmark

# 2. Maximize Jetson clock speeds to prevent thermal throttling
sudo nvpmodel -m 0
sudo jetson_clocks

# 3. Install Python packaging dependencies and Ultralytics
sudo apt update
sudo apt install python3-pip -y
pip3 install pyinstaller pyyaml ultralytics

# 4. Export the ONNX model with dynamic batching enabled
yolo export model=yolov8n.pt format=onnx opset=12 dynamic=True

# 5. Create benchmark.py, config.yaml, and build.spec in this folder
# (Use nano, vim, or scp to transfer the files here)

# 6. Compile the Python script into a standalone Linux ELF binary
pyinstaller build.spec

# 7. Run the benchmark pipeline
sudo ./dist/trt-benchmarker