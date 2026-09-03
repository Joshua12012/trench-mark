# Trench-Mark

[![PyPI version](https://img.shields.io/pypi/v/trench-mark.svg)](https://pypi.org/project/trench-mark/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson-green.svg)](https://developer.nvidia.com/embedded-computing)

Trench-Mark is an automated, hardware-safe TensorRT benchmarking and deployment CLI designed specifically for resource-constrained NVIDIA Jetson edge devices (Orin Nano, Orin NX, Xavier, AGX).

The tool automates the evaluation loop: compiling hardware-fused TensorRT engines without triggering Out-Of-Memory (OOM) crashes, and collecting microsecond-accurate GPU execution telemetry alongside peak unified memory consumption.

---

## Key Capabilities

* **OOM-Safe Engine Compilation:** Caps workspace limits and redirects intermediate build artifacts directly to physical disk storage, preventing the Linux kernel from killing processes on unified memory hardware.
* **Pure Compute Telemetry:** Uses native CUDA hardware events with isolated PCIe data transfers to measure pure GPU execution time without host-to-device transfer noise.
* **Hardware Telemetry:** Polls the Tegra kernel interface asynchronously in a dedicated monitoring thread to log real-time unified memory spikes.
* **True Throughput Metrics:** Automatically translates raw TensorRT Queries Per Second (QPS) into True Frame Throughput (FPS = QPS * Batch Size).

---

## Installation

### From PyPI

```bash
pip install trench_mark
```

### System Prerequisites

* NVIDIA Jetson running JetPack 5.x or JetPack 6.x
* TensorRT and `trtexec` installed and accessible in system `$PATH`
* Python 3.8+
* `ultralytics` package (for exporting YOLO models)

---

## Quick Start

### Step 1: Export the YOLO Model to ONNX

Before running Trench-Mark, you must export your YOLO model to the ONNX format with dynamic batching enabled. Run the following command in your terminal:

```bash
yolo export model=yolov8n.pt format=onnx opset=12 dynamic=True
```

### Step 2: Run the Benchmark

Once the `.onnx` file is generated, pass it to Trench-Mark. Profile the model across batch sizes 1, 4, and 8 for FP32, FP16, and INT8:

```bash
sudo trench-mark -m yolov8n.onnx -b 1 4 8 -p fp32 fp16 int8
```

Running with `sudo` is recommended so the background monitoring thread has permission to poll the `tegrastats` kernel interface.

---

## CLI Reference

```text
trench-mark [-h] -m MODEL [-s INPUT_SHAPE] [--input-name INPUT_NAME]
            [-p {fp32,fp16,int8} [{fp32,fp16,int8} ...]]
            [-b BATCH_SIZES [BATCH_SIZES ...]] [-i ITERATIONS]
            [-w WARMUP] [--workspace WORKSPACE]
```

### Parameter Breakdown

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `-m`, `--model` | `str` | **Required** | Local path to the exported `.onnx` file. |
| `-s`, `--input-shape` | `str` | `3x640x640` | Single image tensor dimensions in Channel x Height x Width format (`CHW`). |
| `--input-name` | `str` | `images` | Name of the primary input node inside the ONNX graph. Ultralytics YOLO models default to `images`. |
| `-p`, `--precisions` | `list` | `fp32 fp16 int8` | Precision modes to compile and test. Supported values: `fp32`, `fp16`, `int8`. |
| `-b`, `--batch-sizes` | `int` | `1 4 8` | Space-separated list of dynamic batch sizes to evaluate sequentially. |
| `-i`, `--iterations` | `int` | `500` | Number of execution cycles per test to ensure statistical stability. |
| `-w`, `--warmup` | `int` | `50` | Hardware pre-conditioning window in milliseconds to stabilize GPU clock speeds. |
| `--workspace` | `int` | `2048` | Maximum memory pool allocated to TensorRT tactic selection in megabytes (MB). |

---

## Common Use Cases

### 1. High-Resolution Model Evaluation

To test higher input resolutions (such as 1280x1280), make sure to export your model at that resolution first, then run:

```bash
sudo trench-mark -m yolov8s_highres.onnx -s 3x1280x1280 -b 1 2 -p fp16
```

### 2. Rapid Single-Frame Latency Testing

To verify real-time response time on batch size 1 with fewer iterations:

```bash
sudo trench-mark -m yolov8n.onnx -b 1 -p fp16 int8 -i 200
```

---

## Understanding Output Metrics

```text
Precision  | Batch   | Latency (ms)    | Throughput (FPS)   | Peak RAM (MB)
---------------------------------------------------------------------------
FP16       | 1       | 6.79            | 147.20             | 5328
FP16       | 4       | 24.88           | 160.76             | 5497
FP16       | 8       | 47.77           | 167.44             | 5345
```

* **Latency (ms):** The mean hardware compute time required to execute the entire batch on the GPU. Batch 1 represents your single-frame reaction time.
* **Throughput (FPS):** Represents True Frames Per Second, calculated as `QPS * Batch Size`. Increasing batch size allows Tensor Cores to process multiple images per kernel launch, improving overall pipeline throughput at the expense of per-batch latency.
* **Peak RAM (MB):** The maximum unified memory footprint captured by `tegrastats`. TensorRT INT8 and FP16 tactics often allocate larger scratchpad memory pools to accelerate matrix operations on Tensor Cores.

---

## Memory Safety and Jetson Optimization

1. **Workspace Boundary:** The `--workspace 2048` flag restricts the builder memory pool to 2GB. This prevents TensorRT from consuming all unified RAM during tactic autotuning and crashing the host Linux system.
2. **Temporary Disk Allocation:** Engine compilation directs temporary intermediate files directly to disk rather than holding them in memory.
3. **Swap Memory Recommended:** When building larger models on 8GB Orin Nano boards, verify you have configured an SSD swap file:

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

4. **Execution Audit Logs:** Every run writes a full trace containing system metadata, environment flags, and raw command logs to `./logs/benchmark_<TIMESTAMP>.log`.

---

## Troubleshooting

### Corrupted 0-Byte Engines

If an engine build aborts unexpectedly, TensorRT may leave behind an empty 0-byte file. Trench-Mark checks file sizes before benchmarking and marks corrupted engines for recompilation. To manually clean your engine cache, run:

```bash
rm -rf models/*.engine trt.cache
```

### Shape Profile Violations

If a test fails with shape dimension errors, the requested batch size exceeds the upper limit built into the cached engine. Delete the older engine file or use the automatic versioned naming convention so a new profile covering the higher batch size can be compiled.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for full terms.