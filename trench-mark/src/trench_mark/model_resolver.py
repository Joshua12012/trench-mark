"""
model_resolver.py
=================
This module handles autonomous model ingestion and ONNX conversion:
1. Takes a model name (e.g. 'yolov8n', 'resnet50') or local model file (.onnx, .pt, .tflite, .h5).
2. Automatically downloads weights into a local 'models/' directory.
3. Automatically exports/converts the model to ONNX with dynamic batching enabled.
4. Inspects the ONNX graph to discover input tensor name (e.g. 'images') and nominal CHW shape (e.g. '3x640x640').
5. Provides the Model Zoo catalog displayed in 'trench-mark --help'.
"""

import ast
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple
import onnx
import torch


# ==============================================================================
# 1. MODEL CATALOG (Displayed in trench-mark --help)
# ==============================================================================
MODEL_CATALOG: Dict[str, List[str]] = {
    "YOLO Object Detection": [
        "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
        "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
        "yolov9t", "yolov9s", "yolov9m", "yolov9c",
        "yolov10n", "yolov10s", "yolov10m",
        "yolov8n-seg", "yolov8n-pose", "yolov8n-cls", "rtdetr-l",
    ],
    "Standard CNN Backbones (Torchvision)": [
        "resnet18", "resnet50", "resnet101",
        "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
        "efficientnet_b0", "convnext_tiny", "vit_b_16",
    ],
    "Supported File Formats": [
        ".onnx (direct inference)",
        ".pt (PyTorch checkpoint & YOLO weights)",
        ".torchscript (TorchScript binary)",
        ".tflite (TensorFlow Lite)",
        ".tf / SavedModel (TensorFlow)",
        ".h5 / .keras (Keras)",
    ],
}


def get_model_zoo_help() -> str:
    """
    Formats the available Model Zoo models into a clear terminal text block
    to be displayed at the end of 'trench-mark --help'.
    """
    lines = ["\nAvailable Model Zoo (Automatic Download & Ingestion):"]
    for category, models in MODEL_CATALOG.items():
        lines.append(f"\n  {category}:")
        lines.append(f"    {', '.join(models)}")
    return "\n".join(lines)


# Backwards compatibility helper
format_catalog_help = get_model_zoo_help
get_model_catalog_summary = lambda: MODEL_CATALOG


# ==============================================================================
# 2. MODEL METADATA & GRAPH INSPECTION
# ==============================================================================
class ModelMetadata:
    """
    Stores metadata discovered from an ONNX model file.
    
    Attributes:
        model_name: Base stem of the model (e.g. 'yolov8n').
        onnx_path: Absolute or relative path to the resolved .onnx file.
        input_name: The name of the input node expected by TensorRT (e.g. 'images').
        input_shape: The nominal channel-height-width dimensions (e.g. '3x640x640').
        is_dynamic_batch: True if the batch dimension is dynamic (allows -b 1 4 8).
    """
    def __init__(
        self,
        model_name: str,
        onnx_path: str,
        input_name: str,
        input_shape: str,
        is_dynamic_batch: bool = True,
    ):
        self.model_name = model_name
        self.onnx_path = onnx_path
        self.input_name = input_name
        self.input_shape = input_shape
        self.is_dynamic_batch = is_dynamic_batch

    def __repr__(self) -> str:
        return (
            f"ModelMetadata(name='{self.model_name}', onnx='{self.onnx_path}', "
            f"input='{self.input_name}', shape='{self.input_shape}', dynamic={self.is_dynamic_batch})"
        )


def inspect_onnx_graph(onnx_path: str) -> ModelMetadata:
    """
    Inspects an ONNX model file using the standard 'onnx' library.
    
    Finds:
    1. Input tensor name (filters out weights/biases initializers).
    2. Channels, Height, Width (CHW).
    3. Dynamic batch status (checks if batch dimension is dynamic or -1).
    """
    model = onnx.load(onnx_path)
    graph = model.graph
    stem = Path(onnx_path).stem.lower()

    # Initializers store constants, weights, and biases.
    # We filter them out so only the true camera/data input is picked.
    initializer_names = {init.name for init in graph.initializer}
    real_inputs = [inp for inp in graph.input if inp.name not in initializer_names]

    if not real_inputs:
        return ModelMetadata(stem, onnx_path, "images", "3x640x640", True)

    primary_input = real_inputs[0]
    input_name = primary_input.name

    # Check if Ultralytics embedded original image size in metadata_props
    metadata_props = {prop.key: prop.value for prop in model.metadata_props}
    default_chw = "3x640x640"
    is_dynamic = True

    # Read tensor dimensions
    tensor_type = primary_input.type.tensor_type
    raw_dims: List[int] = []
    if tensor_type.HasField("shape"):
        for i, dim in enumerate(tensor_type.shape.dim):
            if dim.HasField("dim_value") and dim.dim_value > 0:
                raw_dims.append(dim.dim_value)
                if i == 0:
                    is_dynamic = False  # Explicit fixed batch dimension
            else:
                raw_dims.append(-1)
                if i == 0:
                    is_dynamic = True

    # For standard 4D image input: [batch, channels, height, width]
    if len(raw_dims) == 4:
        c = raw_dims[1] if raw_dims[1] > 0 else 3
        h = raw_dims[2]
        w = raw_dims[3]

        # If height or width are dynamic, check embedded metadata or default to 640
        if h <= 0 or w <= 0:
            if "imgsz" in metadata_props:
                try:
                    val = ast.literal_eval(metadata_props["imgsz"])
                    h, w = (val[0], val[1]) if isinstance(val, (list, tuple)) else (val, val)
                except Exception:
                    h, w = 640, 640
            else:
                h, w = 640, 640

        default_chw = f"{c}x{h}x{w}"

    return ModelMetadata(stem, onnx_path, input_name, default_chw, is_dynamic)


def get_onnx_input_details(onnx_path: str) -> Tuple[str, str]:
    """
    Convenience function returning (input_name, input_shape) for an ONNX file.
    """
    meta = inspect_onnx_graph(onnx_path)
    return meta.input_name, meta.input_shape


# ==============================================================================
# 3. AUTONOMOUS MODEL RESOLVER
# ==============================================================================
def resolve_model(
    model_input: str,
    models_dir: str = "models",
    user_shape: Optional[str] = None,
    user_input_name: Optional[str] = None,
) -> Tuple[str, str, str, str]:
    """
    Resolves, downloads, or converts any model input into an ONNX file.

    Parameters:
        model_input: Model name (e.g. 'yolov8n', 'resnet50') or file path (.onnx, .pt, .tflite).
        models_dir: Directory where models are downloaded and cached (default: 'models').
        user_shape: Optional manual override for CHW shape (e.g. '3x640x640').
        user_input_name: Optional manual override for input node name (e.g. 'images').

    Returns:
        (model_name, onnx_file_path, input_name, input_shape)
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    input_path = Path(model_input)
    model_stem = input_path.stem.lower()

    # Case A: User passed an existing .onnx file directly
    if model_input.endswith(".onnx") and input_path.exists():
        onnx_file = str(input_path.resolve())
        meta = inspect_onnx_graph(onnx_file)
        return (
            model_stem,
            onnx_file,
            user_input_name or meta.input_name,
            user_shape or meta.input_shape,
        )

    target_onnx = models_path / f"{model_stem}.onnx"

    # Case B: Model was previously downloaded and converted in models_dir
    if target_onnx.exists() and target_onnx.stat().st_size > 0:
        meta = inspect_onnx_graph(str(target_onnx))
        return (
            model_stem,
            str(target_onnx),
            user_input_name or meta.input_name,
            user_shape or meta.input_shape,
        )

    # Case C: Ultralytics YOLO models (e.g. yolov8n, yolo11s, rtdetr-l) or .pt files
    is_yolo = any(model_stem.startswith(prefix) for prefix in ("yolo", "rtdetr", "fastsam", "sam"))
    if is_yolo or (input_path.exists() and input_path.suffix == ".pt"):
        from ultralytics import YOLO

        target_pt = models_path / f"{model_stem}.pt"
        load_source = str(input_path) if input_path.exists() else (str(target_pt) if target_pt.exists() else model_stem)

        print(f"📦 [Model Resolver] Loading/Downloading YOLO model weights: '{model_stem}'...")
        model = YOLO(load_source)

        # Move downloaded checkpoint into the user's models/ directory
        local_pt = Path(f"{model_stem}.pt")
        if local_pt.exists() and local_pt.resolve() != target_pt.resolve():
            shutil.move(str(local_pt), str(target_pt))

        print(f"⚙️  [Model Resolver] Exporting '{model_stem}' to ONNX (dynamic batch enabled)...")
        # Ultralytics export generates dynamic-batch ONNX
        exported = model.export(
            format="onnx",
            dynamic=True,      # Dynamic batch support: enables benchmarking across batches 1, 4, 8
            simplify=True,     # Constant folding and dead-node cleanup with onnxslim
            opset=12,          # Broad compatibility across all TensorRT versions
            device="cpu",      # Perform export on CPU to avoid allocating GPU VRAM
            half=False,        # Baseline export in FP32
        )

        exported_path = Path(exported)
        if exported_path.resolve() != target_onnx.resolve():
            shutil.move(str(exported_path), str(target_onnx))

        meta = inspect_onnx_graph(str(target_onnx))
        return (
            model_stem,
            str(target_onnx),
            user_input_name or meta.input_name,
            user_shape or meta.input_shape,
        )

    # Case D: Torchvision models (e.g. resnet18, resnet50, mobilenet_v3_small)
    try:
        import torchvision.models as tvm
        torchvision_models = set(tvm.list_models())
    except Exception:
        torchvision_models = set()

    if model_stem in torchvision_models:
        import torchvision.models as tvm

        print(f"📦 [Model Resolver] Downloading Torchvision model: '{model_stem}'...")
        try:
            model = tvm.get_model(model_stem, weights="DEFAULT")
        except Exception:
            model = tvm.get_model(model_stem, pretrained=True)
        model.eval()

        # Determine nominal dimensions (default 3x224x224 for standard CNNs)
        chw = user_shape or ("3x299x299" if "inception" in model_stem else "3x224x224")
        c, h, w = [int(x) for x in chw.split("x")[-3:]]
        dummy_input = torch.randn(1, c, h, w)

        print(f"⚙️  [Model Resolver] Exporting '{model_stem}' to ONNX (dynamic batch enabled)...")
        input_node = user_input_name or "input"
        # Using dynamo=False ensures legacy TorchScript ONNX exporter is used (PyTorch 2.x+)
        torch.onnx.export(
            model,
            dummy_input,
            str(target_onnx),
            dynamo=False,
            export_params=True,
            opset_version=12,
            input_names=[input_node],
            output_names=["output"],
            dynamic_axes={input_node: {0: "batch"}, "output": {0: "batch"}},
        )

        meta = inspect_onnx_graph(str(target_onnx))
        return (
            model_stem,
            str(target_onnx),
            user_input_name or meta.input_name,
            user_shape or meta.input_shape,
        )

    # Case E: TensorFlow / TFLite / Keras models
    if input_path.suffix in [".tflite", ".h5", ".keras", ".tf"]:
        print(f"📦 [Model Resolver] Converting TensorFlow file '{input_path}' to ONNX...")
        try:
            import tf2onnx
            if input_path.suffix == ".tflite":
                import subprocess
                cmd = ["python", "-m", "tf2onnx.convert", "--tflite", str(input_path), "--output", str(target_onnx), "--opset", "13"]
                subprocess.run(cmd, check=True)
            else:
                import tensorflow as tf
                m = tf.keras.models.load_model(str(input_path))
                tf2onnx.convert.from_keras(m, output_path=str(target_onnx), opset=13)
        except ImportError:
            raise ImportError(
                "TensorFlow / TFLite conversion requires 'tf2onnx'. Install via: pip install tf2onnx"
            )

        meta = inspect_onnx_graph(str(target_onnx))
        return (
            model_stem,
            str(target_onnx),
            user_input_name or meta.input_name,
            user_shape or meta.input_shape,
        )

    # If model is not recognized
    raise ValueError(
        f"Could not recognize model: '{model_input}'.\n"
        f"You can provide a model name like 'yolov8n' or 'resnet50', or pass a file (.onnx, .pt, .tflite).\n"
        f"Run 'trench-mark --help' to see all available models in the zoo."
    )


class ModelResolver:
    """
    Object-oriented wrapper around resolve_model for clean API usage.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir

    def resolve(
        self,
        model_input: str,
        user_shape: Optional[str] = None,
        user_input_name: Optional[str] = None,
    ) -> ModelMetadata:
        name, onnx_file, in_name, in_shape = resolve_model(
            model_input=model_input,
            models_dir=self.models_dir,
            user_shape=user_shape,
            user_input_name=user_input_name,
        )
        return ModelMetadata(
            model_name=name,
            onnx_path=onnx_file,
            input_name=in_name,
            input_shape=in_shape,
            is_dynamic_batch=True,
        )


def convert_onnx_to_fp16(src_onnx: str, dst_onnx: str) -> str:
    """
    Converts an FP32 ONNX graph into an FP16 strongly-typed ONNX graph.
    
    This is required for TensorRT 11+, where NVIDIA removed the --fp16 CLI flag
    in favor of strictly strongly-typed ONNX models.
    """
    dst_path = Path(dst_onnx)
    if dst_path.exists() and dst_path.stat().st_size > 0:
        # Self-healing: Check if existing FP16 model has mismatched Cast ops from previous conversions
        try:
            m_check = onnx.load(str(dst_path))
            repaired = False
            for node in m_check.graph.node:
                if node.op_type == "Cast":
                    for attr in node.attribute:
                        if attr.name == "to" and attr.i == 1:
                            repaired = True
                            attr.i = 10
            if repaired:
                print(f"🔧 [Model Resolver] Repaired mismatched Cast nodes in existing '{dst_path.name}'.")
                onnx.save(m_check, str(dst_path))
            return str(dst_path.resolve())
        except Exception:
            return str(dst_path.resolve())

    src_path = Path(src_onnx)

    # Strategy 1 (Preferred for YOLO): Native Framework Half Export
    # Native YOLO export guarantees 100% type consistency across elementwise layers
    # and properly sets Cast nodes to FLOAT16 without Float vs Half type conflicts.
    pt_candidate = src_path.with_suffix(".pt")
    if not pt_candidate.exists():
        for cand in [src_path.parent / f"{src_path.stem}.pt", Path(f"{src_path.stem}.pt")]:
            if cand.exists():
                pt_candidate = cand
                break

    if pt_candidate.exists():
        try:
            from ultralytics import YOLO
            print(f"⚙️  [Model Resolver] Exporting '{pt_candidate.name}' directly to FP16 ONNX via Ultralytics...")
            model = YOLO(str(pt_candidate))
            exported = model.export(
                format="onnx",
                dynamic=True,
                simplify=True,
                opset=12,
                device="cpu",
                half=True,
            )
            exported_path = Path(exported)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if exported_path.resolve() != dst_path.resolve():
                shutil.move(str(exported_path), str(dst_path))
            return str(dst_path.resolve())
        except Exception as e:
            print(f"⚠️  [Model Resolver] YOLO direct FP16 export note: {e}")

    # Strategy 2: Standard ONNX FP16 conversion via onnxconverter-common
    # with post-processing type patching for ElementWise operations
    try:
        from onnxconverter_common import float16
        print(f"⚙️  [Model Resolver] Converting ONNX graph to FP16 via onnxconverter-common...")
        model = onnx.load(src_onnx)
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)

        # Post-Processing: Fix type inconsistencies for TensorRT 11 strongly-typed parser.
        # In graphs with Cast ops (e.g. Range -> Cast), onnxconverter-common often leaves
        # target attribute 'to' as FLOAT (1). When feeding into an ElementWise Add with Half,
        # TRT 11 throws Error Code 4. We patch all Cast to=FLOAT (1) -> to=FLOAT16 (10).
        for node in model_fp16.graph.node:
            if node.op_type == "Cast":
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == 1:
                        attr.i = 10

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model_fp16, str(dst_path))
        return str(dst_path.resolve())
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  [Model Resolver] onnxconverter-common conversion note: {e}")

    # Fallback to source ONNX if conversion is not possible
    return src_onnx

