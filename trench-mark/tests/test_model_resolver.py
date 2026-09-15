"""
test_model_resolver.py
======================
Unit tests for ModelResolver, graph inspector, and model zoo catalog.
"""

import sys
import unittest
from pathlib import Path

# Add src directory to python search path
src_dir = Path(__file__).resolve().parent.parent / "src"
if not src_dir.exists():
    src_dir = Path(__file__).resolve().parent.parent / "trench-mark" / "src"
sys.path.insert(0, str(src_dir))

from trench_mark.model_resolver import (
    ModelResolver,
    format_catalog_help,
    get_model_catalog_summary,
    inspect_onnx_graph,
    resolve_model,
)


class TestModelResolver(unittest.TestCase):

    def test_catalog_summary(self):
        """Verifies that the catalog includes popular models and formats help text."""
        catalog = get_model_catalog_summary()
        self.assertIn("YOLO Object Detection", catalog)
        self.assertIn("Standard CNN Backbones (Torchvision)", catalog)
        self.assertIn("resnet50", catalog["Standard CNN Backbones (Torchvision)"])
        self.assertIn("yolov8n", catalog["YOLO Object Detection"])

        help_text = format_catalog_help()
        self.assertIn("Available Model Zoo", help_text)
        self.assertIn("yolov8n", help_text)
        self.assertIn("resnet50", help_text)

    def test_existing_onnx_inspection(self):
        """Tests that inspect_onnx_graph extracts correct tensor names and shapes."""
        onnx_file = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"
        if onnx_file.exists():
            meta = inspect_onnx_graph(str(onnx_file))
            self.assertEqual(meta.model_name, "yolov8n")
            self.assertEqual(meta.input_name, "images")
            self.assertTrue("640" in meta.input_shape)
            self.assertTrue(meta.is_dynamic_batch)

    def test_resolver_with_existing_onnx(self):
        """Tests that ModelResolver resolves an existing local ONNX model file."""
        onnx_file = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"
        if onnx_file.exists():
            resolver = ModelResolver(models_dir=str(onnx_file.parent))
            meta = resolver.resolve(str(onnx_file))
            self.assertEqual(meta.model_name, "yolov8n")
            self.assertEqual(meta.input_name, "images")


if __name__ == "__main__":
    unittest.main()
