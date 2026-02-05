import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_metal_marlin_module(module_name: str) -> ModuleType:
    pkg_root = Path(__file__).resolve().parents[1] / "metal_marlin"
    pkg_name = "metal_marlin"
    if pkg_name not in sys.modules:
        pkg = ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_root)]
        sys.modules[pkg_name] = pkg
    if "." in module_name:
        parts = module_name.split(".")
        parent_pkg = pkg_name
        parent_path = pkg_root
        for part in parts[:-1]:
            parent_pkg = f"{parent_pkg}.{part}"
            parent_path = parent_path / part
            if parent_pkg not in sys.modules:
                pkg = ModuleType(parent_pkg)
                pkg.__path__ = [str(parent_path)]
                sys.modules[parent_pkg] = pkg

    module_path = pkg_root / f"{module_name.replace('.', '/')}.py"
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTrellisV3Format:
    def test_detect_v3_format(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
        hf_loader = _load_metal_marlin_module("hf_loader")
        assert hf_loader.detect_trellis_format(tmp_path) == "v3"

    def test_detect_v2_format(self, tmp_path: Path) -> None:
        (tmp_path / "layer_0000").mkdir()
        hf_loader = _load_metal_marlin_module("hf_loader")
        assert hf_loader.detect_trellis_format(tmp_path) == "v2"

    def test_shard_writer_creates_index(self, tmp_path: Path) -> None:
        shard_writer = _load_metal_marlin_module("trellis.shard_writer")
        writer = shard_writer.ShardWriter(tmp_path, max_shard_size_gb=1.0)
        writer.add_tensor(
            "model.layers.0.mlp.down_proj.weight",
            indices=np.zeros((2, 2), dtype=np.uint8),
            scales=np.ones((2, 2), dtype=np.float32),
            su=np.ones((2,), dtype=np.float32),
            sv=np.ones((2,), dtype=np.float32),
            bits=4,
            shape=(2, 2),
            mse=0.0,
        )
        writer.finalize()

        index_path = tmp_path / "trellis_index.json"
        assert index_path.exists()
        index = json.loads(index_path.read_text())
        assert index["num_shards"] == 1
        assert index["tensors"][0]["name"] == "model.layers.0.mlp.down_proj.weight"

    def test_shard_writer_respects_size_limit(self, tmp_path: Path) -> None:
        shard_writer = _load_metal_marlin_module("trellis.shard_writer")
        writer = shard_writer.ShardWriter(tmp_path, max_shard_size_gb=1e-6)
        tensor = np.ones((512,), dtype=np.uint8)
        for idx in range(2):
            writer.add_tensor(
                f"model.layers.{idx}.mlp.down_proj.weight",
                indices=tensor,
                scales=np.ones((16,), dtype=np.float32),
                su=np.ones((16,), dtype=np.float32),
                sv=np.ones((16,), dtype=np.float32),
                bits=4,
                shape=(16, 16),
                mse=0.0,
            )
        writer.finalize()

        index = json.loads((tmp_path / "trellis_index.json").read_text())
        assert index["num_shards"] == 2
        assert (tmp_path / "trellis_shard_00000.safetensors").exists()
        assert (tmp_path / "trellis_shard_00001.safetensors").exists()

    def test_load_v3_weights_round_trip(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")
        pytest.importorskip("safetensors")

        hf_loader = _load_metal_marlin_module("hf_loader")
        base_name = "model.layers.0.mlp.down_proj.weight"
        shard_name = "model-00001-of-00001.safetensors"

        weight_map = {
            f"{base_name}.indices": shard_name,
            f"{base_name}.scales": shard_name,
            f"{base_name}.su": shard_name,
            f"{base_name}.sv": shard_name,
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (tmp_path / "quantization_config.json").write_text(
            json.dumps({"tensors": [{"name": base_name, "bits": 4, "shape": [2, 3]}]})
        )

        tensors = {
            f"{base_name}.indices": torch.zeros((1, 1), dtype=torch.int32),
            f"{base_name}.scales": torch.ones((1, 3), dtype=torch.float32),
            f"{base_name}.su": torch.ones((2,), dtype=torch.float32),
            f"{base_name}.sv": torch.ones((3,), dtype=torch.float32),
        }
        from safetensors.torch import save_file

        save_file(tensors, str(tmp_path / shard_name))

        weights = hf_loader.load_trellis_v3_weights(tmp_path, device="cpu")
        assert base_name in weights
        weight = weights[base_name]
        assert weight.original_shape == (2, 3)
