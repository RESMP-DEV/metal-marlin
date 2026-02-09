from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


class MMFP4ModelLoader:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._parse_index()
        
    def _parse_index(self) -> None:
        """Parse model.safetensors.index.json for tensor->shard mapping."""
        index_path = self.model_path / "model.safetensors.index.json"
        
        if not index_path.exists():
            # Support single-file safetensors if index is missing
            single_file = self.model_path / "model.safetensors"
            if single_file.exists():
                self._tensor_to_shard = {}
                self._layer_to_tensors = defaultdict(list)
                self._shard_handles = {}
                self._register_shard(single_file)
            return
        
        with index_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
        
        weight_map = index_data.get("weight_map", {})
        self._tensor_to_shard = {}
        self._layer_to_tensors = defaultdict(list)
        self._shard_handles = {}
        
        for tensor_name, shard_name in weight_map.items():
            shard_path = self.model_path / shard_name
            self._tensor_to_shard[tensor_name] = shard_path
            
            layer_idx = self._extract_layer_index(tensor_name)
            if layer_idx is not None:
                self._layer_to_tensors[layer_idx].append(tensor_name)
    
    def _register_shard(self, shard_path: Path) -> None:
        """Register all tensors in a single shard."""
        with safe_open(shard_path, framework="pt") as f:
            for name in f.keys():
                self._tensor_to_shard[name] = shard_path
                layer_idx = self._extract_layer_index(name)
                if layer_idx is not None:
                    self._layer_to_tensors[layer_idx].append(name)
    
    def _extract_layer_index(self, name: str) -> int | None:
        """Extract layer index from tensor name using common patterns."""
        patterns = [
            r"layers\.(\d+)\.",
            r"h\.(\d+)\.",
            r"blocks\.(\d+)\.",
        ]
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        return None
    
    def _get_shard_handle(self, shard_path: Path) -> Any:
        if shard_path not in self._shard_handles:
            self._shard_handles[shard_path] = safe_open(shard_path, framework="pt")
        return self._shard_handles[shard_path]
    
    def load_layer(self, layer_idx: int, device: str = "mps") -> dict[str, torch.Tensor]:
        """Load all tensors for a single layer."""
        tensors = {}
        for name in self._layer_to_tensors.get(layer_idx, []):
            shard_path = self._tensor_to_shard[name]
            handle = self._get_shard_handle(shard_path)
            tensor = handle.get_tensor(name)
            
            if device != "cpu":
                if device == "mps" and torch.backends.mps.is_available():
                    tensor = tensor.to("mps")
                elif device.startswith("cuda") and torch.cuda.is_available():
                    tensor = tensor.to(device)
                    
            tensors[name] = tensor
        return tensors
    
    def get_quantized_weight(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (packed_weights, scales) tuple for a quantized weight."""
        # Look for packed weights (uint32 [K, N//8] where 8 FP4 nibbles packed per uint32)
        qweight_suffixes = [".qweight", ".weight", ".packed_weight"]
        scales_suffixes = [".scales", ".weight_scale", ".scales_fp4"]
        
        base_name = name
        for s in qweight_suffixes + scales_suffixes:
            if name.endswith(s):
                base_name = name[:-len(s)]
                break
        
        qweight_key = None
        for s in qweight_suffixes:
            cand = base_name + s
            if cand in self._tensor_to_shard:
                qweight_key = cand
                break
        
        scales_key = None
        for s in scales_suffixes:
            cand = base_name + s
            if cand in self._tensor_to_shard:
                scales_key = cand
                break
            
        if qweight_key is None or scales_key is None:
            raise KeyError(f"Could not find both qweight and scales for {name}")
            
        qweight = self._get_shard_handle(self._tensor_to_shard[qweight_key]).get_tensor(qweight_key)
        scales = self._get_shard_handle(self._tensor_to_shard[scales_key]).get_tensor(scales_key)
            
        return qweight, scales
    
    def __iter__(self) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
        """Iterate over layers for streaming loading."""
        for layer_idx in sorted(self._layer_to_tensors.keys()):
            yield layer_idx, self.load_layer(layer_idx)