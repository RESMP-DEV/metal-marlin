"""HuggingFace integration tests for EXL3 loader."""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from metal_marlin.hf_exl3_loader import (
    detect_exl3_format,
    download_exl3_model,
    list_exl3_models,
    load_exl3_from_hub,
)


class TestFormatDetection:
    """Test EXL3 format detection functionality."""

    @pytest.fixture
    def mock_model_dir(self, tmp_path):
        """Create a mock model directory with config.json."""
        config = {"hidden_size": 2048, "num_hidden_layers": 32}
        (tmp_path / "config.json").write_text(json.dumps(config))
        return tmp_path

    def test_detect_single_file(self, mock_model_dir):
        """Test detection of single-file EXL3 models."""
        (mock_model_dir / "model.safetensors").touch()

        result = detect_exl3_format(mock_model_dir)

        assert result["format"] == "single"
        assert len(result["files"]) == 1
        assert "config" in result
        assert result["config"]["hidden_size"] == 2048
        assert result["config"]["num_hidden_layers"] == 32

    def test_detect_sharded(self, mock_model_dir):
        """Test detection of sharded EXL3 models."""
        for i in range(4):
            (mock_model_dir / f"model-{i + 1:05d}-of-00004.safetensors").touch()

        result = detect_exl3_format(mock_model_dir)

        assert result["format"] == "sharded"
        assert result["num_shards"] == 4
        assert len(result["files"]) == 4
        assert "config" in result

    def test_detect_layerwise(self, mock_model_dir):
        """Test detection of layerwise EXL3 models."""
        for i in range(10):
            layer_dir = mock_model_dir / f"layer_{i:04d}"
            layer_dir.mkdir()
            (layer_dir / "index.json").write_text(json.dumps({"layer_idx": i}))

        result = detect_exl3_format(mock_model_dir)

        assert result["format"] == "layerwise"
        assert result["num_layers"] == 10
        assert len(result["files"]) == 10
        assert "config" in result

    def test_missing_config(self, tmp_path):
        """Test error handling when config.json is missing."""
        with pytest.raises(FileNotFoundError, match="config.json not found"):
            detect_exl3_format(tmp_path)

    def test_empty_directory(self, mock_model_dir):
        """Test detection in directory with only config."""
        result = detect_exl3_format(mock_model_dir)

        # Should not crash, but format detection might be ambiguous
        assert "format" in result
        assert "config" in result


class TestDownloadFunctionality:
    """Test model download functionality."""

    @patch("huggingface_hub.snapshot_download")
    def test_download_exl3_model(self, mock_snapshot, tmp_path):
        """Test downloading EXL3 model from HuggingFace."""
        mock_snapshot.return_value = str(tmp_path / "downloaded_model")

        result = download_exl3_model(
            model_id="test/model",
            local_dir=str(tmp_path / "custom_dir"),
            revision="main",
            token="test_token",
        )

        mock_snapshot.assert_called_once_with(
            repo_id="test/model",
            local_dir=str(tmp_path / "custom_dir"),
            revision="main",
            token="test_token",
            allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
        )
        assert result == tmp_path / "downloaded_model"


class TestModelListing:
    """Test model listing and discovery functionality."""

    @patch("huggingface_hub.list_models")
    def test_list_exl3_models(self, mock_list_models):
        """Test listing EXL3 models from HuggingFace."""
        # Mock the HuggingFace model list response
        mock_model = Mock()
        mock_model.id = "test/Llama-3.1-8B-EXL3"
        mock_model.author = "test"
        mock_model.downloads = 1000
        mock_model.tags = ["exl3", "quantization"]

        mock_list_models.return_value = [mock_model]

        result = list_exl3_models(author="test", search="EXL3", limit=10)

        mock_list_models.assert_called_once_with(
            author="test",
            search="EXL3",
            limit=10,
        )

        assert len(result) == 1
        assert result[0]["id"] == "test/Llama-3.1-8B-EXL3"
        assert result[0]["author"] == "test"
        assert result[0]["downloads"] == 1000
        assert result[0]["tags"] == ["exl3", "quantization"]


class TestModelCardParsing:
    """Test model card parsing functionality."""

    @patch("huggingface_hub.model_info")
    def test_parse_model_card(self, mock_info):
        """Test parsing model card to extract base model."""
        from metal_marlin.hf_exl3_loader import parse_model_card

        mock_model = Mock()
        mock_model.card_data = Mock()
        mock_model.card_data.base_model = "meta-llama/Llama-3.1-8B"
        mock_info.return_value = mock_model

        result = parse_model_card("turboderp/Llama-3.1-8B-EXL3")

        mock_info.assert_called_once_with("turboderp/Llama-3.1-8B-EXL3")
        assert result["base_model"] == "meta-llama/Llama-3.1-8B"

    @patch("huggingface_hub.model_info")
    def test_parse_model_card_no_base_model(self, mock_info):
        """Test parsing model card when no base model is specified."""
        from metal_marlin.hf_exl3_loader import parse_model_card

        mock_model = Mock()
        mock_model.card_data = Mock()
        mock_model.card_data.base_model = None
        mock_info.return_value = mock_model

        result = parse_model_card("test/model")

        assert result["base_model"] is None

    @patch("huggingface_hub.model_info")
    def test_parse_model_card_no_card_data(self, mock_info):
        """Test parsing model card when no card data exists."""
        from metal_marlin.hf_exl3_loader import parse_model_card

        mock_model = Mock()
        mock_model.card_data = None
        mock_info.return_value = mock_model

        result = parse_model_card("test/model")

        assert result["base_model"] is None


@pytest.mark.slow
class TestHuggingFaceIntegration:
    """Integration tests requiring network access."""

    @pytest.mark.skipif(
        not os.environ.get("RUN_HF_TESTS"),
        reason="Set RUN_HF_TESTS=1 to run HuggingFace integration tests",
    )
    def test_find_exl3_models(self):
        """Test finding EXL3 models on HuggingFace Hub."""
        from metal_marlin.hf_exl3_loader import find_exl3_models

        models = find_exl3_models(limit=5)

        assert len(models) > 0
        assert all("id" in m for m in models)
        assert all("author" in m for m in models)

    @pytest.mark.skipif(
        not os.environ.get("RUN_HF_TESTS"),
        reason="Set RUN_HF_TESTS=1 to run HuggingFace integration tests",
    )
    @patch("metal_marlin.hf_exl3_loader.snapshot_download")
    @patch("metal_marlin.hf_exl3_loader.TrellisModelLoader")
    def test_load_exl3_from_hub(self, mock_loader_class, mock_download, tmp_path):
        """Test loading EXL3 model directly from HuggingFace."""
        # Setup mocks
        mock_download.return_value = str(tmp_path / "test_model")

        # Create a mock model directory
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        config = {"hidden_size": 2048, "num_hidden_layers": 32}
        (model_dir / "config.json").write_text(json.dumps(config))

        # Mock TrellisModelLoader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        result = load_exl3_from_hub("test/model", device="mps")

        # Verify download was called
        mock_download.assert_called_once()

        # Verify loader was created
        mock_loader_class.assert_called_once_with(model_dir)

        assert result == mock_loader

    @pytest.mark.skipif(
        not os.environ.get("RUN_HF_TESTS"),
        reason="Set RUN_HF_TESTS=1 to run HuggingFace integration tests",
    )
    def test_list_exl3_models_integration(self):
        """Test actual listing of EXL3 models from HuggingFace."""
        models = list_exl3_models(limit=3)

        assert isinstance(models, list)
        assert len(models) <= 3

        for model in models:
            assert isinstance(model, dict)
            assert "id" in model
            assert "author" in model
            assert "downloads" in model
            assert "tags" in model
            assert isinstance(model["downloads"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
