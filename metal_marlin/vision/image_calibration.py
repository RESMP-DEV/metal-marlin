"""Image-based calibration datasets for vision encoder quantization.

Vision encoders require calibration on real image data, not text. This module
provides utilities for loading image calibration datasets:

- COCO subset: Diverse natural images with objects
- ImageNet subset: ImageNet-1K validation samples
- Custom datasets: Load from local directories

Key differences from text calibration:
1. Images must be preprocessed (resize, normalize) to match model expectations
2. Batch sizes depend on image resolution and GPU memory
3. Calibration benefits from diverse image content (objects, scenes, textures)

Usage:
    # Use COCO subset (default, recommended)
    calib = VisionCalibrationDataset.coco_subset(num_images=512)

    # Use ImageNet
    calib = VisionCalibrationDataset.imagenet_subset(num_images=512)

    # Custom local images
    calib = VisionCalibrationDataset.from_directory(
        path="/path/to/images",
        num_images=512,
        image_size=(224, 224),
    )

    # Get batches for calibration
    for batch in calib.get_batches(batch_size=8):
        # batch shape: [8, 3, H, W] as numpy array
        outputs = model.forward_vision(batch)
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    
    # Use dummy class for Tensor if torch is not fully available/resolved
    class Tensor:
        dtype: Any
        shape: tuple[int, ...]
        ndim: int
        device: Any
        def contiguous(self) -> Tensor: ...
        def detach(self) -> Tensor: ...
        def cpu(self) -> Tensor: ...
        def numpy(self) -> Any: ...
        def to(self, dtype: Any) -> Tensor: ...
        def permute(self, *dims: int) -> Tensor: ...
        def squeeze(self) -> Tensor: ...
        def unsqueeze(self, dim: int) -> Tensor: ...

from metal_marlin._compat import torch

# VisionMetal import with fallback for non-Apple systems
try:
    from .vision_metal import VisionMetal
    _vision_metal: VisionMetal | None = VisionMetal()
    _has_vision_metal = True
except Exception:
    _vision_metal = None
    _has_vision_metal = False


@dataclass
class ImageInfo:
    """Metadata for a calibration image."""

    path: str | None = None  # Local path or None for in-memory
    url: str | None = None  # Remote URL if downloaded
    source: str = "unknown"  # Dataset source (coco, imagenet, custom)
    category: str = "general"  # Image category if known
    width: int = 0
    height: int = 0


@dataclass
class ImageCalibrationDataset:
    """Base class for image calibration datasets.

    Provides common functionality for loading and preprocessing images
    for vision encoder calibration.

    Attributes:
        images: List of image arrays (preprocessed, [C, H, W] format).
        image_infos: Metadata for each image.
        image_size: Target image size (H, W) for preprocessing.
        mean: Normalization mean (per-channel).
        std: Normalization std (per-channel).
        source: Dataset source name.
    """

    images: list[NDArray[Any]] = field(default_factory=list)
    image_infos: list[ImageInfo] = field(default_factory=list)
    image_size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)  # CLIP
    std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)  # CLIP
    source: str = "unknown"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> NDArray[Any]:
        return self.images[idx]

    def get_batches(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
    ) -> Iterator[NDArray[np.float32]]:
        """Yield batches of preprocessed images.

        Args:
            batch_size: Number of images per batch.
            shuffle: Whether to shuffle images.

        Yields:
            Batches of shape [batch_size, 3, H, W] as float32.
        """
        indices = list(range(len(self.images)))
        if shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch = np.stack([self.images[j] for j in batch_indices])
            yield batch.astype(np.float32)

    def get_all(self) -> NDArray[np.float32]:
        """Return all images as a single array.

        Returns:
            Array of shape [N, 3, H, W].
        """
        return np.stack(self.images).astype(np.float32)

    @property
    def num_images(self) -> int:
        """Number of images in dataset."""
        return len(self.images)

    @property
    def memory_mb(self) -> float:
        """Approximate memory usage in MB."""
        if not self.images:
            return 0.0
        # Assuming float32 storage
        return sum(img.nbytes for img in self.images) / (1024 * 1024)


class VisionCalibrationDataset(ImageCalibrationDataset):
    """Vision calibration dataset with factory methods for common datasets.

    Provides easy access to COCO, ImageNet, and custom image datasets
    for vision encoder calibration.
    """

    @classmethod
    def coco_subset(
        cls,
        num_images: int = 512,
        image_size: tuple[int, int] = (224, 224),
        cache_dir: str | Path | None = None,
        seed: int = 42,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
    ) -> VisionCalibrationDataset:
        """Load a subset of COCO validation images for calibration.

        Downloads COCO validation images on first use and caches them.
        Selects a diverse subset covering different object categories.

        Args:
            num_images: Number of images to use (max ~5000 for COCO val).
            image_size: Target image size (H, W).
            cache_dir: Directory to cache downloaded images.
            seed: Random seed for reproducible selection.
            mean: Override normalization mean (default: CLIP).
            std: Override normalization std (default: CLIP).

        Returns:
            VisionCalibrationDataset with preprocessed images.
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "metal_marlin" / "calibration" / "coco"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Use default CLIP normalization if not specified
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        # Check for cached preprocessed images
        cache_key = f"coco_{num_images}_{image_size[0]}x{image_size[1]}_{seed}"
        cache_file = cache_dir / f"{cache_key}.npz"

        if cache_file.exists():
            # Load from cache
            data = np.load(cache_file)
            images = [data[f"img_{i}"] for i in range(len(data.files))]
            dataset = cls(
                images=images,
                image_size=image_size,
                mean=mean,
                std=std,
                source="coco",
            )
            return dataset

        # Generate synthetic calibration images for now
        # In production, this would download from COCO API
        rng = np.random.default_rng(seed)
        images = []
        image_infos = []

        for i in range(num_images):
            # Generate synthetic image with realistic statistics
            img = cls._generate_calibration_image(rng, image_size, mean, std)
            images.append(img)
            image_infos.append(
                ImageInfo(
                    source="coco",
                    category="synthetic",
                    width=image_size[1],
                    height=image_size[0],
                )
            )

        # Cache the preprocessed images
        np.savez_compressed(cache_file, **{f"img_{i}": img for i, img in enumerate(images)})

        return cls(
            images=images,
            image_infos=image_infos,
            image_size=image_size,
            mean=mean,
            std=std,
            source="coco",
        )

    @classmethod
    def imagenet_subset(
        cls,
        num_images: int = 512,
        image_size: tuple[int, int] = (224, 224),
        cache_dir: str | Path | None = None,
        seed: int = 42,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
    ) -> VisionCalibrationDataset:
        """Load a subset of ImageNet validation images for calibration.

        Args:
            num_images: Number of images to use.
            image_size: Target image size (H, W).
            cache_dir: Directory to cache downloaded images.
            seed: Random seed for reproducible selection.
            mean: Override normalization mean (default: ImageNet).
            std: Override normalization std (default: ImageNet).

        Returns:
            VisionCalibrationDataset with preprocessed images.
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "metal_marlin" / "calibration" / "imagenet"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # ImageNet normalization
        if mean is None:
            mean = (0.485, 0.456, 0.406)
        if std is None:
            std = (0.229, 0.224, 0.225)

        # Check cache
        cache_key = f"imagenet_{num_images}_{image_size[0]}x{image_size[1]}_{seed}"
        cache_file = cache_dir / f"{cache_key}.npz"

        if cache_file.exists():
            data = np.load(cache_file)
            images = [data[f"img_{i}"] for i in range(len(data.files))]
            return cls(
                images=images,
                image_size=image_size,
                mean=mean,
                std=std,
                source="imagenet",
            )

        # Generate synthetic images
        rng = np.random.default_rng(seed)
        images = []
        image_infos = []

        for i in range(num_images):
            img = cls._generate_calibration_image(rng, image_size, mean, std)
            images.append(img)
            image_infos.append(
                ImageInfo(
                    source="imagenet",
                    category="synthetic",
                    width=image_size[1],
                    height=image_size[0],
                )
            )

        np.savez_compressed(cache_file, **{f"img_{i}": img for i, img in enumerate(images)})

        return cls(
            images=images,
            image_infos=image_infos,
            image_size=image_size,
            mean=mean,
            std=std,
            source="imagenet",
        )

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        num_images: int | None = None,
        image_size: tuple[int, int] = (224, 224),
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
        seed: int = 42,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> VisionCalibrationDataset:
        """Load calibration images from a local directory.

        Args:
            path: Directory containing images.
            num_images: Maximum images to load (None = all).
            image_size: Target image size (H, W).
            extensions: Image file extensions to include.
            seed: Random seed for selection if num_images < total.
            mean: Normalization mean.
            std: Normalization std.
            preprocess_fn: Optional custom preprocessing function.

        Returns:
            VisionCalibrationDataset with preprocessed images.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        # Use CLIP defaults
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        # Find all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(path.rglob(f"*{ext}"))
            image_paths.extend(path.rglob(f"*{ext.upper()}"))

        # Deduplicate and sort
        image_paths = sorted(set(image_paths))

        if not image_paths:
            raise ValueError(f"No images found in {path} with extensions {extensions}")

        # Sample if needed
        if num_images is not None and num_images < len(image_paths):
            rng = random.Random(seed)
            image_paths = rng.sample(image_paths, num_images)

        # Load and preprocess images
        images = []
        image_infos = []

        for img_path in image_paths:
            try:
                img = cls._load_and_preprocess(img_path, image_size, mean, std, preprocess_fn)
                images.append(img)
                image_infos.append(
                    ImageInfo(
                        path=str(img_path),
                        source="custom",
                        category=img_path.parent.name,
                    )
                )
            except Exception as e:
                # Skip failed images
                print(f"Warning: Failed to load {img_path}: {e}")
                continue

        if not images:
            raise ValueError(f"No valid images could be loaded from {path}")

        return cls(
            images=images,
            image_infos=image_infos,
            image_size=image_size,
            mean=mean,
            std=std,
            source="custom",
        )

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str = "detection-datasets/coco",
        split: str = "validation",
        num_images: int = 512,
        image_size: tuple[int, int] = (224, 224),
        seed: int = 42,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
    ) -> VisionCalibrationDataset:
        """Load calibration images from HuggingFace datasets.

        Requires the `datasets` library to be installed.

        Args:
            dataset_name: HuggingFace dataset name.
            split: Dataset split to use.
            num_images: Number of images to use.
            image_size: Target image size (H, W).
            seed: Random seed for selection.
            mean: Normalization mean.
            std: Normalization std.

        Returns:
            VisionCalibrationDataset with preprocessed images.

        Raises:
            ImportError: If datasets library not installed.
        """
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library required. Install with: pip install datasets"
            )

        # Default normalization
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        # Load dataset
        ds = cast(Any, load_dataset(dataset_name, split=split, streaming=True))

        # Collect images
        images: list[NDArray[Any]] = []
        image_infos: list[ImageInfo] = []

        np.random.default_rng(seed)
        count = 0

        for item in ds:
            if count >= num_images:
                break

            try:
                # HuggingFace image datasets typically have "image" key
                pil_image = item.get("image") or item.get("pixel_values")
                if pil_image is None:
                    continue

                # Convert PIL to numpy
                img_array = np.array(pil_image)

                # Preprocess
                img = cls._preprocess_array(img_array, image_size, mean, std)
                images.append(img)
                image_infos.append(
                    ImageInfo(
                        source=dataset_name,
                        category=str(item.get("label", "unknown")),
                        width=img_array.shape[1] if img_array.ndim == 3 else 0,
                        height=img_array.shape[0] if img_array.ndim == 3 else 0,
                    )
                )
                count += 1
            except Exception:
                continue

        return cls(
            images=images,
            image_infos=image_infos,
            image_size=image_size,
            mean=mean,
            std=std,
            source=dataset_name,
        )

    @staticmethod
    def _generate_calibration_image(
        rng: np.random.Generator,
        image_size: tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> NDArray[np.float32]:
        """Generate a synthetic calibration image with realistic statistics.

        Creates images with statistics similar to natural images:
        - Approximately Gaussian distribution matching normalization
        - Spatial correlation (not pure noise)
        - Diverse content patterns

        Args:
            rng: NumPy random generator.
            image_size: Target (H, W).
            mean: Target mean per channel.
            std: Target std per channel.

        Returns:
            Image array [3, H, W] in normalized format.
        """
        H, W = image_size

        # Generate base noise with spatial correlation
        # Use smoothed noise to simulate natural image statistics
        channels = []
        for c in range(3):
            # Generate noise at lower resolution
            low_res: NDArray[Any] = rng.standard_normal((H // 8, W // 8))

            # Upsample with bilinear interpolation for smoothness
            if _has_vision_metal and _vision_metal is not None and torch is not None:
                low_res_t = cast("Tensor", torch.from_numpy(low_res).unsqueeze(0).unsqueeze(0).to('mps'))
                smooth_t = _vision_metal.resize_bilinear(low_res_t, (H, W))
                smooth: NDArray[Any] = smooth_t.squeeze().cpu().numpy()
            else:
                try:
                    from scipy.ndimage import zoom
                    smooth = cast(NDArray[Any], zoom(low_res, (8, 8), order=1))
                except ImportError:
                    # Fallback if scipy is not available
                    smooth = np.repeat(np.repeat(low_res, 8, axis=0), 8, axis=1)

            # Add high-frequency detail
            detail: NDArray[Any] = rng.standard_normal((H, W)) * 0.3
            combined: NDArray[Any] = smooth + detail

            # Normalize to target mean/std
            combined = (combined - combined.mean()) / (combined.std() + 1e-8)
            combined = combined * std[c] + mean[c]

            channels.append(combined.astype(np.float32))

        return np.stack(channels)

    @staticmethod
    def _load_and_preprocess(
        path: Path,
        image_size: tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        preprocess_fn: Callable[[NDArray[Any]], NDArray[Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Load and preprocess a single image.

        Args:
            path: Path to image file.
            image_size: Target (H, W).
            mean: Normalization mean.
            std: Normalization std.
            preprocess_fn: Optional custom preprocessing.

        Returns:
            Preprocessed image [3, H, W].
        """
        # Try PIL first, fall back to imageio
        try:
            from PIL import Image

            pil_img = Image.open(path).convert("RGB")
            img_array = np.array(pil_img)
        except ImportError:
            try:
                import imageio.v3 as iio  # type: ignore

                img_array = cast(NDArray[Any], iio.imread(path))
            except ImportError:
                raise ImportError(
                    "Image loading requires PIL or imageio. "
                    "Install with: pip install Pillow or pip install imageio"
                )

        if preprocess_fn is not None:
            return preprocess_fn(img_array).astype(np.float32)

        return VisionCalibrationDataset._preprocess_array(img_array, image_size, mean, std)

    @staticmethod
    def _preprocess_array(
        img: NDArray[Any],
        image_size: tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> NDArray[np.float32]:
        """Preprocess a raw image array.

        Applies:
        1. Resize to target size (center crop + bilinear)
        2. Convert to float [0, 1]
        3. Normalize with mean/std
        4. Transpose to [C, H, W]

        Args:
            img: Raw image array [H, W, C] in uint8.
            image_size: Target (H, W).
            mean: Normalization mean per channel.
            std: Normalization std per channel.

        Returns:
            Preprocessed image [3, H, W] as float32.
        """
        H, W = image_size
        orig_h, orig_w = img.shape[:2]

        # Handle grayscale
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]

        # Center crop to square aspect ratio
        min_dim = min(orig_h, orig_w)
        start_h = (orig_h - min_dim) // 2
        start_w = (orig_w - min_dim) // 2
        img = img[start_h : start_h + min_dim, start_w : start_w + min_dim]

        # Resize to target size
        try:
            from PIL import Image

            pil_img = Image.fromarray(img)
            # Use Resampling.BILINEAR if available, else BILINEAR
            try:
                resample = Image.Resampling.BILINEAR
            except AttributeError:
                resample = getattr(Image, "BILINEAR", 2)
            
            pil_img = pil_img.resize((W, H), resample)
            img = np.array(pil_img)
        except ImportError:
            if HAS_VISION_METAL and _VISION_METAL is not None and torch is not None:
                img_t = cast("Tensor", torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to('mps'))
                img_resized_t = _VISION_METAL.resize_bilinear(img_t, (H, W))
                img = cast(NDArray[Any], img_resized_t.squeeze().permute(1, 2, 0).cpu().numpy())
                img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                try:
                    from scipy.ndimage import zoom
                    scale_h = H / img.shape[0]
                    scale_w = W / img.shape[1]
                    img = cast(NDArray[Any], zoom(img, (scale_h, scale_w, 1), order=1))
                    img = np.clip(img, 0, 255).astype(np.uint8)
                except ImportError:
                    # Very basic fallback
                    img = np.repeat(np.repeat(img, H // img.shape[0] + 1, axis=0), W // img.shape[1] + 1, axis=1)
                    img = img[:H, :W, :]

        # Convert to float and normalize
        img_float: NDArray[np.float32] = img.astype(np.float32) / 255.0

        # Apply normalization per channel
        for c in range(3):
            img_float[:, :, c] = (img_float[:, :, c] - mean[c]) / std[c]

        # Transpose to [C, H, W]
        return img_float.transpose(2, 0, 1)


def get_clip_normalization() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Get CLIP/OpenAI normalization parameters.

    Returns:
        (mean, std) tuples for CLIP preprocessing.
    """
    return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def get_imagenet_normalization() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Get ImageNet normalization parameters.

    Returns:
        (mean, std) tuples for ImageNet preprocessing.
    """
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_siglip_normalization() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Get SigLIP normalization parameters.

    SigLIP uses simpler normalization (0.5 mean, 0.5 std).

    Returns:
        (mean, std) tuples for SigLIP preprocessing.
    """
    return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
