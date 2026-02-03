"""
Metal-accelerated vision preprocessing operations.

GPU-accelerated image preprocessing for vision models (ViT, CLIP, Qwen2-VL).
Replaces CPU-bound PIL/torchvision resize/normalize pipeline with Metal compute shaders.

Memory Layouts:
    - NCHW: [batch, channels, height, width] - PyTorch default
    - NHWC: [batch, height, width, channels] - Common image format

All methods accept PyTorch tensors on MPS device and return tensors on the same device.

Usage:
    from metal_marlin.vision.vision_metal import VisionMetal, preprocess_for_vit

    vision = VisionMetal()

    # Single operations
    resized = vision.resize_bilinear(image, (224, 224))
    normalized = vision.normalize(resized, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # Fused resize + normalize (faster, less memory)
    result = vision.resize_and_normalize(image, (224, 224), mean, std)

    # ViT patch extraction
    patches = vision.extract_patches(image, patch_size=16)

    # Convenience function for standard ImageNet preprocessing
    batch = preprocess_for_vit([image1, image2], target_size=(224, 224))
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from metal_marlin._compat import HAS_MPS, HAS_PYOBJC_METAL, Metal, torch

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from metal_marlin._compat import Tensor

    # Use string forward references for Metal types if needed,
    # or just use Any if they are hard to type without the framework.
    MTLDevice = Any
    MTLCommandQueue = Any
    MTLComputePipelineState = Any
    MTLBuffer = Any

_SHADER_DIR = Path(__file__).parent.parent.parent / "src"

T = TypeVar("T", bound="Tensor")

class VisionMetal:
    """Metal-accelerated vision preprocessing operations.

    This class provides GPU-accelerated image preprocessing kernels for vision models.
    All operations run on Apple Silicon Metal GPU and support both NCHW and NHWC
    memory layouts.

    Supported operations:
        - resize_bilinear: Bilinear interpolation resize
        - resize_bicubic: Bicubic interpolation resize (higher quality)
        - normalize: Channel-wise mean/std normalization
        - resize_and_normalize: Fused resize + normalize (single-pass)
        - extract_patches: Extract non-overlapping patches for ViT
        - preprocess_qwen2vl: Dynamic resolution for Qwen2-VL
        - uint8_to_float: Convert uint8 [0,255] to float [0,1]
        - center_crop: Center crop to target size

    Example:
        >>> vision = VisionMetal()
        >>> image = torch.randn(1, 3, 256, 256, device="mps")  # NCHW
        >>> resized = vision.resize_bilinear(image, (224, 224))
        >>> print(resized.shape)
        torch.Size([1, 3, 224, 224])
    """

    def __init__(self, device: MTLDevice | None = None):
        """Initialize Vision Metal dispatcher.

        Args:
            device: Metal device to use. If None, uses default system device.

        Raises:
            RuntimeError: If Metal or PyTorch MPS is not available.
        """
        if not HAS_PYOBJC_METAL or Metal is None:
            raise RuntimeError(
                "VisionMetal requires PyObjC Metal. Install with:\n"
                "  pip install pyobjc-framework-Metal"
            )
        if not HAS_MPS or torch is None:
            raise RuntimeError("VisionMetal requires PyTorch MPS backend (Apple Silicon).")

        self._device: MTLDevice = device or Metal.MTLCreateSystemDefaultDevice()
        self._command_queue: MTLCommandQueue = self._device.newCommandQueue()

        # Load shader source
        shader_path = _SHADER_DIR / "vision_preprocess.metal"
        if not shader_path.exists():
            raise FileNotFoundError(f"Shader not found: {shader_path}")
        self._shader_source: str = shader_path.read_text()

        # Compile pipelines lazily
        self._pipelines: dict[str, MTLComputePipelineState] = {}

    def _get_pipeline(self, function_name: str) -> MTLComputePipelineState:
        """Compile and cache a compute pipeline for the given kernel function."""
        if function_name in self._pipelines:
            return self._pipelines[function_name]

        if Metal is None:
             raise RuntimeError("Metal is not available")

        options = Metal.MTLCompileOptions.new()
        library, error = self._device.newLibraryWithSource_options_error_(
            self._shader_source, options, None
        )
        if library is None:
            error_msg = str(error) if error else "Unknown error"
            raise RuntimeError(f"Failed to compile Metal shader for {function_name}: {error_msg}")

        function = library.newFunctionWithName_(function_name)
        if function is None:
            raise RuntimeError(f"Function {function_name} not found in shader")

        pipeline, error = self._device.newComputePipelineStateWithFunction_error_(function, None)
        if pipeline is None:
            error_msg = str(error) if error else "Unknown error"
            raise RuntimeError(f"Failed to create pipeline for {function_name}: {error_msg}")

        self._pipelines[function_name] = pipeline
        return pipeline

    def _ensure_mps_tensor(
        self, tensor: Tensor | NDArray[Any], dtype: Any = None
    ) -> Tensor:
        """Ensure tensor is on MPS device with correct dtype.

        Args:
            tensor: Input tensor or numpy array
            dtype: Target dtype

        Returns:
            Tensor on MPS device
        """
        if torch is None:
            raise RuntimeError("Torch is not available")

        if dtype is None:
            dtype = torch.float32
        
        t: Tensor
        if isinstance(tensor, np.ndarray):
            t = torch.from_numpy(tensor)
        else:
            t = tensor
            
        if t.device.type != "mps":
            t = t.to("mps")
        if t.dtype != dtype:
            t = t.to(dtype)
        return t.contiguous()

    def _create_buffer(self, data: bytes | NDArray[Any]) -> MTLBuffer:
        """Create a Metal buffer from bytes or numpy array.

        Args:
            data: Bytes or numpy array to copy to GPU

        Returns:
            Metal buffer object
        """
        if Metal is None:
            raise RuntimeError("Metal is not available")

        if isinstance(data, np.ndarray):
            data = data.tobytes()
        return self._device.newBufferWithBytes_length_options_(
            data, len(data), Metal.MTLResourceStorageModeShared
        )

    def _tensor_to_buffer(self, tensor: Tensor) -> MTLBuffer:
        """Convert a PyTorch tensor to a Metal buffer.

        Args:
            tensor: PyTorch tensor on CPU or MPS

        Returns:
            Metal buffer containing tensor data
        """
        if torch is None:
            raise RuntimeError("Torch is not available")

        # Ensure tensor is on CPU for buffer creation
        if tensor.device.type == "mps":
            torch.mps.synchronize()
        np_array = tensor.detach().cpu().numpy()
        return self._create_buffer(np_array)

    def _buffer_to_tensor(
        self, buffer: MTLBuffer, shape: tuple[int, ...], dtype: Any = np.float32
    ) -> Tensor:
        """Copy data from Metal buffer back to PyTorch tensor.

        Args:
            buffer: Metal buffer with results
            shape: Shape of the output tensor
            dtype: Numpy dtype of the data

        Returns:
            PyTorch tensor on MPS device
        """
        if torch is None:
            raise RuntimeError("Torch is not available")

        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        result_bytes = buffer.contents().as_buffer(nbytes)
        np_array = np.frombuffer(result_bytes, dtype=dtype).reshape(shape)
        return torch.from_numpy(np_array.copy()).to("mps")

    def resize_bilinear(
        self,
        input: Tensor | NDArray[Any],
        target_size: tuple[int, int],
        nhwc: bool = False,
    ) -> Tensor:
        """Resize image using bilinear interpolation.

        Args:
            input: Input image tensor [N, C, H_in, W_in] (NCHW) or [N, H_in, W_in, C] (NHWC)
            target_size: Target (height, width)
            nhwc: If True, input is in NHWC format. Default is NCHW.

        Returns:
            Resized image [N, C, H_out, W_out] (NCHW) or [N, H_out, W_out, C] (NHWC)

        Example:
            >>> image = torch.randn(2, 3, 256, 256, device="mps")
            >>> resized = vision.resize_bilinear(image, (224, 224))
            >>> print(resized.shape)
            torch.Size([2, 3, 224, 224])
        """
        input_tensor = self._ensure_mps_tensor(input)
        batch_size = input_tensor.shape[0]

        if nhwc:
            h_in, w_in, channels = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
            output_shape = (batch_size, target_size[0], target_size[1], channels)
        else:
            channels, h_in, w_in = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
            output_shape = (batch_size, channels, target_size[0], target_size[1])

        h_out, w_out = target_size

        # Create output buffer
        output_np = np.empty(output_shape, dtype=np.float32)
        output_buffer = self._create_buffer(output_np)

        # Pack parameters: [batch_size, H_in, W_in, H_out, W_out, channels, nhwc]
        params_bytes = struct.pack("7I", batch_size, h_in, w_in, h_out, w_out, channels, 1 if nhwc else 0)
        params_buffer = self._create_buffer(params_bytes)

        # Create input buffer
        input_buffer = self._tensor_to_buffer(input_tensor)

        # Get pipeline and dispatch
        pipeline = self._get_pipeline("image_resize_bilinear")

        if Metal is None:
            raise RuntimeError("Metal is not available")

        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 2)

        # Grid: one thread per output pixel
        grid_size = Metal.MTLSizeMake(w_out, h_out, batch_size)
        tg_size = Metal.MTLSizeMake(16, 16, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        return self._buffer_to_tensor(output_buffer, output_shape)

    def resize_bicubic(
        self,
        input: Tensor | NDArray[Any],
        target_size: tuple[int, int],
        nhwc: bool = False,
    ) -> Tensor:
        """Resize image using bicubic interpolation (higher quality).

        Args:
            input: Input image tensor [N, C, H_in, W_in] (NCHW) or [N, H_in, W_in, C] (NHWC)
            target_size: Target (height, width)
            nhwc: If True, input is in NHWC format. Default is NCHW.

        Returns:
            Resized image [N, C, H_out, W_out] (NCHW) or [N, H_out, W_out, C] (NHWC)

        Example:
            >>> image = torch.randn(1, 3, 512, 512, device="mps")
            >>> resized = vision.resize_bicubic(image, (224, 224))
        """
        input_tensor = self._ensure_mps_tensor(input)
        batch_size = input_tensor.shape[0]

        if nhwc:
            h_in, w_in, channels = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
            output_shape = (batch_size, target_size[0], target_size[1], channels)
        else:
            channels, h_in, w_in = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
            output_shape = (batch_size, channels, target_size[0], target_size[1])

        h_out, w_out = target_size

        output_np = np.empty(output_shape, dtype=np.float32)
        output_buffer = self._create_buffer(output_np)

        params_bytes = struct.pack("7I", batch_size, h_in, w_in, h_out, w_out, channels, 1 if nhwc else 0)
        params_buffer = self._create_buffer(params_bytes)
        input_buffer = self._tensor_to_buffer(input_tensor)

        pipeline = self._get_pipeline("image_resize_bicubic")

        if Metal is None:
            raise RuntimeError("Metal is not available")

        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 2)

        grid_size = Metal.MTLSizeMake(w_out, h_out, batch_size)
        tg_size = Metal.MTLSizeMake(16, 16, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        return self._buffer_to_tensor(output_buffer, output_shape)

    def normalize(
        self,
        image: Tensor | NDArray[Any],
        mean: tuple[float, ...] | list[float] | Tensor,
        std: tuple[float, ...] | list[float] | Tensor,
        nhwc: bool = False,
    ) -> Tensor:
        """Apply channel-wise normalization: output = (input - mean) / std.

        Args:
            image: Input image [N, C, H, W] (NCHW) or [N, H, W, C] (NHWC)
            mean: Per-channel mean values [C]
            std: Per-channel standard deviation values [C]
            nhwc: If True, input is in NHWC format. Default is NCHW.

        Returns:
            Normalized image with same shape as input

        Example:
            >>> image = torch.randn(1, 3, 224, 224, device="mps")
            >>> mean = (0.485, 0.456, 0.406)
            >>> std = (0.229, 0.224, 0.225)
            >>> normalized = vision.normalize(image, mean, std)
        """
        image_tensor = self._ensure_mps_tensor(image)
        batch_size = image_tensor.shape[0]

        if nhwc:
            h, w, channels = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
        else:
            channels, h, w = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]

        # Convert mean/std to numpy arrays
        if isinstance(mean, (tuple, list)):
            mean_np = np.array(mean, dtype=np.float32)
        elif torch is not None and isinstance(mean, torch.Tensor):
            mean_np = mean.detach().cpu().numpy().astype(np.float32)
        else:
            mean_np = np.asarray(mean, dtype=np.float32)

        if isinstance(std, (tuple, list)):
            std_np = np.array(std, dtype=np.float32)
        elif torch is not None and isinstance(std, torch.Tensor):
            std_np = std.detach().cpu().numpy().astype(np.float32)
        else:
            std_np = np.asarray(std, dtype=np.float32)

        # Precompute 1/std for efficiency
        std_inv_np = 1.0 / std_np

        # Output buffer
        output_shape = image_tensor.shape
        output_np = np.empty(output_shape, dtype=np.float32)
        output_buffer = self._create_buffer(output_np)

        # Input buffer
        input_buffer = self._tensor_to_buffer(image_tensor)

        # Mean and std_inv buffers
        mean_buffer = self._create_buffer(mean_np)
        std_inv_buffer = self._create_buffer(std_inv_np)

        # Params: [batch_size, height, width, channels, nhwc]
        params_bytes = struct.pack("5I", batch_size, h, w, channels, 1 if nhwc else 0)
        params_buffer = self._create_buffer(params_bytes)

        pipeline = self._get_pipeline("image_normalize")

        if Metal is None:
            raise RuntimeError("Metal is not available")

        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(mean_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(std_inv_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 4)

        grid_size = Metal.MTLSizeMake(w, h, batch_size)
        tg_size = Metal.MTLSizeMake(16, 16, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        return self._buffer_to_tensor(output_buffer, output_shape)

    def resize_and_normalize(
        self,
        image: Tensor | NDArray[Any],
        size: tuple[int, int],
        mean: tuple[float, ...] | list[float] | Tensor,
        std: tuple[float, ...] | list[float] | Tensor,
        nhwc: bool = False,
    ) -> Tensor:
        """Fused resize and normalize in a single pass (faster, less memory).

        This performs bilinear resize and channel-wise normalization in a single
        kernel dispatch, eliminating intermediate buffer allocation.

        Args:
            image: Input image [N, C, H_in, W_in] (NCHW) or [N, H_in, W_in, C] (NHWC)
            size: Target (height, width)
            mean: Per-channel mean values [C]
            std: Per-channel standard deviation values [C]
            nhwc: If True, input is in NHWC format. Default is NCHW.

        Returns:
            Resized and normalized image [N, C, H_out, W_out] (NCHW) or [N, H_out, W_out, C] (NHWC)

        Example:
            >>> image = torch.randn(1, 3, 256, 256, device="mps")
            >>> mean = (0.485, 0.456, 0.406)
            >>> std = (0.229, 0.224, 0.225)
            >>> result = vision.resize_and_normalize(image, (224, 224), mean, std)
        """
        image_tensor = self._ensure_mps_tensor(image)
        batch_size = image_tensor.shape[0]

        if nhwc:
            h_in, w_in, channels = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
            output_shape = (batch_size, size[0], size[1], channels)
        else:
            channels, h_in, w_in = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
            output_shape = (batch_size, channels, size[0], size[1])

        h_out, w_out = size

        # Convert mean/std to numpy arrays
        if isinstance(mean, (tuple, list)):
            mean_np = np.array(mean, dtype=np.float32)
        elif torch is not None and isinstance(mean, torch.Tensor):
            mean_np = mean.detach().cpu().numpy().astype(np.float32)
        else:
            mean_np = np.asarray(mean, dtype=np.float32)

        if isinstance(std, (tuple, list)):
            std_np = np.array(std, dtype=np.float32)
        elif torch is not None and isinstance(std, torch.Tensor):
            std_np = std.detach().cpu().numpy().astype(np.float32)
        else:
            std_np = np.asarray(std, dtype=np.float32)

        std_inv_np = 1.0 / std_np

        output_np = np.empty(output_shape, dtype=np.float32)
        output_buffer = self._create_buffer(output_np)
        input_buffer = self._tensor_to_buffer(image_tensor)
        mean_buffer = self._create_buffer(mean_np)
        std_inv_buffer = self._create_buffer(std_inv_np)

        params_bytes = struct.pack("7I", batch_size, h_in, w_in, h_out, w_out, channels, 1 if nhwc else 0)
        params_buffer = self._create_buffer(params_bytes)

        pipeline = self._get_pipeline("image_resize_normalize_fused")

        if Metal is None:
            raise RuntimeError("Metal is not available")

        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(mean_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(std_inv_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 4)

        grid_size = Metal.MTLSizeMake(w_out, h_out, batch_size)
        tg_size = Metal.MTLSizeMake(16, 16, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        return self._buffer_to_tensor(output_buffer, output_shape)

    def extract_patches(
        self,
        image: Tensor | NDArray[Any],
        patch_size: int,
    ) -> Tensor:
        """Extract non-overlapping patches for Vision Transformer models.

        Converts image [N, H, W, C] -> patches [N, num_patches, patch_dim]

        For a 224x224 image with patch_size=16:
            num_patches = (224/16) * (224/16) = 196
            patch_dim = 16 * 16 * 3 = 768

        Args:
            image: Input image [N, H, W, C] (NHWC format required)
            patch_size: Size of each square patch (e.g., 16 for ViT-Base)

        Returns:
            Patches tensor [N, num_patches, patch_size * patch_size * C]

        Raises:
            ValueError: If image height or width is not divisible by patch_size.

        Example:
            >>> image = torch.randn(1, 224, 224, 3, device="mps")
            >>> patches = vision.extract_patches(image, patch_size=16)
            >>> print(patches.shape)
            torch.Size([1, 196, 768])
        """
        image_tensor = self._ensure_mps_tensor(image)
        batch_size = image_tensor.shape[0]
        h, w, channels = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]

        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(
                f"Image dimensions (h={h}, w={w}) must be divisible by patch_size={patch_size}"
            )

        patches_h = h // patch_size
        patches_w = w // patch_size
        num_patches = patches_h * patches_w
        patch_dim = patch_size * patch_size * channels

        output_shape = (batch_size, num_patches, patch_dim)
        output_np = np.empty(output_shape, dtype=np.float32)
        output_buffer = self._create_buffer(output_np)
        input_buffer = self._tensor_to_buffer(image_tensor)

        # Params: [batch_size, height, width, channels, patch_size]
        params_bytes = struct.pack("5I", batch_size, h, w, channels, patch_size)
        params_buffer = self._create_buffer(params_bytes)

        pipeline = self._get_pipeline("vit_patch_extract")

        if Metal is None:
            raise RuntimeError("Metal is not available")

        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 2)

        grid_size = Metal.MTLSizeMake(patches_w, patches_h, batch_size)
        tg_size = Metal.MTLSizeMake(8, 8, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        return self._buffer_to_tensor(output_buffer, output_shape)

    def preprocess_qwen2vl(
        self,
        image: Tensor | NDArray[Any],
        max_pixels: int = 1024 * 1024,
        patch_size: int = 14,
        nhwc: bool = False,
    ) -> Tensor:
        """Dynamic resolution preprocessing for Qwen2-VL style models.

        Qwen2-VL approach:
        1. Keep aspect ratio, resize to fit max_pixels while being divisible by patch_size
        2. Extract patches at multiple scales
        3. Add position embeddings based on actual (h, w) not fixed 224x224

        This kernel handles the resize to nearest valid resolution.

        Args:
            image: Input image [N, C, H_in, W_in] (NCHW) or [N, H_in, W_in, C] (NHWC)
            max_pixels: Maximum number of pixels (default 1MP = 1024x1024)
            patch_size: Patch size for ViT (default 14 for Qwen2-VL)
            nhwc: If True, input is in NHWC format. Default is NCHW.

        Returns:
            Resized image at dynamic resolution [N, C, H_out, W_out] or [N, H_out, W_out, C]

        Example:
            >>> image = torch.randn(1, 3, 1080, 1920, device="mps")  # HD image
            >>> processed = vision.preprocess_qwen2vl(image, max_pixels=512*512)
        """
        image_tensor = self._ensure_mps_tensor(image)
        batch_size = image_tensor.shape[0]

        if nhwc:
            h_in, w_in, channels = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
        else:
            channels, h_in, w_in = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]

        # Calculate target size maintaining aspect ratio
        total_pixels = h_in * w_in

        if total_pixels > max_pixels:
            scale = (max_pixels / total_pixels) ** 0.5
            h_out_raw = int(h_in * scale)
            w_out_raw = int(w_in * scale)
        else:
            h_out_raw, w_out_raw = h_in, w_in

        # Round to be divisible by patch_size
        h_out = (h_out_raw // patch_size) * patch_size
        w_out = (w_out_raw // patch_size) * patch_size
        h_out = max(h_out, patch_size)
        w_out = max(w_out, patch_size)

        if nhwc:
            output_shape = (batch_size, h_out, w_out, channels)
        else:
            output_shape = (batch_size, channels, h_out, w_out)

        output_np = np.empty(output_shape, dtype=np.float32)
        output_buffer = self._create_buffer(output_np)
        input_buffer = self._tensor_to_buffer(image_tensor)

        params_bytes = struct.pack("7I", batch_size, h_in, w_in, h_out, w_out, channels, 1 if nhwc else 0)
        params_buffer = self._create_buffer(params_bytes)

        pipeline = self._get_pipeline("dynamic_resize_qwen2vl")

        if Metal is None:
            raise RuntimeError("Metal is not available")

        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 2)

        grid_size = Metal.MTLSizeMake(w_out, h_out, batch_size)
        tg_size = Metal.MTLSizeMake(16, 16, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        return self._buffer_to_tensor(output_buffer, output_shape)

    def uint8_to_float(
        self,
        image: Tensor,
    ) -> Tensor:
        """Convert uint8 [0, 255] to float [0, 1] range.

        Common preprocessing step before resize/normalize for images loaded
        from JPEG/PNG files.

        Args:
            image: Input image as uint8 tensor of any shape

        Returns:
            Float tensor in [0, 1] range with same shape

        Example:
            >>> uint8_image = torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.uint8, device="mps")
            >>> float_image = vision.uint8_to_float(uint8_image)
            >>> print(float_image.min(), float_image.max())
            tensor(0.) tensor(0.9961)
        """
        if torch is None:
            raise RuntimeError("Torch is not available")

        if image.dtype != torch.uint8:
            raise ValueError(f"Input must be uint8, got {image.dtype}")

        image_contig = image.contiguous()
        total_elements = int(np.prod(image_contig.shape))

        output_np = np.empty(image_contig.shape, dtype=np.float32)
        output_buffer = self._create_buffer(output_np)
        input_buffer = self._tensor_to_buffer(image_contig)

        params_bytes = struct.pack("I", total_elements)
        params_buffer = self._create_buffer(params_bytes)

        pipeline = self._get_pipeline("uint8_to_float")

        if Metal is None:
            raise RuntimeError("Metal is not available")

        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 2)

        # 1D grid for element-wise operation
        threads_per_group = 256
        num_groups = (total_elements + threads_per_group - 1) // threads_per_group
        grid_size = Metal.MTLSizeMake(num_groups, 1, 1)
        tg_size = Metal.MTLSizeMake(threads_per_group, 1, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        return self._buffer_to_tensor(output_buffer, image_contig.shape)

    def center_crop(
        self,
        image: Tensor | NDArray[Any],
        size: tuple[int, int],
        nhwc: bool = False,
    ) -> Tensor:
        """Center crop an image to target size.

        Crops from the center, discarding border pixels equally from all sides.

        Args:
            image: Input image [N, C, H_in, W_in] (NCHW) or [N, H_in, W_in, C] (NHWC)
            size: Target (height, width)
            nhwc: If True, input is in NHWC format. Default is NCHW.

        Returns:
            Cropped image [N, C, H_out, W_out] (NCHW) or [N, H_out, W_out, C] (NHWC)

        Raises:
            ValueError: If target size is larger than input size.

        Example:
            >>> image = torch.randn(1, 3, 256, 256, device="mps")
            >>> cropped = vision.center_crop(image, (224, 224))
            >>> print(cropped.shape)
            torch.Size([1, 3, 224, 224])
        """
        image_tensor = self._ensure_mps_tensor(image)
        batch_size = image_tensor.shape[0]

        if nhwc:
            h_in, w_in, channels = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
            output_shape = (batch_size, size[0], size[1], channels)
        else:
            channels, h_in, w_in = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
            output_shape = (batch_size, channels, size[0], size[1])

        h_out, w_out = size

        if h_out > h_in or w_out > w_in:
            raise ValueError(
                f"Target size ({h_out}, {w_out}) cannot be larger than input size ({h_in}, {w_in})"
            )

        output_np = np.empty(output_shape, dtype=np.float32)
        output_buffer = self._create_buffer(output_np)
        input_buffer = self._tensor_to_buffer(image_tensor)

        params_bytes = struct.pack("7I", batch_size, h_in, w_in, h_out, w_out, channels, 1 if nhwc else 0)
        params_buffer = self._create_buffer(params_bytes)

        pipeline = self._get_pipeline("center_crop")

        if Metal is None:
            raise RuntimeError("Metal is not available")

        command_buffer = self._command_queue.commandQueue().commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 2)

        grid_size = Metal.MTLSizeMake(w_out, h_out, batch_size)
        tg_size = Metal.MTLSizeMake(16, 16, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, tg_size)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        return self._buffer_to_tensor(output_buffer, output_shape)


def preprocess_for_vit(
    images: list[Tensor] | Tensor | list[NDArray[Any]] | NDArray[Any],
    target_size: tuple[int, int] = (224, 224),
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tensor:
    """Standard ImageNet preprocessing pipeline on GPU.

    Performs the following operations:
    1. Resize images to target_size using bilinear interpolation
    2. Normalize with ImageNet mean/std

    This is a convenience function for the most common vision model preprocessing.
    For more control, use the VisionMetal class directly.

    Args:
        images: List of image tensors [C, H, W] or batched tensor [N, C, H, W]
        target_size: Target (height, width), default (224, 224)
        mean: Per-channel mean for normalization, ImageNet default
        std: Per-channel std for normalization, ImageNet default

    Returns:
        Batched preprocessed tensor [N, C, H, W] on MPS device
    """
    if torch is None:
        raise RuntimeError("preprocess_for_vit requires PyTorch")

    vision = VisionMetal()

    image_list: list[Tensor]
    
    if isinstance(images, list):
        image_list = []
        for img in images:
            if isinstance(img, np.ndarray):
                image_list.append(cast("Tensor", torch.from_numpy(img)))
            elif isinstance(img, (torch.Tensor, Tensor)):
                image_list.append(img)
            else:
                raise TypeError(f"Unsupported type in list: {type(img)}")
    elif isinstance(images, (torch.Tensor, Tensor)):
        img_tensor = cast("Tensor", images)
        if img_tensor.ndim == 4:
            # Already batched [N, C, H, W]
            return vision.resize_and_normalize(img_tensor, target_size, mean, std)
        elif img_tensor.ndim == 3:
            # Single image [C, H, W], add batch dim
            image_list = [img_tensor]
        else:
            raise ValueError(f"Input tensor must be 3D or 4D, got {img_tensor.ndim}D")
    elif isinstance(images, np.ndarray):
        img_np = images
        if img_np.ndim == 4:
            return vision.resize_and_normalize(img_np, target_size, mean, std)
        elif img_np.ndim == 3:
            image_list = [cast("Tensor", torch.from_numpy(img_np))]
        else:
            raise ValueError(f"Input array must be 3D or 4D, got {img_np.ndim}D")
    else:
        raise TypeError(f"Unsupported type for images: {type(images)}")

    # Stack list of images into batch
    # First ensure all are on MPS and same dtype
    processed: list[Tensor] = []
    for img in image_list:
        t = img
        if t.device.type != "mps":
            t = t.to("mps")
        if t.dtype != torch.float32:
            t = t.to(torch.float32)
        # Ensure NCHW format [C, H, W]
        if t.ndim == 3 and t.shape[-1] in [1, 3, 4] and t.shape[0] not in [1, 3, 4]:
            # Likely HWC format, convert to CHW
            t = t.permute(2, 0, 1)
        processed.append(t)

    # Stack into batch [N, C, H, W]
    batch = torch.stack(processed, dim=0)

    return vision.resize_and_normalize(batch, target_size, mean, std)
