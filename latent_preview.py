"""
Latent Preview for Video Sampling - Provides animated previews during KSampler execution for video models.
Based on VideoHelperSuite implementation, with compatibility checks to avoid conflicts.
"""

from PIL import Image
import time
import io
import struct
from threading import Thread
import functools

import torch
import torch.nn.functional as F

# Rates table for different video models (frames per second / temporal compression)
RATES_TABLE = {
    'Mochi': 24 // 6,
    'LTXV': 24 // 8,
    'HunyuanVideo': 24 // 4,
    'Cosmos1CV8x8x8': 24 // 8,
    'Wan21': 16 // 4,
    'Wan22': 24 // 4
}

# Global flag to track if we've hooked the previewer
_hook_installed = False
_original_get_previewer = None


def hook(obj, attr):
    """Decorator to hook/wrap an existing function on an object."""
    def dec(f):
        f = functools.update_wrapper(f, getattr(obj, attr))
        setattr(obj, attr, f)
        return f
    return dec


# Import latent_preview at module level for proper inheritance
import latent_preview as _latent_preview_module


class WrappedPreviewer(_latent_preview_module.LatentPreviewer):
    """
    Wraps the standard latent previewer to provide animated video previews.
    This class intercepts preview requests and sends frames as an animation
    instead of static images.
    """

    def __init__(self, previewer, rate=8, server_instance=None):
        # Don't call super().__init__() as we're wrapping an existing previewer
        self.first_preview = True
        self.last_time = 0
        self.c_index = 0
        self.rate = rate
        self.server = server_instance
        self.is_video_taesd = False

        # Copy decoder attributes from the original previewer
        # Check what type of previewer we have
        if hasattr(previewer, 'taesd'):
            self.taesd = previewer.taesd
            # Detect if this is a video TAESD (VAE-based) vs regular TAESD
            # Video TAESDs use a full VAE with first_stage_model
            if hasattr(self.taesd, 'first_stage_model'):
                self.is_video_taesd = True
        elif hasattr(previewer, 'latent_rgb_factors'):
            self.latent_rgb_factors = previewer.latent_rgb_factors
            self.latent_rgb_factors_bias = previewer.latent_rgb_factors_bias
            self.latent_rgb_factors_reshape = getattr(previewer, 'latent_rgb_factors_reshape', None)
        else:
            raise Exception('Unsupported preview type for animated latent previews')

    def decode_latent_to_preview_image(self, preview_format, x0):
        """
        Main preview method - intercepts the standard preview call and
        converts it to animated frame previews for video latents.
        """
        import server

        if x0.ndim == 5:
            # Keep batch major for video tensors
            x0 = x0.movedim(2, 1)
            x0 = x0.reshape((-1,) + x0.shape[-3:])

        num_images = x0.size(0)
        new_time = time.time()
        num_previews = int((new_time - self.last_time) * self.rate)
        self.last_time = self.last_time + num_previews / self.rate

        if num_previews > num_images:
            num_previews = num_images
        elif num_previews <= 0:
            return None

        if self.first_preview:
            self.first_preview = False
            # Send initialization message to frontend
            self.server.send_sync('PM_latentpreview', {
                'length': num_images,
                'rate': self.rate,
                'id': self.server.last_node_id
            })
            self.last_time = new_time + 1 / self.rate

        if self.c_index + num_previews > num_images:
            x0 = x0.roll(-self.c_index, 0)[:num_previews]
        else:
            x0 = x0[self.c_index:self.c_index + num_previews]

        # Process previews in a thread
        if hasattr(self, 'latent_rgb_factors'):
            Thread(target=self._process_latent2rgb_batch, args=(x0, self.c_index, num_images)).run()
        else:
            Thread(target=self._process_taesd_batch, args=(x0, self.c_index, num_images)).run()
        self.c_index = (self.c_index + num_previews) % num_images
        return None

    def _process_taesd_batch(self, latent_frames, ind, leng):
        """Process and send preview frames to the frontend.

        Args:
            latent_frames: Latent tensor, shape (N, C, H, W) where N is number of frames
            ind: Current frame index
            leng: Total number of frames
        """
        import server

        # For TAESD, decode each frame (don't change this - it works)
        for i in range(latent_frames.size(0)):
            frame_latent = latent_frames[i:i + 1]  # Keep batch dim: (1, C, H, W)

            # Decode single frame to PIL Image (same as ComfyUI does)
            preview_image = self.decode_single_frame(frame_latent)

            if preview_image is None:
                continue

            # Resize if needed
            if preview_image.width > 512 or preview_image.height > 512:
                if preview_image.width > preview_image.height:
                    new_width = 512
                    new_height = int(512 * preview_image.height / preview_image.width)
                else:
                    new_height = 512
                    new_width = int(512 * preview_image.width / preview_image.height)
                preview_image = preview_image.resize((new_width, new_height), Image.LANCZOS)

            # Send to frontend
            message = io.BytesIO()
            message.write((1).to_bytes(length=4, byteorder='big') * 2)
            message.write(ind.to_bytes(length=4, byteorder='big'))
            message.write(struct.pack('16p', self.server.last_node_id.encode('ascii')))
            preview_image.save(message, format="JPEG", quality=95)

            self.server.send_sync(
                server.BinaryEventTypes.PREVIEW_IMAGE,
                message.getvalue(),
                self.server.client_id
            )
            ind = (ind + 1) % leng

    def _process_latent2rgb_batch(self, x0, ind, leng):
        """Process Latent2RGB previews using batch decode (VHS style)."""
        import server

        # Apply reshape if needed
        if self.latent_rgb_factors_reshape is not None:
            x0 = self.latent_rgb_factors_reshape(x0)

        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

        # Batch decode: x0 is (N, C, H, W), output is (N, H, W, 3)
        image_tensor = F.linear(x0.movedim(1, -1), self.latent_rgb_factors,
                                bias=self.latent_rgb_factors_bias)

        # Resize if needed
        if image_tensor.size(1) > 512 or image_tensor.size(2) > 512:
            image_tensor = image_tensor.movedim(-1, 0)
            if image_tensor.size(2) < image_tensor.size(3):
                height = (512 * image_tensor.size(2)) // image_tensor.size(3)
                image_tensor = F.interpolate(image_tensor, (height, 512), mode='bilinear')
            else:
                width = (512 * image_tensor.size(3)) // image_tensor.size(2)
                image_tensor = F.interpolate(image_tensor, (512, width), mode='bilinear')
            image_tensor = image_tensor.movedim(0, -1)

        # Convert to uint8 (scale from -1..1 to 0..255)
        previews_ubyte = (((image_tensor + 1.0) / 2.0).clamp(0, 1).mul(0xFF)).to(device="cpu", dtype=torch.uint8)

        # Send each frame
        for preview in previews_ubyte:
            i = Image.fromarray(preview.numpy())
            message = io.BytesIO()
            message.write((1).to_bytes(length=4, byteorder='big') * 2)
            message.write(ind.to_bytes(length=4, byteorder='big'))
            message.write(struct.pack('16p', self.server.last_node_id.encode('ascii')))
            i.save(message, format="JPEG", quality=95)
            self.server.send_sync(
                server.BinaryEventTypes.PREVIEW_IMAGE,
                message.getvalue(),
                self.server.client_id
            )
            ind = (ind + 1) % leng

    def decode_single_frame(self, x0):
        """Decode a single latent frame to PIL Image.

        This mirrors exactly what ComfyUI's preview classes do.

        Args:
            x0: Latent tensor, shape (1, C, H, W) for images or (1, C, T, H, W) for video

        Returns:
            PIL.Image
        """
        if hasattr(self, 'taesd'):
            if self.is_video_taesd:
                # TAEHVPreviewerImpl style: x0[:1, :, :1] then [0][0]
                # Video latent input is (1, C, H, W) - we need to add temporal dim
                # Actually for video models, x0 coming in may already be flattened
                # Let's add a temporal dim: (1, C, H, W) -> (1, C, 1, H, W)
                if x0.ndim == 4:
                    x0 = x0.unsqueeze(2)  # (1, C, H, W) -> (1, C, 1, H, W)

                x_sample = self.taesd.decode(x0[:1, :, :1])[0][0]
                # x_sample should now be (C, H, W), we need (H, W, C) for PIL
                # Video TAEs output 0-1 range, not -1 to 1
                return self._tensor_to_image(x_sample, do_scale=False)
            else:
                # TAESDPreviewerImpl style: decode(x0[:1])[0].movedim(0, 2)
                x_sample = self.taesd.decode(x0[:1])[0].movedim(0, 2)
                # x_sample is (H, W, C)
                return self._tensor_to_image(x_sample, do_scale=True)
        else:
            # Latent2RGBPreviewer style
            if self.latent_rgb_factors_reshape is not None:
                x0 = self.latent_rgb_factors_reshape(x0)

            self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
            if self.latent_rgb_factors_bias is not None:
                self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

            # Handle 5D (video) and 4D (image) latents
            if x0.ndim == 5:
                x0_frame = x0[0, :, 0]  # (C, H, W)
            else:
                x0_frame = x0[0]  # (C, H, W)

            latent_image = F.linear(
                x0_frame.movedim(0, -1),  # (H, W, C)
                self.latent_rgb_factors,
                bias=self.latent_rgb_factors_bias
            )
            # latent_image is (H, W, 3)
            return self._tensor_to_image(latent_image, do_scale=True)

    def _tensor_to_image(self, tensor, do_scale=True):
        """Convert tensor to PIL Image, matching ComfyUI's preview_to_image."""
        import comfy.model_management

        if do_scale:
            latents_ubyte = (((tensor + 1.0) / 2.0).clamp(0, 1).mul(0xFF))
        else:
            latents_ubyte = (tensor.clamp(0, 1).mul(0xFF))

        if comfy.model_management.directml_enabled:
            latents_ubyte = latents_ubyte.to(dtype=torch.uint8)

        latents_ubyte = latents_ubyte.to(
            device="cpu",
            dtype=torch.uint8,
            non_blocking=comfy.model_management.device_supports_non_blocking(tensor.device)
        )

        return Image.fromarray(latents_ubyte.numpy())


def is_vhs_installed():
    """Check if VideoHelperSuite is installed and has latent preview enabled."""
    try:
        # Check if VHS module is loaded
        import sys
        for module_name in sys.modules:
            if 'videohelpersuite' in module_name.lower():
                # VHS is installed, check if it has hooked the previewer
                func = getattr(_latent_preview_module, 'get_previewer', None)
                if func and hasattr(func, '__wrapped__'):
                    # VHS has already hooked the previewer
                    return True
        return False
    except:
        return False


def install_latent_preview_hook():
    """
    Install the latent preview hook if:
    1. It's not already installed by us
    2. VideoHelperSuite hasn't already installed it
    """
    global _hook_installed, _original_get_previewer

    if _hook_installed:
        print("[PromptManager] Latent preview hook already installed")
        return True

    if is_vhs_installed():
        print("[PromptManager] VideoHelperSuite detected with latent preview - skipping to avoid conflict")
        return False

    try:
        import server
        serv = server.PromptServer.instance

        # Store original function
        _original_get_previewer = _latent_preview_module.get_previewer

        @hook(_latent_preview_module, 'get_previewer')
        def get_latent_video_previewer(device, latent_format, *args, **kwargs):
            """Wrapped previewer that checks settings and returns animated previewer if enabled."""
            previewer = get_latent_video_previewer.__wrapped__(device, latent_format, *args, **kwargs)

            try:
                extra_info = next(serv.prompt_queue.currently_running.values().__iter__())[3]['extra_pnginfo']['workflow']['extra']
                prev_setting = extra_info.get('PM_latentpreview', False)

                if extra_info.get('PM_latentpreviewrate', 0) != 0:
                    rate_setting = extra_info['PM_latentpreviewrate']
                else:
                    rate_setting = RATES_TABLE.get(latent_format.__class__.__name__, 8)
            except:
                # For safety since there's lots of keys, any of which can fail
                prev_setting = False

            if not prev_setting or not hasattr(previewer, "decode_latent_to_preview"):
                return previewer

            return WrappedPreviewer(previewer, rate_setting, serv)

        _hook_installed = True
        print("[PromptManager] Latent preview hook installed successfully")
        return True

    except Exception as e:
        print(f"[PromptManager] Failed to install latent preview hook: {e}")
        return False


def uninstall_latent_preview_hook():
    """Restore the original get_previewer function."""
    global _hook_installed, _original_get_previewer

    if not _hook_installed or _original_get_previewer is None:
        return

    try:
        _latent_preview_module.get_previewer = _original_get_previewer
        _hook_installed = False
        _original_get_previewer = None
        print("[PromptManager] Latent preview hook uninstalled")
    except Exception as e:
        print(f"[PromptManager] Failed to uninstall latent preview hook: {e}")