"""
Save Video H264/H265 Node
A custom node that saves videos with H.264 or H.265 (HEVC) codec with quality control.
Based on ComfyUI's built-in SaveVideo node but with added H.265 codec and CRF quality options.
"""

from __future__ import annotations

import os
import re
import av
import json
import math
import torch
from datetime import datetime
import folder_paths
from fractions import Fraction
from comfy.cli_args import args


class SaveVideoH26x:
    """
    Save video with H.264 or H.265 codec and quality control.
    Takes VIDEO input (with audio support) and saves workflow metadata.
    Clone of ComfyUI's SaveVideo with H.265 and CRF quality control added.
    """

    CODECS = ["h264", "h265"]
    CHROMA_SUBSAMPLING = ["yuv420", "yuv422", "yuv444"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The video to save."}),
                "filename_prefix": ("STRING", {
                    "default": "video/vid_%date:yy-MM-dd_hh-mm-ss%",
                    "tooltip": "The prefix for the file to save.\nSupports %date:format% patterns."
                }),
                "codec": (cls.CODECS, {
                    "default": "h264",
                    "tooltip": "Video codec.\nh264 = 8-bit, better compatibility.\nh265/HEVC = 10-bit, better compression & gradients."
                }),
                "chroma": (cls.CHROMA_SUBSAMPLING, {
                    "default": "yuv420",
                    "tooltip": "Chroma subsampling.\nyuv420 = most compatible.\nyuv422 = better color for video editing.\nyuv444 = best color, no chroma loss.\n"
                }),
                "crf": ("INT", {
                    "default": 18,
                    "min": 0,
                    "max": 51,
                    "step": 1,
                    "tooltip": "Constant Rate Factor (quality).\nLower = better quality, larger file.\n0 = lossless\n18-23 = high quality\n28+ = lower quality.\nRecommended: 18-28."
                }),
                "use_counter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add incrementing counter to filename.\nOff = cleaner names like '2026-01-16.mp4' (adds counter only if file exists).\nOn = always add counter like 'ComfyUI_00001_.mp4'."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "image/video"
    DESCRIPTION = "Saves video with H.264 or H.265 codec and quality control. Includes audio and workflow metadata."

    def save_video(self, video, filename_prefix, codec, chroma, crf, use_counter, prompt=None, extra_pnginfo=None):
        # Expand %date:format% patterns (e.g., %date:yy-MM-dd_hh-mm%)
        # This mimics ComfyUI's frontend JS date expansion
        def expand_date_format(text):
            def replace_date(match):
                fmt = match.group(1)
                now = datetime.now()
                # Convert common date format tokens to Python strftime
                fmt = fmt.replace('yyyy', '%Y').replace('yy', '%y')
                fmt = fmt.replace('MM', '%m').replace('dd', '%d')
                fmt = fmt.replace('HH', '%H').replace('hh', '%I')
                fmt = fmt.replace('mm', '%M').replace('ss', '%S')
                return now.strftime(fmt)
            return re.sub(r'%date:([^%]+)%', replace_date, text)

        filename_prefix = expand_date_format(filename_prefix)

        # Bit depth determined by codec: h264=8-bit, h265=10-bit
        is_10bit = (codec == "h265")
        pixel_format = f"{chroma}p" if not is_10bit else f"{chroma}p10le"

        # Get video components (images, audio, frame_rate)
        components = video.get_components()
        images = components.images
        audio = components.audio
        frame_rate = components.frame_rate

        # Get dimensions
        width, height = video.get_dimensions()

        # Get output path
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height
        )

        # Build filename based on use_counter setting
        if use_counter:
            # Always use counter with trailing underscore (ComfyUI standard pattern)
            file = f"{filename}_{counter:05}_.mp4"
        else:
            # Try without counter first
            file = f"{filename}.mp4"
            file_path_test = os.path.join(full_output_folder, file)
            if os.path.exists(file_path_test):
                # File exists, find next available counter
                pattern = re.compile(rf"^{re.escape(filename)}_(\d+)\.mp4$")
                existing_counters = []
                if os.path.isdir(full_output_folder):
                    for f in os.listdir(full_output_folder):
                        match = pattern.match(f)
                        if match:
                            existing_counters.append(int(match.group(1)))
                # Start from 1 if no counters found, otherwise next available
                next_counter = max(existing_counters, default=0) + 1
                file = f"{filename}_{next_counter:05}.mp4"

        file_path = os.path.join(full_output_folder, file)

        # Build metadata
        metadata = {}
        if not args.disable_metadata:
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)
            if prompt is not None:
                metadata["prompt"] = prompt

        # Select codec
        codec_name = "libx264" if codec == "h264" else "libx265"

        # Prepare frame rate as fraction
        frame_rate_frac = Fraction(round(float(frame_rate) * 1000), 1000)

        # Open output container with movflags for metadata
        with av.open(file_path, mode='w', options={'movflags': 'use_metadata_tags'}) as output:
            # Add metadata
            if metadata:
                for key, value in metadata.items():
                    output.metadata[key] = json.dumps(value) if not isinstance(value, str) else value

            # Create video stream
            video_stream = output.add_stream(codec_name, rate=frame_rate_frac)
            video_stream.width = width
            video_stream.height = height
            video_stream.pix_fmt = pixel_format

            # Set encoding options
            video_stream.options = {
                'crf': str(crf),
                'preset': 'medium',
            }

            # For H.265, add tag for Apple/browser compatibility
            if codec == "h265":
                video_stream.options['tag'] = 'hvc1'
                video_stream.options['x265-params'] = 'log-level=error'

            # Create audio stream if audio is present
            audio_stream = None
            audio_sample_rate = 1
            if audio is not None:
                audio_sample_rate = int(audio['sample_rate'])
                audio_stream = output.add_stream('aac', rate=audio_sample_rate)

            # Encode video frames
            for frame_tensor in images:
                if is_10bit:
                    # Convert tensor to 16-bit for 10-bit encoding
                    img = (frame_tensor[..., :3] * 65535).clamp(0, 65535).to(torch.int16).cpu().numpy().astype('uint16')
                    frame = av.VideoFrame.from_ndarray(img, format='rgb48le')
                else:
                    # Convert tensor to 8-bit
                    img = (frame_tensor[..., :3] * 255).clamp(0, 255).byte().cpu().numpy()
                    frame = av.VideoFrame.from_ndarray(img, format='rgb24')

                # IMPORTANT: Explicitly reformat frame to target pixel format
                # Without this, the encoder may not respect the stream's pix_fmt
                frame = frame.reformat(format=pixel_format)

                # Encode and mux
                for packet in video_stream.encode(frame):
                    output.mux(packet)

            # Flush video encoder
            for packet in video_stream.encode(None):
                output.mux(packet)

            # Encode audio if present
            if audio_stream is not None and audio is not None:
                waveform = audio['waveform']
                # Trim audio to match video duration
                waveform = waveform[:, :, :math.ceil((audio_sample_rate / float(frame_rate)) * images.shape[0])]

                # Create audio frame
                layout = 'mono' if waveform.shape[1] == 1 else 'stereo'
                frame = av.AudioFrame.from_ndarray(
                    waveform.movedim(2, 1).reshape(1, -1).float().numpy(),
                    format='flt',
                    layout=layout
                )
                frame.sample_rate = audio_sample_rate
                frame.pts = 0

                # Encode audio
                for packet in audio_stream.encode(frame):
                    output.mux(packet)

                # Flush audio encoder
                for packet in audio_stream.encode(None):
                    output.mux(packet)

        # h265 + yuv422/444 won't play in browser, so generate a browser-compatible preview in temp
        if codec == "h264" or (codec == "h265" and chroma == "yuv420"):
            # Browser can play this directly
            return {"ui": {"images": [{"filename": file, "subfolder": subfolder, "type": "output"}], "animated": (True,)}}
        else:
            # Create a browser-compatible h264 preview in temp folder
            temp_dir = folder_paths.get_temp_directory()
            preview_file = os.path.basename(file)
            # preview_file = f"preview_{counter:05}_{width}x{height}.mp4"
            preview_path = os.path.join(temp_dir, preview_file)

            with av.open(preview_path, mode='w', format='mp4') as preview_output:
                preview_stream = preview_output.add_stream('libx264', rate=frame_rate)
                preview_stream.width = width
                preview_stream.height = height
                preview_stream.pix_fmt = 'yuv420p'
                preview_stream.options = {'crf': '23', 'preset': 'fast'}  # Fast, smaller preview

                # Add audio stream if we have audio
                preview_audio_stream = None
                if audio is not None:
                    preview_audio_stream = preview_output.add_stream('aac', rate=int(audio['sample_rate']))

                for img_tensor in images:
                    frame_np = (img_tensor[..., :3] * 255).clamp(0, 255).byte().cpu().numpy()
                    frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
                    frame = frame.reformat(format='yuv420p')
                    for packet in preview_stream.encode(frame):
                        preview_output.mux(packet)

                for packet in preview_stream.encode(None):
                    preview_output.mux(packet)

                # Encode audio for preview
                if audio is not None and preview_audio_stream is not None:
                    waveform = audio['waveform']
                    preview_sample_rate = int(audio['sample_rate'])
                    # Trim audio to match video duration
                    waveform = waveform[:, :, :math.ceil((preview_sample_rate / float(frame_rate)) * images.shape[0])]

                    layout = 'mono' if waveform.shape[1] == 1 else 'stereo'
                    frame = av.AudioFrame.from_ndarray(
                        waveform.movedim(2, 1).reshape(1, -1).float().numpy(),
                        format='flt',
                        layout=layout
                    )
                    frame.sample_rate = preview_sample_rate
                    frame.pts = 0
                    for packet in preview_audio_stream.encode(frame):
                        preview_output.mux(packet)
                    for packet in preview_audio_stream.encode(None):
                        preview_output.mux(packet)

            return {"ui": {"images": [{"filename": preview_file, "subfolder": "", "type": "temp"}], "animated": (True,)}}

# Node registration
NODE_CLASS_MAPPINGS = {
    "SaveVideoH26x": SaveVideoH26x,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveVideoH26x": "Save Video H264/H265",
}
