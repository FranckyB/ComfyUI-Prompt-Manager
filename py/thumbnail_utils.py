import base64
from io import BytesIO

try:
    import numpy as np
    from PIL import Image
    IMAGE_THUMBNAIL_SUPPORT = True
except ImportError:
    IMAGE_THUMBNAIL_SUPPORT = False


def image_to_base64_thumbnail(image_tensor, max_size=200, jpeg_quality=85, log_prefix="ThumbnailUtils"):
    if not IMAGE_THUMBNAIL_SUPPORT or image_tensor is None:
        return None

    try:
        img_array = image_tensor[0] if len(image_tensor.shape) == 4 else image_tensor
        if hasattr(img_array, "cpu"):
            img_array = img_array.cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)

        img = Image.fromarray(img_array)

        width, height = img.size
        min_dim = min(width, height)
        if min_dim > max_size:
            scale = max_size / min_dim
            img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=jpeg_quality)
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        print(f"[{log_prefix}] Error converting image to thumbnail: {e}")
        return None