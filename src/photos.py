"""Opening photos. Importing this registers the HEIC opener, so other modules don't have to."""
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def open_photo(path) -> Image.Image:
    """The photo upright (EXIF orientation applied), in RGB."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")
