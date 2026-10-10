"""Vision-language models behind one call, so experiments swap them by name.

    vlm = get_vlm("opus")
    answer = vlm.ask([image], "prompt", MyPydanticSchema)

Names: opus, sonnet, haiku, or any full Anthropic model id. Another provider is one more class with `ask`.
Answers are cached in runs/vlm_cache/ by model + prompt + images, so re-running geometry changes is free.
"""
import base64
import hashlib
import io
from pathlib import Path
from typing import Protocol, TypeVar

import anthropic
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)

ALIASES = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}
MAX_SIDE = 2576  # high-res limit: under it, coordinates in the answer are 1:1 with the pixels sent
MAX_PIXELS = 3_750_000
CACHE = Path("runs/vlm_cache")


class VLM(Protocol):
    name: str

    def ask(self, images: list[Image.Image], prompt: str, schema: type[T]) -> T: ...


class Claude:
    def __init__(self, model: str):
        self.name = model
        self._client = None  # created on first call, so local-only runs need no key

    def ask(self, images: list[Image.Image], prompt: str, schema: type[T]) -> T:
        data = [jpeg(img) for img in images]
        key = hashlib.sha256("\0".join([self.name, prompt, schema.__name__, *data]).encode()).hexdigest()
        cached = CACHE / f"{key}.json"
        if cached.exists():
            return schema.model_validate_json(cached.read_text())
        self._client = self._client or anthropic.Anthropic()
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": d}} for d in data]
        response = self._client.messages.parse(
            model=self.name,
            max_tokens=16000,
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": [*content, {"type": "text", "text": prompt}]}],
            output_format=schema,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"{self.name} refused")
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(response.parsed_output.model_dump_json())
        return response.parsed_output


def jpeg(img: Image.Image) -> str:
    if max(img.size) > MAX_SIDE or img.width * img.height > MAX_PIXELS:
        raise ValueError(f"image {img.size} is over the size limit; resize it and rescale the coordinates yourself")
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return base64.standard_b64encode(buf.getvalue()).decode()


def get_vlm(name: str) -> VLM:
    return Claude(ALIASES.get(name, name))
