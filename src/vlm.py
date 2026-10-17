"""Vision-language models behind one call, so experiments swap them by name.

    vlm = get_vlm("opus")
    answer = vlm.ask([image], "prompt", MyPydanticSchema)

Names: opus, sonnet, haiku, or a full Claude model id. Another provider is one more class with `ask`.
Answers are cached in runs/vlm_cache/ by model, prompt and images, so re-running a geometry change costs nothing.
"""
import base64
import hashlib
import io
import os
from pathlib import Path
from typing import Protocol, TypeVar

import anthropic
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

T = TypeVar("T", bound=BaseModel)

ALIASES = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}
MAX_SIDE = 2576  # high-resolution limit: below it, coordinates in an answer are 1:1 with the pixels sent
MAX_PIXELS = 3_750_000
MAX_TOKENS = 16000
CACHE = ROOT / "runs/vlm_cache"


class Refusal(Exception):
    """The model declined to answer."""


ERRORS = (anthropic.APIError, Refusal, ValidationError)  # what one failed question can raise


class VLM(Protocol):
    name: str

    def ask(self, images: list[Image.Image], prompt: str, schema: type[T]) -> T: ...


def known_model(name: str) -> bool:
    return name in ALIASES or name.startswith("claude-")


def has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class Claude:
    def __init__(self, model: str):
        self.name = model
        self._client = None  # created on first call, so runs that only hit the cache need no key

    def ask(self, images: list[Image.Image], prompt: str, schema: type[T]) -> T:
        data = [jpeg(img) for img in images]
        key = hashlib.sha256("\0".join([self.name, prompt, schema.__name__, *data]).encode()).hexdigest()
        cached = CACHE / f"{key}.json"
        if cached.exists():
            try:
                return schema.model_validate_json(cached.read_text())
            except ValidationError:
                pass  # an unreadable cache entry is asked again and overwritten
        self._client = self._client or anthropic.Anthropic()
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": d}} for d in data]
        response = self._client.messages.parse(
            model=self.name,
            max_tokens=MAX_TOKENS,
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": [*content, {"type": "text", "text": prompt}]}],
            output_format=schema,
        )
        if response.stop_reason == "refusal":
            raise Refusal(f"{self.name} declined to answer")
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(response.parsed_output.model_dump_json())
        tmp.replace(cached)  # atomic: a crash never leaves half an answer behind
        return response.parsed_output


def jpeg(img: Image.Image) -> str:
    if max(img.size) > MAX_SIDE or img.width * img.height > MAX_PIXELS:
        raise ValueError(f"image {img.size} is over the size limit; resize it and rescale the coordinates yourself")
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return base64.standard_b64encode(buf.getvalue()).decode()


def get_vlm(name: str) -> VLM:
    return Claude(ALIASES.get(name, name))
