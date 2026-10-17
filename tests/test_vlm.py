from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from PIL import Image

import vlm


def test_vlm_cache_write_leaves_no_temp_files(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm, "CACHE", tmp_path)

    class Answer(vlm.BaseModel):
        x: int

    reply = SimpleNamespace(stop_reason="end_turn", parsed_output=Answer(x=1))
    model = vlm.get_vlm("opus")
    model._client = SimpleNamespace(messages=SimpleNamespace(parse=lambda **_: reply))
    img = Image.new("RGB", (8, 8))
    with ThreadPoolExecutor(8) as pool:  # the same uncached question from many threads at once
        answers = list(pool.map(lambda _: model.ask([img], "q", Answer), range(16)))
    assert all(a.x == 1 for a in answers)
    assert [p.suffix for p in tmp_path.iterdir()] == [".json"]
