"""Minimal requests-like session over the system curl (normal TLS verification via the OS trust store)."""
import os
import subprocess
import tempfile
import urllib.parse


class Resp:
    def __init__(self, body: bytes, code: int, ctype: str):
        self.content = body
        self.status_code = code
        self.headers = {"content-type": ctype}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")


class CurlSession:
    def __init__(self, ua: str):
        self.ua = ua
        fd, self.jar = tempfile.mkstemp(suffix=".cj")  # cookie jar, so the session survives across requests
        os.close(fd)
        self.headers = {"User-Agent": ua}
        self.verify = True

    def _run(self, url: str, data: dict | None = None, timeout: int = 60) -> Resp:
        cmd = ["curl", "-sL", "-b", self.jar, "-c", self.jar, "-A", self.ua, "--max-time", str(timeout),
               "-w", "\n%{http_code} %{content_type}"]
        if data is not None:
            cmd += ["-X", "POST", "--data", urllib.parse.urlencode(data)]
        out = subprocess.run([*cmd, url], capture_output=True, check=False).stdout
        body, _, tail = out.rpartition(b"\n")
        parts = tail.decode().split(" ", 1)
        return Resp(body, int(parts[0] or 0), parts[1] if len(parts) > 1 else "")

    def get(self, url: str, timeout: int = 60) -> Resp:
        return self._run(url, timeout=timeout)

    def post(self, url: str, data: dict | None = None, timeout: int = 60) -> Resp:
        return self._run(url, data=data, timeout=timeout)
