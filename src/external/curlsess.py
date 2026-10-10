import subprocess, tempfile, urllib.parse, os
class Resp:
    def __init__(s, body, code, ctype): s.content=body; s.status_code=code; s.headers={"content-type":ctype}
    @property
    def text(s): return s.content.decode("utf-8","replace")
class CurlSession:
    """Minimal requests-like session over system curl (normal TLS verification via the OS trust store)."""
    def __init__(s, ua):
        s.ua=ua; s.jar=tempfile.mktemp(suffix=".cj"); s.headers={"User-Agent":ua}; s.verify=True
    def _run(s, url, data=None, timeout=60):
        cmd=["curl","-sL","-b",s.jar,"-c",s.jar,"-A",s.ua,"--max-time",str(timeout),"-w","\n%{http_code} %{content_type}"]
        if data is not None: cmd+=["-X","POST","--data",urllib.parse.urlencode(data)]
        out=subprocess.run(cmd+[url],capture_output=True).stdout
        body,_,tail=out.rpartition(b"\n"); parts=tail.decode().split(" ",1)
        return Resp(body,int(parts[0] or 0),parts[1] if len(parts)>1 else "")
    def get(s,url,timeout=60): return s._run(url,timeout=timeout)
    def post(s,url,data=None,timeout=60): return s._run(url,data=data,timeout=timeout)
