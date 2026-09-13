import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path

key = ""
for line in Path(".env").read_text(encoding="utf-8-sig").splitlines():
    if line.strip().startswith("OPENAI_API_KEY="):
        key = line.split("=", 1)[1].strip().strip('"').strip("'")
        break

print("key", key[:8] + "..." + key[-4:], "len", len(key))
ctx = ssl.create_default_context()

bases = [
    "https://api.hcnsec.cn/v1",
    "https://api.hcnsec.cn",
    "https://api.hcnsec.cn/openai/v1",
    "https://api.hcnsec.cn/api/v1",
    "https://hcnsec.cn/v1",
]

for base in bases:
    url = base.rstrip("/") + "/models"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
            data = json.loads(r.read().decode())
            ids = []
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                ids = [m.get("id") for m in data["data"] if isinstance(m, dict)]
            print("OK_MODELS", base, "count", len(ids))
            for m in ids[:25]:
                print(" ", m)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:200]
        print("FAIL_MODELS", base, e.code, body.replace("\n", " "))
    except Exception as e:
        print("FAIL_MODELS", base, str(e)[:150])

# try chat on /v1
print("\n--- chat tests ---")
chat_bases = [
    "https://api.hcnsec.cn/v1",
    "https://api.hcnsec.cn",
]
models_try = [
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-4o",
    "deepseek-chat",
    "gpt-4",
    "gemini-pro",
    "claude-3-haiku",
]
for base in chat_bases:
    for model in models_try:
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "Say ok"}],
                "max_tokens": 8,
            }
        ).encode()
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=25) as r:
                data = json.loads(r.read().decode())
                msg = data["choices"][0]["message"]["content"]
                print("CHAT_OK", base, model, "->", repr(msg)[:60])
                raise SystemExit(0)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:120].replace("\n", " ")
            print("CHAT_FAIL", base, model, e.code, body)
        except Exception as e:
            print("CHAT_FAIL", base, model, str(e)[:100])
