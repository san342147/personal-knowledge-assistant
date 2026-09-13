"""Detect which OpenAI-compatible endpoint accepts the key in .env and list models."""
from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_key() -> str:
    env = ROOT / ".env"
    if not env.exists():
        return ""
    for line in env.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() in {"OPENAI_API_KEY", "XAI_API_KEY", "GROK_API_KEY"}:
            key = v.strip().strip('"').strip("'")
            if key and "paste" not in key.lower() and not key.endswith("here"):
                return key
    return ""


def try_models(base: str, key: str) -> tuple[bool, list[str], str]:
    url = base.rstrip("/") + "/models"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            ids: list[str] = []
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                for m in data["data"]:
                    if isinstance(m, dict) and m.get("id"):
                        ids.append(str(m["id"]))
            elif isinstance(data, list):
                for m in data:
                    if isinstance(m, dict) and m.get("id"):
                        ids.append(str(m["id"]))
            return True, ids, ""
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:220]
        return False, [], f"HTTP {e.code}: {body}"
    except Exception as e:
        return False, [], str(e)


def main() -> None:
    key = load_key()
    if not key:
        print("NO_VALID_KEY")
        return

    print(f"Key: {key[:8]}...{key[-4:]} (len={len(key)})")
    print()

    bases = [
        ("DeepSeek", "https://api.deepseek.com/v1"),
        ("Moonshot/Kimi", "https://api.moonshot.cn/v1"),
        ("Qwen DashScope", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ("SiliconFlow", "https://api.siliconflow.cn/v1"),
        ("Zhipu GLM", "https://open.bigmodel.cn/api/paas/v4"),
        ("OpenRouter", "https://openrouter.ai/api/v1"),
        ("Groq", "https://api.groq.com/openai/v1"),
        ("Together", "https://api.together.xyz/v1"),
        ("Fireworks", "https://api.fireworks.ai/inference/v1"),
        ("ChatAnywhere", "https://api.chatanywhere.tech/v1"),
        ("ChatAnywhere CN", "https://api.chatanywhere.com.cn/v1"),
        ("API2D", "https://oa.api2d.net/v1"),
        ("CloseAI", "https://api.closeai-proxy.xyz/v1"),
        ("AiHubMix", "https://aihubmix.com/v1"),
        ("OpenAI", "https://api.openai.com/v1"),
    ]

    success = []
    for name, base in bases:
        ok, ids, err = try_models(base, key)
        if ok:
            print(f"[OK] {name}")
            print(f"     BASE: {base}")
            print(f"     Models: {len(ids)}")
            preferred = [
                i
                for i in ids
                if any(
                    x in i.lower()
                    for x in (
                        "chat",
                        "plus",
                        "turbo",
                        "flash",
                        "instruct",
                        "deepseek",
                        "qwen",
                        "glm",
                        "moonshot",
                        "gpt-4o",
                        "gpt-3.5",
                    )
                )
            ]
            show = (preferred or ids)[:20]
            for mid in show:
                print(f"       - {mid}")
            success.append((name, base, show[0] if show else "gpt-3.5-turbo"))
            print()
        else:
            short = (err or "").replace("\n", " ")[:110]
            print(f"[FAIL] {name}: {short}")

    print()
    if success:
        name, base, model = success[0]
        print("=== RECOMMENDED ===")
        print(f"OPENAI_BASE_URL={base}")
        print(f"LLM_MODEL={model}")
    else:
        print("NONE_WORKED")
        print("Key is saved but rejected by common providers (invalid/expired/wrong site).")
        print("Open the website where you bought/created the key and copy:")
        print("  - API Base URL (OpenAI compatible)")
        print("  - A chat model name")
        print("  - Or create a new key")


if __name__ == "__main__":
    main()
