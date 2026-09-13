from pathlib import Path

raw = Path(".env").read_bytes()
print("file_bytes", len(raw))
print("has_utf8_bom", raw.startswith(b"\xef\xbb\xbf"))
text = Path(".env").read_text(encoding="utf-8-sig", errors="replace")
print("--- lines ---")
for i, line in enumerate(text.splitlines(), 1):
    s = line.strip()
    if not s:
        print(f"{i}: (empty)")
        continue
    if s.startswith("#"):
        print(f"{i}: # comment")
        continue
    if "=" not in s:
        print(f"{i}: (no equals)")
        continue
    k, v = s.split("=", 1)
    k, v = k.strip(), v.strip().strip('"').strip("'")
    if "KEY" in k.upper() or "TOKEN" in k.upper():
        print(
            f"{i}: {k} len={len(v)} starts={v[:8]!r} ends={v[-4:]!r} "
            f"spaces={(' ' in v)} paste_word={'paste' in v.lower()}"
        )
    else:
        print(f"{i}: {k}={v[:80]}")
