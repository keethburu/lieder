#!/usr/bin/env python3
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp-extract" / "vision"


def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def data_url(path):
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def call_openai(image_path, ocr_hint=""):
    key = os.environ.get("OPEN_API_SECRET") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPEN_API_SECRET or OPENAI_API_KEY is required")

    prompt = (
        "Transcribe every visible song lyric on this page or photo. The page may contain "
        "multiple songs. Split the output by song. Preserve titles, verse numbers, refrains, "
        "repeat markers such as |: :|, punctuation, umlauts, and line breaks as accurately as "
        "possible. Ignore page numbers, decorative marks, and non-lyric advertising text. "
        "If a word is unreadable, write [?] rather than guessing. Before returning, verify "
        "that every visible title and every visible numbered verse on the page appears in "
        "your answer. Output markdown only."
    )
    if ocr_hint:
        prompt += "\n\nTesseract OCR draft, useful only as a hint and not authoritative:\n" + ocr_hint[:6000]

    payload = {
        "model": os.environ.get("VISION_MODEL", "gpt-5"),
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url(image_path), "detail": "high"},
                ],
            }
        ],
        "max_output_tokens": 6000,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as res:
                data = json.load(res)
            if data.get("status") != "completed" or data.get("incomplete_details"):
                raise RuntimeError(
                    "OpenAI response was incomplete: "
                    + json.dumps(
                        {
                            "status": data.get("status"),
                            "incomplete_details": data.get("incomplete_details"),
                        },
                        ensure_ascii=False,
                    )
                )
            chunks = []
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        chunks.append(content.get("text", ""))
            return "\n".join(chunks).strip(), data.get("id", "")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc


def ocr_hint_for(path):
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return ""
    ocr = ROOT / "tmp-extract" / "ocr" / f"{path.stem}.txt"
    return ocr.read_text(encoding="utf-8", errors="replace") if ocr.exists() else ""


def safe_name(path):
    parts = [p for p in path.relative_to(ROOT).parts]
    return "__".join(parts).replace("/", "__") + ".md"


def main(paths):
    load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        out_path = OUT / safe_name(path)
        if out_path.exists():
            print(f"skip {path} -> {out_path}")
            continue
        print(f"vision {path}")
        text, response_id = call_openai(path, ocr_hint_for(path))
        out_path.write_text(f"<!-- response: {response_id} -->\n\n{text}\n", encoding="utf-8")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
