"""Common utilities for benchmark: text normalization, WER, RTF, metadata."""

import datetime
import hashlib
import json
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Optional


def home_dir() -> Path:
    return Path.home()


def manifest_path() -> Path:
    p = home_dir() / ".gigastt/benchmarks/golos_wav/manifest.json"
    if p.exists():
        return p
    # fallback to bundled fixtures
    return Path(__file__).parent.parent / "crates/gigastt/tests/fixtures/manifest.json"


def load_manifest(max_samples: Optional[int] = None):
    with open(manifest_path(), encoding="utf-8") as f:
        data = json.load(f)
    if max_samples and max_samples > 0:
        data = data[:max_samples]
    return data


# --- Russian number-to-words (simplified, matching Rust logic) ---

ONES = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
         "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят",
        "семьдесят", "восемьдесят", "девяносто"]
HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот",
            "шестьсот", "семьсот", "восемьсот", "девятьсот"]


def number_to_words(n: int) -> str:
    if n == 0:
        return "ноль"
    if n > 999_999:
        return str(n)
    parts = []
    rem = n

    if rem >= 1000:
        thousands = rem // 1000
        rem %= 1000
        if thousands >= 100:
            parts.append(HUNDREDS[thousands // 100])
        t = thousands % 100
        if t >= 20:
            parts.append(TENS[t // 10])
            o = t % 10
            if o == 1:
                parts.append("одна")
            elif o == 2:
                parts.append("две")
            elif 3 <= o <= 9:
                parts.append(ONES[o])
        elif t >= 10:
            parts.append(TEENS[t - 10])
        elif t > 0:
            if t == 1:
                parts.append("одна")
            elif t == 2:
                parts.append("две")
            else:
                parts.append(ONES[t])

        last_two = thousands % 100
        last_one = thousands % 10
        if 11 <= last_two <= 19:
            parts.append("тысяч")
        elif last_one == 1:
            parts.append("тысяча")
        elif 2 <= last_one <= 4:
            parts.append("тысячи")
        else:
            parts.append("тысяч")

    r = rem
    if r >= 100:
        parts.append(HUNDREDS[r // 100])
    t = r % 100
    if t >= 20:
        parts.append(TENS[t // 10])
        if t % 10 != 0:
            parts.append(ONES[t % 10])
    elif t >= 10:
        parts.append(TEENS[t - 10])
    elif t > 0:
        parts.append(ONES[t])

    return " ".join(parts)


ORDINALS = {
    1: "первый", 2: "второй", 3: "третий", 4: "четвертый", 5: "пятый",
    6: "шестой", 7: "седьмой", 8: "восьмой", 9: "девятый", 10: "десятый",
    11: "одиннадцатый", 12: "двенадцатый", 13: "тринадцатый", 14: "четырнадцатый",
    15: "пятнадцатый", 16: "шестнадцатый", 17: "семнадцатый", 18: "восемнадцатый",
    19: "девятнадцатый", 20: "двадцатый",
}

ANGLICISMS = {
    "synergy": "синергия", "tv": "тв", "pink": "пинк", "sony": "сони",
    "samsung": "самсунг", "apple": "эпл", "iphone": "айфон", "google": "гугл",
    "youtube": "ютуб", "facebook": "фейсбук", "instagram": "инстаграм",
    "netflix": "нетфликс", "spotify": "спотифай", "whatsapp": "ватсап",
    "telegram": "телеграм", "vk": "вк", "ok": "ок", "aliexpress": "алиэкспресс",
}


def normalize_for_wer(text: str) -> list[str]:
    """Normalize text to word list for WER computation.

    Mirrors the logic in crates/gigastt/tests/benchmark.rs as closely as
    possible so cross-tool numbers are comparable.
    """
    text = text.lower()
    text = text.replace("ё", "е")
    text = text.replace("-", " ")
    # keep only alphanumerics and whitespace
    text = "".join(c for c in text if c.isalnum() or c.isspace())

    words = text.split()

    # Merge digit groups: "60 000" -> "60000"
    merged = []
    i = 0
    while i < len(words):
        w = words[i]
        if w.isdigit():
            m = w
            while i + 1 < len(words) and words[i + 1].isdigit() and len(words[i + 1]) == 3:
                i += 1
                m += words[i]
            merged.append(m)
        else:
            merged.append(w)
        i += 1

    # Resolve ordinals: "5 й" -> "пятый"
    resolved = []
    i = 0
    while i < len(merged):
        if i + 1 < len(merged) and merged[i + 1] == "й" and merged[i].isdigit():
            n = int(merged[i])
            if n in ORDINALS:
                resolved.append(ORDINALS[n])
                i += 2
                continue
        resolved.append(merged[i])
        i += 1

    # Convert cardinal numbers to words
    converted = []
    for w in resolved:
        if w.isdigit():
            converted.extend(number_to_words(int(w)).split())
        else:
            converted.append(w)

    # Transliterate anglicisms
    final = [ANGLICISMS.get(w, w) for w in converted]
    return final


def word_edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance at word level."""
    m, n = len(reference), len(hypothesis)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def compute_wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Returns (wer_percent, errors, ref_word_count)."""
    ref_words = normalize_for_wer(reference)
    hyp_words = normalize_for_wer(hypothesis)
    errors = word_edit_distance(ref_words, hyp_words)
    ref_count = len(ref_words)
    wer = (errors / ref_count * 100.0) if ref_count > 0 else 0.0
    return wer, errors, ref_count


def audio_duration(wav_path: str) -> float:
    """Get duration in seconds using ffprobe or wave module."""
    try:
        import wave
        with wave.open(wav_path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return frames / rate
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# --- Reproducibility metadata helpers ---


def file_sha256(path: str) -> Optional[str]:
    """Return the SHA-256 hex digest of a file, or None if unavailable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def collect_host_metadata() -> dict:
    """Collect host hardware and OS metadata."""
    ram_bytes: Optional[int] = None
    try:
        import psutil

        ram_bytes = psutil.virtual_memory().total
    except Exception:
        pass

    return {
        "cpu": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "ram_bytes": ram_bytes,
        "os": platform.platform(),
        "python_version": platform.python_version(),
    }


def collect_dataset_metadata(
    dataset_name: str = "golos", version: Optional[str] = None
) -> dict:
    """Collect dataset source metadata.

    Defaults to the Golos crowd subset by SberDevices.
    """
    return {
        "name": dataset_name,
        "version": version,
        "source": "https://github.com/salute-developers/golos",
        "attribution": "Golos by SberDevices",
        "license": "CC-BY-4.0",
        "manifest_path": str(manifest_path()),
    }


def collect_engine_metadata(runner) -> dict:
    """Collect engine name, binary/model paths, version, and model hashes."""
    meta: dict[str, object] = {"name": getattr(runner, "name", type(runner).__name__)}

    # Model identifiers
    for attr in ("model_dir", "model_name", "model_size", "download_dir"):
        val = getattr(runner, attr, None)
        if val is not None:
            meta[attr] = str(val)

    private_model_path = getattr(runner, "_model_path", None)
    if private_model_path is not None:
        meta["model_path"] = str(private_model_path)

    # Binary / version
    binary = getattr(runner, "_binary", None)
    if binary is not None:
        meta["binary"] = str(binary)
        try:
            result = subprocess.run(
                [str(binary), "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            output = (result.stdout + result.stderr).strip()
            if output:
                meta["version"] = output.splitlines()[0]
        except Exception:
            pass

    # Model hash (file or directory contents)
    model_path = private_model_path or getattr(runner, "model_dir", None)
    if model_path is None and hasattr(runner, "model_size"):
        # faster-whisper stores under cache by model_size
        model_path = Path.home() / ".cache" / "huggingface" / "hub"
    if model_path is not None:
        p = Path(model_path)
        if p.is_file():
            meta["model_sha256"] = file_sha256(str(p))
        elif p.is_dir():
            meta["model_path"] = str(p)
            # Hash the first ONNX/bin file found to give a stable fingerprint
            for candidate in sorted(p.rglob("*")):
                if candidate.is_file() and candidate.suffix in {".onnx", ".bin", ".pt", ".ckpt"}:
                    meta["model_sha256"] = file_sha256(str(candidate))
                    meta["model_hashed_file"] = str(candidate.relative_to(p))
                    break

    return meta


def collect_repro_metadata(
    runners: list,
    dataset_name: str = "golos",
    dataset_version: Optional[str] = None,
) -> dict:
    """Aggregate reproducibility metadata for a benchmark run."""
    return {
        "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": collect_host_metadata(),
        "dataset": collect_dataset_metadata(dataset_name, dataset_version),
        "engines": [collect_engine_metadata(r) for r in runners],
    }
