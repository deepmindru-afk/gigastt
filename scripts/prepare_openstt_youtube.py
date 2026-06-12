#!/usr/bin/env python3
"""Download OpenSTT public_youtube700_val validation set and prepare a manifest.

Dataset provenance:
  - Name: OpenSTT (Russian Open Speech To Text)
  - Authors: Alexander Veysov, Anna Slizhikova, Diliara Nurtdinova, Dmitry Voronin
    (snakers4)
  - Repository: https://github.com/snakers4/open_stt
  - Paper: Slizhikova et al., "Russian Open Speech To Text (STT/ASR) Dataset"
  - License: CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
    Commercial usage is available after agreement with the dataset authors.

This script downloads the official public_youtube700_val validation subset, which
contains ~7 300 manually-annotated YouTube utterances (~4.5 hours).

Primary source (direct Azure Open Datasets links):
  - Archive: https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/public_youtube700_val.tar.gz
  - Manifest: https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/manifests/public_youtube700_val.csv
  - Unpacked files: https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus_unpacked/

The audio is normalized to 16 kHz mono PCM16 WAV. A deterministic 1000-sample
slice is selected with random.seed(42) and written to
benchmark/manifests/openstt_youtube.json.
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SUBSET = "public_youtube700_val"
OPENSTT_ARCHIVE_URL = (
    "https://azureopendatastorage.blob.core.windows.net/openstt/"
    f"ru_open_stt_opus/archives/{SUBSET}.tar.gz"
)
OPENSTT_MANIFEST_URL = (
    "https://azureopendatastorage.blob.core.windows.net/openstt/"
    f"ru_open_stt_opus/manifests/{SUBSET}.csv"
)
OPENSTT_UNPACKED_BASE_URL = (
    "https://azureopendatastorage.blob.core.windows.net/openstt/"
    "ru_open_stt_opus_unpacked"
)

SAMPLE_RATE = 16000
SEED = 42
SLICE_SIZE = 1000


def home_dir() -> Path:
    return Path.home()


def default_benchmark_manifest_dir() -> Path:
    return Path(__file__).parent.parent / "benchmark" / "manifests"


def download_url(
    url: str,
    dest: Path,
    chunk_size: int = 65536,
    max_retries: int = 5,
    backoff: float = 1.0,
) -> None:
    """Download a file from *url* to *dest* with retries on transient errors."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": f"gigastt-prepare-{SUBSET}"})
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=60) as response:
                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
            return
        except (HTTPError, URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                break
            sleep_sec = backoff * (2 ** attempt)
            print(f"  retry {url} in {sleep_sec:.1f}s ({exc})", file=sys.stderr)
            time.sleep(sleep_sec)
    raise last_exc or RuntimeError(f"Failed to download {url}")


def _is_normalized_wav(path: Path) -> bool:
    """Return True if *path* is a 16 kHz mono 16-bit PCM WAV."""
    try:
        with wave.open(str(path), "rb") as w:
            return (
                w.getnchannels() == 1
                and w.getframerate() == SAMPLE_RATE
                and w.getsampwidth() == 2
            )
    except Exception:
        return False


def _copy_normalized_wav(src: Path, dst: Path) -> None:
    """Copy a normalized WAV, ensuring the RIFF header is well-formed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(src), "rb") as in_wav:
        with wave.open(str(dst), "wb") as out_wav:
            out_wav.setnchannels(in_wav.getnchannels())
            out_wav.setsampwidth(in_wav.getsampwidth())
            out_wav.setframerate(in_wav.getframerate())
            out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))


def convert_to_wav(src: Path, dst: Path) -> None:
    """Convert *src* to a 16 kHz mono PCM16 WAV at *dst*."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _is_normalized_wav(src):
        _copy_normalized_wav(src, dst)
        return

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                str(dst),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to convert non-standard audio files"
        ) from exc


def load_manifest_csv(path: Path) -> list[dict]:
    """Load the OpenSTT manifest CSV.

    Columns: wav_path, text_path, duration
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            wav_rel, txt_rel, duration = (x.strip() for x in row)
            rows.append(
                {
                    "wav_rel": wav_rel,
                    "txt_rel": txt_rel,
                    "duration": float(duration),
                }
            )
    return rows


def _download_one_unpacked(row: dict, extracted_dir: Path) -> None:
    """Download a single wav+txt pair from the unpacked Azure blob."""
    for rel_key in ("wav_rel", "txt_rel"):
        rel = row[rel_key]
        dest = extracted_dir / rel
        if dest.exists():
            continue
        url = f"{OPENSTT_UNPACKED_BASE_URL}/{rel}"
        try:
            download_url(url, dest)
        except HTTPError as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _download_unpacked(rows: list[dict], extracted_dir: Path, workers: int = 8) -> None:
    """Download selected wav+txt files from the unpacked Azure blob in parallel."""
    extracted_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one_unpacked, row, extracted_dir): row
            for row in rows
        }
        for future in as_completed(futures):
            future.result()


def build_sample(row: dict, extracted_dir: Path, dst_dir: Path) -> dict:
    """Copy/convert one audio file and read its reference text."""
    src_wav = extracted_dir / row["wav_rel"]
    src_txt = extracted_dir / row["txt_rel"]

    if not src_txt.exists():
        raise FileNotFoundError(f"Missing transcript file: {src_txt}")
    reference = src_txt.read_text(encoding="utf-8").strip()

    dst_wav = dst_dir / row["wav_rel"]
    if src_wav.exists():
        convert_to_wav(src_wav, dst_wav)
    else:
        _download_one_unpacked(row, extracted_dir)
        convert_to_wav(src_wav, dst_wav)

    return {
        "filename": str(row["wav_rel"]),
        "reference": reference,
        "duration": round(row["duration"], 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Prepare OpenSTT {SUBSET} benchmark subset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=home_dir() / ".gigastt" / "benchmarks" / "openstt_youtube",
        help="Directory to store converted WAV files",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=default_benchmark_manifest_dir() / "openstt_youtube.json",
        help="Output path for the committed manifest JSON",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help=f"Path to a pre-downloaded {SUBSET}.tar.gz archive",
    )
    parser.add_argument(
        "--use-unpacked-source",
        action="store_true",
        help=(
            "Download individual wav/txt files from the unpacked Azure blob "
            "instead of the full archive. Useful for creating the 1000-sample "
            "manifest without fetching the full archive."
        ),
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=SLICE_SIZE,
        help="Number of deterministic samples to include in the manifest",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for sample selection",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel download workers when using --use-unpacked-source",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    work_dir = Path(tempfile.mkdtemp(prefix=f"{SUBSET}_"))
    try:
        csv_path = work_dir / f"{SUBSET}.csv"
        print(f"Downloading manifest from {OPENSTT_MANIFEST_URL} ...")
        download_url(OPENSTT_MANIFEST_URL, csv_path)

        rows = load_manifest_csv(csv_path)
        total = len(rows)
        if total == 0:
            print("Manifest is empty", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {total} entries from manifest")

        rng = random.Random(args.seed)
        selected = rng.sample(rows, min(args.slice_size, total))
        selected.sort(key=lambda r: r["wav_rel"])

        extracted_dir = work_dir / "extracted"
        if args.use_unpacked_source:
            print(
                f"Downloading {len(selected)} wav+txt pairs from "
                f"{OPENSTT_UNPACKED_BASE_URL} ..."
            )
            _download_unpacked(selected, extracted_dir, workers=args.workers)
        else:
            archive_path = args.archive
            if archive_path is None:
                archive_path = work_dir / f"{SUBSET}.tar.gz"
                print(f"Downloading archive from {OPENSTT_ARCHIVE_URL} ...")
                download_url(OPENSTT_ARCHIVE_URL, archive_path)
            print(f"Extracting {archive_path} ...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=extracted_dir)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        for row in selected:
            sample = build_sample(row, extracted_dir, args.output_dir)
            samples.append(sample)
            if (len(samples) % 100) == 0 or len(samples) == len(selected):
                print(f"  processed {len(samples)}/{len(selected)} samples")

        manifest = {
            "dataset": "openstt_youtube",
            "audio_root": "~/.gigastt/benchmarks/openstt_youtube",
            "slice_seed": args.seed,
            "slice_size": len(samples),
            "total_available": total,
            "license": "CC BY-NC 4.0",
            "source": "https://github.com/snakers4/open_stt",
            "samples": samples,
        }
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest_output, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(samples)} samples to {args.manifest_output}")
        print(f"Audio files in {args.output_dir}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
