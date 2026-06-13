"""Runner for Vosk 0.54 (Zipformer2) — a newer Russian model alongside 0.42.

Reuses ``VoskRunner``'s alphacephei ZIP download + ``KaldiRecognizer`` path; only
the default model name differs, so both Vosk versions show up side-by-side in the
results table. The exact model id should be confirmed against
https://alphacephei.com/vosk/models (override with ``BENCHMARK_VOSK054_MODEL``)
before a full run.
"""

import os

from .vosk import VoskRunner


class Vosk054Runner(VoskRunner):
    name = "vosk-0.54"

    def __init__(self, model_name: str | None = None, download_dir: str | None = None):
        if model_name is None:
            model_name = os.environ.get("BENCHMARK_VOSK054_MODEL", "vosk-model-ru-0.54")
        super().__init__(model_name=model_name, download_dir=download_dir)
