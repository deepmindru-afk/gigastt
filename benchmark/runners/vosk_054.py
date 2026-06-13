"""Runner for Vosk 0.54 — NOTE: NOT a drop-in for the Kaldi vosk-api.

De-risk finding (2026-06-14): ``vosk-model-ru-0.54`` on alphacephei is a **Zipformer2
ONNX** bundle — it ships ``am-onnx/``, ``decode-onnx.py``, ``lang/``, ``lm/`` and is
*not* a Kaldi model. ``vosk.Model()`` rejects it ("Folder does not contain model
files"), so this runner cannot reuse the 0.42 ``KaldiRecognizer`` path. Integrating
0.54 needs an onnxruntime backend (its bundled ``decode-onnx.py``) or sherpa-onnx,
both of which support Zipformer2 ONNX models.

Until that backend is wired, ``is_available()`` returns ``False`` so the full suite
skips Vosk 0.54 cleanly instead of recording 100% failures. Tracked as a follow-up.
"""

import os

from .vosk import VoskRunner


class Vosk054Runner(VoskRunner):
    name = "vosk-0.54"

    def __init__(self, model_name: str | None = None, download_dir: str | None = None):
        if model_name is None:
            model_name = os.environ.get("BENCHMARK_VOSK054_MODEL", "vosk-model-ru-0.54")
        super().__init__(model_name=model_name, download_dir=download_dir)

    def is_available(self) -> bool:
        # vosk-model-ru-0.54 is a Zipformer2 ONNX bundle, not a Kaldi model the
        # vosk-api can load. Skip until a sherpa-onnx / onnxruntime backend is wired.
        print(
            "[vosk-0.54] Not available: vosk-model-ru-0.54 is a Zipformer2 ONNX model, "
            "not loadable by vosk-api; needs a sherpa-onnx/onnxruntime backend (follow-up)."
        )
        return False
