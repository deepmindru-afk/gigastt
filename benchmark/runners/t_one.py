"""Runner for T-one (``voicekit-team/T-one``) — T-Bank's streaming CTC Conformer.

T-one is Apache-2.0 (code *and* weights) and purpose-built for Russian telephony /
call-center streaming — exactly the niche gigastt targets, which is why it belongs
in the comparison. This runner loads it via ``transformers`` + ``torch`` as a
Wav2Vec2-style CTC model.

Best-effort by design: the exact processor / model classes and HF repo id should be
confirmed against the T-one model card before a full run (override the repo with
``BENCHMARK_TONE_MODEL``). Until ``torch`` + ``transformers`` + the weights are
present, ``is_available()`` returns ``False`` and the suite skips T-one. All heavy
imports are lazy so importing this module never fails.
"""

import os
import time
import wave


class TOneRunner:
    name = "t-one"

    def __init__(self, model_id: str | None = None, device: str = "cpu"):
        self.model_id = model_id or os.environ.get("BENCHMARK_TONE_MODEL", "voicekit-team/T-one")
        self.device = device
        self._model = None
        self._processor = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except Exception as e:
            print(f"[t-one] Not available: {e}")
            return False

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForCTC, AutoProcessor
            print(f"[t-one] Loading {self.model_id} ...")
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForCTC.from_pretrained(self.model_id).to(self.device).eval()
        return self._model, self._processor

    def _read_wav_16k_mono(self, wav_path: str):
        import numpy as np

        with wave.open(wav_path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                raise ValueError("T-one runner expects 16kHz mono 16-bit WAV")
            frames = wf.readframes(wf.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe(self, wav_path: str) -> tuple[str, float]:
        import torch

        model, processor = self._load()
        audio = self._read_wav_16k_mono(wav_path)
        start = time.perf_counter()
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(self.device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]
        elapsed = time.perf_counter() - start
        return text.strip().lower(), elapsed
