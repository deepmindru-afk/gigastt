"""Runner for T-one (``t-tech/T-one``) — T-Bank's streaming CTC Conformer (Apache-2.0).

T-one is purpose-built for Russian telephony / call-center streaming — exactly the
niche gigastt targets, which is why it belongs in the comparison. It ships its **own**
inference package ``tone``; it is NOT a generic ``transformers`` model (the HF repo
``t-tech/T-one`` declares a custom ``ToneForCTC`` architecture). The documented offline
path is::

    from tone import StreamingCTCPipeline, read_audio
    pipeline = StreamingCTCPipeline.from_hugging_face()      # pulls t-tech/T-one
    text = pipeline.forward_offline(read_audio("clip.wav"))

Install (pulls torch)::

    uv pip install "git+https://github.com/voicekit-team/T-one.git"

Until ``tone`` is importable, ``is_available()`` returns ``False`` and the suite skips
T-one. All heavy imports are lazy so importing this module never fails.
"""

import time


class TOneRunner:
    name = "t-one"

    def __init__(self, device: str | None = None):
        self.device = device
        self._pipeline = None

    def is_available(self) -> bool:
        try:
            import tone  # noqa: F401
            return True
        except Exception as e:
            print(
                "[t-one] Not available (install: "
                "uv pip install 'git+https://github.com/voicekit-team/T-one.git'): "
                f"{e}"
            )
            return False

    def _load(self):
        if self._pipeline is None:
            from tone import StreamingCTCPipeline
            print("[t-one] Loading StreamingCTCPipeline from t-tech/T-one ...")
            self._pipeline = StreamingCTCPipeline.from_hugging_face()
        return self._pipeline

    @staticmethod
    def _extract_text(result) -> str:
        """``forward_offline`` returns phrase objects (or strings); join defensively."""
        if isinstance(result, str):
            return result
        parts = []
        for item in result or []:
            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(
                    getattr(item, "text", None)
                    or getattr(item, "transcription", None)
                    or str(item)
                )
        return " ".join(p for p in parts if p).strip()

    def transcribe(self, wav_path: str) -> tuple[str, float]:
        from tone import read_audio

        pipeline = self._load()
        audio = read_audio(wav_path)
        start = time.perf_counter()
        result = pipeline.forward_offline(audio)
        elapsed = time.perf_counter() - start
        return self._extract_text(result).lower(), elapsed
