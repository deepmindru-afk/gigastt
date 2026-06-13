"""Runner for faster-whisper ``large-v3-turbo`` (faster distilled decoder).

A thin specialization of ``FasterWhisperRunner`` with a fixed model size so the
engine appears as a distinct, stably-named row in the results table instead of
being toggled via an environment variable.
"""

from .faster_whisper import FasterWhisperRunner


class FasterWhisperTurboRunner(FasterWhisperRunner):
    name = "faster-whisper-turbo"

    def __init__(self, device: str = "cpu", compute_type: str = "int8"):
        super().__init__(model_size="large-v3-turbo", device=device, compute_type=compute_type)
