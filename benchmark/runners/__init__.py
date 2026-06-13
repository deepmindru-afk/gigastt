"""ASR benchmark runners."""

from .gigastt import GigasttRunner, GigasttCoreMLRunner
from .whisper_cpp import WhisperCppRunner
from .faster_whisper import FasterWhisperRunner
from .faster_whisper_turbo import FasterWhisperTurboRunner
from .vosk import VoskRunner
from .vosk_054 import Vosk054Runner
from .t_one import TOneRunner

__all__ = [
    "GigasttRunner",
    "GigasttCoreMLRunner",
    "WhisperCppRunner",
    "FasterWhisperRunner",
    "FasterWhisperTurboRunner",
    "VoskRunner",
    "Vosk054Runner",
    "TOneRunner",
]
