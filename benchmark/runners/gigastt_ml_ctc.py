"""Runners for gigastt's GigaAM Multilingual charwise-CTC heads (ml_ctc / ml_ctc_large).

Both reuse the base :class:`GigasttRunner` (it starts ``gigastt serve`` and
transcribes over the REST ``/v1/transcribe`` endpoint). The only difference is
the model directory the server is pointed at: ``gigastt serve`` auto-detects the
recognition head from the encoder file on disk, so a directory holding the
Multilingual CTC INT8 encoder + vocab makes the server run that head.

Model directories are configurable via env vars and default to where
``gigastt download --model-variant ml_ctc[_large] --model-dir <dir>`` places them
in this repo's benchmark setup:

- ``BENCHMARK_GIGASTT_ML_CTC_MODEL_DIR``       (default ``~/.gigastt/models-mlctc``)
- ``BENCHMARK_GIGASTT_ML_CTC_LARGE_MODEL_DIR`` (default ``~/.gigastt/models-mlctc-large``)

`subprocess` does not expand ``~`` (no shell), so the paths are expanded here
before being passed to ``--model-dir``.
"""

import os
from pathlib import Path

from .gigastt import GigasttRunner

# Bump when the CTC transcription behavior changes in a way that invalidates
# previously cached results (model files, preprocessing, decode).
GIGASTT_ML_CTC_CACHE_SCHEMA_VERSION = "v2.11.0"


def _expand(path: str) -> str:
    return str(Path(path).expanduser())


class GigasttMlCtcRunner(GigasttRunner):
    """gigastt with the GigaAM Multilingual charwise-CTC 220M head (``ml_ctc``)."""

    name = "gigastt-ml-ctc"

    def __init__(self, model_dir: str | None = None, port: int = 9879):
        model_dir = model_dir or os.environ.get(
            "BENCHMARK_GIGASTT_ML_CTC_MODEL_DIR", "~/.gigastt/models-mlctc"
        )
        super().__init__(model_dir=_expand(model_dir), use_int8=True, port=port)

    @property
    def cache_config(self) -> str:
        return f"{self.model_dir}:ml_ctc:{GIGASTT_ML_CTC_CACHE_SCHEMA_VERSION}"


class GigasttMlCtcLargeRunner(GigasttRunner):
    """gigastt with the GigaAM Multilingual charwise-CTC 600M head (``ml_ctc_large``)."""

    name = "gigastt-ml-ctc-large"

    def __init__(self, model_dir: str | None = None, port: int = 9880):
        model_dir = model_dir or os.environ.get(
            "BENCHMARK_GIGASTT_ML_CTC_LARGE_MODEL_DIR", "~/.gigastt/models-mlctc-large"
        )
        super().__init__(model_dir=_expand(model_dir), use_int8=True, port=port)

    @property
    def cache_config(self) -> str:
        return f"{self.model_dir}:ml_ctc_large:{GIGASTT_ML_CTC_CACHE_SCHEMA_VERSION}"
