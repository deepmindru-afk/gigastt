"""gigastt WebSocket streaming runner (same server as REST, different cache key)."""

from streaming import STREAM_CHUNK_MS, STREAM_PROTOCOL_VERSION, STREAM_SAMPLE_RATE, transcribe_ws

from .gigastt import GIGASTT_CACHE_SCHEMA_VERSION, GigasttRunner


class GigasttStreamRunner(GigasttRunner):
    """Live `/v1/ws` path: real-time 100 ms PCM16 chunks, then Stop.

    Shares process start/stop with :class:`GigasttRunner`. ``name`` and
    ``cache_config`` differ so REST hypotheses never poison the stream cache.
    """

    name = "gigastt-stream"

    def __init__(
        self,
        model_dir: str | None = None,
        use_int8: bool = True,
        port: int = 9877,
        manage_server: bool = True,
        pace: bool = True,
    ):
        super().__init__(model_dir=model_dir, use_int8=use_int8, port=port)
        self.manage_server = manage_server
        self.pace = pace

    @property
    def cache_config(self) -> str:
        return (
            f"{self.model_dir}:{self.use_int8}:{GIGASTT_CACHE_SCHEMA_VERSION}:"
            f"stream:{STREAM_PROTOCOL_VERSION}:chunk{STREAM_CHUNK_MS}:sr{STREAM_SAMPLE_RATE}"
        )

    def is_available(self) -> bool:
        # Do not start a server here. GigasttRunner.is_available() binds
        # :9877 during runner selection; a second serve on the same port
        # would race `--mode both` (REST then WS, sequential).
        return self._find_binary()

    def _start_server(self):
        if not self.manage_server:
            return
        super()._start_server()

    def _stop_server(self):
        if not self.manage_server:
            return
        super()._stop_server()

    def transcribe(self, wav_path: str) -> tuple[str, float]:
        text, elapsed, _session = transcribe_ws(
            wav_path, port=self.port, chunk_ms=STREAM_CHUNK_MS, pace=self.pace
        )
        return text, elapsed
