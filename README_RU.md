<p align="center">
  <h1 align="center">gigastt</h1>
  <p align="center"><strong>Встраиваемое локальное распознавание русской речи — один бинарник на Rust, без облака, MIT-чистые веса.</strong></p>
  <p align="center">
    <a href="https://github.com/ekhodzitsky/gigastt/actions"><img src="https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://codecov.io/gh/ekhodzitsky/gigastt"><img src="https://codecov.io/gh/ekhodzitsky/gigastt/branch/main/graph/badge.svg" alt="codecov"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/v/gigastt.svg" alt="crates.io"></a>
    <a href="https://docs.rs/gigastt-core"><img src="https://docs.rs/gigastt-core/badge.svg" alt="docs.rs"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
  </p>
  <p align="center"><a href="README.md">English</a> | <b>Русский</b></p>
</p>

---

gigastt превращает любую машину в приватный сервер распознавания русской речи — или встраивает тот же движок в Rust-приложение или Android-бинарник. Открытая модель **GigaAM v3** работает полностью локально через ONNX Runtime: без облака, без API-ключей.

## Обзор

| Локально, приватно | Встраивание + стриминг | Точный русский | Маленький и real-time |
|---|---|---|---|
| Без облака и ключей — после разовой загрузки модели инференс 100% локальный. MIT-движок на MIT-весах, пригоден для коммерции. | Один бинарник, C-ABI FFI для мобильных или крейт `gigastt-core` — с инкрементальными partial'ами по WebSocket, без Python. | Самый точный на 3 из 4 русских доменов: far-field 4.08%, телефон 18.50%, YouTube 10.91%; ничья на чистой речи. | ~225 МБ INT8, RTF ~0.10 (~10× быстрее реального времени на CPU), холодный старт 0.94 с. |

**WER** чистая 3.55% / far-field 4.08% / телефон 18.50% / YouTube 10.91%  ·  **held-out** CV **2.63%** (лучше Vosk+FW) · FLEURS 5.26% (лидер FW 3.84) · RuLS **4.21%** (лучше Vosk+FW) · SOVA device: Vosk впереди  · ToneWebinars: лидер FW 8.33 (gigastt 13.0)  ·  **RTF** ~0.10  ·  **Модель** ~225 МБ INT8  ·  **Холодный старт** 0.94 с  ·  **RAM** ~46 МБ resident (~277 МБ `ps`) · ~66 МБ pool-2 (~510 МБ `ps`)  ·  **Стриминг** первый partial ~0.78 с

> Голова GigaAM v3 `rnnt`, INT8, Apple M1 CPU, 1000 сэмплов на домен (FLEURS n=775), отказы = 100% WER, 95% bootstrap CI. Все конкуренты замерены одинаково — тем же [харнессом](docs/benchmarks.md), манифестами и нормализацией.

## Как сравнивается

WER (%) на четырёх русских доменах, меньше — лучше, плюс все оси, по которым выбирают движок. gigastt — голова `rnnt`, INT8.

| Движок | Чистая | Far-field | Телефон | YouTube | RTF | Диск | Пик RAM | Холодный старт | Стриминг | Пункт. |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| **gigastt** (GigaAM v3 `rnnt`) | 3.55 | **4.08** | **18.50** | **10.91** | 0.10 | ~225 МБ | ~46 МБ · ~66 МБ pool-2 | **0.94 с** | **Да** — инкр. WS | **Да** |
| Vosk 0.54 (Zipformer2) | **2.97** | 6.29 | 22.74 | 17.24 | ~0.03 | 966 МБ | 560 МБ | 1.16 с | Да (сервер) | Аддон |
| T-one (beam + LM) | 6.61 | 14.62 | 21.73 | 23.23 | 0.065 | 138 МБ + 5.5 ГБ LM | — | — | Да (300 мс) | Нет |
| T-one (greedy, без LM) | 7.85 | 17.22 | 22.37 | 26.54 | 0.065 | 138 МБ | 672 МБ | 1.87 с | Да (300 мс) | Нет |
| whisper.cpp (Large v3) | 15.26 | 17.91 | 32.73 | 22.61 | 0.36–0.77 | 2.9 ГБ | — | — | Нет | Да |
| faster-whisper (Large v3) | 15.53 | 17.34 | 24.93 | 15.45 | &gt;1.0 | 2.9 ГБ | 2619 МБ | 8.2 с | Нет | Да |
| faster-whisper-turbo | 14.45 | 18.30 | 26.58 | 15.45 | &gt;1.0 | 1.6 ГБ | 2154 МБ | 6.8 с | Нет | Да |

Условия: Apple M1, CPU EP, INT8/greedy, 1000 сэмплов на домен (чистая речь 992; turbo — срез 300), 95% bootstrap CI. Чистая речь 3.55 (2.9–4.2) пересекается с Vosk 0.54 2.97 (2.4–3.6) — статистическая ничья; победы на far-field / телефоне / YouTube CI-раздельны. RTF &gt; 1.0 = медленнее реального времени на CPU. RAM gigastt — resident footprint (dirty + сжатые страницы, macOS `footprint`) после тёплых декодов, замер на Apple M1 Pro (INT8): ~46 МБ при `--pool-size 1`, ~66 МБ при дефолтном `--pool-size 2`; `ps` RSS показывает ~277 / ~510 МБ, потому что считает общий memory-mapped образ модели — эти чистые страницы ОС забирает под давлением памяти. «—» = не замерялось. Полная методология и оговорки: [Benchmarks](docs/benchmarks.md).

**Raspberry Pi / edge:** на «железе» Pi ещё **не замерялось** — никаких заявлений про RTF, RAM или холодный старт на edge-устройствах; статус и протокол: [Benchmarks § Edge / Raspberry Pi](docs/benchmarks.md#edge--raspberry-pi) и [edge-роадмап](specs/edge-raspberry-pi-roadmap.md).

**Стриминг:** Whisper-движки работают только офлайн — никаких partial'ов во время речи. gigastt отдаёт настоящие инкрементальные partial'ы по WebSocket (первый ~0.78 с на CPU) из одного самодостаточного бинарника без Python; Vosk-server и T-one (чанки 300 мс) тоже стримят. То есть стриминг — чистая победа над Whisper-семейством; а перед Vosk / T-one преимущество в упаковке — инкрементальные partial'ы плюс C-ABI FFI в одном бинарнике, а не в меньшей задержке.

**Пунктуация и регистр:** gigastt выдаёт читаемый русский из коробки — нативно на голове `e2e_rnnt` или маленьким встроенным проходом RuPunct + ITN на дефолтной `rnnt` (`--punctuation` / `--itn`, авто-докачка). Это на уровне Whisper-движков (у них пунктуация нативная) и лучше русских специалистов — Vosk требует отдельный аддон `recasepunc` (отдельная модель, сопоставимая по размеру с распознавателем), а T-one не даёт пунктуации вовсе.

## Область применения и честные оговорки

Где выигрывают конкуренты и когда gigastt не нужен:

- **Чистая речь — ничья, не победа** — gigastt 3.55% (2.9–4.2) vs Vosk 0.54 2.97% (2.4–3.6); CI пересекаются, точечная оценка Vosk чуть впереди.
- **Русский в первую очередь, узкий multilingual** — дефолтные головы `rnnt` / `e2e_rnnt` только русские; опциональные `ml_ctc` / `ml_ctc_large` добавляют лишь ru/en/kk/ky/uz. Для реальной широты языков — Vosk (20+) или whisper.cpp / faster-whisper / sherpa-onnx (~99). gigastt — специалист.
- **Не лидер по скорости** — Vosk (RTF ~0.03) и T-one (~0.06) быстрее; gigastt (~0.10) уверенно real-time, но не самый быстрый.
- **RAM крошечная, но легко читается неправильно** — resident footprint ~46 МБ при `--pool-size 1` / ~66 МБ при дефолтном pool 2, самый лёгкий в таблице (Vosk 0.54 — 560 МБ, T-one greedy — 672 МБ; дополнительный слот пула стоит всего ~20 МБ resident). Но `ps` / Мониторинг системы показывает ~277 / ~510 МБ, потому что RSS считает общий memory-mapped образ модели; ОС забирает эти чистые страницы под давлением памяти, так что закладывать нужно именно resident-цифру.
- **Стриминг — буферизованный/чанковый** поверх офлайн RNN-T, не нативно-стримящая акустическая модель; ~0.78 с до первого partial — не заявка на минимальную задержку.
- **Пересечение с обучающими данными** — GigaAM v3 обучена в основном на Golos; Golos / OpenSTT — in-distribution upper bound. **Held-out** (CV / FLEURS / RuLS / SOVA / Podlodka / ToneWebinars) — второй столбец; см. [Benchmarks](docs/benchmarks.md#held-out--additional-public-sets--wer--95-ci).

## Установка

```sh
# Homebrew (macOS arm64 / Linux x86_64)
brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt && brew install gigastt

# Windows x86_64 (и другие готовые тройки) — tarball из Releases
# https://github.com/ekhodzitsky/gigastt/releases

# crates.io — нужен protoc в PATH (brew install protobuf / apt install protobuf-compiler)
cargo install gigastt

# Готовый образ из GHCR (CPU, multi-arch amd64+arm64; CUDA-вариант: :cuda)
docker pull ghcr.io/ekhodzitsky/gigastt:latest

# Или соберите свой образ (CUDA: Dockerfile.cuda; вшить модель в образ: --build-arg GIGASTT_BAKE_MODEL=1)
docker build -t gigastt . && docker run -p 9876:9876 gigastt
```

Встраивание вместо сервера? `npm install gigastt` (Node.js) · `pip install gigastt` (Python на PyPI) · Swift / Kotlin биндинги в работе — все оборачивают тот же движок, модель подкладывается отдельно: [In-process quickstarts](docs/quickstarts.md).

Модель GigaAM v3 INT8 (~225 МБ) скачивается при первом запуске (lean-бандл с GitHub Releases). Runtime — **только INT8**: нет загрузки и инференса FP32.

> Сборка также тянет prebuilt onnxruntime по сети (ort `download-binaries`); гарантия on-device / без облака покрывает **runtime-инференс**, а не сборку. Air-gapped-сборка — в [Architecture](docs/architecture.md).

## Быстрый старт

```sh
$ gigastt transcribe recording.wav
Привет, как дела?

# Пакетная обработка папки (txt + json на файл, 2 воркера):
$ gigastt transcribe-batch samples/ out/

# Или watch: файлы, появившиеся в inbox/, распознаются и переносятся:
$ gigastt watch inbox/ out/ --move-to inbox/done/

# Или сервер — WebSocket + REST + SSE на одном порту (только loopback):
$ gigastt serve
# WebSocket  ws://127.0.0.1:9876/v1/ws
# REST       http://127.0.0.1:9876/v1/transcribe
# OpenAI     http://127.0.0.1:9876/v1/audio/transcriptions
```

## Возможности

| Возможность | Поддержка |
|---|---|
| Головы | `rnnt` (34-токенный char, дефолт — ниже всех WER) · `e2e_rnnt` (1025-токенный BPE, пунктуация / регистр / ITN встроены) · `ml_ctc` / `ml_ctc_large` (GigaAM Multilingual charwise-CTC, 220M / 600M, 71-токенный multilingual char — ru/en/kk/ky/uz) |
| Постобработка | опциональные пунктуация, регистр и русский ITN — нативно на `e2e_rnnt` или встроенный проход RuPunct + ITN на `rnnt` (авто-докачка; `--punctuation` / `--itn`), переопределяемо на каждый запрос (`?punctuation=` / `?itn=` / `?vad=`) |
| Доставка | статический бинарник · C-ABI FFI `cdylib` (Android / mobile) · крейт `gigastt-core` (без серверных зависимостей) |
| Провайдеры исполнения | CPU (любая платформа) · CoreML EP (macOS ARM64) · CUDA 12+ (Linux x86_64) · NNAPI (Android) · [ANE](docs/ane-backend.md) (`--features ane`, macOS ARM64 — энкодер ≈15.6× на Neural Engine, тёплый e2e ≈10× быстрее CPU-сборки, WER ≈1.11% против `ort`; только файловый режим) · [Candle/Metal](docs/candle-backend.md) (`--features candle`, экспериментальный — вывод побайтово совпадает с `ort`) |
| Стриминг | инкрементальные partial'ы по WebSocket · REST + SSE для файлов · OpenAI-совместимый `/v1/audio/transcriptions` · один порт 9876 |
| Аудио на вход | WAV · M4A/AAC · MP3 · OGG/Vorbis · OGG/Opus (`.opus`) · WebM/Opus (браузерный `MediaRecorder`) · FLAC (авто-микс в моно) |
| Стерео-телефония | Опциональный режим «канал = спикер» (`--stereo-speakers` в CLI / `channels=split` в REST) помечает левый/правый каналы как `speaker_0` и `speaker_1` |
| Диаризация | Эмбеддинги WeSpeaker ResNet34 + кластеризация polyvoice, встроена по умолчанию (speaker-модель скачивается командой `gigastt download`, отказ — `--skip-diarization`) — файлы включают её на каждый запрос (`?diarization=true`, несовместимо с `channels=split`), живые сессии — через WS `Configure`; слова и сегменты получают метки `speaker` |
| Асинхронные задачи | Очередь для длинных файлов / batch-распознавания через `/v1/jobs` (включается `--enable-jobs`): submit, poll, отмена, SSE-прогресс, retry и TTL-евикция |
| Клиентские SDK | Типизированные WebSocket-клиенты для протокола v1.0 с переподключением по `retry_after_ms`: [Go (`sdks/go`)](sdks/go) · [TypeScript `@gigastt/client` (`sdks/js`)](sdks/js) |
| Экспорт | JSON · TXT · SRT · VTT · Markdown — пословные тайминги + confidence или посегментно (`?segments=true` JSON, `### [mm:ss]` Markdown) |
| Защита сервера | loopback по умолчанию · origin-allowlist · rate-limiting по IP · graceful drain · Prometheus `/metrics` на отдельном порту · loopback-only горячая перезагрузка модели (`POST /v1/admin/reload`) |

## Документация

| Гайд | Содержание |
|---|---|
| **[Индекс docs](docs/README.md)** | Полная карта гайдов в `docs/` |
| **[Книга рецептов](https://ekhodzitsky.github.io/gigastt/)** | Сценарные рецепты (EN + RU): установка → транскрибация → стриминг → деплой |
| **[API](docs/api.md)** | WebSocket-протокол, REST + SSE, jobs, admin reload, коды ошибок, клиенты |
| **[Benchmarks](docs/benchmarks.md)** | WER / RTF / footprint против 6 движков на 4 русских доменах, с оговорками |
| **[Architecture](docs/architecture.md)** | Пайплайн, crates, аппаратное ускорение, INT8, структура проекта |
| **[Android / FFI](ANDROID.md)** | Встраивание через C-ABI на Android |
| **[CLI](docs/cli.md)** · **[Deployment](docs/deployment.md)** · **[Security](SECURITY.md)** · **[Troubleshooting](docs/troubleshooting.md)** | Справочник и эксплуатация |

## Требования

Rust **1.88+**, `protoc` в `PATH` (только на этапе сборки — крейт quantize регенерирует ONNX-типы). macOS 14+ (Apple Silicon, CoreML), Linux x86_64 (опц. NVIDIA CUDA 12+ через образ GHCR `:cuda`), Linux aarch64 или Windows x86_64 CPU (готовый tarball). **~250–400 МБ диска** для lean INT8 + бинарника (опциональные punct/VAD — отдельно), ~66 МБ resident RAM при дефолтном `--pool-size 2` (~46 МБ на одну сессию; `ps` RSS показывает ~510 / ~277 МБ из-за общего memory-mapped образа модели). Крейт `gigastt-core` без серверных зависимостей: `gigastt-core = "2.17"`.

## Лицензия

MIT — см. [LICENSE](LICENSE).

> **Данные бенчмарка** под `benchmark/` — **не** MIT: транскрипты OpenSTT (`openstt_*`, CC BY-NC 4.0) и Golos (`golos_*`, Sber Public License) сохраняют свои non-commercial лицензии. См. [`NOTICE`](NOTICE) и [`benchmark/DATA_LICENSE`](benchmark/DATA_LICENSE).

## Благодарности

- [**GigaAM**](https://github.com/salute-developers/GigaAM) от [SberDevices](https://github.com/salute-developers) — модель распознавания
- [**onnx-asr**](https://github.com/istupakov/onnx-asr) от [@istupakov](https://github.com/istupakov) — ONNX-экспорт и референс
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) · [**ort**](https://github.com/pykeio/ort) — движок инференса и Rust-биндинги
