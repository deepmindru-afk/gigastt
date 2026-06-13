<p align="center">
  <h1 align="center">gigastt</h1>
  <p align="center"><strong>Распознавание русской речи на устройстве — WER 8.60% (renorm, golos_crowd_1k, 1k образцов, 95% ДИ 7.5–9.7%)</strong></p>
  <p align="center">Локальный STT-сервер на базе GigaAM v3 — без облака, без API-ключей, полная приватность</p>
  <p align="center">
    <a href="https://github.com/ekhodzitsky/gigastt/actions"><img src="https://github.com/ekhodzitsky/gigastt/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/v/gigastt.svg" alt="crates.io"></a>
    <a href="https://crates.io/crates/gigastt"><img src="https://img.shields.io/crates/d/gigastt.svg" alt="crates.io downloads"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
    <a href="https://github.com/ekhodzitsky/gigastt/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-Keep%20a%20Changelog-orange" alt="Changelog"></a>
  </p>
  <p align="center"><a href="README.md">English</a> | <b>Русский</b></p>
</p>

---

**gigastt** превращает любой компьютер в сервер распознавания русской речи. Один бинарник, одна команда, MIT-чистые веса — всё работает локально.

```sh
brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt
brew install gigastt && gigastt serve
# WebSocket: ws://127.0.0.1:9876/v1/ws
# REST API:  http://127.0.0.1:9876/v1/transcribe
```

## Почему gigastt?

| | gigastt | whisper.cpp | faster-whisper | Vosk | sherpa-onnx | Облачные API |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Модель** | GigaAM v3 | Whisper large-v3 | Whisper large-v3 | Vosk models | разные | вендор |
| **WER (golos_crowd_1k, renorm)** | 8.60% | 15.26% | 15.53% | **4.82%** | зависит от модели | 5–10% |
| **Языки** | русский | 99 | 99 | 20+ | 10+ | 100+ |
| **Стриминг** | буферизованный WebSocket (offline RNN-T) | — | — | WebSocket + gRPC | WebSocket + TCP | по-разному |
| **Batch-задержка (16с, M1)** | ~700мс (encoder) | — | — | — | — | сеть |
| **Приватность** | 100% локально | 100% локально | 100% локально | 100% локально | 100% локально | данные уходят наружу |
| **Установка** | `cargo install` | cmake + make | `pip install` | `pip install` | cmake или pip | API-ключ + биллинг |
| **Реализация** | Rust | C/C++ | Python/C++ | C++/Java | C++ | N/A |
| **Биндинги** | Rust, C FFI | C, Python, Go, JS… | Python | Python, Java, JS, Go… | C, Python, Java, Swift… | SDK |
| **INT8 квантизация** | авто, 0% потери WER | GGML quant | CTranslate2 quant | — | — | N/A |
| **Параллельные сессии** | настраиваемый пул | 1 | 1 | 1 | 1 | лимиты провайдера |
| **Стоимость** | бесплатно | бесплатно | бесплатно | бесплатно | бесплатно | от $0.006/мин |

> **Компромисс:** gigastt поддерживает только русский. Для мультиязычного распознавания подойдут whisper.cpp или sherpa-onnx — либо NVIDIA Parakeet-TDT / Canary для большей мультиязычной точности. На чистой начитке **Vosk точнее** (4.82% против 8.60% у gigastt, renorm на golos_crowd_1k). Ниша gigastt — **встраиваемый сервер в одном Rust/FFI-бинарнике** с **MIT-чистыми весами** (gigastt MIT + веса GigaAM v3 MIT), **малым INT8-footprint** (~225 МБ против 1.3 ГБ у Vosk) и **лидерством по точности на far-field и телефонии** (см. таблицу по доменам ниже). Для специализированного стримингового телефонного ASR оцените T-one (T-Bank, Apache-2.0); для нового стримингового Vosk — Vosk 0.54 Zipformer2. GigaAM v3 обучена на 700K+ часах русской речи.

## Кому подойдёт?

- **Голосовые ассистенты в реальном времени** — WebSocket-стриминг с инкрементальными частичными результатами (измеренный TTFP ~0.7 с на CPU)
- **Транскрипция колл-центров** — диаризация спикеров + REST batch-обработка
- **Офлайн обработка документов** — транскрипция записей совещаний без загрузки в облако
- **Приватные мобильные приложения** — встраивание через C-ABI FFI на Android с on-device инференсом
- **Исследования и ML-пайплайны** — автономная библиотека `gigastt-core` для Rust ML-стеков

## Возможности

- **Потоковая транскрипция** — инкрементальные частичные результаты по WebSocket поверх offline RNN-T со скользящим окном (real-time на CPU)
- **REST API + SSE** — транскрипция файлов с мгновенным или потоковым ответом
- **Аппаратное ускорение** — CoreML + Neural Engine (macOS), CUDA 12+ (Linux), CPU везде
- **INT8 квантизация** — модель в 4 раза меньше, на 43% быстрее
- **Множество форматов** — WAV, M4A/AAC, MP3, OGG/Vorbis, FLAC
- **Диаризация спикеров** — определение кто говорит (опциональная фича)
- **Автопунктуация** — модель GigaAM v3 выдаёт текст с пунктуацией и нормализацией
- **Автозагрузка** — модель скачивается с HuggingFace при первом запуске (~850 МБ)
- **Docker** — образы для CPU и CUDA с многоэтапной сборкой
- **Защищённость** — лимиты соединений, ограничения фреймов, таймауты, санитизация ошибок

## Быстрый старт

### Установка и запуск

```sh
# Homebrew (macOS ARM64 / Linux x86_64)
brew tap ekhodzitsky/gigastt https://github.com/ekhodzitsky/gigastt
brew install gigastt
gigastt serve

# Из crates.io (нужен `protoc`: `brew install protobuf` / `apt install protobuf-compiler`)
cargo install gigastt
gigastt serve

# Из исходников
git clone https://github.com/ekhodzitsky/gigastt
cd gigastt
cargo run --release -- serve
```

Модель (~850 МБ) скачивается автоматически при первом запуске.

### Docker

```sh
# CPU (любая платформа)
docker build -t gigastt .
docker run -p 9876:9876 gigastt

# CUDA (Linux, требуется NVIDIA Container Toolkit)
docker build -f Dockerfile.cuda -t gigastt-cuda .
docker run --gpus all -p 9876:9876 gigastt-cuda

# Модель скачивается при первом запуске (~850 МБ)
```

#### Образ с моделью внутри (baked)

```sh
# Обычный образ (модель скачивается при первом запуске, ~850 МБ)
docker build -t gigastt .

# Образ с моделью (нет задержки при старте, ~1.1 ГБ)
docker build --build-arg GIGASTT_BAKE_MODEL=1 -t gigastt:baked .
```

### Транскрипция файла

```sh
# CLI
gigastt transcribe recording.wav

# REST API
curl -X POST http://127.0.0.1:9876/v1/transcribe \
  -H "Content-Type: application/octet-stream" \
  --data-binary @recording.wav
# {"text":"Привет, как дела?","words":[{"word":"Привет,","start":0.5,"end":0.9,"confidence":0.97},{"word":"как","start":1.0,"end":1.2,"confidence":0.95},{"word":"дела?","start":1.3,"end":1.7,"confidence":0.93}],"duration":3.5}
```

## API

### WebSocket — стриминг в реальном времени

Подключение к `ws://127.0.0.1:9876/v1/ws`, отправка PCM16 аудио-фреймов, получение транскрипции в реальном времени.

```
Клиент                            Сервер
  |                                 |
  |-------- connect --------------> |
  |                                 |
  | <------- ready ----------------- |
  | {type:"ready", version:"1.0"}  |
  |                                 |
  |------- configure (опционально)-> |
  | {type:"configure",              |
  |  sample_rate:16000}             |
  |                                 |
  |-------- binary PCM16 --------> |
  |                                 |
  | <------- partial --------------- |
  | {type:"partial", text:"привет"} |
  |                                 |
  | <------- final ----------------- |
  | {type:"final",                  |
  |  text:"Привет, как дела?"}      |
```

**Поддерживаемые частоты дискретизации:** 8, 16, 24, 44.1, 48 кГц (по умолчанию 48 кГц, внутри ресемплируется в 16 кГц).

### REST API

| Эндпоинт | Метод | Описание |
|---|---|---|
| `/health` | GET | Проверка состояния (`{"status":"ok"}`) |
| `/ready` | GET | Проба готовности (200 когда пул движка инициализирован) |
| `/v1/models` | GET | Информация о модели (тип encoder, размер пула, возможности) |
| `/v1/transcribe` | POST | Транскрипция файла, полный JSON-ответ |
| `/v1/transcribe/stream` | POST | Транскрипция файла с SSE-стримингом |
| `/v1/ws` | GET | WebSocket-апгрейд для стриминга в реальном времени |
| `/metrics` | GET | Prometheus-метрики (включается `--metrics`) |

**Пример SSE-стриминга:**

```sh
curl -X POST http://127.0.0.1:9876/v1/transcribe/stream \
  -H "Content-Type: application/octet-stream" \
  --data-binary @recording.wav
# data: {"type":"partial","text":"привет как"}
# data: {"type":"partial","text":"привет как дела"}
# data: {"type":"final","text":"Привет, как дела?"}
```

Полная спецификация протокола: [`docs/asyncapi.yaml`](docs/asyncapi.yaml)

#### Коды ошибок

| HTTP | Код | Когда |
|---|---|---|
| 400 | `bad_request` | Неверный формат аудио или некорректный запрос |
| 413 | `payload_too_large` | Файл превышает `--body-limit-bytes` (по умолчанию 50 МиБ) |
| 429 | `rate_limit_exceeded` | Исчерпан per-IP token bucket; заголовок `Retry-After` включён |
| 503 | `pool_saturated` | Все сессии инференса заняты; `Retry-After: 30` |
| 503 | `pool_closed` | Сервер завершает работу, пул закрыт для новых запросов |

```json
// Пример: насыщение пула
HTTP/1.1 503 Service Unavailable
Retry-After: 30

{"code":"pool_saturated","message":"All inference sessions are busy"}
```

### Клиентские библиотеки

Готовые WebSocket-клиенты в [`examples/`](examples/):

#### Python
```sh
pip install websockets
python examples/python_client.py recording.wav
```

#### Bun (TypeScript)
```sh
bun examples/bun_client.ts recording.wav
```

#### Go
```sh
# go mod init gigastt-client && go get github.com/gorilla/websocket
go run examples/go_client.go recording.wav
```

#### Kotlin
```sh
# Зависимости — см. заголовок KotlinClient.kt (Gradle/Maven)
kotlinc examples/KotlinClient.kt -include-runtime -d client.jar
java -jar client.jar recording.wav
```

## Производительность

| Метрика | Значение |
|---|---|
| **WER (флагман, renorm)** | 8.60% (golos_crowd_1k, 1000 образцов, 95% ДИ [7.51%, 9.66%]) |
| **WER (raw, полный набор)** | 11.4% (9 994 записи Golos, 50 394 слова, 95% ДИ [10.9%, 11.9%], без ITN-нормализации) |
| **INT8 vs FP32** | 0% деградации WER (11.4% vs 11.5% на 9 994 записях) |
| **Batch-задержка (16с аудио, M1)** | ~700 мс compute (encoder 667 мс + decode 31 мс) |
| **Streaming TTFP (smoke, golos_00)** | ~0.7 с time-to-first-partial, real-time на CPU (RTF 0.49) |
| **Память (RSS)** | ~560 МБ |
| **Размер модели** | 851 МБ (FP32) / 225 МБ (INT8) |
| **Параллельные сессии** | до 4 (настраивается через `--pool-size`) |

### Сравнение Cross-ASR (9 994 сэмпла, Golos crowd, raw WER, M1 CPU)

| Движок | Модель | WER | RTF | Размер |
|---|---|---|---|---|
| **Vosk** | vosk-model-ru-0.42 | **4.27%** | **0.035x** | 1.3 ГБ |
| **gigastt** | GigaAM v3 (INT8) | **11.37%** | **0.157x** | **225 МБ** |
| whisper.cpp | Large v3 | 14.96% | 0.357x | ~3 ГБ |
| faster-whisper | Large v3 (INT8) | 15.73% | 1.187x | ~3 ГБ |

> **Примечание:** Vosk лидирует на этом подмножестве чистой начитки (Golos crowd) с отличной точностью, но требует 1.3 ГБ. Преимущество gigastt здесь — в ~6× меньшем INT8-футпринте (~225 МБ), аппаратном ускорении (CoreML/CUDA/NNAPI), лидерстве по точности на дальнем поле/телефонии (см. таблицу по доменам) и встраиваемом single-binary Rust/FFI-сервере — не в WER на чистой речи.

### Сравнение по доменам (срезы по 1 000 образцов, CPU)

Один чистый датасет не даёт полной картины. Бенчмарк теперь запускается на
четырёх контрастных доменах русской речи:

| Домен | Датасет | Условия | Лицензия |
|---|---|---|---|
| Чистая начитка | `golos_crowd_1k` | студийный/крауд close-mic | Sber Public License |
| Дальний микрофон | `golos_farfield` | команды умному устройству на расстоянии | Sber Public License |
| Телефонные звонки | `openstt_calls` | шумные телефонные записи | CC BY-NC 4.0 |
| YouTube-речь | `openstt_youtube` | спонтанная речь из видео | CC BY-NC 4.0 |

Таблица WER ниже заполняется воспроизводимым сьютом
(`benchmark/run_full_suite.sh`) и публикуется в ветке
[`benchmark-results-local`](https://github.com/ekhodzitsky/gigastt/tree/benchmark-results-local).
Результаты измеряются с учётом падений как 100% WER и 95% bootstrap
доверительными интервалами.

| Движок | Модель | Чистая начитка | Дальний микрофон | Телефония | YouTube | Размер |
|---|---|---|---|---|---|---|
| gigastt | GigaAM v3 (INT8) | 8.60% (7.51–9.66) | 5.90% (5.09–6.83) | 19.28% (17.88–20.67) | 11.35% (10.32–12.31) | 225 МБ |
| whisper.cpp | Whisper Large v3 | 15.26% (13.74–16.71) | 17.91% (16.29–19.57) | 32.73% (30.69–34.91) | 22.61% (20.97–24.20) | ~3 ГБ |
| faster-whisper | Whisper Large v3 (INT8) | 15.53% (13.94–17.10) | 17.34% (15.62–19.07) | 24.93% (23.32–26.57) | 15.45% (14.15–16.62) | ~3 ГБ |
| Vosk | vosk-model-ru-0.42 | 4.82% (4.03–5.60) | 13.93% (12.49–15.47) | 38.57% (36.72–40.64) | 20.65% (19.38–21.98) | 1.3 ГБ |

Полная методология и подготовка датасетов:
[`benchmark/README.md`](benchmark/README.md).

### Аппаратное ускорение

| Платформа | Флаг компиляции | Execution Provider |
|---|---|---|
| macOS ARM64 (M1-M4) | `--features coreml` | CoreML + Neural Engine |
| Linux x86_64 + NVIDIA | `--features cuda` | CUDA 12+ |
| Android / ARM64 | `--features nnapi` | NNAPI (NPU/DSP) |
| Любая платформа | _(по умолчанию)_ | CPU |

```sh
cargo build --release --features coreml   # macOS: CoreML + Neural Engine
cargo build --release --features cuda     # Linux: NVIDIA CUDA 12+
cargo build --release                     # CPU (любая платформа)
```

Фичи компилируются статически. `coreml` и `cuda` взаимоисключающие; `nnapi` можно сочетать с любой из них.

**Как работает CoreML-путь.** Conformer-энкодер имеет динамическую временную ось, а CoreML не умеет надёжно исполнять партиции, скомпилированные с динамическими шейпами — они падают на этапе предсказания (issue #42). Поэтому gigastt компилирует модель в формате `MLProgram` и ограничивает CoreML подграфами со статическими шейпами: тяжёлые свёртки и матричные блоки идут на Neural Engine, операции с динамическими шейпами остаются на CPU EP. Замер на Apple M1 Pro (INT8-энкодер, release-сборка, медиана 5 прогонов): энкодер **в ~3 раза быстрее** на WAV 4 с (~210 мс против ~690 мс) и **в ~5.6 раза быстрее** на 2-минутном файле (~5.5 с против ~31 с) по сравнению с чистым CPU-билдом.

**Автоматический фолбэк на CPU.** При старте движок прогоняет ~1 с тишины через весь пайплайн (warmup-проба). Если CoreML не загрузился или проба упала, движок пишет предупреждение (`falling back to CPU execution provider`) и прозрачно пересоздаёт сессии на CPU EP — сломанный CoreML-стек снижает производительность, а не роняет процесс. Поддержка CoreML остаётся **зависимой от модели**: будущая ревизия модели может перенести на Neural Engine больше (или меньше) операций.

### INT8 квантизация

Квантизированный encoder: в 4 раза меньше, ~43% быстрее, 0% деградации WER (проверено на 9 994 записях Golos / 50 394 слова). Автоматически определяется при запуске.

Начиная с v0.9.0 квантизация всегда компилируется и автоматически вызывается при первом `download` или `serve` — ни feature-флага, ни ручных шагов не нужно. Cargo-фича `quantize` оставлена как no-op для обратной совместимости.

```sh
# Автоматически (рекомендуется)
cargo install gigastt
gigastt serve           # скачивает модель + автоквантизация при первом запуске

# Отключить автоквантизацию (оставить только FP32)
gigastt serve --skip-quantize
# или: GIGASTT_SKIP_QUANTIZE=1 gigastt serve

# Ручная переквантизация
gigastt quantize                     # нативная квантизация на Rust
gigastt quantize --force             # переквантизировать даже при наличии INT8-модели
```

## Структура проекта

gigastt организован как Cargo workspace из 3 крейтов:

| Крейт | Тип | Назначение |
|---|---|---|
| [`gigastt-core`](crates/gigastt-core) | lib (rlib) | Движок инференса, загрузка модели, квантизация, протокольные типы |
| [`gigastt-ffi`](crates/gigastt-ffi) | lib (cdylib) | C-ABI FFI-слой для встраивания в Android / мобильные приложения |
| [`gigastt`](crates/gigastt) | bin | Серверный бинарник (axum HTTP/WS) + CLI |

`gigastt-core` не зависит от серверных библиотек — встраивайте инференс в любой Rust-проект через `gigastt-core = "2.0"`.

## Архитектура

```
                    Аудио-вход
                   (PCM16, разные частоты)
                        |
                        v
               +-----------------+
               | Мел-спектрограмма |  64 bin, FFT=320, hop=160
               +-----------------+
                        |
                        v
            +------------------------+
            |   Conformer Encoder    |  16 слоёв, 768-dim, 240M параметров
            |  (ONNX Runtime)        |  CoreML | CUDA | CPU
            +------------------------+
                        |
                        v
            +------------------------+
            | RNN-T Decoder + Joiner |  Stateful: h/c сохраняются
            |  (ONNX Runtime)        |  между стриминг-чанками
            +------------------------+
                        |
                        v
            +------------------------+
            |   BPE-токенайзер       |  1025 токенов
            |   + автопунктуация     |
            +------------------------+
                        |
                        v
                  Русский текст
```

## Android / FFI

gigastt можно встроить в Android-приложения через C-ABI FFI-слой (без HTTP-сервера, без JNI).

```sh
# Собрать libgigastt_ffi.so для Android (arm64)
cargo ndk -t arm64-v8a -o ./jniLibs build --release -p gigastt-ffi
```

| Функция | Назначение |
|---|---|
| `gigastt_engine_new(model_dir)` | Загрузить движок (pool_size = 4 по умолчанию) |
| `gigastt_engine_new_with_pool_size(model_dir, pool_size)` | Загрузить с кастомным RAM-лимитом |
| `gigastt_transcribe_file(engine, wav_path)` | Синхронная транскрипция файла |
| `gigastt_stream_new(engine)` | Начать стриминговую сессию |
| `gigastt_stream_process_chunk(...)` | Передать PCM16-аудио, получить JSON-сегменты |
| `gigastt_stream_flush(...)` | Завершить стрим |

Фича `nnapi` на `gigastt-ffi` включает `ort/nnapi` для NPU/DSP-ускорения на Android: `cargo ndk ... build -p gigastt-ffi --features nnapi`. Для мобильных устройств рекомендуется `pool_size = 1` (~350 МБ RAM).

Полное руководство по интеграции: [`ANDROID.md`](ANDROID.md)  
Kotlin-мост: [`ffi/android/GigasttBridge.kt`](ffi/android/GigasttBridge.kt)

## Справка по CLI

Ключевые флаги для самых распространённых команд. У каждого флага есть переменная окружения — полный справочник в [`docs/cli.md`](docs/cli.md).

```sh
# Запустить сервер
gigastt serve --port 9876 --bind-all --metrics

# Транскрибировать файл
gigastt transcribe recording.wav

# Переквантизировать encoder (нативный Rust, ~2 мин одноразово)
gigastt quantize --force
```

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--port` | 9876 | Порт |
| `--host` | 127.0.0.1 | Адрес привязки (по умолчанию только loopback) |
| `--bind-all` | — | Разрешить привязку к не-loopback адресам |
| `--pool-size` | 4 | Параллельные сессии инференса |
| `--metrics` | — | Включить Prometheus на `/metrics` |
| `--idle-timeout-secs` | 300 | Таймаут неактивного WebSocket-соединения |
| `--max-session-secs` | 3600 | Максимальная длительность сессии |
| `--rate-limit-per-minute` | 0 | Rate limit по IP (0 = выключен) |
| `--skip-quantize` | — | Пропустить INT8-квантизацию при первом запуске |

## Модель

[**GigaAM v3 e2e_rnnt**](https://huggingface.co/istupakov/gigaam-v3-onnx) от [SberDevices](https://github.com/salute-developers/GigaAM):

| Свойство | Значение |
|---|---|
| Архитектура | RNN-T (Conformer encoder + LSTM decoder + joiner) |
| Encoder | 16-слойный Conformer, 768-dim, 240M параметров |
| Данные обучения | 700K+ часов русской речи |
| Словарь | 1025 BPE-токенов |
| Вход | 16 кГц моно PCM16 |
| Квантизация | INT8 доступна (v0.2+) |
| Лицензия | MIT |
| Размер загрузки | ~850 МБ (encoder 844 МБ, decoder 4.4 МБ, joiner 2.6 МБ) |

## Требования

| | macOS ARM64 | Linux x86_64 |
|---|---|---|
| **ОС** | macOS 14+ (Sonoma) | Любой современный дистрибутив |
| **CPU** | Apple Silicon (M1-M4) | x86_64 |
| **GPU** | _(встроенный, через CoreML)_ | NVIDIA + CUDA 12+ (опционально) |
| **Диск** | ~1.5 ГБ | ~1.5 ГБ |
| **RAM** | ~560 МБ | ~560 МБ |
| **Rust** | 1.88+ | 1.88+ |

## Безопасность

- **Loopback-only по умолчанию.** Сервер откажется слушать любой адрес кроме
  `127.0.0.1` / `::1` / `localhost`, пока оператор явно не передал `--bind-all`
  (или не задал `GIGASTT_ALLOW_BIND_ANY=1`). Защита от случайного публичного
  экспонирования за reverse-прокси или забытым port-forward.
- **Cross-origin запросы отклоняются по умолчанию.** Страница на
  `https://evil.example.com` больше не может drive-by-подключиться к локальному
  WebSocket / REST API. Loopback-источники всегда разрешены; остальные — через
  `--allow-origin https://app.example.com` (повторяемый флаг). Legacy-поведение
  `Access-Control-Allow-Origin: *` — opt-in через `--cors-allow-any`.
- **Retry-After при перегрузке.** Насыщение пула возвращает HTTP 503 с
  заголовком `Retry-After: 30`, а WebSocket-payload `error` теперь содержит
  `retry_after_ms: 30000` — клиенты могут делать back-off без угадывания.
- **Лимит WebSocket-фрейма:** 512 КБ.
- **Пул сессий:** максимум 4 параллельных сессии (настраивается через `--pool-size`).
- **Ограничение аудио-буфера:** 5 с (стриминг) / 10 мин (загрузка файла).
- **Внутренние ошибки санитизируются** — пути и данные модели не утекают клиентам.
- **Таймаут неактивного соединения:** 300 с.
- **Rate limiting по IP** (опционально, по умолчанию выключено): `--rate-limit-per-minute N`
  включает token-bucket-лимитер на всех эндпоинтах `/v1/*`; `/health` исключён.
  Возвращает HTTP 429 при исчерпании bucket. Privacy-first по умолчанию: выключено.

Удалённое развёртывание (TLS + reverse proxy): см. [`docs/deployment.md`](docs/deployment.md).

## Диагностика

| Симптом | Причина | Решение |
|---|---|---|
| `protoc` not found during build | Отсутствует Protocol Buffers compiler | `brew install protobuf` (macOS) или `apt install protobuf-compiler` (Debian/Ubuntu) |
| Загрузка модели зависает или падает | Сеть / доступность HuggingFace | Повторить `gigastt download`; проверить права `~/.gigastt/models/` |
| `Cannot quantize: FP32 encoder not found` | Частичная загрузка | Удалить `~/.gigastt/models/` и повторить `gigastt download` |
| OOM при старте | Pool size слишком большой для доступной RAM | Уменьшить `--pool-size` (по умолчанию 4); каждая сессия загружает полный encoder |
| CoreML не используется на macOS | Собрано без `--features coreml` | Пересобрать: `cargo build --release --features coreml` |
| `falling back to CPU execution provider` в логах | CoreML не смог скомпилировать или исполнить модель на этой связке macOS/модель | Транскрипция продолжает работать на CPU; очистите `~/.gigastt/models/coreml_cache/` и повторите, либо откройте issue с текстом предупреждения |
| CUDA недоступен на Linux | Собрано без `--features cuda` или отсутствует CUDA 12+ | Пересобрать: `cargo build --release --features cuda`; проверить `nvidia-smi` |
| WebSocket закрывается с 1008 | Сессия превысила `--max-session-secs` | Увеличить `--max-session-secs` или отправлять более короткие потоки |
| 429 Too Many Requests | Rate limiter включён и bucket исчерпан | Дождаться `Retry-After` или отключить `--rate-limit-per-minute 0` |
| Пустая транскрипция для шумного аудио | Слишком тихий вход или неверный формат | Убедиться в 16-bit PCM; нормализовать уровень; проверить поддерживаемые форматы |

## Тестирование

270+ юнит-тестов (включая property-based через proptest) + 40+ e2e/load/soak-тестов + WER-бенчмарк:

```sh
cargo test --workspace               # 270+ юнит-тестов (модель не нужна)
cargo clippy --workspace --all-targets  # Линтер (ноль предупреждений)

# E2E-тесты (требуется модель, последовательно во избежание OOM)
cargo run -p gigastt -- download
cargo test -p gigastt --test e2e_rest --test e2e_ws --test e2e_errors --test e2e_shutdown --test e2e_rate_limit -- --ignored --test-threads=1

# Нагрузочные и стресс-тесты (только локально)
cargo test -p gigastt --test load_test -- --ignored
cargo test -p gigastt --test soak_test -- --ignored
```

## Участие в разработке

См. [CONTRIBUTING.md](CONTRIBUTING.md) — настройка окружения, правила PR и чеклист релиза.

## Лицензия

MIT — см. [LICENSE](LICENSE)

> **Данные бенчмарка:** исходный код — MIT, но датасеты под `benchmark/` — нет. Транскрипты OpenSTT (`openstt_*`) под CC BY-NC 4.0, транскрипты Golos (`golos_*`) под Sber Public License — оба non-commercial. См. [`NOTICE`](NOTICE) и [`benchmark/DATA_LICENSE`](benchmark/DATA_LICENSE).

## Благодарности

- [**GigaAM**](https://github.com/salute-developers/GigaAM) от [SberDevices](https://github.com/salute-developers) — модель распознавания речи
- [**onnx-asr**](https://github.com/istupakov/onnx-asr) от [@istupakov](https://github.com/istupakov) — экспорт ONNX-модели и референсная реализация
- [**ONNX Runtime**](https://github.com/microsoft/onnxruntime) — движок инференса
- [**ort**](https://github.com/pykeio/ort) — Rust-биндинги для ONNX Runtime
