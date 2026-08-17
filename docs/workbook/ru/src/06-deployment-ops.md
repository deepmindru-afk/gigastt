# Развёртывание и эксплуатация

## Сценарий

Вы — администратор, который запускает gigastt на сервере, а не на ноутбуке.
Маршрут этой главы: **установить → ограничить → наблюдать → обновлять** —
один управляемый сервис (systemd или Docker), метрики в Prometheus/Grafana,
алерты на важные режимы отказа и процедура обновления, не обрывающая живые
сессии транскрибации.

Каждый рецепт заканчивается шагом **«Проверить»**. Флаги сверены с
`gigastt serve --help`; полный справочник флагов живёт в
[docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md)
и здесь не повторяется.

## Предпосылки

- gigastt установлен (бинарник, пакет или образ) — см.
  [Начало работы](01-getting-started.md).
- Linux-хост с **4+ ГБ RAM** — обычный production-пол (ОС + пики).
  RAM процесса: ~46 / ~66 МБ resident при пуле 1 / 2. Цифры:
  [docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md).
- Для пути с systemd: systemd 241 или новее (любой современный дистрибутив,
  включая Astra Linux, RED OS, ALT) и root-доступ.
- Для пути с Docker: Docker 20.10+; NVIDIA Container Toolkit — только для
  CUDA-варианта.
- Модель либо скачивается один раз (~225 МБ lean INT8), либо предустановлена
  из офлайн-бандла / deb.

## Рецепт

### Docker

Каждый тегированный релиз публикует мультиархитектурные образы в GHCR —
предпочтительнее тянуть готовое, а не собирать:

```sh
TAG=$(gh api repos/ekhodzitsky/gigastt/releases/latest -q .tag_name)  # e.g. v2.18.0
VER=${TAG#v}
docker pull ghcr.io/ekhodzitsky/gigastt:${VER}        # CPU, linux/amd64 + linux/arm64
docker pull ghcr.io/ekhodzitsky/gigastt:${VER}-cuda   # CUDA, linux/amd64
```

Закрепляйте `$VER` для воспроизводимых развёртываний; `:latest` /
`:cuda` — плавающие.

Запускайте с именованным томом, чтобы lean INT8-модель (~225 МБ) переживала
замену контейнера:

```sh
docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:${VER}
```

Примечания:

- Команда образа по умолчанию — `serve --port 9876 --host 0.0.0.0
  --bind-all` (это нужно контейнерной сети); `-p 127.0.0.1:9876:9876`
  оставляет доступ с хоста только на loopback. TLS-прокси ставится впереди
  точно так же, как при установке без контейнера.
- Контейнер работает под непривилегированным пользователем `gigastt`;
  каталог моделей внутри — `/home/gigastt/.gigastt/models`, именно он
  монтируется томом.
- В образ встроен `HEALTHCHECK` на `/health`, так что `docker ps` покажет
  `healthy`, как только порт начнёт отвечать. Во время первичного скачивания
  lean INT8 (~225 МБ, если том пуст) `/health` отвечает `200` с
  `model:"loading"`, а `/ready` —
  `503 {"status":"not_ready","reason":"initializing"}` — пропускайте трафик
  по `/ready`, а не по `/health`.
- **Baked-образ** (нулевой холодный старт, +~225 МБ INT8): соберите локально с
  моделью внутри — `docker build --build-arg GIGASTT_BAKE_MODEL=1 -t
  gigastt:baked .`
- **CUDA**: `docker run --gpus all -p 127.0.0.1:9876:9876
  ghcr.io/ekhodzitsky/gigastt:${VER}-cuda` (требуется NVIDIA Container
  Toolkit; при отсутствии GPU бинарник откатывается на CPU).

**Проверить:**

```sh
curl -s http://127.0.0.1:9876/ready
# {"status":"ready","pool_available":2,"pool_total":2}
curl -s http://127.0.0.1:9876/health
# {"status":"ok","model":"gigaam-v3-rnnt","variant":"rnnt","version":"...","punctuation":true,"itn":true}
```

### Установка без сети (замкнутый контур)

Для головы `rnnt` достаточно **lean INT8-набора** (~220 МБ):
`v3_rnnt_encoder_int8.onnx`, `v3_rnnt_decoder.onnx`, `v3_rnnt_joint.onnx`,
`v3_vocab.txt`. Рекомендуется `gigastt download`. Подробности:
[deployment.md — Lean INT8-only install](../../../deployment.md#lean-int8-only-install).


Для хостов без доступа в интернет каждый релиз публикует самодостаточный
tarball под каждую Linux-цель — бинарник + предквантованная INT8-модель
`rnnt` + модель пунктуации + systemd-юнит + установщик — и два Debian-пакета
(`gigastt_<ver>_<arch>.deb` + `gigastt-model-int8_<ver>_all.deb`). Полный
состав бандла приведён в
[README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md)
и здесь не повторяется.

На машине с сетью скачайте файлы и **проверьте их до** переноса в контур
(зачем и от каких угроз:
[docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md)):

```sh
TAG=$(gh api repos/ekhodzitsky/gigastt/releases/latest -q .tag_name)  # e.g. v2.18.0
VER=${TAG#v}
gh release download "$TAG" -R ekhodzitsky/gigastt \
    -p "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz" \
    -p "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz.sha256" \
    -p "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz.minisig"
sha256sum -c "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz.sha256"
minisign -Vm "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz" -p gigastt.pub
gh attestation verify "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz" \
    --repo ekhodzitsky/gigastt
```

На целевом хосте:

```sh
tar xf "gigastt-${VER}-offline-x86_64-unknown-linux-gnu.tar.gz"
cd "gigastt-${VER}-offline"
sudo ./install.sh    # verifies SHA256SUMS.txt, then installs binary + models + unit
sudo systemctl enable --now gigastt
```

Альтернатива для Debian-семейства:

```sh
sudo dpkg -i "gigastt_${VER}_amd64.deb" "gigastt-model-int8_${VER}_all.deb"
sudo systemctl enable --now gigastt
```

Модель уже в INT8 — ни скачивания, ни квантования, ни сети. Установленный
юнит выставляет `GIGASTT_OFFLINE=1` через `/etc/gigastt/gigastt.env`, поэтому
любой путь кода, который попытался бы скачать модель (включение `--vad`,
диаризация, альтернативная голова распознавания), **падает быстро с ошибкой,
называющей нужный файл**, вместо зависания на connect timeout. Чтобы добавить
опциональные модели позже, выполните `gigastt download` на машине с сетью и
скопируйте файлы в `/usr/share/gigastt/models/`.

Типовые ошибки:

- `install.sh` прерывается с `sha256sum: WARNING: 1 computed checksum did NOT
  match` — tarball повреждён при переносе в замкнутый контур. Проверьте
  внешний `.sha256`, скопируйте заново и повторите; частичной установки не
  происходит.
- Ошибка офлайн-режима, называющая отсутствующий файл (например, модель VAD
  после включения `--vad`) — этой модели нет в бандле; скачайте её на машине
  с сетью и скопируйте.

**Проверить:**

```sh
systemctl is-active gigastt
# active
curl -s http://127.0.0.1:9876/health
# {"status":"ok",...} — served immediately, the model is pre-installed
```

### systemd-сервис

Усиленный юнит лежит в
[packaging/systemd/](https://github.com/ekhodzitsky/gigastt/tree/main/packaging/systemd)
и устанавливается как deb-пакетом, так и офлайн-бандлом. Ключевые свойства
(сам юнит короткий и с комментариями — полный список читайте в нём):

- Запуск под непривилегированным пользователем `gigastt`; модели в
  `/usr/share/gigastt/models` доступны ему только на чтение.
- Прослушивание только loopback (`127.0.0.1:9876`); наружу API выставляется
  через reverse proxy.
- `Restart=on-failure`, `RestartSec=5` — падение перезапускается, чистый
  `systemctl stop` — нет.
- Набор харденинга, совместимый с systemd 241 (`ProtectSystem=strict`,
  `NoNewPrivileges`, `PrivateTmp`, …), поэтому юнит работает без изменений на
  Astra Linux, RED OS и ALT.
- Переопределения живут в `/etc/gigastt/gigastt.env` (переменные `GIGASTT_*`,
  `RUST_LOG`) и подхватываются через `EnvironmentFile`.

Логи идут в журнал:

```sh
journalctl -u gigastt -f          # follow
journalctl -u gigastt -n 100      # recent
```

Уровень логирования меняется правкой `/etc/gigastt/gigastt.env`
(`RUST_LOG=gigastt=debug`), затем `sudo systemctl restart gigastt`.

Флаги меняйте через drop-in — никогда не правьте поставляемый юнит (обновление
пакета его перезапишет). `ExecStart` нужно сначала очистить, потом задать
заново:

```ini
# sudo systemctl edit gigastt
[Service]
ExecStart=
ExecStart=/usr/bin/gigastt serve --model-dir /usr/share/gigastt/models --punct-model-dir /usr/share/gigastt/models/punct --metrics
```

`systemctl restart gigastt` шлёт `SIGTERM`; сервер дренирует живые
WebSocket/SSE-сессии — каждый клиент получает кадр `Final` +
`Close(1001 Going Away)` — в течение `--shutdown-drain-secs` (по умолчанию
10 с), что с запасом укладывается в стандартный стоп-таймаут systemd 90 с.
Как использовать это при обновлении версий:
[Обновление и откат](#обновление-и-откат) ниже.

**Проверить:**

```sh
systemctl status gigastt --no-pager
curl -s http://127.0.0.1:9876/health
```

### Наблюдаемость

Метрики включаются опционально и отдаются на **отдельном слушателе** — никогда
на порту API, поэтому они находятся вне CORS-allowlist и per-IP
rate-лимитера:

```sh
gigastt serve --metrics                                  # http://127.0.0.1:9090/metrics
gigastt serve --metrics --metrics-listen 127.0.0.1:9100  # custom port
```

Держите слушатель на loopback, если только ваш Prometheus не на другом хосте —
и даже тогда привязывайте его к доверенному интерфейсу, никогда к публичному.

Минимальная проводка Prometheus (`prometheus.yml`):

```yaml
scrape_configs:
  - job_name: gigastt
    static_configs:
      - targets: ["127.0.0.1:9090"]

rule_files:
  - /etc/prometheus/rules/gigastt-alerts.yml   # copy of docs/observability/alerts.yml
```

Следите за **насыщением пула** (`gigastt_pool_available == 0`), **5xx** и
RAM на уровне узла. Правила и дашборд Grafana импортируйте из
[`docs/observability/`](https://github.com/ekhodzitsky/gigastt/tree/main/docs/observability)
— каталог метрик сюда не копируйте. Логи: `RUST_LOG=gigastt=info` (debug
для разбора); текста транскриптов нет
([docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md)).

**Проверить:**

```sh
curl -s http://127.0.0.1:9876/ready > /dev/null   # samples the pool gauges once
curl -s http://127.0.0.1:9090/metrics | grep '^gigastt_pool_available'
# gigastt_pool_available 2
```

### Безопасность по умолчанию

Не ослабляйте значения по умолчанию: loopback, origin-allowlist, лимиты
тела/кадра. Сниппеты прокси и rate-limit:
[docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md).
Проверяйте релиз перед установкой:
[docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md).
Минимальный ритуал:

```sh
TAG=$(gh api repos/ekhodzitsky/gigastt/releases/latest -q .tag_name)  # e.g. v2.18.0
VER=${TAG#v}
minisign -Vm "gigastt-${VER}-x86_64-unknown-linux-gnu.tar.gz" -p gigastt.pub
gh attestation verify "gigastt-${VER}-x86_64-unknown-linux-gnu.tar.gz" \
    --repo ekhodzitsky/gigastt
```

- **Приватность.** Нет телеметрии, нет исходящих соединений после разового
  скачивания модели, транскрипты не логируются
  ([docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md)).

**Проверить:**

```sh
ss -ltnp | grep 9876
# tcp LISTEN 0 ... 127.0.0.1:9876 ...   (loopback only)
curl -s -o /dev/null -w '%{http_code}\n' \
    -H 'Origin: https://attacker.example' http://127.0.0.1:9876/v1/models
# 403
```

### Горячая перезагрузка модели без рестарта

Когда вы заменили файлы модели на диске (новый INT8-энкодер, другая голова,
обновлённая punct-модель), движок можно пересобрать **на месте**, не останавливая
`serve`:

```sh
# Только с loopback — не-loopback клиенты получают 403 loopback_only
# даже при --bind-all.
curl -s -X POST http://127.0.0.1:9876/v1/admin/reload
# {"reloaded":true,"variant":"rnnt","encoder":"int8"}
```

Сервер пересобирает engine по **boot-рецепту** (каталог моделей, размер пула,
punct / ITN / VAD / hotwords), греет новый engine и атомарно подменяет его.
Запросы в полёте дорабатывают на старом. Ошибка сборки оставляет прежнюю
модель (`503 reload_failed`). Параллельные reload → `409 reload_in_progress`.
Полный контракт:
[docs/api.md — Admin reload](https://github.com/ekhodzitsky/gigastt/blob/main/docs/api.md#admin-reload).

**Проверка:**

```sh
curl -s -X POST http://127.0.0.1:9876/v1/admin/reload | tee /tmp/reload.json
python3 -c "import json; d=json.load(open('/tmp/reload.json')); assert d['reloaded'] is True"
# From a non-loopback bind (only if you deliberately opened one): expect 403.
```

### Обновление и откат

Закрепляйте то, что разворачиваете (тег образа, версию deb), чтобы обновление
было осознанным и обратимым шагом. Каталог моделей — это состояние: он
переживает обновления, движок сам определяет установленную голову
распознавания, и **никакого молчаливого перекачивания** при смене бинарника
не происходит. В скриптах установки предпочтите резолв latest-тега:

```sh
TAG=$(gh api repos/ekhodzitsky/gigastt/releases/latest -q .tag_name)   # e.g. v2.18.0
VER=${TAG#v}
```

Docker (обновление до `$VER` из блока выше):

```sh
docker pull ghcr.io/ekhodzitsky/gigastt:${VER}
docker stop --time 15 gigastt && docker rm gigastt
docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:${VER}
```

`docker stop` шлёт `SIGTERM`; `--time 15` даёт окну дренажа
(`--shutdown-drain-secs`, по умолчанию 10 с) завершиться до `SIGKILL` —
стандартные 10 с Docker соревнуются с дренажом. Клиенты получают `Final` +
`Close(1001)` и переподключаются; короткие REST-загрузки в полёте, возможно,
придётся повторить.

systemd / deb:

```sh
sudo dpkg -i "gigastt_${VER}_amd64.deb"
sudo systemctl restart gigastt
journalctl -u gigastt -f    # expect a clean drain, no "Drain window expired"
```

В Kubernetes то же правило действует со стороны оркестратора:
`terminationGracePeriodSeconds` ≥ `shutdown_drain_secs + 5` (полный манифест:
[docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md#graceful-shutdown--session-caps)).

**Откат.** Разверните **предыдущий** тег или пакет — набор моделей на диске не
менялся, поэтому старый бинарник стартует на тех же файлах:

```sh
docker run -d --name gigastt \
  -p 127.0.0.1:9876:9876 \
  -v gigastt-models:/home/gigastt/.gigastt/models \
  ghcr.io/ekhodzitsky/gigastt:${PREV_TAG#v}   # e.g. previous release tag
# or: sudo dpkg -i gigastt_${PREV_TAG#v}_amd64.deb && sudo systemctl restart gigastt
```

Если регрессия дренажа ломает ваших WebSocket-клиентов после обновления,
аварийный выход — `--shutdown-drain-secs 0` (прижимается к 1 с); полная
таблица симптомов —
[docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md).

**Проверить** (после каждого обновления или отката):

```sh
curl -s http://127.0.0.1:9876/health
# "version" is the release you deployed; "model"/"variant" are unchanged
curl -s http://127.0.0.1:9876/ready
```

## Проверка результата

Сквозной смоук после любого рецепта выше:

```sh
systemctl is-active gigastt || docker ps --filter name=gigastt --format '{{.Status}}'
curl -s http://127.0.0.1:9876/health     # status ok, expected version
curl -s http://127.0.0.1:9876/ready      # ready, pool_available >= 1
curl -s http://127.0.0.1:9090/metrics | grep '^gigastt_pool_available'
```

Затем транскрибируйте один короткий файл через тот API, который реально
выставлен (REST-рецепты:
[CLI и пакетная обработка](02-cli-batch.md); проверка через CLI:
[Начало работы](01-getting-started.md)).

## Частые ошибки

- **OOM — контейнер или сервис убит.** Считайте resident (~46 / ~66 МБ при
  пуле 1 / 2). `--pool-min-size 1` поднимает деградированный пул. Цифры и
  OOMKilled:
  [docs/benchmarks.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/benchmarks.md),
  [docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md).
- **503 `timeout` под нагрузкой.** Все триплеты заняты, и вызывающий дождался
  конца `--pool-checkout-timeout-secs` (30 с): REST получает `503` +
  `Retry-After`, WebSocket — ошибку с `retry_after_ms`. Это противодавление,
  а не баг — поднимите `--pool-size`, отделите пакетную работу через
  `--batch-pool-size` и следите за `gigastt_pool_waiters` /
  `gigastt_pool_timeouts_total`.
- **`/metrics` недоступен с хоста Prometheus.** Так задумано: слушатель по
  умолчанию — `127.0.0.1:9090`. Направьте скрейпер на сам хост gigastt или
  осознанно перепривяжите порт через `--metrics-listen` на доверенном
  интерфейсе — никогда на публичном. Скрейпер, по-прежнему смотрящий на
  `:9876/metrics`, получит 404: метрики убраны с порта API.
- **Флапающие readiness-пробы при первом запуске.** Первый запуск может
  скачать lean INT8 (~225 МБ), если каталог модели пуст. Всё это время
  `/health` возвращает `200` с `model:"loading"`, но `/ready` — `503
  initializing`. Если балансировщик смотрит на `/health`, ранние клиенты
  получат 503 — стробируйте `/ready` или предустановите / запеките модель.
- **Rate-лимитер штрафует всех за прокси.** Без `--trust-proxy` — и без
  прокси, перезаписывающего, а не дописывающего `X-Forwarded-For` — все
  клиенты делят один бакет, ключованный адресом прокси. Симптомы и точная
  конфигурация прокси:
  [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md#rate-limiter--x-forwarded-for).

## Ссылки

- [Начало работы](01-getting-started.md) — установка и первая транскрибация
- [CLI и пакетная обработка](02-cli-batch.md) — рецепты REST / SSE / jobs
- [Стриминг по WebSocket](04-streaming-ws.md) — паттерны WebSocket-протокола
- [Приложение A — Коды ошибок](appendix-error-codes.md) — jump table HTTP/WS/close
- [Приложение B — Offline-чеклист](appendix-offline-checklist.md) — air-gapped список
- [docs/deployment.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/deployment.md) — reverse proxy, TLS, манифесты Kubernetes
- [docs/runbook.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/runbook.md) — симптом → причина → аварийный выход
- [docs/cli.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/cli.md) — полный справочник флагов `serve`
- [docs/observability/alerts.yml](https://github.com/ekhodzitsky/gigastt/blob/main/docs/observability/alerts.yml) — правила алертинга Prometheus
- [docs/observability/dashboard.json](https://github.com/ekhodzitsky/gigastt/blob/main/docs/observability/dashboard.json) — дашборд Grafana
- [docs/verifying-releases.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/verifying-releases.md) — minisign, SBOM, SLSA-провенанс
- [docs/privacy.md](https://github.com/ekhodzitsky/gigastt/blob/main/docs/privacy.md) — какие данные куда движутся
- [packaging/systemd/](https://github.com/ekhodzitsky/gigastt/tree/main/packaging/systemd) — юнит + env-файл
- [packaging/offline/README-OFFLINE.md](https://github.com/ekhodzitsky/gigastt/blob/main/packaging/offline/README-OFFLINE.md) — состав офлайн-бандла
