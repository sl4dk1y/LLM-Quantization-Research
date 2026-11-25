# Экспериментальный стенд

На этапе M3 здесь размещается минимальный локальный программный стенд. Реализация инфраструктуры не означает, что датасеты, модели или экспериментальный протокол утверждены. Запуск модели и оценочных серий требует отдельного решения на следующем этапе.

## Подготовка среды

Минимальная версия Python пока формально не закреплена. Проверенная среда подготовки данных — Python 3.9 с зависимостями из `requirements-dev.txt` и указанными в [руководстве подготовки датасетов](../docs/dataset-pipeline.md) пакетами. CLI может скачать только перечисленные там небольшие файлы датасетов, конфигурации/tokenizer и исходники scorer с проверкой ревизий и SHA-256; веса моделей и полные архивы датасетов не загружаются. Runtime модели — отдельный локальный `llama-server`; фактическая совместимость с закреплёнными артефактами не проверена.

Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Запуск справки и безопасного dry-run:

```bash
python -m quantbench --help
python -m quantbench validate --config experiments/config.example.json
python -m quantbench plan --config experiments/config.example.json
```

Dry-run валидирует конфигурацию и показывает план; он не запускает runtime, не загружает веса и не выполняет инференс.

## Очередь и аналитика

Приоритеты P0–P6 и команды подготовки job manifest, прогресса, snapshot и offline merge описаны в [плане очереди](../docs/experiment-queue.md). Файл `queue.blocks.example.json` — незаполненный шаблон, он не допускает запуск. MERA/LIBRA exports и task-specific scorer подключены к исполнителю; полный fake-server E2E проверяет цикл без научных измерений. Команды `run` и `resume` требуют явное разрешение владельца и остаются закрытыми до этапа M4.

## Состояние реализации

Схемы MERA/LIBRA, pinned exports и scorer описаны в [руководстве подготовки датасетов](../docs/dataset-pipeline.md). На macOS начальная `memory_pressure.category` может быть неизвестна; Safety Guard допускает `null` только для Apple-specific safety model при действующем Dispatch listener, свежих обязательных метриках и безопасных значениях reclaimable memory/swap/thermal. Watchdog связан с жизненным циклом `llama-server`; production executor остаётся закрытым до проверки допуска в M4. См. [архитектуру](../docs/benchmark-architecture.md), [экспериментальный протокол](../docs/experimental-protocol.md) и [протокол измерения ресурсов](../docs/measurement-protocol.md).
# Исполнение и проверка очереди

CLI использует один platform-specific job manifest. Сначала проверьте план и локальные артефакты:

```sh
PYTHONPATH=experiments python -m quantbench dry-run \
  --config experiments/config.example.json \
  --jobs results/queue/apple-metal.jobs.json

PYTHONPATH=experiments python -m quantbench preflight \
  --config experiments/config.example.json
```

Fake-server интеграция использует совместимый локальный тестовый сервер и сохраняет только синтетические записи. Указывайте каталог вне `results/`:

```sh
PYTHONPATH=experiments python -m quantbench fake-run \
  --config /local/path/experiment-config.json \
  --jobs /local/path/platform.jobs.json \
  --run-dir /local/path/fake-runs/m34
```

Production-команды требуют файл разрешения владельца для точного хеша очереди и платформы. Текущая сборка M3.4 всё равно блокирует реальный инференс; снятие блокировки относится к M4 после интеграционного аудита.

```sh
PYTHONPATH=experiments python -m quantbench run \
  --config /local/path/experiment-config.json \
  --jobs /local/path/platform.jobs.json \
  --run-dir /local/path/results/raw/campaign \
  --authorization-file /local/path/owner-authorization.json

PYTHONPATH=experiments python -m quantbench resume \
  --config /local/path/experiment-config.json \
  --jobs /local/path/platform.jobs.json \
  --run-dir /local/path/results/raw/campaign \
  --authorization-file /local/path/owner-authorization.json

PYTHONPATH=experiments python -m quantbench stop \
  --run-dir /local/path/results/raw/campaign

PYTHONPATH=experiments python -m quantbench status \
  --config /local/path/experiment-config.json \
  --run-dir /local/path/results/raw/campaign

PYTHONPATH=experiments python -m quantbench integrity \
  --config /local/path/experiment-config.json \
  --run-dir /local/path/results/raw/campaign

PYTHONPATH=experiments python -m quantbench slice \
  --config /local/path/experiment-config.json \
  --run-dir /local/path/results/raw/campaign \
  --output /local/path/intermediate-snapshot.json
```

`--start-block P1` выбирает начальный блок, а `--max-runtime` ограничивает время между наблюдениями. Остановка через `stop` и лимит времени не прерывают запрос посреди observation. `resume` проверяет неизменность плана и пропускает завершённые observation IDs.
