# Реализация стенда M3.2–M3.5

**Статус:** M3.2–M3.5 реализованы; полный suite прошёл (93 теста). Реальный инференс остаётся закрытым до отдельного допуска M4. Эта страница описывает техническую готовность, не экспериментальные результаты.

## Что доступно сейчас

```bash
PYTHONPATH=experiments python3 -m quantbench --help
PYTHONPATH=experiments python3 -m quantbench validate --config experiments/config.example.json
PYTHONPATH=experiments python3 -m quantbench plan --config experiments/config.example.json
PYTHONPATH=experiments .venv/bin/python -m pytest experiments/tests -q
```

Read-only queue/progress/snapshot/compare/merge команды и требования к входным job manifest описаны в [плане очереди](experiment-queue.md). Пример P0–P6 находится в `experiments/queue.blocks.example.json`, он явно имеет `ready_for_execution: false`. Он не является завершённым или исполняемым job-файлом. `run`/`resume` CLI закрыты; общий `run_queue` требует явного `owner_authorized=True`, `preflight`, safety callback и внедрённый executor. Не включать исполнение до владельческого разрешения.

## Общая архитектура и очередь

`PlatformAdapter` отделяет manifest, capabilities, runtime args, ресурсные снимки и preflight. Записи метрик содержат статус, единицу и причину; для обязательных метрик сохраняется fail-closed поведение. Logical observation ID сопоставляет научную конфигурацию между стендами; platform observation ID и manifest hash различают платформенные записи.

`orchestrator.py` реализует детерминированную сортировку, барьер приоритетов, обязательную проверку разрешения владельца, неблокирующий lock одного run directory, ограниченные повторы, resume по append-only JSONL, проверку hash job plan и атомарный checkpoint. Остановка по запросу применяется между наблюдениями. Исполнитель обязан предоставить `preflight`, per-observation `safety_check` и `execute`; backend lifecycle не зашит в planner. Ограниченный retry по умолчанию допускается только для timeout/connection/OS ошибок, максимум 5 попыток суммарно на observation. Структурные и safety ошибки не повторяются автоматически.

`planning.py` оценивает оставшееся время только по фактическим duration в тех же конфигурационных strata. `analysis.py` фильтрует synthetic/fake и pilot записи из подтверждающего среза, читает вложенные `score` и `metrics`, проверяет совместимость production-наблюдений по этапу и контролируемым полям, выявляет пропуски и дубликаты. Snapshot сохраняет hash исходного журнала и создаёт новый файл; парные permutation, Holm и McNemar остаются для M6.

## Данные и scorer

`dataset_pipeline.py` проверяет pinned source hashes и schemas, обязательные поля, split, уникальные IDs и provenance; фиксирует prompt/template, seed, исходные labels, scorer revision и tokenizer counts в JSONL/manifest. Экспорты P0/P1 и полные M2 LIBRA cells (3 задачи × 4K/8K/16K) подготовлены локально. `score_mera` использует pinned task-specific scoring; `LibraScorer` сохраняет multiple-reference процедуру. Реестр M3.3 подтверждает, что локальные MERA `test` exports не содержат доступных labels (`outputs` пусты). Для локальной оценки применяется отдельная выборка из размеченного `train`; это не официальный `test`. Dataset/scorer подключены к executor; доступность источников и готовность exports остаются отдельными статусами.

## Runtime и safety

Runtime сохраняет loopback, `shell=False`, один активный запрос и завершает только принадлежащий ему subprocess. SSE cancellation и общий deadline ограничивают зависшие запросы. Production executor подключает `PlatformAdapter → SafetyWatchdog → LlamaServer → task scorer → append-only observation store`; один runtime и его watchdog используются в пределах совместимого блока и выгружаются при его завершении или ошибке. Warm-up выполняется один раз при старте блока, а для performance — один раз на измерительную ячейку. Safety Guard сохраняет состояние и swap baseline на срок runtime. CLI `fake-run` использует только `SyntheticRuntime`, который не запускает процессы, не обращается к сети и не читает GGUF.

Watchdog агрегирует дискретные пики system-used memory, physical footprint, RSS и swap за всё время жизни runtime, включая загрузку. Сводка содержит фактические интервалы выборок, thermal/pressure states и семантику наблюдаемого максимума; это не гарантия непрерывного пика. Для Apple Safety Guard отсутствие свежих обязательных метрик, Dispatch listener или безопасного состояния блокирует продолжение. Публичный MERA test label недоступен для локального scoring; при совпадении условий сравниваются одинаковые заранее экспортированные локальные IDs.

## Аппаратная диагностика

Локальный нативный helper компилируется и возвращает `proc_pid_rusage/ri_phys_footprint`, thermal state, `vm.swapusage` и `hw.memsize` через `sysctlbyname`. Безопасный внешний снимок 24.11.2026 для текущего Python-процесса: physical footprint 13,681,072 bytes; RSS 14,467,072 bytes; thermal `nominal`; swap used 4,131,651,584 bytes; физическая память 17,179,869,184 bytes; non-free physical pages (total minus `vm_stat` free) 17,092,395,008 bytes. Последний read-only preflight в 21:08 MSK сообщил system-used 17,034,608,640 bytes, swap 4,450,156,544 bytes, 44% free по `memory_pressure -Q`, но category отсутствует. Это исторический снимок общего preflight до Apple-specific safety model, он не является проверкой текущего production executor. Прежний расчёт system-used, складывавший неполный перечень страниц, заменён явно определённым total-minus-free; это значение включает reclaimable страницы и не трактуется как pressure category. SDK source подтверждает, что dispatch pressure API уведомляет об изменениях, но не обещает начальный категориальный снимок ([Apple Dispatch memory pressure source](https://developer.apple.com/documentation/dispatch/dispatchsource/makememorypressuresource%28eventmask%3Aqueue%3A%29)). Произвольная категория не назначается; Safety Guard оценивает fully-reclaimable XNU subset и требует действующий Dispatch listener. Внешние диагностические команды в sandbox ранее вернули `Operation not permitted`; read-only проверка вне sandbox успешно прочитала swap. Модельный preflight не запускался, реальный inference не запускался.

## Платформенные задания и исходники

CUDA контракт и техническое задание для второго разработчика: [docs/platforms/nvidia-cuda.md](platforms/nvidia-cuda.md). Фактическая NVIDIA-машина не инвентаризирована. Кандидатные GGUF источники, revisions/hashes и непроверенная quantization provenance описаны в [реестре моделей](model-artifact-registry.md). Unsloth BF16 не эквивалентен требуемому F16; `ggml-org` имеет точные P1 типы, но quantizer args и общая процедура создания файлов остаются неподтверждёнными. Ни одна полная когорта не зафиксирована, ничего не скачивалось.

## Проверки

Ранее зафиксированный прогон **72 passed** относился к предыдущему состоянию suite. Текущий полный прогон M3.5: **93 passed, 0 failed, 0 skipped**, Python 3.9; тесты включали CLI fake-run с trap-исполняемым файлом, fake-server E2E, runtime reuse, truncation rejection, полную агрегацию macro-F1, фильтрацию snapshot и resource peak summary. `compileall`, CLI `validate`, CLI `plan` и `git diff --check` также прошли. Тесты не запускают модель и не подтверждают совместимость реального CUDA backend или длительную серию.

## Конкурс и научная рукопись

Официальный срок IT OPEN 2026 — 26 ноября 2026 года; на дату текущей редакции (24 ноября) до него два дня. Реальные экспериментальные результаты отсутствуют. Официальный файл требований к оформлению не удалось прочитать; сообщаемые параметры оформления отмечены как ожидающие подтверждения. Конкурсный план и checklist: `competition/it-open-2026/`. Заявка и документы не отправлялись.

## Незакрытые условия M3

1. Завершить M3.5 regression и полный suite; production runtime остаётся закрыт (`REAL_EXECUTION_ENABLED=False`).
2. Владелец утверждает протокол и разрешает подготовку крупных model artifacts перед пилотом; ничего не загружать заранее.
3. Quantization provenance candidate GGUF (quantizer args, intermediate, imatrix/corpus) остаётся неполной и блокирует подтверждающую серию.
4. Общая ревизия llama.cpp ожидает результата проверки CUDA-разработчика; CUDA PR не меняется в этой ветке.
5. Для P1 MERA локальный вариант из размеченного `train` не является official test и требует явного protocol addendum до запуска.
6. Реальный Safety Guard допуск должен пройти аппаратный preflight и пилот только после разрешения владельца.

**Вердикт общей основы:** очередь, общие интерфейсы, dataset/scorer pipeline, executor и safety hooks реализованы; реальный запуск остаётся закрыт до разрешения владельца и закрытия M4 gate.
