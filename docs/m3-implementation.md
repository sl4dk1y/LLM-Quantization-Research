# Реализация стенда M3.2–M3.4

**Статус:** M3.2 не завершён. По прямому изменению задачи начаты M3.3 и M3.4; реализованы повторно используемые основы планирования и анализа, но производственный runtime executor и полная система safety monitoring пока не подключены. Это не готовый к инференсу стенд.

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

`planning.py` оценивает оставшееся время только по фактическим duration в той же модели/квантовании/длине контекста/платформе. Загрузка, прогрев, паузы и recovery пока не включены, а неизвестные strata не получают выдуманную оценку. `analysis.py` проверяет совместимость независимых platform JSONL по protocol/artifact/dataset/scorer/prompt контексту и ключам, выявляет пропуски и дубликаты, создаёт новый merge package, не меняя исходные файлы. Snapshot включает source hash, manifest/runtime/artifact metadata, ошибки и описательные CI по task-group bootstrap, если переданы item scores.

## Данные и scorer

`datasets.py` уже проверяет локальный JSONL SHA-256, revision, split и уникальные IDs. Новый `task_data.py` добавляет task-wise SHA-ranked выборку через уже имеющийся sampler; schema полей задаётся закреплённым dataset manifest и не угадывается. Новые `scoring.py` метрики — NFKC/casefold exact match, token F1 и Accuracy для уже нормализованных prediction/reference. Это базовые метрики, а не утверждение, что они заменяют официальные scorer MERA или LIBRA. Парсеры ответов и официальные task-specific scorer не подключены; их интерфейсы должны быть подтверждены по закреплённым разрешённым exports до запуска.

## Runtime и safety

Runtime сохраняет loopback, `shell=False`, один активный запрос, readiness/request timeout, обязательный SSE `[DONE]`, finish reason, клиентские timings и завершение только собственного subprocess. `SafetyWatchdog` умеет периодически опрашивать переданный sampler, fail-closed останавливать runtime через callback при BLOCKED/EMERGENCY_STOP и сохранять lockout после safety event. `LlamaServer.generate` принимает cancellation event.

Пока нет единой production связки `PlatformAdapter → SafetyWatchdog → LlamaServer → task scorer → observation store`. Текущая macOS реализация сообщает, что `memory_pressure.category` недоступна; существующий `SafetyGuard` требует категорию `green/normal` и поэтому не разрешит запуск. Этот блокер нельзя обходить пониженными требованиями или фиктивной категорией. Активный streaming-запрос ещё не покрыт тестом аварийной остановки watchdog-end-to-end. Для NVIDIA неизвестны ОС и hardware manifest, CUDA адаптер остаётся самостоятельной задачей второго участника.

## Аппаратная диагностика

Локальная нативная macOS проверка подтвердила компиляцию helper и доступ к одному измерению `physical_footprint_bytes` и `thermal_state=nominal`. `vm_stat` сообщил page size 16,384 bytes; предыдущий снимок `memory_pressure -Q` показывал 45% свободной памяти. Последний `preflight` от 24.11.2026 подтвердил `llama-server 0.1.1-dev`, build 10470, commit `34af94cd9`; текущий снимок показал 40% свободной памяти и оценку `system_used_bytes=13,144,145,920`. `sysctl vm.swapusage` запрещён sandbox; у процесса нет PID, поэтому RSS/physical footprint процесса не получены. Preflight вернул exit code 2 и `allowed_for_inference=false`: обязательные `memory_pressure.category` и `swap_used_bytes` отсутствуют, platform adapter не подключён к CLI, локальные пути модели/dataset из примера не существуют, разрешение владельца не зарегистрировано. Это диагностика API, не baseline и не результат измерения модели. Стоимость sampler не оценивалась. Никакие модели, полные датасеты и инференс не запускались.

## Платформенные задания и исходники

CUDA контракт и диагностическая анкета для второго разработчика: [docs/platforms/nvidia-cuda.md](platforms/nvidia-cuda.md). Фактическая машина не инвентаризирована. Кандидатные GGUF источники, partial revisions/hashes и непроверенная quantization provenance описаны в [реестре моделей](model-artifact-registry.md). Единый предварительный кандидат Unsloth не утверждён: общая команда конвертации, calibration corpus, imatrix provenance и полные hashes всех нужных файлов остаются неподтверждёнными. Ничего не скачивалось.

## Проверки

Полный прогон: **42 passed, 0 failed, 0 skipped**; Python 3.9.19, pytest 8.4.2. Loopback fake server требует запуск тестов вне обычных ограничений sandbox. Suite покрывает config, выборку, append-only storage, safety thresholds, платформенный контракт, SSE/fake server, ограниченные retries, resume, блокировку параллельного runner, fail-closed watchdog hook, queue compiler, offline snapshot/merge validation, task export checks и базовые метрики. Он не проверяет реальную модель, GPU/Metal/CUDA, Windows, полный runtime-watchdog abort, official dataset scorer или многосуточный запуск. Финальные проверки `compileall`, `validate` и `plan` ещё должны быть повторены после последних изменений.

## Конкурс и научная рукопись

Официальный срок IT OPEN 2026 — 26 ноября 2026 года; на дату текущей редакции (24 ноября) до него два дня. Реальные экспериментальные результаты отсутствуют. Официальный файл требований к оформлению не удалось прочитать; сообщаемые параметры оформления отмечены как ожидающие подтверждения. Конкурсный план и checklist: `competition/it-open-2026/`. Заявка и документы не отправлялись.

## Незакрытые условия M3

1. Доступность и семантика обязательной категории memory pressure и корректные платформенные safety metrics.
2. End-to-end watchdog, который останавливает зависший активный HTTP streaming запрос и гарантированно удерживает lockout.
3. Интеграция платформенных профилей с config, preflight, runtime и выводом метрик.
4. Production executor с последовательной загрузкой одной модели, сбором manifest/метрик, task prompt/scorer и записью полных observation records.
5. Официальные task adapters/scorer, revision и checksum разрешённых dataset exports.
6. Полный immutable model artifact registry, единая provenance/quantization процедура и независимые local checksums.
7. Инвентаризация второй машины и CUDA adapter/контрактные тесты разработчика.
8. Job compiler, который строит exact matrix jobs из утверждённого протокола и локальных manifest без декартова расширения.
9. Тесты повреждённого checkpoint, safety event during active generation, power interruption и resume на обеих ОС.
10. Owner-reviewed protocol lock и отдельное разрешение перед любым скачиванием крупных артефактов или реальным инференсом.

**Вердикт общей основы:** планировщик, общие интерфейсы и техническое задание дают основу для передачи после проверки владельцем; производственная готовность к самостоятельным запускам пока отсутствует.

**Вердикт M3.2:** не готов к независимому аудиту как завершённый этап. M3.3/M3.4 частично реализованы; M4 эксперименты не начаты. Условия допуска, согласование protocol addendum и перехода к экспериментам перечислены в `docs/experiment-queue.md` и `docs/roadmap.md`.
