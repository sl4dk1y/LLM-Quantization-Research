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

`datasets.py` уже проверяет локальный JSONL SHA-256, revision, split и уникальные IDs. `task_data.py` добавляет task-wise SHA-ranked выборку; schema полей задаётся закреплённым dataset manifest. `scoring.py` содержит NFKC/casefold exact match, token F1 и Accuracy для уже нормализованных prediction/reference. Это базовые функции, а не официальные scorer MERA/LIBRA. Task-specific parsers и official scorer не подключены; датасеты не закреплены локальными exports. P1 MathLogicQA + MultiQ заранее зафиксированы как приоритетный срез M2 без исключения оставшихся held-out задач.

## Runtime и safety

Runtime сохраняет loopback, `shell=False`, один активный запрос и завершает только принадлежащий ему subprocess. Исправлена блокировка аварийной остановки во время readiness: lifecycle lock не удерживается на время ожидания HTTP health; `stop()` ограничен 2 s TERM grace + 1 s после KILL. SSE-чтение перенесено в отдельный reader, cancellation закрывает принадлежащий соединению socket, а общий монотонный deadline ограничивает весь запрос, в том числе поток с регулярно поступающими строками. `SafetyWatchdog` callback может останавливать runtime независимо от основного потока; fake-server тест покрывает watchdog stop при зависшем streaming. Build args теперь требуют явные GPU layers, CPU threads, batch/ubatch, KV cache precision, flash attention, reasoning и источник chat template; generation payload требует явные seed/temp/top-p/top-k/max tokens.

Пока нет единой production связки `PlatformAdapter → SafetyWatchdog → LlamaServer → task scorer → observation store`. `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` — event source для изменений состояния, а не гарантированный синхронный начальный снимок; одноразовый collector не подменяет неизвестную категорию. Поэтому `memory_pressure.category` остаётся недоступной и Safety Guard блокирует запуск. Потоковый watchdog stop проверен на fake HTTP server, но не подключён к production executor. Поддерживаемые CLI-флаги проверены на локальной сборке, однако она не совпадает с предложенной M2 ревизией llama.cpp; immutable общий source revision для Metal/CUDA не зафиксирован. `--jinja` и prompt generation path не прошли проверку на закреплённых Qwen GGUF и официальных task prompts. Для NVIDIA по-прежнему нет локального hardware manifest или готового адаптера.

## Аппаратная диагностика

Локальный нативный helper компилируется и возвращает `proc_pid_rusage/ri_phys_footprint`, thermal state, `vm.swapusage` и `hw.memsize` через `sysctlbyname`. Безопасный внешний снимок 24.11.2026 для текущего Python-процесса: physical footprint 13,681,072 bytes; RSS 14,467,072 bytes; thermal `nominal`; swap used 4,131,651,584 bytes; физическая память 17,179,869,184 bytes; non-free physical pages (total minus `vm_stat` free) 17,092,395,008 bytes. Последний read-only preflight в 21:08 MSK сообщил system-used 17,034,608,640 bytes, swap 4,450,156,544 bytes, 44% free по `memory_pressure -Q`, но category отсутствует. Прежний расчёт system-used, складывавший неполный перечень страниц, заменён явно определённым total-minus-free; это значение включает reclaimable страницы и не трактуется как pressure category. Оно превышает предварительный порог 12 GiB, поэтому также блокирует запуск до проверки safety semantics. SDK source подтверждает, что dispatch pressure API уведомляет об изменениях, но не обещает начальный категориальный снимок ([Apple Dispatch memory pressure source](https://developer.apple.com/documentation/dispatch/dispatchsource/makememorypressuresource%28eventmask%3Aqueue%3A%29)). Произвольная категория не назначалась. Предыдущая попытка `sysctl` внутри sandbox вернула `Operation not permitted`; разрешённая read-only проверка вне sandbox успешно прочитала swap. Preflight завершился exit code 2, `allowed_for_inference=false`: отсутствует `memory_pressure.category`, не подключён платформенный adapter, placeholder пути модели/датасета не существуют, разрешение владельца не зарегистрировано. Ни одна модель или полный датасет не скачивались; inference не запускался.

## Платформенные задания и исходники

CUDA контракт и техническое задание для второго разработчика: [docs/platforms/nvidia-cuda.md](platforms/nvidia-cuda.md). Фактическая NVIDIA-машина не инвентаризирована. Кандидатные GGUF источники, revisions/hashes и непроверенная quantization provenance описаны в [реестре моделей](model-artifact-registry.md). Unsloth BF16 не эквивалентен требуемому F16; `ggml-org` имеет точные P1 типы, но quantizer args и общая процедура создания файлов остаются неподтверждёнными. Ни одна полная когорта не зафиксирована, ничего не скачивалось.

## Проверки

Полный прогон после изменений: **46 passed, 0 failed, 0 skipped**; Python 3.9, pytest 8.4.2. Тесты включают fake HTTP server, stop во время readiness, cancellation зависшего streaming, абсолютный deadline при продолжающемся потоке и остановку runtime callback’ом SafetyWatchdog. Текущий аппаратный снимок — диагностика одного локального процесса, не baseline модели и не экспериментальный результат. Suite не проверяет реальную модель, Metal/CUDA runtime compatibility, официальные датасетные scorer или длительный процесс.

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
