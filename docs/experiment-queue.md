# Очередь экспериментальной программы

**Статус:** план исполнения; запуск не разрешён. Источник состава матрицы — предварительный протокол M2 и `docs/experiment-matrix.md`. Очередь приоритизирует работу, но не исключает строки и не меняет выборки. Допуск каждой строки требует закреплённых артефактов, обоих аппаратных manifest и успешного платформенного preflight.

## Приоритетные блоки

| Блок | Содержание | Условие завершения и граница вывода |
| --- | --- | --- |
| P0 | Изолированный пилот на одной короткой MERA задаче для проверки загрузки, scorer, sampler, мониторинга, остановки и повторного старта. Предварительный кандидат — Qwen3-4B Q4_K_M; конкретные элементы и число наблюдений закрепить отдельной версией до запуска. | Только техническая/пилотная серия. Никогда не объединять с подтверждающей оценкой качества. |
| P1 | Основной законченный парный срез: Qwen3-4B F16, Q8_0, Q4_K_M; MERA MathLogicQA `test` и MultiQ `test`, по 200 одних и тех же item IDs на конфигурацию; контекст 4096; Apple Metal и NVIDIA CUDA. | Завершить все разрешённые ячейки обеих платформ до P2. Межплатформенный вывод относится к двум конкретным стендам. При невозможности F16 на одной машине остановить парную P1 серию. Qwen3-1.7B может быть отдельным fallback только после протокольного addendum и не заменяет 4B в полной программе. |
| P2 | Добавить Q6_K и Q5_K_M для P1 выборок Qwen3-4B. | Переиспользовать точные логические наблюдения только при полном совпадении ключевых параметров. |
| P3 | Исследование размера модели в рамках M2: Qwen3-1.7B со всеми пятью квантованиями; Qwen3-8B с Q4_K_M/Q6_K; Q8_0/Q5_K_M условны по safety/preflight; F16 для 8B остаётся исключённым по ресурсному обоснованию M2. | Не расширять матрицу полным декартовым произведением. Условные строки регистрировать до просмотра качества. |
| P4 | Остальные MERA ячейки M2: MathLogicQA `train` (100), ruOpenBookQA `train` (100), ruOpenBookQA `test` (200), PARus `dev` (100); оценки и выборки остаются разделёнными по ролям. | Не менять `development` и `held-out` после открытия результатов. Использовать только опубликованные локальные/разрешённые ответы по закреплённому scorer. |
| P5 | LIBRA Mini: `ru_babilong_qa3`, `long_context_multiq`, `matreshka_names`, классы 4K/8K/16K; три строки Qwen3-4B F16/Q6_K/Q4_K_M, по 30 общих items на ячейку. Отдельная performance серия M2: контексты 512/4096/8192/16384, 30 повторов + 3 прогрева, для Qwen3-4B F16/Q4_K_M/Q6_K. | Не смешивать качество на естественных prompts и стандартизованную производительность. Фактические токены определяются закреплённым tokenizer и опубликованным fixture. |
| P6 | Валидационная когорта Qwen3.5-4B по M2 (F16/Q6_K/Q4_K_M; Q8_0 условен), текстовый режим и проверка переносимости выбора конфигурации. | Отдельная валидационная серия; не объединять без отдельной модели анализа. |

### Предварительная фиксация состава P1

Для приоритетного законченного блока P1 до открытия любых модельных результатов выбраны MathLogicQA `test` и MultiQ `test`. Это заранее ограниченный срез полного M2 набора held-out оценок: он включает повторную оценку задачи, участвовавшей в выборе (MathLogicQA), и одну задачу переноса, не участвовавшую в выборе (MultiQ). Выбор сохраняет научные роли, заданные M2: отбор конфигурации остаётся только на MathLogicQA и ruOpenBookQA `train`; полный M2 held-out набор дополнительно сохраняет ruOpenBookQA `test` и PARus `dev` для P4. P1 не заменяет эти строки и не разрешает менять состав после просмотра результатов.

Для P1 первичны Accuracy в MathLogicQA и token-F1 в MultiQ; вторичные метрики остаются согласно M2. До подтверждающего запуска должны быть закреплены разрешённые split exports, item IDs, task prompts и проверенные версии официальных scorer. Пока это не выполнено, текущая запись является предварительной фиксацией плана, а не допуском к запуску.

Приоритетный файл `experiments/queue.blocks.example.json` фиксирует P0–P6; `experiments/queue.plan.template.json` показывает поля одной P1 ячейки, оставляя все непроверенные данные пустыми. Оба файла имеют `ready_for_execution: false`. Перед компиляцией нужны protocol hash, source checkpoint revision, GGUF repo/revision/file SHA-256, dataset revision/hash, item IDs, scorer version, tokenizer revision, chat-template hash, reasoning mode, llama.cpp commit, параметры генерации, comparison-series type, platform IDs и hardware manifest hashes. `quantbench compile-queue --plan <approved-plan.json> --platform-id apple-metal --output <metal-jobs.json>` компилирует только явные ячейки для одной машины; на второй машине используется собственный platform ID и отдельный файл. Компилятор отказывается перезаписывать output. Каждая запись job содержит общий логический `observation_id`, уникальный `job_id`, приоритет, блок и контролируемые параметры. Если export содержит несколько задач, dataset manifest должен закреплять колонку задачи; однозадачный export может явно указывать единственную задачу без такой колонки.

## Очередность, повторы и восстановление

`quantbench.orchestrator.order_jobs` сортирует задания стабильно и открывает только минимальный оставшийся priority. Поэтому P2 не начинается, пока все P1 задания не завершены. `run_queue` запускает один job за раз, требует отдельные `preflight`, per-observation `safety_check` и `execute` callbacks, берёт неблокирующую межпроцессную блокировку run directory, пишет append-only `observations.jsonl` и атомарный `checkpoint.json`. Повторные попытки ограничены 1–5 суммарно, включая рестарты; исчерпанные задания остаются незавершёнными и блокируют более поздний приоритет. Плановая остановка действует между наблюдениями. Safety callbacks ещё не соединены с активным model executor.

Это общий оркестрационный контур, не подключённый к реальному `llama-server`, Safety Guard и платформенным sampler. `run`/`resume` CLI остаются закрытыми. Поэтому ночное и длительное выполнение пока запрещено; не запускать `run_queue` с внешним model executor до интеграции мониторинга и владельческого разрешения.

## Команды планирования и проверки

После утверждения полного job manifest локальные read-only команды имеют вид:

```bash
PYTHONPATH=experiments python3 -m quantbench queue --config experiments/config.example.json --jobs results/queue/jobs.json
PYTHONPATH=experiments python3 -m quantbench compile-queue --config experiments/config.example.json --plan results/queue/approved-plan.json --platform-id apple-metal --output results/queue/metal-jobs.json
PYTHONPATH=experiments python3 -m quantbench queue --config experiments/config.example.json --jobs results/queue/metal-jobs.json --platform-id apple-metal --run-dir results/local-run --start-block P1
PYTHONPATH=experiments python3 -m quantbench progress --config experiments/config.example.json --jobs results/queue/metal-jobs.json --run-dir results/local-run
PYTHONPATH=experiments python3 -m quantbench status --config experiments/config.example.json --run-dir results/local-run
PYTHONPATH=experiments python3 -m quantbench integrity --config experiments/config.example.json --run-dir results/local-run
PYTHONPATH=experiments python3 -m quantbench slice --config experiments/config.example.json --run-dir results/local-run --output results/slices/snapshot.json
PYTHONPATH=experiments python3 -m quantbench compare --config experiments/config.example.json --left results/metal/observations.jsonl --left-platform apple-metal --right results/cuda/observations.jsonl --right-platform nvidia-cuda
PYTHONPATH=experiments python3 -m quantbench merge --config experiments/config.example.json --left results/metal/observations.jsonl --left-platform apple-metal --right results/cuda/observations.jsonl --right-platform nvidia-cuda --output results/merged/pairs.json
```

Сейчас эти команды могут показать только dry-run job plan, прочитать уже существующий JSONL и создать отдельный описательный snapshot. `slice` создаёт новый файл и отказывается перезаписывать существующий путь. Публичные результаты не публикуются автоматически. Наличие незакреплённых/плейсхолдерных артефактов блокирует исполнение.

## Время и экспорт

`estimate_remaining` оценивает длительность только по фактически завершённым наблюдениям соответствующей модели/квантования/контекста/платформы. Если замеров такой группы нет, оценка времени отсутствует. Текущая оценка исключает загрузку, прогрев, задержки, повторы и выключения; она не является обещанием времени завершения. Для полноценной оценки после пилота потребуется отдельно измерить загрузку, prefill/decode, длину ответа, паузы и восстановление для существенных групп.

Snapshot содержит hash и число исходных строк, timestamp и собственный ID, количество успешных/ошибочных попыток, manifest/runtime/artifact metadata при их наличии и описательные агрегаты. Парное сопоставление проверяет protocol/artifact/dataset/scorer и общие поля, показывает пропуски и не изменяет исходные файлы. Подтверждающие выводы требуют предварительно зарегистрированной статистической процедуры и достаточной полноты данных; общий экспорт сам по себе не является статистическим отчётом.

## Граница допуска

До разрешения владельца не загружать крупные модели или полные наборы данных, не запускать инференс и не проводить серии. Для перехода к автономному выполнению сначала закрыть в M3.2/M3.3: обязательные safety metrics на обеих платформах, watchdog с прерыванием активного запроса, runtime/model scorer integration, полную проверку завершения/остановки, реальный platform preflight, dataset/scorer contracts, идемпотентную компиляцию всех jobs и тесты отказов. После технической готовности требуется отдельное разрешение владельца перед подготовкой артефактов и первым инференсом.
