# Архитектура будущего стенда оценивания

**Статус:** проект спецификации; код не создавался, зависимости не устанавливались, benchmark не запускался.

## Минимальная схема

Будущий стенд должен быть одним локальным Python-проектом с последовательным runner, локальным `llama-server`, append-only JSONL файлами и отдельным анализом. Микросервисы, web UI, удалённые очереди и база данных для этой задачи не требуются.

| Компонент | Ответственность |
| --- | --- |
| Manifest / protocol config | Хранит protocol version, model/dataset revisions, quant, generation, budget thresholds, contexts, seeds, file hashes. Настройка read-only для исполнителя; изменение создаёт новую версию. |
| Dataset loader | Читает уже закреплённую локальную копию/экспорт, валидирует schema и split, сохраняет source revision/hash и список выбранных item IDs. Не переписывает source data. |
| Prompt renderer | Применяет официальный task template и model chat template, сохраняет rendered text/hash и tokenizer token count. Отдельные правила для Qwen3 и Qwen3.5. |
| Runtime adapter | Один локальный HTTP-клиент к `llama-server`; принимает один запрос, получает streaming chunks и timing, различает load/request timeout и runtime error. |
| Safety/resource collector | Сэмплирует process footprint, system memory, pressure, swap, thermal state и backend log; может запретить следующий запрос по safety threshold. |
| Orchestrator / resume | Последовательный порядок конфигураций и запросов, checkpoint после 20 items/на блоке, stop/recovery, resume только отсутствующих compound keys. |
| Quality evaluator | Task-native scorer, валидатор label, official MultiQ/MERA/LIBRA scoring, журнал invalid/truncated/failed outputs. Никакого LLM judge. |
| Raw store | Append-only JSONL: один record на запрос, отдельные resource/performance samples, manifest и diagnostic events. Сырьё не перезаписывается. |
| Statistical analysis | Воспроизводимые скрипты/ноутбук только после M3; строят таблицы/графики из неизменённых raw files и создают derived-data manifest с исходными hash. |

## Поток данных

Закреплённый dataset export → проверка schema/revision → детерминированные item IDs → официальный task prompt → model-specific chat template → local server → streamed response/timing + ресурсные срезы → raw JSONL → scorer → derived summary/CI/Pareto tables.

Отдельные channels записи: `manifest.json`, `requests.jsonl`, `resource_samples.jsonl`, `events.jsonl`, `analysis_manifest.json`. Отсутствующие метрики хранятся как отсутствующие с кодом причины, не `0`. Большие GGUF/модельные веса/исходные датасеты не кладутся в Git; в manifest сохраняются локальный URI/path, размер, hash, лицензия и revision.

## Идемпотентность и восстановление

Compound key `(protocol version, model revision, quant, dataset revision, split, task, item_id, generation seed)` уникален. Runner сначала проверяет завершённые записи, затем добавляет только отсутствующие. Повтор после аварии создаёт новую event запись с новым attempt и ссылкой на исходный ключ; оригинальный результат не удаляется. Derived artifacts должны ссылаться на точные входные файлы и scorer hash.

## Технологический стек-кандидат

- `llama.cpp` локальный сервер в точном commit, собранный для macOS arm64 с Metal; backend build log сохраняется.
- Python standard tooling для runner, JSONL/manifest, анализа; конкретные версии и прямые зависимости утверждаются в M3, без установки на этапе M2.
- Официальные dataset/scorer исходники локально после утверждения и checksum; отдельная ручная сверка scorer на фиксированных examples.
- macOS system tools/API и `proc_pid_rusage`-совместимый сборщик метрик; ресурсный sampling должен проверяться отдельным M3 validation test до основного бенчмарка.
- Git для версионирования исходников по обычному workflow владельца; большие веса, corpora, generated runs исключить из репозитория.

## Контроли, необходимые до реализации

До M3 проверить API metrics, semantics scorer и chat template на маленьком ручном наборе, file locking/append safety, resume key behavior, timeout/TERM path, process cleanup, output decoding и корректное прохождение текстового Qwen3.5 prompt. Эти проверки — будущие функциональные проверки стенда; это не запуск оценочного benchmark.
