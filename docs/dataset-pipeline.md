# Подготовка выборок и оценка ответов

## Зафиксированные источники

| Источник | Ревизия | Лицензия карточки | Подготовка |
| --- | --- | --- | --- |
| `MERA-evaluation/MERA` | `15c0e5b8e27551b50aa27f5497074810c8740661` | MIT | Используются исходные JSONL task splits. В `test` файлах MathLogicQA, MultiQ, ruOpenBookQA и PARus поле `outputs` присутствует, но пусто; scoring этих файлов локально невозможен. |
| `ai-forever/LIBRA` | `10d668ab44165bbab73ce806aa536294abedc1ba` | MIT | Берутся только parquet трёх задач. Строки преобразуются в JSONL с сохранением полей, исходного порядка, полного контекста и `positive_outputs`. Так как исходный ID отсутствует, используется revision-scoped ID с индексом строки и SHA-256 записи. |

Официальный генеративный evaluator MERA закреплён на `MERA-Evaluation/MERA` commit `8698451dd7c281462c684190c9e38d2fae0f6d7a` и harness submodule `artemorloff/lm-evaluation-harness` commit `d85f735223f11699765e2f5490ee526a3fe1b1e5`. LIBRA templates и EM scorer сверены с `EleutherAI/lm-evaluation-harness` commit `d6de81643928d653435c431bae19945d41d32520`. Файлы evaluator и карточки источников сохранены локально вместе с SHA-256 provenance manifest.

## Выборки

- **P0:** 20 MathLogicQA и 20 ruOpenBookQA из `train`. Элементы не пересекаются с демонстрацией, development и held-out выборкой.
- **P1:** 200 MathLogicQA и 200 MultiQ из размеченного открытого `train`. Это `local_heldout`, а не официальный test. Для MathLogicQA исключены демонстрация, P0 и development ID; MultiQ использует 0-shot.
- **M2 development:** по 100 MathLogicQA и ruOpenBookQA из train, отдельно от P0 и demonstration.
- **M2 held-out:** 200 ruOpenBookQA из открытого train, отдельно от P0/development/demonstration. PARus — все 100 размеченных `dev` элементов; few-shot берётся только из train.
- **LIBRA:** по 30 элементов на каждую ячейку 3 задачи × длина 4K/8K/16K; смешанные исходные классы отфильтровываются перед детерминированным выбором. Выбранные строки выводятся в исходном порядке.

Состав каждой выборки фиксируется до получения предсказаний. Манифест содержит source repository/revision/path, SHA-256 оригинала и локального преобразования, SHA-256 JSONL, item IDs и их hash, seed, prompt/template hash, few-shot ID и hash, evaluator revision, tokenizer revision/hash и число токенов отдельно для каждой модели. Платформенные адаптеры читают один и тот же JSONL и item IDs.

## Официальная оценка

- **MathLogicQA:** генерация, официальное извлечение первого отдельного `A`–`D` по pinned regex, fallback `-1`, официальный Exact Match (численно равен Accuracy на label-классе).
- **MultiQ:** zero-shot; исходный текст обрезается по `\n`, обрезаются внешние пробелы, сохраняются все ответы из `outputs` через `;`. Применяются опубликованные SQuAD Exact Match и token F1, максимум по эталонным ответам.
- **ruOpenBookQA:** генерация и извлечение первого отдельного `A`–`D`; официальный Exact Match и macro-F1 через `scikit-learn`.
- **PARus:** генерация и извлечение первой отдельной цифры по официальному regex; официальный Exact Match (численно равен Accuracy на label-классе).
- **LIBRA:** официальные task prompts для `ru_babilong_qa3`, `long_context_multiq` и `matreshka_names`; EM с `pymorphy3` лемматизацией, пунктуационной нормализацией и всеми `positive_outputs`, агрегирование по исходному классу длины.

Loglikelihood MERA и генеративные task configs являются разными процедурами и не смешиваются. Ни один локальный scorer не исполняет inference.

## Установка зависимостей и подготовка

Сначала скачиваются только разрешённые файлы с pinned revisions/hashes: task JSONL/parquet, evaluator source, model configs и tokenizers. Скрипт не использует рекурсивную загрузку репозиториев, поэтому веса моделей не затрагиваются. Подготовка выполняется в виртуальном окружении проекта:

```sh
PYTHONPATH=experiments .venv/bin/python -m quantbench download-research-sources \
  --registry experiments/quantbench/source_registry.json \
  --source-root datasets/source
.venv/bin/python -m pip install pyarrow==21.0.0 pymorphy3==2.0.6 pymorphy3-dicts-ru==2.4.417150.4580142 transformers==4.57.6 tokenizers==0.22.2 jinja2==3.1.6 scikit-learn==1.6.1 scipy==1.13.1
PYTHONPATH=experiments .venv/bin/python -m quantbench prepare-research-data \
  --source-root datasets/source \
  --output-root results/raw/dataset-exports/m3.3.1 \
  --seed 20261124
```

Команда проверяет локальные hashes pinned source manifest, преобразует только три указанных LIBRA parquet, проверяет воспроизводимость конвертации, загружает tokenizer только из локальных каталогов и формирует все P0/P1/M2 exports. Каталоги tokenizer должны содержать файлы соответствующих закреплённых ревизий Qwen3-1.7B, Qwen3-4B, Qwen3-8B и Qwen3.5-4B; никаких весов для этого не требуется. Для повторного запуска существующий экспорт не перезаписывается при изменении содержимого.

Экспортный каталог размещается в `results/raw/` и не предназначен для добавления в репозиторий. Сводка находится в `export-summary.json`; каждый JSONL сопровождается manifest. Значения токенов нельзя переносить между моделями: они рассчитаны с соответствующими tokenizer и chat template.

## Scorer API

`quantbench.dataset_pipeline.score_mera()` возвращает post-processed ответы, per-item метрики и агрегаты. `LibraScorer.score_item()` и `LibraScorer.aggregate()` реализуют LIBRA procedure. Пустая MERA reference, неизвестная задача или недоступная обязательная зависимость приводят к явной ошибке; универсальная Exact Match как замена не применяется.

Synthetic fixtures используются только unit-тестами. Они не попадают в реальные exports и не считаются результатами исследования.
