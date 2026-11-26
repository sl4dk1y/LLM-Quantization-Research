# Реестр артефактов моделей

Проверены публичные model cards и списки файлов Hugging Face 24.11.2026. Веса не загружались. Все описанные репозитории используют Apache-2.0 по HF metadata и ссылаются на исходные Qwen checkpoints. Названия `Q4_K_M` и т. п. сами по себе не доказывают одинаковую процедуру конвертации или квантизации.

## Сопоставление источников

| Источник | Покрытие предварительной когорты | Подготовка и ограничения |
| --- | --- | --- |
| [Unsloth](https://huggingface.co/unsloth) | Qwen3-1.7B, 4B, 8B и Qwen3.5-4B; найден общий набор BF16 и стандартных Q4_K_M/Q5_K_M/Q6_K/Q8_0; также опубликованы UD-типы | У моделей Qwen3 имеются `imatrix_unsloth.dat`, у Qwen3.5 — `imatrix_unsloth.gguf_file`; calibration corpus/точные quantizer args/полная воспроизводимая команда не подтверждены. Не смешивать UD-* с обычными llama.cpp типами. |
| [ggml-org](https://huggingface.co/ggml-org) | Официальная conversion organization; Qwen3-4B в текущем репозитории содержит Q4_K_M/Q8_0/F16, но не весь набор Q5_K_M/Q6_K; для Qwen3.5 видны Q4_K_M/Q8_0/BF16 и отдельные MTP/mmproj артефакты | README Qwen3-4B указывает базовый checkpoint и Apache-2.0, но не описывает quantization args/imatrix. Для полного одинакового ряда across checkpoints текущие списки файлов требуют дополнительной сверки. |
| [Qwen](https://huggingface.co/Qwen) | Qwen3 checkpoints и GGUF репозитории доступны в официальной организации; 4B/8B cards перечисляют обычные типы, включая Q4_K_M/Q5_K_M/Q6_K/Q8_0. Для 1.7B card показывает Q8_0. Qwen3.5-4B-GGUF в этой организации не найден | Карточки связывают файлы с `Qwen/Qwen3-*`, указывают Apache-2.0 и llama.cpp. F16 и одинаковая квантовочная процедура полного ряда не подтверждены по всем размерам. |

## Предварительное сравнение источников

Unsloth — предварительный кандидат на широкое покрытие всех четырёх когорт стандартными квантованиями, но файлы BF16 нельзя считать схемой F16 из M2. Поэтому этот набор сам по себе не закрывает точный P1 baseline. Для Qwen3-4B на P0/P1 найден отдельный `ggml-org` repo с точными F16/Q8_0/Q4_K_M типами; его conversion/quantization provenance также ещё не подтверждён. Ни один источник пока не утверждён для подтверждающей серии.

До фиксации полной когорты нужно проверить source checkpoint revision, общий F16 GGUF intermediate, converter revision, quantizer args, назначение и provenance imatrix, tokenizer/template, reasoning mode и совместимость с зафиксированным llama.cpp commit. Если происхождение файлов не удаётся подтвердить, подготовить общий набор из официальных Qwen checkpoints одним воспроизводимым llama.cpp pipeline после отдельного ресурсного решения. Сейчас веса не скачивать и не конвертировать.

| Модель | Repo ID | Обнаруженные стандартные файлы и размер из listing | Кандидатная revision для file pages | Состояние |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B | `unsloth/Qwen3-1.7B-GGUF` | `Qwen3-1.7B-Q4_K_M.gguf` 1.11 GB; Q5_K_M 1.26 GB; Q6_K 1.42 GB; Q8_0 1.83 GB; BF16 3.45 GB | Q4 page: `cc27747d7419139e44ba97777c2f2fd5dca92ee1`; repo listing: `d7f544e` | Кандидат, источник quant procedure не проверен |
| Qwen3-4B | `unsloth/Qwen3-4B-GGUF` | Q4_K_M 2.50 GB; Q5_K_M 2.89 GB; Q6_K 3.31 GB; Q8_0 4.28 GB; BF16 8.05 GB | Q4 page: `4074b6e5af21b4f7f5019a565af9cf73a4b71882` | Кандидат, файл Q4 SHA страницы: `96fd39fa356d936f2d1c6ddd513c699b5a7ad662c9812728d5f4b98907f1a45d`; нужен локальный пересчёт |
| Qwen3-8B | `unsloth/Qwen3-8B-GGUF` | Q4_K_M 5.03 GB; Q5_K_M 5.85 GB; Q6_K 6.73 GB; Q8_0 8.71 GB; BF16 16.4 GB | Q5 page: `d431b2be484113b464bb912d85c78cffe7d80eb6`; listing: `a6adef1` | Кандидат, Q4 revision/hash следует получить на file page перед фиксацией |
| Qwen3.5-4B | `unsloth/Qwen3.5-4B-GGUF` | Q4_K_M 2.74 GB; Q5_K_M 3.14 GB; Q6_K 3.53 GB; Q8_0 4.48 GB; BF16 8.42 GB | Q4 page: `e87f176479d0855a907a41277aca2f8ee7a09523` | Валидационный кандидат; требуется проверить текстовый режим, MTP/mmproj sidecars и поведение llama.cpp |

Размеры — округлённое отображение Hugging Face, не точные byte counts. Полные SHA-256 Q4_K_M, доступные в file page metadata сейчас: Qwen3-1.7B — `ba491cf470c3cadc624e4c8d6c9a27c998809e8ba8eb938d1689ae87e024b6b7`; Qwen3-4B — `96fd39fa356d936f2d1c6ddd513c699b5a7ad662c9812728d5f4b98907f1a45d`; Qwen3.5-4B — `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`. Это метаданные страниц, не checksum файлов, пересчитанный исследователями. Для 8B точный Q4 hash не считан. Перед запуском оба участника скачивают только зафиксированный файл в рамках отдельно разрешённой подготовки и независимо сверяют SHA-256; несовпадение блокирует парное сравнение.

## Кандидатный набор для P0/P1

Публичный listing `ggml-org/Qwen3-4B-GGUF` содержит все три точных типа P1, включая **F16** (это не BF16), на ревизии `2f3b082b1356a6123f7ed71e65aea340da25d53c`. Карточка связывает модель с `Qwen/Qwen3-4B`, указывает Apache-2.0 и файлы `Qwen3-4B-Q4_K_M.gguf`, `Qwen3-4B-Q8_0.gguf` и `Qwen3-4B-f16.gguf`.

| Очерёдность | Имя файла | Тип | Отображаемый размер | SHA-256 из remote metadata |
| --- | --- | --- | ---: | --- |
| P0 | `Qwen3-4B-Q4_K_M.gguf` | Q4_K_M | 2.50 GB | `ab27b9bfa375a178d6cba48f3ad892b94b7739659dcc7aae8058ce0ffed6b328` |
| P1 | `Qwen3-4B-f16.gguf` | F16 | 8.05 GB | `086aef7ff71d2463d3eaf333b9df7f90914957cb3d78cb46dfd642425bbe4674` |
| P1 | `Qwen3-4B-Q8_0.gguf` | Q8_0 | 4.28 GB | `865fbdcb8119c5cd5ba557b139d49351aa2c4e33bf93bdd7b08e24b573dd1dd4` |

Округлённый общий объём P1 — около 14.83 GB. Remote SHA не заменяют независимый local SHA после получения. Это наиболее прямой кандидат для сохранения F16/Q8_0/Q4_K_M состава, но не утверждённый источник: публичные метаданные не закрепляют quantizer args, общий F16 промежуточный файл, imatrix status/provenance и точную процедуру создания отдельных квантизованных файлов. Все эти пункты остаются блокером для подтверждающей серии.

После отдельного разрешения владельца команды с продолжением частичной загрузки и локальной проверкой SHA-256:

```bash
MODEL_DIR="/path/to/approved/model-cache"
mkdir -p "$MODEL_DIR"
MODEL_REV="2f3b082b1356a6123f7ed71e65aea340da25d53c"
curl -fL -C - "https://huggingface.co/ggml-org/Qwen3-4B-GGUF/resolve/$MODEL_REV/Qwen3-4B-Q4_K_M.gguf" -o "$MODEL_DIR/Qwen3-4B-Q4_K_M.gguf"
printf '%s  %s\n' 'ab27b9bfa375a178d6cba48f3ad892b94b7739659dcc7aae8058ce0ffed6b328' "$MODEL_DIR/Qwen3-4B-Q4_K_M.gguf" | shasum -a 256 -c -
```

После проверки P0 и отдельного допуска к P1:

```bash
curl -fL -C - "https://huggingface.co/ggml-org/Qwen3-4B-GGUF/resolve/$MODEL_REV/Qwen3-4B-f16.gguf" -o "$MODEL_DIR/Qwen3-4B-f16.gguf"
curl -fL -C - "https://huggingface.co/ggml-org/Qwen3-4B-GGUF/resolve/$MODEL_REV/Qwen3-4B-Q8_0.gguf" -o "$MODEL_DIR/Qwen3-4B-Q8_0.gguf"
printf '%s  %s\n' '086aef7ff71d2463d3eaf333b9df7f90914957cb3d78cb46dfd642425bbe4674' "$MODEL_DIR/Qwen3-4B-f16.gguf" | shasum -a 256 -c -
printf '%s  %s\n' '865fbdcb8119c5cd5ba557b139d49351aa2c4e33bf93bdd7b08e24b573dd1dd4' "$MODEL_DIR/Qwen3-4B-Q8_0.gguf" | shasum -a 256 -c -
```

Эти команды подготовлены, но не запускались. Ни одна из двух машин пока не имеет закреплённого согласованного GGUF набора.

## Имматрица и UD

Публичные списки файлов показывают, что Unsloth публикует `imatrix_unsloth.dat` для Qwen3 и `imatrix_unsloth.gguf_file` для Qwen3.5; например, Qwen3-4B revision `4074b6e5af21b4f7f5019a565af9cf73a4b71882` содержит `imatrix_unsloth.dat` (3.84 MB, страница указывает SHA-256 `96fd39fa...` для GGUF Q4; у самой imatrix собственный отдельный hash). Это подтверждает доступность артефакта, но не corpus, calibration prompts, версию генератора matrix и точные параметры использования при каждом quant.

`UD-Q4_K_XL`, `UD-Q5_K_XL` и прочие `UD-*` — отдельная Dynamic Quantization, а не синонимы `Q4_K_M`/`Q5_K_M`. Их исключить из основной матрицы.

## Структура записи до допуска

Для каждого файла зарегистрировать: source checkpoint/revision; GGUF repo ID и полный immutable commit; точное имя; точный byte size; standard quant type; лицензия; converter и llama.cpp commit; quantizer args; imatrix hash/corpus/method или `unknown`; tokenizer/template/reasoning mode; remote file SHA; независимо проверенный локальный SHA-256; дата проверки; статус и причина. Исходные и derived GGUF хранятся вне Git, не перезаписываются.
