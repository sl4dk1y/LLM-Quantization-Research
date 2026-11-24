# Экспериментальный стенд

На этапе M3 здесь размещается минимальный локальный программный стенд. Реализация инфраструктуры не означает, что датасеты, модели или экспериментальный протокол утверждены. Запуск модели и оценочных серий требует отдельного решения на следующем этапе.

## Подготовка среды

Требуется Python 3.11 или новее. Код стенда использует стандартную библиотеку Python. Для автоматизированных проверок нужен `pytest` (разработка); runtime модели — отдельный локально собранный `llama-server` с Metal на macOS Apple Silicon. Ни модели, ни наборы данных стенд не загружает.

Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pytest
```

Запуск справки и безопасного dry-run:

```bash
python -m quantbench --help
python -m quantbench validate --config experiments/config.example.json
python -m quantbench plan --config experiments/config.example.json
```

Dry-run валидирует конфигурацию и показывает план; он не запускает runtime, не загружает веса и не выполняет инференс.

## Состояние реализации

Точные схемы датасетов MERA/LIBRA, официальные scorer, измерение macOS physical footprint, pressure и thermal state требуют локальной валидации на целевом устройстве. До неё запуск runtime-команды блокируется. См. [архитектуру](../docs/benchmark-architecture.md), [экспериментальный протокол](../docs/experimental-protocol.md) и [протокол измерения ресурсов](../docs/measurement-protocol.md).
