# Экспериментальный стенд

На этапе M3 здесь размещается минимальный локальный программный стенд. Реализация инфраструктуры не означает, что датасеты, модели или экспериментальный протокол утверждены. Запуск модели и оценочных серий требует отдельного решения на следующем этапе.

## Подготовка среды

Код стенда использует стандартную библиотеку Python; минимальная версия Python пока формально не закреплена. Последний автоматизированный прогон был выполнен в локальном окружении Python 3.9 с `pytest==8.4.2` из `requirements-dev.txt`. Runtime модели — отдельный локальный `llama-server`; фактические совместимость и версии не подтверждены. Стенд не скачивает модели и наборы данных.

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

Приоритеты P0–P6 и команды подготовки job manifest, прогресса, snapshot и offline merge описаны в [плане очереди](../docs/experiment-queue.md). Файл `queue.blocks.example.json` — незаполненный шаблон, он не допускает запуск. CLI `run` и `resume` остаются заблокированы: production executor, task-specific official scorers и safety watchdog ещё не интегрированы.

## Состояние реализации

Точные схемы датасетов MERA/LIBRA и официальные scorer требуют закреплённых локальных exports. Текущая macOS реализация не предоставляет требуемую `memory_pressure.category`, поэтому Safety Guard fail-closed блокирует запуск. См. [архитектуру](../docs/benchmark-architecture.md), [экспериментальный протокол](../docs/experimental-protocol.md) и [протокол измерения ресурсов](../docs/measurement-protocol.md).
