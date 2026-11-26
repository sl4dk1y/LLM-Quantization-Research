# Политика безопасности памяти Apple Metal

## Версия и статус

Текущая версия — `macos-safety-v2.0`. Она сохраняет исправленный учёт свободных страниц, исходные показатели `memory_pressure -Q` и полный временной ряд ресурсных измерений.

Владелец оборудования утвердил для Apple Metal следующие пороги: перед запуском требуется не менее 4 ГиБ освобождаемой памяти; во время выполнения ниже 3 ГиБ формируется предупреждение; ниже 2 ГиБ выполняется аварийная остановка. Само по себе нахождение между 2 и 3 ГиБ не запускает прежний пятиисекундный таймер остановки. Начальное неизвестное состояние Memory Pressure допустимо, если Dispatch listener доступен, все обязательные числовые метрики доступны и остальные условия Safety Guard соблюдены. Неизвестное значение сохраняется как `null` и не переименовывается в `normal`.

Метрики проверяются каждую секунду. Для начального допуска требуется 30 секунд непрерывного безопасного состояния с запасом не менее 4 ГиБ, без роста swap и при допустимой температуре. Во время исполнения значения не ниже 3 ГиБ допускают продолжение; в диапазоне от 2 до 3 ГиБ выставляется предупреждение. Критическое событие Dispatch, значение ниже 2 ГиБ, потеря обязательной телеметрии и прочие действующие аварийные условия по-прежнему немедленно останавливают runtime. Ограничения по swap и температуре не ослаблены.

## Достоверность Memory Pressure

На исследуемом Mac команда `/usr/sbin/sysctl -n kern.memorystatus_level` вернула `75`, а `/usr/bin/memory_pressure -Q` сообщила `System-wide memory free percentage: 75%`. Это показатель свободной памяти в процентах, а не именованный уровень `normal`, `warning` или `critical`; он сохраняется только как диагностическое значение.

Публичный `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` сообщает о событиях изменения давления. Он не предоставляет отдельного запроса первоначального текущего уровня. Поэтому отсутствие события после запуска источника не трактуется как нормальное состояние. В версии v2.0 начальная категория может оставаться `null` при работающем listener; любое полученное событие `critical` немедленно останавливает исполнение, а потеря listener блокирует продолжение.

`kern.memorystatus_level` и `memory_pressure -Q` не используются для присвоения категории Memory Pressure. Их значения сохраняются только как диагностические показатели.

## Учёт страниц

`vm_stat` печатает `Pages free` без speculative pages и печатает `Pages speculative` отдельно. В структуре XNU `vm_statistics64` speculative pages уже входят в `free_count`. Сборщик восстанавливает свободный счётчик один раз как `Pages free + Pages speculative`, затем вычисляет:

- `system_used_bytes = physical_memory - reconstructed_free_count`;
- `reclaimable_memory_estimate_bytes = reconstructed_free_count + file_backed + purgeable`.

Active и inactive pages остаются только в широкой диагностической оценке доступной памяти и не добавляются к консервативной reclaimable-оценке. Сохраняются исходные строки `memory_pressure -Q`, её код завершения и исходные счётчики страниц.

## Сохранение измерений

При production-запуске SafetyWatchdog добавляет каждое измерение в `resource_samples.jsonl`: временную метку, состояние Safety Guard, состояние Memory Pressure и исходный набор метрик. Сводка пиков остаётся дополнительным производным представлением.

Фактическая argv-команда сохраняется в `runtime.log` до запуска сервера. Строки журнала llama-server о буферах сохраняются в том же журнале и индексируются в итоговой записи `block_metrics.jsonl`, когда runtime штатно завершён либо остановлен защитой.

## Источники

- [XNU: структура vm_statistics64](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/vm_statistics.h)
- [XNU: уведомления и уровни давления памяти](https://github.com/apple-oss-distributions/xnu/blob/main/doc/vm/memorystatus_notify.md)
- [Apple Developer Documentation: DISPATCH_SOURCE_TYPE_MEMORYPRESSURE](https://developer.apple.com/documentation/Dispatch/DISPATCH_SOURCE_TYPE_MEMORYPRESSURE)
- [Apple Developer Documentation: Dispatch memory pressure flags](https://developer.apple.com/documentation/dispatch/dispatch_source_memorypressure_flags_t)
