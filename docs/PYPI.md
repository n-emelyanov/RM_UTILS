# Библиотека утилит для риск-менеджмента ML-моделей

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)

Библиотека разработана для помощи в реализации и мониторинге рисковых ML-моделей в финансовой и других риск-ориентированных областях. Включает инструменты для расчета метрик, анализа стабильности, генерации отчетов и работы с данными.

## Ключевые возможности
- Расчет PSI (Population Stability Index) и других метрик стабильности
- Специализированные метрики для оценки качества ML-моделей в риск-менеджменте
- Генерация автоматизированных отчетов
- Утилиты для работы с SQL-запросами
- Инструменты для обработки и анализа данных


## Установка
```bash
pip install rm-utils
```


## Использование

```python
from rm_utils.reports import ExcelReporter

# Пример создания отчета в Excel
path = r"/path_to_excel/report.xlsx"
writer = ExcelReporter(path)

# Добавление датафреймов
writer.add_dataframe(data=your_dataframe, row_offset=4, col_offset=2)

# Сохранение отчета
writer.save()
```
Подробные примеры использования смотрите в [examples/usage_examples.ipynb](./examples/usage_examples.ipynb).


## История изменений

Все изменения подробно описаны в [CHANGELOG.md](https://github.com/n-emelyanov/RM_UTILS/blob/master/CHANGELOG.md)
