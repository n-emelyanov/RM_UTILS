.DEFAULT_GOAL := help

help:  ## Показать доступные команды
	@echo "Доступные команды:"
	@echo "  make install    - Установить зависимости"
	@echo "  make lint       - Проверить код линтером"
	@echo "  make format     - Отформатировать код"
	@echo "  make clean      - Очистить временные файлы"

install:  ## Установить зависимости через Poetry
	poetry install

lint:  ## Проверить код (когда добавите линтеры)
	poetry run flake8 src/ || echo "Установите flake8: poetry add flake8 --dev"
	poetry run black --check src/ || echo "Установите black: poetry add black --dev"

# format:  ## Форматировать код
# 	poetry run black src/

# test:  ## Запустить тесты (когда добавите pytest)
# 	poetry run pytest tests/ || echo "Установите pytest: poetry add pytest --dev"

clean:  ## Очистить временные файлы
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/ dist/ build/ *.egg-info

# publish:  ## Собрать и опубликовать пакет
# 	poetry build
# 	poetry publish

.PHONY: help install lint format test clean publish
