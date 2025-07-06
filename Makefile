.PHONY: install test lint run

install:
	pip install -e .

test:
	pytest tests/ -v

lint:
	flake8 src/ tests/
	black --check src/ tests/

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

setup-db:
	python scripts/setup_db.py