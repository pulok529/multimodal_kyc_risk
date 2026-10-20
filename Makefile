.PHONY: install install-cpu lint test train api docker-build docker-run clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

install-cpu:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest -q

train:
	$(PYTHON) -m kyc_risk.train --n 6000 --epochs 8 --out artifacts/

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t kyc-risk:latest .

docker-run:
	docker run -p 8000:8000 --name kyc-risk-app kyc-risk:latest

clean:
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__ api/__pycache__
	rm -rf .pytest_cache .ruff_cache *.egg-info build dist
