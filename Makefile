# VerifAI — Developer commands

.PHONY: install run-api run-dashboard run docker-up test

install:
	pip install -r requirements.txt

run-api:
	uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard/app.py

run: ## Run API + dashboard in parallel
	make run-api & make run-dashboard

docker-up:
	docker compose up --build

docker-down:
	docker compose down

test:
	python -m pytest tests/ -v
