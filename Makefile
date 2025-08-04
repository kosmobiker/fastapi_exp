.PHONY: fmt test verify

fmt:
	ruff format .
	ruff check --fix .

test:
	docker-compose up -d test_db --wait
	pytest --cov=app --cov-report=term --cov-fail-under=85


verify: fmt test
	@echo "Verification complete"