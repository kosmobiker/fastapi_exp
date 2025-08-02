SHELL := /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "  build         : Build the Docker image"
	@echo "  run           : Run the Docker container"
	@echo "  stop          : Stop the Docker container"
	@echo "  test          : Run tests inside the Docker container"
	@echo "  lint          : Lint the code inside the Docker container"

.PHONY: build
build:
	@docker build -t fastapi_exp .

.PHONY: run
run:
	@docker run -d -p 8000:8000 --name fastapi_exp fastapi_exp

.PHONY: stop
stop:
	@docker stop fastapi_exp

.PHONY: clean
clean:
	@docker rm fastapi_exp

.PHONY: test
test:
	@docker exec fastapi_exp pytest

.PHONY: ci-test
ci-test:
	@pytest --cov=fastapi_exp --cov-report=term-missing --cov-fail-under=85

.PHONY: lint
lint:
	@docker exec fastapi_exp ruff . --fix

.PHONY: ci-lint
ci-lint:
	@ruff . --fix

.PHONY: verify
verify: lint test

.PHONY: ci-verify
ci-verify: ci-lint ci-test
