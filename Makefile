.PHONY: setup
setup:
	export PYTHONPATH="$$PYTHONPATH:$$PWD"
	pipenv	install
	@echo $$PYTHONPATH
	@echo $$PWD
.PHONY: docker
docker:
	docker build -t my_postgres_image .
	docker run -d -p 5432:5432 --name my_postgres_container my_postgres_image
.PHONY: fmt
fmt:
	pipenv run black	src/
	pipenv run black	tests/
.PHONY: test
test:
	pipenv run pytest	-v
.PHONY: coverage
coverage:
	pipenv run pytest --cov=src --cov-fail-under=80
	rm -f .coverage*
	rm -rf htmlcov/
.PHONY: pylint
pylint:
	pylint src/ --fail-under=8.0
.PHONY: train
train:
		pipenv run python	src/train/trainer.py
.PHONY: verify
verify: fmt test coverage pylint