.PHONY: setup
setup:
	export PYTHONPATH="$$PYTHONPATH:$$PWD"
	pipenv	install
	@echo $$PYTHONPATH
	@echo $$PWD
.PHONY: fmt
fmt:
	pipenv run black	src/
	pipenv run black	tests/
.PHONY: test
test:
	pipenv run pytest	-v
.PHONY: coverage
coverage:
	pipenv run pytest	--cov=src
	rm -f .coverage*
	rm -rf htmlcov/
.PHONY: train
train:
		pipenv run python	src/train/train.py
