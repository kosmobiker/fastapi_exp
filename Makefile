.PHONY: setup
setup:
	export PYTHONPATH="$$PYTHONPATH:$$PWD"
	pipenv	install
	@echo $$PYTHONPATH
	@echo $$PWD
.PHONY: fmt
fmt:
	black	src/
	black	tests/
.PHONY: test
test:
	pytest	-v
.PHONY: coverage
coverage:
	pytest	--cov=src
	rm -f .coverage*
	rm -rf htmlcov/
.PHONY: train
train:
		pipenv run python	src/train/train.py
