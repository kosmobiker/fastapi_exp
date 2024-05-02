.EXPORT_ALL_VARIABLES:

PYTHONPATH = "$$PYTHONPATH:$PPYTHONPATHWD"

.PHONY: setup
setup:
	pipenv	install
	@echo $$PYTHONPATH
	pipenv	shell
.PHONY: fmt
fmt:
	black	.
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
	python	src/train/train.py
