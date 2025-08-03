.PHONY: fmt test verify

fmt:
	ruff format .
	ruff check --fix .

test:
	# Add your test command here later
	@echo "Tests not implemented yet"

verify: fmt test
	@echo "Verification complete"