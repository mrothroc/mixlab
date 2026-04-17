BIN ?= mixlab
CGO_ENABLED ?= 1
TAGS ?= mlx

.PHONY: build test vet lint setup clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## Build mixlab binary
	CGO_ENABLED=$(CGO_ENABLED) go build $(if $(TAGS),-tags $(TAGS),) -o $(BIN) ./cmd/mixlab

test: ## Run all tests
	go test ./... -count=1 -timeout 120s

vet: ## Run go vet
	go vet ./...

lint: ## Run all lint checks (golangci-lint + file size)
	golangci-lint run ./...
	@find . -name "*.go" -not -path "./vendor/*" -exec wc -l {} + | sort -rn | \
		awk '$$1 > 1000 && $$2 != "total" {printf "OVER 1000 LINES: %s (%d)\n", $$2, $$1; err=1} END {exit err}'

setup: ## Set up pre-commit hooks and install tools
	git config core.hooksPath .githooks
	@which golangci-lint > /dev/null 2>&1 || echo "Install golangci-lint: brew install golangci-lint"
	@echo "Pre-commit hooks enabled."

clean: ## Remove build artifacts
	rm -f $(BIN)
