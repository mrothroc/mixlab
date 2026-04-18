BIN ?= mixlab
CGO_ENABLED ?= 1
TAGS ?= mlx

# Auto-detect MLX path from Homebrew, pip, or site-packages.
# Override with: make build MLX_PREFIX=/path/to/mlx
MLX_PREFIX ?= $(shell \
	([ -d /opt/homebrew/opt/mlx/include/mlx ] && echo /opt/homebrew/opt/mlx) || \
	python3 -c "import mlx, os; print(os.path.dirname(mlx.__file__))" 2>/dev/null || \
	python3.12 -c "import mlx, os; print(os.path.dirname(mlx.__file__))" 2>/dev/null || \
	python3.11 -c "import mlx, os; print(os.path.dirname(mlx.__file__))" 2>/dev/null || \
	python3.13 -c "import mlx, os; print(os.path.dirname(mlx.__file__))" 2>/dev/null || \
	(ls -d /opt/homebrew/lib/python3.*/site-packages/mlx 2>/dev/null | tail -1) \
)

# Set CGO flags if MLX was found and we're on macOS
ifneq ($(MLX_PREFIX),)
ifeq ($(shell uname),Darwin)
export CGO_CFLAGS   += -I$(MLX_PREFIX)/include
export CGO_CXXFLAGS += -I$(MLX_PREFIX)/include
export CGO_LDFLAGS  += -L$(MLX_PREFIX)/lib -Wl,-rpath,$(MLX_PREFIX)/lib
endif
endif

.PHONY: build test vet lint setup clean help check-mlx benchmark benchmark-all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

check-mlx: ## Show detected MLX path and CGO flags
	@echo "MLX_PREFIX: $(MLX_PREFIX)"
	@if [ -z "$(MLX_PREFIX)" ]; then \
		echo "WARNING: MLX not found. Install with: pip install mlx"; \
		echo "Or set manually: make build MLX_PREFIX=/path/to/mlx"; \
	else \
		echo "CGO_CFLAGS:   $(CGO_CFLAGS)"; \
		echo "CGO_CXXFLAGS: $(CGO_CXXFLAGS)"; \
		echo "CGO_LDFLAGS:  $(CGO_LDFLAGS)"; \
	fi

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

benchmark: build ## Run Shakespeare nanoGPT benchmark
	./benchmarks/run.sh

benchmark-all: build ## Train all example configs and compare (requires example data)
	./$(BIN) -mode arch_race -configs examples/ -train 'data/example/train_*.bin'

clean: ## Remove build artifacts
	rm -f $(BIN)
