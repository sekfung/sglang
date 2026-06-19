# =============================================================================
# SGLang + FlashInfer — Build, Deploy & Bench
# =============================================================================
# 硬件: 4×RTX PRO 6000 (96GB)  |  基础镜像: sglang:dev-cu13
# 模型: deepseek-v4-flash (FP8) |  构建上下文: /opt/models
# =============================================================================

SHELL := /bin/bash

# ---- Image & Registry ------------------------------------------------------
IMAGE_NAME  ?= sglang
IMAGE_TAG   ?= local
REGISTRY    ?= docker.io/sekfung
FULL_IMAGE  := $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# ---- Arch & Parallelism ----------------------------------------------------
TORCH_CUDA_ARCH_LIST ?= 9.0\;10.0\;10.3
MAX_JOBS            ?= $(shell nproc)

# ---- Mirrors ---------------------------------------------------------------
USE_MIRRORS    ?= 1
PIP_INDEX_URL  ?= https://mirrors.aliyun.com/pypi/simple/
HF_ENDPOINT    ?= https://hf-mirror.com

# ---- Paths -----------------------------------------------------------------
BUILD_CONTEXT  := /opt/models
DOCKERFILE     := $(CURDIR)/Dockerfile.local
FLASHINFER_DIR := /opt/models/flashinfer

# ---- Compose files ---------------------------------------------------------
COMPOSE_128K   := $(CURDIR)/docker-compose-local-128k.yml
COMPOSE_256K   := $(CURDIR)/docker-compose-local-256k.yml
CONTAINER_128K := deepseek-v4-flash-local-128k
CONTAINER_256K := deepseek-v4-flash-local-256k

export DOCKER_BUILDKIT ?= 1
.DEFAULT_GOAL := help

# =============================================================================
# Build
# =============================================================================

.PHONY: build build-no-cache push

build: ## 构建本地镜像
	@echo "=== Building $(FULL_IMAGE) ==="
	@echo "  Arch: $(TORCH_CUDA_ARCH_LIST)  MaxJobs: $(MAX_JOBS)"
	@echo "  Mirrors: $(if $(filter 1,$(USE_MIRRORS)),enabled,disabled)"
	docker build \
	  -f "$(DOCKERFILE)" -t "$(FULL_IMAGE)" \
	  --build-arg TORCH_CUDA_ARCH_LIST="$(TORCH_CUDA_ARCH_LIST)" \
	  --build-arg MAX_JOBS="$(MAX_JOBS)" \
	  --build-arg USE_MIRRORS="$(USE_MIRRORS)" \
	  --build-arg PIP_INDEX_URL="$(PIP_INDEX_URL)" \
	  --build-arg HF_ENDPOINT="$(HF_ENDPOINT)" \
	  --build-arg SGLANG_BUILD_COMMIT="$$(cd $(CURDIR) && git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
	  --build-arg SGLANG_IMAGE_TAG="$(FULL_IMAGE)" \
	  --progress=plain "$(BUILD_CONTEXT)"

build-no-cache: ## 无缓存构建
	docker build --no-cache \
	  -f "$(DOCKERFILE)" -t "$(FULL_IMAGE)" \
	  --build-arg TORCH_CUDA_ARCH_LIST="$(TORCH_CUDA_ARCH_LIST)" \
	  --build-arg MAX_JOBS="$(MAX_JOBS)" \
	  --build-arg USE_MIRRORS="$(USE_MIRRORS)" \
	  --build-arg PIP_INDEX_URL="$(PIP_INDEX_URL)" \
	  --build-arg HF_ENDPOINT="$(HF_ENDPOINT)" \
	  --progress=plain "$(BUILD_CONTEXT)"

push: ## 推送镜像
	docker push "$(FULL_IMAGE)"

# =============================================================================
# Deploy: 128K
# =============================================================================

.PHONY: up-128k down-128k restart-128k logs-128k ps-128k

up-128k: ## 启动 128K (端口 30000)
	docker compose -f $(COMPOSE_128K) up -d
	@echo "128K 启动中 → make logs-128k"

down-128k: ## 停止 128K
	docker compose -f $(COMPOSE_128K) down

restart-128k: ## 重启 128K
	docker compose -f $(COMPOSE_128K) restart

logs-128k: ## 128K 日志
	docker logs -f $(CONTAINER_128K)

ps-128k: ## 128K 状态
	docker compose -f $(COMPOSE_128K) ps

# =============================================================================
# Deploy: 256K
# =============================================================================

.PHONY: up-256k down-256k restart-256k logs-256k ps-256k

up-256k: ## 启动 256K (端口 30000)
	docker compose -f $(COMPOSE_256K) up -d
	@echo "256K 启动中 → make logs-256k"

down-256k: ## 停止 256K
	docker compose -f $(COMPOSE_256K) down

restart-256k: ## 重启 256K
	docker compose -f $(COMPOSE_256K) restart

logs-256k: ## 256K 日志
	docker logs -f $(CONTAINER_256K)

ps-256k: ## 256K 状态
	docker compose -f $(COMPOSE_256K) ps

# =============================================================================
# Convenience
# =============================================================================

.PHONY: up down restart ps logs
up: up-128k
down: down-128k
restart: restart-128k
ps: ps-128k
logs: logs-128k

# =============================================================================
# Benchmark
# =============================================================================

.PHONY: bench-128k bench-256k bench

bench-128k: ## 128K 压测
	CONTAINER=$(CONTAINER_128K) bash $(CURDIR)/bench_sglang.sh

bench-256k: ## 256K 压测
	CONTAINER=$(CONTAINER_256K) bash $(CURDIR)/bench_sglang.sh

bench: bench-128k

# =============================================================================
# Run / Shell
# =============================================================================

.PHONY: run shell serve verify

run: ## 交互式进入容器
	docker run -it --rm --gpus all --ipc=host \
	  --ulimit memlock=-1 --ulimit stack=67108864 \
	  -v $(CURDIR):/src/sglang \
	  -v $(FLASHINFER_DIR):/src/flashinfer \
	  -e HF_ENDPOINT=$(HF_ENDPOINT) \
	  --name sglang-local $(FULL_IMAGE) bash

shell: ## 进入运行中容器
	docker exec -it sglang-local bash

serve: ## 启动服务 (MODEL=path ARGS="...")
	docker run -it --rm --gpus all --ipc=host --network host \
	  --ulimit memlock=-1 --ulimit stack=67108864 \
	  -v $(CURDIR):/src/sglang \
	  -v $(FLASHINFER_DIR):/src/flashinfer \
	  -e HF_ENDPOINT=$(HF_ENDPOINT) \
	  -e SGLANG_SM120_FLASHMLA_BACKEND=flashinfer \
	  --name sglang-server $(FULL_IMAGE) \
	  python3 -m sglang.launch_server --model $(MODEL) --trust-remote-code $(ARGS)

verify: ## 验证 flashinfer SM120 MLA
	docker run --rm --gpus all $(FULL_IMAGE) python3 -c "\
import flashinfer, flashinfer.mla as m; \
print('flashinfer', flashinfer.__version__); \
assert hasattr(m, 'trtllm_batch_decode_sparse_mla_dsv4'); \
print('OK: FlashInfer DSv4 sparse MLA entrypoint present')"

# =============================================================================
# Clean
# =============================================================================

.PHONY: clean clean-all clean-cache

clean: ## 停止容器
	docker compose -f $(COMPOSE_128K) down -v 2>/dev/null || true
	docker compose -f $(COMPOSE_256K) down -v 2>/dev/null || true

clean-all: clean ## 停止容器并删除镜像
	docker rmi $(FULL_IMAGE) 2>/dev/null || true
	docker builder prune -f

clean-cache: ## 清理构建缓存
	docker builder prune -f

# =============================================================================
# Info
# =============================================================================

.PHONY: info

info: ## 显示配置
	@echo "=== Build & Deploy ==="
	@echo "  Image:       $(FULL_IMAGE)"
	@echo "  Arch:        $(TORCH_CUDA_ARCH_LIST)"
	@echo "  MaxJobs:     $(MAX_JOBS)"
	@echo "  Mirrors:     $(if $(filter 1,$(USE_MIRRORS)),enabled,disabled)"
	@echo "  Container:   $(CONTAINER_128K)"
	@echo ""
	@echo "=== Source ==="
	@cd $(CURDIR) && echo "  SGLang:     $$(git rev-parse --abbrev-ref HEAD) ($$(git rev-parse --short HEAD))"
	@cd $(FLASHINFER_DIR) && echo "  FlashInfer: $$(git rev-parse --abbrev-ref HEAD) ($$(git rev-parse --short HEAD))"

# =============================================================================
# Help
# =============================================================================

.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'
