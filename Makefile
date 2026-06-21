# =============================================================================
# SGLang + Gateway — 镜像构建
# 运维管理请用 ./manage.sh (或 ./manage.sh -h 查看帮助)
# =============================================================================
SHELL := /bin/bash

IMAGE_NAME  ?= sglang
IMAGE_TAG   ?= local
REGISTRY    ?= docker.io/sekfung
FULL_IMAGE  := $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

BUILD_DATE      ?= $(shell date +%Y%m%d)
GIT_SHA         ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
GIT_DIRTY       := $(shell test -n "$$(git status --porcelain 2>/dev/null)" && echo -dirty)
VERSION_TAG     ?= $(IMAGE_TAG)-$(BUILD_DATE)-$(GIT_SHA)$(GIT_DIRTY)
VERSIONED_IMAGE := $(REGISTRY)/$(IMAGE_NAME):$(VERSION_TAG)

TORCH_CUDA_ARCH_LIST ?= 9.0\;10.0\;10.3
MAX_JOBS            ?= $(shell nproc)
USE_MIRRORS    ?= 1
PIP_INDEX_URL  ?= https://mirrors.aliyun.com/pypi/simple/
HF_ENDPOINT    ?= https://hf-mirror.com
BUILD_CONTEXT  := /opt/models
DOCKERFILE     := $(CURDIR)/Dockerfile.local

export DOCKER_BUILDKIT ?= 1
.DEFAULT_GOAL := help

.PHONY: help build build-no-cache push info

help: ## 显示帮助
	@echo "make build        构建本地镜像"
	@echo "make push         推送镜像"
	@echo "make info         显示配置"
	@echo ""
	@echo "运维管理: ./manage.sh -h"

build: ## 构建本地镜像 (同时打 :local 与版本化标签)
	@echo "=== Building $(FULL_IMAGE) ==="
	@echo "  Versioned: $(VERSIONED_IMAGE)"
	@echo "  Arch: $(TORCH_CUDA_ARCH_LIST)  MaxJobs: $(MAX_JOBS)"
	docker build \
	  -f "$(DOCKERFILE)" -t "$(FULL_IMAGE)" -t "$(VERSIONED_IMAGE)" \
	  --build-arg TORCH_CUDA_ARCH_LIST="$(TORCH_CUDA_ARCH_LIST)" \
	  --build-arg MAX_JOBS="$(MAX_JOBS)" \
	  --build-arg USE_MIRRORS="$(USE_MIRRORS)" \
	  --build-arg PIP_INDEX_URL="$(PIP_INDEX_URL)" \
	  --build-arg HF_ENDPOINT="$(HF_ENDPOINT)" \
	  --build-arg SGLANG_BUILD_COMMIT="$(GIT_SHA)" \
	  --build-arg SGLANG_IMAGE_TAG="$(VERSIONED_IMAGE)" \
	  --progress=plain "$(BUILD_CONTEXT)"

build-no-cache: ## 无缓存构建
	docker build --no-cache \
	  -f "$(DOCKERFILE)" -t "$(FULL_IMAGE)" -t "$(VERSIONED_IMAGE)" \
	  --build-arg TORCH_CUDA_ARCH_LIST="$(TORCH_CUDA_ARCH_LIST)" \
	  --build-arg MAX_JOBS="$(MAX_JOBS)" \
	  --build-arg USE_MIRRORS="$(USE_MIRRORS)" \
	  --build-arg PIP_INDEX_URL="$(PIP_INDEX_URL)" \
	  --build-arg HF_ENDPOINT="$(HF_ENDPOINT)" \
	  --build-arg SGLANG_BUILD_COMMIT="$(GIT_SHA)" \
	  --build-arg SGLANG_IMAGE_TAG="$(VERSIONED_IMAGE)" \
	  --progress=plain "$(BUILD_CONTEXT)"

push: ## 推送镜像 (:local + 版本化标签)
	docker push "$(FULL_IMAGE)"
	docker push "$(VERSIONED_IMAGE)"

info: ## 显示配置
	@echo "Image:       $(FULL_IMAGE)"
	@echo "Versioned:   $(VERSIONED_IMAGE)"
	@echo "Arch:        $(TORCH_CUDA_ARCH_LIST)"
	@echo "MaxJobs:     $(MAX_JOBS)"
	@echo "Mirrors:     $(if $(filter 1,$(USE_MIRRORS)),enabled,disabled)"
