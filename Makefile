# SGLang Docker build with source-built FlashInfer
# ================================================
#
# Usage:
#   make build             Build the full dev image (framework_final target)
#   make build-runtime     Build the production runtime image
#   make shell             Run the built dev image interactively
#   make shell-runtime     Run the built runtime image interactively
#   make help              Show this help

IMAGE_NAME         ?= local/sglang
IMAGE_TAG          ?= dev
CUDA_VERSION       ?= 13.0.1

FLASHINFER_REPO    ?= https://github.com/sekfung/flashinfer.git
FLASHINFER_BRANCH  ?= feat/sm120

DOCKERFILE         ?= docker/Dockerfile.flashinfer_src
BUILD_TYPE         ?= all
SGL_KERNEL_VERSION ?= 0.4.4
GITHUB_ARTIFACTORY ?= github.com

DOCKER_BUILD_ARGS = \
	--build-arg CUDA_VERSION=$(CUDA_VERSION) \
	--build-arg FLASHINFER_REPO=$(FLASHINFER_REPO) \
	--build-arg FLASHINFER_BRANCH=$(FLASHINFER_BRANCH) \
	--build-arg BUILD_TYPE=$(BUILD_TYPE) \
	--build-arg SGL_KERNEL_VERSION=$(SGL_KERNEL_VERSION) \
	--build-arg GITHUB_ARTIFACTORY=$(GITHUB_ARTIFACTORY)

.PHONY: build build-runtime shell shell-runtime help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; \
		       {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

build: ## Build the full development image
	docker build \
		-f $(DOCKERFILE) \
		--target framework_final \
		$(DOCKER_BUILD_ARGS) \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		.

build-runtime: ## Build the production runtime image
	docker build \
		-f $(DOCKERFILE) \
		--target runtime \
		$(DOCKER_BUILD_ARGS) \
		-t $(IMAGE_NAME):$(IMAGE_TAG)-runtime \
		.

shell: ## Run the built dev image interactively
	docker run --gpus all -it --rm \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		/bin/bash

shell-runtime: ## Run the built runtime image interactively
	docker run --gpus all -it --rm \
		$(IMAGE_NAME):$(IMAGE_TAG)-runtime \
		/bin/bash
