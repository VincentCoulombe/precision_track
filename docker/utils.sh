#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'


: "${IMAGE_NAME:=precisiontrack}" 
: "${GPU_CHECK_IMAGE:=nvidia/cuda:12.4.1-base-ubuntu22.04}"
: "${CUDA_DOCKERFILE:=dockerfile.cuda}"
: "${CPU_DOCKERFILE:=dockerfile.cpu}"
: "${TAG_CPU:=cpu}"
: "${TAG_CUDA:=cuda}"
: "${BUILD_ARGS:=}"

DOCKER_BIN="${DOCKER:-docker}"

err()  { printf '[ERROR] %s\n' "$*" >&2; }
info() { printf '[INFO]  %s\n' "$*"; }
warning() { printf '[WARNING] %s\n' "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [--cpu|--cuda|--both] [--skip-tests]
Environment overrides:
  IMAGE_NAME           Image name (default: ${IMAGE_NAME})
  CUDA_DOCKERFILE      Path to CUDA Dockerfile (default: ${CUDA_DOCKERFILE})
  CPU_DOCKERFILE       Path to CPU Dockerfile (default: ${CPU_DOCKERFILE})
  BUILD_ARGS           Extra args for docker build
  PYTEST_ARGS          Args passed to pytest inside container
  DOCKER               Docker CLI (e.g. "sudo docker")
EOF
}

trap 'err "Failed at line $LINENO"; exit 1' ERR


check_system() {
  # Docker binary installed? Check the actual docker binary (last word of DOCKER_BIN).
  if ! have "${DOCKER_BIN##* }"; then
    err "Docker CLI not found. Install Docker first."
    return 1
  fi

  # Can we talk to the Docker daemon?
  if ! ${DOCKER_BIN} info >/dev/null 2>&1; then
    local os
    os="$(uname -s)"
    case "$os" in
      Linux*)
        # Detect WSL2 — uname reports Linux but the issue is Docker Desktop
        if grep -qi microsoft /proc/version 2>/dev/null; then
          err "WSL2 detected. Make sure Docker Desktop is running and WSL integration is enabled."
          err "  Docker Desktop → Settings → Resources → WSL Integration"
        elif id -nG "$(id -un)" | grep -qw docker; then
          err "Docker group present but daemon unreachable. Is the service running?"
        else
          err "User not in docker group. Options:"
          err "  • Add user to docker group and re-login"
          err "  • Use rootless mode"
          err "  • Or run with DOCKER='sudo docker'"
        fi
        ;;
      MINGW*|MSYS*|CYGWIN*|Windows_NT)
        err "Start Docker Desktop (WSL2 backend recommended for GPU)."
        ;;
      *)
        err "Unsupported/unknown OS. MacOS is not yet supported."
        ;;
    esac
    return 1
  fi
}

is_cuda_accelerated_host() { have nvidia-smi; }

nvidia_container_toolkit_missing(){
    err "Docker lacks --gpus support. Install NVIDIA Container Toolkit."
    err "Docs:"
    err " - PrecisionTrack: https://github.com/VincentCoulombe/precision_track"
    err " - NVIDIA Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
}
check_docker_gpu() {
  if ! ${DOCKER_BIN} run --rm --gpus all "$GPU_CHECK_IMAGE" nvidia-smi >/dev/null 2>&1; then
    nvidia_container_toolkit_missing
    return 1
  fi
}

build_image() {
  local dockerfile_path="$1" tag="$2"
  local context_dir="."
  info "Building ${IMAGE_NAME}:${tag} with Dockerfile '${dockerfile_path}'..."
  ${DOCKER_BIN} build --pull --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" ${BUILD_ARGS} \
   -t "${IMAGE_NAME}:${tag}" -f "${dockerfile_path}" "${context_dir}"
}

run_pytest() {
  local tag="$1"
  info "Running pytest inside ${IMAGE_NAME}:${tag}. This last sanity check ensures everything is properly installed."
  if [[ "$tag" == "$TAG_CPU" ]]; then
    ${DOCKER_BIN} run --rm --ipc=host "${IMAGE_NAME}:${tag}" pytest -q -x
  else
    ${DOCKER_BIN} run --rm --gpus all --ipc=host "${IMAGE_NAME}:${tag}" pytest -q -x
  fi
}

image_exists(){
    local tag="$1"
    ${DOCKER_BIN} image inspect "${IMAGE_NAME}:$tag"  >/dev/null 2>&1
}

ensuring_image_exists(){
    local tag="$1"
    if ! image_exists "$tag"; then
        warning "The ${IMAGE_NAME}:$tag Docker image was not found, creating it..."
        local docker_file=""
        if [[ "$tag" == "${TAG_CPU}" ]]; then
            docker_file="${CPU_DOCKERFILE}"
        else
            docker_file="${CUDA_DOCKERFILE}"
            check_docker_gpu || exit 1
        fi
        build_image "$docker_file" "$tag"
        run_pytest "$tag"
    else
        info "Found the ${IMAGE_NAME}:$tag Docker image."
    fi
}

launching_container() {
  local tag="$1"
  local update="$2"
  local repo_root="$3"

  info "Launching the ${IMAGE_NAME}:$tag Docker container..."
  info "The environment will stay active for as long as the terminal remains open."

  local base_opts=(
    --rm --ipc=host
    --mount type=bind,source="${repo_root}",target=/workspace/precision_track
    --user "$(id -u):$(id -g)"
    -w /workspace/precision_track
    -e HOME=/workspace
  )

  local gpu_opts=()
  if [[ "$tag" == "$TAG_CUDA" ]]; then
    check_docker_gpu || exit 1
    gpu_opts+=(--gpus all --group-add video) # Needed for the container to access the gpus
  fi

  local pre_cmd='true'
  if [[ "$update" == "yes" ]]; then
    info "Updatng the ${IMAGE_NAME}:$tag Docker container's PrecisionTrack..."
    if [[ "$tag" == "$TAG_CPU" ]]; then
      pre_cmd='git pull origin main && python -m pip install -e ".[cpu,test]"'
    else
      pre_cmd='git pull origin main && python -m pip install -e ".[cuda,test]"'
    fi
  fi

  ${DOCKER_BIN} run "${base_opts[@]}" "${gpu_opts[@]}" -it \
    --entrypoint bash "${IMAGE_NAME}:$tag" -lc "$pre_cmd; exec bash"
}