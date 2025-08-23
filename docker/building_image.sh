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
  # Docker binary installed?
  if ! have "${DOCKER_BIN%% *}"; then
    err "Docker CLI not found. Install Docker first."
    return 1
  fi

  # Can we talk to the Docker daemon?
  if ! ${DOCKER_BIN} info >/dev/null 2>&1; then
    case "$(uname -s)" in
      Linux*)
        if id -nG "$(id -un)" | grep -qw docker; then
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
docker_supports_gpus() { ${DOCKER_BIN} run --help | grep -q -- '--gpus'; }

check_docker_gpu() {
  if ! ${DOCKER_BIN} run --rm --gpus all --pull=never "$GPU_CHECK_IMAGE" nvidia-smi >/dev/null 2>&1; then
    nvidia_container_toolkit_missing
    return 1
  fi
}

build_image() {
  local dockerfile_path="$1" tag="$2"
  local context_dir="."
  info "Building ${IMAGE_NAME}:${tag} with Dockerfile '${dockerfile_path}' ..."
  ${DOCKER_BIN} build --pull -t "${IMAGE_NAME}:${tag}" -f "${dockerfile_path}" ${BUILD_ARGS} "${context_dir}"
}

run_pytest() {
  local tag="$1"
  info "Running pytest inside ${IMAGE_NAME}:${tag}. This last sanity check ensures everything installed properly."
  if [[ "$tag" == "$TAG_CPU" ]]; then
    ${DOCKER_BIN} run --rm "${IMAGE_NAME}:${tag}" pytest -q -x
  else
    ${DOCKER_BIN} run --rm --gpus all "${IMAGE_NAME}:${tag}" pytest -q -x
  fi
}

main() {
  check_system || exit 1

  local mode="auto" run_tests="yes"

  # Parse the provided arguments
  while [[ "${1-}" =~ ^- ]]; do
    case "${1}" in
      --cpu)        mode="cpu" ;;
      --cuda)       mode="cuda" ;;
      --both)       mode="both" ;;
      --skip-tests) run_tests="no" ;;
      -h|--help)    usage; exit 0 ;;
      *)            err "Unknown flag: $1"; usage; exit 2 ;;
    esac
    shift
  done

  # Auto-detect mode (if no other supported arguments were provided)
  if [[ "$mode" == "auto" ]]; then
    info "Automatically selecting the build."
    if is_cuda_accelerated_host; then 
      info "Your device is CUDA-accelerated..."
      mode="cuda"
    else
      info "Your device is not CUDA-accelerated..."
      mode="cpu"
    fi
  fi

  case "$mode" in
    cpu)
      info "CPU build selected."
      build_image "${CPU_DOCKERFILE}" "${TAG_CPU}"
      [[ "$run_tests" == "yes" ]] && run_pytest "${TAG_CPU}"
      ;;
    cuda)
      info "CUDA build selected."
      if ! docker_supports_gpus; then
        nvidia_container_toolkit_missing
        exit 1
      fi
      check_docker_gpu
      build_image "${CUDA_DOCKERFILE}" "${TAG_CUDA}"
      [[ "$run_tests" == "yes" ]] && run_pytest "${TAG_CUDA}"
      ;;
    both)
      info "Building both CPU and CUDA images..."
      build_image "${CPU_DOCKERFILE}" "${TAG_CPU}"
      [[ "$run_tests" == "yes" ]] && run_pytest "${TAG_CPU}"

      if docker_supports_gpus; then
        check_docker_gpu
        build_image "${CUDA_DOCKERFILE}" "${TAG_CUDA}"
        [[ "$run_tests" == "yes" ]] && run_pytest "${TAG_CUDA}"
      else
        nvidia_container_toolkit_missing
      fi
      ;;
  esac

  info "All done."
}

main "$@"
