#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

: "${IMAGE_NAME:=precisiontrack}"
: "${GPU_CHECK_IMAGE:=nvidia/cuda:12.4.1-base-ubuntu22.04}"
: "${CUDA_DOCKERFILE:=dockerfile.cuda}"
: "${CPU_DOCKERFILE:=dockerfile.cpu}"

err(){ printf '[ERROR] %s\n' "$*" >&2; }
info(){ printf '[INFO] %s\n' "$*"; }

have(){ command -v "$1" >/dev/null 2>&1; }

check_system(){
  if ! have docker; then
    err "Docker CLI not found. Install Docker Desktop (Windows) or Docker Engine (Linux)."
    return 1
  fi

  # Daemon reachable?
  if ! docker info >/dev/null 2>&1; then
    case "$(uname -s)" in
      Linux)
        if id -nG "$(id -un)" 2>/dev/null | grep -qw docker; then
          err "Docker group present but daemon unreachable. Is the service running?"
        else
          err "User not in 'docker' group (or using rootless elsewhere)."
          err "Options:"
          err "  • Add user to docker group and re-login"
          err "  • Use rootless mode"
          err "  • Set DOCKER='sudo docker' (last resort)"
        fi
        ;;
      MINGW*|MSYS*|CYGWIN*|Windows_NT)
        err "Start Docker Desktop (WSL2 backend recommended for GPU)."
        ;;
      *)
        err "Unsupported OS."
        ;;
    esac
    return 1
  else
    return 0
  fi
}

is_cuda_accelerated_host(){ have nvidia-smi; }

check_docker_gpu(){
  if ! docker run --rm --gpus all --pull=never "$GPU_CHECK_IMAGE" nvidia-smi >/dev/null 2>&1; then
    err "Docker GPU test failed. Ensure NVIDIA Container Toolkit is installed & configured."
    err "Docs: Either refer to PrecisionTrack's or NVIDIA's Container Toolkit & Docker GPU usage:"
    err " - https://github.com/VincentCoulombe/precision_track"
    err " - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    return 1
  fi
}

build_image(){
    dockerfile_path="$1"
    tag="$2"
    docker build --pull -t "$IMAGE_NAME:$tag" -f "$dockerfile_path"
}

run_pytest(){
  echo "Environment sanity check. We will run our automated tests to ensure that PrecisionTrack will run smoothly going forward."
  tag=$1
  if [[ "$tag" == "cpu" ]]; then
    docker run --rm $IMAGE_NAME:$tag pytest -q -x
  else
    docker run --rm --gpus all $IMAGE_NAME:$tag pytest -q -x

  fi
}


check_system || exit 1

if is_cuda_accelerated_host; then
    echo "Your machine is CUDA-accelerated. Building PrecisionTrack's CUDA environment..."
    if docker run --help 2>/dev/null | grep -q -- '--gpus'; then
      check_docker_gpu || {
        err "GPU not usable by Docker. See PrecisionTrack's or NVIDIA's container toolkit install guide."
        exit 1
      }
    else
      err "Your Docker lacks --gpus support. Install NVIDIA Container Toolkit and configure Docker."
      err "Docs: Either refer to PrecisionTrack's or NVIDIA's Container Toolkit & Docker GPU usage:"
      err " - https://github.com/VincentCoulombe/precision_track"
      err " - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
      exit 1
    fi
    build_image $CUDA_DOCKERFILE "cuda"
else
    echo "Your machine is not CUDA-accelerated. Building PrecisionTrack's CPU environment..."
    build_image $CPU_DOCKERFILE "cpu"
fi

run_pytest
