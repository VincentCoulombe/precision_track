#!/usr/bin/env bash

SCRIPT_DIR="$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)"
source "$SCRIPT_DIR/utils.sh"

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
      check_docker_gpu || exit 1
      build_image "${CUDA_DOCKERFILE}" "${TAG_CUDA}"
      [[ "$run_tests" == "yes" ]] && run_pytest "${TAG_CUDA}"
      ;;
    both)
      info "Building both CPU and CUDA images..."
      build_image "${CPU_DOCKERFILE}" "${TAG_CPU}"
      [[ "$run_tests" == "yes" ]] && run_pytest "${TAG_CPU}"

      if docker_supports_gpus; then
        check_docker_gpu || exit 1
        build_image "${CUDA_DOCKERFILE}" "${TAG_CUDA}"
        [[ "$run_tests" == "yes" ]] && run_pytest "${TAG_CUDA}"
      else
        nvidia_container_toolkit_missing
        exit 1
      fi
      ;;
  esac

  info "All done."
}

main "$@"