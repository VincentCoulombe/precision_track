source ./utils.sh

main() {
  check_system || exit 1

  local mode="auto" update="no"

  # Parse the provided arguments
  while [[ "${1-}" =~ ^- ]]; do
    case "${1}" in
      --cpu)        mode="cpu" ;;
      --cuda)       mode="cuda" ;;
      --update)     update="yes" ;;
      -h|--help)    usage; exit 0 ;;
      *)            err "Unknown flag: $1"; usage; exit 2 ;;
    esac
    shift
  done

  # Auto-detect mode (if no other supported arguments were provided)
  if [[ "$mode" == "auto" ]]; then
    info "Automatically selecting the container."
    if is_cuda_accelerated_host; then 
      info "Your device is CUDA-accelerated..."
      mode="cuda"
    else
      info "Your device is not CUDA-accelerated..."
      mode="cpu"
    fi
  fi

  ensuring_image_exists "$mode"
  launching_container "$mode" "$update"
}

main "$@"