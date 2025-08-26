# Docker guide — Build & launch your PrecisionTrack development environment

### 1) Build your Docker image (one time)

- **Make the build script executable (once):**

```bash
chmod +x ./docker/building_image.sh
```

- **Build options:**

  - Device target (choose one):

    - `--cpu` → build CPU image
    - `--cuda` → build CUDA image
    - `--both` → build both images

  - Auto-detect default:

    - No device flag → tries CUDA; falls back to CPU if CUDA isn’t available.

  - Faster build (skip sanity checks):

    - `--skip-tests` → skip automatic checks (faster, but you lose verification).

  - **Examples:**

  ```bash
  ./docker/building_image.sh # Auto-detect default
  ./docker/building_image.sh --cuda
  ./docker/building_image.sh --cpu --skip-tests
  ./docker/building_image.sh --both
  ```

### 2) Launch your development container (each session)

- **What it does:**

  - Starts your PrecisionTrack environment (container).
  - Keeps it running while your terminal stays open.
  - Safe to close/relaunch. Your outputs are save directly on the host.

- **Launch options:**

  - Device target (choose one):

    - `--cpu` → launch CPU image
    - `--cuda` → launch CUDA image

  - Auto-detect default:

    - No device flag → tries CUDA; falls back to CPU if CUDA isn’t available.

  - Update code (optional):

    - `--update` → pulls the latest PrecisionTrack code before starting.

  - Auto-build if missing:

    - If the requested image doesn’t exist, the script will build it first.

  - **Examples:**

  ```bash
  ./docker/launching_container.sh # Auto-detect default
  ./docker/launching_container.sh --cuda
  ./docker/launching_container.sh --cpu --update
  ```

  - **Notes:**

    - Closing the terminal stops the container (no data loss, the outputs are on host).
    - Re-run the launch command to start again.

---

### File layout & persistence (recommended)

- **Everything under `precision_track/`:**

```
precision_track/
├─ configs/
│  ├─ settings/
│  └─ metadata/
├─ datasets/
├─ work_dir/         # logs, checkpoints, results
└─ docker/
   ├─ building_image.sh
   └─ launching_container.sh
```

- **Why:** The container is restricted to this directory. Keeping datasets, configs, and outputs here guarantees read/write access and clean portability.

---

### Common workflows

- **Edit configs on host → run in container:**

  - Edit files in `configs/settings/` and `configs/metadata/` with your usual editor.
  - Inside the running container, launch the [tools](https://github.com/VincentCoulombe/precision_track/tree/main/tools)
    - Find outputs fast:
      - Everything lands in `work_dir/` on your host (logs, checkpoints, tracking results).

---

### Troubleshooting

- **CUDA/GPU not detected:**

  - Verify NVIDIA driver + Container Toolkit.
  - Rebuild with `--cuda` and relaunch with `--cuda`.

- **Permission issues on scripts:**

  - `chmod +x ./docker/*.sh`

- **Image missing:**

  - The launch script auto-builds if needed; otherwise run the build script explicitly.

- **Rebuild from scratch:**
  - Re-run `building_image.sh` with your desired flags.
