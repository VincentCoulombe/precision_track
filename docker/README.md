# Docker guide — Build & launch your PrecisionTrack development environment

### 1) Install and configure Docker

- **Install Docker:**
  - Simply run the following:

  ```bash
  sudo apt install docker.io
  sudo systemctl enable docker
  ```

- **Add user to the Docker group:**

  ```bash
  sudo groupadd docker        # may already exist
  sudo usermod -aG docker $USER
  newgrp docker               # apply group change immediately
  ```

- **Restart your shell**

### 2) Build your Docker image (one time)

- **Make the build script executable (once):**

```bash
(cd ./docker && chmod +x ./building_image.sh)
```

- **Build options:**
  - Device target (choose one):
    - `--cpu` → build CPU image
    - `--cuda` → build CUDA image
    - `--both` → build both images

  - Auto-detect default:
    - No device flag → tries CUDA; falls back to CPU if CUDA isn’t available.

  - Slower build (not recommended, could take hours):
    - `--test` → Perform automatic testing.

  - **Examples:**

  ```bash
  (cd ./docker && bash ./building_image.sh) # Auto-detect default

  (cd ./docker && bash ./building_image.sh --cuda)

  (cd ./docker && bash ./building_image.sh --cpu --test)

  (cd ./docker && bash ./building_image.sh --both)
  ```

### 3) Launch your development container (each session)

- **What it does:**
  - Starts your PrecisionTrack environment (container).
  - Keeps it running while your terminal stays open.
  - Safe to close/relaunch. Your outputs are save directly on the host.

- **Make the build script executable (once):**

```bash
chmod +x ./docker/launching_container.sh
```

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
  bash ./docker/launching_container.sh # Auto-detect default
  bash ./docker/launching_container.sh --cuda
  bash ./docker/launching_container.sh --cpu --update
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

### Environment variable overrides

All defaults can be overridden at call-site without editing the scripts:

| Variable          | Default           | Purpose                                              |
| ----------------- | ----------------- | ---------------------------------------------------- |
| `IMAGE_NAME`      | `precisiontrack`  | Docker image name                                    |
| `CUDA_DOCKERFILE` | `dockerfile.cuda` | Path to the CUDA Dockerfile                          |
| `CPU_DOCKERFILE`  | `dockerfile.cpu`  | Path to the CPU Dockerfile                           |
| `BUILD_ARGS`      | _(empty)_         | Extra arguments forwarded to `docker build`          |
| `PYTEST_ARGS`     | _(empty)_         | Arguments forwarded to `pytest` inside the container |
| `DOCKER`          | `docker`          | Docker CLI binary (e.g. `"sudo docker"`)             |

Example — build under a custom image name without sudo:

```bash
IMAGE_NAME=myproject DOCKER="sudo docker" bash ./docker/building_image.sh --cuda
```

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
