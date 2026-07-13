# banana-in-pot EEF experiments — thin CPU/GPU "toolbox" image.
#
# Design: the CUDA userspace ships *inside* the pip torch wheels (cu128); only the
# host NVIDIA driver + libcuda are used at runtime (injected by nvidia-container-
# toolkit). This was verified empirically on this shared box:
#   host driver 550.144.03 (CUDA 12.4)  +  torch 2.11.0+cu128 (CUDA 12.8)
#   -> torch.cuda.is_available() == True, real matmul on RTX A4000 OK
# (CUDA 12.x minor-version compatibility; driver floor for 12.x is ~525).
# So a plain python:3.12 base is enough — no nvidia/cuda base image needed.
#
# The Python env (lr_env) and datasets are NOT baked in; they are built at run
# time into the mounted /workspace by the repo's own setup.sh / fetch_data.sh,
# so they persist on the host (they are gitignored) and match the repo exactly.
FROM python:3.12

# System libs:
#   git, git-lfs      -> clone lerobot (pinned) + HF large-file downloads
#   ffmpeg            -> torchcodec / PyAV video decode (required by lerobot)
#   build-essential   -> any source builds during pip resolve
#   curl, ca-certs    -> uv installer + HuggingFace downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs ffmpeg build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv (Astral), installed to a world-accessible location so the container can run
# as the HOST user (--user $(id -u):$(id -g)) and NOT leave root-owned lr_env/
# datasets on the shared PC's mounted repo. (setup.sh also auto-installs uv, but
# baking it here avoids a per-run download.)
RUN curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh

# Caches on the mounted workspace so re-runs reuse downloads AND stay user-owned.
# (HOME is set to /workspace at run time via `-e HOME=/workspace` so tools that
# fall back to ~/ still write inside the mount.)
#
# UV_INDEX_STRATEGY=unsafe-best-match: requirements-lock.txt pins exact versions
# (e.g. certifi==2026.6.17) that live on PyPI, while torch/vision (+cu128) live on
# the PyTorch extra index. Newer uv (>=0.11.x) defaults to "first-index only" per
# package (dependency-confusion guard), which wrongly binds certifi to the PyTorch
# index and fails to resolve. Both indexes here are official/trusted, so allowing
# best-match across them is safe and reproduces the original lockfile exactly.
# USER: the container runs as the host UID (--user $(id -u)), which has no entry
# in the image's /etc/passwd. Some libs (torch inductor cache_dir, HF) call
# getpass.getuser() -> pwd.getpwuid() and crash with "uid not found". getpass
# checks $USER (and LOGNAME/…) BEFORE the pwd lookup, so a non-empty USER value
# short-circuits it. The value is only used to name cache subdirs — cosmetic.
ENV HF_HOME=/workspace/.cache/hf \
    UV_CACHE_DIR=/workspace/.cache/uv \
    UV_INDEX_STRATEGY=unsafe-best-match \
    USER=lerobot

# Entrypoint puts the mounted lr_env's pip NVIDIA libs (incl. NPP) on the loader
# path so torchcodec's CUDA decoder can dlopen libnppicc.so.12 (see the script).
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
