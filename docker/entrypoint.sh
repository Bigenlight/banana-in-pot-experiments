#!/usr/bin/env bash
# Container entrypoint: put the pip-provided NVIDIA CUDA libs that live inside the
# mounted lr_env onto the dynamic-loader path, then exec the requested command.
#
# WHY: torchcodec's CUDA video decoder dlopen()s libnppicc.so.12 (NVIDIA NPP) and
# friends at first-decode time. On the ORIGINAL bare-metal box those libs came
# from a system-wide CUDA toolkit, so they never made it into requirements-lock.txt
# (a pip freeze). This thin python:3.12 image has NO system CUDA, so we instead
# pip-install nvidia-npp-cu12 into lr_env (see docker/requirements-extra.txt) and
# expose every nvidia/*/lib dir here. torch itself preloads its own bundled libs,
# but NPP is not one of them — hence this shim.
set -e

VENV="${VIRTUAL_ENV:-/workspace/lr_env}"
NVIDIA_ROOT="$VENV/lib/python3.12/site-packages/nvidia"
if [ -d "$NVIDIA_ROOT" ]; then
  EXTRA="$(find "$NVIDIA_ROOT" -maxdepth 2 -name lib -type d | paste -sd: -)"
  if [ -n "$EXTRA" ]; then
    export LD_LIBRARY_PATH="${EXTRA}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

exec "$@"
