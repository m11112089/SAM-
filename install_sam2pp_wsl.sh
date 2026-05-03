#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR=".venv"
TORCH_FLAVOR="cpu"
DOWNLOAD_CHECKPOINT=0
CHECKPOINT_ONLY=0
BUILD_EXT=0
VERIFY=1

usage() {
  cat <<'EOF'
Usage: bash install_sam2pp_wsl.sh [options]

Create a WSL Python virtual environment and install SAM2-Plus dependencies.

Options:
  --venv DIR              Virtualenv directory. Default: .venv
  --cpu                   Install CPU-only PyTorch wheels. Default.
  --cuda 121              Install PyTorch CUDA 12.1 wheels and build optional CUDA extension.
  --download-checkpoint   Download MCG-NJU/SAM2-Plus checkpoint to checkpoints/SAM2-Plus.
  --checkpoint-only       Only download the checkpoint; do not install packages.
  --skip-build-ext        Skip python setup.py build_ext --inplace.
  --skip-verify           Skip post-install import verification.
  -h, --help              Show this help.

Examples:
  bash install_sam2pp_wsl.sh
  bash install_sam2pp_wsl.sh --download-checkpoint
  bash install_sam2pp_wsl.sh --cuda 121 --venv .venv-cu121

After install:
  source .venv/bin/activate
  python interactive_video_box_track.py input.mp4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="${2:?missing value for --venv}"
      shift 2
      ;;
    --cuda)
      CUDA_VERSION="${2:?missing value for --cuda}"
      case "$CUDA_VERSION" in
        121) TORCH_FLAVOR="cu121"; BUILD_EXT=1 ;;
        *) echo "Unsupported --cuda value: $CUDA_VERSION. Use 121 or --cpu." >&2; exit 2 ;;
      esac
      shift 2
      ;;
    --cpu)
      TORCH_FLAVOR="cpu"
      shift
      ;;
    --download-checkpoint)
      DOWNLOAD_CHECKPOINT=1
      shift
      ;;
    --checkpoint-only)
      CHECKPOINT_ONLY=1
      DOWNLOAD_CHECKPOINT=1
      VERIFY=0
      shift
      ;;
    --skip-build-ext)
      BUILD_EXT=0
      shift
      ;;
    --skip-verify)
      VERIFY=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required in WSL." >&2
  exit 1
fi

PYTHON_VERSION="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
case "$PYTHON_VERSION" in
  3.10|3.11|3.12|3.13) ;;
  *)
    echo "SAM2-Plus requires Python >= 3.10; found Python $PYTHON_VERSION." >&2
    exit 1
    ;;
esac

download_checkpoint() {
  echo "[INFO] Downloading SAM2-Plus checkpoint"
  mkdir -p checkpoints/SAM2-Plus
  if ! command -v hf >/dev/null 2>&1; then
    python3 -m pip install --user "huggingface_hub[cli]"
    export PATH="$HOME/.local/bin:$PATH"
  fi
  hf download MCG-NJU/SAM2-Plus --local-dir checkpoints/SAM2-Plus
}

if [[ "$CHECKPOINT_ONLY" == "1" ]]; then
  download_checkpoint
  echo "[OK] Checkpoint download completed."
  exit 0
fi

if [[ -d "$VENV_DIR" && ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "[WARN] Removing incomplete virtual environment: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[INFO] Creating virtual environment: $VENV_DIR"
  if ! python3 -m venv "$VENV_DIR"; then
    echo "[WARN] python3 -m venv failed; trying virtualenv fallback."
    python3 -m pip install --user virtualenv
    python3 -m virtualenv "$VENV_DIR"
  fi
else
  echo "[INFO] Reusing virtual environment: $VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ "$TORCH_FLAVOR" == "cpu" ]]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
else
  TORCH_INDEX_URL="https://download.pytorch.org/whl/$TORCH_FLAVOR"
fi

echo "[INFO] Installing PyTorch 2.5.1 from $TORCH_INDEX_URL"
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url "$TORCH_INDEX_URL"

echo "[INFO] Installing SAM2-Plus and runtime dependencies"
python -m pip install -r sam2_plus/requirements.txt
python -m pip install -r sav_dataset/requirements.txt
python -m pip install opencv-python Pillow tqdm pyyaml eva-decord "huggingface_hub[cli]"
python -m pip install -e .

if [[ "$TORCH_FLAVOR" == "cpu" ]]; then
  export SAM2_BUILD_CUDA=0
fi

if [[ "$BUILD_EXT" == "1" ]]; then
  echo "[INFO] Building optional SAM2 CUDA extension"
  export SAM2_BUILD_ALLOW_ERRORS="${SAM2_BUILD_ALLOW_ERRORS:-1}"
  python setup.py build_ext --inplace
fi

if [[ "$DOWNLOAD_CHECKPOINT" == "1" ]]; then
  download_checkpoint
fi

if [[ "$VERIFY" == "1" ]]; then
  echo "[INFO] Verifying installation"
  python - <<'PY'
import importlib
import torch

print("python ok")
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())

for name in [
    "cv2",
    "decord",
    "hydra",
    "iopath",
    "einops",
    "natsort",
    "peft",
    "pandas",
    "sam2",
    "sam2_plus",
]:
    importlib.import_module(name)
    print("import ok:", name)

from sam2_plus.build_sam import build_sam2_video_predictor_plus
print("import ok: build_sam2_video_predictor_plus")

try:
    from sam2 import _C
    print("import ok: sam2._C")
except Exception as exc:
    print("warning: sam2._C is unavailable:", exc)
PY
fi

cat <<EOF
[OK] SAM2-Plus environment is ready.

Activate it with:
  cd "$ROOT_DIR"
  source "$VENV_DIR/bin/activate"

Run the interactive tracker with:
  python interactive_video_box_track.py /path/to/input.mp4
EOF
