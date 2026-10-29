#!/bin/bash
# Sets up the study's Python environment on CERES; run once on the login node:
#   bash scripts/ceres/setup.sh            (from the repository folder)
#
# CERES runs CentOS 7 (Rocks 7, glibc 2.17) with Python 3.6, too old for this code,
# so the environment comes from conda-forge (Miniforge) in the home folder, which
# every compute node shares. conda only installs builds that work with the node's
# glibc, so it may pick versions older than on Michael's Mac; the tests at the end
# show whether they behave the same.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PREFIX="$HOME/miniforge3"
ENV=rfgp

if [ ! -x "$PREFIX/bin/conda" ]; then
    curl -fsSL -o "$HOME/miniforge-installer.sh" \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$HOME/miniforge-installer.sh" -b -p "$PREFIX"
    rm -f "$HOME/miniforge-installer.sh"
fi
source "$PREFIX/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^$ENV "; then
    conda create -y -n "$ENV" -c conda-forge python=3.12 numpy pandas pyarrow numba scipy pyyaml \
        exchange-calendars scikit-learn lightgbm pytest
fi
conda activate "$ENV"
# The neural benchmarks need PyTorch (CPU); skipped with a message if conda cannot install it here.
python -c "import torch" 2>/dev/null || conda install -y -n "$ENV" -c conda-forge "pytorch=*=cpu*" \
    || echo "PyTorch could not be installed: the neural benchmarks (ML-GATED, ML-TFT) will not run on CERES"
pip install --no-deps -e "$REPO"
python - <<'PY'
import numba, numpy, pandas, regime_fusion
print("environment ready:", "pandas", pandas.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
PY
cd "$REPO" && python -m pytest -q -p no:warnings
