#!/bin/bash
# Sets up the study's Python environment on CERES, once. CERES asks for conda environments to be
# built on a compute node, so start an interactive session first:
#   qrsh -q interactive.q
#   cd ~/regime-fusion-gp && bash scripts/ceres/setup.sh
#
# CERES runs CentOS 7 (Rocks 7, glibc 2.17). Its Pythons (3.6, and 3.9 on the compute nodes)
# are too old for this code (3.11 or later), and CERES recommends conda environments, so the
# environment comes from conda-forge (Miniforge) in the home folder, which every node shares.
# conda-forge still builds for glibc 2.17, and conda installs only builds that work with it, so
# some versions may be older than on Michael's Mac; the tests at the end show whether they
# behave the same.
set -euo pipefail
case "$(hostname)" in
    compute-*) ;;
    *) echo "Please run this on a compute node: start one with 'qrsh -q interactive.q' and run it there." >&2
       exit 1 ;;
esac
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
# (Re)built unless it already has every package, so a set-up cut short can simply be run again.
if ! conda run -n "$ENV" python -c "import numpy, pandas, pyarrow, numba, scipy, yaml, exchange_calendars, sklearn, lightgbm, pytest" 2>/dev/null; then
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
