#!/bin/bash
# 환경 설치 + Genesis two-field 패치 적용. 노트북(conda)과 Colab 공용.
#   노트북: conda create -n cutsim python=3.11 -y && conda activate cutsim && bash scripts/setup_env.sh
#   Colab : !bash scripts/setup_env.sh   (기본 torch 를 그대로 쓰려면 SKIP_TORCH=1)
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
if [ -z "$SKIP_TORCH" ]; then
  $PY -m pip install torch --index-url https://download.pytorch.org/whl/cu128
fi
$PY -m pip install -r requirements.txt
GS=$($PY -c "import genesis, os; print(os.path.dirname(genesis.__file__))")
$PY patches/make_genesis_multifield.py "$GS"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_capability() if torch.cuda.is_available() else '')"
echo "다음: python scripts/00_smoke_test.py"
